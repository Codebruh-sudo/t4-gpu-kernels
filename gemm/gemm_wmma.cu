// =====================================================================
// LEARNING STEP 3: tensor cores with WMMA (fp16 in, fp32 accumulate)
//
//   C[N][N] (fp32) = A[N][N] (fp16) * B[N][N] (fp16)     row-major, N = 4096
//
// Contenders:
//   wmma simple : each warp computes one 16x16 tile of C, loading its
//                 A and B tiles straight from global memory (no reuse)
//   wmma shared : block of 4 warps computes a 64x64 tile of C.
//                 A and B tiles are staged in shared memory, and each warp
//                 computes a 32x32 piece = 2x2 tensor-core tiles
//   cuBLAS      : cublasGemmEx, fp16 in / fp32 out, tensor cores
//
// T4: fp16 tensor-core peak ~65 TFLOP/s (fp32 CUDA cores only ~8).
//
// compile:  nvcc -O3 -arch=sm_75 gemm_wmma.cu -o gemm_wmma -lcublas
// run:      ./gemm_wmma
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

constexpr int N = 4096;
constexpr int WM = 16, WN = 16, WK = 16;      // one tensor-core tile = 16x16x16
constexpr double PEAK_TFLOPS = 65.0;          // T4 fp16 tensor cores (on paper)

// ------------------------------ version 1 -----------------------------
// 4 warps per block, arranged 2x2 -> the block covers 32x32 of C.
// Each warp: one 16x16 tile of C, walks K in steps of 16.
__global__ void wmma_simple(const half* A, const half* B, float* C) {
    int warpId  = threadIdx.x / 32;           // 0..3
    int tileRow = blockIdx.y * 2 + warpId / 2;
    int tileCol = blockIdx.x * 2 + warpId % 2;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < N; k += WK) {
        wmma::load_matrix_sync(a, A + (size_t)(tileRow * WM) * N + k, N);   // 16x16 of A
        wmma::load_matrix_sync(b, B + (size_t)k * N + tileCol * WN, N);     // 16x16 of B
        wmma::mma_sync(acc, a, b, acc);                                     // tensor core!
    }
    wmma::store_matrix_sync(C + (size_t)(tileRow * WM) * N + tileCol * WN, acc, N,
                            wmma::mem_row_major);
}

// ------------------------------ version 2 -----------------------------
// Block tile 64x64, BK = 32, 4 warps (2x2). Each warp owns 32x32 of C
// = 2x2 tensor-core tiles, so every A fragment is used twice and every
// B fragment is used twice (reuse in registers, like register tiling).
constexpr int BM = 64, BN = 64, BK = 32;
constexpr int PAD = 8;                         // padding against bank conflicts
constexpr int LDA_S = BK + PAD;                // 40 halves per As row (80 bytes)
constexpr int LDB_S = BN + PAD;                // 72 halves per Bs row (144 bytes)
constexpr int THREADS = 128;

__global__ void __launch_bounds__(THREADS)
wmma_shared(const half* A, const half* B, float* C) {
    __shared__ __align__(16) half As[BM * LDA_S];
    __shared__ __align__(16) half Bs[BK * LDB_S];

    int tid     = threadIdx.x;
    int warpId  = tid / 32;
    int warpRow = warpId / 2;                  // 0..1
    int warpCol = warpId % 2;                  // 0..1
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b[2];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < N; k0 += BK) {
        // ---- load 64x32 of A and 32x64 of B into shared memory ----
        //      16-byte chunks (8 halves); 256 chunks each -> 2 per thread
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 8), c = idx % (BK / 8);          // A: 32 cols = 4 chunks/row
            *reinterpret_cast<uint4*>(&As[r * LDA_S + c * 8]) =
                *reinterpret_cast<const uint4*>(&A[(size_t)(blockRow + r) * N + k0 + c * 8]);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / (BN / 8), c = idx % (BN / 8);          // B: 64 cols = 8 chunks/row
            *reinterpret_cast<uint4*>(&Bs[r * LDB_S + c * 8]) =
                *reinterpret_cast<const uint4*>(&B[(size_t)(k0 + r) * N + blockCol + c * 8]);
        }
        __syncthreads();

        // ---- two 16-deep steps through the 32-wide slice ----
#pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
#pragma unroll
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(a[i], &As[(warpRow * 32 + i * 16) * LDA_S + kk], LDA_S);
#pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b[j], &Bs[kk * LDB_S + warpCol * 32 + j * 16], LDB_S);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);   // 4 tensor-core ops
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(
                C + (size_t)(blockRow + warpRow * 32 + i * 16) * N + blockCol + warpCol * 32 + j * 16,
                acc[i][j], N, wmma::mem_row_major);
}

// ------------------------------ timing --------------------------------
template <typename F>
float timeIt(F launch, int iters = 10) {
    for (int i = 0; i < 5; i++) launch();      // warm-up: let clocks ramp up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

double relDiff(const std::vector<float>& ref, const std::vector<float>& got) {
    double maxErr = 0, maxRef = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        maxErr = std::max(maxErr, (double)std::fabs(got[i] - ref[i]));
        maxRef = std::max(maxRef, (double)std::fabs(ref[i]));
    }
    return maxErr / maxRef;
}

void report(const char* name, float ms, double err, float cublasMs) {
    double tflops = 2.0 * N * N * (double)N / (ms * 1e-3) / 1e12;
    printf("%-14s %8.3f ms   %6.2f TFLOP/s (%4.1f%% of peak)   %5.1f%% of cuBLAS   diff %.1e\n",
           name, ms, tflops, 100.0 * tflops / PEAK_TFLOPS, 100.0 * cublasMs / ms, err);
}

int main() {
    size_t n = (size_t)N * N;
    std::vector<half> hA(n), hB(n);
    srand(1);
    for (auto& v : hA) v = __float2half(rand() / (float)RAND_MAX - 0.5f);
    for (auto& v : hB) v = __float2half(rand() / (float)RAND_MAX - 0.5f);

    half *dA, *dB;
    float* dC;
    CHECK(cudaMalloc(&dA, n * sizeof(half)));
    CHECK(cudaMalloc(&dB, n * sizeof(half)));
    CHECK(cudaMalloc(&dC, n * sizeof(float)));
    CHECK(cudaMemcpy(dA, hA.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));
    const float alpha = 1.0f, beta = 0.0f;
    auto cublasRun = [&] {   // row-major trick: swap A and B
        CHECK_BLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, N,
            &beta,  dC, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    printf("Tensor-core GEMM (WMMA): C = A x B, N = %d, fp16 in / fp32 out (T4 peak ~%.0f TFLOP/s)\n\n",
           N, PEAK_TFLOPS);

    std::vector<float> ref(n), out(n);

    float tB1 = timeIt(cublasRun);
    CHECK(cudaMemcpy(ref.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));

    float tS = timeIt([&] { wmma_simple<<<dim3(N / 32, N / 32), 128>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    double eS = relDiff(ref, out);

    float tSh = timeIt([&] { wmma_shared<<<dim3(N / BN, N / BM), THREADS>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    double eSh = relDiff(ref, out);

    float tB2 = timeIt(cublasRun);
    float tB = 0.5f * (tB1 + tB2);

    report("cuBLAS before", tB1, 0.0, tB);
    report("wmma simple", tS, eS, tB);
    report("wmma shared", tSh, eSh, tB);
    report("cuBLAS after", tB2, 0.0, tB);
    printf("\nFor comparison: your fp32 register-tiled kernel reached ~4.3 TFLOP/s on CUDA cores.\n");

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
