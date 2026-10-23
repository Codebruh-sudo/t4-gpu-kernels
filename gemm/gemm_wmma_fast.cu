// =====================================================================
// LEARNING STEP 4: faster tensor-core GEMM (WMMA) on the T4
//
//   C[N][N] (fp32) = A[N][N] (fp16) * B[N][N] (fp16)     row-major, N = 4096
//
// Upgrades over step 3 ("wmma shared", 64x64 block, 2x2 fragments/warp):
//   1. BIGGER TILES:  block 128x128, 8 warps, each warp 32x64 of C
//                     = 2x4 = 8 tensor-core tiles per warp (more reuse)
//   2. DOUBLE BUFFERING: two shared-memory buffers. While the tensor cores
//                     work on buffer "cur", the next tile is already being
//                     fetched from global memory into registers, then
//                     stored into buffer "next" after the math.
//                     (The T4 has no cp.async, so the prefetch goes
//                      global -> registers -> shared.)
//
// compile:  nvcc -O3 -arch=sm_75 gemm_wmma_fast.cu -o gemm_fast -lcublas
// run:      ./gemm_fast
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
constexpr int WM = 16, WN = 16, WK = 16;
constexpr double PEAK_TFLOPS = 65.0;

// ===================== step 3 kernel (for comparison) ==================
namespace s3 {
constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;
constexpr int LDA_S = BK + PAD, LDB_S = BN + PAD, THREADS = 128;

__global__ void __launch_bounds__(THREADS)
wmma_shared(const half* A, const half* B, float* C) {
    __shared__ __align__(32) half As[BM * LDA_S];
    __shared__ __align__(32) half Bs[BK * LDB_S];
    int tid = threadIdx.x, warpId = tid / 32;
    int warpRow = warpId / 2, warpCol = warpId % 2;
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b[2];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS, r = idx / (BK / 8), c = idx % (BK / 8);
            *reinterpret_cast<uint4*>(&As[r * LDA_S + c * 8]) =
                *reinterpret_cast<const uint4*>(&A[(size_t)(blockRow + r) * N + k0 + c * 8]);
        }
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS, r = idx / (BN / 8), c = idx % (BN / 8);
            *reinterpret_cast<uint4*>(&Bs[r * LDB_S + c * 8]) =
                *reinterpret_cast<const uint4*>(&B[(size_t)(k0 + r) * N + blockCol + c * 8]);
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += WK) {
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(a[i], &As[(warpRow * 32 + i * 16) * LDA_S + kk], LDA_S);
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b[j], &Bs[kk * LDB_S + warpCol * 32 + j * 16], LDB_S);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
                wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
        wmma::store_matrix_sync(
            C + (size_t)(blockRow + warpRow * 32 + i * 16) * N + blockCol + warpCol * 32 + j * 16,
            acc[i][j], N, wmma::mem_row_major);
}
}  // namespace s3

// ========================= step 4: the fast kernel =====================
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int PAD = 8;
constexpr int LDA_S = BK + PAD;          // 40 halves per As row
constexpr int LDB_S = BN + PAD;          // 136 halves per Bs row
constexpr int THREADS = 256;             // 8 warps
constexpr int FRAG_M = 2, FRAG_N = 4;    // each warp: 2x4 tiles = 32x64 of C

__global__ void __launch_bounds__(THREADS)
wmma_fast(const half* A, const half* B, float* C) {
    // two buffers ("tables") for A and for B
    __shared__ __align__(32) half As[2][BM * LDA_S];
    __shared__ __align__(32) half Bs[2][BK * LDB_S];

    int tid     = threadIdx.x;
    int warpId  = tid / 32;
    int warpRow = warpId / 2;            // 0..3 -> rows warpRow*32
    int warpCol = warpId % 2;            // 0..1 -> cols warpCol*64
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[FRAG_M];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b[FRAG_N];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; i++)
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    // each thread moves 2 chunks (16 bytes) of A and 2 chunks of B per tile
    //   A tile 128x32 = 512 chunks (4 per row), B tile 32x128 = 512 chunks (16 per row)
    uint4 regA[2], regB[2];

    auto loadGlobal = [&](int k0) {               // global -> registers
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / 4, c = idx % 4;
            regA[i] = *reinterpret_cast<const uint4*>(&A[(size_t)(blockRow + r) * N + k0 + c * 8]);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / 16, c = idx % 16;
            regB[i] = *reinterpret_cast<const uint4*>(&B[(size_t)(k0 + r) * N + blockCol + c * 8]);
        }
    };
    auto storeShared = [&](int buf) {             // registers -> shared buffer
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / 4, c = idx % 4;
            *reinterpret_cast<uint4*>(&As[buf][r * LDA_S + c * 8]) = regA[i];
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS;
            int r = idx / 16, c = idx % 16;
            *reinterpret_cast<uint4*>(&Bs[buf][r * LDB_S + c * 8]) = regB[i];
        }
    };

    const int numTiles = N / BK;

    // prologue: fill buffer 0
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
        int cur = t & 1;

        // 1. start fetching the NEXT tile now (it travels while we compute)
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

        // 2. tensor-core math on the CURRENT buffer
#pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
#pragma unroll
            for (int i = 0; i < FRAG_M; i++)
                wmma::load_matrix_sync(a[i], &As[cur][(warpRow * 32 + i * 16) * LDA_S + kk], LDA_S);
#pragma unroll
            for (int j = 0; j < FRAG_N; j++)
                wmma::load_matrix_sync(b[j], &Bs[cur][kk * LDB_S + warpCol * 64 + j * 16], LDB_S);
#pragma unroll
            for (int i = 0; i < FRAG_M; i++)
#pragma unroll
                for (int j = 0; j < FRAG_N; j++)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);   // 8 tensor-core ops
        }

        // 3. put the prefetched tile into the OTHER buffer
        //    (nobody reads it now: it was last read in iteration t-1,
        //     and everyone passed the barrier at the end of t-1)
        if (t + 1 < numTiles) storeShared(cur ^ 1);

        // 4. one barrier per tile: next buffer complete AND current buffer
        //    no longer being read before it is overwritten next iteration
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAG_M; i++)
#pragma unroll
        for (int j = 0; j < FRAG_N; j++)
            wmma::store_matrix_sync(
                C + (size_t)(blockRow + warpRow * 32 + i * 16) * N + blockCol + warpCol * 64 + j * 16,
                acc[i][j], N, wmma::mem_row_major);
}

// ------------------------------ timing --------------------------------
template <typename F>
float timeIt(F launch, int iters = 10) {
    for (int i = 0; i < 5; i++) launch();
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
    auto cublasRun = [&] {
        CHECK_BLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, N,
            &beta,  dC, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    printf("Fast tensor-core GEMM: C = A x B, N = %d, fp16 in / fp32 out (T4 peak ~%.0f TFLOP/s)\n\n",
           N, PEAK_TFLOPS);

    std::vector<float> ref(n), out(n);
    float tB1 = timeIt(cublasRun);
    CHECK(cudaMemcpy(ref.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));

    // clear C before each of our kernels, so a kernel that writes nothing cannot "pass"
    CHECK(cudaMemset(dC, 0, n * sizeof(float)));
    float t3 = timeIt([&] { s3::wmma_shared<<<dim3(N / s3::BN, N / s3::BM), s3::THREADS>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    double e3 = relDiff(ref, out);

    CHECK(cudaMemset(dC, 0, n * sizeof(float)));
    float t4 = timeIt([&] { wmma_fast<<<dim3(N / BN, N / BM), THREADS>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    double e4 = relDiff(ref, out);

    float tB2 = timeIt(cublasRun);
    float tB = 0.5f * (tB1 + tB2);

    report("cuBLAS before", tB1, 0.0, tB);
    report("step 3 shared", t3, e3, tB);
    report("step 4 fast", t4, e4, tB);
    report("cuBLAS after", tB2, 0.0, tB);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
