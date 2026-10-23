// =====================================================================
// LEARNING STEP 1: tiled GEMM with shared memory (fp32, CUDA cores)
//
//   C[N][N] = A[N][N] * B[N][N]      row-major,  N = 2048
//
// Contenders:
//   naive      : each thread computes one C value, reads A and B straight
//                from global memory (every value fetched N times)
//   tiled<16>  : 16x16 blocks share tiles of A and B in shared memory
//   tiled<32>  : same idea, 32x32 tiles (1024 threads per block)
//   cuBLAS     : cublasSgemm, the target
//
// T4 fp32 peak (CUDA cores) is about 8.1 TFLOP/s.
//
// compile:  nvcc -O3 -arch=sm_75 gemm_tiled.cu -o gemm_tiled -lcublas
// run:      ./gemm_tiled
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

constexpr int N = 2048;
constexpr double PEAK_TFLOPS = 8.1;   // T4 fp32

// ------------------------------ naive --------------------------------
// One thread = one C value. Reads a full row of A and a full column of B
// from global memory.
__global__ void matmul_naive(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ------------------------------ tiled --------------------------------
// A block of TILE x TILE threads computes a TILE x TILE piece of C.
// It walks along K one tile at a time:
//   1. every thread loads ONE value of the A tile and ONE of the B tile
//   2. wait until the whole tile is on the "table"   (__syncthreads)
//   3. every thread does TILE multiply-adds from shared memory
//   4. wait before the tile is replaced              (__syncthreads)
template <int TILE>
__global__ void matmul_tiled(const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];      // the shared "table" for A
    __shared__ float Bs[TILE][TILE];      // the shared "table" for B

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;     // which C row this thread owns
    int col = blockIdx.x * TILE + tx;     // which C column this thread owns

    float sum = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        // step 1: each thread fetches one value of each tile
        //         (tx moves along a row -> neighbouring threads read neighbouring addresses -> coalesced)
        As[ty][tx] = A[row * N + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();                  // step 2: tile complete

        // step 3: reuse the tile TILE times from fast shared memory
#pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();                  // step 4: everyone done before overwrite
    }
    C[row * N + col] = sum;
}

// ------------------------------ timing --------------------------------
template <typename F>
float timeIt(F launch, int iters = 10) {
    launch();                                  // warm-up
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

void report(const char* name, float ms, double err) {
    double tflops = 2.0 * N * N * (double)N / (ms * 1e-3) / 1e12;
    printf("%-12s %9.3f ms   %5.2f TFLOP/s  (%4.1f%% of peak)   rel. diff vs cuBLAS %.1e\n",
           name, ms, tflops, 100.0 * tflops / PEAK_TFLOPS, err);
}

int main() {
    size_t n = (size_t)N * N, bytes = n * sizeof(float);
    std::vector<float> hA(n), hB(n), ref(n), out(n);
    srand(1);
    for (auto& v : hA) v = rand() / (float)RAND_MAX - 0.5f;
    for (auto& v : hB) v = rand() / (float)RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));
    const float alpha = 1.0f, beta = 0.0f;

    printf("Tiled GEMM study: C = A x B, N = %d, fp32 (T4 peak ~%.1f TFLOP/s)\n\n", N, PEAK_TFLOPS);

    // cuBLAS (row-major trick: swap A and B)
    float tB = timeIt([&] {
        CHECK_BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                               &alpha, dB, N, dA, N, &beta, dC, N));
    });
    CHECK(cudaMemcpy(ref.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // naive
    dim3 blk(16, 16), grd(N / 16, N / 16);
    float tN = timeIt([&] { matmul_naive<<<grd, blk>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    double eN = relDiff(ref, out);

    // tiled 16
    float t16 = timeIt([&] { matmul_tiled<16><<<dim3(N / 16, N / 16), dim3(16, 16)>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    double e16 = relDiff(ref, out);

    // tiled 32
    float t32 = timeIt([&] { matmul_tiled<32><<<dim3(N / 32, N / 32), dim3(32, 32)>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    double e32 = relDiff(ref, out);

    report("naive", tN, eN);
    report("tiled 16", t16, e16);
    report("tiled 32", t32, e32);
    report("cuBLAS", tB, 0.0);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
