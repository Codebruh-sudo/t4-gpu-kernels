
// T4 DECODE KERNELS - BATCH SWEEP- our GEMV-style kernels vs cuBLAS

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

constexpr int K = 4096;
constexpr int N = 4096;
constexpr int MAX_M = 512;
constexpr int THREADS = 256;                  // 8 warps = 8 rows per block
constexpr int ROWS_PER_BLOCK = THREADS / 32;
constexpr int TB = 8;                         // tokens per warp in v1b

//  v1- Kernel 1, one token per blockIdx.y 
__global__ void gemv_v1(const __half* __restrict__ W, const __half* __restrict__ X,
                        __half* __restrict__ Y) {
    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int t    = blockIdx.y;                                   // which token
    if (row >= N) return;

    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    const uint4* x4 = reinterpret_cast<const uint4*>(X + (size_t)t * K);

    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < K / 8; i += 32) {
        uint4 wv = w4[i];
        uint4 xv = __ldg(&x4[i]);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 a = __half22float2(wh[j]);
            float2 b = __half22float2(xh[j]);
            acc += a.x * b.x + a.y * b.y;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) Y[(size_t)t * N + row] = __float2half(acc);
}

// v1b: load each weight chunk once, use it for TB tokens
__global__ void gemv_v1b(const __half* __restrict__ W, const __half* __restrict__ X,
                         __half* __restrict__ Y, int M) {
    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int t0   = blockIdx.y * TB;                              // first token of this group
    if (row >= N) return;

    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * K);

    float acc[TB];
#pragma unroll
    for (int t = 0; t < TB; t++) acc[t] = 0.0f;

    for (int i = lane; i < K / 8; i += 32) {
        uint4 wv = w4[i];                                    // weights read ONCE ...
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        float2 wf[4];
#pragma unroll
        for (int j = 0; j < 4; j++) wf[j] = __half22float2(wh[j]);

#pragma unroll
        for (int t = 0; t < TB; t++) {                       // ... and used for TB tokens
            if (t0 + t < M) {                                // same for the whole warp
                const uint4* x4 = reinterpret_cast<const uint4*>(X + (size_t)(t0 + t) * K);
                uint4 xv = __ldg(&x4[i]);
                const __half2* xh = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    float2 b = __half22float2(xh[j]);
                    acc[t] += wf[j].x * b.x + wf[j].y * b.y;
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < TB; t++) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0 && t0 + t < M) Y[(size_t)(t0 + t) * N + row] = __float2half(v);
    }
}

// cuBLAS
void gemmCublas(cublasHandle_t h, int M, const __half* W, const __half* X, __half* Y) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_BLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, W, CUDA_R_16F, K,
                X, CUDA_R_16F, K,
        &beta,  Y, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

//  timing 
template <typename F>
float timeIt(F launch, int iters) {
    for (int i = 0; i < 3; i++) launch();                    // warm-up
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

// relative difference vs cuBLAS result (cuBLAS was checked against the CPU before)
double relDiff(const std::vector<__half>& ref, const std::vector<__half>& got, size_t n) {
    double maxErr = 0, maxRef = 0;
    for (size_t i = 0; i < n; i++) {
        double r = __half2float(ref[i]);
        maxErr = std::max(maxErr, std::fabs(__half2float(got[i]) - r));
        maxRef = std::max(maxRef, std::fabs(r));
    }
    return maxErr / maxRef;
}

int main() {
    std::vector<__half> hW((size_t)N * K), hX((size_t)MAX_M * K);
    srand(42);
    for (auto& v : hW) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
    for (auto& v : hX) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);

    __half *dW, *dX, *dYb, *dY1, *dY2;
    size_t ySize = (size_t)MAX_M * N * sizeof(__half);
    CHECK(cudaMalloc(&dW, hW.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dX, hX.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dYb, ySize));
    CHECK(cudaMalloc(&dY1, ySize));
    CHECK(cudaMalloc(&dY2, ySize));
    CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));

    const int blocksX = N / ROWS_PER_BLOCK;                  // 512
    int Ms[] = {1, 2, 4, 8, 16, 64, 256, 512};

    printf("Batch sweep: Y[M x %d] = X[M x %d] * W^T  (fp16, T4)\n", N, K);
    printf("times in ms;  'x' = speed vs cuBLAS (above 1.00 = we are faster)\n\n");
    printf("    M |   cuBLAS |   v1 naive        |   v1b reuse x8    | diff v1 / v1b\n");
    printf("------+----------+-------------------+-------------------+-----------------\n");

    for (int M : Ms) {
        int iters = (M <= 16) ? 200 : (M <= 64 ? 50 : 10);  // v1 gets slow at big M
        size_t n = (size_t)M * N;
        std::vector<__half> yB(n), y1(n), y2(n);

        float tB = timeIt([&] { gemmCublas(h, M, dW, dX, dYb); }, iters);
        float t1 = timeIt([&] { gemv_v1<<<dim3(blocksX, M), THREADS>>>(dW, dX, dY1); }, iters);
        float t2 = timeIt([&] {
            gemv_v1b<<<dim3(blocksX, (M + TB - 1) / TB), THREADS>>>(dW, dX, dY2, M);
        }, iters);

        CHECK(cudaMemcpy(yB.data(), dYb, n * sizeof(__half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(y1.data(), dY1, n * sizeof(__half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(y2.data(), dY2, n * sizeof(__half), cudaMemcpyDeviceToHost));

        printf(" %4d | %8.4f | %8.4f (%5.2fx) | %8.4f (%5.2fx) | %.1e / %.1e\n",
               M, tB, t1, tB / t1, t2, tB / t2,
               relDiff(yB, y1, n), relDiff(yB, y2, n));
    }

    cublasDestroy(h);
    cudaFree(dW); cudaFree(dX); cudaFree(dYb); cudaFree(dY1); cudaFree(dY2);
    return 0;
}
