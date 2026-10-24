
// T4 DECODE KERNELS - KERNEL 1- hand-written fp16 GEMV VS cuBLAS

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
constexpr int THREADS = 256;                 // 8 warps per block = 8 rows per block
constexpr int ROWS_PER_BLOCK = THREADS / 32;

//  our kernel
__global__ void gemv_warp_per_row(const __half* __restrict__ W,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ y) {
    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // which output this warp computes
    int lane = threadIdx.x & 31;                               // 0..31 inside the warp
    if (row >= N) return;                                      // whole warp exits together

    // view the row of W and the vector x as chunks of 8 halves (16 bytes)
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    float acc = 0.0f;                                          // running sum in fp32
#pragma unroll 4
    for (int i = lane; i < K / 8; i += 32) {                   // lane i, i+32, i+64, ...
        uint4 wv = w4[i];                                      // 8 weights
        uint4 xv = __ldg(&x4[i]);                              // 8 x values (read-only cache)
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; j++) {                          // 4 pairs of halves
            float2 a = __half22float2(wh[j]);
            float2 b = __half22float2(xh[j]);
            acc += a.x * b.x + a.y * b.y;
        }
    }

    // warp reduction: 32 partial sums -> 1
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset);

    if (lane == 0) y[row] = __float2half(acc);
}

// cuBLAS
// Same layout trick as Stage 4: W[N][K] row-major -> OP_T in column-major.
void gemvCublas(cublasHandle_t h, const __half* W, const __half* x, __half* y) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_BLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K,
        &alpha, W, CUDA_R_16F, K,
                x, CUDA_R_16F, K,
        &beta,  y, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// timing
template <typename F>
float timeIt(F launch, int iters = 200) {
    for (int i = 0; i < 10; i++) launch();     // warm-up
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

// relative error vs exact CPU answer: max|gpu - exact| / max|exact|
double relError(const std::vector<float>& ref, const std::vector<__half>& got) {
    double maxErr = 0, maxRef = 0;
    for (int n = 0; n < N; n++) {
        maxErr = std::max(maxErr, (double)std::fabs(__half2float(got[n]) - ref[n]));
        maxRef = std::max(maxRef, (double)std::fabs(ref[n]));
    }
    return maxErr / maxRef;
}

int main() {
    // data
    std::vector<__half> hW((size_t)N * K), hx(K);
    srand(42);
    for (auto& v : hW) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
    for (auto& v : hx) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);

    // exact CPU answer (double precision, from the same fp16 inputs)
    std::vector<float> ref(N);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int k = 0; k < K; k++)
            s += (double)__half2float(hW[(size_t)n * K + k]) * (double)__half2float(hx[k]);
        ref[n] = (float)s;
    }

    __half *dW, *dx, *dy;
    CHECK(cudaMalloc(&dW, hW.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dx, K * sizeof(__half)));
    CHECK(cudaMalloc(&dy, N * sizeof(__half)));
    CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));

    const double bytes = 2.0 * ((double)N * K + K + N);   // read W + x, write y (fp16)
    const int blocks = N / ROWS_PER_BLOCK;                // 4096 / 8 = 512 blocks
    std::vector<__half> out(N);

    printf("Kernel 1: fp16 GEMV, y[%d] = W[%d x %d] * x[%d]\n", N, N, K, K);
    printf("Weights = %.1f MB (far bigger than the 4 MB L2, so this measures DRAM)\n\n",
           2.0 * N * K / 1e6);

    // cuBLAS
    float tB = timeIt([&] { gemvCublas(h, dW, dx, dy); });
    CHECK(cudaMemcpy(out.data(), dy, N * sizeof(__half), cudaMemcpyDeviceToHost));
    double eB = relError(ref, out);
    printf("cuBLAS        %8.4f ms   %6.1f GB/s   rel. error %.2e\n",
           tB, bytes / (tB * 1e-3) / 1e9, eB);

    //  ours 
    CHECK(cudaMemset(dy, 0, N * sizeof(__half)));
    float tO = timeIt([&] { gemv_warp_per_row<<<blocks, THREADS>>>(dW, dx, dy); });
    CHECK(cudaMemcpy(out.data(), dy, N * sizeof(__half), cudaMemcpyDeviceToHost));
    double eO = relError(ref, out);
    printf("our kernel    %8.4f ms   %6.1f GB/s   rel. error %.2e\n",
           tO, bytes / (tO * 1e-3) / 1e9, eO);

    printf("\nours vs cuBLAS: %.2fx  (above 1.00 = we are faster)\n", tB / tO);

    cublasDestroy(h);
    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    return 0;
}
