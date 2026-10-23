// =====================================================================
// FFN PROJECT — STAGE 1: one FFN GEMM in fp16 with cublasGemmEx
//
//   Y[M x N] = X[M x K] * W[K x N]        (row-major thinking)
//
//   X = activations (M tokens, K features)
//   W = weights     (K x N = 4096 x 4096 = 32 MB in fp16)
//   M = number of tokens in the "bus":
//       M = 1   -> decode, batch 1     (expect MEMORY-bound)
//       M = 512 -> prefill / big batch (expect COMPUTE-bound)
//
// What you measure for each M:
//   1. kernel time (printed by this program)
//   2. DRAM throughput %      (ncu)
//   3. tensor pipe %          (ncu)
//
// compile:  nvcc -O3 -arch=sm_75 ffn_stage1_gemm.cu -o ffn1 -lcublas
// run:      ./ffn1            (runs M=1 and M=512)
//           ./ffn1 1          (only M=1   -> use this with ncu)
//           ./ffn1 512        (only M=512 -> use this with ncu)
//
// ncu (one GEMM, after warm-up):
//   ncu --launch-skip 5 --launch-count 3 \
//       --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
//       ./ffn1 1
//   (count 3: cuBLAS sometimes launches 2 kernels for one GEMM, e.g. split-K + reduce)
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

constexpr int K = 4096;              // input features
constexpr int N = 4096;              // output features
constexpr int MAX_M = 512;
constexpr double PEAK_GBS    = 320.0;   // T4 memory bandwidth
constexpr double PEAK_TFLOPS = 65.0;    // T4 fp16 tensor-core peak (on paper)

// ---------------------------------------------------------------------
// The one important call.
// cuBLAS is COLUMN-major. Our arrays are ROW-major.
// Trick (same as the 4x4 Sgemm example): a row-major matrix is the
// transpose in column-major, so we compute  Y^T = W^T * X^T,
// which means: pass W first, X second, and swap M and N.
// ---------------------------------------------------------------------
void gemm(cublasHandle_t h, int M, const __half* dX, const __half* dW, __half* dY) {
    const float alpha = 1.0f, beta = 0.0f;   // float because compute type is 32F
    CHECK_BLAS(cublasGemmEx(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                          // (rows, cols, inner) of the column-major view
        &alpha,
        dW, CUDA_R_16F, N,                // W first  (leading dim = N)
        dX, CUDA_R_16F, K,                // X second (leading dim = K)
        &beta,
        dY, CUDA_R_16F, N,                // Y        (leading dim = N)
        CUBLAS_COMPUTE_32F,               // accumulate in fp32 (accurate)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));  // allow tensor cores
}

// Average time of one GEMM over many runs (after a warm-up)
float timeGemm(cublasHandle_t h, int M, const __half* dX, const __half* dW, __half* dY,
               int iters = 50) {
    for (int i = 0; i < 5; i++) gemm(h, M, dX, dW, dY);   // warm-up
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) gemm(h, M, dX, dW, dY);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

// Check a few output rows against the CPU (catches layout mistakes)
double checkRows(int M, const std::vector<__half>& hX, const std::vector<__half>& hW,
                 const std::vector<__half>& hY) {
    int rows[3] = {0, M / 2, M - 1};
    double maxErr = 0;
    for (int r : rows) {
        for (int n = 0; n < N; n += 7) {               // every 7th column is plenty
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)__half2float(hX[(size_t)r * K + k]) *
                       (double)__half2float(hW[(size_t)k * N + n]);
            double got = __half2float(hY[(size_t)r * N + n]);
            maxErr = fmax(maxErr, fabs(got - ref));
        }
    }
    return maxErr;
}

void runOne(cublasHandle_t h, int M, const std::vector<__half>& hX, const std::vector<__half>& hW,
            const __half* dX, const __half* dW, __half* dY) {
    float ms = timeGemm(h, M, dX, dW, dY);

    std::vector<__half> hY((size_t)M * N);
    CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    double err = checkRows(M, hX, hW, hY);

    double flops = 2.0 * M * N * K;                                    // multiply + add
    double bytes = 2.0 * ((double)K * N + (double)M * K + (double)M * N); // W + X + Y, fp16
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbs    = bytes / (ms * 1e-3) / 1e9;
    double ai     = flops / bytes;                                      // FLOP per byte

    printf("M = %4d | time %8.4f ms | %6.2f TFLOP/s (%4.1f%% of peak) | %6.1f GB/s (%4.1f%% of peak)"
           " | intensity %6.1f FLOP/B | max err %.3g\n",
           M, ms, tflops, 100 * tflops / PEAK_TFLOPS, gbs, 100 * gbs / PEAK_GBS, ai, err);
}

int main(int argc, char** argv) {
    // ---- host data: small random values so fp16 doesn't overflow ----
    std::vector<__half> hX((size_t)MAX_M * K), hW((size_t)K * N);
    srand(42);
    for (auto& v : hX) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
    for (auto& v : hW) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);

    // ---- device data ----
    __half *dX, *dW, *dY;
    CHECK(cudaMalloc(&dX, hX.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dW, hW.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dY, (size_t)MAX_M * N * sizeof(__half)));
    CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));

    printf("FFN Stage 1: Y[M x %d] = X[M x %d] * W[%d x %d], fp16 in / fp32 accumulate\n", N, K, K, N);
    printf("Weights = %.1f MB.  Ideal time to just READ them at %.0f GB/s = %.3f ms\n\n",
           2.0 * K * N / 1e6, PEAK_GBS, 2.0 * K * N / (PEAK_GBS * 1e9) * 1e3);

    if (argc > 1) {
        int M = atoi(argv[1]);
        if (M < 1 || M > MAX_M) { printf("M must be 1..%d\n", MAX_M); return 1; }
        runOne(h, M, hX, hW, dX, dW, dY);
    } else {
        runOne(h, 1, hX, hW, dX, dW, dY);
        runOne(h, MAX_M, hX, hW, dX, dW, dY);
    }

    cublasDestroy(h);
    cudaFree(dX); cudaFree(dW); cudaFree(dY);
    return 0;
}
