// =====================================================================
// FFN PROJECT — STAGE 4: INT8 quantization vs FP16, same GEMM
//
//   Y[M x N] = X[M x K] * W^T          K = N = 4096
//
// W is stored as [N][K] (one row of K weights per output feature).
// This is the PyTorch nn.Linear layout, and it is the layout cuBLAS
// int8 GEMM wants (the "TN" format).
//
// FP16 path:  fp16 x fp16 -> fp16              (1 cuBLAS call)
// INT8 path:  int8 x int8 -> int32  (cuBLAS)  +  dequantize -> fp16  (our kernel)
//
// Quantization (symmetric, done ONCE on the CPU, like a real model):
//   weights:      one scale per output feature  sW[n] = max|W[n][:]| / 127
//   activations:  one scale per token           sX[m] = max|X[m][:]| / 127
//   q = round(value / scale)        value ≈ q * scale
//   Y[m][n] ≈ (sum_k Xq*Wq) * sX[m] * sW[n]     <- the dequantize kernel
//
// (In a real model the activation quantization runs on the GPU every
//  step — one more small kernel, and a fusion opportunity.)
//
// compile:  nvcc -O3 -arch=sm_75 ffn_stage4_int8.cu -o ffn4 -lcublas
// run:      ./ffn4          (M = 1 and M = 512)
//           ./ffn4 1        (only M = 1, use with ncu)
// ncu:      ncu --launch-skip 10 --launch-count 6 \
//               --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed ./ffn4 1
//           -> look at the KERNEL NAMES: does cuBLAS use an int8 tensor-core kernel?
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// ------------------------------ GEMMs --------------------------------
// Row-major Y[M][N] = X[M][K] * W[N][K]^T
// cuBLAS (column-major) sees:  Y^T (N x M) = op(W) (N x K) * X^T (K x M)
// W[N][K] row-major looks like a K x N column-major matrix -> op = T.
void gemmFP16(cublasHandle_t h, int M, const __half* W, const __half* X, __half* Y) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_BLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, W, CUDA_R_16F, K,
                X, CUDA_R_16F, K,
        &beta,  Y, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// int8 x int8 -> int32. alpha/beta are int32 because the compute type is 32I.
cublasStatus_t gemmINT8(cublasHandle_t h, int M, const int8_t* W, const int8_t* X, int32_t* Y) {
    const int32_t alpha = 1, beta = 0;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, W, CUDA_R_8I, K,
                X, CUDA_R_8I, K,
        &beta,  Y, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
}

// --------------------------- dequantize ------------------------------
// One thread per output element: int32 -> real value -> fp16
__global__ void dequant(const int32_t* Y32, const float* sX, const float* sW,
                        __half* Y, int M) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) {
        int m = (int)(i / N), n = (int)(i % N);
        Y[i] = __float2half((float)Y32[i] * sX[m] * sW[n]);
    }
}

// ----------------------------- timing --------------------------------
template <typename F>
float timeIt(F launch, int iters = 50) {
    for (int i = 0; i < 5; i++) launch();          // warm-up
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

// --------------------------- quality check ---------------------------
// Compare GPU result with an exact CPU answer (double, original floats).
// Returns relative error = max|gpu - exact| / max|exact|
double relError(int M, const std::vector<float>& X, const std::vector<float>& W,
                const std::vector<__half>& Ygpu) {
    int rows[3] = {0, M / 2, M - 1};
    double maxErr = 0, maxRef = 0;
    for (int r : rows)
        for (int n = 0; n < N; n += 7) {
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)X[(size_t)r * K + k] * (double)W[(size_t)n * K + k];
            double got = __half2float(Ygpu[(size_t)r * N + n]);
            maxErr = std::max(maxErr, std::fabs(got - ref));
            maxRef = std::max(maxRef, std::fabs(ref));
        }
    return maxErr / maxRef;
}

int main(int argc, char** argv) {
    // ---------------- original float data ----------------
    std::vector<float> X((size_t)MAX_M * K), W((size_t)N * K);
    srand(42);
    for (auto& v : X) v = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : W) v = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;

    // ---------------- fp16 copies ----------------
    std::vector<__half> X16(X.size()), W16(W.size());
    for (size_t i = 0; i < X.size(); i++) X16[i] = __float2half(X[i]);
    for (size_t i = 0; i < W.size(); i++) W16[i] = __float2half(W[i]);

    // ---------------- int8 quantization (CPU, once) ----------------
    std::vector<int8_t> X8(X.size()), W8(W.size());
    std::vector<float> sX(MAX_M), sW(N);
    auto quantRow = [](const float* src, int8_t* dst, int len) {
        float mx = 0;
        for (int k = 0; k < len; k++) mx = std::max(mx, std::fabs(src[k]));
        float scale = mx / 127.0f;
        for (int k = 0; k < len; k++) {
            int q = (int)lrintf(src[k] / scale);
            dst[k] = (int8_t)std::min(127, std::max(-127, q));
        }
        return scale;
    };
    for (int n = 0; n < N; n++)     sW[n] = quantRow(&W[(size_t)n * K], &W8[(size_t)n * K], K);
    for (int m = 0; m < MAX_M; m++) sX[m] = quantRow(&X[(size_t)m * K], &X8[(size_t)m * K], K);

    // ---------------- device buffers ----------------
    __half *dX16, *dW16, *dY16;
    int8_t *dX8, *dW8;
    int32_t* dY32;
    float *dsX, *dsW;
    CHECK(cudaMalloc(&dX16, X16.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dW16, W16.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dY16, (size_t)MAX_M * N * sizeof(__half)));
    CHECK(cudaMalloc(&dX8, X8.size()));
    CHECK(cudaMalloc(&dW8, W8.size()));
    CHECK(cudaMalloc(&dY32, (size_t)MAX_M * N * sizeof(int32_t)));
    CHECK(cudaMalloc(&dsX, MAX_M * sizeof(float)));
    CHECK(cudaMalloc(&dsW, N * sizeof(float)));
    CHECK(cudaMemcpy(dX16, X16.data(), X16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW16, W16.data(), W16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX8, X8.data(), X8.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW8, W8.data(), W8.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsX, sX.data(), MAX_M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsW, sW.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));

    printf("FFN Stage 4: FP16 vs INT8,  Y[M x %d] = X[M x %d] * W^T\n", N, K);
    printf("Weights: fp16 = %.1f MB, int8 = %.1f MB  (half the bytes)\n\n",
           2.0 * N * K / 1e6, 1.0 * N * K / 1e6);

    std::vector<int> Ms;
    if (argc > 1) Ms.push_back(std::max(1, std::min(MAX_M, atoi(argv[1]))));
    else          Ms = {1, MAX_M};

    for (int M : Ms) {
        int dqBlocks = (int)(((size_t)M * N + 255) / 256);
        std::vector<__half> out((size_t)M * N);

        // ---- FP16 ----
        float tFP16 = timeIt([&] { gemmFP16(h, M, dW16, dX16, dY16); });
        CHECK(cudaMemcpy(out.data(), dY16, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        double eFP16 = relError(M, X, W, out);

        // ---- INT8: does cuBLAS support this shape? ----
        cublasStatus_t st = gemmINT8(h, M, dW8, dX8, dY32);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("M = %4d | fp16 %.4f ms | int8 GEMM NOT SUPPORTED by cuBLAS for this shape (status %d)\n",
                   M, tFP16, (int)st);
            continue;
        }
        float tGemm8 = timeIt([&] { gemmINT8(h, M, dW8, dX8, dY32); });
        float tFull8 = timeIt([&] {
            gemmINT8(h, M, dW8, dX8, dY32);
            dequant<<<dqBlocks, 256>>>(dY32, dsX, dsW, dY16, M);
        });
        CHECK(cudaMemcpy(out.data(), dY16, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        double eINT8 = relError(M, X, W, out);

        printf("M = %4d\n", M);
        printf("   fp16 GEMM             %8.4f ms                      rel. error %.2e\n", tFP16, eFP16);
        printf("   int8 GEMM only        %8.4f ms   (%.2fx vs fp16)\n", tGemm8, tFP16 / tGemm8);
        printf("   int8 GEMM + dequant   %8.4f ms   (%.2fx vs fp16)   rel. error %.2e\n\n",
               tFull8, tFP16 / tFull8, eINT8);
    }

    cublasDestroy(h);
    cudaFree(dX16); cudaFree(dW16); cudaFree(dY16);
    cudaFree(dX8); cudaFree(dW8); cudaFree(dY32);
    cudaFree(dsX); cudaFree(dsW);
    return 0;
}
