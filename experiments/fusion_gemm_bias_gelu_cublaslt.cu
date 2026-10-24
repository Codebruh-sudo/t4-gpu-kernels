
// FUSION EXAMPLE 2 (library)- GEMM + bias + GELU via cuBLASLt epilogue
//
// The first half of every transformer FFN-   D = GELU(A * B + bias)
//
// UNFUSED- cuBLASLt matmul writes D - a separate kernel reads D,
//          adds bias, applies GELU, writes D again   (2 launches,
//          one extra read+write of D)
// FUSED-  CUBLASLT_EPILOGUE_GELU_BIAS - bias + GELU are applied
//          while the tile is still in registers   (1 launch, D written once)
//
// The lesson: fusion saves the extra D traffic. Whether that matters
// depends on how big that traffic is compared to the GEMM itself:
//   - n = 4096 (prefill-like): GEMM is compute-heavy, gain is small
//   - n = 64   (decode-like):  GEMM is tiny, the extra kernel and its
//                              launch are a bigger fraction - larger gain
//
// compile:  nvcc -O3 -arch=sm_75 fusion_gemm_bias_gelu_cublaslt.cu -o fusion_lt -lcublasLt
// run:      ./fusion_lt


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)
#define CHECK_LT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLASLt error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

// Column-major convention (cuBLAS)- D[m x n] = A[m x k] * B[k x n]
// Think- m = output features, k = input features, n = tokens.
// bias has one value per output feature (length m).
constexpr int M = 4096, K = 4096;
constexpr int MAX_N = 4096;

__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608f;            // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// The separate epilogue kernel used by the UNFUSED path
__global__ void bias_gelu(__half* D, const __half* bias, int m, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float v = __half2float(D[i]) + __half2float(bias[i % m]);   // column-major: row = i % m
        D[i] = __float2half(gelu_tanh(v));
    }
}

template <typename F>
float timeIt(F launch, int iters = 20) {
    launch();
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

// Build a matmul descriptor: fp16 in, fp32 accumulate, optional epilogue.
cublasLtMatmulDesc_t makeDesc(bool fused, const __half* biasPtr) {
    cublasLtMatmulDesc_t desc;
    CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op = CUBLAS_OP_N;
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));
    if (fused) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;      // <-- the fusion
        CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        const void* bp = biasPtr;
        CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
        cudaDataType_t bt = CUDA_R_16F;
        CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    }
    return desc;
}

int main() {
    // ---- host data ----
    std::vector<__half> hA((size_t)M * K), hB((size_t)K * MAX_N), hBias(M);
    srand(7);
    auto rnd = [] { return (rand() / (float)RAND_MAX - 0.5f) * 0.2f; };
    for (auto& v : hA) v = __float2half(rnd());
    for (auto& v : hB) v = __float2half(rnd());
    for (auto& v : hBias) v = __float2half(rnd() * 5.0f);

    // ---- device data ----
    __half *dA, *dB, *dBias, *dD;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dBias, M * sizeof(__half)));
    CHECK(cudaMalloc(&dD, (size_t)M * MAX_N * sizeof(__half)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dBias, hBias.data(), M * sizeof(__half), cudaMemcpyHostToDevice));

    size_t wsSize = 32 << 20;                 // 32 MB workspace for cuBLASLt
    void* dWs;
    CHECK(cudaMalloc(&dWs, wsSize));

    cublasLtHandle_t lt;
    CHECK_LT(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t descPlain = makeDesc(false, nullptr);
    cublasLtMatmulDesc_t descFused = makeDesc(true, dBias);

    float alpha = 1.0f, beta = 0.0f;
    int shapes[] = {4096, 64};               // prefill-like, decode-like

    for (int n : shapes) {
        cublasLtMatrixLayout_t la, lb, ld;
        CHECK_LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, M, K, M));
        CHECK_LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, n, K));
        CHECK_LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16F, M, n, M));
        size_t total = (size_t)M * n;
        int ewBlocks = (int)((total + 255) / 256);

        auto unfused = [&] {
            CHECK_LT(cublasLtMatmul(lt, descPlain, &alpha, dA, la, dB, lb, &beta,
                                    dD, ld, dD, ld, nullptr, dWs, wsSize, 0));
            bias_gelu<<<ewBlocks, 256>>>(dD, dBias, M, total);
        };
        auto fused = [&] {
            CHECK_LT(cublasLtMatmul(lt, descFused, &alpha, dA, la, dB, lb, &beta,
                                    dD, ld, dD, ld, nullptr, dWs, wsSize, 0));
        };

        std::vector<__half> outU(total), outF(total);
        float tU = timeIt(unfused);
        CHECK(cudaMemcpy(outU.data(), dD, total * sizeof(__half), cudaMemcpyDeviceToHost));
        float tF = timeIt(fused);
        CHECK(cudaMemcpy(outF.data(), dD, total * sizeof(__half), cudaMemcpyDeviceToHost));

        double maxDiff = 0, maxVal = 0;
        for (size_t i = 0; i < total; i++) {
            float u = __half2float(outU[i]), f = __half2float(outF[i]);
            maxDiff = std::max(maxDiff, (double)std::fabs(u - f));
            maxVal  = std::max(maxVal,  (double)std::fabs(u));
        }
        double flops = 2.0 * M * K * n;
        printf("n = %4d | unfused %8.3f ms | fused %8.3f ms | speedup %.2fx | "
               "fused %.1f TFLOP/s | max |unfused-fused| %.3g (max |val| %.3g)\n",
               n, tU, tF, tU / tF, flops / (tF * 1e-3) / 1e12, maxDiff, maxVal);

        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld);
    }

    cublasLtMatmulDescDestroy(descPlain);
    cublasLtMatmulDescDestroy(descFused);
    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dD); cudaFree(dWs);
    return 0;
}
