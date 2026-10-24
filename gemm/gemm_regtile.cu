// =====================================================================
// LEARNING STEP 2: register-tiled GEMM (fp32, CUDA cores)
//
//   C[N][N] = A[N][N] * B[N][N]      row-major,  N = 2048
//
// Step 1 (tiled 32) was load-bound: 1 output per thread meant
//   2 shared-memory loads for every 1 multiply-add.
//
// Fix: each thread computes an 8x8 grid of C (64 outputs) held in
// registers. Per k-step it loads 8 A values + 8 B values from shared
// memory into registers, then does 8*8 = 64 multiply-adds:
//   16 loads -> 64 FMAs  = 4 FMAs per load   (was 0.5)
//
// Block tile: BM x BN = 128 x 128, k-step BK = 8, 256 threads.
//
// compile:  nvcc -O3 -arch=sm_75 gemm_regtile.cu -o gemm_reg -lcublas
// run:      ./gemm_reg
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

// ------------------- step 1 kernel, for comparison -------------------
template <int TILE>
__global__ void matmul_tiled(const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        As[ty][tx] = A[row * N + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// ------------------------ register-tiled kernel -----------------------
constexpr int BM = 128, BN = 128, BK = 8;   // block tile
constexpr int TM = 8,   TN = 8;             // thread tile (outputs per thread)
constexpr int THREADS = (BM * BN) / (TM * TN);   // 256

__global__ void __launch_bounds__(THREADS)
matmul_regtile(const float* A, const float* B, float* C) {
    // shared "table": A slice stored transposed (As[k][m]) so that a
    // thread's 8 A values for one k sit next to each other
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid       = threadIdx.x;
    const int threadCol = tid % (BN / TN);    // 0..15 : which 8-wide column strip
    const int threadRow = tid / (BN / TN);    // 0..15 : which 8-tall row strip

    // move the pointers to this block's 128x128 piece
    A += blockIdx.y * BM * N;                 // first row of the block
    B += blockIdx.x * BN;                     // first column of the block
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {0.0f};               // the 8x8 answers, in registers
    float regA[TM], regB[TN];                 // the numbers "in hand"

    for (int k0 = 0; k0 < N; k0 += BK) {
        // ---- 1. load the 128x8 slice of A and the 8x128 slice of B ----
        //         256 threads, 1024 values each -> 4 values per thread
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;              // row r, column c of the A slice
            As[c][r] = A[r * N + k0 + c];            // store transposed
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int r = i / BN, c = i % BN;              // row r, column c of the B slice
            Bs[r][c] = B[(k0 + r) * N + c];          // neighbouring threads -> neighbouring addresses
        }
        __syncthreads();

        // ---- 2. 8 k's: pick up 8 + 8 values, do 64 multiply-adds ----
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[k][threadRow * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[k][threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];   // outer product: 64 FMAs
        }
        __syncthreads();
    }

    // ---- 3. write the 8x8 answers ----
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}

// ------------------------------ timing --------------------------------
template <typename F>
float timeIt(F launch, int iters = 10) {
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
    printf("%-14s %8.3f ms   %5.2f TFLOP/s (%4.1f%% of peak)   %5.1f%% of cuBLAS   diff %.1e\n",
           name, ms, tflops, 100.0 * tflops / PEAK_TFLOPS, 100.0 * cublasMs / ms, err);
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

    printf("Register-tiled GEMM: C = A x B, N = %d, fp32 (T4 peak ~%.1f TFLOP/s)\n\n", N, PEAK_TFLOPS);

    float tB = timeIt([&] {
        CHECK_BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                               &alpha, dB, N, dA, N, &beta, dC, N));
    });
    CHECK(cudaMemcpy(ref.data(), dC, bytes, cudaMemcpyDeviceToHost));

    float t32 = timeIt([&] { matmul_tiled<32><<<dim3(N / 32, N / 32), dim3(32, 32)>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    double e32 = relDiff(ref, out);

    float tR = timeIt([&] { matmul_regtile<<<dim3(N / BN, N / BM), THREADS>>>(dA, dB, dC); });
    CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    double eR = relDiff(ref, out);

    report("tiled 32", t32, e32, tB);
    report("register 8x8", tR, eR, tB);
    report("cuBLAS", tB, 0.0, tB);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
