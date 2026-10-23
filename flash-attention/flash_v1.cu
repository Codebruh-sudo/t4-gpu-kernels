// =====================================================================
// FLASHATTENTION FOR THE T4 — STAGE 1: correct first (CUDA cores)
//
//   O = softmax(Q K^T / sqrt(d)) V        per (batch, head)
//   layout: [B*H][N][D] contiguous (same as PyTorch [B, H, N, D])
//
// The FlashAttention idea, in its simplest form:
//   - one THREAD owns one query row: keeps q (64 values) and its running
//     output o (64 values) in registers, plus running max m and sum l
//   - a BLOCK of 64 threads = 64 query rows
//   - K and V arrive TILE BY TILE (64 keys at a time) in shared memory
//   - for every key: score s = q.k, then the ONLINE SOFTMAX update:
//         m_new = max(m, s)
//         corr  = exp(m - m_new)          (rescale everything seen so far)
//         p     = exp(s - m_new)
//         l     = l * corr + p
//         o     = o * corr + p * v
//   - at the end: O = o / l
//   - the N x N score matrix NEVER exists
//
// Stage 1 is deliberately simple (fp32 math on CUDA cores, rescale per key).
// It is the correctness baseline. Speed comes in stages 2-3 (tensor cores).
//
// compile:  nvcc -O3 -arch=sm_75 flash_v1.cu -o flash1
// run:      ./flash1
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int B  = 1;      // batch
constexpr int H  = 8;      // heads
constexpr int D  = 64;     // head dimension
constexpr int BR = 64;     // query rows per block (= threads per block)
constexpr int BC = 64;     // keys per K/V tile

// ------------------------------ kernel --------------------------------
// grid: (N / BR, B*H)    block: BR threads
__global__ void __launch_bounds__(BR)
flash_v1(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    int bh  = blockIdx.y;                          // which (batch, head)
    int row = blockIdx.x * BR + threadIdx.x;       // which query row this thread owns

    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;

    __shared__ float Ks[BC][D];                    // the K tile on the "table"
    __shared__ float Vs[BC][D];                    // the V tile on the "table"

    // this thread's query row (pre-scaled) and running output, in registers
    float q[D], o[D];
    const half* qp = Q + ((size_t)bh * N + row) * D;
#pragma unroll
    for (int d = 0; d < D; d++) { q[d] = __half2float(qp[d]) * scale; o[d] = 0.0f; }
    float m = -INFINITY;                           // running max
    float l = 0.0f;                                // running sum of exp

    for (int t0 = 0; t0 < N; t0 += BC) {
        // ---- load one K tile and one V tile (64 keys x 64 dims) ----
        //      neighbouring threads read neighbouring addresses -> coalesced
        for (int i = threadIdx.x; i < BC * D; i += BR) {
            int r = i / D, c = i % D;
            Ks[r][c] = __half2float(Kb[(size_t)(t0 + r) * D + c]);
            Vs[r][c] = __half2float(Vb[(size_t)(t0 + r) * D + c]);
        }
        __syncthreads();

        // ---- process the 64 keys of this tile ----
        for (int j = 0; j < BC; j++) {
            // score = q . k_j   (every thread reads the same Ks[j][d] -> broadcast)
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; d++) s += q[d] * Ks[j][d];

            // online softmax update
            float m_new = fmaxf(m, s);
            float corr  = __expf(m - m_new);       // first key: exp(-inf) = 0
            float p     = __expf(s - m_new);
            l = l * corr + p;
#pragma unroll
            for (int d = 0; d < D; d++) o[d] = o[d] * corr + p * Vs[j][d];
            m = m_new;
        }
        __syncthreads();                           // tile used up before it is replaced
    }

    // ---- final normalize and write ----
    half* op = O + ((size_t)bh * N + row) * D;
#pragma unroll
    for (int d = 0; d < D; d++) op[d] = __float2half(o[d] / l);
}

// ------------------------------ helpers --------------------------------
template <typename F>
float timeIt(F launch, int iters = 10) {
    for (int i = 0; i < 3; i++) launch();
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

// exact attention for ONE query row on the CPU (double precision)
void cpuRow(const std::vector<half>& Q, const std::vector<half>& K, const std::vector<half>& V,
            int N, int bh, int row, float scale, double* out) {
    std::vector<double> s(N);
    double mx = -1e300;
    for (int j = 0; j < N; j++) {
        double acc = 0;
        for (int d = 0; d < D; d++)
            acc += (double)__half2float(Q[((size_t)bh * N + row) * D + d]) *
                   (double)__half2float(K[((size_t)bh * N + j) * D + d]);
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
    }
    double sum = 0;
    for (int j = 0; j < N; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
    for (int d = 0; d < D; d++) {
        double acc = 0;
        for (int j = 0; j < N; j++) acc += s[j] * (double)__half2float(V[((size_t)bh * N + j) * D + d]);
        out[d] = acc / sum;
    }
}

int main() {
    const float scale = 1.0f / std::sqrt((float)D);
    int Ns[] = {512, 1024, 2048, 4096, 8192};

    printf("FlashAttention stage 1 (CUDA cores, fp32 math): B=%d H=%d D=%d, fp16 in/out\n\n", B, H, D);
    printf("%6s | %10s | %9s | %16s | %22s | %s\n",
           "N", "time ms", "TFLOP/s", "memory used MB", "naive N x N would need", "max error");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int N : Ns) {
        size_t elems = (size_t)B * H * N * D;
        std::vector<half> hQ(elems), hK(elems), hV(elems), hO(elems);
        srand(N);
        for (size_t i = 0; i < elems; i++) {
            hQ[i] = __float2half(rand() / (float)RAND_MAX - 0.5f);
            hK[i] = __float2half(rand() / (float)RAND_MAX - 0.5f);
            hV[i] = __float2half(rand() / (float)RAND_MAX - 0.5f);
        }

        half *dQ, *dK, *dV, *dO;
        CHECK(cudaMalloc(&dQ, elems * sizeof(half)));
        CHECK(cudaMalloc(&dK, elems * sizeof(half)));
        CHECK(cudaMalloc(&dV, elems * sizeof(half)));
        CHECK(cudaMalloc(&dO, elems * sizeof(half)));
        CHECK(cudaMemcpy(dQ, hQ.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dK, hK.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dV, hV.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dO, 0, elems * sizeof(half)));

        dim3 grid(N / BR, B * H);
        float ms = timeIt([&] { flash_v1<<<grid, BR>>>(dQ, dK, dV, dO, N, scale); });
        CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(half), cudaMemcpyDeviceToHost));

        // check 8 rows (different heads, start / middle / end) against the CPU
        double maxErr = 0;
        int checks[8][2] = {{0, 0}, {0, N / 2}, {0, N - 1}, {3, 7}, {5, N / 3},
                            {H - 1, 0}, {H - 1, N / 2}, {H - 1, N - 1}};
        double ref[D];
        for (auto& c : checks) {
            cpuRow(hQ, hK, hV, N, c[0], c[1], scale, ref);
            for (int d = 0; d < D; d++)
                maxErr = std::max(maxErr, std::fabs(
                    (double)__half2float(hO[((size_t)c[0] * N + c[1]) * D + d]) - ref[d]));
        }

        double flops  = 4.0 * B * H * (double)N * N * D;        // QK^T + PV
        double usedMB = 4.0 * elems * sizeof(half) / 1e6;       // Q, K, V, O only
        double naiveMB = (double)B * H * N * N * sizeof(half) / 1e6;   // the score matrix alone (fp16)
        printf("%6d | %10.3f | %9.2f | %16.1f | %19.1f MB | %.1e\n",
               N, ms, flops / (ms * 1e-3) / 1e12, usedMB, naiveMB, maxErr);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    printf("\nMemory stays linear in N (only Q, K, V, O) while the naive score matrix grows as N^2.\n");
    printf("Max error should be around 1e-3 or smaller (fp16 output rounding).\n");
    return 0;
}
