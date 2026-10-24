
// FLASHATTENTION FOR THE T4 — STAGE 2: tensor cores + per-tile rescale


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int B  = 1;
constexpr int H  = 8;
constexpr int D  = 64;
constexpr int BR = 64;           // query rows per block
constexpr int BC = 64;           // keys per tile

// ============================ stage 1 ==================================
__global__ void __launch_bounds__(BR)
flash_v1(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    int bh = blockIdx.y, row = blockIdx.x * BR + threadIdx.x;
    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];
    float q[D], o[D];
    const half* qp = Q + ((size_t)bh * N + row) * D;
#pragma unroll
    for (int d = 0; d < D; d++) { q[d] = __half2float(qp[d]) * scale; o[d] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int t0 = 0; t0 < N; t0 += BC) {
        for (int i = threadIdx.x; i < BC * D; i += BR) {
            int r = i / D, c = i % D;
            Ks[r][c] = __half2float(Kb[(size_t)(t0 + r) * D + c]);
            Vs[r][c] = __half2float(Vb[(size_t)(t0 + r) * D + c]);
        }
        __syncthreads();
        for (int j = 0; j < BC; j++) {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; d++) s += q[d] * Ks[j][d];
            float m_new = fmaxf(m, s), corr = __expf(m - m_new), p = __expf(s - m_new);
            l = l * corr + p;
#pragma unroll
            for (int d = 0; d < D; d++) o[d] = o[d] * corr + p * Vs[j][d];
            m = m_new;
        }
        __syncthreads();
    }
    half* op = O + ((size_t)bh * N + row) * D;
#pragma unroll
    for (int d = 0; d < D; d++) op[d] = __float2half(o[d] / l);
}

// ============================ stage 2 ==================================
constexpr int WARPS   = 4;
constexpr int THREADS = WARPS * 32;     // 128
constexpr int LDK = D + 8;              // 72 halves  (K and V tiles, padded)
constexpr int LDS = BC + 4;             // 68 floats  (S tile, padded)
constexpr int LDP = BC + 8;             // 72 halves  (P tile, lives inside the S region)
constexpr int LDO = D + 4;              // 68 floats  (running O, padded)

// shared memory layout (bytes)
constexpr int KS_OFF = 0;
constexpr int VS_OFF = KS_OFF + BC * LDK * 2;        //  9216
constexpr int SS_OFF = VS_OFF + BC * LDK * 2;        // 18432
constexpr int OS_OFF = SS_OFF + BR * LDS * 4;        // 35840
constexpr int SMEM_BYTES = OS_OFF + BR * LDO * 4;    // 53248  (> 48 KB -> opt-in)

__global__ void __launch_bounds__(THREADS)
flash_v2(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    extern __shared__ __align__(32) unsigned char smem[];
    half*  Ks = reinterpret_cast<half*>(smem + KS_OFF);
    half*  Vs = reinterpret_cast<half*>(smem + VS_OFF);
    float* Ss = reinterpret_cast<float*>(smem + SS_OFF);
    float* Os = reinterpret_cast<float*>(smem + OS_OFF);

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh   = blockIdx.y;
    const int row0 = blockIdx.x * BR;                    // first query row of this block

    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;

    // this warp's private slices of S, P (aliases S) and O
    float* Sw = Ss + warp * 16 * LDS;
    half*  Pw = reinterpret_cast<half*>(Sw);            // P overwrites S after S is consumed
    float* Ow = Os + warp * 16 * LDO;

    // ---- Q fragments for this warp's 16 rows: loaded once, kept in registers ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
    const half* Qw = Q + ((size_t)bh * N + row0 + warp * 16) * D;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) wmma::load_matrix_sync(qf[kk], Qw + kk * 16, D);

    // ---- softmax bookkeeping: 2 lanes per row ----
    const int r = lane >> 1;          // row 0..15 within this warp
    const int h = lane & 1;           // which half of the 64 columns (32 each)
    float m_r = -INFINITY;            // running max of row r   (both lanes keep a copy)
    float l_r = 0.0f;                 // running sum of row r

    // zero this warp's O rows
    for (int c = 0; c < 32; c++) Ow[r * LDO + h * 32 + c] = 0.0f;
    __syncwarp();

    for (int t0 = 0; t0 < N; t0 += BC) {
        // ---------- 1. load K and V tiles (64 x 64 halves each) ----------
        //            512 chunks of 16 bytes each -> 4 per thread, coalesced
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * THREADS;
            int rr = idx / (D / 8), cc = idx % (D / 8);
            *reinterpret_cast<uint4*>(&Ks[rr * LDK + cc * 8]) =
                *reinterpret_cast<const uint4*>(&Kb[(size_t)(t0 + rr) * D + cc * 8]);
            *reinterpret_cast<uint4*>(&Vs[rr * LDK + cc * 8]) =
                *reinterpret_cast<const uint4*>(&Vb[(size_t)(t0 + rr) * D + cc * 8]);
        }
        __syncthreads();

        // ---------- 2. S = Q K^T on tensor cores (16 rows x 64 keys per warp) ----------
#pragma unroll
        for (int j = 0; j < BC / 16; j++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
            wmma::fill_fragment(sacc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                // col_major view of K rows (j*16 .. +16), dims (kk*16 .. +16)  ==  K^T block
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, &Ks[(j * 16) * LDK + kk * 16], LDK);
                wmma::mma_sync(sacc, qf[kk], kf, sacc);
            }
            wmma::store_matrix_sync(&Sw[j * 16], sacc, LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // ---------- 3. online softmax, once per tile ----------
        float sv[32];
        float mx = -INFINITY;
#pragma unroll
        for (int c = 0; c < 32; c++) {
            sv[c] = Sw[r * LDS + h * 32 + c] * scale;
            mx = fmaxf(mx, sv[c]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));    // combine the two halves of the row

        float m_new = fmaxf(m_r, mx);
        float corr  = __expf(m_r - m_new);                      // first tile: exp(-inf) = 0
        __syncwarp();                                           // all of S read before P overwrites it

        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 32; c++) {
            float p = __expf(sv[c] - m_new);
            sum += p;
            Pw[r * LDP + h * 32 + c] = __float2half(p);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        l_r = l_r * corr + sum;
        m_r = m_new;

#pragma unroll
        for (int c = 0; c < 32; c++) Ow[r * LDO + h * 32 + c] *= corr;   // rescale ONCE per tile
        __syncwarp();

        // ---------- 4. O += P V on tensor cores (16 rows x 64 dims per warp) ----------
#pragma unroll
        for (int c = 0; c < D / 16; c++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc;
            wmma::load_matrix_sync(oacc, &Ow[c * 16], LDO, wmma::mem_row_major);
#pragma unroll
            for (int k = 0; k < BC / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &Pw[k * 16], LDP);
                wmma::load_matrix_sync(vf, &Vs[(k * 16) * LDK + c * 16], LDK);
                wmma::mma_sync(oacc, pf, vf, oacc);
            }
            wmma::store_matrix_sync(&Ow[c * 16], oacc, LDO, wmma::mem_row_major);
        }
        __syncwarp();
        __syncthreads();      // everyone done with K/V before the next tile overwrites them
    }

    // ---------- final: O = O / l, write fp16 ----------
    half* Og = O + ((size_t)bh * N + row0 + warp * 16 + r) * D + h * 32;
    float inv = 1.0f / l_r;
#pragma unroll
    for (int c = 0; c < 32; c++) Og[c] = __float2half(Ow[r * LDO + h * 32 + c] * inv);
}

// ============================ helpers ==================================
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

double checkRows(const std::vector<half>& hQ, const std::vector<half>& hK, const std::vector<half>& hV,
                 const std::vector<half>& hO, int N, float scale) {
    int checks[8][2] = {{0, 0}, {0, N / 2}, {0, N - 1}, {3, 7}, {5, N / 3},
                        {H - 1, 0}, {H - 1, N / 2}, {H - 1, N - 1}};
    double ref[D], maxErr = 0;
    for (auto& c : checks) {
        cpuRow(hQ, hK, hV, N, c[0], c[1], scale, ref);
        for (int d = 0; d < D; d++)
            maxErr = std::max(maxErr, std::fabs(
                (double)__half2float(hO[((size_t)c[0] * N + c[1]) * D + d]) - ref[d]));
    }
    return maxErr;
}

int main() {
    const float scale = 1.0f / std::sqrt((float)D);
    int Ns[] = {512, 1024, 2048, 4096, 8192};

    CHECK(cudaFuncSetAttribute(flash_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    printf("FlashAttention on the T4: stage 1 (CUDA cores) vs stage 2 (tensor cores)\n");
    printf("B=%d H=%d D=%d, fp16 in/out. PyTorch efficient kernel reached ~13 TFLOP/s on your T4.\n\n", B, H, D);
    printf("%6s | %22s | %22s | %8s\n", "N", "stage 1  ms (TFLOP/s)", "stage 2  ms (TFLOP/s)", "speedup");
    printf("        | %22s | %22s |\n", "max error", "max error");
    printf("--------------------------------------------------------------------------\n");

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

        dim3 grid(N / BR, B * H);
        double flops = 4.0 * B * H * (double)N * N * D;

        CHECK(cudaMemset(dO, 0, elems * sizeof(half)));
        float t1 = timeIt([&] { flash_v1<<<grid, BR>>>(dQ, dK, dV, dO, N, scale); });
        CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(half), cudaMemcpyDeviceToHost));
        double e1 = checkRows(hQ, hK, hV, hO, N, scale);

        CHECK(cudaMemset(dO, 0, elems * sizeof(half)));
        float t2 = timeIt([&] { flash_v2<<<grid, THREADS, SMEM_BYTES>>>(dQ, dK, dV, dO, N, scale); });
        CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(half), cudaMemcpyDeviceToHost));
        double e2 = checkRows(hQ, hK, hV, hO, N, scale);

        printf("%6d | %10.3f (%6.2f)    | %10.3f (%6.2f)    | %6.2fx\n",
               N, t1, flops / (t1 * 1e-3) / 1e12, t2, flops / (t2 * 1e-3) / 1e12, t1 / t2);
        printf("        | %22.1e | %22.1e |\n", e1, e2);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }
    printf("\nErrors should stay around 1e-3 or smaller (stage 2 rounds P to fp16, so expect a bit more than stage 1).\n");
    return 0;
}
