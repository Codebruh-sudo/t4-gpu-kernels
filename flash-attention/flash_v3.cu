
// FLASHATTENTION FOR THE T4 -STAGE 3- occupancy + O in registers + prefetch

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
constexpr int BR = 64;                  // query rows per block (both stages)
constexpr int WARPS   = 4;
constexpr int THREADS = WARPS * 32;     // 128

// ============================ stage 2 (unchanged) =======================
namespace s2 {
constexpr int BC  = 64;
constexpr int LDK = D + 8, LDS = BC + 4, LDP = BC + 8, LDO = D + 4;
constexpr int KS_OFF = 0;
constexpr int VS_OFF = KS_OFF + BC * LDK * 2;
constexpr int SS_OFF = VS_OFF + BC * LDK * 2;
constexpr int OS_OFF = SS_OFF + BR * LDS * 4;
constexpr int SMEM_BYTES = OS_OFF + BR * LDO * 4;   // 53248

__global__ void __launch_bounds__(THREADS)
flash_v2(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    extern __shared__ __align__(32) unsigned char smem[];
    half*  Ks = reinterpret_cast<half*>(smem + KS_OFF);
    half*  Vs = reinterpret_cast<half*>(smem + VS_OFF);
    float* Ss = reinterpret_cast<float*>(smem + SS_OFF);
    float* Os = reinterpret_cast<float*>(smem + OS_OFF);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int bh = blockIdx.y, row0 = blockIdx.x * BR;
    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;
    float* Sw = Ss + warp * 16 * LDS;
    half*  Pw = reinterpret_cast<half*>(Sw);
    float* Ow = Os + warp * 16 * LDO;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
    const half* Qw = Q + ((size_t)bh * N + row0 + warp * 16) * D;
    for (int kk = 0; kk < D / 16; kk++) wmma::load_matrix_sync(qf[kk], Qw + kk * 16, D);

    const int r = lane >> 1, h = lane & 1;
    float m_r = -INFINITY, l_r = 0.0f;
    for (int c = 0; c < 32; c++) Ow[r * LDO + h * 32 + c] = 0.0f;
    __syncwarp();

    for (int t0 = 0; t0 < N; t0 += BC) {
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * THREADS, rr = idx / (D / 8), cc = idx % (D / 8);
            *reinterpret_cast<uint4*>(&Ks[rr * LDK + cc * 8]) =
                *reinterpret_cast<const uint4*>(&Kb[(size_t)(t0 + rr) * D + cc * 8]);
            *reinterpret_cast<uint4*>(&Vs[rr * LDK + cc * 8]) =
                *reinterpret_cast<const uint4*>(&Vb[(size_t)(t0 + rr) * D + cc * 8]);
        }
        __syncthreads();
        for (int j = 0; j < BC / 16; j++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
            wmma::fill_fragment(sacc, 0.0f);
            for (int kk = 0; kk < D / 16; kk++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, &Ks[(j * 16) * LDK + kk * 16], LDK);
                wmma::mma_sync(sacc, qf[kk], kf, sacc);
            }
            wmma::store_matrix_sync(&Sw[j * 16], sacc, LDS, wmma::mem_row_major);
        }
        __syncwarp();
        float sv[32], mx = -INFINITY;
        for (int c = 0; c < 32; c++) { sv[c] = Sw[r * LDS + h * 32 + c] * scale; mx = fmaxf(mx, sv[c]); }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        float m_new = fmaxf(m_r, mx), corr = __expf(m_r - m_new);
        __syncwarp();
        float sum = 0.0f;
        for (int c = 0; c < 32; c++) {
            float p = __expf(sv[c] - m_new);
            sum += p;
            Pw[r * LDP + h * 32 + c] = __float2half(p);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        l_r = l_r * corr + sum;
        m_r = m_new;
        for (int c = 0; c < 32; c++) Ow[r * LDO + h * 32 + c] *= corr;
        __syncwarp();
        for (int c = 0; c < D / 16; c++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc;
            wmma::load_matrix_sync(oacc, &Ow[c * 16], LDO, wmma::mem_row_major);
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
        __syncthreads();
    }
    half* Og = O + ((size_t)bh * N + row0 + warp * 16 + r) * D + h * 32;
    float inv = 1.0f / l_r;
    for (int c = 0; c < 32; c++) Og[c] = __float2half(Ow[r * LDO + h * 32 + c] * inv);
}
}  // namespace s2

// ============================ stage 3 ==================================
constexpr int BC  = 32;                 // keys per tile (smaller -> less shared memory)
constexpr int LDK = D + 8;              // 72 halves  (K, V tiles)
constexpr int LDS = BC + 4;             // 36 floats  (S tile)
constexpr int LDP = BC + 8;             // 40 halves  (P tile, inside each warp's S region)

constexpr int KS_OFF   = 0;
constexpr int VS_OFF   = KS_OFF + BC * LDK * 2;          //  4608
constexpr int SS_OFF   = VS_OFF + BC * LDK * 2;          //  9216
constexpr int CORR_OFF = SS_OFF + BR * LDS * 4;          // 18432
constexpr int SMEM3    = CORR_OFF + WARPS * 16 * 4;      // 18688 bytes
constexpr int OUT_BYTES = BR * D * 4;                    // 16384: final O staging (reuses K/V/S space)
static_assert(OUT_BYTES <= CORR_OFF, "final O staging must fit in the K/V/S region");

__global__ void __launch_bounds__(THREADS, 3)
flash_v3(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    __shared__ __align__(32) unsigned char smem[SMEM3];
    half*  Ks    = reinterpret_cast<half*>(smem + KS_OFF);
    half*  Vs    = reinterpret_cast<half*>(smem + VS_OFF);
    float* Ss    = reinterpret_cast<float*>(smem + SS_OFF);
    float* CorrS = reinterpret_cast<float*>(smem + CORR_OFF);

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh   = blockIdx.y;
    const int row0 = blockIdx.x * BR;

    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;

    float* Sw    = Ss + warp * 16 * LDS;            // this warp's S region (16 x 32)
    half*  Pw    = reinterpret_cast<half*>(Sw);     // P reuses it
    float* corrW = CorrS + warp * 16;               // per-row factors for this warp

    // ---- row map: which row does each accumulator element belong to? ----
    //      load a 16x16 matrix whose value at (r, c) is r, then read it back
    for (int idx = lane; idx < 256; idx += 32) Sw[(idx / 16) * LDS + (idx % 16)] = (float)(idx / 16);
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowFrag;
    wmma::load_matrix_sync(rowFrag, Sw, LDS, wmma::mem_row_major);
    int rowOf[8];                                   // accumulator 16x16 float: 8 elements per thread
#pragma unroll
    for (int i = 0; i < 8; i++) rowOf[i] = (int)rowFrag.x[i];
    __syncwarp();

    // ---- Q fragments (registers, whole kernel) ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
    const half* Qw = Q + ((size_t)bh * N + row0 + warp * 16) * D;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) wmma::load_matrix_sync(qf[kk], Qw + kk * 16, D);

    // ---- O accumulators (registers, whole kernel): 16 rows x 64 dims = 4 fragments ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D / 16];
#pragma unroll
    for (int c = 0; c < D / 16; c++) wmma::fill_fragment(oacc[c], 0.0f);

    // ---- softmax bookkeeping: 2 lanes per row, 16 columns each ----
    const int r = lane >> 1;
    const int h = lane & 1;
    float m_r = -INFINITY, l_r = 0.0f;

    // ---- K/V prefetch registers: each tile is 32x64 halves = 256 chunks of 16 B -> 2 per thread ----
    uint4 kReg[2], vReg[2];
    auto loadKV = [&](int t0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS, rr = idx / (D / 8), cc = idx % (D / 8);
            kReg[i] = *reinterpret_cast<const uint4*>(&Kb[(size_t)(t0 + rr) * D + cc * 8]);
            vReg[i] = *reinterpret_cast<const uint4*>(&Vb[(size_t)(t0 + rr) * D + cc * 8]);
        }
    };
    auto storeKV = [&]() {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * THREADS, rr = idx / (D / 8), cc = idx % (D / 8);
            *reinterpret_cast<uint4*>(&Ks[rr * LDK + cc * 8]) = kReg[i];
            *reinterpret_cast<uint4*>(&Vs[rr * LDK + cc * 8]) = vReg[i];
        }
    };

    // prologue: tile 0 into shared memory
    loadKV(0);
    storeKV();
    __syncthreads();

    for (int t0 = 0; t0 < N; t0 += BC) {
        const bool hasNext = (t0 + BC < N);
        if (hasNext) loadKV(t0 + BC);               // next tile travels while we compute

        // ---------- S = Q K^T  (16 rows x 32 keys per warp) ----------
#pragma unroll
        for (int j = 0; j < BC / 16; j++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
            wmma::fill_fragment(sacc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, &Ks[(j * 16) * LDK + kk * 16], LDK);
                wmma::mma_sync(sacc, qf[kk], kf, sacc);
            }
            wmma::store_matrix_sync(&Sw[j * 16], sacc, LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // ---------- online softmax for this tile ----------
        float sv[16], mx = -INFINITY;
#pragma unroll
        for (int c = 0; c < 16; c++) {
            sv[c] = Sw[r * LDS + h * 16 + c] * scale;
            mx = fmaxf(mx, sv[c]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        float m_new = fmaxf(m_r, mx);
        float corr  = __expf(m_r - m_new);
        __syncwarp();                               // S fully read before P overwrites it

        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 16; c++) {
            float p = __expf(sv[c] - m_new);
            sum += p;
            Pw[r * LDP + h * 16 + c] = __float2half(p);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        l_r = l_r * corr + sum;
        m_r = m_new;
        if (h == 0) corrW[r] = corr;
        __syncwarp();

        // ---------- rescale O in registers, using the row map ----------
#pragma unroll
        for (int c = 0; c < D / 16; c++)
#pragma unroll
            for (int i = 0; i < 8; i++) oacc[c].x[i] *= corrW[rowOf[i]];

        // ---------- O += P V  (16 rows x 64 dims per warp) ----------
#pragma unroll
        for (int k = 0; k < BC / 16; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, &Pw[k * 16], LDP);
#pragma unroll
            for (int c = 0; c < D / 16; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, &Vs[(k * 16) * LDK + c * 16], LDK);
                wmma::mma_sync(oacc[c], pf, vf, oacc[c]);
            }
        }

        __syncthreads();                            // all warps done with this K/V tile
        if (hasNext) {
            storeKV();                              // prefetched tile -> shared memory
            __syncthreads();
        }
    }

    // ---------- final: O / l, staged through shared memory, write fp16 ----------
    if (h == 0) corrW[r] = 1.0f / l_r;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < D / 16; c++)
#pragma unroll
        for (int i = 0; i < 8; i++) oacc[c].x[i] *= corrW[rowOf[i]];

    float* Ostage = reinterpret_cast<float*>(smem) + warp * 16 * D;   // reuses K/V/S space (loop is done)
#pragma unroll
    for (int c = 0; c < D / 16; c++)
        wmma::store_matrix_sync(&Ostage[c * 16], oacc[c], D, wmma::mem_row_major);
    __syncwarp();

    half* Og = O + ((size_t)bh * N + row0 + warp * 16 + r) * D + h * 32;
#pragma unroll
    for (int c = 0; c < 32; c++) Og[c] = __float2half(Ostage[r * D + h * 32 + c]);
}

// ============================ helpers ==================================
template <typename F>
float timeIt(F launch, int iters = 20) {
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
    int checks[10][2] = {{0, 0}, {0, 15}, {0, N / 2}, {0, N - 1}, {3, 7}, {5, N / 3},
                         {H - 1, 0}, {H - 1, 17}, {H - 1, N / 2}, {H - 1, N - 1}};
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

    CHECK(cudaFuncSetAttribute(s2::flash_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, s2::SMEM_BYTES));

    int bps2 = 0, bps3 = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps2, s2::flash_v2, THREADS, s2::SMEM_BYTES));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps3, flash_v3, THREADS, 0));

    printf("FlashAttention on the T4: stage 2 vs stage 3   (B=%d H=%d D=%d, fp16)\n", B, H, D);
    printf("Blocks per SM:  stage 2 = %d (%d warps)   stage 3 = %d (%d warps)\n", bps2, bps2 * WARPS, bps3, bps3 * WARPS);
    printf("PyTorch efficient kernel on your T4: ~0.093 / 0.319 / 0.765 / 2.636 / 10.452 ms (~13 TFLOP/s)\n\n");
    printf("%6s | %22s | %22s | %8s\n", "N", "stage 2  ms (TFLOP/s)", "stage 3  ms (TFLOP/s)", "speedup");
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
        float t2 = timeIt([&] { s2::flash_v2<<<grid, THREADS, s2::SMEM_BYTES>>>(dQ, dK, dV, dO, N, scale); });
        CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(half), cudaMemcpyDeviceToHost));
        double e2 = checkRows(hQ, hK, hV, hO, N, scale);

        CHECK(cudaMemset(dO, 0, elems * sizeof(half)));
        float t3 = timeIt([&] { flash_v3<<<grid, THREADS>>>(dQ, dK, dV, dO, N, scale); });
        CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(half), cudaMemcpyDeviceToHost));
        double e3 = checkRows(hQ, hK, hV, hO, N, scale);

        printf("%6d | %10.3f (%6.2f)    | %10.3f (%6.2f)    | %6.2fx\n",
               N, t2, flops / (t2 * 1e-3) / 1e12, t3, flops / (t3 * 1e-3) / 1e12, t2 / t3);
        printf("        | %22.1e | %22.1e |\n", e2, e3);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }
    return 0;
}
