// =====================================================================
// FUSION EXAMPLE 1: residual add + RMSNorm  (every Llama-style layer)
//
//   h = x + residual                       (the residual stream)
//   y = h / sqrt(mean(h^2) + eps) * gamma  (RMSNorm)
//
// Both h and y are needed afterwards: h feeds the next residual add,
// y feeds the next matmul.
//
// UNFUSED = 3 kernels, every intermediate round-trips through DRAM:
//   k1 add:    read x, read r, write h      -> 3 tensor passes
//   k2 rms:    read h                       -> 1
//   k3 scale:  read h, write y              -> 2
//                                   total   =  6 tensor passes, 3 launches
// FUSED = 1 kernel, h stays in registers between the steps:
//   read x, read r, write h, write y        -> 4 tensor passes, 1 launch
//
// Predicted speedup ≈ 6/4 = 1.5x (each tensor is 64 MB, far too big
// for the 4 MB L2 to catch the round-trips).
//
// compile:  nvcc -O3 -arch=sm_75 fusion_add_rmsnorm.cu -o fusion_rms
// run:      ./fusion_rms
// profile:  nsys profile ./fusion_rms      (count the kernels)
//           ncu --set full ./fusion_rms    (DRAM bytes per kernel)
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

constexpr int ROWS = 4096;        // tokens
constexpr int H = 4096;           // hidden size
constexpr int THREADS = 256;
constexpr float EPS = 1e-6f;
constexpr double PEAK_GBS = 320.0;   // T4

__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ float blockSum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warpSum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / 32) ? s[lane] : 0.0f;
    if (warp == 0) v = warpSum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    float r = s[0];
    __syncthreads();
    return r;
}

// ---------------------------- UNFUSED -------------------------------
__global__ void k_add(const float* x, const float* r, float* h, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = x[i] + r[i];
}

__global__ void k_rms(const float* h, float* invRms, int hidden) {
    const float* row = h + (size_t)blockIdx.x * hidden;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) ss += row[j] * row[j];
    ss = blockSum(ss);
    if (threadIdx.x == 0) invRms[blockIdx.x] = rsqrtf(ss / hidden + EPS);
}

__global__ void k_scale(const float* h, const float* invRms, const float* gamma,
                        float* y, int hidden, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int row = (int)(i / hidden), col = (int)(i % hidden);
        y[i] = h[i] * invRms[row] * gamma[col];
    }
}

// ----------------------------- FUSED --------------------------------
// One block per row. The row of h lives in registers (float4 x 4 per
// thread) from the add, through the reduction, to the normalize.
template <int HIDDEN>
__global__ void k_fused_add_rmsnorm(const float* __restrict__ x, const float* __restrict__ r,
                                    const float* __restrict__ gamma,
                                    float* __restrict__ h, float* __restrict__ y) {
    static_assert(HIDDEN % (THREADS * 4) == 0, "HIDDEN must be a multiple of THREADS*4");
    constexpr int VEC = HIDDEN / (THREADS * 4);
    size_t base = (size_t)blockIdx.x * HIDDEN;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* r4 = reinterpret_cast<const float4*>(r + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    float4* h4 = reinterpret_cast<float4*>(h + base);
    float4* y4 = reinterpret_cast<float4*>(y + base);

    float4 v[VEC];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
        int idx = threadIdx.x + i * THREADS;
        float4 a = x4[idx], b = r4[idx];
        v[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);   // residual add
        h4[idx] = v[i];                                                   // h is needed later
        ss += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
    }
    float inv = rsqrtf(blockSum(ss) / HIDDEN + EPS);                      // RMS, no re-read

#pragma unroll
    for (int i = 0; i < VEC; i++) {
        int idx = threadIdx.x + i * THREADS;
        float4 g = g4[idx];
        y4[idx] = make_float4(v[i].x * inv * g.x, v[i].y * inv * g.y,
                              v[i].z * inv * g.z, v[i].w * inv * g.w);    // normalize from registers
    }
}

// ----------------------------- HOST ---------------------------------
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

double maxErr(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) e = std::max(e, (double)std::fabs(a[i] - b[i]));
    return e;
}

int main() {
    size_t n = (size_t)ROWS * H, bytes = n * sizeof(float);
    std::vector<float> hx(n), hr(n), hg(H), refH(n), refY(n), outH(n), outY(n);
    srand(1);
    for (size_t i = 0; i < n; i++) { hx[i] = rand() / (float)RAND_MAX - 0.5f;
                                     hr[i] = rand() / (float)RAND_MAX - 0.5f; }
    for (int j = 0; j < H; j++) hg[j] = 0.5f + rand() / (float)RAND_MAX;

    // CPU reference
    for (int row = 0; row < ROWS; row++) {
        double ss = 0;
        for (int j = 0; j < H; j++) {
            size_t i = (size_t)row * H + j;
            refH[i] = hx[i] + hr[i];
            ss += (double)refH[i] * refH[i];
        }
        double inv = 1.0 / std::sqrt(ss / H + EPS);
        for (int j = 0; j < H; j++) {
            size_t i = (size_t)row * H + j;
            refY[i] = (float)(refH[i] * inv * hg[j]);
        }
    }

    float *dx, *dr, *dg, *dh, *dy, *dinv;
    CHECK(cudaMalloc(&dx, bytes)); CHECK(cudaMalloc(&dr, bytes));
    CHECK(cudaMalloc(&dh, bytes)); CHECK(cudaMalloc(&dy, bytes));
    CHECK(cudaMalloc(&dg, H * sizeof(float))); CHECK(cudaMalloc(&dinv, ROWS * sizeof(float)));
    CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dr, hr.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dg, hg.data(), H * sizeof(float), cudaMemcpyHostToDevice));

    int ewBlocks = (int)((n + THREADS - 1) / THREADS);

    auto unfused = [&] {
        k_add  <<<ewBlocks, THREADS>>>(dx, dr, dh, n);
        k_rms  <<<ROWS,     THREADS>>>(dh, dinv, H);
        k_scale<<<ewBlocks, THREADS>>>(dh, dinv, dg, dy, H, n);
    };
    auto fused = [&] {
        k_fused_add_rmsnorm<H><<<ROWS, THREADS>>>(dx, dr, dg, dh, dy);
    };

    double mb = bytes / 1e6;
    printf("Each tensor: %.0f MB.  Ideal at %.0f GB/s:  unfused (6 passes) %.3f ms,  fused (4 passes) %.3f ms\n\n",
           mb, PEAK_GBS, 6 * bytes / (PEAK_GBS * 1e9) * 1e3, 4 * bytes / (PEAK_GBS * 1e9) * 1e3);

    float tU = timeIt(unfused);
    CHECK(cudaMemcpy(outH.data(), dh, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outY.data(), dy, bytes, cudaMemcpyDeviceToHost));
    printf("UNFUSED (3 kernels): %7.3f ms   eff %6.1f GB/s   err h %.1e  y %.1e\n",
           tU, 6 * bytes / (tU * 1e-3) / 1e9, maxErr(outH, refH), maxErr(outY, refY));

    CHECK(cudaMemset(dh, 0, bytes)); CHECK(cudaMemset(dy, 0, bytes));
    float tF = timeIt(fused);
    CHECK(cudaMemcpy(outH.data(), dh, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outY.data(), dy, bytes, cudaMemcpyDeviceToHost));
    printf("FUSED   (1 kernel):  %7.3f ms   eff %6.1f GB/s   err h %.1e  y %.1e\n",
           tF, 4 * bytes / (tF * 1e-3) / 1e9, maxErr(outH, refH), maxErr(outY, refY));

    printf("\nSpeedup: %.2fx   (prediction from bytes alone: 1.50x)\n", tU / tF);

    cudaFree(dx); cudaFree(dr); cudaFree(dg); cudaFree(dh); cudaFree(dy); cudaFree(dinv);
    return 0;
}
