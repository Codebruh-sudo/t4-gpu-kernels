// =====================================================================
// Softmax optimization walkthrough — 4 versions, timed and verified.
//
// Problem: row-wise softmax on a [R x C] fp32 matrix (R = C = 4096).
//
// compile:  nvcc -O3 -arch=sm_75 softmax_optimization.cu -o softmax_opt
//           (sm_75 = Turing / T4. Use your GPU's arch otherwise.)
// run:      ./softmax_opt
// profile:  ncu --set full -k regex:softmax ./softmax_opt
//           (look at: DRAM throughput %, L1/L2 hit rate, memory
//            transactions per request, achieved occupancy)
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

constexpr int R = 4096;          // rows (e.g. tokens)
constexpr int C = 4096;          // cols (e.g. vocab / sequence length)
constexpr int THREADS = 256;     // threads per block for v2..v4
constexpr double PEAK_GBS = 320.0;   // T4 peak DRAM bandwidth (GB/s)

// ---------------------------------------------------------------------
// Block-wide reductions built from warp shuffles (no shared-memory tree,
// only one __syncthreads round). Used by v2..v4.
// ---------------------------------------------------------------------
__device__ __forceinline__ float warpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ float blockMax(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warpMax(v);                          // 1) reduce inside each warp
    if (lane == 0) s[warp] = v;              // 2) one value per warp -> shared
    __syncthreads();
    int nWarps = blockDim.x / 32;
    v = (threadIdx.x < nWarps) ? s[lane] : -INFINITY;
    if (warp == 0) v = warpMax(v);           // 3) warp 0 reduces the warp results
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    float r = s[0];                          // 4) broadcast to every thread
    __syncthreads();                         //    (so s can be reused safely)
    return r;
}
__device__ float blockSum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warpSum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    int nWarps = blockDim.x / 32;
    v = (threadIdx.x < nWarps) ? s[lane] : 0.0f;
    if (warp == 0) v = warpSum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    float r = s[0];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------------
// v1 — NAIVE: one thread per row, 3 passes over global memory.
// Bottleneck: UNCOALESCED. Neighbouring threads read different rows,
// C floats apart, so each warp load touches 32 different cache lines.
// ---------------------------------------------------------------------
__global__ void softmax_v1(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* x = in  + (size_t)row * cols;
    float*       y = out + (size_t)row * cols;

    float m = -INFINITY;
    for (int j = 0; j < cols; j++) m = fmaxf(m, x[j]);           // pass 1: max
    float s = 0.0f;
    for (int j = 0; j < cols; j++) s += expf(x[j] - m);          // pass 2: sum
    for (int j = 0; j < cols; j++) y[j] = expf(x[j] - m) / s;    // pass 3: write
}

// ---------------------------------------------------------------------
// v2 — ONE BLOCK PER ROW: threads stride across the row -> COALESCED.
// Still 3 reads + 1 write of the row (but re-reads may hit L1/L2).
// ---------------------------------------------------------------------
__global__ void softmax_v2(const float* in, float* out, int cols) {
    const float* x = in  + (size_t)blockIdx.x * cols;
    float*       y = out + (size_t)blockIdx.x * cols;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, x[j]);
    m = blockMax(m);

    float s = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) s += expf(x[j] - m);
    s = blockSum(s);

    for (int j = threadIdx.x; j < cols; j += blockDim.x) y[j] = expf(x[j] - m) / s;
}

// ---------------------------------------------------------------------
// v3 — ONLINE SOFTMAX: max and sum computed in ONE pass.
// Each thread keeps a running (m, l), rescaling l when its max grows.
// Then the block combines them. 2 reads + 1 write.
// ---------------------------------------------------------------------
__global__ void softmax_v3(const float* in, float* out, int cols) {
    const float* x = in  + (size_t)blockIdx.x * cols;
    float*       y = out + (size_t)blockIdx.x * cols;

    float m = -INFINITY, l = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = x[j];
        float mNew = fmaxf(m, v);
        l = l * expf(m - mNew) + expf(v - mNew);    // the online-softmax rescale
        m = mNew;
    }
    float M = blockMax(m);                           // row max
    float L = blockSum(l * expf(m - M));             // move every partial sum to the row max
    float inv = 1.0f / L;

    for (int j = threadIdx.x; j < cols; j += blockDim.x) y[j] = expf(x[j] - M) * inv;
}

// ---------------------------------------------------------------------
// v4 — ROW IN REGISTERS + VECTORIZED float4 LOADS.
// The whole row (16 KB) is held in registers across the block:
// 256 threads x 4 float4 = 4096 floats. Exactly 1 read + 1 write.
// This is the bandwidth-optimal version.
// ---------------------------------------------------------------------
template <int COLS>
__global__ void softmax_v4(const float* __restrict__ in, float* __restrict__ out) {
    static_assert(COLS % (THREADS * 4) == 0, "COLS must be a multiple of THREADS*4");
    constexpr int VEC = COLS / (THREADS * 4);        // float4 per thread (= 4 here)

    const float4* x = reinterpret_cast<const float4*>(in  + (size_t)blockIdx.x * COLS);
    float4*       y = reinterpret_cast<float4*>      (out + (size_t)blockIdx.x * COLS);

    float4 v[VEC];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
        v[i] = x[threadIdx.x + i * THREADS];         // coalesced 16-byte loads
        m = fmaxf(m, fmaxf(fmaxf(v[i].x, v[i].y), fmaxf(v[i].z, v[i].w)));
    }
    m = blockMax(m);

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
        v[i].x = expf(v[i].x - m); v[i].y = expf(v[i].y - m);
        v[i].z = expf(v[i].z - m); v[i].w = expf(v[i].w - m);
        s += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    s = blockSum(s);
    float inv = 1.0f / s;

#pragma unroll
    for (int i = 0; i < VEC; i++) {
        v[i].x *= inv; v[i].y *= inv; v[i].z *= inv; v[i].w *= inv;
        y[threadIdx.x + i * THREADS] = v[i];        // coalesced 16-byte stores
    }
}

// ---------------------------------------------------------------------
// Host helpers: CPU reference, verification, timing
// ---------------------------------------------------------------------
void cpuSoftmax(const std::vector<float>& in, std::vector<float>& out) {
    for (int r = 0; r < R; r++) {
        const float* x = &in[(size_t)r * C];
        double m = -INFINITY;
        for (int j = 0; j < C; j++) m = std::max(m, (double)x[j]);
        double s = 0;
        for (int j = 0; j < C; j++) s += std::exp(x[j] - m);
        for (int j = 0; j < C; j++) out[(size_t)r * C + j] = (float)(std::exp(x[j] - m) / s);
    }
}

double maxAbsErr(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) e = std::max(e, (double)std::fabs(a[i] - b[i]));
    return e;
}

template <typename F>
float timeIt(F launch, int iters = 20) {
    launch();                                         // warm-up
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

void report(const char* name, float ms, const std::vector<float>& got, const std::vector<float>& ref) {
    // "Effective" bandwidth counts only the IDEAL traffic: 1 read + 1 write.
    double idealBytes = 2.0 * R * C * sizeof(float);
    double gbs = idealBytes / (ms * 1e-3) / 1e9;
    printf("%-28s %8.3f ms   eff. %6.1f GB/s  (%5.1f%% of peak)   max err %.2e\n",
           name, ms, gbs, 100.0 * gbs / PEAK_GBS, maxAbsErr(got, ref));
}

int main() {
    size_t n = (size_t)R * C, bytes = n * sizeof(float);
    std::vector<float> h_in(n), h_out(n), h_ref(n);
    srand(42);
    for (size_t i = 0; i < n; i++) h_in[i] = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;

    printf("Computing CPU reference...\n");
    cpuSoftmax(h_in, h_ref);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    double idealMs = 2.0 * bytes / (PEAK_GBS * 1e9) * 1e3;
    printf("\nIdeal (1 read + 1 write at %.0f GB/s peak): %.3f ms\n\n", PEAK_GBS, idealMs);

    float ms;

    CHECK(cudaMemset(d_out, 0, bytes));
    ms = timeIt([&] { softmax_v1<<<(R + THREADS - 1) / THREADS, THREADS>>>(d_in, d_out, R, C); });
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    report("v1 naive (thread/row)", ms, h_out, h_ref);

    CHECK(cudaMemset(d_out, 0, bytes));
    ms = timeIt([&] { softmax_v2<<<R, THREADS>>>(d_in, d_out, C); });
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    report("v2 block/row, coalesced", ms, h_out, h_ref);

    CHECK(cudaMemset(d_out, 0, bytes));
    ms = timeIt([&] { softmax_v3<<<R, THREADS>>>(d_in, d_out, C); });
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    report("v3 online softmax", ms, h_out, h_ref);

    CHECK(cudaMemset(d_out, 0, bytes));
    ms = timeIt([&] { softmax_v4<C><<<R, THREADS>>>(d_in, d_out); });
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    report("v4 registers + float4", ms, h_out, h_ref);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    return 0;
}
