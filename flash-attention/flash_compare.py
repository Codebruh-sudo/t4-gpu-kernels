# =====================================================================
# FAIR COMPARISON: your FlashAttention (stage 3) vs PyTorch, same run
#
# - compiles flash_v3 INSIDE PyTorch (load_inline) -> callable as flash(q, k, v)
# - times your kernel and PyTorch's memory-efficient SDPA kernel in
#   ALTERNATING rounds on the same GPU, and reports the MEDIAN
#   (so a GPU clock change affects both equally)
# - checks correctness against an fp32 reference and against PyTorch
#
# run in Colab:   !python flash_compare.py
# (first run compiles the kernel: about 1-2 minutes)
# =====================================================================

import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")   # T4: compile only for sm_75 (faster build)

import math
import statistics
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
using namespace nvcuda;

constexpr int D  = 64;
constexpr int BR = 64;
constexpr int WARPS   = 4;
constexpr int THREADS = WARPS * 32;
constexpr int BC  = 32;
constexpr int LDK = D + 8;
constexpr int LDS = BC + 4;
constexpr int LDP = BC + 8;
constexpr int KS_OFF   = 0;
constexpr int VS_OFF   = KS_OFF + BC * LDK * 2;
constexpr int SS_OFF   = VS_OFF + BC * LDK * 2;
constexpr int CORR_OFF = SS_OFF + BR * LDS * 4;
constexpr int SMEM3    = CORR_OFF + WARPS * 16 * 4;

__global__ void __launch_bounds__(THREADS, 3)
flash_v3(const half* __restrict__ Q, const half* __restrict__ K,
         const half* __restrict__ V, half* __restrict__ O, int N, float scale) {
    __shared__ __align__(32) unsigned char smem[SMEM3];
    half*  Ks    = reinterpret_cast<half*>(smem + KS_OFF);
    half*  Vs    = reinterpret_cast<half*>(smem + VS_OFF);
    float* Ss    = reinterpret_cast<float*>(smem + SS_OFF);
    float* CorrS = reinterpret_cast<float*>(smem + CORR_OFF);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int bh = blockIdx.y, row0 = blockIdx.x * BR;
    const half* Kb = K + (size_t)bh * N * D;
    const half* Vb = V + (size_t)bh * N * D;
    float* Sw    = Ss + warp * 16 * LDS;
    half*  Pw    = reinterpret_cast<half*>(Sw);
    float* corrW = CorrS + warp * 16;

    for (int idx = lane; idx < 256; idx += 32) Sw[(idx / 16) * LDS + (idx % 16)] = (float)(idx / 16);
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowFrag;
    wmma::load_matrix_sync(rowFrag, Sw, LDS, wmma::mem_row_major);
    int rowOf[8];
#pragma unroll
    for (int i = 0; i < 8; i++) rowOf[i] = (int)rowFrag.x[i];
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
    const half* Qw = Q + ((size_t)bh * N + row0 + warp * 16) * D;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) wmma::load_matrix_sync(qf[kk], Qw + kk * 16, D);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D / 16];
#pragma unroll
    for (int c = 0; c < D / 16; c++) wmma::fill_fragment(oacc[c], 0.0f);

    const int r = lane >> 1, h = lane & 1;
    float m_r = -INFINITY, l_r = 0.0f;

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

    loadKV(0);
    storeKV();
    __syncthreads();

    for (int t0 = 0; t0 < N; t0 += BC) {
        const bool hasNext = (t0 + BC < N);
        if (hasNext) loadKV(t0 + BC);
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
        float sv[16], mx = -INFINITY;
#pragma unroll
        for (int c = 0; c < 16; c++) { sv[c] = Sw[r * LDS + h * 16 + c] * scale; mx = fmaxf(mx, sv[c]); }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        float m_new = fmaxf(m_r, mx), corr = __expf(m_r - m_new);
        __syncwarp();
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
#pragma unroll
        for (int c = 0; c < D / 16; c++)
#pragma unroll
            for (int i = 0; i < 8; i++) oacc[c].x[i] *= corrW[rowOf[i]];
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
        __syncthreads();
        if (hasNext) { storeKV(); __syncthreads(); }
    }

    if (h == 0) corrW[r] = 1.0f / l_r;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < D / 16; c++)
#pragma unroll
        for (int i = 0; i < 8; i++) oacc[c].x[i] *= corrW[rowOf[i]];
    float* Ostage = reinterpret_cast<float*>(smem) + warp * 16 * D;
#pragma unroll
    for (int c = 0; c < D / 16; c++)
        wmma::store_matrix_sync(&Ostage[c * 16], oacc[c], D, wmma::mem_row_major);
    __syncwarp();
    half* Og = O + ((size_t)bh * N + row0 + warp * 16 + r) * D + h * 32;
#pragma unroll
    for (int c = 0; c < 32; c++) Og[c] = __float2half(Ostage[r * D + h * 32 + c]);
}

// PyTorch-facing wrapper: q, k, v are [B, H, N, 64] fp16 contiguous CUDA tensors
torch::Tensor flash(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "q must be fp16 CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(q.size(3) == D, "head dim must be 64");
    TORCH_CHECK(q.size(2) % BR == 0, "sequence length must be a multiple of 64");
    int BH = q.size(0) * q.size(1), N = q.size(2);
    auto o = torch::empty_like(q);
    dim3 grid(N / BR, BH);
    float scale = 1.0f / sqrtf((float)D);
    flash_v3<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(o.data_ptr<at::Half>()), N, scale);
    return o;
}
'''

cpp_src = "torch::Tensor flash(torch::Tensor q, torch::Tensor k, torch::Tensor v);"

print("Compiling your kernel inside PyTorch (first time: ~1-2 min)...")
ext = load_inline(name="flash_t4", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["flash"], extra_cuda_cflags=["-O3"], verbose=False)
print("Done.  GPU:", torch.cuda.get_device_name(0), "| PyTorch", torch.__version__, "\n")

from torch.nn.attention import sdpa_kernel, SDPBackend

def pytorch_efficient(q, k, v):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)

def time_ms(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# warm the GPU up so clocks are at full speed before measuring anything
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(50):
    x @ x
torch.cuda.synchronize()

B, H, D = 1, 8, 64
ROUNDS = 7
print(f"B={B} H={H} D={D} fp16 | {ROUNDS} alternating rounds, median shown\n")
print(f"{'N':>6} | {'yours ms':>9} {'TFLOP/s':>8} | {'PyTorch ms':>10} {'TFLOP/s':>8} | {'ratio':>6} | "
      f"{'err vs fp32':>11} {'diff vs PyTorch':>15}")
print("-" * 100)

for N in [512, 1024, 2048, 4096, 8192]:
    torch.manual_seed(N)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q) * 0.5

    # correctness
    out_yours = ext.flash(q, k, v)
    out_torch = pytorch_efficient(q, k, v)
    diff_torch = (out_yours.float() - out_torch.float()).abs().max().item()
    if N <= 4096:
        scale = 1.0 / math.sqrt(D)
        ref = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * scale, dim=-1) @ v.float()
        err_ref = f"{(out_yours.float() - ref).abs().max().item():11.1e}"
        del ref
    else:
        err_ref = f"{'(skipped)':>11}"

    # alternating rounds: yours, PyTorch, yours, PyTorch, ...
    t_yours, t_torch = [], []
    for _ in range(ROUNDS):
        t_yours.append(time_ms(lambda: ext.flash(q, k, v)))
        t_torch.append(time_ms(lambda: pytorch_efficient(q, k, v)))
    ty, tt = statistics.median(t_yours), statistics.median(t_torch)

    flops = 4.0 * B * H * N * N * D
    print(f"{N:>6} | {ty:9.3f} {flops / ty / 1e9:8.2f} | {tt:10.3f} {flops / tt / 1e9:8.2f} | "
          f"{tt / ty:5.2f}x | {err_ref} {diff_torch:15.1e}")

print("\nratio = PyTorch time / your time   (above 1.00x = yours is faster)")
print("Run this script 2-3 times: a real result should repeat.")
