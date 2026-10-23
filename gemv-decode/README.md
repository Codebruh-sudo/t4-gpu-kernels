# Batch-1 decode GEMV on the T4

`y = W·x` with W stored `[N][K]` (PyTorch layout), N = K = 4096, fp16. At batch 1 (one token per step, as in LLM decode) this is memory-bound: about 1 FLOP per byte.

## Files

| File | Contents |
|---|---|
| `gemv_kernel1.cu` | One warp per output row, 16-byte vectorized loads, fp32 accumulation, warp-shuffle reduction, vs cuBLAS |
| `gemv_batch_sweep.cu` | Batch sizes M = 1 → 512: the kernel above (naive per token), a weight-reuse variant (8 tokens per warp), and cuBLAS |

## Batch 1 result

| | Time | Bandwidth |
|---|---|---|
| cuBLAS | 0.1355 ms | 247.7 GB/s |
| **This kernel** | **0.1269 ms** | **264.6 GB/s (~83% of 320 GB/s)** |

**1.07× faster than cuBLAS**, same accuracy (relative error 2.9e-4 for both). Reproduced in a second run (0.1262 vs 0.1350 ms).

**Why:** profiling cuBLAS at batch 1 with Nsight Compute shows a split-K GEMV plus a separate `splitKreduce` kernel, used to create enough blocks for 40 SMs. With one warp per row, 4096 warps already fill the GPU, so the kernel computes each row completely and writes it directly: one kernel, no partial sums.

## Batch sweep (time in ms)

| M | cuBLAS | One warp per row, per token | Weight reuse ×8 |
|---|---|---|---|
| 1 | 0.135 | **0.126** | 0.141 |
| 2 | 0.159 | 0.248 | 0.159 |
| 4 | 0.155 | 0.491 | 0.217 |
| 8 | 0.154 | 0.978 | 0.326 |
| 16 | 0.156 | 1.951 | 0.607 |
| 64 | 0.173 | 7.795 | 2.318 |
| 256 | 0.395 | 31.16 | 9.121 |
| 512 | 0.522 | 62.31 | 17.98 |

**Findings**
- The custom kernel wins **only at batch 1**.
- Re-reading the weights per token makes its time grow linearly with M.
- cuBLAS is nearly flat up to M = 64 (64 tokens cost 1.28× one token): weight reuse plus tensor cores. This is why batching makes LLM serving cheaper.
- The weight-reuse variant did not stay flat: the bottleneck moved from DRAM to reloading the activations from cache. It is also slower at M = 1 because its outer loop is not unrolled (fewer loads in flight).

## Run

```bash
nvcc -O3 -arch=sm_75 gemv_kernel1.cu -o gemv1 -lcublas && ./gemv1
nvcc -O3 -arch=sm_75 gemv_batch_sweep.cu -o sweep -lcublas && ./sweep
```
