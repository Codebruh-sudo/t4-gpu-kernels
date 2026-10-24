# FlashAttention forward for the T4 (Turing, sm_75)

FlashAttention-2 does not support Turing GPUs. This folder builds a FlashAttention forward kernel for the T4 in CUDA, in three stages, and compares it fairly against PyTorch.

`O = softmax(Q·Kᵀ / √d) · V`, with tiles of Q, K and V in on-chip memory and **online softmax**, so the N×N score matrix is never written to memory.

## Files

| File | What it does |
|---|---|
| `attention_starter.py` | Baseline- naive attention vs PyTorch's memory-efficient SDPA (time and memory), plus a Triton `tl.dot` test on the T4 |
| `flash_v1.cu` | Stage 1: one thread per query row, online softmax, CUDA cores (correctness baseline) |
| `flash_v2.cu` | Stage 2- WMMA tensor cores for Q·Kᵀ and P·V, rescaling once per 64-key tile (includes stage 1 for comparison) |
| `flash_v3.cu` | Stage 3- 3× occupancy, output kept in registers, K/V prefetch (includes stage 2 for comparison) |
| `flash_compare.py` | Fair benchmark: stage 3 compiled inside PyTorch, timed in alternating rounds against PyTorch's kernel |

## Stage by stage (N = 8192, B = 1, H = 8, D = 64, fp16)

| Stage | Key idea | Time | TFLOP/s |
|---|---|---|---|
| 1 | online softmax on CUDA cores, one thread per query row | 61.6 ms | 2.2 |
| 2 | WMMA tensor cores, rescale once per tile instead of per key | ~15 ms | 9.1 |
| 3 | 3 blocks per SM (was 1), O in registers, K/V prefetch | 8.8 ms | 15.7 |

All stage- max error ~1e-5 against an exact CPU calculation.

### What changed in stage 3

- **Occupancy:** 32-key tiles and output accumulators in registers cut shared memory from ~52 KB to ~18.7 KB per block, so 3 blocks (12 warps) fit per SM instead of 1 block (4 warps).
- **Output in registers:** WMMA does not publish which row each accumulator element belongs to, and per-row rescaling needs that. The kernel discovers it once at start-up by loading a 16×16 matrix whose value at (r, c) is r into an accumulator fragment and reading it back (a "row map").
- **Prefetch:** the next K/V tile is loaded from global memory into registers while the current tile is computed, then stored to shared memory (register-staged, since the T4 has no `cp.async`).

## Fair comparison with PyTorch (`flash_compare.py`)

Same GPU, same script, warmed-up clocks, alternating rounds, median of 7.

| N | This kernel | PyTorch memory-efficient | Ratio |
|---|---|---|---|
| 512  | 0.057 ms | 0.052 ms | 0.92× |
| 1024 | 0.192 ms | 0.201 ms | **1.05×** |
| 2048 | 0.651 ms | 0.692 ms | **1.06×** |
| 4096 | 2.361 ms | 2.535 ms | **1.07×** |
| 8192 | 8.927 ms | 9.981 ms | **1.12×** |

## Memory (from `attention_starter.py`)

| N | Naive attention (peak) | Memory-efficient attention |
|---|---|---|
| 512  | 21 MB   | 0.5 MB |
| 2048 | 336 MB  | 2.1 MB |
| 8192 | 5369 MB | 8.4 MB |

Naive memory grows as N² (4× per doubling); the fused kernel grows as N.

## Known limits and next steps

- Short sequences (N ≤ 512) too few blocks to fill 40 SMs. Splitting the keys across blocks (split-KV) would help.
- Not yet supported: causal masking, head dimensions other than 64, backward pass.

## Run

```bash
nvcc -O3 -arch=sm_75 flash_v3.cu -o flash3 && ./flash3
python flash_compare.py        # needs PyTorch; first run compiles (~1–2 min)
```
