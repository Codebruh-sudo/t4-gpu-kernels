# GEMM- from naive to tensor cores on the T4

`C = A × B`, square matrices, row-major. Each step fixes the bottleneck the previous step exposed.

## Files

| File | Contents |
|---|---|
| `gemm_tiled.cu` | fp32: naive, shared-memory tiled (16×16 and 32×32), vs cuBLAS. N = 2048 |
| `gemm_regtile.cu` | fp32: register tiling, 8×8 outputs per thread, vs tiled 32 and cuBLAS. N = 2048 |
| `gemm_wmma.cu` | fp16 → fp32: WMMA tensor cores, simple (no reuse) and shared-memory versions, vs cuBLAS. N = 4096 |
| `gemm_wmma_fast.cu` | 128×128 block tiles, 32×64 warp tiles, register-staged double buffering, vs cuBLAS. N = 4096 |

## Results (T4)

| Step | Kernel | TFLOP/s | vs cuBLAS |
|---|---|---|---|
| 0 | fp32 naive | 0.44 | |
| 1 | fp32 shared-memory tiling, 16×16 | 0.71 | |
| 1 | fp32 shared-memory tiling, 32×32 | 0.93 | |
| 2 | fp32 register tiling, 8×8 per thread | ~4.3 | ~75% of cuBLAS SGEMM |
| 3a | WMMA, tiles loaded straight from global memory | 1.67 | |
| 3b | WMMA + shared memory, 2×2 fragments per warp | ~11 | |
| 4 | WMMA, 128×128 block, 32×64 warp tile, double buffering | **17.9** | **~60%** of cuBLAS (~31 TFLOP/s) |

Tensor-core outputs are bit-identical to cuBLAS (checked with C cleared before each run).

## What each step taught

- **32×32 tiles beat 16×16:** half the global loads and half the barriers, with more work between barriers.
- **Tiling alone gave only ~2×:** with one output per thread, every multiply-add needs two shared-memory loads, so the kernel became shared-memory-bound. Register tiling raised math per load from 0.5 to 4.
- **The first tensor-core kernel was slower than fp32 register tiling:** each warp fetched its own 16×16 tiles with no reuse, so the tensor cores starved for data.
- **Arithmetic intensity predicts each step.** For a BM × BN block tile in fp16, FLOPs per byte ≈ BM·BN / (BM + BN), and the ceiling ≈ bandwidth × that:

| Tile | FLOPs/byte | Predicted ceiling (×320 GB/s) | Measured |
|---|---|---|---|
| 16×16 | 8 | ~2.6 TFLOP/s | 1.67 |
| 64×64 | 32 | ~10 | ~11 |
| 128×128 | 64 | ~20 | 17.9 |
| 256×128 (cuBLAS) | ~85 | ~27 | ~31 |

(Measured values can exceed the DRAM-based ceiling slightly because L2 catches repeated reads.)

- **Measurement:** an early run showed the register-tiled kernel faster than cuBLAS; it was GPU clock ramp-up from idle (300 MHz) and did not replicate. Later files warm up longer and time cuBLAS before and after.

## Remaining gap to cuBLAS

Occupancy (one 37 KB block per SM), `ldmatrix` with swizzled shared memory instead of WMMA loads from padded rows, larger warp tiles, and per-shape tuning.

## Run

```bash
nvcc -O3 -arch=sm_75 gemm_wmma_fast.cu -o gemm_fast -lcublas && ./gemm_fast
```
