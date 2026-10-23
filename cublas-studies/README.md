# cuBLAS studies: one FFN GEMM, decode vs prefill, fp16 vs int8

One transformer FFN layer, 4096 × 4096 weights (33.6 MB in fp16), at M = 1 (decode) and M = 512 (prefill or a large batch). Profiled with Nsight Compute.

## Files

| File | Contents |
|---|---|
| `ffn_stage1_gemm.cu` | fp16 `cublasGemmEx`, M = 1 vs M = 512, W stored `[K][N]` |
| `ffn_stage4_int8.cu` | fp16 vs int8 (`CUBLAS_COMPUTE_32I`) with per-row scales and a dequantize kernel, W stored `[N][K]` |

## fp16 (`ffn_stage1_gemm.cu`)

| | M = 1 | M = 512 |
|---|---|---|
| Kernel cuBLAS picks | `gemv2N` + `splitKreduce` | `turing_fp16_s1688gemm_fp16_256x128` (split-K ×5) |
| Time | 0.168 ms | 0.872 ms |
| DRAM throughput (ncu) | 88% | 49% |
| Tensor pipe (ncu) | 0.6% | 77% |
| Bound by | memory | compute |

512× the work in 5.2× the time: about **99× cheaper per token** at M = 512. Split-K is used at M = 1 to fill 40 SMs (32 blocks otherwise) and at M = 512 to avoid wave quantization (64 tiles on 40 SMs).

## fp16 vs int8 (`ffn_stage4_int8.cu`)

| | M = 1 | M = 512 |
|---|---|---|
| fp16 | 0.135 ms | 0.579 ms |
| int8 GEMM only | 0.101 ms (1.34×) | 0.398 ms (1.45×) |
| int8 + dequantize | 0.103 ms (1.31×) | 0.438 ms (1.32×) |
| Relative error | 0.05% → 0.55% | 0.05% → 0.61% |

**Findings**
- Storing W as `[N][K]` alone made the fp16 GEMM 1.5× faster at M = 512 (0.872 → 0.579 ms): cuBLAS chose a better kernel.
- At M = 1, fp16 runs at the bandwidth wall (~248 GB/s) but cuBLAS int8 reaches only ~166 GB/s, so int8 gives 1.34× instead of the ideal 2×. A custom decode kernel could close that gap.
- At M = 512 the separate dequantize kernel costs about 10%: its cost scales with the output size.
- int8 raised the error about 10×, so it would need a model-quality evaluation before use.

## Run

```bash
nvcc -O3 -arch=sm_75 ffn_stage4_int8.cu -o ffn4 -lcublas && ./ffn4
```
