T4 GPU Kernels
Hand-written CUDA kernels for AI workloads on the NVIDIA Tesla T4 (Turing, sm_75), a free and widely available GPU (Google Colab, Kaggle) that newer libraries often leave behind. I have benchmarked every kernel against the library it competes with (cuBLAS or PyTorch), checked it for correctness, and explained it step by step.
Highlights
FlashAttention forward for Turing, written in CUDA because FlashAttention-2 does not support sm_75. Built in three stages, from 2.2 to 15.4 TFLOP/s. In a same-run, alternating benchmark it is 5–12% faster than PyTorch's memory-efficient attention kernel for sequences of 1024 tokens and longer, with errors around 1e-5. - `flash-attention/`
Batch-1 decode GEMV, one warp per row: 265 GB/s (~83% of peak bandwidth), 1.07× faster than cuBLAS at batch 1, because cuBLAS uses split-K plus a separate reduction kernel there. A batch sweep shows where it stops winning. → `gemv-decode/`
GEMM from naive to tensor cores: 0.44 to 17.9 TFLOP/s (about 40×) through shared-memory tiling, register tiling, WMMA tensor cores, larger warp tiles and double buffering. Output is bit-identical to cuBLAS. - `gemm/`
FlashAttention results
T4, same run, alternating rounds, median of 7. B = 1, H = 8, head dim 64, fp16, non-causal forward pass.
Sequence length	This kernel	PyTorch memory-efficient SDPA	Ratio (PyTorch / ours)	Max error vs fp32
512	0.057 ms (9.5 TFLOP/s)	0.052 ms (10.3)	0.92×	4.2e-05
1024	0.192 ms (11.2)	0.201 ms (10.7)	1.05×	3.4e-05
2048	0.651 ms (13.2)	0.692 ms (12.4)	1.06×	2.2e-05
4096	2.361 ms (14.6)	2.535 ms (13.6)	1.07×	1.4e-05
8192	8.927 ms (15.4)	9.981 ms (13.8)	1.12×	1.5e-05 [^1]
[^1]: At N = 8192 the error is measured against PyTorch, not against the double-precision CPU reference used for the other rows.
Memory
Like all FlashAttention variants, it never builds the N×N score matrix. At N = 8192 (8 heads) naive PyTorch attention peaked at 5.4 GB, while the fused kernels need only Q, K, V and O.
All projects
Folder	What it is	Key result
`flash-attention/`	FlashAttention forward for the T4, stages 1–3 plus a fair PyTorch comparison	1.05–1.12× vs PyTorch for N ≥ 1024
`gemv-decode/`	Batch-1 decode GEMV plus a batch-size sweep (M = 1 → 512)	1.07× vs cuBLAS at batch 1
`gemm/`	GEMM optimization ladder, fp32 CUDA cores → fp16 tensor cores	17.9 TFLOP/s, ~60% of cuBLAS
`cublas-studies/`	Profiling cuBLAS FFN GEMMs: decode vs prefill, fp16 vs int8	memory- vs compute-bound with Nsight Compute
`experiments/`	Softmax and fusion kernels	not yet benchmarked
Methodology
Timing: CUDA events, after warm-up, averaged over many iterations.
Fair baselines: the library is timed in the same run, before and after (or alternating with) the custom kernel. Early runs showed GPU clock ramp-up from idle skewing results, so the GPU is warmed up first.
Replication: results are only reported if they repeat. One early result, a register-tiled SGEMM appearing faster than cuBLAS, was a clock-ramp artifact and was discarded.
Correctness- every kernel is checked against cuBLAS, PyTorch or an exact double-precision CPU calculation. Output buffers are cleared before each run, so a kernel that writes nothing cannot pass.
Profiling- Nsight Compute (DRAM throughput, tensor-pipe utilization, warp stalls) to explain why a kernel is fast or slow.
Hardware and software
NVIDIA Tesla T4 (Google Colab): 40 SMs, 320 GB/s, 16 GB, compute capability 7.5
CUDA (`nvcc -O3 -arch=sm_75`), PyTorch 2.11.0+cu130, Triton 3.6.0
How to run
Each CUDA file is a standalone benchmark:
```bash
nvcc -O3 -arch=sm_75 flash-attention/flash_v3.cu -o flash3 && ./flash3
nvcc -O3 -arch=sm_75 gemv-decode/gemv_kernel1.cu -o gemv1 -lcublas && ./gemv1
```
Python comparison (compiles the CUDA kernel inside PyTorch on first run):
```bash
python flash-attention/flash_compare.py
```
In Colab, use a T4 runtime, save each file with `%%writefile`, and prefix commands with `!`.
Limitations
FlashAttention: forward pass only, non-causal, head dimension 64, sequence length a multiple of 64.
Results are specific to the T4 and to the shapes tested. The GEMV kernel only wins at batch 1.
The GEMM kernels reach about 60% of cuBLAS; the remaining gap (`ldmatrix`, swizzled shared memory, per-shape tuning) is not closed.
About
Built by Abhishek Kumar, working on AI kernel and inference performance.
