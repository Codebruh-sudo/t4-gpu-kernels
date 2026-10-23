# =====================================================================
# PyTorch starter for the FlashAttention work (run in Colab on the T4)
#
# Part 1: the PyTorch basics you need (GPU tensors, @, softmax, timing,
#         checking results)
# Part 2: naive attention vs PyTorch's memory-efficient attention
#         (the FlashAttention idea: tiles + online softmax, no N x N matrix)
#         -> time AND memory as the sequence length N grows
# Part 3: does Triton's tl.dot (tensor cores) work on this GPU?
#         (Triton officially needs compute capability 8.0+, the T4 is 7.5)
#
# run in a Colab cell:   !python attention_starter.py
# =====================================================================

import math
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "Switch the Colab runtime to a GPU"
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0),
      "| compute capability", torch.cuda.get_device_capability(0))


# ------------------------------- Part 1 --------------------------------
def time_ms(fn, iters=20):
    """Average GPU time of fn() in ms, using CUDA events (after warm-up)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def peak_mem_mb(fn):
    """Extra GPU memory (MB) used while running fn(), above what already exists."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def naive_attention(q, k, v):
    """softmax(Q K^T / sqrt(d)) V, building the full N x N score matrix."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-2, -1)) * scale            # [B, H, N, N]  <- the big matrix
    p = torch.softmax(s.float(), dim=-1).to(q.dtype) # softmax in fp32 for accuracy
    return p @ v


# ------------------------------- Part 2 --------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def efficient_attention(q, k, v):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):   # force the FlashAttention-style kernel
            return F.scaled_dot_product_attention(q, k, v)
    backend_note = "forced EFFICIENT_ATTENTION backend"
except Exception:
    def efficient_attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)
    backend_note = "default SDPA backend (older PyTorch)"

B, H, D = 1, 8, 64            # batch, heads, head dimension
print(f"\nPart 2: naive vs memory-efficient attention  (B={B}, H={H}, D={D}, fp16, {backend_note})")
print(f"{'N':>6} | {'naive ms':>9} {'naive MB':>9} | {'efficient ms':>12} {'efficient MB':>12} | max diff")
print("-" * 72)

for N in [512, 1024, 2048, 4096, 8192]:
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        out_naive = naive_attention(q, k, v)
        t_naive = time_ms(lambda: naive_attention(q, k, v))
        m_naive = peak_mem_mb(lambda: naive_attention(q, k, v))
        naive_txt = f"{t_naive:9.3f} {m_naive:9.1f}"
    except torch.cuda.OutOfMemoryError:
        out_naive, naive_txt = None, f"{'OOM':>9} {'OOM':>9}"
        torch.cuda.empty_cache()

    out_eff = efficient_attention(q, k, v)
    t_eff = time_ms(lambda: efficient_attention(q, k, v))
    m_eff = peak_mem_mb(lambda: efficient_attention(q, k, v))

    diff = (out_naive.float() - out_eff.float()).abs().max().item() if out_naive is not None else float("nan")
    print(f"{N:>6} | {naive_txt} | {t_eff:12.3f} {m_eff:12.1f} | {diff:.1e}")

print("\nExpect: naive memory grows ~N^2 (the score matrix); efficient grows ~N.")


# ------------------------------- Part 3 --------------------------------
print("\nPart 3: Triton tensor-core test (tl.dot) on this GPU")
try:
    import triton
    import triton.language as tl

    @triton.jit
    def dot_test_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + offs[:, None] * BLOCK + offs[None, :])
        b = tl.load(b_ptr + offs[:, None] * BLOCK + offs[None, :])
        c = tl.dot(a, b)                                   # tensor-core matmul in Triton
        tl.store(c_ptr + offs[:, None] * BLOCK + offs[None, :], c)

    BLOCK = 32
    a = torch.randn(BLOCK, BLOCK, device="cuda", dtype=torch.float16)
    b = torch.randn(BLOCK, BLOCK, device="cuda", dtype=torch.float16)
    c = torch.empty(BLOCK, BLOCK, device="cuda", dtype=torch.float32)
    dot_test_kernel[(1,)](a, b, c, BLOCK=BLOCK)
    torch.cuda.synchronize()
    err = (c - (a.float() @ b.float())).abs().max().item()
    print(f"Triton {triton.__version__}: tl.dot ran, max error {err:.1e}",
          "-> Triton route looks usable on this GPU" if err < 1e-1 else "-> WRONG RESULT, don't trust Triton here")
except Exception as e:
    print("Triton tl.dot FAILED on this GPU:", type(e).__name__, str(e)[:300])
    print("-> use the CUDA (WMMA) route for FlashAttention on the T4")
