# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Venue**: NeurIPS '22  
**Paper**: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)  
**Code**: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

---

## TL;DR

Standard attention is slow not because of FLOPs, but because of memory bandwidth: it materializes the full N×N attention matrix in HBM (GPU DRAM), which is orders of magnitude slower than on-chip SRAM. FlashAttention restructures the computation to tile across SRAM — the attention matrix is never fully materialized. For the backward pass, intermediate values are recomputed on-the-fly rather than stored. Result: exact (not approximate) attention that is 2–4x faster than standard PyTorch attention on A100, and uses 10–20x less memory — making long-context training practical.

---

## Infra Analogy

> For engineers coming from distributed systems / OS / backend infra:

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| HBM (GPU DRAM) reads | Disk I/O / DRAM access | Slow, high-latency, bandwidth-limited — the bottleneck |
| SRAM (on-chip shared memory) | L1/L2 CPU cache | Fast, small, explicit management needed |
| Tiled attention computation | Cache-oblivious / blocked matrix multiply | Restructure access pattern so working set fits in fast cache; avoid round-trips to slow memory |
| Recomputing softmax in backward | Log-structured / write-ahead patterns | Trade compute for storage: recompute cheaply rather than persist large intermediates |
| IO-awareness | Storage-aware query planning (e.g., index scan vs seq scan) | Optimize for memory access pattern, not just FLOP count |

The insight is the same as blocked matrix multiply in HPC: the algorithm is mathematically equivalent, but accessing data in cache-friendly tiles instead of row-by-row drops memory traffic dramatically.

---

## Problem

**What gap does this paper address?**

Standard self-attention has O(N²) time and space complexity in sequence length N. In practice, GPU memory is the binding constraint — not FLOPs. The standard implementation:

1. Computes the full Q·Kᵀ matrix → writes N×N floats to HBM
2. Applies softmax row-wise → reads and writes N×N floats again
3. Multiplies by V → reads N×N floats a third time

For N=2048 on a 40GB A100, attention intermediates alone consume ~1.3GB per layer. For N=8192 it's ~21GB — impossible to run.

More critically, each HBM access is ~10–20x slower bandwidth than on-chip SRAM. The algorithm is memory-bandwidth bound, not compute bound.

**Why is this hard?**

Softmax is a reduction over the full row — you need to know the maximum and sum of all logits before you can output any normalized weight. This inter-element dependency seems to force materializing the full row. Breaking this dependency requires tracking running statistics (online softmax), which is non-trivial to implement correctly in a GPU kernel and even harder to extend to the backward pass.

---

## Key Ideas

### Idea 1: Tiling — Fit Working Set in SRAM

Divide the Q, K, V matrices into blocks that fit in SRAM. Process one tile of Q against all tiles of K and V in an outer loop:

```
SRAM size: ~20MB (A100)
Block size chosen so Q_block + K_block + V_block + O_block fits in SRAM

for each Q_block (rows of output):
    for each K_block, V_block:
        compute local attention scores: S_ij = Q_i · K_j^T
        update running (max, sum) for online softmax
        accumulate output: O_i += softmax(S_ij) · V_j
    write O_i to HBM  ← only 1 HBM write per output row
```

HBM accesses: O(N × d) instead of O(N²). For N=2048, d=64, this is a ~32x reduction in HBM traffic.

### Idea 2: Online Softmax — Computing Without Full Materialization

Standard softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

You can compute this in one pass with running statistics. For each new tile, update:
- `m_new = max(m_old, max(S_new_tile))`
- `l_new = exp(m_old - m_new) * l_old + sum(exp(S_new_tile - m_new))`
- `O_new = diag(exp(m_old - m_new)) * O_old + exp(S_new_tile - m_new) * V_new_tile`

This is the key mathematical trick that makes tiling possible. Same idea as computing streaming statistics (Welford's algorithm for variance).

### Idea 3: Recomputation in Backward Pass

Standard attention saves the N×N attention matrix for the backward pass (needed to compute gradients). FlashAttention discards it and recomputes it during backward using the saved softmax statistics (m, l per row — O(N) storage vs O(N²)).

This trades compute for memory: the backward pass does ~2x the FLOPs of standard attention, but avoids HBM traffic for the large intermediate. At memory-bandwidth-bound regimes (which attention always is), this is a net win.

```
Forward: save (O, l, m) per row — O(N·d + N) storage
Backward: recompute S, P from (Q, K, V, l, m) on-the-fly
         → no N×N matrix stored, ever
```

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| HBM bandwidth (main bottleneck) | Higher FLOP count in backward (~2x) |
| Memory footprint (O(N) vs O(N²)) | Complex custom CUDA kernel (hard to maintain) |
| Exact correctness (not approximate) | Block size tuning needed per hardware generation |
| Long-context training feasibility | Less flexible for attention variants (requires kernel rewrite) |

**Design decisions worth questioning:**

- The block size is hardware-specific (SRAM size varies: A100 ~20MB, H100 ~50MB). The paper provides a formula but auto-tuning is still needed.
- The custom CUDA kernel is notoriously hard to extend. FlashAttention-2 and -3 rewrote the kernel substantially for better occupancy on H100 (tensor core utilization, warp specialization).
- For very short sequences (N < 512), the tiling overhead can negate the bandwidth savings — standard attention may be faster.

---

## Connections

**Builds on:**
- Online softmax algorithm (Milakov & Gimelshein, 2018) — the streaming normalization trick
- Blocked / tiled matrix multiply in BLAS (GEMM) — the tiling pattern is identical in spirit
- IO-complexity analysis (Aggarwal & Vitter, 1988) — formal framework for counting memory transfers

**Inspired / Followed by:**
- **FlashAttention-2** (Dao, 2023) — 2x speedup via better parallelism (query-dimension parallelism, fewer non-matmul FLOPs)
- **FlashAttention-3** (Shah et al., 2024) — H100-specific: warp specialization, pingpong GEMM/softmax, FP8 support
- **FlashDecoding** — extends tiling to autoregressive decode (different parallelism strategy: parallelism over KV sequence length)
- **Ring Attention** — extends tiling across multiple GPUs for sequences too long for one device
- **PagedAttention** (vLLM) — uses Flash-style kernels but adds non-contiguous block table lookup

**Production systems:**
- Used in virtually every production LLM training stack (PyTorch SDPA, Megatron-LM, DeepSpeed)
- Default attention implementation in HuggingFace Transformers (via `torch.nn.functional.scaled_dot_product_attention`)

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Speedup vs standard attention | 2–4x | A100, forward + backward, BERT/GPT-style |
| Memory reduction | 10–20x | Sequence length 1K–8K |
| HBM reads/writes reduction | ~5–10x | O(N·d) vs O(N²) |
| Backward FLOP overhead | ~2x | Due to recomputation |
| Max sequence length increase | ~8x | Same GPU memory budget |

---

## Questions & Open Problems

- [ ] How does optimal block size change as SRAM grows (H100 has ~2.5x more SRAM than A100)? Is there a formula that generalizes?
- [ ] FlashDecoding parallelizes over KV sequence length for decode — how does this interact with PagedAttention's non-contiguous blocks?
- [ ] For very long sequences (>100K tokens), is online softmax numerically stable enough? What's the precision floor?
- [ ] Ring Attention extends this to multi-GPU — what's the communication overhead vs the memory savings tradeoff?

---

## Reading Notes

The framing shift is: **attention is an I/O problem, not a compute problem**. Most ML papers optimize FLOPs. This paper counts HBM reads/writes — a memory systems lens, not an ML lens.

The recomputation trick (discard intermediates, recompute in backward) is exactly what gradient checkpointing does at the layer level. FlashAttention applies the same principle within a single attention operation. The "waste" compute in backward is invisible compared to the bandwidth savings.

If you've ever optimized a hot loop by restructuring memory access (loop tiling, cache-oblivious algorithms, prefetching), this paper will feel immediately familiar. The novelty is applying it to attention, which has the online-softmax dependency that makes tiling non-obvious.
