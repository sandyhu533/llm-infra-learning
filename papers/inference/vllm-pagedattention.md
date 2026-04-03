# Efficient Memory Management for Large Language Model Serving with PagedAttention

**Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica  
**Venue**: SOSP '23  
**Paper**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)  
**Code**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

## TL;DR

LLM inference is bottlenecked by GPU memory, not compute. The KV cache (key/value tensors stored per token per layer) is the main culprit — existing systems pre-allocate contiguous memory per request, leading to 60–80% memory waste from fragmentation and over-reservation. vLLM introduces **PagedAttention**, which manages KV cache like OS virtual memory: non-contiguous physical blocks mapped via a page table. This enables near-zero waste, allows memory sharing across requests (e.g., beam search, prefix sharing), and increases effective throughput by 2–4x over HuggingFace Transformers and 1.3–2.5x over FasterTransformer.

---

## Problem

**What gap does this paper address?**

Before vLLM, LLM serving systems pre-allocated a contiguous chunk of GPU memory per request at arrival time, sized for the *maximum possible* sequence length. This caused three types of waste:

1. **Internal fragmentation**: reserved memory never actually used (prompt shorter than max)
2. **External fragmentation**: free memory exists but is split into pieces too small to use
3. **No sharing**: beam search generates multiple sequences from the same prefix, but each sequence stores a full independent copy of the shared KV cache

Result: GPU memory utilization was 20–40%, bottlenecking throughput.

**Why is this hard?**

KV cache is dynamic — you don't know how long a sequence will be until it finishes generating. Standard memory allocators (malloc-style) can't handle this gracefully at GPU memory speeds without significant overhead.

---

## Key Ideas

### Idea 1: PagedAttention — Virtual Memory for KV Cache

Divide KV cache into fixed-size **blocks** (analogous to OS pages). Each block holds KV vectors for a fixed number of tokens (e.g., 16 tokens × all layers × 2 for K and V).

```
Logical KV sequence (request view):
[ tok0 | tok1 | tok2 | tok3 | tok4 | tok5 | ... ]

Physical block allocation (actual GPU memory):
Block 7:  [ tok0 | tok1 | tok2 | tok3 ]
Block 3:  [ tok4 | tok5 | tok6 | tok7 ]
Block 12: [ tok8 | ... ]

Block table (per request):
logical block 0 → physical block 7
logical block 1 → physical block 3
logical block 2 → physical block 12
```

The attention kernel is modified to look up the block table during computation — reads K/V from non-contiguous physical addresses.

### Idea 2: Memory Sharing via Copy-on-Write

For beam search (k beams from one prompt) or parallel sampling, the prompt's KV blocks are **shared** across all sequences with a reference count. A new block is only allocated (copy-on-write) when a sequence diverges and needs to write new tokens.

This reduces memory for beam search from O(k × prompt_length) to O(prompt_length + k × generated_length).

### Idea 3: Preemption with Swap and Recompute

When memory runs out (a new high-priority request arrives), vLLM can:
- **Swap**: move a blocked request's KV cache to CPU memory, swap back when GPU memory is available
- **Recompute**: simply drop the KV cache and recompute the prefill when the request resumes

The scheduler uses a First-Come-First-Served policy with preemption to maximize utilization.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| GPU memory utilization | Modified attention kernel (non-contiguous memory access) |
| Throughput (more requests in flight) | Added block table lookup overhead (~1–2%) |
| Flexibility (variable length) | Slightly more complex memory management logic |
| Sharing (beam/prefix cache) | Copy-on-write coordination complexity |

**Design decisions worth questioning:**

- Block size is a fixed hyperparameter. Small blocks = less internal fragmentation but more table entries; large blocks = faster access but more waste. vLLM defaults to 16 tokens — not always optimal for very long contexts.
- Swap-to-CPU assumes CPU memory is large enough. In practice, swapping latency can hurt P99 significantly under bursty load.
- FCFS scheduling is simple but not optimal; later work (e.g., Sarathi-Serve, S3) shows smarter scheduling improves latency distribution.

---

## Connections

**Builds on:**
- OS virtual memory (paging) — the core abstraction is directly borrowed
- [Orca](orca-continuous-batching.md) — vLLM uses continuous batching from Orca as its batching strategy

**Inspired / Followed by:**
- **Prefix caching / RadixAttention** (SGLang) — extends block sharing to arbitrary prefix trees
- **Chunked prefill** — splits long prefill into chunks to reduce interference with decode latency
- **Splitwise / Disaggregated serving** — separates prefill and decode to dedicated pools

**Production systems:**
- vLLM is used in production at Anyscale, various cloud providers, and as the backbone for many LLM serving deployments

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Memory waste (baseline) | 60–80% | Pre-vLLM systems on OPT-13B |
| Memory waste (vLLM) | <4% | Block-level allocation |
| Throughput improvement | 2–4x | vs HuggingFace Transformers |
| Throughput improvement | 1.3–2.5x | vs FasterTransformer |
| Block table overhead | ~1–2% | Compared to contiguous attention |

---

## Questions & Open Problems

- [ ] How does block size interact with different hardware (H100 vs A100)? What's the optimal block size for 128K context models?
- [ ] Swap latency under bursty load — how does vLLM's P99 degrade when swap is triggered frequently?
- [ ] Is there a better preemption policy than FCFS that accounts for request priority and remaining generation length?
- [ ] How does PagedAttention interact with speculative decoding (draft model generates multiple tokens; which blocks are tentative)?

---

## Reading Notes

The core insight — "KV cache is like process memory, apply OS paging" — feels obvious in hindsight but wasn't for the ML systems community, which had been thinking about this as an ML problem rather than a systems problem.

The block table lookup adds memory indirection. The natural follow-on optimization is prefix caching (SGLang's RadixAttention): co-scheduling requests with shared prefixes to the same worker eliminates redundant lookups.
