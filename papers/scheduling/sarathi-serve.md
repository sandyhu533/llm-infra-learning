# Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes on Chunked Prefills

**Authors**: Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee  
**Venue**: OSDI '24  
**Paper**: [https://arxiv.org/abs/2308.16369](https://arxiv.org/abs/2308.16369)  
**Code**: [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve)

---

## TL;DR

In continuous batching systems like vLLM, long prefill requests monopolize the GPU for hundreds of milliseconds, blocking decode tokens for other in-flight requests and spiking their inter-token latency (TPOT). Sarathi-Serve fixes this with two ideas: **chunked prefill** (split a large prefill into small fixed-size chunks spread across multiple iterations) and **decode piggybacking** (run decode tokens alongside every prefill chunk, so decodes proceed every iteration). The result: TPOT P99 drops by up to 9x with negligible throughput loss.

---

## Problem

**What gap does this paper address?**

In vLLM-style continuous batching, the scheduler picks a batch each iteration and executes a full forward pass. If a new long-context request arrives (e.g., 8K-token prompt), its *prefill* dominates the entire iteration — potentially 300–500ms on a single GPU — during which all other in-flight requests' decode steps are blocked. This inflates TPOT for every other user sharing the system.

The fundamental tension:
- **TTFT** (time-to-first-token): want to finish prefill fast → prefer large batches, few interruptions
- **TPOT** (time-per-output-token): want short decode latency → want decode to run every iteration

Existing systems trade one for the other. Sarathi-Serve breaks the tradeoff.

**Why is this hard?**

Attention has a quadratic cost in sequence length for prefill but a linear (memory-bandwidth-bound) cost per token for decode. Mixing them naively in one pass is complex: the prefill chunk attends only to its own tokens, while decodes attend to their full cached KV history. The batch computation must handle both attention patterns in a single kernel call.

---

## Key Ideas

### Idea 1: Chunked Prefill

Instead of processing the full prompt in one shot, split it into chunks of at most `C` tokens (e.g., C=512) and schedule one chunk per iteration until the prefill is complete.

```
Without chunked prefill (C = full prompt length):

Iter 1: [PREFILL: tok0..tok4095]         ← blocks decode for ~300ms
Iter 2: [DECODE: req_A tok5, req_B tok3] ← now decode can proceed

With chunked prefill (C = 512):

Iter 1: [PREFILL chunk 0: tok0..tok511]  [DECODE: req_A, req_B]
Iter 2: [PREFILL chunk 1: tok512..1023]  [DECODE: req_A, req_B]
...
Iter 8: [PREFILL chunk 7: tok3584..4095] [DECODE: req_A, req_B]
```

Every iteration, decode tokens from all in-flight requests are processed alongside the current prefill chunk.

### Idea 2: Decode Piggybacking

Each iteration, after the scheduler selects the next prefill chunk, it **appends all pending decode tokens** to the batch. The forward pass handles a mixed batch: prefill tokens for one request + single decode tokens for all others.

The attention kernel must distinguish between:
- **Prefill tokens**: attend to each other (within the chunk) + their prior KV cache
- **Decode tokens**: attend to their full KV cache (single token, no self-attention)

This is handled via attention masking and careful batching in the CUDA kernels.

### Idea 3: Stall-Free Scheduling Invariant

Sarathi-Serve enforces the invariant: **every in-flight decode request runs in every iteration**. A request is never "stalled" waiting for a prefill to complete. This bounds TPOT jitter to at most one chunk's worth of compute overhead, independent of how long the concurrent prefill is.

```
Scheduling policy:
1. If any request has a pending prefill chunk → schedule next chunk (size C)
2. Always append decode tokens for ALL in-flight requests to the batch
3. Bound: batch compute per iteration ≤ C prefill tokens + N decode tokens
```

The chunk size C becomes the primary knob to trade TTFT vs TPOT.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| TPOT P99 / decode latency variance | Slightly higher TTFT for long prompts (prefill takes more iterations) |
| SLO compliance under mixed workloads | More complex scheduler and attention kernel |
| GPU utilization across all requests | Small overhead from mixed-batch attention computation |
| Predictable latency | Marginal throughput loss at very high loads (~2–5%) |

**Design decisions worth questioning:**

- Chunk size C is a static hyperparameter. Dynamic chunk sizing (adaptive to current GPU load and SLO targets) could be better but adds complexity.
- Piggybacking works well when decode batch is large. With very few in-flight decode requests, piggybacking gains little — the prefill dominates either way.
- The paper targets A100/H100 GPUs. On memory-bandwidth-constrained hardware, smaller C might be needed to avoid compute bottlenecks during chunked prefill.

---

## Connections

**Builds on:**
- [Orca](../inference/orca-continuous-batching.md) — iteration-level scheduling; Sarathi-Serve adds chunk-level scheduling within each iteration
- [vLLM/PagedAttention](../inference/vllm-pagedattention.md) — KV cache management; Sarathi-Serve uses paged KV cache as the memory backend

**Inspired / Followed by:**
- **SGLang RadixAttention** — extends prefix-aware scheduling; chunk size interacts with prefix cache hit rates
- **Mooncake** (ByteDance/Moonshot) — disaggregated prefill/decode; chunking enables better prefill routing
- **ASTRA-sim / Vidur** — simulators can now model chunked prefill behavior

**Production systems:**
- vLLM v0.4+ implements chunked prefill as an optional feature
- TensorRT-LLM and DeepSpeed-FastGen have similar chunking mechanisms

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| TPOT P99 reduction | up to 9.2x | LLaMA-34B, mixed short/long request workload |
| TTFT increase | ~1.2x | Due to spreading prefill across more iterations |
| Throughput loss | <5% | At 95th percentile load |
| Chunk size (default) | 512 tokens | Balances TTFT vs TPOT on A100 |

---

## Questions & Open Problems

- [ ] How should chunk size be set dynamically based on current SLO targets and queue depth?
- [ ] Does chunked prefill interact with speculative decoding? (draft model generates tokens that may be invalidated by the next prefill chunk's context)
- [ ] At very long context (100K+ tokens), how many chunks does prefill split into, and does the total TTFT become unacceptable?
- [ ] How does this interact with prefix caching? A chunk boundary may fall in the middle of a cached prefix, requiring partial cache hits.

---

## Reading Notes

The core problem — one large job monopolizing the queue and blocking all smaller ones — is well-known in scheduling literature, and chunking is the standard fix. What makes this non-trivial in LLM serving is the attention kernel: you need one kernel call that handles prefill tokens (QKV self-attention within the chunk) and decode tokens (single-token attention over full KV cache) simultaneously. This is essentially what FlashAttention's variable-length kernel does — Sarathi-Serve extends it to the mixed-batch case.

The chunk size C is the primary production tuning knob: larger C means fewer iterations to complete a prefill (lower TTFT) but worse decode interference; smaller C means smoother decode latency (lower TPOT P99) but more iterations per prefill. Tune it based on observed TTFT/TPOT SLO targets for your workload.
