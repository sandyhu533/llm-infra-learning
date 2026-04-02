# ORCA: A Distributed Serving System for Transformer-based Generative Models

**Authors**: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun  
**Venue**: OSDI '22  
**Paper**: [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)

---

## TL;DR

LLM inference has a fundamental mismatch with traditional serving systems: each request generates a *variable number of tokens*, so requests in the same batch finish at unpredictably different times. Pre-Orca systems used static batching — waiting until all requests in a batch finish before starting the next batch — leaving GPU idle whenever some requests finish early. Orca introduces **continuous batching** (called "iteration-level scheduling"): the scheduler makes a new batching decision at *every single decode step*, immediately inserting waiting requests into vacant slots. This keeps GPU utilization near 100% and improves throughput by up to 36.9x over existing systems.

---

## Problem

**What gap does this paper address?**

Transformer-based generative models produce output token-by-token (autoregressive decoding). In a static batch:

```
Request A: [prompt → 5 tokens generated] ✓ DONE at step 5
Request B: [prompt → 50 tokens generated] ... still running at step 5
Request C: [prompt → 30 tokens generated] ... still running at step 5

Steps 6–50: GPU runs B and C, but slot A is WASTED.
New requests sit in queue even though there's capacity.
```

**Why is this hard?**

Each decode step processes *all active requests together* through the full model. Requests have different KV cache sizes (different sequence lengths), so adding or removing a request mid-batch requires careful memory management and batching logic. Prior to Orca, this complexity led systems to take the simpler (but wasteful) static approach.

---

## Key Ideas

### Idea 1: Iteration-Level Scheduling

At every decode step (one forward pass), the scheduler:
1. Checks which requests just finished (generated EOS token)
2. Evicts those requests from the batch
3. Immediately admits new waiting requests to fill the freed slots
4. Runs the next decode step with the updated batch

```
Step t:   [A, B, C, D]
Step t+1: A finishes → evict A, admit E
          [E, B, C, D]
Step t+2: C finishes → evict C, admit F, G
          [E, B, F, G]
```

GPU is never idle waiting for stragglers.

### Idea 2: Selective Batching for Heterogeneous Operations

Prefill and decode have different compute characteristics:
- **Prefill**: processes all prompt tokens in parallel (compute-intensive, embarrassingly parallel)
- **Decode**: generates one token per step per request (memory-bandwidth-bound)

Orca uses **selective batching**: operations that have compatible shapes are batched together; operations that don't (e.g., mixing a long prefill with decode steps) are handled separately. This avoids padding waste when batching requests of very different lengths.

### Idea 3: Distributed Execution with Pipeline Parallelism

For large models that don't fit on one GPU, Orca uses pipeline parallelism: layers are distributed across GPUs, and micro-batches are pipelined to keep all GPUs busy. The iteration-level scheduler operates at the pipeline level, coordinating admission/eviction across all pipeline stages.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| GPU utilization (near 100%) | More complex scheduler (per-step decisions) |
| Throughput | Prefill latency can increase (preempted by decode-heavy batches) |
| Request latency (queue wait time reduced) | Memory pressure (more concurrent requests) |

**Design decisions worth questioning:**

- No explicit KV cache memory management — Orca assumes contiguous memory, which limits how many requests can be in-flight. vLLM's PagedAttention fixes this.
- The "selective batching" heuristic for mixing prefill/decode is approximate. Later work (Sarathi-Serve, Chunked Prefill) shows more principled approaches.
- FCFS admission policy doesn't account for request priority or SLO differentiation.

---

## Connections

**Builds on:**
- Clipper, TensorRT Serving — prior static batching systems that Orca improves upon
- GPipe — pipeline parallelism for distributed execution

**Inspired / Followed by:**
- [vLLM](vllm-pagedattention.md) — adopts continuous batching + adds PagedAttention for memory management
- **Sarathi-Serve** — improves on Orca by chunking prefill to reduce decode latency interference
- **[Splitwise](splitwise-pd-disaggregation.md)** — takes disaggregation further by routing prefill/decode to separate clusters

**Production systems:**
- Continuous batching is now the default in virtually all production LLM serving systems (vLLM, TGI, TensorRT-LLM)

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Throughput improvement | up to 36.9x | vs static batching on GPT-3 scale |
| GPU idle time (static batching) | high | proportional to output length variance |
| GPU idle time (Orca) | near 0% | continuous admission |

---

## Questions & Open Problems

- [ ] How does continuous batching interact with SLO-aware scheduling? If some requests have latency SLOs and others are best-effort, how should the scheduler prioritize?
- [ ] Prefill-decode interference: inserting a long prefill request stalls all decode requests for that step. What's the right policy — always defer prefill, or chunk it?
- [ ] How does the iteration-level scheduler scale when there are thousands of concurrent requests? Is per-step scheduling overhead significant?

---

## Reading Notes

The key insight is deceptively simple: *make scheduling decisions as frequently as possible*. This is the same principle behind preemptive OS scheduling vs. cooperative multitasking. The LLM serving community was effectively using cooperative multitasking (each batch runs to completion) — Orca introduced preemption at the iteration granularity.

The paper doesn't fully solve the memory problem (it assumes contiguous allocation). Reading this alongside vLLM's PagedAttention gives a complete picture: Orca provides the scheduling framework, vLLM provides the memory management. Together they form the foundation of modern LLM serving.
