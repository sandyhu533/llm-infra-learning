# LLM Infra Learning

A curated collection of paper notes and insights on **LLM Infrastructure** — covering inference systems, distributed training, and foundational architectures.

> Built for personal learning and shared for the community. Notes focus on *engineering decisions and system tradeoffs*, not just algorithm descriptions.

---

## How to Use This Repo

Each paper note follows a consistent structure (see [Template](papers/TEMPLATE.md)):

1. **TL;DR** — one paragraph summary of the key contribution
2. **Problem** — what gap does this paper address
3. **Key Ideas** — core technical decisions, with diagrams where helpful
4. **System Tradeoffs** — what the design optimizes for, and at what cost
5. **Connections** — how this relates to other papers or production systems
6. **Questions** — open questions and things to dig deeper on

---

## Reading Roadmap

If you're coming from backend/distributed systems, follow this path. Each paper builds on the previous — jumping ahead without the prerequisites means missing the "why."

### Quick Reference — Flat Sequence

| # | Paper | Why this order |
|---|-------|---------------|
| 1 | Attention Is All You Need | Defines KV cache, MHA, and the decode loop — the vocabulary for everything else |
| 2 | GQA | Modern production models all use GQA; you need this to reason about KV cache sizing |
| 3 | FlashAttention | Teaches GPU memory hierarchy (HBM vs SRAM); prerequisite for understanding any serving bottleneck |
| 4 | ORCA | First principles of serving: iteration-level scheduling, the reactor pattern for GPUs |
| 5 | vLLM / PagedAttention | Fixes KV cache fragmentation with virtual memory; builds directly on ORCA's scheduling model |
| 6 | Speculative Decoding | A latency-only optimization; orthogonal to memory, easier to understand after vLLM |
| 7 | Splitwise | Cluster-level CQRS: only makes sense once you understand prefill vs decode resource profiles |
| 8 | ZeRO | Training track starts here: memory is the binding constraint; partition it first |
| 9 | Megatron-LM | Model parallelism strategies (TP/PP/DP); builds on ZeRO's memory intuition |
| 10 | MegaScale | What ZeRO + Megatron look like at 10K GPUs in production; engineering > algorithms |
| 11 | Sarathi-Serve | Scheduling refinement: chunked prefill prevents head-of-line blocking; needs ORCA + vLLM first |
| 12 | Vidur | Simulation framework; only useful if you understand what it's simulating (papers 4–7) |
| 13 | GPTQ | Post-training quantization; independent thread, but easier after you understand serving memory pressure |
| 14 | AWQ | Refines GPTQ by protecting salient weights; read GPTQ first |

---

### Detailed Roadmap

**Phase 1 — Foundations** (papers 1–3)

Goal: understand the KV cache, attention variants, and GPU memory hierarchy. Without these, serving paper motivations are opaque.

- **1. [Attention Is All You Need](papers/foundations/attention-is-all-you-need.md)**
  Defines the Transformer, the KV cache, and the autoregressive decode loop. Everything downstream is an optimization of this baseline.

- **2. [GQA](papers/foundations/gqa.md)**
  LLaMA 3, Mistral, Gemma all use GQA instead of MHA. Read before vLLM so you understand what's being cached and at what size.

- **3. [FlashAttention](papers/foundations/flash-attention.md)**
  HBM is slow; SRAM is fast; tile the attention computation to stay in SRAM. After this, "memory-bandwidth bound" has a concrete meaning.

**Phase 2 — Inference Systems** (papers 4–7)

Goal: understand how a production serving system works end-to-end. Read in strict order — each paper patches a bottleneck exposed by the previous.

- **4. [ORCA](papers/inference/orca-continuous-batching.md)**
  Iteration-level scheduling: don't wait for a full batch to finish. Establishes continuous batching as the foundational serving primitive.

- **5. [vLLM / PagedAttention](papers/inference/vllm-pagedattention.md)**
  KV cache fragmentation wastes 60–80% of GPU memory in ORCA's world. Fixed-size blocks + a page table eliminate waste and enable prefix sharing.

- **6. [Speculative Decoding](papers/inference/speculative-decoding.md)**
  Draft model generates k candidate tokens; target model verifies all k in one pass. Pure latency win; output distribution is mathematically identical to the target alone.

- **7. [Splitwise](papers/inference/splitwise-pd-disaggregation.md)**
  Prefill is compute-bound; decode is memory-bandwidth-bound. Route them to separate hardware pools sized for each workload.

**Phase 3 — Distributed Training** (papers 8–10)

Goal: understand how to train models that don't fit on one GPU or one node. Independent of Phase 2 — can be read in parallel if preferred.

- **8. [ZeRO](papers/training/zero-memory-optimization.md)**
  Optimizer states are 12× the parameter size at fp16 + Adam. Partition them across GPUs — each rank owns 1/N, gathers on demand.

- **9. [Megatron-LM](papers/training/megatron-lm.md)**
  3D parallelism: tensor parallel (within node), pipeline parallel (across nodes), data parallel (replicas). The canonical mental model for large-scale training. Read ZeRO first — it explains why naive data-parallel alone fails.

- **10. [MegaScale](papers/training/megascale.md)**
  Applies Megatron + ZeRO at 12K GPUs in production at ByteDance. The paper is about fault tolerance, straggler mitigation, and operational visibility — engineering problems that only appear at this scale.

**Phase 4 — Scheduling & SLOs** (papers 11–12, after Phase 2)

Goal: meet latency SLOs while maximizing GPU utilization. Requires Phase 2 as background.

- **11. [Sarathi-Serve](papers/scheduling/sarathi-serve.md)**
  A long prefill monopolizes the GPU for hundreds of ms, blocking all decode tokens. Chunked prefill + piggybacking ensures decodes proceed every iteration.

- **12. [Vidur](papers/scheduling/vidur.md)**
  Calibrated discrete-event simulator for LLM serving. Answers "how many GPUs do I need for X RPS at P99 < Y ms?" without real hardware.

**Phase 5 — Memory & Compression** (papers 13–14, can read any time after Phase 1)

Goal: fit larger models into the same hardware budget. Independent track — GPU memory pressure is the forcing function.

- **13. [GPTQ](papers/compression/gptq.md)**
  Layer-wise INT4 quantization using second-order (Hessian) information. Near-lossless quality; the principled baseline for post-training quantization.

- **14. [AWQ](papers/compression/awq.md)**
  1% of weight channels (those with large activation magnitudes) cause most quantization error. Scale them before quantizing — no Hessian needed, runs in minutes.

---

## Navigation

| # | Category | Topics | Papers |
|---|----------|--------|--------|
| 1–3 | [Foundations](#foundations) | Transformer, Attention Mechanisms, GPU Memory | 3 |
| 4–7 | [Inference Systems](#inference-systems) | Serving, KV Cache, Scheduling, Batching | 4 |
| 8–10 | [Distributed Training](#distributed-training) | Parallelism, Memory Optimization, Fault Tolerance | 3 |
| 11–12 | [Scheduling & SLOs](#scheduling--slos) | Latency Targets, Chunked Prefill, Capacity Planning | 2 |
| 13–14 | [Memory & Compression](#memory--compression) | Quantization, Weight Compression | 2 |

---

## Key Terminology

Core concepts that appear across all papers in this repo.

| Term | What it means |
|------|--------------|
| **KV cache** | Key and Value tensors from the attention mechanism, cached per token per layer during decoding. The central memory resource managed by vLLM, GQA, and every serving system here. |
| **Prefill** | Processing the input prompt — all tokens in parallel, one forward pass. Compute-bound. Produces the initial KV cache. |
| **Decode** | Generating output tokens one at a time, each requiring a forward pass over the full model. Memory-bandwidth-bound. |
| **TTFT / TPOT** | Time-to-first-token (prefill latency) and time-per-output-token (decode latency). The two SLO metrics for LLM serving. |
| **Continuous batching** | Making a new batching decision at every decode step instead of waiting for a full batch to finish. Eliminates GPU idle time. |
| **PagedAttention** | Dividing KV cache into fixed-size blocks mapped via a page table. Eliminates fragmentation and enables KV sharing. |
| **MFU** | Model FLOP Utilization — actual throughput as a fraction of theoretical peak. The efficiency headline in training papers. |
| **MHA / GQA / MQA** | Multi-head attention (h K/V heads), grouped-query attention (G K/V heads, G < h), multi-query attention (1 shared K/V head). Trade KV cache size for model quality. |
| **Tensor / Pipeline / Data parallelism** | Three orthogonal axes for distributing model training. TP splits layer weight matrices; PP splits layers across nodes; DP replicates the model across data shards. |
| **ZeRO** | Shards optimizer state, gradients, and parameters across data-parallel ranks to eliminate memory redundancy without changing the data-parallel programming model. |
| **Speculative decoding** | A small draft model generates k candidate tokens; the target model verifies all k in one pass. Provides a latency speedup with zero quality loss. |
| **Chunked prefill** | Splitting a long prefill into fixed-size chunks, interleaved with decode steps across iterations. Prevents prefill from monopolizing the GPU and blocking other requests. |
| **PD disaggregation** | Routing prefill and decode to separate hardware pools. Enables right-sizing each phase independently and eliminates prefill/decode interference. |
| **PTQ (post-training quantization)** | Compressing model weights to INT4/INT8 after training, without gradient updates. GPTQ uses second-order optimization; AWQ uses activation-guided scaling. |
| **HBM / SRAM** | GPU DRAM (large, slow, ~2–3 TB/s) vs. on-chip shared memory (small, fast, ~10–20 TB/s). The bandwidth gap between these two is the core bottleneck FlashAttention addresses. |

---

## Foundations

> Core architecture papers every LLM infra engineer should know deeply.
>
> Read in this order: the original architecture (what the KV cache is and why it exists) → modern attention variants (how KV cache size is controlled in production) → GPU memory optimization (how the attention kernel actually runs efficiently).

| # | Paper | Venue | Key Idea | Note |
|---|-------|-------|----------|------|
| 1 | [Attention Is All You Need](papers/foundations/attention-is-all-you-need.md) | NeurIPS '17 | Original Transformer architecture; origin of the KV cache and all modern LLMs | ✅ |
| 2 | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](papers/foundations/gqa.md) | EMNLP '23 | Group query heads to share K/V heads; 4–8× KV cache reduction with near-MHA quality | ✅ |
| 3 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](papers/foundations/flash-attention.md) | NeurIPS '22 | Tiling + recomputation to reduce HBM reads/writes | ✅ |

---

## Inference Systems

> How to serve LLMs efficiently at scale: latency, throughput, memory, and scheduling.
>
> Read in this order: continuous batching (scheduling model) → paged KV cache (memory model) → speculative decoding (latency trick) → PD disaggregation (cluster-level split).

| # | Paper | Venue | Key Idea | Note |
|---|-------|-------|----------|------|
| 1 | [ORCA: A Distributed Serving System for Transformer-based Generative Models](papers/inference/orca-continuous-batching.md) | OSDI '22 | Continuous batching (iteration-level scheduling) | ✅ |
| 2 | [Efficient Memory Management for LLM Serving with PagedAttention](papers/inference/vllm-pagedattention.md) | SOSP '23 | Virtual memory for KV cache; eliminates fragmentation | ✅ |
| 3 | [Fast Inference from Transformers via Speculative Decoding](papers/inference/speculative-decoding.md) | ICML '23 | Draft model speculatively generates k tokens; target model verifies in one pass | ✅ |
| 4 | [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](papers/inference/splitwise-pd-disaggregation.md) | ISCA '24 | Prefill-Decode disaggregation for resource efficiency | ✅ |

---

## Distributed Training

> Techniques to train 10B–1T parameter models across hundreds/thousands of GPUs.
>
> Read in this order: memory partitioning (the binding constraint) → model parallelism strategies → production systems engineering.

| # | Paper | Venue | Key Idea | Note |
|---|-------|-------|----------|------|
| 1 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](papers/training/zero-memory-optimization.md) | SC '20 | Partition optimizer states/gradients/parameters across GPUs | ✅ |
| 2 | [Megatron-LM: Training Multi-Billion Parameter Language Models](papers/training/megatron-lm.md) | SC '21 | 3D parallelism: tensor + pipeline + data | ✅ |
| 3 | [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](papers/training/megascale.md) | NSDI '24 | Production LLM training at ByteDance scale | ✅ |

---

## Scheduling & SLOs

> How to meet latency SLOs while maximizing GPU utilization — the production serving challenge.
>
> Read after Inference Systems (papers 4–7): Sarathi-Serve patches a specific bottleneck in the ORCA + vLLM model; Vidur is only useful once you understand what it simulates.

| # | Paper | Venue | Key Idea | Note |
|---|-------|-------|----------|------|
| 1 | [Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](papers/scheduling/sarathi-serve.md) | OSDI '24 | Chunk long prefills to prevent head-of-line blocking; piggyback decode steps | ✅ |
| 2 | [Vidur: A Large-Scale Simulation Framework For LLM Inference](papers/scheduling/vidur.md) | MLSys '24 | Simulate serving clusters to answer capacity planning questions without real hardware | ✅ |

---

## Memory & Compression

> Reduce model footprint without sacrificing quality — critical for serving large models on limited hardware.
>
> Can be read any time after Foundations. Read GPTQ before AWQ — AWQ's design is a direct critique of GPTQ's blind quantization.

| # | Paper | Venue | Key Idea | Note |
|---|-------|-------|----------|------|
| 1 | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](papers/compression/gptq.md) | ICLR '23 | Layer-wise INT4 quantization using second-order information; near-lossless at 4-bit | ✅ |
| 2 | [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](papers/compression/awq.md) | MLSys '24 | Protect salient weights (by activation magnitude) during INT4 quantization | ✅ |

---

## Contributing

Found an error? Have a paper to add? See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## About

Notes written by [@sandyhu533](https://github.com/sandyhu533) — Staff Engineer building large-scale AI/ML infrastructure. Focus on system design decisions and engineering tradeoffs.
