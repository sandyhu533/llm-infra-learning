# LLM Infra Learning

A curated collection of paper notes and insights on **LLM Infrastructure** — covering inference systems, distributed training, and foundational architectures.

> Built for personal learning and shared for the community. Notes focus on *engineering decisions and system tradeoffs*, not just algorithm descriptions.

---

## Navigation

| Category | Topics | Papers |
|----------|--------|--------|
| [Inference Systems](#inference-systems) | Serving, KV Cache, Scheduling, Batching | 3 |
| [Distributed Training](#distributed-training) | Parallelism, Memory Optimization, Fault Tolerance | 3 |
| [Foundations](#foundations) | Transformer, Attention Mechanisms | 1 |
| [Scheduling & SLOs](#scheduling--slos) | Latency Targets, Chunked Prefill, Capacity Planning | 2 |
| [Memory & Compression](#memory--compression) | Quantization, Weight Compression | 2 |

---

## Concept Analogy Map

> If you're coming from backend / distributed systems / OS engineering, this is your translation guide. LLM infra reuses the same ideas — different names, same tradeoffs.

| LLM Infra Concept | Traditional Infra Analogy | Key Insight |
|-------------------|--------------------------|-------------|
| **KV cache** | Buffer pool (InnoDB) / page cache | Hot working-set data cached near compute; eviction when memory pressure hits |
| **Continuous batching** | Event-driven I/O — epoll / reactor | Don't block a thread per request; multiplex many in-flight requests over shared GPU |
| **PagedAttention** | OS virtual memory / paging | Non-contiguous physical memory mapped via page table; copy-on-write for sharing |
| **Prefill phase** | Query parse + compile / transaction begin | CPU-bound setup before the streaming result starts; run once, amortized |
| **Decode phase** | Cursor iteration / streaming result set | Incremental output, one token at a time; memory-bandwidth bound, not compute |
| **Speculative decoding** | Branch prediction / prefetching | Speculatively produce output, verify cheaply; roll back on mispredict |
| **Tensor parallelism** | Horizontal DB sharding | Partition a large weight matrix across devices; all-reduce = scatter-gather |
| **Pipeline parallelism** | CPU pipeline stages / staged request processing | Split model layers across devices; bubble = pipeline stall |
| **Data parallelism** | Replica sets / read replicas | Each replica holds the full model; gradients = write quorum sync |
| **ZeRO optimizer** | Distributed slab allocator | Shard optimizer state across nodes; all-gather before use, reduce-scatter after |
| **PD disaggregation** | CQRS — separate read/write paths | Prefill (write-heavy, compute-bound) and decode (read-heavy, memory-bound) have different resource profiles; split them |
| **Quantization (INT8/INT4)** | Lossy compression / float→int narrowing | Trade precision for throughput/memory; same as JPEG vs PNG for weights |
| **FlashAttention tiling** | Cache-oblivious / blocked matrix multiply | Restructure computation to fit working set in L1/SRAM; avoid HBM round-trips |
| **HBM (GPU main memory)** | NUMA memory / main DRAM | Large, slow-ish, expensive; bandwidth is the bottleneck |
| **Shared memory (SRAM)** | L1/L2 cache | Small, fast, on-chip; explicit management in CUDA = manual cache control |
| **GPU Streaming Multiprocessor** | CPU core with wide SIMD | Execute thousands of threads in lockstep; warp divergence = branch misprediction penalty |
| **MFU (Model FLOP Utilization)** | CPU utilization / IPC | How much of peak theoretical throughput you're actually using |
| **KV cache eviction** | Buffer pool eviction (LRU/LFU) | When memory is full, decide which cached state to drop |
| **Prefix caching** | Query result cache / memoization | Cache the KV state of a shared prefix; hit = skip recomputation |

---

## Inference Systems

> How to serve LLMs efficiently at scale: latency, throughput, memory, and scheduling.

| Paper | Venue | Key Idea | Note |
|-------|-------|----------|------|
| [Efficient Memory Management for LLM Serving with PagedAttention](papers/inference/vllm-pagedattention.md) | SOSP '23 | Virtual memory for KV cache; eliminates fragmentation | ✅ |
| [ORCA: A Distributed Serving System for Transformer-based Generative Models](papers/inference/orca-continuous-batching.md) | OSDI '22 | Continuous batching (iteration-level scheduling) | ✅ |
| [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](papers/inference/splitwise-pd-disaggregation.md) | ISCA '24 | Prefill-Decode disaggregation for resource efficiency | ✅ |

---

## Distributed Training

> Techniques to train 10B–1T parameter models across hundreds/thousands of GPUs.

| Paper | Venue | Key Idea | Note |
|-------|-------|----------|------|
| [Megatron-LM: Training Multi-Billion Parameter Language Models](papers/training/megatron-lm.md) | SC '21 | 3D parallelism: tensor + pipeline + data | ✅ |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](papers/training/zero-memory-optimization.md) | SC '20 | Partition optimizer states/gradients/parameters across GPUs | ✅ |
| [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](papers/training/megascale.md) | NSDI '24 | Production LLM training at ByteDance scale | ✅ |


## Foundations

> Core architecture papers every LLM infra engineer should know deeply.

| Paper | Venue | Key Idea | Note |
|-------|-------|----------|------|
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](papers/foundations/flash-attention.md) | NeurIPS '22 | Tiling + recomputation to reduce HBM reads/writes | ✅ |

---

## Scheduling & SLOs

> How to meet latency SLOs while maximizing GPU utilization — the production serving challenge.

| Paper | Venue | Key Idea | Note |
|-------|-------|----------|------|
| [Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](papers/scheduling/sarathi-serve.md) | OSDI '24 | Chunk long prefills to prevent head-of-line blocking; piggyback decode steps | 📋 |
| [Vidur: A Large-Scale Simulation Framework For LLM Inference](papers/scheduling/vidur.md) | MLSys '24 | Simulate serving clusters to answer capacity planning questions without real hardware | 📋 |

---

## Memory & Compression

> Reduce model footprint without sacrificing quality — critical for serving large models on limited hardware.

| Paper | Venue | Key Idea | Note |
|-------|-------|----------|------|
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](papers/compression/gptq.md) | ICLR '23 | Layer-wise INT4 quantization using second-order information; near-lossless at 4-bit | 📋 |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](papers/compression/awq.md) | MLSys '24 | Protect salient weights (by activation magnitude) during INT4 quantization | 📋 |

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

If you're coming from backend/distributed systems, suggested reading order with the lens to apply:

```
Foundations
└── FlashAttention          ← "cache-oblivious blocked matmul"
                              Understand GPU memory hierarchy (HBM vs SRAM)
                              before everything else

Inference Systems
├── Orca                    ← "event loop / reactor pattern"
│                             Continuous batching = don't block on one request
├── vLLM (PagedAttention)   ← "OS virtual memory"
│                             KV cache fragmentation = heap fragmentation
└── Splitwise               ← "CQRS / read-write separation"
                              Prefill ≠ Decode; give them different hardware

Distributed Training
├── ZeRO                    ← "distributed slab allocator"
│                             Shard optimizer state like you'd shard a hash table
├── Megatron-LM             ← "sharding + pipelining"
│                             Tensor parallel = column-wise DB sharding
└── MegaScale               ← "production SRE at 10K-GPU scale"
                              Same lessons as any large distributed system:
                              fault tolerance, observability, config management

Scheduling & SLOs (after inference basics)
└── Sarathi-Serve           ← "head-of-line blocking prevention"
                              Chunked prefill = HTTP/2 stream multiplexing
```

---

## Contributing

Found an error? Have a paper to add? See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## About

Notes written by [@sandyhu533](https://github.com/sandyhu533) — Staff Engineer building large-scale AI/ML infrastructure. Focus on system design decisions and engineering tradeoffs.
