# LLM Infra Learning

A curated collection of paper notes and insights on **LLM Infrastructure** — covering inference systems, distributed training, and foundational architectures.

> Notes focus on *engineering decisions and system tradeoffs*, not just algorithm descriptions.

---

## Reading Roadmap

Each paper builds on the previous — jumping ahead without the prerequisites means missing the "why."

| # | Paper | Why this order |
|---|-------|---------------|
| 1 | [Attention Is All You Need](papers/foundations/attention-is-all-you-need.md) | Defines KV cache, MHA, and the decode loop — the vocabulary for everything else |
| 2 | [GQA](papers/foundations/gqa.md) | Modern production models all use GQA; you need this to reason about KV cache sizing |
| 3 | [FlashAttention](papers/foundations/flash-attention.md) | Teaches GPU memory hierarchy (HBM vs SRAM); prerequisite for understanding any serving bottleneck |
| 4 | [ORCA](papers/inference/orca-continuous-batching.md) | First principles of serving: iteration-level scheduling, the reactor pattern for GPUs |
| 5 | [vLLM / PagedAttention](papers/inference/vllm-pagedattention.md) | Fixes KV cache fragmentation with virtual memory; builds directly on ORCA's scheduling model |
| 6 | [Speculative Decoding](papers/inference/speculative-decoding.md) | A latency-only optimization; orthogonal to memory, easier to understand after vLLM |
| 7 | [Splitwise](papers/inference/splitwise-pd-disaggregation.md) | Cluster-level CQRS: only makes sense once you understand prefill vs decode resource profiles |
| 8 | [ZeRO](papers/training/zero-memory-optimization.md) | Training track starts here: memory is the binding constraint; partition it first |
| 9 | [Megatron-LM](papers/training/megatron-lm.md) | Model parallelism strategies (TP/PP/DP); builds on ZeRO's memory intuition |
| 10 | [MegaScale](papers/training/megascale.md) | What ZeRO + Megatron look like at 10K GPUs in production; engineering > algorithms |
| 11 | [Sarathi-Serve](papers/scheduling/sarathi-serve.md) | Scheduling refinement: chunked prefill prevents head-of-line blocking; needs ORCA + vLLM first |
| 12 | [Vidur](papers/scheduling/vidur.md) | Simulation framework; only useful if you understand what it's simulating (papers 4–7) |
| 13 | [GPTQ](papers/compression/gptq.md) | Post-training quantization; independent thread, but easier after you understand serving memory pressure |
| 14 | [AWQ](papers/compression/awq.md) | Refines GPTQ by protecting salient weights; read GPTQ first |

---

## Source Code Reading

Paper notes explain what and why. The [code/](code/README.md) section covers how — tracing key frameworks from entry point to kernel call.

| Framework | Learning Goal | Read After |
|-----------|--------------|------------|
| [vLLM](code/inference/vllm.md) | Continuous batching + paged KV cache in production code | ORCA + vLLM papers |
| [DeepSpeed ZeRO](code/training/deepspeed-zero.md) | Stage 2/3 optimizer: reduce-scatter, all-gather, param fetch/release | ZeRO paper |
| [Megatron-LM](code/training/megatron-lm.md) | ColumnParallelLinear, RowParallelLinear, and 1F1B pipeline schedule | Megatron-LM paper |

---

## Key Terminology

| Term | What it means |
|------|--------------|
| **KV cache** | Key and Value tensors cached per token per layer during decoding. The central memory resource managed by vLLM, GQA, and every serving system here. |
| **Prefill** | Processing the input prompt — all tokens in parallel, one forward pass. Compute-bound. |
| **Decode** | Generating output tokens one at a time. Memory-bandwidth-bound. |
| **TTFT / TPOT** | Time-to-first-token (prefill latency) and time-per-output-token (decode latency). The two SLO metrics for LLM serving. |
| **Continuous batching** | Making a new batching decision at every decode step. Eliminates GPU idle time. |
| **PagedAttention** | KV cache in fixed-size blocks mapped via a page table. Eliminates fragmentation and enables prefix sharing. |
| **MFU** | Model FLOP Utilization — actual throughput as a fraction of theoretical peak. |
| **MHA / GQA / MQA** | Multi-head / grouped-query / multi-query attention. Trade KV cache size for model quality. |
| **Tensor / Pipeline / Data parallelism** | TP splits layer weight matrices; PP splits layers across nodes; DP replicates the model across data shards. |
| **ZeRO** | Shards optimizer state, gradients, and parameters across DP ranks to eliminate memory redundancy. |
| **Speculative decoding** | Draft model generates k candidates; target model verifies all k in one pass. Latency speedup with zero quality loss. |
| **Chunked prefill** | Split a long prefill into fixed-size chunks interleaved with decode steps. Prevents head-of-line blocking. |
| **PD disaggregation** | Route prefill and decode to separate hardware pools sized for each workload. |
| **PTQ** | Post-training quantization — compress weights to INT4/INT8 after training, without gradient updates. |
| **HBM / SRAM** | GPU DRAM (large, slow, ~2–3 TB/s) vs. on-chip shared memory (small, fast, ~10–20 TB/s). |

---

## About

Notes written by [@sandyhu533](https://github.com/sandyhu533) — Staff Engineer building large-scale AI/ML infrastructure.
