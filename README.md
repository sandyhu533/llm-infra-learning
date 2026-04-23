# LLM Infra Learning

A curated collection of paper notes and insights on **LLM Infrastructure** — covering inference systems, distributed training, and foundational architectures.

> Notes focus on *engineering decisions and system tradeoffs*, not just algorithm descriptions.

---

## Reading Roadmap

Each paper builds on the previous — jumping ahead without the prerequisites means missing the "why."

Priority levels:
- **★★★ Core** — foundational vocabulary; every other paper assumes this knowledge
- **★★☆ Important** — essential for understanding system design at depth; read in full
- **★☆☆ Selective** — read the conclusion and key numbers; go deep only if directly relevant to your work

| # | Paper | Priority | Why this order |
|---|-------|----------|---------------|
| 1 | [Attention Is All You Need](papers/foundations/attention-is-all-you-need.md) | ★★★ Core | Defines KV cache, MHA, and the decode loop — the vocabulary for everything else |
| 2 | [GQA](papers/foundations/gqa.md) | ★★★ Core | Modern production models all use GQA; you need this to reason about KV cache sizing |
| 3 | [FlashAttention](papers/foundations/flash-attention.md) | ★★★ Core | Teaches GPU memory hierarchy (HBM vs SRAM); prerequisite for understanding any serving bottleneck |
| 4 | [ORCA](papers/inference/orca-continuous-batching.md) | ★★★ Core | First principles of serving: iteration-level scheduling, the reactor pattern for GPUs |
| 5 | [vLLM / PagedAttention](papers/inference/vllm-pagedattention.md) | ★★★ Core | Fixes KV cache fragmentation with virtual memory; builds directly on ORCA's scheduling model |
| 6 | [Speculative Decoding](papers/inference/speculative-decoding.md) | ★★☆ Important | A latency-only optimization; orthogonal to memory, easier to understand after vLLM |
| 7 | [Splitwise](papers/inference/splitwise-pd-disaggregation.md) | ★★☆ Important | Cluster-level CQRS: only makes sense once you understand prefill vs decode resource profiles |
| 8 | [ZeRO](papers/training/zero-memory-optimization.md) | ★★★ Core | Training track starts here: memory is the binding constraint; partition it first |
| 9 | [Megatron-LM](papers/training/megatron-lm.md) | ★★★ Core | Model parallelism strategies (TP/PP/DP); builds on ZeRO's memory intuition |
| 10 | [MegaScale](papers/training/megascale.md) | ★★☆ Important | What ZeRO + Megatron look like at 10K GPUs in production; engineering > algorithms |
| 11 | [Sarathi-Serve](papers/scheduling/sarathi-serve.md) | ★★☆ Important | Scheduling refinement: chunked prefill prevents head-of-line blocking; needs ORCA + vLLM first |
| 12 | [Vidur](papers/scheduling/vidur.md) | ★☆☆ Selective | Simulation framework; useful if you do capacity planning, otherwise skim the key ideas |
| 13 | [GPTQ](papers/compression/gptq.md) | ★☆☆ Selective | Understand the conclusion (Hessian-guided INT4); go deep only if you own a quantization pipeline |
| 14 | [AWQ](papers/compression/awq.md) | ★★☆ Important | Core insight (1% salient channels) is worth understanding; more elegant and practical than GPTQ |
| 15 | [InstructGPT](papers/rlhf/instructgpt.md) | ★★★ Core | Defines the canonical RLHF pipeline (SFT → RM → PPO); the 4-model setup is the infra baseline |
| 16 | [DPO](papers/rlhf/dpo.md) | ★★☆ Important | Eliminates reward model and PPO loop; reduces 4 models to 2 — a 2x memory simplification |
| 17 | [GRPO](papers/rlhf/grpo.md) | ★★★ Core | Eliminates critic via group sampling; online RLHF with only 2 models — the approach behind DeepSeek-R1 |
| 18 | [HybridFlow](papers/rlhf/hybridflow.md) | ★★★ Core | RLHF as a dataflow scheduling problem; 3D-HybridEngine reshards actor between train and generate — the veRL paper |

---

## Source Code Reading

Paper notes explain what and why. The [code/](code/README.md) section covers how — tracing key frameworks from entry point to kernel call.

Priority follows the same logic: Core if the framework is part of your daily stack, Selective if it's background knowledge.

| Framework | Priority | Learning Goal | Read After |
|-----------|----------|--------------|------------|
| [vLLM](code/inference/vllm.md) | ★★★ Core | Continuous batching + paged KV cache in production code | ORCA + vLLM papers |
| [Megatron-LM](code/training/megatron-lm.md) | ★★☆ Important | ColumnParallelLinear, RowParallelLinear, and 1F1B pipeline schedule | Megatron-LM paper |
| [DeepSpeed ZeRO](code/training/deepspeed-zero.md) | ★★☆ Important | Stage 2/3 optimizer: reduce-scatter, all-gather, param fetch/release | ZeRO paper |
| [SGLang](code/inference/sglang.md) | ★★☆ Important | RadixAttention prefix caching, cache-aware scheduling, FlashInfer integration | ORCA + vLLM papers |
| [TorchTitan](code/training/torchtitan.md) | ★★☆ Important | PyTorch-native composable parallelism (FSDP2 + TP + PP) — the anti-Megatron | ZeRO + Megatron-LM papers |
| [veRL](code/training/verl.md) | ★★★ Core | Multi-model RLHF orchestration, hybrid inference-training engine, GRPO implementation | InstructGPT + GRPO papers |

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
| **RLHF** | Reinforcement Learning from Human Feedback — fine-tuning with human preference data via a reward signal. |
| **PPO** | Proximal Policy Optimization — classic RLHF algorithm requiring actor, critic, reward, and reference models. |
| **DPO** | Direct Preference Optimization — eliminates reward model; optimizes preferences directly with 2 models. |
| **GRPO** | Group Relative Policy Optimization — eliminates critic via group sampling; online RLHF with 2 models. |
| **KL penalty** | Constrains the policy from diverging too far from the reference model; prevents reward hacking. |

---

## About

Notes written by [@sandyhu533](https://github.com/sandyhu533) — Staff Engineer building large-scale AI/ML infrastructure.
