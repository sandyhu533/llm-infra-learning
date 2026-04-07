# Paper Notes

Notes on LLM infrastructure papers, organized by topic. Each note follows a consistent structure — see [TEMPLATE.md](TEMPLATE.md).

---

## Reading Order

Read in sequence. Each paper builds on the previous — jumping ahead without the prerequisites means missing the "why."

| # | Paper | Why this order |
|---|-------|---------------|
| 1 | [Attention Is All You Need](foundations/attention-is-all-you-need.md) | Defines KV cache, MHA, and the decode loop — the vocabulary for everything else |
| 2 | [GQA](foundations/gqa.md) | Modern production models all use GQA; you need this to reason about KV cache sizing |
| 3 | [FlashAttention](foundations/flash-attention.md) | Teaches GPU memory hierarchy (HBM vs SRAM); prerequisite for understanding any serving bottleneck |
| 4 | [ORCA](inference/orca-continuous-batching.md) | First principles of serving: iteration-level scheduling, the reactor pattern for GPUs |
| 5 | [vLLM / PagedAttention](inference/vllm-pagedattention.md) | Fixes KV cache fragmentation with virtual memory; builds directly on ORCA's scheduling model |
| 6 | [Speculative Decoding](inference/speculative-decoding.md) | A latency-only optimization; orthogonal to memory, easier to understand after vLLM |
| 7 | [Splitwise](inference/splitwise-pd-disaggregation.md) | Cluster-level CQRS: only makes sense once you understand prefill vs decode resource profiles |
| 8 | [ZeRO](training/zero-memory-optimization.md) | Training track starts here: memory is the binding constraint; partition it first |
| 9 | [Megatron-LM](training/megatron-lm.md) | Model parallelism strategies (TP/PP/DP); builds on ZeRO's memory intuition |
| 10 | [MegaScale](training/megascale.md) | What ZeRO + Megatron look like at 10K GPUs in production; engineering > algorithms |
| 11 | [Sarathi-Serve](scheduling/sarathi-serve.md) | Scheduling refinement: chunked prefill prevents head-of-line blocking; needs ORCA + vLLM first |
| 12 | [Vidur](scheduling/vidur.md) | Simulation framework; only useful if you understand what it's simulating (papers 4–7) |
| 13 | [GPTQ](compression/gptq.md) | Post-training quantization; independent thread, but easier after you understand serving memory pressure |
| 14 | [AWQ](compression/awq.md) | Refines GPTQ by protecting salient weights; read GPTQ first |
| 15 | [InstructGPT](rlhf/instructgpt.md) | Defines the canonical 3-stage RLHF pipeline (SFT → RM → PPO); the 4-model infra baseline |
| 16 | [DPO](rlhf/dpo.md) | Eliminates reward model and PPO; reduces to supervised-style training with 2 models |
| 17 | [GRPO](rlhf/grpo.md) | Eliminates critic via group sampling; online RLHF with 2 models — powers DeepSeek-R1 |

---

## What's Not Included

- Pure ML architecture papers without a systems contribution
- Papers without a clear engineering tradeoff story
- Blog posts or secondary sources

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full scope criteria and how to add a note.

---

## Note Structure

Each paper note follows this template:

1. **TL;DR** — one paragraph readable in 60 seconds
2. **Problem** — what was broken before this paper, and why naive solutions fail
3. **Key Ideas** — core technical decisions with data flow diagrams where helpful
4. **System Tradeoffs** — what the design optimizes for, at what explicit cost
5. **Connections** — what this builds on and what it enabled
6. **Key Numbers** — benchmark figures from the paper
7. **Questions & Open Problems** — non-trivial open questions
8. **Reading Notes** — personal observations; include infra analogies here where they naturally fit
