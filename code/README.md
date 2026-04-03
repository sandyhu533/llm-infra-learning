# Source Code Reading

Paper notes explain *what* systems do and *why*. Code reading shows *how* — the implementation decisions, performance knobs, and corner cases that papers skip.

This section covers three frameworks, each anchored to papers in the reading roadmap.

---

## Reading Order

Read the corresponding paper(s) first, then the code. Each framework picks up exactly where its paper left off.

| # | Framework | Learning Goal | Read After |
|---|-----------|--------------|------------|
| 1 | [vLLM](inference/vllm.md) | Continuous batching + paged KV cache in production code | ORCA + vLLM/PagedAttention papers |
| 2 | [DeepSpeed ZeRO](training/deepspeed-zero.md) | How Stage 2/3 partition optimizer state and params at runtime | ZeRO paper |
| 3 | [Megatron-LM](training/megatron-lm.md) | How TP layers and the 1F1B pipeline schedule work in code | Megatron-LM paper |

---

## What's Not Included

- **PyTorch FSDP** — implements ZeRO Stage 3 semantics, but DeepSpeed is the canonical reference for the ZeRO paper; reading both would duplicate the same concept
- **SGLang** — worth following as a modern inference engine (RadixAttention, prefix caching), but limited paper anchors in this repo currently
- **TensorRT-LLM** — NVIDIA-specific; the concepts are covered by vLLM with a cleaner codebase
- **FlashAttention CUDA kernels** — the paper itself is the better reference; reading raw CUDA/Triton for IO-aware tiling is below this repo's abstraction level

---

## Note Structure

Each code note follows this template:

1. **Learning Goal** — one sentence: what you'll understand after reading
2. **Entry Point** — single file + function to start from
3. **Key Files** — 4–6 files with specific guidance on what to look for
4. **Step Walkthrough** — trace a request or training step end-to-end
5. **Paper → Code Mapping** — explicit table linking paper concepts to code locations
6. **What the Paper Doesn't Tell You** — engineering decisions visible only in code
