# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**Authors**: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro  
**Venue**: SC '21 (original: NeurIPS Workshop '19, extended SC '21)  
**Paper**: [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)  
**Code**: [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

---

## TL;DR

Training models with 10B+ parameters requires distributing computation across many GPUs. Naive data parallelism fails when the model itself doesn't fit on a single GPU. Megatron-LM introduces **tensor parallelism** (splitting individual layers across GPUs) combined with pipeline parallelism, enabling efficient training of GPT-3 scale models on thousands of GPUs. The key insight is that Transformer's matrix multiplications can be partitioned in ways that require only *two* AllReduce operations per Transformer block — minimizing communication overhead while keeping all GPUs busy.

---

## Problem

**What gap does this paper address?**

A 175B-parameter GPT-3 requires ~350GB in fp16 — far exceeding the memory of any single GPU (A100 has 80GB). Data parallelism (replicate model on each GPU, split data) doesn't help when the *model* doesn't fit on one GPU.

Prior model parallelism approaches:
- **Naive layer splitting** (each GPU holds a subset of layers): causes sequential execution (one GPU runs while others wait), low utilization
- **GPipe pipeline parallelism**: reduces idle time with micro-batching but still has pipeline bubbles and requires careful memory management

**Why is this hard?**

Splitting a layer across GPUs introduces communication at every forward and backward pass. If communication is expensive relative to computation, the speedup from adding GPUs is cancelled by communication overhead. The challenge is finding tensor split patterns where communication is minimal.

---

## Key Ideas

### Idea 1: Tensor Parallelism for Transformer Blocks

The core observation: Transformer blocks contain two types of large matrix multiplications that can be split column-wise or row-wise with only one synchronization point each.

**MLP block** (two linear layers + GeLU):

```
Input X → [A1|A2] → GeLU → [B1; B2] → Output Y

GPU 1: X → A1 → GeLU(Z1) → B1 → partial Y1
GPU 2: X → A2 → GeLU(Z2) → B2 → partial Y2

AllReduce(Y1 + Y2) → Y
```

Column-parallel first layer, row-parallel second layer → 1 AllReduce in forward, 1 in backward.

**Self-Attention block**:

Similarly, split attention heads across GPUs. Each GPU computes a subset of heads, then AllReduce on the output projection.

### Idea 2: 3D Parallelism (Tensor + Pipeline + Data)

Real large-scale training combines three parallelism dimensions:

```
┌─────────────────────────────────────────────────────┐
│                   Data Parallel                      │  ← replicate across DP groups
│  ┌───────────────────────────────────────────────┐   │
│  │              Pipeline Parallel                │   │  ← split layers into stages
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │Stage 0  │→ │Stage 1  │→ │Stage 2  │      │   │
│  │  │[Tensor  │  │[Tensor  │  │[Tensor  │      │   │
│  │  │Parallel]│  │Parallel]│  │Parallel]│      │   │
│  │  └─────────┘  └─────────┘  └─────────┘      │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- **Tensor parallelism** (TP): within a node (NVLink bandwidth; typically TP=8)
- **Pipeline parallelism** (PP): across nodes (slower inter-node; split model layers into stages)
- **Data parallelism** (DP): across replicas (gradients averaged via AllReduce)

### Idea 3: Interleaved Pipeline Scheduling

Standard pipeline parallelism (GPipe) has large "pipeline bubbles" where GPUs wait at the start/end of each batch. Megatron-LM introduces **interleaved scheduling** with multiple micro-batches to reduce bubble fraction:

```
Bubble fraction ≈ (p-1) / (m + p - 1)
```

where p = pipeline stages, m = micro-batches. More micro-batches → smaller bubble.

---

## System Tradeoffs

| Parallelism | Communication | Memory savings | Use when |
|-------------|---------------|----------------|----------|
| Tensor (TP) | AllReduce each layer | Linear in TP degree | Within a node (fast NVLink) |
| Pipeline (PP) | P2P between stages | Linear in PP degree | Across nodes |
| Data (DP) | AllReduce gradients | None (full model replicated) | Always, as outer loop |

**Design decisions worth questioning:**

- TP requires fast intra-node communication (NVLink). On commodity clusters with PCIe only, TP overhead is much higher. What's the right TP degree for a given interconnect?
- Pipeline bubbles waste up to ~10% of GPU time even with interleaved scheduling. ZeRO (data parallelism with sharded optimizer states) can sometimes be more efficient than PP for models that fit on fewer GPUs.
- Tensor parallelism for attention is clean for MHA but gets more complex for GQA (grouped query attention) used in LLaMA/Mistral. How does the split change?

---

## Connections

**Builds on:**
- GPipe — pipeline parallelism concepts
- Model parallelism in Megatron (NVIDIA's internal work)

**Inspired / Followed by:**
- [ZeRO](zero-memory-optimization.md) — alternative approach: keep full parallelism but shard optimizer state
- **Megatron-LM v3** — adds sequence parallelism for LayerNorm/Dropout
- [MegaScale](megascale.md) — production application of these techniques at ByteDance scale

**Production systems:**
- GPT-3, GPT-4 training (OpenAI)
- LLaMA training (Meta)
- Most large-scale LLM training pipelines use some form of 3D parallelism

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Training scale | 8.3B params | Original paper; later versions up to 530B |
| GPU efficiency | ~50% of theoretical peak | On 3072 A100s |
| TP communication overhead | ~1–2 AllReduce per layer | In forward pass |
| Pipeline bubble | ~5–10% | With interleaved scheduling |

---

## Questions & Open Problems

- [ ] How does TP degree interact with batch size? Very small batches may not keep tensor-parallel GPUs busy.
- [ ] Sequence parallelism (splitting the sequence dimension) is needed for very long contexts — how does it compose with TP and PP?
- [ ] For MoE (Mixture of Experts) models, expert parallelism is needed. How does 4D parallelism (TP + PP + DP + EP) compose?
- [ ] What's the sweet spot for each parallelism dimension given different hardware topologies (NVLink vs InfiniBand vs PCIe)?

---

## Reading Notes

The "two AllReduce per Transformer block" insight is the core of this paper and worth dwelling on. The reason it works is that Transformer's MLP is just two GEMMs with an elementwise nonlinearity in between — the column/row split can be arranged so that the intermediate activation is *never communicated*, only the final output is.

The 3D parallelism diagram is the most important mental model for LLM training infrastructure. Every conversation about "how do you train a 100B model" should start here.
