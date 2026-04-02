# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Authors**: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai  
**Venue**: EMNLP '23  
**Paper**: [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)  
**Code**: [https://github.com/google-research/t5x](https://github.com/google-research/t5x)

---

## TL;DR

Multi-head attention (MHA) gives every attention head its own K and V projections — during autoregressive decoding, all of these must be cached and read from HBM on every step, causing a memory-bandwidth bottleneck. Multi-query attention (MQA) collapses all K and V heads into one shared head, slashing KV cache size by the number of heads (e.g., 32×) but degrading model quality. **GQA** (Grouped-Query Attention) splits the difference: group query heads into G groups (G > 1), each sharing one K/V head. With G=8, KV cache is 4× smaller than MHA (for 32 query heads) with near-MHA accuracy. GQA is the attention variant used in LLaMA 3, Mistral, Mixtral, Gemma, Falcon, and virtually every modern open-weight LLM — making it essential for understanding current KV cache sizing and memory requirements.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| MHA: one K/V per query head | Separate read replica per client | Every client gets its own data copy; maximum quality, maximum memory |
| MQA: one K/V for all heads | Single shared read replica (MySQL read replica) | All clients share one replica; minimum memory, possible degradation under high contention |
| GQA: one K/V per group of heads | Read replica per shard / region | Groups of clients share a replica; balanced memory vs quality tradeoff |
| KV cache per head | Per-replica buffer pool | Each replica (K/V head) has its own in-memory cache; proportional to number of replicas |
| Uptraining MHA → GQA | Online schema migration | Convert existing data (checkpoint) to new format with minimal reprocessing |
| G (number of groups) | Replication factor | Tunable: more groups = more memory, higher quality; fewer groups = less memory, lower quality |

---

## Problem

**What gap does this paper address?**

Modern LLMs use multi-head attention (MHA) with h=32–128 query heads. During autoregressive decode:
- Each layer stores K and V tensors for all h heads and all previous tokens
- KV cache size per layer = 2 × h × seq_len × d_k × sizeof(dtype)
- For LLaMA-2-70B (h=64, d_k=128, FP16): **~1MB per token across all layers**
- At sequence length 4096, batch size 32: **~130GB KV cache** — more than the model weights

This forces a choice: reduce batch size (less throughput) or context length (less capability).

Multi-query attention (MQA, Shazeer 2019) uses h=1 K/V head shared across all query heads. KV cache shrinks 64× but accuracy degrades noticeably — especially on tasks requiring nuanced context retrieval.

**Why is this hard?**

You can't simply train fewer K/V heads without losing the diversity of attention patterns that different heads capture. The challenge is to reduce KV cache significantly while preserving the representational capacity that MHA's independent heads provide.

---

## Key Ideas

### Idea 1: Grouped-Query Attention

Divide the h query heads into G groups. Each group shares a single K/V head. Each K/V head serves h/G query heads.

```
MHA (h=8, G=8):           GQA (h=8, G=2):           MQA (h=8, G=1):
Q1 → K1, V1              Q1, Q2, Q3, Q4 → K1, V1   Q1..Q8 → K1, V1
Q2 → K2, V2              Q5, Q6, Q7, Q8 → K2, V2
Q3 → K3, V3
Q4 → K4, V4
...
Q8 → K8, V8

KV heads: 8               KV heads: 2                KV heads: 1
KV cache: baseline        KV cache: baseline/4       KV cache: baseline/8
Quality: best             Quality: ~MHA               Quality: degraded
```

GQA is a strict generalization: G=h recovers MHA, G=1 recovers MQA. The parameter G is chosen to match hardware and quality constraints.

### Idea 2: Uptraining — Converting MHA Checkpoints to GQA

Training GQA from scratch requires significant compute. The paper proposes **uptraining**: take an existing MHA checkpoint and convert it to GQA with minimal additional training.

Conversion recipe:
1. **Mean-pool K/V heads** within each group: new K_g = mean(K_{g×(h/G)}, ..., K_{(g+1)×(h/G)-1})
2. Continue training for ~5% of the original training tokens (uptraining step)

This allows organizations to convert their existing MHA models (GPT-2, T5, early LLaMA) to GQA without full retraining.

```
MHA checkpoint (8 K/V heads):
  K heads: [K1, K2, K3, K4, K5, K6, K7, K8]

Convert to GQA (G=2, 2 K/V heads):
  K_g0 = mean(K1, K2, K3, K4)
  K_g1 = mean(K5, K6, K7, K8)

Uptrain for 5% of training steps → GQA checkpoint
```

### Idea 3: Memory-Bandwidth Impact on Decode Latency

GQA's primary benefit is at inference, not training. During decode:
- HBM reads per step (MHA) = 2 × h × seq_len × d_k × bytes + model_weights
- HBM reads per step (GQA, G groups) = 2 × G × seq_len × d_k × bytes + model_weights

For LLaMA-3-70B (h=64, G=8, d_k=128, FP16, batch=1, seq=4096):
```
MHA KV read: 2 × 64 × 4096 × 128 × 2 bytes = 134 MB per layer per step
GQA KV read: 2 × 8  × 4096 × 128 × 2 bytes = 17 MB per layer per step  ← 8× reduction
```

This directly reduces the memory-bandwidth bottleneck during decode, increasing tokens/sec and allowing larger batch sizes without exceeding GPU memory.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| KV cache memory (h/G × reduction) | Slight accuracy degradation vs MHA (typically <1% on benchmarks) |
| Decode throughput (less HBM BW per token) | G×h/G = h query heads → same Q projection cost |
| Larger batch size and/or longer context | Must choose G at training time (not tunable at serving time) |
| Training efficiency (smaller K/V matrices) | Less expressive attention than MHA at same model size |

**Design decisions worth questioning:**

- The choice of G is a training-time decision baked into the checkpoint. Serving teams are stuck with whatever the model was trained with. Ideally, G should be tunable per deployment scenario.
- Mean pooling for uptraining conversion may not be optimal. Learned aggregation (e.g., weighted sum of existing heads) could preserve more information.
- GQA reduces K/V cache but not the Q projection. For very large h, Q projection cost starts to matter — some architectures further reduce Q heads.
- All groups have the same number of query heads (h/G). Non-uniform grouping (some groups larger, some smaller) is unexplored but could help for heterogeneous tasks.

---

## Connections

**Builds on:**
- [Attention Is All You Need](attention-is-all-you-need.md) — defines MHA that GQA generalizes
- **Multi-Query Attention** (Shazeer, 2019) — the G=1 extreme that GQA generalizes from

**Inspired / Followed by:**
- [vLLM / PagedAttention](../inference/vllm-pagedattention.md) — KV cache block sizing must account for GQA's reduced K/V heads
- **LLaMA 3** (Meta, 2024) — uses GQA with G=8 for all model sizes
- **Mistral-7B** — one of the first models to popularize GQA in open-weight LLMs
- **Gemma, Falcon, Qwen** — all use GQA variants

**Production systems:**
- vLLM, TensorRT-LLM, SGLang all handle GQA-format KV cache natively
- Flash Attention-2 implements an efficient GQA kernel
- LLaMA.cpp supports GQA for all modern LLaMA variants

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| KV cache reduction | h/G × (e.g., 4-8×) | LLaMA-3-70B: h=64, G=8 → 8× reduction |
| Quality vs MHA | <1% degradation | On standard benchmarks (MMLU, HellaSwag) with G=8 |
| Quality vs MQA | Significantly better | GQA preserves diversity that MQA collapses |
| Uptraining cost | ~5% of pretraining tokens | To convert MHA checkpoint to GQA |
| Decode throughput increase | ~2-4× | At same batch size and sequence length vs MHA |

---

## Questions & Open Problems

- [ ] Is there a principled way to choose G given hardware constraints (HBM bandwidth, SRAM size, compute budget)?
- [ ] How does GQA interact with speculative decoding? The draft model also needs GQA-compatible KV cache management.
- [ ] Can the number of K/V heads be varied per layer (more heads in early layers, fewer in later layers) without quality loss?
- [ ] GQA reduces memory bandwidth for K/V reads. But for very long contexts (1M tokens), even 8× reduction may be insufficient. What's the right next step: further K/V compression, attention sparsity, or KV offloading?

---

## Reading Notes

GQA is one of the highest-leverage architectural decisions in modern LLM design — it's a simple idea (share K/V heads within groups) that directly cuts the dominant memory bottleneck in autoregressive decoding. The fact that virtually every post-2023 open-weight LLM uses it is a strong signal of its practical value.

For infrastructure engineers, GQA changes the KV cache sizing calculation. You can't use the "MHA formula" (2 × h × d_k × n_layers per token) for modern models — you must check the model's `num_key_value_heads` config (in HuggingFace terminology). For LLaMA-3-70B, that's 8 instead of 64, an 8× reduction in cache sizing.

The uptraining recipe is practically important: it means you can retrofit GQA onto existing MHA models without starting from scratch. If your team is serving a large MHA model and KV cache is the bottleneck, uptraining to GQA with 5% of the original training compute is likely worth it.
