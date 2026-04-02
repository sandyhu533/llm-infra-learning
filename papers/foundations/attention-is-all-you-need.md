# Attention Is All You Need

**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Venue**: NeurIPS '17  
**Paper**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
**Code**: [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

---

## TL;DR

Before 2017, sequence modeling relied on RNNs and LSTMs — inherently sequential architectures that cannot be parallelized over time steps and struggle to learn long-range dependencies. This paper introduces the **Transformer**: an encoder-decoder architecture built entirely on **self-attention**, with no recurrence or convolution. Self-attention computes pairwise interactions between all tokens in O(1) sequential steps (vs O(n) for RNNs), enabling full parallelization during training and better long-range dependency capture. The Transformer became the universal backbone for all modern LLMs — GPT, BERT, LLaMA, and every serving paper in this repo builds directly on this architecture.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| Self-attention (all-to-all token interaction) | Full-mesh network topology | Every node can directly communicate with every other node in one hop; contrast with RNN's linked-list |
| Multi-head attention | RAID striping / parallel query execution | Run multiple attention "heads" in parallel, each capturing different relationship types |
| KV cache (K and V matrices cached at decode time) | Read-through cache for database rows | Computed once during prefill, reused for every decode step — the direct origin of the KV cache in serving systems |
| Positional encoding | Sequence numbers / timestamps | Attention is order-invariant; positional encoding injects token position information explicitly |
| Transformer layer (attention + FFN) | Pipeline stage | A self-contained compute unit; stack N of them for depth; the natural unit for pipeline parallelism |
| Encoder-decoder vs decoder-only | Request-response vs streaming | Encoder-decoder: full input before output; decoder-only (GPT): generate token-by-token from the start |

---

## Problem

**What gap does this paper address?**

RNN/LSTM-based sequence models had three fundamental limitations:
1. **Sequential computation**: hidden state `h_t` depends on `h_{t-1}` — you cannot parallelize across timesteps during training. Training large models on long sequences was slow.
2. **Long-range dependencies**: gradient signal over long sequences vanishes or explodes. Even LSTMs with gating struggle with dependencies >500 tokens.
3. **Memory of the past**: everything must be compressed into a fixed-size hidden vector — information bottleneck for long sequences.

Convolutional approaches (ByteNet, ConvS2S) improved parallelism but required O(log n) or O(n) stacked layers to connect distant tokens.

**Why is this hard?**

The challenge is to capture arbitrary pairwise dependencies between tokens *efficiently*. Naive all-pairs attention is O(n²) in sequence length — this was acceptable for 512-token sequences but becomes the key bottleneck for long-context models (exactly what FlashAttention, GQA, and other papers in this repo address).

---

## Key Ideas

### Idea 1: Scaled Dot-Product Attention

The core operation: for each token, compute a weighted sum of all value vectors, where weights are proportional to the dot product with each key vector.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Shapes:
  Q: [seq_len, d_k]   — query vectors (what am I looking for?)
  K: [seq_len, d_k]   — key vectors   (what do I offer to match against?)
  V: [seq_len, d_v]   — value vectors (what information do I carry?)

Output: [seq_len, d_v] — each token's contextual representation

Complexity:
  Time:   O(n² · d)   — n² attention scores, each a d-dim dot product
  Memory: O(n²)       — the n×n attention matrix
```

The `/√d_k` scaling prevents dot products from growing too large in high dimensions (which would push softmax into near-zero gradient regions).

**Why this matters for serving**: During autoregressive decoding, we generate one token at a time. For each new token, we need Q (new token), K (all previous tokens), and V (all previous tokens). Storing K and V for all previous tokens is exactly the **KV cache** that vLLM, PagedAttention, GQA, and every serving paper manages.

### Idea 2: Multi-Head Attention

Instead of one attention function over d_model dimensions, use h parallel attention heads over d_k = d_model/h dimensions each:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q · W_Q_i, K · W_K_i, V · W_V_i)
```

Different heads learn to attend to different types of relationships (syntactic structure, coreference, semantic similarity). The h heads run in parallel — this is the natural unit of tensor parallelism in multi-GPU serving.

**KV cache implication**: Each head has its own K and V matrices. For h=32 heads, d_model=4096, seq_len=n: KV cache size = 2 × h × n × d_k × sizeof(dtype) per layer. This is why GQA (sharing K/V across query heads) is so impactful for memory.

### Idea 3: Feed-Forward Network + Layer Stack

Each Transformer layer consists of:
1. Multi-head attention (with residual + LayerNorm)
2. Position-wise FFN: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2` (with residual + LayerNorm)

The FFN is applied independently to each token position — it's a dense computation that dominates FLOPs (typically 4× larger than attention). Stacking N layers gives the model representational depth.

```
Transformer layer:
  x → LayerNorm → MultiHeadAttn → + residual → LayerNorm → FFN → + residual → output

Stack N of these = full Transformer encoder (or decoder with causal masking)
```

**Decoder-only vs encoder-decoder**: GPT-family models use decoder-only (no cross-attention, just causal self-attention). This is now the standard for LLMs — one unified stack, no separate encoder.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Training parallelism (all tokens in parallel) | O(n²) attention memory — bottleneck for long contexts |
| Long-range dependencies (direct path between any two tokens) | Quadratic compute scaling with sequence length |
| Simple, uniform architecture | No inductive bias for locality — needs positional encoding |
| Flexible attention patterns | Autoregressive decode is sequential (one token at a time) |

**Design decisions worth questioning:**

- Learned absolute positional encoding (used in the original paper) doesn't generalize to longer sequences than seen during training. Modern LLMs use RoPE (rotary position embeddings) or ALiBi instead.
- The FFN intermediate size (4× d_model) is empirical. Some models use different ratios; MoE replaces FFN with sparse expert layers.
- The original paper uses encoder-decoder for translation. Modern LLMs are decoder-only — this simplification enables much more efficient training and serving.

---

## Connections

**Builds on:**
- **Bahdanau Attention** (2015) — introduced attention for seq2seq; the Transformer generalizes and removes the RNN
- **ByteNet / ConvS2S** — convolutional sequence models that inspired the parallelism goal

**Inspired / Followed by:**
- **BERT** (encoder-only, bidirectional) and **GPT** (decoder-only, causal) — the two dominant architectures derived from this paper
- [FlashAttention](flash-attention.md) — IO-aware rewrite of the O(n²) attention kernel to reduce HBM bandwidth
- [GQA](gqa.md) — reduces KV cache size by sharing K/V heads across query heads
- Every serving paper in this repo (vLLM, Orca, Sarathi-Serve) — all manage the KV cache that exists because of this paper's architecture

**Production systems:**
- Every major LLM in production (GPT-4, Claude, Gemini, LLaMA, Mistral) is a Transformer variant
- The KV cache (K and V matrices from this paper) is the central resource managed by all modern serving infrastructure

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| BLEU score (EN-DE translation) | 28.4 | State of the art in 2017; surpassed all prior RNN/CNN models |
| Training cost (base model) | $4,500 (estimated) | 0.5 days on 8 P100s; vs weeks for comparable RNN models |
| Attention complexity | O(n² · d) | Dominant term for long sequences — motivation for FlashAttention |
| KV cache per token per layer | 2 × d_model × sizeof(dtype) | Origin of the memory bottleneck in modern LLM serving |

---

## Questions & Open Problems

- [ ] Positional encoding: learned absolute (original) vs RoPE vs ALiBi vs NTK-scaling — what's the theoretically principled choice for long-context generalization?
- [ ] The attention operation is the only weight-free operation in the Transformer — its expressivity comes entirely from Q/K/V projections. Can we design sparser or more structured attention patterns without losing capability?
- [ ] Does the O(n²) attention complexity fundamentally limit LLMs to moderate context lengths, or can linear attention variants (Mamba, RWKV) match quality at linear cost?
- [ ] Multi-head attention runs h independent small attentions. Is there evidence that different heads actually specialize, or is this just an implementation artifact?

---

## Reading Notes

This is the paper that started everything — reading it is mandatory not for novelty but for understanding why every subsequent paper makes the architectural choices it does.

From an infra perspective, the most important contribution is not the accuracy improvement — it's the **parallelizability**. RNNs cannot be distributed across sequence dimension; Transformers can be distributed across sequence, batch, and model dimensions simultaneously. This is what makes trillion-parameter models trainable (see Megatron-LM, ZeRO) and what makes all the serving optimizations in this repo possible.

The KV cache is not mentioned in this paper — it's an obvious optimization when you notice that K and V for prior tokens don't change during autoregressive decode. It took the serving community a few years to realize how central this optimization is, and then a few more to realize it was the primary memory bottleneck (vLLM, 2023).
