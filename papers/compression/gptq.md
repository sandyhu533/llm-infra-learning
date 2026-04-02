# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh  
**Venue**: ICLR '23  
**Paper**: [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)  
**Code**: [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)

---

## TL;DR

Running a 175B-parameter LLM requires ~350GB of GPU memory in FP16 — far exceeding what's available on a single node. GPTQ is a post-training quantization method that compresses weights to 4-bit integers (INT4), reducing model size by 4x, while preserving accuracy to within ~1% perplexity degradation. It works by solving a layer-wise weight quantization problem using second-order (Hessian) information from a small calibration dataset, without any gradient updates or full model retraining. An OPT-175B model can be quantized on a single A100 in under 4 hours and then served with near-FP16 quality on 2-3x fewer GPUs.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| INT4 weight quantization | Lossy compression (JPEG, MP3) | Trade some precision for 4x size reduction; quality loss is bounded and acceptable for most use cases |
| Layer-wise quantization | Incremental database migration | Process one table (layer) at a time; don't need to rewrite the whole system |
| Hessian-guided error correction | Error-correcting codes / adaptive compression | Use statistical structure of the data to minimize information loss |
| Calibration dataset | Representative query set for query plan optimization | A small but representative sample is enough to estimate the cost landscape |
| Weight-only quantization | Read-only replica compression | Compress the static data (weights); keep the dynamic data (activations) at full precision |

---

## Problem

**What gap does this paper address?**

Quantization of LLMs was either:
1. **Training-time quantization**: requires full retraining with quantization-aware training (QAT) — impractical for 175B models
2. **Simple round-to-nearest (RTN)**: fast but causes catastrophic accuracy loss below 8-bit
3. **GPTQ's predecessor (OBQ)**: theoretically sound but O(d_col³) complexity per weight — too slow for billion-parameter models

None of these worked for 4-bit quantization of GPT-scale models within practical compute budgets.

**Why is this hard?**

Weights interact non-linearly with activations. Quantizing one weight changes the effective error on all subsequent computations in the layer. A naive greedy approach (quantize each weight independently) ignores these interactions and accumulates large errors. The correct formulation is: *find INT4 weights that minimize the output error of each layer given typical inputs*. This is a combinatorial optimization problem that naive methods cannot solve efficiently at scale.

---

## Key Ideas

### Idea 1: Layer-wise Quantization via OBS Framework

GPTQ decomposes the global problem into per-layer problems:

```
Global problem: minimize ||WX - Ŵ_q X||²
              over all layers simultaneously

GPTQ: minimize ||W_l X_l - Ŵ_q,l X_l||² independently for each layer l
```

This is sound because layer inputs (`X_l`) are treated as fixed during quantization of layer `l` (no retraining, no backprop). The layer-wise error bound does not accumulate unboundedly because errors are small in practice.

For each layer, GPTQ uses the **Optimal Brain Surgeon (OBS)** framework:
- Quantize one weight `w_q` at a time
- Compensate other weights in the same row to cancel the error introduced by quantizing `w_q`
- The compensation update uses the inverse Hessian of the layer's loss surface

```
After quantizing weight w_ij:
  Δw = -(w_ij - quant(w_ij)) / [H⁻¹]_jj × [H⁻¹]_j,·

where H = 2 X X^T  (Hessian of output MSE w.r.t. weights)
```

### Idea 2: Lazy Batch Updates (Efficient Hessian Inversion)

Computing the full Hessian inverse is O(d²) memory and O(d³) compute per layer. GPTQ exploits structure:

1. **Precompute H⁻¹ once per layer** using the Cholesky decomposition — O(d³) but done once
2. **Process weights column-by-column**: quantize a block of B columns (e.g., B=128), then apply the compensating update to all remaining columns in one BLAS operation
3. **Lazy batching**: defer the update for the current block until the block is fully quantized — this batches BLAS operations for GPU efficiency

```
for col_block in range(0, d_col, B):
    quantize weights in col_block (round to nearest INT4)
    error = W[:, col_block] - quant(W[:, col_block])
    W[:, col_block+B:] -= error @ H_inv[col_block, col_block+B:]
```

This reduces wall-clock time for OPT-175B quantization from weeks (naive OBS) to ~4 hours on a single A100.

### Idea 3: Weight Ordering Heuristic

GPTQ quantizes weights in order of their **diagonal Hessian entries** (ascending). Weights with smaller curvature (less sensitive to quantization) are quantized first, leaving compensation budget for the most sensitive weights. This simple ordering reduces quantization error by ~10–15% compared to left-to-right column order.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Model size (4x compression) | ~0.5–1% perplexity degradation at INT4 |
| No retraining required | Calibration dataset required (500–1024 samples) |
| GPU memory during inference | Dequantization overhead at runtime (~20% slowdown without INT4 kernels) |
| Quantization speed (hours, not weeks) | Less accurate than QAT at aggressive bit-widths (<4-bit) |

**Design decisions worth questioning:**

- GPTQ quantizes weights only (not activations). For very long sequences, activation memory can still dominate — weight-only quantization doesn't help there.
- The Hessian approximation assumes the calibration set is representative. Distribution shift between calibration and production traffic can degrade quality unexpectedly.
- At 3-bit, GPTQ degrades significantly. Below 4-bit, the information-theoretic floor of INT representation starts to matter.

---

## Connections

**Builds on:**
- **Optimal Brain Surgeon (OBS)** (Hassibi & Stork, 1993) — the second-order weight pruning framework that GPTQ adapts for quantization
- **SparseGPT** (same authors) — applies the same layer-wise OBS approach to unstructured pruning

**Inspired / Followed by:**
- [AWQ](awq.md) — identifies that 1% of weights (salient channels) dominate quantization error; achieves better quality than GPTQ without Hessian computation
- **GGUF/llama.cpp** — uses GPTQ-style INT4 for edge deployment on CPU/MPS
- **ExLlamaV2** — optimized CUDA INT4 kernels that make GPTQ inference 2–3x faster

**Production systems:**
- vLLM supports GPTQ-quantized models via AutoGPTQ integration
- Text Generation Inference (TGI, HuggingFace) ships GPTQ support in production
- Most open-weight model repos (LLaMA, Mistral, Mixtral) offer GPTQ-quantized variants

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Model size reduction | 4x | FP16 → INT4 |
| Perplexity degradation | <1% | OPT-175B, WikiText2 |
| Quantization time | ~4 hours | OPT-175B on single A100 |
| Memory reduction | ~350GB → ~90GB | OPT-175B FP16 → INT4 |
| Inference speedup | ~3.25x | A100, INT4 vs FP16 (with INT4 kernels) |

---

## Questions & Open Problems

- [ ] How does GPTQ interact with KV cache quantization? The paper quantizes only weights; activations (including KV cache) remain FP16 — is there a joint weight+KV quantization approach?
- [ ] The calibration assumption: how many samples are needed, and how sensitive is quality to calibration set distribution? What happens when production prompts are very different from calibration?
- [ ] At 2-bit, quantization error becomes severe. Can GPTQ be combined with low-rank residuals (LoRA-style) to recover quality at extreme compression?
- [ ] GPTQ processes layers independently. Does an end-to-end layer-dependent optimization (accounting for error propagation across layers) significantly improve quality?

---

## Reading Notes

The key insight is that OBS (a 1993 neural network pruning technique) directly solves the quantization error minimization problem — you're not pruning weights to zero but quantizing them to a discrete grid. The authors deserve credit for recognizing this connection.

The practical contribution is making OBS tractable at 175B scale: the lazy batch update trick turns an O(d³ × n_weights) algorithm into something that runs in hours. This is a good example of algorithm engineering: the math was known for 30 years; the contribution is efficient implementation.

For infrastructure engineers: GPTQ is worth knowing because it's the baseline most production quantization pipelines compare against. If you're serving a 70B model and don't have 4× A100s, GPTQ gets you to 2 A100s with acceptable quality loss. The tradeoff is well-understood and production-validated.
