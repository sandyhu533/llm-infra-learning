# AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration

**Authors**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han  
**Venue**: MLSys '24  
**Paper**: [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)  
**Code**: [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

---

## TL;DR

GPTQ minimizes per-layer quantization error but treats all weights symmetrically. AWQ observes that only ~1% of weight channels — those corresponding to large activation magnitudes — are responsible for most of the quantization error. Instead of using expensive Hessian computations to compensate post-quantization, AWQ **scales these salient channels before quantization** (by activation magnitude) so they are quantized more accurately. The result: better accuracy than GPTQ at INT4 with no Hessian computation, hardware-friendly pure INT4 output, and faster inference on edge devices (1.45x vs GPTQ on mobile GPUs). AWQ is now the dominant quantization method for on-device LLM deployment.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| Salient weights (1% of channels) | Hot paths in a profiler (80/20 rule) | 1% of code/weights causes 80–99% of the quality loss; optimize the hot path |
| Per-channel activation scaling | Adaptive precision in floating-point (exponent bias) | Shift the numeric range where it matters most before quantizing to fixed-point |
| Hardware-friendly pure INT4 | Uniform data encoding for SIMD efficiency | Mixed-precision (some FP16, some INT4) breaks vectorized kernel execution; uniform INT4 is faster |
| Grid search over scaling factors | Auto-tuning (e.g., TVM, Halide schedule search) | Search over a small parameter space (per-channel scales) to minimize a loss metric |
| Calibration-based optimization | SLA-based traffic shaping | Use observed statistics (activation magnitude distribution) to inform static decisions |

---

## Problem

**What gap does this paper address?**

GPTQ achieves good INT4 accuracy via expensive Hessian-guided weight compensation, but it has practical limitations:
1. **Speed**: Hessian inversion is O(d³) per layer — quantization takes hours for large models
2. **Hardware efficiency**: GPTQ sometimes keeps a small fraction of weights in FP16 (mixed precision) for better accuracy — this breaks INT4 kernel optimizations
3. **Root cause ignored**: GPTQ compensates for quantization error after the fact; it doesn't ask *why* some weights are harder to quantize

AWQ asks the question: **which weights actually matter, and why?**

**Why is this hard?**

You can't simply keep the most important weights in FP16 and quantize the rest — hardware INT4 kernels require uniform data layout. The challenge is to preserve accuracy for salient weights *while still quantizing everything to INT4*.

---

## Key Ideas

### Idea 1: Identifying Salient Weights via Activation Magnitude

AWQ's key empirical observation: **weights connected to input channels with large activation magnitudes are disproportionately important**.

Quantization error for a linear layer `Y = WX`:
```
ΔY = ΔW · X

For channel j: ΔY_j = Δw_j · x_j

If x_j is large (large activation), even small Δw_j (quantization rounding)
causes large ΔY_j — the error is amplified by the input magnitude.
```

By profiling activation magnitudes on a calibration set, AWQ identifies which input channels have large `|x_j|` — these are the salient channels whose corresponding weight columns must be quantized more carefully.

### Idea 2: Per-Channel Scaling Before Quantization

Instead of keeping salient weights in FP16, AWQ multiplies each salient weight column by a scale factor `s_j > 1` (expanding its range) and divides the corresponding input activation by `s_j` (preserving the product `W·X`):

```
Y = W X = (W · diag(s)) · (diag(s)⁻¹ · X)

Quantize: Ŵ = quant(W · diag(s))

At inference: Y ≈ Ŵ · (diag(s)⁻¹ · X)
```

By scaling up the salient weight columns before quantization, the rounding error `ΔŴ` is a smaller *fraction* of the (larger) scaled weight values. This effectively allocates more quantization precision to channels that matter more.

```
Without scaling:
  w_j = 0.02  →  quant(0.02) = 0.019  →  Δw_j = 0.001
  x_j = 100   →  ΔY_j = 0.001 × 100 = 0.1   ← large error

With scaling (s_j = 4):
  w_j × s_j = 0.08  →  quant(0.08) = 0.080  →  Δw_j = 0.000
  x_j / s_j = 25    →  ΔY_j ≈ 0             ← small error
```

The key: the scale factors `s_j` are absorbed into the quantized weights before deployment — no FP16 channels remain, everything is INT4.

### Idea 3: Grid Search Over Scale Factors

AWQ searches for optimal per-channel scales `s_j` by minimizing quantization error on a small calibration set:

```
s* = argmin_s || quant(W · diag(s)) · X_calib - W · X_calib ||²

Search space: s_j ∈ {α^k : k ∈ 0..N} where α is based on activation magnitude
```

This is a simple 1D grid search per channel (not gradient-based, not Hessian-based) — it runs in minutes even for 70B models.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Hardware-friendly pure INT4 output | Less fine-grained error control than GPTQ |
| Fast quantization (minutes, not hours) | Grid search may miss globally optimal scales |
| Edge/mobile deployment (no Hessian) | Calibration dataset still required |
| Better accuracy than RTN at INT4 | Slightly below GPTQ on some benchmarks (model-dependent) |

**Design decisions worth questioning:**

- The "1% salient channels" heuristic: what if a model's architecture doesn't exhibit this sparsity pattern? Newer architectures may not have the same activation magnitude distribution as GPT-style models.
- Scaling factors are per-channel (not per-weight). A finer-grained approach (per-weight scaling) could help at lower bit-widths but would complicate the INT4 kernel.
- AWQ focuses on weight quantization only. Combined KV cache + weight quantization (both at INT4/INT8) is more complex and not addressed here.

---

## Connections

**Builds on:**
- [GPTQ](gptq.md) — establishes layer-wise weight quantization as the standard approach; AWQ proposes a simpler, faster alternative
- **SmoothQuant** — precursor idea of input-weight migration via scaling (applied to activation quantization, not weight quantization)

**Inspired / Followed by:**
- **QuIP# / AQLM** — vector quantization approaches that push below 4-bit further
- **MLC-LLM** — deploys AWQ-quantized models on mobile devices (iPhone, Android, WebGPU)
- **TinyChat** (MIT HAN Lab) — real-time LLM inference on edge GPUs using AWQ

**Production systems:**
- vLLM supports AWQ natively
- Ollama ships AWQ variants for popular models (Llama 3, Phi, Mistral)
- HuggingFace `transformers` library supports AutoAWQ
- Used by Apple for on-device LLM inference experiments

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Quantization time | Minutes | vs hours for GPTQ; no Hessian required |
| Inference speedup vs FP16 | ~3x | A100 with INT4 kernels |
| Speedup vs GPTQ | 1.45x | Edge GPU (RTX 4090); pure INT4 vs mixed precision |
| Accuracy vs GPTQ | Comparable or better | LLaMA-2-7B/13B/70B on MMLU and perplexity |
| Salient weight fraction | ~1% | Channels with top activation magnitudes |

---

## Questions & Open Problems

- [ ] Does the 1% salience rule hold for all model families? MoE models (Mixtral), multimodal models (LLaVA), and embedding-heavy architectures may have different activation distributions.
- [ ] Can AWQ scales be learned jointly with model fine-tuning (QLoRA-style) for better accuracy at extreme compression (2-bit)?
- [ ] How does AWQ interact with prefix caching? KV cache entries are computed from quantized weights — does quantization error compound over long cached prefixes?
- [ ] Is there a principled way to set the calibration dataset size? Too few samples may miss activation outliers in tail prompts.

---

## Reading Notes

The core insight — "identify the 1% of weights that matter via activation statistics, then scale before quantizing" — is elegant in its simplicity. It avoids the O(d³) Hessian computation of GPTQ while achieving similar or better results by attacking the root cause rather than compensating after the fact.

This is a recurring pattern in systems optimization: profile to find the hot path, optimize the hot path specifically. The same logic applies to database index selection, JIT compilation hot loops, and network traffic shaping. AWQ applies it to weight quantization.

For infrastructure engineers: AWQ is the right choice when deployment targets edge devices or when you need fast quantization turnaround (minutes vs hours). For maximum accuracy on datacenter GPUs, GPTQ's Hessian compensation can still win on some benchmarks. In practice, run both and pick based on your quality/latency budget.
