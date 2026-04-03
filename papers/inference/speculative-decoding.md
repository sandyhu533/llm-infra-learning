# Fast Inference from Transformers via Speculative Decoding

**Authors**: Yaniv Leviathan, Matan Kalman, Yossi Matias  
**Venue**: ICML '23  
**Paper**: [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)  
**Code**: [https://github.com/google-deepmind/speculative_decoding](https://github.com/google-deepmind/speculative_decoding)

---

## TL;DR

Autoregressive LLM decoding generates one token per forward pass — the GPU runs a massive model to produce a single scalar. For small batch sizes (latency-sensitive requests), this is memory-bandwidth bound: most GPU compute is wasted. Speculative decoding fixes this by using a small **draft model** to speculatively generate k tokens, then verifying all k tokens with a single forward pass of the large **target model**. If the draft was right (high acceptance rate), you get k tokens for the cost of ~1 pass. A rejection sampling scheme guarantees the output distribution is *identical* to running the target model alone — no quality loss. 2–3x decode speedup in practice.

---

## Problem

**What gap does this paper address?**

For a single request (batch size 1), autoregressive decode is *memory-bandwidth bound*, not compute bound:
- A 70B model has ~140GB of weights (FP16)
- Each forward pass reads all 140GB from HBM to compute one token
- On an A100 (2TB/s HBM bandwidth), this takes ~70ms per token
- The ~312 TFLOPS of tensor cores sit largely idle — the bottleneck is memory reads, not arithmetic

The GPU is a powerful compute engine being used as a memory reader. There's significant headroom to do *more compute per token* without increasing latency, if only we could generate multiple tokens per pass.

**Why is this hard?**

Autoregressive generation has a sequential dependency: token `t+1` depends on token `t`. You cannot parallelize generation directly. Any speculative approach must:
1. Generate speculative tokens without knowing the target model's output
2. Accept/reject them in a way that *provably* preserves the target model's distribution
3. Handle rejection without wasting the compute spent so far

The key theoretical challenge is the rejection criterion: naive accept/reject would change the output distribution unless done carefully.

---

## Key Ideas

### Idea 1: Speculative Generation with a Draft Model

Use a small, fast model (the **draft model**, typically 1/10 to 1/100 the size of the target) to speculatively generate k tokens:

```
State: prefix tokens x_1, ..., x_t (already committed)

Step 1 — Draft: run draft model autoregressively k times
  ã_{t+1} = draft_model(x_1..x_t)
  ã_{t+2} = draft_model(x_1..x_t, ã_{t+1})
  ...
  ã_{t+k} = draft_model(x_1..x_t, ã_{t+1}, ..., ã_{t+k-1})

Step 2 — Verify: run target model ONCE on the full extended sequence
  p(x_{t+1}..x_{t+k+1} | x_1..x_t, ã_{t+1}..ã_{t+k})
  ← one forward pass, all k+1 positions in parallel
```

The verification pass costs roughly the same as a single target model decode step (slightly more due to larger batch, but well within the compute headroom).

### Idea 2: Rejection Sampling with Guaranteed Correctness

The theoretical contribution: an acceptance criterion that makes the accepted output provably identical in distribution to running the target model alone.

```
For each speculative token ã_{t+i}:
  q(x) = draft_model probability for token x at position t+i
  p(x) = target_model probability for token x at position t+i

  Accept ã_{t+i} with probability min(1, p(ã_{t+i}) / q(ã_{t+i}))

  If rejected: sample the next token from a corrected distribution:
    p'(x) = normalize(max(0, p(x) - q(x)))
            ← "residual" probability not covered by draft
```

This scheme guarantees: for any acceptance pattern, the output distribution equals the target model's distribution. There's no accuracy-speed tradeoff — the *only* effect of the draft model is latency, never quality.

### Idea 3: Efficiency Analysis — When Speculative Decoding Wins

Let `α` = average acceptance rate (fraction of draft tokens accepted). Expected tokens generated per verification step:

```
E[tokens per step] = (1 - α^{k+1}) / (1 - α)   ← geometric series

k=5, α=0.8: E = (1 - 0.8^6) / (1 - 0.8) = (1 - 0.262) / 0.2 ≈ 3.7 tokens/step
k=5, α=0.6: E = (1 - 0.6^6) / (1 - 0.6) = (1 - 0.047) / 0.4 ≈ 2.4 tokens/step
```

Speedup = `E[tokens per step] × (cost_draft × k + cost_target)⁻¹ × cost_target`

Speculative decoding wins when:
- Draft model is much cheaper than target (10x+ size ratio)
- Acceptance rate is high (similar domain, good draft model)
- Batch size is small (target model is memory-BW bound, not compute bound)
- k is tuned to match acceptance rate and GPU utilization

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Decode latency (tokens/sec, single request) | Additional draft model memory footprint |
| GPU compute utilization | Complexity: two models, rejection sampling logic |
| Identical output distribution (zero quality loss) | Draft model must be co-located with target model |
| Latency-sensitive, small-batch serving | Less effective at large batch sizes (target is already compute-bound) |

**Design decisions worth questioning:**

- The draft model must be from the same model family (same tokenizer, same vocabulary) as the target. You can't use a Mistral-7B draft with a LLaMA-70B target without special handling.
- At large batch sizes (e.g., 32+), the target model is already compute-bound — speculative decoding doesn't help and adds overhead. It's primarily a small-batch optimization.
- The optimal k depends on acceptance rate, which depends on the input distribution. Static k is suboptimal; dynamic k selection adds complexity.
- Draft model must be kept in GPU memory alongside the target — this eats into the memory budget available for KV cache and batch size.

---

## Connections

**Builds on:**
- **Autoregressive LLM decoding** — the sequential bottleneck this paper breaks
- **Rejection sampling** (classic statistics) — the acceptance criterion is a classical rejection sampling technique applied to language model token distributions

**Inspired / Followed by:**
- **Medusa** (Cai et al., 2024) — replaces the draft model with lightweight prediction heads on the target model itself; no separate draft model needed
- **EAGLE** (Li et al., 2024) — uses a single-layer draft head with feature reuse from the target model; higher acceptance rates
- **SpecInfer** — extends speculative decoding to tree-structured speculation (multiple draft sequences in parallel)
- **Self-speculative decoding** — the target model generates its own draft using early exit

**Production systems:**
- vLLM supports speculative decoding natively
- TensorRT-LLM implements speculative decoding for latency-optimized deployments
- Google's TPU serving uses speculative decoding in Gemini production
- Anthropic uses speculative decoding in Claude serving

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Speedup (T5-XXL, XL as draft) | 2–3x | Token generation speed on TPU |
| Acceptance rate (typical) | 0.6–0.85 | Depends on domain alignment between draft and target |
| Draft model size ratio | ~1:10 to 1:100 | Draft must be significantly smaller to be beneficial |
| Quality loss | 0 | Guaranteed identical distribution to target model |
| Optimal k (speculation window) | 4–8 tokens | Diminishing returns beyond this for typical acceptance rates |

---

## Questions & Open Problems

- [ ] How does speculative decoding interact with PagedAttention? The draft model's KV cache and the target model's KV cache must both be managed — does this complicate block allocation?
- [ ] What's the right draft model selection strategy? Size ratio, vocabulary match, and domain alignment all matter — is there a principled way to select or train the optimal draft?
- [ ] For extremely long outputs (1000+ tokens), does acceptance rate stay stable or drift as the output distribution shifts?
- [ ] Can speculative decoding be combined with chunked prefill (Sarathi-Serve)? The speculation window creates a new source of head-of-line blocking if draft token generation takes variable time.

---

## Reading Notes

The core idea is directly borrowed from CPU microarchitecture: speculative execution. The GPU equivalent is: use a cheap "branch predictor" (draft model) to speculatively produce output, then verify in bulk using the expensive wide unit (target model).

The theoretical guarantee (identical output distribution) is what makes this deployable in production. Without it, you'd be trading quality for speed — unacceptable for a production LLM serving system. The rejection sampling construction is elegant: it's essentially saying "accept the draft when the target would have likely generated the same token anyway."

The practical implication: speculative decoding is most valuable for interactive, low-latency use cases (chat, code completion) where requests are individual and batch size is small. For throughput-optimized batch inference, the gains are minimal. In production, you should profile batch size distributions before deciding whether to deploy speculative decoding.
