# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)

**Authors**: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo  
**Venue**: arXiv 2024 (GRPO later used at scale in DeepSeek-R1)  
**Paper**: [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)  
**Code**: [https://github.com/deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)

---

## TL;DR

GRPO (Group Relative Policy Optimization) eliminates the critic/value model from PPO by estimating advantages through **group sampling**: for each prompt, generate G outputs (e.g., G=64), score them with a reward function, and normalize rewards within the group (z-score) to compute advantages. This removes one of the two trainable models in PPO (the critic), leaving only the actor + frozen reference model — the same memory footprint as DPO but with **online generation**, enabling exploration beyond the preference dataset. The catch: generating G samples per prompt makes generation throughput the dominant bottleneck, which is exactly the problem veRL's hybrid inference-training engine addresses.

---

## Problem

**What gap does this paper address?**

PPO requires a critic (value) model to estimate advantages. The critic:
- Doubles the trainable parameter count (actor + critic)
- Must be trained jointly with the actor, adding complexity
- Introduces a second source of approximation error (critic estimation error propagates to advantage estimates, destabilizing actor training)

DPO eliminates both the critic and reward model, but sacrifices online generation — it's offline, training on a fixed preference dataset. For tasks like mathematical reasoning where exploration is essential (the model must discover novel solution paths), offline methods hit a ceiling.

**Why is this hard?**

Without a critic, you need another way to estimate "how good is this output relative to what we expected?" The naive approach (use raw reward as advantage) has high variance because absolute reward values vary widely across prompts. A prompt with all-correct answers and a prompt with all-wrong answers would produce very different gradient magnitudes, making training unstable.

---

## Key Ideas

### Idea 1: Group Relative Policy Optimization

For each prompt x, generate G outputs {y₁, y₂, ..., y_G}. Score each with reward function r(x, yᵢ). Compute advantages:

```
Aᵢ = (r(x, yᵢ) - mean(r(x, y₁..G))) / std(r(x, y₁..G))
```

This is z-score normalization within the group. The group mean serves as the baseline that the critic would have estimated in PPO.

```
PPO:                              GRPO:
                                  For prompt x, generate G=64 outputs:
                                  y₁ → r₁ = 0.0 (wrong)
critic(x, y) → baseline          y₂ → r₂ = 1.0 (correct)
advantage = reward - baseline     y₃ → r₃ = 0.0 (wrong)
                                  ...
                                  y₆₄ → r₆₄ = 1.0 (correct)
                                  
                                  mean = 0.45, std = 0.50
                                  A₁ = (0.0 - 0.45)/0.50 = -0.90
                                  A₂ = (1.0 - 0.45)/0.50 = +1.10
```

### Idea 2: The RLHF Simplification Progression

```
InstructGPT (PPO):   4 models — actor✏️, critic✏️, reward❄️, reference❄️    [online]
DPO:                 2 models — policy✏️, reference❄️                        [offline]
GRPO:                2 models — actor✏️, reference❄️                         [online]

✏️ = trainable, ❄️ = frozen
```

GRPO combines the memory efficiency of DPO (2 models) with the online exploration of PPO (fresh generation each iteration).

### Idea 3: Generation as the Bottleneck

With G=64 and a batch of 512 prompts, each GRPO step requires **32,768 generations**. This is the dominant cost:

| Operation | Compute Profile | Time Share |
|-----------|----------------|------------|
| Generate G×B outputs | Autoregressive, memory-bound | ~70-80% |
| Score with reward function | Single forward pass (or rule-based) | ~5% |
| Compute ref log-probs | Forward pass through reference model | ~5-10% |
| Policy gradient update | Backward pass through actor | ~10-15% |

This is why veRL's hybrid engine — using vLLM for generation and switching to training mode for updates — is essential for making GRPO practical.

### Idea 4: Rule-Based Rewards for Math

For mathematical reasoning, the reward can be binary and rule-based:
```
r(x, y) = 1.0 if final_answer(y) == ground_truth(x) else 0.0
```

No learned reward model needed. This further simplifies the system: the only neural networks are the actor and reference model. The reward function is a deterministic string comparison.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Memory efficiency (no critic) | Higher generation throughput demand (G samples per prompt) |
| Online exploration (unlike DPO) | Generation dominates training time (70-80%) |
| Simplicity (no critic training, no GAE) | Variance in advantage estimates (group stats are noisier than a trained critic) |
| Works with rule-based rewards | Not directly applicable when rewards are subjective (need a learned RM) |

**Design decisions I'd question:**

- G=64 is a hyperparameter that trades compute for variance reduction. What's the minimum G that works? (G=16 is reported as viable but noisier)
- The z-score normalization assumes roughly Gaussian reward distributions within each group. For binary rewards (math), the distribution is bimodal — does this hurt?
- Group-level normalization means prompts where the model gets everything right (or everything wrong) contribute zero gradient. Is this wasted compute, or a feature?

---

## Connections

**Builds on:**
- [InstructGPT](instructgpt.md) — the PPO baseline that GRPO simplifies
- [DPO](dpo.md) — showed 2-model RLHF is viable, but only offline
- REINFORCE — GRPO is essentially REINFORCE with a group-relative baseline

**Inspired / Followed by:**
- DeepSeek-R1 — scaled GRPO to very large models for reasoning
- DeepSeek-R1-Zero — showed GRPO alone (without SFT) can produce chain-of-thought reasoning
- veRL — the primary open-source framework for efficient GRPO training
- OpenRLHF — another framework implementing GRPO

**Production systems using these ideas:**
- DeepSeek-R1 / DeepSeek-V3 (DeepSeek) — GRPO is the core alignment algorithm
- Open-source reproductions (Open-R1, etc.)

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| DeepSeekMath-7B accuracy (MATH) | 51.7% | Competitive with much larger models |
| Group size G | 64 | Default; 16 also works but noisier |
| Generations per step (G=64, B=512) | 32,768 | The dominant compute cost |
| Memory vs PPO | ~0.5× | No critic model; only actor + reference |
| Training time (generation share) | ~70-80% | Generation is the bottleneck, not backprop |

---

## Questions & Open Problems

- [ ] What's the optimal group size G for different tasks? Binary rewards (math) vs continuous rewards (general RLHF) may have different optima
- [ ] Can the generation bottleneck be mitigated by async generation (generate while training on previous batch)?
- [ ] Group normalization with binary rewards creates a bimodal advantage distribution. Is there a better normalization for discrete reward spaces?
- [ ] GRPO works well for math/code where rewards are verifiable. For open-ended generation (dialogue, creative writing), you still need a learned reward model — does GRPO compose well with learned RMs?
- [ ] DeepSeek-R1-Zero showed GRPO can produce chain-of-thought without SFT. What are the conditions under which this works vs fails?

---

## Reading Notes

- GRPO is conceptually simple: it's REINFORCE with a group-mean baseline. The innovation is recognizing that this is sufficient — you don't need a learned critic, and the variance from group statistics is manageable.
- The 4 → 2 model progression (InstructGPT → DPO / GRPO) mirrors a common systems pattern: start with a general solution (PPO), identify which components are load-bearing, eliminate the rest.
- The generation step is a classic fan-out pattern: each prompt fans out to G=64 samples, creating a G× amplification in compute. This shifts the infra challenge from "how to fit 4 models in memory" to "how to generate 32K samples efficiently per step" — which is why veRL integrates vLLM as a generation backend.
- DeepSeek-R1's success with GRPO at scale validated this approach as production-ready. The open-source release of DeepSeek-R1 weights made GRPO the most replicated alignment method in early 2025.
