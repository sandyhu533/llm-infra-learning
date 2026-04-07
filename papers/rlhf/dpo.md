# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn  
**Venue**: NeurIPS '23  
**Paper**: [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)  
**Code**: [https://github.com/eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)

---

## TL;DR

DPO eliminates the reward model and PPO loop from RLHF entirely. It shows that the optimal policy under the standard RLHF objective (reward maximization + KL penalty) has a closed-form solution, allowing the reward to be expressed as a function of the policy and reference model log-probabilities. This transforms RLHF from a complex RL problem into a simple classification-style loss on preference pairs. The infra impact: **4 models → 2 models** (policy + frozen reference), no online generation during training, and the training loop looks like standard supervised fine-tuning. A 2× memory reduction and dramatic simplification of the training pipeline.

---

## Problem

**What gap does this paper address?**

The standard RLHF pipeline ([InstructGPT](instructgpt.md)) requires:
1. Training a separate reward model
2. Running PPO with 4 models in memory (actor, critic, reward, reference)
3. Online generation of rollouts during training
4. Careful tuning of RL hyperparameters (clipping, GAE lambda, KL coefficient)

This is expensive, complex, and unstable. PPO training is notoriously sensitive to hyperparameters, and the 4-model setup makes it accessible only to large-resource labs.

**Why is this hard?**

The naive alternative — just fine-tune on chosen responses and ignore rejected ones — doesn't work because the model doesn't learn *why* responses are preferred. The loss must encode the preference relationship between chosen and rejected outputs relative to a baseline (the reference model).

---

## Key Ideas

### Idea 1: Closed-Form Optimal Policy

The standard RLHF objective is: `max_π E[r(x,y)] - β * KL(π || π_ref)`

The optimal policy has the closed-form solution:
```
π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x,y) / β)
```

Rearranging for the reward:
```
r(x,y) = β * log(π(y|x) / π_ref(y|x)) + β * log(Z(x))
```

The partition function Z(x) cancels out when computing reward differences between two responses — which is exactly what the Bradley-Terry preference model needs.

### Idea 2: DPO Loss Function

The loss reduces to binary cross-entropy on log-probability margins:

```
L_DPO = -log σ(β * [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

where y_w = chosen (winner), y_l = rejected (loser)
```

This is a supervised loss. No RL, no generation, no value function.

### Idea 3: Training Pipeline Simplification

```
InstructGPT (PPO):                    DPO:
┌─────────────────────┐              ┌──────────────────┐
│ 4 models in memory   │              │ 2 models in memory│
│ Actor    (trainable) │              │ Policy (trainable)│
│ Critic   (trainable) │              │ Ref    (frozen)   │
│ Reward   (frozen)    │              └──────────────────┘
│ Reference(frozen)    │
└─────────────────────┘              Training loop:
                                     for (x, y_w, y_l) in dataset:
PPO loop per step:                     logp_w = policy(y_w|x)
1. generate rollouts                   logp_l = policy(y_l|x)
2. score with reward model             ref_w  = ref(y_w|x)
3. compute values with critic          ref_l  = ref(y_l|x)
4. compute advantages (GAE)            loss = -log σ(β * (Δpolicy - Δref))
5. PPO clipped update                  loss.backward()  # only policy
6. update critic
```

### Idea 4: Memory and Compute Profile

| Resource | PPO (InstructGPT) | DPO |
|----------|-------------------|-----|
| Models in memory | 4 (2 trainable + 2 frozen) | 2 (1 trainable + 1 frozen) |
| Per-step forward passes | 4 (actor, critic, reward, ref) | 4 (policy×2 + ref×2, for chosen and rejected) |
| Backward passes | 2 (actor + critic) | 1 (policy only) |
| Online generation | Yes (bottleneck) | No |
| Dataset | Prompts only (generate on-the-fly) | Fixed (prompt, chosen, rejected) triples |
| RL hyperparameters | Many (clip, GAE λ, value coeff, etc.) | One (β) |

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Memory efficiency (2× reduction) | No online exploration — can't improve beyond preference data distribution |
| Training simplicity (SFT-like loop) | Offline — preference data quality is the ceiling |
| Stability (no RL instabilities) | Less flexibility — can't incorporate arbitrary reward signals |
| Compute efficiency (no generation step) | Requires pre-collected preference data with (chosen, rejected) pairs |

**Design decisions I'd question:**

- DPO is offline — it trains on static preference data. Does the lack of online exploration limit alignment quality in practice? (Some empirical evidence says yes for complex reasoning tasks)
- The reference model is frozen and used at every step. For very large models, even the inference cost of the reference model is significant. Could you approximate it?
- β is the only hyperparameter, but it's critical. Too low → reward hacking (same as without KL). Too high → no learning. How sensitive is it in practice?

---

## Connections

**Builds on:**
- [InstructGPT](instructgpt.md) — the RLHF pipeline that DPO simplifies
- Bradley-Terry model — the preference model underlying the reward formulation

**Inspired / Followed by:**
- IPO (Identity Preference Optimization) — further simplification
- KTO (Kahneman-Tversky Optimization) — works with binary feedback instead of pairwise
- [GRPO](grpo.md) — takes a different path: keeps online generation but removes the critic
- ORPO, SimPO — other DPO variants with different loss formulations

**Production systems using these ideas:**
- Llama 2/3 (Meta) — used DPO in alignment pipeline
- Zephyr (HuggingFace) — DPO-trained model
- Most open-source alignment work uses DPO as the default method (simpler than PPO)

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Win rate vs PPO (summarization) | ~60% | DPO matches or exceeds PPO on TL;DR summarization |
| Win rate vs PPO (dialogue) | ~55% | Comparable on Anthropic HH dialogue task |
| GPU memory reduction | ~2× | 2 models vs 4 models |
| Training time reduction | ~3–5× | No generation step; the SFT-like loop is much faster per iteration |
| Hyperparameters | 1 (β) | vs ~6+ for PPO |

---

## Questions & Open Problems

- [ ] Offline vs online: DPO can't explore beyond the preference data distribution. For complex reasoning (math, code), does this ceiling matter? (GRPO's online approach may be better here)
- [ ] On-policy DPO variants (iterative DPO, online DPO) try to bridge this gap — do they preserve DPO's simplicity?
- [ ] The reference model must be loaded at every step. For 70B+ models, can you amortize or approximate reference log-probs without quality loss?
- [ ] DPO assumes Bradley-Terry preferences (pairwise). Human preferences are often intransitive or context-dependent — does this assumption break down at scale?

---

## Reading Notes

- DPO's main contribution is a mathematical insight, not a systems innovation. But the infra consequences are profound: it made RLHF accessible to anyone who can run supervised fine-tuning.
- The "Your Language Model is Secretly a Reward Model" subtitle is the key intuition: if you have a policy and a reference model, the log-prob ratio *is* the reward. You never needed a separate reward model.
- The shift from PPO to DPO mirrors the monolith-vs-microservices tradeoff: PPO's 4-model architecture is flexible but operationally complex; DPO collapses it into a simpler system at the cost of losing online exploration. The reference model acts like a read replica — a frozen snapshot used only for inference, never updated.
- In practice, DPO's simplicity made it the default alignment method for the open-source community (Llama 2, Zephyr, etc.). PPO remained dominant at labs with more compute (OpenAI, Anthropic) because online exploration matters for frontier models.
- The tension between DPO (offline, simple) and PPO/GRPO (online, complex) is the central divide in alignment infrastructure. Understanding both sides is essential.
