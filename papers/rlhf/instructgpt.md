# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**Authors**: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, et al.  
**Venue**: NeurIPS '22  
**Paper**: [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)  
**Code**: N/A (OpenAI internal; open-source implementations: trl, DeepSpeed-Chat)

---

## TL;DR

InstructGPT defines the canonical three-stage pipeline for aligning language models with human intent: (1) Supervised Fine-Tuning (SFT) on demonstration data, (2) training a Reward Model (RM) on human preference rankings, (3) optimizing the policy via Proximal Policy Optimization (PPO) against the reward model with a KL penalty to prevent divergence from the SFT baseline. The key infra challenge is the PPO stage, which requires **four models in GPU memory simultaneously** — actor, critic, reward model, and reference model — making it the most resource-intensive alignment method and the baseline that all subsequent work (DPO, GRPO) aims to simplify.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| 3-stage RLHF pipeline (SFT → RM → PPO) | ETL pipeline (Extract → Transform → Load) | Both are multi-stage data processing pipelines where each stage produces artifacts consumed by the next |
| KL penalty against reference model | Rate limiter / circuit breaker | Prevents the policy from drifting too far too fast; bounds the update magnitude per iteration |
| Reward model | Quality scoring service / anomaly detector | A stateless scorer invoked on every output; must be fast and reliable since it's in the hot path |
| 4-model PPO setup | Microservice architecture with tight coupling | Four interdependent services (actor, critic, reward, ref) that must coordinate within each training step |
| PPO rollout generation | Online feature computation | Generate fresh data each iteration rather than reading from a static dataset; the generation step dominates latency |

---

## Problem

**What gap does this paper address?**

Large language models trained on internet text are good at predicting next tokens but poor at following user instructions. They hallucinate, generate harmful content, and often ignore what the user actually asked. Scaling model size alone doesn't fix this — a 175B model is no more "aligned" than a 1.3B model.

**Why is this hard?**

- Human preferences are subjective and hard to formalize as a loss function
- Supervised fine-tuning on demonstration data has limited coverage — you can't demonstrate every possible instruction
- Naive RL (optimize reward directly) leads to **reward hacking**: the model finds adversarial outputs that score high on the reward model but are low quality to humans
- The KL penalty is necessary to prevent reward hacking but introduces a multi-model coordination problem that multiplies memory requirements

---

## Key Ideas

### Idea 1: Three-Stage Pipeline

```
Stage 1: SFT (Supervised Fine-Tuning)
  Input: (prompt, human-written response) pairs
  Output: SFT model — the starting point for RL
  Infra: standard fine-tuning, nothing special

Stage 2: Reward Model Training
  Input: (prompt, response_A, response_B, human_preference) tuples
  Output: RM that outputs scalar reward for (prompt, response) pairs
  Infra: pairwise forward passes, contrastive loss on reward differences

Stage 3: PPO Optimization
  Input: prompts from training distribution
  Output: aligned policy model
  Infra: 4 models in memory, online generation, multi-step update loop
```

### Idea 2: The 4-Model PPO Architecture

The PPO stage is the infra bottleneck. Every training step requires:

| Model | Role | Trainable | Memory |
|-------|------|-----------|--------|
| **Actor** (policy) | Generates responses, receives gradient updates | Yes | Full training memory (params + grads + optimizer state) |
| **Critic** (value function) | Estimates expected return at each token; used for advantage computation | Yes | Full training memory |
| **Reward Model** | Scores generated responses | No (frozen) | Inference-only memory (params only) |
| **Reference Model** | Computes KL penalty baseline (the SFT checkpoint) | No (frozen) | Inference-only memory (params only) |

For a 7B model, this means ~4× the memory of standard fine-tuning. For 175B, this is only feasible with aggressive model parallelism and offloading.

### Idea 3: PPO Training Loop

```
for each batch of prompts:
  1. Actor.generate(prompts)           → responses    [autoregressive, slow]
  2. Reward.forward(prompts, responses) → rewards      [single forward pass]
  3. Ref.forward(prompts, responses)    → ref_logprobs [single forward pass]
  4. Actor.forward(prompts, responses)  → actor_logprobs
  5. Critic.forward(prompts, responses) → values
  6. Compute advantages using GAE(rewards, values, ref_logprobs)
  7. PPO update: actor.backward() + critic.backward()
     with KL penalty = β * (actor_logprobs - ref_logprobs)
```

Step 1 (generation) dominates wall-clock time. Steps 2–5 are forward passes that can be batched. Step 7 is the only backward pass.

### Idea 4: KL Penalty as Regularizer

The objective is: `maximize E[reward(x, y)] - β * KL(π || π_ref)`

Without KL penalty, the policy would exploit weaknesses in the reward model. The KL term ensures the policy stays close to the SFT model, trading reward maximization for output distribution stability.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Alignment quality (human-rated) | 4× GPU memory vs standard fine-tuning |
| Online exploration (generates fresh data) | Generation throughput bottleneck |
| Stability (KL penalty) | Slower convergence than unconstrained RL |
| Generality (works for any reward signal) | Complexity: 4 models, multiple stages |

**Design decisions I'd question:**

- Why a separate critic model instead of using the reward model as the value function? The critic adds memory and complexity. GRPO later shows you can estimate advantages without a critic entirely.
- The PPO loop requires synchronous generation → scoring → training. Could the generation be pipelined or made asynchronous?

---

## Connections

**Builds on:**
- PPO (Schulman et al., 2017) — the RL algorithm
- GPT-3 (Brown et al., 2020) — the base model being aligned

**Inspired / Followed by:**
- [DPO](dpo.md) — eliminates reward model and PPO loop entirely
- [GRPO](grpo.md) — eliminates critic model; uses group sampling for advantage estimation
- DeepSpeed-Chat — first open-source implementation of the full 3-stage pipeline
- trl (HuggingFace) — popular open-source RLHF library
- veRL — production-grade RLHF framework with hybrid inference-training engine

**Production systems using these ideas:**
- ChatGPT (OpenAI) — the original application
- Claude (Anthropic) — RLHF + Constitutional AI
- Gemini (Google) — RLHF-based alignment

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Labeler preference (InstructGPT 1.3B vs GPT-3 175B) | InstructGPT wins 85% | A 100× smaller aligned model beats the unaligned giant |
| Training data (SFT) | ~13K demonstrations | Small dataset, high-quality human demonstrations |
| Training data (RM) | ~33K comparisons | Human preference rankings |
| Training data (PPO) | ~31K prompts | No human labels needed — reward model provides the signal |
| Model size (final) | 175B parameters | 4× this in GPU memory during PPO training |

---

## Questions & Open Problems

- [ ] The 4-model setup is the dominant cost. Can we reduce this without sacrificing alignment quality? (DPO and GRPO answer this differently)
- [ ] Generation throughput is the bottleneck — how to pipeline generation and training efficiently? (veRL's hybrid engine addresses this)
- [ ] The reward model is a single point of failure — reward hacking is well-documented. How robust is the KL penalty in practice at scale?
- [ ] Online vs offline: PPO generates fresh data each iteration, but is this necessary? DPO suggests offline training on static preferences works comparably.

---

## Reading Notes

- The paper is more about the methodology and results than the systems engineering. The infra implications (4-model memory, generation bottleneck) are not discussed in the paper but are the dominant concern when implementing RLHF at scale.
- The 1.3B InstructGPT beating 175B GPT-3 is the headline result, but the infra story is: how do you run PPO at 175B scale with 4 copies of the model? The answer at the time was: with great difficulty, and only OpenAI had the resources.
- This paper is the "why" for every subsequent RLHF simplification (DPO, GRPO). Understanding the 4-model PPO baseline is essential context for appreciating what those papers eliminate.
