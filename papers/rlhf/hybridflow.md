# HybridFlow: A Flexible and Efficient RLHF Framework

**Authors**: Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, Chuan Wu  
**Venue**: EuroSys '25 (ByteDance Seed + HKU)  
**Paper**: [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)  
**Code**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)

---

## TL;DR

RLHF has **4 heterogeneous models** (actor, critic, reward, reference) executing **two very different workloads** (training backward passes and autoregressive generation) on the same GPU cluster. Existing frameworks either monolithically collocate everything (DeepSpeed-Chat — rigid parallelism, can't use vLLM for generation) or flatten everything into one controller (OpenRLHF — loses fine-grained scheduling). HybridFlow introduces a **hybrid single-controller + multi-controller programming model**: a single Ray driver orchestrates the dataflow (rollout → log-probs → reward → advantage → update), while each role runs its own multi-controller SPMD workers (FSDP for training, vLLM for generation). The key system contribution is the **3D-HybridEngine**: the actor's training-mode sharding (e.g., TP=8, FSDP=16) is *resharded in place* to generation-mode sharding (e.g., TP=2, DP=64) without duplicating parameters. This single idea makes the framework now known as **veRL** — the open-source RLHF infrastructure behind DeepSeek-V3-style post-training.

---

## Problem

**What gap does this paper address?**

RLHF is fundamentally a *dataflow orchestration problem*, not a single training loop. A PPO step looks like:

```
1. rollout:       actor.generate(prompts)         → responses (autoregressive, memory-BW bound)
2. score:         reward_model.forward(responses) → rewards   (single forward pass)
3. reference:     ref_model.forward(responses)    → ref_logp  (frozen, single forward)
4. value:         critic.forward(responses)       → values    (needs training later)
5. advantage:     GAE(rewards, values)            → advantages (CPU compute)
6. policy update: actor.train(advantages)         → new weights (backward pass)
7. critic update: critic.train(returns)           → new weights (backward pass)
```

Each stage has a different optimal execution profile:
- **Generation** wants vLLM/SGLang with small TP, large KV cache, paged attention
- **Training (backward)** wants FSDP or Megatron with TP+PP+DP tuned for gradients and optimizer states
- **Frozen forward passes** (ref/reward) want minimal parallelism, no optimizer state

Existing frameworks forced a single parallelism strategy on all four models. DeepSpeed-Chat uses ZeRO-3 everywhere — fine for training, but painfully slow for generation (ZeRO-3's per-layer all-gather kills decode latency). OpenRLHF uses Ray actors but with rigid placement and no parameter resharding between train/generate modes — meaning the actor must keep two copies of its weights in memory (one sharded for training, one replicated for generation).

**Why is this hard?**

Three specific tensions:

1. **Programming model tension**: high-level RLHF algorithms change rapidly (PPO → DPO → GRPO → RLOO). Users want to express the dataflow in a few dozen lines of Python, like a single-controller notebook. But the underlying compute (billions of parameters on thousands of GPUs) requires SPMD multi-controller execution. Purely single-controller = slow; purely multi-controller = unreadable.

2. **Parameter resharding tension**: the actor must be ready to switch between "training" (sharded for FSDP/Megatron) and "generation" (resharded for vLLM's TP). Naive approach: keep both copies → 2× memory. Smart approach: reshard in place with collective communication → complex.

3. **Resource placement tension**: 4 models with different sizes and different temporal usage patterns. Critic is idle during rollout; reward model is idle during training. Collocate them to share GPUs? Then you need to swap one out of memory when the other runs. Disaggregate them? Then you waste GPUs.

---

## Key Ideas

### Idea 1: Hybrid Single-Controller + Multi-Controller Programming Model

The high-level RLHF dataflow runs on a **single Ray driver** — one Python process that reads like pseudocode. Each compute-heavy operation dispatches to **a multi-controller SPMD group** (one process per GPU, running the same training/inference code).

```python
# Single-controller side (the driver, runs on one process):
for step in range(num_steps):
    batch = sample_batch()
    responses = actor_group.generate(batch)       # dispatches to actor SPMD workers
    rewards   = reward_group.compute(responses)   # dispatches to reward SPMD workers
    ref_logp  = ref_group.compute(responses)      # dispatches to ref SPMD workers
    values    = critic_group.compute(responses)   # dispatches to critic SPMD workers
    advantages = compute_gae(rewards, values, ref_logp)
    actor_group.update(batch, advantages)         # dispatches backward to actor SPMD
    critic_group.update(batch, returns)           # dispatches backward to critic SPMD
```

Each `*_group` is a Ray actor collection — 8, 16, or 128 SPMD workers cooperating via NCCL. The driver never touches tensors; it only passes handles (futures). This gives:
- **Readable dataflow**: the PPO loop is 10 lines, not buried in distributed plumbing
- **Fine-grained scheduling**: the driver controls ordering (e.g., overlap critic.update with actor.generate)
- **SPMD efficiency inside each stage**: each role uses the parallelism strategy optimal for its workload

```
Layer architecture:

    [ Single Ray Driver ]   ← dataflow orchestration, Python
           │
           │  (RPC: "generate", "train", "forward")
           ▼
  ┌────────┼─────────┬──────────┐
  │        │         │          │
 Actor   Critic   Reward    Reference
 group   group    group     group       ← multi-controller SPMD groups
 (TP+FSDP) (FSDP) (TP)     (TP)
  │        │         │          │
 8 GPUs   8 GPUs   4 GPUs    4 GPUs     ← each group = N processes, NCCL collectives
```

This is the same pattern as a distributed database: a coordinator issues logical plans; storage nodes execute physical operations in parallel. The LLM concept doesn't need a new name — it's just client/server separation at the training-job level.

### Idea 2: 3D-HybridEngine — Parameter Resharding Between Train and Generate

The actor model is used two ways per step:
- **Generation** (rollout): runs through vLLM for max throughput. Wants small TP (2–4), large KV cache, no optimizer state.
- **Training** (policy update): runs through FSDP/Megatron. Wants TP+PP+DP tuned for gradient sync, full optimizer state in memory.

Naively, you'd keep two copies of the actor weights — one sharded for training, one sharded for generation — doubling memory. 3D-HybridEngine eliminates the second copy by **resharding the training-mode weights into generation-mode layout** via a single all-gather + re-scatter, done once at each rollout boundary.

```
Train mode (FSDP × TP):             Generate mode (vLLM TP):
                                    
GPU 0:  [layer shard A₀, optim A₀]  →   [full layer A, KV cache]
GPU 1:  [layer shard A₁, optim A₁]  →   [full layer B, KV cache]
GPU 2:  [layer shard B₀, optim B₀]  →   [full layer C, KV cache]
GPU 3:  [layer shard B₁, optim B₁]  →   [full layer D, KV cache]
  ...                                       ...

Reshard step = NCCL all-gather + redistribute
Memory cost = weight-layout buffer (O(layer_size)), not 2× full weights
```

The "3D" refers to the three parallelism dimensions involved (TP, PP, DP), and the engine handles resharding across all three. The practical win: the actor can use TP=8 during training (fits 70B gradients) and TP=2 during generation (bigger batch, lower latency) *without paying for two separate weight copies*.

This is the single most important system contribution of the paper. Every subsequent RLHF framework (OpenRLHF, slime, NeMo-Aligner) now implements some version of it.

### Idea 3: Resource Pool Abstraction — Collocate vs Disaggregate Per Role

HybridFlow lets users declaratively assign each model to a **resource pool** (a set of GPUs). Multiple models can share a pool (collocation) or have their own (disaggregation):

```
Example layout (32 GPUs, 70B actor + 70B critic + 7B reward + 70B ref):

Pool A (16 GPUs): actor + reference    ← share: ref is frozen, actor ∉ generation when training
Pool B (8 GPUs):  critic               ← dedicated: critic training conflicts with actor update
Pool C (8 GPUs):  reward               ← dedicated: small model, always-on inference

Or alternative: all on one 32-GPU pool with time-division
```

The framework handles:
- **Parameter offload**: when a collocated model is idle, its weights can be moved to CPU to free GPU memory
- **NCCL group management**: each resource pool has its own communicator; avoids cross-pool interference
- **Failure isolation**: per-pool restart if one role's workers crash

The tradeoff matrix is explicit: collocation saves GPUs but adds swap latency; disaggregation wastes GPUs but simplifies scheduling. HybridFlow exposes it as a config knob rather than a hardcoded policy.

### Idea 4: Auto-Mapping Algorithm

Given cluster size, model sizes, and parallelism preferences, HybridFlow's auto-mapper chooses:
1. Which models to collocate on shared pools
2. TP/PP/DP degrees per model per pool
3. Whether to offload idle models to CPU

It's a simple search over a small discrete config space (dozens of candidates), scored by simulated memory + latency. Not as fancy as Alpa's ILP, but sufficient for the RLHF topology where the search space is small.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Generation throughput (uses vLLM with optimal TP) | Resharding overhead at each rollout boundary (~1–3% wall time) |
| Memory efficiency (no duplicate weight copies) | Engine complexity: must support both FSDP and vLLM layouts |
| Dataflow readability (single-controller driver) | Ray/RPC overhead for small operations (mitigated by only dispatching batch-level ops) |
| Flexibility (swap PPO ↔ GRPO ↔ DPO with 50 lines) | Steeper learning curve than DeepSpeed-Chat (must understand Ray + hybrid controller model) |
| Collocation (share GPUs across models) | CPU offload latency when swapping models in/out of GPU memory |

**Design decisions I'd question:**

- The resharding happens at every rollout boundary. For small rollouts (e.g., 8-token generations in continuous training), this overhead dominates. Is there a cheaper "partial reshard" when only some layers changed?
- Ray is the default orchestration layer. Ray has its own overhead (hundreds of µs per RPC). For very small RLHF jobs (< 8 GPUs), is the single-controller even worth it?
- The auto-mapper doesn't model communication costs across pools (cross-pool NCCL is slower than intra-pool). For very large clusters, naive placement produces suboptimal network paths.
- Collocation via CPU offload assumes PCIe bandwidth is sufficient. On H100 nodes with NVLink, this is fine; on older A100 nodes, offload latency can spike unpredictably.

---

## Connections

**Builds on:**
- **Ray** (RISELab) — the multi-process actor framework; HybridFlow is essentially "Ray for RLHF" with smart resource management on top
- **FSDP** (PyTorch) — used as the training backend for each role
- **vLLM** / **SGLang** — used as the generation backend; the resharding target layout
- **DeepSpeed-Chat** — the monolithic baseline HybridFlow replaces
- **Alpa** (OSDI '22) — auto-parallelism inspiration, though HybridFlow's search is simpler

**Inspired / Followed by:**
- **OpenRLHF** — earlier framework with similar goals; HybridFlow's parameter resharding improves on OpenRLHF's two-copy approach
- **NeMo-Aligner** (NVIDIA) — adopts similar hybrid controller design
- **slime** (ByteDance, newer) — internal evolution of veRL for RL-scaling experiments
- Most production RLHF infra teams (Anthropic, OpenAI, Mistral) appear to have built something structurally similar internally

**Production systems using these ideas:**
- **veRL** (the public release) — used to train reasoning models across ByteDance Seed
- DeepSeek-V3 / R1 post-training (not veRL directly, but similar architecture by report)
- Most open-source reasoning model replications (Open-R1, etc.) now use veRL

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Throughput vs DeepSpeed-Chat | up to 20.57× | PPO on Llama-70B across various cluster sizes |
| Throughput vs OpenRLHF | up to 2.14× | Same PPO workload |
| Memory savings from 3D-HybridEngine | ~50% of actor footprint | vs keeping two weight copies for train + generate |
| Resharding overhead | ~1–3% of step time | Amortized across long rollouts |
| Supported models | 7B–70B+ | Tested up to Llama-2-70B |
| Reward/KL/advantage compute share | <10% | Generation + backward dominate |

---

## Questions & Open Problems

- [ ] Resharding does one all-gather per rollout boundary. For RL with very frequent policy updates (e.g., continuous rollout), is streaming reshard (only updated layers) feasible?
- [ ] The 4-model RLHF baseline is increasingly replaced by 2-model setups (GRPO). Does the resource-pool abstraction still deliver value at 2 models, or is vanilla Ray + FSDP + vLLM enough?
- [ ] The paper focuses on homogeneous clusters (all H100 or all A100). For heterogeneous clusters (some H100, some A100), how should auto-mapping assign roles?
- [ ] Ray's single-controller has a theoretical central-bottleneck problem. At 10K+ GPU clusters, does the driver become a scheduling bottleneck, or is RLHF's low RPC rate safe?
- [ ] How does HybridFlow handle async RL (generation and training run concurrently on different stale policies)? The paper focuses on synchronous PPO/GRPO — async adds a whole new consistency dimension.

---

## Reading Notes

- The hybrid controller split is the same idea as Spark's driver + executors, or a distributed database's coordinator + workers. What's surprising is how long it took the RLHF community to adopt it — DeepSpeed-Chat (the prior standard) is structurally an MPI program with everything flat. HybridFlow is the first RLHF framework to treat the high-level dataflow as a first-class scheduling problem.
- The 3D-HybridEngine is the killer contribution. Without it, veRL would be just "another Ray wrapper." With it, you can train a 70B actor on 16 GPUs that would otherwise need 32 — the 2× memory saving is the difference between "feasible" and "infeasible" for many labs.
- The decision to use vLLM as the generation backend is non-obvious in hindsight but was controversial at the time. Other frameworks used HuggingFace `generate()` (slow) or built their own inference engine (maintenance burden). Leveraging vLLM meant inheriting PagedAttention + continuous batching for free.
- For staff-level interview prep: this paper is a canonical example of **dataflow-aware distributed systems design**. The questions to prepare for: "how would you design an RLHF framework from scratch?", "how do you schedule heterogeneous workloads on shared GPUs?", "how do you reshard model parameters between parallelism strategies?" — HybridFlow has the production-validated answer to each.
- PR opportunity lens: most of veRL's current issues (e.g., #5872, #144, #5750) touch exactly these design seams — hybrid engine edge cases, resource pool scheduling under heterogeneous loads, and resharding under new parallelism strategies. Understanding the paper is prerequisite; the PRs are where the judgment calls live.
