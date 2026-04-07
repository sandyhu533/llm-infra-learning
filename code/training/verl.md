# veRL

**Repo**: [github.com/volcengine/verl](https://github.com/volcengine/verl)  
**Version**: v0.3.x (hybrid engine architecture stable; active development on new features)  
**Learning goal**: Understand how veRL orchestrates multi-model RLHF training — how it manages actor/reference/reward model placement, switches between inference and training on the same GPUs, and implements the GRPO/PPO rollout-to-update loop efficiently.  
**Prerequisites**: [InstructGPT](../../papers/rlhf/instructgpt.md), [GRPO](../../papers/rlhf/grpo.md), [Megatron-LM](../../papers/training/megatron-lm.md)

---

## Entry Point

`verl/trainer/ppo/ray_trainer.py` — `RayPPOTrainer.fit()`

The main training loop lives here. `RayPPOTrainer` orchestrates the entire RLHF pipeline: it holds references to actor, reference, reward, and critic workers (as Ray actors), and drives the generate → score → compute-advantages → update cycle. User code creates this trainer and calls `fit()`, which runs the training loop until completion.

---

## Key Files

### `verl/trainer/ppo/ray_trainer.py` — The orchestrator

**What to look for:**
- `RayPPOTrainer.__init__()`: accepts worker classes for actor, critic, reward, reference. Workers are Ray remote actors that wrap the actual models.
- `RayPPOTrainer.fit()`: the main training loop — iterates over data, calls workers in sequence
- `training_step()`: a single RLHF iteration — generate rollouts, compute rewards, compute advantages, update policy
- Resource group management: how workers are placed on GPUs and how resources are shared between generation and training

**Data flow**: prompts → actor.generate() → reward.compute() → ref.log_probs() → advantage computation → actor.update() → next iteration

**Key insight**: The orchestrator is deliberately thin — it doesn't hold any model weights or tensors. All heavy compute happens in workers. This separation allows flexible placement: workers can be on different GPU groups, or share GPUs with careful memory management.

---

### `verl/workers/actor/dp_actor.py` — Actor worker (data-parallel mode)

**What to look for:**
- `DataParallelPPOActor`: wraps the model and handles both generation and training
- `generate_sequences()`: switches model to inference mode, runs autoregressive generation. May use vLLM or HuggingFace generate
- `update_policy()`: switches to training mode, runs PPO/GRPO gradient update
- Memory management between inference and training: how KV cache allocations are freed before training buffers are allocated

**Data flow**: prompts → generate_sequences() → (sequences, log_probs) → update_policy(sequences, advantages) → updated weights

**Key insight**: The **hybrid engine** concept — the same GPU hardware runs both inference (generation) and training (gradient updates) within a single RLHF step. This avoids the memory waste of dedicating separate GPU pools to generation and training, but requires careful lifecycle management: vLLM's KV cache must be fully released before PyTorch training allocations begin.

---

### `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — vLLM-based generation

**What to look for:**
- `vLLMRollout`: wraps a vLLM engine for efficient batched generation
- `generate()`: handles the prompt → completion generation with vLLM's continuous batching
- Weight synchronization: how updated actor weights are loaded into the vLLM engine after each training step
- Engine lifecycle: how the vLLM engine is initialized, used for generation, and then released for training

**Data flow**: prompts + sampling_params → vLLM engine → generated sequences + per-token log-probs

**Key insight**: Using vLLM (or similar inference engine) for generation is critical because GRPO requires G samples per prompt. Naive `model.generate()` with PyTorch is 5-10× slower than vLLM's optimized inference with PagedAttention and continuous batching. But integrating vLLM into a training loop means managing two memory regimes (inference KV cache vs training gradients) on the same GPUs.

---

### `verl/trainer/ppo/core_algos.py` — PPO/GRPO loss computation

**What to look for:**
- `compute_grpo_advantages()` or equivalent: implements group-relative advantage normalization (z-score within each prompt's group of G samples)
- `compute_policy_loss()`: PPO clipped surrogate loss with KL penalty
- `compute_kl_penalty()`: KL divergence between current policy and reference model
- The actual algorithmic core is small — most lines are tensor manipulation

**Data flow**: (log_probs, ref_log_probs, rewards, group_info) → advantages → policy_gradient_loss → scalar loss

**Key insight**: The algorithmic code is less than 200 lines. The overwhelming majority of veRL's complexity is in orchestration (ray_trainer.py), memory management (hybrid engine), and communication (weight sync). This validates the GRPO paper's claim that the algorithm is simple — the engineering challenge is making it efficient.

---

### `verl/utils/model/update_model.py` — Weight synchronization

**What to look for:**
- How updated actor weights are propagated from training (PyTorch FSDP/DDP state) to inference (vLLM engine)
- NCCL-based weight transfer vs shared memory approaches
- Handling of sharded weights: if training uses FSDP (sharded params), the full weights must be gathered before loading into vLLM

**Data flow**: FSDP shards → all-gather → full weights → vLLM engine.load_weights()

**Key insight**: Weight synchronization is a hidden bottleneck. After each actor update, the new weights must be transferred to the vLLM engine. For a 7B model this is ~14GB of data; for 70B it's ~140GB. veRL uses NCCL for efficient GPU-to-GPU transfer, but this still adds non-trivial overhead per iteration.

---

## Step Walkthrough

Trace one GRPO training iteration:

```
RayPPOTrainer.fit()
  └─ for each batch of prompts:

       1. Generate rollouts
          actor_worker.generate_sequences(prompts, G=64)
            └─ vLLMRollout.generate()
                 ├─ load latest actor weights into vLLM engine
                 ├─ allocate KV cache blocks
                 ├─ generate G sequences per prompt (continuous batching)
                 └─ return sequences + per-token log_probs

       2. Compute rewards
          reward_worker.compute_rewards(sequences)
            └─ rule-based scoring (math) or reward_model.forward()
            └─ return per-sequence scalar rewards

       3. Compute reference log-probs
          ref_worker.forward(sequences)
            └─ frozen reference model forward pass
            └─ return per-token ref_log_probs

       4. Compute advantages (GRPO)
          core_algos.compute_grpo_advantages(rewards, group_size=G)
            ├─ group rewards by prompt
            ├─ advantage_i = (r_i - mean(group)) / std(group)
            └─ return per-sequence advantages

       5. Update actor
          actor_worker.update_policy(sequences, advantages, ref_log_probs)
            ├─ free vLLM KV cache memory
            ├─ switch to training mode
            ├─ for epoch in range(ppo_epochs):
            │    ├─ actor_log_probs = actor.forward(sequences)
            │    ├─ kl = actor_log_probs - ref_log_probs
            │    ├─ ratio = exp(actor_log_probs - old_log_probs)
            │    ├─ loss = -min(ratio * adv, clip(ratio) * adv) + β * kl
            │    └─ loss.backward(); optimizer.step()
            └─ sync updated weights to vLLM engine

       6. (Optional) Update reference model periodically
          ref_worker.load_weights(actor.state_dict())
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| GRPO group sampling (G outputs per prompt) | `vLLMRollout.generate()` in `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — generates G completions per prompt |
| Group-relative advantage (z-score) | `compute_grpo_advantages()` in `verl/trainer/ppo/core_algos.py` — normalizes rewards within each prompt group |
| KL penalty against reference model | `compute_kl_penalty()` in `verl/trainer/ppo/core_algos.py` — per-token KL divergence |
| PPO clipped surrogate loss | `compute_policy_loss()` in `verl/trainer/ppo/core_algos.py` — clipped ratio × advantage |
| Actor-critic decoupling | Worker classes in `verl/workers/` — separate Ray actors for each model role |
| Online rollout generation | `actor_worker.generate_sequences()` called each iteration in `ray_trainer.py` |

---

## What the Paper Doesn't Tell You

- **Hybrid engine memory lifecycle**: The biggest engineering challenge is switching between inference and training on the same GPUs. vLLM pre-allocates a large KV cache pool (often 60-80% of GPU memory). Before training can begin, this pool must be completely freed, training buffers allocated (gradients, optimizer state, activations), and after training, the KV cache pool re-allocated. This allocation/deallocation cycle adds overhead and risks fragmentation. veRL handles this with explicit memory management and `torch.cuda.empty_cache()` calls at transition points.

- **Weight sync overhead**: After each actor update, weights must be synchronized from the FSDP training state to the vLLM inference engine. For FSDP (ZeRO-3 style), parameters are sharded across GPUs — a full all-gather is needed to reconstruct complete weights before loading them into vLLM. For a 70B model, this means gathering and transferring ~140GB of weights, which can take several seconds even over NVLink.

- **Batch size asymmetry**: With G=64, a batch of 512 prompts produces 32K sequences for training. But the training batch size is typically much smaller (micro-batch of 4-8 sequences). This means the training phase iterates over the rollout data in many mini-batches, often for multiple epochs (ppo_epochs=2-4). The rollout data must be stored in CPU or GPU memory across these iterations.

- **Ray serialization overhead**: Ray remote actors serialize/deserialize tensors when transferring between workers. For large tensors (sequences, log-probs), this can be a bottleneck. veRL mitigates this by using shared memory for co-located workers and NCCL collectives for cross-node transfers, bypassing Ray's default serialization path.

- **vLLM engine lifecycle**: vLLM is designed as a long-running server, not as a library called within a training loop. veRL adapts vLLM by carefully managing its initialization, weight loading, and memory allocation to coexist with PyTorch training. This includes patching vLLM's memory profiling to account for training memory reservations.

- **Generation-training GPU utilization gap**: During generation, the GPU is memory-bandwidth-bound (autoregressive decoding). During training, it's compute-bound (matmul-heavy backward pass). Neither phase fully utilizes all GPU resources, but they can't overlap because they compete for the same memory. This fundamental inefficiency is the motivation for disaggregated RLHF architectures (separate generation and training clusters), which veRL also supports as an alternative to the hybrid engine.
