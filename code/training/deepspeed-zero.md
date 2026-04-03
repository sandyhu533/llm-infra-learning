# DeepSpeed ZeRO

**Repo**: [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)  
**Version**: v0.14.x (Stage 2/3 APIs stable; internal refactors ongoing)  
**Learning goal**: Understand how ZeRO Stage 2 and Stage 3 are implemented — how parameters are partitioned at init, how reduce-scatter and all-gather fire during the training step, and how Stage 3 manages the fetch/release lifecycle per layer.  
**Prerequisites**: [ZeRO](../../papers/training/zero-memory-optimization.md)

---

## Entry Point

`deepspeed/runtime/engine.py` — `DeepSpeedEngine.__init__()` and `DeepSpeedEngine.backward()`

`DeepSpeedEngine` wraps the user's `nn.Module` and optimizer, intercepting `forward()`, `backward()`, and `step()` to inject ZeRO logic. The ZeRO optimizer is instantiated inside `__init__` based on the `zero_optimization.stage` config value.

---

## Key Files

### `deepspeed/runtime/zero/partition_parameters.py` — Parameter partitioning at init

**What to look for:**
- `Init` context manager (used as `with deepspeed.zero.Init()`): intercepts `nn.Parameter` creation and immediately shards each param across ranks
- `ZeroParamStatus` enum: `NOT_AVAILABLE` (param is partitioned, not on this GPU), `INFLIGHT` (all-gather in progress), `AVAILABLE` (full param is present, safe to use in compute)
- `_partition_param()`: splits a parameter tensor into `world_size` equal chunks; this rank keeps only its own shard, frees the rest
- `param.ds_tensor`: the local shard stored on-device; `param.data` is zeroed out until an all-gather makes it available

**Data flow**: `nn.Parameter` created → `_partition_param()` → `param.data` = zero tensor, `param.ds_tensor` = local shard

**Key insight**: Stage 3 partitions parameters at module creation time, not at optimizer init. Every `nn.Parameter` in the model becomes a "ghost" tensor — only the shard is real; the full param is reconstituted on demand.

---

### `deepspeed/runtime/zero/stage_1_and_2.py` — Stage 2 optimizer

**What to look for:**
- `DeepSpeedZeroOptimizer.__init__()`: partitions gradient buckets across ranks; each rank is responsible for a contiguous slice of the parameter list
- `reduce_gradients()`: called during backward; accumulates local gradients and fires `reduce_scatter` when a bucket is full — each rank receives only the gradient slice it owns
- `DeepSpeedZeroOptimizer.step()`: runs the Adam update locally on owned gradient + optimizer state slices, then calls `all_gather` to broadcast updated weights back to all ranks
- `_release_ipg_buffers()`: frees the temporary gradient accumulation buffers after reduce-scatter completes

**Data flow (one training step):**
```
Backward pass
  → gradient hooks accumulate into buckets
  → bucket full → reduce_scatter(bucket) → each rank gets its gradient slice
  → all gradients processed

Optimizer step
  → each rank: Adam(own_params, own_grads, own_optimizer_state)
  → all_gather(updated weights) → all ranks have full updated params

Next forward: weights are already on all ranks (replicated)
```

**Key insight**: The reduce-scatter + all-gather pattern has identical total communication volume to standard all-reduce (both are `2Ψ` bytes), but the intermediate state after reduce-scatter is that each rank holds only 1/N of the full gradient — which is all it needs for its local optimizer step. Memory of gradients drops from `Ψ` to `Ψ/N`.

---

### `deepspeed/runtime/zero/stage3.py` — Stage 3 optimizer

**What to look for:**
- `DeepSpeedZeroOptimizer_Stage3._pre_forward_hook()` / `_post_forward_hook()`: registered as forward hooks on every `nn.Module`; fire before and after each layer's forward
  - `_pre_forward_hook`: calls `_fetch_sub_module()` — triggers all-gather to materialize the layer's params
  - `_post_forward_hook`: calls `_release_sub_module()` — frees the non-owned param copies, reverting to shards
- `_fetch_sub_module()`: issues an all-gather for all params in the module; transitions their `ZeroParamStatus` from `NOT_AVAILABLE` → `INFLIGHT` → `AVAILABLE`
- `_release_sub_module()`: frees `param.data`, sets status back to `NOT_AVAILABLE`
- `get_param_coordinator()`: the prefetch coordinator — starts all-gathering the *next* layer's params while the current layer is computing, overlapping communication with compute

**Data flow (one training step, Stage 3):**
```
Forward pass (layer by layer):
  pre-forward hook  → all-gather layer params → ZeroParamStatus = AVAILABLE
  layer forward     → use full params
  post-forward hook → free non-owned param copies → ZeroParamStatus = NOT_AVAILABLE
  [prefetch next layer's params in background]

Backward pass (reverse layer order):
  pre-backward hook → all-gather layer params again (same lifecycle)
  layer backward    → compute gradients
  post-backward hook → reduce-scatter gradients → rank keeps its own grad slice
                    → free params

Optimizer step:
  fully local — each rank has only its param shard, grad slice, optimizer state
  (no all-gather of weights after step — next forward will gather again)
```

**Key insight**: The forward and backward passes each require an all-gather per layer, so Stage 3 has 3× the all-gather operations vs Stage 2. The prefetch coordinator pipelines the next layer's gather behind the current layer's compute to hide latency — the paper doesn't describe this, but it's essential for achieving the ~90% throughput retention the paper claims.

---

### `deepspeed/runtime/engine.py` — Training loop integration

**What to look for:**
- `DeepSpeedEngine.__init__()`: inspects `zero_optimization.stage` config and instantiates `DeepSpeedZeroOptimizer` (Stage 1/2) or `DeepSpeedZeroOptimizer_Stage3` accordingly
- `DeepSpeedEngine.backward(loss)`: calls `optimizer.backward(loss)` which internally manages gradient hooks + reduce-scatter
- `DeepSpeedEngine.step()`: delegates to the ZeRO optimizer's `step()` (local Adam + all-gather for Stage 2; pure local for Stage 3)
- `DeepSpeedEngine.forward()`: for Stage 3, this is a no-op wrapper — the actual fetch/release lifecycle is driven by the module-level hooks registered in `stage3.py`

**Data flow**: User calls `engine.backward(loss)` + `engine.step()` → ZeRO optimizer intercepts → communication + memory management transparent to user

---

## Step Walkthrough (Stage 2)

One training step with Stage 2, 4 GPUs, parameter owned by rank 0:

```
[All ranks] Forward: use local (replicated) params → no communication

[All ranks] Backward:
  grad hooks fire as each param's gradient is computed
  DeepSpeedZeroOptimizer.reduce_gradients():
    bucket accumulates grads from params owned by rank 0
    when bucket full:
      reduce_scatter(bucket)  ← 8B sent, 2B received per rank
      rank 0 now holds g0 (the reduced gradient for its owned param slice)
      ranks 1-3 discard non-owned slices

[Rank 0] step():
  Adam(w0, g0, os0)  ← fully local, no communication
  updated_w0 ready

[All ranks] all_gather(updated weights):
  rank 0 broadcasts updated_w0, rank 1 broadcasts updated_w1, ...
  all ranks now hold full updated model  ← 8B received per rank

Total communication: 8B (reduce_scatter) + 8B (all_gather) = 16B = same as standard all-reduce
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| Stage 1/2: optimizer state partitioning | `DeepSpeedZeroOptimizer.__init__()` — gradient bucket assignment per rank |
| Stage 2: reduce-scatter during backward | `DeepSpeedZeroOptimizer.reduce_gradients()` |
| Stage 2: all-gather after optimizer step | `DeepSpeedZeroOptimizer.step()` |
| Stage 3: parameter partitioning at init | `Init` context manager in `partition_parameters.py` |
| Stage 3: all-gather before layer forward | `_fetch_sub_module()` via pre-forward hook |
| Stage 3: free params after layer forward | `_release_sub_module()` via post-forward hook |
| Stage 3: reduce-scatter in backward | Post-backward hook in `stage3.py` |
| ZeroParamStatus state machine | `ZeroParamStatus` enum in `partition_parameters.py` |
| Communication-computation overlap | `get_param_coordinator()` prefetch in `stage3.py` |

---

## What the Paper Doesn't Tell You

**The `ZeroParamStatus` state machine is central to correctness.** Any code that accesses a Stage 3 parameter must check that its status is `AVAILABLE` first. DeepSpeed adds assertions throughout; in user code, accessing `param.data` outside a `fetch` context silently returns zeros. This is the most common source of bugs when integrating custom layers with Stage 3.

**Stage 3 hooks fire on `nn.Module` boundaries, not on individual parameters.** A module with 100 params gets one all-gather for all 100, not 100 individual gathers. This means granularity of the fetch lifecycle is determined by how you structure your `nn.Module` hierarchy — very large modules keep params in memory longer, very small modules (leaf ops) add gather overhead.

**The prefetch coordinator uses a configurable lookahead.** `deepspeed_config["zero_optimization"]["prefetch_bucket_size"]` controls how many bytes to prefetch ahead. Too small → communication not hidden; too large → memory spike from holding multiple layers' params simultaneously. The paper's throughput numbers assume optimal prefetch tuning.

**Stage 2's `overlap_comm` flag** enables overlapping reduce-scatter with the backward pass computation. Without it, the reduce-scatter is synchronous (waits for the full bucket before continuing backward). With it, gradient hooks fire immediately and communication runs in a background CUDA stream — at the cost of slightly higher peak memory for the gradient buffers.
