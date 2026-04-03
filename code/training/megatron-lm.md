# Megatron-LM

**Repo**: [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)  
**Version**: core library (`megatron/core/`) introduced in 2023; paths below reflect the `core` refactor  
**Learning goal**: Understand how tensor parallelism splits weight matrices and where collective communication fires, and how the 1F1B pipeline schedule interleaves micro-batches to minimize bubble fraction.  
**Prerequisites**: [Megatron-LM](../../papers/training/megatron-lm.md), [ZeRO](../../papers/training/zero-memory-optimization.md) (background)

---

## Entry Point

`pretrain_gpt.py` → `megatron/training/training.py` — `train()`

`pretrain_gpt.py` sets up the model/optimizer/data factories and calls `pretrain()`. The actual loop is `train()` in `training.py`, which calls `train_step()` repeatedly. Follow `train_step()` down through the schedule into the model.

---

## Key Files

### `megatron/core/tensor_parallel/layers.py` — Tensor parallel linear layers

**What to look for:**
- `ColumnParallelLinear`: splits the weight matrix column-wise across TP ranks. Each rank computes `Y_i = X @ W_i^T` (a partial output). An all-gather on the output reconstructs the full `Y`. Used in the first linear of MLP (expand to 4h) and in QKV projection.
- `RowParallelLinear`: splits the weight matrix row-wise. Each rank receives a slice of `X` (no gather needed on input since the input was already column-split). Each rank computes a partial sum; an all-reduce sums the partials. Used in the second linear of MLP (contract back to h) and in the output projection.
- `VocabParallelEmbedding`: vocabulary embedding table split column-wise across ranks; output logits similarly split.
- The communication ops (`_gather_along_first_dim`, `_reduce`) are in `mappings.py` in the same package — they call `torch.distributed` collectives tagged with the TP process group.

**Pattern**: Every MLP block has exactly one all-reduce (after RowParallelLinear). Every attention block has one all-reduce (after output projection). TP degree = number of GPUs over which the weight is split.

**Data flow:**
```
Input X (shape: [B, S, H]) — identical on all TP ranks (replicated)
  → ColumnParallelLinear: each rank computes X @ W_col_i  (partial output)
  → RowParallelLinear: each rank computes partial_input_i @ W_row_i  (partial sum)
  → all_reduce(partial sums) → full output on all TP ranks
```

---

### `megatron/core/pipeline_parallel/schedules.py` — Pipeline schedule

**What to look for:**
- `forward_backward_no_pipelining()`: the baseline — single pipeline stage, standard backward. Use this to understand the interface before looking at pipelined versions.
- `forward_backward_pipelining_without_interleaving()`: classic 1F1B schedule. Ranks fill the pipeline during the "warmup" phase (all doing forward), then enter steady state (each rank does one forward, one backward per micro-batch), then drain.
- `forward_backward_pipelining_with_interleaving()`: interleaved 1F1B (each GPU holds multiple non-consecutive pipeline stages, called "virtual pipeline stages"). Reduces bubble fraction from `(p-1)/p` to `(p-1)/(mp)` where `m` = virtual stages per GPU.
- `get_forward_backward_func()`: dispatcher that selects the right schedule based on pipeline parallelism config.

**What the schedule manages:**
- Micro-batch ordering across pipeline stages
- When to send/receive activations between adjacent pipeline ranks (`p2p_communication.py`)
- When to free activation memory (during backward) to bound peak memory

**Data flow:**
```
Steady-state 1F1B (rank i in a 4-stage pipeline):
  receive activations from rank i-1
  → forward micro-batch k  → send activations to rank i+1
  receive grad from rank i+1
  → backward micro-batch k-1  → send grad to rank i-1
```

---

### `megatron/core/models/gpt/gpt_model.py` — GPT model with TP+PP

**What to look for:**
- `GPTModel.__init__()`: instantiates `TransformerBlock` with the correct pipeline stage assignment — only the layers assigned to this GPU's pipeline stage are materialized
- `TransformerBlock`: each layer is a `TransformerLayer` containing `SelfAttention` + `MLP`, both using `ColumnParallelLinear` / `RowParallelLinear`
- `pre_process` / `post_process` flags: the first pipeline stage embeds tokens; the last stage computes logits. Middle stages skip both. This is how PP avoids storing the embedding and head on every GPU.

**Data flow**: input token IDs (first stage only) → embedding → transformer layers (this stage's subset) → pass activations to next stage → final stage outputs logits

---

### `megatron/training/training.py` — Training loop

**What to look for:**
- `train_step()`: calls the schedule's `forward_backward_func` with a list of micro-batches; the schedule handles all pipelining, gradient accumulation, and communication internally
- `train_step()` then calls `optimizer.step()` — if using ZeRO (via DeepSpeed integration), this triggers the reduce-scatter + all-gather; in Megatron-standalone mode, a standard all-reduce on gradients happens here
- `save_checkpoint()` / `load_checkpoint()`: pipeline-parallel checkpointing saves each rank's model shard independently — worth reading to understand how PP state is serialized

**Data flow**: micro-batch list → pipeline schedule → accumulated gradients → optimizer step → weight update

---

### `pretrain_gpt.py` — Entry point and parallelism config

**What to look for:**
- `initialize_megatron()`: sets up all three process groups — TP group (within node), PP group (across nodes), DP group (across replicas)
- Rank assignment: `get_tensor_model_parallel_rank()`, `get_pipeline_model_parallel_rank()` — every GPU has a 3-tuple of (TP rank, PP rank, DP rank) identifying its role
- `model_provider()`: constructs the model; only layers for this GPU's PP stage are instantiated

---

## Training Step Walkthrough

One gradient accumulation cycle with `TP=2, PP=4, DP=2` (16 GPUs total):

```
pretrain_gpt.py: each GPU knows its (TP=x, PP=y, DP=z) rank

train_step() → forward_backward_pipelining_without_interleaving(micro_batches=[mb0..mb7])

Warmup phase (PP fills the pipeline):
  Stage 0: forward mb0 → send activations to stage 1
  Stage 1: forward mb0 → send to stage 2  [stage 0: forward mb1]
  Stage 2: forward mb0 → send to stage 3  [...]
  Stage 3: forward mb0 → loss computed, backward starts

Steady state (1F1B per stage):
  Each stage: receive activations from prev → forward mbn
              receive grads from next → backward mb(n-1) → send grads to prev

Drain phase: backward propagates back through stages

Within each stage, per transformer layer:
  ColumnParallelLinear: Y_partial = X @ W_col_local
  RowParallelLinear: Z_partial = Y_partial @ W_row_local
  all_reduce(Z_partial across TP group) → Z_full  [2 all-reduces per layer: attn + MLP]

After all micro-batches:
  gradients accumulated across micro-batches
  all_reduce(gradients across DP group)  [or ZeRO reduce-scatter if using DeepSpeed]
  optimizer.step()
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| Column-parallel linear (split output dim) | `ColumnParallelLinear` in `tensor_parallel/layers.py` |
| Row-parallel linear (all-reduce output) | `RowParallelLinear` in `tensor_parallel/layers.py` |
| TP group all-reduce per layer | `_reduce()` in `tensor_parallel/mappings.py` |
| 1F1B pipeline schedule | `forward_backward_pipelining_without_interleaving()` in `schedules.py` |
| Interleaved schedule (lower bubble fraction) | `forward_backward_pipelining_with_interleaving()` in `schedules.py` |
| Pipeline stage layer assignment | `pre_process`/`post_process` flags in `GPTModel` |
| 3D process group setup | `initialize_megatron()` in `pretrain_gpt.py` |
| Gradient accumulation across micro-batches | `train_step()` in `training.py` |

---

## What the Paper Doesn't Tell You

**TP requires NVLink to be practical.** The paper discusses TP communication cost theoretically. In practice, each transformer layer has 2 all-reduces for attention and 2 for MLP = 4 all-reduces per layer. For a 96-layer model with TP=8, that's 384 all-reduces per step. Over PCIe (32 GB/s), this is a 10–20× slowdown vs NVLink (600 GB/s). This is why TP degree is always constrained to GPUs within a single node (8 GPUs via NVLink).

**The interleaved schedule trades memory for lower bubble fraction.** Virtual pipeline stages (`--num-layers-per-virtual-pipeline-stage`) reduce bubble from `(p-1)/p` to `(p-1)/(mp)`, but each GPU now holds activations from `m` different micro-batches simultaneously — peak activation memory increases `m×`. This is a fundamental knob that the paper derives analytically but the code makes concrete.

**Sequence parallelism (`--sequence-parallel`) is a TP extension not in the original paper.** LayerNorm and dropout are not naturally tensor-parallel (they operate on the full hidden dim). Sequence parallelism keeps the activation *sequence*-split between TP all-gather and reduce-scatter, so LayerNorm/dropout operate on 1/TP of the sequence rather than the full tensor. This cuts activation memory by TP× for these ops but adds two more collectives per layer.

**Gradient accumulation is micro-batch granular, not token granular.** The pipeline schedule accumulates gradients across `num_micro_batches` micro-batches before doing the optimizer step. This means gradient memory = `sizeof(gradients)` regardless of global batch size — the tradeoff is longer time between weight updates.
