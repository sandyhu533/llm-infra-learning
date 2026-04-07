# TorchTitan

**Repo**: [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)  
**Version**: main branch (actively developed, PyTorch-native; paths below reflect early 2025 layout)  
**Learning goal**: Understand how TorchTitan composes PyTorch-native parallelism primitives (FSDP2, tensor parallel, pipeline parallel) into a clean, modular training stack — and how it differs from Megatron-LM's monolithic approach.  
**Prerequisites**: [ZeRO](../../papers/training/zero-memory-optimization.md), [Megatron-LM](../../papers/training/megatron-lm.md)

---

## Entry Point

`train.py` — `main()`

Unlike Megatron-LM's deeply abstracted entry point (where `pretrain_gpt.py` hands off to `training.py` which dispatches into a pipeline schedule), TorchTitan's `train.py` is a single, readable file (~300 lines) that you can follow top-to-bottom. It explicitly constructs the model, applies parallelism transforms, and runs the training loop — no framework magic. Start here and you can trace every decision.

---

## Key Files

### `train.py` — Main training loop

**What to look for:**
- `main()`: the entire training pipeline in one function — config loading, model construction, parallelism application, optimizer setup, and the training loop
- The composition order is explicit: build a standard `nn.Module` → apply parallelism transforms → wrap with optimizer → train. Each step is visible in the source, not hidden behind a `pretrain()` abstraction.
- Mixed precision via `torch.amp`, gradient clipping, and optional `torch.compile()` are applied as standard PyTorch calls — no custom wrappers

**Data flow**: config → model (standard nn.Module) → parallelized model (FSDP2 + TP + PP applied) → training loop (forward → loss → backward → optimizer step)

**Key insight**: TorchTitan is a reference implementation, not a framework. It shows how to compose PyTorch distributed primitives rather than hiding them behind abstractions. This is why it's excellent for learning — every parallelism decision is visible in the call chain, and you can modify any part without understanding a framework's internal contracts.

---

### `torchtitan/parallelisms/parallelize_llama.py` — Model-specific parallelism composition

**What to look for:**
- `parallelize_llama()`: the orchestrator function that applies TP, PP, and FSDP2 to a Llama model in a specific order
- TP application: replaces standard `nn.Linear` layers in attention (QKV projections, output projection) and MLP (gate/up/down projections) with DTensor-based parallel versions. Uses `ColwiseParallel` and `RowwiseParallel` from `torch.distributed.tensor.parallel` — these are *transforms*, not custom layer classes.
- PP application: splits transformer blocks across pipeline stages using `torch.distributed.pipelining`. Each stage gets a contiguous range of transformer layers.
- FSDP2 application: wraps modules with `fully_shard()` for ZeRO-3 semantics — per-parameter sharding across the data-parallel group
- The ORDER of application matters: TP first (innermost), then PP, then FSDP2 (outermost). This is because TP operates within a node on individual layers, PP splits layers across stages, and FSDP2 shards the resulting per-stage parameters across DP replicas.

**Data flow**: clean nn.Module → TP transforms replace Linear layers with DTensor parallels → PP splits the module into stages → FSDP2 wraps each stage for data-parallel sharding

**Key insight**: Parallelism is applied as composable transforms on a standard `nn.Module`, not baked into custom layer implementations like Megatron's `ColumnParallelLinear`/`RowParallelLinear`. This means the model code is completely agnostic to parallelism — you can change from TP=4 to TP=8 without touching the model definition.

---

### `torchtitan/models/llama/` — Model definition

**What to look for:**
- `Transformer`: standard PyTorch `nn.Module` subclass — embedding, N transformer blocks, output head. No parallelism annotations, no custom parallel layers, no process group references.
- `TransformerBlock`: standard self-attention + MLP + RMSNorm. Uses plain `nn.Linear`, not `ColumnParallelLinear`.
- `Attention`, `FeedForward`: clean implementations using standard PyTorch ops. RoPE, GQA, SwiGLU — all in pure PyTorch.

**Data flow**: token IDs → embedding → N × (attention + MLP) → RMSNorm → linear head → logits. Identical to a single-GPU model.

**Key insight**: This is the fundamental philosophical difference from Megatron-LM. In Megatron, parallelism is part of the model definition — `ColumnParallelLinear` and `RowParallelLinear` are the model's building blocks, and removing them means rewriting the model. In TorchTitan, the model is clean and parallelism is a separate concern applied externally. The cost is that the parallelism transforms must understand the model's structure (which layers to split, which to replicate), but the model itself remains testable and readable as standard PyTorch.

---

### `torchtitan/config_manager.py` — Configuration system

**What to look for:**
- TOML-based config files defining all parallelism dimensions: `data_parallel_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree`
- Device mesh construction: maps logical parallelism dimensions (DP, TP, PP) to physical GPU topology via `torch.distributed.device_mesh.init_device_mesh()`
- The mesh abstraction: a 3D grid where each axis corresponds to a parallelism dimension. A GPU's position in the mesh determines its TP group, PP stage, and DP replica index — analogous to Megatron's `initialize_megatron()` but using PyTorch-native `DeviceMesh`.

**Data flow**: TOML config → parsed dimensions → `DeviceMesh(mesh_shape=(DP, PP, TP))` → submeshes extracted for each parallelism strategy

**Key insight**: The `DeviceMesh` abstraction maps logical parallelism dimensions to physical GPU topology. When you say `TP=4, PP=2, DP=2` on 16 GPUs, the mesh determines which GPUs share a TP group (typically within a node for NVLink bandwidth) and which form PP stages (typically across nodes). This is the same concern Megatron handles with custom process group initialization, but expressed declaratively.

---

### `torchtitan/checkpoint.py` — Distributed checkpointing

**What to look for:**
- Uses PyTorch's DCP (Distributed Checkpoint, `torch.distributed.checkpoint`) for async, sharded checkpoint saving
- `CheckpointManager`: handles save/load with configurable frequency, async saving to avoid blocking training
- Resharding support: DCP saves metadata about how tensors were sharded. When loading, it can reshard to a different parallelism config — save with TP=4 and load with TP=8 without manual conversion.
- Async checkpointing: `save()` returns immediately; the actual I/O happens in a background thread. This is critical for large models where checkpoint writes can take minutes.

**Data flow**: model state dict + optimizer state → DCP serializes per-rank shards with shard metadata → filesystem. On load: DCP reads metadata → determines resharding plan → each rank loads only its needed shards.

**Key insight**: DCP solves a real production pain point — changing parallelism strategy between training runs without manual checkpoint conversion. In Megatron, switching from TP=4 to TP=8 requires a checkpoint conversion script that understands the exact sharding layout. With DCP, the sharding metadata is stored alongside the data, and the framework handles resharding automatically.

---

## Step Walkthrough

One complete training step with TP=2, PP=2, FSDP2 (DP=2), 8 GPUs total:

```
train.py main()
  ├─ Build model: Transformer() — standard nn.Module, no parallelism
  │    └─ All layers instantiated on meta device, then materialized
  │
  ├─ Apply parallelisms: parallelize_llama(model, world_mesh, parallel_dims)
  │    ├─ TP (innermost): for each transformer block:
  │    │    ├─ Attention: QKV proj → ColwiseParallel, output proj → RowwiseParallel
  │    │    └─ MLP: gate/up proj → ColwiseParallel, down proj → RowwiseParallel
  │    │    (DTensor replaces nn.Linear weight with sharded DTensor on TP mesh)
  │    │
  │    ├─ PP: split transformer blocks into stages
  │    │    └─ Stage 0 gets blocks 0..N/2-1, Stage 1 gets blocks N/2..N-1
  │    │
  │    └─ FSDP2 (outermost): fully_shard() each transformer block
  │         └─ Per-parameter sharding across DP group (ZeRO-3 semantics)
  │
  └─ Training loop (each step):
       ├─ data_loader.next_batch()
       │    └─ Each DP rank gets a different micro-batch
       │
       ├─ model.forward(batch)  — parallelism is transparent to the call
       │    ├─ FSDP2: all-gather params for current block before its forward
       │    ├─ TP: matmuls execute on sharded weights within TP group
       │    │    └─ ColwiseParallel: each rank computes partial output
       │    │    └─ RowwiseParallel: each rank computes partial sum → all-reduce
       │    ├─ PP: stages execute in schedule (1F1B)
       │    │    └─ Stage 0 sends activations to Stage 1 via p2p
       │    └─ FSDP2: free non-owned param shards after block's forward
       │
       ├─ loss.backward()
       │    ├─ FSDP2: all-gather params again for each block's backward
       │    ├─ TP: gradient communication mirrors forward (all-reduce per layer)
       │    ├─ PP: gradients flow backward through pipeline stages
       │    └─ FSDP2: reduce-scatter gradients after each block's backward
       │
       ├─ torch.nn.utils.clip_grad_norm_()  — gradient clipping
       ├─ optimizer.step()  — each rank updates only its owned param shards
       └─ optional: checkpoint_manager.save()  — async DCP, non-blocking
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| ZeRO Stage 3 (partition params, grads, optimizer state) | `fully_shard()` in `parallelize_llama.py` — wraps each transformer block with FSDP2 for per-parameter sharding |
| Megatron TP: ColumnParallelLinear / RowParallelLinear | `ColwiseParallel` / `RowwiseParallel` transforms from `torch.distributed.tensor.parallel`, applied in `parallelize_llama()` |
| 1F1B pipeline schedule | `torch.distributed.pipelining` schedule, configured in `parallelize_llama.py` for PP stages |
| Mixed precision training (loss scaling, fp16/bf16 compute) | `torch.amp` integration in `train.py` training loop |
| Activation checkpointing (recompute activations in backward) | `torch.utils.checkpoint` applied per transformer block in `parallelize_llama()` |
| 3D parallelism mesh (DP × TP × PP) | `init_device_mesh()` in `config_manager.py` — constructs `DeviceMesh` with named dimensions |
| Distributed checkpointing with resharding | `torch.distributed.checkpoint` (DCP) in `checkpoint.py` — save/load with automatic shard remapping |

---

## What the Paper Doesn't Tell You

**DTensor dispatch overhead is real but intentional.** PyTorch's DTensor abstraction for tensor parallel adds a dispatch layer compared to Megatron's hand-written collectives. Each `DTensor` operation goes through a placement-aware dispatch that determines whether to all-gather, all-reduce, or redistribute — this adds microseconds per op. For large models where compute dominates, this overhead is negligible (~1-2%). For small models or profiling micro-benchmarks, the overhead is measurable and can mislead. TorchTitan accepts this tradeoff explicitly: composability and correctness over raw kernel-level performance. The bet is that `torch.compile()` will eventually fuse through the dispatch overhead.

**FSDP2 is not FSDP1 with a version bump.** FSDP2 uses per-parameter sharding (each `nn.Parameter` is individually sharded as a DTensor), while FSDP1 flattened all parameters in a module into a single `FlatParameter` and sharded that. The difference matters for TP composition: FSDP1's flat parameter broke TP because TP needs to shard individual weight matrices along specific dimensions, which is impossible when they're concatenated into a flat buffer. FSDP2's per-parameter approach composes naturally with TP because each parameter retains its identity and can be independently sharded along different axes for different parallelism strategies.

**Pipeline parallel with FSDP2+TP is the hardest composition to get right.** The pipeline schedule must account for FSDP2 parameter fetching (all-gather before each stage's forward) and TP communication (all-reduce within each stage) simultaneously. The memory budget for pipeline buffers (activations held across micro-batches) must coexist with FSDP2 all-gather buffers (full parameters temporarily materialized) and TP communication buffers. Getting peak memory right requires understanding the interleaving of all three systems — the pipeline schedule determines *when* FSDP2 fetches fire, which determines peak memory. This is not derived in any paper.

**`torch.compile()` compatibility is a design constraint, not an afterthought.** TorchTitan's parallelism transforms produce standard PyTorch operations that the compiler can trace and optimize. This is a deliberate contrast with Megatron, whose custom CUDA kernels (`fused_kernels/`, custom all-reduce implementations) are opaque to `torch.compile()`. The payoff is that compiled TorchTitan can fuse communication with compute (e.g., overlapping the next layer's all-gather with the current layer's matmul) without hand-written overlap code — the compiler discovers these opportunities automatically.

**Float8 training via torchao shows the composability payoff.** TorchTitan has experimental support for float8 training through the `torchao` library. Because parallelism is applied as external transforms on standard modules, swapping `nn.Linear` for a float8 linear is orthogonal to the parallelism config — you can run float8 + TP + FSDP2 without any parallelism-aware changes to the float8 code. In Megatron, adding a new numeric format means writing new `ColumnParallelLinear` and `RowParallelLinear` variants that handle the format-specific casting and communication. This is where the composable-transforms philosophy pays its biggest dividend.
