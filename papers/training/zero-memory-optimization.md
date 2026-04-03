# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Authors**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He  
**Venue**: SC '20  
**Paper**: [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)  
**Code**: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

## TL;DR

Training large models hits a GPU memory wall long before hitting a compute wall. ZeRO (Zero Redundancy Optimizer) identifies that data-parallel training wastes memory by replicating the same optimizer state, gradients, and parameters on every GPU. It eliminates this redundancy by partitioning these states across GPUs — each GPU owns only 1/N of the total — and gathering what it needs just-in-time via collective communication. Stage 3 reduces per-GPU memory from ~16 bytes/param to ~1.9 bytes/param at 64 GPUs, enabling training of models 8x larger with the same hardware at comparable throughput.

---

## Problem

**What gap does this paper address?**

In standard data-parallel training, every GPU holds:
- **Parameters**: 2 bytes/param (fp16) or 4 bytes/param (fp32)
- **Gradients**: same as parameters
- **Optimizer state**: for Adam, 2× fp32 copies (momentum + variance) = 8 bytes/param, plus fp32 master weights = 4 bytes/param

Total: **~16 bytes/param** for fp16 training with Adam. A 7B parameter model needs ~112GB per GPU. Even an H100 (80GB) can't fit it with a single GPU, and data parallelism across GPUs doesn't help — each GPU still holds the full 112GB.

Model parallelism (tensor/pipeline) helps but adds complexity and communication overhead. ZeRO achieves similar memory reduction while keeping the simpler data-parallel programming model.

**Why is this hard?**

The optimizer step requires the full gradient vector to update each parameter. If you shard parameters across GPUs, each GPU must still see the gradient for its assigned parameters — which may have been computed on other GPUs. The challenge is designing communication patterns that are (a) correct, (b) have the same total communication volume as standard data parallelism, and (c) overlap with computation.

---

## Key Ideas

### Idea 1: Three Stages of Partitioning

ZeRO partitions training state progressively across N GPUs (data-parallel rank N):

```
Standard DP:  [params | grads | optimizer_state] × N replicas
              Memory: 16 bytes/param per GPU (N GPUs, all identical)

Stage 1 (P_os):  params + grads replicated, optimizer_state partitioned
              Memory: 4 + 12/N bytes/param per GPU
              At N=64: ~4.2 bytes/param

Stage 2 (P_os+g): params replicated, grads + optimizer_state partitioned
              Memory: 2 + 14/N bytes/param per GPU
              At N=64: ~2.2 bytes/param

Stage 3 (P_os+g+p): everything partitioned
              Memory: 16/N bytes/param per GPU
              At N=64: ~0.25 bytes/param
```

### Idea 2: Communication Pattern — Same Volume as Standard DP

Standard DP does one all-reduce of gradients per backward pass = 2Ψ bytes (Ψ = param count).

ZeRO replaces this with:
- **Forward pass (Stage 3)**: all-gather parameters before each layer = Ψ bytes
- **Backward pass**: reduce-scatter gradients after backward = Ψ bytes

Total: 2Ψ — exactly the same as standard data parallelism. No communication overhead for Stage 1 and 2; Stage 3 has 1.5x overhead (extra all-gather in backward for param reconstruction).

```
Stage 3 forward pass:
  for each layer:
    all-gather(params_layer)  ← assemble from all ranks
    compute forward
    discard non-owned params  ← free immediately

Stage 3 backward pass:
  for each layer (reverse):
    all-gather(params_layer)  ← need params again for grad computation
    compute backward
    reduce-scatter(grads_layer) → each rank gets its slice
    update owned optimizer state (local)
```

### Idea 3: ZeRO-R — Reducing Activation and Fragmentation Memory

Separate from optimizer state, ZeRO-R addresses:
- **Activation partitioning**: instead of replicating activations across DP replicas (needed for backward), partition them and all-gather during backward
- **Activation checkpointing integration**: only checkpoint partitioned activations (reduces checkpointing overhead)
- **Memory defragmentation**: pre-allocate buffers to avoid fragmentation during dynamic tensor creation

### Idea 4: ZeRO-Offload and ZeRO-Infinity (extensions)

When GPU memory is still insufficient, offload to CPU RAM (ZeRO-Offload) or NVMe (ZeRO-Infinity). The compute/communication overlap strategy ensures CPU Adam updates happen asynchronously while GPU computes the next forward pass.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Per-GPU memory footprint | Added all-gather / reduce-scatter communication |
| Simple data-parallel programming model | Stage 3 adds 1.5x communication vs standard DP |
| Linear memory scaling with GPU count | Communication latency sensitive to network bandwidth |
| Training arbitrarily large models | Optimizer step latency increases with gather overhead |

**Design decisions worth questioning:**

- Stage 3 has ~1.5x communication overhead vs Stage 1/2. For bandwidth-constrained clusters (e.g., 100Gbps Ethernet vs 400Gbps InfiniBand), Stage 3 can significantly hurt throughput. Often Megatron-LM tensor parallelism is preferred within a node (NVLink) + ZeRO Stage 1/2 across nodes.
- The all-gather before each layer in forward requires holding the full layer's parameters in memory during that layer's compute. For very wide layers (large hidden dim), this can be a spike.
- ZeRO-Offload's CPU Adam update assumes CPU memory is ample. In practice, large models require >1TB CPU RAM — not always available.

---

## Connections

**Builds on:**
- Data parallelism (standard distributed training baseline)
- AllReduce / reduce-scatter / all-gather collective communication (MPI primitives)
- Activation checkpointing — ZeRO-R integrates with this

**Inspired / Followed by:**
- **[Megatron-LM](megatron-lm.md)** — combines tensor/pipeline parallelism with ZeRO Stage 1; complementary approaches
- **FSDP** (PyTorch Fully Sharded Data Parallel) — PyTorch's native implementation of ZeRO Stage 3 semantics
- **ZeRO++** — reduces communication by quantizing gradients and using hierarchical all-gather
- **MegaScale** — uses ZeRO-style memory management in production at ByteDance

**Production systems:**
- DeepSpeed (Microsoft) — reference implementation, widely used in industry
- PyTorch FSDP — used in Meta's LLaMA training
- Megatron-DeepSpeed — combined tensor parallel + ZeRO

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Memory per param (standard DP, Adam) | 16 bytes | fp16 training: params + grads + optimizer state |
| Memory per param (Stage 1, 64 GPUs) | ~4.2 bytes | 75% reduction |
| Memory per param (Stage 3, 64 GPUs) | ~0.25 bytes | 94% reduction |
| Communication overhead vs standard DP | 0% (Stage 1/2), ~50% (Stage 3) | Per-iteration added traffic |
| Max model size trained (paper claim) | 1 trillion params | With 1024 GPUs, Stage 3 |
| Throughput retention (Stage 3, 64 GPUs) | ~90% of standard DP | On 100Gbps IB, GPT-style model |

---

## Questions & Open Problems

- [ ] Stage 3 all-gather before each layer creates a memory spike for large layers. How do frameworks pipeline this to reduce peak?
- [ ] ZeRO vs tensor parallelism: what's the crossover point where TP is better than Stage 3? Depends on network topology (NVLink vs IB)?
- [ ] FSDP (PyTorch) vs DeepSpeed ZeRO Stage 3: are there correctness differences or just API differences?
- [ ] ZeRO++ uses quantized gradients for communication. What's the accuracy cost at large scale?

---

## Reading Notes

The framing is elegant: standard data parallelism is "replicate everything, sync gradients." ZeRO asks "what's the minimum state each GPU needs to own?" The answer: 1/N of the optimizer state, 1/N of the gradients, and optionally 1/N of the parameters. The rest can be fetched on demand via collective ops whose total volume matches what you'd do anyway.

This is the distributed-systems instinct of "don't replicate data you don't need." In storage systems it's erasure coding vs full replication; in databases it's sharding vs replica sets. The GPU memory crisis in LLM training is the same fundamental problem: too much redundancy, not enough partitioning.

The interaction with Megatron-LM's 3D parallelism is important: tensor parallelism addresses the within-node memory problem (via NVLink), while ZeRO Stage 1/2 addresses the across-node optimizer memory problem. They're not alternatives — they're complementary axes of the memory optimization space.
