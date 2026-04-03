# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Authors**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He  
**Venue**: SC '20  
**Paper**: [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)  
**Code**: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

## TL;DR

Training large models hits a GPU memory wall long before hitting a compute wall. ZeRO (Zero Redundancy Optimizer) identifies that data-parallel training wastes memory by replicating the same optimizer state, gradients, and parameters on every GPU. It eliminates this redundancy by partitioning these states across GPUs — each GPU owns only 1/N of the total — and gathering what it needs just-in-time via collective communication. Stage 3 reduces per-GPU memory from ~16 bytes/param to ~1.9 bytes/param at 64 GPUs, enabling training of models 8x larger with the same hardware at comparable throughput.

---

## Background: DP and MP Baselines

Before ZeRO, two approaches existed for training large models:

### Data Parallelism (DP)

Each GPU holds a **full copy** of the model. The input batch is split across GPUs — each GPU computes its own forward and backward pass on its shard, then gradients are synchronized via **all-reduce** so every GPU ends up with identical updated weights.

```
GPU 0: [full model] ← batch shard 0
GPU 1: [full model] ← batch shard 1
GPU 2: [full model] ← batch shard 2
GPU 3: [full model] ← batch shard 3
         ↓ all-reduce gradients ↓
         all GPUs now identical
```

**Memory problem**: every GPU stores full parameters + gradients + optimizer state = 16 bytes/param. Adding GPUs increases throughput but does **not** reduce per-GPU memory — you still need 112GB for a 7B model on every single GPU.

### Model Parallelism (MP)

Split the model itself across GPUs. Two variants: **Tensor Parallelism** (split a single layer's weight matrix across GPUs, all-reduce partial results after each matmul) and **Pipeline Parallelism** (assign different layers to different GPUs, pass activations between stages). Both reduce per-GPU memory, but add communication overhead, pipeline bubbles, and implementation complexity — typically practical only within a single node (8–16 GPUs) where NVLink bandwidth is available.

**ZeRO's insight**: keep DP's simple programming model, eliminate its memory redundancy by partitioning state across DP ranks and fetching on demand.

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

### Idea 1: Three Stages of Partitioning — with Data Flow Examples

**Setup**: 4 GPUs (rank 0–3), model has 4 parameters [p0, p1, p2, p3].  
Each param: 2B weight (fp16) + 2B grad (fp16) + 12B optimizer state (fp32) = **16B total**.  
Naive DP memory = 16B × 4 = **64B per GPU**.

---

**Standard DP (baseline)**

```
State at rest (all GPUs identical):
  GPU 0: [w0 w1 w2 w3] [g0 g1 g2 g3] [os0 os1 os2 os3]  = 64B
  GPU 1:       (same)                                      = 64B
  GPU 2:       (same)                                      = 64B
  GPU 3:       (same)                                      = 64B

Forward:
  Each GPU processes its batch shard using its local (replicated) weights.
  No cross-GPU communication needed.

Backward:
  Each GPU computes gradients for ALL 4 params w.r.t. its own batch shard.
  GPU j produces local grads: [g0_j, g1_j, g2_j, g3_j]

Sync:
  all-reduce(grads) — every GPU sends its 4 local grads and receives the sum:
    g0 = g0_0 + g0_1 + g0_2 + g0_3   (summed across all GPUs)
    g1 = g1_0 + g1_1 + g1_2 + g1_3
    g2, g3 similarly
  → all GPUs now hold identical summed gradients [g0 g1 g2 g3]
  Communication: 2 × 8B = 16B  (all-reduce = reduce then broadcast)

Update:
  Every GPU independently runs Adam on all 4 params using [g0,g1,g2,g3] and [os0..os3].
  No communication needed — results are identical across GPUs.
```
**Memory = 64B/GPU. Communication = 16B/step.**

---

**Stage 2 (P_os+g) — partition gradients + optimizer state**

Weights remain replicated. Each GPU owns only 1 param's gradient bucket and optimizer state.

```
State at rest:
  GPU 0: [w0 w1 w2 w3] [g0]  [os0]  = 8B + 2B + 3B = 13B
  GPU 1: [w0 w1 w2 w3] [g1]  [os1]  = 13B
  GPU 2: [w0 w1 w2 w3] [g2]  [os2]  = 13B
  GPU 3: [w0 w1 w2 w3] [g3]  [os3]  = 13B

Forward:
  Same as standard DP — each GPU uses its local replicated weights.

Backward:
  Each GPU computes gradients for all 4 params (still needed for correct results).
  As gradients are produced, immediately reduce-scatter them:
    GPU 0 accumulates: g0 = g0_0 + g0_1 + g0_2 + g0_3  ← only keeps this
    GPU 1 accumulates: g1 = g1_0 + g1_1 + g1_2 + g1_3
    GPU 2 accumulates: g2 = ...
    GPU 3 accumulates: g3 = ...
  Non-owned gradient contributions are sent out and discarded immediately.
  Communication: reduce-scatter = 8B  (each GPU sends 8B, receives 2B)

Update:
  GPU i runs Adam(w_i, g_i, os_i) — fully local, zero communication.
  Each GPU updates only its owned param.

Sync:
  all-gather(updated weights):
    GPU 0 broadcasts updated w0, GPU 1 broadcasts w1, etc.
    → all GPUs receive [w0_new, w1_new, w2_new, w3_new]
  Communication: all-gather = 8B
```
**Memory = 13B/GPU (80% reduction). Communication = 8B + 8B = 16B — same as standard DP.**

The key difference from standard DP: instead of all-reduce (reduce → broadcast the same gradient to everyone), ZeRO Stage 2 does reduce-scatter (each GPU gets only its slice) + all-gather (broadcast updated weights). Same total bytes, but gradient memory drops from 8B to 2B per GPU.

---

**Stage 3 (P_os+g+p) — partition everything**

Weights are no longer replicated. Each GPU holds only 1 param in full at rest. Weights must be gathered on demand before each compute step.

```
State at rest:
  GPU 0: [w0] [g0] [os0]  = 4B
  GPU 1: [w1] [g1] [os1]  = 4B
  GPU 2: [w2] [g2] [os2]  = 4B
  GPU 3: [w3] [g3] [os3]  = 4B

Forward pass (done layer-by-layer; treating all 4 params as one layer here):
  Step 1 — all-gather(weights):
    GPU 0 broadcasts w0, GPU 1 broadcasts w1, GPU 2 broadcasts w2, GPU 3 broadcasts w3.
    → Every GPU now temporarily holds [w0, w1, w2, w3]  (8B extra)
    Communication: 8B
  Step 2 — compute:
    Each GPU runs its batch shard through the full weight matrix [w0,w1,w2,w3].
  Step 3 — discard non-owned weights:
    GPU 0 drops [w1,w2,w3], keeping only w0.
    GPU memory returns to near-rest state (+ activations for backward).

Backward pass (reverse layer order):
  Step 1 — all-gather(weights) again:
    Need all weights to compute gradients — same gather as forward.
    Communication: 8B
  Step 2 — compute backward:
    Each GPU computes gradient contributions for all 4 params w.r.t. its batch shard.
  Step 3 — reduce-scatter(grads):
    GPU 0 accumulates: g0 = g0_0 + g0_1 + g0_2 + g0_3
    GPU 1 accumulates: g1 = ...
    Non-owned grad contributions sent out and discarded.
    Communication: 8B
  Step 4 — discard non-owned weights.

Update:
  GPU i runs Adam(w_i, g_i, os_i) — fully local, zero communication.
  (No all-gather of weights at end — next forward will gather again.)
```
**Memory = 4B/GPU (16× reduction). Communication = 8B + 8B + 8B = 24B ≈ 1.5× standard DP.**

---

**Summary**

| Stage | What's partitioned | Memory/GPU | Comm/step | vs standard DP |
|-------|-------------------|-----------|-----------|---------------|
| Standard DP | nothing | 64B | 16B | baseline |
| Stage 2 (P_os+g) | grads + optimizer state | 13B | 16B | same comm, 80% less memory |
| Stage 3 (P_os+g+p) | everything | 4B | 24B | 1.5× comm, 94% less memory |

**Key insight**: Stage 2 achieves 80% memory reduction at **zero communication overhead** by restructuring who accumulates gradients. Stage 3 adds weight sharding for another 3× memory gain, but pays a 1.5× communication tax because weights must be gathered before every layer in both forward and backward.

### Idea 3: ZeRO-R — Activation and Fragmentation Memory

After ZeRO-DP, activations become the next bottleneck (`num_layers × batch × seq_len × hidden_dim`). ZeRO-R addresses three residual memory sources:

- **Partitioned activations (Pa)**: after each layer's forward, scatter activations across DP ranks (each GPU stores 1/N). Backward all-gathers on demand — trades GPU-to-GPU comm for memory savings.
- **Checkpointing integration**: partition the activation checkpoints themselves across DP ranks. Combined with recompute, per-GPU activation memory ≈ `O(checkpoint_size / N)`.
- **Memory defragmentation (Mb)**: pre-allocate contiguous buffers for gradients and optimizer states upfront to prevent fragmentation-caused OOM from dynamic alloc/free cycles.

### Idea 4: ZeRO-Offload and ZeRO-Infinity (extensions)

When GPU memory is still insufficient:
- **ZeRO-Offload**: offload optimizer states and gradients to CPU RAM; CPU runs Adam asynchronously while GPU computes the next forward pass.
- **ZeRO-Infinity**: further offload to NVMe when CPU RAM is also insufficient (requires NVMe bandwidth >> compute time per step).

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Per-GPU memory footprint | Added all-gather / reduce-scatter communication |
| Simple data-parallel programming model | Stage 3 adds 1.5x communication vs standard DP |
| Linear memory scaling with GPU count | Communication latency sensitive to network bandwidth |
| Training arbitrarily large models | Optimizer step latency increases with gather overhead |

**Design decisions worth questioning:**

- Stage 3 has ~1.5x communication overhead vs Stage 2. For bandwidth-constrained clusters (e.g., 100Gbps Ethernet vs 400Gbps InfiniBand), Stage 3 can significantly hurt throughput. Often Megatron-LM tensor parallelism is preferred within a node (NVLink) + ZeRO Stage 2 across nodes.
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
| Memory per param (Stage 2, 64 GPUs) | ~2.2 bytes | 86% reduction |
| Memory per param (Stage 3, 64 GPUs) | ~0.25 bytes | 94% reduction |
| Communication overhead vs standard DP | 0% (Stage 2), ~50% (Stage 3) | Per-iteration added traffic |
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

The interaction with Megatron-LM's 3D parallelism is important: tensor parallelism addresses the within-node memory problem (via NVLink), while ZeRO Stage 2 addresses the across-node optimizer memory problem. They're not alternatives — they're complementary axes of the memory optimization space.
