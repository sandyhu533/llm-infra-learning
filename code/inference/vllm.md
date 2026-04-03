# vLLM

**Repo**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)  
**Version**: v0.4.x (concepts stable across versions; file paths may shift in v0.5+)  
**Learning goal**: Understand how continuous batching and PagedAttention are implemented — from request intake to the attention kernel call.  
**Prerequisites**: [ORCA](../../papers/inference/orca-continuous-batching.md), [vLLM/PagedAttention](../../papers/inference/vllm-pagedattention.md)

---

## Entry Point

`vllm/engine/llm_engine.py` — `LLMEngine.step()`

Each call to `step()` is one iteration of the continuous batching loop: schedule → execute → decode → return finished outputs. Everything else is plumbing in support of this loop.

---

## Key Files

### `vllm/engine/llm_engine.py` — Top-level engine

**What to look for:**
- `LLMEngine.__init__`: how the scheduler, tokenizer, and worker pool are wired together
- `LLMEngine.step()`: the main loop body — calls `scheduler.schedule()`, dispatches to workers, processes outputs
- `LLMEngine.add_request()`: how incoming requests are queued as `SequenceGroup` objects

**Data flow**: External request → `add_request()` → waiting queue → `step()` pulls scheduled sequences → worker executes → outputs returned

---

### `vllm/core/scheduler.py` — Continuous batching scheduler

**What to look for:**
- `Scheduler.schedule()`: the core scheduling decision. Returns a `SchedulerOutputs` with three lists: `scheduled_seq_groups` (running), `prefill_seq_groups` (new), `preempted` (evicted due to OOM)
- `_schedule_prefills()` and `_schedule_running()`: priority and preemption logic
- `Scheduler._preempt()`: what happens when KV blocks run out — either recompute (drop and re-prefill) or swap to CPU

**Data flow**: Waiting/running/swapped queues → scheduling policy → block allocation requests → `SchedulerOutputs`

**Key insight vs. ORCA paper**: The paper describes iteration-level scheduling abstractly. The code shows the preemption policy concretely: sequences are evicted in FCFS order, and eviction means either recomputing from scratch (cheap for short contexts) or swapping KV blocks to CPU (cheaper for long contexts). This is a real engineering tradeoff the paper doesn't detail.

---

### `vllm/core/block_manager_v1.py` — PagedAttention block allocator

**What to look for:**
- `BlockSpaceManagerV1.allocate()`: assign physical blocks to a new sequence; called when a sequence enters the running state
- `BlockSpaceManagerV1.append_slots()`: extend block allocation as a sequence generates new tokens
- `BlockSpaceManagerV1.fork()`: copy-on-write for beam search — logical blocks are shared until a write forces a physical copy
- `self.gpu_allocator` / `self.cpu_allocator`: two separate free lists; swapping moves blocks between them

**Data flow**: `SequenceGroup` → `allocate()` → `BlockTable` (logical → physical mapping) → passed to `ModelRunner` for attention

**Key insight**: The block table is just a list of integers (physical block IDs). The indirection from logical to physical is resolved at the attention kernel call, not earlier.

---

### `vllm/worker/model_runner.py` — Forward pass executor

**What to look for:**
- `ModelRunner.prepare_input_tensors()`: builds the actual CUDA tensors from scheduled sequences — `input_ids`, `positions`, `slot_mapping`, `block_tables`
- `slot_mapping`: one slot ID per token in the current batch — tells the attention kernel exactly which physical KV slot to write to during prefill/decode
- `block_tables`: per-sequence list of physical block IDs — tells the attention kernel where to read cached KV during decode
- `ModelRunner.execute_model()`: calls the model forward with these tensors; handles both prefill and decode in the same call

**Data flow**: `SchedulerOutputs` → tensor construction → model forward → logits → `SamplingMetadata` → sampled tokens

**Key insight**: Prefill and decode sequences are batched together in a single forward pass using attention masking. The `is_prompt` flag switches the attention backend between a prefill mode (compute new KV, write to slots) and decode mode (read from block_tables).

---

### `vllm/attention/backends/flash_attn.py` — Paged attention kernel interface

**What to look for:**
- `FlashAttentionBackend.forward()`: the actual kernel dispatch
- For decode: calls `flash_attn_with_kvcache()` from the `flash_attn` package, passing `block_table` so the kernel can gather KV blocks across non-contiguous physical memory
- For prefill: calls standard `flash_attn_varlen_func()` with `slot_mapping` to write newly computed KV into the correct slots
- `kv_cache`: a pre-allocated tensor of shape `[num_blocks, 2, block_size, num_heads, head_dim]` — the physical KV pool

**Data flow**: `slot_mapping` + `block_tables` + `kv_cache` → kernel → attention output

---

## Request Walkthrough

Trace a single decode-phase request through one `step()` call:

```
LLMEngine.step()
  └─ Scheduler.schedule()
       ├─ _schedule_running(): sequence is running, allocate next block if needed
       └─ returns SchedulerOutputs{running=[seq], prefill=[], preempted=[]}

  └─ Worker.execute_model(SchedulerOutputs)
       └─ ModelRunner.prepare_input_tensors()
            ├─ input_ids = [last_generated_token]
            ├─ slot_mapping = [physical_slot_for_new_kv]   ← where to write new KV
            └─ block_tables = [[blk0, blk1, blk2, ...]]   ← where to read cached KV

       └─ GPTModel.forward(input_ids, ...)
            └─ for each layer:
                 └─ Attention.forward()
                      └─ FlashAttentionBackend.forward()
                           ├─ write new KV to slot_mapping location
                           └─ read past KV from block_tables → compute attention

  └─ Sampler.forward(logits) → next token id

  └─ LLMEngine._process_model_outputs() → update SequenceGroup state
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| Iteration-level scheduling | `Scheduler.schedule()` in `vllm/core/scheduler.py` |
| KV cache as fixed-size blocks | `BlockAllocator` in `vllm/core/block_manager_v1.py` |
| Logical → physical block table | `BlockTable` (list of `PhysicalTokenBlock`) per sequence |
| Block allocation on sequence arrival | `BlockSpaceManagerV1.allocate()` |
| Preemption on OOM | `Scheduler._preempt()` — recompute or swap-to-CPU |
| Copy-on-write for beam search | `BlockSpaceManagerV1.fork()` |
| slot_mapping for KV write | `ModelRunner.prepare_input_tensors()` |
| Paged KV read in attention | `FlashAttentionBackend.forward()` with `block_tables` |

---

## What the Paper Doesn't Tell You

**Preemption policy is not greedy-optimal.** The paper says sequences are preempted when memory runs out, but doesn't specify which ones. vLLM preempts the last-arrived sequence (LCFS within running queue), which avoids starving long-running requests but isn't optimal for throughput in all workloads.

**Prefill and decode share a single forward pass.** The paper discusses them separately, but the implementation batches them together. The attention backend switches mode per-sequence using metadata, not per-batch. This is what makes "chunked prefill" (Sarathi-Serve) a natural extension — you just limit how many prefill tokens enter any given step.

**Block size is a tunable latency/memory tradeoff.** Default block size is 16 tokens. Smaller blocks reduce internal fragmentation (last block is ~half-empty on average) but increase the block table length and kernel overhead. Larger blocks do the opposite. This knob doesn't appear in the paper.

**`slot_mapping` is an implementation detail that makes zero-copy KV writing possible.** Rather than copying KV into a contiguous buffer post-attention, the kernel writes directly to the correct physical slot during the attention pass itself. The indirection is resolved inside the CUDA kernel via `slot_mapping`.
