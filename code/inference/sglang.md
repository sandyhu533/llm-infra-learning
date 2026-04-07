# SGLang

**Repo**: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Version**: v0.4.x (concepts stable; file paths may shift in later releases)  
**Learning goal**: Understand how SGLang's serving architecture works — how RadixAttention enables automatic prefix caching, how the scheduler manages requests with prefix-aware continuous batching, and how the runtime orchestrates multi-call LLM programs.  
**Prerequisites**: [ORCA](../../papers/inference/orca-continuous-batching.md), [vLLM/PagedAttention](../../papers/inference/vllm-pagedattention.md), FlashAttention paper

---

## Entry Point

`python/sglang/srt/server.py` — `launch_server()` / `launch_engine()`

The SRT (SGLang Runtime) server is launched from this file. It spins up the `TokenizerManager` (front-end process handling HTTP/API), one or more `Scheduler` processes (one per GPU or tensor-parallel group), and a `DetokenizerManager`. The architecture is multi-process: tokenization, scheduling/model execution, and detokenization each run in separate processes communicating via ZMQ. Start here to understand how the components are wired, then jump to `scheduler.py` for the core loop.

---

## Key Files

### `python/sglang/srt/managers/scheduler.py` — Central scheduler and batch loop

**What to look for:**
- `Scheduler.event_loop_normal()`: the main scheduling loop. Each iteration processes new requests, builds the next batch, runs model forward, and processes outputs — analogous to vLLM's `LLMEngine.step()` but running as a standalone process
- `Scheduler.process_input_requests()`: intake of new requests from the tokenizer manager. Each request's token sequence is matched against the radix cache to determine how many tokens are already cached
- `Scheduler.get_next_batch_to_run()`: the core scheduling decision. Selects requests for prefill (only the cache-miss suffix) and combines them with running decode requests into a single batch. Implements chunked prefill to prevent long prompts from starving decode
- `Scheduler.process_batch_result()`: post-forward processing — sampling, cache insertion, completion checks

**Data flow**: New requests arrive via ZMQ → `process_input_requests()` performs prefix matching → requests enter `waiting_queue` (sorted by cache locality) → `get_next_batch_to_run()` builds a `ScheduleBatch` → model forward → `process_batch_result()` updates cache and returns completions

**Key insight**: Unlike vLLM's scheduler which treats scheduling and prefix caching as separate concerns, SGLang's scheduler is fundamentally prefix-aware. It considers cache hit rates when ordering requests in the waiting queue, preferring those that share prefixes with the current radix cache state. This is cache-locality-aware scheduling — the scheduler acts like an OS page-replacement-aware process scheduler, not a simple FIFO.

---

### `python/sglang/srt/mem_cache/radix_cache.py` — RadixAttention / Radix Tree Cache

**What to look for:**
- `RadixCache`: the top-level cache class. Wraps a radix tree where each edge is labeled with a token subsequence and each node points to physical KV cache block IDs
- `RadixCache.match_prefix()`: given an input token sequence, walks the tree to find the longest cached prefix. Returns the matched length and the corresponding KV block indices — these blocks are reused without recomputation
- `RadixCache.insert()`: after a forward pass, extends the tree with newly computed KV blocks. If a prefix path already exists, it shares the existing nodes
- `RadixCache.evict()`: LRU eviction at the node level. When GPU memory runs low, the least recently used leaf nodes are pruned and their KV blocks returned to the free pool
- `TreeNode`: the internal node structure — stores a token key segment, reference count, physical block pointers, and child pointers

**Data flow**: Token sequence → `match_prefix()` → (matched_length, cached_block_ids) → only the unmatched suffix needs prefill → after forward, `insert()` adds new KV blocks to the tree

**Key insight**: The radix tree naturally deduplicates shared prefixes across all requests. Unlike vLLM's block-level hash table for prefix caching (which requires exact block-aligned matches), SGLang's radix tree provides semantic-level caching — the tree structure itself encodes prefix relationships. This means two requests sharing a system prompt automatically share KV blocks without any explicit user hints or block-boundary alignment. Think of it as a filesystem's prefix tree (trie) for path lookups, but applied to token sequences and their associated KV cache blocks.

---

### `python/sglang/srt/managers/tokenizer_manager.py` — Request front-end and lifecycle

**What to look for:**
- `TokenizerManager`: runs in its own process, owns the tokenizer and the HTTP-facing event loop
- `TokenizerManager.generate_request()`: handles incoming generate/chat API calls — tokenizes the input, creates a request object with a unique ID, sends it to the scheduler via ZMQ, and awaits the response
- Streaming support: for streaming responses, the tokenizer manager detokenizes incremental token chunks and sends them back to the HTTP connection as SSE events
- `TokenizerManager.handle_batch()`: batch API handling for multi-turn and multi-call SGLang programs

**Data flow**: HTTP request → tokenize → ZMQ send to Scheduler → await response → detokenize → HTTP response

**Key insight**: Separating tokenization into its own process ensures CPU-bound tokenization (especially for large vocabularies or chat template rendering) never blocks the GPU-bound model execution process. This is the same principle as putting a web server's request parsing on a separate thread pool from its compute handlers, but applied at the process level with explicit IPC.

---

### `python/sglang/srt/model_executor/model_runner.py` — Model forward pass execution

**What to look for:**
- `ModelRunner`: manages the loaded model, KV cache memory pool, and attention backend
- `ModelRunner.forward()`: executes a single model forward pass for a `ScheduleBatch`. Handles both prefill tokens and decode tokens in the same call
- KV cache pool management: pre-allocates a large contiguous GPU buffer for KV cache blocks, then manages allocation/deallocation via the radix cache's block pointers
- Attention backend selection: configures FlashInfer (default), Triton, or FlashAttention backends based on hardware and model type

**Data flow**: `ScheduleBatch` (token IDs, positions, block tables, prefix lengths) → model forward → logits → returned to scheduler for sampling

**Key insight**: SGLang uses FlashInfer as its primary attention backend, which provides ragged tensor support for variable-length sequences in a batch. Rather than padding all sequences to the same length (wasteful), FlashInfer operates on a packed tensor with a separate indptr array describing sequence boundaries. This is more memory-efficient than vLLM's padded approach and enables better GPU utilization with heterogeneous batch compositions.

---

### `python/sglang/srt/layers/attention/` — Attention layer and backend dispatch

**What to look for:**
- Attention layer implementations that dispatch to different backends (FlashInfer, Triton)
- `flashinfer_backend.py`: the FlashInfer integration — sets up paged KV cache wrappers, handles the ragged batch format, dispatches prefill and decode to different FlashInfer kernels
- Paged KV cache layout: SGLang uses paged KV cache (like vLLM) for the physical storage, but the radix tree provides the higher-level prefix-aware indexing on top of it. The page table maps logical pages to physical GPU memory blocks

**Data flow**: Attention layer receives Q, K, V tensors + block table metadata → dispatches to FlashInfer → paged attention kernel reads/writes KV cache blocks → returns attention output

**Key insight**: SGLang's KV cache is a two-level structure: the physical layer is a paged block pool (same concept as vLLM's PagedAttention), while the logical layer is the radix tree that maps token sequences to blocks. This separation means SGLang gets both memory-efficient block allocation (no contiguous allocation needed per sequence) and automatic prefix sharing (the radix tree deduplicates at the token level, not the block level).

---

## Step Walkthrough

Trace one complete inference request through the system:

```
HTTP POST /generate arrives at the API server
  └─ TokenizerManager.generate_request()
       ├─ apply chat template, tokenize input → token_ids
       ├─ create Request object with unique rid
       └─ send Request to Scheduler via ZMQ

Scheduler.event_loop_normal()
  └─ Scheduler.process_input_requests()
       ├─ receive Request from ZMQ
       ├─ radix_cache.match_prefix(token_ids)
       │    └─ walk radix tree: find longest cached prefix
       │    └─ return (matched_length, cached_kv_block_ids)
       ├─ only token_ids[matched_length:] needs prefill
       ├─ allocate KV cache blocks for the unmatched suffix
       └─ add to waiting_queue (sorted by prefix match length for locality)

  └─ Scheduler.get_next_batch_to_run()
       ├─ pop requests from waiting_queue for prefill
       │    └─ respect chunked prefill budget: cap total prefill tokens
       ├─ combine with currently running decode requests
       ├─ build ScheduleBatch: token_ids, positions, block_tables,
       │    prefix_lens, seq_lens for the entire mixed batch
       └─ return ScheduleBatch

  └─ ModelRunner.forward(ScheduleBatch)
       ├─ embed tokens → hidden states
       ├─ for each transformer layer:
       │    └─ Attention.forward()
       │         ├─ prefill tokens: compute Q, K, V → write K, V to
       │         │    allocated cache blocks → full attention over prompt
       │         └─ decode tokens: compute Q → read K, V from cached
       │              blocks via block_table → attend to all prior tokens
       └─ return logits for last token of each sequence

  └─ Scheduler.process_batch_result()
       ├─ sample next token from logits (greedy / top-p / etc.)
       ├─ radix_cache.insert(new_token, new_kv_block)
       │    └─ extend the radix tree with the new token edge
       ├─ check stopping criteria (EOS, max_tokens, stop strings)
       ├─ if complete: remove from running batch, send result to TokenizerManager
       └─ if not complete: sequence stays in running batch for next decode step

TokenizerManager receives completed result via ZMQ
  └─ detokenize output token_ids → text
  └─ return HTTP response to client
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| RadixAttention (prefix tree for KV cache) | `RadixCache` class in `python/sglang/srt/mem_cache/radix_cache.py` |
| Automatic prefix caching (match longest prefix) | `RadixCache.match_prefix()` in `radix_cache.py` |
| LRU eviction of cached prefixes | `RadixCache.evict()` — prunes least-recently-used leaf nodes |
| Continuous batching (iteration-level scheduling) | `Scheduler.get_next_batch_to_run()` in `scheduler.py` |
| Prefix-aware scheduling (prioritize cache hits) | Waiting queue ordering logic in `Scheduler.process_input_requests()` |
| Compressed FSM for constrained decoding | `python/sglang/srt/constrained/` — jump-forward decoding via precompiled FSM |
| Multi-call LLM programs (SGLang frontend) | `python/sglang/lang/` — the SGLang DSL that compiles multi-step programs into batched runtime calls |

---

## What the Paper Doesn't Tell You

**Radix tree memory overhead is non-trivial.** Each `TreeNode` stores a token key segment, a list of child pointers, a reference count, and a list of physical KV block IDs. For workloads with many unique prefixes (e.g., thousands of distinct system prompts), the tree becomes wide and deep, consuming significant CPU memory. The tree is also protected by a lock, making concurrent access a potential bottleneck — SGLang mitigates this by confining all cache operations to the scheduler process.

**FlashInfer is not optional — it's load-bearing.** SGLang's batch construction assumes ragged tensor format (variable-length sequences packed contiguously with an indptr array). FlashInfer's paged attention kernels accept this format natively, avoiding the need to pad sequences to a common length. If FlashInfer is unavailable, SGLang falls back to Triton kernels, but performance degrades noticeably. This tight coupling to FlashInfer is a significant difference from vLLM, which abstracts attention backends more cleanly.

**Chunked prefill prevents head-of-line blocking.** A naive scheduler would process an entire long prompt (e.g., 32K tokens) in a single prefill step, blocking all decode requests for that iteration. SGLang caps the number of prefill tokens per iteration (`chunked_prefill_size`), interleaving prefill chunks with decode steps. This is similar to Sarathi-Serve's approach, but the implementation is more complex because the chunk boundary must align with the radix cache's block granularity — you can't split a block mid-way through.

**Cache-aware scheduling is a form of bin packing.** The scheduler doesn't simply FIFO the waiting queue. It scores requests by how many tokens would be cache hits, effectively preferring requests that "fit" into the current cache state. This is analogous to a CPU scheduler that prefers threads whose working sets are already in the L2 cache. The benefit is dramatic in multi-turn chat workloads where many requests share conversation history prefixes.

**Multi-modal inputs complicate the cache key scheme.** When an image is part of the input, its visual tokens need to be incorporated into the radix tree's key. SGLang handles this by hashing image embeddings into pseudo-token IDs that are inserted into the token sequence at the correct positions. This means the radix tree can cache image prefixes just like text prefixes, but the hash collision risk and variable embedding sizes add complexity not discussed in the serving papers.

**The ZMQ-based multi-process architecture has tradeoffs.** Unlike vLLM's asyncio-based single-process engine (pre-v2), SGLang uses separate OS processes for tokenization, scheduling, and detokenization, connected by ZMQ sockets. This provides true CPU parallelism (no GIL contention) and process isolation, but introduces serialization overhead for every request and result. For high-throughput workloads, the ZMQ message passing can become a bottleneck — SGLang mitigates this with batched message sends and careful buffer management.
