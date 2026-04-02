# Splitwise: Efficient Generative LLM Inference Using Phase Splitting

**Authors**: Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini  
**Venue**: ISCA '24  
**Paper**: [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)

---

## TL;DR

LLM inference has two fundamentally different phases: **prefill** (process the input prompt — compute-bound) and **decode** (generate tokens one by one — memory-bandwidth-bound). Running them together on the same GPU cluster forces a hardware configuration that's suboptimal for both. Splitwise proposes **phase splitting (PD disaggregation)**: route prefill requests to a dedicated "prompt" cluster and decode requests to a "token" cluster, each sized for its own workload characteristics. This achieves better hardware utilization and allows operators to independently scale each phase, reducing cost by ~20% while meeting the same SLOs.

---

## Problem

**What gap does this paper address?**

In mixed prefill+decode serving:
- **Prefill** is compute-bound: processes N tokens in one forward pass; benefits from high FLOPS/memory-bandwidth ratio (favor A100 compute)
- **Decode** is memory-bandwidth-bound: loads the entire model per step to generate 1 token; benefits from raw memory bandwidth

When mixed on the same hardware:
- The hardware is a compromise, optimal for neither
- A long prefill request "stalls" all decode requests in the batch (latency spikes)
- Scaling to handle more decode load also scales prefill capacity, often wastefully

**Why is this hard?**

Separating phases requires *KV cache transfer* between clusters: after prefill, the KV cache must be sent from the prompt machine to the token machine. This transfer must be fast enough to not dominate end-to-end latency. The system must also route requests, track state across two clusters, and handle failures at the boundary.

---

## Key Ideas

### Idea 1: Phase Splitting Architecture

```
                    ┌─────────────────┐
Request arrives ──► │  Load Balancer  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Prompt Machine │  ← prefill phase
                    │  (compute-opt)  │    e.g., A100 80GB
                    └────────┬────────┘
                             │ KV cache transfer (RDMA/NVLink)
                    ┌────────▼────────┐
                    │  Token Machine  │  ← decode phase
                    │  (memory-opt)   │    e.g., A100 40GB × 2
                    └────────┬────────┘
                             │
                    Response returned
```

Each cluster can be independently scaled: if decode is the bottleneck, add more token machines without touching prompt machines.

### Idea 2: Hardware Heterogeneity

Different GPU configurations can be optimal for different phases:
- Prompt machines: fewer, higher-compute GPUs (more CUDA cores per memory)
- Token machines: more, memory-bandwidth-optimized GPUs (or even lower-cost GPUs)

Splitwise shows that mixing GPU types — previously impractical in a monolithic serving setup — becomes natural under PD disaggregation.

### Idea 3: KV Cache Transfer Pipeline

The KV cache (generated during prefill) must be transferred to the token machine before decode can start. Splitwise overlaps this transfer with the last few prefill steps to hide transfer latency. Uses RDMA over InfiniBand for ~200 GB/s transfer rates.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Hardware efficiency (right-sized per phase) | KV transfer overhead (~1–5ms per request) |
| Tail latency stability (no prefill/decode interference) | More complex cluster management |
| Cost efficiency (~20% savings) | Need to operate two separate pools |
| Independent scaling | Request routing and state tracking complexity |

**Design decisions worth questioning:**

- KV transfer latency is workload-dependent. For short prompts, the overhead is proportionally large. Is there a threshold below which mixed serving is still better?
- What happens when one phase is the bottleneck and the other is idle? The paper assumes a static split ratio — dynamic load balancing between phases is not fully addressed.
- RDMA requires specialized networking infrastructure. Not all clusters have this — what's the fallback?

---

## Connections

**Builds on:**
- [Orca](orca-continuous-batching.md) — continuous batching within each phase
- [vLLM](vllm-pagedattention.md) — PagedAttention for memory management within each cluster

**Similar work:**
- **DistServe** (2024) — concurrent independent work on PD disaggregation, similar conclusions
- **Mooncake** (Moonshot AI, 2024) — KV cache-centric disaggregation for large-scale deployment
- **Llumnix** — adds live migration of KV cache across instances for load balancing

**Production systems:**
- PD disaggregation is becoming standard in large-scale deployments (adopted by major cloud providers)
- Connects to my work: **tidal scheduling** in our graph mining system routes stream tasks (latency-sensitive) and batch tasks (throughput-sensitive) to different resource pools — structurally identical to PD disaggregation

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Cost reduction | ~20% | vs monolithic serving at same SLO |
| KV transfer overhead | 1–5ms | RDMA, for typical prompt lengths |
| Prefill stall elimination | near 100% | decode latency no longer affected by prefill |

---

## Questions & Open Problems

- [ ] How do you determine the optimal ratio of prompt machines to token machines for a given workload distribution?
- [ ] For very long context (128K+ tokens), KV cache transfer becomes huge. Is there a hybrid approach where only partial KV is transferred?
- [ ] How does PD disaggregation interact with speculative decoding (draft model runs on which cluster)?
- [ ] Dynamic re-routing: if a request's prompt is shorter than expected, can it be routed directly to a token machine that also handles prefill?

---

## Reading Notes

This paper makes explicit what was implicitly known: prefill and decode are different *workload types* that happen to be packaged in the same serving binary. The insight — *treat them as separate services* — is a classic distributed systems move (microservices, service decomposition). It's surprising this wasn't the default architecture from the start.

The KV cache transfer is the linchpin. The whole architecture is feasible only because:
1. RDMA makes transfer fast enough (~200 GB/s)
2. KV cache size is bounded (unlike, say, activations during training)

This paper is required reading before designing any production LLM serving system at scale.
