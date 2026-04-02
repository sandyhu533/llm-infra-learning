# Vidur: A Large-Scale Simulation Framework for LLM Inference

**Authors**: Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav Gulavani, Ramachandran Ramjee, Alexey Tumanov  
**Venue**: MLSys '24  
**Paper**: [https://arxiv.org/abs/2405.05465](https://arxiv.org/abs/2405.05465)  
**Code**: [https://github.com/microsoft/vidur](https://github.com/microsoft/vidur)

---

## TL;DR

Capacity planning for LLM serving clusters is expensive: running real workloads on expensive GPU hardware to evaluate scheduling policies, replica configurations, and SLO compliance is slow and wasteful. Vidur is a high-fidelity discrete-event simulator for LLM serving that accurately predicts latency distributions (TTFT, TPOT, request completion time) without executing on real GPUs. It models LLM execution at the iteration level — including chunked prefill, continuous batching, and KV cache pressure — and enables rapid iteration over scheduling algorithms, hardware configs, and capacity estimates. Simulation accuracy is within 5% of real execution for end-to-end metrics.

---

## Infra Analogy

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| Vidur simulator | ns-3 / DES network simulator | Discrete-event simulation of packet scheduling; predict latency without running real traffic |
| Iteration-level execution model | CPU cycle-accurate simulator | Model each compute "tick" precisely enough to predict real execution time |
| Scheduling policy evaluation | Queuing theory / Little's Law simulation | Evaluate different dispatch algorithms before production rollout |
| Capacity planning via simulation | Traffic modeling + load testing | Answer "how many GPUs do I need for X RPS at P99 < Y ms?" without provisioning the cluster |
| Workload trace replay | tcpreplay / load generator | Replay historical request traces to benchmark a new scheduler or hardware config |

---

## Problem

**What gap does this paper address?**

LLM serving clusters are expensive to experiment with:
- Renting 100s of A100s for scheduling experiments costs thousands of dollars per experiment
- A/B testing scheduling algorithms in production risks SLO violations
- Hardware procurement decisions (how many GPUs, what topology, which parallelism strategy) must be made months in advance

Existing simulators (for networking, CPUs, distributed systems) don't model LLM-specific behaviors: KV cache pressure, variable prefill/decode ratios, token-level batching, and memory-bandwidth-bound decoding.

**Why is this hard?**

LLM execution time is highly variable and input-dependent:
- Prefill cost scales quadratically with sequence length
- Decode cost scales linearly with batch size × KV cache footprint
- Scheduling decisions (which requests to batch) affect memory pressure, which affects future scheduling options — the system is stateful and the state space is large

A simple queueing model (M/M/1 or M/D/1) cannot capture these dependencies. You need iteration-level simulation with a calibrated performance model.

---

## Key Ideas

### Idea 1: Execution Time Predictor

Vidur builds a performance model for each operation in LLM inference:
- **Prefill**: time to compute attention + FFN for a chunk of tokens
- **Decode**: time to run one auto-regressive step for a batch of requests
- **KV cache transfer** (for disaggregated systems)

The model is calibrated by profiling a small set of (batch_size, sequence_length) points on the target hardware, then interpolating. This avoids running full workloads while achieving <5% prediction error.

```
Execution time model:
  T_prefill(n_tokens) = α × n_tokens² + β × n_tokens   (quadratic in seqlen)
  T_decode(batch_size, kv_len) = γ × batch_size × kv_len  (memory-BW bound)
  T_sampler(batch_size) = δ × batch_size               (linear)

Calibration: run microbenchmarks for 10-20 (n, b) points → fit α, β, γ, δ
```

### Idea 2: Discrete Event Simulation

Vidur models the serving system as a discrete event loop:
1. **Events**: request arrival, iteration complete, request departure
2. **State**: KV cache occupancy, in-flight request queue, scheduler state
3. **Scheduler plugin**: any scheduling policy (FCFS, chunked prefill, priority queues) can be plugged in

Each simulated "iteration" advances the clock by the predicted execution time, updates KV cache state, and fires the next batch scheduling decision. This accurately captures the interaction between scheduling and memory pressure.

```
Event loop:
  while events remain:
    event = pop_next_event()
    match event:
      ARRIVAL → enqueue request
      ITERATION_DONE → 
        release KV cache for completed requests
        schedule next batch (via scheduler plugin)
        compute T_next = predict_iteration_time(batch)
        push ITERATION_DONE event at now + T_next
      DEPARTURE → record metrics (TTFT, TPOT, e2e latency)
```

### Idea 3: Workload and Configuration Search Space

Vidur supports:
- **Workload traces**: real request logs (arrival time, prompt length, output length) or synthetic distributions
- **Hardware configs**: single node, tensor parallelism, pipeline parallelism, replica sets
- **Scheduling policies**: FCFS, preemption, chunked prefill, SJF variants
- **Capacity planning**: sweep over (num_replicas, parallelism_degree) to find minimum cost meeting SLO targets

This makes it a practical tool for infrastructure teams to answer: *"For Black Friday traffic, do we need 32 or 48 A100s?"*

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| Cheap, fast scheduling policy evaluation | Simulation accuracy depends on calibration quality |
| Hardware-independent planning | Must re-calibrate when switching GPU type |
| Rapid iteration over configs | Does not model network topology or collective communication exactly |
| SLO compliance prediction | Tail latency estimates may drift at extreme percentiles (P99.9) |

**Design decisions worth questioning:**

- The execution time model uses simple polynomial fits. For heterogeneous hardware (NVLink vs. PCIe) or sparse attention patterns, these fits may be inaccurate.
- The simulator is single-threaded Python; for very high-RPS simulations (>10K req/s), wall-clock simulation time can be slow.
- Calibration assumes hardware is homogeneous across all nodes in the cluster — this fails for mixed GPU clusters.

---

## Connections

**Builds on:**
- [Orca](../inference/orca-continuous-batching.md) — Vidur models Orca-style iteration-level scheduling as one of its built-in policies
- [Sarathi-Serve](sarathi-serve.md) — chunked prefill is a first-class scheduling mode in Vidur
- [vLLM](../inference/vllm-pagedattention.md) — paged KV cache memory model is incorporated

**Inspired / Followed by:**
- Production capacity planning tools at Microsoft Azure and similar cloud providers
- **AlpaServe** — earlier simulation-based approach for model parallelism selection

**Production systems:**
- Used internally at Microsoft for Azure OpenAI capacity planning
- Enables offline evaluation of new scheduling algorithms before production deployment

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| TTFT prediction error | <5% | vs real execution on LLaMA-70B |
| TPOT prediction error | <5% | vs real execution |
| Simulation speed | ~100x faster than real | 1-hour workload simulated in <1 minute |
| Calibration cost | ~30 min profiling | Per hardware type; amortized across all experiments |

---

## Questions & Open Problems

- [ ] How does Vidur handle KV cache eviction and recomputation latency? These are bursty events that may be hard to model analytically.
- [ ] Can the performance model generalize across GPU generations (A100 → H100) without full re-calibration?
- [ ] How accurate is the tail latency (P99.9) prediction vs. median? Tail events are driven by rare scheduling collisions that may be underrepresented in calibration data.
- [ ] Can Vidur model disaggregated prefill/decode (Splitwise-style) accurately, including the KV cache transfer cost over NVLink or InfiniBand?

---

## Reading Notes

Vidur is essentially the LLM-serving equivalent of ns-3 for networking or gem5 for CPUs — a calibrated discrete-event simulator that lets you explore design spaces without paying the real hardware cost. The insight that LLM inference is predictable enough (once you profile the execution time primitives) to simulate accurately is non-trivial.

For infrastructure teams, the most valuable use case is capacity planning: instead of over-provisioning "just in case," you can simulate traffic growth scenarios and find the minimum-cost configuration that keeps P99 TTFT under your SLO. This directly translates to GPU cost savings.

The calibration step (30 min of profiling) is cheap insurance. Any team running LLM inference in production should have a Vidur-like simulator as part of their capacity planning toolkit.
