# [Paper Title]

**Authors**: Author A, Author B, ...  
**Venue**: Conference/Journal 'YY  
**Paper**: [Link](https://arxiv.org/abs/xxxx.xxxxx)  
**Code**: [Link](https://github.com/...)  

---

## TL;DR

> One paragraph. What does this paper do, and why does it matter? Write as if explaining to a senior engineer who has 60 seconds.

---

## Infra Analogy

> For engineers coming from distributed systems / OS / backend infra — what does this map to in your existing mental model?

| LLM Concept | Traditional Infra Analogy | Why It Maps |
|-------------|--------------------------|-------------|
| e.g., KV cache | Buffer pool (InnoDB / PostgreSQL shared_buffers) | Both cache hot working-set data close to compute to avoid re-reading from slow storage |
| e.g., continuous batching | Event-driven I/O (epoll / reactor pattern) | Both multiplex many in-flight requests over a shared resource instead of one-request-one-thread |

---

## Problem

**What gap does this paper address?**

Describe the status quo before this paper. What was broken, slow, or impossible?

**Why is this hard?**

What makes this problem non-trivial? What constraints make naive solutions fail?

---

## Key Ideas

### Idea 1: [Name]

Description. Include diagrams in ASCII or link to images if helpful.

```
Example: architecture diagram, pseudocode, or data flow
```

### Idea 2: [Name]

Description.

### Idea 3: [Name]

Description.

---

## System Tradeoffs

| Optimizes For | At the Cost of |
|---------------|----------------|
| e.g., throughput | e.g., latency tail |
| e.g., memory efficiency | e.g., compute overhead |

**Design decisions I'd question:**

- Why did they choose X over Y?
- Under what workload does this design break down?

---

## Connections

**Builds on:**
- [Paper A](link) — how this work extends it

**Inspired / Followed by:**
- [Paper B](link) — what came after

**Production systems using these ideas:**
- System X at Company Y

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| e.g., throughput improvement | 2.7x | vs baseline on A100 |

---

## Questions & Open Problems

- [ ] Question 1 — something not fully explained in the paper
- [ ] Question 2 — an extension worth exploring
- [ ] Question 3 — limitation that future work should address

---

## Reading Notes

*Personal observations, things that surprised me, or other thoughts.*

