# Contributing

Contributions welcome. This repo is a curated collection of LLM infra paper notes targeting engineers with backgrounds in distributed systems, OS, and backend infra. Quality and depth matter more than quantity.

---

## Scope

**In scope:**
- Systems and infrastructure papers: LLM serving, training systems, hardware efficiency, memory management, scheduling
- Papers from top venues: OSDI, SOSP, NSDI, USENIX ATC, EuroSys, SC, ISCA, MLSys, NeurIPS (systems track), ICLR (systems track)
- Papers that are in production use or have clear production applicability

**Out of scope:**
- Pure ML theory or architecture papers (e.g., new attention variants without systems contribution)
- Papers without a clear engineering tradeoff story
- Blog posts, tutorials, or secondary sources (link to the primary paper)

---

## How to Add a Paper Note

### 1. Copy the template

```bash
cp papers/TEMPLATE.md papers/<category>/<short-name>.md
```

Categories: `inference/`, `training/`, `foundations/`, `scheduling/`, `compression/`

Create a new category directory if the paper doesn't fit existing ones.

### 2. Fill in all sections

Every section in the template is required. The two most important:

**Infra Analogy** — This is the differentiating section of this repo. Map each key concept in the paper to a traditional infra analog (OS, distributed systems, databases, storage). Be specific: don't say "like caching" — say "like InnoDB buffer pool, because both cache hot working-set data near compute and evict under memory pressure." If you can't find an analogy, describe what makes the concept genuinely novel.

**System Tradeoffs** — What does the design optimize for, and at what explicit cost? Include the "Design decisions worth questioning" subsection with real critiques — don't just summarize what the paper says about itself.

### 3. Add an entry to README.md

Add a row to the appropriate table in `README.md`:

```markdown
| [Paper Title](papers/<category>/<short-name>.md) | Venue 'YY | One-line key idea | ✅ |
```

If adding to the Reading Roadmap, annotate with the infra analogy lens.

---

## Quality Bar

Before submitting, check:

- [ ] **Infra Analogy section is filled** with at least 2 specific mappings (not just "like caching" — be precise)
- [ ] **TL;DR** is one paragraph that a senior engineer can read in 60 seconds and understand the contribution
- [ ] **System Tradeoffs** includes at least one genuine critique of the design
- [ ] **Key Numbers** table is populated with actual benchmark numbers from the paper
- [ ] **Questions & Open Problems** includes at least 2 non-trivial questions (not answered in the paper)
- [ ] Paper link (arXiv or venue) is correct and resolves
- [ ] README entry added

---

## Style Notes

- Write for engineers, not researchers. Assume the reader knows distributed systems but not ML.
- Prefer concrete over abstract: "reduces HBM reads by 10x" > "improves memory efficiency."
- ASCII diagrams are encouraged for data flow and architecture — they render everywhere.
- Use the "Connections" section to link papers explicitly. A paper note without outgoing links is incomplete.
- Personal observations in "Reading Notes" are valuable — include them.

---

## What Not to Include

- Do not copy-paste from the paper abstract. Rephrase in your own words.
- Do not include author affiliations or institutional context unless directly relevant to understanding the work.
- Do not evaluate papers on ML metrics (accuracy, perplexity) — only on systems metrics (throughput, latency, memory, utilization).
