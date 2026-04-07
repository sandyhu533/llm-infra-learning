# Contributing

Contributions welcome. This repo is a curated collection of LLM infra paper notes and code walkthroughs targeting engineers with backgrounds in distributed systems, OS, and backend infra. Quality and depth matter more than quantity.

---

## Scope

**In scope:**
- Systems and infrastructure papers: LLM serving, training systems, hardware efficiency, memory management, scheduling
- Papers from top venues: OSDI, SOSP, NSDI, USENIX ATC, EuroSys, SC, ISCA, MLSys, NeurIPS (systems track), ICLR (systems track)
- Papers that are in production use or have clear production applicability
- Code walkthroughs for production open-source frameworks anchored to existing paper notes

**Out of scope:**
- Pure ML theory or architecture papers without a systems contribution
- Papers without a clear engineering tradeoff story
- Blog posts, tutorials, or secondary sources
- Research prototype codebases or framework code that duplicates a concept already covered

---

## How to Add a Paper Note

### 1. Copy the template

```bash
cp papers/TEMPLATE.md papers/<category>/<short-name>.md
```

Categories: `foundations/`, `inference/`, `training/`, `scheduling/`, `compression/`

Create a new category directory if the paper doesn't fit existing ones.

### 2. Fill in all sections

Every section in [papers/TEMPLATE.md](papers/TEMPLATE.md) is required. The most important:

**System Tradeoffs** — What does the design optimize for, at what explicit cost? The "Design decisions worth questioning" subsection should include real critiques, not a summary of what the paper says about itself.

**Reading Notes** — Where infra analogies naturally fit, include them here. Be specific: don't say "like caching" — say "like InnoDB buffer pool, because both cache hot working-set data near compute and evict under memory pressure." Don't force analogies where they don't add insight.

### 3. Update the reading roadmap

Add a row to the sequence table in [papers/README.md](papers/README.md) and the root [README.md](README.md), with a "why this order" note that situates the paper relative to its prerequisites.

### 4. Quality checklist

- [ ] TL;DR is one paragraph readable in 60 seconds
- [ ] System Tradeoffs includes at least one genuine critique
- [ ] Key Numbers is populated with actual benchmark figures from the paper
- [ ] Questions & Open Problems has at least 2 questions not answered in the paper
- [ ] Paper link resolves (arXiv or venue page)
- [ ] Reading roadmap updated in both `papers/README.md` and root `README.md`

---

## How to Add a Code Note

### 1. Check scope

A framework qualifies if:
- It's a **production open-source** codebase, not a research prototype
- It has a **direct anchor in an existing paper note** — the code note deepens understanding of that paper, not introduces a new concept
- The concept is **not already covered** by another code note — if two frameworks implement the same idea, pick one

### 2. Create the file

```bash
cp code/TEMPLATE.md code/<inference|training>/<framework-name>.md
```

### 3. Fill in all sections

Every section in [code/TEMPLATE.md](code/TEMPLATE.md) is required:

**Entry Point** — one file and one function. If you can't identify a single entry point, the scope is too broad.

**Key Files** — 4–6 files maximum. For each: name the specific class or function, describe what data enters and exits, and include a "Key insight" line connecting it back to the paper. Don't just describe what the code does.

**Step Walkthrough** — trace one complete request or training step through the key files in sequence using pseudocode. The goal is to show how modules connect at runtime, not reproduce the code.

**Paper → Code Mapping** — at least 4 rows. Each row maps a named paper concept to a specific file + function. "ZeRO partitioning → `stage3.py`" is too vague; "Stage 3: all-gather before layer forward → `_fetch_sub_module()` via pre-forward hook in `stage3.py`" is the target.

**What the Paper Doesn't Tell You** — at least 3 entries that are only visible in the code: config knobs, performance tradeoffs, correctness gotchas. Do not repeat anything already in the paper note.

### 4. Update the index

Add a row to [code/README.md](code/README.md) and the Source Code Reading table in the root [README.md](README.md).

### 5. Quality checklist

- [ ] Entry Point is a single file + function, not a directory or module
- [ ] Key Files has 4–6 entries (no more), each with a named class/function and a "Key insight" line
- [ ] Step Walkthrough traces one complete step end-to-end using pseudocode
- [ ] Paper → Code Mapping has at least 4 rows with specific file + function references
- [ ] "What the Paper Doesn't Tell You" has at least 3 entries absent from the paper note
- [ ] Index updated in both `code/README.md` and root `README.md`

---

## Style

- **Write for engineers, not researchers.** Assume the reader knows distributed systems but not ML.
- **Prefer concrete over abstract.** "Reduces HBM reads by 10×" beats "improves memory efficiency."
- **Use ASCII diagrams** for data flow and architecture — they render everywhere and don't rot.
- **Link explicitly.** A paper note without outgoing links to prerequisites and follow-on work is incomplete. A code note without a filled Paper → Code Mapping table is incomplete.
- **Include personal observations.** The "Reading Notes" section and "What the Paper Doesn't Tell You" section exist for a reason — opinions and surprises are more valuable than neutral summaries.

**Do not:**
- Copy-paste from the paper abstract. Rephrase in your own words.
- Include author affiliations or institutional context unless directly relevant.
- Evaluate papers on ML metrics (accuracy, perplexity) — only on systems metrics (throughput, latency, memory, utilization).
