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

## How to Add a Code Note

### 1. Check scope first

A framework qualifies if:
- It's a **production open-source** codebase (not a research prototype)
- It has a **direct anchor in an existing paper note** in this repo — the code note should deepen understanding of that paper, not introduce a new concept
- The concept it covers is **not already covered** by another framework note — if two frameworks implement the same idea (e.g., FSDP and DeepSpeed both implement ZeRO Stage 3), pick one

### 2. Create the file

```bash
# inference frameworks
touch code/inference/<framework-name>.md

# training frameworks
touch code/training/<framework-name>.md
```

### 3. Fill in all sections

Every section in the [note template](code/README.md#note-structure) is required:

**Entry Point** — one file and one function. If you can't identify a single entry point, the scope is too broad; narrow it.

**Key Files** — 4–6 files maximum. For each file:
- Name the specific class or function to read, not just the file
- Explain what data enters and exits the module
- Include a "Key insight" line connecting the code to the paper — don't just describe what the code does

**Step Walkthrough** — trace one complete request (inference) or one complete training step (training) through the key files in sequence. Use the pseudocode block style from existing notes. The goal is to show how modules connect at runtime, not to reproduce the code.

**Paper → Code Mapping** — at least 4 rows. Each row maps a named concept from the paper note to a specific file + class/function. Vague entries like "ZeRO partitioning → stage3.py" don't count — be specific: "Stage 3: all-gather before layer forward → `_fetch_sub_module()` via pre-forward hook in `stage3.py`".

**What the Paper Doesn't Tell You** — at least 3 entries. These must be things that are only visible in the code: config knobs, performance tradeoffs, correctness gotchas, implementation constraints the paper glosses over. Do not repeat anything already stated in the paper note.

### 4. Update README entries

Add a row to the table in `code/README.md` and add/update the framework row in the root `README.md` Source Code Reading table.

---

## Code Note Quality Bar

Before submitting, check:

- [ ] Entry Point is a single file + function, not a directory
- [ ] Key Files section has 4–6 files (no more), each with a named class/function and a "Key insight" line
- [ ] Step Walkthrough traces one complete step end-to-end using pseudocode
- [ ] Paper → Code Mapping has at least 4 rows with specific file + function references
- [ ] "What the Paper Doesn't Tell You" has at least 3 entries not present in the paper note
- [ ] No key file entry just describes what the code does without connecting it back to a paper concept

---

## What Not to Include

- Do not copy-paste from the paper abstract. Rephrase in your own words.
- Do not include author affiliations or institutional context unless directly relevant to understanding the work.
- Do not evaluate papers on ML metrics (accuracy, perplexity) — only on systems metrics (throughput, latency, memory, utilization).
