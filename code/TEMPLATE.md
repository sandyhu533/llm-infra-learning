# [Framework Name]

**Repo**: [github.com/org/repo](https://github.com/org/repo)  
**Version**: vX.Y.Z or commit hash (pin for stability)  
**Learning goal**: One sentence — what you'll understand after reading this.  
**Prerequisites**: [Paper A](../../papers/category/paper-a.md), [Paper B](../../papers/category/paper-b.md)

---

## Entry Point

`path/to/entry.py` — `ClassName.method_name()`

One paragraph: what this function does and why it's the right place to start. Explain what the call site looks like from user code so the reader knows how to find it in practice.

---

## Key Files

### `path/to/file.py` — What this file owns

**What to look for:**
- `ClassName` or `function_name`: what it does, 2–3 bullet points
- Where the key data structure is defined or mutated
- Where the communication primitive or memory operation fires

**Data flow**: what enters this module → what happens → what exits

**Key insight**: one sentence connecting this code to the paper — what the paper describes abstractly that this file makes concrete.

---

### `path/to/file2.py` — What this file owns

**What to look for:**
- ...

**Data flow**: ...

**Key insight**: ...

---

*(4–6 files total. No more — if you need more, the scope is too broad.)*

---

## Step Walkthrough

Trace one complete inference request or one complete training step through the key files in sequence. Use pseudocode — don't paste real code.

```
EntryPoint.method()
  └─ ModuleA.step()
       ├─ does X → produces Y
       └─ calls ModuleB.op()
            ├─ does P
            └─ returns Q

  └─ ModuleC.finalize(Q)
       └─ ...
```

---

## Paper → Code Mapping

| Paper concept | Code location |
|---------------|--------------|
| Concept from paper (use the paper's terminology) | `ClassName.method()` in `path/to/file.py` |
| ... | ... |

*(Minimum 4 rows. Each row must name a specific class or function, not just a file.)*

---

## What the Paper Doesn't Tell You

Engineering decisions that are only visible in the code — config knobs, performance tradeoffs, correctness constraints, corner cases the paper glosses over.

- **Point 1**: ...
- **Point 2**: ...
- **Point 3**: ...

*(Minimum 3 entries. Do not repeat anything already stated in the paper note.)*
