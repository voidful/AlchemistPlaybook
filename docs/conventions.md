---
type: conventions
title: Documentation and content conventions
status: active
updated: 2026-09-07
---

# Conventions

## Evidence marking (applies to all skill content)

Every concrete value in `alchemist-playbook/` carries exactly one marking.
Definitions live in [glossary.md](glossary.md); the authoritative statement
is `references/sources.md` header. In short: `[config]` > `[paper]` >
`[reported]`, and anything unbacked is `[heuristic]`. A number with no
marking cannot merge (constitution invariant 6).

## Reference-file structure

Each `references/*.md` follows the same shape:

1. `# Title` — one line stating the topic.
2. `## Contents` — numbered section list (files ≥ ~100 lines).
3. Numbered `## N. Section` bodies; cite sources inline by name and mark
   values. Cross-reference other files as `references/<file>.md §N`.
4. One topic per file. If a section outgrows its file's topic, split it into
   a new file **and** add a routing-table row in the same commit.

## docs/ frontmatter

Every markdown file in `docs/` starts with YAML frontmatter:

```yaml
---
type: index | constitution | architecture | conventions | glossary | workflow | decision
title: <short title>
status: active | superseded-by: NNNN
updated: YYYY-MM-DD        # date of last substantive edit
---
```

Decision records add `decided: YYYY-MM-DD` and optionally
`supersedes: NNNN` (or a sequence of IDs for several records). For a partial
supersession, name the affected clauses in the new record and index; preserve
the historical record's body. Links between docs are relative paths; a link to a
missing file fails `scripts/check_docs.py`.

## Writing rules

- Skill workflows describe applicable defaults and completion criteria.
  Preserve evidence requirements, but avoid unconditional intake, format,
  tool, and approval rules where the user's task supplies the choice.
  See [decision 0006](decisions/0006-task-scoped-skill-guidance.md).
- **Human-road prose follows ASD-STE100 (Simplified Technical English)
  principles.** This applies to `docs/index.html` visible text and
  `README.md`: short sentences (≤ 20 words for instructions, ≤ 25 for
  description), one instruction per sentence, active voice, simple verb
  tenses, explicit articles, one name per thing, no idioms. Technical names
  (SFT, DPO, learning rate, …) are allowed as-is. Agent-road docs may be
  denser but keep the one-name-per-thing rule.
- **Docs describe the repo, not training.** A hyperparameter value appearing
  in `docs/` (outside a quoted example) violates one-home-per-fact.
- Dates are absolute (`2026-08-19`), never relative ("last month").
- 正體中文 terms keep their English gloss on first use per file
  (e.g., 煉丹調參 — "alchemy tuning": training-run hyperparameter craft).
- The SKILL.md `description:` field must stay **≤ 1024 characters** — this
  is a hard packaging constraint (see decision 0001; it has been hit twice
  in history).

## docs/index.html maintenance

The human report is a single self-contained file: no external requests, all
data in the `const DATA = {…}` block at the top, rendering code below it.
When headline facts change (source count, reference files, eval results,
decision records, routing rows), edit **only the DATA block**. Charts follow
the dataviz discipline: validated categorical palette (2 series max used),
one axis per chart, legend + direct labels, light/dark via
`prefers-color-scheme`, tooltips on hover, and a table fallback for every
chart.

## Commit style

Imperative subject describing the substance ("Add GLM-5 recipe: …"), one
logical change per commit — the commit log doubles as the ablation/decision
log (skill principle 8 applied to this repo itself).
