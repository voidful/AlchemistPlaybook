# AGENTS.md — working contract for AI agents

You are working on **AlchemistPlaybook**, the repo that develops the
`alchemist-playbook` training-recipe skill. Read
[`docs/index.md`](docs/index.md) first — it indexes everything; the
[constitution](docs/constitution.md) defines the governance invariants your
change must preserve.

## The six invariants (short form)

1. **Docs-first** — the `.skill` archive and the Pages site are build
   outputs, never edited directly.
2. **One home per fact** — training knowledge lives in
   `alchemist-playbook/` (SKILL.md + references); repo knowledge lives in
   `docs/`. Never duplicate a number across files.
3. **Indexed or it doesn't exist** — new doc → link it from
   `docs/index.md`; new reference file → row in the SKILL.md routing table;
   new source → row in `references/sources.md`.
4. **Supersede, never rewrite** — `docs/decisions/` is append-only.
5. **No structural change without its doc** — same commit.
6. **Evidence-marked or labeled heuristic** — every value carries
   `[config]` / `[paper]` / `[reported]` / `[heuristic]`.

## Hard constraints

- SKILL.md `description:` ≤ **1024 characters**.
- After any change under `alchemist-playbook/`, repackage the archive
  ([workflow](docs/workflows/release.md)).
- Before committing structural changes, run:

```bash
python3 scripts/check_docs.py
```

- When headline facts change (sources, evals, decisions, routing), update
  the `DATA` block at the top of [`docs/index.html`](docs/index.html) — and
  only that block.

## Common tasks → workflows

| Task | Follow |
| --- | --- |
| Add a paper/recipe to the skill | [docs/workflows/add-source.md](docs/workflows/add-source.md) |
| Repackage `alchemist-playbook.skill` | [docs/workflows/release.md](docs/workflows/release.md) |
| Add or run evals | [docs/workflows/run-evals.md](docs/workflows/run-evals.md) |
| Change direction / structure | Decision record first ([conventions](docs/conventions.md) → frontmatter format) |

## Provenance

The founding charter (imported Claude Cowork instruction, 2026-06) is
preserved verbatim in [docs/constitution.md](docs/constitution.md).
