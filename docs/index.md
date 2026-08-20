---
type: index
title: Documentation index
status: active
updated: 2026-08-19
---

# Documentation index

Master index of the AlchemistPlaybook repository documentation. **Every doc in
`docs/` must be reachable from this file** — if it is not linked here, it does
not exist (see [constitution](constitution.md), invariant 3).

Two roads into this repo:

- **Humans** → [`docs/index.html`](index.html) — self-contained interactive
  report (stats, routing explorer, source bibliography, eval results, decision
  timeline). Published via GitHub Pages when enabled
  (Settings → Pages → deploy from branch → `main`, `/docs`).
- **AI agents** → this file, then [AGENTS.md](../AGENTS.md) for the working
  contract. Read [constitution.md](constitution.md) before making any
  structural change.

## Core documents

| Doc | Type | What it answers |
| --- | --- | --- |
| [constitution.md](constitution.md) | constitution | What this project is, its governance invariants, and the original charter |
| [architecture.md](architecture.md) | architecture | Repo layout, component map, how a question routes through the skill |
| [conventions.md](conventions.md) | conventions | Evidence marking, reference-file structure, sync rules, frontmatter format |
| [glossary.md](glossary.md) | glossary | One home per term: `[config]`, WSD, LR ladder, three-tier eval, 煉丹調參 … |

## Workflows

| Doc | What it covers |
| --- | --- |
| [workflows/add-source.md](workflows/add-source.md) | Adding a new paper/recipe to the skill (the most common change) |
| [workflows/release.md](workflows/release.md) | Repackaging `alchemist-playbook.skill` after changes |
| [workflows/run-evals.md](workflows/run-evals.md) | Running the with/without-skill eval suite in the workspace |

## Decision records

Append-only. Never rewrite a decision — supersede it with a new record that
links back (invariant 4).

| # | Decision | Status |
| --- | --- | --- |
| [0001](decisions/0001-evidence-based-routing-skill.md) | Evidence-based skill with routing-table architecture and verification levels | active |
| [0002](decisions/0002-three-tier-evaluation-framework.md) | Three-tier per-stage evaluation framework (monitoring ≠ evaluation ≠ release gate) | active |
| [0003](decisions/0003-html-artifact-experiment-tracking.md) | Experiment tracking = HTML artifacts + git commits + GitHub Pages | active |
| [0004](decisions/0004-two-road-documentation.md) | Two-road documentation: interactive HTML for humans, indexed markdown for agents | active |
| [0005](decisions/0005-plugin-packaging-tests-ci.md) | Plugin packaging (Claude Code + Codex manifests), MIT license, versioned releases, test suite, CI | active |

## Knowledge homes outside `docs/`

`docs/` documents the *repo*. Training knowledge itself lives in exactly one
place each — never duplicate these facts into `docs/`:

| Fact family | Single home |
| --- | --- |
| Skill trigger, operating principles, consensus baseline, LR ladder, output format | [alchemist-playbook/SKILL.md](../alchemist-playbook/SKILL.md) |
| Stage-specific recipes and evidence | [alchemist-playbook/references/](../alchemist-playbook/references/) (10 files; routing table in SKILL.md) |
| Bibliography + verification status | [alchemist-playbook/references/sources.md](../alchemist-playbook/references/sources.md) |
| Quantitative estimation (FLOPs, Chinchilla, batch, LR ladder) | [alchemist-playbook/scripts/estimate.py](../alchemist-playbook/scripts/estimate.py) |
| Ready-to-edit training configs + experiment dashboard | [alchemist-playbook/assets/templates/](../alchemist-playbook/assets/templates/) |
| Eval definitions and results | [alchemist-playbook-workspace/](../alchemist-playbook-workspace/) |

## Validation

Run the deterministic doc checker and the test suite before committing
structural changes:

```bash
python3 scripts/check_docs.py
```

```bash
python3 -m unittest discover -s tests
```

The checker enforces the indexing, routing-sync, and packaging invariants
listed in [constitution.md](constitution.md). The tests cover `estimate.py`
math, template validity, the SKILL.md ↔ `estimate.py` LR lock, §-citation
integrity, and version sync. CI (`.github/workflows/ci.yml`) runs both on
every push. User-visible changes are recorded in
[CHANGELOG.md](../CHANGELOG.md); the release version lives in the plugin
manifests (see [decision 0005](decisions/0005-plugin-packaging-tests-ci.md)).
