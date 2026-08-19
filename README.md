# Alchemist Playbook

This repository contains **alchemist-playbook**, an evidence-based
training-recipe skill (煉丹調參) for AI coding agents. The skill distills
more than 70 published sources — LLaMA 1–3, OLMo 1–3, DeepSeek-V3, SmolLM2,
Kimi K2, GLM-5, Zephyr, Tulu 3, QLoRA, Whisper, OWSM, and more. It gives a
diagnostic workflow for pretraining, SFT, DPO/RLHF/RLVR, LoRA/QLoRA, and
speech training. Each number comes from an official config, a paper, or a
labeled heuristic.

## Documentation — two roads

| Audience | Start here |
| --- | --- |
| **Humans** | [Interactive report](docs/index.html) (`docs/index.html`). It shows the stats, the routing table, the source list, the eval results, and the decision history. To publish it, use GitHub Pages: main branch, `/docs` folder. |
| **AI agents** | [`docs/index.md`](docs/index.md). It links to the [constitution](docs/constitution.md), [architecture](docs/architecture.md), [conventions](docs/conventions.md), [glossary](docs/glossary.md), workflows, and decision records. The agent contract is [AGENTS.md](AGENTS.md). |

## Contents

- [`alchemist-playbook/`](alchemist-playbook/SKILL.md) — the skill source.
  It contains SKILL.md, 10 reference files, the `estimate.py` calculator,
  and 8 config templates with an experiment dashboard.
- `alchemist-playbook.skill` — the packaged archive. It is a build output.
  Do not edit it. See the [release workflow](docs/workflows/release.md).
- [`alchemist-playbook-workspace/`](alchemist-playbook-workspace/evals/evals.json) —
  the eval suite. Iteration 1 result: **15/15** expectations passed with the
  skill, **11/15** without it.
- [`docs/`](docs/index.md) — the repository documentation (both roads).

## Use

Install `alchemist-playbook.skill` in an agent environment that accepts
skill archives (for example, Claude Code or Codex). As an alternative, read
[`alchemist-playbook/SKILL.md`](alchemist-playbook/SKILL.md) directly.

For quantitative estimates, run:

```bash
python3 alchemist-playbook/scripts/estimate.py batch --gpus 8 --micro 4 --seq 4096 --accum 8
```

```bash
python3 alchemist-playbook/scripts/estimate.py lr --params 8e9 --stage sft
```

## Contribute and validate

To add a source, follow the
[add-source workflow](docs/workflows/add-source.md). Before you commit,
run the doc checker:

```bash
python3 scripts/check_docs.py
```
