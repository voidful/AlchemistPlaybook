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

## Install

### Claude Code (recommended)

```
/plugin marketplace add voidful/AlchemistPlaybook
/plugin install alchemist-playbook@alchemist-playbook
```

### Codex

```bash
git clone https://github.com/voidful/AlchemistPlaybook ~/AlchemistPlaybook
```

```bash
ln -s ~/AlchemistPlaybook/alchemist-playbook ~/.agents/skills/alchemist-playbook
```

A `.codex-plugin/plugin.json` manifest is also included for Codex plugin
installs.

### Other environments

Install `alchemist-playbook.skill` in any environment that accepts skill
archives (for example, claude.ai). As a last resort, read
[`alchemist-playbook/SKILL.md`](alchemist-playbook/SKILL.md) directly — the
skill is plain markdown.

## What you can ask

| You say | You get |
| --- | --- |
| "7B pretrain, loss spiked at 80B tokens, grad-norm jumps to 40, betas (0.9, 0.999), eps 1e-5" | Ranked diagnosis, a config diff (β₂→0.95, ε→1e-8) with LLaMA/OLMo citations, and a verify-or-rollback plan |
| "Review my DPO config: beta=5, lr=2e-5, 3 epochs, sigmoid loss" | Flags the LR as ~40× too high and the beta/loss-type mismatch (Zephyr vs Tulu 3), with sources |
| "Fine-tuning Whisper on my domain and it diverges during warmup" | Speech-specific LR/warmup recipe (Whisper/OWSM anchors) |
| "What LR should I use for SFT on an 8B model?" / "幫我設計 1B 模型的 pretraining recipe" | Ladder lookup or full recipe from the consensus baseline, every number tagged `[config]`/`[paper]`/`[reported]`/`[heuristic]` |
| "Which benchmarks should I run during midtraining?" | The three-tier eval framework: monitoring vs capability eval vs release gate |
| "How many GPU-hours does 7B on 2T tokens need?" | `estimate.py` arithmetic (FLOPs, Chinchilla check, batch/warmup math) |

The first three are the eval suite (iteration 1: **15/15** expectations
passed with the skill, **11/15** without it).

For quantitative estimates, run:

```bash
python3 alchemist-playbook/scripts/estimate.py batch --gpus 8 --micro 4 --seq 4096 --accum 8
```

```bash
python3 alchemist-playbook/scripts/estimate.py lr --params 8e9 --stage sft
```

## Integrations (optional)

[research-cards](https://github.com/SungFeng-Huang/research-cards) uses this
playbook as the hyperparameter-citation advisor for its `research-campaign`
skill: every LR/schedule/batch decision in a campaign cites a sourced
recommendation. This skill works standalone; campaigns only cite it. Install
both for autonomous experiment campaigns.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The skill does not trigger | Use its trigger words: 煉丹, 調參, "what LR should I use", "loss spike" — or name the stage (pretrain / SFT / DPO / RLVR / LoRA / speech) |
| Numbers come back without evidence tags | Every value must carry `[config]` / `[paper]` / `[reported]` / `[heuristic]`. Ask for the tags — untagged numbers violate the skill's honesty rules |
| `estimate.py` errors | It needs Python 3 stdlib only. Check the path: `alchemist-playbook/scripts/estimate.py` |
| Plugin install fails | Fall back to the `.skill` archive, or the clone+symlink path above |

## Contribute and validate

To add a source, follow the
[add-source workflow](docs/workflows/add-source.md). Record user-visible
changes in [CHANGELOG.md](CHANGELOG.md). Before you commit, run the doc
checker and the tests:

```bash
python3 scripts/check_docs.py
```

```bash
python3 -m unittest discover -s tests
```

## License

[MIT](LICENSE).
