---
type: glossary
title: Glossary
status: active
updated: 2026-10-27
---

# Glossary

One home per term. Add new terms here, alphabetically within each group,
rather than redefining them inside other docs.

## Project terms

- **煉丹調參** ("alchemy tuning") — Chinese ML-community idiom for the craft
  of training-run hyperparameter tuning; 煉丹 = "refining the elixir"
  (running training), 調參 = "adjusting parameters". The skill's founding
  frame: turn this folklore into cited engineering.
- **The skill** — `alchemist-playbook/`, the source directory packaged into
  `alchemist-playbook.skill`.
- **Routing table** — the topic → reference-file map in SKILL.md that tells
  an agent which evidence file to read before answering.
- **Workspace** — `alchemist-playbook-workspace/`, holding eval definitions
  and per-iteration results; never packaged into the archive.
- **Two roads** — this repo's documentation split: interactive HTML for
  humans (`docs/index.html`), indexed markdown for agents (`docs/index.md`).
  See decision 0004.

## Verification levels

- **`[config]`** — value read directly from an official config file,
  training script, or model card maintained by the releasing org. Highest
  trust.
- **`[paper]`** — value stated in the technical report/paper.
- **`[reported]`** — secondary source or community-standard value; flag when
  precision matters.
- **`[heuristic]`** — no published backing; an explicitly labeled judgment
  call. Also used for practitioner syntheses (e.g., the three-tier eval
  framework).

## Training terms the docs rely on

(Defined here only as needed to read the docs; the skill files are the home
for their full treatment.)

- **Consensus baseline** — a starting optimizer/schedule/batch stack
  synthesized from published text-LLM runs. Home:
  [text-llm-baselines.md](../alchemist-playbook/references/text-llm-baselines.md).
- **LR ladder** — illustrative learning-rate anchors across training
  stages, with separate adapter-tuning regimes. It is a lookup aid, not a
  universal stage ratio. Home: the baseline reference above.
- **NeMo stack** — NVIDIA-NeMo/Nemotron (stage recipes and the `nemotron
  steps` catalog over Megatron-Bridge, NeMo AutoModel, and NeMo-RL) plus
  NVIDIA-NeMo/Gym (environments, verifiers, agent harnesses, evaluation);
  the skill's default implementation route (decision 0007). Home:
  `references/nemo-stack.md`.
- **NeMo Gym environment** — dataset (JSONL tasks) + agent harness +
  verifier with per-task state + model server, wired by one YAML config;
  the unit NeMo Gym evaluates and trains against.
- **Nemotron step** — a named unit of work in the Nemotron repository
  (`nemotron steps run <id>`) with declared consumed and produced artifact
  types; steps chain by matching those types.
- **Three-tier evaluation** — health monitoring ("is the run alive") vs
  capability evaluation ("is it getting stronger without forgetting") vs
  release gate ("can it ship" — a private holdout). Home:
  `references/evaluation.md`.
- **WSD** — warmup–stable–decay LR schedule; alternative to cosine when the
  total token budget is undecided.
- **With/without-skill eval** — the workspace's A/B method: the same prompt
  graded against expert expectations with the skill installed and without;
  the delta is the skill's measured value.
