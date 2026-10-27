---
type: decision
title: "Default implementation stack: Nemotron recipes + NeMo-RL + NeMo Gym"
status: active
decided: 2026-10-27
updated: 2026-10-27
---

# 0007 — Default implementation stack: NeMo Gym + Nemotron recipes

## Context

The skill recommends recipe values but left the *implementation* — which
trainer, which configs, which environments, which evaluation harness — to
generic templates (`assets/templates/`) or the user's own stack. On
2026-10-27 the repository owner directed that the skill should train models
with [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/gym) and
[NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) "as much as
possible".

Both repositories were read at pinned commits (Gym `11c874b`, 2026-10-26,
release v0.7.0; Nemotron `1c3e781`, 2026-10-21, package 0.2.0). Nemotron
supplies reproducible stage recipes for the Nemotron 3 / 3.5 family
(Megatron-Bridge pretraining and SFT; NeMo-RL GRPO, DPO, and RLHF; NeMo Gym
evaluation) and a step catalog (`nemotron steps`) with declared artifact
contracts. Gym supplies environments (dataset + agent harness + verifier +
state), 168 resource servers, token-level trajectory capture for external
harnesses, and the reproducibility recipes behind the published Nemotron
evaluations. The repositories also carry repo-internal ablations (continued
pretraining and SFT guidebooks) and a team-authored training FAQ that hold
training evidence of their own.

Decision 0006 established that the user's task supplies tool choices and
that the skill must not impose unconditional tool rules. "As much as
possible" has to be reconciled with that.

## Decision

- **Default route.** When the user has not fixed a trainer, or asks how to
  implement, launch, or evaluate a recipe, the skill maps the recipe onto
  the NeMo stack: Nemotron steps and recipes (Megatron-Bridge, NeMo
  AutoModel, NeMo-RL) for pretraining, continued pretraining, SFT, PEFT,
  DPO, RLVR, and RLHF; NeMo Gym for environments, verifiers, rollouts, and
  evaluation. Recipe values keep their homes in the existing references.
  The new `references/nemo-stack.md` says which config key carries each
  value and which repo entry point runs it.
- **Precedence unchanged.** A user's existing trainer or config still wins
  (decision 0006). The skill then names the NeMo-stack equivalent only
  where it adds a capability the current stack lacks (Gym environments with
  exact token-ID capture, multi-environment RLVR, pinned evaluation
  recipes) and shows the migration path. Gym also integrates with verl and
  Unsloth, so environments can be adopted without changing trainers.
- **Evidence rules unchanged, with pinning.** Values read from repository
  configs and step manifests are `[config]` only together with the
  repository, path, and commit they were read from; README, guide, FAQ,
  and guidebook statements are `[reported]`. Repo-internal ablations are
  folded into the topic references where each fact belongs (pretraining,
  post-training, peft, evaluation), marked `[reported — repo ablation]`,
  never promoted to `[paper]`. A repo default is a fact about the repo, not
  a published training result; the open-data recipes do not reproduce the
  model-card numbers, and the skill says so.
- **Scope honesty.** The skill states the stack's requirements as the
  repositories state them (steps need multi-node GPU clusters on Slurm,
  DGX Cloud Lepton, or Run:ai; recipes are tested on Slurm; Gym needs
  Python 3.13.14) and routes small hardware to AutoModel PEFT or the DGX
  Station recipes. It never invents paths, flags, or defaults: commands
  are verified against the live checkout (`nemotron steps show`,
  `--dry-run`, `gym env resolve`) before they are emitted, and stale pins
  are re-read.
- **Structure.** One new reference file and routing row; the trigger names
  the stack; `sources.md` gains a "Training stack" section; the generic
  templates stay for non-NeMo trainers.

This decision supplements 0006; it supersedes no clause of it.

## Consequences and validation

New skill coverage: implementation mapping for every stage the skill
already advises; SFT, PEFT, DPO, RLVR/GRPO, RLHF, and CPT defaults at
`[config]` grade; Gym environment, data-format, and evaluation guidance;
hardware routing. New training evidence from the repositories lands in
`pretraining.md` §4–5, `post-training.md` §2 and §5, `peft.md` §1,
`evaluation.md` §2 and §5, and `diagnostics.md`. The LR ladder reference
gains a regime note because the Nemotron MoE GRPO configs run above the
dense RL anchors.

Validation: `scripts/check_docs.py` (routing sync, links, description cap,
archive freshness) and the test suite (section citations, version sync,
ladder lock). A new workspace eval case defines the expected implementation
answer; it is defined, not executed. The pinned commits are the validation
boundary: a reader of a newer checkout re-verifies before quoting.
