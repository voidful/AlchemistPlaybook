---
name: alchemist-playbook
description: >-
  Evidence-based model-training recipes and run diagnosis (煉丹調參).
  Use for pretraining, midtraining, SFT, preference tuning, RLVR,
  LoRA/QLoRA, or speech training; training hyperparameters, loss spikes,
  data curation, compute budgets, stage evaluation, and experiment reports.
  Pure inference setup or general prompt writing is outside this skill's
  scope.
---

# Alchemist Playbook

Turn published training runs and official configs into recommendations
that fit the user's model, data, and budget. Keep source evidence separate
from observations about the user's run and from proposed experiments.

## Task handling

- Use the conversation and available configs or logs before asking for
  information. Ask only when a missing fact could change the next decision;
  otherwise state the assumption and continue with a provisional answer.
- Match the requested deliverable. A request to fix a local config includes
  inspecting it, editing it, and performing relevant validation. Carry that
  work through; an advisory question calls for an answer.
- The user's instructions take precedence over this skill's workflow
  defaults. Existing authorization persists; host permissions and required
  repository checks still apply. Prepare useful local work while any
  necessary approval for a training launch or publication remains pending.
- If an instruction here blocks the task, identify the exact skill or
  reference file, quote the applicable instruction, and explain the missing
  decision. Do not turn a preferred workflow into an approval requirement.
- Incorporate corrections and answer side questions while retaining the
  unfinished task. State what was changed, what was verified, and what
  remains unobserved before claiming completion.

These workflow choices draw on the instruction-design sources in
`references/sources.md`; they do not require a particular agent model.

## Training principles

1. **Trace prescribed values.** Name the source run and preserve its
   `[config]`, `[paper]`, or `[reported]` marking. Mark an extrapolation or
   unsupported choice `[heuristic]`. If sources disagree, explain which
   regime fits. Evidence levels are defined in `references/sources.md`.
2. **Diagnose with available evidence.** Use the intake below as context,
   not a prerequisite questionnaire. Separate plausible causes from a
   confirmed cause; identify the observation that would distinguish them.
3. **Check data and loss construction.** Inspect data quality, mixing,
   template alignment, packing, and masking when symptoms support it.
   `references/diagnostics.md` gives a triage order, not a required sequence
   of experiments for every question.
4. **Account for tokens and assumptions.** Use `scripts/estimate.py` for
   its supported FLOPs, batch, budget, and LR lookups. Report input
   assumptions and units. For unsupported calculations, use another
   executable calculation and show the formula; if execution is unavailable,
   provide the formula and label the result unverified.
5. **Match the training regime.** Stage LR bands are starting anchors,
   not universal laws. Full tuning, adapters, speech, MoE, and different
   preference objectives are not interchangeable. Read the matching route
   before transferring a recipe.
6. **Make changes attributable.** Prefer an isolated change per diagnostic
   rerun. If a repair needs coupled changes, explain the dependency and
   avoid attributing the outcome to a single knob. Define a success metric,
   observation window, and rollback or stop condition before launch.
7. **Separate monitoring, evaluation, and release.** Healthy loss curves
   do not establish capability gains. Use frozen capability slices and an
   independent release holdout; see `references/evaluation.md`.
8. **Keep experiment records useful.** Follow the user's existing tracker
   and requested format. For a new report or campaign, the HTML dashboard
   and git workflow in `references/experiment-tracking.md` are defaults.
   Keep provenance and decision history in any format.

## Intake

Recover the facts relevant to the task: training stage; model size,
architecture and checkpoint; data source, volume and cleaning; hardware
and trainer; current optimizer, LR, schedule, batch, sequence length and
precision; observed symptom, timing, and recent changes. Missing logs can
limit certainty without preventing a useful diagnosis or config draft.

## Routing

Read the reference sections needed for the task; the table is not a reading
checklist. Use multiple routes when the problem crosses topics.

| Topic | File |
| --- | --- |
| Text-LLM starting config, consensus baseline, peak LR by size, cross-stage LR comparison | `references/text-llm-baselines.md` |
| Pretraining, cosine/WSD, batch ramp, data mixes, midtraining/annealing (incl. 256K agentic midtraining), distillation, curriculum, edge architectures | `references/pretraining.md` |
| SFT, DPO, SimPO, ORPO, KTO, PPO, GRPO, RLVR, reward models, model merging, reasoning-data curation | `references/post-training.md` |
| Loss spikes, NaN, divergence, grad norms, precision, QK-norm, z-loss, MuonClip | `references/stability.md` |
| LoRA, QLoRA, low-budget fine-tuning | `references/peft.md` |
| ASR, speech translation, speech SSL, Whisper/OWSM, ESPnet | `references/speech.md` |
| Compute/token budgets, Chinchilla, critical batch size, LR–batch scaling, muP, MFU | `references/scaling-and-batch.md` |
| Symptom → cause → fix, monitoring setup, pre-launch checks, incomplete logs | `references/diagnostics.md` |
| Per-stage metrics and benchmarks, retention suites, midtraining decision protocol, independent release gates | `references/evaluation.md` |
| Experiment records, optional HTML dashboards, git checkpoints, publishing an authorized report | `references/experiment-tracking.md` |
| Bibliography and verification status, including workflow-only sources | `references/sources.md` |

## Deliver and verify

For a substantive tuning or debugging answer, cover the diagnosis, proposed
config changes, their evidence, and how to verify or roll back. Use
**Diagnosis / Recipe / Evidence / Verify** headings when they help; a short
question or a requested YAML diff can carry the same essentials directly.
Do not invent a diagnosis for a pure estimate or rewrite a user's format.

For file changes, run the relevant parser, calculator, or existing checks.
Complete repository-required checks, then stop expanding verification unless
new edits, failures, or unresolved concerns justify more. A config parsing
success is not a successful training run. A reviewed eval definition is not
an executed model evaluation. Parallel source reviews or independent evals
can help larger tasks when the host permits them; they are not prerequisites.

## Resources

- `scripts/estimate.py`: `flops`, `chinchilla`, `batch`, and `lr` commands;
  use `--help` for inputs. Keep assumed utilization and prices explicit.
  `references/scaling-and-batch.md` explains the estimates' limits.
- `assets/templates/`: editable pretrain, SFT, DPO, QLoRA, Accelerate and
  DeepSpeed configs. Adapt a matching template to the installed trainer;
  preserve the user's existing config when editing one.
- `assets/templates/experiment_dashboard.html`: optional report starter.
  Copy it for a requested dashboard, edit its data block, and consult
  `references/experiment-tracking.md` for applicable reporting detail.
