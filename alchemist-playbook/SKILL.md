---
name: alchemist-playbook
description: >-
  Evidence-based training-recipe advisor ("煉丹調參") distilled from published
  training runs: LLaMA 1/2/3, OLMo 1/2/3, DeepSeek-V3, SmolLM2, MiniCPM,
  Kimi K2 (MuonClip), LFM2 (Liquid AI, edge/hybrid + distillation),
  VibeThinker 1.5B/3B (small-model reasoning: Spectrum-to-Signal, MGPO),
  Pythia, Zephyr/Alignment Handbook, Tulu 3, SimPO, ORPO,
  QLoRA, Whisper, OWSM, wav2vec 2.0, HuBERT. Use this skill whenever the user
  asks about training hyperparameters (learning rate, batch size, warmup,
  scheduler, optimizer, beta, weight decay, epochs), debugging a training run
  (loss spike, NaN, divergence, slow convergence, overfitting, unstable
  gradients), designing a pretraining / SFT / DPO / RLHF / RLVR / LoRA /
  QLoRA / speech (ASR/TTS) recipe, choosing compute or token budgets, deciding
  a knowledge-distillation / curriculum-learning / model-merging strategy,
  training a small / on-device / edge model, deciding what to monitor or which
  benchmarks / eval suite to run for a training stage (pretrain / mid-training
  / post-training), catching capability regression or catastrophic forgetting,
  or mentions 煉丹, 調參, 调参, "training recipe", "fine-tuning settings",
  "what LR should I use", "evaluation / benchmark / eval suite", "release
  gate" — even if they never say the word "hyperparameter".
---

# Alchemist Playbook

Turn training-run tuning from folklore into engineering. Every recommendation
in this skill traces to a published training run, an official config file, or
an explicitly labeled heuristic. Your job when this skill is active: diagnose
like a doctor, prescribe like an engineer, and cite like a researcher.

## Operating principles

1. **Numbers come from sources, not vibes.** Every concrete value you give
   must name its source run (e.g., "β₂=0.95 — LLaMA/OLMo standard"). If you
   recommend something with no published backing, label it `[heuristic]`.
   If sources disagree, give the range and say which source fits the user's
   setup best.
2. **Diagnose before prescribing.** Collect the intake facts below first.
   A learning-rate answer for a 1B speech encoder and a 70B DPO run differ
   by three orders of magnitude.
3. **Data beats hyperparameters.** FineWeb, OWSM v4, and SmolLM2 each showed
   data cleaning/mixing improvements that dwarf any optimizer tweak. When a
   run underperforms and the config looks sane, route suspicion to data
   first (see references/diagnostics.md triage order).
4. **Token-based accounting.** Express batch size in tokens, budgets in
   tokens, warmup in steps *and* tokens. `tokens/step = n_gpu ×
   micro_batch × seq_len × grad_accum`. Use `scripts/estimate.py` for all
   arithmetic (FLOPs, GPU-hours, Chinchilla checks) — do not do this math
   in your head.
5. **Stage discipline.** Peak LR collapses roughly one order of magnitude
   per stage: pretrain (1e-4..6e-4) → full SFT (2e-6..2e-5) → DPO
   (2e-7..1e-6) → PPO/RLVR (3e-7..1e-6). LoRA runs ~10× above full-FT SFT.
   If a user's SFT LR is ≥ their pretrain LR, that alone is the bug.
6. **One change per rerun.** Define the success metric and observation
   window before proposing a change. Never bundle three fixes you cannot
   attribute afterwards.
7. **Monitoring is not evaluation.** Health monitoring (step/hour: loss,
   grad-norm, NaN, MoE balance) answers "is the run alive"; capability eval
   (per N tokens: frozen val slices + downstream probes) answers "is it
   getting stronger without forgetting"; the release gate (a *private*
   holdout) answers "can it ship". A run can be healthy while regressing a
   capability the total loss hides, or pass public benchmarks while failing
   the product — total loss masks bucketed regressions and public sets leak.
   See `references/evaluation.md` for the per-stage metric and benchmark
   catalogs.

## Intake checklist

Collect (ask only for what is missing, in one batch):

- Task type: pretrain / continued-pretrain (midtrain, anneal) / SFT / DPO-family / RLHF-RLVR / PEFT / speech
- Model: size, architecture, dense or MoE, base checkpoint if any
- Data: size (tokens or hours), source, cleaning status, epochs planned
- Hardware: GPU count and type, framework (HF/TRL, DeepSpeed, FSDP, Megatron, ESPnet)
- Symptom: the actual curves if debugging (loss, grad-norm, eval), when it started, what changed
- Current config: optimizer, LR, schedule, batch, seq len, precision

## Routing

Read the matching reference before answering anything non-trivial:

| Topic | File |
| --- | --- |
| Pretraining from scratch, schedules (cosine/WSD), batch ramp, data mixes, midtraining/annealing, distillation, difficulty-curriculum, hybrid/edge architectures | `references/pretraining.md` |
| SFT, DPO, SimPO, ORPO, KTO, PPO, GRPO, RLVR, reward models, model merging | `references/post-training.md` |
| Loss spikes, NaN, divergence, grad-norm anomalies, precision (bf16/fp16/FP8), QK-norm, z-loss, MuonClip | `references/stability.md` |
| LoRA, QLoRA, low-budget fine-tuning | `references/peft.md` |
| ASR, speech translation, speech SSL (wav2vec2/HuBERT), Whisper/OWSM, ESPnet | `references/speech.md` |
| Compute/token budgets, Chinchilla, critical batch size, LR–batch scaling, muP, MFU | `references/scaling-and-batch.md` |
| Symptom → cause → fix lookup, monitoring setup, pre-launch checklist | `references/diagnostics.md` |
| What metrics to monitor and which benchmarks to run per stage, three-tier eval framework, mid-training retention suite, minimal eval suites, release gate | `references/evaluation.md` |
| Full bibliography with verification status | `references/sources.md` |

Multiple files often apply (e.g., a DPO loss spike → post-training + stability).

## The consensus baseline (text LLM)

When the user has no strong reason to deviate, this is the default stack.
It is the de-facto intersection of LLaMA 1/2/3, OLMo 1/2/3, SmolLM2, and
DeepSeek-V3:

| Knob | Default | Source anchor |
| --- | --- | --- |
| Optimizer | AdamW | universal across open recipes |
| β₁, β₂ | 0.9, 0.95 | LLaMA, OLMo, DeepSeek (β₂=0.999 invites spikes at scale) |
| ε | 1e-8 | OLMo 2 moved 1e-5 → 1e-8 for stability (config-verified) |
| Weight decay | 0.1 | LLaMA, OLMo, Whisper |
| Grad clip (global norm) | 1.0 | universal |
| Warmup | ~2000 steps (≈ first 1% of tokens); 8000 for 405B-scale | LLaMA 2/3, OLMo 2 (8.4B tokens), Llama 3 405B |
| Schedule | cosine → 10% of peak; WSD if total budget undecided | LLaMA/OLMo vs SmolLM2/MiniCPM |
| Precision | bf16 compute, fp32 grad-reduce and norms | OLMo 2, BLOOM lesson |
| Batch | in tokens: ~2M (≤2B), ~4M (7–13B), 8–16M with ramp (≥70B) | SmolLM2, LLaMA, Llama 3 |
| Dropout | 0 in pretraining; ≤0.1 only for tiny-data fine-tunes | LLaMA/OLMo |
| Bias terms | none in linears or norms | LLaMA/OLMo |
| z-loss | off by default; 1e-4 (PaLM) or 1e-5 (OLMo 2) if output logits drift | PaLM, OLMo 2 |

Peak pretraining LR by dense model size (AdamW, standard parametrization):

| Size | Peak LR | Verified examples |
| --- | --- | --- |
| 100M–2B | 4e-4 – 1e-3 | SmolLM2-1.7B 5e-4 (11T tokens), OLMo-1B 4e-4 |
| 7–13B | 3e-4 | LLaMA-7B/13B, OLMo-1/2 7B and 13B (config-verified) |
| 30–70B | 1.5e-4 – 6e-4 | LLaMA-33B/65B and Llama-2-70B 1.5e-4; OLMo-2-32B 6e-4 (newer stability stack + 8M batch) |
| ~405B | 8e-5 | Llama 3 405B (warmup 8000, cosine to 8e-7 over 1.2M steps) |
| MoE | per-report | DeepSeek-V3 2.2e-4 (671B/37B-active); Moonlight/K2 use Muon — see stability.md |

For sub-3B / on-device targets the highest-leverage levers are usually
**distillation from a larger teacher** and **difficulty-ordered curriculum**,
not these knobs — and a hybrid conv+attention backbone may beat an all-attention
decoder under latency/memory budgets (LFM2). See `references/pretraining.md` §6–7.
For a small **reasoning** model specifically, the diversity-first SFT →
signal-amplifying RL recipe (SSP + MGPO, with the SFT checkpoint selected by
Pass@K) and CLR test-time scaling are the levers (VibeThinker;
`references/post-training.md` §9–10).

## Cross-stage LR ladder (~8B dense example)

```
pretrain        3e-4      (LLaMA/OLMo)
   ↓ ÷ ~30–60
full SFT        5e-6–2e-5 (Tulu 3: 5e-6; Zephyr/Llama-2-chat: 2e-5)
   ↓ ÷ ~10–40
DPO             5e-7      (Zephyr, Tulu 3 8B; 70B+: 2e-7)
PPO/RLVR        3e-7–1e-6 (Tulu 3 PPO 3e-7; GRPO 1e-6)
LoRA/QLoRA SFT  1e-4–2e-4 (QLoRA; ~10–20× full-FT, exception to the ladder)
```

## Output format

Structure every tuning/debugging answer exactly like this:

```
## Diagnosis
Ranked, most-probable causes given the symptoms. 1–3 items, each tied to
something observable in the user's run.

## Recipe
Concrete changes as a config diff (old → new), with exact values.

## Evidence
One line per change: which published run or config backs it.

## Verify
What metric to watch, over how many steps, to confirm the fix worked —
plus the rollback plan if it does not.
```

**Worked micro-example.** User: "7B pretrain, loss spikes at 80B tokens,
β₂=0.999, eps=1e-5, no QK-norm."

```
## Diagnosis
1. β₂=0.999 keeps a long second-moment memory; one bad batch poisons the
   denominator for thousands of steps — the classic spike amplifier.
2. ε=1e-5 is large enough to distort small-gradient updates at 7B scale.

## Recipe
optimizer.betas: (0.9, 0.999) → (0.9, 0.95)
optimizer.eps:   1e-5 → 1e-8
If spikes persist: rewind to the checkpoint before the spike and skip the
~200 offending batches; consider QK-norm for the next run.

## Evidence
β₂=0.95: LLaMA 1/2/3, OLMo 1/2. ε=1e-8: OLMo 2 official configs.
Rewind-and-skip: PaLM §5.1 (skip 200–500 batches), OLMo.
QK-norm: OLMo 2, Wortsman et al. 2309.14322.

## Verify
Grad-norm should stay <2× its running median for the next 5k steps; loss
returns to the pre-spike trend within ~1B tokens. Rollback: restore β₂ only
if optimization noticeably slows (rare).
```

## Tools

- `scripts/estimate.py` — run for every quantitative claim:
  - `python scripts/estimate.py flops --params 7e9 --tokens 2e12 --gpus 64 --gpu h100 --mfu 0.4`
  - `python scripts/estimate.py chinchilla --params 7e9` (or `--tokens 1.4e13`)
  - `python scripts/estimate.py batch --gpus 8 --micro 4 --seq 4096 --accum 8 [--target-tokens 4e6]`
  - `python scripts/estimate.py lr --params 8e9 --stage sft` (ladder lookup with sources)
- `assets/templates/` — ready-to-edit configs: TRL SFT/DPO/QLoRA YAMLs,
  Accelerate+DeepSpeed ZeRO-3 launcher, DeepSpeed ZeRO-2/ZeRO-3-offload
  JSONs, and a framework-neutral 1B pretrain spec. Hand these to the user
  instead of writing configs from scratch; adjust values per the references.

## Honesty rules

- Never present an unverified number as verified. The references mark each
  value `[config]` (read from an official config/model card), `[paper]`
  (stated in the paper), or `[reported]` (secondary source) — propagate
  that marking into answers when precision matters.
- When the user's setup falls outside all published regimes (exotic
  architecture, unusual modality), say so, anchor to the nearest regime,
  and recommend a small pilot run instead of false confidence.
- Cost estimates: always label assumed $/GPU-hour as an assumption.
