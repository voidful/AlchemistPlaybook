# Text-LLM baselines and LR ladder

Read for a text-LLM starting recipe or a stage-to-stage LR comparison.
These anchors describe published regimes; transfer to another setup is a
`[heuristic]`, not a guarantee. Confirm the optimizer, parameterization,
trainable parameters, batch, and objective before using a row. Speech,
MoE, and PEFT may require their own references.

## Contents

1. Consensus baseline
2. Peak pretraining LR
3. Cross-stage LR ladder

## 1. Consensus baseline

This is a starting stack synthesized from LLaMA, OLMo, SmolLM2, and
DeepSeek-V3. Choosing it for a new run is `[heuristic]`; the source column
marks the evidence for the individual anchors. It is not a claim that
every source used the same combination.

| Knob | Default | Source anchor |
| --- | --- | --- |
| Optimizer | AdamW | LLaMA `[paper]`, OLMo `[config]` |
| β₁, β₂ | 0.9, 0.95 | LLaMA `[paper]`, OLMo `[config]`; assess stability in the actual run |
| ε | 1e-8 | OLMo 2 `[config]`; stability context in `stability.md` |
| Weight decay | 0.1 | LLaMA `[paper]`, OLMo `[config]` |
| Grad clip (global norm) | 1.0 | LLaMA `[paper]`, OLMo `[config]` |
| Warmup | ~2000 steps as a starting anchor; 8000 for the 405B run | LLaMA 2/3 `[paper]`; compute warmup tokens from the actual batch, not a fixed percentage |
| Schedule | cosine → 10% of peak; consider WSD for an undecided budget | LLaMA `[paper]`, OLMo 2 `[config]`; WSD choice `[heuristic]` from SmolLM2/MiniCPM |
| Precision | bf16 compute, fp32 grad-reduce and norms | OLMo 2 `[config]`, BLOOM `[paper]` |
| Batch | in tokens: ~2M (≤2B), ~4M (7–13B), 8–16M with ramp (≥70B) | Size bands are `[heuristic]` synthesis; source runs in `scaling-and-batch.md` |
| Dropout | 0 in pretraining; consider ≤0.1 for tiny-data fine-tunes | Pretrain: LLaMA/OLMo `[paper]`; fine-tune cap `[heuristic]` |
| Bias terms | none in linears or norms | LLaMA/OLMo `[paper]` |
| z-loss | off by default; 1e-4 (PaLM) or 1e-5 (OLMo 2) if output logits drift | Default choice `[heuristic]`; PaLM `[paper]`, OLMo 2 `[config]` |

## 2. Peak pretraining LR

Dense model size, AdamW, standard parametrization. The size bands and
transfer ranges are `[heuristic]`; the examples retain their source marks:

| Size | Peak LR | Verified examples |
| --- | --- | --- |
| 100M–2B | 2e-4 – 1e-3 | SmolLM2-1.7B 5e-4 (11T tokens) `[paper]`, OLMo-1B 4e-4 `[paper]`, EvoLM-1B 2e-4 `[config]` |
| 7–13B | 3e-4 | LLaMA-7B/13B `[paper]`, OLMo-1/2 7B and 13B `[config]` |
| 30–70B | 1.5e-4 – 6e-4 | LLaMA-33B/65B and Llama-2-70B 1.5e-4 `[paper]`; OLMo-2-32B 6e-4 `[config]` with a different stability stack and batch |
| ~405B | 8e-5 | Llama 3 405B `[paper]` |
| MoE | per-report | DeepSeek-V3 2.2e-4 (671B/37B-active) `[paper]`; Moonlight/K2 use Muon — see `stability.md` |

For small or on-device targets, consider teacher distillation, curriculum,
and architecture alongside LR (`pretraining.md` §6–7). For small reasoning
models, consult `post-training.md` §9–10 for VibeThinker and §13 for the
MiniCPM teacher pipeline. Reasoning-SFT data curation is in §16. Selection
among these recipes is `[heuristic]` and depends on the target task.

## 3. Cross-stage LR ladder (~8B dense example)

An illustrative comparison `[heuristic]`, not a universal stage ratio:

```
pretrain        3e-4      (LLaMA [paper], OLMo [config])
full SFT        5e-6–2e-5 (Tulu 3 / Zephyr [config], Llama-2-chat [paper])
DPO             5e-7      (Zephyr / Tulu 3 [config]; larger Tulu runs: 2e-7 [config])
PPO/RLVR        3e-7–1e-6 (Tulu 3 PPO [config], DeepSeekMath GRPO [paper])
LoRA/QLoRA SFT  1e-4–2e-4 (QLoRA [paper]; adapter tuning is a separate regime)
```

Use `scripts/estimate.py lr` for a lookup. Check the objective and parameter
regime in the corresponding reference before selecting a value. An SFT LR
above a pretrain LR warrants investigation; it does not by itself prove a
bug.
