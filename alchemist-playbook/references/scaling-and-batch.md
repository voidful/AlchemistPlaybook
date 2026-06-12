# Scaling, compute budgets, and batch–LR coupling

Verification tags: `[config]`, `[paper]`, `[reported]`.

## Contents
1. Token budgets: Chinchilla and deliberate overtraining
2. Compute math: 6ND, MFU, GPU-hours
3. Critical batch size
4. LR–batch–size coupling: what is actually known
5. muP and hyperparameter transfer
6. WSD as a scaling-law instrument
7. Worked examples with estimate.py

## 1. Token budgets

Chinchilla `[paper 2203.15556]`: compute-optimal training puts tokens ≈
**20× params** (their 70B/1.4T point; derived three ways — fixed-model
curves, isoFLOP valleys, parametric loss fit). But compute-optimal ≠
deployment-optimal: if the model will serve many queries, overtrain past
Chinchilla because inference cost dominates lifetime cost (LLaMA-1's
explicit rationale `[paper]`). Reference points: LLaMA-7B 1T (~140×);
Llama 3 8B 15.6T (~1950×); SmolLM2-1.7B 11T (~6500×). Loss still improved
log-linearly at these ratios — undertraining is the common error, not
overtraining. Rule for users: never train a from-scratch model below ~20
tokens/param; go far above it for small deployed models.

## 2. Compute math

Training FLOPs ≈ **6 × N × D** (N params, D tokens; forward 2ND + backward
4ND). GPU-hours = FLOPs / (peak_FLOPs × MFU × 3600).

MFU anchors: Llama 3 405B achieved 38–43% BF16 MFU on up to 16K H100s
`[paper]`; a well-tuned single-node dense run reaches 45–55% `[reported]`;
below ~30% on ≤64 GPUs means a systems problem (input pipeline, comm
overlap, activation checkpointing granularity), not a model problem.
Peak dense BF16: H100 ≈ 989 TFLOPs, A100 ≈ 312 TFLOPs.

Always compute with `scripts/estimate.py flops` and label any $ figure as
an assumption.

## 3. Critical batch size

Gradient-noise-scale theory (McCandlish et al. 1812.06162 `[paper]`):
below the critical batch size, doubling batch ≈ halves steps (efficient);
above it, returns vanish — you spend FLOPs without learning faster. The
noise scale grows as loss falls, which is why frontier runs **ramp** batch
mid-training rather than starting huge: Llama 3 405B 4M→8M→16M tokens
`[paper]`; DeepSeek-V3 3072→15360 sequences over the first 469B tokens
`[paper]`; PaLM 512→1024→2048 `[paper]`. Symmetric advice: a tiny global
batch (<0.5M tokens) on a big model wastes wall-clock and produces noisy
grad-norms — raise accumulation first.

## 4. LR–batch coupling

Honest status: **no universal rule survives at LLM scale.** Linear-scaling
(LR ∝ batch) is a small-batch SGD result, not an AdamW-at-2M-tokens
result. What the published record supports:

- Within the 2–8M token band, flagship runs kept LR fixed while ramping
  batch (Llama 3, DeepSeek-V3) `[paper]` — mild under-scaling of LR is
  safe; aggressive LR raises with batch are where spikes live.
- DeepSeek LLM scaling laws `[paper 2401.02954]` fit optimal batch rising
  and optimal LR falling slowly with compute budget — direction, not a
  formula to copy blind.
- OLMo 2 13B doubled batch vs 7B (8.4M vs 4.2M tokens) at the **same**
  peak LR 3e-4 `[config]` — a clean published data point.

Default advice: pick batch from hardware+noise considerations, pick LR from
the size ladder (SKILL.md), and only co-tune them in a pilot sweep if the
run is expensive enough to justify it.

## 5. muP / hyperparameter transfer

muP (Tensor Programs V, 2203.03466 `[paper]`): with width-dependent
init/LR scaling, the optimal LR becomes approximately width-invariant —
tune on a small proxy, transfer to the big model. MiniCPM used a
muP-style parametrization ("wind-tunnel" sweeps at tiny scale; base LR
0.01 in their parametrization) `[paper]`. Caution for users: muP LRs are
**not comparable** to standard-parametrization LRs — never mix tables.
If a user is not using a muP-aware framework, the practical alternative is
the Wortsman proxy-sweep method (stability.md §8).

## 6. WSD as a scaling-law instrument

With cosine, measuring loss at budget B requires a full run decayed for B
— k budgets cost k full runs. With WSD (MiniCPM `[paper]`), one stable
trunk + k short decay branches (~10% each) yields k decayed-loss
measurements at ~(1 + 0.1k)× cost. This is the cheapest credible way for a
small lab to fit its own data-mix scaling curves; SmolLM2 ran its 11T-token
production training this way (stable LR 5e-4, final-10% decay) `[paper]`.

## 7. Worked examples

```
# 7B on 2T tokens, 64×H100 @ 40% MFU
python scripts/estimate.py flops --params 7e9 --tokens 2e12 --gpus 64 --gpu h100 --mfu 0.40
# → 8.4e22 FLOPs, ~59K GPU-hours (~2458 GPU-days) → ~38 days wall-clock

# Is 1.7B/11T over Chinchilla?
python scripts/estimate.py chinchilla --params 1.7e9
# → optimal ≈ 34B tokens; 11T = ~324× over → deliberate overtraining regime

# tokens/step sanity
python scripts/estimate.py batch --gpus 8 --micro 4 --seq 4096 --accum 8
# → 1,048,576 tokens/step; warmup 2000 steps ≈ 2.1B tokens
```
