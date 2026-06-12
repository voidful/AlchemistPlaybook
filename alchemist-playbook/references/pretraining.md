# Pretraining recipes

Verification tags: `[config]` = read from official config/model card,
`[paper]` = stated in the technical report, `[reported]` = secondary source.

## Contents
1. Master recipe table
2. LR schedules: cosine vs linear vs WSD vs multi-phase constant
3. Batch size: token math and ramp schedules
4. Multi-stage curricula: midtraining and annealing
5. Data: the highest-leverage knob
6. Architecture hygiene checklist
7. Pre-launch checklist

## 1. Master recipe table

| Run | Params | Tokens | Peak LR | Warmup | Schedule | Global batch | Seq | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA-1 7B/13B `[paper]` | 6.7/13B | 1.0T | 3.0e-4 | 2000 steps | cosine → 10% | 4M tokens | 2048 | AdamW (0.9, 0.95), wd 0.1, clip 1.0 |
| LLaMA-1 33B/65B `[paper]` | 32.5/65.2B | 1.4T | 1.5e-4 | 2000 | cosine → 10% | 4M | 2048 | same optimizer block |
| Llama 2 (all) `[paper]` | 7–70B | 2.0T | 3e-4 (7/13B), 1.5e-4 (34/70B) | 2000 | cosine → 10% | 4M | 4096 | GQA on 34/70B |
| Llama 3 8B/70B `[paper]` | 8/70B | 15.6T | 3e-4 / 1.5e-4 | 2000 | cosine | 4–8M | 8192 | GQA everywhere |
| Llama 3 405B `[paper]` | 405B | 15.6T | 8e-5 | 8000 | cosine → 8e-7 over 1.2M steps | 4M → 8M → 16M (see §3) | 4096→8192 | 16K H100, 4D parallelism, BF16 MFU 38–43% |
| OLMo-1 1B/7B `[paper]` | 1/7B | 2T/2.46T | 4e-4 / 3e-4 | 5000 steps | linear → 10% | ~4M | 2048 | ε=1e-5 (later fixed in OLMo 2), in-loop eval every 1000 steps |
| OLMo 2 7B `[config]` | 7B | 4T + stage 2 | 3.0e-4 | 8.4B tokens (≈2000 steps @4.2M) | cosine → 10% (calibrated to 5T) | 1024 seq × 4096 ≈ 4.2M | 4096 | AdamW (0.9,0.95), wd 0.1, ε=1e-8, z-loss 1e-5, QK-norm |
| OLMo 2 13B `[config]` | 13B | 5T + stage 2 | 3.0e-4 | 8.4B tokens | cosine → 10% | 2048 × 4096 ≈ 8.4M | 4096 | same stability stack |
| OLMo 2 32B `[config]` | 32B | 6.5T max | 6.0e-4 | 2000 steps | cosine | 8.4M tokens | 4096 | HSDP, full activation ckpt, rank microbatch 16K tokens |
| OLMo 3 7B/32B `[paper]` | 7/32B | ≤5.9T + stages | see report | — | — | — | — | 3 stages: Dolma 3 pretrain → 100B midtrain → 50B (7B) / 100B (32B) long-context |
| SmolLM2 1.7B `[paper]` | 1.7B | 11T | 5.0e-4 | 2000 steps | WSD, final 10% decay | 2M tokens | 2048 | 256 H100, Nanotron; 4-stage data mixture |
| Pythia suite `[paper]` | 70M–12B | 300B | ~1e-3 → ~1.2e-4 by size | 1% steps | cosine → 10% | 2M (1024×2048) | 2048 | controlled data order, 154 checkpoints — best ablation sandbox |
| DeepSeek-V3 `[paper]` | 671B MoE (37B active) | 14.8T | 2.2e-4 | 2K steps | see §2 multi-phase | 3072 → 15360 seqs over first 469B tokens | 4096 | FP8 training, MTP auxiliary loss |
| MiniCPM `[paper]` | 1.2/2.4B | ~1T | 0.01 under their muP-style parametrization (not comparable to SP LRs) | — | WSD, decay ≈ last 10% | ~4M | — | origin of WSD; sharp loss drop during decay phase |
| Kimi K2 `[paper]` | 1T MoE (32B active) | 15.5T | — | — | — | — | — | Muon + QK-Clip (τ=100): zero loss spikes; see stability.md |

Cross-checks worth quoting: tokens-per-parameter ranges from Chinchilla-optimal
~20 (Chinchilla 70B/1.4T) to heavily overtrained ~6500 (SmolLM2). Overtraining
small models is deliberate: inference cost dominates lifetime cost
(LLaMA-1's stated philosophy).

## 2. LR schedules

**Cosine to 10% of peak** — the LLaMA/OLMo default. Choose when total token
budget is fixed in advance. Decay end-point matters: decaying to 0 mid-run
wastes the tail; 10% keeps the tail useful.

**Linear to 10%** — OLMo-1 used it; indistinguishable from cosine in
practice at equal budget. Not worth debating.

**WSD (warmup–stable–decay)** — SmolLM2 `[paper]`, MiniCPM `[paper]`.
Constant LR after warmup; decay only the final ~10% of steps. Two real
advantages: (a) total budget need not be fixed up front — keep training the
stable trunk and branch a decay anytime; (b) the decay branch gives a
scaling-law measurement per branch (MiniCPM). MiniCPM observed most of the
loss drop happens inside the short decay phase. Use WSD when budget is
open-ended or when you plan repeated data-mix experiments off one trunk.

**Multi-phase constant (DeepSeek-V3)** `[paper]` — warmup 2K steps → constant
2.2e-4 until 10T tokens → cosine to 2.2e-5 over 4.3T → constant 2.2e-5 for
333B → constant 7.3e-6 for final 167B. Effectively WSD with a structured
tail. Evidence that "constant + planned anneals" scales to frontier runs.

**Speech differs**: Whisper used linear-to-zero after only 2048 warmup steps;
OWSM v3.1 needed 60K warmup steps (piecewise). See references/speech.md.

## 3. Batch size: token math and ramps

Always compute: `tokens/step = n_gpu × micro_batch × seq_len × grad_accum`
(`scripts/estimate.py batch`).

Verified ramp schedules:

- Llama 3 405B `[paper]`: 4M tokens @ seq 4096 → 8M @ seq 8192 after 252M
  tokens → 16M after 2.87T tokens.
- DeepSeek-V3 `[paper]`: 3072 → 15360 sequences during first 469B tokens,
  then constant.
- PaLM `[paper]`: 512 → 1024 → 2048 sequences in phases.

Why ramp: early in training gradients are large and noisy-but-informative —
small batches buy more updates per FLOP; later, gradient noise dominates and
large batches average it while improving throughput. This is the
critical-batch-size argument (McCandlish et al. 1812.06162) applied as
engineering. For runs ≤ ~30B params a fixed 2–4M token batch is the simpler,
equally defensible choice (LLaMA, OLMo).

Practical rules:
- Keep effective batch constant when re-laying-out hardware (trade micro
  batch against grad-accum).
- If memory forces a smaller micro batch, raise grad_accum, not LR.
- Changing global batch mid-run without a plan = a silent LR change;
  if you double batch, the safest published practice is to keep LR and
  accept slightly slower per-token progress, not to √2/2× the LR by rote
  `[heuristic — no consistent published rule at LLM scale]`.

## 4. Multi-stage curricula: midtraining and annealing

The single biggest recipe shift of 2024–2025: pretraining is no longer one
homogeneous phase.

- **OLMo 2** `[paper/config]`: stage 1 = 3.9T-token web mix (90%+ of budget);
  stage 2 = "Dolmino" anneals on 50–300B of high-quality + math/QA/
  instruction data, run 3× with different seeds/mixes and **model-souped**
  (weight-averaged) into the final checkpoint. 7B: 3×50B; 13B: 3×100B + 1×300B.
- **OLMo 3** `[paper]`: pretrain (≤5.9T, Dolma 3) → midtraining (100B,
  math/code/reasoning-heavy) → long-context extension (50B for 7B / 100B
  for 32B).
- **Llama 3** `[paper]`: long-context extension done in six staged increases
  up to 128K context (405B); final anneal: LR → 0 over the last 40M tokens
  on upsampled highest-quality data. Also used "annealing runs" as a cheap
  data-quality evaluator: anneal a checkpoint on a candidate mix, measure.
- **SmolLM2** `[paper]`: 4-stage mixture rebalancing across 11T tokens —
  math/code upweighted in later stages once general English saturates.

Recipe guidance when a user asks "should I add my domain data during
pretraining?": put small, high-quality, capability-targeted data in a
**late anneal stage with decaying LR**, not uniformly through the run.
For continued-pretraining of an existing base: 10–100B tokens with a fresh
warmup (shorter, e.g., a few hundred steps) and cosine/WSD decay to ~0,
mixing ~10–30% of original-distribution data to limit forgetting
`[heuristic anchored to OLMo 2 stage-2 / Llama 3 anneal practice]`.

## 5. Data: the highest-leverage knob

- FineWeb `[paper]`: 15T tokens from 96 CommonCrawl snapshots; the ablations
  show per-snapshot dedup + quality filtering beat global aggressive dedup.
  FineWeb-Edu (1.3T): an educational-quality classifier filter produced
  outsized MMLU/ARC gains — classifier-based filtering is cheap leverage.
- Multi-epoch: up to ~4 epochs over deduped data is nearly as good as fresh
  data (Muennighoff et al. 2305.16264 `[paper]`); beyond that returns decay
  fast. OLMo 2 13B ran 1.2 epochs of its mix without issue.
- Contamination: decontaminate eval sets from training data; Llama 3 ran
  contamination analyses per benchmark `[paper]`.
- Domain mixes are usually stated, rarely ablated publicly. Llama 3's final
  mix: roughly 50% general knowledge, 25% math/reasoning, 17% code, 8%
  multilingual `[paper]`.

## 6. Architecture hygiene (modern decoder defaults)

RMSNorm (no bias) + SwiGLU + RoPE + GQA; no dropout; no linear bias; untied
embeddings at ≥1B. Stability extras when justified: QK-norm, z-loss 1e-5,
post-sublayer norm reordering (OLMo 2 — norm the output of attention/FFN
before the residual add), ε=1e-8. Details and evidence: references/stability.md.

Init: OLMo 2 uses plain normal(0, 0.02) everywhere `[paper]`, dropping
scaled-init schemes — simplicity won. Tokenizer/vocab: round vocab to a
multiple of 128 for kernel efficiency `[heuristic]`.

## 7. Pre-launch checklist

1. Smoke run: 1–5% of budget at full distributed layout. Must see: smooth
   loss, grad-norm flat after warmup, no memory growth, expected tok/s.
2. Overfit test: 1 batch to ~0 loss (catches masking/shift bugs).
3. In-loop eval every ~1000 steps (OLMo practice): val ppl on a frozen
   slice + 2–3 cheap downstream probes; log tokens-seen, not just steps.
4. Checkpoint cadence sized so a spike costs <1% of budget to rewind.
5. Log data order/seed so any batch window can be replayed (required for
   the PaLM/OLMo skip-batches spike SOP).
6. Record the config in the run dir. Future-you is the next user.
