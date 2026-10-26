# Pretraining recipes

Verification tags: `[config]` = read from official config/model card,
`[paper]` = stated in the technical report, `[reported]` = secondary source.

## Contents
1. Master recipe table
2. LR schedules: cosine vs linear vs WSD vs multi-phase constant
3. Batch size: token math and ramp schedules
4. Multi-stage curricula: midtraining and annealing
5. Data: the highest-leverage knob
6. Distillation and difficulty-curriculum for small / edge models
7. Architecture hygiene checklist (incl. hybrid conv+attention)
8. Pre-launch checklist

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
| LFM2 dense `[paper]` | 0.35–2.6B | 10T + 1T mid | not disclosed (distillation-driven) | — | accel. decay in midtrain | — | 4096 → 32768 | AdamW (0.9,0.95) wd 0.1 clip 1.0; hybrid gated short-conv + GQA (6–8 attn of 16–30 layers); distilled from LFM1-7B; ~10% input dropout; see §6 |
| LFM2-8B-A1B MoE `[paper]` | 8B (1B active) | 12T + 1T mid | not disclosed | — | accel. decay in midtrain | — | 4096 → 32768 | 32 experts/layer, top-4; same hybrid backbone; edge/on-device target |
| GLM-5 `[paper]` | 744B MoE (40B active) | 28.5T total = 27T pre + 1.55T mid + 20B DSA adapt | not disclosed (Muon family) | — | — | — | 4K → 200K | 256 experts, 80 layers (fewer layers to cut EP comm); MLA-256 (head dim 192→256, heads −1/3 for cheap decoding); Muon **Split** (stability.md §5); MTP with 3 parameter-shared layers (accept len 2.76); DSA sparse-attention retrofit (§7); INT4 QAT during SFT |
| Nemotron 3.5 agentic midtrain — stage only `[paper 2610.22019]` | 30B-A3B (Lightning) / 120B-A12B (Super) MoE | up to 500B / 65–280B midtrain tokens on a converged base | 5e-5 / 1e-5 | 32K / 30K sequences (≈500 steps at batch 64) | **constant** after warmup — no decay inside the stage | 64 seqs × 256K ≈ 16.8M tokens | 262,144 packed, inter-document masking | Adam (0.9, 0.95), wd 0.1, clip 1.0, BF16, 2 MTP layers kept (loss weight 0.1 / 0.3), 20% pretraining replay; pretraining LRs are not in this paper; recipe + ablations in §4 |
| EvoLM `[config]` | 0.5/1/4B | 10–320B (swept) | 2.5e-4 / 2e-4 / 1.5e-4 | ratio 0.1 | cosine → 10% | 512 seqs (0.5/1B) / 1024 (4B) × 2048 ≈ 1–2M tokens | 2048 | AdamW (0.9,0.95) wd 0.1 clip 1.0, bf16-mixed, FineWeb-Edu; 100+-run controlled full-pipeline ablation (LLaMA arch, GQA 4 groups); downstream saturation ~80–160 tok/param (scaling-and-batch.md §1); CPT replay ablation in §4 |

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
- **GLM-5** `[paper]`: 27T-token pretrain @4K (code/reasoning prioritized
  early) → three-stage long-context midtrain: **32K (1T tokens) → 128K
  (500B) → 200K (50B)**, upsampling long documents and synthetic agent
  trajectories in later stages. Agentic SWE midtrain data = repo-level
  concatenation of issues + PRs + diffs + relevant files (~10M issue–PR
  pairs, ~160B unique tokens). Two transferable findings: the extra 200K
  stage *improved 128K performance* (training past the target length helps
  at the target length), and long-context gains tracked data *diversity*
  (NextLong/EntropyLong-style synthetic long-range dependencies + MRCR-like
  multi-turn recall data at the 200K stage).
- **MiniCPM5-1B** `[paper — OpenBMB/MiniCPM README + official recipe
  figure; no dedicated arXiv report — do not cite the MiniCPM4 paper
  (2506.07900) for it]`: the WSD lineage (MiniCPM, §2) grown into a
  staged decay curriculum that folds the long-context extension *into the
  decay phases*: stable training → **short decay @4K ctx (200B tokens) →
  long decay @32K (75B) → long decay @128K (25B)** → base checkpoint →
  mid-training (200B) → 200B deep-thinking SFT + 200B hybrid-thinking SFT.
  Data: Ultra-FineWeb / Ultra-FineWeb-L3 / UltraData-Math under the
  "UltraData tiered data management" scheme (arXiv 2602.09003). Stable-
  phase token count, optimizer, and all LRs are **not stated**.
  Architecture `[config — HF config.json]`: 1.08B total / 0.68B
  non-embedding, 24 layers, GQA 16/2 heads, vocab 130,560, rope_theta 5e6,
  ctx 131,072. Result context: averages 42.57 vs Qwen3-0.6B 26.77 /
  LFM2.5-1.2B 35.61 — the strongest published ≤1B recipe as of mid-2026;
  its RL+OPD post-training is the other half of the story
  (post-training.md §13).

- **EvoLM (the controlled CPT data point)** `[config/paper 2506.16029]`:
  CPT re-warms the LR to the **full pretraining peak** (2e-4, warmup ratio
  0.1, fresh cosine to 10%) — config-verified, not a reduced-LR restart.
  Replay is **non-monotonic**: on a 50B-token FineMath CPT over a
  160B-token 1B base, **8B tokens of FineWeb-Edu replay (16% of the CPT
  mix, ≈5% of the pretraining budget) beat no replay AND beat both smaller
  (1.6B) and larger (16B) replay** on downstream math after SFT (Pass@1
  21.01 vs 19.27 / 16.21 / 15.22). More replay is not automatically safer —
  it displaces target-domain tokens.

**Agentic-coding midtraining — the Nemotron 3.5 open recipe** `[paper
2610.22019, v1 2026-10-15]`. A dedicated stage between a converged
short-context base (Nemotron pretrains at 8K and sees almost no agent
interactions) and SFT (mostly synthetic chat; very long SFT stages erode
later RL trainability — the team's own observation, citing Liu 2026b /
Kang 2025 / Zhang 2026 `[reported via the paper]`). It trains at **256K**
context for 65–500B tokens so whole repositories, commit histories, and
long agent traces fit in one sequence; learning that would otherwise
inflate SFT moves here, so SFT stays short (33–100B tokens). Developed on
Nemotron 3.5 Lightning-30B-A3B, then scaled to Super-120B-A12B.

*Blend by share of training tokens* (Table 1; ≈380B unique natural tokens,
so a 500B run sees each natural source about once):

| Source | Unique tokens | Share | Epochs @500B |
| --- | --- | --- | --- |
| Issue–PR — merged PRs with linked issues, review discussion, and observed patch evolution; 12.1M PRs from ≥10-star repos; an LLM judge checks issue↔patch match with seven failure modes (revert, incidental link, wrong issue, partial fix, no clear action, underspecified, PR does far more); PR-only examples drop README-only or ≤3-line changes | 216.9B | 40.8% | 0.95 |
| Repositories — only import-graph-*connected* repos (≥20% of code files import another file in the repo; 10% for Ruby); README first, then depth-first over the import graph so dependent files sit together; long repos split between connected components rather than truncated; >2M-token repos dropped; Python upweighted 23.5% → 31% | 104.1B | 20.0% | 0.97 |
| Agentic traces — >1M trajectories, ~500K distinct queries, 30M tool calls from >10 scaffolds (Codex 19%, OpenHands 14%, Gemini CLI 12%, Terminus 12% …); the best trajectories are *reserved for SFT*, midtraining gets the rest incl. older-teacher traces; the only source rendered in the chat template | 39.4B | 10.0% | 1.28 |
| Diff data — multi-file edits (import-linked files before → after a merged PR) and code-evolution documents (initial file → real commits oldest-to-newest → final state; median 3 / max 59 commits, median ≈6K tokens) | 16.0B | 3.0% | 0.95 |
| Build & CI logs — LogHub archives + GitHub Actions run logs / workflow histories cut into 120 context lines → 16 target lines, plus 632 frontier-model-written diagnoses | 6.5B | 1.2% | 0.95 |
| Pretraining replay — the base model's final-phase blend | ~2.0T | 20.0% | 0.05 |
| Long-context data | 67.4B | 5.0% | 0.37 |

Traces are deliberately capped at **10–15%** of the blend — more would turn
the stage into a long SFT and risk the RL-trainability loss above.
Everything but traces stays in natural format with no chat template.
Hygiene applied to every source: permissive licenses only; decontamination
against SWE-bench and Terminal-Bench (diff data also removes SWE-bench
Verified/Pro, Multi-SWE-bench, Terminal-Bench, and LoCoDiff repositories;
repositories drop files containing HumanEval / MBPP / LiveCodeBench
solutions); secret redaction; and **loss-masking of anything the model
should condition on but not imitate** — file-path headers, diff headers,
commit hashes, authors, dates, timestamps, repeated delimiters, license
blocks. Trace filtering removes solution leakage through git history or
web lookup, unfinished runs, malformed tool calls, and repetitive
file-inspection loops (≈1% of an early snapshot).

*Training* (§1 row; Table 17): 256K packed sequences with inter-document
masking, batch 64 sequences, a short warmup, then a **constant** LR — 5e-5
on the 30B-A3B ("favors agentic gains") and 1e-5 on the 120B-A12B ("favors
preserving general capabilities"); both keep their MTP layers; Megatron-LM
on GB300. In a mature pipeline the stage sits **before the final
long-context stage**; for evaluation simplicity most runs here started
from the 1M long-context checkpoint instead.

*Ablations worth transferring* (all measured after *identical* SFT; single
runs):
1. **Traces carry most of the post-SFT gain.** 100B tokens on the 30B-A3B:
   natural data alone gave +0.8 SWE-bench Verified pass@1 after full SFT
   (59.7 → 60.5) and a Terminal-Bench *regression* (22.9 → 18.5); the full
   blend gave +3.8 (63.5) and +2.2 (25.1). Natural data still earns its
   75%: it drives file-state tracking (LoCoDiff exact match 1.3% → 29.8%
   from diff data alone in a 10B-token comparison), cross-file modeling,
   and harness robustness.
2. **Mask tool responses; train on assistant turns and tool calls.**
   Putting environment outputs in the loss cost ≈10 points of SWE-bench
   Verified pass@1 (27.5 vs 37.8; pass@3 41.6 vs 53.0) — long, noisy file
   dumps and test logs take capacity from the agent's own actions.
3. **Replay 20%**: no general-capability regression through 65B tokens on
   the 120B-A12B (knowledge / math / code / commonsense groups all within
   1.2 points of the stage input); **40% replay gave no clear benefit**.
4. **LR**: a moderate increase over the decayed end-of-pretraining LR,
   tuned against the general-capability groups — the Table 11 sweep is in
   the continued-pretraining guidance below. The 30B-A3B kept 5e-5 anyway,
   trading general capability for agentic gains.
5. **Midtraining doubles as a long-context stage.** From the 8K base, 100B
   tokens at 256K alone landed within ≈5 RULER points of the dedicated
   long-context model up to 128K (14 behind at 256K); midtrain → long-
   context then **beat** long-context-only at every length from 64K to
   256K (86.6 / 83.8 / 79.7 vs 85.9 / 81.3 / 76.6). Ordering matters the
   other way too: midtraining a 1M checkpoint at 256K dropped RULER@1M
   79.9 → 33.0 within 25B tokens, and a ≈9B-token long-context-only
   touch-up restored 81.0 — training past the target length helps at the
   target (GLM-5's 200K finding above) and training *below* it hurts until
   you re-extend.

*What it bought* (identical full SFT, 100B tokens): 30B-A3B at 500B
midtrain tokens — SWE-bench Verified 60.7 → 64.5, Multilingual 55.8 →
62.8, Pro 32.0 → 35.5, Terminal-Bench 2.0 22.6 → 23.6, with pass@k up
2.3–4.7 points on all four (it solves *more*, not just more reliably);
most of the Verified/Pro gain had arrived by 100B, Multilingual kept
climbing to 500B, Terminal-Bench is noisy and peaked at 300B. 120B-A12B at
280B tokens — DeepSWE 11.7 → 15.7, Verified 72.9 → 74.3, BrowseComp +3;
Multilingual/Pro −1.1 / −2.1, which the authors treat as noise (run-to-run
≈1 point on Verified/Pro, ≈2 on DeepSWE). Every agent harness improved on
both models (mean +8.6 Verified / +6.8 Terminal-Bench on the 120B after
short SFT; +11.6 Verified across six harnesses on the 30B), with the
largest gains where the base was weakest. The agent also became more
economical with context: SWE-bench Pro rollouts that failed by filling the
256K window fell 91 → 13 of 2,193 (4.1% → 0.6%), turn-cap hits on Verified
11 → 1. A 20-step GRPO check kept the ≈2-point Verified lead and held
Terminal-Bench pass@8 where the base's fell (post-training.md §5).

*Honesty notes*: every comparison is a single run (1–2-point differences
are within noise, Terminal-Bench especially); there is no equal-token
comparison against longer pretraining or longer SFT; Nemotron models only,
agentic coding only; the pretraining peak LR is not in this paper, so
"moderate increase" is relative to the decayed end-of-pretraining LR; data
scripts and training-set metadata were "planned for a future version" at
v1. Context from the paper's related work `[reported — not independently
verified here]`: Qwen3-Coder-Next midtrains at 256K on trillions of tokens
incl. 600B repository-level tokens across 370 languages; Kimi-Dev uses
≈150B tokens of diff patches, commit sequences, and synthetic agent
interactions and then needs far fewer SFT tokens; daVinci-Dev matches it
with trajectory-style data at under half the tokens; GLM-5's ≈160B SWE
tokens are the bullet above. How to *judge* such a stage: evaluation.md §4.

Recipe guidance when a user asks "should I add my domain data during
pretraining?": put small, high-quality, capability-targeted data in a
**late anneal stage with decaying LR**, not uniformly through the run.
For continued-pretraining of an existing base: 10–100B tokens with a fresh
warmup and cosine/WSD decay — or, when a long-context stage and SFT still
follow, a short warmup to a **constant** moderate LR with no decay inside
the stage (Nemotron 3.5 midtraining `[paper]`, 65–500B tokens) — mixing
original-distribution replay to limit forgetting. Two controlled replay
points: EvoLM (1B dense, 50B math CPT) says ~5–15% replay, tuned, and
warns against reflexively going higher; Nemotron (30B-A3B / 120B-A12B MoE,
agentic midtraining) ran **20%** with no general-capability regression
through 65B tokens and found **40% no better** `[paper]`. The older 10–30%
band from OLMo 2 stage-2 / Llama 3 anneal practice is the upper limit, not
the default `[EvoLM config-verified; Nemotron paper; band remains
heuristic]`.

**CPT learning rate — two published regimes, not one rule.** EvoLM
re-warmed a small dense base to its *full* pretraining peak (2e-4).
Nemotron, starting from a converged large MoE whose pretraining ended at a
small decayed LR, used only a *moderate* increase and showed why `[paper —
Table 11, 30B-A3B, 100B tokens, 20% replay]`: against the stage input
(MMLU 79.4 / MMLU-Pro 64.3 / MATH-500 79.7), a constant 1e-5 landed at
78.5 / 66.3 / 78.8, warming up to 3e-5 at 76.7 / 61.5 / 74.3, and 5e-5 at
76.8 / 63.3 / **61.7** — an 18-point MATH-500 loss that replay did not
prevent — even though 1e-5 had given smaller post-SFT agentic gains in
their early runs. `[heuristic]` When preserving general capability is a
hard constraint on a large converged base, start in the 1e-5 class and
raise only while the general-capability groups hold; when the base is small
and the CPT domain is the whole goal, EvoLM's full re-warm is the published
pattern.

**Evaluating a midtraining/anneal stage is a Pareto problem, not a single
number.** The success criterion is `target capability ↑ − general capability
forgotten − safety regression − added cost`, so a stage that boosts the
target domain while silently regressing general ability, Chinese, safety, or
tool use is a *failure* even if the target benchmark rose. Every midtraining
run must own a fixed **retention suite** (general knowledge + Chinese + math +
code + long-context + safety + target domain) measured against the *stage
input* checkpoint. Catalog and minimal mid-training suite: `references/evaluation.md` §4.
For agentic midtraining the Nemotron protocol is the operational form of
this rule: identical short SFT after every candidate checkpoint as the
decision signal, general-capability groups as regression checks, and
base-model proxies (bits-per-byte, LoCoDiff, RepoBench) only as trend
monitors — they rose monotonically yet did not predict post-SFT results
`[paper 2610.22019]`.

Note this section reorders **domains** over the run. Reordering **individual
examples** by measured difficulty is a separate, composable lever — see §6.

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

## 6. Distillation and difficulty-curriculum for small / edge models

Below ~3B, two levers from the 2025 small-model playbook (LFM2; consistent
with Gemma 2/3, MiniCPM, SmolLM3, Llama 3.2 1B/3B) move the needle more than
any optimizer tweak. For the post-training half of the current-best ≤1B
pipeline (parallel RL teachers + OPD, MiniCPM5-1B), see post-training.md
§13; its staged-decay pretraining curriculum is in §4 above. If the user is training a sub-3B model and a strong
same-family teacher exists, **distillation is usually the single highest-
leverage change available** — recommend it before debating LR.

**Knowledge distillation — LFM2 "Decoupled Top-K" `[paper]`:**
- Student matches a larger teacher's next-token distribution, blended with
  ordinary next-token cross-entropy on hard labels (keep the CE term so the
  student isn't capped by teacher errors).
- The KL is split over the teacher's **top-32** tokens into two terms:
  (a) a *binary* KL matching the total probability **mass** the teacher puts
  on its top-32 set vs. the tail — left **untempered**; (b) a *conditional*
  KL matching the **shape within** the top-32, with temperature τ applied
  **only here**. Decoupling sidesteps the "support mismatch" that plain
  tempered full-vocab KL suffers when teacher and student disagree on which
  tokens are even plausible.
- Steal-this-even-if-you-don't-copy-the-loss: (1) truncate to top-K (K≈32–64)
  — cheaper and more stable than full-vocab KL; (2) always keep a hard-label
  CE term; (3) put temperature on the *shape*, not the *mass*.
- Teacher in LFM2 was LFM1-7B (≈3–20× the student). `[heuristic]` Pick a
  teacher 3–10× larger from the same tokenizer/family; mismatched tokenizers
  force logit re-alignment and usually aren't worth it.

**Difficulty-ordered curriculum — LFM2 `[paper]`:**
- Score each example by an *empirical* success rate p_i: run an ensemble of
  ~12 diverse LMs and set p_i = fraction that solve/produce the item. Train
  easy→hard (high p_i first), introducing items only the strongest models
  handle later.
- This is **orthogonal** to §4's data-mix annealing: that reorders *domains*
  over the run; this reorders *individual examples* by measured difficulty.
  They compose.
- `[heuristic]` Cheap proxy when a 12-model ensemble is overkill: use one
  judge model's per-example loss/perplexity (or length) as the difficulty
  surrogate, bucket into 3–4 tiers, and ramp.

**Input/embedding dropout for small models `[paper]`:** LFM2 applied ~10%
input dropout during training — a deliberate exception to the "no dropout in
pretraining" consensus baseline. Below ~2B, where capacity is small relative
to a 10T+ token budget, light input dropout is a defensible regularizer;
at ≥7B keep it off by default.

**Two-Stage Diversity-Exploring Distillation (VibeThinker SSP, the SFT
"spectrum" stage) `[paper 2511.06221]`:** builds a *diverse* SFT checkpoint
(the Pass@K-coverage goal of post-training.md §9) by checkpoint-selection +
merging, with **no external teacher** — this is *internal* self-distillation,
distinct from LFM2's logit distillation from a larger teacher above.
- *Stage 1 — Domain-Aware Diversity Probing:* partition the domain into N
  subdomains (math used **N=4**: algebra, geometry, calculus, statistics),
  build a per-subdomain probing set, periodically score intermediate SFT
  checkpoints by **Pass@K**, and keep the Pass@K-maximizing checkpoint as that
  subdomain's specialist (`M_i* = argmax_t P_i(t)`).
- *Stage 2 — Expert Model Fusion:* merge the specialists by plain unweighted
  parameter averaging (`w_i = 1/N`) — the model-soup of post-training.md §8
  applied to diversity-selected experts that share an SFT trunk.
- (1.5B SFT LR/schedule/epochs/batch/seq-len are **not stated** — don't assume.)

**Reference-model rollout difficulty filtering (VibeThinker-3B curriculum SFT)
`[paper 2606.16140]`:** a *cheaper* way to score per-example difficulty than
LFM2's 12-model success-rate ensemble above — they sit side by side as two
instances of an empirical difficulty curriculum.
- Score difficulty with a **single smaller reference model**
  (VibeThinker-1.5B): 8 rollouts/query, **keep problems with error-rate ≥0.75**
  (drop the easy ones), and discard reasoning traces **shorter than 5K tokens**.
- Two-stage schedule: **5 epochs** on the full curriculum, then **+2 epochs**
  on the hard-sample subset at the same hyperparameters.
- SFT hyperparameters (3B, verified): global **batch 128**, **LR 5e-5 →
  cosine → 8e-8**, **5% linear warmup**, sequence packing. Base =
  **Qwen2.5-Coder-3B**; the 1.5B started from **Qwen2.5-Math-1.5B** — starting
  a small reasoning model from a domain-specialized (math/code) base matters.

**Why bother with a tiny reasoning model — and where it stops `[heuristic /
hypothesis]`:** VibeThinker reports a 1.5B reaching math/code/science reasoning
"comparable to models tens to hundreds of times larger" via "meticulous
algorithmic design" `[paper-claim — single run, no scale ablation]`. The 3B
paper frames this as the **Parametric Compression-Coverage Hypothesis**:
capabilities differ not only in *how much* parameter capacity they need but in
the *structural form* of the demand — verifiable reasoning (search, constraint
satisfaction, error correction, multi-step composition) is *compressible* into
a compact reasoning core, while open-domain knowledge and long-tail facts need
*broad parameter coverage* and therefore scale. Practitioner use: it is
reasonable to pour RL/distillation effort into a small **reasoning** model, but
don't expect a small model to close **knowledge-coverage** gaps (the 3B matches
giant models on math/code yet trails on knowledge-heavy GPQA-Diamond). Honesty:
supported only observationally — no mechanism-isolating ablations in either
paper — so treat it as a hypothesis to test, not an established law.

## 7. Architecture hygiene (modern decoder defaults)

RMSNorm (no bias) + SwiGLU + RoPE + GQA; no dropout; no linear bias; untied
embeddings at ≥1B. Stability extras when justified: QK-norm, z-loss 1e-5,
post-sublayer norm reordering (OLMo 2 — norm the output of attention/FFN
before the residual add), ε=1e-8. Details and evidence: references/stability.md.

Init: OLMo 2 uses plain normal(0, 0.02) everywhere `[paper]`, dropping
scaled-init schemes — simplicity won. Tokenizer/vocab: round vocab to a
multiple of 128 for kernel efficiency `[heuristic]`.

**Hybrid / edge architectures (LFM2) `[paper]`:** when the target is
on-device (latency/memory-bound), the all-attention decoder is no longer the
only default. LFM2 interleaves **gated short convolution** blocks (kernel
size 3) with a *small* number of GQA attention layers (6 of 16 layers at
≤1.2B; 8 of 30 at 2.6B; 6 of 24 in the 8B-A1B MoE), the mix chosen by
hardware-in-the-loop search against on-device latency. Lesson for edge users:
most token-mixing can be done by cheap short convs, with attention reserved
for the few layers that need global context (≈2× faster decode at fixed
quality). For server-side inference this trade rarely pays — keep full
attention. Related families: Mamba/Jamba (SSM+attention), RWKV.

**Sparse-attention retrofit — the DSA continued-pretraining recipe (GLM-5)
`[paper 2602.15763]`:** you do not need to pretrain from scratch to get
O(L²)-breaking attention. GLM-5 adopted DSA (DeepSeek Sparse Attention: a
learned "lightning indexer" selects top-k=2048 KV entries per query,
token-level sparsity on all layers) via CPT from the dense-attention base:
1. **Indexer-only warm-up**: train *only* the indexer, base weights frozen —
   GLM-5: 1000 steps × 14 seqs × 202,752 tokens, peak LR 5e-3 (high LR is
   fine; only the tiny indexer trains). Their GLM-4.7-Flash replication:
   1000 steps, batch 16.
2. **Joint sparse adaptation**: unfreeze and co-train model + indexer on
   midtrain data/hyperparameters — GLM-5 needed only **20B tokens** (their
   small-scale run: 150B; DeepSeek-V3.2 used 943.7B) to match the dense
   model on long-context benchmarks and tie its SFT loss curve.
Their ablation hierarchy for efficient attention under continual adaptation
(190B tokens @64K, half layers efficient): naive interleaved SWA =
catastrophic on retrieval (−30 RULER@128K); search-based SWA layer selection
recovers most of it; linear attention (GDN, and their SimpleGDN which reuses
pretrained QKV weights with no extra parameters) is better still but always
loses a few points on fine-grained retrieval; **DSA was the only lossless
option** because it selects tokens rather than compressing state. Rationale
they measured: ~90% of attention entries at long context are redundant;
DSA cuts long-sequence attention compute ~1.5–2×.

## 8. Pre-launch checklist

1. Smoke run: 1–5% of budget at full distributed layout. Must see: smooth
   loss, grad-norm flat after warmup, no memory growth, expected tok/s.
2. Overfit test: 1 batch to ~0 loss (catches masking/shift bugs).
3. In-loop eval every ~1000 steps (OLMo practice): val ppl on a frozen
   slice + 2–3 cheap downstream probes; log tokens-seen, not just steps.
4. Checkpoint cadence sized so a spike costs <1% of budget to rewind.
5. Log data order/seed so any batch window can be replayed (required for
   the PaLM/OLMo skip-batches spike SOP).
6. Record the config in the run dir. Future-you is the next user.
