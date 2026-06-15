# Diagnostics: symptom → cause → fix

Use with the SKILL.md output format. Triage order when several explanations
fit: **data → batching/masking → LR/schedule → precision → optimizer →
architecture**. This ordering is empirical: FineWeb, OWSM v4, and SmolLM2
all located their wins in data; PaLM/OLMo located spikes in batch×state;
optimizer/architecture causes are real but rarer.

## Symptom table

| Symptom | Most likely causes (ranked) | First checks | Fix anchors |
| --- | --- | --- | --- |
| Spike right after warmup ends | peak LR too high; warmup too short | grad-norm at warmup end | lengthen warmup (2000→5000 steps); −30% peak LR (stability.md §3) |
| Random recoverable spikes | bad batch × stale Adam state (β₂); dirty data | replay the batch window on the same checkpoint | β₂→0.95, ε→1e-8; rewind + skip 200–500 batches (PaLM/OLMo) |
| Divergence (loss climbs, never recovers) | LR too high for current phase; logit growth | max attention/output logits | QK-norm, z-loss; lower LR; check schedule actually decays |
| NaN at step ~1 | masking/label-shift bug; fp16 overflow; div-by-zero | overfit-1-batch test; all-masked rows | fix masks; fp32 norms/softmax; bf16 |
| Loss flat from start | LR far too low (esp. LoRA with full-FT LR); frozen params; data repeated | param-update norm per layer | LoRA→1e-4–2e-4; verify requires_grad; dedup |
| Train loss falls, val flat/rising | data leakage into train metric only; overfitting (small data); val slice unrepresentative | val-slice composition; epochs | more data or fewer epochs; for SFT cap at 1–2 epochs |
| Val ppl fine, generations repetitive/broken | post-training LR too high (DPO 1e-5 syndrome); chat-template/BOS mismatch | decode train batch; token-ID diff train-vs-eval | DPO LR→5e-7 (SimPO evidence); fix template (post-training.md §6) |
| Benchmarks up, real usage worse | contamination; benchmark-overfit; length bias | length stats; fresh prompts | decontaminate; add held-out product-like eval slice |
| Math/code regress after preference tuning | preference data distribution shift | GSM8K probe before/after | mix targeted prefs (Tulu 3); SFT-loss term (SimPO tradeoff) |
| Outputs grow longer each eval | DPO/RLHF length bias | mean response length curve | length-normalized DPO (β≈5) or SimPO |
| Throughput low / step-time spiky | dataloader stalls; comm-compute no overlap; checkpoint granularity; ZeRO-3 all-gather cost | profiler: comm wait %, dataloader wait % | overlap/bucket tuning; coarser activation ckpt; ZeRO-2 if model fits |
| OOM mid-run (not at start) | activation spikes on long sequences; fragmentation | seq-length distribution; reserved-vs-allocated | length bucketing/cap; paged/expandable allocator; micro-batch −1 |
| ASR/speech diverges in warmup | warmup far too short for encoder-decoder | — | staged/longer warmup, 25K–60K steps (OWSM v3.1) |
| Speech model loops/hallucinates text | audio–text misalignment in data | CTC-confidence histogram | OWSM v4 cleaning pipeline (speech.md §3) |
| Spikes after batch/seq-len jump | effective-LR shift at ramp boundary | did spike coincide with ramp? | brief LR re-warm after jumps `[heuristic]`; ramp earlier in training |
| Grad-norm slowly trending up late in run | schedule not decaying (constant LR past intended budget); data shift | LR curve vs plan | start decay phase (WSD); inspect late data shards |

## Fast sanity tests (cheap, run before any expensive theory)

1. **Overfit one batch** to near-zero loss. Fails → wiring bug (labels,
   masks, shift), not hyperparameters.
2. **Decode a training batch** post-collation: print tokens with loss-mask
   highlighted. Catches template, packing, and masking bugs in 5 minutes.
3. **Fixed-seed replay**: same checkpoint + same data window twice → same
   loss? If not, nondeterminism is polluting your A/Bs.
4. **LR range test** (small proxy): sweep LR ×{1,2,4,8} for a few hundred
   steps; pick the largest LR with a stable grad-norm, then back off 2×
   `[heuristic]`.
5. **Token accounting**: `estimate.py batch` — confirm tokens/step matches
   the design doc. Silent grad-accum misconfigurations are endemic.

## Monitoring baseline (every run)

Per ≤50 steps: loss, grad-norm, LR, tokens-seen, tok/s. Per ~1000 steps:
frozen-slice val loss + 1–3 micro-evals (OLMo cadence). For post-training
add: response length, KL-to-reference (DPO/RL), reward mean (RL),
DPO pair accuracy. Keep a frozen eval set for the entire project;
never retune it mid-flight.

This is the health-monitoring floor. For the full per-stage metric catalog
(numerical-health, data-contamination, tokenizer, MoE-specific signals), the
benchmark-by-capability tables, and ready-to-use minimal eval suites, see
`references/evaluation.md`.

## When the user's report is incomplete

Ask for exactly these four artifacts before diagnosing spikes:
(1) loss curve, (2) grad-norm curve, (3) the config (optimizer block +
batch math), (4) what changed relative to the last healthy run. Refuse to
hypothesize from loss alone when grad-norm is one click away — wrong-layer
diagnoses (blaming LR for a data bug) are the main failure mode of
human and AI tuners alike.
