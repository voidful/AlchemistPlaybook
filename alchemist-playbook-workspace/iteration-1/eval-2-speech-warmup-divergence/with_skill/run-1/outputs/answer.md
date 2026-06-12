# NaN during warmup: ~1B E-Branchformer encoder–decoder ASR (ESPnet, 40k h)

Your setup is almost exactly **OWSM v3.1 medium**: 1.02B E-Branchformer
encoder–decoder, batch 256, ~180K h, peak 2e-4. So we have a directly
comparable published run — and it tells us precisely what goes wrong here.

## Diagnosis

1. **Warmup is far too short for a large E-Branchformer.** 5000 steps is a
   text-LLM number. Continuous, noisy acoustic features reshape the loss
   landscape, and big E-Branchformer ASR encoders are warmup-fragile: OWSM
   v3.1 medium *diverged with plain short warmup* and needed a staged
   (piecewise-linear) warmup over **60K steps** to reach 2e-4. This is the
   single most likely cause, and it matches "NaN during warmup almost every
   time."
2. **You transplanted text Adam β₂/ε.** Published speech recipes use a
   *shorter* second-moment memory and larger ε than text: Whisper β₂=0.98,
   ε=1e-6. Text defaults (β₂=0.95–0.999, ε=1e-8) sit outside the validated
   speech regime and make early updates twitchier.
3. **True NaN (not a recoverable spike) often = precision / div-by-zero**,
   not LR alone. With variable-length audio the usual culprits are
   fully-masked attention rows (a padded/empty utterance), log(0) in a
   custom CTC/joint loss term, or fp16 overflow. Worth ruling out in
   parallel — it is cheap.

## Recipe (config diff)

```
scheduler:        warmuplr → keep, but stage it
warmup_steps:     5000 → 25000–40000 (ESPnet default band)
                  if it still diverges → OWSM-style two-segment warmup to 60000
optimizer.beta2:  (text value) → 0.98     # Whisper
optimizer.eps:    1e-8 → 1e-6             # Whisper
peak lr:          2e-4 → keep             # correct for ~1B speech; do NOT cut first
grad_clip:        1.0                     # confirm it is on
precision:        bf16 compute; fp32 norms + attention softmax
batching:         batch_bins (cap by frames/duration), length-bucketed
```

Order of operations: **lengthen/stage warmup first, fix β₂/ε, leave peak LR
at 2e-4.** Only drop peak LR if divergence survives a 60K staged warmup.
Run the two cheap NaN checks below before any of this.

## Evidence

- Staged/long warmup for big E-Branchformer ASR: OWSM v3.1 medium config
  `piecewise_lr2e-4_warmup60k`; plain short warmup diverged `[config/paper]`.
  ESPnet standard warmup band 25K–40K `[reported]`.
- Peak 2e-4 at ~1B speech is *in-regime*: OWSM v3.1 medium 2e-4 `[config]`;
  Whisper-large 1.75e-4 / large-v2 2.0e-4 `[paper]`. Speech LR sits ~3–5×
  above text at equal size — your LR is not the bug.
- β₂=0.98, ε=1e-6: Whisper Appendix F `[paper]`.
- NaN → precision / all-masked rows / log(0): stability SOP §1.4; fp32
  norms+softmax standard in OLMo/Megatron stacks `[paper/reported]`.
- batch_bins / length bucketing: padding waste drives 30–50% throughput
  loss and changes effective frames/step (LR-relevant) `[reported]`.

## Verify

- **Cheap NaN tests first (5 min each):** overfit one batch to near-zero
  loss (fails → masking/label-shift wiring bug, not LR); decode a collated
  training batch and scan for empty/fully-padded utterances and any log(0)
  path in the CTC/joint loss.
- After the warmup change: log loss **and grad-norm** every ≤50 steps.
  Grad-norm precedes loss — a clean run keeps grad-norm under ~3× its
  running median through the entire warmup ramp, with no NaN.
- Confirm the run survives past the old failure point (the warmup-end
  region) and joins a smooth descent.
- **Rollback:** if a 60K staged warmup still NaNs, it is not the schedule —
  go back to the precision/masking checks and inspect data alignment
  (CTC-confidence histogram) before touching peak LR.
