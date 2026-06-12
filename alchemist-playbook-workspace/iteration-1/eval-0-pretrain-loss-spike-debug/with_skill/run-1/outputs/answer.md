# 7B pretrain loss spikes at ~80B tokens — diagnosis and fix

Your recipe is one knob away from Llama 2 7B (2T tokens, 3e-4, 4M batch, cosine) — the deviations are exactly where the spikes come from.

## Diagnosis

1. **β₂ = 0.999 — the spike amplifier.** Your pattern (repeated recoverable spikes, grad-norm jumping to ~40 ≈ 67× its ~0.6 median *before* each loss spike) is the textbook "bad batch × stale Adam second moment" signature. With β₂=0.999 one outlier batch poisons the second-moment denominator for thousands of steps, so the run keeps re-spiking. Every major open recipe (LLaMA 1/2/3, OLMo 1/2, DeepSeek-V3) ships 0.95 at this scale.
2. **ε = 1e-5 — distorts small-gradient updates.** OLMo-1 trained 7B with ε=1e-5 and moved to 1e-8 in OLMo 2 explicitly for stability (config-verified).
3. **Not the cause of these spikes, but fix anyway:** spikes start at step ~20,000 — far past warmup end — so warmup isn't the trigger; still, 1000 steps (4B tokens) is half the standard for this exact recipe (LLaMA 2: 2000 steps; OLMo 2: 8.4B tokens). And confirm global grad clipping is actually on: a measured norm of 40 means either no clip or a seatbelt that can't protect Adam's state from corruption.

## Recipe

```
optimizer.betas:  (0.9, 0.999) → (0.9, 0.95)
optimizer.eps:    1e-5         → 1e-8
grad_clip:        ensure global-norm 1.0 (add if missing)
```

Apply via **rewind + skip**: restart from the last checkpoint *before* the most recent spike, skip the ~200–500 batches around it (requires logged data order/seed), resume with the new optimizer settings. Also verify your bf16 setup does gradient all-reduce and norm/softmax statistics in fp32.

**For the next run (not mid-run):** QK-norm + z-loss 1e-5, warmup 2000 steps. QK-norm changes attention dynamics — adopt at run start only.

**Monitoring:** add max attention logit per layer and clip-event rate. Alert at grad-norm > 3× running median, max attention logit > 50 and rising, clip rate > 2% of steps.

## Evidence

- β₂ = 0.95: LLaMA 1/2/3 `[paper]`, OLMo 1/2 `[config]`, DeepSeek-V3 `[paper]`.
- ε = 1e-8: OLMo 2 official configs `[config]` (changed from OLMo-1's 1e-5).
- Rewind + skip 200–500 batches: PaLM §5.1 `[paper]` — same batches replayed on a *different* checkpoint did not spike, proving batch×state interaction; OLMo applied the same practice `[paper]`.
- Clip 1.0: universal across LLaMA/OLMo/Whisper recipes.
- Warmup 2000 @ 4M batch / 3e-4 / 7B: LLaMA 2 `[paper]`; OLMo 2 8.4B tokens `[config]`.
- fp32 grad-reduce under bf16: OLMo-core `reduce_dtype: fp32` `[config]`.
- QK-norm: OLMo 2 `[paper]`, Wortsman et al. 2309.14322 `[paper]`. z-loss 1e-5: OLMo 2 `[config]`; 1e-4: PaLM `[paper]`.

## Verify

Over the next ~5k steps after resuming: grad-norm stays < 2× its running median (~0.6) with no excursions toward 40; clip events < ~1% of steps; loss rejoins the pre-spike trend within ~1B tokens (~250 steps at 4M). You have ~480k steps left — plenty of signal.

- **If spikes still recur** with the new optimizer block: suspicion routes to data (triage order: data before architecture). Replay the offending windows, inspect/clean those shards, and ship QK-norm + z-loss in the relaunch.
- **Rollback:** restore β₂=0.999 only if convergence measurably slows (rare); keep ε=1e-8 regardless. This keeps the change attributable — one stack change per rerun, with QK-norm deliberately deferred.
