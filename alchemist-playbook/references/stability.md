# Stability: loss spikes, divergence, precision

Verification tags: `[config]`, `[paper]`, `[reported]`.

## Contents
1. Spike triage SOP (do this first)
2. Spike taxonomy and causes
3. The optimizer stability stack
4. The architecture stability stack
5. MuonClip and modern optimizers
6. Precision rules (bf16 / fp16 / FP8)
7. Monitoring: the minimal dashboard
8. Predicting instability before the big run

## 1. Spike triage SOP

When a user reports a spike or NaN, walk this order:

1. **Characterize**: single spike that recovers, repeated spikes, or
   divergence? Pull loss AND grad-norm curves. Grad-norm precedes loss.
2. **Rewind + skip** (the proven emergency fix): restart from the last good
   checkpoint and skip the offending data window. PaLM: rewind ~100 steps
   before the spike, skip 200–500 batches — spikes did not recur `[paper]`.
   OLMo applied the same data-skip practice `[paper]`. Note PaLM's finding:
   the same batches on a *different* checkpoint did not spike — it is a
   batch×state interaction, so skipping is legitimate, not data denial.
3. **If spikes repeat**, stop firefighting and fix the stack: §3 then §4.
4. **If NaN (not spike)**: almost always precision or a div-by-zero
   (attention mask rows fully masked, log(0) in a custom loss, fp16
   overflow). Check §6 before touching LR.

## 2. Spike taxonomy

| Pattern | Mechanism | Primary fix |
| --- | --- | --- |
| Spike at warmup end | LR peaks before optimizer state is calibrated | longer warmup; lower peak LR |
| Random spikes mid-run, recover | bad batch × stale Adam second moment | β₂ 0.95; rewind+skip; data cleaning |
| Spikes correlated with batch-size/seq change | effective-LR shift | re-warm LR briefly after batch jumps `[heuristic]` |
| Slow logit growth → blowup | attention-logit or output-logit divergence | QK-norm; z-loss; see §4 |
| Late-run spikes at large scale | LR too high for late loss landscape | check schedule actually decays; Wortsman: instability grows with scale at fixed LR `[paper]` |
| Immediate NaN step 1 | init/precision/mask bug | overfit-1-batch test; fp32 norms |

## 3. Optimizer stability stack (cheap, do first)

- **β₂ = 0.95** (from 0.999). Long second-moment memory is the spike
  amplifier at scale. LLaMA, OLMo, DeepSeek all ship 0.95 `[paper/config]`.
- **ε = 1e-8** (from 1e-5). OLMo 2 found large ε distorts updates;
  config-verified `[config]`.
- **Global grad clip 1.0** — universal. If clip events are frequent
  (>~1% of steps) the LR is too high; clipping is a seatbelt, not a brake
  `[heuristic]`.
- **Warmup**: 2000 steps standard; 8000 at 405B `[paper]`; OLMo 2 counts
  8.4B tokens `[config]`. Under-warmup shows up as warmup-end spikes.
- **Weight decay 0.1**, applied decoupled (AdamW); commonly skipped for
  norms/embeddings. PaLM instead used dynamic wd ∝ lr² `[paper]` — exotic,
  do not recommend by default.

## 4. Architecture stability stack (ranked by evidence)

1. **QK-norm** — RMSNorm on queries and keys before attention. Direct
   anti-dote to attention-logit growth. Adopted by OLMo 2 `[paper]`;
   validated at small scale by Wortsman et al. 2309.14322 `[paper]`.
   Caveat: changes attention dynamics slightly; adopt at run start, not
   mid-run.
2. **z-loss** — auxiliary `c·log²Z` on the softmax normalizer keeps output
   logits bounded. PaLM c=1e-4 `[paper]`; OLMo 2 c=1e-5 `[paper]`,
   OLMo-core 32B script ships 1e-5 `[config]`.
3. **Norm placement** — OLMo 2 reordered to normalize each sublayer's
   *output* before the residual add (keeps the residual stream clean while
   bounding sublayer output scale) `[paper]`.
4. **Embedding protection** — GLM-130B shrank embedding-layer gradients
   (α=0.1) to stop early spikes `[paper]`; "Spike No More" (2312.16903)
   argues for scaled embeddings (×√d) + small init `[paper]`.
5. **Init** — OLMo 2: plain normal(0, 0.02) everywhere beat clever scaled
   inits for stability `[paper]`.
6. **fp32 in norms and softmax** — RMSNorm/LayerNorm statistics and
   attention softmax in fp32 even under bf16 autocast (standard in
   Megatron/OLMo stacks) `[reported]`.

## 5. MuonClip and modern optimizers

- **Muon** (orthogonalized momentum via Newton–Schulz, applied to 2D weight
  matrices; AdamW retained for embeddings/norms/head). Moonlight
  `[paper 2502.16982]`: Muon reached AdamW-comparable loss with ≈52% of the
  FLOPs (~2× compute efficiency) at 16B-MoE/5.7T-token scale, but required
  (a) weight decay (0.1, including on RMSNorm γ) and (b) rescaling updates
  to match AdamW's typical update RMS (~0.2) so one global LR works across
  parameter shapes.
- **MuonClip / QK-Clip** (Kimi K2 `[paper 2507.20534]`): Muon plus a
  per-head clip on query/key projection weights whenever max attention
  logit exceeds τ=100. K2 (1T-param MoE, 32B active) pretrained on 15.5T
  tokens with **zero loss spikes**; max logits hit the τ=100 cap early,
  then naturally decayed below it after ~30% of training.
- Guidance: for AdamW users with logit-growth symptoms, QK-norm (§4) is the
  established fix; QK-Clip is the Muon-ecosystem equivalent. Do not switch
  optimizer families mid-run.

## 6. Precision rules

- **bf16 everywhere, fp32 where it counts**: master weights and gradient
  all-reduce in fp32 (OLMo-core ships `reduce_dtype: fp32` `[config]`),
  norms/softmax fp32.
- **fp16 is legacy-hazard**: needs dynamic loss scaling; silent overflow
  caused BLOOM-era instabilities — BLOOM-176B chose bf16 for this reason
  `[paper]`. If a user is on V100s (fp16-only), keep loss-scale logs on the
  dashboard.
- **FP8 is real but not free**: DeepSeek-V3 trained with FP8 GEMMs using
  fine-grained (tile/block-wise) scaling and higher-precision accumulation
  `[paper]`. Without those two ingredients, expect divergence. The HF
  Ultra-Scale Playbook's systematic runs also found FP8 less stable than
  bf16 `[reported]`. Recommend FP8 only with a framework that implements
  per-tile scaling (e.g., DeepSeek-style or Transformer Engine), never as
  a flag flip.

## 7. Monitoring: the minimal dashboard

Log every N≤50 steps: loss, global grad-norm, LR, tokens-seen, throughput;
every ~1000 steps: val loss on a frozen slice (OLMo cadence `[paper]`).
If instability is suspected, add: max attention logit per layer, output
logit max/mean, loss-scale (fp16), and clip-event rate. Alert thresholds
`[heuristic]`: grad-norm > 3× running median; max attention logit > 50 and
rising (K2 clipped at 100); clip rate > 2% of steps.

## 8. Predicting instability cheaply

Wortsman et al. 2309.14322 `[paper]`: small models at high LR reproduce
large-model instabilities (attention-logit growth, output-logit
divergence). Practical transfer: before a big run, sweep LR ×{1, 2, 4, 8}
on a proxy (~100M–1B); if the LR-vs-loss bowl is narrow or QK-norm/z-loss
visibly widen it, ship the stability stack in the big run. The same paper
validated that QK-norm and z-loss extend the trainable LR range — the
basis for treating them as default-on at ≥30B scale `[heuristic]`.
