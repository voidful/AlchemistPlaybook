# Debugging recurring loss spikes in a 7B pretrain

## Diagnosis

Recurring spikes with grad norm jumping ~60x (0.6 → 40) just before each one, partial recovery, starting ~20k steps in — this is the classic large-batch Adam instability pattern, not a fundamental architecture or data-scale problem. The biggest red flag in your config is **beta2 = 0.999**.

With a 4M-token batch, gradients are very low-noise, so a rare fat-tailed gradient (bad batch, attention-logit blow-up) dominates. With beta2 = 0.999, Adam's second moment `v` adapts over thousands of steps, so it badly underestimates the new gradient scale → the effective step explodes → loss spikes → `v` slowly catches up → partial recovery → repeat. This is exactly why essentially every modern LLM run (GPT-3, PaLM, LLaMA, OLMo, Falcon) uses **beta2 = 0.95**.

## What to change, in priority order

1. **Roll back; don't ride it out.** Restore a checkpoint from before the first spike (~75–80B tokens). "Mostly recovered" loss usually hides lasting damage, and the Adam state is contaminated. Resume with the fixes below (loading old optimizer state with new betas is fine).

2. **Set beta2 = 0.95** (keep beta1 = 0.9). This is the single highest-impact fix. Keep eps = 1e-5 — that's actually fine and mildly stabilizing (it caps the per-parameter step when `v` is tiny); don't "fix" it to 1e-8.

3. **Gradient clipping at global norm 1.0.** If you're already clipping and still seeing this, note that clipping bounds magnitude but not a garbage direction — you still need (4) and (5).

4. **Automatic skip logic.** Skip the optimizer step (don't update Adam moments either) when the pre-clip grad norm exceeds ~2–3x its running EMA, or loss exceeds a threshold. On a full spike, use the PaLM recipe: rewind to the last good checkpoint and skip the ~100–500 data batches around the spike — spikes are typically reproducible from the same data + state, and skipping that window avoids them.

5. **Add z-loss** (`1e-4 · log²Z` on the output softmax, PaLM-style). Safe to enable mid-run; prevents output-logit divergence, a common bf16 spike mechanism. If you ever restart from scratch, also add **QK-layernorm** (don't bolt it onto an existing checkpoint mid-run).

6. **Audit precision.** bf16 compute is fine, but verify: fp32 master weights, fp32 Adam states, fp32 gradient accumulation and all-reduce, and softmax/LayerNorm/RoPE math in fp32 (FlashAttention already does softmax in fp32). A bf16 optimizer state or bf16 grad reduction is a known spike source.

7. **Inspect the data at spike steps.** Dump the exact batches preceding each spike; look for massively repeated text, garbage encodings, or long near-uniform sequences. If a shard is pathological, filter/dedup it.

8. **Secondary knobs only if spikes persist after the above:** lower peak LR to ~2.5e-4, and extend warmup to 2–5k steps on the restart. Your LR itself isn't wrong — 3e-4 at 4M batch is the LLaMA-7B recipe — and warmup is not the proximate cause (spikes at ~20k steps, warmup ended at 1k), though 1000 steps (~4B tokens) is on the short side by modern standards.

## Monitoring to add

Log per-step: pre-clip global grad norm (and per-layer norms), max attention logit, and output-logit max/z. These rise before the loss does and give you an early-warning trigger for the skip logic.

## Expected outcome

beta2 = 0.95 + clipping + skip-and-rewind eliminates the large majority of these spikes; z-loss and precision fixes cover the logit-growth cases. You should not need to lower the LR.
