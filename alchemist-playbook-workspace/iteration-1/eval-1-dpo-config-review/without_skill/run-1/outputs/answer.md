# DPO Config Review: Llama-3.1-8B (post-SFT)

## TL;DR

Three hyperparameters are far outside the sane DPO regime, and they compound:

| Setting | Yours | Typical | Off by |
|---|---|---|---|
| `beta` | 5 | 0.05–0.1 (TRL default 0.1) | ~50x |
| `learning_rate` | 2e-5 | 5e-7–1e-6 (full FT, 8B) | ~20–40x |
| `num_train_epochs` | 3 | 1 | 3x |

The result is exactly what you observe: the policy drifts catastrophically from your SFT checkpoint, producing degenerate repetition and forgetting reasoning ability (GSM8K −15).

## Why each one hurts

### 1. beta = 5
Theoretically, higher beta means a tighter KL constraint to the reference — but that only holds at the exact optimum, not under SGD dynamics. In the actual loss, beta multiplies the log-ratio margin inside the sigmoid, and the gradient is scaled by `beta * sigmoid(-beta * margin)`. At the start of training (margin ≈ 0) that's `beta/2 = 2.5` versus `0.05` for beta=0.1 — your gradients start ~50x larger. Later, the sigmoid saturates as soon as a tiny margin opens, so the signal becomes spiky and uninformative.

### 2. learning_rate = 2e-5
That's an SFT-scale LR. DPO recipes for 7–8B models (Zephyr, Tulu) use 5e-7, at most ~1e-6 for full fine-tuning (~5e-6 only with LoRA). Combined with the beta-inflated gradients, your effective update size is roughly three orders of magnitude too large. This alone explains the capability collapse.

### 3. 3 epochs
DPO overfits preference data quickly. A well-known sigmoid-DPO failure mode: the loss only optimizes the chosen-vs-rejected *margin*, so the absolute likelihood of **both** chosen and rejected responses can fall, with probability mass leaking onto degenerate high-frequency continuations — i.e., repetition loops. Multiple epochs at a huge LR maximize this effect.

### Why GSM8K specifically dropped
Classic alignment tax amplified by over-optimization: the policy moved far from the SFT model that knew math, and (if your preference data contains little math/reasoning) DPO actively reshaped the output distribution away from CoT-style answers.

### What's fine
- Effective batch 32: fine (32–128 is typical).
- `max_length=4096`: fine, but also set `max_prompt_length` (e.g., 2048) so chosen/rejected completions aren't truncated — truncated pairs give garbage signal.
- Sigmoid loss: a reasonable default; it's not the culprit.

## Fixed config

```python
DPOConfig(
    beta=0.1,                # was 5
    learning_rate=5e-7,      # was 2e-5; ~5e-6 if LoRA
    num_train_epochs=1,      # was 3
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_length=4096,
    max_prompt_length=2048,
    # keep effective batch 32
    rpo_alpha=1.0,           # optional: adds NLL on chosen; protects capabilities
)
```

`rpo_alpha` (DPO + SFT NLL on the chosen response) directly counteracts the chosen-likelihood collapse and is cheap insurance for benchmark regression.

## What to monitor on the rerun

- `rewards/accuracies`: healthy is ~0.6–0.8; pinned at ~1.0 means overfitting.
- `logps/chosen`: should not crater — if it falls steeply, you're in the degenerate regime again.
- `rewards/margins`: should grow slowly and smoothly.
- Run GSM8K (and a sample-generation eye-check for repetition) every ~100–200 steps; early-stop on a >2–3 point drop.

## If repetition persists at sane hyperparameters

- Audit the preference data for length/repetition bias in chosen responses (DPO amplifies dataset artifacts).
- Add math/reasoning preference pairs, or rely on `rpo_alpha`.
- Try `label_smoothing=0.1` (cDPO) or IPO loss if margins overfit.

Bottom line: it's not one bug — beta, LR, and epochs are each individually too aggressive, and multiplied together they nuked your SFT policy. Fix all three.
