# DPO Config Review — Llama-3.1-8B (on SFT checkpoint)

## Diagnosis

1. **learning_rate=2e-5 is ~40× too high for DPO — this is the repetition.** Published 8B DPO runs use 5e-7 (Zephyr, Tulu 3 8B). The SimPO authors documented that 1e-5 reliably lobotomizes a 7–8B model into repetition; you are at 2× that, for 3 epochs. 2e-5 is an *SFT*-scale LR — it violates the stage ladder (DPO ≈ SFT LR ÷ 10–40). This one knob explains both symptoms: degenerate repetitive outputs and most of the 15-point GSM8K capability tax.
2. **β=5 is paired with the wrong loss type.** β=5 is Tulu 3's value for **length-normalized** DPO, where dividing log-probs by length shrinks the logit scale ~100×, so β must grow to compensate. With TRL's standard sigmoid loss, published β is 0.01–0.1 (Zephyr 0.01, DPO paper 0.1). β=5 on un-normalized log-ratio sums saturates the sigmoid almost immediately — gradients vanish on separated pairs and the few remaining active pairs get hammered at LR 2e-5. Never transplant β across loss variants.
3. **3 epochs of preference data compounds the damage.** Every documented DPO run trains 1 epoch (Zephyr, Tulu 3, SimPO baselines). Epochs 2–3 re-optimize already-fit off-policy pairs, amplifying the drift from your SFT reference. Secondary suspect for GSM8K only *if* symptoms persist after the fixes: chat-heavy preference data shifting distribution away from math.

## Recipe

Restart from the **SFT checkpoint** — the current DPO'd weights are not worth salvaging.

```
learning_rate:      2e-5 → 5e-7        # grid 3e-7 / 5e-7 / 8e-7 / 1e-6 if budget allows; lean low (5e-7) for math-heavy data
beta:               5 → 0.1            # keeping loss_type="sigmoid"; grid 0.01–0.1
                                       # (alternative: keep β=5 but switch to length-normalized DPO à la Tulu 3 — pick one pairing, not a mix)
num_train_epochs:   3 → 1
effective batch:    32 → 128           # e.g., per_device 4 × 8 GPUs × grad_accum 4
max_length:         4096 (keep) + set max_prompt_length (Tulu 3 ran 2048 total; 4096 is fine if your pairs are long)
schedule/warmup:    cosine or linear, warmup_ratio 0.1
```

If GSM8K still regresses with the corrected config: mix targeted math/reasoning preference pairs into the data (Tulu 3's approach), or add an SFT/NLL loss term — knowing it costs some chat win-rate (SimPO tradeoff).

## Evidence

- LR 5e-7 @ 8B: Zephyr-7B `[config]`, Tulu 3 8B `[config]`; SimPO baselines grid 3e-7–1e-6 `[config]`. "1e-5 → repetition" : SimPO authors `[config]`.
- β–loss coupling: Zephyr β=0.01 standard sigmoid `[config]`; Tulu 3 β=5 *length-normalized* `[config]`; DPO paper β=0.1 `[paper]`.
- 1 epoch, effective batch 128: Zephyr, Tulu 3, SimPO `[config]`.
- GSM8K-after-DPO mitigations: Tulu 3 targeted preference mixing `[config]`; SFT-loss term tradeoff per SimPO authors `[config]`.

## Verify

In-loop (every ~50 steps): DPO pair accuracy should climb toward ~0.6–0.8 without the implicit-reward margin exploding; mean response length flat (not growing); grad-norm stable. Stage exit: rerun the GSM8K probe and a repetition check (decode ~50 fixed prompts) against the **SFT stage-input model** — GSM8K should land within ~1–3 points of the SFT checkpoint, with chat win-rate improved. Rollback plan: if pair accuracy stalls near 0.5 after the LR fix, raise LR one grid step (8e-7) — one change per rerun. If GSM8K still drops >5 points at LR 5e-7 / β 0.1 / 1 epoch, the cause is preference-data distribution shift, not the optimizer: fix the data mix, not the knobs.
