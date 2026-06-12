# Speech ASR NaN during warmup — what's different and what to set

## Why speech diverges when text doesn't

Your text recipe is well-conditioned by construction: inputs are learned embeddings (bounded scale), the loss is a single cross-entropy, and there's no alignment search. A from-scratch 1B ASR model breaks all three assumptions early, and warmup is exactly when the model is random and gradients are largest, so the effects compound:

1. **CTC loss can be +inf / NaN.** ESPnet's hybrid CTC/attention objective (typically `mtlalpha` ~0.3) has a CTC branch. When the subsampled input length `T` is shorter than the label length `L`, no valid alignment exists and CTC returns inf. With 40k hours of real data you *will* have such utterances (short audio with long/garbage transcripts, empty text). A single inf in a batch produces a NaN gradient. This failure mode does not exist in LM training.

2. **Raw acoustic features are unnormalized with large dynamic range.** Log-mel fbanks aren't bounded like embeddings. Without global CMVN the conv2d front-end and macaron FFNs see huge activations and blow up early.

3. **Extreme, variable sequence lengths.** Utterances range from <1s to tens of seconds. Long ones inflate attention softmax and conv activations; mixed batches make the gradient norm spiky precisely when the model is random.

4. **fp16 overflow.** The CTC log-softmax and long-sequence attention softmax overflow fp16's range far more often in speech than in text.

So "NaN almost every time during warmup" is the expected symptom of the above, not primarily a bad peak LR.

## What to set (rough priority order)

- **Precision: use bf16, not fp16.** This alone fixes a large fraction of ASR NaNs because CTC and softmax keep their dynamic range. If you must use fp16, you need dynamic loss scaling and it's still fragile — consider keeping the front-end/CTC in fp32.
- **Gradient clipping.** `grad_clip: 5.0` (try `1.0` if still spiking). Essential for speech; otherwise one bad batch destroys the model. Text tolerates much more.
- **Longer warmup.** 5000 is a text number. ESPnet Conformer / E-Branchformer recipes use ~25k–40k warmup steps; for a 1B from-scratch model set `warmup_steps: 25000–40000`. This is the most direct fix for "same warmup as text."
- **Global feature normalization.** `normalize: global_mvn` (compute stats first) or `utterance_mvn`.
- **Filter the data / guard CTC.** Run `remove_longshortdata.sh` (or equivalent): drop empty transcripts, cap max audio length (~30s), and remove utterances where subsampled `T < L`. This eliminates the inf-CTC source.
- **De-risk the LR only if needed.** Peak 2e-4 is fine *with* long warmup; if it still diverges, drop to 1e-4 or extend warmup further rather than chasing it elsewhere.
- **Optional:** delay or soften SpecAugment for the first few thousand steps (aggressive masking on short utterances early can create degenerate inputs), and confirm gradient accumulation isn't letting one NaN micro-batch poison the accumulated step.

**Start here:** bf16 + `grad_clip: 5.0` + `warmup_steps: 25000` + `global_mvn` + data length/text filtering. That combination resolves the overwhelming majority of warmup NaNs in ESPnet ASR.
