# Speech model recipes (ASR / ST / speech SSL)

Verification tags: `[config]`, `[paper]`, `[reported]`.

## Contents
1. How speech differs from text
2. Whisper (supervised multitask)
3. OWSM v3.1 / v4 (open Whisper-style)
4. Speech SSL: wav2vec 2.0, HuBERT
5. ESPnet working defaults
6. Batching for variable-length audio
7. Multitask token formats

## 1. How speech differs from text

At equal parameter count, published speech recipes use **higher peak LRs**,
**different Adam β₂/ε**, and sometimes **much longer warmup**:
Whisper-large (1.5B) peak LR 1.75e-4 with β₂=0.98, ε=1e-6 `[paper]`;
OWSM v3.1 (1.02B) peak 2e-4 with 60K-step warmup `[config-name/paper]`.
Continuous, noisy acoustic features change the loss landscape; do not
transplant text-LLM LRs into encoder–decoder ASR. Length variance — not
the optimizer — is usually the throughput and stability story (§6).

## 2. Whisper `[paper 2212.04356, Appendix F]`

AdamW β=(0.9, **0.98**), ε=1e-6, wd 0.1, clip 1.0; warmup 2048 updates,
**linear decay to zero**; batch 256 segments (30s each); 2^20 ≈ 1.05M
updates (~2–3 epochs over 680K hours); no SpecAugment for the main models
(data scale substituted for augmentation); large-v2 added augmentation +
2.5× more epochs.

Peak LR by size:

| Model | Params | Peak LR |
| --- | --- | --- |
| tiny | 39M | 1.5e-3 |
| base | 74M | 1.0e-3 |
| small | 244M | 5e-4 |
| medium | 769M | 2.5e-4 |
| large | 1.55B | 1.75e-4 |
| large-v2 | 1.55B | 2.0e-4 |

Same inverse size↔LR law as text, shifted up ~3–5×.

## 3. OWSM (Open Whisper-style Speech Models)

**v3.1** `[paper 2401.16658 + released config]`: E-Branchformer encoder,
101M (base) and 1.02B (medium); 180K hours, 151 languages; joint
attention+CTC with CTC weight 0.3; FlashAttention; batch 256; total
~675K updates `[reported]`. Scheduler: **piecewise-linear warmup over 60K
steps** to peak 2e-4 (medium) — released config name encodes
`piecewise_lr2e-4_warmup60k` `[config]`; base used 1e-3 `[reported]`.
The two-segment warmup (slow climb to a small LR, then faster climb to
peak) existed because plain short warmup diverged on the large
E-Branchformer — when an encoder–decoder ASR run diverges in warmup,
lengthen and stage the warmup before touching peak LR.

**v4** `[paper 2506.00338, Interspeech 2025 best student paper]`: the gains
came from **data cleaning, not optimizer changes**. Pipeline over YODAS
(wild web audio): (1) re-segment/re-align audio↔text with CTC segmentation;
(2) language-ID filtering (audio LID vs label); (3) CTC-confidence
filtering to drop misaligned pairs. 370K raw hours → 166K clean hours
across 75 languages; combined with existing OWSM data (~320K hours total).
Result: outperforms or matches Whisper and MMS on multilingual benchmarks.
Uncleaned YODAS produced repetition/alignment pathologies (WER can exceed
100 on bad subsets) `[paper]`. Quote this whenever a user wants to "just
add more web audio".

## 4. Speech SSL

**wav2vec 2.0** `[paper 2006.11477]`: contrastive + codebook-diversity loss
(diversity weight 0.1); mask p≈0.065 span starts, span M=10 (~49% of
frames masked); Adam, warmup ~32K steps then polynomial decay; peak LR
5e-4 (Base) / 3e-4 (Large) `[paper]`. Fine-tuning uses **tri-stage** LR
(10% warmup / 40% hold / 50% exponential decay) with peak ~3e-5 and an
initial frozen-encoder phase `[paper]`.

**HuBERT** `[paper 2106.07447]`: masked prediction of clustered targets;
iteration 1 targets = MFCC k-means (100 clusters); iteration 2 = k-means
(500) on layer-6 features of the it-1 model; Large/X-Large trained on
layer-9 features of it-2 Base. Masking and optimization mirror wav2vec 2.0
(p=0.08, span 10) `[paper]`. The recipe lesson: target quality (cluster
source layer) matters more than optimizer detail — same data-first law.

## 5. ESPnet working defaults `[reported — standard recipe values]`

Warmup 25K–40K steps (Noam/warmuplr); joint CTC weight 0.3 (train and
decode); label smoothing 0.1; SpecAugment on for supervised ASR fine-tunes;
attention dropout ~0.1 for small/medium supervised models (unlike text-LLM
pretraining, dropout earns its keep at speech data scales); average the
last ~10 checkpoints for the final model.

## 6. Batching for variable-length audio

Sort or bucket utterances by length so each batch holds similar durations;
shuffle at the bucket level (keeps gradient diversity); cap batches by
**total frames/duration (batch_bins)**, not utterance count — a
256-utterance batch of 30s audio and one of 3s audio differ 10× in compute.
Padding waste is the silent killer: 30–50% throughput losses from naive
batching are common `[reported]`. OWSM-style training standardizes 30s
windows (Whisper convention) which trades padding for simplicity; for
in-house data, dynamic batch_bins is strictly better. After any batching
change, re-check effective tokens/frames per step — it is an LR-relevant
quantity (same logic as text §batch).

## 7. Multitask token formats

Whisper/OWSM serialize task routing into the decoder prefix:
`<|startoftranscript|><|lang|><|task|><|notimestamps|>...`. One model,
many tasks (transcribe, translate, LID, timestamps) — and zero-shot
combinations emerge late in training `[paper]`. When a user builds any
multi-task speech-text model, push them toward a single serialized prompt
format rather than task-specific heads: it is the published-validated
path (Whisper, OWSM) and keeps data pipelines unified.
