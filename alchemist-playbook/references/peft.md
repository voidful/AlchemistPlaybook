# PEFT: LoRA and QLoRA

Verification tags: `[config]`, `[paper]`, `[reported]`.

## Contents
1. LoRA vs full fine-tuning: when each wins
2. LoRA hyperparameters
3. QLoRA stack
4. Memory math
5. LoRA + DPO
6. Pitfalls checklist

## 1. LoRA vs full FT

"LoRA Learns Less and Forgets Less" (2405.09673) `[paper]`: on hard target
domains (code/math continued-pretraining), LoRA underperforms full FT at
equal data; but it forgets the source domain less — it acts as a
regularizer toward the base model. Decision rule:

- Style/format/instruction adaptation, small-to-mid data (≤~1M examples),
  limited VRAM → **LoRA/QLoRA**. Quality gap vs full FT is small here
  (QLoRA's Guanaco matched strong baselines `[paper]`).
- Injecting substantial new capability/knowledge (new language, heavy code
  or math CPT) → **full FT** (or much higher LoRA rank + all modules, and
  accept some gap).
- Preserving base capabilities is itself a goal → LoRA's forgetting
  resistance is a feature.

## 2. LoRA hyperparameters

| Knob | Recommendation | Evidence |
| --- | --- | --- |
| Target modules | **All linear layers**: q,k,v,o + gate,up,down. This matters more than rank. | QLoRA ablation: all-linear needed to match full FT `[paper]`; 2405.09673 concurs |
| Rank r | 16–64 for SFT-style adaptation; 64+ (even 256) for CPT-like jobs | QLoRA used r=64 `[paper]`; 2405.09673 swept 16–256 |
| α | α=16 with r=64 (QLoRA convention) or α=2r rule | `[paper]` / `[heuristic]` |
| Dropout | 0.05 (33–65B) / 0.1 (7–13B) | QLoRA `[paper]` |
| LR | 1e-4 – 3e-4; start 2e-4 (7–13B), 1e-4 (33–65B) | QLoRA `[paper]` |
| LR vs full FT | ~10–20× higher than the full-FT LR for the same job | QLoRA 2e-4 vs Tulu/Zephyr full-SFT 5e-6–2e-5; 2405.09673 LR sweeps |
| Schedule | cosine or constant; warmup ratio 0.03–0.1 | handbook convention `[config]` |
| Epochs | 1–3 (small data tolerates 3) | QLoRA `[paper]` |
| Batch | effective 64–128 sequences via grad-accum | QLoRA/handbook `[config]` |

Embeddings and lm_head: leave frozen unless vocabulary changed; if the chat
template adds new special tokens, train embeddings for those tokens or
resize carefully `[heuristic]`.

## 3. QLoRA stack (single-GPU 7–70B fine-tuning)

All from the QLoRA paper `[paper 2305.14314]`:

- **NF4** 4-bit storage for the frozen base (information-theoretically
  matched to normal-distributed weights); compute still in bf16.
- **Double quantization** of the quantization constants (~0.4 bits/param
  saved).
- **Paged optimizers** (paged AdamW 32-bit) to survive activation spikes.
- Results context: 65B fine-tunable on a single 48GB GPU; their best 33B
  models trained in ~24h on one GPU.

Template (7–8B class): NF4 + double-quant, bf16 compute, r=16–64,
α=16–32, dropout 0.05, all-linear targets, lr 2e-4, cosine, warmup 0.03,
1–2 epochs, effective batch 64–128, paged_adamw_32bit, gradient
checkpointing on, loss masked to assistant tokens. Ready-made:
`assets/templates/qlora_sft.yaml`.

## 4. Memory math (rule-of-thumb, label as estimate)

Full FT (AdamW, bf16 weights + fp32 master+moments): ~16 bytes/param →
7B ≈ 112GB before activations — multi-GPU mandatory.
QLoRA: ~0.55 byte/param base (NF4 + constants) + adapters + activations →
7B fits in ~6–8GB plus activation memory; 70B in ~40–48GB `[paper-anchored
estimate]`. Activations scale with micro_batch × seq²-ish; first lever is
micro_batch=1 + grad-accum, second is shorter seq, third is gradient
checkpointing (already on).

## 5. LoRA + DPO

Works and is common practice (TRL supports adapter-as-policy with the
frozen base as implicit reference). Use preference-stage LRs scaled up the
LoRA way: ~5e-6–5e-5 rather than full-DPO's 5e-7 `[heuristic — TRL
community practice; no flagship published config]`. Flag this clearly as
weaker-evidenced. Merge adapters before final evaluation so eval-time
numerics match deployment.

## 6. Pitfalls checklist

1. Only q,v targeted (old default) → silent underfit. Target all linears.
2. LoRA LR copied from full-FT recipes (5e-6) → adapter barely moves.
3. Forgot prompt-loss masking → model imitates users (same as SFT pitfall).
4. Quantized base for training but fp16 merge for eval → small but real
   metric shifts; evaluate the artifact you ship.
5. Chat-template/BOS mismatches between train and eval — the SimPO repo
   documents a double-BOS eval bug class `[config]`. Decode one training
   batch and one eval prompt; compare token IDs.
6. r raised but α left tiny → effective scale α/r collapsed; keep α/r
   roughly constant when changing r.
