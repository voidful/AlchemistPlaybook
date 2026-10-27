# PEFT: LoRA and QLoRA

Verification tags: `[config]`, `[paper]`, `[reported]`.

## Contents
1. LoRA vs full fine-tuning: when each wins
2. LoRA hyperparameters
3. QLoRA stack
4. Memory math
5. LoRA + DPO
6. Pitfalls checklist
7. Hypernetwork-generated LoRA (Program-as-Weights)

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
- **Scarce supervision (~hundreds of labeled examples) → restrict trainable
  capacity.** FAC's controlled trainable-scale ablation at ~200 samples
  `[paper 2602.10388]`: LoRA (41M params) 62.60 AUPRC > head-only (8K)
  49.12 > last-1/2-layers (218/436M) 47.19/45.79 > **full FT (7.5B)
  collapses to 29.02** — large-scale parameter updates amplify spurious
  correlations when data is tiny. In this regime LoRA is not a memory
  hack, it *is* the regularizer; full FT needs data volume.
- **New-language knowledge, same pack, same base: full SFT ≈3–4× the LoRA
  gain** (Nemotron SFT guidebook `[reported — repo ablation at commit
  1c3e781 on the pre-RL SFT checkpoint of Nemotron-3-Nano-30B-A3B]`): full
  SFT at 1e-5 vs LoRA at 1e-4 (rates deliberately unmatched, as each is
  actually run): Hindi MILU +8.63 vs +2.70, Malayalam MILU +15.50 vs +4.19;
  answers written in the target script +74.9pp vs +18.6pp (Hindi math) and
  +80.7pp vs +4.8pp (Malayalam math) — the gap widens with how unfamiliar
  the language is, as expected if adapter capacity is the limit. LoRA's one
  win, far less English instruction-following loss (−0.61 vs −4.08), is
  bought back for full SFT by 20k English IF replay (post-training.md §2).
  Confirms the decision rule above; many small revertible adapters is a
  different question they did not test.

## 2. LoRA hyperparameters

| Knob | Recommendation | Evidence |
| --- | --- | --- |
| Target modules | **All linear layers**: q,k,v,o + gate,up,down. This matters more than rank. | QLoRA ablation: all-linear needed to match full FT `[paper]`; 2405.09673 concurs |
| Rank r | 16–64 for SFT-style adaptation; 64+ (even 256) for CPT-like jobs | QLoRA used r=64 `[paper]`; 2405.09673 swept 16–256; PAW's controlled per-task baselines at 0.6B: r=128 (0.516) ≤ r=64 (0.521) — rank saturates, don't buy past 64 by default `[paper 2607.02512]` |
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

## 7. Hypernetwork-generated LoRA (Program-as-Weights)

PAW `[paper 2607.02512]` trains a 4B "compiler" (Qwen3-4B-Instruct-2507,
fully unfrozen) that reads a natural-language task spec and **emits a LoRA
adapter** (r=64, all 7 linear modules, shared bases, ~38.5M params per
task) for a frozen 0.6B interpreter. Single SFT-style objective — NLL of
the target output under the frozen interpreter wearing the generated
adapter; an earlier RL stage was tried and *deleted* (it mode-collapsed to
one output format regardless of seed prompt). Training config `[paper —
App. G]`: lr 2e-5, AdamW at PyTorch defaults, **no warmup, no schedule**,
batch 16 × accum 3 = 48 effective, 3 epochs over 10M synthetic examples,
bf16 with the adapter-emitting mapper head kept in **fp32 for numerical
stability** (the hypernet analogue of stability.md §6's "fp32 where it
counts").

What a PEFT user can reuse even without building the hypernetwork:

- **Controlled same-data hierarchy** (exact match, 0.6B interpreter):
  generated-LoRA 0.738 > full FT 0.584 > fixed LoRA r=64 0.521 ≥ r=128
  0.516 > r=18 0.424. Two citable points: rank saturates by 64 (§2), and
  when the workload is *many small disjoint tasks*, per-task adapter
  capacity beats one shared full fine-tune.
- **Simplest adapter-generator wins**: every "more expressive" mapper
  variant (per-position aggregation, per-layer bases, adding
  prefix-tuning) underperformed plain mean-pool + shallow residual MLP +
  shared bases `[paper — Table 4]`. Same lesson at hyper-scale as OLMo 2's
  plain-init story.
- **Quantized deployment is measurably cheap**: Q6_K base + Q4_0 adapter
  statistically matched bf16; Q4_K_M base lost 1.3 points at ~507 MB total
  disk `[paper — Table 8]`. They evaluated the *shipped quantized
  artifact* — which is exactly pitfall #4's rule done right.
- Synthetic-eval hygiene: their test set keeps only examples where **two
  independent strong models agree** on the target, and reports the
  generator model's own score (96.1%) as the empirical ceiling — see
  evaluation.md §9.
