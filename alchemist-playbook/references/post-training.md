# Post-training recipes (SFT / preference / RL)

Verification tags: `[config]` official YAML or model card, `[paper]` technical
report, `[reported]` secondary source.

## Contents
1. Algorithm chooser
2. SFT recipes
3. DPO recipes and the β–loss-type coupling
4. SimPO / ORPO / KTO (reference-free family)
5. PPO / RLVR / GRPO
6. Failure modes specific to post-training
7. Evaluation discipline
8. Model merging as a post-training stage
9. The Spectrum-to-Signal Principle (SSP): SFT = diversity, RL = signal
10. Multi-domain RLVR, length control, and offline self-distillation

## 1. Algorithm chooser

| Situation | Use | Why |
| --- | --- | --- |
| Have instruction data, base model | SFT first, always | every open pipeline (Zephyr, Tulu 3, OLMo, SmolLM2) starts here |
| Have preference pairs + VRAM for 2 models | DPO | best-documented, most reproducible (Zephyr, Tulu 3) |
| Preference pairs, tight VRAM | SimPO or ORPO | no reference model |
| Want SFT+alignment in one stage | ORPO | adds odds-ratio penalty to SFT loss |
| Unpaired thumbs-up/down data | KTO | designed for unpaired signals (β default 0.1) |
| Verifiable answers (math, code, constraints) | RLVR (PPO or GRPO) | Tulu 3's final stage; DeepSeek-R1's engine |
| Iterative quality push with a reward model | PPO / iterative DPO | Llama-2-chat style (rejection sampling + PPO) |
| Several good checkpoints, or capabilities to combine | Model merging (§8) | LFM2 runs soup/TIES/DARE/DELLA in parallel, evals, picks; near-free gains |
| Training a sub-3B / on-device model | Distill first, then SFT→preference | distillation + curriculum beat optimizer tuning at small scale (pretraining.md §6) |
| Building a small **reasoning** model (math/code/STEM) | SSP: diversity-first SFT → signal-amplifying RL (§9–§10) | optimize SFT for Pass@K coverage and select the SFT checkpoint by Pass@K, then let MGPO/GRPO sharpen Pass@1 (VibeThinker) |

## 2. SFT recipes

| Run | Base | LR | Schedule | Warmup | Epochs | Eff. batch | Seq | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zephyr-7B SFT | Mistral-7B | 2.0e-5 | cosine | ratio 0.1 | 1 | 16/dev, accum 1 (handbook 8-GPU default ⇒ 128 seqs) | 2048 | `[config]` alignment-handbook config_full.yaml |
| Tulu 3 8B SFT | Llama-3.1-8B | 5e-6 | linear | ratio 0.03 | 2 | 128 | 4096 | `[config]` model card; loss accumulation = **sum**, not mean |
| Tulu 3 70B/405B SFT | Llama-3.1 | 2e-6 | linear | 0.03 | 2 | 128 / 256 | 4096 | `[config]` |
| Llama-2-chat SFT | Llama-2 | 2e-5 | cosine | — | 2 | 64 seqs | 4096 | `[paper]` wd 0.1 |
| SmolLM2 SFT | SmolLM2-1.7B | — | — | — | 2 | ~128 | 8192 | `[paper]` SmolTalk dataset; exact LR not re-verified — treat as `[reported]` |
| LFM2 SFT | LFM2 base 0.35–8B | 3e-5 → 1e-7 | decay | — | 3 | per-size (micro 1 + accum) | 32768 | `[paper]` difficulty-curriculum-ordered data; base already distilled |

Rules that matter more than the LR:

- **Mask the prompt.** Compute loss only on assistant tokens. Llama-2-chat:
  "zero-out the loss on tokens from the user prompt" `[paper]`. Unmasked
  prompts teach the model to imitate users.
- **Scale LR down with model size**: 5e-6 (8B) → 2e-6 (70B) in Tulu 3 —
  same inverse pattern as pretraining.
- **1–2 epochs.** More epochs overfit style. Exception precedent:
  InstructGPT trained SFT 16 epochs — val loss overfit after 1 epoch but
  RM-judged quality kept improving `[paper]`; if downstream (preference)
  selection follows, mild SFT overfitting can be acceptable. LFM2 ran **3
  epochs** on small models `[paper]` — defensible when the data is
  difficulty-curriculum-ordered (pretraining.md §6) and the model is small,
  but treat 3+ as the exception, not the default.
- **Packing**: pack short examples into full sequences with correct
  attention separation; verify by decoding a batch (diagnostics.md).
- NEFTune (uniform noise on embedding, α=5/10/15) is a cheap sometimes-win
  for chat win-rates `[paper 2310.05914]`; off by default.

## 3. DPO recipes

Loss: `-log σ(β [log πθ(yw)/πref(yw) − log πθ(yl)/πref(yl)])`.
β controls KL-anchoring to the reference (SFT) model.

| Run | LR | β | Loss type | Schedule | Warmup | Epochs | Eff. batch | Max len / prompt | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zephyr-7B DPO | 5.0e-7 | 0.01 | standard DPO | cosine | ratio 0.1 | 1 | 8/dev × accum 2 | 1024 / 512 | `[config]` |
| Tulu 3 8B DPO | 5e-7 | 5 | **length-normalized** DPO | linear | 0.1 | 1 | 128 | 2048 | `[config]` model card |
| Tulu 3 70B/405B DPO | 2e-7 | 5 | length-normalized | linear | 0.1 | 1 | 128 / 256 | 2048 | `[config]` |
| SimPO authors' DPO baselines | 3e-7–7e-7 | 0.01 | standard | — | — | 1 | 128 | — | `[config]` SimPO README |
| DPO original paper | ~1e-6 (RMSprop) | 0.1 | standard | — | — | — | — | — | `[paper]` |
| LFM2 direct alignment | 8e-7 → 8e-8 | 5 | **length-normalized** (generalized DPO + APO-zero) | decay | — | — | — | — | `[paper]` |

**The β–loss-type coupling (critical, frequently misunderstood):**
β=0.01 (Zephyr) and β=5 (Tulu 3) are *not* contradictory tuning opinions —
they belong to different loss definitions. Length-normalized DPO divides
log-probs by sequence length, shrinking the logit-difference scale by
~100×, so β must grow proportionally to keep effective regularization
comparable. Never transplant a β across loss variants. Length-normalization
itself exists to kill DPO's length bias (longer answers accumulate larger
log-prob sums).

LFM2 independently lands on the **same β=5 with length-normalized rewards**
`[paper]` (reward divided by response token count: Δ = r(yw)/|yw| −
r(yl)/|yl|), and frames standard DPO and APO-zero as special cases of one
generalized direct-alignment loss (choices of the comparison function f,
margin m, and an absolute-reward term). Two takeaways: (1) further
confirmation of the coupling — length-normalized ⇒ β≈5, standard ⇒
β≈0.01–0.1; (2) if you want both "push the chosen up" and "don't drift"
behavior, the generalized form (relative term + absolute δ term) is more
expressive than vanilla DPO at no extra model cost.

DPO practical rules:
- LR is the kill switch: 1e-5 reliably lobotomizes a 7–8B model into
  repetition (SimPO authors `[config]`). Grid 3e-7 / 5e-7 / 8e-7 / 1e-6.
- Prefer **on-policy preference data annotated by a strong RM**: SimPO v0.2's
  jump came from re-annotating with ArmoRM, not from algorithm changes.
- 1 epoch. Watch response length and a reasoning probe (GSM8K) for the
  classic verbosity / capability-tax failure.
- Memory: policy + frozen reference both resident → gradient checkpointing
  (non-reentrant; Zephyr `[config]`) + ZeRO-3; 70B-scale needs CPU
  offloading (Tulu 3 used stage3 offload conf `[reported]`).

## 4. Reference-free family

**SimPO** `[config — authors' README]`: reward = (β/|y|)·log πθ(y), margin γ.
No reference model. Verified settings:

| Setting | β | γ/β | LR |
| --- | --- | --- | --- |
| Mistral-Base | 2.0 | 0.8 | 3e-7 |
| Mistral-Instruct | 2.5 | 0.1 | 5e-7 |
| Llama3-Base | 2.0 | 0.5 | 6e-7 |
| Llama3-Instruct | 2.5 | 0.55 | 1e-6 |
| Llama3-Instruct v0.2 | 10 | 0.3 | 1e-6 |
| Gemma-2-9B-it | 10 | 0.5 | 8e-7 |

Authors' tuning order: LR first (most critical), then β ∈ [2, 10], then
γ/β ∈ [0, 1] starting at 0.5; batch fixed at 128. Smaller LR (5e-7) for
math-heavy data. Optional SFT-loss term preserves GSM8K but costs chat
win-rate.

**ORPO**: single-stage SFT+preference via odds-ratio penalty (λ a.k.a. β).
Paper (Mistral-7B): λ=0.1, lr 8e-6 `[paper]`. Zephyr-141B-A35B production
config: β=0.05, lr 5e-6, inverse_sqrt schedule, 3 epochs, warmup 100,
adamw_bnb_8bit, max_len 2048/prompt 1792 `[config]`. Note ORPO LR sits
between SFT (1e-5) and DPO (1e-6) — it is doing both jobs.

**KTO**: β=0.1 default; use when only unpaired good/bad labels exist `[paper]`.

## 5. PPO / RLVR / GRPO

**Tulu 3 RLVR (PPO)** — fully published config `[config — model card]`:
lr 3e-7 linear, eff. batch 224, episodes 100K, KL β=0.05, GAE λ=0.95,
γ=1.0, PPO clip 0.2, 4 PPO epochs per batch, value-coef 0.1, grad clip 1.0,
temp 1.0, max len 2048, no-EOS penalty −10, warmup 0. Rewards are
*verifiable*: exact-match math, constraint checkers for IFEval-style
prompts. Value model initialized from the reward model `[paper]`.

**GRPO (DeepSeekMath)** `[paper]`: drop the value model; advantage = group-
normalized reward over G samples per prompt. lr 1e-6, KL coef 0.04, G=64
samples/question, max len 1024, batch 1024.

**DeepSeek-R1** `[paper]`: R1-Zero = GRPO directly on the base model with
rule-based rewards (answer correctness + format tags) — no neural RM, no
SFT. R1 adds a small cold-start SFT, reasoning-RL, rejection-sampled SFT,
then all-scenario RL. Takeaway for users: rule-verifiable reward + GRPO is
the cheapest credible path to reasoning gains; neural RMs invite reward
hacking on long-horizon tasks.

**MGPO (MaxEnt-Guided Policy Optimization)** `[paper 2511.06221 / 2606.16140]`
— a difficulty-aware *advantage reweighting* on top of GRPO, used by both
VibeThinker models. Keep GRPO's clipped, group-relative objective, then scale
each prompt's advantage by an entropy-deviation weight that up-weights
problems the model currently solves near a 50% rate and exponentially
down-weights trivial (all-correct) and impossible (all-wrong) prompts:

```
p_c(q) = (1/G) Σ_i 1[r_i = 1]        # empirical group pass-rate (binary verifiable reward)
w_ME    = exp(−λ · D_ME(p_c ‖ 0.5))   # D_ME = binary KL to a Bernoulli(0.5) max-entropy target
A'      = w_ME · A                    # A = usual group-normalized (r_i − μ_G)/σ_G
```

λ≥0 is the sharpness knob (λ=0 ⇒ w_ME=1 ⇒ plain GRPO). Its **numeric value is
not stated**, nor are G, RL LR, KL coef, clip ε, batch, or step count — do not
invent them. Why it matters: it is a one-line change on any RLVR/GRPO stack,
reusing the binary verifiable reward already present (no extra reward model),
and a smooth differentiable alternative to DAPO-style hard pass-rate filtering
(dropping all-correct/all-wrong groups). Note the 0.5 is the max-entropy
*target* p₀, **not** an entropy or KL coefficient.

RL guardrails: monitor KL to reference, mean reward, and response length
together — reward↑ + KL↑ + length↑ is the hacking signature. Penalize
missing EOS (Tulu 3's −10) to prevent run-on generations.

## 6. Post-training failure modes

| Symptom | Likely cause | Fix anchor |
| --- | --- | --- |
| Outputs become long and empty | DPO/RLHF length bias | length-normalized DPO (Tulu 3) or SimPO; track length in eval |
| Repetition / incoherence after DPO | LR too high | drop to 5e-7 (SimPO README evidence) |
| JSON/format ability lost after preference tuning | over-large LR + weak anchor on instruct base | SimPO v0.2 caveat: lower LR, or switch base (their Gemma run), or raise β |
| GSM8K drops after DPO | chat-data preference shift | add SFT-loss term (costs chat) or mix math prefs; Tulu 3 mixes targeted prefs |
| SFT model parrots prompts | loss not masked | mask prompt tokens (Llama 2) |
| RM score up, humans unimpressed | reward hacking | verifiable rewards, KL watch, fresh on-policy evals |

## 7. Evaluation discipline

Cheap in-loop: held-out loss on preference pairs (DPO accuracy), length
stats, one instruction-following probe (IFEval subset), one reasoning probe
(GSM8K subset). Full evals only on stage exits. Compare against the
*stage input* model, not only the final target — each stage must justify
itself (Tulu 3's per-stage tables are the model to imitate).

For the full SFT and DPO/RLHF/RL monitoring catalogs (instruction-following,
format, factuality, refusal, reward-hacking, product metrics), the benchmark-
by-capability tables, and the post-training minimal eval suite, see
`references/evaluation.md` §5–8.

## 8. Model merging as a post-training stage

Weight-space merging is now an explicit, *evaluated* stage at the small-model
frontier — not an afterthought.

- **LFM2 `[paper]`** runs merging as its final post-training stage: it applies
  **model soup, task arithmetic, TIES-Merging, DARE, and DELLA** in parallel
  to candidate checkpoints and selects the best by evaluation. No single
  technique wins universally — run several, eval, pick.
- **OLMo 2 `[paper]`** model-souped (uniform weight average) 3 anneal runs
  with different seeds/mixes into the final base checkpoint
  (pretraining.md §4).

When to reach for each:
- **Several near-equivalent checkpoints** (different seeds, data mixes, or
  post-training branches of one parent) → **model soup** (uniform average).
  Nearly free and usually ≥ the best single member. Requires a shared
  ancestor — don't soup unrelated models.
- **Combine separately-trained capabilities** (e.g. math-tuned + chat-tuned
  off the same base) → **task arithmetic** (sum the task vectors θ_ft − θ_base)
  or **TIES** (trim small deltas, resolve sign conflicts, then merge).
- **Many task vectors interfering** → **DARE / DELLA** randomly prune + rescale
  delta weights before merging to cut interference.

Always evaluate the merge against **each input model on every axis you care
about** — merging can silently trade one capability for another. `mergekit`
is the standard tooling `[reference]`.

## 9. The Spectrum-to-Signal Principle (SSP): SFT = diversity, RL = signal

The "SFT first, always" default (§1) tells you to *do* SFT before RL. SSP
(VibeThinker `[paper 2511.06221]`) tells you *what to optimize SFT for* when
the goal is a reasoning model: not single-shot accuracy, but **coverage**.

- **Spectrum (SFT):** SFT's job is "not to converge on a single optimal
  answer, but to generate a rich and diverse 'spectrum' of plausible
  solutions" — maximize **Pass@K** (does *any* of K samples solve it).
- **Signal (RL):** RL then "identif[ies] and amplif[ies] the correct 'signal'
  from within this pre-established spectrum" — sharpen **Pass@1**. RL only
  redistributes probability onto paths the SFT model can already reach, so
  "a model with high Pass@K … raises the upper bound of what RL can achieve."
- **Actionable selection rule:** pick the SFT checkpoint by **Pass@K on a
  held-out probing set, not by val-loss or Pass@1.** The paper argues that
  selecting the Pass@1-maximizing SFT checkpoint "artificially constrains the
  potential performance ceiling for the subsequent RL phase," whereas a
  diversity-optimized checkpoint is "a superior prerequisite for RL … more
  fertile ground for optimization." `[paper]`
- **Diagnosing an RL plateau:** if Pass@1 saturates under RL, widen the SFT
  spectrum upstream (more diverse SFT, select on Pass@K) instead of tuning RL
  harder — RL cannot invent reasoning paths SFT never produced.
- The 1.5B run reports the diversity-optimized model reaching SOTA on *both*
  Pass@K and Pass@1 — diversity and accuracy were not in tension there
  `[reported — single run, no controlled ablation]`.

Implementation: the SFT side is "Two-Stage Diversity-Exploring Distillation"
(pretraining.md §6); the RL side is MGPO (§5). Honest caveat: SSP's evidence is
two strong runs from one team, not an ablation isolating the principle — treat
the Pass@K checkpoint-selection rule as a high-value `[heuristic]` to A/B on
your own task, not a proven law.

## 10. Multi-domain RLVR, length control, and offline self-distillation

VibeThinker-3B `[paper 2606.16140]` extends the SSP recipe with three reusable
post-training moves (it shares the SSP/MGPO core with the 1.5B):

- **Sequential multi-domain RLVR, one deterministic verifier per domain** —
  math = final-answer verification (+ LLM-judge), code = sandbox execution
  against test cases, STEM = answer-matching + option verification. Domains
  trained **sequentially** (math → code → STEM) under one MGPO loop, with
  **zero learned reward model**. Reinforces §5's "rule-verifiable reward, no
  neural RM" lesson and shows it composing across domains. RL ran in a single
  **64K** long-context window, replacing the usual progressive context-length
  staging. (Numeric RL hyperparameters — LR, KL, group size G, batch, steps —
  are **not stated**; do not assume them.)
- **Long2Short reward redistribution** `[paper]` — a post-hoc brevity reward
  applied to *correct* trajectories only, magnitude **λ=0.2**, biasing toward
  shorter correct solutions without a hard length cap. An RL-stage analogue of
  the length-bias remedies in §3/§6: shape length via reward, not truncation.
- **Offline self-distillation to consolidate specialists into one student**
  `[paper]` — collect verified trajectories from the math/code/STEM RL
  checkpoints and re-train a single student by plain SFT, selecting traces by a
  **learning-potential** score `S_LP = −(1/|y|)·Σ_t log π_student(y_t|q,y_<t)`
  (length-normalized NLL — higher = not yet well-modeled by the student),
  ranked *within* domain-specific length buckets and keeping the middle-to-high
  band. This is an alternative to weight-space merging (§8) for unifying
  specialists: distill *behavior* instead of averaging *weights*. (Self-
  distillation LR/epochs/batch are **not stated**.)
