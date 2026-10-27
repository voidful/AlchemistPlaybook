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
11. Asynchronous agentic RL and cross-stage distillation (GLM-5)
12. Turn-level local RL from SFT trajectories (PivotRL)
13. Multi-teacher on-policy distillation: unifying domain specialists (Agents-A1, MiniCPM5)
14. Synthetic-data curation by feature coverage (FAC / SAE)
15. When on-policy distillation works: preconditions, mechanism, recovery (Rethinking OPD)
16. Reasoning-SFT data curation: question quality, answer multiplicity, base-model fit (OpenThoughts / GRAPE)
17. Self-evolving hindsight-skill distillation for agentic RL (SEED)

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
| Long-horizon **agent** tasks (SWE, terminal, search) with slow/uneven rollouts | Asynchronous decoupled RL (§11) | synchronous RL stalls on straggler trajectories; GLM-5's async stack + stability mechanisms is the published recipe |
| Sequential RL stages erode earlier capabilities | On-policy cross-stage distillation as the final stage (§11) | GLM-5 recovers SFT/Reasoning-RL/General-RL skills by distilling from each stage's checkpoint as teacher |
| Have agent SFT trajectories but cannot afford full-trajectory RL; or agent SFT is tanking OOD ability | Turn-level local RL on the same trajectories (PivotRL, §12) | matches E2E-RL accuracy at 4× fewer rollout turns and removes SFT's OOD regression (NVIDIA/Nemotron-3) |
| Several *domain-specialist agents* must become one model; mixed-domain SFT is regressing IF/general ability | Domain teachers → multi-teacher on-policy distillation (§13) | Agents-A1: OPD repaired the multi-domain-SFT regressions weight merging can't route around |
| Small model must gain several RL-trained capabilities without forgetting | Parallel capability-specialist RL teachers → OPD back into one student (§13) | MiniCPM5-1B: +16 avg over SFT at 1B, teacher prompts reused as distillation data |
| Building the SFT *dataset* for a reasoning model | OpenThoughts rules (§16): top 1–2 question sources, LLM difficulty/length filters, ×16 teacher answers, skip answer verification | largest controlled data ablation (1000+ runs); SFT-only result beat RL-trained 7B peers |
| Same instructions exist with several candidate responses (multiple datasets/teachers) | GRAPE (§16): keep the lowest-base-model-perplexity response, then plain SFT | beats picking the strongest teacher's answers on every base tried |
| Picking (or debugging) an OPD teacher for a single student | Check the two preconditions first (§15) | Rethinking-OPD: pattern-mismatched or same-pipeline teachers fail even when they benchmark higher — OPD is reverse KL at *student-visited* states, so the teacher must differ there |
| Agentic RL where sparse episode rewards leave intermediate decisions unguided, and no stronger teacher exists | Self-evolving hindsight-skill OPD joint with GRPO (SEED, §17) | dense token-level signal from the policy's *own* trajectory analysis; beats GRPO on ALFWorld/WebShop/search-QA at 1.7–7B |
| Agentic-coding post-training on a base that pretrained at short context with no agent traces; SFT keeps growing to compensate and later RL gains shrink | Insert a 256K **agentic midtraining** stage before SFT (pretraining.md §4, Nemotron 3.5) and keep SFT short | +3.8 SWE-bench Verified / +7.0 Multilingual / +3.5 Pro after identical SFT at 30B-A3B; traces capped at 10–15% of the blend so the stage does not become a long SFT; tool responses masked |
| Any stage above when the user has not fixed a trainer, or asks how to implement, launch, or evaluate it | NVIDIA NeMo stack: Nemotron steps and recipes (Megatron-Bridge, AutoModel, NeMo-RL) + NeMo Gym environments (nemo-stack.md) | default implementation route (repo decision 0007); the recipe values above map onto its config keys (nemo-stack.md §9); the user's existing trainer still wins |

## 2. SFT recipes

| Run | Base | LR | Schedule | Warmup | Epochs | Eff. batch | Seq | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zephyr-7B SFT | Mistral-7B | 2.0e-5 | cosine | ratio 0.1 | 1 | 16/dev, accum 1 (handbook 8-GPU default ⇒ 128 seqs) | 2048 | `[config]` alignment-handbook config_full.yaml |
| Tulu 3 8B SFT | Llama-3.1-8B | 5e-6 | linear | ratio 0.03 | 2 | 128 | 4096 | `[config]` model card; loss accumulation = **sum**, not mean |
| Tulu 3 70B/405B SFT | Llama-3.1 | 2e-6 | linear | 0.03 | 2 | 128 / 256 | 4096 | `[config]` |
| Llama-2-chat SFT | Llama-2 | 2e-5 | cosine | — | 2 | 64 seqs | 4096 | `[paper]` wd 0.1 |
| SmolLM2 SFT | SmolLM2-1.7B | — | — | — | 2 | ~128 | 8192 | `[paper]` SmolTalk dataset; exact LR not re-verified — treat as `[reported]` |
| LFM2 SFT | LFM2 base 0.35–8B | 3e-5 → 1e-7 | decay | — | 3 | per-size (micro 1 + accum) | 32768 | `[paper]` difficulty-curriculum-ordered data; base already distilled |
| EvoLM SFT | myllama 1B / 4B (after math CPT) | 1e-5 / 7.5e-6 | cosine | ratio 0.1 | 1 (swept 1–32) | 128 | 2048 | `[config]` LLaMA-Factory YAML; full FT + ZeRO-3; math CoT distilled from Qwen2.5-Math-7B-Instruct |
| Agents-A1 full-domain SFT | Qwen3.5-35B-A3B | 1e-5 | cosine + warmup | ratio 0.05 | 1 | 16 seqs | 131072 | `[paper]` Table 3; wd 0.1, response-token loss only, sample packing; ~100K agent trajectories avg 45K tokens |
| OpenThinker3-7B (reasoning distillation) | Qwen2.5-7B-Instruct | 8e-5 | cosine | ratio 0.1 | 5 | 512 seqs | not stated | `[config]` HF model card + paper App. D Table 9; AdamW (0.9, **0.999**), wd 0, packing; 1.2M QwQ-32B traces (§16) |
| GRAPE-curated SFT | Llama3.1-8B / Qwen2.5-7B / Mistral-7B | 1e-5 | — | — | 1 | 256 | — | `[paper]` App. D; batch 256, micro 2; data = lowest-base-model-perplexity responses (§16) |
| Nemotron 3.5 agentic SFT (midtraining eval protocol) | N-3.5 Lightning-30B-A3B / Super-120B-A12B, base or midtrained | 2e-5 | cosine → 5e-6 | — | 33B tokens (short) / 100B (full) | 64 seqs | 262144 | `[paper 2610.22019]` Table 17; Adam (0.9, 0.95), wd 0.1, clip 1.0; ~430K trajectories from several scaffolds, mostly GLM-5.2-generated; assistant turns + tool calls in the loss, tool responses masked |

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
  but treat 3+ as the exception, not the default. EvoLM's controlled epoch
  sweep (1B/4B, math, epochs ∈ {1,2,4,8,16,32}) quantifies the trade
  `[paper 2506.16029]`: in-domain keeps rising to ~8 epochs, but **OOD
  peaks at 2–4 epochs then declines**, and over-SFT'd checkpoints gain
  *less* from subsequent RL (their Takeaways 7–8) — if RL follows, stop SFT
  early rather than squeezing in-domain.
- **Packing**: pack short examples into full sequences with correct
  attention separation; verify by decoding a batch (diagnostics.md).
- **Mask errors, keep them in context** (GLM-5 agent-trajectory SFT
  `[paper]`): erroneous segments inside otherwise-good trajectories are
  *retained in the input* but *excluded from the loss* — the model sees
  mistakes and the recovery that follows, learning self-correction without
  ever being trained to reproduce the mistake. Generalizes the prompt-mask
  rule: loss-mask anything you want conditioned on but not imitated.
- **Mask tool responses; train on tool calls** (Nemotron 3.5 `[paper
  2610.22019]`): including environment outputs (file dumps, test logs,
  tracebacks) in the loss cost ≈10 SWE-bench Verified pass@1 points after
  identical SFT (27.5 vs 37.8) in a 100B-token midtraining ablation —
  predicting noisy tool output takes capacity from the agent's own
  actions. Same rule GLM-5 applies in RL (§11). The *world-model* line of
  work that trains on tool responses on purpose (CWM, daVinci-Dev) did not
  work "out of the box" for them — treat it as a separate recipe, not a
  default.
- **SFT length trades against RL plasticity.** The Nemotron team reports
  (and cites Liu 2026b, Kang 2025, Zhang 2026 for) very long SFT stages
  reducing later RL gains `[paper 2610.22019; the cited works are not
  independently verified here]`; EvoLM's epoch sweep above is the
  controlled version. Their structural answer: move long-context and
  agent-format exposure into a midtraining stage with traces capped at
  10–15% of the blend, so SFT stays at 33–100B tokens (pretraining.md §4).
- **Adapting an already post-trained model (Nemotron SFT guidebook)**
  `[reported — repo ablation at commit 1c3e781, Nemotron-3-Nano-30B-A3B
  instruct, Hindi / Malayalam persona-MCQ packs; implementation in
  nemo-stack.md §4]`: **constant 1e-5**, chosen over cosine so that every
  intermediate checkpoint is a comparable model (under cosine the LR at
  step k depends on the planned total); at the same early checkpoint 1e-5 /
  5e-6 / 3e-6 gave Hindi MILU +2.41 / +2.18 / +1.65, none destructive.
  **50k–100k samples** split 50/50 English:target — 20k already reached
  91% of the knowledge gain of 200k, but the reasoning-off subset must stay
  large enough to hold mode integrity (10% of 20k = 2k samples leaked
  reasoning in 20.1% of cases vs 4.0% at 100k) and script fidelity kept
  climbing to 100k; beyond 100k nothing improved. Keep a **90:10
  reasoning-on : reasoning-off** blend, never 100% on. **Replay 20k English
  instruction-following samples** in the pack: knowledge and math hold
  either way, but instruction following regressed on every run (IndicIFEval
  English −12.2 → −8.4 with replay on the released checkpoint) and replay
  buys back only part of it — the capability was installed by RL, not SFT;
  on the pre-RL internal SFT checkpoint the same replay repairs it to
  within noise. Planning rule: **starting from an RL-trained checkpoint,
  budget an RL stage after SFT.** Replay had a measured cost on open-ended
  target-language generation (IndiVibe 75.9 → 49.1), so evaluate both arms.
  Report **language fidelity** — the share of free-form answers in the
  expected script — next to accuracy (evaluation.md §5). Full SFT, not
  LoRA, for new-language knowledge (peft.md §1).
- **Nemotron team SFT answers** `[reported — Nemotron repo FAQ]`: fine-tune
  from the post-trained checkpoint unless you intend to re-run the official
  post-training stages (a domain fine-tune alone does not recover what
  post-training added); Super 3's published full SFT runs **1e-5 constant
  after a 30,000-sample linear warmup**, 1/45 of its 4.5e-4 pretraining
  peak — convert warmup to tokens and compare with the dataset, because a
  warmup of billions of tokens on a dataset far below tens of billions is
  mostly warmup and overfits; narrow-domain SFT on an MoE should keep the
  mixture close to the official post-training blend (take a slice from
  every other domain and reassign it to the new one) to reduce
  expert-collapse risk — a mixture strategy, not a guarantee.
- **Scale SFT hyperparameters with dataset size, not just model size.**
  OpenThoughts' scale-dependent ladder `[paper — App. D Table 9]`, used
  across their 1000+ reasoning-SFT runs on a 7B: <1K samples → LR 1e-5, batch 32, 13 epochs;
  1K–3.16K → 2e-5 / 96 / 7; 3.16K–31.6K → 4e-5 / 128 / 5; >31.6K (incl.
  the 1.2M run) → 8e-5 / 512 / 5 + packing. LR and batch *grow* and epochs
  *shrink* with dataset size — the many-epoch exception for tiny sets is
  published precedent, and 8e-5 for large-scale reasoning distillation
  sits well above the Tulu-3-style 5e-6 chat-SFT default (long fresh CoT
  traces tolerate it; their runs also kept β₂=0.999 and wd 0). Treat this
  ladder as the reasoning-distillation regime, not a chat-SFT default.
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

**IPO** `[paper 2310.12036, Azar et al.]` — the other reference-model DPO
variant worth knowing: replaces DPO's logistic loss with a **squared loss**
pushing the log-ratio margin toward a fixed target (τ plays β's role). The
objective is bounded, so it does not require the Bradley–Terry assumption
and is more robust when preferences are **near-deterministic or noisy** —
the regime where DPO overfits by driving margins to infinity. TRL ships it
as `loss_type="ipo"`. Same warning as the β–loss-type coupling above: IPO's
τ is *not* a DPO β — never transplant values across the two losses.

DPO practical rules:
- LR is the kill switch: 1e-5 reliably lobotomizes a 7–8B model into
  repetition (SimPO authors `[config]`). Grid 3e-7 / 5e-7 / 8e-7 / 1e-6.
- Prefer **on-policy preference data annotated by a strong RM**: SimPO v0.2's
  jump came from re-annotating with ArmoRM, not from algorithm changes.
  Labels from an AI annotator instead of humans (**RLAIF**, Constitutional-AI
  lineage `[paper 2309.00267 / 2212.08073]`) are standard practice —
  ArmoRM re-annotation above is exactly this pattern; the risk it inherits
  is judge bias, so keep a small human-checked slice.
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

**EvoLM PPO (small-scale RLVR, fully config-verified)** `[config —
zhentingqi/evolm verl scripts + paper Table 8]`: actor lr 2e-6 (1B) /
1e-6 (4B), critic lr = **10× the actor's** (2e-5 / 1e-5), KL coef 1e-4,
**warmup-constant** LR schedule (the only stage in their pipeline without
decay), batch 1024 (1B) / 2048 (4B) with mini-batch = batch, max
prompt/response 1024/1024, binary verifiable math reward. The scaling
findings are the citable part `[paper 2506.16029]`: gains saturate at
**4–8 epochs or 50–100K examples**; past ~4 epochs the correct-ratio keeps
rising while **Pass@16 degrades** — PPO *sharpens* the output distribution
rather than expanding capability (the quantitative counterpart of §9's SSP
claim that RL amplifies what SFT can already reach). Failure mode for the
symptom table: scaling the RL *dataset* to 350–400K examples collapsed
runs — response length grew until it exceeded the context window; scaling
*epochs* instead was stable. And under a fixed downstream data budget,
examples allocated to SFT buy in-domain at the cost of OOD while examples
allocated to RL buy OOD (best OOD split in their sweep: 10K SFT / 90K RL)
`[paper — Takeaway 11]`.

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

**GLM-5 Reasoning RL (GRPO + IcePop-style mismatch masking, no KL)**
`[paper 2602.15763]`: GRPO backbone with three deviations worth knowing.
(1) **Train↔inference mismatch masking**: separate engines mean the sampling
distribution ≠ training distribution; GLM-5 computes the per-token ratio
ρ = π_train_old/π_infer_old and **zeroes the loss** for tokens with
ρ ∉ [1/β, β], β=2 (IcePop mechanism). (2) **KL term removed entirely** to
speed RL improvement — the mismatch mask plus on-policy sampling substitute
for the anchor. (3) **Asymmetric PPO clip**: ε_low=0.2, ε_high=0.28
(clip-higher, DAPO-style, keeps upside exploration). Fully on-policy,
group size 32, batch 32. Mixed-domain RLVR (math / science / code /
tool-integrated reasoning) trained *jointly* with roughly balanced mixture
and binary outcome rewards — they report stable simultaneous gains in all
four, a counterpoint to VibeThinker-3B's sequential domains (§10).
Difficulty filtering mirror of SFT: keep problems the previous model
(GLM-4.7) rarely solves but stronger teachers can.

**Train–inference consistency is a first-class RL stability axis** (GLM-5
`[paper]`): with sparse attention (DSA), a *non-deterministic* CUDA top-k in
the indexer crashed RL within a few steps (entropy collapse, sharp
performance drop); switching to deterministic `torch.topk` fixed it, and
they freeze the indexer during RL. Same family as MoE routing replay. The
general lesson for any RL stack: every stochastic/kernel-level divergence
between rollout engine and trainer (top-k ties, tokenization round-trips,
precision) is a latent RL killer — see also TITO in §11.

**Nemotron 3.5 short RLVR persistence check (GRPO, settings stated)**
`[paper 2610.22019, App. D]`: 20 GRPO steps with a **leave-one-out
baseline and no reference-policy KL**, 256 prompts × 16 generations per
step, 32K-token output cap inside a 131K context, LR 1e-6 with 10 warmup
steps, prompts from 14 verifiable environments (tool use, code, SWE and
search pivots, instruction following, reasoning, formatting). The part to
copy is prompt selection: the seed blend was **reward-profiled against both
starting checkpoints and only prompts with mixed rewards for each were
kept** — the offline cousin of the pivot / MGPO / DAPO mixed-outcome
filters, applied before training so neither model sees trivial or
unsolvable prompts. Used only to ask whether a midtraining advantage
survives the start of RL: it did (≈+2 SWE-bench Verified held at every
step; the base's Terminal-Bench 2.1 pass@8 fell 46.1 → 39.3 while the
midtrained model's held at 45–46), and training-time signals favored the
midtrained start (+4.3% mean reward, 12% shorter generations, ≈10× fewer
format violations, ≈15% faster per trajectory). Single seed, 20 steps —
**not** evidence that midtraining raises RL headroom, and the authors say
so.

**NeMo-RL GRPO as shipped for Nemotron models** `[config — NVIDIA-NeMo/Nemotron
recipe configs at commit 1c3e781; full tables in nemo-stack.md §5]`: the Nano 3 / Lightning 3.5 / Super 3 recipes run GRPO with **KL 0**,
a **constant LR of 3e-6 / 4e-6 / 3e-6** after a 10-iteration warmup, Adam
(0.9, 0.999) ε 1e-8, **weight decay 0**, clip 1.0, asymmetric clip 0.2 /
0.28, leave-one-out baseline, normalized rewards, token-level loss,
importance-sampling correction (truncated IS ratio 5), 16 generations per
prompt at 128 / 512 / 256 prompts per step (GBS 2,048 / 8,192 / 4,096),
sequence lengths 49,152 / 73,728 / 49,152→65,536, `overlong_filtering`
**off**, and MoE hygiene that protects rollout-vs-trainer logprob
agreement — router **frozen**, router dtype fp32, load balancing "none",
aux-loss 0 (the config notes that sequence aux-loss "causes logprob error
divergence for grpo"); Lightning keeps 5 repeated MTP layers training at
loss scale 0.3. These LRs sit **above** the dense 3e-7–1e-6 PPO/GRPO
anchors in text-llm-baselines.md §3: hybrid-MoE policies, no KL anchor,
constant schedule — read the regime before transferring either way.
Multi-environment RLVR is their default — 6 Gym environments for Nano and
Lightning, 21 for Super — because "single-environment training leads to
severe regressions on other benchmarks" `[reported]`; Super's RLHF stage is
the one place KL is non-zero (1e-4, LR 1e-6, GenRM reward).

RL guardrails: monitor KL to reference, mean reward, and response length
together — reward↑ + KL↑ + length↑ is the hacking signature. Penalize
missing EOS (Tulu 3's −10) to prevent run-on generations.

**Overlong-response control on reasoning RLVR** (MiniCPM5-1B `[paper —
OpenBMB/MiniCPM README + figures]`): the SFT checkpoint ran to the
max-token budget on 42.3% of AIME-2025 and 47.7% of HMMT responses —
generating until truncation instead of committing to an answer. Their RL
recipe attacks it with a **two-stage max-response-length schedule**:
reasoning RL stage 1 at 30,720 max tokens with a **repetition penalty**,
then stage 2 (accuracy-focused) with the cap raised to 38,912 — rather
than starting at the full budget. After RL+OPD the overlong rate fell to
9.6% / 7.2% (AIME-2025 / HMMT) while accuracy more than doubled (§13).
Same lever family as Long2Short's λ=0.2 brevity reward (§10) and Tulu 3's
EOS penalty: shape length via reward and schedule, not truncation alone.
The reward's exact shape is **not stated** (an explicit length-penalty
term is `[reported — practitioner-talk reading]`, not in the official
README), nor is the repetition-penalty coefficient.

## 6. Post-training failure modes

| Symptom | Likely cause | Fix anchor |
| --- | --- | --- |
| Outputs become long and empty | DPO/RLHF length bias | length-normalized DPO (Tulu 3) or SimPO; track length in eval |
| Repetition / incoherence after DPO | LR too high | drop to 5e-7 (SimPO README evidence) |
| JSON/format ability lost after preference tuning | over-large LR + weak anchor on instruct base | SimPO v0.2 caveat: lower LR, or switch base (their Gemma run), or raise β |
| GSM8K drops after DPO | chat-data preference shift | add SFT-loss term (costs chat) or mix math prefs; Tulu 3 mixes targeted prefs |
| SFT model parrots prompts | loss not masked | mask prompt tokens (Llama 2) |
| RM score up, humans unimpressed | reward hacking | verifiable rewards, KL watch, fresh on-policy evals |
| RL run collapses; responses grow until they hit the context window | RL dataset scaled far past saturation | scale RL *epochs*, not dataset size (EvoLM: collapse at 350–400K examples; saturation 4–8 ep / 50–100K) |
| Reward climbing but a large share of responses hit the max-token cap | no length shaping under a hard cap — model generates until truncation instead of committing | repetition penalty + staged max-length schedule (MiniCPM5 30,720→38,912, §5); Long2Short brevity reward λ=0.2 (§10) |
| Multi-domain agent SFT regresses IF / general QA | conflicting reasoning patterns in one SFT mix (single-turn long-thinking vs multi-turn tool-use) | domain teachers + multi-teacher OPD (§13, Agents-A1) |
| OPD loss falls but student benchmarks flat despite a stronger teacher | teacher fails an OPD precondition: thinking-pattern mismatch, or same-pipeline teacher with nothing new at student-visited states | check top-k overlap ratio + entropy gap; swap teacher, or off-policy cold start + teacher-aligned prompts (§15, Rethinking-OPD) |
| Coding agent fills the context window or hits turn caps; longer SFT barely moves SWE-bench | base pretrained at short context with no agent traces — SFT alone cannot supply repository-scale context use | 256K agentic midtraining with masked tool responses before SFT (pretraining.md §4): SWE-bench Pro context-fill failures 4.1% → 0.6%, turn-cap hits 11 → 1, Verified +3.8 after identical SFT |
| GRPO on an MoE: rollout-vs-trainer logprob error grows, run destabilizes | sequence-level aux-loss load balancing active during RL shifts routing between inference and training | freeze the router, fp32 router dtype, load balancing "none", aux-loss 0 (Nemotron NeMo-RL configs, nemo-stack.md §5); keep importance-sampling correction on |
| Instruction following drops after domain or language SFT of an instruct model, and replay does not recover it | the capability was installed by RL, not SFT — SFT replay cannot put it back | budget an RL stage after SFT; replay still protects knowledge and math (Nemotron SFT guidebook, §2) |

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
is the standard tooling `[reference]`. When the specialists are *agents*
with conflicting reasoning patterns (not near-equivalent checkpoints),
weight-space merging has no published success — the working recipe is
multi-teacher on-policy distillation with domain routing (§13, Agents-A1;
"teachers are not merged at the parameter level" `[paper]`).

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

## 11. Asynchronous agentic RL and cross-stage distillation (GLM-5)

GLM-5 `[paper 2602.15763]` is the most complete published recipe for RL on
long-horizon agent tasks (SWE, terminal, multi-hop search; 10K+ verifiable
environments). Two problems dominate: rollouts are *slow and wildly uneven*
(GPU idle), and asynchrony makes training *off-policy* (instability). Their
answers:

**Throughput side (why async):** training and inference engines run on
separate GPUs; inference generates continuously; a batch trains whenever
enough trajectories accumulate; new weights push to rollout engines every K
updates. A central Multi-Task Rollout Orchestrator (on the slime framework)
registers each task's rollout/reward logic as a microservice, balances
per-task sampling ratios, and sustains >1K concurrent rollouts. Tail-latency
tricks that transfer: FP8 rollout inference, multi-token prediction (biggest
win on small-batch stragglers), and prefill/decode disaggregation so heavy
prefills never stall ongoing decodes; DP-aware routing pins each agent
instance to one DP rank (consistent hashing) for KV-cache reuse.

**Stability side (the checklist to steal):**
- **Objective**: group-wise policy optimization — K traces per problem,
  advantage = r − group mean; **environment/tool-output tokens excluded from
  the loss** (optimize only what the model generated).
- **TITO (token-in-token-out)**: the trainer consumes the rollout engine's
  exact token IDs; never re-tokenize text round-trips. Re-tokenization
  silently corrupts action↔reward alignment.
- **Double-sided importance masking**: with rollout engines several updates
  stale, tracking true π_old is infeasible; reuse the *rollout logprobs* as
  the behavior policy, compute r_t = π_θ/π_rollout, and **zero out** tokens
  with r_t ∉ [1−ε_l, 1+ε_h] (mask, not clip — simpler than IcePop and
  stable without historical checkpoints).
- **Staleness cut**: log which weight versions produced each trajectory;
  drop samples whose oldest version lags the current policy by more than a
  threshold τ.
- **Environment-noise hygiene**: record failure causes; drop samples that
  failed from sandbox/env crashes (not model error). For group methods,
  pad the group by repeating valid samples if >half survive, else drop the
  whole group — spurious negative rewards from broken environments are
  reward noise, not signal.
- **Optimizer reset** after each rollout-engine weight sync — each sync
  redefines the optimization problem, so stale Adam moments mislead.

**On-policy cross-stage distillation (final stage, anti-forgetting):**
sequential stages (SFT → Reasoning RL → Agentic RL → General RL) each erode
predecessors' skills. GLM-5's fix: a last pass where the *final checkpoints
of earlier stages act as teachers* on their own training prompts, using the
GRPO machinery with the advantage replaced by
`Â = sg[log π_teacher(y_t|·) − log π_student(y_t|·)]` on student-sampled
tokens (on-policy distillation). Group size drops to **1** (no group needed
— the teacher gap *is* the advantage), batch 1024. This is the
weight-free alternative to §8 merging and the multi-teacher generalization
of §10's offline self-distillation: distill *behavior from every stage you
cannot afford to forget*.

GLM-5's General RL is also a useful reward-design reference: a **hybrid
reward system** — rule-based checks + outcome RMs (low variance, hackable) +
generative RMs (robust, high variance) — across three objective tiers
(foundational correctness → emotional intelligence → task-specific quality),
plus **human-written exemplars as style anchors** to counter convergence
toward "model-like" prose `[paper]`.

## 12. Turn-level local RL from SFT trajectories (PivotRL)

PivotRL `[paper 2603.21383, NVIDIA]` sits at the third point of the agent
post-training triangle: SFT is cheap but degrades out-of-domain ability;
end-to-end agent RL (§11) preserves OOD but pays for full-trajectory
rollouts; PivotRL reuses **existing SFT trajectories** as RL states and pays
only for single-turn rollouts. It is the production workhorse (alongside SFT
and E2E RL) for Nemotron-3-Super-120B-A12B agentic post-training `[paper]`.

The documented problem it fixes: on identical agent data (conversational
tool use, agentic coding, terminal, search; base
Qwen3-30B-A3B-Thinking-2507), SFT gained +9.94 in-domain but **−9.83 on
OOD** (average over 8 benchmarks — IFBench, AIME25, MATH500, LiveCodeBench,
SciCode, MMLU-Pro/ProX, WMT24++; Table 2. The paper's intro says −9.48
once — an internal inconsistency; the abstract's +10.04 gap confirms
−9.83). PivotRL gained +14.11 in-domain with **+0.21 OOD**, worst
single-benchmark drop −3.12 vs SFT's worst of **−64.48** (AIME25 after
terminal-domain SFT), and matched E2E RL on SWE-Bench (32.67%) with **~4×
fewer rollout turns and ~5.5× less wall-clock** `[paper]`. Quote this to
any user whose agent SFT is eroding general capability (diagnostics.md).
Two honest nuances: in-domain PivotRL is *not* uniformly better — SFT beat
it on SWE-Bench in-domain (37.40 vs 32.67); the win is the OOD + compute
profile. And reported in-domain numbers are **best-checkpoint ("peak")
values, not final-step** `[paper — Table 3/6 captions]` — mind that when
comparing against papers that report final checkpoints.

Mechanism — two changes to naive "RL on demonstration turns", each fixing a
measured bottleneck:

1. **Pivot filtering (offline, before RL).** Split every SFT trajectory at
   assistant-turn boundaries into (state, demonstrated-action) candidates.
   Under the frozen reference policy, sample K local rollouts per state,
   verifier-score them, and keep only states with **mixed outcomes**
   (empirical reward variance > 0) **and** success mean below a difficulty
   threshold λ_diff. Rationale is arithmetic, not taste: with binary rewards
   and group-normalized advantages, an all-pass or all-fail group has
   advantage ≡ 0 — they measured **71% of random turns yield zero gradient**
   on τ²-bench/SWE-Bench. This is the offline sibling of MGPO's p₀=0.5
   weighting (§5) and DAPO-style group filtering: all three spend compute
   where outcome variance lives.
2. **Functional-equivalence reward, not exact match.** Reward 1[action ∈
   verifier-accepted set] — normalized string/schema check, task-specific
   equivalence rule, or a lightweight LLM judge — instead of exact string
   match with the demonstration. Exact-match local RL performed *worse than
   SFT* (57.34 vs 58.44 on τ²-bench `[paper]`): many tool calls / shell
   commands / queries are locally correct without reproducing the golden
   string. Their theory: functional reward shifts probability mass onto the
   acceptable-action set while preserving the reference policy's ordering
   elsewhere — which is why OOD survives. The verifier can be surprisingly
   coarse and still work: their SWE-Bench local verifier matches the
   **tool-call name only** (search/open/edit/run — no argument or patch
   checking; final success still judged by the full harness), while
   Terminal-Bench used schema validation + normalized string similarity +
   an equivalence LLM-judge `[paper — App. A.2]`.

Objective: standard GRPO-style clipped loss over G sampled actions per pivot
state, group-normalized binary rewards, **plus β·KL to the frozen reference
π₀** (kept, unlike GLM-5's reasoning RL — anchoring matters more when
training states come from off-policy demonstrations). Verified numerics
(Appendix A.2, SWE-Bench run): **G=16**, batch 1024 = 64 prompts × 16
generations at 1 turn/sample, reaching 32.67% at step ~130 with ~133K
cumulative rollout turns — vs the E2E-RL baseline's batch 512 = 16 prompts
× G=32 at 12–25 turns/trajectory, ~542K turns to the same score. **K (the
offline profiling rollout count), λ_diff, LR, β, and clip ε are still not
stated anywhere in v1 — do not invent them.**

Monitoring signal worth stealing for *any* group-normalized RL run:
**per-batch reward standard deviation**. Their Theorem 3.2 shows the
population GRPO learning signal scales as √Var(r)/β² — reward variance *is*
the gradient's natural scale. Empirically, training on random turns let
batch reward-std collapse toward ~0.35 while the pivot-filtered set held
~0.50 across ~600 steps, tracking validation accuracy `[paper — figure-read
values]`. If reward-std is collapsing, the data has stopped teaching:
re-filter for mixed-outcome prompts. (This, MGPO's p₀=0.5 weighting (§5),
DAPO group filtering, and Agents-A1's 5-retry mixed-outcome filter (§13)
are the same lever in four guises.)

When to prefer which (agent post-training):
- Own good trajectories + tight compute → PivotRL-style local RL.
- Verifiable end-to-end environments + budget → E2E async RL (§11).
- Both → GLM-5's ordering: SFT (error-masked) → reasoning RL → agentic RL,
  with PivotRL-style turn-level training as the cheap middle step
  `[heuristic — no published run combines all three yet]`.

## 13. Multi-teacher on-policy distillation: unifying domain specialists (Agents-A1, MiniCPM5)

Agents-A1 (Shanghai AI Lab `[paper 2606.30616]`; 35B-A3B MoE initialized
from Qwen3.5-35B-A3B) is the published recipe for the "several specialists
→ one model" problem when the specialists are *agents* spanning
heterogeneous domains (deep research, MLE/coding, science, instruction
following, long-context, tool use). Three stages:

1. **Full-domain SFT** — lr 1e-5, cosine + warmup ratio 0.05, batch 16
   seqs, 1 epoch, max seq **131,072**, AdamW wd 0.1, loss on response
   tokens only, sample packing; ~100K trajectories averaging **45K tokens**
   (per-domain 3K–48K) `[paper — Tables 2–3]`. Documented failure mode:
   this mixed-domain SFT *regressed* IFBench 70.2 → 68.7 and
   HLE-with-tools 47.4 → 41.6 vs the base — attributed to conflicting
   reasoning patterns (single-turn long-thinking vs multi-turn tool-use
   short-thinking) `[paper — Table 4]`. If your multi-domain agent SFT
   shows the same signature, this is the precedent.
2. **Domain-level teachers** — branch per-domain SFT or GRPO-RL runs off
   the SFT checkpoint. The search teacher's RL config is fully published
   `[paper §4.2.1]`: GRPO, **constant lr 1e-6**, 32 prompts × **G=8**
   rollouts = batch 256, **asymmetric clip [0.2, 0.28]** (the same
   clip-higher as GLM-5/DAPO, §5), **KL 0.001 kept** (unlike GLM-5's
   no-KL), entropy coef 1e-4, max response 4,096, rollout temp 1.0, ≤300
   tool calls/rollout, rollout logprobs used for advantage (§11's
   consistency family). RL data: ~2,000 multi-hop questions kept only if 5
   attempts yield BOTH correct and incorrect trajectories — the
   mixed-outcome filter again (§12). The tool-use teacher adds
   **asymmetric process rewards** (PAPO-style): Â = Â_out +
   λ·1[r_out=0]·Â_proc with **λ=0.5**, rubric-based process advantage
   applied to *failed* trajectories only (on successes it double-counts the
   outcome signal) `[paper — Eq. 7–8]`; its hard-task RL set is just **64
   near-miss samples**, deliberately reused across steps. The IF teacher
   runs two consecutive GRPO stages with dynamic sampling (uniform-reward
   groups dropped).
3. **Multi-teacher on-policy distillation (OPD)** — the student generates
   rollouts (a *frozen* rollout copy generates; a separate optimized copy
   receives gradients); each sample is supervised by *its domain's teacher
   only* — hard routing, never a soft teacher mixture, never parameter
   merging. Loss = **truncated reverse KL on the teacher's top-k support**,
   both distributions renormalized on that support ("salient vocabulary
   alignment"), tool-output/user tokens masked, losses normalized within
   each domain before averaging across domains so no domain dominates.
   Diagnostic they monitor: student probability mass on the teacher's top-k
   set. Rollouts hitting turn/length/context budgets are marked TRUNCATED
   and kept as valid prefixes; system-crashed ones are retried (cf. §11's
   env-noise hygiene). Result: the SFT-stage regressions repair (IFBench
   80.6, HLE 47.6) and one student roughly matches its teachers — though
   not always exceeding them (IFBench 80.6 vs teacher's 82.0) `[paper]`.

Position it among the unification tools: §8 merges *weights* (needs a
shared ancestor; can silently trade capabilities), §10 distills *offline*
from verified trajectories, GLM-5 (§11) distills on-policy from its own
*earlier stages*; Agents-A1 distills on-policy from *parallel domain
teachers* with hard routing; SEED (§17) removes the separate teacher
entirely — the policy supervises itself from a hindsight-skill-augmented
context. Caveats before citing: the OPD stage's
LR/batch/top-k are **not stated**; and the paper is internally inconsistent
in places (the search teacher's GAIA gain appears as both 85.4 and 95.1;
the reproduced τ²-bench baseline as 32.53/33.0) — quote Table values and
name the table.

**MiniCPM5-1B: the same pattern at 1B, with published gains** `[paper —
OpenBMB/MiniCPM README + official figures; no dedicated arXiv report
exists]`. The full post-training pipeline of a 1B on-device model
(released 2026-05): 200B tokens deep-thinking SFT + 200B hybrid-thinking
SFT → **parallel RL teachers** branched off the SFT checkpoint —
Reasoning RL stage 1 (repetition penalty) → stage 2 (reasoning accuracy)
plus RLHF (human preference), IF RL, General RL, and Long-Context RL —
→ **on-policy distillation of all teachers back into one student**.
Deviations from Agents-A1 worth knowing:

- The teachers are not domain-specialist *agents* but capability-specialist
  RL runs of the same 1B; OPD still repairs the unification problem.
- **OPD is run inside the RL framework**: reverse KL to the teacher is used
  *as the advantage estimate*, replacing the verifier-based advantage —
  operationally GLM-5's §11 trick (Â = teacher/student log-prob gap)
  adopted wholesale. Implementation follows the TML blog + Rethinking-OPD
  (§15), with one variant choice made explicit: top-k logits are taken from
  **both** student and teacher and the reverse KL is computed on the
  **union** of the two token sets (Rethinking-OPD's default analyzes
  student-top-k; union is an implemented option in its repo) `[paper —
  README verbatim]`.
- **Prompt reuse**: OPD reuses each RL teacher's in-domain training prompts
  as the distillation data — "no additional data curation is required"
  `[paper]`. The cheapest published answer to "what do I distill on".
- Reasoning RL data/recipe: DAPO-Math-17k prompts, "inspired by JustRL's
  minimalist recipe" (arXiv 2512.16649), plus TriviaQA / NQ-Open /
  LongWriter-Zero-RLData / synthesized verifiable RLVR / pairwise RLHF
  signals `[paper]`. Length control is handled in-RL — see §5 guardrails.

Published effect of RL+OPD over the SFT checkpoint `[paper — official
figures]`: average +16 points with overlong responses (hitting the
max-token budget) down 29 pp; per-benchmark AIME-2025 17.3→40.4,
AIME-2026 15.6→40.4, HMMT-Feb-26 11.9→25.8, LCB-v6 20.6→33.5, IFEval
68.2→80.4, IFBench 33.7→46.7, Multi-IF 28.5→43.5. Final model averages
42.57 vs Qwen3-0.6B 26.77 / Qwen3.5-0.8B 25.14 / LFM2.5-1.2B 35.61
`[paper — eval table]`. Not stated anywhere: OPD top-k value, the RL
algorithm base and KL/clip coefficients, all LRs/batch sizes, teacher
checkpoint selection — **do not invent them**.

For *why* these OPD setups work — and what to check before copying one
with a different teacher — see §15: both recipes' teachers (Agents-A1's
domain branches, MiniCPM5's capability-RL runs) are branched from the
student's own SFT checkpoint (thinking-pattern match by construction) and
carry RL-gained capability the student lacks, i.e. both Rethinking-OPD
preconditions hold by design.

## 14. Synthetic-data curation by feature coverage (FAC / SAE)

"Less is Enough" (FAC Synthesis, ICML 2026 `[paper 2602.10388 +
Zhongzhi660/FAC-Synthesis repo]`) supplies the measurement layer for the
"data beats hyperparameters" principle: **the diversity that matters is
coverage in the *model's own feature space*, not surface variety.**

- **The metric (FAC).** Train a Top-K SAE on mid-depth activations (~50%
  layer: layer 16 of LLaMA-3.1-8B / Mistral-7B, 14 of Qwen2-7B), have an
  annotator LLM label which features are task-relevant, and score a
  dataset by which of those features it activates. FAC tracks downstream
  results at Pearson 0.95 (toxicity) / 0.85 (reward modeling) / 0.72–0.88
  (steering, instruction following), while the classic diversity metrics
  **anti-correlate**: Distinct-1/2 r ≈ −0.58/−0.33, POS-tag Distinct-2
  −0.66; bigram entropy (+0.57) is the only decent generic signal `[paper —
  Table 8, App. L.8]`. If you gate synthetic data on Distinct-n today, you
  may be selecting for *worse* data.
- **The synthesis loop.** Identify task-relevant features *missing* from
  the seed set vs an anchor corpus → per missing feature, build a
  contrastive strong/weak activation pair, use it as a few-shot demo, and
  SAE-filter the candidates (two-step beats one-step direct prompting at
  every threshold: 81.8% vs 69.6% FAC at δ=0) → **one sample per missing
  feature is the efficient point** — data-efficiency falls monotonically
  from 2–5 samples/feature, and with coverage fixed, padding the set to
  200 samples adds only ~+0.5% `[paper — §6, §7.3, Fig. 4/7]`. Coverage,
  not count, drives the gains.
- **Results at very small budgets**: 200 samples → +23.63 AUPRC toxicity;
  300 → +13.32 reward-model accuracy; 2,000 → AlpacaEval 2.0 parity with
  MAGPIE's 300K samples (**150× less data**) `[paper — Table 1, Fig. 1]`.
  Known limit: barely moves GSM8K / LiveCodeBench (+0.0–1.5) — single-layer
  SAE features under-capture multi-step reasoning; don't buy this for
  reasoning data.
- **Generator choice**: the *backbone-aligned* generator
  (LLaMA-3.1-8B-Instruct, temp 0.8, top-p 0.9) beat GPT-4o-mini as data
  generator **at every temperature** — alignment with the student's
  feature space matters more than raw generator strength `[paper — Table
  4]`. Features also transfer across families, weak-to-strong: LLaMA-derived
  features improved Qwen2 by +1.6–5.1 more than Qwen2's own `[paper —
  Table 3]`.
- **SAE cost is one-time and small**: 2^16 dictionary, K=20, ~113M tokens,
  3 epochs, AdamW 1e-3, batch 512 ≈ **4.4 h on 4×H100**, reusable across
  tasks `[paper — App. J/L.3]`. Honesty caveats from the repo audit: the
  released script differs from the paper text (plain Adam not AdamW, top-K
  annealed 200→20, default 5 epochs) and the toxicity fine-tune script is
  not runnable as published — treat repo numbers as `[config]` *with that
  flag*, and the paper's Table values as the citable ones.

## 15. When on-policy distillation works: preconditions, mechanism, recovery (Rethinking OPD)

"Rethinking On-Policy Distillation" (THUNLP `[paper 2604.13016 +
thunlp/OPD repo, verl 0.7.0]`) is the controlled study of *single-teacher*
OPD — the science underneath GLM-5's cross-stage distillation (§11) and
Agents-A1's multi-teacher stage (§13). Setup: math reasoning
(DAPO-Math-17K prompts; AIME24/25, AMC23 avg@16), students
R1-Distill-1.5B and Qwen3-1.7B(-Base), teachers JustRL-1.5B,
R1-Distill-7B, and Qwen3-4B variants. The loss is per-token **reverse KL
from the teacher at student-visited states**, truncated to the student's
top-k support and renormalized on it (Student Top-K) `[paper §2.2]`.

**Two preconditions — check both before spending the compute:**

1. **Thinking-pattern consistency.** A GRPO-trained Qwen3-4B teacher with
   a *lower* benchmark score outproduced a stronger non-thinking Qwen3-4B:
   it started at higher top-k overlap with the student (~72%) and the
   mismatched teacher's deficit was never recovered later in training
   `[paper — figure-read]`. Teacher benchmark score is the wrong selection
   criterion; pattern compatibility at student-visited states is the right
   one.
2. **Genuinely new capability.** Same-pipeline teachers fail even when
   bigger: distilling R1-Distill-1.5B toward R1-Distill-7B stagnated on
   every metric, and reverse-distillation showed why — a same-family 7B
   teacher and a 1.5B pre-RL checkpoint induced *nearly indistinguishable*
   target distributions on the student's own trajectories (JustRL-1.5B
   pushed toward its pre-RL checkpoint regressed almost exactly to pre-RL
   performance) `[paper]`. What works: a teacher that gained capability
   the student's pipeline never saw — e.g. via its own RL.

Global reward quality does NOT substitute for these: the *failing* 7B
teacher separated correct from incorrect rollouts as well as the
succeeding one (AUROC 0.75 vs 0.73) `[paper — Fig. 14]`. A teacher can be
a fine judge and a useless token-level supervisor.

**Mechanism → the monitoring signals.** Successful OPD is progressive
alignment on a small shared high-probability token set: the student–teacher
top-k **overlap ratio** rises (~72% → ~91%), the overlap tokens carry
**97–99% of both models' probability mass** throughout, and the entropy
gap |H(teacher) − H(student)| narrows toward zero. In failing runs all
three curves are **flat from the outset** `[paper — Fig. 6, figure-read]`.
Optimizing *only* the overlap tokens matches full Student Top-K; the
non-overlap tokens contribute little `[paper — Fig. 7]`. So: log overlap
ratio + entropy gap from step 0 (same diagnostic family as Agents-A1's
"student mass on teacher top-k", §13). Flat curves in the first hundreds
of steps → stop the run; a precondition is violated — more steps will not
fix it.

**Recovery recipes when a precondition fails:**

- **Off-policy cold start** — SFT the student on teacher-generated
  responses first (they used 200K teacher responses on an OpenThoughts3
  math subset), then run OPD on held-out prompts (~30K after dedup).
  Overlap starts higher and stays smooth; the cold-started student
  substantially beats pure OPD `[paper — Fig. 8, figure-read]`. This is
  the published rationale for the SFT-before-OPD ordering that GLM-5 and
  Agents-A1 both use implicitly.
- **Teacher-aligned prompts and template** — switching the student to the
  teacher's answer template alone improved all three benchmarks; drawing
  prompts from the teacher's own post-training distribution helps too but
  **collapses student entropy** — mix teacher-aligned prompts with prompts
  outside the teacher's training data `[paper — Fig. 9–10, figure-read]`.

**Knob guidance** (default config `[paper — Table 2]`, repo-confirmed
values marked): lr 1e-6, global batch 64, rollouts/prompt 4 `[config]`,
Student Top-K k=16 `[config]`, max prompt 1,024 / max response 7,168
`[config]`, temp 1.0, 1 epoch, 8×A800-80G; eval avg@16 at temp 0.7,
top-p 0.95, val max len 31,744.

- **k is insensitive above ~4**: top-1 consistently underperforms; top-4
  through top-16 are similar; even sampled-token-only supervision is
  comparable `[paper — Fig. 15]`. Don't sweep k; spend the effort on
  teacher choice.
- **Response-length sweet spot 3K–7K** for this regime: 0.5–1K gives too
  little supervision, 10–15K plateaus or declines `[paper — Fig. 11a]`.
- **The teacher's per-token advantage decays with prefix length**: +0.37
  at 1K-token prefixes → +0.02 at 16K `[paper — Fig. 11b]`. Dense
  token-level supervision fades on long sequences — an open caveat for
  long-horizon agent OPD (Agents-A1 distills at 131K seq; whether the
  signal survives there is untested `[heuristic]`).

Honesty caveat: the paper reports results almost entirely in figures — no
consolidated numeric results table — so benchmark deltas above are
`[paper — figure-read]`; cite the mechanism numbers (97–99% mass,
72→91% overlap) and the config table, which are stated in text.

## 16. Reasoning-SFT data curation: question quality, answer multiplicity, base-model fit (OpenThoughts / GRAPE)

Two sources turn "data beats hyperparameters" into concrete SFT-dataset
rules. OpenThoughts (arXiv 2506.04178, ICLR 2026) is the largest published
controlled ablation of reasoning-SFT data curation — 1000+ runs at 31.6K
datapoints each, culminating in OpenThoughts3-1.2M and OpenThinker3-7B.
GRAPE (arXiv 2502.04194, NeurIPS 2025) supplies the response-*selection*
rule when several candidate answers exist per instruction.

**OpenThoughts' five findings** `[paper]` (each from controlled ablations):

1. **Sample many answers per question from the teacher** — "an effective
   technique to increase the size of a data source by at least 16×"; the
   final recipe annotates every question 16× with QwQ-32B. Scaling answers
   per question ≈ free data, and it is the SFT-data mirror of SSP's
   diversity goal (§9).
2. **Better benchmark scores ≠ better teacher.** QwQ-32B-taught students
   beat DeepSeek-R1-taught students by +2.6 (math) / +1.9 (code) even
   though R1 outscores QwQ-32B by 8–23 points on CodeElo / GPQA-Diamond /
   JEEBench.
   Convergent with GRAPE below and FAC's backbone-aligned-generator result
   (§14): *fit* beats generator strength.
3. **Answer verification/filtering did not help**: "the no-filtering
   strategy … led to performance similar to that of all other methods of
   filtering" (including LLM verification and majority consensus); the
   shipped pipeline applies **no answer filtering**. Don't spend teacher
   compute verifying answers — spend it sampling more of them (finding 1).
4. **Fewer, better question sources.** Selecting questions from the top
   1–2 sources per domain beat mixing the top 4/8/16 — optimizing source
   *diversity* actively hurt.
5. **LLM-based question filters beat classical ones**: LLM-labeled
   difficulty (used for code) and LLM response-length (math/science)
   outperformed embedding/fastText filters. Question difficulty filtering
   *does* pay — contrast with finding 3: filter *questions*, not answers.

Pipeline `[config — dataset card]`: filter → dedup → downsample to 75K
questions → ×16 QwQ-32B annotation → 1.2M (850K math / 250K code / 100K
science). Headline result `[paper — Table 1]`: **SFT-only** OpenThinker3-7B
averages 55.3 vs RL-trained AceReason-7B 52.9, Skywork-7B 51.6, and
SFT+RL Nemotron-Nano-8B 53.2 — at 7B, data curation alone beat RL
pipelines (not a controlled SFT-vs-RL comparison — different bases and
data — but the strongest open-data 7B was pure SFT; cf. §12/§15 for when
RL earns its keep). Hyperparameters: §2 table row and the dataset-size
ladder bullet. The 1.5B sibling used LR 1.6e-4 / batch 256 / 7 epochs
`[config]` — don't mix the recipes.

**GRAPE: pick the response that fits the base model** `[paper 2502.04194;
cite arXiv v3 for the metric]`. For each instruction with multiple
candidate responses (from existing datasets and/or several teachers), score
each candidate's **length-normalized conditional log-probability under the
base model you are about to fine-tune** (= lowest perplexity), keep the
best, then run completely standard SFT. Selection is forward-pass-only —
no gradients, no reward model.

- Controlled setting `[paper — Table 1]`, UltraInteract, size-matched:
  Llama3.1-8B avg 41.5 vs 36.8 (original data) / 38.8 (Llama3.1-405B-
  generated) / 31.7 (3× data); Qwen2.5-7B 56.7 vs 45.1 / 42.9 / 39.4.
  The 405B-generated baseline losing to base-model-fit selection is the
  cleanest published "stronger generator ≠ better data" evidence.
- Realistic setting `[paper — Table 2]`, pool = Tulu-3 + Olmo-2
  post-training data: GRAPE-350K @1 epoch averages 35.9 vs Tulu3-SFT-939K
  @2 epochs 32.4 (~1/6 the compute), and beats S2L/LESS/embedding
  selection baselines on every base tried.
- **Reverse-GRAPE control**: selecting the *highest*-perplexity responses
  hurts (30.8) — the effect is direction-specific, not a data-size
  artifact.
- Extends to reasoning traces `[paper — App. B]`: lowest-perplexity
  correct R1-distill trajectories beat highest-perplexity ones.
- Caveats: the metric is explicitly **length-normalized** (v1/v2 said only
  "conditional probability" — cite v3); do not attribute the S2L
  baseline's scheduler details to GRAPE; scoring used a plain
  "Question:/Answer:" template with no chat-template ablation.

**How they compose** `[heuristic]`: OpenThoughts curates *questions* and
says don't filter answers for a *single* teacher; GRAPE ranks *answers*
when you have several candidates per instruction; FAC (§14) measures
whether the resulting set covers the task's feature space. Hard questions
(finding 5, VibeThinker's ≥0.75-error filter, MGPO's p₀=0.5) + many/long
diverse answers (finding 1, §9) + base-model-likelihood selection (GRAPE)
is the assembled playbook for a reasoning-SFT dataset.

## 17. Self-evolving hindsight-skill distillation for agentic RL (SEED)

SEED (Tsinghua `[paper 2607.14777 v1, 2026-07-16 preprint + jinyangwu/Seed
repo, verl-based]`) attacks the core agentic-RL problem — sparse
trajectory-level rewards give no guidance on *intermediate* decisions —
without any external teacher: the policy analyzes its **own completed
trajectories** into natural-language "hindsight skills" and distills the
skill-induced probability shift back into itself as a dense token-level
signal. It is the fourth teacher construction in the OPD family: §11
distills from earlier-stage checkpoints, §13 from parallel specialist
teachers, §15 studies a single external teacher; SEED's teacher is **the
same policy given privileged hindsight context**. Both Rethinking-OPD
preconditions (§15) hold trivially — perfect pattern match (same model)
and genuinely new information (the skill summarizes the episode outcome
the acting policy could not see).

**Two stages** `[paper]`:

1. **Hindsight-skill SFT.** Collect 1,440 trajectories (180 tasks × 8
   rollouts), annotate each completed trajectory with an episode-level
   hindsight skill using an external analyzer (GLM-5.2; the repo also
   ships Gemini-annotated variants for some envs), and SFT 3 epochs — so
   one backbone can both act and analyze. Ablation: dropping this stage
   costs −5.8 on ALFWorld (91.8→86.0) `[paper — ablation table]`.
2. **Self-evolving OPD inside GRPO.** Each update, the frozen current
   policy π_old both samples rollouts *and* serves as the analyzer
   (repo: `analysis_backend=policy_vllm` `[config]`), extracting
   episode-level + critical-step skills from its own trajectories. The
   *same sampled action tokens* are re-scored under the ordinary context h
   and the skill-augmented context h̃; the shift Δ = sg[log π(a|h̃) −
   log π(a|h)] passes through a sigmoid gate g = σ(β_opd·Δ), which weights
   a distillation loss added to the RL loss: **L = L_rl + λ_opd·L_opd**.
   After the update the improved policy becomes the next analyzer —
   supervision evolves with the policy. Replacing this with a *static*
   offline skill bank costs −7.4 (84.4); removing the OPD term costs −4.8
   (87.0) — the self-evolution is worth more than either component alone
   `[paper — ablation table]`.

**Knobs** (repo `examples/seed_trainer/_common/*.sh` defaults `[config]`
unless marked): **λ_opd = 0.01, gate β_opd = 5.0** — neither value appears
in the paper text, so cite them as repo defaults; actor lr 1e-6 (the §5
GRPO convention), KL loss coef 0.01 (ALFWorld) / 0.001 (search),
`low_var_kl`, no KL-in-reward; group size G=8, train batch 16 (ALFWorld;
paper says 128 for search-QA `[paper]`), γ=0.95, invalid-action penalty
0.1; per-step max prompt 2,048 / response 512; analyzer context 16,384,
max completion 4,096, ≤5 step-skills per trajectory (ALFWorld) / ≤2
(search); rollout temp 1.0, val temp 0.4; ~150 policy updates `[paper]`
(script default 160 epochs — repo/paper divergence, cite the paper).

**Results** `[paper — Table 1]` (bases Qwen2.5-3B/7B-Instruct,
Qwen3-1.7B): ALFWorld macro-avg success 91.8 / 96.1 / 92.0 vs GRPO
46.1–81.2 (vanilla prompting 12.5–21.9); search-QA avg (NQ, TriviaQA,
PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) 45.7 / 48.6 / 42.2 vs GRPO
40.8–42.0; WebShop score 88.5 / 89.7 / 87.1 vs GRPO 79.8–80.9. Two
practitioner-relevant findings beyond the headline: **internalizing skills
beats prompting with them** — SEED trains with skills but deploys bare
(no skill bank, retrieval, or extra prompt at inference) and still beats
skill-prompted baselines (the repo's `run_*_skills.sh` scripts, which
inject a top-k=6 skill memory into the prompt, are that baseline); and
skills act as *privileged training-time supervision only*, so inference
cost is unchanged.

Scope caveats before citing: demonstrated at 1.7–7B on short-horizon envs
(ALFWorld ≤30–50 steps, search ≤4 turns, per-step responses ≤512 tokens) —
untested at Agents-A1's 131K-token trajectory scale, where §15's
prefix-length decay of dense token signals is the open question
`[heuristic]`; preprint, no peer-reviewed venue yet; and the paper reports
β_opd/λ_opd and several batch/epoch values only via the repo, whose
defaults do not everywhere match the paper's stated setup (e.g. the
skills-prompt baseline script runs G=4 on search vs the paper's N=8).
