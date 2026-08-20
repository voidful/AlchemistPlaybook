# Sources

Verification levels used across this skill:
- `[config]` — value read directly from an official config file, training
  script, or model card maintained by the releasing org.
- `[paper]` — value stated in the technical report / paper.
- `[reported]` — secondary source or community-standard value; flag when
  precision matters.

## Pretraining

| Source | ID / link | What this skill uses it for |
| --- | --- | --- |
| LLaMA | arXiv 2302.13971 | per-size LR (3e-4 / 1.5e-4), 4M batch, AdamW (0.9,0.95), wd 0.1, clip 1.0, warmup 2000, cosine→10%, overtraining rationale |
| Llama 2 | arXiv 2307.09288 | 2T tokens, 4K ctx, GQA 34/70B; SFT recipe (2e-5, 2 epochs, batch 64); RLHF structure |
| Llama 3 herd | arXiv 2407.21783 | 405B: 8e-5 peak, warmup 8K, cosine→8e-7/1.2M steps, batch ramp 4M→8M(252M tok)→16M(2.87T); data mix; 6-stage 128K ctx; final 40M-token anneal; MFU 38–43%; contamination analyses |
| OLMo | arXiv 2402.00838 | 1B/7B LRs (4e-4/3e-4), warmup 5000, linear→10%, in-loop eval cadence |
| OLMo 2 | arXiv 2501.00656 + allenai/OLMo configs/official-1124 (config-verified: lr 3.0e-4, wd 0.1, eps 1e-8, cosine alpha_f 0.1, t_warmup 8.39B tokens, batch 1024/2048 seqs) + HF model cards | stability stack (QK-norm, z-loss 1e-5, norm reorder, ε 1e-8, init 0.02), two-stage Dolmino annealing + model souping |
| OLMo 2 32B script | OLMo-core official script | lr 6e-4, 8.4M batch, z-loss 1e-5, HSDP, rank microbatch 16K |
| OLMo 3 | arXiv 2512.13961 + allenai.org/blog/olmo3 | 3-stage curriculum: ≤5.9T pretrain → 100B midtrain → 50/100B long-context; model-flow post-training (SFT→DPO→RLVR) |
| SmolLM2 | arXiv 2502.02737 (COLM 2025) | 1.7B: lr 5e-4, WSD warmup 2000 / final-10% decay, 2M batch, 11T tokens, staged mixtures |
| Pythia | arXiv 2304.01373 | controlled suite, 2M batch, checkpoint density for dynamics studies |
| DeepSeek-V3 | arXiv 2412.19437 | multi-phase LR (2K warmup→2.2e-4 const to 10T→cosine to 2.2e-5 over 4.3T→2.2e-5×333B→7.3e-6×167B), batch ramp 3072→15360/469B, FP8 with tile-wise scaling, MTP |
| DeepSeek LLM scaling | arXiv 2401.02954 | batch↑ / LR↓ trends with compute |
| MiniCPM | arXiv 2404.06395 | WSD origin, decay-phase loss drop, muP-style transfer (base lr 0.01) |
| Chinchilla | arXiv 2203.15556 | ~20 tokens/param compute-optimal |
| Data-constrained scaling | arXiv 2305.16264 | ~4 epochs of repeated data ≈ fresh data |
| FineWeb | arXiv 2406.17557 | dedup/filtering ablations, FineWeb-Edu classifier filtering |
| PaLM | arXiv 2204.02311 | spike SOP (rewind ~100 steps, skip 200–500 batches), z-loss 1e-4, batch ramp |
| Kimi K2 | arXiv 2507.20534 | MuonClip / QK-Clip τ=100, 15.5T tokens zero-spike claim |
| Moonlight (Muon scalable) | arXiv 2502.16982 | Muon at scale: weight decay required, update-RMS matching ~0.2, ≈52% FLOPs vs AdamW |
| muP / Tensor Programs V | arXiv 2203.03466 | width-transferable LR |
| Critical batch size | arXiv 1812.06162 | gradient noise scale; ramp justification |
| GLM-130B | arXiv 2210.02414 | embedding gradient shrink α=0.1 |
| Spike No More | arXiv 2312.16903 | embedding scale + small init for spike prevention |
| Small-scale proxies | arXiv 2309.14322 (Wortsman et al.) | QK-norm + z-loss validation; LR-sensitivity grows with scale; proxy sweeps |
| BLOOM | arXiv 2211.05100 | fp16 instability lesson → bf16 |
| Ultra-Scale Playbook | huggingface.co/spaces/nanotron/ultrascale-playbook | distributed training systematics (4000+ runs, up to 512 GPUs); FP8 < bf16 stability |
| LFM2 (Liquid AI) | arXiv 2511.23404 | edge/on-device hybrid (gated short-conv + GQA, hardware-in-the-loop arch search); 10–12T pretrain @4K ctx + 1T long-ctx midtrain @32K with accelerated LR decay; **Decoupled Top-K distillation** (teacher LFM1-7B, top-32, binary-mass + tempered-shape KL + hard-label CE); **difficulty-ordered curriculum** (12-model success-rate ensemble); ~10% input dropout on small models; SFT 3e-5→1e-7 / 3 epochs; **length-normalized direct alignment** β5, 8e-7→8e-8 (generalizes DPO + APO-zero); **parallel model-merging stage** (soup / task-arithmetic / TIES / DARE / DELLA) |
| GLM-5 | arXiv 2602.15763 | 744B/40B-active MoE (256 experts, 80 layers), 28.5T tokens (27T pre @4K + midtrain 32K/1T→128K/500B→200K/50B); MLA-256; **Muon Split** (per-head orthogonalization ⇒ stable attention logits, no clipping); MTP ×3 shared layers; **DSA sparse-attention retrofit** (indexer-only warmup 1000 steps lr 5e-3 → joint 20B tokens); INT4 QAT at SFT; efficient-attention ablation (SWA/GDN/SimpleGDN vs DSA) |
| MiniCPM5-1B | github.com/OpenBMB/MiniCPM README + huggingface.co/openbmb/MiniCPM5-1B (released 2026-05-19; **no dedicated arXiv report** — the repo's "Tech Report" link is the MiniCPM4 paper 2506.07900, do not cite it for MiniCPM5; data scheme: UltraData arXiv 2602.09003) | staged-decay pretrain curriculum (stable → short decay 4K/200B → long decay 32K/75B → 128K/25B → mid-train 200B); 200B deep-thinking + 200B hybrid-thinking SFT; **parallel capability-specialist RL teachers → OPD** (reverse-KL-as-advantage in the RL framework, top-k **union** of student+teacher token sets, teacher RL prompts reused as distillation data); reasoning RL on DAPO-Math-17k per JustRL minimalist recipe (2512.16649) with **two-stage max-length schedule 30,720→38,912** (stage-1 repetition penalty → stage-2 accuracy; reward shape not stated) cutting overlong responses 42.3→9.6% on AIME-2025; RL+OPD +16 avg over SFT (AIME-2025 17.3→40.4, IFEval 68.2→80.4); final avg 42.57 vs LFM2.5-1.2B 35.61; arch `[config]` 1.08B/0.68B non-emb, GQA 16/2, vocab 130,560, ctx 131,072; optimizer/LRs/OPD-k **not stated**; detailed numbers are official-figure reads (PNGs in repo) |
| EvoLM | arXiv 2506.16029 (NeurIPS 2025 oral) + zhentingqi/evolm repo (config-verified) | controlled full-pipeline ablation, 100+ models at 0.5/1/4B: pretrain lr 2.5e-4 / 2e-4 / 1.5e-4, cosine→10%, warmup ratio 0.1, batch 512–1024 × 2048, bf16-mixed, FineWeb-Edu; **downstream saturation ~80–160 tokens/param**, over-pretraining degrades post-trained OOD; CPT re-warms to full peak LR + non-monotonic replay (8B/50B optimal); SFT 1e-5 (1B) / 7.5e-6 (4B), batch 128, OOD peaks 2–4 ep; verl PPO actor 2e-6 / critic 2e-5 (10×), KL 1e-4, **warmup-constant** schedule; RL saturates 4–8 ep / 50–100K, sharpens not expands (Pass@16 ↓ past ~4 ep); RL dataset-scaling collapse at 350–400K via length explosion; SFT budget→ID vs RL budget→OOD; ORM avg-score > val loss as post-training validation signal |

## Post-training

| Source | ID / link | Used for |
| --- | --- | --- |
| Zephyr | arXiv 2310.16944 + huggingface/alignment-handbook recipes/zephyr-7b-beta (config-verified SFT + DPO YAMLs) | SFT 2e-5/cosine/1ep; DPO 5e-7/β0.01/1ep, lengths 1024/512, non-reentrant grad ckpt |
| Tulu 3 | arXiv 2411.15124 + allenai HF model cards (config-verified) | SFT 5e-6→2e-6, linear, warmup 0.03, 2ep, sum-loss; length-norm DPO β5, 5e-7→2e-7; full RLVR PPO table (3e-7, KL 0.05, batch 224, 100K episodes, EOS −10) |
| DPO | arXiv 2305.18290 | β 0.1 convention, objective |
| IPO | arXiv 2310.12036 (Azar et al.) | squared-loss bounded preference objective; robust to near-deterministic/noisy preferences; τ ≠ DPO β |
| RLAIF / Constitutional AI | arXiv 2309.00267 / 2212.08073 | AI-annotated preference labels as standard practice; judge-bias caveat |
| SimPO | arXiv 2405.14734 + princeton-nlp/SimPO README (config-verified tables) | per-setting β/γβ/LR; DPO baselines β0.01; tuning order; LR-damage evidence |
| ORPO | arXiv 2403.07691 + handbook zephyr-141b-A35b config (config-verified: β0.05, 5e-6, inverse_sqrt, 3ep) | single-stage alignment |
| KTO | arXiv 2402.01306 | unpaired preference |
| DeepSeekMath / GRPO | arXiv 2402.03300 | GRPO lr 1e-6, KL 0.04, G=64, len 1024, batch 1024 |
| DeepSeek-R1 | arXiv 2501.12948 | rule-based rewards, R1-Zero pipeline |
| InstructGPT | arXiv 2203.02155 | SFT-epochs vs RM-quality nuance |
| NEFTune | arXiv 2310.05914 | embedding-noise α 5/10/15 |
| LoRA | arXiv 2106.09685 | low-rank adaptation basics |
| QLoRA | arXiv 2305.14314 | NF4 + double-quant + paged optimizers; r64/α16; lr 2e-4 (7–13B) / 1e-4 (33–65B); all-linear targeting |
| LoRA learns less, forgets less | arXiv 2405.09673 | LoRA-vs-full-FT decision rule; higher-LR requirement |
| VibeThinker-1.5B | arXiv 2511.06221 (HTML live at /html/2511.06221v1; v2 → 404) | Spectrum-to-Signal Principle (SFT=Pass@K diversity / RL=Pass@1 signal, select SFT checkpoint by Pass@K); Two-Stage Diversity-Exploring Distillation SFT (N=4 subdomains, Pass@K specialist selection, uniform-average fusion w_i=1/N; base Qwen2.5-Math-1.5B); MGPO = GRPO + max-entropy advantage reweight (target p₀=0.5, w=exp(−λ·D_ME); λ/G/RL-LR/KL not stated); avg@k mean-Pass@1 eval (64/8/16 samples, temp 1.0 math / 0.6 code, 40k max); ~3,900 H800-hr / <$8K post-training @ $2/hr; scores AIME24 80.3, AIME25 74.4, HMMT25 50.4, MATH-500 95.0, LCB V6 51.1, GPQA-D 46.7 |
| VibeThinker-3B | arXiv 2606.16140 | Parametric Compression-Coverage Hypothesis (parameter-dense reasoning vs parameter-expansive knowledge) `[hypothesis]`; curriculum SFT (base Qwen2.5-Coder-3B, batch 128, LR 5e-5→8e-8 cosine, 5% warmup, 5+2 epochs, reference-model 8-rollout ≥0.75-error difficulty filter, ≥5K-token trace floor); single 64K-window multi-domain MGPO RLVR (math→code→STEM verifiers, no neural RM); Long2Short reward redistribution λ=0.2; offline self-distillation w/ learning-potential NLL trace selection; CLR test-time scaling (K=32, M=5, r=(mean v)^M); scores AIME26 94.3/97.1-CLR, LCB v6 80.2, LeetCode 96.1%, IFEval 93.4; RL hyperparameters + compute not stated |
| PivotRL (NVIDIA, Nemotron-3-Super) | arXiv 2603.21383 (v1; full PDF incl. appendices re-verified 2026-07) | turn-level local RL on SFT trajectories: pivot filtering (keep mixed-outcome, below-λ_diff turns; 71% of random turns give zero group-normalized gradient) + functional-equivalence verifier reward (exact match underperforms SFT: 57.34 vs 58.44 τ²-bench); GRPO-clipped objective + KL to frozen π₀; vs SFT same data (base Qwen3-30B-A3B-Thinking-2507): in-domain +14.11 vs +9.94, OOD **+0.21 vs −9.83** (Table 2 average — the −9.48 in the intro is a paper-internal inconsistency); ≈E2E-RL on SWE-Bench (32.67%) at ~4× fewer rollout turns / ~5.5× less wall-clock — PivotRL batch 1024 = 64 prompts × **G=16** @1 turn/sample (~133K turns) vs E2E 16 × G=32 @12–25 turns/traj (~542K); reward-variance diagnostic (Thm 3.2: GRPO signal ∝ √Var(r)/β²); in-domain numbers are best-checkpoint, not final-step; K/λ_diff/LR/β/clip-ε still unstated in v1 |
| Agents-A1 (Shanghai AI Lab) | arXiv 2606.30616 + InternScience/Agents-A1 README | 35B-A3B agentic MoE from Qwen3.5-35B-A3B; 3-stage: full-domain SFT (1e-5 cosine, warmup 0.05, batch 16, 1 ep, seq 131,072, wd 0.1; ~100K trajectories avg 45K tokens) → domain teachers (search GRPO: constant 1e-6, 32 prompts × G=8 = batch 256, asym clip 0.2/0.28, KL 0.001, entropy 1e-4; 5-retry mixed-outcome difficulty filter; PAPO-style negative-only process reward λ=0.5; 64-sample hard-set RL) → **multi-teacher domain-routed on-policy distillation** (truncated reverse-KL on teacher top-k, hard routing, domain-normalized loss); documented multi-domain SFT regression (IFBench 70.2→68.7, HLE 47.4→41.6) repaired by OPD (80.6 / 47.6); OPD-stage LR/batch/top-k **not stated**; paper internally inconsistent in places (GAIA teacher gain 85.4 vs 95.1; reproduced τ²-bench baseline 32.53/33.0) — cite Table values |
| Program-as-Weights (PAW) | arXiv 2607.02512 | text-to-LoRA hypernetwork with rare controlled PEFT evidence: 4B compiler (fully unfrozen) emits r=64 all-linear LoRA for a frozen 0.6B interpreter; lr 2e-5, AdamW defaults, no warmup/schedule, eff. batch 48, 3 ep, bf16 + fp32 mapper head; same-data baselines: generated-LoRA 73.8 > full FT 58.4 > fixed LoRA r=64 52.1 ≥ r=128 51.6 (rank saturation); simplest mapper beats all expressive variants; Q6_K base + Q4_0 LoRA ≈ bf16; two-model-agreement gate + generator-ceiling reporting for synthetic eval sets |
| FAC Synthesis ("Less is Enough") | arXiv 2602.10388 (ICML 2026) + Zhongzhi660/FAC-Synthesis | SAE feature-coverage (FAC) as data-diversity metric — Pearson ≈0.95 vs downstream while Distinct-n **anti-correlates** (−0.33…−0.66); coverage-guided two-step contrastive synthesis, 1 sample/missing feature; 200 samples +23.63 AUPRC, 2K ≈ MAGPIE 300K on AlpacaEval 2.0 (150×); backbone-aligned generator > GPT-4o-mini at every temp (peak 0.8); weak-to-strong cross-family feature transfer; **LoRA 62.6 vs full-FT collapse 29.0 @~200 samples**; Top-K SAE 2^16 / K=20 @50% depth, ~4.4 h on 4×H100 one-time; AUPRC + A/B-swap robust-accuracy eval discipline; caveats: repo diverges from paper text (Adam vs AdamW, top-K anneal 200→20, 5 vs 3 epochs) and toxicity script unrunnable as released — cite paper Tables |
| Rethinking OPD (THUNLP) | arXiv 2604.13016 + thunlp/OPD repo (verl 0.7.0; config-verified: rollout n=4, student top-k 16, max response 7,168) | when single-teacher on-policy distillation works: **two preconditions** (thinking-pattern consistency at student-visited states + teacher capability the student's pipeline lacks); reverse-distillation evidence that same-pipeline teachers — even 7B vs 1.5B — induce near-identical target distributions on student trajectories; mechanism metrics (top-k overlap 72%→91% in success vs flat in failure, overlap tokens carry 97–99% of both models' prob mass, entropy gap narrows; overlap-only loss ≈ full Student Top-K); recovery recipes: off-policy cold-start SFT (200K teacher responses) + teacher-aligned template/prompts (mix to avoid student-entropy collapse); default OPD config lr 1e-6, batch 64, temp 1.0, 1 ep, 8×A800 `[paper — Table 2]`; response-length sweet spot 3K–7K; teacher token-advantage decay +0.37 @1K → +0.02 @16K prefix; top-1 < top-4 ≈ top-16 ≈ sampled-token; failing teacher still reward-correlated (AUROC 0.75 vs 0.73) — global reward ≠ usable token signal; caveat: benchmark results are figure-only, no consolidated numeric table |
| OpenThoughts / OpenThinker3 | arXiv 2506.04178 (ICLR 2026) + open-thoughts HF model/dataset cards (config-verified) | largest controlled reasoning-SFT data ablation (1000+ runs @31.6K samples): ×16 answers/question from teacher QwQ-32B; **teacher benchmark score ≠ teacher quality** (QwQ-32B-taught beats R1-taught +2.6 math/+1.9 code); **answer verification/filtering gave no gains** (ship unverified); top 1–2 question sources beat top 4/8/16; LLM difficulty/length question filters beat embedding/fastText; pipeline 75K questions ×16 → 1.2M (850K math/250K code/100K science); SFT-only 7B avg 55.3 beats RL-trained AceReason 52.9 / Skywork 51.6; **dataset-size hyperparameter ladder** (App. D Table 9: 1e-5/32/13ep → 8e-5/512/5ep); OpenThinker3-7B config `[config]` 8e-5, 5 ep, batch 512, cosine, warmup 0.1, wd 0, β₂ 0.999; seq len **not stated** (32,768 is eval max_new_tokens); 1.5B sibling differs (1.6e-4/256/7ep) |
| GRAPE | arXiv 2502.04194 (NeurIPS 2025; **cite v3** — v1/v2 omit the length normalization) | base-model-fit response selection: among candidate responses per instruction, keep the highest **length-normalized** base-model log-prob (lowest perplexity), then plain SFT (1 ep, 1e-5, batch 256 `[paper — App. D]`); controlled Table 1 (size-matched): Llama3.1-8B 41.5 vs 36.8 original / 38.8 405B-generated / 31.7 3×-data, Qwen2.5-7B 56.7 vs 45.1/42.9/39.4 — **stronger generator ≠ better data**; realistic Table 2 (Tulu-3+Olmo-2 pool): 35.9 @350K/1ep vs Tulu3-SFT 32.4 @939K/2ep (~1/6 compute), beats S2L/LESS/embedding baselines; Reverse-GRAPE (highest ppl) hurts (30.8); extends to reasoning-trace selection (App. B); caveat: S2L baseline's scheduler details are not GRAPE's |
| On-Policy Distillation (Thinking Machines Lab) | thinkingmachines.ai/blog/on-policy-distillation (Oct 2025) | OPD lineage cited by MiniCPM5: student samples, teacher per-token log-probs give a dense reverse-KL reward; ~7–10× fewer gradient steps than RL, ~9–30× conservative compute advantage; personalization example (midtrain on internal docs regresses IF 85→79, OPD from the pre-midtrain model restores 83 while keeping the knowledge) — anti-forgetting via self-teacher; `[reported]` blog — LR/optimizer/token budgets not stated; AIME'24 GPU-hour table quotes the Qwen3 report, not TML measurements |
| GLM-5 (post-training) | arXiv 2602.15763 | SFT @202,752 ctx w/ **error-masked agent trajectories** + 3 thinking modes; Reasoning RL = GRPO + IcePop mismatch mask ρ∈[1/β,β] β=2, **no KL**, asym clip 0.2/0.28, on-policy G=32 batch 32, mixed 4-domain RLVR; **deterministic top-k / frozen indexer** for DSA-RL stability; async agentic RL (slime): TITO gateway, double-sided IS masking [1−ε_l,1+ε_h] on rollout logprobs, staleness cut, env-crash sample dropping + group pad/drop, optimizer reset on weight sync, DP-aware routing, FP8 rollouts + MTP + PD disaggregation; **on-policy cross-stage distillation** final stage (Â = sg[log π_teacher/π_student], G=1, batch 1024); hybrid rule/ORM/GRM rewards + human style anchors |

## Speech

| Source | ID / link | Used for |
| --- | --- | --- |
| Whisper | arXiv 2212.04356 App. F | per-size LRs (1.5e-3→1.75e-4), β₂0.98, ε1e-6, warmup 2048, linear→0, batch 256 seg, 2^20 updates |
| OWSM v3.1 | arXiv 2401.16658 + ESPnet released config | piecewise warmup 60K, peak 2e-4 (medium), batch 256, CTC 0.3, FlashAttention |
| OWSM v4 | arXiv 2506.00338 (Interspeech 2025) + espnet/yodas_owsmv4 | CTC-segmentation realign → LID filter → CTC-score filter; 370K→166K h; data-first lesson |
| wav2vec 2.0 | arXiv 2006.11477 | SSL pretrain + tri-stage fine-tune |
| HuBERT | arXiv 2106.07447 | iterative cluster targets |
| ESPnet | github.com/espnet/espnet | working defaults (CTC 0.3, label smoothing 0.1, warmup 25–40K, batch_bins) |

## Systems

ZeRO (arXiv 1910.02054), FSDP (arXiv 2304.11277), FlashAttention
(arXiv 2205.14135), activation checkpointing (arXiv 1604.06174):
cited for memory/throughput moves in diagnostics and templates.

## Evaluation & benchmarks

The three-tier framework (health monitoring / capability regression / release
gate), the mid-training retention-suite requirement, and the minimal eval
suites in `references/evaluation.md` are a **practitioner synthesis**
`[heuristic]` consistent with the per-stage eval discipline of OLMo (in-loop
eval cadence), Llama 3 (per-benchmark contamination analysis, private evals),
and Tulu 3 (per-stage delta tables) already cited above. Benchmark facts
stated in that file are properties of the named benchmark:

| Benchmark | ID / link | Note this skill relies on |
| --- | --- | --- |
| MMLU / MMLU-Pro | arXiv 2009.03300 / 2406.01574 | MMLU-Pro: 10 options (vs 4), harder reasoning, better strong-model discrimination |
| IFEval | arXiv 2311.07911 | ~500 prompts, ~25 auto-verifiable instruction types; strict/loose accuracy |
| RULER | arXiv 2404.06654 | configurable length + multi-needle / multi-hop / aggregation; > vanilla NIAH |
| SimpleQA | arXiv 2411.04368 (OpenAI) | short fact-seeking factuality; correct/incorrect/not-attempted |
| HarmBench | arXiv 2402.04249 | standard safety red-team + robust-refusal eval |
| GPQA | arXiv 2311.12022 | graduate-level science reasoning |
| BFCL | gorilla.cs.berkeley.edu/leaderboard | function/tool-calling eval |
| τ-bench | arXiv 2406.12045 | tool-agent task success in dynamic dialogue |
| SWE-bench (Verified) | arXiv 2310.06770 + openai SWE-bench-Verified | repo-level resolved rate |
| LiveCodeBench | arXiv 2403.07974 | contamination-resistant code (time-windowed) |
| AlpacaEval 2 / Arena-Hard / MT-Bench | arXiv 2404.04475 / lmarena Arena-Hard / arXiv 2306.05685 | LLM-judge dialogue win-rate / Elo (relative, judge-dependent) |
| Other named sets | C-Eval, CMMLU, AGIEval, BBH, GSM8K, MATH, HellaSwag, ARC, Winogrande, TruthfulQA, LongBench, FLORES-200, XNLI, WMDP, JailbreakBench, WebArena, GAIA, BFCL | standard benchmarks; cite by their canonical paper when precision matters |

## Workflow & tracking

| Source | ID / link | Used for |
| --- | --- | --- |
| "Using Claude Code: the unreasonable effectiveness of HTML" (Thariq Shihipar, Anthropic) | claude.com/blog/using-claude-code-the-unreasonable-effectiveness-of-html (2026-05) | `experiment-tracking.md`: HTML-over-Markdown artifact discipline — self-contained HTML for plans/reports, side-by-side comparison grids, artifact webs read by later sessions/verification agents. `[reported]` workflow practice, **not training evidence** — never cite it for a hyperparameter. |
| FSQ-campaign dashboard (QwenVocabCodec) | eric-lam.com/vocodec/campaign (live page, retrieved 2026-07-11) | `experiment-tracking.md` §4 report-detail standard: the reference exemplar for dashboard depth — stat tiles with bounds/ETA, per-chart interpretation captions, hover-free eval table, gated experiment ladder, honest decision ledger ("NOT a gated comparison"), Slurm job chain, provenance footer. §4.8 (inference-vs-GT sample gallery) extends past the exemplar's noted gap: it shows no output-vs-ground-truth samples. `[reported]` presentation exemplar, **not training evidence**. |

The git-commit-per-change and GitHub Pages mechanics in that file are
standard git/GitHub product behavior; the Pages-is-public-even-for-private-
repos caveat is from GitHub's own documentation. The commit-point and
artifact-set conventions are `[heuristic]` extensions of principle 6.

## Reviewed and excluded (no training evidence)

`itayinbarr/little-coder` and `Doorman11991/smallcode` (community repos,
2026): inference-time coding-agent harnesses for small local models. Both
were reviewed in full — no training runs, fine-tuning, or configs exist in
either repo, so nothing here meets the skill's evidence bar. Kept only as
`[reported]` triage context in diagnostics.md: harness quality can dominate
small-model agent performance (little-coder's self-published benchmark: the
same quantized Qwen3.5 9.7B scores 19.11% → 46.22% on Aider Polyglot from
the harness change alone). Do not cite these as training evidence.

Anthropic "Natural Language Autoencoders" (anthropic.com/research/
natural-language-autoencoders + transformer-circuits.pub/2026/nla,
2026-05-07): interpretability method (activation verbalizer/reconstructor
trained with RL to round-trip activations through natural language) —
reviewed in full, contains **no LLM-training-recipe evidence**. Relevant
only as background for the SAE-feature diversity metric in
post-training.md §14; do not cite it for any training value.
