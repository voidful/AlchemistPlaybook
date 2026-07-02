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

## Post-training

| Source | ID / link | Used for |
| --- | --- | --- |
| Zephyr | arXiv 2310.16944 + huggingface/alignment-handbook recipes/zephyr-7b-beta (config-verified SFT + DPO YAMLs) | SFT 2e-5/cosine/1ep; DPO 5e-7/β0.01/1ep, lengths 1024/512, non-reentrant grad ckpt |
| Tulu 3 | arXiv 2411.15124 + allenai HF model cards (config-verified) | SFT 5e-6→2e-6, linear, warmup 0.03, 2ep, sum-loss; length-norm DPO β5, 5e-7→2e-7; full RLVR PPO table (3e-7, KL 0.05, batch 224, 100K episodes, EOS −10) |
| DPO | arXiv 2305.18290 | β 0.1 convention, objective |
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
