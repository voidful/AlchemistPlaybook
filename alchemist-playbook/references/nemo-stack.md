# Training on the NVIDIA NeMo stack: Nemotron recipes + NeMo Gym

The default implementation route for this skill (repo decision 0007): when
the user has not fixed a trainer, or asks how to implement, launch, or
evaluate a recipe, map it onto **NVIDIA-NeMo/Nemotron** (stage recipes and
the `nemotron steps` catalog over Megatron-Bridge, NeMo AutoModel, and
NeMo-RL) and **NVIDIA-NeMo/Gym** (environments, verifiers, agent harnesses,
rollout collection, evaluation). Recipe *values* keep their homes in the
other references; this file says where each value goes and what runs it.

Verification tags: `[config]` = read from a YAML / TOML / step manifest in
the repositories at the pinned commits below; `[reported]` = stated in the
repositories' READMEs, guides, FAQ, or guidebooks (repo-internal ablations
and team answers, not papers); `[heuristic]` = this skill's synthesis.
Pinned reads (2026-10-27): NVIDIA-NeMo/Nemotron `1c3e781` (2026-10-21,
package `nemotron` 0.2.0, Python ≥ 3.11); NVIDIA-NeMo/Gym `11c874b`
(2026-10-26; release v0.7.0 on 2026-10-06; Python ≥ 3.13.14). Both move
fast: re-read the live checkout (`uv run nemotron steps show <id>`,
`gym env resolve --config …`) before quoting a path, flag, or default, and
say which commit a value came from.

## Contents
1. When this route applies
2. Stage → entry-point map
3. Pretraining and continued pretraining
4. SFT and PEFT
5. RL: GRPO/RLVR, RLHF, DPO on NeMo-RL with NeMo Gym environments
6. Environments, rollouts, and data formats (NeMo Gym)
7. Evaluation with NeMo Gym
8. Hardware, execution, and containers
9. Config key map: skill knob → stack key
10. Pre-launch verification

## 1. When this route applies

- **Default.** No trainer named, or the user asks "how do I run / launch /
  implement / evaluate this", or names a Nemotron model, NeMo-RL, NeMo
  Gym, or Megatron-Bridge → answer on this stack and map the recipe values
  from the other references into its configs (§9).
- **Precedence.** A user's existing trainer, config, or environment wins
  (SKILL.md task handling; decision 0006). Then name the NeMo equivalent
  only where it adds a capability the current stack lacks — Gym
  environments with exact token-ID capture through external harnesses
  (§6), multi-environment RLVR with verified resource servers, or the
  pinned evaluation recipes (§7) — and show the migration path. Gym also
  integrates with **verl** (pinned commit, `verlai/verl:vllm017.latest`)
  and **Unsloth** (2026.1.4) `[reported — gym docs]`, so a verl or Unsloth
  user can adopt Gym environments without changing trainers.
- **Honest scope.** Nemotron *steps* need at least two nodes of 8 × A100
  80 GB (or better) on Slurm, DGX Cloud Lepton, or NVIDIA Run:ai and do not
  support workstation training; the model *recipes* were tested on Slurm
  only `[reported — docs/train-models, recipe READMEs]`. Small hardware
  routes to AutoModel PEFT/SFT or the DGX Station recipes (§8). A repo
  default is a `[config]` fact about the repo, not a published result: the
  open-data recipes "will not reproduce" the model-card numbers
  `[reported]`.
- **Evidence.** Values quoted here carry the repo, path, and commit. The
  training *findings* inside the repos (CPT and SFT guidebooks, team FAQ)
  are homed in pretraining.md §4–5, post-training.md §2 and §5, peft.md
  §1, and evaluation.md §2 and §5 as `[reported — repo ablation]`.

## 2. Stage → entry-point map

Step IDs and artifact types are from `src/nemotron/steps/*/step.toml`,
`steps/types.toml`, and the `nemotron-customize` catalog `[config]`;
`nemotron <model> <stage>` recipes exist for `nano3`, `super3`, `super35`,
`ultra3`, `lightning35`, `omni3`, `embed`, `rerank`.

| Stage (skill home) | Nemotron entry point | Backend | Consumes → produces | Notes |
| --- | --- | --- | --- | --- |
| Pretraining from scratch (pretraining.md §1–3) | `pretrain/megatron_bridge`; `nemotron <model> pretrain` | Megatron-Bridge | `binidx` (+ optional `checkpoint_megatron`) → `checkpoint_megatron` | `data_prep/pretrain_prep` tokenizes a blend JSON into bin/idx + `blend.json`; rebuild if the tokenizer changes |
| CPT / midtraining / long-context (pretraining.md §4) | same step with `mode: cpt` + `load_hf_weights: true`; `pretrain/automodel` with `load_weights=true` for HF-native CPT | Megatron-Bridge / AutoModel | as above / `binidx` → `checkpoint_hf` | CPT guidebook decision path in §3 |
| SFT (post-training.md §2) | `sft/megatron_bridge` (packed Parquet) or `sft/automodel` (chat JSONL); `nemotron <model> sft` | Megatron-Bridge / AutoModel | `packed_parquet` → `checkpoint_megatron`; `training_jsonl` → `checkpoint_hf` | `data_prep/sft_packing` before Megatron-Bridge; pack size must equal the training sequence length |
| LoRA / PEFT (peft.md) | `peft/automodel` (HF base, 1–8 GPUs) or `peft/megatron_bridge`; `convert/merge_lora` | AutoModel / Megatron-Bridge | → `checkpoint_lora` | Megatron-Bridge PEFT only for a Megatron base or packed multi-node runs `[reported — CATALOG]` |
| DPO (post-training.md §3) | `rl/nemo_rl/dpo` | NeMo-RL | `training_jsonl` (prompt / chosen / rejected) + `checkpoint_megatron` → `checkpoint_megatron` | `data_prep/rl_prep` resolves HF references and shards; validate pair ordering |
| RLVR / GRPO (post-training.md §5) | `rl/nemo_rl/rlvr` with `config/nemo_gym.yaml`; `nemotron <model> rl` | NeMo-RL + Gym resource servers | `training_jsonl` (Gym records) + `checkpoint_megatron` → `checkpoint_megatron` | `env.should_use_nemo_gym: true`; reward path validated on sample rollouts first |
| RLHF with a generative RM (post-training.md §5, §11) | `rl/nemo_rl/rlhf`; `super3 rl rlhf` | NeMo-RL + Gym `genrm_compare` | + `checkpoint_hf` reward model | separate placement for policy, generation, Gym GPU servers |
| On-policy distillation / MOPD (post-training.md §11, §13) | `super35 rl teacher <name>`, `super35 rl mopd [-c phase2]` | NeMo-RL `super-v3.5-posttraining` branch | RL checkpoints | SFT/RL data not public; the agentic teacher's environment is not in public Gym `[reported]` |
| Agentic RL environments (post-training.md §11–12, §17) | Gym resource servers: `swe_gym`, `swebench`, `swebench_pro`, `swe_pivot`, `terminal_bench_2_1`, `workplace_assistant`, `xlam_fc`, `nvarc` … | Gym + NeMo-RL / verl / Unsloth | Responses-API JSONL rows | `swe_pivot` is the PivotRL verifier (post-training.md §12) |
| Evaluation (evaluation.md) | `gym eval run`; `nemotron <model> eval`; `nemotron_recipes/<model>/` | Gym (NeMo Evaluator for base-model suites and RULER) | rollouts + aggregate metrics | §7 |
| Data curation / synthetic data | `curate/*`, `sdg/data_designer`, `sdg/persona_mcq`, `byob/*`, `translate/*` | NeMo Curator / Data Designer | `raw_jsonl` → `filtered_jsonl` / `synthetic_jsonl` / `training_jsonl` | `curate/decontamination` before training when a holdout must stay clean |
| Quantization / pruning / distillation (out of this skill's scope) | `optimize/modelopt/{quantize,prune,distill}` | Model Optimizer | `checkpoint_hf` → … | `fp8` on Hopper, `nvfp4` on Blackwell `[reported — CATALOG]` |

Chain steps by artifact contract, not by fixed recipe: walk back from the
goal and insert the step whose `produces` matches the next `consumes`
(`nemotron steps list --produces <type>` / `--consumes <type>`); insert a
`convert/*` step only when adjacent stages disagree on checkpoint type;
bookend quality-changing stages with `eval/model_eval` `[reported —
CATALOG selection rules]`.

## 3. Pretraining and continued pretraining

**Generic step `pretrain/megatron_bridge`** `[config — config/default.yaml]`:
`mode: cpt`, HF base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with
`load_hf_weights: true`, recipe `nemotron_3_nano_pretrain_config`, seq
8,192, TP 4 / PP 1 / CP 1 with sequence parallel, GBS 64, `train_iters`
20,000, `optimizer.lr 3.0e-4`, `min_lr 3.0e-5`, weight decay 0.1, warmup
500 iterations, cosine; container `nvcr.io/nvidia/nemo:26.04`;
`checkpoint.pretrained_checkpoint` switches the start to a Megatron
checkpoint (fails fast if missing). **Override the LR for CPT of a converged
base**: 3e-4 is a from-scratch-class peak; pretraining.md §4 gives the two
published CPT regimes (moderate 1e-5-class on a large converged MoE vs
EvoLM's full re-warm on a small dense model), and the Nemotron team's own
long-context CPT phases ran constant 1e-5 (Nano), 4.5e-6 (Super), 2.5e-6
(Ultra) `[reported — repo FAQ]`. Set `optimizer.lr`, `optimizer.min_lr`,
`scheduler.lr_warmup_iters`, `scheduler.lr_decay_style` explicitly.

**Lightning 3.5 pretrain recipe** `[config — recipes/lightning35/stage0_pretrain]`:
Megatron-Bridge `nemotron_3_5_lightning_pretrain_config` on the open blend,
container `nemo:26.08` with Megatron-Bridge `0c565c9a` and its pinned
Megatron-LM mounted; MTP heads live (2 repeated layers, 0.3 loss scaling
`[reported — tiny.yaml comments]`); `data_prep` tokenizer
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`, 128 shards, `add_eos`.
The released Lightning base was **not** built this way — Nano 3 Base
init → 100B-token MTP warm-up → 1T-token distillation from Super 3 Base →
long-context training `[reported — docs citing the 3.5 report]`; the recipe
is a reference implementation.

**CPT decision path** (`steps/pretrain/guidebook`, repo ablation on
Nemotron-3-Nano-30B-A3B in four Indic languages; findings homed in
pretraining.md §4) `[reported]`: write acceptance thresholds first
(adaptation floor, retention budget, minimum useful gain per 5B tokens);
pilot ≈5B target tokens at **1:4 English:target** replay holding the target
corpus fixed; evaluate every checkpoint on native quality *and* retained
capability (ProX-en, ARC-Challenge, HellaSwag, fixed-text BPB); stop on
marginal gain; prefer WSD when early target lift matters and cosine for a
longer run when retention matters. Step count per replay ratio:
`total_tokens = target_tokens × (1 + e / (1 − e))` with `e` the English
share, then `train_iters = total_tokens / (GBS × seq_length)`.

**Pretraining data prep keys** `[config — data_prep/pretrain_prep and the
Lightning data_prep configs]`: `blend_path` (JSON of datasets + weights),
`tokenizer.model`, `add_bos` / `add_eos`, `num_shards`, `valid_shards`,
`test_shards`, `text_field`, `min_doc_chars`, `max_doc_tokens`, `sample`
(row cap for smoke runs), `force`; Ray-based tokenization in
`anyscale/ray:2.55.1-py312`.

## 4. SFT and PEFT

**Backend choice** `[reported — steps/sft/guide.md]`: Megatron-Bridge for
TP / PP / CP / EP control, packed Parquet, Megatron checkpoints, official
Nano 3 / Super 3 patterns, minimum 8 GPUs; AutoModel for chat JSONL with no
packing, HF checkpoints, LoRA, fewer GPUs (minimum 4; the hardware guide
routes a single GPU to AutoModel PEFT). Megatron-Bridge needs
`data_prep/sft_packing` upstream (`tokenizer`, `pack_size`,
`chat_template`, split ratios, shard counts `[config]`).

| Step / recipe | Defaults `[config]` | Skill cross-check |
| --- | --- | --- |
| `sft/automodel` | `TrainFinetuneRecipeForNextTokenPrediction`; base `Qwen/Qwen3-30B-A3B`; GBS 32, local batch 1, 500 steps, ckpt/val every 100; AdamW **1e-5**, wd 0.01, β (0.9, 0.999), ε 1e-8, cosine → min 1e-6; FSDP2 with EP 8; `MaskedCrossEntropy`; `ChatDataset` on OpenAI-messages JSONL; `packed_sequence_size: 0`; container `nemo-automodel:26.04` | inside the full-SFT band 5e-6–2e-5 (text-llm-baselines.md §3) |
| `peft/automodel` | same, plus `PeftConfig` `match_all_linear: true`, `dim 16`, `alpha 32`, `dropout 0.1`, lr **1e-4** | matches peft.md §2: all-linear, α = 2r, LoRA LR ≈ 10× full-FT |
| `sft/megatron_bridge` (`config/lightning35.yaml`, recipe `stage1_sft`) | recipe `nemotron_3_5_lightning_sft_config`; lr **5e-6**, min 0, warmup 10 iters, decay over 100 iters, `train_iters 100` (reference length — scale `train_iters` and `lr_decay_iters` to the data); GBS 128, micro 1; pack size 4,096 = `model.seq_length` = `packed_sequence_size`; TP 2 + sequence parallel, EP 8, PP 1; selective recompute (moe, layernorm, core_attn, mlp); AdamW, per-token loss, role-based loss mask; chat template `lightning35`; splits 98/1/1, `first_fit_shuffle` packing; container `nemo:26.08` + pinned Megatron-Bridge mount | Tulu-3-class LR; Lightning is the only model in this step's `[[models]]` list `[reported]` |
| `super3_128k.yaml` / `super3_256k.yaml` | 64- / 128-rank topology references; **dry-run only** until alignment-aware packing, all four sequence-length fields, precision, and topology are validated `[reported — CATALOG, HARDWARE]` | long-context SFT is a topology problem first |
| DGX Station LoRA (`usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/lora.md`) | one GB300 Station, AutoModel + FSDP2, rank 8, TP/CP/EP 1, seq 32,768, 100 steps, `interactive_agent` subset of `Nemotron-SFT-Agentic-v2` `[reported]` | a demonstration, not a reproduction |

**What to set from the skill's SFT guidance** (post-training.md §2): the
instruct-adaptation recipe from the repo's SFT guidebook — constant 1e-5
over cosine so intermediate checkpoints are comparable, 50k–100k samples,
90:10 reasoning-on : reasoning-off, 20k English instruction-following
replay, full SFT rather than LoRA for new knowledge, and a language-fidelity
metric next to accuracy — plus the team FAQ's rules: fine-tune from the
post-trained checkpoint unless you re-run post-training, convert warmup to
tokens and compare with the dataset (Super's published full SFT: 1e-5
constant after a 30,000-sample linear warmup), and keep a narrow-domain MoE
SFT mixture close to the official post-training blend `[reported]`.
AutoModel: `optimizer.lr`, `lr_scheduler.lr_decay_style` (`cosine` default;
set a constant style for checkpoint comparability),
`step_scheduler.global_batch_size`, `max_steps`. Megatron-Bridge:
`optimizer.lr` / `min_lr`, `scheduler.lr_warmup_iters` / `lr_decay_iters`,
`train.global_batch_size`, `train.train_iters`, `checkpoint.finetune: true`
(skip optimizer state from the pretrained checkpoint).

## 5. RL: GRPO/RLVR, RLHF, DPO on NeMo-RL with NeMo Gym environments

**Generic `rl/nemo_rl/rlvr` with Gym rewards** (`config/nemo_gym.yaml`)
`[config]`: 16 prompts × 8 generations per step, `max_num_steps 500`,
`use_leave_one_out_baseline: true`, `normalize_rewards: true`, advantage
clip ±50; `loss_fn`: KL penalty **0** (type `k3` available), asymmetric
`ratio_clip_min 0.2` / `ratio_clip_max 0.28`,
`truncated_importance_sampling_ratio 5` (`tis`), `token_level_loss`,
`use_on_policy_kl_approximation`, `use_importance_sampling_correction`,
`force_on_policy_ratio`; guardrail switches `penalize_invalid_tool_call:
true` (`invalid_tool_call_advantage −5.0`), `penalize_malformed_thinking:
true` (`−5.0`), `overlong_filtering: false`, `reward_shaping` (overlong
buffer 128, penalty 1) off, `use_dynamic_sampling: false`,
`async_grpo.enabled: false`, `use_best_at_k: false` (k 8, m 1000); policy
default `meta-llama/Llama-3.2-1B-Instruct` via `RL_POLICY_MODEL`, seq
4,096, GBS 32; checkpoints ranked by `val:total_reward/mean`, top-3 kept;
container `nemo-rl:v0.6.0`; `env.should_use_nemo_gym` defaults to **false**
in the step manifest — set it true with `env.nemo_gym.config_paths`. Step
strategies `[reported — step.toml]`: low reward variance → raise
generations per prompt and use the leave-one-out baseline; make
`num_prompts_per_step × num_generations_per_prompt` and policy batch sizes
divisible by the policy shard count; run the reward path on sample rollouts
before scaling GPUs.

**Nemotron recipe configs (production-shaped)** — the numbers to start from
when the policy is a Nemotron-class hybrid MoE:

| Recipe | Prompts × gens (GBS) | Seq | LR / schedule | Loss | Parallelism / infra | Environments |
| --- | --- | --- | --- | --- | --- | --- |
| Lightning 3.5 `stage2_rl/config/default.yaml` `[config]` | 512 × 16 (8,192); 4 val gens | 73,728 | Adam **4e-6 constant** (min 4e-6), warmup 10 iters from 4e-7, β (0.9, 0.999), ε 1e-8, **wd 0**, clip 1.0 | KL 0; clip 0.2 / 0.28; TIS ratio 5 (min 0.2); LOO baseline; normalized rewards; token-level; `overlong_filtering false`; sync GRPO | TP 4 / CP 4 / EP 16 / PP 1, sequence parallel; router **frozen**, fp32, load-balancing none, aux-loss 0; MTP kept training (5 repeated layers, scale 0.3, heads detached); vLLM TP 4, temp 1.0, top_p 1.0, `gpu_memory_utilization 0.5`, Mamba SSM cache fp32, tool parser `qwen3_coder`, reasoning parser `nano_v3`; 32 nodes × 8 GPUs `[reported]`; container `nemo-rl:v0.4.0.nemotron_3_5_lightning` | 6 Gym servers: `math_with_judge` (judge off), `code_gen`, `workplace_assistant`, `mcqa`, `instruction_following`, `structured_outputs_json`; `max_rollout_turns 1`; data `Nemotron-3.5-Lightning-RL-Training-Blend` (placeholders from DAPO-Math-17k, Skywork-OR1) |
| Nano 3 `stage2_rl/config/default.yaml` `[config]` | 128 × 16 (2,048) | 49,152 | **3e-6 constant**, warmup 10 iters from 3e-8, same optimizer | same, TIS ratio null | TP 2 / PP 2 / CP 4 / EP 8; same router hygiene; reasoning parser `deepseek_r1`; container `nemo-rl:v0.4.0.nemotron_3_nano` | same 6 environments; then GenRM RLHF (128 × 16, circular comparison, length bonus α 0.5) and DPO against tool hallucination `[reported — docs]` |
| Super 3 RLVR `[reported — docs/nemotron/super3/rl/rlvr.md]` | 256 × 16 (4,096) | 49,152 → 65,536 | 3e-6; KL 0; `overlong_filtering false` | **async GRPO**: decoupled generation, in-flight weight updates, no KV recompute, policy lag ≤ 1 step, IS-ratio masking | TP 4 / CP 8 / EP 8; 109 nodes (872 GPUs) = 72 generation + 5 Gym GPU nodes; judges Qwen3-235B-A22B, Nemotron-Content-Safety-Reasoning-4B, Qwen3-Nemotron-235B-A22B-GenRM | **21 environments / 37 datasets** jointly in 3 sub-stages; prompts the SFT model always solves are dropped, remainder difficulty-sorted; 1–2% low-effort prompts with token-count-aware reward |
| Super 3 RLHF `[reported — rl/rlhf.md]` | 128 × 16 (2,048) | 49,152 | **1e-6**; **KL 1e-4** (the only non-zero KL) | GenRM (init Qwen3-235B-A22B-Thinking-2507; HelpSteer3 + lmarena subsets + human prefs) | 72 nodes; GenRM router DP 8 | `genrm_compare`, `single_step_tool_use_with_argument_comparison` |
| Gym tutorial, Nano 9B v2 on `workplace_assistant` `[reported — gym docs; YAML lives in the NeMo RL repo]` | 4 × 4 | 32,768 | Adam 5e-6 → min 5e-7, wd 0.01, β (0.9, 0.999), clip 1.0; 10 steps | LOO baseline, normalized rewards | TP 8, one node | agent `max_steps 6` set in the Gym config, not in NeMo RL |
| DGX Station GRPO `[reported — usage-cookbook]` | — | 16,384 | — | sync GRPO, Megatron Core | two GB300 Stations, EP 2, colocated vLLM one TP rank per GPU, activation checkpointing + 75% optimizer CPU offload; NeMo RL pinned at `7fa6e55` plus two patches | `nvarc` (ARC-AGI `python_inductive`) |

**LR regime note.** These GRPO LRs (3e-6–5e-6) sit above the dense,
KL-anchored 3e-7–1e-6 PPO/GRPO band in text-llm-baselines.md §3. They go
with **KL 0, constant schedule, weight decay 0, frozen MoE routers, and
importance-sampling correction**, on hybrid Mamba-Transformer MoE policies.
`[heuristic]` Start a Nemotron-class policy from its recipe config; start a
dense HF policy on the generic step from the 1e-6 DeepSeekMath anchor
unless you also adopt the rest of this regime; never mix a 4e-6 LR with a
KL-penalized, cosine-decayed setup by copying one line.

**MoE router hygiene for RL** `[config — both recipe configs]`:
`freeze_moe_router: true`, `moe_router_dtype: fp32`,
`moe_router_load_balancing_type: "none"`, `moe_aux_loss_coeff: 0.0` — the
config comment states that `seq_aux_loss` "causes logprob error divergence
for grpo". Pair with `use_importance_sampling_correction` and the
train-inference consistency rules in post-training.md §5 and §11.

**Guardrail switches mapped to the skill's failure modes** (post-training.md
§5–6, diagnostics.md): responses piling at the cap →
`grpo.overlong_filtering: true` or `reward_shaping.enabled` with
`overlong_buffer_length` / `overlong_buffer_penalty`; reward-variance
collapse (all-pass / all-fail groups) → `grpo.use_dynamic_sampling: true`
with `dynamic_sampling_oversample_ratio` (DAPO-style) and the LOO baseline;
tool-format hacking → the `penalize_invalid_tool_call` /
`penalize_malformed_thinking` advantages; rollout staleness in async mode →
`async_grpo.max_trajectory_age_steps`; best-of-k objectives →
`use_best_at_k` / `use_combined_training`. Monitor `val:total_reward/mean`
plus the KL / length / reward trio from post-training.md §5.

**DPO step** `[config — rl/nemo_rl/dpo/config/default.yaml]`: β
(`dpo.reference_policy_kl_penalty`) **0.05**, AdamW **5e-7**, wd 0.01,
cosine with 50 warmup steps, GBS 32, micro 1, seq 4,096, 500 steps, 1
epoch, `preference_loss_weight 1`, `sft_loss_weight 0`, dataset default
`HelpSteer3`, checkpoint metric `val:validation-default_loss`; strategy
"loss diverges or KL collapses → raise β to 0.1–0.3 or lower the LR"
`[reported — step.toml]`. Consistent with the 5e-7 DPO anchor in
post-training.md §3; standard-sigmoid β 0.05 sits between Zephyr's 0.01 and
the DPO paper's 0.1. Manifest errors to pre-empt: inverted
prompt/chosen/rejected ordering; batch not divisible by policy workers ×
micro batch; reference policy, tokenizer, and chat template drifting from
the SFT checkpoint.

**RLHF step** `[config/reported — rl/nemo_rl/rlhf/step.toml]`:
`env.should_use_nemo_gym` defaults **true**; the GenRM judge is served by
Gym through `env.nemo_gym.genrm_model.responses_api_models.vllm_model.model`
with its own `vllm` block (TP, `max_num_seqs`, `max_model_len`,
`gpu_memory_utilization`); budget separate placement for policy, generation,
Gym GPU servers, and Ray headroom; validate GenRM readiness before policy
optimization; reward hacking → raise KL, lower LR, or clip rewards.

## 6. Environments, rollouts, and data formats (NeMo Gym)

**What an environment is** `[reported — gym README / AGENTS.md]`: dataset
(JSONL, one task per row) + agent harness (`responses_api_agents/`, FastAPI
server) + verifier and per-task state (`resources_servers/`) + model server
(`responses_api_models/`, or your own), wired by a Hydra YAML config; every
resource server ships `configs/*.yaml`, `data/example.jsonl` (5 tasks), and
tests. Row schema: `responses_create_params.input` (OpenAI Responses-API
messages, so multi-turn tool-calling trajectories need no custom
serialization) + `verifier_metadata`; multi-environment datasets add an
`agent_ref` (`{"type": "responses_api_agents", "name": "<agent>"}`) to route
each row.

**Inventory at `11c874b`** `[config — directory listing + README table]`:
168 resource servers, 62 agent harnesses, 10 model servers; the README
training-table lists 227 environment rows (agent 45, knowledge 36, coding
33, math 25, instruction following 16, rlhf 7, games 6, safety 5, long
context 2), about 80 of them with a published train split. Training
anchors: `math_with_judge` (DAPO-17k train, AIME24 validation at
`num_repeats 32` = mean@32; `should_use_judge` off by default), `code_gen`
(competitive coding, unit-test reward), `workplace_assistant`
(`Nemotron-RL-agent-workplace_assistant`, agent `max_steps 6`),
`instruction_following`, `structured_outputs`, `mcqa`, `xlam_fc`,
`reasoning_gym`, `nvarc`, `abstention`, `genrm_compare`; software
engineering: `swe_gym` (2,438 SWE-Gym tasks from 11 Python repos with
prebuilt images, SWE-bench grading, anti-cheat git scrub), `swebench`,
`swebench_pro`, `swe_pivot` (PivotRL verifier: tool-name → target-file →
argument-similarity ≥ 0.8 → diff-size shaping; binary or continuous
reward), `terminal_bench_2_1`, `r2e_gym`. Harnesses include OpenHands
(`swe_agents`), mini-SWE-agent, Codex, Claude Code, OpenCode, Hermes, Pi,
Terminus-2, LangGraph, Harbor, and `verifiers_agent`, many with sandboxed
variants for training.

**Multi-environment training** `[reported — gym docs; Nemotron docs]`:
start several resource servers in one `gym env start`; "single-environment
training may cause unrecoverable degradation of other benchmarks", and
Super 3's docs say single-environment RL "leads to severe regressions".
Global batch = prompts per step × responses per prompt; scale prompts per
step with the number of environments so each sees many prompts per step
(two environments at 64 prompts ≈ 32 each; fewer can starve one). The
Nemotron recipes run 6 (Nano, Lightning) and 21 (Super) environments
jointly (§5).

**NeMo-RL wiring** `[config — gym tutorial excerpts]`:
`data.default.dataset_name: NemoGymDataset`, `env_name: nemo_gym`,
`processor: nemo_gym_data_processor`; `env.should_use_nemo_gym: true`,
`should_log_nemo_gym_responses`, `should_mask_flagged_samples` (false
ignores environment `mask_sample` flags); `env.nemo_gym.config_paths` lists
the model-server config
(`responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`,
required: `return_token_id_information: true`) and the environment configs;
Gym ports 5000–5999 (NeMo RL 3000–4999, vLLM 7000–8999); agent behaviour
such as `max_steps` is set in the Gym config, and Gym keys are overridden
by nesting them under `env.nemo_gym`.

**Training through external agent harnesses** (Claude Code, Codex, …)
`[reported — gym docs]`: harnesses that drive their own model calls return
text without token IDs; re-tokenizing the transcript yields a different
sequence than the policy sampled. Enable `token_id_capture` (`enabled`,
`dir`, `delta_records: true`, `max_mask_fraction`), pin trainer sampling
with `sampling_overrides` (temperature 1.0, top_p 1.0, top_k −1), keep
`policy.generation.vllm_cfg.skip_tokenizer_init: false` in NeMo RL, and
set `propagate_context_overflow_errors: true` so a context-window refusal is
recorded rather than lost. Rollouts whose call chain cannot be proven get
`mask_sample: true` and drop out of the loss — the TITO principle of
post-training.md §11 as infrastructure.

**Offline SFT / DPO from rollouts** `[reported — experimental tutorial]`:
collect (low temperature, one rollout per task for SFT; higher temperature,
two per task for DPO pairs) → filter by reward and turn count → format as
messages or chosen/rejected; the rejection-sampling cousin of the data
curation in post-training.md §16.

**Framework pairings** `[reported — gym compatibility page]`: Gym v0.4.0 ↔
`nemo-rl:v0.7.0` (Super 3); v0.2.0 ↔ `nemo-rl:v0.5.0.nemotron_3_super`;
v0.1.1 ↔ `nemo-rl:v0.4.0.nemotron_3_nano`; v0.3.0 ↔ the `ultra-v3`
Dockerfile; standalone `nemo-gym:<release>`. Match the container to the
model recipe, not only to the Gym version.

## 7. Evaluation with NeMo Gym

**Commands** `[reported — gym README/docs]`: `gym env start
--resources-server <env> --model-type vllm_model|openai_model` (starts
resource, agent, and policy-model servers) → `gym eval run --no-serve
--agent <agent> --input <jsonl> --output <jsonl> --num-repeats k --limit
n`; metrics `mean/reward`, `pass@1/accuracy`, `pass@1[avg-of-k]/accuracy`;
`gym eval profile` (per-task pass rates — the raw material for difficulty
filters), `gym eval reverify` (rescore stored rollouts without inference),
`gym eval health-check` (unreadable records, missing agent activity,
inconsistent token counts; health ≠ correctness), BLADE diagnostics.

**Nemotron reproducibility recipes** (`nemotron_recipes/lightning-3.5/`,
`super-3.5/`) `[config/reported]`: one YAML per benchmark with the published
run's settings — `gym_commit` pinned (the runner checks out that commit),
`benchmark_config` (e.g. `benchmarks/swebench/verified/opencode.yaml`),
`sampling_overrides` (Super 3.5 SWE-bench and Terminal-Bench: temperature
1.0, top_p 0.95), OpenSandbox, harness (OpenCode sandboxed agent × 3 repeats
for SWE-bench Verified; Terminus-2 × 8 repeats for Terminal-Bench 2.1 with
tasks pinned to `36d417f5`). `LIMIT` smoke runs are not scores. Serving
moves scores as much as eval config — vLLM version, TP, KV-cache dtype,
prefix caching, reasoning and tool-call parsers — so serve as the recipe
states (Lightning instruct validated on `vllm/vllm-openai:v0.26.0`, TP 2 ×
DP 2, fp8 KV cache, `qwen3_coder` tool parser, `nemotron_v3` reasoning
parser) and judge differences against run-to-run spread `[reported]`. This
is the worked example of evaluation.md §4 step 2 and §9's pinning rule.

**Recipe eval stage** `[config — lightning35/stage3_eval/config/default.yaml]`:
serves the HF checkpoint with vLLM (TP 4, expert parallel, Mamba SSM cache
fp32, optional MTP speculative decoding with 5 tokens), runs `gpqa` by
default at concurrency 128; `scicode` needs `test_data.h5`; `hle`, `aalcr`,
`omniscience`, `critpt`, `gdpval`, `pinchbench` need judge models;
`browsecomp` a search key; `tau2` tool servers; base-model suites
(lm-evaluation-harness short-context, RULER) still run through NeMo
Evaluator `[reported]`.

## 8. Hardware, execution, and containers

**Execution model** `[reported — Nemotron docs]`: `env.toml` profiles
(`executor = "slurm"`, `account`, `partition`, `nodes`, `ntasks_per_node`,
`gpus_per_node`, `mounts`; `[wandb]` optional — a file-based manifest
registry replaces W&B artifact lineage); `--run <profile>` attached,
`--batch <profile>` detached, `--dry-run` compiles the config; run
overrides `run.env.nodes`, `run.env.nproc_per_node`, `run.env.time`,
`run.env.container_image`; `env/env_toml` generates every canonical
profile. The pretrain and SFT recipes mount a pinned Megatron-Bridge over
the container with `${auto_mount:…}`, which **requires the Slurm executor**;
RL uses Ray and runs separately from the `pipe` experiment.

**Fast routing** `[reported — nemotron-customize HARDWARE.md]`:

| Hardware | Prefer | Avoid / caution |
| --- | --- | --- |
| 1 GPU | AutoModel PEFT/SFT, small LoRA, curation, endpoint smoke evals | Megatron-Bridge full SFT/pretrain, Super 3, RL |
| 2–4 GPUs | AutoModel SFT/PEFT, Llama/Mistral-class LoRA | Nano 3 Megatron-Bridge SFT unless a step strategy supports it |
| 8 GPUs / 1 node | Nano 3 Megatron-Bridge SFT/PEFT, AutoModel on larger models, distributed smoke | Super 3 SFT/RL, large pretraining |
| 16–32 GPUs | Super 3 SFT pilot, Nano 3 RL, larger SFT/PEFT | Super 3 RL at production rollout scale without profiling |
| 64+ GPUs | Super 3 RL, pretraining, large CPT | launching without a written token / reward / eval budget |

GPU memory starting points: A100 40 GB → AutoModel LoRA or Nano 3 with
aggressive checkpointing; A100/H100 80 GB → Nano 3 Megatron-Bridge SFT
(`tp=4`, `cp=2`-class), Super 3 from ≈32 GPUs; H200 141 GB → larger micro
batches; B200/Blackwell → NVFP4 targets. Interconnect: TP within an
NVLink/NVSwitch domain; PP/DP across InfiniBand; avoid TP spans over
RoCE/Ethernet. Guardrails: never infer GPU count from the model name; start
distributed validation at micro batch 1 with the `tiny` config; keep GBS
divisible by the data-parallel size; tiny configs are wiring tests, not
quality evidence. The **Nemotron post-training calculator**
(`docs/train-models/sizing-calculator.md`) gives directional GPU-count,
runtime, and cost estimates for SFT, LoRA, and GRPO from Megatron-Bridge
memory-fit tables and NeMo-RL throughput references (BF16; not a guarantee)
`[reported]` — pair it with `scripts/estimate.py` and state both as
estimates.

**Containers** `[config]`: `nvcr.io/nvidia/nemo:26.08` (Lightning
Megatron-Bridge stages, plus the mounted Megatron-Bridge `0c565c9a`),
`nemo:26.04` (generic pretrain step), `nemo-automodel:26.04` (AutoModel
SFT/PEFT), `nemo-rl:v0.6.0` (generic RL steps),
`nemo-rl:v0.4.0.nemotron_3_5_lightning` (Lightning RL + eval, bundles vLLM
and Gym), `nemo-rl:v0.4.0.nemotron_3_nano`,
`nemo-rl:v0.5.0.nemotron_3_super`, `nemo-rl:v0.7.0`; Gym standalone
`nemo-gym:v0.7.0`; eval serving `vllm/vllm-openai`. Container-cache
pitfalls the configs already encode: FlashInfer / Triton / Inductor caches
must point at writable mounts (`FLASHINFER_WORKSPACE_BASE`,
`TRITON_CACHE_DIR`, …).

**Monitoring on this stack** `[reported — repo FAQ]`: beyond `lm_loss`,
`seq_load_balancing_loss`, `loss_scale`, and `grad_norm`, the Nemotron team
tracks `params_norm`, `num_zeros`, per-head MTP losses, MaxVio expert
imbalance, and residual-stream norms by depth (evaluation.md §2); RL runs
watch `val:total_reward/mean` and the reward / KL / length trio.

## 9. Config key map: skill knob → stack key

| Skill knob (reference) | Megatron-Bridge (`pretrain/*`, `sft/megatron_bridge`, recipes) | AutoModel (`sft/automodel`, `peft/automodel`) | NeMo-RL (`rl/nemo_rl/*`) |
| --- | --- | --- | --- |
| Peak LR / min LR | `optimizer.lr` / `optimizer.min_lr` | `optimizer.lr` / `lr_scheduler.min_lr` | `policy.megatron_cfg.optimizer.lr` / `min_lr` (Megatron backend); `policy.optimizer.lr` (DPO, DTensor) |
| Warmup | `scheduler.lr_warmup_iters` | — (recipe default) | `policy.megatron_cfg.scheduler.lr_warmup_iters` + `lr_warmup_init`; `policy.scheduler.warmup_steps` (DPO) |
| Schedule | `scheduler.lr_decay_style` (`cosine`), `lr_decay_iters` | `lr_scheduler.lr_decay_style` | `policy.megatron_cfg.scheduler.lr_decay_style` (`constant` in the recipes); `policy.scheduler.name` (DPO) |
| Weight decay / β / ε / clip | `optimizer.weight_decay`; recipe defaults | `optimizer.weight_decay`, `betas`, `eps` | `policy.megatron_cfg.optimizer.weight_decay`, `adam_beta1/2`, `adam_eps`; `policy.max_grad_norm` |
| Global / micro batch | `train.global_batch_size`, `train.micro_batch_size` | `step_scheduler.global_batch_size`, `local_batch_size` | `policy.train_global_batch_size`, `train_micro_batch_size`; GRPO rollout batch = `grpo.num_prompts_per_step × num_generations_per_prompt` |
| Steps / epochs | `train.train_iters` | `step_scheduler.max_steps` | `grpo.max_num_steps`, `max_num_epochs`; `dpo.max_num_steps` |
| Sequence length / packing | `model.seq_length` = `dataset.seq_length` = `dataset.packed_sequence_specs.packed_sequence_size` = prep `pack_size` | `packed_sequence.packed_sequence_size` (0 = none) | `policy.max_total_sequence_length` (= vLLM `max_model_len`); `policy.sequence_packing.enabled` |
| Parallelism | `model.tensor_model_parallel_size`, `pipeline_…`, `context_parallel_size`, `expert_model_parallel_size`, `sequence_parallel` | `distributed.tp_size/cp_size/pp_size/ep_size` (FSDP2) | `policy.megatron_cfg.tensor_model_parallel_size`, `context_…`, `expert_model_…`, `pipeline_…`; generation `policy.generation.vllm_cfg.tensor_parallel_size`, `gpu_memory_utilization` |
| Base checkpoint | `checkpoint.pretrained_checkpoint`, `checkpoint.finetune`; `hf_model_path` + `load_hf_weights` | `model.pretrained_model_name_or_path` | `policy.model_name` (+ recipe `initial_checkpoint`) |
| Data | `dataset.data_paths` / `blend.json`; `dataset.lightning35_packed_sft_dir` | `dataset.path_or_dataset_id`, `validation_dataset.…` | `data.train.data_path`, `data.validation.data_path`; DPO `data.train.prompt_file` |
| LoRA r / α / dropout / targets | `peft/megatron_bridge` config | `peft.dim`, `peft.alpha`, `peft.dropout`, `peft.match_all_linear` | — |
| GRPO G, clip ε, KL β, baseline | — | — | `grpo.num_generations_per_prompt`; `loss_fn.ratio_clip_min/max`; `loss_fn.reference_policy_kl_penalty`; `grpo.use_leave_one_out_baseline`, `normalize_rewards` |
| DPO β | — | — | `dpo.reference_policy_kl_penalty` |
| Sampling temperature / top_p | — | — | `policy.generation.temperature`, `top_p` (also pin Gym `sampling_overrides` for external harnesses) |
| Environments | — | — | `env.should_use_nemo_gym`, `env.nemo_gym.config_paths`; agent `max_steps` inside the Gym YAML |
| Checkpoint selection metric | `checkpoint.save_interval` | `step_scheduler.ckpt_every_steps` | `checkpointing.metric_name` (`val:total_reward/mean`), `keep_top_k`, `save_period` |

Keys verified at the pinned commits; a different checkout can rename them —
`nemotron steps show <id> --json` lists the live parameters.

## 10. Pre-launch verification

1. `uv run nemotron steps list --category <cat>` → `steps show <id>
   --json`: confirm `consumes` / `produces`, parameters, run spec, and the
   named configs (`default`, `tiny`, `nemo_gym`, …) before writing YAML.
2. Compile without submitting: `… -c <config> --run <profile> --dry-run`;
   then the **`tiny`** config at micro batch 1 on real hardware — a wiring
   test, not quality evidence.
3. Gym: `gym env start …` → `gym env status` → `gym eval run --limit 5
   --num-repeats 1` on `data/example.jsonl`; validate the verifier on sample
   rollouts before raising rollout count or GPUs; `gym env resolve` dumps
   the merged config.
4. Consistency locks: pack size = `seq_length` = `packed_sequence_size`;
   GBS divisible by data-parallel size and policy shard count; tokenizer,
   chat template, and sequence length identical across every step that
   touches one model line; reference policy = the SFT checkpoint for DPO.
5. Report every repo-derived value with its commit; if the user's checkout
   is newer, re-read the file instead of trusting this page. Never emit a
   flag or path you have not seen in the live repo.
