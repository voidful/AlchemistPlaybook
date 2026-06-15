# Evaluation and monitoring

Three different questions, three different cadences — never conflate them:

- **Health monitoring** — is the run alive and sane *right now*? (step/hour)
- **Capability regression** — is each checkpoint *getting stronger without
  forgetting*? (every N tokens / per epoch)
- **Release gate** — can this checkpoint *ship*? (before release only)

A run can be "healthy" (loss falling, no NaN) while silently *regressing* a
capability the total loss hides, or *passing public benchmarks* while failing
the real product. This file is the catalog of what to watch at each tier and
each stage. Verification tags: `[paper]` lab practice stated in a report,
`[reported]` secondary, `[heuristic]` synthesized best practice with no single
published anchor. Benchmark facts (option counts, prompt counts) are stated
plainly because they are properties of the benchmark, not opinions.

## Contents
1. The three-tier framework
2. Pre-training: what to monitor
3. Pre-training: what to evaluate
4. Mid-training: monitor + the retention suite
5. Post-training: SFT monitoring
6. Post-training: DPO / RLHF / RL monitoring
7. Benchmark catalog by capability
8. Minimal eval suites (the deliverable)
9. Release gate and honesty notes

## 1. The three-tier framework `[heuristic, matches OLMo/Llama 3/Tulu 3 practice]`

| Tier | Purpose | Frequency | Owns |
| --- | --- | --- | --- |
| Training health monitoring | catch crashes, data anomalies, system faults | step / hour | loss, grad-norm, throughput, NaN/Inf, MoE balance |
| Capability regression eval | is the checkpoint stronger / forgetting? | every N tokens / per epoch | frozen val slices + cheap downstream probes |
| Release gate eval | decide go / no-go for ship | before release | full public benchmarks + private holdout + product/safety |

Two rules that prevent most eval mistakes:
- **Public benchmarks are for horizontal comparison; a private holdout eval is
  the release gate.** Public sets leak into training data over time
  (contamination); only a freshly-built, never-published holdout can gate a
  version honestly. (Tulu 3, Llama 3 both maintain private evals `[paper]`.)
- **Compare each stage against its *stage input* model, not just the final
  target.** Each stage must independently justify itself — Tulu 3's per-stage
  delta tables are the model to imitate `[paper]`.

## 2. Pre-training: what to monitor

| Category | Metrics | Why |
| --- | --- | --- |
| Loss / PPL | train loss, val loss, perplexity, **bits-per-byte**, loss spike | is training descending stably (BPB is tokenizer-independent — use it to compare across tokenizers) |
| Bucketed loss | per-domain val loss: zh / en / code / math / web / books / papers / QA | **prevent total loss masking local regression** — the single most missed signal |
| Scaling metrics | loss vs tokens, loss slope, loss vs compute, predicted final loss | is the run on its expected scaling-law trajectory (scaling-and-batch.md) |
| Optimization stability | LR, gradient norm, grad-clipping ratio, **update-to-weight ratio**, Adam moments | catch impending divergence before the spike (stability.md) |
| Numerical health | NaN/Inf, activation norm, weight norm, **logit norm**, entropy, mixed-precision overflow | detect numerical instability early (logit growth precedes divergence) |
| Data quality | dedup rate, low-quality ratio, toxicity ratio, PII ratio, language dist, domain dist, doc-length dist | **most pretraining failures originate in the data pipeline** (diagnostics.md triage order) |
| Data contamination | benchmark overlap, n-gram overlap, train-set hit rate, canary extraction | **prevent inflated benchmarks**; Llama 3 ran per-benchmark contamination analysis `[paper]` |
| Tokenizer | token/char, zh/code compression rate, UNK/abnormal-token rate, special-symbol ratio | matters most for multilingual / Chinese / code |
| System efficiency | tokens/sec/GPU, MFU, GPU util, memory, comm time, checkpoint time, fault-recovery time | controls cost and stability (estimate.py flops for MFU targets) |
| **MoE-specific** | expert load balance, router entropy, dropped-token rate, expert-capacity overflow | **mandatory for MoE** — collapsed routing is invisible in total loss (DeepSeek-V3 aux-loss balancing) |

Cadence: loss/grad-norm/LR/tokens/tok-s every ≤50 steps; numerical-health
flags every step (cheap); bucketed val + scaling check every ~1000 steps
(OLMo in-loop cadence `[paper]`). See diagnostics.md for the symptom→fix map.

## 3. Pre-training: what to evaluate

| Capability | Metric | Benchmark |
| --- | --- | --- |
| Language modeling | NLL, PPL, BPB | held-out validation set |
| General knowledge | accuracy, log-likelihood accuracy | MMLU, MMLU-Pro |
| Commonsense reasoning | accuracy | ARC, HellaSwag, Winogrande, OpenBookQA, BBH |
| Chinese | accuracy, per-subject | C-Eval, CMMLU, AGIEval |
| Math | exact match, pass@k, maj@k | GSM8K, MATH, MATH-500 |
| Code | pass@1, pass@k, compile rate, unit-test pass | HumanEval, MBPP (add LiveCodeBench later) |
| Multilingual | accuracy, F1, cross-lingual transfer | XNLI, TyDiQA, XTREME, FLORES-200 |
| Long context | retrieval accuracy by length, position sensitivity | Needle, LongBench, RULER |
| Safety baseline | toxicity, bias, PII leakage, memorization | RealToxicityPrompts, TruthfulQA, self-built safety smoke test |

`MMLU-Pro` adds harder, reasoning-heavier questions and expands choices from
4 → **10 options**, so it discriminates strong models better than MMLU.
`RULER` beats vanilla Needle-in-a-Haystack for long-context because length and
task complexity are configurable and it covers multi-needle, multi-hop
tracing, and aggregation — not just single-fact retrieval.

For pretraining-stage *eval as a data-quality tool*: anneal a checkpoint on a
candidate data mix and measure the downstream delta (Llama 3 / FineWeb
annealing-as-evaluator `[paper]`; see pretraining.md §4).

## 4. Mid-training: monitor + the retention suite

Mid-training = continued training **after pretraining, before/between
SFT/RLHF**: domain continued-pretraining, math/code enhancement, long-context
extension, multilingual boost, knowledge refresh, tool-use pretraining,
high-quality data annealing. Its success metric is **not** "did the target
capability go up" — it is the **Pareto**:

```
target capability ↑   −   general capability forgotten   −   safety regression   −   added cost
```

| Category | Metrics | Why |
| --- | --- | --- |
| Target-domain loss | domain val loss, target-corpus PPL | is the specialization working at all |
| **General retention** | general val loss + MMLU / C-Eval / GSM8K / HumanEval small-set regression | **prevent catastrophic forgetting** — the defining risk of this stage |
| Data mixing | target/general replay ratio, synthetic/human ratio, difficulty distribution | control transfer vs forgetting (pretraining.md §6 difficulty-curriculum) |
| Synthetic-data quality | teacher score, self-consistency, repetition rate, format-error rate, answer-verifiable rate | **prevent synthetic data from poisoning the model** |
| Long context | loss by position, position coverage, loss by length, attention sink, needle accuracy by depth | **prevent "window extended but effective context not improved"** |
| Math reasoning | final-answer accuracy, CoT length, invalid-reasoning ratio, self-consistency gain | is it really reasoning vs memorizing |
| Code | compile rate, unit-test pass, syntax/import errors, repo-level success | is the code executable, not just plausible |
| Tool prep | tool-schema valid rate, argument exact match, tool-selection accuracy | mandatory if tool/agent ability is a goal |
| Safety regression | harmful compliance, jailbreak ASR, false refusal | **prevent capability gains from eroding the safety boundary** |

**Mid-training must own a fixed retention suite**, minimally:
`general knowledge + Chinese + math + code + long-context + safety + target
domain`. Watching only the target-domain benchmark is dangerous: the model can
gain in-domain while silently regressing general ability, Chinese, safety, or
tool use. Mid-training eval by training type:

| Training type | Core eval |
| --- | --- |
| General continued-pretrain | MMLU-Pro, C-Eval, CMMLU, BBH, GPQA, GSM8K, HumanEval, held-out PPL |
| Math / reasoning boost | GSM8K, MATH, MATH-500, AIME, AMC, OlympiadBench, GPQA |
| Code boost | HumanEval, MBPP, LiveCodeBench, APPS (add SWE-bench Verified) |
| Chinese boost | C-Eval, CMMLU, AGIEval, SuperCLUE, self-built multi-turn / long-text |
| Long-context boost | RULER, LongBench, InfiniteBench, Needle, self-built long-doc QA |
| Tool / Agent prep | BFCL, τ-bench, ToolBench/StableToolBench, self-built API-call set |
| Domain boost | medical (HealthBench/MedQA), legal (LegalBench), finance (FinanceBench/FinQA) + private business set |

## 5. Post-training: SFT monitoring

| Category | Metrics |
| --- | --- |
| SFT loss | **assistant-token loss** (masked), validation loss, per-task loss |
| Instruction following | instruction pass rate, hard-constraint satisfaction, language/length/format-constraint pass |
| Format ability | JSON validity, schema valid rate, markdown/table/XML format correctness |
| Multi-turn | context carry-over, coreference resolution, context consistency, contradiction rate |
| Answer quality | helpfulness judge score, human preference score, win rate vs base |
| Refusal behavior | safe-refusal rate, **benign false-refusal rate**, over-refusal |
| Factuality | hallucination rate, unsupported-claim rate, citation-error rate |
| Style | avg length, verbosity rate, template-ization rate, repetition rate, verbosity drift |
| Tool use | tool-selection accuracy, argument accuracy, invalid-JSON rate, execution success |

Watch refusal *both ways*: a model that refuses harmful prompts but also
refuses benign ones (over-refusal) has not been aligned, it has been crippled.

## 6. Post-training: DPO / RLHF / RL monitoring

| Category | Metrics | Risk it catches |
| --- | --- | --- |
| Preference optimization | preference loss, chosen/rejected margin, reward-model accuracy, AUC, calibration | preference-data noise amplified |
| RL stability | reward, **KL to reference**, policy entropy, value loss, advantage mean/std, clip fraction | KL too high → model drifts; KL too low → learns nothing |
| Reward hacking | reward ↑ but human eval ↓, answers grow longer, template-ization, sycophancy | the canonical post-training failure |
| Safety | harmful compliance, jailbreak ASR, false refusal, policy-violation rate | safety and helpfulness must be watched together |
| Tool / Agent | task success, invalid action, loop rate, timeout, error recovery, cost | agents are judged end-to-end, not per-call |
| Product experience | TTFT, tokens/s, latency, cost, timeout, user satisfaction, complaint rate | the only tier that matters at release |

The reward-hacking signature is **reward ↑ + KL ↑ + length ↑ together** — track
all three on one plot (post-training.md §5 RL guardrails).

## 7. Benchmark catalog by capability (post-training / release)

| Capability | Benchmarks | Metric |
| --- | --- | --- |
| Instruction following | IFEval | strict / loose accuracy, constraint satisfaction |
| Dialogue quality | MT-Bench, MT-Bench-101, Arena-Hard, WildBench, AlpacaEval 2, human pairwise | win rate, judge score, Elo |
| Factuality | SimpleQA, TruthfulQA, FActScore, self-built fact QA | correctness, incorrectness, not-attempted, unsupported claims |
| Math / reasoning | MMLU-Pro, GPQA, BBH, MATH, AIME, OlympiadBench | accuracy, exact match, pass@k, maj@k |
| Code | LiveCodeBench, HumanEval, MBPP, SWE-bench Verified | pass@1, resolved rate, unit-test pass |
| Tool use | BFCL | tool selection, schema match, argument accuracy, execution accuracy |
| Agent | τ-bench, WebArena, GAIA, BrowseComp, self-built business tasks | task success, pass^k, step count, timeout, cost |
| Long context | RULER, LongBench, InfiniteBench, Needle | accuracy by length, position sensitivity, effective context length |
| Safety | HarmBench, JailbreakBench, AdvBench, SafetyBench, WMDP, XSTest, internal red team | harmful compliance, jailbreak ASR, false refusal |
| RAG | self-built evidence set, FreshQA-style | answer correctness, groundedness, citation precision/recall |

`IFEval` evaluates *auto-verifiable* instruction following (length, keyword,
format hard-constraints) — ~500 prompts, ~25 verifiable instruction types.
`SimpleQA` targets short fact-seeking factuality. `HarmBench` is the standard
for safety red-team + robust-refusal evaluation. Prefer benchmarks with
**verifiable** answers (exact-match math, unit-tested code, constraint
checkers) over LLM-judge scores wherever possible — judges drift and can be
gamed.

## 8. Minimal eval suites (the deliverable)

When a user asks "what should I actually run," hand them the right minimal set
for their stage. These are the floor, not the ceiling.

**Pre-training minimal set:**

| Module | Recommended |
| --- | --- |
| Loss | total val loss + per-domain bucket loss (zh/en/code/math/web/books/papers) |
| General | MMLU, MMLU-Pro subset |
| Chinese | C-Eval, CMMLU |
| Math | GSM8K, MATH-500 |
| Code | HumanEval, MBPP |
| Long context | Needle, RULER subset |
| Multilingual | XNLI / TyDiQA subset |
| Safety | toxicity, PII, memorization canary |
| Contamination | benchmark overlap, n-gram overlap, near-dup |

**Mid-training minimal set:**

| Module | Recommended |
| --- | --- |
| Target domain | target held-out loss + target benchmark |
| General retention | MMLU-Pro, C-Eval, CMMLU, GSM8K, HumanEval |
| Math | MATH-500, AIME, GPQA |
| Code | LiveCodeBench, SWE-bench Verified |
| Long context | RULER, LongBench, Needle |
| Tool / Agent | BFCL, τ-bench, self-built tool-call set |
| Safety regression | HarmBench subset, false-refusal set, internal red team |
| Business | private golden set, online-failure samples, hard-case set |

**Post-training minimal set:**

| Module | Recommended |
| --- | --- |
| Instruction following | IFEval |
| Dialogue preference | AlpacaEval 2, Arena-Hard, MT-Bench, human pairwise |
| Factuality | SimpleQA, TruthfulQA, self-built fact QA |
| Reasoning | MMLU-Pro, GPQA, MATH/AIME, BBH |
| Chinese | C-Eval, CMMLU, AGIEval, SuperCLUE, self-built multi-turn |
| Code | LiveCodeBench, SWE-bench Verified |
| Tool / Agent | BFCL, τ-bench, WebArena, GAIA, BrowseComp |
| Safety | HarmBench, JailbreakBench, WMDP, SafetyBench, XSTest, internal red team |
| Product | TTFT, tokens/s, latency, cost, timeout, tool-call success, user satisfaction |

## 9. Release gate and honesty notes

One-line summary of the whole file:

> Pre-training: watch loss, data, stability, base capability. Mid-training:
> watch the **Pareto** of target-capability gain vs general-capability
> retention (plus safety and cost). Post-training: watch instruction
> following, preference win-rate, factuality, safety, tool/Agent, and real
> user experience. Use **public benchmarks for horizontal comparison** and a
> **private holdout for the version release gate**.

Honesty rules specific to evaluation:
- A public-benchmark number with no contamination check is not evidence —
  always pair "benchmark ↑" with "overlap/canary clean" (diagnostics.md:
  "Benchmarks up, real usage worse").
- LLM-judge scores (MT-Bench, AlpacaEval) are relative, not absolute, and
  drift with the judge model — report the judge and version, prefer pairwise.
- Never tune on the release-gate holdout. The moment you do, it stops gating.
