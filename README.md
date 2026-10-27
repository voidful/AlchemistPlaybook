# Alchemist Playbook

**Evidence-based model-training recipes and run diagnosis for AI coding agents.**

Alchemist Playbook supports 煉丹調參 — training-run hyperparameter tuning.
It covers pretraining, SFT, preference tuning, RLVR, LoRA/QLoRA, and speech training.
Recommendations connect your model, data, and budget to published runs and official configs.

You get a sourced recipe or diagnosis, concrete config changes, and a way to verify the result.
The skill can also estimate training compute and help record experiments.

## Install

### Claude Code

```text
/plugin marketplace add voidful/AlchemistPlaybook
/plugin install alchemist-playbook@alchemist-playbook
```

### Codex

Clone the repository, then link the skill into your personal skills directory:

```bash
git clone https://github.com/voidful/AlchemistPlaybook ~/AlchemistPlaybook
mkdir -p ~/.agents/skills
ln -s ~/AlchemistPlaybook/alchemist-playbook ~/.agents/skills/alchemist-playbook
```

For an existing clone, use its path in the link command.
If the skill link already exists, check its target before changing it.
The repository also includes a [Codex plugin manifest](.codex-plugin/plugin.json).

### Skill archives

Download [alchemist-playbook.skill](alchemist-playbook.skill) for environments that accept skill archives.
The [skill source](alchemist-playbook/SKILL.md) is plain Markdown with bundled references, scripts, and templates.

## Use it

Ask a training question, or explicitly invoke `$alchemist-playbook` in Codex:

```text
$alchemist-playbook Review my attached DPO config and training logs.
Fix the local config, explain the evidence, and validate the changes.
Do not launch training.
```

Give the context you have: stage, model, data, hardware, config, and symptoms.
The agent uses available files and asks when a missing fact affects its next decision.
Incomplete logs can still support a provisional diagnosis.

| Task | Example request | Expected deliverable |
| --- | --- | --- |
| Recipe design | "Design a pretraining recipe for this model and token budget." | Sourced starting config, assumptions, pilot checks |
| Run diagnosis | "Loss spikes after warmup. Inspect these curves and the optimizer config." | Ranked causes, relevant config changes, verification and rollback criteria |
| Preference tuning | "Review the learning rate and beta for my DPO loss variant." | Objective-aware review with named source recipes |
| Speech training | "My ASR model diverges during warmup. Review this ESPnet config." | Speech-specific diagnosis and recipe anchors |
| Compute estimates | "Estimate GPU-hours from this model, dataset, and hardware budget." | Reproducible calculation with assumptions |
| Stage evaluation | "Choose an evaluation suite for continued pretraining." | Monitoring, capability checks, and a separate release holdout |
| Experiment records | "Summarize these runs in our existing Markdown notes." | A report in the requested format, with provenance and decision criteria |
| Implementation on NeMo | "Turn this SFT and GRPO plan into Nemotron steps and a NeMo Gym environment config." | Step chain, config overrides with evidence labels, verification commands |

For calculator options:

```bash
python3 alchemist-playbook/scripts/estimate.py --help
```

Use its `flops`, `chinchilla`, `batch`, and `lr` subcommands for supported calculations.
The [scaling reference](alchemist-playbook/references/scaling-and-batch.md) explains the assumptions.

## How the skill works

The entrypoint is a short task guide and reference router.
Detailed recipes load when relevant to the question.
The [text-LLM baseline and LR ladder](alchemist-playbook/references/text-llm-baselines.md) are a separate reference.

Implementation defaults to the NVIDIA NeMo stack.
When you have not fixed a trainer, the skill maps its recipe onto [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) steps and recipes (Megatron-Bridge, NeMo AutoModel, NeMo-RL) and [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) environments and evaluation.
It names the config key that carries each value and the commit the value was read from.
Your own trainer still takes precedence.
See the [NeMo stack reference](alchemist-playbook/references/nemo-stack.md) and [decision 0007](docs/decisions/0007-nemo-stack-default-implementation.md).

Each recommended training value carries an evidence label:

| Label | Meaning |
| --- | --- |
| `[config]` | Read from an official config, training script, or model card |
| `[paper]` | Stated in the source paper or technical report |
| `[reported]` | Reported guidance with its source and limits identified |
| `[heuristic]` | An explicit judgment or extrapolation |

See the [bibliography](alchemist-playbook/references/sources.md) for sources and verification status.
The [routing table](alchemist-playbook/SKILL.md#routing) maps questions to references.

The skill follows your requested format and existing workflow.
A config edit includes relevant local validation.
Larger reports can use the optional HTML dashboard and git workflow.
Training launches, commits, and publishing follow the scope you authorize.

Pure inference configuration and general prompt writing are outside the training scope.
The instruction audit draws on [Eric Provencher's article](https://x.com/pvncher/status/2095991462416490862)
and the [OpenAI Astra guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra).
The resulting choices are recorded in [decision 0006](docs/decisions/0006-task-scoped-skill-guidance.md).
These are workflow sources; they provide no training hyperparameters.
The skill does not require Astra or change your selected model.

## Documentation and evaluation

| Resource | Purpose |
| --- | --- |
| [Documentation index](docs/index.md) | Repository architecture, conventions, governance, and workflows |
| [Interactive report](docs/index.html) | Routing, bibliography, historical evaluations, and decision history; open the file locally |
| [Eval definitions](alchemist-playbook-workspace/evals/evals.json) | Training cases and behavioral regression scenarios |
| [Recorded benchmark](alchemist-playbook-workspace/iteration-1/benchmark.md) | Historical with-skill and without-skill results |
| [Eval workflow](docs/workflows/run-evals.md) | Execution procedure, provenance requirements, and current validation status |
| [Changelog](CHANGELOG.md) | User-visible changes |

The recorded benchmark predates the instruction audit.
The new behavioral cases are defined but have not been run in fresh agent environments.
Local checks validate packaging, references, calculator consistency, and templates.
They do not establish training success or current-model performance.

## Optional integration

[research-cards](https://github.com/SungFeng-Huang/research-cards) uses this playbook as a source of hyperparameter citations for research campaigns.
The playbook also works on its own.

## Contribute

Read [AGENTS.md](AGENTS.md) and the [documentation index](docs/index.md).
Follow the [add-source workflow](docs/workflows/add-source.md) for new training evidence.
Record structural or workflow changes in a decision before implementation.

Run the repository checks:

```bash
python3 scripts/check_docs.py
python3 -m unittest discover -s tests
```

Changes to the skill require [repackaging](docs/workflows/release.md).
User-visible changes also require a changelog entry and matching plugin versions.
The archive and Pages report are outputs of the repository sources.

## License

[MIT](LICENSE).
