# Alchemist Playbook

Evidence-based training-recipe skill for model training and tuning. The skill
turns published training recipes into a practical diagnostic workflow for
pretraining, SFT, DPO/RLHF/RLVR, LoRA/QLoRA, and speech training runs.

## Contents

- `alchemist-playbook/` - source skill directory.
- `alchemist-playbook.skill` - packaged skill archive.
- `alchemist-playbook-workspace/` - evaluation workspace and iteration output.
- `AGENTS.md` - original project instruction.

## Use

Install or import `alchemist-playbook.skill` in a Codex environment that accepts
skill archives, or inspect `alchemist-playbook/SKILL.md` directly for the
workflow and routing table.

For quantitative estimates, use:

```bash
python alchemist-playbook/scripts/estimate.py batch --gpus 8 --micro 4 --seq 4096 --accum 8
python alchemist-playbook/scripts/estimate.py lr --params 8e9 --stage sft
```

## Validation

The packaged archive contains the skill source, references, helper script, and
training templates. The workspace includes a first evaluation iteration for
recipe debugging and config review tasks.
