# Changelog

Knowledge changes (new sources, corrected numbers) and packaging changes are
listed separately in each entry. The version here must match
`.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`
(locked by `tests/test_doc_consistency.py`).

## 1.1.0 — SEED (self-evolving OPD for agentic RL)

Knowledge:

- Add SEED (arXiv 2607.14777 + jinyangwu/Seed repo): new
  `post-training.md` §17 — self-evolving hindsight-skill distillation for
  agentic RL. The policy analyzes its own completed trajectories into
  hindsight skills and distills the skill-induced log-prob shift back into
  itself via a sigmoid-gated OPD loss joint with GRPO (`L = L_rl +
  λ_opd·L_opd`; repo defaults λ_opd 0.01, β_opd 5.0 `[config]` — not in
  the paper text). Positions SEED as the fourth OPD teacher construction
  (self-teacher with privileged hindsight context) alongside §11/§13/§15;
  new algorithm-chooser row for sparse-reward agentic RL without a
  stronger teacher; bibliography row + `SEED` added to the SKILL.md
  trigger source list and the dashboard DATA block.

## 1.0.0 — plugin packaging, tests, CI (state to date)

First versioned release. Knowledge state to date (full history in `git log`):
recipes distilled from LLaMA 1-3, OLMo 1-3, DeepSeek-V3, SmolLM2, Kimi K2,
GLM-5, PivotRL, Agents-A1, EvoLM, LFM2, VibeThinker, FAC-Synthesis, Zephyr,
Tulu 3, SimPO, ORPO, QLoRA, Whisper, OWSM, wav2vec 2.0, HuBERT; three-tier
evaluation framework; HTML-artifact experiment tracking with report-detail
standard and dashboard template.

Knowledge fixes:

- `estimate.py` LR ladder: pretrain ≤2B band corrected to `2e-4 - 1e-3`
  (was `4e-4 - 1e-3`) and EvoLM-1B 2e-4 `[config]` added to its sources —
  realigning the script with the SKILL.md table after the EvoLM integration.
  `tests/test_doc_consistency.py` now locks the two against each other.

Packaging:

- Installable as a Claude Code plugin: `.claude-plugin/plugin.json` +
  `marketplace.json` (`/plugin marketplace add voidful/AlchemistPlaybook`),
  plus a `.codex-plugin/plugin.json` for Codex. The skill stays at
  `alchemist-playbook/`, so the previously documented clone+symlink install
  keeps working.
- MIT `LICENSE` (previously no license — all rights reserved by default).
- `tests/` suite (stdlib `unittest`): `estimate.py` math/CLI behavior,
  template validity (DeepSpeed JSON, TRL YAML, dashboard self-containedness),
  SKILL.md ↔ `estimate.py` consistency, §-citation integrity, version sync.
- GitHub Actions CI running `scripts/check_docs.py` + the test suite on every
  push/PR.
