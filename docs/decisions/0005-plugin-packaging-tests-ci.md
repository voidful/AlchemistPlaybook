---
type: decision
title: "Plugin packaging, versioned releases, tests, and CI (research-cards adoption)"
status: active
decided: 2026-08-19
updated: 2026-08-19
---

# 0005 — Plugin packaging, versioned releases, tests, CI

## Context

The repo shipped only a bare `.skill` archive: no license (all rights
reserved by default), no one-command install, no version identity, and no
tests — `estimate.py` backs every quantitative answer yet nothing checked its
math, and its LR ladder had silently drifted from the SKILL.md table (≤2B
pretrain band: script said `4e-4`, SKILL.md said `2e-4` after the EvoLM
integration). Meanwhile
[research-cards](https://github.com/SungFeng-Huang/research-cards) — whose
`research-campaign` skill requires hyperparameter decisions to cite this
playbook — had to document a manual clone+symlink install for this repo.

A deep analysis of research-cards' engineering (multi-agent: 4 proposal
lenses, 3 adversarial judges over 22 proposals) selected 13 practices worth
adopting and rejected 9 as cargo-culting or duplicate.

## Decision

1. **Plugin manifests** — `.claude-plugin/plugin.json` +
   `.claude-plugin/marketplace.json` (Claude Code) and
   `.codex-plugin/plugin.json` (Codex). The Claude manifest uses a custom
   skills path (`"skills": ["./alchemist-playbook/"]`), so the skill stays
   where it is: no directory move, and research-cards' documented
   clone+symlink install keeps working unchanged.
2. **MIT LICENSE**, declared in both manifests.
3. **Versioned releases** — `CHANGELOG.md` with one entry per version;
   the version string must match in both plugin manifests (locked by
   `tests/test_doc_consistency.py`).
4. **Test suite** (`tests/`, stdlib `unittest`, no hard third-party deps):
   `estimate.py` math and CLI contract; template parse-validity with anchor
   values imported from the `LADDER` (single-sourced, per invariant 2);
   SKILL.md ↔ `estimate.py` LR lock (the drift above is now impossible to
   reintroduce silently); `§N`-citation integrity across the skill;
   version sync.
5. **CI** — GitHub Actions runs `scripts/check_docs.py` and the test suite
   on every push/PR. A red check means an invariant was violated, including
   archive freshness.

Deliberately **not** adopted from research-cards (judge-rejected):
multi-language READMEs (standing translation-sync cost), moving the archive
to GitHub Releases, a `--json` mode for `estimate.py`, eval-harness CLI
automation, and shipping GitHub-Pages workflow assets — each copies
infrastructure whose preconditions this knowledge-skill repo lacks.

## Consequences

- One-command install: `/plugin marketplace add voidful/AlchemistPlaybook`.
- Knowledge corrections become releases: version bump + CHANGELOG entry in
  the same commit as the repackage (release workflow step 4).
- `estimate.py` and the templates are now under test; a number can no longer
  change in one home without the lock failing loudly.
