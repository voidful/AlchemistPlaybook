---
type: workflow
title: Repackage the skill archive
status: active
updated: 2026-08-19
---

# Workflow: release / repackage

`alchemist-playbook.skill` is a **zip archive of the `alchemist-playbook/`
directory** (stored, extension renamed). It is a build output: never edit it
directly, and never let it drift from the source tree (constitution
invariant 1 — docs-first).

## Steps

1. Pre-flight checks:

```bash
python3 scripts/check_docs.py
```

```bash
python3 -m unittest discover -s tests
```

   `check_docs.py` includes the ≤ 1024-char description check and the
   archive-freshness check; the tests lock SKILL.md to `estimate.py` and
   validate the templates.

2. Rebuild the archive from the repo root (exclude Python bytecode):

```bash
rm -f alchemist-playbook.skill && zip -r -X alchemist-playbook.skill alchemist-playbook -x "*__pycache__*" -x "*.pyc"
```

3. Sanity-check the contents list:

```bash
unzip -l alchemist-playbook.skill
```

   Expect: `SKILL.md`, 10 files under `references/`, `scripts/estimate.py`,
   8 files under `assets/templates/`. No `__pycache__`, no workspace, no
   `docs/`.

4. Commit the archive together with the source change that motivated it —
   one commit, one logical change. If the change is user-visible, add a
   `CHANGELOG.md` entry and bump `version` in **both**
   `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json` in the same
   commit (`tests/test_doc_consistency.py` enforces that they match).

## What is deliberately NOT packaged

`alchemist-playbook-workspace/` (eval harness output), `docs/` (repo
documentation), `AGENTS.md`, `README.md`, `CHANGELOG.md`, `LICENSE`,
`scripts/check_docs.py`, `tests/`, the plugin manifests, and
`.github/`. The archive is the skill only.
