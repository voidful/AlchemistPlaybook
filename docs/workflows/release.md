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

   This includes the ≤ 1024-char description check and archive-freshness
   check.

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
   one commit, one logical change.

## What is deliberately NOT packaged

`alchemist-playbook-workspace/` (eval harness output), `docs/` (repo
documentation), `AGENTS.md`, `README.md`, `scripts/check_docs.py`. The
archive is the skill only.
