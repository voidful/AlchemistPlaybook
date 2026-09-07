---
type: workflow
title: Repackage the skill archive
status: active
updated: 2026-09-07
---

# Workflow: release / repackage

`alchemist-playbook.skill` is a **zip archive of the `alchemist-playbook/`
directory** (extension renamed). It is a build output: never edit it
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
   archive-freshness check. Before rebuilding, only the expected freshness
   failure is acceptable; rerun the checker after step 2. The tests lock
   `references/text-llm-baselines.md` to `estimate.py` and validate templates.

2. Rebuild from the repo root. Write a fresh archive, then replace the old
   output; no pre-delete is needed. Exclude Python bytecode:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

source = Path("alchemist-playbook")
target = Path("alchemist-playbook.skill")
staging = target.with_suffix(".skill.tmp")
with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            archive.write(path, path.as_posix())
staging.replace(target)
PY
```

3. Sanity-check the contents list:

```bash
unzip -l alchemist-playbook.skill
```

   Expect `SKILL.md`, all reference files in its routing table,
   `scripts/estimate.py`, and the files under `assets/templates/`.
   No `__pycache__`, workspace, or `docs/`. Run `python3 scripts/check_docs.py`
   again to verify exact source/archive agreement. Repeat tests only if
   source changes or a failure warrant it.

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
