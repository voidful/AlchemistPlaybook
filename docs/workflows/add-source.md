---
type: workflow
title: Add a new source / recipe to the skill
status: active
updated: 2026-08-19
---

# Workflow: add a source

The most common change in this repo (see the commit log: nearly every commit
is "Add X"). Follow the data-model flow direction (constitution → Core data
model).

## Steps

1. **Verify the evidence.** Read the paper *and* hunt for official configs /
   model cards / training scripts. Decide the verification level per value:
   `[config]` beats `[paper]` beats `[reported]`. If the paper states no
   number for a knob, record that explicitly ("LR not stated") rather than
   guessing. If a repo diverges from its paper, note the divergence and cite
   the paper's tables (precedent: FAC-Synthesis entry).
2. **Add the bibliography row** in
   `alchemist-playbook/references/sources.md` under the right section
   (Pretraining / Post-training / Speech / Systems / Evaluation / Workflow).
   One row per source: name, arXiv ID or link, and *what this skill uses it
   for*. A source with no training evidence goes to "Reviewed and excluded"
   instead — with the reason.
3. **Integrate into reference file(s).** Add or extend the relevant
   `references/*.md` section(s), marking every value. Correct any existing
   numbers the new source contradicts — give the range and say which source
   fits which regime.
4. **Update SKILL.md only if scope changed.** New topic keywords → trigger
   `description:` (**hard cap 1024 chars** — trim, don't overflow); new
   routing target or renamed coverage → routing table row; new headline
   source → the source list in the description.
5. **Repackage** the archive: [release.md](release.md).
6. **Check + eval.** Run `python3 scripts/check_docs.py`. If the addition
   claims to improve answers on a failure mode, add an eval for it
   ([run-evals.md](run-evals.md)).
7. **Update docs.** New reference file or structural change → update
   [architecture.md](../architecture.md) and the DATA block in
   [index.html](../index.html) (source count, routing rows).
8. **Commit** everything in one commit: "Add <source>: <what it teaches>".

## Quality bar

- Never present an unverified number as verified.
- Paper-internal inconsistencies: cite table values and note the
  inconsistency (precedents: PivotRL −9.83 vs −9.48; Agents-A1 GAIA 85.4 vs
  95.1).
- Community repos without training runs don't meet the bar — exclude with a
  note (precedent: little-coder / smallcode).
