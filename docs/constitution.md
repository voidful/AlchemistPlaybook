---
type: constitution
title: Project constitution
status: active
updated: 2026-08-19
---

# Constitution

## What this project is

AlchemistPlaybook develops and maintains **alchemist-playbook**, an
evidence-based training-recipe skill (煉丹調參) for AI coding agents. The
skill turns published training runs — LLaMA, OLMo, DeepSeek-V3, SmolLM2,
Kimi K2, GLM-5, Zephyr, Tulu 3, QLoRA, Whisper, OWSM, and dozens more — into a
diagnostic workflow for pretraining, SFT, DPO/RLHF/RLVR, PEFT, and speech
training. Its core bet: **every concrete number must trace to a published
run, an official config, or an explicitly labeled heuristic.**

### Original charter

Imported from the founding Claude Cowork project instruction (2026-06):

> 我想要寫一個煉丹調參skill。
> 幫我仔細研究有發佈過訓練經驗和recipe的所有論文，整理出這個skill。

("I want to write a training-tuning skill. Carefully study all papers that
published training experience and recipes, and distill them into this skill.")

## Governance invariants

Adapted from the [living-docs](https://github.com/ejklock/living-docs-skill)
discipline. Every structural change to this repo must preserve all six:

1. **Docs-first.** Author knowledge in this repo before publishing it
   anywhere else. The packaged `.skill` archive and the GitHub Pages site are
   *outputs* of the repo, never inputs.
2. **One home per fact.** Each fact lives in exactly one file. Training
   knowledge lives in the skill (`SKILL.md` + `references/`); repo knowledge
   lives in `docs/`. If two files state the same number, one of them is a bug.
3. **Indexed or it doesn't exist.** Every doc in `docs/` is reachable from
   [docs/index.md](index.md); every reference file is reachable from the
   SKILL.md routing table; every source is listed in `references/sources.md`.
4. **Supersede, never rewrite.** Decision records in `docs/decisions/` are
   append-only. A reversed decision gets a new record marked
   `supersedes: NNNN`, and the old record's status becomes `superseded-by:
   NNNN`. (Skill *content* may be corrected in place — see invariant 6 — but
   the decision history may not.)
5. **No structural change without its doc.** Adding a reference file, source,
   template, workflow, or eval requires updating the affected docs and the
   routing/index tables in the same commit. `scripts/check_docs.py` enforces
   the mechanical part.
6. **Evidence-marked, or labeled heuristic.** The skill's own honesty rule is
   also a repo governance rule: every value carries `[config]`, `[paper]`,
   `[reported]`, or `[heuristic]` (definitions in
   [glossary.md](glossary.md)). A number without a marking cannot merge.

## Core data model

```
Source (paper / official config / model card)
  └─ appears once in references/sources.md with a verification level
       └─ cited by ≥1 reference file (references/*.md)
            └─ routed to by the SKILL.md routing table
                 └─ exercised by evals (alchemist-playbook-workspace/)
                      └─ packaged into alchemist-playbook.skill
```

A change flows the same direction: new source → sources.md → reference
file(s) → (if scope changed) SKILL.md routing/description → repackage →
eval → docs. See [workflows/add-source.md](workflows/add-source.md).

## Change sequence

Constitution (this file) → decision record → implementation → conventions/
workflow updates → index. Big direction changes get a decision record first;
routine source additions follow the add-source workflow directly.
