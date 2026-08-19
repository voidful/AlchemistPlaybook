---
type: workflow
title: Run the with/without-skill evals
status: active
updated: 2026-08-19
---

# Workflow: run evals

The workspace measures the skill the same way the skill tells users to
measure training changes: define the success metric first, change one thing
(skill installed vs not), grade against fixed expectations.

## Layout

```
alchemist-playbook-workspace/
├── evals/evals.json                     # eval definitions: prompt + expected_output
└── iteration-N/
    ├── eval-<id>-<name>/
    │   ├── eval_metadata.json
    │   ├── with_skill/run-M/            # answer.md + grading.json + timing.json
    │   └── without_skill/run-M/
    ├── benchmark.json / benchmark.md    # iteration summary
    └── review.html                      # human-readable iteration report
```

## Procedure

1. **Define or extend evals** in `evals/evals.json`: a realistic user prompt
   plus an `expected_output` written as gradeable expert expectations
   (specific values, named sources, structural requirements). Each new skill
   capability that claims user-visible value should have an eval.
2. **Run both arms** in a fresh agent environment per arm: one with
   `alchemist-playbook.skill` installed, one without. Save the raw answer as
   `outputs/answer.md`.
3. **Grade** each answer against the expectations into `grading.json`
   (per-expectation `passed` + `evidence` quote, plus a `summary` with
   `pass_rate`). Grade the *without* arm just as strictly — the delta is the
   point.
4. **Summarize** the iteration in `benchmark.json`/`benchmark.md` and render
   `review.html` (self-contained, following the skill's own HTML-artifact
   rules).
5. **Feed back**: every failed expectation in the *with* arm is a skill bug —
   route the fix through [add-source.md](add-source.md) or a reference edit,
   then re-run in `iteration-N+1`. Never edit a past iteration's results.
6. **Sync the report**: update the eval section of the DATA block in
   [docs/index.html](../index.html) with the new iteration's numbers.

## Current state (iteration 1, 2026-06)

3 evals (pretrain loss-spike debug, DPO config review, speech warmup
divergence): with skill **15/15** expectations passed; without skill
**11/15**. The four without-skill misses: kept ε=1e-5 against OLMo 2
evidence, no verification plan, missed the β–loss-variant coupling, and no
named speech recipe.
