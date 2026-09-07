---
type: workflow
title: Run the with/without-skill evals
status: active
updated: 2026-09-07
---

# Workflow: run evals

The workspace measures the skill the same way the skill tells users to
measure training changes: define the success metric first, change one thing
(skill installed vs not), grade against fixed expectations.

## Layout

```
alchemist-playbook-workspace/
├── evals/evals.json                     # prompts, expected_output, optional category
├── evals/fixtures/                      # synthetic inputs copied into each test workspace
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
   capability that claims user-visible value should have an eval. Training
   cases test evidence and diagnosis; behavior cases test scope, completion,
   and reporting. Grade requested content and actions, not headings alone.
2. **Run both arms** in a fresh agent environment per arm: one with
   `alchemist-playbook.skill` installed, one without. Pass the prompt and
   raw fixtures to the performing agent, not `expected_output` or prior
   conclusions. Paths in `files` are repository-relative. Copy each fixture
   into a disposable workspace per arm so edits cannot leak between runs.
   Save the raw answer as `outputs/answer.md` and preserve tool traces and
   before/after files for action-based cases.
   For routing-boundary cases, allow normal skill discovery; do not force
   the skill to load. Record whether it was selected.

   Record the exact model, reasoning settings, available tools, skill
   revision/archive hash, prompt, and execution date. Keep these comparable
   between arms. Delegation or API-backed runs require a host that supports
   and authorizes them; their absence does not block repository edits.
3. **Grade** each answer against the expectations into `grading.json`
   (per-expectation `passed` + `evidence` quote, plus a `summary` with
   `pass_rate`). Grade the *without* arm just as strictly — the delta is the
   point.
4. **Summarize** the iteration in `benchmark.json`/`benchmark.md` and render
   `review.html` (the repository's existing report format). Distinguish
   parsed fixtures and reviewer spot checks from fresh agent executions.
5. **Feed back**: every failed expectation in the *with* arm is a skill bug —
   route the fix through [add-source.md](add-source.md) or a reference edit,
   then re-run in `iteration-N+1`. Never edit a past iteration's results.
6. **Sync the report**: update the eval section of the DATA block in
   [docs/index.html](../index.html) with the new iteration's numbers.

## Recorded results and pending validation

The [iteration-1 benchmark](../../alchemist-playbook-workspace/iteration-1/benchmark.md)
contains the historical training-case measurements. Those results predate
the instruction audit and do not establish performance of the revised skill
or a current agent model. Keep that record unchanged.

Cases with `category: behavior` in
[evals.json](../../alchemist-playbook-workspace/evals/evals.json) cover short
answers, incomplete logs, authorized edits and verification, existing
trackers, unavailable execution, task steering, inference-only routing, and
workflow-source evidence boundaries. They are **defined, not executed** in
fresh with/without-skill agent environments. Do not combine them with the
historical pass rate. A local unit-test pass is not a behavioral eval result.
