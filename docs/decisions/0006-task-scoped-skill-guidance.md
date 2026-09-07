---
type: decision
title: Task-scoped guidance and progressive disclosure
status: active
decided: 2026-09-07
updated: 2026-09-07
supersedes: ["0001", "0003"]
---

# 0006 — Task-scoped guidance and progressive disclosure

## Context

The skill mixed training evidence with unconditional workflow instructions.
Its intake could block a provisional diagnosis. Its fixed answer format
overrode requests for a short answer or a config file. Experiment tracking
required HTML and commits even when a user already had a suitable workflow.
The text-LLM baseline also loaded for unrelated speech and reporting tasks.

The workflow sources listed in
[sources.md](../../alchemist-playbook/references/sources.md#workflow--tracking)
include Eric Provencher's *Rethinking skills and prompts for GPT-6 Astra*
and OpenAI's *Using GPT-6 Astra* guide, both read on 2026-09-07. They motivate
an audit of discovery, context cost, instruction conflicts, task completion,
and verification scope. They provide no training-recipe evidence.

## Decision

- Keep the trigger focused on training tasks. Move the text-LLM baseline
  and LR ladder to `references/text-llm-baselines.md`; route to them only
  when applicable. Preserve the calculator's existing numerical checks
  against that new home.
- Treat intake as information to recover from available context. Ask when
  a missing fact changes the next decision; otherwise state assumptions
  and continue useful work. Incomplete logs allow provisional hypotheses,
  with the missing discriminating observation identified.
- Follow the user's task and requested format. Diagnosis, recipe, evidence,
  and verification remain useful answer content, not mandatory headings.
- Carry authorized local work through edits and relevant verification.
  A request to prepare a config does not itself authorize a training job
  or public deployment. Existing authorization remains valid; explain the
  exact instruction if a skill requirement actually blocks work.
- Use the user's existing experiment records. HTML, git checkpoints, and
  Pages remain available defaults for a suitable reporting task. Preserve
  attribution, raw evidence, preregistered criteria, and reproducibility
  regardless of the presentation format.
- Complete required repository checks. Add or repeat other checks only
  for changed behavior or unresolved evidence. Agent delegation depends on
  the host and task; this skill does not mandate it or select a model.

This partially supersedes [0001](0001-evidence-based-routing-skill.md)
(baseline placement and fixed output format) and
[0003](0003-html-artifact-experiment-tracking.md) (unconditional presentation,
commit, and publishing requirements). Their remaining provisions still
apply. Historical records remain unchanged; the index identifies the
superseded scope.

## Consequences and validation

The skill remains a training advisor usable across agent hosts. This is an
instruction audit, not an API migration or a new Astra training recipe.
The source bibliography distinguishes workflow guidance from training
evidence. Repository indexing, packaging, version sync, and scientific
integrity remain required.

Behavioral regression cases cover short answers, missing logs, local config
edits, existing trackers, blocked calculations, task steering, and routing
boundaries. Definitions are not results. The eval workflow records what
ran, on which model and skill revision, and keeps historical measurements
separate from cases awaiting execution.
