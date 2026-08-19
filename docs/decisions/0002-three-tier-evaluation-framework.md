---
type: decision
title: Three-tier per-stage evaluation framework
status: active
decided: 2026-06-15
updated: 2026-08-19
---

# 0002 — Three-tier evaluation framework

*Reconstructed 2026-08-19 from commit `956de83` ("Add LFM2 lessons and a
per-stage evaluation/monitoring framework").*

## Context

The skill initially answered "what LR should I use" well but had no unified
answer for "what should I watch and which benchmarks should I run". Users
conflate run-health monitoring with capability evaluation with release
readiness — three questions with different cadences and different failure
modes (a healthy run can silently regress a capability; public benchmarks
leak).

## Decision

Adopt a three-tier framework as its own reference file
(`references/evaluation.md`), labeled `[heuristic]` (a practitioner
synthesis consistent with OLMo / Llama 3 / Tulu 3 practice):

1. **Health monitoring** (step/hour): loss, grad-norm, NaN, MoE balance —
   "is the run alive".
2. **Capability evaluation** (per N tokens): frozen validation slices +
   downstream probes, including a mid-training retention suite — "is it
   getting stronger without forgetting".
3. **Release gate**: a private holdout — "can it ship".

Elevate "monitoring is not evaluation" to operating principle 7 in SKILL.md.

## Consequences

- Every stage-specific reference now points to its tier-2 metrics.
- The benchmark catalog entered `sources.md` (benchmarks are cited like any
  other source).
- The eval workspace grades answers on whether they include a verification
  plan — a direct descendant of this framing.
