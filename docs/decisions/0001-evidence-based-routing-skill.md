---
type: decision
title: Evidence-based skill with routing-table architecture and verification levels
status: active
decided: 2026-06-12
updated: 2026-08-19
---

# 0001 — Evidence-based routing skill

*Reconstructed 2026-08-19 from commit `0581fb0` ("Initial alchemist playbook
skill") and the shipped design; recorded here so future changes have a
baseline to supersede.*

## Context

Training-run tuning advice from LLMs tends to be folklore: plausible numbers
with no provenance, one-size answers across stages that differ by three
orders of magnitude in learning rate.

## Decision

1. **Every concrete value traces to a source** — an official config
   (`[config]`), a paper (`[paper]`), a secondary report (`[reported]`) — or
   is explicitly labeled `[heuristic]`. The bibliography with per-source
   verification status is a first-class reference file (`sources.md`).
2. **Thin SKILL.md + routed reference files.** SKILL.md holds only the
   trigger, operating principles, consensus baseline, and a routing table;
   deep evidence lives in one topic file each. This keeps the always-loaded
   surface small while the description stays under the 1024-character
   packaging cap.
3. **Fixed answer format** (Diagnosis / Recipe / Evidence / Verify) and a
   calculator script (`estimate.py`) instead of mental arithmetic.

## Consequences

- Adding knowledge means adding *sources*, not opinions — the add-source
  workflow exists because of this decision.
- The 1024-char description cap is a recurring constraint (trimmed in
  `15b65bb` and `7986b27`).
- Sources without training evidence must be explicitly excluded rather than
  silently ignored ("Reviewed and excluded" section).
