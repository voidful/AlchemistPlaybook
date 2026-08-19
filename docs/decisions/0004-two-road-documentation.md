---
type: decision
title: "Two-road documentation: interactive HTML for humans, indexed markdown for agents"
status: active
decided: 2026-08-19
updated: 2026-08-19
---

# 0004 — Two-road documentation

## Context

Repo documentation was a 32-line README plus an AGENTS.md containing only
the founding one-liner. The skill's *content* had strong governance
(evidence marking, sources index) but the *repo* had none: no index, no
recorded decisions, no change workflows, and the skill's own
HTML-artifact-plus-Pages discipline (decision 0003) was preached but not
practiced on this repo itself.

## Decision

Adopt a two-road documentation system modeled on
[living-docs-skill](https://github.com/ejklock/living-docs-skill) (Cyrille
Martraire's living documentation, Nygard ADRs):

1. **Agent road** — indexed markdown under `docs/` with typed frontmatter:
   constitution (six governance invariants adapted from living-docs' five,
   plus evidence marking), architecture, conventions, glossary, workflows,
   and this append-only decisions directory. Spine: `docs/index.md`;
   contract: `AGENTS.md`.
2. **Human road** — `docs/index.html`, a self-contained interactive report
   (stat tiles, routing explorer, filterable bibliography, eval-result
   chart, LR-ladder chart, decision timeline), publishable via GitHub Pages
   from `/docs` — eating the skill's own dog food per decision 0003. Its
   visible prose (and README.md) follows ASD-STE100 Simplified Technical
   English principles: short sentences, one instruction per sentence,
   active voice (see [conventions](../conventions.md)).
3. **Deterministic checking** — `scripts/check_docs.py` (the lightweight
   analog of living-docs' Rust CLI `check`): index reachability, routing
   sync, link integrity, frontmatter typing, description cap, archive
   freshness.

Deliberately *not* adopted from living-docs: the Rust CLI, OKF bundle
format, SQLite search, HMAC sealing — overkill at this repo's size; the
invariants matter, the machinery doesn't.

## Consequences

- Structural changes now carry a doc obligation, enforced mechanically.
- Repo facts have single homes; README and AGENTS.md became pointers.
- The DATA block in `index.html` must be kept in sync with headline facts —
  a new, small maintenance cost accepted in exchange for a shareable,
  always-current human report.
