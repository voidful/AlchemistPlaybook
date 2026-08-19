---
type: decision
title: Experiment tracking = HTML artifacts + git commits + GitHub Pages
status: active
decided: 2026-07-11
updated: 2026-08-19
---

# 0003 — HTML-artifact experiment tracking

*Reconstructed 2026-08-19 from commits `7986b27` ("Add experiment-tracking
workflow") and `20996d4` ("Raise report standard to FSQ-campaign exemplar and
beyond").*

## Context

The skill prescribed "one change per rerun" (principle 6) but gave users no
mechanism to make that auditable. Markdown walls of run notes don't get
read; screenshots of dashboards go stale.

## Decision

For *tracked experiments* (not one-off advisory answers):

1. Every experiment change gets a **self-contained HTML artifact** (plan,
   config diff, run report, comparison) — no external requests, data in one
   JSON const.
2. Every artifact state change gets a **git commit** — the commit log
   becomes the ablation log, so principle 6 audits itself.
3. The artifact tree publishes via **GitHub Pages** so every run's current
   state is one shareable URL (with an explicit privacy caveat: Pages is
   public even on private repos).
4. A report-detail standard (`experiment-tracking.md` §4) pegged to a named
   exemplar (the FSQ-campaign dashboard) and extended past it (ground-truth
   vs inference sample gallery), with a starter template
   (`assets/templates/experiment_dashboard.html`).

Elevated to operating principle 8 in SKILL.md.

## Consequences

- The dashboard template and its "edit only the DATA block" convention exist.
- Workflow sources entered the bibliography with an explicit "not training
  evidence" firewall so they can never be cited for a hyperparameter.
- This repo later applied the same discipline to itself — see
  [0004](0004-two-road-documentation.md).
