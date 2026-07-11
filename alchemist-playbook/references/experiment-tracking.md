# Experiment tracking: HTML artifacts + git + GitHub Pages

How to *record and present* an experiment. Complements principle 6 (one
change per rerun — this file makes that auditable) and `evaluation.md`
(what to measure). The HTML-artifact discipline follows Thariq Shihipar,
"Using Claude Code: the unreasonable effectiveness of HTML" (claude.com
blog, 2026-05) `[reported]` — a workflow practice, not training evidence.
Git/Pages mechanics are standard GitHub features.

## 1. The rule

**Every change gets an HTML artifact, and every artifact gets a commit.**
When this skill is active and the user is planning, running, or debugging
experiments:

- Render plans, config changes, run results, and comparisons as
  **self-contained HTML files**, not Markdown walls. Markdown plans past
  ~100 lines stop being read; HTML gives density (tables, inline-SVG
  curves, color-coded config diffs, collapsible detail) and turns every
  report into a shareable URL.
- Put the experiment in a **git repo from day one** and commit at each
  state change (plan approved / run launched / run finished / decision
  made). One change per rerun (principle 6) then becomes automatic
  attribution: the commit log *is* the ablation log.
- Recommend the user enable **GitHub Pages** so the artifact tree is
  browsable at a URL — teammates read a dashboard, not a diff (privacy
  caveat in §6 applies before anything is published).

## 2. Repo layout

```
my-experiment/
  docs/                      # GitHub Pages root (Settings → Pages → main, /docs)
    index.html               # dashboard: run table, status, links (see template)
    plan.html                # hypotheses, ablation matrix, success metrics
    runs/
      2026-07-11-lr-3e-4/
        report.html          # per-run report: config diff, curves, verdict
    compare/
      lr-sweep.html          # side-by-side grids across runs
    configs/                 # exact launch config per run (yaml/json)
    results/                 # raw metrics (jsonl/csv) that the HTML embeds
  README.md                  # one paragraph + link to the Pages URL
```

Configs and results live *under* `docs/` so every link in every artifact
resolves identically from `file://` and from the published Pages site —
anything outside the Pages root would 404 on the public URL.

Start from `assets/templates/experiment_dashboard.html` for `index.html`.

## 3. The artifact set

| Stage | Artifact | Must contain |
| --- | --- | --- |
| Plan (before launch) | `plan.html` | hypotheses; ablation matrix (run × knob); per-run success metric **and observation window** (principle 6); baseline + rollback plan |
| Launch (per run) | commit only | exact config committed under `docs/configs/`; run row added to dashboard as `running` |
| Result (per run) | `runs/<id>/report.html` | config diff old→new; loss/eval curves as inline SVG; the pre-registered metric vs. its window; verdict (`kept` / `rolled-back`) and why |
| Comparison | `compare/<topic>.html` | side-by-side grid of the candidate runs — same axes, same metric, one screenful |
| Always | `index.html` | run table: id, one-line change, status, headline metric, links to report/config; newest first |

Update `plan.html` only by appending an amendments section — never rewrite
the pre-registered plan after results exist (that is how post-hoc
rationalization sneaks in).

## 4. Commit discipline

- Commit points: plan approved → run launched → run finished → decision.
- Message convention: `run <id>: <the one change> — <headline result>`
  (e.g., `run 07-11a: beta2 0.999→0.95 — spikes gone, loss -0.02`).
  The log then reads as the experiment history without opening a file.
- Commit the *inputs* (config, data-mix manifest, code SHA of the trainer)
  with the launch, and the *outputs* (metrics file + report.html) with the
  result — a run must be reconstructable from its two commits.
- Never force-push over published reports; corrections are new commits.

## 5. HTML artifact rules

- **One file, no build step.** Inline CSS/JS; embed run data as a JSON
  `const` at the top of the file so updating = editing one block. Must
  render from `file://` and from Pages alike.
- **No external CDNs required.** Prefer hand-rolled inline SVG for curves
  (a loss curve is a `<polyline>`); the file must work offline and not
  leak run metadata to third-party hosts.
- Color-code semantics, not decoration: green/red for kept/rolled-back,
  severity colors for diagnostics — the same encoding across all artifacts.
- Downsample embedded curves (~200–500 points per series) so artifacts
  stay a few hundred KB; link raw jsonl in `docs/results/` for full
  fidelity.
- Prompts that work well for generating these (adapted from the source
  article): "lay out the N candidate recipes as a single HTML grid so I
  can compare side by side"; "make a run report: render the config diff
  color-coded, plot the curves as inline SVG, and state the verdict
  against the pre-registered metric".

## 6. GitHub Pages setup

**Privacy caveat (state it BEFORE recommending Pages):** a Pages site is
publicly reachable on the internet even when the repo is private
(access-controlled Pages is an Enterprise feature) — and publishing Pages
from a private repo at all requires a paid plan (Pro/Team/Enterprise); on
GitHub Free the repo itself must be public. If configs, data mixes, or
eval numbers are proprietary, keep Pages off and open `docs/index.html`
locally or host it on an internal static server — the artifact + commit
discipline works identically without the public URL.

Once the user confirms the artifacts are shareable:

```bash
git init -b main && git add -A && git commit -m "plan: <experiment name>"
gh repo create <name> --private --source . --push
# then: repo Settings → Pages → Source: Deploy from a branch → main, /docs
```

## 7. What this does NOT replace

W&B/TensorBoard-style loggers remain the right tool for live,
high-frequency health monitoring (step-level loss, grad-norm — the
health-monitoring tier in `evaluation.md`). The HTML+git layer sits above them:
it captures the *decisions* — what was changed, why, what the
pre-registered bar was, and whether it was met — in a form teammates and
future runs can actually read. Link out to the logger's run page from
each report rather than duplicating it.
