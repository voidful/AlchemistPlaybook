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
  caveat in §7 applies before anything is published).

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
| Result (per run) | `runs/<id>/report.html` | config diff old→new; loss/eval curves as inline SVG; the pre-registered metric vs. its window; inference-vs-GT samples from the fixed probe set (§4.8); verdict (`kept` / `rolled-back`) and why |
| Comparison | `compare/<topic>.html` | side-by-side grid of the candidate runs — same axes, same metric, one screenful |
| Always | `index.html` | goal progress panel (§4.14: target, gap, trend ETA); run table: id, one-line change, status, headline metric, links to report/config; newest first |

Update `plan.html` only by appending an amendments section — never rewrite
the pre-registered plan after results exist (that is how post-hoc
rationalization sneaks in).

## 4. Report detail standard (meet or exceed)

Reference exemplar: the FSQ-campaign dashboard at
`eric-lam.com/vocodec/campaign` `[reported]` — a live page produced with
this workflow. Generated dashboards and run reports must match or exceed
its level of detail. Concretely, a dashboard is not done until it has:

1. **Status strip** — one row of stat tiles: progress vs. budget (steps or
   tokens, with %), latest train loss (with its components), the headline
   val metric *with the step it was measured at*, run-health stats each
   annotated with its theoretical bound or context (e.g., "entropy 10.08
   bits; single-batch ceiling 11.91, codebook log₂K = 13.64"), and
   throughput + ETA ("52.3 step/min, ~6.6 training-hours left").
2. **Captioned curves** — every chart carries a 1–2 sentence
   interpretation: what it measures, why it matters to the decision at
   hand, the expected shape ("zero until quantization starts at step 5k"),
   and bound/baseline anchors drawn or stated ("old cosine-NN design:
   7.04 bits on the whole corpus"). A chart without interpretation is
   decoration, not reporting.
3. **Eval data table** — the numbers behind the curves at a fixed cadence
   (e.g., every eval step), readable without hover: tooltips die in
   exports, prints, and screenshots.
4. **Experiment ladder** — every planned rung (from `plan.html`) with
   status glyph (✓ done / ▶ running / ○ pending) and its *pre-registered
   gate* ("paired-delta CI must exclude 0"), plus enough per-rung notes to
   reconstruct what was actually done (job ids, run dirs, review fixes).
5. **Decision ledger** — dated entries per completed eval: experiment,
   significance flag, and a decision narrative that names the numbers,
   compares against the baseline, states the next action — and carries
   honesty markers where due ("NOT a gated comparison: no live baseline,
   no CI"). This is the skill's honesty rules applied to reporting.
6. **Infra chain** — the job table (Slurm/K8s ids, step ranges, last log
   update) so a stalled or silently-dead chain is visible at a glance.
7. **Provenance footer** — page-generation timestamp, current queue state,
   sampling/downsampling notes ("curves are 25-step logs, thinned to
   ≤700 points"), and a one-line config summary (data, hardware, key
   hyperparameters).
8. **Inference-vs-GT sample gallery** — the exemplar's own noted gap; go
   beyond it. A fixed, pre-registered probe set (8–16 items, chosen in
   `plan.html` before training and including known-hard cases) rendered
   side-by-side with ground truth at every eval checkpoint: `<audio>`
   players (plus spectrograms where useful) for codec/TTS, hypothesis vs.
   reference text with the divergence noted for ASR/LLM outputs, image
   pairs for vision. For agentic-RL, a "sample" is the full trajectory:
   render the tool-call transcript as a collapsible block next to the
   expected outcome (W&B-Weave-style tracing, statically). Keep the SAME
   probe items across checkpoints so progression is directly
   audible/visible, and store them under `docs/results/samples/<step>/`.
   Never hand-pick wins after the fact — metrics say *how much* is wrong;
   samples say *what kind* of wrong remains, and only an untouched probe
   set can say it honestly.

Logger-grade affordances — the reading habits worth adopting from
TensorBoard/W&B, all implementable in static HTML:

9. **Smoothing with a raw ghost** (TensorBoard scalars): smoothed (EMA)
   curves drawn over the raw series at low opacity — never the smoothed
   line alone, because smoothing is exactly what hides the spikes and
   short regressions this skill exists to catch. Expose the smoothing
   factor as a control, defaulting modest (~0.6).
10. **Log-scale and token axes**: loss-like and LR curves get a log-y
    toggle (power-law decay reads as a line; plateaus and spikes stand
    out); when runs differ in batch size or the schedule is
    token-budgeted, the x-axis must be *tokens*, not steps — steps are
    not comparable across batch sizes (`scaling-and-batch.md` token
    accounting). State the unit on the chart.
11. **Reproducibility block** (W&B run metadata, made mandatory): each
    run records trainer git SHA (+ dirty flag), seed(s), data-manifest
    version/hash, framework + CUDA versions, and parent checkpoint
    (lineage). A report a reader cannot reproduce from is a screenshot,
    not a report.
12. **Sweep view** (TB HParams / W&B sweeps): once ≥3 runs vary the same
    knob, add a knob-value vs. headline-metric scatter with the chosen
    setting marked (log-x when the knob is an LR/β-style magnitude);
    parallel coordinates only if several knobs moved — for a
    one-change-per-rerun ladder, the scatter is the honest view.
13. **Throughput panel and staleness banner** (W&B system metrics +
    alerts, statically): a tokens/sec or step/min curve with the expected
    MFU from `scripts/estimate.py` as an anchor line, so dataloader
    stalls and slow nodes are visible; and the dashboard stamps
    `generatedAt` and computes its own age in the reader's browser,
    showing a stale-warning banner when the regeneration loop has died —
    a static page's substitute for alerts.
14. **Goal progress panel** — the release gate made visible, placed
    directly under the status strip so it is the first thing a reader
    absorbs: one card per pre-registered campaign goal (from
    `plan.html`), each showing the target ("val loss ≤ 2.35"), the
    current value with its measured step, the remaining gap in metric
    units, a progress bar from the starting value to the target, and a
    **trend ETA**: extrapolate the recent eval trend (state the method —
    e.g., linear over the last 4 evals) to "at current trend: ~step X,
    ~N hours to go", colored on-track / at-risk (ETA beyond the training
    budget) / off-track (trend flat or moving away — show that verdict,
    never a fabricated ETA) / achieved. Always carry the caveat that
    extrapolation assumes the trend holds — deceleration is the norm near
    convergence (the exemplar's own WER ladder decelerated
    −7.2 → −3.8 → −0.8 pp per 50k steps). Answering "will we make it
    within budget?" honestly, at a glance, is this panel's whole job.

The same bar applies per artifact: a run report that shows a curve but
not the pre-registered window, a comparison grid without matched axes,
an eval checkpoint with no GT-vs-output samples, or a run without its
reproducibility block, is below standard. Run reports additionally render
the FULL config with changed lines highlighted (W&B run-diff), not just
the headline change — it is the visual check that rule 6 ("one change per
rerun") actually held.

## 5. Commit discipline

- Commit points: plan approved → run launched → run finished → decision.
- Message convention: `run <id>: <the one change> — <headline result>`
  (e.g., `run 07-11a: beta2 0.999→0.95 — spikes gone, loss -0.02`).
  The log then reads as the experiment history without opening a file.
- Commit the *inputs* (config, data-mix manifest, code SHA of the trainer)
  with the launch, and the *outputs* (metrics file + report.html) with the
  result — a run must be reconstructable from its two commits.
- Never force-push over published reports; corrections are new commits.

## 6. HTML artifact rules

- **One file, no build step.** Inline CSS/JS; embed run data as a JSON
  `const` at the top of the file so updating = editing one block. Must
  render from `file://` and from Pages alike.
- **No external CDNs required.** Prefer hand-rolled inline SVG for curves
  (a loss curve is a `<polyline>`); the file must work offline and not
  leak run metadata to third-party hosts.
- Color-code semantics, not decoration: green/red for kept/rolled-back,
  severity colors for diagnostics — the same encoding across all artifacts.
- Downsample embedded curves (≤ ~700 points per series) so artifacts
  stay a few hundred KB; link raw jsonl in `docs/results/` for full
  fidelity — and say so in the provenance footer (§4.7).
- The dashboard template implements the §4 standard end-to-end (§4.1–14:
  tiles, goal progress panel with trend ETA, captioned charts with
  smoothing + log-y, eval table, ladder with gates and repro, ledger,
  sample gallery, sweep scatter, job chain, staleness banner, footer;
  agentic-RL collapsible trajectories are the one per-experiment
  extension) — extend it per experiment; do not strip sections to save
  effort.
- Prompts that work well for generating these (adapted from the source
  article): "lay out the N candidate recipes as a single HTML grid so I
  can compare side by side"; "make a run report: render the config diff
  color-coded, plot the curves as inline SVG with an interpretation
  caption each, and state the verdict against the pre-registered metric".

## 7. GitHub Pages setup

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

## 8. What this does NOT replace

W&B/TensorBoard-style loggers remain the right tool for live,
high-frequency health monitoring (step-level loss, grad-norm — the
health-monitoring tier in `evaluation.md`), and §4.9–13 deliberately
adopts only their *reading habits*, not their jobs: live streaming,
threshold alerts paged to a human, profiler traces, and embedding
projectors stay in the logger. The HTML+git layer sits above it: it
captures the *decisions* — what was changed, why, what the
pre-registered bar was, and whether it was met — in a form teammates and
future runs can actually read. Link out to the logger's run page from
each report rather than duplicating it.
