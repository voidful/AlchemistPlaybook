"""Locks between documents that must not drift:

1. text-llm-baselines.md peak-LR table and cross-stage ladder ↔ estimate.py LADDER
   (this exact drift happened once: EvoLM widened the documented ≤2B band to
   2e-4 while estimate.py still said 4e-4).
2. Consensus-baseline knobs quoted by `estimate.py lr --stage pretrain`
   appear in the baseline reference table.
3. Every `<file>.md §N` citation in the skill resolves to a real numbered
   `## N.` heading in the target reference.
4. Release version is identical in .claude-plugin/plugin.json,
   .codex-plugin/plugin.json, and the top CHANGELOG.md entry.

Comparisons extract scientific-notation numbers rather than exact prose, so
rewording survives but a changed number on one side fails loudly.
"""
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = next(d for d in (ROOT / "skills" / "alchemist-playbook",
                             ROOT / "alchemist-playbook") if d.is_dir())
SKILL_MD = (SKILL_DIR / "SKILL.md").read_text()
REFS = SKILL_DIR / "references"
BASELINES = (REFS / "text-llm-baselines.md").read_text()

spec = importlib.util.spec_from_file_location(
    "estimate", SKILL_DIR / "scripts" / "estimate.py")
estimate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimate)

SCI = re.compile(r"\d+(?:\.\d+)?e-\d+")


def sci_set(text):
    return {float(m) for m in SCI.findall(text)}


def ladder_union(stage):
    return {n for _, lr, _ in estimate.LADDER[stage] for n in sci_set(lr)}


def ladder_row(stage, params):
    for lim, lr, _ in estimate.LADDER[stage]:
        if params <= lim:
            return lr
    raise AssertionError(f"no {stage} band for {params}")


class TestPretrainLrTable(unittest.TestCase):
    """The routed baseline's peak pretraining LR table ↔ LADDER rows."""

    # size label in the baseline table → a params probe inside that band
    BANDS = [("100M–2B", 1e9), ("7–13B", 8e9), ("30–70B", 5e10), ("~405B", 4e11)]

    def peak_lr_cell(self, label):
        for line in BASELINES.splitlines():
            if line.startswith(f"| {label} "):
                return line.split("|")[2]  # the 'Peak LR' column
        raise AssertionError(f"baseline peak-LR table has no '{label}' row")

    def test_each_band_subset_of_ladder_row(self):
        for label, probe in self.BANDS:
            with self.subTest(label):
                skill_nums = sci_set(self.peak_lr_cell(label))
                ladder_nums = sci_set(ladder_row("pretrain", probe))
                self.assertTrue(
                    skill_nums and skill_nums <= ladder_nums,
                    f"'{label}': baseline {sorted(skill_nums)} not covered by "
                    f"estimate.py LADDER {sorted(ladder_nums)}")


class TestCrossStageLadder(unittest.TestCase):
    """The baseline's cross-stage LR ladder code block ↔ LADDER stages."""

    STAGE_PREFIX = [("pretrain", "pretrain"), ("full SFT", "sft"),
                    ("DPO", "dpo"), ("PPO/RLVR", "rl"), ("LoRA/QLoRA", "lora")]

    def test_block_values_subset_of_ladder(self):
        m = re.search(r"## \d+\. Cross-stage LR ladder.*?```\n(.*?)```", BASELINES, re.S)
        self.assertIsNotNone(m, "cross-stage ladder code block missing")
        seen = set()
        for line in m.group(1).splitlines():
            for prefix, stage in self.STAGE_PREFIX:
                if line.startswith(prefix):
                    seen.add(stage)
                    with self.subTest(stage):
                        nums = sci_set(line)
                        self.assertTrue(
                            nums and nums <= ladder_union(stage),
                            f"{stage}: baseline {sorted(nums)} not covered by "
                            f"LADDER {sorted(ladder_union(stage))}")
                    break
        self.assertEqual(seen, {stage for _, stage in self.STAGE_PREFIX},
                         "baseline ladder lost a training stage")


class TestConsensusPairing(unittest.TestCase):
    def test_cli_pairing_knobs_appear_in_baseline_reference(self):
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, str(SKILL_DIR / "scripts" / "estimate.py"),
             "lr", "--params", "8e9", "--stage", "pretrain"],
            capture_output=True, text=True).stdout
        self.assertIn("Pair with:", out)
        for token in ("0.9, 0.95", "eps 1e-8", "wd 0.1", "clip 1.0"):
            self.assertIn(token, out)
        for token in ("0.9, 0.95", "1e-8", "| Weight decay | 0.1 |",
                      "| Grad clip (global norm) | 1.0 |"):
            self.assertIn(token, BASELINES, f"baseline table lost '{token}'")


class TestSectionCitations(unittest.TestCase):
    """Every '<file>.md §N[.M][–K]' citation resolves in its target file."""

    CITE = re.compile(r"([A-Za-z0-9_.-]+\.md)\s*§(\d+)(?:\.\d+)?"
                      r"(?:\s*[–-]\s*(\d+)(?:\.\d+)?)?")

    def headings(self, path):
        return {int(m) for m in re.findall(r"^##\s*(\d+)[.\s]",
                                           path.read_text(), re.M)}

    def test_citations_resolve(self):
        sources = [(SKILL_DIR / "SKILL.md", SKILL_MD)]
        sources += [(p, p.read_text()) for p in sorted(REFS.glob("*.md"))]
        seen_any = False
        for src, text in sources:
            for fname, start, end in self.CITE.findall(text):
                target = REFS / Path(fname).name
                if not target.exists():
                    continue  # link-level checks live in scripts/check_docs.py
                seen_any = True
                nums = self.headings(target)
                with self.subTest(f"{src.name} → {fname} §{start}"):
                    self.assertTrue(
                        nums, f"{fname} is cited with § but has no "
                              f"numbered '## N.' headings")
                    cited = {int(start)} | ({int(end)} if end else set())
                    self.assertTrue(
                        cited <= nums,
                        f"{src.name} cites {fname} §{start}"
                        f"{'–' + end if end else ''} but {fname} only has "
                        f"§{sorted(nums)}")
        self.assertTrue(seen_any, "citation scan matched nothing — regex broken?")


class TestVersionSync(unittest.TestCase):
    def test_manifests_and_changelog_agree(self):
        claude = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
        codex = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
        self.assertEqual(claude["version"], codex["version"],
                         "plugin manifests disagree on version")
        top = next(l for l in (ROOT / "CHANGELOG.md").read_text().splitlines()
                   if l.startswith("## "))
        self.assertIn(claude["version"], top,
                      f"top CHANGELOG entry '{top}' ≠ manifest "
                      f"version {claude['version']}")


if __name__ == "__main__":
    unittest.main()
