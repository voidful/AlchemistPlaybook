"""assets/templates: every shipped config parses; anchor values stay
single-sourced from estimate.py's LADDER and the consensus baseline."""
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = next(d for d in (ROOT / "skills" / "alchemist-playbook",
                             ROOT / "alchemist-playbook") if d.is_dir())
TEMPLATES = SKILL_DIR / "assets" / "templates"

spec = importlib.util.spec_from_file_location(
    "estimate", SKILL_DIR / "scripts" / "estimate.py")
estimate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimate)

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

SCI = re.compile(r"\d+(?:\.\d+)?e-\d+")


def ladder_band(stage, params=8e9):
    """[min, max] of the LR-like numbers in the matching LADDER row."""
    for lim, lr, _ in estimate.LADDER[stage]:
        if params <= lim:
            nums = [float(m) for m in SCI.findall(lr)]
            return min(nums), max(nums)
    raise AssertionError(f"no {stage} band")


class TestDeepSpeedJson(unittest.TestCase):
    def test_zero2_parses_with_stage_2(self):
        d = json.loads((TEMPLATES / "deepspeed_zero2.json").read_text())
        self.assertEqual(d["zero_optimization"]["stage"], 2)

    def test_zero3_offload_parses_with_stage_3(self):
        d = json.loads((TEMPLATES / "deepspeed_zero3_offload.json").read_text())
        self.assertEqual(d["zero_optimization"]["stage"], 3)


@unittest.skipUnless(HAVE_YAML, "pyyaml not installed")
class TestYamlTemplates(unittest.TestCase):
    def load(self, name):
        return yaml.safe_load((TEMPLATES / name).read_text())

    def test_all_yaml_templates_parse(self):
        for p in sorted(TEMPLATES.glob("*.yaml")):
            with self.subTest(p.name):
                self.assertIsInstance(yaml.safe_load(p.read_text()), dict)

    def test_sft_lr_within_ladder_band(self):
        lo, hi = ladder_band("sft")
        lr = float(self.load("sft_full.yaml")["learning_rate"])
        self.assertTrue(lo <= lr <= hi, f"sft lr {lr} outside [{lo}, {hi}]")

    def test_dpo_lr_within_ladder_band_and_standard_beta(self):
        cfg = self.load("dpo_full.yaml")
        lo, hi = ladder_band("dpo")
        lr = float(cfg["learning_rate"])
        self.assertTrue(lo <= lr <= hi, f"dpo lr {lr} outside [{lo}, {hi}]")
        # standard-DPO beta; the length-normalized beta-5 variant must stay
        # in comments only (mixing the two is the classic DPO footgun)
        self.assertEqual(float(cfg["beta"]), 0.01)

    def test_qlora_lr_within_ladder_band(self):
        lo, hi = ladder_band("lora")
        lr = float(self.load("qlora_sft.yaml")["learning_rate"])
        self.assertTrue(lo <= lr <= hi, f"lora lr {lr} outside [{lo}, {hi}]")

    def test_pretrain_consensus_baseline(self):
        cfg = self.load("pretrain_1b.yaml")
        opt = cfg["optimizer"]
        self.assertEqual(opt["betas"], [0.9, 0.95])
        self.assertEqual(float(opt["eps"]), 1e-8)
        self.assertEqual(float(opt["weight_decay"]), 0.1)
        self.assertEqual(float(opt["grad_clip_norm"]), 1.0)
        lo, hi = ladder_band("pretrain", params=1.5e9)
        lr = float(opt["peak_lr"])
        self.assertTrue(lo <= lr <= hi, f"pretrain lr {lr} outside [{lo}, {hi}]")


class TestDashboardTemplate(unittest.TestCase):
    def setUp(self):
        self.html = (TEMPLATES / "experiment_dashboard.html").read_text()

    def test_has_data_block(self):
        self.assertIn("const EXPERIMENT", self.html)

    def test_self_contained(self):
        # must render offline: no external scripts, stylesheets, or imports
        self.assertNotIn("<script src=", self.html)
        self.assertNotIn("@import", self.html)
        for m in re.finditer(r'<link[^>]+rel="stylesheet"[^>]*>', self.html):
            self.assertNotIn("http", m.group(0))


if __name__ == "__main__":
    unittest.main()
