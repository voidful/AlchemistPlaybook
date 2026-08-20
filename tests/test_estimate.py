"""estimate.py: pure math, band selection, clamps, and CLI contract.

Assertions derive expected values from the module's own constants (LADDER,
GPU_PEAK_TFLOPS) or from arithmetic, never from hard-coded knowledge
literals — so a knowledge correction updates one place and the suite still
passes (SKILL.md alignment is locked separately in test_doc_consistency.py).
"""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = next(d for d in (ROOT / "skills" / "alchemist-playbook",
                             ROOT / "alchemist-playbook") if d.is_dir())
ESTIMATE = SKILL_DIR / "scripts" / "estimate.py"

spec = importlib.util.spec_from_file_location("estimate", ESTIMATE)
estimate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimate)


def run_cli(*args):
    r = subprocess.run([sys.executable, str(ESTIMATE), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def ladder_lookup(stage, params):
    for lim, lr, src in estimate.LADDER[stage]:
        if params <= lim:
            return lr, src
    raise AssertionError(f"no {stage} band covers {params}")


class TestFmt(unittest.TestCase):
    def test_unit_thresholds(self):
        self.assertEqual(estimate.fmt(999), "999")
        self.assertEqual(estimate.fmt(1e3), "1K")
        self.assertEqual(estimate.fmt(1.5e6), "1.5M")
        self.assertEqual(estimate.fmt(2e9), "2B")
        self.assertEqual(estimate.fmt(2e12), "2T")

    def test_negative(self):
        self.assertEqual(estimate.fmt(-3e9), "-3B")


class TestLadderStructure(unittest.TestCase):
    def test_stages(self):
        self.assertEqual(set(estimate.LADDER),
                         {"pretrain", "sft", "dpo", "rl", "lora"})

    def test_rows_are_triples_with_ascending_limits(self):
        for stage, rows in estimate.LADDER.items():
            limits = []
            for row in rows:
                self.assertEqual(len(row), 3, f"{stage}: {row}")
                lim, lr, src = row
                self.assertIsInstance(lim, float)
                self.assertTrue(lr and src, f"{stage}: empty lr/source")
                limits.append(lim)
            self.assertEqual(limits, sorted(limits), f"{stage} limits not ascending")
            # catch-all row so any --params value resolves
            self.assertGreaterEqual(limits[-1], 1e15, f"{stage} has no catch-all band")

    def test_band_selection_boundary(self):
        first_lim = estimate.LADDER["pretrain"][0][0]
        at_limit, _ = ladder_lookup("pretrain", first_lim)
        above_limit, _ = ladder_lookup("pretrain", first_lim * 1.01)
        self.assertEqual(at_limit, estimate.LADDER["pretrain"][0][1])
        self.assertEqual(above_limit, estimate.LADDER["pretrain"][1][1])
        self.assertNotEqual(at_limit, above_limit)


class TestFlopsCli(unittest.TestCase):
    def test_6nd_and_gpu_hours(self):
        rc, out, _ = run_cli("flops", "--params", "7e9", "--tokens", "2e12",
                             "--gpus", "64", "--gpu", "h100", "--mfu", "0.4")
        self.assertEqual(rc, 0)
        self.assertIn("8.400e+22", out)  # 6 * 7e9 * 2e12
        peak = estimate.GPU_PEAK_TFLOPS["h100"] * 1e12
        gpu_hours = 6.0 * 7e9 * 2e12 / (peak * 0.4) / 3600
        self.assertIn(f"{gpu_hours:,.0f}", out)
        self.assertIn("Wall-clock on 64 GPUs", out)

    def test_cost_labeled_as_assumption(self):
        rc, out, _ = run_cli("flops", "--params", "1e9", "--tokens", "1e10",
                             "--cost-per-hour", "2.5")
        self.assertEqual(rc, 0)
        self.assertIn("ASSUMED PRICE", out)

    def test_every_gpu_key_resolves(self):
        for gpu in estimate.GPU_PEAK_TFLOPS:
            rc, out, _ = run_cli("flops", "--params", "1e9", "--tokens", "1e9",
                                 "--gpu", gpu)
            self.assertEqual(rc, 0, f"{gpu} failed")
            self.assertIn("GPU-hours", out)


class TestChinchillaCli(unittest.TestCase):
    def test_params_only(self):
        rc, out, _ = run_cli("chinchilla", "--params", "7e9")
        self.assertEqual(rc, 0)
        self.assertIn("140B", out)  # 20 * 7e9

    def test_ratio_branches(self):
        cases = [("1e10", "UNDERTRAINED"),        # ratio 10
                 ("2e10", "Near compute-optimal"),  # ratio 20
                 ("4e10", "Overtraining regime")]   # ratio 40
        for tokens, expected in cases:
            rc, out, _ = run_cli("chinchilla", "--params", "1e9",
                                 "--tokens", tokens)
            self.assertEqual(rc, 0)
            self.assertIn(expected, out, f"tokens={tokens}")

    def test_no_args_errors(self):
        rc, _, err = run_cli("chinchilla")
        self.assertNotEqual(rc, 0)


class TestBatchCli(unittest.TestCase):
    def test_tokens_per_step_product(self):
        rc, out, _ = run_cli("batch", "--gpus", "8", "--micro", "4",
                             "--seq", "4096", "--accum", "8")
        self.assertEqual(rc, 0)
        self.assertIn("1,048,576", out)  # 8*4*4096*8
        self.assertIn("sequences/step = 256", out)

    def test_warmup_clamp_low(self):
        # steps ≈ 4 → 1% rounds to 0 → clamped up to 1000
        rc, out, _ = run_cli("batch", "--gpus", "8", "--micro", "4",
                             "--seq", "4096", "--accum", "8",
                             "--target-tokens", "4e6")
        self.assertEqual(rc, 0)
        self.assertIn("~1000 steps", out)

    def test_warmup_clamp_high(self):
        # steps ≈ 954k → 1% ≈ 9.5k → clamped down to 8000
        rc, out, _ = run_cli("batch", "--gpus", "8", "--micro", "4",
                             "--seq", "4096", "--accum", "8",
                             "--target-tokens", "1e12")
        self.assertEqual(rc, 0)
        self.assertIn("~8000 steps", out)

    def test_warmup_midrange_is_one_percent(self):
        tps = 8 * 4 * 4096 * 8
        target = tps * 300000  # 300k steps → 3000 warmup
        rc, out, _ = run_cli("batch", "--gpus", "8", "--micro", "4",
                             "--seq", "4096", "--accum", "8",
                             "--target-tokens", str(target))
        self.assertEqual(rc, 0)
        self.assertIn("~3000 steps", out)


class TestLrCli(unittest.TestCase):
    def test_every_stage_prints_its_ladder_row(self):
        for stage in estimate.LADDER:
            lr, src = ladder_lookup(stage, 8e9)
            rc, out, _ = run_cli("lr", "--params", "8e9", "--stage", stage)
            self.assertEqual(rc, 0, f"{stage} failed")
            self.assertIn(lr, out, f"{stage}: ladder lr not printed")
            self.assertIn(src, out, f"{stage}: ladder source not printed")

    def test_stage_is_case_insensitive(self):
        rc, out, _ = run_cli("lr", "--params", "8e9", "--stage", "SFT")
        self.assertEqual(rc, 0)

    def test_bad_stage_exits_nonzero(self):
        rc, _, err = run_cli("lr", "--params", "8e9", "--stage", "warmup")
        self.assertNotEqual(rc, 0)
        self.assertIn("stage must be one of", err)


if __name__ == "__main__":
    unittest.main()
