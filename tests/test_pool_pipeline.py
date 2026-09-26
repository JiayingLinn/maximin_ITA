"""Calibration separation and compatibility between pool preparation and allocation."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from pessimism.input_data import load_pool
from pessimism.run import run_comparison
from reward_training.response_pool import calibrate, digest, export, read_json, validate_pool, write_json


ROOT = Path(__file__).resolve().parents[1]


def pool_fixture(domain, split):
    text = f"{domain} {split} prompt"
    return {
        "domain": domain, "split": split,
        "scorers": {"proxy": {"adapter_sha256": "proxy"}, "judge": {"adapter_sha256": "judge"}},
        "prompts": [{"id": digest(text), "prompt": text, "candidates": [
            {"id": str(i), "text": str(i), "proxy_raw": proxy, "judge_raw": judge}
            for i, (proxy, judge) in enumerate(zip([0., 2., 4.], [0., 1., 4.]))]}],
    }


class PoolPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for domain in ("group_a", "group_b"):
            for split in ("validation", "test"):
                write_json(self.root / "pools" / domain / f"{split}.scored.json", pool_fixture(domain, split))
        self.validation = sorted(self.root.glob("pools/*/validation.scored.json"))
        self.test = sorted(self.root.glob("pools/*/test.scored.json"))
        self.calibration = self.root / "calibration.json"

    def prepare(self):
        calibrate(SimpleNamespace(inputs=self.validation, output=self.calibration))
        export(SimpleNamespace(inputs=self.test, calibration=self.calibration, output_dir=self.root / "inputs"))

    def test_frozen_validation_scales_and_rmse_feed_all_algorithms(self):
        for path in self.test:
            pool = read_json(path)
            for row, value in zip(pool["prompts"][0]["candidates"], [-2., 2., 8.]):
                row["proxy_raw"] = value
            write_json(path, pool)
        self.prepare()
        pool = load_pool(self.root / "inputs/pool_00000.json")
        self.assertEqual(pool.groups[0].proxy_scores, (0., 0.5, 1.))
        self.assertAlmostEqual(pool.groups[0].error_bound, (0.25 ** 2 / 3) ** 0.5)
        report = run_comparison(pool, budget=4, seed=2026)
        self.assertEqual(len(report["methods"]), 6)
        self.assertTrue(all(result["total_revealed"] == 4 for result in report["methods"].values()))

    def test_changing_test_judge_does_not_change_proxy_or_error_parameter(self):
        self.prepare()
        before = load_pool(self.root / "inputs/pool_00000.json")
        for path in self.test:
            pool = read_json(path)
            for row in pool["prompts"][0]["candidates"]:
                row["judge_raw"] = 100.0
            write_json(path, pool)
        export(SimpleNamespace(inputs=self.test, calibration=self.calibration, output_dir=self.root / "changed"))
        after = load_pool(self.root / "changed/pool_00000.json")
        for a, b in zip(before.groups, after.groups):
            self.assertEqual(a.proxy_scores, b.proxy_scores)
            self.assertEqual(a.error_bound, b.error_bound)
            self.assertNotEqual(a.judge_scores, b.judge_scores)

    def test_rejects_test_calibration_and_constant_scores(self):
        with self.assertRaisesRegex(ValueError, "never test"):
            calibrate(SimpleNamespace(inputs=self.test, output=self.calibration))
        pool = read_json(self.validation[0])
        for row in pool["prompts"][0]["candidates"]:
            row["proxy_raw"] = 1.0
        write_json(self.validation[0], pool)
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            calibrate(SimpleNamespace(inputs=self.validation, output=self.calibration))

    def test_rejects_overlapping_prompts_and_different_adapters(self):
        calibrate(SimpleNamespace(inputs=self.validation, output=self.calibration))
        original = read_json(self.test[0])
        for kind in ("overlap", "adapter"):
            changed = copy.deepcopy(original)
            if kind == "overlap":
                changed["prompts"] = read_json(self.validation[0])["prompts"]
            else:
                changed["scorers"]["proxy"]["adapter_sha256"] = "different"
            write_json(self.test[0], changed)
            with self.assertRaises(ValueError):
                export(SimpleNamespace(inputs=self.test, calibration=self.calibration, output_dir=self.root / "inputs"))
            self.assertFalse((self.root / "inputs").exists())

    def test_rejects_duplicate_candidates_and_nonfinite_rewards(self):
        for mutation in ("duplicate", "nan"):
            pool = pool_fixture("group_a", "test")
            candidates = pool["prompts"][0]["candidates"]
            if mutation == "duplicate":
                candidates[1]["id"] = candidates[0]["id"]
            else:
                candidates[0]["judge_raw"] = float("nan")
            with self.assertRaises(ValueError):
                validate_pool(pool, scored=True)

    def test_documented_shell_preparation_and_algorithm_stages(self):
        environment = {**os.environ, "PYTHON_BIN": sys.executable, "RUN_ROOT": str(self.root),
                       "DOMAINS": "group_a group_b", "SEED": "2026", "BUDGET": "4"}
        for stage in ("prepare", "run"):
            result = subprocess.run(["bash", "scripts/run_pipeline.sh", stage], cwd=ROOT,
                                    env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = read_json(self.root / "results/seed_2026/pool_00000.json")
        self.assertEqual(len(result["methods"]), 6)
        self.assertFalse(result["settings"]["judge_used_for_allocation"])


if __name__ == "__main__":
    unittest.main()
