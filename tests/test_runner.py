"""Budget, stream isolation, evaluation separation, and command-line checks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pessimism.core.validation import PessimismValidationError
from pessimism.input_data import ProxyStream, load_pool, parse_pool
from pessimism.run import METHODS, run_comparison


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.document = {"r_max": 1.0, "groups": [
            {"id": f"group_{i}", "error_bound": 0.05 * (i + 1),
             "candidates": [{"id": f"candidate_{j}",
                             "proxy_score": ((7 * j + i) % 31) / 31,
                             "judge_score": ((7 * j + i + 2) % 31) / 31}
                            for j in range(32)]}
            for i in range(3)]}
        self.pool = parse_pool(self.document)

    def test_every_method_spends_same_global_budget(self):
        report = run_comparison(self.pool, budget=31, seed=7)
        self.assertEqual(set(report["methods"]), set(METHODS))
        for name, result in report["methods"].items():
            with self.subTest(name=name):
                self.assertEqual(result["total_revealed"], 31)
                self.assertEqual(sum(result["final_counts"].values()), 31)
                for row in result["groups"]:
                    self.assertGreater(row["count"], 0)
                    self.assertLess(row["selected_index"], row["count"])
                    self.assertAlmostEqual(sum(row["selection_probabilities"]), 1)
                    self.assertTrue(all(p >= 0 for p in row["selection_probabilities"]))
        for method in ("uniform_argmax", "uniform_itp_auto", "uniform_itp_fixed"):
            self.assertEqual(list(report["methods"][method]["final_counts"].values()), [11, 10, 10])

    def test_deterministic_and_independent_of_method_order(self):
        first = run_comparison(self.pool, budget=30, seed=8)
        second = run_comparison(self.pool, budget=30, seed=8, methods=reversed(METHODS))
        self.assertEqual(first["methods"], second["methods"])
        solo = run_comparison(self.pool, budget=30, seed=8, methods=["lcb_greedy_auto"])
        self.assertEqual(first["methods"]["lcb_greedy_auto"], solo["methods"]["lcb_greedy_auto"])

    def test_judge_cannot_change_allocation_or_selection(self):
        changed = copy.deepcopy(self.document)
        for group in changed["groups"]:
            for candidate in group["candidates"]:
                candidate["judge_score"] = 1.0 - candidate["judge_score"]
        a = run_comparison(self.pool, budget=35, seed=4)
        b = run_comparison(parse_pool(changed), budget=35, seed=4)
        for name in METHODS:
            before, after = a["methods"][name], b["methods"][name]
            self.assertEqual(before["final_counts"], after["final_counts"])
            self.assertEqual(before.get("allocation_history"), after.get("allocation_history"))
            self.assertEqual(before["metrics"]["expected_proxy"], after["metrics"]["expected_proxy"])
            for row_a, row_b in zip(before["groups"], after["groups"]):
                self.assertEqual(row_a["selected_candidate"], row_b["selected_candidate"])
                self.assertEqual(row_a["selection_probabilities"], row_b["selection_probabilities"])

    def test_expected_scores_match_selection_law(self):
        report = run_comparison(self.pool, budget=30)
        for result in report["methods"].values():
            for pool_group, row in zip(self.pool.groups, result["groups"]):
                count = row["count"]
                probability = np.array(row["selection_probabilities"])
                self.assertAlmostEqual(row["expected_proxy"], probability @ pool_group.proxy_scores[:count])
                self.assertAlmostEqual(row["expected_judge"], probability @ pool_group.judge_scores[:count])
            expected = [row["expected_judge"] for row in result["groups"]]
            self.assertAlmostEqual(result["metrics"]["expected_judge"]["gap"], max(expected) - min(expected))

    def test_absent_judge_is_supported(self):
        for group in self.document["groups"]:
            for candidate in group["candidates"]:
                del candidate["judge_score"]
        report = run_comparison(parse_pool(self.document), budget=20)
        for result in report["methods"].values():
            self.assertNotIn("expected_judge", result["metrics"])
            self.assertTrue(all("expected_judge" not in row for row in result["groups"]))

    def test_stream_reveals_ordered_prefixes_and_fails_before_overrun(self):
        stream = ProxyStream(self.pool)
        ids_a, scores_a = stream("group_0", 2)
        ids_b, scores_b = stream("group_0", 3)
        self.assertEqual(ids_a + ids_b, list(self.pool.groups[0].candidate_ids[:5]))
        self.assertEqual(scores_a + scores_b, list(self.pool.groups[0].proxy_scores[:5]))
        with self.assertRaises(PessimismValidationError):
            stream("group_0", 30)
        self.assertEqual(stream.counts["group_0"], 5)
        self.assertEqual(ProxyStream(self.pool).counts["group_0"], 0)

    def test_capacity_exhaustion_and_uniform_infeasibility(self):
        doc = {"r_max": 1, "groups": [
            {"id": "short", "error_bound": 0.1,
             "candidates": [{"id": str(i), "proxy_score": 0.0} for i in range(4)]},
            {"id": "long", "error_bound": 0.1,
             "candidates": [{"id": str(i), "proxy_score": 0.7} for i in range(20)]},
        ]}
        pool = parse_pool(doc)
        methods = ["greedy_argmax", "lcb_greedy_auto", "lcb_greedy_fixed"]
        result = run_comparison(pool, methods=methods, budget=20)
        for method in methods:
            self.assertEqual(result["methods"][method]["final_counts"], {"short": 4, "long": 16})
            self.assertGreater(result["methods"][method]["capacity_affected_steps"], 0)
        with self.assertRaisesRegex(PessimismValidationError, "equal allocation"):
            run_comparison(pool, budget=20)
        with self.assertRaisesRegex(PessimismValidationError, "exceeds pool capacity"):
            run_comparison(pool, methods=["greedy_argmax"], budget=20, capacity_mode="uncapped")

    def test_initialization_budget_includes_all_groups(self):
        result = run_comparison(self.pool, budget=9, initial_count=3)
        for method in ("lcb_greedy_auto", "lcb_greedy_fixed"):
            self.assertEqual(list(result["methods"][method]["final_counts"].values()), [3, 3, 3])
            self.assertEqual(result["methods"][method]["allocation_history"], [])
        with self.assertRaisesRegex(PessimismValidationError, "initialization"):
            run_comparison(self.pool, budget=8, initial_count=3)

    def test_report_distinguishes_floor_and_certifiable_count(self):
        report = run_comparison(self.pool, budget=30, initial_count=3, slack=1000)
        schedule = report["methods"]["lcb_greedy_auto"]["schedule"]
        self.assertEqual(schedule["initial_count"], 3)
        self.assertEqual(schedule["certifiable_count"], 1)
        self.assertTrue(schedule["guarantee_is_vacuous"])
        self.assertEqual(report["settings"]["slack_source"], "explicit")

    def test_fixed_beta_and_alpha_are_recorded(self):
        report = run_comparison(self.pool, budget=30, fixed_beta=0.0001, alpha=0.5)
        for method in ("uniform_itp_fixed", "lcb_greedy_fixed"):
            for group, row in zip(self.pool.groups, report["methods"][method]["groups"]):
                self.assertEqual(row["beta"], 0.0001)
                self.assertAlmostEqual(row["effective_error"], group.error_bound * 0.5)
            self.assertTrue(any("bypasses" in text for text in report["methods"][method]["caveats"]))

    def test_invalid_run_options(self):
        cases = [
            {"budget": 0}, {"budget": 2}, {"budget": 100}, {"budget": 3.5},
            {"initial_count": 0}, {"alpha": -1}, {"alpha": float("nan")},
            {"seed": -1}, {"seed": True}, {"fixed_beta": 0},
            {"beta_grid": [1, 0.1]}, {"beta_grid": [1, 1]},
            {"slack": 0}, {"confidence_delta": 1}, {"bisection_tol": 0},
            {"allocation_batch_size": 0}, {"index_error_scale": float("inf")},
            {"methods": []}, {"methods": ["unknown"]},
            {"methods": ["uniform_argmax", "uniform_argmax"]}, {"capacity_mode": "unknown"},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(PessimismValidationError):
                run_comparison(self.pool, **options)

    def test_input_validation(self):
        cases = []
        for score in (-0.1, 1.1, float("nan"), float("inf"), True, "0.2", None):
            doc = copy.deepcopy(self.document)
            doc["groups"][0]["candidates"][0]["proxy_score"] = score
            cases.append(doc)
        for field, value in (("id", ""), ("error_bound", -1), ("candidates", [])):
            doc = copy.deepcopy(self.document)
            doc["groups"][0][field] = value
            cases.append(doc)
        cases += [{"r_max": 1, "groups": []}, {"r_max": 0, "groups": []}, {}]
        doc = copy.deepcopy(self.document)
        doc["groups"][1]["id"] = doc["groups"][0]["id"]
        cases.append(doc)
        doc = copy.deepcopy(self.document)
        doc["groups"][0]["candidates"][1]["id"] = doc["groups"][0]["candidates"][0]["id"]
        cases.append(doc)
        doc = copy.deepcopy(self.document)
        del doc["groups"][0]["candidates"][0]["judge_score"]
        cases.append(doc)
        doc = copy.deepcopy(self.document)
        doc["groups"][0]["candidates"][0]["judge_score"] = 2
        cases.append(doc)
        doc = copy.deepcopy(self.document)
        doc["dataset_path"] = "unexpected_field"
        cases.append(doc)
        for document in cases:
            with self.subTest(document=document), self.assertRaises(PessimismValidationError):
                parse_pool(document)

    def test_duplicate_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text('{"r_max": 1, "r_max": 2, "groups": []}', encoding="utf-8")
            with self.assertRaisesRegex(PessimismValidationError, "duplicate JSON field"):
                load_pool(path)

    def test_cli_file_and_function_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pool.json").write_text(json.dumps(self.document))
            common = [sys.executable, "-m", "pessimism.run", "--budget", "21", "--seed", "4"]
            a = subprocess.run(common + ["--input", str(root / "pool.json")], cwd=ROOT,
                               check=True, capture_output=True, text=True)
            expected = run_comparison(self.pool, budget=21, seed=4)
            self.assertEqual(json.loads(a.stdout)["methods"], expected["methods"])
            subprocess.run(common + ["--input", str(root / "pool.json"), "--output", str(root / "nested" / "result.json")],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertTrue((root / "nested" / "result.json").is_file())

    def test_cli_requires_source_and_reports_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text(json.dumps(self.document))
            for args in ([], ["--input", str(path), "--budget", "0"],
                         ["--input", str(path), "--beta-grid", "0.1,nan"]):
                result = subprocess.run([sys.executable, "-m", "pessimism.run", *args], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
