"""Analytical and allocation regression tests using synthetic scores only.

These checks cover the ITP oracle, certification schedule, adaptive allocation,
finite capacity, fixed-beta variants, and equal/greedy best-of-n baselines.
No model, judge, dataset, or original response pool is needed.
"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from pessimism.baselines import even_split, greedy_argmax, uniform_argmax, uniform_itp
from pessimism.core import lcb_greedy as lcb_module
from pessimism.core.itp import itp_certify, itp_sample
from pessimism.core.lcb_greedy import group_index, lcb_greedy
from pessimism.core.radii import (
    CertificationSchedule,
    certified_betas,
    default_bisection_tol,
    delta_itp,
    delta_obj,
    delta_total,
    geometric_beta_grid,
    initial_count,
    log_factor,
    m_beta,
)
from pessimism.core.validation import PessimismValidationError

TIGHT = 1e-12


class PoolExhausted(RuntimeError):
    """The synthetic test stream has no further samples."""


def exact_normalizer(scores, beta, iterations=400):
    """A far tighter bisection than any test tolerance, for comparison."""
    scores = np.asarray(scores, dtype=np.float64)
    low, high = -beta, float(scores.max())
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if float(np.mean(np.maximum(scores - middle, 0.0))) / beta - 1.0 > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class TestITPFormulas(unittest.TestCase):
    """Spec 8.1: Algorithm 2 against its closed forms."""

    def test_constant_scores(self):
        for constant, beta, bound in ((0.7, 0.3, 0.1), (0.0, 1.0, 0.0), (1.0, 0.05, 0.25)):
            with self.subTest(constant=constant, beta=beta):
                certification = itp_certify([constant] * 11, beta, bound, TIGHT)
                self.assertAlmostEqual(certification.normalizer, constant - beta, places=9)
                np.testing.assert_allclose(certification.weights, 1.0, atol=1e-9)
                self.assertAlmostEqual(
                    certification.certificate,
                    constant - bound - bound * bound / (2.0 * beta),
                    places=9,
                )
                sampled = itp_sample(
                    list(range(11)), [constant] * 11, beta, bound, TIGHT,
                    np.random.default_rng(0),
                )
                np.testing.assert_allclose(sampled.probabilities, 1.0 / 11, atol=1e-9)

    def test_two_point_analytic_case(self):
        certification = itp_certify([0.0, 1.0], 1.0, 0.0, TIGHT)
        self.assertAlmostEqual(certification.normalizer, -0.5, places=9)
        np.testing.assert_allclose(certification.weights, [0.5, 1.5], atol=1e-9)
        self.assertAlmostEqual(certification.certificate, 0.625, places=9)
        sampled = itp_sample(
            ["a", "b"], [0.0, 1.0], 1.0, 0.0, TIGHT, np.random.default_rng(1)
        )
        np.testing.assert_allclose(sampled.probabilities, [0.25, 0.75], atol=1e-9)

    def test_thresholded_case(self):
        certification = itp_certify([0.0, 1.0], 0.25, 0.0, TIGHT)
        self.assertAlmostEqual(certification.normalizer, 0.5, places=9)
        np.testing.assert_allclose(certification.weights, [0.0, 2.0], atol=1e-9)
        self.assertAlmostEqual(certification.certificate, 0.875, places=9)
        sampled = itp_sample(
            ["a", "b"], [0.0, 1.0], 0.25, 0.0, TIGHT, np.random.default_rng(2)
        )
        np.testing.assert_allclose(sampled.probabilities, [0.0, 1.0], atol=1e-12)
        # A zero-weight response is never selected, in any draw.
        rng = np.random.default_rng(3)
        for _ in range(200):
            draw = itp_sample(["a", "b"], [0.0, 1.0], 0.25, 0.0, TIGHT, rng)
            self.assertEqual(draw.selected_index, 1)

    def test_weight_invariants(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            n = int(rng.integers(1, 60))
            scores = rng.uniform(0.0, 2.5, size=n)
            beta = float(rng.uniform(0.02, 4.0))
            tol = float(rng.uniform(1e-9, 1e-3))
            certification = itp_certify(scores, beta, 0.0, tol)
            self.assertGreaterEqual(certification.weights.min(), 0.0)
            self.assertGreaterEqual(certification.mean_weight, 1.0 - 1e-9)
            self.assertLessEqual(certification.mean_weight, 1.0 + tol / beta + 1e-9)
            self.assertGreater(certification.weights.sum(), 0.0)
            sampled = itp_sample(
                list(range(n)), scores, beta, 0.0, tol, np.random.default_rng(5)
            )
            self.assertAlmostEqual(float(sampled.probabilities.sum()), 1.0, places=12)
            zero = certification.weights == 0.0
            np.testing.assert_allclose(sampled.probabilities[zero], 0.0, atol=0.0)

    def test_bisection_matches_high_precision_root(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            scores = rng.uniform(0.0, 1.0, size=int(rng.integers(2, 80)))
            beta = float(rng.uniform(0.05, 2.0))
            tol = 1e-7
            certification = itp_certify(scores, beta, 0.0, tol)
            root = exact_normalizer(scores, beta)
            # The returned normalizer is the lower endpoint: at or below the
            # root, and within one tolerance of it.
            self.assertLessEqual(certification.normalizer, root + 1e-12)
            self.assertGreaterEqual(certification.normalizer, root - tol)

    def test_error_bound_shifts_certificate_exactly(self):
        scores = [0.1, 0.4, 0.9, 0.25]
        beta = 0.4
        base = itp_certify(scores, beta, 0.0, TIGHT).certificate
        for bound in (0.0, 0.05, 0.3, 1.2):
            shifted = itp_certify(scores, beta, bound, TIGHT).certificate
            self.assertAlmostEqual(
                shifted, base - bound - bound * bound / (2.0 * beta), places=12
            )

    def test_penalty_scale_only_tempers_reward_mismatch_penalty(self):
        scores = [0.1, 0.4, 0.9, 0.25]
        beta, bound = 0.4, 0.3
        original = itp_certify(scores, beta, bound, TIGHT)
        explicit_one = itp_certify(
            scores, beta, bound, TIGHT, penalty_scale=1.0
        )
        self.assertEqual(original.certificate, explicit_one.certificate)
        np.testing.assert_array_equal(original.weights, explicit_one.weights)
        self.assertEqual(original.raw_penalty, explicit_one.used_penalty)

        quarter = itp_certify(
            scores, beta, bound, TIGHT, penalty_scale=0.25
        )
        self.assertEqual(quarter.normalizer, original.normalizer)
        np.testing.assert_array_equal(quarter.weights, original.weights)
        self.assertAlmostEqual(quarter.objective, original.objective, places=15)
        self.assertAlmostEqual(quarter.raw_penalty, original.raw_penalty, places=15)
        self.assertAlmostEqual(
            quarter.used_penalty, 0.25 * original.raw_penalty, places=15
        )
        self.assertAlmostEqual(
            quarter.certificate,
            quarter.objective - 0.25 * original.raw_penalty,
            places=15,
        )

        oracle = itp_certify(scores, beta, bound, TIGHT, penalty_scale=0.0)
        self.assertEqual(oracle.used_penalty, 0.0)
        self.assertEqual(oracle.certificate, oracle.objective)

    def test_duplicate_responses_are_allowed(self):
        sampled = itp_sample(
            ["same", "same", "same"], [0.2, 0.2, 0.9], 0.5, 0.0, TIGHT,
            np.random.default_rng(4),
        )
        self.assertEqual(sampled.selected_response, "same")
        self.assertEqual(sampled.probabilities.size, 3)


class TestRadii(unittest.TestCase):
    """Spec 8.2: Definitions 3.2 and 4.1."""

    def setUp(self):
        self.grid = geometric_beta_grid(0.05, 2.0, 8)
        self.r_max = 1.0
        self.ell = log_factor(5, self.grid.size, 400, 0.1)

    def test_exact_formulas(self):
        k, n_beta, budget, delta = 5, 8, 400, 0.1
        expected = math.log(
            16.0 * k * n_beta * budget * (budget * budget + 2.0) / delta
        )
        self.assertAlmostEqual(log_factor(k, n_beta, budget, delta), expected, places=12)
        self.assertAlmostEqual(m_beta(0.25, 1.0), 5.0, places=12)
        n, beta = 37, 0.8
        ratio = m_beta(beta, self.r_max) * self.ell / n
        self.assertAlmostEqual(
            delta_obj(n, beta, self.r_max, self.ell),
            5.0 * self.r_max * (math.sqrt(ratio) + ratio),
            places=12,
        )
        self.assertAlmostEqual(
            delta_itp(n, beta, self.r_max, self.ell),
            self.r_max * math.sqrt(ratio),
            places=12,
        )
        self.assertAlmostEqual(
            delta_total(n, beta, self.r_max, self.ell),
            delta_obj(n, beta, self.r_max, self.ell)
            + delta_itp(n, beta, self.r_max, self.ell),
            places=12,
        )
        self.assertAlmostEqual(
            delta_total(n, beta, self.r_max, self.ell),
            6.0 * self.r_max * math.sqrt(ratio) + 5.0 * self.r_max * ratio,
            places=12,
        )

    def test_radius_is_nonincreasing_in_n_and_beta(self):
        for beta in self.grid:
            values = [delta_total(n, float(beta), self.r_max, self.ell) for n in range(1, 200)]
            self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        for n in (1, 5, 50, 400):
            values = [delta_total(n, float(beta), self.r_max, self.ell) for beta in self.grid]
            self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_certified_set_grows_with_n(self):
        slack = 14.0
        previous: set[float] = set()
        for n in range(1, 400):
            current = {
                float(beta)
                for beta in certified_betas(n, self.grid, self.r_max, self.ell, slack)
            }
            self.assertTrue(previous <= current, f"certified set shrank at n={n}")
            previous = current
        # It is an upper segment of the grid, never a gapped subset.
        final = certified_betas(399, self.grid, self.r_max, self.ell, slack)
        self.assertTrue(np.array_equal(final, self.grid[self.grid.size - final.size :]))

    def test_initial_count_is_the_first_certifiable_count(self):
        slack = 14.0
        start = initial_count(self.grid, self.r_max, self.ell, slack, 400)
        self.assertGreater(
            certified_betas(start, self.grid, self.r_max, self.ell, slack).size, 0
        )
        if start > 1:
            self.assertEqual(
                certified_betas(start - 1, self.grid, self.r_max, self.ell, slack).size, 0
            )

    def test_clear_failure_when_nothing_certifies(self):
        with self.assertRaises(PessimismValidationError):
            initial_count(self.grid, self.r_max, self.ell, 0.001, 400)
        # Certifiable, but the k * n_0 initialization does not fit the budget.
        with self.assertRaises(PessimismValidationError) as caught:
            CertificationSchedule.build(
                num_groups=5, total_budget=200, beta_grid=self.grid,
                r_max=1.0, slack=8.0, confidence_delta=0.1,
            )
        self.assertIn("initial_count", str(caught.exception))

    def test_default_bisection_tolerance(self):
        self.assertAlmostEqual(
            default_bisection_tol(self.grid, 1.0, 400), 0.05 / 400, places=15
        )
        self.assertAlmostEqual(
            default_bisection_tol(self.grid, 0.01, 400), 0.01 / 400, places=15
        )

    def test_grid_endpoint_convention(self):
        grid = geometric_beta_grid(0.05, 2.0, 4)
        np.testing.assert_allclose(grid, [0.05, 0.1, 0.2, 0.4])
        self.assertEqual(grid.size, 4)


class Stream:
    """A deterministic fake `sample_and_score` with a per-group fixed order."""

    def __init__(self, tables):
        self.tables = {key: list(value) for key, value in tables.items()}
        self.cursors = {key: 0 for key in tables}
        self.calls = []

    def __call__(self, group, count):
        start = self.cursors[group]
        stop = start + count
        table = self.tables[group]
        if stop > len(table):
            raise PoolExhausted(f"{group} exhausted at {stop} of {len(table)}")
        self.cursors[group] = stop
        self.calls.append((group, count))
        return [(group, i) for i in range(start, stop)], table[start:stop]


def make_tables(seed=0, n=4000, groups=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    offsets = {"a": 0.15, "b": 0.5, "c": 0.8}
    return {
        group: np.clip(rng.normal(offsets.get(group, 0.5), 0.2, size=n), 0.0, 1.0)
        for group in groups
    }


class TestLCBGreedy(unittest.TestCase):
    """Spec 8.3: Algorithm 1's allocation, on a deterministic fake stream."""

    def setUp(self):
        self.grid = geometric_beta_grid(0.05, 2.0, 8)
        self.groups = ["a", "b", "c"]
        self.budget = 900
        self.kwargs = dict(
            group_ids=self.groups,
            total_budget=self.budget,
            error_bounds=[0.0, 0.0, 0.0],
            slack=14.0,
            confidence_delta=0.1,
            beta_grid=self.grid,
            r_max=1.0,
        )

    def run_once(self, seed=0, tables=None):
        stream = Stream(tables if tables is not None else make_tables())
        result = lcb_greedy(
            sample_and_score=stream, rng=np.random.default_rng(seed), **self.kwargs
        )
        return result, stream

    def test_counts_sum_to_budget_and_stream_agrees(self):
        result, stream = self.run_once()
        self.assertEqual(sum(result.final_counts.values()), self.budget)
        self.assertEqual(result.total_revealed, self.budget)
        self.assertEqual(sum(stream.cursors.values()), self.budget)

    def test_every_group_gets_initial_count(self):
        result, _ = self.run_once()
        for group in self.groups:
            self.assertGreaterEqual(
                result.final_counts[group], result.schedule.initial_count
            )

    def test_batches_double_except_the_final_one(self):
        result, _ = self.run_once()
        for step in result.history[:-1]:
            self.assertEqual(step.batch_size, step.old_count)
            self.assertEqual(step.new_count, 2 * step.old_count)
        last = result.history[-1]
        self.assertLessEqual(last.batch_size, last.old_count)
        self.assertEqual(last.revealed_after, self.budget)

    def test_fixed_batch_refreshes_after_each_requested_batch(self):
        stream = Stream(make_tables())
        result = lcb_greedy(
            sample_and_score=stream,
            rng=np.random.default_rng(0),
            allocation_batch_size=8,
            **self.kwargs,
        )
        for step in result.history[:-1]:
            self.assertEqual(step.batch_size, 8)
        self.assertLessEqual(result.history[-1].batch_size, 8)
        self.assertEqual(result.history[-1].revealed_after, self.budget)
        self.assertEqual(sum(result.final_counts.values()), self.budget)

    def test_invalid_fixed_batch_is_refused(self):
        for bad in (0, -1, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises(PessimismValidationError):
                    lcb_greedy(
                        sample_and_score=Stream(make_tables()),
                        rng=np.random.default_rng(0),
                        allocation_batch_size=bad,
                        **self.kwargs,
                    )

    def test_chosen_group_is_always_a_current_minimiser(self):
        """Replay the run from the recorded indices and check every argmin."""
        result, _ = self.run_once()
        current = dict(result.initial_indices)
        self.assertEqual(set(current), set(self.groups))
        for step in result.history:
            self.assertAlmostEqual(current[step.group], step.index_before, places=12)
            minimum = min(current.values())
            self.assertAlmostEqual(step.index_before, minimum, places=12)
            self.assertEqual(
                set(step.tied_groups),
                {group for group, value in current.items() if value == minimum},
            )
            self.assertIn(step.group, step.tied_groups)
            current[step.group] = step.index_after

    def test_only_the_chosen_group_is_recomputed(self):
        calls = []
        original = lcb_module.group_index

        def counting(scores, certified, error_bound, bisection_tol):
            calls.append(len(scores))
            return original(scores, certified, error_bound, bisection_tol)

        with patch.object(lcb_module, "group_index", counting):
            result, _ = self.run_once()
        # k initialization calls plus exactly one per allocation step.
        self.assertEqual(len(calls), len(self.groups) + len(result.history))

    def test_tie_breaking_is_reproducible_and_seed_dependent(self):
        tied = {group: np.full(4000, 0.5) for group in self.groups}
        first, _ = self.run_once(seed=5, tables=tied)
        again, _ = self.run_once(seed=5, tables=tied)
        self.assertEqual(first.final_counts, again.final_counts)
        histories = {
            seed: tuple(step.group for step in self.run_once(seed=seed, tables=tied)[0].history)
            for seed in range(12)
        }
        self.assertGreater(len(set(histories.values())), 1)
        # With identical scores every group ties at every step, so the uniform
        # draw is the only thing deciding, and it must be a real draw.
        for result_history in histories.values():
            self.assertEqual(len(set(result_history)) > 1, True)

    def test_outputs_come_from_the_group_and_cost_no_new_draws(self):
        result, stream = self.run_once()
        revealed_after_allocation = dict(stream.cursors)
        calls_after_allocation = len(stream.calls)
        self.assertEqual(sum(revealed_after_allocation.values()), self.budget)
        for group, outcome in result.outcomes.items():
            self.assertEqual(outcome.selected_response[0], group)
            self.assertLess(outcome.selected_response[1], result.final_counts[group])
            self.assertGreater(outcome.selection_probability, 0.0)
        # lcb_greedy has already returned, so no callback ran after allocation.
        self.assertEqual(len(stream.calls), calls_after_allocation)

    def test_one_categorical_draw_per_group(self):
        draws = []
        original = lcb_module.itp_sample

        def counting(**kwargs):
            draws.append(kwargs["beta"])
            return original(**kwargs)

        with patch.object(lcb_module, "itp_sample", counting):
            result, _ = self.run_once()
        self.assertEqual(len(draws), len(self.groups))
        self.assertEqual(len(result.outcomes), len(self.groups))


    def test_index_uses_only_certified_betas(self):
        result, _ = self.run_once()
        for group, outcome in result.outcomes.items():
            certified = result.schedule.certified(result.final_counts[group])
            self.assertIn(outcome.beta_hat, [float(beta) for beta in certified])

    def test_beta_tie_rule_is_the_smallest_maximiser(self):
        # Constant scores make the certificate `c - bound - bound^2/(2 beta)`,
        # which is strictly increasing in beta, so the largest beta wins; with a
        # zero bound every beta ties and the smallest must be returned.
        constant = np.full(64, 0.6)
        grid = geometric_beta_grid(0.1, 2.0, 4)
        # An exact tie has to be constructed: with a finite bisection tolerance
        # the four analytically equal certificates differ in the last bits, and
        # then the argmax is deciding, not the tie rule.
        original = itp_certify

        def flat(scores, beta, error_bound, bisection_tol):
            certification = original(scores, beta, error_bound, bisection_tol)
            return type(certification)(
                **{**certification.__dict__, "certificate": 0.5}
            )

        with patch.object(lcb_module, "itp_certify", flat):
            tied = group_index(constant, grid, 0.0, TIGHT)
        self.assertAlmostEqual(tied.beta, float(grid[0]), places=12)
        self.assertEqual(len(tied.certificates), grid.size)
        # A real separation still picks the true maximiser: with a positive
        # error bound the penalty -bound^2/(2 beta) shrinks as beta grows.
        separated = group_index(constant, grid, 0.2, TIGHT)
        self.assertAlmostEqual(separated.beta, float(grid[-1]), places=12)

    def test_greedy_favours_the_worst_group(self):
        """The allocation's whole content: the low-scoring group gets the budget."""
        result, _ = self.run_once()
        counts = result.final_counts
        self.assertGreater(counts["a"], counts["c"])

    def test_uniform_baseline_spends_the_same_budget(self):
        schedule = CertificationSchedule.build(
            num_groups=3, total_budget=self.budget, beta_grid=self.grid,
            r_max=1.0, slack=14.0, confidence_delta=0.1,
        )
        stream = Stream(make_tables())
        baseline = uniform_itp(
            self.groups, self.budget, [0.0] * 3, schedule, stream,
            np.random.default_rng(0),
        )
        self.assertEqual(baseline.total_revealed, self.budget)
        self.assertEqual(sorted(baseline.final_counts.values()), [300, 300, 300])
        self.assertEqual(even_split(10, 3), [4, 3, 3])


class TestCapacity(unittest.TestCase):
    """The finite-pool deviation: skip full groups, truncate the last batch."""

    def setUp(self):
        self.grid = geometric_beta_grid(0.0005, 2.0, 12)
        self.groups = ["a", "b", "c"]

    def kwargs(self, budget, cap):
        return dict(
            group_ids=self.groups, total_budget=budget, error_bounds=[0.0] * 3,
            slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
            capacity={g: cap for g in self.groups},
        )

    def test_no_group_exceeds_its_capacity(self):
        tables = make_tables(seed=5, n=4000)
        stream = Stream(tables)
        result = lcb_greedy(
            sample_and_score=stream, rng=np.random.default_rng(0),
            **self.kwargs(600, 210),
        )
        self.assertEqual(sum(result.final_counts.values()), 600)
        for group, count in result.final_counts.items():
            self.assertLessEqual(count, 210)
        # A truncated batch must be recorded, not silently applied.
        self.assertTrue(any(step.capped for step in result.history))

    def test_full_groups_leave_the_argmin(self):
        result = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=6, n=4000)),
            rng=np.random.default_rng(0), **self.kwargs(600, 210),
        )
        for step in result.history:
            self.assertNotIn(step.group, step.excluded)
            for group in step.excluded:
                self.assertNotIn(group, step.tied_groups)

    def test_budget_beyond_the_pool_is_refused(self):
        with self.assertRaises(PessimismValidationError):
            lcb_greedy(
                sample_and_score=Stream(make_tables(seed=7, n=4000)),
                rng=np.random.default_rng(0), **self.kwargs(700, 210),
            )

    def test_capacity_below_initial_count_is_refused(self):
        # Total capacity clears the budget, so this can only fail on the
        # initialization check rather than the cruder total-capacity one.
        with self.assertRaises(PessimismValidationError) as caught:
            lcb_greedy(
                sample_and_score=Stream(make_tables(seed=8, n=4000)),
                rng=np.random.default_rng(0), **self.kwargs(90, 32),
            )
        self.assertIn("initial_count", str(caught.exception))

    def test_budget_beyond_total_capacity_is_refused_up_front(self):
        with self.assertRaises(PessimismValidationError) as caught:
            lcb_greedy(
                sample_and_score=Stream(make_tables(seed=8, n=4000)),
                rng=np.random.default_rng(0), **self.kwargs(300, 20),
            )
        self.assertIn("capacity", str(caught.exception))

    def test_paper_behaviour_is_unchanged_without_capacity(self):
        tables = make_tables(seed=9, n=4000)
        base = dict(
            group_ids=self.groups, total_budget=600, error_bounds=[0.0] * 3,
            slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
        )
        a = lcb_greedy(sample_and_score=Stream(tables), rng=np.random.default_rng(1), **base)
        b = lcb_greedy(
            sample_and_score=Stream(tables), rng=np.random.default_rng(1),
            capacity={g: 10**6 for g in self.groups}, **base,
        )
        self.assertEqual(a.final_counts, b.final_counts)
        self.assertFalse(any(step.capped for step in b.history))

    def test_greedy_argmax_respects_capacity(self):
        stream = Stream(make_tables(seed=10, n=4000))
        picked = greedy_argmax(self.groups, 600, stream, {g: 210 for g in self.groups})
        self.assertEqual(sum(v["final_count"] for v in picked.values()), 600)
        for value in picked.values():
            self.assertLessEqual(value["final_count"], 210)
        with self.assertRaises(PessimismValidationError):
            greedy_argmax(
                self.groups, 700, Stream(make_tables(seed=10, n=4000)),
                {g: 210 for g in self.groups},
            )


class TestFixedBeta(unittest.TestCase):
    """Pinning beta must change the regularization and nothing else."""

    def setUp(self):
        self.grid = geometric_beta_grid(0.0005, 2.0, 12)
        self.base = dict(
            group_ids=["a", "b", "c"], total_budget=600, error_bounds=[0.05] * 3,
            slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
        )

    def test_every_index_uses_the_pinned_beta(self):
        result = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=2, n=4000)),
            rng=np.random.default_rng(0), fixed_beta=0.0063, **self.base,
        )
        for outcome in result.outcomes.values():
            self.assertAlmostEqual(outcome.beta_hat, 0.0063, places=12)
        for step in result.history:
            self.assertAlmostEqual(step.beta_after, 0.0063, places=12)

    def test_the_pinned_beta_need_not_be_certified(self):
        """The point of the flag: reach betas the radius will not certify."""
        result = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=2, n=4000)),
            rng=np.random.default_rng(0), fixed_beta=0.0005, **self.base,
        )
        schedule = result.schedule
        for group, count in result.final_counts.items():
            certified = [float(b) for b in schedule.certified(count)]
            self.assertNotIn(0.0005, certified)
            self.assertAlmostEqual(result.outcomes[group].beta_hat, 0.0005, places=12)

    def test_schedule_and_initial_count_are_untouched(self):
        plain = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=3, n=4000)),
            rng=np.random.default_rng(0), **self.base,
        )
        pinned = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=3, n=4000)),
            rng=np.random.default_rng(0), fixed_beta=0.0063, **self.base,
        )
        self.assertEqual(plain.schedule.initial_count, pinned.schedule.initial_count)
        self.assertAlmostEqual(plain.schedule.slack, pinned.schedule.slack, places=12)
        self.assertEqual(plain.schedule.ell, pinned.schedule.ell)
        self.assertEqual(sum(pinned.final_counts.values()), 600)

    def test_a_pinned_beta_equal_to_the_certified_choice_reproduces_it(self):
        plain = lcb_greedy(
            sample_and_score=Stream(make_tables(seed=4, n=4000)),
            rng=np.random.default_rng(1), **self.base,
        )
        only = {o.beta_hat for o in plain.outcomes.values()}
        if len(only) == 1:
            pinned = lcb_greedy(
                sample_and_score=Stream(make_tables(seed=4, n=4000)),
                rng=np.random.default_rng(1), fixed_beta=only.pop(), **self.base,
            )
            self.assertEqual(plain.final_counts, pinned.final_counts)

    def test_invalid_pinned_beta_is_refused(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(PessimismValidationError):
                lcb_greedy(
                    sample_and_score=Stream(make_tables(seed=2, n=4000)),
                    rng=np.random.default_rng(0), fixed_beta=bad, **self.base,
                )


class TestBaselines(unittest.TestCase):
    """The incumbent rules the judge comparison is run against."""

    def setUp(self):
        self.groups = ["a", "b", "c"]

    def test_greedy_argmax_spends_the_budget_and_helps_the_worst_group(self):
        stream = Stream(make_tables(seed=1))
        picked = greedy_argmax(self.groups, 300, stream)
        self.assertEqual(sum(v["final_count"] for v in picked.values()), 300)
        self.assertEqual(sum(stream.cursors.values()), 300)
        # 'a' is the low-offset group in make_tables, so it must be fed most.
        self.assertGreater(picked["a"]["final_count"], picked["c"]["final_count"])

    def test_greedy_argmax_returns_its_own_prefix_maximum(self):
        stream = Stream(make_tables(seed=2))
        picked = greedy_argmax(self.groups, 300, stream)
        for group, value in picked.items():
            revealed = stream.tables[group][: value["final_count"]]
            self.assertAlmostEqual(
                value["selected_score"], float(np.max(revealed)), places=12
            )
            self.assertEqual(value["selected_response"][0], group)

    def test_greedy_argmax_gives_every_group_one_draw(self):
        stream = Stream(make_tables(seed=3))
        picked = greedy_argmax(self.groups, len(self.groups), stream)
        self.assertEqual([v["final_count"] for v in picked.values()], [1, 1, 1])
        with self.assertRaises(PessimismValidationError):
            greedy_argmax(self.groups, 2, Stream(make_tables(seed=3)))

    def test_uniform_argmax_matches_a_direct_prefix_maximum(self):
        stream = Stream(make_tables(seed=4))
        picked = uniform_argmax(self.groups, 300, stream)
        for group, value in picked.items():
            self.assertEqual(value["final_count"], 100)
            self.assertAlmostEqual(
                value["selected_score"],
                float(np.max(stream.tables[group][:100])),
                places=12,
            )


class TestValidation(unittest.TestCase):
    """Spec 8.4: bad inputs are refused, not repaired."""

    def setUp(self):
        self.grid = geometric_beta_grid(0.05, 2.0, 8)

    def test_bad_scores(self):
        for scores in ([], [float("nan"), 0.1], [float("inf")], [[0.1, 0.2]]):
            with self.subTest(scores=scores):
                with self.assertRaises(PessimismValidationError):
                    itp_certify(scores, 1.0, 0.0, TIGHT)

    def test_bad_scalars(self):
        with self.assertRaises(PessimismValidationError):
            itp_certify([0.1], 0.0, 0.0, TIGHT)
        with self.assertRaises(PessimismValidationError):
            itp_certify([0.1], -1.0, 0.0, TIGHT)
        with self.assertRaises(PessimismValidationError):
            itp_certify([0.1], 1.0, -0.1, TIGHT)
        with self.assertRaises(PessimismValidationError):
            itp_certify([0.1], 1.0, 0.0, 0.0)

    def test_bad_schedule_inputs(self):
        base = dict(
            num_groups=3, total_budget=900, beta_grid=self.grid,
            r_max=1.0, slack=14.0, confidence_delta=0.1,
        )
        for override in (
            {"r_max": 0.0},
            {"r_max": -1.0},
            {"slack": 0.0},
            {"confidence_delta": 0.0},
            {"confidence_delta": 1.0},
            {"confidence_delta": 1.5},
            {"total_budget": 0},
            {"total_budget": 3.5},
            {"beta_grid": [0.2, 0.1]},
            {"beta_grid": [0.1, 0.1]},
            {"beta_grid": [0.0, 0.1]},
            {"beta_grid": []},
        ):
            with self.subTest(override=override):
                with self.assertRaises(PessimismValidationError):
                    CertificationSchedule.build(**{**base, **override})

    def test_out_of_range_scores_are_refused_by_algorithm_one(self):
        stream = Stream({"a": np.full(500, 1.5), "b": np.full(500, 0.5)})
        with self.assertRaises(PessimismValidationError) as caught:
            lcb_greedy(
                group_ids=["a", "b"], total_budget=300, error_bounds=[0.0, 0.0],
                slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
                sample_and_score=stream, rng=np.random.default_rng(0),
            )
        self.assertIn("[0, 1.0]", str(caught.exception))

    def test_short_callback_output_is_refused(self):
        def short(group, count):
            return [0] * (count - 1), np.full(count - 1, 0.5)

        with self.assertRaises(PessimismValidationError):
            lcb_greedy(
                group_ids=["a", "b"], total_budget=300, error_bounds=[0.0, 0.0],
                slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
                sample_and_score=short, rng=np.random.default_rng(0),
            )

    def test_duplicate_groups_and_mismatched_bounds(self):
        stream = Stream(make_tables(groups=("a", "b")))
        with self.assertRaises(PessimismValidationError):
            lcb_greedy(
                group_ids=["a", "a"], total_budget=300, error_bounds=[0.0, 0.0],
                slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
                sample_and_score=stream, rng=np.random.default_rng(0),
            )
        with self.assertRaises(PessimismValidationError):
            lcb_greedy(
                group_ids=["a", "b"], total_budget=300, error_bounds=[0.0],
                slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
                sample_and_score=stream, rng=np.random.default_rng(0),
            )

    def test_unseeded_rng_is_refused(self):
        stream = Stream(make_tables(groups=("a", "b")))
        with self.assertRaises(PessimismValidationError):
            lcb_greedy(
                group_ids=["a", "b"], total_budget=300, error_bounds=[0.0, 0.0],
                slack=14.0, confidence_delta=0.1, beta_grid=self.grid, r_max=1.0,
                sample_and_score=stream, rng=np.random.RandomState(0),
            )


    def test_misaligned_responses_and_scores(self):
        with self.assertRaises(PessimismValidationError):
            itp_sample(["a"], [0.1, 0.2], 1.0, 0.0, TIGHT, np.random.default_rng(0))

    def test_empty_certified_set_is_refused(self):
        with self.assertRaises(PessimismValidationError):
            group_index([0.1, 0.2], [], 0.0, TIGHT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
