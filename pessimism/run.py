"""Run the six allocation/selection comparisons on portable ordered pools."""

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import even_split, greedy_argmax, uniform_argmax, uniform_itp
from .core.lcb_greedy import lcb_greedy
from .core.radii import CertificationSchedule, delta_total, log_factor
from .core.validation import (
    PessimismValidationError, require_beta_grid, require_confidence_delta,
    require_nonnegative_float, require_positive_float, require_positive_int,
)
from .input_data import CandidatePool, ProxyStream, load_pool, parse_pool, synthetic_document


METHODS = (
    "uniform_argmax", "greedy_argmax", "uniform_itp_auto", "uniform_itp_fixed",
    "lcb_greedy_auto", "lcb_greedy_fixed",
)
DEFAULT_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def _metrics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "gap": float(values.max() - values.min())}


def run_comparison(
    pool: CandidatePool, *, methods=METHODS, budget=60, seed=0,
    beta_grid=DEFAULT_GRID, fixed_beta=0.1, alpha=1.0, initial_count=2,
    slack=None, confidence_delta=0.05, bisection_tol=None,
    capacity_mode="capped", allocation_batch_size=None, index_error_scale=None,
):
    """Judge scores enter only after allocation and final selection finish.

    All methods see identical prefixes of the caller's order. Each method gets
    a fresh stream and a generator initialized to the same seed, so requesting
    another method never changes an existing method's result.
    """
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise PessimismValidationError("methods must be distinct supported method names")
    budget = require_positive_int(budget, "budget")
    initial_count = require_positive_int(initial_count, "initial_count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PessimismValidationError("seed must be a nonnegative integer")
    alpha = require_nonnegative_float(alpha, "alpha")
    fixed_beta = require_positive_float(fixed_beta, "fixed_beta")
    grid = require_beta_grid(beta_grid)
    confidence_delta = require_confidence_delta(confidence_delta)
    if slack is not None:
        slack = require_positive_float(slack, "slack")
    if bisection_tol is not None:
        bisection_tol = require_positive_float(bisection_tol, "bisection_tol")
    if allocation_batch_size is not None:
        allocation_batch_size = require_positive_int(allocation_batch_size, "allocation_batch_size")
    if index_error_scale is not None:
        index_error_scale = require_nonnegative_float(index_error_scale, "index_error_scale")
    if capacity_mode not in ("capped", "uncapped"):
        raise PessimismValidationError("capacity_mode must be capped or uncapped")
    groups = [group.id for group in pool.groups]
    bounds = [group.error_bound for group in pool.groups]
    capacities = pool.capacity
    if budget < len(groups) or budget > sum(capacities.values()):
        raise PessimismValidationError("budget must cover all groups and fit the total pool capacity")
    if any(method.startswith("uniform_") for method in methods):
        shares = even_split(budget, len(groups))
        if any(share > capacities[g] for g, share in zip(groups, shares)):
            raise PessimismValidationError(
                "equal allocation exceeds a group's capacity; uniform methods do not redistribute"
            )
    needs_itp = any("itp" in method or method.startswith("lcb_") for method in methods)
    schedule = None
    automatic_slack = slack is None
    if needs_itp:
        if slack is None:
            ell = log_factor(len(groups), len(grid), budget, confidence_delta)
            slack = delta_total(initial_count, float(grid[-1]), pool.r_max, ell)
        schedule = CertificationSchedule.build(
            len(groups), budget, grid, pool.r_max, slack, confidence_delta,
            bisection_tol=bisection_tol, min_initial_count=initial_count,
        )
        if any(method.startswith("lcb_") for method in methods):
            if any(cap < schedule.initial_count for cap in capacities.values()):
                raise PessimismValidationError("initial_count exceeds a group's capacity")

    results = {}
    for method in methods:
        stream = ProxyStream(pool)
        rng = np.random.default_rng(seed)
        pinned = fixed_beta if method.endswith("_fixed") else None
        history = []
        if method == "uniform_argmax":
            result = uniform_argmax(groups, budget, stream)
        elif method == "greedy_argmax":
            result = greedy_argmax(groups, budget, stream,
                                   capacity=capacities if capacity_mode == "capped" else None)
        elif method.startswith("uniform_itp_"):
            result = uniform_itp(groups, budget, bounds, schedule, stream, rng,
                                 fixed_beta=pinned, error_scale=alpha)
        else:
            result = lcb_greedy(
                groups, budget, bounds, slack, confidence_delta, grid, pool.r_max,
                stream, rng, bisection_tol=schedule.bisection_tol,
                capacity=capacities if capacity_mode == "capped" else None,
                fixed_beta=pinned, allocation_batch_size=allocation_batch_size,
                error_scale=alpha, index_error_scale=index_error_scale,
                min_initial_count=initial_count,
            )
            history = [step.as_dict() for step in result.history]
        is_itp = method not in ("uniform_argmax", "greedy_argmax")
        rows = []
        # This is the first point where evaluation scores are accessed.
        for group in pool.groups:
            if is_itp:
                outcome = result.outcomes[group.id]
                count, selected = outcome.final_count, outcome.selected_index
                probabilities = outcome.sample_result.probabilities
                details = {
                    "beta": outcome.beta_hat, "index_value": outcome.index_value,
                    "certificate_minus_slack": outcome.certified_value,
                    "certificate": outcome.certificate,
                    "effective_error": outcome.sample_result.certification.effective_error,
                }
            else:
                outcome = result[group.id]
                count, selected = outcome["final_count"], outcome["selected_index"]
                probabilities = np.zeros(count, dtype=np.float64)
                probabilities[selected] = 1.0
                details = {}
            proxy = np.asarray(group.proxy_scores[:count])
            row = {
                "group": group.id, "count": count,
                "selected_index": selected, "selected_candidate": group.candidate_ids[selected],
                "selected_proxy": float(proxy[selected]),
                "expected_proxy": float(np.dot(probabilities, proxy)),
                "selection_probabilities": probabilities.tolist(),
                **details,
            }
            if group.judge_scores is not None:
                judge = np.asarray(group.judge_scores[:count])
                row["selected_judge"] = float(judge[selected])
                row["expected_judge"] = float(np.dot(probabilities, judge))
            rows.append(row)
        counts = {row["group"]: row["count"] for row in rows}
        if sum(counts.values()) != budget or counts != stream.counts:
            raise RuntimeError("allocation budget accounting is inconsistent")
        metric_names = ["expected_proxy", "selected_proxy"]
        if pool.groups[0].judge_scores is not None:
            metric_names += ["expected_judge", "selected_judge"]
        caveats = [
            "Expected scores integrate the final selection law over revealed candidates, "
            "conditional on this allocation; they are not population reward estimates.",
        ]
        if is_itp:
            caveats.append(
                "Certificates require valid external L2 error bounds, bounded rewards, "
                "the stated sampling assumptions, and the algorithm's theoretical settings. "
                "A stored pool or Judge score alone does not establish these assumptions."
            )
            if automatic_slack:
                caveats.append("Slack was derived from initial_count and the largest grid beta; it is recorded in the schedule.")
            if schedule.guarantee_is_vacuous():
                caveats.append("Slack is at least R_max: the reported certificate minus slack is vacuous.")
            if pinned is not None:
                caveats.append("Fixed beta bypasses the certified grid: this is an empirical comparison.")
            if alpha != 1:
                caveats.append("Alpha rescales the supplied L2 error bound; the original certificate assumptions no longer apply without a justified rescaled bound.")
            if method.startswith("lcb_") and index_error_scale is not None and index_error_scale != alpha:
                caveats.append("Index error scaling is decoupled from beta selection: empirical ablation.")
            if method.startswith("lcb_") and allocation_batch_size is not None:
                caveats.append("Constant allocation batches replace doubling: empirical ablation.")
            if schedule.initial_count > schedule.certifiable_count:
                caveats.append("The requested initialization floor exceeds the minimum certifiable count.")
        capped_steps = (
            sum(bool(step["capped"] or step["excluded"]) for step in history)
            if history else
            (max(value.get("capped_steps", 0) for value in result.values()) if not is_itp else 0)
        )
        if capacity_mode == "capped" and method.startswith(("greedy_", "lcb_")):
            caveats.append("Finite-pool capacity checks can exclude exhausted groups or shorten a batch; this differs from unlimited sampling.")
        method_result = {
            "total_revealed": sum(counts.values()), "final_counts": counts,
            "groups": rows,
            "metrics": {name: _metrics([row[name] for row in rows]) for name in metric_names},
            "capacity_affected_steps": capped_steps, "caveats": caveats,
        }
        if is_itp:
            method_result["schedule"] = schedule.as_dict()
        if method.startswith("lcb_"):
            method_result["allocation_history"] = history
        results[method] = method_result
    return {
        "format_version": 1,
        "settings": {"budget": budget, "seed": seed, "r_max": pool.r_max,
                     "alpha": alpha, "fixed_beta": fixed_beta, "beta_grid": grid.tolist(),
                     "requested_initial_count": initial_count, "slack": slack,
                     "slack_source": "derived_from_initial_count" if needs_itp and automatic_slack else "explicit" if slack is not None else "unused",
                     "confidence_delta": confidence_delta, "capacity_mode": capacity_mode,
                     "capacities": capacities, "allocation_batch_size": allocation_batch_size,
                     "index_error_scale": index_error_scale,
                     "judge_used_for_allocation": False},
        "methods": results,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Portable JSON candidate pool supplied by the caller")
    source.add_argument("--synthetic", action="store_true", help="Use generated demonstration data")
    parser.add_argument("--synthetic-groups", type=int, default=3)
    parser.add_argument("--synthetic-candidates", type=int, default=64)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budget", type=int, default=60, help="Global count, including initialization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", "--error-scale", type=float, default=1.0,
                        help="Multiply group L2 error bounds before forming the penalty")
    parser.add_argument("--initial-count", type=int, default=2, help="Minimum initial responses per LCB group")
    parser.add_argument("--fixed-beta", type=float, default=0.1)
    parser.add_argument("--beta-grid", default=",".join(map(str, DEFAULT_GRID)), help="Strictly increasing comma-separated positive values")
    parser.add_argument("--slack", type=float, help="If omitted, derive from initial count and largest beta (often vacuous)")
    parser.add_argument("--delta", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--bisection-tol", type=float)
    parser.add_argument("--capacity-mode", choices=("capped", "uncapped"), default="capped")
    parser.add_argument("--allocation-batch-size", type=int, help="Empirical constant-batch ablation; default is doubling")
    parser.add_argument("--index-error-scale", type=float, help="Empirical decoupled index-error scaling for LCB")
    parser.add_argument("--output", type=Path, help="Write JSON here; default is stdout")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pool = load_pool(args.input) if args.input is not None else parse_pool(
            synthetic_document(args.synthetic_groups, args.synthetic_candidates, args.synthetic_seed)
        )
        result = run_comparison(
            pool, methods=args.methods, budget=args.budget, seed=args.seed,
            beta_grid=[float(value) for value in args.beta_grid.split(",")],
            fixed_beta=args.fixed_beta, alpha=args.alpha, initial_count=args.initial_count,
            slack=args.slack, confidence_delta=args.delta, bisection_tol=args.bisection_tol,
            capacity_mode=args.capacity_mode, allocation_batch_size=args.allocation_batch_size,
            index_error_scale=args.index_error_scale,
        )
        result["source"] = "synthetic" if args.synthetic else "caller_supplied_pool"
        if args.synthetic:
            result["synthetic_settings"] = {
                "groups": args.synthetic_groups, "candidates_per_group": args.synthetic_candidates,
                "seed": args.synthetic_seed,
            }
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
