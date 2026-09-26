"""Allocation and selection baselines for Greedy-Pessimistic allocation.

None of this is part of Algorithm 1 or Algorithm 2. It exists so that a run can
answer "compared with what?": the allocation rule and the BGP selection rule are
two separate choices, and only a baseline that changes one of them at a time
says which one a difference came from.

* `uniform_bgp` keeps the BGP oracle and replaces the adaptive allocation with
  an equal split. A difference against it is attributable to the allocation.
* `uniform_argmax` keeps the equal split and replaces the BGP draw with
  Best-of-N (BoN) selection. A difference against it is attributable to the
  selection rule.
* `greedy_argmax` implements Greedy-BoN: one draw per group, then every remaining
  unit to whichever group's current Best-of-N score is lowest. It changes both
  the allocation and the selection relative to Greedy-Pessimistic.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np

from .core.bgp import bgp_sample
from .core.greedy_pessimistic import GroupOutcome, group_index
from .core.radii import CertificationSchedule
from .core.validation import PessimismValidationError


@dataclass(frozen=True, eq=False)
class BaselineResult:
    outcomes: dict[Hashable, GroupOutcome]
    final_counts: dict[Hashable, int]
    total_revealed: int
    schedule: Optional[CertificationSchedule] = None
    rule: str = ""


def even_split(total_budget: int, num_groups: int) -> list[int]:
    """`total_budget` shared out as evenly as an integer split allows.

    The remainder goes to the first groups rather than being dropped, so the
    split spends the same global budget the adaptive rule does; with the budgets
    used here the remainder is at most `k - 1` responses.
    """
    base, remainder = divmod(total_budget, num_groups)
    if base <= 0:
        raise PessimismValidationError(
            f"total_budget={total_budget} cannot give {num_groups} groups a draw"
        )
    return [base + (1 if i < remainder else 0) for i in range(num_groups)]


def uniform_bgp(
    group_ids: Sequence[Hashable],
    total_budget: int,
    error_bounds: Sequence[float],
    schedule: CertificationSchedule,
    sample_and_score: Callable[[Hashable, int], tuple[Sequence[Any], Sequence[float]]],
    rng: np.random.Generator,
    fixed_beta: Optional[float] = None,
    penalty_scale: float = 1.0,
    error_scale: float = 1.0,
    error_decay: float = 0.0,
) -> BaselineResult:
    """Equal allocation, then the same INDEX and BGP-Sample per group.

    ``fixed_beta`` mirrors :func:`greedy_pessimistic`: it deliberately bypasses the
    certified beta set and is therefore an empirical/uncertified ablation.
    """
    groups = list(group_ids)
    shares = even_split(total_budget, len(groups))
    bounds = dict(zip(groups, (float(bound) for bound in error_bounds)))
    outcomes: dict[Hashable, GroupOutcome] = {}
    counts: dict[Hashable, int] = {}
    for group, share in zip(groups, shares):
        responses, scores = sample_and_score(group, share)
        responses = list(responses)
        values = np.asarray(scores, dtype=np.float64)
        certified = schedule.certified(share)
        if certified.size == 0:
            raise PessimismValidationError(
                f"An equal split gives each group {share} responses, which "
                "certifies no beta; the comparison is not defined at this budget"
            )
        candidates = (
            certified if fixed_beta is None
            else [fixed_beta[group] if isinstance(fixed_beta, dict) else fixed_beta]
        )
        scale_options: dict = {}
        if penalty_scale != 1.0:
            scale_options["penalty_scale"] = penalty_scale
        if error_scale != 1.0:
            scale_options["error_scale"] = error_scale
        if error_decay != 0.0:
            scale_options["error_decay"] = error_decay
            scale_options["decay_count"] = schedule.initial_count
        index = group_index(
            values, candidates, bounds[group], schedule.bisection_tol,
            **scale_options,
        )
        sampled = bgp_sample(
            responses, values, index.beta, bounds[group], schedule.bisection_tol,
            rng, **scale_options,
        )
        counts[group] = share
        outcomes[group] = GroupOutcome(
            group=group,
            final_count=share,
            beta_hat=index.beta,
            index_value=index.index_value,
            certified_value=index.index_value - schedule.slack,
            error_bound=bounds[group],
            normalizer=sampled.normalizer,
            certificate=sampled.certificate,
            selected_index=sampled.selected_index,
            selection_probability=sampled.selection_probability,
            selected_response=sampled.selected_response,
            selected_score=float(values[sampled.selected_index]),
            sample_result=sampled,
        )
    return BaselineResult(
        outcomes=outcomes,
        final_counts=counts,
        total_revealed=sum(counts.values()),
        schedule=schedule,
        rule="uniform_bgp",
    )


def greedy_argmax(
    group_ids: Sequence[Hashable],
    total_budget: int,
    sample_and_score: Callable[[Hashable, int], tuple[Sequence[Any], Sequence[float]]],
    capacity: Optional[dict] = None,
) -> dict[Hashable, dict[str, Any]]:
    """Greedy-BoN allocation and selection on the same ordered stream.

    Every group starts with one draw; each remaining unit of budget goes to the
    group whose current Best-of-N score is lowest, ties to the lowest group index.
    The scores are already on the shared `[0, R_max]` axis when the adapter has
    rescaled them, which is what makes the cross-group minimum meaningful.

    Unlike Algorithm 1 this reveals one response at a time, so it sees strictly
    more information per unit of budget than the doubling rule does. That is a
    real advantage for the incumbent and is not corrected for.

    `capacity` skips groups that have exhausted their stored pool, the same
    finite-pool deviation `greedy_pessimistic` takes. Without it this rule pours the
    whole budget into one group and overruns a 256-response pool at any budget
    above about 260.
    """
    groups = list(group_ids)
    if total_budget < len(groups):
        raise PessimismValidationError(
            f"total_budget={total_budget} cannot give {len(groups)} groups a draw"
        )
    stored: dict[Hashable, list[Any]] = {}
    values: dict[Hashable, list[float]] = {}
    best: dict[Hashable, int] = {}
    for group in groups:
        responses, scores = sample_and_score(group, 1)
        stored[group] = list(responses)
        values[group] = [float(v) for v in np.asarray(scores, dtype=np.float64)]
        best[group] = 0
    order = {group: index for index, group in enumerate(groups)}
    capped_steps = 0
    for _ in range(total_budget - len(groups)):
        eligible = (
            groups if capacity is None
            else [g for g in groups if len(values[g]) < capacity[g]]
        )
        if not eligible:
            raise PessimismValidationError(
                "Every group is at its stored capacity: the budget exceeds the pool."
            )
        if capacity is not None and len(eligible) < len(groups):
            capped_steps += 1
        target = min(eligible, key=lambda g: (values[g][best[g]], order[g]))
        responses, scores = sample_and_score(target, 1)
        stored[target].extend(responses)
        value = float(np.asarray(scores, dtype=np.float64)[0])
        values[target].append(value)
        if value > values[target][best[target]]:
            best[target] = len(values[target]) - 1
    return {
        group: {
            "final_count": len(values[group]),
            "selected_index": best[group],
            "selected_response": stored[group][best[group]],
            "selected_score": values[group][best[group]],
            "capped_steps": capped_steps,
        }
        for group in groups
    }


def uniform_argmax(
    group_ids: Sequence[Hashable],
    total_budget: int,
    sample_and_score: Callable[[Hashable, int], tuple[Sequence[Any], Sequence[float]]],
) -> dict[Hashable, dict[str, Any]]:
    """Equal allocation, then Best-of-N selection. Ties go to the earlier draw."""
    groups = list(group_ids)
    shares = even_split(total_budget, len(groups))
    picked: dict[Hashable, dict[str, Any]] = {}
    for group, share in zip(groups, shares):
        responses, scores = sample_and_score(group, share)
        values = np.asarray(scores, dtype=np.float64)
        index = int(np.argmax(values))
        picked[group] = {
            "final_count": share,
            "selected_index": index,
            "selected_response": list(responses)[index],
            "selected_score": float(values[index]),
        }
    return picked
