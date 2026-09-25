"""LCB-Greedy allocation over groups with finite-sample ITP oracles.

The paper's rule gives the next batch to the group whose certificate is currently
lowest and makes that batch as large as everything that group already has.  The
optional fixed-batch ablation keeps the same argmin and certificate but refreshes
the index after a constant number of draws.  Doubling remains the default.

Three things are easy to get backwards and are worth stating plainly:

* the budget is global. `total_budget` counts responses across all groups, and
  the run ends when the *sum* of the counts reaches it.
* the argmin is a minimum, not a maximum. The group being helped is the one
  whose guaranteed value is worst, which is the entire fairness content of the
  rule.
* the output is `k` responses, one per group, each drawn from that group's own
  stored candidates after allocation has finished. No response is generated or
  scored during indexing or during the final draw.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np

from .itp import ITPCertification, ITPSampleResult, itp_certify, itp_sample
from .radii import CertificationSchedule
from .validation import (
    PessimismValidationError,
    require_nonnegative_float,
    require_positive_float,
    require_positive_int,
    require_scores,
)

# The paper's argmin is over exact equality. A caller whose scores come from a
# noisy pipeline can widen it, but then the tie set is a modelling choice and
# has to be visible, so it is a named argument with an exact-equality default.
EXACT_TIES = 0.0


@dataclass(frozen=True, eq=False)
class GroupIndex:
    """`L_i(n)`, its certifying beta, and the certification behind it."""

    index_value: float
    beta: float
    certification: ITPCertification
    count: int
    certificates: dict[float, float]

    def as_dict(self) -> dict:
        return {
            "index_value": self.index_value,
            "beta": self.beta,
            "count": self.count,
            "certificates": {str(k): v for k, v in self.certificates.items()},
            "certification": self.certification.as_dict(),
        }


@dataclass(frozen=True, eq=False)
class AllocationStep:
    """One pass of the greedy loop, recorded for audit."""

    step: int
    group: Hashable
    old_count: int
    batch_size: int
    new_count: int
    index_before: float
    index_after: float
    beta_after: float
    tied_groups: tuple
    revealed_before: int
    revealed_after: int
    capped: bool = False
    excluded: tuple = ()

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "group": self.group,
            "old_count": self.old_count,
            "batch_size": self.batch_size,
            "new_count": self.new_count,
            "index_before": self.index_before,
            "index_after": self.index_after,
            "beta_after": self.beta_after,
            "tied_groups": list(self.tied_groups),
            "revealed_before": self.revealed_before,
            "revealed_after": self.revealed_after,
            "capped": self.capped,
            "excluded": list(self.excluded),
        }


@dataclass(frozen=True, eq=False)
class GroupOutcome:
    """What group `i` ended with: its count, its index, and its one response."""

    group: Hashable
    final_count: int
    beta_hat: float
    index_value: float
    certified_value: float
    error_bound: float
    normalizer: float
    certificate: float
    selected_index: int
    selection_probability: float
    selected_response: Any
    selected_score: float
    sample_result: ITPSampleResult = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "final_count": self.final_count,
            "beta_hat": self.beta_hat,
            "index_value": self.index_value,
            "certified_value": self.certified_value,
            "error_bound": self.error_bound,
            "normalizer": self.normalizer,
            "certificate": self.certificate,
            "selected_index": self.selected_index,
            "selection_probability": self.selection_probability,
            "selected_score": self.selected_score,
        }


@dataclass(frozen=True, eq=False)
class LCBGreedyResult:
    """The mechanism's output plus everything needed to audit the run."""

    outcomes: dict[Hashable, GroupOutcome]
    schedule: CertificationSchedule
    history: list[AllocationStep]
    final_counts: dict[Hashable, int]
    total_revealed: int
    initial_indices: dict[Hashable, float] = field(default_factory=dict)
    seed_sequence_entropy: Any = None

    @property
    def selected_responses(self) -> dict[Hashable, Any]:
        return {key: outcome.selected_response for key, outcome in self.outcomes.items()}

    def as_dict(self) -> dict:
        return {
            "schedule": self.schedule.as_dict(),
            "final_counts": {str(k): v for k, v in self.final_counts.items()},
            "total_revealed": self.total_revealed,
            "initial_indices": {str(k): v for k, v in self.initial_indices.items()},
            "groups": [outcome.as_dict() for outcome in self.outcomes.values()],
            "history": [step.as_dict() for step in self.history],
            "seed_sequence_entropy": self.seed_sequence_entropy,
        }


def _pin_betas(fixed_beta, groups) -> Optional[dict]:
    """Normalize a scalar or per-group beta into one single-point grid per group."""
    if fixed_beta is None:
        return None
    if isinstance(fixed_beta, dict):
        missing = [g for g in groups if g not in fixed_beta]
        if missing:
            raise PessimismValidationError(
                f"fixed_beta has no entry for {missing}"
            )
        return {
            g: np.array([require_positive_float(fixed_beta[g], f"fixed_beta[{g}]")])
            for g in groups
        }
    value = np.array([require_positive_float(fixed_beta, "fixed_beta")])
    return {g: value for g in groups}


def group_index(
    scores: Sequence[float],
    certified: Sequence[float],
    error_bound: float,
    bisection_tol: float,
    penalty_scale: float = 1.0,
    error_scale: float = 1.0,
    index_error_scale: Optional[float] = None,
    error_decay: float = 0.0,
    decay_count: int = 1,
) -> GroupIndex:
    """INDEX(i, n): maximize the certificate over the *certified* betas only.

    Maximizing over the whole grid and filtering afterwards would be a
    different quantity: an uncertified beta can carry a larger empirical
    certificate precisely because its radius has not shrunk enough to be
    trusted, and letting it win would put an untrustworthy number into the
    cross-group comparison.

    Ties in beta go to the smallest maximizer. This is an implementation
    convention chosen for reproducibility; the group tie-breaking in Algorithm 1
    is genuinely uniform-random and is handled by the caller.

    `error_scale` and `index_error_scale` are the same knob used for two
    different jobs, and they are separable because the two jobs happen at
    different moments. `error_scale` decides which beta maximizes the
    certificate, and beta is the only channel through which this function
    reaches the within-group selection law. `index_error_scale` decides what
    number the chosen beta reports upward, and that number is what Algorithm 1
    compares across groups. Leaving `index_error_scale` unset ties them
    together, which is the original behaviour: the returned index is then
    `best.certificate` itself, bit for bit.
    """
    array = require_scores(scores)
    grid = np.asarray(certified, dtype=np.float64)
    if grid.size == 0:
        raise PessimismValidationError(
            "INDEX was called with an empty certified set; no beta is "
            "certifiable at this sample count"
        )
    best: Optional[ITPCertification] = None
    certificates: dict[float, float] = {}
    for beta in grid:
        certify_options: dict = {}
        if penalty_scale != 1.0:
            certify_options["penalty_scale"] = penalty_scale
        if error_scale != 1.0:
            certify_options["error_scale"] = error_scale
        if error_decay != 0.0:
            certify_options["error_decay"] = error_decay
            certify_options["decay_count"] = decay_count
        certification = itp_certify(
            array, float(beta), error_bound, bisection_tol, **certify_options
        )
        certificates[float(beta)] = certification.certificate
        # Strict improvement only, and the grid is ascending, so the smallest
        # maximizing beta is the one that survives.
        if best is None or certification.certificate > best.certificate:
            best = certification
    assert best is not None
    # The beta is settled; only the number it reports upward can still change.
    # `certification` stays the one that picked the beta, because that is the
    # object `itp_sample` re-derives at selection time and checks against.
    if index_error_scale is None or index_error_scale == error_scale:
        index_value = best.certificate
    else:
        index_options: dict = {}
        if penalty_scale != 1.0:
            index_options["penalty_scale"] = penalty_scale
        if index_error_scale != 1.0:
            index_options["error_scale"] = index_error_scale
        if error_decay != 0.0:
            index_options["error_decay"] = error_decay
            index_options["decay_count"] = decay_count
        index_value = itp_certify(
            array, best.beta, error_bound, bisection_tol, **index_options
        ).certificate
    return GroupIndex(
        index_value=index_value,
        beta=best.beta,
        certification=best,
        count=int(array.size),
        certificates=certificates,
    )


def lcb_greedy(
    group_ids: Sequence[Hashable],
    total_budget: int,
    error_bounds: Sequence[float],
    slack: float,
    confidence_delta: float,
    beta_grid: Sequence[float],
    r_max: float,
    sample_and_score: Callable[[Hashable, int], tuple[Sequence[Any], Sequence[float]]],
    rng: np.random.Generator,
    bisection_tol: Optional[float] = None,
    tie_tolerance: float = EXACT_TIES,
    capacity: Optional[dict] = None,
    fixed_beta: Optional[float] = None,
    allocation_batch_size: Optional[int] = None,
    penalty_scale: float = 1.0,
    error_scale: float = 1.0,
    index_error_scale: Optional[float] = None,
    error_decay: float = 0.0,
    min_initial_count: Optional[int] = None,
) -> LCBGreedyResult:
    """Run Algorithm 1 and return one response per group with full diagnostics.

    `sample_and_score(group, m)` must return exactly `m` fresh responses and
    their proxy scores, or raise. For a stored pool it reveals the next `m`
    unseen rows of a fixed order; the allocation must never see a score it has
    not paid for, which is why the callback is a stream rather than an index
    into a table.
    """
    groups = list(group_ids)
    if len(groups) == 0:
        raise PessimismValidationError("lcb_greedy needs at least one group")
    if len(set(groups)) != len(groups):
        raise PessimismValidationError("group_ids must be distinct")
    if len(error_bounds) != len(groups):
        raise PessimismValidationError(
            f"{len(error_bounds)} error bounds for {len(groups)} groups"
        )
    bounds = {
        group: require_nonnegative_float(bound, f"error_bound[{group}]")
        for group, bound in zip(groups, error_bounds)
    }
    total_budget = require_positive_int(total_budget, "total_budget")
    penalty_scale = require_nonnegative_float(penalty_scale, "penalty_scale")
    fixed_batch = (
        None
        if allocation_batch_size is None
        else require_positive_int(allocation_batch_size, "allocation_batch_size")
    )
    tie_tolerance = require_nonnegative_float(tie_tolerance, "tie_tolerance")
    if not isinstance(rng, np.random.Generator):
        raise PessimismValidationError(
            "rng must be a numpy.random.Generator so the run is reproducible"
        )

    if capacity is not None:
        missing = [g for g in groups if g not in capacity]
        if missing:
            raise PessimismValidationError(f"capacity is missing groups: {missing}")
        capacity = {g: require_positive_int(capacity[g], f"capacity[{g}]") for g in groups}
        if sum(capacity.values()) < total_budget:
            raise PessimismValidationError(
                f"total_budget={total_budget} exceeds the pool's total capacity "
                f"{sum(capacity.values())}"
            )

    schedule = CertificationSchedule.build(
        num_groups=len(groups),
        total_budget=total_budget,
        beta_grid=beta_grid,
        r_max=r_max,
        slack=slack,
        confidence_delta=confidence_delta,
        bisection_tol=bisection_tol,
        min_initial_count=min_initial_count,
    )
    tol = schedule.bisection_tol

    stored_responses: dict[Hashable, list[Any]] = {group: [] for group in groups}
    stored_scores: dict[Hashable, list[float]] = {group: [] for group in groups}
    counts: dict[Hashable, int] = {group: 0 for group in groups}
    indices: dict[Hashable, GroupIndex] = {}
    certified_cache: dict[int, np.ndarray] = {}

    def certified_at(n: int) -> np.ndarray:
        if n not in certified_cache:
            certified_cache[n] = schedule.certified(n)
        return certified_cache[n]

    def reveal(group: Hashable, requested: int) -> None:
        """Pull exactly `requested` fresh rows, checking before committing."""
        responses, scores = sample_and_score(group, requested)
        responses = list(responses)
        values = require_scores(scores, schedule.r_max, name=f"scores for {group}")
        if len(responses) != requested or values.size != requested:
            raise PessimismValidationError(
                f"sample_and_score({group!r}, {requested}) returned "
                f"{len(responses)} responses and {values.size} scores"
            )
        stored_responses[group].extend(responses)
        stored_scores[group].extend(float(value) for value in values)
        counts[group] += requested

    # `fixed_beta` is either one beta for every group or a per-group mapping.
    # The per-group form exists because the best beta is a property of a group's
    # own score distribution, and pinning all five to one value is a modelling
    # choice rather than a consequence of the method.
    pinned = _pin_betas(fixed_beta, groups)

    def recompute(group: Hashable) -> GroupIndex:
        n = counts[group]
        index_options: dict = {}
        if penalty_scale != 1.0:
            index_options["penalty_scale"] = penalty_scale
        if error_scale != 1.0:
            index_options["error_scale"] = error_scale
        if index_error_scale is not None:
            index_options["index_error_scale"] = index_error_scale
        if error_decay != 0.0:
            index_options["error_decay"] = error_decay
            index_options["decay_count"] = schedule.initial_count
        indices[group] = group_index(
            stored_scores[group][:n],
            certified_at(n) if pinned is None else pinned[group],
            bounds[group],
            tol,
            **index_options,
        )
        return indices[group]

    if capacity is not None:
        short = {g: capacity[g] for g in groups if capacity[g] < schedule.initial_count}
        if short:
            raise PessimismValidationError(
                f"initial_count={schedule.initial_count} exceeds the stored "
                f"capacity of {short}; initialization cannot complete."
            )

    # --- Initialization: every group is given initial_count draws. ------------
    for group in groups:
        reveal(group, schedule.initial_count)
        recompute(group)

    initial_indices = {group: indices[group].index_value for group in groups}

    # --- Greedy loop: help the worst-certified group. -------------------------
    history: list[AllocationStep] = []
    used = sum(counts.values())
    step = 0
    while used < total_budget:
        step += 1
        if capacity is None:
            eligible = list(groups)
            excluded: tuple = ()
        else:
            eligible = [g for g in groups if counts[g] < capacity[g]]
            excluded = tuple(g for g in groups if counts[g] >= capacity[g])
            if not eligible:
                raise PessimismValidationError(
                    f"Every group is at its stored capacity with {used} of "
                    f"{total_budget} revealed: the budget exceeds the pool."
                )
        values = np.array([indices[group].index_value for group in eligible])
        minimum = float(values.min())
        tied = [
            group
            for group, value in zip(eligible, values)
            if value <= minimum + tie_tolerance
        ]
        chosen = tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]

        old_count = counts[chosen]
        index_before = indices[chosen].index_value
        proposed_batch = old_count if fixed_batch is None else fixed_batch
        batch_size = min(proposed_batch, total_budget - used)
        if capacity is not None:
            batch_size = min(batch_size, capacity[chosen] - old_count)
        capped = batch_size < min(proposed_batch, total_budget - used)
        reveal(chosen, batch_size)
        updated = recompute(chosen)
        used += batch_size
        history.append(
            AllocationStep(
                step=step,
                group=chosen,
                old_count=old_count,
                batch_size=batch_size,
                new_count=counts[chosen],
                index_before=index_before,
                index_after=updated.index_value,
                beta_after=updated.beta,
                tied_groups=tuple(tied),
                revealed_before=used - batch_size,
                revealed_after=used,
                capped=capped,
                excluded=excluded,
            )
        )

    if used != total_budget:
        raise PessimismValidationError(
            f"Allocation revealed {used} responses for a budget of {total_budget}"
        )

    # --- One categorical draw per group, from stored candidates only. --------
    outcomes: dict[Hashable, GroupOutcome] = {}
    for group in groups:
        n = counts[group]
        current = indices[group]
        sample_options: dict = {}
        if penalty_scale != 1.0:
            sample_options["penalty_scale"] = penalty_scale
        if error_scale != 1.0:
            sample_options["error_scale"] = error_scale
        if error_decay != 0.0:
            sample_options["error_decay"] = error_decay
            sample_options["decay_count"] = schedule.initial_count
        sampled = itp_sample(
            responses=stored_responses[group][:n],
            scores=stored_scores[group][:n],
            beta=current.beta,
            error_bound=bounds[group],
            bisection_tol=tol,
            rng=rng,
            **sample_options,
        )
        # The cached certification and the one recomputed inside itp_sample come
        # from the same scores and beta, so they must agree bit for bit; a
        # mismatch means the stored state drifted from the index. The comparison
        # is against the certificate that *picked* beta, not against the number
        # reported upward: with `index_error_scale` set the two are deliberately
        # different, and checking the reported one would turn that into a crash.
        if not np.isclose(
            sampled.certificate, current.certification.certificate,
            rtol=0.0, atol=1e-12,
        ):
            raise PessimismValidationError(
                f"Group {group!r}: cached certificate "
                f"{current.certification.certificate!r} does not match the one "
                f"recomputed at sampling time {sampled.certificate!r}"
            )
        outcomes[group] = GroupOutcome(
            group=group,
            final_count=n,
            beta_hat=current.beta,
            index_value=current.index_value,
            certified_value=current.index_value - schedule.slack,
            error_bound=bounds[group],
            normalizer=sampled.normalizer,
            certificate=sampled.certificate,
            selected_index=sampled.selected_index,
            selection_probability=sampled.selection_probability,
            selected_response=sampled.selected_response,
            selected_score=stored_scores[group][sampled.selected_index],
            sample_result=sampled,
        )

    return LCBGreedyResult(
        outcomes=outcomes,
        schedule=schedule,
        history=history,
        final_counts=dict(counts),
        total_revealed=used,
        initial_indices=initial_indices,
    )
