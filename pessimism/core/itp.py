"""Algorithm 2: the finite-sample ITP oracle for one group.

Both entry points read only stored scores. Nothing here draws a response,
calls a reward model, or knows what the responses are; `itp_sample` carries the
response objects along solely so that the categorical draw can return one.

The bisection returns the *lower* endpoint of the final bracket, not its
midpoint. That is the whole reason the weights are usable: `h` is nonincreasing
and `h(lambda_lo) >= 0` is an invariant of the loop, so the returned normalizer
sits at or below the true root and the weights it produces have mean at least
one. A midpoint would be off the safe side of the root roughly half the time
and the one-sided normalization the paper's perturbation bound rests on would
be gone.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .validation import (
    WEIGHT_ABSOLUTE_TOLERANCE,
    PessimismValidationError,
    require_nonnegative_float,
    require_positive_float,
    require_scores,
)

# The loop is bounded by the bracket width, which halves every pass; the margin
# absorbs the case where float64 rounding of the midpoint leaves the bracket a
# fraction wider than the ideal halving.
BISECTION_ITERATION_MARGIN = 8


@dataclass(frozen=True, eq=False)
class ITPCertification:
    """The output of ITP-Certify: normalizer, weights, and the certificate."""

    normalizer: float
    weights: np.ndarray
    certificate: float
    mean_weight: float
    mean_square_weight: float
    num_bisection_iterations: int
    beta: float
    error_bound: float
    penalty_scale: float
    error_scale: float
    effective_error: float
    objective: float
    raw_penalty: float
    used_penalty: float
    bisection_tol: float
    count: int
    bracket: tuple[float, float]

    def as_dict(self) -> dict:
        return {
            "normalizer": self.normalizer,
            "certificate": self.certificate,
            "mean_weight": self.mean_weight,
            "mean_square_weight": self.mean_square_weight,
            "num_bisection_iterations": self.num_bisection_iterations,
            "beta": self.beta,
            "error_bound": self.error_bound,
            "error_scale": self.error_scale,
            "effective_error": self.effective_error,
            "penalty_scale": self.penalty_scale,
            "objective": self.objective,
            "raw_penalty": self.raw_penalty,
            "used_penalty": self.used_penalty,
            "bisection_tol": self.bisection_tol,
            "count": self.count,
        }


@dataclass(frozen=True, eq=False)
class ITPSampleResult:
    """One categorical draw from a group's own stored candidates."""

    selected_response: Any
    selected_index: int
    selection_probability: float
    probabilities: np.ndarray
    certification: ITPCertification

    @property
    def certificate(self) -> float:
        return self.certification.certificate

    @property
    def normalizer(self) -> float:
        return self.certification.normalizer

    @property
    def weights(self) -> np.ndarray:
        return self.certification.weights


def _certify_array(
    scores: np.ndarray,
    beta: float,
    error_bound: float,
    bisection_tol: float,
    penalty_scale: float,
    error_scale: float = 1.0,
    error_decay: float = 0.0,
    decay_count: int = 1,
) -> ITPCertification:
    """ITP-Certify on scores already validated by the caller.

    Two independent knobs shrink the reward-mismatch penalty, and they are not
    the same operation. `penalty_scale` multiplies the finished penalty, so
    both of its terms shrink linearly. `error_scale` multiplies epsilon before
    the penalty is formed, so the `e^2/(2 beta)` term shrinks quadratically --
    which is the term that diverges as beta falls, and therefore the one that
    decides whether small beta can ever win. Scaling epsilon is also the only
    one of the two that corresponds to an assumption anyone can state: that the
    proxy is closer to the true reward than the measured bound says.
    """
    lambda_lo = -beta
    lambda_hi = float(scores.max())
    # h(lambda) = mean(relu((scores - lambda) / beta)) - 1, nonincreasing, with
    # h(-beta) = mean(scores)/beta >= 0 and h(max score) = -1: the root is
    # bracketed before the first iteration.
    inverse_beta = 1.0 / beta

    def h(value: float) -> float:
        return float(np.mean(np.maximum(scores - value, 0.0))) * inverse_beta - 1.0

    width = lambda_hi - lambda_lo
    max_iterations = (
        math.ceil(math.log2(width / bisection_tol)) + BISECTION_ITERATION_MARGIN
        if width > bisection_tol
        else 0
    )
    iterations = 0
    while lambda_hi - lambda_lo > bisection_tol:
        lambda_mid = 0.5 * (lambda_lo + lambda_hi)
        # Float64 has run out of room to split the bracket; another pass would
        # loop forever without narrowing it.
        if lambda_mid <= lambda_lo or lambda_mid >= lambda_hi:
            break
        if h(lambda_mid) > 0.0:
            lambda_lo = lambda_mid
        else:
            lambda_hi = lambda_mid
        iterations += 1
        if iterations >= max_iterations:
            break

    normalizer = lambda_lo
    weights = np.maximum(scores - normalizer, 0.0) * inverse_beta
    mean_weight = float(weights.mean())
    mean_square_weight = float(np.mean(weights * weights))
    objective = normalizer + 0.5 * beta * (1.0 + mean_square_weight)
    # `error_decay` shrinks the bound with the sample count:
    #     eps(n) = error_scale * eps * (decay_count / n) ** error_decay
    # This is a DEPARTURE from what the bound means. `bar_epsilon_i` bounds
    # sqrt(E[(r_hat - r*)^2]), a property of the reward model that no amount of
    # sampling reduces; the finite-sample part of the guarantee is carried
    # separately by the certification radius. The knob exists because the
    # penalty's n-independence is what makes the arg min lock onto one group at
    # alpha = 1, and shrinking it with n turns that positive feedback into a
    # negative one. Any run with error_decay != 0 is outside the stated theory.
    effective_error = (
        error_bound if error_scale == 1.0 else error_scale * error_bound
    )
    if error_decay != 0.0:
        effective_error *= (decay_count / max(scores.size, 1)) ** error_decay
    raw_penalty = (
        effective_error + effective_error * effective_error / (2.0 * beta)
    )
    used_penalty = penalty_scale * raw_penalty
    # Preserve the original operation order when neither knob is engaged, so
    # old and new runs agree bit for bit rather than merely up to rounding.
    certificate = (
        objective - error_bound - error_bound * error_bound / (2.0 * beta)
        if penalty_scale == 1.0 and error_scale == 1.0 and error_decay == 0.0
        else objective - used_penalty
    )

    if float(weights.min()) < -WEIGHT_ABSOLUTE_TOLERANCE:
        raise PessimismValidationError("ITP weights must be nonnegative")
    if mean_weight < 1.0 - WEIGHT_ABSOLUTE_TOLERANCE:
        raise PessimismValidationError(
            f"mean ITP weight {mean_weight!r} fell below 1: the bisection did "
            "not return a normalizer at or below the root"
        )
    upper = 1.0 + bisection_tol / beta + WEIGHT_ABSOLUTE_TOLERANCE
    if mean_weight > upper:
        raise PessimismValidationError(
            f"mean ITP weight {mean_weight!r} exceeds 1 + tol/beta = {upper!r}"
        )
    if not float(weights.sum()) > 0.0:
        raise PessimismValidationError("ITP weights carry no mass")

    return ITPCertification(
        normalizer=normalizer,
        weights=weights,
        certificate=certificate,
        mean_weight=mean_weight,
        mean_square_weight=mean_square_weight,
        num_bisection_iterations=iterations,
        beta=beta,
        error_bound=error_bound,
        penalty_scale=penalty_scale,
        error_scale=error_scale,
        effective_error=effective_error,
        objective=objective,
        raw_penalty=raw_penalty,
        used_penalty=used_penalty,
        bisection_tol=bisection_tol,
        count=int(scores.size),
        bracket=(lambda_lo, lambda_hi),
    )


def itp_certify(
    scores: Sequence[float],
    beta: float,
    error_bound: float,
    bisection_tol: float,
    penalty_scale: float = 1.0,
    error_scale: float = 1.0,
    error_decay: float = 0.0,
    decay_count: int = 1,
) -> ITPCertification:
    """ITP-Certify: empirical normalizer, weights and certificate G_hat(beta; n).

    `scores` are one group's first `n` stored proxy scores. The certificate is

        normalizer + beta/2 (1 + mean(w^2)) - e - e^2/(2 beta),

    where `e = error_scale * error_bound` is the group-level L2 bound
    `bar_epsilon_i`, not a per-response quantity, and enters twice. With
    `error_scale=1` -- the default -- `e` is the bound as measured.
    """
    beta = require_positive_float(beta, "beta")
    error_bound = require_nonnegative_float(error_bound, "error_bound")
    bisection_tol = require_positive_float(bisection_tol, "bisection_tol")
    penalty_scale = require_nonnegative_float(penalty_scale, "penalty_scale")
    error_scale = require_nonnegative_float(error_scale, "error_scale")
    array = require_scores(scores)
    return _certify_array(
        array, beta, error_bound, bisection_tol, penalty_scale, error_scale,
        error_decay, decay_count
    )


def itp_sample(
    responses: Sequence[Any],
    scores: Sequence[float],
    beta: float,
    error_bound: float,
    bisection_tol: float,
    rng: np.random.Generator,
    penalty_scale: float = 1.0,
    error_scale: float = 1.0,
    error_decay: float = 0.0,
    decay_count: int = 1,
) -> ITPSampleResult:
    """ITP-Sample: certify, then draw one stored response by its ITP weight.

    The categorical law is `weights / sum(weights)`, never `weights / n`. With a
    positive bisection tolerance the weights only have mean in `[1, 1 + tau/beta]`,
    so `weights / n` need not sum to one; explicit renormalization is the
    paper's finite-tolerance rule. Responses at or below the normalizer carry
    zero weight and are never returned. Duplicate responses are allowed: the
    law is over the multiset of stored draws.
    """
    array = require_scores(scores)
    if len(responses) != array.size:
        raise PessimismValidationError(
            f"{len(responses)} responses for {array.size} scores: the stored "
            "response/score alignment is broken"
        )
    if not isinstance(rng, np.random.Generator):
        raise PessimismValidationError(
            "rng must be a numpy.random.Generator so the run is reproducible"
        )
    beta = require_positive_float(beta, "beta")
    error_bound = require_nonnegative_float(error_bound, "error_bound")
    bisection_tol = require_positive_float(bisection_tol, "bisection_tol")
    penalty_scale = require_nonnegative_float(penalty_scale, "penalty_scale")

    certification = _certify_array(
        array, beta, error_bound, bisection_tol, penalty_scale, error_scale,
        error_decay, decay_count
    )
    weights = certification.weights
    total = float(weights.sum())
    probabilities = weights / total
    index = int(rng.choice(probabilities.size, p=probabilities))
    return ITPSampleResult(
        selected_response=responses[index],
        selected_index=index,
        selection_probability=float(probabilities[index]),
        probabilities=probabilities,
        certification=certification,
    )
