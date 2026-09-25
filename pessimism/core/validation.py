"""Assumption checks for the algorithm core.

The theoretical statements the two algorithms implement are conditional on
their hypotheses, so the hypotheses are part of the interface: a score outside
[0, R_max] or a nonpositive beta makes the certificate a number without a
meaning attached. Every check here therefore refuses the input rather than
repairing it. Whatever rescaling a caller's rewards need belongs in the
adapter, where the choice of scale is visible and recorded.
"""

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np


class PessimismValidationError(ValueError):
    """An input violates an assumption the algorithms are proved under."""


# Scores are compared against [0, R_max] with a tolerance that scales with the
# interval, so a value the caller's own arithmetic placed one ulp outside is
# accepted while a genuinely negative reward is not.
SCORE_RELATIVE_TOLERANCE = 1e-9

# The weight invariants of Algorithm 2 hold up to bisection tolerance in exact
# arithmetic; this is the extra room float64 accumulation needs on top.
WEIGHT_ABSOLUTE_TOLERANCE = 1e-9


def require_positive_int(value: Any, name: str) -> int:
    """An integer count, rejecting bools, floats that merely look integral."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PessimismValidationError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value <= 0:
        raise PessimismValidationError(f"{name} must be positive, got {value}")
    return value


def require_finite_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise PessimismValidationError(f"{name} must be a real number") from error
    if not math.isfinite(value):
        raise PessimismValidationError(f"{name} must be finite, got {value}")
    return value


def require_positive_float(value: Any, name: str) -> float:
    value = require_finite_float(value, name)
    if value <= 0.0:
        raise PessimismValidationError(f"{name} must be positive, got {value}")
    return value


def require_nonnegative_float(value: Any, name: str) -> float:
    value = require_finite_float(value, name)
    if value < 0.0:
        raise PessimismValidationError(f"{name} must be nonnegative, got {value}")
    return value


def require_confidence_delta(value: Any, name: str = "confidence_delta") -> float:
    value = require_finite_float(value, name)
    if not 0.0 < value < 1.0:
        raise PessimismValidationError(f"{name} must lie in (0, 1), got {value}")
    return value


def require_beta_grid(beta_grid: Iterable[float], name: str = "beta_grid") -> np.ndarray:
    """A finite grid of strictly positive betas, strictly increasing.

    Strict increase is checked after the floating-point construction rather
    than before it: a geometric grid with a ratio close to one can collapse two
    neighbours into the same float64, and a duplicated beta would silently
    double one grid point's weight in the log factor.
    """
    grid = np.asarray(list(beta_grid), dtype=np.float64)
    if grid.ndim != 1:
        raise PessimismValidationError(f"{name} must be one-dimensional")
    if grid.size == 0:
        raise PessimismValidationError(f"{name} must be nonempty")
    if not np.all(np.isfinite(grid)):
        raise PessimismValidationError(f"{name} must be finite")
    if not np.all(grid > 0.0):
        raise PessimismValidationError(f"{name} entries must be strictly positive")
    if grid.size > 1 and not np.all(np.diff(grid) > 0.0):
        raise PessimismValidationError(
            f"{name} must be strictly increasing with no duplicates"
        )
    return grid


def require_scores(
    scores: Sequence[float],
    r_max: Optional[float] = None,
    name: str = "scores",
) -> np.ndarray:
    """Proxy scores as a nonempty finite float64 vector, optionally range-checked.

    `r_max` is optional because Algorithm 2 is stated for stored scores and
    never needs the bound itself; the range assumption is enforced wherever the
    bound is in scope, which is the certification schedule and Algorithm 1.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1:
        raise PessimismValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise PessimismValidationError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise PessimismValidationError(f"{name} must be finite (no NaN or Inf)")
    if r_max is not None:
        tolerance = SCORE_RELATIVE_TOLERANCE * max(1.0, abs(r_max))
        low = float(array.min())
        high = float(array.max())
        if low < -tolerance or high > r_max + tolerance:
            raise PessimismValidationError(
                f"{name} must lie in [0, {r_max}]; observed [{low}, {high}]"
            )
    return array
