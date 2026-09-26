"""Definition 3.2 and Definition 4.1: the log factor, the radii, and which
regularizations may be certified at a given sample count.

The three radii are what makes the mechanism finite-sample rather than
asymptotic, and the only place the slack enters. `delta_total` is nonincreasing
in both the count and the regularization strength, so the certified set is an
upper segment of the grid that only grows with n, and the smallest certifiable
count is decided by the largest beta alone.

Reading these numbers honestly matters as much as computing them: at the sample
sizes a stored response pool affords, `delta_total` is typically larger than
`R_max`, which means the certified lower bound `L_i - slack` is below zero and
says nothing. That does not make the allocation invalid -- Algorithm 1 compares
indices, and the slack is a common shift it cancels -- but a run should report
the gap rather than quietly presenting a vacuous bound as a guarantee.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .validation import (
    PessimismValidationError,
    require_beta_grid,
    require_confidence_delta,
    require_positive_float,
    require_positive_int,
)


def geometric_beta_grid(beta_min: float, ratio: float, num_points: int) -> np.ndarray:
    """`{beta_min * ratio**j : j = 0, ..., num_points - 1}`.

    The endpoint convention is inclusive on both ends and stated in the count,
    not in a maximum: `num_points` grid points, the largest of which is
    `beta_min * ratio ** (num_points - 1)`. The paper writes the grid as
    `j = 0, ..., J_beta`, so `num_points = J_beta + 1`.
    """
    beta_min = require_positive_float(beta_min, "beta_min")
    ratio = require_positive_float(ratio, "ratio")
    num_points = require_positive_int(num_points, "num_points")
    if num_points > 1 and ratio <= 1.0:
        raise PessimismValidationError(f"ratio must exceed 1, got {ratio}")
    grid = beta_min * np.power(ratio, np.arange(num_points, dtype=np.float64))
    return require_beta_grid(grid)


def log_factor(
    num_groups: int,
    num_betas: int,
    total_budget: int,
    confidence_delta: float,
) -> float:
    """`ell = log(16 k N_beta B (B^2 + 2) / delta)`.

    Computed once from the global budget, not from any group's current count:
    the union bound it comes from runs over every group, every grid point and
    every count the run could reach.
    """
    num_groups = require_positive_int(num_groups, "num_groups")
    num_betas = require_positive_int(num_betas, "num_betas")
    total_budget = require_positive_int(total_budget, "total_budget")
    confidence_delta = require_confidence_delta(confidence_delta)
    budget = float(total_budget)
    return math.log(
        16.0
        * num_groups
        * num_betas
        * budget
        * (budget * budget + 2.0)
        / confidence_delta
    )


def m_beta(beta: float, r_max: float) -> float:
    """`M_beta = 1 + R_max / beta`, the sup of the BGP weight at that beta."""
    return 1.0 + r_max / beta


def delta_obj(n: int, beta: float, r_max: float, ell: float) -> float:
    """`5 R_max (sqrt(M ell / n) + M ell / n)`."""
    ratio = m_beta(beta, r_max) * ell / n
    return 5.0 * r_max * (math.sqrt(ratio) + ratio)


def delta_bgp(n: int, beta: float, r_max: float, ell: float) -> float:
    """`R_max sqrt(M ell / n)`."""
    return r_max * math.sqrt(m_beta(beta, r_max) * ell / n)


def delta_total(n: int, beta: float, r_max: float, ell: float) -> float:
    """`delta_obj + delta_bgp = 6 R_max sqrt(M ell / n) + 5 R_max M ell / n`."""
    ratio = m_beta(beta, r_max) * ell / n
    return 6.0 * r_max * math.sqrt(ratio) + 5.0 * r_max * ratio


def certified_betas(
    n: int,
    beta_grid: Sequence[float],
    r_max: float,
    ell: float,
    slack: float,
) -> np.ndarray:
    """`{beta in grid : delta_total(n, beta) <= slack}`, in increasing order."""
    grid = np.asarray(beta_grid, dtype=np.float64)
    keep = [delta_total(n, float(beta), r_max, ell) <= slack for beta in grid]
    return grid[np.asarray(keep, dtype=bool)]


def initial_count(
    beta_grid: Sequence[float],
    r_max: float,
    ell: float,
    slack: float,
    total_budget: int,
) -> int:
    """The smallest `n` in `[1, total_budget]` whose certified set is nonempty.

    Found by binary search on the largest grid point, because `delta_total` is
    nonincreasing in beta, and then checked against the definition itself so
    the shortcut cannot quietly disagree with it.
    """
    grid = require_beta_grid(beta_grid)
    total_budget = require_positive_int(total_budget, "total_budget")
    beta_max = float(grid[-1])

    def certifiable(n: int) -> bool:
        return delta_total(n, beta_max, r_max, ell) <= slack

    if not certifiable(total_budget):
        raise PessimismValidationError(
            "No sample count within the budget certifies any beta: "
            f"delta_total({total_budget}, beta_max={beta_max:g}) = "
            f"{delta_total(total_budget, beta_max, r_max, ell):.4g} > slack={slack:g}. "
            "Raise the slack or the budget, or widen the grid upward."
        )
    low, high = 1, total_budget
    while low < high:
        middle = (low + high) // 2
        if certifiable(middle):
            high = middle
        else:
            low = middle + 1
    if certified_betas(low, grid, r_max, ell, slack).size == 0:
        raise PessimismValidationError(f"Certified set empty at initial_count={low}")
    if low > 1 and certified_betas(low - 1, grid, r_max, ell, slack).size != 0:
        raise PessimismValidationError(
            f"initial_count={low} is not minimal: {low - 1} already certifies"
        )
    return low


def default_bisection_tol(beta_grid: Sequence[float], r_max: float, total_budget: int) -> float:
    """`min(min(beta_grid), R_max) / total_budget`, the paper's safe scale."""
    grid = require_beta_grid(beta_grid)
    r_max = require_positive_float(r_max, "r_max")
    total_budget = require_positive_int(total_budget, "total_budget")
    return min(float(grid[0]), r_max) / total_budget


@dataclass(frozen=True)
class CertificationSchedule:
    """Everything Definitions 3.2 and 4.1 fix before a single response is drawn.

    Holding it as one object keeps the log factor, the grid and the slack that
    produced `initial_count` attached to the run that used them, which is what
    a diagnostic file needs to be reproducible.
    """

    beta_grid: np.ndarray
    r_max: float
    slack: float
    confidence_delta: float
    num_groups: int
    total_budget: int
    ell: float
    initial_count: int
    bisection_tol: float
    # The smallest count that certifies anything, kept separate from the count
    # every group is actually initialised with. The two coincide by default,
    # and Algorithm 1 uses `initial_count` as a floor on every group -- so
    # raising it is a fairness floor, and reading them apart is the only way a
    # run can say which of the two a result came from.
    certifiable_count: int = 0

    @classmethod
    def build(
        cls,
        num_groups: int,
        total_budget: int,
        beta_grid: Sequence[float],
        r_max: float,
        slack: float,
        confidence_delta: float,
        bisection_tol: Optional[float] = None,
        min_initial_count: Optional[int] = None,
    ) -> "CertificationSchedule":
        grid = require_beta_grid(beta_grid)
        num_groups = require_positive_int(num_groups, "num_groups")
        total_budget = require_positive_int(total_budget, "total_budget")
        r_max = require_positive_float(r_max, "r_max")
        slack = require_positive_float(slack, "slack")
        confidence_delta = require_confidence_delta(confidence_delta)
        ell = log_factor(num_groups, grid.size, total_budget, confidence_delta)
        certifiable = initial_count(grid, r_max, ell, slack, total_budget)
        start = certifiable
        if min_initial_count is not None:
            floor = require_positive_int(min_initial_count, "min_initial_count")
            if floor < certifiable:
                raise PessimismValidationError(
                    f"min_initial_count={floor} is below the smallest count "
                    f"that certifies any beta at slack={slack:g}, which is "
                    f"{certifiable}. The certified set is empty at {floor}, so "
                    "BGP-Certify would have no beta to choose from. Raise the "
                    "slack (a larger slack lowers the certifiable count) or "
                    "raise the floor."
                )
            start = floor
        if total_budget < num_groups * start:
            raise PessimismValidationError(
                f"total_budget={total_budget} cannot cover initialization: "
                f"{num_groups} groups x initial_count={start} = "
                f"{num_groups * start} responses are spent before the greedy "
                "loop begins."
            )
        if bisection_tol is None:
            bisection_tol = default_bisection_tol(grid, r_max, total_budget)
        else:
            bisection_tol = require_positive_float(bisection_tol, "bisection_tol")
        return cls(
            beta_grid=grid,
            r_max=r_max,
            slack=slack,
            confidence_delta=confidence_delta,
            num_groups=num_groups,
            total_budget=total_budget,
            ell=ell,
            initial_count=start,
            bisection_tol=bisection_tol,
            certifiable_count=certifiable,
        )

    def certified(self, n: int) -> np.ndarray:
        return certified_betas(n, self.beta_grid, self.r_max, self.ell, self.slack)

    def radius(self, n: int, beta: float) -> float:
        return delta_total(n, beta, self.r_max, self.ell)

    def guarantee_is_vacuous(self) -> bool:
        """True when `L_i - slack` cannot be positive whatever the data says.

        `L_i` is at most `R_max` (Lemma 2.2(iii)), so a slack at or above
        `R_max` leaves the certified value nonpositive by arithmetic alone.
        """
        return self.slack >= self.r_max

    def as_dict(self) -> dict:
        return {
            "beta_grid": [float(beta) for beta in self.beta_grid],
            "r_max": self.r_max,
            "slack": self.slack,
            "confidence_delta": self.confidence_delta,
            "num_groups": self.num_groups,
            "total_budget": self.total_budget,
            "ell": self.ell,
            "initial_count": self.initial_count,
            "certifiable_count": self.certifiable_count,
            "bisection_tol": self.bisection_tol,
            "guarantee_is_vacuous": self.guarantee_is_vacuous(),
        }
