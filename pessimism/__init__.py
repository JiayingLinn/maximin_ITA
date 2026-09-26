"""Paper-faithful LCB-Greedy allocation with finite-sample BGP oracles.

`core` holds Algorithm 1 and Algorithm 2 exactly as the paper states them and
depends on nothing but NumPy. `adapters` turns this repository's stored
response pools into the ordered sample-and-score stream Algorithm 1 consumes.
Nothing in `core` knows what a domain, a prompt, or a reward model is.
"""

from .core.bgp import (
    BGPCertification,
    BGPSampleResult,
    bgp_certify,
    bgp_sample,
)
from .core.lcb_greedy import (
    AllocationStep,
    GroupIndex,
    GroupOutcome,
    LCBGreedyResult,
    group_index,
    lcb_greedy,
)
from .core.radii import (
    CertificationSchedule,
    certified_betas,
    default_bisection_tol,
    delta_bgp,
    delta_obj,
    delta_total,
    geometric_beta_grid,
    initial_count,
    log_factor,
    m_beta,
)
from .core.validation import PessimismValidationError

__all__ = [
    "AllocationStep",
    "CertificationSchedule",
    "GroupIndex",
    "GroupOutcome",
    "BGPCertification",
    "BGPSampleResult",
    "LCBGreedyResult",
    "PessimismValidationError",
    "certified_betas",
    "default_bisection_tol",
    "delta_bgp",
    "delta_obj",
    "delta_total",
    "geometric_beta_grid",
    "group_index",
    "initial_count",
    "bgp_certify",
    "bgp_sample",
    "lcb_greedy",
    "log_factor",
    "m_beta",
]
