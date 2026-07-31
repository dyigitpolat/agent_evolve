"""PyBaMM battery fast-charging workload adapter (typed WorkloadKit).

Three-stage CC(+CV) charging protocol optimization on a lumped-thermal DFN
model of an LG M50 cell, evaluated bi-objectively (charge time, peak
temperature rise) through the isolated PyBaMM domain virtualenv with the
IDAKLU solver, a hard wall-clock timeout, and a termination-validity gate.
The agent_evolve environment never imports pybamm.
"""

from .campaign_workload import (
    CAMPAIGN_WORKLOAD_ID,
    compose_pybamm_fastcharge_campaign_workload,
)
from .finite_variation_catalog import PybammFastChargeFiniteVariationCatalog
from .problem_def import ChargingProtocolCandidate, PybammFastChargeProblem

__all__ = [
    "CAMPAIGN_WORKLOAD_ID",
    "ChargingProtocolCandidate",
    "PybammFastChargeFiniteVariationCatalog",
    "PybammFastChargeProblem",
    "compose_pybamm_fastcharge_campaign_workload",
]
