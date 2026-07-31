"""SCIP/MIPLIB solver-configuration workload adapter (typed WorkloadKit).

Curated 20-parameter + 3-emphasis SCIP 10.0 configuration surface evaluated
as a bi-objective panel over frozen MIPLIB 2017 development instances.  The
evaluator shells out to the isolated SCIP domain virtualenv; the agent_evolve
environment never imports pyscipopt.
"""

from .campaign_workload import (
    CAMPAIGN_WORKLOAD_ID,
    compose_scip_miplib_campaign_workload,
)
from .finite_variation_catalog import ScipMiplibFiniteVariationCatalog
from .problem_def import ScipConfigCandidate, ScipMiplibProblem

__all__ = [
    "CAMPAIGN_WORKLOAD_ID",
    "ScipConfigCandidate",
    "ScipMiplibFiniteVariationCatalog",
    "ScipMiplibProblem",
    "compose_scip_miplib_campaign_workload",
]
