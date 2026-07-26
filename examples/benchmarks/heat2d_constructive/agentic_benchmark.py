"""Public inverted-API composition for constructive Heat2D."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import Heat2DFiniteVariationCatalog
from .phenotype_identity import Heat2DPhenotypeIdentityPolicy
from .problem_def import problem


finite_variation_catalog: FiniteVariationCatalog = Heat2DFiniteVariationCatalog()
phenotype_identity_policy = Heat2DPhenotypeIdentityPolicy(
    resolution=problem.settings.resolution
)

# Objective-only mode is deliberate.  The direct-v3 manifest remains available
# from ``evaluate_detailed`` for trace analysis, while the generic archive uses
# the sole trusted scalar EngiBench objective.
benchmark = AgenticBenchmark(
    problem=problem,
    phenotype_identity=phenotype_identity_policy,
    finite_variation_catalogs=(finite_variation_catalog,),
)


__all__ = [
    "benchmark",
    "finite_variation_catalog",
    "phenotype_identity_policy",
]
