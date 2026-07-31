"""Objective-only AgentEvolve composition for the scip_miplib workload."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import ScipMiplibFiniteVariationCatalog
from .problem_def import ScipMiplibProblem, default_settings


finite_variation_catalog: FiniteVariationCatalog = (
    ScipMiplibFiniteVariationCatalog()
)


def create_benchmark(problem: ScipMiplibProblem) -> AgenticBenchmark:
    """Bind one SCIP problem in objective-only mode with the exact catalog."""

    if type(problem) is not ScipMiplibProblem:
        raise TypeError("problem must be an exact ScipMiplibProblem")
    return AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(ScipMiplibFiniteVariationCatalog(),),
    )


def create_default_benchmark() -> AgenticBenchmark:
    return create_benchmark(ScipMiplibProblem(default_settings()))


__all__ = [
    "create_benchmark",
    "create_default_benchmark",
    "finite_variation_catalog",
]
