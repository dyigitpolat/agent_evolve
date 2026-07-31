"""Objective-only AgentEvolve composition for the pybamm_fastcharge workload."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import PybammFastChargeFiniteVariationCatalog
from .problem_def import PybammFastChargeProblem, default_settings


finite_variation_catalog: FiniteVariationCatalog = (
    PybammFastChargeFiniteVariationCatalog()
)


def create_benchmark(problem: PybammFastChargeProblem) -> AgenticBenchmark:
    """Bind one protocol problem in objective-only mode with the exact catalog."""

    if type(problem) is not PybammFastChargeProblem:
        raise TypeError("problem must be an exact PybammFastChargeProblem")
    return AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(PybammFastChargeFiniteVariationCatalog(),),
    )


def create_default_benchmark() -> AgenticBenchmark:
    return create_benchmark(PybammFastChargeProblem(default_settings()))


__all__ = [
    "create_benchmark",
    "create_default_benchmark",
    "finite_variation_catalog",
]
