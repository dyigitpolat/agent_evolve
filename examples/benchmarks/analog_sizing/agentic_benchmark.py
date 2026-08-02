"""Objective-only AgentEvolve composition for the analog_sizing workload."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import AnalogSizingFiniteVariationCatalog
from .problem_def import AnalogSizingProblem, default_settings


finite_variation_catalog: FiniteVariationCatalog = (
    AnalogSizingFiniteVariationCatalog()
)


def create_benchmark(problem: AnalogSizingProblem) -> AgenticBenchmark:
    """Bind one amplifier topology in objective-only mode with the exact catalog."""

    if type(problem) is not AnalogSizingProblem:
        raise TypeError("problem must be an exact AnalogSizingProblem")
    return AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(AnalogSizingFiniteVariationCatalog(),),
    )


def create_default_benchmark() -> AgenticBenchmark:
    return create_benchmark(AnalogSizingProblem(default_settings()))


__all__ = [
    "create_benchmark",
    "create_default_benchmark",
    "finite_variation_catalog",
]
