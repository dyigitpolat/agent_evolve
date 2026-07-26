"""Objective-only public AgentEvolve composition for Timeloop co-design."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import TimeloopFiniteVariationCatalog
from .problem_def import problem


finite_variation_catalog: FiniteVariationCatalog = TimeloopFiniteVariationCatalog()

benchmark = AgenticBenchmark(
    problem=problem,
    finite_variation_catalogs=(finite_variation_catalog,),
)


__all__ = ["benchmark", "finite_variation_catalog"]
