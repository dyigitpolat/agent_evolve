"""Objective-only AgentEvolve composition for the pinned BOiLS benchmark."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import BoilsFiniteVariationCatalog
from .problem_def import problem


finite_variation_catalog: FiniteVariationCatalog = BoilsFiniteVariationCatalog()

# BOiLS already exposes two trusted numeric objectives.  It intentionally uses
# AgentEvolve's objective-only mode here: no Airfoil evidence, violation,
# relation, or reward policy leaks into this independent domain adapter.
benchmark = AgenticBenchmark(
    problem=problem,
    finite_variation_catalogs=(finite_variation_catalog,),
)


__all__ = ["benchmark", "finite_variation_catalog"]
