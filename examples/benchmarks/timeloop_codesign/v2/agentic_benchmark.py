"""Objective-only public AgentEvolve composition for Timeloop v2."""

from __future__ import annotations

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog

from .finite_variation_catalog import TimeloopV2FiniteVariationCatalog
from .frozen_panels import frozen_network_panel
from .problem_def import problem


finite_variation_catalog: FiniteVariationCatalog = TimeloopV2FiniteVariationCatalog(
    frozen_network_panel("resnet50")
)

benchmark = AgenticBenchmark(
    problem=problem,
    finite_variation_catalogs=(finite_variation_catalog,),
)


__all__ = ["benchmark", "finite_variation_catalog"]
