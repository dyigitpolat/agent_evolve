"""The blackbox entry point: ``optimize(problem, ...) -> SearchResult``.

This is the composition root for library users: it resolves settings, loads
integrations, builds the harness, and runs the loop. Everything below receives
its collaborators explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from agent_evolve import bootstrap
from agent_evolve.core.formatting import format_search_space_description
from agent_evolve.core.problem import Problem
from agent_evolve.core.results import SearchResult
from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.harness.directives import DefaultDirectives
from agent_evolve.harness.registry import harness_registry
from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from agent_evolve.settings import AgentEvolveSettings


def _describe(problem: Problem) -> str:
    """Build the search-space description shown to the LLM."""
    problem_description = None
    if hasattr(problem, "search_space_description"):
        problem_description = problem.search_space_description()

    config_schema = getattr(problem, "config_schema", None)
    candidate_model = getattr(problem, "candidate_model", None)
    if config_schema is None and candidate_model is not None:
        try:
            config_schema = candidate_model.model_json_schema()
        except Exception:
            config_schema = None

    return format_search_space_description(
        list(problem.objectives),
        config_schema=config_schema,
        example_config=getattr(problem, "example_config", None),
        constraints=None,  # constraints flow through HarnessContext to avoid duplication
        problem_description=problem_description,
    )


def optimize(
    problem: Problem,
    *,
    harness: Optional[str] = None,
    model: Optional[str] = None,
    pop_size: int = 8,
    generations: int = 5,
    candidates_per_batch: int = 5,
    max_regen_rounds: int = 10,
    max_failed_examples: int = 5,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    retries: int = 3,
    use_failure_insights: bool = True,
    use_constraint_instruction: bool = True,
    use_performance_insights: bool = True,
    extra: Optional[Dict[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    settings: Optional[AgentEvolveSettings] = None,
) -> SearchResult:
    """Run LLM-driven Pareto-guided optimisation against *problem*.

    ``harness`` selects a registered LLM adapter. The in-tree adapter is
    ``"pydantic_ai"``; external adapters can register through the integration
    entry-point group. Defaults come from :class:`AgentEvolveSettings`.
    """
    settings = settings or AgentEvolveSettings.from_env()
    harness_id = harness or settings.harness
    resolved_model = model or settings.model
    resolved_temp = temperature if temperature is not None else settings.temperature

    bootstrap.load_integrations()
    h = harness_registry.create(harness_id)

    directives = getattr(problem, "directives", None) or DefaultDirectives()
    ctx = HarnessContext(
        objectives=list(problem.objectives),
        search_space_desc=_describe(problem),
        candidate_model=getattr(problem, "candidate_model", None),
        constraints_description=getattr(problem, "constraints_description", "") or "",
        directives=directives,
    )
    h.bind(ctx, LLMConfig(model=resolved_model, temperature=resolved_temp, retries=retries,
                          extra=extra or {}))
    if on_event is not None and hasattr(h, "set_call_observer"):
        h.set_call_observer(on_event)

    config = LoopConfig(
        pop_size=pop_size,
        generations=generations,
        candidates_per_batch=candidates_per_batch,
        max_regen_rounds=max_regen_rounds,
        max_failed_examples=max_failed_examples,
        seed=seed,
        llm_retries=retries,
        use_failure_insights=use_failure_insights,
        use_constraint_instruction=use_constraint_instruction,
        use_performance_insights=use_performance_insights,
    )
    return run_evolution_loop(
        problem=problem,
        harness=h,
        config=config,
        log=log or (lambda m: None),
        on_event=on_event,
    )
