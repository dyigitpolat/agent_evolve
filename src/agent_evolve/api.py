"""The public entry point: ``optimize(problem, budget=...) -> SearchResult``.

One required argument and one number a caller actually knows -- how many
evaluations they can afford. Everything else has a defensible default.

``proposer`` is the one option worth understanding:

``"random"``  samples the candidate schema. No credentials, no network, no
              cost. It is also the control arm: a model that cannot beat it on
              your problem is not earning its price. ``agent_evolve check``
              runs exactly that comparison.
``"llm"``     the model-driven proposer.
``"auto"``    ``llm`` when a provider credential is present, otherwise
              ``random``, said out loud through ``on_progress`` rather than
              silently.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from agent_evolve import bootstrap
from agent_evolve.contract import as_problem
from agent_evolve.core.formatting import format_search_space_description
from agent_evolve.core.results import SearchResult
from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.harness.directives import DefaultDirectives
from agent_evolve.harness.registry import harness_registry
from agent_evolve.session.evaluate import EvaluationCache
from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from agent_evolve.settings import AgentEvolveSettings, credentials_present

__all__ = ["optimize"]

Proposer = Literal["auto", "llm", "random"]

#: Candidates proposed per generation. The budget decides how many generations
#: that buys, so the caller states the number they know and not this one.
_BATCH = 8


def _describe(problem: Any) -> str:
    """Build the search-space description shown to the proposer."""
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
        constraints=None,  # constraints flow through HarnessContext
        problem_description=problem_description,
    )


def _resolve_proposer(proposer: str, announce: Callable[[str], None]) -> str:
    if proposer != "auto" and proposer not in ("llm", "random"):
        # Any harness registered by name is also a proposer. This is how an
        # out-of-tree integration is selected, without a second parameter that
        # means almost the same thing.
        bootstrap.load_integrations()
        if proposer not in harness_registry.ids():
            raise ValueError(
                f"proposer must be 'auto', 'llm', 'random' or a registered "
                f"harness id, got {proposer!r}. Registered: "
                f"{sorted(harness_registry.ids())}"
            )
    if proposer != "auto":
        return proposer
    if credentials_present():
        return "llm"
    announce(
        "No provider credential found, so candidates are being proposed at "
        "random. This costs nothing, and it is the baseline a model has to "
        "beat. Pass proposer='llm' with a credential to use a model."
    )
    return "random"


def _build_harness(kind: str, seed: Optional[int], settings: AgentEvolveSettings) -> Any:
    bootstrap.load_integrations()
    if kind == "random":
        harness_id = "random"
    elif kind == "llm":
        harness_id = settings.harness
    else:
        harness_id = kind  # an explicitly named registered harness
    missing = bootstrap.requirement_failure(harness_id)
    if missing is not None:
        # Fail here, naming the fix, rather than deep inside the first model
        # call with a bare ModuleNotFoundError.
        raise RuntimeError(
            f"the {harness_id!r} proposer {missing}. "
            "Or run with proposer='random', which needs nothing."
        )
    try:
        return harness_registry.create(harness_id, seed=seed)
    except KeyError as error:
        raise KeyError(bootstrap.explain_missing_harness(harness_id)) from error


def optimize(
    problem: Any,
    *,
    budget: int = 40,
    model: Optional[str] = None,
    proposer: str = "auto",
    seed: Optional[int] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> SearchResult:
    """Optimize *problem* within *budget* evaluations.

    *budget* counts artifacts measured, which is the expensive thing and the
    only sizing number the caller supplies. The problem's seeds are evaluated
    before anything is proposed, so the result always answers "did this beat
    what I already had".
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")

    bound = as_problem(problem)
    announce = on_progress or (lambda _message: None)
    settings = AgentEvolveSettings.from_env()
    kind = _resolve_proposer(proposer, announce)
    harness = _build_harness(kind, seed, settings)

    ctx = HarnessContext(
        objectives=list(bound.objectives),
        search_space_desc=_describe(bound),
        candidate_model=getattr(bound, "candidate_model", None),
        constraints_description=getattr(bound, "constraints_description", "") or "",
        directives=getattr(bound, "directives", None) or DefaultDirectives(),
    )
    harness.bind(
        ctx,
        LLMConfig(model=model or settings.model, temperature=settings.temperature),
    )

    seeds = tuple(dict(c) for c in bound.seeds())
    cache = EvaluationCache()
    cache.budget = budget
    config = LoopConfig(
        pop_size=min(budget, _BATCH),
        generations=max(1, budget // _BATCH),
        candidates_per_batch=_BATCH,
        seed=seed,
        seeds=seeds,
        evaluation_budget=budget,
        evaluation_cache=cache,
    )
    return run_evolution_loop(
        problem=bound,
        harness=harness,
        config=config,
        log=announce,
    )
