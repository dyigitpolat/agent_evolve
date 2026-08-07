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

import dataclasses
from typing import Any, Callable, Literal, Optional

from agent_evolve import bootstrap
from agent_evolve.contract import as_problem
from agent_evolve.core.formatting import format_search_space_description
from agent_evolve.core.results import ProviderUsageSummary, SearchResult
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


def _resolve_strategy(strategy: str, has_seeds: bool, announce) -> str:
    """Pick the search loop. ``auto`` prefers genetics wherever they are usable.

    The authoring loop asks a model to write whole configurations from a text
    rendering of the Pareto front. Measured against uniform random sampling on
    every genome length tried, that loses (-0.086 to -0.531 excess capture)
    while recombination over a population wins (+0.0042 to +0.1798). So
    ``genetic`` is preferred wherever it can run, which is wherever the problem
    supplies at least one seed to give a candidate its shape.
    """

    if strategy not in ("auto", "genetic", "authoring"):
        raise ValueError(
            f"strategy must be 'auto', 'genetic' or 'authoring', got {strategy!r}"
        )
    if strategy != "auto":
        return strategy
    if has_seeds:
        return "genetic"
    announce(
        "No seeds were supplied, so candidates are authored from scratch rather "
        "than recombined. Give Problem.seeds() one configuration to use the "
        "genetic loop, which measures better against random search."
    )
    return "authoring"


def optimize(
    problem: Any,
    *,
    budget: int = 40,
    model: Optional[str] = None,
    proposer: str = "auto",
    strategy: str = "auto",
    seed: Optional[int] = None,
    seal: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> SearchResult:
    """Optimize *problem* within *budget* evaluations.

    *budget* counts artifacts measured, which is the expensive thing and the
    only sizing number the caller supplies. The problem's seeds are evaluated
    before anything is proposed, so the result always answers "did this beat
    what I already had".

    *seal* names a file to write the run's proposal journal to: one chained,
    self-authenticating line per model call, holding the exact configuration
    that was emitted, the digest of the prompt that produced it, the digest of
    the schema it was drawn from, and the verdict ``validate`` returned. The run
    then replays from that file with no provider and no credential. Pass it when
    the result has to be checkable by someone who was not there.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")

    bound = as_problem(problem)
    announce = on_progress or (lambda _message: None)
    settings = AgentEvolveSettings.from_env()

    # Arguments are validated before any branching. A caller who passes a
    # nonsense proposer must be told so whichever loop ends up running --
    # skipping validation on one path is how an invalid argument becomes a
    # silent no-op.
    kind = _resolve_proposer(proposer, announce)

    seeds = tuple(dict(c) for c in bound.seeds())
    chosen = _resolve_strategy(strategy, bool(seeds), announce)
    if chosen == "genetic" and seal is not None:
        # The seal journal holds generative proposals; the genetic loop's model
        # calls are operator choices, which that format cannot represent. A
        # journal the caller asked for and never got would be a silent no-op,
        # so refuse loudly and name the two ways out.
        raise ValueError(
            "seal journaling is not supported by the genetic strategy yet: "
            "the seal format records generative proposals, and the genetic "
            "loop makes operator choices instead. Pass strategy='authoring' "
            "to seal a generative run, or drop seal=."
        )
    if chosen == "genetic":
        # Only the loop is imported locally. Importing EvaluationCache here too
        # would make that name function-local for the whole body and break the
        # authoring path below, which uses the module-level import.
        from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

        chooser = None
        # Provider usage is measured from the completion seam's own journal,
        # never declared: a zero here means "counted and none occurred".
        usage_ledger = {"calls": 0, "input": 0, "output": 0, "tokens_known": True}
        if kind == "llm":
            # Guided operator choice: the model picks parents and cut points,
            # reasoning over the accumulated search state. It cannot author a
            # candidate -- OperatorChoice has no field that could hold one.
            from agent_evolve.integrations.completion import completion_for
            from agent_evolve.policies.llm_chooser import llm_chooser

            def _record_usage(record: dict) -> None:
                usage_ledger["calls"] += 1
                usage = record.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    usage_ledger["input"] += prompt_tokens
                    usage_ledger["output"] += completion_tokens
                else:
                    usage_ledger["tokens_known"] = False

            complete = completion_for(model or settings.model, settings,
                                      journal=_record_usage)
            if complete is not None:
                chooser = llm_chooser(
                    complete, objectives=list(bound.objectives), budget=budget,
                    on_shortfall=lambda got, want: announce(
                        f"the model supplied {got} of {want} operator choices; "
                        "the rest were filled at random"),
                )

        cache = EvaluationCache()
        cache.budget = budget
        # Sized from the BUDGET, not from how many seeds the caller happened to
        # supply: one seed would otherwise give a population of two, which
        # cannot recombine into anything its parents do not already contain.
        pop = max(4, min(budget // 4, 12))
        offspring = max(2, pop - 2)
        # `generations` is a cap, not a schedule. Duplicate offspring hit the
        # evaluation cache without spending budget, so a fixed generation count
        # would end the run with budget unspent; the loop's real stop condition
        # is the budget.
        result = run_genetic_loop(
            problem=bound,
            config=GeneticConfig(
                population_size=pop,
                offspring_per_generation=offspring,
                generations=max(1, 4 * budget // max(1, offspring)),
                seed=seed,
                seeds=seeds,
                evaluation_budget=budget,
                evaluation_cache=cache,
            ),
            chooser=chooser,
            log=announce,
        )
        if usage_ledger["calls"] and usage_ledger["tokens_known"]:
            usage = ProviderUsageSummary(
                calls=usage_ledger["calls"],
                input_tokens=usage_ledger["input"],
                output_tokens=usage_ledger["output"],
                model=model or settings.model,
                reported_by="openrouter response usage",
            )
        else:
            usage = ProviderUsageSummary(
                calls=usage_ledger["calls"],
                model=(model or settings.model) if usage_ledger["calls"] else None,
            )
        return dataclasses.replace(result, provider_usage=usage)

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

    seal_handle = None
    if seal is not None:
        from pathlib import Path

        from agent_evolve.application.generative_proposal_journal import journal_line
        from agent_evolve.proposal_mode import build_generative_proposer

        path = Path(seal)
        path.parent.mkdir(parents=True, exist_ok=True)
        seal_handle = path.open("w", encoding="ascii")

        def _write(record: dict) -> None:
            seal_handle.write(journal_line(record) + "\n")
            seal_handle.flush()

        harness = build_generative_proposer(bound, delegate=harness, on_seal=_write)
        harness.bind(
            ctx,
            LLMConfig(model=model or settings.model, temperature=settings.temperature),
        )

    cache = EvaluationCache()   # `seeds` was already read above, once
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
    try:
        return run_evolution_loop(
            problem=bound,
            harness=harness,
            config=config,
            log=announce,
        )
    finally:
        # Closed even when the run raises: a journal truncated by a crash still
        # records every call that did happen, and that is the honest artifact.
        if seal_handle is not None:
            seal_handle.close()
