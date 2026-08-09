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

#: Who turns a screen's evidence into a sampling prior. The llm forms fall
#: back to their rule comparator -- out loud -- when no model call is possible.
_PRIORS = ("rule", "rule-weighted", "llm", "llm-weighted")

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
    structure_budget: int = 0,
    prior: str = "rule",
    effort: Optional[str] = None,
    journal: Any = None,
    authorship: Any = "auto",
) -> SearchResult:
    """Optimize *problem* within *budget* evaluations.

    *budget* counts artifacts measured, which is the expensive thing and the
    only sizing number the caller supplies. The problem's seeds are evaluated
    before anything is proposed, so the result always answers "did this beat
    what I already had".

    *structure_budget* spends that many evaluations -- charged against the same
    *budget* -- on a crossed screen before the population is built; *prior*
    names who turns the screen into a sampling prior: the credential-free
    ``"rule"`` (default) or ``"rule-weighted"``, or their model-backed forms
    ``"llm"`` / ``"llm-weighted"``, which fall back to the rule comparator, out
    loud, when no model call is possible.

    *effort* pins the model's reasoning effort on every completion call, and
    *journal* (a callable, or a path to a JSONL file) receives one record per
    completed model call -- model served plus token usage -- so a run's spend
    is verifiable from its own artifacts. Both belong to the genetic strategy.

    *seal* names a file to write the run's proposal journal to: one chained,
    self-authenticating line per model call, holding the exact configuration
    that was emitted, the digest of the prompt that produced it, the digest of
    the schema it was drawn from, and the verdict ``validate`` returned. The run
    then replays from that file with no provider and no credential. Pass it when
    the result has to be checkable by someone who was not there.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    if (not isinstance(structure_budget, int) or isinstance(structure_budget, bool)
            or structure_budget < 0):
        raise ValueError(
            f"structure_budget must be a non-negative integer, got "
            f"{structure_budget!r}"
        )
    if structure_budget >= budget:
        raise ValueError(
            f"structure_budget ({structure_budget}) must leave room inside the "
            f"budget ({budget}): the screen is charged against the same budget "
            "as the search it informs"
        )
    if prior not in _PRIORS:
        raise ValueError(f"prior must be one of {sorted(_PRIORS)}, got {prior!r}")
    if effort is not None and not isinstance(effort, str):
        raise ValueError(
            f"effort must be a provider effort level as a string, got {effort!r}"
        )
    from agent_evolve.session.authorship import AuthorshipConfig
    if isinstance(authorship, AuthorshipConfig):
        authorship_config = authorship
    elif authorship == "auto":
        # Resolved on the genetic branch: the model-authored surrogate is ON
        # when a model call is possible (the sealed S1 luna-clear row held),
        # off otherwise. The evidence-backed default, not the hopeful one.
        authorship_config = None
    elif isinstance(authorship, str):
        authorship_config = AuthorshipConfig.preset(authorship)
    else:
        raise ValueError(
            "authorship must be an AuthorshipConfig or a preset name, got "
            f"{authorship!r}"
        )

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
    if chosen != "genetic":
        engaged = [name for name, on in (
            ("structure_budget", structure_budget != 0),
            ("prior", prior != "rule"),
            ("effort", effort is not None),
            ("journal", journal is not None),
            ("authorship", authorship_config is not None
             and authorship_config.engaged),
        ) if on]
        if engaged:
            # A knob the run would silently ignore is a silent no-op -- the
            # same defect class the seal refusal above exists to prevent.
            raise ValueError(
                f"{', '.join(engaged)} belong(s) to the genetic strategy, and "
                "this run resolved to 'authoring'. Give the problem a seed to "
                "use the genetic loop, or drop the genetic-only arguments."
            )
    if chosen == "genetic":
        # Only the loop is imported locally. Importing EvaluationCache here too
        # would make that name function-local for the whole body and break the
        # authoring path below, which uses the module-level import.
        from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

        journal_handle = None
        journal_sink: Optional[Callable[[dict], None]] = None
        if callable(journal):
            journal_sink = journal
        elif journal is not None:
            import json as _json
            from pathlib import Path

            journal_path = Path(journal)
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            # Opened eagerly even though the run may make no call: an empty
            # journal is a measured zero, an absent file is "nobody looked".
            journal_handle = journal_path.open("w", encoding="utf-8")

            def journal_sink(record: dict) -> None:
                journal_handle.write(_json.dumps(record, sort_keys=True) + "\n")
                journal_handle.flush()

        try:
            chooser = None
            complete = None
            # Provider usage is measured from the completion seam's own
            # journal, never declared: zero means "counted and none occurred".
            usage_ledger = {"calls": 0, "input": 0, "output": 0, "tokens_known": True}
            if kind == "llm":
                # Guided operator choice: the model picks parents and cut
                # points, reasoning over the accumulated search state. It
                # cannot author a candidate -- OperatorChoice has no field
                # that could hold one.
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
                    if journal_sink is not None:
                        journal_sink(record)

                complete = completion_for(model or settings.model, settings,
                                          journal=_record_usage, effort=effort)
                if complete is not None:
                    from agent_evolve.policies.semantics import domain_card
                    chooser = llm_chooser(
                        complete, objectives=list(bound.objectives), budget=budget,
                        domain_context=domain_card(bound),
                        on_shortfall=lambda got, want: announce(
                            f"the model supplied {got} of {want} operator choices; "
                            "the rest were filled at random"),
                    )
            if effort is not None and complete is None:
                announce(
                    "effort pins model reasoning, and this run makes no model "
                    "calls, so it has no effect here."
                )

            prior_proposer: Any = None
            if prior == "rule-weighted":
                from agent_evolve.policies.weighted_prior import (
                    statistical_weighted_prior)
                prior_proposer = statistical_weighted_prior
            elif prior in ("llm", "llm-weighted"):
                if complete is None:
                    announce(
                        f"prior={prior!r} needs a model call and none is "
                        "available; using the credential-free rule comparator "
                        "instead."
                    )
                    if prior == "llm-weighted":
                        from agent_evolve.policies.weighted_prior import (
                            statistical_weighted_prior)
                        prior_proposer = statistical_weighted_prior
                elif prior == "llm":
                    from agent_evolve.policies.llm_prior import llm_prior_proposer
                    from agent_evolve.policies.semantics import domain_card
                    prior_proposer = llm_prior_proposer(
                        complete, objectives=list(bound.objectives),
                        domain_context=domain_card(bound))
                else:
                    from agent_evolve.policies.semantics import domain_card
                    from agent_evolve.policies.weighted_prior import (
                        llm_weighted_prior_proposer)
                    prior_proposer = llm_weighted_prior_proposer(
                        complete, objectives=list(bound.objectives),
                        domain_context=domain_card(bound))

            if authorship_config is None:
                if complete is not None:
                    authorship_config = AuthorshipConfig(surrogate="llm")
                    announce(
                        "authorship: model-authored surrogate screening is ON "
                        "(the sealed luna-clear row held); pass "
                        "authorship='off' to disable.")
                else:
                    authorship_config = AuthorshipConfig()
            # Sized from the BUDGET, not from how many seeds the caller
            # happened to supply: one seed would otherwise give a population
            # of two, which cannot recombine into anything its parents do not
            # already contain.
            pop = max(4, min(budget // 4, 12))
            offspring = max(2, pop - 2)

            from agent_evolve.policies.semantics import domain_card
            from agent_evolve.session.authorship import build_authorship
            policies = build_authorship(
                authorship_config, complete=complete,
                objectives=list(bound.objectives),
                schema_text=domain_card(bound), seed=seed, announce=announce,
                candidate_model=getattr(bound, "candidate_model", None),
                init_template=(dict(seeds[0]) if seeds else None),
                init_k=max(0, pop - len(seeds)))

            cache = EvaluationCache()
            cache.budget = budget
            # `generations` is a cap, not a schedule. Duplicate offspring hit
            # the evaluation cache without spending budget, so a fixed
            # generation count would end the run with budget unspent; the
            # loop's real stop condition is the budget.
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
                    structure_budget=structure_budget,
                    prior_proposer=prior_proposer,
                    screening=policies.screening,
                    portfolio=policies.portfolio,
                    initial_proposals=policies.initial_proposals,
                ),
                chooser=chooser,
                log=announce,
            )
            if policies.init_author is not None and result.telemetry is not None:
                from agent_evolve.core.telemetry import harvest_telemetry
                extra = harvest_telemetry((policies.init_author,))
                result = dataclasses.replace(
                    result,
                    telemetry=dataclasses.replace(
                        result.telemetry,
                        mechanisms=result.telemetry.mechanisms + extra.mechanisms))
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
        finally:
            # Closed even when the run raises: a journal truncated by a crash
            # still records every call that did happen.
            if journal_handle is not None:
                journal_handle.close()

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
