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
_PRIORS = ("rule", "rule-weighted", "llm", "llm-weighted",
           "llm-weighted-committed")

#: Candidates proposed per generation. The budget decides how many generations
#: that buys, so the caller states the number they know and not this one.
_BATCH = 8

#: The declared parameters of the two slot-taking mechanisms, as constants
#: rather than literals repeated in a signature and two comparisons. Both
#: mechanisms are off by default; these are what they run AT when they are
#: asked for, and every one of them is a measured number rather than a tuned
#: one (see `GeneticConfig.explore` / `.intensify`).
_EXPLORE_SCHEDULE = (0.5, 0.1)
_INTENSIFY_FRACTION = 0.25
_INTENSIFY_PIN_RANGE = (6, 12)

#: The largest budget any SEALED row was measured at. At or below it the
#: genetic sizing is a control arm and may not move; above it there is nothing
#: to hold still. A constant rather than a knob, because a knob is a way to
#: move a sealed arm by accident.
_SEALED_BUDGET_CEILING = 384


def _genetic_sizing(budget: int) -> tuple[int, int]:
    """``(population, offspring per generation)``, from the budget alone.

    Sized from the BUDGET, not from how many seeds the caller happened to
    supply: one seed would otherwise give a population of two, which cannot
    recombine into anything its parents do not already contain.

    Two regimes, and the split is measured rather than tasteful.

    Up to ``_SEALED_BUDGET_CEILING`` the population is the old expression --
    capped at twelve, floored at four -- written here as the literal branch so
    that every budget a sealed row was measured at runs the arithmetic it was
    measured with. The byte fossil and the sizing table both pin it.

    Above that ceiling the cap was a THROTTLE. At B = 2000 twelve members
    converge long before the budget is gone: late generations propose
    recombinations the population already holds, those hit the evaluation
    cache, and the generation count -- which is a cap on generations, not on
    charges -- runs out with the budget unspent. Measured, six of six cheap
    cells spent 969 to 1212 of 2000 charges while the uniform comparator spent
    1696 to 1842, so the matched-budget comparison was decided by how much each
    arm could spend and not by how well it was guided; on recall per
    EVALUATION the same cells read at parity or better. So the population
    grows with the budget (one member per 32 charges, floored at the old cap of
    twelve and ceilinged at 64, where the per-generation selection cost starts
    to be the thing being paid for) and the offspring count follows it. The
    generations formula is not restated here; it is
    :func:`_generation_cap`, which had to be repaired for exactly this reason
    and says so in its own docstring.
    """

    if int(budget) > _SEALED_BUDGET_CEILING:
        pop = min(64, max(12, int(budget) // 32))
        return pop, pop - 2
    pop = max(4, min(budget // 4, 12))
    return pop, max(2, pop - 2)


#: The largest offspring count any sealed row ever ran with -- the old
#: population cap of twelve, less its two elites. Every sealed generation cap
#: was ``4 * budget // offspring`` with an offspring count at or under this
#: number, so it is the divisor that pins what "four times the budget's worth
#: of generations" MEANT when it was measured.
_SEALED_OFFSPRING_CEILING = 10


def _generation_cap(budget: int, offspring: int) -> int:
    """How many generations the loop may run. A CAP, never a schedule.

    Duplicate offspring are served from the evaluation cache without spending
    budget, so the loop's real stop condition is the budget and this number
    exists only to bound a run that has stopped making progress. It is
    therefore a defect when it binds first -- and it did.

    THE DEFECT, and the measurement that found it. The expression was
    ``4 * budget // offspring``. Below the sealed ceiling the offspring count
    is at most ten, so it bought at least ``0.4 * budget`` generations: far
    more than the budget could ever pay for, which is what a cap should be.
    Above the ceiling ``_genetic_sizing`` grows the population with the budget
    -- and the offspring count with it -- so the SAME expression bought fewer
    and fewer generations as the budget rose. At budget 2000 the population is
    62, the offspring count 60, and the cap collapses to 133 generations.
    Measured on the cheap tier: the operator-portfolio arm, whose arms
    re-propose recombinations the population already holds, spent a mean
    1615.1 evaluations of 2000 (minimum 1418) while its comparator -- which
    runs the same sizing but produces fewer duplicates -- spent 2000.0 on 12
    of 12 cells. The matched-budget comparison was decided by how much each
    arm could SPEND, and 12 of 20 truncated cells were still gaining recall on
    their last mark when the cap ended them.

    THE FIX, and its guard. The divisor is bounded at
    :data:`_SEALED_OFFSPRING_CEILING`, so the cap keeps the
    generations-per-budget ratio the sealed rows were measured with instead of
    shrinking underneath the new sizing. The guard is the same one
    :func:`_genetic_sizing` uses and is spelled the same way: at or below
    ``_SEALED_BUDGET_CEILING`` the old expression is evaluated literally, and
    because the population there is capped at twelve the two branches also
    agree everywhere the offspring count is ten or fewer -- inert to budget
    415 inclusive, not merely to 384.

    CHANGELOG-ready wording: *"Above a budget of 384 the generation cap no
    longer shrinks as the population grows. Large-budget runs -- particularly
    with ``authorship.operators``, whose arms produce more duplicates --
    previously ended with up to 20% of the evaluation budget unspent. Sealed
    budgets are unaffected."*
    """

    offspring = max(1, int(offspring))
    if int(budget) <= _SEALED_BUDGET_CEILING:
        return max(1, 4 * int(budget) // offspring)
    return max(1, 4 * int(budget) // min(offspring, _SEALED_OFFSPRING_CEILING))


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


def _priced_usage(
    route: str, input_tokens: int, output_tokens: int
) -> tuple[Optional[str], str]:
    """``(cost_usd, reported_by)`` for a run's token counts.

    The two halves of a cost figure come from different places and the reporter
    string says which is which: the token counts are the provider's own, the
    per-million prices are this package's published table -- the same numbers
    the CLI echoes before it spends anything. Keeping the provenance in the
    field means nobody has to guess later whether a dollar figure was billed or
    computed.

    An unpriced route returns ``None``. A cost that cannot be derived is
    reported as unknown rather than as zero, because zero reads as "nothing was
    spent" when it means "nobody looked".
    """
    from decimal import Decimal

    from agent_evolve.settings import model_price

    measured = "openrouter response usage"
    price = model_price(route)
    if price is None:
        return None, measured
    per_m_in, per_m_out = price
    million = Decimal(1_000_000)
    cost = (
        Decimal(str(per_m_in)) * Decimal(input_tokens) / million
        + Decimal(str(per_m_out)) * Decimal(output_tokens) / million
    )
    return (
        str(cost.quantize(Decimal("0.000001"))),
        f"{measured}; cost derived from the package's published price table",
    )


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


def _llm_refusal_message(*, extra_missing: bool) -> str:
    """The explicit-llm refusal, naming every way out that applies.

    On a core install the stranger who asks for a model is missing TWO
    things, and the fix a message names first should be the one they hit
    first: the optional dependencies, then the credential. On an install
    that already has the extra, naming it would be noise. The CI stranger
    job holds the extra-missing rendering to actually naming the extra.
    """

    fix = (
        "Install the model path's optional dependencies with: pip install "
        "'agentevolve-optimizer[llm]'. Then set OPENROUTER_API_KEY (or "
        "AGENTEVOLVE_DOTENV naming a file that does)"
        if extra_missing else
        "Set OPENROUTER_API_KEY (or AGENTEVOLVE_DOTENV naming a file that "
        "does)"
    )
    return (
        "proposer='llm' was asked for by name, but no provider credential "
        f"is configured, so no model can be called. {fix}, or run with "
        "proposer='random', which needs nothing -- or proposer='auto', "
        "which chooses it out loud."
    )


def _moved(value: Any, default: tuple) -> bool:
    """Whether a pair-valued knob was actually moved off its default.

    ``[0.5, 0.1]`` and ``(0.5, 0.1)`` are the same request, so the comparison
    is on contents rather than on type; anything that is not a sequence at all
    reads as moved, and the loop then refuses it by name. Used only to decide
    whether the caller ASKED for something the authoring strategy cannot
    honour -- a knob nobody moved is this package's own default and refusing a
    run over it would be the package arguing with itself.
    """

    try:
        return tuple(value) != default
    except TypeError:
        return True


def _check_structure_budget(structure_budget: int, budget: int) -> None:
    """The screen is charged against the search it informs, so it must fit."""

    if structure_budget >= budget:
        raise ValueError(
            f"structure_budget ({structure_budget}) must leave room inside the "
            f"budget ({budget}): the screen is charged against the same budget "
            "as the search it informs"
        )


def _resolve_guidance(
    prior: Any,
    structure_budget: Any,
    *,
    budget: int,
    model_calls: bool,
    announce: Callable[[str], None],
) -> tuple[str, int]:
    """Turn the ``"auto"`` sentinels into the stack the measurements bought.

    Two rules, and which one applies is decided by whether a model call is
    actually possible -- not by what the caller hoped for.

    Without a model call the sentinels resolve to ``"rule"`` and ``0``, which
    are the literal pre-sentinel defaults: the credential-free path draws the
    same candidates in the same order, and the fossil stream cannot move.

    With one, the screen is sized from the budget and ``prior`` becomes
    ``"llm-weighted"`` exactly when that screen will run. Below 48 evaluations
    both stay off: the six-arm ablation screened at 15 evaluations of 96, at a
    small budget that share buys less than the initialization seam alone (the
    measured winner there), and the prior seat only ever acts on a screen's
    evidence.
    """

    if not model_calls:
        return ("rule" if prior == "auto" else prior,
                0 if structure_budget == "auto" else structure_budget)
    if structure_budget == "auto":
        structure_budget = 0 if budget < 48 else min(16, max(8, budget // 6))
        if structure_budget:
            announce(
                f"structure_budget={structure_budget} by default at budget "
                f"{budget}: the six-arm ablation screened at 15 evaluations of "
                "96, and the screen is charged against the same budget. Below "
                "48 it is skipped. Pass structure_budget=0 to skip it here."
            )
    if prior == "auto":
        # The prior seat only acts on a screen's evidence, so the model form
        # is bought exactly when a screen will run. Announcing a model prior
        # beside structure_budget=0 would be a promise the run never cashes.
        if structure_budget:
            prior = "llm-weighted"
            announce(
                "prior='llm-weighted' by default on a model run: the model "
                "reads the crossed screen and the screen's own statistics "
                "carry the weights (the six-arm ablation's guidance arm). "
                "Pass prior='rule' for the credential-free comparator."
            )
        else:
            prior = "rule"
    return prior, structure_budget


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
    structure_budget: int | str = "auto",
    prior: str = "auto",
    chooser: str = "off",
    effort: Optional[str] = None,
    journal: Any = None,
    authorship: Any = "auto",
    polish: str = "off",
    survival: str = "count",
    explore: str = "off",
    explore_schedule: tuple = _EXPLORE_SCHEDULE,
    intensify: str = "off",
    intensify_fraction: float = _INTENSIFY_FRACTION,
    intensify_pin_range: tuple = _INTENSIFY_PIN_RANGE,
    intensify_burst: int = 0,
    elite_mix: float = 0.0,
) -> SearchResult:
    """Optimize *problem* within *budget* evaluations.

    *budget* counts artifacts measured, which is the expensive thing and the
    only sizing number the caller supplies. The problem's seeds are evaluated
    before anything is proposed, so the result always answers "did this beat
    what I already had".

    *structure_budget* spends that many evaluations -- charged against the same
    *budget* -- on a crossed screen before the population is built; *prior*
    names who turns the screen into a sampling prior: the credential-free
    ``"rule"`` or ``"rule-weighted"``, or their model-backed forms ``"llm"`` /
    ``"llm-weighted"``, which fall back to the rule comparator, out loud, when
    no model call is possible. Both default to ``"auto"``, which resolves
    against what the run can actually do: without a model call, to ``"rule"``
    and ``0`` -- the literal pre-sentinel defaults, so the credential-free path
    stays byte-identical -- and with one, to ``"llm-weighted"`` and a screen
    sized from the budget, announced through *on_progress* rather than picked
    silently. ``"llm-weighted-committed"`` is the tuning-round variant under
    measurement: ``"llm-weighted"`` with the prompt's leave-a-locus-free
    caution swapped for evidence-proportional commitment.

    *chooser* names who picks parents and cut points inside a generation, and
    defaults to ``"off"``. ``"llm"`` buys the per-offspring chooser, which is
    the one mechanism here that has never earned its price: ten sealed null
    verdicts, Theta(offspring) model calls rather than one, and 61% of the
    six-arm ablation's whole ledger consumed for 0.94x the speed of doing
    nothing. ``"off"`` runs the random control it never beat. It needs a run
    that makes model calls; asking for it on a run that cannot is refused
    rather than ignored.

    *polish* (``"off"`` / ``"sweep"``) lets a stalled endgame enumerate the
    1-mutation neighbourhood of the front instead of breeding it again, which
    is the move NSGA-II already makes: it beats this loop to the EXACT optimum
    6W/4L on ten NAS seeds while losing 1W/9L to it at 10% of optimum, so the
    gap is the last grid step and nothing else. *survival* (``"count"`` /
    ``"crowding"``) decides survival among equally-dominated members by
    NSGA-II crowding distance, which is what makes a many-objective unit
    selectable: on the five-objective fleet unit almost nothing dominates
    anything and the count-only rule is near-random. Both default to the
    byte-identical setting and both belong to the genetic strategy.

    *explore* (``"off"`` / ``"coverage"``) and *intensify* (``"off"`` /
    ``"incumbent"``) are the two halves of one fix, and neither is expected to
    pay alone. Exploration spends a declining share of each generation's
    offspring slots on a fresh draw from each locus's least-measured DECLARED
    values -- the schema's whole domain, never the prior's narrowed support --
    because on the analog venue every screen-fitted prior installed a ceiling
    the venue's optimum sits above: best in-box evaluation -0.4002 reward9
    against -0.0566 outside, with 100 of the pooled top-100 outside.
    Intensification spends a further share on the current best member with 6-12
    of its loci pinned and the rest resampled, which is what a third-party
    optimizer's "coordinated combinations" turn out to be when its winning
    populations are measured. Alone, exploration buys uniform sampling of a
    bigger region: the two of our cells that already spend 82-96% of their
    charges outside the box come back no better than the cells that stay in.
    *explore_schedule* ``(e0, e_min)``, *intensify_fraction* and
    *intensify_pin_range* are the declared parameters, at their measured
    values; both mechanisms take offspring SLOTS rather than extra charges, so
    neither changes what a budget buys. Genetic-strategy knobs, both
    default-off, and off is byte-identical.

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
    if structure_budget != "auto":
        if (not isinstance(structure_budget, int)
                or isinstance(structure_budget, bool) or structure_budget < 0):
            raise ValueError(
                f"structure_budget must be 'auto' or a non-negative integer, "
                f"got {structure_budget!r}"
            )
        _check_structure_budget(structure_budget, budget)
    if prior != "auto" and prior not in _PRIORS:
        raise ValueError(
            f"prior must be 'auto' or one of {sorted(_PRIORS)}, got {prior!r}")
    if chooser not in ("off", "llm"):
        raise ValueError(f"chooser must be 'off' or 'llm', got {chooser!r}")
    if polish not in ("off", "sweep"):
        raise ValueError(f"polish must be 'off' or 'sweep', got {polish!r}")
    if survival not in ("count", "crowding"):
        raise ValueError(
            f"survival must be 'count' or 'crowding', got {survival!r}")
    if explore not in ("off", "coverage"):
        raise ValueError(f"explore must be 'off' or 'coverage', got {explore!r}")
    if intensify not in ("off", "incumbent"):
        raise ValueError(
            f"intensify must be 'off' or 'incumbent', got {intensify!r}")
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
    if chooser == "llm" and kind != "llm":
        # A chooser that cannot call a model is a chooser that never chooses,
        # and the run would look exactly like the one that never asked for it.
        raise ValueError(
            f"chooser='llm' asks a model to pick parents and cut points, and "
            f"this run resolved to the {kind!r} proposer, which makes no model "
            "call. Pass proposer='llm' with a provider credential, or drop "
            "chooser= to keep the random control."
        )

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
        # The sentinels are read as "not asked for": ``auto`` is this package
        # choosing, and refusing a run over a choice the caller never made
        # would be the package arguing with itself.
        engaged = [name for name, on in (
            ("structure_budget", structure_budget not in ("auto", 0)),
            ("prior", prior not in ("auto", "rule")),
            ("chooser", chooser == "llm"),
            ("polish", polish != "off"),
            ("survival", survival != "count"),
            ("explore", explore != "off"),
            ("explore_schedule", _moved(explore_schedule, _EXPLORE_SCHEDULE)),
            ("intensify", intensify != "off"),
            ("intensify_fraction", intensify_fraction != _INTENSIFY_FRACTION),
            ("intensify_pin_range",
             _moved(intensify_pin_range, _INTENSIFY_PIN_RANGE)),
            ("intensify_burst", intensify_burst != 0),
            ("elite_mix", elite_mix != 0.0),
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
            chooser_policy = None
            complete = None
            # Provider usage is measured from the completion seam's own
            # journal, never declared: zero means "counted and none occurred".
            usage_ledger = {"calls": 0, "input": 0, "output": 0, "tokens_known": True}
            if kind == "llm":
                # The completion seam is built for the whole run, not for one
                # consumer. It used to be constructed inside the chooser's own
                # branch, which meant the seams that measured well -- authored
                # initialization, the weighted prior -- could only be bought
                # together with the one that measured null.
                from agent_evolve.integrations.completion import completion_for

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

                # The shipped completion ceiling comes from the profile the
                # product already declares for the route, not from the
                # provider's undeclared default. Sending nothing was never
                # "no cap": it was 65,536 on the default route, against the
                # 128,000 the profile declares -- and the half that went
                # missing was taken from the calls that reasoned longest.
                # An unknown route still declares nothing, and then nothing
                # is sent, so that path keeps the pre-cap body exactly.
                from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E501
                    declared_max_output_tokens)

                route = model or settings.model
                cap = declared_max_output_tokens(route)
                complete = completion_for(route, settings,
                                          journal=_record_usage, effort=effort,
                                          max_output_tokens=cap)
                if complete is None:
                    # The caller asked for a model BY NAME and no credential
                    # can honour it. Falling back to the classical path here
                    # ran to completion and said nothing -- a run launched to
                    # measure a model measured the control instead, and the
                    # only trace was `calls: 0`. Found by the release CI's
                    # stranger job, 2026-08-20.
                    import importlib.util
                    raise RuntimeError(_llm_refusal_message(
                        extra_missing=importlib.util.find_spec("pydantic_ai")
                        is None))
            if chooser == "llm":
                if complete is None:
                    announce(
                        "chooser='llm' needs a model call and none is "
                        "available; operator choices stay random."
                    )
                else:
                    # Guided operator choice: the model picks parents and cut
                    # points, reasoning over the accumulated search state. It
                    # cannot author a candidate -- OperatorChoice has no field
                    # that could hold one. Opt-in, because it is the one seam
                    # here with ten sealed null verdicts against it.
                    from agent_evolve.policies.llm_chooser import llm_chooser
                    from agent_evolve.policies.semantics import domain_card
                    chooser_policy = llm_chooser(
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

            # Resolved here and not earlier: what the sentinels mean depends on
            # whether a model call is actually possible, which is not known
            # until the seam above has either been built or come back empty.
            prior, structure_budget = _resolve_guidance(
                prior, structure_budget, budget=budget,
                model_calls=complete is not None, announce=announce)
            _check_structure_budget(structure_budget, budget)

            prior_proposer: Any = None
            if prior == "rule-weighted":
                from agent_evolve.policies.weighted_prior import (
                    statistical_weighted_prior)
                prior_proposer = statistical_weighted_prior
            elif prior in ("llm", "llm-weighted", "llm-weighted-committed"):
                if complete is None:
                    announce(
                        f"prior={prior!r} needs a model call and none is "
                        "available; using the credential-free rule comparator "
                        "instead."
                    )
                    if prior != "llm":
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
                    # The two model-weighted forms differ by ONE clause of the
                    # prompt; everything downstream of the reply is shared.
                    prior_proposer = llm_weighted_prior_proposer(
                        complete, objectives=list(bound.objectives),
                        domain_context=domain_card(bound),
                        style=("committed"
                               if prior == "llm-weighted-committed"
                               else "cautious"))

            if authorship_config is None:
                if complete is not None:
                    authorship_config = AuthorshipConfig(surrogate="llm",
                                                         initialization="llm")
                    announce(
                        "authorship: model-authored surrogate screening is ON "
                        "(the sealed luna-clear row held), and model-proposed "
                        "initialization is ON -- the six-arm ablation's "
                        "strongest arm, at 11x fewer evaluations to target, "
                        "better on 40 of 40 paired seeds, for one call; pass "
                        "authorship='off' to disable.")
                else:
                    authorship_config = AuthorshipConfig()
            # One rule, stated once, in `_genetic_sizing`: the sealed
            # expression at and below the sealed ceiling, a population that
            # grows with the budget above it.
            pop, offspring = _genetic_sizing(budget)

            from agent_evolve.policies.semantics import domain_card
            from agent_evolve.session.authorship import build_authorship
            policies = build_authorship(
                authorship_config, complete=complete,
                objectives=list(bound.objectives),
                schema_text=domain_card(bound), seed=seed, announce=announce,
                candidate_model=getattr(bound, "candidate_model", None),
                init_template=(dict(seeds[0]) if seeds else None),
                init_k=max(0, pop - len(seeds)),
                budget=budget, population_size=pop)

            cache = EvaluationCache()
            cache.budget = budget
            # `generations` is a cap, not a schedule. Duplicate offspring hit
            # the evaluation cache without spending budget, so a fixed
            # generation count would end the run with budget unspent; the
            # loop's real stop condition is the budget. `_generation_cap` is
            # where that sentence is kept true -- it stopped being true above
            # the sealed ceiling, and its docstring carries the measurement.
            result = run_genetic_loop(
                problem=bound,
                config=GeneticConfig(
                    population_size=pop,
                    offspring_per_generation=offspring,
                    generations=_generation_cap(budget, offspring),
                    seed=seed,
                    seeds=seeds,
                    evaluation_budget=budget,
                    evaluation_cache=cache,
                    structure_budget=structure_budget,
                    prior_proposer=prior_proposer,
                    screening=policies.screening,
                    portfolio=policies.portfolio,
                    initial_proposals=policies.initial_proposals,
                    generator=policies.generator,
                    reguidance=policies.reguidance,
                    polish=polish,
                    survival=survival,
                    explore=explore,
                    # Handed through as given rather than coerced: the loop
                    # validates both pairs by name, and a coercion here would
                    # turn its message into a bare TypeError from a tuple().
                    explore_schedule=explore_schedule,
                    intensify=intensify,
                    intensify_fraction=intensify_fraction,
                    intensify_pin_range=intensify_pin_range,
                    intensify_burst=intensify_burst,
                    elite_mix=elite_mix,
                ),
                chooser=chooser_policy,
                log=announce,
            )
            # Authoring that produced no policy object still produced
            # counters, and the loop can only harvest what it was handed.
            # These are the seams whose failure leaves nothing behind.
            orphaned = tuple(note for note in (policies.init_author,
                                               policies.generator_author,
                                               policies.reguidance_author)
                             if note is not None)
            if orphaned and result.telemetry is not None:
                from agent_evolve.core.telemetry import harvest_telemetry
                extra = harvest_telemetry(orphaned)
                result = dataclasses.replace(
                    result,
                    telemetry=dataclasses.replace(
                        result.telemetry,
                        mechanisms=result.telemetry.mechanisms + extra.mechanisms))
            if usage_ledger["calls"] and usage_ledger["tokens_known"]:
                # Cost is DERIVED, and the reporter says so. The tokens are the
                # provider's own count; the price is this package's published
                # table (`MODEL_PRICES_PER_MTOK`), which is the same number the
                # CLI echoes before spending anything. A route the table does
                # not name reports `cost_usd: null` -- unknown stays unknown
                # rather than becoming a guess with a dollar sign on it.
                route = model or settings.model
                cost_usd, reporter = _priced_usage(
                    route, usage_ledger["input"], usage_ledger["output"])
                usage = ProviderUsageSummary(
                    calls=usage_ledger["calls"],
                    input_tokens=usage_ledger["input"],
                    output_tokens=usage_ledger["output"],
                    cost_usd=cost_usd,
                    model=route,
                    reported_by=reporter,
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
