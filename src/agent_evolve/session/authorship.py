"""The authorship substrate's one public knob, and its factory.

``AuthorshipConfig`` names who authors which machinery -- surrogates today,
variation operators next -- and :func:`build_authorship` turns it into the
policy objects the genetic loop consumes. Everything defaults to off, and
off is byte-identical to the pre-substrate loop (the fossil test holds it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from agent_evolve.infrastructure.authored_runtime import RuntimeLimits

__all__ = ["AuthorshipConfig", "AuthorshipPolicies", "PRESETS",
           "build_authorship"]

_SURROGATE = ("off", "rule", "llm")
_OPERATORS = ("off", "rule", "llm")
_INITIALIZATION = ("off", "llm")
_GENERATION = ("off", "llm")

#: The named compositions, as field settings. A table rather than a method
#: body so that everything downstream -- the CLI's ``--authorship`` choices,
#: any campaign script -- ENUMERATES what exists instead of repeating a list
#: that then drifts.
PRESETS: Mapping[str, Mapping[str, str]] = {
    "off": {},
    "surrogate": {"surrogate": "rule"},
    "surrogate-llm": {"surrogate": "llm"},
    "operators": {"operators": "rule"},
    "operators-llm": {"operators": "llm"},
    "init-llm": {"initialization": "llm"},
    "generation-llm": {"generation": "llm"},
    # The authored sampler under the frozen screen stack: the model writes
    # where candidates come from, and the variance-guarded authored surrogate
    # decides which of them are worth measuring.
    "generative": {"generation": "llm", "surrogate": "llm"},
    "full": {"surrogate": "llm", "operators": "llm", "initialization": "llm"},
}


@dataclass(frozen=True)
class AuthorshipConfig:
    """Who authors which machinery, and under what bounds.

    ``surrogate="rule"`` turns on virtual pre-screening with the shipped,
    credential-free surrogates behind the validation gate. The ``"llm"``
    values put the model in the authoring seat for that piece of machinery --
    the surrogate, the variation operators, the initial population, or
    ``generation``, where it writes the sampler every candidate is drawn
    from. Naming a value that has not landed is an error today rather than a
    silent no-op forever.
    """

    surrogate: str = "off"
    operators: str = "off"
    initialization: str = "off"
    generation: str = "off"
    pool_factor: int = 4
    exploration_floor: float = 0.25
    #: How many objectives the screen's gate must certify before the screen
    #: may order anything: ``0`` = every declared objective (the conjunction),
    #: a positive value = that many, with the screen ordering on exactly the
    #: certified ones. See policies.surrogate.GatePolicy.min_passing_objectives.
    screen_min_passing_objectives: int = 0
    #: The share of a generation reserved from a PARTIAL screen, as a multiple
    #: of the share of objectives it could not see. See
    #: session.screening.Screening.unscreened_objective_floor.
    screen_unscreened_objective_floor: float = 1.0
    authoring_attempts: int = 2
    max_authored_fraction: float = 0.5
    #: How many of the most recent measurements a screen refresh fits and
    #: validates on (see session.screening.Screening.max_training_rows). The
    #: screen re-arbitrates every generation, so this is what decides whether
    #: a high-budget run's screening cost is constant or grows with the run.
    screen_training_rows: int = 1024
    #: Mass generation's pool: ``generation_pool_factor`` times the offspring
    #: a generation can afford, or exactly ``generation_pool_size`` when that
    #: is set. The pool costs no evaluations, so it is sized by what the
    #: sampler and the screen can chew through, not by the budget.
    generation_pool_factor: int = 4
    generation_pool_size: int = 0
    #: One-shot authoring is what the ladder cells measure; revision LEVELS
    #: the rungs (W3), so it is capped here and ablatable to 0.
    generation_revisions: int = 1
    #: Keep an authored revision only when a FROZEN replay measures it
    #: better -- strictly lower defect rate, no loss of novelty. Off by
    #: default because every sealed row is defined on unguarded revision;
    #: it is the third arm of the revision-value row, not a silent change.
    generation_revision_guard: bool = False
    #: Ship the emit harness into the sandbox (the authored code builds
    #: candidates through ``build``) and assemble a partially-correct
    #: emission rather than dropping it whole. Both default ON: they are the
    #: fix for the measured assignment-genome authoring failure, and both are
    #: ablatable so the fix can be measured against its own absence.
    generation_scaffold: bool = True
    generation_repair: bool = True
    #: Echo the sandbox's own wall/CPU/memory budget into the authoring
    #: prompt, and retry a batch that overran it at ``n // 4`` rather than
    #: losing the whole pool. An unstated budget is a budget the author
    #: cannot honour, and on `upms_j14_m3` the overrun -- not the shape --
    #: is what emptied most batches.
    generation_limits_echo: bool = True
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)

    def __post_init__(self) -> None:
        if self.surrogate not in _SURROGATE:
            raise ValueError(
                f"authorship.surrogate must be one of {_SURROGATE}, got "
                f"{self.surrogate!r}"
            )
        if self.initialization not in _INITIALIZATION:
            raise ValueError(
                f"authorship.initialization must be one of {_INITIALIZATION}, "
                f"got {self.initialization!r}"
            )
        if self.operators not in _OPERATORS:
            raise ValueError(
                f"authorship.operators must be one of {_OPERATORS}, got "
                f"{self.operators!r}"
            )
        if self.generation not in _GENERATION:
            raise ValueError(
                f"authorship.generation must be one of {_GENERATION}, got "
                f"{self.generation!r}"
            )
        if self.generation != "off" and self.operators != "off":
            # Both construct the generation's candidates. Accepting the pair
            # would silently run one of them and bill the caller for two.
            raise ValueError(
                "authorship.generation and authorship.operators both "
                "construct the generation's candidates -- the generator draws "
                "the pool, the operator arms recombine parents into it. Ask "
                "for one."
            )

    @property
    def engaged(self) -> bool:
        return (self.surrogate != "off" or self.operators != "off"
                or self.initialization != "off" or self.generation != "off")

    @classmethod
    def preset(cls, name: str) -> "AuthorshipConfig":
        if name not in PRESETS:
            raise ValueError(
                f"authorship preset must be one of {sorted(PRESETS)}, got "
                f"{name!r}"
            )
        return cls(**PRESETS[name])


@dataclass(frozen=True)
class AuthorshipPolicies:
    """What the factory built: policy objects the loop consumes directly."""

    screening: Optional[Any] = None
    portfolio: Optional[Any] = None
    initial_proposals: tuple = ()
    init_author: Optional[Any] = None
    generator: Optional[Any] = None
    #: Set only when generation was asked for and produced no generator: the
    #: authoring counters would otherwise have nowhere to live, and "the
    #: model failed to author a sampler" would be indistinguishable from
    #: "nobody asked". When a generator exists it carries its own note.
    generator_author: Optional[Any] = None


def build_authorship(
    config: AuthorshipConfig,
    *,
    complete: Any = None,
    objectives: Sequence[Any] = (),
    schema_text: str = "",
    seed: Optional[int] = None,
    announce: Optional[Callable[[str], None]] = None,
    candidate_model: Any = None,
    init_template: Any = None,
    init_k: int = 0,
) -> AuthorshipPolicies:
    """The policy objects for *config*; fields are ``None`` where nothing is on.

    With an ``"llm"`` value the model is asked ONCE, before any evaluation,
    to author from the schema and objective meanings; authored machinery then
    competes against the shipped rules under measurement -- the validation
    gate for surrogates, survival credit for operators. No usable authorship
    (no credential, no code block, forbidden imports) degrades to the rules,
    out loud, never silently.
    """

    say = announce or (lambda _m: None)
    proposals, init_note = _build_initialization(
        config, complete, schema_text, say, candidate_model,
        init_template, init_k)
    generator, generator_note = _build_generator(
        config, complete, objectives, schema_text, say,
        candidate_model, init_template)
    return AuthorshipPolicies(
        screening=_build_screening(config, complete, objectives,
                                   schema_text, say),
        portfolio=_build_portfolio(config, complete, objectives,
                                   schema_text, say),
        initial_proposals=proposals,
        init_author=init_note,
        generator=generator,
        generator_author=generator_note,
    )


def _build_initialization(config, complete, schema_text, say,
                          candidate_model, template, k):
    if config.initialization == "off":
        return (), None
    from agent_evolve.policies.llm_init import (InitTelemetry,
                                                author_initial_population)
    telemetry = InitTelemetry()
    note = SimpleNamespace(telemetry=telemetry, mechanism="init_author",
                           authored_by="llm")
    if complete is None:
        say("authorship.initialization='llm' needs a model call and none is "
            "available; initialization stays schema-uniform.")
        return (), note
    if template is None or not k:
        say("authorship.initialization='llm' received no template/size; "
            "initialization stays schema-uniform.")
        return (), note
    proposals = author_initial_population(
        complete, candidate_model=candidate_model, template=dict(template),
        k=int(k), domain_context=schema_text, telemetry=telemetry)
    if not proposals:
        say("the model proposed no usable initial members; initialization "
            "stays schema-uniform.")
    return tuple(proposals), note


def _build_generator(
    config: AuthorshipConfig,
    complete: Any,
    objectives: Sequence[Any],
    schema_text: str,
    say: Callable[[str], None],
    candidate_model: Any = None,
    template: Any = None,
) -> tuple:
    """``(generator, author_note)``; both ``None`` when generation is off.

    One authoring call before any evaluation buys a distribution that shapes
    every candidate the run draws -- the amortization that makes this the
    mechanism for cheap-evaluation venues, where per-decision guidance has
    leverage 1/budget. No usable authorship degrades to the shipped
    schema-uniform sampler, out loud: the loop simply keeps drawing the way
    it always did, and the note carries the counters that say why.
    """

    if config.generation == "off":
        return None, None
    from agent_evolve.policies.llm_generator import AuthoredGenerator
    from agent_evolve.policies.llm_surrogate import AuthorTelemetry

    telemetry = AuthorTelemetry()
    author_note = SimpleNamespace(
        telemetry=telemetry, mechanism="generator_author", authored_by="llm"
    )
    if complete is None:
        say("authorship.generation='llm' needs a model call and none is "
            "available; candidates stay schema-uniform.")
        return None, author_note
    from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
    from agent_evolve.policies.llm_generator import (author_generator,
                                                     revise_generator)

    # The per-locus admissible sets the run will actually pass, echoed into
    # the authoring prompt. Derived from the problem's own schema exactly as
    # the sampler derives them; absent a template there are no loci to name,
    # and the prompt keeps the field-level card it always had.
    domains = _generator_domains(candidate_model, template)
    artifact = author_generator(
        complete, objectives=list(objectives), schema_text=schema_text,
        attempts=config.authoring_attempts, telemetry=telemetry,
        domains=domains, scaffold=config.generation_scaffold,
        limits=(config.limits if config.generation_limits_echo else None),
        max_n=_generator_max_n(config))
    if artifact is None:
        say(f"the model authored no usable generator in "
            f"{config.authoring_attempts} attempt(s); candidates stay "
            "schema-uniform.")
        return None, author_note

    holder: dict = {}

    def revise(current: Any, feedback: str) -> Any:
        # The evolving generator: its own source plus what the harness
        # measured about the candidates it emitted -- which loci rejected,
        # why, with a sample, and which edits already failed. Same gate as
        # authoring. The echo is the RUN's domains when a batch has been
        # emitted (a restriction may have narrowed them), the declared ones
        # otherwise.
        live = getattr(holder.get("generator"), "_domains", None)
        return revise_generator(
            complete, artifact=current, feedback=feedback,
            telemetry=telemetry, domains=live or domains,
            scaffold=config.generation_scaffold,
            limits=(config.limits if config.generation_limits_echo else None),
            max_n=_generator_max_n(config))

    generator = AuthoredGenerator(
        artifact,
        AuthoredRuntime(limits=config.limits),
        pool_factor=config.generation_pool_factor,
        pool_size=config.generation_pool_size,
        revise=revise if config.generation_revisions > 0 else None,
        max_revisions=config.generation_revisions,
        scaffold=config.generation_scaffold,
        repair=config.generation_repair,
        revision_guard=config.generation_revision_guard,
        shrink_on_overrun=(4 if config.generation_limits_echo else 0),
    )
    holder["generator"] = generator
    generator.author = author_note
    return generator, None


def _generator_max_n(config: "AuthorshipConfig") -> int:
    """The largest pool one call may be asked for, for the prompt's echo.

    Sized from the same two knobs the sampler is: an explicit pool size when
    one is set, otherwise the factor times the offspring a generation can
    afford. The population sizing caps offspring at ten, so this is an upper
    bound rather than a guess.
    """

    if config.generation_pool_size:
        return int(config.generation_pool_size)
    return int(config.generation_pool_factor) * 10


def _generator_domains(candidate_model: Any, template: Any) -> dict:
    """``{locus name: admissible values}`` for the authoring prompt's echo.

    Empty whenever the caller supplied no template or no candidate model --
    there is then nothing to echo, and the prompt is exactly the one every
    sealed row was authored under.
    """

    if candidate_model is None or not template:
        return {}
    from agent_evolve.policies.genetic import loci_of, locus_domain

    try:
        return {str(locus): list(locus_domain(candidate_model, locus))
                for locus in loci_of(dict(template))}
    except Exception:
        return {}


def _build_screening(
    config: AuthorshipConfig,
    complete: Any,
    objectives: Sequence[Any],
    schema_text: str,
    say: Callable[[str], None],
) -> Optional[Any]:
    if config.surrogate == "off":
        return None
    from agent_evolve.policies.surrogate import (
        ORDERING_GATE,
        additive_surrogate,
        knn_surrogate,
    )
    from agent_evolve.session.screening import Screening

    # The screen sorts candidates and never reads a predicted magnitude, so
    # it is gated on rank fidelity and arbitrated on error. The revision hook
    # below must judge the artifact under the SAME policy, or the residual
    # feedback the model receives describes a different bar from the one it
    # is actually held to.
    screening_gate = ORDERING_GATE
    if config.screen_min_passing_objectives:
        # A partial verdict certifies the artifact for the objectives it can
        # order and leaves the rest unknown; the screen then orders on those
        # and reserves more of the generation for unscreened picks.
        screening_gate = ORDERING_GATE.replace(
            min_passing_objectives=config.screen_min_passing_objectives)
    builders: list = []
    author_note: Optional[SimpleNamespace] = None
    if config.surrogate == "llm":
        from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
        from agent_evolve.policies.llm_surrogate import (
            AuthorTelemetry,
            author_surrogate,
            authored_surrogate_builder,
        )

        telemetry = AuthorTelemetry()
        author_note = SimpleNamespace(
            telemetry=telemetry, mechanism="surrogate_author", authored_by="llm"
        )
        if complete is None:
            say(
                "authorship.surrogate='llm' needs a model call and none is "
                "available; the rule surrogates carry the screen."
            )
        else:
            artifact = author_surrogate(
                complete,
                objectives=list(objectives),
                schema_text=schema_text,
                attempts=config.authoring_attempts,
                telemetry=telemetry,
            )
            if artifact is None:
                say(
                    "the model authored no usable surrogate in "
                    f"{config.authoring_attempts} attempt(s); the rule "
                    "surrogates carry the screen."
                )
            else:
                runtime = AuthoredRuntime(limits=config.limits)
                builders.append((
                    f"llm:{artifact.source_sha256[:8]}",
                    "llm",
                    authored_surrogate_builder(artifact, runtime),
                ))

    builders.extend([
        ("additive", "rule", additive_surrogate),
        ("knn", "rule", knn_surrogate),
    ])

    revise = None
    if config.surrogate == "llm" and complete is not None and builders[0][1] == "llm":
        # The evolving surrogate: when the authored artifact loses to the
        # rules, the model sees its own source plus the measured residuals
        # and revises it. Structure from meaning, correction from data.
        # builders[0] being llm guarantees the authoring above succeeded.
        state = {"artifact": artifact}

        def revise(evaluated, specs):
            from agent_evolve.policies.llm_surrogate import (
                render_validation_feedback, revise_surrogate)
            from agent_evolve.policies.surrogate import validate_surrogate

            current = state["artifact"]
            if current is None:
                return None
            builder = authored_surrogate_builder(current, runtime)
            # Under the gate the SCREEN uses, or the feedback would describe a
            # bar the artifact is not actually judged against.
            verdict = validate_surrogate(builder, evaluated, specs, seed=1,
                                         policy=screening_gate)
            try:
                predictions = builder(list(evaluated), specs)(
                    [candidate_config for candidate_config, _obj in evaluated])
            except Exception:
                predictions = None
            feedback = render_validation_feedback(
                verdict, evaluated, predictions)
            revised = revise_surrogate(
                complete, artifact=current, feedback=feedback,
                telemetry=telemetry)
            if revised is None:
                return None
            state["artifact"] = revised
            return (f"llm:{revised.source_sha256[:8]}", "llm",
                    authored_surrogate_builder(revised, runtime))

    screening = Screening(
        builders=tuple(builders),
        pool_factor=config.pool_factor,
        exploration_floor=config.exploration_floor,
        unscreened_objective_floor=config.screen_unscreened_objective_floor,
        revise=revise,
        max_training_rows=config.screen_training_rows,
        gate=screening_gate,
    )
    if author_note is not None:
        # Harvested beside the screen's own counters: how authoring went is
        # part of the run's story even when nothing usable came back.
        screening.author = author_note
    return screening


def _build_portfolio(
    config: AuthorshipConfig,
    complete: Any,
    objectives: Sequence[Any],
    schema_text: str,
    say: Callable[[str], None],
) -> Optional[Any]:
    if config.operators == "off":
        return None
    from agent_evolve.policies.operator_portfolio import (
        OperatorPortfolio,
        VariationArm,
        classical_arm,
        segment_arm,
    )

    arms: list = [classical_arm(), segment_arm()]
    runtime = None
    author_note: Optional[SimpleNamespace] = None
    if config.operators == "llm":
        from agent_evolve.policies.llm_operator import author_operators
        from agent_evolve.policies.llm_surrogate import AuthorTelemetry

        telemetry = AuthorTelemetry()
        author_note = SimpleNamespace(
            telemetry=telemetry, mechanism="operator_author", authored_by="llm"
        )
        if complete is None:
            say(
                "authorship.operators='llm' needs a model call and none is "
                "available; the rule arms carry the portfolio."
            )
        else:
            artifacts = author_operators(
                complete,
                objectives=list(objectives),
                schema_text=schema_text,
                attempts=config.authoring_attempts,
                telemetry=telemetry,
            )
            if not artifacts:
                say(
                    "the model authored no usable operator in "
                    f"{config.authoring_attempts} attempt(s); the rule arms "
                    "carry the portfolio."
                )
            else:
                from agent_evolve.infrastructure.authored_runtime import (
                    AuthoredRuntime)
                runtime = AuthoredRuntime(limits=config.limits)
                arms.extend(
                    VariationArm(
                        name=f"{artifact.name}:{artifact.source_sha256[:8]}",
                        kind="authored", artifact=artifact,
                    )
                    for artifact in artifacts
                )
    portfolio = OperatorPortfolio(
        arms, runtime=runtime,
        max_authored_fraction=config.max_authored_fraction,
    )
    if author_note is not None:
        portfolio.author = author_note
    return portfolio
