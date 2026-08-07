"""The authorship substrate's one public knob, and its factory.

``AuthorshipConfig`` names who authors which machinery -- surrogates today,
variation operators next -- and :func:`build_authorship` turns it into the
policy objects the genetic loop consumes. Everything defaults to off, and
off is byte-identical to the pre-substrate loop (the fossil test holds it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from agent_evolve.infrastructure.authored_runtime import RuntimeLimits

__all__ = ["AuthorshipConfig", "AuthorshipPolicies", "build_authorship"]

_SURROGATE = ("off", "rule", "llm")
_OPERATORS = ("off", "rule", "llm")


@dataclass(frozen=True)
class AuthorshipConfig:
    """Who authors which machinery, and under what bounds.

    ``surrogate="rule"`` turns on virtual pre-screening with the shipped,
    credential-free surrogates behind the validation gate. The ``"llm"``
    values and the operator portfolio land behind the same fields as the
    substrate grows; naming one that has not landed is an error today rather
    than a silent no-op forever.
    """

    surrogate: str = "off"
    operators: str = "off"
    pool_factor: int = 4
    exploration_floor: float = 0.25
    authoring_attempts: int = 2
    max_authored_fraction: float = 0.5
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)

    def __post_init__(self) -> None:
        if self.surrogate not in _SURROGATE:
            raise ValueError(
                f"authorship.surrogate must be one of {_SURROGATE}, got "
                f"{self.surrogate!r}"
            )
        if self.operators not in _OPERATORS:
            raise ValueError(
                f"authorship.operators must be one of {_OPERATORS}, got "
                f"{self.operators!r}"
            )

    @property
    def engaged(self) -> bool:
        return self.surrogate != "off" or self.operators != "off"

    @classmethod
    def preset(cls, name: str) -> "AuthorshipConfig":
        presets = {
            "off": cls(),
            "surrogate": cls(surrogate="rule"),
            "surrogate-llm": cls(surrogate="llm"),
            "operators": cls(operators="rule"),
            "operators-llm": cls(operators="llm"),
            "full": cls(surrogate="llm", operators="llm"),
        }
        if name not in presets:
            raise ValueError(
                f"authorship preset must be one of {sorted(presets)}, got "
                f"{name!r}"
            )
        return presets[name]


@dataclass(frozen=True)
class AuthorshipPolicies:
    """What the factory built: policy objects the loop consumes directly."""

    screening: Optional[Any] = None
    portfolio: Optional[Any] = None


def build_authorship(
    config: AuthorshipConfig,
    *,
    complete: Any = None,
    objectives: Sequence[Any] = (),
    schema_text: str = "",
    seed: Optional[int] = None,
    announce: Optional[Callable[[str], None]] = None,
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
    return AuthorshipPolicies(
        screening=_build_screening(config, complete, objectives,
                                   schema_text, say),
        portfolio=_build_portfolio(config, complete, objectives,
                                   schema_text, say),
    )


def _build_screening(
    config: AuthorshipConfig,
    complete: Any,
    objectives: Sequence[Any],
    schema_text: str,
    say: Callable[[str], None],
) -> Optional[Any]:
    if config.surrogate == "off":
        return None
    from agent_evolve.policies.surrogate import additive_surrogate, knn_surrogate
    from agent_evolve.session.screening import Screening

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
    screening = Screening(
        builders=tuple(builders),
        pool_factor=config.pool_factor,
        exploration_floor=config.exploration_floor,
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
