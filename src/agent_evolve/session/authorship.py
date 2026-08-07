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

__all__ = ["AuthorshipConfig", "build_authorship"]

_SURROGATE = ("off", "rule", "llm")
_OPERATORS = ("off",)


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
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)

    def __post_init__(self) -> None:
        if self.surrogate not in _SURROGATE:
            raise ValueError(
                f"authorship.surrogate must be one of {_SURROGATE}, got "
                f"{self.surrogate!r}"
            )
        if self.operators not in _OPERATORS:
            raise ValueError(
                f"authorship.operators must be one of {_OPERATORS} for now "
                f"(the operator portfolio is the next layer), got "
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
        }
        if name not in presets:
            raise ValueError(
                f"authorship preset must be one of {sorted(presets)}, got "
                f"{name!r}"
            )
        return presets[name]


def build_authorship(
    config: AuthorshipConfig,
    *,
    complete: Any = None,
    objectives: Sequence[Any] = (),
    schema_text: str = "",
    seed: Optional[int] = None,
    announce: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """The screening policy for *config*, or ``None`` when nothing is on.

    With ``surrogate="llm"`` the model is asked ONCE, before any evaluation,
    to author a surrogate from the schema and objective meanings; the
    authored builder then competes against the rule builders under the same
    per-generation validation gate, and the best-passing one screens. No
    usable authorship -- no credential, no code block, forbidden imports --
    degrades to the rule builders, out loud, never silently.
    """

    del seed
    say = announce or (lambda _m: None)
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
