"""Problem protocol and objective specification for agent_evolve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, Protocol, Sequence, TypeVar, runtime_checkable

Goal = Literal["min", "max"]
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single optimisation objective."""

    name: str
    goal: Goal


@runtime_checkable
class Problem(Protocol[ConfigT]):
    """Minimal interface that every optimisation problem must satisfy.

    Required
    --------
    objectives : Sequence[ObjectiveSpec]
        The objectives to optimise (at least one).
    evaluate(config) -> Dict[str, float]
        Return objective values for *config*.  Raise ``ValueError`` with a
        descriptive message for invalid / infeasible configurations -- the
        message is forwarded to the LLM as feedback.

    Optional (detected via ``hasattr`` at runtime)
    ------------------------------------------------
    validate(config) -> bool
        Fast feasibility pre-check.  Raise ``ValueError`` for rich feedback.
    search_space_description() -> str
        Human-readable description of the configuration format, valid ranges,
        and constraints.  Included verbatim in LLM prompts.
    """

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]: ...

    def evaluate(self, config: ConfigT) -> Dict[str, float]: ...
