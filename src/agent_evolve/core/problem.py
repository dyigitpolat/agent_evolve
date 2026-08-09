"""Problem protocol, objective specification, and validation outcome."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Literal, Optional, Protocol, Sequence, TypeVar, runtime_checkable

Goal = Literal["min", "max"]
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single optimisation objective.

    ``description`` is the objective's MEANING -- what the number measures,
    its units, what a good value looks like ("spec-attainment reward, sum of
    nine clipped terms, maximised at 0 = every spec met"). It is rendered
    into every model-facing prompt: an optimizer asked to trade objectives
    it cannot interpret is reasoning blindfolded, and the resulting failure
    is unattributable (channel defect vs capability). Optional so existing
    problems keep working; a problem that leaves it empty is telling the
    model "the name is all you get".
    """

    name: str
    goal: Goal
    description: str = ""


class ProblemContractError(RuntimeError):
    """The problem adapter violated its declared objective/evaluation contract."""


def validate_objective_specs(objectives: Sequence[ObjectiveSpec]) -> None:
    """Validate objective declarations before any proposal or evaluation work."""
    if not objectives:
        raise ProblemContractError("Problem must define at least one objective")
    names = [spec.name for spec in objectives]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ProblemContractError("Objective names must be non-empty strings")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ProblemContractError(f"Duplicate objective name(s): {', '.join(duplicates)}")
    invalid_goals = [f"{spec.name}={spec.goal!r}" for spec in objectives
                     if spec.goal not in ("min", "max")]
    if invalid_goals:
        raise ProblemContractError(
            "Objective goals must be 'min' or 'max': " + ", ".join(invalid_goals)
        )


def normalize_objective_values(
    values: Any,
    objectives: Sequence[ObjectiveSpec],
) -> Dict[str, float]:
    """Return a complete finite objective vector or raise ``ProblemContractError``.

    Evaluators must return exactly the declared objectives. Diagnostics belong in
    a separate adapter-level artifact/metadata channel; accepting undeclared keys
    here would make misspelled objective names too easy to overlook.
    """
    validate_objective_specs(objectives)
    if not isinstance(values, Mapping):
        raise ProblemContractError(
            f"Problem.evaluate() must return a mapping, got {type(values).__name__}"
        )

    expected = {spec.name for spec in objectives}
    actual = set(values.keys())
    missing = sorted(expected - actual)
    extra = sorted(str(key) for key in actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("undeclared: " + ", ".join(extra))
        raise ProblemContractError("Invalid objective mapping (" + "; ".join(details) + ")")

    normalized: Dict[str, float] = {}
    for spec in objectives:
        value = values[spec.name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ProblemContractError(
                f"Objective {spec.name!r} must be a real number, got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ProblemContractError(
                f"Objective {spec.name!r} must be finite, got {number!r}"
            )
        normalized[spec.name] = number
    return normalized


@dataclass(frozen=True)
class ValidationOutcome:
    """Structured result of a feasibility pre-check.

    ``failure_phase`` lets a problem label *where* a candidate broke (e.g.
    ``"structural" | "constraint" | "simulation"``) so the loop can feed richer
    failure context to the LLM. ``message`` is forwarded verbatim to the model.
    """

    ok: bool
    failure_phase: Optional[str] = None
    message: Optional[str] = None


@runtime_checkable
class Problem(Protocol[ConfigT]):
    """Minimal interface that every optimisation problem must satisfy.

    Required
    --------
    objectives : Sequence[ObjectiveSpec]
        The objectives to optimise (at least one).
    evaluate(config) -> Dict[str, float]
        Return exactly one finite numeric value for every declared objective.
        Raise ``ValueError`` with a descriptive message only for invalid /
        infeasible configurations -- the message is forwarded to the LLM as
        feedback. Infrastructure and programming errors must use other exception
        types and abort the run rather than becoming candidate feedback.

    Optional (detected via ``hasattr`` at runtime)
    ----------------------------------------------
    validate_detailed(config) -> ValidationOutcome
        Structured feasibility pre-check carrying a ``failure_phase`` label.
    validate(config) -> bool
        Legacy boolean pre-check. **Raise** ``ValueError("...")`` when invalid
        (never return ``False`` silently). Wrapped into a ``ValidationOutcome``.
    search_space_description() -> str
        Human-readable description of the configuration format, valid ranges,
        and constraints. Included verbatim in LLM prompts.
    render_candidate(config) -> str
        Compact one-line summary of a configuration, used in failure / Pareto
        lists shown to the LLM. Defaults to pretty JSON when absent.
    candidate_key(config) -> str
        Canonical key used to de-duplicate proposed candidates across the run
        (so identical configs are not re-evaluated). Defaults to sorted JSON.

    Optional attribute:

    directives
        A ``Directives`` provider supplying prompt wording for this problem.
        When absent, the backbone's generic ``DefaultDirectives`` is used.

    Optional attributes (not part of the protocol check):

    candidate_model : type[pydantic.BaseModel]
        Schema for one candidate; its JSON schema is shown to the LLM.
    constraints_description : str
        Extra free-text constraints injected into prompts.
    example_config : dict
        A reference configuration the model can imitate.
    config_schema : dict
        A pseudo-JSON schema for the configuration.
    """

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]: ...

    def evaluate(self, config: ConfigT) -> Dict[str, float]: ...
