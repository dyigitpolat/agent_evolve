"""Black-box candidate evaluation against a Problem.

Honors (in order) ``validate_detailed`` -> ``validate`` -> ``evaluate``. Expected
infeasibility surfaces as a failed :class:`CandidateResult` carrying a
``failure_phase`` and message; the message is forwarded to the LLM as feedback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from agent_evolve.core.formatting import CandidateResult
from agent_evolve.core.problem import (
    ObjectiveSpec,
    ValidationOutcome,
    normalize_objective_values,
)

INVALID_PENALTY: float = 1e18

_VALIDATION_FALSE_HINT = (
    "validate() returned False without a reason. Raise ValueError('...') from "
    "validate() describing what is wrong and how to fix it (unknown keys, wrong "
    "types, out-of-range values, invalid combinations)."
)


def format_optimizer_error(exc: BaseException) -> str:
    """Format an exception for regeneration prompts with a concrete fix hint."""
    name = type(exc).__name__
    msg = str(exc).strip()
    if msg:
        return f"{name}: {msg}"
    return (
        f"{name} was raised with no message. Raise ValueError('clear explanation of "
        "what failed and valid alternatives') instead."
    )


def _penalty_objectives(objectives: Sequence[ObjectiveSpec]) -> Dict[str, float]:
    return {s.name: (0.0 if s.goal == "max" else INVALID_PENALTY) for s in objectives}


def _failure(
    config: Dict[str, Any],
    objectives: Sequence[ObjectiveSpec],
    message: str,
    phase: str,
    *,
    evaluation_attempted: bool = False,
) -> CandidateResult:
    return CandidateResult(
        configuration=config,
        objectives=_penalty_objectives(objectives),
        is_valid=False,
        error_message=message,
        failure_phase=phase,
        evaluation_attempted=evaluation_attempted,
    )


def _validate(problem: Any, config: Dict[str, Any]) -> ValidationOutcome:
    """Normalise the optional validation hooks into a ValidationOutcome."""
    if hasattr(problem, "validate_detailed"):
        outcome = problem.validate_detailed(config)
        if not isinstance(outcome, ValidationOutcome):
            raise TypeError("validate_detailed must return a ValidationOutcome")
        return outcome
    if hasattr(problem, "validate"):
        ok = problem.validate(config)
        if not ok:
            return ValidationOutcome(False, "validation", _VALIDATION_FALSE_HINT)
    return ValidationOutcome(True)


def evaluate_batch(
    problem: Any,
    candidates: List[Dict[str, Any]],
    objectives: Sequence[ObjectiveSpec],
) -> Tuple[List[CandidateResult], List[CandidateResult], List[CandidateResult]]:
    """Evaluate *candidates* against *problem*.

    Returns ``(valid, failed, ordered)`` where *ordered* has one
    :class:`CandidateResult` per input candidate in input order.
    """
    valid: List[CandidateResult] = []
    failed: List[CandidateResult] = []
    ordered: List[CandidateResult] = []

    for config in candidates:
        try:
            outcome = _validate(problem, config)
        except ValueError as exc:
            outcome = ValidationOutcome(False, "validation", format_optimizer_error(exc))
        if not outcome.ok:
            cr = _failure(
                config,
                objectives,
                outcome.message or "invalid configuration",
                outcome.failure_phase or "validation",
            )
            failed.append(cr)
            ordered.append(cr)
            continue

        try:
            obj = problem.evaluate(config)
        except ValueError as exc:
            cr = _failure(
                config,
                objectives,
                format_optimizer_error(exc),
                "evaluation",
                evaluation_attempted=True,
            )
            failed.append(cr)
            ordered.append(cr)
            continue

        normalized = normalize_objective_values(obj, objectives)
        cr = CandidateResult(
            configuration=config,
            objectives=normalized,
            is_valid=True,
            evaluation_attempted=True,
        )
        valid.append(cr)
        ordered.append(cr)

    return valid, failed, ordered
