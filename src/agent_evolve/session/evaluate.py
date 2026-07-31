"""Black-box candidate evaluation against a Problem.

Runs the three per-candidate obligations in order: ``validate`` rejects cheaply
and explains itself, ``materialize`` builds the artifact, ``evaluate`` measures
it. Expected infeasibility surfaces as a failed :class:`CandidateResult`
carrying a ``failure_phase`` and message; the message is forwarded to the LLM
as feedback.

Evaluation is cached on the identity of the *materialized artifact*, not the
configuration. Distinct configurations that build the same artifact are
therefore measured once. Keying on the configuration -- which is what this did
before -- paid twice for the redundancy the union-over-sum analysis measured as
the dominant loss.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from agent_evolve.contract import artifact_key, as_evaluator
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


class EvaluationCache(dict):
    """Artifact identity -> objective vector, plus what it saved.

    ``hits`` is the number of evaluations not paid for. It is reported rather
    than inferred, so a problem whose ``materialize`` collapses nothing can see
    that immediately instead of assuming a saving it never got.

    ``misses`` is what the budget actually counts: distinct artifacts measured.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hits = 0
        self.misses = 0

    #: Hard ceiling on ``misses``. Once reached, no further artifact is
    #: measured; a budget that is only checked between generations is not a
    #: budget, and a caller who said 40 must not be billed for 48.
    budget: Optional[int] = None

    def exhausted(self) -> bool:
        return self.budget is not None and self.misses >= self.budget


def evaluate_batch(
    problem: Any,
    candidates: List[Dict[str, Any]],
    objectives: Sequence[ObjectiveSpec],
    cache: Optional[MutableMapping[str, Dict[str, float]]] = None,
) -> Tuple[List[CandidateResult], List[CandidateResult], List[CandidateResult]]:
    """Evaluate *candidates* against *problem*.

    Returns ``(valid, failed, ordered)`` where *ordered* has one
    :class:`CandidateResult` per input candidate in input order. When *cache*
    is supplied, artifacts already measured are not measured again.
    """
    bound = as_evaluator(problem)
    valid: List[CandidateResult] = []
    failed: List[CandidateResult] = []
    ordered: List[CandidateResult] = []

    def _record(cr: CandidateResult, bucket: List[CandidateResult]) -> None:
        bucket.append(cr)
        ordered.append(cr)

    for config in candidates:
        try:
            outcome = bound.validate(config)
        except ValueError as exc:
            outcome = ValidationOutcome(False, "validation", format_optimizer_error(exc))
        if not outcome.ok:
            _record(
                _failure(
                    config,
                    objectives,
                    outcome.message or "invalid configuration",
                    outcome.failure_phase or "validation",
                ),
                failed,
            )
            continue

        try:
            artifact = bound.materialize(config)
        except ValueError as exc:
            _record(
                _failure(config, objectives, format_optimizer_error(exc), "materialization"),
                failed,
            )
            continue

        key = artifact_key(artifact) if cache is not None else None
        if key is not None and key in cache:
            normalized = dict(cache[key])
            if isinstance(cache, EvaluationCache):
                cache.hits += 1
        elif isinstance(cache, EvaluationCache) and cache.exhausted():
            # The budget is spent. Report it as a refusal rather than quietly
            # measuring one more, so the number the caller gave is the number
            # they are billed for.
            _record(
                _failure(
                    config,
                    objectives,
                    f"evaluation budget of {cache.budget} exhausted",
                    "budget",
                ),
                failed,
            )
            continue
        else:
            try:
                obj = bound.evaluate(artifact)
            except ValueError as exc:
                _record(
                    _failure(
                        config,
                        objectives,
                        format_optimizer_error(exc),
                        "evaluation",
                        evaluation_attempted=True,
                    ),
                    failed,
                )
                continue
            normalized = normalize_objective_values(obj, objectives)
            if key is not None:
                cache[key] = dict(normalized)
                if isinstance(cache, EvaluationCache):
                    cache.misses += 1

        _record(
            CandidateResult(
                configuration=config,
                objectives=normalized,
                is_valid=True,
                evaluation_attempted=True,
            ),
            valid,
        )

    return valid, failed, ordered
