"""The five typed obligations a problem owes the optimizer, and nothing else.

Everything the optimizer needs from you is here. A problem declares what a
candidate looks like, where to start, which candidates are worth spending on,
how a candidate becomes the thing you measure, and what measuring it returns::

    candidate_model   the schema a proposal must satisfy
    objectives        what is being optimized, and in which direction
    seeds()           the starting points
    validate()        cheap, side-effect free, and it explains itself
    materialize()     candidate -> the artifact that gets measured
    evaluate()        artifact -> objective values

Two of these had no public representation before, and both were load-bearing.

``seeds()`` existed only inside the campaign machinery, so a library user had
no supported way to say "start from what I already have" -- the single most
common thing someone with a real problem wants.

``materialize()`` is not symmetry. The evaluation cache keyed on the
*configuration*, so two configurations that produce the same artifact were paid
for twice. That is the redundancy the union-over-sum analysis measured as the
dominant loss: we buy 2.14 times uniform's individual gain and convert 30
percent of it, against uniform's 68 percent. Separating the step that builds
the artifact from the step that measures it lets the cache key on what is
actually expensive, and it gives constraint repair somewhere honest to live --
cheap and fallible in ``materialize``, expensive and trusted in ``evaluate``.

Legacy problems that predate this contract keep working: :func:`as_problem`
adapts anything exposing ``objectives`` and ``evaluate`` by treating
materialization as the identity and seeds as empty.
"""

from __future__ import annotations

import hashlib
import json
from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from agent_evolve.core.problem import (
    ObjectiveSpec,
    ValidationOutcome,
    normalize_objective_values,
    validate_objective_specs,
)

__all__ = [
    "Problem",
    "ObjectiveSpec",
    "ValidationOutcome",
    "as_problem",
    "as_evaluator",
    "artifact_key",
]

Config = Mapping[str, Any]


@runtime_checkable
class Problem(Protocol):
    """The whole integration surface. Five obligations, all required."""

    #: Pydantic model a proposed candidate must satisfy. The proposer is
    #: handed this schema; nothing that fails it reaches ``validate``.
    candidate_model: Any

    #: What is being optimized. At least one, each ``min`` or ``max``.
    objectives: Sequence[ObjectiveSpec]

    def seeds(self) -> Sequence[Config]:
        """Starting points. Return ``()`` to start from proposals alone.

        Seeds are evaluated before any proposal is made, so a run that starts
        from a known-good configuration reports honestly whether anything it
        proposed beat what you already had.
        """
        ...

    def validate(self, config: Config) -> ValidationOutcome:
        """Cheap feasibility check. No side effects, and it explains itself.

        The message is fed back to the proposer verbatim, so it should say what
        is wrong *and* what would be acceptable.
        """
        ...

    def materialize(self, config: Config) -> Any:
        """Build the artifact that will be measured.

        Anything expensive and deterministic belongs here rather than in
        ``evaluate``: two configurations that materialize to equal artifacts
        are evaluated once, not twice.
        """
        ...

    def evaluate(self, artifact: Any) -> Mapping[str, float]:
        """Measure the artifact. Return exactly the declared objectives.

        Raise ``ValueError`` with a usable message for a candidate that is
        infeasible in a way only measurement could reveal.
        """
        ...


def artifact_key(artifact: Any) -> str:
    """Stable identity for a materialized artifact, used to avoid paying twice.

    JSON-shaped artifacts hash by canonical content, so key order and float
    spelling do not create spurious misses. Anything else defers to the
    artifact's own ``__hash__`` when it has one, and finally to its ``repr``.
    A problem that needs different identity should say so by returning an
    artifact whose equality already means what it wants.
    """
    try:
        payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        try:
            payload = f"h:{hash(artifact)}"
        except TypeError:
            payload = f"r:{artifact!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _LegacyProblem:
    """Presents a pre-contract problem as one satisfying all five obligations.

    Materialization is the identity and seeds are empty, which is exactly what
    the old contract meant. Validation keeps the old precedence
    (``validate_detailed`` before ``validate``) so existing messages are
    unchanged.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def candidate_model(self) -> Any:
        return getattr(self._inner, "candidate_model", None)

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return list(self._inner.objectives)

    def seeds(self) -> Sequence[Config]:
        seeds = getattr(self._inner, "seeds", None)
        if seeds is not None:
            declared = tuple(seeds() if callable(seeds) else seeds)
            if declared:
                return declared
        # A problem that declares an `example_config` has already named a
        # configuration it considers representative, which is exactly what a
        # seed is. Using it means the run reports whether anything it proposed
        # beat what the caller already had, and it lets the genetic loop run at
        # all -- without it such a problem silently falls back to authoring,
        # which measures worse than random search on every genome length tried.
        example = getattr(self._inner, "example_config", None)
        if isinstance(example, Mapping) and example:
            return (dict(example),)
        return ()

    def validate(self, config: Config) -> ValidationOutcome:
        inner = self._inner
        if hasattr(inner, "validate_detailed"):
            outcome = inner.validate_detailed(config)
            if not isinstance(outcome, ValidationOutcome):
                raise TypeError("validate_detailed must return a ValidationOutcome")
            return outcome
        if hasattr(inner, "validate"):
            if not inner.validate(config):
                return ValidationOutcome(
                    False,
                    "validation",
                    "validate() returned False without a reason. Raise "
                    "ValueError('...') describing what is wrong and what would "
                    "be acceptable.",
                )
        return ValidationOutcome(True)

    def materialize(self, config: Config) -> Any:
        inner = getattr(self._inner, "materialize", None)
        if inner is not None:
            return inner(config)
        return config

    def evaluate(self, artifact: Any) -> Mapping[str, float]:
        return self._inner.evaluate(artifact)

    def __getattr__(self, name: str) -> Any:
        # Optional hooks the loop still reads (directives, descriptions, ...)
        return getattr(self._inner, name)


def as_evaluator(problem: Any) -> Problem:
    """Adapt *problem* for the per-candidate obligations only.

    Used where the objectives are supplied separately, so a problem that does
    not declare them is not rejected for it -- the caller already knows them.
    """
    if not hasattr(problem, "evaluate"):
        raise TypeError(
            f"{type(problem).__name__} has no 'evaluate'. A problem must be able "
            "to measure a materialized artifact."
        )
    complete = all(
        callable(getattr(problem, name, None))
        for name in ("seeds", "validate", "materialize")
    )
    return problem if complete else _LegacyProblem(problem)


def as_problem(problem: Any) -> Problem:
    """Return *problem* as a five-obligation problem, adapting it if needed.

    Raises immediately, rather than at first evaluation, when the two
    irreducible members are missing.
    """
    if not hasattr(problem, "objectives"):
        raise TypeError(
            f"{type(problem).__name__} declares no 'objectives'. A problem must "
            "declare at least one ObjectiveSpec."
        )
    adapted = as_evaluator(problem)
    validate_objective_specs(list(problem.objectives))
    return adapted


def evaluate_objectives(
    problem: Problem,
    artifact: Any,
    objectives: Sequence[ObjectiveSpec],
) -> Mapping[str, float]:
    """Evaluate and check the result against the declared objectives."""
    return normalize_objective_values(problem.evaluate(artifact), objectives)
