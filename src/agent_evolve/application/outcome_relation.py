"""Injected, stateless relations over sealed detailed evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from agent_evolve.application.detailed_evaluation import DetailedEvaluation
from agent_evolve.core.problem import (
    ObjectiveSpec,
    normalize_objective_values,
    validate_objective_specs,
)
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PARETO_DEFINITION_DOMAIN = b"agent-evolve:objective-pareto-relation:v1\x00"


class OutcomeRelation(str, Enum):
    BETTER = "better"
    WORSE = "worse"
    EQUIVALENT = "equivalent"
    INCOMPARABLE = "incomparable"


OutcomeComparator = Callable[
    [DetailedEvaluation, DetailedEvaluation],
    OutcomeRelation,
]


@dataclass(frozen=True, slots=True)
class OutcomeRelationPolicyBinding:
    """One comparison callable and its immutable published semantics.

    Archive use requires a deterministic partial order: ``BETTER``/``WORSE``
    are inverse strict relations, ``EQUIVALENT`` is a true transitive
    equivalence relation, and ``INCOMPARABLE`` is symmetric.  Resolution bands
    that are not proven transitive should therefore return ``INCOMPARABLE``,
    not ``EQUIVALENT``.  Each call verifies determinism and inverse symmetry;
    global transitivity remains a declared policy obligation.
    """

    compare: OutcomeComparator
    policy_id: str
    policy_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.compare):
            raise TypeError("compare must be callable")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.policy_id, self.policy_version, self.definition_sha256

    def relate(
        self,
        left: DetailedEvaluation,
        right: DetailedEvaluation,
    ) -> OutcomeRelation:
        if type(left) is not DetailedEvaluation or type(right) is not DetailedEvaluation:
            raise TypeError("outcome comparison requires exact DetailedEvaluation values")
        DetailedEvaluation.__post_init__(left)
        DetailedEvaluation.__post_init__(right)
        relation = self.compare(left, right)
        repeated = self.compare(left, right)
        inverse = self.compare(right, left)
        inverse_repeated = self.compare(right, left)
        if any(
            type(value) is not OutcomeRelation
            for value in (relation, repeated, inverse, inverse_repeated)
        ):
            raise TypeError("outcome relation policy returned an invalid relation")
        if relation is not repeated or inverse is not inverse_repeated:
            raise ValueError("outcome relation policy must be deterministic")
        expected_inverse = {
            OutcomeRelation.BETTER: OutcomeRelation.WORSE,
            OutcomeRelation.WORSE: OutcomeRelation.BETTER,
            OutcomeRelation.EQUIVALENT: OutcomeRelation.EQUIVALENT,
            OutcomeRelation.INCOMPARABLE: OutcomeRelation.INCOMPARABLE,
        }[relation]
        if inverse is not expected_inverse:
            raise ValueError("outcome relation policy violated inverse consistency")
        return relation

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveParetoOutcomePolicy:
    """The legacy objective-only partial order, expressed over detailed records."""

    objectives: tuple[ObjectiveSpec, ...]

    def __post_init__(self) -> None:
        if type(self.objectives) is not tuple:
            raise TypeError("objectives must be an exact tuple")
        validate_objective_specs(self.objectives)

    @property
    def definition_sha256(self) -> str:
        record = [
            {"name": objective.name, "goal": objective.goal}
            for objective in self.objectives
        ]
        encoded = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_PARETO_DEFINITION_DOMAIN + encoded).hexdigest()

    def compare(
        self,
        left: DetailedEvaluation,
        right: DetailedEvaluation,
    ) -> OutcomeRelation:
        if not left.success or not right.success:
            raise ValueError("objective Pareto comparison requires successful records")
        left_values = normalize_objective_values(
            dict(left.objectives),
            self.objectives,
        )
        right_values = normalize_objective_values(
            dict(right.objectives),
            self.objectives,
        )
        left_weak = True
        left_strict = False
        right_weak = True
        right_strict = False
        for objective in self.objectives:
            left_value = left_values[objective.name]
            right_value = right_values[objective.name]
            if objective.goal == "max":
                left_weak &= left_value >= right_value
                left_strict |= left_value > right_value
                right_weak &= right_value >= left_value
                right_strict |= right_value > left_value
            else:
                left_weak &= left_value <= right_value
                left_strict |= left_value < right_value
                right_weak &= right_value <= left_value
                right_strict |= right_value < left_value
        if left_weak and left_strict:
            return OutcomeRelation.BETTER
        if right_weak and right_strict:
            return OutcomeRelation.WORSE
        if left_weak and right_weak:
            return OutcomeRelation.EQUIVALENT
        return OutcomeRelation.INCOMPARABLE

    def binding(self) -> OutcomeRelationPolicyBinding:
        return OutcomeRelationPolicyBinding(
            compare=self.compare,
            policy_id="objective_pareto",
            policy_version=1,
            definition_sha256=self.definition_sha256,
        )


def objective_pareto_outcome_binding(
    objectives: Sequence[ObjectiveSpec],
) -> OutcomeRelationPolicyBinding:
    return ObjectiveParetoOutcomePolicy(tuple(objectives)).binding()


__all__ = [
    "ObjectiveParetoOutcomePolicy",
    "OutcomeComparator",
    "OutcomeRelation",
    "OutcomeRelationPolicyBinding",
    "objective_pareto_outcome_binding",
]
