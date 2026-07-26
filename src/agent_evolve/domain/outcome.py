"""Closed, mutually distinguishable failure taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional

from agent_evolve.domain.ids import ArtifactId


class FailureCategory(str, Enum):
    CANDIDATE = "candidate"
    INFRASTRUCTURE = "infrastructure"
    SYSTEM = "system"


class FailureCode(str, Enum):
    # Candidate-attributable failures.
    SCHEMA_INVALID = "schema_invalid"
    DETERMINISTIC_PRECHECK_INFEASIBLE = "deterministic_precheck_infeasible"
    EVALUATOR_DECLARED_INFEASIBLE = "evaluator_declared_infeasible"
    NUMERICAL_NONCONVERGENCE = "numerical_nonconvergence_attributable_to_candidate"

    # Infrastructure failures.
    PROCESS_START_FAILURE = "process_start_failure"
    TIMEOUT_OR_RESOURCE_FAILURE = "timeout_or_resource_failure"
    CONTAINER_OR_DEPENDENCY_FAILURE = "container_or_dependency_failure"
    TRANSIENT_EXTERNAL_SERVICE_FAILURE = "transient_external_service_failure"

    # System failures.
    EVALUATOR_CONTRACT_VIOLATION = "evaluator_contract_violation"
    PARSER_FAILURE = "parser_failure"
    INTERNAL_BUG = "internal_bug"
    UNKNOWN_UNCLASSIFIED = "unknown_unclassified"


_CODES_BY_CATEGORY: Dict[FailureCategory, FrozenSet[FailureCode]] = {
    FailureCategory.CANDIDATE: frozenset(
        {
            FailureCode.SCHEMA_INVALID,
            FailureCode.DETERMINISTIC_PRECHECK_INFEASIBLE,
            FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
            FailureCode.NUMERICAL_NONCONVERGENCE,
        }
    ),
    FailureCategory.INFRASTRUCTURE: frozenset(
        {
            FailureCode.PROCESS_START_FAILURE,
            FailureCode.TIMEOUT_OR_RESOURCE_FAILURE,
            FailureCode.CONTAINER_OR_DEPENDENCY_FAILURE,
            FailureCode.TRANSIENT_EXTERNAL_SERVICE_FAILURE,
        }
    ),
    FailureCategory.SYSTEM: frozenset(
        {
            FailureCode.EVALUATOR_CONTRACT_VIOLATION,
            FailureCode.PARSER_FAILURE,
            FailureCode.INTERNAL_BUG,
            FailureCode.UNKNOWN_UNCLASSIFIED,
        }
    ),
}


def validate_failure_pair(category: FailureCategory, code: FailureCode) -> None:
    if not isinstance(category, FailureCategory):
        raise TypeError("category must be a FailureCategory")
    if not isinstance(code, FailureCode):
        raise TypeError("code must be a FailureCode")
    if code not in _CODES_BY_CATEGORY[category]:
        raise ValueError(f"Failure code {code.value!r} does not belong to {category.value!r}")


@dataclass(frozen=True, slots=True)
class FailureRecord:
    category: FailureCategory
    code: FailureCode
    message: str
    retryable: bool = False
    exception_type: Optional[str] = None
    diagnostics_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        validate_failure_pair(self.category, self.code)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Failure message must be a non-empty string")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be bool")
        if self.exception_type is not None:
            if not isinstance(self.exception_type, str) or not self.exception_type.strip():
                raise ValueError("exception_type must be non-empty when provided")
        if self.diagnostics_artifact_id is not None and not isinstance(
            self.diagnostics_artifact_id, ArtifactId
        ):
            raise TypeError("diagnostics_artifact_id must be an ArtifactId or None")
