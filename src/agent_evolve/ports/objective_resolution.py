"""Benchmark-neutral objective resolution at the evaluation boundary.

Evaluators report physical measurements.  Optimizers need stable decision
values: repeated measurements that differ only below a benchmark-declared
resolution must not become artificial Pareto improvements.  This port keeps
those two facts separate.  A resolver receives the frozen configuration and
the complete raw objective vector, then returns the values used for selection.

The application calls a resolver repeatedly to fail closed on mutable,
non-deterministic, or non-idempotent policies.  Raw measurements remain in an
immutable receipt even when the decision values are canonicalized.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.core.problem import ObjectiveSpec, validate_objective_specs
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


_POLICY_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_REQUEST_DOMAIN = b"agent-evolve:objective-resolution-request:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:objective-resolution-receipt:v1\x00"

EXACT_OBJECTIVE_RESOLUTION_POLICY_ID = "exact_objective_values"
EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION = 1
EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:exact-objective-resolution:v1\x00"
    b"decision_objectives=raw_objectives"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _validate_objective_vector(
    values: tuple[tuple[str, float], ...],
    *,
    name: str,
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty exact tuple")
    metric_ids: list[str] = []
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name} entries must be exact (metric_id, value) tuples")
        metric_id, value = item
        if type(metric_id) is not str or not metric_id.strip():
            raise ValueError(f"{name} metric IDs must be non-empty exact strings")
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(f"{name} values must be finite canonical floats")
        metric_ids.append(metric_id)
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError(f"{name} metric IDs must be unique")
    return tuple(metric_ids)


def _objective_vector_record(
    values: tuple[tuple[str, float], ...],
) -> list[dict[str, str]]:
    return [
        {"metric_id": metric_id, "value_hex": value.hex()}
        for metric_id, value in values
    ]


@dataclass(frozen=True, slots=True)
class ObjectiveResolutionRequest:
    """One complete raw evaluation projected into decision-value semantics."""

    configuration: FrozenJsonObject
    objectives: tuple[ObjectiveSpec, ...]
    raw_objectives: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        if freeze_json(self.configuration) is not self.configuration:
            raise TypeError("configuration must already be frozen typed JSON")
        if type(self.objectives) is not tuple:
            raise TypeError("objectives must be an exact tuple")
        if any(type(value) is not ObjectiveSpec for value in self.objectives):
            raise TypeError("objectives must contain exact ObjectiveSpec values")
        validate_objective_specs(self.objectives)
        metric_ids = _validate_objective_vector(
            self.raw_objectives,
            name="raw_objectives",
        )
        expected = tuple(value.name for value in self.objectives)
        if metric_ids != expected:
            raise ValueError(
                "raw_objectives must be complete and follow objective declaration order"
            )

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return typed_json_sha256(self.configuration)

    def payload_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "configuration_sha256": self.configuration_sha256,
            "objectives": [
                {"metric_id": objective.name, "goal": objective.goal}
                for objective in self.objectives
            ],
            "raw_objectives": _objective_vector_record(self.raw_objectives),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self.payload_record())

    def to_record(self) -> dict[str, object]:
        return {**self.payload_record(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class ObjectiveResolutionResult:
    """Untrusted policy output before application-owned receipt sealing."""

    request_sha256: str
    decision_objectives: tuple[tuple[str, float], ...]
    evidence: FrozenJsonObject = field(default_factory=lambda: FrozenJsonObject(()))

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _validate_objective_vector(
            self.decision_objectives,
            name="decision_objectives",
        )
        if type(self.evidence) is not FrozenJsonObject:
            raise TypeError("evidence must be an exact FrozenJsonObject")
        if freeze_json(self.evidence) is not self.evidence:
            raise TypeError("evidence must already be frozen typed JSON")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "decision_objectives": _objective_vector_record(
                self.decision_objectives
            ),
            "evidence_sha256": typed_json_sha256(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class ObjectiveResolutionReceipt:
    """Application-sealed link between raw evidence and decision values."""

    request_sha256: str
    configuration_sha256: str
    raw_objectives: tuple[tuple[str, float], ...]
    decision_objectives: tuple[tuple[str, float], ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    evidence: FrozenJsonObject = field(default_factory=lambda: FrozenJsonObject(()))

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        raw_ids = _validate_objective_vector(
            self.raw_objectives,
            name="raw_objectives",
        )
        decision_ids = _validate_objective_vector(
            self.decision_objectives,
            name="decision_objectives",
        )
        if raw_ids != decision_ids:
            raise ValueError(
                "raw and decision objective vectors must have identical metric order"
            )
        _validate_policy_identity(
            self.policy_id,
            self.policy_version,
            self.policy_definition_sha256,
        )
        if type(self.evidence) is not FrozenJsonObject:
            raise TypeError("evidence must be an exact FrozenJsonObject")
        if freeze_json(self.evidence) is not self.evidence:
            raise TypeError("evidence must already be frozen typed JSON")

    @property
    def policy_identity(self) -> tuple[str, int, str]:
        self.__post_init__()
        return (
            self.policy_id,
            self.policy_version,
            self.policy_definition_sha256,
        )

    def payload_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "configuration_sha256": self.configuration_sha256,
            "raw_objectives": _objective_vector_record(self.raw_objectives),
            "decision_objectives": _objective_vector_record(
                self.decision_objectives
            ),
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RECEIPT_DOMAIN, self.payload_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self.payload_record(),
            "evidence": thaw_json(self.evidence),
            "receipt_sha256": self.receipt_sha256,
        }

    def revalidate(self) -> None:
        self.__post_init__()


@runtime_checkable
class ObjectiveResolutionPort(Protocol):
    """Resolve stable decision values from raw benchmark measurements."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> int: ...

    @property
    def definition_sha256(self) -> str: ...

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult: ...


def _validate_policy_identity(
    policy_id: object,
    policy_version: object,
    definition_sha256: object,
) -> tuple[str, int, str]:
    if type(policy_id) is not str or _POLICY_ID.fullmatch(policy_id) is None:
        raise ValueError("objective-resolution policy_id is not canonical")
    if type(policy_version) is not int or policy_version <= 0:
        raise ValueError("objective-resolution policy_version must be positive")
    require_sha256(definition_sha256, "objective-resolution definition_sha256")
    return policy_id, policy_version, definition_sha256


def objective_resolution_policy_metadata(
    policy: ObjectiveResolutionPort,
) -> tuple[str, int, str]:
    """Validate and return the immutable scientific identity of a resolver."""

    if not isinstance(policy, ObjectiveResolutionPort):
        raise TypeError(
            "objective_resolution must implement ObjectiveResolutionPort"
        )
    return _validate_policy_identity(
        policy.policy_id,
        policy.policy_version,
        policy.definition_sha256,
    )


def _validated_result(
    result: object,
    request: ObjectiveResolutionRequest,
) -> ObjectiveResolutionResult:
    if type(result) is not ObjectiveResolutionResult:
        raise TypeError("objective-resolution policy must return an exact result")
    ObjectiveResolutionResult.__post_init__(result)
    if result.request_sha256 != request.request_sha256:
        raise ValueError("objective-resolution result is bound to another request")
    expected = tuple(objective.name for objective in request.objectives)
    actual = tuple(metric_id for metric_id, _ in result.decision_objectives)
    if actual != expected:
        raise ValueError(
            "decision_objectives must be complete and follow declaration order"
        )
    return result


def _resolve_twice(
    policy: ObjectiveResolutionPort,
    request: ObjectiveResolutionRequest,
    metadata: tuple[str, int, str],
) -> ObjectiveResolutionResult:
    first = _validated_result(policy.resolve(request), request)
    second = _validated_result(policy.resolve(request), request)
    if first.to_record() != second.to_record():
        raise ValueError("objective-resolution policy must be deterministic")
    if objective_resolution_policy_metadata(policy) != metadata:
        raise ValueError(
            "objective-resolution policy metadata changed during resolution"
        )
    return first


def resolve_objectives(
    policy: ObjectiveResolutionPort,
    request: ObjectiveResolutionRequest,
) -> ObjectiveResolutionReceipt:
    """Resolve, verify, and seal decision values for one raw evaluation.

    Idempotence is checked by resolving the first decision vector again.  This
    rejects drift-prone transforms (for example ``value + epsilon``) before
    they can corrupt archive ordering or hypervolume accounting.
    """

    if type(request) is not ObjectiveResolutionRequest:
        raise TypeError("request must be an exact ObjectiveResolutionRequest")
    ObjectiveResolutionRequest.__post_init__(request)
    metadata = objective_resolution_policy_metadata(policy)
    first = _resolve_twice(policy, request, metadata)
    idempotence_request = ObjectiveResolutionRequest(
        configuration=request.configuration,
        objectives=request.objectives,
        raw_objectives=first.decision_objectives,
    )
    idempotent = _resolve_twice(policy, idempotence_request, metadata)
    if _objective_vector_record(
        idempotent.decision_objectives
    ) != _objective_vector_record(first.decision_objectives):
        raise ValueError("objective-resolution policy must be idempotent")
    policy_id, policy_version, definition_sha256 = metadata
    return ObjectiveResolutionReceipt(
        request_sha256=request.request_sha256,
        configuration_sha256=request.configuration_sha256,
        raw_objectives=request.raw_objectives,
        decision_objectives=first.decision_objectives,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=definition_sha256,
        evidence=first.evidence,
    )


@dataclass(frozen=True, slots=True)
class ExactObjectiveResolution:
    """Identity resolver for explicit experiments and compatibility checks."""

    policy_id: str = EXACT_OBJECTIVE_RESOLUTION_POLICY_ID
    policy_version: int = EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION
    definition_sha256: str = EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256

    def __post_init__(self) -> None:
        identity = _validate_policy_identity(
            self.policy_id,
            self.policy_version,
            self.definition_sha256,
        )
        if identity != (
            EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
            EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
            EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
        ):
            raise ValueError("ExactObjectiveResolution identity is closed")

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult:
        if type(request) is not ObjectiveResolutionRequest:
            raise TypeError("request must be an exact ObjectiveResolutionRequest")
        ObjectiveResolutionRequest.__post_init__(request)
        return ObjectiveResolutionResult(
            request_sha256=request.request_sha256,
            decision_objectives=request.raw_objectives,
        )


__all__ = [
    "EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256",
    "EXACT_OBJECTIVE_RESOLUTION_POLICY_ID",
    "EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION",
    "ExactObjectiveResolution",
    "ObjectiveResolutionPort",
    "ObjectiveResolutionReceipt",
    "ObjectiveResolutionRequest",
    "ObjectiveResolutionResult",
    "objective_resolution_policy_metadata",
    "resolve_objectives",
]
