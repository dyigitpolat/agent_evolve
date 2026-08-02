"""Immutable evaluator evidence sealed by the agentic engine.

An adapter owns domain semantics and returns :class:`DetailedEvaluationPayload`.
It never sees the engine's injected phenotype policy.  The engine measures the
around-port wall time, binds its phenotype identity, and caches that complete
physical-evaluation record.  Candidate occurrence identity deliberately lives
outside this module because multiple configurations may share one phenotype.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import Protocol, runtime_checkable

from agent_evolve.core.problem import ObjectiveSpec, normalize_objective_values
from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.domain.outcome import FailureRecord
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
)
from agent_evolve.policies.selection.phenotype_recourse import PhenotypeIdentity


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"agent-evolve:detailed-evaluation:v1\x00"
_MAX_OBSERVED_VALUE_BYTES = 8_192


def _finite_nonnegative(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _named_finite_values(
    values: tuple[tuple[str, float], ...],
    *,
    name: str,
    nonnegative: bool,
    require_sorted: bool,
) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    names: list[str] = []
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name} must contain exact name/value pairs")
        key, value = item
        if type(key) is not str or _TOKEN.fullmatch(key) is None:
            raise ValueError(f"{name} names must use the closed token grammar")
        if type(value) is not float:
            raise TypeError(f"{name} values must be canonical floats")
        if not math.isfinite(value) or (nonnegative and value < 0):
            qualifier = "finite non-negative" if nonnegative else "finite"
            raise ValueError(f"{name} values must be {qualifier}")
        names.append(key)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} names must be unique")
    if require_sorted and names != sorted(names):
        raise ValueError(f"{name} names must be canonically ordered")


class EvaluationCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class EvaluationCheck:
    """One compact, receipt-addressable evaluator fact."""

    name: str
    status: EvaluationCheckStatus
    observed_value: FrozenJsonValue
    receipt_locator: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or _TOKEN.fullmatch(self.name) is None:
            raise ValueError("check name must use the closed token grammar")
        if type(self.status) is not EvaluationCheckStatus:
            raise TypeError("check status must be an EvaluationCheckStatus")
        frozen = freeze_json(self.observed_value)
        if frozen is not self.observed_value:
            raise TypeError("observed_value must already be frozen typed JSON")
        if len(canonical_typed_json_bytes(frozen)) > _MAX_OBSERVED_VALUE_BYTES:
            raise ValueError("observed_value exceeds the compact evidence limit")
        if (
            type(self.receipt_locator) is not str
            or not self.receipt_locator.strip()
            or self.receipt_locator != self.receipt_locator.strip()
            or len(self.receipt_locator.encode("utf-8")) > 512
        ):
            raise ValueError("receipt_locator must be compact canonical text")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "observed_value": thaw_json(self.observed_value),
            "receipt_locator": self.receipt_locator,
        }


@dataclass(frozen=True, slots=True)
class EvaluationTimings:
    """Non-additive timing observations for one physical evaluation."""

    total_wall_seconds: float
    active_wall_seconds: float | None = None
    resource_queue_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative(self.total_wall_seconds, name="total_wall_seconds")
        if type(self.total_wall_seconds) is not float:
            raise TypeError("total_wall_seconds must be a canonical float")
        for name in ("active_wall_seconds", "resource_queue_wall_seconds"):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name=name)
                if type(value) is not float:
                    raise TypeError(f"{name} must be a canonical float or None")

    def to_record(self) -> dict[str, float | None]:
        return {
            "total_wall_seconds": self.total_wall_seconds,
            "active_wall_seconds": self.active_wall_seconds,
            "resource_queue_wall_seconds": self.resource_queue_wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class EvaluatorIdentity:
    """Stable evaluator implementation/task context used in cache identity."""

    evaluator_id: str
    evaluator_version: int
    evaluator_context_sha256: str

    def __post_init__(self) -> None:
        if type(self.evaluator_id) is not str or _TOKEN.fullmatch(
            self.evaluator_id
        ) is None:
            raise ValueError("evaluator_id must use the closed token grammar")
        if type(self.evaluator_version) is not int or self.evaluator_version <= 0:
            raise ValueError("evaluator_version must be a positive exact integer")
        if (
            type(self.evaluator_context_sha256) is not str
            or _SHA256.fullmatch(self.evaluator_context_sha256) is None
        ):
            raise ValueError(
                "evaluator_context_sha256 must be a lowercase SHA-256 digest"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "evaluator_context_sha256": self.evaluator_context_sha256,
        }


def _artifact_record(value: ArtifactRef | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "artifact_id": value.artifact_id.value,
        "sha256_hex": value.sha256_hex,
        "size_bytes": value.size_bytes,
        "media_type": value.media_type,
    }


def _failure_record(value: FailureRecord | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "category": value.category.value,
        "code": value.code.value,
        "message": value.message,
        "retryable": value.retryable,
        "exception_type": value.exception_type,
        "diagnostics_artifact_id": (
            None
            if value.diagnostics_artifact_id is None
            else value.diagnostics_artifact_id.value
        ),
    }


@dataclass(frozen=True, slots=True)
class DetailedEvaluationPayload:
    """Adapter-authored evidence before engine identity and timing binding."""

    failure: FailureRecord | None
    objectives: tuple[tuple[str, float], ...]
    violations: tuple[tuple[str, float], ...]
    checks: tuple[EvaluationCheck, ...]
    receipt: ArtifactRef | None
    evaluator: EvaluatorIdentity
    active_wall_seconds: float | None = None
    resource_queue_wall_seconds: float | None = None
    observations: tuple[tuple[str, float], ...] = ()
    """Sub-problem measurements the evaluator already computed, carried but
    **never scored**.

    An evaluation unit that aggregates sub-problems -- a panel of layers, a set
    of instances, a multi-medoid network -- publishes the parts and then throws
    them away, because the only typed channel out of an adapter was the
    aggregate objective vector.  On Timeloop that discarded exactly the
    per-medoid split, measured to be worth 43.1 % of that domain's attainable
    range.

    `observations` is that channel.  It is deliberately parallel to
    `violations`: a named finite vector, adapter-authored, sealed with the
    evaluation.  It differs from `objectives` in the one way that matters --
    **nothing reads it to compute reward**.  The reward path consumes
    `objectives` alone, so widening the observation does not move the target,
    and every existing `objectives` equality invariant holds unchanged.

    Not required to be sorted, because sub-problem order is meaningful (medoid
    0, 1, 2 is not an alphabetical fact), and not required nonnegative, because
    a sub-problem measurement may legitimately be signed.
    """

    def __post_init__(self) -> None:
        if self.failure is not None:
            if type(self.failure) is not FailureRecord:
                raise TypeError("failure must be an exact FailureRecord or None")
            FailureRecord.__post_init__(self.failure)
        _named_finite_values(
            self.objectives,
            name="objectives",
            nonnegative=False,
            require_sorted=False,
        )
        _named_finite_values(
            self.violations,
            name="violations",
            nonnegative=True,
            require_sorted=True,
        )
        _named_finite_values(
            self.observations,
            name="observations",
            nonnegative=False,
            require_sorted=False,
        )
        if self.failure is None and not self.objectives:
            raise ValueError("successful detailed evaluations require objectives")
        if self.failure is not None and (
            self.objectives or self.violations or self.observations
        ):
            raise ValueError("failed detailed evaluations cannot carry projections")
        observation_names = tuple(name for name, _ in self.observations)
        if len(set(observation_names)) != len(observation_names):
            raise ValueError("observation names must be unique")
        objective_names = {name for name, _ in self.objectives}
        collisions = objective_names & set(observation_names)
        if collisions:
            raise ValueError(
                "observations must not shadow objective names: "
                f"{sorted(collisions)}"
            )
        if type(self.checks) is not tuple or any(
            type(check) is not EvaluationCheck for check in self.checks
        ):
            raise TypeError("checks must contain exact EvaluationCheck values")
        for check in self.checks:
            EvaluationCheck.__post_init__(check)
        check_names = tuple(check.name for check in self.checks)
        if check_names != tuple(sorted(set(check_names))):
            raise ValueError("checks must be unique and canonically ordered")
        if self.receipt is not None:
            if type(self.receipt) is not ArtifactRef:
                raise TypeError("receipt must be an exact ArtifactRef or None")
            ArtifactRef.__post_init__(self.receipt)
        if self.checks and self.receipt is None:
            raise ValueError("receipt-addressable checks require a receipt ArtifactRef")
        if type(self.evaluator) is not EvaluatorIdentity:
            raise TypeError("evaluator must be an exact EvaluatorIdentity")
        EvaluatorIdentity.__post_init__(self.evaluator)
        for name in ("active_wall_seconds", "resource_queue_wall_seconds"):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name=name)
                if type(value) is not float:
                    raise TypeError(f"{name} must be a canonical float or None")


@runtime_checkable
class DetailedEvaluationAdapter(Protocol):
    """Domain port returning evidence without engine-owned phenotype identity."""

    evaluator_identity: EvaluatorIdentity

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload: ...


def normalize_detailed_payload(
    payload: DetailedEvaluationPayload,
    objectives: tuple[ObjectiveSpec, ...],
) -> DetailedEvaluationPayload:
    """Validate the complete objective vector and freeze declaration order."""

    if type(payload) is not DetailedEvaluationPayload:
        raise TypeError("adapter must return an exact DetailedEvaluationPayload")
    DetailedEvaluationPayload.__post_init__(payload)
    if payload.failure is not None:
        return payload
    normalized = normalize_objective_values(dict(payload.objectives), objectives)
    ordered = tuple((spec.name, normalized[spec.name]) for spec in objectives)
    return replace(payload, objectives=ordered)


@dataclass(frozen=True, slots=True)
class DetailedEvaluation:
    """One engine-sealed, phenotype-cached physical evaluation record."""

    phenotype: PhenotypeIdentity
    payload: DetailedEvaluationPayload
    timings: EvaluationTimings

    def __post_init__(self) -> None:
        if type(self.phenotype) is not PhenotypeIdentity:
            raise TypeError("phenotype must be an exact PhenotypeIdentity")
        PhenotypeIdentity.__post_init__(self.phenotype)
        if type(self.payload) is not DetailedEvaluationPayload:
            raise TypeError("payload must be an exact DetailedEvaluationPayload")
        DetailedEvaluationPayload.__post_init__(self.payload)
        if type(self.timings) is not EvaluationTimings:
            raise TypeError("timings must be an exact EvaluationTimings")
        EvaluationTimings.__post_init__(self.timings)
        if self.timings.active_wall_seconds != self.payload.active_wall_seconds:
            raise ValueError("active timing must be copied unchanged from the adapter")
        if (
            self.timings.resource_queue_wall_seconds
            != self.payload.resource_queue_wall_seconds
        ):
            raise ValueError("queue timing must be copied unchanged from the adapter")

    @property
    def failure(self) -> FailureRecord | None:
        return self.payload.failure

    @property
    def objectives(self) -> tuple[tuple[str, float], ...]:
        return self.payload.objectives

    @property
    def violations(self) -> tuple[tuple[str, float], ...]:
        return self.payload.violations

    @property
    def checks(self) -> tuple[EvaluationCheck, ...]:
        return self.payload.checks

    @property
    def receipt(self) -> ArtifactRef | None:
        return self.payload.receipt

    @property
    def success(self) -> bool:
        return self.failure is None

    @property
    def observations(self) -> tuple[tuple[str, float], ...]:
        return self.payload.observations

    def _identity_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "phenotype": {
                **self.phenotype.to_trace_record(),
                "identity_sha256": self.phenotype.identity_sha256,
            },
            "failure": _failure_record(self.failure),
            "objectives": [[name, value.hex()] for name, value in self.objectives],
            "violations": [[name, value.hex()] for name, value in self.violations],
            "checks": [check.to_record() for check in self.checks],
            "receipt": _artifact_record(self.payload.receipt),
            "evaluator": self.payload.evaluator.to_record(),
            "timings": {
                name: None if value is None else value.hex()
                for name, value in self.timings.to_record().items()
            },
        }
        # `observations` enters the seal only when an adapter actually supplies
        # one.  `evidence_sha256` is a published identity: it is carried in
        # sealed campaign journals and in the prompt-shape commitment as
        # `parent_evidence_sha256s` and `common_ancestor_evidence_sha256`.
        # Adding an always-present key would re-hash every evaluation ever
        # recorded and invalidate that material.  Omitting the key when the
        # tuple is empty makes the serialization byte-identical for every
        # existing payload, so old records verify unchanged while new
        # observations are still sealed rather than merely reported.
        if self.observations:
            record["observations"] = [
                [name, value.hex()] for name, value in self.observations
            ]
        return record

    @property
    def evidence_sha256(self) -> str:
        canonical = json.dumps(
            self._identity_record(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_HASH_DOMAIN + canonical).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            **self._identity_record(),
            "objectives": {name: value for name, value in self.objectives},
            "violations": {name: value for name, value in self.violations},
            "observations": {name: value for name, value in self.observations},
            "timings": self.timings.to_record(),
            "evidence_sha256": self.evidence_sha256,
        }


__all__ = [
    "DetailedEvaluation",
    "DetailedEvaluationAdapter",
    "DetailedEvaluationPayload",
    "EvaluationCheck",
    "EvaluationCheckStatus",
    "EvaluationTimings",
    "EvaluatorIdentity",
    "normalize_detailed_payload",
]
