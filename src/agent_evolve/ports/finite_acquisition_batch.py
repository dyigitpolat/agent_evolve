"""Workload-neutral scoring boundary for finite acquisition slates.

The ordinary finite-acquisition port asks a numerical optimizer to construct a
batch.  AgentEvolve also needs a stricter comparison primitive: score several
already-enumerated, equally sized slates under one fitted model and one Monte
Carlo draw.  Keeping that operation behind a port lets an application policy
retain an incumbent optimizer slate while testing agent-proposed alternatives
without importing BoTorch or a workload schema into the core.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionRequest,
)


_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$")
_REQUEST_DOMAIN = b"agent-evolve:finite-acquisition-batch-score-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:finite-acquisition-batch-score-decision:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionSlate:
    """One unordered, canonical candidate-ID set to score jointly."""

    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty exact tuple")
        for value in self.candidate_ids:
            _require_token(value, name="candidate_id")
        if self.candidate_ids != tuple(sorted(set(self.candidate_ids))):
            raise ValueError("candidate_ids must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"candidate_ids": list(self.candidate_ids)}


@dataclass(frozen=True, slots=True, eq=False)
class FiniteAcquisitionBatchScoreRequest:
    """Score a sealed family of equal-cardinality finite slates."""

    campaign_scope_sha256: str
    cutoff_index: int
    seed: int
    objectives: tuple[FiniteAcquisitionObjective, ...]
    observations: tuple[FiniteAcquisitionObservation, ...]
    candidates: tuple[FiniteAcquisitionCandidate, ...]
    slates: tuple[FiniteAcquisitionSlate, ...]
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.cutoff_index) is not int or self.cutoff_index < 1:
            raise ValueError("cutoff_index must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        if type(self.slates) is not tuple or not self.slates:
            raise ValueError("slates must be a non-empty exact tuple")
        for value in self.slates:
            if type(value) is not FiniteAcquisitionSlate:
                raise TypeError("slates must contain exact finite slates")
            value.__post_init__()
        if self.slates != tuple(
            sorted(set(self.slates), key=lambda value: value.candidate_ids)
        ):
            raise ValueError("slates must be unique and canonically ordered")
        widths = {len(value.candidate_ids) for value in self.slates}
        if len(widths) != 1:
            raise ValueError("all acquisition slates must have one cardinality")
        batch_size = next(iter(widths))
        # Reuse the ordinary ask boundary as the single validation law for
        # objectives, observations, features, and candidate uniqueness.
        base = FiniteAcquisitionRequest(
            campaign_scope_sha256=self.campaign_scope_sha256,
            cutoff_index=self.cutoff_index,
            batch_size=batch_size,
            seed=self.seed,
            objectives=self.objectives,
            observations=self.observations,
            candidates=self.candidates,
        )
        available = {value.candidate_id for value in base.candidates}
        if any(not set(value.candidate_ids) <= available for value in self.slates):
            raise ValueError("an acquisition slate escapes the sealed candidate pool")
        computed = hashlib.sha256(
            _REQUEST_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    @property
    def batch_size(self) -> int:
        return len(self.slates[0].candidate_ids)

    @property
    def base_request(self) -> FiniteAcquisitionRequest:
        """The ordinary finite-acquisition inputs shared by every slate."""

        return FiniteAcquisitionRequest(
            campaign_scope_sha256=self.campaign_scope_sha256,
            cutoff_index=self.cutoff_index,
            batch_size=self.batch_size,
            seed=self.seed,
            objectives=self.objectives,
            observations=self.observations,
            candidates=self.candidates,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "base_request": self.base_request.to_record(),
            "slates": [value.to_record() for value in self.slates],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FiniteAcquisitionBatchScoreRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionSlateScore:
    slate: FiniteAcquisitionSlate
    log_acquisition_value: float

    def __post_init__(self) -> None:
        if type(self.slate) is not FiniteAcquisitionSlate:
            raise TypeError("slate must be an exact FiniteAcquisitionSlate")
        self.slate.__post_init__()
        _finite(self.log_acquisition_value, name="log_acquisition_value")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "slate": self.slate.to_record(),
            "log_acquisition_value_hex": self.log_acquisition_value.hex(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class FiniteAcquisitionBatchScoreDecision:
    request_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    scores: tuple[FiniteAcquisitionSlateScore, ...]
    diagnostics: tuple[tuple[str, str], ...] = ()
    decision_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise ValueError("policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        if type(self.scores) is not tuple or not self.scores:
            raise ValueError("scores must be a non-empty exact tuple")
        for value in self.scores:
            if type(value) is not FiniteAcquisitionSlateScore:
                raise TypeError("scores must contain exact slate scores")
            value.__post_init__()
        slates = tuple(value.slate for value in self.scores)
        if slates != tuple(sorted(set(slates), key=lambda value: value.candidate_ids)):
            raise ValueError("score slates must be unique and canonical")
        if type(self.diagnostics) is not tuple:
            raise TypeError("diagnostics must be an exact tuple")
        keys: list[str] = []
        for key, value in self.diagnostics:
            _require_token(key, name="diagnostic key")
            if type(value) is not str or not value.isascii():
                raise ValueError("diagnostic values must be ASCII strings")
            keys.append(key)
        if tuple(keys) != tuple(sorted(set(keys))):
            raise ValueError("diagnostics must be unique and key-sorted")
        computed = hashlib.sha256(
            _DECISION_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.decision_sha256 not in ("", computed):
            raise ValueError("decision_sha256 does not authenticate the decision")
        object.__setattr__(self, "decision_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "scores": [value.to_record() for value in self.scores],
            "diagnostics": [
                {"key": key, "value": value} for key, value in self.diagnostics
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FiniteAcquisitionBatchScoreDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


def validate_finite_acquisition_batch_score_decision(
    request: FiniteAcquisitionBatchScoreRequest,
    decision: FiniteAcquisitionBatchScoreDecision,
) -> None:
    """Require exact coverage of every requested slate and policy binding."""

    if type(request) is not FiniteAcquisitionBatchScoreRequest:
        raise TypeError("request must be exact")
    request.__post_init__()
    if type(decision) is not FiniteAcquisitionBatchScoreDecision:
        raise TypeError("decision must be exact")
    decision.__post_init__()
    if decision.request_sha256 != request.request_sha256:
        raise ValueError("batch-score decision targets another request")
    if tuple(value.slate for value in decision.scores) != request.slates:
        raise ValueError("batch-score decision must cover every requested slate")


@runtime_checkable
class FiniteAcquisitionBatchScorePolicy(Protocol):
    """Score fixed slates under one past-only acquisition realization."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def score(
        self,
        request: FiniteAcquisitionBatchScoreRequest,
    ) -> FiniteAcquisitionBatchScoreDecision: ...


__all__ = [
    "FiniteAcquisitionBatchScoreDecision",
    "FiniteAcquisitionBatchScorePolicy",
    "FiniteAcquisitionBatchScoreRequest",
    "FiniteAcquisitionSlate",
    "FiniteAcquisitionSlateScore",
    "validate_finite_acquisition_batch_score_decision",
]
