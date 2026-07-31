"""Generic campaign boundary for filling an under-realized candidate stage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import OptimizerState
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_REQUEST_DOMAIN = b"agent-evolve:campaign-capacity-recourse-request:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:campaign-capacity-recourse-result:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    EvolutionCandidate.__post_init__(candidate)
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "generation": candidate.generation,
        "valid": candidate.valid,
        "objectives": [
            {"metric_id": key, "value_hex": value.hex()}
            for key, value in candidate.objectives
        ],
    }


def _policy_identity(
    port: "CampaignCapacityRecoursePort",
) -> tuple[str, int, str]:
    if not isinstance(port, CampaignCapacityRecoursePort):
        raise TypeError("capacity recourse must implement its port")
    values = (
        getattr(port, "policy_id", None),
        getattr(port, "policy_version", None),
        getattr(port, "definition_sha256", None),
    )
    if type(values[0]) is not str or _TOKEN.fullmatch(values[0]) is None:
        raise ValueError("capacity recourse policy_id has invalid syntax")
    if type(values[1]) is not int or values[1] <= 0:
        raise ValueError("capacity recourse policy_version must be positive")
    require_sha256(values[2], "capacity recourse definition_sha256")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CampaignCapacityRecourseRequest:
    campaign_scope_sha256: str
    preparation_sha256: str
    stage_request_sha256: str
    generation: int
    planned_candidate_occurrences: int
    realized_candidate_occurrences: int
    state: OptimizerState
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(self.stage_request_sha256, "stage_request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in (
            "planned_candidate_occurrences",
            "realized_candidate_occurrences",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.realized_candidate_occurrences >= self.planned_candidate_occurrences:
            raise ValueError("capacity recourse requires a strictly underfilled stage")
        if type(self.state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        OptimizerState.__post_init__(self.state)
        if self.state.generation != self.generation:
            raise ValueError("optimizer state generation differs from recourse stage")
        computed = hashlib.sha256(
            _REQUEST_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    @property
    def missing_candidate_occurrences(self) -> int:
        return self.planned_candidate_occurrences - self.realized_candidate_occurrences

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "preparation_sha256": self.preparation_sha256,
            "stage_request_sha256": self.stage_request_sha256,
            "generation": self.generation,
            "planned_candidate_occurrences": self.planned_candidate_occurrences,
            "realized_candidate_occurrences": self.realized_candidate_occurrences,
            "missing_candidate_occurrences": self.missing_candidate_occurrences,
            "state": {
                "archive_snapshot_hash": self.state.archive_snapshot_hash,
                "unique_evaluations": self.state.unique_evaluations,
                "logical_llm_calls": self.state.logical_llm_calls,
                "candidates": [_candidate_record(value) for value in self.state.candidates],
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class CampaignCapacityRecourseResult:
    request_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    candidates: tuple[EvolutionCandidate, ...]
    evidence: FrozenJsonObject
    result_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id has invalid syntax")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("capacity recourse must return a non-empty exact tuple")
        if any(type(value) is not EvolutionCandidate for value in self.candidates):
            raise TypeError("capacity recourse candidates must be exact")
        for value in self.candidates:
            EvolutionCandidate.__post_init__(value)
        if len({value.candidate_id for value in self.candidates}) != len(self.candidates):
            raise ValueError("capacity recourse candidate IDs must be unique")
        if len({value.occurrence.configuration_hash for value in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("capacity recourse configurations must be unique")
        if type(self.evidence) is not FrozenJsonObject or freeze_json(self.evidence) is not self.evidence:
            raise TypeError("evidence must be a frozen typed-JSON object")
        computed = hashlib.sha256(
            _RESULT_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.result_sha256 not in ("", computed):
            raise ValueError("result_sha256 does not authenticate the result")
        object.__setattr__(self, "result_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "candidates": [_candidate_record(value) for value in self.candidates],
            "evidence": thaw_json(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "result_sha256": self.result_sha256}


@runtime_checkable
class CampaignCapacityRecoursePort(Protocol):
    """Fill every requested stage occurrence from the current optimizer state."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    async def fill(
        self,
        request: CampaignCapacityRecourseRequest,
    ) -> CampaignCapacityRecourseResult: ...


def validate_campaign_capacity_recourse_result(
    *,
    port: CampaignCapacityRecoursePort,
    request: CampaignCapacityRecourseRequest,
    result: CampaignCapacityRecourseResult,
) -> None:
    identity = _policy_identity(port)
    if type(request) is not CampaignCapacityRecourseRequest:
        raise TypeError("request must be exact")
    request.__post_init__()
    if type(result) is not CampaignCapacityRecourseResult:
        raise TypeError("capacity recourse returned a foreign result")
    result.__post_init__()
    if result.request_sha256 != request.request_sha256:
        raise ValueError("capacity recourse result targets another request")
    if (
        result.policy_id,
        result.policy_version,
        result.policy_definition_sha256,
    ) != identity:
        raise ValueError("capacity recourse result has a foreign policy identity")
    if len(result.candidates) != request.missing_candidate_occurrences:
        raise ValueError("capacity recourse did not fill every missing occurrence")
    if any(value.generation != request.generation for value in result.candidates):
        raise ValueError("capacity recourse candidate has the wrong generation")
    known_ids = {value.candidate_id for value in request.state.candidates}
    known_configurations = {
        value.occurrence.configuration_hash for value in request.state.candidates
    }
    if any(value.candidate_id in known_ids for value in result.candidates):
        raise ValueError("capacity recourse repeated a known candidate occurrence")
    if any(
        value.occurrence.configuration_hash in known_configurations
        for value in result.candidates
    ):
        raise ValueError("capacity recourse repeated an evaluated configuration")


def validate_campaign_capacity_recourse_port(
    port: CampaignCapacityRecoursePort,
) -> tuple[str, int, str]:
    return _policy_identity(port)


__all__ = [
    "CampaignCapacityRecoursePort",
    "CampaignCapacityRecourseRequest",
    "CampaignCapacityRecourseResult",
    "validate_campaign_capacity_recourse_port",
    "validate_campaign_capacity_recourse_result",
]
