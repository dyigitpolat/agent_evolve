"""Outcome-blind exact-feasibility boundary for candidate screening.

This port is deliberately narrower than a learned proxy.  A workload adapter
may reject a configuration only when it can publish an exact, evaluator-bound
proof of infeasibility.  ``UNKNOWN`` is the safe default and must remain in the
ordinary evaluation pool.  Objective values and predicted quality never cross
this boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_REQUEST_DOMAIN = b"agent-evolve:hard-feasibility-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:hard-feasibility-decision:v1\x00"
_DECISION_BATCH_DOMAIN = b"agent-evolve:hard-feasibility-decision-batch:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _identity(port: "HardFeasibilityPort") -> tuple[str, int, str]:
    if not isinstance(port, HardFeasibilityPort):
        raise TypeError("hard feasibility adapter must implement its port")
    values = (
        getattr(port, "policy_id", None),
        getattr(port, "policy_version", None),
        getattr(port, "definition_sha256", None),
    )
    if type(values[0]) is not str or _TOKEN.fullmatch(values[0]) is None:
        raise ValueError("hard feasibility policy_id has invalid syntax")
    if type(values[1]) is not int or values[1] <= 0:
        raise ValueError("hard feasibility policy_version must be positive")
    require_sha256(values[2], "hard feasibility definition_sha256")
    return values  # type: ignore[return-value]


class HardFeasibilityVerdict(str, Enum):
    """Three-valued proof outcome; only ``INFEASIBLE`` authorizes rejection."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HardFeasibilityRequest:
    campaign_scope_sha256: str
    cutoff_index: int
    configuration: FrozenJsonObject
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.cutoff_index) is not int or self.cutoff_index < 1:
            raise ValueError("cutoff_index must be positive")
        if (
            type(self.configuration) is not FrozenJsonObject
            or freeze_json(self.configuration) is not self.configuration
        ):
            raise TypeError("configuration must be a frozen typed-JSON object")
        computed = hashlib.sha256(
            _REQUEST_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "cutoff_index": self.cutoff_index,
            "configuration_sha256": typed_json_sha256(self.configuration),
            "outcome_access": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class HardFeasibilityDecision:
    request_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    verdict: HardFeasibilityVerdict
    proof: FrozenJsonObject
    decision_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id has invalid syntax")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        if type(self.verdict) is not HardFeasibilityVerdict:
            raise TypeError("verdict must be HardFeasibilityVerdict")
        if type(self.proof) is not FrozenJsonObject or freeze_json(self.proof) is not self.proof:
            raise TypeError("proof must be a frozen typed-JSON object")
        proof_record = thaw_json(self.proof)
        if self.verdict is HardFeasibilityVerdict.INFEASIBLE and not proof_record:
            raise ValueError("infeasible decisions require a non-empty exact proof")
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
            "verdict": self.verdict.value,
            "proof": thaw_json(self.proof),
            "rejection_authorized": (
                self.verdict is HardFeasibilityVerdict.INFEASIBLE
            ),
            "objective_access": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}


@runtime_checkable
class HardFeasibilityPort(Protocol):
    """Publish exact static feasibility evidence for one configuration."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def assess(self, request: HardFeasibilityRequest) -> HardFeasibilityDecision: ...


def validate_hard_feasibility_port(
    port: HardFeasibilityPort,
) -> tuple[str, int, str]:
    return _identity(port)


def assess_hard_feasibility(
    port: HardFeasibilityPort,
    request: HardFeasibilityRequest,
) -> HardFeasibilityDecision:
    identity = _identity(port)
    if type(request) is not HardFeasibilityRequest:
        raise TypeError("request must be exact")
    request.__post_init__()
    decision = port.assess(request)
    if type(decision) is not HardFeasibilityDecision:
        raise TypeError("hard feasibility adapter returned a foreign decision")
    decision.__post_init__()
    if decision.request_sha256 != request.request_sha256:
        raise ValueError("hard feasibility decision targets another request")
    if (
        decision.policy_id,
        decision.policy_version,
        decision.policy_definition_sha256,
    ) != identity:
        raise ValueError("hard feasibility decision has a foreign policy identity")
    return decision


def hard_feasibility_decision_batch_sha256(
    decision_sha256s: Sequence[str],
) -> str:
    """Commit an ordered feasibility audit without inlining every decision."""

    if isinstance(decision_sha256s, (str, bytes)) or not isinstance(
        decision_sha256s,
        Sequence,
    ):
        raise TypeError("decision_sha256s must be a non-string sequence")
    values = tuple(decision_sha256s)
    for index, value in enumerate(values):
        require_sha256(value, f"decision_sha256s[{index}]")
    return hashlib.sha256(
        _DECISION_BATCH_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "decision_sha256s": values,
            }
        )
    ).hexdigest()


__all__ = [
    "HardFeasibilityDecision",
    "HardFeasibilityPort",
    "HardFeasibilityRequest",
    "HardFeasibilityVerdict",
    "assess_hard_feasibility",
    "hard_feasibility_decision_batch_sha256",
    "validate_hard_feasibility_port",
]
