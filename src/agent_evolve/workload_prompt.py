"""Provenance-bound optional workload prompting and ablation views.

AgentEvolve's behavioral workflow remains workload-neutral.  A benchmark may,
however, publish static domain semantics that were frozen before an optimization
campaign: metric meanings, candidate-field semantics, legal invariants, action
semantics, or general reasoning guidance.  These facts are context, not measured
search evidence, and must never be assembled from evaluator outcomes or prior
campaign traces.

The extension is deliberately a typed-JSON view rather than a free callback.
This makes its exact bytes precommittable, keeps it inside the existing request
and prompt hashes, and supports three explicit experimental arms:

``SCHEMA_ONLY``
    Omit the extension entirely.
``SEMANTIC``
    Render the workload-authored semantic payload.
``MATCHED_CONTROL``
    Render an optional workload-authored structure-matched control payload,
    typically with stable opaque labels or neutral wording.

The API can enforce provenance declarations and structural matching.  It
cannot prove that prose was written honestly; confirmatory experiments must
still freeze source artifacts and prompt bytes before optimizer outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY = "workload_prompt_extension"

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_LOWER_SHA256 = frozenset("0123456789abcdef")
_MAX_PAYLOAD_BYTES = 65_536
_PROVENANCE_DOMAIN = b"agent-evolve:workload-prompt-provenance:v1\x00"
_EXTENSION_DOMAIN = b"agent-evolve:workload-prompt-extension:v1\x00"
_VIEW_DOMAIN = b"agent-evolve:workload-prompt-extension-view:v1\x00"

_ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        "action_semantics",
        "candidate_semantics",
        "constraints_and_invariants",
        "domain_context",
        "metric_semantics",
        "reasoning_guidance",
        "reference_facts",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_token(value: object, *, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")
    return value


def _payload_shape(value: object) -> object:
    """Return a value-free recursive shape for a matched prompt control."""

    if type(value) is FrozenJsonObject:
        return {
            key: _payload_shape(child)
            for key, child in value.items
        }
    if type(value) is FrozenJsonArray:
        return [_payload_shape(child) for child in value.items]
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    raise TypeError("prompt payload escaped frozen typed JSON")


def _validate_payload(value: object, *, name: str) -> FrozenJsonObject:
    if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
        raise TypeError(f"{name} must be an exact frozen typed-JSON object")
    if not value.items:
        raise ValueError(f"{name} must not be empty")
    unknown = {key for key, _ in value.items} - _ALLOWED_PAYLOAD_KEYS
    if unknown:
        raise ValueError(
            f"{name} contains unsupported top-level keys: "
            + ",".join(sorted(unknown))
        )
    size = len(_canonical_bytes(thaw_json(value)))
    if size > _MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"{name} exceeds the {_MAX_PAYLOAD_BYTES}-byte static prompt limit"
        )
    return value


class WorkloadPromptSourceKind(str, Enum):
    """Closed provenance classes that predate campaign outcomes."""

    BENCHMARK_SPECIFICATION = "benchmark_specification"
    PREEXISTING_DOMAIN_KNOWLEDGE = "preexisting_domain_knowledge"
    PUBLIC_REFERENCE = "public_reference"


class WorkloadPromptArm(str, Enum):
    """Prospective arms for the workload-semantics ablation."""

    SCHEMA_ONLY = "schema_only"
    SEMANTIC = "semantic"
    MATCHED_CONTROL = "matched_control"


@dataclass(frozen=True, slots=True)
class WorkloadPromptProvenance:
    """Fail-closed declaration for static, outcome-free prompt knowledge."""

    source_kind: WorkloadPromptSourceKind
    source_artifact_sha256s: tuple[str, ...]
    frozen_before_campaign: bool = True
    evaluator_outcomes_accessed: bool = False
    campaign_traces_accessed: bool = False
    tuned_on_benchmark_outcomes: bool = False

    def __post_init__(self) -> None:
        if type(self.source_kind) is not WorkloadPromptSourceKind:
            raise TypeError("source_kind must be an exact WorkloadPromptSourceKind")
        if (
            type(self.source_artifact_sha256s) is not tuple
            or not self.source_artifact_sha256s
        ):
            raise ValueError("source_artifact_sha256s must be a non-empty tuple")
        for value in self.source_artifact_sha256s:
            _require_sha256(value, name="source_artifact_sha256")
        if self.source_artifact_sha256s != tuple(
            sorted(set(self.source_artifact_sha256s))
        ):
            raise ValueError("source artifact hashes must be unique and canonical")
        for name in (
            "frozen_before_campaign",
            "evaluator_outcomes_accessed",
            "campaign_traces_accessed",
            "tuned_on_benchmark_outcomes",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if not self.frozen_before_campaign:
            raise ValueError("workload prompt knowledge must be frozen before campaign")
        if self.evaluator_outcomes_accessed:
            raise ValueError("workload prompt knowledge cannot access evaluator outcomes")
        if self.campaign_traces_accessed:
            raise ValueError("workload prompt knowledge cannot access campaign traces")
        if self.tuned_on_benchmark_outcomes:
            raise ValueError("workload prompt knowledge cannot be outcome-tuned")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_kind": self.source_kind.value,
            "source_artifact_sha256s": list(self.source_artifact_sha256s),
            "frozen_before_campaign": self.frozen_before_campaign,
            "evaluator_outcomes_accessed": self.evaluator_outcomes_accessed,
            "campaign_traces_accessed": self.campaign_traces_accessed,
            "tuned_on_benchmark_outcomes": self.tuned_on_benchmark_outcomes,
        }

    @property
    def provenance_sha256(self) -> str:
        return _sha256(_PROVENANCE_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class WorkloadPromptExtensionView:
    """One immutable model-visible arm derived from a complete extension."""

    extension_id: str
    extension_version: int
    extension_definition_sha256: str
    arm: WorkloadPromptArm
    payload: FrozenJsonObject
    provenance: WorkloadPromptProvenance
    payload_sha256: str = field(init=False)
    view_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.extension_id, name="extension_id")
        if type(self.extension_version) is not int or self.extension_version <= 0:
            raise ValueError("extension_version must be a positive exact integer")
        _require_sha256(
            self.extension_definition_sha256,
            name="extension_definition_sha256",
        )
        if self.arm not in {
            WorkloadPromptArm.SEMANTIC,
            WorkloadPromptArm.MATCHED_CONTROL,
        }:
            raise ValueError("a rendered extension view must use a visible arm")
        _validate_payload(self.payload, name="payload")
        if type(self.provenance) is not WorkloadPromptProvenance:
            raise TypeError("provenance must be exact WorkloadPromptProvenance")
        WorkloadPromptProvenance.__post_init__(self.provenance)
        payload_sha256 = typed_json_sha256(self.payload)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(
            self,
            "view_sha256",
            _sha256(_VIEW_DOMAIN, self._unsigned_binding_record()),
        )

    def _unsigned_binding_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "extension_id": self.extension_id,
            "extension_version": self.extension_version,
            "extension_definition_sha256": self.extension_definition_sha256,
            "arm": self.arm.value,
            "payload_sha256": typed_json_sha256(self.payload),
            "provenance_sha256": self.provenance.provenance_sha256,
        }

    def to_binding_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_binding_record(),
            "provenance": self.provenance.to_record(),
            "view_sha256": self.view_sha256,
        }

    def to_prompt_record(self) -> dict[str, object]:
        """Return the explicit static-context record rendered to the model."""

        self.__post_init__()
        return {
            "schema_version": 1,
            "extension_id": self.extension_id,
            "extension_version": self.extension_version,
            "extension_definition_sha256": self.extension_definition_sha256,
            "arm": self.arm.value,
            "view_sha256": self.view_sha256,
            "payload": thaw_json(self.payload),
            "evidence_status": (
                "static_preoptimization_context_not_measured_search_evidence"
            ),
            "provenance": {
                "source_kind": self.provenance.source_kind.value,
                "source_artifact_sha256s": list(
                    self.provenance.source_artifact_sha256s
                ),
                "frozen_before_campaign": True,
                "evaluator_outcomes_accessed": False,
                "campaign_traces_accessed": False,
                "tuned_on_benchmark_outcomes": False,
            },
        }


@dataclass(frozen=True, slots=True)
class WorkloadPromptExtension:
    """Static semantic prompt plus an optional structure-matched control."""

    extension_id: str
    extension_version: int
    semantic_payload: FrozenJsonObject
    provenance: WorkloadPromptProvenance
    matched_control_payload: FrozenJsonObject | None = None
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.extension_id, name="extension_id")
        if type(self.extension_version) is not int or self.extension_version <= 0:
            raise ValueError("extension_version must be a positive exact integer")
        _validate_payload(self.semantic_payload, name="semantic_payload")
        if type(self.provenance) is not WorkloadPromptProvenance:
            raise TypeError("provenance must be exact WorkloadPromptProvenance")
        WorkloadPromptProvenance.__post_init__(self.provenance)
        matched = self.matched_control_payload
        if matched is not None:
            _validate_payload(matched, name="matched_control_payload")
            if _payload_shape(matched) != _payload_shape(self.semantic_payload):
                raise ValueError(
                    "matched control payload must preserve the semantic payload shape"
                )
            if typed_json_sha256(matched) == typed_json_sha256(
                self.semantic_payload
            ):
                raise ValueError("matched control must differ from semantic payload")
        definition = {
            "schema_version": 1,
            "extension_id": self.extension_id,
            "extension_version": self.extension_version,
            "semantic_payload_sha256": typed_json_sha256(self.semantic_payload),
            "matched_control_payload_sha256": (
                None if matched is None else typed_json_sha256(matched)
            ),
            "provenance": self.provenance.to_record(),
            "allowed_payload_keys": sorted(_ALLOWED_PAYLOAD_KEYS),
            "outcome_access": False,
        }
        object.__setattr__(
            self,
            "definition_sha256",
            _sha256(_EXTENSION_DOMAIN, definition),
        )

    def view(
        self,
        arm: WorkloadPromptArm,
    ) -> WorkloadPromptExtensionView | None:
        """Resolve one prospective ablation arm without inspecting outcomes."""

        self.__post_init__()
        if type(arm) is not WorkloadPromptArm:
            raise TypeError("arm must be an exact WorkloadPromptArm")
        if arm is WorkloadPromptArm.SCHEMA_ONLY:
            return None
        if arm is WorkloadPromptArm.SEMANTIC:
            payload = self.semantic_payload
        else:
            payload = self.matched_control_payload
            if payload is None:
                raise ValueError(
                    "matched-control arm requires matched_control_payload"
                )
        return WorkloadPromptExtensionView(
            extension_id=self.extension_id,
            extension_version=self.extension_version,
            extension_definition_sha256=self.definition_sha256,
            arm=arm,
            payload=payload,
            provenance=self.provenance,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "extension_id": self.extension_id,
            "extension_version": self.extension_version,
            "definition_sha256": self.definition_sha256,
            "semantic_payload_sha256": typed_json_sha256(self.semantic_payload),
            "matched_control_payload_sha256": (
                None
                if self.matched_control_payload is None
                else typed_json_sha256(self.matched_control_payload)
            ),
            "provenance": self.provenance.to_record(),
        }


__all__ = [
    "WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY",
    "WorkloadPromptArm",
    "WorkloadPromptExtension",
    "WorkloadPromptExtensionView",
    "WorkloadPromptProvenance",
    "WorkloadPromptSourceKind",
]
