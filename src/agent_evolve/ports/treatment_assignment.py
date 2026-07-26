"""Provider-neutral contracts for prospective blinded treatment assignment.

The types in this module distinguish scientific treatments, their repeatable
occurrences, and provider-facing opaque slots.  They authenticate exact input
order and both directions of the resulting bijection.  The receipt's claim is
deliberately narrow: it establishes label/order assignment blinding, never
content blinding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re

from agent_evolve.domain.patch import require_sha256


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_SEED_MATERIAL = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,255}$")
_POLICY_BINDING_DOMAIN = b"agent-evolve:treatment-assignment-policy:v1\x00"
_INPUT_DOMAIN = b"agent-evolve:treatment-assignment-input:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:treatment-assignment-receipt:v1\x00"


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


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must use the canonical lowercase identifier grammar")
    return value


@dataclass(frozen=True, slots=True)
class TreatmentId:
    """Nominal identifier for a scientific treatment; repeats are permitted."""

    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "treatment_id")


@dataclass(frozen=True, slots=True)
class TreatmentOccurrenceId:
    """Nominal identifier for one unique occurrence of a treatment."""

    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "occurrence_id")


@dataclass(frozen=True, slots=True)
class OpaqueProviderSlotId:
    """Nominal provider-facing slot identifier with no treatment semantics."""

    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "opaque_provider_slot_id")


@dataclass(frozen=True, slots=True)
class TreatmentOccurrence:
    """One canonical input occurrence, optionally bound to call/request identity."""

    occurrence_id: TreatmentOccurrenceId
    treatment_id: TreatmentId
    call_identity: str | None = None
    request_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.occurrence_id) is not TreatmentOccurrenceId:
            raise TypeError("occurrence_id must be an exact TreatmentOccurrenceId")
        if type(self.treatment_id) is not TreatmentId:
            raise TypeError("treatment_id must be an exact TreatmentId")
        self.occurrence_id.__post_init__()
        self.treatment_id.__post_init__()
        if self.call_identity is not None:
            _require_identifier(self.call_identity, "call_identity")
        if self.request_identity_sha256 is not None:
            require_sha256(
                self.request_identity_sha256,
                "request_identity_sha256",
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "occurrence_id": self.occurrence_id.value,
            "treatment_id": self.treatment_id.value,
            "call_identity": self.call_identity,
            "request_identity_sha256": self.request_identity_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class TreatmentAssignmentPolicyBinding:
    """Identified public permutation policy and domain separation label."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    permutation_domain: str

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        _require_identifier(self.permutation_domain, "permutation_domain")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "permutation_domain": self.permutation_domain,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_POLICY_BINDING_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TreatmentAssignmentPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TreatmentAssignmentInput:
    """Exact prospective input order and public seed material."""

    experiment_commitment_sha256: str
    public_seed_material: str
    occurrences: tuple[TreatmentOccurrence, ...]
    provider_slot_ids: tuple[OpaqueProviderSlotId, ...]

    def __post_init__(self) -> None:
        require_sha256(
            self.experiment_commitment_sha256,
            "experiment_commitment_sha256",
        )
        if (
            type(self.public_seed_material) is not str
            or _SEED_MATERIAL.fullmatch(self.public_seed_material) is None
        ):
            raise ValueError(
                "public_seed_material must use the canonical public-seed grammar"
            )
        if type(self.occurrences) is not tuple or not self.occurrences or any(
            type(value) is not TreatmentOccurrence for value in self.occurrences
        ):
            raise ValueError("occurrences must be a non-empty exact tuple")
        if type(self.provider_slot_ids) is not tuple or any(
            type(value) is not OpaqueProviderSlotId
            for value in self.provider_slot_ids
        ):
            raise TypeError("provider_slot_ids must be an exact tuple")
        if len(self.provider_slot_ids) != len(self.occurrences):
            raise ValueError("provider slot and occurrence counts must match exactly")
        for value in self.occurrences:
            value.__post_init__()
        for value in self.provider_slot_ids:
            value.__post_init__()

        occurrence_ids = tuple(value.occurrence_id.value for value in self.occurrences)
        slot_ids = tuple(value.value for value in self.provider_slot_ids)
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("occurrence IDs cannot repeat")
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("opaque provider slot IDs cannot repeat")

        treatment_ids = {value.treatment_id.value for value in self.occurrences}
        occurrence_id_set = set(occurrence_ids)
        slot_id_set = set(slot_ids)
        if (
            treatment_ids & occurrence_id_set
            or treatment_ids & slot_id_set
            or occurrence_id_set & slot_id_set
        ):
            raise ValueError(
                "treatment, occurrence, and provider-slot identifier namespaces "
                "must be disjoint"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "experiment_commitment_sha256": self.experiment_commitment_sha256,
            "public_seed_material": self.public_seed_material,
            "occurrence_input_order": [
                value.to_record() for value in self.occurrences
            ],
            "provider_slot_input_order": [
                value.value for value in self.provider_slot_ids
            ],
        }

    @property
    def input_sha256(self) -> str:
        return _hash(_INPUT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "input_sha256": self.input_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TreatmentAssignmentInput
            and self.input_sha256 == other.input_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class TreatmentAssignment:
    """One authenticated edge in the occurrence/provider-slot bijection."""

    occurrence_id: TreatmentOccurrenceId
    treatment_id: TreatmentId
    opaque_provider_slot_id: OpaqueProviderSlotId
    occurrence_input_index: int
    provider_slot_input_index: int
    call_identity: str | None = None
    request_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.occurrence_id) is not TreatmentOccurrenceId:
            raise TypeError("occurrence_id must be an exact TreatmentOccurrenceId")
        if type(self.treatment_id) is not TreatmentId:
            raise TypeError("treatment_id must be an exact TreatmentId")
        if type(self.opaque_provider_slot_id) is not OpaqueProviderSlotId:
            raise TypeError(
                "opaque_provider_slot_id must be an exact OpaqueProviderSlotId"
            )
        self.occurrence_id.__post_init__()
        self.treatment_id.__post_init__()
        self.opaque_provider_slot_id.__post_init__()
        for name in ("occurrence_input_index", "provider_slot_input_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.call_identity is not None:
            _require_identifier(self.call_identity, "call_identity")
        if self.request_identity_sha256 is not None:
            require_sha256(
                self.request_identity_sha256,
                "request_identity_sha256",
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "occurrence_id": self.occurrence_id.value,
            "treatment_id": self.treatment_id.value,
            "opaque_provider_slot_id": self.opaque_provider_slot_id.value,
            "occurrence_input_index": self.occurrence_input_index,
            "provider_slot_input_index": self.provider_slot_input_index,
            "call_identity": self.call_identity,
            "request_identity_sha256": self.request_identity_sha256,
        }


class TreatmentAssignmentBlindingScope(str, Enum):
    """The only blinding scope this receipt is capable of establishing."""

    LABEL_AND_ORDER_ONLY = "label_and_order_only"


def _blinding_claim_record() -> dict[str, object]:
    return {
        "scope": TreatmentAssignmentBlindingScope.LABEL_AND_ORDER_ONLY.value,
        "opaque_provider_slot_ids": True,
        "content_blinding_claimed": False,
    }


@dataclass(frozen=True, slots=True, eq=False)
class ProspectiveTreatmentAssignmentReceipt:
    """Authenticated prospective assignment with explicit forward and inverse maps."""

    policy: TreatmentAssignmentPolicyBinding
    assignment_input_sha256: str
    experiment_commitment_sha256: str
    public_seed_material: str
    occurrence_input_order: tuple[TreatmentOccurrence, ...]
    provider_slot_input_order: tuple[OpaqueProviderSlotId, ...]
    ordinal_permutation: tuple[int, ...]
    inverse_ordinal_permutation: tuple[int, ...]
    slot_to_occurrence: tuple[TreatmentAssignment, ...]
    occurrence_to_slot: tuple[TreatmentAssignment, ...]

    def __post_init__(self) -> None:
        if type(self.policy) is not TreatmentAssignmentPolicyBinding:
            raise TypeError("policy must be an exact TreatmentAssignmentPolicyBinding")
        self.policy.__post_init__()
        require_sha256(self.assignment_input_sha256, "assignment_input_sha256")
        assignment_input = TreatmentAssignmentInput(
            experiment_commitment_sha256=self.experiment_commitment_sha256,
            public_seed_material=self.public_seed_material,
            occurrences=self.occurrence_input_order,
            provider_slot_ids=self.provider_slot_input_order,
        )
        if self.assignment_input_sha256 != assignment_input.input_sha256:
            raise ValueError(
                "assignment input commitment does not match receipt inputs"
            )

        count = len(self.occurrence_input_order)
        for name in ("ordinal_permutation", "inverse_ordinal_permutation"):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                type(index) is not int for index in value
            ):
                raise TypeError(f"{name} must be an exact integer tuple")
            if len(value) != count or tuple(sorted(value)) != tuple(range(count)):
                raise ValueError(f"{name} must be a complete exact permutation")
        for provider_index, occurrence_index in enumerate(self.ordinal_permutation):
            if self.inverse_ordinal_permutation[occurrence_index] != provider_index:
                raise ValueError("ordinal permutation and inverse disagree")

        for name in ("slot_to_occurrence", "occurrence_to_slot"):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                type(item) is not TreatmentAssignment for item in value
            ):
                raise TypeError(f"{name} must be an exact assignment tuple")
            if len(value) != count:
                raise ValueError(f"{name} must cover every occurrence exactly once")
            for item in value:
                item.__post_init__()

        expected_by_occurrence: list[TreatmentAssignment | None] = [None] * count
        for provider_index, assignment in enumerate(self.slot_to_occurrence):
            occurrence_index = self.ordinal_permutation[provider_index]
            occurrence = self.occurrence_input_order[occurrence_index]
            if (
                assignment.provider_slot_input_index != provider_index
                or assignment.opaque_provider_slot_id
                != self.provider_slot_input_order[provider_index]
                or assignment.occurrence_input_index != occurrence_index
                or assignment.occurrence_id != occurrence.occurrence_id
                or assignment.treatment_id != occurrence.treatment_id
                or assignment.call_identity != occurrence.call_identity
                or assignment.request_identity_sha256
                != occurrence.request_identity_sha256
            ):
                raise ValueError("slot-to-occurrence mapping differs from bound inputs")
            if expected_by_occurrence[occurrence_index] is not None:
                raise ValueError("slot-to-occurrence mapping repeats an occurrence")
            expected_by_occurrence[occurrence_index] = assignment

        if any(value is None for value in expected_by_occurrence):
            raise ValueError("slot-to-occurrence mapping is not a complete bijection")
        for occurrence_index, assignment in enumerate(self.occurrence_to_slot):
            if assignment is not expected_by_occurrence[occurrence_index]:
                if assignment != expected_by_occurrence[occurrence_index]:
                    raise ValueError(
                        "occurrence-to-slot mapping is not the exact inverse"
                    )
            if assignment.provider_slot_input_index != self.inverse_ordinal_permutation[
                occurrence_index
            ]:
                raise ValueError("inverse mapping differs from inverse permutation")

        if len(
            {value.occurrence_id.value for value in self.slot_to_occurrence}
        ) != count:
            raise ValueError("assignment repeats an occurrence ID")
        if len(
            {
                value.opaque_provider_slot_id.value
                for value in self.slot_to_occurrence
            }
        ) != count:
            raise ValueError("assignment repeats an opaque provider slot ID")

    @property
    def blinding_scope(self) -> TreatmentAssignmentBlindingScope:
        return TreatmentAssignmentBlindingScope.LABEL_AND_ORDER_ONLY

    @property
    def content_blinding_claimed(self) -> bool:
        return False

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "prospective": True,
            "policy": self.policy.to_record(),
            "assignment_input_sha256": self.assignment_input_sha256,
            "experiment_commitment_sha256": self.experiment_commitment_sha256,
            "public_seed_material": self.public_seed_material,
            "occurrence_input_order": [
                value.to_record() for value in self.occurrence_input_order
            ],
            "provider_slot_input_order": [
                value.value for value in self.provider_slot_input_order
            ],
            "ordinal_permutation": list(self.ordinal_permutation),
            "inverse_ordinal_permutation": list(self.inverse_ordinal_permutation),
            "slot_to_occurrence": [
                value.to_record() for value in self.slot_to_occurrence
            ],
            "occurrence_to_slot": [
                value.to_record() for value in self.occurrence_to_slot
            ],
            "blinding_claim": _blinding_claim_record(),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ProspectiveTreatmentAssignmentReceipt
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


__all__ = [
    "OpaqueProviderSlotId",
    "ProspectiveTreatmentAssignmentReceipt",
    "TreatmentAssignment",
    "TreatmentAssignmentBlindingScope",
    "TreatmentAssignmentInput",
    "TreatmentAssignmentPolicyBinding",
    "TreatmentId",
    "TreatmentOccurrence",
    "TreatmentOccurrenceId",
]
