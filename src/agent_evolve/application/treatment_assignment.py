"""Deterministic prospective treatment-to-provider-slot assignment.

The permutation is derived only from public committed material, an identified
policy/domain, the occurrence count, and canonical input ordinals.  Treatment,
occurrence, provider-slot, call, and request labels are intentionally excluded
from permutation ranks.  They remain fully bound by the resulting receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from agent_evolve.ports.treatment_assignment import (
    ProspectiveTreatmentAssignmentReceipt,
    TreatmentAssignment,
    TreatmentAssignmentInput,
    TreatmentAssignmentPolicyBinding,
)


HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_ID = (
    "hash_ranked_treatment_occurrence_assignment"
)
HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_VERSION = 1
HASH_RANKED_TREATMENT_ASSIGNMENT_PERMUTATION_DOMAIN = (
    "agent-evolve:prospective-treatment-assignment-rank:v1"
)
HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:hash-ranked-treatment-occurrence-assignment:v1;"
    b"rank=sha256(domain,policy_binding,experiment_commitment,public_seed_"
    b"material,occurrence_count,occurrence_input_index);"
    b"sort=rank_digest_then_occurrence_input_index;"
    b"provider_slots=canonical_input_order;labels_excluded_from_rank;"
    b"runtime_rng=forbidden"
).hexdigest()


class TreatmentAssignmentVerificationError(RuntimeError):
    """A supplied receipt differs from deterministic prospective replay."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def hash_ranked_treatment_assignment_policy() -> TreatmentAssignmentPolicyBinding:
    """Return the identified domain-neutral public permutation policy."""

    return TreatmentAssignmentPolicyBinding(
        policy_id=HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_ID,
        policy_version=HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_VERSION,
        policy_definition_sha256=(
            HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_DEFINITION_SHA256
        ),
        permutation_domain=HASH_RANKED_TREATMENT_ASSIGNMENT_PERMUTATION_DOMAIN,
    )


def treatment_assignment_ordinal_permutation(
    assignment_input: TreatmentAssignmentInput,
    *,
    policy: TreatmentAssignmentPolicyBinding,
) -> tuple[int, ...]:
    """Derive the provider-slot-to-occurrence permutation without runtime RNG."""

    if type(assignment_input) is not TreatmentAssignmentInput:
        raise TypeError("assignment_input must be an exact TreatmentAssignmentInput")
    assignment_input.__post_init__()
    if type(policy) is not TreatmentAssignmentPolicyBinding:
        raise TypeError("policy must be an exact TreatmentAssignmentPolicyBinding")
    policy.__post_init__()

    count = len(assignment_input.occurrences)
    domain = policy.permutation_domain.encode("ascii") + b"\x00"
    ranked: list[tuple[bytes, int]] = []
    for occurrence_input_index in range(count):
        rank_payload = {
            "schema_version": 1,
            "policy_binding_sha256": policy.binding_sha256,
            "experiment_commitment_sha256": (
                assignment_input.experiment_commitment_sha256
            ),
            "public_seed_material": assignment_input.public_seed_material,
            "occurrence_count": count,
            "occurrence_input_index": occurrence_input_index,
        }
        rank_digest = hashlib.sha256(
            domain + _canonical_json(rank_payload)
        ).digest()
        ranked.append((rank_digest, occurrence_input_index))
    return tuple(index for _, index in sorted(ranked))


def assign_treatment_occurrences(
    assignment_input: TreatmentAssignmentInput,
    *,
    policy: TreatmentAssignmentPolicyBinding | None = None,
) -> ProspectiveTreatmentAssignmentReceipt:
    """Create one prospective, replayable assignment and authenticated inverse."""

    if type(assignment_input) is not TreatmentAssignmentInput:
        raise TypeError("assignment_input must be an exact TreatmentAssignmentInput")
    assignment_input.__post_init__()
    effective_policy = (
        hash_ranked_treatment_assignment_policy() if policy is None else policy
    )
    if type(effective_policy) is not TreatmentAssignmentPolicyBinding:
        raise TypeError("policy must be an exact TreatmentAssignmentPolicyBinding")
    effective_policy.__post_init__()

    permutation = treatment_assignment_ordinal_permutation(
        assignment_input,
        policy=effective_policy,
    )
    inverse_values = [0] * len(permutation)
    slot_assignments: list[TreatmentAssignment] = []
    for provider_slot_input_index, occurrence_input_index in enumerate(permutation):
        inverse_values[occurrence_input_index] = provider_slot_input_index
        occurrence = assignment_input.occurrences[occurrence_input_index]
        slot_assignments.append(
            TreatmentAssignment(
                occurrence_id=occurrence.occurrence_id,
                treatment_id=occurrence.treatment_id,
                opaque_provider_slot_id=(
                    assignment_input.provider_slot_ids[provider_slot_input_index]
                ),
                occurrence_input_index=occurrence_input_index,
                provider_slot_input_index=provider_slot_input_index,
                call_identity=occurrence.call_identity,
                request_identity_sha256=occurrence.request_identity_sha256,
            )
        )
    inverse = tuple(inverse_values)
    assignments_by_occurrence: list[TreatmentAssignment | None] = [
        None
    ] * len(permutation)
    for assignment in slot_assignments:
        assignments_by_occurrence[assignment.occurrence_input_index] = assignment
    if any(value is None for value in assignments_by_occurrence):
        raise RuntimeError("internal assignment construction lost an occurrence")

    return ProspectiveTreatmentAssignmentReceipt(
        policy=effective_policy,
        assignment_input_sha256=assignment_input.input_sha256,
        experiment_commitment_sha256=(
            assignment_input.experiment_commitment_sha256
        ),
        public_seed_material=assignment_input.public_seed_material,
        occurrence_input_order=assignment_input.occurrences,
        provider_slot_input_order=assignment_input.provider_slot_ids,
        ordinal_permutation=permutation,
        inverse_ordinal_permutation=inverse,
        slot_to_occurrence=tuple(slot_assignments),
        occurrence_to_slot=tuple(
            value for value in assignments_by_occurrence if value is not None
        ),
    )


def verify_treatment_assignment_receipt(
    assignment_input: TreatmentAssignmentInput,
    receipt: ProspectiveTreatmentAssignmentReceipt | Mapping[str, object],
    *,
    policy: TreatmentAssignmentPolicyBinding | None = None,
) -> ProspectiveTreatmentAssignmentReceipt:
    """Fail closed unless an object or serialized record equals public replay."""

    expected = assign_treatment_occurrences(assignment_input, policy=policy)
    if type(receipt) is ProspectiveTreatmentAssignmentReceipt:
        receipt.__post_init__()
        matches = receipt == expected
    elif isinstance(receipt, Mapping):
        matches = type(receipt) is dict and receipt == expected.to_record()
    else:
        raise TypeError("receipt must be an exact receipt or canonical dict record")
    if not matches:
        raise TreatmentAssignmentVerificationError(
            "treatment assignment receipt differs from deterministic replay"
        )
    return expected


__all__ = [
    "HASH_RANKED_TREATMENT_ASSIGNMENT_PERMUTATION_DOMAIN",
    "HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_DEFINITION_SHA256",
    "HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_ID",
    "HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_VERSION",
    "TreatmentAssignmentVerificationError",
    "assign_treatment_occurrences",
    "hash_ranked_treatment_assignment_policy",
    "treatment_assignment_ordinal_permutation",
    "verify_treatment_assignment_receipt",
]
