from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from agent_evolve.application.treatment_assignment import (
    HASH_RANKED_TREATMENT_ASSIGNMENT_PERMUTATION_DOMAIN,
    HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_DEFINITION_SHA256,
    TreatmentAssignmentVerificationError,
    assign_treatment_occurrences,
    hash_ranked_treatment_assignment_policy,
    treatment_assignment_ordinal_permutation,
    verify_treatment_assignment_receipt,
)
from agent_evolve.ports.treatment_assignment import (
    OpaqueProviderSlotId,
    ProspectiveTreatmentAssignmentReceipt,
    TreatmentAssignment,
    TreatmentAssignmentBlindingScope,
    TreatmentAssignmentInput,
    TreatmentId,
    TreatmentOccurrence,
    TreatmentOccurrenceId,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _assignment_input(
    *,
    treatment_labels: tuple[str, ...] = (
        "treatment.control",
        "treatment.memory",
        "treatment.memory",
        "treatment.recombination",
        "treatment.control",
        "treatment.mutation",
    ),
    occurrence_prefix: str = "occurrence.generic",
    slot_prefix: str = "opaque.slot",
    seed: str = "public.seed.20260715",
    experiment_commitment: str | None = None,
    with_call_bindings: bool = True,
) -> TreatmentAssignmentInput:
    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"{occurrence_prefix}.{index:02d}"),
            treatment_id=TreatmentId(label),
            call_identity=(
                f"call.generic.{index:02d}" if with_call_bindings else None
            ),
            request_identity_sha256=(
                _sha(f"request-generic-{index}") if index % 2 == 0 else None
            ),
        )
        for index, label in enumerate(treatment_labels)
    )
    return TreatmentAssignmentInput(
        experiment_commitment_sha256=(
            _sha("generic-experiment-v1")
            if experiment_commitment is None
            else experiment_commitment
        ),
        public_seed_material=seed,
        occurrences=occurrences,
        provider_slot_ids=tuple(
            OpaqueProviderSlotId(f"{slot_prefix}.{index:02d}")
            for index in range(len(occurrences))
        ),
    )


def _receipt_with_permutation(
    original: ProspectiveTreatmentAssignmentReceipt,
    permutation: tuple[int, ...],
) -> ProspectiveTreatmentAssignmentReceipt:
    inverse_values = [0] * len(permutation)
    by_slot: list[TreatmentAssignment] = []
    for slot_index, occurrence_index in enumerate(permutation):
        inverse_values[occurrence_index] = slot_index
        occurrence = original.occurrence_input_order[occurrence_index]
        by_slot.append(
            TreatmentAssignment(
                occurrence_id=occurrence.occurrence_id,
                treatment_id=occurrence.treatment_id,
                opaque_provider_slot_id=original.provider_slot_input_order[
                    slot_index
                ],
                occurrence_input_index=occurrence_index,
                provider_slot_input_index=slot_index,
                call_identity=occurrence.call_identity,
                request_identity_sha256=occurrence.request_identity_sha256,
            )
        )
    by_occurrence: list[TreatmentAssignment | None] = [None] * len(permutation)
    for value in by_slot:
        by_occurrence[value.occurrence_input_index] = value
    return replace(
        original,
        ordinal_permutation=permutation,
        inverse_ordinal_permutation=tuple(inverse_values),
        slot_to_occurrence=tuple(by_slot),
        occurrence_to_slot=tuple(
            value for value in by_occurrence if value is not None
        ),
    )


def test_assignment_is_prospective_deterministic_bijective_and_replayable() -> None:
    assignment_input = _assignment_input()
    policy = hash_ranked_treatment_assignment_policy()
    first = assign_treatment_occurrences(assignment_input, policy=policy)
    second = assign_treatment_occurrences(assignment_input, policy=policy)

    assert policy.policy_id == "hash_ranked_treatment_occurrence_assignment"
    assert policy.policy_version == 1
    assert policy.permutation_domain == (
        "agent-evolve:prospective-treatment-assignment-rank:v1"
    )
    assert policy.permutation_domain == (
        HASH_RANKED_TREATMENT_ASSIGNMENT_PERMUTATION_DOMAIN
    )
    assert policy.policy_definition_sha256 == _sha(
        "agent-evolve:hash-ranked-treatment-occurrence-assignment:v1;"
        "rank=sha256(domain,policy_binding,experiment_commitment,public_seed_"
        "material,occurrence_count,occurrence_input_index);"
        "sort=rank_digest_then_occurrence_input_index;"
        "provider_slots=canonical_input_order;labels_excluded_from_rank;"
        "runtime_rng=forbidden"
    )
    assert policy.policy_definition_sha256 == (
        HASH_RANKED_TREATMENT_ASSIGNMENT_POLICY_DEFINITION_SHA256
    )
    assert first == second
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.ordinal_permutation == treatment_assignment_ordinal_permutation(
        assignment_input,
        policy=policy,
    )
    assert sorted(first.ordinal_permutation) == list(range(6))
    assert sorted(first.inverse_ordinal_permutation) == list(range(6))
    for slot_index, occurrence_index in enumerate(first.ordinal_permutation):
        assert first.inverse_ordinal_permutation[occurrence_index] == slot_index
        assignment = first.slot_to_occurrence[slot_index]
        assert assignment.provider_slot_input_index == slot_index
        assert assignment.occurrence_input_index == occurrence_index
        assert assignment.opaque_provider_slot_id == (
            assignment_input.provider_slot_ids[slot_index]
        )
        assert first.occurrence_to_slot[occurrence_index] == assignment

    assert verify_treatment_assignment_receipt(
        assignment_input,
        first,
        policy=policy,
    ) == first
    assert verify_treatment_assignment_receipt(
        assignment_input,
        first.to_record(),
        policy=policy,
    ) == first


def test_repeated_treatments_have_unique_occurrences_and_optional_bindings() -> None:
    assignment_input = _assignment_input()
    receipt = assign_treatment_occurrences(assignment_input)

    assert len({value.occurrence_id for value in receipt.slot_to_occurrence}) == 6
    assert len(
        {value.opaque_provider_slot_id for value in receipt.slot_to_occurrence}
    ) == 6
    assert [
        value.treatment_id.value for value in assignment_input.occurrences
    ].count("treatment.memory") == 2
    assert [
        value.treatment_id.value for value in assignment_input.occurrences
    ].count("treatment.control") == 2
    for assignment in receipt.slot_to_occurrence:
        source = assignment_input.occurrences[assignment.occurrence_input_index]
        assert assignment.treatment_id == source.treatment_id
        assert assignment.call_identity == source.call_identity
        assert assignment.request_identity_sha256 == (
            source.request_identity_sha256
        )
    assert any(
        value.request_identity_sha256 is None
        for value in receipt.slot_to_occurrence
    )
    assert any(
        value.request_identity_sha256 is not None
        for value in receipt.slot_to_occurrence
    )


def test_ordinal_permutation_is_invariant_to_domain_and_identifier_labels() -> None:
    generic = _assignment_input()
    renamed = _assignment_input(
        treatment_labels=(
            "solver.baseline",
            "solver.crossover",
            "solver.crossover",
            "solver.archive",
            "solver.baseline",
            "solver.mutate",
        ),
        occurrence_prefix="trial.database",
        slot_prefix="dispatch.blind",
        with_call_bindings=False,
    )
    generic_receipt = assign_treatment_occurrences(generic)
    renamed_receipt = assign_treatment_occurrences(renamed)

    assert generic_receipt.ordinal_permutation == renamed_receipt.ordinal_permutation
    assert generic_receipt.inverse_ordinal_permutation == (
        renamed_receipt.inverse_ordinal_permutation
    )
    assert generic.input_sha256 != renamed.input_sha256
    assert generic_receipt.receipt_sha256 != renamed_receipt.receipt_sha256
    assert [
        value.occurrence_id.value for value in generic_receipt.slot_to_occurrence
    ] != [
        value.occurrence_id.value for value in renamed_receipt.slot_to_occurrence
    ]


def test_exact_input_order_seed_and_experiment_commitment_are_bound() -> None:
    original = _assignment_input()
    reordered = replace(
        original,
        occurrences=tuple(reversed(original.occurrences)),
    )
    reseeded = _assignment_input(seed="public.seed.changed")
    recommitted = _assignment_input(
        experiment_commitment=_sha("generic-experiment-v2")
    )

    original_receipt = assign_treatment_occurrences(original)
    reordered_receipt = assign_treatment_occurrences(reordered)
    reseeded_receipt = assign_treatment_occurrences(reseeded)
    recommitted_receipt = assign_treatment_occurrences(recommitted)

    assert original.to_record()["occurrence_input_order"] == [
        value.to_record() for value in original.occurrences
    ]
    assert original_receipt.occurrence_input_order == original.occurrences
    assert original_receipt.provider_slot_input_order == original.provider_slot_ids
    assert original.input_sha256 != reordered.input_sha256
    assert original_receipt.receipt_sha256 != reordered_receipt.receipt_sha256
    assert original_receipt.ordinal_permutation != reseeded_receipt.ordinal_permutation
    assert original_receipt.ordinal_permutation != (
        recommitted_receipt.ordinal_permutation
    )


def test_invalid_counts_repeats_namespaces_and_canonical_ids_fail() -> None:
    valid = _assignment_input()
    with pytest.raises(ValueError, match="counts must match"):
        replace(valid, provider_slot_ids=valid.provider_slot_ids[:-1])
    with pytest.raises(ValueError, match="occurrence IDs cannot repeat"):
        replace(
            valid,
            occurrences=(valid.occurrences[0], valid.occurrences[0]),
            provider_slot_ids=valid.provider_slot_ids[:2],
        )
    with pytest.raises(ValueError, match="slot IDs cannot repeat"):
        replace(
            valid,
            provider_slot_ids=(
                valid.provider_slot_ids[0],
                valid.provider_slot_ids[0],
                *valid.provider_slot_ids[2:],
            ),
        )
    with pytest.raises(ValueError, match="namespaces must be disjoint"):
        TreatmentAssignmentInput(
            experiment_commitment_sha256=_sha("namespace-collision"),
            public_seed_material="public.seed",
            occurrences=(
                TreatmentOccurrence(
                    TreatmentOccurrenceId("shared.identifier"),
                    TreatmentId("treatment.unique"),
                ),
            ),
            provider_slot_ids=(OpaqueProviderSlotId("shared.identifier"),),
        )
    with pytest.raises(ValueError, match="canonical lowercase"):
        TreatmentId("Airfoil.Control")
    with pytest.raises(ValueError, match="canonical lowercase"):
        TreatmentOccurrenceId("occurrence with spaces")
    with pytest.raises(ValueError, match="request_identity_sha256"):
        TreatmentOccurrence(
            TreatmentOccurrenceId("occurrence.valid"),
            TreatmentId("treatment.valid"),
            request_identity_sha256="not-a-sha",
        )
    with pytest.raises(TypeError, match="exact TreatmentOccurrenceId"):
        TreatmentOccurrence(  # type: ignore[arg-type]
            TreatmentId("treatment.wrong.nominal.type"),
            TreatmentId("treatment.valid"),
        )


def test_internal_and_replay_validation_reject_tampering() -> None:
    assignment_input = _assignment_input()
    receipt = assign_treatment_occurrences(assignment_input)
    with pytest.raises(ValueError, match="inverse disagree"):
        replace(
            receipt,
            inverse_ordinal_permutation=tuple(
                reversed(receipt.inverse_ordinal_permutation)
            ),
        )
    with pytest.raises(ValueError, match="exact inverse"):
        replace(
            receipt,
            occurrence_to_slot=(
                receipt.occurrence_to_slot[1],
                receipt.occurrence_to_slot[0],
                *receipt.occurrence_to_slot[2:],
            ),
        )

    rotated = receipt.ordinal_permutation[1:] + receipt.ordinal_permutation[:1]
    internally_valid_but_wrong = _receipt_with_permutation(receipt, rotated)
    assert internally_valid_but_wrong.receipt_sha256 != receipt.receipt_sha256
    with pytest.raises(TreatmentAssignmentVerificationError, match="replay"):
        verify_treatment_assignment_receipt(
            assignment_input,
            internally_valid_but_wrong,
        )

    tampered_record = deepcopy(receipt.to_record())
    tampered_record["slot_to_occurrence"][0]["treatment_id"] = (
        "treatment.tampered"
    )
    with pytest.raises(TreatmentAssignmentVerificationError, match="replay"):
        verify_treatment_assignment_receipt(assignment_input, tampered_record)


def test_receipt_claims_label_and_order_only_never_content_blinding() -> None:
    receipt = assign_treatment_occurrences(_assignment_input())
    record = receipt.to_record()

    assert receipt.blinding_scope is (
        TreatmentAssignmentBlindingScope.LABEL_AND_ORDER_ONLY
    )
    assert receipt.content_blinding_claimed is False
    assert record["prospective"] is True
    assert record["blinding_claim"] == {
        "scope": "label_and_order_only",
        "opaque_provider_slot_ids": True,
        "content_blinding_claimed": False,
    }
