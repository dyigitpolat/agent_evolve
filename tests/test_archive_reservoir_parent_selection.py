from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest

from agent_evolve.agentic import (
    ARCHIVE_RESERVOIR_PARENT_POLICY_DEFINITION_SHA256,
    ARCHIVE_RESERVOIR_PARENT_POLICY_ID,
    ARCHIVE_RESERVOIR_PARENT_POLICY_VERSION,
    ArchiveReservoirCrowdingKind,
    ArchiveReservoirParentSelection,
    ArchiveReservoirParentSelectionReceipt,
    ArchiveReservoirParentSelector,
    ArchiveReservoirRankedCandidate,
    TaskKeyedArchiveReservoirParentPolicy,
    validate_archive_reservoir_parent_selection,
)
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import (
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.archive_elite import (
    RESERVOIR_POLICY_DEFINITION_SHA256,
    RESERVOIR_POLICY_ID,
    RESERVOIR_POLICY_VERSION,
)


_SCALAR_OBJECTIVES = (ObjectiveSpec("score", "max"),)
_TRADEOFF_OBJECTIVES = (
    ObjectiveSpec("benefit", "max"),
    ObjectiveSpec("burden", "min"),
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate(
    ids: DeterministicIdFactory,
    sequence: int,
    objective_values: tuple[tuple[str, float], ...],
    *,
    configuration_token: int | None = None,
    valid: bool = True,
) -> EvolutionCandidate:
    token = sequence if configuration_token is None else configuration_token
    configuration = freeze_json({"choice": token})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=ids.new_candidate_id(),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                canonical_typed_json_bytes(configuration)
            ).hexdigest(),
            proposal_sequence=sequence,
        ),
        configuration=configuration,
        objectives=objective_values if valid else (),
        valid=valid,
        generation=sequence,
        label=f"candidate-{sequence}",
        failure_message=None if valid else "invalid fixture candidate",
    )


def _state(
    objectives: tuple[ObjectiveSpec, ...],
    rows: tuple[tuple[tuple[str, float], ...], ...],
    *,
    namespace: str,
) -> OptimizerState:
    ids = DeterministicIdFactory(namespace)
    candidates = tuple(
        _candidate(ids, index, row) for index, row in enumerate(rows, start=1)
    )
    archive = ParetoArchive(objectives)
    for candidate in candidates:
        archive.consider(candidate)
    snapshot = archive.snapshot()
    return OptimizerState(
        generation=len(candidates),
        candidates=candidates,
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=len(candidates),
        logical_llm_calls=0,
    )


def _scalar_state(values: tuple[float, ...], *, namespace: str) -> OptimizerState:
    return _state(
        _SCALAR_OBJECTIVES,
        tuple((("score", value),) for value in values),
        namespace=namespace,
    )


def test_singleton_history_caps_two_parent_request_at_one_real_parent() -> None:
    state = _scalar_state((7.0,), namespace="reservoir_singleton")

    selection = TaskKeyedArchiveReservoirParentPolicy().select(
        state,
        task_sha256=_sha("singleton-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=8,
        parent_count=2,
        rotation_index=99,
    )

    assert len(state.archive.front_candidates) == 1
    assert selection.parents == state.candidates
    assert selection.receipt.requested_parent_count == 2
    assert selection.receipt.returned_parent_count == 1
    assert selection.receipt.selected_ordinals == (0,)
    assert selection.receipt.reservoir == state.archive.front_references
    assert len(selection.receipt.receipt_sha256) == 64
    validate_archive_reservoir_parent_selection(state, selection)


def test_scalar_singleton_front_uses_ranked_history_reservoir_without_duplicates() -> (
    None
):
    state = _scalar_state((1.0, 4.0, 2.0, 3.0), namespace="reservoir_scalar")
    policy = TaskKeyedArchiveReservoirParentPolicy()

    selection = policy.select(
        state,
        task_sha256=_sha("scalar-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=3,
        parent_count=9,
    )
    repeated = TaskKeyedArchiveReservoirParentPolicy().select(
        state,
        task_sha256=_sha("scalar-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=3,
        parent_count=9,
    )

    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in state.candidates
    }
    ranked_labels = [
        candidates_by_id[member.reference.candidate_id].label
        for member in selection.receipt.eligible_ranking
    ]
    assert len(state.archive.front_candidates) == 1
    assert ranked_labels == ["candidate-2", "candidate-4", "candidate-3", "candidate-1"]
    assert [
        member.nondomination_rank for member in selection.receipt.eligible_ranking
    ] == [1, 2, 3, 4]
    assert all(
        member.crowding_kind is ArchiveReservoirCrowdingKind.NOT_APPLICABLE
        for member in selection.receipt.eligible_ranking
    )
    assert selection.receipt.returned_parent_count == 3
    assert len({parent.candidate_id for parent in selection.parents}) == 3
    assert {parent.candidate_id for parent in selection.parents} == {
        member.reference.candidate_id
        for member in selection.receipt.eligible_ranking[:3]
    }
    assert repeated.parents == selection.parents
    assert repeated.receipt.receipt_sha256 == selection.receipt.receipt_sha256


def test_multiobjective_reservoir_orders_rank_then_crowding_quality_and_recency() -> (
    None
):
    rows = tuple(
        (("benefit", value), ("burden", value)) for value in (0.0, 4.0, 5.0, 6.0, 10.0)
    )
    state = _state(
        _TRADEOFF_OBJECTIVES,
        rows,
        namespace="reservoir_tradeoff",
    )

    selection = TaskKeyedArchiveReservoirParentPolicy().select(
        state,
        task_sha256=_sha("tradeoff-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=3,
        parent_count=3,
    )

    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in state.candidates
    }
    ranked_labels = [
        candidates_by_id[member.reference.candidate_id].label
        for member in selection.receipt.eligible_ranking
    ]
    assert len(state.archive.front_candidates) == 5
    assert ranked_labels == [
        "candidate-5",
        "candidate-1",
        "candidate-4",
        "candidate-2",
        "candidate-3",
    ]
    assert [member.crowding_kind for member in selection.receipt.eligible_ranking] == [
        ArchiveReservoirCrowdingKind.BOUNDARY,
        ArchiveReservoirCrowdingKind.BOUNDARY,
        ArchiveReservoirCrowdingKind.FINITE,
        ArchiveReservoirCrowdingKind.FINITE,
        ArchiveReservoirCrowdingKind.FINITE,
    ]
    assert selection.receipt.reservoir == tuple(
        member.reference for member in selection.receipt.eligible_ranking[:3]
    )


def test_archive_rejected_invalid_and_duplicate_occurrences_are_not_eligible() -> None:
    ids = DeterministicIdFactory("reservoir_eligibility")
    accepted = _candidate(ids, 1, (("score", 1.0),))
    duplicate = _candidate(
        ids,
        2,
        (("score", 100.0),),
        configuration_token=1,
    )
    invalid = _candidate(ids, 3, (), valid=False)
    stronger = _candidate(ids, 4, (("score", 2.0),))
    candidates = (accepted, duplicate, invalid, stronger)
    archive = ParetoArchive(_SCALAR_OBJECTIVES)
    for candidate in candidates:
        archive.consider(candidate)
    snapshot = archive.snapshot()
    state = OptimizerState(
        generation=4,
        candidates=candidates,
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=3,
        logical_llm_calls=0,
    )

    selection = TaskKeyedArchiveReservoirParentPolicy().select(
        state,
        task_sha256=_sha("eligibility-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=8,
        parent_count=8,
    )

    assert tuple(
        member.reference.candidate_id for member in selection.receipt.eligible_ranking
    ) == (stronger.candidate_id, accepted.candidate_id)
    assert {parent.candidate_id for parent in selection.parents} == {
        stronger.candidate_id,
        accepted.candidate_id,
    }


def test_stale_foreign_and_tampered_reservoir_choices_fail_closed() -> None:
    earlier = _scalar_state((1.0, 2.0), namespace="reservoir_stale")
    later = _scalar_state((1.0, 2.0, 3.0), namespace="reservoir_stale")
    policy = TaskKeyedArchiveReservoirParentPolicy()
    selection = policy.select(
        earlier,
        task_sha256=_sha("stale-task"),
        expected_archive_snapshot_hash=earlier.archive_snapshot_hash,
        reservoir_limit=3,
        parent_count=2,
    )

    with pytest.raises(ValueError, match="expected archive snapshot is stale"):
        policy.select(
            later,
            task_sha256=_sha("stale-task"),
            expected_archive_snapshot_hash=earlier.archive_snapshot_hash,
            reservoir_limit=3,
            parent_count=2,
        )
    with pytest.raises(ValueError, match="stale|complete current eligible ranking"):
        validate_archive_reservoir_parent_selection(later, selection)

    ids = DeterministicIdFactory("reservoir_foreign")
    foreign = _candidate(ids, 1, (("score", 999.0),))
    foreign_state = replace(
        earlier,
        candidates=(foreign, earlier.candidates[1]),
    )
    with pytest.raises(ValueError, match="history differs|foreign candidate"):
        policy.select(
            foreign_state,
            task_sha256=_sha("foreign-task"),
            expected_archive_snapshot_hash=foreign_state.archive_snapshot_hash,
            reservoir_limit=2,
        )

    object.__setattr__(
        selection.receipt.eligible_ranking[0],
        "nondomination_rank",
        91,
    )
    with pytest.raises(ValueError, match="policy order|non-contiguous|sha256"):
        selection.receipt.revalidate()


def test_reservoir_exact_type_and_policy_identity_boundaries_reject_subclasses() -> (
    None
):
    state = _scalar_state((1.0, 2.0), namespace="reservoir_types")

    class ForeignPolicy(TaskKeyedArchiveReservoirParentPolicy):
        pass

    with pytest.raises(TypeError, match="exact TaskKeyedArchiveReservoir"):
        ForeignPolicy().select(
            state,
            task_sha256=_sha("type-task"),
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            reservoir_limit=2,
        )

    policy = TaskKeyedArchiveReservoirParentPolicy()
    object.__setattr__(policy, "policy_version", 999)
    with pytest.raises(ValueError, match="foreign identity"):
        policy.to_record()

    selection = TaskKeyedArchiveReservoirParentPolicy().select(
        state,
        task_sha256=_sha("receipt-type-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        reservoir_limit=2,
    )

    class ForeignReceipt(ArchiveReservoirParentSelectionReceipt):
        pass

    receipt_arguments = {
        item.name: getattr(selection.receipt, item.name)
        for item in fields(ArchiveReservoirParentSelectionReceipt)
        if item.init
    }
    foreign_receipt = ForeignReceipt(**receipt_arguments)
    with pytest.raises(TypeError, match="exact ArchiveReservoirParentSelectionReceipt"):
        foreign_receipt.revalidate()
    with pytest.raises(TypeError, match="exact archive reservoir receipt"):
        ArchiveReservoirParentSelection(
            parents=selection.parents,
            receipt=foreign_receipt,
        )

    class ForeignRankedCandidate(ArchiveReservoirRankedCandidate):
        pass

    member_arguments = {
        item.name: getattr(selection.receipt.eligible_ranking[0], item.name)
        for item in fields(ArchiveReservoirRankedCandidate)
        if item.init
    }
    foreign_member = ForeignRankedCandidate(**member_arguments)
    object.__setattr__(
        selection.receipt,
        "eligible_ranking",
        (foreign_member, *selection.receipt.eligible_ranking[1:]),
    )
    with pytest.raises(TypeError, match="exact ranked candidates"):
        selection.receipt.revalidate()


def test_public_reservoir_identity_and_inverted_api_are_stable() -> None:
    policy: ArchiveReservoirParentSelector = TaskKeyedArchiveReservoirParentPolicy()

    assert policy.to_record() == {
        "policy_id": RESERVOIR_POLICY_ID,
        "policy_version": RESERVOIR_POLICY_VERSION,
        "definition_sha256": RESERVOIR_POLICY_DEFINITION_SHA256,
    }
    assert ARCHIVE_RESERVOIR_PARENT_POLICY_ID == RESERVOIR_POLICY_ID
    assert ARCHIVE_RESERVOIR_PARENT_POLICY_VERSION == RESERVOIR_POLICY_VERSION
    assert (
        ARCHIVE_RESERVOIR_PARENT_POLICY_DEFINITION_SHA256
        == RESERVOIR_POLICY_DEFINITION_SHA256
    )
