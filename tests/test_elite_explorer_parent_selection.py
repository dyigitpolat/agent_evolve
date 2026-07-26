from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace

import pytest

import agent_evolve.agentic as agentic
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
from agent_evolve.policies.selection.elite_explorer import (
    ArchiveEliteExplorerParentSelectionReceipt,
    EliteExplorerFallbackReason,
    EliteExplorerLaneId,
    EliteExplorerLaneSource,
    POLICY_DEFINITION_SHA256,
    POLICY_ID,
    POLICY_VERSION,
    ROTATION_LAW_ID,
    TaskKeyedArchiveEliteExplorerParentPolicy,
    validate_archive_elite_explorer_parent_selection,
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
) -> EvolutionCandidate:
    configuration = freeze_json({"choice": sequence})
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
        objectives=objective_values,
        valid=True,
        generation=sequence,
        label=f"candidate-{sequence}",
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


def _tradeoff_state(*, namespace: str) -> OptimizerState:
    # Candidates 1 and 2 form the exact front.  Candidates 3 and 4 form the
    # best dominated rank, while candidate 5 is dominated again at rank 3.
    rows = (
        (("benefit", 10.0), ("burden", 10.0)),
        (("benefit", 8.0), ("burden", 5.0)),
        (("benefit", 9.0), ("burden", 12.0)),
        (("benefit", 7.0), ("burden", 8.0)),
        (("benefit", 6.0), ("burden", 20.0)),
    )
    return _state(_TRADEOFF_OBJECTIVES, rows, namespace=namespace)


def test_explicit_lanes_select_front_elite_and_best_dominated_explorer() -> None:
    state = _tradeoff_state(namespace="elite_explorer_ranked")
    task_sha256 = _sha("ranked-task")
    policy = TaskKeyedArchiveEliteExplorerParentPolicy()

    selection = policy.select(
        state,
        task_sha256=task_sha256,
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        rotation_index=0,
    )
    repeated = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=task_sha256,
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        rotation_index=0,
    )

    elite_lane, explorer_lane = selection.receipt.lanes
    assert elite_lane.lane_id is EliteExplorerLaneId.ELITE
    assert elite_lane.source is EliteExplorerLaneSource.CURRENT_PARETO_FRONT
    assert elite_lane.nondomination_rank == 1
    assert elite_lane.fallback_reason is EliteExplorerFallbackReason.NONE
    assert elite_lane.selected_parent in state.archive.front_references
    assert selection.elite is selection.parent_for(EliteExplorerLaneId.ELITE)

    assert explorer_lane.lane_id is EliteExplorerLaneId.EXPLORER
    assert explorer_lane.source is EliteExplorerLaneSource.BEST_DOMINATED_RANK
    assert explorer_lane.nondomination_rank == 2
    assert explorer_lane.fallback_reason is EliteExplorerFallbackReason.NONE
    assert all(
        member.nondomination_rank == 2
        for member in selection.receipt.eligible_ranking
        if member.reference in explorer_lane.selection_pool
    )
    assert selection.explorer is selection.parent_for(EliteExplorerLaneId.EXPLORER)
    assert selection.elite.candidate_id != selection.explorer.candidate_id

    assert selection.receipt.eligible_front == state.archive.front_references
    assert len(selection.receipt.eligible_ranking) == len(state.candidates)
    assert repeated.parents == selection.parents
    assert repeated.receipt.receipt_sha256 == selection.receipt.receipt_sha256
    validate_archive_elite_explorer_parent_selection(state, selection)

    trace = selection.receipt.to_trace_record()
    assert trace["rotation_law_id"] == ROTATION_LAW_ID
    assert [lane["lane_id"] for lane in trace["lanes"]] == ["elite", "explorer"]
    assert "propensity" not in json.dumps(trace, sort_keys=True)


def test_task_archive_keyed_rotation_covers_each_best_rank_tie() -> None:
    state = _tradeoff_state(namespace="elite_explorer_rotation")
    policy = TaskKeyedArchiveEliteExplorerParentPolicy()
    task_sha256 = _sha("rotation-task")

    selections = tuple(
        policy.select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            rotation_index=index,
        )
        for index in range(2)
    )

    explorer_pool = selections[0].receipt.lanes[1].selection_pool
    assert len(explorer_pool) == 2
    assert {
        selection.receipt.lanes[1].selected_parent for selection in selections
    } == set(explorer_pool)
    assert [
        selection.receipt.lanes[1].selected_ordinal for selection in selections
    ] == [
        (selections[0].receipt.lanes[1].rotation_anchor + index) % 2
        for index in range(2)
    ]


def test_no_dominated_history_uses_a_distinct_exact_front_fallback() -> None:
    rows = tuple((("benefit", value), ("burden", value)) for value in (1.0, 2.0, 3.0))
    state = _state(
        _TRADEOFF_OBJECTIVES,
        rows,
        namespace="elite_explorer_front_fallback",
    )

    selection = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=_sha("front-fallback-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )

    elite_lane, explorer_lane = selection.receipt.lanes
    assert len(state.archive.front_references) == 3
    assert {
        member.nondomination_rank for member in selection.receipt.eligible_ranking
    } == {1}
    assert explorer_lane.source is EliteExplorerLaneSource.DISTINCT_FRONT_FALLBACK
    assert (
        explorer_lane.fallback_reason
        is EliteExplorerFallbackReason.NO_DOMINATED_HISTORY
    )
    assert explorer_lane.selection_pool == tuple(
        reference
        for reference in state.archive.front_references
        if reference != elite_lane.selected_parent
    )
    assert elite_lane.selected_parent != explorer_lane.selected_parent


def test_singleton_reuse_is_explicit_and_preserves_two_stable_lanes() -> None:
    state = _state(
        _SCALAR_OBJECTIVES,
        ((("score", 7.0),),),
        namespace="elite_explorer_singleton",
    )

    selection = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=_sha("singleton-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        rotation_index=71,
    )

    assert len(selection.parents) == 2
    assert selection.parents[0] is selection.parents[1]
    assert tuple(lane.lane_id for lane in selection.receipt.lanes) == (
        EliteExplorerLaneId.ELITE,
        EliteExplorerLaneId.EXPLORER,
    )
    explorer_lane = selection.receipt.lanes[1]
    assert explorer_lane.source is EliteExplorerLaneSource.SINGLETON_REUSE_FALLBACK
    assert explorer_lane.fallback_reason is EliteExplorerFallbackReason.SINGLETON_FRONT
    assert explorer_lane.selection_pool == state.archive.front_references
    validate_archive_elite_explorer_parent_selection(state, selection)


def test_rank_one_objective_tie_never_substitutes_for_exact_front_authority() -> None:
    state = _state(
        _SCALAR_OBJECTIVES,
        ((("score", 7.0),), (("score", 7.0),)),
        namespace="elite_explorer_objective_tie",
    )

    selection = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=_sha("objective-tie-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )

    assert len(state.archive.front_references) == 1
    assert len(selection.receipt.eligible_ranking) == 2
    assert {
        member.nondomination_rank for member in selection.receipt.eligible_ranking
    } == {1}
    assert selection.receipt.lanes[0].selection_pool == state.archive.front_references
    assert (
        selection.receipt.lanes[1].source
        is EliteExplorerLaneSource.SINGLETON_REUSE_FALLBACK
    )
    assert selection.parents == (state.archive.front_candidates[0],) * 2


def test_stale_foreign_and_tampered_receipts_fail_closed() -> None:
    earlier = _tradeoff_state(namespace="elite_explorer_stale")
    later_rows = tuple(candidate.objectives for candidate in earlier.candidates) + (
        (("benefit", 11.0), ("burden", 4.0)),
    )
    later = _state(
        _TRADEOFF_OBJECTIVES,
        later_rows,
        namespace="elite_explorer_stale",
    )
    policy = TaskKeyedArchiveEliteExplorerParentPolicy()
    selection = policy.select(
        earlier,
        task_sha256=_sha("stale-task"),
        expected_archive_snapshot_hash=earlier.archive_snapshot_hash,
    )

    with pytest.raises(ValueError, match="expected archive snapshot is stale"):
        policy.select(
            later,
            task_sha256=_sha("stale-task"),
            expected_archive_snapshot_hash=earlier.archive_snapshot_hash,
        )
    with pytest.raises(ValueError, match="stale|complete current"):
        validate_archive_elite_explorer_parent_selection(later, selection)

    foreign_state = replace(
        earlier,
        candidates=(later.candidates[-1], *earlier.candidates[1:]),
    )
    with pytest.raises(ValueError, match="history differs|foreign|stale candidate"):
        policy.select(
            foreign_state,
            task_sha256=_sha("foreign-task"),
            expected_archive_snapshot_hash=foreign_state.archive_snapshot_hash,
        )

    object.__setattr__(selection.receipt.lanes[1], "selected_ordinal", 99)
    with pytest.raises(ValueError, match="selected_ordinal|selection_pool"):
        selection.receipt.revalidate()


def test_ranking_and_policy_identity_tampering_fail_closed() -> None:
    state = _tradeoff_state(namespace="elite_explorer_tamper")
    selection = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=_sha("ranking-tamper-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )
    object.__setattr__(
        selection.receipt.eligible_ranking[-1],
        "nondomination_rank",
        99,
    )
    with pytest.raises(ValueError, match="policy order|non-contiguous|sha256"):
        selection.receipt.revalidate()

    policy = TaskKeyedArchiveEliteExplorerParentPolicy()
    object.__setattr__(policy, "policy_id", "foreign-policy")
    with pytest.raises(ValueError, match="foreign identity"):
        policy.to_record()
    with pytest.raises(ValueError, match="foreign identity"):
        policy.select(
            state,
            task_sha256=_sha("foreign-policy-task"),
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )


def test_receipt_subclasses_and_foreign_parent_payloads_are_rejected() -> None:
    state = _tradeoff_state(namespace="elite_explorer_types")
    selection = TaskKeyedArchiveEliteExplorerParentPolicy().select(
        state,
        task_sha256=_sha("type-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )

    class ForeignReceipt(ArchiveEliteExplorerParentSelectionReceipt):
        pass

    receipt_arguments = {
        item.name: getattr(selection.receipt, item.name)
        for item in fields(ArchiveEliteExplorerParentSelectionReceipt)
        if item.init
    }
    foreign_receipt = ForeignReceipt(**receipt_arguments)
    with pytest.raises(TypeError, match="exact ArchiveEliteExplorer"):
        foreign_receipt.revalidate()

    object.__setattr__(selection, "parents", tuple(reversed(selection.parents)))
    with pytest.raises(ValueError, match="lane receipts"):
        selection.revalidate()


def test_policy_record_exposes_stable_identity_and_lane_contract() -> None:
    record = TaskKeyedArchiveEliteExplorerParentPolicy().to_record()

    assert record == {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "definition_sha256": POLICY_DEFINITION_SHA256,
        "lane_ids": ["elite", "explorer"],
        "rotation_law_id": ROTATION_LAW_ID,
    }
    assert (
        agentic.TaskKeyedArchiveEliteExplorerParentPolicy
        is TaskKeyedArchiveEliteExplorerParentPolicy
    )
    assert agentic.ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_ID == POLICY_ID
