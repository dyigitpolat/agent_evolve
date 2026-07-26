from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace

import pytest

from agent_evolve.agentic import (
    ARCHIVE_ELITE_PARENT_POLICY_DEFINITION_SHA256,
    ARCHIVE_ELITE_PARENT_POLICY_ID,
    ARCHIVE_ELITE_PARENT_POLICY_VERSION,
    ArchiveEliteParentSelection,
    ArchiveEliteParentSelectionReceipt,
    ArchiveEliteParentSelector,
    TaskKeyedArchiveEliteParentPolicy,
    validate_archive_elite_parent_selection,
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
    POLICY_DEFINITION_SHA256,
    POLICY_ID,
    POLICY_VERSION,
)


_OBJECTIVES = (
    ObjectiveSpec("benefit", "max"),
    ObjectiveSpec("burden", "min"),
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate(
    ids: DeterministicIdFactory,
    sequence: int,
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
        # Every row is nondominated: benefit and burden rise together.
        objectives=(
            ("benefit", float(sequence)),
            ("burden", float(sequence)),
        ),
        valid=True,
        generation=0,
        label=f"elite-{sequence}",
    )


def _state(
    cardinality: int,
    *,
    namespace: str = "archive_elite",
    generation: int = 0,
) -> OptimizerState:
    ids = DeterministicIdFactory(namespace)
    candidates = tuple(_candidate(ids, index) for index in range(1, cardinality + 1))
    archive = ParetoArchive(_OBJECTIVES)
    for candidate in candidates:
        archive.consider(candidate)
    snapshot = archive.snapshot()
    return OptimizerState(
        generation=generation,
        candidates=candidates,
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=cardinality,
        logical_llm_calls=0,
    )


def _candidate_subclass(candidate: EvolutionCandidate) -> EvolutionCandidate:
    class ForeignEvolutionCandidate(EvolutionCandidate):
        pass

    constructor = {
        item.name: getattr(candidate, item.name)
        for item in fields(EvolutionCandidate)
        if item.init
    }
    return ForeignEvolutionCandidate(**constructor)


def test_singleton_front_returns_the_only_authenticated_elite() -> None:
    state = _state(1)
    policy = TaskKeyedArchiveEliteParentPolicy()

    selection = policy.select(
        state,
        task_sha256=_sha("singleton-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        parent_count=1,
        rotation_index=37,
    )

    assert selection.parents == state.archive.front_candidates
    assert selection.receipt.eligible_front == state.archive.front_references
    assert selection.receipt.selected_ordinals == (0,)
    assert selection.receipt.rotation_anchor == 0
    assert selection.receipt.archive_snapshot_hash == state.archive_snapshot_hash
    assert selection.receipt.policy_id == POLICY_ID
    assert selection.receipt.policy_version == POLICY_VERSION
    assert selection.receipt.policy_definition_sha256 == POLICY_DEFINITION_SHA256
    assert len(selection.receipt.receipt_sha256) == 64
    assert (
        json.loads(json.dumps(selection.receipt.to_trace_record()))["receipt_sha256"]
        == selection.receipt.receipt_sha256
    )
    validate_archive_elite_parent_selection(state, selection)


def test_multi_front_task_keyed_rotation_is_deterministic_and_supports_k() -> None:
    state = _state(4)
    task_sha256 = _sha("multi-front-task")
    policy = TaskKeyedArchiveEliteParentPolicy()

    singles = [
        policy.select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            rotation_index=index,
        )
        for index in range(4)
    ]
    repeated = TaskKeyedArchiveEliteParentPolicy().select(
        state,
        task_sha256=task_sha256,
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        rotation_index=0,
    )
    pair = policy.select(
        state,
        task_sha256=task_sha256,
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
        parent_count=2,
        rotation_index=1,
    )

    anchor = singles[0].receipt.rotation_anchor
    assert [value.receipt.selected_ordinals[0] for value in singles] == [
        (anchor + index) % 4 for index in range(4)
    ]
    assert {value.parents[0].candidate_id for value in singles} == {
        candidate.candidate_id for candidate in state.archive.front_candidates
    }
    assert repeated.receipt.receipt_sha256 == singles[0].receipt.receipt_sha256
    assert repeated.parents == singles[0].parents
    assert pair.receipt.selected_ordinals == (
        (anchor + 1) % 4,
        (anchor + 2) % 4,
    )
    assert len({parent.candidate_id for parent in pair.parents}) == 2
    assert pair.receipt.eligible_front == state.archive.front_references
    assert len(pair.receipt.to_trace_record()["eligible_front"]) == 4


def test_empty_front_and_invalid_parent_cardinality_fail_closed() -> None:
    state = _state(0)
    policy = TaskKeyedArchiveEliteParentPolicy()

    with pytest.raises(ValueError, match="non-empty front"):
        policy.select(
            state,
            task_sha256=_sha("empty-task"),
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )

    nonempty = _state(2, namespace="archive_elite_count")
    for count in (0, 3):
        with pytest.raises(ValueError, match="parent_count"):
            policy.select(
                nonempty,
                task_sha256=_sha("count-task"),
                expected_archive_snapshot_hash=nonempty.archive_snapshot_hash,
                parent_count=count,
            )


def test_stale_snapshot_and_stale_receipt_are_rejected() -> None:
    earlier = _state(1, namespace="archive_elite_stale", generation=1)
    later = _state(2, namespace="archive_elite_stale", generation=2)
    policy = TaskKeyedArchiveEliteParentPolicy()
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
    with pytest.raises(ValueError, match="stale"):
        validate_archive_elite_parent_selection(later, selection)


def test_foreign_front_candidate_and_tampered_selection_are_rejected() -> None:
    state = _state(3, namespace="archive_elite_tamper")
    foreign = _candidate(DeterministicIdFactory("foreign_elite"), 99)
    damaged_snapshot = replace(
        state.archive,
        front_candidates=(foreign, *state.archive.front_candidates[1:]),
    )
    # The archive hash intentionally excludes full candidate payloads and still
    # verifies here.  The selection boundary must join them back to references.
    damaged_state = replace(state, archive=damaged_snapshot)
    assert damaged_state.archive_snapshot_hash == pareto_archive_snapshot_hash(
        damaged_snapshot
    )

    with pytest.raises(ValueError, match="does not match its reference"):
        TaskKeyedArchiveEliteParentPolicy().select(
            damaged_state,
            task_sha256=_sha("tamper-task"),
            expected_archive_snapshot_hash=damaged_state.archive_snapshot_hash,
        )

    selection = TaskKeyedArchiveEliteParentPolicy().select(
        state,
        task_sha256=_sha("tamper-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )
    object.__setattr__(selection.receipt, "rotation_index", 12)
    with pytest.raises(ValueError, match="selected_ordinals|receipt_sha256"):
        selection.receipt.revalidate()

    type_tampered = TaskKeyedArchiveEliteParentPolicy().select(
        state,
        task_sha256=_sha("tamper-type-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )
    original_ordinal = type_tampered.receipt.selected_ordinals[0]
    object.__setattr__(
        type_tampered.receipt,
        "selected_ordinals",
        (bool(original_ordinal),),
    )
    with pytest.raises(TypeError, match="exact ints"):
        type_tampered.receipt.revalidate()

    hash_tampered = TaskKeyedArchiveEliteParentPolicy().select(
        state,
        task_sha256=_sha("tamper-hash-task"),
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )
    object.__setattr__(hash_tampered.receipt, "receipt_sha256", "0" * 64)
    with pytest.raises(ValueError, match="receipt_sha256 does not verify"):
        hash_tampered.receipt.revalidate()


def test_tampered_policy_identity_fails_closed() -> None:
    state = _state(2, namespace="archive_elite_policy_tamper")
    policy = TaskKeyedArchiveEliteParentPolicy()
    object.__setattr__(policy, "policy_id", "foreign-policy")

    with pytest.raises(ValueError, match="foreign policy identity"):
        policy.to_record()
    with pytest.raises(ValueError, match="foreign policy identity"):
        policy.select(
            state,
            task_sha256=_sha("policy-tamper-task"),
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )


def test_exact_type_boundaries_reject_state_candidate_policy_and_receipt_subclasses() -> (
    None
):
    state = _state(2, namespace="archive_elite_subclass")
    task_sha256 = _sha("subclass-task")

    class ForeignOptimizerState(OptimizerState):
        pass

    foreign_state = ForeignOptimizerState(
        generation=state.generation,
        candidates=state.candidates,
        archive=state.archive,
        archive_snapshot_hash=state.archive_snapshot_hash,
        unique_evaluations=state.unique_evaluations,
        logical_llm_calls=state.logical_llm_calls,
    )
    with pytest.raises(TypeError, match="exact OptimizerState"):
        TaskKeyedArchiveEliteParentPolicy().select(
            foreign_state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )

    damaged_snapshot = replace(
        state.archive,
        front_candidates=(
            _candidate_subclass(state.archive.front_candidates[0]),
            state.archive.front_candidates[1],
        ),
    )
    subclass_candidate_state = replace(state, archive=damaged_snapshot)
    with pytest.raises(TypeError, match="exact EvolutionCandidate"):
        TaskKeyedArchiveEliteParentPolicy().select(
            subclass_candidate_state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )

    class ForeignPolicy(TaskKeyedArchiveEliteParentPolicy):
        pass

    with pytest.raises(TypeError, match="exact TaskKeyed"):
        ForeignPolicy().select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
        )

    selection = TaskKeyedArchiveEliteParentPolicy().select(
        state,
        task_sha256=task_sha256,
        expected_archive_snapshot_hash=state.archive_snapshot_hash,
    )

    class ForeignReceipt(ArchiveEliteParentSelectionReceipt):
        pass

    receipt_arguments = {
        item.name: getattr(selection.receipt, item.name)
        for item in fields(ArchiveEliteParentSelectionReceipt)
        if item.init
    }
    foreign_receipt = ForeignReceipt(**receipt_arguments)
    with pytest.raises(TypeError, match="exact ArchiveEliteParentSelectionReceipt"):
        foreign_receipt.revalidate()
    with pytest.raises(TypeError, match="exact archive elite receipt"):
        ArchiveEliteParentSelection(
            parents=selection.parents,
            receipt=foreign_receipt,
        )


def test_public_policy_identity_and_inverted_selector_seam_are_stable() -> None:
    policy: ArchiveEliteParentSelector = TaskKeyedArchiveEliteParentPolicy()

    assert policy.to_record() == {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "definition_sha256": POLICY_DEFINITION_SHA256,
    }
    assert ARCHIVE_ELITE_PARENT_POLICY_ID == POLICY_ID
    assert ARCHIVE_ELITE_PARENT_POLICY_VERSION == POLICY_VERSION
    assert ARCHIVE_ELITE_PARENT_POLICY_DEFINITION_SHA256 == POLICY_DEFINITION_SHA256
