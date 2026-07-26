from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
    ParetoDecisionAction,
    ParetoDecisionReason,
    pareto_candidate_hash,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory


OBJECTIVES = (
    ObjectiveSpec("quality", "max"),
    ObjectiveSpec("cost", "min"),
)


def _candidate(
    ids: DeterministicIdFactory,
    sequence: int,
    configuration: dict[str, object],
    *,
    quality: float = 0.0,
    cost: float = 0.0,
    objectives: tuple[tuple[str, float], ...] | None = None,
    valid: bool = True,
    operator_compliant: bool = True,
    evidence_compliant: bool = True,
) -> EvolutionCandidate:
    frozen = freeze_json(configuration)
    canonical = canonical_typed_json_bytes(frozen)
    occurrence = CandidateOccurrence(
        candidate_id=ids.new_candidate_id(),
        configuration_hash=typed_json_sha256(frozen),
        configuration_artifact_hash=hashlib.sha256(canonical).hexdigest(),
        proposal_sequence=sequence,
    )
    if objectives is None:
        objectives = (
            (("quality", quality), ("cost", cost)) if valid else ()
        )
    return EvolutionCandidate(
        occurrence=occurrence,
        configuration=frozen,
        objectives=objectives,
        valid=valid,
        generation=sequence,
        label=f"candidate-{sequence}",
        failure_message=None if valid else "structural validation failed",
        operator_compliant=operator_compliant,
        operator_failure=(
            None if operator_compliant else "mutation escaped its declared path"
        ),
        evidence_compliant=evidence_compliant,
        evidence_failure=(
            None if evidence_compliant else "source attribution was unsupported"
        ),
    )


def test_gate_rejection_reports_every_failed_gate_and_does_not_claim_config() -> None:
    ids = DeterministicIdFactory("pareto_gate")
    archive = ParetoArchive(OBJECTIVES)
    rejected = _candidate(
        ids,
        1,
        {"design": "same"},
        valid=False,
        operator_compliant=False,
        evidence_compliant=False,
    )

    decision, = archive.consider(rejected)

    assert decision.action is ParetoDecisionAction.REJECTED
    assert decision.reasons == (
        ParetoDecisionReason.REJECTED_INVALID,
        ParetoDecisionReason.REJECTED_OPERATOR_NONCOMPLIANT,
        ParetoDecisionReason.REJECTED_EVIDENCE_NONCOMPLIANT,
    )
    assert tuple(reason for reason, _ in decision.failure_details) == decision.reasons
    assert decision.candidate.candidate_hash == pareto_candidate_hash(rejected)
    assert decision.candidate.configuration_hash == (
        rejected.occurrence.configuration_hash
    )

    # A rejected workflow occurrence must not prevent a later compliant occurrence
    # of the same configuration from entering the scientific archive.
    accepted = _candidate(
        ids,
        2,
        {"design": "same"},
        quality=4.0,
        cost=2.0,
    )
    admitted, = archive.consider(accepted)
    assert admitted.action is ParetoDecisionAction.ADMITTED
    assert archive.front == (accepted,)
    assert archive.snapshot().eligible_configuration_count == 1


def test_invalid_objective_vector_is_an_explicit_rejection() -> None:
    ids = DeterministicIdFactory("pareto_objectives")
    archive = ParetoArchive(OBJECTIVES)
    candidate = _candidate(
        ids,
        1,
        {"design": "missing-cost"},
        objectives=(("quality", 2.0),),
    )

    decision, = archive.consider(candidate)

    assert decision.action is ParetoDecisionAction.REJECTED
    assert decision.reasons == (
        ParetoDecisionReason.REJECTED_OBJECTIVE_CONTRACT,
    )
    assert "missing: cost" in decision.failure_details[0][1]
    assert not archive.front


def test_configuration_deduplication_persists_after_dominated_rejection() -> None:
    ids = DeterministicIdFactory("pareto_dedup")
    archive = ParetoArchive(OBJECTIVES)
    strong = _candidate(
        ids,
        1,
        {"design": "strong"},
        quality=10.0,
        cost=1.0,
    )
    weak = _candidate(
        ids,
        2,
        {"design": "weak"},
        quality=1.0,
        cost=10.0,
    )
    contradictory_duplicate = _candidate(
        ids,
        3,
        {"design": "weak"},
        quality=100.0,
        cost=0.0,
    )

    archive.consider(strong)
    dominated, = archive.consider(weak)
    duplicate, = archive.consider(contradictory_duplicate)

    assert dominated.reasons == (ParetoDecisionReason.REJECTED_DOMINATED,)
    assert dominated.dominators == (archive.snapshot().front_references[0],)
    assert duplicate.reasons == (
        ParetoDecisionReason.REJECTED_DUPLICATE_CONFIGURATION,
    )
    assert duplicate.duplicate_of is not None
    assert duplicate.duplicate_of.candidate_id == weak.candidate_id
    assert archive.front == (strong,)
    assert archive.snapshot().eligible_configuration_count == 2


def test_admission_reports_every_dominated_removal_as_an_atomic_decision() -> None:
    ids = DeterministicIdFactory("pareto_removal")
    archive = ParetoArchive(OBJECTIVES)
    quality_specialist = _candidate(
        ids,
        1,
        {"design": "quality"},
        quality=6.0,
        cost=6.0,
    )
    cost_specialist = _candidate(
        ids,
        2,
        {"design": "cost"},
        quality=5.0,
        cost=5.0,
    )
    dominator = _candidate(
        ids,
        3,
        {"design": "both"},
        quality=7.0,
        cost=4.0,
    )
    archive.consider(quality_specialist)
    archive.consider(cost_specialist)

    update = archive.consider(dominator)

    assert [item.action for item in update] == [
        ParetoDecisionAction.ADMITTED,
        ParetoDecisionAction.REMOVED,
        ParetoDecisionAction.REMOVED,
    ]
    admission = update[0]
    assert admission.reasons == (
        ParetoDecisionReason.ADMITTED_NONDOMINATED,
    )
    assert {item.candidate_id for item in admission.removed_candidates} == {
        quality_specialist.candidate_id,
        cost_specialist.candidate_id,
    }
    for removal in update[1:]:
        assert removal.reasons == (ParetoDecisionReason.REMOVED_DOMINATED,)
        assert removal.caused_by == admission.candidate
        assert removal.front_after == (admission.candidate,)
    assert archive.front == (dominator,)
    assert [item.decision_sequence for item in archive.decisions] == [1, 2, 3, 4, 5]


def test_exact_objective_ties_have_arrival_independent_representative() -> None:
    ids = DeterministicIdFactory("pareto_tie")
    first = _candidate(
        ids,
        1,
        {"design": "alpha"},
        quality=8.0,
        cost=3.0,
    )
    second = _candidate(
        ids,
        2,
        {"design": "beta"},
        quality=8.0,
        cost=3.0,
    )
    winner, loser = sorted(
        (first, second),
        key=lambda item: (
            item.occurrence.configuration_hash,
            item.candidate_id.value,
            pareto_candidate_hash(item),
        ),
    )

    loser_first = ParetoArchive(OBJECTIVES)
    loser_first.consider(loser)
    replacement = loser_first.consider(winner)
    winner_first = ParetoArchive(OBJECTIVES)
    winner_first.consider(winner)
    tied_rejection, = winner_first.consider(loser)

    assert loser_first.front == winner_first.front == (winner,)
    assert replacement[0].reasons == (
        ParetoDecisionReason.ADMITTED_TIE_BREAK_REPLACEMENT,
    )
    assert replacement[1].reasons == (
        ParetoDecisionReason.REMOVED_TIE_BREAK,
    )
    assert tied_rejection.reasons == (
        ParetoDecisionReason.REJECTED_OBJECTIVE_TIE,
    )
    assert tied_rejection.tie_with[0].candidate_id == winner.candidate_id


def test_snapshot_and_decisions_are_immutable_json_safe_views() -> None:
    ids = DeterministicIdFactory("pareto_snapshot")
    archive = ParetoArchive(OBJECTIVES)
    candidate = _candidate(
        ids,
        1,
        {"design": "traceable"},
        quality=3.0,
        cost=7.0,
    )
    decision, = archive.consider(candidate)
    snapshot = archive.snapshot()

    assert archive.decisions == snapshot.decisions == (decision,)
    assert snapshot.front_candidates == (candidate,)
    assert snapshot.front_references == (decision.candidate,)
    assert json.loads(json.dumps(decision.to_trace_record()))["candidate_hash"] == (
        pareto_candidate_hash(candidate)
    )
    assert json.loads(json.dumps(snapshot.to_trace_record()))["front_size"] == 1
    with pytest.raises(FrozenInstanceError):
        snapshot.consideration_count = 99  # type: ignore[misc]


def test_record_only_evidence_policy_keeps_annotation_errors_out_of_quality_gate() -> None:
    ids = DeterministicIdFactory("pareto_record_only_evidence")
    candidate = _candidate(
        ids,
        1,
        {"design": "valid-objectives-bad-explanation"},
        quality=4.0,
        cost=2.0,
        evidence_compliant=False,
    )
    strict = ParetoArchive(OBJECTIVES)
    strict_decision, = strict.consider(candidate)
    assert strict_decision.reasons == (
        ParetoDecisionReason.REJECTED_EVIDENCE_NONCOMPLIANT,
    )

    record_only = ParetoArchive(
        OBJECTIVES,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    admitted, = record_only.consider(candidate)
    assert admitted.action is ParetoDecisionAction.ADMITTED
    assert record_only.front == (candidate,)
    snapshot = record_only.snapshot()
    assert snapshot.evidence_admission_policy is EvidenceAdmissionPolicy.RECORD_ONLY
    assert snapshot.to_trace_record()["evidence_admission_policy"] == "record_only"
