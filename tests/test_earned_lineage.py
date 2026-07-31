"""Workload- and model-neutral contracts for earned challenger lineage."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.earned_lineage import (
    CandidateConservedCredit,
    CandidateProposalProvenance,
    ConservedBackboneCeilingReceipt,
    ConservedStageCreditReceipt,
    EarnedLineageLedger,
    EarnedLineagePolicy,
    ProposalLineageRole,
    allocate_earned_parents,
)
from agent_evolve.domain.ids import CandidateId


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _provenance(
    label: str,
    *,
    generation: int,
    role: ProposalLineageRole,
    parents: tuple[CandidateId, ...] = (),
) -> CandidateProposalProvenance:
    return CandidateProposalProvenance(
        candidate_id=CandidateId(f"candidate_{label}"),
        configuration_sha256=_sha(f"configuration:{label}"),
        generation=generation,
        source_role=role,
        proposal_expert_id=f"expert.{label}",
        proposal_expert_version=1,
        proposal_expert_definition_sha256=_sha(f"expert:{label}"),
        operator_id="semantic_mutation",
        parent_candidate_ids=parents,
        decision_cutoff_sha256=_sha(f"cutoff:{label}"),
        source_receipt_sha256=_sha(f"source:{label}"),
    )


def _credit_receipt(
    *,
    generation: int,
    candidate_id: CandidateId,
    gain: float,
) -> ConservedStageCreditReceipt:
    return ConservedStageCreditReceipt(
        generation=generation,
        utility_id="archive_hypervolume",
        utility_version=1,
        utility_definition_sha256=_sha("utility"),
        pre_archive_sha256=_sha(f"pre:{generation}"),
        post_archive_sha256=_sha(f"post:{generation}"),
        pre_utility=1.0,
        post_utility=1.0 + gain,
        contribution_policy_id="exclusive_hypervolume",
        contribution_policy_version=1,
        contribution_policy_definition_sha256=_sha("credit-policy"),
        candidate_credits=(
            CandidateConservedCredit(
                candidate_id=candidate_id,
                contribution=gain,
                admitted_to_archive=gain > 0.0,
                outcome_receipt_sha256=_sha(
                    f"outcome:{generation}:{candidate_id.value}"
                ),
            ),
        ),
    )


def test_positive_challenger_credit_earns_one_transactional_parent_exposure() -> None:
    backbone = _provenance(
        "backbone",
        generation=0,
        role=ProposalLineageRole.BACKBONE,
    )
    challenger = _provenance(
        "challenger",
        generation=1,
        role=ProposalLineageRole.CHALLENGER,
        parents=(backbone.candidate_id,),
    )
    displaced = CandidateId("candidate_displaced")
    ledger = EarnedLineageLedger()
    ledger.register((backbone, challenger))

    issuance = ledger.observe(
        _credit_receipt(
            generation=1,
            candidate_id=challenger.candidate_id,
            gain=0.25,
        )
    )
    assert tuple(value.candidate_id for value in issuance.tickets) == (
        challenger.candidate_id,
    )

    reservation = ledger.prepare(
        generation=2,
        available_candidate_ids=(backbone.candidate_id, challenger.candidate_id),
        maximum_tickets=1,
    )
    allocation = allocate_earned_parents(
        generation=2,
        base_parent_ids=(backbone.candidate_id, displaced),
        reservation=reservation,
    )
    assert allocation.selected_parent_ids == (
        backbone.candidate_id,
        challenger.candidate_id,
    )

    ledger.abort(reservation)
    retry = ledger.prepare(
        generation=2,
        available_candidate_ids=(backbone.candidate_id, challenger.candidate_id),
        maximum_tickets=1,
    )
    assert retry == reservation
    commit = ledger.commit(
        retry,
        parent_exposure_receipt_sha256=_sha("parent-exposure"),
    )
    assert commit.consumed_ticket_sha256s == (
        issuance.tickets[0].ticket_sha256,
    )

    empty = ledger.prepare(
        generation=2,
        available_candidate_ids=(backbone.candidate_id, challenger.candidate_id),
        maximum_tickets=1,
    )
    assert empty.tickets == ()
    ledger.abort(empty)
    with pytest.raises(ValueError, match="nothing to commit"):
        ledger.commit(empty, parent_exposure_receipt_sha256=_sha("unused"))


def test_backbone_or_zero_credit_never_manufactures_a_ticket() -> None:
    backbone = _provenance(
        "anchor",
        generation=0,
        role=ProposalLineageRole.BACKBONE,
    )
    challenger = _provenance(
        "zero",
        generation=1,
        role=ProposalLineageRole.CHALLENGER,
        parents=(backbone.candidate_id,),
    )
    ledger = EarnedLineageLedger()
    ledger.register((backbone, challenger))

    assert ledger.observe(
        _credit_receipt(
            generation=1,
            candidate_id=challenger.candidate_id,
            gain=0.0,
        )
    ).tickets == ()
    assert ledger.observe(
        _credit_receipt(
            generation=2,
            candidate_id=backbone.candidate_id,
            gain=0.125,
        )
    ).tickets == ()


def test_one_generation_credit_receipt_can_rank_multiple_challengers() -> None:
    challengers = tuple(
        _provenance(
            label,
            generation=1,
            role=ProposalLineageRole.CHALLENGER,
        )
        for label in ("batch_a", "batch_b", "batch_c")
    )
    ledger = EarnedLineageLedger(
        policy=EarnedLineagePolicy(max_tickets_per_stage=2)
    )
    ledger.register(challengers)
    contributions = {
        challengers[0].candidate_id: 0.125,
        challengers[1].candidate_id: 0.0,
        challengers[2].candidate_id: 0.25,
    }
    receipt = ConservedStageCreditReceipt(
        generation=1,
        utility_id="archive_hypervolume",
        utility_version=1,
        utility_definition_sha256=_sha("utility"),
        pre_archive_sha256=_sha("batch-pre"),
        post_archive_sha256=_sha("batch-post"),
        pre_utility=1.0,
        post_utility=1.375,
        contribution_policy_id="sequential_exact_complement",
        contribution_policy_version=1,
        contribution_policy_definition_sha256=_sha("batch-credit-policy"),
        candidate_credits=tuple(
            CandidateConservedCredit(
                candidate_id=candidate_id,
                contribution=contributions[candidate_id],
                admitted_to_archive=contributions[candidate_id] > 0.0,
                outcome_receipt_sha256=_sha(
                    f"batch-outcome:{candidate_id.value}"
                ),
            )
            for candidate_id in sorted(
                contributions,
                key=lambda value: value.value,
            )
        ),
    )

    issuance = ledger.observe(receipt)

    assert tuple(value.candidate_id for value in issuance.tickets) == (
        challengers[2].candidate_id,
        challengers[0].candidate_id,
    )
    assert ledger.to_record(current_generation=1)["stage_credits"] == [
        receipt.to_record()
    ]


def test_expired_unconsumed_ticket_does_not_block_new_evidence() -> None:
    challenger = _provenance(
        "renewable",
        generation=0,
        role=ProposalLineageRole.CHALLENGER,
    )
    ledger = EarnedLineageLedger(
        policy=EarnedLineagePolicy(ticket_ttl_generations=2)
    )
    ledger.register((challenger,))

    first = ledger.observe(
        _credit_receipt(
            generation=1,
            candidate_id=challenger.candidate_id,
            gain=0.25,
        )
    )
    second = ledger.observe(
        _credit_receipt(
            generation=4,
            candidate_id=challenger.candidate_id,
            gain=0.125,
        )
    )
    assert len(first.tickets) == len(second.tickets) == 1
    assert first.tickets[0].ticket_sha256 != second.tickets[0].ticket_sha256


def test_provenance_rejects_same_generation_parent_cycles() -> None:
    left_id = CandidateId("candidate_cycle_left")
    right_id = CandidateId("candidate_cycle_right")
    left = _provenance(
        "cycle_left",
        generation=1,
        role=ProposalLineageRole.CHALLENGER,
        parents=(right_id,),
    )
    right = _provenance(
        "cycle_right",
        generation=1,
        role=ProposalLineageRole.CHALLENGER,
        parents=(left_id,),
    )
    with pytest.raises(ValueError, match="earlier generation"):
        EarnedLineageLedger().register((left, right))


def test_provenance_accepts_parent_registered_in_an_earlier_batch() -> None:
    backbone = _provenance(
        "registered_backbone",
        generation=0,
        role=ProposalLineageRole.BACKBONE,
    )
    challenger = _provenance(
        "later_challenger",
        generation=1,
        role=ProposalLineageRole.CHALLENGER,
        parents=(backbone.candidate_id,),
    )
    ledger = EarnedLineageLedger()

    ledger.register((backbone,))
    version_after_backbone = ledger.version
    ledger.register((challenger,))

    assert ledger.version == version_after_backbone + 1
    assert {
        value["candidate_id"]
        for value in ledger.to_record(current_generation=1)["provenance"]
    } == {
        backbone.candidate_id.value,
        challenger.candidate_id.value,
    }


def test_conserved_backbone_ceiling_is_additive_and_domain_neutral() -> None:
    receipt = ConservedBackboneCeilingReceipt(
        generation=2,
        backbone_state_before_sha256=_sha("state-before"),
        backbone_ask_receipt_sha256=_sha("ask"),
        backbone_outcome_receipt_sha256=_sha("outcome"),
        backbone_state_after_sha256=_sha("state-after"),
        backbone_candidate_ids=(CandidateId("candidate_backbone_a"),),
        challenger_candidate_ids=(CandidateId("candidate_challenger_a"),),
        utility_definition_sha256=_sha("utility"),
        backbone_utility=0.75,
        union_utility=0.875,
    )
    record = receipt.to_record()
    assert receipt.complement == 0.125
    assert record["backbone_state_excludes_challenger_outcomes"] is True
    assert not {"workload", "model", "provider"}.intersection(record)
    with pytest.raises(ValueError, match="cannot trail"):
        ConservedBackboneCeilingReceipt(
            generation=2,
            backbone_state_before_sha256=_sha("state-before"),
            backbone_ask_receipt_sha256=_sha("ask"),
            backbone_outcome_receipt_sha256=_sha("outcome"),
            backbone_state_after_sha256=_sha("state-after"),
            backbone_candidate_ids=(CandidateId("candidate_backbone_a"),),
            challenger_candidate_ids=(CandidateId("candidate_challenger_a"),),
            utility_definition_sha256=_sha("utility"),
            backbone_utility=0.75,
            union_utility=0.70,
        )
