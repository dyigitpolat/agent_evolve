"""Exact accounting gates for trajectory-preserving evaluation escrow."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.evaluation_escrow import (
    BackboneSettlementUnit,
    ChallengerEvaluationLoan,
    ChallengerLoanKind,
    EvaluationEscrowLedger,
    EvaluationEscrowPolicy,
    TerminalSettlementCapacityReceipt,
    TerminalSettlementMode,
)
from agent_evolve.domain.ids import CandidateId


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _capacity(
    mode: TerminalSettlementMode,
    *,
    budget: int = 38,
    maximum: int = 2,
) -> TerminalSettlementCapacityReceipt:
    return TerminalSettlementCapacityReceipt(
        physical_evaluation_budget=budget,
        maximum_settlement_units=maximum,
        mode=mode,
        backbone_schedule_definition_sha256=_sha("schedule"),
        campaign_scope_sha256=_sha("campaign"),
        capability_receipt_sha256=_sha(f"capacity:{mode.value}"),
    )


def _loan(
    policy: EvaluationEscrowPolicy,
    label: str,
    *,
    kind: ChallengerLoanKind = ChallengerLoanKind.COLD_START_EXPLORATION,
) -> ChallengerEvaluationLoan:
    return ChallengerEvaluationLoan(
        candidate_id=CandidateId(f"candidate_{label}"),
        proposal_provenance_sha256=_sha(f"provenance:{label}"),
        evaluation_receipt_sha256=_sha(f"evaluation:{label}"),
        issued_after_backbone_evaluation_count=20,
        generation=3,
        kind=kind,
        source_ticket_sha256=(
            _sha(f"ticket:{label}")
            if kind is ChallengerLoanKind.EARNED_REPRODUCTION
            else None
        ),
        backbone_state_sha256=_sha("backbone-state-g3"),
        policy_definition_sha256=policy.definition_sha256,
    )


def test_final_batch_settlement_closes_one_challenger_at_exact_budget() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=2)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(TerminalSettlementMode.FINAL_BATCH_WITHHOLDING),
    )
    ledger.record_loan(_loan(policy, "semantic_composite"))
    unit = BackboneSettlementUnit(
        unit_index=1,
        backbone_sequence_index=38,
        mode=TerminalSettlementMode.FINAL_BATCH_WITHHOLDING,
        terminal_state_sha256=_sha("terminal-state"),
        settlement_decision_receipt_sha256=_sha("settlement-decision"),
        candidate_id=CandidateId("candidate_withheld_anchor"),
        ask_receipt_sha256=_sha("final-batch-ask"),
        prior_ranking_receipt_sha256=_sha("prior-ranking"),
    )

    settlement = ledger.settle(
        settlement_units=(unit,),
        actual_backbone_evaluations=37,
        retained_backbone_prefix_receipt_sha256=_sha("retained-prefix"),
        final_union_archive_receipt_sha256=_sha("union-archive"),
    )

    record = settlement.to_record()
    assert record["physical_budget_closed"] is True
    assert record["actual_backbone_evaluations"] == 37
    assert record["actual_challenger_evaluations"] == 1
    assert record["path_interference_for_retained_backbone"] is False
    assert ledger.open_debt_units == 0
    assert not {"workload", "model", "provider"}.intersection(record)


def test_sequential_tail_can_pay_for_cold_start_and_earned_child() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=2)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(TerminalSettlementMode.SEQUENTIAL_TRUNCATION),
    )
    ledger.record_loan(_loan(policy, "parent"))
    ledger.record_loan(
        _loan(
            policy,
            "child",
            kind=ChallengerLoanKind.EARNED_REPRODUCTION,
        )
    )
    units = tuple(
        BackboneSettlementUnit(
            unit_index=index,
            backbone_sequence_index=36 + index,
            mode=TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
            terminal_state_sha256=_sha("sequential-terminal-state"),
            settlement_decision_receipt_sha256=_sha(
                f"sequential-settlement:{index}"
            ),
        )
        for index in (1, 2)
    )

    settlement = ledger.settle(
        settlement_units=units,
        actual_backbone_evaluations=36,
        retained_backbone_prefix_receipt_sha256=_sha("official-prefix-b36"),
        final_union_archive_receipt_sha256=_sha("official-plus-lineage"),
    )
    assert len(settlement.loans) == len(settlement.settlement_units) == 2
    assert settlement.to_record()["open_debt_units"] == 0


def test_debt_cap_and_candidate_identity_are_fail_closed() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=1)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(
            TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
            maximum=1,
        ),
    )
    loan = _loan(policy, "only")
    ledger.record_loan(loan)
    with pytest.raises(ValueError, match="debt cap"):
        ledger.record_loan(_loan(policy, "excess"))

    wider = EvaluationEscrowLedger(
        policy=EvaluationEscrowPolicy(maximum_open_debt=2),
        capacity=_capacity(TerminalSettlementMode.SEQUENTIAL_TRUNCATION),
    )
    foreign = _loan(EvaluationEscrowPolicy(maximum_open_debt=1), "foreign")
    with pytest.raises(ValueError, match="another escrow policy"):
        wider.record_loan(foreign)


def test_settlement_rejects_wrong_mode_or_unclosed_debt() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=2)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(TerminalSettlementMode.FINAL_BATCH_WITHHOLDING),
    )
    ledger.record_loan(_loan(policy, "left"))
    ledger.record_loan(_loan(policy, "right"))
    sequential = BackboneSettlementUnit(
        unit_index=1,
        backbone_sequence_index=38,
        mode=TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
        terminal_state_sha256=_sha("terminal"),
        settlement_decision_receipt_sha256=_sha("decision"),
    )
    with pytest.raises(ValueError, match="mode differs"):
        ledger.settle(
            settlement_units=(sequential,),
            actual_backbone_evaluations=37,
            retained_backbone_prefix_receipt_sha256=_sha("prefix"),
            final_union_archive_receipt_sha256=_sha("union"),
        )

    final_batch = BackboneSettlementUnit(
        unit_index=1,
        backbone_sequence_index=38,
        mode=TerminalSettlementMode.FINAL_BATCH_WITHHOLDING,
        terminal_state_sha256=_sha("terminal"),
        settlement_decision_receipt_sha256=_sha("decision"),
        candidate_id=CandidateId("candidate_withheld"),
        ask_receipt_sha256=_sha("ask"),
        prior_ranking_receipt_sha256=_sha("ranking"),
    )
    with pytest.raises(ValueError, match="close every debt"):
        ledger.settle(
            settlement_units=(final_batch,),
            actual_backbone_evaluations=37,
            retained_backbone_prefix_receipt_sha256=_sha("prefix"),
            final_union_archive_receipt_sha256=_sha("union"),
        )


def test_sequential_settlement_requires_literal_unique_terminal_tail() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=2)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(TerminalSettlementMode.SEQUENTIAL_TRUNCATION),
    )
    ledger.record_loan(_loan(policy, "left"))
    ledger.record_loan(_loan(policy, "right"))
    repeated = tuple(
        BackboneSettlementUnit(
            unit_index=index,
            backbone_sequence_index=37,
            mode=TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
            terminal_state_sha256=_sha("terminal"),
            settlement_decision_receipt_sha256=_sha(f"decision:{index}"),
        )
        for index in (1, 2)
    )
    with pytest.raises(ValueError, match="repeats a backbone sequence index"):
        ledger.settle(
            settlement_units=repeated,
            actual_backbone_evaluations=36,
            retained_backbone_prefix_receipt_sha256=_sha("prefix"),
            final_union_archive_receipt_sha256=_sha("union"),
        )

    skipped = tuple(
        BackboneSettlementUnit(
            unit_index=index,
            backbone_sequence_index=sequence_index,
            mode=TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
            terminal_state_sha256=_sha("terminal"),
            settlement_decision_receipt_sha256=_sha(f"skipped:{index}"),
        )
        for index, sequence_index in ((1, 36), (2, 38))
    )
    with pytest.raises(ValueError, match="exact terminal backbone tail"):
        ledger.settle(
            settlement_units=skipped,
            actual_backbone_evaluations=36,
            retained_backbone_prefix_receipt_sha256=_sha("prefix"),
            final_union_archive_receipt_sha256=_sha("union"),
        )


def test_settlement_rejects_a_loan_issued_beyond_retained_prefix() -> None:
    policy = EvaluationEscrowPolicy(maximum_open_debt=1)
    ledger = EvaluationEscrowLedger(
        policy=policy,
        capacity=_capacity(
            TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
            maximum=1,
        ),
    )
    loan = _loan(policy, "future")
    object.__setattr__(loan, "issued_after_backbone_evaluation_count", 38)
    ledger.record_loan(loan)
    unit = BackboneSettlementUnit(
        unit_index=1,
        backbone_sequence_index=38,
        mode=TerminalSettlementMode.SEQUENTIAL_TRUNCATION,
        terminal_state_sha256=_sha("terminal"),
        settlement_decision_receipt_sha256=_sha("decision"),
    )
    with pytest.raises(ValueError, match="after the retained backbone prefix"):
        ledger.settle(
            settlement_units=(unit,),
            actual_backbone_evaluations=37,
            retained_backbone_prefix_receipt_sha256=_sha("prefix"),
            final_union_archive_receipt_sha256=_sha("union"),
        )
