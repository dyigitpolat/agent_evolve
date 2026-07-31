"""Workload-neutral trajectory-preserving evaluation escrow contracts.

Challenger evaluations are recorded as debt against a protected native
optimizer.  Debt is settled only by withholding terminal backbone units, so
every retained backbone action has the same ask/tell history it would have had
without challengers.  This module owns accounting and evidence; workload
adapters own candidates, evaluators, and native optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_POLICY_DOMAIN = b"agent-evolve:evaluation-escrow-policy:v1\x00"
_CAPACITY_DOMAIN = b"agent-evolve:terminal-settlement-capacity:v1\x00"
_LOAN_DOMAIN = b"agent-evolve:challenger-evaluation-loan:v1\x00"
_UNIT_DOMAIN = b"agent-evolve:backbone-settlement-unit:v1\x00"
_SETTLEMENT_DOMAIN = b"agent-evolve:evaluation-escrow-settlement:v1\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


class ChallengerLoanKind(str, Enum):
    COLD_START_EXPLORATION = "cold_start_exploration"
    EARNED_REPRODUCTION = "earned_reproduction"


class TerminalSettlementMode(str, Enum):
    SEQUENTIAL_TRUNCATION = "sequential_truncation"
    FINAL_BATCH_WITHHOLDING = "final_batch_withholding"


@dataclass(frozen=True, slots=True)
class EvaluationEscrowPolicy:
    """Frozen debt authority; model and workload identity are intentionally absent."""

    policy_id: str = "trajectory_preserving_evaluation_escrow"
    policy_version: int = 1
    maximum_open_debt: int = 2

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        if type(self.maximum_open_debt) is not int or self.maximum_open_debt <= 0:
            raise ValueError("maximum_open_debt must be positive")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "maximum_open_debt": self.maximum_open_debt,
            "challenger_outcomes_visible_to_backbone": False,
            "settlement_scope": "terminal_only_after_last_retained_transition",
            "workload_model_provider_fields_consulted": False,
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "definition_sha256": self.definition_sha256}


@dataclass(frozen=True, slots=True)
class TerminalSettlementCapacityReceipt:
    """Adapter proof that a terminal tail can repay a bounded debt."""

    physical_evaluation_budget: int
    maximum_settlement_units: int
    mode: TerminalSettlementMode
    backbone_schedule_definition_sha256: str
    campaign_scope_sha256: str
    capability_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.physical_evaluation_budget) is not int
            or self.physical_evaluation_budget <= 0
        ):
            raise ValueError("physical_evaluation_budget must be positive")
        if (
            type(self.maximum_settlement_units) is not int
            or not 0 < self.maximum_settlement_units
            < self.physical_evaluation_budget
        ):
            raise ValueError(
                "maximum_settlement_units must lie inside the physical budget"
            )
        if type(self.mode) is not TerminalSettlementMode:
            raise TypeError("mode must be an exact TerminalSettlementMode")
        for name in (
            "backbone_schedule_definition_sha256",
            "campaign_scope_sha256",
            "capability_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "physical_evaluation_budget": self.physical_evaluation_budget,
            "maximum_settlement_units": self.maximum_settlement_units,
            "mode": self.mode.value,
            "backbone_schedule_definition_sha256": (
                self.backbone_schedule_definition_sha256
            ),
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "capability_receipt_sha256": self.capability_receipt_sha256,
            "settlement_after_last_retained_state_transition": True,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CAPACITY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ChallengerEvaluationLoan:
    """One completed real challenger evaluation and its debt unit."""

    candidate_id: CandidateId
    proposal_provenance_sha256: str
    evaluation_receipt_sha256: str
    issued_after_backbone_evaluation_count: int
    generation: int
    kind: ChallengerLoanKind
    source_ticket_sha256: str | None
    backbone_state_sha256: str
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        for name in (
            "proposal_provenance_sha256",
            "evaluation_receipt_sha256",
            "backbone_state_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.issued_after_backbone_evaluation_count) is not int
            or self.issued_after_backbone_evaluation_count < 0
        ):
            raise ValueError(
                "issued_after_backbone_evaluation_count must be non-negative"
            )
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.kind) is not ChallengerLoanKind:
            raise TypeError("kind must be an exact ChallengerLoanKind")
        if self.kind is ChallengerLoanKind.EARNED_REPRODUCTION:
            if self.source_ticket_sha256 is None:
                raise ValueError("earned reproduction requires its source ticket")
            require_sha256(self.source_ticket_sha256, "source_ticket_sha256")
        elif self.source_ticket_sha256 is not None:
            raise ValueError("cold-start exploration cannot consume a lineage ticket")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "candidate_id": self.candidate_id.value,
            "proposal_provenance_sha256": self.proposal_provenance_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "issued_after_backbone_evaluation_count": (
                self.issued_after_backbone_evaluation_count
            ),
            "generation": self.generation,
            "kind": self.kind.value,
            "source_ticket_sha256": self.source_ticket_sha256,
            "backbone_state_sha256": self.backbone_state_sha256,
            "policy_definition_sha256": self.policy_definition_sha256,
            "physical_evaluation_units": 1,
            "outcome_visible_to_backbone_optimizer": False,
        }

    @property
    def loan_sha256(self) -> str:
        return _hash(_LOAN_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "loan_sha256": self.loan_sha256}


@dataclass(frozen=True, slots=True)
class BackboneSettlementUnit:
    """One terminal native evaluation withheld to repay one challenger loan."""

    unit_index: int
    backbone_sequence_index: int
    mode: TerminalSettlementMode
    terminal_state_sha256: str
    settlement_decision_receipt_sha256: str
    candidate_id: CandidateId | None = None
    ask_receipt_sha256: str | None = None
    prior_ranking_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.unit_index) is not int or self.unit_index <= 0:
            raise ValueError("unit_index must be positive")
        if type(self.backbone_sequence_index) is not int or (
            self.backbone_sequence_index <= 0
        ):
            raise ValueError("backbone_sequence_index must be positive")
        if type(self.mode) is not TerminalSettlementMode:
            raise TypeError("mode must be an exact TerminalSettlementMode")
        require_sha256(self.terminal_state_sha256, "terminal_state_sha256")
        require_sha256(
            self.settlement_decision_receipt_sha256,
            "settlement_decision_receipt_sha256",
        )
        if self.mode is TerminalSettlementMode.FINAL_BATCH_WITHHOLDING:
            if type(self.candidate_id) is not CandidateId:
                raise TypeError("final-batch withholding requires a candidate_id")
            CandidateId.__post_init__(self.candidate_id)
            if self.ask_receipt_sha256 is None:
                raise ValueError("final-batch withholding requires an ask receipt")
            if self.prior_ranking_receipt_sha256 is None:
                raise ValueError(
                    "final-batch withholding requires a prior ranking receipt"
                )
            require_sha256(self.ask_receipt_sha256, "ask_receipt_sha256")
            require_sha256(
                self.prior_ranking_receipt_sha256,
                "prior_ranking_receipt_sha256",
            )
        elif any(
            value is not None
            for value in (
                self.candidate_id,
                self.ask_receipt_sha256,
                self.prior_ranking_receipt_sha256,
            )
        ):
            raise ValueError(
                "sequential truncation cannot name an unasked future candidate"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "unit_index": self.unit_index,
            "backbone_sequence_index": self.backbone_sequence_index,
            "mode": self.mode.value,
            "terminal_state_sha256": self.terminal_state_sha256,
            "settlement_decision_receipt_sha256": (
                self.settlement_decision_receipt_sha256
            ),
            "candidate_id": (
                None if self.candidate_id is None else self.candidate_id.value
            ),
            "ask_receipt_sha256": self.ask_receipt_sha256,
            "prior_ranking_receipt_sha256": self.prior_ranking_receipt_sha256,
            "physical_evaluation_units_withheld": 1,
            "later_backbone_state_transitions_allowed": False,
        }

    @property
    def unit_sha256(self) -> str:
        return _hash(_UNIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "unit_sha256": self.unit_sha256}


@dataclass(frozen=True, slots=True)
class EvaluationEscrowSettlement:
    """Exact physical-budget and no-path-interference closure receipt."""

    policy_definition_sha256: str
    capacity_receipt_sha256: str
    physical_evaluation_budget: int
    actual_backbone_evaluations: int
    loans: tuple[ChallengerEvaluationLoan, ...]
    settlement_units: tuple[BackboneSettlementUnit, ...]
    retained_backbone_prefix_receipt_sha256: str
    final_union_archive_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        require_sha256(self.capacity_receipt_sha256, "capacity_receipt_sha256")
        if (
            type(self.physical_evaluation_budget) is not int
            or self.physical_evaluation_budget <= 0
        ):
            raise ValueError("physical_evaluation_budget must be positive")
        if (
            type(self.actual_backbone_evaluations) is not int
            or self.actual_backbone_evaluations < 0
        ):
            raise ValueError("actual_backbone_evaluations must be non-negative")
        if type(self.loans) is not tuple or not self.loans or any(
            type(value) is not ChallengerEvaluationLoan for value in self.loans
        ):
            raise ValueError("loans must be a non-empty exact tuple")
        if type(self.settlement_units) is not tuple or not self.settlement_units or any(
            type(value) is not BackboneSettlementUnit
            for value in self.settlement_units
        ):
            raise ValueError("settlement_units must be a non-empty exact tuple")
        for value in self.loans:
            ChallengerEvaluationLoan.__post_init__(value)
            if value.policy_definition_sha256 != self.policy_definition_sha256:
                raise ValueError("loan uses another escrow policy")
        for value in self.settlement_units:
            BackboneSettlementUnit.__post_init__(value)
        if len(self.loans) != len(self.settlement_units):
            raise ValueError("settlement units must close every debt unit exactly")
        if len({value.candidate_id for value in self.loans}) != len(self.loans):
            raise ValueError("settlement repeats a challenger loan candidate")
        if any(
            value.issued_after_backbone_evaluation_count
            > self.actual_backbone_evaluations
            for value in self.loans
        ):
            raise ValueError(
                "challenger loan was issued after the retained backbone prefix"
            )
        if tuple(value.unit_index for value in self.settlement_units) != tuple(
            range(1, len(self.settlement_units) + 1)
        ):
            raise ValueError("settlement unit indices must be canonical")
        sequence_indices = tuple(
            value.backbone_sequence_index for value in self.settlement_units
        )
        if len(set(sequence_indices)) != len(sequence_indices):
            raise ValueError("settlement repeats a backbone sequence index")
        modes = {value.mode for value in self.settlement_units}
        if len(modes) != 1:
            raise ValueError("one settlement cannot mix adapter modes")
        if self.actual_backbone_evaluations + len(self.loans) != (
            self.physical_evaluation_budget
        ):
            raise ValueError("real backbone and challenger evaluations exceed budget")
        if self.actual_backbone_evaluations + len(self.settlement_units) != (
            self.physical_evaluation_budget
        ):
            raise ValueError("withheld units do not close the planned backbone budget")
        mode = next(iter(modes))
        if mode is TerminalSettlementMode.SEQUENTIAL_TRUNCATION and (
            sequence_indices
            != tuple(
                range(
                    self.actual_backbone_evaluations + 1,
                    self.physical_evaluation_budget + 1,
                )
            )
        ):
            raise ValueError(
                "sequential settlement must name the exact terminal backbone tail"
            )
        withheld_candidate_ids = tuple(
            value.candidate_id
            for value in self.settlement_units
            if value.candidate_id is not None
        )
        if len(set(withheld_candidate_ids)) != len(withheld_candidate_ids):
            raise ValueError("settlement repeats a withheld backbone candidate")
        require_sha256(
            self.retained_backbone_prefix_receipt_sha256,
            "retained_backbone_prefix_receipt_sha256",
        )
        require_sha256(
            self.final_union_archive_receipt_sha256,
            "final_union_archive_receipt_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_definition_sha256": self.policy_definition_sha256,
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "physical_evaluation_budget": self.physical_evaluation_budget,
            "actual_backbone_evaluations": self.actual_backbone_evaluations,
            "actual_challenger_evaluations": len(self.loans),
            "loans": [value.to_record() for value in self.loans],
            "settlement_units": [
                value.to_record() for value in self.settlement_units
            ],
            "retained_backbone_prefix_receipt_sha256": (
                self.retained_backbone_prefix_receipt_sha256
            ),
            "final_union_archive_receipt_sha256": (
                self.final_union_archive_receipt_sha256
            ),
            "physical_budget_closed": True,
            "open_debt_units": 0,
            "path_interference_for_retained_backbone": False,
        }

    @property
    def settlement_sha256(self) -> str:
        return _hash(_SETTLEMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "settlement_sha256": self.settlement_sha256,
        }


@dataclass(slots=True)
class EvaluationEscrowLedger:
    """Mutable campaign ledger with one final, immutable settlement."""

    policy: EvaluationEscrowPolicy
    capacity: TerminalSettlementCapacityReceipt
    _loans: list[ChallengerEvaluationLoan] = field(
        init=False,
        default_factory=list,
    )
    _settlement: EvaluationEscrowSettlement | None = field(
        init=False,
        default=None,
    )
    _version: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if type(self.policy) is not EvaluationEscrowPolicy:
            raise TypeError("policy must be an exact EvaluationEscrowPolicy")
        if type(self.capacity) is not TerminalSettlementCapacityReceipt:
            raise TypeError(
                "capacity must be an exact TerminalSettlementCapacityReceipt"
            )
        EvaluationEscrowPolicy.__post_init__(self.policy)
        TerminalSettlementCapacityReceipt.__post_init__(self.capacity)
        if self.policy.maximum_open_debt > self.capacity.maximum_settlement_units:
            raise ValueError("policy debt cap exceeds terminal settlement capacity")

    @property
    def version(self) -> int:
        return self._version

    @property
    def open_debt_units(self) -> int:
        return 0 if self._settlement is not None else len(self._loans)

    def record_loan(self, loan: ChallengerEvaluationLoan) -> None:
        if self._settlement is not None:
            raise RuntimeError("settled escrow cannot accept another loan")
        if type(loan) is not ChallengerEvaluationLoan:
            raise TypeError("loan must be an exact ChallengerEvaluationLoan")
        ChallengerEvaluationLoan.__post_init__(loan)
        if loan.policy_definition_sha256 != self.policy.definition_sha256:
            raise ValueError("loan uses another escrow policy")
        if len(self._loans) >= self.policy.maximum_open_debt:
            raise ValueError("challenger loan exceeds the frozen debt cap")
        if any(value.candidate_id == loan.candidate_id for value in self._loans):
            raise ValueError("one challenger candidate cannot incur debt twice")
        self._loans.append(loan)
        self._version += 1

    def settle(
        self,
        *,
        settlement_units: tuple[BackboneSettlementUnit, ...],
        actual_backbone_evaluations: int,
        retained_backbone_prefix_receipt_sha256: str,
        final_union_archive_receipt_sha256: str,
    ) -> EvaluationEscrowSettlement:
        if self._settlement is not None:
            raise RuntimeError("evaluation escrow was already settled")
        if not self._loans:
            raise ValueError("debt-free escrow needs no settlement")
        if any(value.mode is not self.capacity.mode for value in settlement_units):
            raise ValueError("settlement mode differs from adapter capacity")
        if len(settlement_units) > self.capacity.maximum_settlement_units:
            raise ValueError("settlement exceeds adapter terminal capacity")
        settlement = EvaluationEscrowSettlement(
            policy_definition_sha256=self.policy.definition_sha256,
            capacity_receipt_sha256=self.capacity.receipt_sha256,
            physical_evaluation_budget=self.capacity.physical_evaluation_budget,
            actual_backbone_evaluations=actual_backbone_evaluations,
            loans=tuple(self._loans),
            settlement_units=settlement_units,
            retained_backbone_prefix_receipt_sha256=(
                retained_backbone_prefix_receipt_sha256
            ),
            final_union_archive_receipt_sha256=final_union_archive_receipt_sha256,
        )
        self._settlement = settlement
        self._version += 1
        return settlement

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": self.policy.to_record(),
            "capacity": self.capacity.to_record(),
            "ledger_version": self._version,
            "loans": [value.to_record() for value in self._loans],
            "open_debt_units": self.open_debt_units,
            "settlement": (
                None if self._settlement is None else self._settlement.to_record()
            ),
            "workload_model_provider_fields_consulted": False,
        }


__all__ = [
    "BackboneSettlementUnit",
    "ChallengerEvaluationLoan",
    "ChallengerLoanKind",
    "EvaluationEscrowLedger",
    "EvaluationEscrowPolicy",
    "EvaluationEscrowSettlement",
    "TerminalSettlementCapacityReceipt",
    "TerminalSettlementMode",
]
