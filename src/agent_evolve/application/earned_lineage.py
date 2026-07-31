"""Workload-neutral conserved-backbone and earned-lineage control contracts.

This module deliberately contains no candidate schema, objective name,
workload identity, model identity, or provider field.  Adapters authenticate
where a materialized candidate came from and an archive-utility policy supplies
conserved per-stage credit.  The controller turns positive challenger credit
into expiring, one-use parent-exposure tickets.

Tickets reserve *proposal opportunities*, not evaluations.  Descendants still
pass through the global feasibility, deduplication, and allocation broker.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PROVENANCE_DOMAIN = b"agent-evolve:candidate-proposal-provenance:v1\x00"
_STAGE_CREDIT_DOMAIN = b"agent-evolve:conserved-stage-credit:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:earned-lineage-policy:v1\x00"
_TICKET_DOMAIN = b"agent-evolve:earned-reproduction-ticket:v1\x00"
_ISSUANCE_DOMAIN = b"agent-evolve:earned-reproduction-issuance:v1\x00"
_RESERVATION_DOMAIN = b"agent-evolve:earned-reproduction-reservation:v1\x00"
_COMMIT_DOMAIN = b"agent-evolve:earned-reproduction-commit:v1\x00"
_PARENT_ALLOCATION_DOMAIN = b"agent-evolve:earned-parent-allocation:v1\x00"
_CEILING_DOMAIN = b"agent-evolve:conserved-backbone-ceiling:v1\x00"


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


def _canonical_float(value: float, name: str, *, nonnegative: bool = True) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _candidate_ids(
    values: tuple[CandidateId, ...],
    name: str,
    *,
    allow_empty: bool,
) -> None:
    if type(values) is not tuple or (not allow_empty and not values):
        raise ValueError(f"{name} must be an exact non-empty tuple")
    if any(type(value) is not CandidateId for value in values):
        raise TypeError(f"{name} must contain exact CandidateId values")
    for value in values:
        CandidateId.__post_init__(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")


class ProposalLineageRole(str, Enum):
    """Scientific role of a proposal, independent of its implementation."""

    BACKBONE = "backbone"
    CHALLENGER = "challenger"


@dataclass(frozen=True, slots=True)
class CandidateProposalProvenance:
    """Immutable source identity for one fully materialized candidate."""

    candidate_id: CandidateId
    configuration_sha256: str
    generation: int
    source_role: ProposalLineageRole
    proposal_expert_id: str
    proposal_expert_version: int
    proposal_expert_definition_sha256: str
    operator_id: str
    parent_candidate_ids: tuple[CandidateId, ...]
    decision_cutoff_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        require_sha256(self.configuration_sha256, "configuration_sha256")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative exact integer")
        if type(self.source_role) is not ProposalLineageRole:
            raise TypeError("source_role must be an exact ProposalLineageRole")
        _token(self.proposal_expert_id, "proposal_expert_id")
        if (
            type(self.proposal_expert_version) is not int
            or self.proposal_expert_version <= 0
        ):
            raise ValueError("proposal_expert_version must be positive")
        require_sha256(
            self.proposal_expert_definition_sha256,
            "proposal_expert_definition_sha256",
        )
        _token(self.operator_id, "operator_id")
        _candidate_ids(
            self.parent_candidate_ids,
            "parent_candidate_ids",
            allow_empty=True,
        )
        if self.candidate_id in self.parent_candidate_ids:
            raise ValueError("a proposal cannot be its own parent")
        require_sha256(self.decision_cutoff_sha256, "decision_cutoff_sha256")
        require_sha256(self.source_receipt_sha256, "source_receipt_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "candidate_id": self.candidate_id.value,
            "configuration_sha256": self.configuration_sha256,
            "generation": self.generation,
            "source_role": self.source_role.value,
            "proposal_expert": {
                "expert_id": self.proposal_expert_id,
                "expert_version": self.proposal_expert_version,
                "definition_sha256": self.proposal_expert_definition_sha256,
            },
            "operator_id": self.operator_id,
            "parent_candidate_ids": [
                value.value for value in self.parent_candidate_ids
            ],
            "decision_cutoff_sha256": self.decision_cutoff_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "workload_model_provider_fields_present": False,
        }

    @property
    def provenance_sha256(self) -> str:
        return _hash(_PROVENANCE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateConservedCredit:
    """One candidate's non-negative share of a stage archive gain."""

    candidate_id: CandidateId
    contribution: float
    admitted_to_archive: bool
    outcome_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        _canonical_float(self.contribution, "contribution")
        if type(self.admitted_to_archive) is not bool:
            raise TypeError("admitted_to_archive must be an exact bool")
        if self.contribution > 0.0 and not self.admitted_to_archive:
            raise ValueError("positive archive credit requires archive admission")
        require_sha256(self.outcome_receipt_sha256, "outcome_receipt_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate_id": self.candidate_id.value,
            "contribution_hex": self.contribution.hex(),
            "admitted_to_archive": self.admitted_to_archive,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ConservedStageCreditReceipt:
    """Candidate credits that close to one authenticated archive transition."""

    generation: int
    utility_id: str
    utility_version: int
    utility_definition_sha256: str
    pre_archive_sha256: str
    post_archive_sha256: str
    pre_utility: float
    post_utility: float
    contribution_policy_id: str
    contribution_policy_version: int
    contribution_policy_definition_sha256: str
    candidate_credits: tuple[CandidateConservedCredit, ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        _token(self.utility_id, "utility_id")
        if type(self.utility_version) is not int or self.utility_version <= 0:
            raise ValueError("utility_version must be positive")
        require_sha256(self.utility_definition_sha256, "utility_definition_sha256")
        require_sha256(self.pre_archive_sha256, "pre_archive_sha256")
        require_sha256(self.post_archive_sha256, "post_archive_sha256")
        _canonical_float(self.pre_utility, "pre_utility")
        _canonical_float(self.post_utility, "post_utility")
        if self.post_utility < self.pre_utility:
            raise ValueError("archive utility cannot decrease")
        _token(self.contribution_policy_id, "contribution_policy_id")
        if (
            type(self.contribution_policy_version) is not int
            or self.contribution_policy_version <= 0
        ):
            raise ValueError("contribution_policy_version must be positive")
        require_sha256(
            self.contribution_policy_definition_sha256,
            "contribution_policy_definition_sha256",
        )
        if type(self.candidate_credits) is not tuple or not self.candidate_credits:
            raise ValueError("candidate_credits must be a non-empty exact tuple")
        if any(
            type(value) is not CandidateConservedCredit
            for value in self.candidate_credits
        ):
            raise TypeError("candidate_credits must contain exact values")
        for value in self.candidate_credits:
            CandidateConservedCredit.__post_init__(value)
        ids = tuple(value.candidate_id.value for value in self.candidate_credits)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("candidate credits must use canonical unique ID order")
        gain = self.post_utility - self.pre_utility
        total = math.fsum(value.contribution for value in self.candidate_credits)
        tolerance = 8.0 * math.ulp(max(1.0, self.post_utility, self.pre_utility))
        if abs(total - gain) > tolerance:
            raise ValueError("candidate contributions do not conserve stage gain")

    @property
    def stage_gain(self) -> float:
        return self.post_utility - self.pre_utility

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "utility": {
                "utility_id": self.utility_id,
                "utility_version": self.utility_version,
                "definition_sha256": self.utility_definition_sha256,
                "pre_archive_sha256": self.pre_archive_sha256,
                "post_archive_sha256": self.post_archive_sha256,
                "pre_utility_hex": self.pre_utility.hex(),
                "post_utility_hex": self.post_utility.hex(),
                "stage_gain_hex": self.stage_gain.hex(),
            },
            "contribution_policy": {
                "policy_id": self.contribution_policy_id,
                "policy_version": self.contribution_policy_version,
                "definition_sha256": self.contribution_policy_definition_sha256,
            },
            "candidate_credits": [
                value.to_record() for value in self.candidate_credits
            ],
            "credit_conservation_verified": True,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_STAGE_CREDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class EarnedLineagePolicy:
    """Frozen rule mapping positive real challenger credit to tickets."""

    positive_credit_floor: float = 0.0
    max_tickets_per_stage: int = 2
    max_open_tickets_per_candidate: int = 1
    ticket_ttl_generations: int = 2
    descendant_discount: Fraction = Fraction(1, 2)

    def __post_init__(self) -> None:
        _canonical_float(self.positive_credit_floor, "positive_credit_floor")
        for name in (
            "max_tickets_per_stage",
            "max_open_tickets_per_candidate",
            "ticket_ttl_generations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.descendant_discount) is not Fraction:
            raise TypeError("descendant_discount must be an exact Fraction")
        if not Fraction(0, 1) < self.descendant_discount <= Fraction(1, 1):
            raise ValueError("descendant_discount must lie in (0, 1]")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "positive_credit_floor_hex": self.positive_credit_floor.hex(),
            "max_tickets_per_stage": self.max_tickets_per_stage,
            "max_open_tickets_per_candidate": (
                self.max_open_tickets_per_candidate
            ),
            "ticket_ttl_generations": self.ticket_ttl_generations,
            "descendant_discount": {
                "numerator": self.descendant_discount.numerator,
                "denominator": self.descendant_discount.denominator,
            },
            "quality_signal": "conserved_real_archive_contribution_only",
            "ticket_scope": "one_parent_exposure_not_evaluation_guarantee",
            "workload_model_provider_fields_consulted": False,
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "definition_sha256": self.definition_sha256}


@dataclass(frozen=True, slots=True)
class ReproductionTicket:
    candidate_id: CandidateId
    issued_after_generation: int
    eligible_generation: int
    expires_after_generation: int
    conserved_credit: float
    stage_credit_receipt_sha256: str
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        for name in (
            "issued_after_generation",
            "eligible_generation",
            "expires_after_generation",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.eligible_generation != self.issued_after_generation + 1:
            raise ValueError("a ticket must first become eligible next generation")
        if self.expires_after_generation < self.eligible_generation:
            raise ValueError("ticket expiry precedes eligibility")
        _canonical_float(self.conserved_credit, "conserved_credit")
        if self.conserved_credit <= 0.0:
            raise ValueError("a reproduction ticket requires positive credit")
        require_sha256(
            self.stage_credit_receipt_sha256,
            "stage_credit_receipt_sha256",
        )
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "candidate_id": self.candidate_id.value,
            "issued_after_generation": self.issued_after_generation,
            "eligible_generation": self.eligible_generation,
            "expires_after_generation": self.expires_after_generation,
            "conserved_credit_hex": self.conserved_credit.hex(),
            "stage_credit_receipt_sha256": self.stage_credit_receipt_sha256,
            "policy_definition_sha256": self.policy_definition_sha256,
            "entitlement": "one_parent_exposure_not_evaluation_guarantee",
        }

    @property
    def ticket_sha256(self) -> str:
        return _hash(_TICKET_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "ticket_sha256": self.ticket_sha256}


@dataclass(frozen=True, slots=True)
class ReproductionTicketIssuance:
    generation: int
    stage_credit_receipt_sha256: str
    policy_definition_sha256: str
    tickets: tuple[ReproductionTicket, ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        require_sha256(
            self.stage_credit_receipt_sha256,
            "stage_credit_receipt_sha256",
        )
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.tickets) is not tuple or any(
            type(value) is not ReproductionTicket for value in self.tickets
        ):
            raise TypeError("tickets must contain exact ReproductionTicket values")
        for value in self.tickets:
            ReproductionTicket.__post_init__(value)
            if (
                value.issued_after_generation != self.generation
                or value.stage_credit_receipt_sha256
                != self.stage_credit_receipt_sha256
                or value.policy_definition_sha256 != self.policy_definition_sha256
            ):
                raise ValueError("issued ticket differs from issuance identity")
        ids = tuple(value.ticket_sha256 for value in self.tickets)
        if len(set(ids)) != len(ids):
            raise ValueError("issuance cannot repeat a ticket")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "stage_credit_receipt_sha256": self.stage_credit_receipt_sha256,
            "policy_definition_sha256": self.policy_definition_sha256,
            "tickets": [value.to_record() for value in self.tickets],
        }

    @property
    def issuance_sha256(self) -> str:
        return _hash(_ISSUANCE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "issuance_sha256": self.issuance_sha256}


@dataclass(frozen=True, slots=True)
class PreparedReproductionReservation:
    generation: int
    ledger_version: int
    available_candidate_ids_sha256: str
    tickets: tuple[ReproductionTicket, ...]
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.ledger_version) is not int or self.ledger_version < 0:
            raise ValueError("ledger_version must be non-negative")
        require_sha256(
            self.available_candidate_ids_sha256,
            "available_candidate_ids_sha256",
        )
        if type(self.tickets) is not tuple or any(
            type(value) is not ReproductionTicket for value in self.tickets
        ):
            raise TypeError("tickets must contain exact values")
        for value in self.tickets:
            ReproductionTicket.__post_init__(value)
            if not value.eligible_generation <= self.generation <= (
                value.expires_after_generation
            ):
                raise ValueError("reservation contains an ineligible ticket")
        if len({value.candidate_id for value in self.tickets}) != len(self.tickets):
            raise ValueError("reservation repeats a candidate")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    @property
    def candidate_ids(self) -> tuple[CandidateId, ...]:
        return tuple(value.candidate_id for value in self.tickets)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "ledger_version": self.ledger_version,
            "available_candidate_ids_sha256": self.available_candidate_ids_sha256,
            "ticket_sha256s": [value.ticket_sha256 for value in self.tickets],
            "candidate_ids": [value.value for value in self.candidate_ids],
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def reservation_sha256(self) -> str:
        return _hash(_RESERVATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "reservation_sha256": self.reservation_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReproductionReservationCommit:
    reservation_sha256: str
    generation: int
    consumed_ticket_sha256s: tuple[str, ...]
    parent_exposure_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.reservation_sha256, "reservation_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if (
            type(self.consumed_ticket_sha256s) is not tuple
            or not self.consumed_ticket_sha256s
        ):
            raise ValueError("consumed_ticket_sha256s must be non-empty")
        for value in self.consumed_ticket_sha256s:
            require_sha256(value, "consumed_ticket_sha256")
        if len(set(self.consumed_ticket_sha256s)) != len(
            self.consumed_ticket_sha256s
        ):
            raise ValueError("commit repeats a consumed ticket")
        require_sha256(
            self.parent_exposure_receipt_sha256,
            "parent_exposure_receipt_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "reservation_sha256": self.reservation_sha256,
            "generation": self.generation,
            "consumed_ticket_sha256s": list(self.consumed_ticket_sha256s),
            "parent_exposure_receipt_sha256": (
                self.parent_exposure_receipt_sha256
            ),
        }

    @property
    def commit_sha256(self) -> str:
        return _hash(_COMMIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "commit_sha256": self.commit_sha256}


@dataclass(frozen=True, slots=True)
class EarnedParentAllocation:
    """A fixed-width parent set containing every reserved earned parent."""

    generation: int
    base_parent_ids: tuple[CandidateId, ...]
    earned_parent_ids: tuple[CandidateId, ...]
    selected_parent_ids: tuple[CandidateId, ...]
    reservation_sha256: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in (
            "base_parent_ids",
            "earned_parent_ids",
            "selected_parent_ids",
        ):
            _candidate_ids(
                getattr(self, name),
                name,
                allow_empty=name == "earned_parent_ids",
            )
        if len(self.selected_parent_ids) != len(self.base_parent_ids):
            raise ValueError("earned parent allocation must preserve width")
        if not set(self.earned_parent_ids).issubset(self.selected_parent_ids):
            raise ValueError("selected parents omit an earned parent")
        require_sha256(self.reservation_sha256, "reservation_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "base_parent_ids": [value.value for value in self.base_parent_ids],
            "earned_parent_ids": [value.value for value in self.earned_parent_ids],
            "selected_parent_ids": [
                value.value for value in self.selected_parent_ids
            ],
            "reservation_sha256": self.reservation_sha256,
            "workload_model_provider_fields_consulted": False,
        }

    @property
    def allocation_sha256(self) -> str:
        return _hash(_PARENT_ALLOCATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "allocation_sha256": self.allocation_sha256}


def allocate_earned_parents(
    *,
    generation: int,
    base_parent_ids: tuple[CandidateId, ...],
    reservation: PreparedReproductionReservation,
) -> EarnedParentAllocation:
    """Inject all earned parents while preserving the base selector's width."""

    reservation.__post_init__()
    if reservation.generation != generation:
        raise ValueError("reservation belongs to another generation")
    _candidate_ids(base_parent_ids, "base_parent_ids", allow_empty=False)
    earned = reservation.candidate_ids
    if len(earned) > len(base_parent_ids):
        raise ValueError("earned tickets exceed parent-selection width")
    retained = [value for value in base_parent_ids if value not in set(earned)]
    retained = retained[: len(base_parent_ids) - len(earned)]
    selected = tuple(retained) + earned
    if len(selected) != len(base_parent_ids):
        raise ValueError("base and earned parents cannot fill the required width")
    return EarnedParentAllocation(
        generation=generation,
        base_parent_ids=base_parent_ids,
        earned_parent_ids=earned,
        selected_parent_ids=selected,
        reservation_sha256=reservation.reservation_sha256,
    )


@dataclass(frozen=True, slots=True)
class ConservedBackboneCeilingReceipt:
    """One dual-ledger checkpoint proving a protected additive ceiling."""

    generation: int
    backbone_state_before_sha256: str
    backbone_ask_receipt_sha256: str
    backbone_outcome_receipt_sha256: str
    backbone_state_after_sha256: str
    backbone_candidate_ids: tuple[CandidateId, ...]
    challenger_candidate_ids: tuple[CandidateId, ...]
    utility_definition_sha256: str
    backbone_utility: float
    union_utility: float

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in (
            "backbone_state_before_sha256",
            "backbone_ask_receipt_sha256",
            "backbone_outcome_receipt_sha256",
            "backbone_state_after_sha256",
            "utility_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _candidate_ids(
            self.backbone_candidate_ids,
            "backbone_candidate_ids",
            allow_empty=False,
        )
        _candidate_ids(
            self.challenger_candidate_ids,
            "challenger_candidate_ids",
            allow_empty=True,
        )
        if set(self.backbone_candidate_ids).intersection(
            self.challenger_candidate_ids
        ):
            raise ValueError("backbone and challenger candidate sets must be disjoint")
        _canonical_float(self.backbone_utility, "backbone_utility")
        _canonical_float(self.union_utility, "union_utility")
        if self.union_utility < self.backbone_utility:
            raise ValueError("additive union utility cannot trail its backbone")

    @property
    def complement(self) -> float:
        return self.union_utility - self.backbone_utility

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "backbone_state_before_sha256": self.backbone_state_before_sha256,
            "backbone_ask_receipt_sha256": self.backbone_ask_receipt_sha256,
            "backbone_outcome_receipt_sha256": (
                self.backbone_outcome_receipt_sha256
            ),
            "backbone_state_after_sha256": self.backbone_state_after_sha256,
            "backbone_candidate_ids": [
                value.value for value in self.backbone_candidate_ids
            ],
            "challenger_candidate_ids": [
                value.value for value in self.challenger_candidate_ids
            ],
            "utility_definition_sha256": self.utility_definition_sha256,
            "backbone_utility_hex": self.backbone_utility.hex(),
            "union_utility_hex": self.union_utility.hex(),
            "complement_hex": self.complement.hex(),
            "backbone_state_excludes_challenger_outcomes": True,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CEILING_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(slots=True)
class EarnedLineageLedger:
    """Transactional ticket ledger driven only by authenticated prior credit."""

    policy: EarnedLineagePolicy = field(default_factory=EarnedLineagePolicy)
    _provenance: dict[CandidateId, CandidateProposalProvenance] = field(
        init=False,
        default_factory=dict,
    )
    _credits: list[ConservedStageCreditReceipt] = field(
        init=False,
        default_factory=list,
    )
    _tickets: dict[str, ReproductionTicket] = field(
        init=False,
        default_factory=dict,
    )
    _consumed: set[str] = field(init=False, default_factory=set)
    _reservations: dict[str, PreparedReproductionReservation] = field(
        init=False,
        default_factory=dict,
    )
    _reserved_ticket_ids: set[str] = field(init=False, default_factory=set)
    _commits: list[ReproductionReservationCommit] = field(
        init=False,
        default_factory=list,
    )
    _version: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if type(self.policy) is not EarnedLineagePolicy:
            raise TypeError("policy must be an exact EarnedLineagePolicy")
        EarnedLineagePolicy.__post_init__(self.policy)

    @property
    def version(self) -> int:
        return self._version

    def register(self, values: Iterable[CandidateProposalProvenance]) -> None:
        batch = tuple(values)
        if not batch:
            raise ValueError("provenance registration batch cannot be empty")
        for value in batch:
            if type(value) is not CandidateProposalProvenance:
                raise TypeError("provenance batch contains an invalid value")
            CandidateProposalProvenance.__post_init__(value)
        if len({value.candidate_id for value in batch}) != len(batch):
            raise ValueError("provenance registration batch repeats a candidate")
        known = set(self._provenance)
        batch_ids = {value.candidate_id for value in batch}
        batch_by_id = {value.candidate_id: value for value in batch}
        for value in batch:
            missing = set(value.parent_candidate_ids) - known - batch_ids
            if missing:
                raise ValueError("proposal provenance names unknown parents")
            for parent_id in value.parent_candidate_ids:
                # Do not use ``dict.get(parent_id, batch_by_id[parent_id])``:
                # Python evaluates the default expression eagerly, so that
                # form raises for a parent that is already registered but is
                # intentionally absent from the current batch.
                if parent_id in self._provenance:
                    parent = self._provenance[parent_id]
                else:
                    parent = batch_by_id[parent_id]
                if parent.generation >= value.generation:
                    raise ValueError(
                        "proposal parents must come from an earlier generation"
                    )
            previous = self._provenance.get(value.candidate_id)
            if previous is not None and previous != value:
                raise ValueError("candidate provenance is immutable")
        changed = False
        for value in batch:
            if value.candidate_id not in self._provenance:
                self._provenance[value.candidate_id] = value
                changed = True
        if changed:
            self._version += 1

    def observe(
        self,
        receipt: ConservedStageCreditReceipt,
    ) -> ReproductionTicketIssuance:
        if type(receipt) is not ConservedStageCreditReceipt:
            raise TypeError("receipt must be an exact ConservedStageCreditReceipt")
        ConservedStageCreditReceipt.__post_init__(receipt)
        if self._credits and receipt.generation <= self._credits[-1].generation:
            raise ValueError("stage credit generations must increase strictly")
        unknown = {
            value.candidate_id
            for value in receipt.candidate_credits
            if value.candidate_id not in self._provenance
        }
        if unknown:
            raise ValueError("stage credit names candidates without provenance")
        eligible = [
            value
            for value in receipt.candidate_credits
            if value.contribution > self.policy.positive_credit_floor
            and self._provenance[value.candidate_id].source_role
            is ProposalLineageRole.CHALLENGER
        ]
        eligible.sort(
            key=lambda value: (-value.contribution, value.candidate_id.value)
        )
        tickets: list[ReproductionTicket] = []
        for credit in eligible:
            open_count = sum(
                value.candidate_id == credit.candidate_id
                and ticket_id not in self._consumed
                and value.expires_after_generation >= receipt.generation + 1
                for ticket_id, value in self._tickets.items()
            )
            if open_count >= self.policy.max_open_tickets_per_candidate:
                continue
            ticket = ReproductionTicket(
                candidate_id=credit.candidate_id,
                issued_after_generation=receipt.generation,
                eligible_generation=receipt.generation + 1,
                expires_after_generation=(
                    receipt.generation + self.policy.ticket_ttl_generations
                ),
                conserved_credit=credit.contribution,
                stage_credit_receipt_sha256=receipt.receipt_sha256,
                policy_definition_sha256=self.policy.definition_sha256,
            )
            if ticket.ticket_sha256 in self._tickets:
                raise RuntimeError("ticket identity collision")
            tickets.append(ticket)
            if len(tickets) == self.policy.max_tickets_per_stage:
                break
        issuance = ReproductionTicketIssuance(
            generation=receipt.generation,
            stage_credit_receipt_sha256=receipt.receipt_sha256,
            policy_definition_sha256=self.policy.definition_sha256,
            tickets=tuple(tickets),
        )
        self._credits.append(receipt)
        self._tickets.update({value.ticket_sha256: value for value in tickets})
        self._version += 1
        return issuance

    def prepare(
        self,
        *,
        generation: int,
        available_candidate_ids: tuple[CandidateId, ...],
        maximum_tickets: int,
    ) -> PreparedReproductionReservation:
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be positive")
        _candidate_ids(
            available_candidate_ids,
            "available_candidate_ids",
            allow_empty=False,
        )
        if type(maximum_tickets) is not int or maximum_tickets <= 0:
            raise ValueError("maximum_tickets must be positive")
        available = set(available_candidate_ids)
        tickets = [
            value
            for ticket_id, value in self._tickets.items()
            if ticket_id not in self._consumed
            and ticket_id not in self._reserved_ticket_ids
            and value.candidate_id in available
            and value.eligible_generation <= generation
            <= value.expires_after_generation
        ]
        tickets.sort(
            key=lambda value: (
                -value.conserved_credit,
                value.issued_after_generation,
                value.candidate_id.value,
                value.ticket_sha256,
            )
        )
        selected = tuple(tickets[:maximum_tickets])
        available_sha256 = hashlib.sha256(
            _canonical_bytes(sorted(value.value for value in available_candidate_ids))
        ).hexdigest()
        reservation = PreparedReproductionReservation(
            generation=generation,
            ledger_version=self._version,
            available_candidate_ids_sha256=available_sha256,
            tickets=selected,
            policy_definition_sha256=self.policy.definition_sha256,
        )
        if reservation.tickets:
            if reservation.reservation_sha256 in self._reservations:
                raise RuntimeError("reservation identity collision")
            self._reservations[reservation.reservation_sha256] = reservation
            self._reserved_ticket_ids.update(
                value.ticket_sha256 for value in reservation.tickets
            )
        return reservation

    def abort(self, reservation: PreparedReproductionReservation) -> None:
        reservation.__post_init__()
        if not reservation.tickets:
            return
        known = self._reservations.pop(reservation.reservation_sha256, None)
        if known != reservation:
            raise ValueError("reservation is absent, stale, or already closed")
        self._reserved_ticket_ids.difference_update(
            value.ticket_sha256 for value in reservation.tickets
        )

    def commit(
        self,
        reservation: PreparedReproductionReservation,
        *,
        parent_exposure_receipt_sha256: str,
    ) -> ReproductionReservationCommit:
        reservation.__post_init__()
        require_sha256(
            parent_exposure_receipt_sha256,
            "parent_exposure_receipt_sha256",
        )
        if not reservation.tickets:
            raise ValueError("an empty reservation has nothing to commit")
        known = self._reservations.pop(reservation.reservation_sha256, None)
        if known != reservation:
            raise ValueError("reservation is absent, stale, or already closed")
        ticket_ids = tuple(value.ticket_sha256 for value in reservation.tickets)
        if any(value in self._consumed for value in ticket_ids):
            raise RuntimeError("a reserved ticket was consumed concurrently")
        commit = ReproductionReservationCommit(
            reservation_sha256=reservation.reservation_sha256,
            generation=reservation.generation,
            consumed_ticket_sha256s=ticket_ids,
            parent_exposure_receipt_sha256=parent_exposure_receipt_sha256,
        )
        self._reserved_ticket_ids.difference_update(ticket_ids)
        self._consumed.update(ticket_ids)
        self._commits.append(commit)
        self._version += 1
        return commit

    def to_record(self, *, current_generation: int) -> dict[str, object]:
        if type(current_generation) is not int or current_generation < 0:
            raise ValueError("current_generation must be non-negative")
        expired = {
            ticket_id
            for ticket_id, value in self._tickets.items()
            if value.expires_after_generation < current_generation
            and ticket_id not in self._consumed
        }
        return {
            "schema_version": 1,
            "policy": self.policy.to_record(),
            "ledger_version": self._version,
            "provenance": [
                value.to_record()
                for value in sorted(
                    self._provenance.values(),
                    key=lambda item: item.candidate_id.value,
                )
            ],
            "stage_credits": [value.to_record() for value in self._credits],
            "tickets": [
                value.to_record()
                for value in sorted(
                    self._tickets.values(),
                    key=lambda item: item.ticket_sha256,
                )
            ],
            "consumed_ticket_sha256s": sorted(self._consumed),
            "expired_unconsumed_ticket_sha256s": sorted(expired),
            "open_reservations": [
                value.to_record()
                for value in sorted(
                    self._reservations.values(),
                    key=lambda item: item.reservation_sha256,
                )
            ],
            "commits": [value.to_record() for value in self._commits],
        }


__all__ = [
    "CandidateConservedCredit",
    "CandidateProposalProvenance",
    "ConservedBackboneCeilingReceipt",
    "ConservedStageCreditReceipt",
    "EarnedLineageLedger",
    "EarnedLineagePolicy",
    "EarnedParentAllocation",
    "PreparedReproductionReservation",
    "ProposalLineageRole",
    "ReproductionReservationCommit",
    "ReproductionTicket",
    "ReproductionTicketIssuance",
    "allocate_earned_parents",
]
