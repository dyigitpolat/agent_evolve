"""Outcome-blind structural reservations inside a model-proposed slate.

The model normally chooses a K-member proposal from a larger finite action
contract.  Complete K8 outcome panels show that this truncation can remove the
useful action before the evaluator allocator ever sees it.  This policy
reserves two *proposal* positions for archive novelty and structural coverage.
The reservations do not force an evaluator slot and are not quality rankings;
the downstream allocator retains authority over the evaluated subset.

All inputs are sealed before provider dispatch.  The policy reads no objective
value, evaluator outcome, workload identifier, model profile, action name, or
free-form memory text.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.patch import require_sha256


POLICY_ID = "structural_proposal_support_reservation"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:structural-proposal-support-reservation:v1;"
    b"reservation-count=2;roles=archive-novelty,structural-coverage;"
    b"first=lexicographic(novelty,coverage);"
    b"second=prefer-distinct-family-and-locus-then-family-then-locus;"
    b"proposal-reservation-only=true;evaluator-slot-authority=false;"
    b"objective-values=false;outcomes=false;workload-and-model-fields=false"
).hexdigest()

_DECISION_DOMAIN = b"agent-evolve:proposal-support-reservation:decision:v1\x00"
_RESERVATION_COUNT = 2


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _finite_unit(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProposalSupportCandidate:
    """One pre-provider structural row from the common candidate universe."""

    option_id: str
    option_identity_sha256: str
    family: str
    locus_key: str
    phenotype_identity_sha256: str
    frozen_archive_snapshot_sha256: str
    structural_evidence_receipt_sha256: str
    archive_novelty_score: float
    structural_coverage_score: float

    def __post_init__(self) -> None:
        for name in ("option_id", "family", "locus_key"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "option_identity_sha256",
            "phenotype_identity_sha256",
            "frozen_archive_snapshot_sha256",
            "structural_evidence_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _finite_unit(self.archive_novelty_score, name="archive_novelty_score")
        _finite_unit(
            self.structural_coverage_score,
            name="structural_coverage_score",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "locus_key": self.locus_key,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "frozen_archive_snapshot_sha256": (
                self.frozen_archive_snapshot_sha256
            ),
            "structural_evidence_receipt_sha256": (
                self.structural_evidence_receipt_sha256
            ),
            "archive_novelty_score_hex": self.archive_novelty_score.hex(),
            "structural_coverage_score_hex": (
                self.structural_coverage_score.hex()
            ),
        }


class ProposalSupportRole(str, Enum):
    """Why one action is protected in the proposal rather than the evaluator."""

    ARCHIVE_NOVELTY = "archive_novelty"
    STRUCTURAL_COVERAGE = "structural_coverage"


@dataclass(frozen=True, slots=True)
class ProposalSupportReservation:
    """One selected proposal-only reservation and its exact structural row."""

    role: ProposalSupportRole
    candidate: ProposalSupportCandidate

    def __post_init__(self) -> None:
        if type(self.role) is not ProposalSupportRole:
            raise TypeError("role must be an exact ProposalSupportRole")
        if type(self.candidate) is not ProposalSupportCandidate:
            raise TypeError("candidate must be an exact ProposalSupportCandidate")
        self.candidate.__post_init__()

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"role": self.role.value, "candidate": self.candidate.to_record()}


def _select_reservations(
    candidates: tuple[ProposalSupportCandidate, ...],
) -> tuple[ProposalSupportReservation, ProposalSupportReservation]:
    first = min(
        candidates,
        key=lambda value: (
            -value.archive_novelty_score,
            -value.structural_coverage_score,
            value.option_id,
        ),
    )
    remaining = tuple(value for value in candidates if value != first)
    preference_pools = (
        tuple(
            value
            for value in remaining
            if value.family != first.family and value.locus_key != first.locus_key
        ),
        tuple(value for value in remaining if value.family != first.family),
        tuple(value for value in remaining if value.locus_key != first.locus_key),
        remaining,
    )
    second_pool = next(value for value in preference_pools if value)
    second = min(
        second_pool,
        key=lambda value: (
            -value.structural_coverage_score,
            -value.archive_novelty_score,
            value.option_id,
        ),
    )
    return (
        ProposalSupportReservation(ProposalSupportRole.ARCHIVE_NOVELTY, first),
        ProposalSupportReservation(ProposalSupportRole.STRUCTURAL_COVERAGE, second),
    )


@dataclass(frozen=True, slots=True, eq=False)
class ProposalSupportDecision:
    """Replayable proposal-reservation decision over one exact common pool."""

    request_sha256: str
    common_candidate_pool_decision_sha256: str
    model_selection_size: int
    candidates: tuple[ProposalSupportCandidate, ...]
    reservations: tuple[ProposalSupportReservation, ...]
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    policy_definition_sha256: str = POLICY_DEFINITION_SHA256
    decision_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(
            self.common_candidate_pool_decision_sha256,
            "common_candidate_pool_decision_sha256",
        )
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.policy_definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("proposal-support policy identity drifted")
        if type(self.model_selection_size) is not int or self.model_selection_size < 4:
            raise ValueError("model_selection_size must be an exact integer >= 4")
        if type(self.candidates) is not tuple or len(self.candidates) < (
            _RESERVATION_COUNT
        ):
            raise ValueError("candidates must contain at least two exact rows")
        if any(type(value) is not ProposalSupportCandidate for value in self.candidates):
            raise TypeError("candidates must contain exact structural rows")
        for value in self.candidates:
            value.__post_init__()
        if self.candidates != tuple(
            sorted(self.candidates, key=lambda value: value.option_id)
        ):
            raise ValueError("candidates must preserve canonical option order")
        if len({value.option_id for value in self.candidates}) != len(self.candidates):
            raise ValueError("candidates cannot repeat an option")
        if len(self.candidates) < self.model_selection_size:
            raise ValueError("model selection exceeds the candidate universe")
        snapshots = {
            value.frozen_archive_snapshot_sha256 for value in self.candidates
        }
        if len(snapshots) != 1:
            raise ValueError("proposal-support candidates must share one archive cutoff")
        if type(self.reservations) is not tuple or len(self.reservations) != (
            _RESERVATION_COUNT
        ):
            raise ValueError("reservations must contain the exact two roles")
        if any(
            type(value) is not ProposalSupportReservation
            for value in self.reservations
        ):
            raise TypeError("reservations must contain exact reservation rows")
        for value in self.reservations:
            value.__post_init__()
        if tuple(value.role for value in self.reservations) != tuple(
            ProposalSupportRole
        ):
            raise ValueError("reservations must preserve the closed role order")
        if len({value.candidate.option_id for value in self.reservations}) != (
            _RESERVATION_COUNT
        ):
            raise ValueError("proposal-support reservations must be distinct")
        if self.reservations != _select_reservations(self.candidates):
            raise ValueError("reservations differ from the structural policy")
        computed = hashlib.sha256(
            _DECISION_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.decision_sha256 not in ("", computed):
            raise ValueError("decision_sha256 does not authenticate the decision")
        object.__setattr__(self, "decision_sha256", computed)

    @property
    def required_option_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(
            sorted(value.candidate.option_id for value in self.reservations)
        )

    @property
    def membership_constraint_effective(self) -> bool:
        self.__post_init__()
        return len(self.candidates) > self.model_selection_size

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "common_candidate_pool_decision_sha256": (
                self.common_candidate_pool_decision_sha256
            ),
            "model_selection_size": self.model_selection_size,
            "candidates": [value.to_record() for value in self.candidates],
            "reservations": [value.to_record() for value in self.reservations],
            "required_option_ids": [
                value.candidate.option_id for value in self.reservations
            ],
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "evaluator_slot_authority": False,
            "objective_or_outcome_values_consulted": False,
            "workload_or_model_fields_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def to_prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "decision_sha256": self.decision_sha256,
            "required_option_ids": list(self.required_option_ids),
            "reservation_roles": [value.role.value for value in self.reservations],
            "reservations_are_quality_rankings": False,
            "reservations_force_evaluator_slots": False,
            "model_may_rank_reserved_options_anywhere": True,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ProposalSupportDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class StructuralProposalSupportPolicy:
    """Reserve two structurally complementary candidates in the model slate."""

    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("proposal-support policy identity drifted")

    def select(
        self,
        *,
        request_sha256: str,
        common_candidate_pool_decision_sha256: str,
        model_selection_size: int,
        candidates: tuple[ProposalSupportCandidate, ...],
    ) -> ProposalSupportDecision:
        self.__post_init__()
        canonical = tuple(sorted(candidates, key=lambda value: value.option_id))
        return ProposalSupportDecision(
            request_sha256=request_sha256,
            common_candidate_pool_decision_sha256=(
                common_candidate_pool_decision_sha256
            ),
            model_selection_size=model_selection_size,
            candidates=canonical,
            reservations=_select_reservations(canonical),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "reservation_count": _RESERVATION_COUNT,
            "evaluator_slot_authority": False,
        }


__all__ = [
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "ProposalSupportCandidate",
    "ProposalSupportDecision",
    "ProposalSupportReservation",
    "ProposalSupportRole",
    "StructuralProposalSupportPolicy",
]
