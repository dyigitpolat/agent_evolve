"""Workload-neutral complete blocks for diagnostic insight exposure.

A campaign may admit one or more quarantined cards after reflection, while its
portfolio stage exposes a fixed number of independently assigned lanes.  This
module fills any unused lanes with preregistered control cards and constructs a
complete singleton-subset block: each eligible card is shown exactly once and
withheld from every other lane.  Consequently every card has treated and
control support without depending on provider completion order or outcomes.

The caller owns cohort selection, lifecycle admission, stable lane identities,
and the externally generated permutation rank.  The planner owns only the
portable experimental design.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.application.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSemantics,
    derive_portfolio_memory_dose_card_support,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.balanced_subset_blocks import (
    BalancedSubsetBlockPlan,
    BalancedSubsetBlockPlanner,
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.compatibility_matching import (
    CanonicalLaneCardMatchingPlanner,
    LaneCardCompatibility,
    LaneCardMatchingCard,
    LaneCardMatchingInput,
    LaneCardMatchingLane,
    LaneCardMatchingReceipt,
)
from agent_evolve.policies.memory.staged_causal import CausalSearchScorePolicy
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSupport,
)


CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_ID = (
    "campaign_diagnostic_singleton_complete_block"
)
CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_VERSION = 1
CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:campaign-diagnostic-singleton-complete-block:v1:"
    b"active-quarantine-plus-preregistered-controls:one-treatment-per-lane:"
    b"complete-balanced-subset-block:external-permutation-rank"
).hexdigest()

CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_ID = (
    "campaign_diagnostic_complete_bipartite_support_audit"
)
CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_VERSION = 2
CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:campaign-diagnostic-complete-bipartite-support-audit:v2:"
    b"prospective-positive-lane-card-graph:equal-card-and-lane-cardinality:"
    b"minimum-two-units:every-card-compatible-with-every-lane:"
    b"eligible-for-balanced-randomized-singleton-assignment-or-fail-closed:"
    b"assignment-audit-is-not-a-card-vs-neutral-causal-effect"
).hexdigest()
CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_ID = (
    "campaign_diagnostic_complete_support_cohort_selection"
)
CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_VERSION = 1
CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:campaign-diagnostic-complete-support-cohort-selection:v1:"
    b"prospective-positive-lane-card-graph:card-supported-by-every-lane:"
    b"externally-keyed-card-order:select-exact-lane-count:"
    b"provider-outcome-and-workload-neutral:fail-closed"
).hexdigest()
_COMPATIBILITY_AUDIT_DOMAIN = (
    b"agent-evolve:campaign-diagnostic-compatibility-audit:v1\x00"
)
_COHORT_SELECTION_DOMAIN = (
    b"agent-evolve:campaign-diagnostic-cohort-selection:v1\x00"
)
_COHORT_CARD_ORDER_DOMAIN = (
    b"agent-evolve:campaign-diagnostic-cohort-card-order:v1\x00"
)
CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_ID = (
    "campaign_diagnostic_complete_support_resolution"
)
CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_VERSION = 1
CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:campaign-diagnostic-complete-support-resolution:v1:"
    b"typed-current-finite-contract-per-stable-lane:typed-card-semantics:"
    b"prospective-positive-support-derivation:outcome-blind-complete-support-"
    b"cohort-selection:canonical-matching:complete-bipartite-audit:fail-closed"
).hexdigest()
_SUPPORT_RESOLUTION_DOMAIN = (
    b"agent-evolve:campaign-diagnostic-support-resolution:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


class CampaignDiagnosticCompatibilityStatus(str, Enum):
    """Whether the positive support graph admits the declared randomization."""

    ELIGIBLE = "eligible_complete_bipartite_support"
    INELIGIBLE = "ineligible_fail_closed_no_causal_credit"


class CampaignDiagnosticCohortSelectionStatus(str, Enum):
    """Whether enough universally supported cards form one complete block."""

    ELIGIBLE = "eligible_complete_support_cohort_selected"
    INELIGIBLE = "ineligible_insufficient_complete_support_cards"


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticSupportLaneInput:
    """One stable lane and its current, parent-bound finite action contract."""

    lane: LaneCardMatchingLane
    finite_variation_contract: FiniteVariationContract

    def __post_init__(self) -> None:
        if type(self.lane) is not LaneCardMatchingLane:
            raise TypeError("lane must be an exact LaneCardMatchingLane")
        self.lane.__post_init__()
        if type(self.finite_variation_contract) is not FiniteVariationContract:
            raise TypeError(
                "finite_variation_contract must be an exact FiniteVariationContract"
            )
        validate_finite_variation_contract(self.finite_variation_contract)


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticSupportCardInput:
    """One source-bound card and its trusted executable action semantics."""

    card: LaneCardMatchingCard
    semantics: PortfolioMemoryDoseCardSemantics

    def __post_init__(self) -> None:
        if type(self.card) is not LaneCardMatchingCard:
            raise TypeError("card must be an exact LaneCardMatchingCard")
        self.card.__post_init__()
        if type(self.semantics) is not PortfolioMemoryDoseCardSemantics:
            raise TypeError(
                "semantics must be exact PortfolioMemoryDoseCardSemantics"
            )
        self.semantics.__post_init__()
        if self.card.card_key != self.semantics.card_key:
            raise ValueError("matching card and executable semantics differ")


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticDerivedSupportEdge:
    """One positive lane/card edge and its exact finite-option support."""

    lane_id: str
    card_key: str
    support: PortfolioMemoryDoseCardSupport

    def __post_init__(self) -> None:
        if type(self.support) is not PortfolioMemoryDoseCardSupport:
            raise TypeError("support must be exact PortfolioMemoryDoseCardSupport")
        self.support.__post_init__()
        if self.support.card_key != self.card_key:
            raise ValueError("support edge and support card key differ")

    @property
    def evidence_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.support.to_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "lane_id": self.lane_id,
            "card_key": self.card_key,
            "compatibility_evidence_sha256": self.evidence_sha256,
            "support": self.support.to_record(),
        }


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticRejectedSupportEdge:
    """One prospectively incompatible lane/card pair, without raw error text."""

    lane_id: str
    card_key: str
    reason_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.reason_sha256, "reason_sha256")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "lane_id": self.lane_id,
            "card_key": self.card_key,
            "status": "incompatible",
            "reason_sha256": self.reason_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticCompleteSupportCohortSelection:
    """Select a randomized, replayable complete-support card cohort.

    Reflection may yield more candidate insights than a diagnostic wave has
    stable lanes.  Compatibility is necessarily parent-local, so selecting
    the experimental cohort before those parents exist either broadens action
    semantics or creates positivity violations.  This policy filters only on
    the prospective positive compatibility graph, then uses an external key to
    sample exactly one lane-sized cohort before provider dispatch.
    """

    matching_input: LaneCardMatchingInput
    cohort_size: int
    selection_key_sha256: str
    full_support_card_keys: tuple[str, ...] = field(init=False)
    selected_card_keys: tuple[str, ...] = field(init=False)
    selected_matching_input: LaneCardMatchingInput | None = field(init=False)
    status: CampaignDiagnosticCohortSelectionStatus = field(init=False)
    receipt_sha256: str = field(init=False)
    policy_id: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if type(self.matching_input) is not LaneCardMatchingInput:
            raise TypeError("matching_input must be exact LaneCardMatchingInput")
        self.matching_input.__post_init__()
        if type(self.cohort_size) is not int or self.cohort_size < 2:
            raise ValueError("cohort_size must be an exact integer of at least two")
        if self.cohort_size != len(self.matching_input.lanes):
            raise ValueError("cohort_size must exactly fill the stable lanes")
        require_sha256(self.selection_key_sha256, "selection_key_sha256")
        positive = {
            (value.lane_id, value.card_key)
            for value in self.matching_input.compatibilities
        }
        lane_ids = tuple(value.lane_id for value in self.matching_input.lanes)
        full_support = tuple(
            card.card_key
            for card in self.matching_input.cards
            if all((lane_id, card.card_key) in positive for lane_id in lane_ids)
        )
        object.__setattr__(self, "full_support_card_keys", full_support)
        eligible = len(full_support) >= self.cohort_size
        if eligible:
            cards_by_key = {
                value.card_key: value for value in self.matching_input.cards
            }
            ordered = tuple(
                sorted(
                    full_support,
                    key=lambda card_key: (
                        hashlib.sha256(
                            _COHORT_CARD_ORDER_DOMAIN
                            + bytes.fromhex(self.selection_key_sha256)
                            + card_key.encode("ascii", errors="strict")
                            + bytes.fromhex(
                                cards_by_key[card_key].card_identity_sha256
                            )
                        ).digest(),
                        card_key,
                    ),
                )
            )
            selected_set = frozenset(ordered[: self.cohort_size])
            selected_keys = tuple(sorted(selected_set))
            selected_input = LaneCardMatchingInput(
                lanes=self.matching_input.lanes,
                cards=tuple(
                    value
                    for value in self.matching_input.cards
                    if value.card_key in selected_set
                ),
                compatibilities=tuple(
                    value
                    for value in self.matching_input.compatibilities
                    if value.card_key in selected_set
                ),
            )
            status = CampaignDiagnosticCohortSelectionStatus.ELIGIBLE
        else:
            selected_keys = ()
            selected_input = None
            status = CampaignDiagnosticCohortSelectionStatus.INELIGIBLE
        object.__setattr__(self, "selected_card_keys", selected_keys)
        object.__setattr__(self, "selected_matching_input", selected_input)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "receipt_sha256",
            hashlib.sha256(
                _COHORT_SELECTION_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def eligible(self) -> bool:
        return self.status is CampaignDiagnosticCohortSelectionStatus.ELIGIBLE

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_definition_sha256": self.policy_definition_sha256,
            },
            "matching_input_sha256": self.matching_input.input_sha256,
            "compatibility_graph_sha256": (
                self.matching_input.compatibility_graph_sha256
            ),
            "cohort_size": self.cohort_size,
            "selection_key_sha256": self.selection_key_sha256,
            "full_support_card_keys": list(self.full_support_card_keys),
            "selected_card_keys": list(self.selected_card_keys),
            "status": self.status.value,
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticCompatibilityAudit:
    """Authenticate complete support before randomized card assignment.

    A maximum matching is enough to administer one compatible card per lane,
    but it is not enough to randomize every card over every lane.  This audit
    closes that distinction.  The existing balanced singleton law may be used
    only when lane/card cardinality is equal, at least two experimental units
    exist, and the prospective positive graph is complete bipartite.

    Passing this audit authorizes the assignment law.  It does not identify a
    card-versus-no-memory effect: every lane may still receive another active
    card, and one realized block supplies no matched neutral potential outcome.
    """

    matching_input: LaneCardMatchingInput
    status: CampaignDiagnosticCompatibilityStatus = field(init=False)
    missing_pairs: tuple[tuple[str, str], ...] = field(init=False)
    audit_sha256: str = field(init=False)
    policy_id: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if type(self.matching_input) is not LaneCardMatchingInput:
            raise TypeError("matching_input must be exact LaneCardMatchingInput")
        self.matching_input.__post_init__()
        observed = {
            (value.lane_id, value.card_key)
            for value in self.matching_input.compatibilities
        }
        expected = tuple(
            (lane.lane_id, card.card_key)
            for lane in self.matching_input.lanes
            for card in self.matching_input.cards
        )
        missing = tuple(value for value in expected if value not in observed)
        eligible = (
            len(self.matching_input.lanes) >= 2
            and len(self.matching_input.lanes) == len(self.matching_input.cards)
            and not missing
        )
        object.__setattr__(self, "missing_pairs", missing)
        object.__setattr__(
            self,
            "status",
            (
                CampaignDiagnosticCompatibilityStatus.ELIGIBLE
                if eligible
                else CampaignDiagnosticCompatibilityStatus.INELIGIBLE
            ),
        )
        object.__setattr__(
            self,
            "audit_sha256",
            hashlib.sha256(
                _COMPATIBILITY_AUDIT_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def eligible(self) -> bool:
        return self.status is CampaignDiagnosticCompatibilityStatus.ELIGIBLE

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_definition_sha256": self.policy_definition_sha256,
            },
            "matching_input_sha256": self.matching_input.input_sha256,
            "compatibility_graph_sha256": (
                self.matching_input.compatibility_graph_sha256
            ),
            "lane_count": len(self.matching_input.lanes),
            "card_count": len(self.matching_input.cards),
            "observed_positive_edge_count": len(
                self.matching_input.compatibilities
            ),
            "required_complete_edge_count": (
                len(self.matching_input.lanes) * len(self.matching_input.cards)
            ),
            "minimum_two_units": len(self.matching_input.lanes) >= 2,
            "balanced_cardinality": (
                len(self.matching_input.lanes) == len(self.matching_input.cards)
            ),
            "complete_bipartite_support": not self.missing_pairs,
            "missing_pairs": [
                {"lane_id": lane_id, "card_key": card_key}
                for lane_id, card_key in self.missing_pairs
            ],
            "status": self.status.value,
            "randomized_singleton_assignment_allowed": self.eligible,
            "card_vs_neutral_effect_identified": False,
            "causal_credit_allowed": False,
            "online_score_update_allowed": False,
            "required_successor_design": (
                "prospective_randomized_prompt_redacted_matched_control"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "audit_sha256": self.audit_sha256}


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticCompleteSupportResolution:
    """Authenticated prospective resolution of one lane/card diagnostic block."""

    full_matching_input: LaneCardMatchingInput
    cohort_selection: CampaignDiagnosticCompleteSupportCohortSelection
    matching: LaneCardMatchingReceipt
    compatibility_audit: CampaignDiagnosticCompatibilityAudit
    supports: tuple[CampaignDiagnosticDerivedSupportEdge, ...]
    rejected_edges: tuple[CampaignDiagnosticRejectedSupportEdge, ...]
    policy_id: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if type(self.full_matching_input) is not LaneCardMatchingInput:
            raise TypeError("full_matching_input must be exact LaneCardMatchingInput")
        self.full_matching_input.__post_init__()
        if type(self.cohort_selection) is not (
            CampaignDiagnosticCompleteSupportCohortSelection
        ):
            raise TypeError("cohort_selection must be exact")
        self.cohort_selection.__post_init__()
        if self.cohort_selection.matching_input != self.full_matching_input:
            raise ValueError("cohort selection differs from the full support graph")
        if type(self.matching) is not LaneCardMatchingReceipt:
            raise TypeError("matching must be an exact LaneCardMatchingReceipt")
        self.matching.__post_init__()
        selected_input = self.cohort_selection.selected_matching_input
        expected_matching_input = (
            self.full_matching_input if selected_input is None else selected_input
        )
        if self.matching.matching_input != expected_matching_input:
            raise ValueError("matching differs from selected diagnostic population")
        if type(self.compatibility_audit) is not CampaignDiagnosticCompatibilityAudit:
            raise TypeError("compatibility_audit must be exact")
        self.compatibility_audit.__post_init__()
        if self.compatibility_audit.matching_input != self.matching.matching_input:
            raise ValueError("compatibility audit differs from the matching population")
        if type(self.supports) is not tuple or any(
            type(value) is not CampaignDiagnosticDerivedSupportEdge
            for value in self.supports
        ):
            raise TypeError("supports must be an exact support-edge tuple")
        for value in self.supports:
            value.__post_init__()
        support_keys = tuple((value.lane_id, value.card_key) for value in self.supports)
        if support_keys != tuple(sorted(set(support_keys))):
            raise ValueError("supports must use unique canonical lane/card order")
        graph_keys = tuple(
            (value.lane_id, value.card_key)
            for value in self.full_matching_input.compatibilities
        )
        if support_keys != graph_keys:
            raise ValueError("derived supports differ from positive graph edges")
        if type(self.rejected_edges) is not tuple or any(
            type(value) is not CampaignDiagnosticRejectedSupportEdge
            for value in self.rejected_edges
        ):
            raise TypeError("rejected_edges must be an exact tuple")
        for value in self.rejected_edges:
            value.__post_init__()
        rejected_keys = tuple(
            (value.lane_id, value.card_key) for value in self.rejected_edges
        )
        if rejected_keys != tuple(sorted(set(rejected_keys))):
            raise ValueError("rejected edges must use unique canonical lane/card order")
        if set(support_keys).intersection(rejected_keys):
            raise ValueError("one lane/card pair cannot be supported and rejected")
        all_pairs = {
            (lane.lane_id, card.card_key)
            for lane in self.full_matching_input.lanes
            for card in self.full_matching_input.cards
        }
        if set((*support_keys, *rejected_keys)) != all_pairs:
            raise ValueError("support resolution must classify every lane/card pair")
        if (
            self.policy_id != CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_ID
            or self.policy_version
            != CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_VERSION
            or self.policy_definition_sha256
            != CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported complete-support resolution policy")

    @property
    def eligible(self) -> bool:
        self.__post_init__()
        return (
            self.cohort_selection.eligible
            and self.matching.is_full
            and self.compatibility_audit.eligible
        )

    def support_for(
        self,
        lane_id: str,
        card_key: str,
    ) -> PortfolioMemoryDoseCardSupport:
        self.__post_init__()
        values = tuple(
            value.support
            for value in self.supports
            if value.lane_id == lane_id and value.card_key == card_key
        )
        if len(values) != 1:
            raise KeyError("lane/card pair has no derived positive support")
        return values[0]

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_definition_sha256": self.policy_definition_sha256,
            },
            "full_matching_input": self.full_matching_input.to_record(),
            "cohort_selection": self.cohort_selection.to_record(),
            "matching": self.matching.to_record(),
            "compatibility_audit": self.compatibility_audit.to_record(),
            "supports": [value.to_record() for value in self.supports],
            "rejected_edges": [
                value.to_record() for value in self.rejected_edges
            ],
            "eligible": self.eligible,
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _SUPPORT_RESOLUTION_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticCompleteSupportResolver:
    """Derive and resolve a complete-support cohort before provider dispatch."""

    def resolve(
        self,
        *,
        lanes: tuple[CampaignDiagnosticSupportLaneInput, ...],
        cards: tuple[CampaignDiagnosticSupportCardInput, ...],
        cohort_size: int,
        selection_key_sha256: str,
    ) -> CampaignDiagnosticCompleteSupportResolution:
        if type(lanes) is not tuple or not lanes or any(
            type(value) is not CampaignDiagnosticSupportLaneInput for value in lanes
        ):
            raise ValueError("lanes must be a non-empty exact support-lane tuple")
        if type(cards) is not tuple or not cards or any(
            type(value) is not CampaignDiagnosticSupportCardInput for value in cards
        ):
            raise ValueError("cards must be a non-empty exact support-card tuple")
        for value in lanes:
            value.__post_init__()
        for value in cards:
            value.__post_init__()
        lane_ids = tuple(value.lane.lane_id for value in lanes)
        card_keys = tuple(value.card.card_key for value in cards)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("support lanes must be unique and canonical")
        if card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("support cards must be unique and canonical")
        if type(cohort_size) is not int or cohort_size != len(lanes):
            raise ValueError("cohort_size must exactly equal the stable lane count")
        require_sha256(selection_key_sha256, "selection_key_sha256")

        supports: list[CampaignDiagnosticDerivedSupportEdge] = []
        rejected: list[CampaignDiagnosticRejectedSupportEdge] = []
        for lane in lanes:
            for card in cards:
                try:
                    support = derive_portfolio_memory_dose_card_support(
                        card.semantics,
                        lane.finite_variation_contract,
                    )
                except ValueError as error:
                    rejected.append(
                        CampaignDiagnosticRejectedSupportEdge(
                            lane_id=lane.lane.lane_id,
                            card_key=card.card.card_key,
                            reason_sha256=hashlib.sha256(
                                str(error).encode("utf-8", errors="strict")
                            ).hexdigest(),
                        )
                    )
                    continue
                supports.append(
                    CampaignDiagnosticDerivedSupportEdge(
                        lane_id=lane.lane.lane_id,
                        card_key=card.card.card_key,
                        support=support,
                    )
                )
        canonical_supports = tuple(
            sorted(supports, key=lambda value: (value.lane_id, value.card_key))
        )
        canonical_rejected = tuple(
            sorted(rejected, key=lambda value: (value.lane_id, value.card_key))
        )
        matching_input = LaneCardMatchingInput(
            lanes=tuple(value.lane for value in lanes),
            cards=tuple(value.card for value in cards),
            compatibilities=tuple(
                LaneCardCompatibility(
                    lane_id=value.lane_id,
                    card_key=value.card_key,
                    compatibility_evidence_sha256=value.evidence_sha256,
                )
                for value in canonical_supports
            ),
        )
        selection = CampaignDiagnosticCompleteSupportCohortSelection(
            matching_input=matching_input,
            cohort_size=cohort_size,
            selection_key_sha256=selection_key_sha256,
        )
        selected_input = selection.selected_matching_input
        diagnostic_input = matching_input if selected_input is None else selected_input
        matching = CanonicalLaneCardMatchingPlanner().plan(
            lanes=diagnostic_input.lanes,
            cards=diagnostic_input.cards,
            compatibilities=diagnostic_input.compatibilities,
        )
        audit = CampaignDiagnosticCompatibilityAudit(matching.matching_input)
        return CampaignDiagnosticCompleteSupportResolution(
            full_matching_input=matching_input,
            cohort_selection=selection,
            matching=matching,
            compatibility_audit=audit,
            supports=canonical_supports,
            rejected_edges=canonical_rejected,
        )


def _canonical_references(
    values: tuple[InsightRef, ...],
    *,
    name: str,
    empty: bool,
) -> tuple[InsightRef, ...]:
    if type(values) is not tuple or any(
        type(value) is not InsightRef for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of InsightRef values")
    for value in values:
        InsightRef.__post_init__(value)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")
    if not empty and not values:
        raise ValueError(f"{name} must be non-empty")
    return values


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticSingletonBlock:
    """Authenticated complete singleton block over active and control cards."""

    active_references: tuple[InsightRef, ...]
    control_references: tuple[InsightRef, ...]
    assignment_plan: BalancedSubsetBlockPlan
    policy_id: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        active = _canonical_references(
            self.active_references,
            name="active_references",
            empty=False,
        )
        controls = _canonical_references(
            self.control_references,
            name="control_references",
            empty=True,
        )
        if set(active).intersection(controls):
            raise ValueError("active and control references must be disjoint")
        if type(self.assignment_plan) is not BalancedSubsetBlockPlan:
            raise TypeError("assignment_plan must be exact BalancedSubsetBlockPlan")
        self.assignment_plan.__post_init__()
        eligible = tuple(
            entry.reference for entry in self.assignment_plan.snapshot.entries
        )
        if eligible != tuple(sorted((*active, *controls))):
            raise ValueError("assignment snapshot differs from the diagnostic cohort")
        if (
            self.assignment_plan.subset_size != 1
            or self.assignment_plan.full_block_count != 1
            or self.assignment_plan.remainder_size != 0
            or len(self.assignment_plan.ordered_units) != len(eligible)
        ):
            raise ValueError(
                "diagnostic singleton design must be exactly one complete block"
            )
        if (
            self.policy_id != CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_ID
            or self.policy_version != CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_VERSION
            or self.policy_definition_sha256
            != CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported diagnostic singleton block policy")

    @property
    def plan_sha256(self) -> str:
        self.__post_init__()
        return self.assignment_plan.receipt_sha256

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "active_references": [
                {
                    "insight_id": value.insight_id.value,
                    "version": value.version,
                }
                for value in self.active_references
            ],
            "control_references": [
                {
                    "insight_id": value.insight_id.value,
                    "version": value.version,
                }
                for value in self.control_references
            ],
            "assignment_plan": self.assignment_plan.to_record(),
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticSingletonBlockPlanner:
    """Fill stable lanes and delegate exact randomization to the block policy."""

    def plan(
        self,
        *,
        active_references: tuple[InsightRef, ...],
        control_references: tuple[InsightRef, ...],
        exact_context_sha256: str,
        estimand_stratum_sha256: str,
        ordered_units: tuple[StableMemoryAssignmentUnit, ...],
        full_block_permutation_rank: int,
    ) -> CampaignDiagnosticSingletonBlock:
        active = _canonical_references(
            active_references,
            name="active_references",
            empty=False,
        )
        controls = _canonical_references(
            control_references,
            name="control_references",
            empty=True,
        )
        if set(active).intersection(controls):
            raise ValueError("active and control references must be disjoint")
        require_sha256(exact_context_sha256, "exact_context_sha256")
        require_sha256(estimand_stratum_sha256, "estimand_stratum_sha256")
        if (
            type(ordered_units) is not tuple
            or len(ordered_units) < 2
            or any(
                type(value) is not StableMemoryAssignmentUnit for value in ordered_units
            )
        ):
            raise ValueError(
                "ordered_units must contain at least two exact stable lanes"
            )
        for value in ordered_units:
            StableMemoryAssignmentUnit.__post_init__(value)
        eligible = tuple(sorted((*active, *controls)))
        if len(eligible) != len(ordered_units):
            raise ValueError(
                "active plus control references must exactly fill the stable lanes"
            )
        if type(full_block_permutation_rank) is not int or not (
            0 <= full_block_permutation_rank < math.factorial(len(eligible))
        ):
            raise ValueError("full_block_permutation_rank is outside the exact law")
        snapshot = CausalSearchScorePolicy(
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        ).genesis(
            exact_context_hash=exact_context_sha256,
            estimand_stratum_hash=estimand_stratum_sha256,
            priors={reference: 0.0 for reference in eligible},
        )
        assignment_plan = BalancedSubsetBlockPlanner().plan(
            snapshot=snapshot,
            ordered_units=ordered_units,
            subset_size=1,
            full_block_permutation_ranks=(full_block_permutation_rank,),
        )
        return CampaignDiagnosticSingletonBlock(
            active_references=active,
            control_references=controls,
            assignment_plan=assignment_plan,
        )


__all__ = [
    "CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_DEFINITION_SHA256",
    "CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_ID",
    "CAMPAIGN_DIAGNOSTIC_COHORT_SELECTION_POLICY_VERSION",
    "CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_DEFINITION_SHA256",
    "CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_ID",
    "CAMPAIGN_DIAGNOSTIC_COMPATIBILITY_AUDIT_POLICY_VERSION",
    "CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_DEFINITION_SHA256",
    "CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_ID",
    "CAMPAIGN_DIAGNOSTIC_SINGLETON_BLOCK_POLICY_VERSION",
    "CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_DEFINITION_SHA256",
    "CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_ID",
    "CAMPAIGN_DIAGNOSTIC_SUPPORT_RESOLUTION_POLICY_VERSION",
    "CampaignDiagnosticCompleteSupportResolution",
    "CampaignDiagnosticCompleteSupportResolver",
    "CampaignDiagnosticSingletonBlock",
    "CampaignDiagnosticSingletonBlockPlanner",
    "CampaignDiagnosticCohortSelectionStatus",
    "CampaignDiagnosticCompleteSupportCohortSelection",
    "CampaignDiagnosticCompatibilityAudit",
    "CampaignDiagnosticCompatibilityStatus",
    "CampaignDiagnosticDerivedSupportEdge",
    "CampaignDiagnosticRejectedSupportEdge",
    "CampaignDiagnosticSupportCardInput",
    "CampaignDiagnosticSupportLaneInput",
]
