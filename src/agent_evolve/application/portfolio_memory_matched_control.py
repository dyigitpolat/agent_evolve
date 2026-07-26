"""Prospective active-versus-neutral experiments for portfolio memory.

The ordinary memory-credit path measures the utility of a complete candidate
wave.  It cannot tell whether the visible memory card, its explicitly supported
action, or an unrelated slate member created that utility.  This module defines
the narrow workload-neutral successor experiment:

* one exact source-bound card is chosen before provider dispatch;
* two stable lane units are randomized between MEMORY and canonical NEUTRAL;
* both arms retain the same source identity and required finite candidate-pool
  actions, while the neutral prompt redacts prompt, evidence, score, and action
  compartments; and
* one realized pair is recorded as experimental evidence, never as an
  identified effect or an online score update.

The optimizer may administer a hard supported-action dose in the MEMORY arm.
If it does, the estimand is the complete memory-guided dose package versus the
same-pool neutral arm, not prose alone.  Workload adapters still own finite
contract construction and evaluation; no benchmark or provider is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from agent_evolve.application.campaign_diagnostic_blocks import (
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
)
from agent_evolve.application.portfolio_memory_dose import (
    derive_portfolio_memory_dose_card_support,
)
from agent_evolve.application.portfolio_projection import (
    bind_portfolio_experimental_view,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.balanced_subset_blocks import (
    StableMemoryAssignmentUnit,
)
from agent_evolve.ports.portfolio_selection import (
    CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
    CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
    PortfolioCard,
    PortfolioCardSourceRegistry,
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    PortfolioExperimentalViewReceipt,
    derive_portfolio_card_view,
)
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSupport,
)


PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID = (
    "portfolio_memory_randomized_active_neutral_pair"
)
PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION = 1
PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-randomized-active-neutral-pair:v1:"
    b"two-stable-lane-units;external-binary-permutation-rank;"
    b"one-source-bound-card;memory-versus-canonical-redacted-neutral;"
    b"same-required-common-pool-actions;compound-memory-dose-estimand;"
    b"lane-randomized-not-parent-or-full-pool-matched;"
    b"single-block-not-identified;no-online-score-update"
).hexdigest()
PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID = (
    "portfolio_memory_independent_exact_lane_support"
)
PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION = 1
PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-independent-exact-lane-support:v1:"
    b"one-current-lane;source-bound-cards;exact-finite-action-support;"
    b"task-lane-keyed-card-selection;provider-and-outcome-blind;"
    b"optimization-exposure-only;card-vs-neutral-effect=false;"
    b"online-causal-credit=false"
).hexdigest()
_PLAN_DOMAIN = b"agent-evolve:portfolio-memory-matched-control-plan:v1\x00"
_VIEW_DOMAIN = b"agent-evolve:portfolio-memory-matched-control-view:v1\x00"
_OUTCOME_DOMAIN = b"agent-evolve:portfolio-memory-matched-control-outcome:v2\x00"
_SUPPORT_DOMAIN = b"agent-evolve:portfolio-memory-matched-support:v1\x00"
_CARD_ORDER_DOMAIN = b"agent-evolve:portfolio-memory-matched-card-order:v1\x00"
_LANE_SUPPORT_DOMAIN = b"agent-evolve:portfolio-memory-lane-support:v1\x00"
_LANE_CARD_ORDER_DOMAIN = b"agent-evolve:portfolio-memory-lane-card-order:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _reference_record(reference: InsightRef) -> dict[str, object]:
    if type(reference) is not InsightRef:
        raise TypeError("reference must be an exact InsightRef")
    InsightRef.__post_init__(reference)
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedArmAssignment:
    """One pre-provider binary arm assignment for a stable lane unit."""

    unit: StableMemoryAssignmentUnit
    arm: PortfolioExperimentalArm
    schedule_position: int
    assignment_probability_numerator: int = field(init=False, default=1)
    assignment_probability_denominator: int = field(init=False, default=2)

    def __post_init__(self) -> None:
        if type(self.unit) is not StableMemoryAssignmentUnit:
            raise TypeError("unit must be an exact StableMemoryAssignmentUnit")
        self.unit.__post_init__()
        if self.arm not in (
            PortfolioExperimentalArm.MEMORY,
            PortfolioExperimentalArm.NEUTRAL,
        ):
            raise ValueError("matched memory assignments admit only M or N")
        if type(self.schedule_position) is not int or self.schedule_position < 0:
            raise ValueError("schedule_position must be nonnegative")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "unit": self.unit.to_record(),
            "arm": self.arm.value,
            "schedule_position": self.schedule_position,
            "assignment_probability": [
                self.assignment_probability_numerator,
                self.assignment_probability_denominator,
            ],
        }


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedControlPlan:
    """One authenticated randomized M/N lane pair for an exact card."""

    reference: InsightRef
    exact_context_sha256: str
    ordered_units: tuple[StableMemoryAssignmentUnit, ...]
    active_unit_rank: int
    assignments: tuple[PortfolioMemoryMatchedArmAssignment, ...]
    policy_id: str = field(
        init=False,
        default=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256,
    )
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        InsightRef.__post_init__(self.reference)
        require_sha256(self.exact_context_sha256, "exact_context_sha256")
        if (
            type(self.ordered_units) is not tuple
            or len(self.ordered_units) != 2
            or any(
                type(value) is not StableMemoryAssignmentUnit
                for value in self.ordered_units
            )
        ):
            raise ValueError("a matched block requires exactly two stable units")
        for value in self.ordered_units:
            value.__post_init__()
        if len({value.unit_key for value in self.ordered_units}) != 2:
            raise ValueError("matched units must have distinct identities")
        if len({value.lane_id for value in self.ordered_units}) != 2:
            raise ValueError("matched units must occupy distinct stable lanes")
        if len({value.generation for value in self.ordered_units}) != 1:
            raise ValueError("one matched block cannot span generations")
        if (
            type(self.active_unit_rank) is not int
            or self.active_unit_rank not in (0, 1)
        ):
            raise ValueError("active_unit_rank must be one exact binary rank")
        if (
            type(self.assignments) is not tuple
            or len(self.assignments) != 2
            or any(
                type(value) is not PortfolioMemoryMatchedArmAssignment
                for value in self.assignments
            )
        ):
            raise ValueError("assignments must cover exactly two units")
        expected = tuple(
            PortfolioMemoryMatchedArmAssignment(
                unit=unit,
                arm=(
                    PortfolioExperimentalArm.MEMORY
                    if position == self.active_unit_rank
                    else PortfolioExperimentalArm.NEUTRAL
                ),
                schedule_position=position,
            )
            for position, unit in enumerate(self.ordered_units)
        )
        if self.assignments != expected:
            raise ValueError("assignments do not replay from the external rank")
        if (
            self.policy_id != PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID
            or self.policy_version != PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION
            or self.policy_definition_sha256
            != PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
        ):
            raise ValueError("unsupported matched-control policy")
        object.__setattr__(
            self,
            "plan_sha256",
            _hash(_PLAN_DOMAIN, self._unsigned_record()),
        )

    @property
    def generation(self) -> int:
        return self.ordered_units[0].generation

    def assignment_for(
        self,
        *,
        generation: int,
        lane_id: str,
    ) -> PortfolioMemoryMatchedArmAssignment:
        self.__post_init__()
        matches = tuple(
            value
            for value in self.assignments
            if value.unit.generation == generation and value.unit.lane_id == lane_id
        )
        if len(matches) != 1:
            raise KeyError("generation/lane does not resolve one matched assignment")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "exact_context_sha256": self.exact_context_sha256,
            "generation": self.generation,
            "active_unit_rank": self.active_unit_rank,
            "assignments": [value.to_record() for value in self.assignments],
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "estimand": (
                "memory_guided_action_dose_package_vs_neutral_with_shared_"
                "required_actions"
            ),
            "provider_and_outcome_blind_assignment": True,
            "same_parent_matched": False,
            "full_candidate_pool_matched": False,
            "single_block_card_effect_identified": False,
            "online_score_update_allowed": False,
            "required_analysis": (
                "aggregate_repeated_precommitted_blocks_with_arm_aware_outcomes"
            ),
            "required_stronger_successor": (
                "replicated_same_parent_same_full_pool_active_neutral_slots"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedControlPlanner:
    """Map one external binary rank to a replayable M/N assignment."""

    def plan(
        self,
        *,
        reference: InsightRef,
        exact_context_sha256: str,
        ordered_units: tuple[StableMemoryAssignmentUnit, ...],
        active_unit_rank: int,
    ) -> PortfolioMemoryMatchedControlPlan:
        assignments = tuple(
            PortfolioMemoryMatchedArmAssignment(
                unit=unit,
                arm=(
                    PortfolioExperimentalArm.MEMORY
                    if position == active_unit_rank
                    else PortfolioExperimentalArm.NEUTRAL
                ),
                schedule_position=position,
            )
            for position, unit in enumerate(ordered_units)
        )
        return PortfolioMemoryMatchedControlPlan(
            reference=reference,
            exact_context_sha256=exact_context_sha256,
            ordered_units=ordered_units,
            active_unit_rank=active_unit_rank,
            assignments=assignments,
        )


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedLaneSupport:
    """Exact support for the selected source card under one current lane."""

    lane_id: str
    support: PortfolioMemoryDoseCardSupport

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or not self.lane_id:
            raise ValueError("lane_id must be a nonempty string")
        if type(self.support) is not PortfolioMemoryDoseCardSupport:
            raise TypeError("support must be exact PortfolioMemoryDoseCardSupport")
        self.support.__post_init__()

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"lane_id": self.lane_id, "support": self.support.to_record()}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedSupportResolution:
    """Outcome-blind selection of one card supported by every matched lane."""

    lane_ids: tuple[str, ...]
    eligible_card_keys: tuple[str, ...]
    selected_card_key: str | None
    selected_lane_supports: tuple[PortfolioMemoryMatchedLaneSupport, ...]
    selection_key_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.lane_ids) is not tuple
            or len(self.lane_ids) != 2
            or self.lane_ids != tuple(sorted(set(self.lane_ids)))
        ):
            raise ValueError("matched support requires two canonical lane IDs")
        if (
            type(self.eligible_card_keys) is not tuple
            or self.eligible_card_keys
            != tuple(sorted(set(self.eligible_card_keys)))
        ):
            raise ValueError("eligible card keys must be unique and canonical")
        require_sha256(self.selection_key_sha256, "selection_key_sha256")
        if self.selected_card_key is None:
            if self.eligible_card_keys or self.selected_lane_supports:
                raise ValueError("ineligible resolution cannot carry selected support")
        else:
            if self.selected_card_key not in self.eligible_card_keys:
                raise ValueError("selected card is outside the eligible set")
            if (
                type(self.selected_lane_supports) is not tuple
                or len(self.selected_lane_supports) != 2
                or tuple(value.lane_id for value in self.selected_lane_supports)
                != self.lane_ids
            ):
                raise ValueError("selected support must cover both lanes exactly")
            for value in self.selected_lane_supports:
                value.__post_init__()
                if value.support.card_key != self.selected_card_key:
                    raise ValueError("lane support names another selected card")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_SUPPORT_DOMAIN, self._unsigned_record()),
        )

    @property
    def eligible(self) -> bool:
        return self.selected_card_key is not None

    def support_for(self, lane_id: str) -> PortfolioMemoryDoseCardSupport:
        self.__post_init__()
        matches = tuple(
            value.support
            for value in self.selected_lane_supports
            if value.lane_id == lane_id
        )
        if len(matches) != 1:
            raise KeyError("lane lacks selected matched-card support")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "eligible": self.eligible,
            "lane_ids": list(self.lane_ids),
            "eligible_card_keys": list(self.eligible_card_keys),
            "selected_card_key": self.selected_card_key,
            "selected_lane_supports": [
                value.to_record() for value in self.selected_lane_supports
            ],
            "selection_key_sha256": self.selection_key_sha256,
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
            "card_vs_neutral_effect_identified": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedSupportResolver:
    """Choose one complete-support card for a two-lane M/N block."""

    def resolve(
        self,
        *,
        lanes: tuple[CampaignDiagnosticSupportLaneInput, ...],
        cards: tuple[CampaignDiagnosticSupportCardInput, ...],
        selection_key_sha256: str,
    ) -> PortfolioMemoryMatchedSupportResolution:
        if (
            type(lanes) is not tuple
            or len(lanes) != 2
            or any(
                type(value) is not CampaignDiagnosticSupportLaneInput
                for value in lanes
            )
        ):
            raise ValueError("matched resolution requires exactly two lane inputs")
        if type(cards) is not tuple or not cards or any(
            type(value) is not CampaignDiagnosticSupportCardInput for value in cards
        ):
            raise ValueError("matched resolution requires source card inputs")
        for value in lanes:
            value.__post_init__()
        for value in cards:
            value.__post_init__()
        lane_ids = tuple(sorted(value.lane.lane_id for value in lanes))
        if len(set(lane_ids)) != 2:
            raise ValueError("matched lanes must have distinct identities")
        card_keys = tuple(sorted(value.card.card_key for value in cards))
        if len(set(card_keys)) != len(card_keys):
            raise ValueError("matched source cards must have distinct keys")
        require_sha256(selection_key_sha256, "selection_key_sha256")

        by_lane = {value.lane.lane_id: value for value in lanes}
        by_card = {value.card.card_key: value for value in cards}
        complete: dict[str, tuple[PortfolioMemoryMatchedLaneSupport, ...]] = {}
        for card_key in card_keys:
            card = by_card[card_key]
            supports: list[PortfolioMemoryMatchedLaneSupport] = []
            for lane_id in lane_ids:
                try:
                    support = derive_portfolio_memory_dose_card_support(
                        card.semantics,
                        by_lane[lane_id].finite_variation_contract,
                    )
                except ValueError:
                    break
                supports.append(
                    PortfolioMemoryMatchedLaneSupport(
                        lane_id=lane_id,
                        support=support,
                    )
                )
            if len(supports) == len(lane_ids):
                complete[card_key] = tuple(supports)

        eligible_keys = tuple(sorted(complete))
        if not eligible_keys:
            return PortfolioMemoryMatchedSupportResolution(
                lane_ids=lane_ids,
                eligible_card_keys=(),
                selected_card_key=None,
                selected_lane_supports=(),
                selection_key_sha256=selection_key_sha256,
            )
        key_bytes = bytes.fromhex(selection_key_sha256)
        selected = min(
            eligible_keys,
            key=lambda card_key: (
                hashlib.sha256(
                    _CARD_ORDER_DOMAIN
                    + key_bytes
                    + card_key.encode("ascii", errors="strict")
                    + bytes.fromhex(by_card[card_key].card.card_identity_sha256)
                ).digest(),
                card_key,
            ),
        )
        return PortfolioMemoryMatchedSupportResolution(
            lane_ids=lane_ids,
            eligible_card_keys=eligible_keys,
            selected_card_key=selected,
            selected_lane_supports=complete[selected],
            selection_key_sha256=selection_key_sha256,
        )


@dataclass(frozen=True, slots=True)
class PortfolioMemoryLaneSupportResolution:
    """Outcome-blind card choice for one independently supportable lane.

    This resolution is an optimization exposure, not an active-versus-neutral
    causal comparison.  It exists so an unrelated second parent cannot erase a
    useful, exactly supported memory treatment from the first parent.
    """

    lane_id: str
    eligible_card_keys: tuple[str, ...]
    selected_card_key: str | None
    selected_support: PortfolioMemoryDoseCardSupport | None
    selection_key_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or not self.lane_id:
            raise ValueError("lane_id must be a nonempty string")
        if (
            type(self.eligible_card_keys) is not tuple
            or self.eligible_card_keys
            != tuple(sorted(set(self.eligible_card_keys)))
        ):
            raise ValueError("eligible card keys must be unique and canonical")
        require_sha256(self.selection_key_sha256, "selection_key_sha256")
        if self.selected_card_key is None:
            if self.eligible_card_keys or self.selected_support is not None:
                raise ValueError("ineligible lane cannot carry selected support")
        else:
            if self.selected_card_key not in self.eligible_card_keys:
                raise ValueError("selected lane card is outside the eligible set")
            if type(self.selected_support) is not PortfolioMemoryDoseCardSupport:
                raise TypeError("eligible lane requires exact selected support")
            self.selected_support.__post_init__()
            if self.selected_support.card_key != self.selected_card_key:
                raise ValueError("selected support names another lane card")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_LANE_SUPPORT_DOMAIN, self._unsigned_record()),
        )

    @property
    def eligible(self) -> bool:
        return self.selected_card_key is not None

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "lane_id": self.lane_id,
            "eligible": self.eligible,
            "eligible_card_keys": list(self.eligible_card_keys),
            "selected_card_key": self.selected_card_key,
            "selected_support": (
                None
                if self.selected_support is None
                else self.selected_support.to_record()
            ),
            "selection_key_sha256": self.selection_key_sha256,
            "policy": {
                "policy_id": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID,
                "policy_version": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION,
                "definition_sha256": (
                    PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256
                ),
            },
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
            "card_vs_neutral_effect_identified": False,
            "online_causal_credit_allowed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryLaneSupportResolver:
    """Select one exactly supported card for one lane without outcome access."""

    def resolve(
        self,
        *,
        lane: CampaignDiagnosticSupportLaneInput,
        cards: tuple[CampaignDiagnosticSupportCardInput, ...],
        selection_key_sha256: str,
    ) -> PortfolioMemoryLaneSupportResolution:
        if type(lane) is not CampaignDiagnosticSupportLaneInput:
            raise TypeError("lane must be an exact diagnostic support lane input")
        lane.__post_init__()
        if type(cards) is not tuple or not cards or any(
            type(value) is not CampaignDiagnosticSupportCardInput for value in cards
        ):
            raise ValueError("lane resolution requires source card inputs")
        for value in cards:
            value.__post_init__()
        card_keys = tuple(sorted(value.card.card_key for value in cards))
        if len(set(card_keys)) != len(card_keys):
            raise ValueError("lane source cards must have distinct keys")
        require_sha256(selection_key_sha256, "selection_key_sha256")

        by_card = {value.card.card_key: value for value in cards}
        supported: dict[str, PortfolioMemoryDoseCardSupport] = {}
        for card_key in card_keys:
            card = by_card[card_key]
            try:
                supported[card_key] = derive_portfolio_memory_dose_card_support(
                    card.semantics,
                    lane.finite_variation_contract,
                )
            except ValueError:
                continue
        eligible_keys = tuple(sorted(supported))
        if not eligible_keys:
            return PortfolioMemoryLaneSupportResolution(
                lane_id=lane.lane.lane_id,
                eligible_card_keys=(),
                selected_card_key=None,
                selected_support=None,
                selection_key_sha256=selection_key_sha256,
            )

        key_bytes = bytes.fromhex(selection_key_sha256)
        selected = min(
            eligible_keys,
            key=lambda card_key: (
                hashlib.sha256(
                    _LANE_CARD_ORDER_DOMAIN
                    + key_bytes
                    + bytes.fromhex(lane.lane.lane_identity_sha256)
                    + card_key.encode("ascii", errors="strict")
                    + bytes.fromhex(by_card[card_key].card.card_identity_sha256)
                ).digest(),
                card_key,
            ),
        )
        return PortfolioMemoryLaneSupportResolution(
            lane_id=lane.lane.lane_id,
            eligible_card_keys=eligible_keys,
            selected_card_key=selected,
            selected_support=supported[selected],
            selection_key_sha256=selection_key_sha256,
        )


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedArmView:
    """One materialized M or N view with a shared candidate-pool obligation."""

    plan_sha256: str
    assignment: PortfolioMemoryMatchedArmAssignment
    cards: tuple[PortfolioCard, ...]
    source_registry: PortfolioCardSourceRegistry
    experimental_view_receipt: PortfolioExperimentalViewReceipt
    required_common_pool_option_ids: tuple[str, ...]
    view_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.plan_sha256, "plan_sha256")
        if type(self.assignment) is not PortfolioMemoryMatchedArmAssignment:
            raise TypeError("assignment must be exact")
        self.assignment.__post_init__()
        if type(self.cards) is not tuple or len(self.cards) != 1:
            raise ValueError("a matched arm view requires exactly one card")
        card = self.cards[0]
        if type(card) is not PortfolioCard:
            raise TypeError("matched arm card must be exact")
        card.__post_init__()
        if type(self.source_registry) is not PortfolioCardSourceRegistry:
            raise TypeError("source_registry must be exact")
        self.source_registry.__post_init__()
        if type(self.experimental_view_receipt) is not PortfolioExperimentalViewReceipt:
            raise TypeError("experimental_view_receipt must be exact")
        self.experimental_view_receipt.__post_init__()
        if self.experimental_view_receipt.arm is not self.assignment.arm:
            raise ValueError("experimental view arm differs from its assignment")
        if (
            type(self.required_common_pool_option_ids) is not tuple
            or not self.required_common_pool_option_ids
            or self.required_common_pool_option_ids
            != tuple(sorted(set(self.required_common_pool_option_ids)))
        ):
            raise ValueError(
                "required common-pool actions must be nonempty and canonical"
            )
        if self.assignment.arm is PortfolioExperimentalArm.MEMORY:
            if card.derived_view_receipt is not None or not card.finite_action_evidence:
                raise ValueError("M requires the pristine evidence-bearing source card")
        else:
            if (
                card.prompt_payload != CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD
                or card.evidence_sha256
                != CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256
                or card.score_components
                or card.assigned_score is not None
                or card.finite_action_evidence
            ):
                raise ValueError("N does not use the canonical redacted card view")
        object.__setattr__(
            self,
            "view_sha256",
            _hash(_VIEW_DOMAIN, self._unsigned_record()),
        )

    @property
    def memory_dose_allowed(self) -> bool:
        return self.assignment.arm is PortfolioExperimentalArm.MEMORY

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_sha256": self.plan_sha256,
            "assignment": self.assignment.to_record(),
            "card_snapshot_sha256": (
                self.experimental_view_receipt.card_snapshot_sha256
            ),
            "source_registry_sha256": self.source_registry.registry_sha256,
            "experimental_view_receipt_sha256": (
                self.experimental_view_receipt.receipt_sha256
            ),
            "required_common_pool_option_ids": list(
                self.required_common_pool_option_ids
            ),
            "same_required_common_pool_actions_across_arms": True,
            "memory_dose_allowed": self.memory_dose_allowed,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "view_sha256": self.view_sha256}


def materialize_portfolio_memory_matched_arm(
    *,
    plan: PortfolioMemoryMatchedControlPlan,
    assignment: PortfolioMemoryMatchedArmAssignment,
    source_card: PortfolioCard,
    source_registry: PortfolioCardSourceRegistry,
    finite_variation_contract: FiniteVariationContract,
) -> PortfolioMemoryMatchedArmView:
    """Create one source-identical M/N view for a precommitted assignment."""

    if type(plan) is not PortfolioMemoryMatchedControlPlan:
        raise TypeError("plan must be exact")
    plan.__post_init__()
    if assignment not in plan.assignments:
        raise ValueError("assignment is outside the matched plan")
    if type(source_card) is not PortfolioCard:
        raise TypeError("source_card must be exact")
    source_card.__post_init__()
    if source_card.reference != plan.reference:
        raise ValueError("source card differs from the planned reference")
    if (
        source_card.source_binding is None
        or source_card.derived_view_receipt is not None
    ):
        raise ValueError("matched views require a pristine source-bound card")
    if type(source_registry) is not PortfolioCardSourceRegistry:
        raise TypeError("source_registry must be exact")
    source_registry.__post_init__()
    if len(source_registry.source_bindings) != 1 or (
        source_registry.source_bindings[0] != source_card.source_binding
    ):
        raise ValueError("matched view registry must contain exactly the source card")
    validate_finite_variation_contract(finite_variation_contract)
    required_ids = tuple(
        sorted(
            {
                value.option_id
                for value in source_card.source_binding.finite_action_evidence
            }
        )
    )
    available_ids = {value.option_id for value in finite_variation_contract.options}
    if not required_ids or not set(required_ids).issubset(available_ids):
        raise ValueError("source card lacks complete support in the current contract")

    if assignment.arm is PortfolioExperimentalArm.MEMORY:
        cards = (source_card,)
    else:
        transforms = tuple(
            sorted(
                (
                    PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION,
                    PortfolioCardViewTransform.EVIDENCE_REDACTION,
                    PortfolioCardViewTransform.PROMPT_REDACTION,
                    PortfolioCardViewTransform.SCORE_REDACTION,
                ),
                key=lambda value: value.value,
            )
        )
        cards = (
            derive_portfolio_card_view(
                source_card,
                prompt_payload=CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
                evidence_sha256=CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
                score_components=(),
                assigned_score=None,
                finite_action_evidence=(),
                transforms=transforms,
                policy_id=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
                policy_version=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
                policy_definition_sha256=(
                    PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                ),
            ),
        )
    receipt = bind_portfolio_experimental_view(
        arm=assignment.arm,
        cards=cards,
        finite_variation_contract=finite_variation_contract,
        source_registry=source_registry,
        policy_id=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
        policy_version=PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
        policy_definition_sha256=(
            PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
        ),
    )
    return PortfolioMemoryMatchedArmView(
        plan_sha256=plan.plan_sha256,
        assignment=assignment,
        cards=cards,
        source_registry=source_registry,
        experimental_view_receipt=receipt,
        required_common_pool_option_ids=required_ids,
    )


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedControlOutcome:
    """Observed M-minus-N contrast from one precommitted lane pair."""

    plan_sha256: str
    generation: int
    reference: InsightRef
    aggregation_binding_sha256: str
    active_view_sha256: str
    neutral_view_sha256: str
    active_result_receipt_sha256: str
    neutral_result_receipt_sha256: str
    active_wave_reward: float
    neutral_wave_reward: float
    outcome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "aggregation_binding_sha256",
            "active_view_sha256",
            "neutral_view_sha256",
            "active_result_receipt_sha256",
            "neutral_result_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        for name in ("active_wave_reward", "neutral_wave_reward"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
        object.__setattr__(
            self,
            "outcome_sha256",
            _hash(_OUTCOME_DOMAIN, self._unsigned_record()),
        )

    @property
    def observed_active_minus_neutral(self) -> float:
        return self.active_wave_reward - self.neutral_wave_reward

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "plan_sha256": self.plan_sha256,
            "generation": self.generation,
            "reference": _reference_record(self.reference),
            "aggregation_binding_sha256": self.aggregation_binding_sha256,
            "active_view_sha256": self.active_view_sha256,
            "neutral_view_sha256": self.neutral_view_sha256,
            "active_result_receipt_sha256": self.active_result_receipt_sha256,
            "neutral_result_receipt_sha256": self.neutral_result_receipt_sha256,
            "active_wave_reward_hex": self.active_wave_reward.hex(),
            "neutral_wave_reward_hex": self.neutral_wave_reward.hex(),
            "observed_active_minus_neutral_hex": (
                self.observed_active_minus_neutral.hex()
            ),
            "single_block_card_effect_identified": False,
            "online_score_update_allowed": False,
            "analysis_scope": "append_only_arm_aware_experimental_observation",
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "outcome_sha256": self.outcome_sha256}


__all__ = [
    "PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID",
    "PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION",
    "PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID",
    "PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION",
    "PortfolioMemoryMatchedArmAssignment",
    "PortfolioMemoryMatchedArmView",
    "PortfolioMemoryMatchedControlOutcome",
    "PortfolioMemoryMatchedControlPlan",
    "PortfolioMemoryMatchedControlPlanner",
    "PortfolioMemoryMatchedLaneSupport",
    "PortfolioMemoryMatchedSupportResolution",
    "PortfolioMemoryMatchedSupportResolver",
    "PortfolioMemoryLaneSupportResolution",
    "PortfolioMemoryLaneSupportResolver",
    "materialize_portfolio_memory_matched_arm",
]
