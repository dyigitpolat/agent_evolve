"""Deterministic lane-level resolution over the typed memory transfer ladder.

This module is deliberately independent of campaign workloads and providers.
A caller supplies one current finite-action lane plus source-bound card
semantics.  The resolver prefers exact replay support, then local advisory
support, then path/family advisory support, and uses an authenticated key only
to break ties inside the best available tier.

The result is a retrieval decision, not a causal memory result.  Advisory
support can be rendered into a prompt but cannot become a bounded forced-action
dose.  No tier authorizes causal card credit without a separate exposure design.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.application.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryTransferLadderAssessment,
    PortfolioMemoryTransferTier,
    assess_portfolio_memory_transfer_ladder,
    derive_portfolio_memory_advisory_card_support,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.portfolio_memory_dose import PortfolioMemoryDoseCardSupport


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_ID = (
    "typed_portfolio_memory_transfer_lane_resolver"
)
PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_VERSION = 1
PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:typed-portfolio-memory-transfer-lane-resolver:v1;"
    b"tier-order=exact,local-advisory,path-family-advisory;"
    b"best-tier-only=true;within-tier=authenticated-keyed-order;"
    b"provider-and-outcome-blind=true;causal-memory-credit=false"
).hexdigest()
_SELECTION_DOMAIN = (
    b"agent-evolve:typed-portfolio-memory-transfer-lane-card-order:v1\x00"
)
_RECEIPT_DOMAIN = b"agent-evolve:typed-portfolio-memory-transfer-lane:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _tier_rank(value: PortfolioMemoryTransferTier) -> int:
    return {
        PortfolioMemoryTransferTier.EXACT_ACTION_REPLAY: 0,
        PortfolioMemoryTransferTier.LOCAL_ACTION_ADVISORY: 1,
        PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY: 2,
        PortfolioMemoryTransferTier.UNSUPPORTED: 3,
    }[value]


@dataclass(frozen=True, slots=True)
class PortfolioMemoryTransferCard:
    """One source-bound card available for typed transfer into a lane."""

    card_key: str
    card_identity_sha256: str
    semantics: PortfolioMemoryDoseCardSemantics

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed token grammar")
        require_sha256(self.card_identity_sha256, "card_identity_sha256")
        if type(self.semantics) is not PortfolioMemoryDoseCardSemantics:
            raise TypeError("semantics must be exact PortfolioMemoryDoseCardSemantics")
        self.semantics.__post_init__()
        if self.semantics.card_key != self.card_key:
            raise ValueError("card and transfer semantics use different keys")


@dataclass(frozen=True, slots=True)
class PortfolioMemoryTransferLane:
    """One stable lane and its complete current finite-action contract."""

    lane_id: str
    lane_identity_sha256: str
    finite_variation_contract: FiniteVariationContract

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or _TOKEN.fullmatch(self.lane_id) is None:
            raise ValueError("lane_id must use the closed token grammar")
        require_sha256(self.lane_identity_sha256, "lane_identity_sha256")
        if type(self.finite_variation_contract) is not FiniteVariationContract:
            raise TypeError("finite_variation_contract must be exact")
        validate_finite_variation_contract(self.finite_variation_contract)


@dataclass(frozen=True, slots=True)
class PortfolioMemoryTransferLaneResolution:
    """Replayable result of one best-tier, keyed lane/card resolution."""

    lane_id: str
    lane_identity_sha256: str
    selection_key_sha256: str
    eligible_assessments: tuple[PortfolioMemoryTransferLadderAssessment, ...]
    selected_card_key: str | None
    selected_assessment: PortfolioMemoryTransferLadderAssessment | None
    selected_support: PortfolioMemoryDoseCardSupport | None
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or _TOKEN.fullmatch(self.lane_id) is None:
            raise ValueError("lane_id must use the closed token grammar")
        require_sha256(self.lane_identity_sha256, "lane_identity_sha256")
        require_sha256(self.selection_key_sha256, "selection_key_sha256")
        values = self.eligible_assessments
        if type(values) is not tuple or any(
            type(value) is not PortfolioMemoryTransferLadderAssessment
            for value in values
        ):
            raise TypeError("eligible_assessments must contain exact assessments")
        for value in values:
            value.__post_init__()
            if not value.deliverable_option_ids:
                raise ValueError("eligible assessment is not deliverable")
        if tuple(value.card_key for value in values) != tuple(
            sorted({value.card_key for value in values})
        ):
            raise ValueError("eligible assessments must use canonical unique cards")
        selected = self.selected_assessment
        support = self.selected_support
        if self.selected_card_key is None:
            if values or selected is not None or support is not None:
                raise ValueError("ineligible resolution cannot contain selected evidence")
        else:
            if type(selected) is not PortfolioMemoryTransferLadderAssessment:
                raise TypeError("eligible resolution requires a selected assessment")
            selected.__post_init__()
            if selected.card_key != self.selected_card_key or selected not in values:
                raise ValueError("selected assessment is outside the eligible cards")
            if type(support) is not PortfolioMemoryDoseCardSupport:
                raise TypeError("eligible resolution requires exact selected support")
            support.__post_init__()
            if (
                support.card_key != self.selected_card_key
                or tuple(value[0] for value in support.compatible_options)
                != selected.deliverable_option_ids
            ):
                raise ValueError("selected support differs from its ladder assessment")
            best_rank = min(_tier_rank(value.tier) for value in values)
            if _tier_rank(selected.tier) != best_rank:
                raise ValueError("selected assessment does not use the best tier")
        object.__setattr__(
            self,
            "receipt_sha256",
            hashlib.sha256(_RECEIPT_DOMAIN + _canonical_json(self._unsigned_record()))
            .hexdigest(),
        )

    @property
    def eligible(self) -> bool:
        return self.selected_card_key is not None

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "lane_id": self.lane_id,
            "lane_identity_sha256": self.lane_identity_sha256,
            "selection_key_sha256": self.selection_key_sha256,
            "eligible": self.eligible,
            "eligible_assessments": [
                value.to_record() for value in self.eligible_assessments
            ],
            "selected_card_key": self.selected_card_key,
            "selected_assessment_sha256": (
                None
                if self.selected_assessment is None
                else self.selected_assessment.assessment_sha256
            ),
            "selected_support": (
                None if self.selected_support is None else self.selected_support.to_record()
            ),
            "policy": {
                "policy_id": PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_ID,
                "policy_version": PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_VERSION,
                "definition_sha256": (
                    PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_DEFINITION_SHA256
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
class PortfolioMemoryTransferLaneResolver:
    """Resolve one lane using typed authority and deterministic keyed ties."""

    def resolve(
        self,
        *,
        lane: PortfolioMemoryTransferLane,
        cards: tuple[PortfolioMemoryTransferCard, ...],
        selection_key_sha256: str,
    ) -> PortfolioMemoryTransferLaneResolution:
        if type(lane) is not PortfolioMemoryTransferLane:
            raise TypeError("lane must be exact PortfolioMemoryTransferLane")
        lane.__post_init__()
        if type(cards) is not tuple or not cards or any(
            type(value) is not PortfolioMemoryTransferCard for value in cards
        ):
            raise ValueError("cards must contain exact transfer cards")
        for value in cards:
            value.__post_init__()
        if tuple(value.card_key for value in cards) != tuple(
            sorted({value.card_key for value in cards})
        ):
            raise ValueError("transfer cards must be unique and canonical")
        require_sha256(selection_key_sha256, "selection_key_sha256")
        assessments = tuple(
            assessment
            for card in cards
            if (
                assessment := assess_portfolio_memory_transfer_ladder(
                    card.semantics,
                    lane.finite_variation_contract,
                )
            ).deliverable_option_ids
        )
        if not assessments:
            return PortfolioMemoryTransferLaneResolution(
                lane_id=lane.lane_id,
                lane_identity_sha256=lane.lane_identity_sha256,
                selection_key_sha256=selection_key_sha256,
                eligible_assessments=(),
                selected_card_key=None,
                selected_assessment=None,
                selected_support=None,
            )
        best_rank = min(_tier_rank(value.tier) for value in assessments)
        best = tuple(value for value in assessments if _tier_rank(value.tier) == best_rank)
        cards_by_key = {value.card_key: value for value in cards}
        selected = min(
            best,
            key=lambda value: (
                hashlib.sha256(
                    _SELECTION_DOMAIN
                    + bytes.fromhex(selection_key_sha256)
                    + bytes.fromhex(lane.lane_identity_sha256)
                    + value.card_key.encode("ascii", errors="strict")
                    + bytes.fromhex(
                        cards_by_key[value.card_key].card_identity_sha256
                    )
                ).digest(),
                value.card_key,
            ),
        )
        support = derive_portfolio_memory_advisory_card_support(
            cards_by_key[selected.card_key].semantics,
            lane.finite_variation_contract,
        )
        return PortfolioMemoryTransferLaneResolution(
            lane_id=lane.lane_id,
            lane_identity_sha256=lane.lane_identity_sha256,
            selection_key_sha256=selection_key_sha256,
            eligible_assessments=assessments,
            selected_card_key=selected.card_key,
            selected_assessment=selected,
            selected_support=support,
        )


__all__ = [
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_ID",
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_VERSION",
    "PortfolioMemoryTransferCard",
    "PortfolioMemoryTransferLane",
    "PortfolioMemoryTransferLaneResolution",
    "PortfolioMemoryTransferLaneResolver",
]
