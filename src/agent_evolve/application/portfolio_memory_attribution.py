"""Post-treatment attribution audit for portfolio memory waves.

Whole-wave reward is the correct endpoint for an intention-to-treat study of
prompt-visible memory.  It is not, however, evidence that an explicitly cited
card-supported candidate created the reward: an uncited member may be the only
frontier contributor, and another prompt-wide lane may independently select
the card's exact option.  This module records those distinctions at the common
generation barrier without importing a benchmark, provider, prompt renderer,
or archive implementation.

The leave-one-out endpoint reuses each wave's precommitted aggregation binding.
It is descriptive performance attribution, never a causal action effect.  A
true card-effect estimate still requires a prospectively randomized,
prompt-redacted matched control call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.portfolio_evolution import (
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_CANDIDATE_DOMAIN = b"agent-evolve:portfolio-memory-candidate-contribution:v1\x00"
_CARD_DOMAIN = b"agent-evolve:portfolio-memory-card-performance:v1\x00"
_AUDIT_DOMAIN = b"agent-evolve:portfolio-memory-attribution-audit:v1\x00"

PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_ID = (
    "post_treatment_memory_performance_attribution"
)
PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_VERSION = 1
PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:post-treatment-memory-performance-attribution:v1;"
    b"precommitted-wave-aggregator;leave-one-out-member-contribution;"
    b"explicit-card-citation;cross-lane-same-option-interference;"
    b"descriptive-only;no-causal-action-or-card-credit;"
    b"no-online-score-consumption"
).hexdigest()


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


def _finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCandidateContribution:
    """One evaluated member's descriptive contribution to its memory wave."""

    generation: int
    request_sha256: str
    result_receipt_sha256: str
    parent_candidate_id: str
    candidate_id: str
    rank: int
    option_id: str
    option_identity_sha256: str
    supporting_card_keys: tuple[str, ...]
    supporting_references: tuple[InsightRef, ...]
    member_reward: float
    joint_wave_reward: float
    leave_one_out_wave_reward: float
    leave_one_out_contribution: float
    better_than_any_parent: bool
    dominates_any_parent: bool
    contribution_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in ("request_sha256", "result_receipt_sha256"):
            require_sha256(getattr(self, name), name)
        for name in ("parent_candidate_id", "candidate_id"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the closed token grammar")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be positive")
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the closed option grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if (
            type(self.supporting_card_keys) is not tuple
            or self.supporting_card_keys != tuple(sorted(set(self.supporting_card_keys)))
        ):
            raise ValueError("supporting_card_keys must be unique and canonical")
        for value in self.supporting_card_keys:
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError("supporting card key is invalid")
        if (
            type(self.supporting_references) is not tuple
            or len(set(self.supporting_references))
            != len(self.supporting_references)
        ):
            raise ValueError("supporting_references must be unique")
        for value in self.supporting_references:
            InsightRef.__post_init__(value)
        if len(self.supporting_card_keys) != len(self.supporting_references):
            raise ValueError("supporting card keys and references differ")
        for name in (
            "member_reward",
            "joint_wave_reward",
            "leave_one_out_wave_reward",
            "leave_one_out_contribution",
        ):
            _finite(getattr(self, name), name=name)
        if type(self.better_than_any_parent) is not bool:
            raise TypeError("better_than_any_parent must be an exact bool")
        if type(self.dominates_any_parent) is not bool:
            raise TypeError("dominates_any_parent must be an exact bool")
        object.__setattr__(
            self,
            "contribution_sha256",
            _hash(_CANDIDATE_DOMAIN, self._unsigned_record()),
        )

    @property
    def explicitly_card_supported(self) -> bool:
        return bool(self.supporting_references)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "request_sha256": self.request_sha256,
            "result_receipt_sha256": self.result_receipt_sha256,
            "parent_candidate_id": self.parent_candidate_id,
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "supporting_card_keys": list(self.supporting_card_keys),
            "supporting_references": [
                _reference_record(value) for value in self.supporting_references
            ],
            "member_reward_hex": self.member_reward.hex(),
            "joint_wave_reward_hex": self.joint_wave_reward.hex(),
            "leave_one_out_wave_reward_hex": self.leave_one_out_wave_reward.hex(),
            "leave_one_out_contribution_hex": self.leave_one_out_contribution.hex(),
            "better_than_any_parent": self.better_than_any_parent,
            "dominates_any_parent": self.dominates_any_parent,
            "explicitly_card_supported": self.explicitly_card_supported,
            "attribution_scope": "post_treatment_descriptive_not_causal_credit",
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "contribution_sha256": self.contribution_sha256,
        }


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCardPerformanceAudit:
    """One explicit card/action join plus cross-lane interference evidence."""

    reference: InsightRef
    card_key: str
    contribution_sha256: str
    generation: int
    request_sha256: str
    candidate_id: str
    option_id: str
    candidate_member_reward: float
    candidate_leave_one_out_contribution: float
    joint_wave_reward: float
    unsupported_positive_member_count: int
    cross_lane_same_option_candidate_ids: tuple[str, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        InsightRef.__post_init__(self.reference)
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed token grammar")
        for name in ("contribution_sha256", "request_sha256"):
            require_sha256(getattr(self, name), name)
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.candidate_id) is not str or _TOKEN.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must use the closed token grammar")
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the closed option grammar")
        for name in (
            "candidate_member_reward",
            "candidate_leave_one_out_contribution",
            "joint_wave_reward",
        ):
            _finite(getattr(self, name), name=name)
        if (
            type(self.unsupported_positive_member_count) is not int
            or self.unsupported_positive_member_count < 0
        ):
            raise ValueError("unsupported_positive_member_count must be nonnegative")
        if (
            type(self.cross_lane_same_option_candidate_ids) is not tuple
            or self.cross_lane_same_option_candidate_ids
            != tuple(sorted(set(self.cross_lane_same_option_candidate_ids)))
        ):
            raise ValueError("cross-lane candidate IDs must be unique and canonical")
        for value in self.cross_lane_same_option_candidate_ids:
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError("cross-lane candidate ID is invalid")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_CARD_DOMAIN, self._unsigned_record()),
        )

    @property
    def cross_lane_action_spillover(self) -> bool:
        return bool(self.cross_lane_same_option_candidate_ids)

    @property
    def positive_wave_with_nonpositive_supported_action(self) -> bool:
        return self.joint_wave_reward > 0.0 and self.candidate_member_reward <= 0.0

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "card_key": self.card_key,
            "contribution_sha256": self.contribution_sha256,
            "generation": self.generation,
            "request_sha256": self.request_sha256,
            "candidate_id": self.candidate_id,
            "option_id": self.option_id,
            "candidate_member_reward_hex": self.candidate_member_reward.hex(),
            "candidate_leave_one_out_contribution_hex": (
                self.candidate_leave_one_out_contribution.hex()
            ),
            "joint_wave_reward_hex": self.joint_wave_reward.hex(),
            "unsupported_positive_member_count": (
                self.unsupported_positive_member_count
            ),
            "cross_lane_same_option_candidate_ids": list(
                self.cross_lane_same_option_candidate_ids
            ),
            "cross_lane_action_spillover": self.cross_lane_action_spillover,
            "positive_wave_with_nonpositive_supported_action": (
                self.positive_wave_with_nonpositive_supported_action
            ),
            "causal_action_effect_identified": False,
            "eligible_for_online_score_update": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryAttributionAudit:
    """Generation-wide descriptive audit with an explicit causal boundary."""

    generation: int
    wave_request_sha256s: tuple[str, ...]
    wave_result_receipt_sha256s: tuple[str, ...]
    candidate_contributions: tuple[PortfolioMemoryCandidateContribution, ...]
    card_performance: tuple[PortfolioMemoryCardPerformanceAudit, ...]
    policy_id: str = field(
        init=False,
        default=PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_VERSION,
    )
    policy_definition_sha256: str = field(
        init=False,
        default=PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_DEFINITION_SHA256,
    )
    audit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in ("wave_request_sha256s", "wave_result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be non-empty")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if (
            type(self.candidate_contributions) is not tuple
            or not self.candidate_contributions
            or any(
                type(value) is not PortfolioMemoryCandidateContribution
                for value in self.candidate_contributions
            )
        ):
            raise ValueError("candidate_contributions must contain exact values")
        for value in self.candidate_contributions:
            value.__post_init__()
            if value.generation != self.generation:
                raise ValueError("candidate contribution generation differs")
        contribution_ids = tuple(
            value.contribution_sha256 for value in self.candidate_contributions
        )
        if contribution_ids != tuple(sorted(set(contribution_ids))):
            raise ValueError("candidate contributions must use canonical identities")
        if type(self.card_performance) is not tuple or any(
            type(value) is not PortfolioMemoryCardPerformanceAudit
            for value in self.card_performance
        ):
            raise TypeError("card_performance must contain exact values")
        known = set(contribution_ids)
        for value in self.card_performance:
            value.__post_init__()
            if value.generation != self.generation:
                raise ValueError("card performance generation differs")
            if value.contribution_sha256 not in known:
                raise ValueError("card performance cites a foreign contribution")
        card_ids = tuple(value.receipt_sha256 for value in self.card_performance)
        if card_ids != tuple(sorted(set(card_ids))):
            raise ValueError("card performance must use canonical identities")
        if (
            self.policy_id != PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_ID
            or self.policy_version
            != PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_VERSION
            or self.policy_definition_sha256
            != PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_DEFINITION_SHA256
        ):
            raise ValueError("unsupported memory attribution audit policy")
        object.__setattr__(self, "audit_sha256", _hash(_AUDIT_DOMAIN, self._unsigned_record()))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "wave_result_receipt_sha256s": list(
                self.wave_result_receipt_sha256s
            ),
            "candidate_contribution_sha256s": [
                value.contribution_sha256 for value in self.candidate_contributions
            ],
            "card_performance_receipt_sha256s": [
                value.receipt_sha256 for value in self.card_performance
            ],
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "cross_lane_action_spillover_count": sum(
                value.cross_lane_action_spillover for value in self.card_performance
            ),
            "positive_wave_nonpositive_supported_action_count": sum(
                value.positive_wave_with_nonpositive_supported_action
                for value in self.card_performance
            ),
            "causal_card_effect_identified": False,
            "causal_action_effect_identified": False,
            "online_score_update_allowed": False,
            "required_successor_design": (
                "prospective_randomized_prompt_redacted_matched_control"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "candidate_contributions": [
                value.to_record() for value in self.candidate_contributions
            ],
            "card_performance": [
                value.to_record() for value in self.card_performance
            ],
            "audit_sha256": self.audit_sha256,
        }


def audit_portfolio_memory_attribution(
    *,
    waves: tuple[PortfolioVariationWaveRequest, ...],
    results: tuple[PortfolioVariationWaveResult, ...],
) -> PortfolioMemoryAttributionAudit | None:
    """Audit memory-supported action performance at a sealed generation."""

    if type(waves) is not tuple or type(results) is not tuple:
        raise TypeError("waves/results must be exact tuples")
    if not waves or len(waves) != len(results):
        raise ValueError("waves/results must be non-empty and equally sized")
    if any(type(value) is not PortfolioVariationWaveRequest for value in waves):
        raise TypeError("waves must contain exact requests")
    if any(type(value) is not PortfolioVariationWaveResult for value in results):
        raise TypeError("results must contain exact results")
    for value in waves:
        PortfolioVariationWaveRequest.__post_init__(value)
    for value in results:
        PortfolioVariationWaveResult.__post_init__(value)
    result_by_request = {value.receipt.request_sha256: value for value in results}
    if len(result_by_request) != len(results):
        raise ValueError("results repeat a wave request")
    if set(result_by_request) != {
        value.selection_request.request_sha256 for value in waves
    }:
        raise ValueError("wave requests and results do not form an exact join")
    generations = {value.generation for value in waves}
    if len(generations) != 1:
        raise ValueError("one attribution audit cannot span generations")
    generation = next(iter(generations))

    contributions: list[PortfolioMemoryCandidateContribution] = []
    for wave in waves:
        credit = wave.memory_credit
        if credit is None:
            continue
        result = result_by_request[wave.selection_request.request_sha256]
        aggregation = credit.aggregation.aggregate
        joint_reward = aggregation(result.outcomes)
        if type(joint_reward) is not float or not math.isfinite(joint_reward):
            raise TypeError("memory aggregation replay must return a finite float")
        committed = result.receipt.memory_credit
        pending = result.pending_memory_credit
        recorded_reward = (
            committed.reward
            if committed is not None
            else (None if pending is None else pending.reward)
        )
        if recorded_reward is None or recorded_reward != joint_reward:
            raise ValueError("memory aggregation replay differs from recorded reward")
        cards = {value.card_key: value.reference for value in wave.selection_request.cards}
        for index, (member, attribution) in enumerate(
            zip(
                result.receipt.members,
                result.action_attributions,
                strict=True,
            )
        ):
            remaining = result.outcomes[:index] + result.outcomes[index + 1 :]
            if not remaining:
                raise ValueError(
                    "leave-one-out memory attribution requires portfolio_size >= 2"
                )
            without = aggregation(remaining)
            if type(without) is not float or not math.isfinite(without):
                raise TypeError(
                    "leave-one-out aggregation replay must return a finite float"
                )
            supporting_keys = attribution.supporting_card_keys
            references = tuple(cards[value] for value in supporting_keys)
            contributions.append(
                PortfolioMemoryCandidateContribution(
                    generation=generation,
                    request_sha256=wave.selection_request.request_sha256,
                    result_receipt_sha256=result.receipt.receipt_sha256,
                    parent_candidate_id=wave.parent.candidate_id.value,
                    candidate_id=member.materialization.candidate_id.value,
                    rank=member.materialization.rank,
                    option_id=member.materialization.option_id,
                    option_identity_sha256=(
                        member.materialization.option_identity_sha256
                    ),
                    supporting_card_keys=supporting_keys,
                    supporting_references=references,
                    member_reward=member.reward,
                    joint_wave_reward=joint_reward,
                    leave_one_out_wave_reward=without,
                    leave_one_out_contribution=joint_reward - without,
                    better_than_any_parent=member.better_than_any_parent,
                    dominates_any_parent=member.dominates_any_parent,
                )
            )

    if not contributions:
        return None
    canonical_contributions = tuple(
        sorted(contributions, key=lambda value: value.contribution_sha256)
    )
    card_rows: list[PortfolioMemoryCardPerformanceAudit] = []
    for contribution in canonical_contributions:
        if not contribution.explicitly_card_supported:
            continue
        same_lane = tuple(
            value
            for value in canonical_contributions
            if value.request_sha256 == contribution.request_sha256
        )
        other_lanes = tuple(
            value
            for value in canonical_contributions
            if value.request_sha256 != contribution.request_sha256
        )
        spillovers = tuple(
            sorted(
                value.candidate_id
                for value in other_lanes
                if value.option_id == contribution.option_id
            )
        )
        unsupported_positive = sum(
            not value.explicitly_card_supported and value.member_reward > 0.0
            for value in same_lane
        )
        for card_key, reference in zip(
            contribution.supporting_card_keys,
            contribution.supporting_references,
            strict=True,
        ):
            card_rows.append(
                PortfolioMemoryCardPerformanceAudit(
                    reference=reference,
                    card_key=card_key,
                    contribution_sha256=contribution.contribution_sha256,
                    generation=generation,
                    request_sha256=contribution.request_sha256,
                    candidate_id=contribution.candidate_id,
                    option_id=contribution.option_id,
                    candidate_member_reward=contribution.member_reward,
                    candidate_leave_one_out_contribution=(
                        contribution.leave_one_out_contribution
                    ),
                    joint_wave_reward=contribution.joint_wave_reward,
                    unsupported_positive_member_count=unsupported_positive,
                    cross_lane_same_option_candidate_ids=spillovers,
                )
            )
    return PortfolioMemoryAttributionAudit(
        generation=generation,
        wave_request_sha256s=tuple(
            sorted(value.selection_request.request_sha256 for value in waves)
        ),
        wave_result_receipt_sha256s=tuple(
            sorted(value.receipt.receipt_sha256 for value in results)
        ),
        candidate_contributions=canonical_contributions,
        card_performance=tuple(
            sorted(card_rows, key=lambda value: value.receipt_sha256)
        ),
    )


__all__ = [
    "PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_ID",
    "PORTFOLIO_MEMORY_ATTRIBUTION_AUDIT_POLICY_VERSION",
    "PortfolioMemoryAttributionAudit",
    "PortfolioMemoryCandidateContribution",
    "PortfolioMemoryCardPerformanceAudit",
    "audit_portfolio_memory_attribution",
]
