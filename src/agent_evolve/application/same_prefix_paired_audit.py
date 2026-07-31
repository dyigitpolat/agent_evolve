"""Outcome-blind paired audits for one frozen adaptive-action prefix.

This module is an experimental interception primitive, not a production
allocation policy.  It converts the final factor-stratified audit decision
into two distinct arms:

* the frozen legacy audit anchor; and
* one factor-stratified alternative selected without either arm's outcome.

Callers must durably commit the returned plan before evaluating either arm.
Both arms are then valued independently against the same pre-audit evaluation
prefix.  Their union is never a budget-matched optimizer result and must not be
published to the authoritative archive.

The design consumes only portable action descriptors and authenticated racing
evidence.  It has no workload, objective, configuration, prompt, model, or
provider branch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
    AdaptiveActionRacingDecision,
    AdaptiveActionSetOutcome,
    AdaptiveActionWave,
    OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_ID,
    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


SAME_PREFIX_PAIRED_AUDIT_DESIGNER_ID = (
    "factor_stratified_same_prefix_paired_audit"
)
SAME_PREFIX_PAIRED_AUDIT_DESIGNER_VERSION = 1
FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID = (
    "forecast_opportunity_same_prefix_shadow"
)
FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_VERSION = 1
FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID = (
    "forecast_stratified_same_prefix_audit"
)
FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_VERSION = 1
FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS = frozenset(
    {
        FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID,
        FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID,
    }
)
SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_ID = (
    "same_prefix_paired_audit_adjudicator"
)
SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_VERSION = 1

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DESIGNER_DEFINITION_DOMAIN = (
    b"agent-evolve:same-prefix-paired-audit-designer-definition:v1\x00"
)
_FORECAST_SHADOW_DESIGNER_DEFINITION_DOMAIN = (
    b"agent-evolve:forecast-opportunity-same-prefix-shadow-designer:"
    b"definition:v1\x00"
)
_FORECAST_STRATIFIED_DESIGNER_DEFINITION_DOMAIN = (
    b"agent-evolve:forecast-stratified-same-prefix-audit-designer:"
    b"definition:v1\x00"
)
_PLAN_DOMAIN = b"agent-evolve:same-prefix-paired-audit-plan:v1\x00"
_ADJUDICATOR_DEFINITION_DOMAIN = (
    b"agent-evolve:same-prefix-paired-audit-adjudicator-definition:v1\x00"
)
_OBSERVATION_DOMAIN = (
    b"agent-evolve:same-prefix-paired-audit-observation:v1\x00"
)


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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_probability(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 < value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite positive probability")


def _read_nonnegative_hex(record: dict[str, object], key: str) -> float:
    raw = record.get(key)
    if type(raw) is not str:
        raise TypeError(f"{key} must be an exact hexadecimal float")
    try:
        value = float.fromhex(raw)
    except ValueError as error:
        raise ValueError(f"{key} is not a hexadecimal float") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return float(value)


def _stable_unit_interval(*parts: object) -> float:
    payload = _canonical_json(list(parts))
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / float(2**64)


def _canonical_hash_tuple(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or (not allow_empty and not values)
        or values != tuple(sorted(set(values)))
    ):
        raise ValueError(f"{name} must be a canonical exact tuple")
    for value in values:
        require_sha256(value, name)
    return values


class SamePrefixPairedAuditArm(str, Enum):
    """The arm retained by the original budget-matched decision."""

    LEGACY = "legacy"
    EXPLORATION = "exploration"


class SamePrefixPairedAuditWinner(str, Enum):
    """Winner under conditional utility at the common prefix."""

    LEGACY = "legacy"
    EXPLORATION = "exploration"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class SamePrefixPairedAuditPlan:
    """Hash-bound plan that must be committed before arm evaluation."""

    designer_id: str
    designer_version: int
    designer_definition_sha256: str
    residual_request_sha256: str
    racing_decision_sha256: str
    common_prefix_action_sha256s: tuple[str, ...]
    authoritative_arm: SamePrefixPairedAuditArm
    authoritative_action_sha256: str
    legacy_action_sha256: str
    exploration_action_sha256: str
    exploration_stratum_key: tuple[str, ...]
    distinct_exploration_support_action_sha256s: tuple[str, ...]
    exploration_selection_propensity: float
    evidence: FrozenJsonObject
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.designer_id, name="designer_id")
        if type(self.designer_version) is not int or self.designer_version <= 0:
            raise ValueError("designer_version must be positive")
        for value, name in (
            (
                self.designer_definition_sha256,
                "designer_definition_sha256",
            ),
            (self.residual_request_sha256, "residual_request_sha256"),
            (self.racing_decision_sha256, "racing_decision_sha256"),
            (self.authoritative_action_sha256, "authoritative_action_sha256"),
            (self.legacy_action_sha256, "legacy_action_sha256"),
            (self.exploration_action_sha256, "exploration_action_sha256"),
        ):
            require_sha256(value, name)
        prefix = _canonical_hash_tuple(
            self.common_prefix_action_sha256s,
            name="common_prefix_action_sha256s",
            allow_empty=False,
        )
        support = _canonical_hash_tuple(
            self.distinct_exploration_support_action_sha256s,
            name="distinct_exploration_support_action_sha256s",
            allow_empty=False,
        )
        if type(self.authoritative_arm) is not SamePrefixPairedAuditArm:
            raise TypeError("authoritative_arm must be exact")
        if self.legacy_action_sha256 == self.exploration_action_sha256:
            raise ValueError("paired audit arms must be distinct")
        if {
            self.legacy_action_sha256,
            self.exploration_action_sha256,
        } & set(prefix):
            raise ValueError("paired audit arm is already in the common prefix")
        if self.exploration_action_sha256 not in support:
            raise ValueError("exploration arm is outside its distinct support")
        expected_authoritative = (
            self.legacy_action_sha256
            if self.authoritative_arm is SamePrefixPairedAuditArm.LEGACY
            else self.exploration_action_sha256
        )
        if self.authoritative_action_sha256 != expected_authoritative:
            raise ValueError("authoritative action does not match its arm")
        if (
            type(self.exploration_stratum_key) is not tuple
            or not self.exploration_stratum_key
        ):
            raise ValueError("exploration_stratum_key must be non-empty")
        for value in self.exploration_stratum_key:
            _require_token(value, name="exploration stratum level")
        _require_probability(
            self.exploration_selection_propensity,
            name="exploration_selection_propensity",
        )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "plan_sha256",
            _hash(_PLAN_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "designer": {
                "designer_id": self.designer_id,
                "designer_version": self.designer_version,
                "definition_sha256": self.designer_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "racing_decision_sha256": self.racing_decision_sha256,
            "common_prefix_action_sha256s": list(
                self.common_prefix_action_sha256s
            ),
            "authoritative_arm": self.authoritative_arm.value,
            "authoritative_action_sha256": (
                self.authoritative_action_sha256
            ),
            "legacy_action_sha256": self.legacy_action_sha256,
            "exploration_action_sha256": self.exploration_action_sha256,
            "exploration_stratum_key": list(
                self.exploration_stratum_key
            ),
            "distinct_exploration_support_action_sha256s": list(
                self.distinct_exploration_support_action_sha256s
            ),
            "exploration_selection_propensity_hex": (
                self.exploration_selection_propensity.hex()
            ),
            "evidence_sha256": typed_json_sha256(self.evidence),
            "current_arm_outcomes_observed": False,
            "assay_union_may_enter_authoritative_archive": False,
            "workload_objective_model_provider_prompt_config_branches": False,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        result = {
            **self._unsigned_record(),
            "plan_sha256": self.plan_sha256,
        }
        if include_evidence:
            result["evidence"] = thaw_json(self.evidence)
        return result


@runtime_checkable
class SamePrefixPairedAuditDesignerPort(Protocol):
    """Inverted port for an outcome-blind paired-audit design."""

    designer_id: str
    designer_version: int
    definition_sha256: str

    def design(
        self,
        *,
        decision: AdaptiveActionRacingDecision,
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> SamePrefixPairedAuditPlan: ...


@runtime_checkable
class ForecastOpportunitySamePrefixShadowDesignerPort(Protocol):
    """Inverted port for a pre-outcome challenger-versus-fallback shadow."""

    designer_id: str
    designer_version: int
    definition_sha256: str

    def design(
        self,
        *,
        adaptive_step: int,
        remaining_authoritative_slots_after_decision: int,
        decision: AdaptiveActionRacingDecision,
        fallback: AdaptiveActionRacingDecision,
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> SamePrefixPairedAuditPlan | None: ...


ForecastOpportunitySamePrefixAuditDesignerPort = (
    ForecastOpportunitySamePrefixShadowDesignerPort
)


@dataclass(frozen=True, slots=True)
class ForecastOpportunitySamePrefixShadowDesigner:
    """Freeze a final-step forecast challenger against its exact fallback.

    Final-step interception prevents a shadow outcome from becoming a later
    authoritative action in the same stage.  Only the challenger remains in
    the budget-matched optimizer archive.
    """

    final_continuation_only: bool = True
    designer_id: str = (
        FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID
    )
    designer_version: int = (
        FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.final_continuation_only) is not bool:
            raise TypeError("final_continuation_only must be exact")
        if (
            self.designer_id
            != FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID
            or self.designer_version
            != FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_VERSION
        ):
            raise ValueError("forecast shadow designer identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _FORECAST_SHADOW_DESIGNER_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "designer_id": self.designer_id,
                    "designer_version": self.designer_version,
                    "final_continuation_only": self.final_continuation_only,
                    "authoritative_arm": "forecast_opportunity_challenger",
                    "counterfactual_arm": "authenticated_fallback",
                    "common_prefix": "exact-real-selected-prefix",
                    "plan_frozen_before_either_arm_outcome": True,
                    "assay_union_may_enter_authoritative_archive": False,
                    "workload_objective_model_provider_prompt_config_"
                    "branches": False,
                },
            ),
        )

    @staticmethod
    def _validate_action_market(
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> dict[str, AdaptiveActionDescriptor]:
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        action_by_sha256: dict[str, AdaptiveActionDescriptor] = {}
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
            if value.action_sha256 in action_by_sha256:
                raise ValueError("actions repeat an identity")
            action_by_sha256[value.action_sha256] = value
        return action_by_sha256

    def design(
        self,
        *,
        adaptive_step: int,
        remaining_authoritative_slots_after_decision: int,
        decision: AdaptiveActionRacingDecision,
        fallback: AdaptiveActionRacingDecision,
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> SamePrefixPairedAuditPlan | None:
        """Return no assay for abstention, unchanged action, or an early step."""

        self.__post_init__()
        if type(adaptive_step) is not int or adaptive_step <= 0:
            raise ValueError("adaptive_step must be positive")
        if (
            type(remaining_authoritative_slots_after_decision) is not int
            or remaining_authoritative_slots_after_decision < 0
        ):
            raise ValueError(
                "remaining_authoritative_slots_after_decision must be "
                "non-negative"
            )
        if (
            self.final_continuation_only
            and remaining_authoritative_slots_after_decision != 0
        ):
            return None
        if type(decision) is not AdaptiveActionRacingDecision:
            raise TypeError("decision must be exact")
        if type(fallback) is not AdaptiveActionRacingDecision:
            raise TypeError("fallback must be exact")
        decision.__post_init__()
        fallback.__post_init__()
        if (
            decision.wave is AdaptiveActionWave.DIAGNOSTIC
            or fallback.wave is AdaptiveActionWave.DIAGNOSTIC
            or decision.wave is not fallback.wave
        ):
            raise ValueError("forecast shadow requires one continuation wave")
        if (
            len(decision.selected_action_sha256s) != 1
            or len(fallback.selected_action_sha256s) != 1
        ):
            raise ValueError("forecast shadow arms must each select one action")
        if (
            decision.residual_request_sha256
            != fallback.residual_request_sha256
            or decision.prior_selected_action_sha256s
            != fallback.prior_selected_action_sha256s
            or decision.observed_outcome_sha256s
            != fallback.observed_outcome_sha256s
            or decision.observed_set_outcome_sha256s
            != fallback.observed_set_outcome_sha256s
        ):
            raise ValueError("forecast challenger and fallback cutoffs differ")
        challenger_action_sha256 = decision.selected_action_sha256s[0]
        fallback_action_sha256 = fallback.selected_action_sha256s[0]
        if challenger_action_sha256 == fallback_action_sha256:
            return None
        action_by_sha256 = self._validate_action_market(actions)
        prefix = set(decision.prior_selected_action_sha256s)
        if (
            not prefix
            or not prefix.issubset(action_by_sha256)
            or challenger_action_sha256 not in action_by_sha256
            or fallback_action_sha256 not in action_by_sha256
            or {
                challenger_action_sha256,
                fallback_action_sha256,
            }
            & prefix
        ):
            raise ValueError("forecast shadow arms are outside the open market")
        evidence = thaw_json(decision.evidence)
        if (
            evidence.get("selection_source")
            != "current_prefix_forecast_opportunity"
            or evidence.get("fallback_preserved_on_abstention") is not True
            or evidence.get("eligible_candidate_outcomes_observed") is not False
        ):
            raise ValueError(
                "decision lacks the protected forecast-opportunity contract"
            )
        embedded_fallback = evidence.get("fallback_decision")
        expected_fallback = fallback.to_record(include_evidence=True)
        if embedded_fallback != expected_fallback:
            raise ValueError(
                "decision does not authenticate the supplied fallback"
            )
        return SamePrefixPairedAuditPlan(
            designer_id=self.designer_id,
            designer_version=self.designer_version,
            designer_definition_sha256=self.definition_sha256,
            residual_request_sha256=decision.residual_request_sha256,
            racing_decision_sha256=decision.decision_sha256,
            common_prefix_action_sha256s=(
                decision.prior_selected_action_sha256s
            ),
            authoritative_arm=SamePrefixPairedAuditArm.EXPLORATION,
            authoritative_action_sha256=challenger_action_sha256,
            legacy_action_sha256=fallback_action_sha256,
            exploration_action_sha256=challenger_action_sha256,
            exploration_stratum_key=(
                "selection_source",
                "forecast_opportunity",
            ),
            distinct_exploration_support_action_sha256s=(
                challenger_action_sha256,
            ),
            exploration_selection_propensity=1.0,
            evidence=freeze_json(
                {
                    "adaptive_step": adaptive_step,
                    "remaining_authoritative_slots_after_decision": (
                        remaining_authoritative_slots_after_decision
                    ),
                    "challenger_decision_sha256": decision.decision_sha256,
                    "fallback_decision_sha256": fallback.decision_sha256,
                    "challenger_action_sha256": challenger_action_sha256,
                    "fallback_action_sha256": fallback_action_sha256,
                    "legacy_arm_semantic_role": "authenticated_fallback",
                    "exploration_arm_semantic_role": (
                        "forecast_opportunity_challenger"
                    ),
                    "final_continuation_only": (
                        self.final_continuation_only
                    ),
                    "current_arm_outcomes_observed": False,
                    "plan_must_be_committed_before_arm_evaluation": True,
                    "assay_union_may_enter_authoritative_archive": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ForecastStratifiedSamePrefixAuditDesigner:
    """Acquire one same-prefix forecast audit even when CPO abstains.

    One continuation position is selected uniformly and deterministically from
    the opaque residual-request identity.  At that position, an unchanged
    protected fallback remains authoritative while a forecast-covered action
    is sampled by a uniform-nonempty-stratum, uniform-within-stratum design.
    If the protected challenger already changed fallback, its selected action
    is reused as the exploration arm.  Neither arm outcome is available while
    the plan is constructed.
    """

    random_seed: int = 0
    designer_id: str = (
        FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
    )
    designer_version: int = (
        FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if (
            self.designer_id
            != FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            or self.designer_version
            != FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_VERSION
        ):
            raise ValueError(
                "forecast-stratified audit designer identity is immutable"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _FORECAST_STRATIFIED_DESIGNER_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "designer_id": self.designer_id,
                    "designer_version": self.designer_version,
                    "random_seed": self.random_seed,
                    "schedule": (
                        "one-hash-uniform-continuation-position-per-request"
                    ),
                    "strata": [
                        "recommended",
                        "adverse_positive",
                        "central_positive_adverse_zero",
                        "favorable_positive_central_zero",
                        "forecast_zero",
                    ],
                    "sampling": (
                        "uniform-nonempty-stratum-then-uniform-action"
                    ),
                    "abstention_authoritative_arm": "protected_fallback",
                    "intervention_authoritative_arm": (
                        "forecast_opportunity_challenger"
                    ),
                    "counterfactual_action_quarantined_after_assay": True,
                    "plan_frozen_before_either_arm_outcome": True,
                    "assay_union_may_enter_authoritative_archive": False,
                    "workload_objective_model_provider_prompt_config_"
                    "branches": False,
                },
            ),
        )

    @staticmethod
    def _validate_action_market(
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> dict[str, AdaptiveActionDescriptor]:
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        action_by_sha256: dict[str, AdaptiveActionDescriptor] = {}
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
            if value.action_sha256 in action_by_sha256:
                raise ValueError("actions repeat an identity")
            action_by_sha256[value.action_sha256] = value
        return action_by_sha256

    @staticmethod
    def _score_stratum(
        *,
        score: dict[str, object],
        recommended_action_sha256s: set[str],
    ) -> str:
        action_sha256 = score.get("action_sha256")
        if type(action_sha256) is not str:
            raise TypeError("opportunity score lacks an action identity")
        if action_sha256 in recommended_action_sha256s:
            return "recommended"
        adverse = _read_nonnegative_hex(score, "adverse_gain_hex")
        central = _read_nonnegative_hex(score, "central_gain_hex")
        favorable = _read_nonnegative_hex(score, "favorable_gain_hex")
        if adverse > 0.0:
            return "adverse_positive"
        if central > 0.0:
            return "central_positive_adverse_zero"
        if favorable > 0.0:
            return "favorable_positive_central_zero"
        return "forecast_zero"

    @classmethod
    def _read_forecast_strata(
        cls,
        *,
        evidence: dict[str, object],
        action_by_sha256: dict[str, AdaptiveActionDescriptor],
        prefix: set[str],
        legacy_action_sha256: str,
    ) -> tuple[
        tuple[str, tuple[str, ...]],
        ...,
    ]:
        ranking = evidence.get("opportunity_ranking")
        if type(ranking) is not dict:
            raise TypeError("decision lacks an opportunity ranking")
        if (
            ranking.get("eligible_candidate_outcomes_observed") is not False
        ):
            raise ValueError(
                "forecast audit ranking observed eligible outcomes"
            )
        raw_recommended = ranking.get("recommended_action_sha256s")
        raw_eligible = ranking.get("eligible_action_sha256s")
        raw_scores = ranking.get("scores")
        if (
            type(raw_recommended) is not list
            or type(raw_eligible) is not list
            or type(raw_scores) is not list
        ):
            raise TypeError("opportunity ranking is incomplete")
        recommended = set(raw_recommended)
        eligible = tuple(raw_eligible)
        if (
            len(recommended) != len(raw_recommended)
            or eligible != tuple(sorted(set(eligible)))
            or not recommended <= set(eligible)
        ):
            raise ValueError("opportunity ranking support is not canonical")
        grouped: dict[str, list[str]] = {}
        seen_scores: set[str] = set()
        for raw_score in raw_scores:
            if type(raw_score) is not dict:
                raise TypeError("opportunity score must be an exact object")
            action_sha256 = raw_score.get("action_sha256")
            if type(action_sha256) is not str:
                raise TypeError("opportunity score lacks an action identity")
            require_sha256(action_sha256, "opportunity action_sha256")
            if (
                action_sha256 in seen_scores
                or action_sha256 not in set(eligible)
            ):
                raise ValueError(
                    "opportunity scores differ from eligible support"
                )
            seen_scores.add(action_sha256)
            if (
                action_sha256 == legacy_action_sha256
                or action_sha256 in prefix
                or action_sha256 not in action_by_sha256
            ):
                continue
            stratum = cls._score_stratum(
                score=raw_score,
                recommended_action_sha256s=recommended,
            )
            grouped.setdefault(stratum, []).append(action_sha256)
        if seen_scores != set(eligible):
            raise ValueError(
                "opportunity scores do not cover eligible support"
            )
        return tuple(
            sorted(
                (
                    stratum,
                    tuple(sorted(action_sha256s)),
                )
                for stratum, action_sha256s in grouped.items()
                if action_sha256s
            )
        )

    def design(
        self,
        *,
        adaptive_step: int,
        remaining_authoritative_slots_after_decision: int,
        decision: AdaptiveActionRacingDecision,
        fallback: AdaptiveActionRacingDecision,
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> SamePrefixPairedAuditPlan | None:
        """Freeze a propensity-logged audit at one rotated position."""

        self.__post_init__()
        if type(adaptive_step) is not int or adaptive_step <= 0:
            raise ValueError("adaptive_step must be positive")
        if (
            type(remaining_authoritative_slots_after_decision) is not int
            or remaining_authoritative_slots_after_decision < 0
        ):
            raise ValueError(
                "remaining_authoritative_slots_after_decision must be "
                "non-negative"
            )
        if type(decision) is not AdaptiveActionRacingDecision:
            raise TypeError("decision must be exact")
        if type(fallback) is not AdaptiveActionRacingDecision:
            raise TypeError("fallback must be exact")
        decision.__post_init__()
        fallback.__post_init__()
        if (
            decision.wave is AdaptiveActionWave.DIAGNOSTIC
            or fallback.wave is AdaptiveActionWave.DIAGNOSTIC
            or decision.wave is not fallback.wave
            or len(decision.selected_action_sha256s) != 1
            or len(fallback.selected_action_sha256s) != 1
        ):
            raise ValueError(
                "forecast-stratified audit requires one continuation action"
            )
        if (
            decision.residual_request_sha256
            != fallback.residual_request_sha256
            or decision.prior_selected_action_sha256s
            != fallback.prior_selected_action_sha256s
            or decision.observed_outcome_sha256s
            != fallback.observed_outcome_sha256s
            or decision.observed_set_outcome_sha256s
            != fallback.observed_set_outcome_sha256s
        ):
            raise ValueError("forecast challenger and fallback cutoffs differ")
        total_continuation_count = (
            adaptive_step
            + remaining_authoritative_slots_after_decision
        )
        position_draw = _stable_unit_interval(
            self.random_seed,
            decision.residual_request_sha256,
            "forecast_stratified_audit_position",
            total_continuation_count,
        )
        target_adaptive_step = 1 + min(
            int(position_draw * total_continuation_count),
            total_continuation_count - 1,
        )
        if adaptive_step != target_adaptive_step:
            return None

        action_by_sha256 = self._validate_action_market(actions)
        prefix = set(decision.prior_selected_action_sha256s)
        authoritative_action_sha256 = (
            decision.selected_action_sha256s[0]
        )
        legacy_action_sha256 = fallback.selected_action_sha256s[0]
        if (
            not prefix
            or not prefix.issubset(action_by_sha256)
            or authoritative_action_sha256 not in action_by_sha256
            or legacy_action_sha256 not in action_by_sha256
            or authoritative_action_sha256 in prefix
            or legacy_action_sha256 in prefix
        ):
            raise ValueError(
                "forecast audit arms are outside the open market"
            )
        # One authoritative action and one quarantined shadow must leave
        # enough distinct actions for every remaining authoritative slot.
        if (
            len(action_by_sha256) - len(prefix)
            < remaining_authoritative_slots_after_decision + 2
        ):
            return None
        evidence = thaw_json(decision.evidence)
        if (
            evidence.get("fallback_preserved_on_abstention") is not True
            or evidence.get("eligible_candidate_outcomes_observed") is not False
            or evidence.get("fallback_decision")
            != fallback.to_record(include_evidence=True)
        ):
            raise ValueError(
                "decision lacks the protected forecast-opportunity contract"
            )
        selection_source = evidence.get("selection_source")
        if selection_source not in {
            "current_prefix_forecast_opportunity",
            "protected_fallback",
        }:
            raise ValueError("decision has an unknown selection source")
        if (
            selection_source == "protected_fallback"
            and authoritative_action_sha256 != legacy_action_sha256
        ):
            raise ValueError(
                "protected fallback source changed the fallback action"
            )
        strata = self._read_forecast_strata(
            evidence=evidence,
            action_by_sha256=action_by_sha256,
            prefix=prefix,
            legacy_action_sha256=legacy_action_sha256,
        )
        if not strata:
            return None
        stratum_by_action = {
            action_sha256: stratum
            for stratum, action_sha256s in strata
            for action_sha256 in action_sha256s
        }
        reused_challenger = (
            authoritative_action_sha256 != legacy_action_sha256
        )
        stratum_draw: float | None = None
        action_draw: float | None = None
        selected_stratum_index: int
        selected_action_index: int
        if reused_challenger:
            stratum = stratum_by_action.get(
                authoritative_action_sha256
            )
            if stratum is None:
                raise ValueError(
                    "forecast challenger is outside forecast audit support"
                )
            selected_stratum_index = tuple(
                value[0] for value in strata
            ).index(stratum)
            selected_actions = strata[selected_stratum_index][1]
            selected_action_index = selected_actions.index(
                authoritative_action_sha256
            )
            exploration_action_sha256 = authoritative_action_sha256
            exploration_propensity = 1.0
        else:
            stratum_draw = _stable_unit_interval(
                self.random_seed,
                decision.residual_request_sha256,
                decision.decision_sha256,
                "forecast_stratified_audit_stratum",
                [
                    [stratum, list(action_sha256s)]
                    for stratum, action_sha256s in strata
                ],
            )
            selected_stratum_index = min(
                int(stratum_draw * len(strata)),
                len(strata) - 1,
            )
            selected_actions = strata[selected_stratum_index][1]
            action_draw = _stable_unit_interval(
                self.random_seed,
                decision.residual_request_sha256,
                decision.decision_sha256,
                "forecast_stratified_audit_action",
                strata[selected_stratum_index][0],
                list(selected_actions),
            )
            selected_action_index = min(
                int(action_draw * len(selected_actions)),
                len(selected_actions) - 1,
            )
            exploration_action_sha256 = selected_actions[
                selected_action_index
            ]
            exploration_propensity = (
                1.0 / len(strata) / len(selected_actions)
            )
        exploration_stratum = strata[selected_stratum_index][0]
        if exploration_action_sha256 == legacy_action_sha256:
            raise RuntimeError("forecast audit arms unexpectedly coincide")
        authoritative_arm = (
            SamePrefixPairedAuditArm.EXPLORATION
            if reused_challenger
            else SamePrefixPairedAuditArm.LEGACY
        )
        distinct_support = tuple(
            sorted(
                action_sha256
                for _, action_sha256s in strata
                for action_sha256 in action_sha256s
            )
        )
        ranking = evidence["opportunity_ranking"]
        if type(ranking) is not dict:
            raise TypeError("decision lacks an opportunity ranking")
        ranking_sha256 = ranking.get("ranking_sha256")
        if type(ranking_sha256) is not str:
            raise TypeError("opportunity ranking lacks its identity")
        require_sha256(ranking_sha256, "opportunity ranking_sha256")
        return SamePrefixPairedAuditPlan(
            designer_id=self.designer_id,
            designer_version=self.designer_version,
            designer_definition_sha256=self.definition_sha256,
            residual_request_sha256=decision.residual_request_sha256,
            racing_decision_sha256=decision.decision_sha256,
            common_prefix_action_sha256s=(
                decision.prior_selected_action_sha256s
            ),
            authoritative_arm=authoritative_arm,
            authoritative_action_sha256=authoritative_action_sha256,
            legacy_action_sha256=legacy_action_sha256,
            exploration_action_sha256=exploration_action_sha256,
            exploration_stratum_key=(
                "forecast_geometry",
                exploration_stratum,
            ),
            distinct_exploration_support_action_sha256s=(
                distinct_support
            ),
            exploration_selection_propensity=float(
                exploration_propensity
            ),
            evidence=freeze_json(
                {
                    "adaptive_step": adaptive_step,
                    "total_continuation_count": (
                        total_continuation_count
                    ),
                    "target_adaptive_step": target_adaptive_step,
                    "remaining_authoritative_slots_after_decision": (
                        remaining_authoritative_slots_after_decision
                    ),
                    "position_draw_hex": position_draw.hex(),
                    "selection_source": selection_source,
                    "fallback_decision_sha256": (
                        fallback.decision_sha256
                    ),
                    "opportunity_ranking_sha256": ranking_sha256,
                    "strata": [
                        {
                            "stratum": stratum,
                            "action_sha256s": list(action_sha256s),
                            "conditional_stratum_propensity_hex": (
                                (1.0 / len(strata)).hex()
                            ),
                            "conditional_action_propensity_hex": (
                                (1.0 / len(action_sha256s)).hex()
                            ),
                        }
                        for stratum, action_sha256s in strata
                    ],
                    "stratum_draw_hex": (
                        None
                        if stratum_draw is None
                        else stratum_draw.hex()
                    ),
                    "action_draw_hex": (
                        None
                        if action_draw is None
                        else action_draw.hex()
                    ),
                    "selected_stratum_index": (
                        selected_stratum_index
                    ),
                    "selected_action_index": selected_action_index,
                    "challenger_reused": reused_challenger,
                    "authoritative_semantic_role": (
                        "forecast_opportunity_challenger"
                        if reused_challenger
                        else "protected_fallback"
                    ),
                    "counterfactual_semantic_role": (
                        "protected_fallback"
                        if reused_challenger
                        else "forecast_stratum_action"
                    ),
                    "current_arm_outcomes_observed": False,
                    "plan_must_be_committed_before_arm_evaluation": True,
                    "counterfactual_action_must_be_quarantined": True,
                    "assay_union_may_enter_authoritative_archive": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FactorStratifiedSamePrefixPairedAuditDesigner:
    """Choose one distinct exploration arm from a v6 frozen audit support."""

    random_seed: int = 0
    designer_id: str = SAME_PREFIX_PAIRED_AUDIT_DESIGNER_ID
    designer_version: int = SAME_PREFIX_PAIRED_AUDIT_DESIGNER_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.designer_id != SAME_PREFIX_PAIRED_AUDIT_DESIGNER_ID:
            raise ValueError("designer_id is immutable")
        if (
            self.designer_version
            != SAME_PREFIX_PAIRED_AUDIT_DESIGNER_VERSION
        ):
            raise ValueError("designer_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DESIGNER_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "designer_id": self.designer_id,
                    "designer_version": self.designer_version,
                    "random_seed": self.random_seed,
                    "input": (
                        "authenticated-v6-decision-and-portable-action-market"
                    ),
                    "legacy_arm": "frozen-v6-legacy-audit-anchor",
                    "exploration_arm": (
                        "uniform-nonempty-stratum-then-uniform-action-"
                        "conditioned-distinct-from-legacy"
                    ),
                    "current_arm_outcomes_observed": False,
                    "assay_union_may_enter_authoritative_archive": False,
                    "workload_objective_model_provider_prompt_config_branches": (
                        False
                    ),
                },
            ),
        )

    @staticmethod
    def _read_strata(
        *,
        evidence: dict[str, object],
        action_by_sha256: dict[str, AdaptiveActionDescriptor],
        prefix: set[str],
        legacy_action_sha256: str,
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
        raw_strata = evidence.get("audit_strata")
        if type(raw_strata) is not list or not raw_strata:
            raise ValueError("v6 decision does not expose audit strata")
        strata: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        seen_actions: set[str] = set()
        for raw_stratum in raw_strata:
            if type(raw_stratum) is not dict:
                raise TypeError("audit stratum must be an exact object")
            raw_key = raw_stratum.get("stratum_key")
            raw_actions = raw_stratum.get("action_sha256s")
            if (
                type(raw_key) is not list
                or not raw_key
                or type(raw_actions) is not list
                or not raw_actions
            ):
                raise ValueError("audit stratum is incomplete")
            key = tuple(raw_key)
            for value in key:
                _require_token(value, name="audit stratum level")
            action_sha256s = tuple(sorted(set(raw_actions)))
            if len(action_sha256s) != len(raw_actions):
                raise ValueError("audit stratum repeats an action")
            for value in action_sha256s:
                require_sha256(value, "audit stratum action_sha256")
                if value not in action_by_sha256:
                    raise ValueError(
                        "audit stratum action is outside the frozen market"
                    )
                if value in prefix:
                    raise ValueError(
                        "audit stratum action is already in the prefix"
                    )
                if value in seen_actions:
                    raise ValueError(
                        "audit action appears in multiple strata"
                    )
                seen_actions.add(value)
            distinct_actions = tuple(
                value
                for value in action_sha256s
                if value != legacy_action_sha256
            )
            if distinct_actions:
                strata.append((key, distinct_actions))
        if not strata:
            raise ValueError(
                "paired audit has no exploration action distinct from legacy"
            )
        return tuple(sorted(strata))

    def design(
        self,
        *,
        decision: AdaptiveActionRacingDecision,
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> SamePrefixPairedAuditPlan:
        """Freeze two distinct arms without accepting either arm's outcome."""

        self.__post_init__()
        if type(decision) is not AdaptiveActionRacingDecision:
            raise TypeError("decision must be exact")
        decision.__post_init__()
        if (
            decision.policy_id != OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_ID
            or decision.policy_version
            != OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
            or decision.wave is not AdaptiveActionWave.RANDOMIZED_AUDIT
            or len(decision.selected_action_sha256s) != 1
        ):
            raise ValueError(
                "paired audit requires one final factor-stratified v6 decision"
            )
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        action_by_sha256: dict[str, AdaptiveActionDescriptor] = {}
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
            if value.action_sha256 in action_by_sha256:
                raise ValueError("actions repeat an identity")
            action_by_sha256[value.action_sha256] = value
        prefix = set(decision.prior_selected_action_sha256s)
        if not prefix.issubset(action_by_sha256):
            raise ValueError("decision prefix is outside the frozen market")
        evidence = thaw_json(decision.evidence)
        if (
            evidence.get("risk_controlled_stratified_audit") is not True
            or evidence.get("candidate_factor_cells_outcome_blind") is not True
        ):
            raise ValueError("decision lacks the v6 outcome-blind audit contract")
        legacy_action_sha256 = evidence.get(
            "legacy_audit_anchor_action_sha256"
        )
        if type(legacy_action_sha256) is not str:
            raise TypeError("decision lacks a legacy audit anchor")
        require_sha256(
            legacy_action_sha256,
            "legacy_audit_anchor_action_sha256",
        )
        if (
            legacy_action_sha256 not in action_by_sha256
            or legacy_action_sha256 in prefix
        ):
            raise ValueError("legacy audit anchor is outside the remaining market")
        authoritative_action_sha256 = decision.selected_action_sha256s[0]
        if (
            authoritative_action_sha256 not in action_by_sha256
            or authoritative_action_sha256 in prefix
        ):
            raise ValueError(
                "authoritative audit action is outside the remaining market"
            )
        strata = self._read_strata(
            evidence=evidence,
            action_by_sha256=action_by_sha256,
            prefix=prefix,
            legacy_action_sha256=legacy_action_sha256,
        )
        authoritative_is_distinct_exploration = (
            evidence.get("audit_exploration_branch") is True
            and authoritative_action_sha256 != legacy_action_sha256
        )
        selected_stratum_index: int | None = None
        selected_action_index: int | None = None
        stratum_draw: float | None = None
        action_draw: float | None = None
        if authoritative_is_distinct_exploration:
            for stratum_index, (_, stratum_actions) in enumerate(strata):
                if authoritative_action_sha256 in stratum_actions:
                    selected_stratum_index = stratum_index
                    selected_action_index = stratum_actions.index(
                        authoritative_action_sha256
                    )
                    break
            if selected_stratum_index is None:
                raise ValueError(
                    "selected exploration action is outside distinct support"
                )
        else:
            stratum_draw = _stable_unit_interval(
                self.random_seed,
                decision.residual_request_sha256,
                decision.decision_sha256,
                "paired_audit_distinct_stratum",
                [
                    [list(key), list(stratum_actions)]
                    for key, stratum_actions in strata
                ],
            )
            selected_stratum_index = min(
                int(stratum_draw * len(strata)),
                len(strata) - 1,
            )
            selected_key, selected_actions = strata[selected_stratum_index]
            action_draw = _stable_unit_interval(
                self.random_seed,
                decision.residual_request_sha256,
                decision.decision_sha256,
                "paired_audit_distinct_action",
                list(selected_key),
                list(selected_actions),
            )
            selected_action_index = min(
                int(action_draw * len(selected_actions)),
                len(selected_actions) - 1,
            )
        if selected_stratum_index is None or selected_action_index is None:
            raise RuntimeError("paired audit did not resolve one exploration arm")
        exploration_stratum_key, stratum_actions = strata[
            selected_stratum_index
        ]
        exploration_action_sha256 = stratum_actions[selected_action_index]
        exploration_propensity = 1.0 / len(strata) / len(stratum_actions)
        authoritative_arm = (
            SamePrefixPairedAuditArm.LEGACY
            if authoritative_action_sha256 == legacy_action_sha256
            else SamePrefixPairedAuditArm.EXPLORATION
        )
        distinct_support = tuple(
            sorted(
                value
                for _, stratum_actions in strata
                for value in stratum_actions
            )
        )
        return SamePrefixPairedAuditPlan(
            designer_id=self.designer_id,
            designer_version=self.designer_version,
            designer_definition_sha256=self.definition_sha256,
            residual_request_sha256=decision.residual_request_sha256,
            racing_decision_sha256=decision.decision_sha256,
            common_prefix_action_sha256s=(
                decision.prior_selected_action_sha256s
            ),
            authoritative_arm=authoritative_arm,
            authoritative_action_sha256=authoritative_action_sha256,
            legacy_action_sha256=legacy_action_sha256,
            exploration_action_sha256=exploration_action_sha256,
            exploration_stratum_key=exploration_stratum_key,
            distinct_exploration_support_action_sha256s=distinct_support,
            exploration_selection_propensity=float(
                exploration_propensity
            ),
            evidence=freeze_json(
                {
                    "source_racing_decision": decision.to_record(
                        include_evidence=False
                    ),
                    "source_audit_exploration_branch": evidence.get(
                        "audit_exploration_branch"
                    ),
                    "source_audit_branch_draw_hex": evidence.get(
                        "audit_branch_draw_hex"
                    ),
                    "conditioned_distinct_from_legacy": True,
                    "distinct_strata": [
                        {
                            "stratum_key": list(key),
                            "action_sha256s": list(stratum_actions),
                            "conditional_action_propensity_hex": (
                                (1.0 / len(stratum_actions)).hex()
                            ),
                        }
                        for key, stratum_actions in strata
                    ],
                    "stratum_draw_hex": (
                        None
                        if stratum_draw is None
                        else stratum_draw.hex()
                    ),
                    "action_draw_hex": (
                        None
                        if action_draw is None
                        else action_draw.hex()
                    ),
                    "selected_stratum_index": selected_stratum_index,
                    "selected_action_index": selected_action_index,
                    "exploration_selection_propensity_hex": (
                        exploration_propensity.hex()
                    ),
                    "authoritative_action_reused_when_exploration": (
                        authoritative_is_distinct_exploration
                    ),
                    "current_arm_outcomes_observed": False,
                    "plan_must_be_committed_before_arm_evaluation": True,
                    "assay_union_may_enter_authoritative_archive": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SamePrefixPairedAuditObservation:
    """Two independently valued arms joined to one frozen common prefix."""

    plan: SamePrefixPairedAuditPlan
    legacy_outcome: AdaptiveActionOutcome
    exploration_outcome: AdaptiveActionOutcome
    legacy_set_outcome: AdaptiveActionSetOutcome
    exploration_set_outcome: AdaptiveActionSetOutcome
    adjudicator_id: str
    adjudicator_version: int
    adjudicator_definition_sha256: str
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not SamePrefixPairedAuditPlan:
            raise TypeError("plan must be exact")
        self.plan.__post_init__()
        for value, name in (
            (self.legacy_outcome, "legacy_outcome"),
            (self.exploration_outcome, "exploration_outcome"),
        ):
            if type(value) is not AdaptiveActionOutcome:
                raise TypeError(f"{name} must be exact")
            value.__post_init__()
        for value, name in (
            (self.legacy_set_outcome, "legacy_set_outcome"),
            (self.exploration_set_outcome, "exploration_set_outcome"),
        ):
            if type(value) is not AdaptiveActionSetOutcome:
                raise TypeError(f"{name} must be exact")
            value.__post_init__()
        if (
            self.legacy_outcome.action_sha256
            != self.plan.legacy_action_sha256
            or self.exploration_outcome.action_sha256
            != self.plan.exploration_action_sha256
        ):
            raise ValueError("arm outcome does not match the frozen plan")
        expected_prefix = set(self.plan.common_prefix_action_sha256s)
        prior_bindings = self.legacy_set_outcome.prior_action_evaluation_bindings
        if (
            prior_bindings
            != self.exploration_set_outcome.prior_action_evaluation_bindings
            or {value[0] for value in prior_bindings} != expected_prefix
        ):
            raise ValueError("paired arms do not share the exact common prefix")
        if not math.isclose(
            self.legacy_set_outcome.prior_selected_set_gain,
            self.exploration_set_outcome.prior_selected_set_gain,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("paired arms disagree on common-prefix utility")
        expected_current = (
            (
                self.plan.legacy_action_sha256,
                self.legacy_outcome.evaluation_sha256,
            ),
        )
        if self.legacy_set_outcome.current_action_evaluation_bindings != (
            expected_current
        ):
            raise ValueError("legacy set outcome does not join its evaluation")
        expected_current = (
            (
                self.plan.exploration_action_sha256,
                self.exploration_outcome.evaluation_sha256,
            ),
        )
        if (
            self.exploration_set_outcome.current_action_evaluation_bindings
            != expected_current
        ):
            raise ValueError(
                "exploration set outcome does not join its evaluation"
            )
        _require_token(self.adjudicator_id, name="adjudicator_id")
        if (
            type(self.adjudicator_version) is not int
            or self.adjudicator_version <= 0
        ):
            raise ValueError("adjudicator_version must be positive")
        require_sha256(
            self.adjudicator_definition_sha256,
            "adjudicator_definition_sha256",
        )
        object.__setattr__(
            self,
            "observation_sha256",
            _hash(_OBSERVATION_DOMAIN, self._unsigned_record()),
        )

    @property
    def conditional_gain_delta(self) -> float:
        return (
            self.exploration_set_outcome.conditional_set_gain
            - self.legacy_set_outcome.conditional_set_gain
        )

    @property
    def winner(self) -> SamePrefixPairedAuditWinner:
        delta = self.conditional_gain_delta
        if math.isclose(delta, 0.0, rel_tol=1e-12, abs_tol=1e-15):
            return SamePrefixPairedAuditWinner.TIE
        if delta > 0.0:
            return SamePrefixPairedAuditWinner.EXPLORATION
        return SamePrefixPairedAuditWinner.LEGACY

    @property
    def authoritative_set_outcome(self) -> AdaptiveActionSetOutcome:
        if self.plan.authoritative_arm is SamePrefixPairedAuditArm.LEGACY:
            return self.legacy_set_outcome
        return self.exploration_set_outcome

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "adjudicator": {
                "adjudicator_id": self.adjudicator_id,
                "adjudicator_version": self.adjudicator_version,
                "definition_sha256": self.adjudicator_definition_sha256,
            },
            "plan_sha256": self.plan.plan_sha256,
            "legacy_outcome_sha256": self.legacy_outcome.outcome_sha256,
            "exploration_outcome_sha256": (
                self.exploration_outcome.outcome_sha256
            ),
            "legacy_set_outcome_sha256": (
                self.legacy_set_outcome.set_outcome_sha256
            ),
            "exploration_set_outcome_sha256": (
                self.exploration_set_outcome.set_outcome_sha256
            ),
            "common_prefix_selected_set_gain_hex": (
                self.legacy_set_outcome.prior_selected_set_gain.hex()
            ),
            "legacy_conditional_gain_hex": (
                self.legacy_set_outcome.conditional_set_gain.hex()
            ),
            "exploration_conditional_gain_hex": (
                self.exploration_set_outcome.conditional_set_gain.hex()
            ),
            "conditional_gain_delta_hex": (
                self.conditional_gain_delta.hex()
            ),
            "winner": self.winner.value,
            "authoritative_arm": self.plan.authoritative_arm.value,
            "authoritative_set_outcome_sha256": (
                self.authoritative_set_outcome.set_outcome_sha256
            ),
            "assay_union_admitted_to_authoritative_archive": False,
            "counterfactual_endpoints_share_one_prefix": True,
            "workload_objective_model_provider_prompt_config_branches": False,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "plan": self.plan.to_record(include_evidence=include_evidence),
            "legacy_outcome": self.legacy_outcome.to_record(),
            "exploration_outcome": self.exploration_outcome.to_record(),
            "legacy_set_outcome": self.legacy_set_outcome.to_record(),
            "exploration_set_outcome": (
                self.exploration_set_outcome.to_record()
            ),
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True, slots=True)
class SamePrefixPairedAuditAdjudicator:
    """Build one workload-opaque observation from real arm outcomes."""

    adjudicator_id: str = SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_ID
    adjudicator_version: int = SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.adjudicator_id != SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_ID:
            raise ValueError("adjudicator_id is immutable")
        if (
            self.adjudicator_version
            != SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_VERSION
        ):
            raise ValueError("adjudicator_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _ADJUDICATOR_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "adjudicator_id": self.adjudicator_id,
                    "adjudicator_version": self.adjudicator_version,
                    "comparison": (
                        "conditional-set-gain-at-identical-prior-bindings"
                    ),
                    "assay_union_admitted_to_authoritative_archive": False,
                    "workload_objective_model_provider_prompt_config_branches": (
                        False
                    ),
                },
            ),
        )

    def adjudicate(
        self,
        *,
        plan: SamePrefixPairedAuditPlan,
        legacy_outcome: AdaptiveActionOutcome,
        exploration_outcome: AdaptiveActionOutcome,
        legacy_set_outcome: AdaptiveActionSetOutcome,
        exploration_set_outcome: AdaptiveActionSetOutcome,
    ) -> SamePrefixPairedAuditObservation:
        self.__post_init__()
        return SamePrefixPairedAuditObservation(
            plan=plan,
            legacy_outcome=legacy_outcome,
            exploration_outcome=exploration_outcome,
            legacy_set_outcome=legacy_set_outcome,
            exploration_set_outcome=exploration_set_outcome,
            adjudicator_id=self.adjudicator_id,
            adjudicator_version=self.adjudicator_version,
            adjudicator_definition_sha256=self.definition_sha256,
        )


__all__ = [
    "FactorStratifiedSamePrefixPairedAuditDesigner",
    "FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS",
    "FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID",
    "FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_VERSION",
    "FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID",
    "FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_VERSION",
    "ForecastOpportunitySamePrefixAuditDesignerPort",
    "ForecastOpportunitySamePrefixShadowDesigner",
    "ForecastOpportunitySamePrefixShadowDesignerPort",
    "ForecastStratifiedSamePrefixAuditDesigner",
    "SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_ID",
    "SAME_PREFIX_PAIRED_AUDIT_ADJUDICATOR_VERSION",
    "SAME_PREFIX_PAIRED_AUDIT_DESIGNER_ID",
    "SAME_PREFIX_PAIRED_AUDIT_DESIGNER_VERSION",
    "SamePrefixPairedAuditAdjudicator",
    "SamePrefixPairedAuditArm",
    "SamePrefixPairedAuditDesignerPort",
    "SamePrefixPairedAuditObservation",
    "SamePrefixPairedAuditPlan",
    "SamePrefixPairedAuditWinner",
]
