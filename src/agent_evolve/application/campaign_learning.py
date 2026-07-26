"""Closed-loop lifecycle for reflected campaign insights.

This module joins three evidence channels without conflating them:

* a reflection creates an unverified card that remains quarantined;
* a campaign admission barrier may expose that exact version only to a
  randomized diagnostic assignment;
* a later generation barrier supplies causal search-utility evidence, while a
  separate global audit supplies semantic support or counterexamples.

Normal retrieval becomes possible only when both semantic support and minimum
causal usefulness are present.  All decisions in one generation are prepared
before the memory bank publishes a canonical lifecycle batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

from agent_evolve.application.insight_memory import (
    InsightLifecycleChangeRequest,
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
    QuarantineTestAdmissionReceipt,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryCreditBatchReceipt,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.global_falsification import (
    GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256,
    GLOBAL_FALSIFICATION_POLICY_ID,
    GLOBAL_FALSIFICATION_POLICY_VERSION,
    EvidenceDisposition,
    GlobalHypothesisAuditReceipt,
    GlobalHypothesisAuditRequest,
    GlobalHypothesisVerdict,
    HypothesisAuditScope,
    HypothesisClaimStrength,
    HypothesisMetricPrediction,
    TypedEvidencePredicate,
    TypedInterventionSignature,
)
from agent_evolve.policies.memory.randomized_subset import (
    InsightTrial,
    MarginalEffectEstimate,
    estimate_marginal_effect,
)
from agent_evolve.policies.memory.staged_causal import (
    insight_selection_decision_sha256,
)
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
    ReflectionInsightKind,
)


_REGISTRATION_DOMAIN = b"agent-evolve:campaign-insight-registration:v2\x00"
_ADMISSION_DOMAIN = b"agent-evolve:campaign-diagnostic-admission:v1\x00"
_CAUSAL_DOMAIN = b"agent-evolve:campaign-causal-usefulness:v2\x00"
_PROMOTION_EVIDENCE_DOMAIN = (
    b"agent-evolve:campaign-randomized-promotion-evidence:v1\x00"
)
_DECISION_DOMAIN = b"agent-evolve:campaign-insight-lifecycle-decision:v1\x00"
_BARRIER_DOMAIN = b"agent-evolve:campaign-learning-barrier:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:campaign-insight-promotion-policy:v2\x00"
_SEMANTIC_AUDIT_PLAN_DOMAIN = b"agent-evolve:campaign-semantic-audit-plan:v1\x00"
_PREPARED_BARRIER_DOMAIN = b"agent-evolve:campaign-prepared-learning-barrier:v1\x00"


def _exact_one_sided_sign_p_value(*, successes: int, blocks: int) -> Fraction:
    """Exact upper binomial tail under a fair-sign null.

    A block whose effect is equal to the practical margin is conservatively a
    failure.  The returned rational is retained in receipts so promotion does
    not depend on platform-specific special functions or rounded p-values.
    """

    if type(successes) is not int or type(blocks) is not int:
        raise TypeError("sign-test counts must be exact integers")
    if successes < 0 or blocks < 0 or successes > blocks:
        raise ValueError("sign-test counts are invalid")
    if blocks == 0:
        return Fraction(1, 1)
    return Fraction(
        sum(math.comb(blocks, value) for value in range(successes, blocks + 1)),
        2**blocks,
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


def _reference_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _trial_record(trial: InsightTrial) -> dict[str, object]:
    InsightTrial.__post_init__(trial)
    return {
        "credit_unit_id": trial.credit_unit_id.value,
        "candidate_ids": [value.value for value in trial.candidate_ids],
        "reward_definition_sha256": trial.reward_definition_hash,
        "selection_decision_sha256": insight_selection_decision_sha256(trial.decision),
        "reward_hex": trial.reward.hex(),
        "treatment_binding_sha256": trial.treatment_binding_sha256,
    }


@dataclass(frozen=True, slots=True)
class CampaignSemanticAuditPlan:
    """Pre-outcome identity for the exact semantic claim that may be audited.

    Registry identity and the inclusive audit cutoff are intentionally absent:
    those values exist only once the later evidence universe is sealed.  Every
    semantic degree of freedom, including the origin cutoff, is fixed before a
    quarantined card may enter diagnostic testing.

    The plan reuses the global falsification layer's typed predicate,
    intervention, prediction, strength, and scope constructs.  It deliberately
    does not reuse the parent-bound executable-treatment compilation receipt:
    that receipt answers whether one current finite palette can execute a card,
    whereas this plan identifies the claim audited across a sealed registry.
    """

    reference: InsightRef
    draft_content_sha256: str
    draft_hypothesis_sha256: str
    trigger: TypedEvidencePredicate
    intervention: TypedInterventionSignature
    predictions: tuple[HypothesisMetricPrediction, ...]
    claim_strength: HypothesisClaimStrength
    scope: HypothesisAuditScope
    matcher_definition_sha256: str
    origin_cutoff_event_index: int
    minimum_support_clusters: int
    minimum_support_instances: int
    audit_policy_definition_sha256: str = GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.draft_content_sha256, "draft_content_sha256")
        require_sha256(self.draft_hypothesis_sha256, "draft_hypothesis_sha256")
        if type(self.trigger) is not TypedEvidencePredicate:
            raise TypeError("trigger must be an exact TypedEvidencePredicate")
        TypedEvidencePredicate.__post_init__(self.trigger)
        if type(self.intervention) is not TypedInterventionSignature:
            raise TypeError("intervention must be an exact TypedInterventionSignature")
        TypedInterventionSignature.__post_init__(self.intervention)
        if (
            type(self.predictions) is not tuple
            or not self.predictions
            or any(
                type(value) is not HypothesisMetricPrediction
                for value in self.predictions
            )
        ):
            raise ValueError("predictions must contain exact typed predictions")
        for prediction in self.predictions:
            HypothesisMetricPrediction.__post_init__(prediction)
        if type(self.claim_strength) is not HypothesisClaimStrength:
            raise TypeError("claim_strength must be exact")
        HypothesisClaimStrength.__post_init__(self.claim_strength)
        if type(self.scope) is not HypothesisAuditScope:
            raise TypeError("scope must be exact")
        HypothesisAuditScope.__post_init__(self.scope)
        require_sha256(self.matcher_definition_sha256, "matcher_definition_sha256")
        if (
            type(self.origin_cutoff_event_index) is not int
            or self.origin_cutoff_event_index < 0
        ):
            raise ValueError("origin_cutoff_event_index must be non-negative")
        for name in ("minimum_support_clusters", "minimum_support_instances"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            self.audit_policy_definition_sha256
            != GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported global falsification policy definition")
        object.__setattr__(
            self,
            "plan_sha256",
            _hash(_SEMANTIC_AUDIT_PLAN_DOMAIN, self._unsigned_record()),
        )

    @classmethod
    def from_request(
        cls,
        request: GlobalHypothesisAuditRequest,
        *,
        draft_hypothesis_sha256: str,
    ) -> "CampaignSemanticAuditPlan":
        if type(request) is not GlobalHypothesisAuditRequest:
            raise TypeError("request must be an exact GlobalHypothesisAuditRequest")
        GlobalHypothesisAuditRequest.__post_init__(request)
        return cls(
            reference=request.reference,
            draft_content_sha256=request.draft_content_sha256,
            draft_hypothesis_sha256=draft_hypothesis_sha256,
            trigger=request.trigger,
            intervention=request.intervention,
            predictions=request.predictions,
            claim_strength=request.claim_strength,
            scope=request.scope,
            matcher_definition_sha256=request.matcher_definition_sha256,
            origin_cutoff_event_index=request.origin_cutoff_event_index,
            minimum_support_clusters=request.minimum_support_clusters,
            minimum_support_instances=request.minimum_support_instances,
            audit_policy_definition_sha256=(request.audit_policy_definition_sha256),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "draft_content_sha256": self.draft_content_sha256,
            "draft_hypothesis_sha256": self.draft_hypothesis_sha256,
            "trigger_predicate_sha256": self.trigger.predicate_sha256,
            "intervention_signature_sha256": self.intervention.signature_sha256,
            "predictions": [value.to_record() for value in self.predictions],
            "claim_strength": self.claim_strength.to_record(),
            "scope_sha256": self.scope.scope_sha256,
            "matcher_definition_sha256": self.matcher_definition_sha256,
            "origin_cutoff_event_index": self.origin_cutoff_event_index,
            "minimum_support_clusters": self.minimum_support_clusters,
            "minimum_support_instances": self.minimum_support_instances,
            "audit_policy_definition_sha256": (self.audit_policy_definition_sha256),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "plan_sha256": self.plan_sha256}

    def admits(self, request: GlobalHypothesisAuditRequest) -> bool:
        """Whether a later sealed request preserves every planned semantic field."""

        if type(request) is not GlobalHypothesisAuditRequest:
            return False
        GlobalHypothesisAuditRequest.__post_init__(request)
        observed = type(self).from_request(
            request,
            draft_hypothesis_sha256=self.draft_hypothesis_sha256,
        )
        return observed.plan_sha256 == self.plan_sha256


@dataclass(frozen=True, slots=True)
class CampaignRandomizedPromotionEvidence:
    """Exact robustness evidence for one randomized insight contrast.

    This is intentionally separate from ``MarginalEffectEstimate``.  The
    latter remains useful for scoring, while this receipt records the stronger
    evidence required to make a quarantined card normally retrievable.
    """

    reference: InsightRef
    exact_context_sha256: str
    source_trial_set_sha256: str
    promotion_policy_definition_sha256: str
    practical_effect_margin: float
    independent_block_unit: str
    block_effects: tuple[tuple[int, float], ...]
    excluded_block_generations: tuple[int, ...]
    missing_generation_trial_count: int
    successful_block_count: int
    exact_one_sided_p_value: Fraction
    family_hypothesis_count: int
    adjusted_alpha: float
    failure_reasons: tuple[str, ...]
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.exact_context_sha256, "exact_context_sha256")
        require_sha256(self.source_trial_set_sha256, "source_trial_set_sha256")
        require_sha256(
            self.promotion_policy_definition_sha256,
            "promotion_policy_definition_sha256",
        )
        for name in ("practical_effect_margin", "adjusted_alpha"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
        if self.practical_effect_margin < 0:
            raise ValueError("practical_effect_margin must be non-negative")
        if not 0 < self.adjusted_alpha <= 0.5:
            raise ValueError("adjusted_alpha must lie in (0, 0.5]")
        if self.independent_block_unit != "campaign_generation":
            raise ValueError("unsupported independent_block_unit")
        if type(self.block_effects) is not tuple:
            raise TypeError("block_effects must be an exact tuple")
        generations: list[int] = []
        effects: list[float] = []
        for item in self.block_effects:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("block effects must be generation/effect pairs")
            generation, effect = item
            if type(generation) is not int or generation <= 0:
                raise ValueError("block generation must be positive")
            if type(effect) is not float or not math.isfinite(effect):
                raise TypeError("block effect must be a finite canonical float")
            generations.append(generation)
            effects.append(effect)
        if tuple(generations) != tuple(sorted(set(generations))):
            raise ValueError("block generations must be unique and canonical")
        if (
            type(self.excluded_block_generations) is not tuple
            or self.excluded_block_generations
            != tuple(sorted(set(self.excluded_block_generations)))
            or set(self.excluded_block_generations).intersection(generations)
        ):
            raise ValueError(
                "excluded block generations must be disjoint and canonical"
            )
        for value in self.excluded_block_generations:
            if type(value) is not int or value <= 0:
                raise ValueError("excluded block generation must be positive")
        if (
            type(self.missing_generation_trial_count) is not int
            or self.missing_generation_trial_count < 0
        ):
            raise ValueError("missing_generation_trial_count must be non-negative")
        expected_successes = sum(
            value > self.practical_effect_margin for value in effects
        )
        if self.successful_block_count != expected_successes:
            raise ValueError("successful_block_count differs from block effects")
        if type(self.exact_one_sided_p_value) is not Fraction:
            raise TypeError("exact_one_sided_p_value must be an exact Fraction")
        expected_p_value = _exact_one_sided_sign_p_value(
            successes=self.successful_block_count,
            blocks=len(self.block_effects),
        )
        if self.exact_one_sided_p_value != expected_p_value:
            raise ValueError("exact sign p-value differs from block effects")
        if (
            type(self.family_hypothesis_count) is not int
            or self.family_hypothesis_count <= 0
        ):
            raise ValueError("family_hypothesis_count must be positive")
        if type(self.failure_reasons) is not tuple or any(
            type(value) is not str or not value for value in self.failure_reasons
        ):
            raise TypeError("failure_reasons must contain non-empty exact strings")
        if self.failure_reasons != tuple(dict.fromkeys(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique and canonical")
        object.__setattr__(
            self,
            "evidence_sha256",
            _hash(_PROMOTION_EVIDENCE_DOMAIN, self._unsigned_record()),
        )

    @property
    def passes_policy(self) -> bool:
        return not self.failure_reasons

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "exact_context_sha256": self.exact_context_sha256,
            "source_trial_set_sha256": self.source_trial_set_sha256,
            "promotion_policy_definition_sha256": (
                self.promotion_policy_definition_sha256
            ),
            "practical_effect_margin_hex": self.practical_effect_margin.hex(),
            "independent_block_unit": self.independent_block_unit,
            "block_effects": [
                {"generation": generation, "effect_hex": effect.hex()}
                for generation, effect in self.block_effects
            ],
            "excluded_block_generations": list(self.excluded_block_generations),
            "missing_generation_trial_count": self.missing_generation_trial_count,
            "successful_block_count": self.successful_block_count,
            "exact_one_sided_p_value": {
                "numerator": self.exact_one_sided_p_value.numerator,
                "denominator": self.exact_one_sided_p_value.denominator,
            },
            "family_hypothesis_count": self.family_hypothesis_count,
            "adjusted_alpha_hex": self.adjusted_alpha.hex(),
            "failure_reasons": list(self.failure_reasons),
            "passes_policy": self.passes_policy,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class CampaignInsightPromotionPolicy:
    """Predeclared conservative evidence for normal insight retrieval.

    Promotion is deliberately harder than estimating a marginal effect.  The
    pooled stabilized-IPW contrast must clear a practical margin, both arms
    must have adequate support, and that margin must recur across distinct
    campaign-generation randomization blocks.  The latter conjunct uses an
    exact one-sided sign test and Bonferroni family-wise error control.

    ``generation`` is the narrowest authenticated block identifier carried by
    every production ``InsightTrial`` today.  Trials without that identifier
    can still update diagnostic scores, but they cannot promote a card.
    """

    minimum_treated_trials: int = 5
    minimum_control_trials: int = 5
    minimum_effective_support: float = 4.0
    minimum_effect: float = 0.01
    minimum_independent_blocks: int = 5
    minimum_treated_trials_per_block: int = 1
    minimum_control_trials_per_block: int = 1
    familywise_error_rate: float = 0.05
    independent_block_unit: str = "campaign_generation"
    evidence_rule: str = "exact_one_sided_block_sign_test"
    multiplicity_correction: str = "bonferroni"
    policy_id: str = "semantic_support_and_robust_randomized_usefulness"
    policy_version: int = 2

    def __post_init__(self) -> None:
        for name in (
            "minimum_treated_trials",
            "minimum_control_trials",
            "minimum_independent_blocks",
            "minimum_treated_trials_per_block",
            "minimum_control_trials_per_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "minimum_effective_support",
            "minimum_effect",
            "familywise_error_rate",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
        if self.minimum_effective_support <= 0:
            raise ValueError("minimum_effective_support must be positive")
        if self.minimum_effect < 0:
            raise ValueError("minimum_effect must be non-negative")
        if not 0 < self.familywise_error_rate <= 0.5:
            raise ValueError("familywise_error_rate must lie in (0, 0.5]")
        if self.independent_block_unit != "campaign_generation":
            raise ValueError("unsupported independent_block_unit")
        if self.evidence_rule != "exact_one_sided_block_sign_test":
            raise ValueError("unsupported promotion evidence_rule")
        if self.multiplicity_correction != "bonferroni":
            raise ValueError("unsupported multiplicity_correction")
        if self.policy_id != "semantic_support_and_robust_randomized_usefulness":
            raise ValueError("unsupported promotion policy_id")
        if type(self.policy_version) is not int or self.policy_version != 2:
            raise ValueError("unsupported promotion policy_version")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "minimum_treated_trials": self.minimum_treated_trials,
            "minimum_control_trials": self.minimum_control_trials,
            "minimum_effective_support_hex": self.minimum_effective_support.hex(),
            "minimum_effect_hex": self.minimum_effect.hex(),
            "minimum_independent_blocks": self.minimum_independent_blocks,
            "minimum_treated_trials_per_block": (self.minimum_treated_trials_per_block),
            "minimum_control_trials_per_block": (self.minimum_control_trials_per_block),
            "familywise_error_rate_hex": self.familywise_error_rate.hex(),
            "independent_block_unit": self.independent_block_unit,
            "evidence_rule": self.evidence_rule,
            "multiplicity_correction": self.multiplicity_correction,
            "effect_estimand": (
                "stabilized_ipw_selected_minus_unselected_portfolio_reward"
            ),
            "block_effect_estimand": (
                "within_generation_stabilized_ipw_selected_minus_unselected_reward"
            ),
            "tie_handling": "effect_equal_to_margin_is_not_success",
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_POLICY_DOMAIN, self.to_record())

    def evaluate(
        self,
        *,
        estimate: MarginalEffectEstimate,
        source_trials: tuple[InsightTrial, ...],
        source_trial_set_sha256: str,
        family_hypothesis_count: int,
    ) -> CampaignRandomizedPromotionEvidence:
        """Build exact, replayable promotion evidence for one insight."""

        self.__post_init__()
        if type(estimate) is not MarginalEffectEstimate:
            raise TypeError("estimate must be an exact MarginalEffectEstimate")
        if estimate.context_hash is None:
            raise ValueError("promotion requires an exact context stratum")
        if type(source_trials) is not tuple or any(
            type(value) is not InsightTrial for value in source_trials
        ):
            raise TypeError("source_trials must contain exact InsightTrial values")
        for value in source_trials:
            InsightTrial.__post_init__(value)
        require_sha256(source_trial_set_sha256, "source_trial_set_sha256")
        if type(family_hypothesis_count) is not int or family_hypothesis_count <= 0:
            raise ValueError("family_hypothesis_count must be positive")

        replayed = estimate_marginal_effect(
            source_trials,
            estimate.insight,
            context_hash=estimate.context_hash,
        )
        if replayed != estimate:
            raise ValueError("promotion estimate differs from its source trials")

        trials_by_generation: dict[int, list[InsightTrial]] = {}
        missing_generation_trial_count = 0
        for trial in source_trials:
            if trial.generation is None:
                missing_generation_trial_count += 1
            else:
                trials_by_generation.setdefault(trial.generation, []).append(trial)

        block_effects: list[tuple[int, float]] = []
        excluded_generations: list[int] = []
        for generation in sorted(trials_by_generation):
            block_estimate = estimate_marginal_effect(
                tuple(trials_by_generation[generation]),
                estimate.insight,
                context_hash=estimate.context_hash,
            )
            if (
                block_estimate.effect is None
                or block_estimate.treated_trials < self.minimum_treated_trials_per_block
                or block_estimate.control_trials < self.minimum_control_trials_per_block
            ):
                excluded_generations.append(generation)
                continue
            block_effects.append((generation, block_estimate.effect))

        successes = sum(effect > self.minimum_effect for _, effect in block_effects)
        p_value = _exact_one_sided_sign_p_value(
            successes=successes,
            blocks=len(block_effects),
        )
        adjusted_alpha = self.familywise_error_rate / family_hypothesis_count
        failure_reasons = self._failure_reasons(
            estimate=estimate,
            block_count=len(block_effects),
            missing_generation_trial_count=missing_generation_trial_count,
            exact_one_sided_p_value=p_value,
            adjusted_alpha=adjusted_alpha,
        )
        return CampaignRandomizedPromotionEvidence(
            reference=estimate.insight,
            exact_context_sha256=estimate.context_hash,
            source_trial_set_sha256=source_trial_set_sha256,
            promotion_policy_definition_sha256=self.definition_sha256,
            practical_effect_margin=self.minimum_effect,
            independent_block_unit=self.independent_block_unit,
            block_effects=tuple(block_effects),
            excluded_block_generations=tuple(excluded_generations),
            missing_generation_trial_count=missing_generation_trial_count,
            successful_block_count=successes,
            exact_one_sided_p_value=p_value,
            family_hypothesis_count=family_hypothesis_count,
            adjusted_alpha=adjusted_alpha,
            failure_reasons=failure_reasons,
        )

    def _failure_reasons(
        self,
        *,
        estimate: MarginalEffectEstimate,
        block_count: int,
        missing_generation_trial_count: int,
        exact_one_sided_p_value: Fraction,
        adjusted_alpha: float,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if estimate.effect is None:
            reasons.append("pooled_effect_not_identified")
        elif estimate.effect <= self.minimum_effect:
            reasons.append("pooled_effect_does_not_clear_practical_margin")
        if estimate.treated_trials < self.minimum_treated_trials:
            reasons.append("insufficient_treated_trials")
        if estimate.control_trials < self.minimum_control_trials:
            reasons.append("insufficient_control_trials")
        if (
            min(
                estimate.treated_effective_sample_size,
                estimate.control_effective_sample_size,
            )
            < self.minimum_effective_support
        ):
            reasons.append("insufficient_effective_support")
        if missing_generation_trial_count:
            reasons.append("missing_authenticated_generation_block")
        if block_count < self.minimum_independent_blocks:
            reasons.append("insufficient_independent_blocks")
        if float(exact_one_sided_p_value) > adjusted_alpha:
            reasons.append("exact_block_sign_evidence_exceeds_adjusted_alpha")
        return tuple(reasons)

    def admits(self, evidence: CampaignRandomizedPromotionEvidence) -> bool:
        self.__post_init__()
        if type(evidence) is not CampaignRandomizedPromotionEvidence:
            raise TypeError(
                "evidence must be exact CampaignRandomizedPromotionEvidence"
            )
        CampaignRandomizedPromotionEvidence.__post_init__(evidence)
        if evidence.promotion_policy_definition_sha256 != self.definition_sha256:
            raise ValueError("promotion evidence was built under another policy")
        return evidence.passes_policy


@dataclass(frozen=True, slots=True)
class CampaignInsightRegistrationReceipt:
    origin_generation: int
    entries: tuple[tuple[InsightRef, str], ...]
    semantic_audit_plans: tuple[tuple[InsightRef, str, str], ...]
    memory_trial_count_at_registration: int
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.origin_generation) is not int or self.origin_generation <= 0:
            raise ValueError("origin_generation must be positive")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("registration entries must be a non-empty exact tuple")
        references: list[InsightRef] = []
        for item in self.entries:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("registration entries must be reference/hash pairs")
            reference, content_sha256 = item
            if type(reference) is not InsightRef:
                raise TypeError("registration reference must be exact")
            require_sha256(content_sha256, "draft_content_sha256")
            references.append(reference)
        if tuple(references) != tuple(sorted(set(references))):
            raise ValueError("registration references must be unique and canonical")
        if type(self.semantic_audit_plans) is not tuple:
            raise TypeError("semantic_audit_plans must be an exact tuple")
        planned_references: list[InsightRef] = []
        for item in self.semantic_audit_plans:
            if type(item) is not tuple or len(item) != 3:
                raise TypeError(
                    "semantic audit plans must be reference/hypothesis/plan triples"
                )
            reference, hypothesis_sha256, plan_sha256 = item
            if type(reference) is not InsightRef:
                raise TypeError("semantic audit plan reference must be exact")
            require_sha256(hypothesis_sha256, "draft_hypothesis_sha256")
            require_sha256(plan_sha256, "semantic_audit_plan_sha256")
            planned_references.append(reference)
        if tuple(planned_references) != tuple(references):
            raise ValueError(
                "semantic audit plans must exactly cover registration references"
            )
        if (
            type(self.memory_trial_count_at_registration) is not int
            or self.memory_trial_count_at_registration < 0
        ):
            raise ValueError("memory trial count must be non-negative")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_REGISTRATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "origin_generation": self.origin_generation,
            "entries": [
                {
                    "reference": _reference_record(reference),
                    "draft_content_sha256": content_sha256,
                }
                for reference, content_sha256 in self.entries
            ],
            "semantic_audit_plans": [
                {
                    "reference": _reference_record(reference),
                    "draft_hypothesis_sha256": hypothesis_sha256,
                    "semantic_audit_plan_sha256": plan_sha256,
                }
                for reference, hypothesis_sha256, plan_sha256 in (
                    self.semantic_audit_plans
                )
            ],
            "memory_trial_count_at_registration": (
                self.memory_trial_count_at_registration
            ),
            "lifecycle_state": "quarantined",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticAdmissionReceipt:
    admission_generation: int
    references: tuple[InsightRef, ...]
    registration_receipt_sha256s: tuple[str, ...]
    campaign_admission_request_sha256: str
    memory_admission_receipt_sha256: str
    operator_kind: str
    editable_paths: tuple[str, ...]
    memory_trial_count_cutoff: int
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.admission_generation) is not int or self.admission_generation <= 0:
            raise ValueError("admission_generation must be positive")
        if (
            type(self.references) is not tuple
            or not self.references
            or any(type(value) is not InsightRef for value in self.references)
        ):
            raise ValueError("references must be a non-empty exact tuple")
        if self.references != tuple(sorted(set(self.references))):
            raise ValueError("references must be unique and canonical")
        if (
            self.registration_receipt_sha256s
            != tuple(sorted(set(self.registration_receipt_sha256s)))
            or not self.registration_receipt_sha256s
        ):
            raise ValueError("registration receipt hashes must be canonical")
        for value in self.registration_receipt_sha256s:
            require_sha256(value, "registration_receipt_sha256")
        require_sha256(
            self.campaign_admission_request_sha256,
            "campaign_admission_request_sha256",
        )
        require_sha256(
            self.memory_admission_receipt_sha256,
            "memory_admission_receipt_sha256",
        )
        if type(self.operator_kind) is not str or not self.operator_kind:
            raise ValueError("operator_kind must be non-empty")
        if type(self.editable_paths) is not tuple:
            raise TypeError("editable_paths must be an exact tuple")
        if self.editable_paths != tuple(sorted(set(self.editable_paths))):
            raise ValueError("editable_paths must be unique and canonical")
        if (
            type(self.memory_trial_count_cutoff) is not int
            or self.memory_trial_count_cutoff < 0
        ):
            raise ValueError("memory trial cutoff must be non-negative")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_ADMISSION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "admission_generation": self.admission_generation,
            "references": [_reference_record(value) for value in self.references],
            "registration_receipt_sha256s": list(self.registration_receipt_sha256s),
            "campaign_admission_request_sha256": (
                self.campaign_admission_request_sha256
            ),
            "memory_admission_receipt_sha256": (self.memory_admission_receipt_sha256),
            "operator_kind": self.operator_kind,
            "editable_paths": list(self.editable_paths),
            "memory_trial_count_cutoff": self.memory_trial_count_cutoff,
            "visibility": "diagnostic_only_not_normal_retrieval",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignInsightAuditBinding:
    request: GlobalHypothesisAuditRequest
    receipt: GlobalHypothesisAuditReceipt
    exact_context_sha256: str

    def __post_init__(self) -> None:
        if type(self.request) is not GlobalHypothesisAuditRequest:
            raise TypeError("request must be an exact GlobalHypothesisAuditRequest")
        if type(self.receipt) is not GlobalHypothesisAuditReceipt:
            raise TypeError("receipt must be an exact GlobalHypothesisAuditReceipt")
        GlobalHypothesisAuditRequest.__post_init__(self.request)
        GlobalHypothesisAuditReceipt.__post_init__(self.receipt)
        if self.receipt.request_sha256 != self.request.request_sha256:
            raise ValueError("global audit receipt belongs to a different request")
        if (
            self.receipt.registry_snapshot_sha256
            != self.request.registry_snapshot_sha256
        ):
            raise ValueError("global audit receipt names a different registry")
        if (
            self.request.audit_policy_definition_sha256
            != GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("global audit request substituted the sealed policy")
        expected_policy = (
            GLOBAL_FALSIFICATION_POLICY_ID,
            GLOBAL_FALSIFICATION_POLICY_VERSION,
            GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256,
        )
        observed_policy = (
            self.receipt.audit_policy_id,
            self.receipt.audit_policy_version,
            self.receipt.audit_policy_definition_sha256,
        )
        if observed_policy != expected_policy:
            raise ValueError("global audit receipt substituted the sealed policy")
        require_sha256(self.exact_context_sha256, "exact_context_sha256")


@dataclass(frozen=True, slots=True)
class CampaignCausalUsefulnessReceipt:
    reference: InsightRef
    exact_context_sha256: str
    admission_generation: int
    barrier_generation: int
    admission_trial_count_cutoff: int
    barrier_trial_count_cutoff: int
    source_trial_set_sha256: str
    estimate: MarginalEffectEstimate
    randomized_promotion_evidence: CampaignRandomizedPromotionEvidence
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be exact")
        require_sha256(self.exact_context_sha256, "exact_context_sha256")
        if (
            type(self.admission_generation) is not int
            or type(self.barrier_generation) is not int
            or self.admission_generation <= 0
            or self.barrier_generation <= self.admission_generation
        ):
            raise ValueError("causal evidence requires a later barrier generation")
        if (
            type(self.admission_trial_count_cutoff) is not int
            or type(self.barrier_trial_count_cutoff) is not int
            or self.admission_trial_count_cutoff < 0
            or self.barrier_trial_count_cutoff < self.admission_trial_count_cutoff
        ):
            raise ValueError("causal trial cutoffs are invalid")
        require_sha256(self.source_trial_set_sha256, "source_trial_set_sha256")
        if type(self.estimate) is not MarginalEffectEstimate:
            raise TypeError("estimate must be exact")
        if (
            self.estimate.insight != self.reference
            or self.estimate.context_hash != self.exact_context_sha256
        ):
            raise ValueError("causal estimate differs from its insight/context")
        if (
            type(self.randomized_promotion_evidence)
            is not CampaignRandomizedPromotionEvidence
        ):
            raise TypeError("randomized_promotion_evidence must be exact")
        CampaignRandomizedPromotionEvidence.__post_init__(
            self.randomized_promotion_evidence
        )
        evidence = self.randomized_promotion_evidence
        if (
            evidence.reference != self.reference
            or evidence.exact_context_sha256 != self.exact_context_sha256
            or evidence.source_trial_set_sha256 != self.source_trial_set_sha256
        ):
            raise ValueError("randomized promotion evidence differs from causal source")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_CAUSAL_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        estimate = self.estimate
        return {
            "schema_version": 2,
            "reference": _reference_record(self.reference),
            "exact_context_sha256": self.exact_context_sha256,
            "admission_generation": self.admission_generation,
            "barrier_generation": self.barrier_generation,
            "admission_trial_count_cutoff": self.admission_trial_count_cutoff,
            "barrier_trial_count_cutoff": self.barrier_trial_count_cutoff,
            "source_trial_set_sha256": self.source_trial_set_sha256,
            "effect_hex": None if estimate.effect is None else estimate.effect.hex(),
            "treated_trials": estimate.treated_trials,
            "control_trials": estimate.control_trials,
            "treated_ess_hex": estimate.treated_effective_sample_size.hex(),
            "control_ess_hex": estimate.control_effective_sample_size.hex(),
            "eligible_trials": estimate.eligible_trials,
            "overlap_trials": estimate.overlap_trials,
            "estimand": "randomized_selected_minus_unselected_portfolio_reward",
            "randomized_promotion_evidence": (
                self.randomized_promotion_evidence.to_record()
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


class CampaignInsightLifecycleDecision(str, Enum):
    PROMOTE = "promote"
    DEPRECATE = "deprecate"
    RETAIN_QUARANTINE = "retain_quarantine"


@dataclass(frozen=True, slots=True)
class CampaignInsightDecisionReceipt:
    reference: InsightRef
    semantic_audit_request_sha256: str
    semantic_audit_receipt_sha256: str
    causal_usefulness: CampaignCausalUsefulnessReceipt
    decision: CampaignInsightLifecycleDecision
    reason: str
    lifecycle_transition_sequence: int | None
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be exact")
        require_sha256(
            self.semantic_audit_request_sha256,
            "semantic_audit_request_sha256",
        )
        require_sha256(
            self.semantic_audit_receipt_sha256,
            "semantic_audit_receipt_sha256",
        )
        if type(self.causal_usefulness) is not CampaignCausalUsefulnessReceipt:
            raise TypeError("causal_usefulness must be exact")
        if self.causal_usefulness.reference != self.reference:
            raise ValueError("causal usefulness belongs to another insight")
        if type(self.decision) is not CampaignInsightLifecycleDecision:
            raise TypeError("decision must be exact")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        transitioned = self.decision is not (
            CampaignInsightLifecycleDecision.RETAIN_QUARANTINE
        )
        if transitioned != (self.lifecycle_transition_sequence is not None):
            raise ValueError("lifecycle transition sequence differs from decision")
        if self.lifecycle_transition_sequence is not None and (
            type(self.lifecycle_transition_sequence) is not int
            or self.lifecycle_transition_sequence <= 0
        ):
            raise ValueError("lifecycle transition sequence must be positive")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "semantic_audit_request_sha256": self.semantic_audit_request_sha256,
            "semantic_audit_receipt_sha256": self.semantic_audit_receipt_sha256,
            "causal_usefulness_receipt_sha256": (self.causal_usefulness.receipt_sha256),
            "decision": self.decision.value,
            "reason": self.reason,
            "lifecycle_transition_sequence": self.lifecycle_transition_sequence,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignLearningBarrierReceipt:
    generation: int
    memory_credit_batch_receipt_sha256: str
    promotion_policy_definition_sha256: str
    decisions: tuple[CampaignInsightDecisionReceipt, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        require_sha256(
            self.memory_credit_batch_receipt_sha256,
            "memory_credit_batch_receipt_sha256",
        )
        require_sha256(
            self.promotion_policy_definition_sha256,
            "promotion_policy_definition_sha256",
        )
        if (
            type(self.decisions) is not tuple
            or not self.decisions
            or any(
                type(value) is not CampaignInsightDecisionReceipt
                for value in self.decisions
            )
        ):
            raise ValueError("decisions must be a non-empty exact tuple")
        references = tuple(value.reference for value in self.decisions)
        if references != tuple(sorted(set(references))):
            raise ValueError("decisions must use unique canonical insight order")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_BARRIER_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "memory_credit_batch_receipt_sha256": (
                self.memory_credit_batch_receipt_sha256
            ),
            "promotion_policy_definition_sha256": (
                self.promotion_policy_definition_sha256
            ),
            "decisions": [value.to_record() for value in self.decisions],
            "publication_scope": "post_generation_barrier_atomic_lifecycle_batch",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignPreparedLearningBarrier:
    """Pure, sealed lifecycle decision awaiting memory-credit publication.

    ``batch_trials`` are the exact trials expected to become the memory suffix.
    Preparing this value does not mutate either the bank or coordinator. Commit
    succeeds only if that exact suffix was published and lifecycle state stayed
    unchanged, leaving a deterministic atomic lifecycle batch.
    """

    memory_credit_batch: PortfolioMemoryCreditBatchReceipt
    audits: tuple[CampaignInsightAuditBinding, ...]
    batch_trials: tuple[InsightTrial, ...]
    lifecycle_requests: tuple[InsightLifecycleChangeRequest, ...]
    barrier_receipt: CampaignLearningBarrierReceipt
    memory_trial_count_at_prepare: int
    transition_count_at_prepare: int
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.memory_credit_batch) is not PortfolioMemoryCreditBatchReceipt:
            raise TypeError("memory_credit_batch must be exact")
        PortfolioMemoryCreditBatchReceipt.__post_init__(self.memory_credit_batch)
        if (
            type(self.audits) is not tuple
            or not self.audits
            or any(
                type(value) is not CampaignInsightAuditBinding for value in self.audits
            )
        ):
            raise ValueError("audits must contain exact audit bindings")
        for value in self.audits:
            CampaignInsightAuditBinding.__post_init__(value)
        audit_references = tuple(value.request.reference for value in self.audits)
        if audit_references != tuple(sorted(set(audit_references))):
            raise ValueError("audits must use unique canonical insight order")
        if (
            type(self.batch_trials) is not tuple
            or not self.batch_trials
            or any(type(value) is not InsightTrial for value in self.batch_trials)
        ):
            raise ValueError("batch_trials must contain exact trials")
        for value in self.batch_trials:
            InsightTrial.__post_init__(value)
        credit_ids = tuple(value.credit_unit_id.value for value in self.batch_trials)
        if credit_ids != tuple(sorted(set(credit_ids))):
            raise ValueError("batch_trials must use canonical unique credit order")
        if len(self.batch_trials) != len(self.memory_credit_batch.credits):
            raise ValueError("batch_trials differ from the memory credit batch")
        if type(self.lifecycle_requests) is not tuple or any(
            type(value) is not InsightLifecycleChangeRequest
            for value in self.lifecycle_requests
        ):
            raise TypeError("lifecycle_requests must contain exact requests")
        for value in self.lifecycle_requests:
            InsightLifecycleChangeRequest.__post_init__(value)
        lifecycle_references = tuple(
            value.reference for value in self.lifecycle_requests
        )
        if lifecycle_references != tuple(sorted(set(lifecycle_references))):
            raise ValueError("lifecycle requests must use canonical insight order")
        if type(self.barrier_receipt) is not CampaignLearningBarrierReceipt:
            raise TypeError("barrier_receipt must be exact")
        CampaignLearningBarrierReceipt.__post_init__(self.barrier_receipt)
        if (
            self.barrier_receipt.generation != self.memory_credit_batch.generation
            or self.barrier_receipt.memory_credit_batch_receipt_sha256
            != self.memory_credit_batch.receipt_sha256
        ):
            raise ValueError("prepared barrier differs from its memory credit batch")
        transitioned = tuple(
            value.reference
            for value in self.barrier_receipt.decisions
            if value.lifecycle_transition_sequence is not None
        )
        if transitioned != lifecycle_references:
            raise ValueError("prepared lifecycle requests differ from decisions")
        for name in (
            "memory_trial_count_at_prepare",
            "transition_count_at_prepare",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.memory_trial_count_at_prepare not in {
            self.memory_credit_batch.memory_trial_count_before,
            self.memory_credit_batch.memory_trial_count_after,
        }:
            raise ValueError("prepare trial count is outside the batch boundary")
        object.__setattr__(
            self,
            "preparation_sha256",
            _hash(_PREPARED_BARRIER_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "memory_credit_batch_receipt_sha256": (
                self.memory_credit_batch.receipt_sha256
            ),
            "semantic_audit_request_sha256s": [
                value.request.request_sha256 for value in self.audits
            ],
            "semantic_audit_receipt_sha256s": [
                value.receipt.audit_receipt_sha256 for value in self.audits
            ],
            "batch_trials": [_trial_record(value) for value in self.batch_trials],
            "lifecycle_requests": [
                {
                    "reference": _reference_record(value.reference),
                    "new_state": value.new_state.value,
                    "reason": value.reason,
                    "supporting_evidence": list(value.supporting_evidence),
                }
                for value in self.lifecycle_requests
            ],
            "barrier_receipt_sha256": self.barrier_receipt.receipt_sha256,
            "memory_trial_count_at_prepare": self.memory_trial_count_at_prepare,
            "transition_count_at_prepare": self.transition_count_at_prepare,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "preparation_sha256": self.preparation_sha256,
        }


@dataclass(slots=True)
class ClosedLoopCampaignLearning:
    """Stateful coordinator around one campaign-owned insight memory bank."""

    memory: InsightMemoryBank
    promotion_policy: CampaignInsightPromotionPolicy = field(
        default_factory=CampaignInsightPromotionPolicy
    )
    _registrations: dict[InsightRef, CampaignInsightRegistrationReceipt] = field(
        init=False,
        default_factory=dict,
    )
    _admissions: dict[InsightRef, CampaignDiagnosticAdmissionReceipt] = field(
        init=False,
        default_factory=dict,
    )
    _audit_plans: dict[InsightRef, CampaignSemanticAuditPlan] = field(
        init=False,
        default_factory=dict,
    )
    _closed_generations: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if type(self.promotion_policy) is not CampaignInsightPromotionPolicy:
            raise TypeError("promotion_policy must be exact")
        self.promotion_policy.__post_init__()

    def register_quarantined_reflections(
        self,
        *,
        origin_generation: int,
        references: tuple[InsightRef, ...],
        semantic_audit_plans: tuple[CampaignSemanticAuditPlan, ...],
    ) -> CampaignInsightRegistrationReceipt:
        self.__post_init__()
        if type(origin_generation) is not int or origin_generation <= 0:
            raise ValueError("origin_generation must be positive")
        if type(references) is not tuple or not references:
            raise ValueError("references must be a non-empty exact tuple")
        canonical = tuple(sorted(references))
        if len(set(canonical)) != len(canonical):
            raise ValueError("references cannot contain duplicates")
        if any(value in self._registrations for value in canonical):
            raise ValueError("an insight was already registered")
        if type(semantic_audit_plans) is not tuple or any(
            type(value) is not CampaignSemanticAuditPlan
            for value in semantic_audit_plans
        ):
            raise TypeError(
                "semantic_audit_plans must contain exact CampaignSemanticAuditPlan values"
            )
        for value in semantic_audit_plans:
            CampaignSemanticAuditPlan.__post_init__(value)
        plans = tuple(sorted(semantic_audit_plans, key=lambda value: value.reference))
        if tuple(value.reference for value in plans) != canonical:
            raise ValueError(
                "semantic audit plans must exactly cover registration references"
            )
        entries = self.memory.entries_for(canonical)
        if any(
            entry.lifecycle_state is not InsightLifecycleState.QUARANTINED
            or entry.origin is not InsightOrigin.REFLECTION
            for entry in entries
        ):
            raise ValueError("registration requires reflection-origin quarantine cards")
        for entry, plan in zip(entries, plans, strict=True):
            self._validate_semantic_audit_plan(entry, plan)
        receipt = CampaignInsightRegistrationReceipt(
            origin_generation=origin_generation,
            entries=tuple(
                (entry.reference, entry.draft.content_sha256) for entry in entries
            ),
            semantic_audit_plans=tuple(
                (
                    entry.reference,
                    entry.draft.hypothesis_sha256,
                    plan.plan_sha256,
                )
                for entry, plan in zip(entries, plans, strict=True)
            ),
            memory_trial_count_at_registration=len(self.memory.trials),
        )
        for reference, plan in zip(canonical, plans, strict=True):
            self._registrations[reference] = receipt
            self._audit_plans[reference] = plan
        return receipt

    @staticmethod
    def _validate_semantic_audit_plan(
        entry: InsightMemoryEntry,
        plan: CampaignSemanticAuditPlan,
    ) -> None:
        """Join a pre-outcome typed audit plan to one exact actionable card."""

        draft = entry.draft
        if not draft.has_semantic_contract or not draft.has_intervention_contract:
            raise ValueError(
                "closed-loop registration requires an actionable semantic-v3 card"
            )
        if draft.insight_kind not in {
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            ReflectionInsightKind.MECHANISTIC_CONJECTURE,
        }:
            raise ValueError(
                "closed-loop registration rejects heuristic or invariant card kinds"
            )
        if not entry.applicable_operator_kinds:
            raise ValueError(
                "closed-loop registration requires explicit operator applicability"
            )
        if draft.affected_paths != tuple(sorted(set(draft.affected_paths))):
            raise ValueError("closed-loop affected paths must be canonical")
        if any(
            value.direction is MetricEffectDirection.UNKNOWN
            for value in draft.effect_predictions
        ):
            raise ValueError(
                "closed-loop promotion requires known auditable metric predictions"
            )
        expected_predictions = tuple(
            (value.metric_id, value.direction, None, None)
            for value in draft.effect_predictions
        )
        observed_predictions = tuple(
            (
                value.metric_id,
                value.direction,
                value.minimum_delta,
                value.maximum_delta,
            )
            for value in plan.predictions
        )
        expected_strength = HypothesisClaimStrength(
            sufficiency=True,
            necessity=False,
            invariance=False,
            mechanistic_or_causal=(
                draft.insight_kind is ReflectionInsightKind.MECHANISTIC_CONJECTURE
            ),
        )
        expected = (
            entry.reference,
            draft.content_sha256,
            draft.hypothesis_sha256,
            draft.affected_paths,
            entry.applicable_operator_kinds,
            expected_predictions,
            expected_strength,
        )
        observed = (
            plan.reference,
            plan.draft_content_sha256,
            plan.draft_hypothesis_sha256,
            plan.intervention.affected_paths,
            plan.intervention.admissible_operator_families,
            observed_predictions,
            plan.claim_strength,
        )
        if observed != expected:
            raise ValueError(
                "semantic audit plan differs from the registered card hypothesis"
            )

    def audit_plans_for(
        self,
        references: tuple[InsightRef, ...],
    ) -> tuple[CampaignSemanticAuditPlan, ...]:
        """Return revalidated pre-outcome plans in exact canonical order.

        This narrow projection lets an outcome-audit adapter compile a later
        sealed request without reaching into coordinator state.  It never
        sorts on the caller's behalf: order substitution is rejected so the
        returned tuple can be joined positionally to a canonical audit family.
        """

        self.__post_init__()
        if type(references) is not tuple or not references:
            raise ValueError("references must be a non-empty exact tuple")
        if any(type(value) is not InsightRef for value in references):
            raise TypeError("references must contain exact InsightRef values")
        for value in references:
            InsightRef.__post_init__(value)
        if references != tuple(sorted(set(references))):
            raise ValueError("references must be unique and canonically ordered")
        try:
            plans = tuple(self._audit_plans[value] for value in references)
        except KeyError as error:
            raise ValueError(
                "reference has no registered semantic audit plan"
            ) from error
        entries = self.memory.entries_for(references)
        for entry, plan in zip(entries, plans, strict=True):
            CampaignSemanticAuditPlan.__post_init__(plan)
            self._validate_semantic_audit_plan(entry, plan)
        return plans

    def admit_for_diagnostic_testing(
        self,
        *,
        admission_generation: int,
        references: tuple[InsightRef, ...],
        campaign_admission_request_sha256: str,
        operator_kind: str,
        editable_paths: tuple[str, ...] = (),
    ) -> CampaignDiagnosticAdmissionReceipt:
        self.__post_init__()
        if type(references) is not tuple or not references:
            raise ValueError("references must be a non-empty exact tuple")
        canonical = tuple(sorted(references))
        if len(set(canonical)) != len(canonical):
            raise ValueError("references cannot contain duplicates")
        if any(value not in self._registrations for value in canonical):
            raise ValueError("diagnostic admission names an unregistered insight")
        if any(value in self._admissions for value in canonical):
            raise ValueError("an insight was already admitted for diagnostics")
        if any(
            admission_generation < self._registrations[value].origin_generation
            for value in canonical
        ):
            raise ValueError("diagnostic admission precedes reflection origin")
        memory_admission: QuarantineTestAdmissionReceipt = (
            self.memory.admit_quarantine_test_assignment(
                canonical,
                operator_kind=operator_kind,
                source_admission_request_sha256=(campaign_admission_request_sha256),
                editable_paths=editable_paths or None,
            )
        )
        receipt = CampaignDiagnosticAdmissionReceipt(
            admission_generation=admission_generation,
            references=memory_admission.references,
            registration_receipt_sha256s=tuple(
                sorted(
                    {
                        self._registrations[value].receipt_sha256
                        for value in memory_admission.references
                    }
                )
            ),
            campaign_admission_request_sha256=(campaign_admission_request_sha256),
            memory_admission_receipt_sha256=(memory_admission.receipt_sha256),
            operator_kind=operator_kind,
            editable_paths=tuple(sorted(set(editable_paths))),
            memory_trial_count_cutoff=(memory_admission.memory_trial_count_cutoff),
        )
        for reference in memory_admission.references:
            self._admissions[reference] = receipt
        return receipt

    def prepare_generation_close(
        self,
        *,
        memory_credit_batch: PortfolioMemoryCreditBatchReceipt,
        audits: tuple[CampaignInsightAuditBinding, ...],
        prospective_trials: tuple[InsightTrial, ...] = (),
    ) -> CampaignPreparedLearningBarrier:
        """Prepare all semantic/causal decisions without mutating campaign state."""

        self.__post_init__()
        if type(memory_credit_batch) is not PortfolioMemoryCreditBatchReceipt:
            raise TypeError("memory_credit_batch must be exact")
        PortfolioMemoryCreditBatchReceipt.__post_init__(memory_credit_batch)
        generation = memory_credit_batch.generation
        if generation in self._closed_generations:
            raise ValueError("campaign learning generation was already closed")
        if type(prospective_trials) is not tuple or any(
            type(value) is not InsightTrial for value in prospective_trials
        ):
            raise TypeError("prospective_trials must contain exact InsightTrial values")
        for value in prospective_trials:
            InsightTrial.__post_init__(value)
        memory_trial_count = len(self.memory.trials)
        if memory_trial_count == memory_credit_batch.memory_trial_count_after:
            committed = self.memory.trials[
                memory_credit_batch.memory_trial_count_before : memory_credit_batch.memory_trial_count_after
            ]
            if prospective_trials and prospective_trials != committed:
                raise ValueError("prospective trials differ from committed memory")
            batch_trials = committed
            trial_universe = self.memory.trials
        elif memory_trial_count == memory_credit_batch.memory_trial_count_before:
            if not prospective_trials:
                raise ValueError(
                    "precommit campaign learning requires prospective memory trials"
                )
            batch_trials = prospective_trials
            trial_universe = (*self.memory.trials, *prospective_trials)
        else:
            raise ValueError("memory bank is outside the sealed batch boundary")
        self._validate_memory_credit_batch_trials(
            memory_credit_batch,
            batch_trials,
        )
        if len(trial_universe) != memory_credit_batch.memory_trial_count_after:
            raise ValueError("prospective trials do not close the memory batch")
        if type(audits) is not tuple or not audits:
            raise ValueError("audits must be a non-empty exact tuple")
        if any(type(value) is not CampaignInsightAuditBinding for value in audits):
            raise TypeError("audits must contain exact bindings")
        for value in audits:
            CampaignInsightAuditBinding.__post_init__(value)
        canonical = tuple(sorted(audits, key=lambda value: value.request.reference))
        references = tuple(value.request.reference for value in canonical)
        if len(set(references)) != len(references):
            raise ValueError("audits cannot repeat an insight reference")

        prepared: list[
            tuple[
                CampaignInsightAuditBinding,
                CampaignCausalUsefulnessReceipt,
                CampaignInsightLifecycleDecision,
                str,
            ]
        ] = []
        lifecycle_requests: list[InsightLifecycleChangeRequest] = []
        for binding in canonical:
            reference = binding.request.reference
            admission = self._admissions.get(reference)
            if admission is None:
                raise ValueError("audit names an insight not admitted for diagnostics")
            if generation <= admission.admission_generation:
                raise ValueError("same-generation diagnostic outcomes are ineligible")
            entry = self.memory.entries_for((reference,))[0]
            if entry.lifecycle_state is not InsightLifecycleState.QUARANTINED:
                raise ValueError("audit insight is no longer quarantined")
            if binding.request.draft_content_sha256 != entry.draft.content_sha256:
                raise ValueError("global audit is bound to different card content")
            plan = self._audit_plans.get(reference)
            if plan is None:
                raise ValueError("audit insight has no pre-outcome semantic audit plan")
            self._validate_semantic_audit_plan(entry, plan)
            if not plan.admits(binding.request):
                raise ValueError(
                    "global audit request substituted the registered semantic plan"
                )
            if (
                binding.request.origin_cutoff_event_index
                >= binding.request.audit_cutoff_event_index
            ):
                raise ValueError("semantic replay lacks a post-origin audit interval")
            self._validate_semantic_verdict(binding)

            source_trials = tuple(
                trial
                for trial in trial_universe[
                    admission.memory_trial_count_cutoff : memory_credit_batch.memory_trial_count_after
                ]
                if trial.decision.context_hash == binding.exact_context_sha256
                and reference in trial.decision.eligible
            )
            estimate = estimate_marginal_effect(
                source_trials,
                reference,
                context_hash=binding.exact_context_sha256,
            )
            source_trial_set_sha256 = _hash(
                b"agent-evolve:campaign-causal-trial-set:v1\x00",
                [_trial_record(value) for value in source_trials],
            )
            randomized_promotion_evidence = self.promotion_policy.evaluate(
                estimate=estimate,
                source_trials=source_trials,
                source_trial_set_sha256=source_trial_set_sha256,
                family_hypothesis_count=len(canonical),
            )
            causal = CampaignCausalUsefulnessReceipt(
                reference=reference,
                exact_context_sha256=binding.exact_context_sha256,
                admission_generation=admission.admission_generation,
                barrier_generation=generation,
                admission_trial_count_cutoff=(admission.memory_trial_count_cutoff),
                barrier_trial_count_cutoff=(
                    memory_credit_batch.memory_trial_count_after
                ),
                source_trial_set_sha256=source_trial_set_sha256,
                estimate=estimate,
                randomized_promotion_evidence=randomized_promotion_evidence,
            )
            verdict = binding.receipt.verdict
            if verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE:
                decision = CampaignInsightLifecycleDecision.DEPRECATE
                reason = "global semantic audit found a counterexample"
                lifecycle_requests.append(
                    InsightLifecycleChangeRequest(
                        reference=reference,
                        new_state=InsightLifecycleState.DEPRECATED,
                        reason=reason,
                        supporting_evidence=(binding.receipt.audit_receipt_sha256,),
                    )
                )
            elif (
                verdict is GlobalHypothesisVerdict.SUPPORT
                and self.promotion_policy.admits(randomized_promotion_evidence)
            ):
                decision = CampaignInsightLifecycleDecision.PROMOTE
                reason = (
                    "semantic replay, practical effect, and exact block robustness "
                    "all passed"
                )
                lifecycle_requests.append(
                    InsightLifecycleChangeRequest(
                        reference=reference,
                        new_state=InsightLifecycleState.PROMOTED,
                        reason=reason,
                        supporting_evidence=tuple(
                            sorted(
                                (
                                    binding.receipt.audit_receipt_sha256,
                                    causal.receipt_sha256,
                                )
                            )
                        ),
                    )
                )
            else:
                decision = CampaignInsightLifecycleDecision.RETAIN_QUARANTINE
                if verdict is GlobalHypothesisVerdict.SUPPORT:
                    detail = ",".join(randomized_promotion_evidence.failure_reasons)
                    reason = f"randomized promotion evidence insufficient: {detail}"
                else:
                    reason = "semantic promotion evidence remains insufficient"
            prepared.append((binding, causal, decision, reason))
        transition_count_before = len(self.memory.transitions)
        canonical_lifecycle_requests = tuple(
            sorted(lifecycle_requests, key=lambda value: value.reference)
        )
        transition_by_reference = {
            value.reference: transition_count_before + index
            for index, value in enumerate(
                canonical_lifecycle_requests,
                start=1,
            )
        }
        decisions = tuple(
            CampaignInsightDecisionReceipt(
                reference=binding.request.reference,
                semantic_audit_request_sha256=binding.request.request_sha256,
                semantic_audit_receipt_sha256=(binding.receipt.audit_receipt_sha256),
                causal_usefulness=causal,
                decision=decision,
                reason=reason,
                lifecycle_transition_sequence=transition_by_reference.get(
                    binding.request.reference
                ),
            )
            for binding, causal, decision, reason in prepared
        )
        receipt = CampaignLearningBarrierReceipt(
            generation=generation,
            memory_credit_batch_receipt_sha256=(memory_credit_batch.receipt_sha256),
            promotion_policy_definition_sha256=(
                self.promotion_policy.definition_sha256
            ),
            decisions=decisions,
        )
        return CampaignPreparedLearningBarrier(
            memory_credit_batch=memory_credit_batch,
            audits=canonical,
            batch_trials=batch_trials,
            lifecycle_requests=canonical_lifecycle_requests,
            barrier_receipt=receipt,
            memory_trial_count_at_prepare=memory_trial_count,
            transition_count_at_prepare=transition_count_before,
        )

    def commit_generation_close(
        self,
        preparation: CampaignPreparedLearningBarrier,
    ) -> CampaignLearningBarrierReceipt:
        """Commit one pure preparation after its exact memory batch is published."""

        self.__post_init__()
        if type(preparation) is not CampaignPreparedLearningBarrier:
            raise TypeError("preparation must be exact")
        CampaignPreparedLearningBarrier.__post_init__(preparation)
        generation = preparation.memory_credit_batch.generation
        if generation in self._closed_generations:
            raise ValueError("campaign learning generation was already closed")
        if (
            len(self.memory.trials)
            != preparation.memory_credit_batch.memory_trial_count_after
        ):
            raise ValueError("memory batch was not published before lifecycle commit")
        self._validate_memory_credit_batch(preparation.memory_credit_batch)
        committed_trials = self.memory.trials[
            preparation.memory_credit_batch.memory_trial_count_before : preparation.memory_credit_batch.memory_trial_count_after
        ]
        if committed_trials != preparation.batch_trials:
            raise ValueError("published memory trials differ from learning preparation")
        if len(self.memory.transitions) != preparation.transition_count_at_prepare:
            raise ValueError("lifecycle state changed after learning preparation")
        if any(
            self.memory.entries_for((value.reference,))[0].lifecycle_state
            is not InsightLifecycleState.QUARANTINED
            for value in preparation.lifecycle_requests
        ):
            raise ValueError("prepared lifecycle card changed before commit")
        if preparation.lifecycle_requests:
            self.memory.apply_lifecycle_batch(preparation.lifecycle_requests)
        self._closed_generations.add(generation)
        return preparation.barrier_receipt

    def close_generation(
        self,
        *,
        memory_credit_batch: PortfolioMemoryCreditBatchReceipt,
        audits: tuple[CampaignInsightAuditBinding, ...],
    ) -> CampaignLearningBarrierReceipt:
        """Compatibility wrapper for an already-committed memory batch."""

        preparation = self.prepare_generation_close(
            memory_credit_batch=memory_credit_batch,
            audits=audits,
        )
        return self.commit_generation_close(preparation)

    @staticmethod
    def _validate_semantic_verdict(binding: CampaignInsightAuditBinding) -> None:
        request = binding.request
        receipt = binding.receipt
        if receipt.verdict is GlobalHypothesisVerdict.SUPPORT:
            post_origin_support = tuple(
                value
                for value in receipt.decisions
                if value.disposition is EvidenceDisposition.SUPPORT
                and value.post_origin_revision_evidence
                and value.event_index > request.origin_cutoff_event_index
            )
            post_origin_support_ids = tuple(
                value.source_evidence_id for value in post_origin_support
            )
            post_origin_clusters = {
                value.effective_cluster_sha256 for value in post_origin_support
            }
            post_origin_instances = {
                value.workload_instance_sha256 for value in post_origin_support
            }
            if (
                not receipt.support_ids
                or receipt.effective_support_cluster_count
                < request.minimum_support_clusters
                or receipt.support_instance_count < request.minimum_support_instances
                or receipt.counterexample_ids
                or receipt.necessity_contradicted
            ):
                raise ValueError(
                    "support verdict lacks its declared evidence threshold"
                )
            if (
                not post_origin_support_ids
                or len(post_origin_clusters) < request.minimum_support_clusters
                or len(post_origin_instances) < request.minimum_support_instances
            ):
                raise ValueError(
                    "support verdict lacks post-origin evidence thresholds"
                )
        elif receipt.verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE:
            if not receipt.counterexample_ids and not receipt.necessity_contradicted:
                raise ValueError("counterexample verdict contains no counterexample")

    def _validate_memory_credit_batch(
        self,
        batch: PortfolioMemoryCreditBatchReceipt,
    ) -> None:
        trials = self.memory.trials[
            batch.memory_trial_count_before : batch.memory_trial_count_after
        ]
        self._validate_memory_credit_batch_trials(batch, trials)

    @staticmethod
    def _validate_memory_credit_batch_trials(
        batch: PortfolioMemoryCreditBatchReceipt,
        trials: tuple[InsightTrial, ...],
    ) -> None:
        if type(batch) is not PortfolioMemoryCreditBatchReceipt:
            raise TypeError("batch must be exact")
        PortfolioMemoryCreditBatchReceipt.__post_init__(batch)
        if type(trials) is not tuple or any(
            type(value) is not InsightTrial for value in trials
        ):
            raise TypeError("trials must contain exact InsightTrial values")
        for value in trials:
            InsightTrial.__post_init__(value)
        if len(trials) != len(batch.credits):
            raise ValueError("memory batch differs from its exact trial suffix")
        trial_by_credit = {value.credit_unit_id: value for value in trials}
        if len(trial_by_credit) != len(trials):
            raise ValueError("trial suffix repeats a credit unit")
        for credit in batch.credits:
            trial = trial_by_credit.get(credit.credit_unit_id)
            if trial is None:
                raise ValueError("memory batch credit is absent from the trial suffix")
            if (
                trial.generation != batch.generation
                or trial.candidate_ids != credit.candidate_ids
                or trial.reward_definition_hash != credit.aggregation_definition_sha256
                or insight_selection_decision_sha256(trial.decision)
                != credit.selection_decision_sha256
                or trial.decision.context_hash
                != credit.selection_decision_context_sha256
                or trial.reward != credit.reward
                or trial.treatment_binding_sha256 != credit.treatment_binding_sha256
                or trial.treatment_binding_sha256 is None
            ):
                raise ValueError(
                    "memory batch credit differs from its exact committed trial"
                )


__all__ = [
    "CampaignCausalUsefulnessReceipt",
    "CampaignDiagnosticAdmissionReceipt",
    "CampaignInsightAuditBinding",
    "CampaignInsightDecisionReceipt",
    "CampaignInsightLifecycleDecision",
    "CampaignInsightPromotionPolicy",
    "CampaignInsightRegistrationReceipt",
    "CampaignLearningBarrierReceipt",
    "CampaignPreparedLearningBarrier",
    "CampaignRandomizedPromotionEvidence",
    "CampaignSemanticAuditPlan",
    "ClosedLoopCampaignLearning",
]
