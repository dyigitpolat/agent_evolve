from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from agent_evolve.application.campaign_learning import (
    CampaignInsightAuditBinding,
    CampaignInsightLifecycleDecision,
    CampaignInsightPromotionPolicy,
    CampaignSemanticAuditPlan,
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditBatchReceipt,
    PortfolioMemoryCreditReceipt,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.global_falsification import (
    GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256,
    GLOBAL_FALSIFICATION_POLICY_ID,
    GLOBAL_FALSIFICATION_POLICY_VERSION,
    EvidenceDisposition,
    GlobalEvidenceDecision,
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
    InsightSelectionDecision,
    InsightSelectionMode,
    InsightTrial,
)
from agent_evolve.policies.memory.staged_causal import (
    insight_selection_decision_sha256,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionInsightKind,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _NoRandom:
    def randrange(self, stop: int) -> int:  # pragma: no cover - exploit only.
        raise AssertionError(f"unexpected draw with stop={stop}")

    def sample(self, population, k: int):  # pragma: no cover - exploit only.
        raise AssertionError((population, k))


def _reflection_entry(memory, ids, label: str):
    contrast = _sha(f"contrast:{label}")
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId(f"call_reflection_{label}"),
        source_operator_invocation_ids=(),
        source_candidate_ids=(),
        available_contrast_ids=(contrast,),
        cited_contrast_ids=(contrast,),
    )
    entry, added = memory.add(
        InsightDraft(
            claim=f"Reflected hypothesis {label}",
            trigger="the typed action is available",
            mechanism="the action may improve the evaluated frontier",
            affected_paths=("$.sequence[0]",),
            evidence_summary=f"sealed contrast {label}",
            confidence=0.5,
            evidence_contrast_ids=(contrast,),
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="quality",
                    direction=MetricEffectDirection.INCREASE,
                    comparison_anchor=MetricComparisonAnchor(
                        MetricComparisonAnchorKind.CURRENT_PARENT,
                    ),
                ),
            ),
            recommended_option_families=("sequence_rewrite",),
            action_template="Apply the typed sequence rewrite.",
            falsification_condition="The child quality does not increase.",
            insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
            factor_capabilities=("sequence_rewrite",),
        ),
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=lineage,
        applicable_operator_kinds=("typed_mutation",),
    )
    assert added
    return entry


def _decision(references, selected, context):
    canonical = tuple(sorted(references))
    exploitation = (canonical[0],)
    return InsightSelectionDecision(
        context_hash=context,
        eligible=canonical,
        selected=(selected,),
        exploitation_subset=exploitation,
        score_snapshot=tuple((value, 0.0) for value in canonical),
        subset_size=1,
        exploration_probability=Fraction(1, 1),
        mode=InsightSelectionMode.EXPLORE_UNIFORM,
        selected_subset_probability=Fraction(1, len(canonical)),
    )


def _publish_trials(memory, ids, references, context, *, generation: int = 3):
    memory_trial_count_before = len(memory.trials)
    first = tuple(sorted(references))[0]
    trials = tuple(
        InsightTrial(
            credit_unit_id=ids.new_operator_invocation_id(),
            candidate_ids=(ids.new_candidate_id(),),
            reward_definition_hash=_sha("closed-loop-reward"),
            decision=_decision(references, selected, context),
            reward=1.0 if selected == first else 0.0,
            treatment_binding_sha256=_sha(
                f"treatment:{selected.insight_id.value}:{selected.version}"
            ),
            generation=generation,
        )
        for selected in (*tuple(sorted(references)), *tuple(sorted(references)))
    )
    committed = memory.record_trials_batch(trials)
    projection = PortfolioMemoryContextProjectionBinding.exact_identity(context)
    credits = tuple(
        PortfolioMemoryCreditReceipt(
            credit_unit_id=trial.credit_unit_id,
            selection_decision_sha256=insight_selection_decision_sha256(trial.decision),
            selection_decision_context_sha256=context,
            candidate_ids=trial.candidate_ids,
            aggregation_id="closed_loop_reward",
            aggregation_version=1,
            aggregation_definition_sha256=trial.reward_definition_hash,
            aggregation_binding_sha256=_sha("closed-loop-reward-binding"),
            context_projection=projection,
            reward=trial.reward,
            treatment_binding_sha256=trial.treatment_binding_sha256,
            generation=trial.generation,
        )
        for trial in committed
    )
    return PortfolioMemoryCreditBatchReceipt(
        generation=generation,
        credits=credits,
        memory_trial_count_before=memory_trial_count_before,
        memory_trial_count_after=memory_trial_count_before + len(credits),
    )


def _predicate(name: str, payload: dict[str, object]) -> TypedEvidencePredicate:
    return TypedEvidencePredicate(
        schema_id=name,
        schema_version=1,
        schema_definition_sha256=_sha(f"schema:{name}"),
        payload=freeze_json(payload),
    )


def _audit_binding(entry, verdict, context, *, content_sha256=None):
    trigger = _predicate("campaign_trigger", {"path": "$.sequence[0]"})
    old = _predicate("campaign_old", {"equals": "rewrite"})
    action = _predicate("campaign_action", {"replace_with": "balance"})
    registry = _sha(f"registry:{entry.reference.insight_id.value}:{verdict.value}")
    request = GlobalHypothesisAuditRequest(
        reference=entry.reference,
        draft_content_sha256=(
            entry.draft.content_sha256 if content_sha256 is None else content_sha256
        ),
        trigger=trigger,
        intervention=TypedInterventionSignature(
            affected_paths=("$.sequence[0]",),
            old_value_predicate=old,
            new_action=action,
            admissible_operator_families=("typed_mutation",),
        ),
        predictions=(
            HypothesisMetricPrediction(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
            ),
        ),
        claim_strength=HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(_sha("workload"),),
            evaluator_contract_sha256=_sha("evaluator"),
            metric_adjudicator_definition_sha256=_sha("adjudicator"),
            campaign_sha256s=(_sha("campaign"),),
        ),
        matcher_definition_sha256=_sha("matcher"),
        origin_cutoff_event_index=2,
        audit_cutoff_event_index=3,
        registry_snapshot_sha256=registry,
        minimum_support_clusters=1,
        minimum_support_instances=1,
    )
    source = _sha(f"evidence:{entry.reference.insight_id.value}:{verdict.value}")
    decisions = ()
    support_ids = ()
    counterexample_ids = ()
    raw_support = 0
    effective_support = 0
    support_instances = 0
    if verdict is GlobalHypothesisVerdict.SUPPORT:
        decisions = (
            GlobalEvidenceDecision(
                source_evidence_id=source,
                observation_sha256=_sha(f"observation:{source}"),
                event_index=3,
                workload_instance_sha256=_sha("workload"),
                itt_estimand_unit_sha256=_sha(f"unit:{source}"),
                disposition=EvidenceDisposition.SUPPORT,
                match_receipt_sha256=_sha(f"match:{source}"),
                metric_assessments=(),
                effective_cluster_sha256=_sha(f"cluster:{source}"),
                post_origin_revision_evidence=True,
                predictions_match=True,
            ),
        )
        support_ids = (source,)
        raw_support = effective_support = support_instances = 1
    elif verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE:
        decisions = (
            GlobalEvidenceDecision(
                source_evidence_id=source,
                observation_sha256=_sha(f"observation:{source}"),
                event_index=3,
                workload_instance_sha256=_sha("workload"),
                itt_estimand_unit_sha256=_sha(f"unit:{source}"),
                disposition=EvidenceDisposition.COUNTEREXAMPLE,
                match_receipt_sha256=_sha(f"match:{source}"),
                metric_assessments=(),
                effective_cluster_sha256=None,
                post_origin_revision_evidence=True,
                predictions_match=False,
            ),
        )
        counterexample_ids = (source,)
    receipt = GlobalHypothesisAuditReceipt(
        request_sha256=request.request_sha256,
        registry_snapshot_sha256=registry,
        audit_policy_id=GLOBAL_FALSIFICATION_POLICY_ID,
        audit_policy_version=GLOBAL_FALSIFICATION_POLICY_VERSION,
        audit_policy_definition_sha256=(GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256),
        verdict=verdict,
        decisions=decisions,
        support_ids=support_ids,
        counterexample_ids=counterexample_ids,
        off_trigger_control_ids=(),
        non_identifiable_ids=(),
        raw_support_count=raw_support,
        effective_support_cluster_count=effective_support,
        support_instance_count=support_instances,
        untested_workload_instance_sha256s=(),
        coverage_gaps=(),
        necessity_contradicted=False,
        mechanism_identified=False,
        lifecycle_decision="provider-free closed-loop test audit",
    )
    return CampaignInsightAuditBinding(
        request=request,
        receipt=receipt,
        exact_context_sha256=context,
    )


def _rebind_audit_request(binding, request):
    receipt = replace(
        binding.receipt,
        request_sha256=request.request_sha256,
        registry_snapshot_sha256=request.registry_snapshot_sha256,
    )
    return CampaignInsightAuditBinding(
        request=request,
        receipt=receipt,
        exact_context_sha256=binding.exact_context_sha256,
    )


def _setup(
    *,
    trial_generation: int = 3,
    trial_generations: tuple[int, ...] | None = None,
):
    ids = DeterministicIdFactory("closed_loop_campaign")
    memory = InsightMemoryBank(id_factory=ids)
    entries = tuple(
        _reflection_entry(memory, ids, label)
        for label in ("promote", "deprecate", "retain")
    )
    learning = ClosedLoopCampaignLearning(
        memory=memory,
        promotion_policy=CampaignInsightPromotionPolicy(
            minimum_treated_trials=2,
            minimum_control_trials=2,
            minimum_effective_support=2.0,
            minimum_effect=0.0,
        ),
    )
    references = tuple(value.reference for value in entries)
    context = _sha("closed-loop-context")
    semantic_audit_plans = tuple(
        CampaignSemanticAuditPlan.from_request(
            _audit_binding(
                entry,
                GlobalHypothesisVerdict.INSUFFICIENT,
                context,
            ).request,
            draft_hypothesis_sha256=entry.draft.hypothesis_sha256,
        )
        for entry in entries
    )
    registration = learning.register_quarantined_reflections(
        origin_generation=2,
        references=tuple(reversed(references)),
        semantic_audit_plans=tuple(reversed(semantic_audit_plans)),
    )
    admission = learning.admit_for_diagnostic_testing(
        admission_generation=2,
        references=tuple(reversed(references)),
        campaign_admission_request_sha256=_sha("campaign-admission-g2"),
        operator_kind="typed_mutation",
        editable_paths=("$.sequence",),
    )
    generations = (
        (trial_generation,) if trial_generations is None else trial_generations
    )
    if not generations:
        raise ValueError("trial_generations cannot be empty")
    barrier = None
    for generation in generations:
        barrier = _publish_trials(
            memory,
            ids,
            references,
            context,
            generation=generation,
        )
    assert barrier is not None
    return memory, learning, entries, registration, admission, context, barrier


def test_closed_loop_promotes_only_supported_useful_card_after_later_barrier() -> None:
    memory, learning, entries, registration, admission, context, barrier = _setup(
        trial_generations=(3, 5, 7, 9, 11, 13)
    )
    promote, deprecate, retain = entries
    assert tuple(value[0] for value in registration.entries) == tuple(
        sorted(value.reference for value in entries)
    )
    assert admission.references == tuple(sorted(value.reference for value in entries))
    assert all(
        value.lifecycle_state is InsightLifecycleState.QUARANTINED
        for value in memory.entries_for(tuple(value.reference for value in entries))
    )

    audits = (
        _audit_binding(retain, GlobalHypothesisVerdict.INSUFFICIENT, context),
        _audit_binding(promote, GlobalHypothesisVerdict.SUPPORT, context),
        _audit_binding(deprecate, GlobalHypothesisVerdict.COUNTEREXAMPLE, context),
    )
    result = learning.close_generation(memory_credit_batch=barrier, audits=audits)
    by_reference = {value.reference: value for value in result.decisions}
    assert tuple(value.reference for value in result.decisions) == tuple(
        sorted(value.reference for value in entries)
    )
    assert by_reference[promote.reference].decision is (
        CampaignInsightLifecycleDecision.PROMOTE
    )
    assert by_reference[deprecate.reference].decision is (
        CampaignInsightLifecycleDecision.DEPRECATE
    )
    assert by_reference[retain.reference].decision is (
        CampaignInsightLifecycleDecision.RETAIN_QUARANTINE
    )
    assert by_reference[promote.reference].causal_usefulness.estimate.effect == 1.0
    assert len(memory.transitions) == 2
    assert memory.entries_for((promote.reference,))[0].lifecycle_state is (
        InsightLifecycleState.PROMOTED
    )
    assert memory.entries_for((deprecate.reference,))[0].lifecycle_state is (
        InsightLifecycleState.DEPRECATED
    )
    assert memory.entries_for((retain.reference,))[0].lifecycle_state is (
        InsightLifecycleState.QUARANTINED
    )

    eligible = memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.sequence",),
    )
    assert eligible == (promote.reference,)
    selected = memory.select(
        context_hash=_sha("g5-normal-retrieval"),
        subset_size=1,
        rng=_NoRandom(),
        exploration_probability=Fraction(0, 1),
        eligible_references=eligible,
    )
    assert selected.selected == (promote.reference,)


def test_diagnostic_admission_remains_valid_after_one_card_is_deprecated() -> None:
    memory, _, entries, _, admission, _, _ = _setup()
    retained, deprecated, companion = entries
    memory.deprecate(deprecated.reference, reason="prospective counterexample")
    receipt = memory.quarantine_test_admission_receipt(
        admission.memory_admission_receipt_sha256
    )

    validated = memory.validate_quarantine_test_admission(
        receipt,
        eligible_references=tuple(sorted((retained.reference, companion.reference))),
    )

    assert validated == tuple(sorted((retained.reference, companion.reference)))


def test_diagnostic_admission_requires_explicit_authorization_for_subset() -> None:
    memory, _, entries, _, admission, _, _ = _setup()
    selected = (entries[0].reference,)
    receipt = memory.quarantine_test_admission_receipt(
        admission.memory_admission_receipt_sha256
    )

    with pytest.raises(ValueError, match="differ from the still-active"):
        memory.validate_quarantine_test_admission(
            receipt,
            eligible_references=selected,
        )
    assert memory.validate_quarantine_test_admission(
        receipt,
        eligible_references=selected,
        subset_authorization_sha256=_sha("prospective-complete-support-subset"),
    ) == selected


def test_closed_loop_rejects_same_generation_and_is_atomic_on_foreign_content() -> None:
    memory, learning, entries, _, _, context, barrier = _setup(trial_generation=2)
    promote, deprecate, _ = entries
    with pytest.raises(ValueError, match="same-generation"):
        learning.close_generation(
            memory_credit_batch=barrier,
            audits=(_audit_binding(promote, GlobalHypothesisVerdict.SUPPORT, context),),
        )
    assert memory.transitions == ()

    memory, learning, entries, _, _, context, barrier = _setup()
    promote, deprecate, _ = entries
    with pytest.raises(ValueError, match="different card content"):
        learning.close_generation(
            memory_credit_batch=barrier,
            audits=(
                _audit_binding(promote, GlobalHypothesisVerdict.SUPPORT, context),
                _audit_binding(
                    deprecate,
                    GlobalHypothesisVerdict.COUNTEREXAMPLE,
                    context,
                    content_sha256=_sha("foreign-card-content"),
                ),
            ),
        )
    assert memory.transitions == ()
    assert all(
        value.lifecycle_state is InsightLifecycleState.QUARANTINED
        for value in memory.entries_for(tuple(value.reference for value in entries))
    )


def test_registration_rejects_legacy_and_model_authored_invariant_cards() -> None:
    ids = DeterministicIdFactory("closed_loop_non_actionable")
    memory = InsightMemoryBank(id_factory=ids)
    exemplar = _reflection_entry(memory, ids, "exemplar")
    assert exemplar.evidence_lineage is not None
    invalid_drafts = (
        replace(
            exemplar.draft,
            claim="Legacy prose-only reflection",
            effect_predictions=(),
            recommended_option_families=(),
            action_template=None,
            falsification_condition=None,
            insight_kind=None,
            consumer_scopes=(),
            factor_capabilities=(),
        ),
        replace(
            exemplar.draft,
            claim="Model-authored invariant",
            insight_kind=ReflectionInsightKind.CONTRACT_INVARIANT,
        ),
    )
    learning = ClosedLoopCampaignLearning(memory=memory)
    context = _sha("non-actionable-context")
    for draft in invalid_drafts:
        entry, added = memory.add(
            draft,
            origin=InsightOrigin.REFLECTION,
            evidence_lineage=exemplar.evidence_lineage,
            applicable_operator_kinds=("typed_mutation",),
        )
        assert added
        plan = CampaignSemanticAuditPlan.from_request(
            _audit_binding(
                entry,
                GlobalHypothesisVerdict.INSUFFICIENT,
                context,
            ).request,
            draft_hypothesis_sha256=entry.draft.hypothesis_sha256,
        )
        with pytest.raises(
            ValueError,
            match="actionable semantic-v3|heuristic or invariant",
        ):
            learning.register_quarantined_reflections(
                origin_generation=2,
                references=(entry.reference,),
                semantic_audit_plans=(plan,),
            )


def test_close_rejects_a_different_predeclared_semantic_claim_atomically() -> None:
    memory, learning, entries, _, _, context, barrier = _setup()
    promote = entries[0]
    valid = _audit_binding(promote, GlobalHypothesisVerdict.SUPPORT, context)
    foreign_trigger = _predicate(
        "campaign_trigger",
        {"path": "$.sequence[1]"},
    )
    substituted_request = replace(valid.request, trigger=foreign_trigger)
    substituted = _rebind_audit_request(valid, substituted_request)

    with pytest.raises(ValueError, match="substituted the registered semantic plan"):
        learning.close_generation(
            memory_credit_batch=barrier,
            audits=(substituted,),
        )
    assert memory.transitions == ()


def test_campaign_audit_binding_rejects_policy_identity_substitution() -> None:
    _, _, entries, _, _, context, _ = _setup()
    valid = _audit_binding(
        entries[0],
        GlobalHypothesisVerdict.INSUFFICIENT,
        context,
    )
    substituted_request = replace(
        valid.request,
        audit_policy_definition_sha256=_sha("foreign-audit-policy"),
    )
    substituted_receipt = replace(
        valid.receipt,
        request_sha256=substituted_request.request_sha256,
        audit_policy_id="foreign_audit_policy",
        audit_policy_definition_sha256=_sha("foreign-audit-policy"),
    )
    with pytest.raises(ValueError, match="substituted the sealed policy"):
        CampaignInsightAuditBinding(
            request=substituted_request,
            receipt=substituted_receipt,
            exact_context_sha256=context,
        )
