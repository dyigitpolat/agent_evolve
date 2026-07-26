"""Provider-free global evidence joining and append-only hypothesis revision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.policies.memory.global_falsification import (
    AmbiguousInterventionIdentityError,
    AppendOnlyHypothesisRevision,
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceDisposition,
    EvidenceProvenance,
    GlobalEvidenceRegistrySnapshot,
    GlobalHypothesisAuditRequest,
    GlobalHypothesisFalsificationGate,
    GlobalHypothesisVerdict,
    HypothesisAuditScope,
    HypothesisClaimStrength,
    HypothesisEvidenceMatchReceipt,
    HypothesisMetricPrediction,
    InterventionIdentifiability,
    InterventionMatch,
    ObservedMetricEffect,
    RevisionEvidenceTiming,
    TriggerMatch,
    TypedEvidencePredicate,
    TypedInterventionSignature,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


MATCHER_DEFINITION = _sha("matcher-definition")
EVALUATOR = _sha("evaluator")
ADJUDICATOR = _sha("adjudicator")
CAMPAIGN = _sha("campaign")
INSTANCE = _sha("instance")
REFERENCE = InsightRef(InsightId("insight_global_gate"), 1)


def _predicate(schema_id: str, payload: dict[str, object]) -> TypedEvidencePredicate:
    return TypedEvidencePredicate(
        schema_id=schema_id,
        schema_version=1,
        schema_definition_sha256=_sha(f"{schema_id}-schema"),
        payload=freeze_json(payload),
    )


TRIGGER = _predicate(
    "synthetic_trigger",
    {"all": [{"path": "$.sequence[0]", "equals": "fraig"}]},
)
OLD = _predicate(
    "synthetic_old_value",
    {"path": "$.sequence[5]", "equals": "rewrite_z"},
)
ACTION = _predicate(
    "synthetic_action",
    {"path": "$.sequence[5]", "replace_with": "balance"},
)
INTERVENTION = TypedInterventionSignature(
    affected_paths=("$.sequence[5]",),
    old_value_predicate=OLD,
    new_action=ACTION,
    admissible_operator_families=("replace",),
)


@dataclass(frozen=True)
class _ScriptedMatcher:
    classifications: dict[str, tuple[TriggerMatch, InterventionMatch]]
    policy_id: str = "synthetic_exact_matcher"
    policy_version: int = 1
    definition_sha256: str = MATCHER_DEFINITION

    def classify(self, request, observation):
        trigger, intervention = self.classifications[observation.source_evidence_id]
        return HypothesisEvidenceMatchReceipt(
            request_sha256=request.request_sha256,
            observation_sha256=observation.observation_sha256,
            trigger_match=trigger,
            intervention_match=intervention,
            matcher_policy_id=self.policy_id,
            matcher_policy_version=self.policy_version,
            matcher_definition_sha256=self.definition_sha256,
        )


def _metrics(
    *,
    levels: tuple[MetricEffectDirection, float] = (
        MetricEffectDirection.UNCHANGED,
        0.0,
    ),
    lut: tuple[MetricEffectDirection, float] = (
        MetricEffectDirection.DECREASE,
        -1.0,
    ),
    include_lut: bool = True,
) -> tuple[ObservedMetricEffect, ...]:
    values = [
        ObservedMetricEffect(
            metric_id="levels",
            direction=levels[0],
            delta=float(levels[1]),
            adjudicator_definition_sha256=ADJUDICATOR,
        )
    ]
    if include_lut:
        values.append(
            ObservedMetricEffect(
                metric_id="lut",
                direction=lut[0],
                delta=float(lut[1]),
                adjudicator_definition_sha256=ADJUDICATOR,
            )
        )
    return tuple(sorted(values, key=lambda value: value.metric_id))


def _observation(
    label: str,
    *,
    event_index: int,
    metrics: tuple[ObservedMetricEffect, ...] | None = None,
    trigger_value: str = "fraig",
    identifiability: InterventionIdentifiability = (
        InterventionIdentifiability.EXACT_SINGLE
    ),
    wave: str = "wave-a",
    portfolio: str = "portfolio-a",
    lineage: str = "lineage-a",
    block: str = "block-a",
    mechanism_identifying: bool = False,
    provenance: EvidenceProvenance = EvidenceProvenance.DIRECT_MUTATION,
    operator_family: str = "replace",
    affected_paths: tuple[str, ...] = ("$.sequence[5]",),
    parent_configuration=None,
    child_configuration=None,
) -> AuthenticatedHypothesisObservation:
    parent = freeze_json(
        parent_configuration
        or {
            "label": label,
            "sequence": [trigger_value, "rewrite_z"],
            "event": event_index,
        }
    )
    child = freeze_json(
        child_configuration
        or {
            "label": label,
            "sequence": [trigger_value, "balance"],
            "event": event_index,
        }
    )
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"evidence-{label}"),
        event_index=event_index,
        workload_instance_sha256=INSTANCE,
        evaluator_contract_sha256=EVALUATOR,
        campaign_sha256=CAMPAIGN,
        parent_candidate_id=CandidateId(f"candidate_parent_{label}"),
        child_candidate_id=CandidateId(f"candidate_child_{label}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{label}"),
        finite_contract_identity_sha256=_sha("finite-contract"),
        provenance=provenance,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(wave),
            estimand_unit=CausalEstimandUnit.PORTFOLIO,
            portfolio_sha256=_sha(portfolio),
            prospective_assignment_sha256=_sha(f"assignment-{portfolio}"),
        ),
        parent_configuration=parent,
        child_configuration=child,
        parent_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(parent)
        ),
        child_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(child)
        ),
        parent_outcome_sha256=_sha(f"parent-outcome-{label}"),
        child_outcome_sha256=_sha(f"child-outcome-{label}"),
        operator_family=operator_family,
        affected_paths=affected_paths,
        observed_action=freeze_json(
            {"path": "$.sequence[5]", "replace_with": "balance"}
        ),
        action_semantics_compiler_id="test_action_semantics",
        action_semantics_compiler_version=1,
        action_semantics_definition_sha256=_sha("action-semantics-definition"),
        intervention_identifiability=identifiability,
        metrics=_metrics() if metrics is None else metrics,
        lineage_cluster_sha256=_sha(lineage),
        factorial_block_sha256=_sha(block),
        mechanism_identifying_design=mechanism_identifying,
    )


def _predictions(*, lut_interval=None):
    lower, upper = (None, None) if lut_interval is None else lut_interval
    return (
        HypothesisMetricPrediction(
            metric_id="levels",
            direction=MetricEffectDirection.UNCHANGED,
        ),
        HypothesisMetricPrediction(
            metric_id="lut",
            direction=MetricEffectDirection.DECREASE,
            minimum_delta=lower,
            maximum_delta=upper,
        ),
    )


def _request(
    registry,
    *,
    claim_strength=None,
    origin_cutoff=None,
    audit_cutoff=50,
    minimum_clusters=2,
    minimum_instances=1,
    predictions=None,
):
    origin_cutoff = 0 if origin_cutoff is None else origin_cutoff
    return GlobalHypothesisAuditRequest(
        reference=REFERENCE,
        draft_content_sha256=_sha("draft-v1"),
        trigger=TRIGGER,
        intervention=INTERVENTION,
        predictions=_predictions() if predictions is None else predictions,
        claim_strength=claim_strength or HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(INSTANCE,),
            evaluator_contract_sha256=EVALUATOR,
            metric_adjudicator_definition_sha256=ADJUDICATOR,
            campaign_sha256s=(CAMPAIGN,),
        ),
        matcher_definition_sha256=MATCHER_DEFINITION,
        origin_cutoff_event_index=origin_cutoff,
        audit_cutoff_event_index=audit_cutoff,
        registry_snapshot_sha256=registry.snapshot_sha256,
        minimum_support_clusters=minimum_clusters,
        minimum_support_instances=minimum_instances,
    )


def _audit(observations, classifications, **request_kwargs):
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=max(
            request_kwargs.get("audit_cutoff", 50),
            *(value.event_index for value in observations),
        ),
        observations=observations,
    )
    request = _request(registry, **request_kwargs)
    receipt = GlobalHypothesisFalsificationGate().audit(
        request=request,
        registry=registry,
        matcher=_ScriptedMatcher(classifications),
    )
    return request, registry, receipt


def test_global_join_finds_omitted_earlier_off_trigger_counterexample_to_necessity() -> (
    None
):
    """Synthetic analogue of BOiLS H2's omitted cycle-2 evidence pattern."""

    latest_trigger_cell = _observation(
        "latest-trigger-cell",
        event_index=30,
        trigger_value="fraig",
        metrics=_metrics(lut=(MetricEffectDirection.DECREASE, -185.0)),
        wave="wave-3",
        portfolio="portfolio-3",
        lineage="lineage-local",
        block="factorial-3",
        provenance=EvidenceProvenance.RECOMBINATION_EXACT_ABLATION,
    )
    omitted_earlier_control = _observation(
        "earlier-off-trigger-benefit",
        event_index=12,
        trigger_value="balance",
        metrics=_metrics(lut=(MetricEffectDirection.DECREASE, -162.0)),
        wave="wave-2",
        portfolio="portfolio-2",
        lineage="lineage-earlier",
        block="factorial-2",
    )
    classifications = {
        latest_trigger_cell.source_evidence_id: (
            TriggerMatch.EXACT,
            InterventionMatch.EXACT,
        ),
        omitted_earlier_control.source_evidence_id: (
            TriggerMatch.OFF_TRIGGER,
            InterventionMatch.EXACT,
        ),
    }

    _, _, receipt = _audit(
        (latest_trigger_cell, omitted_earlier_control),
        classifications,
        claim_strength=HypothesisClaimStrength(necessity=True),
        minimum_clusters=1,
    )

    assert receipt.raw_support_count == 1
    assert receipt.support_ids == (latest_trigger_cell.source_evidence_id,)
    assert receipt.off_trigger_control_ids == (
        omitted_earlier_control.source_evidence_id,
    )
    control = next(
        value
        for value in receipt.decisions
        if value.source_evidence_id == omitted_earlier_control.source_evidence_id
    )
    assert control.disposition is EvidenceDisposition.OFF_TRIGGER_CONTROL
    assert control.predictions_match is True
    assert receipt.necessity_contradicted is True
    assert receipt.verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE
    assert receipt.causal_credit_updates == ()
    assert receipt.to_record()["causal_credit_updates"] == []


def test_correlated_supports_count_as_one_effective_portfolio_cluster() -> None:
    observations = tuple(
        _observation(
            f"correlated-{index}",
            event_index=10 + index,
            metrics=_metrics(lut=(MetricEffectDirection.DECREASE, -float(index + 1))),
            wave="same-wave",
            portfolio="same-portfolio",
            lineage="same-lineage",
            block="same-factorial-block",
            provenance=(
                EvidenceProvenance.DIRECT_MUTATION
                if index < 2
                else EvidenceProvenance.RECOMBINATION_EXACT_ABLATION
            ),
        )
        for index in range(5)
    )
    classifications = {
        value.source_evidence_id: (TriggerMatch.EXACT, InterventionMatch.EXACT)
        for value in observations
    }

    _, _, receipt = _audit(
        observations,
        classifications,
        minimum_clusters=2,
    )

    assert receipt.raw_support_count == 5
    assert receipt.effective_support_cluster_count == 1
    assert receipt.support_instance_count == 1
    assert receipt.verdict is GlobalHypothesisVerdict.INSUFFICIENT
    assert "support_cluster_threshold_not_met" in receipt.coverage_gaps
    assert (
        len(
            {
                value.effective_cluster_sha256
                for value in receipt.decisions
                if value.disposition is EvidenceDisposition.SUPPORT
            }
        )
        == 1
    )


def test_origin_only_support_is_reported_as_insufficient_revision_evidence() -> None:
    origin_support = _observation(
        "origin-only-support",
        event_index=4,
        wave="origin-wave",
        portfolio="origin-portfolio",
        lineage="origin-lineage",
        block="origin-block",
    )
    _, _, receipt = _audit(
        (origin_support,),
        {
            origin_support.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            )
        },
        origin_cutoff=5,
        audit_cutoff=10,
        minimum_clusters=1,
    )

    assert receipt.raw_support_count == 1
    assert receipt.effective_support_cluster_count == 1
    assert receipt.verdict is GlobalHypothesisVerdict.INSUFFICIENT
    assert "post_origin_support_absent" in receipt.coverage_gaps
    assert "post_origin_support_cluster_threshold_not_met" in receipt.coverage_gaps
    assert "post_origin_support_instance_threshold_not_met" in receipt.coverage_gaps
    assert receipt.lifecycle_decision == (
        "quarantined__undersupported_or_scope_restricted"
    )


def test_direction_and_magnitude_are_adjudicated_separately() -> None:
    observation = _observation(
        "magnitude-counterexample",
        event_index=4,
        metrics=_metrics(lut=(MetricEffectDirection.DECREASE, -4.0)),
    )
    classifications = {
        observation.source_evidence_id: (
            TriggerMatch.EXACT,
            InterventionMatch.EXACT,
        )
    }
    _, _, receipt = _audit(
        (observation,),
        classifications,
        predictions=_predictions(lut_interval=(-200.0, -50.0)),
        minimum_clusters=1,
    )

    assert receipt.verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE
    decision = receipt.decisions[0]
    lut = next(
        value for value in decision.metric_assessments if value.metric_id == "lut"
    )
    assert lut.direction_matches is True
    assert lut.magnitude_matches is False
    assert decision.disposition is EvidenceDisposition.COUNTEREXAMPLE


def test_missing_metric_and_joint_intervention_are_non_identifiable() -> None:
    missing_metric = _observation(
        "missing-metric",
        event_index=3,
        metrics=_metrics(include_lut=False),
    )
    joint = _observation(
        "joint",
        event_index=4,
        identifiability=InterventionIdentifiability.JOINT_WITHOUT_ABLATION,
    )
    classifications = {
        missing_metric.source_evidence_id: (
            TriggerMatch.EXACT,
            InterventionMatch.EXACT,
        ),
        joint.source_evidence_id: (
            TriggerMatch.EXACT,
            InterventionMatch.NON_IDENTIFIABLE,
        ),
    }

    _, _, receipt = _audit(
        (missing_metric, joint),
        classifications,
        minimum_clusters=1,
    )

    assert receipt.verdict is GlobalHypothesisVerdict.NON_IDENTIFIABLE
    assert receipt.non_identifiable_ids == tuple(
        sorted((missing_metric.source_evidence_id, joint.source_evidence_id))
    )


def test_ambiguous_intervention_cannot_be_declared_an_exact_match() -> None:
    ambiguous = _observation(
        "ambiguous",
        event_index=3,
        identifiability=InterventionIdentifiability.AMBIGUOUS,
    )
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=5,
        observations=(ambiguous,),
    )
    request = _request(registry, audit_cutoff=5, minimum_clusters=1)

    with pytest.raises(AmbiguousInterventionIdentityError, match="fail closed"):
        GlobalHypothesisFalsificationGate().audit(
            request=request,
            registry=registry,
            matcher=_ScriptedMatcher(
                {
                    ambiguous.source_evidence_id: (
                        TriggerMatch.EXACT,
                        InterventionMatch.EXACT,
                    )
                }
            ),
        )


def test_exact_match_cannot_override_sealed_path_or_operator_family() -> None:
    wrong_path = _observation(
        "wrong-path",
        event_index=3,
        affected_paths=("$.sequence[6]",),
    )
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=5,
        observations=(wrong_path,),
    )
    request = _request(registry, audit_cutoff=5, minimum_clusters=1)
    matcher = _ScriptedMatcher(
        {
            wrong_path.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            )
        }
    )

    with pytest.raises(ValueError, match="sealed path or operator family"):
        GlobalHypothesisFalsificationGate().audit(
            request=request,
            registry=registry,
            matcher=matcher,
        )


def test_mechanism_language_requires_randomized_identifying_design() -> None:
    observational_support = _observation("observational", event_index=3)
    classifications = {
        observational_support.source_evidence_id: (
            TriggerMatch.EXACT,
            InterventionMatch.EXACT,
        )
    }
    _, _, receipt = _audit(
        (observational_support,),
        classifications,
        claim_strength=HypothesisClaimStrength(mechanistic_or_causal=True),
        minimum_clusters=1,
    )
    assert receipt.raw_support_count == 1
    assert receipt.mechanism_identified is False
    assert receipt.verdict is GlobalHypothesisVerdict.NON_IDENTIFIABLE
    assert "mechanism_identifying_trial_absent" in receipt.coverage_gaps

    randomized = _observation(
        "randomized",
        event_index=3,
        provenance=EvidenceProvenance.RANDOMIZED_ADMINISTRATION,
        mechanism_identifying=True,
    )
    _, _, identified = _audit(
        (randomized,),
        {
            randomized.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            )
        },
        claim_strength=HypothesisClaimStrength(mechanistic_or_causal=True),
        minimum_clusters=1,
    )
    assert identified.mechanism_identified is True
    assert identified.verdict is GlobalHypothesisVerdict.SUPPORT


def test_randomized_evidence_requires_a_prospective_itt_assignment() -> None:
    boundary = EvidenceCausalBoundary(
        wave_sha256=_sha("unassigned-wave"),
        estimand_unit=CausalEstimandUnit.WAVE,
    )
    parent = freeze_json({"x": 1})
    child = freeze_json({"x": 2})
    with pytest.raises(ValueError, match="prospective ITT assignment"):
        AuthenticatedHypothesisObservation(
            source_evidence_id=_sha("unassigned-evidence"),
            event_index=1,
            workload_instance_sha256=INSTANCE,
            evaluator_contract_sha256=EVALUATOR,
            campaign_sha256=CAMPAIGN,
            parent_candidate_id=CandidateId("candidate_unassigned_parent"),
            child_candidate_id=CandidateId("candidate_unassigned_child"),
            operator_invocation_id=OperatorInvocationId("operator_unassigned"),
            finite_contract_identity_sha256=_sha("finite-contract"),
            provenance=EvidenceProvenance.RANDOMIZED_ADMINISTRATION,
            causal_boundary=boundary,
            parent_configuration=parent,
            child_configuration=child,
            parent_configuration_sha256=(
                AuthenticatedHypothesisObservation.configuration_sha256(parent)
            ),
            child_configuration_sha256=(
                AuthenticatedHypothesisObservation.configuration_sha256(child)
            ),
            parent_outcome_sha256=_sha("unassigned-parent-outcome"),
            child_outcome_sha256=_sha("unassigned-child-outcome"),
            operator_family="replace",
            affected_paths=("$.x",),
            observed_action=freeze_json({"replace_with": 2}),
            action_semantics_compiler_id="test_action_semantics",
            action_semantics_compiler_version=1,
            action_semantics_definition_sha256=_sha("action-semantics-definition"),
            intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
            metrics=_metrics(),
            lineage_cluster_sha256=_sha("unassigned-lineage"),
            factorial_block_sha256=_sha("unassigned-block"),
            mechanism_identifying_design=True,
        )


def test_registry_order_is_canonical_and_exact_repeats_are_deduplicated() -> None:
    parent = {"shared": "parent"}
    child = {"shared": "child"}
    earlier = _observation(
        "z-earlier",
        event_index=2,
        parent_configuration=parent,
        child_configuration=child,
    )
    later = _observation(
        "a-later",
        event_index=9,
        parent_configuration=parent,
        child_configuration=child,
    )
    forward = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=10,
        observations=(earlier, later),
    )
    reverse = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=10,
        observations=(later, earlier),
    )
    assert forward.snapshot_sha256 == reverse.snapshot_sha256
    request = _request(forward, audit_cutoff=10, minimum_clusters=1)
    matcher = _ScriptedMatcher(
        {
            earlier.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            ),
            later.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            ),
        }
    )
    receipt = GlobalHypothesisFalsificationGate().audit(
        request=request,
        registry=forward,
        matcher=matcher,
    )
    by_id = {value.source_evidence_id: value for value in receipt.decisions}
    assert by_id[earlier.source_evidence_id].disposition is EvidenceDisposition.SUPPORT
    assert by_id[later.source_evidence_id].disposition is EvidenceDisposition.DUPLICATE
    assert by_id[later.source_evidence_id].duplicate_of_source_evidence_id == (
        earlier.source_evidence_id
    )


def test_identical_configurations_across_campaigns_are_independent_evidence() -> None:
    parent = {"shared": "parent"}
    child = {"shared": "child"}
    first = _observation(
        "campaign-one",
        event_index=3,
        parent_configuration=parent,
        child_configuration=child,
    )
    second = replace(
        _observation(
            "campaign-two",
            event_index=4,
            parent_configuration=parent,
            child_configuration=child,
        ),
        campaign_sha256=_sha("independent-campaign-two"),
    )
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=5,
        observations=(first, second),
    )
    request = GlobalHypothesisAuditRequest(
        reference=REFERENCE,
        draft_content_sha256=_sha("draft-v1"),
        trigger=TRIGGER,
        intervention=INTERVENTION,
        predictions=_predictions(),
        claim_strength=HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(INSTANCE,),
            evaluator_contract_sha256=EVALUATOR,
            metric_adjudicator_definition_sha256=ADJUDICATOR,
            campaign_sha256s=tuple(
                sorted((first.campaign_sha256, second.campaign_sha256))
            ),
        ),
        matcher_definition_sha256=MATCHER_DEFINITION,
        origin_cutoff_event_index=0,
        audit_cutoff_event_index=5,
        registry_snapshot_sha256=registry.snapshot_sha256,
        minimum_support_clusters=2,
        minimum_support_instances=1,
    )
    matcher = _ScriptedMatcher(
        {
            first.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            ),
            second.source_evidence_id: (
                TriggerMatch.EXACT,
                InterventionMatch.EXACT,
            ),
        }
    )
    receipt = GlobalHypothesisFalsificationGate().audit(
        request=request,
        registry=registry,
        matcher=matcher,
    )

    assert receipt.raw_support_count == 2
    assert all(
        value.disposition is EvidenceDisposition.SUPPORT for value in receipt.decisions
    )


def test_append_only_revision_preserves_lineage_and_resets_trial_credit() -> None:
    origin_support = _observation("origin-support", event_index=4)
    later_counterexample = _observation(
        "later-counterexample",
        event_index=8,
        metrics=_metrics(
            levels=(MetricEffectDirection.INCREASE, 1.0),
            lut=(MetricEffectDirection.INCREASE, 7.0),
        ),
        wave="later-wave",
        portfolio="later-portfolio",
    )
    observations = (origin_support, later_counterexample)
    classifications = {
        value.source_evidence_id: (TriggerMatch.EXACT, InterventionMatch.EXACT)
        for value in observations
    }
    request, _, receipt = _audit(
        observations,
        classifications,
        origin_cutoff=5,
        audit_cutoff=10,
        minimum_clusters=1,
    )
    successor_trigger = _predicate(
        "narrower_trigger",
        {"all": [{"path": "$.sequence[0]", "equals": "fraig"}, {"held_out": True}]},
    )
    revision = AppendOnlyHypothesisRevision.from_audit(
        request=request,
        receipt=receipt,
        successor=InsightRef(REFERENCE.insight_id, 2),
        successor_draft_content_sha256=_sha("draft-v2-narrower"),
        successor_trigger_predicate_sha256=successor_trigger.predicate_sha256,
        successor_intervention_signature_sha256=INTERVENTION.signature_sha256,
        successor_scope_sha256=request.scope.scope_sha256,
        claim_diff="Remove invariance and state a narrower local implication.",
        scope_diff="Reset scope to a held-out local trial.",
    )

    assert revision.predecessor == REFERENCE
    assert revision.successor == InsightRef(REFERENCE.insight_id, 2)
    assert revision.successor_lifecycle_state == "quarantined"
    assert revision.trial_eligibility_reset is True
    assert revision.inherited_confirmation_count == 0
    assert revision.inherited_causal_credit is False
    timing = {value.source_evidence_id: value.timing for value in revision.evidence}
    assert timing[origin_support.source_evidence_id] is (
        RevisionEvidenceTiming.AVAILABLE_AT_ORIGIN
    )
    assert timing[later_counterexample.source_evidence_id] is (
        RevisionEvidenceTiming.POST_ORIGIN_REVISION_EVIDENCE
    )
    assert revision.to_record()["inherited_confirmation_count"] == 0

    with pytest.raises(ValueError, match="exact next version"):
        AppendOnlyHypothesisRevision.from_audit(
            request=request,
            receipt=receipt,
            successor=InsightRef(REFERENCE.insight_id, 3),
            successor_draft_content_sha256=_sha("invalid-v3"),
            successor_trigger_predicate_sha256=successor_trigger.predicate_sha256,
            successor_intervention_signature_sha256=INTERVENTION.signature_sha256,
            successor_scope_sha256=request.scope.scope_sha256,
            claim_diff="Invalid lineage skip.",
            scope_diff="Invalid lineage skip.",
        )


def test_request_rejects_foreign_registry_or_matcher_identity() -> None:
    observation = _observation("identity", event_index=2)
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=3,
        observations=(observation,),
    )
    request = _request(registry, audit_cutoff=3, minimum_clusters=1)
    foreign = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=4,
        observations=(observation,),
    )
    with pytest.raises(ValueError, match="different registry"):
        GlobalHypothesisFalsificationGate().audit(
            request=request,
            registry=foreign,
            matcher=_ScriptedMatcher({}),
        )

    with pytest.raises(ValueError, match="matcher identity"):
        GlobalHypothesisFalsificationGate().audit(
            request=request,
            registry=registry,
            matcher=_ScriptedMatcher(
                {},
                definition_sha256=_sha("foreign-matcher"),
            ),
        )
