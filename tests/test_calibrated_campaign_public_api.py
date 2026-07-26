from __future__ import annotations

import agent_evolve as public_api

from agent_evolve.application.calibrated_campaign import (
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
    equal_weight_slate_objectives_from_decision_metrics,
    equal_weight_slate_objectives_from_optimization_semantics,
)
from agent_evolve.application.decision_metric_projection import (
    ProjectedDecisionMetrics,
    project_candidate_decision_metrics,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    CalibratedCampaignOutcomeUpdater,
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.application.parent_measurement import (
    PARENT_MEASUREMENT_CONTEXT_KEY,
    attach_parent_measurement_to_context,
    bind_parent_measurement,
    create_parent_measurement_projection,
)
from agent_evolve.policies.objective_resolution import (
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.policies.memory import (
    AppendOnlyHypothesisRevision,
    AuthenticatedHypothesisObservation,
    EvidenceCausalBoundary,
    GlobalEvidenceRegistrySnapshot,
    GlobalHypothesisAuditReceipt,
    GlobalHypothesisAuditRequest,
    GlobalHypothesisEvidenceMatcher,
    GlobalHypothesisFalsificationGate,
    GlobalHypothesisVerdict,
    HypothesisAuditScope,
    HypothesisClaimStrength,
    HypothesisMetricPrediction,
    TypedEvidencePredicate,
    TypedInterventionSignature,
)
from agent_evolve.policies.selection import (
    AbsoluteToleranceDirectionAdjudicator,
    FiniteOptionPromptProjection,
    FiniteOptionPromptProjectionPolicy,
    FiniteOptionPromptRecord,
    FinitePaletteStructuralEvidencePolicy,
    ModelAnchoredCalibratedSlatePolicy,
    ModelAnchoredSlateDecision,
    ModelAnchoredSlateRole,
    PromptMetadataProjectionMode,
)
from agent_evolve.ports.decision_metric_projection import (
    DecisionMetricBinding,
    DecisionMetricProjection,
    DecisionMetricValueSource,
)
from agent_evolve.ports.parent_measurement import (
    ParentCandidateMeasurementIdentity,
    ParentDecisionMetricValue,
    ParentMeasurementBinding,
    ParentMeasurementProjection,
    ParentRawScientificMetricValue,
)


def test_calibrated_campaign_provider_neutral_facade_is_explicit() -> None:
    expected = {
        "AbsoluteToleranceDirectionAdjudicator": (
            AbsoluteToleranceDirectionAdjudicator
        ),
        "AppendOnlyHypothesisRevision": AppendOnlyHypothesisRevision,
        "AuthenticatedHypothesisObservation": AuthenticatedHypothesisObservation,
        "CalibratedCampaignBindingFactory": CalibratedCampaignBindingFactory,
        "CalibratedCampaignOutcomeUpdater": CalibratedCampaignOutcomeUpdater,
        "DecisionMetricBinding": DecisionMetricBinding,
        "DecisionMetricProjection": DecisionMetricProjection,
        "DecisionMetricValueSource": DecisionMetricValueSource,
        "EvidenceCausalBoundary": EvidenceCausalBoundary,
        "FinitePaletteStructuralEvidencePolicy": (
            FinitePaletteStructuralEvidencePolicy
        ),
        "FiniteOptionPromptProjection": FiniteOptionPromptProjection,
        "FiniteOptionPromptProjectionPolicy": FiniteOptionPromptProjectionPolicy,
        "FiniteOptionPromptRecord": FiniteOptionPromptRecord,
        "FixedGridMetricSpec": FixedGridMetricSpec,
        "FixedGridObjectiveResolution": FixedGridObjectiveResolution,
        "FixedGridRoundingLaw": FixedGridRoundingLaw,
        "GlobalEvidenceRegistrySnapshot": GlobalEvidenceRegistrySnapshot,
        "GlobalHypothesisAuditReceipt": GlobalHypothesisAuditReceipt,
        "GlobalHypothesisAuditRequest": GlobalHypothesisAuditRequest,
        "GlobalHypothesisEvidenceMatcher": GlobalHypothesisEvidenceMatcher,
        "GlobalHypothesisFalsificationGate": GlobalHypothesisFalsificationGate,
        "GlobalHypothesisVerdict": GlobalHypothesisVerdict,
        "HypothesisAuditScope": HypothesisAuditScope,
        "HypothesisClaimStrength": HypothesisClaimStrength,
        "HypothesisMetricPrediction": HypothesisMetricPrediction,
        "ModelAnchoredCalibratedSlatePolicy": ModelAnchoredCalibratedSlatePolicy,
        "ModelAnchoredSlateDecision": ModelAnchoredSlateDecision,
        "ModelAnchoredSlateRole": ModelAnchoredSlateRole,
        "PortfolioOutcomeFeedbackLedger": PortfolioOutcomeFeedbackLedger,
        "PARENT_MEASUREMENT_CONTEXT_KEY": PARENT_MEASUREMENT_CONTEXT_KEY,
        "ParentCandidateMeasurementIdentity": ParentCandidateMeasurementIdentity,
        "ParentDecisionMetricValue": ParentDecisionMetricValue,
        "ParentMeasurementBinding": ParentMeasurementBinding,
        "ParentMeasurementProjection": ParentMeasurementProjection,
        "ParentRawScientificMetricValue": ParentRawScientificMetricValue,
        "ProjectedDecisionMetrics": ProjectedDecisionMetrics,
        "PromptMetadataProjectionMode": PromptMetadataProjectionMode,
        "TypedEvidencePredicate": TypedEvidencePredicate,
        "TypedInterventionSignature": TypedInterventionSignature,
        "equal_weight_slate_objectives": equal_weight_slate_objectives,
        "equal_weight_slate_objectives_from_decision_metrics": (
            equal_weight_slate_objectives_from_decision_metrics
        ),
        "equal_weight_slate_objectives_from_optimization_semantics": (
            equal_weight_slate_objectives_from_optimization_semantics
        ),
        "project_candidate_decision_metrics": project_candidate_decision_metrics,
        "attach_parent_measurement_to_context": (attach_parent_measurement_to_context),
        "bind_parent_measurement": bind_parent_measurement,
        "create_parent_measurement_projection": create_parent_measurement_projection,
    }

    assert set(expected).issubset(public_api.__all__)
    for name, value in expected.items():
        assert getattr(public_api, name) is value


def test_provider_specific_calibrated_adapter_stays_in_integration_facade() -> None:
    import agent_evolve.integrations.pydantic_ai as pydantic_ai_api

    assert "CalibratedPortfolioCampaignCoordinator" in pydantic_ai_api.__all__
    assert "CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256" in (
        pydantic_ai_api.__all__
    )
    assert "CalibratedPortfolioFeasibilityWitnessMode" in pydantic_ai_api.__all__
    assert "calibrated_portfolio_prompt_definition_sha256" in (pydantic_ai_api.__all__)
    assert "PydanticAICalibratedPortfolioSelectionPolicy" in (pydantic_ai_api.__all__)
    assert not hasattr(public_api, "PydanticAICalibratedPortfolioSelectionPolicy")
