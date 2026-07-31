from __future__ import annotations

import agent_evolve as public_api

from agent_evolve.application.campaign_contextual_outcomes import (
    AvailableFamilyOutcomeRelevance,
    CampaignOutcomeRelevanceResolver,
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.campaign_learning import (
    CampaignCausalUsefulnessReceipt,
    CampaignDiagnosticAdmissionReceipt,
    CampaignInsightAuditBinding,
    CampaignInsightDecisionReceipt,
    CampaignInsightLifecycleDecision,
    CampaignInsightPromotionPolicy,
    CampaignInsightRegistrationReceipt,
    CampaignLearningBarrierReceipt,
    CampaignPreparedLearningBarrier,
    CampaignRandomizedPromotionEvidence,
    CampaignSemanticAuditPlan,
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.campaign_learning_runtime import (
    CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY,
    CampaignDiagnosticExposureReceipt,
    CampaignGenerationAuditProjection,
    CampaignInsightSemanticCompiler,
    CampaignReflectedInsightProjection,
    CampaignReflectionLearningRecord,
    CampaignReflectionLearningRecordCodec,
    CampaignReflectionLearningProjection,
    CampaignReflectionLearningProjectionPort,
    CampaignRuntimeReflectionRegistrationReceipt,
    CampaignSemanticAuditPlanTemplate,
    ClosedLoopCampaignLearningRuntime,
    CompiledCampaignInsightSemantics,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.campaign_generation_audit import (
    CampaignDiagnosticContextBinding,
    CampaignGenerationAuditPreparation,
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.insight_memory import InsightLifecycleChangeRequest
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignDecisionSlot,
    CampaignLearningLifecyclePort,
    CampaignParentLane,
    CampaignPortfolioContextEnricher,
    CampaignPortfolioMemoryEstimandProjection,
    CampaignPortfolioMemoryEstimandProjector,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryCreditBatchReceipt,
    PortfolioPendingMemoryCredit,
)
from agent_evolve.ports.agentic_generator import (
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    ReflectionConsumerScope,
    ReflectionInsightKind,
)


def test_closed_loop_campaign_facade_is_provider_and_workload_neutral() -> None:
    expected = {
        "CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY": (
            CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY
        ),
        "AvailableFamilyOutcomeRelevance": AvailableFamilyOutcomeRelevance,
        "CampaignCausalUsefulnessReceipt": CampaignCausalUsefulnessReceipt,
        "CampaignDecisionSlot": CampaignDecisionSlot,
        "CampaignDiagnosticAdmissionReceipt": CampaignDiagnosticAdmissionReceipt,
        "CampaignInsightAuditBinding": CampaignInsightAuditBinding,
        "CampaignInsightDecisionReceipt": CampaignInsightDecisionReceipt,
        "CampaignInsightLifecycleDecision": CampaignInsightLifecycleDecision,
        "CampaignInsightPromotionPolicy": CampaignInsightPromotionPolicy,
        "CampaignInsightRegistrationReceipt": CampaignInsightRegistrationReceipt,
        "CampaignLearningBarrierReceipt": CampaignLearningBarrierReceipt,
        "CampaignPreparedLearningBarrier": CampaignPreparedLearningBarrier,
        "CampaignRandomizedPromotionEvidence": (CampaignRandomizedPromotionEvidence),
        "CampaignSemanticAuditPlan": CampaignSemanticAuditPlan,
        "CampaignSemanticAuditPlanTemplate": CampaignSemanticAuditPlanTemplate,
        "CampaignDiagnosticContextBinding": CampaignDiagnosticContextBinding,
        "CampaignDiagnosticExposureReceipt": CampaignDiagnosticExposureReceipt,
        "CampaignGenerationAuditPreparation": CampaignGenerationAuditPreparation,
        "CampaignGenerationAuditProjection": CampaignGenerationAuditProjection,
        "CampaignInsightSemanticCompiler": CampaignInsightSemanticCompiler,
        "CampaignReflectedInsightProjection": CampaignReflectedInsightProjection,
        "CampaignReflectionLearningRecord": CampaignReflectionLearningRecord,
        "CampaignReflectionLearningRecordCodec": (
            CampaignReflectionLearningRecordCodec
        ),
        "CampaignReflectionLearningProjection": (CampaignReflectionLearningProjection),
        "CampaignReflectionLearningProjectionPort": (
            CampaignReflectionLearningProjectionPort
        ),
        "CampaignRuntimeReflectionRegistrationReceipt": (
            CampaignRuntimeReflectionRegistrationReceipt
        ),
        "CampaignLearningLifecyclePort": CampaignLearningLifecyclePort,
        "CampaignOutcomeRelevanceResolver": CampaignOutcomeRelevanceResolver,
        "CampaignParentLane": CampaignParentLane,
        "CampaignPortfolioContextEnricher": CampaignPortfolioContextEnricher,
        "CampaignPortfolioMemoryEstimandProjection": (
            CampaignPortfolioMemoryEstimandProjection
        ),
        "CampaignPortfolioMemoryEstimandProjector": (
            CampaignPortfolioMemoryEstimandProjector
        ),
        "ClosedLoopCampaignLearning": ClosedLoopCampaignLearning,
        "ClosedLoopCampaignLearningRuntime": ClosedLoopCampaignLearningRuntime,
        "CompiledCampaignInsightSemantics": CompiledCampaignInsightSemantics,
        "ContextualOutcomeCampaignEnricher": ContextualOutcomeCampaignEnricher,
        "InsightLifecycleChangeRequest": InsightLifecycleChangeRequest,
        "MetricComparisonAnchor": MetricComparisonAnchor,
        "MetricComparisonAnchorKind": MetricComparisonAnchorKind,
        "PortfolioMemoryCreditBatchReceipt": PortfolioMemoryCreditBatchReceipt,
        "PortfolioPendingMemoryCredit": PortfolioPendingMemoryCredit,
        "ReflectionConsumerScope": ReflectionConsumerScope,
        "ReflectionInsightKind": ReflectionInsightKind,
        "StructuredCampaignReflectionLearningProjector": (
            StructuredCampaignReflectionLearningProjector
        ),
        "TransactionalPortfolioGenerationAuditor": (
            TransactionalPortfolioGenerationAuditor
        ),
    }
    # ``__all__`` is the small supported surface now; these research symbols
    # resolve lazily instead. Reachability is the invariant that matters --
    # the measured stack must keep importing exactly what it always did.
    reachable = set(public_api.__all__) | set(public_api._LEGACY_EXPORTS)
    assert set(expected).issubset(reachable)
    for name, value in expected.items():
        assert getattr(public_api, name) is value
