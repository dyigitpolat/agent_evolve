from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
)
from agent_evolve.application.portfolio_optimization_memory import (
    PortfolioOptimizationMemoryDirective,
    PortfolioOptimizationMemoryDisposition,
    assess_portfolio_optimization_memory,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import freeze_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _semantics() -> OptimizationSemantics:
    metrics = tuple(
        MetricSemantics(
            metric_id=f"objective:{name}",
            name=name,
            role=MetricRole.OBJECTIVE,
            sense=sense,
            definition=f"Test objective {name}.",
            aggregation="One deterministic observation.",
            witness_interpretation="Declared direction determines improvement.",
            tolerance=0.0,
        )
        for name, sense in (
            ("cost", MetricSense.MINIMIZE),
            ("quality", MetricSense.MAXIMIZE),
        )
    )
    return OptimizationSemantics(
        semantics_id="signed_memory_test",
        semantics_version=1,
        metrics=metrics,
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=tuple(value.metric_id for value in metrics),
            description="Pareto ordering for the signed-memory unit test.",
            equivalence="Exact equality on both test objectives.",
            policy_id="signed_memory_test_relation",
            policy_version=1,
            definition_sha256=_sha("signed-memory-test-relation"),
        ),
    )


def _lineage(
    semantics: OptimizationSemantics,
    *,
    cost: str,
    quality: str,
    semantics_sha256: str | None = None,
) -> InsightEvidenceLineage:
    contrast_id = _sha(f"{cost}:{quality}:{semantics_sha256}")
    snapshot = EmpiricalEvidenceSnapshot(
        contrast_id=contrast_id,
        fact_schema_id=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
        fact_schema_version=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
        fact_schema_definition_sha256=(
            IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        ),
        facts=freeze_json(
            {
                "observed_metric_effects": [
                    {
                        "metric_id": "cost",
                        "direction": cost,
                        "delta_hex": "0x0.0p+0",
                        "adjudicator_definition_sha256": _sha("adjudicator"),
                    },
                    {
                        "metric_id": "quality",
                        "direction": quality,
                        "delta_hex": "0x0.0p+0",
                        "adjudicator_definition_sha256": _sha("adjudicator"),
                    },
                ]
            }
        ),
        optimization_semantics_definition_sha256=(
            semantics.definition_sha256
            if semantics_sha256 is None
            else semantics_sha256
        ),
    )
    return InsightEvidenceLineage(
        reflection_call_id=LLMCallId("call_signed_memory_test"),
        source_operator_invocation_ids=(),
        source_candidate_ids=(),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
        empirical_evidence=(snapshot,),
    )


@pytest.mark.parametrize(
    ("cost", "quality", "disposition", "directive", "dose"),
    (
        (
            "decrease",
            "increase",
            PortfolioOptimizationMemoryDisposition.DOMINANCE_FAVORABLE,
            PortfolioOptimizationMemoryDirective.PREFER_OR_TEST,
            True,
        ),
        (
            "increase",
            "decrease",
            PortfolioOptimizationMemoryDisposition.DOMINANCE_UNFAVORABLE,
            PortfolioOptimizationMemoryDirective.AVOID_EXCEPT_FALSIFICATION,
            False,
        ),
        (
            "decrease",
            "decrease",
            PortfolioOptimizationMemoryDisposition.TRADEOFF,
            PortfolioOptimizationMemoryDirective.CONSIDER_FOR_FRONTIER_TRADEOFF,
            False,
        ),
        (
            "unchanged",
            "unchanged",
            PortfolioOptimizationMemoryDisposition.NEUTRAL,
            PortfolioOptimizationMemoryDirective.DO_NOT_PRIORITIZE,
            False,
        ),
    ),
)
def test_authenticated_metric_senses_control_memory_dose(
    cost: str,
    quality: str,
    disposition: PortfolioOptimizationMemoryDisposition,
    directive: PortfolioOptimizationMemoryDirective,
    dose: bool,
) -> None:
    semantics = _semantics()
    assessment = assess_portfolio_optimization_memory(
        _lineage(semantics, cost=cost, quality=quality),
        semantics,
    )

    assert assessment.disposition is disposition
    assert assessment.directive is directive
    assert assessment.forced_action_dose_allowed is dose
    assert assessment.to_record()["model_prose_used_for_sign"] is False


def test_foreign_semantics_cannot_authorize_a_forced_dose() -> None:
    semantics = _semantics()
    assessment = assess_portfolio_optimization_memory(
        _lineage(
            semantics,
            cost="decrease",
            quality="increase",
            semantics_sha256=_sha("foreign-semantics"),
        ),
        semantics,
    )

    assert assessment.disposition is PortfolioOptimizationMemoryDisposition.UNRESOLVED
    assert assessment.forced_action_dose_allowed is False
    assert assessment.unresolved_reason == "optimization_semantics_mismatch"
