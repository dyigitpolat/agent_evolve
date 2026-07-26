from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.calibrated_campaign import (
    equal_weight_slate_objectives,
    equal_weight_slate_objectives_from_decision_metrics,
    equal_weight_slate_objectives_from_optimization_semantics,
)
from agent_evolve.application.decision_metric_projection import (
    project_candidate_decision_metrics,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluatorIdentity,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioEvolution,
    PortfolioVariationWaveRequest,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    observe_selected_portfolio_forecasts,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.meaningful_direction import (
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    resolve_ranked_portfolio_decision,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


def _objective_only_semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="boils_like_objective_only",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:total_levels",
                name="total_levels",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Mapped logic depth.",
                aggregation="One scalar.",
                witness_interpretation="Lower depth is better.",
                tolerance=0.0,
            ),
            MetricSemantics(
                metric_id="objective:total_lut_count",
                name="total_lut_count",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Mapped LUT count.",
                aggregation="One scalar.",
                witness_interpretation="Lower area is better.",
                tolerance=0.0,
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=(
                "objective:total_levels",
                "objective:total_lut_count",
            ),
            description="Minimize both mapped circuit objectives.",
            equivalence="Both objective values match exactly.",
            policy_id="objective_pareto",
            policy_version=1,
            definition_sha256=_sha("boils-like-objective-relation"),
        ),
    )


def test_objective_only_projection_preserves_legacy_ids_slate_and_direction_bytes() -> (
    None
):
    semantics = _objective_only_semantics()
    projection = DecisionMetricProjection.from_optimization_semantics(semantics)
    assert projection.objective_only_legacy_metric_ids is True
    assert projection.metric_ids == ("total_levels", "total_lut_count")

    legacy_slate = equal_weight_slate_objectives(
        (
            ObjectiveSpec("total_levels", "min"),
            ObjectiveSpec("total_lut_count", "min"),
        )
    )
    projected_slate = equal_weight_slate_objectives_from_decision_metrics(projection)
    semantic_slate = equal_weight_slate_objectives_from_optimization_semantics(
        semantics
    )
    assert projected_slate == legacy_slate == semantic_slate
    assert tuple(value.to_record() for value in projected_slate) == tuple(
        value.to_record() for value in legacy_slate
    )

    expected = AbsoluteToleranceDirectionAdjudicator(
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
        resolutions=(
            MetricDirectionResolution("total_levels", 0.0),
            MetricDirectionResolution("total_lut_count", 0.0),
        ),
    )
    observed = AbsoluteToleranceDirectionAdjudicator.from_optimization_semantics(
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
        semantics=semantics,
    )
    assert observed.to_record() == expected.to_record()


class _AirfoilProjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    step: int


class _AirfoilProjectionProblem:
    candidate_model = _AirfoilProjectionConfig
    objectives = (ObjectiveSpec(OBJECTIVE_NAME, "min"),)

    @staticmethod
    def validate(configuration: object) -> bool:
        _AirfoilProjectionConfig.model_validate(configuration, strict=True)
        return True

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        parsed = _AirfoilProjectionConfig.model_validate(configuration, strict=True)
        return {OBJECTIVE_NAME: 1.0 - 0.1 * parsed.step}


class _AirfoilProjectionDetailedEvaluator:
    evaluator_identity = EvaluatorIdentity(
        evaluator_id="airfoil_projection_test",
        evaluator_version=1,
        evaluator_context_sha256=_sha("airfoil-projection-evaluator"),
    )

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        parsed = _AirfoilProjectionConfig.model_validate(configuration, strict=True)
        return DetailedEvaluationPayload(
            failure=None,
            objectives=((OBJECTIVE_NAME, 1.0 - 0.1 * parsed.step),),
            violations=((VIOLATION_NAME, 0.5 - 0.1 * parsed.step),),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


class _NoCandidateGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("finite portfolio children are engine materialized")

    async def reflect(self, request):
        del request
        raise AssertionError("this projection test does not reflect")


class _SingleAirfoilSelector:
    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        option = request.finite_variation_contract.options[0]
        decision = resolve_ranked_portfolio_decision(
            request,
            (
                PortfolioMemberDraft(
                    option_id=option.option_id,
                    supporting_card_keys=(),
                    effect_predictions=tuple(
                        MetricEffectPrediction(
                            metric_id,
                            MetricEffectDirection.DECREASE,
                        )
                        for metric_id in request.required_metric_ids
                    ),
                    design_rationale="Exercise the sealed Airfoil outcome projection.",
                ),
            ),
            policy_id="airfoil_projection_selector",
            policy_version=1,
            policy_definition_sha256=_sha("airfoil-projection-selector"),
        )
        return PortfolioSelectionResult(
            decision=decision,
            telemetry=AgenticCallTelemetry(
                requested_model="provider-free/projection-test",
                resolved_model="provider-free/projection-test",
                resolved_provider="provider-free",
                provider_response_id="projection-test-response",
                finish_reason="stop",
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=1,
            ),
        )


def _frozen_object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


async def _airfoil_wave():
    ids = DeterministicIdFactory("airfoil_decision_metric_projection")
    engine = AgenticEvolutionEngine(
        problem=_AirfoilProjectionProblem(),
        generator=_NoCandidateGenerator(),
        id_factory=ids,
        memory=None,
        seed=7,
        detailed_evaluator=_AirfoilProjectionDetailedEvaluator(),
        outcome_relation_binding=AIRFOIL_V7_ARCHIVE_RELATION,
        optimization_semantics=AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    )
    parent = await engine.register_seed({"step": 0}, label="airfoil_parent")
    child_configuration = _frozen_object({"step": 1})
    contract = FiniteVariationContract(
        catalog_id="airfoil_projection_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("airfoil-projection-catalog"),
        parent_configuration=parent.configuration,
        options=(
            FiniteVariationOption(
                option_id="trim.step.one",
                parent_configuration_sha256=parent.occurrence.configuration_hash,
                child_configuration=child_configuration,
                family="trim",
                description="Apply one sealed trim step.",
            ),
        ),
    )
    request = PortfolioSelectionRequest(
        call_id=ids.new_llm_call_id(),
        operation="select_airfoil_projection",
        instruction="Select the one sealed Airfoil projection action.",
        context=_frozen_object({"benchmark": "airfoil-v7-projection"}),
        finite_variation_contract=contract,
        cards=(
            PortfolioCard(
                card_key="card.airfoil",
                reference=InsightRef(InsightId("insight_airfoil_projection"), 1),
                content_sha256=_sha("airfoil-projection-card-content"),
                evidence_sha256=_sha("airfoil-projection-card-evidence"),
                prompt_payload=_frozen_object(
                    {"claim": "Exercise the sealed Airfoil metric projection."}
                ),
            ),
        ),
        portfolio_size=1,
        required_metric_ids=(
            f"objective:{OBJECTIVE_NAME}",
            f"violation:{VIOLATION_NAME}",
        ),
        require_supporting_cards=False,
    )
    wave = PortfolioVariationWaveRequest(
        selection_request=request,
        parent=parent,
        generation=1,
        label_prefix="airfoil_projection",
    )
    result = await PortfolioEvolution(
        engine=engine,
        selector=_SingleAirfoilSelector(),
        ids=ids,
    ).run(wave)
    return parent, result


def _scope() -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("airfoil-benchmark"),
        session_sha256=_sha("session"),
    )


def _predictions(
    *,
    parent_identity: str,
    option_id: str,
    option_identity: str,
    metric_ids: tuple[str, ...],
) -> tuple[ForecastPredictionReceipt, ...]:
    return tuple(
        ForecastPredictionReceipt(
            scope=_scope(),
            wave_index=1,
            selector_decision_sha256=_sha("pre-evaluation-proposal"),
            parent_candidate_identity_sha256=parent_identity,
            option_id=option_id,
            option_identity_sha256=option_identity,
            family="trim",
            metric_id=metric_id,
            asserted_direction=MetricEffectDirection.DECREASE,
            confidence=ForecastConfidenceBin.HIGH,
        )
        for metric_id in metric_ids
    )


def test_airfoil_semantics_project_objective_and_violation_into_feedback() -> None:
    parent, result = asyncio.run(_airfoil_wave())
    child = result.candidates[0]
    decision = result.selection_decision
    assert decision is not None
    selected = decision.members[0]
    projection = DecisionMetricProjection.from_optimization_semantics(
        AIRFOIL_V7_OPTIMIZATION_SEMANTICS
    )
    assert projection.objective_only_legacy_metric_ids is False
    assert projection.metric_ids == (
        f"objective:{OBJECTIVE_NAME}",
        f"violation:{VIOLATION_NAME}",
    )
    assert project_candidate_decision_metrics(parent, projection).metric_map == {
        f"objective:{OBJECTIVE_NAME}": 1.0,
        f"violation:{VIOLATION_NAME}": 0.5,
    }
    assert project_candidate_decision_metrics(child, projection).metric_map == {
        f"objective:{OBJECTIVE_NAME}": 0.9,
        f"violation:{VIOLATION_NAME}": 0.4,
    }
    predictions = _predictions(
        parent_identity=parent.occurrence.configuration_hash,
        option_id=selected.option_id,
        option_identity=selected.option_identity_sha256,
        metric_ids=projection.metric_ids,
    )
    adjudicator = AbsoluteToleranceDirectionAdjudicator.from_optimization_semantics(
        benchmark_sha256=_scope().benchmark_sha256,
        session_sha256=_scope().session_sha256,
        semantics=AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    )
    with pytest.raises(ValueError, match="evaluated decision metrics"):
        observe_selected_portfolio_forecasts(
            wave_index=1,
            parent=parent,
            result=result,
            selected_predictions=predictions,
            adjudicator=adjudicator,
        )
    feedback = observe_selected_portfolio_forecasts(
        wave_index=1,
        parent=parent,
        result=result,
        selected_predictions=predictions,
        adjudicator=adjudicator,
        decision_metric_projection=projection,
    )
    observations = feedback.actions[0].observations
    assert tuple(value.prediction.metric_id for value in observations) == (
        f"objective:{OBJECTIVE_NAME}",
        f"violation:{VIOLATION_NAME}",
    )
    assert all(
        value.adjudication.actual_direction is MetricEffectDirection.DECREASE
        and value.correctness is True
        for value in observations
    )


def test_objective_only_feedback_is_byte_identical_with_or_without_projection() -> None:
    parent, result = asyncio.run(_airfoil_wave())
    decision = result.selection_decision
    assert decision is not None
    selected = decision.members[0]
    semantics = OptimizationSemantics(
        semantics_id="airfoil_objective_only_compatibility",
        semantics_version=1,
        metrics=(AIRFOIL_V7_OPTIMIZATION_SEMANTICS.metrics[0],),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.SCALAR,
            metric_priority=(f"objective:{OBJECTIVE_NAME}",),
            description="Objective-only compatibility projection.",
            equivalence="Exact objective equality.",
            policy_id="objective_only_compatibility",
            policy_version=1,
            definition_sha256=_sha("objective-only-compatibility"),
        ),
    )
    projection = DecisionMetricProjection.from_optimization_semantics(semantics)
    predictions = _predictions(
        parent_identity=parent.occurrence.configuration_hash,
        option_id=selected.option_id,
        option_identity=selected.option_identity_sha256,
        metric_ids=(OBJECTIVE_NAME,),
    )
    adjudicator = AbsoluteToleranceDirectionAdjudicator.from_optimization_semantics(
        benchmark_sha256=_scope().benchmark_sha256,
        session_sha256=_scope().session_sha256,
        semantics=semantics,
    )
    legacy = observe_selected_portfolio_forecasts(
        wave_index=1,
        parent=parent,
        result=result,
        selected_predictions=predictions,
        adjudicator=adjudicator,
    )
    projected = observe_selected_portfolio_forecasts(
        wave_index=1,
        parent=parent,
        result=result,
        selected_predictions=predictions,
        adjudicator=adjudicator,
        decision_metric_projection=projection,
    )
    assert projected.to_record() == legacy.to_record()
    assert projected.receipt_sha256 == legacy.receipt_sha256


def test_constraint_metrics_share_sealed_detailed_projection_and_fail_closed_sense() -> (
    None
):
    parent, _ = asyncio.run(_airfoil_wave())
    constraint = MetricSemantics(
        metric_id=f"constraint:{VIOLATION_NAME}",
        name=VIOLATION_NAME,
        role=MetricRole.CONSTRAINT,
        sense=MetricSense.MINIMIZE,
        definition="Normalized residual treated as a decision constraint metric.",
        aggregation="Sum of normalized absolute residuals.",
        witness_interpretation="Lower residual is better.",
        tolerance=0.0,
    )
    semantics = OptimizationSemantics(
        semantics_id="airfoil_constraint_projection",
        semantics_version=1,
        metrics=(AIRFOIL_V7_OPTIMIZATION_SEMANTICS.metrics[0], constraint),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.LEXICOGRAPHIC,
            metric_priority=(
                f"constraint:{VIOLATION_NAME}",
                f"objective:{OBJECTIVE_NAME}",
            ),
            description="Minimize the residual before drag.",
            equivalence="Both values match exactly.",
            policy_id="constraint_then_objective",
            policy_version=1,
            definition_sha256=_sha("constraint-then-objective"),
        ),
    )
    projection = DecisionMetricProjection.from_optimization_semantics(semantics)
    assert (
        project_candidate_decision_metrics(parent, projection).metric_map[
            f"constraint:{VIOLATION_NAME}"
        ]
        == 0.5
    )
    assert tuple(
        value.goal.value
        for value in equal_weight_slate_objectives_from_decision_metrics(projection)
    ) == ("minimize", "minimize")

    target_constraint = MetricSemantics(
        metric_id="constraint:setpoint",
        name="setpoint",
        role=MetricRole.CONSTRAINT,
        sense=MetricSense.TARGET,
        definition="Raw target measurement.",
        aggregation="One scalar.",
        witness_interpretation="Move toward the target.",
        reference_target=1.0,
        tolerance=0.01,
    )
    unsupported = OptimizationSemantics(
        semantics_id="target_constraint_projection",
        semantics_version=1,
        metrics=(target_constraint,),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.CUSTOM,
            metric_priority=("constraint:setpoint",),
            description="Custom target relation.",
            equivalence="Custom target equivalence.",
            policy_id="target_constraint_relation",
            policy_version=1,
            definition_sha256=_sha("target-constraint-relation"),
        ),
    )
    with pytest.raises(ValueError, match="no meaningful min/max"):
        equal_weight_slate_objectives_from_optimization_semantics(unsupported)
