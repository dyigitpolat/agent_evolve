from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest

from agent_evolve.application.action_allocation import (
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightOrigin,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    bind_portfolio_experimental_view,
    portfolio_card_from_insight_entry,
)
from agent_evolve.application.two_stage_action_evolution import (
    ActionEvaluationReuseMode,
    ActionEvaluationReusePolicyBinding,
    ActionForecastArmPlan,
    DurablePhaseCommitRequirement,
    FiniteActionEvaluatorBinding,
    PreparedTwoStageActionEvolution,
    PreparedTwoStageActionEvolutionRequest,
    SCIENTIFIC_ARM_ORDER,
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    TwoStageActionPhaseCommitError,
    required_scientific_phase_commit_policy,
)
from agent_evolve.core.action_semantics import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    bind_finite_action_evidence,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
    RunId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
)
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastRequest,
    ActionForecastResult,
    ActionMetricForecast,
    MetricForecastScale,
    ParentMetricValue,
    resolve_action_forecasts,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    derive_portfolio_card_view,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _contract() -> FiniteVariationContract:
    parent = _frozen({"coordinate": 0})
    parent_sha256 = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="two_stage_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("two-stage-fixture-catalog-v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen({"coordinate": coordinate}),
                family=family,
                description=f"Use sealed fixture coordinate {coordinate}.",
            )
            for option_id, family, coordinate in (
                ("action.a", "alpha", 1),
                ("action.b", "beta", 2),
                ("action.c", "gamma", 3),
                ("action.d", "delta", 4),
            )
        ),
    )


def _semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="two_stage_fixture_semantics",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Deterministic fixture cost.",
                aggregation="One scalar.",
                witness_interpretation="Lower is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:cost",),
            description="Minimize fixture cost.",
            equivalence="Equal costs are equivalent.",
            policy_id="two_stage_fixture_order",
            policy_version=1,
            definition_sha256=_sha("two-stage-fixture-order-v1"),
        ),
    )


def _action_semantics(contract: FiniteVariationContract) -> ActionSpaceSemantics:
    return ActionSpaceSemantics(
        semantics_id="two_stage_fixture_action_space",
        semantics_version=1,
        catalog_identities=(
            (
                contract.catalog_id,
                contract.catalog_version,
                contract.catalog_definition_sha256,
            ),
        ),
        axes=(
            ActionAxisSemantics(
                axis_id="fixture_coordinate",
                configuration_paths=("$.coordinate",),
                option_families=("alpha", "beta", "delta", "gamma"),
                definition="One scalar fixture coordinate selects the child.",
                independence=(
                    "Each sealed action is a mutually exclusive replacement of "
                    "the same coordinate."
                ),
                excluded_interpretations=(
                    "Family names do not imply metric quality.",
                ),
            ),
        ),
    )


def _entry(contract: FiniteVariationContract, index: int) -> InsightMemoryEntry:
    option = contract.options[index]
    contrast_id = _sha(f"two-stage-contrast-{index}")
    binding = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id=option.option_id,
    )
    draft = InsightDraft(
        claim=f"Generic fixture mechanism {index}.",
        trigger="The local response pattern is present.",
        mechanism="Test the cited finite perturbation against the parent.",
        affected_paths=("$.coordinate",),
        evidence_summary="One exact contrast supports this mechanism.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=(
            MetricEffectPrediction(
                "objective:cost",
                MetricEffectDirection.DECREASE,
            ),
        ),
        recommended_option_families=(option.family,),
        recommended_option_ids=(option.option_id,),
        action_template="Choose a sealed action with the same mechanism.",
        falsification_condition="Reject when cost does not decrease.",
    )
    return InsightMemoryEntry(
        reference=InsightRef(InsightId(f"insight_two_stage_{index}"), 1),
        draft=draft,
        initial_score=float(index),
        lifecycle_state=InsightLifecycleState.QUARANTINED,
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=InsightEvidenceLineage(
            reflection_call_id=LLMCallId("call_two_stage_reflection"),
            source_operator_invocation_ids=(
                OperatorInvocationId(f"operator_two_stage_{index}"),
            ),
            source_candidate_ids=(CandidateId(f"candidate_two_stage_{index}"),),
            available_contrast_ids=(contrast_id,),
            cited_contrast_ids=(contrast_id,),
            finite_action_bindings=(binding,),
        ),
    )


def _arm_requests() -> tuple[ActionForecastRequest, ...]:
    contract = _contract()
    entries = tuple(_entry(contract, index) for index in range(2))
    cards = tuple(
        portfolio_card_from_insight_entry(
            entry,
            card_key=f"card.{index}",
            prompt_payload=_frozen(
                {"mechanism": f"generic local response pattern {index}"}
            ),
            evidence_sha256=_sha(f"two-stage-evidence-{index}"),
            source_receipt_sha256=_sha(f"two-stage-source-receipt-{index}"),
        )
        for index, entry in enumerate(entries)
    )
    registry = admit_portfolio_card_sources(entries, cards)
    m_receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.MEMORY,
        cards=cards,
        finite_variation_contract=contract,
        source_registry=registry,
        policy_id="two_stage_memory_view",
        policy_version=1,
        policy_definition_sha256=_sha("two-stage-memory-view-v1"),
    )
    common = dict(
        operation="forecast_all_actions",
        instruction="Forecast every sealed action and the required metric.",
        context=_frozen({"benchmark": "generic_fixture", "stage": "g2"}),
        optimization_semantics=_semantics(),
        action_semantics=_action_semantics(contract),
        finite_variation_contract=contract,
        parent_metric_values=(ParentMetricValue("objective:cost", 10.0),),
        metric_scales=(
            MetricForecastScale(
                "objective:cost",
                5.0,
                _sha("two-stage-cost-scale-v1"),
            ),
        ),
        temperature=0.0,
    )
    memory = ActionForecastRequest(
        call_id=LLMCallId("call_two_stage_m"),
        cards=cards,
        source_registry=registry,
        evidence_mode=ActionForecastEvidenceMode.GROUNDED,
        experimental_view_receipt=m_receipt,
        **common,
    )
    transforms = tuple(
        sorted(
            (
                PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
                PortfolioCardViewTransform.PROMPT_PERMUTATION,
                PortfolioCardViewTransform.SCORE_PERMUTATION,
            ),
            key=lambda value: value.value,
        )
    )
    placebo_cards = tuple(
        derive_portfolio_card_view(
            source,
            prompt_payload=donor.prompt_payload,
            evidence_sha256=donor.evidence_sha256,
            score_components=donor.score_components,
            assigned_score=donor.assigned_score,
            transforms=transforms,
            policy_id="two_stage_placebo_view",
            policy_version=1,
            policy_definition_sha256=_sha("two-stage-placebo-view-v1"),
            prompt_source_card=donor,
            evidence_source_card=donor,
            score_source_card=donor,
        )
        for source, donor in ((cards[0], cards[1]), (cards[1], cards[0]))
    )
    p_receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
        cards=placebo_cards,
        finite_variation_contract=contract,
        source_registry=registry,
        policy_id="two_stage_placebo_population",
        policy_version=1,
        policy_definition_sha256=_sha("two-stage-placebo-population-v1"),
    )
    placebo = ActionForecastRequest(
        call_id=LLMCallId("call_two_stage_p"),
        cards=placebo_cards,
        source_registry=registry,
        evidence_mode=ActionForecastEvidenceMode.GROUNDED,
        experimental_view_receipt=p_receipt,
        **common,
    )
    neutral = ActionForecastRequest(
        call_id=LLMCallId("call_two_stage_n"),
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
        **common,
    )
    return memory, placebo, neutral


_DELTAS = {
    PortfolioExperimentalArm.MEMORY: {
        "action.a": 0.0,
        "action.b": -5.0,
        "action.c": -2.0,
        "action.d": -1.0,
    },
    PortfolioExperimentalArm.PERMUTED_PLACEBO: {
        "action.a": 0.0,
        "action.b": -1.0,
        "action.c": -6.0,
        "action.d": -2.0,
    },
    PortfolioExperimentalArm.NEUTRAL: {
        "action.a": 0.0,
        "action.b": -1.0,
        "action.c": -4.0,
        "action.d": -2.0,
    },
}


class _ConcurrentForecaster:
    def __init__(self) -> None:
        self.started = 0
        self.active = 0
        self.maximum_active = 0
        self._release = asyncio.Event()

    @staticmethod
    def _arm(request: ActionForecastRequest) -> PortfolioExperimentalArm:
        receipt = request.experimental_view_receipt
        return PortfolioExperimentalArm.NEUTRAL if receipt is None else receipt.arm

    async def forecast(self, request: ActionForecastRequest) -> ActionForecastResult:
        self.started += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.started == 3:
            self._release.set()
        await self._release.wait()
        arm = self._arm(request)
        citation = ()
        if request.cards:
            binding = request.cards[0].finite_action_evidence[0]
            citation = (
                ActionEvidenceCitation(
                    request.cards[0].card_key,
                    binding.identity_sha256,
                ),
            )
        drafts = tuple(
            ActionForecastDraft(
                option_id=option.option_id,
                probability_valid=1.0,
                metric_forecasts=(
                    ActionMetricForecast(
                        "objective:cost",
                        _DELTAS[arm][option.option_id],
                        _DELTAS[arm][option.option_id],
                        _DELTAS[arm][option.option_id],
                        0.8,
                        citation,
                    ),
                ),
            )
            for option in request.finite_variation_contract.options
        )
        result = ActionForecastResult(
            resolve_action_forecasts(
                request,
                drafts,
                policy_id="two_stage_fixture_forecaster",
                policy_version=1,
                policy_definition_sha256=_sha("two-stage-fixture-forecaster-v1"),
            ),
            None,
        )
        self.active -= 1
        return result


class _ConcurrentEvaluator:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started = 0
        self.active = 0
        self.maximum_active = 0
        self.option_ids: list[str] = []
        self._release = asyncio.Event()

    async def evaluate(self, request):
        self.started += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.option_ids.append(request.option.option_id)
        if self.started == self.expected:
            self._release.set()
        await self._release.wait()
        result = _frozen(
            {
                "cost": float(int(request.option.child_configuration_sha256[0:2], 16)),
                "valid": True,
            }
        )
        self.active -= 1
        return result


def _utility(value: ForecastPortfolioUtilityInput) -> float:
    total = 0.0
    for member in value.members:
        forecast = member.metric_forecasts[0]
        delta = {
            "p10": forecast.p10_delta,
            "p50": forecast.p50_delta,
            "p90": forecast.p90_delta,
        }[value.quantile.value]
        total -= value.parent_metric_values[0].value + delta
    return float(total)


def _run_request(evaluator: _ConcurrentEvaluator) -> PreparedTwoStageActionEvolutionRequest:
    requests = _arm_requests()
    return PreparedTwoStageActionEvolutionRequest(
        run_id=RunId("run_two_stage_fixture"),
        arm_plans=tuple(
            ActionForecastArmPlan(arm, request)
            for arm, request in zip(SCIENTIFIC_ARM_ORDER, requests, strict=True)
        ),
        g1_option_ids=("action.a",),
        portfolio_size=1,
        utility=ForecastPortfolioUtilityBinding(
            utility=_utility,
            policy_id="two_stage_fixture_utility",
            policy_version=1,
            definition_sha256=_sha("two-stage-fixture-utility-v1"),
        ),
        evaluator=FiniteActionEvaluatorBinding(
            evaluator=evaluator,
            evaluator_id="two_stage_fixture_evaluator",
            evaluator_version=1,
            definition_sha256=_sha("two-stage-fixture-evaluator-v1"),
        ),
        evaluation_context=_frozen({"budget": "fixture", "prospective": True}),
    )


def test_prepared_two_stage_run_is_concurrent_g1_excluding_and_per_arm_by_default() -> None:
    forecaster = _ConcurrentForecaster()
    evaluator = _ConcurrentEvaluator(expected=3)
    request = _run_request(evaluator)
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=forecaster,
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )

    result = asyncio.run(coordinator.run(request))

    assert forecaster.maximum_active == 3
    assert evaluator.maximum_active == 3
    assert evaluator.option_ids == ["action.b", "action.c", "action.c"]
    assert tuple(value.arm for value in result.forecasts) == SCIENTIFIC_ARM_ORDER
    assert tuple(value.arm for value in result.allocations) == SCIENTIFIC_ARM_ORDER
    assert [value.result.decision.members[0].option_id for value in result.allocations] == [
        "action.b",
        "action.c",
        "action.c",
    ]
    assert all(
        "action.a" not in {
            member.option_id for member in allocation.result.decision.members
        }
        for allocation in result.allocations
    )
    assert [value.request.selected_by_arms for value in result.evaluations] == [
        (PortfolioExperimentalArm.MEMORY,),
        (PortfolioExperimentalArm.PERMUTED_PLACEBO,),
        (PortfolioExperimentalArm.NEUTRAL,),
    ]
    assert result.evaluation_reuse.mode is ActionEvaluationReuseMode.PER_ARM
    assert tuple(value.phase for value in result.phase_receipts) == tuple(
        TwoStageActionPhase
    )
    assert len(result.receipt_sha256) == 64
    assert result.to_record()["request_sha256"] == request.request_sha256


def test_explicit_unique_action_policy_reuses_cross_arm_evaluation_once() -> None:
    forecaster = _ConcurrentForecaster()
    evaluator = _ConcurrentEvaluator(expected=2)
    request = replace(
        _run_request(evaluator),
        evaluation_reuse=ActionEvaluationReusePolicyBinding(
            ActionEvaluationReuseMode.UNIQUE_ACTION
        ),
    )
    result = asyncio.run(
        PreparedTwoStageActionEvolution(
            forecaster=forecaster,
            allocator=GreedyRiskAdjustedDiversityAllocator(
                risk_aversion=0.0,
                diversity_weight=0.0,
            ),
        ).run(request)
    )

    assert evaluator.maximum_active == 2
    assert evaluator.option_ids == ["action.b", "action.c"]
    assert len(result.evaluations) == 2
    assert result.evaluations[1].request.selected_by_arms == (
        PortfolioExperimentalArm.PERMUTED_PLACEBO,
        PortfolioExperimentalArm.NEUTRAL,
    )
    assert result.evaluation_reuse.to_record() == request.evaluation_reuse.to_record()


def test_prepared_request_rejects_noncomparable_arm_prompt_or_g1_leakage() -> None:
    evaluator = _ConcurrentEvaluator(expected=1)
    request = _run_request(evaluator)
    altered_n = replace(
        request.arm_plans[2].request,
        instruction="Use a different neutral-arm instruction.",
    )
    with pytest.raises(ValueError, match="may differ only"):
        replace(
            request,
            arm_plans=(
                request.arm_plans[0],
                request.arm_plans[1],
                ActionForecastArmPlan(PortfolioExperimentalArm.NEUTRAL, altered_n),
            ),
        )
    with pytest.raises(ValueError, match="non-G1"):
        replace(request, g1_option_ids=("action.a", "action.b", "action.c"), portfolio_size=2)


def test_evaluator_must_return_frozen_typed_json() -> None:
    class _BadEvaluator:
        async def evaluate(self, request):
            del request
            return {"not": "frozen"}

    forecaster = _ConcurrentForecaster()
    request = _run_request(_ConcurrentEvaluator(expected=2))
    request = replace(
        request,
        evaluator=FiniteActionEvaluatorBinding(
            evaluator=_BadEvaluator(),
            evaluator_id="bad_fixture_evaluator",
            evaluator_version=1,
            definition_sha256=_sha("bad-fixture-evaluator-v1"),
        ),
    )
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=forecaster,
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )
    with pytest.raises(TypeError, match="non-FrozenJsonObject"):
        asyncio.run(coordinator.run(request))


class _ImmediateEvaluator:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.capability_open = False
        self.require_open_capability = False

    async def evaluate(self, request):
        if self.require_open_capability and not self.capability_open:
            raise AssertionError("evaluation started before allocation durability")
        self.calls.append(request.option.option_id)
        return _frozen({"cost": 1.0, "valid": True})


def _required_request(evaluator) -> PreparedTwoStageActionEvolutionRequest:
    return replace(
        _run_request(evaluator),
        phase_commit_policy=required_scientific_phase_commit_policy(),
    )


def test_required_phase_sink_absence_fails_before_forecast_or_evaluation() -> None:
    forecaster = _ConcurrentForecaster()
    evaluator = _ImmediateEvaluator()
    request = _required_request(evaluator)
    assert (
        request.phase_commit_policy.requirement
        is DurablePhaseCommitRequirement.REQUIRED
    )
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=forecaster,
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )

    with pytest.raises(TwoStageActionPhaseCommitError, match="sink is absent"):
        asyncio.run(coordinator.run(request))

    assert forecaster.started == 0
    assert evaluator.calls == []


def test_forecast_failure_settles_every_sibling_and_salvages_none() -> None:
    class _SettlingFailureForecaster:
        def __init__(self) -> None:
            self.started = 0
            self.settled: list[PortfolioExperimentalArm] = []
            self._release = asyncio.Event()

        async def forecast(self, request: ActionForecastRequest):
            arm = _ConcurrentForecaster._arm(request)
            self.started += 1
            if self.started == 3:
                self._release.set()
            await self._release.wait()
            if arm is PortfolioExperimentalArm.MEMORY:
                raise RuntimeError("memory forecast failed")
            await asyncio.sleep(0.01)
            self.settled.append(arm)
            raise RuntimeError(f"{arm.value} forecast failed")

    forecaster = _SettlingFailureForecaster()
    evaluator = _ImmediateEvaluator()
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=forecaster,
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="memory forecast failed"):
        asyncio.run(coordinator.run(_run_request(evaluator)))

    assert forecaster.started == 3
    assert set(forecaster.settled) == {
        PortfolioExperimentalArm.PERMUTED_PLACEBO,
        PortfolioExperimentalArm.NEUTRAL,
    }
    assert evaluator.calls == []


def test_evaluation_failure_settles_every_sibling_and_publishes_no_result() -> None:
    class _SettlingFailureEvaluator:
        def __init__(self) -> None:
            self.started = 0
            self.settled: list[PortfolioExperimentalArm] = []
            self._release = asyncio.Event()

        async def evaluate(self, request):
            arm = request.selected_by_arms[0]
            self.started += 1
            if self.started == 3:
                self._release.set()
            await self._release.wait()
            if arm is PortfolioExperimentalArm.MEMORY:
                raise RuntimeError("memory evaluation failed")
            await asyncio.sleep(0.01)
            self.settled.append(arm)
            return _frozen({"cost": 1.0, "valid": True})

    class _RecordingSink:
        def __init__(self) -> None:
            self.phases: list[TwoStageActionPhase] = []

        def commit(self, phase_commit: TwoStageActionPhaseCommit) -> None:
            self.phases.append(phase_commit.receipt.phase)

    evaluator = _SettlingFailureEvaluator()
    sink = _RecordingSink()
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=_ConcurrentForecaster(),
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="memory evaluation failed"):
        asyncio.run(
            coordinator.run(
                _required_request(evaluator),
                phase_commit_sink=sink,
            )
        )

    assert evaluator.started == 3
    assert set(evaluator.settled) == {
        PortfolioExperimentalArm.PERMUTED_PLACEBO,
        PortfolioExperimentalArm.NEUTRAL,
    }
    assert sink.phases == [
        TwoStageActionPhase.FORECAST,
        TwoStageActionPhase.ALLOCATE,
    ]


def test_external_cancellation_cancels_and_settles_every_forecast_sibling() -> None:
    async def scenario() -> None:
        class _CancellationForecaster:
            def __init__(self) -> None:
                self.started = 0
                self.started_all = asyncio.Event()
                self.never = asyncio.Event()
                self.settled: list[PortfolioExperimentalArm] = []

            async def forecast(self, request: ActionForecastRequest):
                arm = _ConcurrentForecaster._arm(request)
                self.started += 1
                if self.started == 3:
                    self.started_all.set()
                try:
                    await self.never.wait()
                finally:
                    self.settled.append(arm)

        forecaster = _CancellationForecaster()
        evaluator = _ImmediateEvaluator()
        task = asyncio.create_task(
            PreparedTwoStageActionEvolution(
                forecaster=forecaster,
                allocator=GreedyRiskAdjustedDiversityAllocator(
                    risk_aversion=0.0,
                    diversity_weight=0.0,
                ),
            ).run(_run_request(evaluator))
        )
        await forecaster.started_all.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(forecaster.settled) == set(SCIENTIFIC_ARM_ORDER)
        assert evaluator.calls == []

    asyncio.run(scenario())


def test_async_allocation_commit_failure_causes_zero_evaluations() -> None:
    class _FailingSink:
        def __init__(self) -> None:
            self.phases: list[TwoStageActionPhase] = []

        async def commit(self, phase_commit: TwoStageActionPhaseCommit) -> None:
            assert typed_json_sha256(phase_commit.payload) == (
                phase_commit.receipt.output_sha256
            )
            self.phases.append(phase_commit.receipt.phase)
            await asyncio.sleep(0)
            if phase_commit.receipt.phase is TwoStageActionPhase.ALLOCATE:
                raise OSError("simulated fsync failure")

    forecaster = _ConcurrentForecaster()
    evaluator = _ImmediateEvaluator()
    sink = _FailingSink()
    coordinator = PreparedTwoStageActionEvolution(
        forecaster=forecaster,
        allocator=GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=0.0,
        ),
    )

    with pytest.raises(TwoStageActionPhaseCommitError, match="allocate phase"):
        asyncio.run(
            coordinator.run(
                _required_request(evaluator),
                phase_commit_sink=sink,
            )
        )

    assert sink.phases == [
        TwoStageActionPhase.FORECAST,
        TwoStageActionPhase.ALLOCATE,
    ]
    assert evaluator.calls == []


def test_sync_sink_opens_evaluator_only_after_durable_allocation_decision() -> None:
    evaluator = _ImmediateEvaluator()
    evaluator.require_open_capability = True

    class _CapabilityOpeningSink:
        def __init__(self) -> None:
            self.commits: list[TwoStageActionPhaseCommit] = []

        def commit(self, phase_commit: TwoStageActionPhaseCommit) -> None:
            phase_commit.__post_init__()
            self.commits.append(phase_commit)
            if phase_commit.receipt.phase is TwoStageActionPhase.ALLOCATE:
                payload = thaw_json(phase_commit.payload)
                assert type(payload) is dict
                executions = payload["arm_executions"]
                assert type(executions) is list and len(executions) == 3
                assert [value["arm"] for value in executions] == ["m", "p", "n"]
                assert all("decision" in value for value in executions)
                evaluator.capability_open = True

    sink = _CapabilityOpeningSink()
    request = _required_request(evaluator)
    result = asyncio.run(
        PreparedTwoStageActionEvolution(
            forecaster=_ConcurrentForecaster(),
            allocator=GreedyRiskAdjustedDiversityAllocator(
                risk_aversion=0.0,
                diversity_weight=0.0,
            ),
        ).run(request, phase_commit_sink=sink)
    )

    assert [value.receipt.phase for value in sink.commits] == list(
        TwoStageActionPhase
    )
    assert evaluator.calls == ["action.b", "action.c", "action.c"]
    assert result.phase_commit_policy == request.phase_commit_policy
    assert [
        value.receipt.receipt_sha256 for value in sink.commits
    ] == [value.receipt_sha256 for value in result.phase_receipts]
    assert all(
        typed_json_sha256(value.payload) == value.receipt.output_sha256
        for value in sink.commits
    )


def test_forecast_commit_contains_full_batches_citations_and_telemetry_before_allocation() -> None:
    class _TelemetryForecaster(_ConcurrentForecaster):
        async def forecast(self, request: ActionForecastRequest) -> ActionForecastResult:
            result = await super().forecast(request)
            arm = self._arm(request)
            return replace(
                result,
                telemetry=AgenticCallTelemetry(
                    requested_model="deepseek/deepseek-v4-pro",
                    resolved_model="deepseek/deepseek-v4-pro",
                    resolved_provider="fixture_provider",
                    provider_response_id=f"response_{arm.value}",
                    finish_reason="tool_call",
                    input_tokens=101,
                    output_tokens=202,
                    reasoning_tokens=303,
                    cache_read_tokens=11,
                    cache_write_tokens=12,
                    cost_usd=Decimal("0.00012300"),
                    latency_ns=4_000_005,
                    attempt_count=2,
                ),
            )

    class _ForecastTraceSink:
        def __init__(self) -> None:
            self.forecast_committed = False
            self.forecast_payload: dict[str, object] | None = None

        def commit(self, phase_commit: TwoStageActionPhaseCommit) -> None:
            if phase_commit.receipt.phase is not TwoStageActionPhase.FORECAST:
                return
            payload = thaw_json(phase_commit.payload)
            assert type(payload) is dict
            assert payload["schema_version"] == 2
            executions = payload["arm_executions"]
            assert type(executions) is list and len(executions) == 3
            for index, execution in enumerate(executions):
                assert execution["arm"] == ("m", "p", "n")[index]
                batch = execution["resolved_action_forecast_batch"]
                assert len(batch["forecasts"]) == 4
                assert "receipt_sha256" in batch
                first_metric = batch["forecasts"][0]["metric_forecasts"][0]
                if execution["arm"] == "n":
                    assert first_metric["citations"] == []
                else:
                    assert len(first_metric["citations"]) == 1
                    assert "source_option_identity_sha256" in (
                        first_metric["citations"][0]
                    )
                telemetry = execution["telemetry"]
                assert telemetry == {
                    "requested_model": "deepseek/deepseek-v4-pro",
                    "resolved_model": "deepseek/deepseek-v4-pro",
                    "resolved_provider": "fixture_provider",
                    "provider_response_id": f"response_{execution['arm']}",
                    "finish_reason": "tool_call",
                    "input_tokens": 101,
                    "output_tokens": 202,
                    "reasoning_tokens": 303,
                    "cache_read_tokens": 11,
                    "cache_write_tokens": 12,
                    "cost_usd": "0.00012300",
                    "latency_ns": 4_000_005,
                    "attempt_count": 2,
                }
            self.forecast_payload = payload
            self.forecast_committed = True

    sink = _ForecastTraceSink()
    delegate_allocator = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=0.0,
    )

    class _CommitGatedAllocator:
        def __init__(self) -> None:
            self.calls = 0

        def allocate(self, request):
            assert sink.forecast_committed
            self.calls += 1
            return delegate_allocator.allocate(request)

    allocator = _CommitGatedAllocator()
    evaluator = _ImmediateEvaluator()
    result = asyncio.run(
        PreparedTwoStageActionEvolution(
            forecaster=_TelemetryForecaster(),
            allocator=allocator,
        ).run(
            _required_request(evaluator),
            phase_commit_sink=sink,
        )
    )

    assert allocator.calls == 3
    assert sink.forecast_payload is not None
    committed = sink.forecast_payload["arm_executions"]
    assert [value["resolved_action_forecast_batch"] for value in committed] == [
        value.result.forecasts.to_record() for value in result.forecasts
    ]
