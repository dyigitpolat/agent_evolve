from __future__ import annotations

import hashlib
import json

from agent_evolve.application.action_target_realization import TargetMetricAlias
from agent_evolve.application.empirical_consequence_calibration import (
    EMPIRICAL_CONSEQUENCE_POLICY_ID,
    HierarchicalEmpiricalConsequenceCalibrationPolicy,
)
from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.portfolio_outcome_feedback import (
    DecisionMetricTransition,
    PortfolioActionOutcomeFeedback,
    PortfolioOutcomeFeedbackLedger,
    PortfolioOutcomeFeedbackReceipt,
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
)
from agent_evolve.domain.ids import ArtifactId, LLMCallId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.artifacts import InMemoryArtifactStore
from agent_evolve.ports.artifact_store import read_json
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
    MeaningfulDirectionRequest,
    observe_forecast,
)
from agent_evolve.policies.selection.meaningful_direction import (
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastRequest,
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionMetricForecast,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _scope() -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )


def _contract() -> FiniteVariationContract:
    parent = _frozen({"x": 0, "y": 0})
    parent_sha = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="empirical_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("catalog"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="action.a",
                parent_configuration_sha256=parent_sha,
                child_configuration=_frozen({"x": 1, "y": 0}),
                family="geometry",
                description="Set fixture x coordinate to one.",
            ),
            FiniteVariationOption(
                option_id="action.b",
                parent_configuration_sha256=parent_sha,
                child_configuration=_frozen({"x": 0, "y": 1}),
                family="geometry",
                description="Set fixture y coordinate to one.",
            ),
        ),
    )


def _semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="empirical_fixture",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Fixture scalar cost.",
                aggregation="One deterministic scalar.",
                witness_interpretation="Lower is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:cost",),
            description="Minimize fixture cost.",
            equivalence="Equal scalar costs are equivalent.",
            policy_id="fixture_pareto",
            policy_version=1,
            definition_sha256=_sha("ordering"),
        ),
    )


def _request() -> ActionForecastRequest:
    contract = _contract()
    return ActionForecastRequest(
        call_id=LLMCallId("call_empirical_fixture"),
        operation="forecast_all_actions",
        instruction="Forecast every sealed action.",
        context=_frozen({"fixture": "empirical"}),
        optimization_semantics=_semantics(),
        action_semantics=ActionSpaceSemantics(
            semantics_id="empirical_fixture_actions",
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
                    axis_id="coordinate",
                    configuration_paths=("$.x", "$.y"),
                    option_families=("geometry",),
                    definition="A sealed scalar coordinate replacement.",
                    independence="Exactly one coordinate is replaced.",
                    excluded_interpretations=(
                        "Option IDs do not reveal objective outcomes.",
                    ),
                ),
            ),
        ),
        finite_variation_contract=contract,
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
        parent_metric_values=(ParentMetricValue("objective:cost", 10.0),),
        metric_scales=(MetricForecastScale("objective:cost", 10.0, _sha("scale")),),
        temperature=0.0,
    )


def _batch(request: ActionForecastRequest) -> ResolvedActionForecastBatch:
    return ResolvedActionForecastBatch(
        request_sha256=request.request_sha256,
        context_sha256=request.context_sha256,
        optimization_semantics_definition_sha256=(
            request.optimization_semantics.definition_sha256
        ),
        action_semantics_definition_sha256=(request.action_semantics.definition_sha256),
        finite_contract_identity_sha256=(
            request.finite_variation_contract.identity_sha256
        ),
        card_snapshot_sha256=request.card_snapshot_sha256,
        forecasts=tuple(
            ResolvedActionForecast(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                family=option.family,
                probability_valid=0.9,
                metric_forecasts=(
                    ResolvedActionMetricForecast(
                        metric_id="objective:cost",
                        p10_delta=1.0,
                        p50_delta=2.0,
                        p90_delta=3.0,
                        confidence=0.8,
                        citations=(),
                    ),
                ),
            )
            for option in request.finite_variation_contract.options
        ),
        policy_id="fixture_model",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-model"),
    )


def _action(
    *,
    wave: int,
    option_id: str,
    parent_value: float,
    child_value: float,
    changed_path: str = "$.x",
    metric_id: str = "objective:cost",
) -> PortfolioActionOutcomeFeedback:
    scope = _scope()
    option_identity = _sha(f"historical:{wave}:{option_id}")
    proposal = _sha(f"proposal:{wave}:{option_id}")
    parent_outcome = _sha(f"parent:{wave}:{option_id}")
    child_outcome = _sha(f"child:{wave}:{option_id}")
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=wave,
        selector_decision_sha256=proposal,
        parent_candidate_identity_sha256=_sha(f"parent-identity:{wave}"),
        option_id=option_id,
        option_identity_sha256=option_identity,
        family="geometry",
        metric_id=metric_id,
        asserted_direction=MetricEffectDirection.INCREASE,
        confidence=ForecastConfidenceBin.HIGH,
    )
    direction_request = MeaningfulDirectionRequest(
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        wave_index=wave,
        parent_candidate_identity_sha256=prediction.parent_candidate_identity_sha256,
        option_id=option_id,
        option_identity_sha256=option_identity,
        metric_id=metric_id,
        parent_outcome_sha256=parent_outcome,
        child_outcome_sha256=child_outcome,
        parent_metric_value=parent_value,
        child_metric_value=child_value,
    )
    observation = observe_forecast(
        prediction,
        direction_request,
        AbsoluteToleranceDirectionAdjudicator(
            benchmark_sha256=scope.benchmark_sha256,
            session_sha256=scope.session_sha256,
            resolutions=(MetricDirectionResolution(metric_id, 0.0),),
        ),
    )
    return PortfolioActionOutcomeFeedback(
        wave_index=wave,
        request_sha256=_sha(f"request:{wave}:{option_id}"),
        ranked_decision_sha256=_sha(f"ranked:{wave}:{option_id}"),
        proposal_sha256=proposal,
        parent_candidate_id=f"candidate_parent_{wave}",
        parent_candidate_identity_sha256=(prediction.parent_candidate_identity_sha256),
        parent_outcome_sha256=parent_outcome,
        candidate_id=f"candidate_child_{wave}",
        candidate_outcome_sha256=child_outcome,
        option_id=option_id,
        option_identity_sha256=option_identity,
        family="geometry",
        changed_paths=(changed_path,),
        observations=(observation,),
        parent_relation=OutcomeRelation.BETTER,
        reward=1.0,
        dominates_parent=True,
        better_than_parent=True,
        metric_transitions=(
            DecisionMetricTransition(
                metric_id=metric_id,
                parent_value=parent_value,
                child_value=child_value,
                actual_direction=observation.adjudication.actual_direction,
                adjudication_receipt_sha256=(observation.adjudication.receipt_sha256),
            ),
        ),
    )


def _ledger(*actions: PortfolioActionOutcomeFeedback) -> PortfolioOutcomeFeedbackLedger:
    ledger = PortfolioOutcomeFeedbackLedger()
    for action in actions:
        ledger.append(
            PortfolioOutcomeFeedbackReceipt(
                wave_index=action.wave_index,
                request_sha256=action.request_sha256,
                ranked_decision_sha256=action.ranked_decision_sha256,
                scope=_scope(),
                actions=(action,),
            )
        )
    return ledger


def test_no_prior_evidence_preserves_model_forecast() -> None:
    request = _request()
    source = _batch(request)
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=PortfolioOutcomeFeedbackLedger(),
        scope=_scope(),
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=1,
    )

    assert result.forecasts.policy_id == EMPIRICAL_CONSEQUENCE_POLICY_ID
    assert result.forecasts.forecasts[0].metric_forecasts[0].p50_delta == 2.0
    assert result.forecasts.forecasts[0].probability_valid == 0.9
    # Regression: a live BOiLS wave reached this receipt only after all model
    # blocks completed, then failed because FrozenJsonObject was passed to the
    # standard JSON encoder.  Receipt hashing and publication must both cross
    # the plain-JSON boundary successfully.
    assert len(result.receipt_sha256) == 64
    json.dumps(result.to_record(), allow_nan=False, sort_keys=True)


def test_large_action_audit_is_externalized_behind_bounded_manifest() -> None:
    request = _request()
    source = _batch(request)
    store = InMemoryArtifactStore()
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=PortfolioOutcomeFeedbackLedger(),
        scope=_scope(),
        audit_artifact_store=store,
        maximum_embedded_action_audits=1,
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=1,
    )

    audit = thaw_json(result.audit)
    storage = audit["action_audit_storage"]
    assert storage["mode"] == "content_addressed_external"
    assert storage["action_count"] == 2
    assert "actions" not in audit
    assert len(storage["artifacts"]) == 2
    first = storage["artifacts"][0]
    payload = read_json(
        store,
        ArtifactId(first["artifact"]["artifact_id"]),
    )
    assert payload["action"]["option_id"] == first["option_id"]
    assert payload["scope_sha256"] == _scope().scope_sha256
    # The decision-facing result remains small enough to cross typed-JSON
    # evidence boundaries even when detailed action records live elsewhere.
    freeze_json(result.to_record())


def test_prior_family_outcomes_can_reverse_a_wrong_sign_model_forecast() -> None:
    request = _request()
    source = _batch(request)
    ledger = _ledger(
        _action(wave=1, option_id="history.a", parent_value=10.0, child_value=0.0),
        _action(wave=2, option_id="history.b", parent_value=11.0, child_value=1.0),
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=3,
    )

    metric = result.forecasts.forecasts[0].metric_forecasts[0]
    assert metric.p10_delta <= metric.p50_delta <= metric.p90_delta
    assert metric.p50_delta < 0.0
    assert result.forecasts.forecasts[0].probability_valid > 0.9


def test_explicit_metric_alias_joins_forecast_ids_to_outcome_ids() -> None:
    request = _request()
    source = _batch(request)
    ledger = _ledger(
        _action(
            wave=1,
            option_id="history.a",
            parent_value=10.0,
            child_value=0.0,
            metric_id="cost",
        ),
        _action(
            wave=2,
            option_id="history.b",
            parent_value=11.0,
            child_value=1.0,
            metric_id="cost",
        ),
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=3,
        metric_aliases=(
            TargetMetricAlias(
                target_metric_id="cost",
                forecast_metric_id="objective:cost",
            ),
        ),
    )

    metric = result.forecasts.forecasts[0].metric_forecasts[0]
    assert metric.metric_id == "objective:cost"
    assert metric.p50_delta < 0.0
    audit = thaw_json(result.audit)
    cell = audit["actions"][0]["metric_cells"][0]
    assert cell["forecast_metric_id"] == "objective:cost"
    assert cell["outcome_metric_id"] == "cost"
    assert cell["stratum"] == "exact_path_family"


def test_prequential_model_skill_inverts_anti_calibrated_signal_without_transfer() -> (
    None
):
    request = _request()
    source = _batch(request)
    ledger = _ledger(
        _action(wave=1, option_id="history.a", parent_value=10.0, child_value=0.0),
        _action(wave=2, option_id="history.b", parent_value=11.0, child_value=1.0),
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
        minimum_path_support=99,
        minimum_family_support=99,
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=3,
    )

    metric = result.forecasts.forecasts[0].metric_forecasts[0]
    assert (metric.p10_delta, metric.p50_delta, metric.p90_delta) == (
        -1.5,
        -1.0,
        -0.5,
    )
    audit = thaw_json(result.audit)
    model = audit["actions"][0]["metric_cells"][0]["model_calibration"]
    assert model["scorable_count"] == 2
    assert model["correct_count"] == 0
    assert model["negative_skill_inversion"] is True
    assert float.fromhex(model["signed_skill_hex"]) == -0.5


def test_empirical_expert_loses_authority_after_prior_wave_errors() -> None:
    request = _request()
    source = _batch(request)
    deltas = (10.0, 10.0, -10.0, -10.0, 10.0, 10.0)
    ledger = _ledger(
        *(
            _action(
                wave=wave,
                option_id=f"history.{wave}",
                parent_value=10.0,
                child_value=10.0 + delta,
            )
            for wave, delta in enumerate(deltas, start=1)
        )
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
        minimum_path_support=2,
        minimum_family_support=2,
        minimum_empirical_score_support=4,
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=7,
    )

    audit = thaw_json(result.audit)
    metric = audit["actions"][0]["metric_cells"][0]
    skill = metric["empirical_prequential_skill"]
    assert skill["scorable_count"] == 4
    assert skill["correct_count"] == 1
    assert skill["score_identified"] is True
    assert float.fromhex(skill["authority_multiplier_hex"]) == 0.0
    assert float.fromhex(metric["empirical_authority_hex"]) == 0.0


def test_current_and_future_wave_outcomes_are_excluded() -> None:
    request = _request()
    source = _batch(request)
    ledger = _ledger(
        _action(wave=1, option_id="history.a", parent_value=10.0, child_value=9.0),
        _action(wave=2, option_id="history.b", parent_value=10.0, child_value=0.0),
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
        minimum_family_support=2,
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=2,
    )

    assert result.forecasts.forecasts[0].metric_forecasts[0].p50_delta == 2.0


def test_exact_option_history_precedes_family_history() -> None:
    request = _request()
    source = _batch(request)
    ledger = _ledger(
        _action(wave=1, option_id="action.a", parent_value=10.0, child_value=0.0),
        _action(wave=2, option_id="action.a", parent_value=10.0, child_value=0.0),
        _action(
            wave=3,
            option_id="history.opposite",
            parent_value=10.0,
            child_value=20.0,
            changed_path="$.y",
        ),
        _action(
            wave=4,
            option_id="history.opposite2",
            parent_value=10.0,
            child_value=20.0,
            changed_path="$.y",
        ),
    )
    result = HierarchicalEmpiricalConsequenceCalibrationPolicy(
        ledger=ledger,
        scope=_scope(),
    ).calibrate(
        request=request,
        forecasts=source,
        cutoff_wave_index_exclusive=5,
    )

    first, second = result.forecasts.forecasts
    assert first.metric_forecasts[0].p50_delta < second.metric_forecasts[0].p50_delta
