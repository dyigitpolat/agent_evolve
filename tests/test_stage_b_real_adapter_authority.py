"""Real Airfoil/BOiLS provider-free transfer gate for Stage-B action sets."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from agent_evolve.agentic import (
    AgenticCallTelemetry,
    AgenticBenchmark,
    EngineFiniteActionRequest,
    FiniteActionSelectorKind,
    FiniteActionSourceMode,
    FiniteVariationSelectionDraft,
    InsightDraft,
    InsightMemoryBank,
    MetricEffectDirection,
    MetricEffectPrediction,
    OperatorKind,
    TaskKeyedUniformFiniteActionPolicy,
    validate_finite_action_decision,
)
from agent_evolve.application.finite_action_selection import (
    model_finite_action_telemetry_sha256,
    seal_model_finite_action_decision,
)
from agent_evolve.application.insight_memory import InsightOrigin
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.agentic_benchmark import (
    benchmark as boils_base_benchmark,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import FINITE_CATALOG_ID
from examples.benchmarks.boils_abc.stage_b_action_set import (
    BoilsPositionHypothesisCompiler,
    BoilsPositionLocalSupportCompiler,
)
from examples.benchmarks.boils_abc.variation_catalog import ACTION_FAMILIES
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    AirfoilV7TrimHypothesisCompiler,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    benchmark as airfoil_base_benchmark,
)
from examples.benchmarks.engibench_airfoil.v7_stage_b_action_set import (
    AirfoilTrimLocalSupportCompiler,
)


AIRFOIL_PARENT = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [0.0] * 10,
    "lower_coefficients": [0.0] * 10,
    "alpha_deg": [2.5, 2.5, 2.5],
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="deepseek/deepseek-v4-pro",
        resolved_model="deepseek/deepseek-v4-pro-20260423",
        resolved_provider="StreamLake",
        provider_response_id="stage-b-fixture-response",
        finish_reason="stop",
        input_tokens=1_234,
        output_tokens=56,
        reasoning_tokens=34,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.00012340"),
        latency_ns=12_345_678,
        attempt_count=1,
    )


def _entry(
    *,
    namespace: str,
    draft: InsightDraft,
    operator_kinds: tuple[str, ...],
):
    ids = DeterministicIdFactory(namespace)
    memory = InsightMemoryBank(id_factory=ids)
    entry, added = memory.add(
        draft,
        applicable_operator_kinds=operator_kinds,
        origin=InsightOrigin.MANUAL,
    )
    assert added
    return entry


def _airfoil_authority():
    benchmark = AgenticBenchmark(
        problem=airfoil_base_benchmark.problem,
        reward=airfoil_base_benchmark.reward,
        detailed_evaluator=airfoil_base_benchmark.detailed_evaluator,
        outcome_relation=airfoil_base_benchmark.outcome_relation,
        optimization_semantics=airfoil_base_benchmark.optimization_semantics,
        action_semantics=airfoil_base_benchmark.action_semantics,
        phenotype_identity=airfoil_base_benchmark.phenotype_identity,
        finite_variation_catalogs=airfoil_base_benchmark.finite_variation_catalogs,
        hypothesis_compiler=AirfoilV7TrimHypothesisCompiler(),
        finite_action_set_compiler=AirfoilTrimLocalSupportCompiler(),
    )
    anchor = "trim.p025.n025.p050"
    entry = _entry(
        namespace="stage_b_airfoil_real_adapter",
        draft=InsightDraft(
            claim="The signed pointwise trim pattern should reduce lift mismatch.",
            trigger="The held-out parent retains pointwise lift residuals.",
            mechanism="Outer angle increases and a central decrease rebalance lift.",
            affected_paths=(
                "$.alpha_deg[0]",
                "$.alpha_deg[1]",
                "$.alpha_deg[2]",
            ),
            evidence_summary="A prior randomized diagnostic supports this trim sign pattern.",
            confidence=0.8,
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="objective:normalized_multipoint_drag",
                    direction=MetricEffectDirection.INCREASE,
                ),
                MetricEffectPrediction(
                    metric_id="violation:normalized_lift_equality",
                    direction=MetricEffectDirection.DECREASE,
                ),
            ),
            recommended_option_families=("trim_only",),
            recommended_option_ids=(anchor,),
            action_template="Apply the exact signed pointwise trim template.",
            falsification_condition="Lift mismatch does not decrease.",
        ),
        operator_kinds=("mutation",),
    )
    compiled = benchmark.compile_registered_hypothesis_treatment(
        catalog_id="airfoil_v7_trim",
        parent_candidate_id=(
            DeterministicIdFactory("stage_b_airfoil_parent").new_candidate_id()
        ),
        parent_configuration=AIRFOIL_PARENT,
        entry=entry,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256=_sha("stage B Airfoil context"),
        endpoint_definition_sha256=airfoil_base_benchmark.reward.definition_hash,
    )
    authority, draft = benchmark.compile_finite_action_set(
        compiled_anchor=compiled,
        required_cardinality=8,
        source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
    )
    return compiled, authority, draft


def _boils_authority():
    benchmark = AgenticBenchmark(
        problem=boils_base_benchmark.problem,
        phenotype_identity=boils_base_benchmark.phenotype_identity,
        finite_variation_catalogs=boils_base_benchmark.finite_variation_catalogs,
        hypothesis_compiler=BoilsPositionHypothesisCompiler(),
        finite_action_set_compiler=BoilsPositionLocalSupportCompiler(),
    )
    parent = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}
    contract = benchmark.bind_finite_variation(FINITE_CATALOG_ID, parent)
    anchor = next(
        option.option_id
        for option in contract.options
        if option.option_id.startswith("boils_abc.p00.")
    )
    entry = _entry(
        namespace="stage_b_boils_real_adapter",
        draft=InsightDraft(
            claim="The first synthesis position should use a better local action.",
            trigger="The parent sequence exposes position zero for replacement.",
            mechanism="Early rewriting changes the logic graph seen downstream.",
            affected_paths=("$.sequence[0]",),
            evidence_summary="A prior diagnostic identifies position zero as actionable.",
            confidence=0.7,
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="objective:total_levels",
                    direction=MetricEffectDirection.DECREASE,
                ),
                MetricEffectPrediction(
                    metric_id="objective:total_lut_count",
                    direction=MetricEffectDirection.DECREASE,
                ),
            ),
            recommended_option_families=tuple(
                sorted(set(ACTION_FAMILIES.values()))
            ),
            recommended_option_ids=(anchor,),
            action_template="Select one exact action at sequence position zero.",
            falsification_condition="Neither circuit objective improves.",
        ),
        operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
    )
    compiled = benchmark.compile_registered_hypothesis_treatment(
        catalog_id=FINITE_CATALOG_ID,
        parent_candidate_id=(
            DeterministicIdFactory("stage_b_boils_parent").new_candidate_id()
        ),
        parent_configuration=parent,
        entry=entry,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256=_sha("stage B BOiLS context"),
        endpoint_definition_sha256=_sha("stage B BOiLS Pareto endpoint"),
    )
    authority, draft = benchmark.compile_finite_action_set(
        compiled_anchor=compiled,
        required_cardinality=8,
        source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
    )
    return compiled, authority, draft


@pytest.mark.parametrize("factory", [_airfoil_authority, _boils_authority])
def test_real_adapters_seal_the_same_generic_k8_authority(factory) -> None:
    compiled, authority, draft = factory()

    assert authority.support.cardinality == 8
    assert authority.support.compatible_option_count == 8
    assert len({row.option.option_id for row in authority.support.options}) == 8
    assert len(
        {row.option.child_configuration_sha256 for row in authority.support.options}
    ) == 8
    assert len(
        {row.phenotype_identity_sha256 for row in authority.support.options}
    ) == 8
    assert authority.support.anchor_option_id in draft.ordered_option_ids
    assert authority.support.presentation.ordered_option_ids == draft.ordered_option_ids
    assert authority.card.reference == compiled.request.reference
    assert authority.current_outcome_access is False
    assert compiled.requirement.allowed_actions[0].option_id == (
        authority.support.anchor_option_id
    )
    assert tuple(
        action.option_id for action in compiled.requirement.allowed_actions
    ) == (authority.support.anchor_option_id,)


def test_stage_b_adapter_modules_depend_only_on_public_agentic_facade() -> None:
    root = Path(__file__).parents[1] / "examples" / "benchmarks"
    paths = (
        root / "engibench_airfoil" / "v7_stage_b_action_set.py",
        root / "boils_abc" / "stage_b_action_set.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        agent_evolve_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("agent_evolve")
        }
        assert agent_evolve_imports == {"agent_evolve.agentic"}


@pytest.mark.parametrize("factory", [_airfoil_authority, _boils_authority])
def test_prospective_uniform_engine_uses_the_exact_real_adapter_support(factory) -> None:
    _, authority, _ = factory()
    policy = TaskKeyedUniformFiniteActionPolicy(
        schedule_seed_sha256=_sha("prospective Stage-B uniform schedule seed"),
    )
    token = policy.freeze_rank(
        authority,
        task_sha256=_sha("matched Stage-B block task"),
        pre_outcome_phase_commit_sha256=_sha("pre-G1 outcome-free phase commit"),
    )
    decision = policy.choose(
        EngineFiniteActionRequest(
            authority=authority,
            prospective_rank=token,
        )
    )

    validate_finite_action_decision(authority, decision)
    selected = authority.support.options[token.selected_ordinal]
    assert decision.option_id == selected.option.option_id
    assert decision.option_identity_sha256 == selected.option.identity_sha256
    assert decision.child_configuration_sha256 == (
        selected.option.child_configuration_sha256
    )
    assert decision.phenotype_identity_sha256 == (
        selected.phenotype_identity_sha256
    )
    assert (decision.propensity_numerator, decision.propensity_denominator) == (1, 8)
    assert decision.current_outcome_access is False
    assert policy.choose(
        EngineFiniteActionRequest(authority, token)
    ) == decision

    forged = replace(
        token,
        selected_ordinal=(token.selected_ordinal + 1) % token.cardinality,
    )
    with pytest.raises(ValueError, match="does not replay its public rank"):
        policy.choose(EngineFiniteActionRequest(authority, forged))


@pytest.mark.parametrize("factory", [_airfoil_authority, _boils_authority])
def test_model_choice_seals_one_exact_real_adapter_support_row(factory) -> None:
    _, authority, _ = factory()
    ordinal = next(
        index
        for index, row in enumerate(authority.support.options)
        if row.option.option_id != authority.support.anchor_option_id
    )
    row = authority.support.options[ordinal]
    draft = FiniteVariationSelectionDraft(
        option_id=row.option.option_id,
        option_identity_sha256=row.option.identity_sha256,
        contract_identity_sha256=(
            authority.support.support_contract.identity_sha256
        ),
        design_rationale="Select a non-anchor local action from the matched support.",
        claimed_insight_ids=(authority.card.reference.insight_id.value,),
    )
    telemetry = _telemetry()
    prompt_sha256 = _sha("exact Stage-B semantic prompt")
    decision = seal_model_finite_action_decision(
        authority=authority,
        call_id=LLMCallId("call_stage_b_model_fixture"),
        prompt_sha256=prompt_sha256,
        draft=draft,
        telemetry=telemetry,
    )

    validate_finite_action_decision(authority, decision)
    assert decision.selector_kind is FiniteActionSelectorKind.MODEL
    assert decision.selected_ordinal == ordinal
    assert decision.option_id == row.option.option_id
    assert decision.model_prompt_sha256 == prompt_sha256
    assert decision.model_telemetry_sha256 == (
        model_finite_action_telemetry_sha256(telemetry)
    )
    assert decision.prospective_token_sha256 is None
    assert decision.propensity_numerator is None
    assert decision.propensity_denominator is None

    with pytest.raises(ValueError, match="different finite contract"):
        seal_model_finite_action_decision(
            authority=authority,
            call_id=LLMCallId("call_stage_b_model_forged_contract"),
            prompt_sha256=prompt_sha256,
            draft=replace(draft, contract_identity_sha256="f" * 64),
            telemetry=telemetry,
        )
    with pytest.raises(ValueError, match="exact prompt identity"):
        replace(decision, model_prompt_sha256=None)
