"""Provider-free conformance for the frozen cross-workload experiment profile."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve import (
    AuthenticatedAffineFrontierContextProjector,
    CampaignExperimentProfile,
    CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID,
    CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID,
    CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID,
    EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID,
    HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID,
    MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID,
    OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID,
    REFERENCE_36_OFFSPRING_SCALE_SHAPE,
    RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID,
    STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID,
    ReferenceCampaignImplementations,
    ResidualHypervolumeCampaignParentSelector,
    StagnationAwareDiverseCampaignParentSelector,
    rebind_reference_campaign_implementations,
    reference_atomic_variation_topology_binding,
    reference_campaign_experiment_profile,
    reference_contextual_outcomes_binding,
    reference_hierarchical_r2_variation_topology_binding,
)
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    CampaignPolicyBinding,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    GPT_OSS_20B_GROQ_HIGH_SERIAL,
    QWEN_3_7_MAX_ALIBABA_XHIGH,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    build_controller_owned_family_exposure_phases,
    build_terminal_tapered_family_exposure_phases,
)
from agent_evolve.workload_prompt import WorkloadPromptArm


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _binding(name: str) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=object(),
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"policy:{name}"),
    )


class _ArchiveUtility:
    utility_id = "profile_test_hypervolume"
    utility_version = 1
    definition_sha256 = _sha("profile-test-hypervolume")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object({"test_only": True}),
        )


def _profile(*, profile_id: str, model) -> CampaignExperimentProfile:
    return CampaignExperimentProfile(
        profile_id=profile_id,
        profile_version=1,
        method_id="generic_successor_reference",
        method_version=1,
        scale_shape=REFERENCE_36_OFFSPRING_SCALE_SHAPE,
        candidate_pool_size=24,
        model_selection_size=8,
        prompt_arm=WorkloadPromptArm.SEMANTIC,
        parent_selection=_binding("archive_elite_explorer_common"),
        memory_assignment=_binding("predictive_memory_diagnostic"),
        portfolio_selection=_binding("common_pool_m24_model_k8_engine_k4"),
        recombination=_binding("adaptive_disjoint_union"),
        reflection=_binding("delayed_quarantined_reflection"),
        model_execution=model,
    )


def test_method_identity_is_stable_across_models_but_execution_is_not() -> None:
    deepseek = _profile(
        profile_id="successor_deepseek",
        model=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    )
    qwen = _profile(
        profile_id="successor_qwen",
        model=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )

    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert deepseek.to_record()["method"]["workload_specific_fields"] == []
    assert deepseek.to_record()["method"]["model_specific_optimizer_fields"] == []


def test_reference_profile_derives_current_38_candidate_7_call_schedule() -> None:
    profile = _profile(
        profile_id="successor_schedule",
        model=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )

    preset = profile.preset(outer_seed=20260770)
    budget = preset.budget(required_seed_count=2)
    behavior = profile.behavior(archive_utility=_ArchiveUtility())

    assert profile.scale_shape.planned_offspring_occurrences == 36
    assert budget.max_unique_evaluations == 38
    assert budget.max_logical_llm_calls == 7
    assert behavior.parent_selection.policy_id == "archive_elite_explorer_common"
    assert len(profile.method_definition_sha256) == 64
    assert len(profile.experiment_definition_sha256) == 64


def test_profile_rejects_hidden_width_and_route_capacity_changes() -> None:
    profile = _profile(
        profile_id="successor_validation",
        model=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )

    with pytest.raises(ValueError, match="M >= K >= k"):
        replace(profile, model_selection_size=3)
    with pytest.raises(ValueError, match="route cap"):
        replace(
            profile,
            model_execution=GPT_OSS_20B_GROQ_HIGH_SERIAL,
            agent_concurrency=3,
        )


def test_serial_route_is_allowed_only_as_an_execution_level_change() -> None:
    base = _profile(
        profile_id="successor_qwen_base",
        model=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    serial = replace(
        base,
        profile_id="successor_oss20_serial",
        model_execution=GPT_OSS_20B_GROQ_HIGH_SERIAL,
        agent_concurrency=1,
        agent_queue_capacity=1,
    )

    assert serial.method_definition_sha256 == base.method_definition_sha256
    assert serial.experiment_definition_sha256 != base.experiment_definition_sha256
    assert serial.to_record()["systems"]["agent_concurrency"] == 1


def test_profile_authenticates_complete_finite_contract_acquisition() -> None:
    base = _profile(
        profile_id="successor_complete_support",
        model=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    )
    complete = replace(base, candidate_pool_size=None)

    method = complete.to_record()["method"]
    assert method["candidate_pool_size"] is None
    assert method["candidate_pool_mode"] == "complete_finite_contract"
    assert complete.method_definition_sha256 != base.method_definition_sha256


def _implementations(
    *,
    archive_context=None,
    context_local: bool = False,
    hierarchical: bool = False,
    portfolio_selection: object | None = None,
    parent_selection: object | None = None,
) -> ReferenceCampaignImplementations:
    return ReferenceCampaignImplementations(
        parent_selection=(object() if parent_selection is None else parent_selection),
        memory_assignment=object(),
        portfolio_selection=(
            object() if portfolio_selection is None else portfolio_selection
        ),
        recombination=object(),
        reflection=object(),
        archive_context=archive_context,
        variation_topology=(
            reference_hierarchical_r2_variation_topology_binding(
                object(),
                max_composite_options=16,
                required_composite_proposals=2,
            )
            if hierarchical
            else reference_atomic_variation_topology_binding(object())
            if context_local
            else None
        ),
        contextual_outcomes=(
            reference_contextual_outcomes_binding(object())
            if context_local or hierarchical
            else None
        ),
    )


def test_reference_profile_rebinds_runtime_objects_without_method_drift() -> None:
    first = reference_campaign_experiment_profile(
        profile_id="reference_cross_workload",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=_implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
    )
    rebound = rebind_reference_campaign_implementations(
        first,
        _implementations(),
    )

    assert rebound.experiment_definition_sha256 == (first.experiment_definition_sha256)
    assert rebound.method_definition_sha256 == first.method_definition_sha256
    assert rebound.parent_selection.implementation is not (
        first.parent_selection.implementation
    )
    assert rebound.to_record()["method"]["workload_specific_fields"] == []


def test_reference_profile_authenticates_frontier_context_as_method_not_system() -> (
    None
):
    without_context = reference_campaign_experiment_profile(
        profile_id="reference_without_frontier",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=_implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
    )
    deepseek = reference_campaign_experiment_profile(
        profile_id="reference_frontier_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=_implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector()
        ),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
    )
    qwen = reference_campaign_experiment_profile(
        profile_id="reference_frontier_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
        implementations=_implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector()
        ),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
    )

    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert deepseek.method_definition_sha256 != (
        without_context.method_definition_sha256
    )
    assert deepseek.archive_context_projector is not None
    assert deepseek.to_record()["method"]["policies"]["archive_context"] == {
        "policy_id": "authenticated_affine_frontier_context",
        "policy_version": 1,
        "definition_sha256": deepseek.archive_context.definition_sha256,
    }


def test_reference_profile_rebinds_frontier_runtime_without_identity_drift() -> None:
    first_projector = AuthenticatedAffineFrontierContextProjector()
    second_projector = AuthenticatedAffineFrontierContextProjector()
    first = reference_campaign_experiment_profile(
        profile_id="reference_frontier_rebind",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=_implementations(archive_context=first_projector),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
    )
    rebound = rebind_reference_campaign_implementations(
        first,
        _implementations(archive_context=second_projector),
    )

    assert rebound.experiment_definition_sha256 == first.experiment_definition_sha256
    assert rebound.archive_context_projector is second_projector
    assert rebound.archive_context_projector is not first.archive_context_projector
    with pytest.raises(ValueError, match="differs from method identity"):
        rebind_reference_campaign_implementations(first, _implementations())


def test_context_local_successor_authenticates_generic_transfer_boundaries() -> None:
    deepseek = reference_campaign_experiment_profile(
        profile_id="context_local_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=_implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            context_local=True,
        ),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
    )
    qwen = replace(
        deepseek,
        profile_id="context_local_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        _implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            context_local=True,
        ),
    )

    method = deepseek.to_record()["method"]
    policies = method["policies"]
    assert deepseek.method_id == CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID
    assert method["schema_version"] == 3
    assert policies["memory_assignment"]["policy_id"] == (
        "exact_context_predictive_memory"
    )
    assert policies["variation_topology"]["policy_id"] == (
        "parent_bound_atomic_finite_variation"
    )
    assert policies["contextual_outcomes"]["policy_id"] == (
        "parent_local_prior_outcome_history"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == (
        deepseek.experiment_definition_sha256
    )


def test_operator_stratified_successor_is_generic_and_rebindable() -> None:
    implementations = _implementations(
        archive_context=AuthenticatedAffineFrontierContextProjector(),
        hierarchical=True,
        portfolio_selection=OperatorStratifiedStructuralPosteriorSlatePolicy(),
    )
    deepseek = reference_campaign_experiment_profile(
        profile_id="operator_stratified_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations,
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        operator_stratified_acquisition=True,
    )
    qwen = replace(
        deepseek,
        profile_id="operator_stratified_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        _implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=(OperatorStratifiedStructuralPosteriorSlatePolicy()),
        ),
    )

    policies = deepseek.to_record()["method"]["policies"]
    assert deepseek.method_id == OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID
    assert policies["portfolio_selection"]["policy_id"] == (
        "operator_stratified_hierarchical_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_horizon_bounded_successor_is_generic_model_invariant_and_rebindable() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )

    def implementations():
        return _implementations(
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="horizon_bounded_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
    )
    qwen = replace(
        deepseek,
        profile_id="horizon_bounded_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    policies = deepseek.to_record()["method"]["policies"]
    assert deepseek.method_id == HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID
    assert policies["portfolio_selection"]["policy_id"] == (
        "horizon_bounded_hierarchical_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_stagnation_aware_successor_has_exact_cross_model_parent_identity() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )

    def implementations():
        return _implementations(
            parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="stagnation_aware_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
    )
    qwen = replace(
        deepseek,
        profile_id="stagnation_aware_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 8
    assert deepseek.parent_selection.policy_id == (
        "stagnation_aware_diverse_campaign_parent"
    )
    assert deepseek.parent_selection.definition_sha256 == (
        StagnationAwareDiverseCampaignParentSelector.definition_sha256
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_constraint_decoupled_successor_has_new_generic_method_identity() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )

    def implementations():
        return _implementations(
            parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="constraint_decoupled_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
        constraint_decoupled_acquisition=True,
    )
    qwen = replace(
        deepseek,
        profile_id="constraint_decoupled_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 9
    assert deepseek.portfolio_selection.policy_id == (
        "constraint_decoupled_horizon_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_minimum_intervention_successor_has_v10_generic_method_identity() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )

    def implementations():
        return _implementations(
            parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="minimum_intervention_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
        constraint_decoupled_acquisition=True,
        minimum_intervention_projection=True,
    )
    qwen = replace(
        deepseek,
        profile_id="minimum_intervention_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 10
    assert deepseek.portfolio_selection.policy_id == (
        "minimum_intervention_horizon_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_source_mix_successor_has_v11_generic_method_identity() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )

    def implementations():
        return _implementations(
            parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="source_mix_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
        constraint_decoupled_acquisition=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
    )
    qwen = replace(
        deepseek,
        profile_id="source_mix_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 11
    assert deepseek.portfolio_selection.policy_id == (
        "evidence_calibrated_source_mix_horizon_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_contextual_search_successor_has_v21_model_neutral_method_identity() -> None:
    phases = build_controller_owned_family_exposure_phases(family="composite_r2")

    def implementations():
        return _implementations(
            parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="contextual_search_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
        constraint_decoupled_acquisition=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    qwen = replace(
        deepseek,
        profile_id="contextual_search_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 21
    assert deepseek.portfolio_selection.policy_id == (
        "contextual_source_operator_horizon_k8_engine_k4"
    )
    assert deepseek.contextual_outcomes is not None
    assert deepseek.contextual_outcomes.policy_id == (
        "multi_horizon_parent_local_outcomes"
    )
    assert deepseek.contextual_outcomes.policy_version == 2
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_residual_frontier_successor_has_v22_model_neutral_method_identity() -> None:
    phases = build_controller_owned_family_exposure_phases(family="composite_r2")

    def implementations():
        return _implementations(
            parent_selection=ResidualHypervolumeCampaignParentSelector(),
            archive_context=AuthenticatedAffineFrontierContextProjector(),
            hierarchical=True,
            portfolio_selection=HorizonBoundedStructuralPosteriorSlatePolicy(phases),
        )

    deepseek = reference_campaign_experiment_profile(
        profile_id="residual_frontier_deepseek",
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        implementations=implementations(),
        candidate_pool_size=None,
        evaluator_concurrency=1,
        agent_concurrency=3,
        agent_queue_capacity=8,
        hierarchical_proposal_support=True,
        horizon_bounded_acquisition=True,
        constraint_decoupled_acquisition=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    qwen = replace(
        deepseek,
        profile_id="residual_frontier_qwen",
        model_execution=QWEN_3_7_MAX_ALIBABA_XHIGH,
    )
    rebound = rebind_reference_campaign_implementations(
        deepseek,
        implementations(),
    )

    assert deepseek.method_id == RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID
    assert deepseek.method_version == 23
    assert deepseek.parent_selection.policy_id == (
        "residual_hypervolume_campaign_parent"
    )
    assert deepseek.portfolio_selection.policy_id == (
        "residual_frontier_contextual_source_operator_horizon_k8_engine_k4"
    )
    assert deepseek.method_definition_sha256 == qwen.method_definition_sha256
    assert deepseek.experiment_definition_sha256 != qwen.experiment_definition_sha256
    assert rebound.experiment_definition_sha256 == deepseek.experiment_definition_sha256


def test_stagnation_aware_parent_cannot_be_silently_labeled_as_older_stack() -> None:
    with pytest.raises(ValueError, match="requires hierarchical support"):
        reference_campaign_experiment_profile(
            profile_id="stagnation_aware_incomplete",
            model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
            implementations=_implementations(
                parent_selection=StagnationAwareDiverseCampaignParentSelector(),
            ),
            candidate_pool_size=None,
            evaluator_concurrency=1,
            agent_concurrency=3,
            agent_queue_capacity=8,
        )


def test_operator_stratified_successor_rejects_atomic_topology() -> None:
    with pytest.raises(ValueError, match="complete context-local hierarchical"):
        reference_campaign_experiment_profile(
            profile_id="operator_stratified_atomic_invalid",
            model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
            implementations=_implementations(
                archive_context=AuthenticatedAffineFrontierContextProjector(),
                context_local=True,
                portfolio_selection=(
                    OperatorStratifiedStructuralPosteriorSlatePolicy()
                ),
            ),
            candidate_pool_size=None,
            evaluator_concurrency=1,
            agent_concurrency=3,
            agent_queue_capacity=8,
            hierarchical_proposal_support=True,
            operator_stratified_acquisition=True,
        )


def test_operator_assay_floor_changes_method_identity_and_cannot_rebind() -> None:
    def build(minimum: int) -> CampaignExperimentProfile:
        return reference_campaign_experiment_profile(
            profile_id=f"operator_floor_{minimum}",
            model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
            implementations=_implementations(
                archive_context=AuthenticatedAffineFrontierContextProjector(),
                hierarchical=True,
                portfolio_selection=(
                    OperatorStratifiedStructuralPosteriorSlatePolicy(
                        (("composite_r2", minimum),)
                    )
                ),
            ),
            candidate_pool_size=None,
            evaluator_concurrency=1,
            agent_concurrency=3,
            agent_queue_capacity=8,
            hierarchical_proposal_support=True,
            operator_stratified_acquisition=True,
        )

    one = build(1)
    two = build(2)

    assert one.method_id == two.method_id
    assert one.method_definition_sha256 != two.method_definition_sha256
    with pytest.raises(AssertionError, match="changed experiment identity"):
        rebind_reference_campaign_implementations(
            one,
            _implementations(
                archive_context=AuthenticatedAffineFrontierContextProjector(),
                hierarchical=True,
                portfolio_selection=(
                    OperatorStratifiedStructuralPosteriorSlatePolicy(
                        (("composite_r2", 2),)
                    )
                ),
            ),
        )
