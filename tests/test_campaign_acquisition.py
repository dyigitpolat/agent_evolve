from __future__ import annotations

import pytest

from agent_evolve.integrations.pydantic_ai.campaign_acquisition import (
    CampaignAcquisitionMode,
    build_campaign_acquisition_allocator,
    campaign_constraint_decoupled_acquisition_from_environment,
    campaign_contextual_search_allocation_from_environment,
    campaign_evidence_calibrated_source_mix_from_environment,
    campaign_minimum_intervention_projection_from_environment,
    campaign_operator_assay_minimum_from_environment,
    campaign_residual_frontier_planning_from_environment,
    campaign_selector_policy_definition_sha256,
)
from agent_evolve.policies.selection.full_support_slate import FullSupportSlatePolicy
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlatePolicy,
    build_controller_owned_family_exposure_phases,
    build_terminal_tapered_family_exposure_phases,
)


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    (
        (CampaignAcquisitionMode.MODEL_TOP_K, ModelAnchoredCalibratedSlatePolicy),
        (CampaignAcquisitionMode.CALIBRATED_FRONTIER, StructuralPosteriorSlatePolicy),
        (
            CampaignAcquisitionMode.OPERATOR_STRATIFIED,
            OperatorStratifiedStructuralPosteriorSlatePolicy,
        ),
    ),
)
def test_common_pool_acquisition_modes_have_distinct_exact_allocators(
    mode: CampaignAcquisitionMode,
    expected_type: type[object],
) -> None:
    allocator = build_campaign_acquisition_allocator(
        mode,
        common_pool_enabled=True,
    )

    assert type(allocator) is expected_type
    assert len(campaign_selector_policy_definition_sha256(allocator)) == 64


def test_full_support_is_the_only_non_common_pool_acquisition() -> None:
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.FULL_SUPPORT,
        common_pool_enabled=False,
    )
    assert type(allocator) is FullSupportSlatePolicy

    with pytest.raises(ValueError, match="K8-to-K4"):
        build_campaign_acquisition_allocator(
            CampaignAcquisitionMode.CALIBRATED_FRONTIER,
            common_pool_enabled=False,
        )
    with pytest.raises(ValueError, match="incompatible with K4"):
        build_campaign_acquisition_allocator(
            CampaignAcquisitionMode.FULL_SUPPORT,
            common_pool_enabled=True,
        )


def test_operator_assay_minimum_is_generic_configured_method_state() -> None:
    one = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        common_pool_enabled=True,
        operator_assay_minimum=1,
    )
    two = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        common_pool_enabled=True,
        operator_assay_minimum=2,
    )

    assert one.required_family_minimums == (("composite_r2", 1),)
    assert two.required_family_minimums == (("composite_r2", 2),)
    assert one.configuration_sha256 != two.configuration_sha256
    assert campaign_operator_assay_minimum_from_environment({}) == 1
    assert campaign_operator_assay_minimum_from_environment(
        {"AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM": "2"}
    ) == 2
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        campaign_operator_assay_minimum_from_environment(
            {"AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM": "5"}
        )


def test_horizon_bounded_acquisition_requires_an_explicit_generic_schedule() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        common_pool_enabled=True,
        family_exposure_phases=phases,
    )

    assert type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy
    assert allocator.exposure_phase_for_wave(1).bounds[0].minimum_evaluations == 2
    assert allocator.exposure_phase_for_wave(5).bounds[0].maximum_evaluations == 0
    assert len(campaign_selector_policy_definition_sha256(allocator)) == 64
    with pytest.raises(ValueError, match="explicit family_exposure_phases"):
        build_campaign_acquisition_allocator(
            CampaignAcquisitionMode.HORIZON_BOUNDED,
            common_pool_enabled=True,
        )


def test_constraint_decoupled_acquisition_is_one_generic_opt_in() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        common_pool_enabled=True,
        family_exposure_phases=phases,
    )

    strict = campaign_selector_policy_definition_sha256(allocator)
    reconciled = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
    )

    assert strict != reconciled
    assert campaign_constraint_decoupled_acquisition_from_environment({}) is False
    assert campaign_constraint_decoupled_acquisition_from_environment(
        {"AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION": "1"}
    ) is True
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        campaign_constraint_decoupled_acquisition_from_environment(
            {"AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION": "true"}
        )
    with pytest.raises(ValueError, match="horizon-bounded"):
        campaign_selector_policy_definition_sha256(
            StructuralPosteriorSlatePolicy(),
            constraint_decoupled=True,
        )


def test_minimum_intervention_projection_is_a_versioned_generic_opt_in() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        common_pool_enabled=True,
        family_exposure_phases=phases,
    )

    v9 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
    )
    v10 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
    )

    assert v10 != v9
    assert campaign_minimum_intervention_projection_from_environment({}) is False
    assert campaign_minimum_intervention_projection_from_environment(
        {"AGENT_EVOLVE_MINIMUM_INTERVENTION_PROJECTION": "1"}
    ) is True
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        campaign_minimum_intervention_projection_from_environment(
            {"AGENT_EVOLVE_MINIMUM_INTERVENTION_PROJECTION": "true"}
        )
    with pytest.raises(ValueError, match="constraint-decoupled"):
        campaign_selector_policy_definition_sha256(
            allocator,
            minimum_intervention_projection=True,
        )


def test_evidence_calibrated_source_mix_is_a_versioned_generic_opt_in() -> None:
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        common_pool_enabled=True,
        family_exposure_phases=phases,
    )
    v10 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
    )
    v11 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
    )

    assert v11 != v10
    assert campaign_evidence_calibrated_source_mix_from_environment({}) is False
    assert campaign_evidence_calibrated_source_mix_from_environment(
        {"AGENT_EVOLVE_EVIDENCE_CALIBRATED_SOURCE_MIX": "1"}
    ) is True
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        campaign_evidence_calibrated_source_mix_from_environment(
            {"AGENT_EVOLVE_EVIDENCE_CALIBRATED_SOURCE_MIX": "true"}
        )
    with pytest.raises(ValueError, match="minimum intervention"):
        campaign_selector_policy_definition_sha256(
            allocator,
            constraint_decoupled=True,
            evidence_calibrated_source_mix=True,
        )


def test_contextual_search_allocation_owns_exact_dose_inside_broad_envelope() -> None:
    phases = build_controller_owned_family_exposure_phases(
        family="composite_r2"
    )
    allocator = build_campaign_acquisition_allocator(
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        common_pool_enabled=True,
        family_exposure_phases=phases,
    )
    bound = allocator.exposure_phase_for_wave(99).bounds[0]
    v11 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
    )
    v12 = campaign_selector_policy_definition_sha256(
        allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )

    assert (bound.minimum_evaluations, bound.maximum_evaluations) == (0, 4)
    assert v12 != v11
    assert campaign_contextual_search_allocation_from_environment({}) is False
    assert campaign_contextual_search_allocation_from_environment(
        {"AGENT_EVOLVE_CONTEXTUAL_SEARCH_ALLOCATION": "1"}
    ) is True
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        campaign_contextual_search_allocation_from_environment(
            {"AGENT_EVOLVE_CONTEXTUAL_SEARCH_ALLOCATION": "true"}
        )
    assert campaign_residual_frontier_planning_from_environment({}) is False
    assert campaign_residual_frontier_planning_from_environment(
        {"AGENT_EVOLVE_RESIDUAL_FRONTIER_PLANNING": "1"}
    ) is True
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        campaign_residual_frontier_planning_from_environment(
            {"AGENT_EVOLVE_RESIDUAL_FRONTIER_PLANNING": "yes"}
        )
    with pytest.raises(ValueError, match="evidence-calibrated source mix"):
        campaign_selector_policy_definition_sha256(
            allocator,
            constraint_decoupled=True,
            minimum_intervention_projection=True,
            contextual_search_allocation=True,
        )
