"""Workload-neutral regression checks for systematic campaign trace analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.development.analyze_systematic_campaign_trace import (
    _action_forecast_information,
    analyze_run,
)


WORKSPACE = Path(__file__).resolve().parents[2]
LOGS = WORKSPACE / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"


def test_action_forecast_information_detects_structured_prediction_collapse() -> None:
    forecast = {
        "authenticated_record": {
            "call_id": "forecast_1",
            "operation": "forecast_target_realization",
            "typed_output": {
                "probability_valid_codes": ["p0_95", "p0_8", "p0_95"],
                "median_effect_codes": [
                    ["z", "z"],
                    ["p1", "n0_5"],
                    ["z", "z"],
                ],
                "lower_uncertainty_codes": [
                    ["u0", "u0"],
                    ["u1", "u0_5"],
                    ["u0", "u0"],
                ],
                "upper_uncertainty_codes": [
                    ["u0", "u0"],
                    ["u2", "u0_5"],
                    ["u0", "u0"],
                ],
            },
        }
    }
    ignored_selector = {
        "authenticated_record": {
            "call_id": "selector_1",
            "operation": "select_portfolio",
            "typed_output": {},
        }
    }

    result = _action_forecast_information([forecast, ignored_selector])

    assert result["forecast_call_count"] == 1
    assert result["action_count"] == 3
    assert result["metric_cell_count"] == 6
    assert result["effect_code_counts"] == {"n0_5": 1, "p1": 1, "z": 4}
    assert result["effect_sign_counts"] == {
        "negative": 1,
        "positive": 1,
        "zero": 4,
    }
    assert result["zero_effect_cell_rate"] == pytest.approx(2 / 3)
    assert result["high_validity_action_rate"] == pytest.approx(2 / 3)
    assert result["zero_uncertainty_cell_rate"] == pytest.approx(2 / 3)
    assert result["asymmetric_uncertainty_cell_rate"] == pytest.approx(1 / 6)
    assert result["distinct_full_action_signature_count"] == 2
    assert result["most_common_full_action_signature_rate"] == pytest.approx(2 / 3)
    assert result["all_zero_call_rate"] == 0.0
    assert result["constant_effect_call_rate"] == 0.0


def test_action_forecast_information_has_typed_empty_state() -> None:
    result = _action_forecast_information([])

    assert result["forecast_call_count"] == 0
    assert result["action_count"] == 0
    assert result["effect_entropy_nats"] is None
    assert result["zero_effect_cell_rate"] is None
    assert result["most_common_full_action_signature_rate"] is None


@pytest.mark.parametrize(
    ("workload_id", "relative", "expected_hv", "unique_evaluations"),
    (
        (
            "boils_abc",
            "boils_abc/generic_campaign/grid_boils_abc_mistral_s20260740_r2_live",
            0.09381041666666665,
            62,
        ),
        (
            "heat2d",
            "benchmark_q1/engibench_heat2d/generic_campaign/grid_heat2d_mistral_s20260740_r3_live",
            0.8840841557290414,
            62,
        ),
        (
            "timeloop_v2",
            "benchmark_q1/timeloop_codesign/full_support_g6/grid_timeloop_v2_mistral_s20260740_r2_live",
            0.8618923597113196,
            58,
        ),
    ),
)
def test_repaired_mistral_runs_normalize_without_workload_metric_names(
    workload_id: str,
    relative: str,
    expected_hv: float,
    unique_evaluations: int,
) -> None:
    row = analyze_run(
        LOGS / relative,
        workload_id=workload_id,
        model_profile="mistral",
        replicate_seed=20_260_740,
    )
    assert row["status"] == "completed_healthy"
    assert row["health_all_true"] is True
    assert row["evaluation"]["candidate_occurrences"] == 62
    assert row["evaluation"]["unique_evaluations"] == unique_evaluations
    assert row["provider"]["logical_calls"] == 7
    assert row["provider"]["physical_attempts"] == 7
    assert row["quality"]["final_hypervolume"] == pytest.approx(expected_hv)
    assert len(row["quality"]["trajectory"]) == 6
    assert (
        sum(
            value["physical_evaluation_span"]
            for value in row["quality"]["physical_evaluation_trajectory"]
        )
        == unique_evaluations
    )
    assert (
        row["quality"]["physical_evaluation_trajectory"][-1]["physical_evaluation"]
        == unique_evaluations
    )
    assert row["quality"]["physical_evaluation_trajectory"][-1][
        "hypervolume"
    ] == pytest.approx(expected_hv)
    assert row["memory_and_reflection"]["hex_exponent_interpretation_risk"] is True


def test_selector_trace_recovers_canonical_witness_copy_and_lane_collapse() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "grid_boils_abc_gpt_oss_120b_s20260740_wave2_repair_r2_live",
        workload_id="boils_abc",
        model_profile="gpt_oss_120b",
        replicate_seed=20_260_740,
    )
    behavior = row["selector_behavior"]

    assert behavior["selector_call_count"] == 6
    assert behavior["proposal_member_count"] == 48
    assert behavior["unique_option_count"] == 13
    assert behavior["exact_ordered_witness_copy_rate"] == 1.0
    assert behavior["exact_set_witness_copy_rate"] == 1.0
    assert behavior["witness_mode_counts"] == {"canonical": 6}
    assert [
        value["mean_lane_option_jaccard"]
        for value in behavior["generation_lane_diversity"]
    ] == pytest.approx([1.0, 7 / 9, 1 / 3])


def test_analyzer_retains_typed_candidate_infeasibility_without_fabricating_metrics() -> (
    None
):
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "mechanism_witness_boils_oss120_s20260741_keyed_live",
        workload_id="boils_abc",
        model_profile="gpt_oss_120b",
        replicate_seed=20_260_741,
    )

    assert row["evaluation"]["candidate_occurrences"] == 62
    assert row["evaluation"]["unique_evaluations"] == 62
    assert row["evaluation"]["scored_candidates"] == 60
    assert row["evaluation"]["typed_candidate_infeasible"] == 2
    assert row["evaluation"]["runtime_failures"] == 0
    assert (
        sum(
            value["physical_evaluation_span"]
            for value in row["quality"]["physical_evaluation_trajectory"]
        )
        == 62
    )
    assert row["quality"]["physical_evaluation_trajectory"][-1][
        "hypervolume"
    ] == pytest.approx(row["quality"]["final_hypervolume"])
    assert (
        sum(
            value["typed_candidate_infeasible_count"]
            for value in row["model_rank_calibration"]
        )
        == 2
    )


def test_analyzer_normalizes_provider_free_selector_decisions_without_a_witness() -> (
    None
):
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "mechanism_witness_boils_uniform_s20260742_control_v2",
        workload_id="boils_abc",
        model_profile="provider_free_conditional_uniform",
        replicate_seed=20_260_742,
        arm="conditional_uniform_reference",
    )

    behavior = row["selector_behavior"]
    assert row["status"] == "completed_healthy"
    assert row["provider"]["logical_calls"] == 0
    assert behavior["selector_call_count"] == 6
    assert behavior["proposal_member_count"] == 48
    assert behavior["unique_option_count"] > 8
    assert behavior["exact_ordered_witness_copy_rate"] is None
    assert behavior["exact_set_witness_copy_rate"] is None
    assert behavior["witness_mode_counts"] == {"provider_free_no_witness": 6}


def test_analyzer_normalizes_outcome_conditioned_all_action_selection() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "jul23_boils_ocafe_bootstrap_v4_deepseek_g6_b38_live_v2_20260723",
        workload_id="boils_abc",
        model_profile="deepseek_v4_pro_streamlake_xhigh",
        replicate_seed=20_260_723,
    )

    behavior = row["selector_behavior"]
    assert row["status"] == "completed_healthy"
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.08894375)
    assert behavior["selector_call_count"] == 6
    assert behavior["proposal_member_count"] == 24
    assert behavior["proposal_member_count_semantics"] == "trusted_evaluated_k_set"
    assert behavior["selection_mode_counts"] == {
        "outcome_conditioned_trusted_all_action": 6
    }
    assert behavior["outcome_conditioned_call_count"] == 6
    assert behavior["outcome_conditioned_forecast_universe_row_count_total"] == 1_385
    assert behavior["outcome_conditioned_forecast_universe_size_counts"] == {
        "230": 3,
        "231": 1,
        "232": 2,
    }
    assert (
        behavior["outcome_conditioned_allocator_candidate_evaluations_total"] == 285_061
    )
    assert behavior["outcome_conditioned_physical_forecast_call_count"] == 48

    calibration = row["forecast_calibration"]
    assert calibration["evaluated_forecast_member_count"] == 24
    assert calibration["effect_prediction_count"] == 48
    assert calibration["known_direction_forecast_count"] == 12
    assert calibration["direction_accuracy"] == pytest.approx(7 / 12)
    assert calibration["unknown_direction_forecast_rate"] == pytest.approx(0.75)
    assert calibration["high_confidence_known_direction_count"] == 3
    assert calibration["high_confidence_direction_error_count"] == 3
    assert calibration["high_confidence_direction_accuracy"] == 0.0

    information = row["action_forecast_information"]
    assert information["forecast_call_count"] == 48
    assert information["action_count"] == 1_385
    assert information["distinct_effect_code_count"] == 10
    assert information["effect_entropy_nats"] == pytest.approx(1.6004839389105696)


def test_analyzer_uses_campaign_event_accounting_for_workload_owned_summary() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/timeloop_codesign/full_support_g6/matched_uniform_control/"
        "grid_timeloop_v2_shared_s20260743_v2wave1_live",
        workload_id="timeloop_v2",
        model_profile="provider_free_conditional_uniform",
        replicate_seed=20_260_743,
        arm="conditional_uniform_reference",
    )

    assert row["status"] == "completed_healthy"
    expected_evaluation = {
        "candidate_occurrences": 62,
        "unique_evaluations": 62,
        "cache_reuse_occurrences": 0,
        # Candidate accounting includes the two authenticated seeds, matching
        # candidate_occurrences and the affine anytime trajectory.
        "scored_candidates": 62,
        "typed_candidate_infeasible": 0,
        "runtime_failures": 0,
        "generations": 6,
    }
    assert {
        key: row["evaluation"][key] for key in expected_evaluation
    } == expected_evaluation
    assert row["evaluation"]["physical_evaluator_latency_count"] == 62
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.8353211631883075)


def test_analyzer_recovers_common_universe_selection_and_allocator_layers() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "boils_common_universe_k24_qwen_s20260745_live_r1",
        workload_id="boils_abc",
        model_profile="qwen",
        replicate_seed=20_260_745,
    )

    behavior = row["selector_behavior"]
    assert row["status"] == "completed_healthy"
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.08771875)
    assert behavior["witness_mode_counts"] == {"task_keyed_common_candidate_pool": 6}
    assert behavior["common_pool_call_count"] == 6
    assert behavior["common_pool_candidate_universe_size_counts"] == {"24": 6}
    assert behavior["common_pool_model_selection_size_counts"] == {"8": 6}
    assert behavior["common_pool_evaluation_size_counts"] == {"4": 6}
    assert behavior["mean_common_universe_selection_fraction"] == pytest.approx(1 / 3)
    assert behavior["mean_common_universe_evaluation_fraction"] == pytest.approx(1 / 6)
    assert behavior["common_pool_prompt_projection_match_rate"] == 1.0
    assert behavior["common_pool_model_provider_blind_rate"] == 1.0
    assert behavior["common_pool_outcome_blind_rate"] == 1.0
    assert behavior["common_pool_hidden_witness_rate"] == 1.0
    assert behavior["common_pool_allocator_replacement_count"] == 2
    assert behavior["common_pool_allocator_replacement_rate"] == pytest.approx(2 / 24)
    assert behavior[
        "common_pool_literal_model_top_evaluation_size_preserved_rate"
    ] == pytest.approx(5 / 6)
    for call in behavior["calls"]:
        common = call["common_candidate_pool"]
        assert len(common["candidate_universe_option_ids"]) == 24
        assert len(call["option_ids"]) == 8
        assert len(common["evaluated_option_ids"]) == 4
        assert set(call["option_ids"]).issubset(common["candidate_universe_option_ids"])
        assert set(common["evaluated_option_ids"]).issubset(call["option_ids"])


def test_analyzer_separates_proposal_reservations_from_observed_k4_preference() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "gate2_hierarchical_boils_deepseek_s20260717_v4_live",
        workload_id="boils_abc",
        model_profile="deepseek_v4_pro_streamlake_xhigh",
        replicate_seed=20_260_717,
        arm="gate2_hierarchical_successor",
    )

    behavior = row["selector_behavior"]
    calibration = row["proposal_support_calibration"]
    assert behavior["proposal_support_call_count"] == 6
    assert behavior["proposal_support_reservation_count"] == 12
    assert behavior["proposal_support_selected_inclusion_rate"] == 1.0
    assert behavior["proposal_support_prompt_projection_match_rate"] == 1.0
    assert behavior["proposal_support_evaluated_reservation_count"] == 10
    assert behavior["proposal_support_reservation_evaluation_rate"] == pytest.approx(
        5 / 6
    )
    assert behavior["proposal_support_evaluator_slot_share"] == pytest.approx(5 / 12)
    assert calibration["selected_nonreservation_count"] == 36
    assert calibration["evaluated_nonreservation_count"] == 14
    assert calibration["reservation_to_nonreservation_evaluation_rate_ratio"] == (
        pytest.approx(15 / 7)
    )
    assert calibration["positive_individual_marginal_rate"] == pytest.approx(0.3)
    assert calibration[
        "nonreservation_positive_individual_marginal_rate"
    ] == pytest.approx(4 / 7)
    assert calibration["final_front_admission_count"] == 1
    assert all(
        call["proposal_support"]["reservations_force_evaluator_slots"] is False
        for call in behavior["calls"]
    )


def test_analyzer_joins_resolved_slots_to_original_model_ranks_and_roles() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/timeloop_codesign/full_support_g6/"
        "grid_timeloop_v2_deepseek_s20260770_v18r1_live",
        workload_id="timeloop_codesign_v2",
        model_profile="deepseek_v4_pro",
        replicate_seed=20_260_770,
    )

    assert row["model_rank_calibration_semantics"] == (
        "legacy_resolved_k4_candidate_label_rank_not_original_k8_model_rank"
    )
    join = row["rank_role_join"]
    assert join["join_semantics"] == (
        "authenticated_generation_parent_resolved_slot_to_original_k8_rank"
    )
    assert join["allocated_slot_count"] == 24
    assert join["eligible_candidate_label_count"] == 24
    assert join["joined_candidate_count"] == 24
    assert join["unjoined_candidate_label_count"] == 0
    assert (
        sum(
            value["candidate_count"] for value in row["original_model_rank_calibration"]
        )
        == 24
    )
    assert (
        sum(value["candidate_count"] for value in row["allocator_role_calibration"])
        == 24
    )
    assert {
        value["allocator_role"]: value["candidate_count"]
        for value in row["allocator_role_calibration"]
    } == {
        "calibrated_exploit": 6,
        "calibrated_frontier": 6,
        "epistemic_structural": 6,
        "structural_coverage": 6,
    }
    rank_one = next(
        value
        for value in row["original_model_rank_calibration"]
        if value["original_model_rank"] == 1
    )
    assert rank_one["candidate_count"] == 4
    assert rank_one["positive_individual_marginal_count"] == 4


def test_analyzer_separates_v9_model_suggestions_from_engine_reconciliation() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/timeloop_codesign/full_support_g6/"
        "jul21_v9r1_timeloop_deepseek_live",
        workload_id="timeloop_codesign_v2",
        model_profile="deepseek_v4_pro",
        replicate_seed=20_260_784,
    )

    reconciliation = row["selector_behavior"]["semantic_reconciliation"]
    assert reconciliation["call_count"] == 6
    assert reconciliation["call_coverage_rate"] == 1.0
    assert reconciliation["original_member_count"] == 48
    assert reconciliation["original_unique_member_count"] == 48
    assert reconciliation["duplicate_model_member_count"] == 0
    assert reconciliation["reconciled_member_count"] == 48
    assert reconciliation["retained_model_member_count"] == 31
    assert reconciliation["engine_inserted_member_count"] == 17
    assert reconciliation["engine_insertion_rate"] == pytest.approx(17 / 48)
    assert reconciliation["evaluated_member_count"] == 24
    assert reconciliation["evaluated_model_member_count"] == 15
    assert reconciliation["evaluated_engine_member_count"] == 9
    assert reconciliation["evaluated_engine_member_rate"] == pytest.approx(3 / 8)
    assert reconciliation["objective_blind_call_rate"] == 1.0
    assert reconciliation["workload_identifier_blind_call_rate"] == 1.0
    assert reconciliation["model_card_attribution_rewrite_count"] == 0
    assert reconciliation["origin_counts"] == {
        "engine_feasibility": 17,
        "model": 31,
    }

    outcomes = row["semantic_reconciliation_outcomes"]
    assert outcomes["reconciliation_call_count"] == 6
    assert outcomes["eligible_candidate_count"] == 24
    assert outcomes["joined_candidate_count"] == 24
    by_origin = {
        value["semantic_origin_group"]: value for value in outcomes["origin_group_rows"]
    }
    assert by_origin["engine"]["candidate_count"] == 9
    assert by_origin["engine"]["positive_individual_marginal_count"] == 5
    assert by_origin["engine"]["final_front_admission_count"] == 3
    assert by_origin["engine"][
        "descriptive_sum_individual_marginal_hypervolume"
    ] == pytest.approx(0.06990610906207617)
    assert by_origin["model"]["candidate_count"] == 15
    assert by_origin["model"]["positive_individual_marginal_count"] == 3
    assert by_origin["model"]["final_front_admission_count"] == 2
    assert by_origin["model"][
        "descriptive_sum_individual_marginal_hypervolume"
    ] == pytest.approx(0.0032883870145606897)
    assert (
        sum(
            value["candidate_count"]
            for value in outcomes["semantic_original_model_rank_rows"]
        )
        == 15
    )


def test_analyzer_preserves_v11_protected_global_source_provenance() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/jul22_v11_boils_deepseek_s20260787_live",
        workload_id="boils_abc_log2",
        model_profile="deepseek_v4_pro_streamlake_xhigh",
        replicate_seed=20_260_787,
    )

    reconciliation = row["selector_behavior"]["semantic_reconciliation"]
    assert reconciliation["call_count"] == 6
    assert reconciliation["origin_counts"]["engine_global_coverage"] == 2
    assert reconciliation["evaluated_engine_member_count"] == 2
    assert reconciliation["objective_blind_call_rate"] == 1.0
    assert reconciliation["workload_identifier_blind_call_rate"] == 1.0

    outcomes = row["semantic_reconciliation_outcomes"]
    by_origin = {
        value["semantic_origin_group"]: value for value in outcomes["origin_group_rows"]
    }
    assert by_origin["engine"]["candidate_count"] == 2


def test_analyzer_authenticates_v12_contextual_allocation_recourse() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "jul22_v12r2_boils_contextual_deepseek_s20260780_live",
        workload_id="boils_abc_log2",
        model_profile="deepseek_v4_pro_streamlake_xhigh",
        replicate_seed=20_260_780,
    )

    reconciliation = row["selector_behavior"]["semantic_reconciliation"]
    assert reconciliation["origin_counts"] == {
        "engine_contextual_allocation": 5,
        "engine_refill": 1,
        "model": 42,
    }
    projection = reconciliation["contextual_allocation_projection"]
    assert projection["call_count"] == 6
    assert projection["call_coverage_rate"] == 1.0
    assert projection["exact_call_count"] == 5
    assert projection["exact_call_rate"] == pytest.approx(5 / 6)
    assert projection["source_l1_deviation"] == 0
    assert projection["operator_l1_deviation"] == 2
    assert projection["requested_source_target_counts"] == {
        "engine": 5,
        "model": 19,
    }
    assert projection["realized_source_target_counts"] == {
        "engine": 5,
        "model": 19,
    }
    assert projection["requested_operator_target_counts"] == {
        "atomic": 11,
        "composite": 13,
    }
    assert projection["realized_operator_target_counts"] == {
        "atomic": 12,
        "composite": 12,
    }
    assert projection["objective_blind_call_rate"] == 1.0
    assert projection["workload_identifier_blind_call_rate"] == 1.0

    controller = row["contextual_search_controller"]
    assert controller["enabled"] is True
    assert controller["plan_count"] == 3
    assert controller["observation_count"] == 24
    assert controller["exact_source_realization_wave_rate"] == 1.0
    assert controller["exact_operator_realization_wave_rate"] == pytest.approx(2 / 3)
    assert {
        value["arm_id"]: (
            value["observation_count"],
            value["positive_marginal_utility_count"],
        )
        for value in controller["source_rows"]
    } == {"engine": (5, 1), "model": (19, 9)}
    assert [value["phase"] for value in controller["plan_rows"]] == [
        "basin_acquisition",
        "composition",
        "terminal_conversion",
    ]


def test_analyzer_joins_v14_frontier_targets_to_real_candidate_outcomes() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/jul22_v14_frontier_deepseek_s20260780_live",
        workload_id="boils_abc_log2",
        model_profile="deepseek_v4_pro_xhigh",
        replicate_seed=20_260_780,
    )

    behavior = row["selector_behavior"]
    assert behavior["frontier_target_call_count"] == 6
    assert behavior["frontier_target_enabled_rate"] == 1.0
    assert behavior["frontier_target_distinct_target_count"] == 6
    assert behavior["frontier_target_direction_counts"] == {
        "axis_1_extreme": 3,
        "balanced_tradeoff": 3,
    }
    assert behavior["frontier_target_future_outcome_leak_count"] == 0
    assert behavior["frontier_target_workload_identifier_consulted_count"] == 0
    assert behavior["frontier_target_model_or_provider_consulted_count"] == 0

    outcomes = row["frontier_target_outcomes"]
    assert outcomes["target_call_count"] == 6
    assert outcomes["joined_candidate_count"] == 24
    assert outcomes["scored_candidate_count"] == 24
    assert outcomes["improves_assigned_parent_count"] == 9
    assert outcomes["beats_prior_archive_best_count"] == 4
    by_direction = {
        value["target_direction_id"]: value for value in outcomes["direction_rows"]
    }
    assert by_direction["axis_1_extreme"]["final_front_admission_count"] == 3
    assert by_direction["balanced_tradeoff"]["final_front_admission_count"] == 0


def test_analyzer_measures_residual_aspiration_closure_in_v22_heat() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/engibench_heat2d/generic_campaign/"
        "jul22_heat2d_qwen_rhfp_v22_live_v1_20260722",
        workload_id="engibench_heat2d",
        model_profile="qwen_3_7_max_alibaba_xhigh",
        replicate_seed=20_260_716,
        arm="residual_hypervolume_frontier_planning_v22",
    )

    outcomes = row["frontier_target_outcomes"]
    assert outcomes["target_call_count"] == 6
    assert outcomes["residual_target_joined_candidate_count"] == 24
    assert outcomes["attains_or_dominates_residual_aspiration_count"] == 2
    assert outcomes["reduces_residual_aspiration_shortfall_count"] == 9
    assert outcomes[
        "median_residual_aspiration_shortfall_reduction_over_parent"
    ] == pytest.approx(0.0)
    attained = [
        value
        for value in outcomes["rows"]
        if value["attains_or_dominates_residual_aspiration"]
    ]
    assert {value["generation"] for value in attained} == {3}
    assert not any(
        value["attains_or_dominates_residual_aspiration"]
        for value in outcomes["rows"]
        if value["generation"] == 5
    )


def test_analyzer_recovers_frontier_context_and_empirical_memory_use() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "boils_common_m24_frontier_deepseek_s20260750_e2e_r1",
        workload_id="boils_abc",
        model_profile="deepseek",
        replicate_seed=20_260_750,
    )

    behavior = row["selector_behavior"]
    assert row["status"] == "completed_healthy"
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.088653125)
    assert behavior["frontier_context_call_count"] == 6
    assert behavior["frontier_context_enabled_rate"] == 1.0
    assert behavior["frontier_context_distinct_projection_count"] == 6
    assert behavior["frontier_context_dimension_counts"] == {"2": 6}
    assert behavior["frontier_context_projector_counts"] == {
        "authenticated_affine_frontier_context:v1": 6
    }
    assert behavior["frontier_context_future_outcome_leak_count"] == 0
    assert behavior["empirical_card_available_call_count"] == 2
    assert behavior["empirical_card_selected_citation_member_count"] == 4
    assert behavior["empirical_card_evaluated_citation_member_count"] == 3
    assert behavior["empirical_card_selected_exact_target_member_count"] == 1
    assert behavior["empirical_card_evaluated_exact_target_member_count"] == 1
    assert (
        behavior["empirical_card_selected_cross_target_generalization_member_count"]
        == 3
    )
    assert (
        behavior["empirical_card_evaluated_cross_target_generalization_member_count"]
        == 2
    )


def test_analyzer_recovers_test_only_advisory_memory_without_causal_credit() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "grid_boils_abc_deepseek_s20260782_v16s0_v6q1e_live",
        workload_id="boils_abc",
        model_profile="deepseek_v4_pro_streamlake_xhigh",
        replicate_seed=20_260_782,
        arm="operator_stratified_v6",
    )

    memory = row["memory_and_reflection"]
    assert memory["reflected_entry_count"] == 4
    assert memory["reflected_entry_source"] == "summary.reflection_records"
    assert memory["reflected_retrievable_count"] == 0
    assert memory["reflected_lifecycle_state_counts"] == {"quarantined": 4}
    assert memory["advisory_memory_lane_count"] == 2
    assert memory["advisory_memory_selected_card_count"] == 2
    assert memory["advisory_memory_exact_parent_match_count"] == 0
    assert memory["advisory_memory_exact_replay_authorized_count"] == 0
    assert memory["advisory_memory_causal_claim_allowed"] is False
    assert memory["memory_supported_action_count"] == 1
    assert memory["memory_supported_action_positive_reward_count"] == 0
    supported = memory["memory_action_performance_rows"][0]
    assert supported["joint_wave_reward"] is None
    assert supported["joint_wave_positive_reward"] is None
    assert supported["joint_wave_reward_source"] == ("advisory_exposure_no_wave_credit")
    assert memory["memory_action_performance_causal_claim_allowed"] is False


def test_analyzer_separates_memory_wave_credit_from_semantic_falsification() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/engibench_heat2d/generic_campaign/"
        "grid_heat2d_deepseek_json_s20260750_"
        "frontier_v4_deepseek_json_heat_e2e_r1_live",
        workload_id="heat2d",
        model_profile="deepseek_json",
        replicate_seed=20_260_750,
    )

    evaluation = row["evaluation"]
    memory = row["memory_and_reflection"]
    assert evaluation["physical_evaluator_latency_count"] == 38
    assert evaluation["physical_evaluator_latency_median_s"] == pytest.approx(
        14.8258468905
    )
    assert memory["reflected_entry_count"] == 2
    assert memory["reflected_retrievable_count"] == 0
    assert memory["reflected_lifecycle_state_counts"] == {"deprecated": 2}
    assert memory["memory_trial_count"] == 6
    assert memory["adaptive_score_consumption"] is False
    assert memory["causal_claim_allowed"] is False
    assert memory["memory_assignment_credit_count"] == 6
    assert memory["memory_assignment_positive_wave_reward_count"] == 6
    generation_five_credit = next(
        value
        for value in memory["memory_assignment_credit_by_generation"]
        if value["generation"] == 5
    )
    assert generation_five_credit["credit_count"] == 2
    assert generation_five_credit["positive_wave_reward_count"] == 2
    assert generation_five_credit["total_wave_reward"] == pytest.approx(
        0.0018577446774189488
    )
    assert memory["semantic_audit_verdict_counts"] == {"contradicted": 2}
    assert memory["lifecycle_request_state_counts"] == {"deprecated": 2}


def test_analyzer_preserves_completed_evolution_after_reporting_failure() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/engibench_heat2d/generic_campaign/"
        "grid_heat2d_v3_qwen_s20260716_"
        "typed_reflection_repair_e2e_r1_live",
        workload_id="heat2d",
        model_profile="qwen",
        replicate_seed=20_260_716,
    )

    # Never relabel the failed convenience summary as a healthy run.
    assert row["status"] == "failed"
    assert row["health_all_true"] is None
    assert row["completion_evidence"] == {
        "summary_status": "failed",
        "summary_health_available": False,
        "evolution_completed": True,
        "runtime_released": True,
        "durable_campaign_complete": True,
        "wall_s": pytest.approx(889.804418267),
        "wall_time_source": (
            "campaign_events.observation.max_monotonic_ns_since_execution_start"
        ),
    }
    assert row["evaluation"]["unique_evaluations"] == 38
    assert row["evaluation"]["generations"] == 6
    assert row["provider"]["logical_calls"] == 7
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.8789746104193821)


def test_analyzer_keeps_censored_quality_and_retry_failure_separate() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "grid_boils_abc_gpt_oss_20b_s20260761_qf_v9_r1_oss20_boils_live",
        workload_id="boils_abc",
        model_profile="gpt_oss_20b",
        replicate_seed=20_260_761,
    )

    assert row["status"] == "failed_before_completion"
    assert row["health_all_true"] is None
    assert row["quality"]["is_final"] is False
    assert row["quality"]["endpoint_kind"] == "censored_latest_sealed_archive"
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.08830729166666668)
    assert row["evaluation"]["generations"] == 4
    assert row["evaluation"]["unique_evaluations"] == 42
    assert row["failure_endpoint"]["campaign_failure_type"] == (
        "QueuedStructuredGenerationError"
    )
    assert row["failure_endpoint"]["failed_generation"] == 5
    assert row["failure_endpoint"]["queue_task_status_counts"] == {
        "attempts_exhausted": 1,
        "cancelled": 1,
        "succeeded": 5,
    }
    exhausted = row["failure_endpoint"]["failed_queue_tasks"][0]
    assert [value["failure_kind"] for value in exhausted["attempts"]] == [
        "output_invalid",
        "output_invalid",
        "provider_unavailable",
    ]
    assert exhausted["attempts"][1]["request_variant"] == "schema_repair_v3"
    assert exhausted["attempts"][2]["will_retry"] is False


def test_analyzer_retains_a_censored_pre_stage_provider_failure() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "grid_boils_abc_gpt_oss_20b_s20260770_v10r1_live",
        workload_id="boils_abc",
        model_profile="gpt_oss_20b_groq_high",
        replicate_seed=20_260_770,
    )

    assert row["status"] == "failed_before_completion"
    assert row["completion_evidence"]["durable_campaign_complete"] is False
    assert row["evaluation"] == {
        **row["evaluation"],
        "candidate_occurrences": 2,
        "unique_evaluations": 2,
        "cache_reuse_occurrences": 0,
        "scored_candidates": 2,
        "typed_candidate_infeasible": 0,
        "runtime_failures": 0,
        "generations": 0,
    }
    assert row["quality"]["endpoint_kind"] == "censored_pre_stage_seed_archive"
    assert row["quality"]["seed_hypervolume"] == pytest.approx(0.046475)
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.046475)
    assert row["quality"]["absolute_gain"] == 0.0
    assert row["failure_endpoint"]["last_sealed_generation"] == 0
    assert row["failure_endpoint"]["failed_generation"] == 1
    assert row["failure_endpoint"]["queue_task_status_counts"] == {
        "attempts_exhausted": 1,
        "cancelled": 1,
    }
    assert row["failure_endpoint"]["queue_attempt_failure_kind_counts"] == {
        "provider_unavailable": 1,
        "rate_limited": 2,
    }


def test_analyzer_separates_unsealed_real_evaluations_from_censored_endpoint() -> None:
    row = analyze_run(
        LOGS / "boils_abc/generic_campaign/"
        "grid_boils_abc_gpt_oss_20b_serial_s20260770_v11r1_live",
        workload_id="boils_abc",
        model_profile="gpt_oss_20b_groq_high_serial",
        replicate_seed=20_260_770,
    )

    assert row["status"] == "failed_before_completion"
    assert row["evaluation"]["unique_evaluations"] == 2
    assert row["evaluation"]["observed_physical_evaluations"] == 6
    assert row["evaluation"]["unsealed_physical_evaluations"] == 4
    assert row["evaluation"]["scored_candidates"] == 2
    assert row["evaluation"]["observed_scored_candidates"] == 6
    assert row["quality"]["endpoint_kind"] == "censored_pre_stage_seed_archive"
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.046475)
    assert row["quality"]["unsealed_observed_hypervolume"] == pytest.approx(
        0.07735104166666666
    )
    assert (
        sum(
            value["positive_individual_marginal_count"]
            for value in row["model_rank_calibration"]
        )
        == 2
    )
    assert row["failure_endpoint"]["queue_attempt_failure_kind_counts"] == {
        "provider_unavailable": 2,
        "rate_limited": 1,
    }


def test_v9_analysis_exposes_forecast_calibration_and_noncausal_memory_assay() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/engibench_heat2d/generic_campaign/"
        "grid_heat2d_gpt_sol_s20260761_qf_v9_r1_gptsol_heat_live",
        workload_id="heat2d",
        model_profile="gpt_5.6_sol_xhigh",
        replicate_seed=20_260_761,
    )

    assert row["quality"]["is_final"] is True
    assert row["quality"]["final_hypervolume"] == pytest.approx(0.8762061422258063)
    forecast = row["forecast_calibration"]
    assert forecast["evaluated_forecast_member_count"] == 48
    assert forecast["unscorable_missing_event_member_count"] == 0
    assert forecast["effect_prediction_count"] == 96
    assert forecast["direction_accuracy"] == pytest.approx(0.7752808988764045)
    assert forecast["high_confidence_direction_accuracy"] == 1.0
    assert forecast["high_confidence_direction_error_count"] == 0

    behavior = row["selector_behavior"]
    assert behavior["evaluated_card_citation_member_count"] == 41
    assert behavior["evaluated_card_citation_without_exact_finite_target_count"] == 40

    memory = row["memory_and_reflection"]
    assert memory["matched_memory_control_block_count"] == 1
    assert memory["matched_memory_total_active_minus_neutral"] == pytest.approx(
        -0.00069233557419357
    )
    assert memory["matched_memory_identified_effect_count"] == 0
    assert memory["matched_memory_causal_claim_allowed"] is False


def test_forecast_analysis_censors_exact_pre_simulator_infeasibility() -> None:
    row = analyze_run(
        LOGS / "benchmark_q1/timeloop_codesign/full_support_g6/"
        "grid_timeloop_v2_gpt_sol_s20260770_v10r1_live",
        workload_id="timeloop_codesign_v2",
        model_profile="gpt_5.6_sol_xhigh",
        replicate_seed=20_260_770,
    )

    assert row["status"] == "completed_unhealthy"
    assert row["completion_evidence"]["durable_campaign_complete"] is True
    assert row["evaluation"]["unique_evaluations"] == 38
    assert row["evaluation"]["typed_candidate_infeasible"] == 1
    assert row["forecast_calibration"]["evaluated_forecast_member_count"] == 24
    forecast = row["forecast_calibration"]
    assert forecast["unscorable_forecast_member_count"] == 1
    assert forecast["unscorable_missing_event_member_count"] == 0
    assert forecast["unscorable_invalid_candidate_member_count"] == 1
    assert forecast["unscorable_objective_payload_member_count"] == 0
    assert forecast["effect_prediction_count"] == 69
