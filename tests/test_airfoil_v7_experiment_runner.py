"""Provider/CFD-free proof of the exact Airfoil-v7 orchestration block."""

from __future__ import annotations

import pytest

from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    DIAGNOSTIC_SLOT_IDS,
    DIAGNOSTIC_SHAPE_OPTION_ID,
    DIAGNOSTIC_TRIM_OPTION_ID,
    HELD_OUT_PARENT_CANDIDATE_SHA256,
    HELD_OUT_PARENT_NONCE,
    HELD_OUT_PARENT_TYPED_SHA256,
    MEMORY_CARD_MASK,
    MAX_OUTPUT_TOKENS,
    MODEL,
    REFLECTION_INSIGHT_CONTRACT,
    SHAM_INSIGHT_CONTRACT,
    SHAM_OPTION_ID,
    STRUCTURED_OUTPUT_BUDGET_POLICY,
    materialize_held_out_parent,
    run_offline_verification_sync,
    validation_record,
)
from examples.development import run_airfoil_v7_reflective_feedback as runner


@pytest.fixture(scope="module")
def offline_verification() -> dict[str, object]:
    result = run_offline_verification_sync()
    assert result["overall_pass"] is True
    return result


def test_outcome_blind_held_out_parent_is_frozen_nonce_zero() -> None:
    materialized = materialize_held_out_parent()
    assert materialized.nonce == HELD_OUT_PARENT_NONCE == 0
    assert materialized.candidate_sha256 == HELD_OUT_PARENT_CANDIDATE_SHA256
    assert materialized.typed_configuration_sha256 == HELD_OUT_PARENT_TYPED_SHA256
    assert materialized.rejected_nonces == ()
    assert materialized.validation.area_ratio == pytest.approx(1.0019623987565598)


def test_validate_mode_freezes_budget_catalogs_and_policy_separation() -> None:
    record = validation_record()
    assert record["provider_io_performed"] is False
    assert record["cfd_calls"] == 0
    assert record["credentials_read"] is False
    assert record["live_authorized"] is False
    assert record["model"] == MODEL == "deepseek/deepseek-v4-pro"
    assert MAX_OUTPUT_TOKENS == 384_000
    assert record["structured_output_budget_policy"] == {
        "policy_id": STRUCTURED_OUTPUT_BUDGET_POLICY.policy_id,
        "policy_version": STRUCTURED_OUTPUT_BUDGET_POLICY.policy_version,
        "proposal_max_output_tokens": 384_000,
        "reflection_max_output_tokens": 384_000,
        "ceiling_semantics": "provider_maximum_not_expected_usage",
    }
    assert record["budget"] == {
        "max_unique_evaluations": 7,
        "max_logical_llm_calls": 7,
        "max_generations": 2,
    }
    assert {
        key: value["option_count"] for key, value in record["catalogs"].items()
    } == {
        "diagnostic_shape": 16,
        "diagnostic_trim": 64,
        "held_out_union": 80,
    }
    assert record["policy_separation"]["distinct"] is True
    assert record["reflection_insight_contract"] == (
        REFLECTION_INSIGHT_CONTRACT.to_record()
    )
    assert REFLECTION_INSIGHT_CONTRACT.allowed_option_ids == (
        DIAGNOSTIC_SHAPE_OPTION_ID,
        DIAGNOSTIC_TRIM_OPTION_ID,
    )
    assert record["sham_insight_contract"] == SHAM_INSIGHT_CONTRACT.to_record()
    assert (
        record["waves"]["g1"]["slot_ids"] == list(DIAGNOSTIC_SLOT_IDS) == ["D-S", "D-T"]
    )
    assert record["waves"]["g1"]["prospective_option_ids"] == [
        DIAGNOSTIC_SHAPE_OPTION_ID,
        DIAGNOSTIC_TRIM_OPTION_ID,
    ]
    assert record["waves"]["g2"]["prospective_sham_option_id"] == SHAM_OPTION_ID
    assert set(record["waves"]["g2"]["clean_early_stop_reason_codes"]) == {
        "reflected_card_batch_unavailable",
        "equal_origin_scores",
        "structurally_inapplicable_assignment",
    }


def test_full_path_is_exactly_seven_calls_seven_evaluations_and_matched_prompts(
    offline_verification: dict[str, object],
) -> None:
    full = offline_verification["full_seven_call_path"]
    assert full["accounting"] == {
        "seed_evaluations": 2,
        "unique_evaluations": 7,
        "logical_llm_calls": 7,
        "proposal_calls": 5,
        "reflection_calls": 2,
        "generation_widths": [2, 3],
        "feedback_calls_by_generation": [2, 0],
    }
    assert full["g1_rewards"] == [1.0, -1.0]
    assert full["concurrency"] == {
        "planned_g1_width": 2,
        "planned_g2_width": 3,
        "max_generator_in_flight": 3,
        "max_evaluator_in_flight": 1,
        "concurrency_ready": True,
    }
    prompt = full["prompt_difference"]
    assert prompt["mask_sentinel"] == MEMORY_CARD_MASK
    assert len(set(prompt["raw_prompt_sha256s"])) == 3
    assert len(set(prompt["masked_prompt_sha256s"])) == 1
    assert prompt["invariant_pass"] is True
    assert full["trace_checks_pass"] is True
    assert full["trace_checks"][
        "diagnostics_execute_prospective_exact_actions"
    ] is True
    assert full["trace_checks"][
        "held_out_actions_equal_card_exact_option_ids"
    ] is True
    union_contracts = full["catalogs"]["held_out_union_contract_sha256s"]
    assert len(union_contracts) == 3
    assert len(set(union_contracts)) == 1
    assert full["trace_checks"]["held_out_arms_share_one_union_contract"] is True
    assert full["reflection_insight_contract"] == (
        REFLECTION_INSIGHT_CONTRACT.to_record()
    )


def test_equal_origin_scores_stop_before_g2_without_spending_reserved_calls(
    offline_verification: dict[str, object],
) -> None:
    tied = offline_verification["equal_score_early_stop_path"]
    assert tied["accounting"] == {
        "seed_evaluations": 2,
        "unique_evaluations": 4,
        "logical_llm_calls": 4,
        "proposal_calls": 2,
        "reflection_calls": 2,
        "generation_widths": [2, 0],
        "feedback_calls_by_generation": [2, 0],
    }
    assert tied["g1_rewards"] == [0.0, 0.0]
    assert "tied origin scores" in tied["early_stop_reason"]
    assert tied["early_stop_reason_code"] == "equal_origin_scores"
    assert tied["prompt_difference"]["raw_prompt_sha256s"] == []
    assert tied["concurrency"]["max_generator_in_flight"] == 2
    assert tied["concurrency"]["max_evaluator_in_flight"] == 1


def test_live_mode_is_fail_closed_before_queue_or_credentials(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("production dependencies must remain untouched")

    monkeypatch.setattr(runner, "production_live_dependencies", forbidden_factory)
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--live"])

    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--launch-manifest is required for the selected mode" in captured.err
    assert factory_calls == 0
