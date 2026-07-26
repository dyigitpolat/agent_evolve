from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.domain.ids import LLMCallId
from examples.development import airfoil_v7_two_stage_agent_evolution as airfoil
from examples.development import (
    run_airfoil_v7_v5_sealed_end_to_end_replay as replay,
)
from examples.benchmarks.engibench_airfoil.v7_finite_oracle import (
    OBJECTIVE_NAME,
    PARENT_METRICS,
    VIOLATION_NAME,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _forecast(option_id: str, objective: float, violation: float):
    objective_delta = objective - float(PARENT_METRICS[OBJECTIVE_NAME])
    violation_delta = violation - float(PARENT_METRICS[VIOLATION_NAME])
    return SimpleNamespace(
        option_id=option_id,
        probability_valid=1.0,
        metric_forecasts=(
            SimpleNamespace(
                metric_id=airfoil.OBJECTIVE_METRIC_ID,
                p10_delta=objective_delta,
                p50_delta=objective_delta,
                p90_delta=objective_delta,
            ),
            SimpleNamespace(
                metric_id=airfoil.VIOLATION_METRIC_ID,
                p10_delta=violation_delta,
                p50_delta=violation_delta,
                p90_delta=violation_delta,
            ),
        ),
    )


def _arm_replay(
    arm: str,
    selected: tuple[str, str, str],
    forecasts: tuple[SimpleNamespace, ...],
):
    members = tuple(SimpleNamespace(option_id=value) for value in selected)
    return SimpleNamespace(
        arm=arm,
        block_request=SimpleNamespace(
            request=SimpleNamespace(
                metric_scales=(
                    SimpleNamespace(
                        metric_id=airfoil.OBJECTIVE_METRIC_ID,
                        delta_scale=0.001,
                    ),
                    SimpleNamespace(
                        metric_id=airfoil.VIOLATION_METRIC_ID,
                        delta_scale=0.005,
                    ),
                )
            )
        ),
        forecasts=SimpleNamespace(forecasts=forecasts),
        execution=SimpleNamespace(
            result=SimpleNamespace(
                decision=SimpleNamespace(members=members),
            )
        ),
    )


def test_posthoc_analyzer_uses_exact_block_cardinality_ranks_and_overlap() -> None:
    rows: list[dict[str, object]] = []
    forecasts: list[SimpleNamespace] = []
    for index in range(80):
        option_id = f"action.{index:02d}"
        objective = 1.0 + index * 0.00001
        violation = 0.4900 + index * 0.0001
        rows.append(
            {
                "option_id": option_id,
                "family": "alpha" if index % 2 == 0 else "beta",
                "objectives": {OBJECTIVE_NAME: objective},
                "violations": {VIOLATION_NAME: violation},
            }
        )
        if index < 18:
            forecasts.append(_forecast(option_id, objective, violation))

    eligible = tuple(f"action.{index:02d}" for index in range(18))
    arms = (
        _arm_replay("m", eligible[0:3], tuple(forecasts)),
        _arm_replay("p", eligible[3:6], tuple(forecasts)),
        _arm_replay(
            "n",
            (eligible[2], eligible[6], eligible[7]),
            tuple(forecasts),
        ),
    )
    result = replay._posthoc_analysis(
        oracle_result={"results": rows},
        eligible_option_ids=eligible,
        arm_replays=arms,  # type: ignore[arg-type]
    )

    assert result["authenticated_subset_size"] == 18
    assert result["three_set_count"] == 816
    assert result["block_optimal_three_set"]["option_ids"] == list(eligible[0:3])
    assert result["primary_best_within_subset_rank_contrasts_positive_favors_m"] == {
        "p_minus_m": 3,
        "n_minus_m": 2,
    }
    assert result["overlap"]["m_n"] == {
        "intersection_count": 1,
        "union_count": 5,
        "jaccard": 0.2,
        "shared_option_ids": [eligible[2]],
    }
    for arm in ("m", "p", "n"):
        calibration = result["arms"][arm][
            "calibration_over_all_18_block_eligible_actions"
        ]
        assert calibration["eligible_action_count"] == 18
        assert calibration["metric_cell_count"] == 36
        assert calibration["p50_normalized_mae"] == pytest.approx(0.0)
        assert calibration["direction_accuracy"] == pytest.approx(1.0)
        assert calibration["p10_p90_coverage"] == pytest.approx(1.0)
        assert calibration["validity_brier_all_cached_actions_valid"] == 0.0
        assert calibration["predicted_actual_order_spearman"] == pytest.approx(1.0)


def test_joint_failure_preserves_saturated_set_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use log failure for claims when every complement utility rounds to one."""

    row_by_id = {
        "shared.a": {"quality": 30.0},
        "shared.b": {"quality": 30.0},
        "n": {"quality": 30.0},
        "m": {"quality": 12.0},
        "p": {"quality": -1.0},
    }
    monkeypatch.setattr(
        replay,
        "_actual_member_quality",
        lambda row: float(row["quality"]),
    )
    sets = {
        "n": ("shared.a", "shared.b", "n"),
        "m": ("shared.a", "shared.b", "m"),
        "p": ("shared.a", "shared.b", "p"),
    }

    failures = {
        arm: replay._actual_set_joint_failure(option_ids, row_by_id)
        for arm, option_ids in sets.items()
    }
    log_failures = {
        arm: replay._actual_set_log_joint_failure(option_ids, row_by_id)
        for arm, option_ids in sets.items()
    }
    utilities = {
        arm: replay._actual_set_utility(option_ids, row_by_id)
        for arm, option_ids in sets.items()
    }

    assert all(value > 0.0 for value in failures.values())
    assert failures["n"] < failures["m"] < failures["p"]
    assert log_failures["n"] < log_failures["m"] < log_failures["p"]
    assert utilities == {"n": 1.0, "m": 1.0, "p": 1.0}
    assert failures["m"] - failures["n"] > 0.0
    assert failures["p"] - failures["n"] > 0.0


def test_assignment_is_prospective_to_replay_and_binds_physical_blocks() -> None:
    block_requests = tuple(
        SimpleNamespace(
            block_call_id=LLMCallId(f"call_replay_block_{index}"),
            block_request_sha256=_sha(f"block-request-{index}"),
        )
        for index in range(3)
    )
    bundle = SimpleNamespace(selected_block_requests=block_requests)
    commitment = _sha("sealed-replay-experiment")

    first = replay._treatment_assignment(bundle, commitment)  # type: ignore[arg-type]
    second = replay._treatment_assignment(bundle, commitment)  # type: ignore[arg-type]

    assert first == second
    assert first.experiment_commitment_sha256 == commitment
    assert tuple(
        value.treatment_id.value for value in first.occurrence_input_order
    ) == ("m", "p", "n")
    assert tuple(
        value.call_identity for value in first.occurrence_input_order
    ) == tuple(value.block_call_id.value for value in block_requests)
    assert tuple(
        value.request_identity_sha256 for value in first.occurrence_input_order
    ) == tuple(value.block_request_sha256 for value in block_requests)
    assert sorted(first.ordinal_permutation) == [0, 1, 2]
    assert first.content_blinding_claimed is False


def test_harness_is_provider_and_credential_free_and_historical_block_bound() -> None:
    source = Path(replay.__file__).read_text(encoding="utf-8")

    assert "OPENROUTER_API_KEY" not in source
    assert "load_dotenv" not in source
    assert "create_progress_aware_openrouter_runner" not in source
    assert replay.EXPECTED_ELIGIBLE_COUNT == 18
    assert replay.EXPECTED_CANDIDATE_SCORES_PER_ARM == 51
    assert replay.EXPECTED_THREE_SET_COUNT == 816
    assert replay.EXPECTED_LOGICAL_EVALUATION_SLOTS == 9
    assert "posthoc" in replay.REPLAY_SCOPE
    assert "causal" in replay.REPLAY_SCOPE
