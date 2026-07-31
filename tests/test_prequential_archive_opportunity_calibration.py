from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
    ArchiveOpportunityCalibrationObservation,
    ArchiveOpportunityCalibrationRequest,
    HierarchicalPrequentialArchiveOpportunityCalibration,
    validate_archive_opportunity_calibration_port,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _request(
    *,
    label: str,
    decision_index: int = 2,
    lane_id: str = "model.residual_local_exploit",
    operator_id: str = "components:1",
    raw_acquisition: float = 0.01,
    prefix_gain: float = 0.04,
    prefix_action_count: int = 4,
) -> ArchiveOpportunityCalibrationRequest:
    return ArchiveOpportunityCalibrationRequest(
        context=ArchiveOpportunityActionContext(
            action_sha256=_sha(f"action:{label}"),
            decision_index=decision_index,
            lane_id=lane_id,
            operator_id=operator_id,
            native_rank=1,
            lane_size=4,
            prior_score=0.75,
            parent_generated_in_current_run=decision_index > 1,
        ),
        forecast_reliability=0.8,
        raw_adverse_gain=raw_acquisition * 0.8,
        raw_central_gain=raw_acquisition,
        raw_favorable_gain=raw_acquisition * 1.2,
        raw_acquisition_value=raw_acquisition,
        prefix_gain=prefix_gain,
        prefix_action_count=prefix_action_count,
    )


def _observation(
    *,
    ordinal: int,
    realized: float,
    decision_index: int = 2,
    lane_id: str = "model.residual_local_exploit",
    operator_id: str = "components:1",
    raw_acquisition: float = 0.01,
) -> ArchiveOpportunityCalibrationObservation:
    return ArchiveOpportunityCalibrationObservation(
        request=_request(
            label=f"observation:{ordinal}",
            decision_index=decision_index,
            lane_id=lane_id,
            operator_id=operator_id,
            raw_acquisition=raw_acquisition,
        ),
        realized_conditional_gain=realized,
        decision_sha256=_sha(f"decision:{ordinal}"),
        outcome_sha256=_sha(f"outcome:{ordinal}"),
        evidence_cutoff_ordinal=ordinal,
    )


def _calibrator(
    observations: tuple[ArchiveOpportunityCalibrationObservation, ...],
    **kwargs: object,
) -> HierarchicalPrequentialArchiveOpportunityCalibration:
    return HierarchicalPrequentialArchiveOpportunityCalibration(
        observations=tuple(
            sorted(
                observations,
                key=lambda value: value.observation_sha256,
            )
        ),
        maximum_evidence_cutoff_ordinal=max(
            value.evidence_cutoff_ordinal for value in observations
        ),
        **kwargs,
    )


def test_calibrator_abstains_without_global_support() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            realized=0.005 * index,
            raw_acquisition=0.004 * index,
        )
        for index in range(1, 4)
    )
    result = _calibrator(observations).calibrate(
        _request(label="cold")
    )

    assert result.abstained is True
    assert result.abstention_reason == "insufficient_global_support"
    assert result.stratum_support_count == 3
    assert result.result_sha256 == result.to_record()["result_sha256"]


def test_calibrator_abstains_for_negative_prequential_rank_skill() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            raw_acquisition=0.001 * index,
            realized=0.001 * (7 - index),
        )
        for index in range(1, 7)
    )
    result = _calibrator(observations).calibrate(
        _request(label="anti-skilled", raw_acquisition=0.003)
    )

    assert result.prequential_rank_skill == pytest.approx(-1.0)
    assert result.abstained is True
    assert (
        result.abstention_reason
        == "nonpositive_prequential_rank_skill"
    )


def test_calibrator_recommends_with_positive_skill_and_support() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            raw_acquisition=0.001 * index,
            realized=0.0005 * index,
        )
        for index in range(1, 7)
    )
    calibrator = _calibrator(observations)
    result = calibrator.calibrate(
        _request(label="skilled", raw_acquisition=0.004)
    )

    assert result.prequential_rank_skill == pytest.approx(1.0)
    assert result.abstained is False
    assert result.lower_positive_probability > 0.0
    assert result.lower_expected_gain > 0.0
    assert result.calibrated_acquisition_value > 0.0
    assert result.calibrated_upper_gain >= result.lower_expected_gain
    assert validate_archive_opportunity_calibration_port(calibrator) == (
        calibrator.calibration_id,
        calibrator.calibration_version,
        calibrator.definition_sha256,
    )


def test_calibrator_rejects_forecast_scale_outside_prior_support() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            raw_acquisition=0.001 * index,
            realized=0.0005 * index,
        )
        for index in range(1, 7)
    )
    result = _calibrator(
        observations,
        maximum_support_log_distance=0.5,
    ).calibrate(
        _request(label="outlier", raw_acquisition=10.0)
    )

    assert result.support_log_distance > 0.5
    assert result.abstained is True
    assert result.abstention_reason == "forecast_scale_out_of_support"


def test_calibrator_prefers_the_most_specific_supported_stratum() -> None:
    exact = tuple(
        _observation(
            ordinal=index,
            lane_id="model.residual_interaction",
            operator_id="components:2",
            raw_acquisition=0.001 * index,
            realized=0.0005 * index,
        )
        for index in range(1, 5)
    )
    global_only = tuple(
        _observation(
            ordinal=index,
            lane_id="model.residual_counterfactual_coverage",
            operator_id="components:3",
            raw_acquisition=0.001 * index,
            realized=0.0002 * index,
        )
        for index in range(5, 9)
    )
    result = _calibrator((*exact, *global_only)).calibrate(
        _request(
            label="exact-cell",
            lane_id="model.residual_interaction",
            operator_id="components:2",
            raw_acquisition=0.003,
        )
    )

    assert result.selected_stratum == "stage_lane_operator"
    assert result.stratum_support_count == 4


def test_calibrator_rejects_observation_after_sealed_cutoff() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            raw_acquisition=0.001 * index,
            realized=0.0005 * index,
        )
        for index in range(1, 7)
    )

    with pytest.raises(
        ValueError,
        match="observation crosses the sealed evidence cutoff",
    ):
        HierarchicalPrequentialArchiveOpportunityCalibration(
            observations=tuple(
                sorted(
                    observations,
                    key=lambda value: value.observation_sha256,
                )
            ),
            maximum_evidence_cutoff_ordinal=5,
        )


def test_definition_and_results_are_deterministic() -> None:
    observations = tuple(
        _observation(
            ordinal=index,
            raw_acquisition=0.001 * index,
            realized=0.0005 * index,
        )
        for index in range(1, 7)
    )
    left = _calibrator(observations)
    right = _calibrator(tuple(reversed(observations)))
    request = _request(label="deterministic", raw_acquisition=0.004)

    assert left.observation_snapshot_sha256 == (
        right.observation_snapshot_sha256
    )
    assert left.definition_sha256 == right.definition_sha256
    assert left.calibrate(request) == right.calibrate(request)
