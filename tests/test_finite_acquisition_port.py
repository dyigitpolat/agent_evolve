from __future__ import annotations

import hashlib

import pytest

from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionRequest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _request() -> FiniteAcquisitionRequest:
    objectives = (
        FiniteAcquisitionObjective("cost", "min", 0.0, 2.0),
        FiniteAcquisitionObjective("quality", "max", 2.0, 0.0),
    )
    observation = FiniteAcquisitionObservation(
        candidate_id="observed-1",
        configuration_sha256=_sha("observed-1"),
        features=(0.25, 0.75),
        objectives=(("cost", 1.0), ("quality", 1.0)),
    )
    candidates = (
        FiniteAcquisitionCandidate("candidate-1", _sha("candidate-1"), (0.0, 0.0)),
        FiniteAcquisitionCandidate("candidate-2", _sha("candidate-2"), (1.0, 1.0)),
    )
    return FiniteAcquisitionRequest(
        campaign_scope_sha256=_sha("scope"),
        cutoff_index=1,
        batch_size=2,
        seed=7,
        objectives=objectives,
        observations=(observation,),
        candidates=candidates,
    )


def test_finite_acquisition_request_is_replay_identical() -> None:
    first = _request()
    second = _request()

    assert first == second
    assert first.request_sha256 == second.request_sha256
    assert first.objectives[0].maximize_value(2.0) == 0.0
    assert first.objectives[0].maximize_value(0.0) == 1.0
    assert first.objectives[1].maximize_value(0.0) == 0.0
    assert first.objectives[1].maximize_value(2.0) == 1.0


def test_finite_acquisition_rejects_duplicate_feature_rows() -> None:
    request = _request()
    duplicate = FiniteAcquisitionCandidate(
        "candidate-3",
        _sha("candidate-3"),
        request.candidates[0].features,
    )

    with pytest.raises(ValueError, match="features must be unique"):
        FiniteAcquisitionRequest(
            campaign_scope_sha256=request.campaign_scope_sha256,
            cutoff_index=1,
            batch_size=2,
            seed=7,
            objectives=request.objectives,
            observations=request.observations,
            candidates=(request.candidates[0], duplicate),
        )
