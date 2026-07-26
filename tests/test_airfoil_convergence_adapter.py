"""No-CFD probes for the convergence-qualified Airfoil adapter boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    AirfoilConvergenceEvaluationError,
    ConvergenceQualifiedAirfoilPanelEvaluator,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    AirfoilEvaluationError,
    AirfoilPanelEvaluation,
    AirfoilPanelEvaluator,
    candidate_sha256,
)


CANDIDATE = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [0.0] * 10,
    "lower_coefficients": [0.0] * 10,
    "alpha_deg": [2.5, 2.5, 2.5],
}


def _bare_evaluator(tmp_path):
    evaluator = object.__new__(ConvergenceQualifiedAirfoilPanelEvaluator)
    evaluator.settings = SimpleNamespace(output_root=tmp_path)
    evaluator._run_id = "unit-run"
    return evaluator


def _receipt_path(tmp_path):
    return tmp_path / "unit-run" / f"{candidate_sha256(CANDIDATE)}.json"


def test_preserves_durable_authoritative_failure_receipt(tmp_path, monkeypatch) -> None:
    receipt_path = _receipt_path(tmp_path)
    receipt_path.parent.mkdir()
    receipt = {
        "schema_version": 2,
        "evaluator_id": V2_EVALUATOR_ID,
        "status": "candidate_invalid",
        "failure_classification": "authoritative_solver_failure",
        "evaluator_calls": 1,
        "evaluator_evidence": {
            "authoritative_status": {"solve_failed": True, "fatal_fail": False}
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def fail(_self, _config):
        raise AirfoilEvaluationError("solver failed", candidate_invalid=True)

    monkeypatch.setattr(AirfoilPanelEvaluator, "evaluate", fail)
    with pytest.raises(AirfoilConvergenceEvaluationError) as captured:
        _bare_evaluator(tmp_path).evaluate(CANDIDATE)
    assert captured.value.candidate_invalid is True
    assert captured.value.record_path == receipt_path
    assert captured.value.record == receipt


def test_accepts_only_success_receipts_with_three_evidence_objects(
    tmp_path, monkeypatch
) -> None:
    receipt_path = _receipt_path(tmp_path)
    evidence = {
        "contract_id": EVIDENCE_CONTRACT_ID,
        "evaluator_id": ADFLOW_EVALUATOR_ID,
        "accepted": True,
        "witness": {"schema_version": 1},
    }
    record = {
        "schema_version": 2,
        "evaluator_id": V2_EVALUATOR_ID,
        "status": "evaluated",
        "points": [{"evaluator_evidence": evidence} for _ in range(3)],
    }
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(record), encoding="utf-8")
    expected = AirfoilPanelEvaluation(
        candidate_sha256=candidate_sha256(CANDIDATE),
        objective_values={"mean_drag_coefficient": 0.02, "max_lift_target_error": 0.1},
        wall_seconds=20.0,
        record_path=receipt_path,
        record=record,
    )

    monkeypatch.setattr(AirfoilPanelEvaluator, "evaluate", lambda _self, _config: expected)
    assert _bare_evaluator(tmp_path).evaluate(CANDIDATE) is expected
