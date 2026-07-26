"""Focused provider/CFD-free tests for the Airfoil-v7 physics H arm."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import product

import pytest

from agent_evolve.agentic import (
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluatorIdentity,
    FiniteVariationContract,
    artifact_ref_for_bytes,
    freeze_json,
)
from examples.benchmarks.engibench_airfoil.v7_contract import LIFT_TARGET
from examples.benchmarks.engibench_airfoil.v7_physics_baseline import (
    AirfoilV7ParentEvidence,
    AirfoilV7PhysicsBaselineError,
    AirfoilV7PointEvidence,
    AirfoilV7TrainingSourceSeal,
    AirfoilV7TrimOutcome,
    AirfoilV7TrimPhysicsResponseSelector,
    AirfoilV7TrimTrainingSet,
    fit_airfoil_v7_trim_response_model,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7TrimVariationCatalog,
    TRIM_DELTAS_DEG,
)


_PART = {-0.5: "n050", -0.25: "n025", 0.25: "p025", 0.5: "p050"}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _option_id(deltas: tuple[float, float, float]) -> str:
    return "trim." + ".".join(_PART[value] for value in deltas)


def _point(residual: float, drag_ratio: float) -> AirfoilV7PointEvidence:
    return AirfoilV7PointEvidence(
        cl=float(LIFT_TARGET + residual),
        signed_lift_residual=float(residual),
        normalized_drag_ratio=float(drag_ratio),
    )


def _parent(
    *,
    configuration_sha256: str,
    residuals: tuple[float, float, float],
    drag_ratios: tuple[float, float, float],
    receipt_label: str,
) -> AirfoilV7ParentEvidence:
    points = tuple(
        _point(residual, drag_ratio)
        for residual, drag_ratio in zip(residuals, drag_ratios, strict=True)
    )
    return AirfoilV7ParentEvidence(
        parent_configuration_sha256=configuration_sha256,
        evaluator_id=EVALUATOR_IDENTITY.evaluator_id,
        evaluator_version=EVALUATOR_IDENTITY.evaluator_version,
        evaluator_context_sha256=EVALUATOR_IDENTITY.evaluator_context_sha256,
        receipt_sha256=_sha(f"receipt:{receipt_label}"),
        source_evidence_sha256=_sha(f"evidence:{receipt_label}"),
        objective=float(sum(drag_ratios) / 3.0),
        violation=float(sum(abs(value) / abs(LIFT_TARGET) for value in residuals)),
        points=points,  # type: ignore[arg-type]
    )


def _lift_response(point_index: int, delta: float) -> float:
    return float((0.11 + point_index * 0.02) * delta)


def _drag_response(point_index: int, delta: float) -> float:
    return float((0.025 + point_index * 0.005) * delta + 0.018 * delta * delta)


def _training_set() -> AirfoilV7TrimTrainingSet:
    parent_sha = _sha("nonce-zero-parent-configuration")
    parent = _parent(
        configuration_sha256=parent_sha,
        residuals=(-0.22, -0.15, -0.08),
        drag_ratios=(1.02, 1.01, 0.99),
        receipt_label="nonce-zero-parent",
    )
    outcomes: list[AirfoilV7TrimOutcome] = []
    for ordinal, deltas in enumerate(product(TRIM_DELTAS_DEG, repeat=3), start=1):
        canonical_deltas = tuple(float(value) for value in deltas)
        option_id = _option_id(canonical_deltas)  # type: ignore[arg-type]
        points = tuple(
            _point(
                parent.points[point_index].signed_lift_residual
                + _lift_response(point_index, delta),
                parent.points[point_index].normalized_drag_ratio
                + _drag_response(point_index, delta),
            )
            for point_index, delta in enumerate(canonical_deltas)
        )
        outcomes.append(
            AirfoilV7TrimOutcome(
                ordinal=ordinal,
                option_id=option_id,
                option_identity_sha256=_sha(f"option:{option_id}"),
                parent_configuration_sha256=parent_sha,
                child_configuration_sha256=_sha(f"child:{option_id}"),
                delta_alpha_deg=canonical_deltas,  # type: ignore[arg-type]
                points=points,  # type: ignore[arg-type]
                raw_receipt_sha256=_sha(f"raw:{option_id}"),
                terminal_record_sha256=_sha(f"terminal:{option_id}"),
            )
        )
    return AirfoilV7TrimTrainingSet(
        source=AirfoilV7TrainingSourceSeal(
            oracle_run_id="synthetic_nonce_zero",
            nonce=0,
            manifest_sha256=_sha("manifest"),
            oracle_result_sha256=_sha("result"),
            oracle_recursive_content_sha256=_sha("recursive"),
            oracle_finalization_record_sha256=_sha("finalization"),
            parent_result_file_sha256=_sha("parent-result-file"),
        ),
        parent=parent,
        outcomes=tuple(outcomes),
    )


def _target_contract() -> FiniteVariationContract:
    parent = freeze_json(
        {
            "representation_id": "external_bernstein_y_panel_v1",
            "upper_coefficients": [0.0] * 10,
            "lower_coefficients": [0.0] * 10,
            "alpha_deg": [2.1, 2.6, 3.0],
        }
    )
    return FiniteVariationContract(
        catalog_id=AirfoilV7TrimVariationCatalog.catalog_id,
        catalog_version=AirfoilV7TrimVariationCatalog.catalog_version,
        catalog_definition_sha256=AirfoilV7TrimVariationCatalog.definition_sha256,
        parent_configuration=parent,
        options=AirfoilV7TrimVariationCatalog.options(parent),
    )


def _target_payload(
    contract: FiniteVariationContract,
    *,
    residuals: tuple[float, float, float] = (-0.04, -0.075, 0.025),
    drag_ratios: tuple[float, float, float] = (1.03, 1.00, 1.04),
    evaluator: EvaluatorIdentity = EVALUATOR_IDENTITY,
    receipt_label: str = "target-parent",
) -> DetailedEvaluationPayload:
    del contract
    checks = tuple(
        EvaluationCheck(
            name=f"point_{point_index}_convergence",
            status=EvaluationCheckStatus.PASS,
            observed_value=freeze_json(
                {
                    "cl": LIFT_TARGET + residual,
                    "signed_lift_residual": residual,
                    "normalized_drag_ratio": drag_ratio,
                }
            ),
            receipt_locator=f"$.points[{point_index}]",
        )
        for point_index, (residual, drag_ratio) in enumerate(
            zip(residuals, drag_ratios, strict=True)
        )
    )
    receipt_bytes = f"synthetic-parent-receipt:{receipt_label}".encode("ascii")
    return DetailedEvaluationPayload(
        failure=None,
        objectives=((OBJECTIVE_NAME, float(sum(drag_ratios) / 3.0)),),
        violations=(
            (
                VIOLATION_NAME,
                float(sum(abs(value) / abs(LIFT_TARGET) for value in residuals)),
            ),
        ),
        checks=checks,
        receipt=artifact_ref_for_bytes(receipt_bytes, media_type="application/json"),
        evaluator=evaluator,
        active_wall_seconds=1.0,
        resource_queue_wall_seconds=None,
    )


def _expected_order(
    model: object,
    residuals: tuple[float, float, float],
    drag_ratios: tuple[float, float, float],
) -> tuple[str, ...]:
    rows: list[tuple[float, float, str]] = []
    for deltas in product(TRIM_DELTAS_DEG, repeat=3):
        predicted_residuals = tuple(
            residuals[index]
            + model.cell(index, delta).delta_signed_lift_residual
            for index, delta in enumerate(deltas)
        )
        predicted_drag = tuple(
            drag_ratios[index]
            + model.cell(index, delta).delta_normalized_drag_ratio
            for index, delta in enumerate(deltas)
        )
        rows.append(
            (
                sum(abs(value) / abs(LIFT_TARGET) for value in predicted_residuals),
                sum(predicted_drag) / 3.0,
                _option_id(tuple(float(value) for value in deltas)),  # type: ignore[arg-type]
            )
        )
    return tuple(row[2] for row in sorted(rows))


def test_response_model_predicts_all_64_and_ranks_exact_violation_then_drag() -> None:
    training = _training_set()
    model = fit_airfoil_v7_trim_response_model(training)
    contract = _target_contract()
    target_residuals = (-0.04, -0.075, 0.025)
    target_drag = (1.03, 1.00, 1.04)
    payload = _target_payload(
        contract,
        residuals=target_residuals,
        drag_ratios=target_drag,
    )
    selector = AirfoilV7TrimPhysicsResponseSelector(model)

    decision = selector.select(
        target_contract=contract,
        target_parent_evaluation=payload,
        top_k=3,
    )
    expected = _expected_order(model, target_residuals, target_drag)
    assert len(decision.predictions) == 64
    assert decision.selected_option_ids == expected[:3]
    assert tuple(row.option_id for row in decision.predictions) == expected
    assert tuple(row.rank for row in decision.predictions) == tuple(range(1, 65))
    assert all(cell.replicate_count == 16 for cell in model.cells)
    assert len(model.cells) == 12

    # At least one lower-violation option has worse drag than a later option;
    # the ordering must still follow violation first without scalarization.
    assert any(
        left.predicted_violation < right.predicted_violation
        and left.predicted_objective > right.predicted_objective
        for left in decision.predictions
        for right in decision.predictions[left.rank :]
    )
    trace = decision.to_trace_record()
    assert trace["information_boundary"]["target_child_outcomes_observed"] == 0
    assert trace["selected_option_ids"] == list(expected[:3])
    assert trace["model_sha256"] == model.model_sha256
    assert trace["decision_sha256"] == decision.decision_sha256
    assert selector.select(
        target_contract=contract,
        target_parent_evaluation=payload,
        top_k=3,
    ).decision_sha256 == decision.decision_sha256


def test_fit_rejects_one_nonseparable_replicate_and_incomplete_grid() -> None:
    training = _training_set()
    outcomes = list(training.outcomes)
    first = outcomes[0]
    changed_points = list(first.points)
    point = changed_points[0]
    changed_points[0] = replace(
        point,
        normalized_drag_ratio=point.normalized_drag_ratio + 1e-9,
    )
    outcomes[0] = replace(first, points=tuple(changed_points))  # type: ignore[arg-type]
    contaminated = replace(training, outcomes=tuple(outcomes))
    with pytest.raises(AirfoilV7PhysicsBaselineError, match="separability failed"):
        fit_airfoil_v7_trim_response_model(contaminated)

    with pytest.raises(TypeError, match="exactly 64"):
        replace(training, outcomes=training.outcomes[:-1])


def test_target_identity_changes_without_refitting_and_context_mismatch_fails() -> None:
    model = fit_airfoil_v7_trim_response_model(_training_set())
    selector = AirfoilV7TrimPhysicsResponseSelector(model)
    contract = _target_contract()
    first = selector.select(
        target_contract=contract,
        target_parent_evaluation=_target_payload(contract, receipt_label="first"),
    )
    second = selector.select(
        target_contract=contract,
        target_parent_evaluation=_target_payload(
            contract,
            residuals=(-0.02, -0.075, 0.025),
            receipt_label="second",
        ),
    )
    assert first.model_sha256 == second.model_sha256 == model.model_sha256
    assert first.request_sha256 != second.request_sha256
    assert first.decision_sha256 != second.decision_sha256

    wrong_evaluator = EvaluatorIdentity(
        evaluator_id=EVALUATOR_IDENTITY.evaluator_id,
        evaluator_version=EVALUATOR_IDENTITY.evaluator_version,
        evaluator_context_sha256=_sha("different-evaluator-context"),
    )
    with pytest.raises(AirfoilV7PhysicsBaselineError, match="context differs"):
        selector.select(
            target_contract=contract,
            target_parent_evaluation=_target_payload(
                contract,
                evaluator=wrong_evaluator,
                receipt_label="wrong-context",
            ),
        )


def test_selector_rejects_partial_trim_contract_before_prediction() -> None:
    model = fit_airfoil_v7_trim_response_model(_training_set())
    contract = _target_contract()
    partial = FiniteVariationContract(
        catalog_id=contract.catalog_id,
        catalog_version=contract.catalog_version,
        catalog_definition_sha256=contract.catalog_definition_sha256,
        parent_configuration=contract.parent_configuration,
        options=contract.options[:-1],
    )
    with pytest.raises(ValueError, match="exact 64-option set"):
        AirfoilV7TrimPhysicsResponseSelector(model).select(
            target_contract=partial,
            target_parent_evaluation=_target_payload(contract),
        )
