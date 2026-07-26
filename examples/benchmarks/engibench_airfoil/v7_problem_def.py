"""Airfoil-v7 problem and generic detailed-evaluation composition.

All aerodynamic receipt interpretation stays in this benchmark adapter.  The
AgentEvolve engine sees only its generic evidence, identity, relation, reward,
and finite-variation ports.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

from agent_evolve.agentic import (
    ActionAxisCoordinateSemantics,
    ActionAxisSemantics,
    ActionSpaceSemantics,
    AgenticBenchmark,
    ArtifactRef,
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluatorIdentity,
    FailureCategory,
    FailureCode,
    FailureRecord,
    MetricRole,
    MetricSemantics,
    MetricSense,
    ObjectiveSpec,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    artifact_ref_for_bytes,
    freeze_json,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
    AirfoilConvergenceEvaluationError,
    create_default_converged_problem,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    AirfoilPanelCandidate,
    AirfoilPanelEvaluation,
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    COMPATIBILITY_ADAPTER_SHA256,
    EXTERNAL_DECODER_EVALUATOR_SHA256,
    LIFT_TARGET,
    NEUTRAL_POINT_DRAGS,
    TASK_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
    SHAPE_CATALOG_DEFINITION_SHA256,
    TRIM_CATALOG_DEFINITION_SHA256,
    UNION_CATALOG_DEFINITION_SHA256,
)


OBJECTIVE_NAME = "normalized_multipoint_drag"
VIOLATION_NAME = "normalized_lift_equality"
V2_EVALUATOR_SOURCE_SHA256 = (
    "6cd2c9e891ba1fdcca0f1f44f83165bd8550a159095e0d0b5638da727060fa5f"
)
V2_CONVERGENCE_CONTRACT_SHA256 = (
    "e53e1f2ae9738bbebf78e39e22d2f20153301b8a5978cab77c394a7dee0a5df8"
)
V2_CONVERGENCE_OVERLAY_SHA256 = (
    "b2c5c26bb0186f85d16f8ebecce06775ee68c0591d02d796e0ca0c2a9d9947ab"
)
_EVALUATOR_CONTEXT_HASH_DOMAIN = b"agent-evolve:airfoil-v7-evaluator-context:v1\x00"


AIRFOIL_V7_OPTIMIZATION_SEMANTICS = OptimizationSemantics(
    semantics_id="airfoil_v7_exact_optimization",
    semantics_version=1,
    metrics=(
        MetricSemantics(
            metric_id=f"objective:{OBJECTIVE_NAME}",
            name=OBJECTIVE_NAME,
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition=(
                "f = (1/3) * sum_i(cd_i / cd_neutral_i) for the three "
                f"operating points, with cd_neutral={list(NEUTRAL_POINT_DRAGS)}."
            ),
            aggregation="Arithmetic mean of the three pointwise drag ratios.",
            witness_interpretation=(
                "normalized_drag_ratio at point i is cd_i / cd_neutral_i; "
                "decreasing a ratio decreases the objective, all else equal."
            ),
            tolerance=0.0,
        ),
        MetricSemantics(
            metric_id=f"violation:{VIOLATION_NAME}",
            name=VIOLATION_NAME,
            role=MetricRole.VIOLATION,
            sense=MetricSense.MINIMIZE,
            definition=(
                "V = sum_i(abs(cl_i - lift_target) / abs(lift_target)) "
                f"for lift_target={LIFT_TARGET}; V=0 is ideal."
            ),
            aggregation=(
                "Sum, not a maximum or cross-point spread, of the three "
                "pointwise normalized absolute target residuals."
            ),
            reference_target=LIFT_TARGET,
            tolerance=0.0,
            witness_interpretation=(
                f"signed_lift_residual is cl_i - {LIFT_TARGET}. A negative "
                "value means that point is below target: increasing cl_i "
                "toward the target reduces normalized_abs_lift_residual and V, "
                "whereas decreasing cl_i moves farther from target and raises V. "
                "A positive residual reverses that directional interpretation."
            ),
        ),
    ),
    outcome_ordering=OutcomeOrderingSemantics(
        kind=OutcomeOrderingKind.LEXICOGRAPHIC,
        metric_priority=(
            f"violation:{VIOLATION_NAME}",
            f"objective:{OBJECTIVE_NAME}",
        ),
        description=(
            "Compare V exactly first: lower V is better. Only when V is "
            "exactly equal compare f: lower f is better. This is not a "
            "drag-first or weighted-sum policy."
        ),
        equivalence="Two successful outcomes are equivalent only when V and f match exactly.",
        policy_id=AIRFOIL_V7_ARCHIVE_RELATION.policy_id,
        policy_version=AIRFOIL_V7_ARCHIVE_RELATION.policy_version,
        definition_sha256=AIRFOIL_V7_ARCHIVE_RELATION.definition_sha256,
    ),
)


AIRFOIL_V7_ACTION_SEMANTICS = ActionSpaceSemantics(
    semantics_id="airfoil_v7_finite_action_space",
    semantics_version=1,
    catalog_identities=(
        (
            "airfoil_v7_shape",
            2,
            SHAPE_CATALOG_DEFINITION_SHA256,
        ),
        (
            "airfoil_v7_trim",
            2,
            TRIM_CATALOG_DEFINITION_SHA256,
        ),
        (
            "airfoil_v7_union",
            2,
            UNION_CATALOG_DEFINITION_SHA256,
        ),
    ),
    axes=(
        ActionAxisSemantics(
            axis_id="shared_bernstein_profile",
            configuration_paths=(
                "$.lower_coefficients",
                "$.upper_coefficients",
            ),
            option_families=("shape_only",),
            definition=(
                "Ten upper and ten lower degree-9 Bernstein y-displacement "
                "coefficients define one shared two-dimensional airfoil profile "
                "used at all three operating points."
            ),
            independence=(
                "A shape option applies one catalog-defined camber or thickness "
                "mode to a coupled set of eight coefficients; the resulting "
                "profile is shared unchanged by all three operating points."
            ),
            unit="Chord-normalized y displacement.",
            excluded_interpretations=(
                "The three operating points do not have separate shapes.",
                "These coefficients are not spanwise stations, time steps, or "
                "three-dimensional geometry.",
            ),
        ),
        ActionAxisSemantics(
            axis_id="three_point_trim",
            configuration_paths=("$.alpha_deg",),
            option_families=("trim_only",),
            definition=(
                "Three ordered angles of attack trim the lower, central, and "
                "higher Mach-Reynolds operating points respectively."
            ),
            independence=(
                "The three coordinates are point-specific and independently "
                "selected from the catalog delta set; a trim option adds one "
                "possibly different delta to each coordinate without broadcasting."
            ),
            unit="Degrees.",
            coordinates=(
                ActionAxisCoordinateSemantics(
                    index=0,
                    label="lower_mach_reynolds_point",
                    definition=(
                        "Angle of attack for the lower Mach-Reynolds operating point."
                    ),
                ),
                ActionAxisCoordinateSemantics(
                    index=1,
                    label="central_mach_reynolds_point",
                    definition=(
                        "Angle of attack for the central Mach-Reynolds operating point."
                    ),
                ),
                ActionAxisCoordinateSemantics(
                    index=2,
                    label="higher_mach_reynolds_point",
                    definition=(
                        "Angle of attack for the higher Mach-Reynolds operating point."
                    ),
                ),
            ),
            excluded_interpretations=(
                "A trim vector is not one angle broadcast across all three "
                "operating points.",
                "Coordinate order is not interchangeable: indices 0, 1, and "
                "2 bind the lower, central, and higher Mach-Reynolds points "
                "respectively.",
                "The three trim coordinates are not spanwise or chordwise "
                "stations, wing-twist sections, temporal stages, or "
                "section-incidence controls.",
            ),
        ),
    ),
)


_EVALUATOR_CONTEXT_DEFINITION = {
    "v2_evaluator_id": V2_EVALUATOR_ID,
    "v2_evaluator_source_sha256": V2_EVALUATOR_SOURCE_SHA256,
    "convergence_contract_sha256": V2_CONVERGENCE_CONTRACT_SHA256,
    "convergence_overlay_sha256": V2_CONVERGENCE_OVERLAY_SHA256,
    "task_sha256": TASK_SHA256,
    "external_decoder_evaluator_sha256": EXTERNAL_DECODER_EVALUATOR_SHA256,
    "compatibility_adapter_sha256": COMPATIBILITY_ADAPTER_SHA256,
    "neutral_point_drags": NEUTRAL_POINT_DRAGS,
    "lift_target": LIFT_TARGET,
    "objective": {
        "name": OBJECTIVE_NAME,
        "formula": "mean_k(cd_k/cd_neutral_k)",
    },
    "violation": {
        "name": VIOLATION_NAME,
        "formula": "sum_k(abs(cl_k-target)/abs(target))",
    },
}


def _context_sha256(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_EVALUATOR_CONTEXT_HASH_DOMAIN + encoded).hexdigest()


EVALUATOR_IDENTITY = EvaluatorIdentity(
    evaluator_id="airfoil_v7_convergence_projection",
    evaluator_version=1,
    evaluator_context_sha256=_context_sha256(_EVALUATOR_CONTEXT_DEFINITION),
)


class RawConvergedAirfoilProblem(Protocol):
    def evaluate_raw(self, config: object) -> AirfoilPanelEvaluation: ...


class AirfoilV7ReceiptError(RuntimeError):
    """The raw v2 receipt cannot support the generic evidence projection."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise AirfoilV7ReceiptError(f"{label} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AirfoilV7ReceiptError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise AirfoilV7ReceiptError(f"{label} must be finite")
    return number


def _exact_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AirfoilV7ReceiptError(f"{label} must be an object")
    return value


def _receipt_ref(
    path: Path,
    expected_record: Mapping[str, Any],
) -> ArtifactRef:
    if not path.is_file():
        raise AirfoilV7ReceiptError(f"durable evaluator receipt is missing: {path}")
    try:
        content = path.read_bytes()
        parsed = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilV7ReceiptError(
            f"durable evaluator receipt is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if parsed != dict(expected_record):
        raise AirfoilV7ReceiptError("in-memory and durable evaluator receipts disagree")
    return artifact_ref_for_bytes(content, media_type="application/json")


def _check(
    name: str,
    status: EvaluationCheckStatus,
    observed: object,
    locator: str,
) -> EvaluationCheck:
    return EvaluationCheck(
        name=name,
        status=status,
        observed_value=freeze_json(observed),
        receipt_locator=locator,
    )


def _failure_record(
    category: FailureCategory,
    code: FailureCode,
    message: str,
    *,
    exception_type: str,
    receipt: ArtifactRef | None,
    retryable: bool = False,
) -> FailureRecord:
    return FailureRecord(
        category=category,
        code=code,
        message=message,
        retryable=retryable,
        exception_type=exception_type,
        diagnostics_artifact_id=None if receipt is None else receipt.artifact_id,
    )


def _failed_payload(
    failure: FailureRecord,
    *,
    checks: tuple[EvaluationCheck, ...] = (),
    receipt: ArtifactRef | None = None,
) -> DetailedEvaluationPayload:
    return DetailedEvaluationPayload(
        failure=failure,
        objectives=(),
        violations=(),
        checks=checks,
        receipt=receipt,
        evaluator=EVALUATOR_IDENTITY,
    )


def _raw_failure_payload(
    error: AirfoilConvergenceEvaluationError,
) -> DetailedEvaluationPayload:
    record = error.record
    receipt: ArtifactRef | None = None
    if record is not None:
        try:
            receipt = _receipt_ref(error.record_path, record)
        except AirfoilV7ReceiptError:
            return _failed_payload(
                _failure_record(
                    FailureCategory.SYSTEM,
                    FailureCode.PARSER_FAILURE,
                    "Airfoil v2 failure receipt could not be verified",
                    exception_type=type(error).__name__,
                    receipt=None,
                )
            )

    failure_type = None
    failure_classification = None
    evaluator_calls = 0
    if record is not None:
        failure = record.get("failure")
        if isinstance(failure, Mapping):
            failure_type = failure.get("type")
        failure_classification = record.get("failure_classification")
        calls = record.get("evaluator_calls")
        if type(calls) is int and calls >= 0:
            evaluator_calls = calls

    checks: tuple[EvaluationCheck, ...]
    if error.candidate_invalid and failure_classification == "authoritative_solver_failure":
        category = FailureCategory.CANDIDATE
        code = FailureCode.NUMERICAL_NONCONVERGENCE
        point_index = 0 if record is None else record.get("failed_point_index", 0)
        witness = {} if record is None else record.get("evaluator_evidence", {})
        status = witness.get("authoritative_status", {}) if isinstance(witness, Mapping) else {}
        checks = (
            _check(
                f"point_{point_index}_convergence",
                EvaluationCheckStatus.FAIL,
                {
                    "solve_failed": status.get("solve_failed"),
                    "fatal_fail": status.get("fatal_fail"),
                    "check_solution_failure": status.get("check_solution_failure"),
                    "evaluator_calls": evaluator_calls,
                },
                "$.evaluator_evidence.authoritative_status",
            ),
        ) if receipt is not None else ()
    elif error.candidate_invalid:
        category = FailureCategory.CANDIDATE
        code = FailureCode.DETERMINISTIC_PRECHECK_INFEASIBLE
        checks = (
            _check(
                "deterministic_precheck",
                EvaluationCheckStatus.FAIL,
                {"failure_type": failure_type, "evaluator_calls": evaluator_calls},
                "$.failure",
            ),
        ) if receipt is not None else ()
    elif failure_type == "WitnessBoundaryFailure":
        category = FailureCategory.SYSTEM
        code = FailureCode.EVALUATOR_CONTRACT_VIOLATION
        checks = (
            _check(
                "evaluator_contract",
                EvaluationCheckStatus.FAIL,
                {"failure_type": failure_type, "evaluator_calls": evaluator_calls},
                "$.failure",
            ),
        ) if receipt is not None else ()
    elif "exceeded" in str(error).lower() or "timeout" in str(error).lower():
        category = FailureCategory.INFRASTRUCTURE
        code = FailureCode.TIMEOUT_OR_RESOURCE_FAILURE
        checks = ()
    elif receipt is None:
        category = FailureCategory.INFRASTRUCTURE
        code = FailureCode.PROCESS_START_FAILURE
        checks = ()
    else:
        category = FailureCategory.INFRASTRUCTURE
        code = FailureCode.CONTAINER_OR_DEPENDENCY_FAILURE
        checks = (
            _check(
                "evaluator_process",
                EvaluationCheckStatus.FAIL,
                {"failure_type": failure_type, "evaluator_calls": evaluator_calls},
                "$.failure",
            ),
        )
    return _failed_payload(
        _failure_record(
            category,
            code,
            str(error),
            exception_type=type(error).__name__,
            receipt=receipt,
        ),
        checks=tuple(sorted(checks, key=lambda item: item.name)),
        receipt=receipt,
    )


def _success_payload(
    evaluation: AirfoilPanelEvaluation,
    candidate: Mapping[str, Any],
) -> DetailedEvaluationPayload:
    if type(evaluation) is not AirfoilPanelEvaluation:
        raise AirfoilV7ReceiptError("raw Airfoil evaluator returned the wrong value type")
    record = evaluation.record
    if (
        record.get("schema_version") != 2
        or record.get("evaluator_id") != V2_EVALUATOR_ID
        or record.get("status") != "evaluated"
        or record.get("candidate_sha256") != candidate_sha256(candidate)
        or record.get("task_sha256") != TASK_SHA256
        or record.get("evaluator_calls") != 3
    ):
        raise AirfoilV7ReceiptError("Airfoil v2 receipt identity/status mismatch")
    receipt = _receipt_ref(evaluation.record_path, record)
    decoder = _exact_mapping(record.get("decoder_audit"), "decoder_audit")
    area_ratio = _finite(decoder.get("area_ratio"), "decoder_audit.area_ratio")
    area_bounds = decoder.get("area_ratio_bounds")
    if not isinstance(area_bounds, list) or len(area_bounds) != 2:
        raise AirfoilV7ReceiptError("decoder area bounds are malformed")
    lower_area = _finite(area_bounds[0], "area lower bound")
    upper_area = _finite(area_bounds[1], "area upper bound")
    if not lower_area <= area_ratio <= upper_area:
        raise AirfoilV7ReceiptError("successful receipt violates its area bounds")
    decoded_sha256 = decoder.get("decoded_coords_sha256")
    if (
        not isinstance(decoded_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", decoded_sha256) is None
    ):
        raise AirfoilV7ReceiptError("decoded-coordinate identity is malformed")

    points = record.get("points")
    if not isinstance(points, list) or len(points) != 3:
        raise AirfoilV7ReceiptError("Airfoil v2 receipt lacks exactly three points")
    cds: list[float] = []
    cls: list[float] = []
    checks: list[EvaluationCheck] = [
        _check(
            "area_bounds",
            EvaluationCheckStatus.PASS,
            {"area_ratio": area_ratio, "bounds": [lower_area, upper_area]},
            "$.decoder_audit",
        ),
        _check(
            "geometry",
            EvaluationCheckStatus.PASS,
            {
                "decoded_coords_sha256": decoded_sha256,
                "external_representation": decoder.get(
                    "external_representation_not_upstream_ffd"
                ),
            },
            "$.decoder_audit",
        ),
    ]
    for index, point_value in enumerate(points):
        point = _exact_mapping(point_value, f"points[{index}]")
        if point.get("index") != index:
            raise AirfoilV7ReceiptError("Airfoil point order/index mismatch")
        cd = _finite(point.get("cd"), f"points[{index}].cd")
        cl = _finite(point.get("cl"), f"points[{index}].cl")
        evidence = _exact_mapping(
            point.get("evaluator_evidence"), f"points[{index}].evaluator_evidence"
        )
        if (
            evidence.get("contract_id") != EVIDENCE_CONTRACT_ID
            or evidence.get("evaluator_id") != ADFLOW_EVALUATOR_ID
            or evidence.get("accepted") is not True
        ):
            raise AirfoilV7ReceiptError("Airfoil point evidence identity/status mismatch")
        witness = _exact_mapping(evidence.get("witness"), "point witness")
        status = _exact_mapping(witness.get("authoritative_status"), "authoritative_status")
        for field_name in (
            "solve_failed",
            "fatal_fail",
            "check_solution_failure",
        ):
            value = status.get(field_name)
            if type(value) is not bool or value is not False:
                raise AirfoilV7ReceiptError(
                    "successful Airfoil receipt requires exact false "
                    f"authoritative_status.{field_name}"
                )
        residual = _exact_mapping(witness.get("residual_evidence"), "residual_evidence")
        history = _exact_mapping(residual.get("convergence_history"), "convergence_history")
        series = _exact_mapping(history.get("series"), "convergence_history.series")
        linear = _exact_mapping(series.get("linear_res"), "linear_res")
        resrho_value = series.get("resrho")
        resrhoe_value = series.get("resrhoe")
        resrho = (
            None if resrho_value is None else _exact_mapping(resrho_value, "resrho")
        )
        resrhoe = (
            None
            if resrhoe_value is None
            else _exact_mapping(resrhoe_value, "resrhoe")
        )
        ratio = cd / NEUTRAL_POINT_DRAGS[index]
        signed_lift_residual = cl - LIFT_TARGET
        normalized_abs_lift_residual = abs(signed_lift_residual) / abs(LIFT_TARGET)
        checks.append(
            _check(
                f"point_{index}_convergence",
                EvaluationCheckStatus.PASS,
                {
                    "solve_failed": status.get("solve_failed"),
                    "fatal_fail": status.get("fatal_fail"),
                    "check_solution_failure": status.get("check_solution_failure"),
                    "free_stream_total_residual_reference": _finite(
                        residual.get("free_stream_total_residual_reference"),
                        "free-stream residual reference",
                    ),
                    "history_rows": history.get("history_rows"),
                    "final_total_minor_iters": history.get("final_total_minor_iters"),
                    "final_linear_residual": _finite(linear.get("final"), "linear residual"),
                    "final_resrho": (
                        None
                        if resrho is None
                        else _finite(resrho.get("final"), "resrho")
                    ),
                    "final_resrhoe": (
                        None
                        if resrhoe is None
                        else _finite(resrhoe.get("final"), "resrhoe")
                    ),
                    "cd": cd,
                    "cl": cl,
                    "normalized_drag_ratio": ratio,
                    "signed_lift_residual": signed_lift_residual,
                    "normalized_abs_lift_residual": normalized_abs_lift_residual,
                },
                f"$.points[{index}].evaluator_evidence.witness",
            )
        )
        cds.append(cd)
        cls.append(cl)
    checks.append(
        _check(
            "three_point_panel",
            EvaluationCheckStatus.PASS,
            {"evaluator_calls": 3, "point_count": 3, "task_sha256": TASK_SHA256},
            "$.points",
        )
    )
    normalized_drag = sum(
        value / neutral
        for value, neutral in zip(cds, NEUTRAL_POINT_DRAGS, strict=True)
    ) / 3.0
    normalized_lift_violation = sum(
        abs(value - LIFT_TARGET) / abs(LIFT_TARGET) for value in cls
    )
    active_seconds = _finite(evaluation.wall_seconds, "Airfoil active wall seconds")
    return DetailedEvaluationPayload(
        failure=None,
        objectives=((OBJECTIVE_NAME, float(normalized_drag)),),
        violations=((VIOLATION_NAME, float(normalized_lift_violation)),),
        checks=tuple(sorted(checks, key=lambda item: item.name)),
        receipt=receipt,
        evaluator=EVALUATOR_IDENTITY,
        active_wall_seconds=active_seconds,
        resource_queue_wall_seconds=None,
    )


class AirfoilV7DetailedEvaluationAdapter:
    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(self, raw_problem: RawConvergedAirfoilProblem) -> None:
        if not callable(getattr(raw_problem, "evaluate_raw", None)):
            raise TypeError("raw Airfoil problem must implement evaluate_raw")
        self.raw_problem = raw_problem

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        try:
            candidate = normalize_candidate(configuration)
        except ValueError as exc:
            return _failed_payload(
                _failure_record(
                    FailureCategory.CANDIDATE,
                    FailureCode.SCHEMA_INVALID,
                    str(exc),
                    exception_type=type(exc).__name__,
                    receipt=None,
                )
            )
        try:
            evaluation = self.raw_problem.evaluate_raw(candidate)
        except AirfoilConvergenceEvaluationError as exc:
            return _raw_failure_payload(exc)
        except Exception as exc:
            return _failed_payload(
                _failure_record(
                    FailureCategory.SYSTEM,
                    FailureCode.UNKNOWN_UNCLASSIFIED,
                    str(exc),
                    exception_type=type(exc).__name__,
                    receipt=None,
                )
            )
        try:
            return _success_payload(evaluation, candidate)
        except AirfoilV7ReceiptError as exc:
            receipt: ArtifactRef | None = None
            try:
                receipt = _receipt_ref(evaluation.record_path, evaluation.record)
            except AirfoilV7ReceiptError:
                pass
            return _failed_payload(
                _failure_record(
                    FailureCategory.SYSTEM,
                    FailureCode.EVALUATOR_CONTRACT_VIOLATION,
                    str(exc),
                    exception_type=type(exc).__name__,
                    receipt=receipt,
                ),
                checks=() if receipt is None else (
                    _check(
                        "evaluator_contract",
                        EvaluationCheckStatus.FAIL,
                        {"projection_error": type(exc).__name__},
                        "$",
                    ),
                ),
                receipt=receipt,
            )


def replay_airfoil_v7_durable_receipt(
    path: Path,
    configuration: object,
) -> DetailedEvaluationPayload:
    """Rebuild the typed v7 evidence projection from one durable raw receipt.

    This is deliberately narrower than an evaluation cache.  It performs no
    search and never invokes the external evaluator; callers must already own
    the receipt path and the exact candidate they expect it to describe.  The
    same projection functions used on the live evaluator path authenticate the
    receipt before returning a payload, which makes a crash between raw receipt
    publication and a higher-level journal commit safely recoverable.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    resolved = path.expanduser().resolve(strict=True)
    candidate = normalize_candidate(configuration)
    try:
        record = json.loads(resolved.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilV7ReceiptError(
            "durable Airfoil-v7 receipt is unreadable"
        ) from exc
    if type(record) is not dict:
        raise AirfoilV7ReceiptError("durable Airfoil-v7 receipt must be an object")
    if (
        record.get("schema_version") != 2
        or record.get("evaluator_id") != V2_EVALUATOR_ID
        or record.get("mode") != "evaluate"
        or record.get("candidate_sha256") != candidate_sha256(candidate)
    ):
        raise AirfoilV7ReceiptError(
            "durable Airfoil-v7 receipt identity does not match the candidate"
        )

    status = record.get("status")
    if status == "evaluated":
        objectives = record.get("objectives")
        if type(objectives) is not dict:
            raise AirfoilV7ReceiptError(
                "successful durable Airfoil-v7 receipt lacks objectives"
            )
        evaluation = AirfoilPanelEvaluation(
            candidate_sha256=candidate_sha256(candidate),
            objective_values={
                str(name): _finite(value, f"objectives.{name}")
                for name, value in objectives.items()
            },
            wall_seconds=_finite(record.get("wall_seconds"), "wall_seconds"),
            record_path=resolved,
            record=record,
        )
        return _success_payload(evaluation, candidate)

    if status not in {
        "candidate_invalid",
        "infrastructure_or_evaluator_failure",
    }:
        raise AirfoilV7ReceiptError(
            "durable Airfoil-v7 receipt has an unsupported terminal status"
        )
    failure = record.get("failure")
    if not isinstance(failure, Mapping):
        raise AirfoilV7ReceiptError(
            "failed durable Airfoil-v7 receipt lacks failure evidence"
        )
    failure_type = failure.get("type")
    failure_message = failure.get("message")
    if type(failure_type) is not str or type(failure_message) is not str:
        raise AirfoilV7ReceiptError(
            "failed durable Airfoil-v7 receipt has malformed failure evidence"
        )
    error = AirfoilConvergenceEvaluationError(
        f"{failure_type}: {failure_message}",
        candidate_invalid=status == "candidate_invalid",
        record_path=resolved,
        record=record,
    )
    return _raw_failure_payload(error)


class AirfoilV7Problem:
    """Single-objective candidate space for the v7 detailed-evidence workflow."""

    candidate_model = AirfoilPanelCandidate
    optimization_semantics = AIRFOIL_V7_OPTIMIZATION_SEMANTICS
    action_semantics = AIRFOIL_V7_ACTION_SEMANTICS
    example_config = {
        "representation_id": "external_bernstein_y_panel_v1",
        "upper_coefficients": [0.0] * 10,
        "lower_coefficients": [0.0] * 10,
        "alpha_deg": [2.5, 2.5, 2.5],
    }
    constraints_description = (
        "External degree-9 Bernstein coefficients lie in [-0.025,0.025]; three "
        "pointwise angles lie in [0,10]. No clipping or silent repair is allowed."
    )

    def __init__(
        self,
        raw_problem: RawConvergedAirfoilProblem | None = None,
    ) -> None:
        self.raw_problem = (
            create_default_converged_problem() if raw_problem is None else raw_problem
        )
        self.detailed_evaluator = AirfoilV7DetailedEvaluationAdapter(self.raw_problem)

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (ObjectiveSpec(OBJECTIVE_NAME, "min"),)

    @staticmethod
    def validate(config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate(self, config: object) -> dict[str, float]:
        payload = self.detailed_evaluator.evaluate_evidence(normalize_candidate(config))
        if payload.failure is not None:
            if payload.failure.category is FailureCategory.CANDIDATE:
                raise ValueError(payload.failure.message)
            raise RuntimeError(payload.failure.message)
        return dict(payload.objectives)

    @staticmethod
    def candidate_key(config: object) -> str:
        return candidate_sha256(config)

    @staticmethod
    def render_candidate(config: object) -> str:
        candidate = normalize_candidate(config)
        upper_peak = max(abs(item) for item in candidate["upper_coefficients"])
        lower_peak = max(abs(item) for item in candidate["lower_coefficients"])
        return (
            f"external Bernstein shape |upper|max={upper_peak:.5f}, "
            f"|lower|max={lower_peak:.5f}, alpha={candidate['alpha_deg']}"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Convergence-qualified three-point Airfoil v7. One shared external "
            "Bernstein shape has 10 upper and 10 lower coefficients; three "
            "point-specific angles control trim. The sole optimizer objective is "
            "normalized multipoint drag; normalized lift equality is retained as "
            "a separate detailed-evidence violation."
        )


def create_default_v7_problem() -> AirfoilV7Problem:
    return AirfoilV7Problem()


problem = create_default_v7_problem()
detailed_evaluator = problem.detailed_evaluator
phenotype_identity_policy = AirfoilV7PhenotypeIdentityPolicy()
outcome_relation_binding = AIRFOIL_V7_ARCHIVE_RELATION
reward_binding = AIRFOIL_V7_REWARD_BINDING
shape_variation_catalog = AirfoilV7ShapeVariationCatalog()
trim_variation_catalog = AirfoilV7TrimVariationCatalog()
union_variation_catalog = AirfoilV7UnionVariationCatalog()
benchmark = AgenticBenchmark(
    problem=problem,
    reward=reward_binding,
    detailed_evaluator=detailed_evaluator,
    outcome_relation=outcome_relation_binding,
    action_semantics=AIRFOIL_V7_ACTION_SEMANTICS,
    phenotype_identity=phenotype_identity_policy,
    finite_variation_catalogs=(
        shape_variation_catalog,
        trim_variation_catalog,
        union_variation_catalog,
    ),
)


__all__ = [
    "AIRFOIL_V7_ACTION_SEMANTICS",
    "AIRFOIL_V7_OPTIMIZATION_SEMANTICS",
    "AirfoilV7DetailedEvaluationAdapter",
    "AirfoilV7Problem",
    "EVALUATOR_IDENTITY",
    "OBJECTIVE_NAME",
    "VIOLATION_NAME",
    "benchmark",
    "create_default_v7_problem",
    "detailed_evaluator",
    "outcome_relation_binding",
    "phenotype_identity_policy",
    "problem",
    "replay_airfoil_v7_durable_receipt",
    "reward_binding",
    "shape_variation_catalog",
    "trim_variation_catalog",
    "union_variation_catalog",
]
