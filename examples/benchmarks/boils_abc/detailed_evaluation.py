"""Generic detailed-evaluation composition for pinned BOiLS/ABC panels.

The subprocess evaluator remains responsible for executing ABC and checking
equivalence.  This module is the benchmark-owned projection boundary: it binds
the exact evaluator/task identity, persists the complete raw observation, and
publishes only generic AgentEvolve evidence and decision-metric contracts.
Nothing here depends on a campaign, model, selector, or particular circuit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from agent_evolve.agentic import (
    AgenticBenchmark,
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluatorIdentity,
    FailureCategory,
    FailureCode,
    FailureRecord,
    FiniteVariationCatalog,
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    freeze_json,
    objective_pareto_outcome_binding,
)
from agent_evolve.ports.artifact_store import ArtifactStore, put_json
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection

from .actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    CONFIG_SCHEMA_ID,
    SEQUENCE_LENGTH,
    config_sha256,
    normalize_candidate,
)
from .evaluator import (
    ABC_SOURCE_COMMIT,
    BOILS_SOURCE_COMMIT,
    EPFL_SOURCE_COMMIT,
    LUT_INPUTS,
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsEvaluation,
    CircuitEvaluation,
)
from .finite_variation_catalog import BoilsFiniteVariationCatalog
from .problem_def import BoilsAbcProblem, DetailedBoilsEvaluator


TOTAL_LUT_COUNT = "total_lut_count"
TOTAL_LEVELS = "total_levels"
BOILS_DETAILED_EVALUATOR_ID = "boils_abc_detailed_projection"
BOILS_DETAILED_EVALUATOR_VERSION = 1
BOILS_DETAILED_RECEIPT_SCHEMA_VERSION = 1
_CONTEXT_HASH_DOMAIN = b"agent-evolve:boils-abc:evaluator-context:v1\x00"
_ACTION_HASH_DOMAIN = b"agent-evolve:boils-abc:executable-actions:v1\x00"


class BoilsDetailedProjectionError(RuntimeError):
    """A raw ABC observation cannot support the declared generic projection."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


BOILS_EXECUTABLE_ACTIONS_SHA256 = hashlib.sha256(
    _ACTION_HASH_DOMAIN
    + _canonical_json_bytes(
        {
            "configuration_schema_id": CONFIG_SCHEMA_ID,
            "sequence_length": SEQUENCE_LENGTH,
            "actions": [
                {
                    "action_id": action_id,
                    "abc_commands": list(ACTION_COMMANDS[action_id]),
                }
                for action_id in ACTION_IDS
            ],
        }
    )
).hexdigest()


def boils_evaluator_context_record(
    settings: AbcEvaluatorSettings,
) -> dict[str, object]:
    """Return the portable scientific identity of one evaluator declaration.

    Filesystem locations, CPU placement, and the receipt store are deliberately
    absent: they can change without changing the function being evaluated.
    Timeout and diagnostic bounds remain included because they change the
    adapter's observable success/failure evidence.
    """

    if type(settings) is not AbcEvaluatorSettings:
        raise TypeError("settings must be an exact AbcEvaluatorSettings")
    settings.__post_init__()
    return {
        "schema_version": 1,
        "adapter": {
            "evaluator_id": BOILS_DETAILED_EVALUATOR_ID,
            "evaluator_version": BOILS_DETAILED_EVALUATOR_VERSION,
            "receipt_schema_version": BOILS_DETAILED_RECEIPT_SCHEMA_VERSION,
        },
        "candidate_contract": {
            "configuration_schema_id": CONFIG_SCHEMA_ID,
            "sequence_length": SEQUENCE_LENGTH,
            "executable_actions_sha256": BOILS_EXECUTABLE_ACTIONS_SHA256,
        },
        "implementation_provenance": {
            "boils_source_commit": BOILS_SOURCE_COMMIT,
            "adapter_abc_source_commit": ABC_SOURCE_COMMIT,
            "adapter_epfl_source_commit": EPFL_SOURCE_COMMIT,
            "declared_abc_source_identity": settings.abc_source_identity,
            "declared_circuit_suite_identity": settings.circuit_suite_identity,
            "abc_binary_sha256": settings.expected_abc_sha256,
        },
        "evaluation_contract": {
            "ordered_circuits": [
                {
                    "name": circuit.name,
                    "source_sha256": circuit.expected_sha256,
                }
                for circuit in settings.circuits
            ],
            "lut_inputs": LUT_INPUTS,
            "prelude": ["read", "strash"],
            "postlude": ["if_lut_map", "print_stats", "write_blif", "cec"],
            "objectives": [TOTAL_LUT_COUNT, TOTAL_LEVELS],
            "per_circuit_timeout_s_hex": float(
                settings.per_circuit_timeout_s
            ).hex(),
            "max_diagnostic_chars": settings.max_diagnostic_chars,
        },
    }


def boils_evaluator_identity(settings: AbcEvaluatorSettings) -> EvaluatorIdentity:
    """Bind a BOiLS evaluator identity to exact portable panel semantics."""

    context = boils_evaluator_context_record(settings)
    return EvaluatorIdentity(
        evaluator_id=BOILS_DETAILED_EVALUATOR_ID,
        evaluator_version=BOILS_DETAILED_EVALUATOR_VERSION,
        evaluator_context_sha256=hashlib.sha256(
            _CONTEXT_HASH_DOMAIN + _canonical_json_bytes(context)
        ).hexdigest(),
    )


def boils_optimization_semantics(
    problem: BoilsAbcProblem,
) -> OptimizationSemantics:
    """Describe raw LUT/depth Pareto semantics for any declared BOiLS panel."""

    if type(problem) is not BoilsAbcProblem:
        raise TypeError("problem must be an exact BoilsAbcProblem")
    objectives = tuple(problem.objectives)
    relation = objective_pareto_outcome_binding(objectives)
    circuit_names = tuple(circuit.name for circuit in problem.settings.circuits)
    panel_text = ", ".join(circuit_names)
    metrics = (
        MetricSemantics(
            metric_id=f"objective:{TOTAL_LUT_COUNT}",
            name=TOTAL_LUT_COUNT,
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition=(
                "Raw sum of mapped LUT node counts reported as nd after the "
                f"pinned LUT-{LUT_INPUTS} mapping for ordered panel [{panel_text}]."
            ),
            aggregation=(
                "Integer sum over exactly one evaluation of every panel circuit."
            ),
            witness_interpretation="A lower raw mapped LUT count is better.",
            tolerance=0.0,
        ),
        MetricSemantics(
            metric_id=f"objective:{TOTAL_LEVELS}",
            name=TOTAL_LEVELS,
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition=(
                "Raw sum of mapped logic levels reported as lev after the "
                f"pinned LUT-{LUT_INPUTS} mapping for ordered panel [{panel_text}]."
            ),
            aggregation=(
                "Integer sum over exactly one evaluation of every panel circuit."
            ),
            witness_interpretation="A lower raw mapped level sum is better.",
            tolerance=0.0,
        ),
    )
    return OptimizationSemantics(
        semantics_id="boils_abc_raw_panel_pareto",
        semantics_version=1,
        metrics=metrics,
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=tuple(metric.metric_id for metric in metrics),
            description=(
                "Pareto minimization of raw total LUT count and raw total mapped "
                "levels; neither objective is lexicographically privileged."
            ),
            equivalence=(
                "Two successful outcomes are equivalent exactly when both raw "
                "integer objective values agree."
            ),
            policy_id=relation.policy_id,
            policy_version=relation.policy_version,
            definition_sha256=relation.definition_sha256,
        ),
    )


def _check(
    name: str,
    status: EvaluationCheckStatus,
    observed_value: object,
    receipt_locator: str,
) -> EvaluationCheck:
    return EvaluationCheck(
        name=name,
        status=status,
        observed_value=freeze_json(observed_value),
        receipt_locator=receipt_locator,
    )


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoilsDetailedProjectionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BoilsDetailedProjectionError(f"{label} must be finite and non-negative")
    return result


def _validate_circuit_result(
    result: CircuitEvaluation,
    *,
    expected_name: str,
    expected_sha256: str,
    expected_affinity: tuple[int, ...] | None,
    expected_timeout_s: float,
) -> None:
    if type(result) is not CircuitEvaluation:
        raise BoilsDetailedProjectionError(
            "circuit_results must contain exact CircuitEvaluation values"
        )
    if result.circuit_name != expected_name:
        raise BoilsDetailedProjectionError("circuit result order or name changed")
    if result.circuit_sha256 != expected_sha256:
        raise BoilsDetailedProjectionError("circuit result provenance changed")
    for name in ("inputs", "outputs", "lut_count", "edge_count", "aig_count", "levels"):
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise BoilsDetailedProjectionError(
                f"circuit result {name} must be a non-negative exact integer"
            )
    diagnostics = result.diagnostics
    if diagnostics.status != "passed" or not diagnostics.equivalent:
        raise BoilsDetailedProjectionError(
            "successful result lacks a passed equivalence receipt"
        )
    if diagnostics.returncode != 0 or diagnostics.error_signatures:
        raise BoilsDetailedProjectionError(
            "successful result carries ABC failure diagnostics"
        )
    if diagnostics.cpu_affinity != expected_affinity:
        raise BoilsDetailedProjectionError(
            "circuit and aggregate CPU-affinity observations differ"
        )
    _finite_nonnegative(diagnostics.elapsed_s, "circuit elapsed_s")
    _finite_nonnegative(diagnostics.timeout_s, "circuit timeout_s")
    if diagnostics.timeout_s != expected_timeout_s:
        raise BoilsDetailedProjectionError("circuit timeout declaration changed")


def _validate_success_observation(
    result: BoilsEvaluation,
    *,
    configuration: object,
    settings: AbcEvaluatorSettings,
) -> None:
    if type(result) is not BoilsEvaluation:
        raise BoilsDetailedProjectionError(
            "evaluator must return an exact BoilsEvaluation"
        )
    expected_sequence = normalize_candidate(configuration)
    if result.sequence != expected_sequence:
        raise BoilsDetailedProjectionError(
            "evaluation sequence differs from requested configuration"
        )
    if result.configuration_sha256 != config_sha256(configuration):
        raise BoilsDetailedProjectionError(
            "evaluation configuration identity differs from the request"
        )
    if result.abc_binary_sha256 != settings.expected_abc_sha256:
        raise BoilsDetailedProjectionError("evaluation ABC provenance changed")
    if result.lut_inputs != LUT_INPUTS:
        raise BoilsDetailedProjectionError("evaluation LUT mapping contract changed")
    if type(result.circuit_results) is not tuple:
        raise BoilsDetailedProjectionError("circuit_results must be an exact tuple")
    if len(result.circuit_results) != len(settings.circuits):
        raise BoilsDetailedProjectionError("evaluation circuit panel width changed")
    if settings.affinity_sets and result.cpu_affinity not in settings.affinity_sets:
        raise BoilsDetailedProjectionError(
            "evaluation did not use one declared CPU affinity"
        )
    if not settings.affinity_sets and result.cpu_affinity is not None:
        raise BoilsDetailedProjectionError("unpinned evaluator reported CPU affinity")
    for circuit_result, expected in zip(
        result.circuit_results,
        settings.circuits,
        strict=True,
    ):
        _validate_circuit_result(
            circuit_result,
            expected_name=expected.name,
            expected_sha256=expected.expected_sha256,
            expected_affinity=result.cpu_affinity,
            expected_timeout_s=float(settings.per_circuit_timeout_s),
        )
    expected_luts = sum(item.lut_count for item in result.circuit_results)
    expected_levels = sum(item.levels for item in result.circuit_results)
    expected_max_levels = max(item.levels for item in result.circuit_results)
    for name in ("total_lut_count", "total_levels", "max_levels"):
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise BoilsDetailedProjectionError(
                f"{name} must be a non-negative exact integer"
            )
    if result.total_lut_count != expected_luts:
        raise BoilsDetailedProjectionError("total_lut_count projection changed")
    if result.total_levels != expected_levels:
        raise BoilsDetailedProjectionError("total_levels projection changed")
    if result.max_levels != expected_max_levels:
        raise BoilsDetailedProjectionError("max_levels diagnostic projection changed")
    _finite_nonnegative(result.elapsed_s, "evaluation elapsed_s")
    _finite_nonnegative(result.affinity_queue_wait_s, "affinity_queue_wait_s")


def _success_receipt_record(
    result: BoilsEvaluation,
    evaluator_identity: EvaluatorIdentity,
) -> dict[str, object]:
    return {
        "schema_version": BOILS_DETAILED_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "boils_abc_detailed_evaluation",
        "status": "passed",
        "evaluator": evaluator_identity.to_record(),
        "evaluation": result.as_dict(),
    }


def _failure_receipt_record(
    *,
    configuration: object,
    error: AbcEvaluationError,
    evaluator_identity: EvaluatorIdentity,
) -> dict[str, object]:
    sequence = normalize_candidate(configuration)
    return {
        "schema_version": BOILS_DETAILED_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "boils_abc_detailed_evaluation",
        "status": "failed",
        "evaluator": evaluator_identity.to_record(),
        "configuration_sha256": config_sha256(configuration),
        "sequence": list(sequence),
        "failed_circuit_name": error.circuit_name,
        "diagnostics": asdict(error.diagnostics),
    }


@dataclass(frozen=True, slots=True)
class BoilsDetailedEvaluationAdapter:
    """Project a typed BOiLS evaluator through the generic evidence boundary."""

    problem: BoilsAbcProblem
    artifact_store: ArtifactStore
    evaluator_identity: EvaluatorIdentity

    def __post_init__(self) -> None:
        if type(self.problem) is not BoilsAbcProblem:
            raise TypeError("problem must be an exact BoilsAbcProblem")
        if not isinstance(self.artifact_store, ArtifactStore):
            raise TypeError("artifact_store must implement ArtifactStore")
        if type(self.evaluator_identity) is not EvaluatorIdentity:
            raise TypeError("evaluator_identity must be an exact EvaluatorIdentity")
        expected = boils_evaluator_identity(self.problem.settings)
        if self.evaluator_identity != expected:
            raise ValueError("evaluator_identity differs from the BOiLS problem")

    @classmethod
    def build(
        cls,
        *,
        problem: BoilsAbcProblem,
        artifact_store: ArtifactStore,
    ) -> "BoilsDetailedEvaluationAdapter":
        return cls(
            problem=problem,
            artifact_store=artifact_store,
            evaluator_identity=boils_evaluator_identity(problem.settings),
        )

    def _schema_failure(self, error: Exception) -> DetailedEvaluationPayload:
        return DetailedEvaluationPayload(
            failure=FailureRecord(
                category=FailureCategory.CANDIDATE,
                code=FailureCode.SCHEMA_INVALID,
                message=str(error),
                retryable=False,
                exception_type=type(error).__name__,
            ),
            objectives=(),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )

    def _abc_failure(
        self,
        configuration: object,
        error: AbcEvaluationError,
    ) -> DetailedEvaluationPayload:
        record = _failure_receipt_record(
            configuration=configuration,
            error=error,
            evaluator_identity=self.evaluator_identity,
        )
        receipt = put_json(self.artifact_store, record)
        return DetailedEvaluationPayload(
            failure=FailureRecord(
                category=FailureCategory.CANDIDATE,
                code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                message=(
                    f"ABC rejected circuit {error.circuit_name!r}: "
                    f"{error.diagnostics.status}"
                ),
                retryable=False,
                exception_type=type(error).__name__,
                diagnostics_artifact_id=receipt.artifact_id,
            ),
            objectives=(),
            violations=(),
            checks=(
                _check(
                    "abc_evaluation",
                    EvaluationCheckStatus.FAIL,
                    {
                        "failed_circuit_name": error.circuit_name,
                        "status": error.diagnostics.status,
                    },
                    "$.diagnostics",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
        )

    def _contract_failure(
        self,
        error: BoilsDetailedProjectionError,
        *,
        receipt,
    ) -> DetailedEvaluationPayload:
        return DetailedEvaluationPayload(
            failure=FailureRecord(
                category=FailureCategory.SYSTEM,
                code=FailureCode.EVALUATOR_CONTRACT_VIOLATION,
                message=str(error),
                retryable=False,
                exception_type=type(error).__name__,
                diagnostics_artifact_id=receipt.artifact_id,
            ),
            objectives=(),
            violations=(),
            checks=(
                _check(
                    "evaluator_contract",
                    EvaluationCheckStatus.FAIL,
                    {"projection_error": str(error)},
                    "$.evaluation",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
        )

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        self.__post_init__()
        try:
            normalize_candidate(configuration)
        except (TypeError, ValueError) as error:
            return self._schema_failure(error)
        try:
            result = self.problem.evaluator.evaluate(configuration)
        except AbcEvaluationError as error:
            return self._abc_failure(configuration, error)
        if type(result) is not BoilsEvaluation:
            # No typed raw observation exists to persist safely.
            return DetailedEvaluationPayload(
                failure=FailureRecord(
                    category=FailureCategory.SYSTEM,
                    code=FailureCode.EVALUATOR_CONTRACT_VIOLATION,
                    message="BOiLS evaluator returned a foreign observation type",
                    retryable=False,
                    exception_type=type(result).__name__,
                ),
                objectives=(),
                violations=(),
                checks=(),
                receipt=None,
                evaluator=self.evaluator_identity,
            )

        receipt = put_json(
            self.artifact_store,
            _success_receipt_record(result, self.evaluator_identity),
        )
        try:
            _validate_success_observation(
                result,
                configuration=configuration,
                settings=self.problem.settings,
            )
        except BoilsDetailedProjectionError as error:
            return self._contract_failure(error, receipt=receipt)

        circuit_names = [item.circuit_name for item in result.circuit_results]
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(
                (TOTAL_LUT_COUNT, float(result.total_lut_count)),
                (TOTAL_LEVELS, float(result.total_levels)),
            ),
            violations=(),
            checks=(
                _check(
                    "abc_provenance",
                    EvaluationCheckStatus.PASS,
                    {
                        "abc_binary_sha256": result.abc_binary_sha256,
                        "lut_inputs": result.lut_inputs,
                    },
                    "$.evaluation.abc_binary_sha256",
                ),
                _check(
                    "cec_equivalence",
                    EvaluationCheckStatus.PASS,
                    {
                        "circuit_names": circuit_names,
                        "equivalent_count": len(circuit_names),
                    },
                    "$.evaluation.circuit_results",
                ),
                _check(
                    "configuration_identity",
                    EvaluationCheckStatus.PASS,
                    {"configuration_sha256": result.configuration_sha256},
                    "$.evaluation.configuration_sha256",
                ),
                _check(
                    "objective_projection",
                    EvaluationCheckStatus.PASS,
                    {
                        TOTAL_LEVELS: result.total_levels,
                        TOTAL_LUT_COUNT: result.total_lut_count,
                    },
                    "$.evaluation",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=float(result.elapsed_s),
            resource_queue_wall_seconds=float(result.affinity_queue_wait_s),
        )


@dataclass(frozen=True, slots=True)
class BoilsScientificWorkload:
    """Ready-to-inject BOiLS benchmark plus its exact forecast metric space."""

    benchmark: AgenticBenchmark
    decision_metrics: DecisionMetricProjection

    def __post_init__(self) -> None:
        if type(self.benchmark) is not AgenticBenchmark:
            raise TypeError("benchmark must be an exact AgenticBenchmark")
        self.benchmark.validate_binding()
        if type(self.decision_metrics) is not DecisionMetricProjection:
            raise TypeError(
                "decision_metrics must be an exact DecisionMetricProjection"
            )
        semantics = self.benchmark.optimization_semantics
        if type(semantics) is not OptimizationSemantics:
            raise ValueError(
                "BOiLS scientific benchmark requires optimization semantics"
            )
        expected = DecisionMetricProjection.from_optimization_semantics(semantics)
        if self.decision_metrics != expected:
            raise ValueError("decision metric projection differs from BOiLS semantics")


def compose_boils_scientific_workload(
    settings: AbcEvaluatorSettings,
    *,
    artifact_store: ArtifactStore,
    evaluator: DetailedBoilsEvaluator | None = None,
    finite_variation_catalog: FiniteVariationCatalog | None = None,
) -> BoilsScientificWorkload:
    """Compose any pinned BOiLS panel behind the generic AgentEvolve API.

    ``finite_variation_catalog`` is an optional framework-owned search-policy
    injection point.  The benchmark retains schema validation and evaluation
    authority; callers may replace the default atomic palette without adding
    BOiLS knowledge to the evolutionary runtime.
    """

    problem = BoilsAbcProblem(settings, evaluator=evaluator)
    detailed = BoilsDetailedEvaluationAdapter.build(
        problem=problem,
        artifact_store=artifact_store,
    )
    relation = objective_pareto_outcome_binding(tuple(problem.objectives))
    semantics = boils_optimization_semantics(problem)
    benchmark = AgenticBenchmark(
        problem=problem,
        detailed_evaluator=detailed,
        outcome_relation=relation,
        finite_variation_catalogs=(
            BoilsFiniteVariationCatalog()
            if finite_variation_catalog is None
            else finite_variation_catalog,
        ),
        optimization_semantics=semantics,
    )
    return BoilsScientificWorkload(
        benchmark=benchmark,
        decision_metrics=(
            DecisionMetricProjection.from_optimization_semantics(semantics)
        ),
    )


def create_current_sqrt_workload(
    *,
    artifact_store: ArtifactStore,
    affinity_sets: tuple[tuple[int, ...], ...] = (),
    per_circuit_timeout_s: float = 60.0,
    cache_root: Path = Path.home() / ".cache" / "agent_evolve_aaai2027",
) -> BoilsScientificWorkload:
    """Compose the hash-pinned single-``sqrt`` workload.

    The default sequence has measured near 25 seconds on the development host,
    but optimized sequences can be much faster.  Campaigns must therefore
    qualify the latency distribution of their candidate region rather than
    treating the instance name as a 10--30 second guarantee.
    """

    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("sqrt",),
        affinity_sets=affinity_sets,
        per_circuit_timeout_s=per_circuit_timeout_s,
        cache_root=cache_root,
    )
    return compose_boils_scientific_workload(
        settings,
        artifact_store=artifact_store,
    )


__all__ = [
    "BOILS_DETAILED_EVALUATOR_ID",
    "BOILS_DETAILED_EVALUATOR_VERSION",
    "BOILS_DETAILED_RECEIPT_SCHEMA_VERSION",
    "BOILS_EXECUTABLE_ACTIONS_SHA256",
    "BoilsDetailedEvaluationAdapter",
    "BoilsDetailedProjectionError",
    "BoilsScientificWorkload",
    "TOTAL_LEVELS",
    "TOTAL_LUT_COUNT",
    "boils_evaluator_context_record",
    "boils_evaluator_identity",
    "boils_optimization_semantics",
    "compose_boils_scientific_workload",
    "create_current_sqrt_workload",
]
