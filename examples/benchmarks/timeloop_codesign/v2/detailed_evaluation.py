"""Detailed Timeloop v2 evidence behind AgentEvolve's generic evaluator port.

The Docker evaluator owns simulation and validates its pinned runtime boundary.
This module owns the workload-to-framework projection: it binds the exact task
identity, copies the evidence required to replay or audit an observation into a
content-addressed store, and maps only exact static empty-mapspace proofs or
authenticated mapper exhaustion to candidate-attributable failures. Campaign
chronology and model policy remain outside the workload adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    freeze_json,
    objective_pareto_outcome_binding,
)
from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.ports.artifact_store import (
    ArtifactStore,
    decode_json_bytes,
    put_json,
)

from .candidate import (
    REPRESENTATION_ID,
    SCHEMA_VERSION,
    candidate_sha256,
    normalize_candidate,
)
from .compiler import COMPILER_DEFINITION_SHA256
from .container_runner import (
    EVALUATOR_ID,
    MAX_CONSECUTIVE_INVALID_MAPPINGS,
    MAPPER_ALGORITHM,
    MAPPER_THREADS,
    OPTIMIZATION_METRICS,
    SEARCH_SIZE,
)
from .evaluator import (
    OBJECTIVE_NAMES,
    PINNED_IMAGE_ID,
    PINNED_IMAGE_REF,
    ContainerEvaluationResult,
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2ContractError,
    TimeloopV2Evaluation,
    TimeloopV2EvaluatorPort,
    TimeloopV2InfeasibleEvaluation,
    TimeloopV2Settings,
    TimeloopV2StaticInfeasibleEvaluation,
    analyze_static_mapspace_feasibility,
    build_evaluation_bundle,
    canonical_evaluation_bundle_bytes,
)
from .finite_variation_catalog import TimeloopV2FiniteVariationCatalog
from .hard_feasibility import TimeloopV2HardFeasibility
from .network_panel import NetworkLayerPanel, panel_sha256
from .problem_def import TimeloopV2CoDesignProblem
from .runtime_manifest import (
    RUNTIME_TEMPLATE_ID,
    RUNTIME_TEMPLATE_SHA256,
    RUNTIME_TRANSLATOR_DEFINITION_SHA256,
)


TIMELOOP_V2_DETAILED_EVALUATOR_ID = "timeloop_v2_detailed_projection"
TIMELOOP_V2_DETAILED_EVALUATOR_VERSION = 2
TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION = 2
_CONTEXT_HASH_DOMAIN = b"agent-evolve:timeloop-v2:evaluator-context:v2\x00"
_CONTAINER_RUNNER = Path(__file__).with_name("container_runner.py").resolve(strict=True)
_JSON_MEDIA_TYPE = "application/json"
_YAML_MEDIA_TYPE = "application/yaml"
_TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"


class TimeloopV2DetailedProjectionError(RuntimeError):
    """An evaluator observation cannot support the declared projection."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(reference: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": reference.artifact_id.value,
        "sha256_hex": reference.sha256_hex,
        "size_bytes": reference.size_bytes,
        "media_type": reference.media_type,
    }


def _check(
    name: str,
    status: EvaluationCheckStatus,
    observed_value: object,
    locator: str,
) -> EvaluationCheck:
    return EvaluationCheck(
        name=name,
        status=status,
        observed_value=freeze_json(observed_value),
        receipt_locator=locator,
    )


def timeloop_v2_evaluator_context_record(
    settings: TimeloopV2Settings,
    panel: NetworkLayerPanel,
) -> dict[str, object]:
    """Return the portable scientific identity of one Timeloop declaration."""

    if type(settings) is not TimeloopV2Settings:
        raise TypeError("settings must be an exact TimeloopV2Settings")
    if type(panel) is not NetworkLayerPanel:
        raise TypeError("panel must be an exact NetworkLayerPanel")
    settings.__post_init__()
    return {
        "schema_version": 1,
        "adapter": {
            "evaluator_id": TIMELOOP_V2_DETAILED_EVALUATOR_ID,
            "evaluator_version": TIMELOOP_V2_DETAILED_EVALUATOR_VERSION,
            "receipt_schema_version": TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION,
        },
        "candidate_contract": {
            "representation_id": REPRESENTATION_ID,
            "schema_version": SCHEMA_VERSION,
            "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
            "runtime_translator_definition_sha256": (
                RUNTIME_TRANSLATOR_DEFINITION_SHA256
            ),
            "runtime_template_id": RUNTIME_TEMPLATE_ID,
            "runtime_template_sha256": RUNTIME_TEMPLATE_SHA256,
        },
        "task": {
            "panel_id": panel.panel_id,
            "network_id": panel.network_id,
            "panel_role": panel.role,
            "panel_sha256": panel_sha256(panel),
            "supported_conv_layer_count": panel.supported_conv_layer_count,
            "medoid_multiplicities": [item.multiplicity for item in panel.medoids()],
            "objective_names": list(OBJECTIVE_NAMES),
            "aggregation": (
                "multiplicity_weighted_energy_and_latency_chip_area_counted_once"
            ),
        },
        "runtime": {
            "container_evaluator_id": EVALUATOR_ID,
            "image_ref": PINNED_IMAGE_REF,
            "image_id": PINNED_IMAGE_ID,
            "container_runner_sha256": _sha256_file(_CONTAINER_RUNNER),
            "network_access": "none",
            "outer_timeout_s_hex": float(settings.timeout_s).hex(),
        },
        "mapper_protocol": {
            "search_size": SEARCH_SIZE,
            "threads": MAPPER_THREADS,
            "algorithm": MAPPER_ALGORITHM,
            "max_consecutive_invalid_mappings": (
                MAX_CONSECUTIVE_INVALID_MAPPINGS
            ),
            "optimization_metrics": list(OPTIMIZATION_METRICS),
            "seed_law": (
                "pinned_binary_random_pruned_default_constructed_cpp_engine"
            ),
        },
        "candidate_infeasibility": {
            "static_mapspace_law": (
                "candidate_infeasible_iff_a_primary_axis_extent_has_no_integer_"
                "divisor_inside_its_compiler_emitted_inclusive_spatial_bounds"
            ),
            "static_check_scope": "before_native_timeloop_invocation",
            "dynamic_check_scope": (
                "authenticated_frozen_consecutive_invalid_mapping_budget"
            ),
            "ambiguous_native_failure_category": "system",
        },
    }


def timeloop_v2_evaluator_identity(
    settings: TimeloopV2Settings,
    panel: NetworkLayerPanel,
) -> EvaluatorIdentity:
    context = timeloop_v2_evaluator_context_record(settings, panel)
    return EvaluatorIdentity(
        evaluator_id=TIMELOOP_V2_DETAILED_EVALUATOR_ID,
        evaluator_version=TIMELOOP_V2_DETAILED_EVALUATOR_VERSION,
        evaluator_context_sha256=hashlib.sha256(
            _CONTEXT_HASH_DOMAIN + _canonical_json_bytes(context)
        ).hexdigest(),
    )


def timeloop_v2_optimization_semantics(
    problem: TimeloopV2CoDesignProblem,
) -> OptimizationSemantics:
    """Publish the exact raw network-level Pareto semantics.

    The semantics live beside the detailed evaluator because this adapter owns
    the projection from Timeloop/Accelergy evidence into stable decision
    metrics.  Campaigns and model transports consume this versioned contract;
    they do not infer units, aggregation, or objective senses from names.
    """

    if type(problem) is not TimeloopV2CoDesignProblem:
        raise TypeError("problem must be an exact TimeloopV2CoDesignProblem")
    objectives = tuple(problem.objectives)
    relation = objective_pareto_outcome_binding(objectives)
    panel = problem.panel
    panel_text = f"{panel.network_id} ({panel.role})"
    metrics = tuple(
        sorted(
            (
                MetricSemantics(
                    metric_id="objective:area_square_meters",
                    name="area_square_meters",
                    role=MetricRole.OBJECTIVE,
                    sense=MetricSense.MINIMIZE,
                    definition=(
                        "Chip area in square meters reported by the pinned "
                        "Accelergy/Timeloop evaluation for the shared accelerator "
                        f"architecture on {panel_text}."
                    ),
                    aggregation=(
                        "Counted once for the shared chip architecture; it is not "
                        "multiplied by layer-medoid multiplicity."
                    ),
                    witness_interpretation="A lower finite chip area is better.",
                    tolerance=0.0,
                ),
                MetricSemantics(
                    metric_id="objective:energy_joules",
                    name="energy_joules",
                    role=MetricRole.OBJECTIVE,
                    sense=MetricSense.MINIMIZE,
                    definition=(
                        "Network inference energy in joules from the pinned "
                        f"Timeloop mapping protocol on {panel_text}."
                    ),
                    aggregation=(
                        "Sum of medoid energy observations weighted by the frozen "
                        "supported-layer multiplicities."
                    ),
                    witness_interpretation="A lower finite network energy is better.",
                    tolerance=0.0,
                ),
                MetricSemantics(
                    metric_id="objective:latency_seconds",
                    name="latency_seconds",
                    role=MetricRole.OBJECTIVE,
                    sense=MetricSense.MINIMIZE,
                    definition=(
                        "Network inference latency in seconds from the pinned "
                        f"Timeloop mapping protocol on {panel_text}."
                    ),
                    aggregation=(
                        "Sum of medoid latency observations weighted by the frozen "
                        "supported-layer multiplicities."
                    ),
                    witness_interpretation="A lower finite network latency is better.",
                    tolerance=0.0,
                ),
            ),
            key=lambda value: value.metric_id,
        )
    )
    return OptimizationSemantics(
        semantics_id="timeloop_v2_raw_network_pareto",
        semantics_version=1,
        metrics=metrics,
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=tuple(value.metric_id for value in metrics),
            description=(
                "Pareto minimization of raw network energy, raw network latency, "
                "and shared chip area; no objective is lexicographically privileged."
            ),
            equivalence=(
                "Two successful outcomes are equivalent exactly when all three "
                "raw floating-point decision values agree."
            ),
            policy_id=relation.policy_id,
            policy_version=relation.policy_version,
            definition_sha256=relation.definition_sha256,
        ),
    )


def _required_evidence_files(
    observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation,
) -> tuple[tuple[str, str], ...]:
    paths: list[tuple[str, str]] = [
        ("evaluation-bundle.json", _JSON_MEDIA_TYPE),
        ("result.json", _JSON_MEDIA_TYPE),
        ("host_receipt.json", _JSON_MEDIA_TYPE),
    ]
    for layer in observation.layer_results:
        prefix = f"timeloop-output/{layer.output_subdirectory}"
        paths.extend(
            (
                (f"{prefix}/timeloop-mapper.map.yaml", _YAML_MEDIA_TYPE),
                (f"{prefix}/parsed-processed-input.yaml", _YAML_MEDIA_TYPE),
                (f"{prefix}/output.log", _TEXT_MEDIA_TYPE),
            )
        )
    return tuple(paths)


def _persist_evidence_files(
    store: ArtifactStore,
    observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative_path, media_type in _required_evidence_files(observation):
        path = observation.output_dir / relative_path
        if not path.is_file():
            raise TimeloopV2DetailedProjectionError(
                f"required evaluator evidence is missing: {relative_path}"
            )
        reference = store.put_bytes(path.read_bytes(), media_type=media_type)
        records.append(
            {
                "relative_path": relative_path,
                "artifact": _artifact_record(reference),
            }
        )
    return records


def _validate_common_observation(
    configuration: object,
    observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation,
    *,
    panel: NetworkLayerPanel,
    expected_status: str,
) -> dict[str, object]:
    expected_bundle = build_evaluation_bundle(configuration, panel)
    if (
        observation.candidate_sha256 != expected_bundle.candidate_sha256
        or observation.compiled_plan_sha256 != expected_bundle.compiled_plan_sha256
        or observation.panel_sha256 != expected_bundle.panel_sha256
    ):
        raise TimeloopV2DetailedProjectionError(
            "evaluation identity differs from the requested candidate"
        )
    if type(observation.manifest) is not ContainerEvaluationResult:
        raise TimeloopV2DetailedProjectionError(
            "evaluation manifest must be an exact ContainerEvaluationResult"
        )
    manifest = observation.manifest
    if (
        manifest.candidate_sha256 != observation.candidate_sha256
        or manifest.compiled_plan_sha256 != observation.compiled_plan_sha256
        or manifest.panel_sha256 != observation.panel_sha256
        or manifest.protocol != expected_bundle.protocol
    ):
        raise TimeloopV2DetailedProjectionError("evaluation manifest provenance drift")
    if manifest.layers != observation.layer_results:
        raise TimeloopV2DetailedProjectionError("layer evidence differs from manifest")
    for name in ("evaluator_elapsed_s", "queue_wait_s"):
        value = getattr(observation, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TimeloopV2DetailedProjectionError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise TimeloopV2DetailedProjectionError(
                f"{name} must be finite and non-negative"
            )

    result_path = observation.output_dir / "result.json"
    host_path = observation.output_dir / "host_receipt.json"
    if not result_path.is_file() or not host_path.is_file():
        raise TimeloopV2DetailedProjectionError(
            "result or host receipt is missing from the evaluation directory"
        )
    host = decode_json_bytes(host_path.read_bytes())
    if not isinstance(host, dict):
        raise TimeloopV2DetailedProjectionError("host receipt is not a JSON object")
    expected_host = {
        "schema_version": 2,
        "evaluator_id": EVALUATOR_ID,
        "status": expected_status,
        "candidate_sha256": observation.candidate_sha256,
        "compiled_plan_sha256": observation.compiled_plan_sha256,
        "panel_sha256": observation.panel_sha256,
        "result_sha256": _sha256_file(result_path),
        "image_ref": PINNED_IMAGE_REF,
        "image_id": PINNED_IMAGE_ID,
        "runner_sha256": _sha256_file(_CONTAINER_RUNNER),
        "search_size": SEARCH_SIZE,
        "mapper_threads": MAPPER_THREADS,
        "max_consecutive_invalid_mappings": MAX_CONSECUTIVE_INVALID_MAPPINGS,
        "evaluator_elapsed_s": observation.evaluator_elapsed_s,
        "queue_wait_s": observation.queue_wait_s,
    }
    for key, expected in expected_host.items():
        if host.get(key) != expected:
            raise TimeloopV2DetailedProjectionError(
                f"host receipt field {key!r} changed"
            )

    for ordinal, layer in enumerate(observation.layer_results):
        if (
            layer.medoid_ordinal != ordinal
            or layer.layer_manifest_sha256
            != expected_bundle.layer_manifest_sha256[ordinal]
        ):
            raise TimeloopV2DetailedProjectionError("layer provenance drift")
        layer_root = observation.output_dir / "timeloop-output" / f"medoid-{ordinal}"
        mapping_path = layer_root / "timeloop-mapper.map.yaml"
        processed_path = layer_root / "parsed-processed-input.yaml"
        if (
            not mapping_path.is_file()
            or _sha256_file(mapping_path) != layer.mapping_sha256
            or not processed_path.is_file()
            or _sha256_file(processed_path) != layer.processed_input_sha256
        ):
            raise TimeloopV2DetailedProjectionError("layer evidence digest drift")
    return host


def _observation_receipt_record(
    *,
    configuration: object,
    observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation,
    status: str,
    identity: EvaluatorIdentity,
    store: ArtifactStore,
    panel: NetworkLayerPanel,
) -> dict[str, object]:
    host = _validate_common_observation(
        configuration,
        observation,
        panel=panel,
        expected_status=status,
    )
    return {
        "schema_version": TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "timeloop_v2_detailed_evaluation",
        "status": status,
        "evaluator": identity.to_record(),
        "configuration": normalize_candidate(configuration).model_dump(mode="json"),
        "evaluation": {
            "candidate_sha256": observation.candidate_sha256,
            "compiled_plan_sha256": observation.compiled_plan_sha256,
            "panel_sha256": observation.panel_sha256,
            "evaluator_elapsed_s": observation.evaluator_elapsed_s,
            "queue_wait_s": observation.queue_wait_s,
            "manifest": observation.manifest.model_dump(mode="json"),
            "host_receipt": host,
        },
        "source_artifacts": _persist_evidence_files(store, observation),
    }


@dataclass(frozen=True, slots=True)
class TimeloopV2DetailedEvaluationAdapter:
    """Project one pinned Timeloop problem through generic detailed evidence."""

    problem: TimeloopV2CoDesignProblem
    artifact_store: ArtifactStore
    evaluator_identity: EvaluatorIdentity

    def __post_init__(self) -> None:
        if type(self.problem) is not TimeloopV2CoDesignProblem:
            raise TypeError("problem must be an exact TimeloopV2CoDesignProblem")
        if not isinstance(self.artifact_store, ArtifactStore):
            raise TypeError("artifact_store must implement ArtifactStore")
        if type(self.evaluator_identity) is not EvaluatorIdentity:
            raise TypeError("evaluator_identity must be an exact EvaluatorIdentity")
        expected = timeloop_v2_evaluator_identity(
            self.problem.settings,
            self.problem.panel,
        )
        if self.evaluator_identity != expected:
            raise ValueError("evaluator_identity differs from the Timeloop problem")

    @classmethod
    def build(
        cls,
        *,
        problem: TimeloopV2CoDesignProblem,
        artifact_store: ArtifactStore,
    ) -> "TimeloopV2DetailedEvaluationAdapter":
        return cls(
            problem=problem,
            artifact_store=artifact_store,
            evaluator_identity=timeloop_v2_evaluator_identity(
                problem.settings,
                problem.panel,
            ),
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

    def _contract_failure(
        self,
        configuration: object,
        error: Exception,
    ) -> DetailedEvaluationPayload:
        receipt = put_json(
            self.artifact_store,
            {
                "schema_version": TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION,
                "receipt_kind": "timeloop_v2_detailed_evaluation",
                "status": "evaluator_contract_failure",
                "evaluator": self.evaluator_identity.to_record(),
                "candidate_sha256": candidate_sha256(configuration),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
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
                    {"error_type": type(error).__name__, "message": str(error)},
                    "$.message",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
        )

    def _candidate_infeasible(
        self,
        configuration: object,
        error: TimeloopV2CandidateInfeasibleError,
    ) -> DetailedEvaluationPayload:
        observation = error.observation
        if type(observation) is TimeloopV2StaticInfeasibleEvaluation:
            return self._static_candidate_infeasible(
                configuration,
                error,
                observation,
            )
        try:
            record = _observation_receipt_record(
                configuration=configuration,
                observation=observation,
                status="candidate_infeasible",
                identity=self.evaluator_identity,
                store=self.artifact_store,
                panel=self.problem.panel,
            )
        except TimeloopV2DetailedProjectionError as projection_error:
            return self._contract_failure(configuration, projection_error)
        receipt = put_json(self.artifact_store, record)
        return DetailedEvaluationPayload(
            failure=FailureRecord(
                category=FailureCategory.CANDIDATE,
                code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                message=str(error),
                retryable=False,
                exception_type=type(error).__name__,
                diagnostics_artifact_id=receipt.artifact_id,
            ),
            objectives=(),
            violations=(),
            checks=(
                _check(
                    "candidate_identity",
                    EvaluationCheckStatus.PASS,
                    {"candidate_sha256": observation.candidate_sha256},
                    "$.evaluation.candidate_sha256",
                ),
                _check(
                    "mapping_budget",
                    EvaluationCheckStatus.FAIL,
                    {
                        "incomplete_medoid_ordinals": list(
                            observation.incomplete_medoid_ordinals
                        ),
                        "requested_valid_mapping_count": SEARCH_SIZE,
                    },
                    "$.evaluation.manifest.layers",
                ),
                _check(
                    "runtime_provenance",
                    EvaluationCheckStatus.PASS,
                    {"runner_sha256": _sha256_file(_CONTAINER_RUNNER)},
                    "$.evaluation.manifest.provenance",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=float(observation.evaluator_elapsed_s),
            resource_queue_wall_seconds=float(observation.queue_wait_s),
        )

    def _static_candidate_infeasible(
        self,
        configuration: object,
        error: TimeloopV2CandidateInfeasibleError,
        observation: TimeloopV2StaticInfeasibleEvaluation,
    ) -> DetailedEvaluationPayload:
        try:
            bundle = build_evaluation_bundle(configuration, self.problem.panel)
            bundle_bytes = canonical_evaluation_bundle_bytes(bundle)
            if (
                bundle.candidate_sha256 != observation.candidate_sha256
                or bundle.compiled_plan_sha256 != observation.compiled_plan_sha256
                or bundle.panel_sha256 != observation.panel_sha256
                or hashlib.sha256(bundle_bytes).hexdigest()
                != observation.evaluation_bundle_sha256
                or analyze_static_mapspace_feasibility(bundle) != observation
            ):
                raise TimeloopV2DetailedProjectionError(
                    "static infeasibility evidence differs from the requested candidate"
                )
            bundle_ref = self.artifact_store.put_bytes(
                bundle_bytes + b"\n",
                media_type=_JSON_MEDIA_TYPE,
            )
            witness_records = [
                {
                    "medoid_ordinal": value.medoid_ordinal,
                    "primary_axis": value.primary_axis,
                    "axis_extent": value.axis_extent,
                    "minimum_parallelism": value.minimum_parallelism,
                    "maximum_parallelism": value.maximum_parallelism,
                    "admissible_spatial_factors": list(
                        value.admissible_spatial_factors
                    ),
                }
                for value in observation.witnesses
            ]
            record = {
                "schema_version": TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION,
                "receipt_kind": "timeloop_v2_detailed_evaluation",
                "status": "candidate_infeasible_static_mapspace",
                "evaluator": self.evaluator_identity.to_record(),
                "configuration": normalize_candidate(configuration).model_dump(
                    mode="json"
                ),
                "evaluation": {
                    "candidate_sha256": observation.candidate_sha256,
                    "compiled_plan_sha256": observation.compiled_plan_sha256,
                    "panel_sha256": observation.panel_sha256,
                    "evaluation_bundle_sha256": (
                        observation.evaluation_bundle_sha256
                    ),
                    "static_mapspace_witnesses": witness_records,
                    "native_simulator_invoked": False,
                },
                "source_artifacts": [
                    {
                        "relative_path": "evaluation-bundle.json",
                        "artifact": _artifact_record(bundle_ref),
                    }
                ],
            }
        except TimeloopV2DetailedProjectionError as projection_error:
            return self._contract_failure(configuration, projection_error)
        receipt = put_json(self.artifact_store, record)
        return DetailedEvaluationPayload(
            failure=FailureRecord(
                category=FailureCategory.CANDIDATE,
                code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                message=str(error),
                retryable=False,
                exception_type=type(error).__name__,
                diagnostics_artifact_id=receipt.artifact_id,
            ),
            objectives=(),
            violations=(),
            checks=(
                _check(
                    "candidate_identity",
                    EvaluationCheckStatus.PASS,
                    {"candidate_sha256": observation.candidate_sha256},
                    "$.evaluation.candidate_sha256",
                ),
                _check(
                    "mapspace_nonempty",
                    EvaluationCheckStatus.FAIL,
                    {"static_mapspace_witnesses": witness_records},
                    "$.evaluation.static_mapspace_witnesses",
                ),
                _check(
                    "native_simulator_invocation",
                    EvaluationCheckStatus.NOT_APPLICABLE,
                    {"native_simulator_invoked": False},
                    "$.evaluation.native_simulator_invoked",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=0.0,
            resource_queue_wall_seconds=0.0,
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
            bundle = build_evaluation_bundle(configuration, self.problem.panel)
            static_infeasibility = analyze_static_mapspace_feasibility(bundle)
        except (TypeError, ValueError, TimeloopV2ContractError) as error:
            return self._contract_failure(configuration, error)
        if static_infeasibility is not None:
            return self._candidate_infeasible(
                configuration,
                TimeloopV2CandidateInfeasibleError(static_infeasibility),
            )
        try:
            observation = self.problem.evaluate_detailed(configuration)
        except TimeloopV2CandidateInfeasibleError as error:
            return self._candidate_infeasible(configuration, error)
        except TimeloopV2ContractError as error:
            return self._contract_failure(configuration, error)
        if type(observation) is not TimeloopV2Evaluation:
            return self._contract_failure(
                configuration,
                TimeloopV2DetailedProjectionError(
                    "Timeloop evaluator returned a foreign observation type"
                ),
            )
        try:
            record = _observation_receipt_record(
                configuration=configuration,
                observation=observation,
                status="passed",
                identity=self.evaluator_identity,
                store=self.artifact_store,
                panel=self.problem.panel,
            )
            if any(
                not layer.mapping_budget_complete
                for layer in observation.layer_results
            ):
                raise TimeloopV2DetailedProjectionError(
                    "successful evaluation contains an incomplete mapper budget"
                )
            objective_values = observation.objective_values
            if set(objective_values) != set(OBJECTIVE_NAMES):
                raise TimeloopV2DetailedProjectionError(
                    "successful observation changed the objective projection"
                )
            for name in OBJECTIVE_NAMES:
                value = objective_values[name]
                manifest_value = float(getattr(observation.manifest.objectives, name))
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) != manifest_value
                ):
                    raise TimeloopV2DetailedProjectionError(
                        f"objective {name!r} differs from the raw manifest"
                    )
        except TimeloopV2DetailedProjectionError as error:
            return self._contract_failure(configuration, error)
        receipt = put_json(self.artifact_store, record)
        return DetailedEvaluationPayload(
            failure=None,
            objectives=tuple(
                (name, float(objective_values[name])) for name in OBJECTIVE_NAMES
            ),
            violations=(),
            checks=(
                _check(
                    "candidate_identity",
                    EvaluationCheckStatus.PASS,
                    {"candidate_sha256": observation.candidate_sha256},
                    "$.evaluation.candidate_sha256",
                ),
                _check(
                    "compiled_plan_identity",
                    EvaluationCheckStatus.PASS,
                    {"compiled_plan_sha256": observation.compiled_plan_sha256},
                    "$.evaluation.compiled_plan_sha256",
                ),
                _check(
                    "evidence_files",
                    EvaluationCheckStatus.PASS,
                    {"preserved_file_count": len(record["source_artifacts"])},
                    "$.source_artifacts",
                ),
                _check(
                    "mapping_budget",
                    EvaluationCheckStatus.PASS,
                    {
                        "complete_medoid_count": len(observation.layer_results),
                        "requested_valid_mapping_count": SEARCH_SIZE,
                    },
                    "$.evaluation.manifest.layers",
                ),
                _check(
                    "objective_projection",
                    EvaluationCheckStatus.PASS,
                    {
                        name: objective_values[name] for name in OBJECTIVE_NAMES
                    },
                    "$.evaluation.manifest.objectives",
                ),
                _check(
                    "runtime_provenance",
                    EvaluationCheckStatus.PASS,
                    {
                        "image_id": PINNED_IMAGE_ID,
                        "runner_sha256": _sha256_file(_CONTAINER_RUNNER),
                    },
                    "$.evaluation.host_receipt",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=float(observation.evaluator_elapsed_s),
            resource_queue_wall_seconds=float(observation.queue_wait_s),
        )


def compose_timeloop_v2_detailed_benchmark(
    settings: TimeloopV2Settings,
    panel: NetworkLayerPanel,
    *,
    artifact_store: ArtifactStore,
    evaluator: TimeloopV2EvaluatorPort | None = None,
) -> AgenticBenchmark:
    """Compose any Timeloop v2 panel behind the generic AgentEvolve API."""

    problem = TimeloopV2CoDesignProblem(settings, panel, evaluator=evaluator)
    detailed = TimeloopV2DetailedEvaluationAdapter.build(
        problem=problem,
        artifact_store=artifact_store,
    )
    relation = objective_pareto_outcome_binding(tuple(problem.objectives))
    return AgenticBenchmark(
        problem=problem,
        detailed_evaluator=detailed,
        outcome_relation=relation,
        optimization_semantics=timeloop_v2_optimization_semantics(problem),
        finite_variation_catalogs=(TimeloopV2FiniteVariationCatalog(panel),),
        hard_feasibility=TimeloopV2HardFeasibility(panel),
    )


__all__ = [
    "TIMELOOP_V2_DETAILED_EVALUATOR_ID",
    "TIMELOOP_V2_DETAILED_EVALUATOR_VERSION",
    "TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION",
    "TimeloopV2DetailedEvaluationAdapter",
    "TimeloopV2DetailedProjectionError",
    "compose_timeloop_v2_detailed_benchmark",
    "timeloop_v2_evaluator_context_record",
    "timeloop_v2_evaluator_identity",
    "timeloop_v2_optimization_semantics",
]
