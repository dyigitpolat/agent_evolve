"""Host-side Docker evaluator for the frozen Timeloop v2 contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Annotated, Any, Literal, Protocol, runtime_checkable
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate import normalize_candidate
from .compiler import TimeloopV2Compiler
from .container_runner import (
    EVALUATOR_ID,
    MAX_CONSECUTIVE_INVALID_MAPPINGS,
    MAPPER_ALGORITHM,
    MAPPER_THREADS,
    OPTIMIZATION_METRICS,
    SEARCH_SIZE,
)
from .network_panel import NetworkLayerPanel
from .runtime_manifest import (
    RuntimeLayerManifest,
    RuntimeSpatialConstraint,
    compile_runtime_layer_manifests,
    runtime_layer_manifest_sha256,
)


PINNED_IMAGE_REF = (
    "timeloopaccelergy/accelergy-timeloop-infrastructure@"
    "sha256:c027f7f57af124ca8500f05f6687ffb0875da4a5b1186934d7daf6a18b143ea2"
)
PINNED_IMAGE_ID = (
    "sha256:c027f7f57af124ca8500f05f6687ffb0875da4a5b1186934d7daf6a18b143ea2"
)
OBJECTIVE_NAMES = (
    "energy_joules",
    "latency_seconds",
    "area_square_meters",
)
_CPU_SET = re.compile(r"^[0-9]+$")
_EXTERNAL_LOCK = threading.Lock()

PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]
PositiveStrictFloat = Annotated[float, Field(strict=True, gt=0)]


class TimeloopV2ContractError(RuntimeError):
    """The pinned simulator or result violated the v2 boundary."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class MapperProtocol(_ClosedModel):
    search_size: Literal[SEARCH_SIZE] = SEARCH_SIZE
    mapper_threads: Literal[MAPPER_THREADS] = MAPPER_THREADS
    mapper_algorithm: Literal[MAPPER_ALGORITHM] = MAPPER_ALGORITHM
    max_consecutive_invalid_mappings: Literal[MAX_CONSECUTIVE_INVALID_MAPPINGS] = (
        MAX_CONSECUTIVE_INVALID_MAPPINGS
    )
    optimization_metrics: tuple[Literal["edp"], ...] = OPTIMIZATION_METRICS


class EvaluationBundle(_ClosedModel):
    schema_version: Literal[1] = 1
    evaluator_id: Literal[EVALUATOR_ID] = EVALUATOR_ID
    candidate_sha256: str
    compiled_plan_sha256: str
    panel_sha256: str
    layer_manifest_sha256: tuple[str, str, str]
    layer_manifests: tuple[
        RuntimeLayerManifest,
        RuntimeLayerManifest,
        RuntimeLayerManifest,
    ]
    protocol: MapperProtocol = Field(default_factory=MapperProtocol)

    @model_validator(mode="after")
    def _bind_manifests(self) -> "EvaluationBundle":
        for ordinal, (manifest, digest) in enumerate(
            zip(self.layer_manifests, self.layer_manifest_sha256, strict=True)
        ):
            if manifest.medoid_ordinal != ordinal:
                raise ValueError("layer manifests must use ordinal order")
            if (
                manifest.candidate_sha256 != self.candidate_sha256
                or manifest.compiled_plan_sha256 != self.compiled_plan_sha256
                or manifest.panel_sha256 != self.panel_sha256
            ):
                raise ValueError("layer manifest provenance drift")
            if runtime_layer_manifest_sha256(manifest) != digest:
                raise ValueError("layer manifest digest mismatch")
        return self


class LayerContainerResult(_ClosedModel):
    medoid_ordinal: Literal[0, 1, 2]
    layer_multiplicity: PositiveStrictInt
    layer_manifest_sha256: str
    energy_joules: PositiveStrictFloat
    latency_seconds: PositiveStrictFloat
    area_square_meters: PositiveStrictFloat
    cycles: PositiveStrictInt
    computes: PositiveStrictInt
    requested_valid_mapping_count: Literal[SEARCH_SIZE]
    reported_valid_mapping_count: PositiveStrictInt | None
    consecutive_invalid_mapping_count: int
    mapping_budget_complete: bool
    termination_reason: Literal[
        "valid_mapping_target",
        "consecutive_invalid_limit",
    ]
    elapsed_s: PositiveStrictFloat
    mapping_sha256: str
    processed_input_sha256: str
    output_subdirectory: Literal["medoid-0", "medoid-1", "medoid-2"]
    front_end_projection_exact: Literal[True]

    @model_validator(mode="after")
    def _validate_termination(self) -> "LayerContainerResult":
        if self.consecutive_invalid_mapping_count < 0:
            raise ValueError("invalid mapping count cannot be negative")
        if self.mapping_budget_complete:
            if (
                self.termination_reason != "valid_mapping_target"
                or self.reported_valid_mapping_count != SEARCH_SIZE
                or self.consecutive_invalid_mapping_count != 0
            ):
                raise ValueError("completed mapping-budget receipt is inconsistent")
        elif (
            self.termination_reason != "consecutive_invalid_limit"
            or self.reported_valid_mapping_count is not None
            or self.consecutive_invalid_mapping_count <= 0
        ):
            raise ValueError("invalid-limit receipt is inconsistent")
        return self


class ObjectiveValues(_ClosedModel):
    energy_joules: PositiveStrictFloat
    latency_seconds: PositiveStrictFloat
    area_square_meters: PositiveStrictFloat


class ProvenanceReceipt(_ClosedModel):
    asset_sha256: dict[str, str]
    runner_sha256: str
    python_hash_seed: Literal["0"]
    mapper_seed_law: Literal[
        "pinned_binary_random_pruned_default_constructed_cpp_engine"
    ]


class ContainerEvaluationResult(_ClosedModel):
    schema_version: Literal[1]
    evaluator_id: Literal[EVALUATOR_ID]
    candidate_sha256: str
    compiled_plan_sha256: str
    panel_sha256: str
    objectives: ObjectiveValues
    layers: tuple[LayerContainerResult, LayerContainerResult, LayerContainerResult]
    protocol: MapperProtocol
    provenance: ProvenanceReceipt


def _require_complete_mapping_budget(layer: LayerContainerResult) -> None:
    if not layer.mapping_budget_complete:
        raise TimeloopV2ContractError(
            "layer did not complete the frozen valid-mapping budget: "
            f"medoid={layer.medoid_ordinal}, termination={layer.termination_reason}, "
            "consecutive_invalid_mappings="
            f"{layer.consecutive_invalid_mapping_count}"
        )


def canonical_evaluation_bundle_bytes(bundle: EvaluationBundle) -> bytes:
    if type(bundle) is not EvaluationBundle:
        raise TypeError("bundle must be an exact EvaluationBundle")
    return json.dumps(
        bundle.model_dump(mode="python"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def build_evaluation_bundle(
    config: object, panel: NetworkLayerPanel
) -> EvaluationBundle:
    candidate = normalize_candidate(config)
    if type(panel) is not NetworkLayerPanel:
        raise TypeError("panel must be an exact NetworkLayerPanel")
    compilation = TimeloopV2Compiler.compile(candidate, panel)
    manifests = compile_runtime_layer_manifests(compilation)
    return EvaluationBundle(
        candidate_sha256=compilation.candidate_sha256,
        compiled_plan_sha256=compilation.compiled_plan_sha256,
        panel_sha256=compilation.panel_sha256,
        layer_manifest_sha256=tuple(
            runtime_layer_manifest_sha256(manifest) for manifest in manifests
        ),
        layer_manifests=manifests,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise TimeloopV2ContractError(f"nonstandard JSON constant: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TimeloopV2ContractError(f"duplicate result key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )
    if type(value) is not dict:
        raise TimeloopV2ContractError("result must be an exact JSON object")
    return value


def _atomic_json_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload + b"\n")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class TimeloopV2Settings:
    output_root: Path
    cpu_set: str = "8"
    timeout_s: float = 180.0
    image_ref: str = PINNED_IMAGE_REF
    expected_image_id: str = PINNED_IMAGE_ID
    search_size: int = SEARCH_SIZE
    mapper_threads: int = MAPPER_THREADS
    max_consecutive_invalid_mappings: int = MAX_CONSECUTIVE_INVALID_MAPPINGS
    external_concurrency: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a pathlib.Path")
        if type(self.cpu_set) is not str or _CPU_SET.fullmatch(self.cpu_set) is None:
            raise ValueError("cpu_set must identify exactly one logical CPU")
        if isinstance(self.timeout_s, bool) or not isinstance(
            self.timeout_s, (int, float)
        ):
            raise TypeError("timeout_s must be numeric")
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if (
            self.image_ref != PINNED_IMAGE_REF
            or self.expected_image_id != PINNED_IMAGE_ID
        ):
            raise ValueError("the v2 evaluator image is pinned")
        if self.search_size != SEARCH_SIZE or self.mapper_threads != MAPPER_THREADS:
            raise ValueError("the v2 mapper protocol is frozen")
        if self.max_consecutive_invalid_mappings != MAX_CONSECUTIVE_INVALID_MAPPINGS:
            raise ValueError("the v2 invalid-mapping limit is frozen")
        if self.external_concurrency != 1:
            raise ValueError("the initial v2 adapter is serialized")


@dataclass(frozen=True, slots=True)
class TimeloopV2Evaluation:
    objective_values: dict[str, float]
    output_dir: Path
    candidate_sha256: str
    compiled_plan_sha256: str
    panel_sha256: str
    evaluator_elapsed_s: float
    queue_wait_s: float
    layer_results: tuple[
        LayerContainerResult, LayerContainerResult, LayerContainerResult
    ]
    manifest: ContainerEvaluationResult


@dataclass(frozen=True, slots=True)
class TimeloopV2InfeasibleEvaluation:
    """Authenticated mapper exhaustion attributable to one candidate.

    This is deliberately not a successful objective observation.  The pinned
    container completed normally and published hash-bound evidence, but at
    least one medoid hit the frozen consecutive-invalid limit before finding
    the requested number of valid mappings.  Keeping this record separate
    prevents partial mapper objectives from entering Pareto selection while
    still preserving the evidence needed for candidate-level recourse.
    """

    output_dir: Path
    candidate_sha256: str
    compiled_plan_sha256: str
    panel_sha256: str
    evaluator_elapsed_s: float
    queue_wait_s: float
    layer_results: tuple[
        LayerContainerResult, LayerContainerResult, LayerContainerResult
    ]
    manifest: ContainerEvaluationResult
    incomplete_medoid_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TimeloopV2StaticMapspaceWitness:
    """One compiler-bound proof that a medoid's spatial mapspace is empty."""

    medoid_ordinal: int
    primary_axis: str
    axis_extent: int
    minimum_parallelism: int
    maximum_parallelism: int
    admissible_spatial_factors: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.medoid_ordinal not in {0, 1, 2}:
            raise ValueError("medoid_ordinal must identify the frozen panel")
        if self.primary_axis not in {"C", "M", "P", "Q"}:
            raise ValueError("primary_axis is outside the candidate contract")
        for name in ("axis_extent", "minimum_parallelism", "maximum_parallelism"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.minimum_parallelism > self.maximum_parallelism:
            raise ValueError("parallelism lower bound exceeds its upper bound")
        expected = tuple(
            factor
            for factor in range(
                self.minimum_parallelism,
                self.maximum_parallelism + 1,
            )
            if self.axis_extent % factor == 0
        )
        if self.admissible_spatial_factors != expected:
            raise ValueError("admissible factor witness is not exact")
        if self.admissible_spatial_factors:
            raise ValueError("a static infeasibility witness must prove an empty set")


@dataclass(frozen=True, slots=True)
class TimeloopV2StaticInfeasibleEvaluation:
    """Simulator-free proof that selected spatial constraints admit no mapping."""

    candidate_sha256: str
    compiled_plan_sha256: str
    panel_sha256: str
    evaluation_bundle_sha256: str
    witnesses: tuple[TimeloopV2StaticMapspaceWitness, ...]

    def __post_init__(self) -> None:
        for value in (
            self.candidate_sha256,
            self.compiled_plan_sha256,
            self.panel_sha256,
            self.evaluation_bundle_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("static infeasibility identity must use SHA-256")
        if not self.witnesses:
            raise ValueError("static infeasibility requires at least one witness")
        ordinals = tuple(value.medoid_ordinal for value in self.witnesses)
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError("static infeasibility witnesses must be unique and sorted")


def analyze_static_mapspace_feasibility(
    bundle: EvaluationBundle,
) -> TimeloopV2StaticInfeasibleEvaluation | None:
    """Prove empty primary-axis factor ranges before invoking native Timeloop.

    The v2 compiler closes every non-primary spatial rank to one.  Therefore a
    layer has a non-empty spatial factorization iff its primary-axis extent has
    at least one integer divisor inside the emitted inclusive bounds.  This is
    an exact workload check, not a heuristic feasibility proxy.
    """

    if type(bundle) is not EvaluationBundle:
        raise TypeError("bundle must be an exact EvaluationBundle")
    witnesses: list[TimeloopV2StaticMapspaceWitness] = []
    for manifest in bundle.layer_manifests:
        spatial = manifest.constraints[1]
        if type(spatial) is not RuntimeSpatialConstraint:
            raise TimeloopV2ContractError(
                "runtime manifest changed its spatial-constraint position"
            )
        primary_axis = spatial.permutation[0]
        axis_extent = manifest.problem_instance[primary_axis]
        admissible = tuple(
            factor
            for factor in range(
                spatial.minimum_parallelism,
                spatial.maximum_parallelism + 1,
            )
            if axis_extent % factor == 0
        )
        if not admissible:
            witnesses.append(
                TimeloopV2StaticMapspaceWitness(
                    medoid_ordinal=manifest.medoid_ordinal,
                    primary_axis=primary_axis,
                    axis_extent=axis_extent,
                    minimum_parallelism=spatial.minimum_parallelism,
                    maximum_parallelism=spatial.maximum_parallelism,
                    admissible_spatial_factors=(),
                )
            )
    if not witnesses:
        return None
    return TimeloopV2StaticInfeasibleEvaluation(
        candidate_sha256=bundle.candidate_sha256,
        compiled_plan_sha256=bundle.compiled_plan_sha256,
        panel_sha256=bundle.panel_sha256,
        evaluation_bundle_sha256=hashlib.sha256(
            canonical_evaluation_bundle_bytes(bundle)
        ).hexdigest(),
        witnesses=tuple(witnesses),
    )


class TimeloopV2CandidateInfeasibleError(RuntimeError):
    """The candidate has a proven empty mapspace or exhausted mapper budget."""

    def __init__(
        self,
        observation: (
            TimeloopV2InfeasibleEvaluation
            | TimeloopV2StaticInfeasibleEvaluation
        ),
    ) -> None:
        if type(observation) not in {
            TimeloopV2InfeasibleEvaluation,
            TimeloopV2StaticInfeasibleEvaluation,
        }:
            raise TypeError(
                "observation must be an exact Timeloop infeasibility record"
            )
        self.observation = observation
        if type(observation) is TimeloopV2StaticInfeasibleEvaluation:
            details = ", ".join(
                (
                    f"medoid={value.medoid_ordinal}:axis={value.primary_axis}:"
                    f"extent={value.axis_extent}:"
                    f"bounds={value.minimum_parallelism}-"
                    f"{value.maximum_parallelism}"
                )
                for value in observation.witnesses
            )
            super().__init__("candidate has a statically empty mapspace: " + details)
        else:
            details = ", ".join(
                (
                    f"medoid={layer.medoid_ordinal}:"
                    f"consecutive_invalid={layer.consecutive_invalid_mapping_count}"
                )
                for layer in observation.layer_results
                if not layer.mapping_budget_complete
            )
            super().__init__(
                "candidate did not complete the frozen valid-mapping budget: "
                + details
            )


@runtime_checkable
class TimeloopV2EvaluatorPort(Protocol):
    def evaluate(self, config: object) -> TimeloopV2Evaluation: ...


class TimeloopV2DockerEvaluator:
    """Evaluate one closed candidate over one frozen three-medoid panel."""

    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: TimeloopV2Settings, panel: NetworkLayerPanel) -> None:
        if type(settings) is not TimeloopV2Settings:
            raise TypeError("settings must be an exact TimeloopV2Settings")
        if type(panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        self.settings = settings
        self.panel = panel
        self._runner = (
            Path(__file__).with_name("container_runner.py").resolve(strict=True)
        )
        self._runner_sha256 = _sha256(self._runner)

    def preflight(self) -> dict[str, object]:
        docker = shutil.which("docker")
        if docker is None:
            raise TimeloopV2ContractError("docker executable is unavailable")
        if _sha256(self._runner) != self._runner_sha256:
            raise TimeloopV2ContractError("container runner changed after construction")
        completed = subprocess.run(
            [docker, "image", "inspect", "--format={{.Id}}", self.settings.image_ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        observed = completed.stdout.strip()
        if completed.returncode != 0 or observed != self.settings.expected_image_id:
            raise TimeloopV2ContractError(
                "pinned Timeloop image is unavailable or has the wrong identity"
            )
        return {
            "docker_executable": docker,
            "image_ref": self.settings.image_ref,
            "image_id": observed,
            "runner_sha256": self._runner_sha256,
            "search_size": self.settings.search_size,
            "mapper_threads": self.settings.mapper_threads,
            "max_consecutive_invalid_mappings": (
                self.settings.max_consecutive_invalid_mappings
            ),
            "cpu_set": self.settings.cpu_set,
        }

    def evaluate(self, config: object) -> TimeloopV2Evaluation:
        bundle = build_evaluation_bundle(config, self.panel)
        static_infeasibility = analyze_static_mapspace_feasibility(bundle)
        if static_infeasibility is not None:
            raise TimeloopV2CandidateInfeasibleError(static_infeasibility)
        call_started = time.perf_counter()
        with _EXTERNAL_LOCK:
            acquired = time.perf_counter()
            queue_wait_s = acquired - call_started
            preflight = self.preflight()
            call_dir = (
                self.settings.output_root.resolve() / f"timeloop-v2-{uuid.uuid4().hex}"
            )
            call_dir.mkdir(parents=True, exist_ok=False)
            cacti_scratch = call_dir / "cacti-scratch"
            cacti_scratch.mkdir()
            input_path = call_dir / "evaluation-bundle.json"
            _atomic_json_bytes(input_path, canonical_evaluation_bundle_bytes(bundle))
            command = [
                str(preflight["docker_executable"]),
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--pids-limit",
                "256",
                "--cpuset-cpus",
                self.settings.cpu_set,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=256m",
                "--env",
                "LD_LIBRARY_PATH=/usr/local/lib",
                "--env",
                "HOME=/tmp",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "PYTHONHASHSEED=0",
                "--volume",
                f"{self._runner.parent}:/adapter:ro",
                "--volume",
                f"{call_dir}:/exchange:rw",
                "--volume",
                (
                    f"{cacti_scratch}:"
                    "/usr/local/share/accelergy/estimation_plug_ins/"
                    "accelergy-cacti-plug-in/cacti_inputs_outputs:rw"
                ),
                "--entrypoint",
                "/usr/bin/python3",
                self.settings.image_ref,
                "/adapter/container_runner.py",
                "--input",
                "/exchange/evaluation-bundle.json",
                "--output-dir",
                "/exchange/timeloop-output",
                "--result",
                "/exchange/result.json",
                "--failure",
                "/exchange/failure.json",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(self.settings.timeout_s),
                )
            except subprocess.TimeoutExpired as error:
                raise TimeloopV2ContractError(
                    "Timeloop v2 container exceeded the fixed outer timeout"
                ) from error
            evaluator_elapsed_s = time.perf_counter() - acquired
            if completed.returncode != 0:
                failure_path = call_dir / "failure.json"
                failure = (
                    failure_path.read_text(encoding="utf-8")[:8_000]
                    if failure_path.is_file()
                    else ""
                )
                raise TimeloopV2ContractError(
                    "Timeloop v2 container failed: "
                    f"returncode={completed.returncode}, failure={failure!r}, "
                    f"stderr={completed.stderr[:2_000]!r}"
                )
            result_path = call_dir / "result.json"
            # First reject duplicate/nonstandard JSON with the strict loader,
            # then let Pydantic's JSON boundary map arrays to immutable tuples.
            _load_json(result_path)
            result = ContainerEvaluationResult.model_validate_json(
                result_path.read_bytes(),
                strict=True,
            )
            try:
                evaluation = self._validate_result(
                    bundle,
                    call_dir,
                    result,
                    evaluator_elapsed_s=evaluator_elapsed_s,
                    queue_wait_s=queue_wait_s,
                )
            except TimeloopV2CandidateInfeasibleError as error:
                observation = error.observation
                self._write_host_receipt(
                    call_dir=call_dir,
                    result_path=result_path,
                    preflight=preflight,
                    status="candidate_infeasible",
                    candidate_sha256=observation.candidate_sha256,
                    compiled_plan_sha256=observation.compiled_plan_sha256,
                    panel_sha256=observation.panel_sha256,
                    evaluator_elapsed_s=observation.evaluator_elapsed_s,
                    queue_wait_s=observation.queue_wait_s,
                    incomplete_medoid_ordinals=(
                        observation.incomplete_medoid_ordinals
                    ),
                )
                raise
            self._write_host_receipt(
                call_dir=call_dir,
                result_path=result_path,
                preflight=preflight,
                status="passed",
                candidate_sha256=evaluation.candidate_sha256,
                compiled_plan_sha256=evaluation.compiled_plan_sha256,
                panel_sha256=evaluation.panel_sha256,
                evaluator_elapsed_s=evaluation.evaluator_elapsed_s,
                queue_wait_s=evaluation.queue_wait_s,
                incomplete_medoid_ordinals=(),
            )
            return evaluation

    def _write_host_receipt(
        self,
        *,
        call_dir: Path,
        result_path: Path,
        preflight: dict[str, object],
        status: Literal["passed", "candidate_infeasible"],
        candidate_sha256: str,
        compiled_plan_sha256: str,
        panel_sha256: str,
        evaluator_elapsed_s: float,
        queue_wait_s: float,
        incomplete_medoid_ordinals: tuple[int, ...],
    ) -> None:
        _atomic_json_bytes(
            call_dir / "host_receipt.json",
            json.dumps(
                {
                    "schema_version": 2,
                    "evaluator_id": EVALUATOR_ID,
                    "status": status,
                    "candidate_sha256": candidate_sha256,
                    "compiled_plan_sha256": compiled_plan_sha256,
                    "panel_sha256": panel_sha256,
                    "result_sha256": _sha256(result_path),
                    "image_ref": self.settings.image_ref,
                    "image_id": preflight["image_id"],
                    "runner_sha256": self._runner_sha256,
                    "cpu_set": self.settings.cpu_set,
                    "search_size": self.settings.search_size,
                    "mapper_threads": self.settings.mapper_threads,
                    "max_consecutive_invalid_mappings": (
                        self.settings.max_consecutive_invalid_mappings
                    ),
                    "evaluator_elapsed_s": evaluator_elapsed_s,
                    "queue_wait_s": queue_wait_s,
                    "incomplete_medoid_ordinals": list(
                        incomplete_medoid_ordinals
                    ),
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
        )

    def _validate_result(
        self,
        bundle: EvaluationBundle,
        call_dir: Path,
        result: ContainerEvaluationResult,
        *,
        evaluator_elapsed_s: float,
        queue_wait_s: float,
    ) -> TimeloopV2Evaluation:
        if (
            result.candidate_sha256 != bundle.candidate_sha256
            or result.compiled_plan_sha256 != bundle.compiled_plan_sha256
            or result.panel_sha256 != bundle.panel_sha256
        ):
            raise TimeloopV2ContractError("container evaluated different provenance")
        if result.protocol != bundle.protocol:
            raise TimeloopV2ContractError("container mapper protocol drift")
        if result.provenance.runner_sha256 != self._runner_sha256:
            raise TimeloopV2ContractError("container runner digest drift")
        for ordinal, layer in enumerate(result.layers):
            if (
                layer.medoid_ordinal != ordinal
                or layer.layer_manifest_sha256 != bundle.layer_manifest_sha256[ordinal]
                or layer.layer_multiplicity
                != bundle.layer_manifests[ordinal].layer_multiplicity
                or layer.output_subdirectory != f"medoid-{ordinal}"
            ):
                raise TimeloopV2ContractError("layer result provenance drift")
            layer_dir = call_dir / "timeloop-output" / layer.output_subdirectory
            mapping_path = layer_dir / "timeloop-mapper.map.yaml"
            processed_path = layer_dir / "parsed-processed-input.yaml"
            if (
                not mapping_path.is_file()
                or _sha256(mapping_path) != layer.mapping_sha256
                or not processed_path.is_file()
                or _sha256(processed_path) != layer.processed_input_sha256
            ):
                raise TimeloopV2ContractError("layer evidence file digest drift")
        incomplete_medoid_ordinals = tuple(
            layer.medoid_ordinal
            for layer in result.layers
            if not layer.mapping_budget_complete
        )
        if incomplete_medoid_ordinals:
            raise TimeloopV2CandidateInfeasibleError(
                TimeloopV2InfeasibleEvaluation(
                    output_dir=call_dir,
                    candidate_sha256=result.candidate_sha256,
                    compiled_plan_sha256=result.compiled_plan_sha256,
                    panel_sha256=result.panel_sha256,
                    evaluator_elapsed_s=evaluator_elapsed_s,
                    queue_wait_s=queue_wait_s,
                    layer_results=result.layers,
                    manifest=result,
                    incomplete_medoid_ordinals=incomplete_medoid_ordinals,
                )
            )
        return TimeloopV2Evaluation(
            objective_values={
                name: float(getattr(result.objectives, name))
                for name in OBJECTIVE_NAMES
            },
            output_dir=call_dir,
            candidate_sha256=result.candidate_sha256,
            compiled_plan_sha256=result.compiled_plan_sha256,
            panel_sha256=result.panel_sha256,
            evaluator_elapsed_s=evaluator_elapsed_s,
            queue_wait_s=queue_wait_s,
            layer_results=result.layers,
            manifest=result,
        )


__all__ = [
    "OBJECTIVE_NAMES",
    "PINNED_IMAGE_ID",
    "PINNED_IMAGE_REF",
    "ContainerEvaluationResult",
    "EvaluationBundle",
    "LayerContainerResult",
    "MapperProtocol",
    "TimeloopV2CandidateInfeasibleError",
    "TimeloopV2ContractError",
    "TimeloopV2DockerEvaluator",
    "TimeloopV2Evaluation",
    "TimeloopV2EvaluatorPort",
    "TimeloopV2InfeasibleEvaluation",
    "TimeloopV2StaticInfeasibleEvaluation",
    "TimeloopV2StaticMapspaceWitness",
    "TimeloopV2Settings",
    "analyze_static_mapspace_feasibility",
    "build_evaluation_bundle",
    "canonical_evaluation_bundle_bytes",
]
