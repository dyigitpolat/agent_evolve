"""Problem and serialized direct-v3 evaluator for constructive Heat2D."""

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
import sys
import threading
import time
from typing import Any, Protocol, runtime_checkable
import uuid

from agent_evolve.agentic import (
    ChildProcessPolicy,
    ExplicitEnvironmentSubprocessBoundary,
    ObjectiveSpec,
)

from .artifact_boundary import (
    artifact_scripts_dir,
    decode_mapping,
    decoder_source_sha256,
    direct_v3_runner_path,
    direct_v3_runner_sha256,
    save_design,
)
from .candidate import (
    CandidateConfig,
    SEED_LAYOUT_A,
    canonical_candidate_json,
    normalize_candidate,
)


OBJECTIVE_NAME = "engibench_objective"
EVALUATOR_ID = "engibench-heatconduction2d-direct-v3"
_EXTERNAL_EVALUATOR_LOCK = threading.Lock()
_CHILD_PATH = "/usr/bin:/bin"
_EXPECTED_DOCKER_EXECUTABLE = "/usr/bin/docker"
_PYTHON_PROBE_TIMEOUT_S = 15.0
_RUNNER_HELP_PROBE_TIMEOUT_S = 15.0
_PYTHON_PROBE_SOURCE = (
    "import hashlib\n"
    "import json\n"
    "from pathlib import Path\n"
    "import sys\n"
    "import numpy\n"
    "module_path = Path(numpy.__file__).resolve(strict=True)\n"
    "print(json.dumps("
    "{'numpy_version': numpy.__version__,"
    "'numpy_module_path': str(module_path),"
    "'numpy_module_sha256': hashlib.sha256(module_path.read_bytes()).hexdigest(),"
    "'python_prefix': sys.prefix, 'python_base_prefix': sys.base_prefix},"
    "sort_keys=True, separators=(',', ':')))\n"
)
_DEPENDENCY_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DirectV3ContractError(RuntimeError):
    """Raised when direct-v3 fails or returns an untrusted result."""


def _reject_json_constant(value: str) -> None:
    raise DirectV3ContractError(f"nonstandard JSON constant in manifest: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectV3ContractError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _invocation_path(path: Path) -> Path:
    """Return an absolute invocation path without dereferencing symlinks.

    Virtual-environment launchers are commonly symlinks whose path determines
    the active environment.  Resolving such a path before execution silently
    changes the process boundary to the base interpreter.
    """

    return Path(os.path.abspath(os.path.expanduser(path)))


@dataclass(frozen=True, slots=True)
class Heat2DDirectV3Settings:
    """Frozen developmental evaluator settings.

    External evaluator concurrency is intentionally fixed at one until PDE RSS
    and runtime qualification explicitly authorizes a wider CPU lease pool.
    """

    output_root: Path
    resolution: int = 1001
    width: float = 0.5
    volume_tolerance: float = 1.0e-10
    cpu_set: str = "8"
    timeout_s: float = 180.0
    python_executable: Path = Path(sys.executable)
    required_numpy_version: str = "2.3.5"
    external_concurrency: int = 1

    def __post_init__(self) -> None:
        if type(self.resolution) is not int or self.resolution < 3:
            raise ValueError("resolution must be an integer at least three")
        for value, name in (
            (self.width, "width"),
            (self.volume_tolerance, "volume_tolerance"),
            (self.timeout_s, "timeout_s"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a pathlib.Path")
        if not isinstance(self.python_executable, Path):
            raise TypeError("python_executable must be a pathlib.Path")
        if self.required_numpy_version != "2.3.5":
            raise ValueError("required_numpy_version is pinned to exactly 2.3.5")
        if type(self.cpu_set) is not str or not self.cpu_set.strip():
            raise ValueError("cpu_set must be a canonical non-empty string")
        if self.cpu_set != self.cpu_set.strip():
            raise ValueError("cpu_set must not contain surrounding whitespace")
        if self.external_concurrency != 1:
            raise ValueError(
                "developmental direct-v3 external_concurrency is fixed at one"
            )


@dataclass(frozen=True, slots=True)
class Heat2DDirectV3Evaluation:
    objective_values: dict[str, float]
    output_dir: Path
    genotype_sha256: str
    phenotype_sha256: str
    raw_array_sha256: str
    representation_spec_sha256: str
    finite_element_volume: float
    grayness: float
    gray_fraction_005_095: float
    adapter_elapsed_s: float
    evaluator_elapsed_s: float
    elapsed_inside_container_s: float
    queue_wait_s: float
    peak_rss_bytes: int
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PythonEnvironmentProbe:
    numpy_version: str
    numpy_module_path: str
    numpy_module_sha256: str
    python_prefix: str
    python_base_prefix: str


@runtime_checkable
class Heat2DEvaluator(Protocol):
    def evaluate(self, config: object) -> Heat2DDirectV3Evaluation: ...


class DirectV3Evaluator:
    """Invoke the strict direct-v3 host runner through unique directories."""

    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: Heat2DDirectV3Settings) -> None:
        if type(settings) is not Heat2DDirectV3Settings:
            raise TypeError("settings must be exact Heat2DDirectV3Settings")
        self.settings = settings
        self._python_executable = _invocation_path(settings.python_executable)
        self._python_probe_lock = threading.Lock()
        self._python_probe: _PythonEnvironmentProbe | None = None
        self._runner_help_probe_lock = threading.Lock()
        self._runner_help_stdout_sha256: str | None = None
        self._runner_path = direct_v3_runner_path().resolve(strict=True)
        self._runner_sha256 = direct_v3_runner_sha256()
        self._process_boundary = ExplicitEnvironmentSubprocessBoundary(
            policy=ChildProcessPolicy(
                policy_id="heat2d_direct_v3_host_child",
                policy_version=1,
                inherited_environment_allowlist=("HOME",),
                fixed_environment=(
                    ("LANG", "C.UTF-8"),
                    ("PATH", _CHILD_PATH),
                ),
            ),
            working_directory=self._runner_path.parent,
        )

    def _probe_python_environment(self) -> _PythonEnvironmentProbe:
        with self._python_probe_lock:
            if self._python_probe is not None:
                return self._python_probe
            try:
                completed = self._process_boundary.run(
                    (str(self._python_executable), "-c", _PYTHON_PROBE_SOURCE),
                    timeout_s=_PYTHON_PROBE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                raise DirectV3ContractError(
                    "Python environment dependency probe timed out"
                ) from None
            except OSError:
                raise DirectV3ContractError(
                    "Python environment dependency probe could not start"
                ) from None
            if completed.returncode != 0:
                raise DirectV3ContractError(
                    "Python environment dependency probe failed"
                )
            if len(completed.stdout) > 4096:
                raise DirectV3ContractError(
                    "Python environment dependency probe returned invalid output"
                )
            try:
                payload = json.loads(
                    completed.stdout,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_strict_json_object,
                )
            except (DirectV3ContractError, json.JSONDecodeError):
                raise DirectV3ContractError(
                    "Python environment dependency probe returned invalid output"
                ) from None
            if (
                not isinstance(payload, dict)
                or set(payload)
                != {
                    "numpy_version",
                    "numpy_module_path",
                    "numpy_module_sha256",
                    "python_prefix",
                    "python_base_prefix",
                }
                or type(payload.get("numpy_version")) is not str
                or _DEPENDENCY_VERSION_PATTERN.fullmatch(payload["numpy_version"])
                is None
                or payload["numpy_version"] != self.settings.required_numpy_version
                or type(payload.get("numpy_module_path")) is not str
                or not payload["numpy_module_path"]
                or type(payload.get("numpy_module_sha256")) is not str
                or _SHA256_PATTERN.fullmatch(payload["numpy_module_sha256"]) is None
                or type(payload.get("python_prefix")) is not str
                or not payload["python_prefix"]
                or type(payload.get("python_base_prefix")) is not str
                or not payload["python_base_prefix"]
            ):
                raise DirectV3ContractError(
                    "Python environment dependency probe returned invalid output"
                )
            self._python_probe = _PythonEnvironmentProbe(
                numpy_version=payload["numpy_version"],
                numpy_module_path=payload["numpy_module_path"],
                numpy_module_sha256=payload["numpy_module_sha256"],
                python_prefix=payload["python_prefix"],
                python_base_prefix=payload["python_base_prefix"],
            )
            return self._python_probe

    def _probe_runner_imports(self) -> str:
        with self._runner_help_probe_lock:
            if self._runner_help_stdout_sha256 is not None:
                return self._runner_help_stdout_sha256
            try:
                completed = self._process_boundary.run(
                    (
                        str(self._python_executable),
                        str(self._runner_path),
                        "--help",
                    ),
                    timeout_s=_RUNNER_HELP_PROBE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                raise DirectV3ContractError(
                    "direct-v3 transitive-import probe timed out"
                ) from None
            except OSError:
                raise DirectV3ContractError(
                    "direct-v3 transitive-import probe could not start"
                ) from None
            if completed.returncode != 0:
                raise DirectV3ContractError(
                    "direct-v3 transitive-import probe failed"
                )
            if "usage:" not in completed.stdout or self._runner_path.name not in (
                completed.stdout
            ):
                raise DirectV3ContractError(
                    "direct-v3 transitive-import probe returned invalid output"
                )
            self._runner_help_stdout_sha256 = hashlib.sha256(
                completed.stdout.encode("utf-8", errors="strict")
            ).hexdigest()
            return self._runner_help_stdout_sha256

    def invocation_observation(self) -> dict[str, object]:
        return self._process_boundary.invocation_observation(
            str(self._python_executable)
        )

    def preflight(self) -> dict[str, object]:
        try:
            resolved_python_executable = self._python_executable.resolve(strict=True)
        except OSError as error:
            raise DirectV3ContractError(
                f"Python executable is missing: {self._python_executable}"
            ) from error
        if not self._python_executable.is_file() or not os.access(
            self._python_executable, os.X_OK
        ):
            raise DirectV3ContractError(
                f"Python executable is not executable: {self._python_executable}"
            )
        docker = shutil.which("docker", path=_CHILD_PATH)
        if (
            docker != _EXPECTED_DOCKER_EXECUTABLE
            or not Path(docker).is_file()
            or not os.access(docker, os.X_OK)
        ):
            raise DirectV3ContractError("docker executable is unavailable")
        observed_runner_sha256 = direct_v3_runner_sha256()
        if observed_runner_sha256 != self._runner_sha256:
            raise DirectV3ContractError(
                "direct-v3 runner changed after evaluator construction"
            )
        dependency = self._probe_python_environment()
        runner_help_sha256 = self._probe_runner_imports()
        return {
            "docker_executable": docker,
            "python_executable_resolved": str(resolved_python_executable),
            "process_boundary": self._process_boundary.stable_record(),
            "required_dependencies": {
                "numpy": {
                    "required_version": self.settings.required_numpy_version,
                    "observed_version": dependency.numpy_version,
                    "module_file_sha256": dependency.numpy_module_sha256,
                }
            },
            "runner_help_transitive_import_probe": {
                "passed": True,
                "stdout_sha256": runner_help_sha256,
            },
            "runner_path": str(self._runner_path),
            "runner_sha256": observed_runner_sha256,
            "resolution": self.settings.resolution,
            "external_concurrency": self.settings.external_concurrency,
        }

    def evaluate(self, config: object) -> Heat2DDirectV3Evaluation:
        candidate = normalize_candidate(config)
        call_started = time.perf_counter_ns()
        with _EXTERNAL_EVALUATOR_LOCK:
            acquired = time.perf_counter_ns()
            queue_wait_s = (acquired - call_started) / 1.0e9
            self.preflight()
            decoded = decode_mapping(
                candidate.decoder_mapping(),
                resolution=self.settings.resolution,
            )
            call_id = uuid.uuid4().hex
            root = self.settings.output_root.resolve()
            input_path = root / "inputs" / f"candidate-{call_id}.npy"
            output_dir = root / "evaluations" / f"direct-v3-{call_id}"
            save_design(input_path, decoded)
            command = (
                str(self._python_executable),
                str(self._runner_path),
                "--design",
                str(input_path),
                "--resolution",
                str(self.settings.resolution),
                "--width",
                repr(float(self.settings.width)),
                "--expected-fe-volume",
                repr(float(candidate.material_fraction)),
                "--volume-tolerance",
                repr(float(self.settings.volume_tolerance)),
                "--mode",
                "forward",
                "--cpu-set",
                self.settings.cpu_set,
                "--timeout-s",
                repr(float(self.settings.timeout_s)),
                "--output-dir",
                str(output_dir),
            )
            try:
                completed = self._process_boundary.run(
                    command,
                    timeout_s=float(self.settings.timeout_s) + 45.0,
                )
            except subprocess.TimeoutExpired as error:
                raise DirectV3ContractError(
                    "direct-v3 host runner exceeded its bounded outer timeout"
                ) from error
            if completed.returncode != 0:
                failure_path = output_dir / "failure.json"
                failure = failure_path.read_text(encoding="utf-8") if failure_path.is_file() else ""
                raise DirectV3ContractError(
                    "direct-v3 host runner failed: "
                    f"returncode={completed.returncode}, failure={failure[:2000]!r}, "
                    f"stderr={completed.stderr[:1000]!r}"
                )

            manifest_path = output_dir / "manifest.json"
            if not manifest_path.is_file():
                raise DirectV3ContractError("direct-v3 did not publish manifest.json")
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_json_object,
            )
            if not isinstance(manifest, dict):
                raise DirectV3ContractError("direct-v3 manifest must be an object")
            container_result = manifest.get("container_result")
            candidate_record = manifest.get("candidate")
            if not isinstance(container_result, dict) or not isinstance(
                candidate_record, dict
            ):
                raise DirectV3ContractError("direct-v3 manifest payload is incomplete")
            result = container_result.get("result")
            measurement = container_result.get("resource_measurement")
            if not isinstance(result, dict) or not isinstance(measurement, dict):
                raise DirectV3ContractError("direct-v3 scientific result is incomplete")
            objective = result.get(OBJECTIVE_NAME)
            evaluator_elapsed_s = manifest.get("elapsed_s")
            inside_s = container_result.get("elapsed_inside_container_s")
            peak_rss_bytes = measurement.get(
                "peak_rss_bytes_by_linux_kib_convention"
            )
            numeric_values = {
                OBJECTIVE_NAME: objective,
                "evaluator_elapsed_s": evaluator_elapsed_s,
                "elapsed_inside_container_s": inside_s,
            }
            for name, value in numeric_values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise DirectV3ContractError(f"direct-v3 {name} is not numeric")
                if not math.isfinite(float(value)):
                    raise DirectV3ContractError(f"direct-v3 {name} is not finite")
            if type(peak_rss_bytes) is not int or peak_rss_bytes <= 0:
                raise DirectV3ContractError("direct-v3 peak RSS is invalid")
            checks = (
                manifest.get("schema_version") == 3,
                manifest.get("evaluator_id") == EVALUATOR_ID,
                manifest.get("mode") == "forward",
                manifest.get("all_checks_pass") is True,
                manifest.get("full_pde_solve_count") == 1,
                candidate_record.get("resolution") == self.settings.resolution,
                candidate_record.get("raw_array_sha256") == decoded.raw_array_sha256,
                abs(
                    float(candidate_record.get("expected_finite_element_volume"))
                    - candidate.material_fraction
                )
                <= self.settings.volume_tolerance,
            )
            if not all(checks):
                raise DirectV3ContractError(
                    "direct-v3 manifest does not bind the requested decoded candidate"
                )
            adapter_elapsed_s = (time.perf_counter_ns() - call_started) / 1.0e9
            return Heat2DDirectV3Evaluation(
                objective_values={OBJECTIVE_NAME: float(objective)},
                output_dir=output_dir,
                genotype_sha256=decoded.genotype_sha256,
                phenotype_sha256=decoded.phenotype_sha256,
                raw_array_sha256=decoded.raw_array_sha256,
                representation_spec_sha256=decoded.representation_spec_sha256,
                finite_element_volume=float(decoded.finite_element_volume),
                grayness=float(decoded.grayness),
                gray_fraction_005_095=float(decoded.gray_fraction_005_095),
                adapter_elapsed_s=adapter_elapsed_s,
                evaluator_elapsed_s=float(evaluator_elapsed_s),
                elapsed_inside_container_s=float(inside_s),
                queue_wait_s=queue_wait_s,
                peak_rss_bytes=peak_rss_bytes,
                manifest=manifest,
            )


class Heat2DConstructiveProblem:
    candidate_model = CandidateConfig
    example_config = SEED_LAYOUT_A
    constraints_description = (
        "fixed four-primitive CSG topology; all scalar fields are finite and "
        "strictly bounded; the decoder exactly projects material fraction; "
        "primitive identity, operation, kind, and order are engine-owned"
    )

    def __init__(
        self,
        settings: Heat2DDirectV3Settings,
        *,
        evaluator: Heat2DEvaluator | None = None,
    ) -> None:
        if type(settings) is not Heat2DDirectV3Settings:
            raise TypeError("settings must be exact Heat2DDirectV3Settings")
        if evaluator is not None and not isinstance(evaluator, Heat2DEvaluator):
            raise TypeError("evaluator must implement Heat2DEvaluator")
        self.settings = settings
        self._evaluator = evaluator
        self._evaluator_lock = threading.Lock()

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (ObjectiveSpec(OBJECTIVE_NAME, "min"),)

    @property
    def evaluator(self) -> Heat2DEvaluator:
        if self._evaluator is None:
            with self._evaluator_lock:
                if self._evaluator is None:
                    self._evaluator = DirectV3Evaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> Heat2DDirectV3Evaluation:
        self.validate(config)
        return self.evaluator.evaluate(config)

    def evaluate(self, config: object) -> dict[str, float]:
        return dict(self.evaluate_detailed(config).objective_values)

    @staticmethod
    def candidate_key(config: object) -> str:
        candidate_json = canonical_candidate_json(config).encode("ascii")
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:heat2d-constructive-candidate-key:v1\x00")
        digest.update(bytes.fromhex(decoder_source_sha256()))
        digest.update(candidate_json)
        return digest.hexdigest()

    @staticmethod
    def render_candidate(config: object) -> str:
        candidate = normalize_candidate(config)
        return (
            f"fixed CSG material_fraction={candidate.material_fraction:.3f}, "
            f"trunk_radius={candidate.trunk.radius:.3f}, "
            f"hole=({candidate.central_hole.radius_x:.3f},"
            f"{candidate.central_hole.radius_y:.3f})"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Constructive HeatConduction2D material co-optimization. A fixed "
            "engine-owned topology contains an additive capsule, ellipse, and "
            "rotated box plus one subtractive ellipse. Candidate fields tune "
            "bounded positions, sizes, angles, and material fraction. The exact "
            "shared decoder projects the dense CG1 nodal field to the requested "
            "material fraction. Minimize the pinned EngiBench thermal objective."
        )


def default_settings() -> Heat2DDirectV3Settings:
    research_artifacts = artifact_scripts_dir().parent
    return Heat2DDirectV3Settings(
        output_root=(
            research_artifacts
            / "experiment_logs"
            / "benchmark_q1"
            / "engibench_heat2d"
            / "agent_evolve_constructive_development"
        ),
    )


def create_default_problem() -> Heat2DConstructiveProblem:
    return Heat2DConstructiveProblem(default_settings())


problem = create_default_problem()


__all__ = [
    "DirectV3ContractError",
    "DirectV3Evaluator",
    "EVALUATOR_ID",
    "Heat2DConstructiveProblem",
    "Heat2DDirectV3Evaluation",
    "Heat2DDirectV3Settings",
    "Heat2DEvaluator",
    "OBJECTIVE_NAME",
    "create_default_problem",
    "default_settings",
    "problem",
]
