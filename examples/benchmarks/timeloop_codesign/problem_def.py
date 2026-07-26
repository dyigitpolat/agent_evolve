"""AgentEvolve problem boundary for pinned Timeloop architecture co-design."""

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
from typing import Any, Protocol, runtime_checkable
import uuid

from agent_evolve.agentic import ObjectiveSpec

from .candidate import (
    DEFAULT_CANDIDATE,
    CandidateConfig,
    candidate_sha256,
    canonical_candidate_bytes,
    normalize_candidate,
)
from .container_runner import (
    ASSET_SHA256,
    EVALUATOR_ID,
    MAPPER_ALGORITHM,
    MAPPER_THREADS,
    OPTIMIZATION_METRICS,
    SEARCH_SIZE,
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


class TimeloopContractError(RuntimeError):
    """The simulator, image, or result violated the frozen adapter contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise TimeloopContractError(f"nonstandard JSON constant: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TimeloopContractError(f"duplicate result key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )
    if type(value) is not dict:
        raise TimeloopContractError("result must be an exact JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def _exact_keys(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise TimeloopContractError(f"{name} has missing or extra fields")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeloopContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise TimeloopContractError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True, slots=True)
class TimeloopSettings:
    output_root: Path
    cpu_set: str = "8"
    timeout_s: float = 120.0
    image_ref: str = PINNED_IMAGE_REF
    expected_image_id: str = PINNED_IMAGE_ID
    search_size: int = SEARCH_SIZE
    mapper_threads: int = MAPPER_THREADS
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
        if self.image_ref != PINNED_IMAGE_REF:
            raise ValueError("the developmental evaluator image is pinned")
        if self.expected_image_id != PINNED_IMAGE_ID:
            raise ValueError("the developmental evaluator image ID is pinned")
        if self.search_size != SEARCH_SIZE or self.mapper_threads != MAPPER_THREADS:
            raise ValueError("mapper search_size and thread count are frozen")
        if self.external_concurrency != 1:
            raise ValueError("the developmental adapter is serialized")


@dataclass(frozen=True, slots=True)
class TimeloopEvaluation:
    objective_values: dict[str, float]
    output_dir: Path
    candidate_sha256: str
    mapping_sha256: str
    evaluator_elapsed_s: float
    elapsed_inside_container_s: float
    queue_wait_s: float
    cycles: int
    computes: int
    manifest: dict[str, Any]


@runtime_checkable
class TimeloopEvaluatorPort(Protocol):
    def evaluate(self, config: object) -> TimeloopEvaluation: ...


class TimeloopDockerEvaluator:
    """Run one fixed-budget mapper transaction in an immutable local image."""

    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: TimeloopSettings) -> None:
        if type(settings) is not TimeloopSettings:
            raise TypeError("settings must be an exact TimeloopSettings")
        self.settings = settings
        self._runner = Path(__file__).with_name("container_runner.py").resolve(
            strict=True
        )
        self._runner_sha256 = _sha256(self._runner)

    def preflight(self) -> dict[str, object]:
        docker = shutil.which("docker")
        if docker is None:
            raise TimeloopContractError("docker executable is unavailable")
        if _sha256(self._runner) != self._runner_sha256:
            raise TimeloopContractError("container runner changed after construction")
        completed = subprocess.run(
            [docker, "image", "inspect", "--format={{.Id}}", self.settings.image_ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        observed = completed.stdout.strip()
        if completed.returncode != 0 or observed != self.settings.expected_image_id:
            raise TimeloopContractError(
                "pinned Timeloop image is unavailable or has the wrong identity"
            )
        return {
            "docker_executable": docker,
            "image_ref": self.settings.image_ref,
            "image_id": observed,
            "runner_sha256": self._runner_sha256,
            "search_size": self.settings.search_size,
            "mapper_threads": self.settings.mapper_threads,
            "cpu_set": self.settings.cpu_set,
        }

    def evaluate(self, config: object) -> TimeloopEvaluation:
        candidate = normalize_candidate(config)
        call_started = time.perf_counter()
        with _EXTERNAL_LOCK:
            acquired = time.perf_counter()
            queue_wait_s = acquired - call_started
            preflight = self.preflight()
            root = self.settings.output_root.resolve()
            call_dir = root / f"timeloop-{uuid.uuid4().hex}"
            call_dir.mkdir(parents=True, exist_ok=False)
            cacti_scratch = call_dir / "cacti-scratch"
            cacti_scratch.mkdir()
            input_path = call_dir / "candidate.json"
            temporary_input = call_dir / "candidate.json.tmp"
            temporary_input.write_bytes(canonical_candidate_bytes(candidate) + b"\n")
            os.replace(temporary_input, input_path)

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
                "/exchange/candidate.json",
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
                raise TimeloopContractError(
                    "Timeloop container exceeded the fixed outer timeout"
                ) from error
            evaluator_elapsed_s = time.perf_counter() - acquired
            if completed.returncode != 0:
                failure_path = call_dir / "failure.json"
                failure = (
                    failure_path.read_text(encoding="utf-8")[:4_000]
                    if failure_path.is_file()
                    else ""
                )
                raise TimeloopContractError(
                    "Timeloop container failed: "
                    f"returncode={completed.returncode}, failure={failure!r}, "
                    f"stderr={completed.stderr[:2_000]!r}"
                )
            result_path = call_dir / "result.json"
            manifest = _load_json(result_path)
            evaluation = self._validate_result(
                candidate,
                call_dir,
                manifest,
                evaluator_elapsed_s=evaluator_elapsed_s,
                queue_wait_s=queue_wait_s,
            )
            _atomic_json(
                call_dir / "host_receipt.json",
                {
                    "schema_version": 1,
                    "evaluator_id": EVALUATOR_ID,
                    "candidate_sha256": evaluation.candidate_sha256,
                    "result_sha256": _sha256(result_path),
                    "image_ref": self.settings.image_ref,
                    "image_id": preflight["image_id"],
                    "runner_sha256": self._runner_sha256,
                    "cpu_set": self.settings.cpu_set,
                    "search_size": self.settings.search_size,
                    "mapper_threads": self.settings.mapper_threads,
                    "evaluator_elapsed_s": evaluation.evaluator_elapsed_s,
                    "elapsed_inside_container_s": (
                        evaluation.elapsed_inside_container_s
                    ),
                    "queue_wait_s": evaluation.queue_wait_s,
                },
            )
            return evaluation

    def _validate_result(
        self,
        candidate: CandidateConfig,
        call_dir: Path,
        manifest: dict[str, Any],
        *,
        evaluator_elapsed_s: float,
        queue_wait_s: float,
    ) -> TimeloopEvaluation:
        top = _exact_keys(
            manifest,
            {
                "schema_version",
                "evaluator_id",
                "candidate",
                "candidate_sha256",
                "objectives",
                "diagnostics",
                "protocol",
                "provenance",
            },
            "result",
        )
        expected_payload = candidate.model_dump(mode="python")
        if top["schema_version"] != 1 or top["evaluator_id"] != EVALUATOR_ID:
            raise TimeloopContractError("result schema or evaluator identity drift")
        if top["candidate"] != expected_payload:
            raise TimeloopContractError("container evaluated a different candidate")
        expected_candidate_sha256 = candidate_sha256(candidate)
        if top["candidate_sha256"] != expected_candidate_sha256:
            raise TimeloopContractError("candidate digest mismatch")

        objectives = _exact_keys(
            top["objectives"], set(OBJECTIVE_NAMES), "objectives"
        )
        normalized_objectives = {
            name: _positive_float(objectives[name], name) for name in OBJECTIVE_NAMES
        }
        diagnostics = _exact_keys(
            top["diagnostics"],
            {"elapsed_s", "cycles", "computes", "mapping_sha256"},
            "diagnostics",
        )
        elapsed_inside = _positive_float(diagnostics["elapsed_s"], "elapsed_s")
        if type(diagnostics["cycles"]) is not int or diagnostics["cycles"] <= 0:
            raise TimeloopContractError("cycles must be a positive integer")
        if type(diagnostics["computes"]) is not int or diagnostics["computes"] <= 0:
            raise TimeloopContractError("computes must be a positive integer")
        mapping_sha256 = diagnostics["mapping_sha256"]
        if (
            type(mapping_sha256) is not str
            or len(mapping_sha256) != 64
            or any(character not in "0123456789abcdef" for character in mapping_sha256)
        ):
            raise TimeloopContractError("mapping_sha256 is invalid")

        protocol = _exact_keys(
            top["protocol"],
            {
                "search_size",
                "mapper_threads",
                "mapper_algorithm",
                "optimization_metrics",
            },
            "protocol",
        )
        expected_protocol = {
            "search_size": SEARCH_SIZE,
            "mapper_threads": MAPPER_THREADS,
            "mapper_algorithm": MAPPER_ALGORITHM,
            "optimization_metrics": list(OPTIMIZATION_METRICS),
        }
        if protocol != expected_protocol:
            raise TimeloopContractError("mapper protocol drift")
        provenance = _exact_keys(
            top["provenance"], {"asset_sha256", "runner_sha256"}, "provenance"
        )
        if provenance["asset_sha256"] != ASSET_SHA256:
            raise TimeloopContractError("asset provenance drift")
        if provenance["runner_sha256"] != self._runner_sha256:
            raise TimeloopContractError("container runner provenance drift")

        return TimeloopEvaluation(
            objective_values=normalized_objectives,
            output_dir=call_dir,
            candidate_sha256=expected_candidate_sha256,
            mapping_sha256=mapping_sha256,
            evaluator_elapsed_s=_positive_float(
                evaluator_elapsed_s, "evaluator_elapsed_s"
            ),
            elapsed_inside_container_s=elapsed_inside,
            queue_wait_s=max(0.0, float(queue_wait_s)),
            cycles=diagnostics["cycles"],
            computes=diagnostics["computes"],
            manifest=manifest,
        )


class TimeloopCoDesignProblem:
    """Three-objective accelerator hardware/mapping co-design problem."""

    candidate_model = CandidateConfig
    example_config = dict(DEFAULT_CANDIDATE)
    constraints_description = (
        "all fields are closed typed choices; the evaluator owns the Timeloop YAML, "
        "fixed EDP mapper, 2,000-valid-mapping budget, and one-thread CPU lease"
    )

    def __init__(
        self,
        settings: TimeloopSettings,
        *,
        evaluator: TimeloopEvaluatorPort | None = None,
    ) -> None:
        if type(settings) is not TimeloopSettings:
            raise TypeError("settings must be an exact TimeloopSettings")
        if evaluator is not None and not isinstance(evaluator, TimeloopEvaluatorPort):
            raise TypeError("evaluator must implement TimeloopEvaluatorPort")
        self.settings = settings
        self._evaluator = evaluator
        self._lock = threading.Lock()

    @property
    def objectives(self):
        return [ObjectiveSpec(name, "min") for name in OBJECTIVE_NAMES]

    @property
    def evaluator(self) -> TimeloopEvaluatorPort:
        if self._evaluator is None:
            with self._lock:
                if self._evaluator is None:
                    self._evaluator = TimeloopDockerEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> TimeloopEvaluation:
        self.validate(config)
        return self.evaluator.evaluate(config)

    def evaluate(self, config: object) -> dict[str, float]:
        return self.evaluate_detailed(config).objective_values

    def candidate_key(self, config: object) -> str:
        return candidate_sha256(config)

    def render_candidate(self, config: object) -> str:
        candidate = normalize_candidate(config)
        return (
            f"PE={candidate.pe_mesh_x}, "
            f"buffer={candidate.global_buffer_depth}x{candidate.global_buffer_width}, "
            f"datawidth={candidate.datawidth_bits}, "
            f"register={str(candidate.register_enabled).lower()}"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Timeloop/Accelergy DNN accelerator architecture and mapping "
            "co-optimization. Candidate fields choose PE parallelism, global-buffer "
            "depth and width, arithmetic precision, and local-register presence. "
            "Each architecture is evaluated by the same pinned random-pruned EDP "
            "mapper for exactly 2,000 valid mappings on a fixed convolution layer. "
            "Minimize energy, latency, and physical area."
        )


def default_settings() -> TimeloopSettings:
    repository_root = Path(__file__).resolve().parents[4]
    return TimeloopSettings(
        output_root=(
            repository_root
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
            / "experiment_logs"
            / "benchmark_q1"
            / "timeloop_codesign"
            / "thin_adapter_development"
        )
    )


def create_default_problem() -> TimeloopCoDesignProblem:
    return TimeloopCoDesignProblem(default_settings())


problem = create_default_problem()


__all__ = [
    "EVALUATOR_ID",
    "OBJECTIVE_NAMES",
    "PINNED_IMAGE_ID",
    "PINNED_IMAGE_REF",
    "TimeloopCoDesignProblem",
    "TimeloopContractError",
    "TimeloopDockerEvaluator",
    "TimeloopEvaluation",
    "TimeloopEvaluatorPort",
    "TimeloopSettings",
    "create_default_problem",
    "default_settings",
    "problem",
]
