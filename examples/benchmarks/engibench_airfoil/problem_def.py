"""AgentEvolve adapter for a real three-point EngiBench Airfoil panel.

The candidate representation is an explicitly external Bernstein surface
parameterization.  It is shared by every optimizer but must not be described
as EngiBench's upstream FFD representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from agent_evolve import ObjectiveSpec


REPRESENTATION_ID = "external_bernstein_y_panel_v1"
EXPECTED_DATASET_SHA256 = "bf9aaf67632a9881cae82ccdbac24a693b1619397d4246342a4f477614ffca51"


class AirfoilPanelCandidate(BaseModel):
    """Twenty smooth shape coefficients and one angle per operating point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_id: Literal["external_bernstein_y_panel_v1"] = REPRESENTATION_ID
    upper_coefficients: list[float]
    lower_coefficients: list[float]
    alpha_deg: list[float]

    @field_validator("upper_coefficients", "lower_coefficients")
    @classmethod
    def _validate_coefficients(cls, value: list[float]) -> list[float]:
        if len(value) != 10:
            raise ValueError("surface coefficient vectors must contain exactly 10 values")
        if any(not math.isfinite(item) or not -0.025 <= item <= 0.025 for item in value):
            raise ValueError("surface coefficients must be finite and in [-0.025, 0.025]")
        return value

    @field_validator("alpha_deg")
    @classmethod
    def _validate_alphas(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("alpha_deg must contain exactly three operating-point angles")
        if any(not math.isfinite(item) or not 0.0 <= item <= 10.0 for item in value):
            raise ValueError("angles must be finite and in [0, 10] degrees")
        return value


def normalize_candidate(config: object) -> dict[str, Any]:
    if isinstance(config, AirfoilPanelCandidate):
        model = config
    elif isinstance(config, Mapping):
        model = AirfoilPanelCandidate.model_validate(dict(config))
    else:
        raise ValueError("Airfoil candidate must be an object/mapping")
    return model.model_dump(mode="json")


def candidate_sha256(config: object) -> str:
    canonical = json.dumps(normalize_candidate(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AirfoilPanelSettings:
    python_executable: Path
    evaluator_script: Path
    dataset_arrow: Path
    output_root: Path
    work_root: Path
    cpu_set: str = "8-15"
    mpi_cores: int = 8
    timeout_seconds: float = 180.0
    expected_dataset_sha256: str = EXPECTED_DATASET_SHA256

    @classmethod
    def local_default(cls) -> "AirfoilPanelSettings":
        workspace = Path(__file__).resolve().parents[4]
        cache = Path.home() / ".cache" / "agent_evolve_aaai2027"
        return cls(
            python_executable=cache / "engibench" / ".venv" / "bin" / "python",
            evaluator_script=(
                workspace
                / "papers"
                / "agent_evolve_aaai_2027"
                / "research_artifacts"
                / "scripts"
                / "airfoil_external_panel_v1.py"
            ),
            dataset_arrow=(
                cache
                / "huggingface"
                / "datasets"
                / "IDEALLab___airfoil_v0"
                / "default"
                / "0.0.0"
                / "8c97a4306cb7246deb3ec560f0c15d9d1eb66e48"
                / "airfoil_v0-train.arrow"
            ),
            output_root=(
                workspace
                / "papers"
                / "agent_evolve_aaai_2027"
                / "research_artifacts"
                / "experiment_logs"
                / "benchmark_q1"
                / "engibench_airfoil"
                / "external_panel_v1"
                / "agent_evolve"
            ),
            work_root=Path("/tmp") / "agent_evolve_airfoil_external_panel_v1",
        )


@dataclass(frozen=True)
class AirfoilPanelEvaluation:
    candidate_sha256: str
    objective_values: dict[str, float]
    wall_seconds: float
    record_path: Path
    record: dict[str, Any]


class AirfoilEvaluationError(RuntimeError):
    def __init__(self, message: str, *, candidate_invalid: bool = False) -> None:
        super().__init__(message)
        self.candidate_invalid = candidate_invalid


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AirfoilPanelEvaluator:
    """Subprocess port keeping EngiBench's heavy dependencies isolated."""

    def __init__(self, settings: AirfoilPanelSettings) -> None:
        self.settings = settings
        for label, path in (
            ("pinned Python", settings.python_executable),
            ("evaluator script", settings.evaluator_script),
            ("dataset Arrow file", settings.dataset_arrow),
        ):
            if not path.is_file():
                raise AirfoilEvaluationError(f"missing {label}: {path}")
        dataset_sha = _sha256_file(settings.dataset_arrow)
        if dataset_sha != settings.expected_dataset_sha256:
            raise AirfoilEvaluationError(
                "dataset SHA-256 mismatch: "
                f"expected {settings.expected_dataset_sha256}, got {dataset_sha}"
            )
        self._script_sha256 = _sha256_file(settings.evaluator_script)
        self._run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"-pid{os.getpid()}"
        )
        self._lock = threading.Lock()

    @property
    def run_directory(self) -> Path:
        """Exact evaluator-owned receipt directory for this invocation."""

        return self.settings.output_root / self._run_id

    def durable_receipt_paths(self) -> tuple[Path, ...]:
        """Return only direct JSON receipts owned by this evaluator instance."""

        directory = self.run_directory
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise AirfoilEvaluationError(
                "Airfoil evaluator run receipt path is not a directory"
            )
        return tuple(
            sorted(
                (
                    path.resolve(strict=True)
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix == ".json"
                ),
                key=lambda path: path.name,
            )
        )

    def evaluate(self, config: object) -> AirfoilPanelEvaluation:
        candidate = normalize_candidate(config)
        key = candidate_sha256(candidate)
        # The pinned image currently uses a fixed container name. Serialize the
        # boundary until a separately validated multi-container port exists.
        with self._lock:
            run_dir = self.run_directory
            run_dir.mkdir(parents=True, exist_ok=True)
            output = run_dir / f"{key}.json"
            work_dir = self.settings.work_root / self._run_id / key
            command = [
                str(self.settings.python_executable),
                str(self.settings.evaluator_script),
                "--mode",
                "evaluate",
                "--candidate-json",
                json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                "--dataset-arrow",
                str(self.settings.dataset_arrow),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--cpu-set",
                self.settings.cpu_set,
                "--mpi-cores",
                str(self.settings.mpi_cores),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(self.settings.evaluator_script.parent),
                    "HF_HOME": str(Path.home() / ".cache" / "agent_evolve_aaai2027" / "huggingface"),
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "MPLCONFIGDIR": "/tmp/agent_evolve_airfoil_mplconfig",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.timeout_seconds,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise AirfoilEvaluationError(
                    f"Airfoil evaluator exceeded {self.settings.timeout_seconds} seconds"
                ) from exc
            if not output.is_file():
                raise AirfoilEvaluationError(
                    "Airfoil evaluator produced no durable record; "
                    f"exit={completed.returncode}, stderr={completed.stderr[-1000:]!r}"
                )
            record = json.loads(output.read_text(encoding="utf-8"))
            if completed.returncode != 0:
                failure = record.get("failure", {})
                raise AirfoilEvaluationError(
                    f"Airfoil evaluator failed: {failure.get('type')}: {failure.get('message')}",
                    candidate_invalid=record.get("status") == "candidate_invalid",
                )
            if record.get("status") != "evaluated" or record.get("candidate_sha256") != key:
                raise AirfoilEvaluationError("Airfoil evaluator record identity/status mismatch")
            objectives = record.get("objectives")
            expected = {"mean_drag_coefficient", "max_lift_target_error"}
            if not isinstance(objectives, dict) or set(objectives) != expected:
                raise AirfoilEvaluationError("Airfoil evaluator returned an invalid objective mapping")
            normalized = {name: float(objectives[name]) for name in sorted(expected)}
            if any(not math.isfinite(value) for value in normalized.values()):
                raise AirfoilEvaluationError("Airfoil evaluator returned non-finite objectives")
            if record.get("evaluator_calls") != 3 or len(record.get("points", [])) != 3:
                raise AirfoilEvaluationError("Airfoil evaluator did not complete exactly three points")
            return AirfoilPanelEvaluation(
                candidate_sha256=key,
                objective_values=normalized,
                wall_seconds=float(record["wall_seconds"]),
                record_path=output,
                record=record,
            )


@runtime_checkable
class DetailedAirfoilEvaluator(Protocol):
    def evaluate(self, config: object) -> AirfoilPanelEvaluation: ...


class AirfoilPanelProblem:
    """Two-objective real-CFD problem consumed directly by AgentEvolve."""

    candidate_model = AirfoilPanelCandidate
    example_config = {
        "representation_id": REPRESENTATION_ID,
        "upper_coefficients": [0.0] * 10,
        "lower_coefficients": [0.0] * 10,
        "alpha_deg": [2.5, 2.5, 2.5],
    }
    constraints_description = (
        "Coefficients use the external degree-9 Bernstein y-displacement decoder, "
        "must lie in [-0.025,0.025], and decoded geometry must pass the frozen "
        "Airfoil adapter plus dataset area-ratio bounds. Angles lie in [0,10]."
    )

    def __init__(
        self,
        settings: AirfoilPanelSettings,
        *,
        evaluator: DetailedAirfoilEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self._evaluator = evaluator
        self._evaluator_lock = threading.Lock()

    @property
    def objectives(self):
        return [
            ObjectiveSpec("mean_drag_coefficient", "min"),
            ObjectiveSpec("max_lift_target_error", "min"),
        ]

    @property
    def evaluator(self) -> DetailedAirfoilEvaluator:
        if self._evaluator is None:
            with self._evaluator_lock:
                if self._evaluator is None:
                    self._evaluator = AirfoilPanelEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> AirfoilPanelEvaluation:
        self.validate(config)
        try:
            return self.evaluator.evaluate(config)
        except AirfoilEvaluationError as exc:
            if exc.candidate_invalid:
                raise ValueError(str(exc)) from exc
            raise

    def evaluate(self, config: object) -> dict[str, float]:
        return self.evaluate_detailed(config).objective_values

    def candidate_key(self, config: object) -> str:
        return candidate_sha256(config)

    def render_candidate(self, config: object) -> str:
        candidate = normalize_candidate(config)
        upper_peak = max(abs(item) for item in candidate["upper_coefficients"])
        lower_peak = max(abs(item) for item in candidate["lower_coefficients"])
        return (
            f"external Bernstein shape |upper|max={upper_peak:.5f}, "
            f"|lower|max={lower_peak:.5f}, alpha={candidate['alpha_deg']}"
        )

    def search_space_description(self) -> str:
        return (
            "Real EngiBench Airfoil three-operating-point RANS panel. One shared shape "
            "uses 10 upper and 10 lower degree-9 Bernstein y coefficients, each in "
            "[-0.025,0.025] chord, plus three point-specific angles in [0,10] degrees. "
            "The smooth decoder is an external method-neutral representation and is NOT "
            "the upstream EngiBench FFD. Minimize mean drag and maximum absolute lift-"
            "target error across the lower/central/higher Mach-Reynolds points."
        )


def create_default_problem() -> AirfoilPanelProblem:
    return AirfoilPanelProblem(AirfoilPanelSettings.local_default())


problem = create_default_problem()


__all__ = [
    "AirfoilPanelCandidate",
    "AirfoilPanelEvaluation",
    "AirfoilPanelProblem",
    "AirfoilPanelSettings",
    "AirfoilPanelEvaluator",
    "create_default_problem",
    "problem",
]
