"""Subprocess evaluator plumbing for the scip_miplib workload.

The agent_evolve environment never imports pyscipopt.  Every evaluation shells
out to the isolated SCIP domain virtualenv python and speaks a strict JSON
contract with :mod:`.eval_runner` (mirroring how timeloop_codesign wraps its
container runner).  The frozen Gate-A policy lives in the runner; this module
owns admission (preflight), the bounded process boundary, response
authentication, and the bi-objective aggregation over the instance panel.

Objective definitions (both minimized), frozen here:

- ``panel_pd_integral_avg_pct``: arithmetic mean over the panel of SCIP's
  primal-dual integral average percent; per-instance fallback when the
  statistics line is unavailable is ``min(final_relative_gap * 100, 100)``
  and ``100`` when no gap exists.
- ``panel_gap_time_score``: arithmetic mean over the panel of the
  best-gap-or-time-to-best composite: ``solving_s / time_limit_s`` for runs
  solved to optimality, else ``1 + min(final_relative_gap, 10)`` (missing
  gap counts as the cap), so just-in-time solves and zero-gap timeouts meet
  continuously at 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time


OBJECTIVE_PD_INTEGRAL = "panel_pd_integral_avg_pct"
OBJECTIVE_GAP_TIME = "panel_gap_time_score"
EVALUATOR_ID = "scip-miplib-panel-config-v1"
RUNNER_SCHEMA_VERSION = 1
_GAP_CAP = 10.0
_ACCEPTED_STATUSES = ("optimal", "timelimit")
_MAX_RESPONSE_BYTES = 4_000_000
_DOMAIN_ENVS_ROOT = Path(
    "/home/yigit/repos/research_stuff/domain_envs/scip"
)
_RUNNER_PATH = Path(__file__).resolve().parent / "eval_runner.py"
_EVALUATION_LOCK = threading.Lock()


class ScipEvaluationError(RuntimeError):
    """The SCIP subprocess contract failed or returned untrusted output."""


def _reject_constant(value: str) -> None:
    raise ScipEvaluationError(f"nonstandard JSON constant in response: {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScipEvaluationError(f"duplicate response key: {key}")
        result[key] = value
    return result


def runner_sha256() -> str:
    return hashlib.sha256(_RUNNER_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ScipEvaluatorSettings:
    """Frozen evaluation admission facts for one panel evaluator."""

    domain_python: Path
    instances_root: Path
    panel: tuple[tuple[str, str], ...]  # (file name, sha256) pairs
    time_limit_s: float = 30.0
    random_seed_shift: int = 0
    startup_grace_s: float = 90.0
    external_concurrency: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.domain_python, Path) or not isinstance(
            self.instances_root, Path
        ):
            raise TypeError("domain_python and instances_root must be Paths")
        if type(self.panel) is not tuple or not self.panel:
            raise ValueError("panel must be a non-empty tuple")
        for item in self.panel:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or len(item[1]) != 64
            ):
                raise ValueError("panel entries must be (file, sha256) pairs")
        for value, name in (
            (self.time_limit_s, "time_limit_s"),
            (self.startup_grace_s, "startup_grace_s"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.random_seed_shift) is not int or self.random_seed_shift < 0:
            raise ValueError("random_seed_shift must be a non-negative integer")
        if self.external_concurrency != 1:
            raise ValueError("scip_miplib external_concurrency is fixed at one")

    @property
    def subprocess_timeout_s(self) -> float:
        return self.startup_grace_s + len(self.panel) * (
            float(self.time_limit_s) + 15.0
        )


@dataclass(frozen=True, slots=True)
class ScipPanelEvaluation:
    objective_values: dict[str, float]
    per_instance: tuple[dict[str, object], ...]
    request_sha256: str
    policy: dict[str, object]
    adapter_elapsed_s: float


def _aggregate(
    results: list[dict[str, object]], time_limit_s: float
) -> dict[str, float]:
    pd_terms: list[float] = []
    gap_time_terms: list[float] = []
    for record in results:
        status = record.get("status")
        if status not in _ACCEPTED_STATUSES:
            raise ScipEvaluationError(
                f"instance {record.get('instance')!r} returned untrusted "
                f"status {status!r}: {json.dumps(record, default=str)[:500]}"
            )
        gap = record.get("gap")
        gap_value = (
            float(gap)
            if isinstance(gap, (int, float)) and not isinstance(gap, bool)
            else None
        )
        pd_avg = record.get("pd_integral_avg_pct")
        if isinstance(pd_avg, (int, float)) and not isinstance(pd_avg, bool):
            pd_terms.append(float(pd_avg))
        elif gap_value is not None:
            pd_terms.append(min(gap_value * 100.0, 100.0))
        else:
            pd_terms.append(100.0)
        if status == "optimal":
            solving = record.get("solving_s")
            if not isinstance(solving, (int, float)) or isinstance(solving, bool):
                raise ScipEvaluationError("optimal run without solving_s")
            gap_time_terms.append(float(solving) / float(time_limit_s))
        else:
            gap_time_terms.append(
                1.0 + (min(gap_value, _GAP_CAP) if gap_value is not None else _GAP_CAP)
            )
    values = {
        OBJECTIVE_PD_INTEGRAL: sum(pd_terms) / len(pd_terms),
        OBJECTIVE_GAP_TIME: sum(gap_time_terms) / len(gap_time_terms),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ScipEvaluationError(f"aggregated objective {name} is not finite")
    return values


class ScipSubprocessEvaluator:
    """Invoke the SCIP domain venv through a bounded JSON subprocess contract."""

    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: ScipEvaluatorSettings) -> None:
        if type(settings) is not ScipEvaluatorSettings:
            raise TypeError("settings must be exact ScipEvaluatorSettings")
        from agent_evolve.agentic import (
            ChildProcessPolicy,
            ExplicitEnvironmentSubprocessBoundary,
        )

        self.settings = settings
        self._runner_sha256 = runner_sha256()
        self._boundary = ExplicitEnvironmentSubprocessBoundary(
            policy=ChildProcessPolicy(
                policy_id="scip_miplib_domain_venv_child",
                policy_version=1,
                inherited_environment_allowlist=("HOME",),
                fixed_environment=(
                    ("LANG", "C.UTF-8"),
                    ("OMP_NUM_THREADS", "1"),
                    ("PATH", "/usr/bin:/bin"),
                ),
            ),
            working_directory=_RUNNER_PATH.parent,
        )

    def preflight(self) -> dict[str, object]:
        python = self.settings.domain_python
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ScipEvaluationError(
                f"SCIP domain python is not executable: {python}"
            )
        observed_runner = runner_sha256()
        if observed_runner != self._runner_sha256:
            raise ScipEvaluationError("eval_runner changed after construction")
        instance_records = []
        for file_name, expected_sha in self.settings.panel:
            path = self.settings.instances_root / file_name
            if not path.is_file():
                raise ScipEvaluationError(f"missing SCIP instance: {path}")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected_sha:
                raise ScipEvaluationError(
                    f"instance {file_name} content drifted from the frozen split"
                )
            instance_records.append({"file": file_name, "sha256": observed})
        return {
            "evaluator_id": EVALUATOR_ID,
            "domain_python": str(python),
            "runner_path": str(_RUNNER_PATH),
            "runner_sha256": observed_runner,
            "instances": instance_records,
            "time_limit_s": float(self.settings.time_limit_s),
            "random_seed_shift": self.settings.random_seed_shift,
            "external_concurrency": self.settings.external_concurrency,
            "process_boundary": self._boundary.stable_record(),
        }

    def evaluate_panel(
        self,
        *,
        setparams: dict[str, object],
        emphasis: dict[str, str],
    ) -> ScipPanelEvaluation:
        started = time.perf_counter()
        with _EVALUATION_LOCK:
            self.preflight()
            request = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "instances": [
                    str(self.settings.instances_root / file_name)
                    for file_name, _ in self.settings.panel
                ],
                "setparams": setparams,
                "emphasis": emphasis,
                "time_limit_s": float(self.settings.time_limit_s),
                "random_seed_shift": self.settings.random_seed_shift,
            }
            request_bytes = json.dumps(
                request, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            request_sha256 = hashlib.sha256(request_bytes).hexdigest()
            request_fd, request_path = tempfile.mkstemp(suffix=".json")
            try:
                with os.fdopen(request_fd, "wb") as handle:
                    handle.write(request_bytes)
                try:
                    completed = self._boundary.run(
                        (
                            str(self.settings.domain_python),
                            str(_RUNNER_PATH),
                            "--request",
                            request_path,
                        ),
                        timeout_s=self.settings.subprocess_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ScipEvaluationError(
                        "SCIP panel subprocess exceeded its bounded timeout of "
                        f"{self.settings.subprocess_timeout_s:.0f}s"
                    ) from exc
            finally:
                if os.path.exists(request_path):
                    os.unlink(request_path)
            if completed.returncode != 0 or len(completed.stdout) > _MAX_RESPONSE_BYTES:
                raise ScipEvaluationError(
                    "SCIP runner failed: returncode="
                    f"{completed.returncode}, stdout={completed.stdout[:1500]!r}, "
                    f"stderr={completed.stderr[:800]!r}"
                )
            try:
                response = json.loads(
                    completed.stdout,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_strict_object,
                )
            except json.JSONDecodeError as exc:
                raise ScipEvaluationError(
                    f"SCIP runner returned invalid JSON: {completed.stdout[:800]!r}"
                ) from exc
            if (
                type(response) is not dict
                or response.get("schema_version") != RUNNER_SCHEMA_VERSION
                or response.get("evaluator_id") != EVALUATOR_ID
                or type(response.get("results")) is not list
                or len(response["results"]) != len(self.settings.panel)
                or type(response.get("policy")) is not dict
            ):
                raise ScipEvaluationError(
                    "SCIP runner response does not bind the requested panel"
                )
            objective_values = _aggregate(
                response["results"], float(self.settings.time_limit_s)
            )
            return ScipPanelEvaluation(
                objective_values=objective_values,
                per_instance=tuple(response["results"]),
                request_sha256=request_sha256,
                policy=response["policy"],
                adapter_elapsed_s=time.perf_counter() - started,
            )


def default_domain_python() -> Path:
    return _DOMAIN_ENVS_ROOT / "venv" / "bin" / "python"


def default_instances_root() -> Path:
    return _DOMAIN_ENVS_ROOT / "instances"


__all__ = [
    "EVALUATOR_ID",
    "OBJECTIVE_GAP_TIME",
    "OBJECTIVE_PD_INTEGRAL",
    "ScipEvaluationError",
    "ScipEvaluatorSettings",
    "ScipPanelEvaluation",
    "ScipSubprocessEvaluator",
    "default_domain_python",
    "default_instances_root",
    "runner_sha256",
]
