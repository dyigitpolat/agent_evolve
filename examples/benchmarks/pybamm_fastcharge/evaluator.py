"""Subprocess evaluator plumbing for the pybamm_fastcharge workload.

The agent_evolve environment never imports pybamm.  Every evaluation shells
out to the isolated PyBaMM domain virtualenv and speaks a strict JSON
contract with :mod:`.eval_runner`.  This module encodes the memo's mandatory
solver policy as typed admission rules:

- hard wall-clock timeout on the whole subprocess (the measured CasadiSolver
  tail motivates it; IDAKLU is tail-free at the frozen fidelities but the cap
  stays as the safety contract) -> informative ``solver-slow`` invalid
  outcome, raised as ``ValueError`` candidate feedback;
- termination-validity gate: IDAKLU convergence failures were measured to
  return silently early-terminated solutions, so a result is admitted only
  when the run reports ``ok``, completed all four experiment steps, and
  terminated on the final CV hold's C-rate cut-off event; anything else is a
  typed early-termination rejection, never an objective value;
- Li-plating feasibility floor (proxy margin >= 0.01 V per the memo's
  section-6 mapping) -> typed infeasible outcome with the measured margin.

Objectives (both minimized): ``charge_time_s`` and ``peak_temp_rise_K``.
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


OBJECTIVE_CHARGE_TIME = "charge_time_s"
OBJECTIVE_TEMP_RISE = "peak_temp_rise_K"
EVALUATOR_ID = "pybamm-fastcharge-dfn-idaklu-v1"
RUNNER_SCHEMA_VERSION = 1
VALID_TERMINATION_MARKER = "C-rate cut-off"
REQUIRED_STEPS_SOLVED = 4
_MAX_RESPONSE_BYTES = 1_000_000
_DOMAIN_ENVS_ROOT = Path(
    "/home/yigit/repos/research_stuff/domain_envs/pybamm"
)
_RUNNER_PATH = Path(__file__).resolve().parent / "eval_runner.py"
_EVALUATION_LOCK = threading.Lock()


class PybammEvaluationError(RuntimeError):
    """The PyBaMM subprocess contract failed or returned untrusted output."""


def _reject_constant(value: str) -> None:
    raise PybammEvaluationError(f"nonstandard JSON constant in response: {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PybammEvaluationError(f"duplicate response key: {key}")
        result[key] = value
    return result


def runner_sha256() -> str:
    return hashlib.sha256(_RUNNER_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PybammEvaluatorSettings:
    """Frozen evaluation admission facts for one protocol evaluator."""

    domain_python: Path
    mesh_factor: int = 2
    timeout_s: float = 120.0
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    initial_soc: float = 0.10
    plating_margin_min_v: float = 0.01
    external_concurrency: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.domain_python, Path):
            raise TypeError("domain_python must be a pathlib.Path")
        if type(self.mesh_factor) is not int or not 1 <= self.mesh_factor <= 3:
            raise ValueError(
                "mesh_factor must be 1..3 (IDAKLU conv-fails were measured "
                "at mesh factor 4 and above)"
            )
        for value, name in (
            (self.timeout_s, "timeout_s"),
            (self.rtol, "rtol"),
            (self.atol, "atol"),
            (self.initial_soc, "initial_soc"),
            (self.plating_margin_min_v, "plating_margin_min_v"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.initial_soc < 1.0:
            raise ValueError("initial_soc must lie strictly inside (0, 1)")
        if self.external_concurrency != 1:
            raise ValueError("pybamm_fastcharge external_concurrency is fixed at one")


@dataclass(frozen=True, slots=True)
class PybammProtocolEvaluation:
    objective_values: dict[str, float]
    result: dict[str, object]
    request_sha256: str
    policy: dict[str, object]
    adapter_elapsed_s: float


def admit_runner_result(
    result: dict[str, object], settings: PybammEvaluatorSettings
) -> dict[str, float]:
    """Apply the frozen validity policy; return objectives or raise typed errors.

    ``ValueError`` carries candidate-level feedback (solver failure, early
    termination, plating infeasibility); ``PybammEvaluationError`` marks
    contract violations that must abort the run.
    """

    if type(result) is not dict:
        raise PybammEvaluationError("runner result must be an object")
    status = result.get("status")
    if status == "solver_error":
        raise ValueError(
            "invalid protocol outcome: the DFN/IDAKLU solve failed "
            f"({str(result.get('error'))[:400]})"
        )
    if status != "ok":
        raise PybammEvaluationError(f"untrusted runner status: {status!r}")
    termination = result.get("termination")
    steps_solved = result.get("n_steps_solved")
    if (
        type(termination) is not str
        or VALID_TERMINATION_MARKER not in termination
        or steps_solved != REQUIRED_STEPS_SOLVED
    ):
        raise ValueError(
            "invalid protocol outcome: the solver terminated early without "
            "completing the protocol (termination="
            f"{str(termination)[:120]!r}, steps_solved={steps_solved!r}); "
            "early-terminated solutions are never scored"
        )
    plating = result.get("plating_proxy_min_V")
    if (
        not isinstance(plating, (int, float))
        or isinstance(plating, bool)
        or not math.isfinite(float(plating))
    ):
        raise PybammEvaluationError("runner omitted a finite plating proxy")
    if float(plating) < float(settings.plating_margin_min_v):
        raise ValueError(
            "infeasible protocol: Li-plating proxy margin "
            f"{float(plating):.6f} V is below the required "
            f"{float(settings.plating_margin_min_v):.3f} V floor"
        )
    values: dict[str, float] = {}
    for name in (OBJECTIVE_CHARGE_TIME, OBJECTIVE_TEMP_RISE):
        value = result.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise PybammEvaluationError(f"runner objective {name} is invalid")
        values[name] = float(value)
    return values


class PybammSubprocessEvaluator:
    """Invoke the PyBaMM domain venv through a bounded JSON subprocess contract."""

    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: PybammEvaluatorSettings) -> None:
        if type(settings) is not PybammEvaluatorSettings:
            raise TypeError("settings must be exact PybammEvaluatorSettings")
        from agent_evolve.agentic import (
            ChildProcessPolicy,
            ExplicitEnvironmentSubprocessBoundary,
        )

        self.settings = settings
        self._runner_sha256 = runner_sha256()
        self._boundary = ExplicitEnvironmentSubprocessBoundary(
            policy=ChildProcessPolicy(
                policy_id="pybamm_fastcharge_domain_venv_child",
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
            raise PybammEvaluationError(
                f"PyBaMM domain python is not executable: {python}"
            )
        observed_runner = runner_sha256()
        if observed_runner != self._runner_sha256:
            raise PybammEvaluationError("eval_runner changed after construction")
        return {
            "evaluator_id": EVALUATOR_ID,
            "domain_python": str(python),
            "runner_path": str(_RUNNER_PATH),
            "runner_sha256": observed_runner,
            "mesh_factor": self.settings.mesh_factor,
            "timeout_s": float(self.settings.timeout_s),
            "plating_margin_min_v": float(self.settings.plating_margin_min_v),
            "external_concurrency": self.settings.external_concurrency,
            "process_boundary": self._boundary.stable_record(),
        }

    def evaluate_protocol(
        self, protocol: dict[str, float]
    ) -> PybammProtocolEvaluation:
        started = time.perf_counter()
        with _EVALUATION_LOCK:
            self.preflight()
            request = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "protocol": protocol,
                "mesh_factor": self.settings.mesh_factor,
                "rtol": float(self.settings.rtol),
                "atol": float(self.settings.atol),
                "initial_soc": float(self.settings.initial_soc),
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
                        timeout_s=float(self.settings.timeout_s),
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError(
                        "invalid protocol outcome: solver-slow; the DFN solve "
                        "exceeded the hard "
                        f"{float(self.settings.timeout_s):.0f}s wall-clock cap"
                    ) from exc
            finally:
                if os.path.exists(request_path):
                    os.unlink(request_path)
            if completed.returncode != 0 or len(completed.stdout) > _MAX_RESPONSE_BYTES:
                raise PybammEvaluationError(
                    "PyBaMM runner failed: returncode="
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
                raise PybammEvaluationError(
                    f"PyBaMM runner returned invalid JSON: {completed.stdout[:800]!r}"
                ) from exc
            if (
                type(response) is not dict
                or response.get("schema_version") != RUNNER_SCHEMA_VERSION
                or response.get("evaluator_id") != EVALUATOR_ID
                or type(response.get("result")) is not dict
                or type(response.get("policy")) is not dict
                or response["policy"].get("solver") != "idaklu"
                or response["policy"].get("mesh_factor") != self.settings.mesh_factor
            ):
                raise PybammEvaluationError(
                    "PyBaMM runner response does not bind the requested policy"
                )
            objective_values = admit_runner_result(
                response["result"], self.settings
            )
            return PybammProtocolEvaluation(
                objective_values=objective_values,
                result=response["result"],
                request_sha256=request_sha256,
                policy=response["policy"],
                adapter_elapsed_s=time.perf_counter() - started,
            )


def default_domain_python() -> Path:
    return _DOMAIN_ENVS_ROOT / "venv" / "bin" / "python"


__all__ = [
    "EVALUATOR_ID",
    "OBJECTIVE_CHARGE_TIME",
    "OBJECTIVE_TEMP_RISE",
    "PybammEvaluationError",
    "PybammEvaluatorSettings",
    "PybammProtocolEvaluation",
    "PybammSubprocessEvaluator",
    "REQUIRED_STEPS_SOLVED",
    "VALID_TERMINATION_MARKER",
    "admit_runner_result",
    "default_domain_python",
    "runner_sha256",
]
