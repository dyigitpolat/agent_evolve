"""AgentEvolve problem boundary for pinned OpenROAD-flow-scripts flow tuning.

One evaluation is one complete RTL-to-GDS run of OpenROAD-flow-scripts inside a
pinned, network-isolated container: Yosys synthesis, floorplan, global and
detailed placement, CTS, global and detailed routing, fill, and final metric
extraction.  Objectives are read from the flow's own machine-readable metric
logs, never from parsed console text.

Measured on this host (nangate45/gcd, 2 CPUs per run, 4 concurrent runs):
median 24.9 s per completed evaluation.  See the venue standup memo at
``papers/agent_evolve_aaai_2027/research_artifacts/wave_j_venues/eda_standup.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    MAKE_VARIABLES,
    CandidateConfig,
    candidate_sha256,
    canonical_candidate_bytes,
    normalize_candidate,
)


PINNED_IMAGE_REF = (
    "openroad/orfs@"
    "sha256:ebc8142da6d65d1a1e9a528aa2cedcde356243465dd859af8d3ade51075f8cb2"
)
EVALUATOR_ID = "orfs-flow-tuning-v1"

#: All four objectives are minimised.  ``achieved_period_ns`` rather than worst
#: slack, because the clock period is itself a knob: slack is not comparable
#: across candidates, achieved critical-path period is.
OBJECTIVE_NAMES = (
    "die_area_um2",
    "achieved_period_ns",
    "power_w",
    "route_wirelength_um",
)

_CPU_SET = re.compile(r"^[0-9]+(-[0-9]+)?$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ORFSContractError(RuntimeError):
    """The image, flow, or result violated the frozen adapter contract."""


class ORFSInfeasible(RuntimeError):
    """The flow refused this configuration (e.g. place density above 1.0).

    This is a property of the search space, not a defect: on the measured
    nangate45/gcd pilot a nonzero fraction of uniform draws are rejected by
    OpenROAD before placement.  The loop must treat these as infeasible
    candidates, never as evaluator faults.
    """


@dataclass(frozen=True, slots=True)
class ORFSSettings:
    output_root: Path
    design: str = "gcd"
    platform: str = "nangate45"
    cpu_set: str = "96-97"
    nproc: int = 2
    timeout_s: float = 1200.0
    image_ref: str = PINNED_IMAGE_REF
    keep_run_dir: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a pathlib.Path")
        if _CPU_SET.fullmatch(self.cpu_set) is None:
            raise ValueError("cpu_set must be a single CPU or an inclusive range")
        for token, name in ((self.design, "design"), (self.platform, "platform")):
            if _SAFE_TOKEN.fullmatch(token) is None:
                raise ValueError(f"{name} must be a bare identifier")
        if self.image_ref != PINNED_IMAGE_REF:
            raise ValueError("the evaluator image is pinned by digest")
        if type(self.nproc) is not int or self.nproc <= 0:
            raise ValueError("nproc must be a positive integer")
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class ORFSEvaluation:
    objective_values: dict[str, float]
    candidate_sha256: str
    evaluator_elapsed_s: float
    drc_errors: int
    worst_setup_slack_ns: float
    total_negative_slack_ns: float
    instance_count: int
    metrics: dict[str, Any]


@runtime_checkable
class ORFSEvaluatorPort(Protocol):
    def evaluate(self, config: object) -> ORFSEvaluation: ...


def _render_overrides(candidate: CandidateConfig) -> list[str]:
    payload = candidate.model_dump(mode="python")
    return [f"{var}={payload[field]}" for field, var in MAKE_VARIABLES.items()]


class ORFSDockerEvaluator:
    """Run one complete ORFS flow in the pinned image and read its metric logs."""

    evaluator_id = EVALUATOR_ID

    def __init__(self, settings: ORFSSettings) -> None:
        if type(settings) is not ORFSSettings:
            raise TypeError("settings must be an exact ORFSSettings")
        self.settings = settings

    def preflight(self) -> dict[str, object]:
        docker = shutil.which("docker")
        if docker is None:
            raise ORFSContractError("docker executable is unavailable")
        completed = subprocess.run(
            [docker, "image", "inspect", "--format={{.Id}}", self.settings.image_ref],
            check=False, capture_output=True, text=True, timeout=30.0,
        )
        if completed.returncode != 0:
            raise ORFSContractError("pinned ORFS image is unavailable")
        return {"docker_executable": docker, "image_id": completed.stdout.strip()}

    def evaluate(self, config: object) -> ORFSEvaluation:
        candidate = normalize_candidate(config)
        preflight = self.preflight()
        root = self.settings.output_root.resolve()
        call_dir = root / f"orfs-{uuid.uuid4().hex}"
        call_dir.mkdir(parents=True, exist_ok=False)
        (call_dir / "candidate.json").write_bytes(
            canonical_candidate_bytes(candidate) + b"\n"
        )
        design = self.settings.design
        platform = self.settings.platform
        period = candidate.sdc_clk_period_ns
        overrides = " ".join(_render_overrides(candidate))
        script = (
            "source ./env.sh >/dev/null 2>&1; cd flow; "
            f"sed 's|^set clk_period .*|set clk_period {period}|' "
            f"designs/{platform}/{design}/constraint.sdc > /work/constraint.sdc; "
            f"make DESIGN_CONFIG=./designs/{platform}/{design}/config.mk "
            f"WORK_HOME=/work NPROC={self.settings.nproc} "
            f"SDC_FILE=/work/constraint.sdc {overrides} > /work/flow.log 2>&1; "
            "echo RC=$?"
        )
        command = [
            str(preflight["docker_executable"]), "run", "--rm",
            "--network", "none",
            "--cpuset-cpus", self.settings.cpu_set,
            "--volume", f"{call_dir}:/work:rw",
            self.settings.image_ref, "bash", "-lc", script,
        ]
        started = time.perf_counter()
        try:
            subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=float(self.settings.timeout_s),
            )
        except subprocess.TimeoutExpired as error:
            raise ORFSContractError("ORFS flow exceeded the outer timeout") from error
        elapsed = time.perf_counter() - started

        logs = call_dir / "logs" / platform / design / "base"
        finish = logs / "6_report.json"
        route = logs / "5_2_route.json"
        if not finish.is_file() or not route.is_file():
            tail = ""
            log = call_dir / "flow.log"
            if log.is_file():
                tail = log.read_text(errors="replace")[-2000:]
            if not self.settings.keep_run_dir:
                shutil.rmtree(call_dir, ignore_errors=True)
            raise ORFSInfeasible(f"ORFS flow did not reach 6_report: {tail!r}")

        fin = json.loads(finish.read_text())
        rt = json.loads(route.read_text())
        worst_slack = float(fin["finish__timing__setup__ws"])
        objectives = {
            "die_area_um2": float(fin["finish__design__die__area"]),
            "achieved_period_ns": period - worst_slack,
            "power_w": float(fin["finish__power__total"]),
            "route_wirelength_um": float(rt["detailedroute__route__wirelength"]),
        }
        for name, value in objectives.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ORFSContractError(f"{name} is not finite and positive")
        evaluation = ORFSEvaluation(
            objective_values=objectives,
            candidate_sha256=candidate_sha256(candidate),
            evaluator_elapsed_s=elapsed,
            drc_errors=int(rt["detailedroute__route__drc_errors"]),
            worst_setup_slack_ns=worst_slack,
            total_negative_slack_ns=float(fin["finish__timing__setup__tns"]),
            instance_count=int(fin["finish__design__instance__count"]),
            metrics={"finish": fin, "route": rt},
        )
        if not self.settings.keep_run_dir:
            shutil.rmtree(call_dir, ignore_errors=True)
        return evaluation


class ORFSFlowTuningProblem:
    """Four-objective OpenROAD physical-design flow-parameter tuning problem."""

    candidate_model = CandidateConfig
    example_config = dict(DEFAULT_CANDIDATE)
    constraints_description = (
        "every field is a typed bounded scalar from the published OpenROAD "
        "AutoTuner search space; the evaluator owns the Makefile, the platform, "
        "the PDK and the SDC template, and rejects configurations OpenROAD "
        "itself refuses (e.g. effective place density above 1.0)"
    )

    def __init__(
        self,
        settings: ORFSSettings,
        *,
        evaluator: ORFSEvaluatorPort | None = None,
    ) -> None:
        if type(settings) is not ORFSSettings:
            raise TypeError("settings must be an exact ORFSSettings")
        if evaluator is not None and not isinstance(evaluator, ORFSEvaluatorPort):
            raise TypeError("evaluator must implement ORFSEvaluatorPort")
        self.settings = settings
        self._evaluator = evaluator
        self._lock = threading.Lock()

    @property
    def objectives(self):
        return [ObjectiveSpec(name, "min") for name in OBJECTIVE_NAMES]

    @property
    def evaluator(self) -> ORFSEvaluatorPort:
        if self._evaluator is None:
            with self._lock:
                if self._evaluator is None:
                    self._evaluator = ORFSDockerEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> ORFSEvaluation:
        self.validate(config)
        return self.evaluator.evaluate(config)

    def evaluate(self, config: object) -> dict[str, float]:
        return self.evaluate_detailed(config).objective_values

    def candidate_key(self, config: object) -> str:
        return candidate_sha256(config)

    def render_candidate(self, config: object) -> str:
        c = normalize_candidate(config)
        return (
            f"clk={c.sdc_clk_period_ns}ns, util={c.core_utilization}%, "
            f"ar={c.core_aspect_ratio}, margin={c.core_margin}, "
            f"pad_gp={c.cell_pad_global_placement}, "
            f"pad_dp={c.cell_pad_detail_placement}, "
            f"lb_addon={c.place_density_lb_addon}, "
            f"cts={c.cts_cluster_size}/{c.cts_cluster_diameter}"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "OpenROAD-flow-scripts physical-design flow tuning. Each candidate "
            "sets the clock constraint, floorplan utilisation, aspect ratio and "
            "margin, global/detail cell padding, the placement density lower "
            "bound addon, and the CTS clustering parameters. The evaluator runs "
            "the complete RTL-to-GDS flow (Yosys, floorplan, global and detail "
            "placement, CTS, global and detail route, fill) in a pinned "
            "container and reads OpenROAD's own metric logs. Minimise die area, "
            "achieved critical-path period, total power, and routed wirelength."
        )


def default_settings() -> ORFSSettings:
    repository_root = Path(__file__).resolve().parents[4]
    return ORFSSettings(
        output_root=(
            repository_root / "papers" / "agent_evolve_aaai_2027"
            / "research_artifacts" / "experiment_logs" / "benchmark_q1"
            / "orfs_flow_tuning" / "thin_adapter_development"
        )
    )


def create_default_problem() -> ORFSFlowTuningProblem:
    return ORFSFlowTuningProblem(default_settings())


__all__ = [
    "EVALUATOR_ID",
    "OBJECTIVE_NAMES",
    "PINNED_IMAGE_REF",
    "ORFSContractError",
    "ORFSDockerEvaluator",
    "ORFSEvaluation",
    "ORFSEvaluatorPort",
    "ORFSFlowTuningProblem",
    "ORFSInfeasible",
    "ORFSSettings",
    "create_default_problem",
    "default_settings",
]
