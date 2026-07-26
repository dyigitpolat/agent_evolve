"""Public AgentEvolve problem boundary for Timeloop v2."""

from __future__ import annotations

from pathlib import Path
import threading

from agent_evolve.agentic import ObjectiveSpec

from .candidate import (
    DEFAULT_CANDIDATE,
    CandidateConfig,
    candidate_sha256,
    normalize_candidate,
)
from .container_runner import SEARCH_SIZE
from .evaluator import (
    OBJECTIVE_NAMES,
    TimeloopV2DockerEvaluator,
    TimeloopV2Evaluation,
    TimeloopV2EvaluatorPort,
    TimeloopV2Settings,
)
from .frozen_panels import frozen_network_panel
from .network_panel import NetworkLayerPanel


class TimeloopV2CoDesignProblem:
    """Three-objective architecture/mapspace-policy co-design problem."""

    candidate_model = CandidateConfig
    example_config = dict(DEFAULT_CANDIDATE)
    constraints_description = (
        "all architecture and three medoid-policy blocks are closed typed choices; "
        "the benchmark compiler owns Timeloop constraints; each medoid uses the "
        "same pinned random-pruned EDP mapper budget; no candidate YAML or command "
        "crosses the evaluator boundary"
    )

    def __init__(
        self,
        settings: TimeloopV2Settings,
        panel: NetworkLayerPanel,
        *,
        evaluator: TimeloopV2EvaluatorPort | None = None,
    ) -> None:
        if type(settings) is not TimeloopV2Settings:
            raise TypeError("settings must be an exact TimeloopV2Settings")
        if type(panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        if evaluator is not None and not isinstance(evaluator, TimeloopV2EvaluatorPort):
            raise TypeError("evaluator must implement TimeloopV2EvaluatorPort")
        self.settings = settings
        self.panel = panel
        self._evaluator = evaluator
        self._lock = threading.Lock()

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        return [ObjectiveSpec(name, "min") for name in OBJECTIVE_NAMES]

    @property
    def evaluator(self) -> TimeloopV2EvaluatorPort:
        if self._evaluator is None:
            with self._lock:
                if self._evaluator is None:
                    self._evaluator = TimeloopV2DockerEvaluator(
                        self.settings,
                        self.panel,
                    )
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> TimeloopV2Evaluation:
        self.validate(config)
        return self.evaluator.evaluate(config)

    def evaluate(self, config: object) -> dict[str, float]:
        return self.evaluate_detailed(config).objective_values

    def candidate_key(self, config: object) -> str:
        return candidate_sha256(config)

    def render_candidate(self, config: object) -> str:
        candidate = normalize_candidate(config)
        policies = tuple(
            getattr(candidate, f"policy_cluster_{ordinal}") for ordinal in range(3)
        )
        policy_text = "; ".join(
            (
                f"m{ordinal}={policy.dataflow_family}/"
                f"{policy.primary_spatial_axis}/"
                f"{policy.spatial_utilization}/"
                f"{policy.buffer_residency_bias}/"
                f"{policy.outer_loop_order}"
            )
            for ordinal, policy in enumerate(policies)
        )
        return (
            f"PE={candidate.pe_mesh_x}, "
            f"buffer={candidate.global_buffer_depth}x{candidate.global_buffer_width}, "
            f"datawidth={candidate.datawidth_bits}, "
            f"register={str(candidate.register_enabled).lower()}; {policy_text}"
        )

    def search_space_description(self) -> str:
        return (
            "Timeloop/Accelergy network-level accelerator architecture and "
            "mapspace-policy co-optimization. Five closed architecture choices "
            "are shared across three outcome-blind Conv medoids. Each medoid has "
            "a closed policy block controlling dataflow, primary spatial axis, "
            "spatial utilization, global-buffer residency, and outer loop order. "
            f"The exact compiled space has 695,784,701,952 records. Each medoid "
            f"is evaluated for {SEARCH_SIZE} valid mappings by the pinned serial "
            "random-pruned EDP mapper; energy and latency are multiplicity-weighted "
            "and chip area is counted once. Minimize energy, latency, and area. "
            f"Frozen instance: {self.panel.network_id} ({self.panel.role})."
        )


def default_settings() -> TimeloopV2Settings:
    repository_root = Path(__file__).resolve().parents[5]
    return TimeloopV2Settings(
        output_root=(
            repository_root
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
            / "experiment_logs"
            / "benchmark_q1"
            / "timeloop_codesign"
            / "v2_agentic_campaign"
        )
    )


def create_default_problem() -> TimeloopV2CoDesignProblem:
    return TimeloopV2CoDesignProblem(
        default_settings(),
        frozen_network_panel("resnet50"),
    )


problem = create_default_problem()


__all__ = [
    "TimeloopV2CoDesignProblem",
    "create_default_problem",
    "default_settings",
    "problem",
]
