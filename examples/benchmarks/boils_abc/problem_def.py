"""AgentEvolve problem adapter for the pinned BOiLS/ABC evaluator."""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from agent_evolve.agentic import ObjectiveSpec

from .actions import (
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
    CandidateConfig,
    config_sha256,
    normalize_candidate,
)
from .evaluator import (
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
)


@runtime_checkable
class DetailedBoilsEvaluator(Protocol):
    """Small injectable port consumed by the AgentEvolve problem boundary."""

    def evaluate(self, config: object) -> BoilsEvaluation: ...


class BoilsAbcProblem:
    """Two-objective raw LUT/depth problem with lazy evaluator preflight."""

    candidate_model = CandidateConfig
    example_config = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}
    constraints_description = (
        f"sequence must contain exactly {SEQUENCE_LENGTH} allowlisted BOiLS action "
        "IDs; order and repetition are meaningful; raw ABC syntax is forbidden"
    )

    def __init__(
        self,
        settings: AbcEvaluatorSettings,
        *,
        evaluator: DetailedBoilsEvaluator | None = None,
    ) -> None:
        if type(settings) is not AbcEvaluatorSettings:
            raise TypeError("settings must be an exact AbcEvaluatorSettings")
        if evaluator is not None and not isinstance(evaluator, DetailedBoilsEvaluator):
            raise TypeError("evaluator must implement DetailedBoilsEvaluator")
        self.settings = settings
        self._evaluator = evaluator
        self._evaluator_lock = threading.Lock()

    @property
    def objectives(self):
        return [
            ObjectiveSpec("total_lut_count", "min"),
            ObjectiveSpec("total_levels", "min"),
        ]

    @property
    def evaluator(self) -> DetailedBoilsEvaluator:
        # Avoid a 157 MiB binary hash and filesystem dependency merely from
        # importing the problem module, while remaining safe under concurrent
        # first evaluations.
        if self._evaluator is None:
            with self._evaluator_lock:
                if self._evaluator is None:
                    self._evaluator = BoilsAbcEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> BoilsEvaluation:
        self.validate(config)
        try:
            return self.evaluator.evaluate(config)
        except AbcEvaluationError as exc:
            # Candidate-local ABC failures and timeouts are ordinary invalid
            # evaluations at the AgentEvolve boundary. Provenance failures and
            # programming errors intentionally remain fatal to the run.
            raise ValueError(
                "ABC candidate evaluation failed for "
                f"{exc.circuit_name}: {exc.diagnostics.status}"
            ) from exc

    def evaluate(self, config: object) -> dict[str, float]:
        return self.evaluate_detailed(config).objective_values

    def candidate_key(self, config: object) -> str:
        return config_sha256(config)

    def render_candidate(self, config: object) -> str:
        return " -> ".join(normalize_candidate(config))

    def search_space_description(self) -> str:
        actions = ", ".join(ACTION_IDS)
        circuits = ", ".join(spec.name for spec in self.settings.circuits)
        return (
            "BOiLS Berkeley ABC logic-synthesis sequence co-optimization.\n"
            f"Configuration: {{\"sequence\": [exactly {SEQUENCE_LENGTH} action IDs]}}\n"
            f"Allowed actions: {actions}.\n"
            "Actions execute after strash, followed by pinned LUT-6 mapping and "
            "CEC against the original circuit. Minimize the raw sum of LUT counts "
            "and the raw sum of mapped logic levels across the panel.\n"
            f"Circuit panel: {circuits}."
        )


def create_default_problem(
    *,
    affinity_sets: tuple[tuple[int, ...], ...] = (),
    per_circuit_timeout_s: float = 60.0,
) -> BoilsAbcProblem:
    return BoilsAbcProblem(
        AbcEvaluatorSettings.current_four_circuit(
            affinity_sets=affinity_sets,
            per_circuit_timeout_s=per_circuit_timeout_s,
        )
    )


# Importing a problem definition must remain cheap.  The pinned binary and
# circuits are cryptographically checked only if this default is evaluated.
problem = create_default_problem()


__all__ = [
    "BoilsAbcProblem",
    "CandidateConfig",
    "DetailedBoilsEvaluator",
    "create_default_problem",
    "problem",
]
