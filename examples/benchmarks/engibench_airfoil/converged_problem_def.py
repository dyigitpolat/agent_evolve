"""Convergence-qualified Airfoil v2 routing for AgentEvolve.

All candidate and problem semantics come from the frozen external Bernstein
panel v1 adapter.  Only the external evaluator executable and durable-output
locations change; ADflow-specific evidence stays outside AgentEvolve core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from examples.benchmarks.engibench_airfoil.problem_def import (
    AirfoilEvaluationError,
    AirfoilPanelEvaluation,
    AirfoilPanelEvaluator,
    AirfoilPanelProblem,
    AirfoilPanelSettings,
    candidate_sha256,
)


V2_EVALUATOR_ID = "engibench_airfoil_external_bernstein_panel_v2_convergence_witness"
EVIDENCE_CONTRACT_ID = "generic_evaluator_evidence_v1"
ADFLOW_EVALUATOR_ID = "adflow_steady_rans_v1"


class AirfoilConvergenceEvaluationError(AirfoilEvaluationError):
    """V2 error that retains any durable raw evaluator receipt."""

    def __init__(
        self,
        message: str,
        *,
        candidate_invalid: bool,
        record_path: Path,
        record: dict[str, Any] | None,
    ) -> None:
        super().__init__(message, candidate_invalid=candidate_invalid)
        self.record_path = record_path
        self.record = record


def _load_raw_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _check_success_evidence(evaluation: AirfoilPanelEvaluation) -> None:
    record = evaluation.record
    if record.get("schema_version") != 2 or record.get("evaluator_id") != V2_EVALUATOR_ID:
        raise AirfoilEvaluationError("Airfoil v2 evaluator identity/schema mismatch")
    points = record.get("points")
    if not isinstance(points, list) or len(points) != 3:
        raise AirfoilEvaluationError("Airfoil v2 receipt lacks exactly three point records")
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise AirfoilEvaluationError(f"Airfoil v2 point {index} is not an object")
        evidence = point.get("evaluator_evidence")
        if not isinstance(evidence, Mapping):
            raise AirfoilEvaluationError(f"Airfoil v2 point {index} lacks evaluator evidence")
        if (
            evidence.get("contract_id") != EVIDENCE_CONTRACT_ID
            or evidence.get("evaluator_id") != ADFLOW_EVALUATOR_ID
            or evidence.get("accepted") is not True
            or not isinstance(evidence.get("witness"), Mapping)
        ):
            raise AirfoilEvaluationError(f"Airfoil v2 point {index} evidence identity rejected")


class ConvergenceQualifiedAirfoilPanelEvaluator(AirfoilPanelEvaluator):
    """V2 subprocess port preserving success and failure evidence receipts."""

    def evaluate(self, config: object) -> AirfoilPanelEvaluation:
        key = candidate_sha256(config)
        record_path = self.settings.output_root / self._run_id / f"{key}.json"
        try:
            evaluation = super().evaluate(config)
            _check_success_evidence(evaluation)
            return evaluation
        except Exception as exc:
            record = _load_raw_receipt(record_path)
            candidate_invalid = (
                isinstance(exc, AirfoilEvaluationError) and exc.candidate_invalid
            )
            raise AirfoilConvergenceEvaluationError(
                str(exc),
                candidate_invalid=candidate_invalid,
                record_path=record_path,
                record=record,
            ) from exc


class ConvergenceQualifiedAirfoilPanelProblem(AirfoilPanelProblem):
    """Airfoil problem whose lazy evaluator preserves raw v2 receipts."""

    @property
    def evaluator(self) -> ConvergenceQualifiedAirfoilPanelEvaluator:
        if self._evaluator is None:
            with self._evaluator_lock:
                if self._evaluator is None:
                    self._evaluator = ConvergenceQualifiedAirfoilPanelEvaluator(self.settings)
        if not isinstance(self._evaluator, ConvergenceQualifiedAirfoilPanelEvaluator):
            raise TypeError("convergence-qualified Airfoil problem received the wrong evaluator")
        return self._evaluator

    def evaluate_raw(self, config: object) -> AirfoilPanelEvaluation:
        """Evaluate without translating a candidate failure into ``ValueError``."""

        self.validate(config)
        return self.evaluator.evaluate(config)


def local_default_converged_settings() -> AirfoilPanelSettings:
    base = AirfoilPanelSettings.local_default()
    workspace = Path(__file__).resolve().parents[4]
    return replace(
        base,
        evaluator_script=(
            workspace
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
            / "scripts"
            / "airfoil_external_panel_v2.py"
        ),
        output_root=(
            workspace
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
            / "experiment_logs"
            / "benchmark_q1"
            / "engibench_airfoil"
            / "external_panel_v2_convergence_witness"
            / "agent_evolve"
        ),
        work_root=Path("/tmp") / "agent_evolve_airfoil_external_panel_v2",
    )


def create_default_converged_problem() -> ConvergenceQualifiedAirfoilPanelProblem:
    return ConvergenceQualifiedAirfoilPanelProblem(local_default_converged_settings())


problem = create_default_converged_problem()


__all__ = [
    "AirfoilConvergenceEvaluationError",
    "ConvergenceQualifiedAirfoilPanelEvaluator",
    "ConvergenceQualifiedAirfoilPanelProblem",
    "create_default_converged_problem",
    "local_default_converged_settings",
    "problem",
]
