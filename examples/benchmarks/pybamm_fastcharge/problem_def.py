"""Typed three-stage fast-charging protocol problem for pybamm_fastcharge.

Candidate = three CC-stage C-rates plus two monotone switch voltages; the
final stage charges to 4.2 V followed by a CV hold to C/50 (fixed evaluator
policy).  Physical bounds and threshold monotonicity are schema facts, so
invalid protocols are rejected instantly with informative reasons and never
reach the solver.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evolve.agentic import ObjectiveSpec

from .evaluator import (
    OBJECTIVE_CHARGE_TIME,
    OBJECTIVE_TEMP_RISE,
    PybammEvaluatorSettings,
    PybammProtocolEvaluation,
    PybammSubprocessEvaluator,
    default_domain_python,
)


VMAX_V = 4.2
CRate = Annotated[float, Field(strict=True, gt=0.0, le=5.0)]
SwitchVoltage = Annotated[float, Field(strict=True, gt=3.0, lt=VMAX_V)]


class ChargingProtocolCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    stage1_c_rate: CRate
    stage2_c_rate: CRate
    stage3_c_rate: CRate
    switch_v1: SwitchVoltage
    switch_v2: SwitchVoltage

    @model_validator(mode="after")
    def validate_monotone_thresholds(self) -> "ChargingProtocolCandidate":
        if not self.switch_v1 < self.switch_v2:
            raise ValueError(
                f"require 3.0 < switch_v1({self.switch_v1}) < "
                f"switch_v2({self.switch_v2}) < {VMAX_V}"
            )
        return self


# Conservative CC-CV-like baseline: uniform 1C to 4.2 V, then the CV hold.
# This is the Gate-A ``baseline_1C`` protocol with full determinism evidence.
SEED_BASELINE_1C: dict[str, float] = {
    "stage1_c_rate": 1.0,
    "stage2_c_rate": 1.0,
    "stage3_c_rate": 1.0,
    "switch_v1": 4.05,
    "switch_v2": 4.15,
}


def normalize_candidate(value: object) -> ChargingProtocolCandidate:
    if isinstance(value, ChargingProtocolCandidate):
        value = value.model_dump(mode="python")
    return ChargingProtocolCandidate.model_validate(
        value, strict=True, by_alias=False, by_name=True
    )


def canonical_candidate_json(value: object) -> str:
    return json.dumps(
        normalize_candidate(value).model_dump(mode="python"),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )


class PybammFastChargeProblem:
    candidate_model = ChargingProtocolCandidate
    example_config = SEED_BASELINE_1C
    constraints_description = (
        "three CC-stage C-rates in (0, 5] and two switch voltages with "
        "3.0 < switch_v1 < switch_v2 < 4.2; the cell, solver, mesh, timeout, "
        "plating-margin floor, and CV cutoff are frozen evaluator facts"
    )

    def __init__(
        self,
        settings: PybammEvaluatorSettings,
        *,
        evaluator: object | None = None,
    ) -> None:
        if type(settings) is not PybammEvaluatorSettings:
            raise TypeError("settings must be exact PybammEvaluatorSettings")
        self.settings = settings
        self._evaluator = evaluator

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (
            ObjectiveSpec(OBJECTIVE_CHARGE_TIME, "min"),
            ObjectiveSpec(OBJECTIVE_TEMP_RISE, "min"),
        )

    @property
    def evaluator(self) -> object:
        if self._evaluator is None:
            self._evaluator = PybammSubprocessEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> PybammProtocolEvaluation:
        candidate = normalize_candidate(config)
        return self.evaluator.evaluate_protocol(
            candidate.model_dump(mode="python")
        )

    def evaluate(self, config: object) -> dict[str, float]:
        return dict(self.evaluate_detailed(config).objective_values)

    @staticmethod
    def candidate_key(config: object) -> str:
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:pybamm-fastcharge-candidate-key:v1\x00")
        digest.update(canonical_candidate_json(config).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def render_candidate(config: object) -> str:
        candidate = normalize_candidate(config)
        return (
            f"CC {candidate.stage1_c_rate:g}C->{candidate.switch_v1:.2f}V, "
            f"{candidate.stage2_c_rate:g}C->{candidate.switch_v2:.2f}V, "
            f"{candidate.stage3_c_rate:g}C->{VMAX_V}V, CV hold to C/50"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Three-stage CC fast-charging protocol for an LG M50 (Chen2020) "
            "cell simulated with a lumped-thermal DFN model from 10% SOC via "
            "the IDAKLU solver.  Stages charge at the candidate C-rates until "
            "the two monotone switch voltages and then 4.2 V, finishing with "
            "a CV hold to C/50.  Minimize total charge time and peak "
            "temperature rise; protocols that plate lithium (proxy margin "
            "below 0.01 V), fail the solver, or exceed the wall-clock cap "
            "are invalid with explicit reasons."
        )


def default_settings(*, mesh_factor: int = 2) -> PybammEvaluatorSettings:
    return PybammEvaluatorSettings(
        domain_python=default_domain_python(),
        mesh_factor=mesh_factor,
    )


__all__ = [
    "ChargingProtocolCandidate",
    "PybammFastChargeProblem",
    "SEED_BASELINE_1C",
    "VMAX_V",
    "canonical_candidate_json",
    "default_settings",
    "normalize_candidate",
]
