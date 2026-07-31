"""Head-mass-conditional deterministic first seat (R2).

Defect D4 (jul28 pareto defect theory): on a head-dominated market whose
rank-1 action carries ~96% of the oracle, the stochastic band-balanced
pilot captures the head with probability ~0.49 where every deterministic
top-rank policy captures it with probability 1 (excess regret +0.0447 on
`trap_gate2_heat2d_gpt_sol_s20260770_w1` alone).  The same stochastic
diversification is why the pilot WINS elsewhere, so the repair must be
conditional, not a return to deterministic heads: before the first pilot
seat is sampled, the calibrated model's predicted positive mass per
candidate is computed outcome-blind; when the top candidate's share of the
total predicted mass exceeds a configured threshold, seat one becomes the
deterministic argmax (an exact point-mass seat, propensity one, logged
with its trigger condition) and every remaining seat keeps the stochastic
band-balanced schedule.

The assessment consumes any calibrated ranking (the base challenger's or
the region-conditional challenger's) and never observes candidate
outcomes.  Composition is config-gated in ``v9_candidate_policy``; no
existing module is modified.  The policy knows no workload, objective
name, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainRanking,
)
from agent_evolve.domain.patch import require_sha256

HEAD_MASS_CONDITIONAL_SEAT_POLICY_ID = "head_mass_conditional_seat"
HEAD_MASS_CONDITIONAL_SEAT_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:head-mass-conditional-seat-definition:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class HeadMassSeatConfig:
    """R2 constants; the threshold is an exact dyadic float."""

    head_mass_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            type(self.head_mass_threshold) is not float
            or not math.isfinite(self.head_mass_threshold)
            or not 0.0 < self.head_mass_threshold < 1.0
        ):
            raise ValueError(
                "head_mass_threshold must lie in (0, 1)"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "head_mass_threshold_hex": (
                self.head_mass_threshold.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class HeadMassAssessment:
    """One outcome-blind head-mass measurement over a scored market."""

    head_mass_fraction: float
    head_mass_threshold: float
    total_predicted_mass: float
    candidate_count: int
    argmax_action_sha256: str
    fired: bool

    def __post_init__(self) -> None:
        for name in ("head_mass_fraction", "total_predicted_mass"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if not 0.0 <= self.head_mass_fraction <= 1.0:
            raise ValueError("head_mass_fraction must lie in [0, 1]")
        if (
            type(self.head_mass_threshold) is not float
            or not 0.0 < self.head_mass_threshold < 1.0
        ):
            raise ValueError(
                "head_mass_threshold must lie in (0, 1)"
            )
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        require_sha256(
            self.argmax_action_sha256,
            "argmax_action_sha256",
        )
        if type(self.fired) is not bool:
            raise TypeError("fired must be exact")
        if self.fired and not (
            self.head_mass_fraction > self.head_mass_threshold
        ):
            raise ValueError(
                "an assessment can fire only strictly above threshold"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": HEAD_MASS_CONDITIONAL_SEAT_POLICY_ID,
            "policy_version": (
                HEAD_MASS_CONDITIONAL_SEAT_POLICY_VERSION
            ),
            "head_mass_fraction_hex": (
                self.head_mass_fraction.hex()
            ),
            "head_mass_threshold_hex": (
                self.head_mass_threshold.hex()
            ),
            "total_predicted_mass_hex": (
                self.total_predicted_mass.hex()
            ),
            "candidate_count": self.candidate_count,
            "argmax_action_sha256": self.argmax_action_sha256,
            "fired": self.fired,
            "trigger_condition": (
                "head_mass_fraction_strictly_above_threshold"
            ),
            "deterministic_seat_propensity_hex": (1.0).hex(),
            "candidate_outcomes_observed": False,
        }


@dataclass(frozen=True, slots=True)
class HeadMassSeatAssessor:
    """Measure predicted-head-mass concentration on a calibrated ranking."""

    config: HeadMassSeatConfig = HeadMassSeatConfig()
    policy_id: str = HEAD_MASS_CONDITIONAL_SEAT_POLICY_ID
    policy_version: int = HEAD_MASS_CONDITIONAL_SEAT_POLICY_VERSION
    definition_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.config) is not HeadMassSeatConfig:
            raise TypeError("config must be exact")
        self.config.__post_init__()
        if (
            type(self.policy_id) is not str
            or _TOKEN.fullmatch(self.policy_id) is None
            or self.policy_id != HEAD_MASS_CONDITIONAL_SEAT_POLICY_ID
            or self.policy_version
            != HEAD_MASS_CONDITIONAL_SEAT_POLICY_VERSION
        ):
            raise ValueError("policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "config": self.config.to_record(),
                    "predicted_mass": (
                        "p_positive_archive_gain_times_expected_"
                        "positive_gain"
                    ),
                    "trigger": (
                        "unique_top_candidate_mass_share_strictly_"
                        "above_threshold_first_seat_only"
                    ),
                    "deterministic_branch": (
                        "point_mass_argmax_propensity_one_logged"
                    ),
                    "remaining_seats": (
                        "stochastic_band_balanced_schedule_unchanged"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    def assess(
        self,
        ranking: CalibratedPositiveGainRanking,
    ) -> HeadMassAssessment:
        """Head-mass fraction of an outcome-blind calibrated ranking.

        Predicted positive mass of one candidate is
        ``p_positive_archive_gain * expected_positive_gain`` (the exploit
        term of the calibrated score, guaranteed non-negative).  The
        assessment fires only when the top candidate's share of the total
        mass strictly exceeds the threshold; exact ties across candidates
        therefore never fire (mass concentrated on ONE candidate is the
        trigger condition).  Zero total mass never fires.
        """

        if type(ranking) is not CalibratedPositiveGainRanking:
            raise TypeError("ranking must be exact")
        ranking.__post_init__()
        masses = tuple(
            (
                value.p_positive_archive_gain
                * value.expected_positive_gain,
                value.action_sha256,
            )
            for value in ranking.scores
        )
        total = math.fsum(mass for mass, _action in masses)
        head_mass = max(mass for mass, _action in masses)
        tied = tuple(
            action for mass, action in masses if mass == head_mass
        )
        # Canonical argmax: the lowest action hash among exact-mass
        # ties, so the assessment is order-independent.  A non-unique
        # maximum is by definition NOT mass concentrated on one
        # candidate and never fires.
        argmax_action = min(tied)
        fraction = head_mass / total if total > 0.0 else 0.0
        fired = (
            total > 0.0
            and len(tied) == 1
            and fraction > self.config.head_mass_threshold
        )
        return HeadMassAssessment(
            head_mass_fraction=float(fraction),
            head_mass_threshold=self.config.head_mass_threshold,
            total_predicted_mass=float(total),
            candidate_count=len(masses),
            argmax_action_sha256=argmax_action,
            fired=fired,
        )


__all__ = [
    "HEAD_MASS_CONDITIONAL_SEAT_POLICY_ID",
    "HEAD_MASS_CONDITIONAL_SEAT_POLICY_VERSION",
    "HeadMassAssessment",
    "HeadMassSeatAssessor",
    "HeadMassSeatConfig",
]
