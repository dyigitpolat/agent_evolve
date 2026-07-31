"""Candidate-ledger view of fixed-reference affine hypervolume."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

from agent_evolve.application.agentic_evolution import EvolutionCandidate

from .affine_hypervolume import (
    AffineHypervolume2DSpec,
    AffineHypervolumeSnapshot2D,
)


AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_ID = (
    "affine_candidate_archive_consequence_2d"
)
AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_VERSION = 1
_DEFINITION_DOMAIN = (
    b"agent-evolve:affine-candidate-archive-consequence:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class AffineCandidateArchiveConsequenceUtility:
    """Evaluate a candidate ledger in the campaign's affine HV currency."""

    spec: AffineHypervolume2DSpec
    utility_id: str = AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_ID
    utility_version: int = AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_VERSION

    def __post_init__(self) -> None:
        if type(self.spec) is not AffineHypervolume2DSpec:
            raise TypeError("spec must be an exact AffineHypervolume2DSpec")
        self.spec.__post_init__()
        if self.utility_id != AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_ID:
            raise ValueError("utility_id is immutable")
        if self.utility_version != (
            AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_VERSION
        ):
            raise ValueError("utility_version is immutable")

    @property
    def definition_sha256(self) -> str:
        self.__post_init__()
        return hashlib.sha256(
            _DEFINITION_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 1,
                    "affine_spec_definition_sha256": (
                        self.spec.definition_sha256
                    ),
                    "candidate_admission": (
                        "valid_and_operator_and_evidence_compliant"
                    ),
                    "hypothetical_admission": "one_raw_objective_point",
                    "result": "dimensionless_nonnegative_hypervolume",
                }
            )
        ).hexdigest()

    def _snapshot(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> AffineHypervolumeSnapshot2D:
        if (
            type(candidates) is not tuple
            or any(
                type(candidate) is not EvolutionCandidate
                for candidate in candidates
            )
        ):
            raise TypeError("candidates must be an exact candidate tuple")
        points: list[dict[str, float]] = []
        for candidate in candidates:
            candidate.__post_init__()
            if (
                candidate.valid
                and candidate.operator_compliant
                and candidate.evidence_compliant
            ):
                points.append(candidate.objective_map)
        return AffineHypervolumeSnapshot2D.create(
            spec=self.spec,
            archive_points=tuple(points),
        )

    def utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> float:
        return float(self._snapshot(candidates).base_hypervolume)

    def marginal_utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
        objective_point: Mapping[str, float],
    ) -> float:
        if not isinstance(objective_point, Mapping):
            raise TypeError("objective_point must be a mapping")
        point = {
            str(metric_id): float(value)
            for metric_id, value in objective_point.items()
        }
        if any(not math.isfinite(value) for value in point.values()):
            raise ValueError("objective_point values must be finite")
        return float(self._snapshot(candidates).marginal_gain(point))

    def portfolio_marginal_utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
        objective_points: tuple[Mapping[str, float], ...],
    ) -> float:
        if type(objective_points) is not tuple:
            raise TypeError("objective_points must be an exact tuple")
        if any(
            not isinstance(objective_point, Mapping)
            for objective_point in objective_points
        ):
            raise TypeError("objective_points must contain mappings")
        points = tuple(
            {
                str(metric_id): float(value)
                for metric_id, value in objective_point.items()
            }
            for objective_point in objective_points
        )
        if any(
            not math.isfinite(value)
            for point in points
            for value in point.values()
        ):
            raise ValueError("objective point values must be finite")
        return float(self._snapshot(candidates).joint_gain(points))


__all__ = [
    "AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_ID",
    "AFFINE_CANDIDATE_CONSEQUENCE_UTILITY_VERSION",
    "AffineCandidateArchiveConsequenceUtility",
]
