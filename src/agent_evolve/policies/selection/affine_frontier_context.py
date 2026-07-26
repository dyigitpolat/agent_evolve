"""Authenticated, workload-neutral Pareto-frontier context projection.

The projector accepts only a frozen affine archive cutoff and the selected
parent's already-evaluated objective vector.  It exposes normalized geometry
in decimal text so language models are not asked to interpret binary64 hex
exponents.  No option outcome, current-wave result, model profile, provider
field, workload identifier, or action name is consulted.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)

if TYPE_CHECKING:
    from agent_evolve.application.agentic_evolution import EvolutionCandidate
    from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot


PROJECTOR_ID = "authenticated_affine_frontier_context"
PROJECTOR_VERSION = 1
PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:authenticated-affine-frontier-context:v1;"
    b"input=frozen-affine-archive-cutoff,selected-evaluated-parent;"
    b"dimensions=2-or-3;orientation=normalized-lower-is-better;"
    b"archive-points,parent-point,axis-gaps,reference-directions=true;"
    b"numeric-rendering=roundtrip-decimal;"
    b"model-provider-workload-action-current-outcome-fields=false"
).hexdigest()


class AffineFrontierContextMode(str, Enum):
    """Closed composition-root switch for the optional generic projection."""

    OFF = "off"
    AUTHENTICATED_AFFINE_V1 = "authenticated_affine_v1"


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("frontier projection did not freeze to an object")
    return result


def _finite_hex(value: object, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be a binary64 hex string")
    result = float.fromhex(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _decimal(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("frontier projection cannot render a non-finite value")
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".17g")


def _normalize(value: float, *, goal: str, ideal: float, reference: float) -> float:
    if goal == "min":
        if not reference > ideal:
            raise ValueError("minimization reference must exceed its ideal")
        return (value - ideal) / (reference - ideal)
    if goal == "max":
        if not ideal > reference:
            raise ValueError("maximization ideal must exceed its reference")
        return (ideal - value) / (ideal - reference)
    raise ValueError("affine frontier axis goal must be min or max")


def _reference_directions(dimension: int) -> list[dict[str, object]]:
    directions: list[tuple[str, tuple[float, ...]]] = [
        (
            f"axis_{index + 1}_extreme",
            tuple(1.0 if index == other else 0.0 for other in range(dimension)),
        )
        for index in range(dimension)
    ]
    if dimension == 3:
        directions.extend(
            (
                f"axes_{left + 1}_{right + 1}_tradeoff",
                tuple(
                    0.5 if index in (left, right) else 0.0
                    for index in range(dimension)
                ),
            )
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    directions.append(
        (
            "balanced_tradeoff",
            tuple(1.0 / dimension for _ in range(dimension)),
        )
    )
    return [
        {
            "direction_id": identifier,
            "normalized_importance_decimal": [_decimal(value) for value in weights],
        }
        for identifier, weights in directions
    ]


@dataclass(frozen=True, slots=True)
class AuthenticatedAffineFrontierContextProjector:
    """Project 2-D/3-D affine archive geometry without domain special cases."""

    projector_id: str = PROJECTOR_ID
    projector_version: int = PROJECTOR_VERSION
    definition_sha256: str = PROJECTOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.projector_id != PROJECTOR_ID
            or self.projector_version != PROJECTOR_VERSION
            or self.definition_sha256 != PROJECTOR_DEFINITION_SHA256
        ):
            raise ValueError("affine frontier projector identity drifted")

    def project(
        self,
        *,
        archive_utility: ArchiveUtilitySnapshot,
        parent: EvolutionCandidate,
    ) -> CampaignPortfolioArchiveContextProjection:
        from agent_evolve.application.agentic_evolution import EvolutionCandidate
        from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot

        self.__post_init__()
        if type(archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("archive_utility must be an exact frozen snapshot")
        ArchiveUtilitySnapshot.__post_init__(archive_utility)
        if type(parent) is not EvolutionCandidate:
            raise TypeError("parent must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(parent)
        if not parent.valid:
            raise ValueError("frontier context requires an evaluated valid parent")

        snapshot = thaw_json(archive_utility.snapshot_receipt)
        if type(snapshot) is not dict:
            raise TypeError("affine snapshot receipt must be an object")
        if snapshot.get("definition_sha256") != archive_utility.definition_sha256:
            raise ValueError("affine snapshot and outer utility definitions differ")
        spec = snapshot.get("spec")
        axes = spec.get("axes") if type(spec) is dict else None
        if type(axes) is not list or len(axes) not in (2, 3):
            raise ValueError("frontier projection supports exact affine 2-D or 3-D")
        dimension = len(axes)
        parsed_axes: list[dict[str, object]] = []
        metric_ids: list[str] = []
        for index, axis in enumerate(axes):
            if type(axis) is not dict or type(axis.get("metric_id")) is not str:
                raise ValueError("affine axis record is malformed")
            metric_id = axis["metric_id"]
            goal = axis.get("goal")
            if goal not in ("min", "max"):
                raise ValueError("affine axis goal is malformed")
            ideal = _finite_hex(axis.get("ideal_hex"), f"axis[{index}].ideal")
            reference = _finite_hex(
                axis.get("reference_hex"), f"axis[{index}].reference"
            )
            _normalize(ideal, goal=goal, ideal=ideal, reference=reference)
            metric_ids.append(metric_id)
            parsed_axes.append(
                {
                    "axis_index": index,
                    "metric_id": metric_id,
                    "source_goal": goal,
                    "normalized_goal": "minimize",
                    "ideal_decimal": _decimal(ideal),
                    "reference_decimal": _decimal(reference),
                    "normalized_ideal_decimal": "0",
                    "normalized_reference_decimal": "1",
                    "_ideal": ideal,
                    "_reference": reference,
                }
            )
        if len(set(metric_ids)) != dimension:
            raise ValueError("affine axes must use distinct metric IDs")
        parent_objectives = parent.objective_map
        if set(parent_objectives) != set(metric_ids):
            raise ValueError("parent objective vector differs from affine axes")
        parent_point = tuple(
            _normalize(
                float(parent_objectives[str(axis["metric_id"])]),
                goal=str(axis["source_goal"]),
                ideal=float(axis["_ideal"]),
                reference=float(axis["_reference"]),
            )
            for axis in parsed_axes
        )

        raw_normalized_points = snapshot.get("normalized_archive_points")
        if type(raw_normalized_points) is not list or not raw_normalized_points:
            raise ValueError("affine snapshot requires normalized archive points")
        normalized_points: tuple[tuple[float, ...], ...] = tuple(
            tuple(
                _finite_hex(value, f"archive[{point_index}][{axis_index}]")
                for axis_index, value in enumerate(point)
            )
            for point_index, point in enumerate(raw_normalized_points)
            if type(point) is list and len(point) == dimension
        )
        if len(normalized_points) != len(raw_normalized_points):
            raise ValueError("normalized archive point dimension is malformed")
        axis_best = tuple(
            min(point[index] for point in normalized_points)
            for index in range(dimension)
        )
        axis_worst = tuple(
            max(point[index] for point in normalized_points)
            for index in range(dimension)
        )
        dominated_by_archive = any(
            all(source <= target for source, target in zip(point, parent_point))
            and any(source < target for source, target in zip(point, parent_point))
            for point in normalized_points
        )
        dominates_archive_count = sum(
            all(source <= target for source, target in zip(parent_point, point))
            and any(source < target for source, target in zip(parent_point, point))
            for point in normalized_points
        )
        public_axes = [
            {key: value for key, value in axis.items() if not key.startswith("_")}
            for axis in parsed_axes
        ]
        inner_snapshot_sha256 = snapshot.get("snapshot_sha256")
        if type(inner_snapshot_sha256) is not str:
            raise ValueError("affine snapshot lacks its inner identity")
        payload = _object(
            {
                "schema_version": 1,
                "epistemic_cutoff": {
                    "generation": archive_utility.generation,
                    "archive_sha256": archive_utility.archive_sha256,
                    "outer_archive_utility_snapshot_sha256": (
                        archive_utility.snapshot_sha256
                    ),
                    "inner_affine_snapshot_sha256": inner_snapshot_sha256,
                    "current_or_future_candidate_outcomes_consulted": False,
                },
                "optimization_frame": {
                    "indicator": spec.get("indicator"),
                    "dimension": dimension,
                    "normalized_orientation": "lower_is_better_on_every_axis",
                    "target": "expand_the_pareto_frontier_under_the_fixed_reference",
                    "simultaneous_improvement_on_every_axis_required": False,
                    "tradeoffs_can_be_frontier_improving": True,
                    "base_hypervolume_decimal": _decimal(
                        _finite_hex(
                            snapshot.get("base_hypervolume_hex"),
                            "base_hypervolume",
                        )
                    ),
                    "axes": public_axes,
                    "reference_directions": _reference_directions(dimension),
                },
                "archive": {
                    "point_count": len(normalized_points),
                    "normalized_points_decimal": [
                        [_decimal(value) for value in point]
                        for point in normalized_points
                    ],
                    "axis_best_decimal": [_decimal(value) for value in axis_best],
                    "axis_worst_decimal": [_decimal(value) for value in axis_worst],
                },
                "parent": {
                    "candidate_id": parent.candidate_id.value,
                    "configuration_sha256": parent.occurrence.configuration_hash,
                    "normalized_point_decimal": [
                        _decimal(value) for value in parent_point
                    ],
                    "axis_gap_from_archive_best_decimal": [
                        _decimal(value - best)
                        for value, best in zip(parent_point, axis_best)
                    ],
                    "axis_slack_to_reference_decimal": [
                        _decimal(1.0 - value) for value in parent_point
                    ],
                    "dominated_by_an_archive_point": dominated_by_archive,
                    "strictly_dominates_archive_point_count": (
                        dominates_archive_count
                    ),
                },
            }
        )
        return CampaignPortfolioArchiveContextProjection(
            projector_id=self.projector_id,
            projector_version=self.projector_version,
            definition_sha256=self.definition_sha256,
            archive_utility_snapshot_sha256=archive_utility.snapshot_sha256,
            parent_configuration_sha256=parent.occurrence.configuration_hash,
            payload=payload,
        )


def affine_frontier_context_projector(
    mode: AffineFrontierContextMode | str,
) -> AuthenticatedAffineFrontierContextProjector | None:
    """Resolve one explicit mechanism arm without reading process state."""

    selected = mode if type(mode) is AffineFrontierContextMode else AffineFrontierContextMode(mode)
    if selected is AffineFrontierContextMode.OFF:
        return None
    if selected is AffineFrontierContextMode.AUTHENTICATED_AFFINE_V1:
        return AuthenticatedAffineFrontierContextProjector()
    raise AssertionError("unreachable affine frontier context mode")


__all__ = [
    "AffineFrontierContextMode",
    "AuthenticatedAffineFrontierContextProjector",
    "PROJECTOR_DEFINITION_SHA256",
    "PROJECTOR_ID",
    "PROJECTOR_VERSION",
    "affine_frontier_context_projector",
]
