"""Generic archive-conditioned target synthesis for action consequence search.

Residual-cell targeting may legitimately fall back when a prior archive has a
single effective frontier point.  Expected-hypervolume action allocation still
has a complete decision frame in that case: the frozen affine archive, fixed
reference, and selected parent's measured objectives.  This adapter binds that
frame into the same authenticated target contract consumed by action forecasts
without inventing an evaluator outcome or a workload-specific aspiration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Mapping

from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.portfolio_campaign_runtime import (
    CAMPAIGN_FRONTIER_TARGET_KEY,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineHypervolumeSnapshot2D,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (
    AffineHypervolumeSnapshot3D,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest


ARCHIVE_CONDITIONED_TARGET_ID = "archive_conditioned_expected_hypervolume"
ARCHIVE_CONDITIONED_TARGET_VERSION = 1
ARCHIVE_CONDITIONED_TARGET_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:archive-conditioned-expected-hypervolume-target:v1;"
    b"inputs=frozen-affine-prior-archive-plus-measured-parent;"
    b"aspiration=parent-neutral-no-invented-outcome;cell=entire-normalized-prior-"
    b"archive;reference=unit-vector;purpose=expected-fixed-reference-hv-gain;"
    b"workload-model-provider-action-identifiers=false;future-outcomes=false"
).hexdigest()
_CELL_DOMAIN = b"agent-evolve:archive-conditioned-cell:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _decimal(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("archive-conditioned values must be finite exact floats")
    return format(0.0 if value == 0.0 else value, ".17g")


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:
        raise TypeError("archive-conditioned target payload must be an object")
    return result


def bind_archive_conditioned_affine_action_target(
    *,
    selection_request: PortfolioSelectionRequest,
    archive_utility: ArchiveUtilitySnapshot,
    affine_snapshot: AffineHypervolumeSnapshot2D | AffineHypervolumeSnapshot3D,
    parent_objectives: Mapping[str, float],
    lane_id: str,
) -> tuple[PortfolioSelectionRequest, CampaignPortfolioFrontierTarget]:
    """Attach a prior-only whole-archive expected-HV target to one request."""

    if type(selection_request) is not PortfolioSelectionRequest:
        raise TypeError("selection_request must be exact")
    selection_request.__post_init__()
    if type(archive_utility) is not ArchiveUtilitySnapshot:
        raise TypeError("archive_utility must be exact")
    archive_utility.__post_init__()
    if type(affine_snapshot) not in {
        AffineHypervolumeSnapshot2D,
        AffineHypervolumeSnapshot3D,
    }:
        raise TypeError("affine_snapshot must be an exact 2-D or 3-D snapshot")
    affine_snapshot.__post_init__()
    if affine_snapshot.to_record() != thaw_json(archive_utility.snapshot_receipt):
        raise ValueError("affine snapshot differs from the archive utility receipt")
    if archive_utility.definition_sha256 != affine_snapshot.spec.definition_sha256:
        raise ValueError("archive utility and affine definition differ")
    if not isinstance(parent_objectives, Mapping):
        raise TypeError("parent_objectives must be a mapping")
    metric_ids = affine_snapshot.spec.metric_ids
    if set(parent_objectives) != set(metric_ids):
        raise ValueError("parent objectives differ from the affine metric frame")
    parent = {metric_id: float(parent_objectives[metric_id]) for metric_id in metric_ids}
    if any(not math.isfinite(value) for value in parent.values()):
        raise ValueError("parent objectives must be finite")
    if type(lane_id) is not str or not lane_id:
        raise ValueError("lane_id must be non-empty")

    anchors = affine_snapshot.normalized_archive_points
    if not anchors:
        raise ValueError("archive-conditioned targeting requires prior archive points")
    normalized_parent = affine_snapshot.spec.normalize(parent)
    cell_payload = {
        "archive_utility_snapshot_sha256": archive_utility.snapshot_sha256,
        "normalized_anchor_points_hex": [
            [value.hex() for value in point] for point in anchors
        ],
        "metric_ids": list(metric_ids),
    }
    cell_sha256 = hashlib.sha256(
        _CELL_DOMAIN + _canonical_json(cell_payload)
    ).hexdigest()
    axes = []
    for axis in affine_snapshot.spec.axes:
        value = parent[axis.metric_id]
        axes.append(
            {
                "metric_id": axis.metric_id,
                "goal": axis.goal,
                "ideal_decimal": _decimal(float(axis.ideal)),
                "reference_decimal": _decimal(float(axis.reference)),
                "parent_value_decimal": _decimal(value),
                "aspiration_value_decimal": _decimal(value),
                "signed_parent_to_aspiration_delta_decimal": _decimal(0.0),
                "improving_raw_delta_sign": (
                    "negative" if axis.goal == "min" else "positive"
                ),
            }
        )
    payload = _object(
        {
            "schema_version": 2,
            "objective_space_target": {
                "purpose": (
                    "score_probabilistic_actions_by_expected_fixed_reference_"
                    "hypervolume_without_inventing_an_aspiration"
                ),
                "axes": axes,
            },
            "residual_frontier_cell": {
                "cell_sha256": cell_sha256,
                "geometry_sha256": affine_snapshot.spec.definition_sha256,
                "policy_definition_sha256": (
                    ARCHIVE_CONDITIONED_TARGET_DEFINITION_SHA256
                ),
                "normalized_anchor_points_decimal": [
                    [_decimal(value) for value in point] for point in anchors
                ],
                "normalized_aspiration_point_decimal": [
                    _decimal(value) for value in normalized_parent
                ],
                "potential_hypervolume_gain_decimal": _decimal(0.0),
                "parent_anchor_binding_distance_decimal": _decimal(0.0),
            },
            "acquisition_instruction": {
                "objective": "maximize_expected_fixed_reference_hypervolume_gain",
                "avoid_candidates_dominated_by_the_prior_archive": True,
                "evaluator_outcomes_remain_unknown": True,
                "target_realization_is_magnitude_sensitive": True,
                "tradeoffs_can_be_frontier_improving": True,
            },
            "normalized_orientation": "lower_is_better_on_every_axis",
            "epistemic_cutoff": {
                "archive_utility_snapshot_sha256": archive_utility.snapshot_sha256,
                "generation": archive_utility.generation,
                "current_or_future_candidate_outcomes_consulted": False,
            },
            "workload_identifiers_consulted": False,
            "model_or_provider_fields_consulted": False,
        }
    )
    target = CampaignPortfolioFrontierTarget(
        allocator_id=ARCHIVE_CONDITIONED_TARGET_ID,
        allocator_version=ARCHIVE_CONDITIONED_TARGET_VERSION,
        definition_sha256=ARCHIVE_CONDITIONED_TARGET_DEFINITION_SHA256,
        archive_utility_snapshot_sha256=archive_utility.snapshot_sha256,
        lane_id=lane_id,
        parent_configuration_sha256=(
            selection_request.finite_variation_contract
            .parent_configuration_sha256
        ),
        direction_id="expected_hypervolume",
        opportunity_rank=1,
        payload=payload,
    )
    context = thaw_json(selection_request.context)
    if type(context) is not dict:
        raise TypeError("selection request context must be an object")
    context[CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
    rebound_context = freeze_json(context)
    if type(rebound_context) is not FrozenJsonObject:
        raise TypeError("rebound selector context must be an object")
    return replace(selection_request, context=rebound_context), target


__all__ = [
    "ARCHIVE_CONDITIONED_TARGET_DEFINITION_SHA256",
    "ARCHIVE_CONDITIONED_TARGET_ID",
    "ARCHIVE_CONDITIONED_TARGET_VERSION",
    "bind_archive_conditioned_affine_action_target",
]
