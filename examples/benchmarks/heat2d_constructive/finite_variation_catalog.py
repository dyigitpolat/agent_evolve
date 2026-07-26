"""Deterministic parent-bound scalar moves for constructive Heat2D search."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_evolve.agentic import (
    FiniteVariationOption,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

from .artifact_boundary import decoder_source_sha256
from .candidate import CandidateConfig, normalize_candidate


CATALOG_ID = "heat2d_constructive_scalar_grid"
CATALOG_VERSION = 1
CATALOG_SCHEMA_ID = "heat2d_constructive_scalar_grid_v1"

# Nine exact alternatives per independently safe locus guarantee at least eight
# moves even when a parent lies exactly on the grid.  Capsule endpoint loci are
# deliberately absent because moving one endpoint independently could violate
# the decoder's minimum-segment-length constraint.
LOCUS_GRIDS: tuple[tuple[str, tuple[float, ...], str], ...] = (
    (
        "material_fraction",
        (0.32, 0.35, 0.38, 0.41, 0.45, 0.49, 0.52, 0.55, 0.58),
        "material_fraction",
    ),
    (
        "trunk.radius",
        (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
        "additive_geometry",
    ),
    (
        "left_lobe.center_x",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "additive_geometry",
    ),
    (
        "left_lobe.center_y",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "additive_geometry",
    ),
    (
        "left_lobe.radius_x",
        (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
        "additive_geometry",
    ),
    (
        "left_lobe.radius_y",
        (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
        "additive_geometry",
    ),
    (
        "left_lobe.angle_radians",
        (-1.20, -0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90, 1.20),
        "additive_geometry",
    ),
    (
        "right_bar.center_x",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "additive_geometry",
    ),
    (
        "right_bar.center_y",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "additive_geometry",
    ),
    (
        "right_bar.half_width",
        (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
        "additive_geometry",
    ),
    (
        "right_bar.half_height",
        (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
        "additive_geometry",
    ),
    (
        "right_bar.angle_radians",
        (-1.20, -0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90, 1.20),
        "additive_geometry",
    ),
    (
        "central_hole.center_x",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "subtractive_geometry",
    ),
    (
        "central_hole.center_y",
        (0.15, 0.24, 0.32, 0.41, 0.50, 0.59, 0.68, 0.76, 0.85),
        "subtractive_geometry",
    ),
    (
        "central_hole.radius_x",
        (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
        "subtractive_geometry",
    ),
    (
        "central_hole.radius_y",
        (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
        "subtractive_geometry",
    ),
    (
        "central_hole.angle_radians",
        (-1.20, -0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90, 1.20),
        "subtractive_geometry",
    ),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


_DEFINITION = {
    "schema_id": CATALOG_SCHEMA_ID,
    "catalog_id": CATALOG_ID,
    "catalog_version": CATALOG_VERSION,
    "decoder_source_sha256": decoder_source_sha256(),
    "candidate_schema": CandidateConfig.model_json_schema(by_alias=False),
    "locus_grids": [
        {"locus": locus, "values": list(values), "family": family}
        for locus, values, family in LOCUS_GRIDS
    ],
    "materialization": "exact_one_locus_replacement_no_clipping_or_repair",
}
CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:heat2d-constructive-catalog:v1\x00"
    + _canonical_bytes(_DEFINITION)
).hexdigest()


def _set_locus(payload: dict[str, Any], locus: str, value: float) -> None:
    path = locus.split(".")
    cursor: dict[str, Any] = payload
    for key in path[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            raise RuntimeError(f"candidate locus is not an object: {locus}")
        cursor = nested
    cursor[path[-1]] = value


class Heat2DFiniteVariationCatalog:
    """Enumerate exact independently valid one-scalar children."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION
    definition_sha256 = CATALOG_DEFINITION_SHA256
    option_families = tuple(sorted({family for _, _, family in LOCUS_GRIDS}))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("Heat2D finite catalog requires an exact FrozenJsonObject")
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        options: list[FiniteVariationOption] = []

        for locus_ordinal, (locus, grid, family) in enumerate(LOCUS_GRIDS):
            path = locus.split(".")
            current: Any = parent_payload
            for key in path:
                current = current[key]
            emitted_for_locus = 0
            for grid_ordinal, value in enumerate(grid):
                if value == current:
                    continue
                # JSON round-trip is an explicit deep copy of this typed tree;
                # it also fails if a future schema introduces a non-JSON value.
                child_payload = json.loads(_canonical_bytes(parent_payload))
                _set_locus(child_payload, locus, value)
                child = normalize_candidate(child_payload)
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("Heat2D candidate did not freeze as an object")
                options.append(
                    FiniteVariationOption(
                        option_id=(
                            f"heat2d.l{locus_ordinal:02d}.v{grid_ordinal:02d}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=family,
                        description=(
                            f"Set {locus} from {float(current):.17g} "
                            f"to {value:.17g}."
                        ),
                        metadata=(
                            ("catalog_definition_sha256", CATALOG_DEFINITION_SHA256),
                            ("locus", locus),
                            ("target_value", f"{value:.17g}"),
                        ),
                    )
                )
                emitted_for_locus += 1
            if emitted_for_locus < 8:
                raise RuntimeError(
                    f"Heat2D catalog emitted fewer than eight moves for {locus}"
                )
        return tuple(options)


__all__ = [
    "CATALOG_DEFINITION_SHA256",
    "CATALOG_ID",
    "CATALOG_SCHEMA_ID",
    "CATALOG_VERSION",
    "LOCUS_GRIDS",
    "Heat2DFiniteVariationCatalog",
]
