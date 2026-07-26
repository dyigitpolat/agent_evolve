"""Strict fixed-topology candidate schema for constructive Heat2D search."""

from __future__ import annotations

import json
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


REPRESENTATION_ID = "heat2d-constructive-layout-v0-qualification"
SCHEMA_VERSION = 1

UnitCoordinate = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
EllipseRadius = Annotated[float, Field(strict=True, ge=0.02, le=0.50)]
CapsuleRadius = Annotated[float, Field(strict=True, ge=0.02, le=0.30)]
BoxHalfExtent = Annotated[float, Field(strict=True, ge=0.02, le=0.50)]
AngleRadians = Annotated[float, Field(strict=True, ge=-math.pi, le=math.pi)]
MaterialFraction = Annotated[float, Field(strict=True, ge=0.30, le=0.60)]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class TrunkCapsule(_StrictFrozenModel):
    start_x: UnitCoordinate
    start_y: UnitCoordinate
    end_x: UnitCoordinate
    end_y: UnitCoordinate
    radius: CapsuleRadius

    @model_validator(mode="after")
    def validate_segment_length(self) -> "TrunkCapsule":
        if math.hypot(self.end_x - self.start_x, self.end_y - self.start_y) < 0.02:
            raise ValueError("trunk capsule segment must be at least 0.02 long")
        return self


class AdditiveEllipse(_StrictFrozenModel):
    center_x: UnitCoordinate
    center_y: UnitCoordinate
    radius_x: EllipseRadius
    radius_y: EllipseRadius
    angle_radians: AngleRadians


class AdditiveBox(_StrictFrozenModel):
    center_x: UnitCoordinate
    center_y: UnitCoordinate
    half_width: BoxHalfExtent
    half_height: BoxHalfExtent
    angle_radians: AngleRadians


class SubtractiveEllipse(AdditiveEllipse):
    pass


class CandidateConfig(_StrictFrozenModel):
    """One immutable four-primitive CSG program.

    Primitive identity, type, operation, and order are not model-authored
    fields.  They are injected by :meth:`decoder_mapping`, which makes
    topological drift impossible in this first developmental benchmark.
    """

    material_fraction: MaterialFraction
    trunk: TrunkCapsule
    left_lobe: AdditiveEllipse
    right_bar: AdditiveBox
    central_hole: SubtractiveEllipse

    def decoder_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "representation_id": REPRESENTATION_ID,
            "material_fraction": self.material_fraction,
            "primitives": [
                {
                    "primitive_id": "trunk",
                    "operation": "add",
                    "kind": "capsule",
                    **self.trunk.model_dump(mode="python"),
                },
                {
                    "primitive_id": "left_lobe",
                    "operation": "add",
                    "kind": "ellipse",
                    **self.left_lobe.model_dump(mode="python"),
                },
                {
                    "primitive_id": "right_bar",
                    "operation": "add",
                    "kind": "box",
                    **self.right_bar.model_dump(mode="python"),
                },
                {
                    "primitive_id": "central_hole",
                    "operation": "remove",
                    "kind": "ellipse",
                    **self.central_hole.model_dump(mode="python"),
                },
            ],
        }


SEED_LAYOUT_A: dict[str, object] = {
    "material_fraction": 0.45,
    "trunk": {
        "start_x": 0.50,
        "start_y": 0.02,
        "end_x": 0.50,
        "end_y": 0.72,
        "radius": 0.08,
    },
    "left_lobe": {
        "center_x": 0.30,
        "center_y": 0.72,
        "radius_x": 0.24,
        "radius_y": 0.13,
        "angle_radians": 0.35,
    },
    "right_bar": {
        "center_x": 0.70,
        "center_y": 0.68,
        "half_width": 0.23,
        "half_height": 0.07,
        "angle_radians": -0.42,
    },
    "central_hole": {
        "center_x": 0.50,
        "center_y": 0.53,
        "radius_x": 0.08,
        "radius_y": 0.10,
        "angle_radians": 0.0,
    },
}

SEED_LAYOUT_B: dict[str, object] = {
    "material_fraction": 0.38,
    "trunk": {
        "start_x": 0.22,
        "start_y": 0.08,
        "end_x": 0.72,
        "end_y": 0.78,
        "radius": 0.11,
    },
    "left_lobe": {
        "center_x": 0.24,
        "center_y": 0.55,
        "radius_x": 0.17,
        "radius_y": 0.25,
        "angle_radians": -0.70,
    },
    "right_bar": {
        "center_x": 0.68,
        "center_y": 0.42,
        "half_width": 0.15,
        "half_height": 0.18,
        "angle_radians": 0.62,
    },
    "central_hole": {
        "center_x": 0.43,
        "center_y": 0.48,
        "radius_x": 0.12,
        "radius_y": 0.06,
        "angle_radians": 0.40,
    },
}


def normalize_candidate(value: object) -> CandidateConfig:
    if isinstance(value, CandidateConfig):
        # Reparse to prevent subclasses or manually constructed instances from
        # bypassing the exact public schema.
        value = value.model_dump(mode="python")
    return CandidateConfig.model_validate(
        value,
        strict=True,
        by_alias=False,
        by_name=True,
    )


def canonical_candidate_json(value: object) -> str:
    candidate = normalize_candidate(value)
    return json.dumps(
        candidate.model_dump(mode="python"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def seed_layouts() -> tuple[CandidateConfig, CandidateConfig]:
    return normalize_candidate(SEED_LAYOUT_A), normalize_candidate(SEED_LAYOUT_B)


__all__ = [
    "CandidateConfig",
    "REPRESENTATION_ID",
    "SCHEMA_VERSION",
    "SEED_LAYOUT_A",
    "SEED_LAYOUT_B",
    "canonical_candidate_json",
    "normalize_candidate",
    "seed_layouts",
]
