"""Closed, immutable v2 Timeloop co-design candidate schema."""

from __future__ import annotations

import hashlib
from itertools import product
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


REPRESENTATION_ID = "timeloop-network-mapspace-policy-codesign-v2"
SCHEMA_VERSION = 2
_HASH_DOMAIN = b"agent-evolve:timeloop-codesign-candidate:v2\x00"

DataflowFamily = Literal[
    "weight_stationary",
    "output_stationary",
    "row_stationary",
    "no_local_reuse",
]
SpatialAxis = Literal["C", "M", "P", "Q"]
SpatialUtilization = Literal["low", "medium", "high", "full"]
ResidencyBias = Literal[
    "balanced",
    "input_reuse",
    "weight_reuse",
    "output_reuse",
]
OuterLoopOrder = Literal[
    "channel_major",
    "channel_then_spatial",
    "input_channel_major",
    "output_spatial_major",
    "filter_major",
    "spatial_major",
]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class MappingPolicyBlock(_ClosedModel):
    """One high-level policy compiled into benchmark-owned constraints."""

    dataflow_family: DataflowFamily = "row_stationary"
    primary_spatial_axis: SpatialAxis = "M"
    spatial_utilization: SpatialUtilization = "medium"
    buffer_residency_bias: ResidencyBias = "balanced"
    outer_loop_order: OuterLoopOrder = "channel_then_spatial"


class CandidateConfig(_ClosedModel):
    """Five architecture choices and one policy for each of three medoids."""

    global_buffer_depth: Literal[256, 512, 1024, 2048] = 512
    global_buffer_width: Literal[64, 128, 256] = 128
    pe_mesh_x: Literal[4, 8, 16, 32] = 8
    datawidth_bits: Literal[8, 16] = 8
    register_enabled: bool = True
    policy_cluster_0: MappingPolicyBlock = Field(default_factory=MappingPolicyBlock)
    policy_cluster_1: MappingPolicyBlock = Field(default_factory=MappingPolicyBlock)
    policy_cluster_2: MappingPolicyBlock = Field(default_factory=MappingPolicyBlock)


ARCHITECTURE_FIELD_GRIDS: tuple[tuple[str, tuple[object, ...], str], ...] = (
    ("global_buffer_depth", (256, 512, 1024, 2048), "memory_capacity"),
    ("global_buffer_width", (64, 128, 256), "memory_bandwidth"),
    ("pe_mesh_x", (4, 8, 16, 32), "compute_parallelism"),
    ("datawidth_bits", (8, 16), "numeric_precision"),
    ("register_enabled", (False, True), "local_storage"),
)

POLICY_FIELD_GRIDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "dataflow_family",
        (
            "weight_stationary",
            "output_stationary",
            "row_stationary",
            "no_local_reuse",
        ),
        "dataflow_family",
    ),
    ("primary_spatial_axis", ("C", "M", "P", "Q"), "spatial_axis"),
    (
        "spatial_utilization",
        ("low", "medium", "high", "full"),
        "spatial_utilization",
    ),
    (
        "buffer_residency_bias",
        ("balanced", "input_reuse", "weight_reuse", "output_reuse"),
        "residency_bias",
    ),
    (
        "outer_loop_order",
        (
            "channel_major",
            "channel_then_spatial",
            "input_channel_major",
            "output_spatial_major",
            "filter_major",
            "spatial_major",
        ),
        "outer_loop_order",
    ),
)

POLICY_BLOCK_FIELDS = (
    "policy_cluster_0",
    "policy_cluster_1",
    "policy_cluster_2",
)

DEFAULT_POLICY: dict[str, str] = {
    "dataflow_family": "row_stationary",
    "primary_spatial_axis": "M",
    "spatial_utilization": "medium",
    "buffer_residency_bias": "balanced",
    "outer_loop_order": "channel_then_spatial",
}

DEFAULT_CANDIDATE: dict[str, object] = {
    "global_buffer_depth": 512,
    "global_buffer_width": 128,
    "pe_mesh_x": 8,
    "datawidth_bits": 8,
    "register_enabled": True,
    **{field: dict(DEFAULT_POLICY) for field in POLICY_BLOCK_FIELDS},
}


def normalize_candidate(value: object) -> CandidateConfig:
    if isinstance(value, CandidateConfig):
        value = value.model_dump(mode="python")
    return CandidateConfig.model_validate(
        value,
        strict=True,
        by_alias=False,
        by_name=True,
    )


def canonical_candidate_bytes(value: object) -> bytes:
    candidate = normalize_candidate(value)
    return json.dumps(
        candidate.model_dump(mode="python"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def candidate_sha256(value: object) -> str:
    return hashlib.sha256(_HASH_DOMAIN + canonical_candidate_bytes(value)).hexdigest()


def iter_policy_blocks() -> tuple[MappingPolicyBlock, ...]:
    """Return the exact 1,536-element closed policy palette."""

    grids = tuple(values for _, values, _ in POLICY_FIELD_GRIDS)
    return tuple(
        MappingPolicyBlock.model_validate(
            dict(
                zip((field for field, _, _ in POLICY_FIELD_GRIDS), values, strict=True)
            ),
            strict=True,
        )
        for values in product(*grids)
    )


def policy_cardinality() -> int:
    cardinality = 1
    for _, values, _ in POLICY_FIELD_GRIDS:
        cardinality *= len(values)
    return cardinality


def architecture_cardinality() -> int:
    cardinality = 1
    for _, values, _ in ARCHITECTURE_FIELD_GRIDS:
        cardinality *= len(values)
    return cardinality


def candidate_cardinality() -> int:
    return architecture_cardinality() * policy_cardinality() ** len(POLICY_BLOCK_FIELDS)


__all__ = [
    "ARCHITECTURE_FIELD_GRIDS",
    "DEFAULT_CANDIDATE",
    "DEFAULT_POLICY",
    "POLICY_BLOCK_FIELDS",
    "POLICY_FIELD_GRIDS",
    "REPRESENTATION_ID",
    "SCHEMA_VERSION",
    "CandidateConfig",
    "MappingPolicyBlock",
    "architecture_cardinality",
    "candidate_cardinality",
    "candidate_sha256",
    "canonical_candidate_bytes",
    "iter_policy_blocks",
    "normalize_candidate",
    "policy_cardinality",
]
