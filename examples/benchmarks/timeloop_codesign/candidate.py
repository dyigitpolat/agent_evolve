"""Strict typed hardware candidates for the Timeloop co-design benchmark."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


REPRESENTATION_ID = "timeloop-accelerator-codesign-v1"
SCHEMA_VERSION = 1
_HASH_DOMAIN = b"agent-evolve:timeloop-codesign-candidate:v1\x00"


class CandidateConfig(BaseModel):
    """One immutable accelerator architecture evaluated by a fixed mapper.

    The fields are deliberately discrete.  They change real compute, storage,
    precision, and local-dataflow resources while preventing arbitrary YAML or
    command text from crossing the evaluator boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    global_buffer_depth: Literal[256, 512, 1024, 2048] = 512
    global_buffer_width: Literal[64, 128, 256] = 128
    pe_mesh_x: Literal[4, 8, 16, 32] = 8
    datawidth_bits: Literal[8, 16] = 8
    register_enabled: bool = True


DEFAULT_CANDIDATE: dict[str, object] = {
    "global_buffer_depth": 512,
    "global_buffer_width": 128,
    "pe_mesh_x": 8,
    "datawidth_bits": 8,
    "register_enabled": True,
}

COMPUTE_HEAVY_SEED: dict[str, object] = {
    **DEFAULT_CANDIDATE,
    "global_buffer_depth": 256,
    "pe_mesh_x": 16,
}

MEMORY_HEAVY_SEED: dict[str, object] = {
    **DEFAULT_CANDIDATE,
    "global_buffer_depth": 1024,
    "pe_mesh_x": 4,
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


def seed_candidates() -> tuple[CandidateConfig, CandidateConfig, CandidateConfig]:
    return (
        normalize_candidate(DEFAULT_CANDIDATE),
        normalize_candidate(COMPUTE_HEAVY_SEED),
        normalize_candidate(MEMORY_HEAVY_SEED),
    )


__all__ = [
    "COMPUTE_HEAVY_SEED",
    "DEFAULT_CANDIDATE",
    "MEMORY_HEAVY_SEED",
    "REPRESENTATION_ID",
    "SCHEMA_VERSION",
    "CandidateConfig",
    "candidate_sha256",
    "canonical_candidate_bytes",
    "normalize_candidate",
    "seed_candidates",
]
