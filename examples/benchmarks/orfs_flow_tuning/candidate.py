"""Strict typed candidates for the OpenROAD-flow-scripts flow-tuning benchmark.

The fields are the published OpenROAD AutoTuner search space, taken verbatim
from two files shipped inside the pinned ORFS image:

* ``flow/designs/nangate45/gcd/autotuner.json`` — ``_SDC_CLK_PERIOD``,
  ``CORE_MARGIN``, ``CELL_PAD_IN_SITES_GLOBAL_PLACEMENT``,
  ``CELL_PAD_IN_SITES_DETAIL_PLACEMENT``, ``PLACE_DENSITY_LB_ADDON``,
  ``CTS_CLUSTER_SIZE``, ``CTS_CLUSTER_DIAMETER``
* ``flow/designs/asap7/ibex/autotuner_new.json`` — additionally
  ``CORE_UTILIZATION`` and ``CORE_ASPECT_RATIO``

Every field crosses the evaluator boundary as a typed scalar that the adapter
renders into ``make VAR=value`` overrides; no Tcl, Makefile, or SDC text is
ever authored by the optimizer.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


REPRESENTATION_ID = "orfs-flow-tuning-v1"
SCHEMA_VERSION = 1
_HASH_DOMAIN = b"agent-evolve:orfs-flow-tuning-candidate:v1\x00"


class CandidateConfig(BaseModel):
    """One immutable ORFS flow configuration evaluated by a full RTL-to-GDS run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    # --- constraint ---
    sdc_clk_period_ns: float = Field(default=0.46, ge=0.30, le=1.00)
    # --- floorplan ---
    core_utilization: int = Field(default=55, ge=20, le=70)
    core_aspect_ratio: float = Field(default=1.0, ge=0.9, le=1.1)
    core_margin: int = Field(default=2, ge=1, le=3)
    # --- placement ---
    cell_pad_global_placement: int = Field(default=0, ge=0, le=3)
    cell_pad_detail_placement: int = Field(default=0, ge=0, le=3)
    place_density_lb_addon: float = Field(default=0.20, ge=0.0, le=0.99)
    # --- clock tree synthesis ---
    cts_cluster_size: int = Field(default=30, ge=10, le=200)
    cts_cluster_diameter: int = Field(default=100, ge=20, le=400)


#: ``make`` variable each field is rendered into. ``sdc_clk_period_ns`` is not
#: here: it is applied by rewriting the design's own ``constraint.sdc``, exactly
#: as ORFS AutoTuner does for ``_SDC_CLK_PERIOD``.
MAKE_VARIABLES: dict[str, str] = {
    "core_utilization": "CORE_UTILIZATION",
    "core_aspect_ratio": "CORE_ASPECT_RATIO",
    "core_margin": "CORE_MARGIN",
    "cell_pad_global_placement": "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT",
    "cell_pad_detail_placement": "CELL_PAD_IN_SITES_DETAIL_PLACEMENT",
    "place_density_lb_addon": "PLACE_DENSITY_LB_ADDON",
    "cts_cluster_size": "CTS_CLUSTER_SIZE",
    "cts_cluster_diameter": "CTS_CLUSTER_DIAMETER",
}

DEFAULT_CANDIDATE: dict[str, object] = CandidateConfig().model_dump(mode="python")

#: Seeds spanning the measured tension: a tight-clock/dense point and a
#: relaxed-clock/loose point.
TIMING_HEAVY_SEED: dict[str, object] = {
    **DEFAULT_CANDIDATE,
    "sdc_clk_period_ns": 0.32,
    "core_utilization": 40,
    "place_density_lb_addon": 0.05,
}
AREA_HEAVY_SEED: dict[str, object] = {
    **DEFAULT_CANDIDATE,
    "sdc_clk_period_ns": 0.90,
    "core_utilization": 68,
    "place_density_lb_addon": 0.02,
}


def normalize_candidate(value: object) -> CandidateConfig:
    if isinstance(value, CandidateConfig):
        value = value.model_dump(mode="python")
    return CandidateConfig.model_validate(
        value, strict=True, by_alias=False, by_name=True
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
        normalize_candidate(TIMING_HEAVY_SEED),
        normalize_candidate(AREA_HEAVY_SEED),
    )


__all__ = [
    "AREA_HEAVY_SEED",
    "DEFAULT_CANDIDATE",
    "MAKE_VARIABLES",
    "REPRESENTATION_ID",
    "SCHEMA_VERSION",
    "TIMING_HEAVY_SEED",
    "CandidateConfig",
    "candidate_sha256",
    "canonical_candidate_bytes",
    "normalize_candidate",
    "seed_candidates",
]
