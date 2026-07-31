"""Outcome-blind finite acquisition reservoir for constructive Heat2D.

This adapter is optional workload integration, not AgentEvolve core logic.  It
maps the typed Heat configuration to a normalized numeric vector and returns a
deterministic mixture of prior-configuration neighborhoods and global legal
configurations.  It receives configuration identities but no objective values.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.finite_acquisition_space import (
    FiniteAcquisitionSpaceRequest,
)

from .candidate import CandidateConfig, normalize_candidate, seed_layouts
from .finite_variation_catalog import LOCUS_GRIDS, Heat2DFiniteVariationCatalog


SPACE_ID = "heat2d_constructive_finite_acquisition"
SPACE_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:heat2d-finite-acquisition-space:v1\x00"
_ORDER_DOMAIN = b"agent-evolve:heat2d-finite-acquisition-local-order:v1\x00"
_SEEDS = seed_layouts()
_LINEAGE_SIGNATURES = tuple(
    (
        value.trunk.start_x,
        value.trunk.start_y,
        value.trunk.end_x,
        value.trunk.end_y,
    )
    for value in _SEEDS
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _set_locus(payload: dict[str, Any], locus: str, value: float) -> None:
    cursor = payload
    path = locus.split(".")
    for name in path[:-1]:
        nested = cursor[name]
        if type(nested) is not dict:
            raise TypeError("Heat2D locus traversed a non-object")
        cursor = nested
    cursor[path[-1]] = value


def _get_locus(payload: dict[str, Any], locus: str) -> float:
    value: object = payload
    for name in locus.split("."):
        if type(value) is not dict:
            raise TypeError("Heat2D locus traversed a non-object")
        value = value[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Heat2D acquisition locus must be numeric")
    return float(value)


def _lineage(candidate: CandidateConfig) -> int:
    signature = (
        candidate.trunk.start_x,
        candidate.trunk.start_y,
        candidate.trunk.end_x,
        candidate.trunk.end_y,
    )
    try:
        return _LINEAGE_SIGNATURES.index(signature)
    except ValueError as error:
        raise ValueError("candidate escaped the two sealed Heat lineages") from error


_DEFINITION = {
    "schema_version": 1,
    "space_id": SPACE_ID,
    "space_version": SPACE_VERSION,
    "lineages": [value.model_dump(mode="json") for value in _SEEDS],
    "locus_grids": [
        {"locus": locus, "grid": list(grid), "family": family}
        for locus, grid, family in LOCUS_GRIDS
    ],
    "features": "lineage_indicator_then_minmax_grid_coordinates",
    "reservoir": (
        "up_to_one_quarter_prior-neighborhood-options-then-seeded-uniform-"
        "legal-grid-fill"
    ),
    "local_order": "request-seed-and-configuration-hash-sha256",
    "outcome_access": False,
}
SPACE_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN + _canonical_json(_DEFINITION)
).hexdigest()


@dataclass(frozen=True, slots=True)
class Heat2DFiniteAcquisitionSpace:
    space_id: str = SPACE_ID
    space_version: int = SPACE_VERSION
    definition_sha256: str = SPACE_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.space_id != SPACE_ID
            or self.space_version != SPACE_VERSION
            or self.definition_sha256 != SPACE_DEFINITION_SHA256
        ):
            raise ValueError("Heat2D acquisition-space identity drifted")

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]:
        self.__post_init__()
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("Heat2D acquisition configuration must be frozen")
        candidate = normalize_candidate(thaw_json(configuration))
        payload = candidate.model_dump(mode="python")
        result = [float(_lineage(candidate))]
        for locus, grid, _family in LOCUS_GRIDS:
            raw = _get_locus(payload, locus)
            normalized = (raw - grid[0]) / (grid[-1] - grid[0])
            if not -1.0e-12 <= normalized <= 1.0 + 1.0e-12:
                raise ValueError("Heat2D candidate escaped its acquisition grid")
            result.append(min(1.0, max(0.0, float(normalized))))
        return tuple(result)

    def candidates(
        self,
        request: FiniteAcquisitionSpaceRequest,
    ) -> tuple[FrozenJsonObject, ...]:
        self.__post_init__()
        if type(request) is not FiniteAcquisitionSpaceRequest:
            raise TypeError("request must be an exact space request")
        FiniteAcquisitionSpaceRequest.__post_init__(request)
        excluded = set(request.excluded_configuration_sha256s)
        selected: dict[str, FrozenJsonObject] = {}
        local: dict[str, FrozenJsonObject] = {}
        catalog = Heat2DFiniteVariationCatalog()
        for observed in request.observed_configurations:
            for option in catalog.options(observed):
                identity = option.child_configuration_sha256
                if identity not in excluded:
                    local.setdefault(identity, option.child_configuration)
        local_limit = min(len(local), request.pool_size // 4)
        ordered_local = sorted(
            local,
            key=lambda identity: (
                hashlib.sha256(
                    _ORDER_DOMAIN
                    + str(request.seed).encode("ascii", errors="strict")
                    + b"\x00"
                    + bytes.fromhex(identity)
                ).digest(),
                identity,
            ),
        )
        for identity in ordered_local[:local_limit]:
            selected[identity] = local[identity]

        rng = random.Random(request.seed)
        maximum_attempts = request.pool_size * 128
        attempts = 0
        while len(selected) < request.pool_size and attempts < maximum_attempts:
            attempts += 1
            lineage = rng.randrange(len(_SEEDS))
            payload = _SEEDS[lineage].model_dump(mode="python")
            for locus, grid, _family in LOCUS_GRIDS:
                _set_locus(payload, locus, grid[rng.randrange(len(grid))])
            candidate = normalize_candidate(payload)
            frozen = freeze_json(candidate.model_dump(mode="python"))
            if type(frozen) is not FrozenJsonObject:  # pragma: no cover
                raise AssertionError("Heat2D candidate did not freeze to an object")
            identity = typed_json_sha256(frozen)
            if identity not in excluded:
                selected.setdefault(identity, frozen)
        if len(selected) != request.pool_size:
            raise RuntimeError("Heat2D acquisition reservoir underfilled")
        return tuple(selected.values())


__all__ = [
    "Heat2DFiniteAcquisitionSpace",
    "SPACE_DEFINITION_SHA256",
    "SPACE_ID",
    "SPACE_VERSION",
]
