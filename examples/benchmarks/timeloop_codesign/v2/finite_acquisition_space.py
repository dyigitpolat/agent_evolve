"""Outcome-blind finite acquisition reservoir for Timeloop v2 co-design."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
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

from .candidate import (
    ARCHITECTURE_FIELD_GRIDS,
    DEFAULT_CANDIDATE,
    POLICY_BLOCK_FIELDS,
    POLICY_FIELD_GRIDS,
    CandidateConfig,
    normalize_candidate,
)
from .finite_variation_catalog import TimeloopV2FiniteVariationCatalog
from .network_panel import NetworkLayerPanel, panel_sha256


SPACE_ID = "timeloop_codesign_v2_finite_acquisition"
SPACE_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:timeloop-v2-finite-acquisition-space:v1\x00"
_ORDER_DOMAIN = b"agent-evolve:timeloop-v2-finite-acquisition-local-order:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _loci() -> tuple[tuple[str, tuple[object, ...], str], ...]:
    values: list[tuple[str, tuple[object, ...], str]] = list(
        ARCHITECTURE_FIELD_GRIDS
    )
    for block in POLICY_BLOCK_FIELDS:
        values.extend(
            (f"{block}.{name}", tuple(grid), family)
            for name, grid, family in POLICY_FIELD_GRIDS
        )
    return tuple(values)


LOCI = _loci()


def _read_locus(payload: dict[str, Any], locus: str) -> object:
    value: Any = payload
    for name in locus.split("."):
        if type(value) is not dict:
            raise TypeError("Timeloop locus traversed a non-object")
        value = value[name]
    return value


def _set_locus(payload: dict[str, Any], locus: str, value: object) -> None:
    cursor = payload
    path = locus.split(".")
    for name in path[:-1]:
        nested = cursor[name]
        if type(nested) is not dict:
            raise TypeError("Timeloop locus traversed a non-object")
        cursor = nested
    cursor[path[-1]] = value


@dataclass(frozen=True, slots=True)
class TimeloopV2FiniteAcquisitionSpace:
    """Expose the sealed 20-locus product space without evaluator outcomes."""

    panel: NetworkLayerPanel
    space_id: str = field(init=False, default=SPACE_ID)
    space_version: int = field(init=False, default=SPACE_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        catalog = TimeloopV2FiniteVariationCatalog(self.panel)
        definition = {
            "schema_version": 1,
            "space_id": SPACE_ID,
            "space_version": SPACE_VERSION,
            "panel_sha256": panel_sha256(self.panel),
            "catalog_definition_sha256": catalog.definition_sha256,
            "candidate_schema": CandidateConfig.model_json_schema(by_alias=False),
            "loci": [
                {"locus": locus, "values": list(values), "family": family}
                for locus, values, family in LOCI
            ],
            "features": "per-locus-grid-index-minmax-normalized",
            "reservoir": (
                "up-to-one-quarter-prior-one-locus-neighborhood-then-seeded-"
                "uniform-legal-product-fill"
            ),
            "local_order": "request-seed-and-configuration-hash-sha256",
            "outcome_access": False,
        }
        expected = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        if self.definition_sha256 not in ("", expected):
            raise ValueError("Timeloop acquisition-space identity drifted")
        object.__setattr__(self, "definition_sha256", expected)

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]:
        self.__post_init__()
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("Timeloop acquisition configuration must be frozen")
        candidate = normalize_candidate(thaw_json(configuration))
        payload = candidate.model_dump(mode="python")
        features = []
        for locus, values, _family in LOCI:
            current = _read_locus(payload, locus)
            try:
                ordinal = next(
                    index
                    for index, value in enumerate(values)
                    if type(value) is type(current) and value == current
                )
            except StopIteration as error:  # pragma: no cover - schema invariant.
                raise ValueError("Timeloop candidate escaped its sealed grid") from error
            features.append(ordinal / float(len(values) - 1))
        return tuple(features)

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
        catalog = TimeloopV2FiniteVariationCatalog(self.panel)
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
            payload = json.loads(_canonical_json(DEFAULT_CANDIDATE))
            for locus, values, _family in LOCI:
                _set_locus(payload, locus, values[rng.randrange(len(values))])
            candidate = normalize_candidate(payload)
            frozen = freeze_json(candidate.model_dump(mode="python"))
            if type(frozen) is not FrozenJsonObject:  # pragma: no cover
                raise AssertionError("Timeloop candidate did not freeze to an object")
            identity = typed_json_sha256(frozen)
            if identity not in excluded:
                selected.setdefault(identity, frozen)
        if len(selected) != request.pool_size:
            raise RuntimeError("Timeloop acquisition reservoir underfilled")
        return tuple(selected.values())


__all__ = [
    "SPACE_ID",
    "SPACE_VERSION",
    "TimeloopV2FiniteAcquisitionSpace",
]
