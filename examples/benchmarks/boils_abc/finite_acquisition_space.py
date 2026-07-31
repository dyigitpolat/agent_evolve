"""Outcome-blind finite acquisition reservoir for BOiLS/ABC.

The adapter exposes only legal typed sequences and normalized configuration
features.  It cannot observe objectives.  A deterministic reservoir combines
one-action neighborhoods of prior real configurations with seeded global
sequence coverage, allowing any finite acquisition policy to use the same
workload-neutral ask/tell port as the other benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.finite_acquisition_space import (
    FiniteAcquisitionSpaceRequest,
)

from .actions import ACTION_IDS, SEQUENCE_LENGTH, CandidateConfig, normalize_candidate
from .finite_variation_catalog import BoilsFiniteVariationCatalog


SPACE_ID = "boils_abc_finite_acquisition"
SPACE_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:boils-abc-finite-acquisition-space:v1\x00"
_ORDER_DOMAIN = b"agent-evolve:boils-abc-finite-acquisition-local-order:v1\x00"
_ACTION_INDEX = {value: ordinal for ordinal, value in enumerate(ACTION_IDS)}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


_DEFINITION = {
    "schema_version": 1,
    "space_id": SPACE_ID,
    "space_version": SPACE_VERSION,
    "sequence_length": SEQUENCE_LENGTH,
    "action_ids": list(ACTION_IDS),
    "features": "per-position-allowlist-index-minmax-normalized",
    "reservoir": (
        "up-to-one-quarter-prior-one-action-neighborhood-then-seeded-"
        "uniform-legal-sequence-fill"
    ),
    "local_order": "request-seed-and-configuration-hash-sha256",
    "outcome_access": False,
}
SPACE_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN + _canonical_json(_DEFINITION)
).hexdigest()


def _frozen_sequence(sequence: tuple[str, ...]) -> FrozenJsonObject:
    candidate = CandidateConfig.model_validate(
        {"sequence": list(sequence)},
        strict=True,
        by_alias=False,
        by_name=True,
    )
    frozen = freeze_json(candidate.model_dump(mode="python"))
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - schema invariant.
        raise AssertionError("BOiLS candidate did not freeze to an object")
    return frozen


@dataclass(frozen=True, slots=True)
class BoilsFiniteAcquisitionSpace:
    """Map the sealed length-20 categorical space to a finite reservoir."""

    space_id: str = SPACE_ID
    space_version: int = SPACE_VERSION
    definition_sha256: str = SPACE_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.space_id != SPACE_ID
            or self.space_version != SPACE_VERSION
            or self.definition_sha256 != SPACE_DEFINITION_SHA256
        ):
            raise ValueError("BOiLS acquisition-space identity drifted")

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]:
        self.__post_init__()
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("BOiLS acquisition configuration must be frozen")
        sequence = normalize_candidate(thaw_json(configuration))
        denominator = float(len(ACTION_IDS) - 1)
        return tuple(_ACTION_INDEX[value] / denominator for value in sequence)

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
        catalog = BoilsFiniteVariationCatalog()
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
            frozen = _frozen_sequence(
                tuple(
                    ACTION_IDS[rng.randrange(len(ACTION_IDS))]
                    for _ in range(SEQUENCE_LENGTH)
                )
            )
            identity = typed_json_sha256(frozen)
            if identity not in excluded:
                selected.setdefault(identity, frozen)
        if len(selected) != request.pool_size:
            raise RuntimeError("BOiLS acquisition reservoir underfilled")
        return tuple(selected.values())


__all__ = [
    "BoilsFiniteAcquisitionSpace",
    "SPACE_DEFINITION_SHA256",
    "SPACE_ID",
    "SPACE_VERSION",
]
