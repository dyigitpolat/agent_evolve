"""Workload-neutral projection port for authenticated archive geometry.

The campaign runtime owns when a projection is attached.  Implementations own
how an already-frozen archive utility and an already-evaluated parent are
translated into a bounded model-visible payload.  The port cannot inspect a
model/provider profile or any current/future candidate outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

if TYPE_CHECKING:
    from agent_evolve.application.agentic_evolution import EvolutionCandidate
    from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PROJECTION_DOMAIN = b"agent-evolve:campaign-archive-context-projection:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CampaignPortfolioArchiveContextProjection:
    """One exact cutoff-valid archive projection for one parent decision."""

    projector_id: str
    projector_version: int
    definition_sha256: str
    archive_utility_snapshot_sha256: str
    parent_configuration_sha256: str
    payload: FrozenJsonObject
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.projector_id) is not str or _TOKEN.fullmatch(
            self.projector_id
        ) is None:
            raise ValueError("projector_id must use the closed token grammar")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be a positive exact integer")
        _require_sha256(self.definition_sha256, "definition_sha256")
        _require_sha256(
            self.archive_utility_snapshot_sha256,
            "archive_utility_snapshot_sha256",
        )
        _require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        if (
            type(self.payload) is not FrozenJsonObject
            or freeze_json(self.payload) is not self.payload
        ):
            raise TypeError("payload must be an exact frozen typed-JSON object")
        object.__setattr__(
            self,
            "projection_sha256",
            hashlib.sha256(
                _PROJECTION_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projector": {
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "definition_sha256": self.definition_sha256,
            },
            "archive_utility_snapshot_sha256": (
                self.archive_utility_snapshot_sha256
            ),
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "payload_sha256": typed_json_sha256(self.payload),
            "payload": thaw_json(self.payload),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "projection_sha256": self.projection_sha256,
        }


@runtime_checkable
class CampaignPortfolioArchiveContextProjector(Protocol):
    """Project only frozen archive state and one evaluated parent."""

    projector_id: str
    projector_version: int
    definition_sha256: str

    def project(
        self,
        *,
        archive_utility: ArchiveUtilitySnapshot,
        parent: EvolutionCandidate,
    ) -> CampaignPortfolioArchiveContextProjection: ...


__all__ = [
    "CampaignPortfolioArchiveContextProjection",
    "CampaignPortfolioArchiveContextProjector",
]
