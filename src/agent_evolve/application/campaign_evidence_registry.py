"""Transactional registry for authenticated campaign hypothesis evidence.

The global falsification policy consumes immutable registry snapshots, while a
campaign stage is published only after candidate evaluation, memory credit,
outcome feedback, and lifecycle preparation have all succeeded.  This module
bridges those requirements: observations are validated and sealed into a
prospective snapshot during the fallible prepare phase, then appended by a
synchronous commit that performs no I/O.

The registry is deliberately ignorant of benchmark configurations, objective
semantics, LLM providers, and insight text.  Workload/application adapters are
responsible for constructing :class:`AuthenticatedHypothesisObservation`
values; this component owns only append order, uniqueness, and snapshot
identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    GlobalEvidenceRegistrySnapshot,
)


_PREPARATION_DOMAIN = b"agent-evolve:campaign-evidence-append-preparation:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(value: object) -> str:
    return hashlib.sha256(_PREPARATION_DOMAIN + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignEvidenceAppendPreparation:
    """Pure append proposal bound to the exact before/after registry states."""

    prior_snapshot_sha256: str
    prior_observation_count: int
    observations: tuple[AuthenticatedHypothesisObservation, ...]
    prospective_snapshot: GlobalEvidenceRegistrySnapshot
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.prior_snapshot_sha256, "prior_snapshot_sha256")
        if (
            type(self.prior_observation_count) is not int
            or self.prior_observation_count < 0
        ):
            raise ValueError("prior_observation_count must be non-negative")
        if type(self.observations) is not tuple or any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in self.observations
        ):
            raise TypeError(
                "observations must contain exact authenticated observations"
            )
        for value in self.observations:
            AuthenticatedHypothesisObservation.__post_init__(value)
        source_ids = tuple(value.source_evidence_id for value in self.observations)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("observations must use canonical unique source IDs")
        if type(self.prospective_snapshot) is not GlobalEvidenceRegistrySnapshot:
            raise TypeError("prospective_snapshot must be exact")
        GlobalEvidenceRegistrySnapshot.__post_init__(self.prospective_snapshot)
        prospective = self.prospective_snapshot.observations
        if len(prospective) != self.prior_observation_count + len(self.observations):
            raise ValueError("prospective snapshot does not close the append count")
        if self.observations and (
            tuple(value.source_evidence_id for value in prospective)[
                -len(self.observations) :
            ]
            != source_ids
        ):
            # The registry snapshot is globally sorted by source ID, so the new
            # observations need not be a physical suffix.  Compare sets below;
            # this fast path merely avoids accepting a malformed count join.
            prospective_ids = {value.source_evidence_id for value in prospective}
            if not set(source_ids).issubset(prospective_ids):
                raise ValueError("prospective snapshot omitted appended observations")
        object.__setattr__(
            self,
            "preparation_sha256",
            _hash(self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "prior_snapshot_sha256": self.prior_snapshot_sha256,
            "prior_observation_count": self.prior_observation_count,
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
            "prospective_snapshot_sha256": self.prospective_snapshot.snapshot_sha256,
            "prospective_captured_through_event_index": (
                self.prospective_snapshot.captured_through_event_index
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "preparation_sha256": self.preparation_sha256,
        }


@dataclass(slots=True)
class CampaignEvidenceRegistry:
    """Append-only in-memory registry with prepare/commit/abort publication."""

    _observations: list[AuthenticatedHypothesisObservation] = field(
        init=False,
        default_factory=list,
    )
    _captured_through_event_index: int = field(init=False, default=0)
    _prepared: dict[str, CampaignEvidenceAppendPreparation] = field(
        init=False,
        default_factory=dict,
    )

    @property
    def observations(self) -> tuple[AuthenticatedHypothesisObservation, ...]:
        return tuple(self._observations)

    @property
    def captured_through_event_index(self) -> int:
        return self._captured_through_event_index

    def snapshot(
        self,
        *,
        captured_through_event_index: int | None = None,
    ) -> GlobalEvidenceRegistrySnapshot:
        cutoff = (
            self.captured_through_event_index
            if captured_through_event_index is None
            else captured_through_event_index
        )
        if type(cutoff) is not int or cutoff < self.captured_through_event_index:
            raise ValueError("snapshot cutoff cannot precede committed evidence")
        return GlobalEvidenceRegistrySnapshot.seal(
            captured_through_event_index=cutoff,
            observations=self._observations,
        )

    def prepare_append(
        self,
        observations: tuple[AuthenticatedHypothesisObservation, ...],
        *,
        captured_through_event_index: int | None = None,
    ) -> CampaignEvidenceAppendPreparation:
        if type(observations) is not tuple or any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in observations
        ):
            raise TypeError(
                "observations must contain exact authenticated observations"
            )
        for value in observations:
            AuthenticatedHypothesisObservation.__post_init__(value)
        canonical = tuple(
            sorted(observations, key=lambda value: value.source_evidence_id)
        )
        source_ids = tuple(value.source_evidence_id for value in canonical)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("one append cannot repeat a source evidence ID")
        committed_ids = {value.source_evidence_id for value in self._observations}
        if committed_ids.intersection(source_ids):
            raise ValueError("source evidence ID is already committed")
        prior_cutoff = self.captured_through_event_index
        if any(value.event_index <= prior_cutoff for value in canonical):
            raise ValueError("new observations must follow the committed event cutoff")
        inferred_cutoff = max(
            (value.event_index for value in canonical),
            default=prior_cutoff,
        )
        cutoff = (
            inferred_cutoff
            if captured_through_event_index is None
            else captured_through_event_index
        )
        if type(cutoff) is not int or cutoff < inferred_cutoff:
            raise ValueError("prospective cutoff does not cover appended evidence")
        if not canonical and captured_through_event_index is None:
            raise ValueError(
                "an empty evidence append requires an explicit event cutoff"
            )
        if not canonical and cutoff <= prior_cutoff:
            raise ValueError(
                "an empty evidence append must advance the committed event cutoff"
            )
        prior = self.snapshot()
        prospective = GlobalEvidenceRegistrySnapshot.seal(
            captured_through_event_index=cutoff,
            observations=(*self._observations, *canonical),
        )
        preparation = CampaignEvidenceAppendPreparation(
            prior_snapshot_sha256=prior.snapshot_sha256,
            prior_observation_count=len(self._observations),
            observations=canonical,
            prospective_snapshot=prospective,
        )
        if preparation.preparation_sha256 in self._prepared:
            raise ValueError("evidence append preparation already exists")
        self._prepared[preparation.preparation_sha256] = preparation
        return preparation

    def commit_append(
        self,
        preparation: CampaignEvidenceAppendPreparation,
    ) -> GlobalEvidenceRegistrySnapshot:
        if type(preparation) is not CampaignEvidenceAppendPreparation:
            raise TypeError("preparation must be exact")
        CampaignEvidenceAppendPreparation.__post_init__(preparation)
        stored = self._prepared.get(preparation.preparation_sha256)
        if stored is not preparation:
            raise ValueError("evidence append preparation is unavailable")
        current = self.snapshot()
        if (
            current.snapshot_sha256 != preparation.prior_snapshot_sha256
            or len(self._observations) != preparation.prior_observation_count
        ):
            raise RuntimeError("evidence registry changed after preparation")
        self._observations.extend(preparation.observations)
        self._observations.sort(key=lambda value: value.source_evidence_id)
        self._captured_through_event_index = (
            preparation.prospective_snapshot.captured_through_event_index
        )
        committed = self.snapshot()
        if (
            committed.snapshot_sha256
            != preparation.prospective_snapshot.snapshot_sha256
        ):
            raise RuntimeError("committed evidence differs from preparation")
        del self._prepared[preparation.preparation_sha256]
        return committed

    def abort_append(self, preparation: CampaignEvidenceAppendPreparation) -> None:
        if type(preparation) is not CampaignEvidenceAppendPreparation:
            raise TypeError("preparation must be exact")
        self._prepared.pop(preparation.preparation_sha256, None)


__all__ = [
    "CampaignEvidenceAppendPreparation",
    "CampaignEvidenceRegistry",
]
