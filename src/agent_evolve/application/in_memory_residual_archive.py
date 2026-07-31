"""Generic in-memory archive publication for residual campaign stages.

The application core owns transactional publication, while a projection port
owns the serialized archive view consumed by an injected utility.  The default
projection is suitable for utilities—such as fixed-reference hypervolume—that
can safely recompute their frontier from every valid evaluated point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilityPort,
    freeze_archive_utility,
)
from agent_evolve.application.residual_campaign_runtime import (
    ResidualArchiveTransitionCommit,
    ResidualArchiveTransitionPreparation,
)
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioEvolutionResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json


ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_ID = (
    "all_valid_candidate_decision_ledger"
)
ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_VERSION = 1
ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:all-valid-candidate-decision-ledger:v1;"
        b"front-candidates=all-valid-points-utility-recomputes-frontier;"
        b"decision-ledger=all-evaluated-candidates;"
        b"objective-values=canonical-hex;"
        b"workload-model-provider-branches=false"
    ).hexdigest()
)
IN_MEMORY_RESIDUAL_ARCHIVE_ID = "in_memory_residual_archive"
IN_MEMORY_RESIDUAL_ARCHIVE_VERSION = 1
IN_MEMORY_RESIDUAL_ARCHIVE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:in-memory-residual-archive:v1;"
    b"state=initial-plus-committed-selected-real-candidates;"
    b"projection=injected-authenticated-port;"
    b"utility=injected-authenticated-archive-utility;"
    b"publication=prepare-commit-abort-no-io;"
    b"workload-model-provider-branches=false"
).hexdigest()


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be exact")
    candidate.__post_init__()
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "valid": candidate.valid,
        "generation": candidate.generation,
        "objectives": [
            {"metric_id": name, "value_hex": float(value).hex()}
            for name, value in candidate.objectives
        ],
    }


@runtime_checkable
class ResidualArchiveRecordProjectionPort(Protocol):
    """Serialize evaluated candidates for an archive-utility boundary."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def project(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> FrozenJsonObject: ...


@dataclass(frozen=True, slots=True)
class AllValidCandidateArchiveProjection:
    """Expose all valid points and let the utility recompute its frontier."""

    projection_id: str = ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_ID
    projection_version: int = (
        ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_VERSION
    )
    definition_sha256: str = (
        ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_DEFINITION_SHA256
    )

    def project(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> FrozenJsonObject:
        if (
            type(candidates) is not tuple
            or any(
                type(candidate) is not EvolutionCandidate
                for candidate in candidates
            )
        ):
            raise TypeError("candidates must be an exact candidate tuple")
        records = tuple(_candidate_record(value) for value in candidates)
        projected = freeze_json(
            {
                "schema_version": 1,
                "front_candidates": [
                    value for value in records if bool(value["valid"])
                ],
                "decision_ledger": list(records),
                "frontier_recomputed_by_utility": True,
            }
        )
        if type(projected) is not FrozenJsonObject:
            raise TypeError("archive projection must have an object root")
        return projected


@dataclass(slots=True)
class InMemoryResidualArchiveTransition:
    """Prepare and synchronously publish selected real candidates."""

    initial_candidates: tuple[EvolutionCandidate, ...]
    archive_utility: ArchiveUtilityPort = field(
        repr=False,
        compare=False,
    )
    benchmark: FrozenJsonObject
    projection: ResidualArchiveRecordProjectionPort = field(
        default_factory=AllValidCandidateArchiveProjection,
        repr=False,
        compare=False,
    )
    archive_id: str = IN_MEMORY_RESIDUAL_ARCHIVE_ID
    archive_version: int = IN_MEMORY_RESIDUAL_ARCHIVE_VERSION
    definition_sha256: str = (
        IN_MEMORY_RESIDUAL_ARCHIVE_DEFINITION_SHA256
    )
    candidates: tuple[EvolutionCandidate, ...] = ()
    _pending: dict[str, tuple[EvolutionCandidate, ...]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if (
            type(self.initial_candidates) is not tuple
            or not self.initial_candidates
            or any(
                type(candidate) is not EvolutionCandidate
                for candidate in self.initial_candidates
            )
        ):
            raise TypeError(
                "initial_candidates must contain exact non-empty candidates"
            )
        for candidate in self.initial_candidates:
            candidate.__post_init__()
        if (
            type(self.candidates) is not tuple
            or any(
                type(candidate) is not EvolutionCandidate
                for candidate in self.candidates
            )
        ):
            raise TypeError("candidates must be an exact candidate tuple")
        for candidate in self.candidates:
            candidate.__post_init__()
        ids = tuple(
            value.candidate_id
            for value in (*self.initial_candidates, *self.candidates)
        )
        if len(ids) != len(set(ids)):
            raise ValueError("archive candidate IDs must be globally unique")
        if not isinstance(self.archive_utility, ArchiveUtilityPort):
            raise TypeError("archive_utility must implement its runtime port")
        if (
            type(self.benchmark) is not FrozenJsonObject
            or freeze_json(self.benchmark) is not self.benchmark
        ):
            raise TypeError("benchmark must be an exact frozen object")
        if not isinstance(
            self.projection,
            ResidualArchiveRecordProjectionPort,
        ):
            raise TypeError("projection must implement its runtime port")
        if (
            type(self.projection.projection_id) is not str
            or not self.projection.projection_id
            or type(self.projection.projection_version) is not int
            or self.projection.projection_version <= 0
        ):
            raise ValueError("projection identity is malformed")
        require_sha256(
            self.projection.definition_sha256,
            "projection definition_sha256",
        )
        if type(self.archive_id) is not str or not self.archive_id:
            raise ValueError("archive_id must be non-empty")
        if type(self.archive_version) is not int or self.archive_version <= 0:
            raise ValueError("archive_version must be positive")
        require_sha256(self.definition_sha256, "definition_sha256")

    @property
    def all_candidates(self) -> tuple[EvolutionCandidate, ...]:
        return (*self.initial_candidates, *self.candidates)

    def _record(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> FrozenJsonObject:
        value = self.projection.project(candidates)
        if type(value) is not FrozenJsonObject:
            raise TypeError("archive projection returned a foreign value")
        return value

    async def prepare(
        self,
        result: ResidualPortfolioEvolutionResult,
    ) -> ResidualArchiveTransitionPreparation:
        self.__post_init__()
        if type(result) is not ResidualPortfolioEvolutionResult:
            raise TypeError("result must be exact")
        result.__post_init__()
        before = self.all_candidates
        after = (*before, *result.candidates)
        ids = tuple(value.candidate_id for value in after)
        if len(ids) != len(set(ids)):
            raise ValueError("archive transition repeats a candidate ID")
        pre_record = self._record(before)
        post_record = self._record(after)
        pre = freeze_archive_utility(
            self.archive_utility,
            benchmark=self.benchmark,
            generation=result.request.decision_index,
            archive=pre_record,
        )
        post = freeze_archive_utility(
            self.archive_utility,
            benchmark=self.benchmark,
            generation=result.request.decision_index,
            archive=post_record,
        )
        post_dynamic = (*self.candidates, *result.candidates)
        preparation = ResidualArchiveTransitionPreparation(
            archive_id=self.archive_id,
            archive_version=self.archive_version,
            archive_definition_sha256=self.definition_sha256,
            residual_result_sha256=result.result_sha256,
            pre_snapshot=pre,
            post_snapshot=post,
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "projection": {
                        "projection_id": self.projection.projection_id,
                        "projection_version": (
                            self.projection.projection_version
                        ),
                        "definition_sha256": (
                            self.projection.definition_sha256
                        ),
                    },
                    "initial_candidate_count": len(
                        self.initial_candidates
                    ),
                    "pre_dynamic_candidate_count": len(self.candidates),
                    "post_dynamic_candidate_count": len(post_dynamic),
                    "selected_only_real_evaluations": len(
                        result.candidates
                    ),
                    "publication_performed": False,
                }
            ),
        )
        self._pending[preparation.preparation_sha256] = post_dynamic
        return preparation

    def commit(
        self,
        preparation: ResidualArchiveTransitionPreparation,
    ) -> ResidualArchiveTransitionCommit:
        if type(preparation) is not ResidualArchiveTransitionPreparation:
            raise TypeError("preparation must be exact")
        preparation.__post_init__()
        try:
            candidates = self._pending.pop(
                preparation.preparation_sha256
            )
        except KeyError as error:
            raise ValueError(
                "archive preparation is absent or already closed"
            ) from error
        self.candidates = candidates
        return ResidualArchiveTransitionCommit(
            preparation_sha256=preparation.preparation_sha256,
            committed_archive_sha256=(
                preparation.post_snapshot.archive_sha256
            ),
            evidence=freeze_json(
                {
                    "published": True,
                    "initial_candidate_count": len(
                        self.initial_candidates
                    ),
                    "dynamic_candidate_count": len(candidates),
                }
            ),
        )

    def abort(
        self,
        preparation: ResidualArchiveTransitionPreparation,
    ) -> None:
        if type(preparation) is not ResidualArchiveTransitionPreparation:
            raise TypeError("preparation must be exact")
        preparation.__post_init__()
        if self._pending.pop(preparation.preparation_sha256, None) is None:
            raise ValueError(
                "archive preparation is absent or already closed"
            )


__all__ = [
    "ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_DEFINITION_SHA256",
    "ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_ID",
    "ALL_VALID_CANDIDATE_ARCHIVE_PROJECTION_VERSION",
    "IN_MEMORY_RESIDUAL_ARCHIVE_DEFINITION_SHA256",
    "IN_MEMORY_RESIDUAL_ARCHIVE_ID",
    "IN_MEMORY_RESIDUAL_ARCHIVE_VERSION",
    "AllValidCandidateArchiveProjection",
    "InMemoryResidualArchiveTransition",
    "ResidualArchiveRecordProjectionPort",
]
