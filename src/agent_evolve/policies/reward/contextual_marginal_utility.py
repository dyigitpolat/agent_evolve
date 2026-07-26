"""Dimension-agnostic fixed-reference marginal utility projection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemberDisposition,
    PortfolioVariationWaveResult,
)


CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID = (
    "fixed_reference_candidate_marginal_utility"
)
CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION = 1
CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:fixed-reference-candidate-marginal-utility:v1;"
    b"snapshot=authenticated-pre-generation-archive-utility;"
    b"candidate=ranked-itt-scored-outcome;infeasible=zero;"
    b"dimension-and-objective-names=opaque;negative-clamp=zero"
).hexdigest()


@runtime_checkable
class MarginalUtilitySnapshot(Protocol):
    def marginal_gain(self, point: Mapping[str, float]) -> float: ...


@runtime_checkable
class ReplayableArchiveUtility(Protocol):
    def require_snapshot(
        self,
        value: ArchiveUtilitySnapshot,
    ) -> MarginalUtilitySnapshot: ...


@dataclass(frozen=True, slots=True)
class FixedReferenceContextualMarginalUtilityProjector:
    """Project ranked evaluated members through an injected utility snapshot."""

    archive_utility: ReplayableArchiveUtility = field(repr=False, compare=False)
    projector_id: str = field(
        init=False,
        default=CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID,
    )
    projector_version: int = field(
        init=False,
        default=CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.archive_utility, ReplayableArchiveUtility):
            raise TypeError("archive_utility must replay marginal-gain snapshots")
        if (
            self.projector_id != CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID
            or self.projector_version
            != CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION
            or self.definition_sha256
            != CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256
        ):
            raise ValueError("unsupported contextual marginal utility projector")

    def project(
        self,
        *,
        snapshot: ArchiveUtilitySnapshot,
        results: tuple[PortfolioVariationWaveResult, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.__post_init__()
        if type(snapshot) is not ArchiveUtilitySnapshot:
            raise TypeError("snapshot must be exact ArchiveUtilitySnapshot")
        snapshot.__post_init__()
        if type(results) is not tuple or not results:
            raise ValueError("results must be a non-empty exact tuple")
        replayed = self.archive_utility.require_snapshot(snapshot)
        if not isinstance(replayed, MarginalUtilitySnapshot):
            raise TypeError("archive utility returned no marginal-gain snapshot")
        projected: list[tuple[float, ...]] = []
        for result in results:
            if type(result) is not PortfolioVariationWaveResult:
                raise TypeError("results must contain exact portfolio results")
            result.__post_init__()
            values: list[float] = []
            for member, outcome in zip(
                result.receipt.members,
                result.outcomes,
                strict=True,
            ):
                if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
                    values.append(0.0)
                    continue
                candidate = outcome.candidate
                if candidate is None:
                    raise ValueError("scored outcome omitted its candidate")
                utility = replayed.marginal_gain(candidate.objective_map)
                if type(utility) is not float or not math.isfinite(utility):
                    raise TypeError("marginal utility must be a finite canonical float")
                normalized = max(0.0, utility)
                if normalized > 1.0:
                    raise ValueError(
                        "archive utility must expose normalized marginal gain in [0, 1]"
                    )
                values.append(normalized)
            projected.append(tuple(values))
        return tuple(projected)


__all__ = [
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION",
    "FixedReferenceContextualMarginalUtilityProjector",
    "MarginalUtilitySnapshot",
    "ReplayableArchiveUtility",
]
