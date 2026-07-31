"""Dimension-agnostic fixed-reference utility attribution.

The legacy projector scores every candidate independently against the same
frozen archive.  Those marginal gains are order invariant, but they are not
coalition efficient: redundant candidates can each receive credit for the
same archive region.  The exact Shapley projector below uses only already
evaluated objective vectors and an injected joint-utility snapshot.  It
therefore allocates the *realized joint stage gain* without another workload
evaluation, simulator call, or objective-specific branch in the framework.
"""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
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

CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_ID = (
    "fixed_reference_exact_coalition_shapley_utility"
)
CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_VERSION = 1
CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:fixed-reference-exact-coalition-shapley-utility:v1;"
    b"snapshot=authenticated-pre-generation-archive-utility;"
    b"players=ranked-itt-scored-outcomes;infeasible=zero;"
    b"coalition-value=joint-normalized-archive-gain;"
    b"credit=exact-shapley;efficiency=true;symmetry=true;dummy=true;"
    b"simulator-calls=zero;dimension-and-objective-names=opaque;"
    b"maximum-exact-players=16;floating-residual=canonical-largest-credit"
).hexdigest()

MAXIMUM_EXACT_SHAPLEY_PLAYERS = 16


@runtime_checkable
class MarginalUtilitySnapshot(Protocol):
    def marginal_gain(self, point: Mapping[str, float]) -> float: ...


@runtime_checkable
class JointUtilitySnapshot(Protocol):
    """Replay normalized archive gain for an already evaluated coalition."""

    def joint_gain(self, points: tuple[Mapping[str, float], ...]) -> float: ...


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


def exact_coalition_shapley_values(
    snapshot: JointUtilitySnapshot,
    points: tuple[Mapping[str, float], ...],
) -> tuple[float, ...]:
    """Allocate one monotone joint gain exactly over at most sixteen players.

    This workload-neutral primitive is public because residual portfolio
    stages no longer necessarily originate as legacy
    ``PortfolioVariationWaveResult`` rows.  Both the legacy projector and the
    mixed-expert stage closer must nevertheless use the identical coalition
    game and floating-residual rule.
    """

    player_count = len(points)
    if player_count == 0:
        return ()
    if player_count > MAXIMUM_EXACT_SHAPLEY_PLAYERS:
        raise ValueError(
            "exact coalition attribution supports at most "
            f"{MAXIMUM_EXACT_SHAPLEY_PLAYERS} scored candidates"
        )

    coalition_values: dict[int, float] = {0: 0.0}

    def coalition_value(mask: int) -> float:
        cached = coalition_values.get(mask)
        if cached is not None:
            return cached
        value = snapshot.joint_gain(
            tuple(points[index] for index in range(player_count) if mask >> index & 1)
        )
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("joint utility must be a finite canonical float")
        if not 0.0 <= value <= 1.0:
            raise ValueError("joint utility must lie in [0, 1]")
        coalition_values[mask] = value
        return value

    full_mask = (1 << player_count) - 1
    values: list[float] = []
    for player in range(player_count):
        others = tuple(index for index in range(player_count) if index != player)
        terms: list[float] = []
        player_bit = 1 << player
        for size in range(player_count):
            weight = 1.0 / (player_count * math.comb(player_count - 1, size))
            for members in combinations(others, size):
                mask = 0
                for member in members:
                    mask |= 1 << member
                marginal = coalition_value(mask | player_bit) - coalition_value(mask)
                if marginal < -1e-12:
                    raise ValueError("joint archive utility is not monotone")
                terms.append(weight * max(0.0, marginal))
        values.append(math.fsum(terms))

    # Shapley efficiency is exact algebraically.  Floating sweeps can leave a
    # few ulps; bind those deterministically to the largest-credit player so
    # serialized credit closes exactly to the observed joint stage gain.
    joint_gain = coalition_value(full_mask)
    residual = joint_gain - math.fsum(values)
    if abs(residual) > 1e-10:
        raise RuntimeError("Shapley credit does not close to joint archive gain")
    correction_index = min(
        range(player_count),
        key=lambda index: (-values[index], index),
    )
    values[correction_index] += residual
    if any(value < 0.0 or value > 1.0 for value in values):
        raise RuntimeError("corrected Shapley credit escapes [0, 1]")
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class ExactCoalitionShapleyContextualUtilityProjector:
    """Conservatively attribute exact joint stage utility to evaluated actions."""

    archive_utility: ReplayableArchiveUtility = field(repr=False, compare=False)
    projector_id: str = field(
        init=False,
        default=CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_ID,
    )
    projector_version: int = field(
        init=False,
        default=CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.archive_utility, ReplayableArchiveUtility):
            raise TypeError("archive_utility must replay utility snapshots")

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
        if not isinstance(replayed, JointUtilitySnapshot):
            raise TypeError("archive utility returned no joint-gain snapshot")

        # Canonical candidate identity makes attribution invariant to concurrent
        # lane completion order.  Infeasible members are retained as explicit
        # zero rows but are not players in the scored coalition.
        scored: list[tuple[str, int, int, Mapping[str, float]]] = []
        rows: list[list[float]] = []
        for result_index, result in enumerate(results):
            if type(result) is not PortfolioVariationWaveResult:
                raise TypeError("results must contain exact portfolio results")
            result.__post_init__()
            row = [0.0] * len(result.receipt.members)
            rows.append(row)
            for member_index, (member, outcome) in enumerate(
                zip(result.receipt.members, result.outcomes, strict=True)
            ):
                if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
                    continue
                candidate = outcome.candidate
                if candidate is None:
                    raise ValueError("scored outcome omitted its candidate")
                scored.append(
                    (
                        candidate.candidate_id.value,
                        result_index,
                        member_index,
                        candidate.objective_map,
                    )
                )
        scored.sort(key=lambda value: value[0])
        if len({value[0] for value in scored}) != len(scored):
            raise ValueError("one stage cannot attribute a candidate more than once")
        credits = exact_coalition_shapley_values(
            replayed,
            tuple(value[3] for value in scored),
        )
        for (_candidate_id, result_index, member_index, _point), credit in zip(
            scored,
            credits,
            strict=True,
        ):
            rows[result_index][member_index] = credit
        return tuple(tuple(row) for row in rows)


__all__ = [
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION",
    "CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_DEFINITION_SHA256",
    "CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_ID",
    "CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_VERSION",
    "ExactCoalitionShapleyContextualUtilityProjector",
    "FixedReferenceContextualMarginalUtilityProjector",
    "JointUtilitySnapshot",
    "MAXIMUM_EXACT_SHAPLEY_PLAYERS",
    "MarginalUtilitySnapshot",
    "ReplayableArchiveUtility",
    "exact_coalition_shapley_values",
]
