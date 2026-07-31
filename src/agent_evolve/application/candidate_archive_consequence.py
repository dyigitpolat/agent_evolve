"""Generic scalar archive-geometry port for pre-evaluation consequence models."""

from __future__ import annotations

import math
import re
from typing import Mapping, Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")


@runtime_checkable
class CandidateArchiveConsequenceUtilityPort(Protocol):
    """Measure prior archive utility and one hypothetical objective point.

    The port receives only evaluated candidates and raw decision objectives.
    It may implement fixed-reference hypervolume or another workload-defined
    archive utility, while the consequence projector remains workload blind.
    """

    utility_id: str
    utility_version: int
    definition_sha256: str

    def utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> float: ...

    def marginal_utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
        objective_point: Mapping[str, float],
    ) -> float: ...


@runtime_checkable
class CandidateArchivePortfolioConsequenceUtilityPort(
    CandidateArchiveConsequenceUtilityPort,
    Protocol,
):
    """Measure the joint marginal value of hypothetical objective points.

    This optional extension preserves the inverted workload boundary while
    allowing a generic allocation policy to value Pareto-complementary sets
    rather than summing member-wise scores.  Implementations remain
    authoritative for objective senses, normalization, and reference points.
    """

    def portfolio_marginal_utility(
        self,
        candidates: tuple[EvolutionCandidate, ...],
        objective_points: tuple[Mapping[str, float], ...],
    ) -> float: ...


def validate_candidate_archive_consequence_utility(
    value: CandidateArchiveConsequenceUtilityPort,
) -> tuple[str, int, str]:
    """Validate and return an injected utility's public identity."""

    if not isinstance(value, CandidateArchiveConsequenceUtilityPort):
        raise TypeError(
            "value must implement CandidateArchiveConsequenceUtilityPort"
        )
    identity = (
        getattr(value, "utility_id", None),
        getattr(value, "utility_version", None),
        getattr(value, "definition_sha256", None),
    )
    if type(identity[0]) is not str or _TOKEN.fullmatch(identity[0]) is None:
        raise ValueError("utility_id must use the closed token grammar")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("utility_version must be positive")
    require_sha256(identity[2], "utility definition_sha256")
    return identity  # type: ignore[return-value]


def validate_candidate_archive_portfolio_consequence_utility(
    value: CandidateArchivePortfolioConsequenceUtilityPort,
) -> tuple[str, int, str]:
    """Validate the joint-set extension and return its public identity."""

    if not isinstance(
        value,
        CandidateArchivePortfolioConsequenceUtilityPort,
    ):
        raise TypeError(
            "value must implement "
            "CandidateArchivePortfolioConsequenceUtilityPort"
        )
    return validate_candidate_archive_consequence_utility(value)


def candidate_archive_utility(
    port: CandidateArchiveConsequenceUtilityPort,
    candidates: tuple[EvolutionCandidate, ...],
) -> float:
    """Call an injected utility and enforce the normalized scalar contract."""

    validate_candidate_archive_consequence_utility(port)
    if (
        type(candidates) is not tuple
        or any(
            type(candidate) is not EvolutionCandidate
            for candidate in candidates
        )
    ):
        raise TypeError("candidates must be an exact candidate tuple")
    for candidate in candidates:
        candidate.__post_init__()
    result = port.utility(candidates)
    if type(result) is not float or not math.isfinite(result) or result < 0.0:
        raise ValueError("archive utility must be finite and non-negative")
    return result


__all__ = [
    "CandidateArchiveConsequenceUtilityPort",
    "CandidateArchivePortfolioConsequenceUtilityPort",
    "candidate_archive_utility",
    "validate_candidate_archive_portfolio_consequence_utility",
    "validate_candidate_archive_consequence_utility",
]
