"""One workload-neutral composition root for the adaptive residual portfolio."""

from __future__ import annotations

from dataclasses import dataclass

from agent_evolve.application.action_score_authorities import (
    NativeRankMaterializedActionScorer,
    TargetEmpiricalReturnMaterializedActionScorer,
)
from agent_evolve.application.frozen_hurdle_score import (
    FrozenHurdleScoreKind,
    FrozenStandardizedLinearModel,
    MaterializedActionFeatureProjectionPort,
)
from agent_evolve.application.materialized_action_broker import (
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.prequential_score_portfolio import (
    ReliabilityAdaptiveScorePortfolioPolicy,
)
from agent_evolve.application.support_guarded_hurdle_score import (
    FrozenFeatureSupportGroup,
    SupportGuardedFrozenHurdleMaterializedActionScorer,
)
from agent_evolve.ports.hard_feasibility import HardFeasibilityPort

from .portable_residual_consequence_features import (
    PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS,
)


PRIMARY_SCORER_ID = "portable_positive_probability"
RELIABILITY_COMPONENT_IDS = (
    "forecast",
    "parent",
    "proposal_structure",
)


@dataclass(frozen=True, slots=True)
class SupportAdaptiveResidualPortfolio:
    """Authenticated scorer bundle and its prequential allocation policy."""

    primary: SupportGuardedFrozenHurdleMaterializedActionScorer
    provider_native: NativeRankMaterializedActionScorer
    target_empirical: TargetEmpiricalReturnMaterializedActionScorer
    allocation: ReliabilityAdaptiveScorePortfolioPolicy

    def __post_init__(self) -> None:
        if (
            type(self.primary)
            is not SupportGuardedFrozenHurdleMaterializedActionScorer
            or type(self.provider_native)
            is not NativeRankMaterializedActionScorer
            or type(self.target_empirical)
            is not TargetEmpiricalReturnMaterializedActionScorer
            or type(self.allocation)
            is not ReliabilityAdaptiveScorePortfolioPolicy
        ):
            raise TypeError(
                "support-adaptive portfolio contains a foreign component"
            )
        self.primary.__post_init__()
        self.provider_native.__post_init__()
        self.target_empirical.__post_init__()
        self.allocation.__post_init__()
        if self.allocation.primary_reliability is not self.primary:
            raise ValueError(
                "allocation reliability must be the primary scorer instance"
            )


def compose_support_adaptive_residual_portfolio(
    *,
    projection: MaterializedActionFeatureProjectionPort,
    positive_model: FrozenStandardizedLinearModel,
    magnitude_model: FrozenStandardizedLinearModel,
    source_fit_sha256: str,
    broker: RegretBrokeredMaterializedActionPolicy,
    minimum_primary_fraction: float = 0.5,
    hard_feasibility: HardFeasibilityPort | None = None,
) -> SupportAdaptiveResidualPortfolio:
    """Compose the same outcome-blind authority market for any workload."""

    groups = tuple(
        FrozenFeatureSupportGroup(
            group_id=group_id,
            feature_names=tuple(feature_names),
        )
        for group_id, feature_names in sorted(
            PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS
        )
    )
    primary = SupportGuardedFrozenHurdleMaterializedActionScorer(
        scorer_id=PRIMARY_SCORER_ID,
        projection=projection,
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=source_fit_sha256,
        support_groups=groups,
    )
    provider_native = NativeRankMaterializedActionScorer()
    target_empirical = TargetEmpiricalReturnMaterializedActionScorer(
        broker=broker
    )
    scorers = tuple(
        sorted(
            (primary, provider_native, target_empirical),
            key=lambda value: value.scorer_id,
        )
    )
    allocation = ReliabilityAdaptiveScorePortfolioPolicy(
        scorers=scorers,
        primary_scorer_id=primary.scorer_id,
        primary_reliability=primary,
        reliability_component_ids=RELIABILITY_COMPONENT_IDS,
        minimum_primary_fraction=minimum_primary_fraction,
        hard_feasibility=hard_feasibility,
    )
    return SupportAdaptiveResidualPortfolio(
        primary=primary,
        provider_native=provider_native,
        target_empirical=target_empirical,
        allocation=allocation,
    )


__all__ = [
    "PRIMARY_SCORER_ID",
    "RELIABILITY_COMPONENT_IDS",
    "SupportAdaptiveResidualPortfolio",
    "compose_support_adaptive_residual_portfolio",
]
