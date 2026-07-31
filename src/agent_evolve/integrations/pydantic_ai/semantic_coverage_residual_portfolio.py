"""Composition root for the generic semantic-coverage residual portfolio."""

from __future__ import annotations

from dataclasses import dataclass

from agent_evolve.application.action_score_authorities import (
    NativeRankMaterializedActionScorer,
    TargetEmpiricalReturnMaterializedActionScorer,
)
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchiveConsequenceUtilityPort,
    CandidateArchivePortfolioConsequenceUtilityPort,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    ForecastGeometryPortfolioMode,
    ForecastGeometryPortfolioPolicy,
    MaterializedForecastGeometryProjectionPort,
)
from agent_evolve.application.frozen_hurdle_score import (
    FrozenHurdleScoreKind,
    FrozenStandardizedLinearModel,
    MaterializedActionFeatureProjectionPort,
    WinsorizedFrozenHurdleMaterializedActionScorer,
)
from agent_evolve.application.materialized_action_broker import (
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.precommitted_portfolio_racing import (
    PortfolioRaceDisagreementPolicyPort,
    PortfolioRacePolicyBinding,
    PortfolioRacePriorProjectionPort,
    PrecommittedPortfolioRacePlanner,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
)
from agent_evolve.application.semantic_coverage_score_portfolio import (
    MaterializedActionSemanticCellProjectionPort,
    SemanticCoverageScorePortfolioPolicy,
)
from agent_evolve.application.sequential_lineage_allocation import (
    CandidateArchiveMarginalPilotOutcomeProjector,
    FrozenBranchSequentialLineagePlanner,
)
from agent_evolve.application.source_exposure_allocation import (
    MaterializedActionSourceGroupProjectionPort,
    MinimumSourceExposureAllocationPolicy,
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
TRANSPORT_SCORER_ID = "portable_cross_domain_expected_gain"
DEFAULT_LINEAGE_CAPACITY_FRACTION = 0.25
DEFAULT_COVERAGE_STRENGTH = 1.0 / 3.0
FORECAST_NEUTRAL_FEATURE_GROUP_AUTHORITIES = tuple(
    sorted(
        (
            group_id,
            0.0 if group_id == "forecast" else 1.0,
        )
        for group_id, _feature_names
        in PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS
    )
)


@dataclass(frozen=True, slots=True)
class SemanticCoverageResidualPortfolio:
    """Four score authorities and one portable semantic allocation policy."""

    primary: SupportGuardedFrozenHurdleMaterializedActionScorer
    transport: WinsorizedFrozenHurdleMaterializedActionScorer
    provider_native: NativeRankMaterializedActionScorer
    target_empirical: TargetEmpiricalReturnMaterializedActionScorer
    allocation: SemanticCoverageScorePortfolioPolicy

    def __post_init__(self) -> None:
        if (
            type(self.primary)
            is not SupportGuardedFrozenHurdleMaterializedActionScorer
            or type(self.transport)
            is not WinsorizedFrozenHurdleMaterializedActionScorer
            or type(self.provider_native)
            is not NativeRankMaterializedActionScorer
            or type(self.target_empirical)
            is not TargetEmpiricalReturnMaterializedActionScorer
            or type(self.allocation)
            is not SemanticCoverageScorePortfolioPolicy
        ):
            raise TypeError(
                "semantic coverage portfolio contains a foreign component"
            )
        self.primary.__post_init__()
        self.transport.__post_init__()
        self.provider_native.__post_init__()
        self.target_empirical.__post_init__()
        self.allocation.__post_init__()


@dataclass(frozen=True, slots=True)
class SequentialSemanticCoverageResidualPortfolio:
    """Two frozen allocation branches and their real-outcome pilot seam."""

    locked: SemanticCoverageResidualPortfolio
    unlocked: SemanticCoverageResidualPortfolio
    planner: FrozenBranchSequentialLineagePlanner
    pilot_outcome_projector: (
        CandidateArchiveMarginalPilotOutcomeProjector
    )

    def __post_init__(self) -> None:
        if type(self.locked) is not SemanticCoverageResidualPortfolio:
            raise TypeError("locked portfolio must be exact")
        if type(self.unlocked) is not SemanticCoverageResidualPortfolio:
            raise TypeError("unlocked portfolio must be exact")
        self.locked.__post_init__()
        self.unlocked.__post_init__()
        if type(self.planner) is not FrozenBranchSequentialLineagePlanner:
            raise TypeError("planner must be exact")
        self.planner.__post_init__()
        if (
            type(self.pilot_outcome_projector)
            is not CandidateArchiveMarginalPilotOutcomeProjector
        ):
            raise TypeError("pilot_outcome_projector must be exact")
        self.pilot_outcome_projector.__post_init__()
        if self.locked.allocation.allow_recursive_score_lane_spillover:
            raise ValueError("locked branch permits recursive spillover")
        if not self.unlocked.allocation.allow_recursive_score_lane_spillover:
            raise ValueError("unlocked branch forbids recursive spillover")

    @property
    def primary(self) -> SupportGuardedFrozenHurdleMaterializedActionScorer:
        """Canonical support telemetry; both branches use the same fit."""

        return self.locked.primary


@dataclass(frozen=True, slots=True)
class RacingSemanticCoverageResidualPortfolio:
    """Calibrated and forecast-neutral slates behind one generic race."""

    calibrated_locked: SemanticCoverageResidualPortfolio
    calibrated_unlocked: SemanticCoverageResidualPortfolio
    forecast_neutral_unlocked: SemanticCoverageResidualPortfolio
    planner: PrecommittedPortfolioRacePlanner
    pilot_outcome_projector: (
        CandidateArchiveMarginalPilotOutcomeProjector
    )

    def __post_init__(self) -> None:
        for value in (
            self.calibrated_locked,
            self.calibrated_unlocked,
            self.forecast_neutral_unlocked,
        ):
            if type(value) is not SemanticCoverageResidualPortfolio:
                raise TypeError("race portfolios must be exact")
            value.__post_init__()
        if type(self.planner) is not PrecommittedPortfolioRacePlanner:
            raise TypeError("race planner must be exact")
        self.planner.__post_init__()
        if (
            type(self.pilot_outcome_projector)
            is not CandidateArchiveMarginalPilotOutcomeProjector
        ):
            raise TypeError("pilot_outcome_projector must be exact")
        self.pilot_outcome_projector.__post_init__()
        if self.calibrated_locked.allocation.allow_recursive_score_lane_spillover:
            raise ValueError("calibrated locked branch permits spillover")
        if not (
            self.calibrated_unlocked.allocation.allow_recursive_score_lane_spillover
            and self.forecast_neutral_unlocked.allocation.allow_recursive_score_lane_spillover
        ):
            raise ValueError("race unlocked branches must permit spillover")


@dataclass(frozen=True, slots=True)
class ForecastGeometryArmSpec:
    """One interceptable, workload-neutral consequence-policy branch."""

    branch_id: str
    mode: ForecastGeometryPortfolioMode
    scenario_id: str
    adverse_scenario_id: str | None = None
    risk_aversion: float = 0.0
    prior_mean: float = 0.5

    def __post_init__(self) -> None:
        if type(self.branch_id) is not str or not self.branch_id:
            raise ValueError("branch_id must be non-empty")
        if type(self.mode) is not ForecastGeometryPortfolioMode:
            raise TypeError("mode must be an exact forecast geometry mode")
        if type(self.scenario_id) is not str or not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if self.adverse_scenario_id is not None and (
            type(self.adverse_scenario_id) is not str
            or not self.adverse_scenario_id
        ):
            raise ValueError("adverse_scenario_id must be non-empty or None")
        for value, name in (
            (self.risk_aversion, "risk_aversion"),
            (self.prior_mean, "prior_mean"),
        ):
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact float")
        if self.risk_aversion < 0.0:
            raise ValueError("risk_aversion must be non-negative")
        if not 0.0 <= self.prior_mean <= 1.0:
            raise ValueError("prior_mean must lie in [0, 1]")


DEFAULT_FORECAST_GEOMETRY_ARM_SPECS = (
    ForecastGeometryArmSpec(
        branch_id="forecast_favorable",
        mode=ForecastGeometryPortfolioMode.SCENARIO,
        scenario_id="favorable",
    ),
    ForecastGeometryArmSpec(
        branch_id="forecast_median",
        mode=ForecastGeometryPortfolioMode.SCENARIO,
        scenario_id="median",
    ),
    ForecastGeometryArmSpec(
        branch_id="forecast_reliability",
        mode=(
            ForecastGeometryPortfolioMode.RELIABILITY_ADJUSTED_SCENARIO
        ),
        scenario_id="median",
    ),
    ForecastGeometryArmSpec(
        branch_id="forecast_risk",
        mode=ForecastGeometryPortfolioMode.RISK_ADJUSTED_SCENARIO,
        scenario_id="median",
        adverse_scenario_id="adverse",
        risk_aversion=1.0,
    ),
)


@dataclass(frozen=True, slots=True)
class ForecastGeometryRacingResidualPortfolio:
    """Protected structural arm plus typed consequence-geometry arms."""

    forecast_neutral: SemanticCoverageResidualPortfolio
    forecast_arms: tuple[
        tuple[str, ForecastGeometryPortfolioPolicy],
        ...,
    ]
    planner: PrecommittedPortfolioRacePlanner
    pilot_outcome_projector: CandidateArchiveMarginalPilotOutcomeProjector

    def __post_init__(self) -> None:
        if type(self.forecast_neutral) is not SemanticCoverageResidualPortfolio:
            raise TypeError("forecast_neutral must be an exact portfolio")
        self.forecast_neutral.__post_init__()
        if not self.forecast_neutral.allocation.allow_recursive_score_lane_spillover:
            raise ValueError("forecast-neutral branch must permit spillover")
        if (
            type(self.forecast_arms) is not tuple
            or not self.forecast_arms
            or self.forecast_arms
            != tuple(sorted(self.forecast_arms, key=lambda value: value[0]))
        ):
            raise ValueError("forecast_arms must be non-empty and canonical")
        branch_ids = []
        for branch_id, policy in self.forecast_arms:
            if type(branch_id) is not str or not branch_id:
                raise ValueError("forecast arm branch IDs must be non-empty")
            if type(policy) is not ForecastGeometryPortfolioPolicy:
                raise TypeError("forecast arms must contain exact policies")
            policy.__post_init__()
            branch_ids.append(branch_id)
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("forecast arms repeat a branch ID")
        if type(self.planner) is not PrecommittedPortfolioRacePlanner:
            raise TypeError("planner must be exact")
        self.planner.__post_init__()
        expected_ids = tuple(sorted(("forecast_neutral", *branch_ids)))
        if tuple(
            value.branch_id for value in self.planner.branch_bindings
        ) != expected_ids:
            raise ValueError("planner branches differ from forecast arms")
        if (
            type(self.pilot_outcome_projector)
            is not CandidateArchiveMarginalPilotOutcomeProjector
        ):
            raise TypeError("pilot_outcome_projector must be exact")
        self.pilot_outcome_projector.__post_init__()


def compose_semantic_coverage_residual_portfolio(
    *,
    projection: MaterializedActionFeatureProjectionPort,
    semantic_projection: MaterializedActionSemanticCellProjectionPort,
    primary_positive_model: FrozenStandardizedLinearModel,
    primary_magnitude_model: FrozenStandardizedLinearModel,
    primary_source_fit_sha256: str,
    transport_positive_model: FrozenStandardizedLinearModel,
    transport_magnitude_model: FrozenStandardizedLinearModel,
    transport_source_fit_sha256: str,
    broker: RegretBrokeredMaterializedActionPolicy,
    lineage_capacity_fraction: float = (
        DEFAULT_LINEAGE_CAPACITY_FRACTION
    ),
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    allow_recursive_score_lane_spillover: bool = False,
    hard_feasibility: HardFeasibilityPort | None = None,
    feature_group_authorities: tuple[tuple[str, float], ...] | None = None,
) -> SemanticCoverageResidualPortfolio:
    """Compose the same four-lane mechanism for any workload or model."""

    expected_group_ids = tuple(
        sorted(
            group_id
            for group_id, _feature_names
            in PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS
        )
    )
    authorities = (
        tuple((group_id, 1.0) for group_id in expected_group_ids)
        if feature_group_authorities is None
        else feature_group_authorities
    )
    if (
        type(authorities) is not tuple
        or authorities != tuple(sorted(authorities))
        or tuple(value[0] for value in authorities) != expected_group_ids
    ):
        raise ValueError(
            "feature_group_authorities must exactly cover portable groups"
        )
    authority_by_group = dict(authorities)
    groups = tuple(
        FrozenFeatureSupportGroup(
            group_id=group_id,
            feature_names=tuple(feature_names),
            base_authority=authority_by_group[group_id],
        )
        for group_id, feature_names in sorted(
            PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS
        )
    )
    primary = SupportGuardedFrozenHurdleMaterializedActionScorer(
        scorer_id=PRIMARY_SCORER_ID,
        projection=projection,
        positive_model=primary_positive_model,
        magnitude_model=primary_magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=primary_source_fit_sha256,
        support_groups=groups,
    )
    transport = WinsorizedFrozenHurdleMaterializedActionScorer(
        scorer_id=TRANSPORT_SCORER_ID,
        projection=projection,
        positive_model=transport_positive_model,
        magnitude_model=transport_magnitude_model,
        score_kind=FrozenHurdleScoreKind.EXPECTED_POSITIVE_MAGNITUDE,
        source_fit_sha256=transport_source_fit_sha256,
        winsorization_limit=3.0,
    )
    provider_native = NativeRankMaterializedActionScorer()
    target_empirical = TargetEmpiricalReturnMaterializedActionScorer(
        broker=broker
    )
    scorers = tuple(
        sorted(
            (primary, transport, provider_native, target_empirical),
            key=lambda value: value.scorer_id,
        )
    )
    score_lane_capacity = 1.0 - lineage_capacity_fraction
    allocation = SemanticCoverageScorePortfolioPolicy(
        scorers=scorers,
        scorer_capacity_fractions=tuple(
            sorted(
                (
                    (
                        primary.scorer_id,
                        score_lane_capacity * (2.0 / 3.0),
                    ),
                    (transport.scorer_id, 0.0),
                    (
                        provider_native.scorer_id,
                        score_lane_capacity * (1.0 / 6.0),
                    ),
                    (
                        target_empirical.scorer_id,
                        score_lane_capacity * (1.0 / 6.0),
                    ),
                )
            )
        ),
        lineage_scorer_id=transport.scorer_id,
        lineage_member_scorer_id=provider_native.scorer_id,
        lineage_deficit_refill_scorer_id=primary.scorer_id,
        lineage_capacity_fraction=lineage_capacity_fraction,
        semantic_projection=semantic_projection,
        coverage_strength=coverage_strength,
        allow_recursive_score_lane_spillover=(
            allow_recursive_score_lane_spillover
        ),
        hard_feasibility=hard_feasibility,
    )
    return SemanticCoverageResidualPortfolio(
        primary=primary,
        transport=transport,
        provider_native=provider_native,
        target_empirical=target_empirical,
        allocation=allocation,
    )


def compose_sequential_semantic_coverage_residual_portfolio(
    *,
    prior_candidates: tuple[EvolutionCandidate, ...],
    archive_utility: CandidateArchiveConsequenceUtilityPort,
    projection: MaterializedActionFeatureProjectionPort,
    semantic_projection: MaterializedActionSemanticCellProjectionPort,
    primary_positive_model: FrozenStandardizedLinearModel,
    primary_magnitude_model: FrozenStandardizedLinearModel,
    primary_source_fit_sha256: str,
    transport_positive_model: FrozenStandardizedLinearModel,
    transport_magnitude_model: FrozenStandardizedLinearModel,
    transport_source_fit_sha256: str,
    broker: RegretBrokeredMaterializedActionPolicy,
    lineage_capacity_fraction: float = (
        DEFAULT_LINEAGE_CAPACITY_FRACTION
    ),
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    hard_feasibility: HardFeasibilityPort | None = None,
) -> SequentialSemanticCoverageResidualPortfolio:
    """Compose outcome-blind locked/unlocked branches for any workload."""

    common = {
        "projection": projection,
        "semantic_projection": semantic_projection,
        "primary_positive_model": primary_positive_model,
        "primary_magnitude_model": primary_magnitude_model,
        "primary_source_fit_sha256": primary_source_fit_sha256,
        "transport_positive_model": transport_positive_model,
        "transport_magnitude_model": transport_magnitude_model,
        "transport_source_fit_sha256": transport_source_fit_sha256,
        "broker": broker,
        "lineage_capacity_fraction": lineage_capacity_fraction,
        "coverage_strength": coverage_strength,
        "hard_feasibility": hard_feasibility,
    }
    locked = compose_semantic_coverage_residual_portfolio(
        **common,
        allow_recursive_score_lane_spillover=False,
    )
    unlocked = compose_semantic_coverage_residual_portfolio(
        **common,
        allow_recursive_score_lane_spillover=True,
    )
    return SequentialSemanticCoverageResidualPortfolio(
        locked=locked,
        unlocked=unlocked,
        planner=FrozenBranchSequentialLineagePlanner(
            locked_policy=locked.allocation,
            unlocked_policy=unlocked.allocation,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=prior_candidates,
                utility=archive_utility,
            )
        ),
    )


def compose_racing_semantic_coverage_residual_portfolio(
    *,
    prior_candidates: tuple[EvolutionCandidate, ...],
    archive_utility: CandidateArchiveConsequenceUtilityPort,
    projection: MaterializedActionFeatureProjectionPort,
    semantic_projection: MaterializedActionSemanticCellProjectionPort,
    primary_positive_model: FrozenStandardizedLinearModel,
    primary_magnitude_model: FrozenStandardizedLinearModel,
    primary_source_fit_sha256: str,
    transport_positive_model: FrozenStandardizedLinearModel,
    transport_magnitude_model: FrozenStandardizedLinearModel,
    transport_source_fit_sha256: str,
    broker: RegretBrokeredMaterializedActionPolicy,
    lineage_capacity_fraction: float = (
        DEFAULT_LINEAGE_CAPACITY_FRACTION
    ),
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    pilot_slots: int = 3,
    disagreement_policy: PortfolioRaceDisagreementPolicyPort | None = None,
    prior_projection: PortfolioRacePriorProjectionPort | None = None,
    source_group_projection: (
        MaterializedActionSourceGroupProjectionPort | None
    ) = None,
    minimum_source_exposures: tuple[tuple[str, int], ...] = (),
    hard_feasibility: HardFeasibilityPort | None = None,
) -> RacingSemanticCoverageResidualPortfolio:
    """Compose three portable complete slates and a branch-level pilot race."""

    common = {
        "projection": projection,
        "semantic_projection": semantic_projection,
        "primary_positive_model": primary_positive_model,
        "primary_magnitude_model": primary_magnitude_model,
        "primary_source_fit_sha256": primary_source_fit_sha256,
        "transport_positive_model": transport_positive_model,
        "transport_magnitude_model": transport_magnitude_model,
        "transport_source_fit_sha256": transport_source_fit_sha256,
        "broker": broker,
        "lineage_capacity_fraction": lineage_capacity_fraction,
        "coverage_strength": coverage_strength,
        "hard_feasibility": hard_feasibility,
    }
    calibrated_locked = compose_semantic_coverage_residual_portfolio(
        **common,
        allow_recursive_score_lane_spillover=False,
    )
    calibrated_unlocked = compose_semantic_coverage_residual_portfolio(
        **common,
        allow_recursive_score_lane_spillover=True,
    )
    forecast_neutral_unlocked = (
        compose_semantic_coverage_residual_portfolio(
            **common,
            allow_recursive_score_lane_spillover=True,
            feature_group_authorities=(
                FORECAST_NEUTRAL_FEATURE_GROUP_AUTHORITIES
            ),
        )
    )
    if (source_group_projection is None) != (
        not minimum_source_exposures
    ):
        raise ValueError(
            "source-group projection and minimum exposures must co-occur"
        )

    def allocation(
        value: SemanticCoverageResidualPortfolio,
    ):
        if source_group_projection is None:
            return value.allocation
        return MinimumSourceExposureAllocationPolicy(
            base_policy=value.allocation,
            priority_scorer=value.primary,
            source_projection=source_group_projection,
            minimum_exposures=minimum_source_exposures,
        )

    return RacingSemanticCoverageResidualPortfolio(
        calibrated_locked=calibrated_locked,
        calibrated_unlocked=calibrated_unlocked,
        forecast_neutral_unlocked=forecast_neutral_unlocked,
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="calibrated_locked",
                    policy=allocation(calibrated_locked),
                ),
                PortfolioRacePolicyBinding(
                    branch_id="calibrated_unlocked",
                    policy=allocation(calibrated_unlocked),
                ),
                PortfolioRacePolicyBinding(
                    branch_id="forecast_neutral_unlocked",
                    policy=allocation(forecast_neutral_unlocked),
                ),
            ),
            pilot_policy=None,
            pilot_slots=pilot_slots,
            disagreement_policy=disagreement_policy,
            prior_projection=prior_projection,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=prior_candidates,
                utility=archive_utility,
            )
        ),
    )


def compose_forecast_geometry_racing_residual_portfolio(
    *,
    prior_candidates: tuple[EvolutionCandidate, ...],
    archive_utility: CandidateArchivePortfolioConsequenceUtilityPort,
    forecast_geometry_projection: (
        MaterializedForecastGeometryProjectionPort
    ),
    projection: MaterializedActionFeatureProjectionPort,
    semantic_projection: MaterializedActionSemanticCellProjectionPort,
    primary_positive_model: FrozenStandardizedLinearModel,
    primary_magnitude_model: FrozenStandardizedLinearModel,
    primary_source_fit_sha256: str,
    transport_positive_model: FrozenStandardizedLinearModel,
    transport_magnitude_model: FrozenStandardizedLinearModel,
    transport_source_fit_sha256: str,
    broker: RegretBrokeredMaterializedActionPolicy,
    lineage_capacity_fraction: float = (
        DEFAULT_LINEAGE_CAPACITY_FRACTION
    ),
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    pilot_slots: int = 3,
    disagreement_policy: PortfolioRaceDisagreementPolicyPort | None = None,
    prior_projection: PortfolioRacePriorProjectionPort | None = None,
    arm_specs: tuple[ForecastGeometryArmSpec, ...] = (
        DEFAULT_FORECAST_GEOMETRY_ARM_SPECS
    ),
    forecast_neutral_prior_mean: float = 0.5,
    max_exact_reliability_members: int = 8,
    source_group_projection: (
        MaterializedActionSourceGroupProjectionPort | None
    ) = None,
    minimum_source_exposures: tuple[tuple[str, int], ...] = (),
    hard_feasibility: HardFeasibilityPort | None = None,
) -> ForecastGeometryRacingResidualPortfolio:
    """Compose protected scenario arms behind the same generic race.

    Workload adapters supply only ordinary objective senses through the
    forecast-geometry projection and the authoritative joint archive utility.
    Arm definitions, source floors, pilot accounting, and current-outcome
    exclusion remain inside reusable application policies.
    """

    if (
        type(arm_specs) is not tuple
        or not arm_specs
        or any(type(value) is not ForecastGeometryArmSpec for value in arm_specs)
    ):
        raise TypeError("arm_specs must contain exact non-empty specs")
    for value in arm_specs:
        value.__post_init__()
    branch_ids = tuple(value.branch_id for value in arm_specs)
    if (
        branch_ids != tuple(sorted(set(branch_ids)))
        or "forecast_neutral" in branch_ids
    ):
        raise ValueError(
            "arm_specs must be canonical, unique, and reserve forecast_neutral"
        )
    if (
        type(forecast_neutral_prior_mean) is not float
        or not 0.0 <= forecast_neutral_prior_mean <= 1.0
    ):
        raise ValueError("forecast_neutral_prior_mean must lie in [0, 1]")
    if (source_group_projection is None) != (
        not minimum_source_exposures
    ):
        raise ValueError(
            "source-group projection and minimum exposures must co-occur"
        )

    forecast_neutral = compose_semantic_coverage_residual_portfolio(
        projection=projection,
        semantic_projection=semantic_projection,
        primary_positive_model=primary_positive_model,
        primary_magnitude_model=primary_magnitude_model,
        primary_source_fit_sha256=primary_source_fit_sha256,
        transport_positive_model=transport_positive_model,
        transport_magnitude_model=transport_magnitude_model,
        transport_source_fit_sha256=transport_source_fit_sha256,
        broker=broker,
        lineage_capacity_fraction=lineage_capacity_fraction,
        coverage_strength=coverage_strength,
        allow_recursive_score_lane_spillover=True,
        hard_feasibility=hard_feasibility,
        feature_group_authorities=(
            FORECAST_NEUTRAL_FEATURE_GROUP_AUTHORITIES
        ),
    )
    forecast_arms = tuple(
        (
            spec.branch_id,
            ForecastGeometryPortfolioPolicy(
                prior_candidates=prior_candidates,
                projection=forecast_geometry_projection,
                archive_utility=archive_utility,
                mode=spec.mode,
                scenario_id=spec.scenario_id,
                adverse_scenario_id=spec.adverse_scenario_id,
                risk_aversion=spec.risk_aversion,
                max_exact_reliability_members=(
                    max_exact_reliability_members
                ),
                hard_feasibility=hard_feasibility,
            ),
        )
        for spec in arm_specs
    )

    def with_source_floor(
        policy: MaterializedActionAllocationPolicyPort,
    ) -> MaterializedActionAllocationPolicyPort:
        if source_group_projection is None:
            return policy
        return MinimumSourceExposureAllocationPolicy(
            base_policy=policy,
            priority_scorer=forecast_neutral.primary,
            source_projection=source_group_projection,
            minimum_exposures=minimum_source_exposures,
        )

    prior_by_branch = {
        value.branch_id: value.prior_mean for value in arm_specs
    }
    bindings = [
        PortfolioRacePolicyBinding(
            branch_id="forecast_neutral",
            policy=with_source_floor(forecast_neutral.allocation),
            prior_mean=forecast_neutral_prior_mean,
        )
    ]
    bindings.extend(
        PortfolioRacePolicyBinding(
            branch_id=branch_id,
            policy=with_source_floor(policy),
            prior_mean=prior_by_branch[branch_id],
        )
        for branch_id, policy in forecast_arms
    )
    planner = PrecommittedPortfolioRacePlanner(
        branch_bindings=tuple(
            sorted(bindings, key=lambda value: value.branch_id)
        ),
        pilot_policy=None,
        pilot_slots=pilot_slots,
        disagreement_policy=disagreement_policy,
        prior_projection=prior_projection,
    )
    return ForecastGeometryRacingResidualPortfolio(
        forecast_neutral=forecast_neutral,
        forecast_arms=forecast_arms,
        planner=planner,
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=prior_candidates,
                utility=archive_utility,
            )
        ),
    )


__all__ = [
    "DEFAULT_FORECAST_GEOMETRY_ARM_SPECS",
    "DEFAULT_COVERAGE_STRENGTH",
    "DEFAULT_LINEAGE_CAPACITY_FRACTION",
    "FORECAST_NEUTRAL_FEATURE_GROUP_AUTHORITIES",
    "ForecastGeometryArmSpec",
    "ForecastGeometryRacingResidualPortfolio",
    "PRIMARY_SCORER_ID",
    "TRANSPORT_SCORER_ID",
    "RacingSemanticCoverageResidualPortfolio",
    "SemanticCoverageResidualPortfolio",
    "SequentialSemanticCoverageResidualPortfolio",
    "compose_forecast_geometry_racing_residual_portfolio",
    "compose_racing_semantic_coverage_residual_portfolio",
    "compose_semantic_coverage_residual_portfolio",
    "compose_sequential_semantic_coverage_residual_portfolio",
]
