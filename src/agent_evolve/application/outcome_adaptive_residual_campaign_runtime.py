"""Transactional campaign adapter for outcome-adaptive residual evolution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionExplorationPort,
    MaterializedActionReturnValuePort,
    MaterializedSlateFeasibilityPort,
    MaterializedSlateValuePort,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    OutcomeAdaptiveActionRacingPolicy,
)
from agent_evolve.application.outcome_adaptive_residual_portfolio_evolution import (
    AdaptiveActionMarketProjectorPort,
    AdaptiveActionOutcomeProjectorPort,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256,
    OutcomeAdaptiveResidualPhaseCommitPort,
    OutcomeAdaptiveResidualPortfolioEvolution,
    OutcomeAdaptiveResidualPortfolioEvolutionResult,
)
from agent_evolve.application.protected_current_prefix_forecast_opportunity import (
    ProtectedCurrentPrefixForecastOpportunityChallenger,
)
from agent_evolve.application.same_prefix_paired_audit import (
    FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID,
    ForecastOpportunitySamePrefixShadowDesignerPort,
    SamePrefixPairedAuditDesignerPort,
)
from agent_evolve.application.prequential_residual_exploration import (
    PrequentialLowDiscrepancyResidualExploration,
)
from agent_evolve.application.residual_campaign_runtime import (
    ResidualArchiveTransitionPort,
    ResidualCampaignStageReceipt,
    commit_residual_campaign_stage,
)
from agent_evolve.application.residual_learning_transaction import (
    TransactionalResidualLearningStore,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
    MaterializedActionProposalExpertPort,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256


OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_ID = (
    "transactional_outcome_adaptive_residual_portfolio_campaign"
)
OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_VERSION = 2
OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:transactional-outcome-adaptive-residual-campaign:v2;"
    b"search=sealed-market-diagnostic-sequential-racing;"
    b"diagnostic-allocation=optional-outcome-blind-required-action-policy;"
    b"publication=shared-generic-residual-archive-learning-transaction;"
    b"broker-cutoff=prior-learning-state-only;"
    b"phase-interception=injected-optional-durable-commit-port;"
    b"workload-objective-model-provider-prompt-config-branches=false"
).hexdigest()
OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_VERSION = 3
OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:transactional-outcome-adaptive-residual-campaign:v3;"
        b"base-definition="
        + OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256.encode(
            "ascii"
        )
        + b";paired-audit=optional-same-prefix-distinct-arm-assay;"
        b"publication=predeclared-authoritative-arm-only;"
        b"physical-extra-evaluations=explicit;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_VERSION = 4
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:transactional-outcome-adaptive-residual-campaign:v4;"
        b"base-definition="
        + OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256.encode(
            "ascii"
        )
        + b";evolution-definition="
        + OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256.encode(
            "ascii"
        )
        + b";continuation=protected-current-prefix-forecast-opportunity;"
        b"publication=precommitted-authoritative-k-only;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_VERSION = 5
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:transactional-outcome-adaptive-residual-campaign:v5;"
        b"base-definition="
        + (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256
        ).encode("ascii")
        + b";evolution-definition="
        + (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256
        ).encode("ascii")
        + b";shadow=final-step-same-prefix-challenger-vs-fallback;"
        b"publication=precommitted-authoritative-k-only;"
        b"calibration-observation=authenticated-causal-memory;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_VERSION = 6
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:transactional-outcome-adaptive-residual-campaign:v6;"
        b"base-definition="
        + (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256
        ).encode("ascii")
        + b";evolution-definition="
        + (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256
        ).encode("ascii")
        + b";audit=hash-rotated-same-prefix-forecast-stratum;"
        b"counterfactual-quarantine=enforced;"
        b"publication=precommitted-authoritative-k-only;"
        b"calibration-observation=paired-role-and-propensity-authenticated;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)

_RECEIPT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-campaign-stage-receipt:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualCampaignStageReceipt:
    """Join one adaptive trace to its one archive/learning transaction."""

    adaptive_result: OutcomeAdaptiveResidualPortfolioEvolutionResult
    campaign_stage: ResidualCampaignStageReceipt
    runtime_definition_sha256: str = (
        OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256
    )
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.adaptive_result)
            is not OutcomeAdaptiveResidualPortfolioEvolutionResult
        ):
            raise TypeError("adaptive_result must be exact")
        self.adaptive_result.__post_init__()
        if type(self.campaign_stage) is not ResidualCampaignStageReceipt:
            raise TypeError("campaign_stage must be exact")
        self.campaign_stage.__post_init__()
        require_sha256(
            self.runtime_definition_sha256,
            "runtime_definition_sha256",
        )
        has_forecast = (
            self.adaptive_result
            .forecast_opportunity_challenger_definition_sha256
            is not None
        )
        has_paired_assay = bool(
            self.adaptive_result.paired_audit_executions
        )
        has_forecast_stratified_audit = any(
            value.plan.designer_id
            == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            for value in self.adaptive_result.paired_audit_executions
        )
        expected_runtime_definition_sha256 = (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256
            if has_forecast_stratified_audit
            else (
                OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                if has_forecast and has_paired_assay
                else (
                    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                    if has_forecast
                    else (
                        OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                        if has_paired_assay
                        else OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                    )
                )
            )
        )
        if (
            self.runtime_definition_sha256
            != expected_runtime_definition_sha256
        ):
            raise ValueError(
                "runtime definition differs from paired-audit execution mode"
            )
        if (
            self.adaptive_result.result.result_sha256
            != self.campaign_stage.result.result_sha256
        ):
            raise ValueError(
                "adaptive trace and campaign transaction name "
                "different results"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        has_forecast = (
            self.adaptive_result
            .forecast_opportunity_challenger_definition_sha256
            is not None
        )
        has_paired_assay = bool(
            self.adaptive_result.paired_audit_executions
        )
        has_forecast_stratified_audit = any(
            value.plan.designer_id
            == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            for value in self.adaptive_result.paired_audit_executions
        )
        return {
            "schema_version": 1,
            "runtime": {
                "runtime_id": (
                    OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_ID
                ),
                "runtime_version": (
                    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_VERSION
                    if has_forecast_stratified_audit
                    else (
                        OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_VERSION
                        if has_forecast and has_paired_assay
                        else (
                            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_VERSION
                            if has_forecast
                            else (
                                OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_VERSION
                                if has_paired_assay
                                else OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_VERSION
                            )
                        )
                    )
                ),
                "definition_sha256": self.runtime_definition_sha256,
            },
            "adaptive_result_sha256": (
                self.adaptive_result.adaptive_result_sha256
            ),
            "campaign_stage_receipt_sha256": (
                self.campaign_stage.receipt_sha256
            ),
            "shared_combined_result_sha256": (
                self.campaign_stage.result.result_sha256
            ),
            "archive_and_learning_published_once": True,
            "workload_objective_model_provider_prompt_fields_present": False,
        }

    @property
    def result(self):
        return self.campaign_stage.result

    @property
    def archive_preparation(self):
        return self.campaign_stage.archive_preparation

    @property
    def archive_commit(self):
        return self.campaign_stage.archive_commit

    @property
    def learning_preparation(self):
        return self.campaign_stage.learning_preparation

    @property
    def learning_commit(self):
        return self.campaign_stage.learning_commit

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "adaptive_result": self.adaptive_result.to_record(
                include_evidence=include_evidence
            ),
            "campaign_stage": self.campaign_stage.to_record(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualPortfolioCampaignStageRuntime:
    """Run and transactionally publish one generic adaptive stage."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    archive: ResidualArchiveTransitionPort = field(
        repr=False,
        compare=False,
    )
    learning: TransactionalResidualLearningStore = field(
        repr=False,
        compare=False,
    )
    racing_policy: OutcomeAdaptiveActionRacingPolicy
    market_projector: AdaptiveActionMarketProjectorPort = field(
        repr=False,
        compare=False,
    )
    outcome_projector: AdaptiveActionOutcomeProjectorPort = field(
        repr=False,
        compare=False,
    )
    slate_value: MaterializedSlateValuePort = field(
        repr=False,
        compare=False,
    )
    slate_feasibility: MaterializedSlateFeasibilityPort = field(
        repr=False,
        compare=False,
    )
    return_value: MaterializedActionReturnValuePort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    exploration_policy: MaterializedActionExplorationPort | None = field(
        default_factory=PrequentialLowDiscrepancyResidualExploration,
        repr=False,
        compare=False,
    )
    diagnostic_allocation_policy: (
        MaterializedActionAllocationPolicyPort | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )
    phase_committer: OutcomeAdaptiveResidualPhaseCommitPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    paired_audit_designer: SamePrefixPairedAuditDesignerPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    forecast_opportunity_challenger: (
        ProtectedCurrentPrefixForecastOpportunityChallenger | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )
    forecast_opportunity_shadow_designer: (
        ForecastOpportunitySamePrefixShadowDesignerPort | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )
    require_durable_phase_commits: bool = False
    hierarchical_kappa: float = 4.0
    confidence_width: float = 1.0
    exact_combination_limit: int = 250_000
    beam_width: int = 512

    def __post_init__(self) -> None:
        if type(self.experts) is not tuple or not self.experts:
            raise ValueError("experts must be a non-empty exact tuple")
        if not isinstance(self.archive, ResidualArchiveTransitionPort):
            raise TypeError(
                "archive must implement ResidualArchiveTransitionPort"
            )
        if type(self.learning) is not TransactionalResidualLearningStore:
            raise TypeError("learning must be an exact transactional store")
        self.learning.__post_init__()
        if type(self.racing_policy) is not OutcomeAdaptiveActionRacingPolicy:
            # Structural admission mirroring the stage-runtime gate: an
            # injected controller must expose the exact racing surface.
            required_attributes = (
                "policy_id",
                "policy_version",
                "definition_sha256",
                "design_diagnostic_pilot",
                "select_next",
                "__post_init__",
            )
            missing = tuple(
                name
                for name in required_attributes
                if not hasattr(self.racing_policy, name)
            )
            if missing:
                raise TypeError(
                    "racing_policy must be the exact racing policy or "
                    "implement its surface; missing: "
                    + ",".join(missing)
                )
        if self.exploration_policy is not None and not isinstance(
            self.exploration_policy,
            MaterializedActionExplorationPort,
        ):
            raise TypeError(
                "exploration_policy must implement "
                "MaterializedActionExplorationPort"
            )
        if self.diagnostic_allocation_policy is not None and not isinstance(
            self.diagnostic_allocation_policy,
            MaterializedActionAllocationPolicyPort,
        ):
            raise TypeError(
                "diagnostic_allocation_policy must implement its port"
            )
        if self.paired_audit_designer is not None and not isinstance(
            self.paired_audit_designer,
            SamePrefixPairedAuditDesignerPort,
        ):
            raise TypeError(
                "paired_audit_designer must implement its port"
            )
        if (
            self.forecast_opportunity_challenger is not None
            and type(self.forecast_opportunity_challenger)
            is not ProtectedCurrentPrefixForecastOpportunityChallenger
        ):
            raise TypeError(
                "forecast_opportunity_challenger must be exact"
            )
        if (
            self.forecast_opportunity_shadow_designer is not None
            and not isinstance(
                self.forecast_opportunity_shadow_designer,
                ForecastOpportunitySamePrefixShadowDesignerPort,
            )
        ):
            raise TypeError(
                "forecast_opportunity_shadow_designer must implement its port"
            )
        if (
            self.forecast_opportunity_shadow_designer is not None
            and self.forecast_opportunity_challenger is None
        ):
            raise ValueError(
                "forecast shadow designer requires a forecast challenger"
            )
        if type(self.require_durable_phase_commits) is not bool:
            raise TypeError("require_durable_phase_commits must be exact")
        for name in ("hierarchical_kappa", "confidence_width"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be a positive finite exact float"
                )
        for name in ("exact_combination_limit", "beam_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> OutcomeAdaptiveResidualCampaignStageReceipt:
        self.__post_init__()
        adaptive = await OutcomeAdaptiveResidualPortfolioEvolution(
            experts=self.experts,
            broker=RegretBrokeredMaterializedActionPolicy(
                ledger=self.learning.state.broker_evidence,
                return_value=self.return_value,
                exploration_policy=self.exploration_policy,
                hierarchical_kappa=self.hierarchical_kappa,
                confidence_width=self.confidence_width,
                exact_combination_limit=self.exact_combination_limit,
                beam_width=self.beam_width,
            ),
            racing_policy=self.racing_policy,
            market_projector=self.market_projector,
            outcome_projector=self.outcome_projector,
            slate_value=self.slate_value,
            slate_feasibility=self.slate_feasibility,
            diagnostic_allocation_policy=(
                self.diagnostic_allocation_policy
            ),
            phase_committer=self.phase_committer,
            paired_audit_designer=self.paired_audit_designer,
            forecast_opportunity_challenger=(
                self.forecast_opportunity_challenger
            ),
            forecast_opportunity_shadow_designer=(
                self.forecast_opportunity_shadow_designer
            ),
            require_durable_phase_commits=(
                self.require_durable_phase_commits
            ),
        ).run(request)
        campaign_stage = await commit_residual_campaign_stage(
            archive=self.archive,
            learning=self.learning,
            result=adaptive.result,
        )
        has_forecast = (
            adaptive.forecast_opportunity_challenger_definition_sha256
            is not None
        )
        has_paired_assay = bool(adaptive.paired_audit_executions)
        has_forecast_stratified_audit = any(
            value.plan.designer_id
            == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            for value in adaptive.paired_audit_executions
        )
        runtime_definition_sha256 = (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256
            if has_forecast_stratified_audit
            else (
                OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                if has_forecast and has_paired_assay
                else (
                    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                    if has_forecast
                    else (
                        OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                        if has_paired_assay
                        else OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256
                    )
                )
            )
        )
        return OutcomeAdaptiveResidualCampaignStageReceipt(
            adaptive_result=adaptive,
            campaign_stage=campaign_stage,
            runtime_definition_sha256=runtime_definition_sha256,
        )


__all__ = [
    "OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_ID",
    "OUTCOME_ADAPTIVE_RESIDUAL_CAMPAIGN_RUNTIME_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_CAMPAIGN_RUNTIME_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_CAMPAIGN_RUNTIME_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_CAMPAIGN_RUNTIME_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_CAMPAIGN_RUNTIME_VERSION",
    "OutcomeAdaptiveResidualCampaignStageReceipt",
    "OutcomeAdaptiveResidualPortfolioCampaignStageRuntime",
]
