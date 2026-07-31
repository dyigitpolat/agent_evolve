"""Transactional campaign runtime for sequential residual evolution.

The sequential search service owns only propose/pilot/gate/continuation
orchestration.  This adapter closes its standard combined result through the
same workload archive and conserved learning transaction used by the
single-wave runtime.  No workload, objective, model, provider, or prompt
identity enters the application policy.
"""

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
    MaterializedActionProposalExpertPort,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.sequential_lineage_allocation import (
    AnyPositiveSequentialLineageGate,
    SequentialAllocationGatePort,
    SequentialLineageAllocationPlannerPort,
    SequentialPilotOutcomeProjectorPort,
)
from agent_evolve.application.sequential_residual_portfolio_evolution import (
    SequentialResidualPhaseCommitPort,
    SequentialResidualPortfolioEvolution,
    SequentialResidualPortfolioEvolutionResult,
)
from agent_evolve.domain.patch import require_sha256


SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_ID = (
    "transactional_sequential_residual_portfolio_campaign"
)
SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_VERSION = 2
SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:transactional-sequential-residual-portfolio-campaign:v2;"
    b"search=sealed-universe-k-pilot-precommitted-branch-k-rest;"
    b"gate=injected-sequential-allocation-gate-port;"
    b"publication=shared-generic-residual-archive-learning-transaction;"
    b"broker-cutoff=prior-learning-state-only;"
    b"phase-interception=injected-optional-durable-commit-port;"
    b"workload-objective-model-provider-prompt-branches=false"
).hexdigest()

_RECEIPT_DOMAIN = (
    b"agent-evolve:sequential-residual-campaign-stage-receipt:v1\x00"
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
class SequentialResidualCampaignStageReceipt:
    """Hash join between the sequential trace and committed campaign state."""

    sequential_result: SequentialResidualPortfolioEvolutionResult
    campaign_stage: ResidualCampaignStageReceipt
    runtime_definition_sha256: str = (
        SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256
    )
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.sequential_result)
            is not SequentialResidualPortfolioEvolutionResult
        ):
            raise TypeError("sequential_result must be exact")
        self.sequential_result.__post_init__()
        if type(self.campaign_stage) is not ResidualCampaignStageReceipt:
            raise TypeError("campaign_stage must be exact")
        self.campaign_stage.__post_init__()
        require_sha256(
            self.runtime_definition_sha256,
            "runtime_definition_sha256",
        )
        if (
            self.sequential_result.result.result_sha256
            != self.campaign_stage.result.result_sha256
        ):
            raise ValueError(
                "sequential trace and campaign transaction name "
                "different results"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtime": {
                "runtime_id": SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_ID,
                "runtime_version": (
                    SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_VERSION
                ),
                "definition_sha256": self.runtime_definition_sha256,
            },
            "sequential_result_sha256": (
                self.sequential_result.sequential_result_sha256
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
        """Drop-in view of the standard combined residual result."""

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
            "sequential_result": self.sequential_result.to_record(
                include_evidence=include_evidence
            ),
            "campaign_stage": self.campaign_stage.to_record(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SequentialResidualPortfolioCampaignStageRuntime:
    """Run and transactionally close one generic sequential campaign stage."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    archive: ResidualArchiveTransitionPort = field(repr=False, compare=False)
    learning: TransactionalResidualLearningStore = field(
        repr=False,
        compare=False,
    )
    planner: SequentialLineageAllocationPlannerPort = field(
        repr=False,
        compare=False,
    )
    pilot_outcome_projector: SequentialPilotOutcomeProjectorPort = field(
        repr=False,
        compare=False,
    )
    slate_value: MaterializedSlateValuePort = field(repr=False, compare=False)
    slate_feasibility: MaterializedSlateFeasibilityPort = field(
        repr=False,
        compare=False,
    )
    gate: SequentialAllocationGatePort = field(
        default_factory=AnyPositiveSequentialLineageGate,
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
    phase_committer: SequentialResidualPhaseCommitPort | None = field(
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
        if not isinstance(self.gate, SequentialAllocationGatePort):
            raise TypeError("gate must implement its application port")
        if self.exploration_policy is not None and not isinstance(
            self.exploration_policy,
            MaterializedActionExplorationPort,
        ):
            raise TypeError(
                "exploration_policy must implement "
                "MaterializedActionExplorationPort"
            )
        if type(self.require_durable_phase_commits) is not bool:
            raise TypeError("require_durable_phase_commits must be exact")
        for name in ("hierarchical_kappa", "confidence_width"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite exact float")
        for name in ("exact_combination_limit", "beam_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> SequentialResidualCampaignStageReceipt:
        self.__post_init__()
        sequential = await SequentialResidualPortfolioEvolution(
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
            planner=self.planner,
            pilot_outcome_projector=self.pilot_outcome_projector,
            slate_value=self.slate_value,
            slate_feasibility=self.slate_feasibility,
            gate=self.gate,
            phase_committer=self.phase_committer,
            require_durable_phase_commits=(
                self.require_durable_phase_commits
            ),
        ).run(request)
        campaign_stage = await commit_residual_campaign_stage(
            archive=self.archive,
            learning=self.learning,
            result=sequential.result,
        )
        return SequentialResidualCampaignStageReceipt(
            sequential_result=sequential,
            campaign_stage=campaign_stage,
        )


__all__ = [
    "SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_ID",
    "SEQUENTIAL_RESIDUAL_CAMPAIGN_RUNTIME_VERSION",
    "SequentialResidualCampaignStageReceipt",
    "SequentialResidualPortfolioCampaignStageRuntime",
]
