"""Durable outcome-adaptive evolution over one sealed proposal market.

This application service turns outcome-adaptive action racing into an honest
live protocol.  Proposal experts materialize every action before a current
outcome exists.  A diagnostic subset is evaluated first, and each subsequent
action is selected only from outcomes that have crossed a durable interception
boundary.

The service is workload-, objective-, model-, provider-, prompt-, and
configuration-schema blind.  Workloads remain behind the existing proposal,
evaluation, archive-utility, slate-value, and slate-feasibility ports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchivePortfolioConsequenceUtilityPort,
    validate_candidate_archive_portfolio_consequence_utility,
)
from agent_evolve.application.forecast_opportunity_shadow_calibration import (
    ForecastOpportunityShadowCalibrationProjector,
)
from agent_evolve.application.materialized_action_broker import (
    BrokerActionScore,
    MaterializedActionAllocationRequirement,
    MaterializedActionBrokerDecision,
    MaterializedActionBrokerRequest,
    MaterializedActionDescriptor,
    MaterializedSlateFeasibilityPort,
    MaterializedSlateValuePort,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionAllocationDirective,
    AdaptiveActionDescriptor,
    AdaptiveActionFactorCell,
    AdaptiveActionOutcome,
    AdaptiveActionRacingDecision,
    AdaptiveActionSetOutcome,
    AdaptiveActionWave,
    OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
    OutcomeAdaptiveActionRacingPolicy,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScoreBatch,
    MaterializedActionScorePort,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityCalibrationObservation,
)
from agent_evolve.application.protected_current_prefix_forecast_opportunity import (
    ProtectedCurrentPrefixForecastOpportunityChallenger,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    MaterializedActionAllocationPolicyPort,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    MaterializedActionProposalExpertPort,
    ResidualPortfolioDecisionRequest,
    ResidualPortfolioEvolutionResult,
    evaluate_materialized_action_counterfactual_subset,
    evaluate_materialized_action_subset,
    propose_materialized_action_batches,
)
from agent_evolve.application.same_prefix_paired_audit import (
    FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS,
    FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID,
    FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID,
    ForecastOpportunitySamePrefixShadowDesignerPort,
    SamePrefixPairedAuditAdjudicator,
    SamePrefixPairedAuditArm,
    SamePrefixPairedAuditDesignerPort,
    SamePrefixPairedAuditObservation,
    SamePrefixPairedAuditPlan,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_ID = (
    "outcome_adaptive_residual_portfolio_evolution"
)
OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_VERSION = 5
OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:outcome-adaptive-residual-portfolio-evolution:v5;"
    b"proposal-market=sealed-once-before-current-outcomes;"
    b"prior-authority=materialized-action-broker-selection-index;"
    b"diagnostic-allocation=optional-outcome-blind-required-action-policy;"
    b"diagnostic=portable-constrained-lane-head-design;"
    b"continuation=one-real-outcome-then-one-action;"
    b"causal-outcomes=fixed-set-and-prior-conditioned-set-lift;"
    b"policy-use-of-causal-set-outcomes=version-gated;"
    b"v4=conditional-opportunity-saturation-and-uniform-audit;"
    b"v5=seed-invariant-directed-selection-and-epsilon-greedy-risk-audit;"
    b"randomized-audit=logged-propensity;"
    b"durability=market-diagnostic-each-adaptive-step-final;"
    b"exactly-once=disjoint-action-evaluation-waves;"
    b"archive-publication=combined-final-slate-only;"
    b"workload-objective-model-provider-prompt-config-branches=false"
).hexdigest()
OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_VERSION = 6
OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:outcome-adaptive-residual-portfolio-evolution:v6;"
        b"base-definition="
        + OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256.encode(
            "ascii"
        )
        + b";paired-audit=optional-same-prefix-distinct-arm-assay;"
        b"paired-audit-plan=durable-before-either-arm-evaluation;"
        b"paired-audit-publication=authoritative-arm-only;"
        b"physical-budget=one-extra-real-evaluation-per-assayed-stage;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_VERSION = 7
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:outcome-adaptive-residual-portfolio-evolution:v7;"
        b"base-definition="
        + OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256.encode(
            "ascii"
        )
        + b";continuation=protected-current-prefix-forecast-opportunity;"
        b"forecast-geometry=frozen-before-diagnostic-outcomes;"
        b"fallback=complete-incumbent-racing-decision;"
        b"abstention=preserve-fallback;"
        b"candidate-outcomes=observed-prefix-only;"
        b"eligible-candidate-outcomes=false;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
_FORECAST_OPPORTUNITY_DEFINITION_SHA256 = (
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256
)
_PAIRED_AUDIT_DEFINITION_SHA256 = (
    OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_DEFINITION_SHA256
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_VERSION = 8
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:outcome-adaptive-residual-portfolio-evolution:v8;"
        b"base-definition="
        + _FORECAST_OPPORTUNITY_DEFINITION_SHA256.encode("ascii")
        + b";shadow=optional-same-prefix-challenger-vs-fallback-assay;"
        b"shadow-plan=durable-before-either-arm-evaluation;"
        b"shadow-schedule=final-continuation-only;"
        b"shadow-publication=authoritative-challenger-only;"
        b"physical-budget=one-extra-real-evaluation-per-assayed-stage;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_VERSION = 9
OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:outcome-adaptive-residual-portfolio-evolution:v9;"
        b"base-definition="
        + _FORECAST_OPPORTUNITY_DEFINITION_SHA256.encode("ascii")
        + b";audit=one-hash-rotated-same-prefix-forecast-stratum-assay;"
        b"audit-plan=durable-before-either-arm-evaluation;"
        b"abstention-authoritative-arm=protected-fallback;"
        b"intervention-authoritative-arm=forecast-challenger;"
        b"counterfactual-action=quarantined-from-later-authoritative-selection;"
        b"sampling=uniform-nonempty-stratum-then-uniform-action;"
        b"calibration=paired-role-and-propensity-authenticated;"
        b"audit-publication=authoritative-arm-only;"
        b"physical-budget=at-most-one-extra-real-evaluation-per-assayed-stage;"
        b"workload-objective-model-provider-prompt-config-branches=false"
    ).hexdigest()
)

PORTABLE_ADAPTIVE_MARKET_PROJECTOR_ID = (
    "portable_materialized_adaptive_market_projector"
)
PORTABLE_ADAPTIVE_MARKET_PROJECTOR_VERSION = 1
PORTABLE_ADAPTIVE_MARKET_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portable-materialized-adaptive-market-projector:v1;"
    b"deduplicate=required-first-then-prior-score-native-rank-action-hash;"
    b"lane=expert;operator=operator-id;"
    b"prior=broker-selection-index;"
    b"lineage=current-run-parent-membership;"
    b"semantics=generic-materialized-action-context-only;"
    b"workload-objective-model-provider-prompt-config-fields=false"
).hexdigest()
PORTABLE_ADAPTIVE_SEMANTIC_VIEW_ID = (
    "portable_materialized_adaptive_semantic_view"
)
PORTABLE_ADAPTIVE_SEMANTIC_VIEW_VERSION = 1
PORTABLE_ADAPTIVE_SEMANTIC_VIEW_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portable-materialized-adaptive-semantic-view:v1;"
    b"inputs=generic-materialized-action-context-lineage-and-role;"
    b"workload-objective-model-provider-prompt-config-fields=false"
).hexdigest()

CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_ID = (
    "candidate_archive_adaptive_action_outcome_projector"
)
CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_VERSION = 2

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_MARKET_STATE_DOMAIN = b"agent-evolve:adaptive-market-state:v1\x00"
_SCORE_ENSEMBLE_PROJECTOR_DEFINITION_DOMAIN = (
    b"agent-evolve:score-ensemble-adaptive-market-projector:v1\x00"
)
_FACTOR_STRATIFIED_PROJECTOR_DEFINITION_DOMAIN = (
    b"agent-evolve:factor-stratified-adaptive-market-projector:v1\x00"
)
_OUTCOME_PROJECTOR_DEFINITION_DOMAIN = (
    b"agent-evolve:adaptive-outcome-projector-definition:v2\x00"
)
_PHASE_RECEIPT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-phase-receipt:v1\x00"
)
_PHASE_ACK_DOMAIN = b"agent-evolve:outcome-adaptive-phase-ack:v1\x00"
_PAIRED_AUDIT_EXECUTION_DOMAIN = (
    b"agent-evolve:outcome-adaptive-paired-audit-execution:v1\x00"
)
_RESULT_DOMAIN = b"agent-evolve:outcome-adaptive-result:v5\x00"
_PAIRED_AUDIT_RESULT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-result:v6\x00"
)
_FORECAST_OPPORTUNITY_RESULT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-result:v7\x00"
)
_FORECAST_SHADOW_RESULT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-result:v8\x00"
)
_FORECAST_STRATIFIED_AUDIT_RESULT_DOMAIN = (
    b"agent-evolve:outcome-adaptive-result:v9\x00"
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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _candidate_id_tuple(
    values: tuple[CandidateId, ...],
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError("current_run_parent_ids must be an exact tuple")
    result: list[str] = []
    for value in values:
        if type(value) is not CandidateId:
            raise TypeError("current_run_parent_ids must contain exact IDs")
        CandidateId.__post_init__(value)
        result.append(value.value)
    if result != sorted(set(result)):
        raise ValueError(
            "current_run_parent_ids must be unique and canonical"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AdaptiveActionSemanticView:
    """Portable operator grouping and semantic cells for one action."""

    operator_id: str
    semantic_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_token(self.operator_id, name="operator_id")
        if (
            type(self.semantic_cell_ids) is not tuple
            or self.semantic_cell_ids
            != tuple(sorted(set(self.semantic_cell_ids)))
        ):
            raise ValueError(
                "semantic_cell_ids must be unique and canonical"
            )
        for value in self.semantic_cell_ids:
            _require_token(value, name="semantic_cell_id")


@runtime_checkable
class AdaptiveActionSemanticViewPort(Protocol):
    """Optional pre-evaluation semantic view; never sees candidate outcomes."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def view(
        self,
        action: MaterializedActionDescriptor,
    ) -> AdaptiveActionSemanticView: ...


@dataclass(frozen=True, slots=True)
class PortableMaterializedAdaptiveSemanticView:
    """Default view requiring no workload-specific integration."""

    projection_id: str = PORTABLE_ADAPTIVE_SEMANTIC_VIEW_ID
    projection_version: int = PORTABLE_ADAPTIVE_SEMANTIC_VIEW_VERSION
    definition_sha256: str = (
        PORTABLE_ADAPTIVE_SEMANTIC_VIEW_DEFINITION_SHA256
    )

    def view(
        self,
        action: MaterializedActionDescriptor,
    ) -> AdaptiveActionSemanticView:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be exact")
        action.__post_init__()
        context = action.context
        return AdaptiveActionSemanticView(
            operator_id=action.operator_id,
            semantic_cell_ids=tuple(
                sorted(
                    {
                        f"role:{action.role_id}",
                        f"residual:{context.residual_frontier_cell}",
                        (
                            "parent_position:"
                            f"{context.parent_position_cell}"
                        ),
                        (
                            "archive_relation:"
                            f"{context.archive_relation_cell}"
                        ),
                        f"patch:{context.patch_compatibility_cell}",
                        (
                            "calibration:"
                            f"{context.forecast_calibration_cell}"
                        ),
                        f"phase:{context.phase.value}",
                        f"arity:{len(action.parent_ids)}",
                        f"reference:{int(action.reference_action)}",
                        (
                            "source_distance:"
                            f"{context.source_distance_bin}"
                        ),
                        f"memory_dose:{context.memory_dose_bin}",
                        "structure:"
                        + context.structural_signature_sha256[:16],
                    }
                )
            ),
        )


@runtime_checkable
class AdaptiveActionMarketProjectorPort(Protocol):
    """Project materialized actions into a portable outcome-blind market."""

    projector_id: str
    projector_version: int
    definition_sha256: str
    state_sha256: str

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: tuple[BrokerActionScore, ...],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]: ...


@dataclass(frozen=True, slots=True)
class PortableMaterializedAdaptiveMarketProjector:
    """Default zero-workload-knowledge projection for adaptive racing."""

    current_run_parent_ids: tuple[CandidateId, ...] = ()
    semantic_view: AdaptiveActionSemanticViewPort = field(
        default_factory=PortableMaterializedAdaptiveSemanticView,
        repr=False,
        compare=False,
    )
    projector_id: str = PORTABLE_ADAPTIVE_MARKET_PROJECTOR_ID
    projector_version: int = PORTABLE_ADAPTIVE_MARKET_PROJECTOR_VERSION
    definition_sha256: str = (
        PORTABLE_ADAPTIVE_MARKET_PROJECTOR_DEFINITION_SHA256
    )
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = _candidate_id_tuple(self.current_run_parent_ids)
        if self.projector_id != PORTABLE_ADAPTIVE_MARKET_PROJECTOR_ID:
            raise ValueError("projector_id is immutable")
        if (
            self.projector_version
            != PORTABLE_ADAPTIVE_MARKET_PROJECTOR_VERSION
        ):
            raise ValueError("projector_version is immutable")
        if (
            self.definition_sha256
            != PORTABLE_ADAPTIVE_MARKET_PROJECTOR_DEFINITION_SHA256
        ):
            raise ValueError("projector definition is immutable")
        if not isinstance(
            self.semantic_view,
            AdaptiveActionSemanticViewPort,
        ):
            raise TypeError("semantic_view must implement its port")
        _require_token(
            self.semantic_view.projection_id,
            name="semantic projection_id",
        )
        if (
            type(self.semantic_view.projection_version) is not int
            or self.semantic_view.projection_version <= 0
        ):
            raise ValueError("semantic projection_version must be positive")
        require_sha256(
            self.semantic_view.definition_sha256,
            "semantic view definition",
        )
        object.__setattr__(
            self,
            "state_sha256",
            _hash(
                _MARKET_STATE_DOMAIN,
                {
                    "schema_version": 1,
                    "current_run_parent_ids": list(ids),
                    "semantic_view_definition_sha256": (
                        self.semantic_view.definition_sha256
                    ),
                },
            ),
        )

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: tuple[BrokerActionScore, ...],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        del request, proposals
        self.__post_init__()
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        action_by_sha: dict[str, MaterializedActionDescriptor] = {}
        for action in actions:
            if type(action) is not MaterializedActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            action.__post_init__()
            if action.action_sha256 in action_by_sha:
                raise ValueError("actions repeat an identity")
            action_by_sha[action.action_sha256] = action
        if (
            type(scores) is not tuple
            or any(type(value) is not BrokerActionScore for value in scores)
        ):
            raise TypeError("scores must contain exact broker scores")
        score_by_action = {value.action_sha256: value for value in scores}
        for value in scores:
            value.__post_init__()
        if set(score_by_action) != set(action_by_sha):
            raise ValueError("broker scores do not exactly cover actions")
        if (
            type(required_action_sha256s) is not tuple
            or required_action_sha256s
            != tuple(sorted(set(required_action_sha256s)))
            or not set(required_action_sha256s).issubset(action_by_sha)
        ):
            raise ValueError(
                "required action hashes must be canonical and in-market"
            )
        return self._project_with_prior(
            actions=actions,
            prior_score_by_action={
                value.action_sha256: value.selection_index
                for value in scores
            },
            required_action_sha256s=required_action_sha256s,
        )

    def _project_with_prior(
        self,
        *,
        actions: tuple[MaterializedActionDescriptor, ...],
        prior_score_by_action: dict[str, float],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        action_by_sha = {
            value.action_sha256: value for value in actions
        }
        if set(prior_score_by_action) != set(action_by_sha):
            raise ValueError("prior scores do not exactly cover actions")
        for value in prior_score_by_action.values():
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError("prior scores must be finite probabilities")
        required = set(required_action_sha256s)
        by_phenotype: dict[str, list[MaterializedActionDescriptor]] = {}
        for action in actions:
            by_phenotype.setdefault(
                action.phenotype_identity_sha256,
                [],
            ).append(action)
        representatives: list[MaterializedActionDescriptor] = []
        for rows in by_phenotype.values():
            required_rows = [
                value
                for value in rows
                if value.action_sha256 in required
            ]
            if len(required_rows) > 1:
                raise ValueError(
                    "fixed constraints repeat one phenotype"
                )
            representatives.append(
                (
                    required_rows[0]
                    if required_rows
                    else max(
                        rows,
                        key=lambda value: (
                            prior_score_by_action[
                                value.action_sha256
                            ],
                            -value.native_rank,
                            value.action_sha256,
                        ),
                    )
                )
            )
        if not required.issubset(
            {value.action_sha256 for value in representatives}
        ):
            raise ValueError(
                "phenotype projection displaced a fixed constraint"
            )
        by_lane: dict[str, list[MaterializedActionDescriptor]] = {}
        for action in representatives:
            by_lane.setdefault(action.expert_id, []).append(action)
        canonical_rank: dict[str, int] = {}
        for rows in by_lane.values():
            for ordinal, action in enumerate(
                sorted(
                    rows,
                    key=lambda value: (
                        value.native_rank,
                        value.action_sha256,
                    ),
                ),
                start=1,
            ):
                canonical_rank[action.action_sha256] = ordinal
        current_ids = set(self.current_run_parent_ids)
        result: list[AdaptiveActionDescriptor] = []
        for action in sorted(
            representatives,
            key=lambda value: value.action_sha256,
        ):
            semantic = self.semantic_view.view(action)
            if type(semantic) is not AdaptiveActionSemanticView:
                raise TypeError("semantic_view returned a foreign value")
            semantic.__post_init__()
            generated_parent = any(
                value in current_ids for value in action.parent_ids
            )
            result.append(
                AdaptiveActionDescriptor(
                action_sha256=action.action_sha256,
                phenotype_sha256=action.phenotype_identity_sha256,
                lane_id=action.expert_id,
                operator_id=semantic.operator_id,
                native_rank=canonical_rank[action.action_sha256],
                lane_size=len(by_lane[action.expert_id]),
                prior_score=float(
                    prior_score_by_action[action.action_sha256]
                ),
                parent_generated_in_current_run=generated_parent,
                semantic_cell_ids=tuple(
                    sorted(
                        {
                            *semantic.semantic_cell_ids,
                            (
                                "parent:generated"
                                if generated_parent
                                else "parent:archive"
                            ),
                        }
                    )
                ),
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ScoreEnsembleMaterializedAdaptiveMarketProjector:
    """Combine opaque pre-evaluation authorities by within-market rank."""

    scorers: tuple[MaterializedActionScorePort, ...] = field(
        repr=False,
        compare=False,
    )
    scorer_weights: tuple[tuple[str, float], ...]
    current_run_parent_ids: tuple[CandidateId, ...] = ()
    semantic_view: AdaptiveActionSemanticViewPort = field(
        default_factory=PortableMaterializedAdaptiveSemanticView,
        repr=False,
        compare=False,
    )
    projector_id: str = (
        "score_ensemble_materialized_adaptive_market_projector"
    )
    projector_version: int = 1
    definition_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = _candidate_id_tuple(self.current_run_parent_ids)
        if type(self.scorers) is not tuple or not self.scorers:
            raise ValueError("scorers must be a non-empty exact tuple")
        identities: list[tuple[str, int, str]] = []
        for scorer in self.scorers:
            if not isinstance(scorer, MaterializedActionScorePort):
                raise TypeError("scorer must implement its port")
            scorer_id = getattr(scorer, "scorer_id", None)
            scorer_version = getattr(scorer, "scorer_version", None)
            definition_sha256 = getattr(
                scorer,
                "definition_sha256",
                None,
            )
            _require_token(scorer_id, name="scorer_id")
            if type(scorer_version) is not int or scorer_version <= 0:
                raise ValueError("scorer_version must be positive")
            require_sha256(
                definition_sha256,
                "scorer definition_sha256",
            )
            identities.append(
                (scorer_id, scorer_version, definition_sha256)
            )
        scorer_ids = tuple(value[0] for value in identities)
        if scorer_ids != tuple(sorted(set(scorer_ids))):
            raise ValueError("scorers must be unique and ID-canonical")
        if (
            type(self.scorer_weights) is not tuple
            or tuple(value[0] for value in self.scorer_weights)
            != scorer_ids
        ):
            raise ValueError("scorer_weights must exactly cover scorers")
        total_weight = 0.0
        for scorer_id, weight in self.scorer_weights:
            _require_token(scorer_id, name="weight scorer_id")
            if (
                type(weight) is not float
                or not math.isfinite(weight)
                or weight <= 0.0
            ):
                raise ValueError("scorer weights must be positive")
            total_weight += weight
        if not isinstance(
            self.semantic_view,
            AdaptiveActionSemanticViewPort,
        ):
            raise TypeError("semantic_view must implement its port")
        require_sha256(
            self.semantic_view.definition_sha256,
            "semantic view definition",
        )
        _require_token(self.projector_id, name="projector_id")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be positive")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _SCORE_ENSEMBLE_PROJECTOR_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "scorers": [
                        {
                            "scorer_id": scorer_id,
                            "scorer_version": scorer_version,
                            "definition_sha256": definition_sha256,
                            "weight_hex": dict(
                                self.scorer_weights
                            )[scorer_id].hex(),
                        }
                        for scorer_id, scorer_version, definition_sha256
                        in identities
                    ],
                    "normalization": (
                        "within-sealed-market-rank-percentile"
                    ),
                    "aggregation": "normalized-positive-weighted-mean",
                    "semantic_view_definition_sha256": (
                        self.semantic_view.definition_sha256
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_prompt_config_branches": False,
                },
            ),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _hash(
                _MARKET_STATE_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_definition_sha256": (
                        self.definition_sha256
                    ),
                    "current_run_parent_ids": list(ids),
                },
            ),
        )

    @staticmethod
    def _rank_percentiles(
        actions: tuple[MaterializedActionDescriptor, ...],
        values: dict[str, float],
    ) -> dict[str, float]:
        ordered = sorted(
            actions,
            key=lambda action: (
                values[action.action_sha256],
                -action.native_rank,
                action.action_sha256,
            ),
        )
        if len(ordered) == 1:
            return {ordered[0].action_sha256: 1.0}
        return {
            action.action_sha256: index / (len(ordered) - 1)
            for index, action in enumerate(ordered)
        }

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: tuple[BrokerActionScore, ...],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        del scores
        self.__post_init__()
        raw = await asyncio.gather(
            *(
                scorer.score(request, proposals)
                for scorer in self.scorers
            )
        )
        batches = tuple(raw)
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        action_sha256s = {
            value.action_sha256 for value in actions
        }
        percentiles: dict[str, dict[str, float]] = {}
        for scorer, batch in zip(
            self.scorers,
            batches,
            strict=True,
        ):
            if type(batch) is not MaterializedActionScoreBatch:
                raise TypeError("scorer returned a foreign batch")
            batch.__post_init__()
            if (
                batch.scorer_id != scorer.scorer_id
                or batch.scorer_version != scorer.scorer_version
                or batch.scorer_definition_sha256
                != scorer.definition_sha256
                or batch.residual_request_sha256
                != request.request_sha256
                or batch.proposal_sha256s != proposal_sha256s
            ):
                raise ValueError("score batch differs from its authority")
            values = {
                value.action_sha256: value.value
                for value in batch.scores
            }
            if set(values) != action_sha256s:
                raise ValueError("score batch does not cover the market")
            percentiles[batch.scorer_id] = self._rank_percentiles(
                actions,
                values,
            )
        weights = dict(self.scorer_weights)
        total_weight = math.fsum(weights.values())
        combined = {
            action.action_sha256: float(
                math.fsum(
                    weights[scorer_id]
                    * percentiles[scorer_id][
                        action.action_sha256
                    ]
                    for scorer_id in weights
                )
                / total_weight
            )
            for action in actions
        }
        base = PortableMaterializedAdaptiveMarketProjector(
            current_run_parent_ids=self.current_run_parent_ids,
            semantic_view=self.semantic_view,
        )
        return base._project_with_prior(
            actions=actions,
            prior_score_by_action=combined,
            required_action_sha256s=required_action_sha256s,
        )


@dataclass(frozen=True, slots=True)
class FactorStratifiedAdaptiveMarketProjector:
    """Add opaque branch/role/rank factor cells to any market projector.

    The wrapper is an integration-layer adapter.  It never parses expert IDs,
    model names, prompts, configurations, objectives, or outcomes.  A
    composition root explicitly binds each proposal expert to an opaque source
    branch; the remaining factors are framework-native action role and
    materialized within-lane rank quantile.
    """

    delegate: AdaptiveActionMarketProjectorPort = field(
        repr=False,
        compare=False,
    )
    source_branch_by_expert: tuple[tuple[str, str], ...]
    rank_layer_count: int = 3
    projector_id: str = "factor_stratified_adaptive_market_projector"
    projector_version: int = 1
    definition_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.delegate, AdaptiveActionMarketProjectorPort):
            raise TypeError("delegate must implement the adaptive market port")
        _require_token(self.delegate.projector_id, name="delegate projector_id")
        if (
            type(self.delegate.projector_version) is not int
            or self.delegate.projector_version <= 0
        ):
            raise ValueError("delegate projector_version must be positive")
        require_sha256(
            self.delegate.definition_sha256,
            "delegate definition_sha256",
        )
        require_sha256(self.delegate.state_sha256, "delegate state_sha256")
        if (
            type(self.source_branch_by_expert) is not tuple
            or not self.source_branch_by_expert
            or self.source_branch_by_expert
            != tuple(sorted(self.source_branch_by_expert))
        ):
            raise ValueError(
                "source_branch_by_expert must be non-empty and canonical"
            )
        experts: list[str] = []
        for value in self.source_branch_by_expert:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError(
                    "source_branch_by_expert must contain exact pairs"
                )
            expert_id, branch_id = value
            _require_token(expert_id, name="source expert_id")
            _require_token(branch_id, name="source branch_id")
            experts.append(expert_id)
        if len(experts) != len(set(experts)):
            raise ValueError("source_branch_by_expert repeats an expert")
        if (
            type(self.rank_layer_count) is not int
            or self.rank_layer_count < 2
            or self.rank_layer_count > 16
        ):
            raise ValueError("rank_layer_count must be an integer in [2, 16]")
        _require_token(self.projector_id, name="projector_id")
        if self.projector_version != 1:
            raise ValueError("projector_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _FACTOR_STRATIFIED_PROJECTOR_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "delegate_definition_sha256": (
                        self.delegate.definition_sha256
                    ),
                    "rank_layer_count": self.rank_layer_count,
                    "factor_families": [
                        "evolutionary_role",
                        "materialized_rank_layer",
                        "source_branch",
                        "source_branch_role",
                    ],
                    "source_branch_semantics": (
                        "opaque_explicit_composition_root_binding"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_config": False,
                },
            ),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _hash(
                _MARKET_STATE_DOMAIN,
                {
                    "schema_version": 2,
                    "projector_definition_sha256": self.definition_sha256,
                    "delegate_state_sha256": self.delegate.state_sha256,
                    "source_branch_by_expert": [
                        {
                            "expert_id": expert_id,
                            "source_branch_id": branch_id,
                        }
                        for expert_id, branch_id in (
                            self.source_branch_by_expert
                        )
                    ],
                },
            ),
        )

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: tuple[BrokerActionScore, ...],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        self.__post_init__()
        projected = await self.delegate.project(
            request,
            proposals,
            actions,
            scores,
            required_action_sha256s,
        )
        action_by_sha = {
            value.action_sha256: value for value in actions
        }
        if len(action_by_sha) != len(actions):
            raise ValueError("materialized actions repeat an identity")
        branch_by_expert = dict(self.source_branch_by_expert)
        observed_experts = {value.lane_id for value in projected}
        if not observed_experts.issubset(branch_by_expert):
            raise ValueError(
                "source branch binding must cover every projected expert"
            )
        result = []
        for value in projected:
            raw = action_by_sha.get(value.action_sha256)
            if raw is None:
                raise ValueError("projected action is absent materialized input")
            branch_id = branch_by_expert[value.lane_id]
            role_id = raw.role_id
            layer = min(
                self.rank_layer_count - 1,
                (
                    (value.native_rank - 1) * self.rank_layer_count
                )
                // value.lane_size,
            )
            added = (
                AdaptiveActionFactorCell(
                    family_id="evolutionary_role",
                    level_id=role_id,
                ),
                AdaptiveActionFactorCell(
                    family_id="materialized_rank_layer",
                    level_id=f"layer{layer}",
                ),
                AdaptiveActionFactorCell(
                    family_id="source_branch",
                    level_id=branch_id,
                ),
                AdaptiveActionFactorCell(
                    family_id="source_branch_role",
                    level_id=(
                        "cell:"
                        + _hash(
                            _FACTOR_STRATIFIED_PROJECTOR_DEFINITION_DOMAIN,
                            {
                                "source_branch_id": branch_id,
                                "evolutionary_role_id": role_id,
                            },
                        )
                    ),
                ),
            )
            combined = tuple(sorted((*value.factor_cells, *added)))
            if len({cell.family_id for cell in combined}) != len(combined):
                raise ValueError(
                    "delegate and stratifier repeat a factor family"
                )
            result.append(
                AdaptiveActionDescriptor(
                    action_sha256=value.action_sha256,
                    phenotype_sha256=value.phenotype_sha256,
                    lane_id=value.lane_id,
                    operator_id=value.operator_id,
                    native_rank=value.native_rank,
                    lane_size=value.lane_size,
                    prior_score=value.prior_score,
                    parent_generated_in_current_run=(
                        value.parent_generated_in_current_run
                    ),
                    semantic_cell_ids=value.semantic_cell_ids,
                    factor_cells=combined,
                )
            )
        return tuple(result)


@runtime_checkable
class AdaptiveActionOutcomeProjectorPort(Protocol):
    """Project real candidate evaluations into portable archive gain."""

    projector_id: str
    projector_version: int
    definition_sha256: str

    def project(
        self,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> tuple[AdaptiveActionOutcome, ...]: ...

    def joint_gain(
        self,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> float: ...

    def project_set_outcome(
        self,
        prior_evaluations: tuple[MaterializedActionEvaluation, ...],
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> AdaptiveActionSetOutcome: ...


@dataclass(frozen=True, slots=True)
class CandidateArchiveAdaptiveActionOutcomeProjector:
    """Use an injected workload utility without exposing objective semantics."""

    prior_candidates: tuple[EvolutionCandidate, ...]
    utility: CandidateArchivePortfolioConsequenceUtilityPort = field(
        repr=False,
        compare=False,
    )
    projector_id: str = CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_ID
    projector_version: int = (
        CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.prior_candidates) is not tuple
            or any(
                type(value) is not EvolutionCandidate
                for value in self.prior_candidates
            )
        ):
            raise TypeError("prior_candidates must be an exact tuple")
        for value in self.prior_candidates:
            value.__post_init__()
        utility_identity = (
            validate_candidate_archive_portfolio_consequence_utility(
                self.utility
            )
        )
        if (
            self.projector_id
            != CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_ID
        ):
            raise ValueError("projector_id is immutable")
        if (
            self.projector_version
            != CANDIDATE_ARCHIVE_ADAPTIVE_OUTCOME_PROJECTOR_VERSION
        ):
            raise ValueError("projector_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _OUTCOME_PROJECTOR_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "prior_candidate_ids": [
                        value.candidate_id.value
                        for value in self.prior_candidates
                    ],
                    "admission": (
                        "valid_and_operator_and_evidence_compliant"
                    ),
                    "currency": (
                        "individual_fixed_joint_and_conditioned_set_utility"
                    ),
                    "set_observation_policy_role": "observational_only",
                },
            ),
        )

    @staticmethod
    def _validate_evaluations(
        evaluations: tuple[MaterializedActionEvaluation, ...],
        *,
        allow_empty: bool = False,
    ) -> None:
        if (
            type(evaluations) is not tuple
            or (not allow_empty and not evaluations)
            or any(
                type(value) is not MaterializedActionEvaluation
                for value in evaluations
            )
        ):
            raise ValueError("evaluations must be a non-empty exact tuple")
        action_ids: list[str] = []
        for value in evaluations:
            value.__post_init__()
            action_ids.append(value.action.action_sha256)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("evaluations repeat an action")

    @staticmethod
    def _feasible(evaluation: MaterializedActionEvaluation) -> bool:
        candidate = evaluation.candidate
        return bool(
            candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
        )

    @staticmethod
    def _bindings(
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    value.action.action_sha256,
                    value.evaluation_sha256,
                )
                for value in evaluations
            )
        )

    def _portfolio_gain(
        self,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> float:
        self._validate_evaluations(evaluations, allow_empty=True)
        points = tuple(
            value.candidate.objective_map
            for value in sorted(
                evaluations,
                key=lambda item: item.action.action_sha256,
            )
            if self._feasible(value)
        )
        gain = (
            0.0
            if not points
            else self.utility.portfolio_marginal_utility(
                self.prior_candidates,
                points,
            )
        )
        if (
            type(gain) is not float
            or not math.isfinite(gain)
            or gain < 0.0
        ):
            raise ValueError("archive utility returned invalid joint gain")
        return float(gain)

    def project(
        self,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> tuple[AdaptiveActionOutcome, ...]:
        self.__post_init__()
        self._validate_evaluations(evaluations)
        outcomes: list[AdaptiveActionOutcome] = []
        for evaluation in sorted(
            evaluations,
            key=lambda value: value.action.action_sha256,
        ):
            feasible = self._feasible(evaluation)
            gain = (
                0.0
                if not feasible
                else self.utility.marginal_utility(
                    self.prior_candidates,
                    evaluation.candidate.objective_map,
                )
            )
            if (
                type(gain) is not float
                or not math.isfinite(gain)
                or gain < 0.0
            ):
                raise ValueError(
                    "archive utility returned invalid individual gain"
                )
            outcomes.append(
                AdaptiveActionOutcome(
                    action_sha256=evaluation.action.action_sha256,
                    evaluation_sha256=evaluation.evaluation_sha256,
                    feasible=feasible,
                    marginal_archive_gain=float(gain),
                )
            )
        return tuple(outcomes)

    def joint_gain(
        self,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> float:
        self.__post_init__()
        self._validate_evaluations(evaluations)
        return self._portfolio_gain(evaluations)

    def project_set_outcome(
        self,
        prior_evaluations: tuple[MaterializedActionEvaluation, ...],
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> AdaptiveActionSetOutcome:
        """Measure fixed opportunity and causal lift at one wave boundary."""

        self.__post_init__()
        self._validate_evaluations(
            prior_evaluations,
            allow_empty=True,
        )
        self._validate_evaluations(evaluations)
        prior_actions = {
            value.action.action_sha256 for value in prior_evaluations
        }
        current_actions = {
            value.action.action_sha256 for value in evaluations
        }
        if prior_actions & current_actions:
            raise ValueError(
                "prior and current evaluations repeat an action"
            )
        prior_receipts = {
            value.evaluation_sha256 for value in prior_evaluations
        }
        current_receipts = {
            value.evaluation_sha256 for value in evaluations
        }
        if prior_receipts & current_receipts:
            raise ValueError(
                "prior and current evaluations repeat a receipt"
            )
        prior_gain = self._portfolio_gain(prior_evaluations)
        fixed_gain = self._portfolio_gain(evaluations)
        augmented = self._portfolio_gain(
            tuple(
                sorted(
                    (*prior_evaluations, *evaluations),
                    key=lambda value: value.action.action_sha256,
                )
            )
        )
        conditional = augmented - prior_gain
        numerical_tolerance = 1e-12 * max(
            1.0,
            abs(prior_gain),
            abs(augmented),
        )
        if conditional < -numerical_tolerance:
            raise ValueError(
                "archive utility is non-monotone across selected sets"
            )
        if conditional < 0.0:
            conditional = 0.0
        return AdaptiveActionSetOutcome(
            prior_action_evaluation_bindings=self._bindings(
                prior_evaluations
            ),
            current_action_evaluation_bindings=self._bindings(
                evaluations
            ),
            prior_selected_set_gain=float(prior_gain),
            current_wave_fixed_set_gain=float(fixed_gain),
            augmented_selected_set_gain=float(augmented),
            conditional_set_gain=float(conditional),
        )


class OutcomeAdaptiveResidualPhase(str, Enum):
    MARKET_FROZEN = "market_frozen"
    DIAGNOSTIC_EVALUATED = "diagnostic_evaluated"
    PAIRED_AUDIT_FROZEN = "paired_audit_frozen"
    ADAPTIVE_STEP_EVALUATED = "adaptive_step_evaluated"
    FINALIZED = "finalized"


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualPhaseReceipt:
    """One causal interception boundary in an adaptive stage."""

    phase: OutcomeAdaptiveResidualPhase
    phase_ordinal: int
    residual_request_sha256: str
    diagnostic_decision_sha256: str
    product_sha256s: tuple[str, ...]
    evidence: FrozenJsonObject
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.phase) is not OutcomeAdaptiveResidualPhase:
            raise TypeError("phase must be exact")
        if type(self.phase_ordinal) is not int or self.phase_ordinal <= 0:
            raise ValueError("phase_ordinal must be positive")
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        require_sha256(
            self.diagnostic_decision_sha256,
            "diagnostic_decision_sha256",
        )
        if (
            type(self.product_sha256s) is not tuple
            or not self.product_sha256s
        ):
            raise ValueError("product_sha256s must be a non-empty tuple")
        for value in self.product_sha256s:
            require_sha256(value, "product_sha256")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("phase evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_PHASE_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "phase_ordinal": self.phase_ordinal,
            "residual_request_sha256": self.residual_request_sha256,
            "diagnostic_decision_sha256": (
                self.diagnostic_decision_sha256
            ),
            "product_sha256s": list(self.product_sha256s),
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "receipt_sha256": self.receipt_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualPhaseCommitAck:
    committer_id: str
    committer_version: int
    committer_definition_sha256: str
    phase_receipt_sha256: str
    durable: bool
    evidence: FrozenJsonObject
    ack_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.committer_id, name="committer_id")
        if type(self.committer_version) is not int or self.committer_version <= 0:
            raise ValueError("committer_version must be positive")
        require_sha256(
            self.committer_definition_sha256,
            "committer_definition_sha256",
        )
        require_sha256(
            self.phase_receipt_sha256,
            "phase_receipt_sha256",
        )
        if type(self.durable) is not bool:
            raise TypeError("durable must be exact")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("commit evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "ack_sha256",
            _hash(_PHASE_ACK_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "committer": {
                "committer_id": self.committer_id,
                "committer_version": self.committer_version,
                "definition_sha256": self.committer_definition_sha256,
            },
            "phase_receipt_sha256": self.phase_receipt_sha256,
            "durable": self.durable,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "ack_sha256": self.ack_sha256}
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class OutcomeAdaptiveResidualPhaseCommitPort(Protocol):
    committer_id: str
    committer_version: int
    definition_sha256: str

    async def commit(
        self,
        receipt: OutcomeAdaptiveResidualPhaseReceipt,
    ) -> OutcomeAdaptiveResidualPhaseCommitAck: ...


def _flatten(
    batches: tuple[MaterializedActionEvaluationBatch, ...],
) -> tuple[MaterializedActionEvaluation, ...]:
    by_action = {
        value.action.action_sha256: value
        for batch in batches
        for value in batch.evaluations
    }
    return tuple(by_action[value] for value in sorted(by_action))


def _merge_waves(
    proposals: tuple[MaterializedActionProposalBatch, ...],
    waves: tuple[tuple[MaterializedActionEvaluationBatch, ...], ...],
) -> tuple[MaterializedActionEvaluationBatch, ...]:
    proposal_by_expert = {value.expert_id: value for value in proposals}
    source_by_expert: dict[str, list[MaterializedActionEvaluationBatch]] = {}
    for wave in waves:
        for batch in wave:
            source_by_expert.setdefault(batch.expert_id, []).append(batch)
    merged: list[MaterializedActionEvaluationBatch] = []
    for expert_id in sorted(source_by_expert):
        source = source_by_expert[expert_id]
        proposal = proposal_by_expert[expert_id]
        evaluations = tuple(
            sorted(
                (
                    value
                    for batch in source
                    for value in batch.evaluations
                ),
                key=lambda value: value.action.action_sha256,
            )
        )
        action_ids = tuple(
            value.action.action_sha256 for value in evaluations
        )
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("adaptive waves repeat an evaluation")
        merged.append(
            MaterializedActionEvaluationBatch(
                proposal_sha256=proposal.proposal_sha256,
                expert_id=proposal.expert_id,
                expert_version=proposal.expert_version,
                expert_definition_sha256=(
                    proposal.expert_definition_sha256
                ),
                selected_action_sha256s=action_ids,
                evaluations=evaluations,
                evidence=freeze_json(
                    {
                        "adaptive_wave_batch_sha256s": [
                            value.batch_sha256 for value in source
                        ],
                        "candidate_outcomes_reused": False,
                        "disjoint_action_waves": True,
                    }
                ),
            )
        )
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class OutcomeAdaptivePairedAuditExecution:
    """One extra real arm excluded from the authoritative K-budget archive."""

    plan: SamePrefixPairedAuditPlan
    observation: SamePrefixPairedAuditObservation
    counterfactual_evaluation_batches: tuple[
        MaterializedActionEvaluationBatch, ...
    ]
    plan_receipt_sha256: str
    execution_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not SamePrefixPairedAuditPlan:
            raise TypeError("plan must be exact")
        self.plan.__post_init__()
        if type(self.observation) is not SamePrefixPairedAuditObservation:
            raise TypeError("observation must be exact")
        self.observation.__post_init__()
        if self.observation.plan != self.plan:
            raise ValueError("paired audit observation changed its frozen plan")
        if (
            type(self.counterfactual_evaluation_batches) is not tuple
            or not self.counterfactual_evaluation_batches
        ):
            raise ValueError(
                "counterfactual_evaluation_batches must be non-empty"
            )
        evaluations: list[MaterializedActionEvaluation] = []
        for value in self.counterfactual_evaluation_batches:
            if type(value) is not MaterializedActionEvaluationBatch:
                raise TypeError(
                    "counterfactual evaluation batches must be exact"
                )
            value.__post_init__()
            evaluations.extend(value.evaluations)
        if len(evaluations) != 1:
            raise ValueError("paired audit must buy exactly one extra action")
        expected_action_sha256 = (
            self.plan.exploration_action_sha256
            if self.plan.authoritative_arm
            is SamePrefixPairedAuditArm.LEGACY
            else self.plan.legacy_action_sha256
        )
        evaluation = evaluations[0]
        if evaluation.action.action_sha256 != expected_action_sha256:
            raise ValueError(
                "counterfactual evaluation differs from the unselected arm"
            )
        expected_outcome = (
            self.observation.exploration_outcome
            if self.plan.authoritative_arm
            is SamePrefixPairedAuditArm.LEGACY
            else self.observation.legacy_outcome
        )
        if expected_outcome.evaluation_sha256 != evaluation.evaluation_sha256:
            raise ValueError(
                "counterfactual outcome does not join its real evaluation"
            )
        require_sha256(
            self.plan_receipt_sha256,
            "plan_receipt_sha256",
        )
        object.__setattr__(
            self,
            "execution_sha256",
            _hash(
                _PAIRED_AUDIT_EXECUTION_DOMAIN,
                self._unsigned_record(),
            ),
        )

    @property
    def counterfactual_action_sha256(self) -> str:
        if self.plan.authoritative_arm is SamePrefixPairedAuditArm.LEGACY:
            return self.plan.exploration_action_sha256
        return self.plan.legacy_action_sha256

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_sha256": self.plan.plan_sha256,
            "observation_sha256": self.observation.observation_sha256,
            "counterfactual_evaluation_batch_sha256s": [
                value.batch_sha256
                for value in self.counterfactual_evaluation_batches
            ],
            "counterfactual_action_sha256": (
                self.counterfactual_action_sha256
            ),
            "plan_receipt_sha256": self.plan_receipt_sha256,
            "physical_extra_real_evaluation_count": 1,
            "assay_union_admitted_to_authoritative_archive": False,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "plan": self.plan.to_record(include_evidence=include_evidence),
            "observation": self.observation.to_record(
                include_evidence=include_evidence
            ),
            "counterfactual_evaluation_batches": [
                value.to_record(include_evidence=include_evidence)
                for value in self.counterfactual_evaluation_batches
            ],
            "execution_sha256": self.execution_sha256,
        }


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualPortfolioEvolutionResult:
    """Complete causal trace plus the standard downstream result."""

    result: ResidualPortfolioEvolutionResult
    prior_broker_decision: MaterializedActionBrokerDecision
    adaptive_actions: tuple[AdaptiveActionDescriptor, ...]
    diagnostic_decision: AdaptiveActionRacingDecision
    continuation_decisions: tuple[AdaptiveActionRacingDecision, ...]
    evaluation_waves: tuple[
        tuple[MaterializedActionEvaluationBatch, ...], ...
    ]
    outcomes: tuple[AdaptiveActionOutcome, ...]
    set_outcomes: tuple[AdaptiveActionSetOutcome, ...]
    diagnostic_joint_gain: float
    allocation_directive: AdaptiveActionAllocationDirective
    phase_receipts: tuple[OutcomeAdaptiveResidualPhaseReceipt, ...]
    phase_commit_acks: tuple[
        OutcomeAdaptiveResidualPhaseCommitAck, ...
    ]
    market_projector_definition_sha256: str
    market_projector_state_sha256: str
    outcome_projector_definition_sha256: str
    forecast_opportunity_challenger_definition_sha256: str | None = None
    paired_audit_executions: tuple[
        OutcomeAdaptivePairedAuditExecution, ...
    ] = ()
    forecast_opportunity_calibration_observations: tuple[
        ArchiveOpportunityCalibrationObservation, ...
    ] = ()
    method_definition_sha256: str = (
        OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256
    )
    adaptive_result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.result) is not ResidualPortfolioEvolutionResult:
            raise TypeError("result must be exact")
        self.result.__post_init__()
        if (
            type(self.prior_broker_decision)
            is not MaterializedActionBrokerDecision
        ):
            raise TypeError("prior_broker_decision must be exact")
        self.prior_broker_decision.__post_init__()
        if (
            type(self.adaptive_actions) is not tuple
            or not self.adaptive_actions
        ):
            raise ValueError("adaptive_actions must be non-empty")
        action_by_sha: dict[str, AdaptiveActionDescriptor] = {}
        for value in self.adaptive_actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("adaptive_actions must be exact")
            value.__post_init__()
            action_by_sha[value.action_sha256] = value
        if len(action_by_sha) != len(self.adaptive_actions):
            raise ValueError("adaptive_actions repeat an identity")
        if type(self.diagnostic_decision) is not AdaptiveActionRacingDecision:
            raise TypeError("diagnostic_decision must be exact")
        self.diagnostic_decision.__post_init__()
        if self.diagnostic_decision.wave is not AdaptiveActionWave.DIAGNOSTIC:
            raise ValueError("first decision must be diagnostic")
        if type(self.continuation_decisions) is not tuple:
            raise TypeError("continuation_decisions must be a tuple")
        for value in self.continuation_decisions:
            if type(value) is not AdaptiveActionRacingDecision:
                raise TypeError("continuation decisions must be exact")
            value.__post_init__()
            if value.wave is AdaptiveActionWave.DIAGNOSTIC:
                raise ValueError("continuation cannot be diagnostic")
        if (
            type(self.evaluation_waves) is not tuple
            or len(self.evaluation_waves)
            != 1 + len(self.continuation_decisions)
        ):
            raise ValueError(
                "evaluation waves must cover diagnostic and continuations"
            )
        evaluation_by_action: dict[str, MaterializedActionEvaluation] = {}
        for wave in self.evaluation_waves:
            if type(wave) is not tuple or not wave:
                raise ValueError("each evaluation wave must be non-empty")
            for batch in wave:
                if type(batch) is not MaterializedActionEvaluationBatch:
                    raise TypeError("evaluation waves must contain exact batches")
                batch.__post_init__()
                for evaluation in batch.evaluations:
                    action_sha = evaluation.action.action_sha256
                    if action_sha in evaluation_by_action:
                        raise ValueError("evaluation waves repeat an action")
                    evaluation_by_action[action_sha] = evaluation
        if type(self.outcomes) is not tuple:
            raise TypeError("outcomes must be a tuple")
        outcome_by_action: dict[str, AdaptiveActionOutcome] = {}
        for value in self.outcomes:
            if type(value) is not AdaptiveActionOutcome:
                raise TypeError("outcomes must be exact")
            value.__post_init__()
            if value.action_sha256 in outcome_by_action:
                raise ValueError("outcomes repeat an action")
            evaluation = evaluation_by_action.get(value.action_sha256)
            if (
                evaluation is None
                or value.evaluation_sha256
                != evaluation.evaluation_sha256
            ):
                raise ValueError("outcome does not join its real evaluation")
            outcome_by_action[value.action_sha256] = value
        if (
            type(self.set_outcomes) is not tuple
            or len(self.set_outcomes) != len(self.evaluation_waves)
        ):
            raise ValueError(
                "set outcomes must cover every causal evaluation wave"
            )
        for value in self.set_outcomes:
            if type(value) is not AdaptiveActionSetOutcome:
                raise TypeError("set outcomes must be exact")
            value.__post_init__()
        selected = set(self.diagnostic_decision.selected_action_sha256s)
        diagnostic_evaluated = {
            value.action.action_sha256
            for value in _flatten(self.evaluation_waves[0])
        }
        if diagnostic_evaluated != selected:
            raise ValueError("diagnostic evaluations differ from decision")
        if not selected.issubset(outcome_by_action):
            raise ValueError("diagnostic outcomes are incomplete")
        for ordinal, decision in enumerate(
            self.continuation_decisions,
            start=1,
        ):
            if decision.prior_selected_action_sha256s != tuple(
                sorted(selected)
            ):
                raise ValueError("continuation decision skips causal prefix")
            expected_outcomes = tuple(
                sorted(
                    outcome_by_action[value].outcome_sha256
                    for value in selected
                )
            )
            if decision.observed_outcome_sha256s != expected_outcomes:
                raise ValueError(
                    "continuation decision observed a foreign outcome cutoff"
                )
            expected_set_outcomes = tuple(
                sorted(
                    value.set_outcome_sha256
                    for value in self.set_outcomes[:ordinal]
                )
            )
            if (
                decision.observed_set_outcome_sha256s
                != expected_set_outcomes
            ):
                raise ValueError(
                    "continuation decision observed a foreign set cutoff"
                )
            evaluated = {
                value.action.action_sha256
                for value in _flatten(self.evaluation_waves[ordinal])
            }
            if evaluated != set(decision.selected_action_sha256s):
                raise ValueError("continuation wave differs from decision")
            selected.update(decision.selected_action_sha256s)
        if selected != set(outcome_by_action):
            raise ValueError("outcome set differs from selected causal chain")
        if type(self.paired_audit_executions) is not tuple:
            raise TypeError("paired_audit_executions must be a tuple")
        if self.forecast_opportunity_challenger_definition_sha256 is not None:
            require_sha256(
                self.forecast_opportunity_challenger_definition_sha256,
                "forecast_opportunity_challenger_definition_sha256",
            )
        continuation_by_sha256 = {
            value.decision_sha256: value
            for value in self.continuation_decisions
        }
        paired_execution_by_decision: dict[
            str, OutcomeAdaptivePairedAuditExecution
        ] = {}
        for value in self.paired_audit_executions:
            if type(value) is not OutcomeAdaptivePairedAuditExecution:
                raise TypeError("paired audit executions must be exact")
            value.__post_init__()
            decision_sha256 = value.plan.racing_decision_sha256
            if decision_sha256 in paired_execution_by_decision:
                raise ValueError(
                    "paired audit executions repeat a racing decision"
                )
            decision = continuation_by_sha256.get(decision_sha256)
            is_forecast_audit = (
                value.plan.designer_id
                in FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS
            )
            if (
                decision is None
                or (
                    not is_forecast_audit
                    and decision.wave
                    is not AdaptiveActionWave.RANDOMIZED_AUDIT
                )
                or decision.selected_action_sha256s
                != (value.plan.authoritative_action_sha256,)
                or decision.prior_selected_action_sha256s
                != value.plan.common_prefix_action_sha256s
            ):
                raise ValueError(
                    "paired audit execution does not join its racing decision"
                )
            if is_forecast_audit != (
                self.forecast_opportunity_challenger_definition_sha256
                is not None
            ):
                raise ValueError(
                    "paired assay role differs from the method execution mode"
                )
            authoritative_evaluation = evaluation_by_action.get(
                value.plan.authoritative_action_sha256
            )
            authoritative_outcome = (
                value.observation.legacy_outcome
                if value.plan.authoritative_arm
                is SamePrefixPairedAuditArm.LEGACY
                else value.observation.exploration_outcome
            )
            if (
                authoritative_evaluation is None
                or authoritative_outcome.evaluation_sha256
                != authoritative_evaluation.evaluation_sha256
            ):
                raise ValueError(
                    "paired audit authoritative arm is not the selected "
                    "real evaluation"
                )
            if value.counterfactual_action_sha256 in evaluation_by_action:
                raise ValueError(
                    "paired audit counterfactual entered the authoritative "
                    "evaluation waves"
                )
            paired_execution_by_decision[decision_sha256] = value
        if (
            type(self.forecast_opportunity_calibration_observations)
            is not tuple
        ):
            raise TypeError(
                "forecast opportunity calibration observations must be a tuple"
            )
        expected_calibration_observations: list[
            ArchiveOpportunityCalibrationObservation
        ] = []
        calibration_projector = (
            ForecastOpportunityShadowCalibrationProjector()
        )
        for execution in self.paired_audit_executions:
            if (
                execution.plan.designer_id
                not in FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS
            ):
                continue
            decision = continuation_by_sha256[
                execution.plan.racing_decision_sha256
            ]
            expected_calibration_observations.append(
                calibration_projector.project(
                    decision_index=self.result.request.decision_index,
                    evidence_cutoff_ordinal=(
                        self.result.request.decision_index
                    ),
                    decision=decision,
                    action=action_by_sha[
                        execution.plan.exploration_action_sha256
                    ],
                    observation=execution.observation,
                )
            )
        for value in self.forecast_opportunity_calibration_observations:
            if type(value) is not ArchiveOpportunityCalibrationObservation:
                raise TypeError(
                    "forecast opportunity calibration observations must "
                    "be exact"
                )
            value.__post_init__()
        if self.forecast_opportunity_calibration_observations != tuple(
            expected_calibration_observations
        ):
            raise ValueError(
                "forecast calibration observations do not close shadow assays"
            )
        prior_bindings: list[tuple[str, str]] = []
        prior_augmented_gain = 0.0
        for ordinal, (wave, set_outcome) in enumerate(
            zip(self.evaluation_waves, self.set_outcomes, strict=True)
        ):
            expected_current_bindings = tuple(
                sorted(
                    (
                        value.action.action_sha256,
                        value.evaluation_sha256,
                    )
                    for value in _flatten(wave)
                )
            )
            if (
                set_outcome.prior_action_evaluation_bindings
                != tuple(sorted(prior_bindings))
                or set_outcome.current_action_evaluation_bindings
                != expected_current_bindings
            ):
                raise ValueError(
                    "set outcome does not join its causal evaluation prefix"
                )
            if not math.isclose(
                set_outcome.prior_selected_set_gain,
                prior_augmented_gain,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "set outcome utility chain is not causally contiguous"
                )
            if ordinal == 0 and (
                set_outcome.prior_action_evaluation_bindings
                or not math.isclose(
                    set_outcome.conditional_set_gain,
                    set_outcome.current_wave_fixed_set_gain,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "diagnostic set outcome must start from an empty prefix"
                )
            prior_bindings.extend(expected_current_bindings)
            prior_augmented_gain = (
                set_outcome.augmented_selected_set_gain
            )
        if (
            type(self.diagnostic_joint_gain) is not float
            or not math.isfinite(self.diagnostic_joint_gain)
            or self.diagnostic_joint_gain < 0.0
        ):
            raise ValueError("diagnostic_joint_gain must be non-negative")
        if not math.isclose(
            self.diagnostic_joint_gain,
            self.set_outcomes[0].current_wave_fixed_set_gain,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "diagnostic joint gain differs from its set observation"
            )
        if (
            type(self.allocation_directive)
            is not AdaptiveActionAllocationDirective
        ):
            raise TypeError("allocation_directive must be exact")
        self.allocation_directive.__post_init__()
        if (
            self.allocation_directive.diagnostic_decision_sha256
            != self.diagnostic_decision.decision_sha256
            or self.allocation_directive.continuation_decision_sha256s
            != tuple(
                sorted(
                    value.decision_sha256
                    for value in self.continuation_decisions
                )
            )
            or self.allocation_directive.observed_outcome_sha256s
            != tuple(
                sorted(value.outcome_sha256 for value in self.outcomes)
            )
            or self.allocation_directive.observed_set_outcome_sha256s
            != tuple(
                sorted(
                    value.set_outcome_sha256
                    for value in self.set_outcomes
                )
            )
            or set(self.allocation_directive.required_action_sha256s)
            != selected
        ):
            raise ValueError("allocation directive does not close causal trace")
        if self.forecast_opportunity_challenger_definition_sha256 is not None:
            if (
                self.allocation_directive.policy_definition_sha256
                != self.forecast_opportunity_challenger_definition_sha256
                or any(
                    value.policy_definition_sha256
                    != self.forecast_opportunity_challenger_definition_sha256
                    for value in self.continuation_decisions
                )
            ):
                raise ValueError(
                    "forecast challenger identity does not cover every "
                    "continuation"
                )
        if (
            self.result.broker_decision.allocation_requirement
            != self.allocation_directive
            or {
                value.action_sha256
                for value in self.result.broker_decision.selected_actions
            }
            != selected
        ):
            raise ValueError("final broker result differs from directive")
        if (
            type(self.phase_receipts) is not tuple
            or len(self.phase_receipts)
            != (
                3
                + len(self.continuation_decisions)
                + len(self.paired_audit_executions)
            )
        ):
            raise ValueError("phase receipts do not cover causal execution")
        expected_phase_values = [
            OutcomeAdaptiveResidualPhase.MARKET_FROZEN,
            OutcomeAdaptiveResidualPhase.DIAGNOSTIC_EVALUATED,
        ]
        for decision in self.continuation_decisions:
            if decision.decision_sha256 in paired_execution_by_decision:
                expected_phase_values.append(
                    OutcomeAdaptiveResidualPhase.PAIRED_AUDIT_FROZEN
                )
            expected_phase_values.append(
                OutcomeAdaptiveResidualPhase.ADAPTIVE_STEP_EVALUATED
            )
        expected_phase_values.append(
            OutcomeAdaptiveResidualPhase.FINALIZED
        )
        expected_phases = tuple(expected_phase_values)
        if tuple(value.phase for value in self.phase_receipts) != expected_phases:
            raise ValueError("phase receipts are out of causal order")
        if tuple(
            value.phase_ordinal for value in self.phase_receipts
        ) != tuple(range(1, len(self.phase_receipts) + 1)):
            raise ValueError("phase receipt ordinals are not contiguous")
        for value in self.phase_receipts:
            value.__post_init__()
            if (
                value.residual_request_sha256
                != self.result.request.request_sha256
                or value.diagnostic_decision_sha256
                != self.diagnostic_decision.decision_sha256
            ):
                raise ValueError("phase receipt names another execution")
        receipt_by_sha256 = {
            value.receipt_sha256: value for value in self.phase_receipts
        }
        for execution in self.paired_audit_executions:
            receipt = receipt_by_sha256.get(execution.plan_receipt_sha256)
            if (
                receipt is None
                or receipt.phase
                is not OutcomeAdaptiveResidualPhase.PAIRED_AUDIT_FROZEN
                or thaw_json(receipt.evidence).get("plan_sha256")
                != execution.plan.plan_sha256
            ):
                raise ValueError(
                    "paired audit plan lacks its pre-evaluation freeze receipt"
                )
        if type(self.phase_commit_acks) is not tuple:
            raise TypeError("phase_commit_acks must be a tuple")
        for value in self.phase_commit_acks:
            value.__post_init__()
        if self.phase_commit_acks and tuple(
            value.phase_receipt_sha256
            for value in self.phase_commit_acks
        ) != tuple(
            value.receipt_sha256 for value in self.phase_receipts
        ):
            raise ValueError("phase commit acknowledgements do not close")
        for name in (
            "market_projector_definition_sha256",
            "market_projector_state_sha256",
            "outcome_projector_definition_sha256",
            "method_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        has_forecast = (
            self.forecast_opportunity_challenger_definition_sha256
            is not None
        )
        has_paired_assay = bool(self.paired_audit_executions)
        has_forecast_stratified_audit = any(
            value.plan.designer_id
            == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            for value in self.paired_audit_executions
        )
        expected_method_definition_sha256 = (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256
            if has_forecast_stratified_audit
            else (
                OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256
                if has_forecast and has_paired_assay
                else (
                    _FORECAST_OPPORTUNITY_DEFINITION_SHA256
                    if has_forecast
                    else (
                        _PAIRED_AUDIT_DEFINITION_SHA256
                        if has_paired_assay
                        else OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256
                    )
                )
            )
        )
        if (
            self.method_definition_sha256
            != expected_method_definition_sha256
        ):
            raise ValueError(
                "method definition does not match paired-audit execution mode"
            )
        object.__setattr__(
            self,
            "adaptive_result_sha256",
            _hash(
                (
                    _FORECAST_STRATIFIED_AUDIT_RESULT_DOMAIN
                    if has_forecast_stratified_audit
                    else (
                        _FORECAST_SHADOW_RESULT_DOMAIN
                        if has_forecast and has_paired_assay
                        else (
                            _FORECAST_OPPORTUNITY_RESULT_DOMAIN
                            if has_forecast
                            else (
                                _PAIRED_AUDIT_RESULT_DOMAIN
                                if has_paired_assay
                                else _RESULT_DOMAIN
                            )
                        )
                    )
                ),
                self._unsigned_record(),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        has_forecast = (
            self.forecast_opportunity_challenger_definition_sha256
            is not None
        )
        has_paired_assay = bool(self.paired_audit_executions)
        has_forecast_stratified_audit = any(
            value.plan.designer_id
            == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
            for value in self.paired_audit_executions
        )
        record: dict[str, object] = {
            "schema_version": (
                8
                if has_forecast_stratified_audit
                else (
                    7
                    if has_forecast and has_paired_assay
                    else (
                        6
                        if has_forecast
                        else (5 if has_paired_assay else 4)
                    )
                )
            ),
            "method": {
                "method_id": OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_ID,
                "method_version": (
                    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_VERSION
                    if has_forecast_stratified_audit
                    else (
                        OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_VERSION
                        if has_forecast and has_paired_assay
                        else (
                            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_VERSION
                            if has_forecast
                            else (
                                OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_VERSION
                                if has_paired_assay
                                else OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_VERSION
                            )
                        )
                    )
                ),
                "definition_sha256": self.method_definition_sha256,
            },
            "combined_result_sha256": self.result.result_sha256,
            "prior_broker_decision_sha256": (
                self.prior_broker_decision.decision_sha256
            ),
            "adaptive_action_sha256s": [
                value.action_sha256 for value in self.adaptive_actions
            ],
            "diagnostic_decision_sha256": (
                self.diagnostic_decision.decision_sha256
            ),
            "continuation_decision_sha256s": [
                value.decision_sha256
                for value in self.continuation_decisions
            ],
            "evaluation_wave_batch_sha256s": [
                [value.batch_sha256 for value in wave]
                for wave in self.evaluation_waves
            ],
            "outcome_sha256s": [
                value.outcome_sha256 for value in self.outcomes
            ],
            "set_outcome_sha256s": [
                value.set_outcome_sha256 for value in self.set_outcomes
            ],
            "diagnostic_joint_gain_hex": self.diagnostic_joint_gain.hex(),
            "allocation_directive_sha256": (
                self.allocation_directive.directive_sha256
            ),
            "phase_receipt_sha256s": [
                value.receipt_sha256 for value in self.phase_receipts
            ],
            "phase_commit_ack_sha256s": [
                value.ack_sha256 for value in self.phase_commit_acks
            ],
            "market_projector_definition_sha256": (
                self.market_projector_definition_sha256
            ),
            "market_projector_state_sha256": (
                self.market_projector_state_sha256
            ),
            "outcome_projector_definition_sha256": (
                self.outcome_projector_definition_sha256
            ),
            "current_outcomes_used_only_after_interception": True,
            "workload_objective_model_provider_prompt_config_branches": False,
        }
        if self.paired_audit_executions:
            record.update(
                {
                    "paired_audit_execution_sha256s": [
                        value.execution_sha256
                        for value in self.paired_audit_executions
                    ],
                    "paired_audit_physical_extra_real_evaluation_count": len(
                        self.paired_audit_executions
                    ),
                    "paired_audit_union_admitted_to_authoritative_archive": (
                        False
                    ),
                }
            )
        if has_forecast_stratified_audit:
            record.update(
                {
                    "forecast_stratified_same_prefix_audit": True,
                    "counterfactual_action_quarantine_enforced": True,
                    "counterfactual_quarantined_action_sha256s": sorted(
                        value.counterfactual_action_sha256
                        for value in self.paired_audit_executions
                        if value.plan.designer_id
                        == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
                    ),
                    "audit_position_schedule": (
                        "one-hash-uniform-continuation-position-per-request"
                    ),
                }
            )
        if self.forecast_opportunity_calibration_observations:
            record[
                "forecast_opportunity_calibration_observation_sha256s"
            ] = [
                value.observation_sha256
                for value in (
                    self.forecast_opportunity_calibration_observations
                )
            ]
        if self.forecast_opportunity_challenger_definition_sha256 is not None:
            record.update(
                {
                    "forecast_opportunity_challenger_definition_sha256": (
                        self.forecast_opportunity_challenger_definition_sha256
                    ),
                    "forecast_geometry_frozen_before_diagnostic_outcomes": True,
                    "current_prefix_candidate_outcomes_used_for_selection": True,
                    "eligible_candidate_outcomes_observed": False,
                    "fallback_preserved_on_abstention": True,
                }
            )
        return record

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "adaptive_actions": [
                value.to_record() for value in self.adaptive_actions
            ],
            "diagnostic_decision": self.diagnostic_decision.to_record(
                include_evidence=include_evidence
            ),
            "continuation_decisions": [
                value.to_record(include_evidence=include_evidence)
                for value in self.continuation_decisions
            ],
            "evaluation_waves": [
                [
                    value.to_record(include_evidence=include_evidence)
                    for value in wave
                ]
                for wave in self.evaluation_waves
            ],
            "outcomes": [value.to_record() for value in self.outcomes],
            "set_outcomes": [
                value.to_record() for value in self.set_outcomes
            ],
            "allocation_directive": self.allocation_directive.to_record(
                include_evidence=include_evidence
            ),
            "phase_receipts": [
                value.to_record(include_evidence=include_evidence)
                for value in self.phase_receipts
            ],
            "phase_commit_acks": [
                value.to_record(include_evidence=include_evidence)
                for value in self.phase_commit_acks
            ],
            "combined_result": self.result.to_record(
                include_allocation_evidence=include_evidence
            ),
            "adaptive_result_sha256": self.adaptive_result_sha256,
        }
        if self.paired_audit_executions:
            record["paired_audit_executions"] = [
                value.to_record(include_evidence=include_evidence)
                for value in self.paired_audit_executions
            ]
        if self.forecast_opportunity_calibration_observations:
            record["forecast_opportunity_calibration_observations"] = [
                value.to_record()
                for value in (
                    self.forecast_opportunity_calibration_observations
                )
            ]
        return record


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveResidualPortfolioEvolution:
    """Execute diagnostic and sequential actions from one sealed market."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    broker: RegretBrokeredMaterializedActionPolicy
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

    def __post_init__(self) -> None:
        if type(self.experts) is not tuple or not self.experts:
            raise ValueError("experts must be a non-empty exact tuple")
        if any(
            getattr(value, "evaluation_wave_semantics", None)
            != DISJOINT_ACTION_EVALUATION_WAVES_V1
            for value in self.experts
        ):
            raise ValueError(
                "adaptive experts must support disjoint action waves"
            )
        if type(self.broker) is not RegretBrokeredMaterializedActionPolicy:
            raise TypeError("broker must be an exact regret-broker policy")
        self.broker.__post_init__()
        if type(self.racing_policy) is not OutcomeAdaptiveActionRacingPolicy:
            # Structural admission for injected allocation controllers:
            # the policy must expose the exact racing surface (identity
            # triple plus both decision methods).  Version-gated
            # features (causal set outcomes, paired audits) still key
            # off policy_version, so a foreign controller only reaches
            # the plain observational path.
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
        self.racing_policy.__post_init__()
        if not isinstance(
            self.market_projector,
            AdaptiveActionMarketProjectorPort,
        ):
            raise TypeError("market_projector must implement its port")
        _require_token(
            self.market_projector.projector_id,
            name="market projector_id",
        )
        if (
            type(self.market_projector.projector_version) is not int
            or self.market_projector.projector_version <= 0
        ):
            raise ValueError("market projector_version must be positive")
        require_sha256(
            self.market_projector.definition_sha256,
            "market projector definition",
        )
        require_sha256(
            self.market_projector.state_sha256,
            "market projector state",
        )
        if not isinstance(
            self.outcome_projector,
            AdaptiveActionOutcomeProjectorPort,
        ):
            raise TypeError("outcome_projector must implement its port")
        _require_token(
            self.outcome_projector.projector_id,
            name="outcome projector_id",
        )
        if (
            type(self.outcome_projector.projector_version) is not int
            or self.outcome_projector.projector_version <= 0
        ):
            raise ValueError("outcome projector_version must be positive")
        require_sha256(
            self.outcome_projector.definition_sha256,
            "outcome projector definition",
        )
        if not isinstance(self.slate_value, MaterializedSlateValuePort):
            raise TypeError("slate_value must implement its port")
        if not isinstance(
            self.slate_feasibility,
            MaterializedSlateFeasibilityPort,
        ):
            raise TypeError("slate_feasibility must implement its port")
        require_sha256(
            self.slate_value.definition_sha256,
            "slate value definition",
        )
        require_sha256(
            self.slate_feasibility.definition_sha256,
            "slate feasibility definition",
        )
        if self.diagnostic_allocation_policy is not None:
            if not isinstance(
                self.diagnostic_allocation_policy,
                MaterializedActionAllocationPolicyPort,
            ):
                raise TypeError(
                    "diagnostic_allocation_policy must implement its port"
                )
            _require_token(
                self.diagnostic_allocation_policy.policy_id,
                name="diagnostic allocation policy_id",
            )
            if (
                type(self.diagnostic_allocation_policy.policy_version)
                is not int
                or self.diagnostic_allocation_policy.policy_version <= 0
            ):
                raise ValueError(
                    "diagnostic allocation policy_version must be positive"
                )
            require_sha256(
                self.diagnostic_allocation_policy.definition_sha256,
                "diagnostic allocation policy definition",
            )
        if self.phase_committer is not None:
            if not isinstance(
                self.phase_committer,
                OutcomeAdaptiveResidualPhaseCommitPort,
            ):
                raise TypeError("phase_committer must implement its port")
            _require_token(
                self.phase_committer.committer_id,
                name="committer_id",
            )
            if (
                type(self.phase_committer.committer_version) is not int
                or self.phase_committer.committer_version <= 0
            ):
                raise ValueError("committer_version must be positive")
            require_sha256(
                self.phase_committer.definition_sha256,
                "committer definition",
            )
        if self.paired_audit_designer is not None:
            if not isinstance(
                self.paired_audit_designer,
                SamePrefixPairedAuditDesignerPort,
            ):
                raise TypeError(
                    "paired_audit_designer must implement its port"
                )
            _require_token(
                self.paired_audit_designer.designer_id,
                name="paired audit designer_id",
            )
            if (
                type(self.paired_audit_designer.designer_version) is not int
                or self.paired_audit_designer.designer_version <= 0
            ):
                raise ValueError(
                    "paired audit designer_version must be positive"
                )
            require_sha256(
                self.paired_audit_designer.definition_sha256,
                "paired audit designer definition",
            )
            if (
                self.racing_policy.policy_version
                != (
                    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
                )
            ):
                raise ValueError(
                    "paired audit interception requires racing policy v6"
                )
        if self.forecast_opportunity_challenger is not None:
            if (
                type(self.forecast_opportunity_challenger)
                is not ProtectedCurrentPrefixForecastOpportunityChallenger
            ):
                raise TypeError(
                    "forecast_opportunity_challenger must be exact"
                )
            self.forecast_opportunity_challenger.__post_init__()
            if self.paired_audit_designer is not None:
                raise ValueError(
                    "forecast challenger and paired audit cannot share one "
                    "authoritative execution"
                )
            if (
                self.forecast_opportunity_challenger.fallback_policy_id
                != self.racing_policy.policy_id
                or self.forecast_opportunity_challenger
                .fallback_policy_version
                != self.racing_policy.policy_version
                or self.forecast_opportunity_challenger
                .fallback_policy_definition_sha256
                != self.racing_policy.definition_sha256
            ):
                raise ValueError(
                    "forecast challenger fallback differs from racing policy"
                )
        if self.forecast_opportunity_shadow_designer is not None:
            if self.forecast_opportunity_challenger is None:
                raise ValueError(
                    "forecast shadow designer requires a forecast challenger"
                )
            if self.paired_audit_designer is not None:
                raise ValueError(
                    "forecast shadow and stratified audit designers are "
                    "mutually exclusive"
                )
            if not isinstance(
                self.forecast_opportunity_shadow_designer,
                ForecastOpportunitySamePrefixShadowDesignerPort,
            ):
                raise TypeError(
                    "forecast_opportunity_shadow_designer must implement "
                    "its port"
                )
            _require_token(
                self.forecast_opportunity_shadow_designer.designer_id,
                name="forecast shadow designer_id",
            )
            if (
                type(
                    self.forecast_opportunity_shadow_designer.designer_version
                )
                is not int
                or (
                    self.forecast_opportunity_shadow_designer.designer_version
                    <= 0
                )
            ):
                raise ValueError(
                    "forecast shadow designer_version must be positive"
                )
            require_sha256(
                self.forecast_opportunity_shadow_designer.definition_sha256,
                "forecast shadow designer definition",
            )
        if type(self.require_durable_phase_commits) is not bool:
            raise TypeError("require_durable_phase_commits must be exact")
        if (
            self.require_durable_phase_commits
            and self.phase_committer is None
        ):
            raise ValueError(
                "required durable phase commits need a committer"
            )

    async def _commit(
        self,
        receipt: OutcomeAdaptiveResidualPhaseReceipt,
    ) -> OutcomeAdaptiveResidualPhaseCommitAck | None:
        if self.phase_committer is None:
            return None
        ack = await self.phase_committer.commit(receipt)
        if type(ack) is not OutcomeAdaptiveResidualPhaseCommitAck:
            raise TypeError("phase committer returned a foreign ack")
        ack.__post_init__()
        if (
            ack.committer_id != self.phase_committer.committer_id
            or ack.committer_version
            != self.phase_committer.committer_version
            or ack.committer_definition_sha256
            != self.phase_committer.definition_sha256
            or ack.phase_receipt_sha256 != receipt.receipt_sha256
        ):
            raise ValueError("phase commit ack differs from request")
        if self.require_durable_phase_commits and not ack.durable:
            raise ValueError("required phase commit was not durable")
        return ack

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> OutcomeAdaptiveResidualPortfolioEvolutionResult:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        uses_causal_set_outcomes = (
            self.racing_policy.policy_version
            in {
                OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
                OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
            }
        )
        proposals = await propose_materialized_action_batches(
            experts=self.experts,
            request=request,
        )
        forecast_geometry = (
            None
            if self.forecast_opportunity_challenger is None
            else await self.forecast_opportunity_challenger.freeze_geometry(
                request=request,
                proposals=proposals,
            )
        )
        actions = tuple(
            action
            for proposal in proposals
            for action in proposal.actions
        )
        diagnostic_requirement = None
        if self.diagnostic_allocation_policy is not None:
            diagnostic_requirement = (
                await self.diagnostic_allocation_policy.require(
                    request,
                    proposals,
                )
            )
            if (
                type(diagnostic_requirement)
                is not MaterializedActionAllocationRequirement
            ):
                raise TypeError(
                    "diagnostic allocation policy returned a foreign "
                    "requirement"
                )
            diagnostic_requirement.__post_init__()
            if (
                diagnostic_requirement.policy_id,
                diagnostic_requirement.policy_version,
                diagnostic_requirement.policy_definition_sha256,
            ) != (
                self.diagnostic_allocation_policy.policy_id,
                self.diagnostic_allocation_policy.policy_version,
                self.diagnostic_allocation_policy.definition_sha256,
            ):
                raise ValueError(
                    "diagnostic allocation requirement changed policy identity"
                )
            proposal_sha256s = tuple(
                sorted(value.proposal_sha256 for value in proposals)
            )
            if (
                diagnostic_requirement.residual_request_sha256
                != request.request_sha256
                or diagnostic_requirement.proposal_sha256s
                != proposal_sha256s
            ):
                raise ValueError(
                    "diagnostic allocation requirement changed its "
                    "sealed cutoff"
                )
            if not set(
                diagnostic_requirement.required_action_sha256s
            ).issubset({value.action_sha256 for value in actions}):
                raise ValueError(
                    "diagnostic allocation selected outside the market"
                )
        prior_decision = self.broker.select(
            MaterializedActionBrokerRequest(
                actions=actions,
                evaluation_slots=request.evaluation_slots,
                slate_value=self.slate_value,
                slate_feasibility=self.slate_feasibility,
                reference_escrow_slots=request.reference_escrow_slots,
                allocation_requirement=diagnostic_requirement,
            )
        )
        fixed_actions = tuple(
            sorted(
                set(prior_decision.required_reference_action_sha256s)
                | (
                    set()
                    if diagnostic_requirement is None
                    else set(
                        diagnostic_requirement.required_action_sha256s
                    )
                )
                | (
                    set()
                    if prior_decision.exploration_requirement is None
                    else set(
                        prior_decision.exploration_requirement
                        .required_action_sha256s
                    )
                )
            )
        )
        adaptive_actions = await self.market_projector.project(
            request,
            proposals,
            actions,
            prior_decision.scores,
            fixed_actions,
        )
        if len(adaptive_actions) < request.evaluation_slots:
            raise ValueError(
                "phenotype-unique adaptive market cannot fill capacity"
            )
        adaptive_action_by_sha256 = {
            value.action_sha256: value for value in adaptive_actions
        }
        diagnostic = self.racing_policy.design_diagnostic_pilot(
            residual_request_sha256=request.request_sha256,
            actions=adaptive_actions,
            evaluation_slots=request.evaluation_slots,
            required_action_sha256s=fixed_actions,
        )
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        receipts: list[OutcomeAdaptiveResidualPhaseReceipt] = []
        acks: list[OutcomeAdaptiveResidualPhaseCommitAck] = []

        market_receipt = OutcomeAdaptiveResidualPhaseReceipt(
            phase=OutcomeAdaptiveResidualPhase.MARKET_FROZEN,
            phase_ordinal=1,
            residual_request_sha256=request.request_sha256,
            diagnostic_decision_sha256=diagnostic.decision_sha256,
            product_sha256s=(
                *proposal_sha256s,
                prior_decision.decision_sha256,
                diagnostic.decision_sha256,
                *(
                    ()
                    if forecast_geometry is None
                    else (forecast_geometry.batch_sha256,)
                ),
            ),
            evidence=freeze_json(
                {
                    "proposal_sha256s": list(proposal_sha256s),
                    "proposal_action_count": len(actions),
                    "adaptive_market_action_count": len(
                        adaptive_actions
                    ),
                    "fixed_action_sha256s": list(fixed_actions),
                    "prior_broker_decision": (
                        prior_decision.to_record(
                            include_allocation_evidence=True
                        )
                    ),
                    "adaptive_actions": [
                        value.to_record()
                        for value in adaptive_actions
                    ],
                    "diagnostic_decision": diagnostic.to_record(
                        include_evidence=True
                    ),
                    "market_projector": {
                        "projector_id": (
                            self.market_projector.projector_id
                        ),
                        "projector_version": (
                            self.market_projector.projector_version
                        ),
                        "definition_sha256": (
                            self.market_projector.definition_sha256
                        ),
                        "state_sha256": (
                            self.market_projector.state_sha256
                        ),
                    },
                    **(
                        {}
                        if forecast_geometry is None
                        else {
                            "forecast_opportunity_challenger": {
                                "challenger_id": (
                                    self.forecast_opportunity_challenger
                                    .challenger_id
                                ),
                                "challenger_version": (
                                    self.forecast_opportunity_challenger
                                    .challenger_version
                                ),
                                "definition_sha256": (
                                    self.forecast_opportunity_challenger
                                    .definition_sha256
                                ),
                            },
                            "forecast_geometry": (
                                forecast_geometry.to_record(
                                    include_evidence=True
                                )
                            ),
                            "forecast_geometry_frozen_before_"
                            "diagnostic_outcomes": True,
                        }
                    ),
                    "current_candidate_outcomes_observed": False,
                    "all_actions_materialized": True,
                }
            ),
        )
        receipts.append(market_receipt)
        ack = await self._commit(market_receipt)
        if ack is not None:
            acks.append(ack)

        diagnostic_batches = await evaluate_materialized_action_subset(
            experts=self.experts,
            proposals=proposals,
            selected_action_sha256s=(
                diagnostic.selected_action_sha256s
            ),
        )
        diagnostic_evaluations = _flatten(diagnostic_batches)
        outcomes = list(
            self.outcome_projector.project(
                diagnostic_evaluations
            )
        )
        diagnostic_joint_gain = self.outcome_projector.joint_gain(
            diagnostic_evaluations
        )
        diagnostic_set_outcome = (
            self.outcome_projector.project_set_outcome(
                (),
                diagnostic_evaluations,
            )
        )
        if not math.isclose(
            diagnostic_joint_gain,
            diagnostic_set_outcome.current_wave_fixed_set_gain,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "diagnostic joint utility projections disagree"
            )
        set_outcomes = [diagnostic_set_outcome]
        selected_evaluations = list(diagnostic_evaluations)
        waves = [diagnostic_batches]
        diagnostic_receipt = OutcomeAdaptiveResidualPhaseReceipt(
            phase=OutcomeAdaptiveResidualPhase.DIAGNOSTIC_EVALUATED,
            phase_ordinal=2,
            residual_request_sha256=request.request_sha256,
            diagnostic_decision_sha256=diagnostic.decision_sha256,
            product_sha256s=(
                *(
                    value.batch_sha256
                    for value in diagnostic_batches
                ),
                *(value.outcome_sha256 for value in outcomes),
                diagnostic_set_outcome.set_outcome_sha256,
            ),
            evidence=freeze_json(
                {
                    "diagnostic_evaluation_batches": [
                        value.to_record(include_evidence=True)
                        for value in diagnostic_batches
                    ],
                    "diagnostic_outcomes": [
                        value.to_record() for value in outcomes
                    ],
                    "diagnostic_joint_gain_hex": (
                        diagnostic_joint_gain.hex()
                    ),
                    "diagnostic_set_outcome": (
                        diagnostic_set_outcome.to_record()
                    ),
                    "causal_set_outcome_policy_role": (
                        "selection_input"
                        if uses_causal_set_outcomes
                        else "observational_only"
                    ),
                    "current_prefix_candidate_outcomes_policy_role": (
                        "selection_input"
                        if self.forecast_opportunity_challenger
                        is not None
                        else "observational_only"
                    ),
                    "later_candidate_outcomes_observed": False,
                    "next_adaptive_decision_exists": False,
                }
            ),
        )
        receipts.append(diagnostic_receipt)
        ack = await self._commit(diagnostic_receipt)
        if ack is not None:
            acks.append(ack)

        selected = set(diagnostic.selected_action_sha256s)
        continuation_decisions: list[AdaptiveActionRacingDecision] = []
        paired_audit_executions: list[
            OutcomeAdaptivePairedAuditExecution
        ] = []
        forecast_calibration_observations: list[
            ArchiveOpportunityCalibrationObservation
        ] = []
        counterfactual_quarantined_action_sha256s: set[str] = set()
        paired_audit_adjudicator = (
            None
            if (
                self.paired_audit_designer is None
                and self.forecast_opportunity_shadow_designer is None
            )
            else SamePrefixPairedAuditAdjudicator()
        )
        while len(selected) < request.evaluation_slots:
            excluded_action_sha256s = tuple(
                sorted(counterfactual_quarantined_action_sha256s)
            )
            fallback_decision = self.racing_policy.select_next(
                residual_request_sha256=request.request_sha256,
                actions=adaptive_actions,
                evaluation_slots=request.evaluation_slots,
                diagnostic_action_sha256s=(
                    diagnostic.selected_action_sha256s
                ),
                diagnostic_joint_gain=diagnostic_joint_gain,
                selected_action_sha256s=tuple(sorted(selected)),
                outcomes=tuple(outcomes),
                set_outcomes=tuple(set_outcomes),
                excluded_action_sha256s=excluded_action_sha256s,
            )
            decision = (
                fallback_decision
                if self.forecast_opportunity_challenger is None
                else self.forecast_opportunity_challenger.challenge(
                    fallback=fallback_decision,
                    geometry=forecast_geometry,
                    adaptive_actions=adaptive_actions,
                    selected_evaluations=tuple(
                        selected_evaluations
                    ),
                    excluded_action_sha256s=(
                        excluded_action_sha256s
                    ),
                )
            )
            continuation_decisions.append(decision)
            paired_plan: SamePrefixPairedAuditPlan | None = None
            paired_plan_receipt: (
                OutcomeAdaptiveResidualPhaseReceipt | None
            ) = None
            if (
                self.paired_audit_designer is not None
                and decision.wave is AdaptiveActionWave.RANDOMIZED_AUDIT
            ):
                paired_plan = self.paired_audit_designer.design(
                    decision=decision,
                    actions=adaptive_actions,
                )
            elif self.forecast_opportunity_shadow_designer is not None:
                paired_plan = (
                    self.forecast_opportunity_shadow_designer.design(
                        adaptive_step=len(continuation_decisions),
                        remaining_authoritative_slots_after_decision=(
                            request.evaluation_slots
                            - len(selected)
                            - len(decision.selected_action_sha256s)
                        ),
                        decision=decision,
                        fallback=fallback_decision,
                        actions=adaptive_actions,
                    )
                )
            if paired_plan is not None:
                paired_plan_receipt = OutcomeAdaptiveResidualPhaseReceipt(
                    phase=(
                        OutcomeAdaptiveResidualPhase.PAIRED_AUDIT_FROZEN
                    ),
                    phase_ordinal=len(receipts) + 1,
                    residual_request_sha256=request.request_sha256,
                    diagnostic_decision_sha256=(
                        diagnostic.decision_sha256
                    ),
                    product_sha256s=(
                        decision.decision_sha256,
                        paired_plan.plan_sha256,
                    ),
                    evidence=freeze_json(
                        {
                            "adaptive_step": len(
                                continuation_decisions
                            ),
                            "racing_decision_sha256": (
                                decision.decision_sha256
                            ),
                            "plan_sha256": paired_plan.plan_sha256,
                            "plan": paired_plan.to_record(
                                include_evidence=True
                            ),
                            "current_arm_outcomes_observed": False,
                            "plan_frozen_before_arm_evaluation": True,
                            "durable_commit_required": (
                                self.require_durable_phase_commits
                            ),
                            "assay_union_may_enter_authoritative_archive": (
                                False
                            ),
                        }
                    ),
                )
                receipts.append(paired_plan_receipt)
                ack = await self._commit(paired_plan_receipt)
                if ack is not None:
                    acks.append(ack)
            step_batches = await evaluate_materialized_action_subset(
                experts=self.experts,
                proposals=proposals,
                selected_action_sha256s=(
                    decision.selected_action_sha256s
                ),
            )
            step_evaluations = _flatten(step_batches)
            step_outcomes = self.outcome_projector.project(
                step_evaluations
            )
            step_set_outcome = (
                self.outcome_projector.project_set_outcome(
                    tuple(selected_evaluations),
                    step_evaluations,
                )
            )
            paired_execution: OutcomeAdaptivePairedAuditExecution | None = None
            forecast_calibration_observation: (
                ArchiveOpportunityCalibrationObservation | None
            ) = None
            if paired_plan is not None:
                if (
                    paired_plan_receipt is None
                    or paired_audit_adjudicator is None
                ):
                    raise RuntimeError(
                        "paired audit plan lacks its frozen execution context"
                    )
                counterfactual_action_sha256 = (
                    paired_plan.exploration_action_sha256
                    if paired_plan.authoritative_arm
                    is SamePrefixPairedAuditArm.LEGACY
                    else paired_plan.legacy_action_sha256
                )
                counterfactual_batches = (
                    await evaluate_materialized_action_counterfactual_subset(
                        experts=self.experts,
                        proposals=proposals,
                        selected_action_sha256s=(
                            counterfactual_action_sha256,
                        ),
                    )
                )
                counterfactual_evaluations = _flatten(
                    counterfactual_batches
                )
                counterfactual_outcomes = (
                    self.outcome_projector.project(
                        counterfactual_evaluations
                    )
                )
                counterfactual_set_outcome = (
                    self.outcome_projector.project_set_outcome(
                        tuple(selected_evaluations),
                        counterfactual_evaluations,
                    )
                )
                if (
                    len(step_outcomes) != 1
                    or len(counterfactual_outcomes) != 1
                ):
                    raise ValueError(
                        "paired audit arms must each contain one action"
                    )
                authoritative_outcome = step_outcomes[0]
                counterfactual_outcome = counterfactual_outcomes[0]
                if (
                    paired_plan.authoritative_arm
                    is SamePrefixPairedAuditArm.LEGACY
                ):
                    legacy_outcome = authoritative_outcome
                    legacy_set_outcome = step_set_outcome
                    exploration_outcome = counterfactual_outcome
                    exploration_set_outcome = (
                        counterfactual_set_outcome
                    )
                else:
                    legacy_outcome = counterfactual_outcome
                    legacy_set_outcome = counterfactual_set_outcome
                    exploration_outcome = authoritative_outcome
                    exploration_set_outcome = step_set_outcome
                paired_observation = (
                    paired_audit_adjudicator.adjudicate(
                        plan=paired_plan,
                        legacy_outcome=legacy_outcome,
                        exploration_outcome=exploration_outcome,
                        legacy_set_outcome=legacy_set_outcome,
                        exploration_set_outcome=(
                            exploration_set_outcome
                        ),
                    )
                )
                paired_execution = OutcomeAdaptivePairedAuditExecution(
                    plan=paired_plan,
                    observation=paired_observation,
                    counterfactual_evaluation_batches=(
                        counterfactual_batches
                    ),
                    plan_receipt_sha256=(
                        paired_plan_receipt.receipt_sha256
                    ),
                )
                paired_audit_executions.append(paired_execution)
                counterfactual_quarantined_action_sha256s.add(
                    paired_execution.counterfactual_action_sha256
                )
                if (
                    paired_plan.designer_id
                    in FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS
                ):
                    forecast_calibration_observation = (
                        ForecastOpportunityShadowCalibrationProjector().project(
                            decision_index=request.decision_index,
                            evidence_cutoff_ordinal=request.decision_index,
                            decision=decision,
                            action=adaptive_action_by_sha256[
                                paired_plan.exploration_action_sha256
                            ],
                            observation=paired_observation,
                        )
                    )
                    forecast_calibration_observations.append(
                        forecast_calibration_observation
                    )
            outcomes.extend(step_outcomes)
            set_outcomes.append(step_set_outcome)
            selected_evaluations.extend(step_evaluations)
            waves.append(step_batches)
            selected.update(decision.selected_action_sha256s)
            step_receipt = OutcomeAdaptiveResidualPhaseReceipt(
                phase=(
                    OutcomeAdaptiveResidualPhase
                    .ADAPTIVE_STEP_EVALUATED
                ),
                phase_ordinal=len(receipts) + 1,
                residual_request_sha256=request.request_sha256,
                diagnostic_decision_sha256=(
                    diagnostic.decision_sha256
                ),
                product_sha256s=(
                    decision.decision_sha256,
                    *(value.batch_sha256 for value in step_batches),
                    *(
                        value.outcome_sha256
                        for value in step_outcomes
                    ),
                    step_set_outcome.set_outcome_sha256,
                    *(
                        ()
                        if paired_execution is None
                        else (
                            paired_execution.execution_sha256,
                            paired_execution.observation.observation_sha256,
                            *(
                                ()
                                if forecast_calibration_observation is None
                                else (
                                    forecast_calibration_observation
                                    .observation_sha256,
                                )
                            ),
                            *(
                                value.batch_sha256
                                for value in (
                                    paired_execution
                                    .counterfactual_evaluation_batches
                                )
                            ),
                        )
                    ),
                ),
                evidence=freeze_json(
                    {
                        "adaptive_step": len(
                            continuation_decisions
                        ),
                        "decision": decision.to_record(
                            include_evidence=True
                        ),
                        "evaluation_batches": [
                            value.to_record(include_evidence=True)
                            for value in step_batches
                        ],
                        "outcomes": [
                            value.to_record()
                            for value in step_outcomes
                        ],
                        "set_outcome": (
                            step_set_outcome.to_record()
                        ),
                        **(
                            {}
                            if paired_execution is None
                            else {
                                "paired_audit_execution": (
                                    paired_execution.to_record(
                                        include_evidence=True
                                    )
                                ),
                                "paired_audit_counterfactual_excluded_from_"
                                "authoritative_archive": True,
                                **(
                                    {}
                                    if forecast_calibration_observation is None
                                    else {
                                        "forecast_opportunity_calibration_"
                                        "observation": (
                                            forecast_calibration_observation
                                            .to_record()
                                        )
                                    }
                                ),
                            }
                        ),
                        "causal_set_outcome_policy_role": (
                            "selection_input"
                            if uses_causal_set_outcomes
                            else "observational_only"
                        ),
                        "current_prefix_candidate_outcomes_policy_role": (
                            "selection_input"
                            if self.forecast_opportunity_challenger
                            is not None
                            else "observational_only"
                        ),
                        "selected_action_count_after_step": len(
                            selected
                        ),
                        "unobserved_candidate_outcomes_available": (
                            False
                        ),
                        "next_adaptive_decision_exists": False,
                    }
                ),
            )
            receipts.append(step_receipt)
            ack = await self._commit(step_receipt)
            if ack is not None:
                acks.append(ack)

        directive = AdaptiveActionAllocationDirective(
            policy_id=(
                self.racing_policy.policy_id
                if self.forecast_opportunity_challenger is None
                else self.forecast_opportunity_challenger.challenger_id
            ),
            policy_version=(
                self.racing_policy.policy_version
                if self.forecast_opportunity_challenger is None
                else self.forecast_opportunity_challenger
                .challenger_version
            ),
            policy_definition_sha256=(
                self.racing_policy.definition_sha256
                if self.forecast_opportunity_challenger is None
                else self.forecast_opportunity_challenger
                .definition_sha256
            ),
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(sorted(selected)),
            diagnostic_decision_sha256=diagnostic.decision_sha256,
            continuation_decision_sha256s=tuple(
                sorted(
                    value.decision_sha256
                    for value in continuation_decisions
                )
            ),
            observed_outcome_sha256s=tuple(
                sorted(value.outcome_sha256 for value in outcomes)
            ),
            observed_set_outcome_sha256s=tuple(
                sorted(
                    value.set_outcome_sha256
                    for value in set_outcomes
                )
            ),
            evidence=freeze_json(
                {
                    "market_receipt_sha256": (
                        market_receipt.receipt_sha256
                    ),
                    "diagnostic_receipt_sha256": (
                        diagnostic_receipt.receipt_sha256
                    ),
                    "adaptive_step_receipt_sha256s": [
                        value.receipt_sha256
                        for value in receipts
                        if value.phase
                        is OutcomeAdaptiveResidualPhase
                        .ADAPTIVE_STEP_EVALUATED
                    ],
                    "diagnostic_joint_gain_hex": (
                        diagnostic_joint_gain.hex()
                    ),
                    "set_outcome_sha256s": [
                        value.set_outcome_sha256
                        for value in set_outcomes
                    ],
                    **(
                        {}
                        if not paired_audit_executions
                        else {
                            "paired_audit_plan_receipt_sha256s": [
                                value.plan_receipt_sha256
                                for value in paired_audit_executions
                            ],
                            "paired_audit_execution_sha256s": [
                                value.execution_sha256
                                for value in paired_audit_executions
                            ],
                            "paired_audit_physical_extra_real_evaluation_count": (
                                len(paired_audit_executions)
                            ),
                            "paired_audit_union_admitted_to_"
                            "authoritative_archive": False,
                        }
                    ),
                    "causal_set_outcomes_used_for_selection": (
                        uses_causal_set_outcomes
                    ),
                    "current_prefix_candidate_outcomes_used_for_selection": (
                        self.forecast_opportunity_challenger is not None
                    ),
                    "eligible_candidate_outcomes_observed": False,
                    "all_selected_actions_really_evaluated": True,
                    "unobserved_candidate_outcomes_available": False,
                    "workload_model_provider_prompt_branches": False,
                }
            ),
        )
        final_decision = self.broker.select(
            MaterializedActionBrokerRequest(
                actions=actions,
                evaluation_slots=request.evaluation_slots,
                slate_value=self.slate_value,
                slate_feasibility=self.slate_feasibility,
                reference_escrow_slots=request.reference_escrow_slots,
                allocation_requirement=directive,
            )
        )
        if {
            value.action_sha256
            for value in final_decision.selected_actions
        } != selected:
            raise ValueError("final broker did not honor adaptive directive")
        merged = _merge_waves(proposals, tuple(waves))
        combined = ResidualPortfolioEvolutionResult(
            request=request,
            proposals=proposals,
            broker_decision=final_decision,
            evaluation_batches=merged,
            slate_value_definition_sha256=(
                self.slate_value.definition_sha256
            ),
            slate_feasibility_definition_sha256=(
                self.slate_feasibility.definition_sha256
            ),
        )
        final_receipt = OutcomeAdaptiveResidualPhaseReceipt(
            phase=OutcomeAdaptiveResidualPhase.FINALIZED,
            phase_ordinal=len(receipts) + 1,
            residual_request_sha256=request.request_sha256,
            diagnostic_decision_sha256=diagnostic.decision_sha256,
            product_sha256s=(
                directive.directive_sha256,
                final_decision.decision_sha256,
                combined.result_sha256,
                *(
                    value.execution_sha256
                    for value in paired_audit_executions
                ),
            ),
            evidence=freeze_json(
                {
                    "allocation_directive": directive.to_record(
                        include_evidence=True
                    ),
                    "final_broker_decision": (
                        final_decision.to_record(
                            include_allocation_evidence=True
                        )
                    ),
                    "combined_result_sha256": combined.result_sha256,
                    "real_evaluation_budget_preserved": (
                        not paired_audit_executions
                    ),
                    **(
                        {}
                        if not paired_audit_executions
                        else {
                            "selected_real_evaluation_budget_preserved": True,
                            "physical_real_evaluation_budget_includes_"
                            "assay_extras": True,
                            "paired_audit_physical_extra_real_"
                            "evaluation_count": len(
                                paired_audit_executions
                            ),
                            "paired_audit_execution_sha256s": [
                                value.execution_sha256
                                for value in paired_audit_executions
                            ],
                            "paired_audit_union_admitted_to_"
                            "authoritative_archive": False,
                        }
                    ),
                    "selected_action_count": len(selected),
                    "set_outcome_sha256s": [
                        value.set_outcome_sha256
                        for value in set_outcomes
                    ],
                    "causal_set_outcomes_used_for_selection": (
                        uses_causal_set_outcomes
                    ),
                    "current_prefix_candidate_outcomes_used_for_selection": (
                        self.forecast_opportunity_challenger is not None
                    ),
                    "eligible_candidate_outcomes_observed": False,
                }
            ),
        )
        receipts.append(final_receipt)
        ack = await self._commit(final_receipt)
        if ack is not None:
            acks.append(ack)
        return OutcomeAdaptiveResidualPortfolioEvolutionResult(
            result=combined,
            prior_broker_decision=prior_decision,
            adaptive_actions=adaptive_actions,
            diagnostic_decision=diagnostic,
            continuation_decisions=tuple(continuation_decisions),
            evaluation_waves=tuple(waves),
            outcomes=tuple(
                sorted(
                    outcomes,
                    key=lambda value: value.action_sha256,
                )
            ),
            set_outcomes=tuple(set_outcomes),
            diagnostic_joint_gain=diagnostic_joint_gain,
            allocation_directive=directive,
            phase_receipts=tuple(receipts),
            phase_commit_acks=tuple(acks),
            market_projector_definition_sha256=(
                self.market_projector.definition_sha256
            ),
            market_projector_state_sha256=(
                self.market_projector.state_sha256
            ),
            outcome_projector_definition_sha256=(
                self.outcome_projector.definition_sha256
            ),
            forecast_opportunity_challenger_definition_sha256=(
                None
                if self.forecast_opportunity_challenger is None
                else self.forecast_opportunity_challenger
                .definition_sha256
            ),
            paired_audit_executions=tuple(paired_audit_executions),
            forecast_opportunity_calibration_observations=tuple(
                forecast_calibration_observations
            ),
            method_definition_sha256=(
                OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256
                if (
                    any(
                        value.plan.designer_id
                        == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
                        for value in paired_audit_executions
                    )
                )
                else (
                    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256
                    if (
                        self.forecast_opportunity_challenger is not None
                        and paired_audit_executions
                    )
                    else (
                        _FORECAST_OPPORTUNITY_DEFINITION_SHA256
                        if self.forecast_opportunity_challenger is not None
                        else (
                            _PAIRED_AUDIT_DEFINITION_SHA256
                            if paired_audit_executions
                            else OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256
                        )
                    )
                )
            ),
        )


__all__ = [
    "AdaptiveActionMarketProjectorPort",
    "AdaptiveActionOutcomeProjectorPort",
    "AdaptiveActionSemanticView",
    "AdaptiveActionSemanticViewPort",
    "CandidateArchiveAdaptiveActionOutcomeProjector",
    "FactorStratifiedAdaptiveMarketProjector",
    "OutcomeAdaptivePairedAuditExecution",
    "OutcomeAdaptiveResidualPhase",
    "OutcomeAdaptiveResidualPhaseCommitAck",
    "OutcomeAdaptiveResidualPhaseCommitPort",
    "OutcomeAdaptiveResidualPhaseReceipt",
    "OutcomeAdaptiveResidualPortfolioEvolution",
    "OutcomeAdaptiveResidualPortfolioEvolutionResult",
    "OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_ID",
    "OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_PAIRED_AUDIT_EVOLUTION_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_VERSION",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256",
    "OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_VERSION",
    "PortableMaterializedAdaptiveMarketProjector",
    "PortableMaterializedAdaptiveSemanticView",
    "ScoreEnsembleMaterializedAdaptiveMarketProjector",
]
