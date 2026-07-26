"""Generic campaign-side construction of calibrated portfolio inputs.

This module joins campaign-owned facts before a selector call.  It has no
provider dependency and no workload vocabulary: benchmark adapters keep
owning candidate generation and evaluation, while this factory binds their
finite palette to prior-wave calibration and structural evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSense,
    OptimizationSemantics,
)
from agent_evolve.core.problem import ObjectiveSpec, validate_objective_specs
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    CalibratedPortfolioAllocationContext,
    CalibratedPortfolioInputBinding,
    common_pool_required_option_ids,
    proposal_support_candidates,
)
from agent_evolve.policies.selection.common_candidate_pool import (
    TaskKeyedCommonCandidatePoolPolicy,
)
from agent_evolve.policies.selection.calibrated_slate import (
    MetricOptimizationGoal,
    SlateMetricObjective,
)
from agent_evolve.policies.selection.finite_palette_evidence import (
    FinitePaletteStructuralEvidencePolicy,
    FinitePaletteStructuralEvidenceRequest,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.policies.selection.forecast_calibration import (
    BetaCorrectnessPrior,
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.proposal_support import (
    StructuralProposalSupportPolicy,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
)


_OBJECTIVE_DOMAIN = b"agent-evolve:equal-weight-slate-objective:v1\x00"
_SEMANTIC_OBJECTIVE_DOMAIN = (
    b"agent-evolve:equal-weight-semantic-slate-objective:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def equal_weight_slate_objectives(
    objectives: tuple[ObjectiveSpec, ...],
) -> tuple[SlateMetricObjective, ...]:
    """Project benchmark objective senses without introducing scalar priority."""

    if type(objectives) is not tuple or any(
        type(value) is not ObjectiveSpec for value in objectives
    ):
        raise TypeError("objectives must contain exact ObjectiveSpec values")
    validate_objective_specs(objectives)
    results = []
    for objective in sorted(objectives, key=lambda value: value.name):
        record = {
            "schema_version": 1,
            "metric_id": objective.name,
            "benchmark_goal": objective.goal,
            "weight_hex": (1.0).hex(),
            "weight_law": "equal_per_metric_no_scalar_objective_claim",
        }
        results.append(
            SlateMetricObjective(
                metric_id=objective.name,
                goal=(
                    MetricOptimizationGoal.MINIMIZE
                    if objective.goal == "min"
                    else MetricOptimizationGoal.MAXIMIZE
                ),
                weight=1.0,
                definition_sha256=hashlib.sha256(
                    _OBJECTIVE_DOMAIN + _canonical_json(record)
                ).hexdigest(),
            )
        )
    return tuple(results)


def equal_weight_slate_objectives_from_decision_metrics(
    projection: DecisionMetricProjection,
) -> tuple[SlateMetricObjective, ...]:
    """Build equal-weight slate goals from one outcome projection.

    Only monotone ``MINIMIZE``/``MAXIMIZE`` senses have an honest mapping to
    the allocator's goal vocabulary. Target, bounded-satisfaction, and
    informational metrics fail closed rather than being silently scalarized.
    """

    if type(projection) is not DecisionMetricProjection:
        raise TypeError("projection must be exact DecisionMetricProjection")
    projection.__post_init__()
    results: list[SlateMetricObjective] = []
    for metric in projection.metrics:
        if metric.sense is MetricSense.MINIMIZE:
            goal = MetricOptimizationGoal.MINIMIZE
            benchmark_goal = "min"
        elif metric.sense is MetricSense.MAXIMIZE:
            goal = MetricOptimizationGoal.MAXIMIZE
            benchmark_goal = "max"
        else:
            raise ValueError(
                "decision metric has no meaningful min/max slate mapping: "
                f"{metric.metric_id} ({metric.sense.value})"
            )
        if (
            projection.objective_only_legacy_metric_ids
            and metric.role is MetricRole.OBJECTIVE
        ):
            # Deliberately reproduce the historical ObjectiveSpec helper's
            # exact authenticated record for objective-only benchmarks.
            record = {
                "schema_version": 1,
                "metric_id": metric.metric_id,
                "benchmark_goal": benchmark_goal,
                "weight_hex": (1.0).hex(),
                "weight_law": "equal_per_metric_no_scalar_objective_claim",
            }
            domain = _OBJECTIVE_DOMAIN
        else:
            record = {
                "schema_version": 1,
                "projection_definition_sha256": projection.definition_sha256,
                "metric": metric.to_record(),
                "goal": goal.value,
                "weight_hex": (1.0).hex(),
                "weight_law": "equal_per_decision_metric_no_scalar_priority_claim",
            }
            domain = _SEMANTIC_OBJECTIVE_DOMAIN
        results.append(
            SlateMetricObjective(
                metric_id=metric.metric_id,
                goal=goal,
                weight=1.0,
                definition_sha256=hashlib.sha256(
                    domain + _canonical_json(record)
                ).hexdigest(),
            )
        )
    return tuple(results)


def equal_weight_slate_objectives_from_optimization_semantics(
    semantics: OptimizationSemantics,
) -> tuple[SlateMetricObjective, ...]:
    """Project benchmark semantics into the calibrated allocator's goals."""

    return equal_weight_slate_objectives_from_decision_metrics(
        DecisionMetricProjection.from_optimization_semantics(semantics)
    )


@dataclass(slots=True)
class CalibratedCampaignBindingFactory:
    """Create one immutable pre-provider K8→K4 binding per campaign wave."""

    scope: ForecastCalibrationScope
    objectives: tuple[SlateMetricObjective, ...]
    ledger: PortfolioOutcomeFeedbackLedger
    structural_evidence: FinitePaletteStructuralEvidencePolicy = field(
        default_factory=FinitePaletteStructuralEvidencePolicy
    )
    prior: BetaCorrectnessPrior = field(default_factory=BetaCorrectnessPrior)
    family_min_support: int = 4
    option_prompt_projection: FiniteOptionPromptProjectionPolicy | None = None
    common_candidate_pool_policy: TaskKeyedCommonCandidatePoolPolicy | None = None
    proposal_support_policy: StructuralProposalSupportPolicy | None = None
    assign_all_cards_by_default: bool = True

    def __post_init__(self) -> None:
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        if (
            type(self.objectives) is not tuple
            or not self.objectives
            or any(type(value) is not SlateMetricObjective for value in self.objectives)
        ):
            raise ValueError("objectives must contain exact slate objectives")
        for value in self.objectives:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.objectives)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("objectives must use unique canonical metric order")
        if type(self.ledger) is not PortfolioOutcomeFeedbackLedger:
            raise TypeError("ledger must be exact PortfolioOutcomeFeedbackLedger")
        if type(self.structural_evidence) is not FinitePaletteStructuralEvidencePolicy:
            raise TypeError("structural_evidence must be exact finite-palette policy")
        self.structural_evidence.__post_init__()
        if type(self.prior) is not BetaCorrectnessPrior:
            raise TypeError("prior must be exact BetaCorrectnessPrior")
        self.prior.__post_init__()
        if type(self.family_min_support) is not int or self.family_min_support <= 0:
            raise ValueError("family_min_support must be positive")
        if self.option_prompt_projection is not None:
            if (
                type(self.option_prompt_projection)
                is not FiniteOptionPromptProjectionPolicy
            ):
                raise TypeError(
                    "option_prompt_projection must be exact projection policy"
                )
            self.option_prompt_projection.__post_init__()
        if self.common_candidate_pool_policy is not None:
            if type(self.common_candidate_pool_policy) is not (
                TaskKeyedCommonCandidatePoolPolicy
            ):
                raise TypeError(
                    "common_candidate_pool_policy must be an exact policy or None"
                )
            self.common_candidate_pool_policy.__post_init__()
        if self.proposal_support_policy is not None:
            if type(self.proposal_support_policy) is not (
                StructuralProposalSupportPolicy
            ):
                raise TypeError(
                    "proposal_support_policy must be an exact policy or None"
                )
            self.proposal_support_policy.__post_init__()
            if self.common_candidate_pool_policy is None:
                raise ValueError(
                    "proposal support requires a common candidate-pool policy"
                )
        if type(self.assign_all_cards_by_default) is not bool:
            raise TypeError("assign_all_cards_by_default must be an exact bool")

    def build(
        self,
        *,
        request: PortfolioSelectionRequest,
        variation: ParentVariationBinding,
        wave_index: int,
        frozen_archive_snapshot_sha256: str,
        assigned_card_keys: tuple[str, ...] | None = None,
        contextual_allocation: ContextualPortfolioAllocationContract | None = None,
    ) -> CalibratedPortfolioInputBinding:
        """Bind request, cutoff, palette evidence, and prior calibration exactly."""

        self.__post_init__()
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be exact PortfolioSelectionRequest")
        request.__post_init__()
        if type(variation) is not ParentVariationBinding:
            raise TypeError("variation must be exact ParentVariationBinding")
        ParentVariationBinding.__post_init__(variation)
        if request.finite_variation_contract != variation.contract:
            raise ValueError("selection request differs from its campaign variation")
        if self.scope.benchmark_sha256 != variation.benchmark_sha256:
            raise ValueError("calibration scope names a foreign benchmark")
        if type(wave_index) is not int or wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        require_sha256(
            frozen_archive_snapshot_sha256,
            "frozen_archive_snapshot_sha256",
        )
        receipt = variation.eligibility_receipt
        if receipt is None:
            raise ValueError(
                "calibrated campaign requires phenotype eligibility evidence"
            )
        receipt.__post_init__()
        phenotype_by_option = {
            value.option_id: (
                value.option_identity_sha256,
                value.phenotype_identity_sha256,
            )
            for value in receipt.option_phenotypes
        }
        option_phenotypes: list[tuple[str, str]] = []
        for option in variation.contract.options:
            source = phenotype_by_option.get(option.option_id)
            if source is None or source[0] != option.identity_sha256:
                raise ValueError(
                    "eligibility evidence differs from the finite contract"
                )
            option_phenotypes.append((option.option_id, source[1]))
        evidence = self.structural_evidence.project(
            FinitePaletteStructuralEvidenceRequest(
                contract=variation.contract,
                option_phenotype_sha256s=tuple(sorted(option_phenotypes)),
                known_phenotype_sha256s=variation.known_phenotype_sha256s,
                eligibility_receipt_sha256=receipt.receipt_sha256,
                frozen_archive_snapshot_sha256=frozen_archive_snapshot_sha256,
            )
        )
        if assigned_card_keys is None:
            if request.memory_dose_contract is not None:
                keys = request.memory_dose_contract.assigned_card_keys
            elif self.assign_all_cards_by_default:
                keys = tuple(sorted(card.card_key for card in request.cards))
            else:
                keys = ()
        else:
            keys = assigned_card_keys
        if type(keys) is not tuple or keys != tuple(sorted(set(keys))):
            raise ValueError("assigned_card_keys must be unique and canonical")
        if not set(keys).issubset({card.card_key for card in request.cards}):
            raise ValueError("assigned cards escape the selector request")
        objective_ids = tuple(value.metric_id for value in self.objectives)
        if objective_ids != request.required_metric_ids:
            raise ValueError("slate objectives differ from requested metrics")
        context = CalibratedPortfolioAllocationContext(
            scope=self.scope,
            wave_index=wave_index,
            parent_candidate_identity_sha256=(variation.parent_configuration_sha256),
            objectives=self.objectives,
            assigned_card_keys=keys,
            calibration_snapshot=self.ledger.calibration_snapshot(
                scope=self.scope,
                cutoff_wave_index_exclusive=wave_index,
                prior=self.prior,
                family_min_support=self.family_min_support,
            ),
        )
        common_pool = (
            None
            if self.common_candidate_pool_policy is None
            else self.common_candidate_pool_policy.select(
                benchmark_sha256=self.scope.benchmark_sha256,
                wave_index=wave_index,
                parent_configuration_sha256=(
                    variation.parent_configuration_sha256
                ),
                contract=variation.contract,
                evaluation_size=request.portfolio_size,
                min_distinct_families=request.min_distinct_families,
                require_pairwise_disjoint_parent_patches=(
                    request.require_pairwise_disjoint_parent_patches
                ),
                required_option_ids=common_pool_required_option_ids(request),
            )
        )
        proposal_support = (
            None
            if self.proposal_support_policy is None
            else self.proposal_support_policy.select(
                request_sha256=request.request_sha256,
                common_candidate_pool_decision_sha256=common_pool.decision_sha256,
                model_selection_size=common_pool.model_selection_size,
                candidates=proposal_support_candidates(
                    request,
                    evidence,
                    common_pool,
                ),
            )
        )
        return CalibratedPortfolioInputBinding(
            request_sha256=request.request_sha256,
            context=context,
            option_evidence=evidence,
            option_prompt_projection=(
                None
                if self.option_prompt_projection is None
                else self.option_prompt_projection.project(variation.contract)
            ),
            common_candidate_pool=common_pool,
            proposal_support=proposal_support,
            contextual_allocation=contextual_allocation,
        )


__all__ = [
    "CalibratedCampaignBindingFactory",
    "equal_weight_slate_objectives",
    "equal_weight_slate_objectives_from_decision_metrics",
    "equal_weight_slate_objectives_from_optimization_semantics",
]
