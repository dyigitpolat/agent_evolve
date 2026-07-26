"""Selected-only campaign lifecycle for target-conditioned acquisition."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Protocol

from agent_evolve.application.campaign_execution import CampaignStageRequest
from agent_evolve.application.evolution_campaign import CampaignGenerationKind
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignPortfolioOutcomePreparation,
    CampaignPortfolioWaveContext,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemberDisposition,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    RegisteredTargetConditionedAllocationContextProvider,
    TargetConditionedAllocationContext,
    TargetConditionedAllocationContextProvider,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    TargetConditionedAcquisitionState,
    TargetConditionedAcquisitionProfile,
    TargetConditionedMetaPrior,
    TargetConditionedSelectedObservation,
    TargetConditionedSlateDecision,
    TargetConditionedStateUpdateReceipt,
    update_target_conditioned_state,
)
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest


_CUTOFF_DOMAIN = b"agent-evolve:target-conditioned-campaign-cutoff:v1\x00"
_SPECIFICATION_DOMAIN = (
    b"agent-evolve:target-conditioned-campaign-specification:v1\x00"
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("target-conditioned evidence did not freeze to an object")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _decimal(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be decimal text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _achievement(point: tuple[float, ...], weights: tuple[float, ...]) -> float:
    active = tuple(
        (value, weight)
        for value, weight in zip(point, weights, strict=True)
        if weight > 0.0
    )
    if not active:
        raise ValueError("target direction must activate at least one axis")
    maximum = max(weight * value for value, weight in active)
    weighted_mean = sum(weight * value for value, weight in active) / sum(
        weight for _value, weight in active
    )
    return maximum + 0.05 * weighted_mean


class SelectedTargetConditionedDecisionProvider(Protocol):
    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> TargetConditionedSlateDecision: ...


class SelectedTargetConditionedContextProvider(Protocol):
    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> TargetConditionedAllocationContext: ...


class SelectedMarginalUtilityProjector(Protocol):
    def project(
        self,
        *,
        snapshot: object,
        results: tuple[PortfolioVariationWaveResult, ...],
    ) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class TargetConditionedCampaignSpecification:
    """Portable frozen prior/profile pair used by every workload launcher."""

    profile: TargetConditionedAcquisitionProfile
    meta_prior: TargetConditionedMetaPrior

    def __post_init__(self) -> None:
        if type(self.profile) is not TargetConditionedAcquisitionProfile:
            raise TypeError("profile must be exact")
        self.profile.__post_init__()
        if type(self.meta_prior) is not TargetConditionedMetaPrior:
            raise TypeError("meta_prior must be exact")
        self.meta_prior.__post_init__()
        if (
            self.meta_prior.marginal_head.feature_names
            != self.meta_prior.direction_head.feature_names
        ):
            raise ValueError("target-conditioned prior feature schemas disagree")

    @classmethod
    def from_freeze_record(
        cls, record: object
    ) -> TargetConditionedCampaignSpecification:
        if type(record) is not dict:
            raise TypeError("target-conditioned freeze record must be an object")
        if record.get("schema_version") != 1:
            raise ValueError("unsupported target-conditioned freeze schema")
        if record.get("artifact_id") != "trap_portable_profile_v1":
            raise ValueError("target-conditioned freeze names a foreign artifact")
        return cls(
            profile=TargetConditionedAcquisitionProfile.from_record(
                record.get("profile")
            ),
            meta_prior=TargetConditionedMetaPrior.from_record(
                record.get("meta_prior")
            ),
        )

    @property
    def specification_sha256(self) -> str:
        self.__post_init__()
        return hashlib.sha256(
            _SPECIFICATION_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 1,
                    "profile": self.profile.to_record(),
                    "meta_prior": self.meta_prior.to_record(),
                }
            )
        ).hexdigest()

    def build_allocator(
        self,
        context_provider: TargetConditionedAllocationContextProvider | None = None,
    ) -> TargetConditionedSlateAllocatorAdapter:
        self.__post_init__()
        return TargetConditionedSlateAllocatorAdapter(
            context_provider=(
                RegisteredTargetConditionedAllocationContextProvider()
                if context_provider is None
                else context_provider
            ),
            profile=self.profile,
        )

    def initial_state(
        self, *, campaign_scope_sha256: str
    ) -> TargetConditionedAcquisitionState:
        self.__post_init__()
        return self.meta_prior.initial_state(
            campaign_scope_sha256=campaign_scope_sha256
        )


def selected_target_improvements(
    decision: TargetConditionedSlateDecision,
    allocation_context: TargetConditionedAllocationContext,
    result: PortfolioVariationWaveResult,
) -> tuple[float, ...]:
    """Project evaluated members onto their pre-call affine target in [-1, 1]."""

    if type(decision) is not TargetConditionedSlateDecision:
        raise TypeError("decision must be exact TargetConditionedSlateDecision")
    decision.revalidate()
    if type(allocation_context) is not TargetConditionedAllocationContext:
        raise TypeError("allocation_context must be exact")
    allocation_context.__post_init__()
    if type(result) is not PortfolioVariationWaveResult:
        raise TypeError("result must be exact PortfolioVariationWaveResult")
    result.__post_init__()
    request = decision.request
    if (
        allocation_context.frontier_target != request.frontier_target
        or allocation_context.state != request.state
        or allocation_context.campaign_generation != request.campaign_generation
        or allocation_context.remaining_proposal_horizon
        != request.remaining_proposal_horizon
    ):
        raise ValueError("allocation context differs from the selected decision")
    allocation_context.require_request(request.allocation_request)

    archive = thaw_json(allocation_context.archive_context.payload)
    target = thaw_json(request.frontier_target.payload)
    if type(archive) is not dict or type(target) is not dict:
        raise TypeError("affine archive and target payloads must be objects")
    frame = archive.get("optimization_frame")
    parent = archive.get("parent")
    target_direction = target.get("target_direction")
    assigned_parent = target.get("assigned_parent")
    if any(
        type(value) is not dict
        for value in (frame, parent, target_direction, assigned_parent)
    ):
        raise ValueError("affine archive or target payload is incomplete")
    axes_raw = frame.get("axes")
    parent_raw = parent.get("normalized_point_decimal")
    weights_raw = target_direction.get("normalized_weights_decimal")
    assigned_parent_raw = assigned_parent.get("normalized_point_decimal")
    if (
        type(axes_raw) is not list
        or len(axes_raw) not in (2, 3)
        or type(parent_raw) is not list
        or type(weights_raw) is not list
        or type(assigned_parent_raw) is not list
        or len(parent_raw) != len(axes_raw)
        or len(weights_raw) != len(axes_raw)
        or len(assigned_parent_raw) != len(axes_raw)
    ):
        raise ValueError("affine target geometry has an unsupported shape")

    axes: list[tuple[str, str, float, float]] = []
    for index, raw in enumerate(axes_raw):
        if type(raw) is not dict:
            raise TypeError(f"archive axis {index} must be an object")
        metric_id = raw.get("metric_id")
        goal = raw.get("source_goal")
        if type(metric_id) is not str or not metric_id:
            raise ValueError(f"archive axis {index} omitted metric_id")
        if goal not in ("min", "max"):
            raise ValueError(f"archive axis {index} has an unsupported source_goal")
        ideal = _decimal(raw.get("ideal_decimal"), name=f"axis[{index}].ideal")
        reference = _decimal(
            raw.get("reference_decimal"), name=f"axis[{index}].reference"
        )
        if (goal == "min" and not reference > ideal) or (
            goal == "max" and not ideal > reference
        ):
            raise ValueError(f"archive axis {index} has invalid affine bounds")
        axes.append((metric_id, goal, ideal, reference))
    metric_ids = tuple(value[0] for value in axes)
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("affine archive axes repeat a metric ID")

    parent_point = tuple(
        _decimal(value, name="archive parent point") for value in parent_raw
    )
    assigned_parent_point = tuple(
        _decimal(value, name="target parent point") for value in assigned_parent_raw
    )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(parent_point, assigned_parent_point, strict=True)
    ):
        raise ValueError("frontier target parent differs from archive context")
    weights = tuple(_decimal(value, name="target weight") for value in weights_raw)
    if any(value < 0.0 for value in weights) or max(weights) <= 0.0:
        raise ValueError("target weights must be non-negative and non-zero")
    parent_achievement = _decimal(
        assigned_parent.get("achievement_decimal"),
        name="parent achievement",
    )
    recomputed_parent_achievement = _achievement(parent_point, weights)
    if not math.isclose(
        parent_achievement,
        recomputed_parent_achievement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("frontier target parent achievement is invalid")
    opportunity = _decimal(
        target_direction.get("opportunity_from_ideal_decimal"),
        name="target opportunity",
    )
    if opportunity < 0.0:
        raise ValueError("target opportunity must be non-negative")
    denominator = max(opportunity, parent_achievement, 1e-12)

    selected_ids = tuple(value.option_id for value in decision.selected)
    received_ids = tuple(
        value.materialization.option_id for value in result.receipt.members
    )
    if selected_ids != received_ids:
        raise ValueError("selected decision differs from evaluated member order")
    values: list[float] = []
    for member, outcome in zip(result.receipt.members, result.outcomes, strict=True):
        if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
            values.append(-1.0)
            continue
        candidate = outcome.candidate
        if candidate is None:
            raise ValueError("scored selected outcome omitted its candidate")
        objective_map = candidate.objective_map
        if set(objective_map) != set(metric_ids):
            raise ValueError("candidate objectives differ from affine archive axes")
        point = []
        for metric_id, goal, ideal, reference in axes:
            raw_value = float(objective_map[metric_id])
            if not math.isfinite(raw_value):
                raise ValueError("candidate objective must be finite")
            normalized = (
                (raw_value - ideal) / (reference - ideal)
                if goal == "min"
                else (ideal - raw_value) / (ideal - reference)
            )
            point.append(normalized)
        candidate_achievement = _achievement(tuple(point), weights)
        improvement = (parent_achievement - candidate_achievement) / denominator
        values.append(max(-1.0, min(1.0, improvement)))
    return tuple(values)


@dataclass(slots=True)
class TargetConditionedCampaignOutcomeUpdater:
    """Prepare and atomically publish selected-only T-RAP posterior updates."""

    state: TargetConditionedAcquisitionState
    selected_decision: SelectedTargetConditionedDecisionProvider
    selected_context: SelectedTargetConditionedContextProvider
    marginal_utility: SelectedMarginalUtilityProjector
    _prepared: dict[
        str,
        tuple[CampaignPortfolioOutcomePreparation, TargetConditionedStateUpdateReceipt],
    ] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.state) is not TargetConditionedAcquisitionState:
            raise TypeError("state must be exact TargetConditionedAcquisitionState")
        self.state.__post_init__()
        if not callable(self.selected_decision):
            raise TypeError("selected_decision must be callable")
        if not callable(self.selected_context):
            raise TypeError("selected_context must be callable")
        if not callable(getattr(self.marginal_utility, "project", None)):
            raise TypeError("marginal_utility must implement its projection port")

    def context_for_wave(
        self,
        *,
        build: CampaignPortfolioWaveContext,
        selection: PortfolioSelectionRequest,
    ) -> TargetConditionedAllocationContext:
        """Build one complete pre-call context from generic campaign facts."""

        self.__post_init__()
        if type(build) is not CampaignPortfolioWaveContext:
            raise TypeError("build must be exact CampaignPortfolioWaveContext")
        build.__post_init__()
        if type(selection) is not PortfolioSelectionRequest:
            raise TypeError("selection must be exact PortfolioSelectionRequest")
        selection.__post_init__()
        archive_context = build.archive_context
        frontier_target = build.frontier_target
        if archive_context is None or frontier_target is None:
            raise ValueError(
                "target-conditioned waves require archive context and frontier target"
            )
        generation = build.stage_request.step.generation
        future_portfolio_generations = sum(
            value.kind is CampaignGenerationKind.PORTFOLIO
            and value.generation > generation
            for value in build.prepared.schedule.steps
        )
        cutoff_record = {
            "schema_version": 1,
            "campaign_state_sha256": self.state.state_sha256,
            "archive_cutoff_request_sha256": (
                build.stage_request.archive_cutoff.request_sha256
            ),
            "archive_utility_snapshot_sha256": (
                build.stage_request.archive_utility.snapshot_sha256
            ),
            "frontier_target_sha256": frontier_target.target_sha256,
            "generation": generation,
            "lane_id": build.parent_lane.lane_id,
            "current_or_future_outcomes_consulted": False,
        }
        return TargetConditionedAllocationContext(
            finite_contract_sha256=(
                selection.finite_variation_contract.identity_sha256
            ),
            cutoff_receipt_sha256=hashlib.sha256(
                _CUTOFF_DOMAIN + _canonical_json(cutoff_record)
            ).hexdigest(),
            archive_context=archive_context,
            frontier_target=frontier_target,
            state=self.state,
            transition_receipts=(),
            campaign_generation=generation,
            lane_slot=build.parent_slot,
            remaining_proposal_horizon=future_portfolio_generations,
        )

    async def prepare_update(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        prior_memory: FrozenJsonObject,
    ) -> CampaignPortfolioOutcomePreparation:
        self.__post_init__()
        if type(request) is not CampaignStageRequest:
            raise TypeError("request must be exact CampaignStageRequest")
        request.__post_init__()
        if request.step.kind is not CampaignGenerationKind.PORTFOLIO:
            raise ValueError("target-conditioned updates require a portfolio stage")
        if (
            type(waves) is not tuple
            or type(results) is not tuple
            or not waves
            or len(waves) != len(results)
        ):
            raise ValueError("waves and results must be equal non-empty tuples")
        if type(prior_memory) is not FrozenJsonObject:
            raise TypeError("prior_memory must be an exact frozen object")
        decisions = tuple(
            self.selected_decision(wave, result)
            for wave, result in zip(waves, results, strict=True)
        )
        contexts = tuple(
            self.selected_context(wave, result)
            for wave, result in zip(waves, results, strict=True)
        )
        if any(type(value) is not TargetConditionedSlateDecision for value in decisions):
            raise TypeError("decision provider returned a foreign allocation")
        if any(type(value) is not TargetConditionedAllocationContext for value in contexts):
            raise TypeError("context provider returned a foreign allocation context")
        marginal = self.marginal_utility.project(
            snapshot=request.archive_utility,
            results=results,
        )
        if len(marginal) != len(results):
            raise ValueError("marginal utility projection differs from wave count")
        observations: list[TargetConditionedSelectedObservation] = []
        for decision, allocation_context, result, utilities in zip(
            decisions, contexts, results, marginal, strict=True
        ):
            if len(utilities) != len(result.receipt.members):
                raise ValueError("marginal utilities differ from selected members")
            selected_ids = tuple(value.option_id for value in decision.selected)
            receipt_ids = tuple(
                value.materialization.option_id for value in result.receipt.members
            )
            if selected_ids != receipt_ids:
                raise ValueError("T-RAP allocation differs from evaluated member order")
            improvements = selected_target_improvements(
                decision,
                allocation_context,
                result,
            )
            features = {
                value.option_id: value for value in decision.request.member_features
            }
            for selected, member, utility, improvement in zip(
                decision.selected,
                result.receipt.members,
                utilities,
                improvements,
                strict=True,
            ):
                feature = features[selected.option_id]
                observations.append(
                    TargetConditionedSelectedObservation(
                        decision_sha256=decision.decision_sha256,
                        campaign_generation=request.step.generation,
                        option_id=selected.option_id,
                        option_identity_sha256=selected.option_identity_sha256,
                        feature_row_sha256=feature.feature_row_sha256,
                        feature_values=feature.values,
                        normalized_marginal_utility=float(utility),
                        normalized_target_improvement=improvement,
                        evaluator_receipt_sha256=member.outcome_sha256,
                    )
                )
        update = update_target_conditioned_state(
            self.state,
            decisions=decisions,
            observations=tuple(observations),
        )
        preparation = CampaignPortfolioOutcomePreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                value.selection_request.request_sha256 for value in waves
            ),
            result_receipt_sha256s=tuple(
                value.receipt.receipt_sha256 for value in results
            ),
            prior_memory_sha256=typed_json_sha256(prior_memory),
            updated_memory=prior_memory,
            evidence=_object(
                {
                    "schema_version": 1,
                    "target_conditioned_state_update": update.to_record(),
                    "marginal_label_scope": (
                        "selected_fixed_reference_candidate_marginal_utility"
                    ),
                    "target_label_scope": (
                        "selected_affine_target_achievement_improvement"
                    ),
                    "rejected_outcomes_consulted": False,
                    "provider_calls": 0,
                }
            ),
        )
        if preparation.preparation_sha256 in self._prepared:
            raise ValueError("target-conditioned update is already pending")
        self._prepared[preparation.preparation_sha256] = (preparation, update)
        return preparation

    def commit_update(self, preparation: CampaignPortfolioOutcomePreparation) -> None:
        pending = self._require_pending(preparation)
        self._prepared.pop(preparation.preparation_sha256)
        self.state = pending.next_state

    def abort_update(self, preparation: CampaignPortfolioOutcomePreparation) -> None:
        self._require_pending(preparation)
        self._prepared.pop(preparation.preparation_sha256)

    def _require_pending(
        self, preparation: CampaignPortfolioOutcomePreparation
    ) -> TargetConditionedStateUpdateReceipt:
        if type(preparation) is not CampaignPortfolioOutcomePreparation:
            raise TypeError("preparation must be exact")
        preparation.__post_init__()
        pending = self._prepared.get(preparation.preparation_sha256)
        if pending is None or pending[0] != preparation:
            raise ValueError("target-conditioned update is foreign or not pending")
        return pending[1]

__all__ = [
    "SelectedMarginalUtilityProjector",
    "SelectedTargetConditionedContextProvider",
    "SelectedTargetConditionedDecisionProvider",
    "TargetConditionedCampaignSpecification",
    "TargetConditionedCampaignOutcomeUpdater",
    "selected_target_improvements",
]
