"""Audited greedy allocation over authenticated forecast frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_evolve.application.action_allocation import (
    GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.ports.action_allocation import (
    AllocatedActionMember,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    PortfolioAllocationScore,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionAllocationSurfaceAudit,
    AllocationCandidateScoreDiagnostic,
    AllocationCandidateScoreDiagnosticInput,
    AllocationScoreDiagnosticBinding,
    AllocationSurfaceGatePolicyBinding,
    AllocationSurfaceStepAudit,
    AuditedFrameActionAllocationResult,
    FrameActionAllocationRequest,
    FrameActionPortfolioDecision,
    allocation_score_multiset_sha256,
    allocation_surface_failure_codes,
    validate_frame_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import ResolvedActionForecast


@dataclass(frozen=True, slots=True)
class AllocationSurfaceGateRejected(RuntimeError):
    """A complete deterministic attempt whose pre-evaluator gate failed."""

    result: AuditedFrameActionAllocationResult

    def __post_init__(self) -> None:
        if type(self.result) is not AuditedFrameActionAllocationResult:
            raise TypeError("result must be an exact audited frame result")
        self.result.__post_init__()
        if self.result.audit.passes:
            raise ValueError("a passing allocation result cannot be rejected")
        RuntimeError.__init__(
            self,
            "allocation surface failed its bound pre-evaluator gate",
        )


@dataclass(frozen=True, slots=True)
class AuditedGreedyForecastFrameAllocator:
    """Allocator-v2-equivalent scoring over a non-forged forecast frame.

    The complete-batch allocator exact-type-checks its request, so forwarding a
    partition block would require fabricating global coverage.  This service
    instead consumes the authenticated frame port directly while retaining the
    same v2 score, attainable-diversity denominator, canonical tie break, and
    configuration identity.
    """

    risk_aversion: float
    diversity_weight: float
    score_diagnostic: AllocationScoreDiagnosticBinding
    gate_policy: AllocationSurfaceGatePolicyBinding

    def __post_init__(self) -> None:
        for name in ("risk_aversion", "diversity_weight"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.score_diagnostic) is not AllocationScoreDiagnosticBinding:
            raise TypeError("score_diagnostic must be an exact binding")
        self.score_diagnostic.__post_init__()
        if type(self.gate_policy) is not AllocationSurfaceGatePolicyBinding:
            raise TypeError("gate_policy must be an exact binding")
        self.gate_policy.__post_init__()

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=self.risk_aversion,
            diversity_weight=self.diversity_weight,
        ).configuration_sha256

    @staticmethod
    def _utility(
        request: FrameActionAllocationRequest,
        members: tuple[ResolvedActionForecast, ...],
        quantile: ForecastQuantile,
    ) -> float:
        canonical_members = tuple(
            sorted(
                members,
                key=lambda value: (value.option_identity_sha256, value.option_id),
            )
        )
        value = request.utility.utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=request.frame.request.optimization_semantics,
                parent_metric_values=request.frame.request.parent_metric_values,
                metric_scales=request.frame.request.metric_scales,
                members=canonical_members,
                quantile=quantile,
            )
        )
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("bound portfolio utility must return a finite float")
        return value

    def _score(
        self,
        request: FrameActionAllocationRequest,
        members: tuple[ResolvedActionForecast, ...],
        *,
        attainable_diversity: int,
    ) -> PortfolioAllocationScore:
        p10 = self._utility(request, members, ForecastQuantile.P10)
        p50 = self._utility(request, members, ForecastQuantile.P50)
        p90 = self._utility(request, members, ForecastQuantile.P90)
        downside = min(p10, p90)
        risk_penalty = self.risk_aversion * max(0.0, p50 - downside)
        diversity_reward = self.diversity_weight * (
            len({member.family for member in members}) / attainable_diversity
        )
        total = p50 - risk_penalty + diversity_reward
        if not all(
            math.isfinite(value)
            for value in (risk_penalty, diversity_reward, total)
        ):
            raise ValueError("allocation score arithmetic became non-finite")
        return PortfolioAllocationScore(
            p10_utility=p10,
            p50_utility=p50,
            p90_utility=p90,
            downside_utility=downside,
            risk_penalty=risk_penalty,
            diversity_reward=diversity_reward,
            total_utility=total,
        )

    def assess(
        self,
        request: FrameActionAllocationRequest,
    ) -> AuditedFrameActionAllocationResult:
        """Return the complete attempt, including a failing gate receipt."""

        if type(request) is not FrameActionAllocationRequest:
            raise TypeError("request must be an exact frame allocation request")
        request.__post_init__()
        self.__post_init__()
        # These authenticated identities recursively validate and hash the
        # complete frame/request graph.  Compute each exactly once after the
        # explicit validation boundary; candidate scoring must remain O(kn),
        # not repeat graph authentication for every greedy extension.
        allocation_request_sha256 = request.request_sha256
        frame_receipt_sha256 = request.frame.receipt_sha256
        source_forecast_receipt_sha256 = (
            request.frame.source_forecast_receipt_sha256
        )
        eligible_options_sha256 = request.eligible_options_sha256
        allocator_configuration_sha256 = self.configuration_sha256
        by_id = {value.option_id: value for value in request.frame.forecasts}
        global_index_by_id = {
            forecast.option_id: global_index
            for global_index, forecast in zip(
                request.frame.global_row_indices,
                request.frame.forecasts,
                strict=True,
            )
        }
        attainable_diversity = min(
            request.portfolio_size,
            len({by_id[value].family for value in request.eligible_option_ids}),
        )
        remaining = sorted(
            (by_id[value] for value in request.eligible_option_ids),
            key=lambda value: (value.option_identity_sha256, value.option_id),
        )
        selected: list[ResolvedActionForecast] = []
        members: list[AllocatedActionMember] = []
        step_audits: list[AllocationSurfaceStepAudit] = []
        previous_total = 0.0

        for step in range(1, request.portfolio_size + 1):
            candidates: list[
                tuple[
                    ResolvedActionForecast,
                    PortfolioAllocationScore,
                    float,
                    str,
                    AllocationCandidateScoreDiagnostic,
                ]
            ] = []
            for candidate in remaining:
                portfolio = tuple((*selected, candidate))
                score = self._score(
                    request,
                    portfolio,
                    attainable_diversity=attainable_diversity,
                )
                marginal = score.total_utility - previous_total
                if not math.isfinite(marginal):
                    raise ValueError("marginal allocation score became non-finite")
                label = f"row_{global_index_by_id[candidate.option_id]:08d}"
                diagnostic_input = AllocationCandidateScoreDiagnosticInput(
                    allocation_request_sha256=allocation_request_sha256,
                    step=step,
                    candidate_label=label,
                    members=tuple(
                        sorted(
                            portfolio,
                            key=lambda value: (
                                value.option_identity_sha256,
                                value.option_id,
                            ),
                        )
                    ),
                    score=score,
                    marginal_total_utility=marginal,
                )
                diagnostic = self.score_diagnostic.diagnostic(diagnostic_input)
                if type(diagnostic) is not AllocationCandidateScoreDiagnostic:
                    raise TypeError(
                        "score diagnostic returned another result type"
                    )
                diagnostic.__post_init__()
                candidates.append((candidate, score, marginal, label, diagnostic))

            # ``remaining`` is already in allocator-v2 canonical tie order.
            top_score = max(value[2] for value in candidates)
            top = tuple(value for value in candidates if value[2] == top_score)
            winner = top[0]
            ranked_scores = sorted(
                (value[2] for value in candidates),
                reverse=True,
            )
            runner_gap = (
                0.0
                if len(ranked_scores) == 1
                else ranked_scores[0] - ranked_scores[1]
            )
            boundary_count = sum(
                value[4].boundary_or_extreme for value in candidates
            )
            boundary_share = boundary_count / len(candidates)
            score_values = tuple(value[2] for value in candidates)
            failure_codes = allocation_surface_failure_codes(
                policy=self.gate_policy,
                candidate_count=len(candidates),
                distinct_finite_score_count=len(set(score_values)),
                top_tie_count=len(top),
                winner_runner_gap=runner_gap,
                boundary_or_extreme_share=boundary_share,
            )
            step_audits.append(
                AllocationSurfaceStepAudit(
                    step=step,
                    candidate_count=len(candidates),
                    distinct_finite_score_count=len(set(score_values)),
                    top_tie_count=len(top),
                    winner_runner_gap=runner_gap,
                    boundary_or_extreme_count=boundary_count,
                    boundary_or_extreme_share=boundary_share,
                    score_multiset_sha256=allocation_score_multiset_sha256(
                        score_values
                    ),
                    winner_candidate_label=winner[3],
                    tie_break_used=len(top) > 1,
                    failure_codes=failure_codes,
                    passes=not failure_codes,
                )
            )

            best_forecast, best_score, best_marginal, _label, _diagnostic = winner
            selected.append(best_forecast)
            remaining.remove(best_forecast)
            members.append(
                AllocatedActionMember(
                    rank=step,
                    option_id=best_forecast.option_id,
                    option_identity_sha256=best_forecast.option_identity_sha256,
                    child_configuration_sha256=(
                        best_forecast.child_configuration_sha256
                    ),
                    family=best_forecast.family,
                    greedy_step_score=best_score,
                    marginal_total_utility=best_marginal,
                )
            )
            previous_total = best_score.total_utility

        candidate_evaluations = sum(value.candidate_count for value in step_audits)
        decision = FrameActionPortfolioDecision(
            allocation_request_sha256=allocation_request_sha256,
            frame_receipt_sha256=frame_receipt_sha256,
            source_forecast_receipt_sha256=source_forecast_receipt_sha256,
            eligible_options_sha256=eligible_options_sha256,
            members=tuple(members),
            final_score=members[-1].greedy_step_score,
            candidate_evaluations=candidate_evaluations,
            utility_policy_id=request.utility.policy_id,
            utility_policy_version=request.utility.policy_version,
            utility_definition_sha256=request.utility.definition_sha256,
            allocator_policy_id=GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
            allocator_policy_version=GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
            allocator_definition_sha256=(
                GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256
            ),
            allocator_configuration_sha256=allocator_configuration_sha256,
        )
        validate_frame_action_portfolio_decision(request, decision)
        audit = ActionAllocationSurfaceAudit(
            allocation_request_sha256=allocation_request_sha256,
            decision_receipt_sha256=decision.receipt_sha256,
            frame_receipt_sha256=frame_receipt_sha256,
            score_diagnostic=self.score_diagnostic,
            gate_policy=self.gate_policy,
            steps=tuple(step_audits),
            candidate_score_count=candidate_evaluations,
            passes=all(value.passes for value in step_audits),
        )
        return AuditedFrameActionAllocationResult(decision=decision, audit=audit)

    def allocate(
        self,
        request: FrameActionAllocationRequest,
    ) -> AuditedFrameActionAllocationResult:
        """Authorize a decision only when every bound surface gate passes."""

        result = self.assess(request)
        if not result.audit.passes:
            raise AllocationSurfaceGateRejected(result)
        return result


__all__ = [
    "AllocationSurfaceGateRejected",
    "AuditedGreedyForecastFrameAllocator",
]
