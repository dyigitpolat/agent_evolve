"""Operational-tie-aware greedy allocation over authenticated forecast frames."""

from __future__ import annotations

import math

from agent_evolve.ports.action_allocation import (
    AllocatedActionMember,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    PortfolioAllocationScore,
)
from agent_evolve.ports.action_allocation_frame import (
    FrameActionPortfolioDecision,
    allocation_score_multiset_sha256,
)
from agent_evolve.ports.action_allocation_frame_v3 import (
    AllocationV3Candidate,
    AllocationV3SeedSamplingLaw,
    AllocationV3StepAudit,
    GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256,
    GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID,
    GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION,
    OperationalFrameActionAllocationAudit,
    OperationalFrameActionAllocationRequest,
    OperationalFrameActionAllocationResult,
    allocation_v3_failure_codes,
    validate_operational_frame_action_allocation_result,
)
from agent_evolve.ports.action_forecast import ResolvedActionForecast


class OperationalTieAllocationRejected(RuntimeError):
    """A fail-closed v3 request produced an operational tie."""

    result: OperationalFrameActionAllocationResult

    def __init__(self, result: OperationalFrameActionAllocationResult) -> None:
        if type(result) is not OperationalFrameActionAllocationResult:
            raise TypeError("result must be an exact operational frame result")
        result.__post_init__()
        if result.audit.passes:
            raise ValueError("a passing operational allocation cannot be rejected")
        self.result = result
        RuntimeError.__init__(
            self,
            "operational allocation tie failed its prospectively bound mode",
        )


class OperationalGreedyForecastFrameAllocator:
    """Stateless allocator-v3 application service.

    Every configuration value capable of changing a decision belongs to the
    request and therefore to its receipt.  The benchmark utility remains an
    injected port, and both ``diversity_weight=0.0`` and arbitrary finite N/k
    are supported.
    """

    @staticmethod
    def _utility(
        request: OperationalFrameActionAllocationRequest,
        members: tuple[ResolvedActionForecast, ...],
        quantile: ForecastQuantile,
    ) -> float:
        base = request.allocation
        canonical_members = tuple(
            sorted(
                members,
                key=lambda value: (value.option_identity_sha256, value.option_id),
            )
        )
        value = base.utility.utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=base.frame.request.optimization_semantics,
                parent_metric_values=base.frame.request.parent_metric_values,
                metric_scales=base.frame.request.metric_scales,
                members=canonical_members,
                quantile=quantile,
            )
        )
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("bound portfolio utility must return a finite float")
        return value

    @classmethod
    def _score(
        cls,
        request: OperationalFrameActionAllocationRequest,
        members: tuple[ResolvedActionForecast, ...],
        *,
        attainable_diversity: int,
    ) -> PortfolioAllocationScore:
        if type(attainable_diversity) is not int or attainable_diversity <= 0:
            raise ValueError("attainable_diversity must be a positive exact integer")
        p10 = cls._utility(request, members, ForecastQuantile.P10)
        p50 = cls._utility(request, members, ForecastQuantile.P50)
        p90 = cls._utility(request, members, ForecastQuantile.P90)
        downside = min(p10, p90)
        risk_penalty = request.risk_aversion * max(0.0, p50 - downside)
        diversity_reward = request.diversity_weight * (
            len({member.family for member in members}) / attainable_diversity
        )
        primary = p50 - risk_penalty
        total = primary + diversity_reward
        if not all(
            math.isfinite(value)
            for value in (risk_penalty, diversity_reward, primary, total)
        ):
            raise ValueError("allocator-v3 score arithmetic became non-finite")
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
        request: OperationalFrameActionAllocationRequest,
    ) -> OperationalFrameActionAllocationResult:
        """Return the complete candidate tables, including a failing attempt."""

        if type(request) is not OperationalFrameActionAllocationRequest:
            raise TypeError("request must be an exact operational frame request")
        request.__post_init__()
        base = request.allocation
        by_id = {value.option_id: value for value in base.frame.forecasts}
        global_index_by_id = {
            forecast.option_id: global_index
            for global_index, forecast in zip(
                base.frame.global_row_indices,
                base.frame.forecasts,
                strict=True,
            )
        }
        attainable_diversity = min(
            base.portfolio_size,
            len({by_id[value].family for value in base.eligible_option_ids}),
        )
        remaining = sorted(
            (by_id[value] for value in base.eligible_option_ids),
            key=lambda value: (value.option_identity_sha256, value.option_id),
        )
        selected: list[ResolvedActionForecast] = []
        members: list[AllocatedActionMember] = []
        step_audits: list[AllocationV3StepAudit] = []
        previous_total = 0.0

        for step in range(1, base.portfolio_size + 1):
            evaluated: list[
                tuple[
                    ResolvedActionForecast,
                    PortfolioAllocationScore,
                    float,
                    str,
                    AllocationV3Candidate,
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
                    raise ValueError(
                        "allocator-v3 marginal arithmetic became non-finite"
                    )
                label = f"row_{global_index_by_id[candidate.option_id]:08d}"
                candidate_record = AllocationV3Candidate(
                    candidate_label=label,
                    option_identity_sha256=candidate.option_identity_sha256,
                    score=score,
                    marginal_total_utility=marginal,
                    public_rank_sha256=request.tie_selection.public_rank_sha256(
                        step=step,
                        option_identity_sha256=candidate.option_identity_sha256,
                    ),
                )
                evaluated.append((candidate, score, marginal, label, candidate_record))

            candidate_table = tuple(
                sorted(
                    (value[4] for value in evaluated),
                    key=lambda value: value.candidate_label,
                )
            )
            raw_top = max(value.marginal_total_utility for value in candidate_table)
            ranked_scores = sorted(
                (value.marginal_total_utility for value in candidate_table),
                reverse=True,
            )
            raw_runner_gap = (
                0.0
                if len(ranked_scores) == 1
                else ranked_scores[0] - ranked_scores[1]
            )
            if not math.isfinite(raw_runner_gap):
                raise ValueError("allocator-v3 runner-gap arithmetic became non-finite")
            raw_top_labels = tuple(
                value.candidate_label
                for value in candidate_table
                if value.marginal_total_utility == raw_top
            )
            operational = tuple(
                value
                for value in candidate_table
                if raw_top - value.marginal_total_utility
                <= request.score_resolution.maximum_indistinguishable_score_gap
            )
            winner_record = min(
                operational,
                key=lambda value: (value.public_rank_sha256, value.candidate_label),
            )
            failures = allocation_v3_failure_codes(
                mode=request.tie_selection.mode,
                operational_top_count=len(operational),
            )
            step_audit = AllocationV3StepAudit(
                step=step,
                candidates=candidate_table,
                distinct_finite_score_count=len(
                    {value.marginal_total_utility for value in candidate_table}
                ),
                raw_top_score=raw_top,
                raw_runner_gap=raw_runner_gap,
                raw_top_candidate_labels=raw_top_labels,
                operational_top_candidate_labels=tuple(
                    value.candidate_label for value in operational
                ),
                selected_candidate_label=winner_record.candidate_label,
                selected_public_rank_sha256=winner_record.public_rank_sha256,
                random_oracle_prior_weight_numerator=(
                    None
                    if request.tie_selection.seed_sampling_law
                    is AllocationV3SeedSamplingLaw.FIXED_PUBLIC
                    else 1
                ),
                random_oracle_prior_weight_denominator=(
                    None
                    if request.tie_selection.seed_sampling_law
                    is AllocationV3SeedSamplingLaw.FIXED_PUBLIC
                    else len(operational)
                ),
                score_multiset_sha256=allocation_score_multiset_sha256(
                    tuple(value.marginal_total_utility for value in candidate_table)
                ),
                failure_codes=failures,
                passes=not failures,
            )
            step_audits.append(step_audit)

            winner = next(
                value
                for value in evaluated
                if value[3] == winner_record.candidate_label
            )
            best_forecast, best_score, best_marginal, _label, _record = winner
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
            allocation_request_sha256=base.request_sha256,
            frame_receipt_sha256=base.frame.receipt_sha256,
            source_forecast_receipt_sha256=(
                base.frame.source_forecast_receipt_sha256
            ),
            eligible_options_sha256=base.eligible_options_sha256,
            members=tuple(members),
            final_score=members[-1].greedy_step_score,
            candidate_evaluations=candidate_evaluations,
            utility_policy_id=base.utility.policy_id,
            utility_policy_version=base.utility.policy_version,
            utility_definition_sha256=base.utility.definition_sha256,
            allocator_policy_id=GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID,
            allocator_policy_version=GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION,
            allocator_definition_sha256=(
                GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256
            ),
            allocator_configuration_sha256=(
                request.allocator_configuration_sha256
            ),
        )
        audit = OperationalFrameActionAllocationAudit(
            operational_request_sha256=request.request_sha256,
            base_allocation_request_sha256=base.request_sha256,
            decision_receipt_sha256=decision.receipt_sha256,
            frame_receipt_sha256=base.frame.receipt_sha256,
            score_resolution=request.score_resolution,
            tie_selection=request.tie_selection,
            steps=tuple(step_audits),
            candidate_score_count=candidate_evaluations,
            passes=all(value.passes for value in step_audits),
        )
        result = OperationalFrameActionAllocationResult(
            decision=decision,
            audit=audit,
        )
        validate_operational_frame_action_allocation_result(request, result)
        return result

    def allocate(
        self,
        request: OperationalFrameActionAllocationRequest,
    ) -> OperationalFrameActionAllocationResult:
        """Authorize only a result permitted by the prospectively bound mode."""

        result = self.assess(request)
        if not result.audit.passes:
            raise OperationalTieAllocationRejected(result)
        return result


__all__ = [
    "OperationalGreedyForecastFrameAllocator",
    "OperationalTieAllocationRejected",
]
