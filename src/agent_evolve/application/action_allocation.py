"""Deterministic bounded-cost allocation over resolved action forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ActionAllocationResult,
    ActionPortfolioDecision,
    AllocatedActionMember,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    PortfolioAllocationScore,
    validate_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import ResolvedActionForecast
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
    validate_pairwise_disjoint_parent_patch_selection,
)


GREEDY_RISK_DIVERSITY_ALLOCATOR_ID = "greedy_risk_diversity"
GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION = 2
GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:greedy-risk-diversity:v2:"
    b"at-each-step-score-every-eligible-extension-at-p10-p50-p90;"
    b"downside=min(p10,p90);risk=max(0,p50-downside);"
    b"attainable-diversity=min(portfolio-size,eligible-distinct-families);"
    b"diversity=selected-distinct-families/attainable-diversity;"
    b"maximize-marginal-total-with-smallest-option-identity-tie-break"
).hexdigest()
_CONFIGURATION_DOMAIN = b"agent-evolve:greedy-risk-diversity-config:v1\x00"
FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_ID = "feasible_beam_risk_diversity"
FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_VERSION = 1
FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:feasible-beam-risk-diversity:v1:"
    b"canonical-combination-beam;every-retained-prefix-has-a-hard-feasible-"
    b"completion;pairwise-parent-patch-disjointness-and-minimum-distinct-"
    b"families-are-trusted-code-constraints;score=p50-minus-quantile-downside-"
    b"risk-plus-family-diversity;canonical-content-identity-tie-break"
).hexdigest()
_FEASIBLE_BEAM_CONFIGURATION_DOMAIN = (
    b"agent-evolve:feasible-beam-risk-diversity-config:v1\x00"
)


def _configuration_sha256(risk_aversion: float, diversity_weight: float) -> str:
    payload = json.dumps(
        {
            "risk_aversion_hex": risk_aversion.hex(),
            "diversity_weight_hex": diversity_weight.hex(),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_CONFIGURATION_DOMAIN + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class GreedyRiskAdjustedDiversityAllocator:
    """Greedy marginal set allocator with O(k^2 n) elementary work.

    The benchmark-owned, identified utility scores the complete forecast set,
    so it can encode hypervolume, feasibility-first orderings, or any other
    domain semantics.  This generic policy adds only a quantile downside term
    and a fixed family-diversity reward.  It never enumerates k-combinations.
    """

    risk_aversion: float = 0.5
    diversity_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("risk_aversion", "diversity_weight"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _configuration_sha256(self.risk_aversion, self.diversity_weight)

    @staticmethod
    def _utility(
        request: ActionAllocationRequest,
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
                optimization_semantics=(
                    request.forecast_request.optimization_semantics
                ),
                parent_metric_values=(
                    request.forecast_request.parent_metric_values
                ),
                metric_scales=request.forecast_request.metric_scales,
                members=canonical_members,
                quantile=quantile,
            )
        )
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("bound portfolio utility must return a finite float")
        return value

    def _score(
        self,
        request: ActionAllocationRequest,
        members: tuple[ResolvedActionForecast, ...],
        *,
        attainable_diversity: int,
    ) -> PortfolioAllocationScore:
        if type(attainable_diversity) is not int or attainable_diversity <= 0:
            raise ValueError("attainable_diversity must be a positive exact integer")
        p10 = self._utility(request, members, ForecastQuantile.P10)
        p50 = self._utility(request, members, ForecastQuantile.P50)
        p90 = self._utility(request, members, ForecastQuantile.P90)
        downside = min(p10, p90)
        risk_penalty = self.risk_aversion * max(0.0, p50 - downside)
        diversity_reward = self.diversity_weight * (
            len({member.family for member in members}) / attainable_diversity
        )
        return PortfolioAllocationScore(
            p10_utility=p10,
            p50_utility=p50,
            p90_utility=p90,
            downside_utility=downside,
            risk_penalty=risk_penalty,
            diversity_reward=diversity_reward,
            total_utility=p50 - risk_penalty + diversity_reward,
        )

    def allocate(self, request: ActionAllocationRequest) -> ActionAllocationResult:
        if type(request) is not ActionAllocationRequest:
            raise TypeError("request must be an exact ActionAllocationRequest")
        request.__post_init__()
        self.__post_init__()
        by_id = {value.option_id: value for value in request.forecasts.forecasts}
        attainable_diversity = min(
            request.portfolio_size,
            len(
                {
                    by_id[option_id].family
                    for option_id in request.eligible_option_ids
                }
            ),
        )
        remaining = sorted(
            (by_id[option_id] for option_id in request.eligible_option_ids),
            key=lambda value: (value.option_identity_sha256, value.option_id),
        )
        required = tuple(
            value
            for value in remaining
            if value.option_id in set(request.required_option_ids)
        )
        selected: list[ResolvedActionForecast] = []
        members: list[AllocatedActionMember] = []
        previous_total = 0.0
        candidate_evaluations = 0
        for rank in range(1, request.portfolio_size + 1):
            if rank <= len(required):
                best_forecast = required[rank - 1]
                best_score = self._score(
                    request,
                    tuple((*selected, best_forecast)),
                    attainable_diversity=attainable_diversity,
                )
                best_marginal = best_score.total_utility - previous_total
                candidate_evaluations += 1
            else:
                best_forecast = None
                best_score = None
                best_marginal = None
                for candidate in remaining:
                    portfolio = tuple((*selected, candidate))
                    score = self._score(
                        request,
                        portfolio,
                        attainable_diversity=attainable_diversity,
                    )
                    marginal = score.total_utility - previous_total
                    candidate_evaluations += 1
                    # ``remaining`` is canonical; strict comparison retains the
                    # smallest option identity on an exact numerical tie.
                    if best_marginal is None or marginal > best_marginal:
                        best_forecast = candidate
                        best_score = score
                        best_marginal = marginal
            assert best_forecast is not None
            assert best_score is not None
            assert best_marginal is not None
            selected.append(best_forecast)
            remaining.remove(best_forecast)
            members.append(
                AllocatedActionMember(
                    rank=rank,
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
        decision = ActionPortfolioDecision(
            allocation_request_sha256=request.request_sha256,
            forecast_receipt_sha256=request.forecasts.receipt_sha256,
            finite_contract_identity_sha256=(
                request.forecast_request.finite_variation_contract.identity_sha256
            ),
            eligible_options_sha256=request.eligible_options_sha256,
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
            allocator_configuration_sha256=self.configuration_sha256,
        )
        validate_action_portfolio_decision(request, decision)
        return ActionAllocationResult(decision=decision)


@dataclass(frozen=True, slots=True)
class FeasibleBeamRiskAdjustedDiversityAllocator:
    """Bounded beam allocation whose every published set is hard feasible.

    The LLM forecasts consequences only.  Trusted application code owns set
    construction and cannot publish a portfolio that violates the generic
    structural constraints sealed into :class:`ActionAllocationRequest`.
    """

    risk_aversion: float = 0.5
    diversity_weight: float = 0.05
    beam_width: int = 256

    def __post_init__(self) -> None:
        for name in ("risk_aversion", "diversity_weight"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.beam_width) is not int or self.beam_width <= 0:
            raise ValueError("beam_width must be a positive exact integer")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        payload = json.dumps(
            {
                "risk_aversion_hex": self.risk_aversion.hex(),
                "diversity_weight_hex": self.diversity_weight.hex(),
                "beam_width": self.beam_width,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(
            _FEASIBLE_BEAM_CONFIGURATION_DOMAIN + payload
        ).hexdigest()

    @staticmethod
    def _allowed_pairs(
        request: ActionAllocationRequest,
    ) -> frozenset[frozenset[str]] | None:
        if not request.require_pairwise_disjoint_parent_patches:
            return None
        return frozenset(
            frozenset(value)
            for value in pairwise_disjoint_parent_patch_pairs(
                request.forecast_request.finite_variation_contract,
                request.eligible_option_ids,
            )
        )

    @staticmethod
    def _pairs_allowed(
        indices: tuple[int, ...],
        options: tuple[ResolvedActionForecast, ...],
        allowed_pairs: frozenset[frozenset[str]] | None,
    ) -> bool:
        return allowed_pairs is None or all(
            frozenset((options[left].option_id, options[right].option_id))
            in allowed_pairs
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        )

    @classmethod
    def _has_completion(
        cls,
        *,
        partial: tuple[int, ...],
        options: tuple[ResolvedActionForecast, ...],
        portfolio_size: int,
        min_distinct_families: int | None,
        allowed_pairs: frozenset[frozenset[str]] | None,
        required_indices: frozenset[int],
    ) -> bool:
        """Exact bounded-depth feasibility oracle for one canonical prefix."""

        slots = portfolio_size - len(partial)
        if slots < 0 or not cls._pairs_allowed(partial, options, allowed_pairs):
            return False
        missing_required = required_indices.difference(partial)
        if len(missing_required) > slots:
            return False
        families = {options[index].family for index in partial}
        if slots == 0:
            return not missing_required and (
                min_distinct_families is None
                or len(families) >= min_distinct_families
            )
        start = partial[-1] + 1 if partial else 0
        if any(index < start for index in missing_required):
            return False
        compatible = tuple(
            index
            for index in range(start, len(options))
            if all(
                allowed_pairs is None
                or frozenset((options[index].option_id, options[chosen].option_id))
                in allowed_pairs
                for chosen in partial
            )
        )
        if len(compatible) < slots:
            return False
        if not missing_required.issubset(compatible):
            return False
        if min_distinct_families is not None and len(
            families | {options[index].family for index in compatible}
        ) < min_distinct_families:
            return False

        def visit(
            chosen: tuple[int, ...],
            remaining: tuple[int, ...],
        ) -> bool:
            needed = slots - len(chosen)
            still_required = missing_required.difference(chosen)
            if len(still_required) > needed or not still_required.issubset(
                remaining
            ):
                return False
            chosen_families = families | {
                options[index].family for index in chosen
            }
            if needed == 0:
                return not still_required and (
                    min_distinct_families is None
                    or len(chosen_families) >= min_distinct_families
                )
            if len(remaining) < needed:
                return False
            if min_distinct_families is not None and len(
                chosen_families
                | {options[index].family for index in remaining}
            ) < min_distinct_families:
                return False
            for position, candidate in enumerate(remaining):
                if all(
                    allowed_pairs is None
                    or frozenset(
                        (options[candidate].option_id, options[prior].option_id)
                    )
                    in allowed_pairs
                    for prior in chosen
                ) and visit(
                    (*chosen, candidate),
                    remaining[position + 1 :],
                ):
                    return True
            return False

        return visit((), compatible)

    def allocate(self, request: ActionAllocationRequest) -> ActionAllocationResult:
        if type(request) is not ActionAllocationRequest:
            raise TypeError("request must be an exact ActionAllocationRequest")
        request.__post_init__()
        self.__post_init__()
        by_id = {value.option_id: value for value in request.forecasts.forecasts}
        options = tuple(
            sorted(
                (by_id[value] for value in request.eligible_option_ids),
                key=lambda item: (item.option_identity_sha256, item.option_id),
            )
        )
        allowed_pairs = self._allowed_pairs(request)
        required_option_ids = set(request.required_option_ids)
        required_indices = frozenset(
            index
            for index, option in enumerate(options)
            if option.option_id in required_option_ids
        )
        scorer = GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=self.risk_aversion,
            diversity_weight=self.diversity_weight,
        )
        attainable_diversity = min(
            request.portfolio_size,
            len({value.family for value in options}),
        )
        beam: list[tuple[tuple[int, ...], PortfolioAllocationScore]] = []
        candidate_evaluations = 0
        for depth in range(1, request.portfolio_size + 1):
            candidates: list[
                tuple[tuple[int, ...], PortfolioAllocationScore]
            ] = []
            prefixes = (((), None),) if depth == 1 else tuple(beam)
            for prefix, _ in prefixes:
                start = prefix[-1] + 1 if prefix else 0
                for index in range(start, len(options)):
                    extension = (*prefix, index)
                    if not self._has_completion(
                        partial=extension,
                        options=options,
                        portfolio_size=request.portfolio_size,
                        min_distinct_families=request.min_distinct_families,
                        allowed_pairs=allowed_pairs,
                        required_indices=required_indices,
                    ):
                        continue
                    score = scorer._score(
                        request,
                        tuple(options[value] for value in extension),
                        attainable_diversity=attainable_diversity,
                    )
                    candidate_evaluations += 1
                    candidates.append((extension, score))
            if not candidates:
                raise RuntimeError(
                    "feasibility oracle admitted no target-allocation prefix"
                )
            candidates.sort(
                key=lambda value: (
                    -value[1].total_utility,
                    tuple(options[index].option_identity_sha256 for index in value[0]),
                )
            )
            beam = candidates[: self.beam_width]

        selected_indices, final_score = beam[0]
        selected = tuple(options[index] for index in selected_indices)
        if request.min_distinct_families is not None and len(
            {value.family for value in selected}
        ) < request.min_distinct_families:
            raise RuntimeError("beam allocator violated minimum family coverage")
        if request.require_pairwise_disjoint_parent_patches:
            validate_pairwise_disjoint_parent_patch_selection(
                request.forecast_request.finite_variation_contract,
                tuple(value.option_id for value in selected),
            )
        if not required_option_ids.issubset(
            value.option_id for value in selected
        ):
            raise RuntimeError("beam allocator omitted a required option")
        prefix_scores = tuple(
            scorer._score(
                request,
                selected[:index],
                attainable_diversity=attainable_diversity,
            )
            for index in range(1, len(selected) + 1)
        )
        members = tuple(
            AllocatedActionMember(
                rank=index,
                option_id=value.option_id,
                option_identity_sha256=value.option_identity_sha256,
                child_configuration_sha256=value.child_configuration_sha256,
                family=value.family,
                greedy_step_score=prefix_scores[index - 1],
                marginal_total_utility=(
                    prefix_scores[index - 1].total_utility
                    - (
                        0.0
                        if index == 1
                        else prefix_scores[index - 2].total_utility
                    )
                ),
            )
            for index, value in enumerate(selected, start=1)
        )
        if final_score != prefix_scores[-1]:
            raise RuntimeError("beam final score differs from its selected prefix")
        decision = ActionPortfolioDecision(
            allocation_request_sha256=request.request_sha256,
            forecast_receipt_sha256=request.forecasts.receipt_sha256,
            finite_contract_identity_sha256=(
                request.forecast_request.finite_variation_contract.identity_sha256
            ),
            eligible_options_sha256=request.eligible_options_sha256,
            members=members,
            final_score=final_score,
            candidate_evaluations=candidate_evaluations,
            utility_policy_id=request.utility.policy_id,
            utility_policy_version=request.utility.policy_version,
            utility_definition_sha256=request.utility.definition_sha256,
            allocator_policy_id=FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_ID,
            allocator_policy_version=(
                FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_VERSION
            ),
            allocator_definition_sha256=(
                FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256
            ),
            allocator_configuration_sha256=self.configuration_sha256,
        )
        validate_action_portfolio_decision(request, decision)
        return ActionAllocationResult(decision=decision)


__all__ = [
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256",
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_ID",
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_VERSION",
    "FeasibleBeamRiskAdjustedDiversityAllocator",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_ID",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION",
    "GreedyRiskAdjustedDiversityAllocator",
]
