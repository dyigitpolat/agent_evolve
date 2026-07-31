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
    single_path_parent_patch_option_ids,
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
FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_VERSION = 3
FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:feasible-beam-risk-diversity:v3:"
    b"canonical-combination-beam;every-retained-prefix-has-a-hard-feasible-"
    b"completion;pairwise-parent-patch-disjointness-and-minimum-distinct-"
    b"families-and-exact-generic-arm-counts-and-minimum-single-path-actions-"
    b"and-minimum-disjoint-parent-patch-pairs-are-trusted-code-constraints;"
    b"score=p50-minus-quantile-downside-"
    b"risk-plus-family-diversity;canonical-content-identity-tie-break"
).hexdigest()
_FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_V2_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:feasible-beam-risk-diversity:v2:"
    b"canonical-combination-beam;every-retained-prefix-has-a-hard-feasible-"
    b"completion;pairwise-parent-patch-disjointness-and-minimum-distinct-"
    b"families-and-exact-generic-arm-counts-are-trusted-code-constraints;"
    b"score=p50-minus-quantile-downside-"
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
                parent_metric_values=(request.forecast_request.parent_metric_values),
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
        if request.exact_arm_count_constraints:
            raise ValueError(
                "GreedyRiskAdjustedDiversityAllocator does not implement exact "
                "arm-count constraints; use FeasibleBeamActionAllocator"
            )
        if (
            request.minimum_single_path_interventions
            or request.minimum_disjoint_parent_patch_pairs
        ):
            raise ValueError(
                "GreedyRiskAdjustedDiversityAllocator does not implement minimum "
                "structural floors; use FeasibleBeamActionAllocator"
            )
        by_id = {value.option_id: value for value in request.forecasts.forecasts}
        attainable_diversity = min(
            request.portfolio_size,
            len({by_id[option_id].family for option_id in request.eligible_option_ids}),
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
                    (*selected, best_forecast),
                    attainable_diversity=attainable_diversity,
                )
                best_marginal = best_score.total_utility - previous_total
                candidate_evaluations += 1
            else:
                best_forecast = None
                best_score = None
                best_marginal = None
                for candidate in remaining:
                    portfolio = (*selected, candidate)
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
        return hashlib.sha256(_FEASIBLE_BEAM_CONFIGURATION_DOMAIN + payload).hexdigest()

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
    def _disjoint_pairs(
        request: ActionAllocationRequest,
    ) -> frozenset[frozenset[str]]:
        if (
            not request.require_pairwise_disjoint_parent_patches
            and request.minimum_disjoint_parent_patch_pairs == 0
        ):
            return frozenset()
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

    @staticmethod
    def _arm_constraints_have_completion(
        *,
        selected: tuple[int, ...],
        remaining: tuple[int, ...],
        needed: int,
        arm_constraints: tuple[tuple[dict[str, int], tuple[str, ...]], ...],
    ) -> bool:
        """Cheap exact marginal-count bound for every generic arm axis."""

        for target, arm_by_index in arm_constraints:
            selected_counts = {arm_id: 0 for arm_id in target}
            for index in selected:
                selected_counts[arm_by_index[index]] += 1
            deficits: dict[str, int] = {}
            for arm_id, count in target.items():
                deficit = count - selected_counts[arm_id]
                if deficit < 0:
                    return False
                deficits[arm_id] = deficit
            if sum(deficits.values()) != needed:
                return False
            available_counts = {arm_id: 0 for arm_id in target}
            for index in remaining:
                available_counts[arm_by_index[index]] += 1
            if any(
                available_counts[arm_id] < deficit
                for arm_id, deficit in deficits.items()
            ):
                return False
        return True

    @staticmethod
    def _structural_floors_have_completion(
        *,
        selected: tuple[int, ...],
        remaining: tuple[int, ...],
        needed: int,
        options: tuple[ResolvedActionForecast, ...],
        single_path_indices: frozenset[int],
        minimum_single_path_interventions: int,
        disjoint_pairs: frozenset[frozenset[str]],
        minimum_disjoint_parent_patch_pairs: int,
    ) -> bool:
        """Return a safe upper-bound test, exact when no slots remain."""

        selected_single = sum(value in single_path_indices for value in selected)
        if selected_single + min(
            needed,
            sum(value in single_path_indices for value in remaining),
        ) < minimum_single_path_interventions:
            return False

        def pair(left: int, right: int) -> frozenset[str]:
            return frozenset(
                (options[left].option_id, options[right].option_id)
            )

        observed_pairs = sum(
            pair(left, right) in disjoint_pairs
            for position, left in enumerate(selected)
            for right in selected[position + 1 :]
        )
        if needed == 0:
            return observed_pairs >= minimum_disjoint_parent_patch_pairs
        optimistic_pairs = (
            observed_pairs
            + len(selected) * needed
            + needed * (needed - 1) // 2
        )
        return optimistic_pairs >= minimum_disjoint_parent_patch_pairs

    @classmethod
    def _has_completion(
        cls,
        *,
        partial: tuple[int, ...],
        options: tuple[ResolvedActionForecast, ...],
        portfolio_size: int,
        min_distinct_families: int | None,
        allowed_pairs: frozenset[frozenset[str]] | None,
        disjoint_pairs: frozenset[frozenset[str]],
        single_path_indices: frozenset[int],
        minimum_single_path_interventions: int,
        minimum_disjoint_parent_patch_pairs: int,
        required_indices: frozenset[int],
        arm_constraints: tuple[tuple[dict[str, int], tuple[str, ...]], ...],
    ) -> bool:
        """Exact bounded-depth feasibility oracle for one canonical prefix."""

        slots = portfolio_size - len(partial)
        if slots < 0 or not cls._pairs_allowed(partial, options, allowed_pairs):
            return False
        missing_required = required_indices.difference(partial)
        if len(missing_required) > slots:
            return False
        families = {options[index].family for index in partial}
        all_remaining = tuple(range(partial[-1] + 1, len(options))) if partial else (
            tuple(range(len(options)))
        )
        if not cls._structural_floors_have_completion(
            selected=partial,
            remaining=all_remaining,
            needed=slots,
            options=options,
            single_path_indices=single_path_indices,
            minimum_single_path_interventions=minimum_single_path_interventions,
            disjoint_pairs=disjoint_pairs,
            minimum_disjoint_parent_patch_pairs=(
                minimum_disjoint_parent_patch_pairs
            ),
        ):
            return False
        if slots == 0:
            return (
                not missing_required
                and (
                    min_distinct_families is None
                    or len(families) >= min_distinct_families
                )
                and cls._arm_constraints_have_completion(
                    selected=partial,
                    remaining=(),
                    needed=0,
                    arm_constraints=arm_constraints,
                )
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
        if not cls._arm_constraints_have_completion(
            selected=partial,
            remaining=compatible,
            needed=slots,
            arm_constraints=arm_constraints,
        ):
            return False
        if (
            min_distinct_families is not None
            and len(families | {options[index].family for index in compatible})
            < min_distinct_families
        ):
            return False

        def visit(
            chosen: tuple[int, ...],
            remaining: tuple[int, ...],
        ) -> bool:
            needed = slots - len(chosen)
            still_required = missing_required.difference(chosen)
            if len(still_required) > needed or not still_required.issubset(remaining):
                return False
            if not cls._arm_constraints_have_completion(
                selected=(*partial, *chosen),
                remaining=remaining,
                needed=needed,
                arm_constraints=arm_constraints,
            ):
                return False
            if not cls._structural_floors_have_completion(
                selected=(*partial, *chosen),
                remaining=remaining,
                needed=needed,
                options=options,
                single_path_indices=single_path_indices,
                minimum_single_path_interventions=(
                    minimum_single_path_interventions
                ),
                disjoint_pairs=disjoint_pairs,
                minimum_disjoint_parent_patch_pairs=(
                    minimum_disjoint_parent_patch_pairs
                ),
            ):
                return False
            chosen_families = families | {options[index].family for index in chosen}
            if needed == 0:
                return not still_required and (
                    min_distinct_families is None
                    or len(chosen_families) >= min_distinct_families
                )
            if len(remaining) < needed:
                return False
            if (
                min_distinct_families is not None
                and len(
                    chosen_families | {options[index].family for index in remaining}
                )
                < min_distinct_families
            ):
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
        disjoint_pairs = self._disjoint_pairs(request)
        single_path_option_ids = set(
            single_path_parent_patch_option_ids(
                request.forecast_request.finite_variation_contract,
                request.eligible_option_ids,
            )
        )
        single_path_indices = frozenset(
            index
            for index, option in enumerate(options)
            if option.option_id in single_path_option_ids
        )
        required_option_ids = set(request.required_option_ids)
        required_indices = frozenset(
            index
            for index, option in enumerate(options)
            if option.option_id in required_option_ids
        )
        arm_constraints = tuple(
            (
                dict(constraint.target_counts),
                tuple(
                    dict(constraint.option_arm_ids)[option.option_id]
                    for option in options
                ),
            )
            for constraint in request.exact_arm_count_constraints
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
            candidates: list[tuple[tuple[int, ...], PortfolioAllocationScore]] = []
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
                        disjoint_pairs=disjoint_pairs,
                        single_path_indices=single_path_indices,
                        minimum_single_path_interventions=(
                            request.minimum_single_path_interventions
                        ),
                        minimum_disjoint_parent_patch_pairs=(
                            request.minimum_disjoint_parent_patch_pairs
                        ),
                        required_indices=required_indices,
                        arm_constraints=arm_constraints,
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
        if (
            request.min_distinct_families is not None
            and len({value.family for value in selected})
            < request.min_distinct_families
        ):
            raise RuntimeError("beam allocator violated minimum family coverage")
        if request.require_pairwise_disjoint_parent_patches:
            validate_pairwise_disjoint_parent_patch_selection(
                request.forecast_request.finite_variation_contract,
                tuple(value.option_id for value in selected),
            )
        if not required_option_ids.issubset(value.option_id for value in selected):
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
                    - (0.0 if index == 1 else prefix_scores[index - 2].total_utility)
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
                if request.minimum_single_path_interventions
                or request.minimum_disjoint_parent_patch_pairs
                else 2
            ),
            allocator_definition_sha256=(
                FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256
                if request.minimum_single_path_interventions
                or request.minimum_disjoint_parent_patch_pairs
                else _FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_V2_DEFINITION_SHA256
            ),
            allocator_configuration_sha256=self.configuration_sha256,
        )
        validate_action_portfolio_decision(request, decision)
        return ActionAllocationResult(decision=decision)


__all__ = [
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256",
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_ID",
    "FEASIBLE_BEAM_RISK_DIVERSITY_ALLOCATOR_VERSION",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_ID",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION",
    "FeasibleBeamRiskAdjustedDiversityAllocator",
    "GreedyRiskAdjustedDiversityAllocator",
]
