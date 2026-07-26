"""Benchmark-neutral ports and receipts for forecast-based action allocation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_forecast import (
    ActionForecastRequest,
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.portfolio_selection import (
    finite_option_ids_have_pairwise_disjoint_parent_patch_subset,
    pairwise_disjoint_parent_patch_witness,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_REQUEST_DOMAIN = b"agent-evolve:action-allocation-request:v3\x00"
_DECISION_DOMAIN = b"agent-evolve:action-portfolio-decision:v1\x00"
_ELIGIBLE_DOMAIN = b"agent-evolve:eligible-action-set:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    return value


class ForecastQuantile(str, Enum):
    P10 = "p10"
    P50 = "p50"
    P90 = "p90"


@dataclass(frozen=True, slots=True)
class ForecastPortfolioUtilityInput:
    """One benchmark utility query over a candidate forecast portfolio."""

    optimization_semantics: OptimizationSemantics
    parent_metric_values: tuple[ParentMetricValue, ...]
    metric_scales: tuple[MetricForecastScale, ...]
    members: tuple[ResolvedActionForecast, ...]
    quantile: ForecastQuantile

    def __post_init__(self) -> None:
        if type(self.optimization_semantics) is not OptimizationSemantics:
            raise TypeError("optimization_semantics must be exact")
        OptimizationSemantics.__post_init__(self.optimization_semantics)
        if type(self.parent_metric_values) is not tuple or not self.parent_metric_values:
            raise ValueError("parent_metric_values must be a non-empty exact tuple")
        if any(type(value) is not ParentMetricValue for value in self.parent_metric_values):
            raise TypeError("parent_metric_values must contain exact values")
        if type(self.metric_scales) is not tuple or not self.metric_scales:
            raise ValueError("metric_scales must be a non-empty exact tuple")
        if any(type(value) is not MetricForecastScale for value in self.metric_scales):
            raise TypeError("metric_scales must contain exact values")
        if type(self.members) is not tuple or not self.members or any(
            type(value) is not ResolvedActionForecast for value in self.members
        ):
            raise ValueError("members must be a non-empty exact resolved tuple")
        for value in self.members:
            value.__post_init__()
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("utility input cannot repeat an action")
        if type(self.quantile) is not ForecastQuantile:
            raise TypeError("quantile must be an exact ForecastQuantile")


@runtime_checkable
class ForecastPortfolioUtility(Protocol):
    """Deterministic benchmark policy; returned utility is higher-is-better."""

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float: ...


@dataclass(frozen=True, slots=True)
class ForecastPortfolioUtilityBinding:
    """Identified executable utility injected by one benchmark adapter."""

    utility: ForecastPortfolioUtility = field(repr=False, compare=False)
    policy_id: str
    policy_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.utility):
            raise TypeError("utility must be callable")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActionAllocationRequest:
    """Allocation input with a cache-safe canonical eligible-action subset."""

    forecast_request: ActionForecastRequest
    forecasts: ResolvedActionForecastBatch
    eligible_option_ids: tuple[str, ...]
    portfolio_size: int
    utility: ForecastPortfolioUtilityBinding
    min_distinct_families: int | None = None
    require_pairwise_disjoint_parent_patches: bool = False
    required_option_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.forecast_request) is not ActionForecastRequest:
            raise TypeError("forecast_request must be exact ActionForecastRequest")
        self.forecast_request.__post_init__()
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be exact ResolvedActionForecastBatch")
        validate_resolved_action_forecasts(self.forecast_request, self.forecasts)
        if type(self.eligible_option_ids) is not tuple or any(
            type(value) is not str or _OPTION_ID.fullmatch(value) is None
            for value in self.eligible_option_ids
        ):
            raise TypeError("eligible_option_ids must be an exact option-ID tuple")
        if not self.eligible_option_ids:
            raise ValueError("eligible_option_ids must be non-empty")
        if self.eligible_option_ids != tuple(sorted(set(self.eligible_option_ids))):
            raise ValueError("eligible_option_ids must be unique and canonical")
        available = {value.option_id for value in self.forecasts.forecasts}
        if not set(self.eligible_option_ids).issubset(available):
            raise ValueError("eligible_option_ids contains a foreign option")
        if type(self.portfolio_size) is not int or self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be a positive exact integer")
        if self.portfolio_size > len(self.eligible_option_ids):
            raise ValueError("portfolio_size exceeds the eligible action count")
        if type(self.utility) is not ForecastPortfolioUtilityBinding:
            raise TypeError("utility must be an exact identified binding")
        self.utility.__post_init__()
        by_id = {
            value.option_id: value
            for value in self.forecast_request.finite_variation_contract.options
        }
        available_families = {
            by_id[option_id].family for option_id in self.eligible_option_ids
        }
        if self.min_distinct_families is not None:
            if (
                type(self.min_distinct_families) is not int
                or self.min_distinct_families <= 0
            ):
                raise ValueError(
                    "min_distinct_families must be positive or None"
                )
            if self.min_distinct_families > self.portfolio_size:
                raise ValueError(
                    "min_distinct_families cannot exceed portfolio_size"
                )
            if self.min_distinct_families > len(available_families):
                raise ValueError(
                    "eligible actions cannot satisfy min_distinct_families"
                )
        if type(self.require_pairwise_disjoint_parent_patches) is not bool:
            raise TypeError(
                "require_pairwise_disjoint_parent_patches must be exact bool"
            )
        if type(self.required_option_ids) is not tuple or any(
            type(value) is not str or _OPTION_ID.fullmatch(value) is None
            for value in self.required_option_ids
        ):
            raise TypeError("required_option_ids must be an exact option-ID tuple")
        if self.required_option_ids != tuple(sorted(set(self.required_option_ids))):
            raise ValueError("required_option_ids must be unique and canonical")
        if not set(self.required_option_ids).issubset(self.eligible_option_ids):
            raise ValueError("required_option_ids contains an ineligible option")
        if len(self.required_option_ids) > self.portfolio_size:
            raise ValueError("required_option_ids exceeds the portfolio size")
        if self.require_pairwise_disjoint_parent_patches and not (
            finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
                self.forecast_request.finite_variation_contract,
                self.eligible_option_ids,
                portfolio_size=self.portfolio_size,
                min_distinct_families=self.min_distinct_families,
            )
        ):
            raise ValueError(
                "eligible actions contain no structurally feasible portfolio"
            )
        if self.require_pairwise_disjoint_parent_patches and self.required_option_ids:
            witness = pairwise_disjoint_parent_patch_witness(
                self.forecast_request.finite_variation_contract,
                self.eligible_option_ids,
                portfolio_size=self.portfolio_size,
                min_distinct_families=self.min_distinct_families,
                required_option_ids=self.required_option_ids,
            )
            if witness is None:
                raise ValueError(
                    "required_option_ids have no structurally feasible completion"
                )
        elif self.min_distinct_families is not None and self.required_option_ids:
            required_families = {
                by_id[value].family for value in self.required_option_ids
            }
            remaining_families = {
                by_id[value].family
                for value in self.eligible_option_ids
                if value not in self.required_option_ids
            }
            slots = self.portfolio_size - len(self.required_option_ids)
            reachable = len(required_families) + min(
                slots,
                len(remaining_families - required_families),
            )
            if reachable < self.min_distinct_families:
                raise ValueError(
                    "required_option_ids cannot complete minimum family coverage"
                )

    @property
    def eligible_options_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _ELIGIBLE_DOMAIN,
            {"eligible_option_ids": list(self.eligible_option_ids)},
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 3,
            "forecast_request_sha256": self.forecast_request.request_sha256,
            "forecast_receipt_sha256": self.forecasts.receipt_sha256,
            "eligible_option_ids": list(self.eligible_option_ids),
            "eligible_options_sha256": self.eligible_options_sha256,
            "portfolio_size": self.portfolio_size,
            "utility": self.utility.to_record(),
            "min_distinct_families": self.min_distinct_families,
            "require_pairwise_disjoint_parent_patches": (
                self.require_pairwise_disjoint_parent_patches
            ),
            "required_option_ids": list(self.required_option_ids),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class PortfolioAllocationScore:
    """Auditable generic score for one greedy marginal candidate."""

    p10_utility: float
    p50_utility: float
    p90_utility: float
    downside_utility: float
    risk_penalty: float
    diversity_reward: float
    total_utility: float

    def __post_init__(self) -> None:
        for name in (
            "p10_utility",
            "p50_utility",
            "p90_utility",
            "downside_utility",
            "risk_penalty",
            "diversity_reward",
            "total_utility",
        ):
            _finite_float(getattr(self, name), name)
        if self.downside_utility != min(self.p10_utility, self.p90_utility):
            raise ValueError("downside_utility must be min(p10_utility,p90_utility)")
        if self.risk_penalty < 0.0:
            raise ValueError("risk_penalty must be non-negative")
        if self.diversity_reward < 0.0:
            raise ValueError("diversity_reward must be non-negative")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            name: getattr(self, name).hex()
            for name in (
                "p10_utility",
                "p50_utility",
                "p90_utility",
                "downside_utility",
                "risk_penalty",
                "diversity_reward",
                "total_utility",
            )
        }


@dataclass(frozen=True, slots=True)
class AllocatedActionMember:
    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str
    greedy_step_score: PortfolioAllocationScore
    marginal_total_utility: float

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the finite-option grammar")
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the closed token grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )
        if type(self.greedy_step_score) is not PortfolioAllocationScore:
            raise TypeError("greedy_step_score must be exact PortfolioAllocationScore")
        self.greedy_step_score.__post_init__()
        _finite_float(self.marginal_total_utility, "marginal_total_utility")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
            "greedy_step_score": self.greedy_step_score.to_record(),
            "marginal_total_utility_hex": self.marginal_total_utility.hex(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class ActionPortfolioDecision:
    """Receipt-bound deterministic k-option allocation decision."""

    allocation_request_sha256: str
    forecast_receipt_sha256: str
    finite_contract_identity_sha256: str
    eligible_options_sha256: str
    members: tuple[AllocatedActionMember, ...]
    final_score: PortfolioAllocationScore
    candidate_evaluations: int
    utility_policy_id: str
    utility_policy_version: int
    utility_definition_sha256: str
    allocator_policy_id: str
    allocator_policy_version: int
    allocator_definition_sha256: str
    allocator_configuration_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "allocation_request_sha256",
            "forecast_receipt_sha256",
            "finite_contract_identity_sha256",
            "eligible_options_sha256",
            "utility_definition_sha256",
            "allocator_definition_sha256",
            "allocator_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.members) is not tuple or not self.members or any(
            type(value) is not AllocatedActionMember for value in self.members
        ):
            raise ValueError("members must be a non-empty exact allocated tuple")
        for value in self.members:
            value.__post_init__()
        if tuple(value.rank for value in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("member ranks must be contiguous and ordered")
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("allocation decision cannot repeat an option")
        if type(self.final_score) is not PortfolioAllocationScore:
            raise TypeError("final_score must be exact PortfolioAllocationScore")
        self.final_score.__post_init__()
        if self.final_score != self.members[-1].greedy_step_score:
            raise ValueError("final_score must equal the last greedy step score")
        if type(self.candidate_evaluations) is not int or self.candidate_evaluations <= 0:
            raise ValueError("candidate_evaluations must be a positive exact integer")
        for prefix in ("utility", "allocator"):
            policy_id = getattr(self, f"{prefix}_policy_id")
            policy_version = getattr(self, f"{prefix}_policy_version")
            if type(policy_id) is not str or _TOKEN.fullmatch(policy_id) is None:
                raise ValueError(f"{prefix}_policy_id must use the token grammar")
            if type(policy_version) is not int or policy_version <= 0:
                raise ValueError(f"{prefix}_policy_version must be positive")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "allocation_request_sha256": self.allocation_request_sha256,
            "forecast_receipt_sha256": self.forecast_receipt_sha256,
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "eligible_options_sha256": self.eligible_options_sha256,
            "members": [value.to_record() for value in self.members],
            "final_score": self.final_score.to_record(),
            "candidate_evaluations": self.candidate_evaluations,
            "utility_policy": {
                "policy_id": self.utility_policy_id,
                "policy_version": self.utility_policy_version,
                "definition_sha256": self.utility_definition_sha256,
            },
            "allocator_policy": {
                "policy_id": self.allocator_policy_id,
                "policy_version": self.allocator_policy_version,
                "definition_sha256": self.allocator_definition_sha256,
                "configuration_sha256": self.allocator_configuration_sha256,
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionPortfolioDecision
            and type(other) is ActionPortfolioDecision
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class ActionAllocationResult:
    decision: ActionPortfolioDecision

    def __post_init__(self) -> None:
        if type(self.decision) is not ActionPortfolioDecision:
            raise TypeError("decision must be exact ActionPortfolioDecision")
        self.decision.__post_init__()


@runtime_checkable
class DeterministicActionAllocator(Protocol):
    def allocate(self, request: ActionAllocationRequest) -> ActionAllocationResult: ...


def validate_action_portfolio_decision(
    request: ActionAllocationRequest,
    decision: ActionPortfolioDecision,
) -> None:
    """Validate receipt bindings and selected action identities without rerunning utility."""

    if type(request) is not ActionAllocationRequest:
        raise TypeError("request must be exact ActionAllocationRequest")
    request.__post_init__()
    if type(decision) is not ActionPortfolioDecision:
        raise TypeError("decision must be exact ActionPortfolioDecision")
    decision.__post_init__()
    if (
        decision.allocation_request_sha256 != request.request_sha256
        or decision.forecast_receipt_sha256 != request.forecasts.receipt_sha256
        or decision.finite_contract_identity_sha256
        != request.forecast_request.finite_variation_contract.identity_sha256
        or decision.eligible_options_sha256 != request.eligible_options_sha256
    ):
        raise ValueError("allocation decision is bound to a different request")
    if len(decision.members) != request.portfolio_size:
        raise ValueError("allocation member count differs from portfolio_size")
    if (
        decision.utility_policy_id != request.utility.policy_id
        or decision.utility_policy_version != request.utility.policy_version
        or decision.utility_definition_sha256 != request.utility.definition_sha256
    ):
        raise ValueError("allocation decision names a different utility policy")
    forecasts = {value.option_id: value for value in request.forecasts.forecasts}
    for member in decision.members:
        if member.option_id not in request.eligible_option_ids:
            raise ValueError("allocation selected an ineligible option")
        forecast = forecasts[member.option_id]
        if (
            member.option_identity_sha256 != forecast.option_identity_sha256
            or member.child_configuration_sha256
            != forecast.child_configuration_sha256
            or member.family != forecast.family
        ):
            raise ValueError("allocation member differs from its resolved forecast")
    if not set(request.required_option_ids).issubset(
        member.option_id for member in decision.members
    ):
        raise ValueError("allocation decision omitted a required option")


__all__ = [
    "ActionAllocationRequest",
    "ActionAllocationResult",
    "ActionPortfolioDecision",
    "AllocatedActionMember",
    "DeterministicActionAllocator",
    "ForecastPortfolioUtility",
    "ForecastPortfolioUtilityBinding",
    "ForecastPortfolioUtilityInput",
    "ForecastQuantile",
    "PortfolioAllocationScore",
    "validate_action_portfolio_decision",
]
