"""Rank-normalized, role-factorized value for small action portfolios.

One scalar acquisition law is brittle when consequence forecasts are weak:
optimistic uncertain actions can monopolize exploitation, while risk aversion
can suppress the global probes that discover new basins.  This module assigns
each member of a small feasible portfolio to one of three workload-neutral
roles: reliable archive exploitation, residual bridging, or epistemic probing.

Role scores are empirical-CDF normalized over the sealed eligible forecast
snapshot.  This avoids workload- and objective-scale weights.  Two exploit
slots retain joint expected-hypervolume value; bridge and probe each receive a
protected slot.  The ordinary feasible beam allocator still owns uniqueness,
family coverage, and patch compatibility.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from agent_evolve.application.action_archive_value import (
    ReliabilityAdjustedResidualCellExpectedHypervolumeUtility,
)
from agent_evolve.application.action_target_realization import (
    ResidualTargetClosurePortfolioUtility,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastRequest,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    validate_resolved_action_forecasts,
)


ROLE_FACTORIZED_ACTION_UTILITY_ID = "role_factorized_action_portfolio"
ROLE_FACTORIZED_ACTION_UTILITY_VERSION = 2
ROLE_FACTORIZED_ACTION_UTILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:role-factorized-action-portfolio:v2;"
    b"roles=reliable-archive-exploit,residual-bridge,epistemic-probe;"
    b"slots=wave-budgeted-with-default-2,1,1;"
    b"normalization=sealed-eligible-empirical-cdf-midrank;"
    b"exploit=joint-reliability-adjusted-fixed-reference-hypervolume;"
    b"bridge=singleton-residual-target-closure;"
    b"probe=one-minus-weakest-target-metric-confidence;"
    b"assignment=maximum-weight-injective-role-assignment;"
    b"workload-model-provider-family-option-text-branches=false;outcomes=false"
).hexdigest()
_BINDING_DOMAIN = b"agent-evolve:role-factorized-action-binding:v2\x00"
_ELIGIBLE_DOMAIN = b"agent-evolve:role-factorized-eligible:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    return value


class ActionAcquisitionRole(str, Enum):
    RELIABLE_ARCHIVE_EXPLOIT = "reliable_archive_exploit"
    RESIDUAL_BRIDGE = "residual_bridge"
    EPISTEMIC_PROBE = "epistemic_probe"


@dataclass(frozen=True, slots=True)
class RoleScoreRow:
    option_id: str
    exploit_rank_scores: tuple[float, float, float]
    bridge_rank_scores: tuple[float, float, float]
    probe_rank_score: float

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        for name in ("exploit_rank_scores", "bridge_rank_scores"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 3:
                raise ValueError(f"{name} must contain p10, p50, and p90")
            for value in values:
                _finite(value, name=name)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must lie in [0,1]")
        _finite(self.probe_rank_score, name="probe_rank_score")
        if not 0.0 <= self.probe_rank_score <= 1.0:
            raise ValueError("probe_rank_score must lie in [0,1]")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "exploit_rank_scores_hex": [
                value.hex() for value in self.exploit_rank_scores
            ],
            "bridge_rank_scores_hex": [
                value.hex() for value in self.bridge_rank_scores
            ],
            "probe_rank_score_hex": self.probe_rank_score.hex(),
        }


@dataclass(frozen=True, slots=True)
class RoleAssignmentMember:
    option_id: str
    role: ActionAcquisitionRole

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        if type(self.role) is not ActionAcquisitionRole:
            raise TypeError("role must be an exact ActionAcquisitionRole")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {"option_id": self.option_id, "role": self.role.value}


@dataclass(frozen=True, slots=True)
class RoleAssignmentAudit:
    """Receipt-bound explanation of one complete role assignment."""

    utility_definition_sha256: str
    forecast_receipt_sha256: str
    quantile: ForecastQuantile
    assignments: tuple[RoleAssignmentMember, ...]
    utility_value: float

    def __post_init__(self) -> None:
        require_sha256(
            self.utility_definition_sha256,
            "utility_definition_sha256",
        )
        require_sha256(self.forecast_receipt_sha256, "forecast_receipt_sha256")
        if type(self.quantile) is not ForecastQuantile:
            raise TypeError("quantile must be an exact ForecastQuantile")
        if type(self.assignments) is not tuple or not self.assignments:
            raise ValueError("assignments must be a non-empty exact tuple")
        for value in self.assignments:
            if type(value) is not RoleAssignmentMember:
                raise TypeError(
                    "assignments must contain exact RoleAssignmentMember values"
                )
            value.__post_init__()
        option_ids = tuple(value.option_id for value in self.assignments)
        if option_ids != tuple(sorted(set(option_ids))):
            raise ValueError("assignments must use unique canonical option order")
        _finite(self.utility_value, name="utility_value")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "utility_definition_sha256": self.utility_definition_sha256,
            "forecast_receipt_sha256": self.forecast_receipt_sha256,
            "quantile": self.quantile.value,
            "assignments": [value.to_record() for value in self.assignments],
            "utility_value_hex": self.utility_value.hex(),
        }


def _quantile_index(value: ForecastQuantile) -> int:
    if value is ForecastQuantile.P10:
        return 0
    if value is ForecastQuantile.P50:
        return 1
    if value is ForecastQuantile.P90:
        return 2
    raise TypeError("quantile must be exact")


def _midrank_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("role score table cannot be empty")
    ordered = sorted(values.values())
    count = len(ordered)
    result: dict[str, float] = {}
    for option_id, value in values.items():
        lower = sum(candidate < value for candidate in ordered)
        equal = sum(candidate == value for candidate in ordered)
        result[option_id] = (lower + 0.5 * equal) / count
    return result


@lru_cache(maxsize=32)
def _cached_role_sequences(
    exploit_slots: int,
    bridge_slots: int,
    probe_slots: int,
    member_count: int,
) -> tuple[tuple[ActionAcquisitionRole, ...], ...]:
    slots = (
        (ActionAcquisitionRole.RELIABLE_ARCHIVE_EXPLOIT,) * exploit_slots
        + (ActionAcquisitionRole.RESIDUAL_BRIDGE,) * bridge_slots
        + (ActionAcquisitionRole.EPISTEMIC_PROBE,) * probe_slots
    )
    return tuple(
        sorted(
            set(itertools.permutations(slots, member_count)),
            key=lambda row: tuple(value.value for value in row),
        )
    )


@dataclass(frozen=True, slots=True)
class RoleFactorizedActionPortfolioUtility:
    """Identified utility with an explicit workload-neutral role budget."""

    exploit_utility: ReliabilityAdjustedResidualCellExpectedHypervolumeUtility
    bridge_utility: ResidualTargetClosurePortfolioUtility
    forecast_receipt_sha256: str
    eligible_option_ids_sha256: str
    score_rows: tuple[RoleScoreRow, ...]
    exploit_denominators: tuple[tuple[float, ...], ...]
    exploit_slots: int = 2
    bridge_slots: int = 1
    probe_slots: int = 1
    _exploit_value_cache: dict[tuple[str, tuple[str, ...]], float] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _row_cache: dict[str, RoleScoreRow] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.exploit_utility)
            is not ReliabilityAdjustedResidualCellExpectedHypervolumeUtility
        ):
            raise TypeError("exploit_utility must be exact")
        self.exploit_utility.__post_init__()
        if type(self.bridge_utility) is not ResidualTargetClosurePortfolioUtility:
            raise TypeError("bridge_utility must be exact")
        self.bridge_utility.__post_init__()
        require_sha256(self.forecast_receipt_sha256, "forecast_receipt_sha256")
        require_sha256(
            self.eligible_option_ids_sha256,
            "eligible_option_ids_sha256",
        )
        if type(self.score_rows) is not tuple or not self.score_rows:
            raise ValueError("score_rows must be non-empty")
        for row in self.score_rows:
            if type(row) is not RoleScoreRow:
                raise TypeError("score_rows must contain exact RoleScoreRow values")
            row.__post_init__()
        option_ids = tuple(value.option_id for value in self.score_rows)
        if option_ids != tuple(sorted(set(option_ids))):
            raise ValueError("score_rows must use unique canonical option order")
        for name in ("exploit_slots", "bridge_slots", "probe_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.exploit_slots <= 0:
            raise ValueError("role allocation requires at least one exploit slot")
        if self.bridge_slots > 1 or self.probe_slots > 1:
            raise ValueError("v2 supports at most one protected bridge/probe slot")
        if self.portfolio_size > 8:
            raise ValueError("role-factorized portfolios are bounded to eight")
        if (
            type(self.exploit_denominators) is not tuple
            or len(self.exploit_denominators) != 3
        ):
            raise ValueError("exploit_denominators must contain three quantiles")
        for values in self.exploit_denominators:
            if type(values) is not tuple or len(values) != self.exploit_slots:
                raise ValueError("exploit denominator width differs from slots")
            previous = 0.0
            for value in values:
                _finite(value, name="exploit denominator")
                if value < previous or value < 0.0:
                    raise ValueError(
                        "exploit denominators must be cumulative and non-negative"
                    )
                previous = value

    @property
    def portfolio_size(self) -> int:
        return self.exploit_slots + self.bridge_slots + self.probe_slots

    @property
    def binding_definition_sha256(self) -> str:
        self.__post_init__()
        payload = {
            "schema_version": 1,
            "base_policy_definition_sha256": (
                ROLE_FACTORIZED_ACTION_UTILITY_DEFINITION_SHA256
            ),
            "exploit_utility_definition_sha256": (
                self.exploit_utility.binding_definition_sha256
            ),
            "bridge_utility_definition_sha256": (
                self.bridge_utility.binding_definition_sha256
            ),
            "forecast_receipt_sha256": self.forecast_receipt_sha256,
            "eligible_option_ids_sha256": self.eligible_option_ids_sha256,
            "role_slots": {
                ActionAcquisitionRole.RELIABLE_ARCHIVE_EXPLOIT.value: (
                    self.exploit_slots
                ),
                ActionAcquisitionRole.RESIDUAL_BRIDGE.value: self.bridge_slots,
                ActionAcquisitionRole.EPISTEMIC_PROBE.value: self.probe_slots,
            },
            "score_rows": [value.to_record() for value in self.score_rows],
            "exploit_denominators_hex": [
                [value.hex() for value in row]
                for row in self.exploit_denominators
            ],
        }
        return hashlib.sha256(
            _BINDING_DOMAIN + _canonical_json(payload)
        ).hexdigest()

    def binding(self) -> ForecastPortfolioUtilityBinding:
        return ForecastPortfolioUtilityBinding(
            utility=self,
            policy_id=ROLE_FACTORIZED_ACTION_UTILITY_ID,
            policy_version=ROLE_FACTORIZED_ACTION_UTILITY_VERSION,
            definition_sha256=self.binding_definition_sha256,
        )

    @property
    def _rows_by_id(self) -> dict[str, RoleScoreRow]:
        if not self._row_cache:
            self._row_cache.update(
                (value.option_id, value) for value in self.score_rows
            )
        return self._row_cache

    def _role_sequences(
        self,
        member_count: int,
    ) -> tuple[tuple[ActionAcquisitionRole, ...], ...]:
        if not 0 < member_count <= self.portfolio_size:
            raise ValueError("member_count lies outside the role portfolio")
        return _cached_role_sequences(
            self.exploit_slots,
            self.bridge_slots,
            self.probe_slots,
            member_count,
        )

    def _subinput(
        self,
        request: ForecastPortfolioUtilityInput,
        members: tuple[ResolvedActionForecast, ...],
    ) -> ForecastPortfolioUtilityInput:
        return ForecastPortfolioUtilityInput(
            optimization_semantics=request.optimization_semantics,
            parent_metric_values=request.parent_metric_values,
            metric_scales=request.metric_scales,
            members=members,
            quantile=request.quantile,
        )

    def _assignment_score(
        self,
        request: ForecastPortfolioUtilityInput,
        roles: tuple[ActionAcquisitionRole, ...],
        *,
        rows: dict[str, RoleScoreRow],
    ) -> float:
        index = _quantile_index(request.quantile)
        exploit_members = tuple(
            member
            for member, role in zip(request.members, roles, strict=True)
            if role is ActionAcquisitionRole.RELIABLE_ARCHIVE_EXPLOIT
        )
        score = 0.0
        if exploit_members:
            cache_key = (
                request.quantile.value,
                tuple(value.option_id for value in exploit_members),
            )
            raw = self._exploit_value_cache.get(cache_key)
            if raw is None:
                raw = self.exploit_utility(
                    self._subinput(request, exploit_members)
                )
                self._exploit_value_cache[cache_key] = raw
            denominator = self.exploit_denominators[index][
                len(exploit_members) - 1
            ]
            if denominator > 0.0:
                score += len(exploit_members) * min(1.0, raw / denominator)
            else:
                score += sum(
                    rows[value.option_id].exploit_rank_scores[index]
                    for value in exploit_members
                )
        for member, role in zip(request.members, roles, strict=True):
            row = rows.get(member.option_id)
            if row is None:
                raise ValueError("utility member is outside the sealed score table")
            if role is ActionAcquisitionRole.RESIDUAL_BRIDGE:
                score += row.bridge_rank_scores[index]
            elif role is ActionAcquisitionRole.EPISTEMIC_PROBE:
                score += row.probe_rank_score
        if not math.isfinite(score):
            raise RuntimeError("role assignment score became non-finite")
        return float(score)

    def _best_assignment(
        self,
        request: ForecastPortfolioUtilityInput,
    ) -> tuple[tuple[ActionAcquisitionRole, ...], float]:
        rows = self._rows_by_id
        if any(member.option_id not in rows for member in request.members):
            raise ValueError("utility member is outside the sealed score table")
        best_roles: tuple[ActionAcquisitionRole, ...] | None = None
        best_score: float | None = None
        for roles in self._role_sequences(len(request.members)):
            score = self._assignment_score(request, roles, rows=rows)
            if best_score is None or score > best_score:
                best_roles = roles
                best_score = score
        if best_roles is None or best_score is None:
            raise AssertionError("role assignment search produced no assignment")
        return best_roles, best_score

    def assign_roles(
        self,
        request: ForecastPortfolioUtilityInput,
    ) -> tuple[RoleAssignmentMember, ...]:
        if type(request) is not ForecastPortfolioUtilityInput:
            raise TypeError("request must be exact ForecastPortfolioUtilityInput")
        request.__post_init__()
        self.__post_init__()
        if len(request.members) > self.portfolio_size:
            raise ValueError("members exceed the role portfolio size")
        best_roles, _ = self._best_assignment(request)
        return tuple(
            RoleAssignmentMember(member.option_id, role)
            for member, role in zip(request.members, best_roles, strict=True)
        )

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        if type(request) is not ForecastPortfolioUtilityInput:
            raise TypeError("request must be exact")
        if len(request.members) > self.portfolio_size:
            raise ValueError("members exceed the role portfolio size")
        _, score = self._best_assignment(request)
        return score


def build_role_factorized_action_utility(
    *,
    forecast_request: ActionForecastRequest,
    forecasts: ResolvedActionForecastBatch,
    eligible_option_ids: tuple[str, ...],
    exploit_utility: ReliabilityAdjustedResidualCellExpectedHypervolumeUtility,
    bridge_utility: ResidualTargetClosurePortfolioUtility,
    exploit_slots: int = 2,
    bridge_slots: int = 1,
    probe_slots: int = 1,
) -> RoleFactorizedActionPortfolioUtility:
    """Freeze rank-normalized role evidence from one forecast cutoff."""

    if type(forecast_request) is not ActionForecastRequest:
        raise TypeError("forecast_request must be exact")
    forecast_request.__post_init__()
    if type(forecasts) is not ResolvedActionForecastBatch:
        raise TypeError("forecasts must be exact")
    validate_resolved_action_forecasts(forecast_request, forecasts)
    if (
        type(eligible_option_ids) is not tuple
        or not eligible_option_ids
        or eligible_option_ids != tuple(sorted(set(eligible_option_ids)))
    ):
        raise ValueError("eligible_option_ids must be a non-empty canonical tuple")
    by_id = {value.option_id: value for value in forecasts.forecasts}
    if not set(eligible_option_ids).issubset(by_id):
        raise ValueError("eligible_option_ids contains a foreign forecast")
    if (
        type(exploit_utility)
        is not ReliabilityAdjustedResidualCellExpectedHypervolumeUtility
    ):
        raise TypeError("exploit_utility must be exact")
    exploit_utility.__post_init__()
    if type(bridge_utility) is not ResidualTargetClosurePortfolioUtility:
        raise TypeError("bridge_utility must be exact")
    bridge_utility.__post_init__()
    if type(exploit_slots) is not int or exploit_slots <= 0:
        raise ValueError("exploit_slots must be a positive exact integer")
    for name, value in (
        ("bridge_slots", bridge_slots),
        ("probe_slots", probe_slots),
    ):
        if type(value) is not int or not 0 <= value <= 1:
            raise ValueError(f"{name} must be zero or one")
    if exploit_slots + bridge_slots + probe_slots > 8:
        raise ValueError("role-factorized portfolios are bounded to eight")

    quantiles = (
        ForecastQuantile.P10,
        ForecastQuantile.P50,
        ForecastQuantile.P90,
    )
    exploit_raw: list[dict[str, float]] = []
    bridge_raw: list[dict[str, float]] = []
    for quantile in quantiles:
        exploit_values: dict[str, float] = {}
        bridge_values: dict[str, float] = {}
        for option_id in eligible_option_ids:
            member = by_id[option_id]
            query = ForecastPortfolioUtilityInput(
                optimization_semantics=forecast_request.optimization_semantics,
                parent_metric_values=forecast_request.parent_metric_values,
                metric_scales=forecast_request.metric_scales,
                members=(member,),
                quantile=quantile,
            )
            exploit_values[option_id] = exploit_utility(query)
            bridge_values[option_id] = bridge_utility(query)
        exploit_raw.append(exploit_values)
        bridge_raw.append(bridge_values)
    exploit_ranks = tuple(_midrank_scores(value) for value in exploit_raw)
    bridge_ranks = tuple(_midrank_scores(value) for value in bridge_raw)

    probe_raw: dict[str, float] = {}
    target_forecast_ids = {
        value.forecast_metric_id for value in exploit_utility.aliases
    } or set(exploit_utility.target.metric_ids)
    for option_id in eligible_option_ids:
        member = by_id[option_id]
        metric_by_id = {
            value.metric_id: value for value in member.metric_forecasts
        }
        if not target_forecast_ids.issubset(metric_by_id):
            raise ValueError("forecast omits a target metric for probing")
        probe_raw[option_id] = 1.0 - min(
            metric_by_id[metric_id].confidence
            for metric_id in target_forecast_ids
        )
    probe_ranks = _midrank_scores(probe_raw)

    score_rows = tuple(
        RoleScoreRow(
            option_id=option_id,
            exploit_rank_scores=tuple(
                values[option_id] for values in exploit_ranks
            ),  # type: ignore[arg-type]
            bridge_rank_scores=tuple(
                values[option_id] for values in bridge_ranks
            ),  # type: ignore[arg-type]
            probe_rank_score=probe_ranks[option_id],
        )
        for option_id in eligible_option_ids
    )
    denominators: list[tuple[float, ...]] = []
    for values in exploit_raw:
        ordered = sorted(values.values(), reverse=True)
        cumulative = []
        total = 0.0
        for index in range(exploit_slots):
            total += ordered[index] if index < len(ordered) else 0.0
            cumulative.append(float(total))
        denominators.append(tuple(cumulative))
    eligible_sha256 = hashlib.sha256(
        _ELIGIBLE_DOMAIN
        + _canonical_json({"eligible_option_ids": list(eligible_option_ids)})
    ).hexdigest()
    return RoleFactorizedActionPortfolioUtility(
        exploit_utility=exploit_utility,
        bridge_utility=bridge_utility,
        forecast_receipt_sha256=forecasts.receipt_sha256,
        eligible_option_ids_sha256=eligible_sha256,
        score_rows=score_rows,
        exploit_denominators=tuple(denominators),
        exploit_slots=exploit_slots,
        bridge_slots=bridge_slots,
        probe_slots=probe_slots,
    )


def audit_role_factorized_action_portfolio(
    *,
    utility: RoleFactorizedActionPortfolioUtility,
    forecast_request: ActionForecastRequest,
    forecasts: ResolvedActionForecastBatch,
    selected_option_ids: tuple[str, ...],
) -> tuple[RoleAssignmentAudit, ...]:
    """Explain a final portfolio at all forecast quantiles without outcomes."""

    if type(utility) is not RoleFactorizedActionPortfolioUtility:
        raise TypeError("utility must be exact")
    utility.__post_init__()
    if type(forecast_request) is not ActionForecastRequest:
        raise TypeError("forecast_request must be exact")
    forecast_request.__post_init__()
    if type(forecasts) is not ResolvedActionForecastBatch:
        raise TypeError("forecasts must be exact")
    validate_resolved_action_forecasts(forecast_request, forecasts)
    if forecasts.receipt_sha256 != utility.forecast_receipt_sha256:
        raise ValueError("utility belongs to a foreign forecast receipt")
    if (
        type(selected_option_ids) is not tuple
        or selected_option_ids != tuple(sorted(set(selected_option_ids)))
        or len(selected_option_ids) != utility.portfolio_size
    ):
        raise ValueError(
            "selected_option_ids must be one complete canonical role portfolio"
        )
    by_id = {value.option_id: value for value in forecasts.forecasts}
    if not set(selected_option_ids).issubset(by_id):
        raise ValueError("selected_option_ids contains a foreign option")
    members = tuple(
        sorted(
            (by_id[option_id] for option_id in selected_option_ids),
            key=lambda value: (value.option_identity_sha256, value.option_id),
        )
    )
    audits: list[RoleAssignmentAudit] = []
    for quantile in (
        ForecastQuantile.P10,
        ForecastQuantile.P50,
        ForecastQuantile.P90,
    ):
        request = ForecastPortfolioUtilityInput(
            optimization_semantics=forecast_request.optimization_semantics,
            parent_metric_values=forecast_request.parent_metric_values,
            metric_scales=forecast_request.metric_scales,
            members=members,
            quantile=quantile,
        )
        assignments = tuple(
            sorted(
                utility.assign_roles(request),
                key=lambda value: value.option_id,
            )
        )
        audits.append(
            RoleAssignmentAudit(
                utility_definition_sha256=utility.binding_definition_sha256,
                forecast_receipt_sha256=forecasts.receipt_sha256,
                quantile=quantile,
                assignments=assignments,
                utility_value=utility(request),
            )
        )
    return tuple(audits)


__all__ = [
    "ActionAcquisitionRole",
    "ROLE_FACTORIZED_ACTION_UTILITY_DEFINITION_SHA256",
    "ROLE_FACTORIZED_ACTION_UTILITY_ID",
    "ROLE_FACTORIZED_ACTION_UTILITY_VERSION",
    "RoleAssignmentAudit",
    "RoleAssignmentMember",
    "RoleFactorizedActionPortfolioUtility",
    "RoleScoreRow",
    "audit_role_factorized_action_portfolio",
    "build_role_factorized_action_utility",
]
