"""Compose forecast-geometry ballots into one protected action committee."""

from __future__ import annotations

from dataclasses import dataclass
import math

from agent_evolve.application.protected_action_committee import (
    ActionCommitteeArmBinding,
    ProtectedActionCommitteePolicy,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedSlateFeasibilityPort,
)

from .semantic_coverage_residual_portfolio import (
    ForecastGeometryRacingResidualPortfolio,
)


@dataclass(frozen=True, slots=True)
class ForecastGeometryActionCommitteePortfolio:
    """The existing consequence arms behind one outcome-blind exact-K policy."""

    source_portfolio: ForecastGeometryRacingResidualPortfolio
    allocation: ProtectedActionCommitteePolicy
    additional_arm_bindings: tuple[ActionCommitteeArmBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.source_portfolio)
            is not ForecastGeometryRacingResidualPortfolio
        ):
            raise TypeError("source_portfolio must be exact")
        self.source_portfolio.__post_init__()
        if type(self.allocation) is not ProtectedActionCommitteePolicy:
            raise TypeError("allocation must be an exact protected committee")
        self.allocation.__post_init__()
        if (
            type(self.additional_arm_bindings) is not tuple
            or any(
                type(value) is not ActionCommitteeArmBinding
                for value in self.additional_arm_bindings
            )
        ):
            raise TypeError(
                "additional_arm_bindings must contain exact bindings"
            )
        for value in self.additional_arm_bindings:
            value.__post_init__()
        additional_ids = tuple(
            value.arm_id for value in self.additional_arm_bindings
        )
        if additional_ids != tuple(sorted(set(additional_ids))):
            raise ValueError(
                "additional arm bindings must be unique and canonical"
            )
        source_identities = tuple(
            sorted(
                (
                    value.branch_id,
                    value.policy.policy_id,
                    value.policy.policy_version,
                    value.policy.definition_sha256,
                )
                for value in self.source_portfolio.planner.branch_bindings
            )
        )
        committee_identities = tuple(
            sorted(
                (
                    value.arm_id,
                    value.policy.policy_id,
                    value.policy.policy_version,
                    value.policy.definition_sha256,
                )
                for value in self.allocation.arm_bindings
            )
        )
        additional_identities = tuple(
            sorted(
                (
                    value.arm_id,
                    value.policy.policy_id,
                    value.policy.policy_version,
                    value.policy.definition_sha256,
                )
                for value in self.additional_arm_bindings
            )
        )
        if committee_identities != tuple(
            sorted((*source_identities, *additional_identities))
        ):
            raise ValueError(
                "committee policies differ from source and additional arms"
            )
        if self.allocation.protected_arm_id != "forecast_neutral":
            raise ValueError("forecast-neutral must remain the protected arm")


def compose_forecast_geometry_action_committee(
    *,
    source_portfolio: ForecastGeometryRacingResidualPortfolio,
    protected_slots: int,
    audit_slots: int,
    audit_seed_sha256: str,
    arm_weights: tuple[tuple[str, float], ...] = (),
    additional_arm_bindings: tuple[
        ActionCommitteeArmBinding,
        ...,
    ] = (),
    slate_feasibility: MaterializedSlateFeasibilityPort | None = None,
) -> ForecastGeometryActionCommitteePortfolio:
    """Replace complete-slate commitment with protected action aggregation.

    ``source_portfolio`` is an interceptable bundle of already-composed
    outcome-blind policies. Its sequential race is not executed. The committee
    reuses the exact branch policies, including any source-exposure or hard-
    feasibility wrappers, so workload adapters do not acquire another seam.
    Optional additional bindings can expose independent, workload-neutral
    support views such as raw prequential score or provider-native rank.
    """

    if (
        type(source_portfolio)
        is not ForecastGeometryRacingResidualPortfolio
    ):
        raise TypeError("source_portfolio must be exact")
    source_portfolio.__post_init__()
    source_bindings = source_portfolio.planner.branch_bindings
    arm_ids = tuple(value.branch_id for value in source_bindings)
    if (
        type(additional_arm_bindings) is not tuple
        or any(
            type(value) is not ActionCommitteeArmBinding
            for value in additional_arm_bindings
        )
    ):
        raise TypeError(
            "additional_arm_bindings must contain exact bindings"
        )
    for value in additional_arm_bindings:
        value.__post_init__()
    additional_ids = tuple(
        value.arm_id for value in additional_arm_bindings
    )
    if additional_ids != tuple(sorted(set(additional_ids))):
        raise ValueError(
            "additional arm bindings must be unique and canonical"
        )
    if set(additional_ids) & set(arm_ids):
        raise ValueError("additional arms collide with consequence arms")
    if not arm_weights:
        weight_by_arm = {arm_id: 1.0 for arm_id in arm_ids}
    else:
        if (
            type(arm_weights) is not tuple
            or arm_weights
            != tuple(sorted(arm_weights, key=lambda value: value[0]))
        ):
            raise ValueError("arm_weights must be canonical")
        weight_by_arm: dict[str, float] = {}
        for arm_id, weight in arm_weights:
            if (
                type(arm_id) is not str
                or arm_id in weight_by_arm
                or type(weight) is not float
                or not math.isfinite(weight)
                or weight <= 0.0
            ):
                raise ValueError(
                    "arm weights require unique IDs and positive finite floats"
                )
            weight_by_arm[arm_id] = weight
        if set(weight_by_arm) != set(arm_ids):
            raise ValueError("arm_weights must cover every consequence arm")

    source_arm_bindings = tuple(
        sorted(
            (
                ActionCommitteeArmBinding(
                    arm_id=value.branch_id,
                    policy=value.policy,
                    weight=weight_by_arm[value.branch_id],
                )
                for value in source_bindings
            ),
            key=lambda value: value.arm_id,
        )
    )
    allocation = ProtectedActionCommitteePolicy(
        arm_bindings=tuple(
            sorted(
                (
                    *source_arm_bindings,
                    *additional_arm_bindings,
                ),
                key=lambda value: value.arm_id,
            )
        ),
        protected_arm_id="forecast_neutral",
        protected_slots=protected_slots,
        audit_slots=audit_slots,
        audit_seed_sha256=audit_seed_sha256,
        slate_feasibility=slate_feasibility,
    )
    return ForecastGeometryActionCommitteePortfolio(
        source_portfolio=source_portfolio,
        allocation=allocation,
        additional_arm_bindings=additional_arm_bindings,
    )


__all__ = [
    "ForecastGeometryActionCommitteePortfolio",
    "compose_forecast_geometry_action_committee",
]
