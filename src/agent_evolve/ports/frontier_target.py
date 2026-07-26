"""Workload-neutral port for coordinated Pareto-frontier search targets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

if TYPE_CHECKING:
    from agent_evolve.application.agentic_evolution import EvolutionCandidate
    from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_TARGET_DOMAIN = b"agent-evolve:campaign-frontier-region-target:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _finite_decimal(value: object, *, name: str) -> float:
    """Decode the canonical decimal text used by frontier-target payloads."""

    if type(value) is not str or not value:
        raise TypeError(f"{name} must be non-empty decimal text")
    try:
        decoded = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be finite decimal text") from error
    if not math.isfinite(decoded):
        raise ValueError(f"{name} must be finite decimal text")
    return decoded


@dataclass(frozen=True, slots=True)
class ObjectiveSpaceTargetAxis:
    """One authenticated parent-to-aspiration axis in raw metric units.

    This is deliberately a port-level value.  Action forecasters and set
    allocators can consume it without importing the residual-frontier policy
    that happened to produce it, and benchmark adapters never need to expose
    workload names or known-good configurations.
    """

    metric_id: str
    goal: str
    ideal: float
    reference: float
    parent_value: float
    aspiration_value: float
    signed_parent_to_aspiration_delta: float
    improving_raw_delta_sign: str

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _TOKEN.fullmatch(self.metric_id) is None:
            raise ValueError("objective target metric_id must use the closed grammar")
        if self.goal not in {"min", "max"}:
            raise ValueError("objective target goal must be min or max")
        for name in (
            "ideal",
            "reference",
            "parent_value",
            "aspiration_value",
            "signed_parent_to_aspiration_delta",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite exact float")
        if self.goal == "min" and not self.reference > self.ideal:
            raise ValueError("minimization target reference must exceed ideal")
        if self.goal == "max" and not self.ideal > self.reference:
            raise ValueError("maximization target ideal must exceed reference")
        expected_sign = "negative" if self.goal == "min" else "positive"
        if self.improving_raw_delta_sign != expected_sign:
            raise ValueError("objective target improving sign differs from its goal")
        if (
            self.aspiration_value - self.parent_value
            != self.signed_parent_to_aspiration_delta
        ):
            raise ValueError("objective target signed delta differs from its endpoints")

    def normalize(self, value: float) -> float:
        """Map raw metric units into the shared lower-is-better affine frame."""

        self.__post_init__()
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("objective target value must be a finite exact float")
        if self.goal == "min":
            return (value - self.ideal) / (self.reference - self.ideal)
        return (self.ideal - value) / (self.ideal - self.reference)

    @property
    def parent_normalized(self) -> float:
        return self.normalize(self.parent_value)

    @property
    def aspiration_normalized(self) -> float:
        return self.normalize(self.aspiration_value)

    @property
    def parent_shortfall(self) -> float:
        """One-sided distance by which the parent fails to dominate the target."""

        return max(0.0, self.parent_normalized - self.aspiration_normalized)

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal,
            "ideal_hex": self.ideal.hex(),
            "reference_hex": self.reference.hex(),
            "parent_value_hex": self.parent_value.hex(),
            "aspiration_value_hex": self.aspiration_value.hex(),
            "signed_parent_to_aspiration_delta_hex": (
                self.signed_parent_to_aspiration_delta.hex()
            ),
            "improving_raw_delta_sign": self.improving_raw_delta_sign,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveSpaceTarget:
    """Typed action-to-frontier target projected from one campaign assignment."""

    campaign_target_sha256: str
    purpose: str
    axes: tuple[ObjectiveSpaceTargetAxis, ...]

    def __post_init__(self) -> None:
        require_sha256(self.campaign_target_sha256, "campaign_target_sha256")
        if type(self.purpose) is not str or not self.purpose:
            raise ValueError("objective target purpose must be non-empty")
        if type(self.axes) is not tuple or not self.axes or any(
            type(value) is not ObjectiveSpaceTargetAxis for value in self.axes
        ):
            raise ValueError("objective target axes must be a non-empty exact tuple")
        for value in self.axes:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.axes)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("objective target axes must be unique and canonical")

    @property
    def metric_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.metric_id for value in self.axes)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "campaign_target_sha256": self.campaign_target_sha256,
            "purpose": self.purpose,
            "axes": [value.to_record() for value in self.axes],
        }


@dataclass(frozen=True, slots=True)
class CampaignPortfolioFrontierTarget:
    """One prior-only frontier target assigned to one concurrent parent lane."""

    allocator_id: str
    allocator_version: int
    definition_sha256: str
    archive_utility_snapshot_sha256: str
    lane_id: str
    parent_configuration_sha256: str
    direction_id: str
    opportunity_rank: int
    payload: FrozenJsonObject
    target_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("allocator_id", "lane_id", "direction_id"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the closed token grammar")
        if type(self.allocator_version) is not int or self.allocator_version <= 0:
            raise ValueError("allocator_version must be positive")
        if type(self.opportunity_rank) is not int or self.opportunity_rank <= 0:
            raise ValueError("opportunity_rank must be positive")
        for name in (
            "definition_sha256",
            "archive_utility_snapshot_sha256",
            "parent_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.payload) is not FrozenJsonObject
            or freeze_json(self.payload) is not self.payload
        ):
            raise TypeError("payload must be an exact frozen typed-JSON object")
        object.__setattr__(
            self,
            "target_sha256",
            hashlib.sha256(
                _TARGET_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "allocator": {
                "allocator_id": self.allocator_id,
                "allocator_version": self.allocator_version,
                "definition_sha256": self.definition_sha256,
            },
            "archive_utility_snapshot_sha256": (self.archive_utility_snapshot_sha256),
            "lane_id": self.lane_id,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "direction_id": self.direction_id,
            "opportunity_rank": self.opportunity_rank,
            "payload_sha256": typed_json_sha256(self.payload),
            "payload": thaw_json(self.payload),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "target_sha256": self.target_sha256}


def objective_space_target_from_campaign_target(
    target: CampaignPortfolioFrontierTarget,
) -> ObjectiveSpaceTarget | None:
    """Project the optional schema-v2 raw target without policy coupling.

    Historical frontier assignments did not expose raw parent/aspiration
    coordinates and therefore return ``None``.  Once the field is present it
    is fail-closed: malformed or internally inconsistent axes cannot silently
    become model or allocation evidence.
    """

    if type(target) is not CampaignPortfolioFrontierTarget:
        raise TypeError("target must be an exact CampaignPortfolioFrontierTarget")
    target.__post_init__()
    payload = thaw_json(target.payload)
    if type(payload) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("campaign frontier target payload is not an object")
    raw = payload.get("objective_space_target")
    if raw is None:
        return None
    if type(raw) is not dict or set(raw) != {"purpose", "axes"}:
        raise ValueError("objective_space_target has an invalid field set")
    rows = raw["axes"]
    if type(rows) is not list or not rows:
        raise ValueError("objective_space_target axes must be a non-empty list")
    expected_fields = {
        "metric_id",
        "goal",
        "ideal_decimal",
        "reference_decimal",
        "parent_value_decimal",
        "aspiration_value_decimal",
        "signed_parent_to_aspiration_delta_decimal",
        "improving_raw_delta_sign",
    }
    axes: list[ObjectiveSpaceTargetAxis] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError("objective_space_target axis has an invalid field set")
        axes.append(
            ObjectiveSpaceTargetAxis(
                metric_id=row["metric_id"],
                goal=row["goal"],
                ideal=_finite_decimal(
                    row["ideal_decimal"], name=f"axes[{index}].ideal_decimal"
                ),
                reference=_finite_decimal(
                    row["reference_decimal"],
                    name=f"axes[{index}].reference_decimal",
                ),
                parent_value=_finite_decimal(
                    row["parent_value_decimal"],
                    name=f"axes[{index}].parent_value_decimal",
                ),
                aspiration_value=_finite_decimal(
                    row["aspiration_value_decimal"],
                    name=f"axes[{index}].aspiration_value_decimal",
                ),
                signed_parent_to_aspiration_delta=_finite_decimal(
                    row["signed_parent_to_aspiration_delta_decimal"],
                    name=(
                        f"axes[{index}].signed_parent_to_aspiration_delta_decimal"
                    ),
                ),
                improving_raw_delta_sign=row["improving_raw_delta_sign"],
            )
        )
    return ObjectiveSpaceTarget(
        campaign_target_sha256=target.target_sha256,
        purpose=raw["purpose"],
        axes=tuple(sorted(axes, key=lambda value: value.metric_id)),
    )


def campaign_frontier_target_from_record(
    record: object,
) -> CampaignPortfolioFrontierTarget:
    """Rebuild and authenticate one serialized campaign target.

    Campaign contexts intentionally carry only typed JSON.  Downstream generic
    policies therefore need a fail-closed inverse of :meth:`to_record` rather
    than reaching back into the planner that created the target.  Recomputing
    both the payload and target digests makes the context record a sufficient
    authenticated boundary for action forecasting and allocation.
    """

    if type(record) is not dict or set(record) != {
        "schema_version",
        "allocator",
        "archive_utility_snapshot_sha256",
        "lane_id",
        "parent_configuration_sha256",
        "direction_id",
        "opportunity_rank",
        "payload_sha256",
        "payload",
        "target_sha256",
    }:
        raise ValueError("campaign frontier target record has an invalid field set")
    if record["schema_version"] != 1:
        raise ValueError("campaign frontier target record has an unsupported schema")
    allocator = record["allocator"]
    if type(allocator) is not dict or set(allocator) != {
        "allocator_id",
        "allocator_version",
        "definition_sha256",
    }:
        raise ValueError("campaign frontier target allocator record is malformed")
    payload = freeze_json(record["payload"])
    if type(payload) is not FrozenJsonObject:
        raise TypeError("campaign frontier target payload must be an object")
    if typed_json_sha256(payload) != record["payload_sha256"]:
        raise ValueError("campaign frontier target payload digest does not match")
    target = CampaignPortfolioFrontierTarget(
        allocator_id=allocator["allocator_id"],
        allocator_version=allocator["allocator_version"],
        definition_sha256=allocator["definition_sha256"],
        archive_utility_snapshot_sha256=record[
            "archive_utility_snapshot_sha256"
        ],
        lane_id=record["lane_id"],
        parent_configuration_sha256=record["parent_configuration_sha256"],
        direction_id=record["direction_id"],
        opportunity_rank=record["opportunity_rank"],
        payload=payload,
    )
    if target.target_sha256 != record["target_sha256"]:
        raise ValueError("campaign frontier target digest does not match")
    return target


@runtime_checkable
class CampaignPortfolioFrontierTargetAllocator(Protocol):
    """Assign distinct generic frontier opportunities across parent lanes."""

    allocator_id: str
    allocator_version: int
    definition_sha256: str

    def allocate(
        self,
        *,
        archive_utility: ArchiveUtilitySnapshot,
        lanes: tuple[tuple[str, EvolutionCandidate], ...],
    ) -> tuple[CampaignPortfolioFrontierTarget, ...]: ...


__all__ = [
    "CampaignPortfolioFrontierTarget",
    "CampaignPortfolioFrontierTargetAllocator",
    "ObjectiveSpaceTarget",
    "ObjectiveSpaceTargetAxis",
    "campaign_frontier_target_from_record",
    "objective_space_target_from_campaign_target",
]
