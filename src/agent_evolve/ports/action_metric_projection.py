"""Typed exact/cheap metric projections for finite action forecasts.

The language model may estimate consequences for every objective, but a
workload can sometimes derive a strict subset exactly from the sealed child
configuration without invoking the expensive evaluator.  This port records
those deltas separately so trusted application code can replace model-authored
cells without giving benchmark adapters control over selection or allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_forecast import ActionForecastRequest


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_BATCH_DOMAIN = b"agent-evolve:exact-action-metric-projection-batch:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class ExactActionMetricProjection:
    """One evaluator-independent, exact child-minus-parent metric delta."""

    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    metric_id: str
    delta: float

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(
            self.option_id
        ) is None:
            raise ValueError("option_id must use the finite-option grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )
        if type(self.metric_id) is not str or _METRIC_ID.fullmatch(
            self.metric_id
        ) is None:
            raise ValueError("metric_id must use the metric identifier grammar")
        if type(self.delta) is not float or not math.isfinite(self.delta):
            raise TypeError("delta must be a finite exact float")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "metric_id": self.metric_id,
            "delta_hex": self.delta.hex(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class ExactActionMetricProjectionBatch:
    """Receipt-bound partial metric authority for one forecast request."""

    forecast_request_sha256: str
    finite_contract_identity_sha256: str
    projections: tuple[ExactActionMetricProjection, ...]
    projector_id: str
    projector_version: int
    projector_definition_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.forecast_request_sha256, "forecast_request_sha256")
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        if type(self.projections) is not tuple or not self.projections or any(
            type(value) is not ExactActionMetricProjection
            for value in self.projections
        ):
            raise ValueError("projections must be a non-empty exact tuple")
        for value in self.projections:
            value.__post_init__()
        keys = tuple(
            (value.option_id, value.metric_id) for value in self.projections
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("projections must be unique and canonical")
        if type(self.projector_id) is not str or _TOKEN.fullmatch(
            self.projector_id
        ) is None:
            raise ValueError("projector_id must use the closed token grammar")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be a positive exact integer")
        require_sha256(
            self.projector_definition_sha256,
            "projector_definition_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "forecast_request_sha256": self.forecast_request_sha256,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "projections": [value.to_record() for value in self.projections],
            "projector": {
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "definition_sha256": self.projector_definition_sha256,
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _BATCH_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ExactActionMetricProjectionBatch
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@runtime_checkable
class ActionMetricProjector(Protocol):
    """Derive only exact/cheap metric cells; never evaluate or allocate."""

    def project(
        self,
        request: ActionForecastRequest,
    ) -> ExactActionMetricProjectionBatch: ...


__all__ = [
    "ActionMetricProjector",
    "ExactActionMetricProjection",
    "ExactActionMetricProjectionBatch",
]
