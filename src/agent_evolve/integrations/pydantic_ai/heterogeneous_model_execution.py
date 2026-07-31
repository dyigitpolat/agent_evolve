"""Workload-neutral execution for heterogeneous model portfolios.

Workload adapters own candidate semantics and evaluators.  This module owns
only opaque model-lane identity, immutable execution profiles, and structured
request routing.  It therefore permits one evolutionary campaign to populate
the same typed proposal market with multiple models without introducing
model-specific branches into an application service or workload adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    OpenRouterModelExecutionProfile,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry

_LANE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_PROFILE_DOMAIN = b"agent-evolve:heterogeneous-model-profile:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class ModelLaneBinding:
    """Bind one opaque lane ID to one complete provider execution contract."""

    lane_id: str
    profile: OpenRouterModelExecutionProfile

    def __post_init__(self) -> None:
        if (
            type(self.lane_id) is not str
            or _LANE_ID.fullmatch(self.lane_id) is None
        ):
            raise ValueError(
                "lane_id must use the closed lowercase token grammar"
            )
        if type(self.profile) is not OpenRouterModelExecutionProfile:
            raise TypeError("profile must be an exact execution profile")
        self.profile.__post_init__()


@dataclass(frozen=True, slots=True)
class HeterogeneousModelExecutionProfile:
    """Canonical, workload-free portfolio of independently queued models."""

    profile_id: str
    lanes: tuple[ModelLaneBinding, ...]

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not str
            or _PROFILE_ID.fullmatch(self.profile_id) is None
        ):
            raise ValueError(
                "profile_id must use the closed lowercase token grammar"
            )
        if (
            type(self.lanes) is not tuple
            or len(self.lanes) < 2
            or any(type(value) is not ModelLaneBinding for value in self.lanes)
        ):
            raise TypeError(
                "lanes must be an exact tuple with at least two bindings"
            )
        for lane in self.lanes:
            lane.__post_init__()
        lane_ids = tuple(value.lane_id for value in self.lanes)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("model lanes must be unique and canonical")
        models = tuple(
            value.profile.requested_model for value in self.lanes
        )
        if len(models) != len(set(models)):
            raise ValueError(
                "each requested model must identify exactly one lane"
            )

    @property
    def lane_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.lane_id for value in self.lanes)

    def lane(self, lane_id: str) -> ModelLaneBinding:
        self.__post_init__()
        matches = tuple(
            value for value in self.lanes if value.lane_id == lane_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown model lane: {lane_id}")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "lanes": [
                {
                    "lane_id": value.lane_id,
                    "profile": value.profile.to_record(),
                }
                for value in self.lanes
            ],
            "routing": "operation_suffix_then_repair_call_lineage",
            "independent_provider_queues": True,
            "workload_specific_fields": [],
        }

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(
            _PROFILE_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        record["profile_sha256"] = self.profile_sha256
        return record

    def validate_telemetry(
        self,
        telemetry: AgenticCallTelemetry,
    ) -> None:
        """Delegate terminal validation to the uniquely identified lane."""

        self.__post_init__()
        matches = tuple(
            value.profile
            for value in self.lanes
            if telemetry.requested_model == value.profile.requested_model
        )
        if len(matches) != 1:
            raise ValueError(
                "telemetry does not identify exactly one model lane"
            )
        matches[0].validate_telemetry(telemetry)


@dataclass(frozen=True, slots=True)
class HeterogeneousRunnerConfig:
    """The only shared state required to instantiate independent queues."""

    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")


class StructuredRunnerPort(Protocol):
    """Minimum async surface required from a model-lane runner."""

    async def __call__(self, request: Any) -> Any:
        """Execute one structured request."""

    async def aclose(self) -> None:
        """Release runner resources."""


LaneRunnerFactory = Callable[
    [ModelLaneBinding, int],
    StructuredRunnerPort,
]


class OperationSuffixDispatchingStructuredRunner:
    """Route calls by authenticated operation and repair-call lineage.

    A proposal expert binds ``_<lane_id>`` to its operation.  Repair calls
    retain the originating call ID plus a configured suffix and are routed to
    the same queue.  No prompt, workload, objective, candidate, or provider
    field is inspected.
    """

    def __init__(
        self,
        *,
        profile: HeterogeneousModelExecutionProfile,
        runners: tuple[tuple[str, StructuredRunnerPort], ...],
        repair_call_id_suffixes: tuple[str, ...] = ("_reground",),
    ) -> None:
        profile.__post_init__()
        if (
            type(runners) is not tuple
            or any(
                type(value) is not tuple or len(value) != 2
                for value in runners
            )
        ):
            raise TypeError("runners must be an exact tuple of lane pairs")
        runner_ids = tuple(value[0] for value in runners)
        if runner_ids != profile.lane_ids:
            raise ValueError(
                "runner lanes must exactly match the canonical profile"
            )
        if (
            type(repair_call_id_suffixes) is not tuple
            or not repair_call_id_suffixes
            or any(
                type(value) is not str
                or not value.startswith("_")
                or len(value) < 2
                for value in repair_call_id_suffixes
            )
            or repair_call_id_suffixes
            != tuple(sorted(set(repair_call_id_suffixes)))
        ):
            raise ValueError(
                "repair suffixes must be a canonical non-empty tuple"
            )
        self._profile = profile
        self._runners = dict(runners)
        self._repair_call_id_suffixes = repair_call_id_suffixes
        self._route_by_call_id: dict[str, str] = {}

    def _route(self, request: Any) -> str:
        operation = getattr(request, "operation", None)
        call_id_value = getattr(getattr(request, "call_id", None), "value", None)
        if type(operation) is not str or type(call_id_value) is not str:
            raise TypeError(
                "structured request must expose string operation and call ID"
            )
        direct = tuple(
            lane_id
            for lane_id in self._profile.lane_ids
            if operation.endswith(f"_{lane_id}")
        )
        if len(direct) == 1:
            route = direct[0]
            self._route_by_call_id[call_id_value] = route
            return route
        for suffix in self._repair_call_id_suffixes:
            if not call_id_value.endswith(suffix):
                continue
            source_call_id = call_id_value.removesuffix(suffix)
            route = self._route_by_call_id.get(source_call_id)
            if route is not None:
                self._route_by_call_id[call_id_value] = route
                return route
        raise ValueError(
            "structured request has no authenticated model-lane route"
        )

    async def __call__(self, request: Any) -> Any:
        return await self._runners[self._route(request)](request)

    async def aclose(self) -> None:
        await asyncio.gather(
            *(self._runners[value].aclose() for value in self._profile.lane_ids)
        )


def compose_operation_suffix_dispatching_runner(
    *,
    profile: HeterogeneousModelExecutionProfile,
    config: HeterogeneousRunnerConfig,
    lane_runner_factory: LaneRunnerFactory,
    repair_call_id_suffixes: tuple[str, ...] = ("_reground",),
) -> OperationSuffixDispatchingStructuredRunner:
    """Instantiate one independently configured queue per opaque model lane."""

    profile.__post_init__()
    config.__post_init__()
    if not callable(lane_runner_factory):
        raise TypeError("lane_runner_factory must be callable")
    runners = tuple(
        (
            binding.lane_id,
            lane_runner_factory(binding, config.seed + ordinal),
        )
        for ordinal, binding in enumerate(profile.lanes)
    )
    return OperationSuffixDispatchingStructuredRunner(
        profile=profile,
        runners=runners,
        repair_call_id_suffixes=repair_call_id_suffixes,
    )


__all__ = [
    "HeterogeneousModelExecutionProfile",
    "HeterogeneousRunnerConfig",
    "LaneRunnerFactory",
    "ModelLaneBinding",
    "OperationSuffixDispatchingStructuredRunner",
    "StructuredRunnerPort",
    "compose_operation_suffix_dispatching_runner",
]
