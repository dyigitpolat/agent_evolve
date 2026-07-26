"""Generic bounded partitioning and reassembly for all-action forecasts.

The application layer owns global coverage, bounded concurrency, deterministic
reassembly, and health diagnostics.  Provider integrations only implement the
single-block port and never own option or metric identities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastBlockPolicy,
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastBlockSpec,
    ActionForecastDraft,
    ActionForecastPartitionLayout,
    ActionForecastPartitionPolicyBinding,
    ActionForecastRequest,
    ActionMetricForecast,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionForecastBlock,
    resolve_action_forecasts,
    validate_action_forecast_partition_layout,
    validate_resolved_action_forecast_block,
    validate_resolved_action_forecasts,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_BLOCK_CALL_ID_DOMAIN = b"agent-evolve:action-forecast-block-call-id:v1\x00"
_PARTITIONED_RESULT_DOMAIN = b"agent-evolve:partitioned-action-forecast-result:v1\x00"
_HEALTH_POLICY_DOMAIN = b"agent-evolve:action-forecast-health-policy:v1\x00"
_METRIC_HEALTH_DOMAIN = b"agent-evolve:action-forecast-metric-health:v2\x00"
_HEALTH_ASSESSMENT_DOMAIN = b"agent-evolve:action-forecast-health-assessment:v4\x00"
_HEALTH_SUBSET_POLICY_DOMAIN = (
    b"agent-evolve:action-forecast-health-subset-policy:v1\x00"
)
_HEALTH_BLOCK_SUBSET_DOMAIN = b"agent-evolve:action-forecast-health-block-subset:v1\x00"

LENIENT_ACTION_FORECAST_HEALTH_POLICY_ID = "lenient_normalized_forecast_health"
LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID = (
    LENIENT_ACTION_FORECAST_HEALTH_POLICY_ID
)
LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION = 2
LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:lenient-normalized-forecast-health:v2;"
    b"minimum_rows=8;extreme_abs_normalized_median=32;"
    b"collapse_share_threshold=0.95;minimum_distinct_signatures=2"
).hexdigest()
LENIENT_ACTION_FORECAST_HEALTH_POLICY_VERSION = 3
LENIENT_ACTION_FORECAST_HEALTH_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:lenient-normalized-forecast-health:v3;"
    b"minimum_rows=8;extreme_abs_normalized_median=32;"
    b"collapse_share_threshold=0.95;minimum_distinct_signatures=2;"
    b"unit_confidence_share_below_collapse_threshold=true"
).hexdigest()


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


def build_action_forecast_partition_layout(
    request: ActionForecastRequest,
    partition_policy: ActionForecastPartitionPolicyBinding,
) -> ActionForecastPartitionLayout:
    """Create the unique maximal contiguous partition under two hard bounds."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    if type(partition_policy) is not ActionForecastPartitionPolicyBinding:
        raise TypeError(
            "partition_policy must be an exact partition-policy binding"
        )
    partition_policy.__post_init__()
    metric_count = len(request.required_metric_ids)
    cell_bounded_rows = partition_policy.max_metric_cells_per_block // metric_count
    rows_per_block = min(
        partition_policy.max_rows_per_block,
        cell_bounded_rows,
    )
    if rows_per_block < 1:
        raise ValueError(
            "max_metric_cells_per_block cannot hold one complete metric row"
        )
    options = request.finite_variation_contract.options
    option_identities = tuple(option.identity_sha256 for option in options)
    blocks = tuple(
        ActionForecastBlockSpec(
            block_index=block_index,
            global_row_start=start,
            global_row_stop=min(start + rows_per_block, len(options)),
            option_identity_sha256s=option_identities[
                start : min(start + rows_per_block, len(options))
            ],
        )
        for block_index, start in enumerate(range(0, len(options), rows_per_block))
    )
    layout = ActionForecastPartitionLayout(
        finite_contract_identity_sha256=(
            request.finite_variation_contract.identity_sha256
        ),
        option_identity_sha256s=option_identities,
        metric_ids=request.required_metric_ids,
        partition_policy=partition_policy,
        blocks=blocks,
    )
    validate_action_forecast_partition_layout(request, layout)
    return layout


def action_forecast_block_call_id(
    request: ActionForecastRequest,
    layout: ActionForecastPartitionLayout,
    block: ActionForecastBlockSpec,
) -> LLMCallId:
    """Derive an opaque stable physical-call identity from sealed receipts."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    validate_action_forecast_partition_layout(request, layout)
    if type(block) is not ActionForecastBlockSpec:
        raise TypeError("block must be an exact ActionForecastBlockSpec")
    block.__post_init__()
    if block.block_index >= layout.block_count or (
        block != layout.blocks[block.block_index]
    ):
        raise ValueError("block differs from its exact partition-layout position")
    digest = _hash(
        _BLOCK_CALL_ID_DOMAIN,
        {
            "request_sha256": request.request_sha256,
            "layout_sha256": layout.layout_sha256,
            "block_spec_sha256": block.block_spec_sha256,
        },
    )
    return LLMCallId(
        f"call_action_forecast_block_{block.block_index:06d}_{digest[:32]}"
    )


def build_action_forecast_block_requests(
    request: ActionForecastRequest,
    layout: ActionForecastPartitionLayout,
) -> tuple[ActionForecastBlockRequest, ...]:
    """Materialize canonical block requests without dispatching provider work."""

    validate_action_forecast_partition_layout(request, layout)
    return tuple(
        ActionForecastBlockRequest(
            request=request,
            layout=layout,
            block=block,
            block_call_id=action_forecast_block_call_id(request, layout, block),
        )
        for block in layout.blocks
    )


def _draft_from_resolved(value: ResolvedActionForecast) -> ActionForecastDraft:
    value.__post_init__()
    return ActionForecastDraft(
        option_id=value.option_id,
        probability_valid=value.probability_valid,
        metric_forecasts=tuple(
            ActionMetricForecast(
                metric_id=metric.metric_id,
                p10_delta=metric.p10_delta,
                p50_delta=metric.p50_delta,
                p90_delta=metric.p90_delta,
                confidence=metric.confidence,
                citations=tuple(
                    ActionEvidenceCitation(
                        card_key=citation.card_key,
                        action_binding_identity_sha256=(
                            citation.action_binding_identity_sha256
                        ),
                    )
                    for citation in metric.citations
                ),
            )
            for metric in value.metric_forecasts
        ),
    )


@dataclass(frozen=True, slots=True, eq=False)
class PartitionedActionForecastResult:
    """Complete batch plus canonical physical-block provenance.

    Telemetry remains available on ``block_results`` but is deliberately not
    part of this scientific forecast receipt.  The receipt binds the resolved
    content and every physical block receipt in canonical global order.
    """

    request_sha256: str
    layout: ActionForecastPartitionLayout
    block_results: tuple[ActionForecastBlockResult, ...]
    forecasts: ResolvedActionForecastBatch

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.layout) is not ActionForecastPartitionLayout:
            raise TypeError("layout must be an exact ActionForecastPartitionLayout")
        self.layout.__post_init__()
        if type(self.block_results) is not tuple or not self.block_results or any(
            type(value) is not ActionForecastBlockResult
            for value in self.block_results
        ):
            raise ValueError(
                "block_results must be a non-empty exact tuple of block results"
            )
        for value in self.block_results:
            value.__post_init__()
        indices = tuple(value.forecasts.block_index for value in self.block_results)
        if indices != tuple(range(len(self.block_results))):
            raise ValueError("block_results must use canonical complete block order")
        if len(self.block_results) != self.layout.block_count:
            raise ValueError("block_results must completely cover the layout")
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be an exact ResolvedActionForecastBatch")
        self.forecasts.__post_init__()
        if self.forecasts.request_sha256 != self.request_sha256:
            raise ValueError("assembled forecasts differ from the partition request")
        if (
            self.forecasts.finite_contract_identity_sha256
            != self.layout.finite_contract_identity_sha256
        ):
            raise ValueError("assembled forecasts differ from the partition contract")
        policy_identity = (
            self.forecasts.policy_id,
            self.forecasts.policy_version,
            self.forecasts.policy_definition_sha256,
        )
        for block_spec, result in zip(
            self.layout.blocks,
            self.block_results,
            strict=True,
        ):
            block = result.forecasts
            if block.request_sha256 != self.request_sha256:
                raise ValueError("a block differs from the partition request")
            if block.layout_sha256 != self.layout.layout_sha256:
                raise ValueError("a block differs from the partition layout")
            if block.block_spec_sha256 != block_spec.block_spec_sha256:
                raise ValueError("a block differs from its partition-layout position")
            if (
                block.policy_id,
                block.policy_version,
                block.policy_definition_sha256,
            ) != policy_identity:
                raise ValueError("a block differs from the assembled policy")
        flattened_ids = tuple(
            forecast.option_id
            for result in self.block_results
            for forecast in result.forecasts.forecasts
        )
        if flattened_ids != tuple(
            forecast.option_id for forecast in self.forecasts.forecasts
        ):
            raise ValueError("assembled option order differs from canonical blocks")
        for forecast in self.forecasts.forecasts:
            if tuple(
                metric.metric_id for metric in forecast.metric_forecasts
            ) != self.layout.metric_ids:
                raise ValueError("assembled metrics differ from the partition layout")

    @property
    def layout_sha256(self) -> str:
        return self.layout.layout_sha256

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "layout_sha256": self.layout.layout_sha256,
            "block_receipt_sha256s": [
                value.forecasts.receipt_sha256 for value in self.block_results
            ],
            "resolved_batch_receipt_sha256": self.forecasts.receipt_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_PARTITIONED_RESULT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PartitionedActionForecastResult
            and type(other) is PartitionedActionForecastResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def assemble_partitioned_action_forecasts(
    request: ActionForecastRequest,
    layout: ActionForecastPartitionLayout,
    block_results: tuple[ActionForecastBlockResult, ...],
) -> PartitionedActionForecastResult:
    """Validate and deterministically reassemble all blocks or publish nothing."""

    validate_action_forecast_partition_layout(request, layout)
    if type(block_results) is not tuple or any(
        type(value) is not ActionForecastBlockResult for value in block_results
    ):
        raise TypeError(
            "block_results must be an exact tuple of ActionForecastBlockResult"
        )
    if len(block_results) != layout.block_count:
        raise ValueError("block results do not completely cover the partition layout")
    for value in block_results:
        value.__post_init__()
    by_index: dict[int, ActionForecastBlockResult] = {}
    for value in block_results:
        index = value.forecasts.block_index
        if index in by_index:
            raise ValueError("block results repeat a partition block")
        by_index[index] = value
    if set(by_index) != set(range(layout.block_count)):
        raise ValueError("block results contain a gap or foreign block index")
    canonical = tuple(by_index[index] for index in range(layout.block_count))
    expected_requests = build_action_forecast_block_requests(request, layout)
    for block_request, result in zip(expected_requests, canonical, strict=True):
        validate_resolved_action_forecast_block(
            block_request,
            result.forecasts,
        )

    policy_identities = {
        (
            result.forecasts.policy_id,
            result.forecasts.policy_version,
            result.forecasts.policy_definition_sha256,
        )
        for result in canonical
    }
    if len(policy_identities) != 1:
        raise ValueError("partition blocks contain policy drift")
    policy_id, policy_version, policy_definition_sha256 = next(
        iter(policy_identities)
    )
    drafts = tuple(
        _draft_from_resolved(forecast)
        for result in canonical
        for forecast in result.forecasts.forecasts
    )
    batch = resolve_action_forecasts(
        request,
        drafts,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
    )
    validate_resolved_action_forecasts(request, batch)
    return PartitionedActionForecastResult(
        request_sha256=request.request_sha256,
        layout=layout,
        block_results=canonical,
        forecasts=batch,
    )


@dataclass(frozen=True, slots=True)
class ActionForecastBlockFailure:
    block_index: int
    block_request_sha256: str
    error_type: str
    error_message: str

    def __post_init__(self) -> None:
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        require_sha256(self.block_request_sha256, "block_request_sha256")
        if type(self.error_type) is not str or not self.error_type:
            raise ValueError("error_type must be non-empty text")
        if type(self.error_message) is not str:
            raise TypeError("error_message must be exact text")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "block_index": self.block_index,
            "block_request_sha256": self.block_request_sha256,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class ActionForecastWaveError(RuntimeError):
    """All blocks settled, but at least one failed and no batch was published."""

    def __init__(
        self,
        *,
        successful_results: tuple[ActionForecastBlockResult, ...],
        failures: tuple[ActionForecastBlockFailure, ...],
    ) -> None:
        if type(successful_results) is not tuple or any(
            type(value) is not ActionForecastBlockResult
            for value in successful_results
        ):
            raise TypeError("successful_results must be an exact result tuple")
        if type(failures) is not tuple or not failures or any(
            type(value) is not ActionForecastBlockFailure for value in failures
        ):
            raise ValueError("failures must be a non-empty exact failure tuple")
        self.successful_results = successful_results
        self.failures = failures
        super().__init__(
            f"{len(failures)} action-forecast block(s) failed after all settled"
        )


@runtime_checkable
class PartitionedActionForecastPolicy(Protocol):
    async def forecast_partitioned(
        self,
        request: ActionForecastRequest,
        layout: ActionForecastPartitionLayout,
    ) -> PartitionedActionForecastResult: ...


@dataclass(slots=True)
class ConcurrentActionForecastWave:
    """Queue-agnostic bounded worker coordinator for one partitioned request."""

    block_policy: ActionForecastBlockPolicy
    max_concurrency: int

    def __post_init__(self) -> None:
        if not callable(getattr(self.block_policy, "forecast_block", None)):
            raise TypeError("block_policy must expose forecast_block")
        if type(self.max_concurrency) is not int or self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive exact integer")

    async def forecast_partitioned(
        self,
        request: ActionForecastRequest,
        layout: ActionForecastPartitionLayout,
    ) -> PartitionedActionForecastResult:
        self.__post_init__()
        block_requests = build_action_forecast_block_requests(request, layout)
        results: list[ActionForecastBlockResult | None] = [
            None for _ in block_requests
        ]
        failures: list[ActionForecastBlockFailure] = []
        next_position = 0

        async def worker() -> None:
            nonlocal next_position
            while next_position < len(block_requests):
                position = next_position
                next_position += 1
                block_request = block_requests[position]
                try:
                    result = await self.block_policy.forecast_block(block_request)
                    if type(result) is not ActionForecastBlockResult:
                        raise TypeError(
                            "block_policy returned a non-ActionForecastBlockResult"
                        )
                    result.__post_init__()
                    validate_resolved_action_forecast_block(
                        block_request,
                        result.forecasts,
                    )
                    results[position] = result
                except Exception as exc:
                    failures.append(
                        ActionForecastBlockFailure(
                            block_index=block_request.block.block_index,
                            block_request_sha256=(
                                block_request.block_request_sha256
                            ),
                            error_type=(
                                f"{type(exc).__module__}.{type(exc).__qualname__}"
                            ),
                            error_message=str(exc),
                        )
                    )

        worker_count = min(self.max_concurrency, len(block_requests))
        workers = tuple(asyncio.create_task(worker()) for _ in range(worker_count))
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        successful = tuple(value for value in results if value is not None)
        if failures:
            raise ActionForecastWaveError(
                successful_results=successful,
                failures=tuple(sorted(failures, key=lambda value: value.block_index)),
            )
        return assemble_partitioned_action_forecasts(
            request,
            layout,
            tuple(value for value in results if value is not None),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastHealthPolicyBinding:
    """Identified, benchmark-neutral normalized forecast health thresholds."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    minimum_rows: int
    extreme_abs_normalized_median: float
    collapse_share_threshold: float
    minimum_distinct_signatures: int

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.minimum_rows) is not int or self.minimum_rows <= 0:
            raise ValueError("minimum_rows must be a positive exact integer")
        _finite_float(
            self.extreme_abs_normalized_median,
            "extreme_abs_normalized_median",
        )
        if self.extreme_abs_normalized_median <= 0.0:
            raise ValueError("extreme_abs_normalized_median must be positive")
        _finite_float(self.collapse_share_threshold, "collapse_share_threshold")
        if not 0.0 < self.collapse_share_threshold <= 1.0:
            raise ValueError("collapse_share_threshold must lie in (0,1]")
        if (
            type(self.minimum_distinct_signatures) is not int
            or self.minimum_distinct_signatures < 2
        ):
            raise ValueError("minimum_distinct_signatures must be at least two")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "minimum_rows": self.minimum_rows,
            "extreme_abs_normalized_median_hex": (
                self.extreme_abs_normalized_median.hex()
            ),
            "collapse_share_threshold_hex": self.collapse_share_threshold.hex(),
            "minimum_distinct_signatures": self.minimum_distinct_signatures,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_HEALTH_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastHealthPolicyBinding
            and type(other) is ActionForecastHealthPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


def lenient_action_forecast_health_policy() -> ActionForecastHealthPolicyBinding:
    return ActionForecastHealthPolicyBinding(
        policy_id=LENIENT_ACTION_FORECAST_HEALTH_POLICY_ID,
        policy_version=LENIENT_ACTION_FORECAST_HEALTH_POLICY_VERSION,
        policy_definition_sha256=(
            LENIENT_ACTION_FORECAST_HEALTH_POLICY_DEFINITION_SHA256
        ),
        minimum_rows=8,
        extreme_abs_normalized_median=32.0,
        collapse_share_threshold=0.95,
        minimum_distinct_signatures=2,
    )


def lenient_action_forecast_health_v2_policy() -> ActionForecastHealthPolicyBinding:
    """Reconstruct the frozen v2 policy binding for historical protocols.

    New experiments must use :func:`lenient_action_forecast_health_policy`.
    This named factory exists only so a sealed v2 protocol never silently
    inherits later default-policy revisions.
    """

    return ActionForecastHealthPolicyBinding(
        policy_id=LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID,
        policy_version=LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION,
        policy_definition_sha256=(
            LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256
        ),
        minimum_rows=8,
        extreme_abs_normalized_median=32.0,
        collapse_share_threshold=0.95,
        minimum_distinct_signatures=2,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastMetricHealth:
    metric_id: str
    row_count: int
    extreme_median_share: float
    largest_median_bucket_share: float
    zero_width_share: float
    unit_confidence_share: float
    distinct_cell_signature_count: int
    distinct_confidence_count: int
    max_abs_normalized_median: float
    threshold_applied: bool
    passes: bool

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be non-empty text")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("row_count must be a positive exact integer")
        for name in (
            "extreme_median_share",
            "largest_median_bucket_share",
            "zero_width_share",
            "unit_confidence_share",
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "distinct_cell_signature_count",
            "distinct_confidence_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        _finite_float(
            self.max_abs_normalized_median,
            "max_abs_normalized_median",
        )
        if self.max_abs_normalized_median < 0.0:
            raise ValueError("max_abs_normalized_median cannot be negative")
        if type(self.threshold_applied) is not bool or type(self.passes) is not bool:
            raise TypeError("threshold_applied and passes must be exact booleans")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "metric_id": self.metric_id,
            "row_count": self.row_count,
            "extreme_median_share_hex": self.extreme_median_share.hex(),
            "largest_median_bucket_share_hex": (
                self.largest_median_bucket_share.hex()
            ),
            "zero_width_share_hex": self.zero_width_share.hex(),
            "unit_confidence_share_hex": self.unit_confidence_share.hex(),
            "distinct_cell_signature_count": self.distinct_cell_signature_count,
            "distinct_confidence_count": self.distinct_confidence_count,
            "max_abs_normalized_median_hex": (
                self.max_abs_normalized_median.hex()
            ),
            "threshold_applied": self.threshold_applied,
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_METRIC_HEALTH_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastMetricHealth
            and type(other) is ActionForecastMetricHealth
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastHealthSubsetPolicyBinding:
    """Identified scientific rule selecting rows from one validated block."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_HEALTH_SUBSET_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastHealthSubsetPolicyBinding
            and type(other) is ActionForecastHealthSubsetPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastBlockHealthSubsetBinding:
    """Authenticated ordered row subset of one exact resolved partition block."""

    subset_policy: ActionForecastHealthSubsetPolicyBinding
    request_sha256: str
    layout_sha256: str
    block_request_sha256: str
    block_spec_sha256: str
    parent_block_receipt_sha256: str
    block_index: int
    global_row_start: int
    global_row_stop: int
    included_global_row_indices: tuple[int, ...]
    included_option_identity_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.subset_policy) is not ActionForecastHealthSubsetPolicyBinding:
            raise TypeError("subset_policy must be an exact subset-policy binding")
        self.subset_policy.__post_init__()
        for name in (
            "request_sha256",
            "layout_sha256",
            "block_request_sha256",
            "block_spec_sha256",
            "parent_block_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        if type(self.global_row_start) is not int or self.global_row_start < 0:
            raise ValueError("global_row_start must be a non-negative exact integer")
        if (
            type(self.global_row_stop) is not int
            or self.global_row_stop <= self.global_row_start
        ):
            raise ValueError("global_row_stop must be greater than global_row_start")
        if type(self.included_global_row_indices) is not tuple or not (
            self.included_global_row_indices
        ) or any(
            type(value) is not int for value in self.included_global_row_indices
        ):
            raise ValueError(
                "included_global_row_indices must be a non-empty exact tuple"
            )
        if self.included_global_row_indices != tuple(
            sorted(set(self.included_global_row_indices))
        ):
            raise ValueError(
                "included global rows must be unique and in block order"
            )
        if any(
            value < self.global_row_start or value >= self.global_row_stop
            for value in self.included_global_row_indices
        ):
            raise ValueError("an included global row is outside the parent block")
        if type(self.included_option_identity_sha256s) is not tuple or any(
            type(value) is not str
            for value in self.included_option_identity_sha256s
        ):
            raise TypeError(
                "included_option_identity_sha256s must be an exact tuple"
            )
        if len(self.included_option_identity_sha256s) != len(
            self.included_global_row_indices
        ):
            raise ValueError("included row and option identity counts must match")
        for index, value in enumerate(self.included_option_identity_sha256s):
            require_sha256(value, f"included_option_identity_sha256s[{index}]")
        if len(set(self.included_option_identity_sha256s)) != len(
            self.included_option_identity_sha256s
        ):
            raise ValueError("an authenticated subset cannot repeat an option")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "subset_policy": self.subset_policy.to_record(),
            "request_sha256": self.request_sha256,
            "layout_sha256": self.layout_sha256,
            "block_request_sha256": self.block_request_sha256,
            "block_spec_sha256": self.block_spec_sha256,
            "parent_block_receipt_sha256": self.parent_block_receipt_sha256,
            "block_index": self.block_index,
            "global_row_start": self.global_row_start,
            "global_row_stop": self.global_row_stop,
            "included_global_row_indices": list(
                self.included_global_row_indices
            ),
            "included_option_identity_sha256s": list(
                self.included_option_identity_sha256s
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_HEALTH_BLOCK_SUBSET_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastBlockHealthSubsetBinding
            and type(other) is ActionForecastBlockHealthSubsetBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


class ActionForecastHealthFrameKind(str, Enum):
    COMPLETE = "complete"
    PARTITION_BLOCK = "partition_block"
    PARTITION_BLOCK_SUBSET = "partition_block_subset"


@dataclass(frozen=True, slots=True)
class _NormalizedActionForecastHealth:
    metric_assessments: tuple[ActionForecastMetricHealth, ...]
    distinct_row_signature_count: int
    distinct_probability_valid_count: int
    constant_confidence_metric_ids: tuple[str, ...]
    threshold_applied: bool
    passes: bool


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedActionForecastHealthAssessment:
    member_id: str
    frame_kind: ActionForecastHealthFrameKind
    frame_receipt_sha256: str
    request_sha256: str
    layout_sha256: str | None
    block_request_sha256: str | None
    block_spec_sha256: str | None
    block_index: int | None
    global_row_start: int
    global_row_stop: int
    subset_binding: ActionForecastBlockHealthSubsetBinding | None
    health_policy: ActionForecastHealthPolicyBinding
    metric_assessments: tuple[ActionForecastMetricHealth, ...]
    distinct_row_signature_count: int
    distinct_probability_valid_count: int
    constant_confidence_metric_ids: tuple[str, ...]
    threshold_applied: bool
    passes: bool

    def __post_init__(self) -> None:
        if type(self.member_id) is not str or _TOKEN.fullmatch(self.member_id) is None:
            raise ValueError("member_id must use the closed token grammar")
        if type(self.frame_kind) is not ActionForecastHealthFrameKind:
            raise TypeError("frame_kind must be an exact health-frame kind")
        require_sha256(self.frame_receipt_sha256, "frame_receipt_sha256")
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.global_row_start) is not int or self.global_row_start < 0:
            raise ValueError("global_row_start must be a non-negative exact integer")
        if (
            type(self.global_row_stop) is not int
            or self.global_row_stop <= self.global_row_start
        ):
            raise ValueError("global_row_stop must be greater than global_row_start")
        if self.frame_kind is ActionForecastHealthFrameKind.COMPLETE:
            if self.global_row_start != 0:
                raise ValueError("complete health frames must start at global row zero")
            if any(
                value is not None
                for value in (
                    self.layout_sha256,
                    self.block_request_sha256,
                    self.block_spec_sha256,
                    self.block_index,
                )
            ):
                raise ValueError("complete health frames forbid block identities")
            if self.subset_binding is not None:
                raise ValueError("complete health frames forbid subset bindings")
        else:
            for name in (
                "layout_sha256",
                "block_request_sha256",
                "block_spec_sha256",
            ):
                value = getattr(self, name)
                if type(value) is not str:
                    raise TypeError(f"partition-block health requires {name}")
                require_sha256(value, name)
            if type(self.block_index) is not int or self.block_index < 0:
                raise ValueError(
                    "partition-block health requires a non-negative block_index"
                )
            if (
                self.frame_kind
                is ActionForecastHealthFrameKind.PARTITION_BLOCK
            ):
                if self.subset_binding is not None:
                    raise ValueError(
                        "full partition-block health forbids subset bindings"
                    )
            else:
                if type(self.subset_binding) is not (
                    ActionForecastBlockHealthSubsetBinding
                ):
                    raise TypeError(
                        "partition-block-subset health requires a subset binding"
                    )
                self.subset_binding.__post_init__()
                if (
                    self.frame_receipt_sha256
                    != self.subset_binding.binding_sha256
                    or self.request_sha256 != self.subset_binding.request_sha256
                    or self.layout_sha256 != self.subset_binding.layout_sha256
                    or self.block_request_sha256
                    != self.subset_binding.block_request_sha256
                    or self.block_spec_sha256
                    != self.subset_binding.block_spec_sha256
                    or self.block_index != self.subset_binding.block_index
                    or self.global_row_start
                    != self.subset_binding.global_row_start
                    or self.global_row_stop != self.subset_binding.global_row_stop
                ):
                    raise ValueError(
                        "subset health frame differs from its authenticated binding"
                    )
        if type(self.health_policy) is not ActionForecastHealthPolicyBinding:
            raise TypeError("health_policy must be an exact health-policy binding")
        self.health_policy.__post_init__()
        if type(self.metric_assessments) is not tuple or not (
            self.metric_assessments
        ) or any(
            type(value) is not ActionForecastMetricHealth
            for value in self.metric_assessments
        ):
            raise ValueError("metric_assessments must be a non-empty exact tuple")
        for value in self.metric_assessments:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.metric_assessments)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("metric assessments must use canonical unique order")
        row_count = (
            len(self.subset_binding.included_global_row_indices)
            if self.frame_kind
            is ActionForecastHealthFrameKind.PARTITION_BLOCK_SUBSET
            and self.subset_binding is not None
            else self.global_row_stop - self.global_row_start
        )
        if any(value.row_count != row_count for value in self.metric_assessments):
            raise ValueError("metric health row counts differ from the bound frame")
        for name in (
            "distinct_row_signature_count",
            "distinct_probability_valid_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
            if value > row_count:
                raise ValueError(f"{name} cannot exceed the bound frame row count")
        if type(self.constant_confidence_metric_ids) is not tuple or any(
            type(value) is not str for value in self.constant_confidence_metric_ids
        ):
            raise TypeError("constant_confidence_metric_ids must be an exact tuple")
        if self.constant_confidence_metric_ids != tuple(
            sorted(set(self.constant_confidence_metric_ids))
        ):
            raise ValueError(
                "constant_confidence_metric_ids must be unique and canonical"
            )
        if not set(self.constant_confidence_metric_ids).issubset(metric_ids):
            raise ValueError("constant-confidence metrics escape the assessment")
        if type(self.threshold_applied) is not bool or type(self.passes) is not bool:
            raise TypeError("threshold_applied and passes must be exact booleans")
        expected_threshold_applied = row_count >= self.health_policy.minimum_rows
        if self.threshold_applied is not expected_threshold_applied:
            raise ValueError("threshold_applied differs from the bound frame size")
        if any(
            value.threshold_applied is not self.threshold_applied
            for value in self.metric_assessments
        ):
            raise ValueError("metric health threshold application is inconsistent")
        rows_pass = not self.threshold_applied or (
            self.distinct_row_signature_count
            >= self.health_policy.minimum_distinct_signatures
        )
        expected_passes = rows_pass and all(
            value.passes for value in self.metric_assessments
        )
        if self.passes is not expected_passes:
            raise ValueError("passes differs from the normalized health gates")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 4,
            "member_id": self.member_id,
            "frame_kind": self.frame_kind.value,
            "frame_receipt_sha256": self.frame_receipt_sha256,
            "request_sha256": self.request_sha256,
            "layout_sha256": self.layout_sha256,
            "block_request_sha256": self.block_request_sha256,
            "block_spec_sha256": self.block_spec_sha256,
            "block_index": self.block_index,
            "global_row_start": self.global_row_start,
            "global_row_stop": self.global_row_stop,
            "subset_binding": (
                None
                if self.subset_binding is None
                else self.subset_binding.to_record()
            ),
            "health_policy": self.health_policy.to_record(),
            "metric_assessments": [
                value.to_record() for value in self.metric_assessments
            ],
            "distinct_row_signature_count": self.distinct_row_signature_count,
            "distinct_probability_valid_count": (
                self.distinct_probability_valid_count
            ),
            "constant_confidence_metric_ids": list(
                self.constant_confidence_metric_ids
            ),
            "threshold_applied": self.threshold_applied,
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_HEALTH_ASSESSMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ResolvedActionForecastHealthAssessment
            and type(other) is ResolvedActionForecastHealthAssessment
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def _normalized(value: float, scale: float) -> float:
    result = value / scale
    if not math.isfinite(result):
        raise ValueError("normalized forecast component must remain finite")
    return 0.0 if result == 0.0 else result


def _validate_health_inputs(
    *,
    member_id: str,
    health_policy: ActionForecastHealthPolicyBinding,
) -> None:
    if type(member_id) is not str or _TOKEN.fullmatch(member_id) is None:
        raise ValueError("member_id must use the closed token grammar")
    if type(health_policy) is not ActionForecastHealthPolicyBinding:
        raise TypeError("health_policy must be an exact health-policy binding")
    health_policy.__post_init__()


def _assess_normalized_forecasts(
    forecasts: tuple[ResolvedActionForecast, ...],
    request: ActionForecastRequest,
    health_policy: ActionForecastHealthPolicyBinding,
) -> _NormalizedActionForecastHealth:
    """Apply one normalized gate implementation to any validated row frame."""

    if type(forecasts) is not tuple or not forecasts or any(
        type(value) is not ResolvedActionForecast for value in forecasts
    ):
        raise ValueError("forecasts must be a non-empty exact resolved tuple")
    for value in forecasts:
        value.__post_init__()
    row_count = len(forecasts)
    threshold_applied = row_count >= health_policy.minimum_rows
    metric_assessments: list[ActionForecastMetricHealth] = []
    constant_confidence_metric_ids: list[str] = []
    row_signatures: set[tuple[object, ...]] = set()
    probability_buckets: set[str] = set()

    for forecast in forecasts:
        probability_buckets.add(forecast.probability_valid.hex())
        row_signature: list[object] = [forecast.probability_valid.hex()]
        for metric, scale in zip(
            forecast.metric_forecasts,
            request.metric_scales,
            strict=True,
        ):
            median = _normalized(metric.p50_delta, scale.delta_scale)
            lower = _normalized(
                metric.p50_delta - metric.p10_delta,
                scale.delta_scale,
            )
            upper = _normalized(
                metric.p90_delta - metric.p50_delta,
                scale.delta_scale,
            )
            row_signature.append(
                (median.hex(), lower.hex(), upper.hex(), metric.confidence.hex())
            )
        row_signatures.add(tuple(row_signature))

    for metric_index, scale in enumerate(request.metric_scales):
        median_buckets: Counter[str] = Counter()
        cell_signatures: set[tuple[str, str, str, str]] = set()
        confidence_buckets: set[str] = set()
        extreme_count = 0
        zero_width_count = 0
        unit_confidence_count = 0
        max_abs_median = 0.0
        for forecast in forecasts:
            metric = forecast.metric_forecasts[metric_index]
            median = _normalized(metric.p50_delta, scale.delta_scale)
            lower = _normalized(
                metric.p50_delta - metric.p10_delta,
                scale.delta_scale,
            )
            upper = _normalized(
                metric.p90_delta - metric.p50_delta,
                scale.delta_scale,
            )
            median_buckets[median.hex()] += 1
            confidence_buckets.add(metric.confidence.hex())
            cell_signatures.add(
                (median.hex(), lower.hex(), upper.hex(), metric.confidence.hex())
            )
            if abs(median) >= health_policy.extreme_abs_normalized_median:
                extreme_count += 1
            if lower == 0.0 and upper == 0.0:
                zero_width_count += 1
            if metric.confidence == 1.0:
                unit_confidence_count += 1
            max_abs_median = max(max_abs_median, abs(median))

        extreme_share = extreme_count / row_count
        largest_bucket_share = max(median_buckets.values()) / row_count
        zero_width_share = zero_width_count / row_count
        unit_confidence_share = unit_confidence_count / row_count
        passes = not threshold_applied or (
            extreme_share < health_policy.collapse_share_threshold
            and largest_bucket_share < health_policy.collapse_share_threshold
            and zero_width_share < health_policy.collapse_share_threshold
            and unit_confidence_share < health_policy.collapse_share_threshold
            and len(cell_signatures)
            >= health_policy.minimum_distinct_signatures
        )
        metric_assessments.append(
            ActionForecastMetricHealth(
                metric_id=scale.metric_id,
                row_count=row_count,
                extreme_median_share=float(extreme_share),
                largest_median_bucket_share=float(largest_bucket_share),
                zero_width_share=float(zero_width_share),
                unit_confidence_share=float(unit_confidence_share),
                distinct_cell_signature_count=len(cell_signatures),
                distinct_confidence_count=len(confidence_buckets),
                max_abs_normalized_median=float(max_abs_median),
                threshold_applied=threshold_applied,
                passes=passes,
            )
        )
        if len(confidence_buckets) == 1:
            constant_confidence_metric_ids.append(scale.metric_id)

    rows_pass = not threshold_applied or (
        len(row_signatures) >= health_policy.minimum_distinct_signatures
    )
    assessments = tuple(metric_assessments)
    return _NormalizedActionForecastHealth(
        metric_assessments=assessments,
        distinct_row_signature_count=len(row_signatures),
        distinct_probability_valid_count=len(probability_buckets),
        constant_confidence_metric_ids=tuple(
            sorted(constant_confidence_metric_ids)
        ),
        threshold_applied=threshold_applied,
        passes=rows_pass and all(value.passes for value in assessments),
    )


def _bind_health_assessment(
    *,
    member_id: str,
    frame_kind: ActionForecastHealthFrameKind,
    frame_receipt_sha256: str,
    request_sha256: str,
    layout_sha256: str | None,
    block_request_sha256: str | None,
    block_spec_sha256: str | None,
    block_index: int | None,
    global_row_start: int,
    global_row_stop: int,
    subset_binding: ActionForecastBlockHealthSubsetBinding | None,
    health_policy: ActionForecastHealthPolicyBinding,
    normalized: _NormalizedActionForecastHealth,
) -> ResolvedActionForecastHealthAssessment:
    return ResolvedActionForecastHealthAssessment(
        member_id=member_id,
        frame_kind=frame_kind,
        frame_receipt_sha256=frame_receipt_sha256,
        request_sha256=request_sha256,
        layout_sha256=layout_sha256,
        block_request_sha256=block_request_sha256,
        block_spec_sha256=block_spec_sha256,
        block_index=block_index,
        global_row_start=global_row_start,
        global_row_stop=global_row_stop,
        subset_binding=subset_binding,
        health_policy=health_policy,
        metric_assessments=normalized.metric_assessments,
        distinct_row_signature_count=normalized.distinct_row_signature_count,
        distinct_probability_valid_count=(
            normalized.distinct_probability_valid_count
        ),
        constant_confidence_metric_ids=(
            normalized.constant_confidence_metric_ids
        ),
        threshold_applied=normalized.threshold_applied,
        passes=normalized.passes,
    )


def assess_resolved_action_forecast_health(
    request: ActionForecastRequest,
    batch: ResolvedActionForecastBatch,
    *,
    member_id: str,
    health_policy: ActionForecastHealthPolicyBinding,
) -> ResolvedActionForecastHealthAssessment:
    """Assess one validated complete batch in metric-scale units."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    validate_resolved_action_forecasts(request, batch)
    _validate_health_inputs(member_id=member_id, health_policy=health_policy)
    normalized = _assess_normalized_forecasts(
        batch.forecasts,
        request,
        health_policy,
    )
    return _bind_health_assessment(
        member_id=member_id,
        frame_kind=ActionForecastHealthFrameKind.COMPLETE,
        frame_receipt_sha256=batch.receipt_sha256,
        request_sha256=request.request_sha256,
        layout_sha256=None,
        block_request_sha256=None,
        block_spec_sha256=None,
        block_index=None,
        global_row_start=0,
        global_row_stop=len(batch.forecasts),
        subset_binding=None,
        health_policy=health_policy,
        normalized=normalized,
    )


def assess_resolved_action_forecast_block_health(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    *,
    member_id: str,
    health_policy: ActionForecastHealthPolicyBinding,
) -> ResolvedActionForecastHealthAssessment:
    """Assess one validated partition block without presenting it as a batch."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be an exact ActionForecastBlockRequest")
    block_request.__post_init__()
    validate_resolved_action_forecast_block(block_request, block)
    _validate_health_inputs(member_id=member_id, health_policy=health_policy)
    normalized = _assess_normalized_forecasts(
        block.forecasts,
        block_request.request,
        health_policy,
    )
    spec = block_request.block
    return _bind_health_assessment(
        member_id=member_id,
        frame_kind=ActionForecastHealthFrameKind.PARTITION_BLOCK,
        frame_receipt_sha256=block.receipt_sha256,
        request_sha256=block_request.request.request_sha256,
        layout_sha256=block_request.layout.layout_sha256,
        block_request_sha256=block_request.block_request_sha256,
        block_spec_sha256=spec.block_spec_sha256,
        block_index=spec.block_index,
        global_row_start=spec.global_row_start,
        global_row_stop=spec.global_row_stop,
        subset_binding=None,
        health_policy=health_policy,
        normalized=normalized,
    )


def assess_resolved_action_forecast_block_subset_health(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    *,
    member_id: str,
    health_policy: ActionForecastHealthPolicyBinding,
    subset_policy: ActionForecastHealthSubsetPolicyBinding,
    included_global_row_indices: tuple[int, ...],
) -> ResolvedActionForecastHealthAssessment:
    """Assess one authenticated ordered subset of a validated partition block."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be an exact ActionForecastBlockRequest")
    block_request.__post_init__()
    validate_resolved_action_forecast_block(block_request, block)
    _validate_health_inputs(member_id=member_id, health_policy=health_policy)
    if type(subset_policy) is not ActionForecastHealthSubsetPolicyBinding:
        raise TypeError("subset_policy must be an exact subset-policy binding")
    subset_policy.__post_init__()
    if type(included_global_row_indices) is not tuple or not (
        included_global_row_indices
    ) or any(type(value) is not int for value in included_global_row_indices):
        raise ValueError(
            "included_global_row_indices must be a non-empty exact tuple"
        )
    if included_global_row_indices != tuple(
        sorted(set(included_global_row_indices))
    ):
        raise ValueError("included global rows must be unique and in block order")
    spec = block_request.block
    if any(
        value < spec.global_row_start or value >= spec.global_row_stop
        for value in included_global_row_indices
    ):
        raise ValueError("an included global row is outside the parent block")

    included_option_identities = tuple(
        block_request.request.finite_variation_contract.options[
            global_index
        ].identity_sha256
        for global_index in included_global_row_indices
    )
    subset_binding = ActionForecastBlockHealthSubsetBinding(
        subset_policy=subset_policy,
        request_sha256=block_request.request.request_sha256,
        layout_sha256=block_request.layout.layout_sha256,
        block_request_sha256=block_request.block_request_sha256,
        block_spec_sha256=spec.block_spec_sha256,
        parent_block_receipt_sha256=block.receipt_sha256,
        block_index=spec.block_index,
        global_row_start=spec.global_row_start,
        global_row_stop=spec.global_row_stop,
        included_global_row_indices=included_global_row_indices,
        included_option_identity_sha256s=included_option_identities,
    )
    selected_forecasts = tuple(
        block.forecasts[global_index - spec.global_row_start]
        for global_index in included_global_row_indices
    )
    normalized = _assess_normalized_forecasts(
        selected_forecasts,
        block_request.request,
        health_policy,
    )
    return _bind_health_assessment(
        member_id=member_id,
        frame_kind=ActionForecastHealthFrameKind.PARTITION_BLOCK_SUBSET,
        frame_receipt_sha256=subset_binding.binding_sha256,
        request_sha256=block_request.request.request_sha256,
        layout_sha256=block_request.layout.layout_sha256,
        block_request_sha256=block_request.block_request_sha256,
        block_spec_sha256=spec.block_spec_sha256,
        block_index=spec.block_index,
        global_row_start=spec.global_row_start,
        global_row_stop=spec.global_row_stop,
        subset_binding=subset_binding,
        health_policy=health_policy,
        normalized=normalized,
    )


__all__ = [
    "ActionForecastBlockFailure",
    "ActionForecastBlockHealthSubsetBinding",
    "ActionForecastHealthFrameKind",
    "ActionForecastHealthPolicyBinding",
    "ActionForecastHealthSubsetPolicyBinding",
    "ActionForecastMetricHealth",
    "ActionForecastWaveError",
    "ConcurrentActionForecastWave",
    "LENIENT_ACTION_FORECAST_HEALTH_POLICY_DEFINITION_SHA256",
    "LENIENT_ACTION_FORECAST_HEALTH_POLICY_ID",
    "LENIENT_ACTION_FORECAST_HEALTH_POLICY_VERSION",
    "LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256",
    "LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID",
    "LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION",
    "PartitionedActionForecastPolicy",
    "PartitionedActionForecastResult",
    "ResolvedActionForecastHealthAssessment",
    "action_forecast_block_call_id",
    "assess_resolved_action_forecast_block_health",
    "assess_resolved_action_forecast_block_subset_health",
    "assess_resolved_action_forecast_health",
    "assemble_partitioned_action_forecasts",
    "build_action_forecast_block_requests",
    "build_action_forecast_partition_layout",
    "lenient_action_forecast_health_policy",
    "lenient_action_forecast_health_v2_policy",
]
