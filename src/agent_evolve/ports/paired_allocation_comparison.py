"""Benchmark-neutral receipts for paired allocator-method comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation_frame import FrameActionAllocationRequest


_METHOD_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_STRUCTURE_DOMAIN = b"agent-evolve:frame-allocation-comparison-structure:v1\x00"
_ELIGIBLE_DOMAIN = b"agent-evolve:frame-allocation-comparison-eligible:v1\x00"
_METHOD_DOMAIN = b"agent-evolve:allocation-comparison-method:v1\x00"
_COMMITMENT_DOMAIN = b"agent-evolve:paired-allocation-comparison:v1\x00"


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


def frame_allocation_comparison_structure_record(
    request: FrameActionAllocationRequest,
) -> dict[str, object]:
    """Return treatment-neutral task, protocol, action, and budget structure.

    Forecast values and source receipts are intentionally excluded.  Exact
    method waves bind those separately; this projection compares whether the
    methods faced the same scientific problem.
    """

    if type(request) is not FrameActionAllocationRequest:
        raise TypeError("request must be an exact frame allocation request")
    request.__post_init__()
    frame = request.frame
    index_by_id = {
        forecast.option_id: index
        for index, forecast in zip(
            frame.global_row_indices,
            frame.forecasts,
            strict=True,
        )
    }
    forecast_by_id = {value.option_id: value for value in frame.forecasts}
    eligible = [
        {
            "global_row_index": index_by_id[option_id],
            "option_id": option_id,
            "option_identity_sha256": (
                forecast_by_id[option_id].option_identity_sha256
            ),
            "child_configuration_sha256": (
                forecast_by_id[option_id].child_configuration_sha256
            ),
            "family": forecast_by_id[option_id].family,
        }
        for option_id in request.eligible_option_ids
    ]
    block = None if frame.block_request is None else frame.block_request.block
    return {
        "schema_version": 1,
        "utility_policy": request.utility.to_record(),
        "portfolio_size": request.portfolio_size,
        "finite_contract_identity_sha256": (
            frame.request.finite_variation_contract.identity_sha256
        ),
        "optimization_semantics": frame.request.optimization_semantics.to_record(),
        "action_semantics": frame.request.action_semantics.to_record(),
        "parent_metric_values": [
            value.to_record() for value in frame.request.parent_metric_values
        ],
        "metric_scales": [
            value.to_record() for value in frame.request.metric_scales
        ],
        "forecast_generation_policy": {
            "policy_id": frame.policy_identity[0],
            "policy_version": frame.policy_identity[1],
            "policy_definition_sha256": frame.policy_identity[2],
        },
        "inference_protocol": {
            "operation": frame.request.operation,
            "max_output_tokens": frame.request.max_output_tokens,
            "temperature_hex": (
                None
                if frame.request.temperature is None
                else float(frame.request.temperature).hex()
            ),
        },
        "frame_kind": frame.frame_kind.value,
        "global_row_indices": list(frame.global_row_indices),
        "eligible_options": eligible,
        "eligible_options_structure_sha256": _hash(_ELIGIBLE_DOMAIN, eligible),
        "subset_policy": (
            None if frame.subset_policy is None else frame.subset_policy.to_record()
        ),
        "partition_structure": (
            None
            if block is None
            else {
                "layout_sha256": frame.block_request.layout.layout_sha256,
                "block_spec_sha256": block.block_spec_sha256,
                "block_index": block.block_index,
                "global_row_start": block.global_row_start,
                "global_row_stop": block.global_row_stop,
            }
        ),
    }


def frame_allocation_comparison_structure_sha256(
    request: FrameActionAllocationRequest,
) -> str:
    return _hash(
        _STRUCTURE_DOMAIN,
        frame_allocation_comparison_structure_record(request),
    )


class AllocationComparisonMethodKind(str, Enum):
    AUDITED_FRAME_V2 = "audited_frame_v2"
    OPERATIONAL_FRAME_V3 = "operational_frame_v3"


@dataclass(frozen=True, slots=True)
class AllocationComparisonSelectedOption:
    treatment_occurrence_id: str
    treatment_id: str
    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str

    def __post_init__(self) -> None:
        for name in ("treatment_occurrence_id", "treatment_id", "option_id", "family"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty exact string")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "treatment_occurrence_id": self.treatment_occurrence_id,
            "treatment_id": self.treatment_id,
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True, eq=False)
class AllocationComparisonMethodReceipt:
    comparison_method_id: str
    method_kind: AllocationComparisonMethodKind
    schedule_binding_sha256: str
    allocation_phase_commit_receipt_sha256: str
    allocation_input_sha256: str
    treatment_assignment_receipt_sha256: str
    upstream_input_sha256: str
    terminal_provider_ledger_commitment_sha256: str
    common_comparison_structure_sha256: str
    allocator_policy_id: str
    allocator_policy_version: int
    allocator_definition_sha256: str
    allocator_configuration_sha256: str
    ordered_frame_receipt_sha256s: tuple[str, ...]
    ordered_base_request_sha256s: tuple[str, ...]
    ordered_method_request_sha256s: tuple[str, ...]
    ordered_decision_receipt_sha256s: tuple[str, ...]
    selected_options: tuple[AllocationComparisonSelectedOption, ...]

    def __post_init__(self) -> None:
        if (
            type(self.comparison_method_id) is not str
            or _METHOD_ID.fullmatch(self.comparison_method_id) is None
        ):
            raise ValueError("comparison_method_id must use the method-ID grammar")
        if type(self.method_kind) is not AllocationComparisonMethodKind:
            raise TypeError("method_kind must be exact")
        for name in (
            "schedule_binding_sha256",
            "allocation_phase_commit_receipt_sha256",
            "allocation_input_sha256",
            "treatment_assignment_receipt_sha256",
            "upstream_input_sha256",
            "terminal_provider_ledger_commitment_sha256",
            "common_comparison_structure_sha256",
            "allocator_definition_sha256",
            "allocator_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.allocator_policy_id) is not str or not self.allocator_policy_id:
            raise ValueError("allocator_policy_id must be non-empty")
        if (
            type(self.allocator_policy_version) is not int
            or self.allocator_policy_version <= 0
        ):
            raise ValueError("allocator_policy_version must be positive")
        digest_tuples = (
            self.ordered_frame_receipt_sha256s,
            self.ordered_base_request_sha256s,
            self.ordered_method_request_sha256s,
            self.ordered_decision_receipt_sha256s,
        )
        if any(type(values) is not tuple or not values for values in digest_tuples):
            raise ValueError("ordered method bindings must be non-empty tuples")
        if len({len(values) for values in digest_tuples}) != 1:
            raise ValueError("ordered method bindings must have equal lengths")
        for values in digest_tuples:
            for value in values:
                require_sha256(value, "ordered method binding")
        if type(self.selected_options) is not tuple or not self.selected_options:
            raise ValueError("selected_options must be a non-empty exact tuple")
        for value in self.selected_options:
            if type(value) is not AllocationComparisonSelectedOption:
                raise TypeError("selected_options must contain exact rows")
            value.__post_init__()

    @property
    def logical_slot_count(self) -> int:
        return len(self.selected_options)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "comparison_method_id": self.comparison_method_id,
            "method_kind": self.method_kind.value,
            "schedule_binding_sha256": self.schedule_binding_sha256,
            "allocation_phase_commit_receipt_sha256": (
                self.allocation_phase_commit_receipt_sha256
            ),
            "allocation_input_sha256": self.allocation_input_sha256,
            "treatment_assignment_receipt_sha256": (
                self.treatment_assignment_receipt_sha256
            ),
            "upstream_input_sha256": self.upstream_input_sha256,
            "terminal_provider_ledger_commitment_sha256": (
                self.terminal_provider_ledger_commitment_sha256
            ),
            "common_comparison_structure_sha256": (
                self.common_comparison_structure_sha256
            ),
            "allocator_policy": {
                "policy_id": self.allocator_policy_id,
                "policy_version": self.allocator_policy_version,
                "definition_sha256": self.allocator_definition_sha256,
                "configuration_sha256": self.allocator_configuration_sha256,
            },
            "ordered_frame_receipt_sha256s": list(
                self.ordered_frame_receipt_sha256s
            ),
            "ordered_base_request_sha256s": list(
                self.ordered_base_request_sha256s
            ),
            "ordered_method_request_sha256s": list(
                self.ordered_method_request_sha256s
            ),
            "ordered_decision_receipt_sha256s": list(
                self.ordered_decision_receipt_sha256s
            ),
            "logical_slot_count": self.logical_slot_count,
            "selected_options": [value.to_record() for value in self.selected_options],
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_METHOD_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True, eq=False)
class PairedAllocationComparisonCommitment:
    """Exact two-method authority receipt created before outcome access."""

    methods: tuple[AllocationComparisonMethodReceipt, ...]

    def __post_init__(self) -> None:
        if type(self.methods) is not tuple or len(self.methods) != 2 or any(
            type(value) is not AllocationComparisonMethodReceipt
            for value in self.methods
        ):
            raise ValueError("paired comparison requires exactly two method receipts")
        for value in self.methods:
            value.__post_init__()
        if self.methods != tuple(
            sorted(self.methods, key=lambda value: value.comparison_method_id)
        ):
            raise ValueError("paired methods must use canonical method-ID order")
        if len({value.comparison_method_id for value in self.methods}) != 2:
            raise ValueError("paired comparison method IDs cannot repeat")
        if {value.method_kind for value in self.methods} != {
            AllocationComparisonMethodKind.AUDITED_FRAME_V2,
            AllocationComparisonMethodKind.OPERATIONAL_FRAME_V3,
        }:
            raise ValueError("paired comparison requires one v2 and one v3 method")
        common_fields = (
            "schedule_binding_sha256",
            "treatment_assignment_receipt_sha256",
            "upstream_input_sha256",
            "terminal_provider_ledger_commitment_sha256",
            "common_comparison_structure_sha256",
            "ordered_frame_receipt_sha256s",
            "ordered_base_request_sha256s",
        )
        for name in common_fields:
            if getattr(self.methods[0], name) != getattr(self.methods[1], name):
                raise ValueError(f"paired methods differ on common {name}")
        identities: dict[str, tuple[str, str, str]] = {}
        for method in self.methods:
            for selected in method.selected_options:
                identity = (
                    selected.option_identity_sha256,
                    selected.child_configuration_sha256,
                    selected.family,
                )
                prior = identities.setdefault(selected.option_id, identity)
                if prior != identity:
                    raise ValueError("selected option identity drifts across methods")

    @property
    def logical_slot_count(self) -> int:
        return sum(value.logical_slot_count for value in self.methods)

    @property
    def selected_option_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    selected.option_id
                    for method in self.methods
                    for selected in method.selected_options
                }
            )
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        first = self.methods[0]
        return {
            "schema_version": 1,
            "method_receipts": [value.to_record() for value in self.methods],
            "common_schedule_binding_sha256": first.schedule_binding_sha256,
            "common_treatment_assignment_receipt_sha256": (
                first.treatment_assignment_receipt_sha256
            ),
            "common_upstream_input_sha256": first.upstream_input_sha256,
            "common_terminal_provider_ledger_commitment_sha256": (
                first.terminal_provider_ledger_commitment_sha256
            ),
            "common_comparison_structure_sha256": (
                first.common_comparison_structure_sha256
            ),
            "logical_slot_count": self.logical_slot_count,
            "selected_option_ids": list(self.selected_option_ids),
            "outcomes_read_by_builder": False,
        }

    @property
    def commitment_sha256(self) -> str:
        return _hash(_COMMITMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "commitment_sha256": self.commitment_sha256,
        }


__all__ = [
    "AllocationComparisonMethodKind",
    "AllocationComparisonMethodReceipt",
    "AllocationComparisonSelectedOption",
    "PairedAllocationComparisonCommitment",
    "frame_allocation_comparison_structure_record",
    "frame_allocation_comparison_structure_sha256",
]
