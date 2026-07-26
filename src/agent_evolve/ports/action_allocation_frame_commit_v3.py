"""Exact treatment executions for durable operational allocator-v3 commits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation_frame_commit import (
    validate_treatment_occurrence_frame_request,
)
from agent_evolve.ports.action_allocation_frame_v3 import (
    OperationalFrameActionAllocationRequest,
    OperationalFrameActionAllocationResult,
    validate_operational_frame_action_allocation_result,
)
from agent_evolve.ports.treatment_assignment import (
    ProspectiveTreatmentAssignmentReceipt,
    TreatmentOccurrence,
)


_COMMIT_INPUT_DOMAIN = b"agent-evolve:operational-frame-allocation-commit-input:v1\x00"
_EXECUTION_DOMAIN = b"agent-evolve:operational-frame-allocation-execution:v1\x00"
_PAIRED_STRUCTURE_DOMAIN = b"agent-evolve:allocation-v3-paired-structure:v1\x00"
_ELIGIBLE_STRUCTURE_DOMAIN = b"agent-evolve:allocation-v3-eligible-structure:v1\x00"
_TERMINAL_LEDGER_DURABILITY_SCOPE = (
    "external_materialization_and_fsync_required_before_commit"
)


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


def paired_comparison_structure_record(
    request: OperationalFrameActionAllocationRequest,
) -> dict[str, object]:
    """Return mechanically comparable task/protocol facts for one v3 arm.

    This record binds provider-neutral forecast-policy identity and the common
    inference protocol, but deliberately excludes forecast values and source
    receipts.  Those treatment-specific values remain authenticated by each
    exact frame and by the upstream durable allocation commit.
    """

    if type(request) is not OperationalFrameActionAllocationRequest:
        raise TypeError("request must be an exact operational frame request")
    request.__post_init__()
    base = request.allocation
    frame = base.frame
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
            "option_identity_sha256": forecast_by_id[
                option_id
            ].option_identity_sha256,
            "child_configuration_sha256": forecast_by_id[
                option_id
            ].child_configuration_sha256,
            "family": forecast_by_id[option_id].family,
        }
        for option_id in base.eligible_option_ids
    ]
    block = None if frame.block_request is None else frame.block_request.block
    record = {
        "schema_version": 1,
        "utility_policy": base.utility.to_record(),
        "portfolio_size": base.portfolio_size,
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
        "eligible_options_structure_sha256": _hash(
            _ELIGIBLE_STRUCTURE_DOMAIN,
            eligible,
        ),
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
    return record


def paired_comparison_structure_sha256(
    request: OperationalFrameActionAllocationRequest,
) -> str:
    return _hash(_PAIRED_STRUCTURE_DOMAIN, paired_comparison_structure_record(request))


def _validate_common_configuration(
    requests: tuple[OperationalFrameActionAllocationRequest, ...],
) -> tuple[str, str]:
    first = requests[0]
    expected_utility_callable = first.allocation.utility.utility
    if any(
        value.allocation.utility.utility is not expected_utility_callable
        for value in requests
    ):
        raise ValueError(
            "all treatment arms must share the same in-process utility executable"
        )
    expected_configuration = first.allocator_configuration_sha256
    if any(
        value.allocator_configuration_sha256 != expected_configuration
        for value in requests
    ):
        raise ValueError(
            "all treatment arms must share one allocator-v3 configuration, "
            "resolution, public seed, mode, and allocation-unit key"
        )
    expected_structure = paired_comparison_structure_sha256(first)
    if any(
        paired_comparison_structure_sha256(value) != expected_structure
        for value in requests
    ):
        raise ValueError(
            "all treatment arms must share utility, portfolio budget, eligible "
            "actions, finite contract, structural frame, and metric semantics"
        )
    return expected_configuration, expected_structure


@dataclass(frozen=True, slots=True, eq=False)
class OperationalFrameActionAllocationCommitInput:
    """Complete common-key v3 allocation inputs for one treatment wave.

    Utility policy metadata is the durable, cross-process source commitment.
    Construction additionally requires object identity for the injected utility
    callable across this in-process wave.  Python object identity is deliberately
    not serialized because it is neither stable nor replayable across processes.
    """

    upstream_input_sha256: str
    terminal_provider_ledger_commitment_sha256: str
    treatment_assignment: ProspectiveTreatmentAssignmentReceipt = field(
        repr=False,
        compare=False,
    )
    allocation_requests: tuple[OperationalFrameActionAllocationRequest, ...] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        require_sha256(self.upstream_input_sha256, "upstream_input_sha256")
        require_sha256(
            self.terminal_provider_ledger_commitment_sha256,
            "terminal_provider_ledger_commitment_sha256",
        )
        if type(self.treatment_assignment) is not ProspectiveTreatmentAssignmentReceipt:
            raise TypeError(
                "treatment_assignment must be an exact prospective receipt"
            )
        self.treatment_assignment.__post_init__()
        if type(self.allocation_requests) is not tuple or not (
            self.allocation_requests
        ) or any(
            type(value) is not OperationalFrameActionAllocationRequest
            for value in self.allocation_requests
        ):
            raise ValueError(
                "allocation_requests must be a non-empty exact v3 request tuple"
            )
        if len(self.allocation_requests) != len(
            self.treatment_assignment.occurrence_input_order
        ):
            raise ValueError("v3 requests must cover every treatment occurrence")
        for occurrence, request in zip(
            self.treatment_assignment.occurrence_input_order,
            self.allocation_requests,
            strict=True,
        ):
            request.__post_init__()
            validate_treatment_occurrence_frame_request(
                occurrence,
                request.allocation,
            )
        _validate_common_configuration(self.allocation_requests)

    @property
    def common_allocator_configuration_sha256(self) -> str:
        self.__post_init__()
        return self.allocation_requests[0].allocator_configuration_sha256

    @property
    def common_paired_comparison_structure_sha256(self) -> str:
        self.__post_init__()
        return paired_comparison_structure_sha256(self.allocation_requests[0])

    @property
    def terminal_provider_ledger_commitment_record(self) -> dict[str, object]:
        return {
            "commitment_sha256": (
                self.terminal_provider_ledger_commitment_sha256
            ),
            "durability_scope": _TERMINAL_LEDGER_DURABILITY_SCOPE,
            "digest_alone_proves_durability": False,
        }

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "upstream_input_sha256": self.upstream_input_sha256,
            "terminal_provider_ledger_commitment": (
                self.terminal_provider_ledger_commitment_record
            ),
            "preregistration_chronology_boundary": {
                "seed_provenance_digest_alone_proves_pre_forecast_timing": False,
                "external_durable_release_required_for_chronology_claim": True,
                "external_release_must_be_bound_by_upstream_input": True,
                "upstream_input_sha256": self.upstream_input_sha256,
            },
            "treatment_assignment": self.treatment_assignment.to_record(),
            "treatment_occurrence_order": [
                value.to_record()
                for value in self.treatment_assignment.occurrence_input_order
            ],
            "common_allocator_configuration_sha256": (
                self.common_allocator_configuration_sha256
            ),
            "utility_executable_boundary": {
                "same_object_identity_required_in_process": True,
                "object_identity_serialized": False,
                "durable_source_commitment": (
                    self.allocation_requests[0].allocation.utility.to_record()
                ),
            },
            "common_score_resolution": (
                self.allocation_requests[0].score_resolution.to_record()
            ),
            "common_tie_selection": (
                self.allocation_requests[0].tie_selection.to_record()
            ),
            "common_paired_comparison_structure": paired_comparison_structure_record(
                self.allocation_requests[0]
            ),
            "common_paired_comparison_structure_sha256": (
                self.common_paired_comparison_structure_sha256
            ),
            "common_public_seed_uint64": (
                self.allocation_requests[0].tie_selection.public_seed
            ),
            "common_allocation_unit_key": (
                self.allocation_requests[0].tie_selection.allocation_unit_key
            ),
            "ordered_allocation_inputs": [
                {
                    "treatment_occurrence_id": occurrence.occurrence_id.value,
                    "operational_request_sha256": request.request_sha256,
                    "base_allocation_request_sha256": (
                        request.allocation.request_sha256
                    ),
                    "frame_receipt_sha256": request.allocation.frame.receipt_sha256,
                    "source_forecast_receipt_sha256": (
                        request.allocation.frame.source_forecast_receipt_sha256
                    ),
                    "eligible_options_sha256": (
                        request.allocation.eligible_options_sha256
                    ),
                }
                for occurrence, request in zip(
                    self.treatment_assignment.occurrence_input_order,
                    self.allocation_requests,
                    strict=True,
                )
            ],
        }

    @property
    def input_sha256(self) -> str:
        return _hash(_COMMIT_INPUT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "input_sha256": _hash(_COMMIT_INPUT_DOMAIN, record)}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperationalFrameActionAllocationCommitInput
            and self.input_sha256 == other.input_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class OperationalFrameActionAllocationTreatmentExecution:
    """One prospective treatment occurrence's exact allocator-v3 execution."""

    treatment_assignment: ProspectiveTreatmentAssignmentReceipt = field(
        repr=False,
        compare=False,
    )
    treatment_occurrence: TreatmentOccurrence
    request: OperationalFrameActionAllocationRequest = field(
        repr=False,
        compare=False,
    )
    result: OperationalFrameActionAllocationResult = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.treatment_assignment) is not ProspectiveTreatmentAssignmentReceipt:
            raise TypeError(
                "treatment_assignment must be an exact prospective receipt"
            )
        self.treatment_assignment.__post_init__()
        if type(self.treatment_occurrence) is not TreatmentOccurrence:
            raise TypeError("treatment_occurrence must be exact")
        self.treatment_occurrence.__post_init__()
        by_id = {
            value.occurrence_id: value
            for value in self.treatment_assignment.occurrence_input_order
        }
        expected = by_id.get(self.treatment_occurrence.occurrence_id)
        if expected is None or expected != self.treatment_occurrence:
            raise ValueError(
                "treatment occurrence differs from its prospective assignment"
            )
        if type(self.request) is not OperationalFrameActionAllocationRequest:
            raise TypeError("request must be an exact operational frame request")
        self.request.__post_init__()
        validate_treatment_occurrence_frame_request(
            self.treatment_occurrence,
            self.request.allocation,
        )
        if type(self.result) is not OperationalFrameActionAllocationResult:
            raise TypeError("result must be an exact operational frame result")
        validate_operational_frame_action_allocation_result(
            self.request,
            self.result,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "treatment_assignment_receipt_sha256": (
                self.treatment_assignment.receipt_sha256
            ),
            "treatment_occurrence": self.treatment_occurrence.to_record(),
            "frame": self.request.allocation.frame.to_record(),
            "base_allocation_request": self.request.allocation.to_record(),
            "operational_allocation_request": self.request.to_record(),
            "decision": self.result.decision.to_record(),
            "audit": self.result.audit.to_record(),
            "operational_result_receipt_sha256": self.result.receipt_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_EXECUTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "receipt_sha256": _hash(_EXECUTION_DOMAIN, record)}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperationalFrameActionAllocationTreatmentExecution
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


__all__ = [
    "OperationalFrameActionAllocationCommitInput",
    "OperationalFrameActionAllocationTreatmentExecution",
    "paired_comparison_structure_record",
    "paired_comparison_structure_sha256",
]
