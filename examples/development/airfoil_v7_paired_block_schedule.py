"""Outcome-free Airfoil-v7 adapter for the generic paired block schedule.

This module materializes only the frozen parent, finite variation catalog,
partition geometry, and outcome-blind G1 sample.  It never opens an oracle
manifest, terminal record, provider credential, forecast, allocation, or
evaluation outcome.  Airfoil-specific candidate/budget facts are projected
after the generic application service has selected a block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

from agent_evolve.application.paired_block_schedule import (
    PAIRED_BLOCK_SCHEDULE_POLICY_ID,
    PAIRED_BLOCK_SCHEDULE_POLICY_VERSION,
    PAIRED_BLOCK_SCHEDULE_RANK_DOMAIN,
    paired_block_schedule_policy,
    rank_paired_benchmark_blocks,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.policies.selection.diagnostic_sampling import (
    DiagnosticActionSample,
    HashStratifiedDiagnosticSampler,
    validate_diagnostic_action_sample,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockSpec,
    ActionForecastPartitionLayout,
    ActionForecastPartitionPolicyBinding,
)
from agent_evolve.ports.paired_block_schedule import (
    CanonicalBlockIdentity,
    ExactEligibleRowMask,
    PairedBlockScheduleRequest,
    PairedBlockScheduleResult,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from examples.benchmarks.engibench_airfoil.v7_contract import TASK_SHA256
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    materialize_held_out_parent,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7UnionVariationCatalog,
)
from examples.development.airfoil_v7_two_stage_agent_evolution import (
    G1_SAMPLE_DESIGN_KEY,
    G1_SAMPLE_SEED,
    G1_SAMPLE_SIZE,
    G2_PORTFOLIO_SIZE,
    OBJECTIVE_METRIC_ID,
    VIOLATION_METRIC_ID,
)


BENCHMARK_ID = "engibench_airfoil_v7"
PARTITION_POLICY_ID = "fixed_contiguous_action_forecast_partition"
PARTITION_POLICY_VERSION = 1
PARTITION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:fixed-contiguous-action-forecast-partition:v1:"
    b"maximal-complete-contiguous-rows-under-row-and-metric-cell-bounds"
).hexdigest()
BLOCK_ROWS = 20
BLOCK_METRIC_CELLS = 40
SCIENTIFIC_ARM_COUNT = 3
ASSIGNMENT_DOMAIN = "agent-evolve:paired-benchmark-block-assignment:v2"
_ASSIGNMENT_FRAMING = ASSIGNMENT_DOMAIN.encode("ascii") + b"\x00"


def _canonical_bytes(value: object) -> bytes:
    import json

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _contract_without_outcomes() -> FiniteVariationContract:
    parent = materialize_held_out_parent()
    frozen_parent = freeze_json(parent.candidate)
    if type(frozen_parent) is not FrozenJsonObject:
        raise AssertionError("held-out parent must freeze to an object")
    return bind_finite_variation_catalog(
        AirfoilV7UnionVariationCatalog(),
        frozen_parent,
    )


def _partition_layout_without_outcomes(
    contract: FiniteVariationContract,
) -> ActionForecastPartitionLayout:
    policy = ActionForecastPartitionPolicyBinding(
        policy_id=PARTITION_POLICY_ID,
        policy_version=PARTITION_POLICY_VERSION,
        policy_definition_sha256=PARTITION_POLICY_DEFINITION_SHA256,
        max_rows_per_block=BLOCK_ROWS,
        max_metric_cells_per_block=BLOCK_METRIC_CELLS,
    )
    option_identity_sha256s = tuple(
        option.identity_sha256 for option in contract.options
    )
    blocks = tuple(
        ActionForecastBlockSpec(
            block_index=block_index,
            global_row_start=start,
            global_row_stop=min(start + BLOCK_ROWS, len(contract.options)),
            option_identity_sha256s=option_identity_sha256s[
                start : min(start + BLOCK_ROWS, len(contract.options))
            ],
        )
        for block_index, start in enumerate(
            range(0, len(contract.options), BLOCK_ROWS)
        )
    )
    return ActionForecastPartitionLayout(
        finite_contract_identity_sha256=contract.identity_sha256,
        option_identity_sha256s=option_identity_sha256s,
        metric_ids=tuple(sorted((OBJECTIVE_METRIC_ID, VIOLATION_METRIC_ID))),
        partition_policy=policy,
        blocks=blocks,
    )


@dataclass(frozen=True, slots=True, eq=False)
class AirfoilV7PairedBlockAssignment:
    """Exact structural schedule plus Airfoil candidate/budget projection."""

    contract: FiniteVariationContract = field(repr=False, compare=False)
    g1_sample: DiagnosticActionSample
    layout: ActionForecastPartitionLayout
    schedule: PairedBlockScheduleResult

    def __post_init__(self) -> None:
        if type(self.contract) is not FiniteVariationContract:
            raise TypeError("contract must be an exact FiniteVariationContract")
        if type(self.g1_sample) is not DiagnosticActionSample:
            raise TypeError("g1_sample must be an exact DiagnosticActionSample")
        validate_diagnostic_action_sample(self.contract, self.g1_sample)
        if type(self.layout) is not ActionForecastPartitionLayout:
            raise TypeError("layout must be an exact ActionForecastPartitionLayout")
        self.layout.__post_init__()
        if type(self.schedule) is not PairedBlockScheduleResult:
            raise TypeError("schedule must be an exact PairedBlockScheduleResult")
        self.schedule.__post_init__()
        request = self.schedule.request
        if (
            request.benchmark_id != BENCHMARK_ID
            or request.task_sha256 != TASK_SHA256
            or request.finite_contract_identity_sha256
            != self.contract.identity_sha256
            or request.partition_layout_sha256 != self.layout.layout_sha256
        ):
            raise ValueError("generic schedule differs from the Airfoil structure")
        sampled = {member.option_id for member in self.g1_sample.members}
        expected_rows = tuple(
            index
            for index, option in enumerate(self.contract.options)
            if option.option_id not in sampled
        )
        if request.eligible_mask.eligible_global_row_indices != expected_rows:
            raise ValueError("generic schedule does not carry the exact non-G1 mask")

    @property
    def selected_block(self) -> CanonicalBlockIdentity:
        return self.schedule.selected_ranked_block.block

    @property
    def eligible_global_row_indices(self) -> tuple[int, ...]:
        selected = self.selected_block
        eligible = set(
            self.schedule.request.eligible_mask.eligible_global_row_indices
        )
        return tuple(
            index
            for index in range(selected.global_row_start, selected.global_row_stop)
            if index in eligible
        )

    @property
    def excluded_g1_global_row_indices(self) -> tuple[int, ...]:
        selected = self.selected_block
        eligible = set(self.eligible_global_row_indices)
        return tuple(
            index
            for index in range(selected.global_row_start, selected.global_row_stop)
            if index not in eligible
        )

    def payload_record(self) -> dict[str, object]:
        self.__post_init__()
        request = self.schedule.request
        selected = self.selected_block
        eligible_rows = self.eligible_global_row_indices
        excluded_rows = self.excluded_g1_global_row_indices
        eligible_count = len(eligible_rows)
        portfolio_size = G2_PORTFOLIO_SIZE
        if eligible_count < portfolio_size:
            raise ValueError("selected block cannot support the G2 portfolio size")
        scores_per_arm = sum(
            eligible_count - offset for offset in range(portfolio_size)
        )
        return {
            "schema_version": 2,
            "policy_id": PAIRED_BLOCK_SCHEDULE_POLICY_ID,
            "policy_version": PAIRED_BLOCK_SCHEDULE_POLICY_VERSION,
            "schedule_rank_domain": PAIRED_BLOCK_SCHEDULE_RANK_DOMAIN,
            "schedule_basis": request.schedule_basis_record(),
            "eligible_mask": request.eligible_mask.to_record(),
            "block_rank_digests": [
                {
                    "block_index": value.block.block_index,
                    "rank_digest_sha256": value.rank_digest_sha256,
                }
                for value in self.schedule.ranked_blocks
            ],
            "block_schedule": list(self.schedule.block_schedule),
            "replicate_index": request.replicate_index,
            "schedule_position": self.schedule.selected_ranked_block.schedule_position,
            "selected_block": {
                "block_index": selected.block_index,
                "block_spec_sha256": selected.block_spec_sha256,
                "global_row_start": selected.global_row_start,
                "global_row_stop": selected.global_row_stop,
            },
            "common_g2_subset": {
                "count": eligible_count,
                "global_row_indices": list(eligible_rows),
                "option_ids": [
                    self.contract.options[index].option_id for index in eligible_rows
                ],
                "excluded_g1_global_row_indices": list(excluded_rows),
                "excluded_g1_option_ids": [
                    self.contract.options[index].option_id for index in excluded_rows
                ],
            },
            "derived_budget": {
                "portfolio_size": portfolio_size,
                "candidate_scores_per_arm": scores_per_arm,
                "candidate_scores_total_mpn": scores_per_arm * SCIENTIFIC_ARM_COUNT,
                "logical_evaluator_slots_per_arm": portfolio_size,
                "logical_evaluator_slots_total": (
                    portfolio_size * SCIENTIFIC_ARM_COUNT
                ),
                "unique_cached_reads_min": portfolio_size,
                "unique_cached_reads_max": (
                    portfolio_size * SCIENTIFIC_ARM_COUNT
                ),
                "postdecision_exact_three_set_count": math.comb(
                    eligible_count,
                    portfolio_size,
                ),
            },
            "selection_seed_excludes_method_identities": True,
        }

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.payload_record())).hexdigest()

    @property
    def assignment_receipt_sha256(self) -> str:
        return hashlib.sha256(
            _ASSIGNMENT_FRAMING + _canonical_bytes(self.payload_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "assignment_domain": ASSIGNMENT_DOMAIN,
            "payload": self.payload_record(),
            "payload_sha256": self.payload_sha256,
            "assignment_receipt_sha256": self.assignment_receipt_sha256,
            "generic_schedule": self.schedule.to_record(),
            "generic_full_schedule_sha256": self.schedule.full_schedule_sha256,
            "generic_selected_block_receipt_sha256": (
                self.schedule.selected_block_receipt_sha256
            ),
            "outcomes_read": False,
            "provider_calls": 0,
            "credentials_read": False,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AirfoilV7PairedBlockAssignment
            and self.assignment_receipt_sha256
            == other.assignment_receipt_sha256
        )

    __hash__ = None


def prepare_airfoil_v7_paired_block_assignment(
    *,
    replicate_index: int = 0,
) -> AirfoilV7PairedBlockAssignment:
    """Prepare one exact assignment using structural, outcome-free inputs only."""

    contract = _contract_without_outcomes()
    sample = HashStratifiedDiagnosticSampler(
        seed=G1_SAMPLE_SEED,
        design_key=G1_SAMPLE_DESIGN_KEY,
    ).sample(contract, sample_size=G1_SAMPLE_SIZE)
    validate_diagnostic_action_sample(contract, sample)
    sampled = {member.option_id for member in sample.members}
    eligible_rows = tuple(
        index
        for index, option in enumerate(contract.options)
        if option.option_id not in sampled
    )
    layout = _partition_layout_without_outcomes(contract)
    request = PairedBlockScheduleRequest(
        benchmark_id=BENCHMARK_ID,
        task_sha256=TASK_SHA256,
        finite_contract_identity_sha256=contract.identity_sha256,
        partition_layout_sha256=layout.layout_sha256,
        policy=paired_block_schedule_policy(),
        eligible_mask=ExactEligibleRowMask(
            finite_contract_identity_sha256=contract.identity_sha256,
            eligible_global_row_indices=eligible_rows,
        ),
        public_seed=G1_SAMPLE_SEED,
        public_seed_source_sha256=sample.receipt_sha256,
        blocks=tuple(
            CanonicalBlockIdentity(
                block_index=block.block_index,
                block_spec_sha256=block.block_spec_sha256,
                global_row_start=block.global_row_start,
                global_row_stop=block.global_row_stop,
            )
            for block in layout.blocks
        ),
        replicate_index=replicate_index,
    )
    assignment = AirfoilV7PairedBlockAssignment(
        contract=contract,
        g1_sample=sample,
        layout=layout,
        schedule=rank_paired_benchmark_blocks(request),
    )
    assignment.__post_init__()
    return assignment


__all__ = [
    "ASSIGNMENT_DOMAIN",
    "AirfoilV7PairedBlockAssignment",
    "BENCHMARK_ID",
    "prepare_airfoil_v7_paired_block_assignment",
]
