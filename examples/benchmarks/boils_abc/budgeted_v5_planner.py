"""Outcome-blind two-generation policy for the BOiLS budgeted-v5 kill test.

This module is deliberately a planner, not an experiment runner.  It imports no
provider adapter, filesystem artifact, or local-oracle table.  Generation one
freezes six atomic interventions from the finite BOiLS catalog.  Generation two
uses only the candidates published by generation one and mechanically composes
every replay-safe disjoint pair before selecting exploit and coverage unions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from fractions import Fraction
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationPlan,
    MaterializedInvocation,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerSlot,
    OptimizerState,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
)
from agent_evolve.domain.typed_json import typed_json_sha256
from agent_evolve.domain.variation_space import AtomicEditOption
from agent_evolve.policies.reward.frozen_archive import (
    FrozenArchiveMarginalHypervolumeReward,
    FrozenArchiveRewardRecord,
    FrozenArchiveSnapshot2D,
)
from agent_evolve.policies.selection.disjoint_pairs import (
    DisjointBranchFacts,
    DisjointPairScoreRow,
    DisjointPairSelectionDecision,
    DisjointParentPairPolicy,
    ReplayVerifiedDisjointPair,
)
from agent_evolve.policies.selection.task_keyed_palette import (
    AtomicPaletteDecision,
    PathFamilyExposure,
    TaskKeyedPalettePolicy,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchMaterialization,
    DisjointPatchRecombinationError,
    DisjointPatchRecombiner,
)
from agent_evolve.ports.agentic_generator import AtomicMutationDraft
from agent_evolve.ports.id_factory import IdFactory
from examples.benchmarks.boils_abc import budgeted_v5_support as support
from examples.benchmarks.boils_abc.variation_catalog import BoilsAtomicVariationCatalog


POLICY_ID = "boils_abc_budgeted_v5_two_generation"
POLICY_VERSION = 2
PALETTE_SEED = 20_260_714
PALETTE_SIZE = 3
MAX_OPTIONS_PER_FAMILY = 1
_HASH_DOMAIN = b"boils-abc:budgeted-v5-planner:v2\x00"
DecisionSink = Callable[[Mapping[str, object]], None]
_G1_SLOT_ORDER = ("G1-A1", "G1-A2", "G1-D1", "G1-D2", "G1-U", "G1-X")
_G1_LABELS = {slot_id: slot_id for slot_id in _G1_SLOT_ORDER}
_SLOT_ROLES = {
    "G1-A1": support.AREA_ROLE.value,
    "G1-A2": support.AREA_ROLE.value,
    "G1-D1": support.DEPTH_ROLE.value,
    "G1-D2": support.DEPTH_ROLE.value,
    "G1-U": support.UNCERTAINTY_ROLE.value,
    "G1-X": support.COVERAGE_ROLE.value,
}


class BoilsV5PlanningError(RuntimeError):
    """The frozen BOiLS protocol cannot truthfully plan the next wave."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _record_hash(kind: str, record: object) -> str:
    if type(kind) is not str or not kind:
        raise ValueError("record kind must be non-empty")
    return hashlib.sha256(
        _HASH_DOMAIN + kind.encode("ascii") + b"\x00" + _canonical_json(record)
    ).hexdigest()


def _path_record(path: JsonPath) -> str:
    for expected, text in (
        (support.AREA_PATH, support.AREA_PATH_TEXT),
        (support.DEPTH_PATH, support.DEPTH_PATH_TEXT),
        (support.UNCERTAINTY_PATH, support.UNCERTAINTY_PATH_TEXT),
        (support.COVERAGE_PATH, support.COVERAGE_PATH_TEXT),
    ):
        if path == expected:
            return text
    return canonical_path_bytes(path).hex()


def _weakly_dominates(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
    objectives: tuple[ObjectiveSpec, ObjectiveSpec],
) -> bool:
    """Return whether ``left`` weakly dominates ``right`` exactly."""

    left_values = dict(left)
    right_values = dict(right)
    return all(
        (
            left_values[objective.name] <= right_values[objective.name]
            if objective.goal == "min"
            else left_values[objective.name] >= right_values[objective.name]
        )
        for objective in objectives
    )


@dataclass(frozen=True, slots=True)
class BoilsV5FrontAlignedRewardRecord:
    """Inspect one composite reward without consulting the live archive."""

    status: str
    reward: float
    base_hypervolume_record: FrozenArchiveRewardRecord
    strictly_extends_frozen_front: bool
    front_extension_raw_credit: float
    reward_definition_hash: str
    reward_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class BoilsV5FrozenFrontAlignedReward:
    """Frozen HV reward with one-unit credit for clipped front extensions.

    BOiLS objectives are integer counts. A candidate that strictly extends the
    pre-wave Pareto front but lies outside the fixed HV reference rectangle
    therefore receives one raw hypervolume quantum instead of zero. Dominated
    and duplicate points receive no such credit, and all candidates in a wave
    are still compared with the same immutable archive snapshot.
    """

    snapshot: FrozenArchiveSnapshot2D
    base_policy: FrozenArchiveMarginalHypervolumeReward = field(init=False)
    definition_hash: str = field(init=False)
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.snapshot) is not FrozenArchiveSnapshot2D:
            raise TypeError("snapshot must be an exact frozen 2D archive snapshot")
        base = FrozenArchiveMarginalHypervolumeReward(self.snapshot)
        definition = {
            "policy_id": support.FRONT_ALIGNED_REWARD_POLICY_ID,
            "policy_version": support.FRONT_ALIGNED_REWARD_POLICY_VERSION,
            "base_reward_definition_sha256": base.definition_hash,
            "composition": "base_hv_plus_clipped_strict_front_extension",
            "front_extension_rule": (
                "not weakly dominated by or equal to any frozen archive point"
            ),
            "front_extension_raw_credit_hex": (
                support.FRONT_EXTENSION_RAW_CREDIT.hex()
            ),
            "normalization": "reuse_frozen_hv_normalization",
        }
        definition_hash = _record_hash("front-aligned-reward-definition", definition)
        snapshot_hash = _record_hash(
            "front-aligned-reward-snapshot",
            {
                "definition_sha256": definition_hash,
                "base_reward_snapshot_sha256": self.snapshot.snapshot_hash,
            },
        )
        object.__setattr__(self, "base_policy", base)
        object.__setattr__(self, "definition_hash", definition_hash)
        object.__setattr__(self, "snapshot_hash", snapshot_hash)

    def record(self, child: EvolutionCandidate) -> BoilsV5FrontAlignedRewardRecord:
        if type(child) is not EvolutionCandidate:
            raise TypeError("child must be an exact EvolutionCandidate")
        base = self.base_policy.record(child)
        candidate_point = base.candidate_point
        extends = bool(
            candidate_point is not None
            and not any(
                _weakly_dominates(
                    archive_point,
                    candidate_point,
                    self.snapshot.objectives,
                )
                for archive_point in self.snapshot.archive_points
            )
        )
        raw_credit = (
            support.FRONT_EXTENSION_RAW_CREDIT
            if base.status == "credited"
            and base.marginal_hypervolume_gain == 0.0
            and extends
            else 0.0
        )
        reward = (
            base.reward + raw_credit / self.snapshot.normalization
            if base.status == "credited"
            else base.reward
        )
        return BoilsV5FrontAlignedRewardRecord(
            status=base.status,
            reward=float(reward),
            base_hypervolume_record=base,
            strictly_extends_frozen_front=extends,
            front_extension_raw_credit=raw_credit,
            reward_definition_hash=self.definition_hash,
            reward_snapshot_hash=self.snapshot_hash,
        )

    def __call__(
        self,
        child: EvolutionCandidate,
        parents: tuple[EvolutionCandidate, ...],
        objectives: Sequence[ObjectiveSpec],
    ) -> float:
        del parents
        if tuple(objectives) != self.snapshot.objectives:
            raise ValueError("reward objectives differ from the frozen snapshot")
        return self.record(child).reward

    def to_trace_record(self) -> dict[str, object]:
        return {
            "policy_id": support.FRONT_ALIGNED_REWARD_POLICY_ID,
            "policy_version": support.FRONT_ALIGNED_REWARD_POLICY_VERSION,
            "definition_sha256": self.definition_hash,
            "snapshot_sha256": self.snapshot_hash,
            "base_reward_definition_sha256": self.base_policy.definition_hash,
            "base_reward_snapshot_sha256": self.snapshot.snapshot_hash,
            "front_extension_raw_credit_hex": (
                support.FRONT_EXTENSION_RAW_CREDIT.hex()
            ),
        }


def _exposure_record(values: Sequence[PathFamilyExposure]) -> list[dict[str, object]]:
    return [value.to_trace_record() for value in values]


def _option_record(option: AtomicEditOption) -> dict[str, object]:
    return {
        "option_id": option.option_id,
        "option_identity_sha256": option.identity_sha256,
        "path": _path_record(option.path),
        "replacement": option.replacement,
        "family": option.family,
    }


@dataclass(frozen=True, slots=True)
class BoilsV5G1SlotDecision:
    """The outcome-blind facts frozen before one G1 proposal starts."""

    slot_id: str
    role: str
    proposal_authority: str
    palette: AtomicPaletteDecision
    exposures_before: tuple[PathFamilyExposure, ...]
    engine_option: AtomicEditOption | None = None

    def __post_init__(self) -> None:
        if self.slot_id not in _G1_SLOT_ORDER:
            raise ValueError("unknown G1 slot_id")
        if self.role != _SLOT_ROLES[self.slot_id]:
            raise ValueError("slot role differs from frozen BOiLS role")
        expected_authority = "engine" if self.slot_id == "G1-X" else "model"
        if self.proposal_authority != expected_authority:
            raise ValueError("slot proposal authority differs from protocol")
        if type(self.palette) is not AtomicPaletteDecision:
            raise TypeError("palette must be an exact AtomicPaletteDecision")
        self.palette.revalidate()
        if type(self.exposures_before) is not tuple or any(
            type(value) is not PathFamilyExposure for value in self.exposures_before
        ):
            raise TypeError("exposures_before must contain exact exposure values")
        if self.slot_id == "G1-X":
            if type(self.engine_option) is not AtomicEditOption:
                raise ValueError("engine coverage slot requires one exact option")
            if self.palette.palette != (self.engine_option,):
                raise ValueError("coverage option must be the singleton palette")
        elif self.engine_option is not None:
            raise ValueError("model slots cannot carry an engine option")

    def to_trace_record(self) -> dict[str, object]:
        body = {
            "slot_id": self.slot_id,
            "label": _G1_LABELS[self.slot_id],
            "role": self.role,
            "proposal_authority": self.proposal_authority,
            "palette_decision_sha256": self.palette.decision_sha256,
            "palette_option_ids": [item.option_id for item in self.palette.palette],
            "palette_options": [_option_record(item) for item in self.palette.palette],
            "exposures_before": _exposure_record(self.exposures_before),
            "engine_option_id": (
                None if self.engine_option is None else self.engine_option.option_id
            ),
        }
        return {**body, "slot_decision_sha256": _record_hash("g1-slot", body)}

    @property
    def decision_sha256(self) -> str:
        return str(self.to_trace_record()["slot_decision_sha256"])


@dataclass(frozen=True, slots=True)
class BoilsV5Generation1Decision:
    """Complete six-slot schedule plus sequential planned-exposure ledger."""

    initial_exposures: tuple[PathFamilyExposure, ...]
    slots: tuple[BoilsV5G1SlotDecision, ...]
    final_exposures: tuple[PathFamilyExposure, ...]
    support_manifest_sha256: str

    def __post_init__(self) -> None:
        if tuple(slot.slot_id for slot in self.slots) != _G1_SLOT_ORDER:
            raise ValueError("G1 decisions must use exact frozen slot order")
        if len(self.slots) != 6:
            raise ValueError("G1 requires six exact slots")
        if self.slots[0].palette != self.slots[1].palette:
            raise ValueError("area control pair must share one exact palette")
        if self.slots[2].palette != self.slots[3].palette:
            raise ValueError("depth control pair must share one exact palette")
        if self.slots[4].palette.chosen_path == self.slots[5].palette.chosen_path:
            raise ValueError("uncertainty and coverage must target distinct paths")
        uncertainty_required = tuple(
            option
            for option in self.slots[4].palette.palette
            if option.replacement == support.UNCERTAINTY_REQUIRED_ACTION
            and option.family == support.UNCERTAINTY_REQUIRED_FAMILY
            and option.path == support.UNCERTAINTY_PATH
        )
        if len(uncertainty_required) != 1 or self.slots[
            4
        ].palette.required_option_ids != (uncertainty_required[0].option_id,):
            raise ValueError(
                "uncertainty palette must bind its extended-family obligation"
            )
        if (
            type(self.support_manifest_sha256) is not str
            or len(self.support_manifest_sha256) != 64
        ):
            raise ValueError("support_manifest_sha256 must be a SHA-256 digest")

    def to_trace_record(self) -> dict[str, object]:
        body = {
            "event_type": "boils_v5_generation1_decided",
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "palette_seed": PALETTE_SEED,
            "palette_size": PALETTE_SIZE,
            "max_options_per_family": MAX_OPTIONS_PER_FAMILY,
            "support_manifest_sha256": self.support_manifest_sha256,
            "initial_exposures": _exposure_record(self.initial_exposures),
            "slots": [slot.to_trace_record() for slot in self.slots],
            "final_exposures": _exposure_record(self.final_exposures),
            "planned_exposure_law": (
                "append every provider-visible option once at each frozen slot; "
                "paired slots reuse their matched palette but remain separate exposures"
            ),
            "uncertainty_palette_obligation": {
                "obligation_id": support.UNCERTAINTY_COVERAGE_OBLIGATION_ID,
                "obligation_version": (support.UNCERTAINTY_COVERAGE_OBLIGATION_VERSION),
                "path": support.UNCERTAINTY_PATH_TEXT,
                "required_action": support.UNCERTAINTY_REQUIRED_ACTION,
                "required_family": support.UNCERTAINTY_REQUIRED_FAMILY,
                "required_option_id": self.slots[4].palette.required_option_ids[0],
                "rationale": support.UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE,
            },
            "post_hoc_development_protocol_correction": True,
            "protocol_correction": support.protocol_correction_record(),
        }
        return {**body, "decision_sha256": _record_hash("generation1", body)}

    @property
    def decision_sha256(self) -> str:
        return str(self.to_trace_record()["decision_sha256"])

    def slot(self, slot_id: str) -> BoilsV5G1SlotDecision:
        return next(item for item in self.slots if item.slot_id == slot_id)


@dataclass(frozen=True, slots=True)
class BoilsV5PairEnumerationRow:
    """One G1 pair attempted by the replay-safe materializer."""

    left_candidate_id: CandidateId
    right_candidate_id: CandidateId
    eligible: bool
    target_candidate_id: CandidateId
    target_configuration_sha256: str | None
    materialization_receipt_sha256: str | None
    rejection_type: str | None
    rejection_message_sha256: str | None

    def __post_init__(self) -> None:
        if self.left_candidate_id >= self.right_candidate_id:
            raise ValueError("enumeration pair IDs must use canonical order")
        if self.target_candidate_id in {
            self.left_candidate_id,
            self.right_candidate_id,
        }:
            raise ValueError("union target occurrence must be distinct")
        success_values = (
            self.target_configuration_sha256,
            self.materialization_receipt_sha256,
        )
        failure_values = (self.rejection_type, self.rejection_message_sha256)
        if self.eligible:
            if any(value is None or len(value) != 64 for value in success_values):
                raise ValueError("eligible row requires target and receipt hashes")
            if any(value is not None for value in failure_values):
                raise ValueError("eligible row cannot carry rejection evidence")
        else:
            if any(value is not None for value in success_values):
                raise ValueError("rejected row cannot carry materialization hashes")
            if any(value is None for value in failure_values):
                raise ValueError("rejected row requires typed failure evidence")

    @property
    def pair_ids(self) -> tuple[CandidateId, CandidateId]:
        return self.left_candidate_id, self.right_candidate_id

    def to_trace_record(self) -> dict[str, object]:
        return {
            "left_candidate_id": self.left_candidate_id.value,
            "right_candidate_id": self.right_candidate_id.value,
            "eligible": self.eligible,
            "target_candidate_id": self.target_candidate_id.value,
            "target_configuration_sha256": self.target_configuration_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "rejection_type": self.rejection_type,
            "rejection_message_sha256": self.rejection_message_sha256,
        }


@dataclass(frozen=True, slots=True)
class BoilsV5G1CheckpointRow:
    """One exact G1 slot state admitted to prospective G2 planning."""

    slot_id: str
    status: str
    candidate_id: CandidateId | None

    def __post_init__(self) -> None:
        if self.slot_id not in _G1_SLOT_ORDER:
            raise ValueError("checkpoint row has an unknown G1 slot")
        allowed = {
            "eligible",
            "missing_candidate",
            "invalid_candidate",
            "operator_noncompliant",
        }
        if self.status not in allowed:
            raise ValueError("checkpoint row has an unknown status")
        if (self.status == "missing_candidate") != (self.candidate_id is None):
            raise ValueError("only a missing-candidate row omits candidate_id")
        if self.candidate_id is not None:
            CandidateId.__post_init__(self.candidate_id)

    def to_trace_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "status": self.status,
            "candidate_id": (
                None if self.candidate_id is None else self.candidate_id.value
            ),
        }


def _pair_id_key(row: DisjointPairScoreRow) -> tuple[str, str]:
    left, right = row.pair_ids
    return left.value, right.value


def _incremental_coverage_metrics(
    row: DisjointPairScoreRow,
    exploit: DisjointPairScoreRow,
    branch_paths: Mapping[CandidateId, str],
) -> dict[str, int]:
    exploit_branches = (exploit.pair.left, exploit.pair.right)
    row_branches = (row.pair.left, row.pair.right)
    exploit_paths = {branch_paths[branch.candidate_id] for branch in exploit_branches}
    row_paths = {branch_paths[branch.candidate_id] for branch in row_branches}
    exploit_families = {branch.family for branch in exploit_branches}
    row_families = {branch.family for branch in row_branches}
    exploit_roles = {branch.role for branch in exploit_branches}
    row_roles = {branch.role for branch in row_branches}
    exploit_parents = set(exploit.pair_ids)
    return {
        "new_path_count": len(row_paths - exploit_paths),
        "new_family_count": len(row_families - exploit_families),
        "new_role_count": len(row_roles - exploit_roles),
        "new_parent_count": len(set(row.pair_ids) - exploit_parents),
    }


def _select_batch_rows(
    rows: tuple[DisjointPairScoreRow, ...],
    branch_paths: Mapping[CandidateId, str],
) -> tuple[DisjointPairScoreRow | None, DisjointPairScoreRow | None]:
    if not rows:
        return None, None
    exploit = min(
        rows,
        key=lambda row: (
            -row.branch_reward_sum,
            -row.distinct_role_count,
            *_pair_id_key(row),
        ),
    )
    coverage_pool = tuple(
        row
        for row in rows
        if row.pair.target_configuration_sha256
        != exploit.pair.target_configuration_sha256
    )
    if not coverage_pool:
        return exploit, None

    def coverage_key(row: DisjointPairScoreRow) -> tuple[object, ...]:
        metrics = _incremental_coverage_metrics(row, exploit, branch_paths)
        return (
            -metrics["new_path_count"],
            -metrics["new_family_count"],
            -metrics["new_role_count"],
            -metrics["new_parent_count"],
            -row.distinct_family_count,
            -row.distinct_role_count,
            row.path_family_exposure_sum,
            *_pair_id_key(row),
        )

    return exploit, min(coverage_pool, key=coverage_key)


@dataclass(frozen=True, slots=True)
class BoilsV5BatchPairSelectionDecision:
    """Select exploit, then maximize coverage incremental to that exploit."""

    eligible_rows: tuple[DisjointPairScoreRow, ...]
    branch_paths: tuple[tuple[CandidateId, str], ...]
    exploit_pair_ids: tuple[CandidateId, CandidateId] | None
    coverage_pair_ids: tuple[CandidateId, CandidateId] | None

    def __post_init__(self) -> None:
        if type(self.eligible_rows) is not tuple or any(
            type(row) is not DisjointPairScoreRow for row in self.eligible_rows
        ):
            raise TypeError("eligible_rows must contain exact score rows")
        for row in self.eligible_rows:
            row.revalidate()
        if tuple(sorted(self.eligible_rows, key=_pair_id_key)) != self.eligible_rows:
            raise ValueError("eligible_rows must use canonical candidate-ID order")
        if type(self.branch_paths) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not CandidateId
            or type(item[1]) is not str
            or not item[1]
            for item in self.branch_paths
        ):
            raise TypeError("branch_paths must contain exact candidate/path pairs")
        if (
            tuple(sorted(self.branch_paths, key=lambda item: item[0]))
            != self.branch_paths
        ):
            raise ValueError("branch_paths must use canonical candidate-ID order")
        path_map = dict(self.branch_paths)
        if len(path_map) != len(self.branch_paths):
            raise ValueError("branch_paths cannot repeat a candidate")
        required_ids = {
            candidate_id for row in self.eligible_rows for candidate_id in row.pair_ids
        }
        if not required_ids.issubset(path_map):
            raise ValueError("every eligible branch requires a frozen path")
        exploit, coverage = _select_batch_rows(self.eligible_rows, path_map)
        if self.exploit_pair_ids != (None if exploit is None else exploit.pair_ids):
            raise ValueError("exploit_pair_ids do not match the frozen rule")
        if self.coverage_pair_ids != (None if coverage is None else coverage.pair_ids):
            raise ValueError("coverage_pair_ids do not match the incremental rule")

    @classmethod
    def from_base_selection(
        cls,
        base: DisjointPairSelectionDecision,
        *,
        branch_paths: Mapping[CandidateId, str],
    ) -> BoilsV5BatchPairSelectionDecision:
        if type(base) is not DisjointPairSelectionDecision:
            raise TypeError("base must be an exact disjoint-pair decision")
        base.revalidate()
        frozen_paths = tuple(sorted(branch_paths.items(), key=lambda item: item[0]))
        exploit, coverage = _select_batch_rows(base.eligible_rows, branch_paths)
        return cls(
            eligible_rows=base.eligible_rows,
            branch_paths=frozen_paths,
            exploit_pair_ids=None if exploit is None else exploit.pair_ids,
            coverage_pair_ids=None if coverage is None else coverage.pair_ids,
        )

    def revalidate(self) -> None:
        if type(self) is not BoilsV5BatchPairSelectionDecision:
            raise TypeError("selection must be an exact BOiLS batch decision")
        BoilsV5BatchPairSelectionDecision.__post_init__(self)

    def _row_for(
        self, pair_ids: tuple[CandidateId, CandidateId] | None
    ) -> DisjointPairScoreRow | None:
        if pair_ids is None:
            return None
        return next(row for row in self.eligible_rows if row.pair_ids == pair_ids)

    @property
    def exploit(self) -> DisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.exploit_pair_ids)

    @property
    def coverage(self) -> DisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.coverage_pair_ids)

    def _trace_payload(self) -> dict[str, object]:
        exploit = self._row_for(self.exploit_pair_ids)
        coverage = self._row_for(self.coverage_pair_ids)
        exploit_target = (
            None if exploit is None else exploit.pair.target_configuration_sha256
        )
        path_map = dict(self.branch_paths)
        coverage_rows = []
        if exploit is not None:
            for row in self.eligible_rows:
                if row.pair.target_configuration_sha256 == exploit_target:
                    continue
                metrics = _incremental_coverage_metrics(row, exploit, path_map)
                coverage_rows.append(
                    {
                        "pair_ids": [value.value for value in row.pair_ids],
                        **metrics,
                        "path_family_exposure_sum": row.path_family_exposure_sum,
                        "tie_key": [
                            -metrics["new_path_count"],
                            -metrics["new_family_count"],
                            -metrics["new_role_count"],
                            -metrics["new_parent_count"],
                            -row.distinct_family_count,
                            -row.distinct_role_count,
                            row.path_family_exposure_sum,
                            *_pair_id_key(row),
                        ],
                    }
                )
        return {
            "event_type": "boils_v5_recombination_pair_selected",
            "policy_id": support.BATCH_INCREMENTAL_COVERAGE_POLICY_ID,
            "policy_version": support.BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION,
            "base_exploit_policy_id": DisjointParentPairPolicy.policy_id,
            "base_exploit_policy_version": DisjointParentPairPolicy.policy_version,
            "branch_paths": [
                {"candidate_id": candidate_id.value, "path": path}
                for candidate_id, path in self.branch_paths
            ],
            "eligible_rows": [
                row.to_trace_record(
                    exploit_target_configuration_sha256=exploit_target,
                )
                for row in self.eligible_rows
            ],
            "exploit_pair_ids": (
                None
                if exploit is None
                else [candidate_id.value for candidate_id in exploit.pair_ids]
            ),
            "coverage_pair_ids": (
                None
                if coverage is None
                else [candidate_id.value for candidate_id in coverage.pair_ids]
            ),
            "batch_incremental_coverage_rows": coverage_rows,
            "exploit_rule": (
                "maximum branch reward sum; maximum distinct roles; canonical IDs"
            ),
            "coverage_rule": (
                "after exploit, maximize new paths, families, roles, and parents; "
                "then within-pair diversity, minimum exposure, and canonical IDs"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        self.revalidate()
        return _record_hash("batch-pair-selection", self._trace_payload())

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._trace_payload(), "decision_sha256": self.decision_sha256}


@dataclass(frozen=True, slots=True)
class BoilsV5Generation2Decision:
    """Complete pair enumeration and exploit/coverage selection evidence."""

    g1_checkpoint: tuple[BoilsV5G1CheckpointRow, ...]
    enumeration: tuple[BoilsV5PairEnumerationRow, ...]
    selection: BoilsV5BatchPairSelectionDecision
    individual_frozen_rewards: tuple[tuple[str, float], ...]
    selected_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(row.slot_id for row in self.g1_checkpoint) != _G1_SLOT_ORDER:
            raise ValueError("G1 checkpoint must record every frozen slot in order")
        if type(self.selection) is not BoilsV5BatchPairSelectionDecision:
            raise TypeError("selection must be an exact BOiLS batch decision")
        self.selection.revalidate()
        if self.selected_slot_ids not in {
            (),
            ("G2-E",),
            ("G2-E", "G2-X"),
        }:
            raise ValueError("G2 selected slots must be a typed prefix of G2-E,G2-X")
        if len(self.selection.eligible_rows) != sum(
            row.eligible for row in self.enumeration
        ):
            raise ValueError("selection must contain every eligible enumeration row")

    def to_trace_record(self) -> dict[str, object]:
        body = {
            "event_type": "boils_v5_generation2_decided",
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "failed_slot_continuation": {
                "policy_id": support.FAILED_SLOT_CONTINUATION_POLICY_ID,
                "policy_version": support.FAILED_SLOT_CONTINUATION_POLICY_VERSION,
                "substitution_allowed": False,
                "g1_checkpoint": [row.to_trace_record() for row in self.g1_checkpoint],
            },
            "enumeration": [row.to_trace_record() for row in self.enumeration],
            "selection": self.selection.to_trace_record(),
            "individual_frozen_rewards": [
                {"candidate_id": candidate_id, "reward_hex": reward.hex()}
                for candidate_id, reward in self.individual_frozen_rewards
            ],
            "selected_slot_ids": list(self.selected_slot_ids),
        }
        return {**body, "decision_sha256": _record_hash("generation2", body)}

    @property
    def decision_sha256(self) -> str:
        return str(self.to_trace_record()["decision_sha256"])


class _ExposureLedger:
    def __init__(self, initial: Sequence[PathFamilyExposure]) -> None:
        self._counts: dict[tuple[bytes, str], tuple[JsonPath, int]] = {}
        for item in initial:
            key = canonical_path_bytes(item.path), item.family
            if key in self._counts:
                raise ValueError("initial exposure cells cannot repeat")
            self._counts[key] = (item.path, item.count)

    def snapshot(self) -> tuple[PathFamilyExposure, ...]:
        return tuple(
            PathFamilyExposure(path, family, count)
            for (path_bytes, family), (path, count) in sorted(
                self._counts.items(), key=lambda item: item[0]
            )
        )

    def expose(self, options: Sequence[AtomicEditOption]) -> None:
        for option in options:
            key = canonical_path_bytes(option.path), option.family
            _, count = self._counts.get(key, (option.path, 0))
            self._counts[key] = (option.path, count + 1)


def _initial_exposures() -> tuple[PathFamilyExposure, ...]:
    values = []
    for path_text, family, count in support.PREORACLE_PATH_FAMILY_EXPOSURES:
        prefix, suffix = "$.sequence[", "]"
        if not path_text.startswith(prefix) or not path_text.endswith(suffix):
            raise RuntimeError("support exposure path is not a BOiLS sequence path")
        ordinal = path_text[len(prefix) : -len(suffix)]
        if not ordinal.isdigit() or (len(ordinal) > 1 and ordinal.startswith("0")):
            raise RuntimeError("support exposure path index is not canonical")
        index = int(ordinal)
        if not 0 <= index < len(support.PARENT_C_SEQUENCE):
            raise RuntimeError("support exposure path index is outside parent C")
        path = JsonPath((ObjectKey("sequence"), ArrayIndex(index)))
        values.append(PathFamilyExposure(path, family, count))
    return tuple(
        sorted(values, key=lambda item: (canonical_path_bytes(item.path), item.family))
    )


def _support_hash() -> str:
    # Evaluator provenance is deliberately a runner-owned gate.  The planner
    # hashes only the evaluator-independent support facts it actually consumes.
    record = {
        "parent_typed_json_sha256": support.PARENT_C_TYPED_JSON_SHA256,
        "parent_objectives": [list(item) for item in support.PARENT_C_OBJECTIVES],
        "reference_point": list(support.REFERENCE_POINT),
        "front_aligned_reward": {
            "policy_id": support.FRONT_ALIGNED_REWARD_POLICY_ID,
            "policy_version": support.FRONT_ALIGNED_REWARD_POLICY_VERSION,
            "front_extension_raw_credit_hex": (
                support.FRONT_EXTENSION_RAW_CREDIT.hex()
            ),
        },
        "failed_slot_continuation": {
            "policy_id": support.FAILED_SLOT_CONTINUATION_POLICY_ID,
            "policy_version": support.FAILED_SLOT_CONTINUATION_POLICY_VERSION,
            "substitution_allowed": False,
        },
        "batch_incremental_coverage": {
            "policy_id": support.BATCH_INCREMENTAL_COVERAGE_POLICY_ID,
            "policy_version": support.BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION,
        },
        "area_required_action": support.AREA_REQUIRED_ACTION,
        "depth_required_action": support.DEPTH_REQUIRED_ACTION,
        "uncertainty_required_action": support.UNCERTAINTY_REQUIRED_ACTION,
        "uncertainty_required_family": support.UNCERTAINTY_REQUIRED_FAMILY,
        "uncertainty_coverage_obligation_id": (
            support.UNCERTAINTY_COVERAGE_OBLIGATION_ID
        ),
        "uncertainty_coverage_obligation_version": (
            support.UNCERTAINTY_COVERAGE_OBLIGATION_VERSION
        ),
        "uncertainty_coverage_obligation_rationale": (
            support.UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE
        ),
        "protocol_correction": support.protocol_correction_record(),
        "preoracle_exposures": [
            list(item) for item in support.PREORACLE_PATH_FAMILY_EXPOSURES
        ],
        "area_cards": [item.to_manifest_record() for item in support.AREA_CARD_PAIR],
        "depth_cards": [item.to_manifest_record() for item in support.DEPTH_CARD_PAIR],
    }
    return _record_hash("support-input", record)


def _atomic_plan(
    *,
    parent: EvolutionCandidate,
    generation: int,
    label: str,
    path: JsonPath,
    palette: Sequence[AtomicEditOption],
    use_memory: bool,
    phase: str,
) -> InvocationPlan:
    options = tuple(palette)
    if not options or any(option.path != path for option in options):
        raise ValueError("atomic palette must be non-empty and single-path")
    return InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=generation,
        label=label,
        allowed_top_level=("sequence",),
        use_memory=use_memory,
        memory_subset_size=1 if use_memory else 0,
        memory_exploration_probability=Fraction(1, 1) if use_memory else None,
        memory_score_phase=phase if use_memory else None,
        phase=phase,
        mutation_contract=MutationContract(
            editable_paths=(path,),
            max_changed_paths=1,
            max_operations=1,
            allow_abstention=False,
        ),
        mutation_response_mode=MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1,
        atomic_replacement_options=tuple(option.replacement for option in options),
    )


def _reward(
    state: OptimizerState,
) -> tuple[FrozenWaveReward, BoilsV5FrozenFrontAlignedReward]:
    snapshot = FrozenArchiveSnapshot2D.create(
        objectives=state.archive.objectives,
        reference_point={
            "total_lut_count": support.REFERENCE_POINT[0],
            "total_levels": support.REFERENCE_POINT[1],
        },
        archive_points=tuple(
            candidate.objective_map for candidate in state.archive.front_candidates
        ),
    )
    policy = BoilsV5FrozenFrontAlignedReward(snapshot)
    return (
        FrozenWaveReward(
            binding=RewardPolicyBinding(policy, policy.definition_hash),
            archive_snapshot_hash=state.archive_snapshot_hash,
            reward_snapshot_hash=policy.snapshot_hash,
        ),
        policy,
    )


def _required_option(
    options: Sequence[AtomicEditOption],
    *,
    path: JsonPath,
    replacement: str,
) -> AtomicEditOption:
    matches = tuple(
        option
        for option in options
        if option.path == path and option.replacement == replacement
    )
    if len(matches) != 1:
        raise BoilsV5PlanningError("required catalog option is missing or ambiguous")
    return matches[0]


def _actual_option(
    *,
    ancestor: EvolutionCandidate,
    child: EvolutionCandidate,
    options: Sequence[AtomicEditOption],
) -> AtomicEditOption:
    matches: list[AtomicEditOption] = []
    for option in options:
        if (
            len(option.path.segments) != 2
            or option.path.segments[0] != ObjectKey("sequence")
            or type(option.path.segments[1]) is not ArrayIndex
        ):
            raise BoilsV5PlanningError("BOiLS catalog contains a non-sequence option")
        sequence = list(ancestor.configuration_dict["sequence"])
        index = option.path.segments[1].value
        sequence[index] = option.replacement
        configuration = {"sequence": sequence}
        if typed_json_sha256(configuration) == child.occurrence.configuration_hash:
            matches.append(option)
    if len(matches) != 1:
        raise BoilsV5PlanningError(
            f"G1 branch {child.candidate_id.value} is not one exact catalog edit"
        )
    return matches[0]


class BoilsBudgetedV5Planner:
    """Stateful two-wave policy consumed by :class:`BudgetedAgenticOptimizer`."""

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __init__(
        self,
        id_factory: IdFactory,
        *,
        catalog: BoilsAtomicVariationCatalog | None = None,
        palette_policy: TaskKeyedPalettePolicy | None = None,
        recombiner: DisjointPatchRecombiner | None = None,
        pair_policy: DisjointParentPairPolicy | None = None,
        decision_sink: DecisionSink | None = None,
    ) -> None:
        if not isinstance(id_factory, IdFactory):
            raise TypeError("id_factory must implement IdFactory")
        self.id_factory = id_factory
        self.catalog = BoilsAtomicVariationCatalog() if catalog is None else catalog
        self.palette_policy = (
            TaskKeyedPalettePolicy(seed=PALETTE_SEED)
            if palette_policy is None
            else palette_policy
        )
        if self.palette_policy.seed != PALETTE_SEED:
            raise ValueError("v5 requires the frozen task-keyed palette seed")
        self.recombiner = (
            DisjointPatchRecombiner() if recombiner is None else recombiner
        )
        self.pair_policy = (
            DisjointParentPairPolicy() if pair_policy is None else pair_policy
        )
        if decision_sink is not None and not callable(decision_sink):
            raise TypeError("decision_sink must be callable")
        self._decision_sink = decision_sink
        self._g1_decision: BoilsV5Generation1Decision | None = None
        self._g2_decision: BoilsV5Generation2Decision | None = None
        self._a0_reward: BoilsV5FrozenFrontAlignedReward | None = None
        self._materializations: dict[
            tuple[CandidateId, CandidateId], DisjointPatchMaterialization
        ] = {}

    @property
    def generation1_decision(self) -> BoilsV5Generation1Decision | None:
        return self._g1_decision

    @property
    def generation2_decision(self) -> BoilsV5Generation2Decision | None:
        return self._g2_decision

    def _publish_decision(self, record: Mapping[str, object]) -> None:
        """Synchronously cross the runner's durable decision boundary."""

        if self._decision_sink is not None:
            self._decision_sink(record)

    def _validate_seed(self, state: OptimizerState) -> EvolutionCandidate:
        if len(state.candidates) != 1:
            raise BoilsV5PlanningError("generation zero requires exact singleton C")
        seed = state.candidates[0]
        if (
            seed.occurrence.configuration_hash != support.PARENT_C_TYPED_JSON_SHA256
            or seed.objectives != support.PARENT_C_OBJECTIVES
            or not seed.valid
            or not seed.operator_compliant
        ):
            raise BoilsV5PlanningError("planner received a seed other than exact C")
        return seed

    def _plan_g1(self, state: OptimizerState) -> GenerationPlan:
        if self._g1_decision is not None:
            raise BoilsV5PlanningError("generation one has already been frozen")
        seed = self._validate_seed(state)
        wave_reward, self._a0_reward = _reward(state)
        catalog_options = self.catalog.options(seed)
        ledger = _ExposureLedger(_initial_exposures())
        initial = ledger.snapshot()

        area_required = _required_option(
            catalog_options,
            path=support.AREA_PATH,
            replacement=support.AREA_REQUIRED_ACTION,
        )
        area = self.palette_policy.select(
            task_key="boils_v5.g1.area_pair",
            options=catalog_options,
            palette_size=PALETTE_SIZE,
            max_options_per_family=MAX_OPTIONS_PER_FAMILY,
            exposures=ledger.snapshot(),
            path=support.AREA_PATH,
            required_option_ids=(area_required.option_id,),
        )
        a1_before = ledger.snapshot()
        ledger.expose(area.palette)
        a2_before = ledger.snapshot()
        ledger.expose(area.palette)

        depth_required = _required_option(
            catalog_options,
            path=support.DEPTH_PATH,
            replacement=support.DEPTH_REQUIRED_ACTION,
        )
        depth = self.palette_policy.select(
            task_key="boils_v5.g1.depth_pair",
            options=catalog_options,
            palette_size=PALETTE_SIZE,
            max_options_per_family=MAX_OPTIONS_PER_FAMILY,
            exposures=ledger.snapshot(),
            path=support.DEPTH_PATH,
            required_option_ids=(depth_required.option_id,),
        )
        d1_before = ledger.snapshot()
        ledger.expose(depth.palette)
        d2_before = ledger.snapshot()
        ledger.expose(depth.palette)

        remaining_paths = (support.UNCERTAINTY_PATH, support.COVERAGE_PATH)
        remaining_options = tuple(
            option for option in catalog_options if option.path in remaining_paths
        )
        uncertainty_required = _required_option(
            remaining_options,
            path=support.UNCERTAINTY_PATH,
            replacement=support.UNCERTAINTY_REQUIRED_ACTION,
        )
        if uncertainty_required.family != support.UNCERTAINTY_REQUIRED_FAMILY:
            raise BoilsV5PlanningError(
                "uncertainty required option family differs from its obligation"
            )
        uncertainty = self.palette_policy.select(
            task_key="boils_v5.g1.uncertainty",
            options=remaining_options,
            palette_size=PALETTE_SIZE,
            max_options_per_family=MAX_OPTIONS_PER_FAMILY,
            exposures=ledger.snapshot(),
            required_option_ids=(uncertainty_required.option_id,),
        )
        if uncertainty.chosen_path != support.UNCERTAINTY_PATH:
            raise BoilsV5PlanningError(
                "frozen task-keyed policy no longer selects uncertainty path 12"
            )
        u_before = ledger.snapshot()
        ledger.expose(uncertainty.palette)
        other_path = next(
            path for path in remaining_paths if path != uncertainty.chosen_path
        )
        coverage = self.palette_policy.select(
            task_key="boils_v5.g1.coverage",
            options=remaining_options,
            palette_size=1,
            max_options_per_family=MAX_OPTIONS_PER_FAMILY,
            exposures=ledger.snapshot(),
            path=other_path,
        )
        if coverage.chosen_path != support.COVERAGE_PATH:
            raise BoilsV5PlanningError(
                "frozen task-keyed policy no longer selects coverage path 18"
            )
        x_before = ledger.snapshot()
        ledger.expose(coverage.palette)
        coverage_option = coverage.palette[0]

        slot_decisions = (
            BoilsV5G1SlotDecision(
                "G1-A1", _SLOT_ROLES["G1-A1"], "model", area, a1_before
            ),
            BoilsV5G1SlotDecision(
                "G1-A2", _SLOT_ROLES["G1-A2"], "model", area, a2_before
            ),
            BoilsV5G1SlotDecision(
                "G1-D1", _SLOT_ROLES["G1-D1"], "model", depth, d1_before
            ),
            BoilsV5G1SlotDecision(
                "G1-D2", _SLOT_ROLES["G1-D2"], "model", depth, d2_before
            ),
            BoilsV5G1SlotDecision(
                "G1-U", _SLOT_ROLES["G1-U"], "model", uncertainty, u_before
            ),
            BoilsV5G1SlotDecision(
                "G1-X",
                _SLOT_ROLES["G1-X"],
                "engine",
                coverage,
                x_before,
                coverage_option,
            ),
        )
        decision = BoilsV5Generation1Decision(
            initial_exposures=initial,
            slots=slot_decisions,
            final_exposures=ledger.snapshot(),
            support_manifest_sha256=_support_hash(),
        )
        self._g1_decision = decision
        self._publish_decision(decision.to_trace_record())

        plans: dict[str, InvocationPlan] = {}
        for item in slot_decisions:
            use_memory = item.slot_id in {"G1-A1", "G1-A2", "G1-D1", "G1-D2"}
            phase = {
                "G1-A1": support.AREA_PHASE,
                "G1-A2": support.AREA_PHASE,
                "G1-D1": support.DEPTH_PHASE,
                "G1-D2": support.DEPTH_PHASE,
                "G1-U": support.UNCERTAINTY_PHASE,
                "G1-X": "boils_v5.g1.coverage",
            }[item.slot_id]
            plans[item.slot_id] = _atomic_plan(
                parent=seed,
                generation=1,
                label=_G1_LABELS[item.slot_id],
                path=item.palette.chosen_path,
                palette=item.palette.palette,
                use_memory=use_memory,
                phase=phase,
            )

        x_plan = plans["G1-X"]
        x_invocation = MaterializedInvocation(
            plan=x_plan,
            draft=AtomicMutationDraft(
                path=coverage_option.path,
                replacement=coverage_option.replacement,
                design_rationale=(
                    "Engine-owned task-keyed representation-coverage intervention."
                ),
            ),
            candidate_id=self.id_factory.new_candidate_id(),
            materialization_policy_id="boils_v5_task_keyed_coverage",
            materialization_policy_version=1,
            materialization_receipt_hash=_record_hash(
                "coverage-materialization",
                {
                    "generation1_decision_sha256": decision.decision_sha256,
                    "slot_decision_sha256": decision.slot("G1-X").decision_sha256,
                    "option_identity_sha256": coverage_option.identity_sha256,
                },
            ),
        )
        slots = (
            OptimizerSlot.model(
                slot_id="G1-A1", role=_SLOT_ROLES["G1-A1"], plan=plans["G1-A1"]
            ),
            OptimizerSlot.model(
                slot_id="G1-A2", role=_SLOT_ROLES["G1-A2"], plan=plans["G1-A2"]
            ),
            OptimizerSlot.model(
                slot_id="G1-D1", role=_SLOT_ROLES["G1-D1"], plan=plans["G1-D1"]
            ),
            OptimizerSlot.model(
                slot_id="G1-D2", role=_SLOT_ROLES["G1-D2"], plan=plans["G1-D2"]
            ),
            OptimizerSlot.model(
                slot_id="G1-U", role=_SLOT_ROLES["G1-U"], plan=plans["G1-U"]
            ),
            OptimizerSlot.engine(
                slot_id="G1-X", role=_SLOT_ROLES["G1-X"], invocation=x_invocation
            ),
        )
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=wave_reward,
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("decision_sha256", decision.decision_sha256),
                        ("palette_policy_id", self.palette_policy.policy_id),
                        (
                            "palette_policy_version",
                            str(self.palette_policy.policy_version),
                        ),
                        (
                            "reward_policy_id",
                            support.FRONT_ALIGNED_REWARD_POLICY_ID,
                        ),
                        (
                            "reward_policy_version",
                            str(support.FRONT_ALIGNED_REWARD_POLICY_VERSION),
                        ),
                        ("support_manifest_sha256", decision.support_manifest_sha256),
                    )
                )
            ),
        )

    def _g1_checkpoint(
        self, state: OptimizerState
    ) -> tuple[
        tuple[BoilsV5G1CheckpointRow, ...],
        tuple[tuple[str, EvolutionCandidate], ...],
    ]:
        if self._g1_decision is None or self._a0_reward is None:
            raise BoilsV5PlanningError("generation one was not frozen by this planner")
        by_label: dict[str, EvolutionCandidate] = {}
        expected_labels = set(_G1_LABELS.values())
        for candidate in state.candidates:
            if candidate.label not in expected_labels:
                continue
            if candidate.label in by_label:
                raise BoilsV5PlanningError("G1 contains duplicate slot labels")
            by_label[candidate.label] = candidate
        if any(candidate.generation != 1 for candidate in by_label.values()):
            raise BoilsV5PlanningError("G1 slot candidate has the wrong generation")
        checkpoint: list[BoilsV5G1CheckpointRow] = []
        eligible: list[tuple[str, EvolutionCandidate]] = []
        for slot_id in _G1_SLOT_ORDER:
            candidate = by_label.get(_G1_LABELS[slot_id])
            if candidate is None:
                status = "missing_candidate"
            elif not candidate.valid:
                status = "invalid_candidate"
            elif not candidate.operator_compliant:
                status = "operator_noncompliant"
            else:
                status = "eligible"
                eligible.append((slot_id, candidate))
            checkpoint.append(
                BoilsV5G1CheckpointRow(
                    slot_id=slot_id,
                    status=status,
                    candidate_id=(
                        None if candidate is None else candidate.candidate_id
                    ),
                )
            )
        return tuple(checkpoint), tuple(eligible)

    def _branch_facts(
        self,
        *,
        seed: EvolutionCandidate,
        candidate: EvolutionCandidate,
        slot_id: str,
        options: Sequence[AtomicEditOption],
    ) -> tuple[DisjointBranchFacts, AtomicEditOption]:
        if not candidate.valid or not candidate.operator_compliant:
            raise BoilsV5PlanningError("ineligible branch passed the G2 fact boundary")
        assert self._g1_decision is not None and self._a0_reward is not None
        option = _actual_option(ancestor=seed, child=candidate, options=options)
        exposure = next(
            (
                item.count
                for item in self._g1_decision.slot(slot_id).exposures_before
                if item.path == option.path and item.family == option.family
            ),
            0,
        )
        return (
            DisjointBranchFacts(
                candidate_id=candidate.candidate_id,
                reward=float(self._a0_reward.record(candidate).reward),
                role=_SLOT_ROLES[slot_id],
                family=option.family,
                path_family_exposure=exposure,
            ),
            option,
        )

    def _plan_g2(self, state: OptimizerState) -> GenerationPlan:
        if self._g2_decision is not None:
            raise BoilsV5PlanningError("generation two has already been frozen")
        if state.generation != 1:
            raise BoilsV5PlanningError("G2 requires the closed G1 checkpoint")
        seed = next(
            (
                candidate
                for candidate in state.candidates
                if candidate.generation == 0
                and candidate.occurrence.configuration_hash
                == support.PARENT_C_TYPED_JSON_SHA256
            ),
            None,
        )
        if seed is None:
            raise BoilsV5PlanningError("exact ancestor C is absent from G2 state")
        g1_checkpoint, eligible = self._g1_checkpoint(state)
        options = self.catalog.options(seed)
        facts: dict[CandidateId, DisjointBranchFacts] = {}
        branch_paths: dict[CandidateId, str] = {}
        for slot_id, candidate in eligible:
            branch, option = self._branch_facts(
                seed=seed,
                candidate=candidate,
                slot_id=slot_id,
                options=options,
            )
            facts[candidate.candidate_id] = branch
            branch_paths[candidate.candidate_id] = _path_record(option.path)

        candidates_by_id = {
            candidate.candidate_id: candidate for _, candidate in eligible
        }
        ordered_ids = tuple(sorted(candidates_by_id))
        rows: list[BoilsV5PairEnumerationRow] = []
        replay_pairs: list[ReplayVerifiedDisjointPair] = []
        materializations: dict[
            tuple[CandidateId, CandidateId], DisjointPatchMaterialization
        ] = {}
        for left_id, right_id in itertools.combinations(ordered_ids, 2):
            target_id = self.id_factory.new_candidate_id()
            try:
                materialization = self.recombiner.materialize(
                    ancestor=seed.configuration,
                    ancestor_candidate_id=seed.candidate_id,
                    left=candidates_by_id[left_id].configuration,
                    left_candidate_id=left_id,
                    right=candidates_by_id[right_id].configuration,
                    right_candidate_id=right_id,
                    target_candidate_id=target_id,
                )
            except DisjointPatchRecombinationError as exc:
                rows.append(
                    BoilsV5PairEnumerationRow(
                        left_id,
                        right_id,
                        False,
                        target_id,
                        None,
                        None,
                        type(exc).__name__,
                        hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                    )
                )
                continue
            target_hash = typed_json_sha256(materialization.configuration)
            row = BoilsV5PairEnumerationRow(
                left_id,
                right_id,
                True,
                target_id,
                target_hash,
                materialization.receipt_sha256,
                None,
                None,
            )
            rows.append(row)
            pair_ids = (left_id, right_id)
            materializations[pair_ids] = materialization
            replay_pairs.append(
                ReplayVerifiedDisjointPair(
                    left=facts[left_id],
                    right=facts[right_id],
                    target_configuration_sha256=target_hash,
                    materialization_receipt_sha256=materialization.receipt_sha256,
                )
            )
        base_selection = self.pair_policy.select(replay_pairs)
        selection = BoilsV5BatchPairSelectionDecision.from_base_selection(
            base_selection,
            branch_paths=branch_paths,
        )
        selected_rows = tuple(
            row for row in (selection.exploit, selection.coverage) if row is not None
        )
        selected_slot_ids = tuple(("G2-E", "G2-X")[: len(selected_rows)])
        individual_rewards = tuple(
            sorted(
                (
                    candidate_id.value,
                    facts[candidate_id].reward,
                )
                for candidate_id in facts
            )
        )
        decision = BoilsV5Generation2Decision(
            g1_checkpoint=g1_checkpoint,
            enumeration=tuple(rows),
            selection=selection,
            individual_frozen_rewards=individual_rewards,
            selected_slot_ids=selected_slot_ids,
        )
        self._g2_decision = decision
        self._materializations = materializations
        self._publish_decision(decision.to_trace_record())
        wave_reward, _ = _reward(state)

        slots: list[OptimizerSlot] = []
        for slot_id, score_row in zip(selected_slot_ids, selected_rows, strict=True):
            left_id, right_id = score_row.pair_ids
            plan = InvocationPlan(
                operator_kind=OperatorKind.THREE_WAY_RECOMBINATION,
                parents=(candidates_by_id[left_id], candidates_by_id[right_id]),
                common_ancestor=seed,
                generation=2,
                label=slot_id,
                phase="boils_v5_recombination",
            )
            invocation = materialized_disjoint_invocation(
                plan=plan,
                materialization=materializations[(left_id, right_id)],
            )
            slots.append(
                OptimizerSlot.engine(
                    slot_id=slot_id,
                    role=("exploit_union" if slot_id == "G2-E" else "coverage_union"),
                    invocation=invocation,
                )
            )
        return GenerationPlan(
            generation=2,
            slots=tuple(slots),
            reward=wave_reward,
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("decision_sha256", decision.decision_sha256),
                        (
                            "base_exploit_policy_id",
                            self.pair_policy.policy_id,
                        ),
                        (
                            "base_exploit_policy_version",
                            str(self.pair_policy.policy_version),
                        ),
                        (
                            "pair_policy_id",
                            support.BATCH_INCREMENTAL_COVERAGE_POLICY_ID,
                        ),
                        (
                            "pair_policy_version",
                            str(support.BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION),
                        ),
                    )
                )
            ),
        )

    def plan(self, state: OptimizerState, budget: OptimizerBudget) -> GenerationPlan:
        if type(state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        if type(budget) is not OptimizerBudget:
            raise TypeError("budget must be an exact OptimizerBudget")
        if state.generation == 0:
            return self._plan_g1(state)
        if state.generation == 1:
            return self._plan_g2(state)
        raise BoilsV5PlanningError("BOiLS v5 stops after exactly two generations")

    def to_summary_record(self) -> dict[str, object]:
        body = {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "palette_seed": PALETTE_SEED,
            "generation1": (
                None
                if self._g1_decision is None
                else self._g1_decision.to_trace_record()
            ),
            "generation2": (
                None
                if self._g2_decision is None
                else self._g2_decision.to_trace_record()
            ),
        }
        return {**body, "summary_sha256": _record_hash("planner-summary", body)}


__all__ = [
    "BoilsV5BatchPairSelectionDecision",
    "BoilsBudgetedV5Planner",
    "BoilsV5FrontAlignedRewardRecord",
    "BoilsV5FrozenFrontAlignedReward",
    "BoilsV5Generation1Decision",
    "BoilsV5Generation2Decision",
    "BoilsV5G1CheckpointRow",
    "BoilsV5G1SlotDecision",
    "BoilsV5PairEnumerationRow",
    "BoilsV5PlanningError",
    "MAX_OPTIONS_PER_FAMILY",
    "PALETTE_SEED",
    "PALETTE_SIZE",
    "POLICY_ID",
    "POLICY_VERSION",
]
