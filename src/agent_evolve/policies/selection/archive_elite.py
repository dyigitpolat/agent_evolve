"""Authenticated, benchmark-neutral parent selection for evolution planners.

The strict elite policy rotates only over the archive's exact current front.
The singleton-safe reservoir policy replays archive admissibility over complete
candidate history and ranks it from declared objective directions, without any
benchmark branches.  Both produce deterministic, snapshot-bound receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import OptimizerState
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoCandidateRef,
    ParetoDecision,
    ParetoDecisionReason,
    pareto_candidate_hash,
)
from agent_evolve.core.problem import (
    ObjectiveSpec,
    ProblemContractError,
    normalize_objective_values,
    validate_objective_specs,
)
from agent_evolve.core.results import dominates
from agent_evolve.domain.patch import require_sha256


POLICY_ID = "task_keyed_archive_elite_parent"
POLICY_VERSION = 1
_SCHEMA_VERSION = 1
_MAX_ROTATION_INDEX = (1 << 63) - 1
_DEFINITION_DOMAIN = b"agent-evolve:archive-elite-parent:def:v1\x00"
_ELIGIBLE_FRONT_DOMAIN = b"agent-evolve:archive-elite-parent:front:v1\x00"
_ROTATION_DOMAIN = b"agent-evolve:archive-elite-parent:rotation:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:archive-elite-parent:receipt:v1\x00"
_DEFINITION = {
    "eligible_source": "complete OptimizerState.archive.front_candidates only",
    "archive_authority": "exact ParetoArchiveSnapshot and snapshot hash",
    "outcome_or_objective_access": False,
    "selection": "task/archive/front-keyed cyclic rotation without replacement",
    "stale_or_foreign_candidate_admission": False,
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN + _canonical_json(_DEFINITION)
).hexdigest()


RESERVOIR_POLICY_ID = "task_keyed_archive_ranked_reservoir_parent"
RESERVOIR_POLICY_VERSION = 1
_RESERVOIR_SCHEMA_VERSION = 1
_RESERVOIR_DEFINITION_DOMAIN = (
    b"agent-evolve:archive-ranked-reservoir-parent:def:v1\x00"
)
_RESERVOIR_RANKING_DOMAIN = (
    b"agent-evolve:archive-ranked-reservoir-parent:ranking:v1\x00"
)
_RESERVOIR_DOMAIN = b"agent-evolve:archive-ranked-reservoir-parent:reservoir:v1\x00"
_RESERVOIR_ROTATION_DOMAIN = (
    b"agent-evolve:archive-ranked-reservoir-parent:rotation:v1\x00"
)
_RESERVOIR_RECEIPT_DOMAIN = (
    b"agent-evolve:archive-ranked-reservoir-parent:receipt:v1\x00"
)
_RESERVOIR_DEFINITION = {
    "eligible_source": (
        "complete archive-authenticated, gate-passing unique-configuration history"
    ),
    "ranking": (
        "nondomination rank; multiobjective crowding when meaningful; "
        "minimax dense objective rank; authenticated proposal-sequence recency; "
        "canonical identity"
    ),
    "reservoir": "first min(reservoir_limit, eligible_count) ranked candidates",
    "selection": (
        "task/archive/ranking-keyed cyclic rotation without replacement; "
        "return min(requested_parent_count, reservoir_cardinality)"
    ),
    "benchmark_branching": False,
    "stale_or_foreign_candidate_admission": False,
}
RESERVOIR_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _RESERVOIR_DEFINITION_DOMAIN + _canonical_json(_RESERVOIR_DEFINITION)
).hexdigest()


def _reference_record(reference: ParetoCandidateRef) -> dict[str, str]:
    if type(reference) is not ParetoCandidateRef:
        raise TypeError("front references must be exact ParetoCandidateRef values")
    ParetoCandidateRef.__post_init__(reference)
    return reference.to_trace_record()


def _reference_key(reference: ParetoCandidateRef) -> tuple[str, str, str]:
    return (
        reference.configuration_hash,
        reference.candidate_id.value,
        reference.candidate_hash,
    )


def _candidate_reference(candidate: EvolutionCandidate) -> ParetoCandidateRef:
    if type(candidate) is not EvolutionCandidate:
        raise TypeError("archive front must contain exact EvolutionCandidate values")
    EvolutionCandidate.__post_init__(candidate)
    return ParetoCandidateRef(
        candidate_id=candidate.candidate_id,
        candidate_hash=pareto_candidate_hash(candidate),
        configuration_hash=candidate.occurrence.configuration_hash,
    )


def _eligible_front_sha256(front: tuple[ParetoCandidateRef, ...]) -> str:
    if type(front) is not tuple or not front:
        raise ValueError("eligible archive front must be a non-empty exact tuple")
    return hashlib.sha256(
        _ELIGIBLE_FRONT_DOMAIN
        + _canonical_json([_reference_record(reference) for reference in front])
    ).hexdigest()


def _rotation_anchor(
    *,
    task_sha256: str,
    archive_snapshot_hash: str,
    eligible_front_sha256: str,
    cardinality: int,
) -> int:
    require_sha256(task_sha256, "task_sha256")
    require_sha256(archive_snapshot_hash, "archive_snapshot_hash")
    require_sha256(eligible_front_sha256, "eligible_front_sha256")
    if type(cardinality) is not int or cardinality <= 0:
        raise ValueError("archive front cardinality must be positive")
    digest = hashlib.sha256(
        _ROTATION_DOMAIN
        + bytes.fromhex(task_sha256)
        + bytes.fromhex(archive_snapshot_hash)
        + bytes.fromhex(eligible_front_sha256)
    ).digest()
    return int.from_bytes(digest, "big", signed=False) % cardinality


def _validated_front(
    state: OptimizerState,
) -> tuple[tuple[EvolutionCandidate, ...], tuple[ParetoCandidateRef, ...]]:
    if type(state) is not OptimizerState:
        raise TypeError("state must be an exact OptimizerState")
    OptimizerState.__post_init__(state)
    candidates = state.archive.front_candidates
    references = state.archive.front_references
    if type(candidates) is not tuple or type(references) is not tuple:
        raise TypeError("archive front and references must be exact tuples")
    if not candidates:
        raise ValueError("archive elite parent selection requires a non-empty front")
    if len(candidates) != len(references):
        raise ValueError("archive front candidates and references have different sizes")

    derived = tuple(_candidate_reference(candidate) for candidate in candidates)
    for observed, expected in zip(references, derived, strict=True):
        _reference_record(observed)
        if observed != expected:
            raise ValueError("archive front candidate does not match its reference")
    if references != tuple(sorted(references, key=_reference_key)):
        raise ValueError("archive front is not in its canonical archive order")
    if len({reference.candidate_id for reference in references}) != len(references):
        raise ValueError("archive front contains duplicate candidate IDs")
    if len({reference.configuration_hash for reference in references}) != len(
        references
    ):
        raise ValueError("archive front contains duplicate configurations")

    history_by_id = {
        candidate.candidate_id: _candidate_reference(candidate)
        for candidate in state.candidates
    }
    for reference in references:
        if history_by_id.get(reference.candidate_id) != reference:
            raise ValueError("archive front contains a foreign or stale candidate")
    return candidates, references


class ArchiveReservoirCrowdingKind(str, Enum):
    """Whether a replayed crowding score is applicable and finite."""

    NOT_APPLICABLE = "not_applicable"
    BOUNDARY = "boundary"
    FINITE = "finite"


@dataclass(frozen=True, slots=True)
class ArchiveReservoirRankedCandidate:
    """One fully ordered, archive-authenticated reservoir candidate."""

    reference: ParetoCandidateRef
    nondomination_rank: int
    crowding_kind: ArchiveReservoirCrowdingKind
    crowding_distance_hex: str | None
    objective_worst_dense_rank: int
    objective_dense_rank_sum: int
    proposal_sequence: int

    def __post_init__(self) -> None:
        _reference_record(self.reference)
        for name in (
            "nondomination_rank",
            "objective_worst_dense_rank",
            "objective_dense_rank_sum",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.proposal_sequence) is not int or self.proposal_sequence < 0:
            raise ValueError("proposal_sequence must be a non-negative exact integer")
        if type(self.crowding_kind) is not ArchiveReservoirCrowdingKind:
            raise TypeError("crowding_kind must be an ArchiveReservoirCrowdingKind")
        if self.crowding_kind is ArchiveReservoirCrowdingKind.FINITE:
            if type(self.crowding_distance_hex) is not str:
                raise TypeError("finite crowding distance requires a hexadecimal float")
            try:
                distance = float.fromhex(self.crowding_distance_hex)
            except ValueError as exc:
                raise ValueError(
                    "crowding_distance_hex is not a hexadecimal float"
                ) from exc
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    "finite crowding distance must be finite and non-negative"
                )
            if distance.hex() != self.crowding_distance_hex:
                raise ValueError("crowding_distance_hex must be canonical")
        elif self.crowding_distance_hex is not None:
            raise ValueError("non-finite crowding kinds cannot carry a distance")

    def to_trace_record(self) -> dict[str, object]:
        ArchiveReservoirRankedCandidate.__post_init__(self)
        return {
            "candidate": _reference_record(self.reference),
            "nondomination_rank": self.nondomination_rank,
            "crowding_kind": self.crowding_kind.value,
            "crowding_distance_hex": self.crowding_distance_hex,
            "objective_worst_dense_rank": self.objective_worst_dense_rank,
            "objective_dense_rank_sum": self.objective_dense_rank_sum,
            "proposal_sequence": self.proposal_sequence,
        }


def _candidate_objective_map(
    candidate: EvolutionCandidate,
    objectives: tuple[ObjectiveSpec, ...],
) -> dict[str, float]:
    if type(candidate.objectives) is not tuple:
        raise ProblemContractError("candidate objectives must be an exact tuple")
    raw: dict[str, object] = {}
    for item in candidate.objectives:
        if type(item) is not tuple or len(item) != 2:
            raise ProblemContractError(
                "candidate objectives must contain exact (name, value) tuples"
            )
        name, value = item
        if type(name) is not str:
            raise ProblemContractError("candidate objective names must be strings")
        if name in raw:
            raise ProblemContractError(f"duplicate candidate objective {name!r}")
        raw[name] = value
    return normalize_objective_values(raw, objectives)


def _validated_eligible_history(
    state: OptimizerState,
) -> tuple[
    tuple[EvolutionCandidate, ...],
    tuple[dict[str, float], ...],
    tuple[ObjectiveSpec, ...],
]:
    """Rejoin complete candidate history to the snapshot's consideration ledger."""

    _validated_front(state)
    snapshot = state.archive
    if not snapshot.objective_pareto_relation:
        raise ValueError(
            "ranked parent reservoirs require objective Pareto archive semantics"
        )
    objectives = snapshot.objectives
    if type(objectives) is not tuple or any(
        type(objective) is not ObjectiveSpec for objective in objectives
    ):
        raise TypeError("archive objectives must contain exact ObjectiveSpec values")
    validate_objective_specs(objectives)
    if type(snapshot.evidence_admission_policy) is not EvidenceAdmissionPolicy:
        raise TypeError(
            "archive evidence_admission_policy must be an EvidenceAdmissionPolicy"
        )
    if (
        type(snapshot.consideration_count) is not int
        or snapshot.consideration_count < 0
    ):
        raise ValueError("archive consideration_count must be non-negative")
    if (
        type(snapshot.eligible_configuration_count) is not int
        or snapshot.eligible_configuration_count < 0
    ):
        raise ValueError("archive eligible_configuration_count must be non-negative")
    decisions = snapshot.decisions
    if type(decisions) is not tuple:
        raise TypeError("archive decisions must be an exact tuple")

    grouped: dict[int, list[ParetoDecision]] = {}
    previous_consideration = 0
    for expected_sequence, decision in enumerate(decisions, start=1):
        if type(decision) is not ParetoDecision:
            raise TypeError(
                "archive decisions must contain exact ParetoDecision values"
            )
        ParetoDecision.__post_init__(decision)
        if decision.decision_sequence != expected_sequence:
            raise ValueError(
                "archive decision sequences must be contiguous and ordered"
            )
        if decision.consideration_sequence < previous_consideration:
            raise ValueError("archive consideration decisions must remain contiguous")
        if decision.consideration_sequence > snapshot.consideration_count:
            raise ValueError("archive decision exceeds consideration_count")
        previous_consideration = decision.consideration_sequence
        grouped.setdefault(decision.consideration_sequence, []).append(decision)
    if tuple(grouped) != tuple(range(1, snapshot.consideration_count + 1)):
        raise ValueError("archive consideration ledger is incomplete")

    history_by_id: dict[str, EvolutionCandidate] = {}
    history_references: dict[str, ParetoCandidateRef] = {}
    for candidate in state.candidates:
        reference = _candidate_reference(candidate)
        candidate_id = candidate.candidate_id.value
        history_by_id[candidate_id] = candidate
        history_references[candidate_id] = reference
        if candidate.generation > state.generation:
            raise ValueError("candidate history cannot be newer than optimizer state")
    primary_ids = tuple(
        grouped[index][0].candidate.candidate_id.value
        for index in range(1, snapshot.consideration_count + 1)
    )
    if len(set(primary_ids)) != len(primary_ids):
        raise ValueError("archive consideration ledger repeats a candidate ID")
    if set(primary_ids) != set(history_by_id):
        raise ValueError("candidate history differs from the complete archive ledger")

    gate_reason_order = (
        ("valid", ParetoDecisionReason.REJECTED_INVALID),
        (
            "operator_compliant",
            ParetoDecisionReason.REJECTED_OPERATOR_NONCOMPLIANT,
        ),
        (
            "evidence_compliant",
            ParetoDecisionReason.REJECTED_EVIDENCE_NONCOMPLIANT,
        ),
    )
    eligible_reasons = {
        ParetoDecisionReason.ADMITTED_NONDOMINATED,
        ParetoDecisionReason.ADMITTED_TIE_BREAK_REPLACEMENT,
        ParetoDecisionReason.REJECTED_DOMINATED,
        ParetoDecisionReason.REJECTED_OBJECTIVE_TIE,
    }
    seen_configurations: set[str] = set()
    eligible_candidates: list[EvolutionCandidate] = []
    objective_maps: list[dict[str, float]] = []
    for consideration in range(1, snapshot.consideration_count + 1):
        primary = grouped[consideration][0]
        candidate_id = primary.candidate.candidate_id.value
        candidate = history_by_id[candidate_id]
        if primary.candidate != history_references[candidate_id]:
            raise ValueError("archive consideration references a foreign candidate")

        expected_gate_reasons = tuple(
            reason
            for attribute, reason in gate_reason_order
            if not getattr(candidate, attribute)
            and (
                attribute != "evidence_compliant"
                or snapshot.evidence_admission_policy
                is EvidenceAdmissionPolicy.REQUIRE_COMPLIANT
            )
        )
        if expected_gate_reasons:
            if primary.reasons != expected_gate_reasons:
                raise ValueError(
                    "archive gate decision differs from candidate evidence"
                )
            continue
        try:
            objective_map = _candidate_objective_map(candidate, objectives)
        except (OverflowError, ProblemContractError, TypeError, ValueError):
            if primary.reasons != (ParetoDecisionReason.REJECTED_OBJECTIVE_CONTRACT,):
                raise ValueError(
                    "archive objective-contract decision differs from candidate evidence"
                )
            continue

        configuration_hash = candidate.occurrence.configuration_hash
        if configuration_hash in seen_configurations:
            if primary.reasons != (
                ParetoDecisionReason.REJECTED_DUPLICATE_CONFIGURATION,
            ):
                raise ValueError(
                    "archive duplicate decision differs from consideration history"
                )
            continue
        seen_configurations.add(configuration_hash)
        if len(primary.reasons) != 1 or primary.reasons[0] not in eligible_reasons:
            raise ValueError("eligible archive candidate has a foreign decision reason")
        eligible_candidates.append(candidate)
        objective_maps.append(objective_map)

    if len(eligible_candidates) != snapshot.eligible_configuration_count:
        raise ValueError("eligible history differs from archive configuration count")
    return tuple(eligible_candidates), tuple(objective_maps), objectives


def _nondomination_ranks(
    objective_maps: tuple[dict[str, float], ...],
    objectives: tuple[ObjectiveSpec, ...],
) -> tuple[int, ...]:
    cardinality = len(objective_maps)
    dominated_sets: list[set[int]] = [set() for _ in range(cardinality)]
    dominator_counts = [0] * cardinality
    for left in range(cardinality):
        for right in range(left + 1, cardinality):
            if dominates(objective_maps[left], objective_maps[right], objectives):
                dominated_sets[left].add(right)
                dominator_counts[right] += 1
            elif dominates(objective_maps[right], objective_maps[left], objectives):
                dominated_sets[right].add(left)
                dominator_counts[left] += 1

    ranks = [0] * cardinality
    current = [index for index, count in enumerate(dominator_counts) if count == 0]
    rank = 1
    ranked_count = 0
    while current:
        following: list[int] = []
        for index in current:
            ranks[index] = rank
            ranked_count += 1
            for dominated in dominated_sets[index]:
                dominator_counts[dominated] -= 1
                if dominator_counts[dominated] == 0:
                    following.append(dominated)
        current = following
        rank += 1
    if ranked_count != cardinality:
        raise RuntimeError("nondomination ranking did not cover eligible history")
    return tuple(ranks)


def _objective_dense_ranks(
    candidates: tuple[EvolutionCandidate, ...],
    objective_maps: tuple[dict[str, float], ...],
    objectives: tuple[ObjectiveSpec, ...],
) -> tuple[tuple[int, ...], ...]:
    ranks = [[0] * len(objectives) for _ in candidates]
    references = tuple(_candidate_reference(candidate) for candidate in candidates)
    for objective_index, objective in enumerate(objectives):
        order = sorted(
            range(len(candidates)),
            key=lambda index: (
                -objective_maps[index][objective.name]
                if objective.goal == "max"
                else objective_maps[index][objective.name],
                _reference_key(references[index]),
            ),
        )
        dense_rank = 1
        for position, index in enumerate(order):
            if position and (
                objective_maps[index][objective.name]
                != objective_maps[order[position - 1]][objective.name]
            ):
                dense_rank += 1
            ranks[index][objective_index] = dense_rank
    return tuple(tuple(value) for value in ranks)


def _crowding_values(
    candidates: tuple[EvolutionCandidate, ...],
    objective_maps: tuple[dict[str, float], ...],
    objectives: tuple[ObjectiveSpec, ...],
    nondomination_ranks: tuple[int, ...],
) -> tuple[tuple[ArchiveReservoirCrowdingKind, float | None], ...]:
    result = [(ArchiveReservoirCrowdingKind.NOT_APPLICABLE, None) for _ in candidates]
    for rank in sorted(set(nondomination_ranks)):
        front = tuple(
            index
            for index, candidate_rank in enumerate(nondomination_ranks)
            if candidate_rank == rank
        )
        if len(objectives) < 2 or len(front) < 3:
            continue
        boundary: set[int] = set()
        distances = {index: 0.0 for index in front}
        varying_objective = False
        for objective in objectives:
            values = {index: objective_maps[index][objective.name] for index in front}
            distinct_values = sorted(set(values.values()))
            if len(distinct_values) < 2:
                continue
            varying_objective = True
            lower = distinct_values[0]
            upper = distinct_values[-1]
            scale = upper - lower
            boundary.update(index for index in front if values[index] in {lower, upper})
            position = {value: index for index, value in enumerate(distinct_values)}
            for index in front:
                value_position = position[values[index]]
                if value_position in (0, len(distinct_values) - 1):
                    continue
                gap = (
                    distinct_values[value_position + 1]
                    - distinct_values[value_position - 1]
                ) / scale
                distances[index] += gap
        if not varying_objective:
            continue
        for index in front:
            result[index] = (
                (ArchiveReservoirCrowdingKind.BOUNDARY, None)
                if index in boundary
                else (ArchiveReservoirCrowdingKind.FINITE, distances[index])
            )
    return tuple(result)


def _reservoir_member_key(
    member: ArchiveReservoirRankedCandidate,
) -> tuple[object, ...]:
    ArchiveReservoirRankedCandidate.__post_init__(member)
    crowding_order = {
        ArchiveReservoirCrowdingKind.BOUNDARY: 0,
        ArchiveReservoirCrowdingKind.FINITE: 1,
        ArchiveReservoirCrowdingKind.NOT_APPLICABLE: 2,
    }[member.crowding_kind]
    crowding_distance = (
        0.0
        if member.crowding_distance_hex is None
        else float.fromhex(member.crowding_distance_hex)
    )
    return (
        member.nondomination_rank,
        crowding_order,
        -crowding_distance,
        member.objective_worst_dense_rank,
        member.objective_dense_rank_sum,
        -member.proposal_sequence,
        *_reference_key(member.reference),
    )


def _ranked_eligible_history(
    state: OptimizerState,
) -> tuple[tuple[EvolutionCandidate, ...], tuple[ArchiveReservoirRankedCandidate, ...]]:
    candidates, objective_maps, objectives = _validated_eligible_history(state)
    if not candidates:
        raise ValueError("ranked parent selection requires eligible candidate history")
    nondomination_ranks = _nondomination_ranks(objective_maps, objectives)
    dense_ranks = _objective_dense_ranks(candidates, objective_maps, objectives)
    crowding = _crowding_values(
        candidates,
        objective_maps,
        objectives,
        nondomination_ranks,
    )
    members = tuple(
        ArchiveReservoirRankedCandidate(
            reference=_candidate_reference(candidate),
            nondomination_rank=nondomination_ranks[index],
            crowding_kind=crowding[index][0],
            crowding_distance_hex=(
                None if crowding[index][1] is None else crowding[index][1].hex()
            ),
            objective_worst_dense_rank=max(dense_ranks[index]),
            objective_dense_rank_sum=sum(dense_ranks[index]),
            proposal_sequence=candidate.occurrence.proposal_sequence,
        )
        for index, candidate in enumerate(candidates)
    )
    ranked_members = tuple(sorted(members, key=_reservoir_member_key))
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ranked_candidates = tuple(
        candidates_by_id[member.reference.candidate_id] for member in ranked_members
    )
    return ranked_candidates, ranked_members


def _eligible_ranking_sha256(
    ranking: tuple[ArchiveReservoirRankedCandidate, ...],
) -> str:
    if type(ranking) is not tuple or not ranking:
        raise ValueError("eligible ranking must be a non-empty exact tuple")
    return hashlib.sha256(
        _RESERVOIR_RANKING_DOMAIN
        + _canonical_json([member.to_trace_record() for member in ranking])
    ).hexdigest()


def _reservoir_sha256(reservoir: tuple[ParetoCandidateRef, ...]) -> str:
    if type(reservoir) is not tuple or not reservoir:
        raise ValueError("parent reservoir must be a non-empty exact tuple")
    return hashlib.sha256(
        _RESERVOIR_DOMAIN
        + _canonical_json([_reference_record(reference) for reference in reservoir])
    ).hexdigest()


def _reservoir_rotation_anchor(
    *,
    task_sha256: str,
    archive_snapshot_hash: str,
    eligible_ranking_sha256: str,
    reservoir_sha256: str,
    cardinality: int,
) -> int:
    require_sha256(task_sha256, "task_sha256")
    require_sha256(archive_snapshot_hash, "archive_snapshot_hash")
    require_sha256(eligible_ranking_sha256, "eligible_ranking_sha256")
    require_sha256(reservoir_sha256, "reservoir_sha256")
    if type(cardinality) is not int or cardinality <= 0:
        raise ValueError("parent reservoir cardinality must be positive")
    digest = hashlib.sha256(
        _RESERVOIR_ROTATION_DOMAIN
        + bytes.fromhex(task_sha256)
        + bytes.fromhex(archive_snapshot_hash)
        + bytes.fromhex(eligible_ranking_sha256)
        + bytes.fromhex(reservoir_sha256)
    ).digest()
    return int.from_bytes(digest, "big", signed=False) % cardinality


@dataclass(frozen=True, slots=True)
class ArchiveEliteParentSelectionReceipt:
    """Replayable binding of one parent choice to the complete archive front."""

    task_sha256: str
    optimizer_generation: int
    archive_snapshot_hash: str
    eligible_front: tuple[ParetoCandidateRef, ...]
    requested_parent_count: int
    rotation_index: int
    rotation_anchor: int
    selected_ordinals: tuple[int, ...]
    selected_parents: tuple[ParetoCandidateRef, ...]
    schema_version: int = field(init=False, default=_SCHEMA_VERSION)
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    policy_definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )
    eligible_front_sha256: str = field(init=False, default="")
    receipt_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if (
            self.schema_version != _SCHEMA_VERSION
            or self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.policy_definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("archive elite receipt uses a foreign policy identity")
        require_sha256(self.task_sha256, "task_sha256")
        require_sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        if type(self.optimizer_generation) is not int or self.optimizer_generation < 0:
            raise ValueError("optimizer_generation must be non-negative")
        if type(self.eligible_front) is not tuple or not self.eligible_front:
            raise ValueError("eligible_front must be a non-empty exact tuple")
        for reference in self.eligible_front:
            _reference_record(reference)
        if self.eligible_front != tuple(
            sorted(self.eligible_front, key=_reference_key)
        ):
            raise ValueError("eligible_front must preserve canonical archive order")
        if len({value.candidate_id for value in self.eligible_front}) != len(
            self.eligible_front
        ):
            raise ValueError("eligible_front contains duplicate candidate IDs")
        if len({value.configuration_hash for value in self.eligible_front}) != len(
            self.eligible_front
        ):
            raise ValueError("eligible_front contains duplicate configurations")

        cardinality = len(self.eligible_front)
        if (
            type(self.requested_parent_count) is not int
            or not 1 <= self.requested_parent_count <= cardinality
        ):
            raise ValueError("requested_parent_count must lie within the front")
        if (
            type(self.rotation_index) is not int
            or not 0 <= self.rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")

        computed_front_sha256 = _eligible_front_sha256(self.eligible_front)
        if self.eligible_front_sha256 not in ("", computed_front_sha256):
            raise ValueError("eligible_front_sha256 does not identify eligible_front")
        object.__setattr__(
            self,
            "eligible_front_sha256",
            computed_front_sha256,
        )
        expected_anchor = _rotation_anchor(
            task_sha256=self.task_sha256,
            archive_snapshot_hash=self.archive_snapshot_hash,
            eligible_front_sha256=computed_front_sha256,
            cardinality=cardinality,
        )
        if (
            type(self.rotation_anchor) is not int
            or self.rotation_anchor != expected_anchor
        ):
            raise ValueError("rotation_anchor does not replay the task-keyed policy")
        expected_ordinals = tuple(
            (expected_anchor + self.rotation_index + offset) % cardinality
            for offset in range(self.requested_parent_count)
        )
        if type(self.selected_ordinals) is not tuple:
            raise TypeError("selected_ordinals must be an exact tuple")
        if any(type(ordinal) is not int for ordinal in self.selected_ordinals):
            raise TypeError("selected_ordinals must contain exact ints")
        if type(self.selected_parents) is not tuple:
            raise TypeError("selected_parents must be an exact tuple")
        for reference in self.selected_parents:
            _reference_record(reference)
        if self.selected_ordinals != expected_ordinals:
            raise ValueError("selected_ordinals do not replay the cyclic rotation")
        if self.selected_parents != tuple(
            self.eligible_front[ordinal] for ordinal in expected_ordinals
        ):
            raise ValueError("selected_parents do not match selected_ordinals")
        if len(set(self.selected_ordinals)) != len(self.selected_ordinals):
            raise ValueError("parent selection must be without replacement")

        computed_receipt_sha256 = hashlib.sha256(
            _RECEIPT_DOMAIN + _canonical_json(self._record_without_hash())
        ).hexdigest()
        if self.receipt_sha256 not in ("", computed_receipt_sha256):
            raise ValueError("receipt_sha256 does not verify")
        object.__setattr__(self, "receipt_sha256", computed_receipt_sha256)

    def _record_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "task_sha256": self.task_sha256,
            "optimizer_generation": self.optimizer_generation,
            "archive_snapshot_hash": self.archive_snapshot_hash,
            "eligible_front": [
                _reference_record(reference) for reference in self.eligible_front
            ],
            "eligible_front_sha256": self.eligible_front_sha256,
            "requested_parent_count": self.requested_parent_count,
            "rotation_index": self.rotation_index,
            "rotation_anchor": self.rotation_anchor,
            "selected_ordinals": list(self.selected_ordinals),
            "selected_parents": [
                _reference_record(reference) for reference in self.selected_parents
            ],
        }

    def revalidate(self) -> None:
        if type(self) is not ArchiveEliteParentSelectionReceipt:
            raise TypeError(
                "receipt must be an exact ArchiveEliteParentSelectionReceipt"
            )
        ArchiveEliteParentSelectionReceipt.__post_init__(self)

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._record_without_hash(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ArchiveEliteParentSelection:
    """Selected immutable parent values plus their audit receipt."""

    parents: tuple[EvolutionCandidate, ...]
    receipt: ArchiveEliteParentSelectionReceipt

    def __post_init__(self) -> None:
        if type(self.parents) is not tuple or not self.parents:
            raise ValueError("parents must be a non-empty exact tuple")
        if any(type(parent) is not EvolutionCandidate for parent in self.parents):
            raise TypeError("parents must contain exact EvolutionCandidate values")
        for parent in self.parents:
            EvolutionCandidate.__post_init__(parent)
        if type(self.receipt) is not ArchiveEliteParentSelectionReceipt:
            raise TypeError("receipt must be an exact archive elite receipt")
        self.receipt.revalidate()
        if len(self.parents) != self.receipt.requested_parent_count:
            raise ValueError("selected parent count differs from its receipt")
        observed = tuple(_candidate_reference(parent) for parent in self.parents)
        if observed != self.receipt.selected_parents:
            raise ValueError("selected parent values differ from their receipt")

    def revalidate(self) -> None:
        if type(self) is not ArchiveEliteParentSelection:
            raise TypeError("selection must be an exact ArchiveEliteParentSelection")
        ArchiveEliteParentSelection.__post_init__(self)


def validate_archive_elite_parent_selection(
    state: OptimizerState,
    selection: ArchiveEliteParentSelection,
) -> None:
    """Reject a stale, foreign, or tampered selection at planner admission."""

    candidates, references = _validated_front(state)
    if type(selection) is not ArchiveEliteParentSelection:
        raise TypeError("selection must be an exact ArchiveEliteParentSelection")
    selection.revalidate()
    receipt = selection.receipt
    if receipt.optimizer_generation != state.generation:
        raise ValueError("archive elite selection is stale for this generation")
    if receipt.archive_snapshot_hash != state.archive_snapshot_hash:
        raise ValueError("archive elite selection is stale for this archive snapshot")
    if receipt.eligible_front != references:
        raise ValueError(
            "archive elite receipt does not bind the complete current front"
        )
    expected_parents = tuple(candidates[index] for index in receipt.selected_ordinals)
    if selection.parents != expected_parents:
        raise ValueError("archive elite parents are foreign to the current front")


class ArchiveEliteParentSelector(Protocol):
    """Inverted planner seam for objective-blind archive parent selection."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        parent_count: int = 1,
        rotation_index: int = 0,
    ) -> ArchiveEliteParentSelection: ...

    def to_record(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class TaskKeyedArchiveEliteParentPolicy:
    """Select distinct elite parents without reading benchmark semantics."""

    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )

    def _validate_identity(self) -> None:
        if type(self) is not TaskKeyedArchiveEliteParentPolicy:
            raise TypeError("policy must be an exact TaskKeyedArchiveEliteParentPolicy")
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("archive elite selector uses a foreign policy identity")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        parent_count: int = 1,
        rotation_index: int = 0,
    ) -> ArchiveEliteParentSelection:
        self._validate_identity()
        require_sha256(task_sha256, "task_sha256")
        require_sha256(
            expected_archive_snapshot_hash,
            "expected_archive_snapshot_hash",
        )
        candidates, references = _validated_front(state)
        if expected_archive_snapshot_hash != state.archive_snapshot_hash:
            raise ValueError("expected archive snapshot is stale")
        if type(parent_count) is not int or not 1 <= parent_count <= len(references):
            raise ValueError("parent_count must lie within the archive front")
        if (
            type(rotation_index) is not int
            or not 0 <= rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")

        front_sha256 = _eligible_front_sha256(references)
        anchor = _rotation_anchor(
            task_sha256=task_sha256,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_front_sha256=front_sha256,
            cardinality=len(references),
        )
        ordinals = tuple(
            (anchor + rotation_index + offset) % len(references)
            for offset in range(parent_count)
        )
        receipt = ArchiveEliteParentSelectionReceipt(
            task_sha256=task_sha256,
            optimizer_generation=state.generation,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_front=references,
            requested_parent_count=parent_count,
            rotation_index=rotation_index,
            rotation_anchor=anchor,
            selected_ordinals=ordinals,
            selected_parents=tuple(references[index] for index in ordinals),
        )
        selection = ArchiveEliteParentSelection(
            parents=tuple(candidates[index] for index in ordinals),
            receipt=receipt,
        )
        validate_archive_elite_parent_selection(state, selection)
        return selection

    def to_record(self) -> dict[str, object]:
        self._validate_identity()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArchiveReservoirParentSelectionReceipt:
    """Replayable binding of a capped parent choice to complete ranked history."""

    task_sha256: str
    optimizer_generation: int
    archive_snapshot_hash: str
    eligible_ranking: tuple[ArchiveReservoirRankedCandidate, ...]
    reservoir_limit: int
    reservoir: tuple[ParetoCandidateRef, ...]
    requested_parent_count: int
    returned_parent_count: int
    rotation_index: int
    rotation_anchor: int
    selected_ordinals: tuple[int, ...]
    selected_parents: tuple[ParetoCandidateRef, ...]
    schema_version: int = field(init=False, default=_RESERVOIR_SCHEMA_VERSION)
    policy_id: str = field(init=False, default=RESERVOIR_POLICY_ID)
    policy_version: int = field(init=False, default=RESERVOIR_POLICY_VERSION)
    policy_definition_sha256: str = field(
        init=False,
        default=RESERVOIR_POLICY_DEFINITION_SHA256,
    )
    eligible_ranking_sha256: str = field(init=False, default="")
    reservoir_sha256: str = field(init=False, default="")
    receipt_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if (
            self.schema_version != _RESERVOIR_SCHEMA_VERSION
            or self.policy_id != RESERVOIR_POLICY_ID
            or self.policy_version != RESERVOIR_POLICY_VERSION
            or self.policy_definition_sha256 != RESERVOIR_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("archive reservoir receipt uses a foreign policy identity")
        require_sha256(self.task_sha256, "task_sha256")
        require_sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        if type(self.optimizer_generation) is not int or self.optimizer_generation < 0:
            raise ValueError("optimizer_generation must be non-negative")
        if type(self.eligible_ranking) is not tuple or not self.eligible_ranking:
            raise ValueError("eligible_ranking must be a non-empty exact tuple")
        if any(
            type(member) is not ArchiveReservoirRankedCandidate
            for member in self.eligible_ranking
        ):
            raise TypeError("eligible_ranking must contain exact ranked candidates")
        for member in self.eligible_ranking:
            ArchiveReservoirRankedCandidate.__post_init__(member)
        if self.eligible_ranking != tuple(
            sorted(self.eligible_ranking, key=_reservoir_member_key)
        ):
            raise ValueError("eligible_ranking does not preserve policy order")
        references = tuple(member.reference for member in self.eligible_ranking)
        if len({value.candidate_id for value in references}) != len(references):
            raise ValueError("eligible_ranking contains duplicate candidate IDs")
        if len({value.configuration_hash for value in references}) != len(references):
            raise ValueError("eligible_ranking contains duplicate configurations")
        observed_ranks = {member.nondomination_rank for member in self.eligible_ranking}
        if observed_ranks != set(range(1, max(observed_ranks) + 1)):
            raise ValueError("eligible_ranking has non-contiguous nondomination ranks")

        computed_ranking_sha256 = _eligible_ranking_sha256(self.eligible_ranking)
        if self.eligible_ranking_sha256 not in ("", computed_ranking_sha256):
            raise ValueError("eligible_ranking_sha256 does not identify ranking")
        object.__setattr__(
            self,
            "eligible_ranking_sha256",
            computed_ranking_sha256,
        )
        if type(self.reservoir_limit) is not int or self.reservoir_limit <= 0:
            raise ValueError("reservoir_limit must be a positive exact integer")
        expected_reservoir = references[: min(self.reservoir_limit, len(references))]
        if type(self.reservoir) is not tuple:
            raise TypeError("reservoir must be an exact tuple")
        for reference in self.reservoir:
            _reference_record(reference)
        if self.reservoir != expected_reservoir:
            raise ValueError("reservoir is not the ranked top-k eligible prefix")
        computed_reservoir_sha256 = _reservoir_sha256(self.reservoir)
        if self.reservoir_sha256 not in ("", computed_reservoir_sha256):
            raise ValueError("reservoir_sha256 does not identify reservoir")
        object.__setattr__(self, "reservoir_sha256", computed_reservoir_sha256)

        if (
            type(self.requested_parent_count) is not int
            or self.requested_parent_count <= 0
        ):
            raise ValueError("requested_parent_count must be a positive exact integer")
        expected_returned = min(self.requested_parent_count, len(self.reservoir))
        if (
            type(self.returned_parent_count) is not int
            or self.returned_parent_count != expected_returned
        ):
            raise ValueError("returned_parent_count does not apply the singleton cap")
        if (
            type(self.rotation_index) is not int
            or not 0 <= self.rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")
        expected_anchor = _reservoir_rotation_anchor(
            task_sha256=self.task_sha256,
            archive_snapshot_hash=self.archive_snapshot_hash,
            eligible_ranking_sha256=computed_ranking_sha256,
            reservoir_sha256=computed_reservoir_sha256,
            cardinality=len(self.reservoir),
        )
        if (
            type(self.rotation_anchor) is not int
            or self.rotation_anchor != expected_anchor
        ):
            raise ValueError("rotation_anchor does not replay the reservoir policy")
        expected_ordinals = tuple(
            (expected_anchor + self.rotation_index + offset) % len(self.reservoir)
            for offset in range(expected_returned)
        )
        if type(self.selected_ordinals) is not tuple or any(
            type(ordinal) is not int for ordinal in self.selected_ordinals
        ):
            raise TypeError("selected_ordinals must be an exact tuple of exact ints")
        if self.selected_ordinals != expected_ordinals:
            raise ValueError("selected_ordinals do not replay the cyclic rotation")
        if len(set(self.selected_ordinals)) != len(self.selected_ordinals):
            raise ValueError("reservoir parent selection must be without replacement")
        if type(self.selected_parents) is not tuple:
            raise TypeError("selected_parents must be an exact tuple")
        for reference in self.selected_parents:
            _reference_record(reference)
        if self.selected_parents != tuple(
            self.reservoir[ordinal] for ordinal in expected_ordinals
        ):
            raise ValueError("selected_parents do not match selected_ordinals")

        computed_receipt_sha256 = hashlib.sha256(
            _RESERVOIR_RECEIPT_DOMAIN + _canonical_json(self._record_without_hash())
        ).hexdigest()
        if self.receipt_sha256 not in ("", computed_receipt_sha256):
            raise ValueError("receipt_sha256 does not verify")
        object.__setattr__(self, "receipt_sha256", computed_receipt_sha256)

    def _record_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "task_sha256": self.task_sha256,
            "optimizer_generation": self.optimizer_generation,
            "archive_snapshot_hash": self.archive_snapshot_hash,
            "eligible_ranking": [
                member.to_trace_record() for member in self.eligible_ranking
            ],
            "eligible_ranking_sha256": self.eligible_ranking_sha256,
            "reservoir_limit": self.reservoir_limit,
            "reservoir": [_reference_record(reference) for reference in self.reservoir],
            "reservoir_sha256": self.reservoir_sha256,
            "requested_parent_count": self.requested_parent_count,
            "returned_parent_count": self.returned_parent_count,
            "rotation_index": self.rotation_index,
            "rotation_anchor": self.rotation_anchor,
            "selected_ordinals": list(self.selected_ordinals),
            "selected_parents": [
                _reference_record(reference) for reference in self.selected_parents
            ],
        }

    def revalidate(self) -> None:
        if type(self) is not ArchiveReservoirParentSelectionReceipt:
            raise TypeError(
                "receipt must be an exact ArchiveReservoirParentSelectionReceipt"
            )
        ArchiveReservoirParentSelectionReceipt.__post_init__(self)

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._record_without_hash(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ArchiveReservoirParentSelection:
    """Up-to-K distinct parents plus their ranked-reservoir receipt."""

    parents: tuple[EvolutionCandidate, ...]
    receipt: ArchiveReservoirParentSelectionReceipt

    def __post_init__(self) -> None:
        if type(self.parents) is not tuple or not self.parents:
            raise ValueError("parents must be a non-empty exact tuple")
        if any(type(parent) is not EvolutionCandidate for parent in self.parents):
            raise TypeError("parents must contain exact EvolutionCandidate values")
        for parent in self.parents:
            EvolutionCandidate.__post_init__(parent)
        if type(self.receipt) is not ArchiveReservoirParentSelectionReceipt:
            raise TypeError("receipt must be an exact archive reservoir receipt")
        self.receipt.revalidate()
        if len(self.parents) != self.receipt.returned_parent_count:
            raise ValueError("selected parent count differs from its receipt")
        if len({parent.candidate_id for parent in self.parents}) != len(self.parents):
            raise ValueError("selected parent occurrences must be distinct")
        if len(
            {parent.occurrence.configuration_hash for parent in self.parents}
        ) != len(self.parents):
            raise ValueError("selected parent configurations must be distinct")
        if tuple(_candidate_reference(parent) for parent in self.parents) != (
            self.receipt.selected_parents
        ):
            raise ValueError("selected parent values differ from their receipt")

    def revalidate(self) -> None:
        if type(self) is not ArchiveReservoirParentSelection:
            raise TypeError(
                "selection must be an exact ArchiveReservoirParentSelection"
            )
        ArchiveReservoirParentSelection.__post_init__(self)


def validate_archive_reservoir_parent_selection(
    state: OptimizerState,
    selection: ArchiveReservoirParentSelection,
) -> None:
    """Reject stale, foreign, incompletely ranked, or tampered reservoir choices."""

    ranked_candidates, ranked_members = _ranked_eligible_history(state)
    if type(selection) is not ArchiveReservoirParentSelection:
        raise TypeError("selection must be an exact ArchiveReservoirParentSelection")
    selection.revalidate()
    receipt = selection.receipt
    if receipt.optimizer_generation != state.generation:
        raise ValueError("archive reservoir selection is stale for this generation")
    if receipt.archive_snapshot_hash != state.archive_snapshot_hash:
        raise ValueError("archive reservoir selection is stale for this snapshot")
    if receipt.eligible_ranking != ranked_members:
        raise ValueError("receipt does not bind the complete current eligible ranking")
    expected_reservoir_members = ranked_members[
        : min(receipt.reservoir_limit, len(ranked_members))
    ]
    if receipt.reservoir != tuple(
        member.reference for member in expected_reservoir_members
    ):
        raise ValueError("receipt reservoir differs from the current ranked top-k")
    expected_parents = tuple(
        ranked_candidates[ordinal] for ordinal in receipt.selected_ordinals
    )
    if selection.parents != expected_parents:
        raise ValueError("archive reservoir parents are foreign to current history")


class ArchiveReservoirParentSelector(Protocol):
    """Inverted planner seam for singleton-safe ranked parent reservoirs."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        reservoir_limit: int,
        parent_count: int = 1,
        rotation_index: int = 0,
    ) -> ArchiveReservoirParentSelection: ...

    def to_record(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class TaskKeyedArchiveReservoirParentPolicy:
    """Rank eligible history generically, then rotate over its top-k reservoir."""

    policy_id: str = field(init=False, default=RESERVOIR_POLICY_ID)
    policy_version: int = field(init=False, default=RESERVOIR_POLICY_VERSION)
    definition_sha256: str = field(
        init=False,
        default=RESERVOIR_POLICY_DEFINITION_SHA256,
    )

    def _validate_identity(self) -> None:
        if type(self) is not TaskKeyedArchiveReservoirParentPolicy:
            raise TypeError(
                "policy must be an exact TaskKeyedArchiveReservoirParentPolicy"
            )
        if (
            self.policy_id != RESERVOIR_POLICY_ID
            or self.policy_version != RESERVOIR_POLICY_VERSION
            or self.definition_sha256 != RESERVOIR_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("archive reservoir selector uses a foreign identity")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        reservoir_limit: int,
        parent_count: int = 1,
        rotation_index: int = 0,
    ) -> ArchiveReservoirParentSelection:
        self._validate_identity()
        require_sha256(task_sha256, "task_sha256")
        require_sha256(
            expected_archive_snapshot_hash,
            "expected_archive_snapshot_hash",
        )
        if type(reservoir_limit) is not int or reservoir_limit <= 0:
            raise ValueError("reservoir_limit must be a positive exact integer")
        if type(parent_count) is not int or parent_count <= 0:
            raise ValueError("parent_count must be a positive exact integer")
        if (
            type(rotation_index) is not int
            or not 0 <= rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")
        ranked_candidates, ranking = _ranked_eligible_history(state)
        if expected_archive_snapshot_hash != state.archive_snapshot_hash:
            raise ValueError("expected archive snapshot is stale")

        reservoir_cardinality = min(reservoir_limit, len(ranking))
        reservoir_candidates = ranked_candidates[:reservoir_cardinality]
        reservoir = tuple(
            member.reference for member in ranking[:reservoir_cardinality]
        )
        ranking_sha256 = _eligible_ranking_sha256(ranking)
        reservoir_hash = _reservoir_sha256(reservoir)
        anchor = _reservoir_rotation_anchor(
            task_sha256=task_sha256,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_ranking_sha256=ranking_sha256,
            reservoir_sha256=reservoir_hash,
            cardinality=reservoir_cardinality,
        )
        returned_parent_count = min(parent_count, reservoir_cardinality)
        ordinals = tuple(
            (anchor + rotation_index + offset) % reservoir_cardinality
            for offset in range(returned_parent_count)
        )
        receipt = ArchiveReservoirParentSelectionReceipt(
            task_sha256=task_sha256,
            optimizer_generation=state.generation,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_ranking=ranking,
            reservoir_limit=reservoir_limit,
            reservoir=reservoir,
            requested_parent_count=parent_count,
            returned_parent_count=returned_parent_count,
            rotation_index=rotation_index,
            rotation_anchor=anchor,
            selected_ordinals=ordinals,
            selected_parents=tuple(reservoir[index] for index in ordinals),
        )
        selection = ArchiveReservoirParentSelection(
            parents=tuple(reservoir_candidates[index] for index in ordinals),
            receipt=receipt,
        )
        validate_archive_reservoir_parent_selection(state, selection)
        return selection

    def to_record(self) -> dict[str, object]:
        self._validate_identity()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


__all__ = [
    "ArchiveEliteParentSelection",
    "ArchiveEliteParentSelectionReceipt",
    "ArchiveEliteParentSelector",
    "ArchiveReservoirCrowdingKind",
    "ArchiveReservoirParentSelection",
    "ArchiveReservoirParentSelectionReceipt",
    "ArchiveReservoirParentSelector",
    "ArchiveReservoirRankedCandidate",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "RESERVOIR_POLICY_DEFINITION_SHA256",
    "RESERVOIR_POLICY_ID",
    "RESERVOIR_POLICY_VERSION",
    "TaskKeyedArchiveEliteParentPolicy",
    "TaskKeyedArchiveReservoirParentPolicy",
    "validate_archive_elite_parent_selection",
    "validate_archive_reservoir_parent_selection",
]
