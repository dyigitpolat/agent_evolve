"""Outcome-blind model anchors with one full-abstention frontier probe.

The policy is intentionally small.  It keeps the model-ranked top three and,
when the model fully abstains on every objective for an outside candidate,
spends the fourth evaluation on the most structurally novel such candidate.
Partial abstention is not an exploration trigger.  In the absence of a full
abstention the policy evaluates the model top four.

Evaluation allocation is a portfolio-learning decision, while pairwise patch
compatibility and minimum family coverage are mating/recombination concerns.
Historical requests conflate those stages, so this module makes their removal
an explicit, authenticated projection.  Phenotype uniqueness and bounded
memory-dose safety remain hard evaluation constraints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from typing import ClassVar

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlateMember,
    SlateAllocationRequest,
    assess_allocated_slate_memory_dose,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseAssessment,
)


POLICY_ID = "model_anchored_full_abstention_frontier_probe"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:model-anchored-full-abstention-frontier-probe:v1;"
    b"slate-size=8;portfolio-size=4;model-anchors=3;"
    b"probe-trigger=full-vector-unknown-outside-anchor-prefix;"
    b"partial-abstention-trigger=false;"
    b"probe-rank=mean-archive-novelty-and-structural-coverage;"
    b"no-probe=model-top-four;phenotype-uniqueness=hard;"
    b"bounded-memory-dose=hard;assigned-cards-without-dose=advisory;"
    b"mating-constraints=explicitly-projected-from-evaluation;"
    b"ties=model-rank,option-id;outcome-blind=true;"
    b"benchmark-specific-parameters=none"
).hexdigest()

PROJECTION_POLICY_ID = "evaluation_allocation_constraint_projection"
PROJECTION_POLICY_VERSION = 1
PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:evaluation-allocation-constraint-projection:v1;"
    b"remove=pairwise-disjoint-option-pairs,min-distinct-families;"
    b"preserve=slate,portfolio-size,objectives,cards,calibration,"
    b"bounded-memory-dose;scope=evaluation-only;"
    b"downstream-mating-compatibility-unchanged=true"
).hexdigest()

_DECISION_DOMAIN = b"agent-evolve:frontier-probe-slate-decision:v1\x00"
_PROJECTION_DOMAIN = b"agent-evolve:evaluation-constraint-projection:v1\x00"
_CONFIGURATION_DOMAIN = b"agent-evolve:frontier-probe-config:v1\x00"
_SLATE_SIZE = 8
_PORTFOLIO_SIZE = 4
_MODEL_ANCHOR_COUNT = 3


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _configuration_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "slate_size": _SLATE_SIZE,
        "portfolio_size": _PORTFOLIO_SIZE,
        "model_anchor_count": _MODEL_ANCHOR_COUNT,
        "probe_trigger": "full_vector_abstention_outside_anchor_prefix",
        "probe_structural_score": (
            "mean_archive_novelty_and_structural_coverage"
        ),
        "partial_abstention_is_probe": False,
        "hard_evaluation_constraints": [
            "distinct_option_identity",
            "distinct_phenotype_identity",
            "bounded_memory_dose_when_present",
        ],
        "projected_mating_constraints": [
            "pairwise_disjoint_option_id_pairs",
            "min_distinct_families",
        ],
        "benchmark_specific_parameters": [],
    }


CONFIGURATION_SHA256 = _hash(_CONFIGURATION_DOMAIN, _configuration_record())


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationAllocationConstraintProjection:
    """Authenticated separation of evaluation and mating constraints."""

    source_request: SlateAllocationRequest
    projected_request: SlateAllocationRequest

    policy_id: ClassVar[str] = PROJECTION_POLICY_ID
    policy_version: ClassVar[int] = PROJECTION_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = PROJECTION_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.source_request) is not SlateAllocationRequest:
            raise TypeError("source_request must be exact SlateAllocationRequest")
        if type(self.projected_request) is not SlateAllocationRequest:
            raise TypeError("projected_request must be exact SlateAllocationRequest")
        self.source_request.revalidate()
        self.projected_request.revalidate()
        expected = replace(
            self.source_request,
            pairwise_disjoint_option_id_pairs=None,
            min_distinct_families=None,
        )
        if self.projected_request != expected:
            raise ValueError(
                "projected request differs beyond mating-only constraints"
            )

    def revalidate(self) -> None:
        if type(self) is not EvaluationAllocationConstraintProjection:
            raise TypeError("projection must be exact")
        EvaluationAllocationConstraintProjection.__post_init__(self)

    @property
    def removed_pairwise_constraint(self) -> bool:
        self.revalidate()
        return self.source_request.pairwise_disjoint_option_id_pairs is not None

    @property
    def removed_pair_count(self) -> int:
        self.revalidate()
        pairs = self.source_request.pairwise_disjoint_option_id_pairs
        return 0 if pairs is None else len(pairs)

    @property
    def removed_min_distinct_families(self) -> int | None:
        self.revalidate()
        return self.source_request.min_distinct_families

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "event_type": "evaluation_allocation_constraints_projected",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "source_request_sha256": self.source_request.request_sha256,
            "projected_request_sha256": self.projected_request.request_sha256,
            "removed_pairwise_constraint": self.removed_pairwise_constraint,
            "removed_pair_count": self.removed_pair_count,
            "removed_min_distinct_families": (
                self.removed_min_distinct_families
            ),
            "preserved_memory_dose_contract_sha256": (
                None
                if self.projected_request.memory_dose_contract is None
                else self.projected_request.memory_dose_contract.contract_sha256
            ),
            "scope": (
                "evaluation_allocation_only_downstream_mating_constraints_"
                "remain_authoritative"
            ),
        }

    @property
    def projection_sha256(self) -> str:
        return _hash(_PROJECTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "projection_sha256": self.projection_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is EvaluationAllocationConstraintProjection
            and self.projection_sha256 == other.projection_sha256
        )

    __hash__ = None


def project_evaluation_allocation_request(
    request: SlateAllocationRequest,
) -> EvaluationAllocationConstraintProjection:
    """Remove only mating-specific constraints from an evaluation request."""

    if type(request) is not SlateAllocationRequest:
        raise TypeError("request must be exact SlateAllocationRequest")
    request.revalidate()
    return EvaluationAllocationConstraintProjection(
        source_request=request,
        projected_request=replace(
            request,
            pairwise_disjoint_option_id_pairs=None,
            min_distinct_families=None,
        ),
    )


class FrontierProbeSlateRole(str, Enum):
    """Engine-owned provenance for each selected evaluation."""

    MODEL_ANCHOR = "model_anchor"
    MODEL_FILL = "model_fill"
    FULL_ABSTENTION_FRONTIER_PROBE = "full_abstention_frontier_probe"
    SAFETY_RECOURSE = "safety_recourse"


@dataclass(frozen=True, slots=True)
class FrontierProbeMemberEvidence:
    """Outcome-free evidence used by the allocator for one member."""

    option_id: str
    option_identity_sha256: str
    phenotype_identity_sha256: str
    model_rank: int
    unknown_metric_ids: tuple[str, ...]
    full_vector_abstention: bool
    structural_frontier_score: float
    supported_assigned_card_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        if type(self.unknown_metric_ids) is not tuple or any(
            type(value) is not str or not value
            for value in self.unknown_metric_ids
        ):
            raise TypeError("unknown_metric_ids must be an exact string tuple")
        if self.unknown_metric_ids != tuple(
            sorted(set(self.unknown_metric_ids))
        ):
            raise ValueError("unknown_metric_ids must be unique and canonical")
        if type(self.full_vector_abstention) is not bool:
            raise TypeError("full_vector_abstention must be exact bool")
        if type(self.structural_frontier_score) is not float or not (
            0.0 <= self.structural_frontier_score <= 1.0
        ):
            raise ValueError("structural_frontier_score must lie in [0, 1]")
        if type(self.supported_assigned_card_keys) is not tuple or any(
            type(value) is not str or not value
            for value in self.supported_assigned_card_keys
        ):
            raise TypeError(
                "supported_assigned_card_keys must be an exact string tuple"
            )
        if self.supported_assigned_card_keys != tuple(
            sorted(set(self.supported_assigned_card_keys))
        ):
            raise ValueError(
                "supported_assigned_card_keys must be unique and canonical"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "model_rank": self.model_rank,
            "unknown_metric_ids": list(self.unknown_metric_ids),
            "full_vector_abstention": self.full_vector_abstention,
            "structural_frontier_score_hex": (
                self.structural_frontier_score.hex()
            ),
            "supported_assigned_card_keys": list(
                self.supported_assigned_card_keys
            ),
        }


@dataclass(frozen=True, slots=True)
class FrontierProbeAllocatedMember:
    """One selected member and its outcome-free selection role."""

    role: FrontierProbeSlateRole
    option_id: str
    option_identity_sha256: str
    model_rank: int
    full_vector_abstention: bool
    structural_frontier_score: float

    def __post_init__(self) -> None:
        if type(self.role) is not FrontierProbeSlateRole:
            raise TypeError("role must be exact FrontierProbeSlateRole")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        if type(self.full_vector_abstention) is not bool:
            raise TypeError("full_vector_abstention must be exact bool")
        if type(self.structural_frontier_score) is not float or not (
            0.0 <= self.structural_frontier_score <= 1.0
        ):
            raise ValueError("structural_frontier_score must lie in [0, 1]")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "role": self.role.value,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "full_vector_abstention": self.full_vector_abstention,
            "structural_frontier_score_hex": (
                self.structural_frontier_score.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class _FrontierProbeAssignment:
    selected: tuple[FrontierProbeAllocatedMember, ...]
    target_option_ids: tuple[str, ...]
    available_full_abstention_option_ids: tuple[str, ...]
    selected_probe_option_id: str | None
    retained_target_count: int
    ideal_target_feasible: bool
    feasible_subset_count: int
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None


def _member_evidence(
    request: SlateAllocationRequest,
    member: CalibratedSlateMember,
) -> FrontierProbeMemberEvidence:
    unknown_metric_ids = tuple(
        prediction.metric_id
        for prediction in member.predictions
        if prediction.asserted_direction is MetricEffectDirection.UNKNOWN
    )
    metric_count = len(member.predictions)
    structural_frontier_score = (
        member.structural_evidence.archive_novelty_score
        + member.structural_evidence.structural_coverage_score
    ) / 2.0
    return FrontierProbeMemberEvidence(
        option_id=member.option_id,
        option_identity_sha256=member.option_identity_sha256,
        phenotype_identity_sha256=member.phenotype_identity_sha256,
        model_rank=member.model_rank,
        unknown_metric_ids=unknown_metric_ids,
        full_vector_abstention=(len(unknown_metric_ids) == metric_count),
        structural_frontier_score=structural_frontier_score,
        supported_assigned_card_keys=tuple(
            card
            for card in request.assigned_card_keys
            if card in member.supporting_card_keys
        ),
    )


def _target(
    rows: tuple[FrontierProbeMemberEvidence, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    anchors = rows[:_MODEL_ANCHOR_COUNT]
    outside_abstentions = tuple(
        row
        for row in rows[_MODEL_ANCHOR_COUNT:]
        if row.full_vector_abstention
    )
    probe = (
        None
        if not outside_abstentions
        else min(
            outside_abstentions,
            key=lambda row: (
                -row.structural_frontier_score,
                row.model_rank,
                row.option_id,
            ),
        )
    )
    fourth = rows[_MODEL_ANCHOR_COUNT] if probe is None else probe
    return (
        tuple(row.option_id for row in anchors) + (fourth.option_id,),
        tuple(row.option_id for row in outside_abstentions),
        None if probe is None else probe.option_id,
    )


def _best_assignment(
    request: SlateAllocationRequest,
    rows: tuple[FrontierProbeMemberEvidence, ...],
) -> _FrontierProbeAssignment:
    row_by_id = {row.option_id: row for row in rows}
    target_ids, available_abstentions, probe_id = _target(rows)
    target_set = set(target_ids)
    anchor_ids = {row.option_id for row in rows[:_MODEL_ANCHOR_COUNT]}
    feasible: list[
        tuple[
            tuple[object, ...],
            tuple[CalibratedSlateMember, ...],
            PortfolioMemoryDoseAssessment | None,
            tuple[str, ...],
            int,
            int,
            int,
        ]
    ] = []
    for subset in combinations(request.slate.members, request.portfolio_size):
        phenotype_count = len(
            {member.phenotype_identity_sha256 for member in subset}
        )
        if phenotype_count != request.portfolio_size:
            continue
        dose_assessment = (
            None
            if request.memory_dose_contract is None
            else assess_allocated_slate_memory_dose(request, subset)
        )
        if dose_assessment is not None and not dose_assessment.passed:
            continue
        subset_ids = {member.option_id for member in subset}
        retained_target_count = len(target_set & subset_ids)
        retained_anchor_count = len(anchor_ids & subset_ids)
        probe_displaced = probe_id is not None and probe_id not in subset_ids
        ranks = tuple(member.model_rank for member in subset)
        structural_sum = sum(
            row_by_id[member.option_id].structural_frontier_score
            for member in subset
        )
        administered = tuple(
            sorted(
                {
                    card
                    for member in subset
                    for card in row_by_id[
                        member.option_id
                    ].supported_assigned_card_keys
                }
            )
        )
        family_count = len({member.family for member in subset})
        locus_count = len({member.locus_key for member in subset})
        key: tuple[object, ...] = (
            -retained_target_count,
            -retained_anchor_count,
            probe_displaced,
            sum(ranks),
            ranks,
            -structural_sum,
            tuple(member.option_id for member in subset),
        )
        feasible.append(
            (
                key,
                subset,
                dose_assessment,
                administered,
                family_count,
                locus_count,
                phenotype_count,
            )
        )
    if not feasible:
        raise ValueError(
            "slate has no phenotype-unique K4 satisfying bounded memory dose"
        )
    (
        _,
        subset,
        dose_assessment,
        administered,
        family_count,
        locus_count,
        phenotype_count,
    ) = min(feasible, key=lambda value: value[0])
    selected_ids = {member.option_id for member in subset}
    selected = tuple(
        FrontierProbeAllocatedMember(
            role=(
                FrontierProbeSlateRole.MODEL_ANCHOR
                if member.option_id in anchor_ids
                else (
                    FrontierProbeSlateRole.FULL_ABSTENTION_FRONTIER_PROBE
                    if member.option_id == probe_id
                    else (
                        FrontierProbeSlateRole.MODEL_FILL
                        if member.option_id in target_set
                        else FrontierProbeSlateRole.SAFETY_RECOURSE
                    )
                )
            ),
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            model_rank=member.model_rank,
            full_vector_abstention=(
                row_by_id[member.option_id].full_vector_abstention
            ),
            structural_frontier_score=(
                row_by_id[member.option_id].structural_frontier_score
            ),
        )
        for member in subset
    )
    return _FrontierProbeAssignment(
        selected=selected,
        target_option_ids=target_ids,
        available_full_abstention_option_ids=available_abstentions,
        selected_probe_option_id=(
            probe_id if probe_id in selected_ids else None
        ),
        retained_target_count=len(target_set & selected_ids),
        ideal_target_feasible=target_set == selected_ids,
        feasible_subset_count=len(feasible),
        distinct_family_count=family_count,
        distinct_locus_count=locus_count,
        distinct_phenotype_count=phenotype_count,
        administered_card_keys=administered,
        memory_dose_assessment=dose_assessment,
    )


def _validate_request(request: SlateAllocationRequest) -> None:
    if len(request.slate.members) != _SLATE_SIZE:
        raise ValueError("frontier-probe policy requires exactly eight members")
    if request.portfolio_size != _PORTFOLIO_SIZE:
        raise ValueError("frontier-probe policy requires four evaluations")
    snapshot = request.calibration_snapshot
    if snapshot is None:
        return
    if snapshot.cutoff_wave_index_exclusive > request.slate.wave_index:
        raise ValueError("calibration snapshot cutoff reaches beyond current wave")
    if any(
        observation.prediction.wave_index >= request.slate.wave_index
        for observation in snapshot.observations
    ):
        raise ValueError("request carries current-wave outcome evidence")


@dataclass(frozen=True, slots=True, eq=False)
class FrontierProbeSlateDecision:
    """Replayable receipt for one projected, outcome-blind K8-to-K4 decision."""

    projection: EvaluationAllocationConstraintProjection
    member_evidence: tuple[FrontierProbeMemberEvidence, ...]
    selected: tuple[FrontierProbeAllocatedMember, ...]
    target_option_ids: tuple[str, ...]
    available_full_abstention_option_ids: tuple[str, ...]
    selected_probe_option_id: str | None
    retained_target_count: int
    ideal_target_feasible: bool
    feasible_subset_count: int
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256
    policy_configuration_sha256: ClassVar[str] = CONFIGURATION_SHA256

    @property
    def source_request(self) -> SlateAllocationRequest:
        return self.projection.source_request

    @property
    def request(self) -> SlateAllocationRequest:
        return self.projection.projected_request

    def __post_init__(self) -> None:
        if type(self.projection) is not EvaluationAllocationConstraintProjection:
            raise TypeError("projection must be exact")
        self.projection.revalidate()
        _validate_request(self.request)
        if type(self.member_evidence) is not tuple or any(
            type(value) is not FrontierProbeMemberEvidence
            for value in self.member_evidence
        ):
            raise TypeError("member_evidence must contain exact rows")
        for value in self.member_evidence:
            value.__post_init__()
        if type(self.selected) is not tuple or any(
            type(value) is not FrontierProbeAllocatedMember
            for value in self.selected
        ):
            raise TypeError("selected must contain exact allocated members")
        for value in self.selected:
            value.__post_init__()
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not PortfolioMemoryDoseAssessment:
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()
        expected_rows = tuple(
            _member_evidence(self.request, member)
            for member in self.request.slate.members
        )
        if self.member_evidence != expected_rows:
            raise ValueError("member evidence differs from the projected request")
        expected = _best_assignment(self.request, expected_rows)
        observed_values = (
            self.selected,
            self.target_option_ids,
            self.available_full_abstention_option_ids,
            self.selected_probe_option_id,
            self.retained_target_count,
            self.ideal_target_feasible,
            self.feasible_subset_count,
            self.distinct_family_count,
            self.distinct_locus_count,
            self.distinct_phenotype_count,
            self.administered_card_keys,
            self.memory_dose_assessment,
        )
        expected_values = (
            expected.selected,
            expected.target_option_ids,
            expected.available_full_abstention_option_ids,
            expected.selected_probe_option_id,
            expected.retained_target_count,
            expected.ideal_target_feasible,
            expected.feasible_subset_count,
            expected.distinct_family_count,
            expected.distinct_locus_count,
            expected.distinct_phenotype_count,
            expected.administered_card_keys,
            expected.memory_dose_assessment,
        )
        if observed_values != expected_values:
            raise ValueError("decision differs from exact frontier-probe allocation")

    def revalidate(self) -> None:
        if type(self) is not FrontierProbeSlateDecision:
            raise TypeError("decision must be exact FrontierProbeSlateDecision")
        FrontierProbeSlateDecision.__post_init__(self)

    @property
    def prior_only(self) -> bool:
        self.revalidate()
        snapshot = self.request.calibration_snapshot
        return snapshot is None or all(
            observation.prediction.wave_index < self.request.slate.wave_index
            for observation in snapshot.observations
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "event_type": "frontier_probe_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "policy_configuration_sha256": (
                self.policy_configuration_sha256
            ),
            "source_request": self.source_request.to_record(),
            "projected_request": self.request.to_record(),
            "constraint_projection": self.projection.to_record(),
            "prior_only": self.prior_only,
            "member_evidence": [
                value.to_record() for value in self.member_evidence
            ],
            "selected": [value.to_record() for value in self.selected],
            "target_option_ids": list(self.target_option_ids),
            "available_full_abstention_option_ids": list(
                self.available_full_abstention_option_ids
            ),
            "selected_probe_option_id": self.selected_probe_option_id,
            "retained_target_count": self.retained_target_count,
            "ideal_target_feasible": self.ideal_target_feasible,
            "feasible_subset_count": self.feasible_subset_count,
            "distinct_family_count": self.distinct_family_count,
            "distinct_locus_count": self.distinct_locus_count,
            "distinct_phenotype_count": self.distinct_phenotype_count,
            "administered_card_keys": list(self.administered_card_keys),
            **(
                {}
                if self.memory_dose_assessment is None
                else {
                    "memory_dose_assessment": (
                        self.memory_dose_assessment.to_record()
                    )
                }
            ),
            "claim_scope": (
                "replayable_outcome_blind_allocation_not_efficacy_or_"
                "outcome_claim"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FrontierProbeSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class FrontierProbeSlatePolicy:
    """Keep model anchors and probe only structurally novel full abstention."""

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256
    configuration_sha256: ClassVar[str] = CONFIGURATION_SHA256

    def __post_init__(self) -> None:
        if type(self) is not FrontierProbeSlatePolicy:
            raise TypeError("policy must be exact FrontierProbeSlatePolicy")

    def select(self, request: SlateAllocationRequest) -> FrontierProbeSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        projection = project_evaluation_allocation_request(request)
        _validate_request(projection.projected_request)
        rows = tuple(
            _member_evidence(projection.projected_request, member)
            for member in projection.projected_request.slate.members
        )
        assignment = _best_assignment(projection.projected_request, rows)
        return FrontierProbeSlateDecision(
            projection=projection,
            member_evidence=rows,
            selected=assignment.selected,
            target_option_ids=assignment.target_option_ids,
            available_full_abstention_option_ids=(
                assignment.available_full_abstention_option_ids
            ),
            selected_probe_option_id=assignment.selected_probe_option_id,
            retained_target_count=assignment.retained_target_count,
            ideal_target_feasible=assignment.ideal_target_feasible,
            feasible_subset_count=assignment.feasible_subset_count,
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            memory_dose_assessment=assignment.memory_dose_assessment,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration": _configuration_record(),
            "configuration_sha256": self.configuration_sha256,
            "constraint_projection_policy": {
                "policy_id": PROJECTION_POLICY_ID,
                "policy_version": PROJECTION_POLICY_VERSION,
                "definition_sha256": PROJECTION_DEFINITION_SHA256,
            },
        }


__all__ = [
    "CONFIGURATION_SHA256",
    "EvaluationAllocationConstraintProjection",
    "FrontierProbeAllocatedMember",
    "FrontierProbeMemberEvidence",
    "FrontierProbeSlateDecision",
    "FrontierProbeSlatePolicy",
    "FrontierProbeSlateRole",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "PROJECTION_DEFINITION_SHA256",
    "PROJECTION_POLICY_ID",
    "PROJECTION_POLICY_VERSION",
    "project_evaluation_allocation_request",
]
