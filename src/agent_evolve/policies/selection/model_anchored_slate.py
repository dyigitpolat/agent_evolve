"""Model-anchored, prior-calibrated allocation over a finite K8 slate.

This opt-in policy deliberately leaves :class:`TraceCalibratedSlatePolicy`
unchanged.  It preserves as many members as possible from a configurable
model-ranked prefix, then fills the remaining K4 capacity using only a sealed
prior-wave calibration snapshot.  Feasibility is always authoritative:
compatibility, family, and assigned-card constraints may displace anchors.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import ClassVar

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationRequest,
    assess_allocated_slate_memory_dose,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationCell,
    ForecastConfidenceBin,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseAssessment,
)


POLICY_ID = "model_anchored_prior_calibrated_slate"
POLICY_VERSION = 2
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:model-anchored-prior-calibrated-slate:v2;"
    b"slate-size=8;portfolio-size=4;model-prefix-anchor-count=configurable;"
    b"full-model-top4-anchor-allowed=true;calibrated-fill=fallback-only;"
    b"feasibility-before-ranking=true;maximize-retained-anchors=true;"
    b"fill-score=prior-only-calibrated-exploitation-sum;"
    b"ties=fill-score-vector,fill-structural-sum,diversity,anchor-ranks,"
    b"selected-ranks,option-ids;card-administration=complete;"
    b"calibration-cutoff-exclusive=true;outcome-blind=true"
).hexdigest()

_CONFIGURATION_DOMAIN = b"agent-evolve:model-anchored-slate-config:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:model-anchored-slate-decision:v1\x00"
_SLATE_SIZE = 8
_PORTFOLIO_SIZE = 4


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


def _require_finite_float(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


class ModelAnchoredSlateRole(str, Enum):
    """Engine-owned provenance of a selected K4 member."""

    MODEL_ANCHOR = "model_anchor"
    PRIOR_CALIBRATED_FILL = "prior_calibrated_fill"


@dataclass(frozen=True, slots=True)
class ModelAnchoredMetricScoreRow:
    """One objective's prior-only contribution to exploitation."""

    metric_id: str
    goal: MetricOptimizationGoal
    asserted_direction: MetricEffectDirection
    confidence: ForecastConfidenceBin
    weight: float
    calibration_cell: ForecastCalibrationCell
    calibration_source: str
    favorable_assertion: bool
    adverse_assertion: bool
    signed_exploitation_score: float

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be a non-empty string")
        if type(self.goal) is not MetricOptimizationGoal:
            raise TypeError("goal must be exact MetricOptimizationGoal")
        if type(self.asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")
        _require_finite_float(self.weight, name="weight")
        if self.weight <= 0.0:
            raise ValueError("weight must be strictly positive")
        if type(self.calibration_cell) is not ForecastCalibrationCell:
            raise TypeError("calibration_cell must be exact ForecastCalibrationCell")
        self.calibration_cell.__post_init__()
        if self.calibration_source not in {
            "supported_family",
            "metric_direction_confidence",
            "declared_prior",
        }:
            raise ValueError("unsupported calibration_source")
        if type(self.favorable_assertion) is not bool:
            raise TypeError("favorable_assertion must be exact bool")
        if type(self.adverse_assertion) is not bool:
            raise TypeError("adverse_assertion must be exact bool")
        if self.favorable_assertion and self.adverse_assertion:
            raise ValueError("one assertion cannot be favorable and adverse")
        _require_finite_float(
            self.signed_exploitation_score,
            name="signed_exploitation_score",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal.value,
            "asserted_direction": self.asserted_direction.value,
            "confidence": self.confidence.value,
            "weight_hex": self.weight.hex(),
            "calibration_cell": self.calibration_cell.to_record(),
            "calibration_source": self.calibration_source,
            "favorable_assertion": self.favorable_assertion,
            "adverse_assertion": self.adverse_assertion,
            "signed_exploitation_score_hex": (
                self.signed_exploitation_score.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelAnchoredMemberScoreRow:
    """Prior-only exploitation and frozen structure for one slate member."""

    option_id: str
    option_identity_sha256: str
    model_rank: int
    metric_scores: tuple[ModelAnchoredMetricScoreRow, ...]
    calibrated_exploitation_score: float
    structural_coverage_score: float
    supported_assigned_card_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        if type(self.metric_scores) is not tuple or not self.metric_scores or any(
            type(value) is not ModelAnchoredMetricScoreRow
            for value in self.metric_scores
        ):
            raise ValueError("metric_scores must contain exact metric score rows")
        for value in self.metric_scores:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.metric_scores)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("metric_scores must use canonical unique metric order")
        _require_finite_float(
            self.calibrated_exploitation_score,
            name="calibrated_exploitation_score",
        )
        _require_finite_float(
            self.structural_coverage_score,
            name="structural_coverage_score",
        )
        if not 0.0 <= self.structural_coverage_score <= 1.0:
            raise ValueError("structural_coverage_score must lie in [0, 1]")
        if type(self.supported_assigned_card_keys) is not tuple or any(
            type(value) is not str or not value
            for value in self.supported_assigned_card_keys
        ):
            raise TypeError("supported_assigned_card_keys must be an exact tuple")
        if self.supported_assigned_card_keys != tuple(
            sorted(set(self.supported_assigned_card_keys))
        ):
            raise ValueError("supported_assigned_card_keys must be canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "metric_scores": [value.to_record() for value in self.metric_scores],
            "calibrated_exploitation_score_hex": (
                self.calibrated_exploitation_score.hex()
            ),
            "structural_coverage_score_hex": (
                self.structural_coverage_score.hex()
            ),
            "supported_assigned_card_keys": list(
                self.supported_assigned_card_keys
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelAnchoredAllocatedMember:
    """One selected member with auditable model/calibration provenance."""

    role: ModelAnchoredSlateRole
    option_id: str
    option_identity_sha256: str
    model_rank: int
    calibrated_exploitation_score: float

    def __post_init__(self) -> None:
        if type(self.role) is not ModelAnchoredSlateRole:
            raise TypeError("role must be exact ModelAnchoredSlateRole")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        _require_finite_float(
            self.calibrated_exploitation_score,
            name="calibrated_exploitation_score",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "role": self.role.value,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "calibrated_exploitation_score_hex": (
                self.calibrated_exploitation_score.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class _AnchoredAssignment:
    selected: tuple[ModelAnchoredAllocatedMember, ...]
    retained_anchor_option_ids: tuple[str, ...]
    calibrated_fill_option_ids: tuple[str, ...]
    calibrated_fill_exploitation_score: float
    calibrated_fill_structural_score: float
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    feasible_subset_count: int
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None


def _configuration_record(model_anchor_count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_anchor_count": model_anchor_count,
        "slate_size": _SLATE_SIZE,
        "portfolio_size": _PORTFOLIO_SIZE,
    }


def _favorable(
    direction: MetricEffectDirection,
    goal: MetricOptimizationGoal,
) -> tuple[bool, bool]:
    if direction in {MetricEffectDirection.UNKNOWN, MetricEffectDirection.UNCHANGED}:
        return False, False
    favorable = (
        goal is MetricOptimizationGoal.MINIMIZE
        and direction is MetricEffectDirection.DECREASE
    ) or (
        goal is MetricOptimizationGoal.MAXIMIZE
        and direction is MetricEffectDirection.INCREASE
    )
    return favorable, not favorable


def _score_member(
    request: SlateAllocationRequest,
    member: CalibratedSlateMember,
) -> ModelAnchoredMemberScoreRow:
    snapshot = request.calibration_snapshot
    if snapshot is None:
        raise ValueError("model-anchored policy requires a calibration snapshot")
    objective_by_metric = {value.metric_id: value for value in request.objectives}
    metric_scores: list[ModelAnchoredMetricScoreRow] = []
    weighted_score = 0.0
    total_weight = sum(value.weight for value in request.objectives)
    for prediction in member.predictions:
        objective = objective_by_metric[prediction.metric_id]
        cell, source = snapshot.lookup(
            metric_id=prediction.metric_id,
            asserted_direction=prediction.asserted_direction,
            confidence=prediction.confidence,
            family=member.family,
        )
        favorable, adverse = _favorable(
            prediction.asserted_direction,
            objective.goal,
        )
        probability = cell.posterior_correctness
        signed_score = probability if favorable else -probability if adverse else 0.0
        weighted_score += objective.weight * signed_score
        metric_scores.append(
            ModelAnchoredMetricScoreRow(
                metric_id=prediction.metric_id,
                goal=objective.goal,
                asserted_direction=prediction.asserted_direction,
                confidence=prediction.confidence,
                weight=objective.weight,
                calibration_cell=cell,
                calibration_source=source,
                favorable_assertion=favorable,
                adverse_assertion=adverse,
                signed_exploitation_score=signed_score,
            )
        )
    supported_cards = tuple(
        card
        for card in request.assigned_card_keys
        if card in member.supporting_card_keys
    )
    structural_score = (
        member.structural_evidence.archive_novelty_score
        + member.structural_evidence.structural_coverage_score
    ) / 2.0
    return ModelAnchoredMemberScoreRow(
        option_id=member.option_id,
        option_identity_sha256=member.option_identity_sha256,
        model_rank=member.model_rank,
        metric_scores=tuple(metric_scores),
        calibrated_exploitation_score=weighted_score / total_weight,
        structural_coverage_score=structural_score,
        supported_assigned_card_keys=supported_cards,
    )


def _compatible(
    members: tuple[CalibratedSlateMember, ...],
    allowed_pairs: set[frozenset[str]] | None,
) -> bool:
    if allowed_pairs is None:
        return True
    return all(
        frozenset((left.option_id, right.option_id)) in allowed_pairs
        for left_index, left in enumerate(members)
        for right in members[left_index + 1 :]
    )


def _best_anchored_assignment(
    request: SlateAllocationRequest,
    score_rows: tuple[ModelAnchoredMemberScoreRow, ...],
    *,
    model_anchor_count: int,
) -> _AnchoredAssignment:
    score_by_id = {value.option_id: value for value in score_rows}
    anchor_ids = {
        value.option_id for value in request.slate.members[:model_anchor_count]
    }
    allowed_pairs = (
        None
        if request.pairwise_disjoint_option_id_pairs is None
        else {
            frozenset(value)
            for value in request.pairwise_disjoint_option_id_pairs
        }
    )
    feasible: list[
        tuple[
            tuple[object, ...],
            tuple[CalibratedSlateMember, ...],
            tuple[ModelAnchoredMemberScoreRow, ...],
            tuple[ModelAnchoredMemberScoreRow, ...],
            tuple[str, ...],
            tuple[str, ...],
            int,
            int,
            int,
            PortfolioMemoryDoseAssessment | None,
        ]
    ] = []
    for subset in combinations(request.slate.members, request.portfolio_size):
        if not _compatible(subset, allowed_pairs):
            continue
        family_count = len({value.family for value in subset})
        if (
            request.min_distinct_families is not None
            and family_count < request.min_distinct_families
        ):
            continue
        rows = tuple(score_by_id[value.option_id] for value in subset)
        administered = tuple(
            sorted(
                {
                    card
                    for row in rows
                    for card in row.supported_assigned_card_keys
                }
            )
        )
        if administered != request.assigned_card_keys:
            continue
        memory_dose_assessment = (
            None
            if request.memory_dose_contract is None
            else assess_allocated_slate_memory_dose(
                request,
                subset,
            )
        )
        if (
            memory_dose_assessment is not None
            and not memory_dose_assessment.passed
        ):
            continue
        retained = tuple(value for value in rows if value.option_id in anchor_ids)
        fills = tuple(value for value in rows if value.option_id not in anchor_ids)
        fill_scores = tuple(
            sorted(
                (value.calibrated_exploitation_score for value in fills),
                reverse=True,
            )
        )
        fill_score = sum(fill_scores, 0.0)
        fill_structural = sum(
            (value.structural_coverage_score for value in fills),
            0.0,
        )
        locus_count = len({value.locus_key for value in subset})
        phenotype_count = len(
            {value.phenotype_identity_sha256 for value in subset}
        )
        retained_ranks = tuple(value.model_rank for value in retained)
        selected_ranks = tuple(value.model_rank for value in rows)
        # ``min`` over this key implements the documented strict priority.
        key: tuple[object, ...] = (
            -len(retained),
            -fill_score,
            tuple(-value for value in fill_scores),
            -fill_structural,
            -family_count,
            -locus_count,
            -phenotype_count,
            retained_ranks,
            selected_ranks,
            tuple(value.option_id for value in rows),
        )
        feasible.append(
            (
                key,
                subset,
                retained,
                fills,
                administered,
                tuple(value.option_id for value in retained),
                family_count,
                locus_count,
                phenotype_count,
                memory_dose_assessment,
            )
        )
    if not feasible:
        raise ValueError(
            "slate has no feasible K4 subset administering every assigned card"
        )
    (
        _,
        subset,
        retained,
        fills,
        administered,
        retained_ids,
        family_count,
        locus_count,
        phenotype_count,
        memory_dose_assessment,
    ) = min(feasible, key=lambda value: value[0])
    fill_ids = tuple(value.option_id for value in fills)
    selected = tuple(
        ModelAnchoredAllocatedMember(
            role=(
                ModelAnchoredSlateRole.MODEL_ANCHOR
                if member.option_id in anchor_ids
                else ModelAnchoredSlateRole.PRIOR_CALIBRATED_FILL
            ),
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            model_rank=member.model_rank,
            calibrated_exploitation_score=(
                score_by_id[member.option_id].calibrated_exploitation_score
            ),
        )
        for member in subset
    )
    return _AnchoredAssignment(
        selected=selected,
        retained_anchor_option_ids=retained_ids,
        calibrated_fill_option_ids=fill_ids,
        calibrated_fill_exploitation_score=sum(
            (value.calibrated_exploitation_score for value in fills),
            0.0,
        ),
        calibrated_fill_structural_score=sum(
            (value.structural_coverage_score for value in fills),
            0.0,
        ),
        distinct_family_count=family_count,
        distinct_locus_count=locus_count,
        distinct_phenotype_count=phenotype_count,
        administered_card_keys=administered,
        feasible_subset_count=len(feasible),
        memory_dose_assessment=memory_dose_assessment,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ModelAnchoredSlateDecision:
    """Replayable receipt for one outcome-blind K8-to-K4 decision."""

    request: SlateAllocationRequest
    model_anchor_count: int
    score_rows: tuple[ModelAnchoredMemberScoreRow, ...]
    selected: tuple[ModelAnchoredAllocatedMember, ...]
    retained_anchor_option_ids: tuple[str, ...]
    calibrated_fill_option_ids: tuple[str, ...]
    calibrated_fill_exploitation_score: float
    calibrated_fill_structural_score: float
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    feasible_subset_count: int
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        self.request.revalidate()
        _validate_policy_request(self.request, self.model_anchor_count)
        if type(self.score_rows) is not tuple or any(
            type(value) is not ModelAnchoredMemberScoreRow
            for value in self.score_rows
        ):
            raise TypeError("score_rows must contain exact member score rows")
        for value in self.score_rows:
            value.__post_init__()
        if type(self.selected) is not tuple or any(
            type(value) is not ModelAnchoredAllocatedMember
            for value in self.selected
        ):
            raise TypeError("selected must contain exact allocated members")
        for value in self.selected:
            value.__post_init__()
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not (
                PortfolioMemoryDoseAssessment
            ):
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()
        expected_rows = tuple(
            _score_member(self.request, value)
            for value in self.request.slate.members
        )
        if self.score_rows != expected_rows:
            raise ValueError("score rows differ from sealed prior calibration")
        expected = _best_anchored_assignment(
            self.request,
            expected_rows,
            model_anchor_count=self.model_anchor_count,
        )
        observed = (
            self.selected,
            self.retained_anchor_option_ids,
            self.calibrated_fill_option_ids,
            self.calibrated_fill_exploitation_score,
            self.calibrated_fill_structural_score,
            self.distinct_family_count,
            self.distinct_locus_count,
            self.distinct_phenotype_count,
            self.administered_card_keys,
            self.feasible_subset_count,
            self.memory_dose_assessment,
        )
        expected_values = (
            expected.selected,
            expected.retained_anchor_option_ids,
            expected.calibrated_fill_option_ids,
            expected.calibrated_fill_exploitation_score,
            expected.calibrated_fill_structural_score,
            expected.distinct_family_count,
            expected.distinct_locus_count,
            expected.distinct_phenotype_count,
            expected.administered_card_keys,
            expected.feasible_subset_count,
            expected.memory_dose_assessment,
        )
        if observed != expected_values:
            raise ValueError("decision differs from exact model-anchored allocation")

    def revalidate(self) -> None:
        if type(self) is not ModelAnchoredSlateDecision:
            raise TypeError("decision must be exact ModelAnchoredSlateDecision")
        ModelAnchoredSlateDecision.__post_init__(self)

    @property
    def policy_configuration_sha256(self) -> str:
        return _hash(
            _CONFIGURATION_DOMAIN,
            _configuration_record(self.model_anchor_count),
        )

    @property
    def prior_only(self) -> bool:
        self.revalidate()
        snapshot = self.request.calibration_snapshot
        assert snapshot is not None
        return all(
            value.prediction.wave_index < self.request.slate.wave_index
            for value in snapshot.observations
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "event_type": "model_anchored_prior_calibrated_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "policy_configuration": _configuration_record(
                self.model_anchor_count
            ),
            "policy_configuration_sha256": (
                self.policy_configuration_sha256
            ),
            "request": self.request.to_record(),
            "request_sha256": self.request.request_sha256,
            "calibration_snapshot_sha256": (
                self.request.calibration_snapshot.snapshot_sha256
                if self.request.calibration_snapshot is not None
                else None
            ),
            "prior_only": self.prior_only,
            "selection_priority": [
                "retained_model_anchor_count",
                "calibrated_fill_exploitation_sum",
                "calibrated_fill_exploitation_vector",
                "calibrated_fill_structural_sum",
                "distinct_family_count",
                "distinct_locus_count",
                "distinct_phenotype_count",
                "retained_anchor_model_ranks",
                "selected_model_ranks",
                "option_ids",
            ],
            "score_rows": [value.to_record() for value in self.score_rows],
            "selected": [value.to_record() for value in self.selected],
            "retained_anchor_option_ids": list(
                self.retained_anchor_option_ids
            ),
            "calibrated_fill_option_ids": list(
                self.calibrated_fill_option_ids
            ),
            "calibrated_fill_exploitation_score_hex": (
                self.calibrated_fill_exploitation_score.hex()
            ),
            "calibrated_fill_structural_score_hex": (
                self.calibrated_fill_structural_score.hex()
            ),
            "distinct_family_count": self.distinct_family_count,
            "distinct_locus_count": self.distinct_locus_count,
            "distinct_phenotype_count": self.distinct_phenotype_count,
            "administered_card_keys": list(self.administered_card_keys),
            "feasible_subset_count": self.feasible_subset_count,
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
                "replayable_prior_only_allocation_not_efficacy_or_outcome_claim"
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
            type(other) is ModelAnchoredSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


def _validate_policy_request(
    request: SlateAllocationRequest,
    model_anchor_count: int,
) -> None:
    if type(model_anchor_count) is not int or not (
        1 <= model_anchor_count <= _PORTFOLIO_SIZE
    ):
        raise ValueError("model_anchor_count must lie in [1, 4]")
    if len(request.slate.members) != _SLATE_SIZE:
        raise ValueError("model-anchored policy requires exactly eight slate members")
    if request.portfolio_size != _PORTFOLIO_SIZE:
        raise ValueError("model-anchored policy requires exactly four evaluations")
    snapshot = request.calibration_snapshot
    if snapshot is None:
        raise ValueError("model-anchored policy requires a calibration snapshot")
    if snapshot.cutoff_wave_index_exclusive > request.slate.wave_index:
        raise ValueError("calibration snapshot cutoff reaches beyond current wave")
    if any(
        value.prediction.wave_index >= request.slate.wave_index
        for value in snapshot.observations
    ):
        raise ValueError("calibration snapshot contains current-wave outcome evidence")


@dataclass(frozen=True, slots=True)
class ModelAnchoredCalibratedSlatePolicy:
    """Preserve model anchors, then use prior calibration for K4 fill slots."""

    model_anchor_count: int = 3

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.model_anchor_count) is not int or not (
            1 <= self.model_anchor_count <= _PORTFOLIO_SIZE
        ):
            raise ValueError("model_anchor_count must lie in [1, 4]")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _CONFIGURATION_DOMAIN,
            _configuration_record(self.model_anchor_count),
        )

    def select(self, request: SlateAllocationRequest) -> ModelAnchoredSlateDecision:
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        _validate_policy_request(request, self.model_anchor_count)
        rows = tuple(_score_member(request, value) for value in request.slate.members)
        assignment = _best_anchored_assignment(
            request,
            rows,
            model_anchor_count=self.model_anchor_count,
        )
        return ModelAnchoredSlateDecision(
            request=request,
            model_anchor_count=self.model_anchor_count,
            score_rows=rows,
            selected=assignment.selected,
            retained_anchor_option_ids=assignment.retained_anchor_option_ids,
            calibrated_fill_option_ids=assignment.calibrated_fill_option_ids,
            calibrated_fill_exploitation_score=(
                assignment.calibrated_fill_exploitation_score
            ),
            calibrated_fill_structural_score=(
                assignment.calibrated_fill_structural_score
            ),
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            feasible_subset_count=assignment.feasible_subset_count,
            memory_dose_assessment=assignment.memory_dose_assessment,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration": _configuration_record(self.model_anchor_count),
            "configuration_sha256": self.configuration_sha256,
        }


__all__ = [
    "ModelAnchoredAllocatedMember",
    "ModelAnchoredCalibratedSlatePolicy",
    "ModelAnchoredMemberScoreRow",
    "ModelAnchoredMetricScoreRow",
    "ModelAnchoredSlateDecision",
    "ModelAnchoredSlateRole",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
]
