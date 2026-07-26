"""Prior-only calibrated acquisition over a finite K8 slate.

The model proposes eight workload-grounded candidates.  This provider-free
policy assigns four distinct evaluation roles using only facts sealed before
the current wave: calibrated forecast correctness, explicit abstention,
structural evidence, and feasibility constraints. Model-authored card
citations remain observable diagnostics but have no allocation authority.

The policy is intentionally workload-neutral and has no fitted benchmark
hyperparameters.  In particular, an ``unknown`` direction is treated as
maximum epistemic uncertainty rather than as zero information.  Epistemic
priority is multiplied by structural novelty/coverage so that uncertainty is
spent on candidates that can distinguish materially different hypotheses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from typing import Any, ClassVar

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
from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITE_OPTION_FAMILY,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseAssessment,
)


POLICY_ID = "calibrated_frontier_four_role_slate"
POLICY_VERSION = 2
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:calibrated-frontier-four-role-slate:v2;"
    b"slate-size=8;portfolio-size=4;outcome-blind=true;"
    b"calibration-cutoff-exclusive=true;"
    b"roles=calibrated-exploit,calibrated-frontier,epistemic-structural,"
    b"structural-coverage;unknown-epistemic-score=1;"
    b"expected-signed-utility=favorable:2p-1,adverse:1-2p;"
    b"known-epistemic-score=mean(posterior-uncertainty,calibrated-disagreement);"
    b"epistemic-structural-coupling=multiply;"
    b"frontier-score=expected-utility+0.5*structural;"
    b"joint-exact-role-assignment=true;diversity-weight=0.25;"
    b"model-card-citations=diagnostic-only;hard-memory-dose=matched-diagnostic-only;"
    b"compatibility-family-and-patch-feasibility=true;"
    b"no-benchmark-specific-parameters=true"
).hexdigest()

OPERATOR_STRATIFIED_POLICY_ID = (
    "operator_stratified_calibrated_frontier_four_role_slate"
)
OPERATOR_STRATIFIED_POLICY_VERSION = 1
OPERATOR_STRATIFIED_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:operator-stratified-calibrated-frontier-four-role-slate:v1;"
    b"base=calibrated-frontier-four-role-v2;"
    b"required-family-minimums=authenticated-configuration;"
    b"minimums-are-evaluation-assay-slots-not-quality-priors;"
    b"joint-feasibility-and-role-optimization=true;outcome-blind=true;"
    b"no-workload-specific-parameters=true"
).hexdigest()

HORIZON_BOUNDED_POLICY_ID = (
    "horizon_bounded_calibrated_frontier_four_role_slate"
)
HORIZON_BOUNDED_POLICY_VERSION = 1
HORIZON_BOUNDED_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:horizon-bounded-calibrated-frontier-four-role-slate:v1;"
    b"base=calibrated-frontier-four-role-v2;"
    b"family-exposure-phases=authenticated-configuration;"
    b"phase-index=sealed-wave-index;lower-and-upper-bounds=true;"
    b"bounds-are-evaluation-assay-slots-not-quality-priors;"
    b"infeasible-bounds=minimum-l1-structural-recourse;"
    b"joint-feasibility-and-role-optimization=true;outcome-blind=true;"
    b"arbitrary-action-family-identifiers=true;"
    b"no-workload-specific-parameters=true"
).hexdigest()

_DECISION_DOMAIN = b"agent-evolve:structural-posterior-slate-decision:v1\x00"
_CONFIGURATION_DOMAIN = b"agent-evolve:calibrated-frontier-configuration:v2\x00"
_OPERATOR_STRATIFIED_CONFIGURATION_DOMAIN = (
    b"agent-evolve:operator-stratified-frontier-configuration:v1\x00"
)
_OPERATOR_STRATIFIED_DECISION_DOMAIN = (
    b"agent-evolve:operator-stratified-frontier-decision:v1\x00"
)
_HORIZON_BOUNDED_CONFIGURATION_DOMAIN = (
    b"agent-evolve:horizon-bounded-frontier-configuration:v1\x00"
)
_HORIZON_BOUNDED_DECISION_DOMAIN = (
    b"agent-evolve:horizon-bounded-frontier-decision:v1\x00"
)
_SLATE_SIZE = 8
_PORTFOLIO_SIZE = 4
_FRONTIER_STRUCTURAL_WEIGHT = 0.5
_DIVERSITY_WEIGHT = 0.25


def _configuration_record() -> dict[str, object]:
    return {
        "schema_version": 2,
        "slate_size": _SLATE_SIZE,
        "portfolio_size": _PORTFOLIO_SIZE,
        "unknown_epistemic_score_hex": (1.0).hex(),
        "epistemic_structural_coupling": "multiply",
        "expected_signed_utility": {
            "favorable": "2p_minus_1",
            "adverse": "1_minus_2p",
        },
        "frontier_structural_weight_hex": _FRONTIER_STRUCTURAL_WEIGHT.hex(),
        "diversity_weight_hex": _DIVERSITY_WEIGHT.hex(),
        "model_card_citation_authority": "diagnostic_only",
        "hard_memory_dose": "matched_diagnostic_only",
        "benchmark_specific_parameters": [],
    }


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


def _require_finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


def _require_unit_interval(value: float, *, name: str) -> None:
    _require_finite(value, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _require_canonical_strings(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or not value for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of non-empty strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


def _validate_required_family_minimums(
    values: tuple[tuple[str, int], ...],
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError("required_family_minimums must be a non-empty exact tuple")
    if values != tuple(sorted(values)):
        raise ValueError("required_family_minimums must be canonical")
    families: list[str] = []
    total = 0
    for family, minimum in values:
        if type(family) is not str or not family:
            raise ValueError("required family must be a non-empty string")
        if type(minimum) is not int or not 1 <= minimum <= _PORTFOLIO_SIZE:
            raise ValueError("required family minimum must lie in [1, 4]")
        families.append(family)
        total += minimum
    if len(set(families)) != len(families):
        raise ValueError("required families cannot repeat")
    if total > _PORTFOLIO_SIZE:
        raise ValueError("required family minimums exceed the evaluation budget")


@dataclass(frozen=True, slots=True)
class FamilyExposureBound:
    """Authenticated lower and upper evaluator exposure for one action family."""

    family: str
    minimum_evaluations: int
    maximum_evaluations: int

    def __post_init__(self) -> None:
        if type(self.family) is not str or not self.family:
            raise ValueError("family must be a non-empty exact string")
        if (
            type(self.minimum_evaluations) is not int
            or not 0 <= self.minimum_evaluations <= _PORTFOLIO_SIZE
        ):
            raise ValueError("minimum_evaluations must lie in [0, 4]")
        if (
            type(self.maximum_evaluations) is not int
            or not 0 <= self.maximum_evaluations <= _PORTFOLIO_SIZE
        ):
            raise ValueError("maximum_evaluations must lie in [0, 4]")
        if self.minimum_evaluations > self.maximum_evaluations:
            raise ValueError("family exposure minimum cannot exceed its maximum")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "family": self.family,
            "minimum_evaluations": self.minimum_evaluations,
            "maximum_evaluations": self.maximum_evaluations,
        }


@dataclass(frozen=True, slots=True)
class FamilyExposurePhase:
    """Family exposure bounds activated at one sealed campaign wave index."""

    start_wave_index: int
    bounds: tuple[FamilyExposureBound, ...]

    def __post_init__(self) -> None:
        if type(self.start_wave_index) is not int or self.start_wave_index < 0:
            raise ValueError("start_wave_index must be a non-negative exact integer")
        if type(self.bounds) is not tuple or not self.bounds:
            raise ValueError("bounds must be a non-empty exact tuple")
        if len(self.bounds) > _PORTFOLIO_SIZE:
            raise ValueError("family exposure bound count exceeds the portfolio size")
        if any(type(value) is not FamilyExposureBound for value in self.bounds):
            raise TypeError("bounds must contain exact FamilyExposureBound values")
        for value in self.bounds:
            value.__post_init__()
        families = tuple(value.family for value in self.bounds)
        if families != tuple(sorted(set(families))):
            raise ValueError("phase families must be unique and canonical")
        if sum(value.minimum_evaluations for value in self.bounds) > _PORTFOLIO_SIZE:
            raise ValueError("phase family minima exceed the evaluation budget")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "start_wave_index": self.start_wave_index,
            "bounds": [value.to_record() for value in self.bounds],
        }


def _validate_family_exposure_phases(
    values: tuple[FamilyExposurePhase, ...],
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError("family_exposure_phases must be a non-empty exact tuple")
    if any(type(value) is not FamilyExposurePhase for value in values):
        raise TypeError(
            "family_exposure_phases must contain exact FamilyExposurePhase values"
        )
    for value in values:
        value.__post_init__()
    starts = tuple(value.start_wave_index for value in values)
    if starts[0] != 0:
        raise ValueError("family exposure phases must start at wave zero")
    if starts != tuple(sorted(set(starts))):
        raise ValueError("family exposure phase starts must be unique and canonical")


def build_terminal_tapered_family_exposure_phases(
    *,
    family: str,
    terminal_wave_index: int,
    discovery_exposure: int = 2,
    terminal_exposure: int = 0,
) -> tuple[FamilyExposurePhase, ...]:
    """Build a generic explore-then-taper schedule for a finite campaign.

    The caller supplies a topology family and the campaign's final selector
    wave.  No workload identifier or objective value enters the schedule.
    """

    if type(terminal_wave_index) is not int or terminal_wave_index <= 0:
        raise ValueError("terminal_wave_index must be a positive exact integer")
    discovery = FamilyExposureBound(
        family=family,
        minimum_evaluations=discovery_exposure,
        maximum_evaluations=discovery_exposure,
    )
    terminal = FamilyExposureBound(
        family=family,
        minimum_evaluations=terminal_exposure,
        maximum_evaluations=terminal_exposure,
    )
    phases = (
        FamilyExposurePhase(start_wave_index=0, bounds=(discovery,)),
        FamilyExposurePhase(
            start_wave_index=terminal_wave_index,
            bounds=(terminal,),
        ),
    )
    _validate_family_exposure_phases(phases)
    return phases


def build_controller_owned_family_exposure_phases(
    *,
    family: str,
) -> tuple[FamilyExposurePhase, ...]:
    """Expose the full K4 family range to an authenticated outer controller.

    The horizon allocator remains responsible for structural feasibility and
    replay, but contributes no fixed family-dose prior.  A prospective
    contextual allocation contract can therefore own the exact evaluator
    exposure without conflicting with a second schedule.
    """

    phases = (
        FamilyExposurePhase(
            start_wave_index=0,
            bounds=(
                FamilyExposureBound(
                    family=family,
                    minimum_evaluations=0,
                    maximum_evaluations=_PORTFOLIO_SIZE,
                ),
            ),
        ),
    )
    _validate_family_exposure_phases(phases)
    return phases


def _active_family_exposure_phase(
    phases: tuple[FamilyExposurePhase, ...],
    *,
    wave_index: int,
) -> FamilyExposurePhase:
    _validate_family_exposure_phases(phases)
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("wave_index must be a non-negative exact integer")
    return max(
        (value for value in phases if value.start_wave_index <= wave_index),
        key=lambda value: value.start_wave_index,
    )


def _horizon_bounded_configuration_record(
    phases: tuple[FamilyExposurePhase, ...],
) -> dict[str, object]:
    _validate_family_exposure_phases(phases)
    return {
        "schema_version": 1,
        "base_allocator": {
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "definition_sha256": POLICY_DEFINITION_SHA256,
            "configuration_sha256": _hash(
                _CONFIGURATION_DOMAIN,
                _configuration_record(),
            ),
        },
        "family_exposure_phases": [value.to_record() for value in phases],
        "phase_authority": "sealed_wave_index",
        "constraint_semantics": "assay_exposure_not_quality_prior",
        "outcomes_consulted": False,
        "workload_identifiers": [],
    }


class StructuralPosteriorSlateRole(str, Enum):
    """Engine-owned purpose of one selected evaluation."""

    CALIBRATED_EXPLOIT = "calibrated_exploit"
    CALIBRATED_FRONTIER = "calibrated_frontier"
    EPISTEMIC_STRUCTURAL = "epistemic_structural"
    STRUCTURAL_COVERAGE = "structural_coverage"


_ROLES = (
    StructuralPosteriorSlateRole.CALIBRATED_EXPLOIT,
    StructuralPosteriorSlateRole.CALIBRATED_FRONTIER,
    StructuralPosteriorSlateRole.EPISTEMIC_STRUCTURAL,
    StructuralPosteriorSlateRole.STRUCTURAL_COVERAGE,
)


@dataclass(frozen=True, slots=True)
class StructuralPosteriorMetricScoreRow:
    """One objective's auditable exploitation and epistemic contributions."""

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
    posterior_uncertainty_score: float
    calibrated_disagreement_score: float
    epistemic_score: float
    explicit_abstention: bool

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be a non-empty string")
        if type(self.goal) is not MetricOptimizationGoal:
            raise TypeError("goal must be exact MetricOptimizationGoal")
        if type(self.asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")
        _require_finite(self.weight, name="weight")
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
            raise TypeError("favorable_assertion must be an exact bool")
        if type(self.adverse_assertion) is not bool:
            raise TypeError("adverse_assertion must be an exact bool")
        if self.favorable_assertion and self.adverse_assertion:
            raise ValueError("one assertion cannot be favorable and adverse")
        _require_finite(
            self.signed_exploitation_score,
            name="signed_exploitation_score",
        )
        for name in (
            "posterior_uncertainty_score",
            "calibrated_disagreement_score",
            "epistemic_score",
        ):
            _require_unit_interval(getattr(self, name), name=name)
        if type(self.explicit_abstention) is not bool:
            raise TypeError("explicit_abstention must be an exact bool")
        expected_abstention = (
            self.asserted_direction is MetricEffectDirection.UNKNOWN
        )
        if self.explicit_abstention is not expected_abstention:
            raise ValueError("explicit_abstention differs from asserted direction")
        if self.explicit_abstention and self.epistemic_score != 1.0:
            raise ValueError("explicit abstention must have maximum epistemic score")

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
            "posterior_uncertainty_score_hex": (
                self.posterior_uncertainty_score.hex()
            ),
            "calibrated_disagreement_score_hex": (
                self.calibrated_disagreement_score.hex()
            ),
            "epistemic_score_hex": self.epistemic_score.hex(),
            "explicit_abstention": self.explicit_abstention,
        }


@dataclass(frozen=True, slots=True)
class StructuralPosteriorMemberScoreRow:
    """All prior-only role scores for one member of the current slate."""

    option_id: str
    option_identity_sha256: str
    model_rank: int
    metric_scores: tuple[StructuralPosteriorMetricScoreRow, ...]
    calibrated_exploitation_score: float
    calibrated_frontier_score: float
    raw_epistemic_score: float
    structural_coverage_score: float
    epistemic_structural_score: float
    model_declared_assigned_card_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        if type(self.metric_scores) is not tuple or not self.metric_scores or any(
            type(value) is not StructuralPosteriorMetricScoreRow
            for value in self.metric_scores
        ):
            raise ValueError("metric_scores must contain exact score rows")
        for value in self.metric_scores:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.metric_scores)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("metric_scores must use canonical unique metric order")
        for name in (
            "calibrated_exploitation_score",
            "calibrated_frontier_score",
        ):
            _require_finite(getattr(self, name), name=name)
        for name in (
            "raw_epistemic_score",
            "structural_coverage_score",
            "epistemic_structural_score",
        ):
            _require_unit_interval(getattr(self, name), name=name)
        if self.epistemic_structural_score != (
            self.raw_epistemic_score * self.structural_coverage_score
        ):
            raise ValueError("epistemic-structural score differs from fixed coupling")
        _require_canonical_strings(
            self.model_declared_assigned_card_keys,
            name="model_declared_assigned_card_keys",
        )

    def score_for(self, role: StructuralPosteriorSlateRole) -> float:
        self.__post_init__()
        if role is StructuralPosteriorSlateRole.CALIBRATED_EXPLOIT:
            return self.calibrated_exploitation_score
        if role is StructuralPosteriorSlateRole.CALIBRATED_FRONTIER:
            return self.calibrated_frontier_score
        if role is StructuralPosteriorSlateRole.EPISTEMIC_STRUCTURAL:
            return self.epistemic_structural_score
        if role is StructuralPosteriorSlateRole.STRUCTURAL_COVERAGE:
            return self.structural_coverage_score
        raise TypeError("role must be exact StructuralPosteriorSlateRole")

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
            "calibrated_frontier_score_hex": (
                self.calibrated_frontier_score.hex()
            ),
            "raw_epistemic_score_hex": self.raw_epistemic_score.hex(),
            "structural_coverage_score_hex": (
                self.structural_coverage_score.hex()
            ),
            "epistemic_structural_score_hex": (
                self.epistemic_structural_score.hex()
            ),
            "model_declared_assigned_card_keys": list(
                self.model_declared_assigned_card_keys
            ),
        }


@dataclass(frozen=True, slots=True)
class StructuralPosteriorAllocatedMember:
    """One selected member with its engine-owned role and score."""

    role: StructuralPosteriorSlateRole
    option_id: str
    option_identity_sha256: str
    model_rank: int
    role_score: float

    def __post_init__(self) -> None:
        if type(self.role) is not StructuralPosteriorSlateRole:
            raise TypeError("role must be exact StructuralPosteriorSlateRole")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        _require_finite(self.role_score, name="role_score")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "role": self.role.value,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "role_score_hex": self.role_score.hex(),
        }


@dataclass(frozen=True, slots=True)
class _StructuralPosteriorAssignment:
    selected: tuple[StructuralPosteriorAllocatedMember, ...]
    joint_score: float
    diversity_score: float
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    feasible_role_assignment_count: int
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None
    family_exposure_violation_count: int


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
) -> StructuralPosteriorMemberScoreRow:
    snapshot = request.calibration_snapshot
    if snapshot is None:
        raise ValueError("structural-posterior policy requires a calibration snapshot")
    objective_by_metric = {value.metric_id: value for value in request.objectives}
    metric_scores: list[StructuralPosteriorMetricScoreRow] = []
    weighted_exploitation = 0.0
    weighted_epistemic = 0.0
    total_weight = sum(value.weight for value in request.objectives)
    for prediction in member.predictions:
        objective = objective_by_metric[prediction.metric_id]
        cell, source = snapshot.lookup(
            metric_id=prediction.metric_id,
            asserted_direction=prediction.asserted_direction,
            confidence=prediction.confidence,
            family=member.family,
        )
        probability = cell.posterior_correctness
        favorable, adverse = _favorable(
            prediction.asserted_direction,
            objective.goal,
        )
        # Posterior correctness is a probability that the asserted direction
        # is right, not a non-negative confidence multiplier. A model family
        # below chance is useful negative evidence: its favorable assertion
        # has negative expected utility and its adverse assertion has positive
        # expected utility. This is the key distinction between calibration
        # and merely shrinking an unreliable model toward zero.
        signed_exploitation = (
            (2.0 * probability) - 1.0
            if favorable
            else 1.0 - (2.0 * probability)
            if adverse
            else 0.0
        )
        posterior_uncertainty = 1.0 - abs((2.0 * probability) - 1.0)
        calibrated_disagreement = max(0.0, 0.5 - probability) * 2.0
        explicit_abstention = (
            prediction.asserted_direction is MetricEffectDirection.UNKNOWN
        )
        epistemic = (
            1.0
            if explicit_abstention
            else (posterior_uncertainty + calibrated_disagreement) / 2.0
        )
        weighted_exploitation += objective.weight * signed_exploitation
        weighted_epistemic += objective.weight * epistemic
        metric_scores.append(
            StructuralPosteriorMetricScoreRow(
                metric_id=prediction.metric_id,
                goal=objective.goal,
                asserted_direction=prediction.asserted_direction,
                confidence=prediction.confidence,
                weight=objective.weight,
                calibration_cell=cell,
                calibration_source=source,
                favorable_assertion=favorable,
                adverse_assertion=adverse,
                signed_exploitation_score=signed_exploitation,
                posterior_uncertainty_score=posterior_uncertainty,
                calibrated_disagreement_score=calibrated_disagreement,
                epistemic_score=epistemic,
                explicit_abstention=explicit_abstention,
            )
        )
    exploitation = weighted_exploitation / total_weight
    raw_epistemic = weighted_epistemic / total_weight
    declared_cards = tuple(
        card
        for card in request.assigned_card_keys
        if card in member.supporting_card_keys
    )
    structural = (
        member.structural_evidence.archive_novelty_score
        + member.structural_evidence.structural_coverage_score
    ) / 2.0
    return StructuralPosteriorMemberScoreRow(
        option_id=member.option_id,
        option_identity_sha256=member.option_identity_sha256,
        model_rank=member.model_rank,
        metric_scores=tuple(metric_scores),
        calibrated_exploitation_score=exploitation,
        calibrated_frontier_score=(
            exploitation + (_FRONTIER_STRUCTURAL_WEIGHT * structural)
        ),
        raw_epistemic_score=raw_epistemic,
        structural_coverage_score=structural,
        epistemic_structural_score=raw_epistemic * structural,
        model_declared_assigned_card_keys=declared_cards,
    )


def score_structural_posterior_slate(
    request: SlateAllocationRequest,
) -> tuple[StructuralPosteriorMemberScoreRow, ...]:
    """Project prior-only calibrated structural evidence without allocating.

    This is the stable projection boundary for downstream acquisition policies.
    It deliberately returns score evidence for the complete sealed slate; it
    does not choose evaluator exposures or consult any candidate outcome.
    """

    if type(request) is not SlateAllocationRequest:
        raise TypeError("request must be exact SlateAllocationRequest")
    request.revalidate()
    return tuple(_score_member(request, member) for member in request.slate.members)


def _best_assignment(
    request: SlateAllocationRequest,
    score_rows: tuple[StructuralPosteriorMemberScoreRow, ...],
    *,
    required_family_minimums: tuple[tuple[str, int], ...] = (),
    required_family_bounds: tuple[FamilyExposureBound, ...] = (),
    minimize_family_bound_violation: bool = False,
) -> _StructuralPosteriorAssignment:
    if required_family_minimums and required_family_bounds:
        raise ValueError("family minima and family bounds are mutually exclusive")
    if required_family_minimums:
        _validate_required_family_minimums(required_family_minimums)
    if required_family_bounds:
        FamilyExposurePhase(0, required_family_bounds).__post_init__()
    if minimize_family_bound_violation and not required_family_bounds:
        raise ValueError("family-bound recourse requires explicit family bounds")
    member_by_id = {value.option_id: value for value in request.slate.members}
    required_option_ids = set(request.required_option_ids)
    allowed_pairs = (
        None
        if request.pairwise_disjoint_option_id_pairs is None
        else {frozenset(value) for value in request.pairwise_disjoint_option_id_pairs}
    )
    memory_dose_by_subset: dict[
        tuple[str, ...], PortfolioMemoryDoseAssessment | None
    ] = {}
    feasible: list[
        tuple[
            tuple[object, ...],
            tuple[StructuralPosteriorMemberScoreRow, ...],
            tuple[float, ...],
            float,
            float,
            int,
            int,
            int,
            tuple[str, ...],
            PortfolioMemoryDoseAssessment | None,
        ]
    ] = []
    for rows in permutations(score_rows, _PORTFOLIO_SIZE):
        if not required_option_ids.issubset(
            {row.option_id for row in rows}
        ):
            continue
        if allowed_pairs is not None and any(
            frozenset((left.option_id, right.option_id)) not in allowed_pairs
            for left_index, left in enumerate(rows)
            for right in rows[left_index + 1 :]
        ):
            continue
        members = tuple(member_by_id[row.option_id] for row in rows)
        family_count = len({value.family for value in members})
        family_exposure_violation = sum(
            max(
                bound.minimum_evaluations
                - sum(value.family == bound.family for value in members),
                0,
                sum(value.family == bound.family for value in members)
                - bound.maximum_evaluations,
            )
            for bound in required_family_bounds
        )
        if (
            request.min_distinct_families is not None
            and family_count < request.min_distinct_families
        ):
            continue
        if any(
            sum(value.family == family for value in members) < minimum
            for family, minimum in required_family_minimums
        ):
            continue
        if family_exposure_violation and not minimize_family_bound_violation:
            continue
        if (
            request.memory_dose_contract is not None
            and not rows[1].model_declared_assigned_card_keys
        ):
            continue
        administered = (
            ()
            if request.memory_dose_contract is None
            else tuple(
                sorted(
                    {
                        card
                        for row in rows
                        for card in row.model_declared_assigned_card_keys
                    }
                )
            )
        )
        if (
            request.memory_dose_contract is not None
            and administered != request.assigned_card_keys
        ):
            continue
        subset_key = tuple(sorted(row.option_id for row in rows))
        if subset_key not in memory_dose_by_subset:
            canonical_members = tuple(
                member_by_id[option_id] for option_id in subset_key
            )
            memory_dose_by_subset[subset_key] = (
                None
                if request.memory_dose_contract is None
                else assess_allocated_slate_memory_dose(
                    request,
                    canonical_members,
                )
            )
        dose_assessment = memory_dose_by_subset[subset_key]
        if dose_assessment is not None and not dose_assessment.passed:
            continue
        locus_count = len({value.locus_key for value in members})
        phenotype_count = len(
            {value.phenotype_identity_sha256 for value in members}
        )
        diversity = (
            family_count + locus_count + phenotype_count
        ) / (3.0 * _PORTFOLIO_SIZE)
        role_scores = tuple(row.score_for(role) for role, row in zip(_ROLES, rows))
        joint_score = sum(role_scores) + (_DIVERSITY_WEIGHT * diversity)
        tie_key: tuple[object, ...] = (
            *((family_exposure_violation,) if minimize_family_bound_violation else ()),
            -joint_score,
            *(-value for value in role_scores),
            tuple(value.model_rank for value in rows),
            tuple(value.option_id for value in rows),
        )
        feasible.append(
            (
                tie_key,
                rows,
                role_scores,
                joint_score,
                diversity,
                family_count,
                locus_count,
                phenotype_count,
                administered,
                dose_assessment,
            )
        )
    if not feasible:
        raise ValueError(
            "slate has no feasible calibrated four-role assignment"
        )
    (
        _,
        rows,
        role_scores,
        joint_score,
        diversity,
        family_count,
        locus_count,
        phenotype_count,
        administered,
        dose_assessment,
    ) = min(feasible, key=lambda value: value[0])
    # Feasibility is invariant to member order and is therefore cached by
    # unordered subset above.  The durable dose receipt is not: its member
    # bindings include the final rank.  Reissue that receipt over the exact
    # four-role order that will cross the decision trust boundary.
    if request.memory_dose_contract is not None:
        dose_assessment = assess_allocated_slate_memory_dose(
            request,
            tuple(member_by_id[row.option_id] for row in rows),
        )
        if not dose_assessment.passed:  # Defensive after subset feasibility.
            raise RuntimeError(
                "winning structural-posterior assignment violated memory dose"
            )
    selected = tuple(
        StructuralPosteriorAllocatedMember(
            role=role,
            option_id=row.option_id,
            option_identity_sha256=row.option_identity_sha256,
            model_rank=row.model_rank,
            role_score=score,
        )
        for role, row, score in zip(_ROLES, rows, role_scores)
    )
    return _StructuralPosteriorAssignment(
        selected=selected,
        joint_score=joint_score,
        diversity_score=diversity,
        distinct_family_count=family_count,
        distinct_locus_count=locus_count,
        distinct_phenotype_count=phenotype_count,
        administered_card_keys=administered,
        feasible_role_assignment_count=len(feasible),
        memory_dose_assessment=dose_assessment,
        family_exposure_violation_count=(
            sum(
                max(
                    bound.minimum_evaluations
                    - sum(
                        member_by_id[row.option_id].family == bound.family
                        for row in rows
                    ),
                    0,
                    sum(
                        member_by_id[row.option_id].family == bound.family
                        for row in rows
                    )
                    - bound.maximum_evaluations,
                )
                for bound in required_family_bounds
            )
        ),
    )


def _validate_request(request: SlateAllocationRequest) -> None:
    if len(request.slate.members) != _SLATE_SIZE:
        raise ValueError("structural-posterior policy requires exactly eight members")
    if request.portfolio_size != _PORTFOLIO_SIZE:
        raise ValueError("structural-posterior policy requires four evaluations")
    snapshot = request.calibration_snapshot
    if snapshot is None:
        raise ValueError("structural-posterior policy requires a calibration snapshot")
    if snapshot.cutoff_wave_index_exclusive > request.slate.wave_index:
        raise ValueError("calibration snapshot cutoff reaches beyond current wave")
    if any(
        value.prediction.wave_index >= request.slate.wave_index
        for value in snapshot.observations
    ):
        raise ValueError("calibration snapshot contains current-wave outcome evidence")


def _validate_structural_decision(
    value: Any,
    *,
    required_family_minimums: tuple[tuple[str, int], ...] = (),
    required_family_bounds: tuple[FamilyExposureBound, ...] = (),
    minimize_family_bound_violation: bool = False,
) -> _StructuralPosteriorAssignment:
    if required_family_minimums and required_family_bounds:
        raise ValueError("family minima and family bounds are mutually exclusive")
    if required_family_minimums:
        _validate_required_family_minimums(required_family_minimums)
    if required_family_bounds:
        FamilyExposurePhase(0, required_family_bounds).__post_init__()
    if type(value.request) is not SlateAllocationRequest:
        raise TypeError("request must be exact SlateAllocationRequest")
    value.request.revalidate()
    _validate_request(value.request)
    if type(value.score_rows) is not tuple or any(
        type(row) is not StructuralPosteriorMemberScoreRow
        for row in value.score_rows
    ):
        raise TypeError("score_rows must contain exact member score rows")
    for row in value.score_rows:
        row.__post_init__()
    if type(value.selected) is not tuple or any(
        type(row) is not StructuralPosteriorAllocatedMember
        for row in value.selected
    ):
        raise TypeError("selected must contain exact allocated members")
    for row in value.selected:
        row.__post_init__()
    if value.memory_dose_assessment is not None:
        if type(value.memory_dose_assessment) is not PortfolioMemoryDoseAssessment:
            raise TypeError("memory_dose_assessment must be exact or None")
        value.memory_dose_assessment.__post_init__()
    expected_rows = tuple(
        _score_member(value.request, member)
        for member in value.request.slate.members
    )
    if value.score_rows != expected_rows:
        raise ValueError("score rows differ from sealed prior evidence")
    expected = _best_assignment(
        value.request,
        expected_rows,
        required_family_minimums=required_family_minimums,
        required_family_bounds=required_family_bounds,
        minimize_family_bound_violation=minimize_family_bound_violation,
    )
    observed_values = (
        value.selected,
        value.joint_score,
        value.diversity_score,
        value.distinct_family_count,
        value.distinct_locus_count,
        value.distinct_phenotype_count,
        value.administered_card_keys,
        value.feasible_role_assignment_count,
        value.memory_dose_assessment,
    )
    expected_values = (
        expected.selected,
        expected.joint_score,
        expected.diversity_score,
        expected.distinct_family_count,
        expected.distinct_locus_count,
        expected.distinct_phenotype_count,
        expected.administered_card_keys,
        expected.feasible_role_assignment_count,
        expected.memory_dose_assessment,
    )
    if observed_values != expected_values:
        raise ValueError("decision differs from exact structural-posterior allocation")
    return expected


@dataclass(frozen=True, slots=True, eq=False)
class StructuralPosteriorSlateDecision:
    """Replayable receipt for one outcome-blind structural-posterior decision."""

    request: SlateAllocationRequest
    score_rows: tuple[StructuralPosteriorMemberScoreRow, ...]
    selected: tuple[StructuralPosteriorAllocatedMember, ...]
    joint_score: float
    diversity_score: float
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    feasible_role_assignment_count: int
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        _validate_structural_decision(self)

    def revalidate(self) -> None:
        if type(self) is not StructuralPosteriorSlateDecision:
            raise TypeError("decision must be exact StructuralPosteriorSlateDecision")
        StructuralPosteriorSlateDecision.__post_init__(self)

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
            "event_type": "structural_posterior_four_role_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "request": self.request.to_record(),
            "request_sha256": self.request.request_sha256,
            "calibration_snapshot_sha256": (
                self.request.calibration_snapshot.snapshot_sha256
                if self.request.calibration_snapshot is not None
                else None
            ),
            "prior_only": self.prior_only,
            "roles": [value.value for value in _ROLES],
            "score_rows": [value.to_record() for value in self.score_rows],
            "selected": [value.to_record() for value in self.selected],
            "joint_score_hex": self.joint_score.hex(),
            "diversity_score_hex": self.diversity_score.hex(),
            "distinct_family_count": self.distinct_family_count,
            "distinct_locus_count": self.distinct_locus_count,
            "distinct_phenotype_count": self.distinct_phenotype_count,
            "administered_card_keys": list(self.administered_card_keys),
            "feasible_role_assignment_count": self.feasible_role_assignment_count,
            **(
                {}
                if self.memory_dose_assessment is None
                else {
                    "memory_dose_assessment": self.memory_dose_assessment.to_record()
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
            type(other) is StructuralPosteriorSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class OperatorStratifiedStructuralPosteriorSlateDecision(
    StructuralPosteriorSlateDecision
):
    """Structural-posterior decision with authenticated assay minima."""

    required_family_minimums: tuple[tuple[str, int], ...] = (
        (COMPOSITE_OPTION_FAMILY, 1),
    )

    policy_id: ClassVar[str] = OPERATOR_STRATIFIED_POLICY_ID
    policy_version: ClassVar[int] = OPERATOR_STRATIFIED_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        OPERATOR_STRATIFIED_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if type(self) is not OperatorStratifiedStructuralPosteriorSlateDecision:
            raise TypeError("decision must be exact operator-stratified decision")
        _validate_required_family_minimums(self.required_family_minimums)
        _validate_structural_decision(
            self,
            required_family_minimums=self.required_family_minimums,
        )

    def revalidate(self) -> None:
        if type(self) is not OperatorStratifiedStructuralPosteriorSlateDecision:
            raise TypeError("decision must be exact operator-stratified decision")
        OperatorStratifiedStructuralPosteriorSlateDecision.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        record = StructuralPosteriorSlateDecision._unsigned_record(self)
        return {
            **record,
            "schema_version": 2,
            "event_type": "operator_stratified_structural_posterior_allocated",
            "operator_stratification": {
                "required_family_minimums": [
                    {"family": family, "minimum_evaluations": minimum}
                    for family, minimum in self.required_family_minimums
                ],
                "semantics": "assay_exposure_not_quality_prior",
                "outcomes_consulted": False,
            },
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(
            _OPERATOR_STRATIFIED_DECISION_DOMAIN,
            self._unsigned_record(),
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperatorStratifiedStructuralPosteriorSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class HorizonBoundedStructuralPosteriorSlateDecision(
    StructuralPosteriorSlateDecision
):
    """Replayable allocation under an authenticated finite-horizon schedule."""

    family_exposure_phases: tuple[FamilyExposurePhase, ...] = ()
    family_exposure_violation_count: int = 0

    policy_id: ClassVar[str] = HORIZON_BOUNDED_POLICY_ID
    policy_version: ClassVar[int] = HORIZON_BOUNDED_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        HORIZON_BOUNDED_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if type(self) is not HorizonBoundedStructuralPosteriorSlateDecision:
            raise TypeError("decision must be exact horizon-bounded decision")
        _validate_family_exposure_phases(self.family_exposure_phases)
        active = _active_family_exposure_phase(
            self.family_exposure_phases,
            wave_index=self.request.slate.wave_index,
        )
        expected = _validate_structural_decision(
            self,
            required_family_bounds=active.bounds,
            minimize_family_bound_violation=True,
        )
        if (
            type(self.family_exposure_violation_count) is not int
            or self.family_exposure_violation_count < 0
        ):
            raise ValueError(
                "family_exposure_violation_count must be a non-negative integer"
            )
        if self.family_exposure_violation_count != (
            expected.family_exposure_violation_count
        ):
            raise ValueError("family exposure recourse differs from exact replay")

    def revalidate(self) -> None:
        if type(self) is not HorizonBoundedStructuralPosteriorSlateDecision:
            raise TypeError("decision must be exact horizon-bounded decision")
        HorizonBoundedStructuralPosteriorSlateDecision.__post_init__(self)

    @property
    def active_exposure_phase(self) -> FamilyExposurePhase:
        self.revalidate()
        return _active_family_exposure_phase(
            self.family_exposure_phases,
            wave_index=self.request.slate.wave_index,
        )

    @property
    def policy_configuration_sha256(self) -> str:
        self.revalidate()
        return _hash(
            _HORIZON_BOUNDED_CONFIGURATION_DOMAIN,
            _horizon_bounded_configuration_record(self.family_exposure_phases),
        )

    def _unsigned_record(self) -> dict[str, object]:
        record = StructuralPosteriorSlateDecision._unsigned_record(self)
        active = self.active_exposure_phase
        family_by_id = {
            value.option_id: value.family for value in self.request.slate.members
        }
        applied_counts = {
            bound.family: sum(
                family_by_id[value.option_id] == bound.family
                for value in self.selected
            )
            for bound in active.bounds
        }
        return {
            **record,
            "schema_version": 3,
            "event_type": "horizon_bounded_structural_posterior_allocated",
            "policy_configuration_sha256": self.policy_configuration_sha256,
            "finite_horizon_exposure": {
                "family_exposure_phases": [
                    value.to_record() for value in self.family_exposure_phases
                ],
                "active_phase": active.to_record(),
                "applied_family_counts": applied_counts,
                "family_exposure_violation_count": (
                    self.family_exposure_violation_count
                ),
                "requested_bounds_satisfied": (
                    self.family_exposure_violation_count == 0
                ),
                "structural_recourse": "minimum_l1_violation_then_role_score",
                "phase_authority": "sealed_wave_index",
                "semantics": "assay_exposure_not_quality_prior",
                "outcomes_consulted": False,
            },
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(
            _HORIZON_BOUNDED_DECISION_DOMAIN,
            self._unsigned_record(),
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is HorizonBoundedStructuralPosteriorSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class StructuralPosteriorSlatePolicy:
    """Allocate K8 to K4 by exact prior-only role assignment."""

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        # The v1 mechanism deliberately has no workload-tuned parameters.
        if type(self) is not StructuralPosteriorSlatePolicy:
            raise TypeError("policy must be exact StructuralPosteriorSlatePolicy")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _hash(_CONFIGURATION_DOMAIN, _configuration_record())

    def select(
        self,
        request: SlateAllocationRequest,
    ) -> StructuralPosteriorSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        _validate_request(request)
        rows = tuple(_score_member(request, member) for member in request.slate.members)
        assignment = _best_assignment(request, rows)
        return StructuralPosteriorSlateDecision(
            request=request,
            score_rows=rows,
            selected=assignment.selected,
            joint_score=assignment.joint_score,
            diversity_score=assignment.diversity_score,
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            feasible_role_assignment_count=(
                assignment.feasible_role_assignment_count
            ),
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
        }


@dataclass(frozen=True, slots=True)
class OperatorStratifiedStructuralPosteriorSlatePolicy:
    """Reserve evaluator assay exposure while preserving prior-only ranking."""

    required_family_minimums: tuple[tuple[str, int], ...] = (
        (COMPOSITE_OPTION_FAMILY, 1),
    )

    policy_id: ClassVar[str] = OPERATOR_STRATIFIED_POLICY_ID
    policy_version: ClassVar[int] = OPERATOR_STRATIFIED_POLICY_VERSION
    definition_sha256: ClassVar[str] = (
        OPERATOR_STRATIFIED_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if type(self) is not OperatorStratifiedStructuralPosteriorSlatePolicy:
            raise TypeError("policy must be exact operator-stratified policy")
        _validate_required_family_minimums(self.required_family_minimums)

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _OPERATOR_STRATIFIED_CONFIGURATION_DOMAIN,
            self._configuration_record(),
        )

    def _configuration_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "base_allocator": {
                "policy_id": POLICY_ID,
                "policy_version": POLICY_VERSION,
                "definition_sha256": POLICY_DEFINITION_SHA256,
                "configuration_sha256": _hash(
                    _CONFIGURATION_DOMAIN,
                    _configuration_record(),
                ),
            },
            "required_family_minimums": [
                {"family": family, "minimum_evaluations": minimum}
                for family, minimum in self.required_family_minimums
            ],
            "constraint_semantics": "assay_exposure_not_quality_prior",
            "outcomes_consulted": False,
            "benchmark_specific_parameters": [],
        }

    def select(
        self,
        request: SlateAllocationRequest,
    ) -> OperatorStratifiedStructuralPosteriorSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        _validate_request(request)
        rows = tuple(_score_member(request, member) for member in request.slate.members)
        assignment = _best_assignment(
            request,
            rows,
            required_family_minimums=self.required_family_minimums,
        )
        return OperatorStratifiedStructuralPosteriorSlateDecision(
            request=request,
            score_rows=rows,
            selected=assignment.selected,
            joint_score=assignment.joint_score,
            diversity_score=assignment.diversity_score,
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            feasible_role_assignment_count=(
                assignment.feasible_role_assignment_count
            ),
            memory_dose_assessment=assignment.memory_dose_assessment,
            required_family_minimums=self.required_family_minimums,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration": self._configuration_record(),
            "configuration_sha256": self.configuration_sha256,
        }


@dataclass(frozen=True, slots=True)
class HorizonBoundedStructuralPosteriorSlatePolicy:
    """Apply pre-registered action-family exposure bounds by campaign phase."""

    family_exposure_phases: tuple[FamilyExposurePhase, ...]

    policy_id: ClassVar[str] = HORIZON_BOUNDED_POLICY_ID
    policy_version: ClassVar[int] = HORIZON_BOUNDED_POLICY_VERSION
    definition_sha256: ClassVar[str] = HORIZON_BOUNDED_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("policy must be exact horizon-bounded policy")
        _validate_family_exposure_phases(self.family_exposure_phases)

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _HORIZON_BOUNDED_CONFIGURATION_DOMAIN,
            self._configuration_record(),
        )

    def _configuration_record(self) -> dict[str, object]:
        return _horizon_bounded_configuration_record(self.family_exposure_phases)

    def exposure_phase_for_wave(self, wave_index: int) -> FamilyExposurePhase:
        self.__post_init__()
        return _active_family_exposure_phase(
            self.family_exposure_phases,
            wave_index=wave_index,
        )

    def select(
        self,
        request: SlateAllocationRequest,
    ) -> HorizonBoundedStructuralPosteriorSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        _validate_request(request)
        active = self.exposure_phase_for_wave(request.slate.wave_index)
        rows = tuple(_score_member(request, member) for member in request.slate.members)
        assignment = _best_assignment(
            request,
            rows,
            required_family_bounds=active.bounds,
            minimize_family_bound_violation=True,
        )
        return HorizonBoundedStructuralPosteriorSlateDecision(
            request=request,
            score_rows=rows,
            selected=assignment.selected,
            joint_score=assignment.joint_score,
            diversity_score=assignment.diversity_score,
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            feasible_role_assignment_count=(
                assignment.feasible_role_assignment_count
            ),
            memory_dose_assessment=assignment.memory_dose_assessment,
            family_exposure_phases=self.family_exposure_phases,
            family_exposure_violation_count=(
                assignment.family_exposure_violation_count
            ),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration": self._configuration_record(),
            "configuration_sha256": self.configuration_sha256,
        }


__all__ = [
    "FamilyExposureBound",
    "FamilyExposurePhase",
    "HORIZON_BOUNDED_POLICY_DEFINITION_SHA256",
    "HORIZON_BOUNDED_POLICY_ID",
    "HORIZON_BOUNDED_POLICY_VERSION",
    "HorizonBoundedStructuralPosteriorSlateDecision",
    "HorizonBoundedStructuralPosteriorSlatePolicy",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "OPERATOR_STRATIFIED_POLICY_DEFINITION_SHA256",
    "OPERATOR_STRATIFIED_POLICY_ID",
    "OPERATOR_STRATIFIED_POLICY_VERSION",
    "OperatorStratifiedStructuralPosteriorSlateDecision",
    "OperatorStratifiedStructuralPosteriorSlatePolicy",
    "StructuralPosteriorAllocatedMember",
    "StructuralPosteriorMemberScoreRow",
    "StructuralPosteriorMetricScoreRow",
    "StructuralPosteriorSlateDecision",
    "StructuralPosteriorSlatePolicy",
    "StructuralPosteriorSlateRole",
    "build_controller_owned_family_exposure_phases",
    "build_terminal_tapered_family_exposure_phases",
]
