"""Prequential calibration for current-prefix archive opportunity.

The raw language-model forecast remains useful semantic evidence, but it is
not assumed to be a calibrated numerical distribution.  This module exposes a
small application port that converts a raw conditional archive-opportunity
forecast into a prior-only predictive distribution and an abstention decision.

All features are workload opaque: evolutionary stage, model/expert lane,
typed operator, native rank, prior score, forecast reliability, and observed
prefix scale.  Objective names, workload identifiers, simulator fields,
provider names, and current eligible-candidate outcomes are absent.

The default empirical implementation uses:

* archive-prefix normalization for cross-workload scale transport;
* log residuals for multiplicative forecast error;
* a hierarchy of stage/lane/operator calibration cells;
* a prequential Spearman skill gate;
* support-distance abstention;
* a finite-sample residual lower quantile; and
* a two-part lower expected-opportunity bound that separates probability of a
  positive contribution from its positive magnitude.

This is recommendation evidence only.  The protected incumbent remains the
selection authority whenever the calibrator abstains.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256


PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_ID = (
    "hierarchical_prequential_archive_opportunity"
)
PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OBSERVATION_DOMAIN = (
    b"agent-evolve:archive-opportunity-calibration-observation:v1\x00"
)
_CONTEXT_DOMAIN = (
    b"agent-evolve:archive-opportunity-calibration-context:v1\x00"
)
_RESULT_DOMAIN = (
    b"agent-evolve:archive-opportunity-calibration-result:v1\x00"
)
_SNAPSHOT_DOMAIN = (
    b"agent-evolve:archive-opportunity-calibration-snapshot:v1\x00"
)
_DEFINITION_DOMAIN = (
    b"agent-evolve:archive-opportunity-calibration-policy:v1\x00"
)


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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_probability(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must lie in [0, 1]")


def _require_nonnegative(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _nearest_rank_quantile(
    values: tuple[float, ...],
    probability: float,
) -> float:
    """Return the deterministic finite-sample nearest-rank quantile."""

    if not values:
        raise ValueError("a quantile requires at least one observation")
    _require_probability(probability, name="probability")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("quantile values must be finite exact floats")
    ordered = tuple(sorted(values))
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


def _rank(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[cursor]]
        ):
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            result[order[position]] = average
        cursor = end
    return tuple(result)


def _pearson(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left)
    )
    right_norm = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right)
    )
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(numerator / (left_norm * right_norm))


def _spearman(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _wilson_lower_bound(
    *,
    positive_count: int,
    observation_count: int,
    z_value: float,
) -> float:
    if (
        type(positive_count) is not int
        or type(observation_count) is not int
        or not 0 <= positive_count <= observation_count
        or observation_count <= 0
    ):
        raise ValueError("Wilson counts are invalid")
    if type(z_value) is not float or not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z_value must be a positive finite float")
    probability = positive_count / observation_count
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / observation_count
    centre = probability + z_squared / (2.0 * observation_count)
    radius = z_value * math.sqrt(
        (
            probability * (1.0 - probability)
            + z_squared / (4.0 * observation_count)
        )
        / observation_count
    )
    return float(max(0.0, (centre - radius) / denominator))


@dataclass(frozen=True, slots=True)
class ArchiveOpportunityActionContext:
    """Outcome-blind generic context for one candidate opportunity."""

    action_sha256: str
    decision_index: int
    lane_id: str
    operator_id: str
    native_rank: int
    lane_size: int
    prior_score: float
    parent_generated_in_current_run: bool
    context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be positive")
        _require_token(self.lane_id, name="lane_id")
        _require_token(self.operator_id, name="operator_id")
        if (
            type(self.native_rank) is not int
            or type(self.lane_size) is not int
            or self.native_rank <= 0
            or self.lane_size <= 0
            or self.native_rank > self.lane_size
        ):
            raise ValueError("native rank must fit the positive lane size")
        _require_probability(self.prior_score, name="prior_score")
        if type(self.parent_generated_in_current_run) is not bool:
            raise TypeError(
                "parent_generated_in_current_run must be exact"
            )
        object.__setattr__(
            self,
            "context_sha256",
            _hash(_CONTEXT_DOMAIN, self._unsigned_record()),
        )

    @property
    def rank_quality(self) -> float:
        if self.lane_size == 1:
            return 1.0
        return 1.0 - (
            (self.native_rank - 1) / float(self.lane_size - 1)
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "decision_index": self.decision_index,
            "lane_id": self.lane_id,
            "operator_id": self.operator_id,
            "native_rank": self.native_rank,
            "lane_size": self.lane_size,
            "rank_quality_hex": self.rank_quality.hex(),
            "prior_score_hex": self.prior_score.hex(),
            "parent_generated_in_current_run": (
                self.parent_generated_in_current_run
            ),
            "workload_objective_provider_prompt_fields_present": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArchiveOpportunityCalibrationRequest:
    """One raw current-prefix forecast submitted for calibration."""

    context: ArchiveOpportunityActionContext
    forecast_reliability: float
    raw_adverse_gain: float
    raw_central_gain: float
    raw_favorable_gain: float
    raw_acquisition_value: float
    prefix_gain: float
    prefix_action_count: int

    def __post_init__(self) -> None:
        if type(self.context) is not ArchiveOpportunityActionContext:
            raise TypeError("context must be exact")
        self.context.__post_init__()
        _require_probability(
            self.forecast_reliability,
            name="forecast_reliability",
        )
        for name in (
            "raw_adverse_gain",
            "raw_central_gain",
            "raw_favorable_gain",
            "raw_acquisition_value",
            "prefix_gain",
        ):
            _require_nonnegative(getattr(self, name), name=name)
        if (
            type(self.prefix_action_count) is not int
            or self.prefix_action_count <= 0
        ):
            raise ValueError("prefix_action_count must be positive")
        if self.raw_adverse_gain > self.raw_favorable_gain:
            raise ValueError("adverse gain cannot exceed favorable gain")

    @property
    def prefix_mean_gain(self) -> float:
        return self.prefix_gain / self.prefix_action_count

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "context": self.context.to_record(),
            "forecast_reliability_hex": self.forecast_reliability.hex(),
            "raw_adverse_gain_hex": self.raw_adverse_gain.hex(),
            "raw_central_gain_hex": self.raw_central_gain.hex(),
            "raw_favorable_gain_hex": self.raw_favorable_gain.hex(),
            "raw_acquisition_value_hex": (
                self.raw_acquisition_value.hex()
            ),
            "prefix_gain_hex": self.prefix_gain.hex(),
            "prefix_action_count": self.prefix_action_count,
            "prefix_mean_gain_hex": self.prefix_mean_gain.hex(),
            "eligible_candidate_outcomes_observed": False,
        }


class ArchiveOpportunityCalibrationEvidenceRole(str, Enum):
    """How the forecast target entered the authenticated evidence stream."""

    AUTHORITATIVE_SELECTED = "authoritative_selected"
    SAME_PREFIX_PAIRED_AUTHORITATIVE = (
        "same_prefix_paired_authoritative"
    )
    SAME_PREFIX_PAIRED_COUNTERFACTUAL = (
        "same_prefix_paired_counterfactual"
    )


@dataclass(frozen=True, slots=True)
class ArchiveOpportunityCalibrationObservation:
    """One forecast joined to a later exact conditional-gain observation."""

    request: ArchiveOpportunityCalibrationRequest
    realized_conditional_gain: float
    decision_sha256: str
    outcome_sha256: str
    evidence_cutoff_ordinal: int
    evidence_role: ArchiveOpportunityCalibrationEvidenceRole = (
        ArchiveOpportunityCalibrationEvidenceRole.AUTHORITATIVE_SELECTED
    )
    sampling_propensity: float = 1.0
    paired_observation_sha256: str | None = None
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.request) is not ArchiveOpportunityCalibrationRequest:
            raise TypeError("request must be exact")
        self.request.__post_init__()
        _require_nonnegative(
            self.realized_conditional_gain,
            name="realized_conditional_gain",
        )
        require_sha256(self.decision_sha256, "decision_sha256")
        require_sha256(self.outcome_sha256, "outcome_sha256")
        if (
            type(self.evidence_cutoff_ordinal) is not int
            or self.evidence_cutoff_ordinal <= 0
        ):
            raise ValueError("evidence_cutoff_ordinal must be positive")
        if (
            type(self.evidence_role)
            is not ArchiveOpportunityCalibrationEvidenceRole
        ):
            raise TypeError("evidence_role must be exact")
        if (
            type(self.sampling_propensity) is not float
            or not math.isfinite(self.sampling_propensity)
            or not 0.0 < self.sampling_propensity <= 1.0
        ):
            raise ValueError(
                "sampling_propensity must be a finite positive probability"
            )
        if self.paired_observation_sha256 is not None:
            require_sha256(
                self.paired_observation_sha256,
                "paired_observation_sha256",
            )
        is_legacy_schema = (
            self.evidence_role
            is (
                ArchiveOpportunityCalibrationEvidenceRole
                .AUTHORITATIVE_SELECTED
            )
            and self.sampling_propensity == 1.0
            and self.paired_observation_sha256 is None
        )
        if (
            not is_legacy_schema
            and self.paired_observation_sha256 is None
        ):
            raise ValueError(
                "paired calibration evidence requires its observation identity"
            )
        object.__setattr__(
            self,
            "observation_sha256",
            _hash(_OBSERVATION_DOMAIN, self._unsigned_record()),
        )

    @property
    def positive(self) -> bool:
        return self.realized_conditional_gain > 0.0

    def _unsigned_record(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "request": self.request.to_record(),
            "realized_conditional_gain_hex": (
                self.realized_conditional_gain.hex()
            ),
            "positive": self.positive,
            "decision_sha256": self.decision_sha256,
            "outcome_sha256": self.outcome_sha256,
            "evidence_cutoff_ordinal": self.evidence_cutoff_ordinal,
            "current_or_future_candidate_outcomes_used": False,
        }
        if (
            self.evidence_role
            is not (
                ArchiveOpportunityCalibrationEvidenceRole
                .AUTHORITATIVE_SELECTED
            )
            or self.sampling_propensity != 1.0
            or self.paired_observation_sha256 is not None
        ):
            result.update(
                {
                    "schema_version": 2,
                    "evidence_role": self.evidence_role.value,
                    "sampling_propensity_hex": (
                        self.sampling_propensity.hex()
                    ),
                    "paired_observation_sha256": (
                        self.paired_observation_sha256
                    ),
                    "same_prefix_counterfactual_is_not_archive_"
                    "publication": (
                        self.evidence_role
                        is (
                            ArchiveOpportunityCalibrationEvidenceRole
                            .SAME_PREFIX_PAIRED_COUNTERFACTUAL
                        )
                    ),
                }
            )
        return result

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArchiveOpportunityCalibrationResult:
    """Authenticated calibrated opportunity and abstention evidence."""

    action_sha256: str
    calibration_id: str
    calibration_version: int
    calibration_definition_sha256: str
    observation_snapshot_sha256: str
    selected_stratum: str
    stratum_support_count: int
    skill_support_count: int
    prequential_rank_skill: float | None
    raw_normalized_acquisition: float
    minimum_supported_normalized_acquisition: float | None
    maximum_supported_normalized_acquisition: float | None
    support_log_distance: float
    positive_count: int
    posterior_positive_probability: float
    lower_positive_probability: float
    conformal_lower_gain: float
    lower_expected_gain: float
    calibrated_acquisition_value: float
    calibrated_upper_gain: float
    abstained: bool
    abstention_reason: str | None
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_token(self.calibration_id, name="calibration_id")
        if (
            type(self.calibration_version) is not int
            or self.calibration_version <= 0
        ):
            raise ValueError("calibration_version must be positive")
        require_sha256(
            self.calibration_definition_sha256,
            "calibration_definition_sha256",
        )
        require_sha256(
            self.observation_snapshot_sha256,
            "observation_snapshot_sha256",
        )
        _require_token(self.selected_stratum, name="selected_stratum")
        for name in (
            "stratum_support_count",
            "skill_support_count",
            "positive_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.positive_count > self.stratum_support_count:
            raise ValueError("positive_count exceeds stratum support")
        if self.prequential_rank_skill is not None and (
            type(self.prequential_rank_skill) is not float
            or not math.isfinite(self.prequential_rank_skill)
            or not -1.0 <= self.prequential_rank_skill <= 1.0
        ):
            raise ValueError("prequential_rank_skill must lie in [-1,1]")
        for name in (
            "raw_normalized_acquisition",
            "support_log_distance",
            "conformal_lower_gain",
            "lower_expected_gain",
            "calibrated_acquisition_value",
            "calibrated_upper_gain",
        ):
            _require_nonnegative(getattr(self, name), name=name)
        for name in (
            "minimum_supported_normalized_acquisition",
            "maximum_supported_normalized_acquisition",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative(value, name=name)
        if (
            self.minimum_supported_normalized_acquisition is None
        ) != (
            self.maximum_supported_normalized_acquisition is None
        ):
            raise ValueError("support bounds must be jointly present")
        _require_probability(
            self.posterior_positive_probability,
            name="posterior_positive_probability",
        )
        _require_probability(
            self.lower_positive_probability,
            name="lower_positive_probability",
        )
        if (
            self.lower_positive_probability
            > self.posterior_positive_probability
        ):
            raise ValueError("lower probability exceeds posterior mean")
        if self.abstained:
            if self.abstention_reason is None:
                raise ValueError("abstention requires a reason")
            _require_token(
                self.abstention_reason,
                name="abstention_reason",
            )
        elif self.abstention_reason is not None:
            raise ValueError(
                "a calibrated recommendation cannot have an abstention reason"
            )
        object.__setattr__(
            self,
            "result_sha256",
            _hash(_RESULT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "calibration": {
                "calibration_id": self.calibration_id,
                "calibration_version": self.calibration_version,
                "definition_sha256": (
                    self.calibration_definition_sha256
                ),
            },
            "observation_snapshot_sha256": (
                self.observation_snapshot_sha256
            ),
            "selected_stratum": self.selected_stratum,
            "stratum_support_count": self.stratum_support_count,
            "skill_support_count": self.skill_support_count,
            "prequential_rank_skill_hex": (
                None
                if self.prequential_rank_skill is None
                else self.prequential_rank_skill.hex()
            ),
            "raw_normalized_acquisition_hex": (
                self.raw_normalized_acquisition.hex()
            ),
            "minimum_supported_normalized_acquisition_hex": (
                None
                if self.minimum_supported_normalized_acquisition is None
                else self.minimum_supported_normalized_acquisition.hex()
            ),
            "maximum_supported_normalized_acquisition_hex": (
                None
                if self.maximum_supported_normalized_acquisition is None
                else self.maximum_supported_normalized_acquisition.hex()
            ),
            "support_log_distance_hex": self.support_log_distance.hex(),
            "positive_count": self.positive_count,
            "posterior_positive_probability_hex": (
                self.posterior_positive_probability.hex()
            ),
            "lower_positive_probability_hex": (
                self.lower_positive_probability.hex()
            ),
            "conformal_lower_gain_hex": self.conformal_lower_gain.hex(),
            "lower_expected_gain_hex": self.lower_expected_gain.hex(),
            "calibrated_acquisition_value_hex": (
                self.calibrated_acquisition_value.hex()
            ),
            "calibrated_upper_gain_hex": (
                self.calibrated_upper_gain.hex()
            ),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "eligible_candidate_outcomes_observed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "result_sha256": self.result_sha256,
        }


@runtime_checkable
class ArchiveOpportunityCalibrationPort(Protocol):
    """Stable inverted API for prior-only opportunity calibration."""

    calibration_id: str
    calibration_version: int
    definition_sha256: str

    def calibrate(
        self,
        request: ArchiveOpportunityCalibrationRequest,
    ) -> ArchiveOpportunityCalibrationResult: ...


def validate_archive_opportunity_calibration_port(
    value: ArchiveOpportunityCalibrationPort,
) -> tuple[str, int, str]:
    if not isinstance(value, ArchiveOpportunityCalibrationPort):
        raise TypeError(
            "calibration must implement ArchiveOpportunityCalibrationPort"
        )
    identity = (
        value.calibration_id,
        value.calibration_version,
        value.definition_sha256,
    )
    _require_token(identity[0], name="calibration_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("calibration_version must be positive")
    require_sha256(identity[2], "calibration definition_sha256")
    return identity


@dataclass(frozen=True, slots=True)
class HierarchicalPrequentialArchiveOpportunityCalibration:
    """Calibrate raw opportunity using a sealed prior observation snapshot."""

    observations: tuple[ArchiveOpportunityCalibrationObservation, ...]
    maximum_evidence_cutoff_ordinal: int
    minimum_global_support: int = 6
    minimum_cell_support: int = 4
    minimum_rank_support: int = 4
    minimum_rank_skill: float = 0.0
    residual_lower_probability: float = 0.1
    residual_upper_probability: float = 0.9
    positive_prior_alpha: float = 1.0
    positive_prior_beta: float = 1.0
    wilson_z_value: float = 1.0
    maximum_support_log_distance: float = 1.0
    minimum_lower_expected_gain: float = 0.0
    scale_floor: float = 1e-15
    calibration_id: str = (
        PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_ID
    )
    calibration_version: int = (
        PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_VERSION
    )
    observation_snapshot_sha256: str = field(init=False)
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.observations) is not tuple
            or any(
                type(value)
                is not ArchiveOpportunityCalibrationObservation
                for value in self.observations
            )
        ):
            raise TypeError("observations must contain exact observations")
        for value in self.observations:
            value.__post_init__()
        hashes = tuple(value.observation_sha256 for value in self.observations)
        if hashes != tuple(sorted(set(hashes))):
            raise ValueError(
                "observations must be unique and hash canonical"
            )
        if (
            type(self.maximum_evidence_cutoff_ordinal) is not int
            or self.maximum_evidence_cutoff_ordinal <= 0
        ):
            raise ValueError(
                "maximum_evidence_cutoff_ordinal must be positive"
            )
        if any(
            value.evidence_cutoff_ordinal
            > self.maximum_evidence_cutoff_ordinal
            for value in self.observations
        ):
            raise ValueError(
                "observation crosses the sealed evidence cutoff"
            )
        for name in (
            "minimum_global_support",
            "minimum_cell_support",
            "minimum_rank_support",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.minimum_rank_skill) is not float
            or not math.isfinite(self.minimum_rank_skill)
            or not -1.0 <= self.minimum_rank_skill <= 1.0
        ):
            raise ValueError("minimum_rank_skill must lie in [-1,1]")
        for name in (
            "residual_lower_probability",
            "residual_upper_probability",
        ):
            _require_probability(getattr(self, name), name=name)
        if (
            self.residual_lower_probability
            >= self.residual_upper_probability
        ):
            raise ValueError("residual quantile order is invalid")
        for name in (
            "positive_prior_alpha",
            "positive_prior_beta",
            "wilson_z_value",
            "maximum_support_log_distance",
            "scale_floor",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        _require_nonnegative(
            self.minimum_lower_expected_gain,
            name="minimum_lower_expected_gain",
        )
        _require_token(self.calibration_id, name="calibration_id")
        if (
            self.calibration_id
            != PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_ID
            or self.calibration_version
            != PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_VERSION
        ):
            raise ValueError("calibration identity is immutable")
        snapshot = _hash(
            _SNAPSHOT_DOMAIN,
            {
                "schema_version": 1,
                "maximum_evidence_cutoff_ordinal": (
                    self.maximum_evidence_cutoff_ordinal
                ),
                "observation_sha256s": list(hashes),
                "forbidden_feature_fields": [
                    "workload_id",
                    "objective_id",
                    "provider_id",
                    "prompt_id",
                ],
            },
        )
        object.__setattr__(
            self,
            "observation_snapshot_sha256",
            snapshot,
        )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "calibration_id": self.calibration_id,
                    "calibration_version": self.calibration_version,
                    "observation_snapshot_sha256": snapshot,
                    "minimum_global_support": self.minimum_global_support,
                    "minimum_cell_support": self.minimum_cell_support,
                    "minimum_rank_support": self.minimum_rank_support,
                    "minimum_rank_skill_hex": (
                        self.minimum_rank_skill.hex()
                    ),
                    "residual_lower_probability_hex": (
                        self.residual_lower_probability.hex()
                    ),
                    "residual_upper_probability_hex": (
                        self.residual_upper_probability.hex()
                    ),
                    "positive_prior_alpha_hex": (
                        self.positive_prior_alpha.hex()
                    ),
                    "positive_prior_beta_hex": (
                        self.positive_prior_beta.hex()
                    ),
                    "wilson_z_value_hex": self.wilson_z_value.hex(),
                    "maximum_support_log_distance_hex": (
                        self.maximum_support_log_distance.hex()
                    ),
                    "minimum_lower_expected_gain_hex": (
                        self.minimum_lower_expected_gain.hex()
                    ),
                    "scale_floor_hex": self.scale_floor.hex(),
                    "normalization": (
                        "raw_and_realized_gain_over_observed_prefix_mean_gain"
                    ),
                    "residual": "log1p_realized_minus_log1p_raw",
                    "strata": [
                        "stage_lane_operator",
                        "stage_operator",
                        "stage",
                        "global",
                    ],
                    "skill": "prequential_spearman_raw_vs_realized",
                    "lower_bound": (
                        "wilson_positive_probability_times_positive_"
                        "residual_lower_magnitude"
                    ),
                    "eligible_candidate_outcomes_observed": False,
                    "workload_objective_provider_prompt_branches": False,
                },
            ),
        )

    def _scale(
        self,
        request: ArchiveOpportunityCalibrationRequest,
    ) -> float:
        return max(self.scale_floor, request.prefix_mean_gain)

    def _rows(
        self,
        request: ArchiveOpportunityCalibrationRequest,
    ) -> tuple[
        str,
        tuple[ArchiveOpportunityCalibrationObservation, ...],
    ]:
        context = request.context
        candidates = (
            (
                "stage_lane_operator",
                tuple(
                    value
                    for value in self.observations
                    if (
                        value.request.context.decision_index
                        == context.decision_index
                        and value.request.context.lane_id
                        == context.lane_id
                        and value.request.context.operator_id
                        == context.operator_id
                    )
                ),
            ),
            (
                "stage_operator",
                tuple(
                    value
                    for value in self.observations
                    if (
                        value.request.context.decision_index
                        == context.decision_index
                        and value.request.context.operator_id
                        == context.operator_id
                    )
                ),
            ),
            (
                "stage",
                tuple(
                    value
                    for value in self.observations
                    if value.request.context.decision_index
                    == context.decision_index
                ),
            ),
        )
        for name, rows in candidates:
            if len(rows) >= self.minimum_cell_support:
                return name, rows
        return "global", self.observations

    def _normalized(
        self,
        observation: ArchiveOpportunityCalibrationObservation,
    ) -> tuple[float, float]:
        scale = self._scale(observation.request)
        return (
            observation.request.raw_acquisition_value / scale,
            observation.realized_conditional_gain / scale,
        )

    def calibrate(
        self,
        request: ArchiveOpportunityCalibrationRequest,
    ) -> ArchiveOpportunityCalibrationResult:
        self.__post_init__()
        if type(request) is not ArchiveOpportunityCalibrationRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        stratum, rows = self._rows(request)
        stage_rows = tuple(
            value
            for value in self.observations
            if (
                value.request.context.decision_index
                == request.context.decision_index
            )
        )
        skill_rows = (
            stage_rows
            if len(stage_rows) >= self.minimum_rank_support
            else self.observations
        )
        rank_skill = (
            _spearman(
                tuple(
                    self._normalized(value)[0]
                    for value in skill_rows
                ),
                tuple(
                    self._normalized(value)[1]
                    for value in skill_rows
                ),
            )
            if len(skill_rows) >= self.minimum_rank_support
            else None
        )
        scale = self._scale(request)
        raw_normalized = request.raw_acquisition_value / scale

        if len(self.observations) < self.minimum_global_support:
            abstention_reason = "insufficient_global_support"
        elif len(rows) < self.minimum_cell_support:
            abstention_reason = "insufficient_cell_support"
        elif (
            rank_skill is None
            or len(skill_rows) < self.minimum_rank_support
        ):
            abstention_reason = "insufficient_rank_skill_support"
        elif rank_skill <= self.minimum_rank_skill:
            abstention_reason = "nonpositive_prequential_rank_skill"
        else:
            abstention_reason = None

        normalized_rows = tuple(self._normalized(value) for value in rows)
        supported_raw = tuple(value[0] for value in normalized_rows)
        minimum_supported = min(supported_raw) if supported_raw else None
        maximum_supported = max(supported_raw) if supported_raw else None
        support_log_distance = 0.0
        if minimum_supported is not None and maximum_supported is not None:
            log_raw = math.log1p(raw_normalized)
            log_minimum = math.log1p(minimum_supported)
            log_maximum = math.log1p(maximum_supported)
            support_log_distance = max(
                0.0,
                log_minimum - log_raw,
                log_raw - log_maximum,
            )
            if (
                abstention_reason is None
                and support_log_distance
                > self.maximum_support_log_distance
            ):
                abstention_reason = "forecast_scale_out_of_support"

        residuals = tuple(
            math.log1p(realized) - math.log1p(raw)
            for raw, realized in normalized_rows
        )
        positive_rows = tuple(
            (raw, realized)
            for raw, realized in normalized_rows
            if realized > 0.0
        )
        positive_count = len(positive_rows)
        posterior_positive = (
            self.positive_prior_alpha + positive_count
        ) / (
            self.positive_prior_alpha
            + self.positive_prior_beta
            + len(rows)
        )
        lower_positive = (
            _wilson_lower_bound(
                positive_count=positive_count,
                observation_count=len(rows),
                z_value=self.wilson_z_value,
            )
            if rows
            else 0.0
        )

        if residuals:
            lower_residual = _nearest_rank_quantile(
                residuals,
                self.residual_lower_probability,
            )
            conformal_lower_normalized = max(
                0.0,
                math.expm1(
                    math.log1p(raw_normalized) + lower_residual
                ),
            )
        else:
            conformal_lower_normalized = 0.0

        if positive_rows:
            positive_residuals = tuple(
                math.log1p(realized) - math.log1p(raw)
                for raw, realized in positive_rows
            )
            lower_positive_residual = _nearest_rank_quantile(
                positive_residuals,
                self.residual_lower_probability,
            )
            median_positive_residual = _nearest_rank_quantile(
                positive_residuals,
                0.5,
            )
            upper_positive_residual = _nearest_rank_quantile(
                positive_residuals,
                self.residual_upper_probability,
            )
            positive_lower_normalized = max(
                0.0,
                math.expm1(
                    math.log1p(raw_normalized)
                    + lower_positive_residual
                ),
            )
            positive_median_normalized = max(
                0.0,
                math.expm1(
                    math.log1p(raw_normalized)
                    + median_positive_residual
                ),
            )
            positive_upper_normalized = max(
                0.0,
                math.expm1(
                    math.log1p(raw_normalized)
                    + upper_positive_residual
                ),
            )
        else:
            positive_lower_normalized = 0.0
            positive_median_normalized = 0.0
            positive_upper_normalized = 0.0
            if abstention_reason is None:
                abstention_reason = "no_positive_calibration_support"

        conformal_lower_gain = scale * conformal_lower_normalized
        lower_expected_gain = (
            scale * lower_positive * positive_lower_normalized
        )
        calibrated_acquisition = (
            scale
            * posterior_positive
            * positive_median_normalized
        )
        calibrated_upper = scale * positive_upper_normalized
        if (
            abstention_reason is None
            and lower_expected_gain
            <= self.minimum_lower_expected_gain
        ):
            abstention_reason = "no_calibrated_lower_opportunity"

        return ArchiveOpportunityCalibrationResult(
            action_sha256=request.context.action_sha256,
            calibration_id=self.calibration_id,
            calibration_version=self.calibration_version,
            calibration_definition_sha256=self.definition_sha256,
            observation_snapshot_sha256=(
                self.observation_snapshot_sha256
            ),
            selected_stratum=stratum,
            stratum_support_count=len(rows),
            skill_support_count=len(skill_rows),
            prequential_rank_skill=rank_skill,
            raw_normalized_acquisition=float(raw_normalized),
            minimum_supported_normalized_acquisition=(
                None
                if minimum_supported is None
                else float(minimum_supported)
            ),
            maximum_supported_normalized_acquisition=(
                None
                if maximum_supported is None
                else float(maximum_supported)
            ),
            support_log_distance=float(support_log_distance),
            positive_count=positive_count,
            posterior_positive_probability=float(
                posterior_positive
            ),
            lower_positive_probability=float(lower_positive),
            conformal_lower_gain=float(conformal_lower_gain),
            lower_expected_gain=float(lower_expected_gain),
            calibrated_acquisition_value=float(
                calibrated_acquisition
            ),
            calibrated_upper_gain=float(calibrated_upper),
            abstained=abstention_reason is not None,
            abstention_reason=abstention_reason,
        )


__all__ = [
    "ArchiveOpportunityActionContext",
    "ArchiveOpportunityCalibrationEvidenceRole",
    "ArchiveOpportunityCalibrationObservation",
    "ArchiveOpportunityCalibrationPort",
    "ArchiveOpportunityCalibrationRequest",
    "ArchiveOpportunityCalibrationResult",
    "HierarchicalPrequentialArchiveOpportunityCalibration",
    "PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_ID",
    "PREQUENTIAL_ARCHIVE_OPPORTUNITY_CALIBRATION_VERSION",
    "validate_archive_opportunity_calibration_port",
]
