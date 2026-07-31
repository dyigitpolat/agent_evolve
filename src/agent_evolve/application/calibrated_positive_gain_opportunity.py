"""Calibrated positive-gain opportunity without adverse-scenario gating.

The forecast-opportunity challenger this module replaces required positive
gain in the ADVERSE scenario before it would recommend anything, so every
uncertain candidate abstained and uncertainty was mapped to zero acquisition
authority.  This policy removes hard abstention: every eligible candidate
receives a defined score, which may be non-positive, and the horizon policy
decides exploit-versus-explore.

Evidence combination:

* forecast geometry — scenario quantile points are valued strictly against
  the CURRENT ARCHIVE through an injected gain port, so parent-relative
  improvement contributes nothing;
* prequential conversion evidence — per (engine, rank-band) conversion rates
  from outcomes observed BEFORE the current decision, with hierarchical
  Beta shrinkage cell -> engine -> global; and
* a configurable mixture of the two (log-odds for probabilities, arithmetic
  for magnitudes).

``score = p_positive_archive_gain * expected_positive_gain
          - lambda_ * tail_risk
          + beta * (future_seats_remaining / horizon_total)
            * value_of_information``

Value of information is exactly zero when no future seat remains, preserving
the V7 terminal semantics that a terminal seat never purchases information.
The policy knows no workload, objective name, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.rank_balanced_causal_pilot import (
    rank_band_index,
)
from agent_evolve.domain.patch import require_sha256


CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_ID = (
    "calibrated_positive_gain_opportunity"
)
CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_VERSION = 1
LOWER_QUANTILE_SCENARIO_ID = "p10"
CENTRAL_QUANTILE_SCENARIO_ID = "p50"
UPPER_QUANTILE_SCENARIO_ID = "p90"
_QUANTILE_SCENARIO_IDS = (
    LOWER_QUANTILE_SCENARIO_ID,
    CENTRAL_QUANTILE_SCENARIO_ID,
    UPPER_QUANTILE_SCENARIO_ID,
)
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:calibrated-positive-gain-policy:v1\x00"
)
_ARCHIVE_DOMAIN = (
    b"agent-evolve:calibrated-positive-gain-archive:v1\x00"
)
_SCORE_DOMAIN = b"agent-evolve:calibrated-positive-gain-score:v1\x00"
_RANKING_DOMAIN = (
    b"agent-evolve:calibrated-positive-gain-ranking:v1\x00"
)

ObjectivePoint = tuple[tuple[str, float], ...]


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


def _require_objective_point(value: ObjectivePoint, *, name: str) -> None:
    if (
        type(value) is not tuple
        or not value
        or value != tuple(sorted(value))
    ):
        raise ValueError(f"{name} must be non-empty and canonical")
    metric_ids: list[str] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name} must contain metric pairs")
        metric_id, metric_value = item
        _require_token(metric_id, name=f"{name} metric_id")
        if type(metric_value) is not float or not math.isfinite(metric_value):
            raise TypeError(f"{name} values must be finite exact floats")
        metric_ids.append(metric_id)
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"{name} repeats a metric")


def _objective_point_record(value: ObjectivePoint) -> list[dict[str, object]]:
    return [
        {"metric_id": metric_id, "value_hex": metric_value.hex()}
        for metric_id, metric_value in value
    ]


def _point_values(
    point: ObjectivePoint,
    metric_ids: tuple[str, ...],
) -> tuple[float, ...]:
    mapping = dict(point)
    if tuple(sorted(mapping)) != metric_ids:
        raise ValueError("objective point uses a foreign metric frame")
    return tuple(mapping[metric_id] for metric_id in metric_ids)


def non_dominated(
    points: tuple[ObjectivePoint, ...],
    metric_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Exact minimization Pareto filter over an archive."""

    raw = sorted({_point_values(point, metric_ids) for point in points})
    return tuple(
        point
        for point in raw
        if not any(
            other != point
            and all(
                o <= p for o, p in zip(other, point, strict=True)
            )
            for other in raw
        )
    )


def chebyshev_excess(
    anchor: ObjectivePoint,
    archive_points: tuple[ObjectivePoint, ...],
    metric_ids: tuple[str, ...],
) -> float:
    """Worst-axis improvement the anchor already holds over the front.

    Minimization convention.  For each non-dominated archive point the
    anchor's worst-axis excess is ``max_j(anchor_j - front_j)``; the returned
    value is the smallest such excess over the front.  It is <= 0 exactly
    when the anchor weakly dominates some front point (the anchor is at
    least as good on every axis), and it grows with how much the anchor
    would have to improve on its worst axis to reach the nearest front
    point.  The archive is Pareto-filtered first: dominated archive points
    would otherwise lower the excess of anchors that no front point can
    reach.

    Scale-free in the sense that matters here: it lives in whatever frame
    the archive lives in, and the policy consumes only its ORDER.
    """

    if not archive_points:
        raise ValueError("archive_points must be non-empty")
    values = _point_values(anchor, metric_ids)
    best: float | None = None
    for other in non_dominated(archive_points, metric_ids):
        worst = max(a - b for a, b in zip(values, other, strict=True))
        if best is None or worst < best:
            best = worst
    return float(best if best is not None else 0.0)


@runtime_checkable
class ArchiveConditionedGainPort(Protocol):
    """Value one hypothetical objective point against an archive.

    Implementations own objective senses, normalization, and reference
    points.  The returned value must be the exact non-negative marginal
    archive utility of adding the point; a point dominated by the archive
    must return exactly zero.  Parents are never visible to this port, so
    parent-relative improvement cannot leak into opportunity.
    """

    utility_id: str
    utility_version: int
    definition_sha256: str

    def marginal_archive_gain(
        self,
        archive_points: tuple[ObjectivePoint, ...],
        objective_point: ObjectivePoint,
    ) -> float: ...


def validate_archive_conditioned_gain_port(
    value: ArchiveConditionedGainPort,
) -> tuple[str, int, str]:
    if not isinstance(value, ArchiveConditionedGainPort):
        raise TypeError(
            "gain port must implement ArchiveConditionedGainPort"
        )
    identity = (
        value.utility_id,
        value.utility_version,
        value.definition_sha256,
    )
    _require_token(identity[0], name="gain utility_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("gain utility_version must be positive")
    require_sha256(identity[2], "gain utility definition_sha256")
    return identity


@dataclass(frozen=True, slots=True)
class PositiveGainForecast:
    """Sealed scenario-quantile forecast for one candidate."""

    quantile_points: tuple[tuple[str, ObjectivePoint], ...]
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.quantile_points) is not tuple
            or tuple(value[0] for value in self.quantile_points)
            != tuple(sorted(_QUANTILE_SCENARIO_IDS))
        ):
            raise ValueError(
                "quantile_points must cover exactly the canonical "
                "p10/p50/p90 scenarios"
            )
        frames = set()
        for scenario_id, point in self.quantile_points:
            _require_objective_point(
                point,
                name=f"{scenario_id} objective point",
            )
            frames.add(tuple(metric_id for metric_id, _value in point))
        if len(frames) != 1:
            raise ValueError(
                "all quantile points must share one objective frame"
            )
        if (
            type(self.reliability) is not float
            or not math.isfinite(self.reliability)
            or not 0.0 <= self.reliability <= 1.0
        ):
            raise ValueError("reliability must lie in [0, 1]")

    @classmethod
    def from_parent_and_deltas(
        cls,
        *,
        parent_point: ObjectivePoint,
        quantile_deltas: tuple[tuple[str, tuple[float, ...]], ...],
        reliability: float = 1.0,
    ) -> PositiveGainForecast:
        """Absolute scenario points from a parent point plus per-metric deltas.

        A proposal-time self-report is a per-metric scenario DELTA against
        the candidate's parent, so the absolute scenario point is the parent
        point plus that delta.  ``quantile_deltas`` pairs each metric with
        one delta per canonical scenario, ordered exactly as
        ``LOWER/CENTRAL/UPPER``; the deltas must live in the same frame as
        ``parent_point``, and the arithmetic is a pure translation, so any
        affine renormalization of the frame commutes with this constructor.

        The only reason a self-report cannot become a forecast is missing
        evidence: a candidate with no parent, or a report that does not cover
        every metric, has no absolute point and must stay forecast-free.  The
        caller decides that by not calling this.
        """

        _require_objective_point(parent_point, name="parent_point")
        if type(quantile_deltas) is not tuple:
            raise TypeError("quantile_deltas must be an exact tuple")
        deltas: dict[str, tuple[float, ...]] = {}
        for item in quantile_deltas:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "quantile_deltas must pair a metric with its deltas"
                )
            metric_id, scenario_deltas = item
            _require_token(metric_id, name="quantile delta metric_id")
            if (
                type(scenario_deltas) is not tuple
                or len(scenario_deltas) != len(_QUANTILE_SCENARIO_IDS)
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in scenario_deltas
                )
            ):
                raise TypeError(
                    "each metric needs one finite delta per scenario"
                )
            if metric_id in deltas:
                raise ValueError("quantile_deltas repeat a metric")
            deltas[metric_id] = scenario_deltas
        parent = dict(parent_point)
        if tuple(sorted(deltas)) != tuple(sorted(parent)):
            raise ValueError(
                "quantile_deltas must cover the parent's exact frame"
            )
        return cls(
            quantile_points=tuple(
                (
                    scenario_id,
                    tuple(
                        sorted(
                            (
                                metric_id,
                                parent[metric_id]
                                + deltas[metric_id][index],
                            )
                            for metric_id in parent
                        )
                    ),
                )
                for index, scenario_id in enumerate(
                    _QUANTILE_SCENARIO_IDS
                )
            ),
            reliability=float(reliability),
        )

    def point(self, scenario_id: str) -> ObjectivePoint:
        for value_id, point in self.quantile_points:
            if value_id == scenario_id:
                return point
        raise ValueError("forecast omits the requested scenario")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "quantile_points": [
                {
                    "scenario_id": scenario_id,
                    "objective_point": _objective_point_record(point),
                }
                for scenario_id, point in self.quantile_points
            ],
            "reliability_hex": self.reliability.hex(),
        }


@dataclass(frozen=True, slots=True)
class PositiveGainCandidate:
    """Outcome-blind view of one unevaluated eligible candidate."""

    action_sha256: str
    engine_id: str
    native_rank: int
    lane_size: int
    forecast: PositiveGainForecast | None = None
    frozen_score: float | None = None
    #: The objective point this action was derived from, known at proposal
    #: time and never an outcome of this action.  Absent when the action has
    #: no parent geometry (a global acquisition lane, say).
    anchor_point: ObjectivePoint | None = None

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_token(self.engine_id, name="engine_id")
        if (
            type(self.native_rank) is not int
            or type(self.lane_size) is not int
            or self.native_rank <= 0
            or self.lane_size <= 0
            or self.native_rank > self.lane_size
        ):
            raise ValueError("native rank must fit the positive lane size")
        if self.forecast is not None:
            if type(self.forecast) is not PositiveGainForecast:
                raise TypeError("forecast must be exact or None")
            self.forecast.__post_init__()
        if self.frozen_score is not None and (
            type(self.frozen_score) is not float
            or not math.isfinite(self.frozen_score)
            or not 0.0 <= self.frozen_score <= 1.0
        ):
            raise ValueError("frozen_score must lie in [0, 1] or be None")
        if self.anchor_point is not None:
            _require_objective_point(self.anchor_point, name="anchor_point")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "engine_id": self.engine_id,
            "native_rank": self.native_rank,
            "lane_size": self.lane_size,
            "forecast": (
                None
                if self.forecast is None
                else self.forecast.to_record()
            ),
            "frozen_score_hex": (
                None
                if self.frozen_score is None
                else self.frozen_score.hex()
            ),
            "anchor_point": (
                None
                if self.anchor_point is None
                else _objective_point_record(self.anchor_point)
            ),
        }


@dataclass(frozen=True, slots=True)
class ObservedConversionOutcome:
    """One real outcome observed strictly before the current decision."""

    observation_ordinal: int
    engine_id: str
    native_rank: int
    lane_size: int
    feasible: bool
    marginal_archive_gain: float

    def __post_init__(self) -> None:
        if (
            type(self.observation_ordinal) is not int
            or self.observation_ordinal <= 0
        ):
            raise ValueError("observation_ordinal must be positive")
        _require_token(self.engine_id, name="engine_id")
        if (
            type(self.native_rank) is not int
            or type(self.lane_size) is not int
            or self.native_rank <= 0
            or self.lane_size <= 0
            or self.native_rank > self.lane_size
        ):
            raise ValueError("native rank must fit the positive lane size")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be exact")
        if (
            type(self.marginal_archive_gain) is not float
            or not math.isfinite(self.marginal_archive_gain)
            or self.marginal_archive_gain < 0.0
        ):
            raise ValueError(
                "marginal_archive_gain must be finite and non-negative"
            )
        if not self.feasible and self.marginal_archive_gain != 0.0:
            raise ValueError("an infeasible outcome cannot contribute gain")

    @property
    def positive(self) -> bool:
        return self.marginal_archive_gain > 0.0

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "observation_ordinal": self.observation_ordinal,
            "engine_id": self.engine_id,
            "native_rank": self.native_rank,
            "lane_size": self.lane_size,
            "feasible": self.feasible,
            "positive": self.positive,
            "marginal_archive_gain_hex": (
                self.marginal_archive_gain.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class CalibratedPositiveGainScore:
    """Complete score card for one eligible candidate; never abstains."""

    action_sha256: str
    p_feasible: float
    p_positive_archive_gain: float
    expected_positive_gain: float
    tail_risk: float
    value_of_information: float
    uncertainty: float
    effective_sample_size: float
    score: float
    forecast_probability: float | None
    forecast_magnitude: float | None
    forecast_nondominated_fraction: float | None
    conversion_probability: float
    conversion_magnitude: float
    frozen_score_probability: float | None
    frozen_evidence_weight: float
    #: Chebyshev excess of the candidate's anchor over the current
    #: archive front; smaller means the parent sits nearer the front.
    anchor_excess: float | None = None
    #: Smallest max-norm distance from the candidate's anchor to any anchor
    #: this market already bought; ``inf`` when nothing is bought yet.
    anchor_dispersion: float | None = None
    score_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        for name in ("p_feasible", "p_positive_archive_gain"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "expected_positive_gain",
            "tail_risk",
            "value_of_information",
            "uncertainty",
            "conversion_magnitude",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            type(self.effective_sample_size) is not float
            or not math.isfinite(self.effective_sample_size)
            or self.effective_sample_size <= 0.0
        ):
            raise ValueError("effective_sample_size must be positive")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise ValueError("score must be a finite float")
        forecast_fields = (
            self.forecast_probability,
            self.forecast_magnitude,
            self.forecast_nondominated_fraction,
        )
        if any(value is None for value in forecast_fields) != all(
            value is None for value in forecast_fields
        ):
            raise ValueError(
                "forecast components must be jointly present or absent"
            )
        for value in forecast_fields:
            if value is not None and (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    "forecast components must be finite and non-negative"
                )
        if (
            type(self.conversion_probability) is not float
            or not math.isfinite(self.conversion_probability)
            or not 0.0 <= self.conversion_probability <= 1.0
        ):
            raise ValueError("conversion_probability must lie in [0, 1]")
        if self.frozen_score_probability is not None and (
            type(self.frozen_score_probability) is not float
            or not math.isfinite(self.frozen_score_probability)
            or not 0.0 <= self.frozen_score_probability <= 1.0
        ):
            raise ValueError(
                "frozen_score_probability must lie in [0, 1] or be None"
            )
        if (
            type(self.frozen_evidence_weight) is not float
            or not math.isfinite(self.frozen_evidence_weight)
            or not 0.0 <= self.frozen_evidence_weight <= 1.0
        ):
            raise ValueError("frozen_evidence_weight must lie in [0, 1]")
        if (
            self.frozen_score_probability is None
            and self.frozen_evidence_weight != 0.0
        ):
            raise ValueError(
                "frozen evidence weight requires a frozen score"
            )
        if self.anchor_excess is not None and (
            type(self.anchor_excess) is not float
            or not math.isfinite(self.anchor_excess)
        ):
            raise ValueError("anchor_excess must be finite or None")
        if self.anchor_dispersion is not None and (
            type(self.anchor_dispersion) is not float
            or math.isnan(self.anchor_dispersion)
            or self.anchor_dispersion < 0.0
        ):
            raise ValueError(
                "anchor_dispersion must be non-negative or None"
            )
        if (self.anchor_excess is None) != (
            self.anchor_dispersion is None
        ):
            raise ValueError(
                "anchor geometry must be jointly present or absent"
            )
        object.__setattr__(
            self,
            "score_sha256",
            _hash(_SCORE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        # The anchor keys are emitted only when the archive-geometry
        # channel is active, so a score card produced with the channel off
        # keeps its pre-existing bytes and hash exactly.
        anchor: dict[str, object] = (
            {}
            if self.anchor_excess is None
            else {
                "anchor_excess_hex": self.anchor_excess.hex(),
                "anchor_dispersion_hex": (
                    "inf"
                    if math.isinf(self.anchor_dispersion)
                    else self.anchor_dispersion.hex()
                ),
            }
        )
        return {
            **anchor,
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "p_feasible_hex": self.p_feasible.hex(),
            "p_positive_archive_gain_hex": (
                self.p_positive_archive_gain.hex()
            ),
            "expected_positive_gain_hex": (
                self.expected_positive_gain.hex()
            ),
            "tail_risk_hex": self.tail_risk.hex(),
            "value_of_information_hex": (
                self.value_of_information.hex()
            ),
            "uncertainty_hex": self.uncertainty.hex(),
            "effective_sample_size_hex": (
                self.effective_sample_size.hex()
            ),
            "score_hex": self.score.hex(),
            "forecast_probability_hex": (
                None
                if self.forecast_probability is None
                else self.forecast_probability.hex()
            ),
            "forecast_magnitude_hex": (
                None
                if self.forecast_magnitude is None
                else self.forecast_magnitude.hex()
            ),
            "forecast_nondominated_fraction_hex": (
                None
                if self.forecast_nondominated_fraction is None
                else self.forecast_nondominated_fraction.hex()
            ),
            "conversion_probability_hex": (
                self.conversion_probability.hex()
            ),
            "conversion_magnitude_hex": (
                self.conversion_magnitude.hex()
            ),
            "frozen_score_probability_hex": (
                None
                if self.frozen_score_probability is None
                else self.frozen_score_probability.hex()
            ),
            "frozen_evidence_weight_hex": (
                self.frozen_evidence_weight.hex()
            ),
            "abstained": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "score_sha256": self.score_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibratedPositiveGainRanking:
    """One cutoff-bound, always-defined ranking of eligible candidates."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    archive_sha256: str
    future_seats_remaining: int
    horizon_total: int
    scores: tuple[CalibratedPositiveGainScore, ...]
    ranked_action_sha256s: tuple[str, ...]
    ranking_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        require_sha256(self.archive_sha256, "archive_sha256")
        if (
            type(self.future_seats_remaining) is not int
            or type(self.horizon_total) is not int
            or self.horizon_total <= 0
            or not 0 <= self.future_seats_remaining <= self.horizon_total
        ):
            raise ValueError(
                "future seats must fit the positive horizon total"
            )
        if (
            type(self.scores) is not tuple
            or not self.scores
            or any(
                type(value) is not CalibratedPositiveGainScore
                for value in self.scores
            )
        ):
            raise TypeError("scores must contain exact score cards")
        for value in self.scores:
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in self.scores)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("scores must be action-canonical")
        if (
            type(self.ranked_action_sha256s) is not tuple
            or tuple(sorted(self.ranked_action_sha256s)) != action_ids
        ):
            raise ValueError(
                "ranking must order the exact eligible market"
            )
        object.__setattr__(
            self,
            "ranking_sha256",
            _hash(_RANKING_DOMAIN, self._unsigned_record()),
        )

    def score_for(self, action_sha256: str) -> CalibratedPositiveGainScore:
        for value in self.scores:
            if value.action_sha256 == action_sha256:
                return value
        raise ValueError("action is outside the ranked market")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "archive_sha256": self.archive_sha256,
            "future_seats_remaining": self.future_seats_remaining,
            "horizon_total": self.horizon_total,
            "score_sha256s": [
                value.score_sha256 for value in self.scores
            ],
            "ranked_action_sha256s": list(self.ranked_action_sha256s),
            "hard_abstention": False,
            "eligible_candidate_outcomes_observed": False,
        }

    def to_record(self, *, include_scores: bool = False) -> dict[str, object]:
        self.__post_init__()
        result = {
            **self._unsigned_record(),
            "ranking_sha256": self.ranking_sha256,
        }
        if include_scores:
            result["scores"] = [
                value.to_record() for value in self.scores
            ]
        return result


def _beta_posterior_mean(
    *,
    prior_mean: float,
    prior_strength: float,
    successes: float,
    failures: float,
) -> float:
    return (prior_strength * prior_mean + successes) / (
        prior_strength + successes + failures
    )


def _shrunk_magnitude(
    *,
    prior_mean: float,
    prior_strength: float,
    positive_gain_sum: float,
    positive_count: float,
) -> float:
    return (prior_strength * prior_mean + positive_gain_sum) / (
        prior_strength + positive_count
    )


@dataclass(frozen=True, slots=True)
class _ConversionCellEvidence:
    observation_count: float
    positive_count: float
    feasible_count: float
    positive_gain_sum: float


@dataclass(frozen=True, slots=True)
class CalibratedPositiveGainOpportunityPolicy:
    """Rank every eligible candidate by calibrated positive archive gain."""

    archive_gain_utility: ArchiveConditionedGainPort = field(
        repr=False,
        compare=False,
    )
    lambda_: float = 1.0
    beta: float = 1.0
    mixture_weight: float = 0.5
    prior_strength: float = 2.0
    band_count: int = 3
    reference_gain_scale: float = 1.0e-4
    root_prior_probability: float = 0.5
    probability_floor: float = 1.0e-6
    scenario_weights: tuple[float, float, float] = (0.25, 0.5, 0.25)
    # The V70 census measured the frozen cross-campaign score as
    # uninformative (pooled Spearman with positivity about -0.01) while
    # native rank was informative, so frozen evidence is a WEAK log-odds
    # feature: a small default weight, and no effect at all unless the
    # frozen fit's training history covers enough runs.
    frozen_score_weight: float = 0.125
    frozen_score_minimum_training_runs: int = 10
    # When cell posteriors tie exactly (no forecasts, shared evidence
    # cell), break ties by the engine's own native-rank order instead
    # of the arbitrary action hash.  Off by default so the r1 ranking
    # and definition hash are preserved bit-for-bit.
    within_cell_rank_tie_break: bool = False
    # ARCHIVE-GEOMETRY TIE-BREAK.  The jul28 diagnosis measured that with
    # forecasts absent, every probability input to this score is a per
    # (engine, rank-band) CELL constant: the score took 1.4 to 2.8 distinct
    # values over markets of 37 to 64 members, 46-83% of every market tied
    # with the argmax, and 23 of 28 live seats were therefore chosen by the
    # TIE-BREAK and not by the score.  The tie-break in use was native-rank
    # quality, a measured non-feature (AUC 0.552), and the seats it bought
    # were geometrically redundant: the panel converted only 30% of the
    # individual gain it purchased into union gain, against a uniform
    # draw's 68%.
    #
    # This flag replaces that tie-break with archive geometry, in the exact
    # place the decision is actually taken.  Among candidates the score
    # cannot separate, prefer, in order:
    #
    #   1. the anchor FARTHEST from the anchors this market already bought
    #      (max-min Chebyshev dispersion).  Provably submodular: a second
    #      seat on an identical anchor has dispersion exactly zero and is
    #      taken last, which is the redundancy the census measured; and
    #   2. failing that -- nothing bought yet, or an exact dispersion tie --
    #      the anchor NEAREST the current archive front (smallest Chebyshev
    #      excess), which is the strongest outcome-blind predictor measured
    #      (pooled AUC 0.732 held-out, 0.700 on the replay corpus).
    #
    # Both quantities are pure archive geometry consumed as an ORDER only:
    # no metric name, no unit, no absolute threshold, no workload constant,
    # and invariant to any monotone rescaling of the objective frame.  A
    # candidate with no anchor takes the market's median position on both,
    # so an anchorless lane is never systematically evicted or protected.
    # Off by default so the r1 and r2 rankings and definition hashes are
    # preserved bit-for-bit.
    anchor_geometry_tie_break: bool = False
    # DOWNSIDE TAIL RISK.  The forecast branch already prices risk as an
    # expected SHORTFALL -- ``sum_s w_s * max(0, central - gain_s)`` -- which
    # is zero when the scenarios agree and never scales with the magnitude on
    # its own.  The conversion branch instead priced it as ``(1 - p) * M``,
    # which is not a shortfall: substituting it into
    # ``score = p*M - lambda*tail_risk`` collapses the score to
    # ``M * (2p - 1)`` at ``lambda = 1``, so the magnitude channel is
    # INVERTED for every candidate with ``p < 0.5`` and a cell that has
    # demonstrated larger gains scores lower.  The jul28 census measured that
    # exposure at 18.1% of multiplier scored candidates.
    #
    # This flag applies the forecast branch's own definition to the
    # conversion branch's own two-point scenario set -- gain ``M`` with
    # probability ``p``, gain ``0`` with probability ``1 - p``, central
    # outcome the expected gain ``p*M`` -- which evaluates in closed form to
    # ``p * (1 - p) * M``.  The score becomes ``p*M*(1 - lambda*(1 - p))``,
    # non-decreasing in both ``p`` and ``M`` at ``lambda <= 1``: risk still
    # penalises an uncertain candidate, but it can no longer reverse the
    # sign of the magnitude channel.  No new constant, no threshold, and the
    # forecast branch's shortfall is untouched.  Off by default so the r1,
    # r2 and r3 rankings and definition hashes are preserved bit-for-bit.
    downside_shortfall_tail_risk: bool = False
    policy_id: str = CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_ID
    policy_version: int = (
        CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        utility_identity = validate_archive_conditioned_gain_port(
            self.archive_gain_utility
        )
        for name in ("lambda_", "beta"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "mixture_weight",
            "root_prior_probability",
            "frozen_score_weight",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            type(self.frozen_score_minimum_training_runs) is not int
            or self.frozen_score_minimum_training_runs < 0
        ):
            raise ValueError(
                "frozen_score_minimum_training_runs must be non-negative"
            )
        for name in (
            "within_cell_rank_tie_break",
            "anchor_geometry_tie_break",
            "downside_shortfall_tail_risk",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exact")
        for name in ("prior_strength", "reference_gain_scale"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if type(self.band_count) is not int or self.band_count <= 0:
            raise ValueError("band_count must be positive")
        if (
            type(self.probability_floor) is not float
            or not math.isfinite(self.probability_floor)
            or not 0.0 < self.probability_floor < 0.5
        ):
            raise ValueError("probability_floor must lie in (0, 0.5)")
        if (
            type(self.scenario_weights) is not tuple
            or len(self.scenario_weights) != 3
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
                for value in self.scenario_weights
            )
            or math.fsum(self.scenario_weights) != 1.0
        ):
            raise ValueError(
                "scenario_weights must be three positive floats "
                "summing to exactly one"
            )
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_id
            != CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_ID
            or self.policy_version
            != CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_VERSION
        ):
            raise ValueError("policy identity is immutable")
        definition = {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "archive_gain_utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "lambda_hex": self.lambda_.hex(),
                    "beta_hex": self.beta.hex(),
                    "mixture_weight_hex": self.mixture_weight.hex(),
                    "prior_strength_hex": self.prior_strength.hex(),
                    "band_count": self.band_count,
                    "reference_gain_scale_hex": (
                        self.reference_gain_scale.hex()
                    ),
                    "root_prior_probability_hex": (
                        self.root_prior_probability.hex()
                    ),
                    "probability_floor_hex": (
                        self.probability_floor.hex()
                    ),
                    "scenario_weights_hex": [
                        value.hex() for value in self.scenario_weights
                    ],
                    "scenario_ids": list(_QUANTILE_SCENARIO_IDS),
                    "frozen_score_weight_hex": (
                        self.frozen_score_weight.hex()
                    ),
                    "frozen_score_minimum_training_runs": (
                        self.frozen_score_minimum_training_runs
                    ),
                    "frozen_score_evidence": (
                        "weak_log_odds_feature_gated_by_training_"
                        "run_floor_probability_only"
                    ),
                    "forecast_geometry": (
                        "archive_conditioned_scenario_quantile_"
                        "nondomination_and_positive_gain"
                    ),
                    "conversion_evidence": (
                        "prequential_engine_rank_band_hierarchical_"
                        "beta_shrinkage"
                    ),
                    "probability_mixture": "log_odds_blend",
                    "magnitude_mixture": "arithmetic_blend",
                    "hard_abstention": False,
                    "terminal_value_of_information": 0.0,
                    "parent_relative_improvement_counts": False,
                    "eligible_candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
        }
        # Emitted only when active so the r1 definition hash is
        # preserved bit-for-bit.
        if self.within_cell_rank_tie_break:
            definition["within_cell_tie_break"] = (
                "native_rank_quality_then_action_sha256"
            )
        if self.anchor_geometry_tie_break:
            definition["anchor_geometry_tie_break"] = (
                "max_min_chebyshev_dispersion_from_bought_anchors_then_"
                "chebyshev_excess_over_current_archive_front"
            )
        if self.downside_shortfall_tail_risk:
            definition["conversion_tail_risk"] = (
                "expected_shortfall_below_the_conversion_branch_own_"
                "expected_gain"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(_DEFINITION_DOMAIN, definition),
        )

    def _clamped(self, probability: float) -> float:
        return min(
            max(probability, self.probability_floor),
            1.0 - self.probability_floor,
        )

    def _conversion_evidence(
        self,
        observed_outcomes: tuple[ObservedConversionOutcome, ...],
    ) -> dict[object, _ConversionCellEvidence]:
        if type(observed_outcomes) is not tuple or any(
            type(value) is not ObservedConversionOutcome
            for value in observed_outcomes
        ):
            raise TypeError(
                "observed_outcomes must contain exact conversion outcomes"
            )
        ordinals = tuple(
            value.observation_ordinal for value in observed_outcomes
        )
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError(
                "observed outcomes must use unique ascending ordinals"
            )
        counts: dict[object, list[float]] = {}
        for value in observed_outcomes:
            value.__post_init__()
            band = rank_band_index(
                value.native_rank,
                value.lane_size,
                self.band_count,
            )
            for key in (
                None,
                value.engine_id,
                (value.engine_id, band),
            ):
                row = counts.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                row[0] += 1.0
                row[1] += float(value.positive)
                row[2] += float(value.feasible)
                row[3] += (
                    value.marginal_archive_gain
                    if value.positive
                    else 0.0
                )
        return {
            key: _ConversionCellEvidence(
                observation_count=row[0],
                positive_count=row[1],
                feasible_count=row[2],
                positive_gain_sum=row[3],
            )
            for key, row in counts.items()
        }

    def _hierarchical_estimates(
        self,
        *,
        evidence: dict[object, _ConversionCellEvidence],
        engine_id: str,
        band: int,
    ) -> dict[str, float]:
        empty = _ConversionCellEvidence(0.0, 0.0, 0.0, 0.0)
        chain = (
            evidence.get(None, empty),
            evidence.get(engine_id, empty),
            evidence.get((engine_id, band), empty),
        )
        p_positive = self.root_prior_probability
        p_feasible = self.root_prior_probability
        magnitude = self.reference_gain_scale
        for level in chain:
            p_positive = _beta_posterior_mean(
                prior_mean=p_positive,
                prior_strength=self.prior_strength,
                successes=level.positive_count,
                failures=level.observation_count - level.positive_count,
            )
            p_feasible = _beta_posterior_mean(
                prior_mean=p_feasible,
                prior_strength=self.prior_strength,
                successes=level.feasible_count,
                failures=level.observation_count - level.feasible_count,
            )
            magnitude = _shrunk_magnitude(
                prior_mean=magnitude,
                prior_strength=self.prior_strength,
                positive_gain_sum=level.positive_gain_sum,
                positive_count=level.positive_count,
            )
        cell = chain[2]
        effective_sample_size = (
            self.prior_strength + cell.observation_count
        )
        return {
            "p_positive": p_positive,
            "p_feasible": p_feasible,
            "magnitude": magnitude,
            "effective_sample_size": effective_sample_size,
        }

    def _forecast_geometry(
        self,
        *,
        archive_points: tuple[ObjectivePoint, ...],
        forecast: PositiveGainForecast,
    ) -> dict[str, float]:
        gains: dict[str, float] = {}
        for scenario_id in _QUANTILE_SCENARIO_IDS:
            gain = self.archive_gain_utility.marginal_archive_gain(
                archive_points,
                forecast.point(scenario_id),
            )
            if type(gain) is not float or not math.isfinite(gain) or gain < 0.0:
                raise ValueError(
                    "gain port returned an invalid archive gain"
                )
            gains[scenario_id] = gain
        weights = dict(
            zip(_QUANTILE_SCENARIO_IDS, self.scenario_weights, strict=True)
        )
        nondominated = math.fsum(
            weights[scenario_id]
            for scenario_id in _QUANTILE_SCENARIO_IDS
            if gains[scenario_id] > 0.0
        )
        positive_weight = nondominated
        magnitude = (
            math.fsum(
                weights[scenario_id] * gains[scenario_id]
                for scenario_id in _QUANTILE_SCENARIO_IDS
                if gains[scenario_id] > 0.0
            )
            / positive_weight
            if positive_weight > 0.0
            else 0.0
        )
        central = gains[CENTRAL_QUANTILE_SCENARIO_ID]
        shortfall = math.fsum(
            weights[scenario_id]
            * max(0.0, central - gains[scenario_id])
            for scenario_id in _QUANTILE_SCENARIO_IDS
        )
        return {
            "nondominated_fraction": nondominated,
            "probability": nondominated,
            "magnitude": magnitude,
            "shortfall": shortfall,
        }

    def _logit(self, probability: float) -> float:
        clamped = self._clamped(probability)
        return math.log(clamped / (1.0 - clamped))

    def _anchor_geometry(
        self,
        *,
        candidates: tuple[PositiveGainCandidate, ...],
        archive_points: tuple[ObjectivePoint, ...],
        covered_anchors: tuple[ObjectivePoint, ...],
    ) -> dict[str, tuple[float, float]]:
        """Per-candidate ``(excess, dispersion)`` in pure archive geometry.

        ``excess`` is the anchor's Chebyshev excess over the CURRENT archive
        front: smaller means the parent sits nearer the front, and it is the
        strongest outcome-blind predictor of realised positivity measured on
        this corpus.

        ``dispersion`` is the smallest max-norm distance from the anchor to
        any anchor this market has already bought, and ``inf`` when nothing
        has been bought.  A second seat on an identical anchor therefore has
        dispersion exactly zero -- the redundancy the census measured -- and
        buying an anchor can only lower the dispersion of the candidates
        near it, never of the ones far from it, so preferring larger
        dispersion is submodular in the bought set by construction.

        A candidate with no anchor is absent from the result and takes the
        market's median position, so an anchorless lane is never
        systematically evicted or systematically protected.
        """

        metric_ids = tuple(
            metric_id for metric_id, _value in archive_points[0]
        )
        covered = tuple(
            _point_values(value, metric_ids) for value in covered_anchors
        )
        result: dict[str, tuple[float, float]] = {}
        for candidate in candidates:
            if candidate.anchor_point is None:
                continue
            values = _point_values(candidate.anchor_point, metric_ids)
            dispersion = (
                min(
                    max(abs(a - b) for a, b in zip(values, other, strict=True))
                    for other in covered
                )
                if covered
                else math.inf
            )
            result[candidate.action_sha256] = (
                chebyshev_excess(
                    candidate.anchor_point,
                    archive_points,
                    metric_ids,
                ),
                float(dispersion),
            )
        return result

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def score_market(
        self,
        *,
        candidates: tuple[PositiveGainCandidate, ...],
        archive_points: tuple[ObjectivePoint, ...],
        observed_outcomes: tuple[ObservedConversionOutcome, ...],
        future_seats_remaining: int,
        horizon_total: int,
        frozen_fit_training_run_count: int = 0,
        covered_anchors: tuple[ObjectivePoint, ...] = (),
    ) -> CalibratedPositiveGainRanking:
        """Score and rank the whole market; never abstain."""

        self.__post_init__()
        if type(candidates) is not tuple or not candidates:
            raise ValueError("candidates must be a non-empty exact tuple")
        for value in candidates:
            if type(value) is not PositiveGainCandidate:
                raise TypeError(
                    "candidates must contain exact eligible candidates"
                )
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in candidates)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("candidate identities repeat")
        if type(archive_points) is not tuple or not archive_points:
            raise ValueError(
                "archive_points must be a non-empty exact tuple"
            )
        for value in archive_points:
            _require_objective_point(value, name="archive point")
        if (
            type(future_seats_remaining) is not int
            or type(horizon_total) is not int
            or horizon_total <= 0
            or not 0 <= future_seats_remaining <= horizon_total
        ):
            raise ValueError(
                "future seats must fit the positive horizon total"
            )
        if (
            type(frozen_fit_training_run_count) is not int
            or frozen_fit_training_run_count < 0
        ):
            raise ValueError(
                "frozen_fit_training_run_count must be non-negative"
            )
        frozen_history_sufficient = (
            frozen_fit_training_run_count
            >= self.frozen_score_minimum_training_runs
        )
        evidence = self._conversion_evidence(observed_outcomes)
        if type(covered_anchors) is not tuple:
            raise TypeError("covered_anchors must be an exact tuple")
        for value in covered_anchors:
            _require_objective_point(value, name="covered anchor")
        anchor_geometry = (
            self._anchor_geometry(
                candidates=candidates,
                archive_points=archive_points,
                covered_anchors=covered_anchors,
            )
            if self.anchor_geometry_tie_break
            else {}
        )
        horizon_fraction = future_seats_remaining / horizon_total
        effective_beta = self.beta * horizon_fraction
        scores: list[CalibratedPositiveGainScore] = []
        for candidate in sorted(
            candidates,
            key=lambda value: value.action_sha256,
        ):
            band = rank_band_index(
                candidate.native_rank,
                candidate.lane_size,
                self.band_count,
            )
            conversion = self._hierarchical_estimates(
                evidence=evidence,
                engine_id=candidate.engine_id,
                band=band,
            )
            p_conversion = conversion["p_positive"]
            conversion_magnitude = conversion["magnitude"]
            if candidate.forecast is None:
                geometry: dict[str, float] | None = None
                raw_forecast_weight = 0.0
            else:
                geometry = self._forecast_geometry(
                    archive_points=archive_points,
                    forecast=candidate.forecast,
                )
                raw_forecast_weight = (
                    self.mixture_weight
                    * candidate.forecast.reliability
                )
            raw_conversion_weight = 1.0 - self.mixture_weight
            frozen_active = (
                candidate.frozen_score is not None
                and frozen_history_sufficient
                and self.frozen_score_weight > 0.0
            )
            raw_frozen_weight = (
                self.frozen_score_weight if frozen_active else 0.0
            )
            anchor = anchor_geometry.get(candidate.action_sha256)
            probability_total = (
                raw_forecast_weight
                + raw_conversion_weight
                + raw_frozen_weight
            )
            if (
                geometry is None and not frozen_active
            ) or probability_total <= 0.0:
                # Only conversion evidence is active: keep the exact
                # hierarchical posterior instead of a clamped round trip.
                p_positive = p_conversion
            else:
                blended_logit = (
                    raw_conversion_weight
                    * self._logit(p_conversion)
                    + (
                        raw_forecast_weight
                        * self._logit(geometry["probability"])
                        if geometry is not None
                        else 0.0
                    )
                    + (
                        raw_frozen_weight
                        * self._logit(candidate.frozen_score)
                        if frozen_active
                        else 0.0
                    )
                ) / probability_total
                p_positive = 1.0 / (1.0 + math.exp(-blended_logit))
            frozen_evidence_weight = (
                raw_frozen_weight / probability_total
                if frozen_active
                else 0.0
            )
            # Magnitudes and tail risk come only from forecast geometry
            # and conversion evidence; a frozen rank prior carries no
            # gain scale.
            magnitude_total = raw_forecast_weight + raw_conversion_weight
            forecast_weight = (
                raw_forecast_weight / magnitude_total
                if magnitude_total > 0.0
                else 0.0
            )
            conversion_weight = 1.0 - forecast_weight
            # The conversion branch's own downside.  Both forms are the
            # weighted mass of scenarios that fall short of a central
            # outcome; they differ only in which central outcome, and the
            # shortfall form is the one that cannot invert the magnitude.
            conversion_shortfall = (
                p_conversion
                * (1.0 - p_conversion)
                * conversion_magnitude
                if self.downside_shortfall_tail_risk
                else (1.0 - p_conversion) * conversion_magnitude
            )
            if geometry is None:
                expected_positive_gain = conversion_magnitude
                tail_risk = conversion_weight * conversion_shortfall
            else:
                expected_positive_gain = (
                    forecast_weight * geometry["magnitude"]
                    + conversion_weight * conversion_magnitude
                )
                tail_risk = (
                    forecast_weight * geometry["shortfall"]
                    + conversion_weight * conversion_shortfall
                )
            uncertainty = math.sqrt(
                p_conversion
                * (1.0 - p_conversion)
                / (conversion["effective_sample_size"] + 1.0)
            )
            value_of_information = (
                uncertainty
                * max(
                    expected_positive_gain,
                    self.reference_gain_scale,
                )
                if future_seats_remaining > 0
                else 0.0
            )
            score = (
                p_positive * expected_positive_gain
                - self.lambda_ * tail_risk
                + effective_beta * value_of_information
            )
            scores.append(
                CalibratedPositiveGainScore(
                    action_sha256=candidate.action_sha256,
                    p_feasible=conversion["p_feasible"],
                    p_positive_archive_gain=float(p_positive),
                    expected_positive_gain=float(
                        expected_positive_gain
                    ),
                    tail_risk=float(tail_risk),
                    value_of_information=float(value_of_information),
                    uncertainty=float(uncertainty),
                    effective_sample_size=float(
                        conversion["effective_sample_size"]
                    ),
                    score=float(score),
                    forecast_probability=(
                        None
                        if geometry is None
                        else float(geometry["probability"])
                    ),
                    forecast_magnitude=(
                        None
                        if geometry is None
                        else float(geometry["magnitude"])
                    ),
                    forecast_nondominated_fraction=(
                        None
                        if geometry is None
                        else float(geometry["nondominated_fraction"])
                    ),
                    conversion_probability=float(p_conversion),
                    conversion_magnitude=float(conversion_magnitude),
                    frozen_score_probability=candidate.frozen_score,
                    frozen_evidence_weight=float(
                        frozen_evidence_weight
                    ),
                    anchor_excess=(
                        None if anchor is None else float(anchor[0])
                    ),
                    anchor_dispersion=(
                        None if anchor is None else float(anchor[1])
                    ),
                )
            )
        rank_quality_by_action = {
            value.action_sha256: (
                1.0
                if value.lane_size == 1
                else 1.0
                - (value.native_rank - 1)
                / float(value.lane_size - 1)
            )
            for value in candidates
        }
        # Candidates the score cannot separate are separated here.  An
        # anchorless candidate takes the market's median position on both
        # geometric keys, so no lane is systematically evicted.
        if self.anchor_geometry_tie_break and anchor_geometry:
            known = list(anchor_geometry.values())
            median_excess = self._median([value[0] for value in known])
            median_dispersion = self._median(
                [value[1] for value in known]
            )
        else:
            median_excess = 0.0
            median_dispersion = math.inf

        def geometry_keys(action_sha256: str) -> tuple[float, float]:
            if not self.anchor_geometry_tie_break:
                return (0.0, 0.0)
            excess, dispersion = anchor_geometry.get(
                action_sha256,
                (median_excess, median_dispersion),
            )
            # Larger dispersion first (novel region), then smaller excess
            # (parent nearer the front).
            return (-dispersion, excess)

        ranked = tuple(
            value.action_sha256
            for value in sorted(
                scores,
                key=lambda value: (
                    -value.score,
                    -value.p_positive_archive_gain,
                    *geometry_keys(value.action_sha256),
                    (
                        -rank_quality_by_action[value.action_sha256]
                        if self.within_cell_rank_tie_break
                        else 0.0
                    ),
                    value.action_sha256,
                ),
            )
        )
        return CalibratedPositiveGainRanking(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            archive_sha256=_hash(
                _ARCHIVE_DOMAIN,
                [
                    _objective_point_record(value)
                    for value in archive_points
                ],
            ),
            future_seats_remaining=future_seats_remaining,
            horizon_total=horizon_total,
            scores=tuple(scores),
            ranked_action_sha256s=ranked,
        )


__all__ = [
    "CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_ID",
    "CALIBRATED_POSITIVE_GAIN_OPPORTUNITY_POLICY_VERSION",
    "CENTRAL_QUANTILE_SCENARIO_ID",
    "LOWER_QUANTILE_SCENARIO_ID",
    "UPPER_QUANTILE_SCENARIO_ID",
    "ArchiveConditionedGainPort",
    "CalibratedPositiveGainOpportunityPolicy",
    "CalibratedPositiveGainRanking",
    "CalibratedPositiveGainScore",
    "ObjectivePoint",
    "ObservedConversionOutcome",
    "PositiveGainCandidate",
    "PositiveGainForecast",
    "chebyshev_excess",
    "non_dominated",
    "validate_archive_conditioned_gain_port",
]
