"""Hierarchical region-conditional credit with learned forecast trust (R1).

Defect D1 (jul28 pareto defect theory): the live broker estimated its
``gain``/``positive`` channels at LANE level with n=1, so a single win
propagated positive=1.0 to every sibling proposal of the lane, and the
``forecast_error`` channel was never populated, so anti-calibrated
forecasts kept full weight.  This module replaces the lane-level leaf with
credit cells keyed by (engine x parent-front-region x radius-or-operator
class), estimated with the EXISTING Beta-shrinkage machinery in a capped
three-level hierarchy (cell -> engine -> global), and adds a learned
forecast-trust channel: per-engine Beta posteriors over forecast direction
correctness, updated only from (predicted, actual) pairs revealed BEFORE
the current decision, applied as a demote-only multiplicative weight on
forecast-derived probability evidence.  With no trust evidence the
multiplier is exactly one, so behavior degrades to the calibrated
challenger's; measured anti-calibration demotes the forecast channel
toward zero.

This module never modifies the calibrated challenger: it composes over an
injected ``CalibratedPositiveGainOpportunityPolicy`` (forecast geometry,
clamping, and score algebra are delegated or mirrored bit-for-bit) and is
enabled only through config-gated composition (``v9_candidate_policy``).
The policy knows no workload, objective name, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainOpportunityPolicy,
    CalibratedPositiveGainRanking,
    CalibratedPositiveGainScore,
    ObjectivePoint,
    PositiveGainCandidate,
    _beta_posterior_mean,
    _require_objective_point,
    _shrunk_magnitude,
)

REGION_CONDITIONAL_CREDIT_POLICY_ID = "region_conditional_credit"
REGION_CONDITIONAL_CREDIT_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:region-conditional-credit-definition:v1\x00"
)

REGION_NO_PARENT = "region.no_parent"
RADIUS_CLASS_NONE = "radius.none"
RADIUS_CLASS_SHORT = "radius.short"
RADIUS_CLASS_MID = "radius.mid"
RADIUS_CLASS_LONG = "radius.long"


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


def _point_values(
    point: ObjectivePoint,
    metric_ids: tuple[str, ...],
) -> tuple[float, ...]:
    mapping = dict(point)
    if tuple(sorted(mapping)) != metric_ids:
        raise ValueError("objective point uses a foreign metric frame")
    return tuple(mapping[metric_id] for metric_id in metric_ids)


def parent_front_distance(
    archive_points: tuple[ObjectivePoint, ...],
    parent_point: ObjectivePoint,
) -> float:
    """Additive-epsilon distance of a parent behind the archive front.

    ``min over archive points a of max over objectives j of (p_j - a_j)``,
    clamped at zero: the smallest uniform improvement that would move the
    parent onto (or past) some archive point.  Zero exactly for parents on
    or beyond the front; dimension-generic and workload-free (operates on
    whatever normalized objective frame the caller uses).
    """

    _require_objective_point(parent_point, name="parent point")
    if type(archive_points) is not tuple or not archive_points:
        raise ValueError("archive_points must be a non-empty exact tuple")
    metric_ids = tuple(metric_id for metric_id, _value in parent_point)
    parent = _point_values(parent_point, metric_ids)
    best = math.inf
    for point in archive_points:
        _require_objective_point(point, name="archive point")
        values = _point_values(point, metric_ids)
        best = min(
            best,
            max(
                p - a
                for p, a in zip(parent, values, strict=True)
            ),
        )
    return float(max(best, 0.0))


def parent_front_region(
    *,
    archive_points: tuple[ObjectivePoint, ...],
    reference_point: ObjectivePoint,
    parent_point: ObjectivePoint | None,
    near_front_epsilon: float,
    extreme_affinity_threshold: float,
) -> str:
    """Classify a parent into a front region token.

    Region = distance band (near/far by additive-epsilon distance) crossed
    with a wedge: the objective extreme the parent has the strongest
    relative affinity to, or interior when no per-objective affinity is
    strong enough.  One wedge per objective plus interior, so the region
    set is dimension-generic (memo R4's wedge construction).  Parents are
    absent for archive-seeded candidates: those map to a dedicated token.
    """

    if parent_point is None:
        return REGION_NO_PARENT
    distance = parent_front_distance(archive_points, parent_point)
    band = "near" if distance <= near_front_epsilon else "far"
    metric_ids = tuple(metric_id for metric_id, _value in parent_point)
    parent = _point_values(parent_point, metric_ids)
    reference = _point_values(reference_point, metric_ids)
    best_per_metric = tuple(
        min(
            _point_values(point, metric_ids)[index]
            for point in archive_points
        )
        for index in range(len(metric_ids))
    )
    affinities: list[tuple[float, str]] = []
    for index, metric_id in enumerate(metric_ids):
        span = reference[index] - best_per_metric[index]
        if span <= 0.0:
            affinity = 0.0
        else:
            affinity = max(
                (parent[index] - best_per_metric[index]) / span,
                0.0,
            )
        affinities.append((affinity, metric_id))
    affinity, metric_id = min(affinities)
    if affinity <= extreme_affinity_threshold:
        return f"region.{band}.extreme:{metric_id}"
    return f"region.{band}.interior"


def radius_operator_class(
    *,
    radius: int | None,
    operator_class: str | None,
    radius_breakpoints: tuple[int, int],
) -> str:
    """Map a move radius (or operator class fallback) into a class token."""

    short_cap, mid_cap = radius_breakpoints
    if radius is not None:
        if type(radius) is not int or radius < 0:
            raise ValueError("radius must be a non-negative integer")
        if radius <= short_cap:
            return RADIUS_CLASS_SHORT
        if radius <= mid_cap:
            return RADIUS_CLASS_MID
        return RADIUS_CLASS_LONG
    if operator_class is not None:
        _require_token(operator_class, name="operator_class")
        return f"op.{operator_class}"
    return RADIUS_CLASS_NONE


@dataclass(frozen=True, slots=True)
class RegionFeatures:
    """Outcome-blind provenance features of one market candidate."""

    parent_point: ObjectivePoint | None = None
    radius: int | None = None
    operator_class: str | None = None

    def __post_init__(self) -> None:
        if self.parent_point is not None:
            _require_objective_point(
                self.parent_point,
                name="parent point",
            )
        if self.radius is not None and (
            type(self.radius) is not int or self.radius < 0
        ):
            raise ValueError("radius must be a non-negative integer")
        if self.operator_class is not None:
            _require_token(self.operator_class, name="operator_class")


@dataclass(frozen=True, slots=True)
class RegionConditionalOutcome:
    """One real outcome observed strictly before the current decision.

    ``region_id`` is None for evidence imported from OUTSIDE the current
    market (warm priors): such outcomes carry no comparable parent
    geometry, so they contribute only to the global and engine levels of
    the hierarchy, never to a leaf cell.  Forecast direction fields are
    jointly present exactly when the outcome's candidate carried a
    forecast: ``forecast_predicted_positive`` is whether the forecast's
    central scenario claimed positive archive gain, and
    ``forecast_actual_positive`` whether the realized point achieved
    positive gain against the same base archive.
    """

    observation_ordinal: int
    engine_id: str
    feasible: bool
    marginal_archive_gain: float
    region_id: str | None = None
    radius_class_id: str | None = None
    forecast_predicted_positive: bool | None = None
    forecast_actual_positive: bool | None = None

    def __post_init__(self) -> None:
        if (
            type(self.observation_ordinal) is not int
            or self.observation_ordinal <= 0
        ):
            raise ValueError("observation_ordinal must be positive")
        _require_token(self.engine_id, name="engine_id")
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
        if (self.region_id is None) != (self.radius_class_id is None):
            raise ValueError(
                "region and radius class must be jointly present or absent"
            )
        if self.region_id is not None:
            _require_token(self.region_id, name="region_id")
            _require_token(
                self.radius_class_id,
                name="radius_class_id",
            )
        forecast_fields = (
            self.forecast_predicted_positive,
            self.forecast_actual_positive,
        )
        if any(value is None for value in forecast_fields) != all(
            value is None for value in forecast_fields
        ):
            raise ValueError(
                "forecast direction fields must be jointly present or absent"
            )
        for value in forecast_fields:
            if value is not None and type(value) is not bool:
                raise TypeError("forecast direction fields must be exact")

    @property
    def positive(self) -> bool:
        return self.marginal_archive_gain > 0.0

    @property
    def forecast_direction_correct(self) -> bool | None:
        if self.forecast_predicted_positive is None:
            return None
        return (
            self.forecast_predicted_positive
            == self.forecast_actual_positive
        )


@dataclass(frozen=True, slots=True)
class RegionScoredCandidate:
    """One eligible candidate with its provenance features attached."""

    candidate: PositiveGainCandidate
    features: RegionFeatures = RegionFeatures()

    def __post_init__(self) -> None:
        if type(self.candidate) is not PositiveGainCandidate:
            raise TypeError("candidate must be exact")
        self.candidate.__post_init__()
        if type(self.features) is not RegionFeatures:
            raise TypeError("features must be exact")
        self.features.__post_init__()


@dataclass(frozen=True, slots=True)
class RegionCreditConfig:
    """All R1 constants; every value is an exact dyadic float."""

    near_front_epsilon: float = 0.125
    extreme_affinity_threshold: float = 0.25
    radius_breakpoints: tuple[int, int] = (1, 3)
    forecast_trust_prior: float = 0.5

    def __post_init__(self) -> None:
        for name in ("near_front_epsilon", "extreme_affinity_threshold"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if (
            type(self.radius_breakpoints) is not tuple
            or len(self.radius_breakpoints) != 2
            or any(
                type(value) is not int or value < 0
                for value in self.radius_breakpoints
            )
            or not self.radius_breakpoints[0]
            <= self.radius_breakpoints[1]
        ):
            raise ValueError(
                "radius_breakpoints must be two ascending non-negative "
                "integers"
            )
        if (
            type(self.forecast_trust_prior) is not float
            or not math.isfinite(self.forecast_trust_prior)
            or not 0.0 < self.forecast_trust_prior < 1.0
        ):
            raise ValueError(
                "forecast_trust_prior must lie in (0, 1)"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "near_front_epsilon_hex": self.near_front_epsilon.hex(),
            "extreme_affinity_threshold_hex": (
                self.extreme_affinity_threshold.hex()
            ),
            "radius_breakpoints": list(self.radius_breakpoints),
            "forecast_trust_prior_hex": (
                self.forecast_trust_prior.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class _RegionCellEvidence:
    observation_count: float
    positive_count: float
    feasible_count: float
    positive_gain_sum: float


@dataclass(frozen=True, slots=True)
class RegionConditionalChallengerPolicy:
    """Score eligible candidates with region-conditional credit.

    The score algebra is the calibrated positive-gain challenger's,
    bit-for-bit, with exactly two substitutions:

    * conversion evidence is estimated over the capped hierarchy
      global -> engine -> (engine x region x radius class) instead of
      global -> engine -> (engine x rank band); and
    * the forecast probability weight is multiplied by the engine's
      learned forecast-trust multiplier (demote-only; exactly one with
      no trust evidence).
    """

    base: CalibratedPositiveGainOpportunityPolicy = field(repr=False)
    credit: RegionCreditConfig = RegionCreditConfig()
    policy_id: str = REGION_CONDITIONAL_CREDIT_POLICY_ID
    policy_version: int = REGION_CONDITIONAL_CREDIT_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.base) is not CalibratedPositiveGainOpportunityPolicy:
            raise TypeError(
                "base must be an exact calibrated challenger"
            )
        self.base.__post_init__()
        if type(self.credit) is not RegionCreditConfig:
            raise TypeError("credit must be an exact region config")
        self.credit.__post_init__()
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_id != REGION_CONDITIONAL_CREDIT_POLICY_ID
            or self.policy_version
            != REGION_CONDITIONAL_CREDIT_POLICY_VERSION
        ):
            raise ValueError("policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "base_challenger": {
                        "policy_id": self.base.policy_id,
                        "policy_version": self.base.policy_version,
                        "definition_sha256": (
                            self.base.definition_sha256
                        ),
                    },
                    "credit": self.credit.to_record(),
                    "conversion_evidence": (
                        "prequential_engine_region_radius_cell_"
                        "hierarchical_beta_shrinkage_capped_three_levels"
                    ),
                    "hierarchy": "cell_engine_global",
                    "prior_market_evidence_leaf_cells": False,
                    "forecast_trust": (
                        "per_engine_direction_correct_beta_posterior_"
                        "demote_only_multiplier_min_one_two_posterior"
                    ),
                    "lane_level_n1_credit_propagation": False,
                    "hard_abstention": False,
                    "eligible_candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    def region_for(
        self,
        *,
        archive_points: tuple[ObjectivePoint, ...],
        reference_point: ObjectivePoint,
        features: RegionFeatures,
    ) -> tuple[str, str]:
        """(region token, radius class token) of one candidate."""

        features.__post_init__()
        return (
            parent_front_region(
                archive_points=archive_points,
                reference_point=reference_point,
                parent_point=features.parent_point,
                near_front_epsilon=self.credit.near_front_epsilon,
                extreme_affinity_threshold=(
                    self.credit.extreme_affinity_threshold
                ),
            ),
            radius_operator_class(
                radius=features.radius,
                operator_class=features.operator_class,
                radius_breakpoints=self.credit.radius_breakpoints,
            ),
        )

    @staticmethod
    def _validated_outcomes(
        observed_outcomes: tuple[RegionConditionalOutcome, ...],
    ) -> tuple[RegionConditionalOutcome, ...]:
        if type(observed_outcomes) is not tuple or any(
            type(value) is not RegionConditionalOutcome
            for value in observed_outcomes
        ):
            raise TypeError(
                "observed_outcomes must contain exact region outcomes"
            )
        ordinals = tuple(
            value.observation_ordinal for value in observed_outcomes
        )
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError(
                "observed outcomes must use unique ascending ordinals"
            )
        for value in observed_outcomes:
            value.__post_init__()
        return observed_outcomes

    def _conversion_evidence(
        self,
        observed_outcomes: tuple[RegionConditionalOutcome, ...],
    ) -> dict[object, _RegionCellEvidence]:
        counts: dict[object, list[float]] = {}
        for value in observed_outcomes:
            keys: list[object] = [None, value.engine_id]
            if value.region_id is not None:
                keys.append(
                    (
                        value.engine_id,
                        value.region_id,
                        value.radius_class_id,
                    )
                )
            for key in keys:
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
            key: _RegionCellEvidence(
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
        evidence: dict[object, _RegionCellEvidence],
        engine_id: str,
        region_id: str,
        radius_class_id: str,
    ) -> dict[str, float]:
        empty = _RegionCellEvidence(0.0, 0.0, 0.0, 0.0)
        chain = (
            evidence.get(None, empty),
            evidence.get(engine_id, empty),
            evidence.get(
                (engine_id, region_id, radius_class_id),
                empty,
            ),
        )
        p_positive = self.base.root_prior_probability
        p_feasible = self.base.root_prior_probability
        magnitude = self.base.reference_gain_scale
        for level in chain:
            p_positive = _beta_posterior_mean(
                prior_mean=p_positive,
                prior_strength=self.base.prior_strength,
                successes=level.positive_count,
                failures=level.observation_count - level.positive_count,
            )
            p_feasible = _beta_posterior_mean(
                prior_mean=p_feasible,
                prior_strength=self.base.prior_strength,
                successes=level.feasible_count,
                failures=level.observation_count - level.feasible_count,
            )
            magnitude = _shrunk_magnitude(
                prior_mean=magnitude,
                prior_strength=self.base.prior_strength,
                positive_gain_sum=level.positive_gain_sum,
                positive_count=level.positive_count,
            )
        cell = chain[2]
        return {
            "p_positive": p_positive,
            "p_feasible": p_feasible,
            "magnitude": magnitude,
            "effective_sample_size": (
                self.base.prior_strength + cell.observation_count
            ),
        }

    def forecast_trust_multiplier(
        self,
        *,
        observed_outcomes: tuple[RegionConditionalOutcome, ...],
        engine_id: str,
    ) -> float:
        """Demote-only trust weight for one engine's forecast channel.

        Beta posterior over forecast direction correctness with prior
        mean ``forecast_trust_prior`` and the base challenger's prior
        strength, mapped through ``min(1, 2 * posterior)``: exactly one
        with no evidence or calibrated forecasts, strictly below one
        once measured accuracy falls below one half (anti-calibration).
        """

        successes = 0.0
        failures = 0.0
        for value in observed_outcomes:
            correct = value.forecast_direction_correct
            if correct is None or value.engine_id != engine_id:
                continue
            if correct:
                successes += 1.0
            else:
                failures += 1.0
        posterior = _beta_posterior_mean(
            prior_mean=self.credit.forecast_trust_prior,
            prior_strength=self.base.prior_strength,
            successes=successes,
            failures=failures,
        )
        return float(min(1.0, 2.0 * posterior))

    def score_market(
        self,
        *,
        candidates: tuple[RegionScoredCandidate, ...],
        archive_points: tuple[ObjectivePoint, ...],
        reference_point: ObjectivePoint,
        observed_outcomes: tuple[RegionConditionalOutcome, ...],
        future_seats_remaining: int,
        horizon_total: int,
        frozen_fit_training_run_count: int = 0,
    ) -> CalibratedPositiveGainRanking:
        """Score and rank the whole market; never abstain."""

        self.__post_init__()
        if type(candidates) is not tuple or not candidates:
            raise ValueError("candidates must be a non-empty exact tuple")
        for value in candidates:
            if type(value) is not RegionScoredCandidate:
                raise TypeError(
                    "candidates must contain exact region candidates"
                )
            value.__post_init__()
        action_ids = tuple(
            value.candidate.action_sha256 for value in candidates
        )
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("candidate identities repeat")
        if type(archive_points) is not tuple or not archive_points:
            raise ValueError(
                "archive_points must be a non-empty exact tuple"
            )
        for value in archive_points:
            _require_objective_point(value, name="archive point")
        _require_objective_point(
            reference_point,
            name="reference point",
        )
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
        outcomes = self._validated_outcomes(observed_outcomes)
        evidence = self._conversion_evidence(outcomes)
        trust_by_engine = {
            engine_id: self.forecast_trust_multiplier(
                observed_outcomes=outcomes,
                engine_id=engine_id,
            )
            for engine_id in {
                value.candidate.engine_id for value in candidates
            }
        }
        frozen_history_sufficient = (
            frozen_fit_training_run_count
            >= self.base.frozen_score_minimum_training_runs
        )
        horizon_fraction = future_seats_remaining / horizon_total
        effective_beta = self.base.beta * horizon_fraction
        scores: list[CalibratedPositiveGainScore] = []
        by_action = {
            value.candidate.action_sha256: value
            for value in candidates
        }
        for scored in sorted(
            candidates,
            key=lambda value: value.candidate.action_sha256,
        ):
            candidate = scored.candidate
            region_id, radius_class_id = self.region_for(
                archive_points=archive_points,
                reference_point=reference_point,
                features=scored.features,
            )
            conversion = self._hierarchical_estimates(
                evidence=evidence,
                engine_id=candidate.engine_id,
                region_id=region_id,
                radius_class_id=radius_class_id,
            )
            p_conversion = conversion["p_positive"]
            conversion_magnitude = conversion["magnitude"]
            if candidate.forecast is None:
                geometry: dict[str, float] | None = None
                raw_forecast_weight = 0.0
            else:
                geometry = self.base._forecast_geometry(
                    archive_points=archive_points,
                    forecast=candidate.forecast,
                )
                raw_forecast_weight = (
                    self.base.mixture_weight
                    * candidate.forecast.reliability
                    * trust_by_engine[candidate.engine_id]
                )
            raw_conversion_weight = 1.0 - self.base.mixture_weight
            frozen_active = (
                candidate.frozen_score is not None
                and frozen_history_sufficient
                and self.base.frozen_score_weight > 0.0
            )
            raw_frozen_weight = (
                self.base.frozen_score_weight if frozen_active else 0.0
            )
            probability_total = (
                raw_forecast_weight
                + raw_conversion_weight
                + raw_frozen_weight
            )
            if (
                geometry is None and not frozen_active
            ) or probability_total <= 0.0:
                p_positive = p_conversion
            else:
                blended_logit = (
                    raw_conversion_weight
                    * self.base._logit(p_conversion)
                    + (
                        raw_forecast_weight
                        * self.base._logit(geometry["probability"])
                        if geometry is not None
                        else 0.0
                    )
                    + (
                        raw_frozen_weight
                        * self.base._logit(candidate.frozen_score)
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
            magnitude_total = raw_forecast_weight + raw_conversion_weight
            forecast_weight = (
                raw_forecast_weight / magnitude_total
                if magnitude_total > 0.0
                else 0.0
            )
            conversion_weight = 1.0 - forecast_weight
            if geometry is None:
                expected_positive_gain = conversion_magnitude
                tail_risk = (
                    conversion_weight
                    * (1.0 - p_conversion)
                    * conversion_magnitude
                )
            else:
                expected_positive_gain = (
                    forecast_weight * geometry["magnitude"]
                    + conversion_weight * conversion_magnitude
                )
                tail_risk = forecast_weight * geometry[
                    "shortfall"
                ] + conversion_weight * (
                    (1.0 - p_conversion) * conversion_magnitude
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
                    self.base.reference_gain_scale,
                )
                if future_seats_remaining > 0
                else 0.0
            )
            score = (
                p_positive * expected_positive_gain
                - self.base.lambda_ * tail_risk
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
                )
            )
        rank_quality_by_action = {
            value.candidate.action_sha256: (
                1.0
                if value.candidate.lane_size == 1
                else 1.0
                - (value.candidate.native_rank - 1)
                / float(value.candidate.lane_size - 1)
            )
            for value in candidates
        }
        ranked = tuple(
            value.action_sha256
            for value in sorted(
                scores,
                key=lambda value: (
                    -value.score,
                    -value.p_positive_archive_gain,
                    (
                        -rank_quality_by_action[value.action_sha256]
                        if self.base.within_cell_rank_tie_break
                        else 0.0
                    ),
                    value.action_sha256,
                ),
            )
        )
        assert set(ranked) == set(by_action)
        return CalibratedPositiveGainRanking(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            archive_sha256=_hash(
                _DEFINITION_DOMAIN,
                [
                    [
                        {
                            "metric_id": metric_id,
                            "value_hex": metric_value.hex(),
                        }
                        for metric_id, metric_value in value
                    ]
                    for value in archive_points
                ],
            ),
            future_seats_remaining=future_seats_remaining,
            horizon_total=horizon_total,
            scores=tuple(scores),
            ranked_action_sha256s=ranked,
        )


__all__ = [
    "RADIUS_CLASS_LONG",
    "RADIUS_CLASS_MID",
    "RADIUS_CLASS_NONE",
    "RADIUS_CLASS_SHORT",
    "REGION_CONDITIONAL_CREDIT_POLICY_ID",
    "REGION_CONDITIONAL_CREDIT_POLICY_VERSION",
    "REGION_NO_PARENT",
    "RegionConditionalChallengerPolicy",
    "RegionConditionalOutcome",
    "RegionCreditConfig",
    "RegionFeatures",
    "RegionScoredCandidate",
    "parent_front_distance",
    "parent_front_region",
    "radius_operator_class",
]
