"""Prior-only empirical calibration for action-consequence forecasts.

The language model is a useful semantic proposal expert, but it must not remain
the numerical oracle after real evaluator evidence exists.  This module adds a
workload-neutral consequence expert over the existing action-outcome ledger.
It transfers only within an authenticated campaign scope and only from waves
strictly earlier than the current one.

The policy is deliberately conservative.  It builds a local empirical
distribution from repeated patch paths when available, otherwise from the
option's sealed family, weights observations by parent-metric proximity and
recency, and fuses it with the model distribution.  Both experts are judged
prequentially: categorical model correctness can reverse an anti-calibrated
model signal, while empirical transfer loses authority when its own strictly
prior-wave predictions are below chance.  Sparse evidence cannot fully replace
the model, and exact metric projectors remain a higher authority when applied
after this policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.application.action_structural_signature import (
    parent_relative_changed_paths_by_option,
)
from agent_evolve.application.action_target_realization import TargetMetricAlias
from agent_evolve.application.portfolio_evolution import PortfolioMemberDisposition
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioActionOutcomeFeedback,
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.policies.selection.forecast_calibration import (
    BetaCorrectnessPrior,
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
    ForecastConfidenceBin,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastRequest,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionMetricForecast,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.artifact_store import ArtifactStore, put_json


EMPIRICAL_CONSEQUENCE_POLICY_ID = "competitive_prequential_consequence"
EMPIRICAL_CONSEQUENCE_POLICY_VERSION = 4
EMPIRICAL_CONSEQUENCE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:competitive-prequential-consequence:v4;"
    b"evidence=prior-wave-engine-metric-transitions;"
    b"metric-identity=explicit-target-to-forecast-alias-contract;"
    b"scope=model-prompt-selector-benchmark-session;"
    b"strata=exact-parent-patch-paths-and-family-then-family;"
    b"locality=parent-metric-distance;recency=geometric;"
    b"model-authority=beta-smoothed-direction-correctness-with-negative-skill;"
    b"empirical-authority=weighted-quantiles-gated-by-prequential-skill;"
    b"fusion=bounded-evidence-and-performance-shrinkage;"
    b"validity=family-frequency-bounded-shrinkage;"
    b"audit=bounded-manifest-plus-content-addressed-per-action-artifacts;"
    b"current-future-outcomes=false;workload-model-branches=false"
).hexdigest()
_RESULT_DOMAIN = b"agent-evolve:empirical-consequence-calibration-result:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("empirical calibration audit did not freeze to an object")
    return frozen


def _artifact_ref_record(value: ArtifactRef) -> dict[str, object]:
    """Project an artifact reference without coupling the policy to storage."""

    return {
        "artifact_id": value.artifact_id.value,
        "sha256_hex": value.sha256_hex,
        "size_bytes": value.size_bytes,
        "media_type": value.media_type,
    }


def _weighted_quantile(
    values: tuple[tuple[float, float], ...],
    probability: float,
) -> float:
    if not values:
        raise ValueError("weighted quantile requires observations")
    if type(probability) is not float or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be a canonical float in [0,1]")
    ordered = tuple(sorted(values, key=lambda item: (item[0], item[1])))
    total = sum(weight for _, weight in ordered)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weighted quantile requires positive finite weight")
    threshold = probability * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _effective_support(weights: tuple[float, ...]) -> float:
    total = sum(weights)
    square_total = sum(value * value for value in weights)
    if square_total <= 0.0:
        return 0.0
    return (total * total) / square_total


def _forecast_direction(
    forecast: ResolvedActionMetricForecast,
) -> MetricEffectDirection:
    """Use the same interval semantics as selected-forecast calibration."""

    if type(forecast) is not ResolvedActionMetricForecast:
        raise TypeError("forecast must be an exact resolved metric forecast")
    forecast.__post_init__()
    if forecast.p10_delta == forecast.p50_delta == forecast.p90_delta == 0.0:
        return MetricEffectDirection.UNCHANGED
    if forecast.p90_delta < 0.0:
        return MetricEffectDirection.DECREASE
    if forecast.p10_delta > 0.0:
        return MetricEffectDirection.INCREASE
    return MetricEffectDirection.UNKNOWN


def _confidence_bin(value: float) -> ForecastConfidenceBin:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("forecast confidence must be a finite canonical float")
    if not 0.0 <= value <= 1.0:
        raise ValueError("forecast confidence must lie in [0,1]")
    if value >= 0.75:
        return ForecastConfidenceBin.HIGH
    if value >= 0.4:
        return ForecastConfidenceBin.MEDIUM
    if value > 0.0:
        return ForecastConfidenceBin.LOW
    return ForecastConfidenceBin.UNKNOWN


def _signed_interval(
    forecast: ResolvedActionMetricForecast,
    factor: float,
) -> tuple[float, float, float]:
    """Project correctness skill onto a quantile interval without reordering it."""

    if type(factor) is not float or not math.isfinite(factor):
        raise TypeError("signed skill factor must be a finite canonical float")
    if not -1.0 <= factor <= 1.0:
        raise ValueError("signed skill factor must lie in [-1,1]")
    values = (
        factor * forecast.p10_delta,
        factor * forecast.p50_delta,
        factor * forecast.p90_delta,
    )
    return tuple(sorted(values))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class ActionConsequenceCalibrationResult:
    """Authenticated calibrated batch plus a complete no-leakage audit."""

    source_forecast_receipt_sha256: str
    cutoff_wave_index_exclusive: int
    forecasts: ResolvedActionForecastBatch
    audit: FrozenJsonObject

    def __post_init__(self) -> None:
        if (
            type(self.source_forecast_receipt_sha256) is not str
            or len(self.source_forecast_receipt_sha256) != 64
        ):
            raise ValueError("source_forecast_receipt_sha256 must be a SHA-256")
        try:
            bytes.fromhex(self.source_forecast_receipt_sha256)
        except ValueError as error:
            raise ValueError(
                "source_forecast_receipt_sha256 must be lowercase hexadecimal"
            ) from error
        if (
            type(self.cutoff_wave_index_exclusive) is not int
            or self.cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be an exact resolved batch")
        self.forecasts.__post_init__()
        if (
            type(self.audit) is not FrozenJsonObject
            or freeze_json(self.audit) is not self.audit
        ):
            raise TypeError("audit must be an exact frozen JSON object")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": EMPIRICAL_CONSEQUENCE_POLICY_ID,
                "policy_version": EMPIRICAL_CONSEQUENCE_POLICY_VERSION,
                "definition_sha256": (EMPIRICAL_CONSEQUENCE_POLICY_DEFINITION_SHA256),
            },
            "source_forecast_receipt_sha256": self.source_forecast_receipt_sha256,
            "cutoff_wave_index_exclusive": self.cutoff_wave_index_exclusive,
            "calibrated_forecast_receipt_sha256": self.forecasts.receipt_sha256,
            # Frozen typed JSON is the trusted in-memory representation, but
            # result records cross the ordinary JSON evidence boundary.  Keep
            # the immutable audit internally and thaw only for serialization.
            "audit": thaw_json(self.audit),
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _RESULT_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ActionConsequenceCalibrationResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@runtime_checkable
class ActionConsequenceCalibrationPolicy(Protocol):
    """Stable application port for prior-only consequence posteriors."""

    def calibrate(
        self,
        *,
        request: ActionForecastRequest,
        forecasts: ResolvedActionForecastBatch,
        cutoff_wave_index_exclusive: int,
        metric_aliases: tuple[TargetMetricAlias, ...] = (),
    ) -> ActionConsequenceCalibrationResult: ...


@dataclass(slots=True)
class HierarchicalEmpiricalConsequenceCalibrationPolicy:
    """Compete LLM and local empirical experts without domain branches."""

    ledger: PortfolioOutcomeFeedbackLedger
    scope: ForecastCalibrationScope
    minimum_path_support: int = 2
    minimum_family_support: int = 2
    prior_strength: float = 3.0
    validity_prior_strength: float = 2.0
    maximum_empirical_authority: float = 0.75
    recency_decay: float = 0.9
    minimum_model_score_support: int = 2
    model_family_min_support: int = 4
    minimum_empirical_score_support: int = 4
    correctness_prior_alpha: float = 1.0
    correctness_prior_beta: float = 1.0
    audit_artifact_store: ArtifactStore | None = None
    maximum_embedded_action_audits: int = 32

    def __post_init__(self) -> None:
        if type(self.ledger) is not PortfolioOutcomeFeedbackLedger:
            raise TypeError("ledger must be exact PortfolioOutcomeFeedbackLedger")
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        for name in (
            "minimum_path_support",
            "minimum_family_support",
            "minimum_model_score_support",
            "model_family_min_support",
            "minimum_empirical_score_support",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "prior_strength",
            "validity_prior_strength",
            "correctness_prior_alpha",
            "correctness_prior_beta",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite float")
        if (
            type(self.maximum_empirical_authority) is not float
            or not 0.0 <= self.maximum_empirical_authority <= 1.0
        ):
            raise ValueError("maximum_empirical_authority must lie in [0,1]")
        if type(self.recency_decay) is not float or not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency_decay must lie in (0,1]")
        if self.audit_artifact_store is not None and not isinstance(
            self.audit_artifact_store,
            ArtifactStore,
        ):
            raise TypeError("audit_artifact_store must implement ArtifactStore")
        if (
            type(self.maximum_embedded_action_audits) is not int
            or self.maximum_embedded_action_audits < 0
        ):
            raise ValueError(
                "maximum_embedded_action_audits must be a non-negative exact integer"
            )

    def _correctness_prior(self) -> BetaCorrectnessPrior:
        return BetaCorrectnessPrior(
            alpha=self.correctness_prior_alpha,
            beta=self.correctness_prior_beta,
        )

    def _model_projection(
        self,
        *,
        snapshot: ForecastCalibrationSnapshot,
        family: str,
        outcome_metric_id: str,
        raw: ResolvedActionMetricForecast,
    ) -> tuple[ResolvedActionMetricForecast, dict[str, object]]:
        direction = _forecast_direction(raw)
        confidence = _confidence_bin(raw.confidence)
        cell, source = snapshot.lookup(
            metric_id=outcome_metric_id,
            asserted_direction=direction,
            confidence=confidence,
            family=family,
        )
        score_identified = (
            direction is not MetricEffectDirection.UNKNOWN
            and cell.scorable_count >= self.minimum_model_score_support
        )
        signed_skill = (
            1.0 if not score_identified else (2.0 * cell.posterior_correctness) - 1.0
        )
        p10, p50, p90 = _signed_interval(raw, float(signed_skill))
        projected = ResolvedActionMetricForecast(
            metric_id=raw.metric_id,
            p10_delta=float(p10),
            p50_delta=float(p50),
            p90_delta=float(p90),
            confidence=float(
                raw.confidence
                if not score_identified
                else raw.confidence * abs(signed_skill)
            ),
            citations=raw.citations,
        )
        return projected, {
            "forecast_metric_id": raw.metric_id,
            "outcome_metric_id": outcome_metric_id,
            "direction": direction.value,
            "confidence_bin": confidence.value,
            "calibration_source": source,
            "scorable_count": cell.scorable_count,
            "correct_count": cell.correct_count,
            "posterior_correctness_hex": cell.posterior_correctness.hex(),
            "score_identified": score_identified,
            "signed_skill_hex": signed_skill.hex(),
            "negative_skill_inversion": signed_skill < 0.0,
            "raw": raw.to_record(),
            "projected": projected.to_record(),
        }

    def _weighted_metric_observations(
        self,
        *,
        actions: tuple[PortfolioActionOutcomeFeedback, ...],
        metric_id: str,
        current_parent_value: float,
        scale: float,
        cutoff_wave_index_exclusive: int,
    ) -> tuple[tuple[PortfolioActionOutcomeFeedback, float, float], ...]:
        observations: list[tuple[PortfolioActionOutcomeFeedback, float, float]] = []
        for action in actions:
            if (
                action.wave_index >= cutoff_wave_index_exclusive
                or action.disposition is not PortfolioMemberDisposition.SCORED
            ):
                continue
            transition = next(
                (
                    value
                    for value in action.metric_transitions
                    if value.metric_id == metric_id
                ),
                None,
            )
            if transition is None:
                continue
            locality = 1.0 / (
                1.0 + abs(transition.parent_value - current_parent_value) / scale
            )
            age = cutoff_wave_index_exclusive - 1 - action.wave_index
            recency = self.recency_decay ** max(0, age)
            observations.append(
                (
                    action,
                    transition.child_value - transition.parent_value,
                    locality * recency,
                )
            )
        return tuple(observations)

    def _empirical_rows(
        self,
        *,
        actions: tuple[PortfolioActionOutcomeFeedback, ...],
        changed_paths: tuple[str, ...],
        family: str,
        metric_id: str,
        current_parent_value: float,
        scale: float,
        cutoff_wave_index_exclusive: int,
    ) -> tuple[str, tuple[tuple[float, float], ...], int]:
        observations = self._weighted_metric_observations(
            actions=actions,
            metric_id=metric_id,
            current_parent_value=current_parent_value,
            scale=scale,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
        )
        path_rows = tuple(
            (delta, weight)
            for action, delta, weight in observations
            if action.changed_paths == changed_paths and action.family == family
        )
        family_rows = tuple(
            (delta, weight)
            for action, delta, weight in observations
            if action.family == family
        )
        if len(path_rows) >= self.minimum_path_support:
            return "exact_path_family", path_rows, len(family_rows)
        if len(family_rows) >= self.minimum_family_support:
            return "family", family_rows, len(family_rows)
        return "model_only", (), len(family_rows)

    def _empirical_prequential_skill(
        self,
        *,
        actions: tuple[PortfolioActionOutcomeFeedback, ...],
        family: str,
        metric_id: str,
        scale: float,
    ) -> dict[str, object]:
        scorable = 0
        correct = 0
        for target in sorted(
            (value for value in actions if value.family == family),
            key=lambda value: (
                value.wave_index,
                value.request_sha256,
                value.option_id,
            ),
        ):
            transition = next(
                (
                    value
                    for value in target.metric_transitions
                    if value.metric_id == metric_id
                ),
                None,
            )
            if transition is None:
                continue
            _, rows, _ = self._empirical_rows(
                actions=actions,
                changed_paths=target.changed_paths,
                family=target.family,
                metric_id=metric_id,
                current_parent_value=transition.parent_value,
                scale=scale,
                cutoff_wave_index_exclusive=target.wave_index,
            )
            if not rows:
                continue
            median = _weighted_quantile(rows, 0.5)
            predicted = (
                MetricEffectDirection.DECREASE
                if median < 0.0
                else MetricEffectDirection.INCREASE
                if median > 0.0
                else MetricEffectDirection.UNCHANGED
            )
            scorable += 1
            correct += predicted is transition.actual_direction
        prior = self._correctness_prior()
        posterior = (prior.alpha + correct) / (prior.alpha + prior.beta + scorable)
        identified = scorable >= self.minimum_empirical_score_support
        authority_multiplier = (
            1.0 if not identified else max(0.0, (2.0 * posterior) - 1.0)
        )
        return {
            "scorable_count": scorable,
            "correct_count": correct,
            "posterior_correctness_hex": posterior.hex(),
            "score_identified": identified,
            "authority_multiplier_hex": authority_multiplier.hex(),
        }

    def _prior_actions(
        self,
        cutoff_wave_index_exclusive: int,
    ) -> tuple[PortfolioActionOutcomeFeedback, ...]:
        return tuple(
            action
            for receipt in self.ledger.receipts
            if receipt.scope == self.scope
            and receipt.wave_index < cutoff_wave_index_exclusive
            for action in receipt.actions
        )

    def _validity_projection(
        self,
        *,
        actions: tuple[PortfolioActionOutcomeFeedback, ...],
        family: str,
        raw_probability: float,
    ) -> tuple[float, dict[str, object]]:
        members = tuple(value for value in actions if value.family == family)
        if not members:
            return raw_probability, {
                "support_count": 0,
                "empirical_authority_hex": (0.0).hex(),
                "posterior_valid_probability_hex": None,
            }
        valid_count = sum(
            value.disposition is PortfolioMemberDisposition.SCORED for value in members
        )
        posterior = valid_count / len(members)
        authority = min(
            self.maximum_empirical_authority,
            len(members) / (len(members) + self.validity_prior_strength),
        )
        calibrated = (1.0 - authority) * raw_probability + authority * posterior
        return calibrated, {
            "support_count": len(members),
            "valid_count": valid_count,
            "empirical_authority_hex": authority.hex(),
            "posterior_valid_probability_hex": posterior.hex(),
        }

    def _metric_projection(
        self,
        *,
        actions: tuple[PortfolioActionOutcomeFeedback, ...],
        model_snapshot: ForecastCalibrationSnapshot,
        changed_paths: tuple[str, ...],
        family: str,
        metric_id: str,
        current_parent_value: float,
        scale: float,
        cutoff_wave_index_exclusive: int,
        raw: ResolvedActionMetricForecast,
    ) -> tuple[ResolvedActionMetricForecast, dict[str, object]]:
        model_projected, model_audit = self._model_projection(
            snapshot=model_snapshot,
            family=family,
            outcome_metric_id=metric_id,
            raw=raw,
        )
        stratum, rows, family_support = self._empirical_rows(
            actions=actions,
            changed_paths=changed_paths,
            family=family,
            metric_id=metric_id,
            current_parent_value=current_parent_value,
            scale=scale,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
        )
        if not rows:
            return model_projected, {
                "metric_id": raw.metric_id,
                "forecast_metric_id": raw.metric_id,
                "outcome_metric_id": metric_id,
                "stratum": stratum,
                "support_count": family_support,
                "effective_support_hex": (0.0).hex(),
                "empirical_authority_hex": (0.0).hex(),
                "model_calibration": model_audit,
                "empirical_prequential_skill": None,
                "raw": raw.to_record(),
                "calibrated": model_projected.to_record(),
            }

        weights = tuple(weight for _, weight in rows)
        effective = _effective_support(weights)
        support_authority = min(
            self.maximum_empirical_authority,
            effective / (effective + self.prior_strength),
        )
        empirical_skill = self._empirical_prequential_skill(
            actions=actions,
            family=family,
            metric_id=metric_id,
            scale=scale,
        )
        authority_multiplier = float.fromhex(
            str(empirical_skill["authority_multiplier_hex"])
        )
        authority = support_authority * authority_multiplier
        empirical = (
            _weighted_quantile(rows, 0.1),
            _weighted_quantile(rows, 0.5),
            _weighted_quantile(rows, 0.9),
        )
        model_values = (
            model_projected.p10_delta,
            model_projected.p50_delta,
            model_projected.p90_delta,
        )
        calibrated_values = tuple(
            (1.0 - authority) * model_value + authority * empirical_value
            for model_value, empirical_value in zip(
                model_values,
                empirical,
                strict=True,
            )
        )
        empirical_agreement = (
            1.0
            if model_projected.p50_delta == 0.0 or empirical[1] == 0.0
            else 1.0
            if math.copysign(1.0, model_projected.p50_delta)
            == math.copysign(1.0, empirical[1])
            else 0.0
        )
        empirical_confidence = effective / (effective + self.prior_strength)
        calibrated_confidence = (
            1.0 - authority
        ) * model_projected.confidence + authority * empirical_confidence * (
            0.5 + 0.5 * empirical_agreement
        )
        calibrated = ResolvedActionMetricForecast(
            metric_id=raw.metric_id,
            p10_delta=float(calibrated_values[0]),
            p50_delta=float(calibrated_values[1]),
            p90_delta=float(calibrated_values[2]),
            confidence=float(min(1.0, max(0.0, calibrated_confidence))),
            citations=raw.citations,
        )
        return calibrated, {
            "metric_id": raw.metric_id,
            "forecast_metric_id": raw.metric_id,
            "outcome_metric_id": metric_id,
            "stratum": stratum,
            "support_count": len(rows),
            "effective_support_hex": effective.hex(),
            "support_authority_hex": support_authority.hex(),
            "empirical_authority_hex": authority.hex(),
            "empirical_quantiles_hex": [value.hex() for value in empirical],
            "parent_metric_value_hex": current_parent_value.hex(),
            "metric_scale_hex": scale.hex(),
            "model_calibration": model_audit,
            "empirical_prequential_skill": empirical_skill,
            "raw": raw.to_record(),
            "calibrated": calibrated.to_record(),
        }

    def calibrate(
        self,
        *,
        request: ActionForecastRequest,
        forecasts: ResolvedActionForecastBatch,
        cutoff_wave_index_exclusive: int,
        metric_aliases: tuple[TargetMetricAlias, ...] = (),
    ) -> ActionConsequenceCalibrationResult:
        self.__post_init__()
        if type(request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        request.__post_init__()
        if type(forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be an exact resolved batch")
        validate_resolved_action_forecasts(request, forecasts)
        if (
            type(cutoff_wave_index_exclusive) is not int
            or cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        if type(metric_aliases) is not tuple or any(
            type(value) is not TargetMetricAlias for value in metric_aliases
        ):
            raise TypeError("metric_aliases must contain exact TargetMetricAlias values")
        if metric_aliases:
            for value in metric_aliases:
                value.__post_init__()
            forecast_ids = tuple(value.forecast_metric_id for value in metric_aliases)
            target_ids = tuple(value.target_metric_id for value in metric_aliases)
            if len(set(forecast_ids)) != len(forecast_ids):
                raise ValueError("metric aliases must have unique forecast metric IDs")
            if len(set(target_ids)) != len(target_ids):
                raise ValueError("metric aliases must have unique target metric IDs")
            if set(forecast_ids) != set(request.required_metric_ids):
                raise ValueError(
                    "metric aliases must cover the forecast request exactly"
                )
            resolved_aliases = metric_aliases
        else:
            resolved_aliases = tuple(
                TargetMetricAlias(
                    target_metric_id=metric_id,
                    forecast_metric_id=metric_id,
                )
                for metric_id in request.required_metric_ids
            )
        target_by_forecast = {
            value.forecast_metric_id: value.target_metric_id
            for value in resolved_aliases
        }

        actions = self._prior_actions(cutoff_wave_index_exclusive)
        model_snapshot = self.ledger.calibration_snapshot(
            scope=self.scope,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
            prior=self._correctness_prior(),
            family_min_support=self.model_family_min_support,
        )
        paths_by_option = parent_relative_changed_paths_by_option(
            request.finite_variation_contract
        )
        parent_by_metric = {
            value.metric_id: value.value for value in request.parent_metric_values
        }
        scale_by_metric = {
            value.metric_id: value.delta_scale for value in request.metric_scales
        }
        calibrated_forecasts: list[ResolvedActionForecast] = []
        action_audits: list[dict[str, object]] = []
        for forecast in forecasts.forecasts:
            probability_valid, validity_audit = self._validity_projection(
                actions=actions,
                family=forecast.family,
                raw_probability=forecast.probability_valid,
            )
            metrics: list[ResolvedActionMetricForecast] = []
            metric_audits: list[dict[str, object]] = []
            for metric in forecast.metric_forecasts:
                outcome_metric_id = target_by_forecast[metric.metric_id]
                calibrated_metric, metric_audit = self._metric_projection(
                    actions=actions,
                    model_snapshot=model_snapshot,
                    changed_paths=paths_by_option[forecast.option_id],
                    family=forecast.family,
                    metric_id=outcome_metric_id,
                    current_parent_value=parent_by_metric[metric.metric_id],
                    scale=scale_by_metric[metric.metric_id],
                    cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
                    raw=metric,
                )
                metrics.append(calibrated_metric)
                metric_audits.append(metric_audit)
            calibrated = ResolvedActionForecast(
                option_id=forecast.option_id,
                option_identity_sha256=forecast.option_identity_sha256,
                child_configuration_sha256=forecast.child_configuration_sha256,
                family=forecast.family,
                probability_valid=float(probability_valid),
                metric_forecasts=tuple(metrics),
            )
            calibrated_forecasts.append(calibrated)
            action_audits.append(
                {
                    "option_id": forecast.option_id,
                    "family": forecast.family,
                    "changed_paths": list(paths_by_option[forecast.option_id]),
                    "validity": validity_audit,
                    "metric_cells": metric_audits,
                }
            )

        calibrated_batch = ResolvedActionForecastBatch(
            request_sha256=forecasts.request_sha256,
            context_sha256=forecasts.context_sha256,
            optimization_semantics_definition_sha256=(
                forecasts.optimization_semantics_definition_sha256
            ),
            action_semantics_definition_sha256=(
                forecasts.action_semantics_definition_sha256
            ),
            finite_contract_identity_sha256=(forecasts.finite_contract_identity_sha256),
            card_snapshot_sha256=forecasts.card_snapshot_sha256,
            forecasts=tuple(calibrated_forecasts),
            policy_id=EMPIRICAL_CONSEQUENCE_POLICY_ID,
            policy_version=EMPIRICAL_CONSEQUENCE_POLICY_VERSION,
            policy_definition_sha256=(EMPIRICAL_CONSEQUENCE_POLICY_DEFINITION_SHA256),
        )
        validate_resolved_action_forecasts(request, calibrated_batch)
        if len(action_audits) <= self.maximum_embedded_action_audits:
            action_audit_storage: dict[str, object] = {
                "mode": "embedded",
                "action_count": len(action_audits),
            }
            embedded_action_audits: dict[str, object] = {
                "actions": action_audits,
            }
        else:
            if self.audit_artifact_store is None:
                raise RuntimeError(
                    "large consequence-calibration audits require an "
                    "audit_artifact_store"
                )
            action_artifacts: list[dict[str, object]] = []
            for ordinal, action_audit in enumerate(action_audits, start=1):
                artifact = put_json(
                    self.audit_artifact_store,
                    {
                        "schema_version": 1,
                        "artifact_kind": (
                            "empirical_consequence_calibration_action_audit"
                        ),
                        "scope_sha256": self.scope.scope_sha256,
                        "cutoff_wave_index_exclusive": (
                            cutoff_wave_index_exclusive
                        ),
                        "source_forecast_receipt_sha256": (
                            forecasts.receipt_sha256
                        ),
                        "action_ordinal": ordinal,
                        "action": action_audit,
                    },
                )
                action_artifacts.append(
                    {
                        "option_id": action_audit["option_id"],
                        "artifact": _artifact_ref_record(artifact),
                    }
                )
            action_audit_storage = {
                "mode": "content_addressed_external",
                "action_count": len(action_audits),
                "artifacts": action_artifacts,
            }
            embedded_action_audits = {}
        audit = _object(
            {
                "schema_version": 2,
                "scope_sha256": self.scope.scope_sha256,
                "cutoff_wave_index_exclusive": cutoff_wave_index_exclusive,
                "eligible_prior_action_count": len(actions),
                "metric_aliases": [
                    {
                        "target_metric_id": value.target_metric_id,
                        "forecast_metric_id": value.forecast_metric_id,
                    }
                    for value in resolved_aliases
                ],
                "minimum_path_support": self.minimum_path_support,
                "minimum_family_support": self.minimum_family_support,
                "prior_strength_hex": self.prior_strength.hex(),
                "validity_prior_strength_hex": (self.validity_prior_strength.hex()),
                "maximum_empirical_authority_hex": (
                    self.maximum_empirical_authority.hex()
                ),
                "recency_decay_hex": self.recency_decay.hex(),
                "minimum_model_score_support": self.minimum_model_score_support,
                "model_family_min_support": self.model_family_min_support,
                "minimum_empirical_score_support": (
                    self.minimum_empirical_score_support
                ),
                "correctness_prior": self._correctness_prior().to_record(),
                "model_calibration_snapshot_sha256": model_snapshot.snapshot_sha256,
                "action_audit_storage": action_audit_storage,
                **embedded_action_audits,
                "leakage_guard": "only_feedback_wave_lt_exclusive_cutoff",
                "workload_or_model_branches": False,
            }
        )
        return ActionConsequenceCalibrationResult(
            source_forecast_receipt_sha256=forecasts.receipt_sha256,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
            forecasts=calibrated_batch,
            audit=audit,
        )


__all__ = [
    "ActionConsequenceCalibrationPolicy",
    "ActionConsequenceCalibrationResult",
    "EMPIRICAL_CONSEQUENCE_POLICY_DEFINITION_SHA256",
    "EMPIRICAL_CONSEQUENCE_POLICY_ID",
    "EMPIRICAL_CONSEQUENCE_POLICY_VERSION",
    "HierarchicalEmpiricalConsequenceCalibrationPolicy",
]
