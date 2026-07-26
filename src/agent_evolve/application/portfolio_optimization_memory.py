"""Workload-neutral signed semantics for optimization-memory consumption.

Reflection cards describe hypotheses, while their evidence lineage contains
engine-authenticated parent-relative metric effects.  Optimization must not
confuse an observed action with a recommendation: a card can be valuable
because it records an action to avoid.  This module therefore derives a
conservative signed disposition exclusively from authenticated evidence and
the benchmark's declared metric senses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
)
from agent_evolve.application.insight_memory import InsightEvidenceLineage
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSense,
    OptimizationSemantics,
)
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.ports.agentic_generator import MetricEffectDirection


PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_ID = (
    "authenticated_dominance_signed_optimization_memory"
)
PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_VERSION = 1
PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:authenticated-dominance-signed-optimization-memory:v1;"
    b"authenticated-identifiable-evidence-only;declared-objective-senses;"
    b"favorable-dose-only;unfavorable-avoidance-advisory;"
    b"tradeoff-neutral-unresolved-no-forced-dose"
).hexdigest()


class PortfolioOptimizationMemoryDisposition(str, Enum):
    """How an empirically observed action relates to declared objectives."""

    DOMINANCE_FAVORABLE = "dominance_favorable"
    DOMINANCE_UNFAVORABLE = "dominance_unfavorable"
    TRADEOFF = "tradeoff"
    NEUTRAL = "neutral"
    UNRESOLVED = "unresolved"


class PortfolioOptimizationMemoryDirective(str, Enum):
    """Prompt-safe consumption rule derived from a signed disposition."""

    PREFER_OR_TEST = "prefer_or_test_supported_action"
    AVOID_EXCEPT_FALSIFICATION = (
        "avoid_supported_action_except_explicit_falsification"
    )
    CONSIDER_FOR_FRONTIER_TRADEOFF = "consider_only_for_frontier_tradeoff"
    DO_NOT_PRIORITIZE = "do_not_prioritize_supported_action"
    TREAT_AS_UNVERIFIED = "treat_as_unverified_context"


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationMetricSign:
    """Authenticated consensus direction for one declared objective."""

    metric_id: str
    metric_name: str
    sense: MetricSense
    observed_direction: MetricEffectDirection
    signed_effect: int
    observation_count: int

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be non-empty")
        if type(self.metric_name) is not str or not self.metric_name:
            raise ValueError("metric_name must be non-empty")
        if type(self.sense) is not MetricSense:
            raise TypeError("sense must be an exact MetricSense")
        if type(self.observed_direction) is not MetricEffectDirection:
            raise TypeError("observed_direction must be exact")
        if self.observed_direction is MetricEffectDirection.UNKNOWN:
            raise ValueError("authenticated observations cannot be unknown")
        if type(self.signed_effect) is not int or self.signed_effect not in {-1, 0, 1}:
            raise ValueError("signed_effect must be -1, zero, or one")
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ValueError("observation_count must be positive")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "sense": self.sense.value,
            "observed_direction": self.observed_direction.value,
            "signed_effect": self.signed_effect,
            "observation_count": self.observation_count,
        }


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationMemoryAssessment:
    """Auditable signed treatment decision for one reflected-card lineage."""

    disposition: PortfolioOptimizationMemoryDisposition
    directive: PortfolioOptimizationMemoryDirective
    metric_signs: tuple[PortfolioOptimizationMetricSign, ...]
    lineage_identity_sha256: str
    optimization_semantics_definition_sha256: str
    evidence_snapshot_count: int
    unresolved_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not PortfolioOptimizationMemoryDisposition:
            raise TypeError("disposition must be exact")
        if type(self.directive) is not PortfolioOptimizationMemoryDirective:
            raise TypeError("directive must be exact")
        if type(self.metric_signs) is not tuple or any(
            type(value) is not PortfolioOptimizationMetricSign
            for value in self.metric_signs
        ):
            raise TypeError("metric_signs must contain exact values")
        for value in self.metric_signs:
            value.__post_init__()
        if self.metric_signs != tuple(
            sorted(self.metric_signs, key=lambda value: value.metric_id)
        ):
            raise ValueError("metric_signs must use canonical metric order")
        for name in (
            "lineage_identity_sha256",
            "optimization_semantics_definition_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if type(self.evidence_snapshot_count) is not int or self.evidence_snapshot_count < 0:
            raise ValueError("evidence_snapshot_count must be non-negative")
        if self.disposition is PortfolioOptimizationMemoryDisposition.UNRESOLVED:
            if type(self.unresolved_reason) is not str or not self.unresolved_reason:
                raise ValueError("unresolved assessment requires a reason")
            if self.metric_signs:
                raise ValueError("unresolved assessment cannot publish partial signs")
        elif self.unresolved_reason is not None or not self.metric_signs:
            raise ValueError("resolved assessment requires signs and no reason")
        expected_directive = {
            PortfolioOptimizationMemoryDisposition.DOMINANCE_FAVORABLE: (
                PortfolioOptimizationMemoryDirective.PREFER_OR_TEST
            ),
            PortfolioOptimizationMemoryDisposition.DOMINANCE_UNFAVORABLE: (
                PortfolioOptimizationMemoryDirective.AVOID_EXCEPT_FALSIFICATION
            ),
            PortfolioOptimizationMemoryDisposition.TRADEOFF: (
                PortfolioOptimizationMemoryDirective.CONSIDER_FOR_FRONTIER_TRADEOFF
            ),
            PortfolioOptimizationMemoryDisposition.NEUTRAL: (
                PortfolioOptimizationMemoryDirective.DO_NOT_PRIORITIZE
            ),
            PortfolioOptimizationMemoryDisposition.UNRESOLVED: (
                PortfolioOptimizationMemoryDirective.TREAT_AS_UNVERIFIED
            ),
        }[self.disposition]
        if self.directive is not expected_directive:
            raise ValueError("directive differs from the signed disposition")

    @property
    def forced_action_dose_allowed(self) -> bool:
        self.__post_init__()
        return self.disposition is (
            PortfolioOptimizationMemoryDisposition.DOMINANCE_FAVORABLE
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_ID,
                "policy_version": PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_VERSION,
                "definition_sha256": (
                    PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_DEFINITION_SHA256
                ),
            },
            "disposition": self.disposition.value,
            "directive": self.directive.value,
            "forced_action_dose_allowed": self.forced_action_dose_allowed,
            "metric_signs": [value.to_record() for value in self.metric_signs],
            "lineage_identity_sha256": self.lineage_identity_sha256,
            "optimization_semantics_definition_sha256": (
                self.optimization_semantics_definition_sha256
            ),
            "evidence_snapshot_count": self.evidence_snapshot_count,
            "unresolved_reason": self.unresolved_reason,
            "evidence_authority": "authenticated_empirical_snapshots_only",
            "model_prose_used_for_sign": False,
        }


def _unresolved(
    lineage: InsightEvidenceLineage,
    semantics: OptimizationSemantics,
    reason: str,
) -> PortfolioOptimizationMemoryAssessment:
    return PortfolioOptimizationMemoryAssessment(
        disposition=PortfolioOptimizationMemoryDisposition.UNRESOLVED,
        directive=PortfolioOptimizationMemoryDirective.TREAT_AS_UNVERIFIED,
        metric_signs=(),
        lineage_identity_sha256=lineage.identity_sha256,
        optimization_semantics_definition_sha256=semantics.definition_sha256,
        evidence_snapshot_count=len(lineage.empirical_evidence),
        unresolved_reason=reason,
    )


def _signed_effect(
    direction: MetricEffectDirection,
    sense: MetricSense,
) -> int | None:
    if direction is MetricEffectDirection.UNCHANGED:
        return 0
    if sense is MetricSense.MINIMIZE:
        return 1 if direction is MetricEffectDirection.DECREASE else -1
    if sense is MetricSense.MAXIMIZE:
        return 1 if direction is MetricEffectDirection.INCREASE else -1
    return None


def assess_portfolio_optimization_memory(
    lineage: InsightEvidenceLineage,
    semantics: OptimizationSemantics,
) -> PortfolioOptimizationMemoryAssessment:
    """Classify one card without consulting provider prose or future outcomes."""

    if type(lineage) is not InsightEvidenceLineage:
        raise TypeError("lineage must be exact")
    if type(semantics) is not OptimizationSemantics:
        raise TypeError("semantics must be exact")
    lineage.__post_init__()
    semantics.__post_init__()
    snapshots = lineage.empirical_evidence
    if not snapshots:
        return _unresolved(lineage, semantics, "no_authenticated_empirical_evidence")

    objective_metrics = tuple(
        metric for metric in semantics.metrics if metric.role is MetricRole.OBJECTIVE
    )
    if not objective_metrics:
        return _unresolved(lineage, semantics, "no_declared_objectives")
    metric_by_external_id = {}
    for metric in objective_metrics:
        for key in (metric.metric_id, metric.name):
            prior = metric_by_external_id.get(key)
            if prior is not None and prior != metric:
                return _unresolved(lineage, semantics, "ambiguous_metric_identifier")
            metric_by_external_id[key] = metric

    directions: dict[str, list[MetricEffectDirection]] = {
        metric.metric_id: [] for metric in objective_metrics
    }
    for snapshot in snapshots:
        if (
            snapshot.fact_schema_id != IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID
            or snapshot.fact_schema_version
            != IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION
            or snapshot.fact_schema_definition_sha256
            != IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        ):
            return _unresolved(lineage, semantics, "foreign_empirical_fact_schema")
        if (
            snapshot.optimization_semantics_definition_sha256
            != semantics.definition_sha256
        ):
            return _unresolved(lineage, semantics, "optimization_semantics_mismatch")
        facts = thaw_json(snapshot.facts)
        rows = facts.get("observed_metric_effects")
        if type(rows) is not list:
            return _unresolved(lineage, semantics, "missing_observed_metric_effects")
        observed_metric_ids: set[str] = set()
        for row in rows:
            if type(row) is not dict:
                return _unresolved(lineage, semantics, "malformed_metric_effect")
            external_id = row.get("metric_id")
            direction_text = row.get("direction")
            if type(external_id) is not str or external_id in observed_metric_ids:
                return _unresolved(lineage, semantics, "duplicate_or_invalid_metric")
            observed_metric_ids.add(external_id)
            metric = metric_by_external_id.get(external_id)
            if metric is None:
                continue
            try:
                direction = MetricEffectDirection(direction_text)
            except (TypeError, ValueError):
                return _unresolved(lineage, semantics, "invalid_metric_direction")
            if direction is MetricEffectDirection.UNKNOWN:
                return _unresolved(lineage, semantics, "unknown_metric_direction")
            directions[metric.metric_id].append(direction)

    signs: list[PortfolioOptimizationMetricSign] = []
    for metric in objective_metrics:
        values = directions[metric.metric_id]
        if len(values) != len(snapshots):
            return _unresolved(lineage, semantics, "incomplete_objective_coverage")
        unique = set(values)
        if len(unique) != 1:
            return _unresolved(lineage, semantics, "conflicting_observed_directions")
        direction = values[0]
        signed = _signed_effect(direction, metric.sense)
        if signed is None:
            return _unresolved(lineage, semantics, "non_monotone_objective_sense")
        signs.append(
            PortfolioOptimizationMetricSign(
                metric_id=metric.metric_id,
                metric_name=metric.name,
                sense=metric.sense,
                observed_direction=direction,
                signed_effect=signed,
                observation_count=len(values),
            )
        )

    positive = sum(value.signed_effect > 0 for value in signs)
    negative = sum(value.signed_effect < 0 for value in signs)
    if positive and not negative:
        disposition = PortfolioOptimizationMemoryDisposition.DOMINANCE_FAVORABLE
        directive = PortfolioOptimizationMemoryDirective.PREFER_OR_TEST
    elif negative and not positive:
        disposition = PortfolioOptimizationMemoryDisposition.DOMINANCE_UNFAVORABLE
        directive = PortfolioOptimizationMemoryDirective.AVOID_EXCEPT_FALSIFICATION
    elif positive and negative:
        disposition = PortfolioOptimizationMemoryDisposition.TRADEOFF
        directive = (
            PortfolioOptimizationMemoryDirective.CONSIDER_FOR_FRONTIER_TRADEOFF
        )
    else:
        disposition = PortfolioOptimizationMemoryDisposition.NEUTRAL
        directive = PortfolioOptimizationMemoryDirective.DO_NOT_PRIORITIZE
    return PortfolioOptimizationMemoryAssessment(
        disposition=disposition,
        directive=directive,
        metric_signs=tuple(sorted(signs, key=lambda value: value.metric_id)),
        lineage_identity_sha256=lineage.identity_sha256,
        optimization_semantics_definition_sha256=semantics.definition_sha256,
        evidence_snapshot_count=len(snapshots),
    )


__all__ = [
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_ID",
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_VERSION",
    "PortfolioOptimizationMemoryAssessment",
    "PortfolioOptimizationMemoryDirective",
    "PortfolioOptimizationMemoryDisposition",
    "PortfolioOptimizationMetricSign",
    "assess_portfolio_optimization_memory",
]
