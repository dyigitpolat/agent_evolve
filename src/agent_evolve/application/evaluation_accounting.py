"""Workload-neutral accounting for candidate occurrences and physical evaluations.

An optimizer can administer two distinct candidate occurrences that resolve to
the same benchmark phenotype.  The evaluation cache should coalesce that work,
but experiment reports must never silently relabel the cache hit as a fresh
physical evaluation.  This module validates the complete seed/stage partition
and publishes both quantities explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CampaignEvaluationAccounting:
    """Exact occurrence/physical-call partition for one completed campaign.

    ``planned_candidate_occurrences`` is an exact commitment by default.  A
    campaign whose evolutionary operators may *typed-abstain* (for example,
    when no legal crossover pair exists) can additionally publish a
    ``minimum_candidate_occurrences``.  In that mode the frozen plan is a hard
    capacity ceiling and the minimum is the mandatory work floor.  This keeps
    legal operator abstention distinct from both silent budget loss and cache
    reuse: every realized occurrence is still partitioned exactly, while the
    unfilled capacity is reported explicitly.
    """

    planned_candidate_occurrences: int
    seed_occurrences: int
    seed_unique_evaluations: int
    stage_occurrences: tuple[int, ...]
    stage_unique_evaluations: tuple[int, ...]
    candidate_occurrences: int
    unique_evaluations: int
    minimum_candidate_occurrences: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "planned_candidate_occurrences",
            "seed_occurrences",
            "seed_unique_evaluations",
            "candidate_occurrences",
            "unique_evaluations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.minimum_candidate_occurrences is not None and (
            type(self.minimum_candidate_occurrences) is not int
            or self.minimum_candidate_occurrences < 0
        ):
            raise ValueError(
                "minimum_candidate_occurrences must be a non-negative exact "
                "integer or None"
            )
        for name in ("stage_occurrences", "stage_unique_evaluations"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"{name} must be an exact non-negative tuple")
        if len(self.stage_occurrences) != len(self.stage_unique_evaluations):
            raise ValueError("stage occurrence and evaluation tuples must align")
        if self.seed_unique_evaluations > self.seed_occurrences:
            raise ValueError("seed physical evaluations exceed seed occurrences")
        if any(
            unique > occurrences
            for occurrences, unique in zip(
                self.stage_occurrences,
                self.stage_unique_evaluations,
                strict=True,
            )
        ):
            raise ValueError("stage physical evaluations exceed occurrences")
        if self.candidate_occurrences != (
            self.seed_occurrences + sum(self.stage_occurrences)
        ):
            raise ValueError("candidate occurrences do not equal the stage partition")
        if self.unique_evaluations != (
            self.seed_unique_evaluations + sum(self.stage_unique_evaluations)
        ):
            raise ValueError("unique evaluations do not equal the stage partition")
        if self.minimum_candidate_occurrences is None:
            if self.candidate_occurrences != self.planned_candidate_occurrences:
                raise ValueError("completed occurrences differ from the frozen plan")
        elif not (
            self.minimum_candidate_occurrences
            <= self.candidate_occurrences
            <= self.planned_candidate_occurrences
        ):
            raise ValueError(
                "completed occurrences escape the frozen capacity envelope"
            )

    @property
    def planned_underfill_occurrences(self) -> int:
        self.__post_init__()
        return self.planned_candidate_occurrences - self.candidate_occurrences

    @property
    def candidate_capacity_utilization(self) -> float:
        self.__post_init__()
        if self.planned_candidate_occurrences == 0:
            return 1.0
        return self.candidate_occurrences / self.planned_candidate_occurrences

    @property
    def cache_reuse_occurrences(self) -> int:
        self.__post_init__()
        return self.candidate_occurrences - self.unique_evaluations

    @property
    def physical_evaluation_utilization(self) -> float:
        self.__post_init__()
        if self.candidate_occurrences == 0:
            return 1.0
        return self.unique_evaluations / self.candidate_occurrences

    def within_cache_reuse_limit(self, maximum: int) -> bool:
        self.__post_init__()
        if type(maximum) is not int or maximum < 0:
            raise ValueError("maximum must be a non-negative exact integer")
        return self.cache_reuse_occurrences <= maximum

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "planned_candidate_occurrences": self.planned_candidate_occurrences,
            "minimum_candidate_occurrences": self.minimum_candidate_occurrences,
            "candidate_plan_mode": (
                "exact"
                if self.minimum_candidate_occurrences is None
                else "typed_operator_abstention_capacity_envelope"
            ),
            "seed_occurrences": self.seed_occurrences,
            "seed_unique_evaluations": self.seed_unique_evaluations,
            "stage_occurrences": list(self.stage_occurrences),
            "stage_unique_evaluations": list(self.stage_unique_evaluations),
            "candidate_occurrences": self.candidate_occurrences,
            "unique_evaluations": self.unique_evaluations,
            "planned_underfill_occurrences": self.planned_underfill_occurrences,
            "candidate_capacity_utilization_hex": (
                self.candidate_capacity_utilization.hex()
            ),
            "cache_reuse_occurrences": self.cache_reuse_occurrences,
            "physical_evaluation_utilization_hex": (
                self.physical_evaluation_utilization.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class CampaignPortfolioEvidenceAccounting:
    """Exact evidence partition for finite-portfolio campaign outcomes.

    Evaluator-declared candidate infeasibility is a terminal scientific
    outcome, but it cannot publish objective deltas or forecast-calibration
    observations.  This accounting keeps those candidates in the proposal and
    feedback action budget without fabricating numeric evidence for them.
    """

    planned_portfolio_occurrences: int
    portfolio_scored_occurrences: int
    portfolio_candidate_infeasible_occurrences: int
    authenticated_mutation_observations: int
    reflection_source_scored_occurrences: int
    reflection_identifiable_contrasts: int
    forecast_enabled: bool
    planned_selector_receipts: int
    forecast_receipts: int
    forecast_actions: int
    forecast_scored_actions: int
    forecast_candidate_infeasible_actions: int
    objective_metric_count: int
    forecast_observations: int
    reflection_typed_exclusions: int = 0

    def __post_init__(self) -> None:
        if type(self.forecast_enabled) is not bool:
            raise TypeError("forecast_enabled must be an exact bool")
        for name in (
            "planned_portfolio_occurrences",
            "portfolio_scored_occurrences",
            "portfolio_candidate_infeasible_occurrences",
            "authenticated_mutation_observations",
            "reflection_source_scored_occurrences",
            "reflection_identifiable_contrasts",
            "reflection_typed_exclusions",
            "planned_selector_receipts",
            "forecast_receipts",
            "forecast_actions",
            "forecast_scored_actions",
            "forecast_candidate_infeasible_actions",
            "objective_metric_count",
            "forecast_observations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.objective_metric_count == 0:
            raise ValueError("objective_metric_count must be positive")
        if self.reflection_source_scored_occurrences > (
            self.portfolio_scored_occurrences
        ):
            raise ValueError(
                "reflection source scored occurrences exceed portfolio scores"
            )

    @property
    def exact_portfolio_outcome_partition(self) -> bool:
        self.__post_init__()
        return self.planned_portfolio_occurrences == (
            self.portfolio_scored_occurrences
            + self.portfolio_candidate_infeasible_occurrences
        )

    @property
    def exact_authenticated_mutation_evidence(self) -> bool:
        self.__post_init__()
        return (
            self.authenticated_mutation_observations
            == self.portfolio_scored_occurrences
        )

    @property
    def exact_reflection_contrast_accounting(self) -> bool:
        """Compatibility name for the exact typed reflection partition."""

        return self.exact_reflection_evidence_partition

    @property
    def exact_reflection_evidence_partition(self) -> bool:
        self.__post_init__()
        return (
            self.reflection_identifiable_contrasts
            + self.reflection_typed_exclusions
            == self.reflection_source_scored_occurrences
        )

    @property
    def exact_forecast_feedback(self) -> bool:
        self.__post_init__()
        if not self.forecast_enabled:
            return (
                self.planned_selector_receipts == 0
                and self.forecast_receipts == 0
                and self.forecast_actions == 0
                and self.forecast_scored_actions == 0
                and self.forecast_candidate_infeasible_actions == 0
                and self.forecast_observations == 0
            )
        return (
            self.forecast_receipts == self.planned_selector_receipts
            and self.forecast_actions == self.planned_portfolio_occurrences
            and self.forecast_scored_actions == self.portfolio_scored_occurrences
            and self.forecast_candidate_infeasible_actions
            == self.portfolio_candidate_infeasible_occurrences
            and self.forecast_observations
            == self.portfolio_scored_occurrences * self.objective_metric_count
        )

    @property
    def all_exact(self) -> bool:
        self.__post_init__()
        return (
            self.exact_portfolio_outcome_partition
            and self.exact_authenticated_mutation_evidence
            and self.exact_reflection_evidence_partition
            and self.exact_forecast_feedback
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "planned_portfolio_occurrences": self.planned_portfolio_occurrences,
            "portfolio_scored_occurrences": self.portfolio_scored_occurrences,
            "portfolio_candidate_infeasible_occurrences": (
                self.portfolio_candidate_infeasible_occurrences
            ),
            "authenticated_mutation_observations": (
                self.authenticated_mutation_observations
            ),
            "reflection_source_scored_occurrences": (
                self.reflection_source_scored_occurrences
            ),
            "reflection_identifiable_contrasts": (
                self.reflection_identifiable_contrasts
            ),
            "reflection_typed_exclusions": self.reflection_typed_exclusions,
            "forecast_enabled": self.forecast_enabled,
            "planned_selector_receipts": self.planned_selector_receipts,
            "forecast_receipts": self.forecast_receipts,
            "forecast_actions": self.forecast_actions,
            "forecast_scored_actions": self.forecast_scored_actions,
            "forecast_candidate_infeasible_actions": (
                self.forecast_candidate_infeasible_actions
            ),
            "objective_metric_count": self.objective_metric_count,
            "forecast_observations": self.forecast_observations,
            "gates": {
                "exact_portfolio_outcome_partition": (
                    self.exact_portfolio_outcome_partition
                ),
                "exact_authenticated_mutation_evidence": (
                    self.exact_authenticated_mutation_evidence
                ),
                "exact_reflection_contrast_accounting": (
                    self.exact_reflection_contrast_accounting
                ),
                "exact_reflection_evidence_partition": (
                    self.exact_reflection_evidence_partition
                ),
                "exact_forecast_feedback": self.exact_forecast_feedback,
            },
            "all_exact": self.all_exact,
        }


__all__ = [
    "CampaignEvaluationAccounting",
    "CampaignPortfolioEvidenceAccounting",
]
