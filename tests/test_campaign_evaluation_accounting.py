"""Tests for occurrence/physical-evaluation separation."""

from __future__ import annotations

import pytest

from agent_evolve.application.evaluation_accounting import (
    CampaignEvaluationAccounting,
    CampaignPortfolioEvidenceAccounting,
)


def test_accounting_preserves_occurrences_and_exposes_cache_reuse() -> None:
    accounting = CampaignEvaluationAccounting(
        planned_candidate_occurrences=62,
        seed_occurrences=2,
        seed_unique_evaluations=2,
        stage_occurrences=(16, 4, 16, 4, 16, 4),
        stage_unique_evaluations=(15, 4, 16, 4, 15, 4),
        candidate_occurrences=62,
        unique_evaluations=60,
    )
    assert accounting.cache_reuse_occurrences == 2
    assert accounting.physical_evaluation_utilization == 60 / 62
    assert accounting.within_cache_reuse_limit(2) is True
    assert accounting.within_cache_reuse_limit(1) is False
    assert accounting.to_record()["cache_reuse_occurrences"] == 2


def test_accounting_exposes_typed_operator_abstention_without_hiding_work() -> None:
    accounting = CampaignEvaluationAccounting(
        planned_candidate_occurrences=38,
        minimum_candidate_occurrences=26,
        seed_occurrences=2,
        seed_unique_evaluations=2,
        stage_occurrences=(8, 1, 8, 0, 8, 1),
        stage_unique_evaluations=(8, 1, 8, 0, 8, 1),
        candidate_occurrences=28,
        unique_evaluations=28,
    )

    assert accounting.planned_underfill_occurrences == 10
    assert accounting.candidate_capacity_utilization == 28 / 38
    assert accounting.to_record()["candidate_plan_mode"] == (
        "typed_operator_abstention_capacity_envelope"
    )


@pytest.mark.parametrize("candidate_occurrences", [25, 39])
def test_accounting_rejects_occurrences_outside_capacity_envelope(
    candidate_occurrences: int,
) -> None:
    with pytest.raises(ValueError, match="capacity envelope"):
        CampaignEvaluationAccounting(
            planned_candidate_occurrences=38,
            minimum_candidate_occurrences=26,
            seed_occurrences=2,
            seed_unique_evaluations=2,
            stage_occurrences=(candidate_occurrences - 2,),
            stage_unique_evaluations=(candidate_occurrences - 2,),
            candidate_occurrences=candidate_occurrences,
            unique_evaluations=candidate_occurrences,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"candidate_occurrences": 61}, "candidate occurrences"),
        ({"unique_evaluations": 61}, "unique evaluations"),
        (
            {"stage_unique_evaluations": (17, 4, 16, 4, 14, 4)},
            "stage physical evaluations",
        ),
        ({"planned_candidate_occurrences": 63}, "frozen plan"),
    ],
)
def test_accounting_rejects_incomplete_or_impossible_partitions(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "planned_candidate_occurrences": 62,
        "seed_occurrences": 2,
        "seed_unique_evaluations": 2,
        "stage_occurrences": (16, 4, 16, 4, 16, 4),
        "stage_unique_evaluations": (15, 4, 16, 4, 15, 4),
        "candidate_occurrences": 62,
        "unique_evaluations": 60,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        CampaignEvaluationAccounting(**values)  # type: ignore[arg-type]


def test_portfolio_evidence_accounts_for_typed_candidate_infeasibility() -> None:
    accounting = CampaignPortfolioEvidenceAccounting(
        planned_portfolio_occurrences=48,
        portfolio_scored_occurrences=46,
        portfolio_candidate_infeasible_occurrences=2,
        authenticated_mutation_observations=46,
        reflection_source_scored_occurrences=14,
        reflection_identifiable_contrasts=14,
        forecast_enabled=True,
        planned_selector_receipts=6,
        forecast_receipts=6,
        forecast_actions=48,
        forecast_scored_actions=46,
        forecast_candidate_infeasible_actions=2,
        objective_metric_count=2,
        forecast_observations=92,
    )

    assert accounting.exact_portfolio_outcome_partition is True
    assert accounting.exact_authenticated_mutation_evidence is True
    assert accounting.exact_reflection_contrast_accounting is True
    assert accounting.exact_forecast_feedback is True
    assert accounting.all_exact is True
    assert accounting.to_record()["all_exact"] is True


def test_portfolio_evidence_does_not_launder_infeasibility_into_metrics() -> None:
    accounting = CampaignPortfolioEvidenceAccounting(
        planned_portfolio_occurrences=48,
        portfolio_scored_occurrences=46,
        portfolio_candidate_infeasible_occurrences=2,
        authenticated_mutation_observations=48,
        reflection_source_scored_occurrences=14,
        reflection_identifiable_contrasts=16,
        forecast_enabled=True,
        planned_selector_receipts=6,
        forecast_receipts=6,
        forecast_actions=48,
        forecast_scored_actions=48,
        forecast_candidate_infeasible_actions=0,
        objective_metric_count=2,
        forecast_observations=96,
    )

    assert accounting.exact_portfolio_outcome_partition is True
    assert accounting.exact_authenticated_mutation_evidence is False
    assert accounting.exact_reflection_contrast_accounting is False
    assert accounting.exact_forecast_feedback is False
    assert accounting.all_exact is False


def test_reflection_accounting_preserves_typed_non_single_exclusions() -> None:
    accounting = CampaignPortfolioEvidenceAccounting(
        planned_portfolio_occurrences=48,
        portfolio_scored_occurrences=48,
        portfolio_candidate_infeasible_occurrences=0,
        authenticated_mutation_observations=48,
        reflection_source_scored_occurrences=16,
        reflection_identifiable_contrasts=11,
        reflection_typed_exclusions=5,
        forecast_enabled=True,
        planned_selector_receipts=6,
        forecast_receipts=6,
        forecast_actions=48,
        forecast_scored_actions=48,
        forecast_candidate_infeasible_actions=0,
        objective_metric_count=2,
        forecast_observations=96,
    )

    assert accounting.exact_reflection_evidence_partition is True
    assert accounting.exact_reflection_contrast_accounting is True
    assert accounting.all_exact is True
    record = accounting.to_record()
    assert record["schema_version"] == 2
    assert record["reflection_typed_exclusions"] == 5


def test_portfolio_evidence_control_requires_no_forecast_side_channel() -> None:
    accounting = CampaignPortfolioEvidenceAccounting(
        planned_portfolio_occurrences=48,
        portfolio_scored_occurrences=47,
        portfolio_candidate_infeasible_occurrences=1,
        authenticated_mutation_observations=47,
        reflection_source_scored_occurrences=15,
        reflection_identifiable_contrasts=15,
        forecast_enabled=False,
        planned_selector_receipts=0,
        forecast_receipts=0,
        forecast_actions=0,
        forecast_scored_actions=0,
        forecast_candidate_infeasible_actions=0,
        objective_metric_count=2,
        forecast_observations=0,
    )

    assert accounting.all_exact is True
