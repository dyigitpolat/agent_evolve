from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_evolve import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    ObjectiveSpec,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    render_optimization_semantics,
)


_RELATION_HASH = "9" * 64


def _semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="generic_targeted_design",
        semantics_version=3,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Total evaluated resource cost.",
                aggregation="Sum across all scheduled resources.",
                witness_interpretation="Each resource_cost witness contributes once.",
                tolerance=0.0,
            ),
            MetricSemantics(
                metric_id="violation:target_error",
                name="target_error",
                role=MetricRole.VIOLATION,
                sense=MetricSense.MINIMIZE,
                definition="Absolute deviation from the response target.",
                aggregation="One absolute residual.",
                witness_interpretation=(
                    "A negative signed residual is below target; move upward."
                ),
                reference_target=2.5,
                tolerance=0.01,
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.LEXICOGRAPHIC,
            metric_priority=("violation:target_error", "objective:cost"),
            description="Minimize target error first, then cost.",
            equivalence="Both values must be exactly equal.",
            policy_id="target_then_cost",
            policy_version=2,
            definition_sha256=_RELATION_HASH,
        ),
    )


def test_semantics_are_frozen_hashable_versioned_and_canonically_rendered() -> None:
    first = _semantics()
    second = _semantics()

    assert first == second
    assert hash(first) == hash(second)
    assert first.identity == (
        "generic_targeted_design",
        3,
        first.definition_sha256,
    )
    assert first.definition_sha256 == second.definition_sha256
    record = first.to_record()
    assert record["definition_sha256"] == first.definition_sha256
    assert record["outcome_ordering"]["relation_policy"] == {
        "policy_id": "target_then_cost",
        "policy_version": 2,
        "definition_sha256": _RELATION_HASH,
    }
    rendered = render_optimization_semantics(first)
    assert rendered.startswith("OPTIMIZATION SEMANTICS (VERSIONED, AUTHORITATIVE)")
    assert rendered.count(first.definition_sha256) == 1
    with pytest.raises(FrozenInstanceError):
        first.semantics_version = 4  # type: ignore[misc]


def test_semantics_binding_rejects_objective_or_relation_drift() -> None:
    semantics = _semantics()
    semantics.validate_binding(
        (ObjectiveSpec("cost", "min"),),
        ("target_then_cost", 2, _RELATION_HASH),
    )

    with pytest.raises(ValueError, match="sense differs"):
        semantics.validate_binding(
            (ObjectiveSpec("cost", "max"),),
            ("target_then_cost", 2, _RELATION_HASH),
        )
    with pytest.raises(ValueError, match="outcome ordering differs"):
        semantics.validate_binding(
            (ObjectiveSpec("cost", "min"),),
            ("different_relation", 2, _RELATION_HASH),
        )


def test_metric_contract_rejects_role_prefix_and_target_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="match role"):
        MetricSemantics(
            metric_id="violation:cost",
            name="cost",
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition="Cost.",
            aggregation="Scalar.",
            witness_interpretation="Lower is better.",
        )
    with pytest.raises(ValueError, match="require reference_target"):
        MetricSemantics(
            metric_id="constraint:setpoint",
            name="setpoint",
            role=MetricRole.CONSTRAINT,
            sense=MetricSense.TARGET,
            definition="Setpoint response.",
            aggregation="Scalar.",
            witness_interpretation="Move toward the target.",
        )
