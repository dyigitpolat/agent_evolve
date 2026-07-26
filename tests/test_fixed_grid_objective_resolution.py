from __future__ import annotations

from decimal import Decimal

import pytest

import agent_evolve.policies as public_policies
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.policies.objective_resolution import (
    FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_ID,
    FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_VERSION,
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionRequest,
    resolve_objectives,
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _request(
    *,
    objectives: tuple[ObjectiveSpec, ...],
    raw: tuple[tuple[str, float], ...],
    configuration: dict[str, object] | None = None,
) -> ObjectiveResolutionRequest:
    return ObjectiveResolutionRequest(
        configuration=_object(configuration or {"candidate": "a"}),
        objectives=objectives,
        raw_objectives=raw,
    )


def _spec(
    metric_id: str,
    *,
    origin: str = "0",
    quantum: str = "0.1",
    law: FixedGridRoundingLaw = FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN,
) -> FixedGridMetricSpec:
    return FixedGridMetricSpec(
        metric_id=metric_id,
        decimal_origin=Decimal(origin),
        decimal_quantum=Decimal(quantum),
        rounding_law=law,
    )


def test_positive_and_negative_values_resolve_while_raw_evidence_is_preserved() -> None:
    objectives = (
        ObjectiveSpec("quality", "max"),
        ObjectiveSpec("loss", "min"),
    )
    policy = FixedGridObjectiveResolution(
        metric_specs=(_spec("loss"), _spec("quality")),
    )
    receipt = resolve_objectives(
        policy,
        _request(
            objectives=objectives,
            raw=(("quality", 1.24), ("loss", -1.26)),
        ),
    )

    assert receipt.raw_objectives == (("quality", 1.24), ("loss", -1.26))
    assert receipt.decision_objectives == (("quality", 1.2), ("loss", -1.3))
    evidence = thaw_json(receipt.evidence)
    assert type(evidence) is dict
    assert evidence["configuration_dependence"] == "none"
    assert [row["metric_id"] for row in evidence["metrics"]] == [
        "quality",
        "loss",
    ]
    assert evidence["metrics"][0]["raw_value_hex"] == 1.24.hex()
    assert evidence["metrics"][0]["resolved_decimal"] == "1.2"
    assert evidence["metrics"][1]["raw_value_hex"] == (-1.26).hex()
    assert evidence["metrics"][1]["resolved_decimal"] == "-1.3"


def test_resolution_is_min_max_agnostic_and_follows_objective_order() -> None:
    policy = FixedGridObjectiveResolution(
        metric_specs=(_spec("a", quantum="1"), _spec("b", quantum="1")),
    )
    raw = (("b", 1.6), ("a", -1.6))
    max_min = resolve_objectives(
        policy,
        _request(
            objectives=(ObjectiveSpec("b", "max"), ObjectiveSpec("a", "min")),
            raw=raw,
        ),
    )
    min_max = resolve_objectives(
        policy,
        _request(
            objectives=(ObjectiveSpec("b", "min"), ObjectiveSpec("a", "max")),
            raw=raw,
        ),
    )

    assert max_min.decision_objectives == (("b", 2.0), ("a", -2.0))
    assert min_max.decision_objectives == max_min.decision_objectives
    assert thaw_json(min_max.evidence) == thaw_json(max_min.evidence)


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((2.5, 2.0), (3.5, 4.0), (-2.5, -2.0), (-3.5, -4.0)),
)
def test_half_boundary_ties_to_even_is_symmetric(raw: float, expected: float) -> None:
    receipt = resolve_objectives(
        FixedGridObjectiveResolution(metric_specs=(_spec("metric", quantum="1"),)),
        _request(
            objectives=(ObjectiveSpec("metric", "max"),),
            raw=(("metric", raw),),
        ),
    )

    assert receipt.decision_objectives == (("metric", expected),)


@pytest.mark.parametrize(("raw", "expected"), ((0.5, 1.0), (-0.5, -1.0)))
def test_half_boundary_ties_away_from_zero_is_declared(
    raw: float,
    expected: float,
) -> None:
    receipt = resolve_objectives(
        FixedGridObjectiveResolution(
            metric_specs=(
                _spec(
                    "metric",
                    quantum="1",
                    law=FixedGridRoundingLaw.NEAREST_TIES_AWAY_FROM_ZERO,
                ),
            )
        ),
        _request(
            objectives=(ObjectiveSpec("metric", "min"),),
            raw=(("metric", raw),),
        ),
    )

    assert receipt.decision_objectives == (("metric", expected),)


def test_nonzero_origin_and_resolve_objectives_idempotence() -> None:
    policy = FixedGridObjectiveResolution(
        metric_specs=(_spec("metric", origin="0.05", quantum="0.1"),),
    )
    request = _request(
        objectives=(ObjectiveSpec("metric", "max"),),
        raw=(("metric", 0.1),),
    )
    receipt = resolve_objectives(policy, request)
    repeated = resolve_objectives(policy, request)
    decision_request = _request(
        objectives=request.objectives,
        raw=receipt.decision_objectives,
        configuration={"candidate": "same-grid-point"},
    )
    decision_receipt = resolve_objectives(policy, decision_request)

    assert receipt.decision_objectives == (("metric", 0.05),)
    assert repeated.receipt_sha256 == receipt.receipt_sha256
    assert decision_receipt.decision_objectives == receipt.decision_objectives


def test_candidate_configuration_is_not_an_implicit_resolution_input() -> None:
    policy = FixedGridObjectiveResolution(metric_specs=(_spec("metric"),))
    objectives = (ObjectiveSpec("metric", "max"),)
    first = resolve_objectives(
        policy,
        _request(
            objectives=objectives,
            raw=(("metric", 1.24),),
            configuration={"mesh": "coarse", "hidden_digits": 1},
        ),
    )
    second = resolve_objectives(
        policy,
        _request(
            objectives=objectives,
            raw=(("metric", 1.24),),
            configuration={"mesh": "fine", "hidden_digits": 9},
        ),
    )

    assert first.request_sha256 != second.request_sha256
    assert first.configuration_sha256 != second.configuration_sha256
    assert first.decision_objectives == second.decision_objectives
    assert thaw_json(first.evidence) == thaw_json(second.evidence)
    assert policy.to_record()["configuration_dependence"] == "none"


@pytest.mark.parametrize(
    "factory",
    (
        lambda: FixedGridMetricSpec(
            "metric", "0", Decimal("1"), FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN
        ),
        lambda: FixedGridMetricSpec(
            "metric", Decimal("NaN"), Decimal("1"), FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN
        ),
        lambda: _spec("metric", quantum="0"),
        lambda: _spec("metric", quantum="-1"),
        lambda: FixedGridMetricSpec(
            "metric", Decimal("0"), Decimal("1"), "nearest_ties_to_even"
        ),
        lambda: _spec(" metric"),
    ),
)
def test_malformed_metric_specs_fail_closed(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_policy_rejects_empty_duplicate_noncanonical_and_incomplete_specs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FixedGridObjectiveResolution(metric_specs=())
    with pytest.raises(ValueError, match="repeat"):
        FixedGridObjectiveResolution(metric_specs=(_spec("a"), _spec("a")))
    with pytest.raises(ValueError, match="canonical"):
        FixedGridObjectiveResolution(metric_specs=(_spec("b"), _spec("a")))

    request = _request(
        objectives=(ObjectiveSpec("a", "max"), ObjectiveSpec("b", "min")),
        raw=(("a", 1.0), ("b", 2.0)),
    )
    with pytest.raises(ValueError, match="cover every objective"):
        resolve_objectives(
            FixedGridObjectiveResolution(metric_specs=(_spec("a"),)),
            request,
        )
    with pytest.raises(ValueError, match=r"missing=b.*extra=c"):
        resolve_objectives(
            FixedGridObjectiveResolution(metric_specs=(_spec("a"), _spec("c"))),
            request,
        )


def test_semantically_equal_decimal_spellings_have_one_policy_identity() -> None:
    first = FixedGridObjectiveResolution(
        metric_specs=(_spec("metric", origin="0.00", quantum="0.10"),),
    )
    second = FixedGridObjectiveResolution(
        metric_specs=(_spec("metric", origin="-0", quantum="0.1"),),
    )

    assert first.definition_sha256 == second.definition_sha256
    assert first.to_record()["metric_specs"] == second.to_record()["metric_specs"]


def test_fixed_grid_policy_is_exported_from_the_public_policy_layer() -> None:
    assert public_policies.FixedGridMetricSpec is FixedGridMetricSpec
    assert public_policies.FixedGridObjectiveResolution is FixedGridObjectiveResolution
    assert public_policies.FixedGridRoundingLaw is FixedGridRoundingLaw
    policy = FixedGridObjectiveResolution(metric_specs=(_spec("metric"),))
    assert policy.policy_id == FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_ID
    assert policy.policy_version == FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_VERSION
    assert len(policy.definition_sha256) == 64
