from __future__ import annotations

import numpy as np

from examples.development import analyze_trap_prequential_k8 as trap


def test_reference_directions_match_authenticated_affine_contract() -> None:
    assert trap._reference_directions(2) == (
        ("axis_1_extreme", (1.0, 0.0)),
        ("axis_2_extreme", (0.0, 1.0)),
        ("balanced_tradeoff", (1.0, 1.0)),
    )
    assert trap._reference_directions(3)[-1] == (
        "balanced_tradeoff",
        (1.0, 1.0, 1.0),
    )


def test_boils_lane_targets_use_only_archive_and_parent_geometry() -> None:
    snapshot = {
        "reference_point": [
            ["first", 1.0.hex()],
            ["second", 1.0.hex()],
        ],
        "archive_points": [
            [["first", 0.8.hex()], ["second", 0.2.hex()]],
        ],
        "snapshot_hash": "fixed",
    }
    pair = [
        {
            "generation": 1,
            "members": [{"parent_slot": 0}],
            "parent_objectives": {"first": 0.7, "second": 0.3},
            "archive_reward_snapshot": snapshot,
        },
        {
            "generation": 1,
            "members": [{"parent_slot": 1}],
            "parent_objectives": {"first": 0.4, "second": 0.6},
            "archive_reward_snapshot": snapshot,
        },
    ]

    targets = trap._lane_targets(workload="boils", raw_pair=pair)

    assert targets[1].direction_id == "axis_1_extreme"
    assert targets[1].opportunity_rank == 1
    assert targets[0].direction_id == "balanced_tradeoff"
    assert targets[0].opportunity_rank == 2
    assert all(value.remaining_proposal_horizon == 2 for value in targets.values())


def test_sequential_ridge_update_is_immutable_and_selected_row_only() -> None:
    prior = trap.SequentialRidgeHead(
        feature_names=("bias", "signal"),
        means=np.asarray([0.0, 0.0]),
        scales=np.asarray([1.0, 1.0]),
        precision=np.eye(2),
        rhs=np.zeros(2),
        covariance=np.eye(2),
        coefficients=np.zeros(2),
        residual_variance=1.0,
    )

    posterior = prior.update([{"bias": 1.0, "signal": 2.0}], [1.0])

    assert prior.predict({"bias": 1.0, "signal": 2.0}) == 0.0
    assert posterior.predict({"bias": 1.0, "signal": 2.0}) > 0.0
    assert np.array_equal(prior.precision, np.eye(2))
    assert not np.array_equal(posterior.precision, prior.precision)
