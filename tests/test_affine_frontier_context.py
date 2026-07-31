from __future__ import annotations

import hashlib
import json

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import (
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.application.portfolio_campaign_runtime import (
    ResidualHypervolumeCampaignParentSelector,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import freeze_json, thaw_json, typed_json_sha256
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineHypervolume2DSpec,
    AffineHypervolumeArchiveUtility,
    AffineObjectiveAxis,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (
    AffineHypervolume3DSpec,
    AffineHypervolumeArchiveUtility3D,
)
from agent_evolve.policies.selection.affine_frontier_context import (
    AffineFrontierContextMode,
    AuthenticatedAffineFrontierContextProjector,
    affine_frontier_context_projector,
)
from agent_evolve.policies.selection.affine_frontier_target import (
    AuthenticatedAffineFrontierTargetAllocator,
    DirectionCoveredAffineFrontierTargetAllocator,
    GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator,
)
from agent_evolve.policies.selection.residual_frontier import (
    residual_anchor_parents,
    residual_frontier_geometry,
)
from agent_evolve.policies.selection.residual_frontier_target import (
    DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID,
    ResidualHypervolumeFrontierTargetAllocator,
)
from agent_evolve.ports.frontier_target import (
    objective_space_target_from_campaign_target,
)


def _archive(points: tuple[dict[str, float], ...]):
    return freeze_json(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": metric_id, "value_hex": value.hex()}
                        for metric_id, value in point.items()
                    ]
                }
                for point in points
            ]
        }
    )


def _parent(objectives: dict[str, float]) -> EvolutionCandidate:
    configuration = freeze_json({"design_variables": [1, 3, 5]})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId("candidate_frontier_parent"),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                b"frontier-parent-artifact"
            ).hexdigest(),
            proposal_sequence=1,
        ),
        configuration=configuration,
        objectives=tuple(objectives.items()),
        valid=True,
        generation=0,
        label="frontier-parent",
    )


def _named_parent(
    name: str,
    objectives: dict[str, float],
) -> EvolutionCandidate:
    configuration = freeze_json({"design_variables": [name, 1, 3, 5]})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_{name}"),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                f"frontier-parent-artifact:{name}".encode("ascii")
            ).hexdigest(),
            proposal_sequence=1,
        ),
        configuration=configuration,
        objectives=tuple(objectives.items()),
        valid=True,
        generation=0,
        label=f"frontier-parent-{name}",
    )


def test_two_dimensional_projection_is_exact_decimal_and_workload_neutral() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("quality", "max", 100.0, 0.0),
            AffineObjectiveAxis("cost", "min", 0.0, 20.0),
        ),
        reference_provenance="fixed before candidate generation",
    )
    points = (
        {"quality": 80.0, "cost": 12.0},
        {"quality": 60.0, "cost": 6.0},
    )
    utility = AffineHypervolumeArchiveUtility(spec)
    snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "generic-2d"}),
        generation=3,
        archive=_archive(points),
    )
    parent = _parent({"quality": 70.0, "cost": 8.0})
    projector = AuthenticatedAffineFrontierContextProjector()

    first = projector.project(archive_utility=snapshot, parent=parent)
    second = projector.project(archive_utility=snapshot, parent=parent)
    record = first.to_record()
    payload = thaw_json(first.payload)

    assert first.projection_sha256 == second.projection_sha256
    assert first.to_record() == second.to_record()
    assert record["archive_utility_snapshot_sha256"] == snapshot.snapshot_sha256
    assert payload["optimization_frame"]["dimension"] == 2
    assert payload["optimization_frame"]["normalized_orientation"] == (
        "lower_is_better_on_every_axis"
    )
    assert payload["parent"]["normalized_point_decimal"] == [
        "0.29999999999999999",
        "0.40000000000000002",
    ]
    assert len(payload["optimization_frame"]["reference_directions"]) == 3
    serialized = json.dumps(record, sort_keys=True)
    assert "0x" not in serialized
    for forbidden in ("workload_id", "model_id", "provider", "action_name"):
        assert forbidden not in serialized
    assert (
        payload["epistemic_cutoff"]["current_or_future_candidate_outcomes_consulted"]
        is False
    )


def test_three_dimensional_projection_uses_same_port_and_generic_contract() -> None:
    spec = AffineHypervolume3DSpec(
        axes=(
            AffineObjectiveAxis("metric_a", "min", 0.0, 10.0),
            AffineObjectiveAxis("metric_b", "min", 0.0, 20.0),
            AffineObjectiveAxis("metric_c", "min", 0.0, 40.0),
        ),
        reference_provenance="fixed before candidate generation",
    )
    points = (
        {"metric_a": 3.0, "metric_b": 12.0, "metric_c": 20.0},
        {"metric_a": 6.0, "metric_b": 5.0, "metric_c": 10.0},
    )
    utility = AffineHypervolumeArchiveUtility3D(spec)
    snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "generic-3d"}),
        generation=4,
        archive=_archive(points),
    )
    projection = AuthenticatedAffineFrontierContextProjector().project(
        archive_utility=snapshot,
        parent=_parent({"metric_a": 4.0, "metric_b": 8.0, "metric_c": 16.0}),
    )
    payload = thaw_json(projection.payload)

    assert payload["optimization_frame"]["dimension"] == 3
    assert payload["parent"]["normalized_point_decimal"] == [
        "0.40000000000000002",
        "0.40000000000000002",
        "0.40000000000000002",
    ]
    assert len(payload["optimization_frame"]["reference_directions"]) == 7
    assert len(payload["archive"]["normalized_points_decimal"]) == 2


def test_projection_rejects_parent_from_a_foreign_objective_space() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("first", "min", 0.0, 1.0),
            AffineObjectiveAxis("second", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed before candidate generation",
    )
    utility = AffineHypervolumeArchiveUtility(spec)
    snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "foreign-axis-rejection"}),
        generation=1,
        archive=_archive(({"first": 0.4, "second": 0.5},)),
    )

    with pytest.raises(ValueError, match="objective vector differs"):
        AuthenticatedAffineFrontierContextProjector().project(
            archive_utility=snapshot,
            parent=_parent({"first": 0.4, "foreign": 0.5}),
        )


def test_closed_mode_factory_has_an_explicit_off_arm() -> None:
    assert affine_frontier_context_projector("off") is None
    assert (
        type(
            affine_frontier_context_projector(
                AffineFrontierContextMode.AUTHENTICATED_AFFINE_V1
            )
        )
        is AuthenticatedAffineFrontierContextProjector
    )
    with pytest.raises(ValueError):
        affine_frontier_context_projector("foreign")


def test_frontier_target_allocator_coordinates_distinct_workload_blind_lanes() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("quality", "max", 100.0, 0.0),
            AffineObjectiveAxis("cost", "min", 0.0, 20.0),
        ),
        reference_provenance="fixed before candidate generation",
    )
    utility = AffineHypervolumeArchiveUtility(spec)
    snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "generic-target-2d"}),
        generation=3,
        archive=_archive(
            (
                {"quality": 80.0, "cost": 12.0},
                {"quality": 60.0, "cost": 6.0},
            )
        ),
    )
    parent = _parent({"quality": 70.0, "cost": 8.0})
    allocator = AuthenticatedAffineFrontierTargetAllocator()

    first = allocator.allocate(
        archive_utility=snapshot,
        lanes=(("elite", parent), ("explorer", parent)),
    )
    second = allocator.allocate(
        archive_utility=snapshot,
        lanes=(("elite", parent), ("explorer", parent)),
    )

    assert first == second
    assert tuple(value.lane_id for value in first) == ("elite", "explorer")
    assert len({value.direction_id for value in first}) == 2
    assert all(
        value.archive_utility_snapshot_sha256 == snapshot.snapshot_sha256
        for value in first
    )
    serialized = json.dumps([value.to_record() for value in first], sort_keys=True)
    assert 'current_or_future_candidate_outcomes_consulted": false' in serialized
    for forbidden in (
        '"workload_id":',
        '"model_id":',
        '"provider_id":',
        '"action_name":',
    ):
        assert forbidden not in serialized


def test_direction_covered_target_allocator_rotates_all_axes_before_repeat() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("quality", "max", 100.0, 0.0),
            AffineObjectiveAxis("cost", "min", 0.0, 20.0),
        ),
        reference_provenance="fixed before candidate generation",
    )
    utility = AffineHypervolumeArchiveUtility(spec)
    benchmark = freeze_json({"contract": "generic-direction-coverage-2d"})
    archive = _archive(
        (
            {"quality": 80.0, "cost": 12.0},
            {"quality": 60.0, "cost": 6.0},
        )
    )
    parent = _parent({"quality": 70.0, "cost": 8.0})
    allocator = DirectionCoveredAffineFrontierTargetAllocator()

    by_generation = tuple(
        allocator.allocate(
            archive_utility=utility.freeze(
                benchmark=benchmark,
                generation=generation,
                archive=archive,
            ),
            lanes=(("elite", parent), ("explorer", parent)),
        )
        for generation in (1, 3, 5)
    )
    direction_ids = tuple(
        value.direction_id for wave in by_generation for value in wave
    )

    assert set(direction_ids) == {
        "axis_1_extreme",
        "axis_2_extreme",
        "balanced_tradeoff",
    }
    assert all(
        direction_ids.count(direction_id) == 2 for direction_id in set(direction_ids)
    )
    assert all(
        value.allocator_id == "direction_covered_affine_frontier_target"
        and value.allocator_version == 2
        for wave in by_generation
        for value in wave
    )


def test_residual_frontier_finds_the_positive_knee_not_a_dominated_long_gap() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("thermal", "min", 0.0, 1.0),
            AffineObjectiveAxis("material", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    points = (
        {"thermal": 0.1069841694, "material": 0.0645161290322581},
        {"thermal": 0.0708174948, "material": 0.2580645161290323},
        {"thermal": 0.0580282620, "material": 0.4838709677419356},
    )
    snapshot = AffineHypervolumeArchiveUtility(spec).freeze(
        benchmark=freeze_json({"contract": "residual-knee-2d"}),
        generation=3,
        archive=_archive(points),
    )

    first = residual_frontier_geometry(snapshot)
    second = residual_frontier_geometry(snapshot)

    assert first == second
    assert first.geometry_sha256 == second.geometry_sha256
    assert first.cells
    best = first.cells[0]
    assert best.anchor_points == (
        (0.0708174948, 0.2580645161290323),
        (0.1069841694, 0.0645161290322581),
    )
    assert best.potential_hypervolume_gain == pytest.approx(
        0.0017500003838709643,
        abs=1e-15,
    )
    assert all(
        cell.anchor_points
        != (
            (0.0580282620, 0.4838709677419356),
            (0.1069841694, 0.0645161290322581),
        )
        for cell in first.cells
    )
    record = json.dumps(first.to_record(), sort_keys=True)
    assert '"current_or_future_candidate_outcomes_consulted": false' in record
    assert '"workload_model_provider_fields_consulted": false' in record


def test_global_target_matching_avoids_the_greedy_parent_direction_mismatch() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("thermal", "min", 0.0, 1.0),
            AffineObjectiveAxis("material", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    points = (
        {"thermal": 0.1069841694, "material": 0.0645161290322581},
        {"thermal": 0.0708174948, "material": 0.2580645161290323},
        {"thermal": 0.0580282620, "material": 0.4838709677419356},
    )
    snapshot = AffineHypervolumeArchiveUtility(spec).freeze(
        benchmark=freeze_json({"contract": "global-target-matching"}),
        generation=5,
        archive=_archive(points),
    )
    low_material = _named_parent("low_material", points[0])
    low_thermal = _named_parent("low_thermal", points[2])
    lanes = (("elite", low_material), ("explorer", low_thermal))

    greedy = AuthenticatedAffineFrontierTargetAllocator().allocate(
        archive_utility=snapshot,
        lanes=lanes,
    )
    global_match = (
        GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator().allocate(
            archive_utility=snapshot,
            lanes=lanes,
        )
    )

    greedy_by_parent = {
        value.parent_configuration_sha256: value.direction_id for value in greedy
    }
    global_by_parent = {
        value.parent_configuration_sha256: value.direction_id for value in global_match
    }
    assert greedy_by_parent[low_material.occurrence.configuration_hash] == (
        "balanced_tradeoff"
    )
    assert greedy_by_parent[low_thermal.occurrence.configuration_hash] == (
        "axis_2_extreme"
    )
    assert global_by_parent[low_material.occurrence.configuration_hash] == (
        "axis_2_extreme"
    )
    assert global_by_parent[low_thermal.occurrence.configuration_hash] == (
        "balanced_tradeoff"
    )
    assert all(
        value.allocator_id == "globally_matched_direction_covered_frontier_target"
        and value.allocator_version == 3
        for value in global_match
    )


def test_residual_frontier_binds_distinct_parents_to_the_best_cell_anchors() -> None:
    spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("first", "min", 0.0, 1.0),
            AffineObjectiveAxis("second", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    objective_rows = (
        {"first": 0.11, "second": 0.06},
        {"first": 0.07, "second": 0.26},
        {"first": 0.05, "second": 0.48},
    )
    snapshot = AffineHypervolumeArchiveUtility(spec).freeze(
        benchmark=freeze_json({"contract": "residual-parent-binding"}),
        generation=3,
        archive=_archive(objective_rows),
    )
    candidates = tuple(
        _named_parent(str(index + 1), objectives)
        for index, objectives in enumerate(objective_rows)
    )

    selected = residual_anchor_parents(
        geometry=residual_frontier_geometry(snapshot),
        candidates=candidates,
    )

    assert selected is not None
    assert {value.candidate_id for value in selected} == {
        CandidateId("candidate_1"),
        CandidateId("candidate_2"),
    }


def test_residual_parent_and_target_policies_form_one_coherent_cell_decision() -> None:
    specs = (
        ObjectiveSpec("first", "min"),
        ObjectiveSpec("second", "min"),
    )
    affine_spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("first", "min", 0.0, 1.0),
            AffineObjectiveAxis("second", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    objective_rows = (
        {"first": 0.32, "second": 0.58},
        {"first": 0.38, "second": 0.35},
        {"first": 0.45, "second": 0.285},
    )
    candidates = tuple(
        _named_parent(f"joint_{index + 1}", objectives)
        for index, objectives in enumerate(objective_rows)
    )
    archive = ParetoArchive(specs)
    for candidate in candidates:
        archive.consider(candidate)
    archive_snapshot = archive.snapshot()
    state = OptimizerState(
        generation=0,
        candidates=candidates,
        archive=archive_snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(archive_snapshot),
        unique_evaluations=3,
        logical_llm_calls=0,
    )
    utility_snapshot = AffineHypervolumeArchiveUtility(affine_spec).freeze(
        benchmark=freeze_json({"contract": "joint-residual-cell"}),
        generation=1,
        archive=_archive(objective_rows),
    )

    selection = ResidualHypervolumeCampaignParentSelector().select(
        state,
        task_sha256=hashlib.sha256(b"joint-residual-task").hexdigest(),
        parent_count=2,
        rotation_index=0,
        archive_utility=utility_snapshot,
    )
    targets = ResidualHypervolumeFrontierTargetAllocator().allocate(
        archive_utility=utility_snapshot,
        lanes=tuple((lane.lane_id, lane.parent) for lane in selection.lanes),
    )
    payloads = [thaw_json(value.payload) for value in targets]
    objective_targets = tuple(
        objective_space_target_from_campaign_target(value) for value in targets
    )

    assert {value.candidate_id for value in selection.parents} == {
        CandidateId("candidate_joint_1"),
        CandidateId("candidate_joint_2"),
    }
    assert tuple(value.lane_id for value in targets) == ("elite", "explorer")
    assert {value.direction_id for value in targets} == {
        "axis_1_extreme",
        "axis_2_extreme",
    }
    assert (
        len({payload["residual_frontier_cell"]["cell_sha256"] for payload in payloads})
        == 1
    )
    assert all(
        payload["residual_frontier_cell"]["normalized_aspiration_point_decimal"]
        == ["0.34999999999999998", "0.46499999999999997"]
        for payload in payloads
    )
    assert all(payload["schema_version"] == 2 for payload in payloads)
    assert all(value is not None for value in objective_targets)
    assert all(value.metric_ids == ("first", "second") for value in objective_targets)
    assert all(
        value.axes[0].aspiration_normalized == 0.35
        and value.axes[1].aspiration_normalized == 0.46499999999999997
        for value in objective_targets
        if value is not None
    )
    assert all(
        payload["objective_space_target"]["axes"]
        == [
            {
                "metric_id": "first",
                "goal": "min",
                "ideal_decimal": "0",
                "reference_decimal": "1",
                "parent_value_decimal": payload["assigned_parent"][
                    "normalized_point_decimal"
                ][0],
                "aspiration_value_decimal": "0.34999999999999998",
                "signed_parent_to_aspiration_delta_decimal": (
                    payload["lane_transition"]["normalized_signed_delta_decimal"][0]
                ),
                "improving_raw_delta_sign": "negative",
            },
            {
                "metric_id": "second",
                "goal": "min",
                "ideal_decimal": "0",
                "reference_decimal": "1",
                "parent_value_decimal": payload["assigned_parent"][
                    "normalized_point_decimal"
                ][1],
                "aspiration_value_decimal": "0.46499999999999997",
                "signed_parent_to_aspiration_delta_decimal": (
                    payload["lane_transition"]["normalized_signed_delta_decimal"][1]
                ),
                "improving_raw_delta_sign": "negative",
            },
        ]
        for payload in payloads
    )
    assert all(
        payload["acquisition_instruction"]["target_realization_is_magnitude_sensitive"]
        and payload["acquisition_instruction"][
            "direction_only_forecasts_are_insufficient"
        ]
        for payload in payloads
    )
    assert {
        tuple(payload["lane_transition"]["improve_metric_ids"]) for payload in payloads
    } == {("first",), ("second",)}
    serialized = json.dumps([value.to_record() for value in targets], sort_keys=True)
    for forbidden in (
        '"workload_id":',
        '"model_id":',
        '"provider_id":',
        '"action_name":',
    ):
        assert forbidden not in serialized


def test_residual_target_chooses_best_cell_reachable_by_supplied_parents() -> None:
    affine_spec = AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("first", "min", 0.0, 1.0),
            AffineObjectiveAxis("second", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    objective_rows = (
        {"first": 0.32, "second": 0.58},
        {"first": 0.38, "second": 0.35},
        {"first": 0.45, "second": 0.285},
    )
    candidates = tuple(
        _named_parent(f"reachable_{index + 1}", objectives)
        for index, objectives in enumerate(objective_rows)
    )
    utility_snapshot = AffineHypervolumeArchiveUtility(affine_spec).freeze(
        benchmark=freeze_json({"contract": "parent-reachable-residual-cell"}),
        generation=3,
        archive=_archive(objective_rows),
    )

    targets = ResidualHypervolumeFrontierTargetAllocator().allocate(
        archive_utility=utility_snapshot,
        lanes=(("alpha", candidates[1]), ("beta", candidates[2])),
    )
    payloads = tuple(thaw_json(value.payload) for value in targets)

    assert tuple(value.lane_id for value in targets) == ("alpha", "beta")
    assert all(value.opportunity_rank > 1 for value in targets)
    assert all(
        payload["residual_frontier_cell"]["selection_scope"]
        == "supplied_parent_lane_anchors"
        and payload["residual_frontier_cell"]["global_opportunity_rank"]
        == target.opportunity_rank
        and payload["residual_frontier_cell"]["parent_anchor_binding_distance_decimal"]
        == "0"
        for target, payload in zip(targets, payloads, strict=True)
    )


def test_residual_frontier_supports_three_objectives_and_singleton_fallback() -> None:
    spec = AffineHypervolume3DSpec(
        axes=(
            AffineObjectiveAxis("a", "min", 0.0, 1.0),
            AffineObjectiveAxis("b", "min", 0.0, 1.0),
            AffineObjectiveAxis("c", "min", 0.0, 1.0),
        ),
        reference_provenance="fixed normalized test box",
    )
    utility = AffineHypervolumeArchiveUtility3D(spec)
    pair_snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "residual-3d"}),
        generation=1,
        archive=_archive(
            (
                {"a": 0.1, "b": 0.7, "c": 0.4},
                {"a": 0.7, "b": 0.1, "c": 0.4},
            )
        ),
    )
    singleton_snapshot = utility.freeze(
        benchmark=freeze_json({"contract": "residual-3d-singleton"}),
        generation=1,
        archive=_archive(({"a": 0.2, "b": 0.3, "c": 0.4},)),
    )

    pair_geometry = residual_frontier_geometry(pair_snapshot)
    singleton_geometry = residual_frontier_geometry(singleton_snapshot)
    bootstrap_targets = ResidualHypervolumeFrontierTargetAllocator().allocate(
        archive_utility=singleton_snapshot,
        lanes=(
            (
                "elite",
                _named_parent(
                    "bootstrap_front",
                    {"a": 0.2, "b": 0.3, "c": 0.4},
                ),
            ),
            (
                "explorer",
                _named_parent(
                    "bootstrap_dominated",
                    {"a": 0.4, "b": 0.5, "c": 0.6},
                ),
            ),
        ),
    )

    assert len(pair_geometry.axes) == 3
    assert len(pair_geometry.cells) == 1
    assert pair_geometry.cells[0].potential_hypervolume_gain > 0.0
    assert singleton_geometry.cells == ()
    assert len(bootstrap_targets) == 2
    for target in bootstrap_targets:
        payload = thaw_json(target.payload)
        weights = tuple(
            float(value)
            for value in payload["target_direction"]["normalized_weights_decimal"]
        )
        parent = tuple(
            float(value)
            for value in payload["assigned_parent"]["normalized_point_decimal"]
        )
        aspiration = tuple(
            float(value)
            for value in payload["frontier_bootstrap"][
                "normalized_aspiration_point_decimal"
            ]
        )
        expected = tuple(
            0.9 * value if weight > 0.0 else 1.0
            for value, weight in zip(parent, weights, strict=True)
        )
        objective_target = objective_space_target_from_campaign_target(target)

        assert target.allocator_id == DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID
        assert payload["schema_version"] == 2
        assert "residual_frontier_cell" not in payload
        assert payload["frontier_bootstrap"]["target_kind"] == (
            "directional_affine_bootstrap"
        )
        assert aspiration == pytest.approx(expected)
        assert objective_target is not None
        assert tuple(
            axis.aspiration_normalized for axis in objective_target.axes
        ) == pytest.approx(expected)
