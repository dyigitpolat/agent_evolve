"""Focused kill tests for deterministic, strictly disjoint recombination."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import (
    InsertSequenceItem,
    JsonPath,
    ObjectKey,
    PatchLimits,
    operation_effect_bytes,
    operation_effect_sha256,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    POLICY_ID,
    POLICY_VERSION,
    DisjointPatchMaterialization,
    DisjointPatchRecombinationError,
    DisjointPatchRecombiner,
    RecombinationBranch,
    SystemPatchAttribution,
)
from agent_evolve.policies.variation.typed_patch import (
    ComponentTagAssignment,
    ThreeWayRelationKind,
    apply_patch,
    derive_patch,
)


ANCESTOR_ID = CandidateId("candidate_disjoint_ancestor")
LEFT_ID = CandidateId("candidate_disjoint_left")
RIGHT_ID = CandidateId("candidate_disjoint_right")
TARGET_ID = CandidateId("candidate_disjoint_target")


def _materialize(
    ancestor: object,
    left: object,
    right: object,
    **kwargs: object,
) -> DisjointPatchMaterialization:
    arguments = {
        "ancestor": ancestor,
        "ancestor_candidate_id": ANCESTOR_ID,
        "left": left,
        "left_candidate_id": LEFT_ID,
        "right": right,
        "right_candidate_id": RIGHT_ID,
        "target_candidate_id": TARGET_ID,
    }
    arguments.update(kwargs)
    return DisjointPatchRecombiner().materialize(**arguments)  # type: ignore[arg-type]


def _path_key(attribution: SystemPatchAttribution) -> str:
    assert len(attribution.path.segments) == 1
    segment = attribution.path.segments[0]
    assert type(segment) is ObjectKey
    return segment.value


def test_safe_union_is_canonical_replayable_rediff_exact_and_system_attributed() -> None:
    ancestor = {"same": 0, "right": 0, "left": 0, "untouched": [1, 2]}
    left = {"same": 1, "right": 0, "left": 7, "untouched": [1, 2]}
    right = {"same": 1, "right": 9, "left": 0, "untouched": [1, 2]}

    result = _materialize(ancestor, left, right)

    assert type(result.configuration) is FrozenJsonObject
    assert thaw_json(result.configuration) == {
        "left": 7,
        "right": 9,
        "same": 1,
        "untouched": [1, 2],
    }
    assert result.policy_id == POLICY_ID == "disjoint_patch_union"
    assert result.policy_version == POLICY_VERSION == 1
    assert result.union_patch.base_candidate_id == ANCESTOR_ID
    assert result.union_patch.target_candidate_id == TARGET_ID
    assert all(
        operation.source_candidate_id == ANCESTOR_ID
        for operation in result.union_patch.operations
    )
    assert {
        kind: len(result.classification.of_kind(kind))
        for kind in ThreeWayRelationKind
    } == {
        ThreeWayRelationKind.IDENTICAL: 1,
        ThreeWayRelationKind.DISJOINT: 2,
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT: 0,
        ThreeWayRelationKind.CONFLICT: 0,
        ThreeWayRelationKind.INVALIDATED: 0,
    }

    sources = {
        _path_key(attribution): tuple(source.value for source in attribution.sources)
        for attribution in result.system_attribution
    }
    assert sources == {
        "left": ("left",),
        "right": ("right",),
        "same": ("left", "right"),
    }
    relation_by_id = {
        relation.relation_id: relation
        for relation in result.classification.relations
    }
    for attribution in result.system_attribution:
        relation = relation_by_id[attribution.relation_id]
        operation = (
            relation.left_operations[0]
            if relation.left_operations
            else relation.right_operations[0]
        )
        assert attribution.path == operation.path
        assert attribution.effect_sha256 == operation_effect_sha256(operation)

    assert apply_patch(ancestor, result.union_patch) == result.configuration
    rediff = derive_patch(
        ancestor,
        result.configuration,
        base_candidate_id=ANCESTOR_ID,
        target_candidate_id=TARGET_ID,
    )
    assert rediff.patch_hash == result.union_patch.patch_hash
    assert tuple(map(operation_effect_bytes, rediff.operations)) == tuple(
        map(operation_effect_bytes, result.union_patch.operations)
    )
    assert len(result.receipt_sha256) == 64
    int(result.receipt_sha256, 16)
    result.revalidate()


def test_materialization_is_deterministic_and_does_not_alias_mutable_inputs() -> None:
    ancestor = {"z": 0, "a": 0, "m": {"keep": True}}
    left = {"z": 4, "a": 0, "m": {"keep": True}}
    right = {"z": 0, "a": 3, "m": {"keep": True}}
    first = _materialize(ancestor, left, right)
    second = _materialize(
        {"a": 0, "m": {"keep": True}, "z": 0},
        {"a": 0, "m": {"keep": True}, "z": 4},
        {"a": 3, "m": {"keep": True}, "z": 0},
    )

    assert first == second
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.union_patch.patch_hash == second.union_patch.patch_hash
    left["z"] = 99
    right["a"] = 99
    ancestor["m"]["keep"] = False  # type: ignore[index]
    assert thaw_json(first.configuration) == {
        "a": 3,
        "m": {"keep": True},
        "z": 4,
    }
    with pytest.raises(FrozenInstanceError):
        first.configuration = freeze_json({"bad": True})  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.union_patch = second.union_patch  # type: ignore[misc]


@pytest.mark.parametrize(
    "left,right,match",
    [
        ({"a": 0, "b": 0}, {"a": 0, "b": 2}, "left branch"),
        ({"a": 1, "b": 0}, {"a": 1, "b": 0}, "left and right"),
        (
            {"a": 1, "b": 0, "same": 1},
            {"a": 0, "b": 0, "same": 1},
            "right branch",
        ),
        (
            {"a": 0, "b": 0, "same": 1},
            {"a": 0, "b": 2, "same": 1},
            "left branch",
        ),
    ],
)
def test_each_branch_must_contribute_a_non_identical_disjoint_effect(
    left: object,
    right: object,
    match: str,
) -> None:
    ancestor = {key: 0 for key in set(left) | set(right)}  # type: ignore[arg-type]
    with pytest.raises(DisjointPatchRecombinationError, match=match):
        _materialize(ancestor, left, right)


def test_conflicting_effects_fail_closed_without_materializing_a_union() -> None:
    with pytest.raises(DisjointPatchRecombinationError, match="conflict"):
        _materialize(
            {"x": 0, "left": 0, "right": 0},
            {"x": 1, "left": 1, "right": 0},
            {"x": 2, "left": 0, "right": 1},
        )


def test_prefix_invalidated_effects_fail_closed() -> None:
    ancestor = {"tree": {"x": 0, "y": 0}, "left": 0, "right": 0}
    left = {
        "tree": {"x": 0, "y": 0, "new": 1},
        "left": 1,
        "right": 0,
    }
    right = {"tree": {"x": 2, "y": 0}, "left": 0, "right": 1}
    with pytest.raises(DisjointPatchRecombinationError, match="invalidated"):
        _materialize(ancestor, left, right)


def test_same_semantic_component_requires_higher_authority_and_fails_closed() -> None:
    path_a = JsonPath((ObjectKey("a"),))
    path_b = JsonPath((ObjectKey("b"),))
    with pytest.raises(
        DisjointPatchRecombinationError,
        match="compatible_same_component",
    ):
        _materialize(
            {"a": 0, "b": 0},
            {"a": 1, "b": 0},
            {"a": 0, "b": 2},
            left_component_tags=(ComponentTagAssignment(path_a, "control"),),
            right_component_tags=(ComponentTagAssignment(path_b, "control"),),
        )


def test_disjoint_sequence_operations_construct_and_replay_safely() -> None:
    ancestor = {
        "left_sequence": [1, 2],
        "right_sequence": [3, 4],
    }
    left = {
        "left_sequence": [0, 1, 2],
        "right_sequence": [3, 4],
    }
    right = {
        "left_sequence": [1, 2],
        "right_sequence": [4],
    }
    result = _materialize(ancestor, left, right)

    assert thaw_json(result.configuration) == {
        "left_sequence": [0, 1, 2],
        "right_sequence": [4],
    }
    assert any(type(operation) is InsertSequenceItem for operation in result.union_patch.operations)
    assert len(result.union_patch.operations) == 2
    result.revalidate()


def test_union_respects_the_same_bounded_patch_algebra_as_each_branch() -> None:
    policy = DisjointPatchRecombiner(limits=PatchLimits(max_operations=2))
    with pytest.raises(ValueError, match="max_operations"):
        policy.materialize(
            ancestor={"a": 0, "b": 0, "c": 0, "d": 0},
            ancestor_candidate_id=ANCESTOR_ID,
            left={"a": 1, "b": 1, "c": 0, "d": 0},
            left_candidate_id=LEFT_ID,
            right={"a": 0, "b": 0, "c": 1, "d": 1},
            right_candidate_id=RIGHT_ID,
            target_candidate_id=TARGET_ID,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"left_candidate_id": ANCESTOR_ID},
        {"right_candidate_id": LEFT_ID},
        {"target_candidate_id": ANCESTOR_ID},
        {"target_candidate_id": LEFT_ID},
        {"target_candidate_id": RIGHT_ID},
    ],
)
def test_all_occurrence_ids_must_be_pairwise_distinct(overrides: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "ancestor": {"a": 0, "b": 0},
        "ancestor_candidate_id": ANCESTOR_ID,
        "left": {"a": 1, "b": 0},
        "left_candidate_id": LEFT_ID,
        "right": {"a": 0, "b": 1},
        "right_candidate_id": RIGHT_ID,
        "target_candidate_id": TARGET_ID,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match="pairwise distinct"):
        DisjointPatchRecombiner().materialize(**arguments)  # type: ignore[arg-type]


def test_candidate_ids_and_policy_limits_use_exact_domain_types() -> None:
    with pytest.raises(TypeError, match="exact CandidateId"):
        DisjointPatchRecombiner().materialize(
            ancestor={"a": 0, "b": 0},
            ancestor_candidate_id="candidate_not_typed",  # type: ignore[arg-type]
            left={"a": 1, "b": 0},
            left_candidate_id=LEFT_ID,
            right={"a": 0, "b": 1},
            right_candidate_id=RIGHT_ID,
            target_candidate_id=TARGET_ID,
        )
    with pytest.raises(TypeError, match="exact PatchLimits"):
        DisjointPatchRecombiner(limits=object())  # type: ignore[arg-type]


def test_materialization_revalidation_rejects_configuration_and_attribution_tampering() -> None:
    result = _materialize(
        {"a": 0, "b": 0, "same": 0},
        {"a": 1, "b": 0, "same": 1},
        {"a": 0, "b": 2, "same": 1},
    )
    wrong_configuration = object.__new__(DisjointPatchMaterialization)
    object.__setattr__(wrong_configuration, "configuration", freeze_json({"a": 1, "b": 0, "same": 1}))
    object.__setattr__(wrong_configuration, "union_patch", result.union_patch)
    object.__setattr__(wrong_configuration, "classification", result.classification)
    object.__setattr__(wrong_configuration, "system_attribution", result.system_attribution)
    with pytest.raises(ValueError, match="configuration"):
        wrong_configuration.revalidate()

    right_attribution_index = next(
        index
        for index, item in enumerate(result.system_attribution)
        if item.sources == (RecombinationBranch.RIGHT,)
    )
    forged = list(result.system_attribution)
    forged[right_attribution_index] = replace(
        forged[right_attribution_index],
        sources=(RecombinationBranch.LEFT,),
    )
    with pytest.raises(ValueError, match="system_attribution"):
        replace(result, system_attribution=tuple(forged))


@pytest.mark.parametrize(
    "sources,error,match",
    [
        ((), ValueError, "non-empty canonical"),
        (
            (RecombinationBranch.RIGHT, RecombinationBranch.LEFT),
            ValueError,
            "canonical",
        ),
        (("left",), TypeError, "RecombinationBranch"),
    ],
)
def test_system_attribution_accepts_only_canonical_exact_source_sets(
    sources: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        SystemPatchAttribution(
            path=JsonPath((ObjectKey("x"),)),
            sources=sources,  # type: ignore[arg-type]
            relation_id="a" * 64,
            effect_sha256="b" * 64,
        )


def test_target_occurrence_changes_receipt_but_not_union_effects() -> None:
    ancestor = {"a": 0, "b": 0}
    left = {"a": 1, "b": 0}
    right = {"a": 0, "b": 1}
    first = _materialize(ancestor, left, right)
    second = _materialize(
        ancestor,
        left,
        right,
        target_candidate_id=CandidateId("candidate_disjoint_target_two"),
    )
    assert first.receipt_sha256 != second.receipt_sha256
    assert first.union_patch.patch_hash != second.union_patch.patch_hash
    assert tuple(map(operation_effect_bytes, first.union_patch.operations)) == tuple(
        map(operation_effect_bytes, second.union_patch.operations)
    )
    assert first.configuration == second.configuration


def test_policy_module_remains_inward_only_and_generator_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "agent_evolve"
        / "policies"
        / "variation"
        / "disjoint_recombination.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    forbidden = (
        "agent_evolve.application",
        "agent_evolve.ports",
        "agent_evolve.infrastructure",
        "agent_evolve.integrations",
        "pydantic",
        "pydantic_ai",
    )
    assert not any(
        imported == blocked or imported.startswith(f"{blocked}.")
        for imported in imports
        for blocked in forbidden
    ), imports
