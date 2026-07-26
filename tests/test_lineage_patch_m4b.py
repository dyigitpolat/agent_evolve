"""Offline kill tests for the M4b typed lineage/patch slice."""

from __future__ import annotations

import itertools
import ast
from dataclasses import fields, replace
from pathlib import Path

import pytest

import agent_evolve.domain.patch as patch_domain
import agent_evolve.policies.variation.typed_patch as typed_patch_policy

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.lineage import (
    MAX_PRESERVATION_OBLIGATIONS,
    MAX_SELECTED_INSIGHTS,
    CandidateOccurrence,
    ParentEdge,
    ParentRole,
    PreservationClaim,
    PreservationExpectation,
    PreservationObligation,
    PreservationSource,
    VariationCase,
    VariationKind,
    VariationParent,
    validate_variation_case,
)
from agent_evolve.domain.patch import (
    ArrayIndex,
    DeleteSequenceItem,
    InsertSequenceItem,
    JsonPath,
    ObjectKey,
    PatchLimits,
    PermuteSequence,
    ReplaceScalar,
    ReplaceSubtree,
    TypedPatch,
    canonical_path_bytes,
    operation_effect_bytes,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    TypedJsonLimits,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.variation.typed_patch import (
    ComponentTagAssignment,
    ParentConfiguration,
    PatchPostconditionError,
    PatchPreconditionError,
    PatchRelation,
    PatchResolution,
    PreservationObligationRequest,
    PreservationError,
    PreservationVerification,
    ResolutionChoice,
    ThreeWayPatchClassification,
    ThreeWayRelationKind,
    apply_patch,
    bind_parent_configuration,
    classify_three_way_patches,
    derive_patch,
    derive_preservation_obligations,
    invert_patch,
    validate_three_way_resolutions,
    validate_patch_relation,
    validate_preservation_verification,
    verify_preservation_claims,
)


BASE_ID = CandidateId("candidate_m4b_base")
TARGET_ID = CandidateId("candidate_m4b_target")
LEFT_ID = CandidateId("candidate_m4b_left")
RIGHT_ID = CandidateId("candidate_m4b_right")
CHILD_ID = CandidateId("candidate_m4b_child")
CONTEXT_HASH = "a" * 64
REWARD_HASH = "b" * 64


def _derive(base, target, *, base_id=BASE_ID, target_id=TARGET_ID, tags=(), limits=None):
    arguments = {
        "base_candidate_id": base_id,
        "target_candidate_id": target_id,
        "component_tags": tags,
    }
    if limits is not None:
        arguments["limits"] = limits
    return derive_patch(base, target, **arguments)


def _occurrence(candidate_id, value, sequence):
    digest = typed_json_sha256(value)
    return CandidateOccurrence(candidate_id, digest, digest, sequence)


def _case(kind, parents, *, obligations=(), ancestor=None, patches=(), insights=()):
    return VariationCase(
        OperatorInvocationId(f"operator_m4b_{kind.value}"),
        kind,
        f"m4b.{kind.value}",
        1,
        parents,
        2,
        CONTEXT_HASH,
        REWARD_HASH,
        common_ancestor=ancestor,
        ancestor_to_parent_patches=patches,
        selected_insights=insights,
        preservation_obligations=obligations,
    )


def test_m4b_slice_dependencies_remain_inward_and_framework_free():
    source_root = Path(__file__).parents[1] / "src" / "agent_evolve"
    relative_paths = (
        "domain/typed_json.py",
        "domain/patch.py",
        "domain/lineage.py",
        "policies/variation/typed_patch.py",
    )
    forbidden = (
        "agent_evolve.ports",
        "agent_evolve.infrastructure",
        "pydantic",
        "pydantic_ai",
        "requests",
        "httpx",
        "pathlib",
        "os",
        "subprocess",
    )
    for relative_path in relative_paths:
        tree = ast.parse((source_root / relative_path).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert not any(
            imported == blocked or imported.startswith(f"{blocked}.")
            for imported in imports
            for blocked in forbidden
        ), (relative_path, imports)


def test_typed_json_hashes_preserve_exact_scalar_types_and_float_bits():
    values = (True, 1, 1.0, False, 0, 0.0, -0.0)
    encodings = [canonical_typed_json_bytes(value) for value in values]
    assert len(set(encodings)) == len(values)
    assert not typed_json_equal(True, 1)
    assert not typed_json_equal(1, 1.0)
    assert not typed_json_equal(0.0, -0.0)


def test_canonical_object_order_is_independent_of_input_insertion_order():
    left = {"z": [1, {"x": True}], "a": "text"}
    right = {"a": "text", "z": [1, {"x": True}]}
    assert canonical_typed_json_bytes(left) == canonical_typed_json_bytes(right)
    assert typed_json_sha256(left) == typed_json_sha256(right)
    assert thaw_json(freeze_json(left)) == right


class _HostileList(list):
    pass


class _HostileDict(dict):
    pass


class _HostileInt(int):
    pass


@pytest.mark.parametrize(
    "value,match",
    [
        (_HostileList([1]), "exact dict/list"),
        (_HostileDict({"a": 1}), "exact dict/list"),
        (_HostileInt(1), "exact dict/list"),
        ((1, 2), "exact dict/list"),
        ({1: "bad"}, "object key"),
        ({"x": float("nan")}, "finite"),
        ({"x": float("inf")}, "finite"),
        ({"\ud800": 1}, "UTF-8"),
        ({"x": "\udfff"}, "UTF-8"),
    ],
)
def test_freeze_rejects_hostile_or_non_json_values(value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        freeze_json(value)


def test_freeze_rejects_direct_and_indirect_cycles():
    direct = []
    direct.append(direct)
    with pytest.raises(ValueError, match="cycles"):
        freeze_json(direct)
    indirect = {"x": []}
    indirect["x"].append(indirect)
    with pytest.raises(ValueError, match="cycles"):
        freeze_json(indirect)


def test_typed_json_enforces_depth_node_string_integer_and_byte_bounds():
    limits = TypedJsonLimits(
        max_depth=2,
        max_nodes=5,
        max_container_items=3,
        max_string_bytes=4,
        max_integer_bits=8,
        max_canonical_bytes=32,
    )
    with pytest.raises(ValueError, match="max_depth"):
        freeze_json([[[[0]]]], limits=limits)
    with pytest.raises(ValueError, match="max_nodes"):
        freeze_json([0, 1, 2], limits=replace(limits, max_nodes=3))
    with pytest.raises(ValueError, match="max_container_items"):
        freeze_json([0, 1, 2, 3], limits=limits)
    with pytest.raises(ValueError, match="max_string_bytes"):
        freeze_json("12345", limits=limits)
    with pytest.raises(ValueError, match="max_integer_bits"):
        freeze_json(256, limits=limits)
    with pytest.raises(ValueError, match="max_canonical_bytes"):
        canonical_typed_json_bytes("1234", limits=replace(limits, max_canonical_bytes=4))


def test_paths_distinguish_object_keys_from_array_indices_and_reject_malformed_values():
    key_path = JsonPath((ObjectKey("0"),))
    index_path = JsonPath((ArrayIndex(0),))
    assert canonical_path_bytes(key_path) != canonical_path_bytes(index_path)
    assert key_path.schema_identity != index_path.schema_identity
    with pytest.raises(TypeError, match="exact tuple"):
        JsonPath([ObjectKey("x")])
    with pytest.raises(TypeError, match="exact integers"):
        ArrayIndex(True)
    with pytest.raises(ValueError, match="UTF-8"):
        ObjectKey("\ud800")


@pytest.mark.parametrize(
    "base,target,operation_type",
    [
        ({"x": True}, {"x": 1}, ReplaceScalar),
        ({"x": 1}, {"x": 1.0}, ReplaceScalar),
        ({"x": {"a": 1}}, {"x": {"a": 1, "b": 2}}, ReplaceSubtree),
        ({"x": [1, 2]}, {"x": [0, 1, 2]}, InsertSequenceItem),
        ({"x": [1, 2]}, {"x": [2]}, DeleteSequenceItem),
        ({"x": [1, 2, 2]}, {"x": [2, 1, 2]}, PermuteSequence),
    ],
)
def test_derivation_selects_the_typed_operation_and_replays_exactly(
    base, target, operation_type
):
    patch = _derive(base, target)
    assert len(patch.operations) == 1
    assert type(patch.operations[0]) is operation_type
    assert thaw_json(apply_patch(base, patch)) == target
    assert typed_json_equal(apply_patch(base, patch), freeze_json(target))


def test_duplicate_insert_delete_and_permutation_use_lowest_source_index_law():
    insert = _derive([1], [1, 1])
    assert type(insert.operations[0]) is InsertSequenceItem
    assert insert.operations[0].index == 0
    delete = _derive([1, 1], [1])
    assert type(delete.operations[0]) is DeleteSequenceItem
    assert delete.operations[0].index == 0
    permutation = _derive([1, 1, 2], [1, 2, 1])
    assert type(permutation.operations[0]) is PermuteSequence
    assert permutation.operations[0].permutation == (0, 2, 1)

    before = freeze_json([1, 1, 2])
    after = freeze_json([1, 2, 1])
    with pytest.raises(ValueError, match="lowest-source-index"):
        PermuteSequence(
            JsonPath(),
            (1, 2, 0),
            before,
            after,
            BASE_ID,
        )


def test_exhaustive_duplicate_binary_sequences_obey_lowest_edit_index():
    for size in range(6):
        for sequence in itertools.product((0, 1), repeat=size):
            for requested_index in range(size + 1):
                for item in (0, 1):
                    target = (
                        sequence[:requested_index]
                        + (item,)
                        + sequence[requested_index:]
                    )
                    patch = _derive(list(sequence), list(target))
                    operation = patch.operations[0]
                    assert type(operation) is InsertSequenceItem
                    valid_indices = [
                        index
                        for index in range(size + 1)
                        if sequence[:index]
                        + (target[index],)
                        + sequence[index:]
                        == target
                    ]
                    assert operation.index == min(valid_indices)
            if size:
                for requested_index in range(size):
                    target = (
                        sequence[:requested_index]
                        + sequence[requested_index + 1 :]
                    )
                    patch = _derive(list(sequence), list(target))
                    operation = patch.operations[0]
                    assert type(operation) is DeleteSequenceItem
                    valid_indices = [
                        index
                        for index in range(size)
                        if sequence[:index] + sequence[index + 1 :] == target
                    ]
                    assert operation.index == min(valid_indices)


def test_object_key_set_change_is_explicit_subtree_replacement_not_insertion():
    patch = _derive({"outer": {"a": 1}}, {"outer": {"a": 1, "b": 2}})
    operation = patch.operations[0]
    assert type(operation) is ReplaceSubtree
    assert operation.path == JsonPath((ObjectKey("outer"),))
    assert thaw_json(apply_patch({"outer": {"a": 1}}, patch)) == {
        "outer": {"a": 1, "b": 2}
    }


def test_nested_disjoint_derivation_is_canonical_across_object_order():
    base_a = {"z": [0, {"b": False}], "a": {"q": 0}}
    target_a = {"z": [2, {"b": True}], "a": {"q": 1}}
    base_b = {"a": {"q": 0}, "z": [0, {"b": False}]}
    target_b = {"a": {"q": 1}, "z": [2, {"b": True}]}
    first = _derive(base_a, target_a)
    second = _derive(base_b, target_b)
    assert first.patch_hash == second.patch_hash
    paths = [canonical_path_bytes(operation.path) for operation in first.operations]
    assert paths == sorted(paths)
    assert thaw_json(apply_patch(base_b, second)) == target_b


def _small_tree_corpus():
    scalars = (None, False, True, -1, 0, 1, -0.0, 0.0, 1.5, "", "x")
    values = list(scalars)
    values.extend([[value] for value in scalars[:6]])
    values.extend({"x": value} for value in scalars[:6])
    values.extend(
        [
            [1, 1, 2],
            [1, 2, 1],
            {"a": [1, {"b": False}], "z": 0.0},
            {"a": [2, {"b": True}], "z": -0.0},
        ]
    )
    return values


def test_exhaustive_small_tree_inverse_round_trips_and_is_deterministic():
    corpus = _small_tree_corpus()
    for index, (base, target) in enumerate(itertools.product(corpus, repeat=2)):
        base_id = CandidateId(f"candidate_property_base_{index}")
        target_id = CandidateId(f"candidate_property_target_{index}")
        first = _derive(base, target, base_id=base_id, target_id=target_id)
        second = _derive(base, target, base_id=base_id, target_id=target_id)
        assert first == second
        result = apply_patch(base, first)
        assert typed_json_equal(result, freeze_json(target))
        restored = apply_patch(result, invert_patch(first))
        assert typed_json_equal(restored, freeze_json(base))


def test_empty_reproduction_patch_retains_distinct_occurrence_endpoints():
    value = {"x": 1}
    patch = _derive(value, value)
    assert not patch.operations
    assert patch.base_candidate_id != patch.target_candidate_id
    assert patch.base_hash == patch.target_hash
    assert typed_json_equal(apply_patch(value, patch), value)


def test_apply_rejects_stale_global_and_operation_local_preconditions():
    valid = _derive({"x": 1}, {"x": 2})
    with pytest.raises(PatchPreconditionError, match="base hash"):
        apply_patch({"x": 0}, valid)

    base = freeze_json({"x": 1})
    fabricated_operation = ReplaceScalar(
        JsonPath((ObjectKey("x"),)),
        9,
        2,
        BASE_ID,
    )
    fabricated = TypedPatch(
        BASE_ID,
        TARGET_ID,
        typed_json_sha256(base),
        typed_json_sha256({"x": 2}),
        (fabricated_operation,),
    )
    with pytest.raises(PatchPreconditionError, match="old-value"):
        apply_patch(base, fabricated)


def test_apply_rejects_missing_object_key_and_fabricated_target_postcondition():
    missing = TypedPatch(
        BASE_ID,
        TARGET_ID,
        typed_json_sha256({"x": 1}),
        typed_json_sha256({"missing": 2, "x": 1}),
        (
            ReplaceScalar(
                JsonPath((ObjectKey("missing"),)),
                0,
                2,
                BASE_ID,
            ),
        ),
    )
    with pytest.raises(PatchPreconditionError, match="does not exist"):
        apply_patch({"x": 1}, missing)

    operation = ReplaceScalar(JsonPath((ObjectKey("x"),)), 1, 2, BASE_ID)
    wrong_target = TypedPatch(
        BASE_ID,
        TARGET_ID,
        typed_json_sha256({"x": 1}),
        "f" * 64,
        (operation,),
    )
    with pytest.raises(PatchPostconditionError, match="target_hash"):
        apply_patch({"x": 1}, wrong_target)


def test_patch_rejects_noncanonical_overlap_wrong_source_and_noop_endpoint_claims():
    root = ReplaceSubtree(JsonPath(), freeze_json({"x": 1}), freeze_json({"x": 2}), BASE_ID)
    child = ReplaceScalar(JsonPath((ObjectKey("x"),)), 1, 2, BASE_ID)
    with pytest.raises(ValueError, match="overlap"):
        TypedPatch(BASE_ID, TARGET_ID, "a" * 64, "b" * 64, (root, child))

    z = ReplaceScalar(JsonPath((ObjectKey("z"),)), 0, 1, BASE_ID)
    a = ReplaceScalar(JsonPath((ObjectKey("a"),)), 0, 1, BASE_ID)
    with pytest.raises(ValueError, match="canonical"):
        TypedPatch(BASE_ID, TARGET_ID, "a" * 64, "b" * 64, (z, a))

    wrong_source = ReplaceScalar(JsonPath(), 0, 1, LEFT_ID)
    with pytest.raises(ValueError, match="base occurrence"):
        TypedPatch(BASE_ID, TARGET_ID, "a" * 64, "b" * 64, (wrong_source,))
    with pytest.raises(ValueError, match="non-empty"):
        TypedPatch(BASE_ID, TARGET_ID, "a" * 64, "a" * 64, (child,))
    with pytest.raises(ValueError, match="empty patch"):
        TypedPatch(BASE_ID, TARGET_ID, "a" * 64, "b" * 64, ())
    with pytest.raises(ValueError, match="distinct candidate occurrences"):
        TypedPatch(BASE_ID, BASE_ID, "a" * 64, "a" * 64, ())


def test_patch_resource_bounds_stop_deep_paths_and_operation_explosion():
    json_limits = TypedJsonLimits(max_depth=8)
    path_limits = PatchLimits(
        json_limits=json_limits,
        max_operations=10,
        max_path_segments=1,
    )
    patch = _derive(
        {"a": {"b": {"c": 1}}},
        {"a": {"b": {"c": 2}}},
        limits=path_limits,
    )
    assert len(patch.operations[0].path.segments) == 1
    assert type(patch.operations[0]) is ReplaceSubtree

    operation_limits = PatchLimits(
        json_limits=json_limits,
        max_operations=1,
        max_path_segments=8,
    )
    with pytest.raises(ValueError, match="max_operations"):
        _derive({"a": 0, "b": 0}, {"a": 1, "b": 1}, limits=operation_limits)
    byte_limits = PatchLimits(
        json_limits=json_limits,
        max_operations=2,
        max_path_segments=8,
        max_patch_bytes=1,
    )
    with pytest.raises(ValueError, match="max_patch_bytes"):
        _derive({"a": 0}, {"a": 1}, limits=byte_limits)


def test_nested_limits_and_paths_are_revalidated_by_consumers():
    malformed_json_limits = object.__new__(TypedJsonLimits)
    for name, value in {
        "max_depth": 1_000_000,
        "max_nodes": 50_000,
        "max_container_items": 10_000,
        "max_string_bytes": 1_048_576,
        "max_integer_bits": 4096,
        "max_canonical_bytes": 8_388_608,
    }.items():
        object.__setattr__(malformed_json_limits, name, value)
    with pytest.raises(ValueError, match="max_depth"):
        PatchLimits(json_limits=malformed_json_limits)

    malformed_path = object.__new__(JsonPath)
    object.__setattr__(malformed_path, "segments", [ObjectKey("x")])
    with pytest.raises(TypeError, match="exact tuple"):
        ReplaceScalar(malformed_path, 0, 1, BASE_ID)


def test_patch_byte_limit_binds_fixed_empty_and_single_operation_preimages_exactly():
    schema = "typed_json_patch_v1"
    fixed = (
        8
        + len(schema)
        + 8
        + len(BASE_ID.value)
        + 8
        + len(TARGET_ID.value)
        + 64
        + 9 * 8
        + 8
    )
    same_hash = typed_json_sha256(0)
    with pytest.raises(ValueError, match="max_patch_bytes"):
        TypedPatch(
            BASE_ID,
            TARGET_ID,
            same_hash,
            same_hash,
            (),
            limits=PatchLimits(max_patch_bytes=fixed - 1),
        )
    empty = TypedPatch(
        BASE_ID,
        TARGET_ID,
        same_hash,
        same_hash,
        (),
        limits=PatchLimits(max_patch_bytes=fixed),
    )
    assert empty.patch_hash

    operation = ReplaceScalar(JsonPath(), 0, 1, BASE_ID)
    exact_single = fixed + 8 + len(operation_effect_bytes(operation))
    with pytest.raises(ValueError, match="max_patch_bytes"):
        TypedPatch(
            BASE_ID,
            TARGET_ID,
            typed_json_sha256(0),
            typed_json_sha256(1),
            (operation,),
            limits=PatchLimits(max_patch_bytes=exact_single - 1),
        )
    assert TypedPatch(
        BASE_ID,
        TARGET_ID,
        typed_json_sha256(0),
        typed_json_sha256(1),
        (operation,),
        limits=PatchLimits(max_patch_bytes=exact_single),
    ).patch_hash


def test_long_admitted_object_keys_bubble_to_nearest_representable_parent():
    boundary_key = "k" * 4096
    exact = _derive({boundary_key: 0}, {boundary_key: 1})
    assert exact.operations[0].path == JsonPath((ObjectKey(boundary_key),))

    long_key = "k" * 4097
    root = _derive({long_key: 0}, {long_key: 1})
    assert type(root.operations[0]) is ReplaceSubtree
    assert root.operations[0].path == JsonPath()
    assert typed_json_equal(apply_patch({long_key: 0}, root), {long_key: 1})

    limits = PatchLimits(
        json_limits=TypedJsonLimits(max_string_bytes=5000),
        max_path_segments=4,
    )
    nested_base = {"outer": {long_key: 0}, "stable": {long_key: 7}, "short": 0}
    nested_target = {"outer": {long_key: 1}, "stable": {long_key: 7}, "short": 1}
    nested = _derive(nested_base, nested_target, limits=limits)
    assert [operation.path for operation in nested.operations] == [
        JsonPath((ObjectKey("outer"),)),
        JsonPath((ObjectKey("short"),)),
    ]
    assert typed_json_equal(apply_patch(nested_base, nested), nested_target)

    tighter_path = replace(limits, max_path_segments=1)
    collapsed = _derive(nested_base, nested_target, limits=tighter_path)
    assert typed_json_equal(apply_patch(nested_base, collapsed), nested_target)


def test_component_assignments_are_exact_canonical_and_cannot_be_fabricated():
    path_a = JsonPath((ObjectKey("a"),))
    path_b = JsonPath((ObjectKey("b"),))
    ordered = tuple(
        sorted(
            (
                ComponentTagAssignment(path_b, "component"),
                ComponentTagAssignment(path_a, "component"),
            ),
            key=lambda value: canonical_path_bytes(value.path),
        )
    )
    patch = _derive({"a": 0, "b": 0}, {"a": 1, "b": 1}, tags=ordered)
    assert {operation.semantic_component for operation in patch.operations} == {
        "component"
    }
    with pytest.raises(ValueError, match="canonical"):
        _derive(
            {"a": 0, "b": 0},
            {"a": 1, "b": 1},
            tags=tuple(reversed(ordered)),
        )
    with pytest.raises(ValueError, match="not a derived operation"):
        _derive(
            {"a": 0},
            {"a": 1},
            tags=(ComponentTagAssignment(path_b, "fabricated"),),
        )
    with pytest.raises(ValueError, match="max_operations"):
        _derive(
            {"a": 0, "b": 0},
            {"a": 1, "b": 1},
            tags=(
                ComponentTagAssignment(path_a, "a"),
                ComponentTagAssignment(path_b, "b"),
            ),
            limits=PatchLimits(max_operations=1),
        )


def test_relation_component_rejects_hostile_string_like_values_at_every_consumer():
    ancestor, _, _, left_patch, right_patch = _three_way_fixture()
    classification = classify_three_way_patches(ancestor, left_patch, right_patch)
    canonical = classification.of_kind(
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT
    )[0]

    class _HostileComponent:
        equality_calls = 0
        encode_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return other == "coupled_control"

        def __ne__(self, other):
            type(self).equality_calls += 1
            return False

        def __hash__(self):
            return hash("different-hash")

        def encode(self, *args, **kwargs):
            type(self).encode_calls += 1
            return b"attacker-selected-relation-preimage"

    hostile = _HostileComponent()
    with pytest.raises(TypeError, match="exact string"):
        PatchRelation(
            canonical.kind,
            canonical.left_operations,
            canonical.right_operations,
            hostile,  # type: ignore[arg-type]
        )
    assert _HostileComponent.equality_calls == 0
    assert _HostileComponent.encode_calls == 0

    malformed = object.__new__(PatchRelation)
    object.__setattr__(malformed, "kind", canonical.kind)
    object.__setattr__(malformed, "left_operations", canonical.left_operations)
    object.__setattr__(malformed, "right_operations", canonical.right_operations)
    object.__setattr__(malformed, "semantic_component", hostile)
    with pytest.raises(TypeError, match="exact string"):
        validate_patch_relation(malformed)
    assert _HostileComponent.equality_calls == 0
    assert _HostileComponent.encode_calls == 0

    relations = tuple(
        malformed if relation is canonical else relation
        for relation in classification.relations
    )
    object.__setattr__(classification, "relations", relations)
    for consumer in (
        classification.revalidate,
        lambda: classification.of_kind(
            ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT
        ),
        lambda: validate_three_way_resolutions(classification, ()),
        lambda: derive_preservation_obligations(classification, ()),
    ):
        with pytest.raises(TypeError, match="exact string"):
            consumer()
    assert _HostileComponent.equality_calls == 0
    assert _HostileComponent.encode_calls == 0


def test_relation_component_uses_exact_nonempty_bounded_utf8():
    ancestor, _, _, left_patch, right_patch = _three_way_fixture()
    classification = classify_three_way_patches(ancestor, left_patch, right_patch)
    canonical = classification.of_kind(
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT
    )[0]

    class _StringSubclass(str):
        pass

    invalid = (
        (_StringSubclass("coupled_control"), TypeError, "exact string"),
        ("", ValueError, "cannot be empty"),
        ("\ud800", ValueError, "UTF-8"),
        ("é" * 129, ValueError, "byte limit"),
    )
    for component, error, message in invalid:
        with pytest.raises(error, match=message):
            PatchRelation(
                canonical.kind,
                canonical.left_operations,
                canonical.right_operations,
                component,
            )


def _three_way_fixture():
    ancestor = {
        "same": 0,
        "left_only": 0,
        "right_only": 0,
        "component": {"a": 0, "b": 0},
        "conflict": 0,
        "tree": {"x": 0, "y": 0},
    }
    left = {
        "same": 1,
        "left_only": 1,
        "right_only": 0,
        "component": {"a": 1, "b": 0},
        "conflict": 1,
        "tree": {"x": 0, "y": 0, "z": 1},
    }
    right = {
        "same": 1,
        "left_only": 0,
        "right_only": 2,
        "component": {"a": 0, "b": 2},
        "conflict": 2,
        "tree": {"x": 2, "y": 0},
    }
    path_a = JsonPath((ObjectKey("component"), ObjectKey("a")))
    path_b = JsonPath((ObjectKey("component"), ObjectKey("b")))
    left_patch = _derive(
        ancestor,
        left,
        base_id=BASE_ID,
        target_id=LEFT_ID,
        tags=(ComponentTagAssignment(path_a, "coupled_control"),),
    )
    right_patch = _derive(
        ancestor,
        right,
        base_id=BASE_ID,
        target_id=RIGHT_ID,
        tags=(ComponentTagAssignment(path_b, "coupled_control"),),
    )
    return ancestor, left, right, left_patch, right_patch


def test_three_way_classification_partitions_every_required_relation_kind():
    ancestor, _, _, left_patch, right_patch = _three_way_fixture()
    classification = classify_three_way_patches(ancestor, left_patch, right_patch)
    counts = {
        kind: len(classification.of_kind(kind)) for kind in ThreeWayRelationKind
    }
    assert counts == {
        ThreeWayRelationKind.IDENTICAL: 1,
        ThreeWayRelationKind.DISJOINT: 2,
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT: 1,
        ThreeWayRelationKind.CONFLICT: 1,
        ThreeWayRelationKind.INVALIDATED: 1,
    }
    covered_left = sum(len(relation.left_operations) for relation in classification.relations)
    covered_right = sum(
        len(relation.right_operations) for relation in classification.relations
    )
    assert covered_left == len(left_patch.operations)
    assert covered_right == len(right_patch.operations)


def test_three_way_rejects_different_ancestors_targets_or_algebra_limits():
    base = {"x": 0}
    left = _derive(base, {"x": 1}, target_id=LEFT_ID)
    different_ancestor = _derive(
        base,
        {"x": 2},
        base_id=CandidateId("candidate_other_ancestor"),
        target_id=RIGHT_ID,
    )
    with pytest.raises(ValueError, match="exact ancestor"):
        classify_three_way_patches(base, left, different_ancestor)
    same_target = _derive(base, {"x": 2}, target_id=LEFT_ID)
    with pytest.raises(ValueError, match="distinct occurrences"):
        classify_three_way_patches(base, left, same_target)
    tighter = _derive(
        base,
        {"x": 2},
        target_id=RIGHT_ID,
        limits=PatchLimits(max_operations=4),
    )
    with pytest.raises(ValueError, match="exact algebra limits"):
        classify_three_way_patches(base, left, tighter)
    valid_right = _derive(base, {"x": 2}, target_id=RIGHT_ID)
    fabricated_target = replace(valid_right, target_hash="f" * 64)
    with pytest.raises(PatchPostconditionError, match="target_hash"):
        classify_three_way_patches(base, left, fabricated_target)


def test_direct_classification_cannot_relabel_relations_or_bypass_replay():
    ancestor = {"x": 0}
    left = _derive(ancestor, {"x": 1}, target_id=LEFT_ID)
    right = _derive(ancestor, {"x": 2}, target_id=RIGHT_ID)
    valid = classify_three_way_patches(ancestor, left, right)
    assert [relation.kind for relation in valid.relations] == [
        ThreeWayRelationKind.CONFLICT
    ]
    forged_relations = (
        PatchRelation(ThreeWayRelationKind.DISJOINT, (left.operations[0],), ()),
        PatchRelation(ThreeWayRelationKind.DISJOINT, (), (right.operations[0],)),
    )
    forged_relations = tuple(
        sorted(
            forged_relations,
            key=lambda relation: (
                min(
                    canonical_path_bytes(operation.path)
                    for operation in relation.left_operations
                    + relation.right_operations
                ),
                relation.kind.value,
                relation.relation_id,
            ),
        )
    )
    with pytest.raises(ValueError, match="canonical global classification"):
        ThreeWayPatchClassification(
            ancestor=freeze_json(ancestor),
            ancestor_candidate_id=BASE_ID,
            ancestor_hash=left.base_hash,
            left_patch_hash=left.patch_hash,
            right_patch_hash=right.patch_hash,
            relations=forged_relations,
            left_patch=left,
            right_patch=right,
        )

    mixed_ancestor, _, _, mixed_left, mixed_right = _three_way_fixture()
    mixed = classify_three_way_patches(mixed_ancestor, mixed_left, mixed_right)
    all_disjoint = []
    for operation in mixed_left.operations:
        all_disjoint.append(
            PatchRelation(ThreeWayRelationKind.DISJOINT, (operation,), ())
        )
    for operation in mixed_right.operations:
        all_disjoint.append(
            PatchRelation(ThreeWayRelationKind.DISJOINT, (), (operation,))
        )
    all_disjoint = tuple(
        sorted(
            all_disjoint,
            key=lambda relation: (
                min(
                    canonical_path_bytes(operation.path)
                    for operation in relation.left_operations
                    + relation.right_operations
                ),
                relation.kind.value,
                relation.relation_id,
            ),
        )
    )
    assert {
        relation.kind for relation in mixed.relations
    } >= {
        ThreeWayRelationKind.IDENTICAL,
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
        ThreeWayRelationKind.CONFLICT,
        ThreeWayRelationKind.INVALIDATED,
    }
    with pytest.raises(ValueError, match="canonical global classification"):
        replace(mixed, relations=all_disjoint)

    fabricated_target = replace(right, target_hash="f" * 64)
    with pytest.raises(PatchPostconditionError, match="target_hash"):
        ThreeWayPatchClassification(
            ancestor=freeze_json(ancestor),
            ancestor_candidate_id=BASE_ID,
            ancestor_hash=left.base_hash,
            left_patch_hash=left.patch_hash,
            right_patch_hash=fabricated_target.patch_hash,
            relations=valid.relations,
            left_patch=left,
            right_patch=fabricated_target,
        )


def test_consumers_reject_wrong_source_relation_and_non_enum_kind():
    ancestor = {"x": 0}
    left = _derive(ancestor, {"x": 1}, target_id=LEFT_ID)
    right = _derive(ancestor, {"x": 2}, target_id=RIGHT_ID)
    classification = classify_three_way_patches(ancestor, left, right)
    wrong_source = CandidateId("candidate_m4b_wrong_relation_source")
    left_operation = replace(
        left.operations[0], source_candidate_id=wrong_source
    )
    right_operation = replace(
        right.operations[0], source_candidate_id=wrong_source
    )
    wrong_relation = PatchRelation(
        ThreeWayRelationKind.CONFLICT,
        (left_operation,),
        (right_operation,),
    )
    object.__setattr__(classification, "relations", (wrong_relation,))
    with pytest.raises(ValueError, match="common ancestor|canonical global|partition"):
        classification.revalidate()

    classification = classify_three_way_patches(ancestor, left, right)

    class _KindLikeConflict:
        value = "conflict"

    malformed_relation = object.__new__(PatchRelation)
    object.__setattr__(malformed_relation, "kind", _KindLikeConflict())
    object.__setattr__(
        malformed_relation, "left_operations", (left.operations[0],)
    )
    object.__setattr__(
        malformed_relation, "right_operations", (right.operations[0],)
    )
    object.__setattr__(malformed_relation, "semantic_component", None)
    object.__setattr__(classification, "relations", (malformed_relation,))
    with pytest.raises(TypeError, match="ThreeWayRelationKind"):
        classification.revalidate()
    with pytest.raises(TypeError, match="ThreeWayRelationKind"):
        validate_three_way_resolutions(classification, ())


def test_consumers_reject_sequential_same_path_and_mistyped_operations():
    ancestor = {"x": 0, "y": 0}
    target = {"x": 2, "y": 0}
    path = JsonPath((ObjectKey("x"),))
    sequential = (
        ReplaceScalar(path, 0, 1, BASE_ID),
        ReplaceScalar(path, 1, 2, BASE_ID),
    )
    malformed_patch = object.__new__(TypedPatch)
    for name, value in {
        "base_candidate_id": BASE_ID,
        "target_candidate_id": LEFT_ID,
        "base_hash": typed_json_sha256(ancestor),
        "target_hash": typed_json_sha256(target),
        "operations": sequential,
        "limits": patch_domain.DEFAULT_PATCH_LIMITS,
        "schema_version": "typed_json_patch_v1",
    }.items():
        object.__setattr__(malformed_patch, name, value)
    right = _derive(ancestor, {"x": 0, "y": 1}, target_id=RIGHT_ID)
    with pytest.raises(ValueError, match="overlap"):
        classify_three_way_patches(ancestor, malformed_patch, right)
    with pytest.raises(ValueError, match="overlap"):
        apply_patch(ancestor, malformed_patch)

    malformed_scalar = object.__new__(ReplaceScalar)
    for name, value in {
        "path": JsonPath(),
        "old_value": freeze_json({"x": 0}),
        "new_value": freeze_json({"x": 1}),
        "source_candidate_id": BASE_ID,
        "semantic_component": None,
    }.items():
        object.__setattr__(malformed_scalar, name, value)
    with pytest.raises(TypeError, match="scalar values"):
        TypedPatch(
            BASE_ID,
            TARGET_ID,
            typed_json_sha256({"x": 0}),
            typed_json_sha256({"x": 1}),
            (malformed_scalar,),
        )


def test_classification_rejects_missing_duplicate_and_changed_effect_coverage():
    ancestor = {"x": 0}
    left = _derive(ancestor, {"x": 1}, target_id=LEFT_ID)
    right = _derive(ancestor, {"x": 2}, target_id=RIGHT_ID)
    valid = classify_three_way_patches(ancestor, left, right)
    with pytest.raises(ValueError, match="partition"):
        replace(valid, relations=())
    changed = replace(left.operations[0], new_value=3)
    changed_relation = PatchRelation(
        ThreeWayRelationKind.CONFLICT,
        (changed,),
        (right.operations[0],),
    )
    with pytest.raises(ValueError, match="partition"):
        replace(valid, relations=(changed_relation,))
    duplicate = (
        PatchRelation(ThreeWayRelationKind.DISJOINT, (left.operations[0],), ()),
        PatchRelation(ThreeWayRelationKind.DISJOINT, (left.operations[0],), ()),
    )
    duplicate = tuple(
        sorted(
            duplicate,
            key=lambda relation: (
                min(
                    canonical_path_bytes(operation.path)
                    for operation in relation.left_operations
                    + relation.right_operations
                ),
                relation.kind.value,
                relation.relation_id,
            ),
        )
    )
    with pytest.raises(ValueError, match="duplicate relation identities"):
        replace(valid, relations=duplicate)


def test_conflict_and_invalidation_resolutions_are_complete_unique_and_closed():
    ancestor, _, _, left_patch, right_patch = _three_way_fixture()
    classification = classify_three_way_patches(ancestor, left_patch, right_patch)
    required = sorted(
        relation.relation_id
        for relation in classification.relations
        if relation.kind
        in (ThreeWayRelationKind.CONFLICT, ThreeWayRelationKind.INVALIDATED)
    )
    valid = tuple(
        PatchResolution(relation_id, ResolutionChoice.CHOOSE_LEFT)
        for relation_id in required
    )
    assert validate_three_way_resolutions(classification, valid) == valid
    with pytest.raises(ValueError, match="missing"):
        validate_three_way_resolutions(classification, valid[:-1])
    duplicate = tuple(
        sorted(valid + (valid[0],), key=lambda resolution: resolution.relation_id)
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_three_way_resolutions(classification, duplicate)
    unknown = tuple(
        sorted(
            valid + (PatchResolution("f" * 64, ResolutionChoice.DROP_BOTH),),
            key=lambda resolution: resolution.relation_id,
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_three_way_resolutions(classification, unknown)
    with pytest.raises(ValueError, match="result hash"):
        PatchResolution(required[0], ResolutionChoice.SYNTHESIZE)

    malformed = object.__new__(PatchResolution)
    object.__setattr__(malformed, "relation_id", required[0])
    object.__setattr__(malformed, "choice", "erase_conflict")
    object.__setattr__(malformed, "synthesized_result_hash", None)
    remaining = tuple(value for value in valid if value.relation_id != required[0])
    malformed_set = tuple(
        sorted((malformed,) + remaining, key=lambda value: value.relation_id)
    )
    with pytest.raises(TypeError, match="ResolutionChoice"):
        validate_three_way_resolutions(classification, malformed_set)


def test_trie_overlap_indexes_have_linear_deterministic_work_counts():
    count = 512
    left = tuple(
        ReplaceScalar(
            JsonPath((ObjectKey(f"left_{index:04d}"),)),
            0,
            1,
            BASE_ID,
        )
        for index in range(count)
    )
    right = tuple(
        ReplaceScalar(
            JsonPath((ObjectKey(f"right_{index:04d}"),)),
            0,
            1,
            BASE_ID,
        )
        for index in range(count)
    )
    patch_work = [0]
    patch_domain._validate_non_overlapping_operation_paths(
        left, work_counter=patch_work
    )
    assert patch_work[0] == count
    indexed = typed_patch_policy._index_cross_path_relations(left, right)
    assert indexed.equal_pairs == ()
    assert indexed.strict_edges == ()
    assert indexed.work_units == 2 * count

    root_left = (ReplaceSubtree(JsonPath(), freeze_json({}), freeze_json({"x": 1}), BASE_ID),)
    descendants = tuple(
        ReplaceScalar(
            JsonPath((ObjectKey(f"child_{index:04d}"),)),
            0,
            1,
            BASE_ID,
        )
        for index in range(count)
    )
    star = typed_patch_policy._index_cross_path_relations(root_left, descendants)
    assert len(star.strict_edges) == count
    assert star.work_units <= 2 * count



def test_candidate_occurrences_keep_duplicate_content_distinct_and_edges_bind_patches():
    value = {"x": 1}
    parent = _occurrence(BASE_ID, value, 1)
    child = _occurrence(CHILD_ID, value, 2)
    patch = _derive(value, value, base_id=BASE_ID, target_id=CHILD_ID)
    edge = ParentEdge(ParentRole.REPRODUCTION_SOURCE, parent, child, patch)
    assert edge.parent.candidate_id != edge.child.candidate_id
    assert edge.parent.configuration_hash == edge.child.configuration_hash
    with pytest.raises(ValueError, match="endpoints"):
        ParentEdge(
            ParentRole.REPRODUCTION_SOURCE,
            parent,
            child,
            replace(patch, target_candidate_id=TARGET_ID),
        )


def test_three_way_variation_case_binds_ordered_roles_ancestor_and_branch_patches():
    ancestor_value, left_value, right_value, left_patch, right_patch = _three_way_fixture()
    ancestor = _occurrence(BASE_ID, ancestor_value, 0)
    left = _occurrence(LEFT_ID, left_value, 1)
    right = _occurrence(RIGHT_ID, right_value, 2)
    case = _case(
        VariationKind.THREE_WAY_RECOMBINATION,
        (
            VariationParent(ParentRole.CROSSOVER_LEFT, left),
            VariationParent(ParentRole.CROSSOVER_RIGHT, right),
        ),
        ancestor=ancestor,
        patches=(left_patch, right_patch),
    )
    assert case.common_ancestor == ancestor
    with pytest.raises(ValueError, match="role order"):
        _case(
            VariationKind.THREE_WAY_RECOMBINATION,
            tuple(reversed(case.parents)),
            ancestor=ancestor,
            patches=(right_patch, left_patch),
        )
    with pytest.raises(ValueError, match="exactly two"):
        _case(
            VariationKind.THREE_WAY_RECOMBINATION,
            case.parents,
            ancestor=ancestor,
            patches=(left_patch,),
        )


def _relation_request(classification, source, path):
    operations_name = (
        "left_operations"
        if source is PreservationSource.LEFT_BRANCH
        else "right_operations"
    )
    if source is PreservationSource.IDENTICAL_NEUTRAL:
        candidates = [
            relation
            for relation in classification.relations
            if relation.kind is ThreeWayRelationKind.IDENTICAL
            and relation.left_operations[0].path.is_prefix_of(path)
        ]
    else:
        candidates = [
            relation
            for relation in classification.relations
            if any(
                operation.path.is_prefix_of(path)
                for operation in getattr(relation, operations_name)
            )
        ]
    assert len(candidates) == 1
    return PreservationObligationRequest(candidates[0].relation_id, source, path)


def _preservation_fixture(*, include_neutral=True, include_absence=True):
    ancestor_value = {
        "a": 0,
        "b": 0,
        "same": 0,
        "left_component": {"keep": 0, "gone": 1},
        "right_component": {"keep": 0, "gone": 2},
    }
    left_value = {
        "a": 0,
        "b": 1,
        "same": 1,
        "left_component": {"keep": 0},
        "right_component": {"keep": 0, "gone": 2},
    }
    right_value = {
        "a": 2,
        "b": 0,
        "same": 1,
        "left_component": {"keep": 0, "gone": 1},
        "right_component": {"keep": 0},
    }
    ancestor = _occurrence(BASE_ID, ancestor_value, 0)
    left = _occurrence(LEFT_ID, left_value, 1)
    right = _occurrence(RIGHT_ID, right_value, 2)
    left_patch = _derive(
        ancestor_value,
        left_value,
        base_id=BASE_ID,
        target_id=LEFT_ID,
    )
    right_patch = _derive(
        ancestor_value,
        right_value,
        base_id=BASE_ID,
        target_id=RIGHT_ID,
    )
    classification = classify_three_way_patches(
        ancestor_value,
        left_patch,
        right_patch,
    )
    requests = [
        _relation_request(
            classification,
            PreservationSource.LEFT_BRANCH,
            JsonPath((ObjectKey("b"),)),
        ),
        _relation_request(
            classification,
            PreservationSource.RIGHT_BRANCH,
            JsonPath((ObjectKey("a"),)),
        ),
    ]
    if include_neutral:
        requests.append(
            _relation_request(
                classification,
                PreservationSource.IDENTICAL_NEUTRAL,
                JsonPath((ObjectKey("same"),)),
            )
        )
    if include_absence:
        requests.extend(
            (
                _relation_request(
                    classification,
                    PreservationSource.LEFT_BRANCH,
                    JsonPath(
                        (ObjectKey("left_component"), ObjectKey("gone"))
                    ),
                ),
                _relation_request(
                    classification,
                    PreservationSource.RIGHT_BRANCH,
                    JsonPath(
                        (ObjectKey("right_component"), ObjectKey("gone"))
                    ),
                ),
            )
        )
    obligations = derive_preservation_obligations(classification, tuple(requests))
    case = _case(
        VariationKind.THREE_WAY_RECOMBINATION,
        (
            VariationParent(ParentRole.CROSSOVER_LEFT, left),
            VariationParent(ParentRole.CROSSOVER_RIGHT, right),
        ),
        ancestor=ancestor,
        patches=(left_patch, right_patch),
        obligations=obligations,
    )
    limits = left_patch.limits.json_limits
    configurations = (
        bind_parent_configuration(left, left_value, limits=limits),
        bind_parent_configuration(right, right_value, limits=limits),
    )
    claims = tuple(PreservationClaim(value.obligation_id) for value in obligations)
    child = {
        "a": 2,
        "b": 1,
        "same": 1,
        "left_component": {"keep": 0},
        "right_component": {"keep": 0},
    }
    return (
        ancestor_value,
        case,
        classification,
        configurations,
        limits,
        claims,
        child,
    )


def test_replay_derived_obligations_verify_both_branches_neutral_and_absence():
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    result = verify_preservation_claims(
        case,
        classification,
        configurations,
        child,
        claims=claims,
        limits=limits,
    )
    assert result.discriminatively_used_parent_ids == (LEFT_ID, RIGHT_ID)
    assert result.child_hash == typed_json_sha256(child)
    neutral = [
        value
        for value in case.preservation_obligations
        if value.source is PreservationSource.IDENTICAL_NEUTRAL
    ]
    absent = [
        value
        for value in case.preservation_obligations
        if value.expected_state is PreservationExpectation.ABSENT
    ]
    assert len(neutral) == 1
    assert len(absent) == 2
    assert all(value.absence_context_path is not None for value in absent)
    assert all(value.absence_context_shape_hash is not None for value in absent)
    validate_preservation_verification(result)


def test_preservation_verification_recursively_rejects_forged_receipts():
    valid_claims = (PreservationClaim("a" * 64), PreservationClaim("b" * 64))
    valid_parents = (LEFT_ID, RIGHT_ID)
    child_hash = "c" * 64
    valid = PreservationVerification(child_hash, valid_claims, valid_parents)
    validate_preservation_verification(valid)

    malformed_claim = object.__new__(PreservationClaim)
    object.__setattr__(malformed_claim, "obligation_id", "not-a-digest")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PreservationVerification(
            child_hash,
            (malformed_claim, valid_claims[1]),
            valid_parents,
        )

    malformed_parent = object.__new__(CandidateId)
    object.__setattr__(malformed_parent, "value", "not-a-candidate-id")
    with pytest.raises(ValueError, match="start with 'candidate'"):
        PreservationVerification(
            child_hash,
            valid_claims,
            (malformed_parent, RIGHT_ID),
        )

    with pytest.raises(ValueError, match="canonical"):
        PreservationVerification(
            child_hash,
            tuple(reversed(valid_claims)),
            valid_parents,
        )
    with pytest.raises(ValueError, match="duplicate"):
        PreservationVerification(
            child_hash,
            (valid_claims[0], valid_claims[0]),
            valid_parents,
        )
    with pytest.raises(ValueError, match="2 to 4096"):
        PreservationVerification(
            child_hash,
            tuple(
                PreservationClaim(f"{index:064x}")
                for index in range(MAX_PRESERVATION_OBLIGATIONS + 1)
            ),
            valid_parents,
        )
    with pytest.raises(ValueError, match="exactly two"):
        PreservationVerification(child_hash, valid_claims, ())
    with pytest.raises(ValueError, match="distinct"):
        PreservationVerification(child_hash, valid_claims, (LEFT_ID, LEFT_ID))

    forged_receipt = object.__new__(PreservationVerification)
    object.__setattr__(forged_receipt, "child_hash", child_hash)
    object.__setattr__(forged_receipt, "verified_claims", (malformed_claim,))
    object.__setattr__(forged_receipt, "discriminatively_used_parent_ids", ())
    with pytest.raises(ValueError, match="2 to 4096"):
        validate_preservation_verification(forged_receipt)


def test_frozen_json_implicit_equality_is_typed_and_values_are_unhashable():
    bool_array = FrozenJsonArray((True,))
    int_array = FrozenJsonArray((1,))
    float_array = FrozenJsonArray((1.0,))
    positive_zero = FrozenJsonArray((0.0,))
    negative_zero = FrozenJsonArray((-0.0,))

    for left, right in (
        (bool_array, int_array),
        (bool_array, float_array),
        (int_array, float_array),
        (positive_zero, negative_zero),
    ):
        assert left != right
        assert typed_json_equal(left, right) is False
    assert FrozenJsonArray((True,)) == bool_array
    with pytest.raises(TypeError, match="unhashable"):
        hash(bool_array)
    with pytest.raises(TypeError, match="unhashable"):
        hash(FrozenJsonObject((("x", True),)))


def test_patch_operation_and_relation_equality_follow_typed_canonical_identity():
    bool_operation = ReplaceScalar(JsonPath(), True, False, BASE_ID)
    int_operation = ReplaceScalar(JsonPath(), 1, 0, BASE_ID)
    float_operation = ReplaceScalar(JsonPath(), 1.0, 0.0, BASE_ID)

    assert bool_operation != int_operation
    assert int_operation != float_operation
    assert operation_effect_bytes(bool_operation) != operation_effect_bytes(int_operation)
    with pytest.raises(TypeError, match="unhashable"):
        hash(bool_operation)

    bool_relation = PatchRelation(
        ThreeWayRelationKind.DISJOINT,
        (bool_operation,),
        (),
    )
    int_relation = PatchRelation(
        ThreeWayRelationKind.DISJOINT,
        (int_operation,),
        (),
    )
    assert bool_relation != int_relation
    assert bool_relation.relation_id != int_relation.relation_id
    with pytest.raises(TypeError, match="unhashable"):
        hash(bool_relation)


def test_forged_mutable_or_cyclic_frozen_json_graphs_fail_closed():
    mutable_array = object.__new__(FrozenJsonArray)
    object.__setattr__(mutable_array, "items", [1])
    mutable_object = object.__new__(FrozenJsonObject)
    object.__setattr__(mutable_object, "items", [("x", 1)])
    cyclic_array = object.__new__(FrozenJsonArray)
    object.__setattr__(cyclic_array, "items", (cyclic_array,))

    for malformed, message in (
        (mutable_array, "exact tuple"),
        (mutable_object, "exact tuple"),
        (cyclic_array, "cycles"),
    ):
        for consumer in (
            freeze_json,
            thaw_json,
            canonical_typed_json_bytes,
            typed_json_sha256,
        ):
            with pytest.raises((TypeError, ValueError), match=message):
                consumer(malformed)
        with pytest.raises(TypeError, match="unhashable"):
            hash(malformed)

    with pytest.raises(TypeError, match="exact tuple"):
        InsertSequenceItem(
            JsonPath(),
            0,
            1,
            mutable_array,
            FrozenJsonArray((1, 1)),
            BASE_ID,
        )
    with pytest.raises(TypeError, match="exact tuple"):
        DeleteSequenceItem(
            JsonPath(),
            0,
            1,
            mutable_array,
            FrozenJsonArray(()),
            BASE_ID,
        )
    with pytest.raises(TypeError, match="exact tuple"):
        PermuteSequence(
            JsonPath(),
            (0,),
            mutable_array,
            FrozenJsonArray((1,)),
            BASE_ID,
        )


def test_implicit_equality_and_mapping_use_reject_hostile_nested_values_without_hooks():
    class _HostileValue:
        equality_calls = 0
        hash_calls = 0
        encode_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return True

        def __hash__(self):
            type(self).hash_calls += 1
            return 0

        def encode(self, *args, **kwargs):
            type(self).encode_calls += 1
            return b"hostile"

    hostile = _HostileValue()
    ancestor, _, _, left_patch, right_patch = _three_way_fixture()
    classification = classify_three_way_patches(ancestor, left_patch, right_patch)
    canonical_relation = classification.of_kind(
        ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT
    )[0]
    forged_relation = object.__new__(PatchRelation)
    object.__setattr__(forged_relation, "kind", canonical_relation.kind)
    object.__setattr__(
        forged_relation,
        "left_operations",
        canonical_relation.left_operations,
    )
    object.__setattr__(
        forged_relation,
        "right_operations",
        canonical_relation.right_operations,
    )
    object.__setattr__(forged_relation, "semantic_component", hostile)

    for comparison in (
        lambda: canonical_relation == forged_relation,
        lambda: forged_relation == canonical_relation,
    ):
        with pytest.raises(TypeError, match="exact string"):
            comparison()
    for mapping_use in (
        lambda: hash(forged_relation),
        lambda: {forged_relation},
        lambda: {forged_relation: "value"},
    ):
        with pytest.raises(TypeError, match="unhashable"):
            mapping_use()

    forged_classification = object.__new__(ThreeWayPatchClassification)
    for name in (
        "ancestor",
        "ancestor_candidate_id",
        "ancestor_hash",
        "left_patch_hash",
        "right_patch_hash",
        "left_patch",
        "right_patch",
    ):
        object.__setattr__(forged_classification, name, getattr(classification, name))
    object.__setattr__(
        forged_classification,
        "relations",
        tuple(
            forged_relation if relation is canonical_relation else relation
            for relation in classification.relations
        ),
    )
    for comparison in (
        lambda: classification == forged_classification,
        lambda: forged_classification == classification,
    ):
        with pytest.raises(TypeError, match="exact string"):
            comparison()
    with pytest.raises(TypeError, match="unhashable"):
        hash(forged_classification)

    valid_claims = (PreservationClaim("a" * 64), PreservationClaim("b" * 64))
    canonical_receipt = PreservationVerification(
        "c" * 64,
        valid_claims,
        (LEFT_ID, RIGHT_ID),
    )
    forged_claim = object.__new__(PreservationClaim)
    object.__setattr__(forged_claim, "obligation_id", hostile)
    forged_receipt = object.__new__(PreservationVerification)
    object.__setattr__(forged_receipt, "child_hash", canonical_receipt.child_hash)
    object.__setattr__(
        forged_receipt,
        "verified_claims",
        (forged_claim, valid_claims[1]),
    )
    object.__setattr__(
        forged_receipt,
        "discriminatively_used_parent_ids",
        canonical_receipt.discriminatively_used_parent_ids,
    )
    for comparison in (
        lambda: valid_claims[0] == forged_claim,
        lambda: forged_claim == valid_claims[0],
        lambda: canonical_receipt == forged_receipt,
        lambda: forged_receipt == canonical_receipt,
    ):
        with pytest.raises((TypeError, ValueError), match="SHA-256"):
            comparison()
    for mapping_use in (
        lambda: hash(forged_claim),
        lambda: {forged_claim},
        lambda: {forged_claim: "value"},
        lambda: hash(forged_receipt),
        lambda: {forged_receipt},
        lambda: {forged_receipt: "value"},
    ):
        with pytest.raises(TypeError, match="unhashable"):
            mapping_use()

    assert _HostileValue.equality_calls == 0
    assert _HostileValue.hash_calls == 0
    assert _HostileValue.encode_calls == 0


def test_paths_reject_forged_leaf_values_before_equality_or_hash_hooks():
    class _HostileValue:
        equality_calls = 0
        hash_calls = 0
        encode_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return True

        def __hash__(self):
            type(self).hash_calls += 1
            return 0

        def encode(self, *args, **kwargs):
            type(self).encode_calls += 1
            return b"hostile"

    hostile = _HostileValue()
    forged_key = object.__new__(ObjectKey)
    object.__setattr__(forged_key, "value", hostile)
    with pytest.raises(TypeError, match="exact string"):
        JsonPath((forged_key,))
    with pytest.raises(TypeError, match="exact string"):
        hash(forged_key)

    forged_path = object.__new__(JsonPath)
    object.__setattr__(forged_path, "segments", (forged_key,))
    canonical = JsonPath((ObjectKey("x"),))
    for implicit_use in (
        lambda: canonical == forged_path,
        lambda: forged_path == canonical,
        lambda: hash(forged_path),
    ):
        with pytest.raises(TypeError, match="exact string"):
            implicit_use()
    assert _HostileValue.equality_calls == 0
    assert _HostileValue.hash_calls == 0
    assert _HostileValue.encode_calls == 0


def test_nested_validation_and_collection_bounds_precede_implicit_operations():
    class _HostileValue:
        equality_calls = 0
        hash_calls = 0
        bool_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return True

        def __hash__(self):
            type(self).hash_calls += 1
            return 0

        def __bool__(self):
            type(self).bool_calls += 1
            return True

    hostile = _HostileValue()
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )

    forged_parent_id = object.__new__(CandidateId)
    object.__setattr__(forged_parent_id, "value", hostile)
    branch_obligation = next(
        value
        for value in case.preservation_obligations
        if value.source is not PreservationSource.IDENTICAL_NEUTRAL
    )
    with pytest.raises(ValueError, match="non-empty string"):
        replace(
            branch_obligation,
            source_parent_candidate_ids=(forged_parent_id,),
        )

    forged_patch = object.__new__(TypedPatch)
    valid_patch = case.ancestor_to_parent_patches[1]
    for name in (
        "base_candidate_id",
        "target_candidate_id",
        "base_hash",
        "target_hash",
        "operations",
        "schema_version",
    ):
        object.__setattr__(forged_patch, name, getattr(valid_patch, name))
    object.__setattr__(forged_patch, "limits", hostile)
    with pytest.raises(TypeError, match="exact PatchLimits"):
        replace(
            case,
            ancestor_to_parent_patches=(
                case.ancestor_to_parent_patches[0],
                forged_patch,
            ),
        )

    forged_limits = object.__new__(TypedJsonLimits)
    for name in (
        "max_depth",
        "max_nodes",
        "max_container_items",
        "max_string_bytes",
        "max_integer_bits",
        "max_canonical_bytes",
    ):
        object.__setattr__(forged_limits, name, getattr(limits, name))
    object.__setattr__(forged_limits, "max_depth", hostile)
    with pytest.raises(TypeError, match="max_depth must be an exact integer"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            child,
            claims=claims,
            limits=forged_limits,
        )

    with pytest.raises(ValueError, match="one or two"):
        replace(case, parents=(hostile, hostile, hostile))
    with pytest.raises(ValueError, match="MAX_PATCH_OPERATIONS"):
        PatchRelation(
            ThreeWayRelationKind.DISJOINT,
            (hostile,) * (patch_domain.MAX_PATCH_OPERATIONS + 1),
            (),
        )
    with pytest.raises(ValueError, match="byte limit"):
        patch_domain.validate_semantic_component("x" * 1_000_000)

    assert _HostileValue.equality_calls == 0
    assert _HostileValue.hash_calls == 0
    assert _HostileValue.bool_calls == 0


def test_deletion_receipts_reject_destroyed_or_reshaped_containers():
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    destroyed = dict(child)
    destroyed["left_component"] = None
    destroyed["right_component"] = False
    with pytest.raises(PreservationError, match="presence-aware"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            destroyed,
            claims=claims,
            limits=limits,
        )

    reshaped = dict(child)
    reshaped["left_component"] = {"keep": 0, "unrelated": 9}
    with pytest.raises(PreservationError, match="presence-aware"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            reshaped,
            claims=claims,
            limits=limits,
        )


def test_ancestor_copy_ignored_branch_and_missing_claim_fail_closed():
    ancestor, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    with pytest.raises(PreservationError, match="presence-aware"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            ancestor,
            claims=claims,
            limits=limits,
        )
    ignored_right = dict(child)
    ignored_right["a"] = 0
    ignored_right["right_component"] = {"keep": 0, "gone": 2}
    with pytest.raises(PreservationError, match="presence-aware"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            ignored_right,
            claims=claims,
            limits=limits,
        )
    with pytest.raises(PreservationError, match="every predeclared obligation"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            child,
            claims=claims[:-1],
            limits=limits,
        )


def test_ancestor_only_identical_branch_and_fabricated_obligations_fail_closed():
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    left_component_relation = next(
        relation
        for relation in classification.relations
        if any(
            operation.path
            == JsonPath((ObjectKey("left_component"),))
            for operation in relation.left_operations
        )
    )
    with pytest.raises(ValueError, match="actual change from the ancestor"):
        derive_preservation_obligations(
            classification,
            (
                PreservationObligationRequest(
                    left_component_relation.relation_id,
                    PreservationSource.LEFT_BRANCH,
                    JsonPath((ObjectKey("left_component"), ObjectKey("keep"))),
                ),
            ),
        )
    identical = classification.of_kind(ThreeWayRelationKind.IDENTICAL)[0]
    with pytest.raises(ValueError, match="neutral"):
        derive_preservation_obligations(
            classification,
            (
                PreservationObligationRequest(
                    identical.relation_id,
                    PreservationSource.LEFT_BRANCH,
                    JsonPath((ObjectKey("same"),)),
                ),
            ),
        )

    first = case.preservation_obligations[0]
    replacement_hash = "f" * 64
    if first.expected_value_hash == replacement_hash:
        replacement_hash = "e" * 64
    fabricated = replace(first, expected_value_hash=replacement_hash)
    fabricated_obligations = tuple(
        sorted(
            (fabricated,) + case.preservation_obligations[1:],
            key=lambda value: value.obligation_id,
        )
    )
    fabricated_case = replace(case, preservation_obligations=fabricated_obligations)
    fabricated_claims = tuple(
        PreservationClaim(value.obligation_id) for value in fabricated_obligations
    )
    with pytest.raises(PreservationError, match="not exact replay-derived"):
        verify_preservation_claims(
            fabricated_case,
            classification,
            configurations,
            child,
            claims=fabricated_claims,
            limits=limits,
        )
    with pytest.raises(PreservationError, match="every predeclared obligation"):
        verify_preservation_claims(
            case,
            classification,
            configurations,
            child,
            claims=tuple(sorted(claims[:-1] + (PreservationClaim("f" * 64),), key=lambda value: value.obligation_id)),
            limits=limits,
        )


def test_variation_case_rejects_duplicate_roles_overlaps_and_unbounded_inputs():
    _, case, classification, _, _, _, _ = _preservation_fixture()
    duplicate_roles = (
        VariationParent(ParentRole.CROSSOVER_LEFT, case.parents[0].occurrence),
        VariationParent(ParentRole.CROSSOVER_LEFT, case.parents[1].occurrence),
    )
    with pytest.raises(ValueError, match="unique"):
        _case(VariationKind.TWO_PARENT_CROSSOVER, duplicate_roles)

    left_component_relation = next(
        relation
        for relation in classification.relations
        if any(
            operation.path
            == JsonPath((ObjectKey("left_component"),))
            for operation in relation.left_operations
        )
    )
    overlapping = derive_preservation_obligations(
        classification,
        (
            PreservationObligationRequest(
                left_component_relation.relation_id,
                PreservationSource.LEFT_BRANCH,
                JsonPath((ObjectKey("left_component"),)),
            ),
            PreservationObligationRequest(
                left_component_relation.relation_id,
                PreservationSource.LEFT_BRANCH,
                JsonPath((ObjectKey("left_component"), ObjectKey("gone"))),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        replace(case, preservation_obligations=overlapping)

    one = case.preservation_obligations[0]
    with pytest.raises(ValueError, match="MAX_PRESERVATION_OBLIGATIONS"):
        replace(
            case,
            preservation_obligations=(one,) * (MAX_PRESERVATION_OBLIGATIONS + 1),
        )
    insight = InsightRef(InsightId("insight_m4b_bound"), 1)
    with pytest.raises(ValueError, match="MAX_SELECTED_INSIGHTS"):
        replace(case, selected_insights=(insight,) * (MAX_SELECTED_INSIGHTS + 1))


def test_lineage_values_reject_same_invocation_and_reverse_chronology():
    value = {"x": 0}
    digest = typed_json_sha256(value)
    invocation = OperatorInvocationId("operator_m4b_cycle")
    self_parent = CandidateOccurrence(BASE_ID, digest, digest, 1, invocation)
    with pytest.raises(ValueError, match="produced itself"):
        VariationCase(
            invocation,
            VariationKind.REPRODUCTION,
            "m4b.reproduction",
            1,
            (VariationParent(ParentRole.REPRODUCTION_SOURCE, self_parent),),
            1,
            CONTEXT_HASH,
            REWARD_HASH,
        )
    later_parent = CandidateOccurrence(BASE_ID, digest, digest, 2)
    earlier_child = CandidateOccurrence(CHILD_ID, digest, digest, 1)
    patch = _derive(value, value, base_id=BASE_ID, target_id=CHILD_ID)
    with pytest.raises(ValueError, match="later proposal sequence"):
        ParentEdge(
            ParentRole.REPRODUCTION_SOURCE,
            later_parent,
            earlier_child,
            patch,
        )

    valid_parent = CandidateOccurrence(BASE_ID, digest, digest, 0)
    case = VariationCase(
        invocation,
        VariationKind.REPRODUCTION,
        "m4b.reproduction",
        1,
        (VariationParent(ParentRole.REPRODUCTION_SOURCE, valid_parent),),
        1,
        CONTEXT_HASH,
        REWARD_HASH,
    )
    same_invocation = replace(valid_parent, operator_invocation_id=invocation)
    object.__setattr__(
        case,
        "parents",
        (VariationParent(ParentRole.REPRODUCTION_SOURCE, same_invocation),),
    )
    with pytest.raises(ValueError, match="produced itself"):
        validate_variation_case(case)


def _forged_dataclass_copy(value, **overrides):
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged,
            field.name,
            overrides.get(field.name, getattr(value, field.name)),
        )
    return forged


def test_every_exported_m4b_value_validates_before_implicit_equality_and_hash():
    class _HostileValue:
        equality_calls = 0
        hash_calls = 0
        bool_calls = 0
        encode_calls = 0
        ordering_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return True

        def __hash__(self):
            type(self).hash_calls += 1
            return 0

        def __bool__(self):
            type(self).bool_calls += 1
            return True

        def encode(self, *args, **kwargs):
            type(self).encode_calls += 1
            return b"hostile"

        def __lt__(self, other):
            type(self).ordering_calls += 1
            return False

    hostile = _HostileValue()
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    verification = verify_preservation_claims(
        case,
        classification,
        configurations,
        child,
        claims=claims,
        limits=limits,
    )
    parent = case.common_ancestor
    assert parent is not None
    edge = ParentEdge(
        ParentRole.CROSSOVER_LEFT,
        parent,
        case.parents[0].occurrence,
        classification.left_patch,
    )
    component = ComponentTagAssignment(JsonPath((ObjectKey("x"),)), "component")
    request = PreservationObligationRequest(
        classification.relations[0].relation_id,
        PreservationSource.LEFT_BRANCH,
        JsonPath((ObjectKey("x"),)),
    )
    resolution = PatchResolution("a" * 64, ResolutionChoice.DROP_BOTH)

    forged_occurrence = _forged_dataclass_copy(
        case.parents[0].occurrence,
        configuration_hash=hostile,
    )
    forged_parent = _forged_dataclass_copy(parent, configuration_hash=hostile)
    cases = (
        (
            "typed-json limits",
            limits,
            _forged_dataclass_copy(limits, max_depth=hostile),
            True,
        ),
        (
            "patch limits",
            classification.left_patch.limits,
            _forged_dataclass_copy(
                classification.left_patch.limits,
                max_operations=hostile,
            ),
            True,
        ),
        (
            "candidate occurrence",
            case.parents[0].occurrence,
            forged_occurrence,
            False,
        ),
        (
            "variation parent",
            case.parents[0],
            _forged_dataclass_copy(
                case.parents[0],
                occurrence=forged_occurrence,
            ),
            False,
        ),
        (
            "parent edge",
            edge,
            _forged_dataclass_copy(edge, parent=forged_parent),
            False,
        ),
        (
            "preservation obligation",
            case.preservation_obligations[0],
            _forged_dataclass_copy(
                case.preservation_obligations[0],
                relation_id=hostile,
            ),
            False,
        ),
        (
            "variation case",
            case,
            _forged_dataclass_copy(case, operator_id=hostile),
            False,
        ),
        (
            "component assignment",
            component,
            _forged_dataclass_copy(component, component=hostile),
            False,
        ),
        (
            "obligation request",
            request,
            _forged_dataclass_copy(request, relation_id=hostile),
            False,
        ),
        (
            "patch resolution",
            resolution,
            _forged_dataclass_copy(resolution, relation_id=hostile),
            False,
        ),
        (
            "parent configuration",
            configurations[0],
            _forged_dataclass_copy(
                configurations[0],
                occurrence=forged_occurrence,
            ),
            False,
        ),
    )

    for label, valid, forged, supports_hash in cases:
        canonical_copy = _forged_dataclass_copy(valid)
        assert valid == canonical_copy, label
        assert canonical_copy == valid, label
        for comparison in (
            lambda valid=valid, forged=forged: valid == forged,
            lambda valid=valid, forged=forged: forged == valid,
        ):
            with pytest.raises((TypeError, ValueError), match="exact|SHA-256|token"):
                comparison()
        for implicit_hash_use in (
            lambda forged=forged: hash(forged),
            lambda forged=forged: {forged},
            lambda forged=forged: {forged: "value"},
        ):
            with pytest.raises(TypeError):
                implicit_hash_use()
        if supports_hash:
            assert hash(valid) == hash(canonical_copy), label
            assert {valid: label}[canonical_copy] == label
        else:
            with pytest.raises(TypeError, match="unhashable"):
                hash(valid)

    assert type(configurations[0]) is ParentConfiguration
    assert verification == _forged_dataclass_copy(verification)
    assert _HostileValue.equality_calls == 0
    assert _HostileValue.hash_calls == 0
    assert _HostileValue.bool_calls == 0
    assert _HostileValue.encode_calls == 0
    assert _HostileValue.ordering_calls == 0


def test_custom_equality_never_delegates_to_a_foreign_reflected_hook():
    class _ForeignValue:
        equality_calls = 0

        def __eq__(self, other):
            type(self).equality_calls += 1
            return True

    foreign = _ForeignValue()
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    verification = verify_preservation_claims(
        case,
        classification,
        configurations,
        child,
        claims=claims,
        limits=limits,
    )
    parent = case.common_ancestor
    assert parent is not None
    edge = ParentEdge(
        ParentRole.CROSSOVER_LEFT,
        parent,
        case.parents[0].occurrence,
        classification.left_patch,
    )
    frozen_object = freeze_json({"x": [1]})
    assert type(frozen_object) is FrozenJsonObject
    frozen_array = frozen_object.items[0][1]
    assert type(frozen_array) is FrozenJsonArray
    values = (
        limits,
        classification.left_patch.limits,
        frozen_array,
        frozen_object,
        ObjectKey("x"),
        ArrayIndex(0),
        JsonPath((ObjectKey("x"),)),
        classification.left_patch.operations[0],
        classification.left_patch,
        case.parents[0].occurrence,
        case.parents[0],
        edge,
        PreservationClaim("a" * 64),
        case.preservation_obligations[0],
        case,
        ComponentTagAssignment(JsonPath((ObjectKey("x"),)), "component"),
        classification.relations[0],
        classification,
        PreservationObligationRequest(
            classification.relations[0].relation_id,
            PreservationSource.LEFT_BRANCH,
            JsonPath((ObjectKey("x"),)),
        ),
        PatchResolution("a" * 64, ResolutionChoice.DROP_BOTH),
        configurations[0],
        verification,
    )
    for value in values:
        assert (value == foreign) is False
    assert _ForeignValue.equality_calls == 0


def test_every_exported_m4b_dataclass_rejects_inheriting_subclass_aliases():
    _, case, classification, configurations, limits, claims, child = (
        _preservation_fixture()
    )
    verification = verify_preservation_claims(
        case,
        classification,
        configurations,
        child,
        claims=claims,
        limits=limits,
    )
    parent = case.common_ancestor
    assert parent is not None
    edge = ParentEdge(
        ParentRole.CROSSOVER_LEFT,
        parent,
        case.parents[0].occurrence,
        classification.left_patch,
    )
    frozen_object = freeze_json({"x": [1]})
    assert type(frozen_object) is FrozenJsonObject
    frozen_array = frozen_object.items[0][1]
    assert type(frozen_array) is FrozenJsonArray
    replace_scalar = ReplaceScalar(JsonPath(), 0, 1, BASE_ID)
    replace_subtree = ReplaceSubtree(
        JsonPath(),
        freeze_json({}),
        freeze_json({"x": 1}),
        BASE_ID,
    )
    insert_item = InsertSequenceItem(
        JsonPath(),
        1,
        1,
        FrozenJsonArray((0,)),
        FrozenJsonArray((0, 1)),
        BASE_ID,
    )
    delete_item = DeleteSequenceItem(
        JsonPath(),
        0,
        0,
        FrozenJsonArray((0, 1)),
        FrozenJsonArray((1,)),
        BASE_ID,
    )
    permute = PermuteSequence(
        JsonPath(),
        (1, 0),
        FrozenJsonArray((0, 1)),
        FrozenJsonArray((1, 0)),
        BASE_ID,
    )
    values = (
        limits,
        classification.left_patch.limits,
        frozen_array,
        frozen_object,
        ObjectKey("x"),
        ArrayIndex(0),
        JsonPath((ObjectKey("x"),)),
        replace_scalar,
        replace_subtree,
        insert_item,
        delete_item,
        permute,
        classification.left_patch,
        case.parents[0].occurrence,
        case.parents[0],
        edge,
        PreservationClaim("a" * 64),
        case.preservation_obligations[0],
        case,
        ComponentTagAssignment(JsonPath((ObjectKey("x"),)), "component"),
        classification.relations[0],
        classification,
        PreservationObligationRequest(
            classification.relations[0].relation_id,
            PreservationSource.LEFT_BRANCH,
            JsonPath((ObjectKey("x"),)),
        ),
        PatchResolution("a" * 64, ResolutionChoice.DROP_BOTH),
        configurations[0],
        verification,
    )
    assert len(values) == 26

    for value in values:
        value_type = type(value)
        subclass = type(f"_Inheriting{value_type.__name__}", (value_type,), {})
        forged = object.__new__(subclass)
        for field in fields(value):
            object.__setattr__(forged, field.name, getattr(value, field.name))

        # Python gives a right-hand proper subclass reflected-method priority.
        # The inherited implementation must therefore reject its own non-exact
        # receiver, not merely inspect the other operand.
        assert (value == forged) is False, value_type.__name__
        assert (forged == value) is False, value_type.__name__
        assert forged not in [value], value_type.__name__
        assert forged not in (value,), value_type.__name__
        assert value not in [forged], value_type.__name__
        assert value not in (forged,), value_type.__name__
        for mapping_use in (
            lambda forged=forged: hash(forged),
            lambda forged=forged: {forged},
            lambda forged=forged: {forged: "value"},
            lambda value=value, forged=forged: {value: "value"}.get(forged),
        ):
            with pytest.raises(TypeError):
                mapping_use()

    obligation_subclass = type(
        "_InheritingPreservationObligation",
        (PreservationObligation,),
        {},
    )
    forged_obligation = object.__new__(obligation_subclass)
    for field in fields(case.preservation_obligations[0]):
        object.__setattr__(
            forged_obligation,
            field.name,
            getattr(case.preservation_obligations[0], field.name),
        )
    with pytest.raises(TypeError, match="exact PreservationObligation"):
        _ = forged_obligation.obligation_id

    classification_subclass = type(
        "_InheritingThreeWayPatchClassification",
        (ThreeWayPatchClassification,),
        {},
    )
    forged_classification = object.__new__(classification_subclass)
    for field in fields(classification):
        object.__setattr__(
            forged_classification,
            field.name,
            getattr(classification, field.name),
        )
    with pytest.raises(TypeError, match="exact ThreeWayPatchClassification"):
        forged_classification.revalidate()


def test_operation_effect_component_flag_requires_an_exact_boolean_without_hooks():
    class _HostileTruth:
        bool_calls = 0

        def __bool__(self):
            type(self).bool_calls += 1
            return type(self).bool_calls % 2 == 1

    path = JsonPath((ObjectKey("x"),))
    patch = _derive(
        {"x": 0},
        {"x": 1},
        tags=(ComponentTagAssignment(path, "component"),),
    )
    operation = patch.operations[0]
    assert operation_effect_bytes(operation, include_component=True) != (
        operation_effect_bytes(operation, include_component=False)
    )
    hostile = _HostileTruth()
    for invalid in (1, 0, None, hostile):
        with pytest.raises(TypeError, match="exact Boolean"):
            operation_effect_bytes(operation, include_component=invalid)
    assert _HostileTruth.bool_calls == 0
