"""Offline adversarial tests for the isolated M4c durable lineage codec."""

from __future__ import annotations

import ast
import json
import math
import os
import random
import struct
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.lineage import (
    AbsenceContextKind,
    AbsenceFailureKind,
    CandidateOccurrence,
    ParentEdge,
    ParentRole,
    PreservationClaim,
    PreservationExpectation,
    PreservationSource,
    VariationCase,
    VariationKind,
    VariationParent,
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
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    TypedJsonLimits,
    freeze_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.infrastructure.lineage_codec import (
    LINEAGE_CODEC_FORMAT,
    LINEAGE_CODEC_SCHEMA_VERSION,
    LINEAGE_WIRE_KINDS,
    M4B_EXPORTED_VALUE_TYPES,
    LineageCodecError,
    LineageCodecLimits,
    decode_lineage_value,
    encode_lineage_value,
)
from agent_evolve.policies.variation.typed_patch import (
    ComponentTagAssignment,
    ParentConfiguration,
    PatchRelation,
    PatchResolution,
    PreservationObligationRequest,
    PreservationVerification,
    ResolutionChoice,
    ThreeWayPatchClassification,
    ThreeWayRelationKind,
    bind_parent_configuration,
    classify_three_way_patches,
    derive_patch,
    derive_preservation_obligations,
    verify_preservation_claims,
)


BASE_ID = CandidateId("candidate_codec_base")
LEFT_ID = CandidateId("candidate_codec_left")
RIGHT_ID = CandidateId("candidate_codec_right")
CONTEXT_HASH = "a" * 64
REWARD_HASH = "b" * 64


def _canonical_wire(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _occurrence(candidate_id: CandidateId, value: object, sequence: int):
    digest = typed_json_sha256(value)
    return CandidateOccurrence(candidate_id, digest, digest, sequence)


def _relation_request(
    classification: ThreeWayPatchClassification,
    source: PreservationSource,
    path: JsonPath,
) -> PreservationObligationRequest:
    attribute = (
        "left_operations"
        if source is PreservationSource.LEFT_BRANCH
        else "right_operations"
    )
    relations = [
        relation
        for relation in classification.relations
        if any(
            operation.path.is_prefix_of(path)
            for operation in getattr(relation, attribute)
        )
    ]
    assert len(relations) == 1
    return PreservationObligationRequest(relations[0].relation_id, source, path)


def _complete_fixture():
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
    left_patch = derive_patch(
        ancestor_value,
        left_value,
        base_candidate_id=BASE_ID,
        target_candidate_id=LEFT_ID,
    )
    right_patch = derive_patch(
        ancestor_value,
        right_value,
        base_candidate_id=BASE_ID,
        target_candidate_id=RIGHT_ID,
    )
    classification = classify_three_way_patches(
        ancestor_value,
        left_patch,
        right_patch,
    )
    requests = (
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
    )
    obligations = derive_preservation_obligations(classification, requests)
    case = VariationCase(
        operator_invocation_id=OperatorInvocationId("operator_codec_three_way"),
        variation_kind=VariationKind.THREE_WAY_RECOMBINATION,
        operator_id="codec.three_way",
        operator_version=7,
        parents=(
            VariationParent(ParentRole.CROSSOVER_LEFT, left),
            VariationParent(ParentRole.CROSSOVER_RIGHT, right),
        ),
        requested_child_count=2,
        context_stratum_hash=CONTEXT_HASH,
        reward_definition_hash=REWARD_HASH,
        common_ancestor=ancestor,
        ancestor_to_parent_patches=(left_patch, right_patch),
        selected_insights=(InsightRef(InsightId("insight_codec_selected"), 3),),
        preservation_obligations=obligations,
    )
    limits = left_patch.limits.json_limits
    configurations = (
        bind_parent_configuration(left, left_value, limits=limits),
        bind_parent_configuration(right, right_value, limits=limits),
    )
    claims = tuple(PreservationClaim(item.obligation_id) for item in obligations)
    child = {
        "a": 2,
        "b": 1,
        "same": 1,
        "left_component": {"keep": 0},
        "right_component": {"keep": 0},
    }
    verification = verify_preservation_claims(
        case,
        classification,
        configurations,
        child,
        claims=claims,
        limits=limits,
    )
    edge = ParentEdge(
        ParentRole.CROSSOVER_LEFT,
        ancestor,
        left,
        left_patch,
    )
    frozen_object = freeze_json({"unicode": [True, 1, 1.0, -0.0, "é漢🙂"]})
    assert type(frozen_object) is FrozenJsonObject
    frozen_array = frozen_object.items[0][1]
    assert type(frozen_array) is FrozenJsonArray
    operations = (
        ReplaceScalar(JsonPath(), 0, 1, BASE_ID),
        ReplaceSubtree(
            JsonPath(),
            freeze_json({}),
            freeze_json({"x": 1}),
            BASE_ID,
        ),
        InsertSequenceItem(
            JsonPath(),
            1,
            1,
            FrozenJsonArray((0,)),
            FrozenJsonArray((0, 1)),
            BASE_ID,
        ),
        DeleteSequenceItem(
            JsonPath(),
            0,
            0,
            FrozenJsonArray((0, 1)),
            FrozenJsonArray((1,)),
            BASE_ID,
        ),
        PermuteSequence(
            JsonPath(),
            (1, 0),
            FrozenJsonArray((0, 1)),
            FrozenJsonArray((1, 0)),
            BASE_ID,
        ),
    )
    values = (
        limits,
        left_patch.limits,
        frozen_array,
        frozen_object,
        ObjectKey("x"),
        ArrayIndex(0),
        JsonPath((ObjectKey("x"), ArrayIndex(0))),
        *operations,
        left_patch,
        left,
        case.parents[0],
        edge,
        PreservationClaim("c" * 64),
        obligations[0],
        case,
        ComponentTagAssignment(JsonPath((ObjectKey("x"),)), "component"),
        classification.relations[0],
        classification,
        PreservationObligationRequest(
            classification.relations[0].relation_id,
            PreservationSource.LEFT_BRANCH,
            JsonPath((ObjectKey("x"),)),
        ),
        PatchResolution("d" * 64, ResolutionChoice.DROP_BOTH),
        configurations[0],
        verification,
    )
    assert len(values) == 26
    return values, case, classification, configurations, verification


def _forged_copy(base: object, cls: type | None = None, **overrides: object):
    forged = object.__new__(type(base) if cls is None else cls)
    for field in fields(base):
        object.__setattr__(
            forged,
            field.name,
            overrides.get(field.name, getattr(base, field.name)),
        )
    return forged


def _assert_decode_rejects(content: object) -> None:
    with pytest.raises(LineageCodecError) as captured:
        decode_lineage_value(content)  # type: ignore[arg-type]
    error = captured.value
    assert str(error) == "lineage decode rejected bytes"
    assert error.__cause__ is None
    assert error.__context__ is None


def _assert_encode_rejects(value: object) -> None:
    with pytest.raises(LineageCodecError) as captured:
        encode_lineage_value(value)
    error = captured.value
    assert str(error) == "lineage encode rejected value"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_closed_inventory_round_trips_all_26_exported_dataclasses_exactly():
    values, _, _, _, _ = _complete_fixture()
    assert len(M4B_EXPORTED_VALUE_TYPES) == len(set(M4B_EXPORTED_VALUE_TYPES)) == 26
    assert {type(value) for value in values} == set(M4B_EXPORTED_VALUE_TYPES)

    for value in values:
        first = encode_lineage_value(value)
        second = encode_lineage_value(value)
        decoded = decode_lineage_value(first)
        assert first == second
        assert type(decoded) is type(value)
        assert decoded == value
        assert encode_lineage_value(decoded) == first


def test_ast_inventory_matches_the_frozen_m4b_export_surface():
    source_root = Path(__file__).parents[1] / "src" / "agent_evolve"
    paths = (
        source_root / "domain" / "typed_json.py",
        source_root / "domain" / "patch.py",
        source_root / "domain" / "lineage.py",
        source_root / "policies" / "variation" / "typed_patch.py",
    )
    exported_dataclasses: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dataclasses = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any("dataclass" in ast.unparse(item) for item in node.decorator_list)
        }
        exported: tuple[str, ...] = ()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                exported = tuple(ast.literal_eval(node.value))
        exported_dataclasses.extend(name for name in exported if name in dataclasses)
    assert len(exported_dataclasses) == len(set(exported_dataclasses)) == 26
    assert set(exported_dataclasses) == {
        value_type.__name__ for value_type in M4B_EXPORTED_VALUE_TYPES
    }


def test_all_relevant_ids_enums_and_insight_references_round_trip():
    values = (
        CandidateId("candidate_codec_direct"),
        OperatorInvocationId("operator_codec_direct"),
        InsightId("insight_codec_direct"),
        InsightRef(InsightId("insight_codec_ref"), 9),
        *tuple(VariationKind),
        *tuple(ParentRole),
        *tuple(PreservationSource),
        *tuple(PreservationExpectation),
        *tuple(AbsenceContextKind),
        *tuple(AbsenceFailureKind),
        *tuple(ThreeWayRelationKind),
        *tuple(ResolutionChoice),
    )
    for value in values:
        encoded = encode_lineage_value(value)
        decoded = decode_lineage_value(encoded)
        assert type(decoded) is type(value)
        assert decoded == value
        assert encode_lineage_value(decoded) == encoded


def test_all_resolution_choices_and_all_five_operations_round_trip():
    resolutions = (
        PatchResolution("1" * 64, ResolutionChoice.CHOOSE_LEFT),
        PatchResolution("2" * 64, ResolutionChoice.CHOOSE_RIGHT),
        PatchResolution(
            "3" * 64,
            ResolutionChoice.SYNTHESIZE,
            synthesized_result_hash="4" * 64,
        ),
        PatchResolution("5" * 64, ResolutionChoice.DROP_BOTH),
    )
    values, _, _, _, _ = _complete_fixture()
    operations = tuple(
        value
        for value in values
        if type(value)
        in (
            ReplaceScalar,
            ReplaceSubtree,
            InsertSequenceItem,
            DeleteSequenceItem,
            PermuteSequence,
        )
    )
    assert len(operations) == 5
    for value in resolutions + operations:
        decoded = decode_lineage_value(encode_lineage_value(value))
        assert type(decoded) is type(value)
        assert decoded == value


def test_scalar_adjacency_signed_zero_and_finite_float_bits_are_lossless():
    values = (
        None,
        False,
        True,
        -1,
        0,
        1,
        1.0,
        0.0,
        -0.0,
        math.nextafter(0.0, 1.0),
        math.nextafter(1.0, 2.0),
        "",
        "é漢🙂",
    )
    encodings = tuple(encode_lineage_value(value) for value in values)
    assert len(encodings) == len(set(encodings))
    for value, encoded in zip(values, encodings):
        decoded = decode_lineage_value(encoded)
        assert type(decoded) is type(value)
        if type(value) is float:
            assert struct.pack(">d", decoded) == struct.pack(">d", value)
        else:
            assert decoded == value
    assert b'"bits":"0000000000000000"' in encode_lineage_value(0.0)
    assert b'"bits":"8000000000000000"' in encode_lineage_value(-0.0)
    for value in (float("nan"), float("inf"), float("-inf")):
        _assert_encode_rejects(value)


def test_random_finite_float_bit_patterns_round_trip_exactly():
    rng = random.Random(20260713)
    checked = 0
    while checked < 1000:
        bits = rng.getrandbits(64).to_bytes(8, "big")
        value = struct.unpack(">d", bits)[0]
        if not math.isfinite(value):
            continue
        decoded = decode_lineage_value(encode_lineage_value(value))
        assert struct.pack(">d", decoded) == bits
        checked += 1


def test_nested_unicode_typed_json_randomized_round_trips():
    rng = random.Random(20260713)
    scalars = (None, False, True, -7, 0, 9, 0.0, -0.0, 1.25, "", "é", "漢", "🙂")
    keys = ("a", "z", "é", "漢", "🙂")

    def tree(depth: int = 0):
        if depth == 4 or rng.random() < 0.42:
            return rng.choice(scalars)
        if rng.random() < 0.5:
            return [tree(depth + 1) for _ in range(rng.randrange(5))]
        chosen = rng.sample(keys, rng.randrange(5))
        return {f"{key}{depth}": tree(depth + 1) for key in chosen}

    for _ in range(500):
        frozen = freeze_json(tree())
        decoded = decode_lineage_value(encode_lineage_value(frozen))
        assert type(decoded) is type(frozen)
        assert typed_json_equal(decoded, frozen)


def test_domain_maximum_limit_fields_and_deep_value_are_explicit_and_lossless():
    json_limits = TypedJsonLimits(
        max_depth=64,
        max_nodes=50_000,
        max_container_items=10_000,
        max_string_bytes=1_048_576,
        max_integer_bits=4096,
        max_canonical_bytes=8_388_608,
    )
    patch_limits = PatchLimits(
        json_limits=json_limits,
        max_operations=4096,
        max_path_segments=64,
        max_patch_bytes=67_108_864,
    )
    for value in (json_limits, patch_limits, ArrayIndex((1 << 63) - 1), 1 << 4095):
        decoded = decode_lineage_value(encode_lineage_value(value))
        assert type(decoded) is type(value)
        assert decoded == value

    nested: object = 0
    for _ in range(64):
        nested = [nested]
    frozen = freeze_json(nested, limits=json_limits)
    decoded = decode_lineage_value(encode_lineage_value(frozen))
    assert typed_json_equal(decoded, frozen, limits=json_limits)

    large_key = "k" * 4096
    large_string = "s" * 1_048_576
    for value in (ObjectKey(large_key), large_string):
        decoded = decode_lineage_value(encode_lineage_value(value))
        assert decoded == value


def test_wire_contains_every_nested_limit_and_patch_schema_field():
    patch = derive_patch(
        {"x": 0},
        {"x": 1},
        base_candidate_id=BASE_ID,
        target_candidate_id=LEFT_ID,
        limits=PatchLimits(
            json_limits=TypedJsonLimits(
                max_depth=9,
                max_nodes=101,
                max_container_items=17,
                max_string_bytes=211,
                max_integer_bits=307,
                max_canonical_bytes=4099,
            ),
            max_operations=13,
            max_path_segments=11,
            max_patch_bytes=5001,
        ),
    )
    record = json.loads(encode_lineage_value(patch))
    patch_record = record["value"]
    assert patch_record["schema_version"] == "typed_json_patch_v1"
    assert patch_record["limits"] == {
        "json_limits": {
            "kind": "typed_json_limits",
            "max_canonical_bytes": 4099,
            "max_container_items": 17,
            "max_depth": 9,
            "max_integer_bits": 307,
            "max_nodes": 101,
            "max_string_bytes": 211,
        },
        "kind": "patch_limits",
        "max_operations": 13,
        "max_patch_bytes": 5001,
        "max_path_segments": 11,
    }


def test_equal_content_occurrences_keep_distinct_ids_roles_and_tuple_order():
    value = {"same": [True, 1, 1.0, -0.0]}
    left = _occurrence(CandidateId("candidate_codec_duplicate_left"), value, 1)
    right = _occurrence(CandidateId("candidate_codec_duplicate_right"), value, 2)
    assert left.configuration_hash == right.configuration_hash
    case = VariationCase(
        OperatorInvocationId("operator_codec_duplicate_case"),
        VariationKind.TWO_PARENT_CROSSOVER,
        "codec.duplicate_content",
        1,
        (
            VariationParent(ParentRole.CROSSOVER_LEFT, left),
            VariationParent(ParentRole.CROSSOVER_RIGHT, right),
        ),
        1,
        CONTEXT_HASH,
        REWARD_HASH,
    )
    decoded = decode_lineage_value(encode_lineage_value(case))
    assert type(decoded) is VariationCase
    assert tuple(parent.role for parent in decoded.parents) == (
        ParentRole.CROSSOVER_LEFT,
        ParentRole.CROSSOVER_RIGHT,
    )
    assert tuple(parent.occurrence.candidate_id for parent in decoded.parents) == (
        left.candidate_id,
        right.candidate_id,
    )
    assert decoded.parents[0].occurrence != decoded.parents[1].occurrence

    tampered = json.loads(encode_lineage_value(case))
    tampered["value"]["parents"].reverse()
    _assert_decode_rejects(_canonical_wire(tampered))


def test_frozen_golden_vectors_are_literal_and_reencode_identically():
    occurrence = CandidateOccurrence(
        CandidateId("candidate_codec_golden"),
        "a" * 64,
        "b" * 64,
        7,
        None,
    )
    vectors = (
        (
            True,
            b'{"format":"agent_evolve.lineage_value","schema_version":1,'
            b'"value":{"kind":"json_bool","value":true}}',
        ),
        (
            -0.0,
            b'{"format":"agent_evolve.lineage_value","schema_version":1,'
            b'"value":{"bits":"8000000000000000","kind":"json_float"}}',
        ),
        (
            occurrence,
            b'{"format":"agent_evolve.lineage_value","schema_version":1,"value":{'
            b'"candidate_id":{"kind":"candidate_id","value":"candidate_codec_golden"},'
            b'"configuration_artifact_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
            b'"configuration_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            b'"kind":"candidate_occurrence","operator_invocation_id":null,"proposal_sequence":7}}',
        ),
    )
    for value, golden in vectors:
        assert encode_lineage_value(value) == golden
        decoded = decode_lineage_value(golden)
        assert type(decoded) is type(value)
        assert encode_lineage_value(decoded) == golden


def test_golden_is_stable_across_process_hash_seeds():
    script = (
        "from agent_evolve.infrastructure.lineage_codec import encode_lineage_value;"
        "import sys;sys.stdout.buffer.write(encode_lineage_value(-0.0))"
    )
    expected = encode_lineage_value(-0.0)
    for seed in ("1", "99991"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            env=environment,
            timeout=10,
        )
        assert completed.stdout == expected
        assert completed.stderr == b""


@pytest.mark.parametrize(
    "mutation",
    (
        "one_byte",
        "key",
        "tag",
        "type",
        "missing_field",
        "extra_field",
        "field_value",
        "field_order",
    ),
)
def test_one_byte_key_tag_type_field_and_order_tampering_rejects(mutation):
    encoded = encode_lineage_value(ArrayIndex(7))
    record = json.loads(encoded)
    if mutation == "one_byte":
        tampered = encoded.replace(b"array_index", b"array_indeX", 1)
    elif mutation == "key":
        record["formaX"] = record.pop("format")
        tampered = _canonical_wire(record)
    elif mutation == "tag":
        record["value"]["kind"] = "array-index"
        tampered = _canonical_wire(record)
    elif mutation == "type":
        record["value"]["value"] = True
        tampered = _canonical_wire(record)
    elif mutation == "missing_field":
        del record["value"]["value"]
        tampered = _canonical_wire(record)
    elif mutation == "extra_field":
        record["value"]["alias"] = 7
        tampered = _canonical_wire(record)
    elif mutation == "field_value":
        record["value"]["value"] = -1
        tampered = _canonical_wire(record)
    else:
        tampered = json.dumps(
            {
                "schema_version": record["schema_version"],
                "format": record["format"],
                "value": record["value"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    _assert_decode_rejects(tampered)


def test_unknown_versions_kinds_aliases_and_operation_retagging_reject():
    operation = ReplaceScalar(JsonPath(), 0, 1, BASE_ID)
    cases = []
    record = json.loads(encode_lineage_value(operation))
    for version in (0, 2, True, "1"):
        changed = json.loads(encode_lineage_value(operation))
        changed["schema_version"] = version
        cases.append(_canonical_wire(changed))
    for tag in ("bool", "replace", "ReplaceScalar", "replace_subtree"):
        changed = json.loads(encode_lineage_value(operation))
        changed["value"]["kind"] = tag
        cases.append(_canonical_wire(changed))
    for content in cases:
        _assert_decode_rejects(content)


def test_noncanonical_json_utf8_bom_duplicates_numbers_and_whitespace_reject():
    encoded = encode_lineage_value("é")
    duplicate = encoded.replace(
        b'"schema_version":1',
        b'"schema_version":1,"schema_version":1',
        1,
    )
    ascii_escaped = json.dumps(
        json.loads(encoded),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    float_version = encoded.replace(b'"schema_version":1', b'"schema_version":1.0')
    constant_version = encoded.replace(b'"schema_version":1', b'"schema_version":NaN')
    escaped_surrogate = encoded.replace("é".encode(), b"\\ud800")
    cases = (
        b"\xef\xbb\xbf" + encoded,
        b"\xff",
        duplicate,
        ascii_escaped,
        float_version,
        constant_version,
        escaped_surrogate,
        b" " + encoded,
        encoded + b"\n",
        json.dumps(json.loads(encoded), indent=2, ensure_ascii=False).encode(),
    )
    for content in cases:
        _assert_decode_rejects(content)
    _assert_decode_rejects(bytearray(encoded))
    _assert_decode_rejects(encoded.decode())


def test_codec_byte_depth_node_container_string_and_integer_bounds():
    array = FrozenJsonArray((0, 1, 2, 3))
    encoded = encode_lineage_value(array)
    exact = LineageCodecLimits(max_bytes=len(encoded))
    assert encode_lineage_value(array, limits=exact) == encoded
    with pytest.raises(LineageCodecError):
        encode_lineage_value(array, limits=LineageCodecLimits(max_bytes=len(encoded) - 1))
    with pytest.raises(LineageCodecError):
        encode_lineage_value(array, limits=LineageCodecLimits(max_container_items=3))
    with pytest.raises(LineageCodecError):
        encode_lineage_value("x" * 27, limits=LineageCodecLimits(max_string_bytes=26))
    with pytest.raises(LineageCodecError):
        encode_lineage_value(123, limits=LineageCodecLimits(max_integer_digits=2))
    with pytest.raises(LineageCodecError):
        encode_lineage_value(
            FrozenJsonArray((FrozenJsonArray((0,)),)),
            limits=LineageCodecLimits(max_depth=4),
        )
    with pytest.raises(LineageCodecError):
        decode_lineage_value(encoded, limits=LineageCodecLimits(max_nodes=5))


def test_constructor_bypassed_valid_exact_copies_revalidate_and_round_trip():
    values, _, _, _, _ = _complete_fixture()
    for value in values:
        copied = _forged_copy(value)
        assert type(copied) is type(value)
        assert encode_lineage_value(copied) == encode_lineage_value(value)


def test_constructor_bypassed_hostile_fields_fail_before_every_hook():
    class Hostile:
        equality_calls = 0
        hash_calls = 0
        bool_calls = 0
        encode_calls = 0
        ordering_calls = 0
        string_calls = 0
        repr_calls = 0
        iteration_calls = 0
        length_calls = 0

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

        def __str__(self):
            type(self).string_calls += 1
            return "hostile"

        def __repr__(self):
            type(self).repr_calls += 1
            return "hostile"

        def __iter__(self):
            type(self).iteration_calls += 1
            return iter(())

        def __len__(self):
            type(self).length_calls += 1
            return 0

    hostile = Hostile()
    values, _, _, _, _ = _complete_fixture()
    invalid = {
        "TypedJsonLimits": {"max_depth": hostile},
        "FrozenJsonArray": {"items": (hostile,)},
        "FrozenJsonObject": {"items": ((hostile, 1),)},
        "ObjectKey": {"value": hostile},
        "ArrayIndex": {"value": hostile},
        "JsonPath": {"segments": (hostile,)},
        "PatchLimits": {"max_operations": hostile},
        "ReplaceScalar": {"old_value": hostile},
        "ReplaceSubtree": {"old_value": hostile},
        "InsertSequenceItem": {"item": hostile},
        "DeleteSequenceItem": {"item": hostile},
        "PermuteSequence": {"permutation": (hostile, 0)},
        "TypedPatch": {"limits": hostile},
        "CandidateOccurrence": {"configuration_hash": hostile},
        "VariationParent": {"occurrence": hostile},
        "ParentEdge": {"parent": hostile},
        "PreservationClaim": {"obligation_id": hostile},
        "PreservationObligation": {"relation_id": hostile},
        "VariationCase": {"operator_id": hostile},
        "ComponentTagAssignment": {"component": hostile},
        "PatchRelation": {"semantic_component": hostile},
        "ThreeWayPatchClassification": {"ancestor_hash": hostile},
        "PreservationObligationRequest": {"relation_id": hostile},
        "PatchResolution": {"relation_id": hostile},
        "ParentConfiguration": {"occurrence": hostile},
        "PreservationVerification": {"child_hash": hostile},
    }
    for value in values:
        _assert_encode_rejects(
            _forged_copy(value, **invalid[type(value).__name__])
        )
    assert (
        Hostile.equality_calls,
        Hostile.hash_calls,
        Hostile.bool_calls,
        Hostile.encode_calls,
        Hostile.ordering_calls,
        Hostile.string_calls,
        Hostile.repr_calls,
        Hostile.iteration_calls,
        Hostile.length_calls,
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_forged_insight_ref_class_hook_rejects_directly_and_inside_case():
    class EvilInsightId:
        class_calls = 0

        @property
        def __class__(self):
            type(self).class_calls += 1
            return InsightId

    evil = EvilInsightId()
    forged_reference = object.__new__(InsightRef)
    object.__setattr__(forged_reference, "insight_id", evil)
    object.__setattr__(forged_reference, "version", 1)
    _assert_encode_rejects(forged_reference)

    _, case, _, _, _ = _complete_fixture()
    forged_case = _forged_copy(case, selected_insights=(forged_reference,))
    _assert_encode_rejects(forged_case)
    assert EvilInsightId.class_calls == 0


def test_inherited_and_overriding_subclasses_reject_before_hooks():
    calls = {
        "eq": 0,
        "hash": 0,
        "bool": 0,
        "encode": 0,
        "lt": 0,
        "str": 0,
        "repr": 0,
        "iter": 0,
        "len": 0,
    }

    def eq(self, other):
        calls["eq"] += 1
        return True

    def hash_value(self):
        calls["hash"] += 1
        return 0

    def bool_value(self):
        calls["bool"] += 1
        return True

    def encode(self, *args, **kwargs):
        calls["encode"] += 1
        return b"hostile"

    def lt(self, other):
        calls["lt"] += 1
        return False

    def string(self):
        calls["str"] += 1
        return "hostile"

    def representation(self):
        calls["repr"] += 1
        return "hostile"

    def iteration(self):
        calls["iter"] += 1
        return iter(())

    def length(self):
        calls["len"] += 1
        return 0

    methods = {
        "__slots__": (),
        "__eq__": eq,
        "__hash__": hash_value,
        "__bool__": bool_value,
        "encode": encode,
        "__lt__": lt,
        "__str__": string,
        "__repr__": representation,
        "__iter__": iteration,
        "__len__": length,
    }
    values, _, _, _, _ = _complete_fixture()
    for value in values:
        inheriting_type = type(
            f"CodecInheriting{type(value).__name__}",
            (type(value),),
            {"__slots__": ()},
        )
        overriding_type = type(
            f"CodecOverriding{type(value).__name__}",
            (type(value),),
            methods,
        )
        _assert_encode_rejects(_forged_copy(value, inheriting_type))
        _assert_encode_rejects(_forged_copy(value, overriding_type))
    assert calls == {name: 0 for name in calls}


def test_subclass_metaclass_hash_and_equality_never_run_during_dispatch():
    class HostileMeta(type):
        hash_calls = 0
        equality_calls = 0

        def __hash__(cls):
            type(cls).hash_calls += 1
            return 0

        def __eq__(cls, other):
            type(cls).equality_calls += 1
            return True

    class HostileObjectKey(ObjectKey, metaclass=HostileMeta):
        __slots__ = ()

    forged = object.__new__(HostileObjectKey)
    object.__setattr__(forged, "value", "x")
    HostileMeta.hash_calls = 0
    HostileMeta.equality_calls = 0
    _assert_encode_rejects(forged)
    assert HostileMeta.hash_calls == 0
    assert HostileMeta.equality_calls == 0


def test_codec_limits_and_bytes_subclasses_are_rejected_without_hooks():
    class LimitsSubclass(LineageCodecLimits):
        bool_calls = 0

        def __bool__(self):
            type(self).bool_calls += 1
            return True

    class BytesSubclass(bytes):
        iteration_calls = 0

        def __iter__(self):
            type(self).iteration_calls += 1
            return super().__iter__()

    limits = LimitsSubclass()
    content = BytesSubclass(encode_lineage_value(True))
    with pytest.raises(LineageCodecError):
        encode_lineage_value(True, limits=limits)
    with pytest.raises(LineageCodecError):
        decode_lineage_value(content)
    assert LimitsSubclass.bool_calls == 0
    assert BytesSubclass.iteration_calls == 0


def test_dependency_barrier_is_closed_and_provider_evaluator_incapable():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "agent_evolve"
        / "infrastructure"
        / "lineage_codec.py"
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
        "agent_evolve.integrations",
        "agent_evolve.ports",
        "agent_evolve.session",
        "agent_evolve.infrastructure.artifacts",
        "agent_evolve.infrastructure.events",
        "pydantic",
        "pydantic_ai",
        "pickle",
        "importlib",
        "inspect",
        "requests",
        "httpx",
        "socket",
        "subprocess",
        "pathlib",
        "os",
    )
    assert not any(
        imported == blocked or imported.startswith(f"{blocked}.")
        for imported in imports
        for blocked in forbidden
    )
    forbidden_calls = {"eval", "exec", "compile", "__import__", "getattr", "setattr"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden_calls
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"__new__", "__setattr__"}
        for node in ast.walk(tree)
    )


def test_registry_is_closed_unique_and_contains_no_alias_tags():
    assert len(LINEAGE_WIRE_KINDS) == len(set(LINEAGE_WIRE_KINDS))
    assert all(type(kind) is str and kind == kind.lower() for kind in LINEAGE_WIRE_KINDS)
    assert "replace_scalar" in LINEAGE_WIRE_KINDS
    assert "ReplaceScalar" not in LINEAGE_WIRE_KINDS
    assert "bool" not in LINEAGE_WIRE_KINDS
    assert LINEAGE_CODEC_FORMAT == "agent_evolve.lineage_value"
    assert type(LINEAGE_CODEC_SCHEMA_VERSION) is int
    assert LINEAGE_CODEC_SCHEMA_VERSION == 1
