"""Closed, canonical durable codec for the isolated M4b value graph.

This outward adapter has no event-store, workflow, model, provider, evaluator,
or benchmark capability.  It converts exact, already-valid M4b values to one
versioned canonical JSON representation and reconstructs them exclusively
through their validated constructors and public revalidation boundaries.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Final

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.lineage import (
    MAX_SELECTED_INSIGHTS,
    AbsenceContextKind,
    AbsenceFailureKind,
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
    validate_json_path,
    validate_patch_limits,
    validate_patch_operation,
    validate_typed_patch,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    TypedJsonLimits,
    freeze_json,
    is_frozen_json_value,
    validate_typed_json_limits,
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
    validate_patch_relation,
    validate_preservation_verification,
)


LINEAGE_CODEC_FORMAT: Final = "agent_evolve.lineage_value"
LINEAGE_CODEC_SCHEMA_VERSION: Final = 1

_MAX_CODEC_BYTES: Final = 1_073_741_824
_MAX_CODEC_DEPTH: Final = 512
_MAX_CODEC_NODES: Final = 5_000_000
_MAX_CODEC_CONTAINER_ITEMS: Final = 1_000_000
_MAX_CODEC_STRING_BYTES: Final = 67_108_864
_MAX_CODEC_INTEGER_DIGITS: Final = 4096


class LineageCodecError(ValueError):
    """A bounded, value-free public codec failure."""


class _CodecFailure(Exception):
    """Private control-flow exception whose details never cross the boundary."""


@dataclass(frozen=True, slots=True)
class LineageCodecLimits:
    """Process-independent resource bounds for canonical wire processing."""

    max_bytes: int = 268_435_456
    max_depth: int = 256
    max_nodes: int = 2_000_000
    max_container_items: int = 100_000
    max_string_bytes: int = 16_777_216
    max_integer_digits: int = 4096

    def __post_init__(self) -> None:
        ceilings = (
            ("max_bytes", self.max_bytes, _MAX_CODEC_BYTES),
            ("max_depth", self.max_depth, _MAX_CODEC_DEPTH),
            ("max_nodes", self.max_nodes, _MAX_CODEC_NODES),
            (
                "max_container_items",
                self.max_container_items,
                _MAX_CODEC_CONTAINER_ITEMS,
            ),
            ("max_string_bytes", self.max_string_bytes, _MAX_CODEC_STRING_BYTES),
            (
                "max_integer_digits",
                self.max_integer_digits,
                _MAX_CODEC_INTEGER_DIGITS,
            ),
        )
        for _, value, ceiling in ceilings:
            if type(value) is not int or value <= 0 or value > ceiling:
                raise ValueError("invalid lineage codec limit")


DEFAULT_LINEAGE_CODEC_LIMITS: Final = LineageCodecLimits()


M4B_EXPORTED_VALUE_TYPES: Final = (
    TypedJsonLimits,
    FrozenJsonArray,
    FrozenJsonObject,
    ObjectKey,
    ArrayIndex,
    JsonPath,
    PatchLimits,
    ReplaceScalar,
    ReplaceSubtree,
    InsertSequenceItem,
    DeleteSequenceItem,
    PermuteSequence,
    TypedPatch,
    CandidateOccurrence,
    VariationParent,
    ParentEdge,
    PreservationClaim,
    PreservationObligation,
    VariationCase,
    ComponentTagAssignment,
    PatchRelation,
    ThreeWayPatchClassification,
    PreservationObligationRequest,
    PatchResolution,
    ParentConfiguration,
    PreservationVerification,
)


# This is the complete closed type/tag registry.  Dispatch walks it with
# identity checks rather than hashing an attacker-controlled subclass type: a
# custom metaclass must not execute ``__hash__`` or ``__eq__`` at the boundary.
_EXACT_TYPE_KIND_REGISTRY: Final = (
    (type(None), "json_none"),
    (bool, "json_bool"),
    (int, "json_int"),
    (float, "json_float"),
    (str, "json_string"),
    (CandidateId, "candidate_id"),
    (OperatorInvocationId, "operator_invocation_id"),
    (InsightId, "insight_id"),
    (InsightRef, "insight_ref"),
    (VariationKind, "variation_kind"),
    (ParentRole, "parent_role"),
    (PreservationSource, "preservation_source"),
    (PreservationExpectation, "preservation_expectation"),
    (AbsenceContextKind, "absence_context_kind"),
    (AbsenceFailureKind, "absence_failure_kind"),
    (ThreeWayRelationKind, "three_way_relation_kind"),
    (ResolutionChoice, "resolution_choice"),
    (TypedJsonLimits, "typed_json_limits"),
    (FrozenJsonArray, "frozen_json_array"),
    (FrozenJsonObject, "frozen_json_object"),
    (ObjectKey, "object_key"),
    (ArrayIndex, "array_index"),
    (JsonPath, "json_path"),
    (PatchLimits, "patch_limits"),
    (ReplaceScalar, "replace_scalar"),
    (ReplaceSubtree, "replace_subtree"),
    (InsertSequenceItem, "insert_sequence_item"),
    (DeleteSequenceItem, "delete_sequence_item"),
    (PermuteSequence, "permute_sequence"),
    (TypedPatch, "typed_patch"),
    (CandidateOccurrence, "candidate_occurrence"),
    (VariationParent, "variation_parent"),
    (ParentEdge, "parent_edge"),
    (PreservationClaim, "preservation_claim"),
    (PreservationObligation, "preservation_obligation"),
    (VariationCase, "variation_case"),
    (ComponentTagAssignment, "component_tag_assignment"),
    (PatchRelation, "patch_relation"),
    (ThreeWayPatchClassification, "three_way_patch_classification"),
    (PreservationObligationRequest, "preservation_obligation_request"),
    (PatchResolution, "patch_resolution"),
    (ParentConfiguration, "parent_configuration"),
    (PreservationVerification, "preservation_verification"),
)

LINEAGE_WIRE_KINDS: Final = tuple(
    sorted(kind for _, kind in _EXACT_TYPE_KIND_REGISTRY)
)


_VARIATION_KINDS: Final = (
    VariationKind.REPRODUCTION,
    VariationKind.TYPED_MUTATION,
    VariationKind.TWO_PARENT_CROSSOVER,
    VariationKind.THREE_WAY_RECOMBINATION,
    VariationKind.REPAIR,
)
_PARENT_ROLES: Final = (
    ParentRole.REPRODUCTION_SOURCE,
    ParentRole.MUTATION_PARENT,
    ParentRole.CROSSOVER_LEFT,
    ParentRole.CROSSOVER_RIGHT,
    ParentRole.COMMON_ANCESTOR,
    ParentRole.REPAIR_TARGET,
)
_PRESERVATION_SOURCES: Final = (
    PreservationSource.LEFT_BRANCH,
    PreservationSource.RIGHT_BRANCH,
    PreservationSource.IDENTICAL_NEUTRAL,
)
_PRESERVATION_EXPECTATIONS: Final = (
    PreservationExpectation.PRESENT,
    PreservationExpectation.ABSENT,
)
_ABSENCE_CONTEXT_KINDS: Final = (
    AbsenceContextKind.OBJECT,
    AbsenceContextKind.ARRAY,
)
_ABSENCE_FAILURE_KINDS: Final = (
    AbsenceFailureKind.MISSING_OBJECT_KEY,
    AbsenceFailureKind.ARRAY_INDEX_OUT_OF_BOUNDS,
)
_THREE_WAY_RELATION_KINDS: Final = (
    ThreeWayRelationKind.IDENTICAL,
    ThreeWayRelationKind.DISJOINT,
    ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
    ThreeWayRelationKind.CONFLICT,
    ThreeWayRelationKind.INVALIDATED,
)
_RESOLUTION_CHOICES: Final = (
    ResolutionChoice.CHOOSE_LEFT,
    ResolutionChoice.CHOOSE_RIGHT,
    ResolutionChoice.SYNTHESIZE,
    ResolutionChoice.DROP_BOTH,
)

_VARIATION_KIND_FROM_WIRE: Final = {
    "reproduction": VariationKind.REPRODUCTION,
    "typed_mutation": VariationKind.TYPED_MUTATION,
    "two_parent_crossover": VariationKind.TWO_PARENT_CROSSOVER,
    "three_way_recombination": VariationKind.THREE_WAY_RECOMBINATION,
    "repair": VariationKind.REPAIR,
}
_PARENT_ROLE_FROM_WIRE: Final = {
    "reproduction_source": ParentRole.REPRODUCTION_SOURCE,
    "mutation_parent": ParentRole.MUTATION_PARENT,
    "crossover_left": ParentRole.CROSSOVER_LEFT,
    "crossover_right": ParentRole.CROSSOVER_RIGHT,
    "common_ancestor": ParentRole.COMMON_ANCESTOR,
    "repair_target": ParentRole.REPAIR_TARGET,
}
_PRESERVATION_SOURCE_FROM_WIRE: Final = {
    "left_branch": PreservationSource.LEFT_BRANCH,
    "right_branch": PreservationSource.RIGHT_BRANCH,
    "identical_neutral": PreservationSource.IDENTICAL_NEUTRAL,
}
_PRESERVATION_EXPECTATION_FROM_WIRE: Final = {
    "present": PreservationExpectation.PRESENT,
    "absent": PreservationExpectation.ABSENT,
}
_ABSENCE_CONTEXT_KIND_FROM_WIRE: Final = {
    "object": AbsenceContextKind.OBJECT,
    "array": AbsenceContextKind.ARRAY,
}
_ABSENCE_FAILURE_KIND_FROM_WIRE: Final = {
    "missing_object_key": AbsenceFailureKind.MISSING_OBJECT_KEY,
    "array_index_out_of_bounds": AbsenceFailureKind.ARRAY_INDEX_OUT_OF_BOUNDS,
}
_THREE_WAY_RELATION_KIND_FROM_WIRE: Final = {
    "identical": ThreeWayRelationKind.IDENTICAL,
    "disjoint": ThreeWayRelationKind.DISJOINT,
    "compatible_same_component": ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
    "conflict": ThreeWayRelationKind.CONFLICT,
    "invalidated": ThreeWayRelationKind.INVALIDATED,
}
_RESOLUTION_CHOICE_FROM_WIRE: Final = {
    "choose_left": ResolutionChoice.CHOOSE_LEFT,
    "choose_right": ResolutionChoice.CHOOSE_RIGHT,
    "synthesize": ResolutionChoice.SYNTHESIZE,
    "drop_both": ResolutionChoice.DROP_BOTH,
}

_PATCH_OPERATION_TYPES: Final = (
    ReplaceScalar,
    ReplaceSubtree,
    InsertSequenceItem,
    DeleteSequenceItem,
    PermuteSequence,
)
_TYPED_JSON_TYPES: Final = (
    type(None),
    bool,
    int,
    float,
    str,
    FrozenJsonArray,
    FrozenJsonObject,
)


def _validate_codec_limits(limits: LineageCodecLimits) -> None:
    if type(limits) is not LineageCodecLimits:
        raise _CodecFailure
    LineageCodecLimits.__post_init__(limits)


def _bounded_string(value: object, limits: LineageCodecLimits) -> str:
    if type(value) is not str:
        raise _CodecFailure
    encoded = value.encode("utf-8", errors="strict")
    if len(encoded) > limits.max_string_bytes:
        raise _CodecFailure
    return value


def _bounded_integer(value: object, limits: LineageCodecLimits) -> int:
    if type(value) is not int:
        raise _CodecFailure
    if len(str(abs(value))) > limits.max_integer_digits:
        raise _CodecFailure
    return value


def _bounded_items(value: object, limits: LineageCodecLimits) -> tuple[object, ...]:
    if type(value) is not tuple or len(value) > limits.max_container_items:
        raise _CodecFailure
    return value


def _validate_enum_member(
    value: object,
    enum_type: type[Enum],
    members: tuple[Enum, ...],
) -> Enum:
    if type(value) is not enum_type or not any(value is member for member in members):
        raise _CodecFailure
    return value


def _kind_for_exact_type(value_type: type) -> str:
    for registered_type, kind in _EXACT_TYPE_KIND_REGISTRY:
        if value_type is registered_type:
            return kind
    raise _CodecFailure


def _prevalidate_insight_ref(value: object, limits: LineageCodecLimits) -> InsightRef:
    """Close InsightRef's upstream ``isinstance`` ordering before it can run."""

    if type(value) is not InsightRef:
        raise _CodecFailure
    if type(value.insight_id) is not InsightId or type(value.version) is not int:
        raise _CodecFailure
    InsightId.__post_init__(value.insight_id)
    if value.version <= 0:
        raise _CodecFailure
    _bounded_integer(value.version, limits)
    InsightRef.__post_init__(value)
    return value


def _prevalidate_selected_insights(
    value: VariationCase,
    limits: LineageCodecLimits,
) -> None:
    references = _bounded_items(value.selected_insights, limits)
    if len(references) > MAX_SELECTED_INSIGHTS:
        raise _CodecFailure
    for reference in references:
        _prevalidate_insight_ref(reference, limits)


@dataclass(slots=True)
class _EncodeState:
    limits: LineageCodecLimits
    logical_nodes: int = 0

    def enter(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise _CodecFailure
        self.logical_nodes += 1
        if self.logical_nodes > self.limits.max_nodes:
            raise _CodecFailure


def _record(kind: str, **fields: object) -> dict[str, object]:
    return {"kind": kind, **fields}


def _encode_optional_node(
    value: object | None,
    state: _EncodeState,
    depth: int,
) -> object:
    if value is None:
        return None
    return _encode_node(value, state, depth)


def _encode_nodes(
    values: object,
    state: _EncodeState,
    depth: int,
) -> list[object]:
    items = _bounded_items(values, state.limits)
    return [_encode_node(item, state, depth) for item in items]


def _encode_strings(values: object, limits: LineageCodecLimits) -> list[str]:
    items = _bounded_items(values, limits)
    return [_bounded_string(item, limits) for item in items]


def _encode_enum(
    value: object,
    enum_type: type[Enum],
    members: tuple[Enum, ...],
    kind: str,
    limits: LineageCodecLimits,
) -> dict[str, object]:
    member = _validate_enum_member(value, enum_type, members)
    return _record(kind, value=_bounded_string(member.value, limits))


def _encode_node(value: object, state: _EncodeState, depth: int) -> dict[str, object]:
    state.enter(depth)
    value_type = type(value)
    kind = _kind_for_exact_type(value_type)
    limits = state.limits

    if value is None:
        return _record("json_none")
    if value_type is bool:
        freeze_json(value)
        return _record("json_bool", value=value)
    if value_type is int:
        freeze_json(value)
        integer = _bounded_integer(value, limits)
        return _record("json_int", value=str(integer))
    if value_type is float:
        freeze_json(value)
        if not math.isfinite(value):
            raise _CodecFailure
        return _record("json_float", bits=struct.pack(">d", value).hex())
    if value_type is str:
        freeze_json(value)
        return _record("json_string", value=_bounded_string(value, limits))

    if value_type is CandidateId:
        CandidateId.__post_init__(value)
        return _record(kind, value=_bounded_string(value.value, limits))
    if value_type is OperatorInvocationId:
        OperatorInvocationId.__post_init__(value)
        return _record(kind, value=_bounded_string(value.value, limits))
    if value_type is InsightId:
        InsightId.__post_init__(value)
        return _record(kind, value=_bounded_string(value.value, limits))
    if value_type is InsightRef:
        _prevalidate_insight_ref(value, limits)
        return _record(
            kind,
            insight_id=_encode_node(value.insight_id, state, depth + 1),
            version=_bounded_integer(value.version, limits),
        )

    if value_type is VariationKind:
        return _encode_enum(value, VariationKind, _VARIATION_KINDS, kind, limits)
    if value_type is ParentRole:
        return _encode_enum(value, ParentRole, _PARENT_ROLES, kind, limits)
    if value_type is PreservationSource:
        return _encode_enum(
            value,
            PreservationSource,
            _PRESERVATION_SOURCES,
            kind,
            limits,
        )
    if value_type is PreservationExpectation:
        return _encode_enum(
            value,
            PreservationExpectation,
            _PRESERVATION_EXPECTATIONS,
            kind,
            limits,
        )
    if value_type is AbsenceContextKind:
        return _encode_enum(
            value,
            AbsenceContextKind,
            _ABSENCE_CONTEXT_KINDS,
            kind,
            limits,
        )
    if value_type is AbsenceFailureKind:
        return _encode_enum(
            value,
            AbsenceFailureKind,
            _ABSENCE_FAILURE_KINDS,
            kind,
            limits,
        )
    if value_type is ThreeWayRelationKind:
        return _encode_enum(
            value,
            ThreeWayRelationKind,
            _THREE_WAY_RELATION_KINDS,
            kind,
            limits,
        )
    if value_type is ResolutionChoice:
        return _encode_enum(
            value,
            ResolutionChoice,
            _RESOLUTION_CHOICES,
            kind,
            limits,
        )

    if value_type is TypedJsonLimits:
        validate_typed_json_limits(value)
        return _record(
            kind,
            max_depth=_bounded_integer(value.max_depth, limits),
            max_nodes=_bounded_integer(value.max_nodes, limits),
            max_container_items=_bounded_integer(value.max_container_items, limits),
            max_string_bytes=_bounded_integer(value.max_string_bytes, limits),
            max_integer_bits=_bounded_integer(value.max_integer_bits, limits),
            max_canonical_bytes=_bounded_integer(value.max_canonical_bytes, limits),
        )
    if value_type is FrozenJsonArray:
        if freeze_json(value) is not value:
            raise _CodecFailure
        return _record(kind, items=_encode_nodes(value.items, state, depth + 1))
    if value_type is FrozenJsonObject:
        if freeze_json(value) is not value:
            raise _CodecFailure
        entries = _bounded_items(value.items, limits)
        return _record(
            kind,
            items=[
                {
                    "key": _bounded_string(entry[0], limits),
                    "value": _encode_node(entry[1], state, depth + 1),
                }
                for entry in entries
            ],
        )

    if value_type is ObjectKey:
        ObjectKey.__post_init__(value)
        return _record(kind, value=_bounded_string(value.value, limits))
    if value_type is ArrayIndex:
        ArrayIndex.__post_init__(value)
        return _record(kind, value=_bounded_integer(value.value, limits))
    if value_type is JsonPath:
        validate_json_path(value)
        return _record(kind, segments=_encode_nodes(value.segments, state, depth + 1))
    if value_type is PatchLimits:
        validate_patch_limits(value)
        return _record(
            kind,
            json_limits=_encode_node(value.json_limits, state, depth + 1),
            max_operations=_bounded_integer(value.max_operations, limits),
            max_path_segments=_bounded_integer(value.max_path_segments, limits),
            max_patch_bytes=_bounded_integer(value.max_patch_bytes, limits),
        )

    if value_type is ReplaceScalar:
        validate_patch_operation(value)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            old_value=_encode_node(value.old_value, state, depth + 1),
            new_value=_encode_node(value.new_value, state, depth + 1),
            source_candidate_id=_encode_node(
                value.source_candidate_id, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is ReplaceSubtree:
        validate_patch_operation(value)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            old_value=_encode_node(value.old_value, state, depth + 1),
            new_value=_encode_node(value.new_value, state, depth + 1),
            source_candidate_id=_encode_node(
                value.source_candidate_id, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is InsertSequenceItem:
        validate_patch_operation(value)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            index=_bounded_integer(value.index, limits),
            item=_encode_node(value.item, state, depth + 1),
            before_sequence=_encode_node(value.before_sequence, state, depth + 1),
            after_sequence=_encode_node(value.after_sequence, state, depth + 1),
            source_candidate_id=_encode_node(
                value.source_candidate_id, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is DeleteSequenceItem:
        validate_patch_operation(value)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            index=_bounded_integer(value.index, limits),
            item=_encode_node(value.item, state, depth + 1),
            before_sequence=_encode_node(value.before_sequence, state, depth + 1),
            after_sequence=_encode_node(value.after_sequence, state, depth + 1),
            source_candidate_id=_encode_node(
                value.source_candidate_id, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is PermuteSequence:
        validate_patch_operation(value)
        permutation = _bounded_items(value.permutation, limits)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            permutation=[_bounded_integer(item, limits) for item in permutation],
            before_sequence=_encode_node(value.before_sequence, state, depth + 1),
            after_sequence=_encode_node(value.after_sequence, state, depth + 1),
            source_candidate_id=_encode_node(
                value.source_candidate_id, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is TypedPatch:
        validate_typed_patch(value)
        return _record(
            kind,
            base_candidate_id=_encode_node(value.base_candidate_id, state, depth + 1),
            target_candidate_id=_encode_node(
                value.target_candidate_id, state, depth + 1
            ),
            base_hash=_bounded_string(value.base_hash, limits),
            target_hash=_bounded_string(value.target_hash, limits),
            operations=_encode_nodes(value.operations, state, depth + 1),
            limits=_encode_node(value.limits, state, depth + 1),
            schema_version=_bounded_string(value.schema_version, limits),
        )

    if value_type is CandidateOccurrence:
        CandidateOccurrence.__post_init__(value)
        return _record(
            kind,
            candidate_id=_encode_node(value.candidate_id, state, depth + 1),
            configuration_hash=_bounded_string(value.configuration_hash, limits),
            configuration_artifact_hash=_bounded_string(
                value.configuration_artifact_hash, limits
            ),
            proposal_sequence=_bounded_integer(value.proposal_sequence, limits),
            operator_invocation_id=_encode_optional_node(
                value.operator_invocation_id, state, depth + 1
            ),
        )
    if value_type is VariationParent:
        VariationParent.__post_init__(value)
        return _record(
            kind,
            role=_encode_node(value.role, state, depth + 1),
            occurrence=_encode_node(value.occurrence, state, depth + 1),
        )
    if value_type is ParentEdge:
        ParentEdge.__post_init__(value)
        return _record(
            kind,
            role=_encode_node(value.role, state, depth + 1),
            parent=_encode_node(value.parent, state, depth + 1),
            child=_encode_node(value.child, state, depth + 1),
            patch=_encode_node(value.patch, state, depth + 1),
        )
    if value_type is PreservationClaim:
        PreservationClaim.__post_init__(value)
        return _record(kind, obligation_id=_bounded_string(value.obligation_id, limits))
    if value_type is PreservationObligation:
        PreservationObligation.__post_init__(value)
        return _record(
            kind,
            source=_encode_node(value.source, state, depth + 1),
            source_parent_candidate_ids=_encode_nodes(
                value.source_parent_candidate_ids, state, depth + 1
            ),
            branch_patch_hashes=_encode_strings(value.branch_patch_hashes, limits),
            operation_effect_hashes=_encode_strings(
                value.operation_effect_hashes, limits
            ),
            relation_id=_bounded_string(value.relation_id, limits),
            path=_encode_node(value.path, state, depth + 1),
            expected_state=_encode_node(value.expected_state, state, depth + 1),
            expected_value_hash=(
                None
                if value.expected_value_hash is None
                else _bounded_string(value.expected_value_hash, limits)
            ),
            ancestor_state=_encode_node(value.ancestor_state, state, depth + 1),
            ancestor_value_hash=(
                None
                if value.ancestor_value_hash is None
                else _bounded_string(value.ancestor_value_hash, limits)
            ),
            absence_context_path=_encode_optional_node(
                value.absence_context_path, state, depth + 1
            ),
            absence_context_kind=_encode_optional_node(
                value.absence_context_kind, state, depth + 1
            ),
            absence_context_shape_hash=(
                None
                if value.absence_context_shape_hash is None
                else _bounded_string(value.absence_context_shape_hash, limits)
            ),
            absence_failure_kind=_encode_optional_node(
                value.absence_failure_kind, state, depth + 1
            ),
        )
    if value_type is VariationCase:
        _prevalidate_selected_insights(value, limits)
        validate_variation_case(value)
        return _record(
            kind,
            operator_invocation_id=_encode_node(
                value.operator_invocation_id, state, depth + 1
            ),
            variation_kind=_encode_node(value.variation_kind, state, depth + 1),
            operator_id=_bounded_string(value.operator_id, limits),
            operator_version=_bounded_integer(value.operator_version, limits),
            parents=_encode_nodes(value.parents, state, depth + 1),
            requested_child_count=_bounded_integer(
                value.requested_child_count, limits
            ),
            context_stratum_hash=_bounded_string(value.context_stratum_hash, limits),
            reward_definition_hash=_bounded_string(
                value.reward_definition_hash, limits
            ),
            common_ancestor=_encode_optional_node(
                value.common_ancestor, state, depth + 1
            ),
            ancestor_to_parent_patches=_encode_nodes(
                value.ancestor_to_parent_patches, state, depth + 1
            ),
            selected_insights=_encode_nodes(
                value.selected_insights, state, depth + 1
            ),
            preservation_obligations=_encode_nodes(
                value.preservation_obligations, state, depth + 1
            ),
        )

    if value_type is ComponentTagAssignment:
        ComponentTagAssignment.__post_init__(value)
        return _record(
            kind,
            path=_encode_node(value.path, state, depth + 1),
            component=_bounded_string(value.component, limits),
        )
    if value_type is PatchRelation:
        validate_patch_relation(value)
        return _record(
            kind,
            relation_kind=_encode_node(value.kind, state, depth + 1),
            left_operations=_encode_nodes(
                value.left_operations, state, depth + 1
            ),
            right_operations=_encode_nodes(
                value.right_operations, state, depth + 1
            ),
            semantic_component=(
                None
                if value.semantic_component is None
                else _bounded_string(value.semantic_component, limits)
            ),
        )
    if value_type is ThreeWayPatchClassification:
        value.revalidate()
        return _record(
            kind,
            ancestor=_encode_node(value.ancestor, state, depth + 1),
            ancestor_candidate_id=_encode_node(
                value.ancestor_candidate_id, state, depth + 1
            ),
            ancestor_hash=_bounded_string(value.ancestor_hash, limits),
            left_patch_hash=_bounded_string(value.left_patch_hash, limits),
            right_patch_hash=_bounded_string(value.right_patch_hash, limits),
            relations=_encode_nodes(value.relations, state, depth + 1),
            left_patch=_encode_node(value.left_patch, state, depth + 1),
            right_patch=_encode_node(value.right_patch, state, depth + 1),
        )
    if value_type is PreservationObligationRequest:
        PreservationObligationRequest.__post_init__(value)
        return _record(
            kind,
            relation_id=_bounded_string(value.relation_id, limits),
            source=_encode_node(value.source, state, depth + 1),
            path=_encode_node(value.path, state, depth + 1),
        )
    if value_type is PatchResolution:
        PatchResolution.__post_init__(value)
        return _record(
            kind,
            relation_id=_bounded_string(value.relation_id, limits),
            choice=_encode_node(value.choice, state, depth + 1),
            synthesized_result_hash=(
                None
                if value.synthesized_result_hash is None
                else _bounded_string(value.synthesized_result_hash, limits)
            ),
        )
    if value_type is ParentConfiguration:
        ParentConfiguration.__post_init__(value)
        return _record(
            kind,
            occurrence=_encode_node(value.occurrence, state, depth + 1),
            configuration=_encode_node(value.configuration, state, depth + 1),
        )
    if value_type is PreservationVerification:
        validate_preservation_verification(value)
        return _record(
            kind,
            child_hash=_bounded_string(value.child_hash, limits),
            verified_claims=_encode_nodes(
                value.verified_claims, state, depth + 1
            ),
            discriminatively_used_parent_ids=_encode_nodes(
                value.discriminatively_used_parent_ids, state, depth + 1
            ),
        )

    raise _CodecFailure


def _validate_wire_tree(value: object, limits: LineageCodecLimits) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        if depth > limits.max_depth:
            raise _CodecFailure
        nodes += 1
        if nodes > limits.max_nodes:
            raise _CodecFailure
        current_type = type(current)
        if current is None or current_type is bool:
            continue
        if current_type is int:
            _bounded_integer(current, limits)
            continue
        if current_type is str:
            _bounded_string(current, limits)
            continue
        if current_type is list:
            if len(current) > limits.max_container_items:
                raise _CodecFailure
            stack.extend((item, depth + 1) for item in reversed(current))
            continue
        if current_type is dict:
            if len(current) > limits.max_container_items:
                raise _CodecFailure
            for key, item in current.items():
                _bounded_string(key, limits)
                stack.append((item, depth + 1))
            continue
        raise _CodecFailure


def _canonical_json_bytes(value: object, limits: LineageCodecLimits) -> bytes:
    _validate_wire_tree(value, limits)
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    chunks: list[bytes] = []
    size = 0
    for text in encoder.iterencode(value):
        if type(text) is not str:
            raise _CodecFailure
        chunk = text.encode("utf-8", errors="strict")
        size += len(chunk)
        if size > limits.max_bytes:
            raise _CodecFailure
        chunks.append(chunk)
    return b"".join(chunks)


def _scan_json_structure(content: bytes, limits: LineageCodecLimits) -> None:
    depth = 0
    in_string = False
    escaped = False
    expect_value = True
    nodes = 0
    containers: list[list[object]] = []
    for byte in content:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte in (0x20, 0x09, 0x0A, 0x0D):
            continue
        if expect_value and byte not in (0x5D, 0x7D):
            nodes += 1
            if nodes > limits.max_nodes:
                raise _CodecFailure
            if containers and containers[-1][0] == "array":
                containers[-1][1] += 1
                if containers[-1][1] > limits.max_container_items:
                    raise _CodecFailure
            expect_value = False
        if byte == 0x22:
            in_string = True
        elif byte == 0x5B:
            depth += 1
            if depth > limits.max_depth:
                raise _CodecFailure
            containers.append(["array", 0])
            expect_value = True
        elif byte == 0x7B:
            depth += 1
            if depth > limits.max_depth:
                raise _CodecFailure
            containers.append(["object", 0])
        elif byte == 0x5D:
            if not containers or containers[-1][0] != "array":
                raise _CodecFailure
            containers.pop()
            depth -= 1
            if depth < 0:
                raise _CodecFailure
            expect_value = False
        elif byte == 0x7D:
            if not containers or containers[-1][0] != "object":
                raise _CodecFailure
            containers.pop()
            depth -= 1
            if depth < 0:
                raise _CodecFailure
            expect_value = False
        elif byte == 0x3A:
            if not containers or containers[-1][0] != "object":
                raise _CodecFailure
            containers[-1][1] += 1
            if containers[-1][1] > limits.max_container_items:
                raise _CodecFailure
            expect_value = True
        elif byte == 0x2C and containers and containers[-1][0] == "array":
            expect_value = True
    if in_string or escaped or depth != 0 or containers or nodes == 0:
        raise _CodecFailure


def _decode_json_bytes(content: object, limits: LineageCodecLimits) -> object:
    if type(content) is not bytes or not content or len(content) > limits.max_bytes:
        raise _CodecFailure
    if content.startswith(b"\xef\xbb\xbf"):
        raise _CodecFailure
    _scan_json_structure(content, limits)
    text = content.decode("utf-8", errors="strict")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _CodecFailure
            result[key] = value
        return result

    def parse_integer(token: str) -> int:
        if type(token) is not str or len(token.lstrip("-")) > limits.max_integer_digits:
            raise _CodecFailure
        return int(token)

    def reject_number(_: str) -> object:
        raise _CodecFailure

    parsed = json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_int=parse_integer,
        parse_float=reject_number,
        parse_constant=reject_number,
    )
    _validate_wire_tree(parsed, limits)
    if _canonical_json_bytes(parsed, limits) != content:
        raise _CodecFailure
    return parsed


def _expect_record(
    value: object,
    kind: str,
    fields: tuple[str, ...],
) -> dict[str, object]:
    if type(value) is not dict:
        raise _CodecFailure
    expected = {"kind", *fields}
    if set(value) != expected or value.get("kind") != kind:
        raise _CodecFailure
    return value


def _decode_string(value: object, limits: LineageCodecLimits) -> str:
    return _bounded_string(value, limits)


def _decode_optional_string(value: object, limits: LineageCodecLimits) -> str | None:
    if value is None:
        return None
    return _bounded_string(value, limits)


def _decode_integer(value: object, limits: LineageCodecLimits) -> int:
    return _bounded_integer(value, limits)


def _decode_list(value: object, limits: LineageCodecLimits) -> list[object]:
    if type(value) is not list or len(value) > limits.max_container_items:
        raise _CodecFailure
    return value


def _decode_as(
    value: object,
    expected_type: type,
    limits: LineageCodecLimits,
    depth: int,
) -> object:
    decoded = _decode_node(value, limits, depth)
    if type(decoded) is not expected_type:
        raise _CodecFailure
    return decoded


def _decode_optional_as(
    value: object,
    expected_type: type,
    limits: LineageCodecLimits,
    depth: int,
) -> object | None:
    if value is None:
        return None
    return _decode_as(value, expected_type, limits, depth)


def _decode_typed_json(
    value: object,
    limits: LineageCodecLimits,
    depth: int,
) -> object:
    decoded = _decode_node(value, limits, depth)
    if type(decoded) not in _TYPED_JSON_TYPES or not is_frozen_json_value(decoded):
        raise _CodecFailure
    freeze_json(decoded)
    return decoded


def _decode_operation(
    value: object,
    limits: LineageCodecLimits,
    depth: int,
) -> object:
    decoded = _decode_node(value, limits, depth)
    if type(decoded) not in _PATCH_OPERATION_TYPES:
        raise _CodecFailure
    validate_patch_operation(decoded)
    return decoded


def _decode_nodes(
    value: object,
    limits: LineageCodecLimits,
    depth: int,
) -> tuple[object, ...]:
    return tuple(_decode_node(item, limits, depth) for item in _decode_list(value, limits))


def _decode_enum(
    value: object,
    kind: str,
    mapping: dict[str, Enum],
    limits: LineageCodecLimits,
) -> Enum:
    record = _expect_record(value, kind, ("value",))
    token = _decode_string(record["value"], limits)
    member = mapping.get(token)
    if member is None:
        raise _CodecFailure
    return member


def _decode_node(
    value: object,
    limits: LineageCodecLimits,
    depth: int,
) -> object:
    if depth > limits.max_depth or type(value) is not dict:
        raise _CodecFailure
    kind = value.get("kind")
    if type(kind) is not str or kind not in LINEAGE_WIRE_KINDS:
        raise _CodecFailure

    if kind == "json_none":
        _expect_record(value, kind, ())
        return None
    if kind == "json_bool":
        record = _expect_record(value, kind, ("value",))
        if type(record["value"]) is not bool:
            raise _CodecFailure
        return freeze_json(record["value"])
    if kind == "json_int":
        record = _expect_record(value, kind, ("value",))
        token = _decode_string(record["value"], limits)
        if token == "0":
            integer = 0
        else:
            negative = token.startswith("-")
            digits = token[1:] if negative else token
            if not digits or digits[0] == "0" or not digits.isascii() or not digits.isdigit():
                raise _CodecFailure
            if len(digits) > limits.max_integer_digits:
                raise _CodecFailure
            integer = int(token)
        return freeze_json(integer)
    if kind == "json_float":
        record = _expect_record(value, kind, ("bits",))
        bits = _decode_string(record["bits"], limits)
        if len(bits) != 16 or any(character not in "0123456789abcdef" for character in bits):
            raise _CodecFailure
        number = struct.unpack(">d", bytes.fromhex(bits))[0]
        if not math.isfinite(number):
            raise _CodecFailure
        return freeze_json(number)
    if kind == "json_string":
        record = _expect_record(value, kind, ("value",))
        return freeze_json(_decode_string(record["value"], limits))

    if kind == "candidate_id":
        record = _expect_record(value, kind, ("value",))
        return CandidateId(_decode_string(record["value"], limits))
    if kind == "operator_invocation_id":
        record = _expect_record(value, kind, ("value",))
        return OperatorInvocationId(_decode_string(record["value"], limits))
    if kind == "insight_id":
        record = _expect_record(value, kind, ("value",))
        return InsightId(_decode_string(record["value"], limits))
    if kind == "insight_ref":
        record = _expect_record(value, kind, ("insight_id", "version"))
        return InsightRef(
            _decode_as(record["insight_id"], InsightId, limits, depth + 1),
            _decode_integer(record["version"], limits),
        )

    if kind == "variation_kind":
        return _decode_enum(value, kind, _VARIATION_KIND_FROM_WIRE, limits)
    if kind == "parent_role":
        return _decode_enum(value, kind, _PARENT_ROLE_FROM_WIRE, limits)
    if kind == "preservation_source":
        return _decode_enum(value, kind, _PRESERVATION_SOURCE_FROM_WIRE, limits)
    if kind == "preservation_expectation":
        return _decode_enum(
            value,
            kind,
            _PRESERVATION_EXPECTATION_FROM_WIRE,
            limits,
        )
    if kind == "absence_context_kind":
        return _decode_enum(value, kind, _ABSENCE_CONTEXT_KIND_FROM_WIRE, limits)
    if kind == "absence_failure_kind":
        return _decode_enum(value, kind, _ABSENCE_FAILURE_KIND_FROM_WIRE, limits)
    if kind == "three_way_relation_kind":
        return _decode_enum(
            value,
            kind,
            _THREE_WAY_RELATION_KIND_FROM_WIRE,
            limits,
        )
    if kind == "resolution_choice":
        return _decode_enum(value, kind, _RESOLUTION_CHOICE_FROM_WIRE, limits)

    if kind == "typed_json_limits":
        record = _expect_record(
            value,
            kind,
            (
                "max_depth",
                "max_nodes",
                "max_container_items",
                "max_string_bytes",
                "max_integer_bits",
                "max_canonical_bytes",
            ),
        )
        return TypedJsonLimits(
            max_depth=_decode_integer(record["max_depth"], limits),
            max_nodes=_decode_integer(record["max_nodes"], limits),
            max_container_items=_decode_integer(
                record["max_container_items"], limits
            ),
            max_string_bytes=_decode_integer(record["max_string_bytes"], limits),
            max_integer_bits=_decode_integer(record["max_integer_bits"], limits),
            max_canonical_bytes=_decode_integer(
                record["max_canonical_bytes"], limits
            ),
        )
    if kind == "frozen_json_array":
        record = _expect_record(value, kind, ("items",))
        items = tuple(
            _decode_typed_json(item, limits, depth + 1)
            for item in _decode_list(record["items"], limits)
        )
        return FrozenJsonArray(items)
    if kind == "frozen_json_object":
        record = _expect_record(value, kind, ("items",))
        entries: list[tuple[str, object]] = []
        for raw_entry in _decode_list(record["items"], limits):
            if type(raw_entry) is not dict or set(raw_entry) != {"key", "value"}:
                raise _CodecFailure
            entries.append(
                (
                    _decode_string(raw_entry["key"], limits),
                    _decode_typed_json(raw_entry["value"], limits, depth + 1),
                )
            )
        return FrozenJsonObject(tuple(entries))

    if kind == "object_key":
        record = _expect_record(value, kind, ("value",))
        return ObjectKey(_decode_string(record["value"], limits))
    if kind == "array_index":
        record = _expect_record(value, kind, ("value",))
        return ArrayIndex(_decode_integer(record["value"], limits))
    if kind == "json_path":
        record = _expect_record(value, kind, ("segments",))
        segments = tuple(
            _decode_node(item, limits, depth + 1)
            for item in _decode_list(record["segments"], limits)
        )
        if any(type(item) not in (ObjectKey, ArrayIndex) for item in segments):
            raise _CodecFailure
        return JsonPath(segments)
    if kind == "patch_limits":
        record = _expect_record(
            value,
            kind,
            (
                "json_limits",
                "max_operations",
                "max_path_segments",
                "max_patch_bytes",
            ),
        )
        return PatchLimits(
            json_limits=_decode_as(
                record["json_limits"], TypedJsonLimits, limits, depth + 1
            ),
            max_operations=_decode_integer(record["max_operations"], limits),
            max_path_segments=_decode_integer(
                record["max_path_segments"], limits
            ),
            max_patch_bytes=_decode_integer(record["max_patch_bytes"], limits),
        )

    if kind in ("replace_scalar", "replace_subtree"):
        record = _expect_record(
            value,
            kind,
            (
                "path",
                "old_value",
                "new_value",
                "source_candidate_id",
                "semantic_component",
            ),
        )
        arguments = (
            _decode_as(record["path"], JsonPath, limits, depth + 1),
            _decode_typed_json(record["old_value"], limits, depth + 1),
            _decode_typed_json(record["new_value"], limits, depth + 1),
            _decode_as(
                record["source_candidate_id"], CandidateId, limits, depth + 1
            ),
            _decode_optional_string(record["semantic_component"], limits),
        )
        return (
            ReplaceScalar(*arguments)
            if kind == "replace_scalar"
            else ReplaceSubtree(*arguments)
        )
    if kind in ("insert_sequence_item", "delete_sequence_item"):
        record = _expect_record(
            value,
            kind,
            (
                "path",
                "index",
                "item",
                "before_sequence",
                "after_sequence",
                "source_candidate_id",
                "semantic_component",
            ),
        )
        arguments = (
            _decode_as(record["path"], JsonPath, limits, depth + 1),
            _decode_integer(record["index"], limits),
            _decode_typed_json(record["item"], limits, depth + 1),
            _decode_as(
                record["before_sequence"], FrozenJsonArray, limits, depth + 1
            ),
            _decode_as(
                record["after_sequence"], FrozenJsonArray, limits, depth + 1
            ),
            _decode_as(
                record["source_candidate_id"], CandidateId, limits, depth + 1
            ),
            _decode_optional_string(record["semantic_component"], limits),
        )
        return (
            InsertSequenceItem(*arguments)
            if kind == "insert_sequence_item"
            else DeleteSequenceItem(*arguments)
        )
    if kind == "permute_sequence":
        record = _expect_record(
            value,
            kind,
            (
                "path",
                "permutation",
                "before_sequence",
                "after_sequence",
                "source_candidate_id",
                "semantic_component",
            ),
        )
        permutation = tuple(
            _decode_integer(item, limits)
            for item in _decode_list(record["permutation"], limits)
        )
        return PermuteSequence(
            _decode_as(record["path"], JsonPath, limits, depth + 1),
            permutation,
            _decode_as(
                record["before_sequence"], FrozenJsonArray, limits, depth + 1
            ),
            _decode_as(
                record["after_sequence"], FrozenJsonArray, limits, depth + 1
            ),
            _decode_as(
                record["source_candidate_id"], CandidateId, limits, depth + 1
            ),
            _decode_optional_string(record["semantic_component"], limits),
        )
    if kind == "typed_patch":
        record = _expect_record(
            value,
            kind,
            (
                "base_candidate_id",
                "target_candidate_id",
                "base_hash",
                "target_hash",
                "operations",
                "limits",
                "schema_version",
            ),
        )
        operations = tuple(
            _decode_operation(item, limits, depth + 1)
            for item in _decode_list(record["operations"], limits)
        )
        return TypedPatch(
            base_candidate_id=_decode_as(
                record["base_candidate_id"], CandidateId, limits, depth + 1
            ),
            target_candidate_id=_decode_as(
                record["target_candidate_id"], CandidateId, limits, depth + 1
            ),
            base_hash=_decode_string(record["base_hash"], limits),
            target_hash=_decode_string(record["target_hash"], limits),
            operations=operations,
            limits=_decode_as(record["limits"], PatchLimits, limits, depth + 1),
            schema_version=_decode_string(record["schema_version"], limits),
        )

    if kind == "candidate_occurrence":
        record = _expect_record(
            value,
            kind,
            (
                "candidate_id",
                "configuration_hash",
                "configuration_artifact_hash",
                "proposal_sequence",
                "operator_invocation_id",
            ),
        )
        return CandidateOccurrence(
            candidate_id=_decode_as(
                record["candidate_id"], CandidateId, limits, depth + 1
            ),
            configuration_hash=_decode_string(record["configuration_hash"], limits),
            configuration_artifact_hash=_decode_string(
                record["configuration_artifact_hash"], limits
            ),
            proposal_sequence=_decode_integer(record["proposal_sequence"], limits),
            operator_invocation_id=_decode_optional_as(
                record["operator_invocation_id"],
                OperatorInvocationId,
                limits,
                depth + 1,
            ),
        )
    if kind == "variation_parent":
        record = _expect_record(value, kind, ("role", "occurrence"))
        return VariationParent(
            _decode_as(record["role"], ParentRole, limits, depth + 1),
            _decode_as(
                record["occurrence"], CandidateOccurrence, limits, depth + 1
            ),
        )
    if kind == "parent_edge":
        record = _expect_record(value, kind, ("role", "parent", "child", "patch"))
        return ParentEdge(
            _decode_as(record["role"], ParentRole, limits, depth + 1),
            _decode_as(record["parent"], CandidateOccurrence, limits, depth + 1),
            _decode_as(record["child"], CandidateOccurrence, limits, depth + 1),
            _decode_as(record["patch"], TypedPatch, limits, depth + 1),
        )
    if kind == "preservation_claim":
        record = _expect_record(value, kind, ("obligation_id",))
        return PreservationClaim(_decode_string(record["obligation_id"], limits))
    if kind == "preservation_obligation":
        record = _expect_record(
            value,
            kind,
            (
                "source",
                "source_parent_candidate_ids",
                "branch_patch_hashes",
                "operation_effect_hashes",
                "relation_id",
                "path",
                "expected_state",
                "expected_value_hash",
                "ancestor_state",
                "ancestor_value_hash",
                "absence_context_path",
                "absence_context_kind",
                "absence_context_shape_hash",
                "absence_failure_kind",
            ),
        )
        source_ids = tuple(
            _decode_as(item, CandidateId, limits, depth + 1)
            for item in _decode_list(record["source_parent_candidate_ids"], limits)
        )
        branch_hashes = tuple(
            _decode_string(item, limits)
            for item in _decode_list(record["branch_patch_hashes"], limits)
        )
        effect_hashes = tuple(
            _decode_string(item, limits)
            for item in _decode_list(record["operation_effect_hashes"], limits)
        )
        return PreservationObligation(
            source=_decode_as(
                record["source"], PreservationSource, limits, depth + 1
            ),
            source_parent_candidate_ids=source_ids,
            branch_patch_hashes=branch_hashes,
            operation_effect_hashes=effect_hashes,
            relation_id=_decode_string(record["relation_id"], limits),
            path=_decode_as(record["path"], JsonPath, limits, depth + 1),
            expected_state=_decode_as(
                record["expected_state"],
                PreservationExpectation,
                limits,
                depth + 1,
            ),
            expected_value_hash=_decode_optional_string(
                record["expected_value_hash"], limits
            ),
            ancestor_state=_decode_as(
                record["ancestor_state"],
                PreservationExpectation,
                limits,
                depth + 1,
            ),
            ancestor_value_hash=_decode_optional_string(
                record["ancestor_value_hash"], limits
            ),
            absence_context_path=_decode_optional_as(
                record["absence_context_path"], JsonPath, limits, depth + 1
            ),
            absence_context_kind=_decode_optional_as(
                record["absence_context_kind"],
                AbsenceContextKind,
                limits,
                depth + 1,
            ),
            absence_context_shape_hash=_decode_optional_string(
                record["absence_context_shape_hash"], limits
            ),
            absence_failure_kind=_decode_optional_as(
                record["absence_failure_kind"],
                AbsenceFailureKind,
                limits,
                depth + 1,
            ),
        )
    if kind == "variation_case":
        record = _expect_record(
            value,
            kind,
            (
                "operator_invocation_id",
                "variation_kind",
                "operator_id",
                "operator_version",
                "parents",
                "requested_child_count",
                "context_stratum_hash",
                "reward_definition_hash",
                "common_ancestor",
                "ancestor_to_parent_patches",
                "selected_insights",
                "preservation_obligations",
            ),
        )
        parents = tuple(
            _decode_as(item, VariationParent, limits, depth + 1)
            for item in _decode_list(record["parents"], limits)
        )
        patches = tuple(
            _decode_as(item, TypedPatch, limits, depth + 1)
            for item in _decode_list(record["ancestor_to_parent_patches"], limits)
        )
        insights = tuple(
            _decode_as(item, InsightRef, limits, depth + 1)
            for item in _decode_list(record["selected_insights"], limits)
        )
        obligations = tuple(
            _decode_as(item, PreservationObligation, limits, depth + 1)
            for item in _decode_list(record["preservation_obligations"], limits)
        )
        result = VariationCase(
            operator_invocation_id=_decode_as(
                record["operator_invocation_id"],
                OperatorInvocationId,
                limits,
                depth + 1,
            ),
            variation_kind=_decode_as(
                record["variation_kind"], VariationKind, limits, depth + 1
            ),
            operator_id=_decode_string(record["operator_id"], limits),
            operator_version=_decode_integer(record["operator_version"], limits),
            parents=parents,
            requested_child_count=_decode_integer(
                record["requested_child_count"], limits
            ),
            context_stratum_hash=_decode_string(
                record["context_stratum_hash"], limits
            ),
            reward_definition_hash=_decode_string(
                record["reward_definition_hash"], limits
            ),
            common_ancestor=_decode_optional_as(
                record["common_ancestor"],
                CandidateOccurrence,
                limits,
                depth + 1,
            ),
            ancestor_to_parent_patches=patches,
            selected_insights=insights,
            preservation_obligations=obligations,
        )
        validate_variation_case(result)
        return result

    if kind == "component_tag_assignment":
        record = _expect_record(value, kind, ("path", "component"))
        return ComponentTagAssignment(
            _decode_as(record["path"], JsonPath, limits, depth + 1),
            _decode_string(record["component"], limits),
        )
    if kind == "patch_relation":
        record = _expect_record(
            value,
            kind,
            (
                "relation_kind",
                "left_operations",
                "right_operations",
                "semantic_component",
            ),
        )
        left = tuple(
            _decode_operation(item, limits, depth + 1)
            for item in _decode_list(record["left_operations"], limits)
        )
        right = tuple(
            _decode_operation(item, limits, depth + 1)
            for item in _decode_list(record["right_operations"], limits)
        )
        return PatchRelation(
            _decode_as(
                record["relation_kind"], ThreeWayRelationKind, limits, depth + 1
            ),
            left,
            right,
            _decode_optional_string(record["semantic_component"], limits),
        )
    if kind == "three_way_patch_classification":
        record = _expect_record(
            value,
            kind,
            (
                "ancestor",
                "ancestor_candidate_id",
                "ancestor_hash",
                "left_patch_hash",
                "right_patch_hash",
                "relations",
                "left_patch",
                "right_patch",
            ),
        )
        relations = tuple(
            _decode_as(item, PatchRelation, limits, depth + 1)
            for item in _decode_list(record["relations"], limits)
        )
        result = ThreeWayPatchClassification(
            ancestor=_decode_typed_json(record["ancestor"], limits, depth + 1),
            ancestor_candidate_id=_decode_as(
                record["ancestor_candidate_id"], CandidateId, limits, depth + 1
            ),
            ancestor_hash=_decode_string(record["ancestor_hash"], limits),
            left_patch_hash=_decode_string(record["left_patch_hash"], limits),
            right_patch_hash=_decode_string(record["right_patch_hash"], limits),
            relations=relations,
            left_patch=_decode_as(
                record["left_patch"], TypedPatch, limits, depth + 1
            ),
            right_patch=_decode_as(
                record["right_patch"], TypedPatch, limits, depth + 1
            ),
        )
        result.revalidate()
        return result
    if kind == "preservation_obligation_request":
        record = _expect_record(value, kind, ("relation_id", "source", "path"))
        return PreservationObligationRequest(
            _decode_string(record["relation_id"], limits),
            _decode_as(
                record["source"], PreservationSource, limits, depth + 1
            ),
            _decode_as(record["path"], JsonPath, limits, depth + 1),
        )
    if kind == "patch_resolution":
        record = _expect_record(
            value,
            kind,
            ("relation_id", "choice", "synthesized_result_hash"),
        )
        return PatchResolution(
            _decode_string(record["relation_id"], limits),
            _decode_as(record["choice"], ResolutionChoice, limits, depth + 1),
            _decode_optional_string(record["synthesized_result_hash"], limits),
        )
    if kind == "parent_configuration":
        record = _expect_record(value, kind, ("occurrence", "configuration"))
        return ParentConfiguration(
            _decode_as(
                record["occurrence"], CandidateOccurrence, limits, depth + 1
            ),
            _decode_typed_json(record["configuration"], limits, depth + 1),
        )
    if kind == "preservation_verification":
        record = _expect_record(
            value,
            kind,
            (
                "child_hash",
                "verified_claims",
                "discriminatively_used_parent_ids",
            ),
        )
        claims = tuple(
            _decode_as(item, PreservationClaim, limits, depth + 1)
            for item in _decode_list(record["verified_claims"], limits)
        )
        parent_ids = tuple(
            _decode_as(item, CandidateId, limits, depth + 1)
            for item in _decode_list(
                record["discriminatively_used_parent_ids"], limits
            )
        )
        result = PreservationVerification(
            _decode_string(record["child_hash"], limits),
            claims,
            parent_ids,
        )
        validate_preservation_verification(result)
        return result

    raise _CodecFailure


def _encode_internal(value: object, limits: LineageCodecLimits) -> bytes:
    _validate_codec_limits(limits)
    state = _EncodeState(limits)
    node = _encode_node(value, state, 1)
    envelope = {
        "format": LINEAGE_CODEC_FORMAT,
        "schema_version": LINEAGE_CODEC_SCHEMA_VERSION,
        "value": node,
    }
    return _canonical_json_bytes(envelope, limits)


def _decode_internal(content: object, limits: LineageCodecLimits) -> object:
    _validate_codec_limits(limits)
    parsed = _decode_json_bytes(content, limits)
    if type(parsed) is not dict or set(parsed) != {
        "format",
        "schema_version",
        "value",
    }:
        raise _CodecFailure
    if parsed["format"] != LINEAGE_CODEC_FORMAT:
        raise _CodecFailure
    if (
        type(parsed["schema_version"]) is not int
        or parsed["schema_version"] != LINEAGE_CODEC_SCHEMA_VERSION
    ):
        raise _CodecFailure
    result = _decode_node(parsed["value"], limits, 1)
    if _encode_internal(result, limits) != content:
        raise _CodecFailure
    return result


def encode_lineage_value(
    value: object,
    *,
    limits: LineageCodecLimits = DEFAULT_LINEAGE_CODEC_LIMITS,
) -> bytes:
    """Encode one exact registered value to canonical version-1 JSON bytes."""

    try:
        return _encode_internal(value, limits)
    except Exception:
        pass
    raise LineageCodecError("lineage encode rejected value")


def decode_lineage_value(
    content: bytes,
    *,
    limits: LineageCodecLimits = DEFAULT_LINEAGE_CODEC_LIMITS,
) -> object:
    """Decode canonical version-1 JSON through validated public constructors."""

    try:
        return _decode_internal(content, limits)
    except Exception:
        pass
    raise LineageCodecError("lineage decode rejected bytes")


__all__ = [
    "DEFAULT_LINEAGE_CODEC_LIMITS",
    "LINEAGE_CODEC_FORMAT",
    "LINEAGE_CODEC_SCHEMA_VERSION",
    "LINEAGE_WIRE_KINDS",
    "M4B_EXPORTED_VALUE_TYPES",
    "LineageCodecError",
    "LineageCodecLimits",
    "decode_lineage_value",
    "encode_lineage_value",
]
