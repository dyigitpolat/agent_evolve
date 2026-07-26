"""Deterministic model-free union of strictly disjoint typed patches.

The three-way patch algebra deliberately classifies rather than merges.  This
module provides the narrower policy that is safe to automate: two branches may
be united only when replay classifies every branch effect as either identical
or disjoint.  Conflicts, prefix invalidation, and semantic-component coupling
remain explicit failures for a higher-authority operator to resolve.

The policy depends only on immutable domain values and the typed-patch policy.
It has no generator, evaluator, archive, or application dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import (
    DEFAULT_PATCH_LIMITS,
    DeleteSequenceItem,
    InsertSequenceItem,
    JsonPath,
    PatchLimits,
    PatchOperation,
    PermuteSequence,
    ReplaceScalar,
    ReplaceSubtree,
    TypedPatch,
    canonical_path_bytes,
    operation_effect_bytes,
    operation_effect_sha256,
    operation_occurrence_bytes,
    operation_sort_key,
    require_sha256,
    validate_json_path,
    validate_patch_limits,
    validate_patch_operation,
    validate_typed_patch,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    freeze_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.variation.typed_patch import (
    ComponentTagAssignment,
    ThreeWayPatchClassification,
    ThreeWayRelationKind,
    apply_patch,
    classify_three_way_patches,
    derive_patch,
    replace_existing_path,
    value_at_path,
)


POLICY_ID = "disjoint_patch_union"
POLICY_VERSION = 1
_RECEIPT_DOMAIN = b"agent-evolve:disjoint-patch-union-receipt:v1\x00"


class DisjointPatchRecombinationError(ValueError):
    """The requested branches do not admit the policy's exact safe union."""


class RecombinationBranch(str, Enum):
    """A branch whose replay-verified effect appears in the union."""

    LEFT = "left"
    RIGHT = "right"


_CANONICAL_BOTH_BRANCHES = (
    RecombinationBranch.LEFT,
    RecombinationBranch.RIGHT,
)


def _frame(value: bytes) -> bytes:
    if type(value) is not bytes:
        raise TypeError("receipt components must be exact bytes")
    return len(value).to_bytes(8, "big", signed=False) + value


def _validate_candidate_id(value: CandidateId, *, name: str) -> None:
    if type(value) is not CandidateId:
        raise TypeError(f"{name} must be an exact CandidateId")
    CandidateId.__post_init__(value)


def _validate_sources(values: tuple[RecombinationBranch, ...]) -> None:
    if type(values) is not tuple or any(
        type(value) is not RecombinationBranch for value in values
    ):
        raise TypeError("sources must be an exact tuple of RecombinationBranch values")
    if values not in {
        (RecombinationBranch.LEFT,),
        (RecombinationBranch.RIGHT,),
        _CANONICAL_BOTH_BRANCHES,
    }:
        raise ValueError("sources must be a non-empty canonical branch set")


@dataclass(frozen=True, slots=True, eq=False)
class SystemPatchAttribution:
    """System-derived provenance for one exact union-patch effect.

    ``sources`` contains both branches only for a replay-classified identical
    relation.  The relation and effect digests make this evidence independently
    bindable without relying on model-authored path strings or explanations.
    """

    path: JsonPath
    sources: tuple[RecombinationBranch, ...]
    relation_id: str
    effect_sha256: str

    def __post_init__(self) -> None:
        validate_json_path(self.path)
        _validate_sources(self.sources)
        require_sha256(self.relation_id, "relation_id")
        require_sha256(self.effect_sha256, "effect_sha256")

    def revalidate(self) -> None:
        if type(self) is not SystemPatchAttribution:
            raise TypeError("attribution must be an exact SystemPatchAttribution")
        SystemPatchAttribution.__post_init__(self)

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not SystemPatchAttribution
            or type(other) is not SystemPatchAttribution
        ):
            return False
        self.revalidate()
        other.revalidate()
        return _attribution_fingerprint(self) == _attribution_fingerprint(other)

    __hash__ = None


def _attribution_fingerprint(
    value: SystemPatchAttribution,
) -> tuple[bytes, tuple[str, ...], str, str]:
    value.revalidate()
    return (
        canonical_path_bytes(value.path),
        tuple(source.value for source in value.sources),
        value.relation_id,
        value.effect_sha256,
    )


def _attribution_sort_key(
    value: SystemPatchAttribution,
) -> tuple[bytes, str, tuple[str, ...], str]:
    value.revalidate()
    return (
        canonical_path_bytes(value.path),
        value.effect_sha256,
        tuple(source.value for source in value.sources),
        value.relation_id,
    )


def _operation_before_after(
    operation: PatchOperation,
) -> tuple[FrozenJsonValue, FrozenJsonValue]:
    validate_patch_operation(operation)
    if type(operation) in (ReplaceScalar, ReplaceSubtree):
        return operation.old_value, operation.new_value
    if type(operation) in (
        InsertSequenceItem,
        DeleteSequenceItem,
        PermuteSequence,
    ):
        return operation.before_sequence, operation.after_sequence
    raise TypeError("unsupported patch operation type")


def _safe_relations(
    classification: ThreeWayPatchClassification,
) -> tuple[PatchOperation, ...]:
    """Return the canonical union operations or reject the classification."""

    if type(classification) is not ThreeWayPatchClassification:
        raise TypeError("classification must be an exact ThreeWayPatchClassification")
    classification.revalidate()
    rejected = tuple(
        relation.kind
        for relation in classification.relations
        if relation.kind
        not in {ThreeWayRelationKind.IDENTICAL, ThreeWayRelationKind.DISJOINT}
    )
    if rejected:
        kinds = ", ".join(sorted({kind.value for kind in rejected}))
        raise DisjointPatchRecombinationError(
            f"branch classification contains unsafe relation kinds: {kinds}"
        )

    left_unique = False
    right_unique = False
    operations: list[PatchOperation] = []
    for relation in classification.relations:
        if relation.kind is ThreeWayRelationKind.IDENTICAL:
            # Relation validation proves these are the exact same effect.  The
            # operation occurrence also has the same common-ancestor source, so
            # choosing the left representation is deterministic and lossless.
            operations.append(relation.left_operations[0])
        elif relation.kind is ThreeWayRelationKind.DISJOINT:
            if relation.left_operations:
                left_unique = True
                operations.append(relation.left_operations[0])
            else:
                right_unique = True
                operations.append(relation.right_operations[0])
        else:  # pragma: no cover - rejected above and enum closes the cases.
            raise AssertionError("unsafe relation passed the fail-closed gate")
    if not left_unique or not right_unique:
        missing = "left" if not left_unique else "right"
        if not left_unique and not right_unique:
            missing = "left and right"
        raise DisjointPatchRecombinationError(
            f"{missing} branch lacks a non-identical disjoint effect"
        )
    return tuple(sorted(operations, key=operation_sort_key))


def _expected_attribution(
    classification: ThreeWayPatchClassification,
) -> tuple[SystemPatchAttribution, ...]:
    classification.revalidate()
    values: list[SystemPatchAttribution] = []
    for relation in classification.relations:
        if relation.kind is ThreeWayRelationKind.IDENTICAL:
            operation = relation.left_operations[0]
            sources = _CANONICAL_BOTH_BRANCHES
        elif relation.kind is ThreeWayRelationKind.DISJOINT:
            if relation.left_operations:
                operation = relation.left_operations[0]
                sources = (RecombinationBranch.LEFT,)
            else:
                operation = relation.right_operations[0]
                sources = (RecombinationBranch.RIGHT,)
        else:
            raise DisjointPatchRecombinationError(
                "system attribution cannot represent an unsafe relation"
            )
        values.append(
            SystemPatchAttribution(
                path=operation.path,
                sources=sources,
                relation_id=relation.relation_id,
                effect_sha256=operation_effect_sha256(
                    operation,
                    limits=classification.left_patch.limits.json_limits,
                ),
            )
        )
    return tuple(sorted(values, key=_attribution_sort_key))


def _construct_target(
    ancestor: FrozenJsonValue,
    operations: tuple[PatchOperation, ...],
    *,
    limits: PatchLimits,
) -> FrozenJsonValue:
    """Construct an endpoint using only exact old/new operation states."""

    validate_patch_limits(limits)
    frozen_ancestor = freeze_json(ancestor, limits=limits.json_limits)
    if frozen_ancestor is not ancestor:
        raise TypeError("ancestor must already be frozen typed JSON")
    current = frozen_ancestor
    for operation in operations:
        validate_patch_operation(operation)
        before, after = _operation_before_after(operation)
        observed = value_at_path(current, operation.path)
        if not typed_json_equal(observed, before, limits=limits.json_limits):
            raise DisjointPatchRecombinationError(
                "canonical union encountered a stale operation precondition"
            )
        current = replace_existing_path(current, operation.path, after)
        if not typed_json_equal(
            value_at_path(current, operation.path),
            after,
            limits=limits.json_limits,
        ):
            raise DisjointPatchRecombinationError(
                "canonical union did not install an exact operation effect"
            )
    return current


def _effect_fingerprint(
    patch: TypedPatch,
) -> tuple[bytes, ...]:
    validate_typed_patch(patch)
    return tuple(
        operation_effect_bytes(operation, limits=patch.limits.json_limits)
        for operation in patch.operations
    )


def _validate_materialization(
    value: "DisjointPatchMaterialization",
) -> None:
    if type(value.union_patch) is not TypedPatch:
        raise TypeError("union_patch must be an exact TypedPatch")
    if type(value.classification) is not ThreeWayPatchClassification:
        raise TypeError("classification must be an exact ThreeWayPatchClassification")
    validate_typed_patch(value.union_patch)
    value.classification.revalidate()
    configuration = freeze_json(
        value.configuration,
        limits=value.union_patch.limits.json_limits,
    )
    if configuration is not value.configuration:
        raise TypeError("configuration must already be frozen typed JSON")
    classification = value.classification
    union_patch = value.union_patch
    if union_patch.limits != classification.left_patch.limits:
        raise ValueError("union patch and branch patches must share exact limits")
    if (
        union_patch.base_candidate_id != classification.ancestor_candidate_id
        or union_patch.base_hash != classification.ancestor_hash
    ):
        raise ValueError("union patch does not bind the classified ancestor")
    if union_patch.target_candidate_id in {
        classification.ancestor_candidate_id,
        classification.left_patch.target_candidate_id,
        classification.right_patch.target_candidate_id,
    }:
        raise ValueError("union target must be a distinct candidate occurrence")
    expected_operations = _safe_relations(classification)
    if tuple(
        operation_occurrence_bytes(
            operation,
            limits=union_patch.limits.json_limits,
        )
        for operation in union_patch.operations
    ) != tuple(
        operation_occurrence_bytes(
            operation,
            limits=union_patch.limits.json_limits,
        )
        for operation in expected_operations
    ):
        raise ValueError(
            "union patch does not contain the canonical classified effects"
        )
    replayed = apply_patch(classification.ancestor, union_patch)
    if not typed_json_equal(
        replayed,
        configuration,
        limits=union_patch.limits.json_limits,
    ):
        raise ValueError("configuration does not equal the replayed union patch")

    rediff = derive_patch(
        classification.ancestor,
        configuration,
        base_candidate_id=classification.ancestor_candidate_id,
        target_candidate_id=union_patch.target_candidate_id,
        limits=union_patch.limits,
    )
    if _effect_fingerprint(rediff) != _effect_fingerprint(union_patch):
        raise ValueError(
            "union endpoint does not re-diff to the exact classified effects"
        )
    if rediff.patch_hash != union_patch.patch_hash:
        raise ValueError("union endpoint does not re-diff to the canonical union patch")

    if type(value.system_attribution) is not tuple or any(
        type(item) is not SystemPatchAttribution for item in value.system_attribution
    ):
        raise TypeError(
            "system_attribution must be an exact tuple of SystemPatchAttribution values"
        )
    for item in value.system_attribution:
        item.revalidate()
    if value.system_attribution != tuple(
        sorted(value.system_attribution, key=_attribution_sort_key)
    ):
        raise ValueError("system_attribution must use canonical order")
    expected_attribution = _expected_attribution(classification)
    if tuple(map(_attribution_fingerprint, value.system_attribution)) != tuple(
        map(_attribution_fingerprint, expected_attribution)
    ):
        raise ValueError(
            "system_attribution does not exactly bind every classified union effect"
        )


@dataclass(frozen=True, slots=True, eq=False)
class DisjointPatchMaterialization:
    """Immutable, replay-complete output of deterministic recombination."""

    configuration: FrozenJsonValue
    union_patch: TypedPatch
    classification: ThreeWayPatchClassification
    system_attribution: tuple[SystemPatchAttribution, ...]

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.union_patch) is not TypedPatch:
            raise TypeError("union_patch must be an exact TypedPatch")
        frozen = freeze_json(
            self.configuration,
            limits=self.union_patch.limits.json_limits,
        )
        object.__setattr__(self, "configuration", frozen)
        _validate_materialization(self)

    def revalidate(self) -> None:
        if type(self) is not DisjointPatchMaterialization:
            raise TypeError(
                "materialization must be an exact DisjointPatchMaterialization"
            )
        _validate_materialization(self)

    @property
    def receipt_sha256(self) -> str:
        """Digest binding the policy, endpoints, effects, and provenance."""

        self.revalidate()
        digest = hashlib.sha256()
        digest.update(_RECEIPT_DOMAIN)
        digest.update(_frame(self.policy_id.encode("ascii", errors="strict")))
        digest.update(self.policy_version.to_bytes(8, "big", signed=False))
        for candidate_id in (
            self.classification.ancestor_candidate_id,
            self.classification.left_patch.target_candidate_id,
            self.classification.right_patch.target_candidate_id,
            self.union_patch.target_candidate_id,
        ):
            digest.update(_frame(candidate_id.value.encode("ascii", errors="strict")))
        digest.update(bytes.fromhex(self.classification.left_patch_hash))
        digest.update(bytes.fromhex(self.classification.right_patch_hash))
        digest.update(bytes.fromhex(self.union_patch.patch_hash))
        digest.update(
            bytes.fromhex(
                typed_json_sha256(
                    self.configuration,
                    limits=self.union_patch.limits.json_limits,
                )
            )
        )
        digest.update(len(self.classification.relations).to_bytes(8, "big"))
        for relation in self.classification.relations:
            digest.update(bytes.fromhex(relation.relation_id))
        digest.update(len(self.system_attribution).to_bytes(8, "big"))
        for attribution in self.system_attribution:
            digest.update(_frame(canonical_path_bytes(attribution.path)))
            digest.update(len(attribution.sources).to_bytes(8, "big"))
            for source in attribution.sources:
                digest.update(_frame(source.value.encode("ascii", errors="strict")))
            digest.update(bytes.fromhex(attribution.relation_id))
            digest.update(bytes.fromhex(attribution.effect_sha256))
        return digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not DisjointPatchMaterialization
            or type(other) is not DisjointPatchMaterialization
        ):
            return False
        return self.receipt_sha256 == other.receipt_sha256

    __hash__ = None


@dataclass(frozen=True, slots=True)
class DisjointPatchRecombiner:
    """Materialize the exact union of two safe ancestor-relative branches."""

    limits: PatchLimits = DEFAULT_PATCH_LIMITS

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        validate_patch_limits(self.limits)

    def materialize(
        self,
        *,
        ancestor: object,
        ancestor_candidate_id: CandidateId,
        left: object,
        left_candidate_id: CandidateId,
        right: object,
        right_candidate_id: CandidateId,
        target_candidate_id: CandidateId,
        left_component_tags: tuple[ComponentTagAssignment, ...] = (),
        right_component_tags: tuple[ComponentTagAssignment, ...] = (),
    ) -> DisjointPatchMaterialization:
        """Derive, classify, construct, replay, and independently re-diff.

        All four occurrence IDs must be distinct.  Component tags are optional
        benchmark semantics; when they induce a same-component relation, this
        mechanical policy intentionally refuses to decide whether combination
        is safe.
        """

        validate_patch_limits(self.limits)
        identifiers = (
            (ancestor_candidate_id, "ancestor_candidate_id"),
            (left_candidate_id, "left_candidate_id"),
            (right_candidate_id, "right_candidate_id"),
            (target_candidate_id, "target_candidate_id"),
        )
        for candidate_id, name in identifiers:
            _validate_candidate_id(candidate_id, name=name)
        if len({candidate_id for candidate_id, _ in identifiers}) != len(identifiers):
            raise ValueError("recombination occurrence IDs must be pairwise distinct")

        frozen_ancestor = freeze_json(ancestor, limits=self.limits.json_limits)
        frozen_left = freeze_json(left, limits=self.limits.json_limits)
        frozen_right = freeze_json(right, limits=self.limits.json_limits)
        left_patch = derive_patch(
            frozen_ancestor,
            frozen_left,
            base_candidate_id=ancestor_candidate_id,
            target_candidate_id=left_candidate_id,
            component_tags=left_component_tags,
            limits=self.limits,
        )
        right_patch = derive_patch(
            frozen_ancestor,
            frozen_right,
            base_candidate_id=ancestor_candidate_id,
            target_candidate_id=right_candidate_id,
            component_tags=right_component_tags,
            limits=self.limits,
        )
        classification = classify_three_way_patches(
            frozen_ancestor,
            left_patch,
            right_patch,
        )
        union_operations = _safe_relations(classification)
        provisional_target = _construct_target(
            frozen_ancestor,
            union_operations,
            limits=self.limits,
        )
        union_patch = TypedPatch(
            base_candidate_id=ancestor_candidate_id,
            target_candidate_id=target_candidate_id,
            base_hash=classification.ancestor_hash,
            target_hash=typed_json_sha256(
                provisional_target,
                limits=self.limits.json_limits,
            ),
            operations=union_operations,
            limits=self.limits,
        )

        # Replay is an independent whole-patch check after the local, safe
        # construction above.  Re-diff then proves the endpoint did not merely
        # happen to match while hiding a different canonical effect script.
        replayed = apply_patch(frozen_ancestor, union_patch)
        if not typed_json_equal(
            replayed,
            provisional_target,
            limits=self.limits.json_limits,
        ):
            raise DisjointPatchRecombinationError(
                "union patch replay disagrees with safe construction"
            )
        rediff = derive_patch(
            frozen_ancestor,
            replayed,
            base_candidate_id=ancestor_candidate_id,
            target_candidate_id=target_candidate_id,
            limits=self.limits,
        )
        if _effect_fingerprint(rediff) != _effect_fingerprint(union_patch):
            raise DisjointPatchRecombinationError(
                "replayed union does not re-diff to the exact classified effects"
            )
        if rediff.patch_hash != union_patch.patch_hash:
            raise DisjointPatchRecombinationError(
                "replayed union does not re-diff to the canonical union patch"
            )

        return DisjointPatchMaterialization(
            configuration=replayed,
            union_patch=union_patch,
            classification=classification,
            system_attribution=_expected_attribution(classification),
        )


__all__ = [
    "DisjointPatchMaterialization",
    "DisjointPatchRecombinationError",
    "DisjointPatchRecombiner",
    "POLICY_ID",
    "POLICY_VERSION",
    "RecombinationBranch",
    "SystemPatchAttribution",
]
