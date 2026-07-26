"""Framework-free occurrence, parentage, and variation-case contracts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    TypedPatch,
    canonical_path_bytes,
    require_sha256,
    validate_json_path,
    validate_typed_patch,
)


_OPERATOR_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
MAX_REQUESTED_CHILDREN = 1024
MAX_SELECTED_INSIGHTS = 1024
MAX_PRESERVATION_OBLIGATIONS = 4096
_PRESERVATION_OBLIGATION_DOMAIN = b"agent-evolve:preservation-obligation:v1\x00"


class VariationKind(str, Enum):
    REPRODUCTION = "reproduction"
    TYPED_MUTATION = "typed_mutation"
    TWO_PARENT_CROSSOVER = "two_parent_crossover"
    THREE_WAY_RECOMBINATION = "three_way_recombination"
    REPAIR = "repair"


class ParentRole(str, Enum):
    REPRODUCTION_SOURCE = "reproduction_source"
    MUTATION_PARENT = "mutation_parent"
    CROSSOVER_LEFT = "crossover_left"
    CROSSOVER_RIGHT = "crossover_right"
    COMMON_ANCESTOR = "common_ancestor"
    REPAIR_TARGET = "repair_target"


class PreservationSource(str, Enum):
    """Exact branch provenance of one factory-derived preservation obligation."""

    LEFT_BRANCH = "left_branch"
    RIGHT_BRANCH = "right_branch"
    IDENTICAL_NEUTRAL = "identical_neutral"


class PreservationExpectation(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class AbsenceContextKind(str, Enum):
    OBJECT = "object"
    ARRAY = "array"


class AbsenceFailureKind(str, Enum):
    MISSING_OBJECT_KEY = "missing_object_key"
    ARRAY_INDEX_OUT_OF_BOUNDS = "array_index_out_of_bounds"


def _validate_candidate_id(value: CandidateId, name: str) -> None:
    if type(value) is not CandidateId:
        raise TypeError(f"{name} must be an exact CandidateId")
    CandidateId.__post_init__(value)


def _validate_operator_invocation_id(
    value: OperatorInvocationId,
    name: str,
) -> None:
    if type(value) is not OperatorInvocationId:
        raise TypeError(f"{name} must be an exact OperatorInvocationId")
    OperatorInvocationId.__post_init__(value)


@dataclass(frozen=True, slots=True, eq=False)
class CandidateOccurrence:
    """One proposal occurrence, distinct even when candidate content repeats."""

    candidate_id: CandidateId
    configuration_hash: str
    configuration_artifact_hash: str
    proposal_sequence: int
    operator_invocation_id: OperatorInvocationId | None = None

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id, "candidate_id")
        require_sha256(self.configuration_hash, "configuration_hash")
        require_sha256(
            self.configuration_artifact_hash,
            "configuration_artifact_hash",
        )
        if type(self.proposal_sequence) is not int or self.proposal_sequence < 0:
            raise ValueError("proposal_sequence must be a non-negative exact integer")
        if self.operator_invocation_id is not None and type(
            self.operator_invocation_id
        ) is not OperatorInvocationId:
            raise TypeError(
                "operator_invocation_id must be an exact OperatorInvocationId or None"
            )
        if self.operator_invocation_id is not None:
            _validate_operator_invocation_id(
                self.operator_invocation_id,
                "operator_invocation_id",
            )

    def _validated_values(self) -> tuple[str, str, str, int, str | None]:
        if type(self) is not CandidateOccurrence:
            raise TypeError("occurrence must be an exact CandidateOccurrence")
        CandidateOccurrence.__post_init__(self)
        return (
            self.candidate_id.value,
            self.configuration_hash,
            self.configuration_artifact_hash,
            self.proposal_sequence,
            None
            if self.operator_invocation_id is None
            else self.operator_invocation_id.value,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not CandidateOccurrence or type(other) is not CandidateOccurrence:
            return False
        return self._validated_values() == other._validated_values()

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class VariationParent:
    role: ParentRole
    occurrence: CandidateOccurrence

    def __post_init__(self) -> None:
        if type(self.role) is not ParentRole:
            raise TypeError("role must be a ParentRole")
        if type(self.occurrence) is not CandidateOccurrence:
            raise TypeError("occurrence must be an exact CandidateOccurrence")
        CandidateOccurrence.__post_init__(self.occurrence)
        if self.role is ParentRole.COMMON_ANCESTOR:
            raise ValueError("common ancestors use VariationCase.common_ancestor")

    def _validated_values(self) -> tuple[str, tuple[str, str, str, int, str | None]]:
        if type(self) is not VariationParent:
            raise TypeError("parent must be an exact VariationParent")
        VariationParent.__post_init__(self)
        return self.role.value, self.occurrence._validated_values()

    def __eq__(self, other: object) -> bool:
        if type(self) is not VariationParent or type(other) is not VariationParent:
            return False
        return self._validated_values() == other._validated_values()

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ParentEdge:
    """A verified occurrence edge carrying its exact parent-to-child patch."""

    role: ParentRole
    parent: CandidateOccurrence
    child: CandidateOccurrence
    patch: TypedPatch

    def __post_init__(self) -> None:
        if type(self.role) is not ParentRole:
            raise TypeError("role must be a ParentRole")
        if self.role is ParentRole.COMMON_ANCESTOR:
            raise ValueError("COMMON_ANCESTOR is not a direct parent edge role")
        if type(self.parent) is not CandidateOccurrence or type(
            self.child
        ) is not CandidateOccurrence:
            raise TypeError("parent and child must be exact CandidateOccurrence values")
        CandidateOccurrence.__post_init__(self.parent)
        CandidateOccurrence.__post_init__(self.child)
        if self.parent.candidate_id == self.child.candidate_id:
            raise ValueError("a lineage edge cannot be self-referential")
        if self.parent.proposal_sequence >= self.child.proposal_sequence:
            raise ValueError("a lineage edge must point to a later proposal sequence")
        validate_typed_patch(self.patch)
        if (
            self.patch.base_candidate_id != self.parent.candidate_id
            or self.patch.target_candidate_id != self.child.candidate_id
            or self.patch.base_hash != self.parent.configuration_hash
            or self.patch.target_hash != self.child.configuration_hash
        ):
            raise ValueError("parent edge endpoints do not match its exact patch")

    def __eq__(self, other: object) -> bool:
        if type(self) is not ParentEdge or type(other) is not ParentEdge:
            return False
        ParentEdge.__post_init__(self)
        ParentEdge.__post_init__(other)
        return (
            self.role.value,
            self.parent._validated_values(),
            self.child._validated_values(),
            self.patch.patch_hash,
        ) == (
            other.role.value,
            other.parent._validated_values(),
            other.child._validated_values(),
            other.patch.patch_hash,
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PreservationClaim:
    """An output claim referencing one immutable predeclared obligation."""

    obligation_id: str

    def __post_init__(self) -> None:
        require_sha256(self.obligation_id, "obligation_id")

    def __eq__(self, other: object) -> bool:
        if type(self) is not PreservationClaim or type(other) is not PreservationClaim:
            return False
        PreservationClaim.__post_init__(self)
        PreservationClaim.__post_init__(other)
        return self.obligation_id == other.obligation_id

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PreservationObligation:
    """A replay-derived exact branch effect that a child is asked to preserve.

    The identity binds the source branch patch, operation effect, three-way
    relation, path, and both ancestor and expected presence-aware states.  A
    branch-specific obligation therefore cannot describe unchanged ancestor
    content.  Identical branch effects are represented once as neutral evidence
    and never count as use of either individual parent.
    """

    source: PreservationSource
    source_parent_candidate_ids: tuple[CandidateId, ...]
    branch_patch_hashes: tuple[str, ...]
    operation_effect_hashes: tuple[str, ...]
    relation_id: str
    path: JsonPath
    expected_state: PreservationExpectation
    expected_value_hash: str | None
    ancestor_state: PreservationExpectation
    ancestor_value_hash: str | None
    absence_context_path: JsonPath | None = None
    absence_context_kind: AbsenceContextKind | None = None
    absence_context_shape_hash: str | None = None
    absence_failure_kind: AbsenceFailureKind | None = None

    def __post_init__(self) -> None:
        if type(self.source) is not PreservationSource:
            raise TypeError("source must be a PreservationSource")
        if type(self.source_parent_candidate_ids) is not tuple:
            raise TypeError("source_parent_candidate_ids must be an exact CandidateId tuple")
        required_sources = (
            2 if self.source is PreservationSource.IDENTICAL_NEUTRAL else 1
        )
        if len(self.source_parent_candidate_ids) != required_sources:
            raise ValueError("preservation source has the wrong number of parent identities")
        if any(
            type(value) is not CandidateId
            for value in self.source_parent_candidate_ids
        ):
            raise TypeError("source_parent_candidate_ids must be an exact CandidateId tuple")
        for value in self.source_parent_candidate_ids:
            _validate_candidate_id(value, "source_parent_candidate_id")
        if len(set(self.source_parent_candidate_ids)) != len(
            self.source_parent_candidate_ids
        ):
            raise ValueError("preservation source parent identities must be distinct")
        for name, values in (
            ("branch_patch_hashes", self.branch_patch_hashes),
            ("operation_effect_hashes", self.operation_effect_hashes),
        ):
            if type(values) is not tuple or len(values) != required_sources:
                raise TypeError(f"{name} must be an exact tuple of source-aligned hashes")
            for value in values:
                require_sha256(value, name)
        require_sha256(self.relation_id, "relation_id")
        validate_json_path(self.path)
        self._validate_state(
            self.expected_state,
            self.expected_value_hash,
            name="expected",
        )
        self._validate_state(
            self.ancestor_state,
            self.ancestor_value_hash,
            name="ancestor",
        )
        if (
            self.expected_state is self.ancestor_state
            and self.expected_value_hash == self.ancestor_value_hash
        ):
            raise ValueError(
                "a preservation obligation must bind an actual change from the ancestor"
            )
        absence_values = (
            self.absence_context_path,
            self.absence_context_kind,
            self.absence_context_shape_hash,
            self.absence_failure_kind,
        )
        if self.expected_state is PreservationExpectation.PRESENT:
            if any(value is not None for value in absence_values):
                raise ValueError(
                    "present preservation state cannot carry an absence-context receipt"
                )
        else:
            if any(value is None for value in absence_values):
                raise ValueError(
                    "absent preservation state requires a complete context receipt"
                )
            validate_json_path(self.absence_context_path)  # type: ignore[arg-type]
            if not self.absence_context_path.is_prefix_of(self.path) or len(
                self.absence_context_path.segments
            ) >= len(self.path.segments):
                raise ValueError(
                    "absence context must be a strict prefix of the obligation path"
                )
            if type(self.absence_context_kind) is not AbsenceContextKind:
                raise TypeError("absence_context_kind must be an AbsenceContextKind")
            if type(self.absence_failure_kind) is not AbsenceFailureKind:
                raise TypeError("absence_failure_kind must be an AbsenceFailureKind")
            require_sha256(
                self.absence_context_shape_hash,  # type: ignore[arg-type]
                "absence_context_shape_hash",
            )
            failed_segment = self.path.segments[
                len(self.absence_context_path.segments)
            ]
            expected_pair = (
                AbsenceContextKind.OBJECT,
                AbsenceFailureKind.MISSING_OBJECT_KEY,
                ObjectKey,
            )
            if self.absence_context_kind is AbsenceContextKind.ARRAY:
                expected_pair = (
                    AbsenceContextKind.ARRAY,
                    AbsenceFailureKind.ARRAY_INDEX_OUT_OF_BOUNDS,
                    ArrayIndex,
                )
            if (
                self.absence_context_kind is not expected_pair[0]
                or self.absence_failure_kind is not expected_pair[1]
                or type(failed_segment) is not expected_pair[2]
            ):
                raise ValueError(
                    "absence receipt kind does not match its failing path segment"
                )

    @staticmethod
    def _validate_state(
        state: PreservationExpectation,
        value_hash: str | None,
        *,
        name: str,
    ) -> None:
        if type(state) is not PreservationExpectation:
            raise TypeError(f"{name}_state must be a PreservationExpectation")
        if state is PreservationExpectation.PRESENT:
            if value_hash is None:
                raise ValueError(f"{name} present state requires a value hash")
            require_sha256(value_hash, f"{name}_value_hash")
        elif value_hash is not None:
            raise ValueError(f"{name} absent state cannot carry a value hash")

    @property
    def obligation_id(self) -> str:
        if type(self) is not PreservationObligation:
            raise TypeError("obligation must be an exact PreservationObligation")
        PreservationObligation.__post_init__(self)
        digest = hashlib.sha256()
        digest.update(_PRESERVATION_OBLIGATION_DOMAIN)
        digest.update(self.source.value.encode("ascii", errors="strict"))
        digest.update(len(self.source_parent_candidate_ids).to_bytes(8, "big"))
        for candidate_id in self.source_parent_candidate_ids:
            encoded = candidate_id.value.encode("ascii", errors="strict")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        for values in (self.branch_patch_hashes, self.operation_effect_hashes):
            digest.update(len(values).to_bytes(8, "big"))
            for value in values:
                digest.update(bytes.fromhex(value))
        digest.update(bytes.fromhex(self.relation_id))
        path_bytes = canonical_path_bytes(self.path)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        for state, value_hash in (
            (self.expected_state, self.expected_value_hash),
            (self.ancestor_state, self.ancestor_value_hash),
        ):
            digest.update(state.value.encode("ascii", errors="strict"))
            digest.update(b"\x00" if value_hash is None else b"\x01")
            if value_hash is not None:
                digest.update(bytes.fromhex(value_hash))
        if self.absence_context_path is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            context_path = canonical_path_bytes(self.absence_context_path)
            digest.update(len(context_path).to_bytes(8, "big"))
            digest.update(context_path)
            digest.update(self.absence_context_kind.value.encode("ascii"))
            digest.update(bytes.fromhex(self.absence_context_shape_hash))
            digest.update(self.absence_failure_kind.value.encode("ascii"))
        return digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not PreservationObligation
            or type(other) is not PreservationObligation
        ):
            return False
        PreservationObligation.__post_init__(self)
        PreservationObligation.__post_init__(other)
        return self.obligation_id == other.obligation_id

    __hash__ = None


def _validate_obligation_path_overlaps(
    obligations: tuple[PreservationObligation, ...],
) -> None:
    """Reject parent-local equal/prefix paths in linear total path length."""

    terminal = object()
    roots: dict[CandidateId, dict[object, object]] = {}
    for obligation in obligations:
        for parent_id in obligation.source_parent_candidate_ids:
            node = roots.setdefault(parent_id, {})
            for segment in obligation.path.segments:
                if terminal in node:
                    raise ValueError(
                        "preservation obligation paths cannot overlap within one parent"
                    )
                child = node.setdefault(segment, {})
                if type(child) is not dict:  # pragma: no cover - local trie invariant.
                    raise AssertionError("invalid preservation-path trie")
                node = child
            if terminal in node or node:
                raise ValueError(
                    "preservation obligation paths cannot overlap within one parent"
                )
            node[terminal] = True


@dataclass(frozen=True, slots=True, eq=False)
class VariationCase:
    """Frozen causal inputs for one isolated variation-policy invocation.

    This first slice deliberately contains no prompt, provider, evaluator, or
    event object.  Those identities remain a later application-layer concern.
    """

    operator_invocation_id: OperatorInvocationId
    variation_kind: VariationKind
    operator_id: str
    operator_version: int
    parents: tuple[VariationParent, ...]
    requested_child_count: int
    context_stratum_hash: str
    reward_definition_hash: str
    common_ancestor: CandidateOccurrence | None = None
    ancestor_to_parent_patches: tuple[TypedPatch, ...] = ()
    selected_insights: tuple[InsightRef, ...] = ()
    preservation_obligations: tuple[PreservationObligation, ...] = ()

    def __post_init__(self) -> None:
        _validate_operator_invocation_id(
            self.operator_invocation_id,
            "operator_invocation_id",
        )
        if type(self.variation_kind) is not VariationKind:
            raise TypeError("variation_kind must be a VariationKind")
        if type(self.operator_id) is not str or _OPERATOR_TOKEN.fullmatch(
            self.operator_id
        ) is None:
            raise ValueError("operator_id must use the closed lowercase token grammar")
        if type(self.operator_version) is not int or self.operator_version <= 0:
            raise ValueError("operator_version must be a positive exact integer")
        if type(self.parents) is not tuple:
            raise TypeError("parents must be an exact tuple of VariationParent values")
        # Every closed variation kind consumes either one or two parents.  Bound
        # the container before inspecting its elements so an oversized hostile
        # tuple cannot force an unbounded validation scan.
        if not 1 <= len(self.parents) <= 2:
            raise ValueError("parents must contain one or two variation parents")
        if any(type(parent) is not VariationParent for parent in self.parents):
            raise TypeError("parents must be an exact tuple of VariationParent values")
        for parent in self.parents:
            VariationParent.__post_init__(parent)
        if type(self.requested_child_count) is not int:
            raise TypeError("requested_child_count must be an exact integer")
        if not 1 <= self.requested_child_count <= MAX_REQUESTED_CHILDREN:
            raise ValueError(
                f"requested_child_count must lie in [1, {MAX_REQUESTED_CHILDREN}]"
            )
        require_sha256(self.context_stratum_hash, "context_stratum_hash")
        require_sha256(self.reward_definition_hash, "reward_definition_hash")

        parent_ids = tuple(parent.occurrence.candidate_id for parent in self.parents)
        roles = tuple(parent.role for parent in self.parents)
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("variation parents must be distinct occurrences")
        if len(set(roles)) != len(roles):
            raise ValueError("variation parent roles must be unique")
        expected_roles = {
            VariationKind.REPRODUCTION: (ParentRole.REPRODUCTION_SOURCE,),
            VariationKind.TYPED_MUTATION: (ParentRole.MUTATION_PARENT,),
            VariationKind.TWO_PARENT_CROSSOVER: (
                ParentRole.CROSSOVER_LEFT,
                ParentRole.CROSSOVER_RIGHT,
            ),
            VariationKind.THREE_WAY_RECOMBINATION: (
                ParentRole.CROSSOVER_LEFT,
                ParentRole.CROSSOVER_RIGHT,
            ),
            VariationKind.REPAIR: (ParentRole.REPAIR_TARGET,),
        }[self.variation_kind]
        if roles != expected_roles:
            raise ValueError(
                "parents must appear in the exact role order required by variation_kind"
            )
        if any(
            parent.occurrence.operator_invocation_id == self.operator_invocation_id
            for parent in self.parents
        ):
            raise ValueError(
                "an operator invocation cannot consume an occurrence it produced itself"
            )

        if type(self.ancestor_to_parent_patches) is not tuple:
            raise TypeError(
                "ancestor_to_parent_patches must be an exact tuple of TypedPatch values"
            )
        if len(self.ancestor_to_parent_patches) > 2:
            raise ValueError("ancestor_to_parent_patches cannot contain more than two patches")
        if any(
            type(patch) is not TypedPatch
            for patch in self.ancestor_to_parent_patches
        ):
            raise TypeError(
                "ancestor_to_parent_patches must be an exact tuple of TypedPatch values"
            )
        for patch in self.ancestor_to_parent_patches:
            validate_typed_patch(patch)
        if self.variation_kind is VariationKind.THREE_WAY_RECOMBINATION:
            if type(self.common_ancestor) is not CandidateOccurrence:
                raise ValueError("three-way recombination requires one common ancestor")
            CandidateOccurrence.__post_init__(self.common_ancestor)
            if self.common_ancestor.candidate_id in parent_ids:
                raise ValueError("common ancestor must be a distinct occurrence")
            if self.common_ancestor.operator_invocation_id == self.operator_invocation_id:
                raise ValueError(
                    "an operator invocation cannot consume its own common ancestor"
                )
            if any(
                self.common_ancestor.proposal_sequence
                >= parent.occurrence.proposal_sequence
                for parent in self.parents
            ):
                raise ValueError(
                    "common ancestor must precede both branch-parent occurrences"
                )
            if len(self.ancestor_to_parent_patches) != 2:
                raise ValueError("three-way recombination requires exactly two branch patches")
            if (
                self.ancestor_to_parent_patches[0].limits
                != self.ancestor_to_parent_patches[1].limits
            ):
                raise ValueError(
                    "three-way branch patches must share exact algebra limits"
                )
            expected_targets = parent_ids
            actual_targets = tuple(
                patch.target_candidate_id for patch in self.ancestor_to_parent_patches
            )
            if actual_targets != expected_targets:
                raise ValueError("branch patches must follow left/right parent order")
            for parent, patch in zip(
                self.parents,
                self.ancestor_to_parent_patches,
            ):
                if (
                    patch.base_candidate_id != self.common_ancestor.candidate_id
                    or patch.base_hash != self.common_ancestor.configuration_hash
                    or patch.target_candidate_id != parent.occurrence.candidate_id
                    or patch.target_hash != parent.occurrence.configuration_hash
                ):
                    raise ValueError("ancestor branch patch endpoints do not match the case")
        elif self.common_ancestor is not None or self.ancestor_to_parent_patches:
            raise ValueError(
                "common ancestor and branch patches are exclusive to three-way recombination"
            )

        if type(self.selected_insights) is not tuple:
            raise TypeError("selected_insights must be an exact tuple of InsightRef values")
        if len(self.selected_insights) > MAX_SELECTED_INSIGHTS:
            raise ValueError("selected_insights exceeds MAX_SELECTED_INSIGHTS")
        if any(type(reference) is not InsightRef for reference in self.selected_insights):
            raise TypeError("selected_insights must be an exact tuple of InsightRef values")
        for reference in self.selected_insights:
            InsightRef.__post_init__(reference)
            if type(reference.insight_id) is not InsightId:
                raise TypeError("selected insight IDs must be exact InsightId values")
            InsightId.__post_init__(reference.insight_id)
        if len(set(self.selected_insights)) != len(self.selected_insights):
            raise ValueError("selected_insights cannot contain duplicate exact versions")

        if type(self.preservation_obligations) is not tuple:
            raise TypeError(
                "preservation_obligations must be an exact tuple of PreservationObligation values"
            )
        if len(self.preservation_obligations) > MAX_PRESERVATION_OBLIGATIONS:
            raise ValueError(
                "preservation_obligations exceeds MAX_PRESERVATION_OBLIGATIONS"
            )
        if any(
            type(obligation) is not PreservationObligation
            for obligation in self.preservation_obligations
        ):
            raise TypeError(
                "preservation_obligations must be an exact tuple of PreservationObligation values"
            )
        for obligation in self.preservation_obligations:
            PreservationObligation.__post_init__(obligation)
        if (
            self.preservation_obligations
            and self.variation_kind is not VariationKind.THREE_WAY_RECOMBINATION
        ):
            raise ValueError(
                "this slice admits preservation obligations only for verified three-way cases"
            )
        canonical_obligations = tuple(
            sorted(
                self.preservation_obligations,
                key=lambda obligation: obligation.obligation_id,
            )
        )
        if self.preservation_obligations != canonical_obligations:
            raise ValueError(
                "preservation_obligations must use canonical obligation-id order"
            )
        obligation_ids = tuple(
            obligation.obligation_id for obligation in self.preservation_obligations
        )
        if len(set(obligation_ids)) != len(obligation_ids):
            raise ValueError("preservation_obligations cannot duplicate an identity")
        for obligation in self.preservation_obligations:
            expected_source_ids = {
                PreservationSource.LEFT_BRANCH: (parent_ids[0],),
                PreservationSource.RIGHT_BRANCH: (parent_ids[1],),
                PreservationSource.IDENTICAL_NEUTRAL: parent_ids,
            }[obligation.source]
            if obligation.source_parent_candidate_ids != expected_source_ids:
                raise ValueError(
                    "preservation obligation branch provenance does not match case roles"
                )
        _validate_obligation_path_overlaps(self.preservation_obligations)

    def __eq__(self, other: object) -> bool:
        if type(self) is not VariationCase or type(other) is not VariationCase:
            return False
        VariationCase.__post_init__(self)
        VariationCase.__post_init__(other)

        def projection(value: VariationCase) -> tuple[object, ...]:
            return (
                value.operator_invocation_id.value,
                value.variation_kind.value,
                value.operator_id,
                value.operator_version,
                tuple(parent._validated_values() for parent in value.parents),
                value.requested_child_count,
                value.context_stratum_hash,
                value.reward_definition_hash,
                None
                if value.common_ancestor is None
                else value.common_ancestor._validated_values(),
                tuple(patch.patch_hash for patch in value.ancestor_to_parent_patches),
                tuple(
                    (reference.insight_id.value, reference.version)
                    for reference in value.selected_insights
                ),
                tuple(
                    obligation.obligation_id
                    for obligation in value.preservation_obligations
                ),
            )

        return projection(self) == projection(other)

    __hash__ = None


def validate_variation_case(value: VariationCase) -> None:
    """Revalidate the complete immutable variation-case graph."""

    if type(value) is not VariationCase:
        raise TypeError("variation_case must be an exact VariationCase")
    VariationCase.__post_init__(value)


__all__ = [
    "AbsenceContextKind",
    "AbsenceFailureKind",
    "MAX_PRESERVATION_OBLIGATIONS",
    "MAX_REQUESTED_CHILDREN",
    "MAX_SELECTED_INSIGHTS",
    "CandidateOccurrence",
    "ParentEdge",
    "ParentRole",
    "PreservationClaim",
    "PreservationExpectation",
    "PreservationObligation",
    "PreservationSource",
    "VariationCase",
    "VariationKind",
    "VariationParent",
    "validate_variation_case",
]
