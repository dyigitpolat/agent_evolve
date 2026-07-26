"""Lifecycle-gated insight memory built around randomized subset credit.

The bank stores immutable versions, logs one assignment per operator invocation,
and updates contextual retrieval scores only from identified selected-vs-unselected
contrasts. Unselected insights are never mechanically penalized, and untested
reflection output cannot enter retrieval without a recorded promotion.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from typing import Mapping, Sequence

from agent_evolve.domain.ids import (
    CandidateId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.finite_variation import FiniteActionEvidenceBinding
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
)
from agent_evolve.policies.memory.randomized_subset import (
    EpsilonGreedySubsetSelector,
    InsightSelectionDecision,
    InsightTrial,
    estimate_marginal_effect,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES,
    ReflectionConsumerScope,
)


_SPACE = re.compile(r"\s+")
_OPERATOR_KIND_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_JSON_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_FACT_SCHEMA_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_FACTOR_CAPABILITY_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_INSIGHT_EVIDENCE_LINEAGE_DOMAIN = b"agent-evolve:insight-evidence-lineage:v1\x00"
_INSIGHT_EVIDENCE_LINEAGE_V2_DOMAIN = (
    b"agent-evolve:insight-evidence-lineage:v2-empirical-snapshots\x00"
)
_EMPIRICAL_EVIDENCE_SNAPSHOT_DOMAIN = b"agent-evolve:empirical-evidence-snapshot:v1\x00"
_QUARANTINE_TEST_ADMISSION_DOMAIN = b"agent-evolve:quarantine-test-admission:v1\x00"


class QuarantineAssignmentStructuralError(ValueError):
    """A valid owned quarantine card cannot act within the requested scope."""


class InsightLifecycleState(str, Enum):
    """Explicit retrieval lifecycle for one exact insight version."""

    SEED = "seed"
    QUARANTINED = "quarantined"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"

    @property
    def retrievable(self) -> bool:
        return self in {type(self).SEED, type(self).PROMOTED}


class InsightOrigin(str, Enum):
    """How an insight entered the bank; lifecycle transitions do not alter it."""

    SEED = "seed"
    REFLECTION = "reflection"
    MANUAL = "manual"


class InsightRelationKind(str, Enum):
    """Declared semantic relationships; the bank never infers these."""

    REVISES = "revises"
    DUPLICATES = "duplicates"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True, slots=True)
class EmpiricalEvidenceSnapshot:
    """Engine-issued facts for one exact contrast, never model-authored prose."""

    contrast_id: str
    fact_schema_id: str
    fact_schema_version: int
    fact_schema_definition_sha256: str
    facts: FrozenJsonObject
    optimization_semantics_definition_sha256: str | None = None
    action_semantics_definition_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 ID")
        if (
            type(self.fact_schema_id) is not str
            or _FACT_SCHEMA_ID.fullmatch(self.fact_schema_id) is None
        ):
            raise ValueError("fact_schema_id must use the closed lowercase grammar")
        if type(self.fact_schema_version) is not int or self.fact_schema_version <= 0:
            raise ValueError("fact_schema_version must be a positive exact integer")
        if (
            type(self.fact_schema_definition_sha256) is not str
            or _LOWER_SHA256.fullmatch(self.fact_schema_definition_sha256) is None
        ):
            raise ValueError(
                "fact_schema_definition_sha256 must be a lowercase SHA-256 ID"
            )
        if type(self.facts) is not FrozenJsonObject:
            raise TypeError("facts must be an exact FrozenJsonObject")
        if freeze_json(self.facts) is not self.facts:
            raise TypeError("facts must already be frozen typed JSON")
        if not self.facts.items:
            raise ValueError("facts must be non-empty")
        for name in (
            "optimization_semantics_definition_sha256",
            "action_semantics_definition_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str or _LOWER_SHA256.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 ID or None")

    def _identity_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "contrast_id": self.contrast_id,
            "fact_schema_id": self.fact_schema_id,
            "fact_schema_version": self.fact_schema_version,
            "fact_schema_definition_sha256": (self.fact_schema_definition_sha256),
            "facts": thaw_json(self.facts),
            "optimization_semantics_definition_sha256": (
                self.optimization_semantics_definition_sha256
            ),
            "action_semantics_definition_sha256": (
                self.action_semantics_definition_sha256
            ),
        }

    @property
    def snapshot_sha256(self) -> str:
        return hashlib.sha256(
            _EMPIRICAL_EVIDENCE_SNAPSHOT_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self._identity_record()))
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._identity_record(), "snapshot_sha256": self.snapshot_sha256}


@dataclass(frozen=True, slots=True)
class InsightRelation:
    kind: InsightRelationKind
    target: InsightRef
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InsightRelationKind):
            raise TypeError("relation kind must be an InsightRelationKind")
        if not isinstance(self.target, InsightRef):
            raise TypeError("relation target must be an InsightRef")
        if self.note is not None and (
            type(self.note) is not str
            or not self.note.strip()
            or self.note != self.note.strip()
        ):
            raise ValueError("relation note must be non-empty canonical text")

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic durable relation projection."""

        self.__post_init__()
        InsightRef.__post_init__(self.target)
        return {
            "kind": self.kind.value,
            "target": {
                "insight_id": self.target.insight_id.value,
                "version": self.target.version,
            },
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class InsightEvidenceLineage:
    """Machine-verifiable evidence boundary supplied to one reflection call.

    ``available_contrast_ids`` records what the reflector was allowed to use;
    ``cited_contrast_ids`` records only exact full IDs supplied through that
    draft's structured citation field.  The distinction avoids treating prompt
    availability or incidental prose substrings as an evidence citation.
    """

    reflection_call_id: LLMCallId
    source_operator_invocation_ids: tuple[OperatorInvocationId, ...]
    source_candidate_ids: tuple[CandidateId, ...]
    available_contrast_ids: tuple[str, ...]
    cited_contrast_ids: tuple[str, ...] = ()
    finite_action_bindings: tuple[FiniteActionEvidenceBinding, ...] = ()
    empirical_evidence: tuple[EmpiricalEvidenceSnapshot, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reflection_call_id) is not LLMCallId:
            raise TypeError("reflection_call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.reflection_call_id)
        _validate_exact_sorted_ids(
            self.source_operator_invocation_ids,
            OperatorInvocationId,
            name="source_operator_invocation_ids",
        )
        _validate_exact_sorted_ids(
            self.source_candidate_ids,
            CandidateId,
            name="source_candidate_ids",
        )
        for name in ("available_contrast_ids", "cited_contrast_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or _LOWER_SHA256.fullmatch(value) is None
                for value in values
            ):
                raise TypeError(f"{name} must be a tuple of lowercase SHA-256 IDs")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonically sorted")
        if not set(self.cited_contrast_ids).issubset(self.available_contrast_ids):
            raise ValueError("cited_contrast_ids must be available to the reflection")
        if type(self.finite_action_bindings) is not tuple or any(
            type(binding) is not FiniteActionEvidenceBinding
            for binding in self.finite_action_bindings
        ):
            raise TypeError(
                "finite_action_bindings must be an exact tuple of "
                "FiniteActionEvidenceBinding values"
            )
        for binding in self.finite_action_bindings:
            FiniteActionEvidenceBinding.__post_init__(binding)
        binding_contrast_ids = tuple(
            binding.contrast_id for binding in self.finite_action_bindings
        )
        if binding_contrast_ids != tuple(sorted(set(binding_contrast_ids))):
            raise ValueError(
                "finite_action_bindings must have unique canonical contrast order"
            )
        if not set(binding_contrast_ids).issubset(self.cited_contrast_ids):
            raise ValueError(
                "finite action evidence must bind a cited reflection contrast"
            )
        if type(self.empirical_evidence) is not tuple or any(
            type(value) is not EmpiricalEvidenceSnapshot
            for value in self.empirical_evidence
        ):
            raise TypeError(
                "empirical_evidence must be an exact tuple of "
                "EmpiricalEvidenceSnapshot values"
            )
        for snapshot in self.empirical_evidence:
            EmpiricalEvidenceSnapshot.__post_init__(snapshot)
        empirical_contrast_ids = tuple(
            value.contrast_id for value in self.empirical_evidence
        )
        if empirical_contrast_ids != tuple(sorted(set(empirical_contrast_ids))):
            raise ValueError(
                "empirical_evidence must have unique canonical contrast order"
            )
        if self.empirical_evidence and empirical_contrast_ids != (
            self.cited_contrast_ids
        ):
            raise ValueError(
                "empirical evidence must exactly cover every cited contrast"
            )

    def _identity_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "schema_version": 2 if self.empirical_evidence else 1,
            "reflection_call_id": self.reflection_call_id.value,
            "source_operator_invocation_ids": [
                value.value for value in self.source_operator_invocation_ids
            ],
            "source_candidate_ids": [
                value.value for value in self.source_candidate_ids
            ],
            "available_contrast_ids": list(self.available_contrast_ids),
            "cited_contrast_ids": list(self.cited_contrast_ids),
            "finite_action_bindings": [
                binding.to_record() for binding in self.finite_action_bindings
            ],
        }
        if self.empirical_evidence:
            record["empirical_evidence"] = [
                snapshot.to_record() for snapshot in self.empirical_evidence
            ]
        return record

    @property
    def identity_sha256(self) -> str:
        """Bind citations, source IDs, and exact action-conditioned evidence."""

        domain = (
            _INSIGHT_EVIDENCE_LINEAGE_V2_DOMAIN
            if self.empirical_evidence
            else _INSIGHT_EVIDENCE_LINEAGE_DOMAIN
        )
        return hashlib.sha256(
            domain + canonical_typed_json_bytes(freeze_json(self._identity_record()))
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        """Return deterministic durable provenance without benchmark objects."""

        return {
            **self._identity_record(),
            "lineage_identity_sha256": self.identity_sha256,
        }

    @property
    def portfolio_action_evidence(
        self,
    ) -> tuple[FiniteActionEvidenceBinding, ...]:
        """Immutable framework-neutral projection for portfolio card adapters."""

        self.__post_init__()
        return self.finite_action_bindings


def compose_epistemic_prompt_payload(
    *,
    empirical_evidence: tuple[EmpiricalEvidenceSnapshot, ...],
    hypothesis: FrozenJsonObject,
) -> FrozenJsonObject:
    """Keep trusted observations structurally separate from model hypotheses.

    Benchmark adapters may scrub or otherwise project a hypothesis before this
    call, but they cannot author empirical snapshots.  The returned typed JSON
    is suitable for a card prompt payload and makes the epistemic boundary
    machine-inspectable rather than relying on cautionary prose.
    """

    if type(empirical_evidence) is not tuple or any(
        type(value) is not EmpiricalEvidenceSnapshot for value in empirical_evidence
    ):
        raise TypeError("empirical_evidence must be an exact tuple of snapshots")
    for snapshot in empirical_evidence:
        EmpiricalEvidenceSnapshot.__post_init__(snapshot)
    contrast_ids = tuple(value.contrast_id for value in empirical_evidence)
    if contrast_ids != tuple(sorted(set(contrast_ids))):
        raise ValueError("empirical_evidence must have unique canonical contrast order")
    if type(hypothesis) is not FrozenJsonObject:
        raise TypeError("hypothesis must be an exact FrozenJsonObject")
    if freeze_json(hypothesis) is not hypothesis:
        raise TypeError("hypothesis must already be frozen typed JSON")
    hypothesis_record = thaw_json(hypothesis)
    if hypothesis_record.get("epistemic_status") != "unverified_hypothesis":
        raise ValueError(
            "hypothesis must declare epistemic_status=unverified_hypothesis"
        )
    payload = freeze_json(
        {
            "schema_version": 1,
            "empirical_facts": [
                snapshot.to_record() for snapshot in empirical_evidence
            ],
            "hypothesis": hypothesis_record,
            "interpretation_policy": {
                "empirical_facts_are_observations": True,
                "hypothesis_is_observation": False,
                "mechanism_requires_independent_validation": True,
            },
        }
    )
    if type(payload) is not FrozenJsonObject:
        raise AssertionError("epistemic prompt payload must freeze to an object")
    return payload


def _validate_reflection_action_lineage(
    draft: InsightDraft,
    lineage: InsightEvidenceLineage,
) -> None:
    """Reject action recommendations detached from their cited finite evidence."""

    if type(draft) is not InsightDraft:
        raise TypeError("draft must be an exact InsightDraft")
    InsightDraft.__post_init__(draft)
    if type(lineage) is not InsightEvidenceLineage:
        raise TypeError("lineage must be an exact InsightEvidenceLineage")
    InsightEvidenceLineage.__post_init__(lineage)
    binding_contrast_ids = tuple(
        binding.contrast_id for binding in lineage.finite_action_bindings
    )
    if not set(binding_contrast_ids).issubset(draft.evidence_contrast_ids):
        raise ValueError(
            "finite action evidence must bind a contrast cited by the draft"
        )
    if not draft.recommended_option_ids:
        return
    if draft.evidence_contrast_ids != lineage.cited_contrast_ids:
        raise ValueError(
            "exact-action draft citations differ from their evidence lineage"
        )
    if binding_contrast_ids != lineage.cited_contrast_ids:
        raise ValueError(
            "exact-action reflection requires one finite action binding per citation"
        )
    bound_option_ids = tuple(
        sorted({binding.option_id for binding in lineage.finite_action_bindings})
    )
    if bound_option_ids != draft.recommended_option_ids:
        raise ValueError(
            "exact-action recommendation differs from its evidence action bindings"
        )


@dataclass(frozen=True, slots=True)
class ReflectedInsightBatchItem:
    """One staged reflection card and its independently verified lineage."""

    draft: InsightDraft
    evidence_lineage: InsightEvidenceLineage

    def __post_init__(self) -> None:
        if type(self.draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(self.draft)
        if type(self.evidence_lineage) is not InsightEvidenceLineage:
            raise TypeError("evidence_lineage must be an exact InsightEvidenceLineage")
        InsightEvidenceLineage.__post_init__(self.evidence_lineage)
        _validate_reflection_action_lineage(self.draft, self.evidence_lineage)


@dataclass(frozen=True, slots=True)
class InsightLifecycleTransition:
    """Append-only audit record for an explicit lifecycle change."""

    sequence: int
    reference: InsightRef
    prior_state: InsightLifecycleState
    new_state: InsightLifecycleState
    reason: str
    supporting_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("transition sequence must be positive")
        if not isinstance(self.reference, InsightRef):
            raise TypeError("transition reference must be an InsightRef")
        if not isinstance(self.prior_state, InsightLifecycleState) or not isinstance(
            self.new_state, InsightLifecycleState
        ):
            raise TypeError("transition states must be InsightLifecycleState values")
        if (
            type(self.reason) is not str
            or not self.reason.strip()
            or self.reason != self.reason.strip()
            or len(self.reason) > 1_024
        ):
            raise ValueError("transition reason must be canonical non-empty text")
        if type(self.supporting_evidence) is not tuple:
            raise TypeError("supporting_evidence must be an exact tuple")
        if _supporting_evidence(self.supporting_evidence) != self.supporting_evidence:
            raise ValueError("supporting_evidence must use canonical sorted order")


@dataclass(frozen=True, slots=True)
class InsightLifecycleChangeRequest:
    """One proposed lifecycle change for atomic batch publication."""

    reference: InsightRef
    new_state: InsightLifecycleState
    reason: str
    supporting_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        if self.new_state not in {
            InsightLifecycleState.PROMOTED,
            InsightLifecycleState.DEPRECATED,
        }:
            raise ValueError("new_state must be promoted or deprecated")
        if (
            type(self.reason) is not str
            or not self.reason.strip()
            or self.reason != self.reason.strip()
            or len(self.reason) > 1_024
        ):
            raise ValueError("reason must be canonical non-empty text")
        if type(self.supporting_evidence) is not tuple:
            raise TypeError("supporting_evidence must be an exact tuple")
        if _supporting_evidence(self.supporting_evidence) != self.supporting_evidence:
            raise ValueError("supporting_evidence must use canonical sorted order")


def _validate_exact_sorted_ids(
    values: tuple, expected_type: type, *, name: str
) -> None:
    if type(values) is not tuple or any(
        type(value) is not expected_type for value in values
    ):
        raise TypeError(
            f"{name} must be an exact tuple of {expected_type.__name__} values"
        )
    for value in values:
        expected_type.__post_init__(value)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _supporting_evidence(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("supporting_evidence must be a sequence of evidence IDs")
    canonical = tuple(values)
    if any(
        type(value) is not str or _EVIDENCE_REFERENCE.fullmatch(value) is None
        for value in canonical
    ):
        raise ValueError("supporting_evidence must contain bounded evidence-ID tokens")
    if len(set(canonical)) != len(canonical):
        raise ValueError("supporting_evidence cannot contain duplicates")
    return tuple(sorted(canonical))


def _claim_key(value: str) -> str:
    return _SPACE.sub(" ", value.strip().casefold())


def _operator_kind_token(value: str) -> str:
    if type(value) is not str or _OPERATOR_KIND_TOKEN.fullmatch(value) is None:
        raise ValueError(
            "operator kinds must be lowercase tokens containing only letters, "
            "digits, and underscores"
        )
    return value


def _applicable_operator_kinds(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("applicable_operator_kinds must be a sequence of strings")
    canonical = tuple(sorted(_operator_kind_token(value) for value in values))
    if len(set(canonical)) != len(canonical):
        raise ValueError("applicable_operator_kinds cannot contain duplicates")
    return canonical


def _canonical_paths(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of canonical JSON paths")
    paths = tuple(values)
    if any(
        type(path) is not str or _CANONICAL_JSON_PATH.fullmatch(path) is None
        for path in paths
    ):
        raise ValueError(
            f"{name} must contain exact canonical JSON paths beginning with $."
        )
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(paths))


def _canonical_factor_capabilities(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("factor_capabilities must be a sequence of strings or None")
    capabilities = tuple(values)
    if any(
        type(value) is not str or _FACTOR_CAPABILITY_TOKEN.fullmatch(value) is None
        for value in capabilities
    ):
        raise ValueError("factor_capabilities must contain canonical capability tokens")
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("factor_capabilities cannot contain duplicates")
    if len(capabilities) > MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES:
        raise ValueError("factor_capabilities exceeds the semantic vocabulary cap")
    return tuple(sorted(capabilities))


def _paths_overlap(first: str, second: str) -> bool:
    """Whether either canonical JSON path is an ancestor of the other."""

    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


@dataclass(frozen=True, slots=True)
class QuarantineTestAdmissionReceipt:
    """Bank-issued authority for randomized use of exact quarantine versions."""

    references: tuple[InsightRef, ...]
    operator_kind: str
    editable_paths: tuple[str, ...]
    source_admission_request_sha256: str
    memory_trial_count_cutoff: int
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.references) is not tuple
            or not self.references
            or any(type(value) is not InsightRef for value in self.references)
        ):
            raise ValueError("references must be a non-empty exact tuple")
        if self.references != tuple(sorted(set(self.references))):
            raise ValueError("references must be unique and canonical")
        operator = _operator_kind_token(self.operator_kind)
        if operator != self.operator_kind:
            raise ValueError("operator_kind must be canonical")
        paths = _canonical_paths(self.editable_paths, name="editable_paths")
        if paths != self.editable_paths:
            raise ValueError("editable_paths must be canonical")
        if (
            type(self.source_admission_request_sha256) is not str
            or _LOWER_SHA256.fullmatch(self.source_admission_request_sha256) is None
        ):
            raise ValueError(
                "source_admission_request_sha256 must be a lowercase SHA-256 ID"
            )
        if (
            type(self.memory_trial_count_cutoff) is not int
            or self.memory_trial_count_cutoff < 0
        ):
            raise ValueError("memory_trial_count_cutoff must be non-negative")
        object.__setattr__(
            self,
            "receipt_sha256",
            hashlib.sha256(
                _QUARANTINE_TEST_ADMISSION_DOMAIN
                + canonical_typed_json_bytes(freeze_json(self._unsigned_record()))
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "references": [
                {
                    "insight_id": value.insight_id.value,
                    "version": value.version,
                }
                for value in self.references
            ],
            "operator_kind": self.operator_kind,
            "editable_paths": list(self.editable_paths),
            "source_admission_request_sha256": (self.source_admission_request_sha256),
            "memory_trial_count_cutoff": self.memory_trial_count_cutoff,
            "scope": "quarantine_diagnostic_only",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class InsightMemoryEntry:
    reference: InsightRef
    draft: InsightDraft
    initial_score: float
    applicable_operator_kinds: tuple[str, ...] = ()
    lifecycle_state: InsightLifecycleState = InsightLifecycleState.SEED
    origin: InsightOrigin = InsightOrigin.SEED
    evidence_lineage: InsightEvidenceLineage | None = None
    relations: tuple[InsightRelation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        if type(self.draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(self.draft)
        if type(self.initial_score) is not float or not math.isfinite(
            self.initial_score
        ):
            raise TypeError("initial_score must be a finite canonical float")
        canonical = _applicable_operator_kinds(self.applicable_operator_kinds)
        if canonical != self.applicable_operator_kinds:
            raise ValueError(
                "applicable_operator_kinds must use canonical sorted order"
            )
        if not isinstance(self.lifecycle_state, InsightLifecycleState):
            raise TypeError("lifecycle_state must be an InsightLifecycleState")
        if not isinstance(self.origin, InsightOrigin):
            raise TypeError("origin must be an InsightOrigin")
        if self.lifecycle_state is InsightLifecycleState.SEED:
            if self.origin is not InsightOrigin.SEED:
                raise ValueError(
                    "only seed-origin insights may have seed lifecycle state"
                )
        elif self.origin is InsightOrigin.SEED and self.lifecycle_state not in {
            InsightLifecycleState.SEED,
            InsightLifecycleState.DEPRECATED,
        }:
            raise ValueError(
                "seed-origin insights can only remain seed or be deprecated"
            )
        if self.origin is InsightOrigin.REFLECTION:
            if self.evidence_lineage is None:
                raise ValueError("reflection-origin insights require evidence lineage")
        elif self.evidence_lineage is not None:
            raise ValueError("reflection evidence lineage requires reflection origin")
        if self.evidence_lineage is not None:
            if type(self.evidence_lineage) is not InsightEvidenceLineage:
                raise TypeError(
                    "evidence_lineage must be an exact InsightEvidenceLineage"
                )
            InsightEvidenceLineage.__post_init__(self.evidence_lineage)
            _validate_reflection_action_lineage(
                self.draft,
                self.evidence_lineage,
            )
        if type(self.relations) is not tuple or any(
            not isinstance(relation, InsightRelation) for relation in self.relations
        ):
            raise TypeError("relations must be a tuple of InsightRelation values")
        relation_keys = tuple(
            (relation.target, relation.kind.value) for relation in self.relations
        )
        if relation_keys != tuple(sorted(set(relation_keys))):
            raise ValueError("relations must be unique and canonically sorted")

    @property
    def retrievable(self) -> bool:
        return self.lifecycle_state.retrievable

    def to_record(self) -> dict[str, object]:
        """Return a complete, lossless, JSON-ready durable projection.

        The draft record carries every immutable model-authored field while
        its hashes make later content drift directly detectable.  Scores use
        hexadecimal float text so a write/read cycle cannot change the exact
        binary value.  Lifecycle eligibility is emitted explicitly because it
        is operational state, even though it is derivable from the lifecycle
        enum.
        """

        self.__post_init__()
        return {
            "schema_version": 1,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "draft": self.draft.content_record(),
            "draft_content_sha256": self.draft.content_sha256,
            "draft_hypothesis_sha256": self.draft.hypothesis_sha256,
            "initial_score_hex": self.initial_score.hex(),
            "applicable_operator_kinds": list(self.applicable_operator_kinds),
            "lifecycle_state": self.lifecycle_state.value,
            "retrievable": self.retrievable,
            "origin": self.origin.value,
            "evidence_lineage": (
                None
                if self.evidence_lineage is None
                else self.evidence_lineage.to_record()
            ),
            "relations": [relation.to_record() for relation in self.relations],
        }


class InsightMemoryBank:
    """Small replaceable policy object for online development experiments."""

    def __init__(
        self,
        *,
        id_factory,
        exploration_probability: Fraction = Fraction(1, 2),
        shrinkage_effective_sample_size: float = 4.0,
    ) -> None:
        if shrinkage_effective_sample_size <= 0:
            raise ValueError("shrinkage_effective_sample_size must be positive")
        self._ids = id_factory
        self._selector = EpsilonGreedySubsetSelector(exploration_probability)
        self._shrinkage_ess = float(shrinkage_effective_sample_size)
        self._entries: dict[InsightRef, InsightMemoryEntry] = {}
        self._claim_index: dict[str, InsightRef] = {}
        self._scores: dict[tuple[str, InsightRef], float] = {}
        self._trials: list[InsightTrial] = []
        self._transitions: list[InsightLifecycleTransition] = []
        self._quarantine_admissions: dict[
            str,
            QuarantineTestAdmissionReceipt,
        ] = {}
        self._quarantine_admission_by_reference: dict[InsightRef, str] = {}

    @property
    def entries(self) -> tuple[InsightMemoryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    @property
    def trials(self) -> tuple[InsightTrial, ...]:
        return tuple(self._trials)

    @property
    def transitions(self) -> tuple[InsightLifecycleTransition, ...]:
        return tuple(self._transitions)

    def entries_for(
        self, references: Sequence[InsightRef]
    ) -> tuple[InsightMemoryEntry, ...]:
        """Return immutable exact-version entries in caller-supplied order."""

        owned = self._owned_reference_sequence(references, name="references")
        return tuple(self._entries[reference] for reference in owned)

    def add(
        self,
        draft: InsightDraft,
        *,
        initial_score: float | None = None,
        applicable_operator_kinds: Sequence[str] = (),
        origin: InsightOrigin = InsightOrigin.SEED,
        lifecycle_state: InsightLifecycleState | None = None,
        evidence_lineage: InsightEvidenceLineage | None = None,
        relations: Sequence[InsightRelation] = (),
    ) -> tuple[InsightMemoryEntry, bool]:
        """Add one immutable v1 insight, or return an exact-claim match.

        Existing callers create eligible seed priors by default. Reflection and
        manual origins are forced to start quarantined; promotion is a separate,
        recorded operation.
        """

        if type(draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(draft)
        _canonical_paths(draft.affected_paths, name="draft.affected_paths")
        operator_kinds = _applicable_operator_kinds(applicable_operator_kinds)
        state = self._initial_state(origin, lifecycle_state, evidence_lineage)
        if origin is InsightOrigin.REFLECTION:
            assert evidence_lineage is not None
            _validate_reflection_action_lineage(draft, evidence_lineage)
        canonical_relations = self._canonical_relations(relations)
        if any(
            relation.kind is InsightRelationKind.REVISES
            for relation in canonical_relations
        ):
            raise ValueError("use add_revision to create a revises relationship")
        key = _claim_key(draft.claim)
        existing = self._claim_index.get(key)
        if existing is not None:
            return self._entries[existing], False
        score = 0.0 if initial_score is None else float(initial_score)
        if not math.isfinite(score):
            raise ValueError("initial_score must be finite")
        reference = InsightRef(self._ids.new_insight_id(), 1)
        entry = InsightMemoryEntry(
            reference=reference,
            draft=draft,
            initial_score=score,
            applicable_operator_kinds=operator_kinds,
            lifecycle_state=state,
            origin=origin,
            evidence_lineage=evidence_lineage,
            relations=canonical_relations,
        )
        self._insert(entry)
        return entry, True

    def add_reflection_batch(
        self,
        items: tuple[ReflectedInsightBatchItem, ...],
        *,
        initial_score: float = 0.0,
        applicable_operator_kinds: Sequence[str] = (),
    ) -> tuple[InsightMemoryEntry, ...]:
        """Publish a complete reflection batch atomically or publish nothing.

        Reflection shards are independently generated, but a scientific batch
        is useful only when every expected origin yielded a new quarantine
        card.  All validation and duplicate checks therefore precede entry
        construction and insertion.
        """

        if type(items) is not tuple or not items:
            raise ValueError("a reflection batch must be a non-empty exact tuple")
        if any(type(item) is not ReflectedInsightBatchItem for item in items):
            raise TypeError(
                "reflection batch must contain exact ReflectedInsightBatchItem values"
            )
        score = float(initial_score)
        if not math.isfinite(score):
            raise ValueError("initial_score must be finite")
        operator_kinds = _applicable_operator_kinds(applicable_operator_kinds)

        claim_keys: list[str] = []
        for item in items:
            ReflectedInsightBatchItem.__post_init__(item)
            _canonical_paths(item.draft.affected_paths, name="draft.affected_paths")
            self._initial_state(
                InsightOrigin.REFLECTION,
                InsightLifecycleState.QUARANTINED,
                item.evidence_lineage,
            )
            claim_keys.append(_claim_key(item.draft.claim))
        if len(set(claim_keys)) != len(claim_keys):
            raise ValueError("reflection batch contains duplicate normalized claims")
        if any(key in self._claim_index for key in claim_keys):
            raise ValueError("reflection batch collides with an existing claim")

        staged = tuple(
            InsightMemoryEntry(
                reference=InsightRef(self._ids.new_insight_id(), 1),
                draft=item.draft,
                initial_score=score,
                applicable_operator_kinds=operator_kinds,
                lifecycle_state=InsightLifecycleState.QUARANTINED,
                origin=InsightOrigin.REFLECTION,
                evidence_lineage=item.evidence_lineage,
            )
            for item in items
        )
        staged_references = tuple(entry.reference for entry in staged)
        if len(set(staged_references)) != len(staged_references) or any(
            reference in self._entries for reference in staged_references
        ):
            raise ValueError(
                "reflection batch ID allocation collided before publication"
            )
        for entry in staged:
            self._insert(entry)
        return staged

    def extend(
        self,
        drafts: Sequence[InsightDraft],
        *,
        initial_score: float | None = None,
        applicable_operator_kinds: Sequence[str] = (),
        origin: InsightOrigin = InsightOrigin.SEED,
        lifecycle_state: InsightLifecycleState | None = None,
        evidence_lineage: InsightEvidenceLineage | None = None,
        relations: Sequence[InsightRelation] = (),
    ) -> tuple[InsightMemoryEntry, ...]:
        added: list[InsightMemoryEntry] = []
        for draft in drafts:
            entry, is_new = self.add(
                draft,
                initial_score=initial_score,
                applicable_operator_kinds=applicable_operator_kinds,
                origin=origin,
                lifecycle_state=lifecycle_state,
                evidence_lineage=evidence_lineage,
                relations=relations,
            )
            if is_new:
                added.append(entry)
        return tuple(added)

    def add_revision(
        self,
        predecessor: InsightRef,
        draft: InsightDraft,
        *,
        initial_score: float | None = None,
        applicable_operator_kinds: Sequence[str] = (),
        origin: InsightOrigin = InsightOrigin.MANUAL,
        evidence_lineage: InsightEvidenceLineage | None = None,
        relations: Sequence[InsightRelation] = (),
        revision_note: str = "explicit semantic revision",
    ) -> InsightMemoryEntry:
        """Create a quarantined next version without superseding its predecessor.

        The caller must separately promote this version and deprecate the old
        version after validation. This keeps semantic edits from silently
        inheriting the predecessor's retrieval eligibility or credit.
        """

        prior = self._owned_entry(predecessor)
        versions = tuple(
            reference.version
            for reference in self._entries
            if reference.insight_id == predecessor.insight_id
        )
        if predecessor.version != max(versions):
            raise ValueError("predecessor must be the latest version of its insight")
        if origin is InsightOrigin.SEED:
            raise ValueError("revisions must use manual or reflection origin")
        if type(draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(draft)
        _canonical_paths(draft.affected_paths, name="draft.affected_paths")
        operator_kinds = _applicable_operator_kinds(applicable_operator_kinds)
        state = self._initial_state(origin, None, evidence_lineage)
        if origin is InsightOrigin.REFLECTION:
            assert evidence_lineage is not None
            _validate_reflection_action_lineage(draft, evidence_lineage)
        user_relations = self._canonical_relations(relations)
        if any(
            relation.kind is InsightRelationKind.REVISES for relation in user_relations
        ):
            raise ValueError("add_revision declares its predecessor automatically")
        declared = (
            *user_relations,
            InsightRelation(
                InsightRelationKind.REVISES,
                predecessor,
                revision_note,
            ),
        )
        canonical_relations = self._canonical_relations(declared)
        score = 0.0 if initial_score is None else float(initial_score)
        if not math.isfinite(score):
            raise ValueError("initial_score must be finite")
        reference = InsightRef(predecessor.insight_id, predecessor.version + 1)
        claim_match = self._claim_index.get(_claim_key(draft.claim))
        if claim_match is not None and claim_match != predecessor:
            raise ValueError(
                "revision claim exactly matches a different insight version"
            )
        entry = InsightMemoryEntry(
            reference=reference,
            draft=draft,
            initial_score=score,
            applicable_operator_kinds=operator_kinds,
            lifecycle_state=state,
            origin=origin,
            evidence_lineage=evidence_lineage,
            relations=canonical_relations,
        )
        self._insert(entry)
        # Retain the predecessor's exact-claim lookup if the prose changed; for
        # an unchanged claim, future exact matches resolve to the latest version.
        if _claim_key(prior.draft.claim) == _claim_key(draft.claim):
            self._claim_index[_claim_key(draft.claim)] = reference
        return entry

    def apply_lifecycle_batch(
        self,
        requests: tuple[InsightLifecycleChangeRequest, ...],
    ) -> tuple[InsightMemoryEntry, ...]:
        """Validate and publish independent lifecycle changes atomically.

        Reflection blocks and generation barriers can resolve several cards at
        once.  Publishing one transition at a time would leave a partially
        mutated bank if a later card were foreign or in the wrong state.  This
        method stages every updated entry and transition first, then commits in
        canonical reference order so gather completion order has no meaning.
        """

        if type(requests) is not tuple or not requests:
            raise ValueError("lifecycle batch must be a non-empty exact tuple")
        if any(type(value) is not InsightLifecycleChangeRequest for value in requests):
            raise TypeError(
                "lifecycle batch must contain exact InsightLifecycleChangeRequest values"
            )
        for value in requests:
            InsightLifecycleChangeRequest.__post_init__(value)
        canonical = tuple(sorted(requests, key=lambda value: value.reference))
        references = tuple(value.reference for value in canonical)
        if len(set(references)) != len(references):
            raise ValueError("lifecycle batch cannot repeat an insight reference")

        updated_entries: list[InsightMemoryEntry] = []
        transitions: list[InsightLifecycleTransition] = []
        for offset, request in enumerate(canonical, start=1):
            entry = self._owned_entry(request.reference)
            if request.new_state is InsightLifecycleState.PROMOTED:
                if entry.lifecycle_state is not InsightLifecycleState.QUARANTINED:
                    raise ValueError("only a quarantined insight can be promoted")
                if not request.supporting_evidence:
                    raise ValueError(
                        "promotion requires at least one supporting evidence ID"
                    )
            elif entry.lifecycle_state is InsightLifecycleState.DEPRECATED:
                raise ValueError("a deprecated insight cannot transition again")
            transition = InsightLifecycleTransition(
                sequence=len(self._transitions) + offset,
                reference=request.reference,
                prior_state=entry.lifecycle_state,
                new_state=request.new_state,
                reason=request.reason,
                supporting_evidence=request.supporting_evidence,
            )
            transitions.append(transition)
            updated_entries.append(replace(entry, lifecycle_state=request.new_state))

        for entry in updated_entries:
            self._entries[entry.reference] = entry
        self._transitions.extend(transitions)
        return tuple(updated_entries)

    def promote(
        self,
        reference: InsightRef,
        *,
        reason: str,
        supporting_evidence: Sequence[str],
    ) -> InsightMemoryEntry:
        """Promote one tested quarantine entry into retrieval eligibility."""

        evidence = _supporting_evidence(supporting_evidence)
        return self.apply_lifecycle_batch(
            (
                InsightLifecycleChangeRequest(
                    reference=reference,
                    new_state=InsightLifecycleState.PROMOTED,
                    reason=reason,
                    supporting_evidence=evidence,
                ),
            )
        )[0]

    def deprecate(
        self,
        reference: InsightRef,
        *,
        reason: str,
        supporting_evidence: Sequence[str] = (),
    ) -> InsightMemoryEntry:
        """Make a seed, quarantine, or promoted entry permanently ineligible."""

        return self.apply_lifecycle_batch(
            (
                InsightLifecycleChangeRequest(
                    reference=reference,
                    new_state=InsightLifecycleState.DEPRECATED,
                    reason=reason,
                    supporting_evidence=_supporting_evidence(supporting_evidence),
                ),
            )
        )[0]

    def _initial_state(
        self,
        origin: InsightOrigin,
        lifecycle_state: InsightLifecycleState | None,
        evidence_lineage: InsightEvidenceLineage | None,
    ) -> InsightLifecycleState:
        if not isinstance(origin, InsightOrigin):
            raise TypeError("origin must be an InsightOrigin")
        expected = (
            InsightLifecycleState.SEED
            if origin is InsightOrigin.SEED
            else InsightLifecycleState.QUARANTINED
        )
        state = expected if lifecycle_state is None else lifecycle_state
        if not isinstance(state, InsightLifecycleState):
            raise TypeError("lifecycle_state must be an InsightLifecycleState")
        if state is not expected:
            raise ValueError(
                f"{origin.value}-origin insights must start in {expected.value} state"
            )
        if origin is InsightOrigin.REFLECTION:
            if evidence_lineage is None:
                raise ValueError("reflection-origin insights require evidence lineage")
            if type(evidence_lineage) is not InsightEvidenceLineage:
                raise TypeError(
                    "evidence_lineage must be an exact InsightEvidenceLineage"
                )
            InsightEvidenceLineage.__post_init__(evidence_lineage)
        elif evidence_lineage is not None:
            raise ValueError("evidence_lineage is only valid for reflection origin")
        return state

    def _canonical_relations(
        self, relations: Sequence[InsightRelation]
    ) -> tuple[InsightRelation, ...]:
        if isinstance(relations, (str, bytes)) or not isinstance(relations, Sequence):
            raise TypeError("relations must be a sequence of InsightRelation values")
        values = tuple(relations)
        if any(not isinstance(relation, InsightRelation) for relation in values):
            raise TypeError("relations must contain InsightRelation values")
        keys = tuple((relation.target, relation.kind.value) for relation in values)
        if len(set(keys)) != len(keys):
            raise ValueError("relations cannot contain duplicate target-kind pairs")
        foreign = tuple(
            relation.target
            for relation in values
            if relation.target not in self._entries
        )
        if foreign:
            raise ValueError("relations cannot target a foreign insight reference")
        return tuple(sorted(values, key=lambda item: (item.target, item.kind.value)))

    def _insert(self, entry: InsightMemoryEntry) -> None:
        if entry.reference in self._entries:
            raise ValueError("insight reference already exists")
        self._entries[entry.reference] = entry
        self._claim_index[_claim_key(entry.draft.claim)] = entry.reference

    def _owned_entry(self, reference: InsightRef) -> InsightMemoryEntry:
        if not isinstance(reference, InsightRef):
            raise TypeError("reference must be an InsightRef")
        try:
            return self._entries[reference]
        except KeyError as exc:
            raise ValueError("reference is foreign to this insight bank") from exc

    def _transition(
        self,
        reference: InsightRef,
        new_state: InsightLifecycleState,
        *,
        reason: str,
        supporting_evidence: tuple[str, ...],
    ) -> InsightMemoryEntry:
        entry = self._owned_entry(reference)
        if new_state is InsightLifecycleState.PROMOTED:
            if entry.lifecycle_state is not InsightLifecycleState.QUARANTINED:
                raise ValueError("only a quarantined insight can be promoted")
        elif new_state is InsightLifecycleState.DEPRECATED:
            if entry.lifecycle_state is InsightLifecycleState.DEPRECATED:
                raise ValueError("a deprecated insight cannot transition again")
        else:  # pragma: no cover - private method has two public call sites.
            raise ValueError("unsupported insight lifecycle transition")
        transition = InsightLifecycleTransition(
            sequence=len(self._transitions) + 1,
            reference=reference,
            prior_state=entry.lifecycle_state,
            new_state=new_state,
            reason=reason,
            supporting_evidence=supporting_evidence,
        )
        updated = replace(entry, lifecycle_state=new_state)
        self._entries[reference] = updated
        self._transitions.append(transition)
        return updated

    def _eligible_reference_subset(
        self,
        eligible_references: Sequence[InsightRef] | None,
    ) -> tuple[InsightRef, ...]:
        if eligible_references is None:
            return tuple(
                reference
                for reference, entry in sorted(self._entries.items())
                if entry.retrievable
            )
        if isinstance(eligible_references, (str, bytes)) or not isinstance(
            eligible_references, Sequence
        ):
            raise TypeError(
                "eligible_references must be a sequence of InsightRef values"
            )
        references = tuple(eligible_references)
        if any(not isinstance(reference, InsightRef) for reference in references):
            raise TypeError("eligible_references must contain InsightRef values")
        if len(set(references)) != len(references):
            raise ValueError("eligible_references cannot contain duplicates")
        foreign = tuple(
            reference for reference in references if reference not in self._entries
        )
        if foreign:
            raise ValueError("eligible_references contains a foreign insight reference")
        ineligible = tuple(
            reference
            for reference in references
            if not self._entries[reference].retrievable
        )
        if ineligible:
            raise ValueError(
                "eligible_references contains a lifecycle-ineligible insight reference"
            )
        return tuple(sorted(references))

    def eligible_references(
        self,
        *,
        operator_kind: str,
        editable_paths: Sequence[str] | None = None,
        consumer_scope: ReflectionConsumerScope | None = None,
        factor_capabilities: Sequence[str] | None = None,
    ) -> tuple[InsightRef, ...]:
        """Return deterministic structurally and semantically compatible cards.

        Applicability is deliberately structural: natural-language triggers are
        not interpreted. An empty operator-kind tuple on an entry means that the
        insight applies to every valid operator token.  The v3 semantic filters
        are opt-in so historical callers retain their exact behavior.  When a
        filter is supplied, semantic-v3 cards fail closed against their declared
        consumer scopes and required factor capabilities; legacy cards remain
        governed by their historical operator/path contract.
        """

        operator = _operator_kind_token(operator_kind)
        paths = (
            None
            if editable_paths is None
            else _canonical_paths(editable_paths, name="editable_paths")
        )
        if consumer_scope is not None and type(consumer_scope) is not (
            ReflectionConsumerScope
        ):
            raise TypeError(
                "consumer_scope must be an exact ReflectionConsumerScope or None"
            )
        capabilities = _canonical_factor_capabilities(factor_capabilities)
        return tuple(
            reference
            for reference, entry in sorted(self._entries.items())
            if entry.retrievable
            and self._structurally_applicable(entry, operator, paths)
            and self._semantically_applicable(
                entry,
                consumer_scope=consumer_scope,
                factor_capabilities=capabilities,
            )
        )

    @staticmethod
    def _semantically_applicable(
        entry: InsightMemoryEntry,
        *,
        consumer_scope: ReflectionConsumerScope | None,
        factor_capabilities: tuple[str, ...] | None,
    ) -> bool:
        draft = entry.draft
        if not draft.has_semantic_contract:
            return True
        return (consumer_scope is None or consumer_scope in draft.consumer_scopes) and (
            factor_capabilities is None
            or set(draft.factor_capabilities).issubset(factor_capabilities)
        )

    @staticmethod
    def _structurally_applicable(
        entry: InsightMemoryEntry,
        operator_kind: str,
        editable_paths: tuple[str, ...] | None,
    ) -> bool:
        return (
            not entry.applicable_operator_kinds
            or operator_kind in entry.applicable_operator_kinds
        ) and (
            editable_paths is None
            or any(
                _paths_overlap(editable, affected)
                for editable in editable_paths
                for affected in entry.draft.affected_paths
            )
        )

    def validate_quarantine_test_assignment(
        self,
        references: Sequence[InsightRef],
        *,
        operator_kind: str,
        editable_paths: Sequence[str] | None = None,
    ) -> tuple[InsightRef, ...]:
        """Validate exact quarantined versions for one isolated test invocation.

        This is deliberately separate from :meth:`eligible_references`: a
        quarantined hypothesis may be named for a controlled test, but remains
        unavailable to normal retrieval.  The returned tuple is an immutable
        value snapshot in caller-supplied order; no entry or mutable bank state
        is exposed.
        """

        assigned = self._owned_reference_sequence(
            references,
            name="quarantine_test_insights",
        )
        operator = _operator_kind_token(operator_kind)
        paths = (
            None
            if editable_paths is None
            else _canonical_paths(editable_paths, name="editable_paths")
        )
        wrong_lifecycle = tuple(
            reference
            for reference in assigned
            if self._entries[reference].lifecycle_state
            is not InsightLifecycleState.QUARANTINED
        )
        if wrong_lifecycle:
            raise ValueError(
                "quarantine_test_insights must contain only quarantined insight "
                "references"
            )
        inapplicable = tuple(
            reference
            for reference in assigned
            if not self._structurally_applicable(
                self._entries[reference], operator, paths
            )
        )
        if inapplicable:
            raise QuarantineAssignmentStructuralError(
                "quarantine_test_insights contains a structurally inapplicable "
                "insight reference"
            )
        return assigned

    def admit_quarantine_test_assignment(
        self,
        references: Sequence[InsightRef],
        *,
        operator_kind: str,
        source_admission_request_sha256: str,
        editable_paths: Sequence[str] | None = None,
    ) -> QuarantineTestAdmissionReceipt:
        """Issue one exact diagnostic authority and retain it for later joins."""

        assigned = self.validate_quarantine_test_assignment(
            references,
            operator_kind=operator_kind,
            editable_paths=editable_paths,
        )
        if any(
            reference in self._quarantine_admission_by_reference
            for reference in assigned
        ):
            raise ValueError("a quarantine insight was already admitted for testing")
        receipt = QuarantineTestAdmissionReceipt(
            references=tuple(sorted(assigned)),
            operator_kind=_operator_kind_token(operator_kind),
            editable_paths=(
                ()
                if editable_paths is None
                else _canonical_paths(editable_paths, name="editable_paths")
            ),
            source_admission_request_sha256=source_admission_request_sha256,
            memory_trial_count_cutoff=len(self._trials),
        )
        if receipt.receipt_sha256 in self._quarantine_admissions:
            raise ValueError("quarantine admission receipt identity collided")
        self._quarantine_admissions[receipt.receipt_sha256] = receipt
        for reference in receipt.references:
            self._quarantine_admission_by_reference[reference] = receipt.receipt_sha256
        return receipt

    def validate_quarantine_test_admission(
        self,
        receipt: QuarantineTestAdmissionReceipt,
        *,
        eligible_references: Sequence[InsightRef],
        subset_authorization_sha256: str | None = None,
    ) -> tuple[InsightRef, ...]:
        """Revalidate bank issuance, lifecycle, and the randomized catalog.

        By default the still-active admitted cohort must equal the assignment
        cohort exactly.  A prospectively selected subset is permitted only
        when its external policy receipt is explicitly bound into memory
        credit.  This preserves the original anti-cherry-picking default while
        allowing parent-local positivity filtering to happen before dispatch.
        """

        if type(receipt) is not QuarantineTestAdmissionReceipt:
            raise TypeError("receipt must be an exact QuarantineTestAdmissionReceipt")
        QuarantineTestAdmissionReceipt.__post_init__(receipt)
        issued = self._quarantine_admissions.get(receipt.receipt_sha256)
        if issued != receipt:
            raise ValueError("quarantine admission was not issued by this memory bank")
        if subset_authorization_sha256 is not None:
            require_sha256(
                subset_authorization_sha256,
                "subset_authorization_sha256",
            )
        eligible = self._owned_reference_sequence(
            eligible_references,
            name="eligible_references",
        )
        quarantined = tuple(
            sorted(
                reference
                for reference in eligible
                if self._entries[reference].lifecycle_state
                is InsightLifecycleState.QUARANTINED
            )
        )
        currently_quarantined_admitted = tuple(
            reference
            for reference in receipt.references
            if self._entries[reference].lifecycle_state
            is InsightLifecycleState.QUARANTINED
        )
        if subset_authorization_sha256 is None:
            if quarantined != currently_quarantined_admitted:
                raise ValueError(
                    "eligible quarantine cards differ from the still-active references "
                    "of the issued admission"
                )
        elif not quarantined or not set(quarantined).issubset(
            currently_quarantined_admitted
        ):
            raise ValueError(
                "authorized quarantine subset must be a non-empty subset of the "
                "still-active issued admission"
            )
        if any(
            self._quarantine_admission_by_reference.get(reference)
            != receipt.receipt_sha256
            for reference in quarantined
        ):
            raise ValueError("quarantine admission reference index changed")
        return quarantined

    def quarantine_test_admission_receipt(
        self,
        receipt_sha256: str,
    ) -> QuarantineTestAdmissionReceipt:
        """Return one immutable bank-issued diagnostic authority by exact hash."""

        if (
            type(receipt_sha256) is not str
            or _LOWER_SHA256.fullmatch(receipt_sha256) is None
        ):
            raise ValueError("receipt_sha256 must be a lowercase SHA-256 ID")
        try:
            return self._quarantine_admissions[receipt_sha256]
        except KeyError as exc:
            raise ValueError(
                "quarantine admission was not issued by this memory bank"
            ) from exc

    def _owned_reference_sequence(
        self,
        references: Sequence[InsightRef],
        *,
        name: str,
    ) -> tuple[InsightRef, ...]:
        if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
            raise TypeError(f"{name} must be a sequence of InsightRef values")
        values = tuple(references)
        if any(not isinstance(reference, InsightRef) for reference in values):
            raise TypeError(f"{name} must contain InsightRef values")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} cannot contain duplicates")
        if any(reference not in self._entries for reference in values):
            raise ValueError(f"{name} contains a foreign insight reference")
        return values

    def score_snapshot(
        self,
        context_hash: str,
        *,
        eligible_references: Sequence[InsightRef] | None = None,
    ) -> Mapping[InsightRef, float]:
        eligible = self._eligible_reference_subset(eligible_references)
        return {
            reference: self._scores.get(
                (context_hash, reference), self._entries[reference].initial_score
            )
            for reference in eligible
        }

    def select(
        self,
        *,
        context_hash: str,
        subset_size: int,
        rng,
        exploration_probability: Fraction | None = None,
        score_context_hash: str | None = None,
        eligible_references: Sequence[InsightRef] | None = None,
    ) -> InsightSelectionDecision:
        eligible = self._eligible_reference_subset(eligible_references)
        selector = (
            self._selector
            if exploration_probability is None
            else EpsilonGreedySubsetSelector(exploration_probability)
        )
        return selector.select(
            context_hash=context_hash,
            eligible=eligible,
            scores=self.score_snapshot(
                score_context_hash or context_hash,
                eligible_references=eligible,
            ),
            subset_size=min(subset_size, len(eligible)),
            rng=rng,
        )

    def prompt_records(
        self, references: Sequence[InsightRef]
    ) -> tuple[dict[str, object], ...]:
        """Return detached, JSON-ready records for exact owned references.

        Every mapping and nested container is freshly allocated.  Mutating a
        returned prompt record therefore cannot mutate lifecycle, provenance,
        scores, or any other state held by this bank.
        """

        owned = self._owned_reference_sequence(references, name="references")
        records = []
        for reference in owned:
            entry = self._entries[reference]
            record: dict[str, object] = {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
                "claim": entry.draft.claim,
                "trigger": entry.draft.trigger,
                "mechanism": entry.draft.mechanism,
                "evidence_summary": entry.draft.evidence_summary,
                "affected_paths": list(entry.draft.affected_paths),
                "confidence_at_creation": entry.draft.confidence,
                "lifecycle_state": entry.lifecycle_state.value,
                "origin": entry.origin.value,
                "semantic_relations": [
                    {
                        "kind": relation.kind.value,
                        "target_insight_id": relation.target.insight_id.value,
                        "target_version": relation.target.version,
                        "note": relation.note,
                    }
                    for relation in entry.relations
                ],
            }
            intervention = entry.draft.intervention_record()
            if intervention is not None:
                record.update(intervention)
            semantic = entry.draft.semantic_record()
            if semantic is not None:
                record.update(semantic)
            records.append(record)
        return tuple(records)

    def selected_prompt_records(
        self, decision: InsightSelectionDecision
    ) -> tuple[dict[str, object], ...]:
        """Backward-compatible prompt projection for retrieval decisions."""

        return self.prompt_records(decision.selected)

    def record_trial(
        self,
        *,
        credit_unit_id: OperatorInvocationId,
        candidate_ids: tuple[CandidateId, ...],
        reward_definition_hash: str,
        decision: InsightSelectionDecision,
        reward: float,
    ) -> InsightTrial:
        trial = InsightTrial(
            credit_unit_id=credit_unit_id,
            candidate_ids=candidate_ids,
            reward_definition_hash=reward_definition_hash,
            decision=decision,
            reward=float(reward),
        )
        return self.record_trials_batch((trial,))[0]

    def _prepare_trials_batch(
        self,
        trials: tuple[InsightTrial, ...],
    ) -> tuple[tuple[InsightTrial, ...], dict[tuple[str, InsightRef], float]]:
        if type(trials) is not tuple or not trials:
            raise ValueError("trials must be a non-empty exact tuple")
        if any(type(trial) is not InsightTrial for trial in trials):
            raise TypeError("trials must contain exact InsightTrial values")
        for trial in trials:
            InsightTrial.__post_init__(trial)
            self.entries_for(trial.decision.eligible)

        canonical = tuple(sorted(trials, key=lambda trial: trial.credit_unit_id.value))
        proposed = (*self._trials, *canonical)
        contexts = tuple(sorted({trial.decision.context_hash for trial in canonical}))

        # Estimator validation also rejects duplicate credit/candidate units
        # across the existing bank and the complete incoming batch.  Compute
        # all score mutations before touching either mutable collection.
        for trial in canonical:
            for reference in trial.decision.eligible:
                estimate_marginal_effect(
                    proposed,
                    reference,
                    context_hash=trial.decision.context_hash,
                )
        score_updates: dict[tuple[str, InsightRef], float] = {}
        for context_hash in contexts:
            score_updates.update(self._context_score_updates(proposed, context_hash))

        return canonical, score_updates

    def preview_trials_batch(
        self,
        trials: tuple[InsightTrial, ...],
    ) -> tuple[InsightTrial, ...]:
        """Validate and canonicalize a prospective batch without mutation."""

        canonical, _ = self._prepare_trials_batch(trials)
        return canonical

    def record_trials_batch(
        self,
        trials: tuple[InsightTrial, ...],
    ) -> tuple[InsightTrial, ...]:
        """Atomically publish a canonical batch of completed credit units.

        Concurrent campaign arms must not update retrieval scores as each arm
        happens to finish.  This boundary validates the complete proposed
        state, computes every affected score against that same state, and only
        then publishes trials in stable credit-unit order.  The returned tuple
        is therefore independent of caller/gather completion order.
        """

        canonical, score_updates = self._prepare_trials_batch(trials)

        self._trials.extend(canonical)
        self._scores.update(score_updates)
        return canonical

    def _refresh_context_scores(self, context_hash: str) -> None:
        self._scores.update(
            self._context_score_updates(tuple(self._trials), context_hash)
        )

    def _context_score_updates(
        self,
        trials: Sequence[InsightTrial],
        context_hash: str,
    ) -> dict[tuple[str, InsightRef], float]:
        updates: dict[tuple[str, InsightRef], float] = {}
        for reference, entry in self._entries.items():
            estimate = estimate_marginal_effect(
                trials,
                reference,
                context_hash=context_hash,
            )
            if not estimate.identified:
                continue
            support = min(
                estimate.treated_effective_sample_size,
                estimate.control_effective_sample_size,
            )
            shrinkage = support / (support + self._shrinkage_ess)
            updates[(context_hash, reference)] = (
                entry.initial_score + shrinkage * float(estimate.effect)
            )
        return updates

    def score_evidence(self, context_hash: str) -> tuple[dict[str, object], ...]:
        evidence = []
        for reference, entry in sorted(self._entries.items()):
            estimate = estimate_marginal_effect(
                self._trials,
                reference,
                context_hash=context_hash,
            )
            evidence.append(
                {
                    "insight_id": reference.insight_id.value,
                    "version": reference.version,
                    "lifecycle_state": entry.lifecycle_state.value,
                    "retrievable": entry.retrievable,
                    "retrieval_score": self._scores.get(
                        (context_hash, reference), entry.initial_score
                    ),
                    "effect": estimate.effect,
                    "treated_trials": estimate.treated_trials,
                    "control_trials": estimate.control_trials,
                    "treated_ess": estimate.treated_effective_sample_size,
                    "control_ess": estimate.control_effective_sample_size,
                    "identified": estimate.identified,
                }
            )
        return tuple(evidence)


def context_stratum_hash(*, problem_id: str, operator_kind: str, phase: str) -> str:
    payload = f"{problem_id}\x00{operator_kind}\x00{phase}".encode(
        "utf-8", errors="strict"
    )
    return hashlib.sha256(b"agent-evolve:insight-context:v1\x00" + payload).hexdigest()


__all__ = [
    "EmpiricalEvidenceSnapshot",
    "InsightEvidenceLineage",
    "InsightLifecycleChangeRequest",
    "InsightLifecycleState",
    "InsightLifecycleTransition",
    "InsightMemoryBank",
    "InsightMemoryEntry",
    "InsightOrigin",
    "InsightRelation",
    "InsightRelationKind",
    "QuarantineAssignmentStructuralError",
    "QuarantineTestAdmissionReceipt",
    "ReflectedInsightBatchItem",
    "compose_epistemic_prompt_payload",
    "context_stratum_hash",
]
