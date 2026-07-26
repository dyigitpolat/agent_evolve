"""Authenticated global falsification for reflected optimization hypotheses.

The gate in this module is deliberately provider- and workload-neutral.  A
workload-owned matcher interprets typed trigger predicates and intervention
signatures.  The engine then joins that classification to every authenticated
observation visible at a sealed cutoff, applies exact metric predictions, and
issues an immutable receipt.

Semantic claim auditing and memory-treatment causality are intentionally
different estimands.  Candidate-level contrasts may support or falsify a
hypothesis, but every observation remains attached to its enclosing wave or
portfolio ITT unit.  Audit receipts contain no reward and cannot assign causal
credit to an individual insight.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_PATH = re.compile(r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$")
_TEXT_LIMIT = 4_096
_MAX_EVENT_INDEX = (1 << 63) - 1

_PREDICATE_DOMAIN = b"agent-evolve:typed-hypothesis-predicate:v1\x00"
_INTERVENTION_DOMAIN = b"agent-evolve:typed-intervention-signature:v1\x00"
_SCOPE_DOMAIN = b"agent-evolve:global-hypothesis-audit-scope:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:global-hypothesis-audit-request:v1\x00"
_BOUNDARY_DOMAIN = b"agent-evolve:hypothesis-evidence-itt-boundary:v1\x00"
_OBSERVATION_DOMAIN = b"agent-evolve:global-hypothesis-observation:v3\x00"
_REGISTRY_DOMAIN = b"agent-evolve:global-hypothesis-registry:v1\x00"
_MATCH_DOMAIN = b"agent-evolve:hypothesis-evidence-match:v1\x00"
_CLUSTER_DOMAIN = b"agent-evolve:hypothesis-effective-cluster:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:global-hypothesis-audit-receipt:v1\x00"
_REVISION_DOMAIN = b"agent-evolve:append-only-hypothesis-revision:v1\x00"


def _hash_record(domain: bytes, record: object) -> str:
    return hashlib.sha256(
        domain + canonical_typed_json_bytes(freeze_json(record))
    ).hexdigest()


GLOBAL_FALSIFICATION_POLICY_ID = "global_hypothesis_falsification"
GLOBAL_FALSIFICATION_POLICY_VERSION = 2
GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256 = _hash_record(
    b"agent-evolve:global-hypothesis-falsification-policy:v2\x00",
    {
        "policy_id": GLOBAL_FALSIFICATION_POLICY_ID,
        "policy_version": GLOBAL_FALSIFICATION_POLICY_VERSION,
        "join": "exhaustive_authenticated_registry_at_inclusive_event_cutoff",
        "repeat_rule": "earliest_event_then_source_id_per_configuration_evaluator",
        "support_rule": "all_required_known_directions_and_magnitudes_match",
        "counterexample_rule": "any_required_direction_or_magnitude_mismatch",
        "necessity_rule": "matching_off_trigger_response_contradicts_necessity",
        "cluster_rule": (
            "instance_evaluator_campaign_lineage_factorial_block_itt_unit"
        ),
        "support_temporality_rule": (
            "declared_support_thresholds_must_be_met_by_post_origin_evidence"
        ),
        "mechanism_temporality_rule": (
            "mechanistic_support_requires_post_origin_identifying_evidence"
        ),
        "causal_credit_rule": "semantic_audit_only_no_retroactive_card_credit",
    },
)


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _require_metric(value: str, *, name: str = "metric_id") -> None:
    if type(value) is not str or _METRIC.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed metric identifier grammar")


def _require_event_index(value: int, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_EVENT_INDEX:
        raise ValueError(f"{name} must be an exact non-negative int63")


def _require_canonical_text(value: str, *, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _TEXT_LIMIT
    ):
        raise ValueError(f"{name} must be bounded canonical non-empty text")


def _require_sha_tuple(values: tuple[str, ...], *, name: str, empty: bool) -> None:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError(f"{name} must be an exact tuple of SHA-256 strings")
    for value in values:
        require_sha256(value, name)
    if not empty and not values:
        raise ValueError(f"{name} cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _require_frozen_object(value: FrozenJsonObject, *, name: str) -> None:
    if type(value) is not FrozenJsonObject:
        raise TypeError(f"{name} must be an exact FrozenJsonObject")
    if freeze_json(value) is not value:
        raise TypeError(f"{name} must already be frozen typed JSON")


@dataclass(frozen=True, slots=True)
class TypedEvidencePredicate:
    """Workload-owned, replayable predicate with engine-authenticated identity."""

    schema_id: str
    schema_version: int
    schema_definition_sha256: str
    payload: FrozenJsonObject
    predicate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.schema_id, name="schema_id")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer")
        require_sha256(self.schema_definition_sha256, "schema_definition_sha256")
        _require_frozen_object(self.payload, name="payload")
        object.__setattr__(
            self,
            "predicate_sha256",
            _hash_record(_PREDICATE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "predicate_schema_id": self.schema_id,
            "predicate_schema_version": self.schema_version,
            "predicate_schema_definition_sha256": self.schema_definition_sha256,
            "payload": thaw_json(self.payload),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "predicate_sha256": self.predicate_sha256}


@dataclass(frozen=True, slots=True)
class TypedInterventionSignature:
    """Exact semantic edit requested by a sealed hypothesis."""

    affected_paths: tuple[str, ...]
    old_value_predicate: TypedEvidencePredicate
    new_action: TypedEvidencePredicate
    admissible_operator_families: tuple[str, ...]
    signature_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.affected_paths) is not tuple or not self.affected_paths:
            raise ValueError("affected_paths must be a non-empty exact tuple")
        if any(
            type(path) is not str or _PATH.fullmatch(path) is None
            for path in self.affected_paths
        ):
            raise ValueError("affected_paths must contain canonical JSON paths")
        if self.affected_paths != tuple(sorted(set(self.affected_paths))):
            raise ValueError("affected_paths must be unique and canonically sorted")
        if type(self.old_value_predicate) is not TypedEvidencePredicate:
            raise TypeError("old_value_predicate must be exact TypedEvidencePredicate")
        if type(self.new_action) is not TypedEvidencePredicate:
            raise TypeError("new_action must be exact TypedEvidencePredicate")
        if (
            type(self.admissible_operator_families) is not tuple
            or not self.admissible_operator_families
        ):
            raise ValueError("admissible_operator_families must be non-empty")
        for family in self.admissible_operator_families:
            _require_token(family, name="admissible_operator_families")
        if self.admissible_operator_families != tuple(
            sorted(set(self.admissible_operator_families))
        ):
            raise ValueError(
                "admissible_operator_families must be unique and canonically sorted"
            )
        object.__setattr__(
            self,
            "signature_sha256",
            _hash_record(_INTERVENTION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "affected_paths": list(self.affected_paths),
            "old_value_predicate_sha256": self.old_value_predicate.predicate_sha256,
            "new_action_sha256": self.new_action.predicate_sha256,
            "admissible_operator_families": list(self.admissible_operator_families),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "signature_sha256": self.signature_sha256}


@dataclass(frozen=True, slots=True)
class HypothesisMetricPrediction:
    """One required direction and optional child-minus-parent magnitude interval."""

    metric_id: str
    direction: MetricEffectDirection
    minimum_delta: float | None = None
    maximum_delta: float | None = None

    def __post_init__(self) -> None:
        _require_metric(self.metric_id)
        if (
            type(self.direction) is not MetricEffectDirection
            or self.direction is MetricEffectDirection.UNKNOWN
        ):
            raise ValueError("direction must be a known MetricEffectDirection")
        if (self.minimum_delta is None) != (self.maximum_delta is None):
            raise ValueError("magnitude claims require both interval endpoints")
        if self.minimum_delta is not None:
            if (
                type(self.minimum_delta) is not float
                or type(self.maximum_delta) is not float
                or not math.isfinite(self.minimum_delta)
                or not math.isfinite(self.maximum_delta)
            ):
                raise TypeError("magnitude endpoints must be finite canonical floats")
            if self.minimum_delta > self.maximum_delta:
                raise ValueError("minimum_delta cannot exceed maximum_delta")

    @property
    def has_magnitude_claim(self) -> bool:
        return self.minimum_delta is not None

    def to_record(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "direction": self.direction.value,
            "minimum_delta_hex": (
                None if self.minimum_delta is None else self.minimum_delta.hex()
            ),
            "maximum_delta_hex": (
                None if self.maximum_delta is None else self.maximum_delta.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class HypothesisClaimStrength:
    """Typed logical flags; prose is never parsed by the gate."""

    sufficiency: bool = True
    necessity: bool = False
    invariance: bool = False
    mechanistic_or_causal: bool = False

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, name)) is not bool
            for name in (
                "sufficiency",
                "necessity",
                "invariance",
                "mechanistic_or_causal",
            )
        ):
            raise TypeError("logical strength flags must be exact booleans")
        if not any(
            (
                self.sufficiency,
                self.necessity,
                self.invariance,
                self.mechanistic_or_causal,
            )
        ):
            raise ValueError("a hypothesis must declare at least one logical claim")

    def to_record(self) -> dict[str, bool]:
        return {
            "sufficiency": self.sufficiency,
            "necessity": self.necessity,
            "invariance": self.invariance,
            "mechanistic_or_causal": self.mechanistic_or_causal,
        }


@dataclass(frozen=True, slots=True)
class HypothesisAuditScope:
    """Closed workload/evaluator/campaign scope for one global join."""

    workload_instance_sha256s: tuple[str, ...]
    evaluator_contract_sha256: str
    metric_adjudicator_definition_sha256: str
    campaign_sha256s: tuple[str, ...] = ()
    cross_instance_claim: bool = False
    scope_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha_tuple(
            self.workload_instance_sha256s,
            name="workload_instance_sha256s",
            empty=False,
        )
        require_sha256(self.evaluator_contract_sha256, "evaluator_contract_sha256")
        require_sha256(
            self.metric_adjudicator_definition_sha256,
            "metric_adjudicator_definition_sha256",
        )
        _require_sha_tuple(self.campaign_sha256s, name="campaign_sha256s", empty=True)
        if type(self.cross_instance_claim) is not bool:
            raise TypeError("cross_instance_claim must be an exact boolean")
        if self.cross_instance_claim and len(self.workload_instance_sha256s) < 2:
            raise ValueError("cross-instance claims require at least two instances")
        object.__setattr__(
            self, "scope_sha256", _hash_record(_SCOPE_DOMAIN, self._unsigned_record())
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "workload_instance_sha256s": list(self.workload_instance_sha256s),
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "metric_adjudicator_definition_sha256": (
                self.metric_adjudicator_definition_sha256
            ),
            "campaign_sha256s": list(self.campaign_sha256s),
            "cross_instance_claim": self.cross_instance_claim,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "scope_sha256": self.scope_sha256}


@dataclass(frozen=True, slots=True)
class GlobalHypothesisAuditRequest:
    """Sealed hypothesis, evidence universe, and predeclared support threshold."""

    reference: InsightRef
    draft_content_sha256: str
    trigger: TypedEvidencePredicate
    intervention: TypedInterventionSignature
    predictions: tuple[HypothesisMetricPrediction, ...]
    claim_strength: HypothesisClaimStrength
    scope: HypothesisAuditScope
    matcher_definition_sha256: str
    origin_cutoff_event_index: int
    audit_cutoff_event_index: int
    registry_snapshot_sha256: str
    minimum_support_clusters: int = 2
    minimum_support_instances: int = 1
    audit_policy_definition_sha256: str = GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.draft_content_sha256, "draft_content_sha256")
        if type(self.trigger) is not TypedEvidencePredicate:
            raise TypeError("trigger must be exact TypedEvidencePredicate")
        if type(self.intervention) is not TypedInterventionSignature:
            raise TypeError("intervention must be exact TypedInterventionSignature")
        if (
            type(self.predictions) is not tuple
            or not self.predictions
            or any(
                type(value) is not HypothesisMetricPrediction
                for value in self.predictions
            )
        ):
            raise ValueError("predictions must contain typed metric predictions")
        metric_ids = tuple(value.metric_id for value in self.predictions)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("predictions must use unique canonical metric order")
        if type(self.claim_strength) is not HypothesisClaimStrength:
            raise TypeError("claim_strength must be exact HypothesisClaimStrength")
        if type(self.scope) is not HypothesisAuditScope:
            raise TypeError("scope must be exact HypothesisAuditScope")
        require_sha256(self.matcher_definition_sha256, "matcher_definition_sha256")
        _require_event_index(
            self.origin_cutoff_event_index, name="origin_cutoff_event_index"
        )
        _require_event_index(
            self.audit_cutoff_event_index, name="audit_cutoff_event_index"
        )
        if self.origin_cutoff_event_index > self.audit_cutoff_event_index:
            raise ValueError("origin cutoff cannot follow the audit cutoff")
        require_sha256(self.registry_snapshot_sha256, "registry_snapshot_sha256")
        for name in ("minimum_support_clusters", "minimum_support_instances"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        require_sha256(
            self.audit_policy_definition_sha256,
            "audit_policy_definition_sha256",
        )
        if self.scope.cross_instance_claim and self.minimum_support_instances < 2:
            raise ValueError(
                "cross-instance claims require support from at least two instances"
            )
        object.__setattr__(
            self,
            "request_sha256",
            _hash_record(_REQUEST_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "draft_content_sha256": self.draft_content_sha256,
            "trigger_predicate_sha256": self.trigger.predicate_sha256,
            "intervention_signature_sha256": self.intervention.signature_sha256,
            "predictions": [value.to_record() for value in self.predictions],
            "claim_strength": self.claim_strength.to_record(),
            "scope_sha256": self.scope.scope_sha256,
            "matcher_definition_sha256": self.matcher_definition_sha256,
            "origin_cutoff_event_index": self.origin_cutoff_event_index,
            "audit_cutoff_event_index": self.audit_cutoff_event_index,
            "registry_snapshot_sha256": self.registry_snapshot_sha256,
            "minimum_support_clusters": self.minimum_support_clusters,
            "minimum_support_instances": self.minimum_support_instances,
            "audit_policy_definition_sha256": (self.audit_policy_definition_sha256),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


class EvidenceProvenance(str, Enum):
    DIRECT_MUTATION = "direct_mutation"
    RECOMBINATION_EXACT_ABLATION = "recombination_exact_ablation"
    FULL_SLATE_DIAGNOSTIC = "full_slate_diagnostic"
    RANDOMIZED_ADMINISTRATION = "randomized_administration"
    OBSERVATIONAL_ASSOCIATION = "observational_association"


class InterventionIdentifiability(str, Enum):
    EXACT_SINGLE = "exact_single_intervention"
    JOINT_WITHOUT_ABLATION = "joint_intervention_without_exact_ablation"
    AMBIGUOUS = "ambiguous_intervention_identity"


class CausalEstimandUnit(str, Enum):
    WAVE = "wave"
    PORTFOLIO = "portfolio"


@dataclass(frozen=True, slots=True)
class EvidenceCausalBoundary:
    """Enclosing prospective ITT unit; never an individual reflected card."""

    wave_sha256: str
    estimand_unit: CausalEstimandUnit
    portfolio_sha256: str | None = None
    prospective_assignment_sha256: str | None = None
    boundary_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.wave_sha256, "wave_sha256")
        if type(self.estimand_unit) is not CausalEstimandUnit:
            raise TypeError("estimand_unit must be exact CausalEstimandUnit")
        if self.portfolio_sha256 is not None:
            require_sha256(self.portfolio_sha256, "portfolio_sha256")
        if (
            self.estimand_unit is CausalEstimandUnit.PORTFOLIO
            and self.portfolio_sha256 is None
        ):
            raise ValueError("a portfolio estimand requires portfolio_sha256")
        if self.prospective_assignment_sha256 is not None:
            require_sha256(
                self.prospective_assignment_sha256,
                "prospective_assignment_sha256",
            )
        object.__setattr__(
            self,
            "boundary_sha256",
            _hash_record(_BOUNDARY_DOMAIN, self._unsigned_record()),
        )

    @property
    def estimand_unit_sha256(self) -> str:
        if self.estimand_unit is CausalEstimandUnit.PORTFOLIO:
            assert self.portfolio_sha256 is not None
            return self.portfolio_sha256
        return self.wave_sha256

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "wave_sha256": self.wave_sha256,
            "estimand_unit": self.estimand_unit.value,
            "portfolio_sha256": self.portfolio_sha256,
            "prospective_assignment_sha256": self.prospective_assignment_sha256,
            "credit_boundary": "semantic_audit_only_no_per_card_credit",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "boundary_sha256": self.boundary_sha256}


@dataclass(frozen=True, slots=True)
class ObservedMetricEffect:
    metric_id: str
    direction: MetricEffectDirection
    delta: float
    adjudicator_definition_sha256: str

    def __post_init__(self) -> None:
        _require_metric(self.metric_id)
        if (
            type(self.direction) is not MetricEffectDirection
            or self.direction is MetricEffectDirection.UNKNOWN
        ):
            raise ValueError("direction must be a known MetricEffectDirection")
        if type(self.delta) is not float or not math.isfinite(self.delta):
            raise TypeError("delta must be a finite canonical float")
        require_sha256(
            self.adjudicator_definition_sha256,
            "adjudicator_definition_sha256",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "direction": self.direction.value,
            "delta_hex": self.delta.hex(),
            "adjudicator_definition_sha256": self.adjudicator_definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedHypothesisObservation:
    """One immutable parent/child fact available to the global evidence join."""

    source_evidence_id: str
    event_index: int
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    campaign_sha256: str
    parent_candidate_id: CandidateId
    child_candidate_id: CandidateId
    operator_invocation_id: OperatorInvocationId
    finite_contract_identity_sha256: str
    provenance: EvidenceProvenance
    causal_boundary: EvidenceCausalBoundary
    parent_configuration: FrozenJsonObject
    child_configuration: FrozenJsonObject
    parent_configuration_sha256: str
    child_configuration_sha256: str
    parent_outcome_sha256: str
    child_outcome_sha256: str
    operator_family: str
    affected_paths: tuple[str, ...]
    observed_action: FrozenJsonObject
    action_semantics_compiler_id: str
    action_semantics_compiler_version: int
    action_semantics_definition_sha256: str
    intervention_identifiability: InterventionIdentifiability
    metrics: tuple[ObservedMetricEffect, ...]
    lineage_cluster_sha256: str
    factorial_block_sha256: str
    mechanism_identifying_design: bool = False
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.source_evidence_id, "source_evidence_id")
        _require_event_index(self.event_index, name="event_index")
        for name in (
            "workload_instance_sha256",
            "evaluator_contract_sha256",
            "campaign_sha256",
            "finite_contract_identity_sha256",
            "parent_configuration_sha256",
            "child_configuration_sha256",
            "parent_outcome_sha256",
            "child_outcome_sha256",
            "lineage_cluster_sha256",
            "factorial_block_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be an exact CandidateId")
        if type(self.child_candidate_id) is not CandidateId:
            raise TypeError("child_candidate_id must be an exact CandidateId")
        if type(self.operator_invocation_id) is not OperatorInvocationId:
            raise TypeError(
                "operator_invocation_id must be an exact OperatorInvocationId"
            )
        CandidateId.__post_init__(self.parent_candidate_id)
        CandidateId.__post_init__(self.child_candidate_id)
        OperatorInvocationId.__post_init__(self.operator_invocation_id)
        if self.parent_candidate_id == self.child_candidate_id:
            raise ValueError(
                "authenticated child occurrence cannot reuse its parent ID"
            )
        if type(self.provenance) is not EvidenceProvenance:
            raise TypeError("provenance must be exact EvidenceProvenance")
        if type(self.causal_boundary) is not EvidenceCausalBoundary:
            raise TypeError("causal_boundary must be exact EvidenceCausalBoundary")
        _require_frozen_object(self.parent_configuration, name="parent_configuration")
        _require_frozen_object(self.child_configuration, name="child_configuration")
        if (
            _hash_record(
                b"agent-evolve:configuration:v1\x00",
                thaw_json(self.parent_configuration),
            )
            != self.parent_configuration_sha256
        ):
            raise ValueError("parent_configuration_sha256 does not match configuration")
        if (
            _hash_record(
                b"agent-evolve:configuration:v1\x00",
                thaw_json(self.child_configuration),
            )
            != self.child_configuration_sha256
        ):
            raise ValueError("child_configuration_sha256 does not match configuration")
        _require_token(self.operator_family, name="operator_family")
        if type(self.affected_paths) is not tuple or not self.affected_paths:
            raise ValueError("affected_paths must be a non-empty exact tuple")
        if any(
            type(path) is not str or _PATH.fullmatch(path) is None
            for path in self.affected_paths
        ):
            raise ValueError("affected_paths must contain canonical JSON paths")
        if self.affected_paths != tuple(sorted(set(self.affected_paths))):
            raise ValueError("affected_paths must be unique and canonically sorted")
        _require_frozen_object(self.observed_action, name="observed_action")
        _require_token(
            self.action_semantics_compiler_id,
            name="action_semantics_compiler_id",
        )
        if (
            type(self.action_semantics_compiler_version) is not int
            or self.action_semantics_compiler_version <= 0
        ):
            raise ValueError("action_semantics_compiler_version must be positive")
        require_sha256(
            self.action_semantics_definition_sha256,
            "action_semantics_definition_sha256",
        )
        if type(self.intervention_identifiability) is not InterventionIdentifiability:
            raise TypeError(
                "intervention_identifiability must be exact InterventionIdentifiability"
            )
        if (
            type(self.metrics) is not tuple
            or not self.metrics
            or any(type(value) is not ObservedMetricEffect for value in self.metrics)
        ):
            raise ValueError("metrics must contain exact ObservedMetricEffect values")
        metric_ids = tuple(value.metric_id for value in self.metrics)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("metrics must use unique canonical metric order")
        if type(self.mechanism_identifying_design) is not bool:
            raise TypeError("mechanism_identifying_design must be an exact boolean")
        if self.mechanism_identifying_design and self.provenance is not (
            EvidenceProvenance.RANDOMIZED_ADMINISTRATION
        ):
            raise ValueError(
                "mechanism-identifying evidence must be randomized administration"
            )
        if (
            self.provenance is EvidenceProvenance.RANDOMIZED_ADMINISTRATION
            and self.causal_boundary.prospective_assignment_sha256 is None
        ):
            raise ValueError(
                "randomized administration requires a prospective ITT assignment"
            )
        object.__setattr__(
            self,
            "observation_sha256",
            _hash_record(_OBSERVATION_DOMAIN, self._unsigned_record()),
        )

    @classmethod
    def configuration_sha256(cls, value: FrozenJsonObject) -> str:
        _require_frozen_object(value, name="configuration")
        return _hash_record(b"agent-evolve:configuration:v1\x00", thaw_json(value))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "source_evidence_id": self.source_evidence_id,
            "event_index": self.event_index,
            "workload_instance_sha256": self.workload_instance_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "campaign_sha256": self.campaign_sha256,
            "parent_candidate_id": self.parent_candidate_id.value,
            "child_candidate_id": self.child_candidate_id.value,
            "operator_invocation_id": self.operator_invocation_id.value,
            "finite_contract_identity_sha256": (self.finite_contract_identity_sha256),
            "provenance": self.provenance.value,
            "causal_boundary": self.causal_boundary.to_record(),
            "parent_configuration": thaw_json(self.parent_configuration),
            "child_configuration": thaw_json(self.child_configuration),
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "parent_outcome_sha256": self.parent_outcome_sha256,
            "child_outcome_sha256": self.child_outcome_sha256,
            "operator_family": self.operator_family,
            "affected_paths": list(self.affected_paths),
            "observed_action": thaw_json(self.observed_action),
            "action_semantics_compiler": {
                "compiler_id": self.action_semantics_compiler_id,
                "compiler_version": self.action_semantics_compiler_version,
                "definition_sha256": self.action_semantics_definition_sha256,
            },
            "intervention_identifiability": self.intervention_identifiability.value,
            "metrics": [value.to_record() for value in self.metrics],
            "lineage_cluster_sha256": self.lineage_cluster_sha256,
            "factorial_block_sha256": self.factorial_block_sha256,
            "mechanism_identifying_design": self.mechanism_identifying_design,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True, slots=True)
class GlobalEvidenceRegistrySnapshot:
    """Complete immutable observation registry visible through one event index."""

    captured_through_event_index: int
    observations: tuple[AuthenticatedHypothesisObservation, ...]
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_event_index(
            self.captured_through_event_index,
            name="captured_through_event_index",
        )
        if type(self.observations) is not tuple or any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in self.observations
        ):
            raise TypeError(
                "observations must be an exact tuple of authenticated observations"
            )
        keys = tuple(value.source_evidence_id for value in self.observations)
        if keys != tuple(sorted(set(keys))):
            raise ValueError(
                "observations must have unique source IDs in canonical order"
            )
        if any(
            value.event_index > self.captured_through_event_index
            for value in self.observations
        ):
            raise ValueError("registry contains evidence after its capture cutoff")
        object.__setattr__(
            self,
            "snapshot_sha256",
            _hash_record(_REGISTRY_DOMAIN, self._unsigned_record()),
        )

    @classmethod
    def seal(
        cls,
        *,
        captured_through_event_index: int,
        observations: Sequence[AuthenticatedHypothesisObservation],
    ) -> "GlobalEvidenceRegistrySnapshot":
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence")
        if any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in observations
        ):
            raise TypeError(
                "observations must contain exact authenticated observations"
            )
        canonical = tuple(
            sorted(observations, key=lambda value: value.source_evidence_id)
        )
        return cls(captured_through_event_index, canonical)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "captured_through_event_index": self.captured_through_event_index,
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}


class TriggerMatch(str, Enum):
    EXACT = "exact_trigger"
    OFF_TRIGGER = "off_trigger"
    AMBIGUOUS = "ambiguous_trigger"


class InterventionMatch(str, Enum):
    EXACT = "exact_intervention"
    NEAR = "near_intervention"
    DIFFERENT = "different_intervention"
    NON_IDENTIFIABLE = "non_identifiable_intervention"


@dataclass(frozen=True, slots=True)
class HypothesisEvidenceMatchReceipt:
    """Workload matcher result bound to one request and one observation."""

    request_sha256: str
    observation_sha256: str
    trigger_match: TriggerMatch
    intervention_match: InterventionMatch
    matcher_policy_id: str
    matcher_policy_version: int
    matcher_definition_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.observation_sha256, "observation_sha256")
        if type(self.trigger_match) is not TriggerMatch:
            raise TypeError("trigger_match must be exact TriggerMatch")
        if type(self.intervention_match) is not InterventionMatch:
            raise TypeError("intervention_match must be exact InterventionMatch")
        _require_token(self.matcher_policy_id, name="matcher_policy_id")
        if (
            type(self.matcher_policy_version) is not int
            or self.matcher_policy_version <= 0
        ):
            raise ValueError("matcher_policy_version must be a positive integer")
        require_sha256(self.matcher_definition_sha256, "matcher_definition_sha256")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash_record(_MATCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "observation_sha256": self.observation_sha256,
            "trigger_match": self.trigger_match.value,
            "intervention_match": self.intervention_match.value,
            "matcher": {
                "policy_id": self.matcher_policy_id,
                "policy_version": self.matcher_policy_version,
                "definition_sha256": self.matcher_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@runtime_checkable
class GlobalHypothesisEvidenceMatcher(Protocol):
    """Inverted workload seam for typed trigger/intervention matching."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def classify(
        self,
        request: GlobalHypothesisAuditRequest,
        observation: AuthenticatedHypothesisObservation,
    ) -> HypothesisEvidenceMatchReceipt: ...


class EvidenceDisposition(str, Enum):
    SUPPORT = "support"
    COUNTEREXAMPLE = "counterexample"
    OFF_TRIGGER_CONTROL = "off_trigger_control"
    NON_IDENTIFIABLE = "non_identifiable"
    NEAR_INTERVENTION = "near_intervention"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "deduplicated_exact_repeat"
    OUT_OF_SCOPE = "out_of_scope"
    AFTER_CUTOFF = "after_cutoff"


@dataclass(frozen=True, slots=True)
class MetricClaimAssessment:
    metric_id: str
    expected_direction: MetricEffectDirection
    actual_direction: MetricEffectDirection
    actual_delta: float
    direction_matches: bool
    magnitude_matches: bool | None

    @property
    def matches(self) -> bool:
        return self.direction_matches and self.magnitude_matches is not False

    def to_record(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "expected_direction": self.expected_direction.value,
            "actual_direction": self.actual_direction.value,
            "actual_delta_hex": self.actual_delta.hex(),
            "direction_matches": self.direction_matches,
            "magnitude_matches": self.magnitude_matches,
            "matches": self.matches,
        }


@dataclass(frozen=True, slots=True)
class GlobalEvidenceDecision:
    source_evidence_id: str
    observation_sha256: str
    event_index: int
    workload_instance_sha256: str
    itt_estimand_unit_sha256: str
    disposition: EvidenceDisposition
    match_receipt_sha256: str | None
    metric_assessments: tuple[MetricClaimAssessment, ...]
    effective_cluster_sha256: str | None
    post_origin_revision_evidence: bool
    duplicate_of_source_evidence_id: str | None = None
    predictions_match: bool | None = None

    def __post_init__(self) -> None:
        require_sha256(self.source_evidence_id, "source_evidence_id")
        require_sha256(self.observation_sha256, "observation_sha256")
        require_sha256(self.workload_instance_sha256, "workload_instance_sha256")
        require_sha256(self.itt_estimand_unit_sha256, "itt_estimand_unit_sha256")
        _require_event_index(self.event_index, name="event_index")
        if type(self.disposition) is not EvidenceDisposition:
            raise TypeError("disposition must be exact EvidenceDisposition")
        for name in (
            "match_receipt_sha256",
            "effective_cluster_sha256",
            "duplicate_of_source_evidence_id",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if type(self.metric_assessments) is not tuple:
            raise TypeError("metric_assessments must be an exact tuple")
        if type(self.post_origin_revision_evidence) is not bool:
            raise TypeError("post_origin_revision_evidence must be an exact boolean")
        if (
            self.predictions_match is not None
            and type(self.predictions_match) is not bool
        ):
            raise TypeError("predictions_match must be an exact boolean or None")
        if self.disposition is EvidenceDisposition.DUPLICATE:
            if self.duplicate_of_source_evidence_id is None:
                raise ValueError("duplicate evidence must identify its representative")
        elif self.duplicate_of_source_evidence_id is not None:
            raise ValueError("only duplicate evidence may identify a representative")

    def to_record(self) -> dict[str, object]:
        return {
            "source_evidence_id": self.source_evidence_id,
            "observation_sha256": self.observation_sha256,
            "event_index": self.event_index,
            "workload_instance_sha256": self.workload_instance_sha256,
            "itt_estimand_unit_sha256": self.itt_estimand_unit_sha256,
            "disposition": self.disposition.value,
            "match_receipt_sha256": self.match_receipt_sha256,
            "metric_assessments": [
                value.to_record() for value in self.metric_assessments
            ],
            "effective_cluster_sha256": self.effective_cluster_sha256,
            "post_origin_revision_evidence": self.post_origin_revision_evidence,
            "duplicate_of_source_evidence_id": self.duplicate_of_source_evidence_id,
            "predictions_match": self.predictions_match,
        }


class GlobalHypothesisVerdict(str, Enum):
    SUPPORT = "globally_audited_support"
    COUNTEREXAMPLE = "contradicted"
    INSUFFICIENT = "insufficient_evidence"
    NON_IDENTIFIABLE = "non_identifiable"


@dataclass(frozen=True, slots=True)
class GlobalHypothesisAuditReceipt:
    """Closed semantic decision with no causal-memory reward channel."""

    request_sha256: str
    registry_snapshot_sha256: str
    audit_policy_id: str
    audit_policy_version: int
    audit_policy_definition_sha256: str
    verdict: GlobalHypothesisVerdict
    decisions: tuple[GlobalEvidenceDecision, ...]
    support_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...]
    off_trigger_control_ids: tuple[str, ...]
    non_identifiable_ids: tuple[str, ...]
    raw_support_count: int
    effective_support_cluster_count: int
    support_instance_count: int
    untested_workload_instance_sha256s: tuple[str, ...]
    coverage_gaps: tuple[str, ...]
    necessity_contradicted: bool
    mechanism_identified: bool
    lifecycle_decision: str
    audit_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.registry_snapshot_sha256, "registry_snapshot_sha256")
        _require_token(self.audit_policy_id, name="audit_policy_id")
        if type(self.audit_policy_version) is not int or self.audit_policy_version <= 0:
            raise ValueError("audit_policy_version must be a positive exact integer")
        require_sha256(
            self.audit_policy_definition_sha256,
            "audit_policy_definition_sha256",
        )
        if type(self.verdict) is not GlobalHypothesisVerdict:
            raise TypeError("verdict must be exact GlobalHypothesisVerdict")
        if type(self.decisions) is not tuple or any(
            type(value) is not GlobalEvidenceDecision for value in self.decisions
        ):
            raise TypeError(
                "decisions must contain exact GlobalEvidenceDecision values"
            )
        decision_ids = tuple(value.source_evidence_id for value in self.decisions)
        if decision_ids != tuple(sorted(set(decision_ids))):
            raise ValueError("decisions must have unique canonical source order")
        for name in (
            "support_ids",
            "counterexample_ids",
            "off_trigger_control_ids",
            "non_identifiable_ids",
        ):
            _require_sha_tuple(getattr(self, name), name=name, empty=True)
        expected_ids = {
            disposition: tuple(
                value.source_evidence_id
                for value in self.decisions
                if value.disposition is disposition
            )
            for disposition in (
                EvidenceDisposition.SUPPORT,
                EvidenceDisposition.COUNTEREXAMPLE,
                EvidenceDisposition.OFF_TRIGGER_CONTROL,
                EvidenceDisposition.NON_IDENTIFIABLE,
            )
        }
        observed_ids = {
            EvidenceDisposition.SUPPORT: self.support_ids,
            EvidenceDisposition.COUNTEREXAMPLE: self.counterexample_ids,
            EvidenceDisposition.OFF_TRIGGER_CONTROL: self.off_trigger_control_ids,
            EvidenceDisposition.NON_IDENTIFIABLE: self.non_identifiable_ids,
        }
        if observed_ids != expected_ids:
            raise ValueError("receipt evidence ID summaries differ from its decisions")
        for name in (
            "raw_support_count",
            "effective_support_cluster_count",
            "support_instance_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        _require_sha_tuple(
            self.untested_workload_instance_sha256s,
            name="untested_workload_instance_sha256s",
            empty=True,
        )
        support_decisions = tuple(
            value
            for value in self.decisions
            if value.disposition is EvidenceDisposition.SUPPORT
        )
        if self.raw_support_count != len(support_decisions):
            raise ValueError("raw_support_count differs from support decisions")
        if any(value.effective_cluster_sha256 is None for value in support_decisions):
            raise ValueError("support decisions require an effective cluster")
        if self.effective_support_cluster_count != len(
            {value.effective_cluster_sha256 for value in support_decisions}
        ):
            raise ValueError(
                "effective_support_cluster_count differs from support decisions"
            )
        if self.support_instance_count != len(
            {value.workload_instance_sha256 for value in support_decisions}
        ):
            raise ValueError("support_instance_count differs from support decisions")
        if type(self.coverage_gaps) is not tuple:
            raise TypeError("coverage_gaps must be an exact tuple")
        for value in self.coverage_gaps:
            _require_token(value, name="coverage_gaps")
        if self.coverage_gaps != tuple(sorted(set(self.coverage_gaps))):
            raise ValueError("coverage_gaps must be unique and canonically sorted")
        if type(self.necessity_contradicted) is not bool:
            raise TypeError("necessity_contradicted must be an exact boolean")
        if type(self.mechanism_identified) is not bool:
            raise TypeError("mechanism_identified must be an exact boolean")
        _require_canonical_text(self.lifecycle_decision, name="lifecycle_decision")
        object.__setattr__(
            self,
            "audit_receipt_sha256",
            _hash_record(_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    @property
    def causal_credit_updates(self) -> tuple[()]:
        """Truth audits can never retroactively credit an individual card."""

        return ()

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "registry_snapshot_sha256": self.registry_snapshot_sha256,
            "audit_policy": {
                "policy_id": self.audit_policy_id,
                "policy_version": self.audit_policy_version,
                "definition_sha256": self.audit_policy_definition_sha256,
            },
            "verdict": self.verdict.value,
            "decisions": [value.to_record() for value in self.decisions],
            "support_ids": list(self.support_ids),
            "counterexample_ids": list(self.counterexample_ids),
            "off_trigger_control_ids": list(self.off_trigger_control_ids),
            "non_identifiable_ids": list(self.non_identifiable_ids),
            "raw_support_count": self.raw_support_count,
            "effective_support_cluster_count": self.effective_support_cluster_count,
            "support_instance_count": self.support_instance_count,
            "untested_workload_instance_sha256s": list(
                self.untested_workload_instance_sha256s
            ),
            "coverage_gaps": list(self.coverage_gaps),
            "necessity_contradicted": self.necessity_contradicted,
            "mechanism_identified": self.mechanism_identified,
            "lifecycle_decision": self.lifecycle_decision,
            "causal_credit_policy": ("semantic_truth_only_no_retroactive_card_credit"),
            "causal_credit_updates": [],
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "audit_receipt_sha256": self.audit_receipt_sha256,
        }


class AmbiguousInterventionIdentityError(ValueError):
    """A matcher attempted to identify an ambiguous or joint intervention."""


@dataclass(frozen=True, slots=True)
class GlobalHypothesisFalsificationGate:
    """Exhaustively adjudicate one sealed hypothesis against a sealed registry."""

    policy_id: str = GLOBAL_FALSIFICATION_POLICY_ID
    policy_version: int = GLOBAL_FALSIFICATION_POLICY_VERSION
    definition_sha256: str = GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if (
            type(self.policy_version) is not int
            or self.policy_version != GLOBAL_FALSIFICATION_POLICY_VERSION
        ):
            raise ValueError("unsupported global falsification policy version")
        if self.definition_sha256 != GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256:
            raise ValueError("unsupported global falsification policy definition")

    def audit(
        self,
        *,
        request: GlobalHypothesisAuditRequest,
        registry: GlobalEvidenceRegistrySnapshot,
        matcher: GlobalHypothesisEvidenceMatcher,
    ) -> GlobalHypothesisAuditReceipt:
        if type(request) is not GlobalHypothesisAuditRequest:
            raise TypeError("request must be exact GlobalHypothesisAuditRequest")
        if type(registry) is not GlobalEvidenceRegistrySnapshot:
            raise TypeError("registry must be exact GlobalEvidenceRegistrySnapshot")
        if registry.snapshot_sha256 != request.registry_snapshot_sha256:
            raise ValueError("request is bound to a different registry snapshot")
        if registry.captured_through_event_index < request.audit_cutoff_event_index:
            raise ValueError("registry does not cover the requested audit cutoff")
        if request.audit_policy_definition_sha256 != self.definition_sha256:
            raise ValueError("audit policy differs from the sealed request")
        if not isinstance(matcher, GlobalHypothesisEvidenceMatcher):
            raise TypeError("matcher must implement GlobalHypothesisEvidenceMatcher")
        _require_token(matcher.policy_id, name="matcher.policy_id")
        if type(matcher.policy_version) is not int or matcher.policy_version <= 0:
            raise ValueError("matcher.policy_version must be a positive integer")
        require_sha256(matcher.definition_sha256, "matcher.definition_sha256")
        if matcher.definition_sha256 != request.matcher_definition_sha256:
            raise ValueError("matcher identity differs from the sealed request")

        representative_by_repeat: dict[tuple[str, str, str, str, str], str] = {}
        for observation in sorted(
            registry.observations,
            key=lambda value: (value.event_index, value.source_evidence_id),
        ):
            if (
                observation.event_index <= request.audit_cutoff_event_index
                and self._in_scope(request.scope, observation)
            ):
                repeat_key = (
                    observation.workload_instance_sha256,
                    observation.campaign_sha256,
                    observation.parent_configuration_sha256,
                    observation.child_configuration_sha256,
                    observation.evaluator_contract_sha256,
                )
                representative_by_repeat.setdefault(
                    repeat_key, observation.source_evidence_id
                )
        decisions: list[GlobalEvidenceDecision] = []
        support_instances: set[str] = set()
        support_clusters: set[str] = set()
        post_origin_support_instances: set[str] = set()
        post_origin_support_clusters: set[str] = set()
        mechanism_identified = False
        post_origin_mechanism_identified = False

        for observation in registry.observations:
            post_origin = observation.event_index > request.origin_cutoff_event_index
            if observation.event_index > request.audit_cutoff_event_index:
                decisions.append(
                    self._simple_decision(
                        observation,
                        EvidenceDisposition.AFTER_CUTOFF,
                        post_origin=post_origin,
                    )
                )
                continue
            if not self._in_scope(request.scope, observation):
                decisions.append(
                    self._simple_decision(
                        observation,
                        EvidenceDisposition.OUT_OF_SCOPE,
                        post_origin=post_origin,
                    )
                )
                continue
            repeat_key = (
                observation.workload_instance_sha256,
                observation.campaign_sha256,
                observation.parent_configuration_sha256,
                observation.child_configuration_sha256,
                observation.evaluator_contract_sha256,
            )
            representative = representative_by_repeat[repeat_key]
            if representative != observation.source_evidence_id:
                decisions.append(
                    self._simple_decision(
                        observation,
                        EvidenceDisposition.DUPLICATE,
                        post_origin=post_origin,
                        duplicate_of=representative,
                    )
                )
                continue
            match = matcher.classify(request, observation)
            self._validate_match(request, observation, matcher, match)
            assessments = self._metric_assessments(request, observation)
            predictions_match = (
                None
                if assessments is None
                else all(value.matches for value in assessments)
            )
            cluster = self._effective_cluster(observation)
            disposition = self._disposition(
                observation=observation,
                match=match,
                assessments=assessments,
            )
            if disposition is EvidenceDisposition.SUPPORT:
                support_instances.add(observation.workload_instance_sha256)
                support_clusters.add(cluster)
                mechanism_identified = mechanism_identified or (
                    observation.mechanism_identifying_design
                )
                if post_origin:
                    post_origin_support_instances.add(
                        observation.workload_instance_sha256
                    )
                    post_origin_support_clusters.add(cluster)
                    post_origin_mechanism_identified = (
                        post_origin_mechanism_identified
                        or observation.mechanism_identifying_design
                    )
            decisions.append(
                GlobalEvidenceDecision(
                    source_evidence_id=observation.source_evidence_id,
                    observation_sha256=observation.observation_sha256,
                    event_index=observation.event_index,
                    workload_instance_sha256=(observation.workload_instance_sha256),
                    itt_estimand_unit_sha256=(
                        observation.causal_boundary.estimand_unit_sha256
                    ),
                    disposition=disposition,
                    match_receipt_sha256=match.receipt_sha256,
                    metric_assessments=() if assessments is None else assessments,
                    effective_cluster_sha256=(
                        cluster
                        if disposition
                        in {
                            EvidenceDisposition.SUPPORT,
                            EvidenceDisposition.COUNTEREXAMPLE,
                            EvidenceDisposition.OFF_TRIGGER_CONTROL,
                        }
                        else None
                    ),
                    post_origin_revision_evidence=post_origin,
                    predictions_match=predictions_match,
                )
            )

        canonical = tuple(sorted(decisions, key=lambda value: value.source_evidence_id))
        support_ids = self._ids(canonical, EvidenceDisposition.SUPPORT)
        counterexample_ids = self._ids(canonical, EvidenceDisposition.COUNTEREXAMPLE)
        control_ids = self._ids(canonical, EvidenceDisposition.OFF_TRIGGER_CONTROL)
        non_identifiable_ids = self._ids(
            canonical, EvidenceDisposition.NON_IDENTIFIABLE
        )
        necessity_contradicted = request.claim_strength.necessity and any(
            decision.disposition is EvidenceDisposition.OFF_TRIGGER_CONTROL
            and decision.predictions_match is True
            for decision in canonical
        )
        exact_trigger_instance_ids = {
            decision.workload_instance_sha256
            for decision in canonical
            if decision.disposition
            in {
                EvidenceDisposition.SUPPORT,
                EvidenceDisposition.COUNTEREXAMPLE,
            }
        }
        untested_instances = tuple(
            sorted(
                set(request.scope.workload_instance_sha256s)
                - exact_trigger_instance_ids
            )
        )
        coverage_gaps = []
        if not support_ids and not counterexample_ids:
            coverage_gaps.append("no_identifiable_exact_trigger_intervention_evidence")
        if len(support_clusters) < request.minimum_support_clusters:
            coverage_gaps.append("support_cluster_threshold_not_met")
        if len(support_instances) < request.minimum_support_instances:
            coverage_gaps.append("support_instance_threshold_not_met")
        post_origin_support_ids = tuple(
            value.source_evidence_id
            for value in canonical
            if value.disposition is EvidenceDisposition.SUPPORT
            and value.post_origin_revision_evidence
            and value.event_index > request.origin_cutoff_event_index
        )
        if not post_origin_support_ids:
            coverage_gaps.append("post_origin_support_absent")
        if len(post_origin_support_clusters) < request.minimum_support_clusters:
            coverage_gaps.append("post_origin_support_cluster_threshold_not_met")
        if len(post_origin_support_instances) < request.minimum_support_instances:
            coverage_gaps.append("post_origin_support_instance_threshold_not_met")
        if request.claim_strength.necessity and not control_ids:
            coverage_gaps.append("necessity_off_trigger_controls_absent")
        if request.claim_strength.mechanistic_or_causal and not mechanism_identified:
            coverage_gaps.append("mechanism_identifying_trial_absent")
        if (
            request.claim_strength.mechanistic_or_causal
            and not post_origin_mechanism_identified
        ):
            coverage_gaps.append("post_origin_mechanism_identifying_trial_absent")
        if untested_instances:
            coverage_gaps.append(
                "workload_instances_without_identifiable_trigger_evidence"
            )
        if counterexample_ids or necessity_contradicted:
            verdict = GlobalHypothesisVerdict.COUNTEREXAMPLE
            lifecycle = "contradicted__retain_or_deprecate_predecessor_and_revise"
        elif (
            request.claim_strength.mechanistic_or_causal
            and not post_origin_mechanism_identified
        ):
            verdict = GlobalHypothesisVerdict.NON_IDENTIFIABLE
            lifecycle = "quarantined__mechanism_requires_post_origin_identifying_trial"
        elif (
            len(post_origin_support_clusters) >= request.minimum_support_clusters
            and len(post_origin_support_instances) >= request.minimum_support_instances
        ):
            verdict = GlobalHypothesisVerdict.SUPPORT
            lifecycle = (
                "globally_audited_local_support__eligible_for_preregistered_trial"
            )
        elif not support_ids and non_identifiable_ids:
            verdict = GlobalHypothesisVerdict.NON_IDENTIFIABLE
            lifecycle = "quarantined__intervention_or_trigger_not_identifiable"
        else:
            verdict = GlobalHypothesisVerdict.INSUFFICIENT
            lifecycle = "quarantined__undersupported_or_scope_restricted"
        return GlobalHypothesisAuditReceipt(
            request_sha256=request.request_sha256,
            registry_snapshot_sha256=registry.snapshot_sha256,
            audit_policy_id=self.policy_id,
            audit_policy_version=self.policy_version,
            audit_policy_definition_sha256=self.definition_sha256,
            verdict=verdict,
            decisions=canonical,
            support_ids=support_ids,
            counterexample_ids=counterexample_ids,
            off_trigger_control_ids=control_ids,
            non_identifiable_ids=non_identifiable_ids,
            raw_support_count=len(support_ids),
            effective_support_cluster_count=len(support_clusters),
            support_instance_count=len(support_instances),
            untested_workload_instance_sha256s=untested_instances,
            coverage_gaps=tuple(sorted(coverage_gaps)),
            necessity_contradicted=necessity_contradicted,
            mechanism_identified=mechanism_identified,
            lifecycle_decision=lifecycle,
        )

    @staticmethod
    def _in_scope(
        scope: HypothesisAuditScope,
        observation: AuthenticatedHypothesisObservation,
    ) -> bool:
        return (
            observation.workload_instance_sha256 in scope.workload_instance_sha256s
            and observation.evaluator_contract_sha256 == scope.evaluator_contract_sha256
            and all(
                metric.adjudicator_definition_sha256
                == scope.metric_adjudicator_definition_sha256
                for metric in observation.metrics
            )
            and (
                not scope.campaign_sha256s
                or observation.campaign_sha256 in scope.campaign_sha256s
            )
        )

    @staticmethod
    def _validate_match(
        request: GlobalHypothesisAuditRequest,
        observation: AuthenticatedHypothesisObservation,
        matcher: GlobalHypothesisEvidenceMatcher,
        receipt: HypothesisEvidenceMatchReceipt,
    ) -> None:
        if type(receipt) is not HypothesisEvidenceMatchReceipt:
            raise TypeError("matcher returned a foreign receipt type")
        observed = (
            receipt.request_sha256,
            receipt.observation_sha256,
            receipt.matcher_policy_id,
            receipt.matcher_policy_version,
            receipt.matcher_definition_sha256,
        )
        expected = (
            request.request_sha256,
            observation.observation_sha256,
            matcher.policy_id,
            matcher.policy_version,
            matcher.definition_sha256,
        )
        if observed != expected:
            raise ValueError("matcher receipt belongs to a foreign request or policy")
        if (
            observation.intervention_identifiability
            is not InterventionIdentifiability.EXACT_SINGLE
            and receipt.intervention_match is not InterventionMatch.NON_IDENTIFIABLE
        ):
            raise AmbiguousInterventionIdentityError(
                "ambiguous or joint interventions must fail closed as non-identifiable"
            )
        if receipt.intervention_match is InterventionMatch.EXACT and (
            observation.affected_paths != request.intervention.affected_paths
            or observation.operator_family
            not in request.intervention.admissible_operator_families
        ):
            raise ValueError(
                "exact intervention match contradicts the sealed path or operator family"
            )

    @staticmethod
    def _metric_assessments(
        request: GlobalHypothesisAuditRequest,
        observation: AuthenticatedHypothesisObservation,
    ) -> tuple[MetricClaimAssessment, ...] | None:
        by_metric = {value.metric_id: value for value in observation.metrics}
        if any(value.metric_id not in by_metric for value in request.predictions):
            return None
        result = []
        for prediction in request.predictions:
            actual = by_metric[prediction.metric_id]
            magnitude_matches = None
            if prediction.has_magnitude_claim:
                assert prediction.minimum_delta is not None
                assert prediction.maximum_delta is not None
                magnitude_matches = (
                    prediction.minimum_delta <= actual.delta <= prediction.maximum_delta
                )
            result.append(
                MetricClaimAssessment(
                    metric_id=prediction.metric_id,
                    expected_direction=prediction.direction,
                    actual_direction=actual.direction,
                    actual_delta=actual.delta,
                    direction_matches=prediction.direction is actual.direction,
                    magnitude_matches=magnitude_matches,
                )
            )
        return tuple(result)

    @staticmethod
    def _disposition(
        *,
        observation: AuthenticatedHypothesisObservation,
        match: HypothesisEvidenceMatchReceipt,
        assessments: tuple[MetricClaimAssessment, ...] | None,
    ) -> EvidenceDisposition:
        if (
            observation.intervention_identifiability
            is not InterventionIdentifiability.EXACT_SINGLE
            or match.intervention_match is InterventionMatch.NON_IDENTIFIABLE
            or match.trigger_match is TriggerMatch.AMBIGUOUS
        ):
            return EvidenceDisposition.NON_IDENTIFIABLE
        if match.intervention_match is InterventionMatch.NEAR:
            return EvidenceDisposition.NEAR_INTERVENTION
        if match.intervention_match is InterventionMatch.DIFFERENT:
            return EvidenceDisposition.IRRELEVANT
        if assessments is None:
            return EvidenceDisposition.NON_IDENTIFIABLE
        if match.trigger_match is TriggerMatch.OFF_TRIGGER:
            return EvidenceDisposition.OFF_TRIGGER_CONTROL
        if all(value.matches for value in assessments):
            return EvidenceDisposition.SUPPORT
        return EvidenceDisposition.COUNTEREXAMPLE

    @staticmethod
    def _effective_cluster(observation: AuthenticatedHypothesisObservation) -> str:
        return _hash_record(
            _CLUSTER_DOMAIN,
            {
                "workload_instance_sha256": observation.workload_instance_sha256,
                "evaluator_contract_sha256": observation.evaluator_contract_sha256,
                "campaign_sha256": observation.campaign_sha256,
                "lineage_cluster_sha256": observation.lineage_cluster_sha256,
                "factorial_block_sha256": observation.factorial_block_sha256,
                "itt_estimand_unit_sha256": (
                    observation.causal_boundary.estimand_unit_sha256
                ),
            },
        )

    @staticmethod
    def _ids(
        decisions: tuple[GlobalEvidenceDecision, ...],
        disposition: EvidenceDisposition,
    ) -> tuple[str, ...]:
        return tuple(
            value.source_evidence_id
            for value in decisions
            if value.disposition is disposition
        )

    @staticmethod
    def _simple_decision(
        observation: AuthenticatedHypothesisObservation,
        disposition: EvidenceDisposition,
        *,
        post_origin: bool,
        duplicate_of: str | None = None,
    ) -> GlobalEvidenceDecision:
        return GlobalEvidenceDecision(
            source_evidence_id=observation.source_evidence_id,
            observation_sha256=observation.observation_sha256,
            event_index=observation.event_index,
            workload_instance_sha256=observation.workload_instance_sha256,
            itt_estimand_unit_sha256=(observation.causal_boundary.estimand_unit_sha256),
            disposition=disposition,
            match_receipt_sha256=None,
            metric_assessments=(),
            effective_cluster_sha256=None,
            post_origin_revision_evidence=post_origin,
            duplicate_of_source_evidence_id=duplicate_of,
        )


class RevisionEvidenceTiming(str, Enum):
    AVAILABLE_AT_ORIGIN = "available_at_origin"
    POST_ORIGIN_REVISION_EVIDENCE = "post_origin_revision_evidence"


@dataclass(frozen=True, slots=True)
class RevisionEvidenceReference:
    source_evidence_id: str
    disposition: EvidenceDisposition
    timing: RevisionEvidenceTiming

    def __post_init__(self) -> None:
        require_sha256(self.source_evidence_id, "source_evidence_id")
        if type(self.disposition) is not EvidenceDisposition:
            raise TypeError("disposition must be exact EvidenceDisposition")
        if type(self.timing) is not RevisionEvidenceTiming:
            raise TypeError("timing must be exact RevisionEvidenceTiming")

    def to_record(self) -> dict[str, str]:
        return {
            "source_evidence_id": self.source_evidence_id,
            "disposition": self.disposition.value,
            "timing": self.timing.value,
        }


@dataclass(frozen=True, slots=True)
class AppendOnlyHypothesisRevision:
    """Successor metadata that never mutates or inherits credit from its parent."""

    predecessor: InsightRef
    successor: InsightRef
    predecessor_draft_content_sha256: str
    successor_draft_content_sha256: str
    predecessor_audit_request_sha256: str
    audit_receipt_sha256: str
    successor_trigger_predicate_sha256: str
    successor_intervention_signature_sha256: str
    successor_scope_sha256: str
    claim_diff: str
    scope_diff: str
    evidence: tuple[RevisionEvidenceReference, ...]
    successor_lifecycle_state: str = field(init=False, default="quarantined")
    trial_eligibility_reset: bool = field(init=False, default=True)
    inherited_confirmation_count: int = field(init=False, default=0)
    inherited_causal_credit: bool = field(init=False, default=False)
    revision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.predecessor) is not InsightRef
            or type(self.successor) is not InsightRef
        ):
            raise TypeError("predecessor and successor must be exact InsightRef values")
        if (
            self.successor.insight_id != self.predecessor.insight_id
            or self.successor.version != self.predecessor.version + 1
        ):
            raise ValueError("successor must be the exact next version of predecessor")
        for name in (
            "predecessor_draft_content_sha256",
            "successor_draft_content_sha256",
            "predecessor_audit_request_sha256",
            "audit_receipt_sha256",
            "successor_trigger_predicate_sha256",
            "successor_intervention_signature_sha256",
            "successor_scope_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_canonical_text(self.claim_diff, name="claim_diff")
        _require_canonical_text(self.scope_diff, name="scope_diff")
        if (
            self.predecessor_draft_content_sha256 == self.successor_draft_content_sha256
            and self.claim_diff == "unchanged"
            and self.scope_diff == "unchanged"
        ):
            raise ValueError("a revision must change claim content or scope")
        if type(self.evidence) is not tuple or any(
            type(value) is not RevisionEvidenceReference for value in self.evidence
        ):
            raise TypeError(
                "evidence must contain exact RevisionEvidenceReference values"
            )
        allowed_evidence_dispositions = {
            EvidenceDisposition.SUPPORT,
            EvidenceDisposition.COUNTEREXAMPLE,
            EvidenceDisposition.OFF_TRIGGER_CONTROL,
            EvidenceDisposition.NON_IDENTIFIABLE,
        }
        if any(
            value.disposition not in allowed_evidence_dispositions
            for value in self.evidence
        ):
            raise ValueError(
                "revision evidence may contain only matched semantic audit rows"
            )
        evidence_ids = tuple(value.source_evidence_id for value in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("revision evidence must use unique canonical source order")
        object.__setattr__(
            self,
            "revision_sha256",
            _hash_record(_REVISION_DOMAIN, self._unsigned_record()),
        )

    @classmethod
    def from_audit(
        cls,
        *,
        request: GlobalHypothesisAuditRequest,
        receipt: GlobalHypothesisAuditReceipt,
        successor: InsightRef,
        successor_draft_content_sha256: str,
        successor_trigger_predicate_sha256: str,
        successor_intervention_signature_sha256: str,
        successor_scope_sha256: str,
        claim_diff: str,
        scope_diff: str,
    ) -> "AppendOnlyHypothesisRevision":
        if type(request) is not GlobalHypothesisAuditRequest:
            raise TypeError("request must be exact GlobalHypothesisAuditRequest")
        if type(receipt) is not GlobalHypothesisAuditReceipt:
            raise TypeError("receipt must be exact GlobalHypothesisAuditReceipt")
        if receipt.request_sha256 != request.request_sha256:
            raise ValueError("audit receipt belongs to a different hypothesis request")
        evidence = tuple(
            RevisionEvidenceReference(
                source_evidence_id=decision.source_evidence_id,
                disposition=decision.disposition,
                timing=(
                    RevisionEvidenceTiming.POST_ORIGIN_REVISION_EVIDENCE
                    if decision.post_origin_revision_evidence
                    else RevisionEvidenceTiming.AVAILABLE_AT_ORIGIN
                ),
            )
            for decision in receipt.decisions
            if decision.disposition
            in {
                EvidenceDisposition.SUPPORT,
                EvidenceDisposition.COUNTEREXAMPLE,
                EvidenceDisposition.OFF_TRIGGER_CONTROL,
                EvidenceDisposition.NON_IDENTIFIABLE,
            }
        )
        return cls(
            predecessor=request.reference,
            successor=successor,
            predecessor_draft_content_sha256=request.draft_content_sha256,
            successor_draft_content_sha256=successor_draft_content_sha256,
            predecessor_audit_request_sha256=request.request_sha256,
            audit_receipt_sha256=receipt.audit_receipt_sha256,
            successor_trigger_predicate_sha256=(successor_trigger_predicate_sha256),
            successor_intervention_signature_sha256=(
                successor_intervention_signature_sha256
            ),
            successor_scope_sha256=successor_scope_sha256,
            claim_diff=claim_diff,
            scope_diff=scope_diff,
            evidence=evidence,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "predecessor": {
                "insight_id": self.predecessor.insight_id.value,
                "version": self.predecessor.version,
            },
            "successor": {
                "insight_id": self.successor.insight_id.value,
                "version": self.successor.version,
            },
            "predecessor_draft_content_sha256": (self.predecessor_draft_content_sha256),
            "successor_draft_content_sha256": self.successor_draft_content_sha256,
            "predecessor_audit_request_sha256": (self.predecessor_audit_request_sha256),
            "audit_receipt_sha256": self.audit_receipt_sha256,
            "successor_trigger_predicate_sha256": (
                self.successor_trigger_predicate_sha256
            ),
            "successor_intervention_signature_sha256": (
                self.successor_intervention_signature_sha256
            ),
            "successor_scope_sha256": self.successor_scope_sha256,
            "claim_diff": self.claim_diff,
            "scope_diff": self.scope_diff,
            "evidence": [value.to_record() for value in self.evidence],
            "successor_lifecycle_state": self.successor_lifecycle_state,
            "trial_eligibility_reset": self.trial_eligibility_reset,
            "inherited_confirmation_count": self.inherited_confirmation_count,
            "inherited_causal_credit": self.inherited_causal_credit,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "revision_sha256": self.revision_sha256}


__all__ = [
    "AmbiguousInterventionIdentityError",
    "AppendOnlyHypothesisRevision",
    "AuthenticatedHypothesisObservation",
    "CausalEstimandUnit",
    "EvidenceCausalBoundary",
    "EvidenceDisposition",
    "EvidenceProvenance",
    "GlobalEvidenceDecision",
    "GlobalEvidenceRegistrySnapshot",
    "GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256",
    "GLOBAL_FALSIFICATION_POLICY_ID",
    "GLOBAL_FALSIFICATION_POLICY_VERSION",
    "GlobalHypothesisAuditReceipt",
    "GlobalHypothesisAuditRequest",
    "GlobalHypothesisEvidenceMatcher",
    "GlobalHypothesisFalsificationGate",
    "GlobalHypothesisVerdict",
    "HypothesisAuditScope",
    "HypothesisClaimStrength",
    "HypothesisEvidenceMatchReceipt",
    "HypothesisMetricPrediction",
    "InterventionIdentifiability",
    "InterventionMatch",
    "MetricClaimAssessment",
    "ObservedMetricEffect",
    "RevisionEvidenceReference",
    "RevisionEvidenceTiming",
    "TriggerMatch",
    "TypedEvidencePredicate",
    "TypedInterventionSignature",
]
