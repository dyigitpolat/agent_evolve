"""Evidence-identifiable, mutation-only inputs for prospective reflection.

The production campaign already records authenticated parent-to-child mutation
observations.  Reflection should consume those exact single-intervention facts
instead of assigning a recombination result to one arbitrarily chosen
coordinate.  This module is a pure projection over sealed observations.  It
does not call a provider, interpret benchmark prose, mutate memory, or decide
when a reflected card becomes eligible for use.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_frozen_json_value,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
    ReflectionInsightKind,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_PATH = re.compile(r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$")
_CONTRAST_DOMAIN = b"agent-evolve:identifiable-reflection-contrast:v3\x00"
_HYPOTHESIS_CLUSTER_DOMAIN = (
    b"agent-evolve:identifiable-reflection-hypothesis-cluster:v1\x00"
)
_SNAPSHOT_DOMAIN = b"agent-evolve:identifiable-reflection-snapshot:v3\x00"
_FEEDBACK_DOMAIN = b"agent-evolve:reflection-falsification-feedback:v1\x00"
MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES = 4_096

IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID = (
    "sealed_direct_single_mutation_reflection_evidence"
)
IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION = 3
IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sealed-direct-single-mutation-reflection-evidence:v3;"
    b"direct-mutation-only;exact-single-intervention;one-affected-path;"
    b"observed-action-path-option-family-and-finite-contract-join;sealed-event-cutoff;"
    b"authenticated-action-semantics-compiler-id-version-definition-join;"
    b"exact-parent-child-candidate-and-operator-invocation-lineage;"
    b"bounded-exact-parent-child-local-values;"
    b"empirical-rule-only-without-mechanism-identifying-design"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_paths(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _PATH.fullmatch(value) is None for value in values
    ):
        raise ValueError(f"{name} must be an exact tuple of canonical JSON paths")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


def _require_sha_tuple(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool,
) -> None:
    if type(values) is not tuple or (not allow_empty and not values):
        raise ValueError(f"{name} must be a canonical SHA-256 tuple")
    for value in values:
        require_sha256(value, name)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


class ReflectionEvidenceExclusionReason(str, Enum):
    """Why an authenticated observation cannot identify a reflection claim."""

    BEFORE_OR_AT_PRIOR_CUTOFF = "before_or_at_prior_cutoff"
    AFTER_SEALED_CUTOFF = "after_sealed_cutoff"
    FOREIGN_SCOPE = "foreign_scope"
    NON_MUTATION_PROVENANCE = "non_mutation_provenance"
    NON_SINGLE_INTERVENTION = "non_single_intervention"
    MULTI_PATH_INTERVENTION = "multi_path_intervention"
    MALFORMED_ACTION_SEMANTICS = "malformed_action_semantics"
    LOCAL_INTERVENTION_UNAVAILABLE = "local_intervention_unavailable"
    LOCAL_INTERVENTION_TOO_LARGE = "local_intervention_too_large"
    NON_CHANGING_LOCAL_INTERVENTION = "non_changing_local_intervention"


@dataclass(frozen=True, slots=True)
class IdentifiableMutationReflectionContrast:
    """One direct, single-path mutation fact suitable for reflection."""

    contrast_id: str
    source_observation_sha256: str
    source_evidence_id: str
    event_index: int
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    campaign_sha256: str
    parent_candidate_id: CandidateId
    child_candidate_id: CandidateId
    operator_invocation_id: OperatorInvocationId
    finite_contract_identity_sha256: str
    action_semantics_compiler_id: str
    action_semantics_compiler_version: int
    action_semantics_definition_sha256: str
    option_id: str
    option_identity_sha256: str
    option_family: str
    affected_path: str
    parent_local_value: FrozenJsonValue
    child_local_value: FrozenJsonValue
    parent_configuration_sha256: str
    child_configuration_sha256: str
    parent_outcome_sha256: str
    child_outcome_sha256: str
    metrics: tuple[ObservedMetricEffect, ...]
    mechanism_identifying_design: bool
    permitted_insight_kinds: tuple[ReflectionInsightKind, ...]
    contrast_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "contrast_id",
            "source_observation_sha256",
            "source_evidence_id",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
            "campaign_sha256",
            "option_identity_sha256",
            "finite_contract_identity_sha256",
            "action_semantics_definition_sha256",
            "parent_configuration_sha256",
            "child_configuration_sha256",
            "parent_outcome_sha256",
            "child_outcome_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.contrast_id != self.source_observation_sha256:
            raise ValueError("contrast_id must be the authenticated observation hash")
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
            raise ValueError("reflection child occurrence cannot reuse its parent ID")
        _require_token(
            self.action_semantics_compiler_id,
            name="action_semantics_compiler_id",
        )
        if (
            type(self.action_semantics_compiler_version) is not int
            or self.action_semantics_compiler_version <= 0
        ):
            raise ValueError("action_semantics_compiler_version must be positive")
        if type(self.event_index) is not int or self.event_index <= 0:
            raise ValueError("event_index must be positive")
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the closed option grammar")
        _require_token(self.option_family, name="option_family")
        _require_paths((self.affected_path,), name="affected_path")
        for name in ("parent_local_value", "child_local_value"):
            value = getattr(self, name)
            if not is_frozen_json_value(value):
                raise TypeError(f"{name} must be exact frozen typed JSON")
            if (
                len(canonical_typed_json_bytes(value))
                > MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES
            ):
                raise ValueError(f"{name} exceeds the local intervention bound")
        if typed_json_equal(self.parent_local_value, self.child_local_value):
            raise ValueError("local intervention must change its affected value")
        if (
            type(self.metrics) is not tuple
            or not self.metrics
            or any(type(value) is not ObservedMetricEffect for value in self.metrics)
        ):
            raise ValueError("metrics must contain exact observed effects")
        for value in self.metrics:
            ObservedMetricEffect.__post_init__(value)
        if tuple(value.metric_id for value in self.metrics) != tuple(
            sorted({value.metric_id for value in self.metrics})
        ):
            raise ValueError("metrics must use unique canonical metric order")
        if type(self.mechanism_identifying_design) is not bool:
            raise TypeError("mechanism_identifying_design must be exact bool")
        if self.mechanism_identifying_design:
            raise ValueError(
                "direct single-mutation contrasts cannot identify mechanisms"
            )
        if (
            type(self.permitted_insight_kinds) is not tuple
            or not self.permitted_insight_kinds
            or any(
                type(value) is not ReflectionInsightKind
                for value in self.permitted_insight_kinds
            )
        ):
            raise ValueError("permitted_insight_kinds must contain exact kinds")
        if self.permitted_insight_kinds != tuple(
            sorted(set(self.permitted_insight_kinds), key=lambda value: value.value)
        ):
            raise ValueError("permitted_insight_kinds must be unique and canonical")
        expected_kinds = (ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,)
        if self.permitted_insight_kinds != expected_kinds:
            raise ValueError("permitted insight kinds overstate evidence design")
        object.__setattr__(
            self,
            "contrast_sha256",
            _hash(_CONTRAST_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "contrast_id": self.contrast_id,
            "source_observation_sha256": self.source_observation_sha256,
            "source_evidence_id": self.source_evidence_id,
            "event_index": self.event_index,
            "workload_instance_sha256": self.workload_instance_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "campaign_sha256": self.campaign_sha256,
            "parent_candidate_id": self.parent_candidate_id.value,
            "child_candidate_id": self.child_candidate_id.value,
            "operator_invocation_id": self.operator_invocation_id.value,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "action_semantics_compiler": {
                "compiler_id": self.action_semantics_compiler_id,
                "compiler_version": self.action_semantics_compiler_version,
                "definition_sha256": self.action_semantics_definition_sha256,
            },
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "option_family": self.option_family,
            "affected_path": self.affected_path,
            "parent_local_value": thaw_json(self.parent_local_value),
            "parent_local_value_sha256": typed_json_sha256(
                self.parent_local_value
            ),
            "child_local_value": thaw_json(self.child_local_value),
            "child_local_value_sha256": typed_json_sha256(
                self.child_local_value
            ),
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "parent_outcome_sha256": self.parent_outcome_sha256,
            "child_outcome_sha256": self.child_outcome_sha256,
            "metrics": [value.to_record() for value in self.metrics],
            "mechanism_identifying_design": self.mechanism_identifying_design,
            "permitted_insight_kinds": [
                value.value for value in self.permitted_insight_kinds
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "contrast_sha256": self.contrast_sha256}

    def to_prompt_record(self, *, evidence_citation_key: str) -> dict[str, object]:
        """Return bounded scientific facts without raw candidate configurations."""

        self.__post_init__()
        _require_token(evidence_citation_key, name="evidence_citation_key")
        return {
            "evidence_citation_key": evidence_citation_key,
            "event_index": self.event_index,
            "option_id": self.option_id,
            "option_family": self.option_family,
            "affected_path": self.affected_path,
            "local_intervention": {
                "parent_value": thaw_json(self.parent_local_value),
                "child_value": thaw_json(self.child_local_value),
            },
            "metric_effects": [
                {
                    "metric_id": value.metric_id,
                    "direction": value.direction.value,
                    # Keep the hexadecimal form for exact replay, but also
                    # expose Python's shortest round-trippable decimal text.
                    # Several otherwise-conformant models read the binary
                    # exponent in ``float.hex`` as a base-10 exponent, which
                    # changes an authenticated measurement by many orders of
                    # magnitude inside the learned prose rule.
                    "delta_decimal": repr(value.delta),
                    "delta_hex": value.delta.hex(),
                }
                for value in self.metrics
            ],
            "permitted_insight_kinds": [
                value.value for value in self.permitted_insight_kinds
            ],
            "comparison_anchor": "current_parent",
        }


def _hypothesis_signature(
    contrast: IdentifiableMutationReflectionContrast,
) -> dict[str, object]:
    """Return the parent-relative empirical hypothesis represented by a fact."""

    if type(contrast) is not IdentifiableMutationReflectionContrast:
        raise TypeError("contrast must be exact")
    IdentifiableMutationReflectionContrast.__post_init__(contrast)
    # Finite-contract and option-identity hashes authenticate one parent-bound
    # occurrence.  They intentionally remain on each contrast rather than in
    # the semantic key, so the same local action observed under two parents can
    # accumulate evidence instead of becoming duplicate prose claims.
    return {
        "schema_version": 1,
        "scope": {
            "campaign_sha256": contrast.campaign_sha256,
            "workload_instance_sha256": contrast.workload_instance_sha256,
            "evaluator_contract_sha256": contrast.evaluator_contract_sha256,
        },
        "finite_action": {
            "option_id": contrast.option_id,
            "option_family": contrast.option_family,
            "affected_path": contrast.affected_path,
            "action_semantics_compiler_id": (
                contrast.action_semantics_compiler_id
            ),
            "action_semantics_compiler_version": (
                contrast.action_semantics_compiler_version
            ),
            "action_semantics_definition_sha256": (
                contrast.action_semantics_definition_sha256
            ),
        },
        "local_intervention": {
            "parent_value_sha256": typed_json_sha256(
                contrast.parent_local_value
            ),
            "child_value_sha256": typed_json_sha256(
                contrast.child_local_value
            ),
        },
        "metric_direction_signature": [
            {
                "metric_id": value.metric_id,
                "direction": value.direction.value,
            }
            for value in contrast.metrics
        ],
        "mechanism_identifying_design": contrast.mechanism_identifying_design,
        "permitted_insight_kinds": [
            value.value for value in contrast.permitted_insight_kinds
        ],
        "comparison_anchor": "current_parent",
    }


@dataclass(frozen=True, slots=True)
class IdentifiableMutationReflectionHypothesisCluster:
    """Repeated direct observations supporting one unique empirical claim."""

    contrasts: tuple[IdentifiableMutationReflectionContrast, ...]
    hypothesis_sha256: str = field(init=False)
    cluster_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.contrasts) is not tuple
            or not self.contrasts
            or any(
                type(value) is not IdentifiableMutationReflectionContrast
                for value in self.contrasts
            )
        ):
            raise ValueError("hypothesis cluster must contain exact contrasts")
        for value in self.contrasts:
            IdentifiableMutationReflectionContrast.__post_init__(value)
        contrast_ids = tuple(value.contrast_id for value in self.contrasts)
        if contrast_ids != tuple(sorted(set(contrast_ids))):
            raise ValueError(
                "hypothesis-cluster contrasts must be unique and canonical"
            )
        signature = _hypothesis_signature(self.contrasts[0])
        if any(
            _hypothesis_signature(value) != signature
            for value in self.contrasts[1:]
        ):
            raise ValueError("hypothesis cluster mixed distinct empirical claims")
        hypothesis_sha256 = _hash(_HYPOTHESIS_CLUSTER_DOMAIN, signature)
        object.__setattr__(self, "hypothesis_sha256", hypothesis_sha256)
        object.__setattr__(
            self,
            "cluster_sha256",
            _hash(
                _HYPOTHESIS_CLUSTER_DOMAIN,
                {
                    "schema_version": 1,
                    "hypothesis_sha256": hypothesis_sha256,
                    "contrast_ids": list(contrast_ids),
                },
            ),
        )

    @property
    def representative(self) -> IdentifiableMutationReflectionContrast:
        self.__post_init__()
        return self.contrasts[0]

    @property
    def contrast_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.contrast_id for value in self.contrasts)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "hypothesis_sha256": self.hypothesis_sha256,
            "cluster_sha256": self.cluster_sha256,
            "hypothesis_signature": _hypothesis_signature(self.contrasts[0]),
            "contrast_ids": list(self.contrast_ids),
            "observation_count": len(self.contrasts),
        }


def cluster_identifiable_mutation_reflection_hypotheses(
    contrasts: tuple[IdentifiableMutationReflectionContrast, ...],
) -> tuple[IdentifiableMutationReflectionHypothesisCluster, ...]:
    """Aggregate repeated interventions without collapsing conflicting effects."""

    if (
        type(contrasts) is not tuple
        or not contrasts
        or any(
            type(value) is not IdentifiableMutationReflectionContrast
            for value in contrasts
        )
    ):
        raise ValueError("contrasts must contain exact identifiable evidence")
    for value in contrasts:
        IdentifiableMutationReflectionContrast.__post_init__(value)
    contrast_ids = tuple(value.contrast_id for value in contrasts)
    if contrast_ids != tuple(sorted(set(contrast_ids))):
        raise ValueError("contrasts must use unique canonical contrast order")

    grouped: dict[str, list[IdentifiableMutationReflectionContrast]] = {}
    for contrast in contrasts:
        hypothesis_sha256 = _hash(
            _HYPOTHESIS_CLUSTER_DOMAIN,
            _hypothesis_signature(contrast),
        )
        grouped.setdefault(hypothesis_sha256, []).append(contrast)
    return tuple(
        IdentifiableMutationReflectionHypothesisCluster(contrasts=tuple(values))
        for values in grouped.values()
    )


@dataclass(frozen=True, slots=True)
class ReflectionFalsificationFeedback:
    """A prior card's authenticated counterexample, safe for later prompts."""

    insight_content_sha256: str
    applicable_workload_instance_sha256s: tuple[str, ...]
    evaluator_contract_sha256: str
    applicable_campaign_sha256s: tuple[str, ...]
    audit_scope_sha256: str
    available_event_index: int
    affected_paths: tuple[str, ...]
    predictions: tuple[tuple[str, MetricEffectDirection], ...]
    counterexample_source_evidence_ids: tuple[str, ...]
    semantic_audit_receipt_sha256: str
    lifecycle_decision_receipt_sha256: str
    deprecation_reason: str
    feedback_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "insight_content_sha256",
            "evaluator_contract_sha256",
            "audit_scope_sha256",
            "semantic_audit_receipt_sha256",
            "lifecycle_decision_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_sha_tuple(
            self.applicable_workload_instance_sha256s,
            name="applicable_workload_instance_sha256s",
            allow_empty=False,
        )
        _require_sha_tuple(
            self.applicable_campaign_sha256s,
            name="applicable_campaign_sha256s",
            allow_empty=True,
        )
        if type(self.available_event_index) is not int or (
            self.available_event_index <= 0
        ):
            raise ValueError("available_event_index must be positive")
        _require_paths(self.affected_paths, name="affected_paths")
        if type(self.predictions) is not tuple or not self.predictions:
            raise ValueError("predictions must be a non-empty exact tuple")
        metric_ids: list[str] = []
        for item in self.predictions:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("predictions must contain exact metric/direction pairs")
            metric_id, direction = item
            _require_token(metric_id, name="metric_id")
            if (
                type(direction) is not MetricEffectDirection
                or direction is MetricEffectDirection.UNKNOWN
            ):
                raise ValueError("prior predictions must use known directions")
            metric_ids.append(metric_id)
        if metric_ids != sorted(set(metric_ids)):
            raise ValueError("predictions must use unique canonical metric order")
        if (
            type(self.counterexample_source_evidence_ids) is not tuple
            or not self.counterexample_source_evidence_ids
        ):
            raise ValueError("counterexample IDs must be a non-empty exact tuple")
        for value in self.counterexample_source_evidence_ids:
            require_sha256(value, "counterexample source evidence ID")
        if self.counterexample_source_evidence_ids != tuple(
            sorted(set(self.counterexample_source_evidence_ids))
        ):
            raise ValueError("counterexample IDs must be unique and canonical")
        if (
            type(self.deprecation_reason) is not str
            or not self.deprecation_reason.strip()
            or self.deprecation_reason != self.deprecation_reason.strip()
        ):
            raise ValueError("deprecation_reason must be canonical non-empty text")
        object.__setattr__(
            self,
            "feedback_sha256",
            _hash(_FEEDBACK_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "insight_content_sha256": self.insight_content_sha256,
            "applicable_workload_instance_sha256s": list(
                self.applicable_workload_instance_sha256s
            ),
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "applicable_campaign_sha256s": list(
                self.applicable_campaign_sha256s
            ),
            "audit_scope_sha256": self.audit_scope_sha256,
            "available_event_index": self.available_event_index,
            "affected_paths": list(self.affected_paths),
            "predictions": [
                {"metric_id": metric_id, "direction": direction.value}
                for metric_id, direction in self.predictions
            ],
            "counterexample_source_evidence_ids": list(
                self.counterexample_source_evidence_ids
            ),
            "semantic_audit_receipt_sha256": self.semantic_audit_receipt_sha256,
            "lifecycle_decision_receipt_sha256": (
                self.lifecycle_decision_receipt_sha256
            ),
            "deprecation_reason": self.deprecation_reason,
            "instruction": (
                "do_not_repeat_without_new_identifiable_counterevidence"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "feedback_sha256": self.feedback_sha256}

    def to_prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "available_event_index": self.available_event_index,
            "affected_paths": list(self.affected_paths),
            "predictions": [
                {"metric_id": metric_id, "direction": direction.value}
                for metric_id, direction in self.predictions
            ],
            "counterexample_count": len(self.counterexample_source_evidence_ids),
            "deprecation_reason": self.deprecation_reason,
            "instruction": (
                "Do not repeat this claim unless the new single-intervention "
                "evidence directly resolves its counterexample."
            ),
        }


@dataclass(frozen=True, slots=True)
class IdentifiableReflectionEvidenceSnapshot:
    """Sealed mutation evidence and prior falsifications for one LLM call."""

    campaign_sha256: str
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    prior_cutoff_event_index_exclusive: int
    sealed_cutoff_event_index_inclusive: int
    contrasts: tuple[IdentifiableMutationReflectionContrast, ...]
    exclusions: tuple[tuple[ReflectionEvidenceExclusionReason, int], ...]
    prior_falsifications: tuple[ReflectionFalsificationFeedback, ...] = ()
    policy_id: str = IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID
    policy_version: int = IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION
    policy_definition_sha256: str = (
        IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256
    )
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_sha256",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "prior_cutoff_event_index_exclusive",
            "sealed_cutoff_event_index_inclusive",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if (
            self.sealed_cutoff_event_index_inclusive
            <= self.prior_cutoff_event_index_exclusive
        ):
            raise ValueError("sealed cutoff must advance beyond the prior cutoff")
        if (
            type(self.contrasts) is not tuple
            or not self.contrasts
            or any(
                type(value) is not IdentifiableMutationReflectionContrast
                for value in self.contrasts
            )
        ):
            raise ValueError("contrasts must contain exact eligible evidence")
        for value in self.contrasts:
            IdentifiableMutationReflectionContrast.__post_init__(value)
        contrast_ids = tuple(value.contrast_id for value in self.contrasts)
        if contrast_ids != tuple(sorted(set(contrast_ids))):
            raise ValueError("contrasts must use unique canonical contrast order")
        if any(
            value.campaign_sha256 != self.campaign_sha256
            or value.workload_instance_sha256 != self.workload_instance_sha256
            or value.evaluator_contract_sha256 != self.evaluator_contract_sha256
            or not (
                self.prior_cutoff_event_index_exclusive
                < value.event_index
                <= self.sealed_cutoff_event_index_inclusive
            )
            for value in self.contrasts
        ):
            raise ValueError("contrast escapes the sealed reflection scope")
        if type(self.exclusions) is not tuple:
            raise TypeError("exclusions must be an exact tuple")
        reasons: list[str] = []
        for reason, count in self.exclusions:
            if type(reason) is not ReflectionEvidenceExclusionReason:
                raise TypeError("exclusion reason must be exact")
            if type(count) is not int or count <= 0:
                raise ValueError("exclusion count must be positive")
            reasons.append(reason.value)
        if reasons != sorted(set(reasons)):
            raise ValueError("exclusions must use canonical unique reason order")
        if type(self.prior_falsifications) is not tuple or any(
            type(value) is not ReflectionFalsificationFeedback
            for value in self.prior_falsifications
        ):
            raise TypeError("prior_falsifications must contain exact feedback")
        for value in self.prior_falsifications:
            ReflectionFalsificationFeedback.__post_init__(value)
            if (
                self.workload_instance_sha256
                not in value.applicable_workload_instance_sha256s
                or value.evaluator_contract_sha256
                != self.evaluator_contract_sha256
                or (
                    value.applicable_campaign_sha256s
                    and self.campaign_sha256
                    not in value.applicable_campaign_sha256s
                )
                or value.available_event_index
                > self.sealed_cutoff_event_index_inclusive
            ):
                raise ValueError(
                    "prior falsification escapes the sealed reflection scope"
                )
        if tuple(value.feedback_sha256 for value in self.prior_falsifications) != tuple(
            sorted({value.feedback_sha256 for value in self.prior_falsifications})
        ):
            raise ValueError("prior falsifications must be unique and canonical")
        if (
            self.policy_id != IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID
            or self.policy_version != IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION
            or self.policy_definition_sha256
            != IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported identifiable reflection evidence policy")
        object.__setattr__(
            self,
            "snapshot_sha256",
            _hash(_SNAPSHOT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "campaign_sha256": self.campaign_sha256,
            "workload_instance_sha256": self.workload_instance_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "prior_cutoff_event_index_exclusive": (
                self.prior_cutoff_event_index_exclusive
            ),
            "sealed_cutoff_event_index_inclusive": (
                self.sealed_cutoff_event_index_inclusive
            ),
            "contrast_sha256s": [value.contrast_sha256 for value in self.contrasts],
            "exclusions": [
                {"reason": reason.value, "count": count}
                for reason, count in self.exclusions
            ],
            "prior_falsification_sha256s": [
                value.feedback_sha256 for value in self.prior_falsifications
            ],
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "contrasts": [value.to_record() for value in self.contrasts],
            "prior_falsifications": [
                value.to_record() for value in self.prior_falsifications
            ],
            "snapshot_sha256": self.snapshot_sha256,
        }


@dataclass(frozen=True, slots=True)
class _AuthenticatedActionFields:
    option_id: str
    option_identity_sha256: str
    finite_contract_identity_sha256: str
    option_family: str
    changed_paths: tuple[str, ...]
    compiler_id: str
    compiler_version: int
    compiler_definition_sha256: str


def _action_fields(
    observation: AuthenticatedHypothesisObservation,
) -> _AuthenticatedActionFields | None:
    raw = thaw_json(observation.observed_action)
    if type(raw) is not dict or raw.get("schema_version") != 2:
        return None
    option_id = raw.get("option_id")
    option_identity = raw.get("option_identity_sha256")
    contract_identity = raw.get("finite_contract_identity_sha256")
    family = raw.get("option_family")
    operator_family = raw.get("operator_family")
    changed_paths = raw.get("changed_paths")
    compiler = raw.get("compiler")
    if (
        type(option_id) is not str
        or _OPTION_ID.fullmatch(option_id) is None
        or type(option_identity) is not str
        or type(contract_identity) is not str
        or type(family) is not str
        or _TOKEN.fullmatch(family) is None
        or operator_family != observation.operator_family
        or type(changed_paths) is not list
        or any(type(value) is not str for value in changed_paths)
        or type(compiler) is not dict
        or set(compiler)
        != {"compiler_id", "compiler_version", "definition_sha256"}
    ):
        return None
    compiler_id = compiler.get("compiler_id")
    compiler_version = compiler.get("compiler_version")
    compiler_definition = compiler.get("definition_sha256")
    if (
        type(compiler_id) is not str
        or _TOKEN.fullmatch(compiler_id) is None
        or type(compiler_version) is not int
        or compiler_version <= 0
        or type(compiler_definition) is not str
    ):
        return None
    try:
        require_sha256(option_identity, "observed option identity")
        require_sha256(contract_identity, "observed finite contract identity")
        require_sha256(
            compiler_definition,
            "observed action semantics definition",
        )
        paths = tuple(changed_paths)
        _require_paths(paths, name="observed changed_paths")
    except (TypeError, ValueError):
        return None
    if paths != observation.affected_paths:
        return None
    if contract_identity != observation.finite_contract_identity_sha256:
        return None
    if (
        compiler_id != observation.action_semantics_compiler_id
        or compiler_version != observation.action_semantics_compiler_version
        or compiler_definition != observation.action_semantics_definition_sha256
    ):
        return None
    return _AuthenticatedActionFields(
        option_id=option_id,
        option_identity_sha256=option_identity,
        finite_contract_identity_sha256=contract_identity,
        option_family=family,
        changed_paths=paths,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        compiler_definition_sha256=compiler_definition,
    )


def _path_parts(path: str) -> tuple[str | int, ...]:
    _require_paths((path,), name="local intervention path")
    parts: list[str | int] = []
    index = 2
    while index < len(path):
        start = index
        while index < len(path) and path[index] not in ".[":
            index += 1
        if start == index:
            raise ValueError("local intervention path has an empty object key")
        parts.append(path[start:index])
        while index < len(path) and path[index] == "[":
            end = path.index("]", index)
            parts.append(int(path[index + 1 : end]))
            index = end + 1
        if index < len(path):
            if path[index] != ".":
                raise ValueError("local intervention path is malformed")
            index += 1
    return tuple(parts)


def _value_at_path(root: object, parts: tuple[str | int, ...]) -> object:
    value = root
    for part in parts:
        if type(part) is str:
            if type(value) is not dict or part not in value:
                raise KeyError(part)
            value = value[part]
        else:
            if (
                type(value) is not list
                or part < 0
                or part >= len(value)
            ):
                raise IndexError(part)
            value = value[part]
    return value


def _local_intervention_values(
    observation: AuthenticatedHypothesisObservation,
    path: str,
) -> (
    tuple[FrozenJsonValue, FrozenJsonValue]
    | ReflectionEvidenceExclusionReason
):
    try:
        parts = _path_parts(path)
        parent = freeze_json(
            _value_at_path(thaw_json(observation.parent_configuration), parts)
        )
        child = freeze_json(
            _value_at_path(thaw_json(observation.child_configuration), parts)
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return ReflectionEvidenceExclusionReason.LOCAL_INTERVENTION_UNAVAILABLE
    if (
        len(canonical_typed_json_bytes(parent))
        > MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES
        or len(canonical_typed_json_bytes(child))
        > MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES
    ):
        return ReflectionEvidenceExclusionReason.LOCAL_INTERVENTION_TOO_LARGE
    if typed_json_equal(parent, child):
        return ReflectionEvidenceExclusionReason.NON_CHANGING_LOCAL_INTERVENTION
    return parent, child


def project_identifiable_reflection_evidence(
    observations: tuple[AuthenticatedHypothesisObservation, ...],
    *,
    campaign_sha256: str,
    workload_instance_sha256: str,
    evaluator_contract_sha256: str,
    prior_cutoff_event_index_exclusive: int,
    sealed_cutoff_event_index_inclusive: int,
    prior_falsifications: tuple[ReflectionFalsificationFeedback, ...] = (),
) -> IdentifiableReflectionEvidenceSnapshot:
    """Select only direct, one-path mutations inside an immutable cutoff."""

    for name in (
        "campaign_sha256",
        "workload_instance_sha256",
        "evaluator_contract_sha256",
    ):
        require_sha256(locals()[name], name)
    if type(prior_cutoff_event_index_exclusive) is not int or (
        prior_cutoff_event_index_exclusive < 0
    ):
        raise ValueError("prior cutoff must be a non-negative exact integer")
    if type(sealed_cutoff_event_index_inclusive) is not int or (
        sealed_cutoff_event_index_inclusive <= prior_cutoff_event_index_exclusive
    ):
        raise ValueError("sealed cutoff must advance beyond the prior cutoff")
    if type(observations) is not tuple or any(
        type(value) is not AuthenticatedHypothesisObservation
        for value in observations
    ):
        raise TypeError("observations must contain exact authenticated evidence")
    exclusions: dict[ReflectionEvidenceExclusionReason, int] = {}
    contrasts: list[IdentifiableMutationReflectionContrast] = []

    def exclude(reason: ReflectionEvidenceExclusionReason) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for observation in observations:
        AuthenticatedHypothesisObservation.__post_init__(observation)
        if observation.event_index <= prior_cutoff_event_index_exclusive:
            exclude(ReflectionEvidenceExclusionReason.BEFORE_OR_AT_PRIOR_CUTOFF)
            continue
        if observation.event_index > sealed_cutoff_event_index_inclusive:
            exclude(ReflectionEvidenceExclusionReason.AFTER_SEALED_CUTOFF)
            continue
        if (
            observation.campaign_sha256 != campaign_sha256
            or observation.workload_instance_sha256 != workload_instance_sha256
            or observation.evaluator_contract_sha256 != evaluator_contract_sha256
        ):
            exclude(ReflectionEvidenceExclusionReason.FOREIGN_SCOPE)
            continue
        if observation.provenance is not EvidenceProvenance.DIRECT_MUTATION:
            exclude(ReflectionEvidenceExclusionReason.NON_MUTATION_PROVENANCE)
            continue
        if observation.intervention_identifiability is not (
            InterventionIdentifiability.EXACT_SINGLE
        ):
            exclude(ReflectionEvidenceExclusionReason.NON_SINGLE_INTERVENTION)
            continue
        if len(observation.affected_paths) != 1:
            exclude(ReflectionEvidenceExclusionReason.MULTI_PATH_INTERVENTION)
            continue
        action = _action_fields(observation)
        if action is None:
            exclude(ReflectionEvidenceExclusionReason.MALFORMED_ACTION_SEMANTICS)
            continue
        local_values = _local_intervention_values(
            observation,
            action.changed_paths[0],
        )
        if type(local_values) is ReflectionEvidenceExclusionReason:
            exclude(local_values)
            continue
        parent_local_value, child_local_value = local_values
        kinds = (ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,)
        contrasts.append(
            IdentifiableMutationReflectionContrast(
                contrast_id=observation.observation_sha256,
                source_observation_sha256=observation.observation_sha256,
                source_evidence_id=observation.source_evidence_id,
                event_index=observation.event_index,
                workload_instance_sha256=observation.workload_instance_sha256,
                evaluator_contract_sha256=observation.evaluator_contract_sha256,
                campaign_sha256=observation.campaign_sha256,
                parent_candidate_id=observation.parent_candidate_id,
                child_candidate_id=observation.child_candidate_id,
                operator_invocation_id=observation.operator_invocation_id,
                finite_contract_identity_sha256=(
                    action.finite_contract_identity_sha256
                ),
                action_semantics_compiler_id=action.compiler_id,
                action_semantics_compiler_version=action.compiler_version,
                action_semantics_definition_sha256=(
                    action.compiler_definition_sha256
                ),
                option_id=action.option_id,
                option_identity_sha256=action.option_identity_sha256,
                option_family=action.option_family,
                affected_path=action.changed_paths[0],
                parent_local_value=parent_local_value,
                child_local_value=child_local_value,
                parent_configuration_sha256=(
                    observation.parent_configuration_sha256
                ),
                child_configuration_sha256=(
                    observation.child_configuration_sha256
                ),
                parent_outcome_sha256=observation.parent_outcome_sha256,
                child_outcome_sha256=observation.child_outcome_sha256,
                metrics=observation.metrics,
                mechanism_identifying_design=(
                    observation.mechanism_identifying_design
                ),
                permitted_insight_kinds=tuple(
                    sorted(kinds, key=lambda value: value.value)
                ),
            )
        )
    if not contrasts:
        raise ValueError("sealed cutoff contains no identifiable mutation evidence")
    return IdentifiableReflectionEvidenceSnapshot(
        campaign_sha256=campaign_sha256,
        workload_instance_sha256=workload_instance_sha256,
        evaluator_contract_sha256=evaluator_contract_sha256,
        prior_cutoff_event_index_exclusive=prior_cutoff_event_index_exclusive,
        sealed_cutoff_event_index_inclusive=sealed_cutoff_event_index_inclusive,
        contrasts=tuple(sorted(contrasts, key=lambda value: value.contrast_id)),
        exclusions=tuple(
            sorted(exclusions.items(), key=lambda value: value[0].value)
        ),
        prior_falsifications=prior_falsifications,
    )


__all__ = [
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256",
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID",
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION",
    "MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES",
    "IdentifiableMutationReflectionContrast",
    "IdentifiableMutationReflectionHypothesisCluster",
    "IdentifiableReflectionEvidenceSnapshot",
    "ReflectionEvidenceExclusionReason",
    "ReflectionFalsificationFeedback",
    "cluster_identifiable_mutation_reflection_hypotheses",
    "project_identifiable_reflection_evidence",
]
