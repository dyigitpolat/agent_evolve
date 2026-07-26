"""Transactional runtime adapter for closed-loop campaign insight learning.

This module is the workload-neutral bridge between campaign runtime hooks and
``ClosedLoopCampaignLearning``.  Workload adapters project opaque reflection
results into typed drafts/evidence/audit-plan templates.  The concrete
transactional generation auditor then registers evaluated finite actions and
issues global audit bindings through the sealed falsification gate.  The bridge
owns memory registration, bank-issued diagnostic admissions, pure lifecycle
preparation, and post-memory commit; it never interprets benchmark metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.campaign_execution import (
    CampaignReflectionReceipt,
    CampaignReflectionRequest,
    CampaignReflectionStatus,
    CampaignReflectionTestAdmissionRequest,
    CampaignStageRequest,
)
from agent_evolve.application.campaign_generation_audit import (
    CampaignGenerationAuditPreparation,
    CampaignGenerationAuditProjection,
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.campaign_learning import (
    CampaignDiagnosticAdmissionReceipt,
    CampaignInsightRegistrationReceipt,
    CampaignPreparedLearningBarrier,
    CampaignSemanticAuditPlan,
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryEntry,
    QuarantineTestAdmissionReceipt,
    ReflectedInsightBatchItem,
)
from agent_evolve.domain.finite_variation import FiniteActionEvidenceBinding
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryCreditBatchPreparation,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignPortfolioLearningPreparation,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.memory.global_falsification import (
    GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256,
    HypothesisAuditScope,
    HypothesisClaimStrength,
    HypothesisMetricPrediction,
    TypedEvidencePredicate,
    TypedInterventionSignature,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionEvidenceCatalog,
    ReflectionEvidenceCatalogEntry,
    ReflectionInsightContract,
    ReflectionInsightKind,
    validate_reflection_insight_draft,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPERATOR = re.compile(r"^[a-z][a-z0-9_]*$")
_PATH = re.compile(r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$")
_TEMPLATE_DOMAIN = b"agent-evolve:campaign-semantic-audit-template:v1\x00"
_REFLECTION_PROJECTION_DOMAIN = (
    b"agent-evolve:campaign-reflection-learning-projection:v1\x00"
)
_REFLECTION_REGISTRATION_DOMAIN = (
    b"agent-evolve:campaign-runtime-reflection-registration:v1\x00"
)
_DIAGNOSTIC_EXPOSURE_DOMAIN = (
    b"agent-evolve:campaign-runtime-diagnostic-exposure:v1\x00"
)
_REFLECTION_LEARNING_RECORD_DOMAIN = (
    b"agent-evolve:campaign-reflection-learning-record:v1\x00"
)
_COMPILED_INSIGHT_SEMANTICS_DOMAIN = (
    b"agent-evolve:campaign-compiled-insight-semantics:v1\x00"
)
CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY = "campaign_reflection_learning"
STRUCTURED_REFLECTION_PROJECTION_POLICY_ID = "structured_reflection_learning"
STRUCTURED_REFLECTION_PROJECTION_POLICY_VERSION = 1
STRUCTURED_REFLECTION_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:structured-reflection-learning-projection:v1"
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


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("runtime learning evidence did not freeze to an object")
    return frozen


def _reference_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _canonical_tokens(
    values: tuple[str, ...],
    *,
    name: str,
    pattern: re.Pattern[str],
    allow_empty: bool = False,
) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or pattern.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must contain canonical strings")
    if not allow_empty and not values:
        raise ValueError(f"{name} cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


def _plain_object(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    return value


def _exact_keys(
    value: dict[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = tuple(sorted(expected - observed))
        foreign = tuple(sorted(observed - expected))
        raise ValueError(
            f"{name} fields differ from the canonical schema; "
            f"missing={missing!r}, foreign={foreign!r}"
        )


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise TypeError(f"{name} must be an exact JSON string array")
    return tuple(value)


def _object_tuple(value: object, *, name: str) -> tuple[dict[str, object], ...]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise TypeError(f"{name} must be an exact JSON object array")
    return tuple(value)


def _decode_metric_prediction(
    value: object,
    *,
    name: str,
) -> MetricEffectPrediction:
    record = _plain_object(value, name=name)
    allowed = {"metric_id", "direction", "comparison_anchor"}
    if set(record) not in (
        {"metric_id", "direction"},
        allowed,
    ):
        _exact_keys(record, allowed, name=name)
    metric_id = record.get("metric_id")
    direction = record.get("direction")
    if type(metric_id) is not str or type(direction) is not str:
        raise TypeError(f"{name} metric_id and direction must be exact strings")
    anchor_record = record.get("comparison_anchor")
    anchor: MetricComparisonAnchor | None
    if anchor_record is None:
        anchor = None
    else:
        raw_anchor = _plain_object(anchor_record, name=f"{name}.comparison_anchor")
        _exact_keys(
            raw_anchor,
            {"kind", "source_role_id"},
            name=f"{name}.comparison_anchor",
        )
        kind = raw_anchor["kind"]
        source_role_id = raw_anchor["source_role_id"]
        if type(kind) is not str or (
            source_role_id is not None and type(source_role_id) is not str
        ):
            raise TypeError(f"{name}.comparison_anchor has non-string fields")
        anchor = MetricComparisonAnchor(
            kind=MetricComparisonAnchorKind(kind),
            source_role_id=source_role_id,
        )
    prediction = MetricEffectPrediction(
        metric_id=metric_id,
        direction=MetricEffectDirection(direction),
        comparison_anchor=anchor,
    )
    if prediction.to_record() != record:
        raise ValueError(f"{name} is not the canonical prediction record")
    return prediction


def _decode_insight_draft(value: object, *, name: str) -> InsightDraft:
    record = _plain_object(value, name=name)
    schema_version = record.get("schema_version")
    base_fields = {
        "schema_version",
        "claim",
        "trigger",
        "mechanism",
        "affected_paths",
        "evidence_summary",
        "confidence",
        "evidence_contrast_ids",
        "effect_predictions",
        "recommended_option_families",
        "recommended_option_ids",
        "action_template",
        "falsification_condition",
    }
    semantic_fields = {
        "insight_kind",
        "consumer_scopes",
        "factor_capabilities",
    }
    if schema_version != 2:
        raise ValueError(f"{name} must use semantic-v3 InsightDraft schema version 2")
    _exact_keys(record, base_fields | semantic_fields, name=name)
    text_fields = ("claim", "trigger", "mechanism", "evidence_summary")
    if any(type(record[field]) is not str for field in text_fields):
        raise TypeError(f"{name} prose fields must be exact strings")
    confidence = record["confidence"]
    if type(confidence) is not float:
        raise TypeError(f"{name}.confidence must be an exact JSON float")
    action_template = record["action_template"]
    falsification_condition = record["falsification_condition"]
    insight_kind = record["insight_kind"]
    if (
        type(action_template) is not str
        or type(falsification_condition) is not str
        or type(insight_kind) is not str
    ):
        raise TypeError(f"{name} actionable semantic fields must be exact strings")
    predictions = tuple(
        _decode_metric_prediction(item, name=f"{name}.effect_predictions[{index}]")
        for index, item in enumerate(
            _object_tuple(
                record["effect_predictions"],
                name=f"{name}.effect_predictions",
            )
        )
    )
    consumer_scopes = tuple(
        ReflectionConsumerScope(item)
        for item in _string_tuple(
            record["consumer_scopes"],
            name=f"{name}.consumer_scopes",
        )
    )
    draft = InsightDraft(
        claim=record["claim"],
        trigger=record["trigger"],
        mechanism=record["mechanism"],
        affected_paths=_string_tuple(
            record["affected_paths"],
            name=f"{name}.affected_paths",
        ),
        evidence_summary=record["evidence_summary"],
        confidence=confidence,
        evidence_contrast_ids=_string_tuple(
            record["evidence_contrast_ids"],
            name=f"{name}.evidence_contrast_ids",
        ),
        effect_predictions=predictions,
        recommended_option_families=_string_tuple(
            record["recommended_option_families"],
            name=f"{name}.recommended_option_families",
        ),
        recommended_option_ids=_string_tuple(
            record["recommended_option_ids"],
            name=f"{name}.recommended_option_ids",
        ),
        action_template=action_template,
        falsification_condition=falsification_condition,
        insight_kind=ReflectionInsightKind(insight_kind),
        consumer_scopes=consumer_scopes,
        factor_capabilities=_string_tuple(
            record["factor_capabilities"],
            name=f"{name}.factor_capabilities",
        ),
    )
    if draft.content_record() != record:
        raise ValueError(f"{name} is not the canonical InsightDraft content record")
    return draft


def _decode_insight_contract(value: object) -> ReflectionInsightContract:
    record = _plain_object(value, name="insight_contract")
    fields = {
        "schema_version",
        "contract_identity_sha256",
        "required_metric_ids",
        "allowed_option_families",
        "allowed_option_ids",
        "direction_vocabulary",
        "allowed_decision_paths",
        "allowed_insight_kinds",
        "allowed_consumer_scopes",
        "allowed_comparison_anchor_kinds",
        "allowed_factor_capabilities",
        "allowed_source_role_ids",
    }
    _exact_keys(record, fields, name="insight_contract")
    if record["schema_version"] != 3:
        raise ValueError("insight_contract must use semantic-v3 schema version 3")
    contract = ReflectionInsightContract(
        required_metric_ids=_string_tuple(
            record["required_metric_ids"],
            name="insight_contract.required_metric_ids",
        ),
        allowed_option_families=_string_tuple(
            record["allowed_option_families"],
            name="insight_contract.allowed_option_families",
        ),
        allowed_option_ids=_string_tuple(
            record["allowed_option_ids"],
            name="insight_contract.allowed_option_ids",
        ),
        allowed_decision_paths=_string_tuple(
            record["allowed_decision_paths"],
            name="insight_contract.allowed_decision_paths",
        ),
        allowed_insight_kinds=tuple(
            ReflectionInsightKind(item)
            for item in _string_tuple(
                record["allowed_insight_kinds"],
                name="insight_contract.allowed_insight_kinds",
            )
        ),
        allowed_consumer_scopes=tuple(
            ReflectionConsumerScope(item)
            for item in _string_tuple(
                record["allowed_consumer_scopes"],
                name="insight_contract.allowed_consumer_scopes",
            )
        ),
        allowed_comparison_anchor_kinds=tuple(
            MetricComparisonAnchorKind(item)
            for item in _string_tuple(
                record["allowed_comparison_anchor_kinds"],
                name="insight_contract.allowed_comparison_anchor_kinds",
            )
        ),
        allowed_factor_capabilities=_string_tuple(
            record["allowed_factor_capabilities"],
            name="insight_contract.allowed_factor_capabilities",
        ),
        allowed_source_role_ids=_string_tuple(
            record["allowed_source_role_ids"],
            name="insight_contract.allowed_source_role_ids",
        ),
    )
    if not contract.is_semantic_v3 or contract.to_record() != record:
        raise ValueError("insight_contract is not its canonical semantic-v3 record")
    return contract


def _decode_evidence_catalog(value: object) -> ReflectionEvidenceCatalog:
    record = _plain_object(value, name="evidence_catalog")
    _exact_keys(
        record,
        {"schema_version", "entries", "catalog_identity_sha256"},
        name="evidence_catalog",
    )
    if record["schema_version"] != 1:
        raise ValueError("evidence_catalog must use schema version 1")
    entries_list: list[ReflectionEvidenceCatalogEntry] = []
    for item in _object_tuple(record["entries"], name="evidence_catalog.entries"):
        _exact_keys(
            item,
            {"citation_key", "contrast_id"},
            name="evidence_catalog entry",
        )
        citation_key = item["citation_key"]
        contrast_id = item["contrast_id"]
        if type(citation_key) is not str or type(contrast_id) is not str:
            raise TypeError("catalog entry fields must be exact strings")
        entries_list.append(ReflectionEvidenceCatalogEntry(citation_key, contrast_id))
    entries = tuple(entries_list)
    catalog = ReflectionEvidenceCatalog(entries)
    if catalog.to_record() != record:
        raise ValueError("evidence_catalog is not its canonical record")
    return catalog


def _decode_finite_action_binding(
    value: object,
    *,
    name: str,
) -> FiniteActionEvidenceBinding:
    record = _plain_object(value, name=name)
    _exact_keys(
        record,
        {
            "schema_version",
            "contrast_id",
            "option_id",
            "family",
            "option_identity_sha256",
            "contract_identity_sha256",
            "binding_identity_sha256",
        },
        name=name,
    )
    for field_name in (
        "contrast_id",
        "option_id",
        "family",
        "option_identity_sha256",
        "contract_identity_sha256",
    ):
        if type(record[field_name]) is not str:
            raise TypeError(f"{name}.{field_name} must be an exact string")
    binding = FiniteActionEvidenceBinding(
        contrast_id=record["contrast_id"],
        option_id=record["option_id"],
        family=record["family"],
        option_identity_sha256=record["option_identity_sha256"],
        contract_identity_sha256=record["contract_identity_sha256"],
    )
    if binding.to_record() != record:
        raise ValueError(f"{name} is not its canonical record")
    return binding


def _decode_empirical_snapshot(
    value: object,
    *,
    name: str,
) -> EmpiricalEvidenceSnapshot:
    record = _plain_object(value, name=name)
    _exact_keys(
        record,
        {
            "schema_version",
            "contrast_id",
            "fact_schema_id",
            "fact_schema_version",
            "fact_schema_definition_sha256",
            "facts",
            "optimization_semantics_definition_sha256",
            "action_semantics_definition_sha256",
            "snapshot_sha256",
        },
        name=name,
    )
    string_fields = (
        "contrast_id",
        "fact_schema_id",
        "fact_schema_definition_sha256",
    )
    if any(type(record[field_name]) is not str for field_name in string_fields):
        raise TypeError(f"{name} identity fields must be exact strings")
    if type(record["fact_schema_version"]) is not int:
        raise TypeError(f"{name}.fact_schema_version must be an exact integer")
    for field_name in (
        "optimization_semantics_definition_sha256",
        "action_semantics_definition_sha256",
    ):
        field_value = record[field_name]
        if field_value is not None and type(field_value) is not str:
            raise TypeError(f"{name}.{field_name} must be a string or null")
    facts = freeze_json(_plain_object(record["facts"], name=f"{name}.facts"))
    if type(facts) is not FrozenJsonObject:
        raise AssertionError("empirical facts did not freeze to an object")
    snapshot = EmpiricalEvidenceSnapshot(
        contrast_id=record["contrast_id"],
        fact_schema_id=record["fact_schema_id"],
        fact_schema_version=record["fact_schema_version"],
        fact_schema_definition_sha256=record["fact_schema_definition_sha256"],
        facts=facts,
        optimization_semantics_definition_sha256=record[
            "optimization_semantics_definition_sha256"
        ],
        action_semantics_definition_sha256=record["action_semantics_definition_sha256"],
    )
    if snapshot.to_record() != record:
        raise ValueError(f"{name} is not its canonical record")
    return snapshot


@dataclass(frozen=True, slots=True)
class CampaignReflectionLearningRecord:
    """Canonical provider-independent input to reflection learning.

    The opaque campaign result may carry telemetry or runner-specific trace data
    beside this record.  Only this exact nested schema may create memory cards.
    Engine-owned source IDs and empirical snapshots are therefore separated from
    model-authored :class:`InsightDraft` content before projection.
    """

    reflection_generation_request_sha256: str
    reflection_call_id: LLMCallId
    source_generation: int
    source_stage_receipt_sha256: str
    origin_cutoff_event_index: int
    source_operator_invocation_ids: tuple[OperatorInvocationId, ...]
    source_candidate_ids: tuple[CandidateId, ...]
    evidence_catalog: ReflectionEvidenceCatalog
    insight_contract: ReflectionInsightContract
    insights: tuple[InsightDraft, ...]
    finite_action_bindings: tuple[FiniteActionEvidenceBinding, ...]
    empirical_evidence: tuple[EmpiricalEvidenceSnapshot, ...]
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(
            self.reflection_generation_request_sha256,
            "reflection_generation_request_sha256",
        )
        if type(self.reflection_call_id) is not LLMCallId:
            raise TypeError("reflection_call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.reflection_call_id)
        if type(self.source_generation) is not int or self.source_generation <= 0:
            raise ValueError("source_generation must be a positive exact integer")
        require_sha256(
            self.source_stage_receipt_sha256,
            "source_stage_receipt_sha256",
        )
        if (
            type(self.origin_cutoff_event_index) is not int
            or self.origin_cutoff_event_index < 0
        ):
            raise ValueError("origin_cutoff_event_index must be non-negative")
        for values, expected_type, name in (
            (
                self.source_operator_invocation_ids,
                OperatorInvocationId,
                "source_operator_invocation_ids",
            ),
            (self.source_candidate_ids, CandidateId, "source_candidate_ids"),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(item) is not expected_type for item in values)
            ):
                raise TypeError(f"{name} must contain exact non-empty IDs")
            for item in values:
                expected_type.__post_init__(item)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if type(self.evidence_catalog) is not ReflectionEvidenceCatalog:
            raise TypeError("evidence_catalog must be exact")
        ReflectionEvidenceCatalog.__post_init__(self.evidence_catalog)
        if type(self.insight_contract) is not ReflectionInsightContract:
            raise TypeError("insight_contract must be exact")
        ReflectionInsightContract.__post_init__(self.insight_contract)
        if not self.insight_contract.is_semantic_v3:
            raise ValueError("campaign reflection learning requires semantic-v3")
        if (
            type(self.insights) is not tuple
            or not self.insights
            or any(type(item) is not InsightDraft for item in self.insights)
        ):
            raise ValueError("insights must contain exact InsightDraft values")
        for draft in self.insights:
            validate_reflection_insight_draft(draft, self.insight_contract)
        if len({draft.content_sha256 for draft in self.insights}) != len(self.insights):
            raise ValueError("reflection learning record repeats an insight draft")
        if type(self.finite_action_bindings) is not tuple or any(
            type(item) is not FiniteActionEvidenceBinding
            for item in self.finite_action_bindings
        ):
            raise TypeError("finite_action_bindings must contain exact bindings")
        for item in self.finite_action_bindings:
            FiniteActionEvidenceBinding.__post_init__(item)
        binding_ids = tuple(item.contrast_id for item in self.finite_action_bindings)
        if binding_ids != tuple(sorted(set(binding_ids))):
            raise ValueError("finite action bindings must use canonical contrast order")
        if type(self.empirical_evidence) is not tuple or any(
            type(item) is not EmpiricalEvidenceSnapshot
            for item in self.empirical_evidence
        ):
            raise TypeError("empirical_evidence must contain exact snapshots")
        for item in self.empirical_evidence:
            EmpiricalEvidenceSnapshot.__post_init__(item)
        empirical_ids = tuple(item.contrast_id for item in self.empirical_evidence)
        if empirical_ids != self.evidence_catalog.contrast_ids:
            raise ValueError(
                "empirical evidence must exactly cover the request evidence catalog"
            )
        if not set(binding_ids).issubset(self.evidence_catalog.contrast_ids):
            raise ValueError("finite action binding names a foreign contrast")
        for draft in self.insights:
            ReflectedInsightBatchItem(draft, self._lineage_for(draft))
        object.__setattr__(
            self,
            "record_sha256",
            _hash(_REFLECTION_LEARNING_RECORD_DOMAIN, self._unsigned_record()),
        )

    def _lineage_for(self, draft: InsightDraft) -> InsightEvidenceLineage:
        cited = draft.evidence_contrast_ids
        return InsightEvidenceLineage(
            reflection_call_id=self.reflection_call_id,
            source_operator_invocation_ids=self.source_operator_invocation_ids,
            source_candidate_ids=self.source_candidate_ids,
            available_contrast_ids=self.evidence_catalog.contrast_ids,
            cited_contrast_ids=cited,
            finite_action_bindings=tuple(
                item
                for item in self.finite_action_bindings
                if item.contrast_id in cited
            ),
            empirical_evidence=tuple(
                item for item in self.empirical_evidence if item.contrast_id in cited
            ),
        )

    def lineage_for(self, draft: InsightDraft) -> InsightEvidenceLineage:
        self.__post_init__()
        if type(draft) is not InsightDraft or draft not in self.insights:
            raise ValueError("draft is not part of this reflection learning record")
        return self._lineage_for(draft)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reflection_generation_request_sha256": (
                self.reflection_generation_request_sha256
            ),
            "reflection_call_id": self.reflection_call_id.value,
            "source_generation": self.source_generation,
            "source_stage_receipt_sha256": self.source_stage_receipt_sha256,
            "origin_cutoff_event_index": self.origin_cutoff_event_index,
            "source_operator_invocation_ids": [
                item.value for item in self.source_operator_invocation_ids
            ],
            "source_candidate_ids": [item.value for item in self.source_candidate_ids],
            "evidence_catalog": self.evidence_catalog.to_record(),
            "insight_contract": self.insight_contract.to_record(),
            "insights": [item.content_record() for item in self.insights],
            "finite_action_bindings": [
                item.to_record() for item in self.finite_action_bindings
            ],
            "empirical_evidence": [
                item.to_record() for item in self.empirical_evidence
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "record_sha256": self.record_sha256}


@dataclass(frozen=True, slots=True)
class CampaignReflectionLearningRecordCodec:
    """Strict codec for the one canonical nested campaign learning record."""

    @staticmethod
    def encode(record: CampaignReflectionLearningRecord) -> FrozenJsonObject:
        if type(record) is not CampaignReflectionLearningRecord:
            raise TypeError("record must be exact CampaignReflectionLearningRecord")
        CampaignReflectionLearningRecord.__post_init__(record)
        return _object({CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY: record.to_record()})

    @staticmethod
    def decode(result: FrozenJsonObject) -> CampaignReflectionLearningRecord:
        if type(result) is not FrozenJsonObject or freeze_json(result) is not result:
            raise TypeError("result must be an exact frozen object")
        outer = thaw_json(result)
        if type(outer) is not dict:  # pragma: no cover - exact root above.
            raise AssertionError("frozen reflection result thawed to a non-object")
        if CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY not in outer:
            raise ValueError("reflection result omits the canonical learning record")
        raw = _plain_object(
            outer[CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY],
            name=CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY,
        )
        fields = {
            "schema_version",
            "reflection_generation_request_sha256",
            "reflection_call_id",
            "source_generation",
            "source_stage_receipt_sha256",
            "origin_cutoff_event_index",
            "source_operator_invocation_ids",
            "source_candidate_ids",
            "evidence_catalog",
            "insight_contract",
            "insights",
            "finite_action_bindings",
            "empirical_evidence",
            "record_sha256",
        }
        _exact_keys(raw, fields, name=CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY)
        if raw["schema_version"] != 1:
            raise ValueError("unsupported campaign reflection learning schema")
        for field_name in (
            "reflection_generation_request_sha256",
            "reflection_call_id",
            "source_stage_receipt_sha256",
        ):
            if type(raw[field_name]) is not str:
                raise TypeError(f"{field_name} must be an exact string")
        if (
            type(raw["source_generation"]) is not int
            or type(raw["origin_cutoff_event_index"]) is not int
        ):
            raise TypeError("reflection generation and cutoff must be exact integers")
        record = CampaignReflectionLearningRecord(
            reflection_generation_request_sha256=raw[
                "reflection_generation_request_sha256"
            ],
            reflection_call_id=LLMCallId(raw["reflection_call_id"]),
            source_generation=raw["source_generation"],
            source_stage_receipt_sha256=raw["source_stage_receipt_sha256"],
            origin_cutoff_event_index=raw["origin_cutoff_event_index"],
            source_operator_invocation_ids=tuple(
                OperatorInvocationId(item)
                for item in _string_tuple(
                    raw["source_operator_invocation_ids"],
                    name="source_operator_invocation_ids",
                )
            ),
            source_candidate_ids=tuple(
                CandidateId(item)
                for item in _string_tuple(
                    raw["source_candidate_ids"],
                    name="source_candidate_ids",
                )
            ),
            evidence_catalog=_decode_evidence_catalog(raw["evidence_catalog"]),
            insight_contract=_decode_insight_contract(raw["insight_contract"]),
            insights=tuple(
                _decode_insight_draft(item, name=f"insights[{index}]")
                for index, item in enumerate(
                    _object_tuple(raw["insights"], name="insights")
                )
            ),
            finite_action_bindings=tuple(
                _decode_finite_action_binding(
                    item,
                    name=f"finite_action_bindings[{index}]",
                )
                for index, item in enumerate(
                    _object_tuple(
                        raw["finite_action_bindings"],
                        name="finite_action_bindings",
                    )
                )
            ),
            empirical_evidence=tuple(
                _decode_empirical_snapshot(
                    item,
                    name=f"empirical_evidence[{index}]",
                )
                for index, item in enumerate(
                    _object_tuple(raw["empirical_evidence"], name="empirical_evidence")
                )
            ),
        )
        if record.to_record() != raw:
            raise ValueError("campaign reflection learning record is not canonical")
        return record


@dataclass(frozen=True, slots=True)
class CompiledCampaignInsightSemantics:
    """Workload compilation output bound to one exact draft and evidence lineage."""

    draft_content_sha256: str
    insight_contract_identity_sha256: str
    evidence_lineage_sha256: str
    trigger: TypedEvidencePredicate
    old_value_predicate: TypedEvidencePredicate
    new_action: TypedEvidencePredicate
    matcher_definition_sha256: str
    compiler_policy_id: str
    compiler_policy_version: int
    compiler_definition_sha256: str
    compilation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "draft_content_sha256",
            "insight_contract_identity_sha256",
            "evidence_lineage_sha256",
            "matcher_definition_sha256",
            "compiler_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("trigger", "old_value_predicate", "new_action"):
            value = getattr(self, name)
            if type(value) is not TypedEvidencePredicate:
                raise TypeError(f"{name} must be an exact TypedEvidencePredicate")
            TypedEvidencePredicate.__post_init__(value)
        if (
            type(self.compiler_policy_id) is not str
            or _TOKEN.fullmatch(self.compiler_policy_id) is None
        ):
            raise ValueError("compiler_policy_id must be canonical")
        if (
            type(self.compiler_policy_version) is not int
            or self.compiler_policy_version <= 0
        ):
            raise ValueError("compiler_policy_version must be positive")
        object.__setattr__(
            self,
            "compilation_sha256",
            _hash(_COMPILED_INSIGHT_SEMANTICS_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "draft_content_sha256": self.draft_content_sha256,
            "insight_contract_identity_sha256": (self.insight_contract_identity_sha256),
            "evidence_lineage_sha256": self.evidence_lineage_sha256,
            "trigger_predicate_sha256": self.trigger.predicate_sha256,
            "old_value_predicate_sha256": self.old_value_predicate.predicate_sha256,
            "new_action_predicate_sha256": self.new_action.predicate_sha256,
            "matcher_definition_sha256": self.matcher_definition_sha256,
            "compiler": {
                "policy_id": self.compiler_policy_id,
                "policy_version": self.compiler_policy_version,
                "definition_sha256": self.compiler_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "compilation_sha256": self.compilation_sha256,
        }


@runtime_checkable
class CampaignInsightSemanticCompiler(Protocol):
    """Irreducible workload seam for replayable trigger/action semantics."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def compile(
        self,
        *,
        draft: InsightDraft,
        insight_contract: ReflectionInsightContract,
        evidence_lineage: InsightEvidenceLineage,
    ) -> CompiledCampaignInsightSemantics: ...


@dataclass(frozen=True, slots=True)
class CampaignSemanticAuditPlanTemplate:
    """Reference-free semantic plan projected before memory allocates an ID."""

    trigger: TypedEvidencePredicate
    intervention: TypedInterventionSignature
    predictions: tuple[HypothesisMetricPrediction, ...]
    claim_strength: HypothesisClaimStrength
    scope: HypothesisAuditScope
    matcher_definition_sha256: str
    origin_cutoff_event_index: int
    minimum_support_clusters: int
    minimum_support_instances: int
    audit_policy_definition_sha256: str = GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
    template_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.trigger) is not TypedEvidencePredicate:
            raise TypeError("trigger must be exact")
        TypedEvidencePredicate.__post_init__(self.trigger)
        if type(self.intervention) is not TypedInterventionSignature:
            raise TypeError("intervention must be exact")
        TypedInterventionSignature.__post_init__(self.intervention)
        if (
            type(self.predictions) is not tuple
            or not self.predictions
            or any(
                type(value) is not HypothesisMetricPrediction
                for value in self.predictions
            )
        ):
            raise ValueError("predictions must contain exact typed predictions")
        for value in self.predictions:
            HypothesisMetricPrediction.__post_init__(value)
        if type(self.claim_strength) is not HypothesisClaimStrength:
            raise TypeError("claim_strength must be exact")
        HypothesisClaimStrength.__post_init__(self.claim_strength)
        if type(self.scope) is not HypothesisAuditScope:
            raise TypeError("scope must be exact")
        HypothesisAuditScope.__post_init__(self.scope)
        require_sha256(self.matcher_definition_sha256, "matcher_definition_sha256")
        if (
            type(self.origin_cutoff_event_index) is not int
            or self.origin_cutoff_event_index < 0
        ):
            raise ValueError("origin_cutoff_event_index must be non-negative")
        for name in ("minimum_support_clusters", "minimum_support_instances"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            self.audit_policy_definition_sha256
            != GLOBAL_FALSIFICATION_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported global falsification policy definition")
        object.__setattr__(
            self,
            "template_sha256",
            _hash(_TEMPLATE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "trigger_predicate_sha256": self.trigger.predicate_sha256,
            "intervention_signature_sha256": self.intervention.signature_sha256,
            "predictions": [value.to_record() for value in self.predictions],
            "claim_strength": self.claim_strength.to_record(),
            "scope_sha256": self.scope.scope_sha256,
            "matcher_definition_sha256": self.matcher_definition_sha256,
            "origin_cutoff_event_index": self.origin_cutoff_event_index,
            "minimum_support_clusters": self.minimum_support_clusters,
            "minimum_support_instances": self.minimum_support_instances,
            "audit_policy_definition_sha256": (self.audit_policy_definition_sha256),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "template_sha256": self.template_sha256}

    def validate_source(
        self,
        draft: InsightDraft,
        applicable_operator_kinds: tuple[str, ...],
    ) -> None:
        """Fail before memory publication if a template changes card semantics."""

        if type(draft) is not InsightDraft:
            raise TypeError("draft must be exact")
        InsightDraft.__post_init__(draft)
        _canonical_tokens(
            applicable_operator_kinds,
            name="applicable_operator_kinds",
            pattern=_OPERATOR,
        )
        if not draft.has_semantic_contract or not draft.has_intervention_contract:
            raise ValueError("runtime reflection must be actionable semantic-v3")
        if draft.insight_kind not in {
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            ReflectionInsightKind.MECHANISTIC_CONJECTURE,
        }:
            raise ValueError("runtime reflection kind is not promotion-eligible")
        expected_strength = HypothesisClaimStrength(
            sufficiency=True,
            necessity=False,
            invariance=False,
            mechanistic_or_causal=(
                draft.insight_kind is ReflectionInsightKind.MECHANISTIC_CONJECTURE
            ),
        )
        expected_predictions = tuple(
            (value.metric_id, value.direction, None, None)
            for value in draft.effect_predictions
        )
        observed_predictions = tuple(
            (
                value.metric_id,
                value.direction,
                value.minimum_delta,
                value.maximum_delta,
            )
            for value in self.predictions
        )
        if any(
            value.direction is MetricEffectDirection.UNKNOWN
            for value in draft.effect_predictions
        ):
            raise ValueError("runtime reflection predictions must be known")
        if (
            draft.affected_paths != tuple(sorted(set(draft.affected_paths)))
            or self.intervention.affected_paths != draft.affected_paths
            or self.intervention.admissible_operator_families
            != applicable_operator_kinds
            or observed_predictions != expected_predictions
            or self.claim_strength != expected_strength
        ):
            raise ValueError("semantic audit template differs from reflection draft")

    def bind(self, entry: InsightMemoryEntry) -> CampaignSemanticAuditPlan:
        if type(entry) is not InsightMemoryEntry:
            raise TypeError("entry must be exact")
        InsightMemoryEntry.__post_init__(entry)
        self.validate_source(entry.draft, entry.applicable_operator_kinds)
        return CampaignSemanticAuditPlan(
            reference=entry.reference,
            draft_content_sha256=entry.draft.content_sha256,
            draft_hypothesis_sha256=entry.draft.hypothesis_sha256,
            trigger=self.trigger,
            intervention=self.intervention,
            predictions=self.predictions,
            claim_strength=self.claim_strength,
            scope=self.scope,
            matcher_definition_sha256=self.matcher_definition_sha256,
            origin_cutoff_event_index=self.origin_cutoff_event_index,
            minimum_support_clusters=self.minimum_support_clusters,
            minimum_support_instances=self.minimum_support_instances,
            audit_policy_definition_sha256=(self.audit_policy_definition_sha256),
        )


@dataclass(frozen=True, slots=True)
class CampaignReflectedInsightProjection:
    draft: InsightDraft
    evidence_lineage: InsightEvidenceLineage
    semantic_audit_template: CampaignSemanticAuditPlanTemplate

    def __post_init__(self) -> None:
        if type(self.draft) is not InsightDraft:
            raise TypeError("draft must be exact")
        InsightDraft.__post_init__(self.draft)
        if type(self.evidence_lineage) is not InsightEvidenceLineage:
            raise TypeError("evidence_lineage must be exact")
        InsightEvidenceLineage.__post_init__(self.evidence_lineage)
        if type(self.semantic_audit_template) is not (
            CampaignSemanticAuditPlanTemplate
        ):
            raise TypeError("semantic_audit_template must be exact")
        CampaignSemanticAuditPlanTemplate.__post_init__(self.semantic_audit_template)
        ReflectedInsightBatchItem(self.draft, self.evidence_lineage)


@dataclass(frozen=True, slots=True)
class CampaignReflectionLearningProjection:
    reflection_request_sha256: str
    reflection_receipt_sha256: str
    reflection_result_sha256: str
    insights: tuple[CampaignReflectedInsightProjection, ...]
    applicable_operator_kinds: tuple[str, ...]
    diagnostic_operator_kind: str
    diagnostic_editable_paths: tuple[str, ...]
    initial_score: float
    projection_policy_id: str
    projection_policy_version: int
    projection_policy_definition_sha256: str
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "reflection_request_sha256",
            "reflection_receipt_sha256",
            "reflection_result_sha256",
            "projection_policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.insights) is not tuple
            or not self.insights
            or any(
                type(value) is not CampaignReflectedInsightProjection
                for value in self.insights
            )
        ):
            raise ValueError("insights must contain exact projections")
        for value in self.insights:
            CampaignReflectedInsightProjection.__post_init__(value)
        _canonical_tokens(
            self.applicable_operator_kinds,
            name="applicable_operator_kinds",
            pattern=_OPERATOR,
        )
        if (
            type(self.diagnostic_operator_kind) is not str
            or _OPERATOR.fullmatch(self.diagnostic_operator_kind) is None
            or self.diagnostic_operator_kind not in self.applicable_operator_kinds
        ):
            raise ValueError("diagnostic_operator_kind must be explicitly applicable")
        _canonical_tokens(
            self.diagnostic_editable_paths,
            name="diagnostic_editable_paths",
            pattern=_PATH,
        )
        if any(
            not any(
                _paths_overlap(affected, editable)
                for editable in self.diagnostic_editable_paths
            )
            for item in self.insights
            for affected in item.draft.affected_paths
        ):
            raise ValueError("diagnostic paths do not cover every reflection path")
        if type(self.initial_score) is not float or not math.isfinite(
            self.initial_score
        ):
            raise TypeError("initial_score must be a finite canonical float")
        if (
            type(self.projection_policy_id) is not str
            or _TOKEN.fullmatch(self.projection_policy_id) is None
        ):
            raise ValueError("projection_policy_id must be canonical")
        if (
            type(self.projection_policy_version) is not int
            or self.projection_policy_version <= 0
        ):
            raise ValueError("projection_policy_version must be positive")
        for item in self.insights:
            item.semantic_audit_template.validate_source(
                item.draft,
                self.applicable_operator_kinds,
            )
        object.__setattr__(
            self,
            "projection_sha256",
            _hash(_REFLECTION_PROJECTION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reflection_request_sha256": self.reflection_request_sha256,
            "reflection_receipt_sha256": self.reflection_receipt_sha256,
            "reflection_result_sha256": self.reflection_result_sha256,
            "insights": [
                {
                    "draft_content_sha256": value.draft.content_sha256,
                    "draft_hypothesis_sha256": value.draft.hypothesis_sha256,
                    "evidence_lineage_sha256": value.evidence_lineage.identity_sha256,
                    "semantic_audit_template_sha256": (
                        value.semantic_audit_template.template_sha256
                    ),
                }
                for value in self.insights
            ],
            "applicable_operator_kinds": list(self.applicable_operator_kinds),
            "diagnostic_operator_kind": self.diagnostic_operator_kind,
            "diagnostic_editable_paths": list(self.diagnostic_editable_paths),
            "initial_score_hex": self.initial_score.hex(),
            "projection_policy": {
                "policy_id": self.projection_policy_id,
                "policy_version": self.projection_policy_version,
                "definition_sha256": self.projection_policy_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "projection_sha256": self.projection_sha256}


@runtime_checkable
class CampaignReflectionLearningProjectionPort(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def project(
        self,
        request: CampaignReflectionRequest,
        receipt: CampaignReflectionReceipt,
        result: FrozenJsonObject,
    ) -> CampaignReflectionLearningProjection: ...


@dataclass(frozen=True, slots=True)
class StructuredCampaignReflectionLearningProjector:
    """Generic strict-record projector with injected workload semantics.

    All structural joins, lineage construction, prediction projection, claim
    strength, and audit-template binding are application-owned.  Only typed
    trigger/old-value/new-action compilation remains workload-owned.
    """

    semantic_compiler: CampaignInsightSemanticCompiler
    scope: HypothesisAuditScope
    applicable_operator_kinds: tuple[str, ...]
    diagnostic_operator_kind: str
    diagnostic_editable_paths: tuple[str, ...]
    initial_score: float = 0.0
    minimum_support_clusters: int = 2
    minimum_support_instances: int = 1
    policy_id: str = STRUCTURED_REFLECTION_PROJECTION_POLICY_ID
    policy_version: int = STRUCTURED_REFLECTION_PROJECTION_POLICY_VERSION
    definition_sha256: str = STRUCTURED_REFLECTION_PROJECTION_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_compiler, CampaignInsightSemanticCompiler):
            raise TypeError("semantic_compiler must implement its exact port")
        compiler = self.semantic_compiler
        if (
            type(compiler.policy_id) is not str
            or _TOKEN.fullmatch(compiler.policy_id) is None
        ):
            raise ValueError("semantic compiler policy_id must be canonical")
        if type(compiler.policy_version) is not int or compiler.policy_version <= 0:
            raise ValueError("semantic compiler policy_version must be positive")
        require_sha256(
            compiler.definition_sha256,
            "semantic_compiler.definition_sha256",
        )
        if type(self.scope) is not HypothesisAuditScope:
            raise TypeError("scope must be an exact HypothesisAuditScope")
        HypothesisAuditScope.__post_init__(self.scope)
        _canonical_tokens(
            self.applicable_operator_kinds,
            name="applicable_operator_kinds",
            pattern=_OPERATOR,
        )
        if (
            type(self.diagnostic_operator_kind) is not str
            or _OPERATOR.fullmatch(self.diagnostic_operator_kind) is None
            or self.diagnostic_operator_kind not in self.applicable_operator_kinds
        ):
            raise ValueError("diagnostic_operator_kind must be explicitly applicable")
        _canonical_tokens(
            self.diagnostic_editable_paths,
            name="diagnostic_editable_paths",
            pattern=_PATH,
        )
        if type(self.initial_score) is not float or not math.isfinite(
            self.initial_score
        ):
            raise TypeError("initial_score must be a finite canonical float")
        for name in ("minimum_support_clusters", "minimum_support_instances"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must be canonical")
        if type(self.policy_version) is not int or self.policy_version != 1:
            raise ValueError("unsupported structured projection policy version")
        if self.definition_sha256 != (
            STRUCTURED_REFLECTION_PROJECTION_DEFINITION_SHA256
        ):
            raise ValueError("unsupported structured projection policy definition")

    def _compile(
        self,
        *,
        draft: InsightDraft,
        contract: ReflectionInsightContract,
        lineage: InsightEvidenceLineage,
    ) -> CompiledCampaignInsightSemantics:
        compiled = self.semantic_compiler.compile(
            draft=draft,
            insight_contract=contract,
            evidence_lineage=lineage,
        )
        if type(compiled) is not CompiledCampaignInsightSemantics:
            raise TypeError("semantic compiler returned a foreign value")
        CompiledCampaignInsightSemantics.__post_init__(compiled)
        observed = (
            compiled.draft_content_sha256,
            compiled.insight_contract_identity_sha256,
            compiled.evidence_lineage_sha256,
            compiled.compiler_policy_id,
            compiled.compiler_policy_version,
            compiled.compiler_definition_sha256,
        )
        expected = (
            draft.content_sha256,
            contract.identity_sha256,
            lineage.identity_sha256,
            self.semantic_compiler.policy_id,
            self.semantic_compiler.policy_version,
            self.semantic_compiler.definition_sha256,
        )
        if observed != expected:
            raise ValueError("semantic compilation belongs to foreign inputs or policy")
        return compiled

    def project(
        self,
        request: CampaignReflectionRequest,
        receipt: CampaignReflectionReceipt,
        result: FrozenJsonObject,
    ) -> CampaignReflectionLearningProjection:
        self.__post_init__()
        if type(request) is not CampaignReflectionRequest:
            raise TypeError("request must be an exact CampaignReflectionRequest")
        CampaignReflectionRequest.__post_init__(request)
        if type(receipt) is not CampaignReflectionReceipt:
            raise TypeError("receipt must be an exact CampaignReflectionReceipt")
        CampaignReflectionReceipt.__post_init__(receipt)
        if type(result) is not FrozenJsonObject or freeze_json(result) is not result:
            raise TypeError("result must be an exact frozen object")
        if (
            receipt.request_sha256 != request.request_sha256
            or receipt.quarantined_result != result
        ):
            raise ValueError("reflection receipt differs from request/result")
        if receipt.status is not CampaignReflectionStatus.COMPLETED:
            raise ValueError("failed reflection cannot enter campaign learning")
        if receipt.logical_agent_calls != 1:
            raise ValueError(
                "reflection learning record v1 requires exactly one agentic call"
            )
        record = CampaignReflectionLearningRecordCodec.decode(result)
        if (
            record.source_generation != request.wave.source_generation
            or record.source_generation != receipt.source_generation
            or record.source_stage_receipt_sha256 != request.source_stage.receipt_sha256
            or record.source_stage_receipt_sha256 != receipt.source_stage_receipt_sha256
        ):
            raise ValueError("reflection learning record names a foreign source stage")
        projected: list[CampaignReflectedInsightProjection] = []
        for draft in record.insights:
            lineage = record.lineage_for(draft)
            compiled = self._compile(
                draft=draft,
                contract=record.insight_contract,
                lineage=lineage,
            )
            predictions = tuple(
                HypothesisMetricPrediction(
                    metric_id=value.metric_id,
                    direction=value.direction,
                )
                for value in draft.effect_predictions
            )
            claim_strength = HypothesisClaimStrength(
                sufficiency=True,
                necessity=False,
                invariance=False,
                mechanistic_or_causal=(
                    draft.insight_kind is ReflectionInsightKind.MECHANISTIC_CONJECTURE
                ),
            )
            template = CampaignSemanticAuditPlanTemplate(
                trigger=compiled.trigger,
                intervention=TypedInterventionSignature(
                    affected_paths=draft.affected_paths,
                    old_value_predicate=compiled.old_value_predicate,
                    new_action=compiled.new_action,
                    admissible_operator_families=self.applicable_operator_kinds,
                ),
                predictions=predictions,
                claim_strength=claim_strength,
                scope=self.scope,
                matcher_definition_sha256=compiled.matcher_definition_sha256,
                origin_cutoff_event_index=record.origin_cutoff_event_index,
                minimum_support_clusters=self.minimum_support_clusters,
                minimum_support_instances=self.minimum_support_instances,
            )
            projected.append(
                CampaignReflectedInsightProjection(
                    draft=draft,
                    evidence_lineage=lineage,
                    semantic_audit_template=template,
                )
            )
        return CampaignReflectionLearningProjection(
            reflection_request_sha256=request.request_sha256,
            reflection_receipt_sha256=receipt.receipt_sha256,
            reflection_result_sha256=typed_json_sha256(result),
            insights=tuple(projected),
            applicable_operator_kinds=self.applicable_operator_kinds,
            diagnostic_operator_kind=self.diagnostic_operator_kind,
            diagnostic_editable_paths=self.diagnostic_editable_paths,
            initial_score=self.initial_score,
            projection_policy_id=self.policy_id,
            projection_policy_version=self.policy_version,
            projection_policy_definition_sha256=self.definition_sha256,
        )


@dataclass(frozen=True, slots=True)
class CampaignRuntimeReflectionRegistrationReceipt:
    reflection_receipt_sha256: str
    reflection_result_sha256: str
    projection: CampaignReflectionLearningProjection
    registration: CampaignInsightRegistrationReceipt
    references: tuple[InsightRef, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.reflection_receipt_sha256, "reflection_receipt_sha256")
        require_sha256(self.reflection_result_sha256, "reflection_result_sha256")
        if type(self.projection) is not CampaignReflectionLearningProjection:
            raise TypeError("projection must be exact")
        CampaignReflectionLearningProjection.__post_init__(self.projection)
        if type(self.registration) is not CampaignInsightRegistrationReceipt:
            raise TypeError("registration must be exact")
        CampaignInsightRegistrationReceipt.__post_init__(self.registration)
        if (
            type(self.references) is not tuple
            or not self.references
            or any(type(value) is not InsightRef for value in self.references)
        ):
            raise ValueError("references must contain exact InsightRef values")
        if self.references != tuple(sorted(set(self.references))):
            raise ValueError("references must be unique and canonical")
        if (
            self.reflection_receipt_sha256 != self.projection.reflection_receipt_sha256
            or self.reflection_result_sha256 != self.projection.reflection_result_sha256
            or self.references != tuple(value[0] for value in self.registration.entries)
        ):
            raise ValueError("reflection registration evidence is not joined")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_REFLECTION_REGISTRATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reflection_receipt_sha256": self.reflection_receipt_sha256,
            "reflection_result_sha256": self.reflection_result_sha256,
            "projection_sha256": self.projection.projection_sha256,
            "registration_receipt_sha256": self.registration.receipt_sha256,
            "references": [_reference_record(value) for value in self.references],
            "lifecycle_state": "quarantined",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticExposureReceipt:
    campaign_admission_request_sha256: str
    barrier_generation: int
    reflection_receipt_sha256s: tuple[str, ...]
    registration_receipt_sha256s: tuple[str, ...]
    admission: CampaignDiagnosticAdmissionReceipt
    memory_admission: QuarantineTestAdmissionReceipt
    references: tuple[InsightRef, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(
            self.campaign_admission_request_sha256,
            "campaign_admission_request_sha256",
        )
        for name in (
            "reflection_receipt_sha256s",
            "registration_receipt_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be non-empty")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if type(self.barrier_generation) is not int or self.barrier_generation <= 0:
            raise ValueError("barrier_generation must be positive")
        if type(self.admission) is not CampaignDiagnosticAdmissionReceipt:
            raise TypeError("admission must be exact")
        CampaignDiagnosticAdmissionReceipt.__post_init__(self.admission)
        if type(self.memory_admission) is not QuarantineTestAdmissionReceipt:
            raise TypeError("memory_admission must be exact")
        QuarantineTestAdmissionReceipt.__post_init__(self.memory_admission)
        if (
            type(self.references) is not tuple
            or not self.references
            or any(type(value) is not InsightRef for value in self.references)
        ):
            raise TypeError("references must contain exact InsightRef values")
        if self.references != tuple(sorted(set(self.references))):
            raise ValueError("references must be unique and canonical")
        if (
            self.admission.campaign_admission_request_sha256
            != self.campaign_admission_request_sha256
            or self.admission.admission_generation != self.barrier_generation
            or self.admission.references != self.references
            or self.memory_admission.references != self.references
            or self.admission.memory_admission_receipt_sha256
            != self.memory_admission.receipt_sha256
            or self.registration_receipt_sha256s
            != self.admission.registration_receipt_sha256s
        ):
            raise ValueError("diagnostic exposure evidence is not joined")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_DIAGNOSTIC_EXPOSURE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_admission_request_sha256": (
                self.campaign_admission_request_sha256
            ),
            "barrier_generation": self.barrier_generation,
            "reflection_receipt_sha256s": list(self.reflection_receipt_sha256s),
            "registration_receipt_sha256s": list(self.registration_receipt_sha256s),
            "campaign_diagnostic_admission_receipt_sha256": (
                self.admission.receipt_sha256
            ),
            "memory_admission_receipt_sha256": self.memory_admission.receipt_sha256,
            "references": [_reference_record(value) for value in self.references],
            "scope": "controlled_future_testing_only",
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class _RuntimeGenerationLearningPreparation:
    audit: CampaignGenerationAuditPreparation
    learning: CampaignPreparedLearningBarrier | None

    def __post_init__(self) -> None:
        if type(self.audit) is not CampaignGenerationAuditPreparation:
            raise TypeError("audit must be exact")
        CampaignGenerationAuditPreparation.__post_init__(self.audit)
        if self.learning is not None:
            if type(self.learning) is not CampaignPreparedLearningBarrier:
                raise TypeError("learning must be exact or None")
            CampaignPreparedLearningBarrier.__post_init__(self.learning)
            projection = self.audit.projection
            if projection is None:
                raise ValueError(
                    "learning preparation requires a real audit projection"
                )
            if self.learning.audits != projection.audits:
                raise ValueError("learning preparation differs from real-gate audits")


@dataclass(slots=True)
class ClosedLoopCampaignLearningRuntime:
    """Real coordinator adapter behind campaign learning lifecycle hooks."""

    learning: ClosedLoopCampaignLearning
    reflection_projection: CampaignReflectionLearningProjectionPort
    generation_auditor: TransactionalPortfolioGenerationAuditor
    _registrations: dict[str, CampaignRuntimeReflectionRegistrationReceipt] = field(
        init=False, default_factory=dict
    )
    _exposures: dict[str, CampaignDiagnosticExposureReceipt] = field(
        init=False,
        default_factory=dict,
    )
    _runtime_preparations: dict[str, _RuntimeGenerationLearningPreparation] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if type(self.learning) is not ClosedLoopCampaignLearning:
            raise TypeError("learning must be an exact ClosedLoopCampaignLearning")
        if not isinstance(
            self.reflection_projection,
            CampaignReflectionLearningProjectionPort,
        ):
            raise TypeError("reflection_projection must implement its port")
        if type(self.generation_auditor) is not TransactionalPortfolioGenerationAuditor:
            raise TypeError("generation_auditor must be exact")
        TransactionalPortfolioGenerationAuditor.__post_init__(self.generation_auditor)
        self._validate_port_identity(self.reflection_projection, "reflection")

    @staticmethod
    def _validate_port_identity(port: object, name: str) -> None:
        policy_id = getattr(port, "policy_id", None)
        policy_version = getattr(port, "policy_version", None)
        definition_sha256 = getattr(port, "definition_sha256", None)
        if type(policy_id) is not str or _TOKEN.fullmatch(policy_id) is None:
            raise ValueError(f"{name} projection policy_id must be canonical")
        if type(policy_version) is not int or policy_version <= 0:
            raise ValueError(f"{name} projection policy_version must be positive")
        require_sha256(definition_sha256, f"{name} projection definition_sha256")

    def reflection_completed(
        self,
        request: CampaignReflectionRequest,
        receipt: CampaignReflectionReceipt,
        result: FrozenJsonObject,
    ) -> FrozenJsonObject:
        self.__post_init__()
        if type(request) is not CampaignReflectionRequest:
            raise TypeError("request must be exact")
        CampaignReflectionRequest.__post_init__(request)
        if type(receipt) is not CampaignReflectionReceipt:
            raise TypeError("receipt must be exact")
        CampaignReflectionReceipt.__post_init__(receipt)
        if type(result) is not FrozenJsonObject or freeze_json(result) is not result:
            raise TypeError("result must be an exact frozen object")
        result_sha256 = typed_json_sha256(result)
        if (
            receipt.request_sha256 != request.request_sha256
            or receipt.quarantined_result != result
        ):
            raise ValueError("reflection receipt differs from request/result")
        if receipt.receipt_sha256 in self._registrations:
            raise ValueError("reflection receipt was already registered")
        projection = self.reflection_projection.project(request, receipt, result)
        if type(projection) is not CampaignReflectionLearningProjection:
            raise TypeError("reflection projector returned a foreign value")
        CampaignReflectionLearningProjection.__post_init__(projection)
        expected_identity = (
            self.reflection_projection.policy_id,
            self.reflection_projection.policy_version,
            self.reflection_projection.definition_sha256,
        )
        observed_identity = (
            projection.projection_policy_id,
            projection.projection_policy_version,
            projection.projection_policy_definition_sha256,
        )
        if (
            projection.reflection_request_sha256 != request.request_sha256
            or projection.reflection_receipt_sha256 != receipt.receipt_sha256
            or projection.reflection_result_sha256 != result_sha256
            or observed_identity != expected_identity
        ):
            raise ValueError("reflection learning projection is foreign")
        entries = self.learning.memory.add_reflection_batch(
            tuple(
                ReflectedInsightBatchItem(value.draft, value.evidence_lineage)
                for value in projection.insights
            ),
            initial_score=projection.initial_score,
            applicable_operator_kinds=projection.applicable_operator_kinds,
        )
        plans = tuple(
            item.semantic_audit_template.bind(entry)
            for item, entry in zip(projection.insights, entries, strict=True)
        )
        registration = self.learning.register_quarantined_reflections(
            origin_generation=request.wave.source_generation,
            references=tuple(entry.reference for entry in entries),
            semantic_audit_plans=plans,
        )
        registered = CampaignRuntimeReflectionRegistrationReceipt(
            reflection_receipt_sha256=receipt.receipt_sha256,
            reflection_result_sha256=result_sha256,
            projection=projection,
            registration=registration,
            references=tuple(sorted(entry.reference for entry in entries)),
        )
        self._registrations[receipt.receipt_sha256] = registered
        return _object(registered.to_record())

    def reflections_admitted(
        self,
        request: CampaignReflectionTestAdmissionRequest,
        contents: tuple[
            tuple[CampaignReflectionReceipt, FrozenJsonObject],
            ...,
        ],
    ) -> FrozenJsonObject:
        self.__post_init__()
        if type(request) is not CampaignReflectionTestAdmissionRequest:
            raise TypeError("request must be exact")
        CampaignReflectionTestAdmissionRequest.__post_init__(request)
        if type(contents) is not tuple or not contents:
            raise ValueError("contents must be a non-empty exact tuple")
        content_hashes = tuple(value[0].receipt_sha256 for value in contents)
        if content_hashes != tuple(sorted(set(content_hashes))):
            raise ValueError("contents must use canonical reflection receipt order")
        registrations: list[CampaignRuntimeReflectionRegistrationReceipt] = []
        for reflection, result in contents:
            if type(reflection) is not CampaignReflectionReceipt:
                raise TypeError("content reflection receipt must be exact")
            CampaignReflectionReceipt.__post_init__(reflection)
            if (
                type(result) is not FrozenJsonObject
                or freeze_json(result) is not result
            ):
                raise TypeError("content result must be frozen")
            registered = self._registrations.get(reflection.receipt_sha256)
            if registered is None:
                raise ValueError("reflection content was not registered")
            if (
                registered.reflection_result_sha256 != typed_json_sha256(result)
                or reflection.quarantined_result != result
            ):
                raise ValueError("admission content differs from registration")
            if reflection.receipt_sha256 in self._exposures:
                raise ValueError("reflection was already admitted")
            registrations.append(registered)
        diagnostic_scopes = {
            (
                value.projection.diagnostic_operator_kind,
                value.projection.diagnostic_editable_paths,
            )
            for value in registrations
        }
        if len(diagnostic_scopes) != 1:
            raise ValueError(
                "one atomic reflection barrier requires one diagnostic scope"
            )
        operator_kind, editable_paths = next(iter(diagnostic_scopes))
        references = tuple(
            sorted(
                reference
                for registered in registrations
                for reference in registered.references
            )
        )
        if len(set(references)) != len(references):
            raise ValueError("reflection barrier repeats an insight reference")
        admission = self.learning.admit_for_diagnostic_testing(
            admission_generation=request.barrier.generation,
            references=references,
            campaign_admission_request_sha256=request.request_sha256,
            operator_kind=operator_kind,
            editable_paths=editable_paths,
        )
        memory_admission = self.learning.memory.quarantine_test_admission_receipt(
            admission.memory_admission_receipt_sha256
        )
        exposure = CampaignDiagnosticExposureReceipt(
            campaign_admission_request_sha256=request.request_sha256,
            barrier_generation=request.barrier.generation,
            reflection_receipt_sha256s=content_hashes,
            registration_receipt_sha256s=tuple(
                sorted(value.registration.receipt_sha256 for value in registrations)
            ),
            admission=admission,
            memory_admission=memory_admission,
            references=admission.references,
        )
        for reflection_sha256 in content_hashes:
            self._exposures[reflection_sha256] = exposure
        return _object(
            {
                "schema_version": 1,
                "campaign_admission_request_sha256": request.request_sha256,
                "diagnostic_exposures": [exposure.to_record()],
                "normal_retrieval_mutated": False,
            }
        )

    def diagnostic_exposure(
        self,
        reflection_receipt_sha256: str,
    ) -> CampaignDiagnosticExposureReceipt:
        require_sha256(reflection_receipt_sha256, "reflection_receipt_sha256")
        try:
            return self._exposures[reflection_receipt_sha256]
        except KeyError as exc:
            raise ValueError("reflection has no issued diagnostic exposure") from exc

    def diagnostic_exposures(
        self,
        reflection_receipt_sha256s: tuple[str, ...],
    ) -> tuple[CampaignDiagnosticExposureReceipt, ...]:
        if reflection_receipt_sha256s != tuple(sorted(set(reflection_receipt_sha256s))):
            raise ValueError("reflection receipt hashes must be unique and canonical")
        by_receipt = {
            exposure.receipt_sha256: exposure
            for exposure in (
                self.diagnostic_exposure(value) for value in reflection_receipt_sha256s
            )
        }
        return tuple(by_receipt[value] for value in sorted(by_receipt))

    def normal_references(self, **eligibility: object) -> tuple[InsightRef, ...]:
        """Expose only the memory bank's normal lifecycle-gated retrieval view."""

        return self.learning.memory.eligible_references(**eligibility)

    def _active_diagnostic_exposures(
        self,
    ) -> tuple[CampaignDiagnosticExposureReceipt, ...]:
        active: list[CampaignDiagnosticExposureReceipt] = []
        for exposure in {
            value.receipt_sha256: value for value in self._exposures.values()
        }.values():
            entries = self.learning.memory.entries_for(exposure.references)
            if any(
                entry.lifecycle_state is InsightLifecycleState.QUARANTINED
                for entry in entries
            ):
                active.append(exposure)
        return tuple(sorted(active, key=lambda value: value.receipt_sha256))

    async def prepare_portfolio_generation_close(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        memory_credit_preparation: PortfolioMemoryCreditBatchPreparation,
    ) -> CampaignPortfolioLearningPreparation:
        """Implement the runtime's pure prepare half without publishing state."""

        self.__post_init__()
        if type(request) is not CampaignStageRequest:
            raise TypeError("request must be exact")
        CampaignStageRequest.__post_init__(request)
        if type(memory_credit_preparation) is not (
            PortfolioMemoryCreditBatchPreparation
        ):
            raise TypeError("memory_credit_preparation must be exact")
        PortfolioMemoryCreditBatchPreparation.__post_init__(memory_credit_preparation)
        if memory_credit_preparation.prepared_results != results:
            raise ValueError("learning results differ from memory preparation")
        wave_request_sha256s = tuple(
            value.selection_request.request_sha256 for value in waves
        )
        result_receipt_sha256s = tuple(
            value.receipt.receipt_sha256 for value in results
        )
        active = self._active_diagnostic_exposures()
        references = tuple(
            sorted(
                {
                    entry.reference
                    for exposure in active
                    for entry in self.learning.memory.entries_for(exposure.references)
                    if entry.lifecycle_state is InsightLifecycleState.QUARANTINED
                }
            )
        )
        entries = self.learning.memory.entries_for(references) if references else ()
        plans = self.learning.audit_plans_for(references) if references else ()
        audit_preparation = self.generation_auditor.prepare_generation_audit(
            request=request,
            waves=waves,
            results=results,
            memory_credit_preparation=memory_credit_preparation,
            entries=entries,
            plans=plans,
        )
        projection = audit_preparation.projection
        try:
            if projection is not None:
                batch = memory_credit_preparation.batch_receipt
                if batch is None:  # Closed by the auditor; retained defensively.
                    raise AssertionError("real audit projection lost its memory batch")
                prepared = self.learning.prepare_generation_close(
                    memory_credit_batch=batch,
                    audits=projection.audits,
                    prospective_trials=memory_credit_preparation.expected_trials,
                )
                evidence = _object(
                    {
                        "schema_version": 2,
                        "status": "prepared_closed_loop_learning_real_gate",
                        "generation_audit_preparation": (audit_preparation.to_record()),
                        "coordinator_preparation": prepared.to_record(),
                        "lifecycle_publication_deferred": True,
                    }
                )
                learning_preparation: CampaignPreparedLearningBarrier | None = prepared
            else:
                evidence = _object(
                    {
                        "schema_version": 2,
                        "status": "evidence_append_prepared_no_diagnostic_assignment",
                        "generation_audit_preparation": (audit_preparation.to_record()),
                        "lifecycle_publication_deferred": False,
                    }
                )
                learning_preparation = None
            internal = _RuntimeGenerationLearningPreparation(
                audit=audit_preparation,
                learning=learning_preparation,
            )
            runtime_preparation = CampaignPortfolioLearningPreparation(
                request_sha256=request.request_sha256,
                generation=request.step.generation,
                wave_request_sha256s=wave_request_sha256s,
                result_receipt_sha256s=result_receipt_sha256s,
                memory_credit_preparation_sha256=(
                    memory_credit_preparation.preparation_sha256
                ),
                evidence=evidence,
            )
            if runtime_preparation.preparation_sha256 in self._runtime_preparations:
                raise ValueError("runtime learning preparation identity collided")
        except BaseException:
            self.generation_auditor.abort_generation_audit(audit_preparation)
            raise
        self._runtime_preparations[runtime_preparation.preparation_sha256] = internal
        return runtime_preparation

    def commit_portfolio_generation_close(
        self,
        preparation: CampaignPortfolioLearningPreparation,
    ) -> None:
        """Publish only the already-sealed coordinator decision after memory."""

        if type(preparation) is not CampaignPortfolioLearningPreparation:
            raise TypeError("preparation must be exact")
        CampaignPortfolioLearningPreparation.__post_init__(preparation)
        if preparation.preparation_sha256 not in self._runtime_preparations:
            raise ValueError("runtime learning preparation is absent")
        internal = self._runtime_preparations[preparation.preparation_sha256]
        if internal.learning is not None:
            self.learning.commit_generation_close(internal.learning)
        self.generation_auditor.commit_generation_audit(internal.audit)
        del self._runtime_preparations[preparation.preparation_sha256]

    def abort_portfolio_generation_close(
        self,
        preparation: CampaignPortfolioLearningPreparation,
    ) -> None:
        """Discard transient preparation state without touching memory/lifecycle."""

        if type(preparation) is not CampaignPortfolioLearningPreparation:
            raise TypeError("preparation must be exact")
        CampaignPortfolioLearningPreparation.__post_init__(preparation)
        internal = self._runtime_preparations.pop(
            preparation.preparation_sha256,
            None,
        )
        if internal is not None:
            self.generation_auditor.abort_generation_audit(internal.audit)


__all__ = [
    "CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY",
    "CampaignDiagnosticExposureReceipt",
    "CampaignGenerationAuditProjection",
    "CampaignInsightSemanticCompiler",
    "CampaignReflectedInsightProjection",
    "CampaignReflectionLearningRecord",
    "CampaignReflectionLearningRecordCodec",
    "CampaignReflectionLearningProjection",
    "CampaignReflectionLearningProjectionPort",
    "CampaignRuntimeReflectionRegistrationReceipt",
    "CampaignSemanticAuditPlanTemplate",
    "ClosedLoopCampaignLearningRuntime",
    "CompiledCampaignInsightSemantics",
    "STRUCTURED_REFLECTION_PROJECTION_DEFINITION_SHA256",
    "STRUCTURED_REFLECTION_PROJECTION_POLICY_ID",
    "STRUCTURED_REFLECTION_PROJECTION_POLICY_VERSION",
    "StructuredCampaignReflectionLearningProjector",
]
