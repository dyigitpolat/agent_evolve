"""Provider-neutral request construction for identifiable campaign reflection.

This boundary turns a sealed, mutation-only evidence snapshot into the exact
structured request consumed by an :class:`AgenticGenerator`.  Workloads inject
optimization semantics and a closed reflection vocabulary; the builder owns no
benchmark prose, provider transport, retry policy, memory state, or evaluator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from agent_evolve.application.identifiable_reflection_evidence import (
    IdentifiableReflectionEvidenceSnapshot,
)
from agent_evolve.core.optimization_semantics import (
    MetricSense,
    OptimizationSemantics,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import (
    MetricComparisonAnchorKind,
    ReflectionEvidenceCatalog,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID = "direct_single_mutation_contrast"
IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION = 2
IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:direct-single-mutation-contrast-facts:v2;"
    b"request-scoped-citation;bounded-local-intervention;"
    b"parent-relative-metric-effects;decimal-and-hex-delta;"
    b"empirical-prediction-only"
).hexdigest()
IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID = (
    "sealed_identifiable_mutation_reflection_request"
)
IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION = 3
IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sealed-identifiable-mutation-reflection-request:v3;"
    b"canonical-typed-evidence-window;request-scoped-citations;"
    b"empirical-only-single-intervention-facts;closed-action-vocabulary;"
    b"explicit-optimization-semantics;decimal-authoritative-magnitude;"
    b"citation-direction-consistency;exact-action-citation-binding;"
    b"exact-insight-cardinality"
).hexdigest()
_REQUEST_IDENTITY_DOMAIN = b"agent-evolve:identifiable-reflection-request:v4\x00"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _identity(value: dict[str, object]) -> str:
    frozen = freeze_json(value)
    return hashlib.sha256(
        _REQUEST_IDENTITY_DOMAIN + canonical_typed_json_bytes(frozen)
    ).hexdigest()


def _validate_semantic_join(
    evidence: IdentifiableReflectionEvidenceSnapshot,
    contract: ReflectionInsightContract,
    semantics: OptimizationSemantics,
) -> None:
    evidence.__post_init__()
    ReflectionInsightContract.__post_init__(contract)
    OptimizationSemantics.__post_init__(semantics)
    if not contract.is_semantic_v3:
        raise ValueError("identifiable reflection requires semantic-v3")
    if contract.allowed_insight_kinds != (
        ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
    ):
        raise ValueError(
            "direct single-mutation evidence permits empirical rules only"
        )
    if contract.allowed_comparison_anchor_kinds != (
        MetricComparisonAnchorKind.CURRENT_PARENT,
    ):
        raise ValueError(
            "identifiable mutation reflection requires current-parent anchors"
        )
    decision_metrics = DecisionMetricProjection.from_optimization_semantics(semantics)
    if decision_metrics.metric_ids != contract.required_metric_ids:
        raise ValueError(
            "decision metric projection differs from the reflection contract"
        )
    if any(
        value.sense not in (MetricSense.MINIMIZE, MetricSense.MAXIMIZE)
        for value in decision_metrics.metrics
    ):
        raise ValueError(
            "identifiable reflection currently supports monotone metrics only"
        )
    allowed_families = set(contract.allowed_option_families)
    allowed_paths = set(contract.allowed_decision_paths)
    allowed_ids = set(contract.allowed_option_ids)
    for contrast in evidence.contrasts:
        if tuple(value.metric_id for value in contrast.metrics) != (
            contract.required_metric_ids
        ):
            raise ValueError("contrast metrics differ from the reflection contract")
        if contrast.option_family not in allowed_families:
            raise ValueError("contrast option family escaped the reflection contract")
        if contrast.affected_path not in allowed_paths:
            raise ValueError("contrast path escaped the reflection contract")
        if allowed_ids and contrast.option_id not in allowed_ids:
            raise ValueError("contrast option ID escaped the reflection contract")
        if contrast.permitted_insight_kinds != contract.allowed_insight_kinds:
            raise ValueError("reflection contract overstates the evidence design")


def bind_reflection_contract_to_evidence_actions(
    contract: ReflectionInsightContract,
    evidence: IdentifiableReflectionEvidenceSnapshot,
) -> ReflectionInsightContract:
    """Require reflections to name an exact action observed in their evidence.

    Family/path-only rules remain useful for descriptive reflection, but they
    are too broad for executable memory: a target-specific prose claim can
    otherwise be silently interpreted as every replacement at the same locus.
    This provider-free projection derives the request-scoped option vocabulary
    from authenticated direct interventions, without parsing model prose or
    consulting outcomes beyond the already sealed evidence window.
    """

    if type(contract) is not ReflectionInsightContract:
        raise TypeError("contract must be an exact ReflectionInsightContract")
    ReflectionInsightContract.__post_init__(contract)
    if type(evidence) is not IdentifiableReflectionEvidenceSnapshot:
        raise TypeError("evidence must be an exact identifiable snapshot")
    evidence.__post_init__()
    exact_ids = tuple(sorted({value.option_id for value in evidence.contrasts}))
    if not exact_ids:  # pragma: no cover - snapshot validation already closes this.
        raise ValueError("exact-action reflection requires at least one action")
    if contract.allowed_option_ids and not set(exact_ids).issubset(
        contract.allowed_option_ids
    ):
        raise ValueError("evidence action IDs escape the adapter vocabulary")
    return replace(contract, allowed_option_ids=exact_ids)


def build_identifiable_reflection_generation_request(
    *,
    call_id: LLMCallId,
    evidence: IdentifiableReflectionEvidenceSnapshot,
    insight_contract: ReflectionInsightContract,
    optimization_semantics: OptimizationSemantics,
    max_output_tokens: int,
    temperature: float | None,
    max_insights: int = 2,
    min_insights: int = 1,
) -> ReflectionGenerationRequest:
    """Build one exact mutation-evidence reflection request without I/O.

    ``min_insights`` remains positive until the campaign learning lifecycle has
    a typed abstention receipt.  Silently accepting an empty batch today would
    fail later at memory publication and misreport a successful reflection.
    """

    if type(call_id) is not LLMCallId:
        raise TypeError("call_id must be an exact LLMCallId")
    LLMCallId.__post_init__(call_id)
    if type(evidence) is not IdentifiableReflectionEvidenceSnapshot:
        raise TypeError("evidence must be an exact identifiable snapshot")
    if type(insight_contract) is not ReflectionInsightContract:
        raise TypeError("insight_contract must be exact")
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    if type(min_insights) is not int or min_insights <= 0:
        raise ValueError(
            "min_insights must stay positive until typed abstention is supported"
        )
    _validate_semantic_join(evidence, insight_contract, optimization_semantics)
    available = tuple(value.contrast_id for value in evidence.contrasts)
    catalog = ReflectionEvidenceCatalog.from_contrast_ids(available)
    decision_metrics = DecisionMetricProjection.from_optimization_semantics(
        optimization_semantics
    )
    prompt = json.dumps(
        {
            "task": (
                "Derive bounded, falsifiable empirical predictive rules for a "
                "later finite-mutation selector. Use only the authenticated "
                "direct single-intervention observations below. Each rule must "
                "name exactly one allowed decision path, predict every required "
                "metric relative to the current parent, recommend only allowed "
                "finite action families or IDs, and cite request-scoped evidence "
                "keys. Each delta_decimal is child minus parent in raw metric "
                "units and is the authoritative text for numerical magnitude; "
                "copy it verbatim when stating a magnitude and do not reinterpret "
                "the binary exponent in delta_hex. Evidence cited by one rule "
                "must agree on affected path, intervention, and predicted metric "
                "directions; otherwise abstain or emit separate bounded rules. "
                "When exact option IDs are available, every rule must recommend "
                "exactly the option ID shared by all evidence keys it cites; do "
                "not generalize that target action to another value at the same "
                "path. "
                "The observations do not identify a causal mechanism: the "
                "required mechanism field is a prospective testable rationale, "
                "not an established mechanistic or causal conclusion."
            ),
            "optimization_semantics": optimization_semantics.to_record(),
            "decision_metric_projection": decision_metrics.to_record(),
            "action_vocabulary": {
                "allowed_decision_paths": list(
                    insight_contract.allowed_decision_paths
                ),
                "allowed_option_families": list(
                    insight_contract.allowed_option_families
                ),
                "allowed_option_ids": list(insight_contract.allowed_option_ids),
                "allowed_insight_kinds": [
                    value.value for value in insight_contract.allowed_insight_kinds
                ],
            },
            "evidence_window": {
                "snapshot_sha256": evidence.snapshot_sha256,
                "prior_cutoff_event_index_exclusive": (
                    evidence.prior_cutoff_event_index_exclusive
                ),
                "sealed_cutoff_event_index_inclusive": (
                    evidence.sealed_cutoff_event_index_inclusive
                ),
                "excluded_observation_counts": [
                    {"reason": reason.value, "count": count}
                    for reason, count in evidence.exclusions
                ],
            },
            "identifiable_mutation_contrasts": [
                value.to_prompt_record(
                    evidence_citation_key=catalog.citation_key_for_contrast_id(
                        value.contrast_id
                    )
                )
                for value in evidence.contrasts
            ],
            "prior_falsifications": [
                value.to_prompt_record() for value in evidence.prior_falsifications
            ],
            "falsification_instruction": (
                "Do not repeat a prior deprecated prediction unless new "
                "single-intervention evidence directly resolves its counterexample."
            ),
            "quarantine": (
                "Outputs are unverified hypotheses and remain quarantined until "
                "a later preregistered diagnostic block closes."
            ),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if any(value in prompt for value in available):
        raise AssertionError("full contrast identities leaked into the model prompt")
    return ReflectionGenerationRequest(
        call_id=call_id,
        operation="extract_identifiable_insights",
        prompt=prompt,
        max_insights=max_insights,
        min_insights=min_insights,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        available_contrast_ids=available,
        insight_contract=insight_contract,
        evidence_catalog=catalog,
    )


def identifiable_reflection_request_construction_record(
    request: ReflectionGenerationRequest,
    evidence: IdentifiableReflectionEvidenceSnapshot,
) -> dict[str, object]:
    """Authenticate the exact prompt, snapshot, contract, and citation map."""

    if type(request) is not ReflectionGenerationRequest:
        raise TypeError("request must be an exact ReflectionGenerationRequest")
    ReflectionGenerationRequest.__post_init__(request)
    if type(evidence) is not IdentifiableReflectionEvidenceSnapshot:
        raise TypeError("evidence must be an exact identifiable snapshot")
    evidence.__post_init__()
    catalog = request.evidence_catalog
    contract = request.insight_contract
    if catalog is None or contract is None:
        raise ValueError("identifiable reflection requires both sealed contracts")
    if request.available_contrast_ids != tuple(
        value.contrast_id for value in evidence.contrasts
    ):
        raise ValueError("request evidence differs from the sealed snapshot")
    parsed = json.loads(request.prompt)
    if type(parsed) is not dict:
        raise ValueError("reflection prompt root must be an object")
    rows = parsed.get("identifiable_mutation_contrasts")
    if type(rows) is not list or any(type(value) is not dict for value in rows):
        raise ValueError("reflection prompt lost its identifiable contrasts")
    observed_keys = tuple(value.get("evidence_citation_key") for value in rows)
    if observed_keys != catalog.citation_keys:
        raise ValueError("prompt citation keys differ from the evidence catalog")
    window = parsed.get("evidence_window")
    if type(window) is not dict or window.get("snapshot_sha256") != (
        evidence.snapshot_sha256
    ):
        raise ValueError("prompt names a foreign evidence snapshot")
    mapping = [value.to_record() for value in catalog.entries]
    identity_record: dict[str, object] = {
        "schema_version": 2,
        "request_builder": {
            "builder_id": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
            "builder_version": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
            "definition_sha256": (
                IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
            ),
        },
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_sha256": _sha_text(request.prompt),
        "max_insights": request.max_insights,
        "min_insights": request.min_insights,
        "max_output_tokens": request.max_output_tokens,
        "temperature_hex": (
            None
            if request.temperature is None
            else float(request.temperature).hex()
        ),
        "evidence_snapshot_sha256": evidence.snapshot_sha256,
        "available_contrast_ids": list(request.available_contrast_ids),
        "evidence_catalog_identity_sha256": catalog.catalog_identity_sha256,
        "insight_contract_identity_sha256": contract.identity_sha256,
        "evidence_citation_mapping_sha256": typed_json_sha256(
            freeze_json({"schema_version": 1, "entries": mapping})
        ),
        "full_contrast_ids_exposed_to_model": False,
    }
    return {
        **identity_record,
        "request_identity_sha256": _identity(identity_record),
        "prompt_utf8_bytes": len(request.prompt.encode("utf-8", errors="strict")),
        "evidence_citation_mapping": mapping,
    }


__all__ = [
    "IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256",
    "IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID",
    "IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION",
    "IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256",
    "IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID",
    "IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION",
    "build_identifiable_reflection_generation_request",
    "bind_reflection_contract_to_evidence_actions",
    "identifiable_reflection_request_construction_record",
]
