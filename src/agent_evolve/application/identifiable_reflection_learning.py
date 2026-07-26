"""Canonical learning projection for identifiable campaign reflection.

The provider-facing reflection request deliberately receives only sealed,
direct, single-mutation evidence.  This module closes the other side of that
boundary: it joins the exact request and validated provider result back to the
typed campaign input, then emits the one canonical learning envelope consumed
by :mod:`agent_evolve.application.campaign_learning_runtime`.

The projection is workload-neutral and side-effect free.  In particular it
does not inspect a source-stage payload, consume recombination results, call a
provider, evaluate a candidate, or mutate campaign memory.
"""

from __future__ import annotations

from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecord,
    CampaignReflectionLearningRecordCodec,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    IdentifiableMutationReflectionContrast,
)
from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
    build_identifiable_reflection_generation_request,
    identifiable_reflection_request_construction_record,
)
from agent_evolve.application.insight_memory import EmpiricalEvidenceSnapshot
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.finite_variation import FiniteActionEvidenceBinding
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightKind,
    validate_reflection_evidence_catalog_result,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("identifiable empirical facts did not freeze to an object")
    return frozen


def _validate_exact_request(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    optimization_semantics: OptimizationSemantics,
) -> dict[str, object]:
    """Replay request construction and return its authenticated identity record."""

    if type(reflection_input) is not CampaignIdentifiableReflectionInput:
        raise TypeError("reflection_input must be exact")
    CampaignIdentifiableReflectionInput.__post_init__(reflection_input)
    if type(request) is not ReflectionGenerationRequest:
        raise TypeError("request must be an exact ReflectionGenerationRequest")
    ReflectionGenerationRequest.__post_init__(request)
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    OptimizationSemantics.__post_init__(optimization_semantics)
    contract = request.insight_contract
    catalog = request.evidence_catalog
    if contract is None or catalog is None:
        raise ValueError(
            "identifiable reflection learning requires request contracts"
        )

    # Rebuilding through the only supported constructor rejects a handcrafted
    # prompt, an overstated epistemic contract, reordered evidence, or foreign
    # optimization semantics even when the outer dataclass remains well typed.
    replay = build_identifiable_reflection_generation_request(
        call_id=request.call_id,
        evidence=reflection_input.evidence,
        insight_contract=contract,
        optimization_semantics=optimization_semantics,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        max_insights=request.max_insights,
        min_insights=request.min_insights,
    )
    if replay != request:
        raise ValueError(
            "reflection request differs from canonical identifiable construction"
        )
    return identifiable_reflection_request_construction_record(
        request,
        reflection_input.evidence,
    )


def _validate_result(
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
) -> None:
    if type(result) is not ReflectionGenerationResult:
        raise TypeError("result must be an exact ReflectionGenerationResult")
    ReflectionGenerationResult.__post_init__(result)
    if type(result.telemetry) is not AgenticCallTelemetry:
        raise TypeError("result telemetry must be exact AgenticCallTelemetry")
    AgenticCallTelemetry.__post_init__(result.telemetry)
    validate_reflection_evidence_catalog_result(request, result)
    if not result.insights:
        raise ValueError(
            "identifiable reflection abstention is unsupported until campaign "
            "memory accepts a typed empty-batch receipt"
        )
    if not request.min_insights <= len(result.insights) <= request.max_insights:
        raise ValueError("reflection result insight count escaped request bounds")
    normalized_claims = tuple(
        " ".join(draft.claim.strip().casefold().split())
        for draft in result.insights
    )
    if len(set(normalized_claims)) != len(normalized_claims):
        raise ValueError("identifiable reflection produced duplicate normalized claims")
    contract = request.insight_contract
    if contract is None:  # pragma: no cover - exact request validation precedes this.
        raise AssertionError("validated identifiable request lost its contract")
    for draft in result.insights:
        validate_reflection_insight_draft(draft, contract)
        if draft.insight_kind is not ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE:
            raise ValueError("identifiable mutation reflection is empirical-only")


def _finite_binding(
    contrast: IdentifiableMutationReflectionContrast,
) -> FiniteActionEvidenceBinding:
    contrast.__post_init__()
    # Finite-contract identity is mandatory in the identifiable contrast v2
    # schema.  Do not silently emit an unbound card if an older projection is
    # injected at runtime.
    require_sha256(
        contrast.finite_contract_identity_sha256,
        "finite_contract_identity_sha256",
    )
    return FiniteActionEvidenceBinding(
        contrast_id=contrast.contrast_id,
        option_id=contrast.option_id,
        family=contrast.option_family,
        option_identity_sha256=contrast.option_identity_sha256,
        contract_identity_sha256=contrast.finite_contract_identity_sha256,
    )


def _empirical_snapshot(
    *,
    contrast: IdentifiableMutationReflectionContrast,
    citation_key: str,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    request_identity_sha256: str,
    optimization_semantics: OptimizationSemantics,
    decision_metrics: DecisionMetricProjection,
    action_semantics_compiler_id: str,
    action_semantics_compiler_version: int,
    action_semantics_definition_sha256: str,
) -> EmpiricalEvidenceSnapshot:
    contract = request.insight_contract
    catalog = request.evidence_catalog
    if contract is None or catalog is None:  # pragma: no cover - validated first.
        raise AssertionError("validated identifiable request lost its contracts")
    facts = _object(
        {
            "schema_version": 1,
            "design_kind": "direct_single_mutation",
            "comparison_anchor": "current_parent",
            "mechanism_identifying_design": False,
            "permitted_insight_kinds": [
                ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE.value
            ],
            "request_binding": {
                "campaign_identifiable_reflection_input_sha256": (
                    reflection_input.input_sha256
                ),
                "identifiable_reflection_request_identity_sha256": (
                    request_identity_sha256
                ),
                "evidence_snapshot_sha256": (
                    reflection_input.evidence.snapshot_sha256
                ),
                "evidence_catalog_identity_sha256": (
                    catalog.catalog_identity_sha256
                ),
                "insight_contract_identity_sha256": contract.identity_sha256,
                "decision_metric_projection_definition_sha256": (
                    decision_metrics.definition_sha256
                ),
                "action_semantics_compiler": {
                    "compiler_id": action_semantics_compiler_id,
                    "compiler_version": action_semantics_compiler_version,
                    "definition_sha256": action_semantics_definition_sha256,
                },
            },
            "source_scope": {
                "source_observation_sha256": (
                    contrast.source_observation_sha256
                ),
                "source_evidence_id": contrast.source_evidence_id,
                "event_index": contrast.event_index,
                "workload_instance_sha256": contrast.workload_instance_sha256,
                "evaluator_contract_sha256": contrast.evaluator_contract_sha256,
                "campaign_sha256": contrast.campaign_sha256,
                "evidence_citation_key": citation_key,
            },
            "occurrence_lineage": {
                "parent_candidate_id": contrast.parent_candidate_id.value,
                "child_candidate_id": contrast.child_candidate_id.value,
                "operator_invocation_id": contrast.operator_invocation_id.value,
            },
            "finite_action": {
                "option_id": contrast.option_id,
                "option_identity_sha256": contrast.option_identity_sha256,
                "option_family": contrast.option_family,
                "finite_contract_identity_sha256": (
                    contrast.finite_contract_identity_sha256
                ),
            },
            "local_intervention": {
                "affected_path": contrast.affected_path,
                "parent_value": thaw_json(contrast.parent_local_value),
                "child_value": thaw_json(contrast.child_local_value),
            },
            "configuration_lineage": {
                "parent_configuration_sha256": (
                    contrast.parent_configuration_sha256
                ),
                "child_configuration_sha256": contrast.child_configuration_sha256,
            },
            "outcome_lineage": {
                "parent_outcome_sha256": contrast.parent_outcome_sha256,
                "child_outcome_sha256": contrast.child_outcome_sha256,
            },
            "observed_metric_effects": [
                value.to_record() for value in contrast.metrics
            ],
        }
    )
    return EmpiricalEvidenceSnapshot(
        contrast_id=contrast.contrast_id,
        fact_schema_id=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
        fact_schema_version=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
        fact_schema_definition_sha256=(
            IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        ),
        facts=facts,
        optimization_semantics_definition_sha256=(
            optimization_semantics.definition_sha256
        ),
        action_semantics_definition_sha256=action_semantics_definition_sha256,
    )


def build_identifiable_campaign_reflection_learning_record(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
    optimization_semantics: OptimizationSemantics,
) -> CampaignReflectionLearningRecord:
    """Join one completed identifiable reflection to canonical campaign learning.

    The trusted action-semantics compiler identity is derived from the
    authenticated contrasts and must be identical across the evidence cohort.
    It is intentionally not inferred from a decision projection, provider
    request, or finite contract; those are distinct authorities.
    """
    construction = _validate_exact_request(
        reflection_input=reflection_input,
        request=request,
        optimization_semantics=optimization_semantics,
    )
    _validate_result(request, result)
    request_identity = construction["request_identity_sha256"]
    if type(request_identity) is not str:  # pragma: no cover - closed helper.
        raise AssertionError("request construction emitted a non-string identity")
    require_sha256(request_identity, "request_identity_sha256")
    evidence = reflection_input.evidence
    catalog = request.evidence_catalog
    contract = request.insight_contract
    if catalog is None or contract is None:  # pragma: no cover - validated first.
        raise AssertionError("validated identifiable request lost its contracts")
    contrasts = evidence.contrasts
    contrast_ids = tuple(value.contrast_id for value in contrasts)
    if contrast_ids != catalog.contrast_ids:
        raise ValueError("evidence catalog order differs from identifiable evidence")
    decision_metrics = DecisionMetricProjection.from_optimization_semantics(
        optimization_semantics
    )
    action_semantics_identities = {
        (
            value.action_semantics_compiler_id,
            value.action_semantics_compiler_version,
            value.action_semantics_definition_sha256,
        )
        for value in contrasts
    }
    if len(action_semantics_identities) != 1:
        raise ValueError(
            "identifiable evidence mixes action-semantics compiler identities"
        )
    (
        action_semantics_compiler_id,
        action_semantics_compiler_version,
        action_semantics_definition_sha256,
    ) = next(iter(action_semantics_identities))
    require_sha256(
        action_semantics_definition_sha256,
        "action_semantics_definition_sha256",
    )
    bindings = tuple(_finite_binding(value) for value in contrasts)
    if tuple(value.contrast_id for value in bindings) != catalog.contrast_ids:
        raise AssertionError("finite binding projection lost canonical catalog order")
    empirical = tuple(
        _empirical_snapshot(
            contrast=value,
            citation_key=catalog.citation_key_for_contrast_id(value.contrast_id),
            reflection_input=reflection_input,
            request=request,
            request_identity_sha256=request_identity,
            optimization_semantics=optimization_semantics,
            decision_metrics=decision_metrics,
            action_semantics_compiler_id=action_semantics_compiler_id,
            action_semantics_compiler_version=(
                action_semantics_compiler_version
            ),
            action_semantics_definition_sha256=(
                action_semantics_definition_sha256
            ),
        )
        for value in contrasts
    )
    query = reflection_input.query
    return CampaignReflectionLearningRecord(
        reflection_generation_request_sha256=request_identity,
        reflection_call_id=request.call_id,
        source_generation=query.wave.source_generation,
        source_stage_receipt_sha256=query.source_stage_receipt_sha256,
        origin_cutoff_event_index=query.sealed_cutoff_event_index_inclusive,
        source_operator_invocation_ids=tuple(
            sorted({value.operator_invocation_id for value in contrasts})
        ),
        source_candidate_ids=tuple(
            sorted(
                {
                    candidate_id
                    for value in contrasts
                    for candidate_id in (
                        value.parent_candidate_id,
                        value.child_candidate_id,
                    )
                }
            )
        ),
        evidence_catalog=catalog,
        insight_contract=contract,
        insights=result.insights,
        finite_action_bindings=bindings,
        empirical_evidence=empirical,
    )


def build_identifiable_campaign_reflection_learning_envelope(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
    optimization_semantics: OptimizationSemantics,
) -> FrozenJsonObject:
    """Return the strict codec envelope expected by campaign memory learning."""

    return CampaignReflectionLearningRecordCodec.encode(
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=optimization_semantics,
        )
    )


__all__ = [
    "build_identifiable_campaign_reflection_learning_envelope",
    "build_identifiable_campaign_reflection_learning_record",
]
