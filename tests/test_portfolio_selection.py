from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightOrigin,
    ReflectedInsightBatchItem,
)
from agent_evolve.agentic import (
    CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
    CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
    CardScoreComponent,
    CardTransferAdjudicationRequest,
    CardTransferScoreReceipt,
    FiniteActionEvidenceBinding,
    InsightEvidenceLineage,
    PortfolioCard,
    PortfolioCardPromptPayload,
    PortfolioCardSourceRegistry,
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    PortfolioMemberDraft,
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
    admit_portfolio_card_sources,
    bind_portfolio_experimental_view,
    bind_finite_action_evidence,
    derive_portfolio_card_view,
    portfolio_card_from_insight_entry,
    project_action_neutral_insight_prompt_payload,
    portfolio_card_snapshot_sha256,
    resolve_ranked_portfolio_decision,
    validate_card_transfer_score_receipt,
    validate_ranked_portfolio_decision,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, typed_json_sha256
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PORTFOLIO_SELECTION_TOOL_NAME,
    PydanticAIPortfolioSelectionPolicy,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


def _frozen_object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _contract() -> FiniteVariationContract:
    parent = _frozen_object({"x": 0, "y": 0})
    parent_sha256 = typed_json_sha256(parent)
    options = tuple(
        FiniteVariationOption(
            option_id=option_id,
            parent_configuration_sha256=parent_sha256,
            child_configuration=_frozen_object(child),
            family=family,
            description=description,
        )
        for option_id, family, child, description in (
            ("alpha.x1", "alpha", {"x": 1, "y": 0}, "Increase x by one."),
            ("alpha.x2", "alpha", {"x": 2, "y": 0}, "Increase x by two."),
            ("beta.y1", "beta", {"x": 0, "y": 1}, "Increase y by one."),
            ("gamma.xy", "gamma", {"x": 1, "y": 1}, "Increase both values."),
        )
    )
    return FiniteVariationContract(
        catalog_id="toy_portfolio",
        catalog_version=1,
        catalog_definition_sha256="a" * 64,
        parent_configuration=parent,
        options=options,
    )


def _cards() -> tuple[PortfolioCard, ...]:
    return (
        PortfolioCard(
            card_key="card.a",
            reference=InsightRef(InsightId("insight_portfolio_a"), 1),
            content_sha256="1" * 64,
            evidence_sha256="2" * 64,
            prompt_payload=_frozen_object(
                {"claim": "x changes are useful", "family": "alpha"}
            ),
            score_components=(
                CardScoreComponent(
                    score_id="origin.rank",
                    value=2.0,
                    definition_sha256="5" * 64,
                    evidence_count=3,
                    receipt_sha256s=("6" * 64,),
                ),
            ),
            assigned_score=0.75,
        ),
        PortfolioCard(
            card_key="card.b",
            reference=InsightRef(InsightId("insight_portfolio_b"), 2),
            content_sha256="3" * 64,
            evidence_sha256="4" * 64,
            prompt_payload=_frozen_object(
                {"claim": "y changes are useful", "family": "beta"}
            ),
            score_components=(
                CardScoreComponent(
                    score_id="origin.rank",
                    value=1.0,
                    definition_sha256="7" * 64,
                    evidence_count=4,
                    receipt_sha256s=("8" * 64, "9" * 64),
                ),
            ),
            assigned_score=-0.25,
        ),
    )


def _request(**changes: Any) -> PortfolioSelectionRequest:
    values: dict[str, Any] = {
        "call_id": LLMCallId("call_portfolio_0001"),
        "operation": "select_portfolio",
        "instruction": "Select a diverse portfolio for the supplied context.",
        "context": _frozen_object(
            {"problem": "toy", "parent_role": "held_out"}
        ),
        "finite_variation_contract": _contract(),
        "cards": _cards(),
        "portfolio_size": 3,
        "required_metric_ids": ("cost", "quality"),
        "min_distinct_families": 2,
        "max_output_tokens": 999_999,
        "temperature": 0.1,
    }
    values.update(changes)
    return PortfolioSelectionRequest(**values)


def _predictions(
    cost: MetricEffectDirection = MetricEffectDirection.DECREASE,
    quality: MetricEffectDirection = MetricEffectDirection.INCREASE,
) -> tuple[MetricEffectPrediction, ...]:
    return (
        MetricEffectPrediction("cost", cost),
        MetricEffectPrediction("quality", quality),
    )


def _action_entry(
    *,
    suffix: str,
    option_id: str,
    contrast_id: str,
) -> InsightMemoryEntry:
    contract = _contract()
    option = contract.resolve(option_id)
    binding = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id=option_id,
    )
    draft = InsightDraft(
        claim=f"Action-conditioned claim {suffix}.",
        trigger=f"A compatible bounded parent exposes case {suffix}.",
        mechanism=f"Reuse the exact sealed action for case {suffix}.",
        affected_paths=("$.x",),
        evidence_summary=f"Contrast {suffix} supports the bounded action.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=_predictions(),
        recommended_option_families=(option.family,),
        recommended_option_ids=(option_id,),
        action_template=f"Select {option_id} when the trigger holds.",
        falsification_condition="Reject when held-out effects reverse.",
    )
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId(f"call_card_projection_{suffix}"),
        source_operator_invocation_ids=(
            OperatorInvocationId(f"operator_card_projection_{suffix}"),
        ),
        source_candidate_ids=(
            CandidateId(f"candidate_card_projection_{suffix}"),
        ),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
        finite_action_bindings=(binding,),
    )
    return InsightMemoryEntry(
        reference=InsightRef(InsightId(f"insight_card_projection_{suffix}"), 1),
        draft=draft,
        initial_score=0.0,
        lifecycle_state=InsightLifecycleState.QUARANTINED,
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=lineage,
    )


def _scientific_sources() -> tuple[
    tuple[InsightMemoryEntry, ...],
    tuple[PortfolioCard, ...],
    PortfolioCardSourceRegistry,
]:
    entries = (
        _action_entry(
            suffix="scientific_a",
            option_id="alpha.x1",
            contrast_id="a" * 64,
        ),
        _action_entry(
            suffix="scientific_b",
            option_id="beta.y1",
            contrast_id="b" * 64,
        ),
        _action_entry(
            suffix="scientific_c",
            option_id="gamma.xy",
            contrast_id="c" * 64,
        ),
    )
    cards = tuple(
        portfolio_card_from_insight_entry(
            entry,
            card_key=f"card.{label}",
            prompt_payload=_frozen_object(
                {
                    "claim": claim,
                    "mechanism": "A bounded coordinate change may transfer.",
                }
            ),
            evidence_sha256=str(index + 1) * 64,
            source_receipt_sha256=str(index + 4) * 64,
            score_components=(
                CardScoreComponent(
                    score_id="transfer.rank",
                    value=float(index),
                    definition_sha256=str(index + 7) * 64,
                    evidence_count=index + 1,
                    receipt_sha256s=(str(index + 1) * 64,),
                ),
            ),
            assigned_score=float(index),
        )
        for index, (entry, label, claim) in enumerate(
            zip(
                entries,
                ("a", "b", "c"),
                (
                    "A larger first coordinate improved the observation.",
                    "A larger second coordinate improved the observation.",
                    "A coupled coordinate change improved the observation.",
                ),
                strict=True,
            )
        )
    )
    registry = admit_portfolio_card_sources(entries, cards)
    return entries, cards, registry


def _draft(
    option_id: str,
    *cards: str,
) -> PortfolioMemberDraft:
    return PortfolioMemberDraft(
        option_id=option_id,
        supporting_card_keys=tuple(sorted(cards)),
        effect_predictions=_predictions(),
        design_rationale=f"Use sealed action {option_id}.",
    )


def test_framework_neutral_resolution_binds_request_cards_actions_and_hashes() -> None:
    request = _request()
    decision = resolve_ranked_portfolio_decision(
        request,
        (
            _draft("beta.y1", "card.b"),
            _draft("alpha.x2", "card.a"),
            _draft("gamma.xy", "card.a", "card.b"),
        ),
        policy_id="test_portfolio",
        policy_version=1,
        policy_definition_sha256="f" * 64,
    )

    assert isinstance(
        PydanticAIPortfolioSelectionPolicy(lambda _: None),
        PortfolioSelectionPolicy,
    )
    assert decision.request_sha256 == request.request_sha256
    assert decision.context_sha256 == request.context_sha256
    assert decision.card_snapshot_sha256 == request.card_snapshot_sha256
    assert decision.finite_contract_identity_sha256 == (
        request.finite_variation_contract.identity_sha256
    )
    assert [member.rank for member in decision.members] == [1, 2, 3]
    assert [member.option_id for member in decision.members] == [
        "beta.y1",
        "alpha.x2",
        "gamma.xy",
    ]
    assert len(decision.decision_sha256) == 64
    assert decision.to_record()["decision_sha256"] == decision.decision_sha256
    committed = decision.to_record()
    audit = decision.to_audit_record()
    committed_members = committed["members"]
    audit_members = audit["members"]
    assert type(committed_members) is list
    assert type(audit_members) is list
    assert audit["decision_sha256"] == committed["decision_sha256"]
    assert "design_rationale" not in committed_members[0]
    assert audit_members[0]["design_rationale"] == (
        "Use sealed action beta.y1."
    )
    assert hashlib.sha256(
        audit_members[0]["design_rationale"].encode("utf-8", errors="strict")
    ).hexdigest() == audit_members[0]["design_rationale_sha256"]
    assert audit_members[0]["supporting_card_keys"] == ["card.b"]
    assert audit_members[0]["effect_predictions"] == [
        {"metric_id": "cost", "direction": "decrease"},
        {"metric_id": "quality", "direction": "increase"},
    ]
    assert decision.to_record() == committed
    assert request.cards[0].to_record()["score_components"] == [
        {
            "score_id": "origin.rank",
            "value_hex": float(2.0).hex(),
            "definition_sha256": "5" * 64,
            "evidence_count": 3,
            "receipt_sha256s": ["6" * 64],
        }
    ]
    validate_ranked_portfolio_decision(request, decision)

    altered = replace(
        decision,
        members=(
            replace(decision.members[0], option_identity_sha256="0" * 64),
            *decision.members[1:],
        ),
    )
    with pytest.raises(ValueError, match="sealed finite option"):
        validate_ranked_portfolio_decision(request, altered)


def test_action_conditioned_lineage_projects_immutably_into_portfolio_cards() -> None:
    contract = _contract()
    contrast_id = "a" * 64
    binding = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id="alpha.x1",
    )
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId("call_action_evidence_0001"),
        source_operator_invocation_ids=(
            OperatorInvocationId("operator_action_evidence_0001"),
        ),
        source_candidate_ids=(
            CandidateId("candidate_action_evidence_child"),
            CandidateId("candidate_action_evidence_parent"),
        ),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
        finite_action_bindings=(binding,),
    )
    replay = InsightEvidenceLineage(
        reflection_call_id=lineage.reflection_call_id,
        source_operator_invocation_ids=lineage.source_operator_invocation_ids,
        source_candidate_ids=lineage.source_candidate_ids,
        available_contrast_ids=lineage.available_contrast_ids,
        cited_contrast_ids=lineage.cited_contrast_ids,
        finite_action_bindings=(
            bind_finite_action_evidence(
                contrast_id=contrast_id,
                contract=contract,
                option_id="alpha.x1",
            ),
        ),
    )

    assert type(binding) is FiniteActionEvidenceBinding
    assert lineage.identity_sha256 == replay.identity_sha256
    assert lineage.to_record() == replay.to_record()
    assert lineage.portfolio_action_evidence == (binding,)
    legacy_lineage = replace(lineage, finite_action_bindings=())
    assert legacy_lineage.portfolio_action_evidence == ()
    assert legacy_lineage.identity_sha256 != lineage.identity_sha256
    assert binding.to_record() == {
        "schema_version": 1,
        "contrast_id": contrast_id,
        "option_id": "alpha.x1",
        "family": "alpha",
        "option_identity_sha256": contract.resolve("alpha.x1").identity_sha256,
        "contract_identity_sha256": contract.identity_sha256,
        "binding_identity_sha256": binding.identity_sha256,
    }

    legacy_cards = _cards()
    assert "finite_action_evidence" not in legacy_cards[0].to_record()
    assert "finite_action_evidence" not in legacy_cards[0].prompt_record()
    assert "source_binding" not in legacy_cards[0].to_record()
    assert "derived_view_receipt" not in legacy_cards[0].to_record()
    with pytest.raises(ValueError, match="requires a source binding"):
        replace(
            legacy_cards[0],
            finite_action_evidence=lineage.portfolio_action_evidence,
        )

    entry = _action_entry(
        suffix="lineage_a",
        option_id="alpha.x1",
        contrast_id=contrast_id,
    )
    projected_card = portfolio_card_from_insight_entry(
        entry,
        card_key=legacy_cards[0].card_key,
        prompt_payload=legacy_cards[0].prompt_payload,
        evidence_sha256=legacy_cards[0].evidence_sha256,
        source_receipt_sha256="c" * 64,
        score_components=legacy_cards[0].score_components,
        assigned_score=legacy_cards[0].assigned_score,
    )
    assert projected_card.to_record()["finite_action_evidence"] == [
        binding.to_record()
    ]
    assert projected_card.prompt_record()["finite_action_evidence"] == [
        binding.to_record()
    ]
    assert portfolio_card_snapshot_sha256(
        (projected_card, legacy_cards[1])
    ) != portfolio_card_snapshot_sha256(legacy_cards)


def test_source_projection_and_request_admission_fail_closed_on_tampering() -> None:
    entry = _action_entry(
        suffix="source_a",
        option_id="alpha.x1",
        contrast_id="a" * 64,
    )
    score = (
        CardScoreComponent(
            score_id="transfer.rank",
            value=0.5,
            definition_sha256="b" * 64,
            evidence_count=2,
            receipt_sha256s=("c" * 64,),
        ),
    )
    values = {
        "card_key": "card.a",
        "prompt_payload": _frozen_object({"claim": entry.draft.claim}),
        "evidence_sha256": "d" * 64,
        "source_receipt_sha256": "e" * 64,
        "score_components": score,
        "assigned_score": 0.25,
    }
    card = portfolio_card_from_insight_entry(entry, **values)
    replay = portfolio_card_from_insight_entry(entry, **values)
    registry = admit_portfolio_card_sources((entry,), (card,))
    replay_registry = admit_portfolio_card_sources((entry,), (replay,))

    assert card.source_binding is not None
    assert card.source_binding.reference == entry.reference
    assert card.source_binding.content_sha256 == entry.draft.content_sha256
    assert card.source_binding.evidence_lineage_identity_sha256 == (
        entry.evidence_lineage.identity_sha256
    )
    assert card.source_binding.finite_action_evidence == (
        entry.evidence_lineage.portfolio_action_evidence
    )
    assert card.source_binding.to_record() == replay.source_binding.to_record()
    assert registry.registry_sha256 == replay_registry.registry_sha256
    assert registry.to_record() == replay_registry.to_record()

    with pytest.raises(TypeError, match="trusted application code"):
        PortfolioCardSourceRegistry(source_bindings=(card.source_binding,))
    with pytest.raises(ValueError, match="trusted application admission"):
        _request(cards=(card,))
    admitted_request = _request(cards=(card,), source_registry=registry)
    assert admitted_request.source_registry is registry
    assert admitted_request.to_record()["source_registry_sha256"] == (
        registry.registry_sha256
    )
    with pytest.raises(ValueError, match="cannot mix bound and legacy cards"):
        _request(cards=(card, _cards()[1]), source_registry=registry)

    with pytest.raises(ValueError, match="identity"):
        replace(card, content_sha256="f" * 64)
    with pytest.raises(ValueError, match="without a derived view receipt"):
        replace(card, finite_action_evidence=())
    with pytest.raises(ValueError, match="without a derived view receipt"):
        replace(card, prompt_payload=_frozen_object({"claim": "substituted"}))
    with pytest.raises(ValueError, match="without a derived view receipt"):
        replace(card, evidence_sha256="0" * 64)


def test_derived_views_preserve_source_and_bind_exact_permutation_donors() -> None:
    first_entry = _action_entry(
        suffix="view_a",
        option_id="alpha.x1",
        contrast_id="a" * 64,
    )
    second_entry = _action_entry(
        suffix="view_b",
        option_id="beta.y1",
        contrast_id="b" * 64,
    )
    first_score = (
        CardScoreComponent(
            score_id="transfer.rank",
            value=1.0,
            definition_sha256="1" * 64,
            evidence_count=1,
            receipt_sha256s=("2" * 64,),
        ),
    )
    second_score = (
        CardScoreComponent(
            score_id="transfer.rank",
            value=-1.0,
            definition_sha256="3" * 64,
            evidence_count=4,
            receipt_sha256s=("4" * 64,),
        ),
    )
    first = portfolio_card_from_insight_entry(
        first_entry,
        card_key="card.a",
        prompt_payload=_frozen_object({"view": "source-a"}),
        evidence_sha256="5" * 64,
        source_receipt_sha256="6" * 64,
        score_components=first_score,
        assigned_score=1.0,
    )
    second = portfolio_card_from_insight_entry(
        second_entry,
        card_key="card.b",
        prompt_payload=_frozen_object({"view": "source-b"}),
        evidence_sha256="7" * 64,
        source_receipt_sha256="8" * 64,
        score_components=second_score,
        assigned_score=-1.0,
    )
    derived = derive_portfolio_card_view(
        first,
        prompt_payload=_frozen_object({"view": "blinded-permutation"}),
        evidence_sha256=second.evidence_sha256,
        score_components=second.score_components,
        assigned_score=second.assigned_score,
        transforms=(
            PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION,
            PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
            PortfolioCardViewTransform.PROMPT_PROJECTION,
            PortfolioCardViewTransform.SCORE_PERMUTATION,
        ),
        policy_id="blind_permutation",
        policy_version=1,
        policy_definition_sha256="9" * 64,
        evidence_source_card=second,
        score_source_card=second,
        finite_action_evidence=second.finite_action_evidence,
        action_evidence_source_card=second,
    )
    registry = admit_portfolio_card_sources(
        (first_entry, second_entry),
        (first, second),
    )

    assert derived.source_binding is first.source_binding
    assert derived.finite_action_evidence == second.finite_action_evidence
    assert derived.derived_view_receipt is not None
    assert derived.derived_view_receipt.evidence_source_binding_sha256 == (
        second.source_binding.binding_sha256
    )
    assert derived.derived_view_receipt.score_source_binding_sha256 == (
        second.source_binding.binding_sha256
    )
    assert (
        derived.derived_view_receipt.action_evidence_source_binding_sha256
        == second.source_binding.binding_sha256
    )
    assert derived.prompt_record()["finite_action_evidence"] == [
        binding.to_record() for binding in second.finite_action_evidence
    ]
    assert derived.to_record()["source_binding"] == first.source_binding.to_record()
    assert _request(
        cards=(derived, second),
        source_registry=registry,
    ).cards == (derived, second)

    first_only_registry = admit_portfolio_card_sources(
        (first_entry,),
        (first,),
    )
    with pytest.raises(ValueError, match="evidence source outside the request"):
        _request(
            cards=(derived,),
            source_registry=first_only_registry,
        )

    action_only_permutation = derive_portfolio_card_view(
        first,
        prompt_payload=first.prompt_payload,
        evidence_sha256=first.evidence_sha256,
        score_components=first.score_components,
        assigned_score=first.assigned_score,
        finite_action_evidence=second.finite_action_evidence,
        transforms=(
            PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION,
        ),
        policy_id="action_only_permutation",
        policy_version=1,
        policy_definition_sha256="9" * 64,
        action_evidence_source_card=second,
    )
    with pytest.raises(ValueError, match="action-evidence source outside"):
        _request(
            cards=(action_only_permutation,),
            source_registry=first_only_registry,
        )

    forged_receipt = replace(
        derived.derived_view_receipt,
        derived_evidence_sha256="f" * 64,
    )
    self_consistent_forgery = replace(
        derived,
        evidence_sha256="f" * 64,
        derived_view_receipt=forged_receipt,
    )
    with pytest.raises(ValueError, match="named source view"):
        _request(
            cards=(self_consistent_forgery, second),
            source_registry=registry,
        )

    with pytest.raises(ValueError, match="named source evidence"):
        derive_portfolio_card_view(
            first,
            prompt_payload=first.prompt_payload,
            evidence_sha256="f" * 64,
            score_components=first.score_components,
            assigned_score=first.assigned_score,
            transforms=(PortfolioCardViewTransform.EVIDENCE_PERMUTATION,),
            policy_id="bad_permutation",
            policy_version=1,
            policy_definition_sha256="0" * 64,
            evidence_source_card=second,
        )

    with pytest.raises(ValueError, match="named source action evidence"):
        derive_portfolio_card_view(
            first,
            prompt_payload=first.prompt_payload,
            evidence_sha256=first.evidence_sha256,
            score_components=first.score_components,
            assigned_score=first.assigned_score,
            finite_action_evidence=first.finite_action_evidence,
            transforms=(
                PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION,
            ),
            policy_id="bad_action_permutation",
            policy_version=1,
            policy_definition_sha256="0" * 64,
            action_evidence_source_card=second,
        )

    redacted = derive_portfolio_card_view(
        first,
        prompt_payload=_frozen_object({"view": "action-redacted"}),
        evidence_sha256=first.evidence_sha256,
        score_components=first.score_components,
        assigned_score=first.assigned_score,
        finite_action_evidence=(),
        transforms=(
            PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION,
            PortfolioCardViewTransform.PROMPT_PROJECTION,
        ),
        policy_id="action_redaction",
        policy_version=1,
        policy_definition_sha256="0" * 64,
    )
    assert "finite_action_evidence" not in redacted.prompt_record()


def test_scientific_m_view_is_pristine_typed_and_request_bound() -> None:
    _, cards, registry = _scientific_sources()
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.MEMORY,
        cards=cards,
        finite_variation_contract=_contract(),
        source_registry=registry,
        policy_id="scientific_mpn",
        policy_version=1,
        policy_definition_sha256="a" * 64,
    )
    request = _request(
        cards=cards,
        source_registry=registry,
        experimental_view_receipt=receipt,
    )

    typed = cards[0].typed_prompt_payload
    assert type(typed) is PortfolioCardPromptPayload
    assert typed.action_neutral_payload is cards[0].prompt_payload
    assert typed.finite_action_evidence == cards[0].finite_action_evidence
    assert typed.prompt_record()["action_neutral_payload"] == {
        "claim": "A larger first coordinate improved the observation.",
        "mechanism": "A bounded coordinate change may transfer.",
    }
    assert request.experimental_view_receipt is receipt
    assert request.to_record()["experimental_view_receipt_sha256"] == (
        receipt.receipt_sha256
    )

    projected = derive_portfolio_card_view(
        cards[0],
        prompt_payload=_frozen_object({"claim": "A generic projection."}),
        evidence_sha256=cards[0].evidence_sha256,
        score_components=cards[0].score_components,
        assigned_score=cards[0].assigned_score,
        transforms=(PortfolioCardViewTransform.PROMPT_PROJECTION,),
        policy_id="legacy_projection",
        policy_version=1,
        policy_definition_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="M requires pristine"):
        bind_portfolio_experimental_view(
            arm=PortfolioExperimentalArm.MEMORY,
            cards=(projected, *cards[1:]),
            finite_variation_contract=_contract(),
            source_registry=registry,
            policy_id="scientific_mpn",
            policy_version=1,
            policy_definition_sha256="a" * 64,
        )

    # The same arbitrary projection remains available to explicitly
    # non-scientific callers that omit an experimental arm receipt.
    non_scientific = _request(
        cards=(projected, *cards[1:]),
        source_registry=registry,
    )
    assert non_scientific.experimental_view_receipt is None


@pytest.mark.parametrize(
    "leaked_value",
    (
        "Recommend alpha.x1 for this parent.",
        "Recommend alpha.x1.",
        "Recommend ALPHA.X1 for this parent.",
        _contract().resolve("alpha.x1").identity_sha256,
        _contract().identity_sha256,
    ),
)
def test_scientific_prompt_payload_rejects_exact_action_attribution_leaks(
    leaked_value: str,
) -> None:
    entry = _action_entry(
        suffix="scientific_leak",
        option_id="alpha.x1",
        contrast_id="a" * 64,
    )
    card = portfolio_card_from_insight_entry(
        entry,
        card_key="card.leak",
        prompt_payload=_frozen_object(
            {
                "claim": "Nominally action-neutral prose.",
                "nested": {"deep": [leaked_value]},
            }
        ),
        evidence_sha256="b" * 64,
        source_receipt_sha256="c" * 64,
    )
    registry = admit_portfolio_card_sources((entry,), (card,))

    with pytest.raises(ValueError, match="action-neutral payload contains"):
        bind_portfolio_experimental_view(
            arm=PortfolioExperimentalArm.MEMORY,
            cards=(card,),
            finite_variation_contract=_contract(),
            source_registry=registry,
            policy_id="scientific_mpn",
            policy_version=1,
            policy_definition_sha256="d" * 64,
        )


def test_action_neutral_insight_projection_redacts_exact_identity_deterministically() -> None:
    contract = _contract()
    contrast_id = "a" * 64
    entry = _action_entry(
        suffix="scientific_projection",
        option_id="alpha.x1",
        contrast_id=contrast_id,
    )
    binding = entry.evidence_lineage.finite_action_bindings[0]
    payload = _frozen_object(
        {
            "claim": (
                "ALPHA.X1 improved the generic mechanism; preserve this prose. "
                f"Evidence {contrast_id}."
            ),
            f"identity.{binding.option_identity_sha256}": [
                contract.identity_sha256,
                contract.resolve("alpha.x2").identity_sha256,
            ],
        }
    )
    projected = project_action_neutral_insight_prompt_payload(
        entry,
        prompt_payload=payload,
        finite_variation_contracts=(contract,),
    )
    replay = project_action_neutral_insight_prompt_payload(
        entry,
        prompt_payload=payload,
        finite_variation_contracts=(contract,),
    )
    assert projected == replay
    rendered = repr(projected).casefold()
    for forbidden in (
        "alpha.x1",
        contrast_id,
        binding.option_identity_sha256,
        contract.identity_sha256,
        contract.resolve("alpha.x2").identity_sha256,
    ):
        assert forbidden not in rendered
    assert "preserve this prose" in rendered

    card = portfolio_card_from_insight_entry(
        entry,
        card_key="card.projected",
        prompt_payload=projected,
        evidence_sha256="b" * 64,
        source_receipt_sha256="c" * 64,
    )
    assert card.finite_action_evidence[0].option_id == "alpha.x1"
    registry = admit_portfolio_card_sources((entry,), (card,))
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.MEMORY,
        cards=(card,),
        finite_variation_contract=contract,
        source_registry=registry,
        policy_id="scientific_mpn",
        policy_version=1,
        policy_definition_sha256="d" * 64,
    )
    assert receipt.arm is PortfolioExperimentalArm.MEMORY


def test_scientific_p_view_is_one_bijective_deranged_donor_per_card() -> None:
    _, cards, registry = _scientific_sources()
    donors = (cards[1], cards[2], cards[0])
    transforms = (
        PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
        PortfolioCardViewTransform.PROMPT_PERMUTATION,
        PortfolioCardViewTransform.SCORE_PERMUTATION,
    )
    placebo = tuple(
        derive_portfolio_card_view(
            source,
            prompt_payload=donor.prompt_payload,
            evidence_sha256=donor.evidence_sha256,
            score_components=donor.score_components,
            assigned_score=donor.assigned_score,
            transforms=transforms,
            policy_id="scientific_p",
            policy_version=1,
            policy_definition_sha256="e" * 64,
            prompt_source_card=donor,
            evidence_source_card=donor,
            score_source_card=donor,
        )
        for source, donor in zip(cards, donors, strict=True)
    )
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
        cards=placebo,
        finite_variation_contract=_contract(),
        source_registry=registry,
        policy_id="scientific_mpn",
        policy_version=1,
        policy_definition_sha256="f" * 64,
    )
    request = _request(
        cards=placebo,
        source_registry=registry,
        experimental_view_receipt=receipt,
    )

    assert request.experimental_view_receipt is receipt
    assert {donor for _, donor in receipt.source_donor_binding_pairs} == {
        source for source, _ in receipt.source_donor_binding_pairs
    }
    assert all(
        source != donor
        for source, donor in receipt.source_donor_binding_pairs
    )
    assert all(
        card.finite_action_evidence
        == card.source_binding.finite_action_evidence
        and card.derived_view_receipt.action_evidence_source_binding_sha256
        is None
        for card in placebo
        if card.source_binding is not None
        and card.derived_view_receipt is not None
    )

    # Rotating the action binding with the donor would preserve a coherent
    # fact/action unit and reduce P to an opaque relabeling of M.
    relabeled_coherent_unit = derive_portfolio_card_view(
        cards[0],
        prompt_payload=cards[1].prompt_payload,
        evidence_sha256=cards[1].evidence_sha256,
        score_components=cards[1].score_components,
        assigned_score=cards[1].assigned_score,
        finite_action_evidence=cards[1].finite_action_evidence,
        transforms=tuple(
            sorted(
                (
                    *transforms,
                    PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION,
                ),
                key=lambda value: value.value,
            )
        ),
        policy_id="scientific_p",
        policy_version=1,
        policy_definition_sha256="e" * 64,
        prompt_source_card=cards[1],
        evidence_source_card=cards[1],
        score_source_card=cards[1],
        action_evidence_source_card=cards[1],
    )
    with pytest.raises(ValueError, match="retaining source action evidence"):
        bind_portfolio_experimental_view(
            arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
            cards=(relabeled_coherent_unit, *placebo[1:]),
            finite_variation_contract=_contract(),
            source_registry=registry,
            policy_id="scientific_mpn",
            policy_version=1,
            policy_definition_sha256="f" * 64,
        )

    # Reusing one donor breaks the population-level bijection even though each
    # individual card still names a valid, non-self source.
    duplicate_donor = derive_portfolio_card_view(
        cards[2],
        prompt_payload=cards[1].prompt_payload,
        evidence_sha256=cards[1].evidence_sha256,
        score_components=cards[1].score_components,
        assigned_score=cards[1].assigned_score,
        transforms=transforms,
        policy_id="scientific_p",
        policy_version=1,
        policy_definition_sha256="e" * 64,
        prompt_source_card=cards[1],
        evidence_source_card=cards[1],
        score_source_card=cards[1],
    )
    with pytest.raises(ValueError, match="bijection"):
        bind_portfolio_experimental_view(
            arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
            cards=(*placebo[:2], duplicate_donor),
            finite_variation_contract=_contract(),
            source_registry=registry,
            policy_id="scientific_mpn",
            policy_version=1,
            policy_definition_sha256="f" * 64,
        )

    # A per-card compartment split is rejected before a scientific request can
    # be issued.
    split_donor = derive_portfolio_card_view(
        cards[0],
        prompt_payload=cards[1].prompt_payload,
        evidence_sha256=cards[2].evidence_sha256,
        score_components=cards[1].score_components,
        assigned_score=cards[1].assigned_score,
        transforms=transforms,
        policy_id="scientific_p",
        policy_version=1,
        policy_definition_sha256="e" * 64,
        prompt_source_card=cards[1],
        evidence_source_card=cards[2],
        score_source_card=cards[1],
    )
    with pytest.raises(ValueError, match="one donor"):
        bind_portfolio_experimental_view(
            arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
            cards=(split_donor, *placebo[1:]),
            finite_variation_contract=_contract(),
            source_registry=registry,
            policy_id="scientific_mpn",
            policy_version=1,
            policy_definition_sha256="f" * 64,
        )


def test_scientific_n_view_requires_canonical_redaction_in_every_compartment() -> None:
    _, cards, registry = _scientific_sources()
    transforms = (
        PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION,
        PortfolioCardViewTransform.EVIDENCE_REDACTION,
        PortfolioCardViewTransform.PROMPT_REDACTION,
        PortfolioCardViewTransform.SCORE_REDACTION,
    )
    neutral = tuple(
        derive_portfolio_card_view(
            source,
            prompt_payload=CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
            evidence_sha256=CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
            score_components=(),
            assigned_score=None,
            finite_action_evidence=(),
            transforms=transforms,
            policy_id="scientific_n",
            policy_version=1,
            policy_definition_sha256="a" * 64,
        )
        for source in cards
    )
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.NEUTRAL,
        cards=neutral,
        finite_variation_contract=_contract(),
        source_registry=registry,
        policy_id="scientific_mpn",
        policy_version=1,
        policy_definition_sha256="b" * 64,
    )
    request = _request(
        cards=neutral,
        source_registry=registry,
        experimental_view_receipt=receipt,
    )
    assert request.cards[0].typed_prompt_payload.prompt_record() == {
        "action_neutral_payload": {},
        "finite_action_evidence": [],
    }

    with pytest.raises(ValueError, match="canonical neutral"):
        derive_portfolio_card_view(
            cards[0],
            prompt_payload=_frozen_object({"neutral": True}),
            evidence_sha256=CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
            score_components=(),
            assigned_score=None,
            finite_action_evidence=(),
            transforms=transforms,
            policy_id="scientific_n",
            policy_version=1,
            policy_definition_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="canonical sentinel"):
        derive_portfolio_card_view(
            cards[0],
            prompt_payload=CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
            evidence_sha256="f" * 64,
            score_components=(),
            assigned_score=None,
            finite_action_evidence=(),
            transforms=transforms,
            policy_id="scientific_n",
            policy_version=1,
            policy_definition_sha256="a" * 64,
        )


def test_action_conditioned_lineage_fails_closed_on_detached_or_noncanonical_data() -> (
    None
):
    contract = _contract()
    first = bind_finite_action_evidence(
        contrast_id="a" * 64,
        contract=contract,
        option_id="alpha.x1",
    )
    second = bind_finite_action_evidence(
        contrast_id="b" * 64,
        contract=contract,
        option_id="beta.y1",
    )
    base = {
        "reflection_call_id": LLMCallId("call_action_evidence_invalid"),
        "source_operator_invocation_ids": (
            OperatorInvocationId("operator_action_evidence_invalid"),
        ),
        "source_candidate_ids": (
            CandidateId("candidate_action_evidence_invalid"),
        ),
        "available_contrast_ids": ("a" * 64, "b" * 64),
        "cited_contrast_ids": ("a" * 64, "b" * 64),
    }

    with pytest.raises(ValueError, match="canonical contrast order"):
        InsightEvidenceLineage(
            **base,
            finite_action_bindings=(second, first),
        )
    with pytest.raises(ValueError, match="bind a cited"):
        InsightEvidenceLineage(
            **{**base, "cited_contrast_ids": ("a" * 64,)},
            finite_action_bindings=(second,),
        )
    with pytest.raises(ValueError, match="outside the sealed contract"):
        bind_finite_action_evidence(
            contrast_id="a" * 64,
            contract=contract,
            option_id="foreign.option",
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        FiniteActionEvidenceBinding(
            contrast_id="A" * 64,
            option_id="alpha.x1",
            family="alpha",
            option_identity_sha256=contract.resolve("alpha.x1").identity_sha256,
            contract_identity_sha256=contract.identity_sha256,
        )


def test_exact_action_reflection_requires_complete_matching_action_evidence() -> None:
    contract = _contract()
    contrast_id = "a" * 64
    draft = InsightDraft(
        claim="The observed alpha action may transfer.",
        trigger="A compatible parent exposes the same bounded action family.",
        mechanism="Select the exact action supported by the cited contrast.",
        affected_paths=("$.x",),
        evidence_summary="One finite-action contrast supports the hypothesis.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=_predictions(),
        recommended_option_families=("alpha",),
        recommended_option_ids=("alpha.x1",),
        action_template="Choose the exact alpha action bound to the evidence.",
        falsification_condition="Falsify if the held-out direction reverses.",
    )
    lineage_values = {
        "reflection_call_id": LLMCallId("call_exact_action_evidence"),
        "source_operator_invocation_ids": (
            OperatorInvocationId("operator_exact_action_evidence"),
        ),
        "source_candidate_ids": (
            CandidateId("candidate_exact_action_evidence"),
        ),
        "available_contrast_ids": (contrast_id,),
        "cited_contrast_ids": (contrast_id,),
    }
    with pytest.raises(ValueError, match="one finite action binding per citation"):
        ReflectedInsightBatchItem(
            draft=draft,
            evidence_lineage=InsightEvidenceLineage(**lineage_values),
        )

    wrong = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id="beta.y1",
    )
    with pytest.raises(ValueError, match="recommendation differs"):
        ReflectedInsightBatchItem(
            draft=draft,
            evidence_lineage=InsightEvidenceLineage(
                **lineage_values,
                finite_action_bindings=(wrong,),
            ),
        )

    correct = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id="alpha.x1",
    )
    accepted = ReflectedInsightBatchItem(
        draft=draft,
        evidence_lineage=InsightEvidenceLineage(
            **lineage_values,
            finite_action_bindings=(correct,),
        ),
    )
    assert accepted.evidence_lineage.portfolio_action_evidence == (correct,)


def test_transfer_score_receipt_binds_typed_projections_and_source_lineage() -> None:
    request = CardTransferAdjudicationRequest(
        card_key="card.a",
        reference=InsightRef(InsightId("insight_portfolio_a"), 1),
        prediction=_frozen_object({"cost": "decrease", "quality": "increase"}),
        outcome=_frozen_object({"cost_delta": -1.0, "quality_delta": 2.0}),
        source_receipt_sha256s=("a" * 64, "b" * 64),
    )
    receipt = CardTransferScoreReceipt(
        request_sha256=request.request_sha256,
        score_component=CardScoreComponent(
            score_id="transfer.direction_accuracy",
            value=1.0,
            definition_sha256="c" * 64,
            evidence_count=2,
            receipt_sha256s=request.source_receipt_sha256s,
        ),
        adjudicator_id="toy_transfer",
        adjudicator_version=1,
        adjudicator_definition_sha256="d" * 64,
    )

    validate_card_transfer_score_receipt(request, receipt)
    assert len(receipt.receipt_sha256) == 64
    with pytest.raises(ValueError, match="different transfer request"):
        validate_card_transfer_score_receipt(
            replace(request, outcome=_frozen_object({"cost_delta": 1.0})),
            receipt,
        )


@pytest.mark.parametrize(
    ("drafts", "message"),
    (
        (
            (
                _draft("alpha.x1", "card.a"),
                _draft("alpha.x1", "card.b"),
                _draft("beta.y1", "card.a"),
            ),
            "repeats a finite option",
        ),
        (
            (
                _draft("alpha.x1", "card.a"),
                _draft("alpha.x2", "card.b"),
                _draft("beta.y1", "card.foreign"),
            ),
            "outside the request snapshot",
        ),
        (
            (
                _draft("alpha.x1", "card.a"),
                _draft("alpha.x2", "card.b"),
                _draft("gamma.xy", "card.a"),
            ),
            "min_distinct_families",
        ),
    ),
)
def test_resolution_is_all_or_nothing(
    drafts: tuple[PortfolioMemberDraft, ...],
    message: str,
) -> None:
    request = _request(min_distinct_families=3)
    with pytest.raises(ValueError, match=message):
        resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id="test_portfolio",
            policy_version=1,
            policy_definition_sha256="f" * 64,
        )


class _FakeRunner:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(self, request: StructuredGenerationRequest[Any]):
        self.requests.append(request)
        return self.handler(request)


def _response(value: Any) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model="deepseek/deepseek-v4-pro",
        resolved_model="deepseek/deepseek-v4-pro",
        resolved_provider="streamlake",
        provider_response_id="response-portfolio",
        finish_reason="stop",
        input_tokens=1_000,
        output_tokens=400,
        reasoning_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.01"),
        latency_ns=2_000_000_000,
    )


def test_pydantic_adapter_runs_one_structured_call_and_returns_ranked_decision() -> None:
    def handle(request: StructuredGenerationRequest[Any]):
        value = request.output_type.model_validate(
            {
                "members": [
                    {
                        "option_id": "beta.y1",
                        "supporting_card_keys": ["card.b"],
                        "effect_predictions": [
                            {"metric_id": "quality", "direction": "increase"},
                            {"metric_id": "cost", "direction": "decrease"},
                        ],
                        "design_rationale": " Best expected balance. ",
                    },
                    {
                        "option_id": "alpha.x2",
                        "supporting_card_keys": ["card.a"],
                        "effect_predictions": [
                            {"metric_id": "cost", "direction": "decrease"},
                            {"metric_id": "quality", "direction": "unchanged"},
                        ],
                        "design_rationale": "Strong single-family alternative.",
                    },
                    {
                        "option_id": "gamma.xy",
                        "supporting_card_keys": ["card.b", "card.a"],
                        "effect_predictions": [
                            {"metric_id": "cost", "direction": "unknown"},
                            {"metric_id": "quality", "direction": "increase"},
                        ],
                        "design_rationale": "Diverse coordinated fallback.",
                    },
                ]
            },
            strict=True,
        )
        return AttemptedStructuredGenerationResponse(
            response=_response(value),
            attempt_count=3,
        )

    runner = _FakeRunner(handle)
    request = _request()
    result = asyncio.run(
        PydanticAIPortfolioSelectionPolicy(runner).select(request)
    )

    assert len(runner.requests) == 1
    low_level = runner.requests[0]
    assert low_level.call_id == request.call_id
    assert low_level.operation == "select_portfolio"
    assert low_level.output_tool_name == PORTFOLIO_SELECTION_TOOL_NAME
    assert low_level.max_output_tokens == 999_999
    assert low_level.temperature == 0.1
    assert request.request_sha256 in low_level.prompt
    assert "alpha.x1" in low_level.prompt
    assert "card.a" in low_level.prompt
    assert '"score_id":"origin.rank"' in low_level.prompt
    schema = low_level.output_type.model_json_schema()
    option_schema = schema["$defs"]["PortfolioMemberSelection"]["properties"][
        "option_id"
    ]
    assert set(option_schema["enum"]) == {
        "alpha.x1",
        "alpha.x2",
        "beta.y1",
        "gamma.xy",
    }

    assert [member.option_id for member in result.decision.members] == [
        "beta.y1",
        "alpha.x2",
        "gamma.xy",
    ]
    assert result.decision.members[2].supporting_card_keys == (
        "card.a",
        "card.b",
    )
    assert tuple(
        prediction.metric_id
        for prediction in result.decision.members[0].effect_predictions
    ) == ("cost", "quality")
    assert result.telemetry is not None
    assert result.telemetry.attempt_count == 3
    assert result.telemetry.output_tokens == 400


def test_pydantic_schema_rejects_incomplete_or_nondiverse_output_before_return() -> None:
    def handle(request: StructuredGenerationRequest[Any]):
        with pytest.raises(ValidationError):
            request.output_type.model_validate(
                {
                    "members": [
                        {
                            "option_id": "alpha.x1",
                            "supporting_card_keys": ["card.a"],
                            "effect_predictions": [
                                {"metric_id": "cost", "direction": "decrease"}
                            ],
                            "design_rationale": "Missing one metric.",
                        },
                        {
                            "option_id": "alpha.x2",
                            "supporting_card_keys": ["card.b"],
                            "effect_predictions": [
                                {"metric_id": "cost", "direction": "decrease"},
                                {"metric_id": "quality", "direction": "increase"},
                            ],
                            "design_rationale": "Second alpha member.",
                        },
                        {
                            "option_id": "beta.y1",
                            "supporting_card_keys": ["card.a"],
                            "effect_predictions": [
                                {"metric_id": "cost", "direction": "decrease"},
                                {"metric_id": "quality", "direction": "increase"},
                            ],
                            "design_rationale": "Third member.",
                        },
                    ]
                },
                strict=True,
            )
        raise RuntimeError("no portfolio response was admitted")

    policy = PydanticAIPortfolioSelectionPolicy(_FakeRunner(handle))
    with pytest.raises(RuntimeError, match="no portfolio response"):
        asyncio.run(policy.select(_request()))
