"""Integrity-bound projections from insight memory into selector cards.

The application owns the trusted join between an immutable memory entry and
the framework-neutral portfolio port.  Benchmark adapters choose only the
prompt/evidence projection and its source receipt; they cannot substitute a
different finite action without invalidating the resulting source binding.
"""

from __future__ import annotations

import math
import re

from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightMemoryEntry,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validated_finite_variation_identity_index,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
)
from agent_evolve.ports.agentic_generator import InsightDraft
from agent_evolve.ports.portfolio_selection import (
    CardScoreComponent,
    PortfolioCard,
    PortfolioExperimentalArm,
    PortfolioExperimentalViewReceipt,
    PortfolioCardSourceBinding,
    PortfolioCardSourceRegistry,
    _bind_portfolio_card_source,
    _issue_portfolio_card_source_registry,
    portfolio_card_snapshot_sha256,
    validate_portfolio_experimental_view,
)


def project_action_neutral_insight_prompt_payload(
    entry: InsightMemoryEntry,
    *,
    prompt_payload: FrozenJsonObject,
    finite_variation_contracts: tuple[FiniteVariationContract, ...],
) -> FrozenJsonObject:
    """Remove exact finite-action identity from free-form insight prose.

    Reflection text is provider-authored and may repeat option IDs or evidence
    hashes that were visible in its evidence catalog.  Scientific M/N views
    carry those identities exclusively through structured
    ``finite_action_evidence``.  This projection therefore replaces every
    exact current-catalog or source-evidence identity in arbitrary typed JSON
    strings (including object keys) with deterministic punctuation-only
    aliases.  Families and other action-neutral mechanism prose are retained.

    Multiple contracts are accepted because a single admitted card population
    can be shared by several stable parent lanes with different parent-bound
    action catalogs.  The output is independent of contract input order.
    """

    lineage = _validate_source_entry(entry)
    if type(prompt_payload) is not FrozenJsonObject:
        raise TypeError("prompt_payload must be an exact FrozenJsonObject")
    if freeze_json(prompt_payload) is not prompt_payload:
        raise TypeError("prompt_payload must already be frozen typed JSON")
    if type(finite_variation_contracts) is not tuple or not (
        finite_variation_contracts
    ):
        raise ValueError(
            "finite_variation_contracts must be a non-empty exact tuple"
        )
    if any(
        type(contract) is not FiniteVariationContract
        for contract in finite_variation_contracts
    ):
        raise TypeError(
            "finite_variation_contracts must contain exact "
            "FiniteVariationContract values"
        )

    indexes_by_identity = {}
    for contract in finite_variation_contracts:
        index = validated_finite_variation_identity_index(contract)
        indexes_by_identity[index.contract_identity_sha256] = index

    forbidden: set[str] = set()
    for contract_identity in sorted(indexes_by_identity):
        index = indexes_by_identity[contract_identity]
        forbidden.update(index.option_ids)
        forbidden.add(index.contract_identity_sha256)
        forbidden.update(index.option_identity_sha256s)
        forbidden.update(index.child_configuration_sha256s)
    for binding in lineage.finite_action_bindings:
        forbidden.update(
            (
                binding.option_id,
                binding.option_identity_sha256,
                binding.contract_identity_sha256,
                binding.contrast_id,
                binding.identity_sha256,
            )
        )

    canonical_tokens = tuple(sorted({value.casefold() for value in forbidden}))
    aliases = {
        token: f"[#{ordinal:04d}]"
        for ordinal, token in enumerate(canonical_tokens, start=1)
    }
    matcher = re.compile(
        "|".join(
            re.escape(token)
            for token in sorted(canonical_tokens, key=lambda value: (-len(value), value))
        ),
        flags=re.IGNORECASE,
    )

    def redact_text(value: str) -> str:
        return matcher.sub(lambda match: aliases[match.group(0).casefold()], value)

    def visit(value: FrozenJsonValue) -> object:
        if type(value) is str:
            return redact_text(value)
        if type(value) is FrozenJsonArray:
            return [visit(item) for item in value.items]
        if type(value) is FrozenJsonObject:
            projected: dict[str, object] = {}
            for key, item in value.items:
                projected_key = redact_text(key)
                if projected_key in projected:
                    raise ValueError(
                        "action-neutral projection caused an object-key collision"
                    )
                projected[projected_key] = visit(item)
            return projected
        return value

    projected = freeze_json(visit(prompt_payload))
    if type(projected) is not FrozenJsonObject:  # pragma: no cover - root sealed.
        raise AssertionError("action-neutral prompt projection lost its object root")
    return projected


def _validate_source_entry(entry: InsightMemoryEntry) -> InsightEvidenceLineage:
    if type(entry) is not InsightMemoryEntry:
        raise TypeError("entry must be an exact InsightMemoryEntry")
    InsightMemoryEntry.__post_init__(entry)
    if type(entry.reference) is not InsightRef:
        raise TypeError("entry.reference must be an exact InsightRef")
    InsightRef.__post_init__(entry.reference)
    if type(entry.draft) is not InsightDraft:
        raise TypeError("entry.draft must be an exact InsightDraft")
    InsightDraft.__post_init__(entry.draft)
    if type(entry.initial_score) is not float or not math.isfinite(
        entry.initial_score
    ):
        raise TypeError("entry.initial_score must be a finite canonical float")
    lineage = entry.evidence_lineage
    if type(lineage) is not InsightEvidenceLineage:
        raise ValueError(
            "source-bound portfolio projection requires reflection evidence lineage"
        )
    InsightEvidenceLineage.__post_init__(lineage)
    return lineage


def portfolio_card_from_insight_entry(
    entry: InsightMemoryEntry,
    *,
    card_key: str,
    prompt_payload: FrozenJsonObject,
    evidence_sha256: str,
    source_receipt_sha256: str,
    score_components: tuple[CardScoreComponent, ...] = (),
    assigned_score: float | None = None,
) -> PortfolioCard:
    """Project one reflection entry with integrity-bound action lineage.

    ``source_receipt_sha256`` identifies the durable evidence/projection receipt
    trusted by the caller.  The returned receipt hash-binds the entry reference,
    complete draft content, evidence-lineage identity, exact finite actions,
    prompt view, evidence identity, and score state.
    """

    lineage = _validate_source_entry(entry)
    if type(prompt_payload) is not FrozenJsonObject:
        raise TypeError("prompt_payload must be an exact FrozenJsonObject")
    if freeze_json(prompt_payload) is not prompt_payload:
        raise TypeError("prompt_payload must already be frozen typed JSON")
    source_binding = _bind_portfolio_card_source(
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_lineage_identity_sha256=lineage.identity_sha256,
        finite_action_evidence=lineage.portfolio_action_evidence,
        prompt_payload=prompt_payload,
        evidence_sha256=evidence_sha256,
        score_components=score_components,
        assigned_score=assigned_score,
        source_receipt_sha256=source_receipt_sha256,
    )
    return PortfolioCard(
        card_key=card_key,
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_sha256=evidence_sha256,
        prompt_payload=prompt_payload,
        score_components=score_components,
        assigned_score=assigned_score,
        finite_action_evidence=lineage.portfolio_action_evidence,
        source_binding=source_binding,
    )


def admit_portfolio_card_sources(
    entries: tuple[InsightMemoryEntry, ...],
    cards: tuple[PortfolioCard, ...],
) -> PortfolioCardSourceRegistry:
    """Admit an exact source-entry set for one scientific selector request.

    The caller designates ``entries`` as trusted memory state.  This function
    performs the authoritative entry-to-binding join and issues a closed
    integrity registry; it does not provide cryptographic authenticity.  Every
    supplied card must be source-bound, and the entry/card reference sets must
    be exactly equal so neither stale nor adapter-synthesized bindings slip in.
    """

    if type(entries) is not tuple or not entries:
        raise ValueError("entries must be a non-empty exact tuple")
    if any(type(entry) is not InsightMemoryEntry for entry in entries):
        raise TypeError("entries must contain exact InsightMemoryEntry values")
    if type(cards) is not tuple or not cards:
        raise ValueError("cards must be a non-empty exact tuple")
    if any(type(card) is not PortfolioCard for card in cards):
        raise TypeError("cards must contain exact PortfolioCard values")

    entries_by_reference: dict[InsightRef, InsightMemoryEntry] = {}
    for entry in entries:
        _validate_source_entry(entry)
        if entry.reference in entries_by_reference:
            raise ValueError("entries cannot repeat an insight reference")
        entries_by_reference[entry.reference] = entry

    bindings: list[PortfolioCardSourceBinding] = []
    card_references: set[InsightRef] = set()
    for card in cards:
        card.__post_init__()
        binding = card.source_binding
        if binding is None:
            raise ValueError("source admission rejects unbound legacy cards")
        if card.reference in card_references:
            raise ValueError("cards cannot repeat an insight reference")
        card_references.add(card.reference)
        entry = entries_by_reference.get(card.reference)
        if entry is None:
            raise ValueError("card source is absent from the trusted entry set")
        lineage = entry.evidence_lineage
        assert type(lineage) is InsightEvidenceLineage
        if (
            binding.reference != entry.reference
            or binding.content_sha256 != entry.draft.content_sha256
            or binding.evidence_lineage_identity_sha256
            != lineage.identity_sha256
            or binding.finite_action_evidence
            != lineage.portfolio_action_evidence
        ):
            raise ValueError("card source binding differs from its trusted entry")
        bindings.append(binding)
    if card_references != set(entries_by_reference):
        raise ValueError("trusted entry set differs from the card source set")

    canonical_bindings = tuple(
        sorted(bindings, key=lambda binding: binding.binding_sha256)
    )
    return _issue_portfolio_card_source_registry(canonical_bindings)


def bind_portfolio_experimental_view(
    *,
    arm: PortfolioExperimentalArm,
    cards: tuple[PortfolioCard, ...],
    finite_variation_contract: FiniteVariationContract,
    source_registry: PortfolioCardSourceRegistry,
    policy_id: str,
    policy_version: int,
    policy_definition_sha256: str,
) -> PortfolioExperimentalViewReceipt:
    """Bind and validate one complete scientific M/P/N card population.

    The finite contract is typed at the port trust boundary so this application
    helper stays a thin issuer rather than duplicating domain validation.
    """

    source_donor_pairs: tuple[tuple[str, str], ...] = ()
    if arm is PortfolioExperimentalArm.PERMUTED_PLACEBO:
        inferred: list[tuple[str, str]] = []
        for card in cards:
            if type(card) is not PortfolioCard or card.source_binding is None:
                raise ValueError("P requires source-bound portfolio cards")
            view = card.derived_view_receipt
            if view is None or view.prompt_source_binding_sha256 is None:
                raise ValueError("P requires donor-bound derived card views")
            inferred.append(
                (
                    card.source_binding.binding_sha256,
                    view.prompt_source_binding_sha256,
                )
            )
        source_donor_pairs = tuple(sorted(inferred))
    receipt = PortfolioExperimentalViewReceipt(
        arm=arm,
        source_registry_sha256=source_registry.registry_sha256,
        card_snapshot_sha256=portfolio_card_snapshot_sha256(cards),
        source_donor_binding_pairs=source_donor_pairs,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
    )
    validate_portfolio_experimental_view(
        cards=cards,
        finite_variation_contract=finite_variation_contract,
        source_registry=source_registry,
        receipt=receipt,
    )
    return receipt


__all__ = [
    "admit_portfolio_card_sources",
    "bind_portfolio_experimental_view",
    "portfolio_card_from_insight_entry",
    "project_action_neutral_insight_prompt_payload",
]
