"""Trusted sealing of model choices from matched finite action sets.

The provider-facing generator already returns an opaque finite-option ID.  This
service joins that untrusted draft to the exact Stage-B authority, logical call,
semantic prompt, and provider telemetry before an evaluator may use the choice.
It deliberately has no evaluator or outcome dependency.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    resolve_finite_variation_selection,
)
from agent_evolve.ports.finite_action_selection import (
    FiniteActionDecision,
    FiniteActionSelectorKind,
    validate_finite_action_decision,
)


MODEL_FINITE_ACTION_SELECTOR_POLICY_ID = "literal_id_model_finite_action"
MODEL_FINITE_ACTION_SELECTOR_POLICY_VERSION = 1
_MODEL_SELECTOR_DEFINITION_DOMAIN = (
    b"agent-evolve:literal-id-model-finite-action:def:v1\x00"
)
_MODEL_TELEMETRY_DOMAIN = (
    b"agent-evolve:model-finite-action-call-telemetry:v1\x00"
)
_MODEL_SELECTOR_DEFINITION = {
    "choice_authority": "one presealed finite action set",
    "model_output": "one opaque option ID plus non-authoritative rationale",
    "materialization": "engine-owned full child",
    "propensity": "unknown",
    "current_outcome_access": False,
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


MODEL_FINITE_ACTION_SELECTOR_DEFINITION_SHA256 = hashlib.sha256(
    _MODEL_SELECTOR_DEFINITION_DOMAIN
    + _canonical_json(_MODEL_SELECTOR_DEFINITION)
).hexdigest()


def _telemetry_record(telemetry: AgenticCallTelemetry) -> dict[str, object]:
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("telemetry must be an exact AgenticCallTelemetry")
    AgenticCallTelemetry.__post_init__(telemetry)
    for name in ("provider_response_id", "finish_reason"):
        value = getattr(telemetry, name)
        if value is not None and type(value) is not str:
            raise TypeError(f"telemetry {name} must be an exact string or None")
    if telemetry.cost_usd is None:
        cost_usd: str | None = None
    else:
        if type(telemetry.cost_usd) is not Decimal:
            raise TypeError("telemetry cost_usd must be an exact Decimal or None")
        if not telemetry.cost_usd.is_finite() or telemetry.cost_usd < 0:
            raise ValueError("telemetry cost_usd must be finite and non-negative")
        cost_usd = str(telemetry.cost_usd)
    return {
        "requested_model": telemetry.requested_model,
        "resolved_model": telemetry.resolved_model,
        "resolved_provider": telemetry.resolved_provider,
        "provider_response_id": telemetry.provider_response_id,
        "finish_reason": telemetry.finish_reason,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "cache_read_tokens": telemetry.cache_read_tokens,
        "cache_write_tokens": telemetry.cache_write_tokens,
        "cost_usd": cost_usd,
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


def model_finite_action_telemetry_sha256(
    telemetry: AgenticCallTelemetry,
) -> str:
    """Hash exact call telemetry without lossy Decimal conversion."""

    return hashlib.sha256(
        _MODEL_TELEMETRY_DOMAIN + _canonical_json(_telemetry_record(telemetry))
    ).hexdigest()


def seal_model_finite_action_decision(
    *,
    authority: FiniteActionSetAuthority,
    call_id: LLMCallId,
    prompt_sha256: str,
    draft: FiniteVariationSelectionDraft,
    telemetry: AgenticCallTelemetry,
) -> FiniteActionDecision:
    """Bind one model-selected ID to its exact K-option authority.

    The option ordinal is derived from the authority's provider-visible order;
    it is never accepted from the model.  All validation and sealing happen
    without current candidate outcomes.
    """

    if type(authority) is not FiniteActionSetAuthority:
        raise TypeError("authority must be an exact FiniteActionSetAuthority")
    FiniteActionSetAuthority.__post_init__(authority)
    if type(call_id) is not LLMCallId:
        raise TypeError("call_id must be an exact LLMCallId")
    LLMCallId.__post_init__(call_id)
    require_sha256(prompt_sha256, "prompt_sha256")
    if type(draft) is not FiniteVariationSelectionDraft:
        raise TypeError("draft must be an exact FiniteVariationSelectionDraft")
    FiniteVariationSelectionDraft.__post_init__(draft)
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("telemetry must be an exact AgenticCallTelemetry")
    AgenticCallTelemetry.__post_init__(telemetry)

    authority_sha256_before = authority.authority_sha256
    support_sha256_before = authority.support.support_sha256
    support_contract = authority.support.support_contract
    option = resolve_finite_variation_selection(support_contract, draft)
    matching_ordinals = tuple(
        ordinal
        for ordinal, row in enumerate(authority.support.options)
        if (
            row.option.option_id == option.option_id
            and row.option.identity_sha256 == option.identity_sha256
        )
    )
    if len(matching_ordinals) != 1:
        raise ValueError("selected option has no unique authority ordinal")
    selected_ordinal = matching_ordinals[0]
    row = authority.support.options[selected_ordinal]
    decision = FiniteActionDecision(
        authority_sha256=authority.authority_sha256,
        support_sha256=authority.support.support_sha256,
        selector_kind=FiniteActionSelectorKind.MODEL,
        selected_ordinal=selected_ordinal,
        option_id=row.option.option_id,
        option_identity_sha256=row.option.identity_sha256,
        child_configuration_sha256=row.option.child_configuration_sha256,
        phenotype_identity_sha256=row.phenotype_identity_sha256,
        selector_policy_id=MODEL_FINITE_ACTION_SELECTOR_POLICY_ID,
        selector_policy_version=MODEL_FINITE_ACTION_SELECTOR_POLICY_VERSION,
        selector_definition_sha256=(
            MODEL_FINITE_ACTION_SELECTOR_DEFINITION_SHA256
        ),
        prospective_token_sha256=None,
        model_call_id=call_id,
        model_prompt_sha256=prompt_sha256,
        model_telemetry_sha256=model_finite_action_telemetry_sha256(telemetry),
        propensity_numerator=None,
        propensity_denominator=None,
        current_outcome_access=False,
    )
    if (
        authority.authority_sha256 != authority_sha256_before
        or authority.support.support_sha256 != support_sha256_before
    ):
        raise ValueError("finite action authority changed during model sealing")
    validate_finite_action_decision(authority, decision)
    return decision


__all__ = [
    "MODEL_FINITE_ACTION_SELECTOR_DEFINITION_SHA256",
    "MODEL_FINITE_ACTION_SELECTOR_POLICY_ID",
    "MODEL_FINITE_ACTION_SELECTOR_POLICY_VERSION",
    "model_finite_action_telemetry_sha256",
    "seal_model_finite_action_decision",
]
