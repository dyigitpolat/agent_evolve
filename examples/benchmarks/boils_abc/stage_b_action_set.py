"""Exact BOiLS anchor compilation and outcome-blind local K-option support."""

from __future__ import annotations

import hashlib
import json

from agent_evolve.agentic import (
    ExecutableHypothesisTestSpec,
    FiniteActionSetCompilationRequest,
    FiniteActionSetDraft,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    OperatorKind,
    TreatmentActionBinding,
    validate_hypothesis_compilation,
    validate_hypothesis_compiler_identity,
)

from .actions import SEQUENCE_LENGTH
from .finite_variation_catalog import FINITE_CATALOG_ID
from .variation_catalog import ACTION_FAMILIES


_EXACT_DOMAIN = b"agent-evolve:boils-position-anchor-compiler:v1\x00"
_SUPPORT_DOMAIN = b"agent-evolve:boils-position-local-support:v1\x00"
_RANK_DOMAIN = b"agent-evolve:boils-position-local-rank:v1\x00"
_PRESENT_DOMAIN = b"agent-evolve:boils-position-local-presentation:v1\x00"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


BOILS_POSITION_ANCHOR_COMPILER_DEFINITION_SHA256 = hashlib.sha256(
    _EXACT_DOMAIN
    + _canonical(
        {
            "anchor": "one exact current-parent option",
            "affected_path": "anchor sequence position",
            "held_fixed": "all other sequence positions",
            "current_outcome_access": False,
        }
    )
).hexdigest()
BOILS_STAGE_B_ACTION_SET_DEFINITION_SHA256 = hashlib.sha256(
    _SUPPORT_DOMAIN
    + _canonical(
        {
            "support": "same sequence position",
            "anchor": "always included",
            "remaining": "task-keyed SHA-256 ranks",
            "cardinality": "request 4 through 8",
            "current_outcome_access": False,
        }
    )
).hexdigest()
BOILS_STAGE_B_PROMPT_SHAPE_SHA256 = hashlib.sha256(
    _PRESENT_DOMAIN
    + _canonical(
        {
            "card_record_count": 1,
            "option_record_count": "request cardinality",
            "option_fields": ["description", "family", "metadata", "option_id"],
            "selected_field": "option_id",
        }
    )
).hexdigest()


def _position(option_id: str) -> int:
    parts = option_id.split(".")
    if len(parts) != 3 or parts[0] != "boils_abc" or not parts[1].startswith("p"):
        raise ValueError("BOiLS option ID does not encode one sequence position")
    try:
        position = int(parts[1][1:])
    except ValueError as exc:
        raise ValueError("BOiLS option position is not an integer") from exc
    if not 0 <= position < SEQUENCE_LENGTH:
        raise ValueError("BOiLS option position is outside the sequence")
    return position


class BoilsPositionHypothesisCompiler:
    """Bind one position-scoped hypothesis to its exact recommended action."""

    policy_id = "boils_position_exact_anchor"
    policy_version = 1
    definition_sha256 = BOILS_POSITION_ANCHOR_COMPILER_DEFINITION_SHA256

    def _inapplicable(
        self,
        request: HypothesisCompilationRequest,
        *reasons: str,
    ) -> HypothesisCompilationReceipt:
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.INAPPLICABLE,
            reason_codes=tuple(sorted(set(reasons))),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=None,
        )

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        if type(request) is not HypothesisCompilationRequest:
            raise TypeError("request must be an exact HypothesisCompilationRequest")
        HypothesisCompilationRequest.__post_init__(request)
        reasons: list[str] = []
        if request.finite_contract.catalog_id != FINITE_CATALOG_ID:
            reasons.append("foreign_catalog")
        if request.requested_operator_kind != OperatorKind.TYPED_MUTATION.value:
            reasons.append("foreign_operator")
        if len(request.insight.recommended_option_ids) != 1:
            reasons.append("non_singleton_anchor")
        option = None
        position = None
        if len(request.insight.recommended_option_ids) == 1:
            option_id = request.insight.recommended_option_ids[0]
            try:
                option = request.finite_contract.resolve(option_id)
                position = _position(option_id)
            except ValueError:
                reasons.append("anchor_absent_or_malformed")
        expected_families = tuple(sorted(set(ACTION_FAMILIES.values())))
        if request.insight.recommended_option_families != expected_families:
            reasons.append("incomplete_local_family_scope")
        if position is not None and request.insight.affected_paths != (
            f"$.sequence[{position}]",
        ):
            reasons.append("foreign_affected_path")
        if reasons:
            return self._inapplicable(request, *reasons)
        assert option is not None and position is not None
        held_fixed = tuple(
            sorted(
                f"$.sequence[{index}]"
                for index in range(SEQUENCE_LENGTH)
                if index != position
            )
        )
        spec = ExecutableHypothesisTestSpec(
            request_sha256=request.request_sha256,
            reference=request.reference,
            insight_content_sha256=request.insight.content_sha256,
            source_evidence_sha256=request.source_evidence_sha256,
            requested_operator_kind=request.requested_operator_kind,
            source_operator_kinds=request.source_operator_kinds,
            executable_operator_kinds=(request.requested_operator_kind,),
            parent_candidate_id=request.parent_candidate_id,
            parent_configuration_sha256=request.parent_configuration_sha256,
            finite_contract_sha256=request.finite_contract.identity_sha256,
            context_projection_sha256=request.context_projection_sha256,
            endpoint_definition_sha256=request.endpoint_definition_sha256,
            allowed_actions=(
                TreatmentActionBinding(option.option_id, option.identity_sha256),
            ),
            recommended_option_families=expected_families,
            affected_paths=(f"$.sequence[{position}]",),
            held_fixed_paths=held_fixed,
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=str(request.insight.falsification_condition),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )
        receipt = HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.APPLICABLE,
            reason_codes=(),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=spec,
        )
        validate_hypothesis_compiler_identity(self, receipt)
        validate_hypothesis_compilation(request, receipt)
        return receipt


class BoilsPositionLocalSupportCompiler:
    """Choose K same-position actions with a prospective task-keyed rank."""

    policy_id = "boils_position_local_support"
    policy_version = 1
    definition_sha256 = BOILS_STAGE_B_ACTION_SET_DEFINITION_SHA256

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft:
        if type(request) is not FiniteActionSetCompilationRequest:
            raise TypeError("request must be an exact FiniteActionSetCompilationRequest")
        FiniteActionSetCompilationRequest.__post_init__(request)
        if request.current_outcome_access:
            raise ValueError("BOiLS support compiler cannot access outcomes")
        if request.finite_contract.catalog_id != FINITE_CATALOG_ID:
            raise ValueError("BOiLS local support requires its single-action catalog")
        position = _position(request.anchor_option_id)
        eligible = tuple(
            option
            for option in request.finite_contract.options
            if _position(option.option_id) == position
        )
        anchor = request.finite_contract.resolve(request.anchor_option_id)
        alternatives = tuple(
            sorted(
                (option for option in eligible if option.option_id != anchor.option_id),
                key=lambda option: hashlib.sha256(
                    _RANK_DOMAIN
                    + bytes.fromhex(request.request_sha256)
                    + bytes.fromhex(option.identity_sha256)
                ).digest(),
            )
        )
        selected = (anchor,) + alternatives[: request.required_cardinality - 1]
        ordered = tuple(
            option.option_id
            for option in sorted(
                selected,
                key=lambda option: hashlib.sha256(
                    _PRESENT_DOMAIN
                    + bytes.fromhex(request.request_sha256)
                    + bytes.fromhex(option.identity_sha256)
                ).digest(),
            )
        )
        if len(ordered) != request.required_cardinality:
            raise ValueError("BOiLS source contract lacks enough same-position actions")
        return FiniteActionSetDraft(
            request_sha256=request.request_sha256,
            ordered_option_ids=ordered,
            anchor_option_id=request.anchor_option_id,
            presentation_policy_id="boils_stage_b_local_presentation",
            presentation_policy_version=1,
            presentation_definition_sha256=self.definition_sha256,
            prompt_shape_sha256=BOILS_STAGE_B_PROMPT_SHAPE_SHA256,
        )


__all__ = [
    "BOILS_POSITION_ANCHOR_COMPILER_DEFINITION_SHA256",
    "BOILS_STAGE_B_ACTION_SET_DEFINITION_SHA256",
    "BOILS_STAGE_B_PROMPT_SHAPE_SHA256",
    "BoilsPositionHypothesisCompiler",
    "BoilsPositionLocalSupportCompiler",
]
