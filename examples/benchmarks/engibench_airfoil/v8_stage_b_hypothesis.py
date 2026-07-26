"""Airfoil Stage-B compiler for reflection-native executable hypotheses.

The immutable G3 compiler maps the historical source vocabulary ``mutation``
to AgentEvolve's executable ``typed_mutation`` operator.  A card revised by the
real G3 reflection loop instead records the operator that actually produced its
evidence: ``typed_mutation``.  This new benchmark policy accepts both exact
source vocabularies while preserving every Airfoil action, metric, path, and
held-fixed constraint.  The generic hypothesis port and core contracts remain
unchanged.
"""

from __future__ import annotations

import hashlib
import json

from agent_evolve.agentic import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    OperatorKind,
    TreatmentActionBinding,
    validate_hypothesis_compilation,
    validate_hypothesis_compiler_identity,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    REQUIRED_METRIC_IDS,
    SHAPE_HELD_FIXED_PATHS,
    TRIM_PATHS,
)


_DEFINITION_DOMAIN = b"agent-evolve:airfoil-v8-stage-b-compiler:def:v1\x00"
_DEFINITION = {
    "policy_id": "airfoil_v8_reflection_native_trim_compiler",
    "policy_version": 1,
    "accepted_source_operator_scopes": [
        ["mutation"],
        [OperatorKind.TYPED_MUTATION.value],
    ],
    "executable_operator": OperatorKind.TYPED_MUTATION.value,
    "required_family": "trim_only",
    "required_affected_paths": list(TRIM_PATHS),
    "required_metric_ids": list(REQUIRED_METRIC_IDS),
    "allowed_action_cardinality": 1,
    "action_binding": "exact_recommended_template_in_current_parent_contract",
    "held_fixed_paths": list(SHAPE_HELD_FIXED_PATHS),
    "outcome_access": False,
    "semantic_change_from_g3_v1": (
        "admit the reflection-native typed_mutation source scope without "
        "rewriting the persisted learned card"
    ),
}
AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN
    + json.dumps(
        _DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


class AirfoilV8ReflectionNativeTrimHypothesisCompiler:
    """Bind historical or reflection-native trim cards to one current parent."""

    policy_id = "airfoil_v8_reflection_native_trim_compiler"
    policy_version = 1
    definition_sha256 = (
        AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256
    )

    def _inapplicable(
        self,
        request: HypothesisCompilationRequest,
        *reason_codes: str,
    ) -> HypothesisCompilationReceipt:
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.INAPPLICABLE,
            reason_codes=tuple(sorted(set(reason_codes))),
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
        request.__post_init__()
        reasons: list[str] = []
        if request.requested_operator_kind != OperatorKind.TYPED_MUTATION.value:
            reasons.append("foreign_executable_operator")
        if request.source_operator_kinds not in {
            ("mutation",),
            (OperatorKind.TYPED_MUTATION.value,),
        }:
            reasons.append("foreign_source_operator_scope")
        insight = request.insight
        if insight.recommended_option_families != ("trim_only",):
            reasons.append("foreign_option_family")
        if tuple(sorted(insight.affected_paths)) != TRIM_PATHS:
            reasons.append("foreign_affected_paths")
        if tuple(value.metric_id for value in insight.effect_predictions) != (
            REQUIRED_METRIC_IDS
        ):
            reasons.append("foreign_metric_scope")
        if len(insight.recommended_option_ids) != 1:
            reasons.append("non_singleton_template")
        option = None
        if len(insight.recommended_option_ids) == 1:
            try:
                option = request.finite_contract.resolve(
                    insight.recommended_option_ids[0]
                )
            except ValueError:
                reasons.append("template_absent_from_parent")
            else:
                if option.family != "trim_only":
                    reasons.append("template_resolves_to_foreign_family")
        if reasons:
            return self._inapplicable(request, *reasons)
        assert option is not None
        spec = ExecutableHypothesisTestSpec(
            request_sha256=request.request_sha256,
            reference=request.reference,
            insight_content_sha256=insight.content_sha256,
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
            recommended_option_families=insight.recommended_option_families,
            affected_paths=TRIM_PATHS,
            held_fixed_paths=SHAPE_HELD_FIXED_PATHS,
            effect_predictions=insight.effect_predictions,
            falsification_condition=str(insight.falsification_condition),
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


__all__ = [
    "AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256",
    "AirfoilV8ReflectionNativeTrimHypothesisCompiler",
]
