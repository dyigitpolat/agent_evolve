"""Fail-closed tests for parent-bound executable hypothesis compilation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId, InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentActionBinding,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    validate_hypothesis_compiler_identity,
    validate_hypothesis_compilation,
)


_COMPILER_ID = "fixture_hypothesis_compiler"
_COMPILER_VERSION = 1
_COMPILER_SHA = "c" * 64


def _contract() -> FiniteVariationContract:
    parent = freeze_json({"model": {"a": 5}, "engine": {"b": 5}})
    parent_sha = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="fixture_parent_bound",
        catalog_version=1,
        catalog_definition_sha256="a" * 64,
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="model.conservative",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json(
                    {"model": {"a": 4}, "engine": {"b": 5}}
                ),
                family="model",
                description="Apply the conservative model intervention.",
            ),
            FiniteVariationOption(
                option_id="model.aggressive",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json(
                    {"model": {"a": 2}, "engine": {"b": 5}}
                ),
                family="model",
                description="Apply the aggressive model intervention.",
            ),
            FiniteVariationOption(
                option_id="engine.orthogonal",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json(
                    {"model": {"a": 5}, "engine": {"b": 3}}
                ),
                family="engine",
                description="Apply the orthogonal engine intervention.",
            ),
        ),
    )


def _insight(*, recommended_ids: tuple[str, ...] = ()) -> InsightDraft:
    return InsightDraft(
        claim="Reducing the model coordinate improves the primary metric.",
        trigger="The current parent retains excessive model capacity.",
        mechanism="A smaller model coordinate reduces the evaluated cost.",
        affected_paths=("$.model.a",),
        evidence_summary="The sealed diagnostic contrast supports the direction.",
        confidence=0.75,
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="primary",
                direction=MetricEffectDirection.DECREASE,
            ),
        ),
        recommended_option_families=("model",),
        recommended_option_ids=recommended_ids,
        action_template="Select a parent-bound model-family reduction.",
        falsification_condition="The primary metric does not decrease.",
    )


def _request(*, insight: InsightDraft | None = None) -> HypothesisCompilationRequest:
    contract = _contract()
    return HypothesisCompilationRequest(
        reference=InsightRef(InsightId("insight_fixture_h1"), 1),
        insight=_insight() if insight is None else insight,
        source_evidence_sha256="b" * 64,
        requested_operator_kind="typed_mutation",
        source_operator_kinds=("mutation",),
        parent_candidate_id=CandidateId("candidate_fixture_parent"),
        parent_configuration_sha256=contract.parent_configuration_sha256,
        finite_contract=contract,
        context_projection_sha256="d" * 64,
        endpoint_definition_sha256="e" * 64,
    )


def _receipt(
    request: HypothesisCompilationRequest,
    *,
    option_id: str = "model.aggressive",
) -> HypothesisCompilationReceipt:
    option = request.finite_contract.resolve(option_id)
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
        recommended_option_families=request.insight.recommended_option_families,
        affected_paths=tuple(sorted(set(request.insight.affected_paths))),
        held_fixed_paths=("$.engine",),
        effect_predictions=request.insight.effect_predictions,
        falsification_condition=request.insight.falsification_condition or "missing",
        compiler_policy_id=_COMPILER_ID,
        compiler_policy_version=_COMPILER_VERSION,
        compiler_definition_sha256=_COMPILER_SHA,
    )
    return HypothesisCompilationReceipt(
        request_sha256=request.request_sha256,
        status=HypothesisApplicabilityStatus.APPLICABLE,
        reason_codes=(),
        compiler_policy_id=_COMPILER_ID,
        compiler_policy_version=_COMPILER_VERSION,
        compiler_definition_sha256=_COMPILER_SHA,
        spec=spec,
    )


class _Compiler:
    policy_id = _COMPILER_ID
    policy_version = _COMPILER_VERSION
    definition_sha256 = _COMPILER_SHA

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        return _receipt(request)


def _rewritten_receipt(
    receipt: HypothesisCompilationReceipt,
    **changes: object,
) -> HypothesisCompilationReceipt:
    assert receipt.spec is not None
    return replace(receipt, spec=replace(receipt.spec, **changes))


def test_valid_parent_bound_compilation_and_port_identity_pass() -> None:
    request = _request()
    compiler = _Compiler()
    receipt = compiler.compile(request)

    validate_hypothesis_compiler_identity(compiler, receipt)
    validate_hypothesis_compilation(request, receipt)


def test_request_rejects_non_intervention_and_empty_operator_scope() -> None:
    plain = InsightDraft(
        claim="A descriptive observation.",
        trigger="Always.",
        mechanism="Description only.",
        affected_paths=("$.model.a",),
        evidence_summary="No intervention contract.",
        confidence=0.5,
    )
    with pytest.raises(ValueError, match="intervention-contract"):
        _request(insight=plain)

    request = _request()
    with pytest.raises(ValueError, match="non-empty"):
        replace(request, source_operator_kinds=())
    # Legacy source aliases are immutable evidence, not execution authority.
    replace(request, requested_operator_kind="two_parent_crossover")


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"recommended_option_families": ("engine",)}, "families"),
        (
            {"affected_paths": ("$.engine.b",), "held_fixed_paths": ()},
            "affected paths",
        ),
        (
            {
                "effect_predictions": (
                    MetricEffectPrediction(
                        metric_id="primary",
                        direction=MetricEffectDirection.INCREASE,
                    ),
                )
            },
            "metric predictions",
        ),
        ({"falsification_condition": "The metric decreases."}, "falsification"),
        ({"source_evidence_sha256": "f" * 64}, "trusted request identity"),
        (
            {
                "requested_operator_kind": "two_parent_crossover",
                "executable_operator_kinds": ("two_parent_crossover",),
            },
            "trusted request identity",
        ),
        (
            {
                "source_operator_kinds": ("mutation", "reproduction")
            },
            "trusted request identity",
        ),
    ),
)
def test_compiler_cannot_rewrite_immutable_hypothesis_semantics(
    changes: dict[str, object],
    message: str,
) -> None:
    request = _request()
    receipt = _rewritten_receipt(_receipt(request), **changes)

    with pytest.raises(ValueError, match=message):
        validate_hypothesis_compilation(request, receipt)


def test_compiler_cannot_map_hypothesis_to_foreign_family() -> None:
    request = _request()
    receipt = _receipt(request, option_id="engine.orthogonal")

    with pytest.raises(ValueError, match="foreign action family"):
        validate_hypothesis_compilation(request, receipt)


def test_compiler_cannot_broaden_executable_operator_authority() -> None:
    receipt = _receipt(_request())
    assert receipt.spec is not None

    with pytest.raises(ValueError, match="requested singleton"):
        replace(
            receipt.spec,
            executable_operator_kinds=("repair", "typed_mutation"),
        )


def test_exact_parent_id_is_enforced_only_when_present_in_current_catalog() -> None:
    current = _request(insight=_insight(recommended_ids=("model.conservative",)))
    with pytest.raises(ValueError, match="exact recommendations"):
        validate_hypothesis_compilation(current, _receipt(current))

    historical = _request(insight=_insight(recommended_ids=("model.old_parent",)))
    validate_hypothesis_compilation(historical, _receipt(historical))


def test_held_fixed_path_rejects_ancestor_and_descendant_overlap() -> None:
    request = _request()
    receipt = _receipt(request)
    assert receipt.spec is not None

    with pytest.raises(ValueError, match="hierarchically disjoint"):
        replace(receipt.spec, held_fixed_paths=("$.model",))
    with pytest.raises(ValueError, match="hierarchically disjoint"):
        replace(
            receipt.spec,
            affected_paths=("$.model",),
            held_fixed_paths=("$.model.a",),
        )


@pytest.mark.parametrize("malformed", ("$", "$.[", "$..model", "$.model[-1]"))
def test_compiler_cannot_bypass_overlap_with_malformed_paths(malformed: str) -> None:
    receipt = _receipt(_request())
    assert receipt.spec is not None

    with pytest.raises((TypeError, ValueError), match="canonical rooted JSON paths"):
        replace(receipt.spec, held_fixed_paths=(malformed,))


def test_orchestration_rejects_receipt_from_different_compiler_identity() -> None:
    class _SwappedCompiler(_Compiler):
        definition_sha256 = "f" * 64

    receipt = _receipt(_request())

    with pytest.raises(ValueError, match="compiler identity changed"):
        validate_hypothesis_compiler_identity(_SwappedCompiler(), receipt)
