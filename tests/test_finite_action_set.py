"""Provider/evaluator-free tests for matched K-option action authorities."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.executable_hypothesis import (
    compile_registered_hypothesis_treatment,
    registered_source_evidence_sha256,
)
from agent_evolve.application.finite_action_set import (
    compile_and_seal_finite_action_set,
)
from agent_evolve.application.insight_memory import InsightMemoryBank, InsightOrigin
from agent_evolve.domain.finite_action_set import FiniteActionSourceMode
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.treatment_compliance import TreatmentActionBinding
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    TypedConfigurationPhenotypeIdentityPolicy,
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
)
from agent_evolve.ports.finite_action_set import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetDraft,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _ExactAnchorCompiler:
    policy_id = "fixture_exact_anchor"
    policy_version = 1
    definition_sha256 = _sha("fixture exact anchor v1")

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        option = request.finite_contract.resolve(request.insight.recommended_option_ids[0])
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
            affected_paths=tuple(sorted(request.insight.affected_paths)),
            held_fixed_paths=("$.held",),
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=str(request.insight.falsification_condition),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.APPLICABLE,
            reason_codes=(),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=spec,
        )


class _LocalSupportCompiler:
    policy_id = "fixture_local_support"
    policy_version = 1
    definition_sha256 = _sha("fixture local support v1")

    def __init__(self, option_ids: tuple[str, ...]) -> None:
        self.option_ids = option_ids

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft:
        return FiniteActionSetDraft(
            request_sha256=request.request_sha256,
            ordered_option_ids=self.option_ids,
            anchor_option_id=request.anchor_option_id,
            presentation_policy_id="fixture_support_presentation",
            presentation_policy_version=1,
            presentation_definition_sha256=_sha("fixture presentation v1"),
            prompt_shape_sha256=_sha("fixture K-option prompt shape v1"),
        )


def _scalar_contract() -> FiniteVariationContract:
    parent = freeze_json({"value": 0, "held": 7})
    parent_sha256 = typed_json_sha256(parent)
    values = (-2, -1, 1, 2)
    return FiniteVariationContract(
        catalog_id="fixture_scalar_local",
        catalog_version=1,
        catalog_definition_sha256=_sha("fixture scalar catalog v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=f"local.value_{'n' if value < 0 else 'p'}{abs(value)}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"value": value, "held": 7}),
                family="local_value",
                description=f"Set the local scalar value to {value}.",
            )
            for value in values
        ),
    )


def _sequence_contract() -> FiniteVariationContract:
    parent = freeze_json({"sequence": ["a", "a"], "held": 7})
    parent_sha256 = typed_json_sha256(parent)
    replacements = ("b", "c", "d", "e")
    return FiniteVariationContract(
        catalog_id="fixture_sequence_local",
        catalog_version=1,
        catalog_definition_sha256=_sha("fixture sequence catalog v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=f"local.sequence_0_{value}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"sequence": [value, "a"], "held": 7}
                ),
                family="local_sequence",
                description=f"Replace sequence position zero with {value}.",
            )
            for value in replacements
        ),
    )


def _compiled_anchor(
    contract: FiniteVariationContract,
    *,
    anchor_option_id: str,
    affected_path: str,
):
    ids = DeterministicIdFactory(f"finite_action_{contract.catalog_id}")
    memory = InsightMemoryBank(id_factory=ids)
    family = contract.resolve(anchor_option_id).family
    entry, added = memory.add(
        InsightDraft(
            claim="The exact anchor is a useful local intervention.",
            trigger="A local parent-bound choice is available.",
            mechanism="The selected local coordinate controls the endpoint.",
            affected_paths=(affected_path,),
            evidence_summary="Fixture source evidence binds one exact anchor.",
            confidence=0.8,
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="objective:score",
                    direction=MetricEffectDirection.DECREASE,
                ),
            ),
            recommended_option_families=(family,),
            recommended_option_ids=(anchor_option_id,),
            action_template="Choose the exact local anchor action.",
            falsification_condition="The endpoint does not improve.",
        ),
        applicable_operator_kinds=("typed_mutation",),
        origin=InsightOrigin.MANUAL,
    )
    assert added
    request = HypothesisCompilationRequest(
        reference=entry.reference,
        insight=entry.draft,
        source_evidence_sha256=registered_source_evidence_sha256(entry),
        requested_operator_kind="typed_mutation",
        source_operator_kinds=entry.applicable_operator_kinds,
        parent_candidate_id=CandidateId(f"candidate_{contract.catalog_id}"),
        parent_configuration_sha256=contract.parent_configuration_sha256,
        finite_contract=contract,
        context_projection_sha256=_sha(f"{contract.catalog_id} context"),
        endpoint_definition_sha256=_sha(f"{contract.catalog_id} endpoint"),
    )
    return compile_registered_hypothesis_treatment(
        entry=entry,
        request=request,
        compiler=_ExactAnchorCompiler(),
    )


def _seal(
    contract: FiniteVariationContract,
    *,
    anchor_option_id: str,
    affected_path: str,
    compiler: _LocalSupportCompiler | None = None,
    phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
):
    compiled = _compiled_anchor(
        contract,
        anchor_option_id=anchor_option_id,
        affected_path=affected_path,
    )
    request = FiniteActionSetCompilationRequest(
        parent_candidate_id=compiled.request.parent_candidate_id,
        finite_contract=contract,
        anchor_option_id=anchor_option_id,
        anchor_option_identity_sha256=contract.resolve(anchor_option_id).identity_sha256,
        exact_anchor_requirement_sha256=compiled.requirement.requirement_sha256,
        card_reference=compiled.request.reference,
        card_content_sha256=compiled.request.insight.content_sha256,
        context_projection_sha256=compiled.request.context_projection_sha256,
        endpoint_definition_sha256=compiled.request.endpoint_definition_sha256,
        required_cardinality=4,
    )
    selected_compiler = compiler or _LocalSupportCompiler(
        tuple(option.option_id for option in contract.options)
    )
    authority, draft = compile_and_seal_finite_action_set(
        compiled_anchor=compiled,
        request=request,
        compiler=selected_compiler,
        phenotype_identity=phenotype_identity,
        source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
    )
    return compiled, request, authority, draft


@pytest.mark.parametrize(
    "contract,anchor,affected",
    [
        (_scalar_contract(), "local.value_n1", "$.value"),
        (_sequence_contract(), "local.sequence_0_c", "$.sequence[0]"),
    ],
)
def test_same_generic_authority_seals_two_structurally_distinct_adapters(
    contract: FiniteVariationContract,
    anchor: str,
    affected: str,
) -> None:
    compiled, request, authority, draft = _seal(
        contract,
        anchor_option_id=anchor,
        affected_path=affected,
    )

    assert authority.support.cardinality == 4
    assert authority.support.anchor_option_id == anchor
    assert authority.support.source_contract_sha256 == contract.identity_sha256
    assert authority.support.support_contract.options == contract.options
    assert authority.support.presentation.ordered_option_ids == tuple(
        option.option_id for option in contract.options
    )
    assert len(
        {value.phenotype_identity_sha256 for value in authority.support.options}
    ) == 4
    assert authority.card.reference == compiled.request.reference
    assert authority.card.exact_anchor_requirement_sha256 == (
        compiled.requirement.requirement_sha256
    )
    assert authority.support_compilation_request_sha256 == request.request_sha256
    assert authority.support_compilation_draft_sha256 == draft.draft_sha256
    assert authority.current_outcome_access is False
    # Sealing the neighbourhood did not widen or rewrite the old exact treatment.
    assert tuple(
        action.option_id for action in compiled.requirement.allowed_actions
    ) == (anchor,)
    assert compiled.treatment_evidence.recommended_option_ids == (anchor,)


def test_catalog_legal_but_support_illegal_and_wrong_anchor_fail_before_sealing() -> None:
    contract = _scalar_contract()
    compiled = _compiled_anchor(
        contract,
        anchor_option_id="local.value_n1",
        affected_path="$.value",
    )
    request = FiniteActionSetCompilationRequest(
        parent_candidate_id=compiled.request.parent_candidate_id,
        finite_contract=contract,
        anchor_option_id="local.value_n1",
        anchor_option_identity_sha256=contract.resolve("local.value_n1").identity_sha256,
        exact_anchor_requirement_sha256=compiled.requirement.requirement_sha256,
        card_reference=compiled.request.reference,
        card_content_sha256=compiled.request.insight.content_sha256,
        context_projection_sha256=compiled.request.context_projection_sha256,
        endpoint_definition_sha256=compiled.request.endpoint_definition_sha256,
        required_cardinality=4,
    )
    outside = _LocalSupportCompiler(
        (
            "local.value_n2",
            "local.value_n1",
            "local.value_p1",
            "local.not_in_contract",
        )
    )
    with pytest.raises(ValueError, match="outside the sealed contract"):
        compile_and_seal_finite_action_set(
            compiled_anchor=compiled,
            request=request,
            compiler=outside,
            phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
            source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
        )

    with pytest.raises(ValueError, match="differs from its exact compiled anchor"):
        compile_and_seal_finite_action_set(
            compiled_anchor=compiled,
            request=replace(
                request,
                anchor_option_id="local.value_p1",
                anchor_option_identity_sha256=(
                    contract.resolve("local.value_p1").identity_sha256
                ),
            ),
            compiler=_LocalSupportCompiler(
                tuple(option.option_id for option in contract.options)
            ),
            phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
            source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
        )


def test_authority_rejects_outcome_access_and_phenotype_aliases() -> None:
    contract = _scalar_contract()
    compiled = _compiled_anchor(
        contract,
        anchor_option_id="local.value_n1",
        affected_path="$.value",
    )
    with pytest.raises(ValueError, match="outcome-blind"):
        FiniteActionSetCompilationRequest(
            parent_candidate_id=compiled.request.parent_candidate_id,
            finite_contract=contract,
            anchor_option_id="local.value_n1",
            anchor_option_identity_sha256=contract.resolve(
                "local.value_n1"
            ).identity_sha256,
            exact_anchor_requirement_sha256=compiled.requirement.requirement_sha256,
            card_reference=compiled.request.reference,
            card_content_sha256=compiled.request.insight.content_sha256,
            context_projection_sha256=compiled.request.context_projection_sha256,
            endpoint_definition_sha256=compiled.request.endpoint_definition_sha256,
            required_cardinality=4,
            current_outcome_access=True,
        )

    class _AliasingPhenotypePolicy:
        policy_id = "fixture_aliasing_phenotype"
        policy_version = 1

        @staticmethod
        def identify(configuration: object) -> PhenotypeIdentity:
            del configuration
            return PhenotypeIdentity(
                policy_id="fixture_aliasing_phenotype",
                policy_version=1,
                value_sha256="a" * 64,
            )

    with pytest.raises(ValueError, match="support phenotypes must be unique"):
        _seal(
            contract,
            anchor_option_id="local.value_n1",
            affected_path="$.value",
            phenotype_identity=_AliasingPhenotypePolicy(),
        )


def test_compiler_identity_toctou_fails_closed() -> None:
    contract = _scalar_contract()

    class _MutatingCompiler(_LocalSupportCompiler):
        def compile(
            self,
            request: FiniteActionSetCompilationRequest,
        ) -> FiniteActionSetDraft:
            draft = super().compile(request)
            self.definition_sha256 = _sha("mutated compiler identity")
            return draft

    compiler = _MutatingCompiler(tuple(option.option_id for option in contract.options))
    with pytest.raises(ValueError, match="identity changed during compile"):
        _seal(
            contract,
            anchor_option_id="local.value_n1",
            affected_path="$.value",
            compiler=compiler,
        )
