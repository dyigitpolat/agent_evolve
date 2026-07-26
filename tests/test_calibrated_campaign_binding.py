from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.calibrated_campaign import (
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.application.finite_variation_eligibility import (
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import InsightId, LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioSelectionRequest,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseCardSupport,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _variation(benchmark_sha256: str) -> ParentVariationBinding:
    base = bind_finite_variation_catalog(
        BoilsFiniteVariationCatalog(),
        _object({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
    )
    view = eligible_finite_variation_view(
        contract=base,
        option_phenotypes=exact_configuration_phenotype_bindings(base),
        known_phenotype_sha256s=(),
    )
    return ParentVariationBinding(
        benchmark_sha256=benchmark_sha256,
        parent_configuration_sha256=(
            view.contract.parent_configuration_sha256
        ),
        known_phenotype_sha256s=(),
        contract=view.contract,
        eligibility_receipt=view.receipt,
    )


def _card() -> PortfolioCard:
    return PortfolioCard(
        card_key="card.a",
        reference=InsightRef(InsightId("insight_campaign_binding"), 1),
        content_sha256=_sha("card-content"),
        evidence_sha256=_sha("card-evidence"),
        prompt_payload=_object({"claim": "Test one mechanistic hypothesis."}),
    )


def _request(variation: ParentVariationBinding) -> PortfolioSelectionRequest:
    return PortfolioSelectionRequest(
        call_id=LLMCallId("call_calibrated_campaign_binding"),
        operation="select_calibrated_portfolio",
        instruction="This caller text remains identity-bound but is not rendered.",
        context=_object({"workload": "provider-free-real-boils-palette"}),
        finite_variation_contract=variation.contract,
        cards=(_card(),),
        portfolio_size=4,
        required_metric_ids=("total_levels", "total_lut_count"),
        require_supporting_cards=True,
        require_pairwise_disjoint_parent_patches=True,
        max_output_tokens=65_536,
        temperature=0.0,
    )


def _factory(benchmark_sha256: str) -> CalibratedCampaignBindingFactory:
    return CalibratedCampaignBindingFactory(
        scope=ForecastCalibrationScope(
            model_profile_sha256=_sha("model-profile"),
            prompt_definition_sha256=(
                CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256
            ),
            selector_policy_definition_sha256=(
                CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            ),
            benchmark_sha256=benchmark_sha256,
            session_sha256=_sha("session"),
        ),
        objectives=equal_weight_slate_objectives(
            (
                ObjectiveSpec("total_levels", "min"),
                ObjectiveSpec("total_lut_count", "min"),
            )
        ),
        ledger=PortfolioOutcomeFeedbackLedger(),
    )


def test_builds_complete_prior_only_binding_for_real_200_option_palette() -> None:
    benchmark_sha256 = _sha("benchmark")
    variation = _variation(benchmark_sha256)
    request = _request(variation)

    binding = _factory(benchmark_sha256).build(
        request=request,
        variation=variation,
        wave_index=1,
        frozen_archive_snapshot_sha256=_sha("archive-g0"),
    )

    binding.require_request(request)
    assert binding.context.wave_index == 1
    assert binding.context.assigned_card_keys == ("card.a",)
    assert binding.context.calibration_snapshot.observation_count == 0
    assert binding.context.calibration_snapshot.cutoff_wave_index_exclusive == 1
    assert len(binding.option_evidence) == 200
    assert len({value.locus_key for value in binding.option_evidence}) == 20
    assert all(
        value.structural_evidence.archive_novelty_score == 1.0
        for value in binding.option_evidence
    )
    assert binding.option_prompt_projection is None
    assert binding.to_record()["schema_version"] == 1
    assert "option_prompt_projection" not in binding.to_record()


def test_factory_owns_and_strictly_replays_opt_in_prompt_projection() -> None:
    benchmark_sha256 = _sha("benchmark")
    variation = _variation(benchmark_sha256)
    request = _request(variation)
    base = _factory(benchmark_sha256)
    policy = FiniteOptionPromptProjectionPolicy(
        metadata_keys=("abc_commands_json", "position", "replacement_action")
    )
    factory = replace(base, option_prompt_projection=policy)

    binding = factory.build(
        request=request,
        variation=variation,
        wave_index=1,
        frozen_archive_snapshot_sha256=_sha("archive-g0"),
    )

    projection = binding.option_prompt_projection
    assert projection is not None
    binding.require_request(request)
    assert binding.prompt_records_for(request) == projection.prompt_records()
    assert binding.prompt_projection_contract_for(request) == (
        projection.to_prompt_contract_record()
    )
    assert binding.to_record()["option_prompt_projection"] == (
        projection.to_binding_record()
    )
    assert binding.to_record()["schema_version"] == 2

    tampered_projection = replace(
        projection,
        records=tuple(reversed(projection.records)),
    )
    tampered_binding = replace(
        binding,
        option_prompt_projection=tampered_projection,
    )
    with pytest.raises(ValueError, match="differs from the sealed contract"):
        tampered_binding.require_request(request)


def test_factory_rejects_foreign_benchmark_and_unassigned_card() -> None:
    benchmark_sha256 = _sha("benchmark")
    variation = _variation(benchmark_sha256)
    request = _request(variation)
    factory = _factory(benchmark_sha256)

    with pytest.raises(ValueError, match="assigned cards escape"):
        factory.build(
            request=request,
            variation=variation,
            wave_index=1,
            frozen_archive_snapshot_sha256=_sha("archive"),
            assigned_card_keys=("card.foreign",),
        )

    foreign = _factory(_sha("foreign-benchmark"))
    with pytest.raises(ValueError, match="foreign benchmark"):
        foreign.build(
            request=request,
            variation=variation,
            wave_index=1,
            frozen_archive_snapshot_sha256=_sha("archive"),
        )


def test_hard_dose_owns_assigned_cards_when_default_assignment_is_disabled() -> None:
    benchmark_sha256 = _sha("benchmark")
    variation = _variation(benchmark_sha256)
    base_request = replace(
        _request(variation),
        require_supporting_cards=False,
    )
    card = base_request.cards[0]
    option = variation.contract.options[0]
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=(
            PortfolioMemoryDoseCardSupport(
                card_key=card.card_key,
                card_content_sha256=card.content_sha256,
                finite_contract_identity_sha256=variation.contract.identity_sha256,
                compatible_options=((option.option_id, option.identity_sha256),),
                support_policy_id="test_exact_support",
                support_policy_version=1,
                support_policy_definition_sha256=_sha("exact-support"),
            ),
        ),
        proposed_supported_member_bounds=(1, 1),
        evaluated_supported_member_bounds=(1, 1),
        minimum_unattributed_proposed_members=7,
        minimum_unattributed_evaluated_members=3,
        maximum_cards_per_member=1,
        require_every_assigned_card=True,
    )
    request = replace(base_request, memory_dose_contract=dose)
    factory = replace(
        _factory(benchmark_sha256),
        assign_all_cards_by_default=False,
    )

    binding = factory.build(
        request=request,
        variation=variation,
        wave_index=5,
        frozen_archive_snapshot_sha256=_sha("archive-g4"),
    )

    assert binding.context.assigned_card_keys == dose.assigned_card_keys

def test_equal_weight_projection_preserves_mixed_objective_senses() -> None:
    objectives = equal_weight_slate_objectives(
        (
            ObjectiveSpec("quality", "max"),
            ObjectiveSpec("runtime", "min"),
        )
    )
    assert tuple(value.metric_id for value in objectives) == ("quality", "runtime")
    assert tuple(value.goal.value for value in objectives) == ("maximize", "minimize")
    assert tuple(value.weight for value in objectives) == (1.0, 1.0)
