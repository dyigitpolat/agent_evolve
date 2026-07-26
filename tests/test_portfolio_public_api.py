from __future__ import annotations

import asyncio
import hashlib
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

import agent_evolve.agentic as ae


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _frozen_object(value: dict[str, object]) -> ae.FrozenJsonObject:
    frozen = ae.freeze_json(value)
    assert type(frozen) is ae.FrozenJsonObject
    return frozen


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    alias: str


class _Problem:
    candidate_model = _Configuration
    objectives = (ae.ObjectiveSpec("score", "min"),)

    @staticmethod
    def search_space_description() -> str:
        return "An integer coordinate plus an evaluator-inert label."

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        del configuration
        raise AssertionError("the benchmark detailed evaluator is authoritative")


class _DetailedEvaluator:
    evaluator_identity = ae.EvaluatorIdentity(
        "portfolio_public_fake",
        1,
        _sha("portfolio-public-fake-evaluator-v1"),
    )

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> ae.DetailedEvaluationPayload:
        parsed = _Configuration.model_validate(configuration, strict=True)
        self.calls += 1
        return ae.DetailedEvaluationPayload(
            failure=None,
            objectives=(("score", float(parsed.x)),),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


def _outcome_relation() -> ae.OutcomeRelationPolicyBinding:
    def compare(left, right):
        left_score = dict(left.objectives)["score"]
        right_score = dict(right.objectives)["score"]
        if left_score < right_score:
            return ae.OutcomeRelation.BETTER
        if left_score > right_score:
            return ae.OutcomeRelation.WORSE
        return ae.OutcomeRelation.EQUIVALENT

    return ae.OutcomeRelationPolicyBinding(
        compare=compare,
        policy_id="portfolio_public_score_order",
        policy_version=1,
        definition_sha256=_sha("portfolio-public-score-order-v1"),
    )


def _reward() -> ae.RewardPolicyBinding:
    def score(child, parents, objectives):
        del objectives
        if not child.valid:
            return -1.0
        parent_scores = tuple(parent.objective_map["score"] for parent in parents)
        return float(min(parent_scores) - child.objective_map["score"])

    return ae.RewardPolicyBinding(
        score=score,
        definition_hash=_sha("portfolio-public-parent-delta-v1"),
    )


def _phenotype_policy() -> ae.SemanticProjectionPhenotypeIdentityPolicy:
    def project(configuration):
        thawed = ae.thaw_json(configuration)
        assert type(thawed) is dict
        return {"x": thawed["x"]}

    return ae.SemanticProjectionPhenotypeIdentityPolicy(
        policy_id="portfolio_public_integer_phenotype",
        policy_version=1,
        projector=project,
    )


class _Catalog:
    catalog_id = "portfolio_public_fake"
    catalog_version = 1
    definition_sha256 = _sha("portfolio-public-fake-catalog-v1")
    option_families = ("integer_step",)

    def options(
        self,
        parent_configuration: ae.FrozenJsonObject,
    ) -> tuple[ae.FiniteVariationOption, ...]:
        parent = ae.thaw_json(parent_configuration)
        assert type(parent) is dict
        x = parent["x"]
        assert type(x) is int
        parent_sha256 = ae.typed_json_sha256(parent_configuration)
        return (
            ae.FiniteVariationOption(
                option_id="move.down_one",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen_object(
                    {"x": x - 1, "alias": "one_step"}
                ),
                family="integer_step",
                description="Decrease the integer coordinate by one.",
            ),
            ae.FiniteVariationOption(
                option_id="move.down_two",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen_object(
                    {"x": x - 2, "alias": "two_steps"}
                ),
                family="integer_step",
                description="Decrease the integer coordinate by two.",
            ),
        )


class _UnusedGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, request):
        del request
        self.calls += 1
        raise AssertionError("ranked portfolio children are engine-materialized")

    async def reflect(self, request):
        del request
        self.calls += 1
        raise AssertionError("this portfolio wave does not request reflection")


class _FirstOptionSelector:
    def __init__(self) -> None:
        self.calls = 0

    async def select(
        self,
        request: ae.PortfolioSelectionRequest,
    ) -> ae.PortfolioSelectionResult:
        self.calls += 1
        option = request.finite_variation_contract.options[0]
        decision = ae.resolve_ranked_portfolio_decision(
            request,
            (
                ae.PortfolioMemberDraft(
                    option_id=option.option_id,
                    supporting_card_keys=(request.cards[0].card_key,),
                    effect_predictions=(
                        ae.MetricEffectPrediction(
                            "score",
                            ae.MetricEffectDirection.DECREASE,
                        ),
                    ),
                    design_rationale="Choose the sealed one-step intervention.",
                ),
            ),
            policy_id="portfolio_public_first_option",
            policy_version=1,
            policy_definition_sha256=_sha("portfolio-public-first-option-v1"),
        )
        return ae.PortfolioSelectionResult(
            decision=decision,
            telemetry=ae.AgenticCallTelemetry(
                requested_model="fake/public-selector",
                resolved_model="fake/public-selector-v1",
                resolved_provider="provider-free",
                provider_response_id="portfolio-public-response",
                finish_reason="stop",
                input_tokens=12,
                output_tokens=4,
                reasoning_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=10,
            ),
        )


def _composition():
    evaluator = _DetailedEvaluator()
    relation = _outcome_relation()
    generator = _UnusedGenerator()
    selector = _FirstOptionSelector()
    benchmark = ae.AgenticBenchmark(
        problem=_Problem(),
        reward=_reward(),
        detailed_evaluator=evaluator,
        outcome_relation=relation,
        phenotype_identity=_phenotype_policy(),
        finite_variation_catalogs=(_Catalog(),),
    )
    ids = ae.DeterministicIdFactory("portfolio_public_composition")
    memory = ae.InsightMemoryBank(id_factory=ids)
    composition = ae.compose_portfolio_evolution(
        benchmark,
        generator=generator,
        selector=selector,
        seed=17,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=2,
    )
    return composition, evaluator, relation, generator, selector


def test_public_composition_executes_a_benchmark_neutral_portfolio_wave() -> None:
    composition, evaluator, relation, generator, selector = _composition()

    async def scenario():
        parent = await composition.engine.register_seed(
            {"x": 3, "alias": "seed"},
            label="seed",
        )
        entry, added = composition.memory.add(
            ae.InsightDraft(
                claim="A smaller coordinate may improve the score.",
                trigger="The current coordinate remains positive.",
                mechanism="A finite decrement directly reduces the evaluator metric.",
                affected_paths=("$.x",),
                evidence_summary="This is a provider-free public API fixture.",
                confidence=0.5,
            )
        )
        assert added
        contract = composition.bind_finite_variation(
            "portfolio_public_fake",
            parent.configuration,
        )
        request = ae.PortfolioSelectionRequest(
            call_id=composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction="Select one ranked sealed intervention.",
            context=_frozen_object({"domain": "benchmark_neutral_fake"}),
            finite_variation_contract=contract,
            cards=(
                ae.PortfolioCard(
                    card_key="card.1",
                    reference=entry.reference,
                    content_sha256=_sha("portfolio-public-card-content"),
                    evidence_sha256=_sha("portfolio-public-card-evidence"),
                    prompt_payload=_frozen_object(
                        {"claim": "Smaller coordinates may be useful."}
                    ),
                ),
            ),
            portfolio_size=1,
            required_metric_ids=("score",),
        )
        return await composition.portfolio.run(
            ae.PortfolioVariationWaveRequest(
                selection_request=request,
                parent=parent,
                generation=1,
                label_prefix="public_portfolio",
            )
        )

    result = asyncio.run(scenario())

    assert type(result) is ae.PortfolioVariationWaveResult
    assert result.receipt.members[0].better_than_any_parent
    assert result.candidates[0].objective_map == {"score": 2.0}
    assert evaluator.calls == 2
    assert selector.calls == 1
    assert generator.calls == 0
    assert composition.outcome_relation is relation
    assert composition.engine.problem is composition.benchmark.problem
    assert composition.engine.ids is composition.id_factory
    assert composition.engine.memory is composition.memory
    assert composition.portfolio.engine is composition.engine
    assert composition.portfolio.ids is composition.id_factory
    assert composition.portfolio.memory is composition.memory
    assert composition.engine.reward_binding.binding_sha256 == (
        composition.benchmark.reward.binding_sha256
    )
    assert composition.engine.identify_phenotype(
        {"x": 2, "alias": "left"}
    ) == composition.engine.identify_phenotype({"x": 2, "alias": "right"})
    assert not hasattr(composition, "optimizer")
    assert not hasattr(composition, "planner")
    with pytest.raises(FrozenInstanceError):
        composition.memory = ae.InsightMemoryBank(  # type: ignore[misc]
            id_factory=composition.id_factory
        )


def test_outcome_blind_eligibility_is_available_from_the_public_facade() -> None:
    composition, *_ = _composition()
    contract = composition.bind_finite_variation(
        "portfolio_public_fake",
        {"x": 3, "alias": "seed"},
    )
    exact_bindings = ae.exact_configuration_phenotype_bindings(contract)
    assert tuple(
        binding.phenotype_identity_sha256 for binding in exact_bindings
    ) == tuple(option.child_configuration_sha256 for option in contract.options)

    semantic_alias = _sha("portfolio-public-semantic-alias")
    bindings = tuple(
        ae.OptionPhenotypeBinding(
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            phenotype_identity_sha256=semantic_alias,
        )
        for option in contract.options
    )
    view = ae.eligible_finite_variation_view(
        contract=contract,
        option_phenotypes=bindings,
        known_phenotype_sha256s=(),
    )

    assert view.receipt.base_contract_identity_sha256 == contract.identity_sha256
    assert view.receipt.eligible_option_ids == ("move.down_one",)
    assert view.receipt.alias_excluded_option_ids == ("move.down_two",)
    assert tuple(option.option_id for option in view.contract.options) == (
        "move.down_one",
    )


def test_package_root_reexports_the_primary_portfolio_surface() -> None:
    import agent_evolve as package

    for name in (
        "PortfolioEvolutionComposition",
        "PortfolioVariationWaveRequest",
        "PortfolioVariationWaveResult",
        "FiniteVariationEligibilityReceipt",
        "compose_portfolio_evolution",
        "eligible_finite_variation_view",
    ):
        assert getattr(package, name) is getattr(ae, name)
