"""Provider-free smoke example for AgentEvolve's public inverted API."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import (
    AgenticBenchmark,
    DetailedEvaluationPayload,
    DeterministicIdFactory,
    EvaluatorIdentity,
    FiniteVariationOption,
    FrozenJsonObject,
    ObjectiveSpec,
    OptimizerBudget,
    REWARD_DEFINITION_HASH,
    RewardPolicyBinding,
    SemanticProjectionPhenotypeIdentityPolicy,
    compose_agentic_optimizer,
    default_parent_relative_reward,
    freeze_json,
    objective_pareto_outcome_binding,
    thaw_json,
    typed_json_sha256,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    setting: int
    inert_label: str


class ExampleProblem:
    candidate_model = CandidateModel
    objectives = (ObjectiveSpec("cost", "min"),)

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        # Detailed mode calls ExampleEvidence instead.  Keeping this method
        # satisfies the generic Problem protocol and supports legacy mode.
        return {"cost": float(configuration["setting"])}


class ExampleEvidence:
    evaluator_identity = EvaluatorIdentity(
        "example_evidence",
        1,
        _sha("example-evaluator-context-v1"),
    )

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("cost", float(configuration["setting"])),),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


class StepCatalog:
    catalog_version = 1

    def __init__(self, catalog_id: str, delta: int) -> None:
        self.catalog_id = catalog_id
        self.delta = delta
        self.definition_sha256 = _sha(f"{catalog_id}:{delta}:v1")

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        child = {
            "setting": int(parent["setting"]) + self.delta,
            "inert_label": f"step_{self.delta:+d}",
        }
        return (
            FiniteVariationOption(
                option_id=f"setting.step_{'up' if self.delta > 0 else 'down'}",
                parent_configuration_sha256=typed_json_sha256(
                    parent_configuration
                ),
                child_configuration=freeze_json(child),
                family="setting_step",
                description="Apply one benchmark-owned coordinated setting step.",
            ),
        )


class NoCallGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("the construction smoke test makes no model calls")

    async def reflect(self, request):
        del request
        raise AssertionError("the construction smoke test makes no model calls")


class NoPlanPlanner:
    def plan(self, state, budget):
        del state, budget
        raise AssertionError("a zero-generation smoke test makes no plan")


def build():
    problem = ExampleProblem()

    def project(configuration):
        value = thaw_json(configuration)
        assert type(value) is dict
        return {"setting": value["setting"]}

    benchmark = AgenticBenchmark(
        problem=problem,
        reward=RewardPolicyBinding(
            default_parent_relative_reward,
            REWARD_DEFINITION_HASH,
        ),
        detailed_evaluator=ExampleEvidence(),
        outcome_relation=objective_pareto_outcome_binding(problem.objectives),
        phenotype_identity=SemanticProjectionPhenotypeIdentityPolicy(
            policy_id="example_setting_semantics",
            policy_version=1,
            projector=project,
        ),
        finite_variation_catalogs=(
            StepCatalog("example_step_up", +1),
            StepCatalog("example_step_down", -1),
        ),
    )
    return compose_agentic_optimizer(
        benchmark,
        generator=NoCallGenerator(),
        planner=NoPlanPlanner(),
        budget=OptimizerBudget(1, 0, 0),
        seed=7,
        id_factory=DeterministicIdFactory("public_api_smoke"),
    )


if __name__ == "__main__":
    composition = build()
    print(composition.benchmark.finite_variation_catalog_identities)
