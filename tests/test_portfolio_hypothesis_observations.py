from __future__ import annotations

import asyncio
import hashlib

from agent_evolve.application.portfolio_evolution import PortfolioEvolution
from agent_evolve.application.portfolio_hypothesis_observations import (
    CampaignHypothesisObservationExclusionReason,
    FinitePortfolioActionSemanticsCompiler,
    ObjectiveDeltaMetricEffectProjector,
    project_portfolio_hypothesis_evidence,
    project_portfolio_hypothesis_observations,
)
from agent_evolve.domain.outcome import FailureCode
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.memory.global_falsification import (
    EvidenceProvenance,
    InterventionIdentifiability,
)
from tests.test_portfolio_evolution import (
    _CandidateInfeasibilityEvaluator,
    _build_wave,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_real_portfolio_result_projects_one_authenticated_observation_per_action() -> (
    None
):
    async def scenario() -> None:
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "hypothesis_observation_join"
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave, defer_memory_credit=True)
        observations = project_portfolio_hypothesis_observations(
            campaign_sha256=_sha("campaign"),
            event_index=1,
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            waves=(wave,),
            results=(result,),
            metric_projector=ObjectiveDeltaMetricEffectProjector(_sha("adjudicator")),
            semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
        )

        assert len(observations) == len(result.receipt.members) == 3
        assert {value.source_evidence_id for value in observations} == {
            value.outcome_sha256 for value in result.receipt.members
        }
        member_by_evidence = {
            value.outcome_sha256: value for value in result.receipt.members
        }
        for observation in observations:
            assert observation.event_index == 1
            assert observation.provenance is EvidenceProvenance.DIRECT_MUTATION
            assert observation.intervention_identifiability is (
                InterventionIdentifiability.EXACT_SINGLE
            )
            assert observation.mechanism_identifying_design is False
            assert observation.metrics[0].metric_id == "loss"
            member = member_by_evidence[observation.source_evidence_id]
            assert observation.parent_candidate_id == wave.parent.candidate_id
            assert (
                observation.child_candidate_id
                == member.materialization.candidate_id
            )
            assert (
                observation.operator_invocation_id
                == member.operator_invocation_id
            )
            assert observation.finite_contract_identity_sha256 == (
                wave.selection_request.finite_variation_contract.identity_sha256
            )
            action = thaw_json(observation.observed_action)
            assert action["operator_family"] == "typed_mutation"
            assert action["option_id"] in {"alpha.x1", "beta.y1", "gamma.xy"}
            assert action["finite_contract_identity_sha256"] == (
                observation.finite_contract_identity_sha256
            )
            assert action["changed_paths"] == list(observation.affected_paths)
            assert action["compiler"] == {
                "compiler_id": observation.action_semantics_compiler_id,
                "compiler_version": observation.action_semantics_compiler_version,
                "definition_sha256": (
                    observation.action_semantics_definition_sha256
                ),
            }
            compiler = FinitePortfolioActionSemanticsCompiler()
            assert observation.action_semantics_compiler_id == compiler.compiler_id
            assert (
                observation.action_semantics_compiler_version
                == compiler.compiler_version
            )
            assert (
                observation.action_semantics_definition_sha256
                == compiler.definition_sha256
            )

    asyncio.run(scenario())


def test_metric_projector_uses_raw_child_minus_parent_direction() -> None:
    async def scenario() -> None:
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "hypothesis_metric_direction"
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave, defer_memory_credit=True)
        observations = project_portfolio_hypothesis_observations(
            campaign_sha256=_sha("campaign"),
            event_index=2,
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            waves=(wave,),
            results=(result,),
            metric_projector=ObjectiveDeltaMetricEffectProjector(_sha("adjudicator")),
            semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
        )
        deltas = sorted(value.metrics[0].delta for value in observations)
        assert deltas == [1.0, 1.0, 4.0]

    asyncio.run(scenario())


def test_infeasible_rank_gets_explicit_exclusion_and_no_metric_observation() -> None:
    async def scenario() -> None:
        evaluator = _CandidateInfeasibilityEvaluator()
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "hypothesis_infeasible_exclusion",
            detailed_evaluator=evaluator,
        )
        evaluator.reset_evidence()
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave, defer_memory_credit=True)
        projection = project_portfolio_hypothesis_evidence(
            campaign_sha256=_sha("campaign"),
            event_index=3,
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            waves=(wave,),
            results=(result,),
            metric_projector=ObjectiveDeltaMetricEffectProjector(_sha("adjudicator")),
            semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
        )

        assert len(result.receipt.members) == 3
        assert len(projection.observations) == 2
        assert len(projection.exclusions) == 1
        assert set(projection.ranked_source_evidence_ids) == {
            value.outcome_sha256 for value in result.receipt.members
        }
        assert {
            value.source_evidence_id for value in projection.observations
        }.isdisjoint(
            value.source_evidence_id for value in projection.exclusions
        )
        exclusion = projection.exclusions[0]
        assert exclusion.rank == 3
        assert exclusion.reason is (
            CampaignHypothesisObservationExclusionReason.CANDIDATE_INFEASIBLE
        )
        assert exclusion.candidate_failure.failure_code is (
            FailureCode.EVALUATOR_DECLARED_INFEASIBLE
        )
        record = projection.to_record()
        assert record["ranked_itt_member_count"] == 3
        assert record["observation_count"] == 2
        assert record["exclusion_count"] == 1
        assert record["resampled_member_count"] == 0
        assert record["exclusions"][0]["metric_projection_executed"] is False
        assert record["exclusions"][0]["semantics_compiler_executed"] is False

    asyncio.run(scenario())
