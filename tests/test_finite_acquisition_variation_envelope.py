from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from agent_evolve.agentic import ObjectiveSpec
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import (
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.campaign_variation_envelope import (
    CampaignVariationEnvelopeLane,
    CampaignVariationEnvelopeRequest,
    campaign_variation_envelope_context_record,
    campaign_variation_envelope_trace_record,
    decode_campaign_variation_envelope_trace_record,
    validate_campaign_variation_envelope_result,
)
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    ParentVariationBinding,
)
from agent_evolve.application.finite_acquisition_variation_envelope import (
    ProtectedFiniteAcquisitionVariationEnvelope,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContextRegistry,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    required_ranked_source_evaluation_option_ids,
    required_source_evaluation_option_ids,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionDecision,
    FiniteAcquisitionObjective,
    FiniteAcquisitionRequest,
    FiniteAcquisitionSelection,
)
from agent_evolve.ports.finite_acquisition_space import (
    FiniteAcquisitionSpaceRequest,
)
from agent_evolve.ports.variation_source import (
    VARIATION_SOURCE_RANK_METADATA_KEY,
    finite_variation_operator_id,
    finite_variation_source_id,
    finite_variation_source_minimum_counts,
    finite_variation_candidate_pool_required_option_ids,
)
from examples.development.analyze_systematic_campaign_trace import (
    _expert_union_support,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(x: float, y: float) -> FrozenJsonObject:
    value = freeze_json({"x": x, "y": y})
    assert type(value) is FrozenJsonObject
    return value


def _candidate(index: int, x: float, y: float) -> EvolutionCandidate:
    configuration = _object(x, y)
    identity = typed_json_sha256(configuration)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_envelope_{index:06d}"),
            configuration_hash=identity,
            configuration_artifact_hash=identity,
            proposal_sequence=index,
        ),
        configuration=configuration,
        objectives=(
            ("cost", float((x - 0.15) ** 2 + y)),
            ("quality", float((y - 0.85) ** 2 + x)),
        ),
        valid=True,
        generation=0,
        label=f"seed_{index}",
    )


def _base_binding(
    *,
    parent: EvolutionCandidate,
    benchmark_sha256: str,
    known: tuple[str, ...],
    offset: float,
) -> ParentVariationBinding:
    parent_configuration = parent.configuration
    assert type(parent_configuration) is FrozenJsonObject
    parent_sha256 = typed_json_sha256(parent_configuration)
    options = tuple(
        FiniteVariationOption(
            option_id=f"base.{parent.candidate_id.value[-1]}.{index}",
            parent_configuration_sha256=parent_sha256,
            child_configuration=_object(offset + index * 0.01, 0.31 + index * 0.01),
            family="local",
            description=f"Base legal option {index}.",
        )
        for index in (1, 2)
    )
    contract = FiniteVariationContract(
        catalog_id="test_base_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("test-base-catalog"),
        parent_configuration=parent_configuration,
        options=options,
    )
    return ParentVariationBinding(
        benchmark_sha256=benchmark_sha256,
        parent_configuration_sha256=parent_sha256,
        known_phenotype_sha256s=known,
        contract=contract,
    )


@dataclass(frozen=True, slots=True)
class _GridSpace:
    space_id = "test_grid_space"
    space_version = 1
    definition_sha256 = _sha("test-grid-space")

    def candidates(
        self,
        request: FiniteAcquisitionSpaceRequest,
    ) -> tuple[FrozenJsonObject, ...]:
        excluded = set(request.excluded_configuration_sha256s)
        rows: list[FrozenJsonObject] = []
        for x_index in range(1, 20):
            for y_index in range(1, 20):
                value = _object(x_index / 20.0, y_index / 20.0)
                if typed_json_sha256(value) in excluded:
                    continue
                rows.append(value)
                if len(rows) == request.pool_size:
                    return tuple(rows)
        raise AssertionError("test grid cannot underfill")

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]:
        value = thaw_json(configuration)
        assert type(value) is dict
        return float(value["x"]), float(value["y"])


@dataclass(frozen=True, slots=True)
class _FirstBatchAcquisition:
    policy_id = "test_first_batch_acquisition"
    policy_version = 1
    definition_sha256 = _sha("test-first-batch-acquisition")

    def select(self, request: FiniteAcquisitionRequest) -> FiniteAcquisitionDecision:
        return FiniteAcquisitionDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            selected=tuple(
                FiniteAcquisitionSelection(
                    candidate_id=value.candidate_id,
                    configuration_sha256=value.configuration_sha256,
                    acquisition_value=float(request.batch_size - index),
                )
                for index, value in enumerate(
                    request.candidates[: request.batch_size]
                )
            ),
        )


def _scenario():
    first = _candidate(1, 0.0, 1.0)
    second = _candidate(2, 1.0, 0.0)
    archive = ParetoArchive(
        (ObjectiveSpec("cost", "min"), ObjectiveSpec("quality", "min"))
    )
    archive.consider(first)
    archive.consider(second)
    snapshot = archive.snapshot()
    state = OptimizerState(
        generation=0,
        candidates=(first, second),
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=2,
        logical_llm_calls=0,
    )
    benchmark_sha256 = _sha("test-benchmark")
    known = tuple(
        sorted(
            (
                first.occurrence.configuration_hash,
                second.occurrence.configuration_hash,
            )
        )
    )
    lanes = tuple(
        sorted(
            (
                CampaignVariationEnvelopeLane(
                    lane_id="elite",
                    parent=first,
                    base_variation=_base_binding(
                        parent=first,
                        benchmark_sha256=benchmark_sha256,
                        known=known,
                        offset=0.70,
                    ),
                ),
                CampaignVariationEnvelopeLane(
                    lane_id="explorer",
                    parent=second,
                    base_variation=_base_binding(
                        parent=second,
                        benchmark_sha256=benchmark_sha256,
                        known=known,
                        offset=0.80,
                    ),
                ),
            ),
            key=lambda value: value.lane_id,
        )
    )
    request = CampaignVariationEnvelopeRequest(
        campaign_scope_sha256=_sha("test-campaign"),
        generation=1,
        evaluation_slots_per_lane=4,
        state=state,
        archive_utility=ArchiveUtilitySnapshot(
            utility_id="test_utility",
            utility_version=1,
            definition_sha256=_sha("test-utility"),
            generation=1,
            benchmark_sha256=benchmark_sha256,
            archive_sha256=_sha("test-archive"),
            snapshot_receipt=freeze_json({"frame": "fixed"}),
            scalar_utility_hex=0.0.hex(),
        ),
        lanes=lanes,
    )
    envelope = ProtectedFiniteAcquisitionVariationEnvelope(
        objectives=(
            FiniteAcquisitionObjective("cost", "min", 0.0, 2.0),
            FiniteAcquisitionObjective("quality", "min", 0.0, 2.0),
        ),
        space=_GridSpace(),
        acquisition=_FirstBatchAcquisition(),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        pool_size=16,
        protected_batch_size=4,
        seed=7,
    )
    return envelope, request


def test_protected_acquisition_envelope_preserves_base_and_partitions_global_batch():
    envelope, request = _scenario()
    result = envelope.enrich(request)
    validate_campaign_variation_envelope_result(
        policy=envelope,
        request=request,
        result=result,
    )

    selected_children: set[str] = set()
    for source_lane, result_lane in zip(request.lanes, result.lanes, strict=True):
        contract = result_lane.variation.contract
        base_ids = {
            value.identity_sha256
            for value in source_lane.base_variation.contract.options
        }
        assert base_ids <= {value.identity_sha256 for value in contract.options}
        acquisition_options = tuple(
            value
            for value in contract.options
            if finite_variation_source_id(value) == "numerical_acquisition"
        )
        assert len(acquisition_options) == 2
        assert finite_variation_source_minimum_counts(contract) == (
            ("numerical_acquisition", 1),
        )
        assert {finite_variation_operator_id(value) for value in acquisition_options} == {
            "global"
        }
        ranks = {
            value.option_id: int(
                dict(value.metadata)[VARIATION_SOURCE_RANK_METADATA_KEY]
            )
            for value in acquisition_options
        }
        required = required_source_evaluation_option_ids(contract)
        assert len(required) == 1
        assert ranks[required[0]] == min(ranks.values())
        for option in acquisition_options:
            assert option.child_configuration_sha256 not in selected_children
            selected_children.add(option.child_configuration_sha256)
    assert len(selected_children) == 4
    evidence = thaw_json(result.evidence)
    assert evidence["strictly_prior_observation_count"] == 2
    assert evidence["reservoir_size"] == 16
    assert evidence["current_future_outcomes_consulted"] is False
    context = campaign_variation_envelope_context_record(result)
    assert context["result_sha256"] == result.result_sha256
    assert context["full_variation_contracts_repeated_in_context"] is False
    assert all(
        set(value)
        == {
            "eligible_option_count",
            "finite_contract_identity_sha256",
            "lane_id",
            "parent_configuration_sha256",
        }
        for value in context["lanes"]
    )

    trace = campaign_variation_envelope_trace_record(
        request=request,
        result=result,
    )
    assert trace["eligible_added_option_count"] == 4
    assert trace["eligible_option_occurrence_count"] == sum(
        len(value.variation.contract.options) for value in result.lanes
    )
    assert trace["full_child_configurations_included"] is True
    assert trace["full_expert_union_included"] is True
    assert trace["exposed_to_model_prompt"] is False
    assert len(trace["trace_receipt_sha256"]) == 64
    assert trace["evaluated_disposition_join"] == {
        "record": "portfolio_wave_receipts.action_attributions",
        "key": "option_identity_sha256",
    }
    assert thaw_json(freeze_json(trace)) == trace
    payload = decode_campaign_variation_envelope_trace_record(trace)
    assert payload["evidence"] == thaw_json(result.evidence)
    for lane, lane_trace in zip(result.lanes, payload["lanes"], strict=True):
        additions = [
            value
            for value in lane.variation.contract.options
            if finite_variation_source_id(value) == "numerical_acquisition"
        ]
        assert lane_trace["eligible_added_option_count"] == 2
        eligible = lane_trace["eligible_options"]
        assert len(eligible) == len(lane.variation.contract.options)
        assert {
            value["option"]["option_identity_sha256"]
            for value in eligible
            if value["support_origin"] == "envelope_addition"
        } == {value.identity_sha256 for value in additions}
        assert sum(value["support_origin"] == "base" for value in eligible) == len(
            lane.variation.contract.options
        ) - len(additions)
        for value in eligible:
            configuration = freeze_json(value["child_configuration"])
            assert typed_json_sha256(configuration) == value["option"][
                "child_configuration_sha256"
            ]
            assert len(value["phenotype_identity_sha256"]) == 64
            assert value["eligibility_disposition"] == "eligible"


def test_expert_union_trace_joins_eligible_support_to_one_materialized_outcome():
    envelope, request = _scenario()
    result = envelope.enrich(request)
    trace = campaign_variation_envelope_trace_record(
        request=request,
        result=result,
    )
    lane = result.lanes[0]
    request_lane = next(value for value in request.lanes if value.lane_id == lane.lane_id)
    selected = lane.variation.contract.options[0]
    candidate_id = "candidate_expert_union_000001"
    stages = [
        {
            "payload": {
                "stage_receipt": {
                    "generation": request.generation,
                    "result": {
                        "variation_envelope_trace_receipt": trace,
                        "portfolio_wave_receipts": [
                            {
                                "parent_candidate_id": (
                                    request_lane.parent.candidate_id.value
                                ),
                                "action_attributions": [
                                    {
                                        "selected_member": {
                                            "option_identity_sha256": (
                                                selected.identity_sha256
                                            )
                                        },
                                        "candidate_id": candidate_id,
                                    }
                                ],
                            }
                        ],
                    },
                }
            }
        }
    ]
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "positive_individual_marginal": True,
            "individual_marginal_hypervolume": 0.125,
            "admitted_to_stage_front": True,
            "admitted_to_final_front": False,
        }
    ]

    support = _expert_union_support(stages, candidate_rows)

    assert support["authenticated_full_union_trace_stage_rate"] == 1.0
    assert support["eligible_option_occurrence_count"] == trace[
        "eligible_option_occurrence_count"
    ]
    assert support["evaluated_option_occurrence_count"] == 1
    assert support["positive_individual_marginal_count"] == 1
    assert support["stage_front_admission_count"] == 1
    assert support["final_front_admission_count"] == 0
    assert sum(
        value["eligible_option_occurrence_count"]
        for value in support["support_origin_rows"]
    ) == trace["eligible_option_occurrence_count"]


def test_expanded_acquisition_supply_protects_each_lanes_native_rank_prefix():
    envelope, request = _scenario()
    expanded = ProtectedFiniteAcquisitionVariationEnvelope(
        objectives=envelope.objectives,
        space=envelope.space,
        acquisition=envelope.acquisition,
        phenotype_identity=envelope.phenotype_identity,
        pool_size=envelope.pool_size,
        protected_batch_size=8,
        source_minimum_per_lane=2,
        seed=envelope.seed,
    )

    result = expanded.enrich(request)
    validate_campaign_variation_envelope_result(
        policy=expanded,
        request=request,
        result=result,
    )

    all_required_ranks: list[int] = []
    for lane in result.lanes:
        contract = lane.variation.contract
        acquisition_options = tuple(
            value
            for value in contract.options
            if finite_variation_source_id(value) == "numerical_acquisition"
        )
        ranks = {
            value.option_id: int(
                dict(value.metadata)[VARIATION_SOURCE_RANK_METADATA_KEY]
            )
            for value in acquisition_options
        }
        assert len(ranks) == 4
        required = required_source_evaluation_option_ids(contract)
        assert required_ranked_source_evaluation_option_ids(contract) == required
        required_ranks = sorted(ranks[value] for value in required)
        assert required_ranks == sorted(ranks.values())[:2]
        all_required_ranks.extend(required_ranks)
    assert sorted(all_required_ranks) == [1, 2, 3, 4]


def test_envelope_atomically_registers_workload_neutral_certification_contexts():
    envelope, request = _scenario()
    registry = AcquisitionCertifiedSlateContextRegistry()
    expanded = ProtectedFiniteAcquisitionVariationEnvelope(
        objectives=envelope.objectives,
        space=envelope.space,
        acquisition=envelope.acquisition,
        phenotype_identity=envelope.phenotype_identity,
        acquisition_certification_context_sink=registry,
        pool_size=envelope.pool_size,
        protected_batch_size=8,
        source_minimum_per_lane=2,
        seed=envelope.seed,
    )

    result = expanded.enrich(request)

    assert registry.to_record()["registered_context_count"] == 2
    for lane in result.lanes:
        context = registry.context_for(lane.variation.contract.identity_sha256)
        assert len(context.reference_option_ids) == 4
        assert {value.candidate_id for value in context.candidates} == {
            value.option_id for value in lane.variation.contract.options
        }
        assert all(
            finite_variation_source_id(
                lane.variation.contract.resolve(option_id)
            )
            == "numerical_acquisition"
            for option_id in context.reference_option_ids
        )
        assert finite_variation_candidate_pool_required_option_ids(
            lane.variation.contract
        ) == context.reference_option_ids
        assert context.cutoff_index == request.state.unique_evaluations
        assert context.observations
    evidence = thaw_json(result.evidence)
    assert evidence["acquisition_certification_contexts_registered"] is True
    assert len(evidence["acquisition_certification_context_sha256s"]) == 2


def test_envelope_rejects_a_protected_batch_that_cannot_cover_every_lane():
    envelope, request = _scenario()
    undercovered = ProtectedFiniteAcquisitionVariationEnvelope(
        objectives=envelope.objectives,
        space=envelope.space,
        acquisition=envelope.acquisition,
        phenotype_identity=envelope.phenotype_identity,
        pool_size=16,
        protected_batch_size=1,
        seed=7,
    )

    with pytest.raises(ValueError, match="cover every active parent lane"):
        undercovered.enrich(request)
