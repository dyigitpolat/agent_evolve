from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.infrastructure.artifacts.in_memory import InMemoryArtifactStore
from examples.benchmarks.timeloop_codesign.v2 import DEFAULT_CANDIDATE
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (
    TimeloopV2DetailedEvaluationAdapter,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (
    frozen_network_panel,
)
from examples.benchmarks.timeloop_codesign.v2.problem_def import (
    TimeloopV2CoDesignProblem,
)
from examples.benchmarks.timeloop_codesign.v2.residual_portfolio_adapter import (
    TimeloopV2ResidualPhenotypeProjection,
    TimeloopV2SelectedMaterializedActionEvaluator,
    timeloop_v2_residual_phenotype_projection_definition_sha256,
)


class _MustNotRunEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, config: object):
        del config
        self.calls += 1
        raise AssertionError("static infeasibility must bypass the simulator")


def _problem(tmp_path: Path) -> tuple[TimeloopV2CoDesignProblem, _MustNotRunEvaluator]:
    evaluator = _MustNotRunEvaluator()
    problem = TimeloopV2CoDesignProblem(
        TimeloopV2Settings(output_root=tmp_path / "unused"),
        frozen_network_panel("resnet50"),
        evaluator=evaluator,
    )
    return problem, evaluator


def _static_infeasible_configuration() -> dict[str, object]:
    return {
        **DEFAULT_CANDIDATE,
        "pe_mesh_x": 4,
        "policy_cluster_1": {
            **DEFAULT_CANDIDATE["policy_cluster_1"],
            "primary_spatial_axis": "Q",
            "spatial_utilization": "full",
        },
    }


def test_projection_binds_compiled_plan_and_panel(tmp_path: Path) -> None:
    problem, _ = _problem(tmp_path)
    projection = TimeloopV2ResidualPhenotypeProjection(problem)
    configuration = freeze_json(DEFAULT_CANDIDATE)

    assert projection.project(configuration) == projection.identify(
        configuration
    ).value_sha256
    assert projection.definition_sha256 == (
        timeloop_v2_residual_phenotype_projection_definition_sha256(problem)
    )
    assert len(projection.definition_sha256) == 64


def test_selected_static_infeasibility_is_an_authenticated_candidate(
    tmp_path: Path,
) -> None:
    problem, simulator = _problem(tmp_path)
    projection = TimeloopV2ResidualPhenotypeProjection(problem)
    detailed = TimeloopV2DetailedEvaluationAdapter.build(
        problem=problem,
        artifact_store=InMemoryArtifactStore(),
    )
    configuration = freeze_json(_static_infeasible_configuration())
    action = MaterializedActionDescriptor(
        context=MaterializedActionContext(
            campaign_scope_sha256="4" * 64,
            decision_index=1,
            phase=SearchPhase.BASIN_EXPANSION,
            remaining_decisions=2,
            remaining_evaluations=8,
            residual_frontier_cell="frontier.reachable",
            parent_position_cell="parent.frontier",
            archive_relation_cell="archive.novel",
            structural_signature_sha256="5" * 64,
            patch_compatibility_cell="compatible",
            forecast_calibration_cell="unseen",
            source_distance_bin=1,
            memory_dose_bin=0,
        ),
        configuration=configuration,
        phenotype_identity_sha256=projection.project(configuration),
        expert_id="local_residual",
        native_rank=1,
        parent_ids=(CandidateId("candidate_timeloop_parent"),),
        operator_id="finite_residual.radius_1",
        target_candidate_id=CandidateId("candidate_timeloop_target"),
        role_id="local_exploit",
        normalized_evaluation_cost=1.0,
    )
    observations: list[dict[str, object]] = []
    evaluator = TimeloopV2SelectedMaterializedActionEvaluator(
        detailed_adapter=detailed,
        phenotype_projection=projection,
        observation_sink=observations.append,
        proposal_sequence_start=7,
    )

    evaluated = asyncio.run(evaluator.evaluate(action))

    assert simulator.calls == 0
    assert evaluator.evaluation_count == 1
    assert evaluated.candidate.valid is False
    assert evaluated.candidate.objectives == ()
    assert evaluated.candidate.failure_message is not None
    assert evaluated.candidate.detailed_evaluation is not None
    assert evaluated.candidate.detailed_evaluation.success is False
    assert evaluated.candidate.occurrence.proposal_sequence == 8
    assert observations[0]["valid"] is False
    assert observations[0]["receipt_artifact_id"] is not None
    json.dumps(observations[0], allow_nan=False, sort_keys=True)
    with pytest.raises(ValueError, match="only once"):
        asyncio.run(evaluator.evaluate(action))
