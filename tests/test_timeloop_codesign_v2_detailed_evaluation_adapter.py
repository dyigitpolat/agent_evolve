"""Provider-free gates for Timeloop v2 detailed evaluator evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agent_evolve.agentic import DetailedEvaluationAdapter, FailureCategory, FailureCode
from agent_evolve.infrastructure.artifacts.in_memory import InMemoryArtifactStore
from agent_evolve.ports.artifact_store import read_json
from examples.benchmarks.timeloop_codesign.v2 import DEFAULT_CANDIDATE
from examples.benchmarks.timeloop_codesign.v2.container_runner import (
    EVALUATOR_ID,
    MAX_CONSECUTIVE_INVALID_MAPPINGS,
    SEARCH_SIZE,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (
    TimeloopV2DetailedEvaluationAdapter,
    compose_timeloop_v2_detailed_benchmark,
    timeloop_v2_evaluator_identity,
    timeloop_v2_optimization_semantics,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (
    PINNED_IMAGE_ID,
    PINNED_IMAGE_REF,
    ContainerEvaluationResult,
    LayerContainerResult,
    MapperProtocol,
    ObjectiveValues,
    ProvenanceReceipt,
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2ContractError,
    TimeloopV2DockerEvaluator,
    TimeloopV2Evaluation,
    TimeloopV2InfeasibleEvaluation,
    TimeloopV2Settings,
    build_evaluation_bundle,
    canonical_evaluation_bundle_bytes,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (
    frozen_network_panel,
)
from examples.benchmarks.timeloop_codesign.v2.problem_def import (
    TimeloopV2CoDesignProblem,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _make_observation(
    root: Path,
    *,
    incomplete_medoid_ordinals: tuple[int, ...] = (),
) -> TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation:
    root.mkdir(parents=True)
    panel = frozen_network_panel("resnet50")
    bundle = build_evaluation_bundle(DEFAULT_CANDIDATE, panel)
    (root / "evaluation-bundle.json").write_bytes(
        canonical_evaluation_bundle_bytes(bundle) + b"\n"
    )
    runner = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmarks"
        / "timeloop_codesign"
        / "v2"
        / "container_runner.py"
    )
    layers: list[LayerContainerResult] = []
    for ordinal, manifest_sha256 in enumerate(bundle.layer_manifest_sha256):
        layer_root = root / "timeloop-output" / f"medoid-{ordinal}"
        layer_root.mkdir(parents=True)
        mapping = f"mapping for medoid {ordinal}\n".encode("ascii")
        processed = f"processed input for medoid {ordinal}\n".encode("ascii")
        (layer_root / "timeloop-mapper.map.yaml").write_bytes(mapping)
        (layer_root / "parsed-processed-input.yaml").write_bytes(processed)
        (layer_root / "output.log").write_text(
            f"termination evidence for medoid {ordinal}\n",
            encoding="utf-8",
        )
        complete = ordinal not in incomplete_medoid_ordinals
        layers.append(
            LayerContainerResult(
                medoid_ordinal=ordinal,
                layer_multiplicity=bundle.layer_manifests[ordinal].layer_multiplicity,
                layer_manifest_sha256=manifest_sha256,
                energy_joules=float(ordinal + 1),
                latency_seconds=float(ordinal + 2),
                area_square_meters=3.0,
                cycles=100 + ordinal,
                computes=200 + ordinal,
                requested_valid_mapping_count=SEARCH_SIZE,
                reported_valid_mapping_count=SEARCH_SIZE if complete else None,
                consecutive_invalid_mapping_count=(
                    0 if complete else MAX_CONSECUTIVE_INVALID_MAPPINGS
                ),
                mapping_budget_complete=complete,
                termination_reason=(
                    "valid_mapping_target"
                    if complete
                    else "consecutive_invalid_limit"
                ),
                elapsed_s=1.0,
                mapping_sha256=_sha_bytes(mapping),
                processed_input_sha256=_sha_bytes(processed),
                output_subdirectory=f"medoid-{ordinal}",
                front_end_projection_exact=True,
            )
        )
    manifest = ContainerEvaluationResult(
        schema_version=1,
        evaluator_id=EVALUATOR_ID,
        candidate_sha256=bundle.candidate_sha256,
        compiled_plan_sha256=bundle.compiled_plan_sha256,
        panel_sha256=bundle.panel_sha256,
        objectives=ObjectiveValues(
            energy_joules=10.0,
            latency_seconds=20.0,
            area_square_meters=3.0,
        ),
        layers=tuple(layers),
        protocol=MapperProtocol(),
        provenance=ProvenanceReceipt(
            asset_sha256={},
            runner_sha256=_sha_file(runner),
            python_hash_seed="0",
            mapper_seed_law=(
                "pinned_binary_random_pruned_default_constructed_cpp_engine"
            ),
        ),
    )
    result_path = root / "result.json"
    result_path.write_bytes(
        json.dumps(
            manifest.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    status = "candidate_infeasible" if incomplete_medoid_ordinals else "passed"
    (root / "host_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "evaluator_id": EVALUATOR_ID,
                "status": status,
                "candidate_sha256": bundle.candidate_sha256,
                "compiled_plan_sha256": bundle.compiled_plan_sha256,
                "panel_sha256": bundle.panel_sha256,
                "result_sha256": _sha_file(result_path),
                "image_ref": PINNED_IMAGE_REF,
                "image_id": PINNED_IMAGE_ID,
                "runner_sha256": _sha_file(runner),
                "cpu_set": "8",
                "search_size": SEARCH_SIZE,
                "mapper_threads": 1,
                "max_consecutive_invalid_mappings": (
                    MAX_CONSECUTIVE_INVALID_MAPPINGS
                ),
                "evaluator_elapsed_s": 3.5,
                "queue_wait_s": 0.25,
                "incomplete_medoid_ordinals": list(incomplete_medoid_ordinals),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common = {
        "output_dir": root,
        "candidate_sha256": bundle.candidate_sha256,
        "compiled_plan_sha256": bundle.compiled_plan_sha256,
        "panel_sha256": bundle.panel_sha256,
        "evaluator_elapsed_s": 3.5,
        "queue_wait_s": 0.25,
        "layer_results": tuple(layers),
        "manifest": manifest,
    }
    if incomplete_medoid_ordinals:
        return TimeloopV2InfeasibleEvaluation(
            **common,
            incomplete_medoid_ordinals=incomplete_medoid_ordinals,
        )
    return TimeloopV2Evaluation(
        **common,
        objective_values={
            "energy_joules": 10.0,
            "latency_seconds": 20.0,
            "area_square_meters": 3.0,
        },
    )


class _FakeEvaluator:
    def __init__(
        self,
        observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation | Exception,
    ) -> None:
        self.observation = observation
        self.calls = 0

    def evaluate(self, config: object) -> TimeloopV2Evaluation:
        del config
        self.calls += 1
        if isinstance(self.observation, Exception):
            raise self.observation
        if type(self.observation) is TimeloopV2InfeasibleEvaluation:
            raise TimeloopV2CandidateInfeasibleError(self.observation)
        return self.observation


def _adapter(
    tmp_path: Path,
    observation: TimeloopV2Evaluation | TimeloopV2InfeasibleEvaluation | Exception,
    store: InMemoryArtifactStore,
) -> TimeloopV2DetailedEvaluationAdapter:
    settings = TimeloopV2Settings(output_root=tmp_path / "unused")
    problem = TimeloopV2CoDesignProblem(
        settings,
        frozen_network_panel("resnet50"),
        evaluator=_FakeEvaluator(observation),
    )
    return TimeloopV2DetailedEvaluationAdapter.build(
        problem=problem,
        artifact_store=store,
    )


def test_identity_is_portable_but_binds_observable_semantics(tmp_path: Path) -> None:
    panel = frozen_network_panel("resnet50")
    settings = TimeloopV2Settings(output_root=tmp_path / "first")
    identity = timeloop_v2_evaluator_identity(settings, panel)

    assert timeloop_v2_evaluator_identity(
        replace(settings, output_root=tmp_path / "relocated", cpu_set="9"),
        panel,
    ) == identity
    assert timeloop_v2_evaluator_identity(
        replace(settings, timeout_s=181.0),
        panel,
    ) != identity


def test_success_preserves_complete_evidence_and_exact_objectives(
    tmp_path: Path,
) -> None:
    observation = _make_observation(tmp_path / "success")
    assert type(observation) is TimeloopV2Evaluation
    store = InMemoryArtifactStore()
    adapter = _adapter(tmp_path, observation, store)
    assert isinstance(adapter, DetailedEvaluationAdapter)

    payload = adapter.evaluate_evidence(dict(DEFAULT_CANDIDATE))

    assert payload.failure is None
    assert payload.objectives == (
        ("energy_joules", 10.0),
        ("latency_seconds", 20.0),
        ("area_square_meters", 3.0),
    )
    assert payload.active_wall_seconds == 3.5
    assert payload.resource_queue_wall_seconds == 0.25
    assert payload.receipt is not None
    receipt = read_json(store, payload.receipt.artifact_id)
    assert receipt["status"] == "passed"
    assert receipt["configuration"] == DEFAULT_CANDIDATE
    assert receipt["evaluation"]["candidate_sha256"] == (
        observation.candidate_sha256
    )
    assert len(receipt["source_artifacts"]) == 12
    for item in receipt["source_artifacts"]:
        assert store.stat(type(payload.receipt.artifact_id)(item["artifact"]["artifact_id"]))


def test_authenticated_mapper_exhaustion_is_receipted_candidate_failure(
    tmp_path: Path,
) -> None:
    observation = _make_observation(
        tmp_path / "infeasible",
        incomplete_medoid_ordinals=(1,),
    )
    assert type(observation) is TimeloopV2InfeasibleEvaluation
    store = InMemoryArtifactStore()
    payload = _adapter(tmp_path, observation, store).evaluate_evidence(
        dict(DEFAULT_CANDIDATE)
    )

    assert payload.failure is not None
    assert payload.failure.category is FailureCategory.CANDIDATE
    assert payload.failure.code is FailureCode.EVALUATOR_DECLARED_INFEASIBLE
    assert payload.objectives == ()
    assert payload.receipt is not None
    receipt = read_json(store, payload.receipt.artifact_id)
    assert receipt["status"] == "candidate_infeasible"
    assert receipt["evaluation"]["host_receipt"][
        "incomplete_medoid_ordinals"
    ] == [1]
    assert len(receipt["source_artifacts"]) == 12


def test_static_empty_mapspace_is_receipted_without_calling_evaluator(
    tmp_path: Path,
) -> None:
    candidate = {
        **DEFAULT_CANDIDATE,
        "pe_mesh_x": 4,
        "policy_cluster_1": {
            **DEFAULT_CANDIDATE["policy_cluster_1"],
            "primary_spatial_axis": "Q",
            "spatial_utilization": "full",
        },
    }
    store = InMemoryArtifactStore()
    observation = _make_observation(tmp_path / "unused-observation")
    adapter = _adapter(tmp_path, observation, store)
    payload = adapter.evaluate_evidence(candidate)

    assert payload.failure is not None
    assert payload.failure.category is FailureCategory.CANDIDATE
    assert payload.failure.code is FailureCode.EVALUATOR_DECLARED_INFEASIBLE
    assert payload.objectives == ()
    assert payload.active_wall_seconds == 0.0
    assert payload.resource_queue_wall_seconds == 0.0
    assert adapter.problem.evaluator.calls == 0
    assert payload.receipt is not None
    receipt = read_json(store, payload.receipt.artifact_id)
    assert receipt["status"] == "candidate_infeasible_static_mapspace"
    assert receipt["evaluation"]["native_simulator_invoked"] is False
    assert receipt["evaluation"]["static_mapspace_witnesses"] == [
        {
            "medoid_ordinal": 1,
            "primary_axis": "Q",
            "axis_extent": 14,
            "minimum_parallelism": 4,
            "maximum_parallelism": 4,
            "admissible_spatial_factors": [],
        }
    ]
    assert len(receipt["source_artifacts"]) == 1


def test_contract_failure_and_schema_failure_remain_distinguishable(
    tmp_path: Path,
) -> None:
    store = InMemoryArtifactStore()
    adapter = _adapter(
        tmp_path,
        TimeloopV2ContractError("pinned result drift"),
        store,
    )
    contract = adapter.evaluate_evidence(dict(DEFAULT_CANDIDATE))
    assert contract.failure is not None
    assert contract.failure.category is FailureCategory.SYSTEM
    assert contract.failure.code is FailureCode.EVALUATOR_CONTRACT_VIOLATION
    assert contract.receipt is not None

    schema = adapter.evaluate_evidence({"pe_mesh_x": 7})
    assert schema.failure is not None
    assert schema.failure.category is FailureCategory.CANDIDATE
    assert schema.failure.code is FailureCode.SCHEMA_INVALID
    assert schema.receipt is None
    assert adapter.problem.evaluator.calls == 1


def test_docker_validator_classifies_only_hash_valid_mapper_exhaustion(
    tmp_path: Path,
) -> None:
    observation = _make_observation(
        tmp_path / "docker-validation",
        incomplete_medoid_ordinals=(2,),
    )
    assert type(observation) is TimeloopV2InfeasibleEvaluation
    settings = TimeloopV2Settings(output_root=tmp_path / "unused")
    evaluator = TimeloopV2DockerEvaluator(
        settings,
        frozen_network_panel("resnet50"),
    )
    bundle = build_evaluation_bundle(
        DEFAULT_CANDIDATE,
        frozen_network_panel("resnet50"),
    )

    with pytest.raises(TimeloopV2CandidateInfeasibleError) as captured:
        evaluator._validate_result(
            bundle,
            observation.output_dir,
            observation.manifest,
            evaluator_elapsed_s=3.5,
            queue_wait_s=0.25,
        )

    assert captured.value.observation.incomplete_medoid_ordinals == (2,)


def test_public_composition_binds_detailed_evidence_and_pareto_relation(
    tmp_path: Path,
) -> None:
    observation = _make_observation(tmp_path / "composition")
    assert type(observation) is TimeloopV2Evaluation
    benchmark = compose_timeloop_v2_detailed_benchmark(
        TimeloopV2Settings(output_root=tmp_path / "unused"),
        frozen_network_panel("resnet50"),
        artifact_store=InMemoryArtifactStore(),
        evaluator=_FakeEvaluator(observation),
    )

    benchmark.validate_binding()
    assert type(benchmark.detailed_evaluator) is TimeloopV2DetailedEvaluationAdapter
    assert benchmark.outcome_relation is not None
    assert benchmark.outcome_relation.policy_id == "objective_pareto"
    assert len(benchmark.finite_variation_catalogs) == 1

    semantics = benchmark.optimization_semantics
    assert semantics == timeloop_v2_optimization_semantics(benchmark.problem)
    assert semantics is not None
    assert semantics.semantics_id == "timeloop_v2_raw_network_pareto"
    assert semantics.outcome_ordering.kind.value == "pareto"
    assert tuple(value.metric_id for value in semantics.metrics) == (
        "objective:area_square_meters",
        "objective:energy_joules",
        "objective:latency_seconds",
    )
    assert all(value.sense.value == "minimize" for value in semantics.metrics)
