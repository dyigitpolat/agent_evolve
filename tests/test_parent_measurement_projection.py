from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
    EvaluatorIdentity,
)
from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.application.parent_measurement import (
    attach_parent_measurement_to_context,
    bind_parent_measurement,
    create_parent_measurement_projection,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.core.problem import ObjectiveSpec
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
from agent_evolve.policies.objective_resolution.fixed_grid import (
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.policies.selection.phenotype_recourse import PhenotypeIdentity
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionRequest,
    objective_resolution_policy_metadata,
    resolve_objectives,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    OBJECTIVE_NAME as AIRFOIL_OBJECTIVE,
    VIOLATION_NAME as AIRFOIL_VIOLATION,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _objective_semantics(
    workload_id: str,
    objectives: tuple[ObjectiveSpec, ...],
) -> OptimizationSemantics:
    metrics = tuple(
        MetricSemantics(
            metric_id=f"objective:{objective.name}",
            name=objective.name,
            role=MetricRole.OBJECTIVE,
            sense=(
                MetricSense.MINIMIZE
                if objective.goal == "min"
                else MetricSense.MAXIMIZE
            ),
            definition=f"{workload_id} measured {objective.name}.",
            aggregation="One evaluator-produced scalar.",
            witness_interpretation="Follow the declared optimization sense.",
            tolerance=0.0,
        )
        for objective in objectives
    )
    return OptimizationSemantics(
        semantics_id=f"{workload_id}_parent_measurement",
        semantics_version=1,
        metrics=metrics,
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=tuple(metric.metric_id for metric in metrics),
            description="Use the declared Pareto objective order.",
            equivalence="Decision values agree exactly.",
            policy_id="objective_pareto",
            policy_version=1,
            definition_sha256=_sha(f"{workload_id}-objective-pareto"),
        ),
    )


def _variation(
    *,
    workload_id: str,
    benchmark_sha256: str,
    parent: FrozenJsonObject,
) -> ParentVariationBinding:
    parent_sha256 = typed_json_sha256(parent)
    child_record = thaw_json(parent)
    child_record["step"] = int(child_record["step"]) + 1
    child = _object(child_record)
    contract = FiniteVariationContract(
        catalog_id=f"{workload_id}_finite",
        catalog_version=1,
        catalog_definition_sha256=_sha(f"{workload_id}-catalog"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id=f"{workload_id}.step",
                parent_configuration_sha256=parent_sha256,
                child_configuration=child,
                family="step",
                description="Apply one sealed test step.",
            ),
        ),
    )
    return ParentVariationBinding(
        benchmark_sha256=benchmark_sha256,
        parent_configuration_sha256=parent_sha256,
        known_phenotype_sha256s=(),
        contract=contract,
    )


def _candidate(
    *,
    workload_id: str,
    configuration: FrozenJsonObject,
    raw_objectives: tuple[tuple[str, float], ...],
    violations: tuple[tuple[str, float], ...],
    evaluator: EvaluatorIdentity,
    objective_specs: tuple[ObjectiveSpec, ...],
    resolver: FixedGridObjectiveResolution | None = None,
) -> EvolutionCandidate:
    configuration_sha256 = typed_json_sha256(configuration)
    resolution = (
        None
        if resolver is None
        else resolve_objectives(
            resolver,
            ObjectiveResolutionRequest(
                configuration=configuration,
                objectives=objective_specs,
                raw_objectives=raw_objectives,
            ),
        )
    )
    detailed = DetailedEvaluation(
        phenotype=PhenotypeIdentity(
            policy_id="typed_configuration",
            policy_version=1,
            value_sha256=configuration_sha256,
        ),
        payload=DetailedEvaluationPayload(
            failure=None,
            objectives=raw_objectives,
            violations=violations,
            checks=(),
            receipt=None,
            evaluator=evaluator,
        ),
        timings=EvaluationTimings(total_wall_seconds=1.0),
    )
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_{workload_id}"),
            configuration_hash=configuration_sha256,
            configuration_artifact_hash=configuration_sha256,
            proposal_sequence=7,
        ),
        configuration=configuration,
        objectives=(
            raw_objectives if resolution is None else resolution.decision_objectives
        ),
        valid=True,
        generation=2,
        label=f"{workload_id}_selected_parent",
        detailed_evaluation=detailed,
        objective_resolution_receipt=resolution,
    )


def _workload_cases():
    boils_objectives = (
        ObjectiveSpec("total_levels", "min"),
        ObjectiveSpec("total_lut_count", "min"),
    )
    heat_objectives = (
        ObjectiveSpec(THERMAL_OBJECTIVE_NAME, "min"),
        ObjectiveSpec(MATERIAL_OBJECTIVE_NAME, "min"),
    )
    heat_resolver = FixedGridObjectiveResolution(
        metric_specs=tuple(
            sorted(
                (
                    FixedGridMetricSpec(
                        metric_id=value.name,
                        decimal_origin=Decimal("0"),
                        decimal_quantum=Decimal("0.000000000001"),
                        rounding_law=FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN,
                    )
                    for value in heat_objectives
                ),
                key=lambda value: value.metric_id,
            )
        )
    )
    return (
        (
            "boils",
            boils_objectives,
            _objective_semantics("boils", boils_objectives),
            (("total_levels", 70.0), ("total_lut_count", 8000.0)),
            (),
            None,
        ),
        (
            "heat2d",
            heat_objectives,
            _objective_semantics("heat2d", heat_objectives),
            (
                (THERMAL_OBJECTIVE_NAME, 0.0012345678904),
                (MATERIAL_OBJECTIVE_NAME, 0.5000000000004),
            ),
            (),
            heat_resolver,
        ),
        (
            "airfoil",
            (ObjectiveSpec(AIRFOIL_OBJECTIVE, "min"),),
            AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
            ((AIRFOIL_OBJECTIVE, 0.031),),
            ((AIRFOIL_VIOLATION, 0.004),),
            None,
        ),
    )


@pytest.mark.parametrize(
    ("workload_id", "objective_specs", "semantics", "raw", "violations", "resolver"),
    _workload_cases(),
)
def test_parent_measurement_crosses_three_workloads_without_evaluation(
    workload_id,
    objective_specs,
    semantics,
    raw,
    violations,
    resolver,
) -> None:
    benchmark_sha256 = _sha(f"{workload_id}-benchmark")
    session_sha256 = _sha(f"{workload_id}-session")
    parent = _object({"step": 0, "workload": workload_id})
    evaluator = EvaluatorIdentity(
        evaluator_id=f"{workload_id}_evaluator",
        evaluator_version=1,
        evaluator_context_sha256=_sha(f"{workload_id}-evaluator-context"),
    )
    candidate = _candidate(
        workload_id=workload_id,
        configuration=parent,
        raw_objectives=raw,
        violations=violations,
        evaluator=evaluator,
        objective_specs=objective_specs,
        resolver=resolver,
    )
    variation = _variation(
        workload_id=workload_id,
        benchmark_sha256=benchmark_sha256,
        parent=parent,
    )
    metric_projection = DecisionMetricProjection.from_optimization_semantics(semantics)
    projection = create_parent_measurement_projection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=session_sha256,
        decision_metrics=metric_projection,
        evaluator=evaluator,
        objective_resolution_identity=(
            None if resolver is None else objective_resolution_policy_metadata(resolver)
        ),
    )
    binding = bind_parent_measurement(
        candidate=candidate,
        variation=variation,
        projection=projection,
    )

    context = _object({"schema_version": 1, "workload_id": workload_id})
    assert attach_parent_measurement_to_context(context, None) is context
    attached = attach_parent_measurement_to_context(context, binding)
    record = thaw_json(attached)["parent_measurement"]
    assert record["binding_sha256"] == binding.binding_sha256
    assert record["candidate"]["configuration_sha256"] == typed_json_sha256(parent)
    assert record["projection"]["evaluator"] == evaluator.to_record()
    assert record["interpretation"]["current_wave_outcomes_included"] is False
    assert [value["metric_id"] for value in record["decision_metrics"]] == list(
        metric_projection.metric_ids
    )
    assert [value["metric_id"] for value in record["raw_scientific_metrics"]] == list(
        metric_projection.metric_ids
    )
    if resolver is not None:
        assert binding.objective_resolution_receipt_sha256 is not None
        assert tuple(value.value for value in binding.raw_scientific_metrics) != tuple(
            value.value for value in binding.decision_metrics
        )


def test_parent_measurement_rejects_configuration_evaluator_resolution_and_schema_mismatch() -> (
    None
):
    workload_id, objectives, semantics, raw, violations, resolver = _workload_cases()[1]
    assert resolver is not None
    benchmark_sha256 = _sha("heat2d-benchmark")
    parent = _object({"step": 0, "workload": workload_id})
    evaluator = EvaluatorIdentity(
        evaluator_id="heat2d_evaluator",
        evaluator_version=1,
        evaluator_context_sha256=_sha("heat2d-evaluator-context"),
    )
    candidate = _candidate(
        workload_id=workload_id,
        configuration=parent,
        raw_objectives=raw,
        violations=violations,
        evaluator=evaluator,
        objective_specs=objectives,
        resolver=resolver,
    )
    projection = create_parent_measurement_projection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=_sha("heat2d-session"),
        decision_metrics=DecisionMetricProjection.from_optimization_semantics(
            semantics
        ),
        evaluator=evaluator,
        objective_resolution_identity=objective_resolution_policy_metadata(resolver),
    )
    foreign_parent = _object({"step": 9, "workload": workload_id})
    with pytest.raises(ValueError, match="variation parent"):
        bind_parent_measurement(
            candidate=candidate,
            variation=_variation(
                workload_id=workload_id,
                benchmark_sha256=benchmark_sha256,
                parent=foreign_parent,
            ),
            projection=projection,
        )

    wrong_evaluator = create_parent_measurement_projection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=_sha("heat2d-session"),
        decision_metrics=projection.decision_metrics,
        evaluator=EvaluatorIdentity(
            evaluator_id="heat2d_evaluator",
            evaluator_version=2,
            evaluator_context_sha256=evaluator.evaluator_context_sha256,
        ),
        objective_resolution_identity=objective_resolution_policy_metadata(resolver),
    )
    with pytest.raises(ValueError, match="evaluator identity"):
        bind_parent_measurement(
            candidate=candidate,
            variation=_variation(
                workload_id=workload_id,
                benchmark_sha256=benchmark_sha256,
                parent=parent,
            ),
            projection=wrong_evaluator,
        )

    wrong_resolution = create_parent_measurement_projection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=_sha("heat2d-session"),
        decision_metrics=projection.decision_metrics,
        evaluator=evaluator,
        objective_resolution_identity=("wrong_resolution", 1, _sha("wrong")),
    )
    with pytest.raises(ValueError, match="objective-resolution identity"):
        bind_parent_measurement(
            candidate=candidate,
            variation=_variation(
                workload_id=workload_id,
                benchmark_sha256=benchmark_sha256,
                parent=parent,
            ),
            projection=wrong_resolution,
        )

    airfoil_projection = create_parent_measurement_projection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=_sha("heat2d-session"),
        decision_metrics=DecisionMetricProjection.from_optimization_semantics(
            AIRFOIL_V7_OPTIMIZATION_SEMANTICS
        ),
        evaluator=evaluator,
        objective_resolution_identity=objective_resolution_policy_metadata(resolver),
    )
    with pytest.raises(ValueError, match="objective is absent"):
        bind_parent_measurement(
            candidate=candidate,
            variation=_variation(
                workload_id=workload_id,
                benchmark_sha256=benchmark_sha256,
                parent=parent,
            ),
            projection=airfoil_projection,
        )

    context = _object({"parent_measurement": {"forged": True}})
    valid = bind_parent_measurement(
        candidate=candidate,
        variation=_variation(
            workload_id=workload_id,
            benchmark_sha256=benchmark_sha256,
            parent=parent,
        ),
        projection=projection,
    )
    with pytest.raises(ValueError, match="reserved parent_measurement"):
        attach_parent_measurement_to_context(context, valid)
