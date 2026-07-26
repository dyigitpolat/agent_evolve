"""Seal completed parent measurements and attach them to selector context."""

from __future__ import annotations

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.decision_metric_projection import (
    project_candidate_decision_metrics,
)
from agent_evolve.application.detailed_evaluation import EvaluatorIdentity
from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.core.optimization_semantics import MetricRole
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.ports.objective_resolution import (
    EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.parent_measurement import (
    ParentCandidateMeasurementIdentity,
    ParentDecisionMetricValue,
    ParentMeasurementBinding,
    ParentMeasurementProjection,
    ParentRawScientificMetricValue,
)


PARENT_MEASUREMENT_CONTEXT_KEY = "parent_measurement"


def create_parent_measurement_projection(
    *,
    benchmark_sha256: str,
    session_sha256: str,
    decision_metrics: DecisionMetricProjection,
    evaluator: EvaluatorIdentity,
    objective_resolution_identity: tuple[str, int, str] | None,
) -> ParentMeasurementProjection:
    """Bind a decision schema to exact campaign evaluation authorities.

    ``None`` denotes the engine's explicit compatibility law in which raw
    evaluator objectives are consumed unchanged.  It never denotes an unknown
    or unauthenticated resolution policy.
    """

    if type(evaluator) is not EvaluatorIdentity:
        raise TypeError("evaluator must be an exact EvaluatorIdentity")
    EvaluatorIdentity.__post_init__(evaluator)
    resolution = (
        (
            EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
            EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
            EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
        )
        if objective_resolution_identity is None
        else objective_resolution_identity
    )
    if (
        type(resolution) is not tuple
        or len(resolution) != 3
        or type(resolution[0]) is not str
        or type(resolution[1]) is not int
        or type(resolution[2]) is not str
    ):
        raise TypeError("objective_resolution_identity must be an exact identity")
    return ParentMeasurementProjection(
        benchmark_sha256=benchmark_sha256,
        session_sha256=session_sha256,
        decision_metrics=decision_metrics,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        evaluator_context_sha256=evaluator.evaluator_context_sha256,
        objective_resolution_policy_id=resolution[0],
        objective_resolution_policy_version=resolution[1],
        objective_resolution_definition_sha256=resolution[2],
    )


def _candidate_identity(
    candidate: EvolutionCandidate,
) -> ParentCandidateMeasurementIdentity:
    occurrence = candidate.occurrence
    return ParentCandidateMeasurementIdentity(
        candidate_id=candidate.candidate_id.value,
        configuration_sha256=occurrence.configuration_hash,
        configuration_artifact_sha256=occurrence.configuration_artifact_hash,
        proposal_sequence=occurrence.proposal_sequence,
        operator_invocation_id=(
            None
            if occurrence.operator_invocation_id is None
            else occurrence.operator_invocation_id.value
        ),
    )


def bind_parent_measurement(
    *,
    candidate: EvolutionCandidate,
    variation: ParentVariationBinding,
    projection: ParentMeasurementProjection,
) -> ParentMeasurementBinding:
    """Project one completed parent and fail closed on every authority mismatch."""

    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be an exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(candidate)
    if type(variation) is not ParentVariationBinding:
        raise TypeError("variation must be an exact ParentVariationBinding")
    ParentVariationBinding.__post_init__(variation)
    if type(projection) is not ParentMeasurementProjection:
        raise TypeError("projection must be exact ParentMeasurementProjection")
    projection.__post_init__()
    if not candidate.valid:
        raise ValueError("parent measurement requires a valid evaluated candidate")
    if candidate.occurrence.configuration_hash != variation.parent_configuration_sha256:
        raise ValueError("parent measurement candidate differs from variation parent")
    if variation.benchmark_sha256 != projection.benchmark_sha256:
        raise ValueError("parent measurement projection names a foreign benchmark")

    detailed = candidate.detailed_evaluation
    if detailed is None or not detailed.success:
        raise ValueError(
            "authenticated parent measurement requires successful detailed evidence"
        )
    evaluator = detailed.payload.evaluator
    if (
        evaluator.evaluator_id,
        evaluator.evaluator_version,
        evaluator.evaluator_context_sha256,
    ) != projection.evaluator_identity:
        raise ValueError(
            "parent measurement evaluator identity differs from projection"
        )

    resolution = candidate.objective_resolution_receipt
    if resolution is None:
        observed_resolution = (
            EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
            EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
            EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
        )
        resolution_receipt_sha256 = None
    else:
        resolution.revalidate()
        observed_resolution = resolution.policy_identity
        resolution_receipt_sha256 = resolution.receipt_sha256
    if observed_resolution != projection.objective_resolution_identity:
        raise ValueError(
            "parent measurement objective-resolution identity differs from projection"
        )

    projected = project_candidate_decision_metrics(
        candidate,
        projection.decision_metrics,
    )
    raw_objectives = candidate.raw_objective_map
    detailed_values = dict(detailed.violations)
    raw_values: list[ParentRawScientificMetricValue] = []
    for metric in projection.decision_metrics.metrics:
        source = (
            raw_objectives if metric.role is MetricRole.OBJECTIVE else detailed_values
        )
        if metric.value_name not in source:
            raise ValueError(
                "raw scientific metric is absent from evaluator evidence: "
                f"{metric.value_name}"
            )
        raw_values.append(
            ParentRawScientificMetricValue(
                metric_id=metric.metric_id,
                semantic_metric_id=metric.semantic_metric_id,
                value_name=metric.value_name,
                role=metric.role,
                value=source[metric.value_name],
            )
        )
    return ParentMeasurementBinding(
        projection=projection,
        candidate=_candidate_identity(candidate),
        detailed_evaluation_sha256=detailed.evidence_sha256,
        objective_resolution_receipt_sha256=resolution_receipt_sha256,
        raw_scientific_metrics=tuple(raw_values),
        decision_metrics=tuple(
            ParentDecisionMetricValue(metric_id, value)
            for metric_id, value in projected.values
        ),
    )


def attach_parent_measurement_to_context(
    context: FrozenJsonObject,
    binding: ParentMeasurementBinding | None,
) -> FrozenJsonObject:
    """Attach one reserved context block, preserving exact legacy bytes if absent."""

    if type(context) is not FrozenJsonObject or freeze_json(context) is not context:
        raise TypeError("context must be an exact frozen typed-JSON object")
    if binding is None:
        return context
    if type(binding) is not ParentMeasurementBinding:
        raise TypeError("binding must be exact ParentMeasurementBinding or None")
    binding.__post_init__()
    record = thaw_json(context)
    if type(record) is not dict:  # pragma: no cover - exact root above.
        raise AssertionError("frozen context did not thaw to an object")
    if PARENT_MEASUREMENT_CONTEXT_KEY in record:
        raise ValueError(
            "workload context already uses reserved parent_measurement key"
        )
    record[PARENT_MEASUREMENT_CONTEXT_KEY] = binding.to_record()
    frozen = freeze_json(record)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed record.
        raise AssertionError("parent measurement context did not freeze to an object")
    return frozen


__all__ = [
    "PARENT_MEASUREMENT_CONTEXT_KEY",
    "attach_parent_measurement_to_context",
    "bind_parent_measurement",
    "create_parent_measurement_projection",
]
