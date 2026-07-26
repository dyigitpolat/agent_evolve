"""Generic, provider-free conformance for optional workload prompt knowledge."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve import (
    WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY,
    WorkloadPromptArm,
    WorkloadPromptExtension,
    WorkloadPromptProvenance,
    WorkloadPromptSourceKind,
)
from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.campaign_workload import (
    compose_boils_campaign_workload,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _extension() -> WorkloadPromptExtension:
    return WorkloadPromptExtension(
        extension_id="boils.public_semantics.v1",
        extension_version=1,
        semantic_payload=_object(
            {
                "domain_context": {
                    "candidate_kind": "logic-synthesis action sequence"
                },
                "metric_semantics": {
                    "metrics": [
                        {"metric_id": "total_lut_count", "goal": "minimize"},
                        {"metric_id": "total_levels", "goal": "minimize"},
                    ]
                },
                "constraints_and_invariants": {
                    "rules": ["Keep the published sequence length fixed."]
                },
            }
        ),
        matched_control_payload=_object(
            {
                "domain_context": {"candidate_kind": "opaque candidate sequence"},
                "metric_semantics": {
                    "metrics": [
                        {"metric_id": "metric_1", "goal": "declared_goal"},
                        {"metric_id": "metric_2", "goal": "declared_goal"},
                    ]
                },
                "constraints_and_invariants": {
                    "rules": ["Respect the published candidate shape."]
                },
            }
        ),
        provenance=WorkloadPromptProvenance(
            source_kind=WorkloadPromptSourceKind.BENCHMARK_SPECIFICATION,
            source_artifact_sha256s=(
                _sha("frozen-public-boils-benchmark-specification"),
            ),
        ),
    )


def _boils_config():
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=((0,),),
        per_circuit_timeout_s=60.0,
    )
    problem = BoilsAbcProblem(settings)
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    config = compose_boils_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "provider_calls": 0,
                "abc_executions": 0,
            }
        ),
        resource_lease_receipt=_object(
            {"resource": "one_pinned_cpu_affinity", "active": True}
        ),
        evaluator_concurrency_cap=1,
    )
    return problem, config


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"frozen_before_campaign": False}, "frozen before campaign"),
        ({"evaluator_outcomes_accessed": True}, "evaluator outcomes"),
        ({"campaign_traces_accessed": True}, "campaign traces"),
        ({"tuned_on_benchmark_outcomes": True}, "outcome-tuned"),
    ),
)
def test_provenance_rejects_post_outcome_prompt_knowledge(
    overrides: dict[str, bool],
    message: str,
) -> None:
    arguments = {
        "source_kind": WorkloadPromptSourceKind.PUBLIC_REFERENCE,
        "source_artifact_sha256s": (_sha("reference"),),
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        WorkloadPromptProvenance(**arguments)


def test_extension_exposes_prospective_schema_semantic_and_control_arms() -> None:
    extension = _extension()

    assert extension.view(WorkloadPromptArm.SCHEMA_ONLY) is None
    semantic = extension.view(WorkloadPromptArm.SEMANTIC)
    control = extension.view(WorkloadPromptArm.MATCHED_CONTROL)
    assert semantic is not None
    assert control is not None
    assert semantic.arm is WorkloadPromptArm.SEMANTIC
    assert control.arm is WorkloadPromptArm.MATCHED_CONTROL
    assert semantic.extension_definition_sha256 == extension.definition_sha256
    assert control.extension_definition_sha256 == extension.definition_sha256
    assert semantic.payload_sha256 != control.payload_sha256
    assert semantic.view_sha256 != control.view_sha256


def test_matched_control_must_preserve_recursive_payload_shape() -> None:
    provenance = WorkloadPromptProvenance(
        source_kind=WorkloadPromptSourceKind.PUBLIC_REFERENCE,
        source_artifact_sha256s=(_sha("public-reference"),),
    )
    with pytest.raises(ValueError, match="preserve.*shape"):
        WorkloadPromptExtension(
            extension_id="shape.check",
            extension_version=1,
            semantic_payload=_object(
                {"metric_semantics": {"metrics": ["one", "two"]}}
            ),
            matched_control_payload=_object(
                {"metric_semantics": {"metrics": ["one"]}}
            ),
            provenance=provenance,
        )


@pytest.mark.parametrize(
    "arm",
    (WorkloadPromptArm.SEMANTIC, WorkloadPromptArm.MATCHED_CONTROL),
)
def test_boils_campaign_authenticates_and_renders_optional_prompt_arm(
    arm: WorkloadPromptArm,
) -> None:
    problem, base = _boils_config()
    view = _extension().view(arm)
    assert view is not None
    treated = replace(base, prompt_extension=view)

    assert base.prompt_extension is None
    assert WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY not in thaw_json(
        base.benchmark_record
    )
    assert treated.configuration_sha256 != base.configuration_sha256
    benchmark_record = thaw_json(treated.benchmark_record)
    assert benchmark_record[WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY] == (
        view.to_binding_record()
    )

    ports = treated.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("prompt-extension-protocol"),
            budget_sha256=_sha("prompt-extension-budget"),
            outer_seed=20260719,
            requested_evaluator_concurrency=1,
        )
    )
    seeds = ports.seeds.load(session)
    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    memory = ports.evidence.initialize_memory(session, seeds)
    context = thaw_json(
        ports.evidence.context(session, parent, variation, memory)
    )

    assert context[WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY] == (
        view.to_prompt_record()
    )
    assert context[WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY]["arm"] == arm.value
    assert context[WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY]["evidence_status"] == (
        "static_preoptimization_context_not_measured_search_evidence"
    )
    assert problem._evaluator is None


def test_schema_only_arm_is_byte_compatible_with_default_campaign() -> None:
    _, base = _boils_config()
    schema_only = replace(
        base,
        prompt_extension=_extension().view(WorkloadPromptArm.SCHEMA_ONLY),
    )

    assert schema_only.to_record() == base.to_record()
    assert schema_only.configuration_sha256 == base.configuration_sha256
