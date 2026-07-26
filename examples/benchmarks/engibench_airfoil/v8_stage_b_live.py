"""Airfoil Stage-B matched model-versus-uniform development composition.

This module is deliberately a thin benchmark/live boundary.  The generic
matched finite-action planner owns the A/U chronology, support equality,
prospective uniform draw, model decision sealing, and collision policy.  This
adapter supplies only Airfoil's learned card, held-out parent, K=8 trim support,
evaluation semantics, and the already authenticated DeepSeek/OpenRouter route.

Importing or composing this module never reads credentials, contacts a model,
or evaluates an Airfoil.  The credentialed transport is created lazily on the
first proposal, after the optimizer has evaluated its seed.  This is a
development block: its learned v2 card and parent come from the disclosed G3
run, so its result is useful for loop debugging but is not fresh paper evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticGenerator,
    AgenticOptimizerComposition,
    CandidateId,
    FiniteActionEvidenceBinding,
    FixedStructuredOutputBudgetPolicy,
    InsightDraft,
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
    LLMCallId,
    MatchedFiniteActionBlockFactory,
    MetricEffectDirection,
    MetricEffectPrediction,
    OperatorKind,
    OperatorInvocationId,
    OptimizerBudget,
    TaskKeyedUniformFiniteActionPolicy,
    TelemetryGatedAgenticGenerator,
    compose_agentic_optimizer,
    thaw_json,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareOpenRouterConfig,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    StructuredEvidencePublicationPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    EVALUATOR_CONCURRENCY,
    MAX_IN_FLIGHT,
    MAX_OUTPUT_TOKENS,
    MAX_PENDING,
    MODEL_ALIAS,
    AirfoilG3LiveSinks,
    LiveGeneratorFactory,
    OwnedAgenticGenerator,
    build_openrouter_config,
    build_telemetry_policy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    AIRFOIL_G3_ABSOLUTE_REWARD,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AirfoilG3RuntimeInputs,
)
from examples.benchmarks.engibench_airfoil.v7_stage_b_action_set import (
    AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256,
    AirfoilTrimLocalSupportCompiler,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_hypothesis import (
    AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256,
    AirfoilV8ReflectionNativeTrimHypothesisCompiler,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
RESEARCH_ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
CANONICAL_G3_RUN_ID = "airfoil_v7_g3_20260715t1204z"
CANONICAL_G3_RUN_ROOT = (
    RESEARCH_ARTIFACT_ROOT / "experiment_logs" / "airfoil_g3" / CANONICAL_G3_RUN_ID
)
CANONICAL_G3_TRACE_PATH = CANONICAL_G3_RUN_ROOT / "execution_traces.jsonl"
CANONICAL_G3_TRACE_FILE_SHA256 = (
    "45e77289802e484dc90f958d38f879a1e3583aa0361da758ed04286e9053f6f8"
)

AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID = "insight_airfoil_twostage_cards_000002"
AIRFOIL_V8_STAGE_B_PREDECESSOR_VERSION = 1
AIRFOIL_V8_STAGE_B_CARD_VERSION = 2
AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256 = (
    "ecc66ffd283a70b675551e2ca8dcbd0dbd803135ccace8d4c06101e26494972d"
)
AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256 = (
    "7353690f460f208a4a27a318551fda4ae309526af3390b9c4fc06c651e6c9bdc"
)
AIRFOIL_V8_STAGE_B_REFLECTION_CALL_ID = "call_airfoil_g3_runtime_000006"
AIRFOIL_V8_STAGE_B_REFLECTION_RECEIPT_SHA256 = (
    "64de474f6548b20c2578a3926abde0b1d4258934a24698203fc338fdec05eafd"
)
AIRFOIL_V8_STAGE_B_CATALOG_ID = "airfoil_v7_trim"
AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY = 8
AIRFOIL_V8_STAGE_B_RUN_SEED = 2_026_071_508
AIRFOIL_V8_STAGE_B_PHASE = "airfoil_v8_stage_b_learned_card_development"
AIRFOIL_V8_STAGE_B_BUDGET = OptimizerBudget(
    max_unique_evaluations=3,
    max_logical_llm_calls=1,
    max_generations=1,
)
AIRFOIL_V8_STAGE_B_OUTPUT_BUDGET = FixedStructuredOutputBudgetPolicy(
    proposal_max_output_tokens=MAX_OUTPUT_TOKENS,
    reflection_max_output_tokens=MAX_OUTPUT_TOKENS,
)

_DEFINITION_DOMAIN = b"agent-evolve:airfoil-v8-stage-b-development:v1\x00"
_SCHEDULE_DOMAIN = b"agent-evolve:airfoil-v8-stage-b-uniform-schedule:v1\x00"
_TASK_DOMAIN = b"agent-evolve:airfoil-v8-stage-b-task:v1\x00"
_COMMIT_DOMAIN = b"agent-evolve:airfoil-v8-stage-b-pre-outcome:v1\x00"


class AirfoilV8StageBError(RuntimeError):
    """The learned-card replay or exact Stage-B composition drifted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


_BLOCK_DEFINITION = {
    "schema_version": 1,
    "status": "development_disclosed_parent_and_memory",
    "source_run_id": CANONICAL_G3_RUN_ID,
    "learned_card": {
        "insight_id": AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID,
        "version": AIRFOIL_V8_STAGE_B_CARD_VERSION,
        "content_sha256": AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
        "lineage_sha256": AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256,
    },
    "parent_role": "G3 P_H disclosed held-out parent",
    "catalog_id": AIRFOIL_V8_STAGE_B_CATALOG_ID,
    "support_cardinality": AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    "support_compiler_sha256": AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256,
    "hypothesis_compiler_sha256": (
        AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256
    ),
    "arms": ["A:model_learned_card", "U:prospective_uniform_same_support"],
    "a_u_alias_policy": "retain_without_resampling_and_use_evaluation_cache",
    "current_stage_b_outcome_access": False,
}
AIRFOIL_V8_STAGE_B_DEFINITION_SHA256 = _hash(
    _DEFINITION_DOMAIN,
    _BLOCK_DEFINITION,
)
AIRFOIL_V8_STAGE_B_SCHEDULE_SEED_SHA256 = _hash(
    _SCHEDULE_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V8_STAGE_B_DEFINITION_SHA256,
        "run_seed": AIRFOIL_V8_STAGE_B_RUN_SEED,
    },
)
AIRFOIL_V8_STAGE_B_TASK_SHA256 = _hash(
    _TASK_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V8_STAGE_B_DEFINITION_SHA256,
        "learned_card_content_sha256": AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
        "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
    },
)
AIRFOIL_V8_STAGE_B_PRE_OUTCOME_COMMIT_SHA256 = _hash(
    _COMMIT_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V8_STAGE_B_DEFINITION_SHA256,
        "task_sha256": AIRFOIL_V8_STAGE_B_TASK_SHA256,
        "uniform_schedule_seed_sha256": (
            AIRFOIL_V8_STAGE_B_SCHEDULE_SEED_SHA256
        ),
        "new_stage_b_outcomes_observed": False,
    },
)


def _exact_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise AirfoilV8StageBError(f"{name} must be an exact object")
    return value


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise AirfoilV8StageBError(f"{name} must be an exact list")
    return value


def _load_canonical_revision_event(path: Path) -> Mapping[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    if resolved == CANONICAL_G3_TRACE_PATH.resolve(strict=True) and (
        hashlib.sha256(payload).hexdigest() != CANONICAL_G3_TRACE_FILE_SHA256
    ):
        raise AirfoilV8StageBError("canonical G3 trace bytes changed")
    rows: list[Mapping[str, object]] = []
    try:
        for line in payload.splitlines():
            raw = json.loads(line)
            if (
                type(raw) is dict
                and raw.get("event_type") == "reflection_completed"
                and raw.get("call_id") == AIRFOIL_V8_STAGE_B_REFLECTION_CALL_ID
            ):
                rows.append(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilV8StageBError("canonical G3 trace is malformed") from exc
    if len(rows) != 1:
        raise AirfoilV8StageBError("canonical G3 revision event is not unique")
    event = rows[0]
    if (
        event.get("reflection_call_receipt_sha256")
        != AIRFOIL_V8_STAGE_B_REFLECTION_RECEIPT_SHA256
        or event.get("label") != "g3_postseal_curation"
    ):
        raise AirfoilV8StageBError("canonical G3 revision receipt changed")
    return event


def _revision_from_event(event: Mapping[str, object]) -> tuple[
    InsightDraft,
    InsightEvidenceLineage,
]:
    raw_insights = _exact_list(event.get("insights"), name="G3 revision insights")
    if len(raw_insights) != 1:
        raise AirfoilV8StageBError("G3 revision event must contain one insight")
    raw = _exact_mapping(raw_insights[0], name="G3 revision insight")
    raw_effects = _exact_list(raw.get("effect_predictions"), name="effect predictions")
    effects = tuple(
        MetricEffectPrediction(
            metric_id=str(_exact_mapping(item, name="effect prediction")["metric_id"]),
            direction=MetricEffectDirection(
                str(_exact_mapping(item, name="effect prediction")["direction"])
            ),
        )
        for item in raw_effects
    )
    draft = InsightDraft(
        claim=str(raw["claim"]),
        trigger=str(raw["trigger"]),
        mechanism=str(raw["mechanism"]),
        affected_paths=tuple(str(value) for value in raw["affected_paths"]),
        evidence_summary=str(raw["evidence_summary"]),
        confidence=float(raw["confidence"]),
        evidence_contrast_ids=tuple(
            str(value) for value in raw["evidence_contrast_ids"]
        ),
        effect_predictions=effects,
        recommended_option_families=tuple(
            str(value) for value in raw["recommended_option_families"]
        ),
        recommended_option_ids=tuple(
            str(value) for value in raw["recommended_option_ids"]
        ),
        action_template=str(raw["action_template"]),
        falsification_condition=str(raw["falsification_condition"]),
    )
    raw_lineage = _exact_mapping(raw.get("evidence_lineage"), name="G3 lineage")
    raw_bindings = _exact_list(
        raw_lineage.get("finite_action_bindings"),
        name="G3 finite action bindings",
    )
    bindings = tuple(
        FiniteActionEvidenceBinding(
            contrast_id=str(_exact_mapping(item, name="action binding")["contrast_id"]),
            option_id=str(_exact_mapping(item, name="action binding")["option_id"]),
            family=str(_exact_mapping(item, name="action binding")["family"]),
            option_identity_sha256=str(
                _exact_mapping(item, name="action binding")[
                    "option_identity_sha256"
                ]
            ),
            contract_identity_sha256=str(
                _exact_mapping(item, name="action binding")[
                    "contract_identity_sha256"
                ]
            ),
        )
        for item in raw_bindings
    )
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId(str(raw_lineage["reflection_call_id"])),
        source_operator_invocation_ids=tuple(
            OperatorInvocationId(str(value))
            for value in raw_lineage["source_operator_invocation_ids"]
        ),
        source_candidate_ids=tuple(
            CandidateId(str(value))
            for value in raw_lineage["source_candidate_ids"]
        ),
        available_contrast_ids=tuple(
            str(value) for value in raw_lineage["available_contrast_ids"]
        ),
        cited_contrast_ids=tuple(
            str(value) for value in raw_lineage["cited_contrast_ids"]
        ),
        finite_action_bindings=bindings,
    )
    if (
        raw.get("insight_id") != AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID
        or raw.get("version") != AIRFOIL_V8_STAGE_B_CARD_VERSION
        or raw.get("lifecycle_state") != InsightLifecycleState.QUARANTINED.value
        or raw.get("origin") != InsightOrigin.REFLECTION.value
        or raw.get("applicable_operator_kinds") != ["typed_mutation"]
        or draft.content_sha256 != AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256
        or lineage.identity_sha256 != AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256
        or raw_lineage != lineage.to_record()
    ):
        raise AirfoilV8StageBError("canonical learned-card identity changed")
    return draft, lineage


def replay_canonical_g3_adaptive_revision(
    memory: InsightMemoryBank,
    *,
    trace_path: Path = CANONICAL_G3_TRACE_PATH,
) -> InsightMemoryEntry:
    """Replay the exact G3 v2 revision into a bank already holding its v1."""

    if type(memory) is not InsightMemoryBank:
        raise TypeError("memory must be an exact InsightMemoryBank")
    predecessors = tuple(
        entry
        for entry in memory.entries
        if (
            entry.reference.insight_id.value == AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID
            and entry.reference.version == AIRFOIL_V8_STAGE_B_PREDECESSOR_VERSION
        )
    )
    if len(predecessors) != 1:
        raise AirfoilV8StageBError("memory lacks the exact learned-card predecessor")
    predecessor = predecessors[0]
    if predecessor.draft.content_sha256 != (
        "45612e513d6a72d66cfb2d99c58541cfb60982d9ab17cd83fed6b5e63e84052c"
    ):
        raise AirfoilV8StageBError("learned-card predecessor content changed")
    draft, lineage = _revision_from_event(_load_canonical_revision_event(trace_path))
    entry = memory.add_revision(
        predecessor.reference,
        draft,
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=lineage,
        revision_note="postseal evidence-guided revision",
    )
    if (
        entry.reference.insight_id != predecessor.reference.insight_id
        or entry.reference.version != AIRFOIL_V8_STAGE_B_CARD_VERSION
        or entry.draft.content_sha256 != AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256
        or entry.evidence_lineage is None
        or entry.evidence_lineage.identity_sha256
        != AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256
        or entry.lifecycle_state is not InsightLifecycleState.QUARANTINED
        or entry.origin is not InsightOrigin.REFLECTION
    ):
        raise AirfoilV8StageBError("replayed learned card differs from canonical G3")
    return entry


def build_airfoil_v8_stage_b_benchmark(base: AgenticBenchmark) -> AgenticBenchmark:
    """Attach only Airfoil's support compiler to an otherwise unchanged bundle."""

    if type(base) is not AgenticBenchmark:
        raise TypeError("base must be an exact AgenticBenchmark")
    base.validate_binding()
    benchmark = replace(
        base,
        hypothesis_compiler=AirfoilV8ReflectionNativeTrimHypothesisCompiler(),
        finite_action_set_compiler=AirfoilTrimLocalSupportCompiler(),
    )
    benchmark.validate_binding()
    if benchmark.reward.binding_sha256 != AIRFOIL_G3_ABSOLUTE_REWARD.binding_sha256:
        raise AirfoilV8StageBError("Stage-B benchmark changed the absolute G3 reward")
    return benchmark


@dataclass(frozen=True, slots=True)
class AirfoilV8StageBInputs:
    """Provider/evaluator-free objects for one disclosed-parent A/U block."""

    benchmark: AgenticBenchmark
    id_factory: Any
    memory: InsightMemoryBank
    learned_card: InsightMemoryEntry
    seed_configuration: dict[str, object]
    planner_factory: MatchedFiniteActionBlockFactory
    source_runtime_inputs_sha256: str

    def __post_init__(self) -> None:
        self.benchmark.validate_binding()
        if type(self.benchmark.finite_action_set_compiler) is not (
            AirfoilTrimLocalSupportCompiler
        ):
            raise ValueError("Stage-B benchmark lacks the Airfoil K=8 compiler")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if self.memory.entries_for((self.learned_card.reference,)) != (
            self.learned_card,
        ):
            raise ValueError("learned card is not owned by the runtime memory")
        if (
            self.learned_card.reference.insight_id.value
            != AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID
            or self.learned_card.reference.version
            != AIRFOIL_V8_STAGE_B_CARD_VERSION
            or self.learned_card.draft.content_sha256
            != AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256
            or self.learned_card.lifecycle_state
            is not InsightLifecycleState.QUARANTINED
        ):
            raise ValueError("Stage-B learned card identity changed")
        if type(self.seed_configuration) is not dict:
            raise TypeError("seed_configuration must be an exact dictionary")
        if self.planner_factory.card_reference != self.learned_card.reference:
            raise ValueError("planner factory is bound to another card")
        if (
            self.planner_factory.catalog_id != AIRFOIL_V8_STAGE_B_CATALOG_ID
            or self.planner_factory.required_cardinality
            != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
        ):
            raise ValueError("planner factory changed the exact K=8 Airfoil design")
        if (
            type(self.source_runtime_inputs_sha256) is not str
            or len(self.source_runtime_inputs_sha256) != 64
        ):
            raise ValueError("source runtime input SHA-256 is malformed")


def compose_airfoil_v8_stage_b_inputs(
    source: AirfoilG3RuntimeInputs,
) -> AirfoilV8StageBInputs:
    """Upgrade fresh frozen G3 inputs with the exact post-G3 learned revision."""

    if type(source) is not AirfoilG3RuntimeInputs:
        raise TypeError("source must be exact AirfoilG3RuntimeInputs")
    source.__post_init__()
    heldout_request = next(
        request
        for request in source.prepared_hypothesis_matrices[1].requests
        if request.reference.insight_id.value == AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID
    )
    learned = replay_canonical_g3_adaptive_revision(source.memory)
    seed = thaw_json(source.preparation.heldout_parent.candidate.configuration)
    if type(seed) is not dict:
        raise AirfoilV8StageBError("held-out Airfoil parent is not an object")
    benchmark = build_airfoil_v8_stage_b_benchmark(source.benchmark)
    factory = MatchedFiniteActionBlockFactory(
        card_reference=learned.reference,
        catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
        required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        context_projection_sha256=heldout_request.context_projection_sha256,
        endpoint_definition_sha256=ABSOLUTE_Q_DEFINITION_SHA256,
        task_sha256=AIRFOIL_V8_STAGE_B_TASK_SHA256,
        pre_outcome_phase_commit_sha256=(
            AIRFOIL_V8_STAGE_B_PRE_OUTCOME_COMMIT_SHA256
        ),
        uniform_policy=TaskKeyedUniformFiniteActionPolicy(
            schedule_seed_sha256=AIRFOIL_V8_STAGE_B_SCHEDULE_SEED_SHA256,
        ),
        phase=AIRFOIL_V8_STAGE_B_PHASE,
    )
    return AirfoilV8StageBInputs(
        benchmark=benchmark,
        id_factory=source.id_factory,
        memory=source.memory,
        learned_card=learned,
        seed_configuration=seed,  # type: ignore[arg-type]
        planner_factory=factory,
        source_runtime_inputs_sha256=source.runtime_inputs_sha256,
    )


def compose_airfoil_v8_stage_b_optimizer(
    inputs: AirfoilV8StageBInputs,
    *,
    generator: AgenticGenerator,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
) -> AgenticOptimizerComposition:
    """Shared provider-free/live optimizer composition for the A/U block."""

    inputs.__post_init__()
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    return compose_agentic_optimizer(
        inputs.benchmark,
        generator=generator,
        planner_factory=inputs.planner_factory,
        budget=AIRFOIL_V8_STAGE_B_BUDGET,
        seed=AIRFOIL_V8_STAGE_B_RUN_SEED,
        id_factory=inputs.id_factory,
        memory=inputs.memory,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        structured_output_budget_policy=AIRFOIL_V8_STAGE_B_OUTPUT_BUDGET,
        temperature=0.2,
    )


def build_airfoil_v8_stage_b_openrouter_config() -> ProgressAwareOpenRouterConfig:
    """Reuse the authenticated provider maximum and queue/backoff parameters."""

    base = build_openrouter_config()
    config = replace(
        base,
        jitter_seed=AIRFOIL_V8_STAGE_B_RUN_SEED,
        jitter_domain="airfoil-v8-stage-b-matched-finite-action-v1",
        app_title="AgentEvolve AAAI 2027 Airfoil Stage-B A/U development",
    )
    if (
        config.model_name != MODEL_ALIAS
        or config.max_connections != MAX_IN_FLIGHT
        or config.max_pending != MAX_PENDING
        or config.reasoning_config is None
        or config.reasoning_config.max_tokens != MAX_OUTPUT_TOKENS
    ):
        raise AirfoilV8StageBError("Stage-B OpenRouter envelope drifted")
    return config


def _production_generator_factory(
    api_key: str,
    config: ProgressAwareOpenRouterConfig,
    sinks: AirfoilG3LiveSinks,
) -> OwnedAgenticGenerator:
    sinks.__post_init__()
    runner = create_progress_aware_openrouter_runner(
        api_key=api_key,
        config=config,
        progress_sink=sinks.progress,
        outcome_sink=sinks.outcome,
        request_evidence_sink=sinks.request_evidence,
        output_evidence_sink=sinks.output_evidence,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    generator = TelemetryGatedAgenticGenerator(
        PydanticAIAgenticGenerator(runner),
        build_telemetry_policy(),
    )
    return OwnedAgenticGenerator(generator=generator, close=runner.aclose)


class LazyAirfoilV8StageBGenerator:
    """Initialize the paid transport only when the one model arm dispatches."""

    def __init__(
        self,
        *,
        credential_loader: Callable[[], str],
        sinks: AirfoilG3LiveSinks,
        factory: LiveGeneratorFactory = _production_generator_factory,
    ) -> None:
        if not callable(credential_loader):
            raise TypeError("credential_loader must be callable")
        sinks.__post_init__()
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.credential_loader = credential_loader
        self.sinks = sinks
        self.factory = factory
        self._owned: OwnedAgenticGenerator | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def initialized(self) -> bool:
        return self._owned is not None

    async def _generator(self) -> AgenticGenerator:
        if self._closed:
            raise AirfoilV8StageBError("lazy Stage-B generator is closed")
        if self._owned is None:
            async with self._lock:
                if self._owned is None:
                    api_key = self.credential_loader()
                    if type(api_key) is not str or not api_key:
                        raise AirfoilV8StageBError(
                            "credential loader returned no OpenRouter key"
                        )
                    owned = self.factory(
                        api_key,
                        build_airfoil_v8_stage_b_openrouter_config(),
                        self.sinks,
                    )
                    if type(owned) is not OwnedAgenticGenerator:
                        raise TypeError("live generator factory returned wrong value")
                    owned.__post_init__()
                    self._owned = owned
        assert self._owned is not None
        return self._owned.generator

    async def propose(self, request):
        return await (await self._generator()).propose(request)

    async def reflect(self, request):
        return await (await self._generator()).reflect(request)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owned is not None:
                value = self._owned.close()
                if hasattr(value, "__await__"):
                    await value


@dataclass(frozen=True, slots=True)
class AirfoilV8StageBLiveComposition:
    """Single-use live wrapper retaining ownership of the queued transport."""

    composition: AgenticOptimizerComposition
    inputs: AirfoilV8StageBInputs
    generator: LazyAirfoilV8StageBGenerator

    async def run(self):
        return await self.composition.optimizer.run((self.inputs.seed_configuration,))

    async def aclose(self) -> None:
        await self.generator.aclose()


def compose_airfoil_v8_stage_b_live(
    inputs: AirfoilV8StageBInputs,
    *,
    credential_loader: Callable[[], str],
    progress_sink,
    outcome_sink,
    request_evidence_sink,
    output_evidence_sink,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
    generator_factory: LiveGeneratorFactory = _production_generator_factory,
) -> AirfoilV8StageBLiveComposition:
    """Compose the lazy paid route without starting credentials, model, or CFD."""

    sinks = AirfoilG3LiveSinks(
        progress=progress_sink,
        outcome=outcome_sink,
        request_evidence=request_evidence_sink,
        output_evidence=output_evidence_sink,
    )
    lazy = LazyAirfoilV8StageBGenerator(
        credential_loader=credential_loader,
        sinks=sinks,
        factory=generator_factory,
    )
    composition = compose_airfoil_v8_stage_b_optimizer(
        inputs,
        generator=lazy,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
    )
    if lazy.initialized:
        raise AirfoilV8StageBError("live composition initialized provider eagerly")
    return AirfoilV8StageBLiveComposition(composition, inputs, lazy)


def airfoil_v8_stage_b_readiness_record(
    inputs: AirfoilV8StageBInputs,
) -> dict[str, object]:
    """Return a provider/CFD-free launch summary for durable preflight."""

    inputs.__post_init__()
    config = build_airfoil_v8_stage_b_openrouter_config()
    compiled = inputs.benchmark.compile_registered_hypothesis_treatment(
        catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
        parent_candidate_id=CandidateId(
            "candidate_airfoil_v8_stage_b_readiness_probe"
        ),
        parent_configuration=inputs.seed_configuration,
        entry=inputs.learned_card,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256=(
            inputs.planner_factory.context_projection_sha256
        ),
        endpoint_definition_sha256=(
            inputs.planner_factory.endpoint_definition_sha256
        ),
    )
    authority, _ = inputs.benchmark.compile_finite_action_set(
        compiled_anchor=compiled,
        required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        source_mode=inputs.planner_factory.source_mode,
    )
    if (
        authority.card.reference != inputs.learned_card.reference
        or authority.support.cardinality
        != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
        or len(
            {
                row.phenotype_identity_sha256
                for row in authority.support.options
            }
        )
        != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
    ):
        raise AirfoilV8StageBError("provider-free K=8 authority probe failed")
    return {
        "schema_version": 1,
        "ready": True,
        "claim_boundary": "development_not_fresh_paper_evidence",
        "definition_sha256": AIRFOIL_V8_STAGE_B_DEFINITION_SHA256,
        "source_runtime_inputs_sha256": inputs.source_runtime_inputs_sha256,
        "learned_card": {
            "insight_id": inputs.learned_card.reference.insight_id.value,
            "version": inputs.learned_card.reference.version,
            "content_sha256": inputs.learned_card.draft.content_sha256,
            "lineage_sha256": inputs.learned_card.evidence_lineage.identity_sha256,
        },
        "budget": AIRFOIL_V8_STAGE_B_BUDGET.to_trace_record(),
        "same_support": {
            "arms": ["A", "U"],
            "catalog_id": AIRFOIL_V8_STAGE_B_CATALOG_ID,
            "cardinality": AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
            "support_compiler_sha256": (
                AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256
            ),
            "resample_on_alias": False,
            "readiness_probe_authority_sha256": authority.authority_sha256,
            "readiness_probe_support_sha256": authority.support.support_sha256,
            "readiness_probe_anchor_option_id": (
                authority.support.anchor_option_id
            ),
        },
        "provider": {
            "model": config.model_name,
            "provider_options": config.provider_options,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "artificial_output_cap": False,
            "queue": config.to_manifest_record()["queue"],
        },
        "evaluator_concurrency": EVALUATOR_CONCURRENCY,
    }


__all__ = [
    "AIRFOIL_V8_STAGE_B_BUDGET",
    "AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256",
    "AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256",
    "AIRFOIL_V8_STAGE_B_CARD_VERSION",
    "AIRFOIL_V8_STAGE_B_DEFINITION_SHA256",
    "AIRFOIL_V8_STAGE_B_PRE_OUTCOME_COMMIT_SHA256",
    "AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY",
    "AIRFOIL_V8_STAGE_B_TASK_SHA256",
    "AirfoilV8StageBError",
    "AirfoilV8StageBInputs",
    "AirfoilV8StageBLiveComposition",
    "LazyAirfoilV8StageBGenerator",
    "airfoil_v8_stage_b_readiness_record",
    "build_airfoil_v8_stage_b_benchmark",
    "build_airfoil_v8_stage_b_openrouter_config",
    "compose_airfoil_v8_stage_b_inputs",
    "compose_airfoil_v8_stage_b_live",
    "compose_airfoil_v8_stage_b_optimizer",
    "replay_canonical_g3_adaptive_revision",
]
