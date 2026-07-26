"""Airfoil composition for generic K-option agentic evolution.

This module owns only Airfoil-specific dependency injection.  The generic
planner owns the G0--G3 chronology, memory checkpoint, finite-option decisions,
uniform comparator, reproduction, recombination, and model crossovers.  The
generic post-evolution interceptor owns terminal reflection.

Provider-free experiments and paid runs call the same
``compose_airfoil_v10_multi_option_optimizer`` function.  The live wrapper
injects a lazy generator, so importing or composing this module does not read a
credential, construct an OpenRouter client, call a model, or evaluate an
Airfoil.  The provider is first initialized only when G1 dispatches its first
model-authored option choice, after both G0 seeds have been admitted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from agent_evolve.agentic import (
    AgenticGenerator,
    AgenticOptimizerComposition,
    compose_agentic_optimizer,
)
from agent_evolve.application.multi_option_evolution import (
    MULTI_OPTION_EVOLUTION_BUDGET,
    MULTI_OPTION_G2_SLOT_IDS,
    MultiOptionEvolutionPlanner,
    MultiOptionEvolutionPlannerFactory,
)
from agent_evolve.application.post_evolution_reflection import (
    PostEvolutionReflectionFactory,
    PostEvolutionReflectionSource,
    PostEvolutionReflectionSourceScope,
    PostEvolutionReflectionSpec,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareOpenRouterConfig,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestSink,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    StructuredEvidencePublicationPolicy,
    StructuredOutputEvidenceSink,
    StructuredRequestEvidenceSink,
)
from agent_evolve.policies.selection.finite_action import (
    TaskKeyedUniformFiniteActionPolicy,
)
from agent_evolve.ports.agentic_generator import (
    ReflectionInsightContract,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import StructuredStreamProgressSink
from agent_evolve.application.gated_agentic_generator import (
    TelemetryGatedAgenticGenerator,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    DEEPSEEK_G3_PROVIDER_PROFILE,
    EVALUATOR_CONCURRENCY,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
    AirfoilG3ProviderProfile,
    OwnedAgenticGenerator,
    build_openrouter_config,
    build_telemetry_policy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    AIRFOIL_G3_ABSOLUTE_REWARD,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AIRFOIL_G3_MODEL_CATALOG_ID,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AIRFOIL_V8_STAGE_B_CATALOG_ID,
    AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    AIRFOIL_V10_CARD_ROLES,
    AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
    AIRFOIL_V10_MULTI_OPTION_PHASE,
    AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256,
    AIRFOIL_V10_RUN_SEED,
    AIRFOIL_V10_SEED_ROLES,
    AirfoilV10MultiOptionInputs,
)


AIRFOIL_V10_REFLECTION_LABEL = "airfoil_v10_terminal_reflection"
AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES = (
    DEEPSEEK_G3_PROVIDER_PROFILE,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
)

_ESTIMAND_DOMAIN = b"agent-evolve:airfoil-v10-multi-option-estimand:v1\x00"
_ESTIMAND_DEFINITION = {
    "schema_version": 1,
    "estimand_id": "airfoil_v10_k_option_memory_guided_evolution",
    "g1": (
        "randomized one-card assignments on a diagnostic parent; the model "
        "chooses one action from each card's authenticated K=8 neighbourhood"
    ),
    "g2": (
        "adaptive and score-shuffled card selection on a held-out parent, "
        "plus a prospective uniform draw on the exact adaptive support and an "
        "outcome-blind orthogonal shape mate"
    ),
    "g3": (
        "reproduction, replay-verified disjoint unions, and model-authored "
        "two-parent crossovers"
    ),
    "terminal_reflection": (
        "one optional revision of the adaptive card using the sealed G2 "
        "adaptive-versus-uniform exact-action evidence"
    ),
    "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
    "current_run_outcome_access_during_composition": False,
}
AIRFOIL_V10_MULTI_OPTION_ESTIMAND_SHA256 = hashlib.sha256(
    _ESTIMAND_DOMAIN
    + json.dumps(
        _ESTIMAND_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


class AirfoilV10MultiOptionLiveError(RuntimeError):
    """The Airfoil v10 composition or its lazy provider route drifted."""


@dataclass(frozen=True, slots=True)
class AirfoilV10MultiOptionLiveSinks:
    """All synchronous durability boundaries required before v10 dispatch."""

    progress: StructuredStreamProgressSink
    outcome: Callable[[Any], None]
    request_evidence: StructuredRequestEvidenceSink
    output_evidence: StructuredOutputEvidenceSink
    outbound_request_manifest: OpenRouterOutboundRequestManifestSink

    def __post_init__(self) -> None:
        for name in (
            "progress",
            "outcome",
            "request_evidence",
            "output_evidence",
            "outbound_request_manifest",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} sink must be callable")


AirfoilV10LiveGeneratorFactory = Callable[
    [
        AirfoilG3ProviderProfile,
        str,
        ProgressAwareOpenRouterConfig,
        AirfoilV10MultiOptionLiveSinks,
    ],
    OwnedAgenticGenerator,
]


def _require_profile(
    provider_profile: AirfoilG3ProviderProfile,
) -> AirfoilG3ProviderProfile:
    if type(provider_profile) is not AirfoilG3ProviderProfile:
        raise TypeError("provider_profile must be an exact AirfoilG3ProviderProfile")
    provider_profile.__post_init__()
    if provider_profile not in AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES:
        raise AirfoilV10MultiOptionLiveError(
            "v10 permits only the authenticated DeepSeek or GPT Sol profile"
        )
    if provider_profile is GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE and (
        provider_profile.reasoning_config.to_model_setting() != {"effort": "xhigh"}
    ):
        raise AirfoilV10MultiOptionLiveError(
            "GPT Sol must use standard mode with xhigh effort only"
        )
    return provider_profile


def build_airfoil_v10_openrouter_config(
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> ProgressAwareOpenRouterConfig:
    """Project either authenticated provider into the same v10 queue policy."""

    profile = _require_profile(provider_profile)
    config = replace(
        build_openrouter_config(profile),
        jitter_seed=AIRFOIL_V10_RUN_SEED,
        jitter_domain="airfoil-v10-multi-option-evolution-v1",
        app_title="AgentEvolve AAAI 2027 Airfoil v10 multi-option evolution",
    )
    config.__post_init__()
    if (
        config.model_name != profile.model_alias
        or config.provider_only != (profile.provider_slug,)
        or config.reasoning_config != profile.reasoning_config
        or config.stream_liveness_policy.absolute_timeout_ns
        != profile.absolute_timeout_seconds * 1_000_000_000
    ):
        raise AirfoilV10MultiOptionLiveError(
            "v10 OpenRouter projection differs from its provider profile"
        )
    if profile is GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE and (
        config.reasoning_config is None
        or config.reasoning_config.to_model_setting() != {"effort": "xhigh"}
    ):
        raise AirfoilV10MultiOptionLiveError(
            "GPT Sol transport must request only xhigh reasoning effort"
        )
    return config


def _planner_context(inputs: AirfoilV10MultiOptionInputs) -> str:
    contexts = tuple(seed.context_projection_sha256 for seed in inputs.g0_seeds)
    if (
        len(set(contexts)) != 1
        or contexts[0] != inputs.context_projection_sha256
        or contexts[0] != AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
    ):
        raise AirfoilV10MultiOptionLiveError(
            "v10 G0 roles do not share the committed memory context stratum"
        )
    return contexts[0]


def build_airfoil_v10_planner_factory(
    inputs: AirfoilV10MultiOptionInputs,
    *,
    planner_trace_sink=None,
) -> MultiOptionEvolutionPlannerFactory:
    """Bind Airfoil policies to the benchmark-neutral evolutionary planner."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    active_references = tuple(card.reference for card in inputs.active_cards)
    if active_references != tuple(sorted(set(active_references))):
        raise AirfoilV10MultiOptionLiveError(
            "active card references are not canonically ordered"
        )
    return MultiOptionEvolutionPlannerFactory(
        reward_binding=AIRFOIL_G3_ABSOLUTE_REWARD,
        active_references=active_references,
        model_catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
        mate_catalog_id=AIRFOIL_G3_MODEL_CATALOG_ID,
        mate_choice=inputs.mate_choice,
        required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        uniform_policy=TaskKeyedUniformFiniteActionPolicy(
            schedule_seed_sha256=AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256,
        ),
        task_sha256=inputs.task_sha256,
        pre_outcome_phase_commit_sha256=inputs.pre_outcome_commit_sha256,
        endpoint_definition_sha256=ABSOLUTE_Q_DEFINITION_SHA256,
        context_projection_sha256=_planner_context(inputs),
        estimand_stratum_sha256=AIRFOIL_V10_MULTI_OPTION_ESTIMAND_SHA256,
        phase=inputs.phase,
        trace_sink=planner_trace_sink,
    )


def _reflection_contract(
    inputs: AirfoilV10MultiOptionInputs,
) -> ReflectionInsightContract:
    hypothesis_role = AIRFOIL_V10_SEED_ROLES[1]
    option_ids = tuple(
        sorted(
            {
                row.option.option_id
                for binding in inputs.authority_bindings
                if binding.seed_role == hypothesis_role
                for row in binding.authority.support.options
            }
        )
    )
    if len(option_ids) != (
        len(AIRFOIL_V10_CARD_ROLES) * AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
    ):
        raise AirfoilV10MultiOptionLiveError(
            "terminal reflection vocabulary is not the full held-out K-option union"
        )
    return ReflectionInsightContract(
        required_metric_ids=(
            "objective:normalized_multipoint_drag",
            "violation:normalized_lift_equality",
        ),
        allowed_option_families=("trim_only",),
        allowed_option_ids=option_ids,
    )


def _adaptive_predecessor(planner: object) -> InsightRef:
    if type(planner) is not MultiOptionEvolutionPlanner:
        raise TypeError("reflection received a foreign evolutionary planner")
    reference = planner.adaptive_reference
    if type(reference) is not InsightRef:
        raise AirfoilV10MultiOptionLiveError(
            "adaptive memory assignment is unavailable at terminal reflection"
        )
    return reference


def build_airfoil_v10_reflection_factory(
    inputs: AirfoilV10MultiOptionInputs,
) -> PostEvolutionReflectionFactory:
    """Build one terminal, receipt-bound adaptive-versus-uniform reflection."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    spec = PostEvolutionReflectionSpec(
        terminal_generation=3,
        source_scope=PostEvolutionReflectionSourceScope(
            sources=(
                PostEvolutionReflectionSource(2, MULTI_OPTION_G2_SLOT_IDS[0]),
                PostEvolutionReflectionSource(2, MULTI_OPTION_G2_SLOT_IDS[2]),
            ),
            policy_id="airfoil_v10_adaptive_uniform_scope",
        ),
        insight_contract=_reflection_contract(inputs),
        policy_id="airfoil_v10_terminal_reflection",
        label=AIRFOIL_V10_REFLECTION_LABEL,
    )
    return PostEvolutionReflectionFactory(
        spec=spec,
        predecessor_resolver=_adaptive_predecessor,
    )


def compose_airfoil_v10_multi_option_optimizer(
    inputs: AirfoilV10MultiOptionInputs,
    *,
    generator: AgenticGenerator,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
    planner_trace_sink=None,
) -> AgenticOptimizerComposition:
    """Shared provider-free/live composition for the complete G0--G3 run."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    profile = _require_profile(provider_profile)
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    return compose_agentic_optimizer(
        inputs.benchmark,
        generator=generator,
        planner_factory=build_airfoil_v10_planner_factory(
            inputs,
            planner_trace_sink=planner_trace_sink,
        ),
        feedback_interceptor_factory=build_airfoil_v10_reflection_factory(inputs),
        budget=MULTI_OPTION_EVOLUTION_BUDGET,
        seed=AIRFOIL_V10_RUN_SEED,
        id_factory=inputs.id_factory,
        memory=inputs.memory,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
        max_output_tokens=profile.max_output_tokens,
        structured_output_budget_policy=profile.output_budget_policy,
        temperature=profile.temperature,
    )


def _production_generator_factory(
    provider_profile: AirfoilG3ProviderProfile,
    api_key: str,
    config: ProgressAwareOpenRouterConfig,
    sinks: AirfoilV10MultiOptionLiveSinks,
) -> OwnedAgenticGenerator:
    sinks.__post_init__()
    runner = create_progress_aware_openrouter_runner(
        api_key=api_key,
        config=config,
        progress_sink=sinks.progress,
        outcome_sink=sinks.outcome,
        request_evidence_sink=sinks.request_evidence,
        output_evidence_sink=sinks.output_evidence,
        outbound_request_manifest_sink=sinks.outbound_request_manifest,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    generator = TelemetryGatedAgenticGenerator(
        PydanticAIAgenticGenerator(runner),
        build_telemetry_policy(provider_profile),
    )
    return OwnedAgenticGenerator(generator=generator, close=runner.aclose)


class LazyAirfoilV10MultiOptionGenerator:
    """Initialize one selected paid transport on the first evolutionary call."""

    def __init__(
        self,
        *,
        credential_loader: Callable[[], str],
        sinks: AirfoilV10MultiOptionLiveSinks,
        provider_profile: AirfoilG3ProviderProfile = (DEEPSEEK_G3_PROVIDER_PROFILE),
        factory: AirfoilV10LiveGeneratorFactory = _production_generator_factory,
    ) -> None:
        if not callable(credential_loader):
            raise TypeError("credential_loader must be callable")
        if type(sinks) is not AirfoilV10MultiOptionLiveSinks:
            raise TypeError("sinks must be exact AirfoilV10MultiOptionLiveSinks")
        sinks.__post_init__()
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.credential_loader = credential_loader
        self.sinks = sinks
        self.provider_profile = _require_profile(provider_profile)
        self.factory = factory
        self._owned: OwnedAgenticGenerator | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def initialized(self) -> bool:
        return self._owned is not None

    async def _generator(self) -> AgenticGenerator:
        if self._closed:
            raise AirfoilV10MultiOptionLiveError("lazy v10 generator is closed")
        if self._owned is None:
            async with self._lock:
                if self._owned is None:
                    api_key = self.credential_loader()
                    if type(api_key) is not str or not api_key:
                        raise AirfoilV10MultiOptionLiveError(
                            "credential loader returned no OpenRouter API key"
                        )
                    owned = self.factory(
                        self.provider_profile,
                        api_key,
                        build_airfoil_v10_openrouter_config(self.provider_profile),
                        self.sinks,
                    )
                    if type(owned) is not OwnedAgenticGenerator:
                        raise TypeError("live generator factory returned wrong value")
                    owned.__post_init__()
                    self._owned = owned
        assert self._owned is not None
        return self._owned.generator

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        return await (await self._generator()).propose(request)

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        return await (await self._generator()).reflect(request)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owned is not None:
                result = self._owned.close()
                if hasattr(result, "__await__"):
                    await result


class AirfoilV10MultiOptionLiveComposition:
    """Single-use live wrapper retaining ownership of the queued transport."""

    def __init__(
        self,
        *,
        composition: AgenticOptimizerComposition,
        inputs: AirfoilV10MultiOptionInputs,
        generator: LazyAirfoilV10MultiOptionGenerator,
        provider_profile: AirfoilG3ProviderProfile,
    ) -> None:
        self.composition = composition
        self.inputs = inputs
        self.generator = generator
        self.provider_profile = provider_profile
        self._run_state = "not_started"
        self._run_lock = threading.Lock()

    @property
    def initialized_provider(self) -> bool:
        return self.generator.initialized

    @property
    def run_state(self) -> str:
        with self._run_lock:
            return self._run_state

    async def run(self):
        with self._run_lock:
            if self._run_state != "not_started":
                raise AirfoilV10MultiOptionLiveError(
                    f"live composition is single-use (state={self._run_state})"
                )
            self._run_state = "running"
        try:
            result = await self.composition.optimizer.run(
                self.inputs.seed_configurations
            )
        except BaseException:
            with self._run_lock:
                self._run_state = "failed"
            raise
        with self._run_lock:
            self._run_state = "completed"
        return result

    async def aclose(self) -> None:
        await self.generator.aclose()


def compose_airfoil_v10_multi_option_live(
    inputs: AirfoilV10MultiOptionInputs,
    *,
    credential_loader: Callable[[], str],
    progress_sink,
    outcome_sink,
    request_evidence_sink,
    output_evidence_sink,
    outbound_request_manifest_sink,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    generator_factory: AirfoilV10LiveGeneratorFactory = _production_generator_factory,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
    planner_trace_sink=None,
) -> AirfoilV10MultiOptionLiveComposition:
    """Compose the paid route without reading credentials or starting work."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    profile = _require_profile(provider_profile)
    sinks = AirfoilV10MultiOptionLiveSinks(
        progress=progress_sink,
        outcome=outcome_sink,
        request_evidence=request_evidence_sink,
        output_evidence=output_evidence_sink,
        outbound_request_manifest=outbound_request_manifest_sink,
    )
    lazy = LazyAirfoilV10MultiOptionGenerator(
        credential_loader=credential_loader,
        sinks=sinks,
        provider_profile=profile,
        factory=generator_factory,
    )
    composition = compose_airfoil_v10_multi_option_optimizer(
        inputs,
        generator=lazy,
        provider_profile=profile,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
        planner_trace_sink=planner_trace_sink,
    )
    if lazy.initialized:
        raise AirfoilV10MultiOptionLiveError(
            "live composition initialized its provider eagerly"
        )
    return AirfoilV10MultiOptionLiveComposition(
        composition=composition,
        inputs=inputs,
        generator=lazy,
        provider_profile=profile,
    )


__all__ = [
    "AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES",
    "AIRFOIL_V10_MULTI_OPTION_ESTIMAND_SHA256",
    "AIRFOIL_V10_MULTI_OPTION_PHASE",
    "AIRFOIL_V10_REFLECTION_LABEL",
    "AirfoilV10MultiOptionLiveComposition",
    "AirfoilV10MultiOptionLiveError",
    "AirfoilV10MultiOptionLiveSinks",
    "AirfoilV10LiveGeneratorFactory",
    "LazyAirfoilV10MultiOptionGenerator",
    "build_airfoil_v10_openrouter_config",
    "build_airfoil_v10_planner_factory",
    "build_airfoil_v10_reflection_factory",
    "compose_airfoil_v10_multi_option_live",
    "compose_airfoil_v10_multi_option_optimizer",
]
