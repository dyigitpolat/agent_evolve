#!/usr/bin/env python3
"""One-selector, non-scientific conformance gate for the BOiLS live stack.

``readiness`` constructs one genuine, registered generation-1 BOiLS
calibrated K8-to-K4 request and executes it only against a provider-free
capture runner.  It reads no credential, constructs no provider client, and
executes no ABC evaluation.  ``live`` binds a finalized readiness directory
and executes exactly that selector path once with the campaign's production
OpenRouter configuration.  Neither mode is optimization or scientific
evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import DeterministicIdFactory  # noqa: E402
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignArchiveCutoffReceipt,
    CampaignStageRequest,
)
from agent_evolve.application.parent_measurement import (  # noqa: E402
    attach_parent_measurement_to_context,
    bind_parent_measurement,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    CampaignPortfolioWaveContext,
)
from agent_evolve.application.portfolio_outcome_feedback import (  # noqa: E402
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.domain.typed_json import thaw_json  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (  # noqa: E402
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (  # noqa: E402
    CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (  # noqa: E402
    build_provider_attempt_terminal_join_receipt,
    validate_provider_attempt_terminal_join_receipt,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
)
from agent_evolve.ports.variation_source import (  # noqa: E402
    finite_variation_source_by_option,
    finite_variation_source_ids,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamProgress,
)
from examples.development import durable_run_artifacts  # noqa: E402
from examples.development import run_boils_generic_campaign as campaign  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    decode_json_bytes,
    file_identity,
    finalize_run_directory,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


KIND = "boils_exact_stack_conformance"
SCHEMA_VERSION = 1
LIVE_AUTHORIZATION = "RUN_ONE_BOILS_SELECTOR_CONFORMANCE_CALL"
DEFAULT_RUN_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/exact_stack_conformance"
)
FRAMEWORK_PACKAGES = ("httpx", "openai", "pydantic", "pydantic-ai")
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")


class BoilsExactStackConformanceError(RuntimeError):
    """A content-safe canary invariant failed."""


class FailIfCalledBoilsEvaluator:
    """Evaluator port used to prove that the conformance path executes no ABC."""

    instances: list["FailIfCalledBoilsEvaluator"] = []

    def __init__(self, settings: object, *, observer: Callable[[object], None]) -> None:
        self.settings = settings
        self.observer = observer
        self.calls = 0
        type(self).instances.append(self)

    def provenance(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "provider_free_fail_if_called_boils_evaluator",
            "abc_executions": 0,
        }

    def evaluate(self, configuration: object) -> object:
        del configuration
        self.calls += 1
        raise BoilsExactStackConformanceError(
            "the conformance gate attempted an ABC evaluation"
        )


@dataclass(frozen=True, slots=True)
class ConformanceInputs:
    request: PortfolioSelectionRequest
    coordinator: CalibratedPortfolioCampaignCoordinator
    evaluator_guard: FailIfCalledBoilsEvaluator
    evaluator_observer: object
    preparation_sha256: str

    def __post_init__(self) -> None:
        if type(self.request) is not PortfolioSelectionRequest:
            raise TypeError("request must be exact PortfolioSelectionRequest")
        self.request.__post_init__()
        if type(self.coordinator) is not CalibratedPortfolioCampaignCoordinator:
            raise TypeError("coordinator must be exact")
        if self.coordinator.registered_request_count != 1:
            raise BoilsExactStackConformanceError(
                "exactly one selector request must be registered"
            )
        if self.evaluator_guard.calls != 0:
            raise BoilsExactStackConformanceError("input construction invoked ABC")


class LiveRunner(Protocol):
    async def __call__(
        self, request: StructuredGenerationRequest[Any]
    ) -> AttemptedStructuredGenerationResponse[Any]: ...

    async def snapshot(self) -> object: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ConformanceDependencies:
    inputs_factory: Callable[[Path, str], ConformanceInputs]
    credential_loader: Callable[[], str]
    runner_factory: Callable[..., LiveRunner]

    def __post_init__(self) -> None:
        if not all(
            callable(getattr(self, name))
            for name in ("inputs_factory", "credential_loader", "runner_factory")
        ):
            raise TypeError("all conformance dependencies must be callable")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(value: str) -> str:
    if type(value) is not str or _SAFE_RUN_ID.fullmatch(value) is None:
        raise BoilsExactStackConformanceError("run_id violates the closed grammar")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "tests/test_boils_exact_stack_conformance.py",
        Path(campaign.__file__),
        Path(durable_run_artifacts.__file__),
        *tuple(sorted((AGENT_EVOLVE_ROOT / "src/agent_evolve").rglob("*.py"))),
        *tuple(
            sorted(
                (AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc").glob("*.py")
            )
        ),
    )


def _source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def _framework_versions() -> dict[str, str]:
    return {name: version(name) for name in FRAMEWORK_PACKAGES}


def _read_openrouter_api_key() -> str:
    """Read the key at the sole credential boundary without persisting it."""

    value = os.environ.get("OPENROUTER_API_KEY")
    for dotenv in (WORKSPACE_ROOT / ".env", AGENT_EVOLVE_ROOT / ".env"):
        if value or not dotenv.is_file():
            continue
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if type(value) is not str or not value:
        raise BoilsExactStackConformanceError("OPENROUTER_API_KEY is unavailable")
    return value


def _production_runner_factory(**kwargs: object) -> LiveRunner:
    kwargs["evidence_publication_policy"] = (
        StructuredEvidencePublicationPolicy.REQUIRED
    )
    return create_progress_aware_openrouter_runner(**kwargs)  # type: ignore[arg-type]


def build_conformance_inputs(run_dir: Path, source_sha256: str) -> ConformanceInputs:
    """Construct one real, registered BOiLS G1 selector request without ABC."""

    preparation = DurableJsonlJournal(run_dir / "campaign_preparation.jsonl")
    evaluator = DurableJsonlJournal(run_dir / "evaluator_observations.jsonl")
    try:
        bundle = campaign._prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator,
            source_closure_sha256=source_sha256,
            arm="live",
            evaluator_factory=FailIfCalledBoilsEvaluator,
        )
        guard = bundle.evaluator
        if type(guard) is not FailIfCalledBoilsEvaluator:
            raise TypeError("BOiLS preparation replaced the evaluator guard")

        session = bundle.prepared.benchmark_session
        parents = tuple(
            campaign._construction_parent(
                ordinal=ordinal,
                configuration=seed.configuration,
                bundle=bundle,
            )
            for ordinal, seed in enumerate(bundle.prepared.seeds.seeds, start=1)
        )
        known = tuple(
            sorted(
                bundle.benchmark.phenotype_identity.identify(
                    thaw_json(parent.configuration)
                ).value_sha256
                for parent in parents
            )
        )
        memory = bundle.workload_ports.evidence.initialize_memory(
            session, bundle.prepared.seeds
        )
        archive = campaign._object(
            {
                "conformance_only": True,
                "front_candidates": [
                    {
                        "objectives": [
                            {"metric_id": name, "value_hex": value.hex()}
                            for name, value in parent.objectives
                        ]
                    }
                    for parent in parents
                ],
            }
        )
        step = next(
            value for value in bundle.prepared.schedule.steps if value.generation == 1
        )
        utility = bundle.utility.freeze(
            benchmark=session.benchmark,
            generation=1,
            archive=archive,
        )
        stage = CampaignStageRequest(
            preparation_sha256=bundle.prepared.preparation_sha256,
            runtime_start_receipt_sha256=campaign._sha(
                "boils-exact-stack-conformance-runtime-start"
            ),
            step=step,
            archive_cutoff=CampaignArchiveCutoffReceipt(
                request_sha256=campaign._sha(
                    "boils-exact-stack-conformance-cutoff-g1"
                ),
                preparation_sha256=bundle.prepared.preparation_sha256,
                generation=1,
                archive=archive,
                evidence=campaign._object(
                    {"conformance_only": True, "abc_executions": 0}
                ),
            ),
            archive_utility=utility,
            source_portfolio=None,
            test_eligible_reflection_receipt_sha256s=(),
            prior_selector_audit_set_sha256=campaign._sha(
                "boils-exact-stack-conformance-prior-audit-g1"
            ),
        )
        parent = parents[0]
        variation = bundle.workload_ports.catalog.bind(
            session.benchmark, parent.configuration, known
        )
        context = bundle.workload_ports.evidence.context(
            session,
            parent.configuration,
            variation,
            memory,
        )
        parent_measurement = bind_parent_measurement(
            candidate=parent,
            variation=variation,
            projection=bundle.parent_measurement_projection,
        )
        context = attach_parent_measurement_to_context(context, parent_measurement)
        cards = bundle.workload_ports.evidence.cards(
            session,
            parent.configuration,
            variation,
            memory,
        )
        binding_factory = CalibratedCampaignBindingFactory(
            scope=bundle.binding_factory.scope,
            objectives=bundle.binding_factory.objectives,
            ledger=PortfolioOutcomeFeedbackLedger(),
            structural_evidence=bundle.binding_factory.structural_evidence,
            prior=bundle.binding_factory.prior,
            family_min_support=bundle.binding_factory.family_min_support,
            option_prompt_projection=bundle.binding_factory.option_prompt_projection,
            common_candidate_pool_policy=(
                bundle.binding_factory.common_candidate_pool_policy
            ),
            proposal_support_policy=bundle.binding_factory.proposal_support_policy,
            assign_all_cards_by_default=(
                bundle.binding_factory.assign_all_cards_by_default
            ),
        )
        coordinator = CalibratedPortfolioCampaignCoordinator(
            allocator=campaign._allocator(),
            constraint_decoupled=campaign.CONSTRAINT_DECOUPLED_ACQUISITION,
            minimum_intervention_projection=(
                campaign.MINIMUM_INTERVENTION_PROJECTION
            ),
            evidence_calibrated_source_mix=(
                campaign.EVIDENCE_CALIBRATED_SOURCE_MIX
            ),
            contextual_search_allocation=campaign.CONTEXTUAL_SEARCH_ALLOCATION,
        )
        wave_context = CampaignPortfolioWaveContext(
            prepared=bundle.prepared,
            stage_request=stage,
            parent_slot=0,
            parent=parent,
            variation=variation,
            evidence_context=context,
            evidence_cards=cards,
            memory=memory,
            parent_measurement=parent_measurement,
        )
        if campaign.CONTEXTUAL_SEARCH_ALLOCATION:
            planner = campaign.CampaignContextualSearchPlanner(
                ledger=campaign.ContextualSearchLedger(),
                campaign_scope_sha256=campaign._sha(
                    "agent-evolve:contextual-search-campaign:"
                    + bundle.prepared.preparation_sha256
                ),
            )
            plan = planner.plan((wave_context,))
            allocation = plan.contracts[0]
            target = plan.frontier_targets[0]
            evidence = thaw_json(wave_context.evidence_context)
            if type(evidence) is not dict:
                raise TypeError("conformance evidence context must be an object")
            evidence[campaign.CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
            wave_context = replace(
                wave_context,
                evidence_context=campaign._object(evidence),
                contextual_allocation=allocation,
                frontier_target=target,
            )
        wave = campaign._WaveFactory(
            bundle=bundle,
            learning_runtime=None,
            records=[],
            ids=DeterministicIdFactory("boils_exact_stack_conformance"),
            binding_factory=binding_factory,
            coordinator=coordinator,
        ).build(wave_context)
        result = ConformanceInputs(
            request=wave.selection_request,
            coordinator=coordinator,
            evaluator_guard=guard,
            evaluator_observer=bundle.evaluator_observer,
            preparation_sha256=bundle.prepared.preparation_sha256,
        )
        if bundle.evaluator_observer.calls != 0:
            raise BoilsExactStackConformanceError(
                "input construction emitted an evaluator observation"
            )
        return result
    finally:
        preparation.close()
        evaluator.close()


_CANONICAL_DEPENDENCIES = (
    build_conformance_inputs,
    _read_openrouter_api_key,
    _production_runner_factory,
)


def _production_dependencies() -> ConformanceDependencies:
    return ConformanceDependencies(*_CANONICAL_DEPENDENCIES)


def _is_production_dependencies(value: ConformanceDependencies) -> bool:
    return (
        type(value) is ConformanceDependencies
        and value.inputs_factory is _CANONICAL_DEPENDENCIES[0]
        and value.credential_loader is _CANONICAL_DEPENDENCIES[1]
        and value.runner_factory is _CANONICAL_DEPENDENCIES[2]
    )


def _proposal_option_ids(request: PortfolioSelectionRequest) -> tuple[str, ...]:
    """Choose a deterministic, feasible, cross-family K8 without evaluations."""

    by_position: dict[str, list[Any]] = {}
    for option in request.finite_variation_contract.options:
        position = dict(option.metadata).get("position")
        if position is None:
            continue
        if type(position) not in (int, str):
            raise BoilsExactStackConformanceError(
                "finite option position metadata must be an integer or string"
            )
        position = str(position)
        by_position.setdefault(position, []).append(option)
    selected: list[str] = []
    used_families: set[str] = set()
    for position in sorted(by_position)[:8]:
        options = sorted(
            by_position[position], key=lambda value: (value.family, value.option_id)
        )
        option = next(
            (value for value in options if value.family not in used_families),
            options[0],
        )
        selected.append(option.option_id)
        used_families.add(option.family)
    if len(selected) != 8 or len(used_families) < 3:
        raise BoilsExactStackConformanceError(
            "provider-free capture could not form a feasible cross-family K8"
        )
    return tuple(selected)


class _ProviderFreeCaptureRunner:
    def __init__(self, request: PortfolioSelectionRequest) -> None:
        self.option_ids = _proposal_option_ids(request)
        self.card_key = request.cards[0].card_key
        self.hierarchical = (
            campaign.VARIATION_TOPOLOGY.mode.value == "hierarchical_r2"
        )
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(
        self, request: StructuredGenerationRequest[Any]
    ) -> StructuredGenerationResponse[Any]:
        self.requests.append(request)
        members = []
        for index, option_id in enumerate(self.option_ids, start=1):
            member = {
                "option_id": option_id,
                "supporting_card_keys": [self.card_key] if index == 1 else [],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": (
                    "exploit"
                    if index <= 3
                    else "falsify"
                    if index <= 6
                    else "coverage"
                ),
                "design_rationale": (
                    f"Provider-free conformance rationale {index} for {option_id}."
                ),
            }
            if self.hierarchical:
                member["action_kind"] = "atomic"
            members.append(member)
        value = request.output_type.model_validate({"members": members}, strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/boils-conformance",
            resolved_model="provider-free/boils-conformance",
            resolved_provider="provider-free",
            provider_response_id="provider-free-boils-conformance-response",
            finish_reason="tool_calls",
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


def _schema_contract(
    low_request: StructuredGenerationRequest[Any],
    option_ids: tuple[str, ...],
) -> dict[str, object]:
    schema = low_request.output_type.model_json_schema()
    members = schema.get("properties", {}).get("members", {})
    definitions = schema.get("$defs", {})
    exposed_option_ids: list[str] = []
    for definition in definitions.values():
        if type(definition) is not dict:
            continue
        properties = definition.get("properties", {})
        if type(properties) is not dict:
            continue
        for field in ("option_id", "composite_option_id"):
            projection = properties.get(field, {})
            if type(projection) is not dict:
                continue
            enum = projection.get("enum")
            if type(enum) is list and all(type(value) is str for value in enum):
                exposed_option_ids.extend(enum)
    gates = {
        "members_exact_k8": (
            members.get("minItems") == 8 and members.get("maxItems") == 8
        ),
        "option_enum_exact_finite_contract": (
            len(exposed_option_ids) == len(option_ids)
            and len(set(exposed_option_ids)) == len(exposed_option_ids)
            and set(exposed_option_ids) == set(option_ids)
        ),
    }
    return {
        "logical_schema_sha256": hashlib.sha256(_canonical_bytes(schema)).hexdigest(),
        "logical_schema_utf8_bytes": len(_canonical_bytes(schema)),
        "gates": gates,
    }


async def _capture_contract(inputs: ConformanceInputs) -> dict[str, object]:
    """Exercise the real registered selector with a provider-free typed K8."""

    request = inputs.request
    capture = _ProviderFreeCaptureRunner(request)
    result = await inputs.coordinator.build_selector(capture).select(request)
    if type(result) is not PortfolioSelectionResult or len(capture.requests) != 1:
        raise BoilsExactStackConformanceError(
            "provider-free selector capture did not complete exactly once"
        )
    low_request = capture.requests[0]
    options = request.finite_variation_contract.options
    option_ids = tuple(value.option_id for value in options)
    config = campaign._provider_config()
    provider = config.to_manifest_record()
    schema_contract = _schema_contract(low_request, option_ids)
    audit = result.supplemental_audit
    if audit is None:
        raise BoilsExactStackConformanceError("selector omitted its K8 audit")
    audit_payload = thaw_json(audit.payload)
    original_members = audit_payload.get("original_k8_response", {}).get("members")
    source_ids = finite_variation_source_ids(request.finite_variation_contract)
    source_by_option = finite_variation_source_by_option(
        request.finite_variation_contract
    )
    selected_option_ids = tuple(member.option_id for member in result.decision.members)
    selected_source_counts = {
        source_id: sum(
            source_by_option[option_id] == source_id
            for option_id in selected_option_ids
        )
        for source_id in source_ids
    }
    contextual_realization = inputs.coordinator.decode_contextual_allocation_realization(
        result
    )
    expected_contextual = campaign.CONTEXTUAL_SEARCH_ALLOCATION
    contextual_exact = (
        contextual_realization is not None
        and contextual_realization.exact
        and dict(contextual_realization.realized_source_target_counts)
        == selected_source_counts
    )
    gates = {
        "one_registered_g1_request": inputs.coordinator.registered_request_count == 1,
        "exact_finite_option_contract": (
            len(options) >= 8 and len(set(option_ids)) == len(options)
        ),
        "exact_one_bootstrap_card": len(request.cards) == 1,
        "engine_k4_contract": request.portfolio_size == 4,
        "minimum_three_families": request.min_distinct_families == 3,
        "pairwise_disjoint_policy_exact": (
            request.require_pairwise_disjoint_parent_patches
            is campaign._require_pairwise_disjoint_evaluation_patches()
        ),
        "finite_variation_sources_exact": source_ids
        == tuple(sorted(selected_source_counts)),
        "contextual_allocation_presence_exact": (
            contextual_realization is not None
        )
        is expected_contextual,
        "contextual_allocation_realized_exactly": (
            contextual_exact if expected_contextual else True
        ),
        "proposal_k8_result": type(original_members) is list
        and len(original_members) == 8,
        "engine_k4_result": len(result.decision.members) == 4,
        "one_low_level_call": len(capture.requests) == 1,
        "exact_tool": (
            low_request.output_tool_name == CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
        ),
        "exact_prompt": low_request.prompt == inputs.coordinator.render(request),
        "max_output_tokens_exact": (
            low_request.max_output_tokens == campaign.MAX_OUTPUT_TOKENS
        ),
        "temperature_exact": low_request.temperature == campaign.TEMPERATURE,
        "model_alias": provider.get("model_name") == campaign.MODEL,
        "provider_route_exact": provider.get("provider_options", {}).get("only")
        == list(campaign.PROVIDER_ONLY),
        "no_provider_fallback": provider.get("provider_options", {}).get(
            "allow_fallbacks"
        )
        is False,
        "xhigh_reasoning": provider.get("reasoning") == {"effort": "xhigh"},
        "reasoning_mode_absent": "mode" not in provider.get("reasoning", {}),
        "queue_concurrency_exact": (
            provider.get("queue", {}).get("max_in_flight")
            == campaign.AGENT_CONCURRENCY
            and provider.get("queue", {}).get("max_pending")
            == campaign.AGENT_QUEUE_CAPACITY
            and provider.get("queue", {}).get("max_attempts")
            == campaign.MAX_ATTEMPTS
        ),
        "evaluator_never_called": inputs.evaluator_guard.calls == 0,
        "evaluator_observer_empty": getattr(inputs.evaluator_observer, "calls") == 0,
        **schema_contract["gates"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "request": {
            "call_id": request.call_id.value,
            "operation": request.operation,
            "request_sha256": request.request_sha256,
            "preparation_sha256": inputs.preparation_sha256,
            "finite_option_count": len(options),
            "finite_option_ids_sha256": hashlib.sha256(
                _canonical_bytes(option_ids)
            ).hexdigest(),
            "card_count": len(request.cards),
            "proposal_width": 8,
            "evaluation_width": request.portfolio_size,
            "prompt_sha256": hashlib.sha256(
                low_request.prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_utf8_bytes": len(low_request.prompt.encode("utf-8")),
            "output_tool_name": low_request.output_tool_name,
            "max_output_tokens": low_request.max_output_tokens,
            "temperature_hex": float(low_request.temperature).hex(),
        },
        "provider_config": provider,
        "framework_versions": _framework_versions(),
        "schema": {
            key: value for key, value in schema_contract.items() if key != "gates"
        },
        "provider_free_decision": {
            "proposed_option_ids": list(capture.option_ids),
            "selected_option_ids": list(selected_option_ids),
            "finite_variation_source_ids": list(source_ids),
            "selected_source_counts": [
                [source_id, selected_source_counts[source_id]]
                for source_id in source_ids
            ],
            "contextual_allocation_realization": (
                None
                if contextual_realization is None
                else contextual_realization.to_record()
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "abc_executions": 0,
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
    }


def _readiness_record(
    run_id: str,
    *,
    source: Mapping[str, object],
    contract: Mapping[str, object],
    production_stack_authenticated: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "mode": "readiness",
        "status": (
            "ready_conformance_only"
            if production_stack_authenticated
            else "ready_offline_test_only"
        ),
        "run_id": run_id,
        "created_at_utc": _utc(),
        "production_stack_authenticated": production_stack_authenticated,
        "source_identity": dict(source),
        "contract": dict(contract),
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "abc_executions": 0,
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
    }
    commitment_input = dict(record)
    commitment_input.pop("created_at_utc")
    record["readiness_sha256"] = hashlib.sha256(
        b"agent-evolve:boils-exact-stack-readiness:v1\x00"
        + _canonical_bytes(commitment_input)
    ).hexdigest()
    return record


def _verify_readiness_record(
    value: Mapping[str, object],
    *,
    expected_source: Mapping[str, object],
    expected_contract: Mapping[str, object],
    production_stack_authenticated: bool,
) -> dict[str, object]:
    record = dict(value)
    supplied = record.pop("readiness_sha256", None)
    commitment_input = dict(record)
    commitment_input.pop("created_at_utc", None)
    expected = hashlib.sha256(
        b"agent-evolve:boils-exact-stack-readiness:v1\x00"
        + _canonical_bytes(commitment_input)
    ).hexdigest()
    status = (
        "ready_conformance_only"
        if production_stack_authenticated
        else "ready_offline_test_only"
    )
    if (
        supplied != expected
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != KIND
        or record.get("mode") != "readiness"
        or record.get("status") != status
        or record.get("production_stack_authenticated")
        is not production_stack_authenticated
        or record.get("source_identity") != dict(expected_source)
        or record.get("contract") != dict(expected_contract)
        or record.get("credentials_read") is not False
        or record.get("provider_client_constructed") is not False
        or record.get("provider_call_attempted") is not False
        or record.get("abc_executions") != 0
        or record.get("evaluator_call_count") != 0
        or record.get("scientific_result_eligible") is not False
        or record.get("optimization_result_eligible") is not False
    ):
        raise BoilsExactStackConformanceError("readiness is stale or invalid")
    return dict(value)


async def _execute_readiness(
    run_id: str,
    *,
    run_root: Path,
    dependencies: ConformanceDependencies,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    """Finalize a zero-credential, zero-provider, zero-ABC exact contract."""

    canonical = _validate_run_id(run_id)
    dependencies.__post_init__()
    if production_stack_authenticated and not _is_production_dependencies(
        dependencies
    ):
        raise BoilsExactStackConformanceError(
            "production readiness dependencies are not authentic"
        )
    run_dir = run_root.expanduser().resolve(strict=False) / canonical
    if run_dir.exists():
        raise FileExistsError(run_dir)
    source = _source_identity()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "source_identity.json", source)
    inputs = dependencies.inputs_factory(run_dir, str(source["aggregate_sha256"]))
    if type(inputs) is not ConformanceInputs:
        raise TypeError("inputs_factory returned a foreign value")
    contract = await _capture_contract(inputs)
    if contract["all_gates_pass"] is not True:
        raise BoilsExactStackConformanceError("readiness contract gates failed")
    record = _readiness_record(
        canonical,
        source=source,
        contract=contract,
        production_stack_authenticated=production_stack_authenticated,
    )
    write_json_atomic(run_dir / "readiness.json", record)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": record["status"],
        "production_stack_authenticated": production_stack_authenticated,
        "all_gates_pass": True,
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "abc_executions": 0,
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
    }
    write_json_atomic(run_dir / "result.json", result)
    if _source_identity() != source:
        raise BoilsExactStackConformanceError("source changed during readiness")
    finalization = finalize_run_directory(run_dir, status=str(record["status"]))
    return {
        "run_dir": str(run_dir),
        "readiness": record,
        "finalization": finalization,
    }


async def execute_readiness(
    run_id: str, *, run_root: Path = DEFAULT_RUN_ROOT
) -> dict[str, object]:
    """Public readiness path sealed to production composition identities."""

    return await _execute_readiness(
        run_id,
        run_root=run_root,
        dependencies=_production_dependencies(),
        production_stack_authenticated=True,
    )


async def _execute_readiness_for_testing(
    run_id: str,
    *,
    run_root: Path,
    dependencies: ConformanceDependencies,
) -> dict[str, object]:
    """Injected readiness whose artifacts are permanently offline-only."""

    if _is_production_dependencies(dependencies):
        raise ValueError("test readiness requires injected dependencies")
    return await _execute_readiness(
        run_id,
        run_root=run_root,
        dependencies=dependencies,
        production_stack_authenticated=False,
    )


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    value.__post_init__()
    return {
        "schema_version": 1,
        "call_id": value.call_id,
        "provider_attempt_id": value.provider_attempt_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
    }


def _queue_snapshot_record(value: object) -> dict[str, object]:
    record = {
        "schema_version": 1,
        "max_in_flight": getattr(value, "max_in_flight", None),
        "max_pending": getattr(value, "max_pending", None),
        "in_flight": getattr(value, "in_flight", None),
        "pending": getattr(value, "pending", None),
        "closed": getattr(value, "closed", None),
    }
    if (
        type(record["max_in_flight"]) is not int
        or record["max_in_flight"] < 1
        or type(record["max_pending"]) is not int
        or record["max_pending"] < 0
        or type(record["in_flight"]) is not int
        or record["in_flight"] < 0
        or type(record["pending"]) is not int
        or record["pending"] < 0
        or type(record["closed"]) is not bool
    ):
        raise BoilsExactStackConformanceError("runner returned an invalid queue snapshot")
    return record


def _expected_transport_settings() -> dict[str, object]:
    return {
        "model": campaign.MODEL,
        "provider": {"only": ["streamlake"], "allow_fallbacks": False},
        "reasoning": {"effort": "xhigh"},
        "usage": {"include": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "tool_choice": "required",
        "response_format": None,
    }


def _validate_outbound_manifest(value: Mapping[str, object]) -> dict[str, object]:
    row = validate_openrouter_outbound_request_manifest_record(value)
    settings = row["settings"]
    tool = row["tool"]
    if (
        row["call_id"] != "call_boils_exact_stack_conformance_000001"
        or row["operation"] != "select_portfolio"
        or any(settings[key] != expected for key, expected in _expected_transport_settings().items())
        or settings["max_completion_tokens"] != 384_000
        or settings.get("temperature_hex") != float(0.2).hex()
        or tool["count"] != 1
        or tool["name"] != CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
        or not all(row["forbidden_fields_absent"].values())
    ):
        raise BoilsExactStackConformanceError(
            "outbound request differs from the sealed BOiLS transport"
        )
    return row


def _load_bound_readiness(
    readiness_dir: Path,
    *,
    source: Mapping[str, object],
    production_stack_authenticated: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    root = readiness_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    expected_status = (
        "ready_conformance_only"
        if production_stack_authenticated
        else "ready_offline_test_only"
    )
    if finalization.get("status") != expected_status:
        raise BoilsExactStackConformanceError("bound readiness did not pass")
    value = decode_json_bytes((root / "readiness.json").read_bytes())
    if type(value) is not dict or type(value.get("contract")) is not dict:
        raise BoilsExactStackConformanceError("bound readiness is unreadable")
    record = _verify_readiness_record(
        value,
        expected_source=source,
        expected_contract=value["contract"],
        production_stack_authenticated=production_stack_authenticated,
    )
    return record, finalization


def _bind_live_inputs_to_readiness(
    inputs: ConformanceInputs, readiness: Mapping[str, object]
) -> None:
    contract = readiness.get("contract")
    request = None if type(contract) is not dict else contract.get("request")
    current = inputs.request
    if (
        type(request) is not dict
        or request.get("call_id") != current.call_id.value
        or request.get("request_sha256") != current.request_sha256
        or request.get("preparation_sha256") != inputs.preparation_sha256
        or request.get("finite_option_count")
        != len(current.finite_variation_contract.options)
        or request.get("card_count") != len(current.cards)
        or request.get("proposal_width") != 8
        or request.get("evaluation_width") != current.portfolio_size
        or request.get("max_output_tokens") != current.max_output_tokens
        or request.get("output_tool_name")
        != CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
    ):
        raise BoilsExactStackConformanceError(
            "live G1 request differs from bound readiness"
        )


def _validate_completed_live_call(
    result: PortfolioSelectionResult,
    *,
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    framework_versions: Mapping[str, object],
) -> dict[str, object]:
    if type(result) is not PortfolioSelectionResult:
        raise TypeError("selector returned a foreign result")
    result.__post_init__()
    if len(request_rows) != 1 or len(outcome_rows) != 1 or len(output_rows) != 1:
        raise BoilsExactStackConformanceError(
            "live call did not emit exactly one logical request/outcome/output"
        )
    if not 1 <= len(outbound_rows) <= campaign.MAX_ATTEMPTS:
        raise BoilsExactStackConformanceError("physical attempt count is invalid")
    request = validate_structured_generation_request_evidence_record(request_rows[0])
    if (
        request["call_id"] != "call_boils_exact_stack_conformance_000001"
        or request["operation"] != "select_portfolio"
        or request["output_tool_name"]
        != CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
        or request["max_output_tokens"] != 384_000
    ):
        raise BoilsExactStackConformanceError("logical request evidence drifted")
    validated_outbound = tuple(_validate_outbound_manifest(row) for row in outbound_rows)
    output = validate_structured_generation_output_evidence_record(
        output_rows[0], request_evidence=request
    )
    members = output["typed_output"].get("members")
    if type(members) is not list or len(members) != 8:
        raise BoilsExactStackConformanceError("typed provider output is not exact K8")
    outcome = dict(outcome_rows[0])
    if outcome.get("status") != "succeeded":
        raise BoilsExactStackConformanceError("terminal queue outcome is not success")
    attempts = outcome.get("attempts")
    response = outcome.get("response")
    if type(attempts) is not list or len(attempts) != len(validated_outbound):
        raise BoilsExactStackConformanceError("attempt evidence does not join")
    attempt_ids = []
    for attempt in attempts:
        evidence = attempt.get("request_evidence") if type(attempt) is dict else None
        attempt_id = (
            evidence.get("provider_attempt_id") if type(evidence) is dict else None
        )
        if type(attempt_id) is not str:
            raise BoilsExactStackConformanceError("attempt identity is absent")
        attempt_ids.append(attempt_id)
    if set(attempt_ids) != {
        row["provider_attempt_id"] for row in validated_outbound
    }:
        raise BoilsExactStackConformanceError("outbound/terminal attempts differ")
    successful_progress = [
        row for row in progress_rows if row.get("provider_attempt_id") == attempt_ids[-1]
    ]
    if (
        not successful_progress
        or successful_progress[-1].get("kind") != "stream_completed"
    ):
        raise BoilsExactStackConformanceError(
            "successful physical attempt lacks stream completion"
        )
    if (
        type(response) is not dict
        or response.get("requested_model") != campaign.MODEL
        or response.get("resolved_model") != campaign.MODEL
        or response.get("resolved_provider") != campaign.RESOLVED_PROVIDER
        or type(response.get("input_tokens")) is not int
        or response["input_tokens"] <= 0
        or type(response.get("output_tokens")) is not int
        or response["output_tokens"] <= 0
        or type(response.get("reasoning_tokens")) is not int
        or response["reasoning_tokens"] <= 0
        or response.get("cost_usd") is None
    ):
        raise BoilsExactStackConformanceError("provider telemetry is incomplete")
    audit = result.supplemental_audit
    payload = None if audit is None else thaw_json(audit.payload)
    original = (
        None if type(payload) is not dict else payload.get("original_k8_response")
    )
    if (
        len(result.decision.members) != 4
        or type(original) is not dict
        or type(original.get("members")) is not list
        or len(original["members"]) != 8
    ):
        raise BoilsExactStackConformanceError("selector result is not K8-to-K4")
    join = validate_provider_attempt_terminal_join_receipt(
        build_provider_attempt_terminal_join_receipt(
            logical_requests=request_rows,
            outbound_manifests=outbound_rows,
            terminal_outcomes=outcome_rows,
            progress_rows=progress_rows,
            explicit_pre_transport_failures=(),
            expected_framework_versions=framework_versions,
            expected_transport_settings=_expected_transport_settings(),
        )
    )
    if join["join_valid"] is not True:
        raise BoilsExactStackConformanceError("provider-attempt terminal join is red")
    return {
        "logical_call_count": 1,
        "physical_attempt_count": len(attempt_ids),
        "provider_attempt_ids": attempt_ids,
        "proposal_width": 8,
        "evaluation_width": 4,
        "selected_option_ids": [
            member.option_id for member in result.decision.members
        ],
        "response": dict(response),
        "provider_attempt_join": join,
    }


def _failure_diagnosis(
    *,
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    framework_versions: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Persist only closed queue failure evidence; never exception text/body."""

    join = validate_provider_attempt_terminal_join_receipt(
        build_provider_attempt_terminal_join_receipt(
            logical_requests=request_rows,
            outbound_manifests=outbound_rows,
            terminal_outcomes=outcome_rows,
            progress_rows=progress_rows,
            explicit_pre_transport_failures=(),
            expected_framework_versions=framework_versions,
            expected_transport_settings=_expected_transport_settings(),
        )
    )
    last_failure: object = None
    if outcome_rows:
        attempts = outcome_rows[-1].get("attempts")
        if type(attempts) is list and attempts and type(attempts[-1]) is dict:
            last_failure = attempts[-1].get("failure")
    diagnosis = {
        "request_evidence_rows": len(request_rows),
        "outbound_manifest_rows": len(outbound_rows),
        "progress_rows": len(progress_rows),
        "terminal_outcome_rows": len(outcome_rows),
        "typed_output_rows": len(output_rows),
        "last_sanitized_failure": last_failure,
        "failure_schema_supports_exception_provenance_v8": True,
        "raw_exception_text_retained": False,
        "raw_provider_body_retained": False,
    }
    return diagnosis, join


async def _execute_live(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path,
    dependencies: ConformanceDependencies,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    """Execute exactly one registered G1 selector call and seal its evidence."""

    canonical = _validate_run_id(run_id)
    if authorization != LIVE_AUTHORIZATION:
        raise BoilsExactStackConformanceError("live authorization string is invalid")
    dependencies.__post_init__()
    if production_stack_authenticated and not _is_production_dependencies(
        dependencies
    ):
        raise BoilsExactStackConformanceError(
            "production live dependencies are not authentic"
        )
    source = _source_identity()
    readiness, readiness_finalization = _load_bound_readiness(
        readiness_dir,
        source=source,
        production_stack_authenticated=production_stack_authenticated,
    )
    run_dir = run_root.expanduser().resolve(strict=False) / canonical
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "source_identity.json", source)
    write_json_atomic(run_dir / "bound_readiness.json", readiness)
    write_json_atomic(
        run_dir / "launch.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "mode": "live",
            "run_id": canonical,
            "created_at_utc": _utc(),
            "production_stack_authenticated": production_stack_authenticated,
            "bound_readiness": {
                "directory": str(readiness_dir.expanduser().resolve(strict=True)),
                "readiness": file_identity(
                    readiness_dir.expanduser().resolve(strict=True)
                    / "readiness.json"
                ),
                "finalization_sha256": readiness_finalization.get(
                    "finalization_sha256"
                ),
            },
            "maximum_logical_calls": 1,
            "abc_executions": 0,
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        },
    )
    inputs = dependencies.inputs_factory(run_dir, str(source["aggregate_sha256"]))
    if type(inputs) is not ConformanceInputs:
        raise TypeError("inputs_factory returned a foreign value")
    _bind_live_inputs_to_readiness(inputs, readiness)
    if _source_identity() != source:
        raise BoilsExactStackConformanceError("source changed before credential access")

    progress_journal = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl", max_unfsynced_rows=32
    )
    request_journal = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    outbound_journal = DurableJsonlJournal(
        run_dir / "provider_attempt_requests.jsonl"
    )
    output_journal = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    outcome_journal = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    progress_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    outbound_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    credential_reads = 0
    client_constructed = False
    provider_call_attempted = False
    runner: LiveRunner | None = None
    failure: BaseException | None = None
    preclose_snapshot: dict[str, object] | None = None
    postclose_snapshot: dict[str, object] | None = None
    result_record: dict[str, object]

    def progress_sink(value: StructuredStreamProgress) -> None:
        row = _progress_record(value)
        progress_rows.append(row)
        progress_journal.append(row)

    def request_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_request_evidence_record(value)
        request_rows.append(row)
        request_journal.append(row)

    def outbound_sink(value: Mapping[str, object]) -> None:
        row = _validate_outbound_manifest(value)
        outbound_rows.append(row)
        outbound_journal.append(row)

    def output_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_output_evidence_record(value)
        output_rows.append(row)
        output_journal.append(row)

    def outcome_sink(value: object) -> None:
        progress_journal.flush()
        row = structured_generation_outcome_record(value)  # type: ignore[arg-type]
        outcome_rows.append(row)
        outcome_journal.append(row)

    try:
        credential_reads = 1
        api_key = dependencies.credential_loader()
        if type(api_key) is not str or not api_key:
            raise BoilsExactStackConformanceError(
                "credential loader returned an empty key"
            )
        write_json_atomic(
            run_dir / "credential_access.json",
            {
                "schema_version": 1,
                "credential_name": "OPENROUTER_API_KEY",
                "read_count": 1,
                "value_persisted": False,
            },
        )
        runner = dependencies.runner_factory(
            api_key=api_key,
            config=campaign._provider_config(),
            progress_sink=progress_sink,
            request_evidence_sink=request_sink,
            outbound_request_manifest_sink=outbound_sink,
            output_evidence_sink=output_sink,
            outcome_sink=outcome_sink,
        )
        client_constructed = True
        selector = inputs.coordinator.build_selector(runner)
        provider_call_attempted = True
        result = await selector.select(inputs.request)
        completed = _validate_completed_live_call(
            result,
            request_rows=request_rows,
            outbound_rows=outbound_rows,
            progress_rows=progress_rows,
            outcome_rows=outcome_rows,
            output_rows=output_rows,
            framework_versions=_framework_versions(),
        )
        if inputs.evaluator_guard.calls != 0 or getattr(
            inputs.evaluator_observer, "calls"
        ) != 0:
            raise BoilsExactStackConformanceError("live selector invoked ABC")
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "completed_conformance_only"
                if production_stack_authenticated
                else "completed_offline_test_only"
            ),
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "abc_executions": 0,
            "evaluator_call_count": 0,
            **completed,
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        }
    except BaseException as exc:
        failure = exc
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_conformance_only",
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "failure_type": type(exc).__name__,
            "abc_executions": inputs.evaluator_guard.calls,
            "evaluator_call_count": getattr(inputs.evaluator_observer, "calls"),
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        }
    finally:
        if runner is not None:
            try:
                preclose_snapshot = _queue_snapshot_record(await runner.snapshot())
                write_json_atomic(
                    run_dir / "queue_snapshot_before_close.json",
                    preclose_snapshot,
                )
                if (
                    preclose_snapshot["pending"] != 0
                    or preclose_snapshot["in_flight"] != 0
                ):
                    raise BoilsExactStackConformanceError(
                        "queue retained work before close"
                    )
            except BaseException as snapshot_error:
                if failure is None:
                    failure = snapshot_error
                    result_record["failure_type"] = type(snapshot_error).__name__
                result_record["status"] = "failed_conformance_only"
                result_record["preclose_snapshot_failure_type"] = type(
                    snapshot_error
                ).__name__
            try:
                await runner.aclose()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                    result_record["failure_type"] = type(close_error).__name__
                result_record["status"] = "failed_conformance_only"
                result_record["runner_close_failure_type"] = type(
                    close_error
                ).__name__
            try:
                postclose_snapshot = _queue_snapshot_record(await runner.snapshot())
                write_json_atomic(
                    run_dir / "queue_snapshot_after_close.json",
                    postclose_snapshot,
                )
                if (
                    postclose_snapshot["pending"] != 0
                    or postclose_snapshot["in_flight"] != 0
                    or postclose_snapshot["closed"] is not True
                ):
                    raise BoilsExactStackConformanceError(
                        "closed queue is not empty and terminal"
                    )
            except BaseException as snapshot_error:
                if failure is None:
                    failure = snapshot_error
                    result_record["failure_type"] = type(snapshot_error).__name__
                result_record["status"] = "failed_conformance_only"
                result_record["postclose_snapshot_failure_type"] = type(
                    snapshot_error
                ).__name__
        for journal in (
            progress_journal,
            request_journal,
            outbound_journal,
            output_journal,
            outcome_journal,
        ):
            journal.close()

    if _source_identity() != source and failure is None:
        failure = BoilsExactStackConformanceError("source changed during live call")
        result_record["status"] = "failed_conformance_only"
        result_record["failure_type"] = type(failure).__name__
    diagnosis, join = _failure_diagnosis(
        request_rows=request_rows,
        outbound_rows=outbound_rows,
        progress_rows=progress_rows,
        outcome_rows=outcome_rows,
        output_rows=output_rows,
        framework_versions=_framework_versions(),
    )
    if failure is not None:
        result_record["diagnosis"] = diagnosis
    result_record["source_identity_verified_before_and_after"] = (
        _source_identity() == source
    )
    result_record["queue_cleanup"] = {
        "before_close": preclose_snapshot,
        "after_close": postclose_snapshot,
        "empty_before_close": (
            preclose_snapshot is not None
            and preclose_snapshot["pending"] == 0
            and preclose_snapshot["in_flight"] == 0
        ),
        "closed_and_empty_after_close": (
            postclose_snapshot is not None
            and postclose_snapshot["pending"] == 0
            and postclose_snapshot["in_flight"] == 0
            and postclose_snapshot["closed"] is True
        ),
    }
    result_record["provider_attempt_join"] = join
    write_json_atomic(run_dir / "provider_attempt_join.json", join)
    write_json_atomic(run_dir / "result.json", result_record)
    finalization = finalize_run_directory(run_dir, status=str(result_record["status"]))
    return {
        "run_dir": str(run_dir),
        "result": result_record,
        "finalization": finalization,
        "failed": failure is not None,
    }


async def execute_live(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, object]:
    """Public one-call path sealed to the production composition."""

    return await _execute_live(
        run_id,
        readiness_dir=readiness_dir,
        authorization=authorization,
        run_root=run_root,
        dependencies=_production_dependencies(),
        production_stack_authenticated=True,
    )


async def _execute_live_for_testing(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path,
    dependencies: ConformanceDependencies,
) -> dict[str, object]:
    """Injected live path whose artifacts remain permanently offline-only."""

    if _is_production_dependencies(dependencies):
        raise ValueError("test live execution requires injected dependencies")
    return await _execute_live(
        run_id,
        readiness_dir=readiness_dir,
        authorization=authorization,
        run_root=run_root,
        dependencies=dependencies,
        production_stack_authenticated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("readiness", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--readiness-dir", type=Path)
    parser.add_argument("--authorization")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "readiness":
            summary = asyncio.run(
                execute_readiness(args.run_id, run_root=args.run_root)
            )
        else:
            if args.readiness_dir is None:
                raise BoilsExactStackConformanceError(
                    "live mode requires --readiness-dir"
                )
            summary = asyncio.run(
                execute_live(
                    args.run_id,
                    readiness_dir=args.readiness_dir,
                    authorization=args.authorization,
                    run_root=args.run_root,
                )
            )
    except (
        BoilsExactStackConformanceError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 1 if summary.get("failed") is True else 0


if __name__ == "__main__":
    raise SystemExit(main())
