#!/usr/bin/env python3
"""Two-lane, non-scientific concurrency canary for the BOiLS G1 stack.

``readiness`` reconstructs the two production G1 parent lanes from frozen,
content-minimized parent fixtures and runs both registered calibrated K8-to-K
selectors concurrently against a provider-free capture runner.  It reads no
credential, creates no provider client, materializes no child, and executes no
ABC process.

``live`` binds a finalized readiness directory, creates the ordinary production
OpenRouter queue, and dispatches exactly those two selector calls concurrently.
It still performs no child materialization or ABC evaluation.  The command is a
transport/orchestration canary only: every artifact is permanently ineligible
for optimization, model-quality, or scientific claims.
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
import inspect
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.application.budgeted_optimizer import (  # noqa: E402
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignArchiveCutoffReceipt,
    CampaignStageRequest,
)
from agent_evolve.application.concurrent_stage import (  # noqa: E402
    gather_concurrent_stage,
)
from agent_evolve.application.detailed_evaluation import (  # noqa: E402
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
)
from agent_evolve.application.parent_measurement import (  # noqa: E402
    attach_parent_measurement_to_context,
    bind_parent_measurement,
)
from agent_evolve.application.pareto_archive import ParetoArchive  # noqa: E402
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    CAMPAIGN_ARCHIVE_CONTEXT_KEY,
    CAMPAIGN_FRONTIER_TARGET_KEY,
    CampaignPortfolioWaveContext,
    CampaignPortfolioWavePreparationReceipt,
    _validated_wave_preparation_receipt,
)
from agent_evolve.application.portfolio_outcome_feedback import (  # noqa: E402
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
)
from agent_evolve.application.agentic_evolution import (  # noqa: E402
    EvolutionCandidate,
)
from agent_evolve.domain.ids import CandidateId  # noqa: E402
from agent_evolve.domain.lineage import CandidateOccurrence  # noqa: E402
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
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    progress_aware_openrouter as _progress_aware_openrouter_module,
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
from agent_evolve.policies.selection.affine_frontier_context import (  # noqa: E402
    AuthenticatedAffineFrontierContextProjector,
)
from agent_evolve.policies.selection.affine_frontier_target import (  # noqa: E402
    AuthenticatedAffineFrontierTargetAllocator,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamProgress,
)
from examples.development import durable_run_artifacts  # noqa: E402
from examples.development import run_boils_exact_stack_conformance as one_call  # noqa: E402
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


_CAPTURED_PRODUCTION_RUNNER_CONSTRUCTOR = (
    _progress_aware_openrouter_module.create_progress_aware_openrouter_runner
)


KIND = "boils_g1_two_call_concurrency_canary"
SCHEMA_VERSION = 1
LIVE_AUTHORIZATION = "RUN_TWO_BOILS_G1_CONCURRENT_CANARY_CALLS"
DEFAULT_RUN_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/g1_two_call_concurrency_canary"
)
FRAMEWORK_PACKAGES = ("httpx", "openai", "pydantic", "pydantic-ai")
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_EXPECTED_CALL_IDS = (
    "call_boils_g6_identifiable_v1_live_000001",
    "call_boils_g6_identifiable_v1_live_000002",
)
_EXPECTED_LANE_IDS = (
    ("elite", "explorer")
    if campaign.COMMON_POOL_ACQUISITION
    else ("reservoir_0001", "reservoir_0002")
)
_EXPECTED_FINITE_OPTION_COUNT = (
    200 + campaign.VARIATION_TOPOLOGY.max_composite_options
)
_INVALID_STREAM_ITEM_SAFE_MESSAGE = "provider stream returned an invalid item"
_RUNNER_BINDING_DOMAIN = (
    b"agent-evolve:boils-g1-two-call-runner-constructor-binding:v1\x00"
)
_BOUNDARY_EVENT_DOMAIN = b"agent-evolve:boils-g1-two-call-boundary-event:v1\x00"
_BOUNDARY_EVENT_COLLECTION_DOMAIN = (
    b"agent-evolve:boils-g1-two-call-boundary-event-collection:v1\x00"
)

# These are content-minimized, frozen G1 seed observations from the sealed R4
# launch.  They let the canary reconstruct completed parent-measurement
# contracts without executing ABC.  They are fixtures, not fresh measurements,
# and no canary result may be used as benchmark evidence.
_FROZEN_PARENT_FIXTURES: tuple[dict[str, object], ...] = (
    {
        "seed_id": "seed_default",
        "candidate_id": "candidate_boils_g6_identifiable_v1_live_000001",
        "configuration_sha256": (
            "91e8e9756403130ae67423409d6e40860228a8adc4a72bce68ac97a41530f878"
        ),
        "configuration_artifact_sha256": (
            "2e53e361dfa82cbe439c29a9ea3b3ecf7c3d68dcfd20b98c1e5816c19fd7054d"
        ),
        "objectives": (("total_levels", 71.0), ("total_lut_count", 8028.0)),
    },
    {
        "seed_id": "seed_parent_c",
        "candidate_id": "candidate_boils_g6_identifiable_v1_live_000002",
        "configuration_sha256": (
            "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a"
        ),
        "configuration_artifact_sha256": (
            "78c782b594ec17b8bb0ef3471ae822ab7b85944321cce14d198979eec79f0a22"
        ),
        "objectives": (("total_levels", 69.0), ("total_lut_count", 7944.0)),
    },
)
_FROZEN_PARENT_FIXTURE_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-g1-two-call-canary-parent-fixtures:v1\x00"
    + json.dumps(
        _FROZEN_PARENT_FIXTURES,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


class BoilsG1TwoCallCanaryError(RuntimeError):
    """A content-safe two-call canary invariant failed."""


class LiveRunner(Protocol):
    async def __call__(
        self, request: StructuredGenerationRequest[Any]
    ) -> AttemptedStructuredGenerationResponse[Any]: ...

    async def snapshot(self) -> object: ...

    async def aclose(self) -> None: ...


class _SelectorOnlyIdFactoryGuard:
    """Delegate selector IDs while making child allocation fail closed.

    The canary never constructs :class:`PortfolioEvolution`; its call graph
    terminates at ``PortfolioSelectionPolicy.select``.  This proxy is still
    passed into the real production wave factory so a future refactor that
    tries to allocate a child candidate at that boundary is observed and
    rejected rather than silently invalidating the selector-only claim.
    """

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.child_candidate_allocation_attempts = 0

    def new_candidate_id(self) -> CandidateId:
        self.child_candidate_allocation_attempts += 1
        raise BoilsG1TwoCallCanaryError(
            "selector-only canary crossed the child materialization boundary"
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def to_record(self) -> dict[str, object]:
        return {
            "boundary": "selector_only_before_portfolio_evolution",
            "portfolio_evolution_constructed": False,
            "child_candidate_allocator_guard_installed": True,
            "child_candidate_allocation_attempts": (
                self.child_candidate_allocation_attempts
            ),
            "claim_scope": (
                "canary-owned selector call graph only; no claim about external "
                "or subsequent campaign materialization"
            ),
        }


@dataclass(frozen=True, slots=True)
class CanaryInputs:
    requests: tuple[PortfolioSelectionRequest, PortfolioSelectionRequest]
    coordinator: CalibratedPortfolioCampaignCoordinator
    wave_preparations: tuple[
        CampaignPortfolioWavePreparationReceipt,
        CampaignPortfolioWavePreparationReceipt,
    ]
    evaluator_guard: one_call.FailIfCalledBoilsEvaluator
    evaluator_observer: object
    materialization_guard: _SelectorOnlyIdFactoryGuard
    preparation_sha256: str
    parent_fixture_sha256: str

    def __post_init__(self) -> None:
        if type(self.requests) is not tuple or len(self.requests) != 2:
            raise ValueError("canary requires exactly two selector requests")
        if any(type(value) is not PortfolioSelectionRequest for value in self.requests):
            raise TypeError("canary requests must be exact")
        for request in self.requests:
            request.__post_init__()
        if tuple(value.call_id.value for value in self.requests) != _EXPECTED_CALL_IDS:
            raise BoilsG1TwoCallCanaryError("production G1 call identities drifted")
        if type(self.coordinator) is not CalibratedPortfolioCampaignCoordinator:
            raise TypeError("coordinator must be exact")
        if self.coordinator.registered_request_count != 2:
            raise BoilsG1TwoCallCanaryError(
                "exactly two selector requests must be registered"
            )
        if type(self.wave_preparations) is not tuple or len(self.wave_preparations) != 2:
            raise ValueError("two wave preparations are required")
        if tuple(value.parent_lane_id for value in self.wave_preparations) != (
            _EXPECTED_LANE_IDS
        ):
            raise BoilsG1TwoCallCanaryError("production parent lanes drifted")
        if self.evaluator_guard.calls != 0:
            raise BoilsG1TwoCallCanaryError("input construction invoked ABC")
        if type(self.materialization_guard) is not _SelectorOnlyIdFactoryGuard:
            raise TypeError("materialization guard must be exact")
        if self.materialization_guard.child_candidate_allocation_attempts != 0:
            raise BoilsG1TwoCallCanaryError(
                "input construction crossed child materialization"
            )
        if self.parent_fixture_sha256 != _FROZEN_PARENT_FIXTURE_SHA256:
            raise BoilsG1TwoCallCanaryError("parent fixture identity drifted")


@dataclass(frozen=True, slots=True)
class CanaryDependencies:
    inputs_factory: Callable[[Path, str], CanaryInputs]
    credential_loader: Callable[[], str]
    runner_factory: Callable[..., LiveRunner]

    def __post_init__(self) -> None:
        if not all(
            callable(getattr(self, name))
            for name in ("inputs_factory", "credential_loader", "runner_factory")
        ):
            raise TypeError("all canary dependencies must be callable")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(value: str) -> str:
    if type(value) is not str or _SAFE_RUN_ID.fullmatch(value) is None:
        raise BoilsG1TwoCallCanaryError("run_id violates the closed grammar")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "tests/test_boils_g1_two_call_canary.py",
        Path(one_call.__file__),
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


def _production_runner_constructor_binding(
    _captured: Callable[..., object] = _CAPTURED_PRODUCTION_RUNNER_CONSTRUCTOR,
) -> tuple[Callable[..., object], dict[str, object]]:
    """Authenticate the import-time production constructor binding."""

    current = _progress_aware_openrouter_module.create_progress_aware_openrouter_runner
    if current is not _captured:
        raise BoilsG1TwoCallCanaryError(
            "production runner constructor binding drifted after import"
        )
    source_file_value = inspect.getsourcefile(_captured)
    if source_file_value is None:
        raise BoilsG1TwoCallCanaryError(
            "production runner constructor has no source identity"
        )
    source_file = Path(source_file_value).expanduser().resolve(strict=True)
    try:
        source_record = file_identity(source_file, relative_to=WORKSPACE_ROOT)
    except ValueError as exc:
        raise BoilsG1TwoCallCanaryError(
            "production runner constructor is outside the source workspace"
        ) from exc
    definition = inspect.getsource(_captured).encode("utf-8", errors="strict")
    record: dict[str, object] = {
        "module": _captured.__module__,
        "qualname": _captured.__qualname__,
        "source_file": source_record,
        "definition_sha256": hashlib.sha256(definition).hexdigest(),
        "captured_at_module_import": True,
        "runtime_module_symbol_is_captured_object": True,
    }
    record["binding_identity_sha256"] = hashlib.sha256(
        _RUNNER_BINDING_DOMAIN + _canonical_bytes(record)
    ).hexdigest()
    return _captured, record


def _production_runner_factory(**kwargs: object) -> LiveRunner:
    constructor, _ = _production_runner_constructor_binding()
    kwargs["evidence_publication_policy"] = (
        StructuredEvidencePublicationPolicy.REQUIRED
    )
    return constructor(**kwargs)  # type: ignore[return-value]


def _fixture_parent(
    *,
    ordinal: int,
    seed: object,
    fixture: Mapping[str, object],
    bundle: object,
) -> EvolutionCandidate:
    configuration = seed.configuration
    configuration_sha256 = str(fixture["configuration_sha256"])
    if (
        seed.seed_id != fixture["seed_id"]
        or seed.configuration_sha256 != configuration_sha256
    ):
        raise BoilsG1TwoCallCanaryError(
            "prepared seed differs from the frozen zero-ABC parent fixture"
        )
    objectives = fixture["objectives"]
    if type(objectives) is not tuple:
        raise TypeError("fixture objectives must be an exact tuple")
    detailed = bundle.benchmark.detailed_evaluator
    if detailed is None:
        raise BoilsG1TwoCallCanaryError("parent fixture requires evaluator identity")
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(str(fixture["candidate_id"])),
            configuration_hash=configuration_sha256,
            configuration_artifact_hash=str(
                fixture["configuration_artifact_sha256"]
            ),
            proposal_sequence=ordinal,
        ),
        configuration=configuration,
        objectives=objectives,
        valid=True,
        generation=0,
        label=f"campaign_seed_{seed.seed_id}",
        design_rationale="frozen_parent_fixture_for_non_scientific_canary",
        detailed_evaluation=DetailedEvaluation(
            phenotype=bundle.benchmark.phenotype_identity.identify(
                thaw_json(configuration)
            ),
            payload=DetailedEvaluationPayload(
                failure=None,
                objectives=objectives,
                violations=(),
                checks=(),
                receipt=None,
                evaluator=detailed.evaluator_identity,
            ),
            timings=EvaluationTimings(total_wall_seconds=0.0),
        ),
    )


def build_canary_inputs(run_dir: Path, source_sha256: str) -> CanaryInputs:
    """Build both production-shaped G1 parent lanes without provider or ABC."""

    preparation = DurableJsonlJournal(run_dir / "campaign_preparation.jsonl")
    evaluator = DurableJsonlJournal(run_dir / "evaluator_observations.jsonl")
    try:
        bundle = campaign._prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator,
            source_closure_sha256=source_sha256,
            arm="live",
            evaluator_factory=one_call.FailIfCalledBoilsEvaluator,
        )
        guard = bundle.evaluator
        if type(guard) is not one_call.FailIfCalledBoilsEvaluator:
            raise TypeError("BOiLS preparation replaced the evaluator guard")
        seeds = bundle.prepared.seeds.seeds
        if len(seeds) != len(_FROZEN_PARENT_FIXTURES):
            raise BoilsG1TwoCallCanaryError("prepared G1 seed cardinality drifted")
        parents = tuple(
            _fixture_parent(
                ordinal=ordinal,
                seed=seed,
                fixture=fixture,
                bundle=bundle,
            )
            for ordinal, (seed, fixture) in enumerate(
                zip(seeds, _FROZEN_PARENT_FIXTURES, strict=True), start=1
            )
        )
        archive = ParetoArchive(
            bundle.benchmark.objectives,
            outcome_relation_binding=bundle.benchmark.outcome_relation,
        )
        for parent in parents:
            archive.consider(parent)
        snapshot = archive.snapshot()
        state = OptimizerState(
            generation=0,
            candidates=parents,
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=2,
            logical_llm_calls=0,
        )
        step = next(
            value for value in bundle.prepared.schedule.steps if value.generation == 1
        )
        selection = bundle.parent_selector.select(
            state,
            task_sha256=campaign.TASK_SHA256,
            parent_count=step.parent_count,
            rotation_index=0,
        )
        if tuple(value.lane_id for value in selection.lanes) != _EXPECTED_LANE_IDS:
            raise BoilsG1TwoCallCanaryError("G1 parent selector lanes drifted")
        observed_parent_ids = tuple(
            value.candidate_id.value for value in selection.parents
        )
        if len(set(observed_parent_ids)) != 2 or set(observed_parent_ids) != {
            str(value["candidate_id"]) for value in _FROZEN_PARENT_FIXTURES
        }:
            raise BoilsG1TwoCallCanaryError(
                f"G1 parent fixture coverage drifted: {observed_parent_ids!r}"
            )

        session = bundle.prepared.benchmark_session
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
        archive_record = campaign._object(
            {
                "canary_only": True,
                "frozen_parent_fixture_sha256": _FROZEN_PARENT_FIXTURE_SHA256,
                "front_candidates": [
                    {
                        "objectives": [
                            {"metric_id": name, "value_hex": value.hex()}
                            for name, value in parent.objectives
                        ]
                    }
                    for parent in snapshot.front_candidates
                ],
            }
        )
        utility = bundle.utility.freeze(
            benchmark=session.benchmark,
            generation=1,
            archive=archive_record,
        )
        stage = CampaignStageRequest(
            preparation_sha256=bundle.prepared.preparation_sha256,
            runtime_start_receipt_sha256=campaign._sha(
                "boils-g1-two-call-canary-runtime-start"
            ),
            step=step,
            archive_cutoff=CampaignArchiveCutoffReceipt(
                request_sha256=campaign._sha("boils-g1-two-call-canary-cutoff"),
                preparation_sha256=bundle.prepared.preparation_sha256,
                generation=1,
                archive=archive_record,
                evidence=campaign._object(
                    {
                        "canary_only": True,
                        "abc_executions": 0,
                        "scientific_result_eligible": False,
                    }
                ),
            ),
            archive_utility=utility,
            source_portfolio=None,
            test_eligible_reflection_receipt_sha256s=(),
            prior_selector_audit_set_sha256=campaign._sha(
                "boils-g1-two-call-canary-prior-audit"
            ),
        )
        binding_factory = replace(
            bundle.binding_factory,
            ledger=PortfolioOutcomeFeedbackLedger(),
        )
        coordinator = CalibratedPortfolioCampaignCoordinator(
            allocator=campaign._allocator()
        )
        target_conditioned_controller = campaign._target_conditioned_controller(
            bundle,
            coordinator,
        )
        materialization_guard = _SelectorOnlyIdFactoryGuard(bundle.ids)
        factory = campaign._WaveFactory(
            bundle=bundle,
            learning_runtime=None,
            records=[],
            ids=materialization_guard,
            binding_factory=binding_factory,
            coordinator=coordinator,
            target_conditioned_controller=target_conditioned_controller,
        )
        lanes_by_id = {value.lane_id: value for value in selection.lanes}
        builds: list[CampaignPortfolioWaveContext] = []
        context_hashes: list[tuple[str, str]] = []
        for parent_slot, decision_slot in enumerate(selection.decision_slots):
            lane = lanes_by_id[decision_slot.lane_id]
            parent = lane.parent
            variation = bundle.workload_ports.catalog.bind(
                session.benchmark, parent.configuration, known
            )
            workload_context = bundle.workload_ports.evidence.context(
                session,
                parent.configuration,
                variation,
                memory,
            )
            workload_context_sha256 = campaign.typed_json_sha256(workload_context)
            parent_measurement = bind_parent_measurement(
                candidate=parent,
                variation=variation,
                projection=bundle.parent_measurement_projection,
            )
            selector_context = attach_parent_measurement_to_context(
                workload_context, parent_measurement
            )
            pre_memory_projection_context_sha256 = campaign.typed_json_sha256(
                selector_context
            )
            cards = bundle.workload_ports.evidence.cards(
                session,
                parent.configuration,
                variation,
                memory,
            )
            builds.append(
                CampaignPortfolioWaveContext(
                    prepared=bundle.prepared,
                    stage_request=stage,
                    parent_slot=parent_slot,
                    parent=parent,
                    variation=variation,
                    evidence_context=selector_context,
                    evidence_cards=cards,
                    memory=memory,
                    parent_measurement=parent_measurement,
                    parent_lane=lane,
                    decision_slot=decision_slot,
                )
            )
            context_hashes.append(
                (workload_context_sha256, pre_memory_projection_context_sha256)
            )
        if target_conditioned_controller is not None:
            targets = AuthenticatedAffineFrontierTargetAllocator().allocate(
                archive_utility=utility,
                lanes=tuple(
                    (value.parent_lane.lane_id, value.parent)
                    for value in builds
                ),
            )
            targets_by_lane = {value.lane_id: value for value in targets}
            projector = AuthenticatedAffineFrontierContextProjector()
            rebound: list[CampaignPortfolioWaveContext] = []
            for value in builds:
                projection = projector.project(
                    archive_utility=utility,
                    parent=value.parent,
                )
                target = targets_by_lane[value.parent_lane.lane_id]
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("canary selector context must be an object")
                evidence[CAMPAIGN_ARCHIVE_CONTEXT_KEY] = projection.to_record()
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
                rebound.append(
                    replace(
                        value,
                        evidence_context=campaign._object(evidence),
                        archive_context=projection,
                        frontier_target=target,
                    )
                )
            builds = rebound
        waves = factory.build_batch(tuple(builds))
        if type(waves) is not tuple or len(waves) != 2:
            raise BoilsG1TwoCallCanaryError("G1 batch factory did not build two waves")
        receipts = tuple(
            _validated_wave_preparation_receipt(
                build=build,
                wave=wave,
                workload_context_sha256=hashes[0],
                pre_memory_projection_context_sha256=hashes[1],
            )
            for build, wave, hashes in zip(
                builds, waves, context_hashes, strict=True
            )
        )
        result = CanaryInputs(
            requests=tuple(value.selection_request for value in waves),  # type: ignore[arg-type]
            coordinator=coordinator,
            wave_preparations=receipts,  # type: ignore[arg-type]
            evaluator_guard=guard,
            evaluator_observer=bundle.evaluator_observer,
            materialization_guard=materialization_guard,
            preparation_sha256=bundle.prepared.preparation_sha256,
            parent_fixture_sha256=_FROZEN_PARENT_FIXTURE_SHA256,
        )
        if bundle.evaluator_observer.calls != 0:
            raise BoilsG1TwoCallCanaryError(
                "input construction emitted an evaluator observation"
            )
        return result
    finally:
        preparation.close()
        evaluator.close()


_CANONICAL_DEPENDENCIES = (
    build_canary_inputs,
    one_call._read_openrouter_api_key,
    _production_runner_factory,
)


def _production_dependencies() -> CanaryDependencies:
    _production_runner_constructor_binding()
    return CanaryDependencies(*_CANONICAL_DEPENDENCIES)


def _is_production_dependencies(value: CanaryDependencies) -> bool:
    return (
        type(value) is CanaryDependencies
        and value.inputs_factory is _CANONICAL_DEPENDENCIES[0]
        and value.credential_loader is _CANONICAL_DEPENDENCIES[1]
        and value.runner_factory is _CANONICAL_DEPENDENCIES[2]
    )


def _proposal_members(
    request: PortfolioSelectionRequest,
    low_request: StructuredGenerationRequest[Any],
) -> tuple[dict[str, object], ...]:
    """Choose one schema-visible feasible K8 without evaluating a child."""

    schema = low_request.output_type.model_json_schema()
    definitions = schema.get("$defs", {})
    hierarchical = "HierarchicalCalibratedAtomicMember" in definitions
    if hierarchical:
        atomic_ids = tuple(
            definitions["HierarchicalCalibratedAtomicMember"]["properties"][
                "option_id"
            ]["enum"]
        )
        composite_ids = tuple(
            definitions["HierarchicalCalibratedCompositeMember"]["properties"][
                "composite_option_id"
            ]["enum"]
        )
        required_composites = (
            campaign.VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
        )
    else:
        atomic_ids = tuple(
            definitions["CalibratedPortfolioSlateMember"]["properties"][
                "option_id"
            ]["enum"]
        )
        composite_ids = ()
        required_composites = 0
    atomic_target = 8 - required_composites
    required_support = tuple(
        sorted(
            getattr(
                low_request.output_type,
                "required_proposal_support_option_ids",
                frozenset(),
            )
        )
    )
    options_by_id = {
        value.option_id: value
        for value in request.finite_variation_contract.options
    }
    by_position: dict[str, list[Any]] = {}
    for option_id in atomic_ids:
        option = options_by_id[option_id]
        metadata = dict(option.metadata)
        position = metadata.get("position")
        if position is None:
            raise BoilsG1TwoCallCanaryError(
                "schema-visible atomic option lacks a BOiLS position"
            )
        by_position.setdefault(str(position), []).append(option)
    required_atomics = tuple(
        options_by_id[value] for value in required_support if value in atomic_ids
    )
    if len(required_atomics) > atomic_target:
        raise BoilsG1TwoCallCanaryError(
            "proposal support exceeds the atomic proposal stratum"
        )
    selected_atomics: list[Any] = list(required_atomics)
    used_families: set[str] = {value.family for value in selected_atomics}
    used_positions = {
        str(dict(value.metadata)["position"]) for value in selected_atomics
    }
    for position in sorted(by_position):
        if position in used_positions:
            continue
        options = sorted(
            by_position[position], key=lambda value: (value.family, value.option_id)
        )
        option = next(
            (value for value in options if value.family not in used_families),
            options[0],
        )
        selected_atomics.append(option)
        used_families.add(option.family)
        used_positions.add(position)
        if len(selected_atomics) == atomic_target:
            break
    if len(selected_atomics) != atomic_target or len(used_families) < 3:
        raise BoilsG1TwoCallCanaryError(
            "provider-free capture could not form its feasible atomic stratum"
        )
    required_composite_ids = tuple(
        value for value in required_support if value in composite_ids
    )
    selected_composites = tuple(
        [*required_composite_ids]
        + [
            value
            for value in composite_ids
            if value not in required_composite_ids
        ][: required_composites - len(required_composite_ids)]
    )
    if len(selected_composites) != required_composites:
        raise BoilsG1TwoCallCanaryError(
            "provider-free capture lacks required hierarchical composites"
        )
    members: list[dict[str, object]] = []
    for option in selected_atomics:
        member: dict[str, object] = {"option_id": option.option_id}
        if hierarchical:
            member["action_kind"] = "atomic"
        members.append(member)
    for option_id in selected_composites:
        metadata = dict(options_by_id[option_id].metadata)
        components = tuple(
            sorted(
                (
                    str(metadata["left_option_id"]),
                    str(metadata["right_option_id"]),
                )
            )
        )
        members.append(
            {
                "action_kind": "compose_r2",
                "composite_option_id": option_id,
                "component_option_ids": list(components),
            }
        )
    return tuple(members)


class _ConcurrentProviderFreeCaptureRunner:
    def __init__(
        self,
        requests: tuple[PortfolioSelectionRequest, PortfolioSelectionRequest],
    ) -> None:
        self._requests = {value.call_id.value: value for value in requests}
        self._both_arrived = asyncio.Event()
        self._arrived = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.low_requests: dict[str, StructuredGenerationRequest[Any]] = {}
        self.proposals: dict[str, tuple[str, ...]] = {}

    async def __call__(
        self, request: StructuredGenerationRequest[Any]
    ) -> StructuredGenerationResponse[Any]:
        call_id = request.call_id.value
        source = self._requests.get(call_id)
        if source is None or call_id in self.low_requests:
            raise BoilsG1TwoCallCanaryError("capture received an unknown/duplicate call")
        self.low_requests[call_id] = request
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self._arrived += 1
        if self._arrived == 2:
            self._both_arrived.set()
        try:
            await asyncio.wait_for(self._both_arrived.wait(), timeout=2.0)
            proposal_members = _proposal_members(source, request)
            self.proposals[call_id] = tuple(
                str(
                    value.get("option_id", value.get("composite_option_id"))
                )
                for value in proposal_members
            )
            card_key = source.cards[0].card_key
            members = []
            for index, proposal in enumerate(proposal_members, start=1):
                members.append(
                    {
                    **proposal,
                    "supporting_card_keys": [card_key] if index == 1 else [],
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
                        f"Provider-free two-call canary member {index}."
                    ),
                    }
                )
            value = request.output_type.model_validate(
                {"members": members}, strict=True
            )
            return StructuredGenerationResponse(
                value=value,
                requested_model="provider-free/boils-two-call-canary",
                resolved_model="provider-free/boils-two-call-canary",
                resolved_provider="provider-free",
                provider_response_id=f"provider-free-{call_id}",
                finish_reason="tool_calls",
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=1,
            )
        finally:
            self.in_flight -= 1


def _schema_contract(
    low_request: StructuredGenerationRequest[Any],
    contract_option_ids: tuple[str, ...],
    proposed_option_ids: tuple[str, ...],
) -> dict[str, object]:
    schema = low_request.output_type.model_json_schema()
    members = schema.get("properties", {}).get("members", {})
    definitions = schema.get("$defs", {})
    if "HierarchicalCalibratedAtomicMember" in definitions:
        atomic_enum = definitions["HierarchicalCalibratedAtomicMember"][
            "properties"
        ]["option_id"]["enum"]
        composite_enum = definitions["HierarchicalCalibratedCompositeMember"][
            "properties"
        ]["composite_option_id"]["enum"]
        enum = [*atomic_enum, *composite_enum]
    else:
        enum = definitions.get("CalibratedPortfolioSlateMember", {}).get(
            "properties", {}
        ).get("option_id", {}).get("enum")
    expected_visible_count = (
        len(contract_option_ids)
        if (
            not campaign.COMMON_POOL_ACQUISITION
            or campaign.COMMON_CANDIDATE_POOL_SIZE is None
        )
        else campaign.COMMON_CANDIDATE_POOL_SIZE
    )
    gates = {
        "members_exact_k8": (
            members.get("minItems") == 8 and members.get("maxItems") == 8
        ),
        "option_enum_is_closed_contract_subset": (
            type(enum) is list
            and len(enum) == expected_visible_count
            and len(set(enum)) == len(enum)
            and set(enum).issubset(contract_option_ids)
            and set(proposed_option_ids).issubset(enum)
        ),
    }
    return {
        "logical_schema_sha256": _sha(schema),
        "logical_schema_utf8_bytes": len(_canonical_bytes(schema)),
        "gates": gates,
    }


async def _capture_contract(inputs: CanaryInputs) -> dict[str, object]:
    """Exercise both real registered selectors concurrently, provider-free."""

    capture = _ConcurrentProviderFreeCaptureRunner(inputs.requests)
    selector = inputs.coordinator.build_selector(capture)
    results = await gather_concurrent_stage(
        selector.select(request) for request in inputs.requests
    )
    if any(type(value) is not PortfolioSelectionResult for value in results):
        raise BoilsG1TwoCallCanaryError("capture returned a foreign result")
    if set(capture.low_requests) != set(_EXPECTED_CALL_IDS):
        raise BoilsG1TwoCallCanaryError("capture did not execute both calls")
    config = campaign._provider_config()
    provider = config.to_manifest_record()
    _, runner_constructor_binding = _production_runner_constructor_binding()
    lanes: list[dict[str, object]] = []
    all_schema_gates: list[bool] = []
    for request, preparation, result in zip(
        inputs.requests, inputs.wave_preparations, results, strict=True
    ):
        call_id = request.call_id.value
        low_request = capture.low_requests[call_id]
        options = request.finite_variation_contract.options
        option_ids = tuple(value.option_id for value in options)
        schema = _schema_contract(
            low_request,
            option_ids,
            capture.proposals[call_id],
        )
        all_schema_gates.extend(schema["gates"].values())
        audit = result.supplemental_audit
        payload = None if audit is None else thaw_json(audit.payload)
        original = (
            None if type(payload) is not dict else payload.get("original_k8_response")
        )
        if (
            type(original) is not dict
            or type(original.get("members")) is not list
            or len(original["members"]) != 8
            or len(result.decision.members) != request.portfolio_size
        ):
            raise BoilsG1TwoCallCanaryError(
                "capture result does not match its sealed K8-to-K contract"
            )
        lanes.append(
            {
                "parent_slot": preparation.parent_slot,
                "parent_lane_id": preparation.parent_lane_id,
                "decision_slot_id": preparation.decision_slot_id,
                "parent_candidate_id": preparation.parent_candidate_id,
                "call_id": call_id,
                "request_sha256": request.request_sha256,
                "preparation_sha256": inputs.preparation_sha256,
                "finite_contract_identity_sha256": (
                    request.finite_variation_contract.identity_sha256
                ),
                "finite_option_count": len(options),
                "finite_option_ids_sha256": _sha(option_ids),
                "card_count": len(request.cards),
                "proposal_width": 8,
                "evaluation_width": request.portfolio_size,
                "prompt_sha256": hashlib.sha256(
                    low_request.prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_utf8_bytes": len(low_request.prompt.encode("utf-8")),
                "output_tool_name": low_request.output_tool_name,
                "max_output_tokens": low_request.max_output_tokens,
                "temperature_hex": (
                    None
                    if low_request.temperature is None
                    else float(low_request.temperature).hex()
                ),
                "schema": {
                    key: value for key, value in schema.items() if key != "gates"
                },
                "schema_gates": schema["gates"],
                "provider_free_proposed_option_ids": list(
                    capture.proposals[call_id]
                ),
                "provider_free_selected_option_ids": [
                    member.option_id for member in result.decision.members
                ],
            }
        )
    gates = {
        "two_registered_g1_requests": inputs.coordinator.registered_request_count == 2,
        "exact_production_call_ids": tuple(value["call_id"] for value in lanes)
        == _EXPECTED_CALL_IDS,
        "exact_parent_lanes": tuple(value["parent_lane_id"] for value in lanes)
        == _EXPECTED_LANE_IDS,
        "both_calls_overlapped": capture.peak_in_flight == 2,
        "both_calls_completed": len(results) == 2,
        "distinct_request_hashes": len({value["request_sha256"] for value in lanes})
        == 2,
        "distinct_prompt_hashes": len({value["prompt_sha256"] for value in lanes})
        == 2,
        "distinct_schema_hashes": len(
            {value["schema"]["logical_schema_sha256"] for value in lanes}
        )
        == 2,
        "exact_expected_options_per_lane": all(
            value["finite_option_count"] == _EXPECTED_FINITE_OPTION_COUNT
            for value in lanes
        ),
        "exact_one_bootstrap_card_per_lane": all(
            value["card_count"] == 1 for value in lanes
        ),
        "k8_to_sealed_evaluation_width_per_lane": all(
            value["proposal_width"] == 8
            and type(value["evaluation_width"]) is int
            and 1 <= value["evaluation_width"] <= value["proposal_width"]
            for value in lanes
        ),
        "all_schema_gates": all(all_schema_gates),
        "exact_tool_per_lane": all(
            value["output_tool_name"] == CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
            for value in lanes
        ),
        "max_output_tokens_match_profile": all(
            value["max_output_tokens"]
            == campaign.MODEL_EXECUTION_PROFILE.max_output_tokens
            for value in lanes
        ),
        "temperature_matches_profile": all(
            value["temperature_hex"]
            == (
                None
                if campaign.MODEL_EXECUTION_PROFILE.temperature is None
                else campaign.MODEL_EXECUTION_PROFILE.temperature.hex()
            )
            for value in lanes
        ),
        "production_model_alias": provider.get("model_name") == campaign.MODEL,
        "profile_provider_only": provider.get("provider_options", {}).get("only")
        == list(campaign.MODEL_EXECUTION_PROFILE.provider_only),
        "no_provider_fallback": provider.get("provider_options", {}).get(
            "allow_fallbacks"
        )
        is False,
        "reasoning_matches_profile": provider.get("reasoning")
        == campaign.MODEL_EXECUTION_PROFILE.outbound_reasoning_setting,
        "reasoning_mode_absent": "mode"
        not in (provider.get("reasoning") or {}),
        "queue_concurrency_supports_two": (
            provider.get("queue", {}).get("max_in_flight")
            == campaign.AGENT_CONCURRENCY
            and campaign.AGENT_CONCURRENCY >= 2
            and provider.get("queue", {}).get("max_pending")
            == campaign.AGENT_QUEUE_CAPACITY
            and provider.get("queue", {}).get("max_attempts")
            == campaign.MAX_ATTEMPTS
        ),
        "evaluator_never_called": inputs.evaluator_guard.calls == 0,
        "evaluator_observer_empty": getattr(inputs.evaluator_observer, "calls") == 0,
        "production_runner_constructor_binding_authenticated": (
            runner_constructor_binding[
                "runtime_module_symbol_is_captured_object"
            ]
            is True
        ),
        "selector_only_materialization_guard_clean": (
            inputs.materialization_guard.child_candidate_allocation_attempts == 0
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_fixture": {
            "identity_sha256": inputs.parent_fixture_sha256,
            "kind": "frozen_prior_measurement_contract_fixture",
            "fresh_measurement": False,
            "scientific_reuse_permitted": False,
        },
        "lanes": lanes,
        "provider_config": provider,
        "production_runner_constructor_binding": runner_constructor_binding,
        "framework_versions": _framework_versions(),
        "provider_free_peak_in_flight": capture.peak_in_flight,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "abc_executions": 0,
        "child_materialization_boundary": (
            inputs.materialization_guard.to_record()
        ),
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
        "model_quality_result_eligible": False,
    }


_READINESS_DOMAIN = b"agent-evolve:boils-g1-two-call-canary-readiness:v1\x00"


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
            "ready_two_call_canary_only"
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
        "logical_call_count": 0,
        "abc_executions": 0,
        "child_materialization_boundary": contract[
            "child_materialization_boundary"
        ],
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
        "model_quality_result_eligible": False,
    }
    commitment = dict(record)
    commitment.pop("created_at_utc")
    record["readiness_sha256"] = hashlib.sha256(
        _READINESS_DOMAIN + _canonical_bytes(commitment)
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
    commitment = dict(record)
    commitment.pop("created_at_utc", None)
    expected = hashlib.sha256(
        _READINESS_DOMAIN + _canonical_bytes(commitment)
    ).hexdigest()
    expected_status = (
        "ready_two_call_canary_only"
        if production_stack_authenticated
        else "ready_offline_test_only"
    )
    if (
        supplied != expected
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != KIND
        or record.get("mode") != "readiness"
        or record.get("status") != expected_status
        or record.get("production_stack_authenticated")
        is not production_stack_authenticated
        or record.get("source_identity") != dict(expected_source)
        or record.get("contract") != dict(expected_contract)
        or record.get("credentials_read") is not False
        or record.get("provider_client_constructed") is not False
        or record.get("provider_call_attempted") is not False
        or record.get("logical_call_count") != 0
        or record.get("abc_executions") != 0
        or record.get("child_materialization_boundary")
        != expected_contract.get("child_materialization_boundary")
        or record.get("evaluator_call_count") != 0
        or record.get("scientific_result_eligible") is not False
        or record.get("optimization_result_eligible") is not False
        or record.get("model_quality_result_eligible") is not False
    ):
        raise BoilsG1TwoCallCanaryError("readiness is stale or invalid")
    return dict(value)


async def _execute_readiness(
    run_id: str,
    *,
    run_root: Path,
    dependencies: CanaryDependencies,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    """Seal both provider-free G1 lane contracts without crossing ABC."""

    canonical = _validate_run_id(run_id)
    dependencies.__post_init__()
    if production_stack_authenticated and not _is_production_dependencies(
        dependencies
    ):
        raise BoilsG1TwoCallCanaryError(
            "production readiness dependencies are not authentic"
        )
    if production_stack_authenticated:
        _production_runner_constructor_binding()
    run_dir = run_root.expanduser().resolve(strict=False) / canonical
    if run_dir.exists():
        raise FileExistsError(run_dir)
    source = _source_identity()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "source_identity.json", source)
    inputs = dependencies.inputs_factory(run_dir, str(source["aggregate_sha256"]))
    if type(inputs) is not CanaryInputs:
        raise TypeError("inputs_factory returned a foreign value")
    contract = await _capture_contract(inputs)
    if contract["all_gates_pass"] is not True:
        failed_gates = tuple(
            sorted(
                key
                for key, value in contract["gates"].items()
                if value is not True
            )
        )
        raise BoilsG1TwoCallCanaryError(
            f"readiness contract gates failed: {failed_gates!r}"
        )
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
        "provider_free_logical_call_count": 2,
        "provider_free_peak_in_flight": contract["provider_free_peak_in_flight"],
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "abc_executions": 0,
        "child_materialization_boundary": contract[
            "child_materialization_boundary"
        ],
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
        "model_quality_result_eligible": False,
    }
    write_json_atomic(run_dir / "result.json", result)
    if _source_identity() != source:
        raise BoilsG1TwoCallCanaryError("source changed during readiness")
    finalization = finalize_run_directory(run_dir, status=str(record["status"]))
    return {
        "run_dir": str(run_dir),
        "readiness": record,
        "finalization": finalization,
    }


async def execute_readiness(
    run_id: str, *, run_root: Path = DEFAULT_RUN_ROOT
) -> dict[str, object]:
    """Public readiness path sealed to the production composition."""

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
    dependencies: CanaryDependencies,
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
        raise BoilsG1TwoCallCanaryError("runner returned an invalid queue snapshot")
    return record


def _expected_transport_settings() -> dict[str, object]:
    config = campaign._provider_config()
    native_json = config.structured_output_mode.value == "native_json_schema"
    return {
        "model": config.model_name,
        "provider": dict(config.provider_options),
        "reasoning": (
            None
            if config.reasoning_config is None
            else config.reasoning_config.to_model_setting()
        ),
        "usage": {"include": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "tool_choice": (
            None
            if native_json
            else "required" if config.supports_forced_tool_choice else "auto"
        ),
        "response_format": "json_schema" if native_json else None,
    }


def _validate_outbound_manifest(value: Mapping[str, object]) -> dict[str, object]:
    row = validate_openrouter_outbound_request_manifest_record(value)
    settings = row["settings"]
    tool = row["tool"]
    profile = campaign.MODEL_EXECUTION_PROFILE
    native_json = profile.structured_output_mode.value == "native_json_schema"
    if (
        row["call_id"] not in _EXPECTED_CALL_IDS
        or row["operation"] != "select_portfolio"
        or any(
            settings[key] != expected
            for key, expected in _expected_transport_settings().items()
        )
        or settings["output_mode"] != profile.structured_output_mode.value
        or settings["max_completion_tokens"] != profile.max_output_tokens
        or settings.get("temperature_hex")
        != (
            None
            if profile.temperature is None
            else profile.temperature.hex()
        )
        or tool["count"] != (0 if native_json else 1)
        or tool["type"]
        != ("native_json_schema" if native_json else "function")
        or tool["name"] != CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
        or not all(row["forbidden_fields_absent"].values())
    ):
        raise BoilsG1TwoCallCanaryError(
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
        "ready_two_call_canary_only"
        if production_stack_authenticated
        else "ready_offline_test_only"
    )
    if finalization.get("status") != expected_status:
        raise BoilsG1TwoCallCanaryError("bound readiness did not pass")
    value = decode_json_bytes((root / "readiness.json").read_bytes())
    if type(value) is not dict or type(value.get("contract")) is not dict:
        raise BoilsG1TwoCallCanaryError("bound readiness is unreadable")
    record = _verify_readiness_record(
        value,
        expected_source=source,
        expected_contract=value["contract"],
        production_stack_authenticated=production_stack_authenticated,
    )
    return record, finalization


async def _bind_live_inputs_to_readiness(
    inputs: CanaryInputs, readiness: Mapping[str, object]
) -> dict[str, object]:
    expected = readiness.get("contract")
    current = await _capture_contract(inputs)
    if type(expected) is not dict or current != expected:
        raise BoilsG1TwoCallCanaryError(
            "live G1 parent-lane requests differ from bound readiness"
        )
    return current


_BOUNDARY_EVENT_KINDS = frozenset(
    {
        "logical_dispatch_started",
        "logical_request_evidence",
        "http_outbound_hook",
        "stream_progress",
        "typed_output_evidence",
        "terminal_outcome",
        "logical_dispatch_terminal",
    }
)


def _boundary_event_record(
    *,
    sequence: int,
    monotonic_ns: int,
    boundary: str,
    call_id: str,
    provider_attempt_id: str | None,
) -> dict[str, object]:
    if type(sequence) is not int or sequence <= 0:
        raise ValueError("boundary event sequence must be positive")
    if type(monotonic_ns) is not int or monotonic_ns < 0:
        raise ValueError("boundary event monotonic_ns must be non-negative")
    if boundary not in _BOUNDARY_EVENT_KINDS:
        raise ValueError("boundary event kind is outside the closed vocabulary")
    if call_id not in _EXPECTED_CALL_IDS:
        raise ValueError("boundary event names a foreign call")
    if boundary in {"http_outbound_hook", "stream_progress"}:
        if type(provider_attempt_id) is not str or not provider_attempt_id:
            raise ValueError("physical boundary event requires an attempt identity")
    elif provider_attempt_id is not None:
        raise ValueError("logical boundary event cannot carry an attempt identity")
    record: dict[str, object] = {
        "schema_version": 1,
        "sequence": sequence,
        "monotonic_ns": monotonic_ns,
        "boundary": boundary,
        "call_id": call_id,
        "provider_attempt_id": provider_attempt_id,
        "raw_provider_content_persisted": False,
    }
    record["event_sha256"] = hashlib.sha256(
        _BOUNDARY_EVENT_DOMAIN + _canonical_bytes(record)
    ).hexdigest()
    return record


class _BoundaryEventLedger:
    """One globally ordered durable clock for every live evidence boundary."""

    def __init__(self, journal: DurableJsonlJournal) -> None:
        self._journal = journal
        self._lock = threading.Lock()
        self._last_monotonic_ns = -1
        self.rows: list[dict[str, object]] = []

    def record(
        self,
        boundary: str,
        *,
        call_id: str,
        provider_attempt_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            observed = time.monotonic_ns()
            monotonic_ns = max(observed, self._last_monotonic_ns + 1)
            row = _boundary_event_record(
                sequence=len(self.rows) + 1,
                monotonic_ns=monotonic_ns,
                boundary=boundary,
                call_id=call_id,
                provider_attempt_id=provider_attempt_id,
            )
            self._journal.append(row)
            self.rows.append(row)
            self._last_monotonic_ns = monotonic_ns
            return row


def _validate_boundary_event_order(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Authenticate global order and the two-before-any-response invariant."""

    if not rows:
        raise BoilsG1TwoCallCanaryError("boundary event ledger is empty")
    detached: list[dict[str, object]] = []
    for index, value in enumerate(rows, start=1):
        row = dict(value)
        try:
            expected = _boundary_event_record(
                sequence=row["sequence"],  # type: ignore[arg-type]
                monotonic_ns=row["monotonic_ns"],  # type: ignore[arg-type]
                boundary=row["boundary"],  # type: ignore[arg-type]
                call_id=row["call_id"],  # type: ignore[arg-type]
                provider_attempt_id=row["provider_attempt_id"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BoilsG1TwoCallCanaryError(
                "boundary event ledger contains a malformed row"
            ) from exc
        if (
            row != expected
            or value.get("sequence") != index
        ):
            raise BoilsG1TwoCallCanaryError(
                "boundary event ledger authentication failed"
            )
        detached.append(dict(value))
    times = [value["monotonic_ns"] for value in detached]
    if any(
        type(value) is not int for value in times
    ) or any(left >= right for left, right in zip(times, times[1:])):
        raise BoilsG1TwoCallCanaryError(
            "boundary event monotonic clock is not strictly increasing"
        )
    first_outbound: dict[str, dict[str, object]] = {}
    for value in detached:
        if value["boundary"] == "http_outbound_hook":
            first_outbound.setdefault(str(value["call_id"]), value)
    if set(first_outbound) != set(_EXPECTED_CALL_IDS):
        raise BoilsG1TwoCallCanaryError(
            "both first HTTP outbound hooks were not observed"
        )
    response_boundaries = [
        value
        for value in detached
        if value["boundary"] in {"stream_progress", "terminal_outcome"}
    ]
    if not response_boundaries:
        raise BoilsG1TwoCallCanaryError(
            "boundary ledger lacks progress or a terminal outcome"
        )
    latest_first_outbound = max(
        int(value["sequence"]) for value in first_outbound.values()
    )
    earliest_response = min(
        int(value["sequence"]) for value in response_boundaries
    )
    if latest_first_outbound >= earliest_response:
        raise BoilsG1TwoCallCanaryError(
            "progress/outcome preceded both first HTTP outbound hooks"
        )
    collection = hashlib.sha256(_BOUNDARY_EVENT_COLLECTION_DOMAIN)
    for value in detached:
        payload = _canonical_bytes(value)
        collection.update(len(payload).to_bytes(8, "big"))
        collection.update(payload)
    return {
        "schema_version": 1,
        "event_count": len(detached),
        "first_http_outbound_sequence_by_call": {
            call_id: first_outbound[call_id]["sequence"]
            for call_id in _EXPECTED_CALL_IDS
        },
        "earliest_progress_or_terminal_outcome_sequence": earliest_response,
        "both_first_http_outbound_before_any_progress_or_terminal_outcome": True,
        "strict_sequence_order": True,
        "strict_monotonic_ns_order": True,
        "collection_sha256": collection.hexdigest(),
    }


def _validate_boundary_evidence_join(
    rows: Sequence[Mapping[str, object]],
    *,
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    receipt = _validate_boundary_event_order(rows)
    counts: dict[str, int] = {
        kind: sum(value.get("boundary") == kind for value in rows)
        for kind in _BOUNDARY_EVENT_KINDS
    }
    outbound_event_ids = {
        value.get("provider_attempt_id")
        for value in rows
        if value.get("boundary") == "http_outbound_hook"
    }
    progress_event_projection = [
        (value.get("call_id"), value.get("provider_attempt_id"))
        for value in rows
        if value.get("boundary") == "stream_progress"
    ]
    if (
        counts["logical_dispatch_started"] != 2
        or counts["logical_dispatch_terminal"] != 2
        or counts["logical_request_evidence"] != 2
        or counts["http_outbound_hook"] != len(outbound_rows)
        or counts["stream_progress"] != len(progress_rows)
        or counts["terminal_outcome"] != 2
        or len(outcome_rows) != 2
        or counts["typed_output_evidence"] != 2
        or len(output_rows) != 2
        or outbound_event_ids
        != {value.get("provider_attempt_id") for value in outbound_rows}
        or sorted(progress_event_projection)
        != sorted(
            (value.get("call_id"), value.get("provider_attempt_id"))
            for value in progress_rows
        )
    ):
        raise BoilsG1TwoCallCanaryError(
            "global boundary events do not join durable evidence channels"
        )
    return {**receipt, "boundary_counts": counts, "evidence_join_valid": True}


class _ObservedConcurrentRunner:
    """Record logical call overlap while delegating every queue operation."""

    def __init__(
        self,
        runner: LiveRunner,
        journal: DurableJsonlJournal,
        boundary_ledger: _BoundaryEventLedger,
    ) -> None:
        self._runner = runner
        self._journal = journal
        self._boundary_ledger = boundary_ledger
        self._lock = asyncio.Lock()
        self._sequence = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.completed_call_ids: list[str] = []

    async def _event(
        self,
        *,
        call_id: str,
        phase: str,
        failure_type: str | None = None,
    ) -> None:
        async with self._lock:
            if phase == "started":
                self.in_flight += 1
                self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            elif phase == "terminal":
                self.in_flight -= 1
                self.completed_call_ids.append(call_id)
            else:  # pragma: no cover - private closed call sites.
                raise AssertionError("unknown dispatch phase")
            self._sequence += 1
            self._journal.append(
                {
                    "schema_version": 1,
                    "sequence": self._sequence,
                    "call_id": call_id,
                    "phase": phase,
                    "logical_in_flight": self.in_flight,
                    "logical_peak_in_flight": self.peak_in_flight,
                    "failure_type": failure_type,
                    "raw_failure_content_persisted": False,
                }
            )
            self._boundary_ledger.record(
                (
                    "logical_dispatch_started"
                    if phase == "started"
                    else "logical_dispatch_terminal"
                ),
                call_id=call_id,
            )

    async def __call__(
        self, request: StructuredGenerationRequest[Any]
    ) -> AttemptedStructuredGenerationResponse[Any]:
        call_id = request.call_id.value
        await self._event(call_id=call_id, phase="started")
        failure_type: str | None = None
        try:
            return await self._runner(request)
        except BaseException as exc:
            failure_type = type(exc).__name__
            raise
        finally:
            await self._event(
                call_id=call_id,
                phase="terminal",
                failure_type=failure_type,
            )

    async def snapshot(self) -> object:
        return await self._runner.snapshot()

    async def aclose(self) -> None:
        await self._runner.aclose()


def _attempt_ids(outcome: Mapping[str, object]) -> tuple[str, ...]:
    attempts = outcome.get("attempts")
    if type(attempts) is not list or not attempts:
        raise BoilsG1TwoCallCanaryError("terminal outcome omitted attempts")
    identities: list[str] = []
    for attempt in attempts:
        evidence = attempt.get("request_evidence") if type(attempt) is dict else None
        attempt_id = (
            evidence.get("provider_attempt_id") if type(evidence) is dict else None
        )
        if type(attempt_id) is not str:
            raise BoilsG1TwoCallCanaryError(
                "non-cancelled canary attempt lacks a physical identity"
            )
        identities.append(attempt_id)
    return tuple(identities)


def _contract_lanes_by_call(
    readiness_contract: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    lanes = readiness_contract.get("lanes")
    if type(lanes) is not list or len(lanes) != 2:
        raise BoilsG1TwoCallCanaryError("readiness contract lanes are malformed")
    result: dict[str, dict[str, object]] = {}
    for value in lanes:
        if type(value) is not dict:
            raise BoilsG1TwoCallCanaryError("readiness contract lane is malformed")
        call_id = value.get("call_id")
        if (
            type(call_id) is not str
            or call_id not in _EXPECTED_CALL_IDS
            or call_id in result
        ):
            raise BoilsG1TwoCallCanaryError(
                "readiness contract lane call identity drifted"
            )
        result[call_id] = value
    if (
        set(result) != set(_EXPECTED_CALL_IDS)
        or tuple(result[value]["parent_lane_id"] for value in _EXPECTED_CALL_IDS)
        != _EXPECTED_LANE_IDS
    ):
        raise BoilsG1TwoCallCanaryError(
            "readiness contract call/lane binding drifted"
        )
    return result


def _require_request_matches_readiness_lane(
    request: Mapping[str, object], lane: Mapping[str, object]
) -> None:
    schema = lane.get("schema")
    if (
        type(schema) is not dict
        or request.get("call_id") != lane.get("call_id")
        or request.get("prompt_sha256") != lane.get("prompt_sha256")
        or request.get("wire_prompt_sha256") != lane.get("prompt_sha256")
        or request.get("prompt_utf8_bytes") != lane.get("prompt_utf8_bytes")
        or request.get("output_schema_sha256")
        != schema.get("logical_schema_sha256")
        or request.get("output_schema_utf8_bytes")
        != schema.get("logical_schema_utf8_bytes")
        or request.get("output_tool_name") != lane.get("output_tool_name")
        or request.get("temperature_hex") != lane.get("temperature_hex")
        or request.get("max_output_tokens") != lane.get("max_output_tokens")
    ):
        raise BoilsG1TwoCallCanaryError(
            "live request evidence differs from its readiness call/lane commitment"
        )


def _require_release_attempt_shape(
    *,
    call_id: str,
    outcome: Mapping[str, object],
    logical_request: Mapping[str, object],
    progress_rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Admit success or one exact invalid-stream-item retry, nothing else."""

    attempts = outcome.get("attempts")
    if type(attempts) is not list or len(attempts) not in {1, 2}:
        raise BoilsG1TwoCallCanaryError(
            "release permits only one attempt or one exact bounded retry"
        )
    attempt_ids = _attempt_ids(outcome)
    if len(set(attempt_ids)) != len(attempt_ids):
        raise BoilsG1TwoCallCanaryError("physical attempt identities collide")

    def require_original_request(attempt: object, number: int) -> dict[str, object]:
        if type(attempt) is not dict or attempt.get("attempt_number") != number:
            raise BoilsG1TwoCallCanaryError("attempt order is malformed")
        evidence = attempt.get("request_evidence")
        if (
            type(evidence) is not dict
            or evidence.get("variant") != "original"
            or evidence.get("prompt_sha256")
            != logical_request.get("prompt_sha256")
            or evidence.get("provider_attempt_id") != attempt_ids[number - 1]
        ):
            raise BoilsG1TwoCallCanaryError(
                "retry changed or lost the original request identity"
            )
        return attempt

    first = require_original_request(attempts[0], 1)
    if len(attempts) == 2:
        classification = first.get("classification")
        failure = first.get("failure")
        expected_failure_fields = {
            "kind",
            "retryable",
            "safe_message",
            "status_code",
            "retry_after_seconds",
            "stream_timeout_phase",
            "output_failure_mode",
            "validation_issues",
            "provider_error_code",
            "provider_error_envelope_sha256",
            "exception_provenance",
        }
        if (
            first.get("status") != "retryable_failure"
            or first.get("will_retry") is not True
            or first.get("error_type") != "StructuredGenerationError"
            or classification != {"disposition": "retry", "reason": "transient"}
            or type(failure) is not dict
            or set(failure) != expected_failure_fields
            or failure.get("kind") != "provider_unavailable"
            or failure.get("retryable") is not True
            or failure.get("safe_message") != _INVALID_STREAM_ITEM_SAFE_MESSAGE
            or failure.get("status_code") is not None
            or failure.get("retry_after_seconds") is not None
            or failure.get("stream_timeout_phase") is not None
            or failure.get("output_failure_mode") is not None
            or failure.get("validation_issues") != []
            or failure.get("provider_error_code") is not None
            or failure.get("provider_error_envelope_sha256") is not None
            or failure.get("exception_provenance") is not None
        ):
            raise BoilsG1TwoCallCanaryError(
                "attempt one is not the exact invalid-stream-item retry class"
            )
        if any(
            value.get("provider_attempt_id") == attempt_ids[0]
            for value in progress_rows
        ):
            raise BoilsG1TwoCallCanaryError(
                "invalid-stream-item attempt emitted forbidden progress"
            )
        successful = require_original_request(attempts[1], 2)
    else:
        successful = first
    if (
        successful.get("status") != "succeeded"
        or successful.get("will_retry") is not False
        or successful.get("classification") is not None
        or successful.get("failure") is not None
        or successful.get("error_type") is not None
    ):
        raise BoilsG1TwoCallCanaryError(
            "final admitted attempt is not an exact success"
        )
    if any(
        value.get("call_id") != call_id
        for value in progress_rows
        if value.get("provider_attempt_id") in set(attempt_ids)
    ):
        raise BoilsG1TwoCallCanaryError("attempt progress crossed logical calls")
    return attempt_ids


def _require_k8_output_audit_join(
    *,
    output: Mapping[str, object],
    response: Mapping[str, object],
    original_audit: object,
) -> list[object]:
    members = output.get("typed_output")
    members = members.get("members") if type(members) is dict else None
    original_members = (
        original_audit.get("members") if type(original_audit) is dict else None
    )
    provider_response_id = response.get("provider_response_id")
    finish_reason = response.get("finish_reason")
    if (
        type(provider_response_id) is not str
        or not provider_response_id.strip()
        or type(finish_reason) is not str
        or not finish_reason.strip()
        or output.get("provider_response_id") != provider_response_id
        or type(members) is not list
        or len(members) != 8
        or type(original_members) is not list
        or len(original_members) != 8
    ):
        raise BoilsG1TwoCallCanaryError(
            "provider response, exact K8 output, and supplemental audit do not join"
        )
    canonical_original: list[dict[str, object]] = []
    for rank, value in enumerate(original_members, start=1):
        if type(value) is not dict or value.get("model_rank") != rank:
            raise BoilsG1TwoCallCanaryError(
                "provider response, exact K8 output, and supplemental audit do not join"
            )
        projected = dict(value)
        del projected["model_rank"]
        hierarchical_action = projected.pop("hierarchical_action", None)
        if hierarchical_action is not None:
            if type(hierarchical_action) is not dict:
                raise BoilsG1TwoCallCanaryError(
                    "hierarchical K8 audit action is malformed"
                )
            action_kind = hierarchical_action.get("action_kind")
            resolved_option_id = projected.pop("option_id", None)
            if action_kind == "atomic":
                projected["action_kind"] = "atomic"
                projected["option_id"] = resolved_option_id
            elif action_kind == "compose_r2":
                projected["action_kind"] = "compose_r2"
                projected["composite_option_id"] = resolved_option_id
                projected["component_option_ids"] = hierarchical_action.get(
                    "component_option_ids"
                )
            else:
                raise BoilsG1TwoCallCanaryError(
                    "hierarchical K8 audit action has an unknown kind"
                )
        canonical_original.append(projected)
    if _canonical_bytes(members) != _canonical_bytes(canonical_original):
        raise BoilsG1TwoCallCanaryError(
            "provider response, exact K8 output, and supplemental audit do not join"
        )
    return members


def _validate_completed_live_calls(
    results: Sequence[PortfolioSelectionResult],
    *,
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    boundary_event_rows: Sequence[Mapping[str, object]],
    framework_versions: Mapping[str, object],
    logical_peak_in_flight: int,
    readiness_contract: Mapping[str, object],
) -> dict[str, object]:
    if len(results) != 2 or any(
        type(value) is not PortfolioSelectionResult for value in results
    ):
        raise BoilsG1TwoCallCanaryError("selector did not return two exact results")
    for result in results:
        result.__post_init__()
    if (
        len(request_rows) != 2
        or len(outcome_rows) != 2
        or len(output_rows) != 2
    ):
        raise BoilsG1TwoCallCanaryError(
            "live calls did not emit two logical requests/outcomes/outputs"
        )
    if not 2 <= len(outbound_rows) <= 4:
        raise BoilsG1TwoCallCanaryError("physical attempt count is invalid")
    if logical_peak_in_flight != 2:
        raise BoilsG1TwoCallCanaryError("both logical calls did not overlap")

    lanes_by_call = _contract_lanes_by_call(readiness_contract)
    requests_by_call: dict[str, dict[str, object]] = {}
    for value in request_rows:
        request = validate_structured_generation_request_evidence_record(value)
        call_id = request["call_id"]
        if (
            call_id not in _EXPECTED_CALL_IDS
            or call_id in requests_by_call
            or request["operation"] != "select_portfolio"
            or request["output_tool_name"]
            != CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME
            or request["max_output_tokens"]
            != campaign.MODEL_EXECUTION_PROFILE.max_output_tokens
        ):
            raise BoilsG1TwoCallCanaryError("logical request evidence drifted")
        _require_request_matches_readiness_lane(
            request, lanes_by_call[call_id]
        )
        requests_by_call[call_id] = request
    if set(requests_by_call) != set(_EXPECTED_CALL_IDS):
        raise BoilsG1TwoCallCanaryError("logical request identities are incomplete")

    outbound_by_call: dict[str, list[dict[str, object]]] = {
        value: [] for value in _EXPECTED_CALL_IDS
    }
    for value in outbound_rows:
        outbound = _validate_outbound_manifest(value)
        outbound_by_call[outbound["call_id"]].append(outbound)
    if any(
        not 1 <= len(values) <= 2
        for values in outbound_by_call.values()
    ):
        raise BoilsG1TwoCallCanaryError("per-call physical attempts are incomplete")

    outputs_by_call: dict[str, dict[str, object]] = {}
    for value in output_rows:
        call_id = value.get("call_id")
        if type(call_id) is not str or call_id not in requests_by_call:
            raise BoilsG1TwoCallCanaryError("typed output names a foreign call")
        if call_id in outputs_by_call:
            raise BoilsG1TwoCallCanaryError("typed output is duplicated")
        outputs_by_call[call_id] = (
            validate_structured_generation_output_evidence_record(
                value,
                request_evidence=requests_by_call[call_id],
            )
        )
    outcomes_by_call: dict[str, dict[str, object]] = {}
    for value in outcome_rows:
        call_id = value.get("task_id")
        if (
            type(call_id) is not str
            or call_id not in requests_by_call
            or call_id in outcomes_by_call
        ):
            raise BoilsG1TwoCallCanaryError("terminal outcome identity drifted")
        outcomes_by_call[call_id] = dict(value)

    calls: list[dict[str, object]] = []
    all_attempt_ids: list[str] = []
    for result, call_id in zip(results, _EXPECTED_CALL_IDS, strict=True):
        output = outputs_by_call[call_id]
        outcome = outcomes_by_call[call_id]
        if outcome.get("status") != "succeeded":
            raise BoilsG1TwoCallCanaryError("terminal queue outcome is not success")
        attempt_ids = _require_release_attempt_shape(
            call_id=call_id,
            outcome=outcome,
            logical_request=requests_by_call[call_id],
            progress_rows=progress_rows,
        )
        outbound_ids = {
            value["provider_attempt_id"] for value in outbound_by_call[call_id]
        }
        if len(attempt_ids) != len(outbound_by_call[call_id]) or set(
            attempt_ids
        ) != outbound_ids:
            raise BoilsG1TwoCallCanaryError("outbound/terminal attempts differ")
        successful_progress = [
            value
            for value in progress_rows
            if value.get("provider_attempt_id") == attempt_ids[-1]
        ]
        if (
            not successful_progress
            or successful_progress[-1].get("kind") != "stream_completed"
        ):
            raise BoilsG1TwoCallCanaryError(
                "successful physical attempt lacks stream completion"
            )
        response = outcome.get("response")
        if (
            type(response) is not dict
            or response.get("requested_model") != campaign.MODEL
            or response.get("resolved_model") != campaign.MODEL
            or response.get("resolved_provider") != campaign.RESOLVED_PROVIDER
            or type(response.get("provider_response_id")) is not str
            or not response["provider_response_id"].strip()
            or type(response.get("finish_reason")) is not str
            or not response["finish_reason"].strip()
            or type(response.get("input_tokens")) is not int
            or response["input_tokens"] <= 0
            or type(response.get("output_tokens")) is not int
            or response["output_tokens"] <= 0
            or type(response.get("reasoning_tokens")) is not int
            or (
                response["reasoning_tokens"] <= 0
                if campaign.MODEL_EXECUTION_PROFILE.require_positive_reasoning_tokens
                else response["reasoning_tokens"] < 0
            )
            or response.get("cost_usd") is None
        ):
            raise BoilsG1TwoCallCanaryError("provider telemetry is incomplete")
        audit = result.supplemental_audit
        payload = None if audit is None else thaw_json(audit.payload)
        original = (
            None
            if type(payload) is not dict
            else payload.get("original_k8_response")
        )
        members = _require_k8_output_audit_join(
            output=output,
            response=response,
            original_audit=original,
        )
        expected_evaluation_width = lanes_by_call[call_id]["evaluation_width"]
        if (
            type(expected_evaluation_width) is not int
            or len(result.decision.members) != expected_evaluation_width
            or len(members) != 8
        ):
            raise BoilsG1TwoCallCanaryError(
                "selector result does not match its sealed K8-to-K contract"
            )
        all_attempt_ids.extend(attempt_ids)
        calls.append(
            {
                "call_id": call_id,
                "parent_lane_id": lanes_by_call[call_id]["parent_lane_id"],
                "physical_attempt_count": len(attempt_ids),
                "provider_attempt_ids": list(attempt_ids),
                "proposal_width": 8,
                "evaluation_width": expected_evaluation_width,
                "selected_option_ids": [
                    member.option_id for member in result.decision.members
                ],
                "response": dict(response),
            }
        )

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
        raise BoilsG1TwoCallCanaryError("provider-attempt terminal join is red")
    boundary_join = _validate_boundary_evidence_join(
        boundary_event_rows,
        outbound_rows=outbound_rows,
        progress_rows=progress_rows,
        outcome_rows=outcome_rows,
        output_rows=output_rows,
    )
    return {
        "logical_call_count": 2,
        "logical_peak_in_flight": logical_peak_in_flight,
        "physical_attempt_count": len(all_attempt_ids),
        "provider_attempt_ids": all_attempt_ids,
        "calls": calls,
        "provider_attempt_join": join,
        "boundary_event_join": boundary_join,
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
    failures: list[object] = []
    for outcome in outcome_rows:
        attempts = outcome.get("attempts")
        if type(attempts) is list and attempts and type(attempts[-1]) is dict:
            failure = attempts[-1].get("failure")
            if failure is not None:
                failures.append(failure)
    diagnosis = {
        "request_evidence_rows": len(request_rows),
        "outbound_manifest_rows": len(outbound_rows),
        "progress_rows": len(progress_rows),
        "terminal_outcome_rows": len(outcome_rows),
        "typed_output_rows": len(output_rows),
        "last_sanitized_failures": failures,
        "failure_schema_supports_exception_provenance_v8": True,
        "raw_exception_text_retained": False,
        "raw_provider_body_retained": False,
    }
    return diagnosis, join


def _readiness_path_binding(
    readiness_dir: Path,
    *,
    run_root: Path,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    resolved = readiness_dir.expanduser().resolve(strict=True)
    workspace = WORKSPACE_ROOT.expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(workspace)
        base = "workspace_root"
    except ValueError:
        if production_stack_authenticated:
            raise BoilsG1TwoCallCanaryError(
                "production readiness directory is outside the workspace"
            )
        offline_root = run_root.expanduser().resolve(strict=True)
        try:
            relative = resolved.relative_to(offline_root)
        except ValueError as exc:
            raise BoilsG1TwoCallCanaryError(
                "offline readiness directory is outside its injected run root"
            ) from exc
        base = "injected_run_root"
    relative_text = relative.as_posix()
    if Path(relative_text).is_absolute() or relative_text.startswith("../"):
        raise BoilsG1TwoCallCanaryError("readiness path binding is not relative")
    return {
        "base": base,
        "relative_path": relative_text,
        "absolute_path_persisted": False,
    }


async def _execute_live(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path,
    dependencies: CanaryDependencies,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    """Execute exactly both bound G1 selector calls and seal all joins."""

    canonical = _validate_run_id(run_id)
    if authorization != LIVE_AUTHORIZATION:
        raise BoilsG1TwoCallCanaryError("live authorization string is invalid")
    dependencies.__post_init__()
    if production_stack_authenticated and not _is_production_dependencies(
        dependencies
    ):
        raise BoilsG1TwoCallCanaryError(
            "production live dependencies are not authentic"
        )
    if production_stack_authenticated:
        _production_runner_constructor_binding()
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
    readiness_path_binding = _readiness_path_binding(
        readiness_dir,
        run_root=run_root,
        production_stack_authenticated=production_stack_authenticated,
    )
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
                "directory_binding": readiness_path_binding,
                "readiness": file_identity(
                    readiness_dir.expanduser().resolve(strict=True)
                    / "readiness.json"
                ),
                "finalization_sha256": readiness_finalization.get(
                    "finalization_sha256"
                ),
            },
            "maximum_logical_calls": 2,
            "maximum_physical_attempts": 2 * campaign.MAX_ATTEMPTS,
            "abc_executions": 0,
            "child_materialization_boundary": (
                readiness["contract"]["child_materialization_boundary"]
            ),
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
            "model_quality_result_eligible": False,
        },
    )
    inputs = dependencies.inputs_factory(run_dir, str(source["aggregate_sha256"]))
    if type(inputs) is not CanaryInputs:
        raise TypeError("inputs_factory returned a foreign value")
    await _bind_live_inputs_to_readiness(inputs, readiness)
    if _source_identity() != source:
        raise BoilsG1TwoCallCanaryError("source changed before credential access")

    progress_journal = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl", max_unfsynced_rows=32
    )
    request_journal = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    outbound_journal = DurableJsonlJournal(
        run_dir / "provider_attempt_requests.jsonl"
    )
    output_journal = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    outcome_journal = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    dispatch_journal = DurableJsonlJournal(run_dir / "provider_dispatch.jsonl")
    boundary_journal = DurableJsonlJournal(
        run_dir / "provider_boundary_events.jsonl"
    )
    boundary_ledger = _BoundaryEventLedger(boundary_journal)
    progress_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    outbound_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    credential_reads = 0
    client_constructed = False
    provider_call_attempted = False
    runner: _ObservedConcurrentRunner | None = None
    failure: BaseException | None = None
    preclose_snapshot: dict[str, object] | None = None
    postclose_snapshot: dict[str, object] | None = None
    result_record: dict[str, object]

    def progress_sink(value: StructuredStreamProgress) -> None:
        row = _progress_record(value)
        progress_rows.append(row)
        progress_journal.append(row)
        boundary_ledger.record(
            "stream_progress",
            call_id=value.call_id,
            provider_attempt_id=value.provider_attempt_id,
        )

    def request_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_request_evidence_record(value)
        request_rows.append(row)
        request_journal.append(row)
        boundary_ledger.record(
            "logical_request_evidence", call_id=str(row["call_id"])
        )

    def outbound_sink(value: Mapping[str, object]) -> None:
        row = _validate_outbound_manifest(value)
        outbound_rows.append(row)
        outbound_journal.append(row)
        boundary_ledger.record(
            "http_outbound_hook",
            call_id=str(row["call_id"]),
            provider_attempt_id=str(row["provider_attempt_id"]),
        )

    def output_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_output_evidence_record(value)
        output_rows.append(row)
        output_journal.append(row)
        boundary_ledger.record(
            "typed_output_evidence", call_id=str(row["call_id"])
        )

    def outcome_sink(value: object) -> None:
        progress_journal.flush()
        row = structured_generation_outcome_record(value)  # type: ignore[arg-type]
        outcome_rows.append(row)
        outcome_journal.append(row)
        boundary_ledger.record(
            "terminal_outcome", call_id=str(row["task_id"])
        )

    try:
        credential_reads = 1
        api_key = dependencies.credential_loader()
        if type(api_key) is not str or not api_key:
            raise BoilsG1TwoCallCanaryError("credential loader returned an empty key")
        write_json_atomic(
            run_dir / "credential_access.json",
            {
                "schema_version": 1,
                "credential_name": "OPENROUTER_API_KEY",
                "read_count": 1,
                "value_persisted": False,
            },
        )
        raw_runner = dependencies.runner_factory(
            api_key=api_key,
            config=campaign._provider_config(),
            progress_sink=progress_sink,
            request_evidence_sink=request_sink,
            outbound_request_manifest_sink=outbound_sink,
            output_evidence_sink=output_sink,
            outcome_sink=outcome_sink,
        )
        runner = _ObservedConcurrentRunner(
            raw_runner, dispatch_journal, boundary_ledger
        )
        client_constructed = True
        selector = inputs.coordinator.build_selector(runner)
        provider_call_attempted = True
        results = await gather_concurrent_stage(
            selector.select(request) for request in inputs.requests
        )
        completed = _validate_completed_live_calls(
            results,
            request_rows=request_rows,
            outbound_rows=outbound_rows,
            progress_rows=progress_rows,
            outcome_rows=outcome_rows,
            output_rows=output_rows,
            boundary_event_rows=boundary_ledger.rows,
            framework_versions=_framework_versions(),
            logical_peak_in_flight=runner.peak_in_flight,
            readiness_contract=readiness["contract"],
        )
        if (
            inputs.evaluator_guard.calls != 0
            or getattr(inputs.evaluator_observer, "calls") != 0
            or inputs.materialization_guard.child_candidate_allocation_attempts != 0
        ):
            raise BoilsG1TwoCallCanaryError(
                "live selectors crossed a forbidden workload boundary"
            )
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "completed_two_call_canary_only"
                if production_stack_authenticated
                else "completed_offline_test_only"
            ),
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "abc_executions": 0,
            "child_materialization_boundary": (
                inputs.materialization_guard.to_record()
            ),
            "evaluator_call_count": 0,
            **completed,
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
            "model_quality_result_eligible": False,
        }
    except BaseException as exc:
        failure = exc
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_two_call_canary_only",
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "failure_type": type(exc).__name__,
            "logical_peak_in_flight": 0 if runner is None else runner.peak_in_flight,
            "abc_executions": inputs.evaluator_guard.calls,
            "child_materialization_boundary": (
                inputs.materialization_guard.to_record()
            ),
            "evaluator_call_count": getattr(inputs.evaluator_observer, "calls"),
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
            "model_quality_result_eligible": False,
        }
    finally:
        if runner is not None:
            try:
                preclose_snapshot = _queue_snapshot_record(await runner.snapshot())
                write_json_atomic(
                    run_dir / "queue_snapshot_before_close.json", preclose_snapshot
                )
                if (
                    preclose_snapshot["pending"] != 0
                    or preclose_snapshot["in_flight"] != 0
                    or runner.in_flight != 0
                ):
                    raise BoilsG1TwoCallCanaryError(
                        "queue retained work before close"
                    )
            except BaseException as snapshot_error:
                if failure is None:
                    failure = snapshot_error
                    result_record["failure_type"] = type(snapshot_error).__name__
                result_record["status"] = "failed_two_call_canary_only"
                result_record["preclose_snapshot_failure_type"] = type(
                    snapshot_error
                ).__name__
            try:
                await runner.aclose()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                    result_record["failure_type"] = type(close_error).__name__
                result_record["status"] = "failed_two_call_canary_only"
                result_record["runner_close_failure_type"] = type(
                    close_error
                ).__name__
            try:
                postclose_snapshot = _queue_snapshot_record(await runner.snapshot())
                write_json_atomic(
                    run_dir / "queue_snapshot_after_close.json", postclose_snapshot
                )
                if (
                    postclose_snapshot["pending"] != 0
                    or postclose_snapshot["in_flight"] != 0
                    or postclose_snapshot["closed"] is not True
                    or runner.in_flight != 0
                ):
                    raise BoilsG1TwoCallCanaryError(
                        "closed queue is not empty and terminal"
                    )
            except BaseException as snapshot_error:
                if failure is None:
                    failure = snapshot_error
                    result_record["failure_type"] = type(snapshot_error).__name__
                result_record["status"] = "failed_two_call_canary_only"
                result_record["postclose_snapshot_failure_type"] = type(
                    snapshot_error
                ).__name__
        for journal in (
            progress_journal,
            request_journal,
            outbound_journal,
            output_journal,
            outcome_journal,
            dispatch_journal,
            boundary_journal,
        ):
            journal.close()

    source_unchanged = _source_identity() == source
    if not source_unchanged and failure is None:
        failure = BoilsG1TwoCallCanaryError("source changed during live calls")
        result_record["status"] = "failed_two_call_canary_only"
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
    result_record["source_identity_verified_before_and_after"] = source_unchanged
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
    if "boundary_event_join" in result_record:
        write_json_atomic(
            run_dir / "provider_boundary_event_join.json",
            result_record["boundary_event_join"],
        )
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
    """Public two-call path sealed to production composition identities."""

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
    dependencies: CanaryDependencies,
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
                raise BoilsG1TwoCallCanaryError(
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
        BoilsG1TwoCallCanaryError,
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
