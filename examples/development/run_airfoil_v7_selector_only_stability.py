"""Frozen nine-call selector-only Airfoil-v7 stability experiment.

The benchmark-local composition authenticates the sealed recovery card bank,
reconstructs the historical M/P/N scientific requests, and dispatches nine
serial execution calls.  Each slot owns a fresh one-connection provider stack;
the historical request identity remains inside the frozen prompt while a fresh
execution call ID is used only for queue accounting.

Manifest construction and verification are provider-free and never read a
credential.  This is a post-hoc transport/stability study, not held-out
efficacy evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Protocol

from pydantic import BaseModel

AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.execution_binding import (
    StructuredScienceRequestBinding,
    rebind_structured_execution_request,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PORTFOLIO_SELECTION_TOOL_NAME,
    PydanticAIPortfolioSelectionPolicy,
    _portfolio_output_type,
    render_portfolio_selection_prompt,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    ExactPayloadAttemptPolicy,
    OutcomePublicationPolicy,
    TransportOnlyStructuredGenerationRetryClassifier,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.portfolio_selection import (
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development import (
    DEFAULT_SEALED_ORACLE_DIR,
    PreparedSelectorViews,
    VerifiedSealedOracle,
    _validate_card_for_call,
    build_card_projection_sources,
    prepare_provider_ready_stage_a,
    prepare_selector_views,
    reflection_result_record,
    selector_result_record,
    telemetry_record,
)
from examples.development import (
    run_airfoil_v7_oracle_portfolio_stage_a as frozen,
)


WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "portfolio_stage_a"
SOURCE_RUN_ID = "ae7_portfolio_stage_a_recovery_0715_0404"
SOURCE_RUN_DIR = RUN_ROOT / SOURCE_RUN_ID
SOURCE_BANK_PATH = SOURCE_RUN_DIR / "reflection_results.json"
PROTOCOL_ARTIFACT = (
    ARTIFACT_ROOT
    / "121_airfoil_v7_selector_only_position_counterbalanced_protocol.md"
)
PROTOCOL_ARTIFACT_SHA256 = (
    "afa937438b1d9c725c91be4f74b3ec4c978c29271f341b2d095f5fec9be86561"
)
LINEAGE_ARTIFACT_SHA256 = {
    "115_generic_portfolio_selector_development_and_transfer_protocol.md": (
        "54c934ed21f6b0757ddbbec51ea2f576311ac88caf560a203e56f23a86fb92a7"
    ),
    "119_airfoil_v7_stage_a_v1r2_single_shard_recovery_protocol.md": (
        "8586bb580510b2429f72eaf05ba6b6aa0f389d5d3c8a3110ffb5330a79354647"
    ),
    "120_airfoil_v7_stage_a_recovery_failure_and_queue_diagnosis.md": (
        "abe8ee05583e08a6f9f1be2ba2864435537f235396d4b6a998068eca0d98aa08"
    ),
    "121_airfoil_v7_selector_only_position_counterbalanced_protocol.md": (
        PROTOCOL_ARTIFACT_SHA256
    ),
}

DESIGN_ID = "airfoil_v7_selector_only_position_counterbalanced_v1"
EXECUTION_CLASS = "post_hoc_selector_transport_and_stability_study"
CLAIM_BOUNDARY = (
    "Post-hoc selector transport/stability development; not held-out "
    "efficacy, genericity, SOTA, or wall-clock evidence."
)
MANIFEST_KIND = "agent_evolve_airfoil_v7_selector_stability_manifest_v1"
MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-selector-stability-manifest:v1\x00"
DEFAULT_RELEASE_EVIDENCE = (
    RUN_ROOT / "manifests" / "selector_stability_release_evidence.json"
)
RELEASE_GATE_PYTEST_COMMAND = (
    ".venv/bin/python -m pytest -q tests/test_llm_task_queue.py "
    "tests/test_queued_structured_runner.py "
    "tests/test_airfoil_v7_selector_only_stability.py"
)
RELEASE_GATE_COMPILE_COMMAND = (
    ".venv/bin/python -m compileall -q src/agent_evolve "
    "examples/development/run_airfoil_v7_selector_only_stability.py "
    "tests/test_llm_task_queue.py tests/test_queued_structured_runner.py "
    "tests/test_airfoil_v7_selector_only_stability.py"
)
RELEASE_GATE_COMMANDS = (
    RELEASE_GATE_PYTEST_COMMAND,
    RELEASE_GATE_COMPILE_COMMAND,
)
REQUIRED_RELEASE_TESTS = (
    "test_recovery_topology_three_active_terminal_then_two_transport_aborts",
    "test_simultaneous_transport_abort_retirement_has_one_owner_and_terminates",
    "test_transport_only_retry_classifier_is_a_positive_allowlist",
    "test_serial_controller_continues_after_timeout_and_defers_analysis",
    "test_complete_batch_analyzes_only_after_all_nine_stacks_close",
)

MODEL = frozen.MODEL
ALLOWED_RESOLVED_MODELS = frozen.ALLOWED_RESOLVED_MODELS
ALLOWED_RESOLVED_PROVIDERS = frozen.ALLOWED_RESOLVED_PROVIDERS
PROVIDER_ONLY = frozen.PROVIDER_ONLY
MAX_INPUT_TOKENS = frozen.MAX_INPUT_TOKENS
MAX_OUTPUT_TOKENS = frozen.MAX_OUTPUT_TOKENS
MAX_IN_FLIGHT = 1
MAX_PENDING = 0
MAX_ATTEMPTS = 2
MAX_CONNECTIONS = 1
ATTEMPT_TIMEOUT_NS = 300_000_000_000
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
QUIET_INTERVAL_SECONDS = 5.0
JITTER_SEED = 20_260_715
JITTER_DOMAIN = "airfoil-v7-selector-stability-v1"

SOURCE_FILE_SHA256 = {
    "launch_manifest.json": (
        "a32b229fdd1a6d14dc91ca1efa53e6c21ec3cf7985db025779609ee662bbc991"
    ),
    "prompt_response_journal.jsonl": (
        "66f98ed1f2f3b624de5e7371d5ff27d84af068df5b077ac3b1fc71c2fa46d805"
    ),
    "provider_queue_outcomes.jsonl": (
        "2585d639985d69e314c7da08d081a8a21433a91d86db6f286b10530831ed5895"
    ),
    "source_verifications.jsonl": (
        "1e07a101fe390426e948bf1ce2557b5c4789cf78e59e27a095b86f8496d1d84b"
    ),
    "failure.json": (
        "df8aeb18fa00cb73a23455ee54c7e98eb3abe47261545d742601113fefb1aa37"
    ),
    "reflection_results.json": (
        "8836115611bae7ccae4fb70184f59cba61dc788ab53f485d28e9abe12f23834c"
    ),
    "finalized.json": (
        "412836f062110aaeaf364473865b6666f20a9d67c156a13039f2f3f13b086025"
    ),
}
SOURCE_MANIFEST_SHA256 = (
    "ebb1b2230c2dc073e2e7b7314ecdf36cd3acf446a6bf81f4063d129db7a0102f"
)
SOURCE_RECURSIVE_SHA256 = (
    "b5010720bed26fffbe9f7f0fcd2ee15a32440c63ef06abcb02bd44144fc11aab"
)
SOURCE_FINALIZATION_SHA256 = (
    "e74942b97807c71d0d13513a3bc5ed7ccd1fbada7267510984f941ab3cb8d6e4"
)
SOURCE_RECURSIVE_FILE_COUNT = 8

CARD_BANK_PROJECTION_SHA256 = (
    "4365264b868a0e1065b2ba1fc27369831786aeb20e75ddce334024aa493f10f4"
)
SELECTOR_VIEWS_SHA256 = (
    "06aa9708a3697cf0f7c645296386b92e8014b96cf5144271b690c93cf11b41cd"
)
VIEW_RECORD_SHA256 = {
    "M": "e888d8e400a8704385b3d8d7fb634fa8bf5f77c08be10168d9dfad03e8d2e214",
    "P": "778725c008a48ac10552a4a902593540bc4e80b3e651fa593515e7eddc6665c2",
    "N": "06758de28075e634d4e24931bc2e8de42ec5a8db49cb69b0afe26125f522f184",
}
ORDERED_OPTIONS_SHA256 = (
    "b86eaacbaed1dd30ab9c8ea4322224d41a1bf650bcb09a6159301f5ee0bc41c9"
)
CONTEXT_SHA256 = (
    "71f8568c0eaac0ff8549bc74fbe8ee91cc7552612d4efbdcb930f12c5adbae03"
)
PROMPT_SHA256 = {
    "M": "f8ddf8a5408c92c01e2f719caa539540ded7c137dcb17c0582fc799fa0f62bcc",
    "P": "afbe4ce1909fd9d535cb129107f482617fb99f871cb7d0c1976bbb55a91876fd",
    "N": "4738f3a6e35e343d75b6f8ea614b80051d693783dd0cd022d19bfea6a3465d9c",
}
PROMPT_UTF8_BYTES = {"M": 66_301, "P": 66_301, "N": 24_343}
OUTPUT_SCHEMA_SHA256 = (
    "278590d5b63de8f02e106fa4f3e79c12c1638fd0bb47edd0d36095b4ad305730"
)
OUTPUT_SCHEMA_UTF8_BYTES = 3_550


def _canonical_bytes(value: object, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii" if ensure_ascii else "utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _projection_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value, ensure_ascii=False))


def _schema_bytes(output_type: type[object]) -> bytes:
    if not issubclass(output_type, BaseModel):
        raise TypeError("selector output type must be a Pydantic BaseModel")
    return _canonical_bytes(output_type.model_json_schema(mode="validation"))


@dataclass(frozen=True, slots=True)
class SelectorSlot:
    absolute_slot: int
    block: int
    position: int
    view_id: str
    replicate: int
    execution_call_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "absolute_slot": self.absolute_slot,
            "block": self.block,
            "position": self.position,
            "view_id": self.view_id,
            "replicate": self.replicate,
            "execution_call_id": self.execution_call_id,
        }


def _build_schedule() -> tuple[SelectorSlot, ...]:
    order = ("M", "P", "N", "P", "N", "M", "N", "M", "P")
    seen: Counter[str] = Counter()
    slots = []
    for index, view_id in enumerate(order, start=1):
        seen[view_id] += 1
        slots.append(
            SelectorSlot(
                absolute_slot=index,
                block=1 + (index - 1) // 3,
                position=1 + (index - 1) % 3,
                view_id=view_id,
                replicate=seen[view_id],
                execution_call_id=f"call_airfoil_selector_stability_{index:06d}",
            )
        )
    return tuple(slots)


SCHEDULE = _build_schedule()


@dataclass(frozen=True, slots=True)
class FrozenBankEvidence:
    oracle: VerifiedSealedOracle
    selector_views: PreparedSelectorViews
    record: dict[str, object]

    @property
    def requests(self) -> dict[str, PortfolioSelectionRequest]:
        return self.selector_views.by_id()


def _structured_response_from_card(
    request: StructuredGenerationRequest[Any],
    card: Mapping[str, object],
) -> AttemptedStructuredGenerationResponse[Any]:
    content = card.get("content")
    telemetry = card.get("telemetry")
    if type(content) is not dict or type(telemetry) is not dict:
        raise RuntimeError("sealed card content or telemetry is malformed")
    output_content = dict(content)
    if output_content.pop("schema_version", None) != 1:
        raise RuntimeError("sealed card content schema is unsupported")
    value = request.output_type.model_validate(
        {"insights": [output_content]},
        strict=True,
    )
    cost = telemetry.get("cost_usd")
    response = StructuredGenerationResponse(
        value=value,
        requested_model=str(telemetry["requested_model"]),
        resolved_model=str(telemetry["resolved_model"]),
        resolved_provider=str(telemetry["resolved_provider"]),
        provider_response_id=telemetry.get("provider_response_id"),
        finish_reason=telemetry.get("finish_reason"),
        input_tokens=int(telemetry["input_tokens"]),
        output_tokens=int(telemetry["output_tokens"]),
        reasoning_tokens=int(telemetry["reasoning_tokens"]),
        cache_read_tokens=int(telemetry["cache_read_tokens"]),
        cache_write_tokens=int(telemetry["cache_write_tokens"]),
        cost_usd=None if cost is None else Decimal(str(cost)),
        latency_ns=int(telemetry["latency_ns"]),
    )
    return AttemptedStructuredGenerationResponse(
        response=response,
        attempt_count=int(telemetry["attempt_count"]),
    )


def _authenticate_source_files() -> dict[str, object]:
    bindings = {
        name: frozen._file_binding(SOURCE_RUN_DIR / name)
        for name in SOURCE_FILE_SHA256
    }
    if any(
        bindings[name]["sha256"] != expected
        for name, expected in SOURCE_FILE_SHA256.items()
    ):
        raise RuntimeError("sealed recovery file identity drifted")
    finalized = frozen._load_object(SOURCE_RUN_DIR / "finalized.json")
    launch = frozen._load_object(SOURCE_RUN_DIR / "launch_manifest.json")
    count, recursive_sha = frozen._recursive_content_identity(SOURCE_RUN_DIR)
    if (
        finalized.get("status") != "failed"
        or finalized.get("finalization_sha256") != SOURCE_FINALIZATION_SHA256
        or finalized.get("recursive_content_sha256") != SOURCE_RECURSIVE_SHA256
        or finalized.get("recursive_file_count") != SOURCE_RECURSIVE_FILE_COUNT
        or count != SOURCE_RECURSIVE_FILE_COUNT
        or recursive_sha != SOURCE_RECURSIVE_SHA256
        or launch.get("manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("sealed recovery finalization or manifest drifted")
    return {
        "run_id": SOURCE_RUN_ID,
        "status": "failed",
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "finalization_sha256": SOURCE_FINALIZATION_SHA256,
        "recursive_content_sha256": SOURCE_RECURSIVE_SHA256,
        "recursive_file_count": SOURCE_RECURSIVE_FILE_COUNT,
        "files": bindings,
    }


def authenticate_frozen_bank() -> FrozenBankEvidence:
    """Strictly reparse all cards and reproduce all historical selector bytes."""

    source_record = _authenticate_source_files()
    sealed = frozen._load_object(SOURCE_BANK_PATH)
    cards_value = sealed.get("cards")
    if type(cards_value) is not list or len(cards_value) != 8:
        raise RuntimeError("sealed recovery bank is not the frozen eight-card bank")
    cards_by_call = {
        str(card.get("request_call_id")): card
        for card in cards_value
        if type(card) is dict
    }
    if len(cards_by_call) != 8:
        raise RuntimeError("sealed card request IDs are not unique")

    ids = DeterministicIdFactory("airfoil_oracle_stage_a")
    oracle, _, reflection_calls = prepare_provider_ready_stage_a(
        run_dir=DEFAULT_SEALED_ORACLE_DIR,
        id_factory=ids,
    )

    async def replay(request: StructuredGenerationRequest[Any]) -> object:
        card = cards_by_call.get(request.call_id.value)
        if card is None:
            raise RuntimeError("reflection replay escaped the sealed bank")
        return _structured_response_from_card(request, card)

    async def revalidate_cards() -> tuple[Any, ...]:
        generator = PydanticAIAgenticGenerator(replay)
        accepted = []
        for prepared in reflection_calls:
            result = await generator.reflect(prepared.request)
            accepted.append(_validate_card_for_call(prepared, result))
        return tuple(sorted(accepted, key=lambda item: item.card_key))

    accepted = asyncio.run(revalidate_cards())
    sources = build_card_projection_sources(accepted)
    selector_views = prepare_selector_views(oracle, sources, id_factory=ids)
    rebuilt = reflection_result_record(oracle, accepted, sources, selector_views)
    if rebuilt != sealed:
        raise RuntimeError("strict card replay did not reproduce reflection_results")

    card_projection = [
        {
            "card_key": card["card_key"],
            "content_sha256": card["content_sha256"],
            "request_call_id": card["request_call_id"],
            "request_prompt_sha256": card["request_prompt_sha256"],
            "shard_id": card["shard_id"],
        }
        for card in cards_value
    ]
    if _projection_sha256(card_projection) != CARD_BANK_PROJECTION_SHA256:
        raise RuntimeError("validated card-bank projection drifted")
    selector_record = sealed.get("selector_views")
    if (
        type(selector_record) is not dict
        or _projection_sha256(selector_record) != SELECTOR_VIEWS_SHA256
    ):
        raise RuntimeError("sealed selector-view object drifted")
    view_rows = selector_record.get("views")
    if type(view_rows) is not list or len(view_rows) != 3:
        raise RuntimeError("sealed selector views are malformed")
    view_by_id = {
        str(row.get("view_id")): row for row in view_rows if type(row) is dict
    }
    if set(view_by_id) != {"M", "P", "N"}:
        raise RuntimeError("sealed selector view IDs drifted")
    if any(
        _projection_sha256(view_by_id[view_id]) != expected
        for view_id, expected in VIEW_RECORD_SHA256.items()
    ):
        raise RuntimeError("sealed full view record drifted")
    if any(
        _projection_sha256(view_by_id[view_id]["ordered_options"])
        != ORDERED_OPTIONS_SHA256
        or _projection_sha256(view_by_id[view_id]["context"]) != CONTEXT_SHA256
        for view_id in ("M", "P", "N")
    ):
        raise RuntimeError("shared selector context or option catalog drifted")

    low_level_bindings = []
    for view_id, request in selector_views.requests:
        prompt = render_portfolio_selection_prompt(request)
        output_type = _portfolio_output_type(request)
        prompt_bytes = prompt.encode("utf-8", errors="strict")
        schema_bytes = _schema_bytes(output_type)
        if (
            len(prompt_bytes) != PROMPT_UTF8_BYTES[view_id]
            or _sha256_bytes(prompt_bytes) != PROMPT_SHA256[view_id]
            or len(schema_bytes) != OUTPUT_SCHEMA_UTF8_BYTES
            or _sha256_bytes(schema_bytes) != OUTPUT_SCHEMA_SHA256
        ):
            raise RuntimeError("historical selector prompt or schema drifted")
        low_level_bindings.append(
            {
                "view_id": view_id,
                "historical_science_call_id": request.call_id.value,
                "prompt_utf8_bytes": len(prompt_bytes),
                "prompt_sha256": _sha256_bytes(prompt_bytes),
                "output_schema_utf8_bytes": len(schema_bytes),
                "output_schema_sha256": _sha256_bytes(schema_bytes),
                "output_tool_name": PORTFOLIO_SELECTION_TOOL_NAME,
                "max_output_tokens": request.max_output_tokens,
                "temperature": request.temperature,
            }
        )

    evidence_record = {
        "schema_version": 1,
        "authenticated_before_credentials": True,
        "source_run": source_record,
        "reflection_results": frozen._file_binding(SOURCE_BANK_PATH),
        "strict_card_count": len(accepted),
        "card_bank_projection_sha256": CARD_BANK_PROJECTION_SHA256,
        "selector_views_sha256": SELECTOR_VIEWS_SHA256,
        "view_record_sha256": dict(VIEW_RECORD_SHA256),
        "ordered_options_sha256": ORDERED_OPTIONS_SHA256,
        "context_sha256": CONTEXT_SHA256,
        "low_level_bindings": low_level_bindings,
        "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
        "output_schema_utf8_bytes": OUTPUT_SCHEMA_UTF8_BYTES,
    }
    return FrozenBankEvidence(
        oracle=oracle,
        selector_views=selector_views,
        record=evidence_record,
    )


def _protocol_binding() -> dict[str, object]:
    binding = frozen._file_binding(PROTOCOL_ARTIFACT)
    if binding["sha256"] != PROTOCOL_ARTIFACT_SHA256:
        raise RuntimeError("artifact 121 drifted")
    return binding


def lineage_artifact_record() -> dict[str, object]:
    bindings = {
        name: frozen._file_binding(ARTIFACT_ROOT / name)
        for name in LINEAGE_ARTIFACT_SHA256
    }
    if any(
        bindings[name]["sha256"] != expected
        for name, expected in LINEAGE_ARTIFACT_SHA256.items()
    ):
        raise RuntimeError("selector protocol lineage artifact drifted")
    return {"schema_version": 1, "artifacts": bindings}


def source_snapshot() -> dict[str, object]:
    paths = (
        Path(__file__),
        AGENT_EVOLVE_ROOT
        / "examples/development/run_airfoil_v7_oracle_portfolio_stage_a.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/llm_task_queue.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/domain/llm_task_queue.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/llm_task_queue.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/infrastructure/asyncio_runtime.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/execution_binding.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/portfolio_selection.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/llm_backoff.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/portfolio_selection.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/structured_generator.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/engibench_airfoil/v7_oracle_portfolio_development.py",
        AGENT_EVOLVE_ROOT / "tests/test_llm_task_queue.py",
        AGENT_EVOLVE_ROOT / "tests/test_queued_structured_runner.py",
        AGENT_EVOLVE_ROOT / "tests/test_airfoil_v7_selector_only_stability.py",
        AGENT_EVOLVE_ROOT / "pyproject.toml",
        AGENT_EVOLVE_ROOT / "uv.lock",
    )
    bindings = [frozen._file_binding(path) for path in paths]
    return {
        "schema_version": 1,
        "files": bindings,
        "file_count": len(bindings),
        "sha256": _projection_sha256(bindings),
    }


def release_gate_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "must_pass_immediately_before_manifest_and_dispatch",
        "commands": list(RELEASE_GATE_COMMANDS),
        "required_tests": list(REQUIRED_RELEASE_TESTS),
        "requirements": {
            "failures": 0,
            "errors": 0,
            "warnings": 0,
            "provider_calls": 0,
        },
    }


def release_evidence_binding(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    content = frozen._load_object(resolved)
    snapshot = source_snapshot()
    expected_keys = {
        "schema_version",
        "kind",
        "source_snapshot_sha256",
        "protocol_artifact_sha256",
        "commands",
        "required_test_names",
        "exit_codes",
        "passed_count",
        "failures",
        "errors",
        "warnings",
        "provider_calls",
        "recorded_at_utc",
    }
    if (
        set(content) != expected_keys
        or
        content.get("schema_version") != 1
        or content.get("kind")
        != "airfoil_v7_selector_stability_release_gate_v1"
        or content.get("source_snapshot_sha256") != snapshot["sha256"]
        or content.get("protocol_artifact_sha256") != PROTOCOL_ARTIFACT_SHA256
        or content.get("commands") != list(RELEASE_GATE_COMMANDS)
        or content.get("required_test_names") != list(REQUIRED_RELEASE_TESTS)
        or content.get("exit_codes") != [0, 0]
        or type(content.get("passed_count")) is not int
        or int(content["passed_count"]) <= 0
        or content.get("failures") != 0
        or content.get("errors") != 0
        or content.get("warnings") != 0
        or content.get("provider_calls") != 0
        or type(content.get("recorded_at_utc")) is not str
    ):
        raise RuntimeError("selector release-gate evidence is absent or invalid")
    return {
        "schema_version": 1,
        "file": frozen._file_binding(resolved),
        "evidence": content,
    }


def authorization_record() -> dict[str, object]:
    return {
        "logical_selector_calls": 9,
        "maximum_physical_attempts": 18,
        "reflection_calls": 0,
        "candidate_evaluations": 0,
        "cfd_calls": 0,
        "future_slots_prequeued": False,
    }


def provider_policy_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "requested_model": MODEL,
        "provider": "openrouter",
        "allowed_resolved_models": list(ALLOWED_RESOLVED_MODELS),
        "provider_routing": {
            "only": list(PROVIDER_ONLY),
            "allow_fallbacks": False,
        },
        "allowed_resolved_providers": list(ALLOWED_RESOLVED_PROVIDERS),
        "route_snapshot": frozen._route_snapshot_binding(),
        "reasoning": {
            "request_control": {"effort": "xhigh"},
            "hard_reasoning_token_cap": None,
            "admission": "0 <= reasoning_tokens <= output_tokens",
        },
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "queue": {
            "fresh_owned_stack_per_logical_slot": True,
            "max_in_flight": MAX_IN_FLIGHT,
            "max_pending": MAX_PENDING,
            "max_connections": MAX_CONNECTIONS,
            "max_attempts": MAX_ATTEMPTS,
            "attempt_timeout_ns": ATTEMPT_TIMEOUT_NS,
            "hard_transport_abort_is_terminal_for_slot": True,
            "retry_classifier": (
                "transport_only_positive_allowlist_v1"
            ),
            "attempt_request_policy": "exact_payload",
            "base_backoff_ns": BASE_BACKOFF_NS,
            "max_backoff_ns": MAX_BACKOFF_NS,
            "jitter_seed": JITTER_SEED,
            "jitter_domain": JITTER_DOMAIN,
            "quiet_interval_seconds": QUIET_INTERVAL_SECONDS,
        },
    }


def build_manifest_record(
    *,
    run_id: str,
    output_dir: Path,
    release_evidence_path: Path = DEFAULT_RELEASE_EVIDENCE,
) -> dict[str, object]:
    """Build the provider-free nine-call commitment."""

    target = frozen._validate_target(run_id, output_dir)
    bank = authenticate_frozen_bank()
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": str(target),
        "design_id": DESIGN_ID,
        "method_design_id": frozen.DEVELOPMENT_DESIGN_ID,
        "execution_class": EXECUTION_CLASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_artifact": _protocol_binding(),
        "lineage_artifacts": lineage_artifact_record(),
        "frozen_bank": bank.record,
        "schedule": [slot.to_record() for slot in SCHEDULE],
        "provider_policy": provider_policy_record(),
        "authorization": authorization_record(),
        "release_gate": release_gate_record(),
        "release_evidence": release_evidence_binding(release_evidence_path),
        "source_snapshot": source_snapshot(),
        "credentials_read": False,
        "provider_dispatch_performed": False,
    }
    record["manifest_sha256"] = frozen._domain_sha256(
        record,
        MANIFEST_FRAMING,
    )
    return record


def write_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    release_evidence_path: Path = DEFAULT_RELEASE_EVIDENCE,
) -> dict[str, object]:
    record = build_manifest_record(
        run_id=run_id,
        output_dir=output_dir,
        release_evidence_path=release_evidence_path,
    )
    frozen.write_json_atomic(path, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    manifest_sha256: str
    bank: FrozenBankEvidence


def verify_manifest(
    path: Path,
    *,
    require_output_absent: bool = True,
) -> VerifiedManifest:
    resolved = path.expanduser().resolve(strict=True)
    record = frozen._load_object(resolved)
    expected_keys = {
        "schema_version",
        "kind",
        "built_at_utc",
        "run_id",
        "output_dir",
        "design_id",
        "method_design_id",
        "execution_class",
        "claim_boundary",
        "protocol_artifact",
        "lineage_artifacts",
        "frozen_bank",
        "schedule",
        "provider_policy",
        "authorization",
        "release_gate",
        "release_evidence",
        "source_snapshot",
        "credentials_read",
        "provider_dispatch_performed",
        "manifest_sha256",
    }
    claimed = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256", None)
    if (
        set(record) != expected_keys
        or type(claimed) is not str
        or claimed != frozen._domain_sha256(unsigned, MANIFEST_FRAMING)
        or record.get("schema_version") != 1
        or record.get("kind") != MANIFEST_KIND
        or record.get("design_id") != DESIGN_ID
        or record.get("method_design_id") != frozen.DEVELOPMENT_DESIGN_ID
        or record.get("execution_class") != EXECUTION_CLASS
        or record.get("claim_boundary") != CLAIM_BOUNDARY
        or record.get("credentials_read") is not False
        or record.get("provider_dispatch_performed") is not False
    ):
        raise RuntimeError("selector-stability manifest identity failed")
    run_id = record.get("run_id")
    output = record.get("output_dir")
    if type(run_id) is not str or type(output) is not str:
        raise RuntimeError("selector-stability manifest target is malformed")
    output_dir = frozen._validate_target(run_id, Path(output))
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)
    bank = authenticate_frozen_bank()
    release_value = record.get("release_evidence")
    if type(release_value) is not dict:
        raise RuntimeError("selector release evidence binding is malformed")
    release_file = release_value.get("file")
    if type(release_file) is not dict or type(release_file.get("path")) is not str:
        raise RuntimeError("selector release evidence path is malformed")
    if (
        record.get("protocol_artifact") != _protocol_binding()
        or record.get("lineage_artifacts") != lineage_artifact_record()
        or record.get("frozen_bank") != bank.record
        or record.get("schedule") != [slot.to_record() for slot in SCHEDULE]
        or record.get("provider_policy") != provider_policy_record()
        or record.get("authorization") != authorization_record()
        or record.get("release_gate") != release_gate_record()
        or record.get("release_evidence")
        != release_evidence_binding(Path(str(release_file["path"])))
        or record.get("source_snapshot") != source_snapshot()
    ):
        raise RuntimeError("selector-stability manifest dependency drifted")
    return VerifiedManifest(
        path=resolved,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        manifest_sha256=claimed,
        bank=bank,
    )


def reverify_source(verified: VerifiedManifest) -> dict[str, object]:
    observed = source_snapshot()
    if observed != verified.record["source_snapshot"]:
        raise RuntimeError("selector-stability source snapshot drifted")
    source = _authenticate_source_files()
    if source != verified.record["frozen_bank"]["source_run"]:
        raise RuntimeError("selector-stability source run drifted")
    bank_binding = frozen._file_binding(SOURCE_BANK_PATH)
    if bank_binding != verified.record["frozen_bank"]["reflection_results"]:
        raise RuntimeError("selector-stability bank drifted")
    return {
        "source_sha256": observed["sha256"],
        "bank_file_sha256": bank_binding["sha256"],
        "selector_views_sha256": SELECTOR_VIEWS_SHA256,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def reverify_dispatch_inputs(verified: VerifiedManifest) -> dict[str, object]:
    """Synchronously recheck immutable dispatch inputs inside an event loop."""

    observed = source_snapshot()
    if observed != verified.record["source_snapshot"]:
        raise RuntimeError("selector-stability source snapshot drifted")
    source = _authenticate_source_files()
    if source != verified.record["frozen_bank"]["source_run"]:
        raise RuntimeError("sealed recovery source drifted before dispatch")
    bank_binding = frozen._file_binding(SOURCE_BANK_PATH)
    if bank_binding != verified.record["frozen_bank"]["reflection_results"]:
        raise RuntimeError("sealed reflection bank drifted before dispatch")
    if _protocol_binding() != verified.record["protocol_artifact"]:
        raise RuntimeError("selector protocol drifted before dispatch")
    if lineage_artifact_record() != verified.record["lineage_artifacts"]:
        raise RuntimeError("selector protocol lineage drifted before dispatch")
    provider_policy = provider_policy_record()
    if provider_policy != verified.record["provider_policy"]:
        raise RuntimeError("selector provider route or queue policy drifted")
    release_value = verified.record.get("release_evidence")
    if type(release_value) is not dict or type(release_value.get("file")) is not dict:
        raise RuntimeError("selector release evidence binding is malformed")
    release_path = release_value["file"].get("path")
    if type(release_path) is not str or (
        release_evidence_binding(Path(release_path)) != release_value
    ):
        raise RuntimeError("selector release evidence drifted before dispatch")
    return {
        "source_sha256": observed["sha256"],
        "bank_file_sha256": bank_binding["sha256"],
        "selector_views_sha256": SELECTOR_VIEWS_SHA256,
        "provider_policy_sha256": _projection_sha256(provider_policy),
        "provider_policy": provider_policy,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


class HistoricalScienceExecutionRunner:
    """Split frozen science identity from fresh queue execution identity."""

    def __init__(
        self,
        runner: Callable[[StructuredGenerationRequest[Any]], Awaitable[object]],
        *,
        slot: SelectorSlot,
        historical_request: PortfolioSelectionRequest,
        pre_dispatch: Callable[
            [SelectorSlot, StructuredGenerationRequest[Any]], Mapping[str, object]
        ],
        journal_sink: Callable[[Mapping[str, object]], None],
    ) -> None:
        self._runner = runner
        self._slot = slot
        self._historical_request = historical_request
        self._pre_dispatch = pre_dispatch
        self._journal_sink = journal_sink
        self._expected_prompt = render_portfolio_selection_prompt(historical_request)
        self._expected_output_type = _portfolio_output_type(historical_request)

    async def __call__(
        self,
        historical_low: StructuredGenerationRequest[Any],
    ) -> object:
        if type(historical_low) is not StructuredGenerationRequest:
            raise TypeError("historical low-level request has an invalid type")
        expected_schema = _schema_bytes(self._expected_output_type)
        if (
            historical_low.call_id != self._historical_request.call_id
            or historical_low.operation != self._historical_request.operation
            or historical_low.operation != "select_portfolio"
            or historical_low.prompt != self._expected_prompt
            or _schema_bytes(historical_low.output_type) != expected_schema
            or historical_low.output_tool_name != PORTFOLIO_SELECTION_TOOL_NAME
            or historical_low.max_output_tokens != MAX_OUTPUT_TOKENS
            or historical_low.temperature != 0.0
            or _sha256_bytes(historical_low.prompt.encode())
            != PROMPT_SHA256[self._slot.view_id]
            or _sha256_bytes(expected_schema) != OUTPUT_SCHEMA_SHA256
        ):
            raise RuntimeError("historical science request escaped its frozen binding")
        science_binding = StructuredScienceRequestBinding.from_request(
            historical_low
        )
        execution_request = rebind_structured_execution_request(
            historical_low,
            expected=science_binding,
            execution_call_id=LLMCallId(self._slot.execution_call_id),
        )
        if (
            execution_request.prompt != historical_low.prompt
            or execution_request.output_type is not historical_low.output_type
            or _schema_bytes(execution_request.output_type) != expected_schema
        ):
            raise AssertionError("execution-ID rebinding changed provider-visible bytes")
        verification = dict(self._pre_dispatch(self._slot, execution_request))
        request_record = {
            "schema_version": 1,
            "record_type": "request",
            "absolute_slot": self._slot.absolute_slot,
            "block": self._slot.block,
            "position": self._slot.position,
            "view_id": self._slot.view_id,
            "replicate": self._slot.replicate,
            "historical_science_call_id": historical_low.call_id.value,
            "execution_call_id": execution_request.call_id.value,
            "operation": execution_request.operation,
            "prompt": execution_request.prompt,
            "prompt_utf8_bytes": len(execution_request.prompt.encode()),
            "prompt_sha256": _sha256_bytes(execution_request.prompt.encode()),
            "output_schema_utf8_bytes": len(expected_schema),
            "output_schema_sha256": _sha256_bytes(expected_schema),
            "output_tool_name": execution_request.output_tool_name,
            "max_output_tokens": execution_request.max_output_tokens,
            "temperature": execution_request.temperature,
            "source_verification": verification,
        }
        self._journal_sink(request_record)
        try:
            raw = await self._runner(execution_request)
        except Exception as exc:
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "response_failure",
                    "absolute_slot": self._slot.absolute_slot,
                    "view_id": self._slot.view_id,
                    "execution_call_id": execution_request.call_id.value,
                    "failure_type": type(exc).__name__,
                }
            )
            raise
        if type(raw) is AttemptedStructuredGenerationResponse:
            response = raw.response
            attempts = raw.attempt_count
        elif type(raw) is StructuredGenerationResponse:
            response = raw
            attempts = 1
        else:
            raise TypeError("queued selector runner returned an unsupported response")
        route_checks = {
            "requested_model": response.requested_model == MODEL,
            "resolved_model": response.resolved_model in ALLOWED_RESOLVED_MODELS,
            "resolved_provider": (
                response.resolved_provider in ALLOWED_RESOLVED_PROVIDERS
            ),
            "input_tokens": response.input_tokens <= MAX_INPUT_TOKENS,
            "output_tokens": response.output_tokens <= MAX_OUTPUT_TOKENS,
            "reasoning_in_output": (
                0 <= response.reasoning_tokens <= response.output_tokens
            ),
            "attempt_count": 1 <= attempts <= MAX_ATTEMPTS,
        }
        if not all(route_checks.values()):
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "response_route_rejected",
                    "absolute_slot": self._slot.absolute_slot,
                    "view_id": self._slot.view_id,
                    "execution_call_id": execution_request.call_id.value,
                    "route_checks": route_checks,
                }
            )
            raise RuntimeError("provider response violated selector route policy")
        content = response.value.model_dump(mode="json")
        self._journal_sink(
            {
                "schema_version": 1,
                "record_type": "response",
                "absolute_slot": self._slot.absolute_slot,
                "view_id": self._slot.view_id,
                "execution_call_id": execution_request.call_id.value,
                "content": content,
                "content_sha256": _sha256_bytes(_canonical_bytes(content)),
                "telemetry": {
                    "requested_model": response.requested_model,
                    "resolved_model": response.resolved_model,
                    "resolved_provider": response.resolved_provider,
                    "provider_response_id": response.provider_response_id,
                    "finish_reason": response.finish_reason,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                    "cache_read_tokens": response.cache_read_tokens,
                    "cache_write_tokens": response.cache_write_tokens,
                    "cost_usd": (
                        None if response.cost_usd is None else str(response.cost_usd)
                    ),
                    "latency_ns": response.latency_ns,
                    "attempt_count": attempts,
                },
            }
        )
        return raw


class SlotExecutor(Protocol):
    async def __call__(
        self,
        *,
        api_key: str,
        slot: SelectorSlot,
        historical_request: PortfolioSelectionRequest,
        queue_sink: Callable[[Mapping[str, object]], None],
        journal_sink: Callable[[Mapping[str, object]], None],
        pre_dispatch: Callable[
            [SelectorSlot, StructuredGenerationRequest[Any]], Mapping[str, object]
        ],
    ) -> PortfolioSelectionResult: ...


async def production_slot_executor(
    *,
    api_key: str,
    slot: SelectorSlot,
    historical_request: PortfolioSelectionRequest,
    queue_sink: Callable[[Mapping[str, object]], None],
    journal_sink: Callable[[Mapping[str, object]], None],
    pre_dispatch: Callable[
        [SelectorSlot, StructuredGenerationRequest[Any]], Mapping[str, object]
    ],
) -> PortfolioSelectionResult:
    """Run one slot in a fresh, independently owned provider/queue stack."""

    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=MAX_CONNECTIONS,
        timeout_seconds=ATTEMPT_TIMEOUT_NS / 1_000_000_000,
        provider_options={"only": list(PROVIDER_ONLY), "allow_fallbacks": False},
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        app_title="AgentEvolve AAAI 2027 selector-only stability",
    )
    queued = create_production_queued_runner(
        generator=structured,
        max_in_flight=MAX_IN_FLIGHT,
        max_pending=MAX_PENDING,
        max_attempts=MAX_ATTEMPTS,
        attempt_timeout_ns=ATTEMPT_TIMEOUT_NS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_policy=DeterministicHashJitter(
            seed=JITTER_SEED,
            domain=JITTER_DOMAIN,
        ),
        close_generator=True,
        outcome_sink=lambda outcome: queue_sink(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
    )
    low_runner = HistoricalScienceExecutionRunner(
        queued,
        slot=slot,
        historical_request=historical_request,
        pre_dispatch=pre_dispatch,
        journal_sink=journal_sink,
    )
    selector = PydanticAIPortfolioSelectionPolicy(low_runner)
    async with queued:
        return await selector.select(historical_request)


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    credential_loader: Callable[[], str]
    slot_executor: SlotExecutor
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic_ns: Callable[[], int] = time.monotonic_ns
    wall_time: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat()
    enforce_accounting: bool = True


def production_dependencies() -> LiveDependencies:
    def load_key() -> str:
        from agent_evolve.settings import load_credentials

        load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
        return os.environ.get("OPENROUTER_API_KEY", "")

    return LiveDependencies(
        credential_loader=load_key,
        slot_executor=production_slot_executor,
    )


def _response_record(
    slot: SelectorSlot,
    result: PortfolioSelectionResult,
) -> dict[str, object]:
    result.__post_init__()
    return {
        "schema_version": 1,
        **slot.to_record(),
        "decision": result.decision.to_record(),
        "telemetry": (
            None if result.telemetry is None else telemetry_record(result.telemetry)
        ),
    }


def _validate_slot_runtime_evidence(
    slot: SelectorSlot,
    *,
    queue_row: Mapping[str, object],
    journal_rows: Sequence[Mapping[str, object]],
    valid_response: bool,
) -> None:
    """Fail closed on missing or treatment-mutating runtime evidence."""

    if (
        queue_row.get("task_id") != slot.execution_call_id
        or type(queue_row.get("published_monotonic_ns")) is not int
        or int(queue_row["published_monotonic_ns"]) < 0
        or type(queue_row.get("published_wall_utc")) is not str
    ):
        raise RuntimeError("slot queue terminal publication evidence is incomplete")
    attempts = queue_row.get("attempts")
    if type(attempts) is not list or not 1 <= len(attempts) <= MAX_ATTEMPTS:
        raise RuntimeError("slot queue attempt ledger escaped the 1..2 bound")
    for ordinal, attempt in enumerate(attempts, start=1):
        if type(attempt) is not dict or attempt.get("attempt_number") != ordinal:
            raise RuntimeError("slot queue attempt ordering drifted")
        evidence = attempt.get("request_evidence")
        if (
            type(evidence) is not dict
            or evidence.get("variant") != "original"
            or evidence.get("prompt_sha256") != PROMPT_SHA256[slot.view_id]
        ):
            raise RuntimeError("slot retry changed or lost the frozen prompt")
        for field in (
            "wait_time_ns",
            "service_time_ns",
            "policy_backoff_ns",
            "retry_after_ns",
            "scheduled_delay_ns",
        ):
            if type(attempt.get(field)) is not int or int(attempt[field]) < 0:
                raise RuntimeError("slot attempt timing/backoff evidence is incomplete")
    if len(attempts) == 2:
        first = attempts[0]
        classification = first.get("classification")
        if (
            first.get("will_retry") is not True
            or type(classification) is not dict
            or classification.get("disposition") != "retry"
            or classification.get("reason")
            not in {"timeout", "rate_limit", "transient"}
        ):
            raise RuntimeError("slot retry predecessor was not allowlisted transport")

    requests = [row for row in journal_rows if row.get("record_type") == "request"]
    terminals = [
        row
        for row in journal_rows
        if row.get("record_type")
        in {"response", "response_failure", "response_route_rejected"}
    ]
    if len(requests) != 1 or len(terminals) != 1:
        raise RuntimeError("slot prompt/response journal is not one request/terminal")
    request = requests[0]
    if (
        request.get("absolute_slot") != slot.absolute_slot
        or request.get("view_id") != slot.view_id
        or request.get("execution_call_id") != slot.execution_call_id
        or request.get("operation") != "select_portfolio"
        or request.get("prompt_sha256") != PROMPT_SHA256[slot.view_id]
        or request.get("prompt_utf8_bytes") != PROMPT_UTF8_BYTES[slot.view_id]
        or request.get("output_schema_sha256") != OUTPUT_SCHEMA_SHA256
        or request.get("output_schema_utf8_bytes") != OUTPUT_SCHEMA_UTF8_BYTES
        or request.get("output_tool_name") != PORTFOLIO_SELECTION_TOOL_NAME
        or request.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or request.get("temperature") != 0.0
    ):
        raise RuntimeError("slot request journal escaped the frozen treatment")
    if valid_response and (
        queue_row.get("status") != "succeeded"
        or terminals[0].get("record_type") != "response"
    ):
        raise RuntimeError("valid logical response disagrees with queue/journal")
    if not valid_response and queue_row.get("status") != "succeeded" and (
        terminals[0].get("record_type") != "response_failure"
    ):
        raise RuntimeError("failed queue outcome disagrees with logical journal")


def _stability_record(
    values: Sequence[tuple[str, ...]],
) -> dict[str, object]:
    if len(values) != 3 or any(len(value) != 3 for value in values):
        raise ValueError("stability requires three complete size-three portfolios")
    pairs = tuple(itertools.combinations(values, 2))
    exact_pairs = sum(left == right for left, right in pairs)
    top_one_counts = Counter(value[0] for value in values)
    modal_frequency = max(top_one_counts.values()) / 3

    def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
        left_set = set(left)
        right_set = set(right)
        return len(left_set & right_set) / len(left_set | right_set)

    mean_jaccard = sum(jaccard(left, right) for left, right in pairs) / 3
    return {
        "ordered_portfolios": [list(value) for value in values],
        "exact_ordered_tuple_pair_count": exact_pairs,
        "modal_top_one_frequency": modal_frequency,
        "mean_pairwise_top_three_set_jaccard": mean_jaccard,
        "stable": modal_frequency >= 2 / 3 and mean_jaccard >= 2 / 3,
    }


def _analyze_complete_batch(
    oracle: VerifiedSealedOracle,
    results: Mapping[int, PortfolioSelectionResult],
) -> dict[str, object]:
    if set(results) != set(range(1, 10)):
        raise ValueError("scientific analysis requires all nine valid slots")
    by_view: dict[str, list[tuple[str, ...]]] = {"M": [], "P": [], "N": []}
    block_records = []
    for block in range(1, 4):
        block_slots = tuple(slot for slot in SCHEDULE if slot.block == block)
        block_results = {
            slot.view_id: results[slot.absolute_slot] for slot in block_slots
        }
        scored = selector_result_record(oracle, block_results)
        views = {
            str(row["view_id"]): row for row in scored["views"]
        }
        for slot in block_slots:
            members = results[slot.absolute_slot].decision.members
            by_view[slot.view_id].append(
                tuple(member.option_id for member in members)
            )
        block_records.append(
            {
                "block": block,
                "slot_by_view": {
                    slot.view_id: slot.absolute_slot for slot in block_slots
                },
                "views": list(scored["views"]),
                "survival_gates": scored["survival_gates"],
                "passes_scientific_gate": scored["survives_stage_a_v1"],
            }
        )
    stability = {
        view_id: _stability_record(values)
        for view_id, values in by_view.items()
    }
    stability_passes = all(row["stable"] for row in stability.values())
    scientific_blocks = sum(
        bool(row["passes_scientific_gate"]) for row in block_records
    )
    if not stability_passes:
        decision = "selector_unstable"
    elif scientific_blocks < 2:
        decision = "do_not_advance"
    else:
        decision = "advance_to_prospective_transfer_only"
    return {
        "schema_version": 1,
        "transport_complete": True,
        "stability": stability,
        "stability_gate_passes": stability_passes,
        "blocks": block_records,
        "scientific_blocks_passing": scientific_blocks,
        "scientific_gate_passes": scientific_blocks >= 2,
        "decision": decision,
        "claim_boundary": "post_hoc_development_not_held_out_efficacy",
    }


def execute_with_dependencies(
    manifest_path: Path,
    dependencies: LiveDependencies,
) -> dict[str, object]:
    """Execute exactly nine serial slots and always seal the resulting ledger."""

    verified = verify_manifest(manifest_path, require_output_absent=True)
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen._directory_fsync(run_dir.parent)
    shutil.copyfile(verified.path, run_dir / "launch_manifest.json")
    with (run_dir / "launch_manifest.json").open("rb") as stream:
        os.fsync(stream.fileno())
    frozen._directory_fsync(run_dir)
    frozen.write_json_atomic(run_dir / "authenticated_bank.json", verified.bank.record)
    frozen.write_json_atomic(
        run_dir / "schedule.json",
        {
            "schema_version": 1,
            "schedule": [slot.to_record() for slot in SCHEDULE],
        },
    )

    queue_writer = frozen.DurableJsonlWriter(
        run_dir / "provider_queue_outcomes.jsonl"
    )
    journal_writer = frozen.DurableJsonlWriter(
        run_dir / "prompt_response_journal.jsonl"
    )
    disposition_writer = frozen.DurableJsonlWriter(
        run_dir / "slot_dispositions.jsonl"
    )
    source_writer = frozen.DurableJsonlWriter(run_dir / "source_verifications.jsonl")
    quiet_writer = frozen.DurableJsonlWriter(run_dir / "quiet_intervals.jsonl")
    queue_rows: list[dict[str, object]] = []
    journal_rows: list[dict[str, object]] = []
    disposition_rows: list[dict[str, object]] = []
    rows_lock = threading.Lock()
    results: dict[int, PortfolioSelectionResult] = {}
    credentials_read = False
    status = "failed"
    summary: dict[str, object] | None = None
    pending: BaseException | None = None
    try:
        source_writer.write(
            {"stage": "post_run_directory_creation", **reverify_source(verified)}
        )
        source_writer.write(
            {"stage": "pre_credential_load", **reverify_source(verified)}
        )
        api_key = dependencies.credential_loader()
        credentials_read = True
        if type(api_key) is not str or not api_key.strip():
            raise RuntimeError("OPENROUTER_API_KEY is unavailable")

        def queue_sink(value: Mapping[str, object]) -> None:
            row = dict(value)
            if "published_monotonic_ns" in row or "published_wall_utc" in row:
                raise RuntimeError("queue outcome supplied reserved publication fields")
            row["published_monotonic_ns"] = dependencies.monotonic_ns()
            row["published_wall_utc"] = dependencies.wall_time()
            queue_writer.write(row)
            with rows_lock:
                queue_rows.append(row)

        def journal_sink(value: Mapping[str, object]) -> None:
            row = dict(value)
            journal_writer.write(row)
            with rows_lock:
                journal_rows.append(row)

        def pre_dispatch(
            slot: SelectorSlot,
            request: StructuredGenerationRequest[Any],
        ) -> Mapping[str, object]:
            if request.call_id.value != slot.execution_call_id:
                raise RuntimeError("pre-dispatch execution call ID drifted")
            verification = {
                "stage": "pre_provider_dispatch",
                "absolute_slot": slot.absolute_slot,
                "view_id": slot.view_id,
                "execution_call_id": slot.execution_call_id,
                **reverify_dispatch_inputs(verified),
            }
            source_writer.write(verification)
            return verification

        async def run_slots() -> None:
            for slot in SCHEDULE:
                queue_before = len(queue_rows)
                journal_before = len(journal_rows)
                historical = verified.bank.requests[slot.view_id]
                failure: Exception | None = None
                value: PortfolioSelectionResult | None = None
                try:
                    value = await dependencies.slot_executor(
                        api_key=api_key,
                        slot=slot,
                        historical_request=historical,
                        queue_sink=queue_sink,
                        journal_sink=journal_sink,
                        pre_dispatch=pre_dispatch,
                    )
                    if type(value) is not PortfolioSelectionResult:
                        raise TypeError("slot executor returned an invalid result")
                    value.__post_init__()
                except Exception as exc:
                    failure = exc

                new_queue_rows = queue_rows[queue_before:]
                if (
                    len(new_queue_rows) != 1
                    or new_queue_rows[0].get("task_id") != slot.execution_call_id
                ):
                    raise RuntimeError(
                        "slot stack did not publish exactly one durable queue outcome"
                    )
                new_journal_rows = journal_rows[journal_before:]
                _validate_slot_runtime_evidence(
                    slot,
                    queue_row=new_queue_rows[0],
                    journal_rows=new_journal_rows,
                    valid_response=failure is None,
                )
                if failure is None:
                    assert value is not None
                    response_record = _response_record(slot, value)
                    response_path = run_dir / f"slot_{slot.absolute_slot:02d}_response.json"
                    frozen.write_json_atomic(response_path, response_record)
                    response_binding = frozen._file_binding(response_path)
                    results[slot.absolute_slot] = value
                    disposition = {
                        "schema_version": 1,
                        **slot.to_record(),
                        "status": "valid_response",
                        "queue_status": new_queue_rows[0].get("status"),
                        "queue_attempt_count": len(
                            new_queue_rows[0].get("attempts", [])
                        ),
                        "response_file": response_binding,
                    }
                else:
                    disposition = {
                        "schema_version": 1,
                        **slot.to_record(),
                        "status": "terminal_failure",
                        "queue_status": new_queue_rows[0].get("status"),
                        "queue_attempt_count": len(
                            new_queue_rows[0].get("attempts", [])
                        ),
                        "failure_type": type(failure).__name__,
                        "safe_message": "logical selector slot ended terminally",
                    }
                disposition_writer.write(disposition)
                disposition_rows.append(disposition)

                # The slot executor has returned only after its fresh stack is
                # closed; the queue row and logical disposition are durable.
                if slot.absolute_slot < len(SCHEDULE):
                    quiet_start_ns = dependencies.monotonic_ns()
                    quiet_start_wall = dependencies.wall_time()
                    await dependencies.sleep(QUIET_INTERVAL_SECONDS)
                    quiet_end_ns = dependencies.monotonic_ns()
                    quiet_end_wall = dependencies.wall_time()
                    if quiet_end_ns - quiet_start_ns < 5_000_000_000:
                        raise RuntimeError("quiet interval was shorter than five seconds")
                    quiet_writer.write(
                        {
                            "schema_version": 1,
                            "after_absolute_slot": slot.absolute_slot,
                            "before_absolute_slot": slot.absolute_slot + 1,
                            "start_monotonic_ns": quiet_start_ns,
                            "end_monotonic_ns": quiet_end_ns,
                            "elapsed_ns": quiet_end_ns - quiet_start_ns,
                            "requested_seconds": QUIET_INTERVAL_SECONDS,
                            "start_wall_utc": quiet_start_wall,
                            "end_wall_utc": quiet_end_wall,
                        }
                    )

        asyncio.run(run_slots())
        disposition_table = {
            "schema_version": 1,
            "schedule_complete": len(disposition_rows) == 9,
            "slots": disposition_rows,
            "valid_response_count": len(results),
            "terminal_failure_count": 9 - len(results),
        }
        frozen.write_json_atomic(
            run_dir / "slot_disposition_table.json",
            disposition_table,
        )
        physical_attempts = sum(
            len(row.get("attempts", [])) for row in queue_rows
        )
        accounting = {
            "logical_slots": len(disposition_rows),
            "terminal_queue_outcomes": len(queue_rows),
            "physical_attempts": physical_attempts,
            "maximum_physical_attempts": 18,
            "all_execution_ids_match": (
                [row.get("task_id") for row in queue_rows]
                == [slot.execution_call_id for slot in SCHEDULE]
            ),
            "nine_dispositions": len(disposition_rows) == 9,
            "within_physical_cap": physical_attempts <= 18,
            "all_attempt_counts_in_range": all(
                type(row.get("attempts")) is list
                and 1 <= len(row["attempts"]) <= MAX_ATTEMPTS
                for row in queue_rows
            ),
            "request_journal_count": sum(
                row.get("record_type") == "request" for row in journal_rows
            ),
            "all_queue_publications_timestamped": all(
                type(row.get("published_monotonic_ns")) is int
                and type(row.get("published_wall_utc")) is str
                for row in queue_rows
            ),
        }
        accounting["passed"] = all(
            bool(accounting[key])
            for key in (
                "all_execution_ids_match",
                "nine_dispositions",
                "within_physical_cap",
                "all_attempt_counts_in_range",
                "all_queue_publications_timestamped",
            )
        ) and len(queue_rows) == 9 and accounting["request_journal_count"] == 9
        if dependencies.enforce_accounting and not accounting["passed"]:
            raise RuntimeError("selector-only provider accounting drifted")

        if len(results) == 9:
            analysis = _analyze_complete_batch(verified.bank.oracle, results)
            frozen.write_json_atomic(run_dir / "analysis.json", analysis)
            scientific_decision = analysis["decision"]
        else:
            analysis = None
            scientific_decision = "transport_incomplete"
        status = "completed_selector_only_study"
        summary = {
            "schema_version": 1,
            "status": status,
            "run_id": verified.run_id,
            "design_id": DESIGN_ID,
            "execution_class": EXECUTION_CLASS,
            "manifest_sha256": verified.manifest_sha256,
            "credentials_read": credentials_read,
            "provider_accounting": accounting,
            "transport_complete": len(results) == 9,
            "valid_response_count": len(results),
            "scientific_analysis_performed": analysis is not None,
            "decision": scientific_decision,
            "reflection_calls": 0,
            "candidate_evaluations": 0,
            "cfd_calls": 0,
            "claim_boundary": verified.record["claim_boundary"],
        }
        frozen.write_json_atomic(run_dir / "summary.json", summary)
    except BaseException as exc:
        pending = exc
        frozen.write_json_atomic(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "design_id": DESIGN_ID,
                "execution_class": EXECUTION_CLASS,
                "failure_type": type(exc).__name__,
                "safe_message": "selector-only infrastructure failed closed",
                "credentials_read": credentials_read,
                "completed_slot_dispositions": len(disposition_rows),
            },
        )
    finally:
        queue_writer.close()
        journal_writer.close()
        disposition_writer.close()
        source_writer.close()
        quiet_writer.close()
        finalization = frozen._finalize(run_dir, status=status)
    if pending is not None:
        raise pending
    assert summary is not None
    return {**summary, "finalization": finalization}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-manifest", type=Path, metavar="PATH")
    modes.add_argument("--verify-manifest", type=Path, metavar="PATH")
    modes.add_argument("--live", type=Path, metavar="MANIFEST")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--release-evidence",
        type=Path,
        default=DEFAULT_RELEASE_EVIDENCE,
    )
    return parser


def _required(parser: argparse.ArgumentParser, value: object, flag: str) -> object:
    if value is None:
        parser.error(f"{flag} is required for the selected mode")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.build_manifest is not None:
        record: object = write_manifest(
            args.build_manifest,
            run_id=str(_required(parser, args.run_id, "--run-id")),
            output_dir=Path(_required(parser, args.output_dir, "--output-dir")),
            release_evidence_path=args.release_evidence,
        )
    elif args.verify_manifest is not None:
        if (
            args.run_id is not None
            or args.output_dir is not None
            or args.release_evidence != DEFAULT_RELEASE_EVIDENCE
        ):
            parser.error("verification rejects build-only target arguments")
        verified = verify_manifest(args.verify_manifest)
        record = {
            "status": "verified_provider_ready",
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            "authenticated_cards": 8,
            "authorized_logical_calls": 9,
            "maximum_physical_attempts": 18,
            "credentials_read": False,
            "provider_dispatch_performed": False,
        }
    else:
        if (
            args.run_id is not None
            or args.output_dir is not None
            or args.release_evidence != DEFAULT_RELEASE_EVIDENCE
        ):
            parser.error("--live rejects build-only target arguments")
        record = execute_with_dependencies(args.live, production_dependencies())
    print(json.dumps(record, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
