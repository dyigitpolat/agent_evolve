from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

from agent_evolve.application.llm_task_queue import AsyncLLMTaskQueue
from agent_evolve.infrastructure.asyncio_runtime import AsyncioRuntime
from agent_evolve.infrastructure.clock import SystemClock
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    OutcomePublicationPolicy,
    QueuedStructuredGenerationRunner,
    StructuredGenerationExecutor,
    TransportOnlyStructuredGenerationRetryClassifier,
)
from agent_evolve.policies.llm_backoff import ExponentialBackoff, NoJitter
from agent_evolve.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import (
    run_airfoil_v7_forecast_wire_v3_pilot as launcher,
)


@pytest.fixture(scope="module")
def bundle() -> launcher.PilotBundle:
    return launcher.build_pilot_bundle()


@pytest.fixture
def prepare_cached(
    bundle: launcher.PilotBundle,
) -> Callable[[Path, Path], dict[str, object]]:
    def prepare(path: Path, target: Path) -> dict[str, object]:
        # One real reconstruction above is sufficient.  Each prepare path
        # still writes and finalizes every exact artifact, using that same
        # immutable bundle, without paying to rebuild the frozen contract.
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                launcher,
                "build_pilot_bundle",
                lambda **_kwargs: bundle,
            )
            result = launcher.execute_prepare(
                run_dir=path,
                target_live_run_dir=target,
            )
        assert result["prepared"]["status"] == "prepared"
        return result

    return prepare


@dataclass(frozen=True)
class _HealthReceipt:
    member_id: str
    passes: bool

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "frame_kind": "partition_block",
            "member_id": self.member_id,
            "passes": self.passes,
        }


def _health_assessor(passes: bool):
    def assess(_request, _block, *, member_id, health_policy):
        health_policy.__post_init__()
        return _HealthReceipt(member_id=member_id, passes=passes)

    return assess


def _subset_health_assessor(passes: bool):
    def assess(
        _request,
        _block,
        *,
        member_id,
        health_policy,
        subset_policy,
        included_global_row_indices,
    ):
        health_policy.__post_init__()
        assert subset_policy == launcher.eligible_subset_policy()
        assert included_global_row_indices == (
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
        )
        return _HealthReceipt(member_id=member_id, passes=passes)

    return assess


class _ThreeCallGenerator:
    def __init__(
        self,
        *,
        progress_sink,
        run_dir: Path,
        state: dict[str, object],
        fail_call_id: str | None = None,
    ) -> None:
        self._progress_sink = progress_sink
        self._run_dir = run_dir
        self._state = state
        self._fail_call_id = fail_call_id
        self._wave = asyncio.Event()
        self._lock = asyncio.Lock()
        self._starts = 0
        self._active = 0

    def _terminal_progress(self, request) -> None:
        attempt_id = request.provider_attempt_id
        assert attempt_id is not None
        digest = hashlib.sha256(b"").hexdigest()
        for sequence, kind in enumerate(
            (
                StructuredStreamProgressKind.OUTPUT_SELECTED,
                StructuredStreamProgressKind.STREAM_COMPLETED,
            ),
            start=1,
        ):
            self._progress_sink(
                StructuredStreamProgress(
                    call_id=request.call_id.value,
                    sequence=sequence,
                    kind=kind,
                    channel=StructuredStreamChannel.OTHER,
                    elapsed_ns=sequence,
                    event_content_utf8_bytes=0,
                    cumulative_content_utf8_bytes=0,
                    rolling_content_sha256=digest,
                    provider_attempt_id=attempt_id.value,
                )
            )

    async def generate_once(self, request):
        # Scientific provider requests and their exact prompts/schemas must be
        # durably frozen before any queue delegate receives a call.
        assert (self._run_dir / "planned_block_wave.json").is_file()
        assert len(launcher.read_jsonl(self._run_dir / "planned_calls.jsonl")) == 3
        async with self._lock:
            self._starts += 1
            self._active += 1
            self._state["starts"] = self._starts
            self._state["max_concurrent"] = max(
                int(self._state.get("max_concurrent", 0)), self._active
            )
            if self._starts == 3:
                self._wave.set()
        await asyncio.wait_for(self._wave.wait(), timeout=5.0)
        try:
            if request.call_id.value == self._fail_call_id:
                raise ValueError("deterministic fake provider failure")
            payload = launcher.v2_launcher._schema_driven_action_forecast_payload(
                request.output_type
            )
            value = request.output_type.model_validate(payload)
            self._terminal_progress(request)
            return StructuredGenerationResponse(
                value=value,
                requested_model=launcher.MODEL,
                resolved_model=launcher.CANONICAL_MODEL,
                resolved_provider=launcher.RESOLVED_PROVIDER,
                provider_response_id=f"fake-{request.call_id.value}",
                finish_reason="tool_call",
                input_tokens=100,
                output_tokens=100,
                reasoning_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0.001"),
                latency_ns=1_000,
            )
        finally:
            async with self._lock:
                self._active -= 1


class _RunnerFactory:
    def __init__(
        self,
        *,
        run_dir: Path,
        state: dict[str, object],
        fail_call_id: str | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._state = state
        self._fail_call_id = fail_call_id

    def __call__(self, *, api_key, config, progress_sink, outcome_sink):
        assert api_key == "fake-key"
        assert config.to_manifest_record() == launcher.build_config().to_manifest_record()
        generator = _ThreeCallGenerator(
            progress_sink=progress_sink,
            run_dir=self._run_dir,
            state=self._state,
            fail_call_id=self._fail_call_id,
        )
        queue = AsyncLLMTaskQueue(
            executor=StructuredGenerationExecutor(generator),
            retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
            backoff_policy=ExponentialBackoff(0, 0, NoJitter()),
            clock=SystemClock(),
            max_in_flight=3,
            max_pending=3,
            attempt_timeout_ns=None,
            runtime=AsyncioRuntime(),
        )
        return QueuedStructuredGenerationRunner(
            queue=queue,
            max_attempts=2,
            outcome_sink=outcome_sink,
            outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        )


def _execute_fake_live(
    *,
    bundle: launcher.PilotBundle,
    prepared_dir: Path,
    live_dir: Path,
    health_passes: bool,
    fail_call_id: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    state: dict[str, object] = {}
    # claim_live still checks finalization, commitments, exact source closure,
    # protocol, arm identities, and provider request wave; only the already
    # tested deterministic reconstruction is memoized in this focused suite.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            launcher,
            "build_pilot_bundle",
            lambda **_kwargs: bundle,
        )
        claim = launcher.claim_live(prepared_dir=prepared_dir, run_dir=live_dir)
    result = launcher.execute_live(
        claim=claim,
        api_key="fake-key",
        dependencies=launcher.LiveDependencies(
            runner_factory=_RunnerFactory(
                run_dir=live_dir,
                state=state,
                fail_call_id=fail_call_id,
            ),
            block_health_assessor=_health_assessor(health_passes),
            block_subset_health_assessor=_subset_health_assessor(
                health_passes
            ),
        ),
    )
    return result, state


def test_reconstructs_frozen_v2_and_rejects_tampering(
    bundle: launcher.PilotBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
) -> None:
    assert bundle.preparation.contract.identity_sha256 == launcher.EXPECTED_CONTRACT_SHA256
    assert bundle.arms.source_registry.registry_sha256 == launcher.EXPECTED_SOURCE_REGISTRY_SHA256
    assert bundle.reflection.logical_llm_calls_used == 1
    assert len(bundle.reflection.shards) == 8

    reflection = launcher._load_object(
        launcher.DEFAULT_FROZEN_V2_RUN / "reflection_result.json"
    )
    expected = tuple(
        sorted(value.contrast_id for value in bundle.preparation.observations)
    )
    cards = reflection["cards"]
    assert type(cards) is list
    cards[0]["draft_content_sha256"] = "0" * 64
    with pytest.raises(launcher.ForecastWirePilotError, match="draft hash"):
        launcher.reflection_from_record(
            reflection,
            expected_contrast_ids=expected,
        )

    prepared_dir = tmp_path / "prepared"
    prepare_cached(prepared_dir, tmp_path / "live")
    tampered = tmp_path / "tampered_prepared"
    shutil.copytree(prepared_dir, tampered)
    protocol = launcher._load_object(tampered / "protocol.json")
    protocol["scientific_scope"] = "changed"
    launcher.write_json_atomic(tampered / "protocol.json", protocol)
    with pytest.raises(Exception):
        launcher.verify_prepared(tampered)


def test_deterministic_enum_only_twenty_row_block_and_blinded_calls(
    bundle: launcher.PilotBundle,
) -> None:
    assert bundle.layout.layout_sha256 == (
        "fd823ec5d0ba9505bd110c2c1ed30c0982c5a9d5f05b422f27908838e2bf4abe"
    )
    assert bundle.selection_record["selection_digest"] == (
        "fb83f355ebc4aac3cc836a8ec3679451309ed9b7efc9e3cc73337204b285aef3"
    )
    assert bundle.selected_block_index == 3
    assert bundle.eligible_g2_global_row_indices == (
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
    )
    assert len(bundle.selected_block_requests) == 3
    for block_request, plan in zip(
        bundle.selected_block_requests, bundle.planned_calls, strict=True
    ):
        assert block_request.block.global_row_start == 60
        assert block_request.block.global_row_stop == 80
        assert plan.max_output_tokens == 384_000
        schema = plan.output_type.model_json_schema()
        assert launcher._schema_contains_provider_number(schema) is False
        properties = schema["properties"]
        assert properties["probability_valid_codes"]["minItems"] == 20
        assert properties["probability_valid_codes"]["maxItems"] == 20
        median_codes = properties["median_effect_codes"]["items"]["items"]["enum"]
        uncertainty_codes = properties["lower_uncertainty_codes"]["items"]["items"]["enum"]
        assert {"n32", "p32"}.issubset(median_codes)
        assert "u32" in uncertainty_codes
        lines = plan.prompt.splitlines()
        contract = json.loads(
            lines[lines.index("ALL-OPTION ACTION FORECAST CONTRACT") + 1]
        )
        assert [row["global_row_index"] for row in contract["ordered_options"]] == list(
            range(60, 80)
        )
        visible = "\n".join(
            (plan.call_id.value, plan.operation, plan.output_tool_name, plan.prompt)
        ).casefold()
        assert not any(
            token in visible
            for token in (
                "permuted_placebo",
                "control_arm",
                "portfolioexperimentalarm",
                '"arm"',
            )
        )
        lowered_call_id = plan.call_id.value.casefold()
        assert not any(
            token in lowered_call_id for token in ("memory", "placebo", "neutral")
        )
        assert not lowered_call_id.endswith(("_m", "_p", "_n"))


def test_historical_v4_v2_bundle_matches_sealed_planned_wave_exactly(
    bundle: launcher.PilotBundle,
) -> None:
    historical_root = launcher.DEFAULT_HISTORICAL_V4_PREPARED_RUN
    historical_prepared = launcher._load_object(historical_root / "prepared.json")
    historical_wave_path = historical_root / "planned_block_wave.json"
    historical_wave = launcher._load_object(historical_wave_path)

    historical_commitment = historical_prepared["preparation_commitment_sha256"]
    unsigned_prepared = dict(historical_prepared)
    del unsigned_prepared["preparation_commitment_sha256"]
    assert historical_commitment == launcher._sha256_record(
        launcher._PREPARED_FRAMING,
        unsigned_prepared,
    )
    assert historical_commitment == (
        launcher.EXPECTED_HISTORICAL_PREPARATION_COMMITMENT_SHA256
    )
    assert hashlib.sha256(historical_wave_path.read_bytes()).hexdigest() == (
        launcher.EXPECTED_HISTORICAL_PLANNED_WAVE_FILE_SHA256
    )
    assert launcher._planned_wave_record(bundle) == historical_wave
    assert bundle.arm_request_sha256s == launcher.EXPECTED_CURRENT_ARM_REQUEST_SHA256S
    for block_request, plan in zip(
        bundle.selected_block_requests,
        bundle.planned_calls,
        strict=True,
    ):
        assert plan == launcher.plan_action_forecast_v4_block_request(block_request)


def test_prepare_seals_exact_three_call_precommit_and_source_closure(
    bundle: launcher.PilotBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    target = (tmp_path / "authorized_live").resolve()
    prepare_cached(prepared_dir, target)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            launcher,
            "build_pilot_bundle",
            lambda **_kwargs: bundle,
        )
        verified = launcher.verify_prepared(prepared_dir)
    assert verified.record["credential_read_attempted"] is False
    assert verified.record["frozen_g1_terminal_records_rehydrated"] == 8
    assert verified.record["new_candidate_evaluations"] == 0
    assert verified.record["authorized_target_live_run_dir"] == str(target)
    assert len(verified.wave["calls"]) == 3
    assert verified.wave["status"] == "durably_precommitted_before_live_credential_read"
    protocol = launcher._load_object(prepared_dir / "protocol.json")
    assert protocol["forecast_policy"]["provider_wire_version"] == 4
    assert protocol["forecast_policy"]["policy_definition_sha256"] == (
        launcher.EXPECTED_PROVIDER_WIRE_POLICY_DEFINITION_SHA256
    )
    assert protocol["eligible_g2_subset_health"]["included_global_row_indices"] == list(
        bundle.eligible_g2_global_row_indices
    )
    paths = {
        row["path"]
        for row in verified.record["closed_source_identity"]["files"]
    }
    assert any(path.endswith("run_airfoil_v7_forecast_wire_v3_pilot.py") for path in paths)
    assert any(path.endswith("test_run_airfoil_v7_forecast_wire_v3_pilot.py") for path in paths)
    assert not any(
        (prepared_dir / name).exists()
        for name in ("allocations.json", "g2.json", "evaluations.json", "cfd.json")
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            launcher,
            "build_pilot_bundle",
            lambda **_kwargs: bundle,
        )
        with pytest.raises(
            launcher.ForecastWirePilotError,
            match="one-shot target",
        ):
            launcher.claim_live(
                prepared_dir=prepared_dir,
                run_dir=tmp_path / "unauthorized_live",
            )


@pytest.mark.parametrize(
    ("health_passes", "expected_status"),
    ((True, "wire_qualified"), (False, "typed_but_semantically_degenerate")),
)
def test_three_calls_settle_concurrently_and_emit_typed_health_statuses(
    bundle: launcher.PilotBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
    health_passes: bool,
    expected_status: str,
) -> None:
    live_dir = tmp_path / expected_status
    prepared_dir = tmp_path / "prepared"
    prepare_cached(prepared_dir, live_dir)
    execution, state = _execute_fake_live(
        bundle=bundle,
        prepared_dir=prepared_dir,
        live_dir=live_dir,
        health_passes=health_passes,
    )
    result = execution["result"]
    assert result["status"] == expected_status
    assert state["starts"] == 3
    assert state["max_concurrent"] == 3
    assert result["planned_logical_call_count"] == 3
    assert result["submitted_logical_call_count"] == 3
    assert result["new_logical_provider_calls"] == 3
    assert result["terminal_queue_outcome_count"] == 3
    assert result["accepted_typed_block_count"] == 3
    assert result["typed_wire_artifact_count"] == 3
    assert result["health_assessment_count"] == 3
    assert result["eligible_subset_health_assessment_count"] == 3
    assert result["qualification_counts"] == {
        "planned": 3,
        "submitted": 3,
        "terminal_outcomes": 3,
        "typed_wires": 3,
        "accepted_blocks": 3,
        "health_assessments": 3,
        "eligible_subset_health_assessments": 3,
    }
    assert result["physical_attempts"]["physical_attempt_count"] == 3
    assert result["physical_attempts"]["scheduled_retry_count"] == 0
    assert result["successful_attempt_validation"]["exact_original_payloads"] is True
    assert result["accepted_usage"]["accepted_response_count"] == 3
    assert result["accepted_usage"]["input_tokens"] == 300
    assert result["accepted_usage"]["output_tokens"] == 300
    assert result["accepted_usage"]["cost_usd"] == "0.003"
    assert result["provider_call_attempted"] is True
    assert len(launcher.read_jsonl(live_dir / "planned_calls.jsonl")) == 3
    assert len(launcher.read_jsonl(live_dir / "submitted_calls.jsonl")) == 3
    assert len(list(live_dir.glob("typed_wire_*.json"))) == 3
    assert len(list(live_dir.glob("resolved_block_*.json"))) == 3
    assert len(list(live_dir.glob("block_health_*.json"))) == 3
    assert len(list(live_dir.glob("eligible_g2_subset_health_*.json"))) == 3
    assert execution["finalization"]["status"] == expected_status
    for counter in (
        "new_candidate_evaluations",
        "allocation_calls",
        "g2_openings",
        "selected_action_evaluator_calls",
        "new_cfd_calls",
    ):
        assert result[counter] == 0


def test_failure_still_settles_three_calls_and_counts_actual_submissions(
    bundle: launcher.PilotBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
) -> None:
    live_dir = tmp_path / "incomplete"
    prepared_dir = tmp_path / "prepared"
    prepare_cached(prepared_dir, live_dir)
    failed_call = bundle.planned_calls[0].call_id.value
    execution, state = _execute_fake_live(
        bundle=bundle,
        prepared_dir=prepared_dir,
        live_dir=live_dir,
        health_passes=True,
        fail_call_id=failed_call,
    )
    result = execution["result"]
    assert result["status"] == "incomplete"
    assert state["starts"] == 3
    assert state["max_concurrent"] == 3
    assert result["planned_logical_call_count"] == 3
    assert result["submitted_logical_call_count"] == 3
    assert result["new_logical_provider_calls"] == 3
    assert result["terminal_queue_outcome_count"] == 3
    assert result["accepted_typed_block_count"] == 2
    assert result["health_assessment_count"] == 0
    assert result["eligible_subset_health_assessment_count"] == 0
    assert result["physical_attempts"]["physical_attempt_count"] == 3
    assert result["physical_attempts"]["scheduled_retry_count"] == 0
    assert result["provider_call_attempted"] is True
    assert execution["finalization"]["status"] == "incomplete"
    assert result["allocation_calls"] == result["g2_openings"] == 0
    assert result["selected_action_evaluator_calls"] == result["new_cfd_calls"] == 0
