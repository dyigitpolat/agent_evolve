from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import Decimal
import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

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
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import (
    run_airfoil_v7_v5_paired_causal_trial as trial,
)


@pytest.fixture(scope="module")
def bundle() -> trial.TrialBundle:
    """Pay the exact frozen-evidence reconstruction cost only once."""

    return trial.build_trial_bundle()


@pytest.fixture
def prepare_cached(
    bundle: trial.TrialBundle,
) -> Callable[[Path, Path], dict[str, object]]:
    def prepare(path: Path, target: Path) -> dict[str, object]:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                trial,
                "build_trial_bundle",
                lambda **_kwargs: bundle,
            )
            result = trial.execute_prepare(
                run_dir=path,
                target_live_run_dir=target,
            )
        assert result["prepared"]["status"] == "prepared"
        return result

    return prepare


def _verify_cached(
    *,
    bundle: trial.TrialBundle,
    prepared_dir: Path,
) -> trial.VerifiedPreparation:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trial,
            "build_trial_bundle",
            lambda **_kwargs: bundle,
        )
        return trial.verify_prepared(prepared_dir)


def _healthy_wire_payload(
    request: StructuredGenerationRequest[Any],
    *,
    arm_index: int,
) -> dict[str, object]:
    """Return a deterministic varied v5 matrix admitted by the real schema."""

    payload = trial.v2_launcher._schema_driven_action_forecast_payload(
        request.output_type
    )
    medians = payload["median_effect_codes"]
    lower = payload["lower_uncertainty_codes"]
    upper = payload["upper_uncertainty_codes"]
    probabilities = payload["probability_valid_codes"]
    assert type(medians) is list
    assert type(lower) is list
    assert type(upper) is list
    assert type(probabilities) is list
    effect_codes = (
        "n2",
        "n1",
        "n0_5",
        "n0_25",
        "z",
        "p0_25",
        "p0_5",
        "p1",
        "p2",
    )
    uncertainty_codes = ("u0_25", "u0_5", "u1", "u2")
    validity_codes = ("p0_6", "p0_8", "p0_95")
    for row_index, row in enumerate(medians):
        assert type(row) is list
        assert type(lower[row_index]) is list
        assert type(upper[row_index]) is list
        probabilities[row_index] = validity_codes[
            (row_index + arm_index) % len(validity_codes)
        ]
        for metric_index in range(len(row)):
            row[metric_index] = effect_codes[
                (row_index * (metric_index + 1) + 2 * arm_index)
                % len(effect_codes)
            ]
            lower[row_index][metric_index] = uncertainty_codes[
                (row_index + metric_index + arm_index)
                % len(uncertainty_codes)
            ]
            upper[row_index][metric_index] = uncertainty_codes[
                (2 * row_index + metric_index + arm_index + 1)
                % len(uncertainty_codes)
            ]
    return payload


class _ThreeCallGenerator:
    def __init__(
        self,
        *,
        progress_sink: Callable[[StructuredStreamProgress], None],
        run_dir: Path,
        state: dict[str, object],
        collapsed: bool,
        fail_call_id: str | None,
    ) -> None:
        self._progress_sink = progress_sink
        self._run_dir = run_dir
        self._state = state
        self._collapsed = collapsed
        self._fail_call_id = fail_call_id
        self._wave = asyncio.Event()
        self._lock = asyncio.Lock()
        self._starts = 0
        self._active = 0

    def _terminal_progress(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> None:
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

    async def generate_once(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        assert (self._run_dir / "planned_opaque_wave.json").is_file()
        assert len(trial.read_jsonl(self._run_dir / "planned_calls.jsonl")) == 3
        async with self._lock:
            self._starts += 1
            self._active += 1
            self._state["starts"] = self._starts
            self._state["max_concurrent"] = max(
                int(self._state.get("max_concurrent", 0)),
                self._active,
            )
            order = list(self._state.get("submission_order", []))
            order.append(request.call_id.value)
            self._state["submission_order"] = order
            if self._starts == 3:
                self._wave.set()
        await asyncio.wait_for(self._wave.wait(), timeout=5.0)
        try:
            if request.call_id.value == self._fail_call_id:
                raise ValueError("deterministic fake provider failure")
            if self._collapsed:
                payload = trial.v2_launcher._schema_driven_action_forecast_payload(
                    request.output_type
                )
            else:
                payload = _healthy_wire_payload(
                    request,
                    arm_index=(self._starts + len(request.call_id.value)) % 3,
                )
            value = request.output_type.model_validate(payload)
            self._terminal_progress(request)
            return StructuredGenerationResponse(
                value=value,
                requested_model=trial.MODEL,
                resolved_model=trial.CANONICAL_MODEL,
                resolved_provider=trial.RESOLVED_PROVIDER,
                provider_response_id=f"provider-free-{request.call_id.value}",
                finish_reason="tool_call",
                input_tokens=100,
                output_tokens=200,
                reasoning_tokens=50,
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
        collapsed: bool = False,
        fail_call_id: str | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._state = state
        self._collapsed = collapsed
        self._fail_call_id = fail_call_id

    def __call__(
        self,
        *,
        api_key: str,
        config: object,
        progress_sink: Callable[[StructuredStreamProgress], None],
        outcome_sink: Callable[[object], None],
    ) -> QueuedStructuredGenerationRunner:
        assert api_key == "provider-free-injected-test-key"
        assert getattr(config, "to_manifest_record")() == (
            trial.build_config().to_manifest_record()
        )
        generator = _ThreeCallGenerator(
            progress_sink=progress_sink,
            run_dir=self._run_dir,
            state=self._state,
            collapsed=self._collapsed,
            fail_call_id=self._fail_call_id,
        )
        queue = AsyncLLMTaskQueue(
            executor=StructuredGenerationExecutor(generator),
            retry_classifier=(
                TransportOnlyStructuredGenerationRetryClassifier()
            ),
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


def _claim_cached(
    *,
    bundle: trial.TrialBundle,
    prepared_dir: Path,
    live_dir: Path,
) -> trial.LiveClaim:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trial,
            "build_trial_bundle",
            lambda **_kwargs: bundle,
        )
        return trial.claim_live(
            prepared_dir=prepared_dir,
            run_dir=live_dir,
        )


def test_prepare_releases_schedule_before_historical_evidence_decode(
    bundle: trial.TrialBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The prospective release must be durable before any frozen G1 decode."""

    prepared_dir = tmp_path / "prepared"
    target = (tmp_path / "authorized-live").resolve()
    observed: list[str] = []

    def build_after_release(**kwargs: object) -> trial.TrialBundle:
        observed.append("historical_evidence_decode")
        release_path = prepared_dir / "paired_block_assignment_release.json"
        assert release_path.is_file()
        release = trial._load_object(release_path)
        assert release["status"] == (
            "released_before_historical_g1_artifact_decode"
        )
        assert release["selected_block_index"] == 2
        assert release["oracle_outcome_file_reads_before_release"] == 0
        assert release["credentials_read_before_release"] is False
        assert release["provider_calls_before_release"] == 0
        schedule = kwargs.get("schedule_assignment")
        assert schedule == bundle.schedule_assignment
        return bundle

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepare crossed a live-only authority boundary")

    monkeypatch.setattr(trial, "build_trial_bundle", build_after_release)
    monkeypatch.setattr(
        trial.airfoil,
        "verify_airfoil_v7_predecision_oracle",
        forbidden,
    )
    monkeypatch.setattr(
        trial,
        "create_progress_aware_openrouter_runner",
        forbidden,
    )

    execution = trial.execute_prepare(
        run_dir=prepared_dir,
        target_live_run_dir=target,
    )

    assert observed == ["historical_evidence_decode"]
    prepared = execution["prepared"]
    assert prepared["credential_read_attempted"] is False
    assert prepared["provider_client_constructed"] is False
    assert prepared["provider_call_attempted"] is False
    assert prepared["oracle_outcome_file_reads"] == 0
    assert prepared["historical_g1_artifact_observations_rehydrated"] == 8
    assert prepared["new_candidate_evaluations"] == 0
    assert prepared["authorized_target_live_run_dir"] == str(target)
    assert not target.exists()

    wave = trial._load_object(prepared_dir / "planned_opaque_wave.json")
    calls = wave["calls_in_opaque_provider_slot_order"]
    assert type(calls) is list and len(calls) == 3
    assert wave["status"] == "durably_precommitted_before_live_credential_read"
    assert wave["provider_visible_treatment_labels"] is False
    assert wave["content_blinding_claimed"] is False

    protocol = trial._load_object(prepared_dir / "protocol.json")
    chronology = protocol["chronology"]
    assert type(chronology) is list
    assert chronology.index("paired_block_assignment_release_fsync") < (
        chronology.index("historical_g1_reflection_card_reconstruction")
    )
    assert chronology.index("exact_v5_provider_wave_precommit_fsync") < (
        chronology.index("live_claim_before_credential_read")
    )
    assert protocol["queue"] == trial.build_config().to_manifest_record()
    assert protocol["primary_endpoint"] == trial.primary_endpoint_record()


def test_preparation_tamper_is_rejected_by_recursive_finalization(
    bundle: trial.TrialBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    prepare_cached(prepared_dir, (tmp_path / "authorized-live").resolve())
    tampered = tmp_path / "tampered"
    shutil.copytree(prepared_dir, tampered)

    release = trial._load_object(
        tampered / "paired_block_assignment_release.json"
    )
    release["selected_block_index"] = 1
    trial.write_json_atomic(
        tampered / "paired_block_assignment_release.json",
        release,
    )

    with pytest.raises(RuntimeError, match="finalized file content changed"):
        _verify_cached(bundle=bundle, prepared_dir=tampered)


def test_live_claim_is_exact_target_one_shot_and_precredential(
    bundle: trial.TrialBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    authorized = (tmp_path / "authorized-live").resolve()
    unauthorized = tmp_path / "unauthorized-live"
    prepare_cached(prepared_dir, authorized)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trial,
            "build_trial_bundle",
            lambda **_kwargs: bundle,
        )
        with pytest.raises(
            trial.FreshPairedTrialError,
            match="live target differs",
        ):
            trial.claim_live(
                prepared_dir=prepared_dir,
                run_dir=unauthorized,
            )
        claim = trial.claim_live(
            prepared_dir=prepared_dir,
            run_dir=authorized,
        )

    assert not unauthorized.exists()
    assert claim.active is True
    assert claim.run_dir == authorized
    assert claim.claim_record["status"] == "claimed_before_credential_read"
    assert claim.claim_record["credential_read_attempted"] is False
    assert claim.claim_record["provider_client_constructed"] is False
    assert claim.claim_record["provider_call_attempted"] is False
    assert claim.claim_record["oracle_outcome_file_reads"] == 0
    assert trial._load_object(authorized / "precredential_claim.json") == (
        claim.claim_record
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trial,
            "build_trial_bundle",
            lambda **_kwargs: bundle,
        )
        with pytest.raises(FileExistsError):
            trial.claim_live(
                prepared_dir=prepared_dir,
                run_dir=authorized,
            )
    claim.close()
    assert claim.active is False


def test_paired_adjudicator_commits_both_methods_before_union_only_oracle(
    bundle: trial.TrialBundle,
    prepare_cached: Callable[[Path, Path], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the opaque queue, v5 decoder, allocators, and union firewall."""

    prepared_dir = tmp_path / "prepared"
    live_dir = (tmp_path / "live").resolve()
    prepare_cached(prepared_dir, live_dir)
    claim = _claim_cached(
        bundle=bundle,
        prepared_dir=prepared_dir,
        live_dir=live_dir,
    )
    state: dict[str, object] = {}

    real_verify = trial.airfoil.verify_airfoil_v7_predecision_oracle
    chronology_observed: list[str] = []
    rejection_observed: list[str] = []

    def verify_only_after_commits(path: Path):
        for name in (
            "durable_allocation_v2_commit.json",
            "durable_allocation_v3_commit.json",
            "paired_allocation_comparison_commitment.json",
            "airfoil_paired_allocation_commitment.json",
        ):
            assert (live_dir / name).is_file()
            assert type(trial._load_object(live_dir / name)) is dict
        access = trial._load_object(
            live_dir / "postcommit_oracle_access_started.json"
        )
        assert access["both_method_commits_read_back"] is True
        assert access["raw_authority"] == "committed_selected_union_only"
        chronology_observed.append("oracle_verified_after_both_commits")
        return real_verify(path)

    def checked_adjudicator(
        context: trial.PostLedgerContext,
    ) -> object:
        assert tuple(value[0] for value in context.accepted) == ("m", "p", "n")
        assert tuple(
            value[1].block_request_sha256 for value in context.accepted
        ) == tuple(
            value.block_request_sha256
            for value in bundle.selected_block_requests
        )

        foreign_result = (
            context.accepted[0][0],
            context.accepted[0][1],
            context.accepted[1][2],
        )
        with pytest.raises(ValueError):
            trial.PostLedgerContext(
                claim=context.claim,
                accepted=(
                    foreign_result,
                    context.accepted[1],
                    context.accepted[2],
                ),
                terminal_ledger=context.terminal_ledger,
            )
        rejection_observed.append("foreign_result")

        fake_ledger = dict(context.terminal_ledger)
        fake_ledger["successful_outcome_count"] = 2
        unsigned = dict(fake_ledger)
        unsigned.pop("commitment_sha256")
        fake_ledger["commitment_sha256"] = trial._hash(
            trial._TERMINAL_LEDGER_FRAMING,
            unsigned,
        )
        with pytest.raises(ValueError, match="durable read-back"):
            trial.PostLedgerContext(
                claim=context.claim,
                accepted=context.accepted,
                terminal_ledger=fake_ledger,
            )
        rejection_observed.append("authenticated_but_undurable_fake_ledger")
        return trial.adjudicate_airfoil_paired_allocations(context)

    monkeypatch.setattr(
        trial.airfoil,
        "verify_airfoil_v7_predecision_oracle",
        verify_only_after_commits,
    )
    execution = trial._execute_live_with_dependencies_for_test(
        claim=claim,
        dependencies=trial.LiveDependencies(
            runner_factory=_RunnerFactory(
                run_dir=live_dir,
                state=state,
            ),
            paired_adjudicator=checked_adjudicator,
        ),
    )
    result = execution["result"]

    assert chronology_observed == ["oracle_verified_after_both_commits"]
    assert rejection_observed == [
        "foreign_result",
        "authenticated_but_undurable_fake_ledger",
    ]
    assert state["starts"] == 3
    assert state["max_concurrent"] == 3
    assert state["submission_order"] == [
        value.plan.call_id.value for value in bundle.opaque_calls
    ]
    assert result["status"] == "provider_free_injected_test_completed"
    assert result["underlying_provider_free_test_status"] == (
        "completed_paired_selected_union_primary_endpoint"
    )
    assert result["release_eligible"] is False
    assert result["paid_provider_evidence"] is False
    assert result["terminal_queue_outcome_count"] == 3
    assert result["accepted_typed_block_count"] == 3
    assert result["health_pass_count"] == 3
    assert result["eligible_subset_health_pass_count"] == 3
    assert result["allocation_execution_count"] == 6
    paired_result = result["paired_result"]
    assert paired_result["allocation_method_count"] == 2
    assert paired_result["allocation_arm_count_per_method"] == 3
    assert paired_result["candidate_score_count_per_arm_per_method"] == 54
    assert paired_result["total_candidate_score_count"] == 324
    assert paired_result["logical_selected_evaluation_slots"] == 18
    assert 3 <= paired_result["unique_selected_cached_reads"] <= 18
    assert result["selected_action_evaluator_calls"] == (
        paired_result["unique_selected_cached_reads"]
    )
    assert result["new_candidate_evaluations"] == 0
    assert result["new_cfd_calls"] == 0
    assert paired_result["exact_three_set_rank_status"] == (
        "not_computed_without_separate_postcommit_reference_release"
    )
    assert result["terminal_provider_ledger_materialized"] is True
    terminal = trial._load_object(live_dir / "terminal_provider_ledger.json")
    assert terminal["successful_outcome_count"] == 3
    assert terminal["allocation_started_before_ledger_fsync"] is False
    assert execution["pending_error_type"] is None

    benchmark_commit = trial._load_object(
        live_dir / "airfoil_paired_allocation_commitment.json"
    )
    outcomes = trial._load_object(live_dir / "selected_union_outcomes.json")
    committed_union = set(benchmark_commit["selected_option_ids"])
    returned_union = {
        row["option_id"] for row in outcomes["unique_evaluations"]
    }
    logical_slot_ids = {
        row["option_id"] for row in outcomes["logical_slots"]
    }
    assert returned_union == committed_union == logical_slot_ids
    assert len(outcomes["logical_slots"]) == 18
    assert outcomes["unique_cached_read_count"] == len(committed_union)
    assert outcomes["raw_outcome_authority"] == (
        "committed_selected_union_only"
    )
    assert outcomes["unselected_outcomes_exposed"] is False
    assert outcomes["new_cfd_calls"] == 0
    endpoint = outcomes["primary_endpoint_analysis"]
    assert set(endpoint["endpoint_by_method_and_arm"]) == {
        "audited_frame_v2",
        "operational_frame_v3",
    }
    assert endpoint["exact_three_set_competition_rank"]["selected_set_ranks"] is None
    assert endpoint["exact_three_set_competition_rank"][
        "raw_unselected_outcomes_returned"
    ] is False


def test_endpoint_contract_uses_observable_selected_union_positive_contrasts() -> None:
    endpoint = trial.primary_endpoint_record()

    assert endpoint["direction"] == "lower_is_better"
    assert endpoint["observable_authority"] == {
        "raw_outcomes": "committed_selected_union_only",
        "portfolio_size": 3,
        "requires_unselected_raw_outcomes": False,
    }
    assert endpoint["primary_contrasts"] == {
        "within_method": [
            "endpoint_p_minus_endpoint_m",
            "endpoint_n_minus_endpoint_m",
        ],
        "paired_method": [
            "endpoint_m_v2_minus_v3",
            "endpoint_p_v2_minus_v3",
            "endpoint_n_v2_minus_v3",
        ],
        "positive_difference_favors_m_or_v3": True,
    }
    rank = endpoint["secondary_endpoints"]["exact_three_set_competition_rank"]
    assert rank["exact_three_set_count"] == 969
    assert rank["status"] == (
        "not_computed_without_separate_postcommit_reference_release"
    )
    assert endpoint["conditional_postcommit_rank_authority"][
        "required_for_primary_endpoint"
    ] is False
    assert endpoint["conditional_postcommit_rank_authority"]["status"] == (
        "not_computed_without_separate_postcommit_reference_release"
    )


def test_exact_three_set_rank_uses_competition_ranking() -> None:
    option_ids = tuple(f"option.{index:02d}" for index in range(19))
    deltas = {
        option_id: (0.0, float(index) * 0.001)
        for index, option_id in enumerate(option_ids)
    }

    best_rank, best_endpoint = trial.exact_three_set_rank(
        selected_option_ids=(option_ids[0], option_ids[1], option_ids[2]),
        eligible_option_ids=option_ids,
        metric_deltas=deltas,
    )
    worst_rank, worst_endpoint = trial.exact_three_set_rank(
        selected_option_ids=(option_ids[16], option_ids[17], option_ids[18]),
        eligible_option_ids=option_ids,
        metric_deltas=deltas,
    )

    assert best_rank == 1
    assert worst_rank == 969
    assert best_endpoint < worst_endpoint
