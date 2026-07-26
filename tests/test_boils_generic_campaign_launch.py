"""Credential-free launch-contract checks for the BOiLS generic campaign."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.development import run_boils_generic_campaign as campaign
from examples.benchmarks.boils_abc.actions import config_sha256, normalize_candidate
from examples.benchmarks.boils_abc.evaluator import (
    LUT_INPUTS,
    BoilsEvaluation,
    CircuitDiagnostics,
    CircuitEvaluation,
)
from examples.development.durable_run_artifacts import (
    DurableJsonlJournal,
    source_identity,
)
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    FailureRecord,
)
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.ports.structured_generator import (
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)


class _FastBoilsEvaluator:
    """Contract-faithful deterministic evaluator for orchestration tests only."""

    def __init__(self, settings, *, observer=None) -> None:
        self.settings = settings
        self.observer = observer

    def provenance(self) -> dict[str, object]:
        return {
            "mode": "test_double",
            "abc_binary_sha256": self.settings.expected_abc_sha256,
            "circuits": [value.name for value in self.settings.circuits],
        }

    def evaluate(self, config: object) -> BoilsEvaluation:
        sequence = normalize_candidate(config)
        digest = config_sha256(config)
        lut_count = 7_700 + int(digest[:4], 16) % 350
        levels = 58 + int(digest[4:8], 16) % 12
        affinity = self.settings.affinity_sets[0]
        stdout = (
            f"i/o = 32/ 16 nd = {lut_count} edge = 1 aig = 1 lev = {levels}; "
            "Networks are equivalent."
        )
        diagnostics = CircuitDiagnostics(
            status="passed",
            returncode=0,
            elapsed_s=0.001,
            timeout_s=float(self.settings.per_circuit_timeout_s),
            equivalent=True,
            error_signatures=(),
            stdout_excerpt=stdout,
            stderr_excerpt="",
            stdout_sha256=hashlib.sha256(stdout.encode("ascii")).hexdigest(),
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            abc_program="test-double",
            argv=("test-double",),
            cpu_affinity=affinity,
        )
        circuit = CircuitEvaluation(
            circuit_name=self.settings.circuits[0].name,
            circuit_sha256=self.settings.circuits[0].expected_sha256,
            inputs=32,
            outputs=16,
            lut_count=lut_count,
            edge_count=1,
            aig_count=1,
            levels=levels,
            diagnostics=diagnostics,
        )
        result = BoilsEvaluation(
            configuration_sha256=digest,
            sequence=sequence,
            abc_binary_sha256=self.settings.expected_abc_sha256,
            lut_inputs=LUT_INPUTS,
            circuit_results=(circuit,),
            total_lut_count=lut_count,
            total_levels=levels,
            max_levels=levels,
            elapsed_s=0.001,
            affinity_queue_wait_s=0.0,
            cpu_affinity=affinity,
        )
        if self.observer is not None:
            self.observer(result)
        return result


def _fake_construction_probe() -> dict[str, object]:
    return {
        "all_gates_pass": True,
        "provider_calls": 0,
        "probe_contract": {
            "probe_id": campaign.CONSTRUCTION_PROBE_ID,
            "probe_version": campaign.CONSTRUCTION_PROBE_VERSION,
            "definition_sha256": campaign.CONSTRUCTION_PROBE_DEFINITION_SHA256,
        },
        "reflection_probe": {
            "request": {"request_identity_sha256": "7" * 64},
            "evidence": {"snapshot_sha256": "8" * 64},
        },
    }


def test_boils_live_progress_journal_projects_current_generic_contract() -> None:
    progress = StructuredStreamProgress(
        call_id="call_boils_progress_test_000001",
        provider_attempt_id="provider_attempt_" + "a" * 64,
        sequence=7,
        kind=StructuredStreamProgressKind.PART_DELTA,
        channel=StructuredStreamChannel.THINKING,
        elapsed_ns=123,
        event_content_utf8_bytes=5,
        cumulative_content_utf8_bytes=17,
        rolling_content_sha256="b" * 64,
    )

    assert campaign._progress_record(progress) == {
        "call_id": "call_boils_progress_test_000001",
        "provider_attempt_id": "provider_attempt_" + "a" * 64,
        "sequence": 7,
        "kind": "part_delta",
        "channel": "thinking",
        "elapsed_ns": 123,
        "event_content_utf8_bytes": 5,
        "cumulative_content_utf8_bytes": 17,
        "rolling_content_sha256": "b" * 64,
    }
    with pytest.raises(TypeError, match="exact StructuredStreamProgress"):
        campaign._progress_record(SimpleNamespace())


def test_boils_g5_compatibility_audit_converts_tuple_to_typed_json_list() -> None:
    rows = (
        {
            "lane_id": "reservoir_0001",
            "card_key": "card.boils.g05.r01",
            "status": "compatible",
            "support_sha256": "a" * 64,
        },
        {
            "lane_id": "reservoir_0002",
            "card_key": "card.boils.g05.r02",
            "status": "incompatible",
            "reason_sha256": "b" * 64,
        },
    )

    assert campaign._compatibility_audit_sha256(rows) == typed_json_sha256(
        freeze_json({"schema_version": 1, "rows": list(rows)})
    )
    with pytest.raises(TypeError, match="exact tuple"):
        campaign._compatibility_audit_sha256(list(rows))


def test_matched_control_health_uses_canonical_serialized_arm_values() -> None:
    assert campaign._is_exact_active_neutral_arm_pair(("m", "n"))
    assert campaign._is_exact_active_neutral_arm_pair(("n", "m"))
    assert not campaign._is_exact_active_neutral_arm_pair(("M", "N"))
    assert not campaign._is_exact_active_neutral_arm_pair(("m", "m"))


def test_boils_forecast_feedback_counts_each_metric_for_each_selected_action() -> None:
    assert campaign._expected_forecast_feedback_counts("live") == (6, 96)
    assert campaign._expected_forecast_feedback_counts("control") == (0, 0)
    with pytest.raises(ValueError, match="live or control"):
        campaign._expected_forecast_feedback_counts("foreign")


def test_boils_live_provider_manifest_is_streamlake_xhigh_max_token() -> None:
    record = campaign._provider_config().to_manifest_record()
    assert record["model_name"] == "deepseek/deepseek-v4-pro"
    assert record["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert record["reasoning"] == {"effort": "xhigh"}
    assert record["supports_forced_tool_choice"] is False
    assert record["queue"]["max_in_flight"] == 3
    assert record["queue"]["max_pending"] == 8
    assert record["queue"]["max_attempts"] == 3
    assert record["queue"]["backoff"]["kind"] == (
        "exponential_deterministic_task_keyed_full_jitter"
    )

    manifest = campaign._manifest(
        run_id="boils_prepare_test",
        mode="prepare",
        source={
            "schema_version": 1,
            "file_count": 1,
            "aggregate_sha256": "0" * 64,
            "files": [],
        },
        source_snapshot={
            "schema_version": 1,
            "file_count": 1,
            "aggregate_sha256": "0" * 64,
            "files": [],
        },
    )
    assert manifest["model"]["max_output_tokens"] == 384_000
    assert manifest["model"]["reasoning_mode"] is None
    assert manifest["queue"]["exponential_backoff"] is True
    assert manifest["protocol"] == {
        **manifest["protocol"],
        "generations": 6,
        "portfolio_generations": [1, 3, 5],
        "recombination_generations": [2, 4, 6],
        "planned_unique_evaluations": 62,
        "planned_logical_calls": 7,
        "terminal_reflection_policy": "require_future_portfolio_consumer",
        "reflection_chronology": {
            "source_generation": 2,
            "sealed_evidence_portfolio_generation": 1,
            "promotion_barrier_generation": 4,
            "first_consumer_generation": 5,
            "terminal_reflection": False,
        },
    }
    assert manifest["reflection"]["semantic_contract_version"] == 3
    assert manifest["reflection"]["insight_cohort_bounds"] == [2, 8]
    assert manifest["reflection"]["request_builder"] == {
        "builder_id": campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
        "builder_version": (
            campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION
        ),
        "definition_sha256": (
            campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
        ),
    }
    assert manifest["reflection"]["source"] == (
        "sealed_direct_single_mutation_observations"
    )
    assert manifest["reflection"]["recombination_results_exposed"] is False
    assert manifest["reflection"]["visibility"] == (
        "g2_quarantined_g4_admitted_g5_first_consumer"
    )
    assert manifest["construction_probe"] == {
        "probe_id": campaign.CONSTRUCTION_PROBE_ID,
        "probe_version": campaign.CONSTRUCTION_PROBE_VERSION,
        "definition_sha256": campaign.CONSTRUCTION_PROBE_DEFINITION_SHA256,
        "content_identity_deferred_until_preparation": True,
    }
    assert manifest["calibrated_selection"]["g5_memory_treatment"] == {
        **manifest["calibrated_selection"]["g5_memory_treatment"],
        "proposed_supported_member_bounds": [1, 1],
        "evaluated_supported_member_bounds": [1, 1],
        "minimum_unattributed_proposed_members": 7,
        "minimum_unattributed_evaluated_members": 7,
        "maximum_cards_per_member": 1,
        "require_every_assigned_card": True,
    }
    matched = manifest["calibrated_selection"]["g5_memory_treatment"][
        "randomized_active_neutral_pair"
    ]
    assert matched == {
        **matched,
        "policy_id": campaign.PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
        "policy_version": campaign.PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
        "definition_sha256": (
            campaign.PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
        ),
        "one_source_bound_card": True,
        "two_stable_lane_units": True,
        "same_parent_matched": False,
        "full_candidate_pool_matched": False,
        "single_block_card_effect_identified": False,
        "online_score_update_allowed": False,
    }
    assert manifest["durable_journals"] == {
        "preparation": "preparation.jsonl",
        "request": "request_evidence.jsonl",
        "output": "output_evidence.jsonl",
        "outcome": "queue_outcomes.jsonl",
        "outbound": "outbound_requests.jsonl",
        "progress": "stream_progress.jsonl",
        "wave_preparation": "wave_preparations.jsonl",
        "campaign": "campaign_events.jsonl",
        "engine": "engine_events.jsonl",
    }


def test_execution_exit_code_fails_closed_on_unhealthy_summary() -> None:
    assert campaign._execution_exit_code("completed_healthy") == 0
    assert campaign._execution_exit_code("completed_unhealthy") == 2
    with pytest.raises(RuntimeError, match="unknown terminal status"):
        campaign._execution_exit_code("prepared")


def test_provider_response_gate_requires_exact_streamlake_reasoning() -> None:
    response = {
        "requested_model": campaign.MODEL,
        "resolved_model": campaign.MODEL,
        "resolved_provider": campaign.RESOLVED_PROVIDER,
        "finish_reason": campaign.MODEL_EXECUTION_PROFILE.accepted_finish_reasons[0],
        "reasoning_tokens": 1,
        "provider_response_id": "response_1",
    }
    rows = tuple(
        {
            "authenticated_record": {
                "status": "succeeded",
                "response": {**response, "provider_response_id": f"response_{index}"},
            }
        }
        for index in range(campaign.PLANNED_LOGICAL_CALLS)
    )
    assert campaign._provider_response_telemetry_gate(
        arm="live", outcome_rows=rows
    )
    drifted = list(rows)
    drifted[0] = {
        "authenticated_record": {
            "status": "succeeded",
            "response": {**response, "reasoning_tokens": 0},
        }
    }
    assert not campaign._provider_response_telemetry_gate(
        arm="live", outcome_rows=tuple(drifted)
    )
    assert campaign._provider_response_telemetry_gate(
        arm="control", outcome_rows=()
    )


def test_preregistration_exactly_joins_prepared_source(tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        prepared=SimpleNamespace(
            protocol=SimpleNamespace(protocol_sha256="1" * 64),
            preparation_sha256="2" * 64,
        ),
        experiment_profile=None,
    )
    expected = campaign._preregistration_contract(
        bundle=bundle,
        source_aggregate_sha256="3" * 64,
        construction_probe=_fake_construction_probe(),
    )
    assert expected["reflection_insight_cohort_bounds"] == [2, 8]
    assert expected["identifiable_request_builder"]["definition_sha256"] == (
        campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
    )
    assert expected["construction_probe"]["probe_id"] == (
        campaign.CONSTRUCTION_PROBE_ID
    )
    assert len(expected["construction_probe"]["probe_sha256"]) == 64
    assert expected["construction_probe"][
        "reflection_request_identity_sha256"
    ] == "7" * 64
    assert expected["diagnostic_memory_assignment_policy"][
        "single_block_card_effect_identified"
    ] is False
    assert expected["diagnostic_memory_assignment_policy"][
        "online_score_update_allowed"
    ] is False
    path = campaign.WORKSPACE_ROOT / "boils_preregistration_test.json"
    try:
        path.write_text(json.dumps(expected), encoding="utf-8")
        receipt = campaign._validate_preregistration(
            path=path,
            bundle=bundle,
            source_aggregate_sha256="3" * 64,
            construction_probe=_fake_construction_probe(),
        )
        assert receipt["validated_contract"] == expected
        drifted = {**expected, "max_output_tokens": 2048}
        path.write_text(json.dumps(drifted), encoding="utf-8")
        try:
            campaign._validate_preregistration(
                path=path,
                bundle=bundle,
                source_aggregate_sha256="3" * 64,
                construction_probe=_fake_construction_probe(),
            )
        except RuntimeError as error:
            assert "differs" in str(error)
        else:  # pragma: no cover - fail-closed assertion.
            raise AssertionError("drifted preregistration was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_prepare_control_flow_never_reaches_credential_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "runs"
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(
        campaign,
        "_source_paths",
        lambda: (Path(campaign.__file__).resolve(),),
    )
    fake_source = {
        "schema_version": 1,
        "file_count": 1,
        "aggregate_sha256": "4" * 64,
        "files": [],
    }
    monkeypatch.setattr(
        campaign, "source_identity", lambda *args, **kwargs: fake_source
    )
    monkeypatch.setattr(
        campaign,
        "_snapshot_sources",
        lambda run_dir, paths: {
            "schema_version": 1,
            "snapshot_directory": "source_snapshot",
            "file_count": 1,
            "aggregate_sha256": "4" * 64,
            "files": [],
        },
    )

    class _Prepared:
        preparation_sha256 = "5" * 64
        protocol = SimpleNamespace(protocol_sha256="6" * 64)

        @staticmethod
        def to_record():
            return {"prepared": True}

    fake_bundle = SimpleNamespace(
        prepared=_Prepared(),
        evaluator_observer=SimpleNamespace(calls=0),
        evaluator=SimpleNamespace(provenance=lambda: {"verified": True}),
        experiment_profile=None,
    )
    monkeypatch.setattr(campaign, "_prepare_bundle", lambda **kwargs: fake_bundle)
    monkeypatch.setattr(
        campaign,
        "_all_wave_probe",
        lambda bundle: _fake_construction_probe(),
    )
    monkeypatch.setattr(
        campaign,
        "_read_live_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("credential read")),
    )
    args = argparse.Namespace(
        mode="prepare", run_id="credential_free_prepare", prereg=None
    )
    assert asyncio.run(campaign._main_async(args)) == 0
    summary = json.loads(
        (root / args.run_id / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["credential_read"] is False
    assert summary["provider_calls"] == 0
    assert summary["abc_executions"] == 0


def test_control_all_wave_probe_is_provider_and_evaluator_free(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "control_probe"
    run_dir.mkdir()
    preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
    evaluator_journal = DurableJsonlJournal(
        run_dir / "real_evaluator_observations.jsonl"
    )
    source = source_identity(
        campaign._source_paths(), relative_to=campaign.WORKSPACE_ROOT
    )
    try:
        bundle = campaign._prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator_journal,
            source_closure_sha256=str(source["aggregate_sha256"]),
            arm="control",
            evaluator_factory=_FastBoilsEvaluator,
        )
        probe = campaign._all_wave_probe(bundle)
    finally:
        preparation.close()
        evaluator_journal.close()

    assert probe["all_gates_pass"] is True
    assert probe["provider_calls"] == 0
    assert probe["abc_executions"] == 0
    assert probe["credential_read"] is False
    assert len(probe["rows"]) == 8
    bootstrap_rows = [
        row for row in probe["rows"] if row["probe_kind"] == "bootstrap_wave"
    ]
    dose_rows = [
        row
        for row in probe["rows"]
        if row["probe_kind"] == "synthetic_g5_bounded_dose"
    ]
    assert [row["generation"] for row in bootstrap_rows] == [1, 1, 3, 3, 5, 5]
    assert len(dose_rows) == 2
    assert all(row["generation"] == 5 for row in dose_rows)
    assert all(
        row["selection_contract"] == "outcome_blind_random_k8"
        and row["policy_output_width"] == 8
        and row["evaluation_width"] == 8
        for row in bootstrap_rows
    )
    assert all(
        row["selection_contract"]
        == "synthetic_bounded_dose_construction_only"
        and row["policy_output_width"] == 8
        and row["evaluation_width"] == 8
        and row["memory_dose_contract"]["proposed_supported_member_bounds"]
        == [1, 1]
        and row["memory_dose_contract"]["evaluated_supported_member_bounds"]
        == [1, 1]
        and row["memory_dose_contract"][
            "minimum_unattributed_proposed_members"
        ]
        == 7
        and row["memory_dose_contract"][
            "minimum_unattributed_evaluated_members"
        ]
        == 7
        for row in dose_rows
    )
    reflection_probe = probe["reflection_probe"]
    assert reflection_probe["all_gates_pass"] is True
    assert reflection_probe["request"]["min_insights"] == 2
    assert reflection_probe["request"]["max_insights"] == 4
    assert reflection_probe["request"]["request_builder"] == {
        "builder_id": campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
        "builder_version": (
            campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION
        ),
        "definition_sha256": (
            campaign.IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
        ),
    }
    assert reflection_probe["evidence"][
        "sealed_cutoff_event_index_inclusive"
    ] == 1
    assert len(reflection_probe["evidence"]["contrasts"]) == 4
    assert all(reflection_probe["gates"].values())
    assert bundle.evaluator_observer.calls == 0


def test_candidate_outcome_health_accepts_typed_infeasibility_only(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "candidate_outcomes"
    run_dir.mkdir()
    preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
    evaluator_journal = DurableJsonlJournal(
        run_dir / "real_evaluator_observations.jsonl"
    )
    source = source_identity(
        campaign._source_paths(), relative_to=campaign.WORKSPACE_ROOT
    )
    try:
        bundle = campaign._prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator_journal,
            source_closure_sha256=str(source["aggregate_sha256"]),
            arm="control",
            evaluator_factory=_FastBoilsEvaluator,
        )
        scored = campaign._construction_parent(
            ordinal=1,
            configuration=bundle.prepared.seeds.seeds[0].configuration,
            bundle=bundle,
        )
    finally:
        preparation.close()
        evaluator_journal.close()

    def failed_candidate(category: FailureCategory, code: FailureCode):
        failure = FailureRecord(
            category=category,
            code=code,
            message=f"typed {category.value} outcome",
        )
        assert scored.detailed_evaluation is not None
        payload = replace(
            scored.detailed_evaluation.payload,
            failure=failure,
            objectives=(),
            violations=(),
            checks=(),
            receipt=None,
        )
        detailed = replace(scored.detailed_evaluation, payload=payload)
        return replace(
            scored,
            objectives=(),
            valid=False,
            failure_message=failure.message,
            detailed_evaluation=detailed,
            objective_resolution_receipt=None,
        )

    infeasible = failed_candidate(
        FailureCategory.CANDIDATE,
        FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
    )
    accounting = campaign._candidate_outcome_accounting([scored, infeasible])
    assert accounting == {
        "evaluated_count": 2,
        "scored_count": 1,
        "typed_candidate_infeasible_count": 1,
        "runtime_failure_count": 0,
        "runtime_failures": [],
    }

    infrastructure = failed_candidate(
        FailureCategory.INFRASTRUCTURE,
        FailureCode.PROCESS_START_FAILURE,
    )
    failed = campaign._candidate_outcome_accounting([scored, infrastructure])
    assert failed["runtime_failure_count"] == 1
    assert failed["typed_candidate_infeasible_count"] == 0
    assert failed["runtime_failures"][0]["failure_category"] == "infrastructure"


def test_matched_control_uses_full_generic_g6_runtime_without_provider(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "control"
    run_dir.mkdir()
    preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
    evaluator_journal = DurableJsonlJournal(
        run_dir / "real_evaluator_observations.jsonl"
    )
    source = source_identity(
        campaign._source_paths(), relative_to=campaign.WORKSPACE_ROOT
    )
    bundle = campaign._prepare_bundle(
        run_dir=run_dir,
        preparation_journal=preparation,
        evaluator_journal=evaluator_journal,
        source_closure_sha256=str(source["aggregate_sha256"]),
        arm="control",
        evaluator_factory=_FastBoilsEvaluator,
    )
    journals = campaign._open_execution_journals(run_dir)
    try:
        summary = asyncio.run(
            campaign._execute(
                bundle=bundle,
                run_dir=run_dir,
                journals=journals,
                expected_source_sha256=str(source["aggregate_sha256"]),
            )
        )
    finally:
        for journal in journals.values():
            journal.close()
        preparation.close()
        evaluator_journal.close()

    assert summary["status"] == "completed_healthy"
    assert all(summary["health"].values())
    assert summary["provider_calls"] == 0
    assert summary["provider_response_telemetry"] == []
    assert summary["stage_candidate_counts"] == [16, 4, 16, 4, 16, 4]
    accounting = summary["evaluation_accounting"]
    assert accounting["planned_candidate_occurrences"] == 62
    assert accounting["candidate_occurrences"] == 62
    assert accounting["unique_evaluations"] == summary["evaluator_observation_count"]
    assert (
        accounting["candidate_occurrences"]
        - accounting["cache_reuse_occurrences"]
        == accounting["unique_evaluations"]
    )
    assert accounting["cache_reuse_occurrences"] <= campaign.MAX_CACHE_REUSE_OCCURRENCES
    assert summary["evaluator_receipt_count"] == 62
    assert len(summary["evaluator_artifact_ids"]) == 62
    assert all(
        type(value) is str and value.startswith("artifact_")
        for value in summary["evaluator_artifact_ids"]
    )
    # FileSystemArtifactStore receipts carry the real typed ArtifactId domain
    # object.  The launcher summary boundary must project it to canonical JSON.
    json.dumps(summary, allow_nan=False, sort_keys=True)
    assert summary["memory_trial_count"] == 0
    assert summary["administered_g5_memory_lane_count"] == 0
    assert summary["typed_infeasible_matching_lane_count"] == 0
    assert summary["g5_causal_memory_credit_claim"] is False
    assert summary["g5_memory_statuses"] == [
        "control_ignores_admitted_reflected_memory",
        "control_ignores_admitted_reflected_memory",
    ]
    assert summary["reflection_eligibility_counts"] == {
        "1": [0, 0],
        "3": [0, 0],
        "5": [1, 1],
    }
    assert len(summary["reflection_records"]) == 1
    reflection = summary["reflection_records"][0]
    assert reflection["source_generation"] == 2
    assert reflection["origin_cutoff_event_index"] == 1
    assert reflection["identifiable_contrast_count"] == 16
    assert reflection["insight_count"] == 2
    assert reflection["recombination_results_exposed"] is False
    assert len(summary["evidence_registry"]["observation_sha256s"]) == 48
    assert summary["evidence_registry"]["captured_through_event_index"] == 5
    assert summary["forecast_feedback"] == {
        "receipt_count": 0,
        "observation_count": 0,
    }
    assert summary["candidate_outcome_accounting"] == {
        "evaluated_count": 62,
        "scored_count": 62,
        "typed_candidate_infeasible_count": 0,
        "runtime_failure_count": 0,
        "runtime_failures": [],
    }
