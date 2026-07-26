from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
from decimal import Decimal
from pathlib import Path

import pytest

from agent_evolve.domain.patch import ArrayIndex
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AtomicMutationDraft,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem
from examples.benchmarks.boils_abc.budgeted_v5_analysis import (
    BoilsV5RunAnalysisInput,
)
from examples.development import run_boils_budgeted_optimizer_v5 as runner


class _DetailedResult:
    def __init__(self, objectives: dict[str, float]) -> None:
        self.objective_values = objectives


class _OfflineEvaluator:
    """Additive deterministic BOiLS-shaped evaluator with no subprocess I/O."""

    _effects = {
        (7, "dsdb"): (-10, 0),
        (7, "resub"): (-19, 0),
        (1, "balance"): (5, -1),
        (1, "fraig"): (-2, -2),
        (12, "refactor_z"): (-4, 0),
        (18, "blut"): (-3, 0),
    }

    def __init__(self, *, fail_multi_edit: bool = False) -> None:
        self._lock = threading.Lock()
        self.calls: list[dict[str, object]] = []
        self.fail_multi_edit = fail_multi_edit

    def evaluate(self, configuration: object) -> _DetailedResult:
        assert type(configuration) is dict
        sequence = configuration["sequence"]
        assert type(sequence) is list
        lut, levels = 7_944, 69
        edit_count = 0
        for index, (parent, child) in enumerate(
            zip(runner.v5.PARENT_C_SEQUENCE, sequence, strict=True)
        ):
            if parent == child:
                continue
            edit_count += 1
            delta = self._effects.get((index, child))
            if delta is None:
                # A palette choice not selected by this fixture remains a valid,
                # deliberately mediocre branch rather than an external failure.
                delta = (7, 1)
            lut += delta[0]
            levels += delta[1]
        with self._lock:
            self.calls.append({"sequence": list(sequence)})
        if self.fail_multi_edit and edit_count > 1:
            raise ValueError("injected evaluator rejection for a composed candidate")
        return _DetailedResult(
            {"total_lut_count": float(lut), "total_levels": float(levels)}
        )


def _telemetry(ordinal: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=runner.MODEL,
        resolved_model=runner.MODEL,
        resolved_provider="StreamLake",
        provider_response_id=f"offline-{ordinal}",
        finish_reason="stop",
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _OfflineGenerator:
    def __init__(
        self,
        *,
        fail_ordinal: int | None = None,
        delays: tuple[float, ...] = (),
    ) -> None:
        self.fail_ordinal = fail_ordinal
        self.calls = 0
        self.reflections = 0
        self._by_path: dict[int, int] = {}
        self.delays = delays

    async def propose(self, request: VariationGenerationRequest):
        self.calls += 1
        ordinal = self.calls
        if ordinal == self.fail_ordinal:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.OUTPUT_INVALID,
                retryable=False,
                safe_message="injected terminal model-output failure",
            )
        contract = request.atomic_mutation_contract
        assert contract is not None
        segment = contract.editable_path.segments[-1]
        assert type(segment) is ArrayIndex
        path_index = segment.value
        path_ordinal = self._by_path.get(path_index, 0)
        self._by_path[path_index] = path_ordinal + 1
        preferred = {
            (7, 0): "dsdb",
            (7, 1): "resub",
            (1, 0): "balance",
            (1, 1): "fraig",
            (12, 0): "refactor_z",
        }[(path_index, path_ordinal)]
        assert preferred in contract.replacement_options
        await asyncio.sleep(self.delays[ordinal - 1] if self.delays else 0)
        return VariationGenerationResult(
            draft=AtomicMutationDraft(
                path=contract.editable_path,
                replacement=preferred,
                design_rationale="Offline exact-palette fixture.",
            ),
            telemetry=_telemetry(ordinal),
        )

    async def reflect(self, request: ReflectionGenerationRequest):
        del request
        self.reflections += 1
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry(99))


def _durable_binding(tmp_path: Path, readiness: dict[str, object]):
    readiness_path = tmp_path / "readiness.json"
    launch_path = tmp_path / "launch.json"
    readiness_hash = runner.durable_write_json(readiness_path, readiness)
    launch_hash = runner.durable_write_json(
        launch_path,
        {"readiness_sha256": readiness["readiness_sha256"]},
    )
    return runner.DurableManifestBinding(
        readiness_path,
        readiness_hash,
        launch_path,
        launch_hash,
    )


def _offline_run(
    tmp_path: Path,
    *,
    fail_ordinal: int | None = None,
    delays: tuple[float, ...] = (),
    evaluator: _OfflineEvaluator | None = None,
):
    readiness = runner.prepare_readiness_manifest()
    binding = _durable_binding(tmp_path, readiness)
    evaluator = _OfflineEvaluator() if evaluator is None else evaluator
    problem = BoilsAbcProblem(runner._settings(), evaluator=evaluator)
    generator = _OfflineGenerator(fail_ordinal=fail_ordinal, delays=delays)
    event_path = tmp_path / "events.jsonl"
    writer = runner.DurableJsonlWriter(event_path)
    try:
        summary = asyncio.run(
            runner.run_workflow(
                problem=problem,
                generator=generator,
                evaluator_provenance_sha256_value="a" * 64,
                readiness=readiness,
                manifests=binding,
                event_writer=writer,
            )
        )
    finally:
        writer.close()
    events = tuple(
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    )
    return evaluator, generator, summary, events


def test_readiness_freezes_five_prompts_and_counterbalanced_cards_without_io() -> None:
    first = runner.prepare_readiness_manifest()
    second = runner.prepare_readiness_manifest()

    assert first == second
    assert first["schema_version"] == runner.READINESS_SCHEMA_VERSION == 2
    assert (
        runner._record_sha256("runner-domain-probe", {"schema_version": 2})
        == "11c9ccf155e2f9c50a07ef343914c77193909caf17d06121c764431576ba4610"
    )
    assert first["no_external_calls_or_evaluations"] is True
    assert len(first["prepared_model_calls"]) == 5
    assert [row["label"] for row in first["prepared_model_calls"]] == [
        "G1-A1",
        "G1-A2",
        "G1-D1",
        "G1-D2",
        "G1-U",
    ]
    assignments = {
        row["label"]: [item["card_id"] for item in row["selected_insights"]]
        for row in first["prepared_model_calls"]
    }
    assert assignments == {
        "G1-A1": ["boils_v5.area.path7.real.v1"],
        "G1-A2": ["boils_v5.area.path7.placebo.v1"],
        "G1-D1": ["boils_v5.depth.path1.placebo.v1"],
        "G1-D2": ["boils_v5.depth.path1.transfer_real.v1"],
        "G1-U": [],
    }
    assert [row["prompt_sha256"] for row in first["prepared_model_calls"]] == [
        "1c20a7b467fa08384276cdbe5e9575f1abafbb568a844d695203f67a44a09290",
        "9915e378cc668997ac21a65409593c9808d7393c478d28a43cb6c37c70c02532",
        "2fe835b0ac29b05abee7eab221261d26122b97278a16ce930dab17c6dd97b026",
        "9b859d4309770be5e874e4b25624592aef7480774fc94b77385555f6af1f37e1",
        "b2fd3926c880c6a7e5de388fcad46740164a69d3dbcdf934cacf345ae69b5f70",
    ]
    assert [row["candidate_id"] for row in first["prepared_model_calls"]] == [
        f"candidate_{runner.ID_NAMESPACE}_{ordinal:06d}" for ordinal in range(3, 8)
    ]
    assert [row["proposal_sequence"] for row in first["prepared_model_calls"]] == [
        2,
        3,
        4,
        5,
        6,
    ]
    assert len({row["candidate_id"] for row in first["prepared_model_calls"]}) == 5
    assert first["engine_cpus"] == list(runner.ENGINE_CPUS)
    assert first["cpu_admission_cpus"] == list(runner.CPU_ADMISSION_CPUS)
    assert first["budget"] == {
        "max_unique_evaluations": 9,
        "max_logical_llm_calls": 5,
        "max_generations": 2,
    }
    assert first["attempt4_pricing_envelope"] == (
        runner.attempt4_pricing_envelope_record()
    )
    assert first["attempt4_mechanism_contract"] == (
        runner.attempt4_mechanism_contract_record()
    )
    assert first["schema_repair_policy"] == runner.schema_repair_policy_record()
    assert first["schema_repair_policy"]["durable_queue_outcome_schema_version"] == 4
    assert first["schema_repair_policy"]["explicit_factory_injection"] is True
    assert len(first["schema_repair_policy"]["template_sha256"]) == 64
    assert len(first["schema_repair_policy"]["policy_sha256"]) == 64
    assert runner.MAX_OUTPUT_TOKENS == 960
    assert runner.MAX_REASONING_TOKENS == 960
    assert first["cpu_topology"]["smt_sibling_pairs"] == [
        list(pair) for pair in runner.CPU_SIBLING_PAIRS
    ]
    assert first["post_hoc_development_protocol_correction"] is True
    assert first["execution_contract"] == runner.execution_contract_record()
    assert first["cpu_sampling_policy"] == runner.cpu_sampling_policy_record()
    assert first["protocol_correction"] == runner.v5.protocol_correction_record()
    assert first["generation_one_decision"]["decision_sha256"] == (
        "a0e4262f501c94df308a7f19f29d1746e8da626ad8a16a85ee00762aa1cd1d44"
    )
    assert first["provider_options"] == {"only": ["streamlake"]}
    assert first["allowed_resolved_providers"] == ["StreamLake"]
    assert "provider_order" not in first
    assert "allow_fallbacks" not in first
    policy = runner.telemetry_policy()
    assert policy.requested_model == "deepseek/deepseek-v4-pro"
    assert policy.allowed_resolved_models == (
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-20260423",
    )
    assert policy.allowed_resolved_providers == ("StreamLake",)
    assert policy.max_cost_usd == Decimal("0.010")


def test_attempt4_binds_new_envelope_to_immutable_streamlake_snapshots() -> None:
    assert runner.RUN_ID == "boils_budgeted_optimizer_v5_attempt4_20260714"
    assert runner._file_sha256(runner.PRICING_SNAPSHOT_PATH) == (
        runner.EXPECTED_PRICING_SNAPSHOT_SHA256
    )
    assert runner._file_sha256(runner.CAPABILITY_SNAPSHOT_PATH) == (
        runner.EXPECTED_CAPABILITY_SNAPSHOT_SHA256
    )
    pricing = json.loads(runner.PRICING_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    capability = json.loads(runner.CAPABILITY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    runner._validate_attempt3_pricing_snapshot(pricing)
    runner._validate_attempt3_capability_snapshot(capability)

    assert pricing["attempt3_gate_derivation"]["provider_routing"] == {
        "eligible_provider_count": 1,
        "only": ["streamlake"],
    }
    assert capability["attempt3_relevance"]["provider_only"] == ["streamlake"]
    assert runner.attempt4_pricing_envelope_record() == {
        "schema_version": 1,
        "envelope_id": "boils_v5_attempt4_streamlake_token_caps_v1",
        "source_pricing_snapshot_sha256": (runner.EXPECTED_PRICING_SNAPSHOT_SHA256),
        "max_logical_calls": 5,
        "max_input_tokens_per_call": 10_000,
        "max_output_tokens_per_call": 960,
        "max_reasoning_tokens_per_call": 960,
        "prompt_usd_per_token": "0.0000007134",
        "completion_usd_per_token": "0.0000014268",
        "cache_read_usd_per_token": "0.00000005945",
        "reasoning_accounting": (
            "reasoning cap charged once more at the completion-token rate"
        ),
        "derived_max_cost_usd_per_call": "0.009873456",
        "frozen_cost_ceiling_usd_per_call": "0.010",
        "derived_max_accepted_run_cost_usd": "0.049367280",
        "frozen_accepted_run_ceiling_usd": "0.050",
        "envelope_sha256": runner.attempt4_pricing_envelope_record()["envelope_sha256"],
    }


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("selected_endpoint", "provider_name", "unexpected-provider"),
        ("selected_endpoint", "provider_request_slug", "unexpected-provider"),
        ("provider_registry", "slug", "unexpected-provider"),
        ("attempt3_relevance", "provider_only", ["unexpected-provider"]),
        ("attempt3_relevance", "allowed_resolved_models", [runner.MODEL]),
        ("attempt3_relevance", "allowed_resolved_providers", ["unexpected"]),
        ("attempt3_relevance", "required_capabilities_present", []),
    ],
)
def test_attempt3_capability_validator_rejects_route_drift(
    section: str,
    field: str,
    replacement: object,
) -> None:
    capability = json.loads(runner.CAPABILITY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    capability[section][field] = replacement

    with pytest.raises(RuntimeError, match="capability snapshot semantics changed"):
        runner._validate_attempt3_capability_snapshot(capability)


def _replay_readiness_preparations(
    verifier: runner.ReadinessTraceVerifier,
    readiness: dict[str, object],
) -> None:
    for row in readiness["prepared_model_calls"]:
        verifier.observe(
            {
                "event_type": "invocation_prepared",
                "proposal_authority": "model",
                **row,
            }
        )


def test_readiness_verifier_requires_exact_decision_and_reserved_sequence() -> None:
    readiness = runner.prepare_readiness_manifest()

    missing_decision = runner.ReadinessTraceVerifier(readiness)
    _replay_readiness_preparations(missing_decision, readiness)
    with pytest.raises(RuntimeError, match="before readiness replay closed"):
        missing_decision.assert_ready()

    altered_decision = runner.ReadinessTraceVerifier(readiness)
    bad_decision = copy.deepcopy(readiness["generation_one_decision"])
    bad_decision["palette_seed"] += 1
    with pytest.raises(RuntimeError, match="decision differs"):
        altered_decision.observe(bad_decision)

    duplicate_decision = runner.ReadinessTraceVerifier(readiness)
    duplicate_decision.observe(readiness["generation_one_decision"])
    with pytest.raises(RuntimeError, match="duplicated"):
        duplicate_decision.observe(readiness["generation_one_decision"])

    altered_sequence = copy.deepcopy(readiness)
    altered_sequence["prepared_model_calls"][0]["proposal_sequence"] += 1
    verifier = runner.ReadinessTraceVerifier(readiness)
    verifier.observe(readiness["generation_one_decision"])
    with pytest.raises(RuntimeError, match="live preparation differs"):
        _replay_readiness_preparations(verifier, altered_sequence)

    exact = runner.ReadinessTraceVerifier(readiness)
    exact.observe(readiness["generation_one_decision"])
    _replay_readiness_preparations(exact, readiness)
    exact.assert_ready()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 1),
        ("run_id", "boils_budgeted_optimizer_v5_attempt3_20260714"),
    ],
)
def test_readiness_rejects_coherently_rehashed_stale_identity(
    field: str, replacement: object
) -> None:
    readiness = runner.prepare_readiness_manifest()
    readiness[field] = replacement
    body = {key: value for key, value in readiness.items() if key != "readiness_sha256"}
    readiness["readiness_sha256"] = runner._record_sha256("readiness", body)

    with pytest.raises(RuntimeError, match="readiness manifest identity changed"):
        runner.validate_readiness_manifest(readiness)


def test_readiness_rejects_corrupt_runner_v2_hash() -> None:
    readiness = runner.prepare_readiness_manifest()
    readiness["readiness_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="readiness manifest identity changed"):
        runner.ReadinessTraceVerifier(readiness)


def test_offline_two_generation_run_closes_exact_budgets_and_engine_only_g2(
    tmp_path: Path,
) -> None:
    evaluator, generator, summary, events = _offline_run(tmp_path)

    assert summary["protocol_acceptance_passed"] is True
    assert summary["execution_contract"]["wall_clock_claims_allowed"] is False
    assert summary["cpu_sampling_policy"]["mode"] == (
        "record_only_shared_host_load_observation"
    )
    assert generator.calls == 5
    assert generator.reflections == 0
    assert len(evaluator.calls) == 9
    assert summary["resources"]["logical_llm_calls"] == 5
    assert summary["resources"]["unique_physical_evaluations"] == 9
    assert summary["resources"]["evaluation_cache"]["misses"] == 9
    assert summary["resources"]["evaluation_cache"]["in_flight"] == 0
    assert summary["attempt4_mechanism_contract"] == (
        runner.attempt4_mechanism_contract_record()
    )
    selection = summary["planner"]["generation2"]["selection"]
    path_by_candidate = {
        row["candidate_id"]: row["path"] for row in selection["branch_paths"]
    }
    exploit_paths = {
        path_by_candidate[candidate_id]
        for candidate_id in selection["exploit_pair_ids"]
    }
    coverage_paths = {
        path_by_candidate[candidate_id]
        for candidate_id in selection["coverage_pair_ids"]
    }
    assert coverage_paths - exploit_paths
    BoilsV5RunAnalysisInput.from_record(summary["offline_analysis_input"])
    assert summary["offline_analysis_input"]["palette_spec"]["uncertainty"] == {
        "index": 12,
        "replacements": ["sopb", "dsdb", "refactor_z"],
    }
    assert [
        row["proposal_authority"] for row in summary["generations"][1]["slots"]
    ] == [
        "engine",
        "engine",
    ]
    assert (
        summary["resources"]["accepted_terminal_response_cost_ceiling_usd"] == "0.050"
    )
    assert (
        summary["resources"]["potentially_billable_attempt_cost_envelope_usd"]
        == "0.100"
    )

    planned = [
        event
        for event in events
        if event["event_type"] == "optimizer_generation_planned"
    ]
    completed_calls = [
        event for event in events if event["event_type"] == "llm_call_completed"
    ]
    decisions = [
        event
        for event in events
        if event["event_type"].startswith("boils_v5_generation")
    ]
    assert len(planned) == 2
    assert len(completed_calls) == 5
    assert len(decisions) == 2
    first_call_index = events.index(completed_calls[0])
    prepared_model = [
        event
        for event in events[:first_call_index]
        if event["event_type"] == "invocation_prepared"
        and event["proposal_authority"] == "model"
    ]
    assert len(prepared_model) == 5
    decision_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "boils_v5_generation1_decided"
    )
    assert decision_index < min(events.index(event) for event in prepared_model)


def test_opposite_provider_delays_preserve_occurrences_g2_archive_and_result(
    tmp_path: Path,
) -> None:
    forward_dir = tmp_path / "forward"
    reverse_dir = tmp_path / "reverse"
    forward_dir.mkdir()
    reverse_dir.mkdir()
    _, _, forward, _ = _offline_run(
        forward_dir,
        delays=(0.05, 0.04, 0.03, 0.02, 0.01),
    )
    _, _, reverse, _ = _offline_run(
        reverse_dir,
        delays=(0.01, 0.02, 0.03, 0.04, 0.05),
    )

    def g1_occurrences(summary: dict[str, object]):
        slots = summary["generations"][0]["slots"]
        return [
            (
                row["slot_id"],
                row["candidate"]["candidate_id"],
                row["candidate"]["proposal_sequence"],
            )
            for row in slots
        ]

    assert g1_occurrences(forward) == g1_occurrences(reverse)
    assert (
        forward["planner"]["generation2"]["decision_sha256"]
        == (reverse["planner"]["generation2"]["decision_sha256"])
    )
    assert (
        forward["generations"][1]["plan_sha256"]
        == (reverse["generations"][1]["plan_sha256"])
    )
    assert (
        forward["generations"][1]["receipt_sha256"]
        == (reverse["generations"][1]["receipt_sha256"])
    )
    assert (
        forward["final_archive_snapshot_sha256"]
        == (reverse["final_archive_snapshot_sha256"])
    )
    assert forward["result_sha256"] == reverse["result_sha256"]


def test_invalid_g2_evaluation_is_protocol_rejected(tmp_path: Path) -> None:
    evaluator = _OfflineEvaluator(fail_multi_edit=True)
    _, _, summary, _ = _offline_run(tmp_path, evaluator=evaluator)

    assert summary["protocol_acceptance_passed"] is False
    assert summary["status"] == "protocol_rejected"
    assert summary["gates"]["generation_two_non_skipped_valid_compliant"] is False
    assert summary["gates"]["no_missing_or_rejected_slot"] is False
    assert summary["offline_analysis_input"] is None


def test_noncompliant_g2_candidate_is_protocol_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner.AgenticEvolutionEngine._operator_compliance

    def inject(self, prepared, draft, occurrence, frozen):
        compliant, failure, hashes, preservation = original(
            self, prepared, draft, occurrence, frozen
        )
        if prepared.plan.generation == 2:
            return False, "injected compliance rejection", hashes, preservation
        return compliant, failure, hashes, preservation

    monkeypatch.setattr(
        runner.AgenticEvolutionEngine,
        "_operator_compliance",
        inject,
    )
    _, _, summary, _ = _offline_run(tmp_path)

    assert summary["protocol_acceptance_passed"] is False
    assert summary["status"] == "protocol_rejected"
    assert summary["gates"]["generation_one_six_valid_compliant"] is True
    assert summary["gates"]["generation_two_non_skipped_valid_compliant"] is False
    assert summary["gates"]["no_missing_or_rejected_slot"] is False
    assert summary["offline_analysis_input"] is None


def test_one_missing_model_call_continues_g2_without_replacement(
    tmp_path: Path,
) -> None:
    readiness = runner.prepare_readiness_manifest()
    binding = _durable_binding(tmp_path, readiness)
    evaluator = _OfflineEvaluator()
    generator = _OfflineGenerator(fail_ordinal=3)
    problem = BoilsAbcProblem(runner._settings(), evaluator=evaluator)
    writer = runner.DurableJsonlWriter(tmp_path / "events.jsonl")
    try:
        summary = asyncio.run(
            runner.run_workflow(
                problem=problem,
                generator=generator,
                evaluator_provenance_sha256_value="b" * 64,
                readiness=readiness,
                manifests=binding,
                event_writer=writer,
            )
        )
    finally:
        writer.close()

    assert generator.calls == 5
    assert generator.reflections == 0
    assert len(evaluator.calls) == 8  # seed + four model + X + two G2 unions
    assert summary["status"] == "protocol_rejected"
    assert summary["protocol_acceptance_passed"] is False
    assert summary["offline_analysis_input"] is None
    assert summary["failed_slot_continuation"] == {
        "missing_g1_slot_ids": ["G1-D1"],
        "substitution_allowed": False,
        "g2_checkpoint_closed": True,
        "protocol_acceptance_requires_no_missing_slot": True,
    }
    checkpoint = summary["planner"]["generation2"]["failed_slot_continuation"][
        "g1_checkpoint"
    ]
    assert [row["status"] for row in checkpoint] == [
        "eligible",
        "eligible",
        "missing_candidate",
        "eligible",
        "eligible",
        "eligible",
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["event_type"] == "llm_call_failed" for row in rows) == 1
    assert any(
        row["event_type"] == "optimizer_generation_planned" and row["generation"] == 2
        for row in rows
    )


def test_manifest_identity_failure_precedes_seed_and_provider(tmp_path: Path) -> None:
    readiness = runner.prepare_readiness_manifest()
    binding = _durable_binding(tmp_path, readiness)
    binding.launch_path.write_text("{}\n", encoding="utf-8")
    evaluator = _OfflineEvaluator()
    generator = _OfflineGenerator()
    problem = BoilsAbcProblem(runner._settings(), evaluator=evaluator)
    writer = runner.DurableJsonlWriter(tmp_path / "events.jsonl")
    try:
        with pytest.raises(RuntimeError, match="manifest identity"):
            asyncio.run(
                runner.run_workflow(
                    problem=problem,
                    generator=generator,
                    evaluator_provenance_sha256_value="c" * 64,
                    readiness=readiness,
                    manifests=binding,
                    event_writer=writer,
                )
            )
    finally:
        writer.close()
    assert evaluator.calls == []
    assert generator.calls == 0


def test_source_bundle_is_deterministic_complete_and_replay_verified(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.tar"
    second_path = tmp_path / "second.tar"
    first = runner.durable_source_bundle(first_path)
    second = runner.durable_source_bundle(second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["entry_count"] >= 90
    paths = [row["path"] for row in first["entries"]]
    assert paths == sorted(paths)
    assert "src/agent_evolve/application/agentic_evolution.py" in paths
    assert "examples/benchmarks/boils_abc/budgeted_v5_planner.py" in paths
    assert "examples/development/run_boils_budgeted_optimizer_v5.py" in paths
    assert "pyproject.toml" in paths
    assert "uv.lock" in paths
    runner.verify_source_bundle(first_path, first)

    for field, replacement in (
        ("schema_version", 2),
        ("metadata_normalization", {}),
        ("record_sha256", "0" * 64),
    ):
        changed_record = copy.deepcopy(first)
        changed_record[field] = replacement
        with pytest.raises(RuntimeError, match="source bundle record changed"):
            runner.verify_source_bundle(first_path, changed_record)

    changed_member_path = tmp_path / "changed-member.tar"
    payload = bytearray(first_path.read_bytes())
    payload[512] ^= 1
    changed_member_path.write_bytes(payload)
    coherent_outer_record = copy.deepcopy(first)
    coherent_outer_record["bundle_sha256"] = hashlib.sha256(payload).hexdigest()
    body = {
        key: value
        for key, value in coherent_outer_record.items()
        if key != "record_sha256"
    }
    coherent_outer_record["record_sha256"] = runner._record_sha256(
        "source-bundle", body
    )
    with pytest.raises(RuntimeError, match="source bundle/live source mismatch"):
        runner.verify_source_bundle(changed_member_path, coherent_outer_record)


def test_durable_mkdir_fsyncs_parent_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = runner.os.fsync

    def observe(fd: int) -> None:
        calls.append(fd)
        original(fd)

    monkeypatch.setattr(runner.os, "fsync", observe)
    destination = tmp_path / "durable"
    runner.durable_mkdir(destination)
    assert destination.is_dir()
    assert len(calls) == 1

    def fail(fd: int) -> None:
        del fd
        raise OSError("injected parent directory fsync failure")

    monkeypatch.setattr(runner.os, "fsync", fail)
    with pytest.raises(OSError, match="parent directory fsync failure"):
        runner.durable_mkdir(tmp_path / "not-durable")


def _synthetic_topology() -> dict[str, object]:
    sibling_by_cpu = {cpu: pair for pair in runner.CPU_SIBLING_PAIRS for cpu in pair}

    def read(path: Path) -> str:
        name = path.parts[-3]
        assert name.startswith("cpu")
        cpu = int(name[3:])
        return ",".join(str(value) for value in sibling_by_cpu[cpu]) + "\n"

    return runner.cpu_topology_record(reader=read)


def _synthetic_source_closure(marker: str = "a") -> dict[str, object]:
    body = {
        "schema_version": 1,
        "entry_count": 1,
        "entries": [
            {
                "path": "fixture.py",
                "bytes": 1,
                "sha256": marker * 64,
            }
        ],
    }
    return {
        **body,
        "closure_sha256": runner._record_sha256("source-closure", body),
    }


def _quality_readiness(
    topology: dict[str, object], closure: dict[str, object]
) -> dict[str, object]:
    body = {
        "schema_version": runner.READINESS_SCHEMA_VERSION,
        "run_id": runner.RUN_ID,
        "cpu_topology": topology,
        "source_closure": closure,
        "execution_contract": runner.execution_contract_record(),
        "cpu_sampling_policy": runner.cpu_sampling_policy_record(),
        "frozen": True,
    }
    return {
        **body,
        "readiness_sha256": runner._record_sha256("readiness", body),
    }


def _synthetic_admission(topology: dict[str, object]) -> dict[str, object]:
    samples: list[str] = []
    for window in range(runner.CPU_ADMISSION_WINDOWS + 1):
        samples.append(
            "\n".join(
                f"cpu{cpu} 0 0 0 {100 + 100 * window} 0 0 0 0"
                for cpu in runner.CPU_ADMISSION_CPUS
            )
        )
    iterator = iter(samples)
    return runner.sample_cpu_admission(
        reader=lambda: next(iterator),
        sleeper=lambda _: None,
        topology=topology,
    )


def test_cpu_admission_records_three_quality_only_windows() -> None:
    samples = []
    for window in range(4):
        rows = []
        for cpu in runner.CPU_ADMISSION_CPUS:
            values = [100 * window, 0, 0, 100, 0, 0, 0, 0]
            rows.append(f"cpu{cpu} " + " ".join(str(value) for value in values))
        samples.append("\n".join(rows))
    iterator = iter(samples)

    record = runner.sample_cpu_admission(
        reader=lambda: next(iterator),
        sleeper=lambda _: None,
        topology=_synthetic_topology(),
    )

    assert record["passed"] is True
    assert record["window_count"] == 3
    assert record["max_busy_fraction"] == 1.0
    assert record["execution_contract"]["wall_clock_claims_allowed"] is False
    assert all(
        row["busy_fraction"] == 1.0 and row["passed"] is True
        for window in record["windows"]
        for row in window["cpus"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evidence_class", "timing_eligible"),
        ("timing_comparison_claim_authorized", True),
        ("wall_clock_claim_authorized", True),
        ("wall_clock_dominance_claim_authorized", True),
    ],
)
def test_quality_only_execution_contract_is_exact(
    field: str, replacement: object
) -> None:
    contract = runner.execution_contract_record()
    contract[field] = replacement

    with pytest.raises(RuntimeError, match="execution claim boundary changed"):
        runner.validate_execution_contract(contract)


def test_cpu_admission_rejects_coherently_rehashed_policy_drift() -> None:
    topology = _synthetic_topology()
    record = _synthetic_admission(topology)
    record["max_busy_fraction"] = 0.1
    body = {key: value for key, value in record.items() if key != "admission_sha256"}
    record["admission_sha256"] = runner._record_sha256("cpu-admission", body)

    with pytest.raises(RuntimeError, match="CPU admission record changed"):
        runner.validate_cpu_admission_record(record, topology=topology)


def test_cpu_admission_rejects_zero_delta_counters() -> None:
    rows = "\n".join(f"cpu{cpu} 0 0 0 100 0 0 0 0" for cpu in runner.CPU_ADMISSION_CPUS)

    with pytest.raises(RuntimeError, match="failed counter-integrity admission"):
        runner.sample_cpu_admission(
            reader=lambda: rows,
            sleeper=lambda _: None,
            topology=_synthetic_topology(),
        )


def test_pre_directory_admission_failure_leaves_run_directory_absent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "attempt"
    topology = _synthetic_topology()
    closure = _synthetic_source_closure()
    readiness = _quality_readiness(topology, closure)
    calls: list[str] = []

    def build_readiness() -> dict[str, object]:
        calls.append("readiness")
        return readiness

    def build_topology() -> dict[str, object]:
        calls.append("topology")
        return topology

    def reject_admission(*, topology: dict[str, object]) -> dict[str, object]:
        calls.append("admission")
        assert topology is readiness["cpu_topology"]
        raise RuntimeError("injected transient CPU contention")

    with pytest.raises(RuntimeError, match="transient CPU contention"):
        runner.prepare_pre_directory_admission(
            run_dir,
            readiness_builder=build_readiness,
            topology_builder=build_topology,
            admission_sampler=reject_admission,
            source_closure_builder=lambda: closure,
        )

    assert calls == ["readiness", "topology", "admission"]
    assert not run_dir.exists()


def test_pre_directory_admission_success_returns_records_without_creating_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "attempt"
    topology = _synthetic_topology()
    closure = _synthetic_source_closure()
    readiness = _quality_readiness(topology, closure)
    admission = _synthetic_admission(topology)

    def sample(*, topology: dict[str, object]) -> dict[str, object]:
        assert topology is readiness["cpu_topology"]
        assert not run_dir.exists()
        return admission

    actual = runner.prepare_pre_directory_admission(
        run_dir,
        readiness_builder=lambda: readiness,
        topology_builder=lambda: topology,
        admission_sampler=sample,
        source_closure_builder=lambda: closure,
    )

    assert actual == (readiness, topology, admission)
    assert actual[0] is readiness
    assert actual[1] is topology
    assert actual[2] is not admission
    assert not run_dir.exists()


def test_source_closure_change_during_admission_leaves_run_directory_absent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "attempt"
    topology = _synthetic_topology()
    frozen_closure = _synthetic_source_closure("a")
    changed_closure = _synthetic_source_closure("b")
    readiness = _quality_readiness(topology, frozen_closure)
    admission = _synthetic_admission(topology)
    admission_calls = 0

    def sample(*, topology: dict[str, object]) -> dict[str, object]:
        nonlocal admission_calls
        admission_calls += 1
        assert topology is readiness["cpu_topology"]
        return admission

    with pytest.raises(
        RuntimeError, match="source closure changed during CPU admission"
    ):
        runner.prepare_pre_directory_admission(
            run_dir,
            readiness_builder=lambda: readiness,
            topology_builder=lambda: topology,
            admission_sampler=sample,
            source_closure_builder=lambda: changed_closure,
        )

    assert admission_calls == 1
    assert not run_dir.exists()


def test_pre_directory_admission_rejects_preexisting_directory_before_builders(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    calls: list[str] = []

    def unexpected_builder() -> dict[str, object]:
        calls.append("builder")
        return {}

    def unexpected_sampler(**kwargs: object) -> dict[str, object]:
        calls.append("sampler")
        return kwargs

    with pytest.raises(FileExistsError, match="run directory already exists"):
        runner.prepare_pre_directory_admission(
            run_dir,
            readiness_builder=unexpected_builder,
            topology_builder=unexpected_builder,
            admission_sampler=unexpected_sampler,
        )

    assert calls == []
    assert run_dir.is_dir()


def test_pre_directory_admission_rejects_non_path_before_builders() -> None:
    calls: list[str] = []

    def unexpected_builder() -> dict[str, object]:
        calls.append("builder")
        return {}

    with pytest.raises(TypeError, match="run_dir must be a Path"):
        runner.prepare_pre_directory_admission(
            "not-a-path",  # type: ignore[arg-type]
            readiness_builder=unexpected_builder,
            topology_builder=unexpected_builder,
        )

    assert calls == []


def test_main_finalizes_owned_directory_after_directory_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / runner.RUN_ID

    class _Args:
        log_root = tmp_path

    class _Parser:
        @staticmethod
        def parse_args() -> _Args:
            return _Args()

    def fail_after_creation(path: Path) -> None:
        assert path == run_dir
        path.mkdir()
        raise runner.DurableDirectoryPublishError("injected dirent fsync failure")

    monkeypatch.setattr(runner, "_parser", _Parser)
    monkeypatch.setattr(
        runner,
        "prepare_pre_directory_admission",
        lambda _: ({}, {}, {}),
    )
    monkeypatch.setattr(runner, "durable_mkdir", fail_after_creation)

    with pytest.raises(
        runner.DurableDirectoryPublishError,
        match="injected dirent fsync failure",
    ):
        runner.main()

    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    finalized = json.loads((run_dir / "finalized.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure_type"] == "DurableDirectoryPublishError"
    assert finalized["status"] == "failed"
    assert set(finalized["files"]) == {"failure.json"}


def test_main_file_exists_collision_does_not_mutate_preexisting_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / runner.RUN_ID
    run_dir.mkdir()
    sentinel = run_dir / "owner.txt"
    sentinel.write_text("preexisting owner\n", encoding="utf-8")

    class _Args:
        log_root = tmp_path

    class _Parser:
        @staticmethod
        def parse_args() -> _Args:
            return _Args()

    monkeypatch.setattr(runner, "_parser", _Parser)
    monkeypatch.setattr(
        runner,
        "prepare_pre_directory_admission",
        lambda _: ({}, {}, {}),
    )

    with pytest.raises(FileExistsError):
        runner.main()

    assert tuple(run_dir.iterdir()) == (sentinel,)
    assert sentinel.read_text(encoding="utf-8") == "preexisting owner\n"


def test_live_queue_composition_is_exact_and_requires_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class _StructuredFactory:
        @staticmethod
        def openrouter(**kwargs):
            calls["openrouter"] = kwargs
            return object()

    class _QueuedRunner:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            del args

    def queue_factory(**kwargs):
        calls["queue"] = kwargs
        return _QueuedRunner()

    async def workflow(**kwargs):
        calls["workflow"] = kwargs
        return {"ok": True}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-secret")
    monkeypatch.setattr(runner, "PydanticAIStructuredGenerator", _StructuredFactory)
    monkeypatch.setattr(runner, "create_production_queued_runner", queue_factory)
    monkeypatch.setattr(runner, "PydanticAIAgenticGenerator", lambda value: value)
    monkeypatch.setattr(runner, "run_workflow", workflow)
    event_writer = runner.DurableJsonlWriter(tmp_path / "events.jsonl")
    queue_writer = runner.DurableJsonlWriter(tmp_path / "queue.jsonl")
    try:
        result = asyncio.run(
            runner._run_live(
                problem=object(),
                provenance_sha256="d" * 64,
                readiness={},
                manifests=object(),
                event_writer=event_writer,
                queue_writer=queue_writer,
            )
        )
    finally:
        event_writer.close()
        queue_writer.close()

    assert result == {"ok": True}
    assert calls["openrouter"]["model_name"] == runner.MODEL
    assert calls["openrouter"]["max_connections"] == 5
    assert calls["openrouter"]["provider_options"] == {
        "only": ["streamlake"],
    }
    queue = calls["queue"]
    assert queue["max_in_flight"] == 5
    assert queue["max_pending"] == 10
    assert queue["max_attempts"] == 2
    assert queue["attempt_timeout_ns"] == 60_000_000_000
    assert type(queue["attempt_request_policy"]) is runner.SchemaRepairAttemptPolicy
    assert (
        queue["attempt_request_policy"].manifest is runner.SCHEMA_REPAIR_POLICY_MANIFEST
    )
    assert (
        queue["outcome_publication_policy"] is runner.OutcomePublicationPolicy.REQUIRED
    )
    assert queue["jitter_policy"] == runner.DeterministicHashJitter(
        seed=runner.JITTER_SEED,
        domain=runner.JITTER_DOMAIN,
    )
