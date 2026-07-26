"""Provider/CFD-free checks for the Airfoil-v7 G2 release barrier."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent_evolve.agentic import TreatmentComplianceViolation

from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    DIAGNOSTIC_SHAPE_OPTION_ID,
    DIAGNOSTIC_TRIM_OPTION_ID,
    MAX_OUTPUT_TOKENS,
    MEMORY_CARD_BEGIN,
    MEMORY_CARD_END,
    NEUTRAL_PARENT,
    OfflineAirfoilV7Generator,
    SHAM_OPTION_ID,
    compose_offline_experiment,
)
from examples.benchmarks.engibench_airfoil.v7_launch import (
    AirfoilV7G2PrequeueGatePolicy,
    DeferredJournaledLiveGenerator,
    G2PrequeueGateError,
    _airfoil_v7_telemetry_policy,
    _held_out_transfer_adjudication,
    _sha256_record,
)


@dataclass(frozen=True)
class _CapturedAirfoilBatch:
    g1_requests: tuple
    reflection_requests: tuple
    g2_requests: tuple
    assignment_commitment: object
    result: object
    memory_entries: tuple


def _execute_offline_fixture(fixture):
    async def execute_with_heartbeat():
        execution = asyncio.create_task(
            fixture.composition.optimizer.run(
                (NEUTRAL_PARENT, fixture.held_out_parent.candidate)
            )
        )
        while not execution.done():
            await asyncio.sleep(0.01)
        return await execution

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(execute_with_heartbeat())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture(scope="module")
def captured_batch() -> _CapturedAirfoilBatch:
    # The tiny yield also proves the planner's wave concurrency without slowing
    # the provider-free fixture materially.
    fixture = compose_offline_experiment(delay_seconds=0.001)
    result = _execute_offline_fixture(fixture)
    assert result.final_state.logical_llm_calls == 7
    g1 = tuple(
        request
        for request in fixture.generator.requests
        if request.finite_variation_contract is not None
        and request.finite_variation_contract.catalog_id
        in {"airfoil_v7_shape", "airfoil_v7_trim"}
    )
    g2 = tuple(
        request
        for request in fixture.generator.requests
        if request.finite_variation_contract is not None
        and request.finite_variation_contract.catalog_id == "airfoil_v7_union"
    )
    assert len(g1) == 2
    target_by_catalog = {
        "airfoil_v7_shape": DIAGNOSTIC_SHAPE_OPTION_ID,
        "airfoil_v7_trim": DIAGNOSTIC_TRIM_OPTION_ID,
    }
    for request in g1:
        contract = request.finite_variation_contract
        assert contract is not None
        target = target_by_catalog[contract.catalog_id]
        assert "PROSPECTIVELY TARGETED DIAGNOSTIC ACTION" in request.prompt
        assert f"Select exact option_id {target}." in request.prompt
    assert len(fixture.generator.reflection_requests) == 2
    assert len(g2) == 3
    assert {
        request.max_output_tokens for request in (*g1, *g2)
    } == {MAX_OUTPUT_TOKENS} == {384_000}
    for reflection in fixture.generator.reflection_requests:
        assert reflection.max_output_tokens == MAX_OUTPUT_TOKENS
        assert reflection.min_insights == reflection.max_insights == 1
        assert len(reflection.available_contrast_ids) == 1
        assert sorted(
            reflection.prompt.count(option_id)
            for option_id in (
                DIAGNOSTIC_SHAPE_OPTION_ID,
                DIAGNOSTIC_TRIM_OPTION_ID,
            )
        ) == [1, 2]
    assert fixture.planner.held_out_assignment_commitment is not None
    return _CapturedAirfoilBatch(
        g1_requests=g1,
        reflection_requests=tuple(fixture.generator.reflection_requests),
        g2_requests=g2,
        assignment_commitment=fixture.planner.held_out_assignment_commitment,
        result=result,
        memory_entries=fixture.composition.memory.entries,
    )


class _JournalDouble:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def write(self, value) -> None:
        self.records.append(dict(value))


class _RunnerDouble:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.exited += 1


def _replace_card(prompt: str, transform) -> str:
    start = prompt.index(MEMORY_CARD_BEGIN) + len(MEMORY_CARD_BEGIN)
    end = prompt.index(MEMORY_CARD_END)
    payload = json.loads(prompt[start:end].strip())
    transform(payload)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{prompt[:start]}\n{encoded}\n{prompt[end:]}"


def _live_double(captured: _CapturedAirfoilBatch, *, gate_policy=None):
    journal = _JournalDouble()
    delegate = OfflineAirfoilV7Generator(delay_seconds=0.0)
    runner = _RunnerDouble()
    telemetry_policy = _airfoil_v7_telemetry_policy()
    counters = {"credentials": 0, "stack": 0, "verifications": 0}

    def credential_loader() -> str:
        counters["credentials"] += 1
        return "test-key-never-sent"

    def stack_factory(api_key: str):
        assert api_key == "test-key-never-sent"
        counters["stack"] += 1
        return SimpleNamespace(
            runner=runner,
            generator=delegate,
            telemetry_policy=telemetry_policy,
        )

    def verifier(stage: str):
        counters["verifications"] += 1
        return {"stage": stage, "verified": True}

    live = DeferredJournaledLiveGenerator(
        credential_loader=credential_loader,
        stack_factory=stack_factory,
        pre_provider_verifier=verifier,
        journal=journal,  # type: ignore[arg-type]
        expected_telemetry_policy=telemetry_policy.to_trace_record(),
        expected_telemetry_policy_sha256=telemetry_policy.policy_sha256,
        g2_gate_policy=gate_policy,
    )
    live.bind_assignment_commitment_supplier(
        lambda: captured.assignment_commitment  # type: ignore[return-value]
    )
    return live, delegate, runner, journal, counters


def test_full_offline_adjudicator_binds_actual_asn_assignments_to_gate(
    captured_batch: _CapturedAirfoilBatch,
) -> None:
    policy = AirfoilV7G2PrequeueGatePolicy()
    gate = policy.validate_batch(
        tuple(policy.prepare(request) for request in captured_batch.g2_requests),
        assignment_commitment=captured_batch.assignment_commitment,
    )
    adjudication = _held_out_transfer_adjudication(
        captured_batch.result,  # type: ignore[arg-type]
        memory_entries=captured_batch.memory_entries,
        g2_gate_record=gate,
    )
    assert adjudication["schema_version"] == 2
    assert adjudication["policy_version"] == 4
    assert adjudication["exact_action_gate_pass"] is True
    assert adjudication["recommended_option_ids"] == [
        adjudication["selected_option_id"]
    ]
    assert all(
        arm["exact_action_gate_pass"] is True
        for arm in adjudication["arms"].values()
    )
    conjuncts = adjudication["artifact_91_promotion_conjuncts"]
    assert adjudication["g2_gate_integrity_pass"] is True
    assert conjuncts["2_registered_as_mapping_only_and_distinct_cards"] is True
    assert adjudication["candidate_assignment_bindings_pass"] is True
    assert adjudication["neutral_sham_gate_pass"] is True

    swapped = json.loads(json.dumps(gate))
    chosen = swapped["assignment_commitment"]["chosen_references"]
    chosen["adaptive"], chosen["score_swapped"] = (
        chosen["score_swapped"],
        chosen["adaptive"],
    )
    unsigned = dict(swapped)
    unsigned.pop("batch_gate_sha256")
    swapped["batch_gate_sha256"] = _sha256_record(
        unsigned,
        domain=b"agent-evolve:airfoil-v7-g2-prequeue-gate:v2\x00",
    )
    rejected = _held_out_transfer_adjudication(
        captured_batch.result,  # type: ignore[arg-type]
        memory_entries=captured_batch.memory_entries,
        g2_gate_record=swapped,
    )
    rejected_conjuncts = rejected["artifact_91_promotion_conjuncts"]
    assert (
        rejected_conjuncts["2_registered_as_mapping_only_and_distinct_cards"]
        is False
    )
    assert rejected["pre_finalization_promotion_eligible"] is False


def test_same_family_wrong_option_is_rejected_before_evaluation_and_not_adjudicated(
) -> None:
    fixture = compose_offline_experiment(delay_seconds=0.0)

    def same_family_wrong_shape_option(prompt: str) -> str:
        start = prompt.index(MEMORY_CARD_BEGIN) + len(MEMORY_CARD_BEGIN)
        end = prompt.index(MEMORY_CARD_END)
        payload = json.loads(prompt[start:end].strip())
        (exact_option_id,) = payload["recommended_option_ids"]
        if exact_option_id == DIAGNOSTIC_SHAPE_OPTION_ID:
            return "shape.camber_aft.p0030"
        return exact_option_id

    fixture.generator._union_option = same_family_wrong_shape_option
    result = _execute_offline_fixture(fixture)
    generation_two = result.generation_receipts[1]
    adaptive = next(
        item.outcome for item in generation_two.slot_results if item.slot.slot_id == "A"
    )

    assert result.final_state.logical_llm_calls == 7
    assert result.final_state.unique_evaluations == 6
    assert fixture.evaluator.calls == 6
    assert adaptive.failure_stage == "treatment_noncompliance"
    assert adaptive.candidate is None
    assert adaptive.terminal_evaluation is None
    admission = adaptive.treatment_admission_receipt
    assert admission is not None and not admission.passed
    assert admission.evaluator_entered is False
    assert admission.selected_action.family == "shape_only"
    assert admission.selected_action.option_id == "shape.camber_aft.p0030"
    assert admission.violations == (
        TreatmentComplianceViolation.SELECTED_ACTION_INCOMPATIBLE,
    )
    preflight = adaptive.prepared.treatment_preflight_receipt
    assert preflight is not None
    assert tuple(action.option_id for action in preflight.compatible_actions) == (
        DIAGNOSTIC_SHAPE_OPTION_ID,
    )
    assert not any(
        event.get("event_type") == "candidate_evaluated"
        and event.get("operator_invocation_id")
        == adaptive.prepared.operator_invocation_id.value
        for event in fixture.engine_events
    )

    g2_requests = tuple(
        request
        for request in fixture.generator.requests
        if request.finite_variation_contract is not None
        and request.finite_variation_contract.catalog_id == "airfoil_v7_union"
    )
    gate_policy = AirfoilV7G2PrequeueGatePolicy()
    gate = gate_policy.validate_batch(
        tuple(gate_policy.prepare(request) for request in g2_requests),
        assignment_commitment=fixture.planner.held_out_assignment_commitment,
    )
    adjudication = _held_out_transfer_adjudication(
        result,
        memory_entries=fixture.composition.memory.entries,
        g2_gate_record=gate,
    )
    assert adjudication["status"] == "not_tested_noncompliance"
    assert adjudication["scientific_verdict"] == "not_tested_noncompliance"
    assert adjudication["metric_adjudication_gate_pass"] is False
    assert adjudication["metric_adjudications"] == []
    assert adjudication["exact_action_gate_pass"] is False
    assert adjudication["recommended_option_ids"] == [
        DIAGNOSTIC_SHAPE_OPTION_ID
    ]
    assert adjudication["treatment_noncompliance_slots"] == ["A"]
    assert adjudication["development_decision"] == "do_not_advance"


class _GatePolicyProtocolDouble:
    """Non-Airfoil concrete type proving policy injection is structural."""

    def __init__(self) -> None:
        self.delegate = AirfoilV7G2PrequeueGatePolicy()
        self.policy_id = self.delegate.policy_id
        self.policy_version = self.delegate.policy_version
        self.prepare_calls = 0
        self.validate_calls = 0

    @property
    def identity_sha256(self):
        return self.delegate.identity_sha256

    def prepare(self, request):
        self.prepare_calls += 1
        return self.delegate.prepare(request)

    def validate_batch(self, envelopes, *, assignment_commitment):
        self.validate_calls += 1
        return self.delegate.validate_batch(
            envelopes,
            assignment_commitment=assignment_commitment,
        )


async def _run_prefix(live, captured: _CapturedAirfoilBatch) -> None:
    await asyncio.gather(*(live.propose(item) for item in captured.g1_requests))
    await asyncio.gather(
        *(live.reflect(item) for item in captured.reflection_requests)
    )


def test_realistic_reflected_cards_pass_versioned_hashed_policy(
    captured_batch: _CapturedAirfoilBatch,
) -> None:
    policy = AirfoilV7G2PrequeueGatePolicy()
    envelopes = tuple(policy.prepare(item) for item in captured_batch.g2_requests)
    record = policy.validate_batch(
        envelopes,
        assignment_commitment=captured_batch.assignment_commitment,  # type: ignore[arg-type]
    )

    assert record["schema_version"] == 2
    assert record["status"] == "accepted"
    assert record["policy_id"] == "airfoil_v7_blinded_g2_prequeue_gate"
    assert record["policy_version"] == 2
    assert record["policy_identity_sha256"] == policy.identity_sha256
    assert record["assignment_sha256"] == (
        captured_batch.assignment_commitment.assignment_sha256
    )
    assert len(record["batch_gate_sha256"]) == 64
    rows = record["requests"]
    assert len(rows) == 3
    assert {row["assignment_role"] for row in rows} == {
        "adaptive",
        "score_swapped",
        "sham",
    }
    assert len({row["raw_prompt_sha256"] for row in rows}) == 3
    assert len({row["masked_prompt_sha256"] for row in rows}) == 1
    assert len({row["payload_sha256"] for row in rows}) == 3
    assert {row["recommended_option_id"] for row in rows} == {
        DIAGNOSTIC_SHAPE_OPTION_ID,
        DIAGNOSTIC_TRIM_OPTION_ID,
        SHAM_OPTION_ID,
    }
    assert all(row["raw_prompt_utf8_bytes"] > 0 for row in rows)
    assert all(row["masked_prompt_utf8_bytes"] > 0 for row in rows)
    assert all(row["payload_utf8_bytes"] > 0 for row in rows)
    sham_id = captured_batch.assignment_commitment.sham_reference.insight_id.value
    sham_payload = next(
        envelope.payload
        for envelope in envelopes
        if envelope.payload["insight_id"] == sham_id
    )
    assert sham_payload == {
        "affected_paths": [
            "$.alpha_deg",
            "$.lower_coefficients",
            "$.upper_coefficients",
        ],
        "action_template": (
            "A legal option is represented by its named family and sealed option "
            "identifier."
        ),
        "claim": (
            "The supplied palette contains legal coordinated interventions in "
            "each listed option family."
        ),
        "effect_predictions": [
            {
                "direction": "unknown",
                "metric_id": "objective:normalized_multipoint_drag",
            },
            {
                "direction": "unknown",
                "metric_id": "violation:normalized_lift_equality",
            },
        ],
        "falsification_condition": (
            "The two named held-out metric values are the complete empirical "
            "check for the intervention."
        ),
        "insight_id": sham_id,
        "mechanism": (
            "Each finite option specifies an internally consistent coordinated "
            "change within its named family."
        ),
        "recommended_option_families": ["trim_only"],
        "recommended_option_ids": [SHAM_OPTION_ID],
        "trigger": "The frozen parent admits the listed finite action families.",
    }
    visible_text = " ".join(
        value for value in sham_payload.values() if type(value) is str
    ).casefold()
    assert all(
        token not in visible_text
        for token in ("select ", "choose ", "explore ", "no action", "control", "sham")
    )


def test_scientific_evidence_and_reflection_words_are_not_treatment_labels(
    captured_batch: _CapturedAirfoilBatch,
) -> None:
    requests = tuple(
        replace(
            request,
            prompt=_replace_card(
                request.prompt,
                lambda card: card.__setitem__(
                    "mechanism",
                    f"{card['mechanism']} Scientific evidence motivates reflection.",
                ),
            ),
        )
        for request in captured_batch.g2_requests
    )
    policy = AirfoilV7G2PrequeueGatePolicy()
    record = policy.validate_batch(
        tuple(policy.prepare(item) for item in requests),
        assignment_commitment=captured_batch.assignment_commitment,  # type: ignore[arg-type]
    )
    assert record["status"] == "accepted"


def test_gate_receipt_is_durable_before_any_g2_delegate_release(
    captured_batch: _CapturedAirfoilBatch,
) -> None:
    live, delegate, runner, journal, counters = _live_double(captured_batch)

    async def scenario() -> None:
        async with live:
            await _run_prefix(live, captured_batch)
            baseline = delegate.propose_calls
            first = asyncio.create_task(live.propose(captured_batch.g2_requests[0]))
            second = asyncio.create_task(live.propose(captured_batch.g2_requests[1]))
            await asyncio.sleep(0)
            assert baseline == delegate.propose_calls == 2
            assert not first.done()
            assert not second.done()
            await asyncio.gather(
                first,
                second,
                live.propose(captured_batch.g2_requests[2]),
            )

    asyncio.run(scenario())

    assert delegate.propose_calls == 5
    assert delegate.reflect_calls == 2
    assert counters["credentials"] == 1
    assert counters["stack"] == 1
    assert runner.entered == runner.exited == 1
    gate_index = next(
        index
        for index, record in enumerate(journal.records)
        if record.get("record_type") == "g2_prequeue_batch_gate"
    )
    g2_request_indices = [
        index
        for index, record in enumerate(journal.records)
        if record.get("record_type") == "request"
        and record.get("logical_call_ordinal") in {5, 6, 7}
    ]
    assert journal.records[gate_index]["status"] == "accepted"
    assert len(g2_request_indices) == 3
    assert gate_index < min(g2_request_indices)
    accepted = live.accepted_g2_gate_record
    assert accepted == journal.records[gate_index]
    assert accepted is not None
    accepted["status"] = "mutated-test-copy"
    assert live.accepted_g2_gate_record["status"] == "accepted"


def test_generator_accepts_structural_batch_policy_promotion_seam(
    captured_batch: _CapturedAirfoilBatch,
) -> None:
    policy = _GatePolicyProtocolDouble()
    live, delegate, _, _, _ = _live_double(
        captured_batch,
        gate_policy=policy,
    )

    async def scenario() -> None:
        async with live:
            await _run_prefix(live, captured_batch)
            await asyncio.gather(
                *(live.propose(item) for item in captured_batch.g2_requests)
            )

    asyncio.run(scenario())
    assert policy.prepare_calls == 3
    assert policy.validate_calls == 1
    assert delegate.propose_calls == 5


def _reserved_arm_request(request):
    return replace(
        request,
        prompt=_replace_card(
            request.prompt,
            lambda card: card.__setitem__(
                "claim", f"{card['claim']} This is the adaptive arm."
            ),
        ),
    )


def _extra_schema_key_request(request):
    return replace(
        request,
        prompt=_replace_card(
            request.prompt,
            lambda card: card.__setitem__("origin", "hidden"),
        ),
    )


def _outside_contract_option_request(request):
    return replace(
        request,
        prompt=_replace_card(
            request.prompt,
            lambda card: card.__setitem__(
                "recommended_option_ids", ["trim.outside.contract"]
            ),
        ),
    )


def _duplicate_insight_request(request, first_request):
    start = first_request.prompt.index(MEMORY_CARD_BEGIN) + len(MEMORY_CARD_BEGIN)
    end = first_request.prompt.index(MEMORY_CARD_END)
    first_id = json.loads(first_request.prompt[start:end].strip())["insight_id"]
    return replace(
        request,
        prompt=_replace_card(
            request.prompt,
            lambda card: card.__setitem__("insight_id", first_id),
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("reserved_arm", "blinded_card_contains_reserved_term"),
        ("schema_key", "blinded_card_schema_keys_differ"),
        ("outside_option", "recommended_option_id_outside_contract"),
        ("duplicate_insight", "insight_ids_not_distinct"),
        ("masked_prompt", "masked_prompt_bytes_mismatch"),
        ("contract", "finite_contract_mismatch"),
        ("delimiter", "memory_card_delimiter_count_invalid"),
    ],
)
def test_any_invalid_arm_fails_closed_before_all_g2_delegate_calls(
    captured_batch: _CapturedAirfoilBatch,
    mutation: str,
    expected_reason: str,
) -> None:
    requests = list(captured_batch.g2_requests)
    if mutation == "reserved_arm":
        requests[2] = _reserved_arm_request(requests[2])
    elif mutation == "schema_key":
        requests[2] = _extra_schema_key_request(requests[2])
    elif mutation == "duplicate_insight":
        requests[2] = _duplicate_insight_request(requests[2], requests[0])
    elif mutation == "outside_option":
        requests[2] = _outside_contract_option_request(requests[2])
    elif mutation == "masked_prompt":
        requests[2] = replace(requests[2], prompt=requests[2].prompt + "\n")
    elif mutation == "contract":
        contract = requests[2].finite_variation_contract
        assert contract is not None
        requests[2] = replace(
            requests[2],
            finite_variation_contract=replace(
                contract,
                options=tuple(reversed(contract.options)),
            ),
        )
    elif mutation == "delimiter":
        requests[2] = replace(
            requests[2],
            prompt=requests[2].prompt.replace(
                "</MEMORY_CARD>",
                "<MEMORY_CARD></MEMORY_CARD>",
            ),
        )
    else:  # pragma: no cover - closed parametrization.
        raise AssertionError(mutation)
    live, delegate, runner, journal, counters = _live_double(captured_batch)

    async def scenario():
        async with live:
            await _run_prefix(live, captured_batch)
            baseline = delegate.propose_calls
            outcomes = await asyncio.gather(
                *(live.propose(item) for item in requests),
                return_exceptions=True,
            )
            assert baseline == delegate.propose_calls == 2
            return outcomes

    outcomes = asyncio.run(scenario())

    assert all(type(item) is G2PrequeueGateError for item in outcomes)
    assert {item.reason_code for item in outcomes} == {expected_reason}
    assert delegate.reflect_calls == 2
    assert counters["credentials"] == 1
    assert counters["stack"] == 1
    assert runner.entered == runner.exited == 1
    rejected = [
        record
        for record in journal.records
        if record.get("record_type") == "g2_prequeue_batch_gate"
    ]
    assert len(rejected) == 1
    gate_policy = AirfoilV7G2PrequeueGatePolicy()
    assert rejected[0] == {
        "schema_version": 2,
        "record_type": "g2_prequeue_batch_gate",
        "status": "rejected",
        "reason_code": expected_reason,
            "request_count": (
                2
                if mutation
                in {"reserved_arm", "schema_key", "outside_option", "delimiter"}
                else 3
            ),
        "policy_id": gate_policy.policy_id,
        "policy_version": gate_policy.policy_version,
        "policy_identity_sha256": gate_policy.identity_sha256,
        "provider_dispatch_performed": False,
    }
