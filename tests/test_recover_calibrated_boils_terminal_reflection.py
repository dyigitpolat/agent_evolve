from __future__ import annotations

from examples.development import (
    recover_calibrated_boils_terminal_reflection as recovery,
)


def test_reconstructs_failed_reflection_and_transient_controls_provider_free() -> None:
    reconstructed = recovery.reconstruct()

    assert reconstructed.engine_request.request_sha256 == recovery.SPEC.request_sha256
    assert (
        reconstructed.engine_request.prompt_sha256
        == recovery.SPEC.semantic_prompt_sha256
    )
    assert reconstructed.contract.identity_sha256 == recovery.SPEC.contract_sha256
    assert reconstructed.provider_request.call_id.value == recovery.SPEC.source_call_id
    assert (
        recovery._sha_text(
            recovery.render_reflection_prompt(reconstructed.provider_request.prompt)
        )
        == recovery.SPEC.wire_prompt_sha256
    )

    verification = reconstructed.verification
    assert verification["engine_summary_queue_join_verified"] is True
    assert verification["exact_provider_request_reconstructed"] is True
    comparison = verification["comparison_evidence"]
    assert comparison["same_route_across_source_and_heat_controls"] is True
    assert comparison["shared_failure_envelope_with_heat_v3"] is True
    assert comparison["failed_prompt_no_larger_than_predecessors"] is True
    assert [
        item["status"] for item in comparison["source_successful_predecessors"]
    ] == ["succeeded_first_attempt", "succeeded_first_attempt"]
    assert comparison["heat_exact_request_failure_then_success"] == {
        "call_id": recovery.SPEC.heat_control_call_id,
        "wire_prompt_sha256": recovery.SPEC.heat_control_wire_prompt_sha256,
        "provider_attempt_id": recovery.SPEC.heat_control_provider_attempt_id,
        "v3_status": "terminal_failure_http_400",
        "v4_status": "succeeded_first_attempt",
        "failure_envelope_sha256": recovery.SPEC.failure_envelope_sha256,
        "same_call_prompt_attempt_identity": True,
    }


def test_recovery_record_changes_only_call_identity_and_allows_one_attempt() -> None:
    reconstructed = recovery.reconstruct()
    call_id = recovery._new_call_id()
    record = recovery.recovery_request_record(reconstructed, call_id=call_id)

    source = record["source_provider_request"]
    replay = record["replay_provider_request"]
    assert source["call_id"] == recovery.SPEC.source_call_id
    assert replay["call_id"] == call_id.value
    assert call_id.value != recovery.SPEC.source_call_id
    assert {
        key: value for key, value in source.items() if key != "call_id"
    } == {key: value for key, value in replay.items() if key != "call_id"}

    config = recovery._config()
    assert config.model_name == recovery.MODEL
    assert config.provider_only == ("streamlake",)
    assert config.reasoning_config is not None
    assert config.reasoning_config.effort == "xhigh"
    assert config.max_attempts == 1
    assert config.to_manifest_record()["reasoning"] == {"effort": "xhigh"}
