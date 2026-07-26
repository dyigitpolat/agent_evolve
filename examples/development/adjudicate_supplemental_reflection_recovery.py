#!/usr/bin/env python3
"""Provider-free adjudication of the sealed BOiLS reflection recovery output."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes  # noqa: E402
from examples.development import (  # noqa: E402
    run_supplemental_reflection_recovery as recovery,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


RECOVERY_FINALIZATION_SHA256 = (
    "8ce385be78e30de169982a51fe3b3ef5bcf9e3b1502e6f7c0817bcc155b3d7ab"
)
RECOVERY_RECURSIVE_SHA256 = (
    "abeb303f9e97f9908e3eee4896d1c2b8c45fe67724ea04e63512c0811398866c"
)
ADJUDICATION_RUN = recovery.OUTPUT_RUN.with_name(
    "boilsq_repaired_v2_deepseek_g6_cycle3_reflection_"
    "recovery_adjudication_20260716"
)


def _one_row(name: str) -> dict[str, object]:
    rows = read_jsonl(recovery.OUTPUT_RUN / name)
    if len(rows) != 1:
        raise RuntimeError(f"{name} must contain exactly one row")
    return rows[0]


def _insight(value: object) -> InsightDraft:
    if type(value) is not dict:
        raise RuntimeError("typed insight is not an exact object")
    raw_predictions = value.get("effect_predictions")
    if type(raw_predictions) is not list:
        raise RuntimeError("typed insight has no prediction list")
    predictions: list[MetricEffectPrediction] = []
    for raw in raw_predictions:
        if type(raw) is not dict:
            raise RuntimeError("typed metric prediction is not an object")
        predictions.append(
            MetricEffectPrediction(
                metric_id=str(raw.get("metric_id")),
                direction=MetricEffectDirection(str(raw.get("direction"))),
            )
        )
    tuple_fields = (
        "affected_paths",
        "evidence_contrast_ids",
        "recommended_option_families",
        "recommended_option_ids",
    )
    for name in tuple_fields:
        if type(value.get(name)) is not list or not all(
            type(item) is str for item in value[name]
        ):
            raise RuntimeError(f"typed insight has malformed {name}")
    return InsightDraft(
        claim=str(value.get("claim")),
        trigger=str(value.get("trigger")),
        mechanism=str(value.get("mechanism")),
        affected_paths=tuple(value["affected_paths"]),
        evidence_summary=str(value.get("evidence_summary")),
        confidence=float(value.get("confidence")),
        evidence_contrast_ids=tuple(sorted(value["evidence_contrast_ids"])),
        effect_predictions=tuple(sorted(predictions, key=lambda item: item.metric_id)),
        recommended_option_families=tuple(
            sorted(value["recommended_option_families"])
        ),
        recommended_option_ids=tuple(sorted(value["recommended_option_ids"])),
        action_template=str(value.get("action_template")),
        falsification_condition=str(value.get("falsification_condition")),
    )


def adjudicate() -> tuple[Path, dict[str, object]]:
    finalization = verify_finalized_run_directory(recovery.OUTPUT_RUN)
    if (
        finalization.get("status") != "failed"
        or finalization.get("finalization_sha256")
        != RECOVERY_FINALIZATION_SHA256
        or finalization.get("recursive_content_sha256")
        != RECOVERY_RECURSIVE_SHA256
    ):
        raise RuntimeError("sealed recovery artifact differs from the adjudication input")
    _, contract, source_request, source_verification = recovery._load_source()

    request_record = json.loads(
        (recovery.OUTPUT_RUN / "recovery_request.json").read_bytes()
    )
    if type(request_record) is not dict:
        raise RuntimeError("recovery request is not an exact object")
    request_sha256 = request_record.pop("recovery_request_sha256", None)
    if request_sha256 != hashlib.sha256(
        recovery.REQUEST_DOMAIN + canonical_json_bytes(request_record)
    ).hexdigest():
        raise RuntimeError("recovery request hash does not authenticate its fields")
    request_record["recovery_request_sha256"] = request_sha256

    request_evidence = validate_structured_generation_request_evidence_record(
        _one_row("request_evidence.jsonl")
    )
    output_evidence = validate_structured_generation_output_evidence_record(
        _one_row("output_evidence.jsonl"),
        request_evidence=request_evidence,
    )
    outbound = validate_openrouter_outbound_request_manifest_record(
        _one_row("outbound_requests.jsonl")
    )
    outcome = _one_row("queue_outcomes.jsonl")
    progress = read_jsonl(recovery.OUTPUT_RUN / "stream_progress.jsonl")
    failure = json.loads((recovery.OUTPUT_RUN / "failed.json").read_bytes())
    lifecycle = read_jsonl(recovery.OUTPUT_RUN / "lifecycle.jsonl")

    call_id = request_record.get("call_id")
    attempts = outcome.get("attempts")
    response = outcome.get("response")
    settings = outbound.get("settings")
    if (
        type(attempts) is not list
        or len(attempts) != 1
        or type(attempts[0]) is not dict
        or type(response) is not dict
        or type(settings) is not dict
        or outcome.get("status") != "succeeded"
        or outcome.get("task_id") != call_id
        or request_evidence.get("call_id") != call_id
        or output_evidence.get("call_id") != call_id
        or outbound.get("call_id") != call_id
        or response.get("requested_model") != recovery.MODEL
        or response.get("resolved_model") != recovery.MODEL
        or response.get("resolved_provider") != recovery.RESOLVED_PROVIDER
        or not isinstance(response.get("reasoning_tokens"), int)
        or response["reasoning_tokens"] <= 0
        or settings.get("model") != recovery.MODEL
        or settings.get("provider")
        != {"only": ["streamlake"], "allow_fallbacks": False}
        or settings.get("reasoning") != {"effort": "xhigh"}
        or settings.get("max_completion_tokens") != recovery.MAX_OUTPUT_TOKENS
        or settings.get("stream") is not True
    ):
        raise RuntimeError("accepted provider output does not join its frozen route")
    forbidden = outbound.get("forbidden_fields_absent")
    if type(forbidden) is not dict or not all(value is True for value in forbidden.values()):
        raise RuntimeError("outbound request contains a forbidden behavior field")
    attempt_request = attempts[0].get("request_evidence")
    if (
        type(attempt_request) is not dict
        or attempt_request.get("provider_attempt_id")
        != outbound.get("provider_attempt_id")
        or attempt_request.get("prompt_sha256")
        != recovery.SOURCE_WIRE_PROMPT_SHA256
        or len(progress) < 1
        or progress[-1].get("kind") != "stream_completed"
        or sum(row.get("kind") == "stream_completed" for row in progress) != 1
    ):
        raise RuntimeError("stream/outcome evidence is incomplete or does not join")
    if (
        type(failure) is not dict
        or failure.get("failure_type") != "AttributeError"
        or len(lifecycle) != 2
        or lifecycle[-1].get("event") != "runner_close_failed"
        or lifecycle[-1].get("failure_type") != "AttributeError"
    ):
        raise RuntimeError("post-success harness failure differs from the diagnosis")

    typed_output = output_evidence.get("typed_output")
    if type(typed_output) is not dict or type(typed_output.get("insights")) is not list:
        raise RuntimeError("accepted output has no exact typed insight list")
    insights = tuple(_insight(value) for value in typed_output["insights"])
    if not recovery.MIN_INSIGHTS <= len(insights) <= recovery.MAX_INSIGHTS:
        raise RuntimeError("accepted insight cardinality violates the frozen request")
    for insight in insights:
        validate_reflection_insight_draft(insight, contract)
        if not set(insight.evidence_contrast_ids).issubset(
            source_request.available_contrast_ids
        ):
            raise RuntimeError("accepted insight cites a foreign contrast")

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "adjudicated_provider_success",
        "epistemic_status": "supplemental_quarantined",
        "source_run_mutated": False,
        "lifecycle_publication_count": 0,
        "provider_call_repeated": False,
        "recovery_artifact": {
            "run_id": recovery.OUTPUT_RUN.name,
            "recorded_status": "failed",
            "finalization_sha256": RECOVERY_FINALIZATION_SHA256,
            "recursive_content_sha256": RECOVERY_RECURSIVE_SHA256,
            "failure_boundary": (
                "provider generation and all required evidence publications "
                "succeeded; post-success QueueSnapshot field mismatch raised "
                "AttributeError before result publication"
            ),
            "explicit_close_authenticated": False,
            "queue_terminal_and_zero_pending_inferred": True,
            "limitation": (
                "The sealed live artifact cannot prove explicit runner/HTTP-client "
                "close because the same snapshot serializer failed before aclose."
            ),
        },
        "source_verification": source_verification,
        "call_id": call_id,
        "source_call_id": recovery.SOURCE_CALL_ID,
        "request_evidence_sha256": request_evidence["request_evidence_sha256"],
        "output_evidence_sha256": output_evidence["output_evidence_sha256"],
        "typed_output_sha256": output_evidence["typed_output_sha256"],
        "outbound_request_manifest_sha256": outbound[
            "outbound_request_manifest_sha256"
        ],
        "provider_attempt_id": outbound["provider_attempt_id"],
        "progress_event_count": len(progress),
        "terminal_progress": progress[-1],
        "telemetry": response,
        "insight_count": len(insights),
        "insights": [
            {
                "content": insight.content_record(),
                "content_sha256": insight.content_sha256,
                "hypothesis_sha256": insight.hypothesis_sha256,
                "epistemic_status": "unverified_supplemental_hypothesis",
                "lifecycle_status": "quarantined",
            }
            for insight in insights
        ],
    }

    ADJUDICATION_RUN.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        ADJUDICATION_RUN / "manifest.json",
        {
            "schema_version": 1,
            "provider_free": True,
            "source_recovery_finalization_sha256": RECOVERY_FINALIZATION_SHA256,
            "source_recovery_recursive_content_sha256": RECOVERY_RECURSIVE_SHA256,
            "source_code": source_identity(
                (Path(__file__), Path(recovery.__file__)),
                relative_to=WORKSPACE_ROOT,
            ),
        },
    )
    write_json_atomic(ADJUDICATION_RUN / "result.json", result)
    seal = finalize_run_directory(
        ADJUDICATION_RUN,
        status="adjudicated_provider_success",
    )
    return ADJUDICATION_RUN, {**result, "finalization": seal}


def main() -> int:
    output_dir, result = adjudicate()
    finalization = result["finalization"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": result["status"],
                "insight_count": result["insight_count"],
                "provider_call_repeated": result["provider_call_repeated"],
                "finalization_sha256": finalization["finalization_sha256"],
                "recursive_content_sha256": finalization[
                    "recursive_content_sha256"
                ],
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
