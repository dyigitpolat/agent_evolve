from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PydanticAIPortfolioSelectionPolicy,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development import (
    execute_provider_ready_stage_a,
)
from examples.development import (
    run_airfoil_v7_oracle_portfolio_stage_a_recovery as recovery,
)


def _typed_value(request: StructuredGenerationRequest[Any]) -> BaseModel:
    if request.operation == "oracle_portfolio_reflect":
        insight_type = get_args(
            request.output_type.model_fields["insights"].annotation
        )[0]
        metric_ids = insight_type.required_metric_ids
        families = insight_type.allowed_option_families
        option_ids = insight_type.allowed_option_ids
        contrast_list = insight_type.model_fields[
            "evidence_contrast_ids"
        ].annotation
        contrast_literal = get_args(contrast_list)[0]
        contrast_id = get_args(contrast_literal)[0]
        return request.output_type.model_validate(
            {
                "insights": [
                    {
                        "claim": "A fixed recovery fixture claim.",
                        "trigger": "Use only for this sealed evidence shard.",
                        "mechanism": "The fixture exercises strict typed replay.",
                        "affected_paths": ["$"],
                        "evidence_summary": "One sealed contrast supports the fixture.",
                        "evidence_contrast_ids": [contrast_id],
                        "confidence": 0.5,
                        "effect_predictions": [
                            {"metric_id": metric_id, "direction": "unchanged"}
                            for metric_id in metric_ids
                        ],
                        "recommended_option_families": [families[0]],
                        "recommended_option_ids": [option_ids[0]],
                        "action_template": "Apply the selected sealed action.",
                        "falsification_condition": "Reject if its metrics disagree.",
                    }
                ]
            },
            strict=True,
        )
    member_type = get_args(request.output_type.model_fields["members"].annotation)[0]
    option_ids = tuple(request.output_type.option_family_by_id)[:3]
    card_key = sorted(member_type.allowed_card_keys)[0]
    return request.output_type.model_validate(
        {
            "members": [
                {
                    "option_id": option_id,
                    "supporting_card_keys": [card_key],
                    "effect_predictions": [
                        {"metric_id": metric_id, "direction": "unchanged"}
                        for metric_id in member_type.required_metric_ids
                    ],
                    "design_rationale": "Deterministic provider-free fixture.",
                }
                for option_id in option_ids
            ]
        },
        strict=True,
    )


def test_manifest_binds_seven_cards_and_only_four_xhigh_live_calls(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "recovery.manifest.json"
    output = tmp_path / "recovery_test"
    record = recovery.write_manifest(
        manifest,
        run_id="recovery_test",
        output_dir=output,
    )
    verified = recovery.verify_manifest(manifest)

    assert len(verified.archive.entries) == 7
    assert len(record["source_archive"]["archived_cards"]) == 7
    assert record["missing_request_binding"]["call_id"] == recovery.MISSING_CALL_ID
    assert record["missing_request_binding"]["prompt_sha256"] == (
        "95e41428a70b5bc5c8c8268d304920c7580ec9fbc13864355134e8c97b228248"
    )
    assert record["experiment"]["authorized_live_call_ids"] == list(
        recovery.LIVE_CALL_IDS
    )
    assert record["provider_policy"]["reasoning"] == {
        "request_control": {"effort": "xhigh"},
        "hard_reasoning_token_cap": None,
        "accounting": "reasoning_tokens_included_in_output_tokens",
        "admission": "0 <= reasoning_tokens <= output_tokens",
    }
    assert record["protocol_artifact"]["sha256"] == (
        recovery.PROTOCOL_ARTIFACT_SHA256
    )
    assert not output.exists()


def test_existing_harness_replays_seven_then_delegates_call6_and_mpn() -> None:
    archive = recovery.authenticate_source_archive()
    replay_rows: list[dict[str, object]] = []
    live_ids: list[str] = []

    async def live(request: StructuredGenerationRequest[Any]) -> object:
        live_ids.append(request.call_id.value)
        return StructuredGenerationResponse(
            value=_typed_value(request),
            requested_model=recovery.MODEL,
            resolved_model=recovery.MODEL,
            resolved_provider="StreamLake",
            provider_response_id=f"fixture-{request.call_id.value}",
            finish_reason="tool_call",
            input_tokens=100,
            output_tokens=5_000,
            reasoning_tokens=4_565,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=None,
            latency_ns=1,
        )

    expected = dict(archive.bindings)
    expected[recovery.MISSING_CALL_ID] = archive.missing_binding
    hybrid = recovery.ArchivedReplayRunner(
        archive.entries,
        expected_bindings=expected,
        live_runner=live,
        replay_sink=lambda row: replay_rows.append(dict(row)),
    )

    result = asyncio.run(
        execute_provider_ready_stage_a(
            generator=PydanticAIAgenticGenerator(hybrid),
            selector=PydanticAIPortfolioSelectionPolicy(hybrid),
            id_factory=DeterministicIdFactory("airfoil_oracle_stage_a"),
        )
    )

    assert len(replay_rows) == 7
    assert set(live_ids) == set(recovery.LIVE_CALL_IDS)
    assert live_ids[0] == recovery.MISSING_CALL_ID
    assert set(hybrid.consumed) == set(recovery.ARCHIVED_CONTENT_SHA256).union(
        recovery.LIVE_CALL_IDS
    )
    assert {row["view_id"] for row in result["views"]} == {"M", "P", "N"}


def test_route_gate_accepts_reasoning_above_old_soft_budget_when_in_output() -> None:
    class Output(BaseModel):
        value: int

    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_recovery_route_test"),
        operation="route_test",
        prompt="provider-free route fixture",
        output_type=Output,
        output_tool_name="route_fixture",
        max_output_tokens=recovery.MAX_OUTPUT_TOKENS,
        temperature=0.0,
    )

    async def inner(_: StructuredGenerationRequest[Any]) -> object:
        return StructuredGenerationResponse(
            value=Output(value=1),
            requested_model=recovery.MODEL,
            resolved_model=recovery.MODEL,
            resolved_provider="StreamLake",
            provider_response_id="fixture-route",
            finish_reason="tool_call",
            input_tokens=1,
            output_tokens=5_527,
            reasoning_tokens=4_565,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=None,
            latency_ns=1,
        )

    journal: list[dict[str, object]] = []
    runner = recovery.RecoveryAuditedStructuredRunner(
        inner,
        pre_dispatch=lambda _: {"verified": True},
        journal_sink=lambda row: journal.append(dict(row)),
    )
    asyncio.run(runner(request))

    assert [row["record_type"] for row in journal] == ["request", "response"]
    assert journal[-1]["telemetry"]["reasoning_tokens"] == 4_565


def test_seven_durable_attestations_precede_credential_load(tmp_path: Path) -> None:
    manifest = tmp_path / "ordering.manifest.json"
    output = tmp_path / "recovery_ordering"
    recovery.write_manifest(
        manifest,
        run_id="recovery_ordering",
        output_dir=output,
    )
    observed: dict[str, object] = {}

    def credential_loader() -> str:
        replay_rows = [
            recovery.json.loads(line)
            for line in (output / "archived_replay_journal.jsonl")
            .read_text()
            .splitlines()
        ]
        source_rows = [
            recovery.json.loads(line)
            for line in (output / "source_verifications.jsonl")
            .read_text()
            .splitlines()
        ]
        observed["replay_rows"] = replay_rows
        observed["source_stages"] = [row["stage"] for row in source_rows]
        return "provider-free-injected-key"

    class Hybrid:
        consumed: set[str] = set()

    class Stack:
        hybrid = Hybrid()
        generator = object()
        selector = object()

        async def __aenter__(self) -> "Stack":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    async def stage_executor(**_: object) -> dict[str, object]:
        return {"schema_version": 1, "survives_stage_a_v1": False}

    recovery.execute_with_dependencies(
        manifest,
        recovery.LiveDependencies(
            credential_loader=credential_loader,
            stack_factory=lambda **_: Stack(),
            stage_executor=stage_executor,
            enforce_accounting=False,
        ),
    )

    replay_rows = observed["replay_rows"]
    assert len(replay_rows) == 7
    assert all(
        row["record_type"] == "archived_response_authenticated_precredential"
        and row["phase"] == "before_credential_load"
        for row in replay_rows
    )
    assert observed["source_stages"] == [
        "post_run_directory_creation",
        "archived_replays_authenticated_precredential",
        "pre_credential_load",
    ]
