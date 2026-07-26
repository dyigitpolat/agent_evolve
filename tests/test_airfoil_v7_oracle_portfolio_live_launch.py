from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from examples.development import run_airfoil_v7_oracle_portfolio_stage_a as launch


def _build(tmp_path: Path, run_id: str) -> tuple[Path, Path, dict[str, object]]:
    manifest = tmp_path / f"{run_id}.manifest.json"
    output = tmp_path / run_id
    record = launch.write_manifest(
        manifest,
        run_id=run_id,
        output_dir=output,
    )
    return manifest, output, record


def test_manifest_is_provider_free_and_binds_exact_live_policy(tmp_path: Path) -> None:
    manifest, output, record = _build(tmp_path, "portfolio_manifest_test")

    policy = record["provider_policy"]
    assert isinstance(policy, dict)
    assert policy["requested_model"] == "deepseek/deepseek-v4-pro"
    assert policy["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert policy["max_output_tokens"] == 384_000
    assert policy["reasoning_max_tokens"] == 4_096
    queue = policy["queue"]
    assert isinstance(queue, dict)
    assert queue["max_in_flight"] == 8
    assert queue["max_pending"] == 16
    assert queue["max_attempts"] == 2
    assert queue["attempt_timeout_seconds"] == 180
    assert record["credentials_read"] is False
    assert record["provider_dispatch_performed"] is False
    assert record["design_id"] == (
        "airfoil_v7_oracle_portfolio_stage_a_v1r2_provider_grammar"
    )
    assert record["base_method_design_id"] == (
        "airfoil_v7_oracle_portfolio_stage_a_v1"
    )
    assert record["execution_revision_class"] == (
        "pre_treatment_provider_grammar_repair"
    )
    assert record["mechanism_revision_ordinal"] == 0
    assert record["experiment"] == {
        "logical_reflection_calls": 8,
        "logical_selector_calls": 3,
        "logical_calls": 11,
        "candidate_evaluations": 0,
        "cfd_calls": 0,
        "execution": "8 concurrent reflections then 3 concurrent selectors",
    }
    lineage = record["revision_lineage"]
    assert isinstance(lineage, dict)
    assert lineage["execution_revision_id"] == record["design_id"]
    assert lineage["mechanism_revision_ordinal"] == 0
    predecessor = lineage["predecessor"]
    assert isinstance(predecessor, dict)
    assert predecessor["run_id"] == (
        "ae7_portfolio_stage_a_v1r1_jsonpath_0715_0315"
    )
    assert predecessor["status"] == (
        "failed_before_any_inference_or_selector_call"
    )
    assert predecessor["logical_reflection_calls"] == 8
    assert predecessor["logical_selector_calls"] == 0
    assert predecessor["physical_provider_attempts"] == 8
    assert predecessor["provider_inferences"] == 0
    assert predecessor["http_status_codes"] == [400]
    assert predecessor["finalization_sha256"] == (
        "71dd532c0604c0bbc3a4dce0085de1cf6a5d69d618bca90247ac22ef94df8936"
    )
    predecessors = lineage["predecessor_runs"]
    assert predecessors["ordered_run_ids"] == [
        "ae7_portfolio_stage_a_0715_0256",
        "ae7_portfolio_stage_a_v1r1_jsonpath_0715_0315",
    ]
    assert predecessors["v1"]["physical_provider_attempts"] == 16
    assert predecessors["v1r1"] == predecessor
    change = lineage["change_commitment"]
    assert isinstance(change, dict)
    wire = change["provider_wire"]
    assert wire == {
        "wire_contract_revision": (
            "reflection_wire_jsonpath_contract_v3_provider_grammar"
        ),
        "wire_contract_revision_sha256": (
            "28563b2b0f118e49d9d245a1fe7bfef52d8ad412cb7ab778e249e5f4e9176760"
        ),
        "provider_path_pattern": r"^\$([.\[].*)?$",
        "provider_path_pattern_sha256": (
            "9774d7fbc3f23aced5abaa6b033060b2bfc9702a235924d8a9459d5cf76d8ba2"
        ),
        "output_contract_note_sha256": (
            "03b92db7cdb9b7c1f92a2508616047c450355f31eb2ff5c660063e1945b48138"
        ),
    }
    prompt_bindings = change["provider_prompt_bindings"]
    assert isinstance(prompt_bindings, list)
    assert len(prompt_bindings) == 8
    assert len({row["provider_prompt_sha256"] for row in prompt_bindings}) == 8
    calls = record["development_plan"]["reflection_calls"]
    for call, binding in zip(calls, prompt_bindings, strict=True):
        rendered = launch.render_reflection_prompt(call["prompt"])
        assert binding["call_id"] == call["call_id"]
        assert binding["high_level_prompt_sha256"] == call["prompt_sha256"]
        assert binding["provider_prompt_utf8_bytes"] == len(rendered.encode())
        assert binding["provider_prompt_sha256"] == launch.hashlib.sha256(
            rendered.encode()
        ).hexdigest()
        assert launch.REFLECTION_OUTPUT_CONTRACT_NOTE in rendered
    assert tuple(
        row["provider_prompt_sha256"] for row in prompt_bindings
    ) == launch.EXPECTED_V1R2_RENDERED_REFLECTION_PROMPT_SHA256
    compatibility = change["provider_schema_compatibility_evidence"]
    assert compatibility["revision_artifact"]["sha256"] == (
        "a9555218e45209a7cfa020a057da5762ca3f5bafb5340066b3d94e879bf40ba3"
    )
    assert compatibility["positive_probe_raw_bytes_cryptographically_bound"] is False
    assert compatibility["prelaunch_provider_acceptance_proven"] is False
    scientific = change["scientific_surface_binding"]
    assert scientific["unchanged"] is True
    assert scientific["v1_development_plan"]["sha256"] == (
        "2d52ee4189443415c2431f720abdd3efdfb4286c7cf071d880f231ebd1229f10"
    )
    assert scientific["v1r1_development_plan"]["sha256"] == (
        "dba61a7accb29088fc5ea61e78f9c5e9fcb55860de1417c4755bdd17e4716b12"
    )
    assert change["no_further_execution_revision_permitted"] is True
    accounting = lineage["cumulative_accounting"]
    assert accounting == {
        "accounting_basis": "logical_calls_separate_from_physical_attempts",
        "v1_realized": {
            "logical_calls": 8,
            "physical_provider_attempts": 16,
            "logical_selector_calls": 0,
        },
        "v1r1_realized": {
            "logical_calls": 8,
            "physical_provider_attempts": 8,
            "http_status_codes": [400],
            "provider_inferences": 0,
            "logical_selector_calls": 0,
        },
        "cumulative_before_current": {
            "logical_calls": 16,
            "physical_provider_attempts": 24,
        },
        "current_plan": {
            "logical_reflection_calls": 8,
            "logical_selector_calls": 3,
            "logical_calls": 11,
            "maximum_physical_provider_attempts": 22,
        },
        "cumulative_logical_calls_after_current": 27,
        "maximum_cumulative_physical_provider_attempts_after_current": 46,
        "artifact_115_old_derived_logical_call_ceiling": 23,
        "logical_calls_above_old_derived_ceiling": 4,
        "old_derived_ceiling_compliance_claimed": False,
        "literal_paid_call_cap_compliance_claimed": False,
        "separate_diagnostic_schema_probe_requests": 2,
        "diagnostic_probe_requests_included_in_stage_a_totals": False,
        "additional_revision_calls_permitted": 0,
        "no_further_execution_revision_permitted": True,
    }
    verified = launch.verify_manifest(manifest)
    assert verified.output_dir == output.resolve()


def test_predecessor_failure_binding_is_recursively_authenticated() -> None:
    binding = launch.predecessor_failure_binding()
    assert binding["v1"]["recursive_file_count"] == 6
    assert binding["v1"]["recursive_content_sha256"] == (
        "4b155380961f420b876da1bb3d48891829aa69b9ed7701b2e002774628a52973"
    )
    assert binding["v1"]["unique_sanitized_validation_issue"] == {
        "category": "bounds_or_length",
        "location": ["insights", "item", "affected_paths", "item"],
    }
    assert binding["v1r1"]["recursive_file_count"] == 6
    assert binding["v1r1"]["recursive_content_sha256"] == (
        "80fcb03f406f162daebcf52270f4ad9fb8a8c2c9882525e98b36a8b47e96ff25"
    )
    assert binding["v1r1"]["provider_inferences"] == 0


def test_v1r2_scientific_projection_and_artifact_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = launch._plan_record(launch.DEFAULT_SEALED_ORACLE_DIR)
    drifted = dict(plan)
    drifted["planned_selector_calls"] = 4
    with pytest.raises(RuntimeError, match="scientific surfaces changed"):
        launch._scientific_surface_binding(drifted)

    monkeypatch.setattr(launch, "V1R2_REVISION_ARTIFACT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="revision artifact drifted"):
        launch._provider_schema_compatibility_evidence()


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("V1_FINALIZED_FILE_SHA256", "predecessor failure artifact drifted"),
        ("V1R1_FINALIZED_FILE_SHA256", "v1r1 failure artifact drifted"),
    ],
)
def test_predecessor_digest_drift_blocks_manifest_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    message: str,
) -> None:
    monkeypatch.setattr(launch, constant, "0" * 64)
    with pytest.raises(RuntimeError, match=message):
        _build(tmp_path, "portfolio_predecessor_drift_test")


def test_source_drift_fails_before_credentials_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, output, _ = _build(tmp_path, "portfolio_drift_test")
    original = launch.source_snapshot
    credentials_loaded = False

    def drifted() -> dict[str, object]:
        record = original()
        record["sha256"] = "0" * 64
        return record

    def credential_loader() -> str:
        nonlocal credentials_loaded
        credentials_loaded = True
        return "must-not-be-read"

    monkeypatch.setattr(launch, "source_snapshot", drifted)
    dependencies = launch.LiveDependencies(
        credential_loader=credential_loader,
        stack_factory=lambda **_: pytest.fail("stack construction is unreachable"),
    )
    with pytest.raises(RuntimeError, match="source snapshot drifted"):
        launch.execute_with_dependencies(manifest, dependencies)
    assert credentials_loaded is False
    assert not output.exists()


class _FakeStack:
    generator = object()
    selector = object()

    async def __aenter__(self) -> "_FakeStack":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


async def _fake_stage_executor(**kwargs: Any) -> dict[str, object]:
    sink = kwargs["sink"]
    sink("development_plan", {"fake": "plan"})
    sink("reflection_results", {"fake": "reflections"})
    sink("selector_results", {"survives_stage_a_v1": True})
    return {"schema_version": 1, "survives_stage_a_v1": True}


def test_injected_provider_free_execution_publishes_result_and_recursive_seal(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path, "portfolio_injected_test")
    events: list[str] = []

    def credential_loader() -> str:
        events.append("credential")
        return "injected-nonsecret-key"

    def stack_factory(**kwargs: Any) -> _FakeStack:
        assert kwargs["api_key"] == "injected-nonsecret-key"
        events.append("stack")
        return _FakeStack()

    summary = launch.execute_with_dependencies(
        manifest,
        launch.LiveDependencies(
            credential_loader=credential_loader,
            stack_factory=stack_factory,
            stage_executor=_fake_stage_executor,
            enforce_provider_accounting=False,
        ),
    )

    assert events == ["credential", "stack"]
    assert summary["status"] == "completed_provider_only_stage_a"
    assert summary["design_id"] == (
        "airfoil_v7_oracle_portfolio_stage_a_v1r2_provider_grammar"
    )
    assert summary["mechanism_revision_ordinal"] == 0
    assert summary["candidate_evaluations"] == 0
    assert summary["cfd_calls"] == 0
    assert (output / "provider_queue_outcomes.jsonl").read_bytes() == b""
    assert (output / "prompt_response_journal.jsonl").read_bytes() == b""
    assert (output / "result.json").is_file()
    final = launch._load_object(output / "finalized.json")
    assert final["status"] == "completed_provider_only_stage_a"
    assert "result.json" in final["files"]
    assert "summary.json" in final["files"]
    assert "finalized.json" not in final["files"]
