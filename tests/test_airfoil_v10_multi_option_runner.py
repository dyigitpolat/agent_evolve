"""Provider-free durability and provider-manifest tests for the v10 runner."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.application.live_runtime_manifest import (
    RuntimeManifestSection,
    build_live_runtime_manifest,
    capture_runtime_source_closure,
)
from agent_evolve.policies.variation.crossover_inheritance import (
    CrossoverInheritanceClaim,
    CrossoverInheritanceSource,
    materialize_crossover_inheritance,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
    exact_parent_import_exclusions_sha256,
    materialize_exact_parent_crossover,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_runner import (
    DEFAULT_PROVIDER_PROFILE_ID,
    GPT_XHIGH_PROVIDER_PROFILE_ID,
    AirfoilV10MultiOptionRunnerError,
    _bind_readiness_selected_provider,
    _readiness_development,
    _execute_live_development,
    _provider_config_record,
    _resolve_profile,
    _slot_record,
    _verified_model_crossover_materialization_record,
    execute_live,
    readiness,
    validate_airfoil_v10_readiness_record,
)


class _Receipt:
    def to_record(self) -> dict[str, object]:
        return {"lease_id": "provider_free_fake"}


class _Qualification:
    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "receipt_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "installed_distributions": {
                "httpx": "0.test",
                "openai": "0.test",
                "pydantic": "0.test",
                "pydantic-ai": "0.test",
                "pytest": "0.test",
            },
            "non_circular_external_receipt": True,
        }


class _Lease:
    def __init__(self) -> None:
        self.active = False

    def acquire(self) -> _Receipt:
        assert not self.active
        self.active = True
        return _Receipt()

    def release(
        self,
        *,
        outcome: str = "completed",
        failure_type: str | None = None,
    ) -> dict[str, object]:
        assert self.active
        self.active = False
        return {"outcome": outcome, "failure_type": failure_type}

    def __enter__(self) -> _Receipt:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        self.release(
            outcome="completed" if exc_type is None else "failed",
            failure_type=None if exc_type is None else exc_type.__name__,
        )
        return False


class _FakeLive:
    def __init__(self, credential_loader) -> None:
        self.initialized_provider = False
        self._credential_loader = credential_loader
        self.closed = 0

    async def run(self):
        assert self._credential_loader() == "fake-openrouter-key"
        self.initialized_provider = True
        return SimpleNamespace(marker="fake_result")

    async def aclose(self) -> None:
        self.closed += 1


class _ManifestVerification:
    def __init__(self, manifest) -> None:
        self._manifest = manifest

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_sha256": self._manifest.manifest_sha256,
            "source_sha256": self._manifest.source_closure.source_sha256,
            "verification_sha256": "a" * 64,
        }


class _ManifestGate:
    def __init__(self, manifest, *, fail_on: tuple[int, ...] = ()) -> None:
        self.manifest = manifest
        self.verifications = 0
        self.fail_on = fail_on

    def verify(self) -> _ManifestVerification:
        self.verifications += 1
        if self.verifications in self.fail_on:
            raise RuntimeError("provider-free injected manifest drift")
        return _ManifestVerification(self.manifest)


def _fake_runtime_manifest() -> object:
    source = capture_runtime_source_closure(
        {
            "test_source": {
                "tests/test_airfoil_v10_multi_option_runner.py": Path(__file__),
            }
        }
    )
    section = RuntimeManifestSection.seal(
        "test_boundary",
        {"schema_version": 1, "provider_called": False},
    )
    return build_live_runtime_manifest(
        manifest_id="airfoil_v10_runner_test",
        manifest_version=1,
        built_at_utc="2026-07-16T02:00:00Z",
        source_closure=source,
        sections=(section,),
        required_section_ids=(section.section_id,),
    )


def test_provider_records_admit_only_max_deepseek_or_standard_xhigh_gpt() -> None:
    deepseek = _provider_config_record(_resolve_profile(DEFAULT_PROVIDER_PROFILE_ID))
    gpt = _provider_config_record(_resolve_profile(GPT_XHIGH_PROVIDER_PROFILE_ID))

    assert deepseek["reasoning"] == {"max_tokens": 384_000}
    assert gpt["reasoning"] == {"effort": "xhigh"}
    assert deepseek["transport"]["stream_liveness"]["absolute_timeout_ns"] == (
        600_000_000_000
    )
    assert gpt["transport"]["stream_liveness"]["absolute_timeout_ns"] == (
        600_000_000_000
    )
    serialized = json.dumps(gpt, sort_keys=True)
    assert '"mode"' not in serialized
    assert '"pro"' not in serialized
    assert gpt["reasoning_mode_or_pro_fields_absent"] is True


def test_readiness_provider_alias_tracks_selected_profile_and_preserves_preflight() -> (
    None
):
    stale_default = {"schema_version": 1, "profile_id": "legacy-gpt-default"}
    selected = _provider_config_record(_resolve_profile(DEFAULT_PROVIDER_PROFILE_ID))

    record = _bind_readiness_selected_provider(
        {"schema_version": 1, "provider": stale_default},
        selected,
    )

    assert record["provider"] == selected
    assert record["input_default_provider_preflight"] == stale_default


@pytest.mark.parametrize(
    ("relations", "dominates_any_parent"),
    (
        (("worse", "better"), False),
        (("better", "better"), True),
    ),
)
def test_slot_projection_rederives_better_parent_relation_without_scalar_reward(
    relations: tuple[str, str],
    dominates_any_parent: bool,
) -> None:
    parent_ids = ("candidate-parent-left", "candidate-parent-right")
    operator_invocation_id = "operator-parent-relation-projection"
    parents = tuple(
        SimpleNamespace(candidate_id=SimpleNamespace(value=value))
        for value in parent_ids
    )
    plan = SimpleNamespace(
        operator_kind=SimpleNamespace(value="two_parent_crossover"),
        phase="relation_projection",
        parents=parents,
        common_ancestor=None,
    )
    prepared = SimpleNamespace(
        operator_invocation_id=SimpleNamespace(value=operator_invocation_id),
        call_id=None,
        proposal_sequence=1,
    )
    outcome = SimpleNamespace(
        prepared=prepared,
        finite_action_decision=None,
        reward=-1.0,
        failure_stage=None,
        call_failure_type=None,
        dominates_any_parent=dominates_any_parent,
        # This legacy scalar-reward projection is deliberately false.  The
        # result field must instead follow the parent-relative relation record.
        better_than_any_parent=False,
        candidate=None,
    )
    value = SimpleNamespace(
        outcome=outcome,
        slot=SimpleNamespace(
            slot_id="g3_relation_projection",
            role="model_selected_exact_parent_crossover",
            proposal_authority=SimpleNamespace(value="model"),
            plan=plan,
        ),
    )
    engine_trace_rows = (
        {
            "event_type": "invocation_completed",
            "operator_invocation_id": operator_invocation_id,
            "parent_ids": list(parent_ids),
            "parent_outcome_relations": [
                {
                    "parent_candidate_id": parent_candidate_id,
                    "candidate_relation": relation,
                }
                for parent_candidate_id, relation in zip(
                    parent_ids,
                    relations,
                    strict=True,
                )
            ],
            "better_relation_any_parent": True,
        },
    )

    record = _slot_record(value, engine_trace_rows)

    assert record["better_than_any_parent"] is True
    assert record["dominates_any_parent"] is dominates_any_parent


def test_public_readiness_exposes_only_sealed_production_inputs() -> None:
    assert set(inspect.signature(readiness).parameters) == {
        "run_id",
        "qualification_dir",
        "provider_profile_id",
        "run_root",
        "work_root",
    }


def test_injected_readiness_is_authenticated_but_never_live_promotable(
    tmp_path: Path,
) -> None:
    manifest = _fake_runtime_manifest()
    record = _readiness_development(
        "offline-readiness",
        qualification_dir=tmp_path / "qualification",
        problem_factory=lambda *_: object(),
        inputs_loader=lambda **_: {"provider_free": True},
        readiness_record_factory=lambda _: {
            "schema_version": 1,
            "ready": True,
        },
        runtime_manifest_factory=lambda **_: manifest,
        qualification_loader=lambda *_args, **_kwargs: _Qualification(),
        run_root=tmp_path / "runs",
        work_root=tmp_path / "work",
    )

    assert record["readiness_boundary"] == {
        "credential_read": False,
        "dependency_mode": "injected_development_dependencies",
        "entrypoint": "_readiness_development",
        "injected_dependencies_allowed": True,
        "live_promotion_eligible": False,
        "physical_evaluator_called": False,
        "production_dependencies_authenticated": False,
        "production_stack_authenticated": False,
        "provider_called": False,
        "qualification_route_source_authenticated": False,
        "schema_version": 1,
        "scientific_result_eligible": False,
    }
    assert (
        validate_airfoil_v10_readiness_record(
            record,
            require_live_promotable=False,
        )
        == record
    )
    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="not eligible for live promotion",
    ):
        validate_airfoil_v10_readiness_record(record)

    from examples.benchmarks.engibench_airfoil import v10_multi_option_runner

    forged = dict(record)
    forged["readiness_boundary"] = v10_multi_option_runner._readiness_boundary_record(
        production=True
    )
    forged.pop("readiness_commitment_sha256")
    forged["readiness_commitment_sha256"] = (
        v10_multi_option_runner._readiness_commitment(forged)
    )
    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="identity is inconsistent",
    ):
        validate_airfoil_v10_readiness_record(forged)

    tampered = dict(record)
    tampered["ready"] = False
    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="commitment is invalid",
    ):
        validate_airfoil_v10_readiness_record(
            tampered,
            require_live_promotable=False,
        )


def test_injected_runner_reads_credential_once_and_finalizes(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    work_root = tmp_path / "work"
    credential_reads = 0
    captured: dict[str, object] = {}
    manifest = _fake_runtime_manifest()
    manifest_gate = _ManifestGate(manifest)
    qualification_loaded = False

    def qualification_loader(*_args, **_kwargs):
        nonlocal qualification_loaded
        qualification_loaded = True
        return _Qualification()

    def lease_factory(_run_id):
        assert qualification_loaded
        assert manifest_gate.verifications == 1
        return _Lease()

    def credential_source() -> str:
        nonlocal credential_reads
        assert manifest_gate.verifications == 3
        credential_reads += 1
        return "fake-openrouter-key"

    def live_factory(inputs, **kwargs):
        captured["inputs"] = inputs
        captured["profile"] = kwargs["provider_profile"]
        captured["outbound_request_manifest_sink"] = kwargs[
            "outbound_request_manifest_sink"
        ]
        return _FakeLive(kwargs["credential_loader"])

    async def result_factory(result, live, inputs, provider_rows, engine_trace_rows):
        assert result.marker == "fake_result"
        assert inputs == {"provider_free": True}
        assert provider_rows == ()
        assert engine_trace_rows == ()
        assert live.initialized_provider
        return {"schema_version": 1, "injected_provider_free_result": True}

    outcome = asyncio.run(
        _execute_live_development(
            "injected-run",
            qualification_dir=tmp_path / "qualification",
            credential_source=credential_source,
            resource_lease_factory=lease_factory,
            problem_factory=lambda *_: object(),
            inputs_loader=lambda **_: {"provider_free": True},
            readiness_record_factory=lambda _: {
                "schema_version": 1,
                "ready": True,
            },
            runtime_manifest_factory=lambda **_: manifest,
            runtime_manifest_gate_factory=lambda **_: manifest_gate,
            qualification_loader=qualification_loader,
            live_factory=live_factory,
            result_record_factory=result_factory,
            run_root=run_root,
            work_root=work_root,
        )
    )

    run_dir = Path(outcome["run_dir"])
    assert credential_reads == 1
    assert manifest_gate.verifications == 5
    assert captured["inputs"] == {"provider_free": True}
    assert callable(captured["outbound_request_manifest_sink"])
    assert run_dir == run_root / "injected-run"
    assert json.loads((run_dir / "result.json").read_text())[
        "injected_provider_free_result"
    ]
    boundary = json.loads((run_dir / "execution_boundary.json").read_text())
    assert boundary == {
        "dependency_mode": "injected_development_dependencies",
        "entrypoint": "_execute_live_development",
        "injected_dependencies_allowed": True,
        "schema_version": 1,
        "scientific_result_eligible": False,
    }
    assert outcome["result"]["execution_boundary"] == boundary
    finalized = json.loads((run_dir / "finalized.json").read_text())
    assert finalized["status"] == "completed"
    credential = json.loads((run_dir / "credential_access.json").read_text())
    assert credential == {
        "credential_name": "OPENROUTER_API_KEY",
        "read_count": 1,
        "schema_version": 1,
        "stage": "first_g1_model_call_after_two_g0_seed_evaluations",
        "value_persisted": False,
    }
    readiness = json.loads((run_dir / "readiness.json").read_text())
    assert readiness["runtime_manifest"]["manifest_sha256"] == (
        manifest.manifest_sha256
    )
    assert (run_dir / "runtime_manifest.json").is_file()
    assert (run_dir / "runtime_manifest_precomposition_verification.json").is_file()
    assert (run_dir / "runtime_manifest_pre_g0_verification.json").is_file()
    assert (run_dir / "runtime_manifest_precredential_verification.json").is_file()
    assert (run_dir / "runtime_manifest_postoptimizer_verification.json").is_file()
    assert (run_dir / "runtime_manifest_terminal_verification.json").is_file()
    assert (run_dir / "provider_attempt_requests.jsonl").read_text() == ""
    terminal_join = json.loads((run_dir / "provider_attempt_join.json").read_text())
    assert terminal_join["join_valid"] is True
    assert terminal_join["source_counts"]["outbound_manifests"] == 0
    assert terminal_join["expected_framework_versions"] == {
        "httpx": "0.test",
        "openai": "0.test",
        "pydantic": "0.test",
        "pydantic-ai": "0.test",
    }
    assert terminal_join["expected_transport_settings"]["model"] == (
        _resolve_profile(DEFAULT_PROVIDER_PROFILE_ID).model_alias
    )
    assert (
        terminal_join["invariants"]["framework_versions_join_qualification_exact"]
        is True
    )
    assert (
        terminal_join["invariants"]["transport_settings_join_selected_profile_exact"]
        is True
    )
    assert outcome["result"]["provider_attempt_join"] == terminal_join

    with pytest.raises(AirfoilV10MultiOptionRunnerError, match="already exists"):
        asyncio.run(
            _execute_live_development(
                "injected-run",
                qualification_dir=tmp_path / "qualification",
                resource_lease_factory=lambda _: _Lease(),
                run_root=run_root,
                work_root=work_root,
            )
        )


def test_qualification_failure_precedes_resource_lease(tmp_path: Path) -> None:
    lease_factory_calls = 0

    def lease_factory(_run_id):
        nonlocal lease_factory_calls
        lease_factory_calls += 1
        return _Lease()

    def reject_qualification(*_args, **_kwargs):
        raise RuntimeError("stale source-bound qualification")

    with pytest.raises(AirfoilV10MultiOptionRunnerError, match="offline_qualification"):
        asyncio.run(
            _execute_live_development(
                "qualification-failure",
                qualification_dir=tmp_path / "qualification",
                resource_lease_factory=lease_factory,
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                qualification_loader=reject_qualification,
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    assert lease_factory_calls == 0
    run_dir = tmp_path / "runs" / "qualification-failure"
    assert not (run_dir / "resource_lease_acquired.json").exists()
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "offline_qualification"
    assert (
        json.loads((run_dir / "provider_attempt_join.json").read_text())["join_valid"]
        is True
    )
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"


def test_live_failure_preserves_primary_error_and_persists_terminal_verification(
    tmp_path: Path,
) -> None:
    manifest = _fake_runtime_manifest()
    gate = _ManifestGate(manifest, fail_on=(4,))

    class FailingLive(_FakeLive):
        async def run(self):
            assert self._credential_loader() == "fake-openrouter-key"
            raise RuntimeError("provider-free injected primary failure")

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="g0_g3_multi_option_execution",
    ):
        asyncio.run(
            _execute_live_development(
                "terminal-failure",
                qualification_dir=tmp_path / "qualification",
                credential_source=lambda: "fake-openrouter-key",
                resource_lease_factory=lambda _: _Lease(),
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                runtime_manifest_factory=lambda **_: manifest,
                runtime_manifest_gate_factory=lambda **_: gate,
                qualification_loader=lambda *_args, **_kwargs: _Qualification(),
                live_factory=lambda _inputs, **kwargs: FailingLive(
                    kwargs["credential_loader"]
                ),
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    run_dir = tmp_path / "runs" / "terminal-failure"
    assert gate.verifications == 4
    assert not (run_dir / "runtime_manifest_terminal_verification.json").exists()
    assert (run_dir / "runtime_manifest_terminal_verification_failure.json").is_file()
    terminal_failure = json.loads(
        (run_dir / "runtime_manifest_terminal_verification_failure.json").read_text()
    )
    assert terminal_failure["primary_failure_preserved"] is True
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "g0_g3_multi_option_execution"
    assert failure["execution_boundary"]["scientific_result_eligible"] is False
    assert (
        json.loads((run_dir / "provider_attempt_join.json").read_text())["join_valid"]
        is True
    )
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"


def test_terminal_join_mismatch_fails_completion_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.benchmarks.engibench_airfoil import v10_multi_option_runner

    manifest = _fake_runtime_manifest()
    gate = _ManifestGate(manifest)
    monkeypatch.setattr(
        v10_multi_option_runner,
        "build_provider_attempt_terminal_join_receipt",
        lambda **_: {"schema_version": 1, "join_valid": False},
    )

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="provider_attempt_terminal_join",
    ):
        asyncio.run(
            _execute_live_development(
                "join-mismatch",
                qualification_dir=tmp_path / "qualification",
                credential_source=lambda: "fake-openrouter-key",
                resource_lease_factory=lambda _: _Lease(),
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                runtime_manifest_factory=lambda **_: manifest,
                runtime_manifest_gate_factory=lambda **_: gate,
                qualification_loader=lambda *_args, **_kwargs: _Qualification(),
                live_factory=lambda _inputs, **kwargs: _FakeLive(
                    kwargs["credential_loader"]
                ),
                result_record_factory=lambda *_: {"schema_version": 1},
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    run_dir = tmp_path / "runs" / "join-mismatch"
    assert (
        json.loads((run_dir / "provider_attempt_join.json").read_text())["join_valid"]
        is False
    )
    assert not (run_dir / "result.json").exists()
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"


def test_postclose_terminal_join_is_the_only_authoritative_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.benchmarks.engibench_airfoil import v10_multi_option_runner

    manifest = _fake_runtime_manifest()
    join_builds = 0
    transport_closed = False

    class PostCloseManifestGate(_ManifestGate):
        def verify(self) -> _ManifestVerification:
            if self.verifications == 4:
                assert transport_closed
            return super().verify()

    gate = PostCloseManifestGate(manifest)

    def build_join(**_kwargs: object) -> dict[str, object]:
        nonlocal join_builds
        join_builds += 1
        return {
            "schema_version": 1,
            "join_valid": transport_closed,
            "recomputation_number": join_builds,
            "close_published_terminal_evidence": transport_closed,
        }

    class ClosePublishingLive(_FakeLive):
        async def aclose(self) -> None:
            nonlocal transport_closed
            await super().aclose()
            transport_closed = True

    monkeypatch.setattr(
        v10_multi_option_runner,
        "build_provider_attempt_terminal_join_receipt",
        build_join,
    )
    outcome = asyncio.run(
        _execute_live_development(
            "postclose-authority",
            qualification_dir=tmp_path / "qualification",
            credential_source=lambda: "fake-openrouter-key",
            resource_lease_factory=lambda _: _Lease(),
            problem_factory=lambda *_: object(),
            inputs_loader=lambda **_: {"provider_free": True},
            readiness_record_factory=lambda _: {
                "schema_version": 1,
                "ready": True,
            },
            runtime_manifest_factory=lambda **_: manifest,
            runtime_manifest_gate_factory=lambda **_: gate,
            qualification_loader=lambda *_args, **_kwargs: _Qualification(),
            live_factory=lambda _inputs, **kwargs: ClosePublishingLive(
                kwargs["credential_loader"]
            ),
            result_record_factory=lambda *_: {"schema_version": 1},
            run_root=tmp_path / "runs",
            work_root=tmp_path / "work",
        )
    )

    assert join_builds == 1
    assert transport_closed is True
    run_dir = Path(outcome["run_dir"])
    terminal_join = json.loads((run_dir / "provider_attempt_join.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    assert terminal_join["recomputation_number"] == 1
    assert terminal_join["close_published_terminal_evidence"] is True
    assert persisted_result["provider_attempt_join"] == terminal_join
    assert outcome["result"]["provider_attempt_join"] == terminal_join


def test_transport_close_failure_publishes_failure_without_result(
    tmp_path: Path,
) -> None:
    manifest = _fake_runtime_manifest()
    gate = _ManifestGate(manifest)

    class CloseFailingLive(_FakeLive):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("provider-free injected transport close failure")

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="transport_close_and_receipts",
    ):
        asyncio.run(
            _execute_live_development(
                "close-failure",
                qualification_dir=tmp_path / "qualification",
                credential_source=lambda: "fake-openrouter-key",
                resource_lease_factory=lambda _: _Lease(),
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                runtime_manifest_factory=lambda **_: manifest,
                runtime_manifest_gate_factory=lambda **_: gate,
                qualification_loader=lambda *_args, **_kwargs: _Qualification(),
                live_factory=lambda _inputs, **kwargs: CloseFailingLive(
                    kwargs["credential_loader"]
                ),
                result_record_factory=lambda *_: {"schema_version": 1},
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    run_dir = tmp_path / "runs" / "close-failure"
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "transport_close_and_receipts"
    assert failure["failure_type"] == "RuntimeError"
    assert not (run_dir / "result.json").exists()
    assert (
        json.loads((run_dir / "provider_attempt_join.json").read_text())["join_valid"]
        is True
    )
    assert (run_dir / "runtime_manifest_terminal_verification.json").is_file()
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"


def test_resource_lease_release_failure_publishes_failure_without_result(
    tmp_path: Path,
) -> None:
    manifest = _fake_runtime_manifest()
    gate = _ManifestGate(manifest)

    class ReleaseFailingLease(_Lease):
        def release(
            self,
            *,
            outcome: str = "completed",
            failure_type: str | None = None,
        ) -> dict[str, object]:
            del outcome, failure_type
            assert self.active
            self.active = False
            raise RuntimeError("provider-free injected lease release failure")

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="resource_lease_release",
    ):
        asyncio.run(
            _execute_live_development(
                "lease-release-failure",
                qualification_dir=tmp_path / "qualification",
                credential_source=lambda: "fake-openrouter-key",
                resource_lease_factory=lambda _: ReleaseFailingLease(),
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                runtime_manifest_factory=lambda **_: manifest,
                runtime_manifest_gate_factory=lambda **_: gate,
                qualification_loader=lambda *_args, **_kwargs: _Qualification(),
                live_factory=lambda _inputs, **kwargs: _FakeLive(
                    kwargs["credential_loader"]
                ),
                result_record_factory=lambda *_: {"schema_version": 1},
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    run_dir = tmp_path / "runs" / "lease-release-failure"
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "resource_lease_release"
    assert failure["failure_type"] == "RuntimeError"
    assert not (run_dir / "result.json").exists()
    assert (
        json.loads((run_dir / "provider_attempt_join.json").read_text())["join_valid"]
        is True
    )
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"


def test_public_scientific_entrypoint_exposes_no_behavioral_injection() -> None:
    import inspect

    parameters = set(inspect.signature(execute_live).parameters)
    assert parameters == {
        "run_id",
        "qualification_dir",
        "provider_profile_id",
        "run_root",
        "work_root",
    }


def _crossover_materialization_trace_fixture() -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    materialization = materialize_crossover_inheritance(
        left={"a": 1, "b": 10, "shared": "same"},
        right={"a": 2, "b": 20, "shared": "same"},
        draft={"a": 1, "b": 20, "shared": "same"},
        claims=(
            CrossoverInheritanceClaim(
                path="$.a",
                source=CrossoverInheritanceSource.LEFT,
            ),
            CrossoverInheritanceClaim(
                path="$.b",
                source=CrossoverInheritanceSource.RIGHT,
            ),
        ),
    )
    operator_id = "operator-crossover-1"
    call_id = "call-crossover-1"
    candidate_id = "candidate-crossover-1"
    parent_ids = ("candidate-left", "candidate-right")
    parent_patch_hashes = ("a" * 64, "b" * 64)
    source_attribution = (("$.a", "left"), ("$.b", "right"))
    trace_rows: tuple[dict[str, object], ...] = (
        {
            "schema_version": 1,
            "source": "engine",
            "sequence": 20,
            "event_type": "invocation_prepared",
            "operator_invocation_id": operator_id,
            "call_id": call_id,
            "candidate_id": candidate_id,
            "proposal_sequence": 12,
            "operator_kind": "two_parent_crossover",
            "proposal_authority": "model",
            "parent_ids": list(parent_ids),
        },
        {
            "schema_version": 1,
            "source": "engine",
            "sequence": 29,
            "event_type": "candidate_evaluated",
            "operator_invocation_id": operator_id,
            "candidate_id": candidate_id,
            "operator_compliant": True,
            "evidence_compliant": True,
            "parent_patch_hashes": list(parent_patch_hashes),
            "source_attribution": [
                {"path": path, "source": source} for path, source in source_attribution
            ],
            "source_attribution_provenance": (
                "engine_materialized_from_model_inheritance_plan"
            ),
            "target_configuration_hash": (
                materialization.materialized_configuration_sha256
            ),
            "crossover_materialization": materialization.to_record(),
            "crossover_materialization_receipt_sha256": (
                materialization.receipt_sha256
            ),
            "crossover_draft_configuration_hash": (
                materialization.draft_configuration_sha256
            ),
            "crossover_materialized_configuration_hash": (
                materialization.materialized_configuration_sha256
            ),
            "crossover_adjusted_float_leaf_count": 0,
        },
    )
    arguments: dict[str, object] = {
        "slot_id": "g3_model_crossover_1",
        "operator_invocation_id": operator_id,
        "llm_call_id": call_id,
        "candidate_id": candidate_id,
        "proposal_sequence": 12,
        "parent_candidate_ids": parent_ids,
        "configuration_sha256": (materialization.materialized_configuration_sha256),
        "parent_patch_sha256s": parent_patch_hashes,
        "source_attribution": source_attribution,
    }
    return trace_rows, arguments


def test_crossover_result_projection_joins_hash_only_engine_receipt() -> None:
    trace_rows, arguments = _crossover_materialization_trace_fixture()

    record = _verified_model_crossover_materialization_record(
        trace_rows,
        **arguments,
    )

    assert record["evidence_protocol"] == (
        "engine_crossover_materialization_trace_join_v1"
    )
    assert record["operator_invocation_id"] == arguments["operator_invocation_id"]
    assert record["llm_call_id"] == arguments["llm_call_id"]
    assert record["candidate_id"] == arguments["candidate_id"]
    assert (
        record["materialization_receipt_sha256"]
        == trace_rows[1]["crossover_materialization_receipt_sha256"]
    )
    assert record["verification_facts"] == {
        "exact_call_operator_candidate_join": True,
        "exact_parent_identity_join": True,
        "exact_materialized_configuration_join": True,
        "receipt_digest_recomputed": True,
        "attribution_exhaustive_and_nonoverlapping": True,
        "both_named_parent_sources_present": True,
        "inherited_path_count": 2,
        "synthesized_path_count": 0,
        "adjusted_float_leaf_count": 0,
    }
    assert "configuration" not in record


def _exact_parent_crossover_trace_fixture(
    *,
    base: FrozenJsonObject | None = None,
    donor: FrozenJsonObject | None = None,
    import_locus_ids: tuple[str, ...] = ("locus_0001",),
    known_import_locus_ids: tuple[str, ...] = ("locus_0002",),
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    if base is None:
        base = freeze_json({"a": 1, "b": 1, "shared": 0})
    if donor is None:
        donor = freeze_json({"a": 2, "b": 2, "shared": 0})
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    materialization = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=import_locus_ids,
    )
    known_target = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=known_import_locus_ids,
    ).configuration
    forbidden_import_locus_sets = (known_import_locus_ids,)
    exclusions_sha256 = exact_parent_import_exclusions_sha256(
        contract,
        forbidden_import_locus_sets,
    )
    receipt = materialization.receipt
    operator_id = "operator-exact-crossover-1"
    call_id = "call-exact-crossover-1"
    candidate_id = "candidate-exact-crossover-1"
    parent_ids = ("candidate-base", "candidate-donor")
    parent_patch_hashes = ("a" * 64, "b" * 64)
    source_attribution = tuple(
        (
            attribution.path_text,
            "left" if attribution.source.value == "base" else "right",
        )
        for attribution in materialization.attributions
    )
    trace_rows = (
        {
            "schema_version": 1,
            "source": "engine",
            "sequence": 20,
            "event_type": "invocation_prepared",
            "operator_invocation_id": operator_id,
            "call_id": call_id,
            "candidate_id": candidate_id,
            "proposal_sequence": 12,
            "operator_kind": "two_parent_crossover",
            "proposal_authority": "model",
            "parent_ids": list(parent_ids),
            "crossover_response_mode": "exact_parent_import_v1",
            "proposal_representation": "exact_parent_import_v1",
            "exact_parent_crossover_contract": contract.to_record(),
            "exact_parent_crossover_contract_sha256": (contract.contract_sha256),
            "forbidden_exact_parent_import_sets": [
                list(value) for value in forbidden_import_locus_sets
            ],
            "exact_parent_import_exclusions_sha256": exclusions_sha256,
        },
        {
            "schema_version": 1,
            "source": "engine",
            "sequence": 29,
            "event_type": "candidate_evaluated",
            "operator_invocation_id": operator_id,
            "candidate_id": candidate_id,
            "operator_compliant": True,
            "evidence_compliant": True,
            "parent_patch_hashes": list(parent_patch_hashes),
            "source_attribution": [
                {"path": path, "source": source} for path, source in source_attribution
            ],
            "source_attribution_provenance": ("engine_derived_exact_parent_import"),
            "target_configuration_hash": (
                materialization.materialized_configuration_sha256
            ),
            "crossover_contract": contract.to_record(),
            "crossover_contract_sha256": contract.contract_sha256,
            "crossover_import_locus_ids": list(import_locus_ids),
            "crossover_forbidden_import_locus_sets": [
                list(value) for value in forbidden_import_locus_sets
            ],
            "crossover_import_exclusions_sha256": exclusions_sha256,
            "crossover_plan_sha256": materialization.plan.plan_sha256,
            "crossover_materialization": materialization.to_record(),
            "crossover_materialization_sha256": (
                materialization.materialization_sha256
            ),
            "crossover_materialization_receipt": receipt.to_record(),
            "crossover_materialization_receipt_sha256": receipt.receipt_sha256,
            "crossover_materialized_configuration_hash": (
                materialization.materialized_configuration_sha256
            ),
            "crossover_base_parent_candidate_id": parent_ids[0],
            "crossover_donor_parent_candidate_id": parent_ids[1],
        },
    )
    return trace_rows, {
        "slot_id": "g3_exact_crossover_1",
        "operator_invocation_id": operator_id,
        "llm_call_id": call_id,
        "candidate_id": candidate_id,
        "proposal_sequence": 12,
        "parent_candidate_ids": parent_ids,
        "parent_configuration_sha256s": (
            typed_json_sha256(base),
            typed_json_sha256(donor),
        ),
        "parent_configurations": (base, donor),
        "known_target_configurations": (known_target,),
        "configuration": materialization.configuration,
        "configuration_sha256": (materialization.materialized_configuration_sha256),
        "parent_patch_sha256s": parent_patch_hashes,
        "source_attribution": source_attribution,
    }


def test_exact_parent_crossover_result_projection_recomputes_every_digest() -> None:
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()

    record = _verified_model_crossover_materialization_record(
        trace_rows,
        **arguments,
    )

    assert record["schema_version"] == 4
    assert record["evidence_protocol"] == ("engine_exact_parent_import_trace_join_v4")
    assert record["verification_facts"] == {
        "exact_call_operator_candidate_join": True,
        "exact_parent_identity_join": True,
        "proper_nonempty_donor_subset": True,
        "known_target_exclusions_complete": True,
        "known_target_exclusions_digest_recomputed": True,
        "selected_import_set_not_forbidden": True,
        "exact_materialized_configuration_join": True,
        "contract_plan_materialization_receipts_recomputed": True,
        "core_parent_rederivation_and_child_replay": True,
        "attribution_exhaustive_and_nonoverlapping": True,
        "both_named_parent_sources_present": True,
        "locus_count": 2,
        "imported_locus_count": 1,
        "retained_locus_count": 1,
        "known_target_count": 1,
        "forbidden_import_set_count": 1,
        "model_authored_configuration_fields": 0,
        "model_authored_rationale_fields": 0,
    }
    assert record["forbidden_import_locus_sets"] == [["locus_0002"]]
    assert (
        record["import_exclusions_sha256"]
        == (trace_rows[0]["exact_parent_import_exclusions_sha256"])
    )
    assert "configuration" not in record


def test_exact_parent_crossover_interleaved_mask_verifies_semantically() -> None:
    base = freeze_json({f"field_{index:02d}": index for index in range(1, 12)})
    donor = freeze_json({f"field_{index:02d}": 100 + index for index in range(1, 12)})
    imported = tuple(f"locus_{index:04d}" for index in range(4, 8))
    trace_rows, arguments = _exact_parent_crossover_trace_fixture(
        base=base,
        donor=donor,
        import_locus_ids=imported,
        known_import_locus_ids=("locus_0001",),
    )

    canonical_attribution = trace_rows[1]["source_attribution"]
    assert [value["source"] for value in canonical_attribution] == [
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "left",
        "left",
        "left",
        "left",
    ]
    record = _verified_model_crossover_materialization_record(
        trace_rows,
        **arguments,
    )
    assert record["verification_facts"]["imported_locus_count"] == 4
    assert record["verification_facts"]["retained_locus_count"] == 7

    # A legacy engine grouped all left paths before all right paths.  Ordering
    # is not semantic, so the terminal projection accepts the same exhaustive
    # path/source mapping while still replaying every pair from both parents.
    legacy_rows = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    legacy_attribution = sorted(
        canonical_attribution,
        key=lambda value: (value["source"] == "right", value["path"]),
    )
    legacy_rows[1]["source_attribution"] = legacy_attribution
    legacy_arguments = dict(arguments)
    legacy_arguments["source_attribution"] = tuple(
        (value["path"], value["source"]) for value in legacy_attribution
    )

    legacy_record = _verified_model_crossover_materialization_record(
        legacy_rows,
        **legacy_arguments,
    )
    assert (
        legacy_record["materialization_receipt_sha256"]
        == (record["materialization_receipt_sha256"])
    )


@pytest.mark.parametrize(
    ("row_index", "field"),
    (
        (0, "exact_parent_import_exclusions_sha256"),
        (1, "crossover_import_exclusions_sha256"),
    ),
)
def test_exact_parent_crossover_result_projection_rejects_exclusion_digest_damage(
    row_index: int,
    field: str,
) -> None:
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    damaged[row_index][field] = "f" * 64

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="exclusion",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rejects_selected_known_child() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    base, donor = arguments["parent_configurations"]
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    selected = (("locus_0001",),)
    selected_digest = exact_parent_import_exclusions_sha256(contract, selected)
    damaged[0]["forbidden_exact_parent_import_sets"] = [["locus_0001"]]
    damaged[0]["exact_parent_import_exclusions_sha256"] = selected_digest
    damaged[1]["crossover_forbidden_import_locus_sets"] = [["locus_0001"]]
    damaged[1]["crossover_import_exclusions_sha256"] = selected_digest
    arguments["known_target_configurations"] = (arguments["configuration"],)

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="selected a forbidden known child",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rejects_incomplete_known_targets() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    base, donor = arguments["parent_configurations"]
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    wrong = (("locus_0001",),)
    wrong_digest = exact_parent_import_exclusions_sha256(contract, wrong)
    damaged[0]["forbidden_exact_parent_import_sets"] = [["locus_0001"]]
    damaged[0]["exact_parent_import_exclusions_sha256"] = wrong_digest
    damaged[1]["crossover_forbidden_import_locus_sets"] = [["locus_0001"]]
    damaged[1]["crossover_import_exclusions_sha256"] = wrong_digest

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="exclusions are incomplete for the known targets",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_binds_lineage_parent_hashes() -> None:
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    arguments["parent_configuration_sha256s"] = ("1" * 64, "2" * 64)

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="lineage parents",
    ):
        _verified_model_crossover_materialization_record(
            trace_rows,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rejects_counterfeit_parent_config() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    arguments["parent_configurations"] = (
        freeze_json({"a": 1, "b": 9, "shared": 0}),
        arguments["parent_configurations"][1],
    )

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="parent configuration hashes do not verify",
    ):
        _verified_model_crossover_materialization_record(
            trace_rows,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rederives_detached_contract() -> None:
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    counterfeit_base = freeze_json({"a": 1, "b": 9, "shared": 0})
    donor = arguments["parent_configurations"][1]
    counterfeit_parent_sha256s = (
        typed_json_sha256(counterfeit_base),
        typed_json_sha256(donor),
    )
    arguments["parent_configurations"] = (counterfeit_base, donor)
    arguments["parent_configuration_sha256s"] = counterfeit_parent_sha256s
    detached_contract = damaged[0]["exact_parent_crossover_contract"]
    detached_contract["base_parent_sha256"] = counterfeit_parent_sha256s[0]
    detached_contract_sha256 = hashlib.sha256(
        b"agent-evolve:exact-parent-crossover-contract:v1\x00"
        + json.dumps(
            detached_contract,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    ).hexdigest()
    damaged[0]["exact_parent_crossover_contract_sha256"] = detached_contract_sha256

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="contract differs from its lineage parents",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rejects_counterfeit_child_config() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    arguments["configuration"] = freeze_json({"a": 2, "b": 9, "shared": 0})

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="child configuration hash does not verify",
    ):
        _verified_model_crossover_materialization_record(
            trace_rows,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_replays_self_consistent_child_hash() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    counterfeit_child = freeze_json({"a": 2, "b": 9, "shared": 0})
    counterfeit_child_sha256 = typed_json_sha256(counterfeit_child)
    arguments["configuration"] = counterfeit_child
    arguments["configuration_sha256"] = counterfeit_child_sha256
    damaged[1]["target_configuration_hash"] = counterfeit_child_sha256

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="child does not replay from its lineage parents",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_exact_parent_crossover_result_projection_rejects_counterfeit_attribution() -> (
    None
):
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    counterfeit = [
        {"path": '$["counterfeit_left"]', "source": "left"},
        {"path": '$["counterfeit_right"]', "source": "right"},
    ]
    damaged[1]["source_attribution"] = counterfeit
    arguments["source_attribution"] = tuple(
        (value["path"], value["source"]) for value in counterfeit
    )

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="source attribution differs from core replay",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


@pytest.mark.parametrize(
    "field",
    (
        "crossover_contract_sha256",
        "crossover_plan_sha256",
        "crossover_materialization_sha256",
        "crossover_materialization_receipt_sha256",
    ),
)
def test_exact_parent_crossover_result_projection_rejects_digest_damage(
    field: str,
) -> None:
    trace_rows, arguments = _exact_parent_crossover_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    damaged[1][field] = "f" * 64

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="evidence failed closed",
    ):
        _verified_model_crossover_materialization_record(
            damaged,
            **arguments,
        )


def test_crossover_result_projection_fails_when_materialization_receipt_missing() -> (
    None
):
    trace_rows, arguments = _crossover_materialization_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    damaged[1].pop("crossover_materialization")

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="materialization receipt is absent",
    ):
        _verified_model_crossover_materialization_record(damaged, **arguments)


def test_crossover_result_projection_fails_when_receipt_digest_mismatches() -> None:
    trace_rows, arguments = _crossover_materialization_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    damaged[1]["crossover_materialization_receipt_sha256"] = "f" * 64

    with pytest.raises(
        AirfoilV10MultiOptionRunnerError,
        match="receipt digest does not verify",
    ):
        _verified_model_crossover_materialization_record(damaged, **arguments)


@pytest.mark.parametrize("damage", ("missing_receipt", "mismatched_digest"))
def test_runner_terminal_result_projection_fails_closed_on_crossover_evidence(
    tmp_path: Path,
    damage: str,
) -> None:
    trace_rows, arguments = _crossover_materialization_trace_fixture()
    damaged = tuple(json.loads(json.dumps(row)) for row in trace_rows)
    if damage == "missing_receipt":
        damaged[1].pop("crossover_materialization")
    else:
        damaged[1]["crossover_materialization_receipt_sha256"] = "f" * 64
    manifest = _fake_runtime_manifest()
    gate = _ManifestGate(manifest)

    class TracePublishingLive(_FakeLive):
        def __init__(self, credential_loader, engine_trace_sink) -> None:
            super().__init__(credential_loader)
            self._engine_trace_sink = engine_trace_sink

        async def run(self):
            result = await super().run()
            for row in damaged:
                payload = dict(row)
                payload.pop("schema_version")
                payload.pop("source")
                self._engine_trace_sink(payload)
            return result

    async def result_factory(
        _result,
        _live,
        _inputs,
        _provider_rows,
        engine_trace_rows,
    ):
        return {
            "schema_version": 1,
            "crossover": _verified_model_crossover_materialization_record(
                engine_trace_rows,
                **arguments,
            ),
        }

    run_id = f"crossover-evidence-{damage}"
    with pytest.raises(AirfoilV10MultiOptionRunnerError, match="result_projection"):
        asyncio.run(
            _execute_live_development(
                run_id,
                qualification_dir=tmp_path / "qualification",
                credential_source=lambda: "fake-openrouter-key",
                resource_lease_factory=lambda _: _Lease(),
                problem_factory=lambda *_: object(),
                inputs_loader=lambda **_: {"provider_free": True},
                readiness_record_factory=lambda _: {
                    "schema_version": 1,
                    "ready": True,
                },
                runtime_manifest_factory=lambda **_: manifest,
                runtime_manifest_gate_factory=lambda **_: gate,
                qualification_loader=lambda *_args, **_kwargs: _Qualification(),
                live_factory=lambda _inputs, **kwargs: TracePublishingLive(
                    kwargs["credential_loader"],
                    kwargs["engine_trace_sink"],
                ),
                result_record_factory=result_factory,
                run_root=tmp_path / "runs",
                work_root=tmp_path / "work",
            )
        )

    run_dir = tmp_path / "runs" / run_id
    assert not (run_dir / "result.json").exists()
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "result_projection"
    assert json.loads((run_dir / "finalized.json").read_text())["status"] == "failed"
