"""Strict sanitizer and explicit role-minimization contracts."""

from __future__ import annotations

import builtins
import time

import pytest

from agent_evolve.domain.artifact import ArtifactRole
from agent_evolve.infrastructure.sanitization import (
    StrictJsonSanitizer,
    TopLevelAllowlistMinimizer,
)
from agent_evolve.ports.artifact_sanitizer import (
    ArtifactMinimizationError,
    ArtifactSanitizationError,
)


def test_allowlist_minimizer_is_role_specific_deterministic_and_fail_closed():
    minimizer = TopLevelAllowlistMinimizer(
        {
            ArtifactRole.DIAGNOSTICS: {"message", "context"},
            ArtifactRole.LLM_RESPONSE: {"response"},
        }
    )
    value = {"unused": "drop", "message": "m", "context": {"z": 1}}

    assert minimizer.minimize_json(value, role=ArtifactRole.DIAGNOSTICS) == {
        "context": {"z": 1},
        "message": "m",
    }
    reordered = TopLevelAllowlistMinimizer(
        {
            ArtifactRole.LLM_RESPONSE: {"response"},
            ArtifactRole.DIAGNOSTICS: {"context", "message"},
        }
    )
    assert minimizer.policy_id == "top-level-allowlist"
    assert minimizer.policy_version == "1"
    assert len(minimizer.policy_config_sha256) == 64
    assert reordered.policy_config_sha256 == minimizer.policy_config_sha256
    with pytest.raises(ArtifactMinimizationError, match="configured role policy"):
        minimizer.minimize_json(value, role=ArtifactRole.RUN_MANIFEST)
    with pytest.raises(ArtifactMinimizationError, match="configured role policy"):
        minimizer.minimize_json("not an object", role=ArtifactRole.DIAGNOSTICS)


def test_strict_sanitizer_recursively_redacts_keys_literals_and_common_credentials():
    exact_secret = "injected-openrouter-value-123"
    sanitizer = StrictJsonSanitizer(exact_secret_values=(exact_secret,))
    raw = {
        "api_key": exact_secret,
        "nested": [
            {"password": "hunter2", "safe": f"before {exact_secret} after"},
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "api_key=another-sensitive-value",
            "sk-abcdefghijklmnopqrstuv",
        ],
        "tuple": ("ok", 2),
    }

    sanitized = sanitizer.sanitize_json(raw, role=ArtifactRole.DIAGNOSTICS)

    assert sanitized == {
        "api_key": "[REDACTED]",
        "nested": [
            {"password": "[REDACTED]", "safe": "before [REDACTED] after"},
            "[REDACTED]",
            "[REDACTED]",
            "[REDACTED]",
        ],
        "tuple": ["ok", 2],
    }
    assert raw["api_key"] == exact_secret  # caller input was not mutated
    assert sanitizer.policy_id == "strict-json-redaction"
    assert sanitizer.policy_version == "1"
    assert exact_secret not in repr(sanitizer)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"value": "-----BEGIN PRIVATE KEY-----\nmaterial"},
        {"value": "https://user:password@example.test/path"},
        {"value": "\ud800"},
    ],
)
def test_strict_sanitizer_rejects_residual_high_risk_or_non_utf8_content(unsafe):
    with pytest.raises(
        ArtifactSanitizationError,
        match="rejected unsafe or unsupported content",
    ) as caught:
        StrictJsonSanitizer().sanitize_json(
            unsafe,
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert "password" not in str(caught.value)
    assert "PRIVATE KEY" not in str(caught.value)


def test_strict_sanitizer_rejects_secret_in_key_cycles_and_non_json_without_leaks():
    exact_secret = "literal-secret-key-fragment"
    sanitizer = StrictJsonSanitizer(exact_secret_values=(exact_secret,))
    cyclic = []
    cyclic.append(cyclic)

    for unsafe in ({f"prefix-{exact_secret}": "value"}, cyclic, {"x": object()}):
        with pytest.raises(ArtifactSanitizationError) as caught:
            sanitizer.sanitize_json(unsafe, role=ArtifactRole.DIAGNOSTICS)
        assert str(caught.value) == (
            "artifact sanitization rejected unsafe or unsupported content"
        )
        assert exact_secret not in str(caught.value)


def test_sanitizer_does_not_consult_files_or_environment(monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("sanitizer attempted external I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    assert StrictJsonSanitizer(exact_secret_values=("runtime-secret",)).sanitize_json(
        {"value": "runtime-secret"},
        role=ArtifactRole.DIAGNOSTICS,
    ) == {"value": "[REDACTED]"}


def test_exact_secret_policy_rejects_unsafe_short_or_metadata_overlaps():
    for constructor_args in (
        {"exact_secret_values": ("short",)},
        {
            "exact_secret_values": ("policy-secret-value",),
            "policy_id": "policy-secret-value",
        },
    ):
        with pytest.raises(ArtifactSanitizationError) as caught:
            StrictJsonSanitizer(**constructor_args)
        assert str(caught.value) == (
            "artifact sanitization rejected unsafe or unsupported content"
        )


@pytest.mark.parametrize(
    "credential_key",
    [
        "sk-abcdefghijklmnopqrstuv",
        "Authorization: Bearer abcdefghijklmnop",
    ],
)
def test_credential_shaped_object_keys_fail_closed(credential_key):
    with pytest.raises(ArtifactSanitizationError):
        StrictJsonSanitizer().sanitize_json(
            {credential_key: "value"},
            role=ArtifactRole.DIAGNOSTICS,
        )


@pytest.mark.parametrize(
    "secret_key",
    [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "webhook_secret",
        "artifact_signing_key",
    ],
)
def test_common_cloud_and_secret_suffix_fields_are_redacted(secret_key):
    sanitized = StrictJsonSanitizer().sanitize_json(
        {secret_key: "credential-material"},
        role=ArtifactRole.DIAGNOSTICS,
    )
    assert sanitized == {secret_key: "[REDACTED]"}


@pytest.mark.parametrize(
    "credential_assignment",
    [
        "token=abcdefghijklmnop",
        "secret=abcdefghijklmnop",
        "authorization=abcdefghijklmnop",
        "AWS_SECRET_ACCESS_KEY=abcdefghijklmnop",
        "OPENROUTER_API_KEY=arbitraryCredentialValue",
    ],
)
def test_common_free_text_credential_assignments_are_redacted(
    credential_assignment,
):
    sanitized = StrictJsonSanitizer().sanitize_json(
        {"message": f"before {credential_assignment} after"},
        role=ArtifactRole.DIAGNOSTICS,
    )
    assert "[REDACTED]" in sanitized["message"]
    assert credential_assignment not in sanitized["message"]


def test_sanitizer_bounds_strings_before_regex_and_before_redaction_shrinks_them():
    sanitizer = StrictJsonSanitizer(
        max_string_bytes=32,
        max_total_string_bytes=48,
    )
    for value in (
        {"m": "token=" + "x" * 40},
        {"a": "x" * 24, "b": "y" * 24},
    ):
        with pytest.raises(ArtifactSanitizationError) as caught:
            sanitizer.sanitize_json(value, role=ArtifactRole.DIAGNOSTICS)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_sanitizer_node_budget_counts_keys_and_redacted_value_slots_before_sort():
    sanitizer = StrictJsonSanitizer(max_nodes=4)
    with pytest.raises(ArtifactSanitizationError):
        sanitizer.sanitize_json(
            {"first_password": "x", "second_password": "y"},
            role=ArtifactRole.DIAGNOSTICS,
        )


def test_assignment_scanner_is_linear_on_long_safe_punctuation_text():
    safe = ("A." * 20_000) + "NOPE"
    started = time.monotonic()
    result = StrictJsonSanitizer().sanitize_json(
        {"message": safe},
        role=ArtifactRole.DIAGNOSTICS,
    )
    elapsed = time.monotonic() - started
    assert result == {"message": safe}
    assert elapsed < 0.5


def test_public_adapter_errors_clear_nested_raw_exception_context():
    raw = "TOP-SECRET-\ud800"
    with pytest.raises(ArtifactSanitizationError) as caught:
        StrictJsonSanitizer().sanitize_json(
            {"value": raw},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "TOP-SECRET" not in str(caught.value)

    class LeakyDict(dict):
        def __contains__(self, key):
            raise ValueError("TOP-SECRET")

    minimizer = TopLevelAllowlistMinimizer(
        {ArtifactRole.DIAGNOSTICS: {"message"}}
    )
    with pytest.raises(ArtifactMinimizationError) as minimizer_error:
        minimizer.minimize_json(
            LeakyDict(message="safe"),
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert minimizer_error.value.__cause__ is None
    assert minimizer_error.value.__context__ is None


def test_object_keys_are_type_and_byte_validated_before_sorting():
    comparison_attempted = False

    class HostileString(str):
        def __lt__(self, other):
            nonlocal comparison_attempted
            comparison_attempted = True
            raise ValueError("unsafe comparison")

    with pytest.raises(ArtifactSanitizationError) as caught:
        StrictJsonSanitizer().sanitize_json(
            {HostileString("key"): "value", "other": "value"},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert comparison_attempted is False
    assert caught.value.__context__ is None

    with pytest.raises(ArtifactSanitizationError):
        StrictJsonSanitizer(max_string_bytes=16).sanitize_json(
            {"x" * 10_000: "value", "y" * 10_000: "value"},
            role=ArtifactRole.DIAGNOSTICS,
        )


def test_json_value_subclasses_cannot_override_strict_utf8_or_container_rules():
    class EvilString(str):
        def encode(self, *args, **kwargs):
            return b"falsely-safe"

    class EvilList(list):
        pass

    for unsafe in (
        {"value": EvilString("\ud800")},
        EvilList(["safe-looking"]),
    ):
        with pytest.raises(ArtifactSanitizationError) as caught:
            StrictJsonSanitizer().sanitize_json(
                unsafe,
                role=ArtifactRole.DIAGNOSTICS,
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "credential_policy_id",
    [
        "sk-abcdefghijklmnopqrstuv",
        "or-v1-abcdefghijklmnopqrstuv",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "prefix_sk-abcdefghijklmnopqrstuv",
    ],
)
def test_policy_metadata_rejects_credential_shapes(credential_policy_id):
    with pytest.raises(ArtifactSanitizationError):
        StrictJsonSanitizer(policy_id=credential_policy_id)
