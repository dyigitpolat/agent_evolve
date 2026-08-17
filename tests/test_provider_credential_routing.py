"""Two questions about a variable name, and they are not the same question.

`is_credential_name` answers **"must this be redacted?"** — a broad, shape-based
rule, where over-classifying is safe because it protects more.

`is_provider_credential_name` answers **"may `proposer='auto'` reach a provider?"**
— where over-classifying is *unsafe*, because it turns off the credential-free
default. Using the broad rule for the narrow question meant that on an ordinary
developer machine, with an editor open and an agent running and no provider
account anywhere, `VSCODE_GIT_IPC_AUTH_TOKEN` and `CLAUDE_CODE_MESSAGING_TOKEN`
read as credentials and `auto` announced it had found one.

The asymmetry is the whole point, so it is asserted in both directions.
"""

from __future__ import annotations

import pytest

from agent_evolve.settings import (
    PROVIDER_CREDENTIAL_VARS,
    credentials_present,
    is_credential_name,
    is_provider_credential_name,
)

# Secret-shaped, and belonging to something that is not a model provider.
NOT_A_PROVIDER = [
    "VSCODE_GIT_IPC_AUTH_TOKEN",
    "CLAUDE_CODE_MESSAGING_TOKEN",   # an agent harness's own IPC token
    "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "DATABASE_PASSWORD",
    "SSH_AUTH_SOCK_TOKEN",
    "NPM_TOKEN",
    "SLACK_API_TOKEN",
]

A_PROVIDER = [
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
]


@pytest.mark.parametrize("name", NOT_A_PROVIDER)
def test_secret_shaped_but_not_a_provider(name):
    assert is_credential_name(name), (
        f"{name} must still be redacted: over-classifying is safe for redaction"
    )
    assert not is_provider_credential_name(name), (
        f"{name} does not address a model provider, so it must not route "
        "proposer='auto' away from the credential-free path"
    )


@pytest.mark.parametrize("name", A_PROVIDER)
def test_a_real_provider_credential_routes(name):
    assert is_credential_name(name)
    assert is_provider_credential_name(name)


def test_every_listed_provider_var_is_also_redacted_or_deliberately_named():
    """Nothing may route without also being protected."""
    for name in PROVIDER_CREDENTIAL_VARS:
        assert is_provider_credential_name(name), name


def test_the_package_prefix_is_the_escape_hatch_for_an_unlisted_provider():
    assert is_provider_credential_name("AGENTEVOLVE_PROVIDER_API_KEY")
    # ... but only for credential-shaped names; plain configuration is not one.
    assert not is_provider_credential_name("AGENTEVOLVE_MODEL")
    assert not is_provider_credential_name("AGENTEVOLVE_DOTENV")


def test_credentials_present_is_false_with_only_non_provider_secrets(monkeypatch):
    for name in list(NOT_A_PROVIDER):
        monkeypatch.setenv(name, "present-but-irrelevant")
    for name in PROVIDER_CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    assert credentials_present() is False


def test_credentials_present_is_true_for_a_real_provider_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    assert credentials_present() is True


def test_a_scrubbed_provider_key_still_does_not_count(monkeypatch):
    """The scrub rule outranks detection, as it did before."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.setenv("AGENTEVOLVE_SCRUBBED", "OPENROUTER_API_KEY")
    for name in PROVIDER_CREDENTIAL_VARS - {"OPENROUTER_API_KEY"}:
        monkeypatch.delenv(name, raising=False)
    assert credentials_present() is False
