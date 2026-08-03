"""The completion seam must fail loudly and degrade honestly."""

from __future__ import annotations

import pytest

from agent_evolve.integrations.completion import completion_for, credential_for


def test_no_credential_returns_none_rather_than_raising(monkeypatch) -> None:
    # A caller without a key should fall back to the unguided loop, which is a
    # working optimizer, rather than fail outright.
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert credential_for() is None
    assert completion_for("openai/gpt-5.6-luna") is None


def test_a_credential_yields_a_callable(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    assert callable(completion_for("openai/gpt-5.6-luna"))


def test_a_provider_that_never_answers_raises_rather_than_returning_empty(
    monkeypatch,
) -> None:
    # Returning "" would look to the chooser like an unparseable reply, and the
    # run would degrade to random choice while reporting a model arm.
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setattr("time.sleep", lambda _s: None)

    import urllib.request

    def refuse(*_a, **_k):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    complete = completion_for("openai/gpt-5.6-luna", attempts=2)
    with pytest.raises(RuntimeError, match="no completion"):
        complete("hello")
