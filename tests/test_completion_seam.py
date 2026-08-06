"""The completion seam must fail loudly and degrade honestly."""

from __future__ import annotations

import json

import pytest

from agent_evolve.integrations.completion import completion_for, credential_for


class _CannedResponse:
    """A urlopen-shaped success carrying one provider payload."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_CannedResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _provider_payload() -> dict:
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [{"message": {"content": "ok"},
                     "finish_reason": "stop",
                     "native_finish_reason": "stop"}],
        "usage": {"completion_tokens": 7,
                  "completion_tokens_details": {"reasoning_tokens": 5},
                  "cost": 0.001},
    }


def _serve_canned(monkeypatch, captured: list) -> None:
    import urllib.request

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return _CannedResponse(_provider_payload())

    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


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


def test_effort_omitted_sends_the_byte_identical_legacy_body(
    monkeypatch,
) -> None:
    # The effort parameter is ADDITIVE: 81 sealed cells ran through this seam
    # with the two-key body, so the default request must not change by a byte.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna")
    assert complete("hello") == "ok"
    assert len(captured) == 1
    legacy = json.dumps({
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()
    assert captured[0].data == legacy
    assert b"reasoning" not in captured[0].data


def test_effort_requested_adds_reasoning_effort_and_nothing_else(
    monkeypatch,
) -> None:
    for level in ("low", "high"):
        captured: list = []
        _serve_canned(monkeypatch, captured)
        complete = completion_for("openai/gpt-5.6-luna", effort=level)
        assert complete("hello") == "ok"
        sent = json.loads(captured[0].data.decode())
        assert sent.pop("reasoning") == {"effort": level}
        assert sent == {
            "model": "openai/gpt-5.6-luna",
            "messages": [{"role": "user", "content": "hello"}],
        }


def test_journal_record_without_effort_keeps_the_legacy_keys(
    monkeypatch,
) -> None:
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append)
    complete("hello")
    assert len(records) == 1
    assert set(records[0]) == {"model_requested", "model_served", "usage"}


def test_journal_echoes_requested_effort_and_finish_reason(
    monkeypatch,
) -> None:
    # The honor gate reads journals alone: the requested effort and the
    # reply's finish_reason must both land there, next to the usage block
    # that carries the measured reasoning tokens.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append,
                              effort="high")
    complete("hello")
    assert len(records) == 1
    record = records[0]
    assert record["effort_requested"] == "high"
    assert record["finish_reason"] == "stop"
    assert record["native_finish_reason"] == "stop"
    assert record["usage"]["completion_tokens_details"][
        "reasoning_tokens"] == 5
