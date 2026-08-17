"""The configured route and the wire model ID are not the same string.

`AGENTEVOLVE_MODEL` feeds two consumers that spell a model differently. The
pydantic-ai harness wants the provider-qualified `openrouter:openai/...`; the
lightweight completion seam posts straight to OpenRouter's REST API, whose model
IDs carry no such prefix. The package's default is written in the harness
spelling, so every call the seam made on the default route came back

    HTTP 400: "openrouter:openai/gpt-5.6-luna is not a valid model ID"

four times per call. Nothing crashed, because the caller falls back to the
classical path when the model gives it nothing -- the fallback guarantee working
exactly as designed, and exactly why this went unnoticed: the run produced a
result and the usage ledger reported `calls: 0`, which reads as "no model was
asked" rather than "every ask was rejected".

The scope of the repair is asserted as tightly as the repair: only the
`openrouter:` prefix comes off, so bodies that already worked are byte-identical
and no sealed replay moves.
"""

from __future__ import annotations

import json

import pytest

from agent_evolve.integrations.completion import completion_for, wire_model
from agent_evolve.settings import _DEFAULT_MODEL


def test_the_shipped_default_route_produces_a_valid_wire_id():
    assert _DEFAULT_MODEL.startswith("openrouter:"), (
        "this test exists because the default carries the harness prefix"
    )
    assert wire_model(_DEFAULT_MODEL) == "openai/gpt-5.6-luna"
    assert ":" not in wire_model(_DEFAULT_MODEL)


@pytest.mark.parametrize("model", [
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-4",
    "meta-llama/llama-3.1-70b-instruct",
])
def test_a_bare_wire_id_passes_through_unchanged(model):
    """Bodies that worked before must be byte-identical after."""
    assert wire_model(model) == model


@pytest.mark.parametrize("model", ["openai:gpt-4o", "anthropic:claude-3"])
def test_another_vendors_prefix_is_left_alone(model):
    """Rewriting it would send one vendor's ID to another vendor, quietly."""
    assert wire_model(model) == model


def test_the_prefix_match_is_case_insensitive():
    assert wire_model("OpenRouter:openai/gpt-5.6-luna") == "openai/gpt-5.6-luna"


def test_a_non_string_route_is_refused_by_type():
    with pytest.raises(TypeError, match="exact string"):
        wire_model(None)  # type: ignore[arg-type]


def test_the_request_body_carries_the_wire_id_not_the_configured_route(monkeypatch):
    """The end-to-end assertion: what actually goes on the wire."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-real-credential")
    sent: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "model": "openai/gpt-5.6-luna",
                "id": "resp-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }).encode()

    def _fake_urlopen(request, timeout=None):
        sent["body"] = json.loads(request.data.decode())
        return _Response()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    complete = completion_for(_DEFAULT_MODEL)
    assert complete is not None
    assert complete("hello") == "ok"
    assert sent["body"]["model"] == "openai/gpt-5.6-luna", (
        "the harness prefix reached the provider again; every call on the "
        "default route would be rejected with HTTP 400"
    )


def test_the_journal_still_records_the_configured_route(monkeypatch):
    """A record must say which route was asked for, prefix and all."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-real-credential")
    records: list[dict] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "model": "openai/gpt-5.6-luna",
                "id": "resp-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }).encode()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout=None: _Response())

    complete = completion_for(_DEFAULT_MODEL, journal=records.append)
    assert complete is not None
    complete("hello")
    assert records, "the call was not journalled"
    assert records[0]["model_requested"] == _DEFAULT_MODEL
    assert records[0]["model_served"] == "openai/gpt-5.6-luna"
