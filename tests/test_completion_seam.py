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
        "id": "gen-abc123",
        "model": "openai/gpt-5.6-luna",
        "provider": "OpenAI",
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
    # The three legacy keys every sealed journal reader indexes by are still
    # there and still carry the same values. The record is WIDER than it was:
    # `provider_served`/`provider_pinned`/`response_id` were added because no
    # journal in the program could previously answer "was this rung served by
    # one provider?" -- see the seam docstring. Readers use .get, so widening
    # is compatible; narrowing would not be, which is what this pins.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append)
    complete("hello")
    assert len(records) == 1
    assert {"model_requested", "model_served", "usage"} <= set(records[0])
    assert records[0]["model_requested"] == "openai/gpt-5.6-luna"
    assert records[0]["model_served"] == "openai/gpt-5.6-luna"
    assert records[0]["usage"]["completion_tokens"] == 7


def test_served_provider_is_journalled_even_when_nothing_is_pinned(
    monkeypatch,
) -> None:
    # M6: a ladder rung that silently spans serving providers spans reasoning
    # IMPLEMENTATIONS, and the dose difference it reports is confounded. The
    # served provider must be readable from the journal alone, pinned or not.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append,
                              effort="high")
    complete("hello")
    assert records[0]["provider_served"] == "OpenAI"
    assert records[0]["provider_pinned"] is None
    assert records[0]["response_id"] == "gen-abc123"
    assert b"provider" not in captured[0].data       # nothing was sent


def test_provider_only_pins_the_route_and_is_echoed_into_the_journal(
    monkeypatch,
) -> None:
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append,
                              effort="xhigh", provider_only=("openai",))
    assert complete("hello") == "ok"
    sent = json.loads(captured[0].data.decode())
    assert sent.pop("provider") == {"only": ["openai"]}
    assert sent.pop("reasoning") == {"effort": "xhigh"}
    assert sent == {"model": "openai/gpt-5.6-luna",
                    "messages": [{"role": "user", "content": "hello"}]}
    assert records[0]["provider_pinned"] == ["openai"]
    assert records[0]["provider_served"] == "OpenAI"


def _truncated_payload() -> dict:
    """What a `max`-effort call actually returned: all budget, no answer.

    Measured on `openai/gpt-5.6-luna` at effort=max: 65,536 completion tokens,
    65,536 of them reasoning, `content: null`, finish_reason `length`. The
    provider default output cap is 65,536 while the route's own
    `max_completion_tokens` is 128,000.
    """
    return {
        "id": "gen-truncated",
        "model": "openai/gpt-5.6-luna",
        "provider": "OpenAI",
        "choices": [{"message": {"content": None},
                     "finish_reason": "length",
                     "native_finish_reason": "max_output_tokens"}],
        "usage": {"completion_tokens": 65536,
                  "completion_tokens_details": {"reasoning_tokens": 65536}},
    }


def test_the_cap_is_additive_and_lands_in_the_body(monkeypatch) -> None:
    # The seam used to send NO completion limit, which is not "no limit" -- it
    # is the provider's default (65,536 here) while the route allows 128,000
    # and the shipped execution profile declares 128,000. Half the declared
    # capacity was unreachable from this seam.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna", effort="max",
                              max_output_tokens=128000)
    assert complete("hello") == "ok"
    sent = json.loads(captured[0].data.decode())
    assert sent["max_tokens"] == 128000
    assert sent.pop("reasoning") == {"effort": "max"}
    sent.pop("max_tokens")
    assert sent == {"model": "openai/gpt-5.6-luna",
                    "messages": [{"role": "user", "content": "hello"}]}

    captured2: list = []
    _serve_canned(monkeypatch, captured2)
    completion_for("openai/gpt-5.6-luna")("hello")
    assert b"max_tokens" not in captured2[0].data


def test_a_reasoning_only_truncated_reply_raises_instead_of_returning_none(
    monkeypatch,
) -> None:
    # The failure this seam is documented to prevent, arriving through a route
    # it did not check: content=None was returned verbatim to the caller, and
    # `author_surrogate` then crashed inside a regex on a NoneType -- or, worse
    # on another path, would have degraded silently to no guidance while the
    # run still reported a model arm.
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setattr("time.sleep", lambda _s: None)

    import urllib.request

    calls: list = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return _CannedResponse(_truncated_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", attempts=3,
                              journal=records.append, effort="max")
    with pytest.raises(RuntimeError, match="empty completion"):
        complete("hello")
    # It retried rather than accepting the non-answer on the first look ...
    assert len(calls) == 3
    # ... and every attempt is still JOURNALLED, so the truncation rate of a
    # dose is measurable from journals instead of being inferred from a crash.
    assert len(records) == 3
    assert records[0]["finish_reason"] == "length"
    assert records[0]["native_finish_reason"] == "max_output_tokens"
    assert records[0]["content_chars"] is None
    assert records[0]["usage"]["completion_tokens_details"][
        "reasoning_tokens"] == 65536


def test_a_blank_completion_is_also_a_non_answer(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setattr("time.sleep", lambda _s: None)

    import urllib.request

    payload = _provider_payload()
    payload["choices"][0]["message"]["content"] = "   \n "
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout=None: _CannedResponse(payload))
    with pytest.raises(RuntimeError, match="empty completion"):
        completion_for("openai/gpt-5.6-luna", attempts=2)("hello")


def test_finish_reason_is_journalled_even_without_an_effort_pin(
    monkeypatch,
) -> None:
    # Truncation is not an effort-only hazard, so the evidence for it must not
    # be gated on the effort pin the way it used to be.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    completion_for("openai/gpt-5.6-luna", journal=records.append)("hello")
    assert records[0]["finish_reason"] == "stop"
    assert records[0]["content_chars"] == 2
    assert "effort_requested" not in records[0]


def test_provider_pin_omitted_leaves_the_body_byte_identical(
    monkeypatch,
) -> None:
    # Both pins are ADDITIVE. With neither, the body is the pre-effort body.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna")
    complete("hello")
    assert captured[0].data == json.dumps({
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()


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


# --- from `wo/completion-max-output-tokens`, kept whole ----------------------
# The second independent fix of this seam. Its parameter was named
# `max_output_tokens` for the profile field it carries, which is the name that
# survived consolidation; its tests are kept verbatim in substance, because
# they pin two things the other branch's did not: the WIRE SPELLING (the body
# key is `max_tokens`, not the parameter's name) and the fact that a truncated
# reply which still carries text is journalled rather than raised.

def test_max_output_tokens_omitted_leaves_the_body_byte_identical(
    monkeypatch,
) -> None:
    # The cap parameter is ADDITIVE, like `effort`: every sealed cell that ran
    # through this seam without it must still produce the same two-key body.
    # THIS IS THE NON-NEGOTIABLE INVARIANT of the consolidation -- no sealed
    # measurement may be retroactively altered by a later product fix.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna")
    assert complete("hello") == "ok"
    legacy = json.dumps({
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()
    assert captured[0].data == legacy
    assert b"max_tokens" not in captured[0].data
    assert b"provider" not in captured[0].data
    assert b"reasoning" not in captured[0].data


def test_max_output_tokens_is_sent_as_max_tokens_and_nothing_else(
    monkeypatch,
) -> None:
    # An unpinned request is cut at the PROVIDER's default (65,536 measured on
    # openai/gpt-5.6-luna), not at the route's advertised 128,000, so a
    # high-effort call can spend its whole budget on reasoning and return a
    # truncated artifact while looking successful. The product's own
    # OpenRouterModelExecutionProfile declares max_output_tokens=128_000; this
    # seam must be able to send it, under OpenRouter's own spelling.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna", max_output_tokens=128_000)
    assert complete("hello") == "ok"
    sent = json.loads(captured[0].data.decode())
    assert sent.pop("max_tokens") == 128_000
    assert sent == {
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_cap_and_effort_compose_into_one_body(monkeypatch) -> None:
    captured: list = []
    _serve_canned(monkeypatch, captured)
    complete = completion_for("openai/gpt-5.6-luna", effort="max",
                              max_output_tokens=128_000)
    assert complete("hello") == "ok"
    sent = json.loads(captured[0].data.decode())
    assert sent.pop("reasoning") == {"effort": "max"}
    assert sent.pop("max_tokens") == 128_000
    assert sent == {
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_a_truncated_reply_is_visible_in_the_journal(monkeypatch) -> None:
    # The defect this fix closes was INVISIBLE without an effort pin: the
    # journal carried usage only, so `finish_reason="length"` never reached
    # disk and a truncated artifact was indistinguishable from a short one.
    # This reply still HAS text, so it is a legitimate return -- and it is
    # exactly the case the "empty completion raises" rule does not cover,
    # which is why the journalling has to stand on its own.
    import urllib.request

    truncated = {
        "model": "openai/gpt-5.6-luna",
        "choices": [{"message": {"content": "def propose("},
                     "finish_reason": "length",
                     "native_finish_reason": "max_output_tokens"}],
        "usage": {"completion_tokens": 65536,
                  "completion_tokens_details": {"reasoning_tokens": 63446}},
    }
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout=None: _CannedResponse(truncated))
    records: list[dict] = []
    complete = completion_for("openai/gpt-5.6-luna", journal=records.append,
                              max_output_tokens=128_000)
    assert complete("hello") == "def propose("
    assert records[0]["finish_reason"] == "length"
    assert records[0]["native_finish_reason"] == "max_output_tokens"
    assert records[0]["max_output_tokens_requested"] == 128_000


def test_an_uncapped_call_journals_no_cap_key(monkeypatch) -> None:
    # A pin is journalled when it is SENT. The absence of the key is the
    # evidence that nothing was pinned -- the same reading the M11 audit used
    # on `effort_requested` to tell an unaffected row from an affected one.
    captured: list = []
    _serve_canned(monkeypatch, captured)
    records: list[dict] = []
    completion_for("openai/gpt-5.6-luna", journal=records.append)("hello")
    assert "max_output_tokens_requested" not in records[0]
    assert "effort_requested" not in records[0]
    assert records[0]["provider_pinned"] is None     # the stated exception


# --- the SHIPPED default: the profile's declaration, not the provider's ------

def test_the_product_declares_the_route_ceiling_it_could_not_send(
) -> None:
    # 128,000 is what OpenRouterModelExecutionProfile has declared for the
    # luna and sol routes all along; the provider's silent default is 65,536.
    # Shipping the declaration is the whole point of the fix.
    from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
        declared_max_output_tokens)

    assert declared_max_output_tokens("openai/gpt-5.6-luna") == 128_000
    assert declared_max_output_tokens("openai/gpt-5.6-sol") == 128_000
    # The harness prefix that `Settings.model` carries must resolve too, or
    # the shipped default would silently miss the shipped default model.
    assert declared_max_output_tokens(
        "openrouter:openai/gpt-5.6-luna") == 128_000
    # An undeclared route declares nothing, and nothing is what gets sent.
    assert declared_max_output_tokens("someone/unknown-model") is None


def test_optimize_sends_the_declared_ceiling_and_never_invents_one(
    monkeypatch,
) -> None:
    # The seam keeps no default, so the shipped cap has to be applied by the
    # caller that knows the route. This is that caller.
    import agent_evolve.integrations.completion as completion_module
    from agent_evolve import optimize
    from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
    from pydantic import BaseModel
    from typing import Literal

    class _Candidate(BaseModel):
        width: Literal[8, 16]

    class _Problem:
        candidate_model = _Candidate
        objectives = (ObjectiveSpec(name="energy", goal="min"),)

        def seeds(self):
            return ({"width": 16},)

        def validate(self, config):
            return ValidationOutcome(ok=True)

        def materialize(self, config):
            return dict(config)

        def evaluate(self, artifact):
            return {"energy": float(artifact["width"])}

    seen: list[dict] = []

    def _capture(model, settings=None, **kwargs):
        seen.append(dict(kwargs, model=model))
        return lambda prompt: "{}"

    monkeypatch.setattr(completion_module, "completion_for", _capture)

    optimize(_Problem(), budget=6, seed=1, proposer="llm",
             model="openai/gpt-5.6-luna")
    assert seen and seen[0]["max_output_tokens"] == 128_000

    seen.clear()
    optimize(_Problem(), budget=6, seed=1, proposer="llm",
             model="someone/unknown-model")
    assert seen and seen[0]["max_output_tokens"] is None
