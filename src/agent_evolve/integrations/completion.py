"""One prompt in, one string out.

The operator chooser needs a single completion, not a proposal protocol. A
narrow seam keeps the chooser testable without a provider and keeps the
credential handling in one place instead of threaded through a harness.

Failures are surfaced, never swallowed: a provider that returns an error body
raises after bounded retries rather than yielding empty text, because a chooser
that silently receives nothing degrades into random choice and produces a null
indistinguishable from an honest one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional, Sequence

__all__ = ["Completion", "completion_for", "credential_for", "wire_model"]

Completion = Callable[[str], str]

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")

#: The harness prefix ``Settings.model`` carries. This seam talks to
#: OpenRouter's REST API directly, and OpenRouter's wire model IDs do not carry
#: it -- so it has to come off before the body is built.
_HARNESS_PREFIX = "openrouter:"


def wire_model(model: str) -> str:
    """The OpenRouter wire model ID for a configured route.

    One setting, ``AGENTEVOLVE_MODEL``, feeds two consumers that spell a model
    differently: the pydantic-ai harness wants the provider-qualified
    ``openrouter:openai/gpt-5.6-luna``, and this raw-HTTP seam wants the bare
    ``openai/gpt-5.6-luna``. The package's *default* is written in the harness
    spelling, so every call this seam made on the default route was rejected:

        HTTP 400: "openrouter:openai/gpt-5.6-luna is not a valid model ID"

    four times per call, after which the caller fell back to the classical path
    -- which is the fallback guarantee doing its job, and also why the failure
    could sit there unnoticed: the run still produced a result, and the usage
    ledger reported ``calls: 0`` because no call ever completed.

    Only the ``openrouter:`` prefix is removed. A different vendor prefix is
    left exactly as written, because rewriting it would quietly send one
    vendor's model ID to another vendor rather than failing where the mistake
    is. Bodies that worked before are byte-identical, so no sealed replay
    moves: the only strings this changes are the ones the provider rejected.
    """
    if type(model) is not str:
        raise TypeError("model must be an exact string")
    if model.lower().startswith(_HARNESS_PREFIX):
        return model[len(_HARNESS_PREFIX):]
    return model


def credential_for() -> Optional[tuple[str, str]]:
    """The first credential present, as ``(variable_name, value)``."""

    for name in _KEYS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def completion_for(
    model: str,
    settings: Any = None,
    *,
    attempts: int = 4,
    timeout_s: float = 300.0,
    journal: Any = None,
    effort: Optional[str] = None,
    provider_only: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
) -> Optional[Completion]:
    """A completion callable for *model*, or ``None`` with no credential.

    ``None`` rather than an exception: a caller without a key should fall back
    to the unguided loop, which is a working optimizer, rather than fail. The
    caller announces the fallback so it is never silent.

    ``effort`` is an ADDITIVE reasoning-effort pin: when given, the request
    body carries ``{"reasoning": {"effort": <level>}}`` and each journalled
    record additionally echoes ``effort_requested`` plus the reply's
    ``finish_reason``/``native_finish_reason`` (so honoring is verifiable
    from journals alone). When omitted, the request body is byte-identical to
    the pre-effort seam.

    ``provider_only`` is the second ADDITIVE pin: when given, the body carries
    ``{"provider": {"only": [...]}}``, which is what the shipped
    ``OpenRouterModelExecutionProfile`` declares for these routes
    (``provider_only=("openai",)``). It matters for any dose or scale ladder:
    a router free to move between serving providers can change the *reasoning
    implementation* underneath a rung, so a measured dose difference would be
    confounded with a routing difference.

    It is deliberately NOT pinned by default. Pinning is a measurement-design
    choice that belongs to the row that needs it, not to the product default:
    the one row that measured the pin (ROW O2) found it a no-op against the
    unpinned request, while a default pin would change the body of every call
    this seam has ever made and so would silently re-cut the sealed rows'
    replay. What ships instead is the EVIDENCE: the served provider is
    journalled UNCONDITIONALLY, as ``provider_served`` (plus the reply
    ``response_id``), whether or not the pin is used. Recording it costs
    nothing and is the only way an auditor can check, from journals alone,
    that a rung did not silently span providers -- which no journal in this
    program could do before.

    ``max_output_tokens`` is the third ADDITIVE pin, and it is the one that
    stops a reasoning model from thinking its way past the answer. It is named
    for the ``OpenRouterModelExecutionProfile`` field it is meant to carry
    (``max_output_tokens``); the OpenRouter wire spelling it becomes is
    ``max_tokens``. This seam used to send no completion limit at all, which
    does NOT mean "no limit": it means the PROVIDER's default, measured at
    65,536 on this route, while the route's own ``max_completion_tokens`` is
    128,000 and the shipped profile declares ``max_output_tokens=128_000``.
    At the top of the reasoning-effort range that gap is fatal -- a measured
    ``max``-effort call spent 63,446 of its 65,536 output tokens on reasoning,
    returned ``finish_reason="length"`` /
    ``native_finish_reason="max_output_tokens"``, and produced **no content at
    all**. Half the declared capacity was being left unused, and the artifact
    was what got dropped. Worse, the truncation SELECTS: it deletes exactly the
    calls that reasoned longest, which is the direction that manufactures a
    null on a reasoning-effort ladder. ``agent_evolve.api.optimize`` therefore
    passes the profile's declared ceiling by default (see
    ``declared_max_output_tokens``) rather than accepting the provider's
    silent one -- but this seam itself keeps NO default, so that a caller who
    omits the cap still sends the sealed rows' exact body.

    Which is why a missing completion now RAISES. ``choices[0].message.content``
    can be ``None``, and this function used to return that ``None`` straight to
    the caller -- who is documented above to never receive empty text, because
    a policy that silently receives nothing degrades into random choice and
    produces a null indistinguishable from an honest one. A ``None`` or blank
    content is now treated exactly like a provider error: retried, and raised
    after ``attempts``, carrying the ``finish_reason`` so the caller can see
    the truncation rather than infer it.

    JOURNAL SHAPE, stated once so readers can rely on it. What was SERVED is
    always recorded -- ``model_served``, ``provider_served``, ``response_id``,
    ``finish_reason``, ``native_finish_reason``, ``content_chars``, ``usage``
    -- because truncation and provider drift are not effort-only hazards and
    their evidence must not be gated on a pin the caller may not have used.
    What was PINNED is recorded when it was sent: ``effort_requested`` and
    ``max_output_tokens_requested`` appear only when the corresponding field
    is in the body, so the ABSENCE of ``effort_requested`` in a record means
    exactly one thing -- no ``reasoning`` field was sent, and the route
    resolved its own ``default_effort``. That reading is what the M11 audit
    used to separate the affected rows from the unaffected ones, and it stays
    readable. ``provider_pinned`` is the single exception: it is always
    present (``None`` when unpinned) because an auditor must pair it with
    ``provider_served`` on EVERY record to ask "did this rung span providers?"
    """

    found = credential_for()
    if found is None:
        return None
    _name, key = found
    # The body carries the wire ID; `model_requested` in the journal keeps the
    # configured route, so a record still says which route was asked for.
    on_the_wire = wire_model(model)

    def complete(prompt: str) -> str:
        import urllib.error
        import urllib.request

        fields: dict[str, Any] = {
            "model": on_the_wire,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effort is not None:
            fields["reasoning"] = {"effort": effort}
        if provider_only is not None:
            fields["provider"] = {"only": list(provider_only)}
        if max_output_tokens is not None:
            fields["max_tokens"] = int(max_output_tokens)
        body = json.dumps(fields).encode()
        last: str | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                _ENDPOINT, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            payload: dict | None = None
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    payload = json.loads(response.read().decode())
            except urllib.error.HTTPError as error:
                last = f"HTTP {error.code}: {error.read()[:200].decode(errors='replace')}"
            except Exception as error:                  # network, decode, timeout
                last = f"{type(error).__name__}: {error}"[:200]
            if payload is not None and "choices" in payload:
                choice = payload["choices"][0]
                text = (choice.get("message") or {}).get("content")
                if journal is not None:
                    record = dict(model_requested=model,
                                  model_served=payload.get("model"),
                                  usage=payload.get("usage") or {},
                                  provider_served=payload.get("provider"),
                                  provider_pinned=(None if provider_only is None
                                                   else list(provider_only)),
                                  response_id=payload.get("id"),
                                  finish_reason=choice.get("finish_reason"),
                                  native_finish_reason=choice.get(
                                      "native_finish_reason"),
                                  content_chars=(None if text is None
                                                 else len(text)))
                    if effort is not None:
                        record["effort_requested"] = effort
                    if max_output_tokens is not None:
                        record["max_output_tokens_requested"] = int(
                            max_output_tokens)
                    journal(record)
                if text is not None and text.strip():
                    return text
                # An empty completion is a NON-ANSWER, not an answer. At the
                # top of the reasoning range the provider can spend the whole
                # output budget on reasoning and return content=None with
                # finish_reason="length"; returning that would hand a policy
                # nothing while it reports a model arm. Retry, then raise.
                last = (f"empty completion (finish_reason="
                        f"{choice.get('finish_reason')!r}, native="
                        f"{choice.get('native_finish_reason')!r}, "
                        f"completion_tokens="
                        f"{(payload.get('usage') or {}).get('completion_tokens')}"
                        f", reasoning_tokens="
                        f"{((payload.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')})")
            elif payload is not None:                   # a provider error body
                last = json.dumps(payload.get("error", payload))[:300]
            time.sleep(2 ** attempt)
        raise RuntimeError(
            f"no completion from {model!r} after {attempts} attempts: {last}"
        )

    return complete
