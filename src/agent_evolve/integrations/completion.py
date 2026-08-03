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
from typing import Any, Callable, Optional

__all__ = ["Completion", "completion_for", "credential_for"]

Completion = Callable[[str], str]

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


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
) -> Optional[Completion]:
    """A completion callable for *model*, or ``None`` with no credential.

    ``None`` rather than an exception: a caller without a key should fall back
    to the unguided loop, which is a working optimizer, rather than fail. The
    caller announces the fallback so it is never silent.
    """

    found = credential_for()
    if found is None:
        return None
    _name, key = found

    def complete(prompt: str) -> str:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
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
                if journal is not None:
                    journal(dict(model_requested=model,
                                 model_served=payload.get("model"),
                                 usage=payload.get("usage") or {}))
                return payload["choices"][0]["message"]["content"]
            if payload is not None:                     # a provider error body
                last = json.dumps(payload.get("error", payload))[:300]
            time.sleep(2 ** attempt)
        raise RuntimeError(
            f"no completion from {model!r} after {attempts} attempts: {last}"
        )

    return complete
