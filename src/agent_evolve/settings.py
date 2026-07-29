"""Runtime settings, read from the process environment at the composition root.

Credential loading **never searches the filesystem**. A ``.env`` file is read
only when the caller names one -- through the ``dotenv_path`` argument or the
``AGENTEVOLVE_DOTENV`` environment variable. Everything else comes from the real
process environment.

This is deliberate. ``dotenv.load_dotenv()`` with no argument walks *upward*
from the calling module until it finds any ``.env``, so a library buried in a
monorepo silently adopts an unrelated project's credentials and bills them. It
also defeats key scrubbing: a run launched as ``env -u OPENAI_API_KEY ...`` to
prove it made no provider call would have the key handed straight back to it.
The loader must therefore stay inside what the caller explicitly points at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_MODEL = "openai:gpt-4o"
_DEFAULT_HARNESS = "pydantic_ai"

_API_KEYS = ("OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")

#: Environment variable naming an explicit ``.env`` file to load.
DOTENV_PATH_VAR = "AGENTEVOLVE_DOTENV"


@dataclass(frozen=True)
class AgentEvolveSettings:
    """Model selection and credentials. Secrets are never echoed by repr helpers."""

    model: str = _DEFAULT_MODEL
    harness: str = _DEFAULT_HARNESS
    temperature: Optional[float] = None
    dotenv_source: Optional[str] = None

    @classmethod
    def from_env(cls, *, dotenv_path: Optional[str] = None) -> "AgentEvolveSettings":
        """Read settings from the process environment.

        ``dotenv_path`` (or ``AGENTEVOLVE_DOTENV``) names a ``.env`` to merge in
        underneath the process environment. Without one, no file is read at all
        and the process environment is the single source of truth.
        """
        source = _load_dotenv(dotenv_path)
        temp = os.environ.get("AGENTEVOLVE_TEMPERATURE")
        return cls(
            model=os.environ.get("AGENTEVOLVE_MODEL", _DEFAULT_MODEL),
            harness=os.environ.get("AGENTEVOLVE_HARNESS", _DEFAULT_HARNESS),
            temperature=float(temp) if temp else None,
            dotenv_source=source,
        )

    def public_metadata(self) -> Dict[str, Any]:
        """Settings safe to log (no secrets)."""
        return {
            "model": self.model,
            "harness": self.harness,
            "temperature": self.temperature,
            "available_keys": [k for k in _API_KEYS if os.environ.get(k)],
            "dotenv_source": self.dotenv_source,
        }


def _load_dotenv(dotenv_path: Optional[str]) -> Optional[str]:
    """Load an explicitly named ``.env``. Never search for one.

    Returns the resolved path actually loaded, or ``None`` when the settings came
    from the process environment alone -- so a caller can record *which* file, if
    any, supplied credentials.

    Raises ``FileNotFoundError`` when a caller names a file that is not there. A
    named path that silently does nothing is exactly how credential bugs hide.
    """
    named = dotenv_path or os.environ.get(DOTENV_PATH_VAR)
    if not named:
        return None
    resolved = Path(named).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"agent_evolve dotenv file not found: {resolved}")
    try:
        from dotenv import load_dotenv
    except Exception as exc:  # pragma: no cover - python-dotenv is a hard dep
        raise RuntimeError(
            f"a dotenv file was requested ({resolved}) but python-dotenv is not installed"
        ) from exc
    # override=False: the real process environment always outranks the file, so
    # scrubbing a key out of the environment cannot be undone by a stale .env.
    load_dotenv(resolved, override=False)
    return str(resolved.resolve())
