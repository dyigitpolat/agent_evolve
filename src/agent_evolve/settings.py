"""Runtime settings, read from environment / ``.env`` at the composition root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

_DEFAULT_MODEL = "openai:gpt-4o"
_DEFAULT_HARNESS = "pydantic_ai"

_API_KEYS = ("OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")


@dataclass(frozen=True)
class AgentEvolveSettings:
    """Model selection and credentials. Secrets are never echoed by repr helpers."""

    model: str = _DEFAULT_MODEL
    harness: str = _DEFAULT_HARNESS
    temperature: Optional[float] = None

    @classmethod
    def from_env(cls, *, dotenv_path: Optional[str] = None) -> "AgentEvolveSettings":
        _load_dotenv(dotenv_path)
        temp = os.environ.get("AGENTEVOLVE_TEMPERATURE")
        return cls(
            model=os.environ.get("AGENTEVOLVE_MODEL", _DEFAULT_MODEL),
            harness=os.environ.get("AGENTEVOLVE_HARNESS", _DEFAULT_HARNESS),
            temperature=float(temp) if temp else None,
        )

    def public_metadata(self) -> Dict[str, Any]:
        """Settings safe to log (no secrets)."""
        return {
            "model": self.model,
            "harness": self.harness,
            "temperature": self.temperature,
            "available_keys": [k for k in _API_KEYS if os.environ.get(k)],
        }


def _load_dotenv(dotenv_path: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
