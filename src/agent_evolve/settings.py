"""Runtime settings and credential loading at the composition root.

Two rules, both learned from defects.

**Never search.** A ``.env`` is read only when the caller names one -- through
``dotenv_path``/``AGENTEVOLVE_DOTENV`` here, or the explicit path handed to
:func:`load_credentials`. ``dotenv.load_dotenv()`` with no argument walks
*upward* from the calling module until it finds any ``.env``, so a library
buried in a monorepo silently adopts an unrelated project's credentials.

**Never re-introduce a name the operator removed.** ``load_dotenv(path,
override=False)`` looks safe and is not. ``override=False`` defers only to
variables that are *present*; a variable deliberately removed with ``env -u
OPENROUTER_API_KEY`` is absent, so the file value is set anyway. A run launched
to demonstrate that it made no provider call was handed the key straight back by
the very next line. A process cannot tell "never set" from "deliberately unset",
so the intent has to be stated -- name the variables in ``AGENTEVOLVE_SCRUBBED``
and they are removed from the environment and never reintroduced from a file,
whatever else the caller asks for::

    AGENTEVOLVE_SCRUBBED=OPENROUTER_API_KEY python run_campaign.py

This module therefore never calls ``dotenv.load_dotenv`` at all. It reads a
named file with ``dotenv_values`` -- which touches nothing -- then applies the
result key by key and returns a record of exactly what it introduced, refused,
and removed, so a run can seal that record as evidence instead of asserting a
constant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

_DEFAULT_MODEL = "openai:gpt-4o"
_DEFAULT_HARNESS = "pydantic_ai"

_API_KEYS = ("OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")

#: Environment variable naming an explicit ``.env`` file to load.
DOTENV_PATH_VAR = "AGENTEVOLVE_DOTENV"

#: Environment variable naming variables that must stay unset (comma separated).
SCRUBBED_VAR = "AGENTEVOLVE_SCRUBBED"

#: Name shapes treated as credentials. Deliberately generic -- no provider is
#: named, so a new provider is covered the day it first appears in a ``.env``.
_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIAL",
    "_CREDENTIALS",
)


def is_credential_name(name: str) -> bool:
    """Whether *name* looks like a secret rather than plain configuration."""
    upper = name.upper()
    return upper.endswith(_CREDENTIAL_SUFFIXES) or upper in {"API_KEY", "TOKEN", "SECRET"}


def scrubbed_names() -> Tuple[str, ...]:
    """Variables the operator declared must stay unset, via ``AGENTEVOLVE_SCRUBBED``."""
    raw = os.environ.get(SCRUBBED_VAR, "")
    return tuple(sorted({part.strip() for part in raw.split(",") if part.strip()}))


def enforce_scrubbed_environment() -> Tuple[str, ...]:
    """Remove every declared-scrubbed variable from the process environment.

    Returns the names actually removed. A no-op when nothing is declared, so
    importing this package cannot surprise a caller who has not asked for it.
    """
    removed = []
    for name in scrubbed_names():
        if name in os.environ:
            del os.environ[name]
            removed.append(name)
    return tuple(removed)


@dataclass(frozen=True)
class CredentialLoad:
    """What a credential load actually did. Names only -- never values."""

    dotenv_path: Optional[str] = None
    introduced: Tuple[str, ...] = ()
    refused_scrubbed: Tuple[str, ...] = ()
    refused_undeclared: Tuple[str, ...] = ()
    already_present: Tuple[str, ...] = ()
    removed_from_environment: Tuple[str, ...] = ()

    def to_record(self) -> Dict[str, Any]:
        """A sealable record of the load, safe to write into a run receipt."""
        return {
            "dotenv_path": self.dotenv_path,
            "introduced": list(self.introduced),
            "refused_scrubbed": list(self.refused_scrubbed),
            "refused_undeclared": list(self.refused_undeclared),
            "already_present": list(self.already_present),
            "removed_from_environment": list(self.removed_from_environment),
        }


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
        named = dotenv_path or os.environ.get(DOTENV_PATH_VAR)
        if named is None:
            enforce_scrubbed_environment()
            source = None
        else:
            source = load_credentials(named).dotenv_path
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
            "scrubbed": list(scrubbed_names()),
        }


def load_credentials(
    dotenv_path: Union[str, Path],
    *,
    allow_credentials: Optional[Sequence[str]] = None,
    override: bool = False,
    optional: bool = False,
) -> CredentialLoad:
    """Merge an explicitly named ``.env`` into the environment, safely.

    This replaces ``dotenv.load_dotenv(path, override=False)`` at every call
    site. The difference that matters: a variable named in
    ``AGENTEVOLVE_SCRUBBED`` is removed from the environment and never set from
    the file, so the scrub actually holds.

    ``allow_credentials`` optionally restricts which credential-shaped names the
    file may introduce -- pass ``()`` for a run that should need none. ``None``
    (the default) allows any, matching the behaviour of the call it replaces.
    ``override`` replaces variables already in the environment; the scrub list
    outranks it either way.

    Raises ``FileNotFoundError`` if the named file is absent, because a named
    path that silently does nothing is how credential bugs hide. Pass
    ``optional=True`` where the file is genuinely a maybe -- a run that layers a
    workspace ``.env`` over a repository one, say -- and its absence returns an
    empty record instead. The scrub list is enforced either way.
    """
    removed = enforce_scrubbed_environment()
    scrubbed = set(scrubbed_names())
    allowed = None if allow_credentials is None else set(allow_credentials)

    resolved = Path(dotenv_path).expanduser()
    if not resolved.is_file():
        if optional:
            return CredentialLoad(removed_from_environment=removed)
        raise FileNotFoundError(f"agent_evolve dotenv file not found: {resolved}")

    introduced, refused_scrubbed, refused_undeclared, present = [], [], [], []
    for name, value in _read_dotenv_file(resolved).items():
        if value is None:
            continue
        if name in scrubbed:
            refused_scrubbed.append(name)
            continue
        if name in os.environ and not override:
            present.append(name)
            continue
        if allowed is not None and is_credential_name(name) and name not in allowed:
            refused_undeclared.append(name)
            continue
        os.environ[name] = value
        introduced.append(name)

    return CredentialLoad(
        dotenv_path=str(resolved.resolve()),
        introduced=tuple(sorted(introduced)),
        refused_scrubbed=tuple(sorted(refused_scrubbed)),
        refused_undeclared=tuple(sorted(refused_undeclared)),
        already_present=tuple(sorted(present)),
        removed_from_environment=removed,
    )


def _read_dotenv_file(resolved: Path) -> Mapping[str, Optional[str]]:
    """Parse a ``.env`` without touching the environment."""
    try:
        from dotenv import dotenv_values
    except Exception as exc:  # pragma: no cover - python-dotenv is a hard dep
        raise RuntimeError(
            f"a dotenv file was requested ({resolved}) but python-dotenv is not installed"
        ) from exc
    return dotenv_values(resolved)


__all__ = [
    "AgentEvolveSettings",
    "CredentialLoad",
    "DOTENV_PATH_VAR",
    "SCRUBBED_VAR",
    "enforce_scrubbed_environment",
    "is_credential_name",
    "load_credentials",
    "scrubbed_names",
]
