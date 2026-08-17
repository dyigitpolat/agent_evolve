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

#: Default model. Cheap by policy, not by accident: a package that defaults to
#: a frontier model hands a stranger a bill they never chose on their first
#: run. This route is $0.10 per million input tokens and $0.60 per million
#: output. Whatever is set here is echoed before any paid call, so nobody is
#: billed by a default they did not see.
_DEFAULT_MODEL = "openrouter:openai/gpt-5.6-luna"
_DEFAULT_HARNESS = "pydantic_ai"

#: Published price per million tokens for the routes we default to or document,
#: so cost can be shown rather than guessed. Absent from this table means
#: unknown, and unknown is reported as unknown.
MODEL_PRICES_PER_MTOK = {
    "openrouter:openai/gpt-5.6-luna": (0.10, 0.60),
    "openai/gpt-5.6-luna": (0.10, 0.60),
}


def model_price(model: str):
    """Return ``(input, output)`` USD per million tokens, or ``None``."""
    return MODEL_PRICES_PER_MTOK.get(model)

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


#: The variable names model providers actually document. Used ONLY to answer the
#: routing question below -- never to decide what to redact.
#:
#: Exact names, not a vendor substring. A substring rule reads
#: `CLAUDE_CODE_MESSAGING_TOKEN` -- an agent harness's own IPC token -- as an
#: Anthropic credential, which is the same false positive one level down. A new
#: provider is one line here, and `proposer="llm"` reaches any provider without
#: consulting this list at all.
PROVIDER_CREDENTIAL_VARS = frozenset({
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "CO_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "PERPLEXITY_API_KEY",
    "CEREBRAS_API_KEY",
    "MOONSHOT_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_BEDROCK_API_KEY",
    "OLLAMA_API_KEY",
})

#: Prefix reserved for this package, so an operator can declare a credential for
#: a provider the list above does not name yet without editing the package.
_PROVIDER_VAR_PREFIX = "AGENTEVOLVE_"


def is_credential_name(name: str) -> bool:
    """Whether *name* looks like a secret rather than plain configuration.

    This is the **redaction** predicate: it decides what must not be echoed and
    what an allowlist has to name explicitly. Over-classifying here is safe --
    it protects more -- so the rule stays deliberately generic and no provider
    is named, which is why a new provider is covered the day it first appears in
    a ``.env``.

    It is *not* the routing predicate. See :func:`is_provider_credential_name`.
    """
    upper = name.upper()
    return upper.endswith(_CREDENTIAL_SUFFIXES) or upper in {"API_KEY", "TOKEN", "SECRET"}


def is_provider_credential_name(name: str) -> bool:
    """Whether *name* plausibly holds a credential for a **model provider**.

    The two questions the generic rule above used to answer at once are not the
    same question, and conflating them had a consequence in exactly one
    direction. `credentials_present()` decides whether ``proposer="auto"`` may
    reach a provider at all; asking "does this look like a secret" answered
    *yes* for `VSCODE_GIT_IPC_AUTH_TOKEN` and `CLAUDE_CODE_MESSAGING_TOKEN`, so
    on an ordinary developer machine -- an editor open, an agent running, no
    provider account anywhere -- ``auto`` stopped choosing the credential-free
    path and announced that it had found a credential. The documented default
    was being flipped by a variable belonging to something else entirely.

    So routing requires *both* a credential-shaped name and a vendor this
    package can reach. Over-classifying is safe for redaction and unsafe for
    routing, and the asymmetry is why these are two functions.

    The escape hatch is explicit rather than inferred: ``proposer="llm"`` never
    consults this at all, so an unrecognized vendor is one flag away and never a
    silent misroute. ``AGENTEVOLVE_*`` credential names also route, so an
    operator can declare one without editing the package.
    """
    upper = name.upper()
    if upper in PROVIDER_CREDENTIAL_VARS:
        return True
    return upper.startswith(_PROVIDER_VAR_PREFIX) and is_credential_name(upper)


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


def credentials_present() -> bool:
    """True when some provider credential is available in the environment.

    Used only to decide whether ``proposer="auto"`` may reach a provider. A
    name the operator scrubbed does not count as present, so a run launched to
    demonstrate that it made no provider call falls back to random rather than
    quietly finding a key somewhere else.

    The test is :func:`is_provider_credential_name`, not the broad redaction
    rule: a secret-shaped variable belonging to an editor or an agent is not
    evidence of a model-provider account, and treating it as one turned the
    credential-free default off on machines that had no provider credential at
    all.
    """
    scrubbed = set(scrubbed_names())
    for name, value in os.environ.items():
        if not value or name in scrubbed:
            continue
        if is_provider_credential_name(name):
            return True
    return False


__all__ = [
    "AgentEvolveSettings",
    "CredentialLoad",
    "DOTENV_PATH_VAR",
    "SCRUBBED_VAR",
    "MODEL_PRICES_PER_MTOK",
    "credentials_present",
    "model_price",
    "enforce_scrubbed_environment",
    "is_credential_name",
    "is_provider_credential_name",
    "PROVIDER_CREDENTIAL_VARS",
    "load_credentials",
    "scrubbed_names",
]
