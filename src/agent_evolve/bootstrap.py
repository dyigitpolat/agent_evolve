"""Idempotent integration loading: in-tree proposers + setuptools entry points.

The core never imports ``integrations``; this module (the composition root)
does, once per process. Out-of-tree integrations advertise themselves through
the ``agent_evolve.integrations`` entry-point group.

A failed optional integration must not stop the others loading, but it must not
vanish either: the reason is kept and re-reported when someone asks for a
harness that failed to register. Swallowing it produced ``Unknown harness
'pydantic_ai'. Registered: []`` as the only symptom of a broken install.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Dict, Optional

from agent_evolve.harness.registry import harness_registry
from agent_evolve.proposers.random_proposer import RandomProposer

_INTREE = ("agent_evolve.integrations.pydantic_ai",)

_loaded = False

#: Import failures, kept so a missing harness can explain itself.
load_failures: Dict[str, str] = {}


def _register_builtin() -> None:
    """Register proposers that have no optional dependencies."""
    try:
        harness_registry.create("random")
    except KeyError:
        harness_registry.register("random", RandomProposer)


_register_builtin()


#: Harness id -> the third-party distribution it cannot work without, and the
#: extra that installs it. An adapter that imports its provider lazily still
#: registers successfully without it, and then fails deep inside the first
#: model call with a bare ModuleNotFoundError. Checking here turns that into
#: one sentence naming the install command.
_REQUIRES = {
    "pydantic_ai": ("pydantic_ai", "agentevolve-optimizer[llm]"),
}


def load_integrations() -> None:
    """Import in-tree integrations and any registered via entry points (once)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    _register_builtin()
    for module in _INTREE:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - recorded, then reported on use
            load_failures[module] = f"{type(exc).__name__}: {exc}"
    _load_entry_points()
    _check_lazy_requirements()


def _check_lazy_requirements() -> None:
    """Record harnesses whose provider package is not installed."""
    for harness_id, (dist, extra) in _REQUIRES.items():
        if harness_id not in harness_registry.ids():
            continue
        if importlib.util.find_spec(dist) is None:
            load_failures[harness_id] = (
                f"needs {dist}, which is not installed; pip install '{extra}'"
            )


def requirement_failure(harness_id: str) -> Optional[str]:
    """Return why *harness_id* cannot run, or ``None`` when it can."""
    return load_failures.get(harness_id)


def _load_entry_points() -> None:
    try:
        from importlib.metadata import entry_points
    except Exception:
        return
    try:
        eps = entry_points(group="agent_evolve.integrations")
    except TypeError:
        eps = entry_points().get("agent_evolve.integrations", [])  # py<3.10 shim
    for ep in eps:
        try:
            ep.load()
        except Exception as exc:  # noqa: BLE001 - recorded, then reported on use
            load_failures[getattr(ep, "name", str(ep))] = f"{type(exc).__name__}: {exc}"


def explain_missing_harness(harness_id: str) -> str:
    """Return a usable message for a harness that is not registered."""
    known = sorted(harness_registry.ids()) if hasattr(harness_registry, "ids") else []
    if not load_failures:
        return f"Unknown proposer {harness_id!r}. Available: {known}."
    reasons = "; ".join(f"{name} failed to load ({why})" for name, why in load_failures.items())
    return (
        f"Unknown proposer {harness_id!r}. Available: {known}. {reasons}. "
        "Install the optional dependency, e.g. pip install 'agentevolve-optimizer[llm]'."
    )
