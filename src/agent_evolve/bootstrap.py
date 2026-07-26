"""Idempotent integration loading: in-tree harnesses + setuptools entry points.

The core never imports ``integrations``; this module (the composition root) does,
once per process. Out-of-tree integrations advertise themselves through the
``agent_evolve.integrations`` entry-point group.
"""

from __future__ import annotations

import importlib

_INTREE = (
    "agent_evolve.integrations.pydantic_ai",
)

_loaded = False


def load_integrations() -> None:
    """Import in-tree integrations and any registered via entry points (once)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _INTREE:
        try:
            importlib.import_module(module)
        except Exception:
            # A broken optional integration must not prevent external adapters
            # from loading through entry points.
            pass
    _load_entry_points()


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
        except Exception:
            pass
