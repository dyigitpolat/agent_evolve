"""Harness registry: register adapters by string id, create them by id."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from agent_evolve.harness.base import Harness

HarnessFactory = Callable[[], Harness]


class HarnessRegistry:
    """Process-wide registry of harness adapters keyed by string id."""

    def __init__(self) -> None:
        self._factories: Dict[str, HarnessFactory] = {}

    def register(self, harness_id: str, factory: HarnessFactory) -> None:
        self._factories[harness_id] = factory

    def create(self, harness_id: str, **kwargs: Any) -> Harness:
        if harness_id not in self._factories:
            raise KeyError(
                f"Unknown harness '{harness_id}'. Registered: {sorted(self._factories)}"
            )
        factory = self._factories[harness_id]
        try:
            return factory(**kwargs) if kwargs else factory()
        except TypeError:
            # A factory that does not take the requested options still builds;
            # the option was advisory (a seed, say), not part of the contract.
            return factory()

    def ids(self) -> List[str]:
        return sorted(self._factories)

    def clear(self) -> None:
        self._factories.clear()


harness_registry = HarnessRegistry()
