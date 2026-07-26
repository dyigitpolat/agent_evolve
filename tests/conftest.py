"""Shared fixtures: reset the harness registry and load in-tree integrations."""

from __future__ import annotations

import os
import sys

import pytest

# Make the test-helper modules importable as top-level modules without adding
# the package root itself to sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evolve import bootstrap
from agent_evolve.harness.registry import harness_registry


@pytest.fixture(autouse=True)
def _fresh_registry():
    harness_registry.clear()
    bootstrap._loaded = False
    bootstrap.load_integrations()
    yield
    harness_registry.clear()
    bootstrap._loaded = False
