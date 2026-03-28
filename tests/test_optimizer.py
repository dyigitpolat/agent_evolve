"""Tests for AgentEvolver problem injection mechanics.

These tests verify the sys.modules injection/cleanup without calling the
kedi runtime (which would require an LLM).
"""

import sys
import pytest

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.optimizer import AgentEvolver, _PROBLEM_DEF_MODULE


class _DummyProblem:
    @property
    def objectives(self):
        return [ObjectiveSpec("score", "max")]

    def evaluate(self, config):
        return {"score": float(config.get("x", 0))}


class TestProblemInjection:
    def test_inject_creates_module(self):
        evolver = AgentEvolver(pop_size=2, generations=1)
        evolver._inject_problem_def(_DummyProblem())
        try:
            assert _PROBLEM_DEF_MODULE in sys.modules
            mod = sys.modules[_PROBLEM_DEF_MODULE]
            assert hasattr(mod, "problem")
            assert hasattr(mod, "config")
            assert mod.config["pop_size"] == 2
            assert mod.config["generations"] == 1
        finally:
            evolver._cleanup_problem_def()

    def test_cleanup_removes_module(self):
        evolver = AgentEvolver()
        evolver._inject_problem_def(_DummyProblem())
        evolver._cleanup_problem_def()
        assert _PROBLEM_DEF_MODULE not in sys.modules

    def test_config_schema_forwarded(self):
        evolver = AgentEvolver(
            config_schema={"x": "int"},
            example_config={"x": 5},
            constraints_description="x > 0",
        )
        evolver._inject_problem_def(_DummyProblem())
        try:
            mod = sys.modules[_PROBLEM_DEF_MODULE]
            assert mod.config["config_schema"] == {"x": "int"}
            assert mod.config["example_config"] == {"x": 5}
            assert mod.config["constraints_description"] == "x > 0"
        finally:
            evolver._cleanup_problem_def()
