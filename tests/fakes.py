"""Fake problem and harness for offline loop tests."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.harness.base import HarnessBase
from agent_evolve.integrations.pydantic_ai.harness import PydanticAIHarness
from _canned import CandidateFactory


class SimpleCandidate(BaseModel):
    a: int
    b: int


class SimpleProblem:
    """Two-objective toy problem over ``{"a": int, "b": int}``.

    Maximise ``score`` (= a), minimise ``penalty`` (= b); feasible iff
    ``0 <= a,b <= 10`` and ``a + b <= 15``. Deterministic and LLM-free.
    """

    candidate_model = SimpleCandidate

    @property
    def objectives(self) -> List[ObjectiveSpec]:
        return [ObjectiveSpec("score", "max"), ObjectiveSpec("penalty", "min")]

    def validate_detailed(self, config: Dict[str, Any]) -> ValidationOutcome:
        a, b = config.get("a"), config.get("b")
        if not isinstance(a, int) or not isinstance(b, int):
            return ValidationOutcome(False, "structural", "'a' and 'b' must be integers")
        if not (0 <= a <= 10 and 0 <= b <= 10):
            return ValidationOutcome(False, "constraint", "'a' and 'b' must be in 0..10")
        if a + b > 15:
            return ValidationOutcome(False, "constraint", "a + b must be <= 15")
        return ValidationOutcome(True)

    def evaluate(self, config: Dict[str, Any]) -> Dict[str, float]:
        return {"score": float(config["a"]), "penalty": float(config["b"])}

    def search_space_description(self) -> str:
        return "Pick integers a,b in 0..10 with a+b<=15."


def default_factory(i: int) -> Dict[str, Any]:
    """Deterministic candidates; every 4th is infeasible to exercise the failure path."""
    if i % 4 == 3:
        return {"a": 99, "b": 99}
    return {"a": i % 8, "b": (i + 1) % 6}


class FakeHarness(HarnessBase):
    """Harness that returns canned typed outputs without any LLM."""

    id = "fake"

    def __init__(self, candidate_factory: CandidateFactory = default_factory) -> None:
        super().__init__()
        self._factory = candidate_factory
        self.calls: List[str] = []

    def _batch(self, op: str, n: int) -> List[Dict[str, Any]]:
        self.calls.append(op)
        return [self._factory(i) for i in range(n)]

    def generate_initial(self, n):
        return self._batch("generate_initial", n)

    def regenerate(self, failed_str, n, constraint_instruction, performance_insights):
        return self._batch("regenerate", n)

    def generate_offspring(self, pareto_str, n, constraint_instruction, performance_insights):
        return self._batch("generate_offspring", n)

    def regenerate_offspring(self, failed_str, pareto_str, n, constraint_instruction, performance_insights):
        return self._batch("regenerate_offspring", n)

    def failure_insights(self, failed_str, n_failed):
        self.calls.append("failure_insights")
        return [f"insight {i}" for i in range(n_failed)]

    def constraint_instruction(self, failed_str, previous: Optional[str] = None):
        self.calls.append("constraint_instruction")
        return "Follow the learned constraints."

    def performance_insights(self, stats_str, pareto_str, previous: Optional[str] = None):
        self.calls.append("performance_insights")
        return "Prefer low-cost, high-value configs."


class CannedPydanticAIHarness(PydanticAIHarness):
    """Exercise pydantic-ai's typed-output path without making a provider call."""

    def __init__(self, candidate_factory: CandidateFactory = default_factory) -> None:
        super().__init__()
        self._factory = candidate_factory

    @staticmethod
    def _requested_count(instruction: str) -> int:
        matches = re.findall(r"exactly\s+(\d+)", instruction, flags=re.IGNORECASE)
        if not matches:
            raise AssertionError(f"Canned instruction has no exact output count: {instruction!r}")
        return int(matches[-1])

    def _output(self, output_type: Any, instruction: str) -> Any:
        fields = getattr(output_type, "model_fields", {})
        if "candidates" in fields:
            count = self._requested_count(instruction)
            return output_type(
                thought_process="Deterministic offline test response.",
                candidates=[self._factory(i) for i in range(count)],
            )
        if output_type == List[str]:
            count = self._requested_count(instruction)
            return [f"insight {i}" for i in range(count)]
        if output_type is str:
            return "Deterministic offline guidance."
        raise AssertionError(f"Unsupported canned output type: {output_type!r}")
