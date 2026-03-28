"""Thin Python wrapper around evolve.kedi for programmatic usage."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from agent_evolve.problem import Problem
from agent_evolve.results import SearchResult

ConfigT = TypeVar("ConfigT")

_EVOLVE_KEDI = Path(__file__).parent / "evolve.kedi"
_PROBLEM_DEF_MODULE = "problem_def"


@dataclass
class AgentEvolver:
    """LLM-driven multi-objective optimiser.

    Compiles ``evolve.kedi``, injects the user's problem as a synthetic
    ``problem_def`` module, and runs the kedi program to completion.
    The kedi program defines LLM prompt procedures and hands them to
    the Python loop in ``agent_evolve.loop``.
    """

    model: str = "openai:gpt-4o"
    adapter_type: str = "pydantic"

    pop_size: int = 8
    generations: int = 5
    candidates_per_batch: int = 5
    max_regen_rounds: int = 10
    max_failed_examples: int = 5
    verbose: bool = True

    config_schema: Optional[Dict[str, Any]] = None
    example_config: Optional[Dict[str, Any]] = None
    constraints_description: Optional[str] = None

    def optimize(self, problem: Problem[ConfigT]) -> SearchResult[ConfigT]:
        from kedi.lang import compile_program, parse_program

        adapter = self._create_adapter()
        self._inject_problem_def(problem)

        try:
            source = _EVOLVE_KEDI.read_text(encoding="utf-8")
            prog = parse_program(source)
            rt = compile_program(prog, adapter=adapter)
            rt.run_main()
            return sys.modules[_PROBLEM_DEF_MODULE]._result
        finally:
            self._cleanup_problem_def()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_adapter(self) -> Any:
        if self.adapter_type == "dspy":
            from kedi.agent_adapter.adapters import DSPyAdapter
            return DSPyAdapter(model=self.model)
        from kedi.agent_adapter.adapters import PydanticAdapter
        return PydanticAdapter(model=self.model, retries=3)

    def _inject_problem_def(self, problem: Any) -> None:
        """Create a synthetic ``problem_def`` module in ``sys.modules``."""
        mod = types.ModuleType(_PROBLEM_DEF_MODULE)
        mod.problem = problem  # type: ignore[attr-defined]
        mod.config = {  # type: ignore[attr-defined]
            "pop_size": self.pop_size,
            "generations": self.generations,
            "candidates_per_batch": self.candidates_per_batch,
            "max_regen_rounds": self.max_regen_rounds,
            "max_failed_examples": self.max_failed_examples,
            "verbose": self.verbose,
            "config_schema": self.config_schema,
            "example_config": self.example_config,
            "constraints_description": self.constraints_description,
        }
        sys.modules[_PROBLEM_DEF_MODULE] = mod

    @staticmethod
    def _cleanup_problem_def() -> None:
        sys.modules.pop(_PROBLEM_DEF_MODULE, None)
