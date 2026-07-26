"""pydantic-ai harness: typed structured outputs per op (no JSON-string round-trip).

Prompt wording comes from the problem's ``Directives`` (``self.directives``); this
adapter runs ``pydantic_ai.Agent`` with a typed ``output_type`` per op:
``list[CandidateConfig]`` for candidate ops, ``list[str]`` / ``str`` for insights.
"""

from __future__ import annotations

from typing import Any, List, Optional

from agent_evolve.harness.base import HarnessBase, HarnessOutputError


_AGENT_RUN_FAILED = object()


def _attempt_agent_run_sync(agent: Any, instruction: str, settings: dict | None) -> Any:
    """Run the framework behind an exception-state-clearing private boundary."""

    try:
        if settings is not None:
            return agent.run_sync(instruction, model_settings=settings)
        return agent.run_sync(instruction)
    except Exception:
        return _AGENT_RUN_FAILED


class PydanticAIHarness(HarnessBase):
    id = "pydantic_ai"

    def __init__(self) -> None:
        super().__init__()
        self._last_output_type_contract = None

    @property
    def last_output_type_contract(self):
        """The immutable contract bound to the most recent Agent execution."""

        return self._last_output_type_contract

    def _model_settings(self) -> Optional[dict]:
        settings: dict = {}
        if self.cfg.temperature is not None:
            settings["temperature"] = self.cfg.temperature
        # OpenRouter provider routing (pin official first-party provider, no fallbacks)
        # flows through LLMConfig.extra as {"extra_body": {"provider": {...}}}.
        extra_body = (self.cfg.extra or {}).get("extra_body")
        if extra_body:
            settings["extra_body"] = extra_body
        return settings or None

    def _output(self, output_type: Any, instruction: str) -> Any:
        from pydantic_ai import Agent

        from agent_evolve.integrations.pydantic_ai.boundary_codec import (
            build_output_type_contract,
            validate_output_type_contract,
        )

        # The application workflow is the sole retry owner. Framework output
        # retries can issue a different repair prompt while the legacy/event
        # layer still believes it is replaying one request, so they must remain
        # disabled at this boundary. Provider-client transport retries are a
        # separate M1e composition-root gate and are not asserted here.
        # Reject unsupported/deferred output workflows and bind their canonical
        # type/schema identity before any Agent request can execute.
        self._last_output_type_contract = None
        output_contract = build_output_type_contract(output_type)
        agent = Agent(
            self.cfg.model,
            output_type=output_contract.output_type,
            retries=0,
        )
        validate_output_type_contract(
            output_contract,
            compiled_json_schema=agent.output_json_schema(),
        )
        self._last_output_type_contract = output_contract
        settings = self._model_settings()
        res = _attempt_agent_run_sync(agent, instruction, settings)
        if res is _AGENT_RUN_FAILED:
            # Raised outside the private handler so framework ExceptionGroups,
            # provider payloads, and secrets cannot survive as context/cause.
            raise HarnessOutputError("pydantic_ai: agent execution failed") from None
        self._observe_usage(res)
        return res.output

    def _observe_usage(self, res: Any) -> None:
        """Surface per-call token usage to the loop's event observer (best-effort)."""
        if self._observer is None:
            return
        try:
            u = res.usage
            if callable(u):  # older pydantic-ai exposed usage() as a method
                u = u()
        except Exception:
            return
        inp = getattr(u, "input_tokens", None)
        if inp is None:
            inp = getattr(u, "request_tokens", None)
        out = getattr(u, "output_tokens", None)
        if out is None:
            out = getattr(u, "response_tokens", None)
        self._observer({"kind": "llm_usage", "model": self.cfg.model,
                        "input_tokens": inp, "output_tokens": out})

    def _candidates(self, op: str, instruction: str) -> List[dict]:
        # Wrap the typed list in a Proposal model with a thought_process field so
        # the model reasons before committing to factors.
        from pydantic import create_model

        proposal = create_model(
            "Proposal",
            thought_process=(str, ...),
            candidates=(List[self.candidate_type], ...),
        )
        result = self._output(proposal, instruction)
        return self._to_dicts(result.candidates, op)

    # -- ops -------------------------------------------------------------
    def generate_initial(self, n):
        self._observe("generate_initial")
        return self._candidates("generate_initial", self.directives.compose_initial(self.context, n))

    def regenerate(self, failed_str, n, constraint_instruction, performance_insights):
        self._observe("regenerate")
        instr = self.directives.compose_regenerate(
            self.context, failed_str, n, constraint_instruction, performance_insights)
        return self._candidates("regenerate", instr)

    def generate_offspring(self, pareto_str, n, constraint_instruction, performance_insights):
        self._observe("generate_offspring")
        instr = self.directives.compose_offspring(
            self.context, pareto_str, n, constraint_instruction, performance_insights)
        return self._candidates("generate_offspring", instr)

    def regenerate_offspring(self, failed_str, pareto_str, n,
                             constraint_instruction, performance_insights):
        self._observe("regenerate_offspring")
        instr = self.directives.compose_regenerate_offspring(
            self.context, failed_str, pareto_str, n, constraint_instruction, performance_insights)
        return self._candidates("regenerate_offspring", instr)

    def failure_insights(self, failed_str, n_failed):
        self._observe("failure_insights")
        instr = self.directives.compose_failure_insights(self.context, failed_str, n_failed)
        return self._to_str_list(self._output(List[str], instr))

    def constraint_instruction(self, failed_str, previous=None):
        self._observe("constraint_instruction")
        instr = self.directives.compose_constraint_instruction(self.context, failed_str, previous or "")
        return str(self._output(str, instr)).strip()

    def performance_insights(self, stats_str, pareto_str, previous=None):
        self._observe("performance_insights")
        instr = self.directives.compose_performance_insights(
            self.context, stats_str, pareto_str, previous or "")
        return str(self._output(str, instr)).strip()
