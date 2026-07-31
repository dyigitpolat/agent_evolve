"""The Harness port: the switchable LLM seam the loop depends on.

A harness turns the per-op inputs into **typed** candidates / insights. The loop
never imports a concrete LLM runtime; it talks only to this interface. Concrete
adapters live under ``agent_evolve.integrations``.

Output typing is the headline: candidate ops return ``list[CandidateConfig]``
(structured), insight ops return ``list[str]`` / ``str`` — no JSON-in-a-string.
"""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.harness.directives import DefaultDirectives, Directives

#: The fixed set of LLM operations exposed by every harness implementation.
OP_NAMES: tuple[str, ...] = (
    "generate_initial",
    "regenerate",
    "generate_offspring",
    "regenerate_offspring",
    "failure_insights",
    "constraint_instruction",
    "performance_insights",
)


class HarnessOutputError(RuntimeError):
    """Raised when a harness call yields no usable candidates (triggers loop retry)."""


@dataclass(frozen=True)
class LLMConfig:
    """Model selection and call parameters, identical across harnesses."""

    model: str
    temperature: Optional[float] = None
    retries: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessContext:
    """Per-run context bound into the harness before any call."""

    objectives: Sequence[ObjectiveSpec]
    search_space_desc: str
    candidate_model: Optional[type] = None
    constraints_description: str = ""
    #: Prompt wording; defaults to the generic backbone directives.
    directives: Directives = field(default_factory=DefaultDirectives)


class _AnyCandidate(BaseModel):
    """Permissive fallback candidate schema for problems without a candidate_model."""

    model_config = ConfigDict(extra="allow")


def build_context(ctx: "HarnessContext") -> str:
    """Compose the static per-run context block (objectives/search space + constraints).

    This is shared *data* prep, not prompt wording — each harness composes its own
    prompts. The candidate schema is conveyed by the typed output, not restated here.
    """
    parts = [ctx.search_space_desc]
    if ctx.constraints_description:
        parts.append(f"DOMAIN CONSTRAINTS:\n{ctx.constraints_description}")
    return "\n\n".join(parts)


@runtime_checkable
class Harness(Protocol):
    """The switchable LLM interface the loop depends on."""

    id: str

    def bind(self, ctx: HarnessContext, cfg: LLMConfig) -> None: ...
    def generate_initial(self, n: int) -> List[Dict[str, Any]]: ...
    def regenerate(
        self, failed_str: str, n: int, constraint_instruction: str, performance_insights: str
    ) -> List[Dict[str, Any]]: ...
    def generate_offspring(
        self, pareto_str: str, n: int, constraint_instruction: str, performance_insights: str
    ) -> List[Dict[str, Any]]: ...
    def regenerate_offspring(
        self, failed_str: str, pareto_str: str, n: int,
        constraint_instruction: str, performance_insights: str,
    ) -> List[Dict[str, Any]]: ...
    def failure_insights(self, failed_str: str, n_failed: int) -> List[str]: ...
    def constraint_instruction(self, failed_str: str, previous: Optional[str] = None) -> str: ...
    def performance_insights(
        self, stats_str: str, pareto_str: str, previous: Optional[str] = None
    ) -> str: ...


CallObserver = Callable[[Dict[str, Any]], None]


class HarnessContractError(TypeError):
    """A harness subclass does not satisfy the operations it inherits."""


def _positional_arity(func: Any) -> Optional[tuple]:
    """Return ``(required, maximum_or_None)`` positional arity, ignoring ``self``.

    ``None`` maximum means ``*args``, which accepts anything.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - builtins, C functions
        return None
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    required = 0
    maximum: Optional[int] = 0
    for p in params:
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return (required, None)
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            maximum = (maximum or 0) + 1
            if p.default is inspect.Parameter.empty:
                required += 1
    return (required, maximum)


class HarnessBase(abc.ABC):
    """Shared lifecycle + result normalization for harness adapters.

    **The operations are enforced, not merely declared.** This class used to
    inherit ``abc.ABC`` while marking nothing abstract, so it promised a
    contract and checked none of it: a subclass could omit an operation, or
    define one with the wrong arity, and the only symptom was a ``TypeError``
    from inside the optimization loop after the run had already started. A
    shipped proposer did exactly that, three times.

    ``__init_subclass__`` now checks, at class-definition time, that every
    operation in :data:`OP_NAMES` exists and accepts the number of positional
    arguments the loop passes it. A wrong signature is a failure to import the
    module that declares it, which is when it is cheap to fix.
    """

    id: str = "base"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # An intermediate base may legitimately be partial; it declares so.
        if getattr(cls, "__abstract_harness__", False):
            return
        problems: List[str] = []
        for op in OP_NAMES:
            impl = getattr(cls, op, None)
            if impl is None or not callable(impl):
                problems.append(f"{op}: not implemented")
                continue
            expected = _positional_arity(getattr(Harness, op, None))
            actual = _positional_arity(impl)
            if expected is None or actual is None:
                continue
            need, _ = expected
            takes_min, takes_max = actual
            if takes_max is not None and takes_max < need:
                problems.append(
                    f"{op}: the loop passes {need} positional argument(s), "
                    f"this accepts at most {takes_max}"
                )
            elif takes_min > need:
                problems.append(
                    f"{op}: requires {takes_min} positional argument(s), "
                    f"the loop passes {need}"
                )
        if problems:
            raise HarnessContractError(
                f"{cls.__name__} does not satisfy the Harness contract:\n  "
                + "\n  ".join(problems)
            )

    def __init__(self) -> None:
        self._ctx: Optional[HarnessContext] = None
        self._cfg: Optional[LLMConfig] = None
        self._observer: Optional[CallObserver] = None
        self.context: str = ""
        self.candidate_type: type = _AnyCandidate

    def bind(self, ctx: HarnessContext, cfg: LLMConfig) -> None:
        self._ctx = ctx
        self._cfg = cfg
        self.context = build_context(ctx)
        self.candidate_type = ctx.candidate_model or _AnyCandidate
        self._on_bind(ctx, cfg)

    def set_call_observer(self, observer: Optional[CallObserver]) -> None:
        self._observer = observer

    @property
    def cfg(self) -> LLMConfig:
        if self._cfg is None:
            raise RuntimeError("Harness used before bind()")
        return self._cfg

    @property
    def directives(self) -> Directives:
        if self._ctx is None:
            raise RuntimeError("Harness used before bind()")
        return self._ctx.directives

    # -- subclass hook ---------------------------------------------------
    def _on_bind(self, ctx: HarnessContext, cfg: LLMConfig) -> None:
        """Optional adapter-specific setup."""

    # -- normalization helpers ------------------------------------------
    def _observe(self, op: str) -> None:
        if self._observer is not None:
            self._observer({"kind": "llm_call", "op": op, "harness": self.id,
                            "model": self.cfg.model})

    @staticmethod
    def _to_dicts(objs: Any, op: str) -> List[Dict[str, Any]]:
        if not isinstance(objs, (list, tuple)) or not objs:
            raise HarnessOutputError(f"{op}: model returned no candidates")
        out: List[Dict[str, Any]] = []
        for o in objs:
            if isinstance(o, BaseModel):
                out.append(o.model_dump())
            elif isinstance(o, dict):
                out.append(o)
            else:
                raise HarnessOutputError(f"{op}: unexpected candidate type {type(o).__name__}")
        return out

    @staticmethod
    def _to_str_list(objs: Any) -> List[str]:
        if isinstance(objs, (list, tuple)):
            return [str(x) for x in objs]
        return [str(objs)]
