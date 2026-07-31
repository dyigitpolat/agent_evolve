"""Record and replay a generative proposer, so its evidence is checkable.

The catalogue path is auditable because the model's whole output is one token
from a sealed list. Generative proposal gives that up on purpose -- the model
authors configurations -- so the audit has to move with it. This wrapper is
where it moves to.

``SealedGenerativeHarness`` sits between the loop and any :class:`Harness` and
does one of two things:

``record``  call the delegate, hash the exact instruction, seal what came back
            (configurations verbatim, guidance text verbatim) into a chained
            journal, and hand the loop the delegate's answer unchanged.
``replay``  serve the sealed answer, after checking that the question matches.
            No provider, no network, no credential. A prompt that has drifted
            from the sealed one is an error, never a live call.

The prompt is hashed rather than stored because it is a *derived* quantity: the
loop composes it from the problem's directives and the campaign's own state, so
a replay that reaches a different hash has already diverged somewhere it can
still be diagnosed. Storing the text as well would make the journal larger and
prove nothing extra.

**Why replay is exact at all.** Nothing here makes a stochastic loop
deterministic. Replay reproduces a run only when everything outside the
provider is already reproducible -- a deterministic evaluator, a fixed seed, and
the same code. That is the standing condition on this project's benchmarks, and
when it does not hold the drift check fails loudly instead of quietly serving
the wrong answer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from agent_evolve.core.problem import ValidationOutcome
from agent_evolve.domain.generative_emission import (
    GENESIS_CALL_SHA256,
    GenerativeEmission,
    GenerativeProposalCall,
    SealedGuidanceCall,
    SealedRunHeader,
    generative_prompt_sha256,
)
from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.harness.base import (
    Harness,
    HarnessBase,
    HarnessContext,
    HarnessOutputError,
    LLMConfig,
)

__all__ = [
    "SealedGenerativeHarness",
    "SealedReplayDriftError",
    "CANDIDATE_OPS",
    "GUIDANCE_OPS",
]

#: Operations whose output is a configuration the model authored. These are the
#: operator under test.
CANDIDATE_OPS = (
    "generate_initial",
    "regenerate",
    "generate_offspring",
    "regenerate_offspring",
)

#: Operations whose output is text that re-enters a later prompt.
GUIDANCE_OPS = (
    "failure_insights",
    "constraint_instruction",
    "performance_insights",
)


class SealedReplayDriftError(RuntimeError):
    """A replayed call is not the call that was sealed.

    Raised rather than falling back to a live call. A silent fallback would let
    a run that claims to be provider-free contact a provider, which is exactly
    the property the seal exists to prove.
    """


def _as_outcome(value: Any) -> ValidationOutcome:
    if isinstance(value, ValidationOutcome):
        return value
    if value is True or value is None:
        return ValidationOutcome(True)
    if value is False:
        return ValidationOutcome(False, "validation", "validate() returned False")
    raise TypeError("validate() must return a ValidationOutcome or a bool")


def _verdict(
    validator: Optional[Callable[[Dict[str, Any]], Any]],
    config: Dict[str, Any],
) -> tuple[bool, str]:
    """Run the problem's own feasibility check, and never let it abort the run."""

    if validator is None:
        return True, ""
    try:
        outcome = _as_outcome(validator(config))
    except ValueError as exc:
        return False, f"ValueError: {exc}" or "ValueError"
    except TypeError as exc:
        return False, f"TypeError: {exc}" or "TypeError"
    if outcome.ok:
        return True, ""
    return False, (outcome.message or outcome.failure_phase or "rejected by validate()")


class SealedGenerativeHarness(HarnessBase):
    """Seal a generative proposer's decisions by content, not by menu index.

    *delegate* is the proposer actually being audited; in ``replay`` mode it is
    never called and may be omitted entirely.

    *validator* should be the problem's ``validate``. Its verdict is sealed
    alongside each emission so the journal records what the loop was told, and a
    replay re-runs it and requires agreement -- feasibility certification
    survives the loss of the catalogue.

    *candidate_schema_sha256* pins the support the emission was drawn from. It
    is the field that makes a matched null checkable: a null sampling a
    different schema is a support mismatch, and this is where that becomes
    visible rather than assumed.
    """

    id = "sealed_generative"

    def __init__(
        self,
        delegate: Optional[Harness] = None,
        *,
        candidate_schema_sha256: str,
        mode: str = "record",
        validator: Optional[Callable[[Dict[str, Any]], Any]] = None,
        sealed_calls: Sequence[Any] = (),
        on_seal: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__()
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        if mode == "record" and delegate is None:
            raise ValueError("recording requires a delegate proposer to record")
        self._delegate = delegate
        self._mode = mode
        self._validator = validator
        self._schema_sha256 = candidate_schema_sha256
        self._on_seal = on_seal
        self._sealed: List[Any] = list(sealed_calls)
        self._calls: List[Any] = []
        self._previous = GENESIS_CALL_SHA256
        self._cursor = 0

    # -- lifecycle --------------------------------------------------------

    def _on_bind(self, ctx: HarnessContext, cfg: LLMConfig) -> None:
        if self._delegate is not None:
            self._delegate.bind(ctx, cfg)
        if self._mode == "record" and not self._calls:
            # The header opens the chain, before any question is asked, because
            # it declares which questions this proposer will be asked at all.
            self._append(
                SealedRunHeader(
                    proposer_id=str(getattr(self._delegate, "id", "unknown")),
                    requested_model=cfg.model,
                    candidate_schema_sha256=self._schema_sha256,
                    provides_insights=bool(
                        getattr(self._delegate, "provides_insights", True)
                    ),
                )
            )
        elif self._mode == "replay":
            if not self._sealed or type(self._sealed[0]) is not SealedRunHeader:
                raise SealedReplayDriftError(
                    "the sealed journal has no run header, so the recorded "
                    "proposer's declarations cannot be recovered"
                )
            header = self._sealed[0]
            if header.candidate_schema_sha256 != self._schema_sha256:
                raise SealedReplayDriftError(
                    "the sealed run was recorded against a different candidate "
                    "schema than the one now bound"
                )
            self._cursor = 1
            self._append(header)

    def set_call_observer(self, observer) -> None:  # noqa: ANN001 - port signature
        super().set_call_observer(observer)
        setter = getattr(self._delegate, "set_call_observer", None)
        if setter is not None:
            setter(observer)

    @property
    def calls(self) -> tuple:
        """The chained journal this run produced, in issue order."""

        return tuple(self._calls)

    @property
    def terminal_sha256(self) -> str:
        """The digest that closes the chain. Publishing it dates the evidence."""

        return self._previous

    @property
    def provides_insights(self) -> bool:
        """Whether the loop should ask this proposer for guidance at all.

        In replay the answer is *read from the sealed run header*, not guessed
        and not inferred. The loop skips guidance calls for a proposer that
        declares it makes none -- an uninformed baseline is exactly that case --
        so a replay that assumed ``True`` because the delegate is absent issues a
        call the recording never made, and then reports a drift it invented
        itself. Inferring it from which guidance calls appear does not work
        either: a proposer that declines *failure* insights is still asked for
        the constraint guide, so the journal holds guidance calls in both cases.
        """

        if self._mode == "replay":
            return bool(self._sealed[0].provides_insights)
        return bool(getattr(self._delegate, "provides_insights", True))

    # -- the seven operations ---------------------------------------------

    def generate_initial(self, n: int) -> List[Dict[str, Any]]:
        return self._candidates(
            "generate_initial",
            self.directives.compose_initial(self.context, n),
            lambda: self._delegate.generate_initial(n),
        )

    def regenerate(
        self,
        failed_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._candidates(
            "regenerate",
            self.directives.compose_regenerate(
                self.context, failed_str, n, constraint_instruction, performance_insights
            ),
            lambda: self._delegate.regenerate(
                failed_str, n, constraint_instruction, performance_insights
            ),
        )

    def generate_offspring(
        self,
        pareto_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._candidates(
            "generate_offspring",
            self.directives.compose_offspring(
                self.context, pareto_str, n, constraint_instruction, performance_insights
            ),
            lambda: self._delegate.generate_offspring(
                pareto_str, n, constraint_instruction, performance_insights
            ),
        )

    def regenerate_offspring(
        self,
        failed_str: str,
        pareto_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._candidates(
            "regenerate_offspring",
            self.directives.compose_regenerate_offspring(
                self.context,
                failed_str,
                pareto_str,
                n,
                constraint_instruction,
                performance_insights,
            ),
            lambda: self._delegate.regenerate_offspring(
                failed_str, pareto_str, n, constraint_instruction, performance_insights
            ),
        )

    def failure_insights(self, failed_str: str, n_failed: int) -> List[str]:
        outputs = self._guidance(
            "failure_insights",
            self.directives.compose_failure_insights(self.context, failed_str, n_failed),
            lambda: self._delegate.failure_insights(failed_str, n_failed),
        )
        return list(outputs)

    def constraint_instruction(
        self, failed_str: str, previous: Optional[str] = None
    ) -> str:
        outputs = self._guidance(
            "constraint_instruction",
            self.directives.compose_constraint_instruction(
                self.context, failed_str, previous or ""
            ),
            lambda: self._delegate.constraint_instruction(failed_str, previous),
        )
        return outputs[0] if outputs else ""

    def performance_insights(
        self, stats_str: str, pareto_str: str, previous: Optional[str] = None
    ) -> str:
        outputs = self._guidance(
            "performance_insights",
            self.directives.compose_performance_insights(
                self.context, stats_str, pareto_str, previous or ""
            ),
            lambda: self._delegate.performance_insights(stats_str, pareto_str, previous),
        )
        return outputs[0] if outputs else ""

    # -- sealing ----------------------------------------------------------

    def _next_sealed(self, op: str, prompt_sha256: str, expected: type) -> Any:
        if self._cursor >= len(self._sealed):
            raise SealedReplayDriftError(
                f"{op}: the sealed journal is exhausted at call "
                f"{self._cursor}. The replayed run asked more of the proposer "
                "than the recorded one did, so it is a different run."
            )
        call = self._sealed[self._cursor]
        self._cursor += 1
        if type(call) is not expected:
            raise SealedReplayDriftError(
                f"call {call.call_ordinal}: sealed as {type(call).__name__}, "
                f"replayed as {expected.__name__}"
            )
        if call.op != op:
            raise SealedReplayDriftError(
                f"call {call.call_ordinal}: sealed op {call.op!r}, replayed op {op!r}"
            )
        if call.prompt_sha256 != prompt_sha256:
            raise SealedReplayDriftError(
                f"call {call.call_ordinal} ({op}): the prompt is not the sealed "
                "prompt. Replay reconstructs the question before it trusts the "
                "answer, and this question differs."
            )
        return call

    def _append(self, call: Any) -> None:
        self._calls.append(call)
        self._previous = call.identity_sha256
        if self._on_seal is not None:
            self._on_seal(call.to_record())

    def _candidates(
        self,
        op: str,
        instruction: str,
        live: Callable[[], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        prompt_sha256 = generative_prompt_sha256(instruction)
        ordinal = len(self._calls)

        if self._mode == "replay":
            sealed = self._next_sealed(op, prompt_sha256, GenerativeProposalCall)
            if sealed.candidate_schema_sha256 != self._schema_sha256:
                raise SealedReplayDriftError(
                    f"call {sealed.call_ordinal}: the emission was drawn from a "
                    "different candidate schema than the one now bound. The "
                    "support moved, so the decision is not the same decision."
                )
            configs = [thaw_json(c) for c in (e.configuration for e in sealed.emissions)]
            self._recheck(sealed, configs)
            call = GenerativeProposalCall(
                call_ordinal=ordinal,
                op=op,
                requested_model=sealed.requested_model,
                prompt_sha256=prompt_sha256,
                candidate_schema_sha256=self._schema_sha256,
                emissions=sealed.emissions,
                previous_call_sha256=self._previous,
            )
            self._append(call)
            return [dict(c) for c in configs]

        configs = live()
        if not isinstance(configs, list) or not configs:
            # An empty answer is a failed call, and the loop already retries it.
            # Sealing it as a success would put a call in the record that
            # produced nothing, which is the shape of fabricated telemetry.
            raise HarnessOutputError(f"{op}: proposer returned no candidates")
        emissions = []
        for config in configs:
            accepted, reason = _verdict(self._validator, config)
            emissions.append(
                GenerativeEmission(
                    configuration=freeze_json(dict(config)),
                    accepted=accepted,
                    rejection_reason=reason,
                )
            )
        call = GenerativeProposalCall(
            call_ordinal=ordinal,
            op=op,
            requested_model=self.cfg.model,
            prompt_sha256=prompt_sha256,
            candidate_schema_sha256=self._schema_sha256,
            emissions=tuple(emissions),
            previous_call_sha256=self._previous,
        )
        self._append(call)
        return configs

    def _recheck(self, sealed: GenerativeProposalCall, configs: List[Any]) -> None:
        """Re-run the feasibility check and require the sealed verdict back.

        This is what replaces the catalogue's enumeration guarantee. The
        catalogue could promise every option was constructible because it built
        them; a generative seal promises only that the emission still earns the
        verdict the run acted on -- and it proves that by asking again.
        """

        if self._validator is None:
            return
        for emission, config in zip(sealed.emissions, configs):
            accepted, reason = _verdict(self._validator, config)
            if accepted != emission.accepted:
                raise SealedReplayDriftError(
                    f"call {sealed.call_ordinal}: a sealed emission validated as "
                    f"{'valid' if emission.accepted else 'invalid'} and now "
                    f"validates as {'valid' if accepted else 'invalid'}. The "
                    "problem's feasibility rule changed under the seal."
                )
            if not accepted and reason != emission.rejection_reason:
                raise SealedReplayDriftError(
                    f"call {sealed.call_ordinal}: the rejection the proposer was "
                    "shown is not the rejection it would be shown now."
                )

    def _guidance(
        self,
        op: str,
        instruction: str,
        live: Callable[[], Any],
    ) -> tuple:
        prompt_sha256 = generative_prompt_sha256(instruction)
        ordinal = len(self._calls)

        if self._mode == "replay":
            sealed = self._next_sealed(op, prompt_sha256, SealedGuidanceCall)
            call = SealedGuidanceCall(
                call_ordinal=ordinal,
                op=op,
                requested_model=sealed.requested_model,
                prompt_sha256=prompt_sha256,
                outputs=sealed.outputs,
                previous_call_sha256=self._previous,
            )
            self._append(call)
            return sealed.outputs

        raw = live()
        outputs = tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else (str(raw),)
        call = SealedGuidanceCall(
            call_ordinal=ordinal,
            op=op,
            requested_model=self.cfg.model,
            prompt_sha256=prompt_sha256,
            outputs=outputs,
            previous_call_sha256=self._previous,
        )
        self._append(call)
        return outputs
