"""Workload-neutral preparation boundary for multi-generation campaigns.

This module deliberately stops before live evolutionary execution.  It seals
the shared chronology, validates budgets and concurrency against a benchmark
preflight, loads an exact seed batch, asks an injected agent runtime to
preflight the composition, and journals the resulting immutable preparation
record.  Existing portfolio-evolution and recombination services remain the
authorities for executing their respective waves.

Benchmark adapters enter only through inverted ports.  Model and provider
configuration belongs to ``CampaignAgentRuntimePort`` implementations and is
therefore absent from campaign protocols and workload ports.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.finite_variation_eligibility import (
    FiniteVariationEligibilityReceipt,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


MIN_CAMPAIGN_GENERATIONS = 3
MAX_CAMPAIGN_GENERATIONS = 24

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_PROTOCOL_DOMAIN = b"agent-evolve:evolution-campaign-protocol:v1\x00"
_SCHEDULE_DOMAIN = b"agent-evolve:evolution-campaign-schedule:v1\x00"
_SESSION_REQUEST_DOMAIN = b"agent-evolve:campaign-session-request:v1\x00"
_SESSION_DOMAIN = b"agent-evolve:campaign-benchmark-session:v1\x00"
_SEED_DOMAIN = b"agent-evolve:campaign-seed:v1\x00"
_SEED_BATCH_DOMAIN = b"agent-evolve:campaign-seed-batch:v1\x00"
_ARCHIVE_UTILITY_SNAPSHOT_DOMAIN = (
    b"agent-evolve:campaign-archive-utility-snapshot:v1\x00"
)
_POLICIES_DOMAIN = b"agent-evolve:campaign-policies:v1\x00"
_REFLECTION_SUPERVISION_CONFIGURATION_DOMAIN = (
    b"agent-evolve:campaign-reflection-supervision-configuration:v1\x00"
)
_WORKLOAD_PORTS_DOMAIN = b"agent-evolve:campaign-workload-ports:v1\x00"
_CONCURRENCY_DOMAIN = b"agent-evolve:campaign-concurrency:v1\x00"
_RUNTIME_REQUEST_DOMAIN = b"agent-evolve:campaign-runtime-request:v1\x00"
_RUNTIME_RECEIPT_DOMAIN = b"agent-evolve:campaign-runtime-receipt:v1\x00"
_PREPARATION_DOMAIN = b"agent-evolve:campaign-preparation:v1\x00"

ALTERNATING_CADENCE_POLICY_ID = "alternating_portfolio_recombination"
ALTERNATING_CADENCE_POLICY_VERSION = 3
ALTERNATING_CADENCE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:alternating-portfolio-recombination:v3;"
    b"actual-generations-not-cycles;portfolio-first;recombination-second;"
    b"odd-terminal-portfolio-permitted;recombination-source-is-prior-portfolio;"
    b"protocol-bound-terminal-reflection-consumer-requirement;"
    b"future-consumer-requires-complete-scheduled-admission-barrier"
).hexdigest()

SEALED_CUTOFF_DELAYED_CADENCE_POLICY_ID = (
    "sealed_cutoff_delayed_portfolio_recombination"
)
SEALED_CUTOFF_DELAYED_CADENCE_POLICY_VERSION = 1
SEALED_CUTOFF_DELAYED_CADENCE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sealed-cutoff-delayed-portfolio-recombination:v1;"
    b"actual-generations-not-cycles;portfolio-first;recombination-second;"
    b"reflection-input-is-exact-sealed-source-stage;"
    b"one-complete-pair-delay;admit-after-next-recombination-seal;"
    b"first-consumer-is-following-portfolio;future-content-not-visible-to-reflector;"
    b"no-online-reflection-without-scheduled-future-portfolio-consumer"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, record: object) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _validate_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed campaign-token grammar")


def _validate_frozen_object(value: object, *, name: str) -> FrozenJsonObject:
    if type(value) is not FrozenJsonObject:
        raise TypeError(f"{name} must be an exact FrozenJsonObject")
    if freeze_json(value) is not value:
        raise TypeError(f"{name} must already be frozen typed JSON")
    return value


class TerminalReflectionPolicy(str, Enum):
    """Whether online reflection may run without a later portfolio consumer."""

    ALLOW_TERMINAL = "allow_terminal"
    REQUIRE_FUTURE_PORTFOLIO_CONSUMER = "require_future_portfolio_consumer"


@dataclass(frozen=True, slots=True)
class CampaignProtocol:
    """Provider- and workload-free behavioral protocol for one campaign.

    ``generation_count`` is the number of actual evolutionary generations,
    not a number of portfolio/recombination cycles.  Consequently a count of
    three means portfolio, recombination, portfolio.
    """

    protocol_id: str
    protocol_version: int
    definition_sha256: str
    outer_seed: int
    generation_count: int
    required_seed_count: int
    parents_per_portfolio_generation: int
    portfolio_width: int
    recombinations_per_parent: int
    reflections_per_recombination_generation: int = 1
    reflection_promotion_block_pairs: int = 3
    terminal_reflection_policy: TerminalReflectionPolicy = (
        TerminalReflectionPolicy.ALLOW_TERMINAL
    )

    def __post_init__(self) -> None:
        _validate_token(self.protocol_id, name="protocol_id")
        if type(self.protocol_version) is not int or self.protocol_version <= 0:
            raise ValueError("protocol_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")
        if type(self.outer_seed) is not int or not -(1 << 127) <= self.outer_seed < (
            1 << 127
        ):
            raise ValueError("outer_seed must be an exact signed int128")
        if type(self.generation_count) is not int or not (
            MIN_CAMPAIGN_GENERATIONS
            <= self.generation_count
            <= MAX_CAMPAIGN_GENERATIONS
        ):
            raise ValueError(
                "generation_count must count 3 through 24 actual generations"
            )
        for name in (
            "required_seed_count",
            "parents_per_portfolio_generation",
            "recombinations_per_parent",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.portfolio_width) is not int or self.portfolio_width < 2:
            raise ValueError("portfolio_width must be an exact integer of at least two")
        if (
            type(self.reflections_per_recombination_generation) is not int
            or self.reflections_per_recombination_generation < 0
        ):
            raise ValueError(
                "reflections_per_recombination_generation must be a "
                "non-negative exact integer"
            )
        if (
            type(self.reflection_promotion_block_pairs) is not int
            or self.reflection_promotion_block_pairs <= 0
        ):
            raise ValueError(
                "reflection_promotion_block_pairs must be a positive exact integer"
            )
        if type(self.terminal_reflection_policy) is not TerminalReflectionPolicy:
            raise TypeError(
                "terminal_reflection_policy must be a TerminalReflectionPolicy"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "definition_sha256": self.definition_sha256,
            "outer_seed": self.outer_seed,
            "generation_count": self.generation_count,
            "required_seed_count": self.required_seed_count,
            "parents_per_portfolio_generation": (self.parents_per_portfolio_generation),
            "portfolio_width": self.portfolio_width,
            "recombinations_per_parent": self.recombinations_per_parent,
            "reflections_per_recombination_generation": (
                self.reflections_per_recombination_generation
            ),
            "reflection_promotion_block_pairs": (self.reflection_promotion_block_pairs),
            "terminal_reflection_policy": self.terminal_reflection_policy.value,
        }

    @property
    def protocol_sha256(self) -> str:
        return _hash(_PROTOCOL_DOMAIN, self.to_record())


class CampaignGenerationKind(str, Enum):
    PORTFOLIO = "portfolio"
    RECOMBINATION = "recombination"


@dataclass(frozen=True, slots=True)
class CampaignGenerationStep:
    """One exact generation in a campaign cadence."""

    generation: int
    kind: CampaignGenerationKind
    source_portfolio_generation: int | None
    parent_count: int
    offspring_per_parent: int
    planned_agent_calls: int

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.kind) is not CampaignGenerationKind:
            raise TypeError("kind must be a CampaignGenerationKind")
        for name in ("parent_count", "offspring_per_parent"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.planned_agent_calls) is not int or self.planned_agent_calls < 0:
            raise ValueError("planned_agent_calls must be non-negative")
        if self.kind is CampaignGenerationKind.PORTFOLIO:
            if self.source_portfolio_generation is not None:
                raise ValueError(
                    "a portfolio generation cannot name a source portfolio"
                )
            if self.planned_agent_calls != self.parent_count:
                raise ValueError(
                    "a portfolio generation requires one agent call per parent"
                )
        else:
            if (
                type(self.source_portfolio_generation) is not int
                or self.source_portfolio_generation <= 0
                or self.source_portfolio_generation >= self.generation
            ):
                raise ValueError(
                    "a recombination generation must name an earlier portfolio"
                )

    @property
    def planned_candidate_evaluations(self) -> int:
        return self.parent_count * self.offspring_per_parent

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "generation": self.generation,
            "kind": self.kind.value,
            "source_portfolio_generation": self.source_portfolio_generation,
            "parent_count": self.parent_count,
            "offspring_per_parent": self.offspring_per_parent,
            "planned_candidate_evaluations": self.planned_candidate_evaluations,
            "planned_agent_calls": self.planned_agent_calls,
        }


@dataclass(frozen=True, slots=True)
class CampaignGenerationPair:
    """One portfolio generation and its immediately paired recombination."""

    portfolio_generation: int
    recombination_generation: int

    def __post_init__(self) -> None:
        if type(self.portfolio_generation) is not int or self.portfolio_generation <= 0:
            raise ValueError("portfolio_generation must be positive")
        if self.recombination_generation != self.portfolio_generation + 1:
            raise ValueError("recombination must immediately follow its portfolio")

    def to_record(self) -> dict[str, int]:
        self.__post_init__()
        return {
            "portfolio_generation": self.portfolio_generation,
            "recombination_generation": self.recombination_generation,
        }


class ReflectionLaunchMode(str, Enum):
    """When a reflection may enter the agent task queue."""

    ASYNC_AFTER_STAGE_SEAL = "async_after_stage_seal"


class ReflectionVisibility(str, Enum):
    """Visibility law for evidence produced by a reflection wave."""

    QUARANTINED_UNTIL_BLOCK_CLOSE = "quarantined_until_block_close"


class ReflectionFailureMode(str, Enum):
    """When a settled reflection failure changes campaign execution."""

    FAIL_AT_NEXT_STAGE_BOUNDARY = "fail_at_next_stage_boundary"
    COLLECT_ALL_AT_BARRIER_THEN_FAIL = "collect_all_at_barrier_then_fail"
    BEST_EFFORT_DEGRADED = "best_effort_degraded"


@dataclass(frozen=True, slots=True)
class CampaignReflectionSupervisionPolicy:
    """Failure supervision independent from reflection-content visibility.

    A visibility barrier controls when successful quarantined content may enter
    later testing.  It must not implicitly define when background-task failures
    become observable or how sibling tasks are drained.
    """

    mode: ReflectionFailureMode = ReflectionFailureMode.COLLECT_ALL_AT_BARRIER_THEN_FAIL
    policy_id: str = field(
        init=False,
        default="campaign_reflection_supervision",
    )
    policy_version: int = field(init=False, default=1)
    definition_sha256: str = field(
        init=False,
        default=hashlib.sha256(
            b"agent-evolve:campaign-reflection-supervision:v1;"
            b"visibility-independent=true;settlement-durable=true;"
            b"safe-fail-checkpoint=after-stage-seal-before-next-stage;"
            b"barrier-drain=all-launched;exception-short-circuit=false;"
            b"failed-content-admission=false;partial-block-admission=false;"
            b"abort-cancellation=typed-and-drained;"
            b"modes=fail-at-next-stage-boundary,collect-all-at-barrier-then-fail,"
            b"best-effort-degraded"
        ).hexdigest(),
    )

    def __post_init__(self) -> None:
        if type(self.mode) is not ReflectionFailureMode:
            raise TypeError("mode must be an exact ReflectionFailureMode")
        _validate_token(self.policy_id, name="reflection_supervision.policy_id")
        if type(self.policy_version) is not int or self.policy_version != 1:
            raise ValueError("unsupported reflection supervision policy version")
        require_sha256(
            self.definition_sha256,
            "reflection_supervision.definition_sha256",
        )

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _REFLECTION_SUPERVISION_CONFIGURATION_DOMAIN,
            {"mode": self.mode.value},
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "mode": self.mode.value,
            "configuration_sha256": self.configuration_sha256,
            "visibility_independent": True,
            "safe_failure_checkpoint": ("after_stage_seal_before_next_archive_cutoff"),
            "barrier_drains_all_launched_reflections": True,
            "failed_or_partial_block_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class CampaignReflectionWave:
    """Nonblocking reflection work and its prospective visibility barrier."""

    source_generation: int
    call_count: int
    launch_mode: ReflectionLaunchMode
    visibility: ReflectionVisibility
    promotion_barrier_generation: int | None

    def __post_init__(self) -> None:
        if type(self.source_generation) is not int or self.source_generation <= 0:
            raise ValueError("reflection source_generation must be positive")
        if type(self.call_count) is not int or self.call_count <= 0:
            raise ValueError("reflection call_count must be positive")
        if type(self.launch_mode) is not ReflectionLaunchMode:
            raise TypeError("launch_mode must be a ReflectionLaunchMode")
        if type(self.visibility) is not ReflectionVisibility:
            raise TypeError("visibility must be a ReflectionVisibility")
        barrier = self.promotion_barrier_generation
        if barrier is not None and (
            type(barrier) is not int or barrier < self.source_generation
        ):
            raise ValueError("promotion barrier cannot precede reflection evidence")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "source_generation": self.source_generation,
            "call_count": self.call_count,
            "launch_mode": self.launch_mode.value,
            "visibility": self.visibility.value,
            "promotion_barrier_generation": self.promotion_barrier_generation,
        }


@dataclass(frozen=True, slots=True)
class CampaignPromotionBarrier:
    """Stage seal after which quarantined reflections may enter testing.

    The legacy cohort cadence closes a barrier at its last reflection source.
    A delayed cadence may close it at a later sealed stage so provider work can
    overlap intervening generations.  In both cases the barrier is prospective:
    admitted evidence can first appear in the *following* stage request.
    """

    generation: int
    reflection_source_generations: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("promotion barrier generation must be positive")
        if (
            type(self.reflection_source_generations) is not tuple
            or not self.reflection_source_generations
            or any(
                type(value) is not int or value <= 0
                for value in self.reflection_source_generations
            )
        ):
            raise ValueError("promotion barrier requires exact source generations")
        if self.reflection_source_generations != tuple(
            sorted(set(self.reflection_source_generations))
        ):
            raise ValueError("reflection source generations must be canonical")
        if self.reflection_source_generations[-1] > self.generation:
            raise ValueError("promotion barrier cannot precede its last source")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "generation": self.generation,
            "reflection_source_generations": list(self.reflection_source_generations),
        }


@dataclass(frozen=True, slots=True)
class CampaignSchedule:
    """Authenticated operator chronology derived from a campaign protocol."""

    protocol_sha256: str
    cadence_policy_id: str
    cadence_policy_version: int
    cadence_definition_sha256: str
    steps: tuple[CampaignGenerationStep, ...]
    recombination_pairs: tuple[CampaignGenerationPair, ...]
    reflection_waves: tuple[CampaignReflectionWave, ...]
    promotion_barriers: tuple[CampaignPromotionBarrier, ...]

    def __post_init__(self) -> None:
        require_sha256(self.protocol_sha256, "protocol_sha256")
        _validate_token(self.cadence_policy_id, name="cadence_policy_id")
        if (
            type(self.cadence_policy_version) is not int
            or self.cadence_policy_version <= 0
        ):
            raise ValueError("cadence_policy_version must be positive")
        require_sha256(
            self.cadence_definition_sha256,
            "cadence_definition_sha256",
        )
        if type(self.steps) is not tuple or not self.steps:
            raise ValueError("steps must be a non-empty exact tuple")
        if any(type(step) is not CampaignGenerationStep for step in self.steps):
            raise TypeError("steps must contain exact CampaignGenerationStep values")
        for step in self.steps:
            CampaignGenerationStep.__post_init__(step)
        if tuple(step.generation for step in self.steps) != tuple(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("schedule generations must be contiguous from one")
        expected_kinds = tuple(
            CampaignGenerationKind.PORTFOLIO
            if index % 2 == 1
            else CampaignGenerationKind.RECOMBINATION
            for index in range(1, len(self.steps) + 1)
        )
        if tuple(step.kind for step in self.steps) != expected_kinds:
            raise ValueError("schedule must alternate portfolio then recombination")
        expected_pairs = tuple(
            CampaignGenerationPair(index, index + 1)
            for index in range(1, len(self.steps), 2)
        )
        if self.recombination_pairs != expected_pairs:
            raise ValueError("recombination_pairs differ from the exact chronology")
        for pair in self.recombination_pairs:
            recombination = self.steps[pair.recombination_generation - 1]
            if recombination.source_portfolio_generation != pair.portfolio_generation:
                raise ValueError(
                    "recombination source differs from its paired portfolio"
                )
        if type(self.reflection_waves) is not tuple or any(
            type(value) is not CampaignReflectionWave for value in self.reflection_waves
        ):
            raise TypeError("reflection_waves must contain exact wave values")
        for value in self.reflection_waves:
            CampaignReflectionWave.__post_init__(value)
        reflection_sources = tuple(
            value.source_generation for value in self.reflection_waves
        )
        if reflection_sources != tuple(sorted(set(reflection_sources))):
            raise ValueError("reflection waves must have unique canonical sources")
        recombination_calls = {
            step.generation: step.planned_agent_calls
            for step in self.steps
            if step.kind is CampaignGenerationKind.RECOMBINATION
            and step.planned_agent_calls > 0
        }
        if reflection_sources != tuple(recombination_calls):
            raise ValueError("reflection waves differ from recombination call plans")
        if any(
            value.call_count != recombination_calls[value.source_generation]
            for value in self.reflection_waves
        ):
            raise ValueError("reflection wave call count differs from its stage")
        if type(self.promotion_barriers) is not tuple or any(
            type(value) is not CampaignPromotionBarrier
            for value in self.promotion_barriers
        ):
            raise TypeError("promotion_barriers must contain exact barrier values")
        for value in self.promotion_barriers:
            CampaignPromotionBarrier.__post_init__(value)
            if value.generation > len(self.steps):
                raise ValueError("promotion barrier escapes the campaign horizon")
            barrier_step = self.steps[value.generation - 1]
            if barrier_step.kind is not CampaignGenerationKind.RECOMBINATION:
                raise ValueError("promotion barrier must follow a recombination seal")
        barrier_generations = tuple(
            value.generation for value in self.promotion_barriers
        )
        if barrier_generations != tuple(sorted(set(barrier_generations))):
            raise ValueError("promotion barriers must be unique and canonical")
        waves_by_barrier: dict[int, list[int]] = {}
        for wave in self.reflection_waves:
            barrier = wave.promotion_barrier_generation
            if barrier is not None:
                waves_by_barrier.setdefault(barrier, []).append(wave.source_generation)
        expected_barriers = tuple(
            CampaignPromotionBarrier(generation, tuple(sources))
            for generation, sources in sorted(waves_by_barrier.items())
        )
        if expected_barriers != self.promotion_barriers:
            raise ValueError("promotion barriers differ from reflection visibility")

    @property
    def portfolio_generations(self) -> tuple[int, ...]:
        return tuple(
            step.generation
            for step in self.steps
            if step.kind is CampaignGenerationKind.PORTFOLIO
        )

    @property
    def paired_recombination_generations(self) -> tuple[int, ...]:
        return tuple(pair.recombination_generation for pair in self.recombination_pairs)

    @property
    def unpaired_terminal_portfolio_generation(self) -> int | None:
        terminal = self.steps[-1]
        return (
            terminal.generation
            if terminal.kind is CampaignGenerationKind.PORTFOLIO
            else None
        )

    @property
    def planned_candidate_evaluations(self) -> int:
        return sum(step.planned_candidate_evaluations for step in self.steps)

    @property
    def planned_agent_calls(self) -> int:
        return sum(step.planned_agent_calls for step in self.steps)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "protocol_sha256": self.protocol_sha256,
            "cadence": {
                "policy_id": self.cadence_policy_id,
                "policy_version": self.cadence_policy_version,
                "definition_sha256": self.cadence_definition_sha256,
            },
            "steps": [step.to_record() for step in self.steps],
            "portfolio_generations": list(self.portfolio_generations),
            "recombination_pairs": [
                pair.to_record() for pair in self.recombination_pairs
            ],
            "paired_recombination_generations": list(
                self.paired_recombination_generations
            ),
            "reflection_waves": [value.to_record() for value in self.reflection_waves],
            "promotion_barriers": [
                value.to_record() for value in self.promotion_barriers
            ],
            "unpaired_terminal_portfolio_generation": (
                self.unpaired_terminal_portfolio_generation
            ),
            "planned_candidate_evaluations": self.planned_candidate_evaluations,
            "planned_agent_calls": self.planned_agent_calls,
        }

    @property
    def schedule_sha256(self) -> str:
        return _hash(_SCHEDULE_DOMAIN, self.to_record())


@runtime_checkable
class CampaignCadence(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def build(self, protocol: CampaignProtocol) -> CampaignSchedule: ...


def _build_alternating_schedule(
    protocol: CampaignProtocol,
    *,
    cadence_policy_id: str,
    cadence_policy_version: int,
    cadence_definition_sha256: str,
    reflection_waves: tuple[CampaignReflectionWave, ...],
    promotion_barriers: tuple[CampaignPromotionBarrier, ...],
) -> CampaignSchedule:
    """Materialize common P/R stage accounting around a reflection policy."""

    reflection_generation_set = {
        value.source_generation for value in reflection_waves
    }
    steps = tuple(
        CampaignGenerationStep(
            generation=generation,
            kind=(
                CampaignGenerationKind.PORTFOLIO
                if generation % 2 == 1
                else CampaignGenerationKind.RECOMBINATION
            ),
            source_portfolio_generation=(
                None if generation % 2 == 1 else generation - 1
            ),
            parent_count=protocol.parents_per_portfolio_generation,
            offspring_per_parent=(
                protocol.portfolio_width
                if generation % 2 == 1
                else protocol.recombinations_per_parent
            ),
            planned_agent_calls=(
                protocol.parents_per_portfolio_generation
                if generation % 2 == 1
                else (
                    protocol.reflections_per_recombination_generation
                    if generation in reflection_generation_set
                    else 0
                )
            ),
        )
        for generation in range(1, protocol.generation_count + 1)
    )
    pairs = tuple(
        CampaignGenerationPair(generation, generation + 1)
        for generation in range(1, protocol.generation_count, 2)
    )
    return CampaignSchedule(
        protocol_sha256=protocol.protocol_sha256,
        cadence_policy_id=cadence_policy_id,
        cadence_policy_version=cadence_policy_version,
        cadence_definition_sha256=cadence_definition_sha256,
        steps=steps,
        recombination_pairs=pairs,
        reflection_waves=reflection_waves,
        promotion_barriers=promotion_barriers,
    )


@dataclass(frozen=True, slots=True)
class AlternatingPortfolioRecombinationCadence:
    """Portfolio-first cadence for three through twenty-four generations."""

    policy_id: str = field(init=False, default=ALTERNATING_CADENCE_POLICY_ID)
    policy_version: int = field(
        init=False,
        default=ALTERNATING_CADENCE_POLICY_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=ALTERNATING_CADENCE_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if (
            self.policy_id != ALTERNATING_CADENCE_POLICY_ID
            or self.policy_version != ALTERNATING_CADENCE_POLICY_VERSION
            or self.definition_sha256 != ALTERNATING_CADENCE_DEFINITION_SHA256
        ):
            raise ValueError("alternating cadence policy identity changed")

    def build(self, protocol: CampaignProtocol) -> CampaignSchedule:
        if type(protocol) is not CampaignProtocol:
            raise TypeError("protocol must be an exact CampaignProtocol")
        CampaignProtocol.__post_init__(protocol)
        paired_recombination_generations = tuple(
            range(2, protocol.generation_count + 1, 2)
        )
        candidate_reflection_generations = tuple(
            generation
            for generation in paired_recombination_generations
            if (
                protocol.reflections_per_recombination_generation > 0
                and (
                    protocol.terminal_reflection_policy
                    is TerminalReflectionPolicy.ALLOW_TERMINAL
                    or generation < protocol.generation_count
                )
            )
        )
        block_size = protocol.reflection_promotion_block_pairs
        barrier_by_source: dict[int, int] = {}
        promotion_barriers: list[CampaignPromotionBarrier] = []
        if protocol.reflections_per_recombination_generation > 0:
            for start in range(
                0,
                len(candidate_reflection_generations),
                block_size,
            ):
                block = candidate_reflection_generations[start : start + block_size]
                if len(block) != block_size:
                    continue
                barrier = block[-1]
                promotion_barriers.append(
                    CampaignPromotionBarrier(
                        generation=barrier,
                        reflection_source_generations=block,
                    )
                )
                barrier_by_source.update({source: barrier for source in block})
        reflection_generations = candidate_reflection_generations
        if (
            protocol.terminal_reflection_policy
            is TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ):
            # Merely occurring before a later portfolio is insufficient.  If a
            # cohort is incomplete, no reflection in it can cross a promotion
            # barrier, so none has an actual consumer under this policy.
            reflection_generations = tuple(
                generation
                for generation in reflection_generations
                if generation in barrier_by_source
                and barrier_by_source[generation] + 1
                <= protocol.generation_count
            )
            retained = set(reflection_generations)
            promotion_barriers = [
                barrier
                for barrier in promotion_barriers
                if set(barrier.reflection_source_generations) <= retained
            ]
        reflection_waves = tuple(
            CampaignReflectionWave(
                source_generation=generation,
                call_count=protocol.reflections_per_recombination_generation,
                launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
                visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
                promotion_barrier_generation=barrier_by_source.get(generation),
            )
            for generation in reflection_generations
        )
        return _build_alternating_schedule(
            protocol,
            cadence_policy_id=self.policy_id,
            cadence_policy_version=self.policy_version,
            cadence_definition_sha256=self.definition_sha256,
            reflection_waves=reflection_waves,
            promotion_barriers=tuple(promotion_barriers),
        )


@dataclass(frozen=True, slots=True)
class SealedCutoffDelayedAdmissionCadence:
    """P/R cadence with one-pair-lagged, prospectively useful reflection.

    A reflection sourced from sealed recombination generation ``g`` is allowed
    to run while generations ``g+1`` and ``g+2`` execute.  Its join barrier is
    the seal of recombination generation ``g+2`` and its first possible
    consumer is portfolio generation ``g+3``.  Sources without that exact
    future consumer are never launched.

    This is a separate injected cadence rather than a flag on the legacy
    cadence, preserving the existing default schedule and policy identity.
    """

    policy_id: str = field(
        init=False,
        default=SEALED_CUTOFF_DELAYED_CADENCE_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=SEALED_CUTOFF_DELAYED_CADENCE_POLICY_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=SEALED_CUTOFF_DELAYED_CADENCE_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if (
            self.policy_id != SEALED_CUTOFF_DELAYED_CADENCE_POLICY_ID
            or self.policy_version != SEALED_CUTOFF_DELAYED_CADENCE_POLICY_VERSION
            or self.definition_sha256
            != SEALED_CUTOFF_DELAYED_CADENCE_DEFINITION_SHA256
        ):
            raise ValueError("sealed-cutoff delayed cadence identity changed")

    def build(self, protocol: CampaignProtocol) -> CampaignSchedule:
        if type(protocol) is not CampaignProtocol:
            raise TypeError("protocol must be an exact CampaignProtocol")
        CampaignProtocol.__post_init__(protocol)
        if protocol.terminal_reflection_policy is not (
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ):
            raise ValueError(
                "delayed admission requires the future-consumer terminal policy"
            )
        if protocol.reflection_promotion_block_pairs != 1:
            raise ValueError(
                "delayed admission requires one reflection per admission barrier"
            )

        sources = tuple(
            generation
            for generation in range(2, protocol.generation_count + 1, 2)
            if (
                protocol.reflections_per_recombination_generation > 0
                and generation + 3 <= protocol.generation_count
            )
        )
        barriers = tuple(
            CampaignPromotionBarrier(
                generation=source + 2,
                reflection_source_generations=(source,),
            )
            for source in sources
        )
        waves = tuple(
            CampaignReflectionWave(
                source_generation=source,
                call_count=protocol.reflections_per_recombination_generation,
                launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
                visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
                promotion_barrier_generation=source + 2,
            )
            for source in sources
        )
        return _build_alternating_schedule(
            protocol,
            cadence_policy_id=self.policy_id,
            cadence_policy_version=self.policy_version,
            cadence_definition_sha256=self.definition_sha256,
            reflection_waves=waves,
            promotion_barriers=barriers,
        )


@dataclass(frozen=True, slots=True)
class CampaignConcurrency:
    """Independent evaluator and agent-queue concurrency caps."""

    evaluator_concurrency: int
    agent_concurrency: int
    agent_queue_capacity: int

    def __post_init__(self) -> None:
        for name in ("evaluator_concurrency", "agent_concurrency"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.agent_queue_capacity) is not int
            or self.agent_queue_capacity < self.agent_concurrency
        ):
            raise ValueError("agent_queue_capacity must be at least agent_concurrency")

    def to_record(self) -> dict[str, int]:
        self.__post_init__()
        return {
            "evaluator_concurrency": self.evaluator_concurrency,
            "agent_concurrency": self.agent_concurrency,
            "agent_queue_capacity": self.agent_queue_capacity,
        }

    @property
    def concurrency_sha256(self) -> str:
        return _hash(_CONCURRENCY_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class BenchmarkSessionRequest:
    """Workload-neutral request passed to a benchmark session port."""

    protocol_sha256: str
    budget_sha256: str
    outer_seed: int
    requested_evaluator_concurrency: int

    def __post_init__(self) -> None:
        require_sha256(self.protocol_sha256, "protocol_sha256")
        require_sha256(self.budget_sha256, "budget_sha256")
        if type(self.outer_seed) is not int or not -(1 << 127) <= self.outer_seed < (
            1 << 127
        ):
            raise ValueError("outer_seed must be an exact signed int128")
        if (
            type(self.requested_evaluator_concurrency) is not int
            or self.requested_evaluator_concurrency <= 0
        ):
            raise ValueError("requested evaluator concurrency must be positive")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "protocol_sha256": self.protocol_sha256,
            "budget_sha256": self.budget_sha256,
            "outer_seed": self.outer_seed,
            "requested_evaluator_concurrency": (self.requested_evaluator_concurrency),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_SESSION_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignBenchmarkSession:
    """Benchmark facts and acquired-resource evidence from preflight."""

    request_sha256: str
    benchmark: FrozenJsonObject
    evaluator_concurrency_cap: int
    preflight_receipt: FrozenJsonObject
    resource_lease: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _validate_frozen_object(self.benchmark, name="benchmark")
        _validate_frozen_object(self.preflight_receipt, name="preflight_receipt")
        _validate_frozen_object(self.resource_lease, name="resource_lease")
        if (
            type(self.evaluator_concurrency_cap) is not int
            or self.evaluator_concurrency_cap <= 0
        ):
            raise ValueError("evaluator_concurrency_cap must be positive")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "benchmark": thaw_json(self.benchmark),
            "benchmark_sha256": typed_json_sha256(self.benchmark),
            "evaluator_concurrency_cap": self.evaluator_concurrency_cap,
            "preflight_receipt": thaw_json(self.preflight_receipt),
            "resource_lease": thaw_json(self.resource_lease),
        }

    @property
    def session_sha256(self) -> str:
        return _hash(_SESSION_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignSeed:
    """One labelled, immutable benchmark seed configuration."""

    seed_id: str
    configuration: FrozenJsonObject

    def __post_init__(self) -> None:
        _validate_token(self.seed_id, name="seed_id")
        _validate_frozen_object(self.configuration, name="configuration")

    @property
    def configuration_sha256(self) -> str:
        return typed_json_sha256(self.configuration)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record = {
            "seed_id": self.seed_id,
            "configuration_sha256": self.configuration_sha256,
        }
        return {**record, "seed_sha256": _hash(_SEED_DOMAIN, record)}


@dataclass(frozen=True, slots=True)
class CampaignSeedBatch:
    """Exact seed batch bound to one opened benchmark session."""

    session_sha256: str
    seeds: tuple[CampaignSeed, ...]

    def __post_init__(self) -> None:
        require_sha256(self.session_sha256, "session_sha256")
        if type(self.seeds) is not tuple or not self.seeds:
            raise ValueError("seeds must be a non-empty exact tuple")
        if any(type(seed) is not CampaignSeed for seed in self.seeds):
            raise TypeError("seeds must contain exact CampaignSeed values")
        for seed in self.seeds:
            CampaignSeed.__post_init__(seed)
        if len({seed.seed_id for seed in self.seeds}) != len(self.seeds):
            raise ValueError("seed IDs must be unique")
        if len({seed.configuration_sha256 for seed in self.seeds}) != len(self.seeds):
            raise ValueError("seed configurations must be unique")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "session_sha256": self.session_sha256,
            "seeds": [seed.to_record() for seed in self.seeds],
        }

    @property
    def batch_sha256(self) -> str:
        return _hash(_SEED_BATCH_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class ParentVariationBinding:
    """Catalog result sealed to one benchmark, parent, and novelty cutoff."""

    benchmark_sha256: str
    parent_configuration_sha256: str
    known_phenotype_sha256s: tuple[str, ...]
    contract: FiniteVariationContract
    eligibility_receipt: FiniteVariationEligibilityReceipt | None = None

    def __post_init__(self) -> None:
        require_sha256(self.benchmark_sha256, "benchmark_sha256")
        require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        if type(self.known_phenotype_sha256s) is not tuple or any(
            type(value) is not str for value in self.known_phenotype_sha256s
        ):
            raise TypeError("known_phenotype_sha256s must be an exact string tuple")
        for value in self.known_phenotype_sha256s:
            require_sha256(value, "known_phenotype_sha256")
        if self.known_phenotype_sha256s != tuple(
            sorted(set(self.known_phenotype_sha256s))
        ):
            raise ValueError("known_phenotype_sha256s must be unique and canonical")
        if type(self.contract) is not FiniteVariationContract:
            raise TypeError("contract must be an exact FiniteVariationContract")
        validate_finite_variation_contract(self.contract)
        if (
            typed_json_sha256(self.contract.parent_configuration)
            != self.parent_configuration_sha256
        ):
            raise ValueError("finite contract is bound to a different parent")
        if self.eligibility_receipt is not None:
            if type(self.eligibility_receipt) is not FiniteVariationEligibilityReceipt:
                raise TypeError("eligibility_receipt must be exact or None")
            FiniteVariationEligibilityReceipt.__post_init__(self.eligibility_receipt)
            if (
                self.eligibility_receipt.eligible_contract_identity_sha256
                != self.contract.identity_sha256
            ):
                raise ValueError("finite contract differs from its eligibility receipt")
            if (
                self.eligibility_receipt.known_phenotype_sha256s
                != self.known_phenotype_sha256s
            ):
                raise ValueError("eligibility receipt differs from the novelty cutoff")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "benchmark_sha256": self.benchmark_sha256,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "known_phenotype_sha256s": list(self.known_phenotype_sha256s),
            "finite_variation_contract_sha256": self.contract.identity_sha256,
            "eligibility_receipt": (
                None
                if self.eligibility_receipt is None
                else self.eligibility_receipt.to_record()
            ),
        }


def _port_record(port: object, *, role: str) -> dict[str, object]:
    for name in ("port_id", "port_version", "definition_sha256"):
        if not hasattr(port, name):
            raise TypeError(f"{role} port is missing {name}")
    port_id = getattr(port, "port_id")
    port_version = getattr(port, "port_version")
    definition_sha256 = getattr(port, "definition_sha256")
    _validate_token(port_id, name=f"{role}.port_id")
    if type(port_version) is not int or port_version <= 0:
        raise ValueError(f"{role}.port_version must be positive")
    require_sha256(definition_sha256, f"{role}.definition_sha256")
    return {
        "role": role,
        "port_id": port_id,
        "port_version": port_version,
        "definition_sha256": definition_sha256,
    }


@runtime_checkable
class CampaignBenchmarkPort(Protocol):
    port_id: str
    port_version: int
    definition_sha256: str

    def open(self, request: BenchmarkSessionRequest) -> CampaignBenchmarkSession: ...


@runtime_checkable
class CampaignSeedPort(Protocol):
    port_id: str
    port_version: int
    definition_sha256: str

    def load(self, session: CampaignBenchmarkSession) -> CampaignSeedBatch: ...


@runtime_checkable
class CampaignCatalogPort(Protocol):
    port_id: str
    port_version: int
    definition_sha256: str

    def bind(
        self,
        benchmark: FrozenJsonObject,
        parent: FrozenJsonObject,
        known_phenotype_sha256s: tuple[str, ...],
    ) -> ParentVariationBinding: ...


@runtime_checkable
class CampaignEvidencePort(Protocol):
    port_id: str
    port_version: int
    definition_sha256: str

    def initialize_memory(
        self,
        session: CampaignBenchmarkSession,
        seeds: CampaignSeedBatch,
    ) -> FrozenJsonObject: ...

    def context(
        self,
        session: CampaignBenchmarkSession,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
        memory: FrozenJsonObject,
    ) -> FrozenJsonObject: ...

    def cards(
        self,
        session: CampaignBenchmarkSession,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
        memory: FrozenJsonObject,
    ) -> tuple[FrozenJsonObject, ...]: ...


@dataclass(frozen=True, slots=True)
class CampaignWorkloadPorts:
    """The complete benchmark-owned side of the inversion boundary."""

    benchmark: CampaignBenchmarkPort
    seeds: CampaignSeedPort
    catalog: CampaignCatalogPort
    evidence: CampaignEvidencePort

    def __post_init__(self) -> None:
        expected = (
            ("benchmark", self.benchmark, CampaignBenchmarkPort),
            ("seeds", self.seeds, CampaignSeedPort),
            ("catalog", self.catalog, CampaignCatalogPort),
            ("evidence", self.evidence, CampaignEvidencePort),
        )
        for role, value, contract in expected:
            if not isinstance(value, contract):
                raise TypeError(f"{role} must implement its campaign port")
            _port_record(value, role=role)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "benchmark": _port_record(self.benchmark, role="benchmark"),
            "seeds": _port_record(self.seeds, role="seeds"),
            "catalog": _port_record(self.catalog, role="catalog"),
            "evidence": _port_record(self.evidence, role="evidence"),
        }

    @property
    def ports_sha256(self) -> str:
        return _hash(_WORKLOAD_PORTS_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignPolicyBinding:
    """An executable policy dependency paired with its public identity."""

    implementation: object = field(repr=False, compare=False)
    policy_id: str
    policy_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if self.implementation is None:
            raise TypeError("policy implementation cannot be None")
        _validate_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArchiveUtilitySnapshot:
    """Frozen pre-generation utility/reference receipt for generic credit.

    ``snapshot_receipt`` is workload-owned data.  It may encode a fixed
    hypervolume reference, normalization constants, or another archive utility
    definition, while this core sees only immutable typed JSON and identities.
    """

    utility_id: str
    utility_version: int
    definition_sha256: str
    generation: int
    benchmark_sha256: str
    archive_sha256: str
    snapshot_receipt: FrozenJsonObject
    scalar_utility_hex: str | None = None

    def __post_init__(self) -> None:
        _validate_token(self.utility_id, name="utility_id")
        if type(self.utility_version) is not int or self.utility_version <= 0:
            raise ValueError("utility_version must be positive")
        require_sha256(self.definition_sha256, "definition_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("utility snapshot generation must be positive")
        require_sha256(self.benchmark_sha256, "benchmark_sha256")
        require_sha256(self.archive_sha256, "archive_sha256")
        _validate_frozen_object(self.snapshot_receipt, name="snapshot_receipt")
        if self.scalar_utility_hex is not None:
            if type(self.scalar_utility_hex) is not str:
                raise TypeError("scalar_utility_hex must be an exact string or None")
            try:
                scalar = float.fromhex(self.scalar_utility_hex)
            except ValueError as error:
                raise ValueError("scalar_utility_hex must be a float hex string") from error
            if not math.isfinite(scalar) or scalar < 0.0:
                raise ValueError(
                    "scalar_utility_hex must encode a finite non-negative utility"
                )
            if scalar.hex() != self.scalar_utility_hex:
                raise ValueError("scalar_utility_hex must be canonical")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        record = {
            "utility_id": self.utility_id,
            "utility_version": self.utility_version,
            "definition_sha256": self.definition_sha256,
            "generation": self.generation,
            "benchmark_sha256": self.benchmark_sha256,
            "archive_sha256": self.archive_sha256,
            "snapshot_receipt": thaw_json(self.snapshot_receipt),
        }
        if self.scalar_utility_hex is not None:
            record["scalar_utility_hex"] = self.scalar_utility_hex
        return record

    @property
    def snapshot_sha256(self) -> str:
        return _hash(_ARCHIVE_UTILITY_SNAPSHOT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}


@runtime_checkable
class ArchiveUtilityPort(Protocol):
    """Freeze one workload-defined archive utility before a generation."""

    utility_id: str
    utility_version: int
    definition_sha256: str

    def freeze(
        self,
        *,
        benchmark: FrozenJsonObject,
        generation: int,
        archive: FrozenJsonObject,
    ) -> ArchiveUtilitySnapshot: ...


def freeze_archive_utility(
    port: ArchiveUtilityPort,
    *,
    benchmark: FrozenJsonObject,
    generation: int,
    archive: FrozenJsonObject,
) -> ArchiveUtilitySnapshot:
    """Call and authenticate an injected pre-generation utility snapshot."""

    if not isinstance(port, ArchiveUtilityPort):
        raise TypeError("port must implement ArchiveUtilityPort")
    _validate_token(port.utility_id, name="archive_utility.utility_id")
    if type(port.utility_version) is not int or port.utility_version <= 0:
        raise ValueError("archive utility version must be positive")
    require_sha256(
        port.definition_sha256,
        "archive_utility.definition_sha256",
    )
    _validate_frozen_object(benchmark, name="benchmark")
    _validate_frozen_object(archive, name="archive")
    if type(generation) is not int or generation <= 0:
        raise ValueError("generation must be positive")
    snapshot = port.freeze(
        benchmark=benchmark,
        generation=generation,
        archive=archive,
    )
    if type(snapshot) is not ArchiveUtilitySnapshot:
        raise TypeError("archive utility must return ArchiveUtilitySnapshot")
    ArchiveUtilitySnapshot.__post_init__(snapshot)
    if (
        snapshot.utility_id != port.utility_id
        or snapshot.utility_version != port.utility_version
        or snapshot.definition_sha256 != port.definition_sha256
    ):
        raise ValueError("archive utility snapshot has a foreign definition")
    if snapshot.generation != generation:
        raise ValueError("archive utility snapshot has a foreign generation")
    if snapshot.benchmark_sha256 != typed_json_sha256(benchmark):
        raise ValueError("archive utility snapshot has a foreign benchmark")
    if snapshot.archive_sha256 != typed_json_sha256(archive):
        raise ValueError("archive utility snapshot has a foreign archive cutoff")
    return snapshot


@dataclass(frozen=True, slots=True)
class CampaignPolicies:
    """Workload-independent behavioral policies injected into a campaign."""

    cadence: CampaignCadence
    parent_selection: CampaignPolicyBinding
    memory_assignment: CampaignPolicyBinding
    portfolio_selection: CampaignPolicyBinding
    recombination: CampaignPolicyBinding
    reflection: CampaignPolicyBinding
    archive_utility: ArchiveUtilityPort
    reflection_supervision: CampaignReflectionSupervisionPolicy = field(
        default_factory=CampaignReflectionSupervisionPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cadence, CampaignCadence):
            raise TypeError("cadence must implement CampaignCadence")
        _port_record(
            _CadencePortView(self.cadence),
            role="cadence",
        )
        for name in (
            "parent_selection",
            "memory_assignment",
            "portfolio_selection",
            "recombination",
            "reflection",
        ):
            value = getattr(self, name)
            if type(value) is not CampaignPolicyBinding:
                raise TypeError(f"{name} must be an exact CampaignPolicyBinding")
            CampaignPolicyBinding.__post_init__(value)
        if type(self.reflection_supervision) is not CampaignReflectionSupervisionPolicy:
            raise TypeError(
                "reflection_supervision must be an exact "
                "CampaignReflectionSupervisionPolicy"
            )
        CampaignReflectionSupervisionPolicy.__post_init__(self.reflection_supervision)
        if not isinstance(self.archive_utility, ArchiveUtilityPort):
            raise TypeError("archive_utility must implement ArchiveUtilityPort")
        _validate_token(
            self.archive_utility.utility_id,
            name="archive_utility.utility_id",
        )
        if (
            type(self.archive_utility.utility_version) is not int
            or self.archive_utility.utility_version <= 0
        ):
            raise ValueError("archive utility version must be positive")
        require_sha256(
            self.archive_utility.definition_sha256,
            "archive_utility.definition_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "cadence": {
                "policy_id": self.cadence.policy_id,
                "policy_version": self.cadence.policy_version,
                "definition_sha256": self.cadence.definition_sha256,
            },
            "parent_selection": self.parent_selection.to_record(),
            "memory_assignment": self.memory_assignment.to_record(),
            "portfolio_selection": self.portfolio_selection.to_record(),
            "recombination": self.recombination.to_record(),
            "reflection": self.reflection.to_record(),
            "reflection_supervision": self.reflection_supervision.to_record(),
            "archive_utility": {
                "utility_id": self.archive_utility.utility_id,
                "utility_version": self.archive_utility.utility_version,
                "definition_sha256": self.archive_utility.definition_sha256,
            },
        }

    @property
    def policies_sha256(self) -> str:
        return _hash(_POLICIES_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class _CadencePortView:
    cadence: CampaignCadence

    @property
    def port_id(self) -> str:
        return self.cadence.policy_id

    @property
    def port_version(self) -> int:
        return self.cadence.policy_version

    @property
    def definition_sha256(self) -> str:
        return self.cadence.definition_sha256


@dataclass(frozen=True, slots=True)
class CampaignAgentRuntimeRequest:
    """Identity-only runtime preflight; it intentionally carries no model config."""

    protocol_sha256: str
    schedule_sha256: str
    session_sha256: str
    seed_batch_sha256: str
    workload_ports_sha256: str
    policies_sha256: str
    budget_sha256: str
    concurrency_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "schedule_sha256",
            "session_sha256",
            "seed_batch_sha256",
            "workload_ports_sha256",
            "policies_sha256",
            "budget_sha256",
            "concurrency_sha256",
        ):
            require_sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "protocol_sha256": self.protocol_sha256,
            "schedule_sha256": self.schedule_sha256,
            "session_sha256": self.session_sha256,
            "seed_batch_sha256": self.seed_batch_sha256,
            "workload_ports_sha256": self.workload_ports_sha256,
            "policies_sha256": self.policies_sha256,
            "budget_sha256": self.budget_sha256,
            "concurrency_sha256": self.concurrency_sha256,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_RUNTIME_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignAgentRuntimeReceipt:
    """Runtime-owned evidence that a prepared composition is executable."""

    request_sha256: str
    runtime_id: str
    runtime_version: int
    definition_sha256: str
    accepted: bool
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _validate_token(self.runtime_id, name="runtime_id")
        if type(self.runtime_version) is not int or self.runtime_version <= 0:
            raise ValueError("runtime_version must be positive")
        require_sha256(self.definition_sha256, "definition_sha256")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact bool")
        _validate_frozen_object(self.evidence, name="evidence")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "request_sha256": self.request_sha256,
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "definition_sha256": self.definition_sha256,
            "accepted": self.accepted,
            "evidence": thaw_json(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record = self._unsigned_record()
        return {**record, "receipt_sha256": _hash(_RUNTIME_RECEIPT_DOMAIN, record)}

    @property
    def receipt_sha256(self) -> str:
        return str(self.to_record()["receipt_sha256"])


@runtime_checkable
class CampaignAgentRuntimePort(Protocol):
    """Agent/runtime preflight outside the workload adapter boundary."""

    def prepare(
        self,
        request: CampaignAgentRuntimeRequest,
    ) -> CampaignAgentRuntimeReceipt: ...


@runtime_checkable
class CampaignJournalPort(Protocol):
    """Append one immutable campaign-level record."""

    def append(self, record: FrozenJsonObject) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedEvolutionCampaign:
    """Complete, replay-identifiable result of campaign preparation."""

    protocol: CampaignProtocol
    schedule: CampaignSchedule
    benchmark_session: CampaignBenchmarkSession
    seeds: CampaignSeedBatch
    workload_ports_sha256: str
    policies_sha256: str
    budget: OptimizerBudget
    concurrency: CampaignConcurrency
    runtime_receipt: CampaignAgentRuntimeReceipt

    def __post_init__(self) -> None:
        if type(self.protocol) is not CampaignProtocol:
            raise TypeError("protocol must be exact")
        CampaignProtocol.__post_init__(self.protocol)
        if type(self.schedule) is not CampaignSchedule:
            raise TypeError("schedule must be exact")
        CampaignSchedule.__post_init__(self.schedule)
        if self.schedule.protocol_sha256 != self.protocol.protocol_sha256:
            raise ValueError("schedule is bound to a different protocol")
        if type(self.benchmark_session) is not CampaignBenchmarkSession:
            raise TypeError("benchmark_session must be exact")
        CampaignBenchmarkSession.__post_init__(self.benchmark_session)
        if type(self.seeds) is not CampaignSeedBatch:
            raise TypeError("seeds must be exact")
        CampaignSeedBatch.__post_init__(self.seeds)
        if self.seeds.session_sha256 != self.benchmark_session.session_sha256:
            raise ValueError("seed batch is bound to a different session")
        for name in ("workload_ports_sha256", "policies_sha256"):
            require_sha256(getattr(self, name), name)
        if type(self.budget) is not OptimizerBudget:
            raise TypeError("budget must be an exact OptimizerBudget")
        OptimizerBudget.__post_init__(self.budget)
        if type(self.concurrency) is not CampaignConcurrency:
            raise TypeError("concurrency must be exact")
        CampaignConcurrency.__post_init__(self.concurrency)
        if type(self.runtime_receipt) is not CampaignAgentRuntimeReceipt:
            raise TypeError("runtime_receipt must be exact")
        CampaignAgentRuntimeReceipt.__post_init__(self.runtime_receipt)
        expected_runtime_request = CampaignAgentRuntimeRequest(
            protocol_sha256=self.protocol.protocol_sha256,
            schedule_sha256=self.schedule.schedule_sha256,
            session_sha256=self.benchmark_session.session_sha256,
            seed_batch_sha256=self.seeds.batch_sha256,
            workload_ports_sha256=self.workload_ports_sha256,
            policies_sha256=self.policies_sha256,
            budget_sha256=self.budget.budget_hash,
            concurrency_sha256=self.concurrency.concurrency_sha256,
        )
        if (
            self.runtime_receipt.request_sha256
            != expected_runtime_request.request_sha256
        ):
            raise ValueError("runtime receipt is bound to a different preparation")
        if not self.runtime_receipt.accepted:
            raise ValueError("a prepared campaign requires runtime acceptance")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "protocol": self.protocol.to_record(),
            "protocol_sha256": self.protocol.protocol_sha256,
            "schedule": self.schedule.to_record(),
            "schedule_sha256": self.schedule.schedule_sha256,
            "benchmark_session": self.benchmark_session.to_record(),
            "benchmark_session_sha256": self.benchmark_session.session_sha256,
            "seeds": self.seeds.to_record(),
            "seed_batch_sha256": self.seeds.batch_sha256,
            "workload_ports_sha256": self.workload_ports_sha256,
            "policies_sha256": self.policies_sha256,
            "budget": self.budget.to_trace_record(),
            "budget_sha256": self.budget.budget_hash,
            "concurrency": self.concurrency.to_record(),
            "concurrency_sha256": self.concurrency.concurrency_sha256,
            "runtime_receipt": self.runtime_receipt.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record = self._unsigned_record()
        return {**record, "preparation_sha256": _hash(_PREPARATION_DOMAIN, record)}

    @property
    def preparation_sha256(self) -> str:
        return str(self.to_record()["preparation_sha256"])


@dataclass(frozen=True, slots=True)
class EvolutionCampaign:
    """Prepare one reusable campaign composition without executing live waves."""

    protocol: CampaignProtocol
    workload: CampaignWorkloadPorts
    policies: CampaignPolicies
    runtime: CampaignAgentRuntimePort
    budget: OptimizerBudget
    concurrency: CampaignConcurrency
    journals: tuple[CampaignJournalPort, ...]

    def __post_init__(self) -> None:
        if type(self.protocol) is not CampaignProtocol:
            raise TypeError("protocol must be an exact CampaignProtocol")
        CampaignProtocol.__post_init__(self.protocol)
        if type(self.workload) is not CampaignWorkloadPorts:
            raise TypeError("workload must be exact CampaignWorkloadPorts")
        CampaignWorkloadPorts.__post_init__(self.workload)
        if type(self.policies) is not CampaignPolicies:
            raise TypeError("policies must be exact CampaignPolicies")
        CampaignPolicies.__post_init__(self.policies)
        if not isinstance(self.runtime, CampaignAgentRuntimePort):
            raise TypeError("runtime must implement CampaignAgentRuntimePort")
        if type(self.budget) is not OptimizerBudget:
            raise TypeError("budget must be an exact OptimizerBudget")
        OptimizerBudget.__post_init__(self.budget)
        if type(self.concurrency) is not CampaignConcurrency:
            raise TypeError("concurrency must be exact CampaignConcurrency")
        CampaignConcurrency.__post_init__(self.concurrency)
        if type(self.journals) is not tuple or not self.journals:
            raise ValueError("journals must be a non-empty exact tuple")
        if any(
            not isinstance(journal, CampaignJournalPort) for journal in self.journals
        ):
            raise TypeError("every journal must implement CampaignJournalPort")

    def prepare(self) -> PreparedEvolutionCampaign:
        """Preflight and seal a campaign; no evolutionary wave runs here."""

        self.__post_init__()
        schedule = self.policies.cadence.build(self.protocol)
        if type(schedule) is not CampaignSchedule:
            raise TypeError("cadence must return an exact CampaignSchedule")
        CampaignSchedule.__post_init__(schedule)
        if schedule.protocol_sha256 != self.protocol.protocol_sha256:
            raise ValueError("cadence returned a schedule for a foreign protocol")

        required_evaluations = (
            self.protocol.required_seed_count + schedule.planned_candidate_evaluations
        )
        if self.budget.max_generations < self.protocol.generation_count:
            raise ValueError("budget cannot cover the protocol generation count")
        if self.budget.max_unique_evaluations < required_evaluations:
            raise ValueError(
                "budget cannot cover the schedule's maximum candidate evaluations"
            )
        if self.budget.max_logical_llm_calls < schedule.planned_agent_calls:
            raise ValueError("budget cannot cover the schedule's planned agent calls")

        session_request = BenchmarkSessionRequest(
            protocol_sha256=self.protocol.protocol_sha256,
            budget_sha256=self.budget.budget_hash,
            outer_seed=self.protocol.outer_seed,
            requested_evaluator_concurrency=(self.concurrency.evaluator_concurrency),
        )
        session = self.workload.benchmark.open(session_request)
        if type(session) is not CampaignBenchmarkSession:
            raise TypeError("benchmark port must return CampaignBenchmarkSession")
        CampaignBenchmarkSession.__post_init__(session)
        if session.request_sha256 != session_request.request_sha256:
            raise ValueError("benchmark session is bound to a foreign request")
        if self.concurrency.evaluator_concurrency > session.evaluator_concurrency_cap:
            raise ValueError("requested evaluator concurrency exceeds benchmark cap")

        seeds = self.workload.seeds.load(session)
        if type(seeds) is not CampaignSeedBatch:
            raise TypeError("seed port must return CampaignSeedBatch")
        CampaignSeedBatch.__post_init__(seeds)
        if seeds.session_sha256 != session.session_sha256:
            raise ValueError("seed batch is bound to a foreign benchmark session")
        if len(seeds.seeds) != self.protocol.required_seed_count:
            raise ValueError("seed batch cardinality differs from the protocol")

        runtime_request = CampaignAgentRuntimeRequest(
            protocol_sha256=self.protocol.protocol_sha256,
            schedule_sha256=schedule.schedule_sha256,
            session_sha256=session.session_sha256,
            seed_batch_sha256=seeds.batch_sha256,
            workload_ports_sha256=self.workload.ports_sha256,
            policies_sha256=self.policies.policies_sha256,
            budget_sha256=self.budget.budget_hash,
            concurrency_sha256=self.concurrency.concurrency_sha256,
        )
        runtime_receipt = self.runtime.prepare(runtime_request)
        if type(runtime_receipt) is not CampaignAgentRuntimeReceipt:
            raise TypeError("runtime must return CampaignAgentRuntimeReceipt")
        CampaignAgentRuntimeReceipt.__post_init__(runtime_receipt)
        if runtime_receipt.request_sha256 != runtime_request.request_sha256:
            raise ValueError("runtime receipt is bound to a foreign request")
        if not runtime_receipt.accepted:
            raise ValueError("agent runtime rejected campaign preparation")

        prepared = PreparedEvolutionCampaign(
            protocol=self.protocol,
            schedule=schedule,
            benchmark_session=session,
            seeds=seeds,
            workload_ports_sha256=self.workload.ports_sha256,
            policies_sha256=self.policies.policies_sha256,
            budget=self.budget,
            concurrency=self.concurrency,
            runtime_receipt=runtime_receipt,
        )
        journal_record = freeze_json(prepared.to_record())
        if type(journal_record) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("campaign preparation did not freeze to an object")
        for journal in self.journals:
            journal.append(journal_record)
        return prepared


__all__ = [
    "ALTERNATING_CADENCE_DEFINITION_SHA256",
    "ALTERNATING_CADENCE_POLICY_ID",
    "ALTERNATING_CADENCE_POLICY_VERSION",
    "SEALED_CUTOFF_DELAYED_CADENCE_DEFINITION_SHA256",
    "SEALED_CUTOFF_DELAYED_CADENCE_POLICY_ID",
    "SEALED_CUTOFF_DELAYED_CADENCE_POLICY_VERSION",
    "MAX_CAMPAIGN_GENERATIONS",
    "MIN_CAMPAIGN_GENERATIONS",
    "AlternatingPortfolioRecombinationCadence",
    "ArchiveUtilityPort",
    "ArchiveUtilitySnapshot",
    "BenchmarkSessionRequest",
    "CampaignAgentRuntimePort",
    "CampaignAgentRuntimeReceipt",
    "CampaignAgentRuntimeRequest",
    "CampaignBenchmarkPort",
    "CampaignBenchmarkSession",
    "CampaignCadence",
    "CampaignCatalogPort",
    "CampaignConcurrency",
    "CampaignEvidencePort",
    "CampaignGenerationKind",
    "CampaignGenerationPair",
    "CampaignGenerationStep",
    "CampaignJournalPort",
    "CampaignPolicies",
    "CampaignPolicyBinding",
    "CampaignPromotionBarrier",
    "CampaignProtocol",
    "CampaignReflectionWave",
    "CampaignReflectionSupervisionPolicy",
    "CampaignSchedule",
    "CampaignSeed",
    "CampaignSeedBatch",
    "CampaignSeedPort",
    "CampaignWorkloadPorts",
    "EvolutionCampaign",
    "ParentVariationBinding",
    "PreparedEvolutionCampaign",
    "ReflectionLaunchMode",
    "ReflectionFailureMode",
    "TerminalReflectionPolicy",
    "ReflectionVisibility",
    "SealedCutoffDelayedAdmissionCadence",
    "freeze_archive_utility",
]
