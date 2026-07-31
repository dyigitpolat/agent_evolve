"""Workload-neutral high-level campaign presets.

This module owns generational shape, delayed reflection chronology, exact
budgets, and concurrency.  Workload adapters remain in ``WorkloadKit`` while
method implementations remain injectable as ``PortfolioCampaignBehavior``.
The split prevents a domain package from copying a thousand-line launcher just
to change seeds, an evaluator, or static prompt semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilityPort,
    CampaignAgentRuntimePort,
    CampaignConcurrency,
    CampaignJournalPort,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignReflectionSupervisionPolicy,
    EvolutionCampaign,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
)
from agent_evolve.campaign_workload import AgenticCampaignWorkloadConfig
from agent_evolve.workload_kit import WorkloadKit


_PRESET_DEFINITION_DOMAIN = b"agent-evolve:delayed-portfolio-preset:v1\x00"
_SCALE_SHAPE_DOMAIN = b"agent-evolve:portfolio-scale-shape:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PortfolioScaleShape:
    """One workload-neutral depth/width/operator opportunity shape."""

    shape_id: str
    generation_count: int
    parents_per_portfolio_generation: int
    portfolio_width: int
    recombinations_per_parent: int

    def __post_init__(self) -> None:
        if type(self.shape_id) is not str or _TOKEN.fullmatch(self.shape_id) is None:
            raise ValueError("shape_id must use the closed lowercase token grammar")
        if (
            type(self.generation_count) is not int
            or not 3 <= self.generation_count <= 24
        ):
            raise ValueError("generation_count must lie in [3, 24]")
        for name in (
            "parents_per_portfolio_generation",
            "portfolio_width",
            "recombinations_per_parent",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.portfolio_width < 2:
            raise ValueError("portfolio_width must be at least two")

    @property
    def portfolio_generation_count(self) -> int:
        self.__post_init__()
        return (self.generation_count + 1) // 2

    @property
    def recombination_generation_count(self) -> int:
        self.__post_init__()
        return self.generation_count // 2

    @property
    def planned_offspring_occurrences(self) -> int:
        self.__post_init__()
        return self.parents_per_portfolio_generation * (
            self.portfolio_generation_count * self.portfolio_width
            + self.recombination_generation_count * self.recombinations_per_parent
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "shape_id": self.shape_id,
            "generation_count": self.generation_count,
            "parents_per_portfolio_generation": (
                self.parents_per_portfolio_generation
            ),
            "portfolio_width": self.portfolio_width,
            "recombinations_per_parent": self.recombinations_per_parent,
            "portfolio_generation_count": self.portfolio_generation_count,
            "recombination_generation_count": self.recombination_generation_count,
            "planned_offspring_occurrences": self.planned_offspring_occurrences,
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_SCALE_SHAPE_DOMAIN, self.to_record())


EQUAL_60_OFFSPRING_SCALE_SHAPES = {
    value.shape_id: value
    for value in (
        # The agentic selector has an authenticated support width of eight.
        # Keep every equal-budget treatment inside that generic interface and
        # shift the short-run opportunity into recombination instead of asking
        # one call for an impossible fourteen-candidate portfolio.
        PortfolioScaleShape("g4_k8_r7", 4, 2, 8, 7),
        PortfolioScaleShape("g6_k8_r2", 6, 2, 8, 2),
        PortfolioScaleShape("g10_k4_r2", 10, 2, 4, 2),
    )
}


# Equal-budget B38 shapes.  The two campaign seeds are outside the 36
# offspring opportunities.  Keeping the alternatives in one workload-neutral
# registry prevents benchmark runners from silently rebuilding different
# schedules with local constants.
#
# ``g6_k5_r1`` is the anchor-heavy successor: for two parent lanes it preserves
# the reference shape's eighteen semantic portfolio positions, doubles the
# capacity available to a four-member protected numerical batch (six to twelve
# evaluated positions over the campaign), and halves compulsory
# recombination from twelve to six positions.  It changes allocation
# opportunity, not the total real-evaluation budget.
EQUAL_36_OFFSPRING_SCALE_SHAPES = {
    value.shape_id: value
    for value in (
        PortfolioScaleShape("g6_k4_r2", 6, 2, 4, 2),
        PortfolioScaleShape("g6_k5_r1", 6, 2, 5, 1),
    )
}


# Truthfully separate near-reference-budget successors from the exactly B38
# registry. Both successors stay inside the already-qualified K8-to-K4
# selector. ``g6_k4_r1`` is a B32 screening rung (24 portfolio + 6
# recombination occurrences); ``g7_k4_r1`` is the B40 expansion rung (32 + 6).
SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES = {
    **EQUAL_36_OFFSPRING_SCALE_SHAPES,
    "g6_k4_r1": PortfolioScaleShape("g6_k4_r1", 6, 2, 4, 1),
    "g7_k4_r1": PortfolioScaleShape("g7_k4_r1", 7, 2, 4, 1),
}


# Anytime-curve rungs.  Everything above is a near-reference budget, so until
# now the largest expressible campaign was B40 -- and that was a **registry
# cap, not a design choice**.  It is the reason every number this project holds
# sits at B=38: a claim about evaluation efficiency is a claim about a curve,
# and no curve past B40 could be expressed at all.
#
# These rungs are additions.  No entry above is modified, so nothing in flight
# changes shape underneath it; a run selects by ``shape_id`` and every existing
# id still resolves to exactly the shape it did before.
#
# Budget, from ``planned_offspring_occurrences``:
#     P * (ceil(G/2) * K + floor(G/2) * R)   plus the two benchmark seeds
#
# ``g24_k17_r8`` is the classical-matched rung: 600 offspring + 2 seeds = 602
# charged occurrences, matching the sealed classical anytime arms' budget of
# 600 so our curve and NSGA-II's are read on the same axis.  It preserves the
# reference rung's 2:1 portfolio-to-recombination balance (g6_k4_r2 is K=4,
# R=2), so it is the same method run longer rather than a different one.
# ``g12_k8_r4`` is the same balance at 144 + 2 = 146, a cheaper rung for
# smoking the path before the full curve is bought.
#
# ``generation_count`` is independently capped at 24 by ``PortfolioScaleShape``
# validation, so depth alone cannot reach 600; these rungs buy the remaining
# budget in width, which is why K is large rather than G.
ANYTIME_CURVE_CAMPAIGN_SCALE_SHAPES = {
    value.shape_id: value
    for value in (
        PortfolioScaleShape("g12_k8_r4", 12, 2, 8, 4),
        PortfolioScaleShape("g24_k17_r8", 24, 2, 17, 8),
    )
}


#: Every registered shape, small-budget and anytime-curve alike.  Lookups that
#: want the full set use this; the two registries above stay exactly as they
#: were for any caller that deliberately wants only the near-reference rungs.
CAMPAIGN_SCALE_SHAPES = {
    **SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES,
    **ANYTIME_CURVE_CAMPAIGN_SCALE_SHAPES,
}


CLASSICAL_MATCHED_B602_SCALE_SHAPE = ANYTIME_CURVE_CAMPAIGN_SCALE_SHAPES[
    "g24_k17_r8"
]
ANYTIME_SMOKE_B146_SCALE_SHAPE = ANYTIME_CURVE_CAMPAIGN_SCALE_SHAPES[
    "g12_k8_r4"
]
REFERENCE_36_OFFSPRING_SCALE_SHAPE = EQUAL_36_OFFSPRING_SCALE_SHAPES[
    "g6_k4_r2"
]
ANCHOR_HEAVY_36_OFFSPRING_SCALE_SHAPE = EQUAL_36_OFFSPRING_SCALE_SHAPES[
    "g6_k5_r1"
]
REFERENCE_HEAVY_B40_SCALE_SHAPE = SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES[
    "g7_k4_r1"
]
REFERENCE_HEAVY_B32_SCALE_SHAPE = SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES[
    "g6_k4_r1"
]


@dataclass(frozen=True, slots=True)
class PortfolioCampaignBehavior:
    """Method-owned collaborators, deliberately separate from a workload."""

    parent_selection: CampaignPolicyBinding
    memory_assignment: CampaignPolicyBinding
    portfolio_selection: CampaignPolicyBinding
    recombination: CampaignPolicyBinding
    reflection: CampaignPolicyBinding
    archive_utility: ArchiveUtilityPort
    reflection_supervision: CampaignReflectionSupervisionPolicy = (
        CampaignReflectionSupervisionPolicy()
    )

    def __post_init__(self) -> None:
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
        if not isinstance(self.archive_utility, ArchiveUtilityPort):
            raise TypeError("archive_utility must implement ArchiveUtilityPort")
        if type(self.reflection_supervision) is not (
            CampaignReflectionSupervisionPolicy
        ):
            raise TypeError(
                "reflection_supervision must be exact "
                "CampaignReflectionSupervisionPolicy"
            )
        CampaignReflectionSupervisionPolicy.__post_init__(
            self.reflection_supervision
        )

    @classmethod
    def from_campaign_policies(
        cls,
        policies: CampaignPolicies,
    ) -> "PortfolioCampaignBehavior":
        """Migrate an existing strict policy bundle without changing identity."""

        if type(policies) is not CampaignPolicies:
            raise TypeError("policies must be an exact CampaignPolicies")
        CampaignPolicies.__post_init__(policies)
        return cls(
            parent_selection=policies.parent_selection,
            memory_assignment=policies.memory_assignment,
            portfolio_selection=policies.portfolio_selection,
            recombination=policies.recombination,
            reflection=policies.reflection,
            archive_utility=policies.archive_utility,
            reflection_supervision=policies.reflection_supervision,
        )

    def bind(self) -> CampaignPolicies:
        """Bind this behavior to the preset's sealed delayed cadence."""

        self.__post_init__()
        return CampaignPolicies(
            cadence=SealedCutoffDelayedAdmissionCadence(),
            parent_selection=self.parent_selection,
            memory_assignment=self.memory_assignment,
            portfolio_selection=self.portfolio_selection,
            recombination=self.recombination,
            reflection=self.reflection,
            archive_utility=self.archive_utility,
            reflection_supervision=self.reflection_supervision,
        )


@dataclass(frozen=True, slots=True)
class DelayedPortfolioCampaignPreset:
    """Exact P/R campaign plan with useful, delayed, quarantined reflection."""

    generation_count: int
    outer_seed: int
    parents_per_portfolio_generation: int = 2
    portfolio_width: int = 4
    recombinations_per_parent: int = 2
    evaluator_concurrency: int = 1
    agent_concurrency: int = 3
    agent_queue_capacity: int = 8

    def __post_init__(self) -> None:
        if type(self.generation_count) is not int or not 3 <= self.generation_count <= 24:
            raise ValueError("generation_count must lie in [3, 24]")
        if type(self.outer_seed) is not int or not -(1 << 127) <= self.outer_seed < (
            1 << 127
        ):
            raise ValueError("outer_seed must be an exact signed int128")
        for name in (
            "parents_per_portfolio_generation",
            "recombinations_per_parent",
            "evaluator_concurrency",
            "agent_concurrency",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.portfolio_width) is not int or self.portfolio_width < 2:
            raise ValueError("portfolio_width must be at least two")
        if (
            type(self.agent_queue_capacity) is not int
            or self.agent_queue_capacity < self.agent_concurrency
        ):
            raise ValueError("agent_queue_capacity must cover agent_concurrency")

    @classmethod
    def generations(
        cls,
        generation_count: int,
        *,
        outer_seed: int,
        parents_per_portfolio_generation: int = 2,
        portfolio_width: int = 4,
        recombinations_per_parent: int = 2,
        evaluator_concurrency: int = 1,
        agent_concurrency: int = 3,
        agent_queue_capacity: int = 8,
    ) -> "DelayedPortfolioCampaignPreset":
        """Readable public constructor used by experiment manifests."""

        return cls(
            generation_count=generation_count,
            outer_seed=outer_seed,
            parents_per_portfolio_generation=parents_per_portfolio_generation,
            portfolio_width=portfolio_width,
            recombinations_per_parent=recombinations_per_parent,
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=agent_concurrency,
            agent_queue_capacity=agent_queue_capacity,
        )

    @classmethod
    def scale_shape(
        cls,
        shape: PortfolioScaleShape,
        *,
        outer_seed: int,
        evaluator_concurrency: int = 1,
        agent_concurrency: int = 3,
        agent_queue_capacity: int = 8,
    ) -> "DelayedPortfolioCampaignPreset":
        """Construct a preset from one authenticated experimental shape."""

        if type(shape) is not PortfolioScaleShape:
            raise TypeError("shape must be an exact PortfolioScaleShape")
        shape.__post_init__()
        return cls.generations(
            shape.generation_count,
            outer_seed=outer_seed,
            parents_per_portfolio_generation=(
                shape.parents_per_portfolio_generation
            ),
            portfolio_width=shape.portfolio_width,
            recombinations_per_parent=shape.recombinations_per_parent,
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=agent_concurrency,
            agent_queue_capacity=agent_queue_capacity,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "preset_id": "sealed_cutoff_delayed_portfolio",
            "preset_version": 1,
            "generation_count": self.generation_count,
            "parents_per_portfolio_generation": (
                self.parents_per_portfolio_generation
            ),
            "portfolio_width": self.portfolio_width,
            "recombinations_per_parent": self.recombinations_per_parent,
            "reflections_per_recombination_generation": 1,
            "reflection_promotion_block_pairs": 1,
            "terminal_reflection_policy": (
                TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER.value
            ),
            "evaluator_concurrency": self.evaluator_concurrency,
            "agent_concurrency": self.agent_concurrency,
            "agent_queue_capacity": self.agent_queue_capacity,
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_PRESET_DEFINITION_DOMAIN, self.to_record())

    def protocol(self, *, required_seed_count: int) -> CampaignProtocol:
        """Build the exact protocol; the outer seed remains separately bound."""

        if type(required_seed_count) is not int or required_seed_count <= 0:
            raise ValueError("required_seed_count must be a positive exact integer")
        return CampaignProtocol(
            protocol_id="sealed_cutoff_delayed_portfolio",
            protocol_version=1,
            definition_sha256=self.definition_sha256,
            outer_seed=self.outer_seed,
            generation_count=self.generation_count,
            required_seed_count=required_seed_count,
            parents_per_portfolio_generation=(
                self.parents_per_portfolio_generation
            ),
            portfolio_width=self.portfolio_width,
            recombinations_per_parent=self.recombinations_per_parent,
            reflections_per_recombination_generation=1,
            reflection_promotion_block_pairs=1,
            terminal_reflection_policy=(
                TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
            ),
        )

    def budget(self, *, required_seed_count: int) -> OptimizerBudget:
        """Derive—not hand-maintain—the exact maximum schedule budget."""

        protocol = self.protocol(required_seed_count=required_seed_count)
        schedule = SealedCutoffDelayedAdmissionCadence().build(protocol)
        return OptimizerBudget(
            max_unique_evaluations=(
                required_seed_count + schedule.planned_candidate_evaluations
            ),
            max_logical_llm_calls=schedule.planned_agent_calls,
            max_generations=self.generation_count,
        )

    def concurrency(self) -> CampaignConcurrency:
        return CampaignConcurrency(
            evaluator_concurrency=self.evaluator_concurrency,
            agent_concurrency=self.agent_concurrency,
            agent_queue_capacity=self.agent_queue_capacity,
        )

    def compose(
        self,
        *,
        workload: WorkloadKit | AgenticCampaignWorkloadConfig,
        behavior: PortfolioCampaignBehavior,
        runtime: CampaignAgentRuntimePort,
        journals: tuple[CampaignJournalPort, ...],
    ) -> EvolutionCampaign:
        """Compose a strict campaign without running preflight or live waves."""

        self.__post_init__()
        if type(workload) is WorkloadKit:
            config = workload.to_campaign_workload()
        elif type(workload) is AgenticCampaignWorkloadConfig:
            config = workload
            AgenticCampaignWorkloadConfig.__post_init__(config)
        else:
            raise TypeError(
                "workload must be exact WorkloadKit or "
                "AgenticCampaignWorkloadConfig"
            )
        if self.evaluator_concurrency > config.evaluator_concurrency_cap:
            raise ValueError(
                "preset evaluator concurrency exceeds workload admission cap"
            )
        if type(behavior) is not PortfolioCampaignBehavior:
            raise TypeError("behavior must be exact PortfolioCampaignBehavior")
        behavior.__post_init__()
        required_seed_count = len(config.seeds)
        return EvolutionCampaign(
            protocol=self.protocol(required_seed_count=required_seed_count),
            workload=config.build_ports(),
            policies=behavior.bind(),
            runtime=runtime,
            budget=self.budget(required_seed_count=required_seed_count),
            concurrency=self.concurrency(),
            journals=journals,
        )


__all__ = [
    "ANCHOR_HEAVY_36_OFFSPRING_SCALE_SHAPE",
    "ANYTIME_CURVE_CAMPAIGN_SCALE_SHAPES",
    "ANYTIME_SMOKE_B146_SCALE_SHAPE",
    "CAMPAIGN_SCALE_SHAPES",
    "CLASSICAL_MATCHED_B602_SCALE_SHAPE",
    "DelayedPortfolioCampaignPreset",
    "EQUAL_36_OFFSPRING_SCALE_SHAPES",
    "EQUAL_60_OFFSPRING_SCALE_SHAPES",
    "PortfolioCampaignBehavior",
    "PortfolioScaleShape",
    "REFERENCE_36_OFFSPRING_SCALE_SHAPE",
]
