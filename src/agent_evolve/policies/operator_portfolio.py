"""Variation arms under measured credit: operators earn their offspring slots.

The classical crossover+mutate path becomes one arm among several. Arms --
rule-written or model-authored -- are chosen per offspring slot by UCB1 over
measured SURVIVAL (did the child outlive truncation against parents and
siblings?), with two guarantees that keep the mechanism honest: the
classical arm holds a floor slot every generation, so the comparator is
always inside the run; and credit is assigned at survival time, never at
proposal time, so an arm cannot look good by proposing loudly.

Authored arms construct through the bounded runtime, batched one process per
generation. Every authored child passes :func:`child_within_material` --
each value inherited from a parent or drawn from the (already restricted)
declared domain, same shape as the template -- or it is rejected, counted,
and its slot falls back to the classical arm. The authoring line holds: an
artifact reaches real evaluations only through slots it earned.

Retirement is preregistered in code, not judged post hoc: an arm with at
least four measured children and zero survivors, while the classical arm has
survivors, retires for the remainder of the run. One way, counted, recorded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.authored import AuthoredArtifact
from agent_evolve.policies.genetic import (
    Locus,
    crossover,
    loci_of,
    locus_domain,
    mutate,
    read_locus,
)

__all__ = [
    "ArmCredit",
    "CreditTable",
    "VariationArm",
    "OperatorPortfolio",
    "child_within_material",
    "segment_arm",
]

Config = Dict[str, Any]

#: A rule arm's constructor: (parent_a, parent_b, candidate_model,
#: restriction, rng) -> child. Classical and rule arms run in-process; they
#: are trusted code shipped with the package.
RuleConstruct = Callable[[Config, Config, Any, Any, random.Random], Config]


@dataclass
class ArmCredit:
    proposed: int = 0
    measured: int = 0
    survived: int = 0
    rejected_out_of_domain: int = 0
    runtime_failures: int = 0


class CreditTable:
    """Per-arm outcomes, and the UCB1 view over them."""

    def __init__(self) -> None:
        self.arms: Dict[str, ArmCredit] = {}

    def arm(self, name: str) -> ArmCredit:
        return self.arms.setdefault(name, ArmCredit())

    def total_measured(self) -> int:
        return sum(credit.measured for credit in self.arms.values())

    def ucb(self, name: str, *, c: float = 1.4) -> float:
        credit = self.arm(name)
        if credit.measured == 0:
            return float("inf")           # every arm gets its first hearing
        rate = credit.survived / credit.measured
        total = max(2, self.total_measured())
        return rate + c * math.sqrt(math.log(total) / credit.measured)

    def as_flat_counters(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, credit in sorted(self.arms.items()):
            for fieldname in ("proposed", "measured", "survived",
                              "rejected_out_of_domain", "runtime_failures"):
                out[f"{name}__{fieldname}"] = getattr(credit, fieldname)
        return out


@dataclass(frozen=True)
class VariationArm:
    """One way to construct a child. ``artifact`` only for authored arms."""

    name: str
    kind: str                                  # "classical" | "rule" | "authored"
    construct: Optional[RuleConstruct] = None  # in-process arms only
    artifact: Optional[AuthoredArtifact] = None


def classical_arm() -> VariationArm:
    """Uniform-mask crossover + rate-1/n mutation: the loop's own operators.

    Note one composition limit, stated rather than hidden: under a portfolio
    the chooser still picks the parents, but the classical arm draws its own
    mask instead of the chooser's -- mask-level guidance and arm-level credit
    do not compose yet.
    """

    def construct(parent_a: Config, parent_b: Config, candidate_model: Any,
                  restriction: Any, rng: random.Random) -> Config:
        child = crossover(parent_a, parent_b, rng=rng)
        return mutate(child, candidate_model, restriction=restriction, rng=rng)

    return VariationArm(name="classical", kind="classical", construct=construct)


def segment_arm() -> VariationArm:
    """Contiguous-block recombination: the shipped rule comparator.

    Uniform masks shred building blocks that live in adjacent loci; a
    contiguous cut preserves them. This is the classic alternative any
    authored operator should at minimum out-survive before its story is
    believed.
    """

    def construct(parent_a: Config, parent_b: Config, candidate_model: Any,
                  restriction: Any, rng: random.Random) -> Config:
        loci = loci_of(parent_a)
        n = len(loci)
        start = rng.randrange(n)
        length = 1 + rng.randrange(max(1, n // 2))
        mask = tuple(start <= i < start + length for i in range(n))
        child = crossover(parent_a, parent_b, mask=mask)
        return mutate(child, candidate_model, restriction=restriction, rng=rng)

    return VariationArm(name="segment", kind="rule", construct=construct)


def child_within_material(
    child: Any,
    parent_a: Config,
    parent_b: Config,
    domains: Mapping[str, Tuple[Any, ...]],
) -> bool:
    """May this authored child enter the run at all?

    Shape must equal the parents' shape, and every locus value must be
    inherited from a parent at that locus or drawn from the provided
    (already restricted) declared domain. This is the widened authoring
    line: an operator may construct any candidate the schema and its parents
    can express, and nothing else.
    """

    if not isinstance(child, dict):
        return False
    if set(child.keys()) != set(parent_a.keys()):
        return False
    try:
        child_loci = loci_of(child)
    except Exception:
        return False
    if child_loci != loci_of(parent_a):
        return False
    for locus in child_loci:
        value = read_locus(child, locus)
        if value == read_locus(parent_a, locus):
            continue
        if value == read_locus(parent_b, locus):
            continue
        domain = domains.get(str(locus), ())
        if domain and value in domain:
            continue
        return False
    return True


class PortfolioTelemetry:
    """Portfolio-level counters; per-arm credit is appended at harvest."""

    __slots__ = ("generations", "constructed", "authored_children",
                 "rejected_out_of_domain", "runtime_failures", "retired")

    def __init__(self) -> None:
        self.generations = 0
        self.constructed = 0
        self.authored_children = 0
        self.rejected_out_of_domain = 0
        self.runtime_failures = 0
        self.retired = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


class OperatorPortfolio:
    """Chooses arms, constructs a generation, and assigns survival credit."""

    def __init__(
        self,
        arms: Sequence[VariationArm],
        *,
        runtime: Any = None,
        max_authored_fraction: float = 0.5,
    ) -> None:
        names = [arm.name for arm in arms]
        if len(names) != len(set(names)):
            raise ValueError(f"arm names must be unique, got {names}")
        if not any(arm.kind == "classical" for arm in arms):
            raise ValueError("the portfolio needs the classical arm: it is "
                             "the comparator every other arm is read against")
        if any(arm.kind == "authored" for arm in arms) and runtime is None:
            raise ValueError("authored arms need a runtime")
        self.arms = tuple(arms)
        self.runtime = runtime
        self.max_authored_fraction = float(max_authored_fraction)
        self.credit = CreditTable()
        self.retired: set[str] = set()
        self.counters = PortfolioTelemetry()
        self.mechanism = "operator_portfolio"
        self.authored_by = (
            "llm" if any(arm.kind == "authored" for arm in self.arms) else "rule"
        )

        # Harvest sees one flat counter map: portfolio totals + per-arm credit.
        outer = self

        class _Merged:
            def as_dict(self) -> dict[str, int]:
                merged = outer.counters.as_dict()
                merged.update(outer.credit.as_flat_counters())
                return merged

        self.telemetry = _Merged()

    # -- choosing -----------------------------------------------------------

    def active_arms(self) -> Tuple[VariationArm, ...]:
        return tuple(arm for arm in self.arms if arm.name not in self.retired)

    def assign(self, count: int, rng: random.Random) -> List[VariationArm]:
        """One arm per offspring slot; slot 0 is always classical.

        Unmeasured arms rank at +inf so every arm gets its first hearing;
        ties break toward the FEWEST proposals so far (including this
        generation's), so first hearings rotate instead of the
        alphabetically-first arm swallowing every slot. Deterministic; *rng*
        is reserved for future stochastic policies.
        """

        del rng
        active = self.active_arms()
        classical = next(arm for arm in active if arm.kind == "classical")
        out: List[VariationArm] = []
        tally: Dict[str, int] = {}
        authored_cap = max(0, int(self.max_authored_fraction * count + 1e-9))
        authored_given = 0

        def take(arm: VariationArm) -> None:
            tally[arm.name] = tally.get(arm.name, 0) + 1
            out.append(arm)

        for slot in range(count):
            if slot == 0:
                take(classical)
                continue
            ranked = sorted(
                active,
                key=lambda arm: (
                    -self.credit.ucb(arm.name),
                    self.credit.arm(arm.name).proposed + tally.get(arm.name, 0),
                    arm.name,
                ),
            )
            pick = ranked[0]
            if pick.kind == "authored" and authored_given >= authored_cap:
                pick = next((arm for arm in ranked
                             if arm.kind != "authored"), classical)
            if pick.kind == "authored":
                authored_given += 1
            take(pick)
        return out

    # -- constructing -------------------------------------------------------

    def construct_generation(
        self,
        parent_pairs: Sequence[Tuple[Config, Config]],
        candidate_model: Any,
        restriction: Any,
        rng: random.Random,
        *,
        generation: int,
    ) -> Tuple[List[Config], List[str]]:
        """Children for the generation, plus the arm that made each."""

        self.counters.generations += 1
        assignments = self.assign(len(parent_pairs), rng)
        classical = next(arm for arm in self.arms if arm.kind == "classical")

        kids: List[Optional[Config]] = [None] * len(parent_pairs)
        origins: List[str] = [""] * len(parent_pairs)
        authored_slots: Dict[str, List[int]] = {}
        for slot, (arm, (parent_a, parent_b)) in enumerate(
                zip(assignments, parent_pairs)):
            if arm.kind == "authored":
                authored_slots.setdefault(arm.name, []).append(slot)
                continue
            kids[slot] = arm.construct(parent_a, parent_b, candidate_model,
                                       restriction, rng)
            origins[slot] = arm.name
            self.credit.arm(arm.name).proposed += 1

        for name, slots in authored_slots.items():
            arm = next(a for a in self.arms if a.name == name)
            self._construct_authored(
                arm, slots, parent_pairs, candidate_model, restriction,
                rng, kids, origins, classical, generation)

        self.counters.constructed += len(kids)
        return [kid for kid in kids if kid is not None], origins

    def _construct_authored(
        self, arm, slots, parent_pairs, candidate_model, restriction,
        rng, kids, origins, classical, generation,
    ) -> None:
        template = parent_pairs[slots[0]][0]
        loci = loci_of(template)
        loci_names = [str(locus) for locus in loci]
        domains = {
            str(locus): list(locus_domain(candidate_model, locus,
                                          restriction=restriction))
            for locus in loci
        }
        calls = []
        for slot in slots:
            parent_a, parent_b = parent_pairs[slot]
            calls.append([dict(parent_a), dict(parent_b), loci_names,
                          domains, generation * 1000 + slot])
            self.credit.arm(arm.name).proposed += 1

        outcome = self.runtime.call(arm.artifact, calls)
        results: Sequence[Any]
        if outcome.ok:
            results = outcome.results
        else:
            self.counters.runtime_failures += len(slots)
            self.credit.arm(arm.name).runtime_failures += len(slots)
            results = [None] * len(slots)

        domain_tuples = {name: tuple(values) for name, values in domains.items()}
        for slot, child in zip(slots, results):
            parent_a, parent_b = parent_pairs[slot]
            if child is not None and child_within_material(
                    child, parent_a, parent_b, domain_tuples):
                kids[slot] = dict(child)
                origins[slot] = arm.name
                self.counters.authored_children += 1
            else:
                if child is not None:
                    self.counters.rejected_out_of_domain += 1
                    self.credit.arm(arm.name).rejected_out_of_domain += 1
                kids[slot] = classical.construct(
                    parent_a, parent_b, candidate_model, restriction, rng)
                origins[slot] = f"{classical.name}(fallback)"
                self.credit.arm(classical.name).proposed += 1

    # -- credit and retirement ---------------------------------------------

    def record_measured(self, origin: str, *, survived: bool) -> None:
        name = origin.split("(")[0]
        credit = self.credit.arm(name)
        credit.measured += 1
        if survived:
            credit.survived += 1

    def review(self) -> Tuple[str, ...]:
        """The preregistered retirement rule; one-way, counted."""

        classical_survived = any(
            self.credit.arm(arm.name).survived > 0
            for arm in self.arms if arm.kind == "classical"
        )
        newly = []
        for arm in self.active_arms():
            if arm.kind == "classical":
                continue
            credit = self.credit.arm(arm.name)
            if (credit.measured >= 4 and credit.survived == 0
                    and classical_survived):
                self.retired.add(arm.name)
                self.counters.retired += 1
                newly.append(arm.name)
        return tuple(newly)

    def summary(self) -> Dict[str, Any]:
        return {
            "arms": {
                name: [credit.proposed, credit.measured, credit.survived]
                for name, credit in sorted(self.credit.arms.items())
            },
            "retired": sorted(self.retired),
        }
