"""A search loop that actually maintains a population and recombines it.

The loop in :mod:`agent_evolve.session.loop` renders the Pareto front as text
and asks a model to author whole configurations. That is whole-artifact rewrite;
it has no parent selection, no recombination, no mutation and no survival rule,
and it is measurably worse than uniform random sampling on every genome length
measured (advantage_theory sweep, 2026-08-03).

This loop supplies those operators. The seam that matters is
:class:`OperatorChooser`: something outside the loop decides *which parents* and
*where to cut*, and the loop never learns what made that decision. A random
chooser is the unguided control; a model-backed chooser is guidance. Neither can
author a candidate, because the chooser's return type cannot express one -- the
distinction is enforced by the type, not by convention.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Set)

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import SearchResult, dominates
from agent_evolve.core.telemetry import harvest_telemetry
from agent_evolve.policies.genetic import (
    EliteMixture,
    Locus,
    coverage_candidate,
    coverage_counts,
    crossover,
    incumbent_candidate,
    local_probe_candidate,
    loci_of,
    read_locus,
    mutate,
    one_mutation_neighbourhood,
    truncation_survival,
    uniform_candidate,
)
from agent_evolve.session.evaluate import EvaluationCache, evaluate_batch

__all__ = ["OperatorChoice", "OperatorChooser", "GeneticConfig", "run_genetic_loop",
           "random_chooser", "domination_rank", "explore_probability",
           "best_member"]

Config = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class OperatorChoice:
    """One offspring, expressed as operator arguments rather than a candidate.

    There is deliberately no field that can hold a configuration. A chooser that
    wanted to author a genome could not say so through this type.
    """

    parent_a: int
    parent_b: int
    mask: tuple[bool, ...]
    mutate_loci: Optional[tuple[Locus, ...]] = None


#: Given the ranked population, how many offspring are wanted, and the search
#: state accumulated so far, return that many operator choices. Indices address
#: the population list. The state argument is explicit rather than captured so
#: that a chooser's inputs are visible at the call site and an ablation can be
#: read off the signature.
OperatorChooser = Callable[
    [Sequence[tuple[Config, float]], int, Any], Sequence[OperatorChoice]
]


@dataclass(slots=True)
class GeneticConfig:
    population_size: int = 8
    offspring_per_generation: int = 6
    generations: int = 5
    mutation_rate: Optional[float] = None      # None -> 1/n_loci
    seed: Optional[int] = None
    seeds: tuple = ()
    evaluation_budget: Optional[int] = None
    evaluation_cache: EvaluationCache = field(default_factory=EvaluationCache)
    #: What the chooser may reason over. Left None for the score-only baseline.
    state: Any = None
    #: Evaluations to spend on a screening design before the population is
    #: built. 0 leaves the loop byte-identical to the pre-structure seam. The
    #: screen is charged against the same budget as everything else: a phase
    #: that spent free evaluations would not be a real operating point.
    structure_budget: int = 0
    #: Turns the screen's evidence into a prior. Defaults to the credential-free
    #: statistical rule, so the phase is useful with no model at all -- and so a
    #: model-proposed prior always has a rule to beat.
    prior_proposer: Any = None
    #: Pool sequence positions by field in the structure phase: the screen
    #: becomes per-value pure/spiked designs and attribution counts every
    #: (candidate, position) pair as one observation. The exchangeability
    #: bet this makes is checked by the same unwind test as any prior.
    structure_pooled: bool = False
    #: A prior over WHERE to sample: narrows the declared domain of any locus it
    #: names, for initialization and mutation alike. Left None, every sampler
    #: sees exactly the schema's own domains, so the loop is byte-identical to
    #: the pre-restriction seam. This is guidance over the sampling
    #: distribution rather than over operator choice within a fixed one.
    restriction: Any = None
    #: Virtual pre-screening (a session.screening.Screening). Each generation
    #: the loop builds pool_factor times the offspring it can afford, asks the
    #: screen's validated surrogate to order the pool, and measures the
    #: exploration floor plus the top of the order. None -- the default --
    #: leaves the loop byte-identical to the pre-screening seam; the pool's
    #: extra construction runs on its own RNG stream for the same reason.
    screening: Any = None
    #: Model-proposed initial population members, validated value-by-value
    #: upstream. Inserted AFTER the caller's seeds and before schema-uniform
    #: fill; () -- the default -- is byte-identical to the pre-init seam.
    initial_proposals: tuple = ()
    #: Variation-arm portfolio (a policies.operator_portfolio
    #: .OperatorPortfolio). When present it constructs the generation's
    #: offspring -- classical, rule, and authored arms under survival credit
    #: -- and the loop reports each measured child's fate back to it. None
    #: leaves offspring construction byte-identical to the classical path.
    portfolio: Any = None
    #: A model-authored SAMPLER (a policies.llm_generator.AuthoredGenerator).
    #: When present it draws the whole generation's candidate POOL -- many
    #: times the offspring the budget can afford -- and the pool then goes
    #: through the same screening path as any other pool, or is taken from
    #: the top when there is no screen. Mass generation charges nothing:
    #: the generator never sees the problem or the cache, and only the
    #: `want` candidates handed to measure() can reach the budget. None --
    #: the default -- leaves the loop byte-identical to the pre-generator
    #: seam. It replaces offspring construction, so it does not compose with
    #: `portfolio`; asking for both is refused rather than silently ignored.
    generator: Any = None
    #: Whether the screen may spend the problem's CHEAPER evaluation fidelity
    #: (``Problem.evaluate_proxy``), and how. "off" leaves the loop
    #: byte-identical to the pre-proxy seam and is what every problem without
    #: a cheap fidelity gets regardless. "rows" buys gate evidence at the
    #: cheap fidelity, "screen" lets the cheap fidelity compete as a
    #: surrogate under the gate, "both" does both. Proxy evaluations are
    #: counted in their own ledger and are NEVER charged to
    #: ``evaluation_budget``: the budget the claims are denominated in counts
    #: real evaluations, and a cheap one is not one of them.
    #:
    #: The default is the MEASURED arm: "screen" won 23/24 seeds against the
    #: no-proxy screen pooled over both budgets (12/0 at B=24, p = 0.00049;
    #: 11/12 at B=16, p = 0.006; wave-K aug14_multifidelity.md), with the
    #: shuffled-proxy control identical to no-proxy in all 12 runs -- the
    #: gain is the cheap MEASUREMENT, not the machinery. The seam is active
    #: iff the problem exposes ``evaluate_proxy``: ``ProxySource.for_problem``
    #: returns None otherwise and nothing attaches, so every problem without
    #: a cheap fidelity keeps the pre-proxy behaviour bit for bit.
    proxy_fidelity: str = "screen"
    #: A hard ceiling on proxy evaluations for the whole run. None -- no
    #: ceiling. A consumer that exhausts it degrades to "no proxy".
    proxy_ceiling: Optional[int] = None
    #: MEASUREMENT-CONDITIONED REVISION of the sampling prior (a
    #: policies.reguidance.Reguidance). Once per generation the policy is
    #: asked whether its declared cadence has come round; when it has, one
    #: model call reads what the run measured and returns a re-weighted prior
    #: -- damped into the one in force, so a revision can tilt the draws and
    #: cannot exclude anything -- plus, when it is bought, a few complete
    #: configurations to measure next. None -- the default -- is
    #: byte-identical to the pre-seam loop: no call fires and no counter
    #: moves. The prior it revises is the same one initialization and
    #: mutation already consult, so one revision reshapes every subsequent
    #: draw with no further calls.
    reguidance: Any = None
    #: ENDGAME POLISH. A breeding loop that has already found the right region
    #: keeps recombining a front it has solved, and the last grid step never
    #: gets tried: measured on ten NAS seeds, NSGA-II reaches the EXACT
    #: optimum 6W/4L against us while we reach within 10% 9W/1L, and it does
    #: it by spending its endgame on cache-fuelled mass proposals around the
    #: front. "sweep" answers with the deterministic version of that: when the
    #: front has stopped moving and enough budget is still in hand, the
    #: generation proposes the 1-mutation neighbourhood of a rank-0 member
    #: instead of breeding. It is classical -- no model call, no RNG draw.
    #: "off" -- the default -- is byte-identical to the pre-polish seam:
    #: nothing is tracked, nothing is deduped and nothing is drawn.
    polish: str = "off"
    #: Consecutive CHARGES with an unchanged rank-0 front before a sweep may
    #: engage. 0 means auto: twice the population size, which is roughly the
    #: charges two generations of a converged population spend re-measuring
    #: what it already holds. Any change to the front's objective-vector set
    #: resets the count -- a front that is still moving is not stalled.
    polish_after: int = 0
    #: The fraction of the budget that must still be unspent for a sweep to
    #: engage. A neighbourhood enumeration is only worth buying while there is
    #: enough left to measure a useful part of it; below the reserve the
    #: generation breeds as usual. Ignored when there is no budget to reserve
    #: a fraction OF.
    polish_reserve: float = 0.15
    #: DECLARED-DOMAIN EXPLORATION. "coverage" spends some of each
    #: generation's offspring slots on a fresh configuration drawn from each
    #: locus's LEAST-MEASURED DECLARED values -- counted over the schema's
    #: whole domain and never over the prior's narrowed support. "off" -- the
    #: default -- is byte-identical to the pre-exploration seam: no slot is
    #: taken, no RNG stream is created, no key is tracked and no counter moves.
    #:
    #: The measurement it answers. On the analog venue every installed prior
    #: allowed at most level 13-15 of a 23-32 level ladder; pooling 12,899 real
    #: evaluations, the best inside that box reads -0.4002 reward9 and the best
    #: outside it -0.0566, and 100 of the pooled top-100 carry at least one
    #: coordinate the box excludes. HEBO reaches 95.6% coverage of the declared
    #: (field, value) cells by charge 80; our arms stall at 72.1%, and the
    #: frozen arm adds no new declared value at all after the screen
    #: (INVESTIGATION.md, S5).
    #:
    #: It is HALF a fix, and the other half is `intensify` below: the two of
    #: our cells that already spend 82-96% of their charges outside the box
    #: come back with -0.457 and -0.452, no better than the cells that stay in.
    #: Leaving the box without intensification pays nothing.
    explore: str = "off"
    #: ``(e0, e_min)``: the probability that any one offspring slot is an
    #: exploration draw, declining LINEARLY in the fraction of the evaluation
    #: budget already charged. Both ends are declared rather than tuned; a run
    #: with no evaluation budget has no horizon to decline over and holds at
    #: ``e0``. See :func:`explore_probability`.
    explore_schedule: tuple = (0.5, 0.1)
    #: INCUMBENT INTENSIFICATION. "incumbent" spends a fraction of each
    #: generation's offspring slots -- after the exploration slots -- on the
    #: current best member with a seeded random subset of its loci PINNED and
    #: the rest resampled through the prior in force. "off" -- the default --
    #: is byte-identical to the pre-intensification seam, on the same terms as
    #: `explore`.
    #:
    #: This is what "coordinated combinations" turned out to be when the
    #: populations that motivated it were actually measured: HEBO's winners are
    #: ONE incumbent-anchored cluster (single-linkage at Hamming <= 10 gives
    #: exactly one cluster in 6 of 6 cells), with zero of 156
    #: entropy-controlled coordinate pairs above z = 2 and a modal purity of
    #: 0.695-0.842 -- a product of marginals pinned to an incumbent, not a
    #: couplings table. And what it does that we do not is collapse: mean
    #: |delta level| to the incumbent 4.21 -> 0.24 across budget quartiles,
    #: against our 3.61 -> 2.98 (INVESTIGATION.md, S5). Classical, zero model
    #: calls.
    intensify: str = "off"
    #: The share of a generation's offspring slots an intensification may take,
    #: FLOORED to a whole slot and capped by what exploration left. 0.25 of six
    #: offspring is one slot, which is the point: intensification is a bet
    #: placed beside the generation, not instead of it.
    intensify_fraction: float = 0.25
    #: ``(low, high)``: how many loci one intensified child pins, drawn uniform
    #: on the closed interval and clamped to the genome's own length. The band
    #: is HEBO's measured modal-purity band over 24 loci -- 6 to 12 fields --
    #: rather than a tuned pair.
    intensify_pin_range: tuple = (6, 12)
    #: CONSOLIDATION BURST, a sub-mode of `intensify` (requires "incumbent";
    #: 0 -- the default -- changes nothing, byte for byte). When any
    #: evaluation strictly improves any single objective's best-so-far, the
    #: next N intensification slots probe THAT member's one-locus
    #: neighbourhood (:func:`local_probe_candidate`) instead of pin-resampling
    #: the ranked incumbent; a newer advance retargets and refills the count.
    #: No extra charges: the burst retargets slots intensification already
    #: holds.
    #:
    #: The measured defect this answers: both runs that ever reached the
    #: analog venue's fully-feasible plateau then spent 3.4% and 12.4% of
    #: their remaining charges on it, while all 44 one-step neighbours of the
    #: discovery measured on-plateau -- because an intensified child resamples
    #: its unpinned loci through a prior that repels from a region it never
    #: predicted, and elite offspring mutate through the same prior. The probe
    #: is prior-free and one declared step wide on one locus. Advances are
    #: front-loaded (24-42 per 320-charge run, 0-7 in the last quarter), so
    #: the burst displaces early intensification -- measured NEGATIVE alone
    #: (5W/7L, median -0.046) -- and costs late budget nearly nothing where
    #: advances are rare.
    intensify_burst: int = 0
    #: ELITE MIXTURE at the value draw: with this weight, mutation and
    #: intensification's unpinned resamples draw each locus from a mixture of
    #: the prior in force and the CURRENT front's empirical values (see
    #: :class:`EliteMixture` for the measured story: the prior repels from
    #: discoveries it never predicted, and no trigger can tell a jackpot from
    #: a routine advance at entry time). 0.0 -- the default -- constructs
    #: nothing and is byte-identical. The mixture only ever rides a prior:
    #: with no restriction in force, draws are untouched.
    elite_mix: float = 0.0
    #: ACTUATION BANDWIDTH (REFINEMENT_ROUND.md X4). "fresh-boost" raises the
    #: mutation rate to min(1/3, 8/n_loci) while an installed or revised
    #: prior is FRESH (its first G generations, G = max(4, budget //
    #: (4*offspring))), reverting to the 1/n default after. The trace
    #: forensics measured the 1/n seam as the ceiling on how fast guidance
    #: can reach evaluations (fresh-draw rate 0.0373 against the 1/24
    #: ceiling; post-revision distribution shifts statistically identical to
    #: untouched loci). "off" -- the default -- is byte-identical.
    actuation: str = "off"
    #: How survival breaks ties inside one domination count. "count" -- the
    #: default -- keeps the measurement order and is byte-identical.
    #: "crowding" prefers the members that are most alone in objective space
    #: (NSGA-II crowding distance), which is what makes a many-objective unit
    #: selectable at all: on the five-objective fleet unit almost nothing
    #: dominates anything, so a count-only rule is near-random.
    survival: str = "count"


def domination_rank(
    objectives_of: Sequence[Mapping[str, float]],
    specs: Sequence[ObjectiveSpec],
) -> List[float]:
    """How many population members dominate each one. Lower is better.

    Pareto-correct and weight-free. A scalarization would need weights nobody
    declared, and an undeclared weight is precisely the kind of hidden choice
    that has silently decided results in this project before.
    """

    def dominates(x: Mapping[str, float], y: Mapping[str, float]) -> bool:
        better_anywhere = False
        for spec in specs:
            # spec.goal, read directly rather than via a default: a missing goal
            # is a contract violation, and defaulting it to "min" would silently
            # invert every maximised objective instead of failing.
            sign = 1.0 if spec.goal == "min" else -1.0
            xi, yi = sign * float(x[spec.name]), sign * float(y[spec.name])
            if xi > yi:
                return False
            if xi < yi:
                better_anywhere = True
        return better_anywhere

    return [
        float(sum(1 for other in objectives_of if dominates(other, this)))
        for this in objectives_of
    ]


#: How many times a slot may redraw before it is given back to breeding. A
#: mechanism that cannot produce anything the run has not already measured has
#: nothing to say this generation, and saying it eight times is enough to
#: establish that; the slot then breeds rather than shrinking the generation,
#: because an arm that spends fewer charges than its control is not a
#: comparison.
_DEDUP_ATTEMPTS = 8


def explore_probability(
    schedule: Sequence[float], *, spent: int, budget: Optional[int]
) -> float:
    """The per-slot exploration probability after *spent* of *budget* charges.

    Linear in the fraction of the budget already charged: ``e0`` at the first
    charge, ``e_min`` at the last, exactly halfway between them at half the
    budget, and pinned at ``e_min`` by a run that somehow charges past its own
    budget. A run with no declared budget has nothing to decline over and holds
    at ``e0`` rather than inventing a horizon -- the same rule the polish
    reserve uses for the same reason.

    A declining schedule rather than a constant because the measured target
    declines: HEBO's rate of proposing a value it has never measured is 0.288
    in its first budget quartile and 0.001 in every later one
    (INVESTIGATION.md, S5). Exploration is a phase, and a phase that never ends
    is a uniform sampler wearing a schedule.
    """

    e0, e_min = float(schedule[0]), float(schedule[1])
    if budget is None or budget <= 0:
        return e0
    progress = min(1.0, max(0.0, float(spent) / float(budget)))
    # Written as the convex combination rather than `e0 + (e_min - e0) * p`,
    # which is the same line of algebra and not the same arithmetic: the second
    # form returns 0.09999999999999998 at the end of a (0.5, 0.1) schedule, so
    # the declared floor would be a number the function never actually reaches.
    return e0 * (1.0 - progress) + e_min * progress


def best_member(
    population: Sequence[tuple[Config, Mapping[str, float]]],
    ranks: Sequence[float],
    specs: Sequence[ObjectiveSpec],
) -> Optional[Config]:
    """The incumbent: the rank-0 member best on the FIRST declared objective.

    A population with several objectives has no single best member, so the one
    an intensification anchors on has to be CHOSEN, and the choice is declared
    here rather than left to whatever order survival happened to produce.
    Among the members nothing dominates, the one with the best value on
    ``specs[0]``, read through that objective's own goal; among those, the one
    measured EARLIEST, because population order is measurement order within a
    rank and a first-wins tiebreak makes the anchor reproducible from the seed
    alone. Returns None for an empty population, which is the only case with no
    incumbent to name.
    """

    if not population or not specs:
        return None
    sign = 1.0 if specs[0].goal == "min" else -1.0
    best: Optional[tuple[float, Config]] = None
    for (config, objectives), rank in zip(population, ranks):
        if rank != 0:
            continue
        value = sign * float(objectives[specs[0].name])
        if best is None or value < best[0]:    # strict: the earliest wins ties
            best = (value, config)
    return None if best is None else best[1]


def _pair(value: Any, name: str) -> tuple:
    """A knob declared as two numbers, read as two numbers or refused by name."""

    try:
        items = tuple(value)
    except TypeError:
        items = ()
    if len(items) != 2:
        raise ValueError(f"{name} must be a pair, got {value!r}")
    return items


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def random_chooser(rng: random.Random, n_loci: int) -> OperatorChooser:
    """The unguided control: tournament parents, uniform mask, random loci."""

    def choose(population: Sequence[tuple[Config, float]], count: int,
               state: Any = None):
        del state                       # the unguided control reasons over nothing
        out: List[OperatorChoice] = []
        for _ in range(count):
            def pick() -> int:
                a, b = rng.sample(range(len(population)), min(2, len(population))) \
                    if len(population) >= 2 else (0, 0)
                return a if population[a][1] <= population[b][1] else b

            out.append(
                OperatorChoice(
                    parent_a=pick(),
                    parent_b=pick(),
                    mask=tuple(bool(rng.getrandbits(1)) for _ in range(n_loci)),
                )
            )
        return out

    return choose


def run_genetic_loop(
    *,
    problem: Any,
    config: GeneticConfig,
    chooser: Optional[OperatorChooser] = None,
    log: Callable[[str], None] = lambda _m: None,
) -> SearchResult:
    """Evolve a population under *problem*, spending at most the budget."""

    from agent_evolve.session.loop import _build_search_result, _default_candidate_key

    if config.generator is not None and config.portfolio is not None:
        raise ValueError(
            "generator and portfolio both construct the generation's "
            "candidates: the generator draws the pool, the portfolio "
            "recombines parents into it. Run one or the other."
        )
    if config.polish not in ("off", "sweep"):
        raise ValueError(
            f"polish must be 'off' or 'sweep', got {config.polish!r}")
    if config.survival not in ("count", "crowding"):
        raise ValueError(
            f"survival must be 'count' or 'crowding', got {config.survival!r}")
    if config.explore not in ("off", "coverage"):
        raise ValueError(
            f"explore must be 'off' or 'coverage', got {config.explore!r}")
    if config.intensify not in ("off", "incumbent"):
        raise ValueError(
            f"intensify must be 'off' or 'incumbent', got {config.intensify!r}")
    polish_on = config.polish == "sweep"
    explore_on = config.explore == "coverage"
    intensify_on = config.intensify == "incumbent"
    # The numeric knobs are validated whether or not their mechanism is on. A
    # nonsense schedule sitting quietly beside `explore="off"` is a bug waiting
    # for the day someone turns the mechanism on, and it costs nothing to say
    # so now.
    schedule = _pair(config.explore_schedule, "explore_schedule")
    if not all(_is_number(x) and 0.0 <= float(x) <= 1.0 for x in schedule):
        raise ValueError(
            "explore_schedule must be (e0, e_min), two probabilities in "
            f"[0, 1], got {config.explore_schedule!r}")
    if not _is_number(config.intensify_fraction) or not (
            0.0 <= float(config.intensify_fraction) <= 1.0):
        raise ValueError(
            "intensify_fraction must be a fraction in [0, 1], got "
            f"{config.intensify_fraction!r}")
    pin_low, pin_high = _pair(config.intensify_pin_range, "intensify_pin_range")
    if not all(isinstance(x, int) and not isinstance(x, bool)
               for x in (pin_low, pin_high)) or pin_low < 1 or pin_high < pin_low:
        raise ValueError(
            "intensify_pin_range must be (low, high), two integers with "
            f"1 <= low <= high, got {config.intensify_pin_range!r}")
    if not isinstance(config.intensify_burst, int) or isinstance(
            config.intensify_burst, bool) or config.intensify_burst < 0:
        raise ValueError(
            "intensify_burst must be an integer >= 0, got "
            f"{config.intensify_burst!r}")
    if config.intensify_burst > 0 and not intensify_on:
        # A burst is a retargeting of intensification's own slots; without
        # them it would be a silent no-op, and this package refuses those.
        raise ValueError(
            "intensify_burst requires intensify='incumbent': the burst "
            "retargets intensification slots and there are none to retarget")
    burst_on = intensify_on and config.intensify_burst > 0
    if not _is_number(config.elite_mix) or not (0.0 <= float(config.elite_mix) < 1.0):
        raise ValueError(
            f"elite_mix must be a mixing weight in [0, 1), got "
            f"{config.elite_mix!r}")
    mixer = (EliteMixture(float(config.elite_mix))
             if float(config.elite_mix) > 0.0 else None)
    if config.actuation not in ("off", "fresh-boost"):
        raise ValueError(
            f"actuation must be 'off' or 'fresh-boost', got {config.actuation!r}")
    boost_on = config.actuation == "fresh-boost"

    specs = list(problem.objectives)
    candidate_model = getattr(problem, "candidate_model", None)
    rng = random.Random(config.seed)
    cache = config.evaluation_cache
    if config.evaluation_budget is not None:
        cache.budget = config.evaluation_budget

    seeds = [dict(c) for c in (config.seeds or tuple(problem.seeds()))]
    if not seeds:
        raise ValueError(
            "the genetic loop needs at least one seed to know the shape of a "
            "candidate. Give Problem.seeds() one configuration, or use "
            "proposer='llm' with the authoring loop."
        )

    def spent() -> int:
        return int(getattr(cache, "misses", 0))

    def budget_left() -> int:
        if config.evaluation_budget is None:
            return 1 << 30
        return max(0, config.evaluation_budget - spent())

    from agent_evolve.policies.search_state import SearchState

    # --- the cheaper evaluation fidelity, if the problem has one -----------
    # The source holds the problem's `evaluate_proxy` bound method and nothing
    # else -- no problem, no cache, no budget -- so a proxy evaluation has no
    # route to a charge. It is attached to the SCREEN, which is the only
    # consumer that can use cheap evidence without putting it in the archive.
    proxy_source: Any = None
    if config.proxy_fidelity != "off" and config.screening is not None:
        from agent_evolve.session.fidelity import ProxySource

        proxy_source = ProxySource.for_problem(
            problem, ceiling=config.proxy_ceiling)
        if proxy_source is not None:
            config.screening.attach_proxy(
                proxy_source, mode=config.proxy_fidelity)

    all_valid: List[Any] = []
    all_meta: List[tuple] = []
    history: List[Dict[str, Any]] = []
    # The state accumulates across generations and is handed to the chooser
    # each time. The unguided chooser ignores it, which is what makes the two
    # arms differ in exactly one thing.
    state = config.state if config.state is not None else SearchState()
    state.history = history

    # What the run has already put in front of the evaluator, by configuration
    # identity. The evaluation cache keys the MATERIALIZED artifact, which the
    # loop cannot compute for a candidate it has not built yet, so every
    # mechanism that has to dedup its own proposals reads this instead: every
    # configuration handed to measure(), whether it was charged, served from
    # cache or refused. Populated only under the mechanisms that need it, so
    # all of them off leaves no side effect behind at all.
    seen_keys: Set[str] = set()
    track_seen = polish_on or explore_on or intensify_on

    def measure(configs: List[Config], gen: int) -> List[Any]:
        room = budget_left()
        if room <= 0:
            return []
        if track_seen:
            seen_keys.update(_default_candidate_key(c) for c in configs[:room])
        valid, failed, _ordered = evaluate_batch(
            problem, configs[:room], specs, cache=cache
        )
        all_valid.extend(valid)
        for result in list(valid) + list(failed):
            all_meta.append((result, {"generation": gen}))
        # Every measured candidate feeds the per-locus table, including ones the
        # population does not keep: what was tried and rejected is exactly the
        # evidence that a locus is saturated.
        state.evaluated.extend((r.configuration, dict(r.objectives)) for r in valid)
        # --- actionable side information ----------------------------------
        # Failures verbatim: what the validator or evaluator said is exactly
        # the diagnostic the score-only condition throws away.
        for result in failed:
            message = getattr(result, "error_message", None)
            if message:
                state.side_information.append(f"rejected: {message}")
        # Optional problem hook -- an opt-in sixth obligation, never required.
        hook = getattr(problem, "side_information", None)
        if callable(hook):
            for result in valid:
                try:
                    text = hook(result.configuration, dict(result.objectives))
                except Exception as exc:            # a hook must not kill a run,
                    state.side_information.append(  # but must not vanish either
                        f"side_information hook raised {type(exc).__name__}: {exc}"
                    )
                    break
                if text:
                    state.side_information.append(str(text))
        del state.side_information[:-64]            # bounded, newest kept
        return valid

    def remember_measured(results: Sequence[Any],
                          surviving: Optional[Set[str]] = None) -> None:
        """Report charged measurements to the generator as EVIDENCE.

        Every charge the run makes, whoever proposed it. A generator that
        reasons over measurements is reasoning about the SPACE, and the space
        does not care which component produced the point: the initial
        population and the structure screen are measurements this run paid
        for, and withholding them leaves the channel blind until the loop has
        spent two generations reproducing evidence it already had. That was
        W11, measured: the locus prior could not be authored before a median
        charge of 40 on a venue whose dominant knob is legible by charge 19.

        Attribution is the separate call: a generator is credited only with
        the children it drew, so its survival counters -- and the revision and
        unwind rules that read them -- are untouched by what it is shown.
        """

        generator = config.generator
        note = getattr(generator, "note_measured", None)
        if generator is None or not results or note is None:
            return
        kept = surviving or set()
        for result in results:
            note(result.configuration,
                 objectives=dict(result.objectives),
                 survived=_default_candidate_key(result.configuration) in kept)

    # --- initial population: the seeds, then SCHEMA-UNIFORM draws -----------
    # Not mutants of the seed. A population of near-copies of one
    # configuration is an anchored cloud around it; measured on a third-party
    # optimizer, correcting exactly this anchor moved its result from +0.095
    # (loses badly to uniform) to +0.0066 (parity), and the standard seed on
    # log2 scores worse than a typical uniform draw.
    n_loci = len(loci_of(seeds[0]))

    # --- optional structure phase: buy a model of the landscape first -------
    # Operator choice cannot escape the distribution it samples from, so before
    # building a population the loop can spend a few evaluations on a CROSSED
    # screen, read which locus values the evidence refutes, and narrow the
    # domains every later draw sees. The screen costs real budget and the prior
    # can be wrong, so both are recorded and the bet is checked below.
    restriction = config.restriction
    screen_front: List[Mapping[str, float]] = []
    structure_record: Dict[str, Any] = {}
    prior_proposer_used: Any = None
    if config.structure_budget > 0 and restriction is None:
        from agent_evolve.policies.structure import (
            attribute, crossed_screen, statistical_prior)

        screened = crossed_screen(seeds[0], candidate_model,
                                  size=config.structure_budget, rng=rng,
                                  pool_by_field=config.structure_pooled)
        screen_valid = measure(screened, 0)
        # Charged, therefore evidence. The screen's points never enter a
        # population, so none of them is marked survived -- an absent verdict
        # reported as absent rather than invented.
        remember_measured(screen_valid)
        if screen_valid:
            attr = attribute(
                [(r.configuration, dict(r.objectives)) for r in screen_valid],
                specs, candidate_model,
                pool_by_field=config.structure_pooled)
            propose = config.prior_proposer or statistical_prior
            prior_proposer_used = propose
            try:
                restriction = propose(attr, candidate_model)
            except Exception as exc:        # a proposer must not kill a run
                structure_record["proposer_error"] = f"{type(exc).__name__}: {exc}"
                restriction = None
            # Keep what the screen itself achieved, as objective vectors: the
            # unwind test below asks whether the restricted search can still
            # match it, and domination is the only comparison that means
            # anything across several objectives.
            # Keep the screen's non-dominated points that the prior EXCLUDES.
            # Those are exactly the claims the prior makes: it asserts this
            # region is not worth sampling. If the restricted search cannot
            # beat even one of them, the assertion is unsupported and the bet
            # comes off. Pooled rank-0 would be the weaker test and a useless
            # one -- a restricted region always holds points that are
            # non-dominated along some other objective, so it could never fire.
            screen_ranks = domination_rank(
                [dict(r.objectives) for r in screen_valid], specs)
            allowed = dict(getattr(restriction, "allowed", {}) or {})

            def _excluded(cfg: Mapping[str, Any]) -> bool:
                return any(cfg.get(k) not in tuple(vals)
                           for k, vals in allowed.items())

            screen_front = [dict(r.objectives)
                            for r, rank in zip(screen_valid, screen_ranks)
                            if rank == 0 and _excluded(r.configuration)]
            structure_record.update(
                screened=len(screened), evaluated=len(screen_valid),
                allowed=dict(getattr(restriction, "allowed", {}) or {}),
                misses=list(getattr(restriction, "misses", []) or []),
            )
            log(f"structure: screened {len(screen_valid)}, prior "
                f"{structure_record.get('allowed') or 'none'}")

    initial = list(seeds)
    for proposal in config.initial_proposals:
        if len(initial) < config.population_size:
            initial.append(dict(proposal))
    while len(initial) < config.population_size:
        template = initial[rng.randrange(len(initial))]
        initial.append(uniform_candidate(template, candidate_model, rng=rng,
                                         restriction=restriction))
    valid = measure(initial, 0)
    if not valid:
        raise RuntimeError(
            "no seed evaluated successfully, so there is nothing to evolve from"
        )

    # The population carries OBJECTIVES, not ranks: a rank is only meaningful
    # relative to the set it was computed in, so it must be recomputed whenever
    # the set changes rather than carried forward from a previous generation.
    def survive(
        pool: List[tuple[Config, Mapping[str, float]]]
    ) -> List[tuple[Config, Mapping[str, float]]]:
        ranks = domination_rank([obj for _c, obj in pool], specs)
        kept = truncation_survival(
            [((c, o), r) for (c, o), r in zip(pool, ranks)],
            keep=config.population_size,
            key_of=lambda pair: _default_candidate_key(pair[0]),
            method=config.survival,
            objectives_of=lambda pair: pair[1],
        )
        return [pair for pair, _rank in kept]

    population = survive([(r.configuration, dict(r.objectives)) for r in valid])
    history.append({"gen": 0, "valid_count": len(valid), "pop": len(population)})
    log(f"generation 0: {len(valid)} evaluated, population {len(population)}")
    if config.generator is not None:
        # What the run has already measured, so the novelty guard can tell a
        # candidate that is new from one the generator is re-proposing.
        config.generator.note_archive([r.configuration for r in valid])
    # ... and what those measurements SAID. The initial population is the only
    # evidence in existence when the first pool is drawn; a channel that cannot
    # see it cannot speak until generation 2, which is the W11 defect.
    remember_measured(valid,
                      {_default_candidate_key(c) for c, _o in population})

    pick = chooser or random_chooser(rng, n_loci)
    # Guidance picks the revision channel returned last generation. They are
    # authored members, like the initial proposals, so they wait for a
    # generation of their own rather than displacing offspring already built.
    immigrants: List[Config] = []
    # Pool extras draw from their own stream: the main stream must spend
    # exactly the same draws whether or not screening is on, or "off" stops
    # being byte-identical to the pre-screening seam.
    rng_pool = random.Random(0 if config.seed is None else (config.seed ^ 0x5CEE11))
    # Each slot-taking mechanism draws from its OWN stream, for the same reason
    # the pool does, and one reason more: the 2x2 ablation that measures them
    # (base / +explore / +intensify / both) must not have one arm's draws move
    # because the other arm was switched on. A stream is created only when its
    # mechanism is on, so "off" constructs nothing.
    rng_explore = (random.Random(0 if config.seed is None
                                 else (config.seed ^ 0xC0FFEE))
                   if explore_on else None)
    rng_intensify = (random.Random(0 if config.seed is None
                                   else (config.seed ^ 0x1CE111))
                     if intensify_on else None)

    # The prior the DRAW SEAMS see: the elite mixture over the restriction in
    # force, or the restriction itself when the mixture is off. Recomputed at
    # every point the restriction rebinds (unwind, revision), because the
    # mixture rides the prior rather than replacing it -- restriction=None
    # keeps the pre-prior stream untouched, mixture or no mixture.
    draw_prior = restriction if mixer is None else mixer.over(restriction)
    cage_streak = 0                     # X2b: consecutive gens best-outside
    #: X4: the generation through which the CURRENT prior counts as fresh.
    boost_gens = max(4, (config.evaluation_budget or 0)
                     // max(1, 4 * config.offspring_per_generation))
    boost_until = boost_gens if (boost_on and restriction is not None) else -1

    # --- consolidation-burst state ------------------------------------------
    # Pure bookkeeping until a burst is live: the tracker reads results and
    # draws nothing, so with the knob at 0 nothing below constructs, consumes,
    # or records anything and "off" stays byte-identical. The bests seeded
    # here are the run's starting bests -- screen rows and the initial
    # population included -- so a burst answers ADVANCES made by the search,
    # which is the measured walk-away defect, not the init pool's luck.
    burst_best: Dict[str, float] = {}
    burst_anchor: Optional[Config] = None
    burst_left = 0
    if burst_on:
        for _row, _objectives in state.evaluated:
            for spec in specs:
                value = _objectives.get(spec.name)
                if value is None:
                    continue
                value = float(value)
                held = burst_best.get(spec.name)
                if held is None or (value > held if spec.goal == "max"
                                    else value < held):
                    burst_best[spec.name] = value

    # --- endgame polish state ------------------------------------------------
    # The stall tracker reads the population and the charge counter and draws
    # nothing: a tracker that touched the RNG would make "off" and "sweep"
    # differ in the control arm's stream, which is the one thing they may not
    # do. `polish_front` is the front's objective-vector SET, so a front that
    # trades one member for another with the same vector is correctly read as
    # unchanged, and a front that actually moved resets the count.
    polish_front: Optional[frozenset] = None
    polish_stall = 0                    # charges since the front last moved
    polish_mark = spent()
    polish_turn = 0                     # the rank-0 member a sweep starts at
    polish_threshold = (config.polish_after if config.polish_after > 0
                        else 2 * config.population_size)

    for gen in range(1, config.generations + 1):
        if budget_left() <= 0:
            log(f"budget exhausted after {spent()} evaluations; stopping at gen {gen}")
            break
        want = min(config.offspring_per_generation, budget_left())
        # The chooser sees ranks, not raw objectives: it decides which parents to
        # combine, and a rank is the comparable form of "how good is this one".
        ranks = domination_rank([obj for _c, obj in population], specs)
        ranked = [(c, r) for (c, _o), r in zip(population, ranks)]
        if mixer is not None:
            # The mixture's elite component: last survival's rank-0 members.
            mixer.front_rows = tuple(
                c for (c, _o), rank in zip(population, ranks) if rank == 0)
        filled = 0
        choices: Sequence[OperatorChoice] = ()

        # --- endgame polish: enumerate, once breeding has stopped paying -----
        # Everything here is deterministic. The sweep is the 1-mutation
        # neighbourhood of a rank-0 member in a fixed order, minus what the run
        # has already put in front of the evaluator, and a generation that
        # sweeps makes no operator choice at all -- so no chooser is consulted
        # and no model call is spent on a decision the sweep does not use.
        polish_note: Optional[Dict[str, Any]] = None
        polish_kids: List[Config] = []
        polishing = False
        if polish_on:
            front_now = frozenset(
                tuple(sorted((name, float(value)) for name, value in obj.items()))
                for (_c, obj), rank in zip(population, ranks) if rank == 0)
            if front_now == polish_front:
                polish_stall += spent() - polish_mark
            else:
                polish_front = front_now
                polish_stall = 0
            polish_mark = spent()
            # No declared budget means nothing to hold a fraction of, so the
            # reserve cannot refuse: an unbounded run is all endgame.
            reserved = (config.evaluation_budget is None
                        or budget_left() >= config.polish_reserve
                        * config.evaluation_budget)
            member_index = -1
            if polish_stall >= polish_threshold and reserved:
                rank0 = [c for (c, _o), rank in zip(population, ranks)
                         if rank == 0]
                # Population order is measurement order within a rank, so the
                # rotation visits the front first-measured first and a run's
                # polish generations are reproducible from the seed alone.
                proposed: Set[str] = set()
                for step in range(len(rank0)):
                    member = rank0[(polish_turn + step) % len(rank0)]
                    for candidate in one_mutation_neighbourhood(
                            member, candidate_model, restriction=restriction):
                        key = _default_candidate_key(candidate)
                        if key in seen_keys or key in proposed:
                            continue
                        proposed.add(key)
                        polish_kids.append(candidate)
                        if len(polish_kids) >= want:
                            break
                    if len(polish_kids) >= want:
                        break
                if polish_kids:
                    member_index = polish_turn % len(rank0)
                    polish_turn += 1
                    polishing = True
            # An exhausted neighbourhood reports `engaged: false` and the
            # generation breeds: polish did not run, and a record that said it
            # had would be the silent no-op this package refuses everywhere
            # else.
            polish_note = {"engaged": polishing,
                           "proposed": len(polish_kids),
                           "member_index": member_index}

        # --- exploration and intensification: two slots, one rule ------------
        # Both mechanisms take offspring SLOTS out of this generation -- never
        # extra charges -- and the order is declared: exploration first, then
        # intensification of whatever is left, then normal breeding fills the
        # rest, so the generation spends exactly the charges it would have
        # spent anyway. Polish supersedes both: a sweep already IS the
        # generation's answer, and taking slots out of an enumerated
        # neighbourhood would leave the enumeration incomplete for no gain.
        # Neither draws a model call.
        extra_kids: List[Config] = []
        explore_count = 0
        intensify_count = 0
        burst_count = 0
        if (explore_on or intensify_on) and not polishing:
            proposed_keys: Set[str] = set()

            def admit(candidate: Config) -> bool:
                """Take *candidate* unless the run has already proposed it."""

                key = _default_candidate_key(candidate)
                if key in seen_keys or key in proposed_keys:
                    return False
                proposed_keys.add(key)
                extra_kids.append(candidate)
                return True

            if explore_on:
                probability = explore_probability(
                    schedule, spent=spent(),
                    budget=config.evaluation_budget)
                # Coverage is counted over the run's OWN trace -- every row it
                # charged, including the screen's and the initial population's
                # -- and against the DECLARED domain. See `coverage_counts`
                # for why the declared domain and not the allowed one.
                counts = coverage_counts(
                    [row for row, _objectives in state.evaluated],
                    seeds[0], candidate_model)
                for _slot in range(want):
                    if rng_explore.random() >= probability:
                        continue
                    for _attempt in range(_DEDUP_ATTEMPTS):
                        candidate = coverage_candidate(
                            seeds[0], candidate_model, counts=counts,
                            rng=rng_explore)
                        if admit(candidate):
                            explore_count += 1
                            # This generation's own draws count against the
                            # next slot's tally. Without that, two slots in one
                            # generation see the same least-measured values and
                            # propose the same configuration twice.
                            coverage_counts([candidate], seeds[0],
                                            candidate_model, counts=counts)
                            break

            if intensify_on:
                # Floored to a whole slot, and rounded to nine places first so
                # that the declared arithmetic is the arithmetic: 0.3 of ten
                # slots is 2.9999999999999996 in binary, and flooring that
                # gives two where three was asked for.
                slots = min(want - len(extra_kids),
                            int(round(float(config.intensify_fraction) * want,
                                      9)))
                incumbent = best_member(population, ranks, specs)
                if incumbent is not None:
                    anchor_loci = loci_of(incumbent)
                    for _slot in range(max(0, slots)):
                        # A live burst retargets this slot at the member that
                        # last advanced an objective best: one-locus probes of
                        # a discovery, instead of a pin-resample through a
                        # prior that repels from it. The counter burns per
                        # SLOT, not per admission, so an anchor whose
                        # neighbourhood the run has exhausted cannot pin the
                        # mechanism to a dead point for the rest of the run.
                        if burst_left > 0 and burst_anchor is not None:
                            burst_left -= 1
                            for _attempt in range(_DEDUP_ATTEMPTS):
                                candidate = local_probe_candidate(
                                    burst_anchor, candidate_model,
                                    rng=rng_intensify)
                                if admit(candidate):
                                    intensify_count += 1
                                    burst_count += 1
                                    break
                            continue
                        for _attempt in range(_DEDUP_ATTEMPTS):
                            # q is clamped to the genome: a pin band read off a
                            # 24-locus venue must not ask a 6-locus one to hold
                            # loci it does not have.
                            pinned = min(len(anchor_loci),
                                         rng_intensify.randint(pin_low, pin_high))
                            candidate = incumbent_candidate(
                                incumbent, candidate_model,
                                pin=rng_intensify.sample(anchor_loci, pinned),
                                rng=rng_intensify, restriction=draw_prior)
                            if admit(candidate):
                                intensify_count += 1
                                break

        # What breeding is left to fill. With both mechanisms off this is
        # `want` exactly, and every arithmetic below it is the arithmetic that
        # was there before they existed.
        want_bred = want - len(extra_kids)

        # An authored generator draws the whole generation, so there is no
        # parent choice to make and the chooser is not consulted -- calling it
        # and discarding the answer would spend a model call on nothing.
        if config.generator is None and not polishing and want_bred > 0:
            choices = list(pick(ranked, want_bred, state))[:want_bred]
            # A chooser that returns too few must not silently shrink the
            # generation: the arm would then spend less budget than the control
            # it is compared against. Top up at random and record how many, so
            # the shortfall shows up in the result instead of in the conclusion.
            if len(choices) < want_bred:
                filled = want_bred - len(choices)
                choices = list(choices) + list(
                    random_chooser(rng, n_loci)(ranked, filled, None)
                )
        def build_kid(choice: OperatorChoice, r: random.Random) -> Config:
            a = population[choice.parent_a % len(population)][0]
            b = population[choice.parent_b % len(population)][0]
            mask = choice.mask
            # The mask is fitted to the parent it is about to be applied to,
            # NOT to `n_loci`. A locus count is a property of a candidate, not
            # of a problem: a field holding a sequence contributes one locus per
            # element, so two candidates of the same problem legitimately have
            # different genome lengths. `n_loci` is read once from seeds[0], and
            # using it here made the shipped knapsack example -- whose seeds are
            # a 1-item and a 3-item selection -- crash deterministically on the
            # first generation, because a 1-bit mask met a 3-locus parent.
            want_bits = len(loci_of(a))
            if len(mask) != want_bits:           # a chooser may be wrong; the
                mask = mask[:want_bits] + tuple( # loop must not crash on it
                    bool(r.getrandbits(1)) for _ in range(want_bits - len(mask))
                )
            kid = crossover(a, b, mask=mask)
            rate = config.mutation_rate
            if boost_on and restriction is not None and gen <= boost_until:
                rate = max(rate or (1.0 / max(1, n_loci)),
                           min(1.0 / 3.0, 8.0 / max(1, n_loci)))
            return mutate(kid, candidate_model, rate=rate,
                          restriction=draw_prior,
                          loci=choice.mutate_loci, rng=r)

        kid_origins: Optional[List[str]] = None
        # --- mass generation: the model wrote the sampler, not the samples --
        # The pool is many times what the budget can afford, and costs the
        # budget nothing: the generator is handed a template, the domains, and
        # the archive -- never the problem and never the cache -- so the only
        # candidates that can become evaluations are the `want` below.
        pool_kids: Optional[List[Config]] = None
        if polishing:
            kids = [dict(kid) for kid in polish_kids]
        elif want_bred <= 0:
            # Exploration and intensification between them filled the
            # generation. There is no breeding left to construct, and asking a
            # constructor for zero candidates -- a generator especially, which
            # would spend a model call on it -- is a call with no answer.
            kids = []
        elif config.generator is not None:
            pool_kids = config.generator.propose(
                template=seeds[0], candidate_model=candidate_model,
                restriction=restriction,
                archive=[c for c, _o in population],
                want=want_bred, rng=rng_pool,
                seed=(config.seed or 0) * 1000 + gen)
            kids = [dict(kid) for kid in pool_kids[:want_bred]]
        elif config.portfolio is not None:
            pairs = [
                (population[choice.parent_a % len(population)][0],
                 population[choice.parent_b % len(population)][0])
                for choice in choices
            ]
            kids, kid_origins = config.portfolio.construct_generation(
                pairs, candidate_model, restriction, rng, generation=gen)
        else:
            kids = [build_kid(choice, rng) for choice in choices]

        # --- virtual pre-screening: build more than we can afford, pay for
        # the promising. The surrogate is re-validated on today's data before
        # it may order anything (the gate is the arbitration), a floor of the
        # chooser's own picks is always measured unscreened (the screen must
        # never own the whole generation), and only measure() below touches
        # the budget -- the screen has no route to it by construction.
        # A swept generation skips the screen. The sweep already IS the
        # generation's answer -- an enumerated neighbourhood, deduped and
        # ordered -- so there is nothing for a surrogate to re-order, and
        # building the screen's larger pool would spend random draws on
        # candidates the sweep deliberately did not ask for.
        screen_note: Optional[Dict[str, Any]] = None
        if config.screening is not None and kids and not polishing:
            # Cheap-fidelity evidence FIRST, so the gate this generation sees
            # the rows the campaign could not afford. It buys evidence about
            # THIS generation's candidates, which is the distribution the
            # screen is about to rank, and it charges nothing.
            if proxy_source is not None:
                config.screening.prime(
                    kids if pool_kids is None else pool_kids,
                    [proxy_source.key(c) for c, _o in state.evaluated])
            active = config.screening.refresh(
                list(state.evaluated), specs,
                seed=(config.seed or 0) + gen)
            screen_note = {"pool": len(kids), "held_out": len(kids),
                           "advanced": 0, "active": bool(active)}
            if active:
                if pool_kids is None:
                    extra_n = (config.screening.pool_factor - 1) * len(kids)
                    extra_choices = random_chooser(rng_pool, n_loci)(
                        ranked, extra_n, None)
                    pool_kids = kids + [build_kid(c, rng_pool)
                                        for c in extra_choices]
                pool_origins = (None if kid_origins is None else
                                list(kid_origins)
                                + ["pool"] * (len(pool_kids) - len(kids)))
                report = config.screening.screen(
                    pool_kids, [obj for _c, obj in population], specs)
                if report is not None:
                    # The floor is the screen's own, not a constant: a screen
                    # certified on some of the objectives is biased against
                    # the ones it cannot see, so it reserves more of the
                    # generation for the chooser's unscreened picks.
                    floor_n = min(len(kids), max(1, math.ceil(
                        config.screening.exploration_floor_for(report)
                        * want_bred)))
                    keep = list(range(floor_n))
                    for index in report.order:
                        if len(keep) >= want_bred:
                            break
                        if index >= floor_n:
                            keep.append(index)
                    kids = [pool_kids[index] for index in keep[:want_bred]]
                    if pool_origins is not None:
                        kid_origins = [pool_origins[index]
                                       for index in keep[:want_bred]]
                    screen_note = {
                        "pool": len(pool_kids), "held_out": floor_n,
                        "advanced": len(kids) - floor_n, "active": True,
                        "surrogate": report.surrogate_name,
                        # Which objectives this order was actually computed
                        # over. A generation that screened on a subset must
                        # not be readable as one that screened on the whole
                        # problem, so the scope travels in the record beside
                        # the count of what it advanced.
                        "objectives": list(report.screened_objectives),
                        "objectives_declared": list(report.declared_objectives),
                        "partial": bool(report.partial),
                    }

        # The slots taken above go in ahead of the offspring and bypass the
        # screen, exactly as the immigrants below do and for the same reason:
        # the screen orders what BREEDING constructed, and a draw made to leave
        # the region the screen's own surrogate was fitted in is not a
        # candidate that surrogate has any standing to re-order.
        if extra_kids:
            kids = [dict(kid) for kid in extra_kids] + list(kids)

        # --- immigrants: authored members, ahead of the offspring -----------
        # They bypass the screen exactly as the initial proposals do: the
        # screen orders what the loop CONSTRUCTED, and a member the model
        # authored from the run's measurements is a guidance pick whose bet is
        # settled by the evaluator, not by a surrogate.
        injected = 0
        if immigrants:
            kids = [dict(member) for member in immigrants] + list(kids)
            kids = kids[:want]
            injected = min(len(immigrants), len(kids))
            immigrants = []

        valid = measure(kids, gen)
        # Survivors compete against this generation's offspring on equal terms;
        # both carry their own objectives, so the rank is recomputed over the
        # merged set rather than inherited from the set it was measured in.
        pool: List[tuple[Config, Mapping[str, float]]] = list(population)
        pool.extend((r.configuration, dict(r.objectives)) for r in valid)
        population = survive(pool)

        # --- did anything just advance an objective best? -------------------
        # Reads only; no RNG. The LAST advancer in evaluation order becomes
        # the burst anchor -- the freshest information wins -- and a new
        # advance during a live burst retargets it and refills the count.
        if burst_on and valid:
            triggered = None
            for result in valid:
                for spec in specs:
                    value = result.objectives.get(spec.name)
                    if value is None:
                        continue
                    value = float(value)
                    held = burst_best.get(spec.name)
                    if held is None or (value > held if spec.goal == "max"
                                        else value < held):
                        burst_best[spec.name] = value
                        triggered = result.configuration
            if triggered is not None:
                burst_anchor = dict(triggered)
                burst_left = config.intensify_burst

        # --- survival credit: a mechanism is what its children survive ------
        # Not for a swept generation: the generator drew none of those
        # children, and crediting it with the sweep's survivors would put a
        # classical enumeration's wins on a model's counter.
        if config.generator is not None and valid and not polishing:
            surviving = {_default_candidate_key(c) for c, _o in population}
            for result in valid:
                config.generator.record_measured(
                    result.configuration,
                    survived=(_default_candidate_key(result.configuration)
                              in surviving),
                    objectives=dict(result.objectives))

        if config.portfolio is not None and kid_origins is not None and valid:
            surviving = {_default_candidate_key(c) for c, _o in population}
            origin_by_key: Dict[str, str] = {}
            for kid, origin in zip(kids, kid_origins):
                origin_by_key.setdefault(_default_candidate_key(kid), origin)
            for result in valid:
                key = _default_candidate_key(result.configuration)
                origin = origin_by_key.get(key)
                if origin and origin != "pool":
                    config.portfolio.record_measured(
                        origin, survived=key in surviving)
            for name in config.portfolio.review():
                log(f"generation {gen}: operator arm {name!r} retired -- no "
                    "survivors in at least 4 measured children while the "
                    "classical arm has some")

        # --- the prior is a bet, and a bet must be checkable -----------------
        # A restriction that removed the good region cannot recover on its own,
        # so once the restricted search has had a generation to show something,
        # ask whether it can still match what the screen already found. If not,
        # drop the restriction for the remainder and say so. Unwinding a prior
        # that is merely unlucky costs less than holding one that is wrong.
        if restriction is not None and screen_front and not structure_record.get("unwound"):
            beat_something = any(
                dominates(obj, excluded, specs)
                for _c, obj in population
                for excluded in screen_front)
            if not beat_something:
                restriction = None
                draw_prior = None if mixer is None else mixer.over(None)
                structure_record["unwound"] = gen
                log(f"generation {gen}: the prior stopped paying; restriction "
                    "dropped for the remainder")
        # --- X2b: the falsified cage (REFINEMENT_ROUND.md) -------------------
        # The bet test above passes too easily (dominating one screen row is
        # cheap): measured, it fired on 2 of 24 NAS cells while 8 held
        # supports that excluded every optimum. The sharper falsification is
        # the run's own scoreboard: if a member that is BEST on some declared
        # objective sits OUTSIDE the prior's support -- and stays best for 3
        # straight generations -- then the exclusion claim ("good is inside")
        # is refuted by the best thing this run has ever measured, and the
        # restriction drops. A transient escape does not fire it.
        if restriction is not None and not structure_record.get("unwound") and gen >= 4:
            allowed_map = dict(getattr(restriction, "allowed", {}) or {})
            def _violates(member: Config) -> bool:
                for locus in loci_of(member):
                    want = allowed_map.get(locus.field)
                    if want and read_locus(member, locus) not in tuple(want):
                        return True
                return False
            falsified = False
            if allowed_map:
                for spec in specs:
                    sign = 1.0 if spec.goal == "min" else -1.0
                    ranked_pop = sorted(
                        population,
                        key=lambda pair: sign * float(pair[1][spec.name]))
                    champion, champ_obj = ranked_pop[0]
                    if not _violates(champion):
                        continue
                    inside = [pair for pair in ranked_pop
                              if not _violates(pair[0])]
                    # The champion sits outside the support AND nothing the
                    # support has produced matches it: the exclusion of the
                    # champion's region is unjustified by the box's own
                    # results, which is the falsification.
                    if not inside or (
                        sign * float(inside[0][1][spec.name])
                        > sign * float(champ_obj[spec.name])):
                        falsified = True
                        break
            cage_streak = cage_streak + 1 if falsified else 0
            if cage_streak >= 3:
                restriction = None
                draw_prior = None if mixer is None else mixer.over(None)
                structure_record["unwound"] = gen
                structure_record["unwound_cage"] = gen
                cage_streak = 0
                log(f"generation {gen}: a per-objective best has sat outside "
                    "the prior's support for 3 generations; the cage is "
                    "falsified and the restriction drops")

        # --- the prior is also REVISABLE ------------------------------------
        # The unwind above can only drop a prior; it cannot correct one. A
        # prior authored before the run measured anything is right about the
        # rows it was authored from and says nothing about the rows that came
        # after, so the channel that reads those rows runs here, after
        # survival, on its own declared cadence.
        reguide_note: Optional[Dict[str, Any]] = None
        if config.reguidance is not None:
            outcome = config.reguidance.maybe_revise(
                rows=list(state.evaluated), population=list(population),
                specs=specs, restriction=restriction, charges=spent(), gen=gen)
            if outcome.restriction is not None:
                restriction = outcome.restriction
                draw_prior = (restriction if mixer is None
                              else mixer.over(restriction))
                if boost_on:
                    boost_until = gen + boost_gens
            if outcome.immigrants:
                immigrants = [dict(member) for member in outcome.immigrants]
            reguide_note = outcome.note
            if reguide_note is not None:
                log(f"generation {gen}: the sampling prior was revisited at "
                    f"{spent()} charged evaluations")

        entry = {"gen": gen, "valid_count": len(valid),
                 "pop": len(population),
                 "choices_filled_at_random": filled}
        if reguide_note is not None:
            entry["reguide"] = reguide_note
        if injected:
            entry["reguide_immigrants"] = injected
        if explore_on:
            # Recorded whenever the mechanism is on, including the zero: a
            # generation whose schedule drew no slot, or whose draws all
            # collided with the trace, is a measured zero and not an absence.
            entry["explore"] = explore_count
        if intensify_on:
            entry["intensify"] = intensify_count
        if burst_on:
            # Same terms as explore/intensify: the zero is a measured zero.
            entry["burst"] = burst_count
        if mixer is not None:
            # Cumulative draws where the front actually held an opinion; the
            # measured zero (prior right, front agreeing) is part of the story.
            entry["elite_mix_opined"] = mixer.opined
        if boost_on:
            entry["actuation_boost"] = bool(
                restriction is not None and gen <= boost_until)
        if screen_note is not None:
            entry["screen"] = screen_note
        if polish_note is not None:
            entry["polish"] = polish_note
            if polishing:
                log(f"generation {gen}: the front has not moved for "
                    f"{polish_stall} charges; sweeping {len(kids)} of the "
                    f"1-mutation neighbourhood of rank-0 member "
                    f"{polish_note['member_index']} instead of breeding")
        if config.generator is not None:
            entry["generate"] = config.generator.note()
        if config.portfolio is not None:
            entry["portfolio"] = config.portfolio.summary()
        history.append(entry)
        if filled:
            log(f"generation {gen}: the chooser supplied {want_bred - filled} "
                f"of {want_bred} choices; {filled} were filled at random")
        log(f"generation {gen}: {len(valid)} evaluated, {spent()} of "
            f"{config.evaluation_budget} budget used")

    # --- X2c: no stranded budget (REFINEMENT_ROUND.md) -----------------------
    # A run that exhausts its generations while budget remains used to stop --
    # measured on the F2 row, seven of twenty-four cells stranded up to 219 of
    # 384 evaluations because a caged mutation channel proposed 78-96%
    # duplicates. Budget honesty, the B-G family: whatever evaluations were
    # promised get spent, here on forced-novel uniform draws over the DECLARED
    # domains (no restriction -- the fill exists precisely because the guided
    # channels ran dry), dedup-checked, until the budget is gone or novelty
    # exhausts. Runs that spent their budget in the loop skip this block
    # byte-identically.
    if config.evaluation_budget is not None and budget_left() > 0:
        fill_spent = 0
        # seen_keys is only maintained under the mechanisms that need it, so
        # the fill derives seen-ness from the run's own measurement record
        # (valid AND failed both went in front of the evaluator).
        fill_seen: Set[str] = set(seen_keys)
        fill_seen.update(
            _default_candidate_key(r.configuration) for r, _m in all_meta)
        while budget_left() > 0:
            before_fill = budget_left()
            batch: List[Config] = []
            batch_keys: Set[str] = set()
            attempts = 0
            want_fill = min(budget_left(), config.offspring_per_generation)
            while len(batch) < want_fill and attempts < 60 * want_fill:
                attempts += 1
                candidate = uniform_candidate(
                    dict(seeds[0]), candidate_model, rng=rng_pool)
                key = _default_candidate_key(candidate)
                if key in fill_seen or key in batch_keys:
                    continue
                batch_keys.add(key)
                batch.append(candidate)
            if not batch:
                log(f"fill: novelty exhausted with {budget_left()} of "
                    f"{config.evaluation_budget} unspent")
                break
            fill_valid = measure(batch, config.generations + 1)
            pool = list(population)
            pool.extend((r.configuration, dict(r.objectives))
                        for r in fill_valid)
            population = survive(pool)
            fill_seen.update(batch_keys)
            fill_spent += before_fill - budget_left()
            if budget_left() >= before_fill:
                # A whole batch bought no evaluation (validation failures
                # charge nothing): chargeability, not novelty, is the
                # blocker, and looping would spin forever. Say so and stop.
                log(f"fill: a batch of {len(batch)} bought no evaluation; "
                    f"{budget_left()} of {config.evaluation_budget} unspent")
                break
        if fill_spent:
            history.append({"fill": fill_spent,
                            "budget_left": budget_left()})
            log(f"fill: spent {fill_spent} stranded evaluations on "
                "forced-novel draws after the generation cap")

    if structure_record:
        history.append({"structure": dict(structure_record)})
    result = _build_search_result(
        all_valid, all_meta, specs, history,
        evaluations=spent(), candidate_key=_default_candidate_key,
    )
    # Telemetry is attached even when every mechanism was counter-free: an
    # empty mechanism list on a random run is a measured zero, and a guided
    # run's counters are the difference between "guidance did not help" and
    # "guidance never arrived".
    virtual = 0
    if config.screening is not None:
        virtual = int(getattr(
            config.screening.telemetry, "virtual_evaluations", 0))
    # Proxy evaluations are reported BESIDE the charged count, never inside
    # it: `real_evaluations` is what the budget bought and what any
    # evaluation-efficiency claim is denominated in.
    proxy_evaluations = (0 if proxy_source is None
                         else int(proxy_source.ledger.evaluations))
    return replace(result, telemetry=harvest_telemetry(
        (pick, prior_proposer_used, config.screening,
         getattr(config.screening, "author", None),
         config.portfolio, getattr(config.portfolio, "author", None),
         config.generator, getattr(config.generator, "author", None),
         config.reguidance, proxy_source),
        real_evaluations=spent(), virtual_evaluations=virtual,
        proxy_evaluations=proxy_evaluations,
    ))
