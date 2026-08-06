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

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import SearchResult
from agent_evolve.policies.genetic import (
    Locus,
    crossover,
    loci_of,
    mutate,
    truncation_survival,
    uniform_candidate,
)
from agent_evolve.session.evaluate import EvaluationCache, evaluate_batch

__all__ = ["OperatorChoice", "OperatorChooser", "GeneticConfig", "run_genetic_loop",
           "random_chooser", "domination_rank"]

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
    #: A prior over WHERE to sample: narrows the declared domain of any locus it
    #: names, for initialization and mutation alike. Left None, every sampler
    #: sees exactly the schema's own domains, so the loop is byte-identical to
    #: the pre-restriction seam. This is guidance over the sampling
    #: distribution rather than over operator choice within a fixed one.
    restriction: Any = None


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

    all_valid: List[Any] = []
    all_meta: List[tuple] = []
    history: List[Dict[str, Any]] = []
    # The state accumulates across generations and is handed to the chooser
    # each time. The unguided chooser ignores it, which is what makes the two
    # arms differ in exactly one thing.
    state = config.state if config.state is not None else SearchState()
    state.history = history

    def measure(configs: List[Config], gen: int) -> List[Any]:
        room = budget_left()
        if room <= 0:
            return []
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

    # --- initial population: the seeds, then SCHEMA-UNIFORM draws -----------
    # Not mutants of the seed. A population of near-copies of one
    # configuration is an anchored cloud around it; measured on a third-party
    # optimizer, correcting exactly this anchor moved its result from +0.095
    # (loses badly to uniform) to +0.0066 (parity), and the standard seed on
    # log2 scores worse than a typical uniform draw.
    n_loci = len(loci_of(seeds[0]))
    initial = list(seeds)
    while len(initial) < config.population_size:
        template = initial[rng.randrange(len(initial))]
        initial.append(uniform_candidate(template, candidate_model, rng=rng,
                                         restriction=config.restriction))
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
        )
        return [pair for pair, _rank in kept]

    population = survive([(r.configuration, dict(r.objectives)) for r in valid])
    history.append({"gen": 0, "valid_count": len(valid), "pop": len(population)})
    log(f"generation 0: {len(valid)} evaluated, population {len(population)}")

    pick = chooser or random_chooser(rng, n_loci)

    for gen in range(1, config.generations + 1):
        if budget_left() <= 0:
            log(f"budget exhausted after {spent()} evaluations; stopping at gen {gen}")
            break
        want = min(config.offspring_per_generation, budget_left())
        # The chooser sees ranks, not raw objectives: it decides which parents to
        # combine, and a rank is the comparable form of "how good is this one".
        ranks = domination_rank([obj for _c, obj in population], specs)
        ranked = [(c, r) for (c, _o), r in zip(population, ranks)]
        choices = list(pick(ranked, want, state))[:want]
        # A chooser that returns too few must not silently shrink the
        # generation: the arm would then spend less budget than the control it
        # is compared against. Top up at random and record how many, so the
        # shortfall shows up in the result instead of in the conclusion.
        filled = 0
        if len(choices) < want:
            filled = want - len(choices)
            choices = list(choices) + list(
                random_chooser(rng, n_loci)(ranked, filled, None)
            )
        kids: List[Config] = []
        for choice in choices:
            a = population[choice.parent_a % len(population)][0]
            b = population[choice.parent_b % len(population)][0]
            mask = choice.mask
            if len(mask) != n_loci:              # a chooser may be wrong; the
                mask = mask[:n_loci] + tuple(    # loop must not crash on it
                    bool(rng.getrandbits(1)) for _ in range(n_loci - len(mask))
                )
            kid = crossover(a, b, mask=mask)
            kid = mutate(kid, candidate_model, rate=config.mutation_rate,
                         restriction=config.restriction,
                         loci=choice.mutate_loci, rng=rng)
            kids.append(kid)

        valid = measure(kids, gen)
        # Survivors compete against this generation's offspring on equal terms;
        # both carry their own objectives, so the rank is recomputed over the
        # merged set rather than inherited from the set it was measured in.
        pool: List[tuple[Config, Mapping[str, float]]] = list(population)
        pool.extend((r.configuration, dict(r.objectives)) for r in valid)
        population = survive(pool)
        history.append({"gen": gen, "valid_count": len(valid),
                        "pop": len(population),
                        "choices_filled_at_random": filled})
        if filled:
            log(f"generation {gen}: the chooser supplied {want - filled} of "
                f"{want} choices; {filled} were filled at random")
        log(f"generation {gen}: {len(valid)} evaluated, {spent()} of "
            f"{config.evaluation_budget} budget used")

    return _build_search_result(
        all_valid, all_meta, specs, history,
        evaluations=spent(), candidate_key=_default_candidate_key,
    )
