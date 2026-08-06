"""Pre-flight problem diagnostic: ``check(problem, budget) -> CheckReport``.

Before any optimizer -- model-guided or not -- is credited with anything on a
problem, two questions have answers the problem itself can give:

1. **What is the search space, actually?** How many heritable loci does a
   candidate carry, how many values does the schema declare at each, and which
   loci declare none (those an uninformed sampler cannot vary at all)?

2. **Is there anything to win at this budget?** If the best value a broad
   uniform probe can find is within noise of what ``budget`` random draws are
   expected to reach anyway, then *no optimizer can demonstrate an advantage
   here at this budget* -- not because optimizers are useless, but because the
   measurement cannot separate one from chance. This project has been burned by
   exactly that: on one benchmark a single median random draw already carried
   roughly 80 percent of the final hypervolume.

The probe draws ``probe`` schema-uniform candidates (via
:func:`agent_evolve.policies.genetic.uniform_candidate`, so only declared
domains are sampled and nothing is invented) and pushes each through the
problem's own ``validate -> materialize -> evaluate`` pipeline, recording
failures at every stage and the wall-clock cost of ``evaluate``.

**What is spent.** The probe costs at most ``probe`` evaluations -- ``budget``
is the optimizer budget *being assessed*, not the probe's spend, exactly as
``agent_evolve check`` spends ``repeats x budget`` evaluations to assess one
budget. A probe larger than the budget is the informative regime: it looks
further than the budget can, and asks whether the extra looking found anything
the budget would miss.

**The headroom estimate.** For each objective, the best value seen anywhere in
the probe is compared with the distribution of the best of ``budget`` uniform
redraws from the probe's own empirical values. That distribution is computed in
closed form (the exact bootstrap, not a Monte Carlo approximation of it), so
the reported noise is the sampling noise of a ``budget``-draw run with nothing
added by the estimator. Headroom at or below that noise means chance already
covers everything the probe found.

Everything here is workload-agnostic and credential-free: no model, no
provider, no network. The loci, domains and objective directions all come from
the problem's own declarations.
"""

from __future__ import annotations

import math
import random
import statistics
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from agent_evolve.contract import as_problem
from agent_evolve.core.problem import normalize_objective_values
from agent_evolve.policies.genetic import loci_of, locus_domain, uniform_candidate

__all__ = [
    "check",
    "CheckReport",
    "LocusDomain",
    "ObjectiveSpread",
    "ObjectiveHeadroom",
]

#: The plain-language conclusion for a problem/budget pair on which the probe
#: found nothing that chance at the same budget would not also find. Tests and
#: callers match on this exact sentence, so it is a named constant.
NO_HEADROOM_VERDICT = "no optimizer can demonstrate an advantage here at this budget"


@dataclass(frozen=True, slots=True)
class LocusDomain:
    """One heritable position and how many values the schema declares for it."""

    locus: str
    domain_size: int  #: 0 means the schema declares no finite set here.

    @property
    def declared(self) -> bool:
        return self.domain_size > 0


@dataclass(frozen=True, slots=True)
class ObjectiveSpread:
    """How one objective varied over the probe's successful evaluations."""

    objective: str
    goal: str
    count: int
    minimum: float
    maximum: float
    mean: float
    stdev: float


@dataclass(frozen=True, slots=True)
class ObjectiveHeadroom:
    """Best-of-probe against the exact best-of-budget bootstrap, one objective."""

    objective: str
    goal: str
    best_of_probe: float
    #: Expected best of ``budget`` uniform redraws from the probe's values.
    best_of_budget: float
    #: Standard deviation of that best-of-budget distribution -- the noise a
    #: single budget-sized random run carries.
    noise: float
    #: Improvement still on the table beyond expected chance, in the improving
    #: direction (always >= 0).
    headroom: float
    below_noise: bool


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Everything :func:`check` measured, plus the plain-language verdict."""

    problem: str
    budget: int
    probe: int
    draws: int
    evaluated: int
    loci: tuple[LocusDomain, ...]
    undeclared_loci: tuple[str, ...]
    validate_failures: int
    materialize_failures: int
    evaluate_failures: int
    failure_rate: float
    failure_examples: tuple[str, ...]
    eval_seconds_median: float
    spread: tuple[ObjectiveSpread, ...]
    headroom: tuple[ObjectiveHeadroom, ...]
    verdict: str

    @property
    def locus_count(self) -> int:
        return len(self.loci)

    def render(self) -> str:
        """The report as plain text, in the same voice as the CLI."""
        lines: list[str] = [f"problem check: {self.problem}"]
        lines.append(f"  budget assessed  {self.budget} evaluations")
        lines.append(
            f"  probe spent      {self.draws} draws, {self.evaluated} evaluated"
            f"  (failure rate {self.failure_rate:.0%})"
        )
        lines.append("")
        lines.append(f"  search space  ({self.locus_count} loci)")
        shown = self.loci[:24]
        entries = " ".join(
            f"{d.locus}:{d.domain_size if d.declared else '?'}" for d in shown
        )
        tail = f"  (+{len(self.loci) - len(shown)} more)" if len(self.loci) > len(shown) else ""
        for row in textwrap.wrap(entries + tail, width=74) or ["(no loci)"]:
            lines.append(f"    {row}")
        undeclared = ", ".join(self.undeclared_loci) if self.undeclared_loci else "none"
        for row in textwrap.wrap(f"undeclared domains: {undeclared}", width=74):
            lines.append(f"    {row}")
        lines.append("")
        lines.append("  probe pipeline")
        lines.append(
            f"    failures      validate {self.validate_failures}, "
            f"materialize {self.materialize_failures}, "
            f"evaluate {self.evaluate_failures}"
        )
        for message in self.failure_examples:
            for row in textwrap.wrap(f"e.g. {message}", width=70):
                lines.append(f"      {row}")
        lines.append(f"    eval seconds  median {self.eval_seconds_median:.6g}")
        lines.append("")
        lines.append(f"  objective spread  (over {self.evaluated} evaluations)")
        if not self.spread:
            lines.append("    (nothing was successfully evaluated)")
        for s in self.spread:
            lines.append(
                f"    {s.objective} ({s.goal})  min {s.minimum:g}  max {s.maximum:g}"
                f"  mean {s.mean:g}  sd {s.stdev:g}"
            )
        lines.append("")
        lines.append(f"  headroom at budget {self.budget}")
        if not self.headroom:
            lines.append("    (no measurements to estimate from)")
        for h in self.headroom:
            flag = "below noise" if h.below_noise else "ABOVE noise"
            lines.append(
                f"    {h.objective} ({h.goal})  best of probe {h.best_of_probe:g}"
                f"  expected best of {self.budget} random {h.best_of_budget:g}"
                f" +/- {h.noise:g}  headroom {h.headroom:g}  [{flag}]"
            )
        lines.append("")
        lines.append("  verdict")
        for row in textwrap.wrap(self.verdict, width=72):
            lines.append(f"    {row}")
        return "\n".join(lines)


def _template_of(problem: Any) -> dict[str, Any]:
    """The configuration whose shape defines the probe's loci.

    A seed is the problem's own statement of what a candidate looks like, so it
    is preferred; ``example_config`` and an all-defaults ``candidate_model``
    construction are accepted in that order for problems that declare no seed.
    """
    seeds = tuple(problem.seeds())
    if seeds:
        return dict(seeds[0])
    example = getattr(problem, "example_config", None)
    if isinstance(example, Mapping) and example:
        return dict(example)
    model = getattr(problem, "candidate_model", None)
    if model is not None:
        try:
            return dict(model().model_dump())
        except Exception:
            pass
    raise ValueError(
        "check() needs one configuration to shape the probe. Give the problem "
        "a seed (Problem.seeds()), an example_config, or a candidate_model "
        "whose fields all carry defaults."
    )


def _best_of_budget(values: Sequence[float], budget: int, goal: str) -> tuple[float, float]:
    """Mean and standard deviation of the best of *budget* uniform redraws.

    This is the exact bootstrap: with the probe's values sorted so the best
    under *goal* comes last, the best of ``k`` redraws lands at rank ``i`` with
    probability ``(i/n)**k - ((i-1)/n)**k``. Summing over ranks gives the
    distribution in closed form, so the returned noise is purely the sampling
    noise of a budget-sized random run -- no Monte Carlo error rides on top.
    """
    n = len(values)
    ordered = sorted(values, reverse=(goal == "min"))  # best is last either way
    mean = 0.0
    second = 0.0
    previous = 0.0
    for rank, value in enumerate(ordered, start=1):
        prefix = (rank / n) ** budget
        p = prefix - previous
        previous = prefix
        mean += p * value
        second += p * value * value
    return mean, math.sqrt(max(0.0, second - mean * mean))


def _verdict(
    *,
    budget: int,
    draws: int,
    evaluated: int,
    validate_failures: int,
    materialize_failures: int,
    evaluate_failures: int,
    failure_rate: float,
    headroom: Sequence[ObjectiveHeadroom],
    loci: Sequence[LocusDomain],
    undeclared: Sequence[str],
) -> str:
    if evaluated == 0:
        base = (
            f"Every one of the {draws} probe draws failed before producing a "
            f"measurement (validate {validate_failures}, materialize "
            f"{materialize_failures}, evaluate {evaluate_failures}). Headroom "
            "cannot be assessed; fix the failures this report names before "
            "spending anything on optimization."
        )
        return base
    above = [h.objective for h in headroom if not h.below_noise]
    below = [h.objective for h in headroom if h.below_noise]
    if not above:
        base = (
            f"The best value the probe found on every objective is within "
            f"noise of what {budget} random draws are expected to reach: "
            f"{NO_HEADROOM_VERDICT}. Raise the budget, or reshape the search "
            "space, before crediting any optimizer with a win."
        )
    elif not below:
        base = (
            f"Every objective carries headroom above noise at budget {budget}: "
            f"the probe found values that {budget} random draws would "
            "typically miss. That gap is what an optimizer has to close to "
            "earn its cost."
        )
    else:
        base = (
            f"Headroom above noise on {', '.join(above)}; none on "
            f"{', '.join(below)}. At this budget an optimizer can only "
            "demonstrate an advantage on the former."
        )
    notes: list[str] = []
    if undeclared:
        named = ", ".join(list(undeclared)[:6])
        more = ", ..." if len(undeclared) > 6 else ""
        notes.append(
            f"Note: {len(undeclared)} of {len(loci)} loci declare no finite "
            f"domain ({named}{more}), so the probe could not vary them and any "
            "headroom along those axes is invisible to this check."
        )
    if failure_rate >= 0.5:
        notes.append(
            f"{failure_rate:.0%} of draws failed before measurement; the "
            f"estimate rests on only {evaluated} evaluations."
        )
    return " ".join([base, *notes])


def check(
    problem: Any,
    budget: int,
    *,
    probe: int = 120,
    seed: Optional[int] = None,
) -> CheckReport:
    """Probe *problem* and report whether *budget* can show anything at all.

    *budget* is the optimizer budget under assessment. The probe itself spends
    at most *probe* evaluations (schema-uniform draws through the problem's own
    ``validate -> materialize -> evaluate``), needs no model, no credentials
    and no network, and is deterministic given *seed*.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    if not isinstance(probe, int) or isinstance(probe, bool) or probe < 1:
        raise ValueError(f"probe must be a positive integer, got {probe!r}")

    name = type(problem).__name__
    bound = as_problem(problem)
    objectives = list(bound.objectives)
    model = getattr(bound, "candidate_model", None)
    template = _template_of(bound)

    loci = loci_of(template)
    domains = tuple(
        LocusDomain(locus=str(locus), domain_size=len(locus_domain(model, locus)))
        for locus in loci
    )
    undeclared = tuple(d.locus for d in domains if not d.declared)

    rng = random.Random(seed)
    validate_failures = 0
    materialize_failures = 0
    evaluate_failures = 0
    failure_examples: list[str] = []
    eval_seconds: list[float] = []
    rows: list[Mapping[str, float]] = []

    def note_failure(message: str) -> None:
        if message and message not in failure_examples and len(failure_examples) < 3:
            failure_examples.append(message)

    for _ in range(probe):
        config = uniform_candidate(template, model, rng=rng)
        try:
            outcome = bound.validate(config)
            ok = bool(getattr(outcome, "ok", outcome))
            message = getattr(outcome, "message", None) or "validate() rejected the draw"
        except Exception as error:  # noqa: BLE001 - a diagnostic records, it does not crash
            ok, message = False, f"validate raised {type(error).__name__}: {error}"
        if not ok:
            validate_failures += 1
            note_failure(f"validate: {message}")
            continue
        try:
            artifact = bound.materialize(config)
        except Exception as error:  # noqa: BLE001
            materialize_failures += 1
            note_failure(f"materialize raised {type(error).__name__}: {error}")
            continue
        started = time.perf_counter()
        try:
            values = bound.evaluate(artifact)
            elapsed = time.perf_counter() - started
            rows.append(normalize_objective_values(values, objectives))
        except Exception as error:  # noqa: BLE001
            evaluate_failures += 1
            note_failure(f"evaluate raised {type(error).__name__}: {error}")
            continue
        eval_seconds.append(elapsed)

    draws = probe
    evaluated = len(rows)
    failures = validate_failures + materialize_failures + evaluate_failures
    failure_rate = failures / draws if draws else 0.0

    spread: list[ObjectiveSpread] = []
    headroom: list[ObjectiveHeadroom] = []
    for spec in objectives:
        values = [row[spec.name] for row in rows]
        if not values:
            continue
        spread.append(
            ObjectiveSpread(
                objective=spec.name,
                goal=spec.goal,
                count=len(values),
                minimum=min(values),
                maximum=max(values),
                mean=statistics.fmean(values),
                stdev=statistics.pstdev(values),
            )
        )
        best_of_probe = max(values) if spec.goal == "max" else min(values)
        expected, noise = _best_of_budget(values, budget, spec.goal)
        gap = (best_of_probe - expected) if spec.goal == "max" else (expected - best_of_probe)
        gap = max(0.0, gap)
        headroom.append(
            ObjectiveHeadroom(
                objective=spec.name,
                goal=spec.goal,
                best_of_probe=best_of_probe,
                best_of_budget=expected,
                noise=noise,
                headroom=gap,
                below_noise=gap <= noise,
            )
        )

    verdict = _verdict(
        budget=budget,
        draws=draws,
        evaluated=evaluated,
        validate_failures=validate_failures,
        materialize_failures=materialize_failures,
        evaluate_failures=evaluate_failures,
        failure_rate=failure_rate,
        headroom=headroom,
        loci=domains,
        undeclared=undeclared,
    )

    return CheckReport(
        problem=name,
        budget=budget,
        probe=probe,
        draws=draws,
        evaluated=evaluated,
        loci=domains,
        undeclared_loci=undeclared,
        validate_failures=validate_failures,
        materialize_failures=materialize_failures,
        evaluate_failures=evaluate_failures,
        failure_rate=failure_rate,
        failure_examples=tuple(failure_examples),
        eval_seconds_median=statistics.median(eval_seconds) if eval_seconds else 0.0,
        spread=tuple(spread),
        headroom=tuple(headroom),
        verdict=verdict,
    )
