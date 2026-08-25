"""Learning the landscape's structure, and spending the knowledge.

The loop this package shipped with could only ever ask "which parents, which
cut points" -- decisions *inside* a fixed schema-uniform sampling distribution.
That caps what any guidance can buy: on an exhaustively-censused venue the best
achievable operator choice reached 7 of 34 frontier configurations, while
simply sampling from the right quarter of the space reaches 11. The ceiling was
the distribution, not the chooser.

This module adds the phase that was missing. A *screen* spends a few
evaluations on a design that can actually identify structure; *attribution*
turns those evaluations into per-level evidence; a *prior proposer* -- a plain
rule or a model -- turns the evidence into a :class:`DomainRestriction` that
narrows where every later operator samples.

Two things are load-bearing and neither is about language models.

A screen must be CROSSED, not merely balanced. Balancing each locus marginally
leaves its contrast confounded with every other locus, and a confounded table
says "nothing is excludable" no matter who reads it. Measured: on the same
venue, a marginally-balanced 8-run screen identified one of two deciding loci,
while a crossed 4-run screen identified both. Fewer evaluations, better design.

Crossing the few loci that fit is only half of it, and the half this module
shipped without: the loci that do NOT fit in the crossing must still be
decorrelated from each other, or the design collapses onto the space's main
diagonal and every prior fit from it lives in a box the diagonal happened to
pass through. That defect shipped, was measured, and is fixed -- see
:func:`crossed_screen`.

A prior governs SAMPLING, not the caller's input. Seeds are data the caller
supplied and survive a restriction untouched; only drawn candidates -- fresh
initialization and mutation -- are confined. A prior that could delete a seed
would be overruling the one thing the problem stated outright.

And the prior must be falsifiable. A restriction that removes the good region
is unrecoverable by construction, so callers get :func:`restriction_is_paying`
to check the bet against the screen's own best and unwind it.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import (
    DomainRestriction,
    Locus,
    loci_of,
    locus_domain,
    read_locus,
    write_locus,
)

__all__ = [
    "LevelSummary",
    "Attribution",
    "crossed_screen",
    "attribute",
    "render_attribution",
    "statistical_prior",
    "restriction_is_paying",
    "PriorProposer",
]


@dataclass(frozen=True)
class LevelSummary:
    """What the screen measured for one value of one locus."""

    locus: str
    value: Any
    n: int
    objective_means: Mapping[str, float]
    nondominated: int


@dataclass(frozen=True)
class Attribution:
    """Per-level evidence from the screened candidates."""

    levels: tuple[LevelSummary, ...] = ()
    n_evaluated: int = 0
    blocking: tuple[str, ...] = ()

    def for_locus(self, name: str) -> tuple[LevelSummary, ...]:
        return tuple(s for s in self.levels if s.locus == name)


#: Turns measured evidence into a prior over where to sample. A rule, a model,
#: anything: the loop only requires that it narrows declared domains.
PriorProposer = Callable[[Attribution, Any], DomainRestriction]


def crossed_screen(
    template: Mapping[str, Any],
    candidate_model: Any,
    *,
    size: int,
    blocking: Sequence[str] | None = None,
    rng: random.Random | None = None,
    pool_by_field: bool = False,
) -> list[dict[str, Any]]:
    """A screening design of *size* candidates that can identify structure.

    The blocking loci are fully crossed -- every combination of their values
    appears equally often -- so their contrasts are not confounded with each
    other. Every remaining locus walks its OWN ladder of values, and the same
    ladder of rows is reused inside each block, so a free locus's levels stay
    matched across blocks rather than correlated with them.

    Chosen automatically when *blocking* is omitted: the loci with the smallest
    declared domains, taking as many as fit in *size*. Small domains are where
    structural switches live, and they are the cheapest to cross.

    **Fixed 2026-08-25 -- the free ladder was a DIAGONAL, and it boxed every
    prior this package ever fit.** Until this change every free locus read the
    same ladder index ``t``, so row ``t`` set *every* free locus to level
    ``t % len(domain)``. On the 24-locus analog venue (20 declared levels per
    locus, ``size=16``, nothing crossable) the 16 "crossed" rows were literally
    the main diagonal: row 0 every locus at its minimum, row 8 every locus at
    mid-ladder, row 15 every locus at level 15 -- byte-reproduced from the
    frozen ``results_probe320`` traces. The consequences were measured, not
    inferred. The installed prior's maximum allowed level index was 13-15 in
    20 of 20 analog cells and never once above 15, while over the venue's
    12,899 valid evaluations the best in-box point (all 24 coordinates at level
    <= 15) scores reward9 **-0.4002** against a true best of **-0.0566**, and
    **100 of the pooled top 100 sit outside the box**. Our own finals sat at
    -0.353 ... -0.555: at the ceiling of the box, not failing inside it. The
    same shared index walked 5 of 6 NAS loci in lockstep, and on the fleet 15
    of 16 screen rows had ``e1=e2=e3=e4=e5``, which handed 7 of 12 seeds
    *identical* allowed sets for those five loci.

    Every screen-using row recorded before 2026-08-25 therefore measured a
    diagonal. Those rows remain quotable as measured on the substrate they ran
    on; they are not evidence about a space-filling screen, and no prior-quality
    claim survives being carried across this fix without a rerun.

    The repair is a deterministic latin-hypercube-style assignment. Each free
    locus gets its own column: its level indices, permuted with the run's
    ``rng``, laid down the rows and re-permuted every time the ladder is longer
    than the domain. Loci are decorrelated from one another because their
    permutations are independent, while each still sweeps its full ladder --
    every level appears at least ``rows // len(domain)`` times, which is the
    per-level n :func:`attribute` needs. The design is default-on and exact
    given the seed. A run with ``structure_budget=0`` never reaches this
    function at all, so the pre-fix fossil is byte-unchanged.

    ``pool_by_field`` changes the design for SEQUENCE fields, whose positions
    share one vocabulary: crossing them is out of reach (k^n cells) and
    per-position ladders leave per-level n tiny, but if positions are roughly
    exchangeable the value effect can be measured POOLED across positions.
    The design is then per-value: first one PURE candidate per value (every
    position set to it -- the cleanest pooled contrast), then SPIKED
    candidates (the value at every other position, the rest cycling through
    the vocabulary) until *size* is spent. Whether positions really are
    exchangeable is the bet this design makes; the caller's prior gets
    unwound by the standard test when they are not.
    """

    r = rng or random.Random()
    if pool_by_field:
        return _pooled_screen(template, candidate_model, size=size)
    all_loci = [lc for lc in loci_of(template) if locus_domain(candidate_model, lc)]
    if not all_loci or size <= 0:
        return []
    domains = {lc: locus_domain(candidate_model, lc) for lc in all_loci}

    if blocking is None:
        block: list[Locus] = []
        cells = 1
        for lc in sorted(all_loci, key=lambda l: len(domains[l])):
            if cells * len(domains[lc]) > size:
                break
            block.append(lc)
            cells *= len(domains[lc])
    else:
        wanted = tuple(blocking)
        block = [lc for lc in all_loci if str(lc) in wanted or lc.field in wanted]
        cells = 1
        for lc in block:
            cells *= len(domains[lc])
    if not block:                       # nothing crossable: fall back to a walk
        block, cells = [], 1

    rest = [lc for lc in all_loci if lc not in block]
    per_cell = max(1, size // max(1, cells))
    # One shared ladder of settings for the non-blocking loci, reused verbatim
    # in every block: that is what keeps their levels matched across blocks.
    # Within the ladder each locus walks its own permutation, which is what
    # keeps the rows off the diagonal (see this function's docstring).
    ladder = _free_ladder(rest, domains, per_cell, random.Random(r.getrandbits(64)))

    out: list[dict[str, Any]] = []
    combos = list(itertools.product(*[domains[lc] for lc in block])) or [()]
    for combo in combos:
        for setting in ladder:
            cfg = dict(template)
            for lc, value in zip(block, combo):
                cfg = write_locus(cfg, lc, value)
            for lc, value in setting.items():
                cfg = write_locus(cfg, lc, value)
            out.append(cfg)
    # Never exceed the budget the caller agreed to spend; if the crossing did
    # not fill it, top up with schema-uniform draws rather than repeating.
    while len(out) < size:
        cfg = dict(template)
        for lc in all_loci:
            cfg = write_locus(cfg, lc, r.choice(domains[lc]))
        out.append(cfg)
    return out[:size]


def _free_ladder(
    loci: Sequence[Locus],
    domains: Mapping[Locus, Sequence[Any]],
    rows: int,
    rng: random.Random,
) -> list[dict[Locus, Any]]:
    """One value per locus per row, sweeping each ladder without the diagonal.

    Every locus gets an independent column: its declared levels shuffled, laid
    down the rows, and reshuffled for each further pass when *rows* outruns the
    domain. Two properties fall out and both are load-bearing. Coverage is
    exactly what a balanced ladder gave -- ``rows // len(domain)`` appearances
    of every level at minimum, because each pass is a whole permutation -- so
    :func:`attribute` keeps its per-level n. And the columns are independent,
    so no locus's contrast is confounded with another's; the shared index that
    made them identical is what put every prior in a box (:func:`crossed_screen`).

    Reshuffling per pass rather than cycling one permutation matters: cycling
    would give only ``len(domain)`` distinct rows repeated, which spends the
    screen's budget on duplicates of the same corner.
    """

    columns: dict[Locus, list[Any]] = {}
    for locus in loci:
        domain = list(domains[locus])
        column: list[Any] = []
        while len(column) < rows:
            rng.shuffle(domain)
            column.extend(domain)
        columns[locus] = column[:rows]
    return [{locus: columns[locus][t] for locus in loci} for t in range(rows)]


def _pooled_screen(
    template: Mapping[str, Any],
    candidate_model: Any,
    *,
    size: int,
) -> list[dict[str, Any]]:
    """Pure and spiked candidates over each sequence field's shared vocabulary."""

    sequence_loci = [lc for lc in loci_of(template) if lc.index is not None
                     and locus_domain(candidate_model, lc)]
    if not sequence_loci or size <= 0:
        return []
    fields: dict[str, list[Locus]] = {}
    for lc in sequence_loci:
        fields.setdefault(lc.field, []).append(lc)

    out: list[dict[str, Any]] = []
    spike_round = 0
    while len(out) < size:
        for name, loci in fields.items():
            domain = locus_domain(candidate_model, loci[0])
            for value_index, value in enumerate(domain):
                if len(out) >= size:
                    break
                cfg = dict(template)
                for position, lc in enumerate(loci):
                    if spike_round == 0:
                        cfg = write_locus(cfg, lc, value)      # pure
                    elif position % 2 == 0:
                        cfg = write_locus(cfg, lc, value)      # spiked
                    else:
                        fill = domain[(value_index + position + spike_round)
                                      % len(domain)]
                        cfg = write_locus(cfg, lc, fill)
                out.append(cfg)
        spike_round += 1
    return out[:size]


def _nondominated(points: Sequence[Sequence[float]]) -> list[bool]:
    out = []
    for a, pa in enumerate(points):
        out.append(not any(
            b != a
            and all(x <= y for x, y in zip(pb, pa))
            and any(x < y for x, y in zip(pb, pa))
            for b, pb in enumerate(points)))
    return out


def attribute(
    evaluated: Sequence[tuple[Mapping[str, Any], Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
    candidate_model: Any,
    *,
    blocking: Sequence[str] = (),
    pool_by_field: bool = False,
) -> Attribution:
    """Summarise screened candidates per locus value.

    Non-domination is computed WITHIN the screen, in minimisation-normalised
    coordinates, because that is the only comparison an optimizer can make from
    its own evaluations -- no global frontier is available at run time.

    ``pool_by_field`` treats a sequence field's positions as exchangeable:
    every (candidate, position) pair becomes one observation of the value it
    holds, keyed by the bare field name. Per-position n on a short screen is
    unusably small; pooled n is positions times larger. The exchangeability
    bet itself is the caller's, and the standard unwind test is its check.
    """

    if not evaluated:
        return Attribution((), 0, tuple(blocking))
    names = [s.name for s in specs]
    sign = {s.name: (-1.0 if s.goal == "max" else 1.0) for s in specs}
    pts = [[sign[n] * float(obj[n]) for n in names] for _, obj in evaluated]
    nd = _nondominated(pts)

    scalar_loci: list[Locus] = []
    pooled_fields: dict[str, list[Locus]] = {}
    for lc in loci_of(evaluated[0][0]):
        if not locus_domain(candidate_model, lc):
            continue
        if pool_by_field and lc.index is not None:
            pooled_fields.setdefault(lc.field, []).append(lc)
        else:
            scalar_loci.append(lc)

    levels: list[LevelSummary] = []
    for lc in scalar_loci:
        seen: dict[Any, list[int]] = {}
        for i, (cfg, _obj) in enumerate(evaluated):
            seen.setdefault(read_locus(cfg, lc), []).append(i)
        for value, idxs in seen.items():
            levels.append(LevelSummary(
                locus=str(lc), value=value, n=len(idxs),
                objective_means={
                    n: sum(float(evaluated[i][1][n]) for i in idxs) / len(idxs)
                    for n in names},
                nondominated=sum(1 for i in idxs if nd[i]),
            ))
    for field_name, loci in pooled_fields.items():
        seen = {}
        for i, (cfg, _obj) in enumerate(evaluated):
            for lc in loci:
                seen.setdefault(read_locus(cfg, lc), []).append(i)
        for value, idxs in seen.items():
            levels.append(LevelSummary(
                locus=field_name, value=value, n=len(idxs),
                objective_means={
                    n: sum(float(evaluated[i][1][n]) for i in idxs) / len(idxs)
                    for n in names},
                nondominated=sum(1 for i in idxs if nd[i]),
            ))
    return Attribution(tuple(levels), len(evaluated), tuple(blocking))


def render_attribution(attr: Attribution) -> str:
    """The screen as text: what a reader (rule or model) gets to reason over."""

    if not attr.levels:
        return "(no screened evaluations)"
    lines: list[str] = []
    for name in dict.fromkeys(s.locus for s in attr.levels):
        lines.append(f"  {name}:")
        for s in attr.for_locus(name):
            means = "  ".join(f"{k}={v:.6g}" for k, v in s.objective_means.items())
            lines.append(f"    {s.value}: n={s.n}  {means}  "
                         f"non-dominated {s.nondominated}/{s.n}")
    return "\n".join(lines)


def statistical_prior(
    attr: Attribution,
    candidate_model: Any = None,
    *,
    min_levels: int = 2,
) -> DomainRestriction:
    """The credential-free prior: keep the levels the screen did not refute.

    For each locus, a level is kept when its within-screen non-domination rate
    ties the best rate for that locus. A locus whose levels all tie is left
    alone -- which is the honest reading of a screen that separated nothing,
    and the reason this rule abstains far more often than it commits.

    This ships as the default so the phase is useful with no model and no
    credential, and so any model-proposed prior has a rule to beat.
    """

    allowed: dict[str, list[Any]] = {}
    for name in dict.fromkeys(s.locus for s in attr.levels):
        summaries = attr.for_locus(name)
        if len(summaries) < min_levels:
            continue
        rates = {s.value: (s.nondominated / s.n if s.n else 0.0) for s in summaries}
        best = max(rates.values())
        if best <= 0.0:
            continue
        keep = [v for v, rate in rates.items() if rate >= best]
        if 0 < len(keep) < len(summaries):
            allowed[name.split("[")[0]] = keep
    return DomainRestriction(allowed)


def restriction_is_paying(
    restriction: DomainRestriction,
    screen_best: float,
    current_best: float,
    *,
    goal: str = "min",
) -> bool:
    """Is the bet still worth holding?

    A restriction that removed the good region cannot recover on its own, so a
    caller that installed one must be able to notice and unwind it. The test is
    deliberately weak -- the restricted search only has to have matched what the
    screen already found -- because unwinding a prior that is merely unlucky
    costs less than holding one that is wrong.
    """

    if not restriction.allowed:
        return True
    if goal == "max":
        return current_best >= screen_best
    return current_best <= screen_best
