"""Generic genetic operators over a typed candidate model.

The loop this package shipped with had no genetic operators at all: it rendered
the Pareto front as text and asked the model to author whole configurations.
That is whole-artifact rewrite, and it is measurably worse than uniform random
sampling on every genome length we have tested (advantage_theory sweep,
2026-08-03: greedy authoring -0.086 to -0.531 excess capture, population with
recombination +0.0042 to +0.1798).

Everything here is workload-agnostic. A locus is an addressable position inside a
candidate, derived from the problem's own ``candidate_model``; no module in this
package knows what a locus *means*. Sequence-valued fields expand element-wise so
that recombination is defined where the structure actually varies, which is the
whole reason a genome differs from a paragraph of text.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "Locus",
    "loci_of",
    "read_locus",
    "write_locus",
    "locus_domain",
    "locus_is_projected",
    "SamplingPrior",
    "DomainRestriction",
    "EliteMixture",
    "crossover",
    "mutate",
    "one_mutation_neighbourhood",
    "local_probe_candidate",
    "uniform_candidate",
    "coverage_counts",
    "coverage_candidate",
    "incumbent_candidate",
    "tournament",
    "crowding_distances",
    "truncation_survival",
]


@dataclass(frozen=True, slots=True)
class Locus:
    """An addressable position in a candidate.

    ``field`` names a top-level key; ``index`` is set only for sequence-valued
    fields, where each element is independently heritable.
    """

    field: str
    index: int | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.field if self.index is None else f"{self.field}[{self.index}]"


def loci_of(config: Mapping[str, Any]) -> tuple[Locus, ...]:
    """Every heritable position in *config*, derived from the value shapes.

    Scalars contribute one locus. Sequences contribute one per element, because
    a genome whose only locus is "the whole list" cannot be recombined -- which
    is exactly the degenerate case that makes whole-artifact rewrite the only
    available move.
    """

    out: list[Locus] = []
    for field, value in config.items():
        if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            out.extend(Locus(field, i) for i in range(len(value)))
        else:
            out.append(Locus(field))
    return tuple(out)


def read_locus(config: Mapping[str, Any], locus: Locus) -> Any:
    value = config[locus.field]
    return value if locus.index is None else value[locus.index]


def write_locus(config: Mapping[str, Any], locus: Locus, value: Any) -> dict[str, Any]:
    """Return a copy of *config* with *locus* set to *value*. Never mutates."""

    out = {k: (list(v) if isinstance(v, (list, tuple)) and not isinstance(v, (str, bytes)) else v)
           for k, v in config.items()}
    if locus.index is None:
        out[locus.field] = value
    else:
        out[locus.field][locus.index] = value
    return out


class SamplingPrior(Protocol):
    """What a sampler needs from a prior over WHERE to sample.

    ``narrow`` confines a declared domain -- always to a subset, never to
    nothing (see :class:`DomainRestriction` for the enforced invariants).
    ``weights_for`` grades the values a draw is about to choose among;
    returning ``None`` means "uniform", and the sampler then draws through
    exactly the call it used before graded priors existed, so a prior with
    nothing to say costs nothing and changes nothing.
    """

    def narrow(self, locus: "Locus", domain: tuple[Any, ...]) -> tuple[Any, ...]:
        ...

    def weights_for(
        self, locus: "Locus", values: Sequence[Any]
    ) -> "tuple[float, ...] | None":
        ...


@dataclass(frozen=True)
class DomainRestriction:
    """A narrowing of declared locus domains -- a prior over where to sample.

    Every operator that draws a value reads its domain through
    :func:`locus_domain`, so restricting there restricts initialization and
    mutation together without either operator knowing a restriction exists.

    Two invariants, both enforced rather than documented. A restriction can only
    ever *narrow*: the result is always a subset of what the problem declared,
    because a caller that could widen a domain would be authoring values the
    problem never allowed. And it can never empty a domain: an empty
    intersection means the restriction is about a schema it does not match, so
    the declared domain is returned unchanged and the miss is counted in
    :attr:`misses` for the caller to act on. Silently sampling nothing, or
    sampling outside the schema, are both worse than ignoring a bad prior.
    """

    allowed: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list, compare=False, repr=False)

    def narrow(self, locus: "Locus", domain: tuple[Any, ...]) -> tuple[Any, ...]:
        want = self.allowed.get(locus.field)
        if want is None or not domain:
            return domain
        kept = tuple(v for v in domain if v in tuple(want))
        if not kept:
            self.misses.append(locus.field)
            return domain
        return kept

    def weights_for(self, locus: "Locus", values: Sequence[Any]) -> None:
        """Uniform over the narrowed support: the hard 0/1 special case."""

        return None


class EliteMixture:
    """``(1-eps)*prior + eps*front-marginal`` at the value draw, two seams only.

    The repair for a measured repulsion, after its triggered form was measured
    and killed. The two runs that ever reached the analog venue's
    fully-feasible plateau then abandoned it (3.4% and 12.4% post-hit
    occupancy against a 44/44-feasible one-step neighbourhood), because every
    prior-mediated resample pulls toward mass the prior assigned BEFORE the
    discovery existed -- 8 of the discovery's 24 fields sat above the
    installed prior's lid. The triggered burst that tried to fix this missed
    its own bar (4W/8L): its trigger fired ~58 times a run, and the census
    that followed showed why no trigger can work -- at entry time a jackpot is
    indistinguishable from routine front entries by any domination or record
    statistic (jackpots purged 0-1 of their fronts and improved 1-2 bests,
    inside a class of 1,400+ routine events).

    So: no trigger. With mixing weight ``epsilon``, a draw's value
    distribution becomes a mixture of the prior in force and the empirical
    marginal of the CURRENT nondominated front, per locus, over exactly the
    values the draw was already choosing among. Where the prior is right the
    front agrees with it and the mixture is nearly the prior; where the front
    has found something the prior repels from, the discovery's values keep
    circulating. This is the classical estimation-of-distribution move, and
    it also widens the measured 1/24-per-locus prior-to-offspring bandwidth
    the revision channel was blocked by.

    Deliberately narrow: ``narrow`` delegates to the base prior untouched (the
    mixture never reopens an exclusion -- that is the floor knob's job, and it
    is a different, measured decision); a locus on which the front holds no
    declared value returns the base's own answer, INCLUDING ``None`` -- so a
    draw that was unweighted stays on the unweighted ``r.choice`` stream and
    an off-front run does not drift byte-wise. The loop threads this wrapper
    into exactly two seams -- mutation and intensification's unpinned resample
    -- and nowhere else: gen-0 fills, polish enumeration, authored generators,
    and the revision channel all keep the raw prior.
    """

    def __init__(self, epsilon: float) -> None:
        eps = float(epsilon)
        if not 0.0 < eps < 1.0:
            raise ValueError(
                f"elite_mix must be a mixing weight in (0, 1), got {epsilon!r}")
        self.epsilon = eps
        self.base: "SamplingPrior | None" = None
        self.front_rows: "Sequence[Mapping[str, Any]]" = ()
        self.opined = 0                 # draws where the front had an opinion

    def over(self, base: "SamplingPrior | None") -> "SamplingPrior | None":
        """The prior draws should go through, given the one in force."""

        if base is None:
            return None                 # mixture rides a prior, never replaces one
        self.base = base
        return self

    def narrow(self, locus: "Locus", domain: tuple[Any, ...]) -> tuple[Any, ...]:
        assert self.base is not None
        return self.base.narrow(locus, domain)

    def weights_for(
        self, locus: "Locus", values: Sequence[Any]
    ) -> "tuple[float, ...] | None":
        assert self.base is not None
        base_w = None
        weigher = getattr(self.base, "weights_for", None)
        if callable(weigher):
            base_w = weigher(locus, values)
        counts = []
        for value in values:
            n = 0
            for row in self.front_rows:
                try:
                    if read_locus(row, locus) == value:
                        n += 1
                except (KeyError, IndexError, TypeError):
                    continue            # a row that does not carry the locus
            counts.append(n)
        total = sum(counts)
        if total == 0:
            return base_w
        self.opined += 1
        if base_w is None:
            base_norm = [1.0 / len(values)] * len(values)
        else:
            s = sum(base_w)
            base_norm = ([w / s for w in base_w] if s > 0.0
                         else [1.0 / len(values)] * len(values))
        eps = self.epsilon
        return tuple((1.0 - eps) * b + eps * (c / total)
                     for b, c in zip(base_norm, counts))


def locus_domain(
    candidate_model: Any,
    locus: Locus,
    *,
    restriction: "SamplingPrior | None" = None,
) -> tuple[Any, ...]:
    """Allowed values for *locus*, read from the problem's own schema.

    Returns ``()`` when the schema does not constrain the value to a finite set;
    callers fall back to recombination-only for that locus rather than inventing
    a domain the problem never declared.

    *restriction* narrows the declared domain (see :class:`DomainRestriction`).
    Omitted, the result is exactly what the schema declares.
    """

    domain = _declared_domain(candidate_model, locus)
    if restriction is None:
        return domain
    return restriction.narrow(locus, domain)


#: Points a bounded numeric axis is projected onto when the schema says nothing
#: else. Integers get 64: an axis like ``ge=0, le=1000`` keeps a tenth of a
#: percent of resolution while its domain still renders in a prompt. Numbers get
#: 16, which is the grid ``from_pymoo`` has quantized continuous decision
#: variables onto since it shipped -- one resolution rule for the package, not
#: two. Both are overridable per field (see :func:`_grid_override`).
_INTEGER_GRID = 64
_NUMBER_GRID = 16

#: A per-field grid outside this range is not a search space: one point is a
#: constant, and a domain nobody can render or enumerate cheaply is not a
#: domain an operator can draw from.
_GRID_MIN, _GRID_MAX = 2, 256

#: Slack for reading multiples off a float bound, whose decimal literal in the
#: schema (0.3) is not the binary value the arithmetic produces.
_TOL = 1e-12


def _quantum(value: float) -> float:
    """A grid float rendered the way every reader will see it.

    Ten significant digits: enough that no two points of a 256-point grid
    collide, few enough that ``repr`` is stable, so a value in a prompt, a value
    in a card and a value compared against a domain are the same string and the
    same float.
    """

    return float(f"{value:.10g}")


def _bound(node: Mapping[str, Any], key: str) -> float | None:
    """A real numeric bound under *key*, or ``None``. ``True`` is not a bound."""

    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _grid_override(node: Mapping[str, Any]) -> int | None:
    """The field's own point count, from ``{"agent_evolve": {"grid": N}}``.

    Written by the problem author as
    ``Field(json_schema_extra={"agent_evolve": {"grid": N}})``: the one place a
    schema can say how finely *this* axis deserves to be searched.
    """

    extra = node.get("agent_evolve")
    if isinstance(extra, Mapping):
        wanted = extra.get("grid")
        if isinstance(wanted, int) and not isinstance(wanted, bool):
            return max(_GRID_MIN, min(_GRID_MAX, wanted))
    return None


def _spread(count: int, points: int) -> tuple[int, ...]:
    """*points* indices spread over ``range(count)``, both extremes included.

    Fewer values than points means every value: a projection never invents
    resolution the range does not have.
    """

    if count <= points:
        return tuple(range(count))
    return tuple(dict.fromkeys(
        round(i * (count - 1) / (points - 1)) for i in range(points)
    ))


def _multiple_indices(
    lo: float, hi: float, step: float, *, open_lo: bool, open_hi: bool
) -> tuple[int, int]:
    """First and last multiple-of-*step* indices inside the interval."""

    if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
        first, last = -((-lo) // step), hi // step   # exact for ints of any size
    else:
        first = math.ceil(lo / step - _TOL)
        last = math.floor(hi / step + _TOL)
    if open_lo and first * step <= lo:
        first += 1
    if open_hi and last * step >= hi:
        last -= 1
    return first, last


def _integer_domain(node: Mapping[str, Any], points: int) -> tuple[int, ...]:
    """Every integer the bounds allow, or *points* of them evenly spaced.

    ``exclusiveMinimum``/``exclusiveMaximum`` tighten by one, which is what they
    mean on an integer axis. ``multipleOf`` restricts to multiples, so the
    emitted values are ones the schema would actually accept.
    """

    lows: list[int] = []
    highs: list[int] = []
    closed_lo = _bound(node, "minimum")
    if closed_lo is not None:
        lows.append(math.ceil(closed_lo))
    open_lo = _bound(node, "exclusiveMinimum")
    if open_lo is not None:
        lows.append(math.floor(open_lo) + 1)
    closed_hi = _bound(node, "maximum")
    if closed_hi is not None:
        highs.append(math.floor(closed_hi))
    open_hi = _bound(node, "exclusiveMaximum")
    if open_hi is not None:
        highs.append(math.ceil(open_hi) - 1)
    if not lows or not highs:
        return ()                           # unbounded on one side: no finite reading
    lo, hi = max(lows), min(highs)
    step = 1
    multiple = _bound(node, "multipleOf")
    if multiple is not None:
        if multiple <= 0 or not float(multiple).is_integer():
            return ()                       # no integer axis reading of this step
        step = int(multiple)
    first, last = _multiple_indices(lo, hi, step, open_lo=False, open_hi=False)
    if first > last:
        return ()                           # the bounds admit nothing
    return tuple((first + i) * step for i in _spread(last - first + 1, points))


def _number_domain(node: Mapping[str, Any], points: int) -> tuple[float, ...]:
    """A *points*-point grid across a doubly bounded continuous axis.

    Inclusive bounds put a point on each endpoint. An excluded endpoint is not
    a legal value, so it becomes one more interval instead: an open interval of
    ``points`` interior points sits on ``points + 1`` intervals.
    """

    lo, open_lo = _bound(node, "minimum"), False
    tighter = _bound(node, "exclusiveMinimum")
    if tighter is not None and (lo is None or tighter >= lo):
        lo, open_lo = tighter, True
    hi, open_hi = _bound(node, "maximum"), False
    tighter = _bound(node, "exclusiveMaximum")
    if tighter is not None and (hi is None or tighter <= hi):
        hi, open_hi = tighter, True
    if lo is None or hi is None or hi < lo:
        return ()                           # one-sided or empty: nothing finite
    multiple = _bound(node, "multipleOf")
    if multiple is not None:
        if multiple <= 0:
            return ()
        first, last = _multiple_indices(lo, hi, multiple,
                                        open_lo=open_lo, open_hi=open_hi)
        if first > last:
            return ()
        return tuple(dict.fromkeys(
            _quantum((first + i) * multiple)
            for i in _spread(last - first + 1, points)
        ))
    if lo == hi:
        return () if (open_lo or open_hi) else (_quantum(lo),)
    edges = points + int(open_lo) + int(open_hi)
    start = 1 if open_lo else 0
    return tuple(dict.fromkeys(
        _quantum(lo + (hi - lo) * i / (edges - 1))
        for i in range(start, start + points)
    ))


def _node_domain(
    node: Mapping[str, Any], override: int | None
) -> tuple[tuple[Any, ...], bool]:
    """One schema node's finite domain, and whether it was projected.

    Projected means "read off bounds": the schema declared a range, not a set of
    values, and what comes back is a finite sample of that range. Enumerations,
    booleans and constants are declared outright and are not projections.
    """

    if "enum" in node:
        return tuple(node["enum"]), False
    if "const" in node:
        # A one-value domain is still a domain. Read as nothing, a Literal of
        # length one freezes its locus and every reader calls it undeclared.
        return (node["const"],), False
    declared = node.get("type")
    types = tuple(declared) if isinstance(declared, list) else (declared,)
    if "boolean" in types:
        # A boolean is a finite domain. Without this, a bool field has no
        # declared values, mutate() leaves it alone, and a real design axis is
        # silently frozen at whatever the seed happened to carry.
        return (False, True), False
    points = _grid_override(node) or override
    if "integer" in types:
        return _integer_domain(node, points or _INTEGER_GRID), True
    if "number" in types:
        return _number_domain(node, points or _NUMBER_GRID), True
    return (), False


def _declared_domain_detail(
    candidate_model: Any, locus: Locus
) -> tuple[tuple[Any, ...], bool]:
    """The schema's own domain for *locus*, and whether it is a projection."""

    if candidate_model is None:
        return (), False
    try:
        schema = candidate_model.model_json_schema()
    except Exception:                       # not a pydantic model
        return (), False
    defs = schema.get("$defs", {}) or schema.get("definitions", {}) or {}
    node = (schema.get("properties", {}) or {}).get(locus.field)
    if node is None:
        return (), False

    def resolve(n: Any) -> Any:
        seen = 0
        while isinstance(n, dict) and "$ref" in n and seen < 10:
            ref = n["$ref"].rsplit("/", 1)[-1]
            n = defs.get(ref, {})
            seen += 1
        return n

    node = resolve(node)
    if locus.index is not None:
        node = resolve(node.get("items", {}) if isinstance(node, dict) else {})
    elif isinstance(node, dict) and node.get("type") == "array":
        # A bare Locus("field") naming a sequence field means the field's
        # shared per-element vocabulary -- what pooled attribution and
        # field-keyed priors reason over. Indexed loci resolve above.
        node = resolve(node.get("items", {}))
    if not isinstance(node, dict):
        return (), False
    override = _grid_override(node)         # a wrapper carries the field's own grid
    domain, projected = _node_domain(node, override)
    if domain:
        return domain, projected
    for key in ("anyOf", "oneOf"):
        for branch in node.get(key, ()) or ():
            branch = resolve(branch)
            if isinstance(branch, dict) and "enum" in branch:
                return tuple(branch["enum"]), False
    for key in ("anyOf", "oneOf"):
        for branch in node.get(key, ()) or ():
            branch = resolve(branch)
            # A "null" branch is Optional's absence marker, not a value to
            # sample: writing None where the problem declared a range is how a
            # sampler produces candidates the problem never allowed.
            if not isinstance(branch, dict) or branch.get("type") == "null":
                continue
            domain, projected = _node_domain(branch, override)
            if domain:
                return domain, projected
    return (), False


def _declared_domain(candidate_model: Any, locus: Locus) -> tuple[Any, ...]:
    """The schema's own domain for *locus*, before any narrowing."""

    return _declared_domain_detail(candidate_model, locus)[0]


def locus_is_projected(candidate_model: Any, locus: Locus) -> bool:
    """True when *locus*'s domain is a finite projection of a declared range.

    A projected domain is searchable but lossy: the optimizer moves on the grid,
    not the continuum. Reports that name domain sizes say so, because "16
    values" means something different when the schema declared 16 values than
    when it declared an interval.
    """

    domain, projected = _declared_domain_detail(candidate_model, locus)
    return bool(domain) and projected


def _draw(
    r: random.Random,
    values: tuple[Any, ...],
    locus: Locus,
    restriction: "SamplingPrior | None",
) -> Any:
    """Draw one of *values*, honouring the prior's weights when it has any.

    The unweighted path is literally ``r.choice`` -- same call, same RNG
    stream -- which is what keeps every run without a graded prior
    byte-identical to the pre-weights seam (the fossil test holds this).
    """

    weights = None
    if restriction is not None:
        weigher = getattr(restriction, "weights_for", None)
        if callable(weigher):
            weights = weigher(locus, values)
    if weights is None:
        return r.choice(values)
    return r.choices(values, weights=weights, k=1)[0]


def crossover(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    *,
    mask: Sequence[bool] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Recombine two parents locus by locus.

    ``mask[i]`` true takes locus *i* from *parent_b*. A caller-supplied mask is
    how an external chooser -- a model, a heuristic -- expresses *where to cut*
    without ever authoring a candidate. With no mask, each locus is drawn
    independently at random, which is the unguided control.

    The mask is indexed by *parent_a*'s loci, and its length must match them:
    a mask of the wrong length is a caller bug and is refused by name rather
    than silently reinterpreted.

    **Ragged genomes.** When a field holds a sequence, each element is a locus,
    so two candidates of the same problem may have different genome lengths.
    A locus that *parent_b* does not have cannot be inherited from it, so the
    child keeps *parent_a*'s value there. Recombination is therefore defined on
    the loci the parents share, which reduces exactly to the fixed-length case
    when the shapes match.
    """

    loci = loci_of(parent_a)
    if mask is None:
        r = rng or random.Random()
        mask = [bool(r.getrandbits(1)) for _ in loci]
    if len(mask) != len(loci):
        raise ValueError(
            f"mask has {len(mask)} bits but the candidate has {len(loci)} loci"
        )
    donor = frozenset(loci_of(parent_b))
    child = dict(parent_a)
    for locus, take_b in zip(loci, mask):
        if take_b and locus in donor:
            child = write_locus(child, locus, read_locus(parent_b, locus))
    return child


def mutate(
    config: Mapping[str, Any],
    candidate_model: Any,
    *,
    rate: float | None = None,
    loci: Iterable[Locus] | None = None,
    rng: random.Random | None = None,
    restriction: SamplingPrior | None = None,
) -> dict[str, Any]:
    """Resample loci from their declared domains.

    *loci* names exactly which positions to change -- how an external chooser
    expresses *which loci* without authoring a candidate. Otherwise every locus
    is resampled independently with probability *rate*, defaulting to 1/n so a
    candidate changes in about one place.
    """

    r = rng or random.Random()
    all_loci = loci_of(config)
    targets = tuple(loci) if loci is not None else tuple(
        lc for lc in all_loci
        if r.random() < (rate if rate is not None else 1.0 / max(1, len(all_loci)))
    )
    out = dict(config)
    for locus in targets:
        domain = locus_domain(candidate_model, locus, restriction=restriction)
        if not domain:
            continue                        # undeclared domain: leave it alone
        current = read_locus(out, locus)
        choices = tuple(v for v in domain if v != current) or domain
        out = write_locus(out, locus, _draw(r, choices, locus, restriction))
    return out


def _nearness(value: Any, current: Any) -> float:
    """How far *value* sits from *current* on a numeric axis; 0 where none exists.

    A projected domain is a GRID, and the move that decides an exact optimum is
    one step along it, so a neighbourhood enumerated nearest-first tries that
    step before it tries the far end of the axis. Values with no distance --
    enumerations, strings, booleans -- all read 0 and therefore keep the
    schema's own declared order under a stable sort.
    """

    if isinstance(value, bool) or isinstance(current, bool):
        return 0.0
    if isinstance(value, (int, float)) and isinstance(current, (int, float)):
        return abs(float(value) - float(current))
    return 0.0


def one_mutation_neighbourhood(
    config: Mapping[str, Any],
    candidate_model: Any,
    *,
    restriction: SamplingPrior | None = None,
) -> Iterable[dict[str, Any]]:
    """Every candidate exactly one declared value away from *config*, in order.

    This is the deterministic complement to :func:`mutate`: same move, no RNG,
    and the whole neighbourhood rather than one sample of it. It exists for the
    endgame, where a population has already found the right region and the only
    thing left is the last grid step -- measured, NSGA-II reaches the EXACT
    optimum on 6 of 10 NAS seeds where our loop reaches within 10% on 9 of 10,
    by spending its endgame enumerating neighbours while a breeding loop keeps
    recombining a front that is already solved.

    The order is fixed and free of chance: loci in :func:`loci_of` order, and
    within a locus the declared values nearest the current one first. Loci the
    schema does not constrain contribute nothing -- there is no neighbour to
    name where there is no declared domain.

    A generator, not a list: the neighbourhood of a long genome over a
    256-point grid is tens of thousands of candidates, and a caller that wants
    the first handful should pay for the first handful.
    """

    for locus in loci_of(config):
        domain = locus_domain(candidate_model, locus, restriction=restriction)
        if not domain:
            continue                        # undeclared domain: no neighbour
        current = read_locus(config, locus)
        for value in sorted((v for v in domain if v != current),
                            key=lambda v: _nearness(v, current)):
            yield write_locus(config, locus, value)


def local_probe_candidate(
    anchor: Mapping[str, Any],
    candidate_model: Any,
    *,
    rng: random.Random,
    span: int = 2,
) -> dict[str, Any]:
    """*anchor* with ONE locus moved to one of its *span* nearest other values.

    The consolidation move, and the measured defect it answers: when a run
    finds a configuration that advances an objective's best, nothing in the
    loop looks next door. Both runs that ever reached the analog venue's
    fully-feasible plateau then spent 3.4% and 12.4% of their remaining
    charges on it -- while every one of the discovery's 44 one-step
    neighbours was measured (post hoc) to be on the plateau too. The
    incumbent path could not hold it: an intensification resamples its
    unpinned loci THROUGH THE PRIOR, and the prior actively repels from a
    discovery it never predicted (8 of that discovery's 24 fields sat above
    the installed prior's lid, at floor-level mass at best). This function is
    the prior-free complement: keep everything, move one locus one or two
    declared steps.

    No restriction, deliberately, for the same reason
    :func:`coverage_candidate` takes none: the whole point of the probe is to
    hold a region the prior may be wrong about. Nearness is
    :func:`one_mutation_neighbourhood`'s -- numeric axes by distance,
    unordered domains falling back to the schema's own declared order under a
    stable sort -- so a probe on a grid tries the adjacent rungs first and a
    probe on an enumeration tries its declared neighbours. Loci with fewer
    than two declared values cannot move and are never picked; an anchor with
    no movable locus at all comes back unchanged, which the caller's dedup
    then correctly refuses.
    """

    movable = []
    for locus in loci_of(anchor):
        domain = locus_domain(candidate_model, locus)
        if domain and len(domain) >= 2:
            movable.append((locus, domain))
    if not movable:
        return dict(anchor)
    locus, domain = movable[rng.randrange(len(movable))]
    current = read_locus(anchor, locus)
    nearest = sorted((v for v in domain if v != current),
                     key=lambda v: _nearness(v, current))[:max(1, span)]
    return write_locus(anchor, locus, nearest[rng.randrange(len(nearest))])


def uniform_candidate(
    template: Mapping[str, Any],
    candidate_model: Any,
    *,
    rng: random.Random | None = None,
    restriction: SamplingPrior | None = None,
) -> dict[str, Any]:
    """A fresh draw over every locus with a declared domain.

    Loci the schema does not constrain keep the template's value: inventing
    values the problem never declared is not sampling, it is guessing. When no
    locus declares a domain at all, fall back to :func:`mutate` so the caller
    still gets diversity rather than a copy.

    This exists because filling a population with *mutants of one seed* builds
    an anchored cloud around that seed. Measured on a third-party optimizer,
    correcting exactly this anchor moved its result from +0.095 (loses badly to
    uniform) to +0.0066 (parity) -- and the standard seed on at least one of
    our own workloads scores worse than a typical uniform draw.
    """

    r = rng or random.Random()
    out = dict(template)
    drew = False
    for locus in loci_of(template):
        domain = locus_domain(candidate_model, locus, restriction=restriction)
        if domain:
            out = write_locus(out, locus, _draw(r, domain, locus, restriction))
            drew = True
    if not drew:
        return mutate(template, candidate_model, rng=r, restriction=restriction)
    return out


def coverage_counts(
    rows: Iterable[Mapping[str, Any]],
    template: Mapping[str, Any],
    candidate_model: Any,
    *,
    counts: "dict[Locus, list[int]] | None" = None,
) -> "dict[Locus, list[int]]":
    """How many of *rows* carry each DECLARED value of each locus.

    The tally is aligned to :func:`locus_domain`'s declared order, one slot per
    declared value, and it is NEVER narrowed by a prior. That is the
    load-bearing choice rather than an oversight: a coverage channel that
    counts only the values a restriction still allows cannot notice that the
    region it was handed is the wrong one. Measured on the analog venue, every
    installed prior's ceiling was level 13-15 of a 23-32 level ladder, the best
    in-box evaluation of 12,899 reads -0.4002 while the best outside reads
    -0.0566, and 100 of the pooled top-100 carry at least one coordinate the
    box excludes (INVESTIGATION.md, S5). Counting inside the box would have
    reported that venue fully covered.

    Loci come from *template*, so a ragged row that does not carry one
    contributes nothing to it rather than an invented zero, and loci the schema
    does not constrain get no tally at all -- there is no coverage to speak of
    where there are no declared values to cover.

    *counts*, when given, is EXTENDED by *rows* in place and returned: a caller
    that wants this generation's own draws counted against the next one does
    not re-walk the whole trace to do it.
    """

    loci = loci_of(template)
    domains = {lc: locus_domain(candidate_model, lc) for lc in loci}
    tally: dict[Locus, list[int]] = {} if counts is None else counts
    for locus, domain in domains.items():
        if domain and (locus not in tally or len(tally[locus]) != len(domain)):
            tally[locus] = [0] * len(domain)
    for row in rows:
        for locus, domain in domains.items():
            if not domain:
                continue
            try:
                value = read_locus(row, locus)
            except (KeyError, IndexError, TypeError):
                continue                    # this row does not carry this locus
            for index, declared in enumerate(domain):
                if declared == value:
                    tally[locus][index] += 1
                    break
    return tally


def coverage_candidate(
    template: Mapping[str, Any],
    candidate_model: Any,
    *,
    counts: "Mapping[Locus, Sequence[int]]",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """A fresh candidate drawn from each locus's LEAST-MEASURED declared values.

    Every locus with a declared domain is resampled -- this is a fresh
    configuration, not a mutant -- and the values it may be resampled to are
    exactly the ones *counts* has seen fewest of. Ties draw uniformly among the
    tied values, so a run that has measured nothing yet draws uniformly over
    the whole declared domain and the channel degrades to what it should
    degrade to.

    No restriction and no weights. The prior is the box, and the whole point of
    this draw is to leave it: a coverage draw graded by the prior's own mass
    would spend its slot re-covering the region the prior already prefers.
    Loci the schema does not constrain keep the template's value, for the same
    reason :func:`uniform_candidate` does -- inventing values the problem never
    declared is not sampling.
    """

    r = rng or random.Random()
    out = dict(template)
    for locus in loci_of(template):
        domain = locus_domain(candidate_model, locus)
        if not domain:
            continue
        seen = tuple(counts.get(locus) or ())
        if len(seen) != len(domain):
            seen = (0,) * len(domain)       # no tally for it is no coverage
        fewest = min(seen)
        least = tuple(value for value, count in zip(domain, seen)
                      if count == fewest)
        out = write_locus(out, locus, r.choice(least))
    return out


def incumbent_candidate(
    incumbent: Mapping[str, Any],
    candidate_model: Any,
    *,
    pin: Iterable[Locus],
    rng: random.Random | None = None,
    restriction: SamplingPrior | None = None,
) -> dict[str, Any]:
    """*incumbent* with the loci in *pin* held, every other locus resampled.

    The classical reading of what a third-party optimizer's winning population
    actually is. Measured against HEBO's own top decile: one incumbent-anchored
    cluster in 6 of 6 cells at single-linkage Hamming <= 10, zero of 156
    entropy-controlled coordinate pairs above z = 2, and a modal purity of
    0.695-0.842 over 6-12 fields (INVESTIGATION.md, S5). That is not a learned
    coupling table -- it is one member with most of its fields frozen and the
    rest redrawn, which is what this function is.

    The unpinned loci are drawn through *restriction* exactly as
    :func:`uniform_candidate` draws them, so an intensification samples the
    prior in force rather than a second distribution nobody declared. Loci with
    no declared domain keep the incumbent's value whether they were pinned or
    not: there is nothing to resample them from.
    """

    r = rng or random.Random()
    held = frozenset(pin)
    out = dict(incumbent)
    for locus in loci_of(incumbent):
        if locus in held:
            continue
        domain = locus_domain(candidate_model, locus, restriction=restriction)
        if not domain:
            continue
        out = write_locus(out, locus, _draw(r, domain, locus, restriction))
    return out


def tournament(
    population: Sequence[tuple[Mapping[str, Any], float]],
    *,
    size: int = 2,
    rng: random.Random | None = None,
) -> Mapping[str, Any]:
    """Pick one parent by a *size*-way tournament on the ranking value (lower wins)."""

    if not population:
        raise ValueError("cannot select a parent from an empty population")
    r = rng or random.Random()
    contenders = r.sample(list(population), min(size, len(population)))
    return min(contenders, key=lambda pair: pair[1])[0]


def crowding_distances(
    vectors: Sequence[Mapping[str, float]]
) -> list[float]:
    """NSGA-II crowding distance for one tied set, one number per member.

    Per objective, sort the set: the two boundary members are infinitely alone
    on that axis and take ``inf``; every interior member accumulates the gap
    between its two neighbours, divided by the objective's range over the set
    so that objectives measured in seconds and objectives measured in watts
    contribute comparably. An axis on which the whole set is constant separates
    nobody and is skipped.

    Direction never enters. Crowding measures how alone a point is, and a
    minimised axis is exactly as wide as a maximised one -- which is also why
    this needs objective VECTORS and cannot be computed from a ranking value.
    """

    n = len(vectors)
    if n <= 2:
        return [math.inf] * n               # every member is a boundary
    distances = [0.0] * n
    for name in sorted({key for vec in vectors for key in vec}):
        column = [vec.get(name) for vec in vectors]
        if any(value is None for value in column):
            continue                        # not an axis the whole set carries
        values = [float(value) for value in column]
        order = sorted(range(n), key=lambda i: values[i])
        span = values[order[-1]] - values[order[0]]
        if span <= 0:
            continue
        distances[order[0]] = math.inf
        distances[order[-1]] = math.inf
        for position in range(1, n - 1):
            index = order[position]
            if distances[index] == math.inf:
                continue                    # a boundary elsewhere stays one
            distances[index] += (values[order[position + 1]]
                                 - values[order[position - 1]]) / span
    return distances


def truncation_survival(
    population: Sequence[tuple[Mapping[str, Any], float]],
    *,
    keep: int,
    key_of,
    method: str = "count",
    objectives_of=None,
) -> list[tuple[Mapping[str, Any], float]]:
    """Deduplicate by candidate identity, then keep the *keep* best.

    Deduplication is not cosmetic: without it a population collapses onto copies
    of one genome and recombination stops producing anything new, which is the
    failure mode a diversity-free loop reaches within a couple of generations.

    *method* decides what happens to members the ranking value cannot separate.
    ``"count"`` -- the default -- keeps them in the order they were measured,
    which is exactly what this function did before the knob existed and is
    therefore byte-identical. ``"crowding"`` breaks ties WITHIN a domination
    count by :func:`crowding_distances`, keeping the members that are most
    alone. That matters where most of the population is non-dominated: on the
    five-objective fleet unit almost nothing dominates anything, count-only
    survival is then near-random, and the unit hosts no claim at all.

    ``"crowding"`` needs *objectives_of*: a callable from a population item to
    its objective vector. Crowding lives in objective space and a ranking value
    is a scalar summary of it, so the vectors have to be supplied rather than
    reconstructed.
    """

    if method not in ("count", "crowding"):
        raise ValueError(
            f"method must be 'count' or 'crowding', got {method!r}")
    best: dict[str, tuple[Mapping[str, Any], float]] = {}
    for config, value in population:
        k = key_of(config)
        if k not in best or value < best[k][1]:
            best[k] = (config, value)
    if method == "count":
        return sorted(best.values(), key=lambda pair: pair[1])[:keep]
    if objectives_of is None:
        raise ValueError(
            "method='crowding' needs objectives_of: crowding distance is "
            "measured in objective space, and a ranking value alone cannot "
            "say how alone a point is"
        )
    tied: dict[float, list[tuple[Mapping[str, Any], float]]] = {}
    for pair in best.values():
        tied.setdefault(pair[1], []).append(pair)
    ordered: list[tuple[Mapping[str, Any], float]] = []
    for value in sorted(tied):
        group = tied[value]
        if len(group) == 1:
            ordered.extend(group)
            continue
        spread = crowding_distances([objectives_of(pair[0]) for pair in group])
        # Stable on equal distance, so a tie the crowding rule also cannot
        # break keeps the order it was measured in rather than an arbitrary one.
        ordered.extend(pair for _d, pair in sorted(
            zip(spread, group), key=lambda item: -item[0]))
    return ordered[:keep]
