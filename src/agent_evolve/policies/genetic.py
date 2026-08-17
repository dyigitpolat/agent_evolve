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

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "Locus",
    "loci_of",
    "read_locus",
    "write_locus",
    "locus_domain",
    "SamplingPrior",
    "DomainRestriction",
    "crossover",
    "mutate",
    "uniform_candidate",
    "tournament",
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


def _declared_domain(candidate_model: Any, locus: Locus) -> tuple[Any, ...]:
    """The schema's own domain for *locus*, before any narrowing."""

    if candidate_model is None:
        return ()
    try:
        schema = candidate_model.model_json_schema()
    except Exception:                       # not a pydantic model
        return ()
    defs = schema.get("$defs", {}) or schema.get("definitions", {}) or {}
    node = (schema.get("properties", {}) or {}).get(locus.field)
    if node is None:
        return ()

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
        return ()
    if "enum" in node:
        return tuple(node["enum"])
    if node.get("type") == "boolean":
        # A boolean is a finite domain. Without this, a bool field has no
        # declared values, mutate() leaves it alone, and a real design axis is
        # silently frozen at whatever the seed happened to carry.
        return (False, True)
    for key in ("anyOf", "oneOf"):
        for branch in node.get(key, ()) or ():
            branch = resolve(branch)
            if isinstance(branch, dict) and "enum" in branch:
                return tuple(branch["enum"])
    return ()


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


def truncation_survival(
    population: Sequence[tuple[Mapping[str, Any], float]],
    *,
    keep: int,
    key_of,
) -> list[tuple[Mapping[str, Any], float]]:
    """Deduplicate by candidate identity, then keep the *keep* best.

    Deduplication is not cosmetic: without it a population collapses onto copies
    of one genome and recombination stops producing anything new, which is the
    failure mode a diversity-free loop reaches within a couple of generations.
    """

    best: dict[str, tuple[Mapping[str, Any], float]] = {}
    for config, value in population:
        k = key_of(config)
        if k not in best or value < best[k][1]:
            best[k] = (config, value)
    return sorted(best.values(), key=lambda pair: pair[1])[:keep]
