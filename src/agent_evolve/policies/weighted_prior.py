"""Graded sampling priors: the generic form of domain restriction.

A hard :class:`~agent_evolve.policies.genetic.DomainRestriction` is a 0/1 bet:
sample these values, never those. The generic form carries per-value WEIGHTS,
so a proposer can grade its confidence instead of gambling the support -- and
the hard restriction is exactly the special case where every kept value weighs
the same. That equivalence is enforced, not aspirational: ``weights_for``
reports ``None`` whenever the mapped weights are uniform, the sampler then
draws through the identical ``r.choice`` call, and a test pins the two streams
draw-for-draw.

Why grading exists at all: the sealed space-prior arm showed a ~20-line rule
matching a model's hard restriction to the last digit on 15 of 15 seeds -- a
hard subset is arithmetic over the screen table, which is rule territory. A
graded prior is the form that can carry what the table does not say (named
idioms, couplings, physics) while the evaluator still arbitrates: support can
only narrow, misses are counted, seeds survive, and the unwind test applies
wherever the prior actually excludes something. A prior that excludes nothing
makes no refutable claim; it only biases draws, and its worst case is bounded
by the declared domain.

THE PRIOR FLOOR (2026-08-25). A prior that excludes IS refutable in principle
and was not refutable in practice: the loop's unwind test only fires when the
restricted search fails to match a screen point the prior excluded, and the
screen is the very instrument whose blind spot put the exclusion there. Twenty
of twenty analog cells never reopened an excluded value for the rest of the
run, and the region the prior had zeroed held the venue's best configurations.
So every prior installed here now keeps a FLOOR of sampling mass -- see
:data:`PRIOR_FLOOR` and :func:`floor_weights` -- on every value the schema
declares. A wrong box then costs draws instead of costing the answer, and the
run itself can refute it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import Locus
# _domains is package-internal on purpose: both prior proposers must resolve
# declared domains identically, or the same reply would be judged two ways.
from agent_evolve.policies.llm_prior import _domains
from agent_evolve.policies.structure import Attribution, render_attribution

__all__ = [
    "WeightedRestriction",
    "statistical_weighted_prior",
    "WeightedPriorTelemetry",
    "llm_weighted_prior_proposer",
    "floor_weights",
    "PRIOR_FLOOR",
    "PROMPT",
    "COMMITTED_PROMPT",
    "PRIOR_STYLES",
]

#: How the prompt's last instruction reads. See :data:`COMMITTED_PROMPT`.
PRIOR_STYLES = ("cautious", "committed")

#: The smallest share of a locus's sampling mass any DECLARED value may hold
#: once a prior is installed. Both prior entry points here read it at CALL
#: time, so it is the one place the floor is set and the one place a study
#: turns it off (``0.0`` restores the pre-floor arithmetic exactly, byte for
#: byte -- see :func:`floor_weights`).
#:
#: 0.02 is a defect fix and is disclosed as one. CHANGELOG-ready wording:
#: *"Installed sampling priors now keep at least 2% of each locus's mass on
#: every declared value (``weighted_prior.PRIOR_FLOOR``). A prior that zeroed
#: a region the screen never measured previously ended the run's ability to
#: sample there at all -- measured on 20 of 20 analog cells, where the zeroed
#: region held the venue's best configurations. Set ``PRIOR_FLOOR = 0.0`` to
#: restore the previous behaviour exactly."*
#:
#: Why 2% and not 10%: the floor is insurance, not exploration. At the analog
#: venue's 24 loci a 2% floor costs a draw its prior-preferred value on about
#: one locus in two, which is inside the mutation noise the loop already runs
#: with; 10% would be a second exploration mechanism wearing this one's name.
PRIOR_FLOOR = 0.02


@dataclass(frozen=True)
class WeightedRestriction:
    """Per-field weights over declared values -- a graded prior over WHERE to sample.

    ``weighted`` maps a field name to a ``(values, weights)`` pair of parallel
    tuples. Weight zero excludes a value from the support exactly as a hard
    restriction would; positive weights keep values in proportion.

    The :class:`~agent_evolve.policies.genetic.DomainRestriction` invariants
    hold unchanged: narrows only (undeclared values contribute nothing), never
    empties (an empty intersection returns the declared domain and counts the
    miss), misses accumulate on the frozen instance, and the key is
    ``Locus.field`` so sequence elements share their field's entry.
    """

    weighted: Mapping[str, tuple[tuple[Any, ...], tuple[float, ...]]] = field(
        default_factory=dict
    )
    misses: list[str] = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name, entry in dict(self.weighted).items():
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError(
                    f"weighted prior for {name!r} must be a (values, weights) "
                    f"pair, got {entry!r}"
                )
            values, weights = entry
            if len(values) != len(weights):
                raise ValueError(
                    f"weighted prior for {name!r}: {len(values)} values but "
                    f"{len(weights)} weights"
                )
            bad = [
                w for w in weights
                if isinstance(w, bool) or not isinstance(w, (int, float))
                or not math.isfinite(float(w)) or float(w) < 0.0
            ]
            if bad:
                raise ValueError(
                    f"weighted prior for {name!r}: weights must be finite and "
                    f"non-negative, got {bad}"
                )
            if not any(float(w) > 0.0 for w in weights):
                raise ValueError(
                    f"weighted prior for {name!r} gives every value weight "
                    "zero, which is a restriction that samples nothing"
                )

    @classmethod
    def hard(cls, allowed: Mapping[str, Sequence[Any]]) -> "WeightedRestriction":
        """The 0/1 special case, from a hard restriction's vocabulary."""

        return cls({
            name: (tuple(values), tuple(1.0 for _ in values))
            for name, values in dict(allowed).items()
        })

    @property
    def allowed(self) -> dict[str, tuple[Any, ...]]:
        """Positive-weight support, in the hard restriction's vocabulary.

        The loop's structure record and its unwind test read ``allowed``, so a
        graded prior takes part in both through exactly the values it
        excludes. A prior that excludes nothing is unwound by nothing: it made
        no refutable claim, only a bias, and a bias's worst case is bounded.
        """

        return {
            name: tuple(v for v, w in zip(values, weights) if float(w) > 0.0)
            for name, (values, weights) in dict(self.weighted).items()
        }

    def narrow(self, locus: Locus, domain: tuple[Any, ...]) -> tuple[Any, ...]:
        entry = self.weighted.get(locus.field)
        if entry is None or not domain:
            return domain
        values, weights = entry
        support = tuple(v for v, w in zip(values, weights) if float(w) > 0.0)
        kept = tuple(v for v in domain if v in support)
        if not kept:
            self.misses.append(locus.field)
            return domain
        return kept

    def weights_for(
        self, locus: Locus, values: Sequence[Any]
    ) -> "tuple[float, ...] | None":
        """Weights parallel to *values*, or ``None`` when the draw is uniform.

        ``None`` in exactly three situations, each deliberate: the prior has no
        entry for this field; the mapped weights are all equal (the hard
        special case -- the sampler must take the byte-identical ``r.choice``
        path); or a zero-weight value is present in *values*, which can only
        happen after ``narrow`` missed, and a prior that missed does not get to
        bias the fallback domain it failed to narrow.
        """

        entry = self.weighted.get(locus.field)
        if entry is None or not values:
            return None
        table = {v: float(w) for v, w in zip(*entry)}
        mapped = [table.get(v, 0.0) for v in values]
        if any(w <= 0.0 for w in mapped):
            return None
        if all(w == mapped[0] for w in mapped):
            return None
        return tuple(mapped)


def floor_weights(
    values: Sequence[Any],
    weights: Sequence[float],
    *,
    domain: Sequence[Any] | None = None,
    floor: float | None = None,
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """``(values, weights)`` with no declared value below *floor* of the mass.

    One generic operation, used by both prior entry points, so a floored prior
    means the same thing whoever proposed it.

    *domain* is the schema's declared vocabulary for the locus. A value the
    proposal OMITS is an exclusion written as an absence, so the domain is
    folded in at weight zero before the floor is applied -- otherwise the
    cheapest way to zero a region would also be the one way to escape the
    floor. Omit *domain* (the screen-derived prior, whose vocabulary is
    whatever the screen measured) and the entry's own values are the domain.

    The floor is applied as a MIXTURE with the uniform distribution -- ``w' =
    (1 - a) * w + a / k``, with ``a`` the smallest weight that lifts the
    minimum to the floor -- and not by clipping. Two properties follow, and
    both are the reason for the shape:

    * **Order is preserved, strictly.** Clipping ties every below-floor value
      at the floor and throws away the proposal's ranking of exactly the
      values it was least sure about. A mixture keeps ``w_i > w_j`` wherever
      the proposal said so.
    * **The concentration cap survives.** For positive vectors the max/min
      ratio of a convex combination with the uniform vector is at most the
      ratio of the input, so a prior admitted under a ``max_weight_ratio``
      bound is still inside it after flooring. Flooring can only make a prior
      flatter.

    ``floor <= 0`` returns *values* and *weights* unchanged, and so does any
    entry already at or above the floor: the return is the caller's own
    tuples, not a rebuilt copy, so a run with nothing to floor takes the
    pre-floor arithmetic byte for byte. The effective floor is capped at
    ``1/k`` -- k values cannot each hold more than an equal share -- so a floor
    larger than that flattens the locus to uniform rather than failing.
    """

    kept_values = tuple(values)
    kept_weights = tuple(float(w) for w in weights)
    floor = PRIOR_FLOOR if floor is None else float(floor)
    if floor <= 0.0 or not kept_values:
        return tuple(values), tuple(weights)

    table = {v: w for v, w in zip(kept_values, kept_weights)}
    full = tuple(domain) if domain else ()
    if full and all(v in full for v in kept_values):
        # Domain order, so a floored entry reads in the schema's own order
        # whoever proposed it. Reached only when the floor actually binds.
        kept_values = full
        kept_weights = tuple(table.get(v, 0.0) for v in full)

    total = math.fsum(kept_weights)
    if total <= 0.0:                    # rejected upstream; nothing to floor
        return tuple(values), tuple(weights)
    k = len(kept_values)
    shares = [w / total for w in kept_weights]
    effective = min(floor, 1.0 / k)
    lowest = min(shares)
    if lowest >= effective:
        return tuple(values), tuple(weights)
    alpha = min(1.0, max(0.0, (effective - lowest) / (1.0 / k - lowest)))
    return kept_values, tuple((1.0 - alpha) * s + alpha / k for s in shares)


def statistical_weighted_prior(
    attr: Attribution,
    candidate_model: Any = None,
    *,
    min_levels: int = 2,
    prior_floor: float | None = None,
) -> WeightedRestriction:
    """The credential-free graded prior: weight levels by the screen's evidence.

    Reads exactly what :func:`~agent_evolve.policies.structure.statistical_prior`
    reads, but grades instead of gambling: each level's weight is its
    Laplace-smoothed within-screen non-domination rate, ``(nd + 0.5)/(n + 1)``,
    so a level a small screen never saw win keeps mass instead of being erased.
    A locus whose rates all tie is left free -- the honest reading of a screen
    that separated nothing. This ships as the rule any model-proposed weighted
    prior has to beat.

    *prior_floor* defaults to :data:`PRIOR_FLOOR`, read at call time. Laplace
    smoothing already keeps every SCREENED level positive, so the floor is
    inert here at any screen small enough for ``0.5/(n+1)`` to stay above it
    and binds only on the wide screens where a single unlucky level would
    otherwise be arithmetically erased. Pass ``0.0`` for the pre-floor numbers.
    """

    weighted: dict[str, tuple[tuple[Any, ...], tuple[float, ...]]] = {}
    for name in dict.fromkeys(s.locus for s in attr.levels):
        summaries = attr.for_locus(name)
        if len(summaries) < min_levels:
            continue
        rates = {
            s.value: (s.nondominated + 0.5) / (s.n + 1.0) for s in summaries
        }
        if len(set(rates.values())) < 2:
            continue
        # No `domain=`: the screen's vocabulary is what this proposer can
        # speak for, and a level it never measured is not one it excluded.
        weighted[name.split("[")[0]] = floor_weights(
            tuple(rates), tuple(rates.values()), floor=prior_floor)
    return WeightedRestriction(weighted)


@dataclass
class WeightedPriorTelemetry:
    """What the proposer's replies did. Counted, never inferred."""

    calls: int = 0
    unparseable: int = 0
    wrote_candidate: int = 0
    out_of_domain: int = 0
    empty: int = 0
    restricted_loci: int = 0
    errors: int = 0
    proposals: list = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "unparseable": self.unparseable,
            "wrote_candidate": self.wrote_candidate,
            "out_of_domain": self.out_of_domain,
            "empty": self.empty,
            "restricted_loci": self.restricted_loci,
            "errors": self.errors,
            "proposals": len(self.proposals),
        }


_PROMPT_BODY = """The search has spent {n} evaluations on a screening design.

OBJECTIVES: {goals}

SEARCH SPACE (declared domains):
{schema}

SCREEN EVIDENCE (per locus value, within-screen):
{screen}

Read the table as evidence about THIS evaluator. It may or may not match how
such problems usually behave; where they disagree, the table wins.

Propose a WEIGHTED sampling prior. For any locus the evidence (or the
parameter's meaning) separates, list the values worth sampling and a
non-negative weight for each: higher weight, more draws. A value you omit gets
weight zero and is not sampled; a locus you omit stays free. You are not
choosing a candidate and must not propose one. {closing}

Reply with ONLY this JSON shape and no other text:
{{"weights": {{"<param>": {{"values": [...], "weights": [...]}}}}, "free": ["<param>", ...]}}"""

_CAUTIOUS_CLOSE = (
    "Over-restricting a\nfrontier-spreading parameter destroys diversity: "
    "leave a locus free unless you\nhave a reason."
)

_COMMITTED_CLOSE = (
    "Commit: for every parameter this\nscreen's evidence separates, give "
    "weights proportional to that evidence.\nList a parameter as free ONLY "
    "when the evidence genuinely does not separate\nits values; hedging every "
    "parameter as free discards the screen this run\npaid for."
)

#: The shipped prompt, and every sealed row's: the caution clause asks for a
#: locus to be left free unless there is a reason to bind it.
PROMPT = _PROMPT_BODY.replace("{closing}", _CAUTIOUS_CLOSE)

#: The same prompt with that one clause swapped for its opposite instruction.
#: The tuning round's target is a measured phenotype, not a taste: the terra
#: tier obeys the caution clause most literally -- the flattest weight ratios
#: of the three tiers (1.82 against luna's 2.67 and sol's 4.0) and a median of
#: five parameters listed free where the others list none -- and it costs that
#: tier at the endpoint (4 of 6 paired seeds). Everything else about the
#: prompt, the reply shape, and the validation is identical, so the arm
#: measures the clause and nothing else.
COMMITTED_PROMPT = _PROMPT_BODY.replace("{closing}", _COMMITTED_CLOSE)


def llm_weighted_prior_proposer(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    telemetry: WeightedPriorTelemetry | None = None,
    domain_context: str = "",
    style: str = "cautious",
    prior_floor: float | None = None,
) -> Callable[[Attribution, Any], WeightedRestriction]:
    """A prior proposer that elicits a GRADED prior and repairs nothing.

    Whole-reply rejection, exactly as the hard proposer: a reply naming
    undeclared values, negative or non-finite weights, mismatched lists, or an
    all-zero field is rejected WHOLE and the run proceeds unguided -- a
    repaired prior is the harness's prior wearing the model's name. A field
    weighted evenly over its full declared domain is recorded as free, not as
    a restriction; a field weighted UNevenly over the full domain is a real
    graded prior and is kept.

    *style* names which closing instruction the prompt carries -- the shipped
    ``"cautious"`` clause or the ``"committed"`` one (see
    :data:`COMMITTED_PROMPT`). It changes the prompt's last sentence and
    nothing else: same reply shape, same parse, same validation, so the two
    arms differ by the clause alone.

    *prior_floor* defaults to :data:`PRIOR_FLOOR`, read at call time, and is
    applied at INSTALL -- after the reply has been accepted whole, never as a
    repair that would let a rejected reply through. This is the path the floor
    exists for: a reply's omission of a value is an exclusion, so the declared
    domain is folded back in at the floor and the accepted weights keep their
    order above it. One consequence, stated rather than discovered: with the
    floor on, a floored locus's ``allowed`` is its full declared domain, so
    the prior stops making an exclusion claim there and the loop's unwind test
    has nothing to unwind on it. That is the trade the floor buys -- an
    unfalsifiable exclusion is replaced by a bounded bias.
    """

    from agent_evolve.policies.semantics import objective_lines, parameter_lines

    if style not in PRIOR_STYLES:
        raise ValueError(
            f"weighted prior style must be one of {PRIOR_STYLES}, got "
            f"{style!r}")
    template = COMMITTED_PROMPT if style == "committed" else PROMPT
    tel = telemetry if telemetry is not None else WeightedPriorTelemetry()
    goals = ", ".join(objective_lines(objectives))
    preamble = f"{domain_context.strip()}\n\n" if domain_context.strip() else ""

    def propose(attr: Attribution, candidate_model: Any) -> WeightedRestriction:
        domains = _domains(candidate_model, attr)
        if not domains:
            return WeightedRestriction({})
        described = parameter_lines(candidate_model, fields=list(domains))
        prompt = preamble + template.format(
            n=attr.n_evaluated,
            goals=goals,
            schema="\n".join(f"  {line}" for line in described)
            if described else
            "\n".join(f"  {k}: one of {list(v)}" for k, v in domains.items()),
            screen=render_attribution(attr),
        )
        tel.calls += 1
        try:
            text = complete(prompt)
        except Exception:
            tel.errors += 1
            return WeightedRestriction({})
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            tel.unparseable += 1
            return WeightedRestriction({})
        try:
            raw = json.loads(match.group(0))
        except (ValueError, TypeError):
            tel.unparseable += 1
            return WeightedRestriction({})
        if not isinstance(raw, dict) or not isinstance(raw.get("weights", {}), dict):
            tel.unparseable += 1
            return WeightedRestriction({})
        entries = raw.get("weights", {}) or {}
        # Every field mapped to a bare value is a configuration, not a prior --
        # the failure mode that collapses this arm into artifact authoring.
        if entries and all(not isinstance(v, dict) for v in entries.values()):
            tel.wrote_candidate += 1
            return WeightedRestriction({})

        weighted: dict[str, tuple[tuple[Any, ...], tuple[float, ...]]] = {}
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                tel.unparseable += 1
                return WeightedRestriction({})
            values = entry.get("values")
            weights = entry.get("weights")
            if (
                not isinstance(values, list) or not isinstance(weights, list)
                or not values or len(values) != len(weights)
            ):
                tel.unparseable += 1
                return WeightedRestriction({})
            if name not in domains:
                tel.out_of_domain += 1
                return WeightedRestriction({})
            domain = tuple(domains[name])
            if any(v not in domain for v in values):
                tel.out_of_domain += 1
                return WeightedRestriction({})
            clean: list[float] = []
            for w in weights:
                if (
                    isinstance(w, bool) or not isinstance(w, (int, float))
                    or not math.isfinite(float(w)) or float(w) < 0.0
                ):
                    tel.out_of_domain += 1
                    return WeightedRestriction({})
                clean.append(float(w))
            if not any(w > 0.0 for w in clean):
                tel.out_of_domain += 1
                return WeightedRestriction({})
            support = {v for v, w in zip(values, clean) if w > 0.0}
            if len(set(clean)) == 1 and support == set(domain):
                continue                      # the full domain, evenly: free
            # The floor lands HERE and not in the parse: what is admitted is
            # what the model wrote, and what is installed is what the loop
            # will sample from. Reading the reply and installing it are two
            # different acts and only the second one is insured.
            weighted[name] = floor_weights(
                values, clean, domain=domain, floor=prior_floor)

        if not weighted:
            tel.empty += 1
            return WeightedRestriction({})
        tel.restricted_loci += len(weighted)
        tel.proposals.append(
            {k: (list(v[0]), list(v[1])) for k, v in weighted.items()}
        )
        return WeightedRestriction(weighted)

    propose.telemetry = tel                    # type: ignore[attr-defined]
    propose.mechanism = "weighted_prior"       # type: ignore[attr-defined]
    propose.authored_by = "llm"                # type: ignore[attr-defined]
    # Which clause this arm bought, readable off the proposer: an arm that
    # cannot say which prompt it ran is not an arm.
    propose.style = style                      # type: ignore[attr-defined]
    return propose
