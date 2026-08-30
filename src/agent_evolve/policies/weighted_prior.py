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
    "band_fill",
    "snap_to_rung",
    "PROMPT",
    "COMMITTED_PROMPT",
    "PRIOR_STYLES",
]

#: How the prompt's last instruction reads. See :data:`COMMITTED_PROMPT`.
PRIOR_STYLES = ("cautious", "committed")

#: The smallest share of a locus's sampling mass any DECLARED value may hold
#: once a prior is installed. Both prior entry points here read it at CALL
#: time, so it is the one place the floor is set and the one place a study
#: turns it on (any positive value; ``0.0`` is the default and keeps the
#: floorless arithmetic exactly, byte for byte -- see :func:`floor_weights`).
#:
#: Default OFF, and that is a measured decision, not a hedge. The floor
#: shipped briefly at 0.02 as a presumed defect fix (a prior that zeroes a
#: region the screen never measured ends the run's ability to sample there
#: at all -- observed on 20 of 20 analog cells, where the zeroed region held
#: the venue's best configurations). The one-factor reading then came back a
#: wash: floor-on vs floor-off on otherwise identical frozen runs read
#: 3W/3L with median delta -0.021, the floor rescuing seeds whose box was
#: wrong (+0.16) and taxing seeds whose box was right (-0.24) in equal
#: measure. Insurance that costs its premium exactly is not a defect fix,
#: so the default reverted. Studies that want a falsifiable-in-run box
#: (feasibility hunting, wrong-prior recovery) opt in with a small value;
#: 0.02 keeps the cost of a draw's prior-preferred value inside the loop's
#: own mutation noise at ~24 loci. Larger values are a second exploration
#: mechanism wearing this one's name -- use ``explore="coverage"`` instead.
PRIOR_FLOOR = 0.0


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
    #: X1, the joint-candidate channel (REFINEMENT_ROUND.md): weighted whole
    #: or partial configurations the reply named DIRECTLY. The trace
    #: forensics found live replies already smuggling 4-8 joint configs
    #: through the marginals (17/24 tapes), which a product-of-marginals
    #: draw reproduces with probability ~1e-17; this field is where they
    #: survive instead. Entries are ({field: value, ...}, weight) pairs,
    #: validated per value at admission; the loop's joint_share knob decides
    #: whether any slot ever draws them, so a populated field with the knob
    #: off changes nothing.
    candidates: tuple = ()
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



def _numeric_domain(domain: "tuple[Any, ...]") -> bool:
    return bool(domain) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in domain)


def snap_to_rung(value: Any, domain: "tuple[Any, ...]") -> Any:
    """*value* snapped to the nearest declared rung of a NUMERIC domain.

    The trace forensics that motivated this (REFINEMENT_ROUND.md, X2a): 11.5%
    of authored init cells and a share of prior values are "about X" numbers
    -- 1.61461 on a ladder whose rungs are 1.50394 and 1.6623 -- which
    whole-reply rejection then discards wholesale. The model speaks
    continuous; the grid is quantized; snapping within the domain's span is a
    codec, not a repair: no value outside [min, max] of the declared rungs is
    ever admitted, and non-numeric domains are untouched (returns *value*
    unchanged, so the caller's own validation still judges it).
    """

    if not _numeric_domain(domain):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    lo, hi = min(domain), max(domain)
    if not (lo <= float(value) <= hi):
        return value
    return min(domain, key=lambda rung: abs(float(rung) - float(value)))


def band_fill(
    values: "tuple[Any, ...] | list",
    weights: "tuple[float, ...] | list",
    *,
    domain: "tuple[Any, ...]",
) -> "tuple[tuple[Any, ...], tuple[float, ...]]":
    """Unnamed rungs INSIDE the reply's own band get interpolated weights.

    The measured defect (trace forensics, 24 tapes): replies name a band by
    listing a few of its rungs, and 68.2% of the declared rungs lying
    STRICTLY INSIDE that band got weight zero -- permanently, since the floor
    is off. The reply's own extremes define the band; the schema's enumeration
    forced the holes. This closes the holes and nothing else: on a NUMERIC
    ordered domain, every unnamed rung between two named ones takes the linear
    interpolation (in ladder-index space) of its named neighbours' weights;
    named entries keep their exact weights, including explicit zeros; rungs
    outside the named band stay excluded, so the prior's outer exclusion claim
    survives and the unwind test still has something to test. Non-numeric
    domains come back unchanged.
    """

    if not _numeric_domain(domain) or len(values) < 2:
        return tuple(values), tuple(float(w) for w in weights)
    index_of = {v: i for i, v in enumerate(domain)}
    named = sorted(
        ((index_of[v], float(w)) for v, w in zip(values, weights)
         if v in index_of),
        key=lambda pair: pair[0])
    if len(named) < 2:
        return tuple(values), tuple(float(w) for w in weights)
    table = dict(named)                      # a duplicated rung: last wins
    lo, hi = named[0][0], named[-1][0]
    anchors = sorted(table)
    out_v: list = []
    out_w: list = []
    for i in range(lo, hi + 1):
        if i in table:
            out_v.append(domain[i]); out_w.append(table[i])
            continue
        left = max(a for a in anchors if a < i)
        right = min(a for a in anchors if a > i)
        t = (i - left) / (right - left)
        out_v.append(domain[i])
        out_w.append((1.0 - t) * table[left] + t * table[right])
    return tuple(out_v), tuple(out_w)


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
    snapped_values: int = 0
    band_filled_rungs: int = 0
    joint_accepted: int = 0
    joint_rejected: int = 0
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
            "snapped_values": self.snapped_values,
            "band_filled_rungs": self.band_filled_rungs,
            "joint_accepted": self.joint_accepted,
            "joint_rejected": self.joint_rejected,
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
{{"weights": {{"<param>": {{"values": [...], "weights": [...]}}}}, "free": ["<param>", ...]{joints_shape}}}"""

_JOINTS_CLAUSE = """

Weights are per-parameter marginals and cannot express COMBINATIONS. If the
evidence suggests specific joint configurations -- parameters that work
together -- name up to {k} of them directly as candidates: each a complete or
partial configuration with a non-negative weight (higher weight, more draws).
Values must come from the declared domains; omitted parameters stay free."""

_JOINTS_SHAPE = (
    ', "candidates": [{{"config": {{"<param>": <value>, ...}}, '
    '"weight": <w>}}, ...]')

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
    joints: int = 0,
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
    if joints > 0:
        template = template.replace(
            "Reply with ONLY this JSON shape",
            _JOINTS_CLAUSE.format(k=int(joints)).strip()
            + "\n\nReply with ONLY this JSON shape")
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
            joints_shape=(_JOINTS_SHAPE if joints > 0 else ""),
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
                # X2a: continuous-to-grid codec, in-span only, numeric only.
                snapped = [snap_to_rung(v, domain) for v in values]
                tel.snapped_values += sum(
                    1 for old, new in zip(values, snapped)
                    if old != new and new in domain)
                values = snapped
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
            # X2a: a snap can land two "about X" values on one rung; the
            # rung keeps their combined intent.
            merged: dict[Any, float] = {}
            for v, w in zip(values, clean):
                merged[v] = merged.get(v, 0.0) + w
            values = list(merged)
            clean = [merged[v] for v in values]
            filled_v, filled_w = band_fill(values, clean, domain=domain)
            tel.band_filled_rungs += max(0, len(filled_v) - len(values))
            weighted[name] = floor_weights(
                filled_v, filled_w, domain=domain, floor=prior_floor)

        joint_pool: list = []
        if joints > 0:
            for item in (raw.get("candidates") or [])[: int(joints)]:
                if not isinstance(item, dict):
                    continue
                fragment = item.get("config")
                weight = item.get("weight", 1.0)
                if (not isinstance(fragment, dict) or not fragment
                        or isinstance(weight, bool)
                        or not isinstance(weight, (int, float))
                        or not math.isfinite(float(weight))
                        or float(weight) < 0.0):
                    tel.joint_rejected += 1
                    continue
                clean_frag: dict = {}
                ok = True
                for fname, fvalue in fragment.items():
                    if fname not in domains:
                        ok = False
                        break
                    fdomain = tuple(domains[fname])
                    snapped = snap_to_rung(fvalue, fdomain)
                    if snapped not in fdomain:
                        ok = False
                        break
                    clean_frag[fname] = snapped
                if ok and clean_frag:
                    joint_pool.append((clean_frag, float(weight)))
                    tel.joint_accepted += 1
                else:
                    tel.joint_rejected += 1

        if not weighted and not joint_pool:
            tel.empty += 1
            return WeightedRestriction({})
        tel.restricted_loci += len(weighted)
        tel.proposals.append(
            {k: (list(v[0]), list(v[1])) for k, v in weighted.items()}
        )
        return WeightedRestriction(weighted, candidates=tuple(joint_pool))

    propose.telemetry = tel                    # type: ignore[attr-defined]
    propose.mechanism = "weighted_prior"       # type: ignore[attr-defined]
    propose.authored_by = "llm"                # type: ignore[attr-defined]
    # Which clause this arm bought, readable off the proposer: an arm that
    # cannot say which prompt it ran is not an arm.
    propose.style = style                      # type: ignore[attr-defined]
    return propose
