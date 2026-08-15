"""What the run MEASURED, rendered so a model can reason about it.

Every authored mechanism in this package so far has been conditioned on
MEANING: the model reads a semantic card and a schema and writes an artifact
before a single evaluation exists. That is a static prior. It is worth a great
deal where a recallable domain prior is strong and it is worth exactly nothing
where one is not, and no amount of re-prompting changes which of those two a
venue is -- because the prompt never contains a measurement.

This module is the other half: it turns the run's own measured
``(configuration -> objectives)`` trace into evidence a model can be asked to
reason over. It is deliberately a pure function of the trace:

- it never sees the problem, the evaluator, the cache or the budget, so
  rendering evidence cannot spend one;
- what it emits is TEXT plus a digest of that text, so a run can record
  exactly what the model was shown and no run can imply it reasoned over
  measurements when it did not;
- it computes nothing a reader cannot recompute from the same rows.

Three things are rendered, in the order the evidence supports them:

``front``       the non-dominated members measured so far -- the run's own
                weight-free verdict on what is good, the same relation
                selection already uses.
``progress``    per objective, the best value in the first half of the trace
                against the best in the second, so "what improved and what
                did not" is a measurement and not an adjective.
``effects``     per (locus, objective) rank correlation over the measured
                rows: WHICH KNOB MOVES WHICH COST. This is the term a genetic
                algorithm can only discover by stumbling and a reader of
                twenty measured points should be able to name.

``coverage`` rides along with ``effects``: which declared values a locus has
actually been measured at, so "this region is exhausted" is checkable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import dominates
from agent_evolve.policies.genetic import Locus, loci_of, read_locus

__all__ = [
    "MeasuredRow",
    "EvidenceView",
    "front_of",
    "spearman",
    "locus_effects",
    "render_measurement_evidence",
    "evidence_digest",
    "WeightedRestriction",
    "PriorVerdict",
    "parse_weighted_restriction",
    "admit_weighted_restriction",
    "apply_weighted_restriction",
    "WEIGHTED_RESTRICTION_PROMPT",
]

#: One measured candidate: its configuration, what the evaluator returned,
#: and whether it survived selection. Survival is the run's own weight-free
#: quality verdict and costs nothing extra to carry.
MeasuredRow = Tuple[Mapping[str, Any], Mapping[str, float], bool]

#: The seam a CONTROL arm wraps. A view receives the rows this run measured
#: and returns the rows the model will be shown. The identity view is the
#: product; a view that returns another run's rows -- same count, same shape,
#: same cost, wrong run -- is the shuffled-evidence control that says whether
#: any measured gain comes from reasoning over THIS run's evidence or merely
#: from making another call. It lives here, as a declared parameter, because a
#: mechanism whose control cannot be built without editing the product is a
#: mechanism whose control will not be built.
EvidenceView = Callable[[Sequence[MeasuredRow]], Sequence[MeasuredRow]]


def front_of(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
) -> List[MeasuredRow]:
    """The non-dominated subset of *rows*, in measurement order."""

    kept: List[MeasuredRow] = []
    for row in rows:
        if any(dominates(dict(other[1]), dict(row[1]), specs)
               for other in rows if other is not row):
            continue
        kept.append(row)
    return kept


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not manufacture correlation."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, or ``None`` when it is not defined.

    Undefined means undefined and is reported as such: fewer than three
    pairs, or a constant column, cannot produce a correlation and must not
    produce a zero that reads like "measured, no effect".
    """

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    if den <= 0.0:
        return None
    return max(-1.0, min(1.0, num / den))


@dataclass(frozen=True)
class LocusEffect:
    """One locus, and what the trace says it does to each objective."""

    locus: str
    #: objective name -> rank correlation between the locus's position in its
    #: DECLARED value order and that objective. ``None`` where undefined.
    correlation: Mapping[str, Optional[float]]
    #: declared value -> how many measured rows hold it.
    coverage: Mapping[str, int]
    unmeasured: Tuple[str, ...]

    @property
    def strength(self) -> float:
        return max((abs(v) for v in self.correlation.values() if v is not None),
                   default=0.0)


def locus_effects(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
    domains: Mapping[str, Sequence[Any]],
) -> List[LocusEffect]:
    """Per-locus rank correlations and coverage, strongest effect first.

    The predictor is a locus's POSITION IN ITS DECLARED VALUE ORDER, not its
    value: the declared order is the only ordering this package is allowed to
    assume about a categorical domain, it is what the schema published, and it
    is the same order the sampler draws from. Where a domain is genuinely
    unordered the correlation is meaningless and the model is told the
    coverage instead -- which is why both travel together.
    """

    if not rows:
        return []
    template = dict(rows[0][0])
    effects: List[LocusEffect] = []
    for locus in loci_of(template):
        key = str(locus)
        domain = list(domains.get(key) or ())
        if len(domain) < 2:
            continue
        index_of = {_token(v): i for i, v in enumerate(domain)}
        positions: List[float] = []
        keep: List[int] = []
        counts: Dict[str, int] = {_token(v): 0 for v in domain}
        for i, (config, _obj, _s) in enumerate(rows):
            try:
                token = _token(read_locus(config, locus))
            except Exception:
                continue
            if token not in index_of:
                continue
            counts[token] += 1
            positions.append(float(index_of[token]))
            keep.append(i)
        correlation: Dict[str, Optional[float]] = {}
        for spec in specs:
            column = [float(rows[i][1].get(spec.name, 0.0)) for i in keep]
            correlation[spec.name] = spearman(positions, column)
        effects.append(LocusEffect(
            locus=key,
            correlation=correlation,
            coverage=dict(counts),
            unmeasured=tuple(t for t, c in counts.items() if c == 0),
        ))
    effects.sort(key=lambda e: (-e.strength, e.locus))
    return effects


def _token(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def render_measurement_evidence(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
    domains: Mapping[str, Sequence[Any]],
    *,
    front_shown: int = 8,
    effects_shown: int = 8,
    charged: Optional[int] = None,
) -> str:
    """The measured trace as prompt text. Pure, and a pure function of *rows*.

    Nothing here is an opinion. Every line is a count, a measured value or a
    rank correlation over the rows passed in, so the same rows always render
    the same bytes and :func:`evidence_digest` of that text identifies the
    evidence a call was conditioned on.
    """

    rows = list(rows)
    if not rows:
        return "  no candidate has been measured yet."
    lines: List[str] = [
        f"  measured so far: {len(rows)} configurations"
        + (f" ({charged} charged evaluations)" if charged is not None else ""),
    ]

    front = front_of(rows, specs)
    lines.append(f"  the current non-dominated front ({len(front)} of "
                 f"{len(rows)} measured):")
    for config, objectives, _s in front[:front_shown]:
        rendered = ", ".join(f"{s.name}={_fmt(objectives.get(s.name, 0.0))}"
                             for s in specs)
        lines.append(f"    {json.dumps(config, sort_keys=True, default=str)}"
                     f"  ->  {rendered}")
    if len(front) > front_shown:
        lines.append(f"    ... and {len(front) - front_shown} more front members")

    lines.append("  progress -- best value per objective, first half of the "
                 "run vs second half:")
    half = max(1, len(rows) // 2)
    for spec in specs:
        early = [float(o.get(spec.name, 0.0)) for _c, o, _s in rows[:half]]
        late = [float(o.get(spec.name, 0.0)) for _c, o, _s in rows[half:]]
        pick = min if spec.goal == "min" else max
        best_early = pick(early) if early else float("nan")
        best_late = pick(late) if late else best_early
        moved = ("IMPROVED" if late and pick([best_early, best_late]) == best_late
                 and best_late != best_early else "did not improve")
        lines.append(f"    {spec.name} ({spec.goal}imise): {_fmt(best_early)} "
                     f"-> {_fmt(best_late)}  [{moved}]")

    effects = locus_effects(rows, specs, domains)
    if effects:
        lines.append("  which parameter moves which cost -- rank correlation "
                     "between the parameter's position in its declared value "
                     "list and the measured cost (+1 = later values cost more, "
                     "-1 = later values cost less, '-' = not determinable "
                     "from these rows):")
        for effect in effects[:effects_shown]:
            body = ", ".join(
                f"{name}={'-' if value is None else f'{value:+.2f}'}"
                for name, value in effect.correlation.items())
            unmeasured = (f"; never measured at: "
                          f"{', '.join(effect.unmeasured)}"
                          if effect.unmeasured else "; every declared value "
                          "has been measured")
            lines.append(f"    {effect.locus}: {body}{unmeasured}")
        weak = [e.locus for e in effects[effects_shown:] if e.strength < 0.2]
        if weak:
            lines.append(f"    (no determinable effect on any cost: "
                         f"{', '.join(weak[:12])})")
    return "\n".join(lines)


def evidence_digest(text: str) -> str:
    """The identity of one evidence rendering. Recorded on every call."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()




# ================================================ the locus-importance channel

WEIGHTED_RESTRICTION_PROMPT = """You are advising a black-box multi-objective \
optimizer about WHERE IN ITS SEARCH SPACE to concentrate the rest of its
evaluations.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE -- every parameter and the values it may take:
{domains}

WHAT THE OPTIMIZER HAS ACTUALLY MEASURED SO FAR:

{evidence}

Read the measurements, not the parameter names. Decide which parameters
actually move the measured costs, and for those parameters how much of the
remaining budget each value deserves.

Reply with ONLY a JSON object, no prose and no code fence, of this shape:

{{"weight": {{"<parameter>": {{"<value>": <positive number>, ...}}, ...}},
 "because": {{"<parameter>": "<one short sentence citing the measurements>"}}}}

Rules, and the harness checks every one of them:
- Name ONLY parameters that appear in the search space above, and ONLY values
  that parameter declares. Anything else and the whole reply is REFUSED.
- Weights BIAS sampling; they exclude nothing. A value you do not mention
  keeps weight 1. Every declared value stays reachable no matter what you
  reply, so you cannot lose the optimum -- but you can waste budget, so
  weight only what the measurements justify.
- Weights must be positive numbers, and within one parameter the heaviest
  value may outweigh the lightest by at most {max_ratio}x (unmentioned values
  count as weight 1). More concentration than that and the whole reply is
  REFUSED: concentration is the point; a de-facto exclusion is not.
- Leave "weight" empty if the measurements do not justify biasing anything.
  An honest empty answer is accepted and costs nothing."""


@dataclass(frozen=True)
class WeightedRestriction:
    """A model-authored GRADED bias over declared domains, with its reasons.

    The graded replacement for the hard-restriction prior: per-locus VALUE
    WEIGHTS that bias where sampling mass lands and exclude NOTHING. Every
    declared value keeps positive weight (implicitly ``1.0`` when the model
    does not name it), so the one unrecoverable failure of the hard form --
    a restriction that excludes a measured optimum -- is structurally
    impossible rather than gated against: the ``excludes_front`` predicate is
    satisfied vacuously because nothing is ever excluded, and the
    vacuous-or-veto pathology it had on large fronts (W1: 81% of a trace
    non-dominated leaves a median of ZERO unexplored values in any admissible
    hard restriction) cannot occur.
    """

    #: parameter -> {value token -> weight}. Positive, finite, capped in
    #: ratio by the gate. Values a parameter declares but the mapping does
    #: not name carry an implicit weight of 1.0.
    weight: Mapping[str, Mapping[str, float]] = ()
    because: Mapping[str, str] = ()

    def as_note(self) -> Dict[str, Any]:
        return {"weight": {k: {t: float(w) for t, w in dict(v).items()}
                           for k, v in dict(self.weight).items()},
                "because": dict(self.because)}


@dataclass(frozen=True)
class PriorVerdict:
    """Admitted or refused, and WHY. The gate refuses; it never trusts."""

    admitted: bool
    reason: str = ""
    prior: Optional[WeightedRestriction] = None
    #: The largest max/min weight ratio any one parameter carries, implicit
    #: weights included. ``1.0`` is "biases nothing".
    concentration: float = 1.0

    def as_note(self) -> Dict[str, Any]:
        note: Dict[str, Any] = {"admitted": bool(self.admitted),
                                "reason": self.reason,
                                "concentration": round(self.concentration, 4)}
        if self.prior is not None:
            note.update(self.prior.as_note())
        return note


def parse_weighted_restriction(text: str) -> Optional[WeightedRestriction]:
    """The model's reply as a typed restriction, or ``None`` if it is not one.

    Whole-reply acceptance, exactly as the authored-artifact gate does it: a
    reply that is not a JSON object of the declared shape -- weights as
    numbers, per parameter, per value -- is not repaired into one, because a
    repaired prior is a prior nobody authored.
    """

    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if body.count("```") >= 2 else body
        if body.startswith("json"):
            body = body[4:]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start:end + 1])
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    weight = parsed.get("weight")
    if weight is None:
        weight = {}
    if not isinstance(weight, dict):
        return None
    typed: Dict[str, Dict[str, float]] = {}
    for name, values in weight.items():
        if not isinstance(values, dict):
            return None
        entry: Dict[str, float] = {}
        for token, w in values.items():
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                return None
            entry[str(token)] = float(w)
        typed[str(name)] = entry
    because = parsed.get("because") or {}
    if not isinstance(because, dict):
        because = {}
    return WeightedRestriction(
        weight=typed, because={str(k): str(v) for k, v in because.items()})


def admit_weighted_restriction(
    prior: Optional[WeightedRestriction],
    *,
    domains: Mapping[str, Sequence[Any]],
    max_weight_ratio: float = 8.0,
) -> PriorVerdict:
    """Refuse, or admit with the concentration measured. Never repair.

    Refusals, in the order a wrong prior does damage:

    ``unparsed``      there is no typed restriction to judge.
    ``empty``         it weights nothing (or weights everything equally), so
                      there is nothing to measure and nothing to admit -- an
                      honest no-op, recorded as one.
    ``undeclared``    a parameter or a value the schema never declared. The
                      whole reply goes, not the offending entry: a prior that
                      is wrong about the schema has not read the schema, and
                      dropping only the bad line would admit the rest on the
                      strength of an author that demonstrably guessed.
    ``invalid_weight``  a weight that is not a positive finite number. Same
                      whole-reply rule.
    ``over_concentrated``  some parameter's heaviest value outweighs its
                      lightest (implicit ``1.0`` included) by more than
                      ``max_weight_ratio``. The cap is what keeps a graded
                      bias from becoming a de-facto exclusion.

    What is NOT here, deliberately: ``excludes_front``. A weighted
    restriction cannot exclude anything -- every declared value keeps
    positive mass -- so the predicate the hard form had to gate on is
    satisfied structurally, and the W1 vacuous-or-veto pathology (a large
    front leaves nothing an admissible hard restriction may drop) has no
    lever to act on.
    """

    if prior is None:
        return PriorVerdict(False, "unparsed")
    weight = {k: dict(v) for k, v in dict(prior.weight).items() if v}
    if not weight:
        return PriorVerdict(False, "empty", prior)

    declared = {str(k): [_token(v) for v in (values or ())]
                for k, values in dict(domains).items()}
    concentration = 1.0
    for name, values in weight.items():
        if name not in declared or not declared[name]:
            return PriorVerdict(False, f"undeclared parameter {name!r}", prior)
        allowed = set(declared[name])
        for token, w in values.items():
            if token not in allowed:
                return PriorVerdict(
                    False, f"undeclared value for {name!r}", prior)
            if not math.isfinite(w) or w <= 0.0:
                return PriorVerdict(
                    False, f"invalid_weight for {name!r}", prior)
        # Implicit 1.0 for every declared value the reply does not name.
        full = [values.get(token, 1.0) for token in allowed]
        ratio = max(full) / min(full)
        concentration = max(concentration, ratio)

    if concentration <= 1.0:
        return PriorVerdict(False, "empty", prior, concentration)
    if concentration > float(max_weight_ratio):
        return PriorVerdict(
            False, f"over_concentrated ({concentration:.3g}x > "
            f"{float(max_weight_ratio):g}x)", prior, concentration)
    return PriorVerdict(True, "admitted", prior, concentration)


def apply_weighted_restriction(
    domains: Mapping[str, Sequence[Any]],
    prior: WeightedRestriction,
) -> Dict[str, List[Any]]:
    """*domains* with sampling mass biased by an ADMITTED *prior*.

    The bias is mechanical and sampler-agnostic: each declared value appears
    in its locus's list a number of times proportional to its weight
    (implicit ``1.0`` where unnamed), scaled so the lightest value appears
    exactly once. A generator that draws uniformly from the list it is given
    -- which is all the authored-sampler contract promises -- therefore lands
    its mass where the weights say, and EVERY declared value remains present,
    so nothing is excluded and validation against the declared domains never
    sees an illegal value. The admission cap on the weight ratio is also the
    cap on how long these lists can grow.
    """

    out: Dict[str, List[Any]] = {}
    for name, values in dict(domains).items():
        weights = dict(prior.weight or {}).get(name)
        if not weights:
            out[name] = list(values)
            continue
        per_value = [(v, float(weights.get(_token(v), 1.0))) for v in values]
        lightest = min((w for _v, w in per_value), default=1.0)
        if lightest <= 0.0:                       # admission forbids this;
            out[name] = list(values)              # degrade, never divide by 0
            continue
        biased: List[Any] = []
        for v, w in per_value:
            biased.extend([v] * max(1, round(w / lightest)))
        out[name] = biased
    return out
