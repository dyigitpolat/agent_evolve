"""Measurement-conditioned revision of the graded prior, mid-run.

Every prior this package installs today is authored ONCE, before the run has
measured anything worth reading: the screen's statistical rule and the graded
prior it feeds both settle at t = 0 and are then held for the whole budget.
That is the right shape when the budget is short and the wrong shape when it
is not -- a prior fitted to the first rows keeps steering draws long after the
rows that justified it stopped being the run's evidence, and staleness is
measured to decide the row at the larger budgets (GE: the best arm at the
small budgets loses decisively by B = 160, with the late half of a run
gaining a fraction of what its early half gained).

This module is the other clock. At a DECLARED cadence in charged evaluations,
one model call reads the domain card, the run's own measured-evidence
rendering (:mod:`agent_evolve.policies.measurement_evidence`) and the weights
currently installed, and replies with a revised graded prior -- optionally
with a few complete configurations worth measuring next. The reply is admitted
or refused WHOLE and then step-damped into the installed weights, so a
revision is a tilt and never a replacement.

Three constraints shape everything here, and each is structural rather than
procedural:

*Damping bounds the damage.* ``w_new = (1 - a) * w_prev + a * w_admitted``
with ``a < 1`` keeps every previously positive weight positive, so a revision
CANNOT introduce an exclusion; the worst case of a wrong revision is wasted
draws inside the declared domain. The same mixture keeps the concentration
cap: a convex combination of two vectors whose max/min ratio is at most ``r``
has ratio at most ``r`` (the mediant inequality), so admitting proposals under
``max_weight_ratio`` bounds every installed prior at that ratio forever.

*A revision is a bet, and a bet is checked.* Each event records the weights it
replaced and where the trace stood. At the next checkpoint, if nothing
measured since is rank-0 in the pooled rows, the weights revert to the
pre-event snapshot before anything new is considered. Nothing here is
unrecoverable, which is exactly why the check can be this cheap.

*The control is a declared parameter.* ``evidence_view`` receives the rows the
run measured and returns the rows the model is shown, so the shuffled-evidence
arm -- same count, same shape, same cost, another run's rows -- is buildable
without editing the product. Every event journals the digest of the rendered
evidence, so no run can imply it reasoned over its own measurements when it
did not.

This is the GRADED, mid-run form of the typed locus restriction, which is the
one measurement-conditioned channel that separated from the unguided null with
semantics removed; it is not the re-authoring channel that lost to its
shuffled-evidence control, and it must not be described as one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import dominates
from agent_evolve.policies.genetic import (
    Locus, loci_of, locus_domain, read_locus)
from agent_evolve.policies.measurement_evidence import (
    MIN_EVIDENCE_ROWS,
    MeasuredRow,
    evidence_digest,
    render_measurement_evidence,
)
from agent_evolve.policies.weighted_prior import WeightedRestriction

__all__ = ["ReguidanceTelemetry", "ReguidanceOutcome", "Reguidance", "PROMPT",
           "IMMIGRANTS_CLAUSE"]

Config = Dict[str, Any]
#: field -> (values, weights), the installed overlay's one representation.
Overlay = Dict[str, Tuple[Tuple[Any, ...], Tuple[float, ...]]]


@dataclass
class ReguidanceTelemetry:
    """What the revision channel did. Counted, never inferred."""

    calls: int = 0
    revisions_admitted: int = 0
    revisions_refused: int = 0
    revisions_reverted: int = 0
    immigrants_proposed: int = 0
    immigrants_accepted: int = 0
    immigrants_rejected: int = 0
    errors: int = 0
    #: One record per event: the cadence position it fired at, the digest of
    #: the evidence the model was shown, the verdict, and what changed.
    events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "revisions_admitted": self.revisions_admitted,
            "revisions_refused": self.revisions_refused,
            "revisions_reverted": self.revisions_reverted,
            "immigrants_proposed": self.immigrants_proposed,
            "immigrants_accepted": self.immigrants_accepted,
            "immigrants_rejected": self.immigrants_rejected,
            "errors": self.errors,
            "events": len(self.events),
        }


@dataclass(frozen=True)
class ReguidanceOutcome:
    """What the loop should do with this checkpoint.

    ``restriction`` is ``None`` for "keep whatever you hold": no call fired, or
    the reply was refused and nothing about the installed weights moved. A
    :class:`~agent_evolve.policies.weighted_prior.WeightedRestriction` is the
    prior the loop installs from here on -- including the empty one, which
    samples exactly as no restriction does.
    """

    restriction: Optional[Any] = None
    immigrants: Tuple[Config, ...] = ()
    note: Optional[Dict[str, Any]] = None


PROMPT = """{context}

You are REVISING the weighted sampling prior mid-run.

The optimizer draws every new candidate from a per-parameter weight table.
That table was set before these measurements existed; you are being shown the
measurements so it can be corrected.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE -- every parameter and the values it may take:
{domains}

THE WEIGHTS CURRENTLY INSTALLED -- a parameter that does not appear here is
sampled UNIFORMLY over its declared values:
{weights}

WHAT THE OPTIMIZER HAS MEASURED SO FAR -- {rows} configurations, {charges} \
charged evaluations:

{evidence}

Read the measurements, not the parameter names. Decide which parameters the
trace says are worth concentrating the remaining budget on, and where.

Reply with ONLY a JSON object of this shape, no prose and no code fence:

{{"weights": {{"<parameter>": {{"values": [...], "weights": [...]}}}},
 "free": ["<parameter>", ...]}}
{immigrants}
Rules, and the harness checks every one of them:
- Name ONLY parameters that appear in the search space above, and ONLY values
  that parameter declares. Anything else and the WHOLE reply is REFUSED.
- "values" and "weights" are parallel lists of the same, non-zero length.
  Weights must be finite and non-negative.
- Within one parameter the heaviest value may outweigh the lightest POSITIVE
  one by at most {max_ratio}x; more concentration than that and the whole
  reply is REFUSED. Concentration is the point; a de-facto exclusion is not.
- For a parameter you name, a value you do NOT list gets weight zero. Never
  give weight zero -- by listing it as zero or by leaving it out -- to a value
  held by any configuration on the front above. That is the one revision that
  could throw away what the run has already measured to be good.
- "free" lists parameters whose weights should move back toward uniform,
  because the measurements no longer justify biasing them.
- Your reply is not installed as written: it is MIXED with the weights above
  at {damping:g}, so every value sampled now stays sampled and a revision is a
  tilt rather than a replacement. Say what the evidence says; the mixture
  supplies the caution."""

IMMIGRANTS_CLAUSE = """
You may also include an "immigrants" key holding up to {m} COMPLETE
configurations worth measuring next -- every parameter present, every value
from that parameter's declared domain, and none of them a configuration the
run has already measured:

{{"immigrants": [{{"<parameter>": <value>, ...}}]}}
"""


class Reguidance:
    """The revision channel: one call per checkpoint, damped into the prior.

    Constructed complete -- the completion callable, the objectives, the
    schema, the cadence -- so a run states what it bought instead of
    assembling it from defaults at three call sites. It is a pure consumer of
    ``complete``: it never sees the problem, the evaluator, the cache or the
    budget, so a revision cannot spend one.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        objectives: Sequence[ObjectiveSpec],
        candidate_model: Any,
        template: Mapping[str, Any],
        domain_context: str = "",
        every: int,
        max_events: int = 4,
        immigrants: int = 0,
        damping: float = 0.5,
        max_weight_ratio: float = 8.0,
        evidence_view: Optional[Callable[
            [Sequence[MeasuredRow]], Sequence[MeasuredRow]]] = None,
        telemetry: Optional[ReguidanceTelemetry] = None,
        min_rows: int = MIN_EVIDENCE_ROWS,
        front_shown: int = 8,
        effects_shown: int = 8,
    ) -> None:
        if int(every) <= 0:
            raise ValueError(
                "reguidance fires on a declared cadence in charged "
                f"evaluations and needs a positive one, got {every!r}")
        if int(max_events) < 0 or int(immigrants) < 0 or int(min_rows) < 0:
            raise ValueError(
                "reguidance max_events, immigrants and min_rows are counts "
                f"and must be non-negative, got {max_events!r}, "
                f"{immigrants!r}, {min_rows!r}")
        if not 0.0 <= float(damping) <= 1.0:
            raise ValueError(
                "reguidance damping mixes the reply into the installed "
                f"weights and must lie in [0, 1], got {damping!r}")
        if float(max_weight_ratio) < 1.0:
            raise ValueError(
                "reguidance max_weight_ratio caps how far one parameter may "
                f"concentrate and must be at least 1, got {max_weight_ratio!r}")

        self.complete = complete
        self.objectives = tuple(objectives)
        self.candidate_model = candidate_model
        self.template = dict(template or {})
        self.domain_context = domain_context
        self.every = int(every)
        self.max_events = int(max_events)
        self.immigrants = int(immigrants)
        self.damping = float(damping)
        self.max_weight_ratio = float(max_weight_ratio)
        self.evidence_view = evidence_view
        self.min_rows = int(min_rows)
        self.front_shown = int(front_shown)
        self.effects_shown = int(effects_shown)

        # The harvest contract (core.telemetry): counters, a name, an author.
        self.telemetry = telemetry if telemetry is not None else ReguidanceTelemetry()
        self.mechanism = "reguidance"
        self.authored_by = "llm"

        #: Per-FIELD vocabulary. Weights are keyed by ``Locus.field`` because
        #: that is the key the sampler consults, so a sequence's elements
        #: share one entry -- as they do in every other prior here.
        self._field_domains: Dict[str, Tuple[Any, ...]] = _field_domains(
            self.candidate_model, self.template)
        #: Per-LOCUS domains, which is what the evidence renderer keys on.
        self._locus_domains: Dict[str, Tuple[Any, ...]] = _locus_domains(
            self.candidate_model, self.template)

        self._installed: Overlay = {}
        self._seeded = False
        self._events_fired = 0
        self._next_at = self.every
        self._rows_at_last_event = 0
        #: The outstanding bet: the weights an event replaced and the row
        #: count at the time. ``None`` whenever no installed revision is
        #: waiting to be judged.
        self._pending: Optional[Dict[str, Any]] = None

    # -- what the loop sees --------------------------------------------------

    @property
    def installed(self) -> Overlay:
        """The graded overlay this policy currently holds. A copy."""

        return {k: (tuple(v[0]), tuple(v[1])) for k, v in self._installed.items()}

    def maybe_revise(
        self,
        rows: Sequence[Tuple[Mapping[str, Any], Mapping[str, float]]],
        population: Sequence[Tuple[Mapping[str, Any], Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
        restriction: Any,
        charges: int,
        gen: int,
    ) -> ReguidanceOutcome:
        """One checkpoint. Cheap and silent unless the cadence says otherwise."""

        measured = [(dict(config), dict(objectives)) for config, objectives in rows]
        if not self._field_domains:
            # Nothing the schema declares finitely: there is no weight table
            # to revise, so no call is worth its cost.
            return ReguidanceOutcome()
        if not self._due(int(charges), len(measured)):
            return ReguidanceOutcome()
        specs = list(specs or self.objectives)

        self._seed_from(restriction)
        note: Dict[str, Any] = {
            "gen": int(gen),
            "at_charges": int(charges),
            "rows": len(measured),
        }
        changed = self._revert_if_the_bet_lost(measured, specs, note)

        self._events_fired += 1
        self._rows_at_last_event = len(measured)
        self._next_at = int(charges) + self.every
        return self._revise(measured, population, specs, note, changed)

    # -- cadence -------------------------------------------------------------

    def _due(self, charges: int, rows: int) -> bool:
        """The declared rule, and only it.

        Three conditions, each declared rather than inherited: the cadence in
        CHARGED evaluations (what the campaign paid for), enough NEW measured
        rows since the last event for the evidence to say anything the last
        rendering did not, and the event cap. A stall trigger would be a
        fourth clock and is deliberately not built here.
        """

        if self._events_fired >= self.max_events:
            return False
        if charges < self._next_at:
            return False
        if rows <= 0 or rows - self._rows_at_last_event < self.min_rows:
            return False
        return True

    # -- state ---------------------------------------------------------------

    def _seed_from(self, restriction: Any) -> None:
        """Adopt whatever prior the loop already holds, once.

        A hard restriction is the 0/1 special case of the graded form, so it
        seeds as such. Its exclusions persist through any revision that stays
        SILENT about them (damping keeps an untouched zero at zero) and regain
        mass exactly when a reply weights them -- ``_damp`` states why that
        direction is the deliberate one. ``None`` seeds the uniform table.
        """

        if self._seeded:
            return
        self._seeded = True
        weighted = getattr(restriction, "weighted", None)
        if weighted:
            self._installed = {str(name): (tuple(values), tuple(float(w) for w in weights))
                               for name, (values, weights) in dict(weighted).items()}
            return
        allowed = getattr(restriction, "allowed", None)
        if allowed:
            hard = WeightedRestriction.hard(dict(allowed))
            self._installed = {str(name): (tuple(values), tuple(float(w) for w in weights))
                               for name, (values, weights) in dict(hard.weighted).items()}

    def _revert_if_the_bet_lost(
        self,
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
        note: Dict[str, Any],
    ) -> bool:
        """Undo the last revision unless its window IMPROVED the front.

        The claim a revision makes is narrow and therefore checkable: draws
        from the tilted prior are worth more than draws from the one it
        replaced. The first reading of "worth more" -- some post-event row is
        rank-0 in the pooled rows -- was measured impotent on the first live
        venue it met: across six revision-carrying runs on a three-objective
        simulator it admitted 23 revisions and reverted 0, because on three
        objectives almost every fresh point is non-dominated, and a run whose
        revisions had locked it flat for 120 charges kept every one of them
        (W1 pilot, seed 20370102). The bet is now the loop's own unwind
        semantics: the revision stands only if some row measured after the
        event STRICTLY DOMINATES a member of the pre-event front -- the
        tilted prior must move the front, not merely land beside it.
        """

        pending = self._pending
        if pending is None:
            return False
        self._pending = None
        cut = int(pending["rows_at_event"])
        before = [dict(row[1]) for row in rows[:cut]]
        front_before = [before[index]
                        for index in _front_indices(rows[:cut], specs)]
        oriented = list(specs)
        if any(dominates(dict(objectives), member, oriented)
               for _config, objectives in rows[cut:]
               for member in front_before):
            return False
        self._installed = {k: (tuple(v[0]), tuple(v[1]))
                           for k, v in dict(pending["weights"]).items()}
        self.telemetry.revisions_reverted += 1
        note["reverted"] = {"rows_at_event": cut,
                            "fields": sorted(self._installed)}
        return True

    # -- the call ------------------------------------------------------------

    def _revise(
        self,
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        population: Sequence[Tuple[Mapping[str, Any], Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
        note: Dict[str, Any],
        changed: bool,
    ) -> ReguidanceOutcome:
        view_rows = self._evidence_rows(rows, population)
        evidence = render_measurement_evidence(
            view_rows, list(specs), self._locus_domains,
            front_shown=self.front_shown, effects_shown=self.effects_shown,
            charged=int(note["at_charges"]))
        note["rows_shown"] = len(view_rows)
        note["evidence_sha256"] = evidence_digest(evidence)

        prompt = self._prompt(evidence, note)
        self.telemetry.calls += 1
        try:
            reply = self.complete(prompt)
        except Exception as exc:            # a policy must never kill a run
            self.telemetry.errors += 1
            note["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.events.append(note)
            return self._outcome(note, changed)

        parsed, refusal = self._parse(reply, rows, specs)
        if parsed is None:
            self.telemetry.revisions_refused += 1
            note["refused"] = refusal
            self.telemetry.events.append(note)
            return self._outcome(note, changed)

        weights, free, raw_immigrants = parsed
        before = self.installed
        mixed = self._damp(weights, free)
        self._installed = mixed
        self.telemetry.revisions_admitted += 1
        self._pending = {"rows_at_event": len(rows), "weights": before}
        note["admitted"] = True
        note["damped_fields"] = sorted(mixed)
        note["proposed_fields"] = sorted(weights)
        note["freed_fields"] = sorted(free)

        immigrants = self._immigrants(raw_immigrants, rows, note)
        self.telemetry.events.append(note)
        return self._outcome(note, True, immigrants)

    def _outcome(self, note: Dict[str, Any], changed: bool,
                 immigrants: Tuple[Config, ...] = ()) -> ReguidanceOutcome:
        """``restriction=None`` means keep; anything else is what to install."""

        restriction = WeightedRestriction(self.installed) if changed else None
        return ReguidanceOutcome(restriction=restriction,
                                 immigrants=tuple(immigrants), note=note)

    def _evidence_rows(
        self,
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        population: Sequence[Tuple[Mapping[str, Any], Mapping[str, float]]],
    ) -> List[MeasuredRow]:
        """The rows the model is shown: identity, unless a view is declared."""

        surviving = {_key(dict(config)) for config, _objectives in population}
        measured: Sequence[MeasuredRow] = [
            (config, objectives, _key(config) in surviving)
            for config, objectives in rows
        ]
        if self.evidence_view is not None:
            try:
                measured = self.evidence_view(measured)
            except Exception:               # a control that throws must not
                measured = ()               # be able to kill a measurement
        return [tuple(row) for row in measured]      # type: ignore[misc]

    def _prompt(self, evidence: str, note: Mapping[str, Any]) -> str:
        clause = ("" if self.immigrants <= 0
                  else IMMIGRANTS_CLAUSE.format(m=self.immigrants))
        return PROMPT.format(
            context=self.domain_context.strip(),
            goals="\n".join(f"  {s.name}: {s.goal}imise" for s in self.objectives),
            domains="\n".join(
                f"  {name}: {json.dumps(list(values), default=str)}"
                + self._shared_note(name)
                for name, values in sorted(self._field_domains.items())),
            weights=self._render_weights(),
            rows=int(note["rows"]),
            charges=int(note["at_charges"]),
            evidence=evidence,
            immigrants=clause,
            max_ratio=f"{self.max_weight_ratio:g}",
            damping=self.damping,
        )

    def _shared_note(self, name: str) -> str:
        """Say where a sequence parameter's one weight table applies.

        The evidence names positions (``genome[3]``) because a correlation is
        per position; the weight table is per PARAMETER, because that is the
        key the sampler consults and because a table per position would be a
        different prior for every genome length. Both facts are in the prompt,
        so the difference cannot read as a contradiction.
        """

        positions = [str(locus) for locus in loci_of(self.template)
                     if locus.field == name and locus.index is not None]
        if len(positions) < 2:
            return ""
        return (f"   (one weight table, used at every position: "
                f"{positions[0]} .. {positions[-1]})")

    def _render_weights(self) -> str:
        if not self._installed:
            return "  (none installed: every parameter is sampled uniformly)"
        lines = []
        for name in sorted(self._installed):
            values, weights = self._installed[name]
            body = ", ".join(f"{json.dumps(v, default=str)}={float(w):.4g}"
                             for v, w in zip(values, weights))
            lines.append(f"  {name}: {body}")
        absent = sorted(set(self._field_domains) - set(self._installed))
        if absent:
            lines.append(f"  (uniform, no entry: {', '.join(absent)})")
        return "\n".join(lines)

    # -- parse and admission: whole-reply, never repaired --------------------

    def _parse(
        self,
        reply: Any,
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
    ) -> Tuple[Optional[Tuple[Dict[str, List[Tuple[Any, float]]],
                              List[str], List[Any]]], str]:
        """The reply, judged whole, in the taxonomy the hard gate established.

        The reasons are the ones
        :func:`~agent_evolve.policies.measurement_evidence.admit_weighted_restriction`
        refuses on, plus the two the GRADED form adds: an all-zero field (a
        restriction that samples nothing) and ``excludes_front`` -- zero mass,
        listed or omitted, on a value some rank-0 configuration holds. Nothing
        is repaired anywhere in here: a repaired prior is the harness's prior
        wearing the model's name.
        """

        text = reply if isinstance(reply, str) else ""
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            return None, "unparsed"
        try:
            raw = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None, "unparsed"
        if not isinstance(raw, dict):
            return None, "unparsed"
        entries = raw.get("weights")
        if entries is None:
            entries = {}
        free_raw = raw.get("free")
        if free_raw is None:
            free_raw = []
        if not isinstance(entries, dict) or not isinstance(free_raw, list):
            return None, "unparsed"
        # Every parameter mapped to a bare value is a CONFIGURATION, not a
        # prior -- the failure mode that collapses this channel into artifact
        # authoring, and the reason the weighted proposer checks for it too.
        if entries and all(not isinstance(v, dict) for v in entries.values()):
            return None, "wrote_candidate"

        front = {index for index in _front_indices(rows, specs)}
        front_values: Dict[str, List[Any]] = {}
        for index in front:
            for name, values in self._field_values(rows[index][0]).items():
                for value in values:
                    if value not in front_values.setdefault(name, []):
                        front_values[name].append(value)

        weights: Dict[str, List[Tuple[Any, float]]] = {}
        for name, entry in entries.items():
            name = str(name)
            if name not in self._field_domains:
                return None, f"undeclared parameter {name!r}"
            domain = self._field_domains[name]
            if not isinstance(entry, dict):
                return None, f"malformed entry for {name!r}"
            values = entry.get("values")
            listed = entry.get("weights")
            if (not isinstance(values, list) or not isinstance(listed, list)
                    or not values or len(values) != len(listed)):
                return None, f"malformed entry for {name!r}"
            clean: List[Tuple[Any, float]] = []
            for value, weight in zip(values, listed):
                if value not in domain:
                    return None, f"undeclared value for {name!r}"
                if (isinstance(weight, bool)
                        or not isinstance(weight, (int, float))
                        or not math.isfinite(float(weight))
                        or float(weight) < 0.0):
                    return None, f"invalid_weight for {name!r}"
                clean.append((value, float(weight)))
            positive = [w for _v, w in clean if w > 0.0]
            if not positive:
                return None, f"all_zero for {name!r}"
            ratio = max(positive) / min(positive)
            if ratio > self.max_weight_ratio:
                return None, (f"over_concentrated ({ratio:.3g}x > "
                              f"{self.max_weight_ratio:g}x) for {name!r}")
            # Zero is zero however it is spelled: a value the reply omits from
            # a parameter it names carries no mass either.
            held = {_token(v): w for v, w in clean}
            for value in front_values.get(name, ()):
                if held.get(_token(value), 0.0) <= 0.0:
                    return None, f"excludes_front for {name!r}"
            weights[name] = clean

        free: List[str] = []
        for name in free_raw:
            name = str(name)
            if name not in self._field_domains:
                return None, f"undeclared parameter {name!r}"
            free.append(name)

        if not weights and not free:
            return None, "empty"

        immigrants = raw.get("immigrants")
        if not isinstance(immigrants, list):
            immigrants = []
        return (weights, free, immigrants), "admitted"

    def _field_values(self, config: Mapping[str, Any]) -> Dict[str, List[Any]]:
        """Which declared values a configuration holds, per FIELD.

        A sequence field holds one value per element and they share the
        field's entry, so a front member pins every value it uses anywhere in
        that field.
        """

        out: Dict[str, List[Any]] = {}
        for locus in loci_of(dict(config)):
            if locus.field not in self._field_domains:
                continue
            try:
                value = read_locus(config, locus)
            except Exception:
                continue
            out.setdefault(locus.field, []).append(value)
        return out

    # -- damping -------------------------------------------------------------

    def _damp(self, proposal: Mapping[str, Sequence[Tuple[Any, float]]],
              free: Sequence[str]) -> Overlay:
        """Mix the admitted reply into the installed weights. Two properties.

        1. With ``damping < 1`` no exclusion can be INTRODUCED: every value
           whose installed weight is positive keeps a positive mixed weight,
           whatever the reply says about it. A revision is therefore a tilt,
           and the worst case of a wrong one is bounded by the declared
           domains rather than by a gate.
        2. The concentration cap survives mixing. For positive vectors ``b``
           and ``p`` with ``max/min <= r`` each, every mixed entry satisfies
           ``min(b_i, p_i) * (something) <=`` ... concretely, the mediant
           inequality gives ``max_i((1-a)b_i + a*p_i) / min_i((1-a)b_i +
           a*p_i) <= r``, so admitting under ``max_weight_ratio`` bounds the
           INSTALLED ratio at that value for the whole run, however many
           revisions land. (A base carrying zeros -- a hard restriction seeded
           in -- is not an ``r``-ratio vector; the bound is over the support
           the two share, which is where the cap has meaning.)

        A field the reply neither weights nor frees is left exactly as it is:
        silence about a parameter is not evidence about it.

        The mixture is directional in one place only: a value the BASE
        excludes -- a hard restriction seeded in at the first event -- regains
        mass when the reply weights it. That is the correction the unwind
        machinery can only make by dropping the whole prior, it is bounded by
        the same cap and the same front check as any other revision, and it is
        the direction that cannot lose the optimum.
        """

        freed = set(free)
        out: Overlay = {}
        names = list(dict.fromkeys(
            list(proposal) + list(free) + list(self._installed)))
        for name in names:
            entry = self._installed.get(name)
            domain = self._field_domains.get(name)
            if not domain:
                if entry is not None:
                    out[name] = entry
                continue
            if name not in proposal and name not in freed:
                if entry is not None:
                    out[name] = entry
                continue
            base = _distribution(entry, domain)
            if name in proposal:
                table = list(proposal[name])
                prop = _normalize([_lookup(table, value) for value in domain])
            else:
                prop = [1.0 / len(domain)] * len(domain)
            mixed = tuple((1.0 - self.damping) * b + self.damping * p
                          for b, p in zip(base, prop))
            if all(w == mixed[0] for w in mixed):
                # Uniform is FREE, and free is the honest reading: an entry
                # here would only make the sampler take the weighted path to
                # reach the draw it would have made anyway.
                continue
            out[name] = (tuple(domain), mixed)
        return out

    # -- immigrants ----------------------------------------------------------

    def _immigrants(
        self,
        raw: Sequence[Any],
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        note: Dict[str, Any],
    ) -> Tuple[Config, ...]:
        """Complete configurations, validated value-by-value like ``llm_init``.

        Same rule, same counters, one addition: a member the run has already
        measured is dropped. It would cost nothing (the cache holds it) and
        buy nothing, and counting it as accepted would report guidance that
        moved no draw.
        """

        if self.immigrants <= 0 or not raw:
            return ()
        measured = {_key(config) for config, _objectives in rows}
        accepted: List[Config] = []
        rejected: List[Dict[str, str]] = []
        loci = loci_of(self.template)
        for member in raw:
            self.telemetry.immigrants_proposed += 1
            reason = _immigrant_reason(member, self.template, loci,
                                       self.candidate_model)
            if reason is None and _key(dict(member)) in measured:
                reason = "already_measured"
            if reason is None and len(accepted) >= self.immigrants:
                reason = "over_cap"
            if reason is not None:
                self.telemetry.immigrants_rejected += 1
                rejected.append({"reason": reason})
                continue
            self.telemetry.immigrants_accepted += 1
            accepted.append(dict(member))
        note["immigrants"] = {"accepted": len(accepted),
                              "rejected": rejected}
        return tuple(accepted)


# ------------------------------------------------------------------ helpers

def _key(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, default=str)


def _token(value: Any) -> str:
    """One declared value's identity, rendered as measurement_evidence does."""

    return value if isinstance(value, str) else json.dumps(value, default=str)


def _front_indices(
    rows: Sequence[Tuple[Mapping[str, Any], Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
) -> List[int]:
    """Row indices dominated by nothing. Goal-aware, weight-free."""

    objectives = [dict(row[1]) for row in rows]
    return [
        index for index, this in enumerate(objectives)
        if not any(dominates(other, this, list(specs))
                   for position, other in enumerate(objectives)
                   if position != index)
    ]


def _field_domains(candidate_model: Any,
                   template: Mapping[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    """Per-field vocabularies, the key the sampler consults.

    Sequence loci share their field's entry: a weight table keyed per element
    would be a different prior for every genome length, which the ragged-genome
    case makes meaningless.
    """

    if candidate_model is None or not template:
        return {}
    out: Dict[str, Tuple[Any, ...]] = {}
    try:
        loci = loci_of(dict(template))
    except Exception:
        return {}
    for locus in loci:
        if locus.field in out:
            continue
        try:
            domain = tuple(locus_domain(candidate_model, locus))
        except Exception:
            domain = ()
        if not domain and locus.index is not None:
            try:
                domain = tuple(locus_domain(candidate_model, Locus(locus.field)))
            except Exception:
                domain = ()
        if domain:
            out[locus.field] = domain
    return out


def _locus_domains(candidate_model: Any,
                   template: Mapping[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    """Per-locus domains, which is what the evidence renderer keys on."""

    if candidate_model is None or not template:
        return {}
    out: Dict[str, Tuple[Any, ...]] = {}
    try:
        loci = loci_of(dict(template))
    except Exception:
        return {}
    for locus in loci:
        try:
            domain = tuple(locus_domain(candidate_model, locus))
        except Exception:
            domain = ()
        if domain:
            out[str(locus)] = domain
    return out


def _lookup(table: Sequence[Tuple[Any, float]], value: Any) -> float:
    for candidate, weight in table:
        if candidate == value:
            return float(weight)
    return 0.0


def _normalize(raw: Sequence[float]) -> List[float]:
    total = float(sum(raw))
    if total <= 0.0 or not raw:
        return [1.0 / max(1, len(raw))] * len(raw)
    return [float(w) / total for w in raw]


def _distribution(
    entry: Optional[Tuple[Tuple[Any, ...], Tuple[float, ...]]],
    domain: Sequence[Any],
) -> List[float]:
    """The installed weights over *domain*, normalized; uniform when absent."""

    if entry is None:
        return [1.0 / len(domain)] * len(domain)
    table = list(zip(entry[0], entry[1]))
    return _normalize([_lookup(table, value) for value in domain])


def _immigrant_reason(member: Any, template: Mapping[str, Any],
                      loci: Sequence[Locus], candidate_model: Any) -> Optional[str]:
    """``None`` when the member is admissible; the refusal reason otherwise."""

    if not isinstance(member, dict) or set(member) != set(template):
        return "shape"
    try:
        member_loci = loci_of(member)
    except Exception:
        return "shape"
    if member_loci != tuple(loci):
        return "shape"
    for locus in member_loci:
        value = read_locus(member, locus)
        domain = locus_domain(candidate_model, locus)
        if domain:
            if value not in domain:
                return "out_of_domain"
        elif value != read_locus(template, locus):
            return "out_of_domain"
    return None
