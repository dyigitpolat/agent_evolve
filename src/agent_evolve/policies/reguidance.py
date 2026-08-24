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
currently installed, and replies with a revised graded prior -- and, where the
run bought that channel, with the complete configurations it was REQUIRED to
propose beside it. The prior half is admitted or refused WHOLE and then
step-damped into the installed weights, so a revision is a tilt and never a
replacement; the proposals beside it are validated one at a time, and a short
list is counted rather than fatal.

The evidence is a BUNDLE of two renderings, and the second one is why this
channel was rebuilt. ``render_measurement_evidence`` supplies the front, the
progress line and the per-locus rank correlations; ``render_elite_table``
supplies value occupancy among the non-dominated configurations. At the 40-90
rows a revision actually holds, over a two-dozen-field space, those
correlations are noise -- the W1 pilot moved less between arms than the same
arm moved between two draws -- while the sealed prior that did separate was
authored in the occupancy format. Both halves read the same viewed rows, one
digest covers the whole bundle, and every event journals ``evidence_text``:
the bundle verbatim. A digest can falsify a reconstruction but cannot produce
one, and the oracle instrument measured that a late checkpoint's prompt is not
reconstructible from the cells beside it -- the row list a checkpoint reads
carries cache-served repeats that the charge log, which counts charged
evaluations, cannot recover.

Three constraints shape everything here, and each is structural rather than
procedural:

*Damping bounds the damage.* ``w_new = (1 - a) * w_prev + a * w_admitted``
with ``a < 1`` keeps every previously positive weight positive, so a revision
CANNOT introduce an exclusion; the worst case of a wrong revision is wasted
draws inside the declared domain. The same mixture keeps the concentration
cap: a convex combination of two vectors whose max/min ratio is at most ``r``
has ratio at most ``r`` (the mediant inequality), so admitting proposals under
``max_weight_ratio`` bounds every installed prior at that ratio forever.

The admission gate is narrow because of that, and deliberately so. A value the
reply leaves OUT of a parameter it names is not a zero -- the mixture leaves it
``(1 - a)`` of the share it held -- so ``excludes_front`` fires only on an
EXPLICIT zero weight for a value some rank-0 configuration holds. Reading
silence as exclusion made the SEMANTICS the binding constraint on this channel
rather than the model: 10 of 11 live refusals were ``excludes_front`` on
subset replies; on the losing taped pair the late revisions were refused
exactly where the oracle's hindsight alignment peaked (delta loglik 2.55 at
k = 2); and the oracle's OWN replies -- authored with the winning run's front
in hand -- were refused at the late checkpoints of BOTH studies (s101 at
k = 3; s105 at k = 2 and k = 3). A rule that refuses hindsight is measuring
itself. The rule is stated as the condition it rests on rather than assumed:
at ``a = 1`` there is no mixture, an omission really does become a zero, and
the gate reads silence the old way because nothing else is left to.

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
did not. ``gate_reads_view`` names which of those two row sets the front check
reads, and its default is the product's safety stance: the gate that protects
a LIVE run reads REALITY, so no revision can write a zero onto a value that
some configuration this run actually measured onto the front, whatever the
prompt happened to show. A CONTROL arm sets it ``True``, because a control
whose prompt reads donor rows while its gate reads this run's front accrues
``excludes_front`` refusals the arm it controls never meets, and its refusal
rate stops being comparable (W1 pilot, seed 20370103: two of four revisions
refused on the shuffled arm alone, on evidence that named no front value).
A control is the only sane user of ``True``.

The joint-proposal half carries its own history, and that history is why an
event marks itself ``v3i`` whenever the channel is bought. Required rather
than offered, the clause was answered on schedule -- and then died at the
door: on the analog venue, twenty-four FLAT fields spelled ``bias_nmos__w``,
every single member was refused as ``shape``, while the same clause over the
six-field NAS schema passed. What the model was failing there was the
SPELLING of the schema, not the search, and the oracle named these proposals
the only carrier of interaction structure in all four usable checkpoints -- so
a channel that refuses every member over its keys is refusing the one thing it
was bought for. The repair is bounded, and every part of it is also stated in
the clause: ONE flatten of a nested member's keys (``{"a": {"b": 1}}`` becomes
``{"a__b": 1}``), completion of a PARTIAL member from the run's current best
measured configuration -- which makes a member naming three fields a
deliberate recombination against the best rather than an error -- and the
template itself echoed as a literal example member. A member that invents a
key this schema does not declare is still ``shape``: the harness repairs
spelling, and never content.

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
    render_elite_table,
    render_measurement_evidence,
)
from agent_evolve.policies.weighted_prior import WeightedRestriction

__all__ = ["ReguidanceTelemetry", "ReguidanceOutcome", "Reguidance", "PROMPT",
           "IMMIGRANTS_CLAUSE", "ELITE_TABLE_TITLE", "EVIDENCE_VERSION",
           "MECHANISM_VERSION", "MECHANISM_VERSION_IMMIGRANTS", "TILT_CAP"]

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
    #: Members the reply OWED and did not write, summed over the events of a
    #: run that bought the channel: the required-k clause's compliance meter.
    #: A shortfall costs the reply nothing else -- the prior half of the same
    #: reply is judged on its own -- so this is the only place the ask's
    #: answer rate is visible.
    immigrants_shortfall: int = 0
    #: Parameters named in ``"weights"``, summed over every reply that came
    #: back, admitted or refused. Divided by the events that carry a breadth
    #: it is the mean tilt; the per-event ``tilt_breadth`` carries the median
    #: a campaign actually reads. Counted, never capped: the focused-tilt ask
    #: is an ask, and a second refusal mode would be a throttle.
    breadth_total: int = 0
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
            "immigrants_shortfall": self.immigrants_shortfall,
            "breadth_total": self.breadth_total,
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
trace says are worth concentrating the remaining budget on, and where. Name
AT MOST {tilt_cap} parameters in "weights" -- the table's strongest cases -- and
leave the rest unlisted.

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
- For a parameter you name, a value you do NOT list keeps the mass it already
  holds, reduced by the mixture below: silence damps a value, it never
  excludes one. List the values the evidence speaks to and stay silent about
  the rest. What IS refused is an EXPLICIT zero weight on a value held by any
  configuration on the front above -- writing that zero is the one revision
  that could throw away what the run has already measured to be good.
- "free" lists parameters whose weights should move back toward uniform,
  because the measurements no longer justify biasing them.
- Your reply is not installed as written: it is MIXED with the weights above
  at {damping:g}, so every value sampled now stays sampled and a revision is a
  tilt rather than a replacement. Say what the evidence says; the mixture
  supplies the caution."""

#: The joint-proposal channel, and the reason it is REQUIRED rather than
#: offered. Both oracle studies name the same standing gap in the model's own
#: words -- per-parameter weights cannot express the interaction structure the
#: front is built out of -- at EVERY checkpoint of both, and three times they
#: name these proposals as its only carrier. Offered, the clause went
#: unanswered: zero proposals across roughly thirty analog calls, by the live
#: model and by the hindsight oracle alike (every admitted checkpoint of both
#: studies reports an immigrant count of 0), while the same clause on the
#: six-field NAS venue was sometimes answered. Optionality, not capability,
#: was suppressing it -- so the clause states a count, and the harness meters
#: the answer instead of refusing over it.
#:
#: The novelty half is the SECOND thing the live measurement forced. Required,
#: the clause was answered on schedule -- twelve proposals per cell -- and
#: accepted 0 of 69: at roughly 320 measured rows, a recombination of the
#: elites the occupancy table shows is usually a configuration the run has
#: already charged, and the dedup drops it. "Never repeats" was already in the
#: prose; what was missing was the GROUND for it, because the model cannot
#: count rows it was shown a digest of. So the clause now states how many
#: configurations the run has measured and what a repeat costs. Nothing about
#: admission moved: a repeat is still dropped, and the drop is still counted.
#:
#: The KEYS are the third thing the live measurement forced, and the largest.
#: With the clause required, the analog venue -- 24 flat fields spelled
#: ``bias_nmos__w`` -- refused EVERY member as ``shape``, where the six-field
#: NAS schema passed: the model was writing partial members and nesting the
#: groups the flat names encode. So the clause now shows the schema instead of
#: describing it (the template rendered as a literal example member), says
#: what a PARTIAL member means rather than treating it as a mistake (it is
#: completed from the run's current best measured configuration, which makes
#: it a recombination against the best), and -- when the last event rejected
#: any -- names the counts by reason, so the next reply is answering a
#: measurement rather than repeating one.
IMMIGRANTS_CLAUSE = """
Your reply MUST also carry an "immigrants" key holding EXACTLY {m} COMPLETE
configurations worth measuring next -- recombinations or refinements of what
the occupancy table says the front rewards, never repeats of configurations
the run has already measured. A per-parameter table cannot say which values
belong TOGETHER; these {m} are where you say it. Every parameter present,
every value from that parameter's declared domain:

{{"immigrants": [{{"<parameter>": <value>, ...}}]}}

Every member must carry these exact keys:

{example}

The keys are FLAT: write "group__field", never {{"group": {{"field": ...}}}},
and never a key this schema does not declare -- an invented key REJECTS the
whole member. A member that names only SOME of these keys is COMPLETED from
the run's current best configuration, so a partial member is a deliberate
recombination against the best rather than an error: name the fields you mean
to move and leave the rest out.

NOVELTY IS THE POINT: this run has ALREADY MEASURED {measured}
configurations, and the evidence above is drawn from them. A proposal that
repeats one of those {measured} is REJECTED without being measured and WASTES
the slot it took. Every one of the {m} must differ from every configuration
this run has measured, in at least one parameter -- recombine what the front
rewards into a joint setting the trace does not already contain.
{feedback}"""

#: How many parameters one reply is ASKED to name in ``"weights"``. Not a
#: refusal threshold and deliberately not one: the harness counts breadth
#: (``tilt_breadth``) and never throttles it, because a second refusal mode is
#: what v3 exists to remove. The number is the oracle's own: over the five
#: usable hindsight checkpoints of the two studies it tilted 2 to 8 focused
#: parameters, where the live replies tilted or freed all 24 fields of the
#: analog venue at once -- a breadth that says nothing a uniform table does
#: not.
TILT_CAP = 4

#: The heading the elite-occupancy half of the evidence bundle carries. It is
#: a constant because the immigrants clause and the analysis tooling both name
#: the section, and a heading two places quote is a heading worth declaring.
ELITE_TABLE_TITLE = "WHAT THE FRONT IS BUILT OUT OF"

#: Which evidence bundle an event was conditioned on, journalled on every
#: event. ``"v2"`` is the measured trace PLUS the elite-occupancy table; the
#: unversioned bundle before it was the trace alone. A study that pools events
#: across the change would otherwise be pooling two different prompts.
EVIDENCE_VERSION = "v2"

#: Which MECHANISM authored an event, journalled beside the evidence version
#: so a cell self-identifies without its campaign's paperwork. ``"v3"`` is
#: silence-keeps-mass admission, required-k joint proposals and the
#: focused-tilt ask; ``"v2"`` before it refused a subset reply whole, offered
#: the proposals and asked for no focus. The two markers move INDEPENDENTLY:
#: v3 changed what the harness asks for and what it admits, not what it shows,
#: so the evidence version stays where it was and a study may pool bundles
#: across the mechanism change while refusing to pool the mechanisms.
MECHANISM_VERSION = "v3"

#: The marker an event carries when the joint-proposal channel is BOUGHT.
#: ``v3i`` is v3 plus the repair of that channel and nothing else: one flatten
#: of a member's keys, template-fill of a partial member from the run's best
#: measured configuration, the literal example member in the clause, and the
#: per-reason feedback line. A run that buys no proposals meets none of those
#: paths and stays ``v3``, so the two populations a study pools are exactly
#: the two the change separates -- the marker names the CHANNEL that moved,
#: not the release the code came from.
MECHANISM_VERSION_IMMIGRANTS = "v3i"


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
        gate_reads_view: bool = False,
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
        #: Which rows the zero-on-front admission check reads. False -- the
        #: default, and the only setting a shipped run should use -- reads the
        #: rows this run really measured. True reads the VIEWED rows instead,
        #: which is what makes a shuffled-evidence CONTROL's refusal rate
        #: comparable to the arm it controls. See the module docstring.
        self.gate_reads_view = bool(gate_reads_view)
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
        evidence = self._evidence_bundle(view_rows, specs,
                                         int(note["at_charges"]))
        note["rows_shown"] = len(view_rows)
        note["evidence"] = EVIDENCE_VERSION
        note["mechanism"] = self._mechanism()
        note["evidence_sha256"] = evidence_digest(evidence)
        # The rendering itself, not a recipe for reconstructing it. The oracle
        # instrument proved a late-checkpoint prompt UNRECONSTRUCTIBLE from the
        # cells it was journalled beside: the row list a checkpoint reads
        # includes cache-served repeats, and the charge log -- which counts
        # charged evaluations -- cannot recover them. A digest can only falsify
        # a reconstruction; the text makes the study exact.
        note["evidence_text"] = evidence

        prompt = self._prompt(evidence, note)
        self.telemetry.calls += 1
        try:
            reply = self.complete(prompt)
        except Exception as exc:            # a policy must never kill a run
            self.telemetry.errors += 1
            note["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.events.append(note)
            return self._outcome(note, changed)

        # WHICH front the admission check protects. Reality by default: a value
        # this run measured onto its own front keeps its mass however the
        # prompt was composed. A control arm hands the gate the same rows it
        # prompted with, so the two arms refuse for the same reasons.
        gate_rows = rows
        if self.gate_reads_view:
            gate_rows = [(dict(row[0]), dict(row[1])) for row in view_rows]
            note["gate_reads_view"] = True
        # Breadth is METERED, not gated: it is read off every reply that came
        # back, whatever the verdict, so a campaign's median tilt is taken
        # over the replies the model wrote rather than over the subset the
        # admission rule happened to keep.
        breadth = _weights_breadth(reply)
        note["tilt_breadth"] = breadth
        self.telemetry.breadth_total += breadth

        parsed, refusal = self._parse(reply, gate_rows, specs)
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

        immigrants = self._immigrants(raw_immigrants, rows, specs, note)
        self.telemetry.events.append(note)
        return self._outcome(note, True, immigrants)

    def _mechanism(self) -> str:
        """Which mechanism authored this event: the channel, not the release.

        The v3i repair touches ONE channel -- the joint proposals -- so only a
        run that bought them can be affected by it, and only such a run says
        so. A run with ``immigrants = 0`` renders the same prompt v3 rendered,
        admits by the same rule and marks itself ``v3``.
        """

        return (MECHANISM_VERSION_IMMIGRANTS if self.immigrants > 0
                else MECHANISM_VERSION)

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

    def _evidence_bundle(
        self,
        view_rows: Sequence[MeasuredRow],
        specs: Sequence[ObjectiveSpec],
        charged: int,
    ) -> str:
        """The v2 bundle: the measured trace, then what the front is made of.

        Both halves read the SAME rows -- the ones ``evidence_view`` returned
        -- so the control parameter transforms the whole of what the model
        sees rather than half of it, and one digest over the concatenation is
        the identity of the whole prompt's evidence. Which is why the digest is
        taken here, over the bundle, and not per section: a bundle whose halves
        were separately digested could report "same evidence" while one half
        had moved.
        """

        measured = render_measurement_evidence(
            view_rows, list(specs), self._locus_domains,
            front_shown=self.front_shown, effects_shown=self.effects_shown,
            charged=int(charged))
        elite = render_elite_table(
            view_rows, list(specs), self._locus_domains)
        return f"{measured}\n\n  {ELITE_TABLE_TITLE}:\n{elite}"

    def _prompt(self, evidence: str, note: Mapping[str, Any]) -> str:
        # The novelty ground is the run's OWN row count -- the same number the
        # prompt states above the evidence -- because that is the count the
        # dedup the proposals will meet actually holds. A view arm changes
        # which rows are RENDERED, never how many the run has measured, so the
        # two arms are asked for novelty against the same standard.
        clause = ("" if self.immigrants <= 0
                  else IMMIGRANTS_CLAUSE.format(
                      m=self.immigrants, measured=int(note["rows"]),
                      example=json.dumps(dict(self.template), default=str),
                      feedback=self._rejection_feedback()))
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
            tilt_cap=TILT_CAP,
            max_ratio=f"{self.max_weight_ratio:g}",
            damping=self.damping,
        )

    def _rejection_feedback(self) -> str:
        """What the LAST event's proposals died of, in one line, or nothing.

        The channel already counted its rejections by reason; until now the
        only reader was the campaign. The model that wrote the rejected
        members never learned they were rejected, so a reply that misread the
        schema misread it again at the next checkpoint -- which is exactly the
        signature the analog cells wrote, twelve ``shape`` rejections a cell,
        every cell. The line costs one sentence of prompt, carries no state
        this policy did not already journal (it is read back off
        ``telemetry.events``), and says nothing when the last event's
        proposals were all admitted.
        """

        for event in reversed(self.telemetry.events):
            record = event.get("immigrants")
            if not isinstance(record, dict):
                continue
            by_reason = dict(record.get("rejected_by_reason") or {})
            if not by_reason:
                return ""
            worst_first = sorted(by_reason.items(),
                                 key=lambda item: (-item[1], item[0]))
            counts = ", ".join(f"{count} rejected as {reason}"
                               for reason, count in worst_first)
            return (f"\nLast time: {counts}. Those members bought this run\n"
                    f"nothing; write these {self.immigrants} so that does not "
                    "happen again.\n")
        return ""

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
        restriction that samples nothing) and ``excludes_front`` -- an EXPLICIT
        zero weight on a value some rank-0 configuration holds, which is the
        only way a reply can take mass off the measured front. A value the
        reply simply omits is damped, not excluded, and is admitted; the module
        docstring records what reading that omission as a zero cost. Nothing is
        repaired anywhere in here: a repaired prior is the harness's prior
        wearing the model's name.
        """

        raw = _json_object(reply)
        if raw is None:
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
            # Only a zero the reply WROTE. Silence about a value is not a
            # zero -- damping leaves an unlisted value ``(1 - a)`` of the mass
            # it holds -- so a subset reply excludes nothing and is admitted.
            # At ``a == 1`` there is no mixture and an omission really does
            # become a zero, so the gate carries the whole guarantee again and
            # reads silence the way the installed weights will.
            held = {_token(v): w for v, w in clean}
            unlisted = 0.0 if self.damping >= 1.0 else None
            for value in front_values.get(name, ()):
                mass = held.get(_token(value), unlisted)
                if mass is not None and mass <= 0.0:
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

        # The required-k clause is NOT enforced here, and that is the design:
        # the two halves of a reply are judged separately, so a model that
        # under-answers the joint-proposal ask does not also lose the prior it
        # got right. ``_immigrants`` counts the shortfall.
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
        silence about a parameter is not evidence about it. A VALUE the reply
        omits from a field it does weight is the same shape one level down --
        it keeps ``(1 - a)`` of its share rather than being zeroed -- which is
        why the admission gate can afford to refuse written zeros only.

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
        specs: Sequence[ObjectiveSpec],
        note: Dict[str, Any],
    ) -> Tuple[Config, ...]:
        """The REQUIRED k, validated value-by-value like ``llm_init``.

        Same rule, same counters, two additions. A member the run has already
        measured is dropped: it would cost nothing (the cache holds it) and
        buy nothing, and counting it as accepted would report guidance that
        moved no draw. And a reply that writes FEWER than k is not refused --
        the prior half of the same reply was judged on its own and is
        installed on its own, so an under-answered ask cannot cost the run the
        channel that did answer. The shortfall is COUNTED instead, on the
        event (proposed against required) and in the run's telemetry, which is
        where a campaign reads how often the required-k ask was met at all.

        What v3i adds happens BEFORE that rule and never inside it:
        :meth:`_prepare` repairs a member's KEYS -- one flatten, then a fill
        of the fields a partial member left out -- and the repaired member
        then meets the same value-by-value gate every whole member meets. A
        filled member is not laundered: its own values are checked exactly as
        written, so a partial member holding an undeclared value is
        ``out_of_domain`` and not an accepted configuration wearing the best
        row's clothes. Both repairs are counted on the event, because a cell
        that accepted twelve members has to be able to say how many of them
        the model actually wrote whole.
        """

        if self.immigrants <= 0:
            return ()
        provided = list(raw or ())
        measured = {_key(config) for config, _objectives in rows}
        # WHERE a partial member's missing fields come from: the run's current
        # best measured configuration, so completing one is a recombination
        # against the best rather than against the seed. With no measured row
        # to read -- the empty-trace boundary -- the template is what is left,
        # and the event says which of the two it used.
        best = self._best_measured(rows, specs)
        fill_from = best if best is not None else dict(self.template)
        accepted: List[Config] = []
        rejected: List[Dict[str, str]] = []
        loci = loci_of(self.template)
        normalized = 0
        template_filled = 0
        for member in provided:
            self.telemetry.immigrants_proposed += 1
            candidate, was_normalized, was_filled = self._prepare(member, fill_from)
            normalized += int(was_normalized)
            template_filled += int(was_filled)
            reason = _immigrant_reason(candidate, self.template, loci,
                                       self.candidate_model)
            if reason is None and _key(dict(candidate)) in measured:
                reason = "already_measured"
            if reason is None and len(accepted) >= self.immigrants:
                reason = "over_cap"
            if reason is not None:
                self.telemetry.immigrants_rejected += 1
                rejected.append({"reason": reason})
                continue
            self.telemetry.immigrants_accepted += 1
            accepted.append(dict(candidate))
        shortfall = max(0, self.immigrants - len(provided))
        self.telemetry.immigrants_shortfall += shortfall
        # WHY the channel bought nothing, per event, in one line. The rejection
        # list already carried the reason on each member; the split is what a
        # campaign reads, because the three reasons name three different
        # failures and one aggregate count names none of them. A member the run
        # has already charged (``already_measured``) says the ask needs more
        # novelty ground -- the live 0-of-69 signature; ``out_of_domain`` says
        # the model misread a declared vocabulary; ``shape`` says it wrote
        # something that is not a configuration of this schema at all. Sparse
        # by construction: a reason that never fired is absent, not zero.
        by_reason: Dict[str, int] = {}
        for entry in rejected:
            reason = entry["reason"]
            by_reason[reason] = by_reason.get(reason, 0) + 1
        record: Dict[str, Any] = {"accepted": len(accepted),
                                  "rejected": rejected,
                                  "rejected_by_reason": by_reason,
                                  "proposed": len(provided),
                                  "required": self.immigrants,
                                  "shortfall": shortfall}
        # Sparse, like the reason split beside it: a repair that never fired
        # is absent rather than zero, so an event that reports a fill is an
        # event where the model wrote a partial member.
        if normalized:
            record["normalized"] = normalized
        if template_filled:
            record["template_filled"] = template_filled
            record["fill_source"] = ("best_measured" if best is not None
                                     else "template")
        note["immigrants"] = record
        return tuple(accepted)

    # -- the key repair, bounded ---------------------------------------------

    def _prepare(self, member: Any,
                 fill_from: Mapping[str, Any]) -> Tuple[Any, bool, bool]:
        """A member's keys, repaired at most this far: flatten once, then fill.

        Two repairs, both about SPELLING, and a hard floor under both. The
        flatten is attempted only when the keys do not already match the
        schema, it goes exactly one level deep (``{"a": {"b": 1}}`` becomes
        ``{"a__b": 1}``), and its result is adopted only when it actually
        lands inside the declared field names -- so a member the flatten does
        not help is judged as written. The fill completes a member whose keys
        are a PROPER SUBSET of the schema's, from *fill_from*, in the
        template's own key order. Anything else -- a key this schema does not
        declare, a member that names nothing at all -- comes back untouched
        and is refused as ``shape`` by the same gate as before: the harness
        repairs how the model spelled the schema and never guesses what it
        meant.

        Key ORDER is normalised on every path that returns a repaired member,
        because a JSON object's order is not information while
        :func:`~agent_evolve.policies.genetic.loci_of` reads it as structure;
        a member written in the alphabetical order the prompt lists the
        domains in would otherwise be ``shape`` against a template that
        happens to be ordered another way.

        Returns the member to validate, whether the flatten was applied, and
        whether fields were filled.
        """

        if not isinstance(member, dict):
            return member, False, False
        written: Dict[str, Any] = {str(name): value
                                   for name, value in member.items()}
        fields = set(self.template)
        normalized = False
        if set(written) != fields:
            flat = _flatten_once(written)
            if set(flat) != set(written) and set(flat) <= fields:
                written, normalized = flat, True
        if not written or not set(written) <= fields:
            return written, normalized, False
        if set(written) == fields:
            ordered = {name: written[name] for name in self.template}
            return ordered, normalized, False
        filled: Dict[str, Any] = {}
        for name, held in self.template.items():
            if name in written:
                filled[name] = written[name]
            else:
                filled[name] = _copy_value(fill_from.get(name, held))
        return filled, normalized, True

    def _best_measured(
        self,
        rows: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
    ) -> Optional[Config]:
        """The run's current best configuration, defined so it cannot drift.

        Rank-0 in the pooled rows, then best on the FIRST declared objective,
        ties broken by the lowest row index. Every part of that is a choice
        and each is the cheap one: the front is what the evidence the model
        just read is about, the first objective is the one the campaign
        declared first, and the index tiebreak makes the fill deterministic
        for a taped replay. ``None`` only when there is nothing measured to
        read, which is the boundary the caller names in ``fill_source``.
        """

        if not rows:
            return None
        front = _front_indices(rows, specs)
        if not front:
            return None
        spec = list(specs)[0] if specs else None
        chosen = front[0]
        if spec is not None:
            direction = 1.0 if str(spec.goal) == "max" else -1.0
            best_score: Optional[float] = None
            for index in front:
                value = dict(rows[index][1]).get(spec.name)
                if value is None or isinstance(value, bool):
                    continue
                try:
                    score = direction * float(value)
                except (TypeError, ValueError):
                    continue
                if best_score is None or score > best_score:
                    best_score, chosen = score, index
        return dict(rows[chosen][0])


# ------------------------------------------------------------------ helpers

def _key(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, default=str)


def _token(value: Any) -> str:
    """One declared value's identity, rendered as measurement_evidence does."""

    return value if isinstance(value, str) else json.dumps(value, default=str)


def _json_object(reply: Any) -> Optional[Dict[str, Any]]:
    """The one JSON object a reply carries, or ``None``.

    The single reader of a raw reply, so the admission gate and the breadth
    meter cannot disagree about what the model actually wrote.
    """

    text = reply if isinstance(reply, str) else ""
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _weights_breadth(reply: Any) -> int:
    """How many parameters a reply names in ``"weights"``. 0 when unreadable.

    A measurement, not a check: nothing in this module refuses over it. It
    exists because the live pilot's replies tilted or freed every field of a
    24-field venue at once, which a per-event count makes visible and an
    admitted/refused verdict does not.
    """

    raw = _json_object(reply)
    entries = raw.get("weights") if raw is not None else None
    return len(entries) if isinstance(entries, dict) else 0


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


def _flatten_once(member: Mapping[str, Any]) -> Dict[str, Any]:
    """One level of nesting, joined with ``__``. The only spelling repaired.

    The analog schema names a group and a field in one flat key
    (``bias_nmos__w``), and a model handed twenty-four of those writes the
    groups back as objects often enough that every member of every cell died
    of it. One level is the whole repair: it inverts exactly the spelling the
    field names encode, and a second level would be the harness inventing a
    structure the schema never declared.
    """

    out: Dict[str, Any] = {}
    for name, value in member.items():
        if isinstance(value, dict):
            for inner, held in value.items():
                out[f"{name}__{inner}"] = held
        else:
            out[str(name)] = value
    return out


def _copy_value(value: Any) -> Any:
    """A filled field's value, never shared with the row it was read from."""

    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


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
