"""A model reading a screen and proposing where to sample.

The blind version of this does not work and the reason is worth stating: asked
from the schema alone, a model reasons about a *plausible* instance of the
domain rather than the one in front of it. Measured on an accelerator venue, it
declined to exclude anything, arguing that registers trade latency against area
-- sound in general, and false there, where latency was a function of one other
locus entirely. What it lacked was a model of the evaluator, not domain
knowledge. So this proposer never runs blind: it reads a screen.

What it returns is a restriction over declared domains -- never a candidate,
never a value the schema does not declare. That is the same choice-not-authoring
line the chooser holds, applied to the generator rather than to the pair.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import DomainRestriction, Locus, locus_domain
from agent_evolve.policies.structure import Attribution, render_attribution

__all__ = ["PriorTelemetry", "llm_prior_proposer", "PROMPT"]

PROMPT = """\
You are advising an optimizer that has already spent {n} of its evaluations on
a screening design. The rest of the budget remains.

OBJECTIVES ({goals}).

THE SEARCH SPACE:
{schema}

WHAT THE SCREEN MEASURED. Per parameter value: the mean of each objective over
the screened candidates carrying that value, and how many of them were
non-dominated within the screen.
{screen}

Read this as evidence about how THIS evaluator behaves -- it may or may not
match how such problems usually behave.

YOUR TASK. Propose a sampling prior: for each parameter, either restrict
sampling to a subset of its values, or leave it free. You are not choosing a
candidate and must not propose one; you are shaping the distribution the
optimizer draws from.

Restricting to a genuinely better region multiplies the remaining budget.
Restricting wrongly can put the best region out of reach. Over-restricting a
parameter that spreads candidates ALONG the frontier destroys the diversity a
multi-objective search needs. Leave a parameter free unless the screen gives
you a reason.

Reply with JSON only:
{{"restrictions": {{"<param>": [allowed values...]}}, "free": ["<param>", ...]}}
Every parameter appears exactly once, in "restrictions" or in "free".
"""


@dataclass
class PriorTelemetry:
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


def _domains(candidate_model: Any, attr: Attribution) -> dict[str, tuple]:
    names = list(dict.fromkeys(s.locus for s in attr.levels))
    out: dict[str, tuple] = {}
    for name in names:
        base = name.split("[")[0]
        domain = locus_domain(candidate_model, Locus(base))
        if not domain:                       # a sequence element's own domain
            index = name[name.find("[") + 1:-1] if "[" in name else None
            domain = locus_domain(
                candidate_model,
                Locus(base, int(index)) if index is not None else Locus(base))
        if domain:
            out[base] = domain
    return out


def llm_prior_proposer(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    telemetry: PriorTelemetry | None = None,
) -> Callable[[Attribution, Any], DomainRestriction]:
    """A :data:`~agent_evolve.policies.structure.PriorProposer` backed by *complete*.

    A reply naming values the schema never declared is REJECTED whole, not
    repaired: a repaired prior is the harness's prior wearing the model's name,
    and the measurement it feeds would be attributing someone else's decision.
    An unusable reply yields an empty restriction, which is exactly the
    unguided sampling distribution -- the safe direction to fail in.
    """

    tel = telemetry if telemetry is not None else PriorTelemetry()
    goals = ", ".join(f"{s.name} ({s.goal})" for s in objectives)

    def propose(attr: Attribution, candidate_model: Any) -> DomainRestriction:
        domains = _domains(candidate_model, attr)
        if not domains:
            return DomainRestriction({})
        prompt = PROMPT.format(
            n=attr.n_evaluated, goals=goals,
            schema="\n".join(f"  {k}: one of {list(v)}" for k, v in domains.items()),
            screen=render_attribution(attr))
        tel.calls += 1
        try:
            reply = complete(prompt)
        except Exception:
            tel.errors += 1
            return DomainRestriction({})

        match = re.search(r"\{.*\}", reply or "", re.S)
        if not match:
            tel.unparseable += 1
            return DomainRestriction({})
        try:
            doc = json.loads(match.group(0))
        except json.JSONDecodeError:
            tel.unparseable += 1
            return DomainRestriction({})
        if not isinstance(doc, dict):
            tel.unparseable += 1
            return DomainRestriction({})

        restrictions = doc.get("restrictions")
        if not isinstance(restrictions, dict):
            tel.unparseable += 1
            return DomainRestriction({})
        # A reply that hands back whole configurations is authoring, which this
        # channel exists not to do. Detect it by shape: a mapping of parameter
        # to a single scalar rather than to a list of allowed values.
        if restrictions and all(
                not isinstance(v, list) for v in restrictions.values()):
            tel.wrote_candidate += 1
            return DomainRestriction({})

        allowed: dict[str, list] = {}
        for key, values in restrictions.items():
            domain = domains.get(key)
            if domain is None or not isinstance(values, list) or not values:
                tel.out_of_domain += 1
                return DomainRestriction({})
            kept = [v for v in values if v in domain]
            if len(kept) != len(values):
                tel.out_of_domain += 1
                return DomainRestriction({})
            if len(kept) < len(domain):
                allowed[key] = kept

        if not allowed:
            tel.empty += 1
        tel.restricted_loci += len(allowed)
        tel.proposals.append(dict(allowed))
        return DomainRestriction(allowed)

    propose.telemetry = tel                  # type: ignore[attr-defined]
    propose.mechanism = "prior"              # type: ignore[attr-defined]
    propose.authored_by = "llm"              # type: ignore[attr-defined]
    return propose
