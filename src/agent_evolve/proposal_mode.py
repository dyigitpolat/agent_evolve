"""Which operator the model is actually running, chosen per campaign.

Two modes, and the difference between them is not a hyper-parameter -- it is
which evolutionary operator exists at all.

``catalogue``   the engine enumerates a finite set of parent-relative edits and
                the model selects among them. Model authority is one opaque
                identifier; every child configuration is engine-owned. Sealed by
                :mod:`agent_evolve.domain.finite_variation`, replayed by
                resolving the identifier against the sealed contract. This is the
                constrained mode: it certifies feasibility by construction and it
                makes allocation a well-posed scoring problem. It also means the
                model never generates anything, so mutation, crossover, offspring
                generation and population initialization are all absent, and only
                selection remains.

``generative``  the model emits configurations conforming to the problem's
                ``candidate_model``, rejected by ``validate`` when illegal and
                built by ``materialize``. Sealed by
                :mod:`agent_evolve.domain.generative_emission`, replayed from the
                emission itself. This is the path the five public obligations
                always admitted; what was missing was the audit trail, which is
                the other half of this module's job.

**The matched null moves with the mode, and this module is where that is
enforced.** A generative arm sampled from an unbounded schema must be compared
against a null drawn from *that* schema, not from the catalogue the treatment no
longer uses. Comparing across supports is failure mode 2 in this project's own
taxonomy, and it is the easiest mistake to make here because the old null still
runs and still produces a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from agent_evolve.application.generative_proposal_journal import (
    candidate_schema_sha256,
)
from agent_evolve.harness.generative_seal import SealedGenerativeHarness
from agent_evolve.proposers.random_proposer import RandomProposer

__all__ = [
    "CATALOGUE",
    "GENERATIVE",
    "PROPOSAL_MODES",
    "ProposalSupport",
    "build_generative_proposer",
    "build_matched_null_proposer",
    "describe_support",
    "require_matched_support",
]

CATALOGUE = "catalogue"
GENERATIVE = "generative"
PROPOSAL_MODES = (GENERATIVE, CATALOGUE)


@dataclass(frozen=True)
class ProposalSupport:
    """The space a proposer is allowed to draw from, named so it can be checked.

    ``schema_sha256`` is present for the generative mode and is the whole story
    there: the support *is* the schema. For the catalogue mode the support is the
    sealed contract, which is parent-relative and therefore changes every
    generation, so ``option_count`` describes one parent's neighbourhood rather
    than the reachable space.
    """

    mode: str
    schema_sha256: Optional[str] = None
    option_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in PROPOSAL_MODES:
            raise ValueError(f"mode must be one of {PROPOSAL_MODES}, got {self.mode!r}")
        if self.mode == GENERATIVE and not self.schema_sha256:
            raise ValueError("a generative support is identified by its schema digest")


def describe_support(problem: Any, mode: str = GENERATIVE) -> ProposalSupport:
    """Name the support a campaign in *mode* would draw from."""

    if mode == GENERATIVE:
        return ProposalSupport(
            mode=GENERATIVE,
            schema_sha256=candidate_schema_sha256(
                getattr(problem, "candidate_model", None)
            ),
        )
    return ProposalSupport(mode=CATALOGUE)


def require_matched_support(
    treatment: ProposalSupport, null: ProposalSupport
) -> None:
    """Refuse a comparison whose arms do not draw from the same space.

    This is a hard failure and not a warning. A support mismatch does not make a
    result noisy, it makes it about something other than what it claims, and the
    number it produces looks entirely ordinary.
    """

    if treatment.mode != null.mode:
        raise ValueError(
            f"support mismatch: the treatment proposes in {treatment.mode!r} mode "
            f"and the null in {null.mode!r} mode. The null has to move with the "
            "treatment or the comparison measures the change of support."
        )
    if treatment.mode == GENERATIVE and treatment.schema_sha256 != null.schema_sha256:
        raise ValueError(
            "support mismatch: treatment and null sample different candidate "
            f"schemas ({treatment.schema_sha256} vs {null.schema_sha256})."
        )


def build_generative_proposer(
    problem: Any,
    *,
    delegate: Optional[Any] = None,
    mode: str = "record",
    validator: Optional[Callable[[Dict[str, Any]], Any]] = None,
    sealed_calls: Sequence[Any] = (),
    on_seal: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SealedGenerativeHarness:
    """Wrap *delegate* so its emissions are sealed against *problem*'s schema.

    ``validator`` defaults to the problem's own ``validate``, which is the point:
    the feasibility rule that gates the run is the same one the seal records, so
    a replay cannot be told a different story about what was legal.
    """

    schema_sha256 = candidate_schema_sha256(getattr(problem, "candidate_model", None))
    if validator is None:
        validator = getattr(problem, "validate", None)
    return SealedGenerativeHarness(
        delegate,
        candidate_schema_sha256=schema_sha256,
        mode=mode,
        validator=validator,
        sealed_calls=sealed_calls,
        on_seal=on_seal,
    )


def build_matched_null_proposer(problem: Any, *, seed: Optional[int] = None):
    """The uninformed arm for a generative campaign: same schema, no model.

    Returned rather than described, so an experiment cannot accidentally use the
    catalogue-era null against a generative treatment. The schema digest it
    reports is the same one :func:`build_generative_proposer` pins, and
    :func:`require_matched_support` will say so.
    """

    candidate_schema_sha256(getattr(problem, "candidate_model", None))
    return RandomProposer(seed=seed)
