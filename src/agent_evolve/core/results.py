"""Result containers and Pareto-front utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from agent_evolve.core.problem import ObjectiveSpec, ProblemContractError

ConfigT = TypeVar("ConfigT")



@dataclass(frozen=True)
class ProviderUsageSummary:
    """What the run spent, so a caller can answer "what did this cost me".

    Counted from the calls actually made, never declared. ``calls`` is zero for
    an uninformed proposer, and that zero is measured: a run that made no model
    call still reports the block rather than omitting it, because an absent
    field cannot be told apart from an unrecorded one.
    """

    calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[str] = None
    model: Optional[str] = None
    reported_by: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.calls) is not int or self.calls < 0:
            raise ValueError("calls must be a non-negative integer")
        figures = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }
        present = {name for name, value in figures.items() if value is not None}
        if present and not self.reported_by:
            # A figure with no reporter is a number nobody measured. Zero is the
            # dangerous case: it reads as "nothing was spent" when it means "no
            # one looked". Naming the reporter is what makes the difference
            # unrepresentable rather than merely documented.
            raise ValueError(
                f"usage figures {sorted(present)} require reported_by naming "
                "what measured them"
            )
        if self.reported_by is not None and not present:
            raise ValueError(
                "reported_by names a reporter that supplied no figure"
            )

    @property
    def provider_free(self) -> bool:
        """True only when calls were counted and none occurred."""

        return self.calls == 0

    @property
    def cost_is_known(self) -> bool:
        return self.cost_usd is not None


@dataclass(frozen=True)
class Candidate(Generic[ConfigT]):
    """A single evaluated configuration."""

    configuration: ConfigT
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult(Generic[ConfigT]):
    """Aggregated output of an optimisation run.

    Attributes
    ----------
    objectives : objectives used during the run.
    best : the single recommended candidate (minimax rank over the Pareto front;
        see :func:`select_minimax_rank`).
    pareto_front : non-dominated set.
    all_candidates : every unique candidate processed across all generations,
        including deterministic validation failures.
    history : per-generation summary dicts.
    best_per_generation : minimax-best candidate on the cumulative Pareto front
        after each generation (same rule as ``best``); useful for progress.
    evaluations : exact number of ``Problem.evaluate`` invocations, including
        calls that raise candidate-level ``ValueError``. Deterministic pre-check
        failures do not increment this evaluator-call budget.
    """

    objectives: Sequence[ObjectiveSpec]
    best: Candidate[ConfigT]
    pareto_front: List[Candidate[ConfigT]] = field(default_factory=list)
    all_candidates: List[Candidate[ConfigT]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_per_generation: List[Candidate[ConfigT]] = field(default_factory=list)
    evaluations: int = 0
    provider_usage: "ProviderUsageSummary | None" = None

    def candidates_by_author(self) -> Dict[str, int]:
        """How many candidates each authoring call produced.

        Publish this beside any comparison that treats "the model proposed it"
        as an arm. A count is checkable; a label is only asserted, and an arm
        named for the component that produced it rather than the authority that
        decided it is a mistake that survives review because the numbers still
        look plausible.
        """
        counts: Dict[str, int] = {}
        for candidate in self.all_candidates:
            author = candidate.metadata.get("authored_by", "unrecorded")
            counts[author] = counts.get(author, 0) + 1
        return counts

    def proposed_candidates(self) -> List[Candidate[ConfigT]]:
        """Only what the proposer authored: the caller's own seeds excluded."""

        return [
            candidate
            for candidate in self.all_candidates
            if candidate.metadata.get("authored_by", "").startswith("proposer_")
        ]


# ------------------------------------------------------------------
# Pareto dominance
# ------------------------------------------------------------------

def objective_value(values: Dict[str, float], name: str) -> float:
    """Read one required finite objective without fabricating a default value."""
    if name not in values:
        raise ProblemContractError(f"Missing declared objective {name!r}")
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProblemContractError(
            f"Objective {name!r} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ProblemContractError(f"Objective {name!r} must be finite, got {number!r}")
    return number

def dominates(
    a: Dict[str, float],
    b: Dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Return *True* if objective vector *a* Pareto-dominates *b*."""
    all_geq = True
    any_better = False
    for spec in objectives:
        va = objective_value(a, spec.name)
        vb = objective_value(b, spec.name)
        if spec.goal == "max":
            if va < vb:
                all_geq = False
            elif va > vb:
                any_better = True
        else:
            if va > vb:
                all_geq = False
            elif va < vb:
                any_better = True
    return all_geq and any_better


def compute_pareto_front(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[Candidate[ConfigT]]:
    """Return the non-dominated subset of *candidates*.

    Exact duplicates — same configuration identity and same measured
    objectives — collapse to their first occurrence, so a configuration a
    population re-visits across generations appears once on the front rather
    than once per visit. A configuration re-evaluated to *different*
    objectives is a genuinely different measurement and both rows remain;
    dropping one silently would be the library's judgement, not the caller's.
    """
    if not candidates:
        return []
    from agent_evolve.contract import artifact_key

    unique: List[Candidate[ConfigT]] = []
    seen: set = set()
    for c in candidates:
        key = (artifact_key(c.configuration), tuple(sorted(c.objectives.items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    front: List[Candidate[ConfigT]] = []
    for i, c in enumerate(unique):
        if not any(
            dominates(other.objectives, c.objectives, objectives)
            for j, other in enumerate(unique)
            if j != i
        ):
            front.append(c)
    return front


# ------------------------------------------------------------------
# Best-candidate selection
# ------------------------------------------------------------------

def select_best_candidate(
    pareto: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
    priority_order: Optional[List[str]] = None,
) -> Optional[Candidate[ConfigT]]:
    """Lexicographic selection from the Pareto front.

    Default priority: maximise objectives first, then minimise objectives.
    """
    if not pareto:
        return None
    if priority_order is None:
        max_objs = [s for s in objectives if s.goal == "max"]
        min_objs = [s for s in objectives if s.goal == "min"]
        priority_order = [s.name for s in max_objs] + [s.name for s in min_objs]
    obj_map = {s.name: s for s in objectives}
    unknown = [name for name in priority_order if name not in obj_map]
    if unknown:
        raise ProblemContractError(
            "Unknown priority objective name(s): " + ", ".join(unknown)
        )

    def _key(c: Candidate[ConfigT]) -> Tuple[float, ...]:
        parts: List[float] = []
        for name in priority_order:
            spec = obj_map[name]
            val = objective_value(c.objectives, name)
            parts.append(-val if spec.goal == "max" else val)
        return tuple(parts)

    return min(pareto, key=_key)


# ------------------------------------------------------------------
# Minimax-rank selection
# ------------------------------------------------------------------

def _rank_candidates(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[List[int]]:
    """``ranks[i][j]`` = 1-based **dense** rank of candidate *i* on objective *j* (1 = best).

    Ties share the same rank; the next distinct value gets the next integer (no gaps
    from skipped positions).
    """
    n = len(candidates)
    ranks: List[List[int]] = [[0] * len(objectives) for _ in range(n)]
    for j, spec in enumerate(objectives):
        values = [objective_value(c.objectives, spec.name) for c in candidates]
        reverse = spec.goal == "max"
        order = sorted(range(n), key=lambda i: values[i], reverse=reverse)
        dense = 1
        for pos, idx in enumerate(order):
            if pos > 0 and values[order[pos]] != values[order[pos - 1]]:
                dense += 1
            ranks[idx][j] = dense
    return ranks


def select_minimax_rank(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Candidate[ConfigT]]:
    """Best pick for multi-objective summaries: minimax over per-objective ranks.

    For each candidate, compute **dense** rank on each objective (1 = best among
    *candidates*). Take the **maximum** rank across objectives (bottleneck / worst
    placement). Prefer the candidate(s) with the **smallest** bottleneck (minimax).

    If several tie, pick the one with the **smallest sum of ranks** (more uniform
    strength, not a spike on one metric).
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranks = _rank_candidates(candidates, objectives)
    worst = [max(r) for r in ranks]
    min_worst = min(worst)
    tied = [i for i in range(len(candidates)) if worst[i] == min_worst]
    if len(tied) == 1:
        return candidates[tied[0]]
    best_idx = min(tied, key=lambda i: sum(ranks[i]))
    return candidates[best_idx]


def sort_by_minimax_rank(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[Candidate[ConfigT]]:
    """Order *candidates* by the same rule as :func:`select_minimax_rank`.

    Primary key: smallest worst per-objective (dense) rank. Secondary: smallest
    sum of per-objective ranks. The first element matches what
    ``select_minimax_rank(candidates, objectives)`` returns when not ``None``.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0]]
    ranks = _rank_candidates(candidates, objectives)
    order = sorted(
        range(len(candidates)),
        key=lambda i: (max(ranks[i]), sum(ranks[i])),
    )
    return [candidates[i] for i in order]
