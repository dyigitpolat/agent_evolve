"""Performance statistics and failure sampling for the insight prompts."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from agent_evolve.core.formatting import (
    CandidateResult,
    candidate_to_result,
    result_to_candidate,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import compute_pareto_front, objective_value, sort_by_minimax_rank


def compute_performance_stats(
    valid_results: Sequence[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Dict[str, Any]]:
    """Compute best/worst per objective and the top Pareto candidates."""
    if not valid_results:
        return None

    stats: Dict[str, Any] = {}

    for spec in objectives:
        key = spec.name
        if spec.goal == "max":
            best = max(valid_results, key=lambda r: objective_value(r.objectives, key))
            worst = min(valid_results, key=lambda r: objective_value(r.objectives, key))
        else:
            best = min(valid_results, key=lambda r: objective_value(r.objectives, key))
            worst = max(valid_results, key=lambda r: objective_value(r.objectives, key))
        stats[f"best_{key}"] = best
        stats[f"worst_{key}"] = worst

    candidates = [result_to_candidate(r) for r in valid_results]
    pareto_candidates = compute_pareto_front(candidates, objectives)
    sorted_pareto = sort_by_minimax_rank(pareto_candidates, objectives)
    pareto_results = [candidate_to_result(c) for c in sorted_pareto]
    stats["top_3_pareto"] = pareto_results[:3]
    stats["pareto_front"] = pareto_results
    stats["pareto_size"] = len(pareto_results)

    return stats


def sample_failed_for_constraint(
    latest_failed: Sequence[CandidateResult],
    all_previous_failed: Sequence[CandidateResult],
    max_examples: int,
    rng: Optional[random.Random] = None,
) -> List[CandidateResult]:
    """Sample failures for constraint-instruction generation.

    Always includes the latest failures; fills remaining slots with random
    previous failures using the injected ``rng`` for reproducibility.
    """
    sampled = list(latest_failed)
    if len(sampled) >= max_examples:
        return sampled[:max_examples]

    remaining = max_examples - len(sampled)
    latest_ids = {id(r) for r in latest_failed}
    previous = [r for r in all_previous_failed if id(r) not in latest_ids]
    if previous and remaining > 0:
        picker = rng or random
        sampled.extend(picker.sample(previous, min(remaining, len(previous))))
    return sampled
