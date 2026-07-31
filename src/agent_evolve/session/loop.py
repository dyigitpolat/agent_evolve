"""Pareto-guided evolutionary loop.

Pure Python orchestration: it depends only on ``core`` and the ``Harness`` port,
and knows nothing about pydantic-ai or any other LLM runtime. Switching the
harness changes nothing in this file.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence

from agent_evolve.core.formatting import (
    CandidateResult,
    prettify_configuration,
    prettify_results,
    result_to_candidate,
)
from agent_evolve.core.problem import ObjectiveSpec, Problem, validate_objective_specs
from agent_evolve.core.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    objective_value,
    select_minimax_rank,
)
from agent_evolve.core.stats import compute_performance_stats, sample_failed_for_constraint
from agent_evolve.harness.base import Harness
from agent_evolve.session.evaluate import evaluate_batch

LogFn = Callable[[str], None]
EventFn = Callable[[Dict[str, Any]], None]
RenderFn = Callable[[Dict[str, Any]], str]


@dataclass(frozen=True)
class LoopConfig:
    """Hyper-parameters for one optimisation run."""

    pop_size: int = 8
    generations: int = 5
    candidates_per_batch: int = 5
    max_regen_rounds: int = 10
    max_failed_examples: int = 5
    seed: Optional[int] = None
    llm_retries: int = 3
    #: Reflection ablation switches (all True = the paper's full reflective loop).
    #: Turning one off cleanly isolates that feedback arm; all-off is the
    #: non-reflective generate-validate-regenerate baseline (the LLM still sees the
    #: raw validator errors, but no LLM-synthesized insights/guide/patterns).
    use_failure_insights: bool = True
    use_constraint_instruction: bool = True
    use_performance_insights: bool = True
    #: Separate, more patient budget for provider rate-limit (HTTP 429) errors,
    #: which are transient and should not exhaust the normal retry budget.
    rate_limit_retries: int = 12
    #: Starting configurations, evaluated before anything is proposed, so a run
    #: can say whether what it proposed beat what the caller already had.
    seeds: tuple = ()
    #: Hard ceiling on artifacts measured. ``None`` means the generation
    #: structure alone decides.
    evaluation_budget: Optional[int] = None
    #: Artifact-identity cache shared across the run. Supplying one makes
    #: repeated materializations free and reports what that saved.
    evaluation_cache: Optional[MutableMapping] = None


def _noop_log(msg: str) -> None:
    pass


def _default_candidate_key(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


def _retry(fn: Callable, args: tuple, retries: int, log: LogFn,
           rate_limit_retries: int = 12,
           backoff_base: float = 3.0, backoff_cap: float = 30.0,
           rate_limit_cap: float = 60.0) -> Any:
    """Call *fn*, retrying on failure with exponential backoff.

    Provider rate-limit (HTTP 429) errors are transient and use a separate, more
    patient budget so they don't exhaust the normal retry budget.
    """
    normal = 0
    limited = 0
    while True:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - retried then re-raised
            if _is_rate_limit(exc):
                limited += 1
                if limited >= rate_limit_retries:
                    raise
                delay = min(15.0 + 10.0 * limited, rate_limit_cap)
                log(f"  [rate-limit {limited}/{rate_limit_retries}] waiting {delay:.0f}s")
                time.sleep(delay)
            else:
                normal += 1
                if normal >= max(retries, 1):
                    raise
                delay = min(backoff_base * (2 ** (normal - 1)), backoff_cap)
                log(f"  [retry {normal}/{retries}] {type(exc).__name__}: {exc}; waiting {delay:.0f}s")
                time.sleep(delay)


@dataclass
class _RunState:
    """Shared collaborators + mutable bookkeeping threaded through the loop."""

    problem: Any
    harness: Harness
    objectives: Sequence[ObjectiveSpec]
    config: LoopConfig
    rng: random.Random
    call: Callable
    log: LogFn
    on_event: Optional[EventFn]
    render: Optional[RenderFn]
    key_fn: Callable[[Dict[str, Any]], str]
    seen: set
    #: Count of valid candidates in the batch that last (re)wrote the constraint
    #: guide; -1 means "no guide created yet" (port of `constraint_valid_count`).
    constraint_valid_count: int = -1

    def failed_str(self, results: Sequence[CandidateResult]) -> str:
        return prettify_results(results, self.objectives, render=self.render)

    def dedup(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop already-seen configs so regen rounds don't re-evaluate duplicates."""
        out: List[Dict[str, Any]] = []
        for c in configs:
            k = self.key_fn(c)
            if k not in self.seen:
                self.seen.add(k)
                out.append(c)
        rejected = len(configs) - len(out)
        if rejected:
            self.log(f"  [dedup] rejected {rejected} duplicate candidate(s)")
        return out


def _snapshot_best(
    pareto: Sequence[Candidate], objectives: Sequence[ObjectiveSpec]
) -> Candidate:
    best = select_minimax_rank(list(pareto), objectives)
    if best is None:
        return Candidate(configuration={}, objectives={}, metadata={})
    return best


def _emit(on_event: Optional[EventFn], event: Dict[str, Any]) -> None:
    if on_event is not None:
        on_event(event)


def _performance_stats_str(
    valid_results: Sequence[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    render: Optional[RenderFn] = None,
) -> tuple[str, str, int, int]:
    """Return ``(stats_str, pareto_str, total_valid, pareto_size)``."""
    stats = compute_performance_stats(valid_results, objectives)
    if not stats:
        return "", "None", 0, 0
    lines: List[str] = [f"TOTAL VALID CANDIDATES: {len(valid_results)}"]
    lines.append(f"PARETO FRONT SIZE: {stats.get('pareto_size', 0)}")
    lines.append("")
    lines.append("BEST AND WORST PER OBJECTIVE:")
    for spec in objectives:
        best = stats.get(f"best_{spec.name}")
        worst = stats.get(f"worst_{spec.name}")
        if best:
            cfg = render(best.configuration) if render else prettify_configuration(best.configuration)
            lines.append(f"  Best {spec.name}: {objective_value(best.objectives, spec.name)}")
            lines.append(f"    Config: {cfg}")
        if worst:
            lines.append(f"  Worst {spec.name}: {objective_value(worst.objectives, spec.name)}")
    top_pareto = stats.get("top_3_pareto", [])
    pareto_str = prettify_results(top_pareto, objectives, render=render) if top_pareto else "None"
    return "\n".join(lines), pareto_str, len(valid_results), stats.get("pareto_size", 0)


def run_evolution_loop(
    *,
    problem: Problem,
    harness: Harness,
    config: LoopConfig,
    log: LogFn = _noop_log,
    on_event: Optional[EventFn] = None,
) -> SearchResult:
    """Run the full Pareto-guided evolutionary loop against *harness*."""
    objectives = list(problem.objectives)
    validate_objective_specs(objectives)
    if config.pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if config.generations <= 0:
        raise ValueError("generations must be positive")
    if config.candidates_per_batch <= 0:
        raise ValueError("candidates_per_batch must be positive")
    if config.max_regen_rounds < 0:
        raise ValueError("max_regen_rounds must be non-negative")

    rng = random.Random(config.seed)
    retries = config.llm_retries

    def call(fn: Callable, *args: Any) -> Any:
        return _retry(fn, args, retries, log, rate_limit_retries=config.rate_limit_retries)

    key_fn = getattr(problem, "candidate_key", None) or _default_candidate_key
    state = _RunState(
        problem=problem,
        harness=harness,
        objectives=objectives,
        config=config,
        rng=rng,
        call=call,
        log=log,
        on_event=on_event,
        render=getattr(problem, "render_candidate", None),
        key_fn=key_fn,
        seen=set(),
    )
    # Seed the dedup set with the example/base config so it is never re-proposed.
    example = getattr(problem, "example_config", None)
    if isinstance(example, dict):
        state.seen.add(key_fn(example))

    all_valid: List[CandidateResult] = []
    all_failed: List[CandidateResult] = []
    all_candidates_meta: List[tuple] = []
    history: List[Dict[str, Any]] = []
    constraint_instruction = ""
    performance_insights = ""
    best_per_generation: List[Candidate] = []

    log(
        f"agent_evolve: {config.generations} generations, pop={config.pop_size}, "
        f"batch={config.candidates_per_batch}, harness={getattr(harness, 'id', '?')}"
    )

    # -- Generation 0: the caller's own starting points -------------------
    # Evaluated before anything is proposed, so the run can say plainly
    # whether what it proposed beat what the caller already had.
    if config.seeds:
        seed_configs = [dict(c) for c in config.seeds]
        for c in seed_configs:
            state.seen.add(key_fn(c))
        seed_valid, seed_failed, _ = evaluate_batch(
            problem, seed_configs, objectives, cache=config.evaluation_cache
        )
        all_valid.extend(seed_valid)
        all_failed.extend(seed_failed)
        for r in seed_valid + seed_failed:
            all_candidates_meta.append((r, _candidate_metadata(r, generation=0)))
        log(f"  [seeds] {len(seed_valid)} valid, {len(seed_failed)} rejected")

    # -- Generation 1: initial sampling with regeneration ----------------
    gen1_valid, gen1_failed, constraint_instruction = _run_initial_generation(
        state=state,
        constraint_instruction=constraint_instruction,
        performance_insights=performance_insights,
    )

    all_valid.extend(gen1_valid)
    all_failed.extend(gen1_failed)
    for r in gen1_valid + gen1_failed:
        all_candidates_meta.append((r, _candidate_metadata(r, generation=1)))

    pareto = compute_pareto_front([result_to_candidate(r) for r in all_valid], objectives)
    best_per_generation.append(_snapshot_best(pareto, objectives))

    if config.use_performance_insights and gen1_valid:
        stats_str, pareto_str, _, _ = _performance_stats_str(all_valid, objectives, state.render)
        performance_insights = call(harness.performance_insights, stats_str, pareto_str, None)

    history.append(
        {
            "gen": 1,
            "valid_count": len(gen1_valid),
            "failed_count": len(gen1_failed),
            "pareto_size": len(pareto),
        }
    )
    _emit(on_event, {"kind": "generation_complete", "gen": 1, "pareto_size": len(pareto)})

    def _spent() -> int:
        """Artifacts actually measured, which is what the budget counts."""
        cache = config.evaluation_cache
        misses = getattr(cache, "misses", None)
        if misses is not None:
            return int(misses)
        return sum(
            1 for r in all_valid + all_failed if getattr(r, "evaluation_attempted", False)
        )

    # -- Generations 2..N: evolution from the Pareto front ---------------
    for gen in range(2, config.generations + 1):
        if config.evaluation_budget is not None and _spent() >= config.evaluation_budget:
            log(
                f"  [budget] {_spent()}/{config.evaluation_budget} evaluations "
                "spent; stopping before generation "
                f"{gen}"
            )
            break
        gen_valid, gen_failed, constraint_instruction = _run_evolution_generation(
            state=state,
            gen=gen,
            prev_pareto=pareto,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
        )

        all_valid.extend(gen_valid)
        all_failed.extend(gen_failed)
        for r in gen_valid + gen_failed:
            all_candidates_meta.append((r, _candidate_metadata(r, generation=gen)))

        pareto = compute_pareto_front([result_to_candidate(r) for r in all_valid], objectives)
        best_per_generation.append(_snapshot_best(pareto, objectives))

        if config.use_performance_insights and all_valid:
            stats_str, pareto_str, _, _ = _performance_stats_str(all_valid, objectives, state.render)
            performance_insights = call(
                harness.performance_insights, stats_str, pareto_str, performance_insights
            )

        history.append(
            {
                "gen": gen,
                "valid_count": len(gen_valid),
                "failed_count": len(gen_failed),
                "pareto_size": len(pareto),
            }
        )
        _emit(on_event, {"kind": "generation_complete", "gen": gen, "pareto_size": len(pareto)})

    result = _build_search_result(
        all_valid,
        all_candidates_meta,
        objectives,
        history,
        best_per_generation=best_per_generation,
        # Artifacts actually measured. A result served from the artifact
        # cache cost nothing, so counting it here would overstate the bill
        # and make a budget look breached when it was honoured exactly.
        evaluations=(
            int(getattr(config.evaluation_cache, 'misses', 0))
            if config.evaluation_cache is not None
            else sum(r.evaluation_attempted for r in (*all_valid, *all_failed))
        ),
        candidate_key=state.key_fn,
    )
    log("")
    log(
        f"Summary: evaluations={result.evaluations}"
        + (
            f" (+{getattr(config.evaluation_cache, 'hits', 0)} served from cache)"
            if getattr(config.evaluation_cache, "hits", 0)
            else ""
        )
        + f", valid={len(all_valid)}, "
        f"pareto={len(result.pareto_front)}, best={result.best.objectives}"
    )
    _emit(
        on_event,
        {
            "kind": "search_complete",
            "evaluations": result.evaluations,
            "pareto_size": len(result.pareto_front),
            "performance_insights": performance_insights,
            "constraint_instruction": constraint_instruction,
        },
    )
    return result


# ------------------------------------------------------------------
# Constraint-instruction learning (port of constraint_valid_count heuristic)
# ------------------------------------------------------------------

def _learn_constraint(
    state: _RunState,
    constraint_instruction: str,
    last_round_failed: Sequence[CandidateResult],
    all_failed: Sequence[CandidateResult],
    batch_valid_count: int,
) -> str:
    """Create the constraint guide if absent, else update it when a batch improves.

    Mirrors the original: only (re)write the guide when this batch produced more
    valid candidates than the batch that last wrote it.
    """
    if not state.config.use_constraint_instruction:
        return constraint_instruction
    if not all_failed:
        return constraint_instruction

    create = not constraint_instruction
    improved = batch_valid_count > state.constraint_valid_count
    if not (create or improved):
        return constraint_instruction

    sampled = sample_failed_for_constraint(
        last_round_failed, all_failed, state.config.max_failed_examples, state.rng
    )
    sampled_str = state.failed_str(sampled)
    previous = None if create else constraint_instruction
    new_ci = state.call(state.harness.constraint_instruction, sampled_str, previous)
    if new_ci and new_ci != constraint_instruction:
        state.constraint_valid_count = batch_valid_count
        verb = "created" if create else "updated"
        state.log(f"  [constraint] guide {verb} (batch valid={batch_valid_count})")
        return new_ci
    return constraint_instruction


# ------------------------------------------------------------------
# Generation helpers
# ------------------------------------------------------------------

def _run_initial_generation(
    *,
    state: _RunState,
    constraint_instruction: str,
    performance_insights: str,
) -> tuple:
    cfg = state.config
    gen_valid: List[CandidateResult] = []
    gen_failed: List[CandidateResult] = []
    last_round_failed: List[CandidateResult] = []

    # ``max_regen_rounds`` means retries *after* one mandatory initial call.
    for regen_round in range(cfg.max_regen_rounds + 1):
        remaining = max(cfg.pop_size - len(gen_valid), 1)
        n = min(cfg.candidates_per_batch, remaining)
        if regen_round == 0:
            configs = state.call(state.harness.generate_initial, n)
        else:
            configs = state.call(
                state.harness.regenerate,
                state.failed_str(last_round_failed),
                n,
                constraint_instruction,
                performance_insights,
            )
        configs = state.dedup(configs)

        valid_batch, failed_batch, _ = evaluate_batch(state.problem, configs, state.objectives, cache=state.config.evaluation_cache)
        _emit_candidates(state.on_event, 1, regen_round, valid_batch, failed_batch)

        if failed_batch:
            _attach_failure_insights(state, failed_batch)
            gen_failed.extend(failed_batch)
        gen_valid.extend(valid_batch)
        last_round_failed = failed_batch

        constraint_instruction = _learn_constraint(
            state, constraint_instruction, last_round_failed, gen_failed, len(valid_batch)
        )

        if len(gen_valid) >= cfg.pop_size:
            break

    # A harness may return more than requested. Every already-evaluated candidate
    # remains in the ledger/archive even if the working population target was met.
    return gen_valid, gen_failed, constraint_instruction


def _run_evolution_generation(
    *,
    state: _RunState,
    gen: int,
    prev_pareto: Sequence[Candidate],
    constraint_instruction: str,
    performance_insights: str,
) -> tuple:
    cfg = state.config
    gen_valid: List[CandidateResult] = []
    gen_failed: List[CandidateResult] = []

    pareto_results = _pareto_as_results(prev_pareto)

    if not prev_pareto:
        configs = state.call(state.harness.generate_initial, cfg.pop_size)
    else:
        pareto_str = prettify_results(pareto_results[:5], state.objectives, render=state.render)
        configs = state.call(
            state.harness.generate_offspring,
            pareto_str,
            cfg.pop_size,
            constraint_instruction,
            performance_insights,
        )
    configs = state.dedup(configs)

    valid_batch, failed_batch, _ = evaluate_batch(state.problem, configs, state.objectives, cache=state.config.evaluation_cache)
    _emit_candidates(state.on_event, gen, 0, valid_batch, failed_batch)

    if failed_batch:
        _attach_failure_insights(state, failed_batch)
        gen_failed.extend(failed_batch)
    gen_valid.extend(valid_batch)
    last_round_failed = failed_batch

    regen_round = 0
    while len(gen_valid) < cfg.pop_size and regen_round < cfg.max_regen_rounds:
        if not last_round_failed:
            break
        failed_str = state.failed_str(last_round_failed)

        if prev_pareto:
            p_str = prettify_results(pareto_results[:3], state.objectives, render=state.render)
            remaining = max(cfg.pop_size - len(gen_valid), 1)
            configs = state.call(
                state.harness.regenerate_offspring,
                failed_str,
                p_str,
                min(cfg.candidates_per_batch, remaining),
                constraint_instruction,
                performance_insights,
            )
        else:
            remaining = max(cfg.pop_size - len(gen_valid), 1)
            configs = state.call(
                state.harness.regenerate,
                failed_str,
                min(cfg.candidates_per_batch, remaining),
                constraint_instruction,
                performance_insights,
            )
        configs = state.dedup(configs)

        valid_batch, failed_batch, _ = evaluate_batch(state.problem, configs, state.objectives, cache=state.config.evaluation_cache)
        _emit_candidates(state.on_event, gen, regen_round + 1, valid_batch, failed_batch)

        if failed_batch:
            _attach_failure_insights(state, failed_batch)
            gen_failed.extend(failed_batch)
        gen_valid.extend(valid_batch)
        last_round_failed = failed_batch

        constraint_instruction = _learn_constraint(
            state, constraint_instruction, last_round_failed, gen_failed, len(valid_batch)
        )
        regen_round += 1

    return gen_valid, gen_failed, constraint_instruction


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _pareto_as_results(pareto: Sequence[Candidate]) -> List[CandidateResult]:
    return [
        CandidateResult(
            configuration=c.configuration,
            objectives=c.objectives,
            is_valid=True,
            evaluation_attempted=True,
        )
        for c in pareto
    ]


def _candidate_metadata(result: CandidateResult, *, generation: int) -> Dict[str, Any]:
    """Preserve trace diagnostics in the public candidate ledger."""
    metadata: Dict[str, Any] = {
        "generation": generation,
        "valid": result.is_valid,
        "is_pareto": False,
        "evaluation_attempted": result.evaluation_attempted,
    }
    if result.failure_phase:
        metadata["failure_phase"] = result.failure_phase
    if result.error_message:
        metadata["error_message"] = result.error_message
    if result.insight:
        metadata["insight"] = result.insight
    return metadata


def _attach_failure_insights(state: _RunState, failed: List[CandidateResult]) -> None:
    if not state.config.use_failure_insights:
        return
    # A proposer may declare that it produces no insights. An uninformed
    # baseline is the case that matters: one that synthesised guidance would
    # not be uninformed, so its empty return is correct and not a fault.
    if not getattr(state.harness, "provides_insights", True):
        return
    insights = state.call(state.harness.failure_insights, state.failed_str(failed), len(failed))
    if isinstance(insights, list) and insights:
        # Broadcast: some models collapse the per-candidate list to one item; reuse the
        # last insight for any remaining failures so every failed candidate carries feedback
        # (otherwise zip() silently dropped feedback for all but the first failure).
        for i, r in enumerate(failed):
            r.insight = str(insights[i]) if i < len(insights) else str(insights[-1])
        if len(insights) != len(failed):
            state.log(f"[agent_evolve] note: {len(insights)} insight(s) for "
                      f"{len(failed)} failures — broadcasting last to remainder")
    else:
        state.log("[agent_evolve] WARNING: failure_insights returned no usable list")


def _emit_candidates(
    on_event: Optional[EventFn],
    gen: Optional[int],
    regen_round: int,
    valid_batch: Sequence[CandidateResult],
    failed_batch: Sequence[CandidateResult],
) -> None:
    if on_event is None:
        return
    for r in valid_batch:
        on_event(
            {
                "kind": "candidate_result",
                "gen": gen,
                "regen_round": regen_round,
                "valid": True,
                "configuration": dict(r.configuration),
                "evaluation_attempted": r.evaluation_attempted,
                "objectives": dict(r.objectives),
            }
        )
    for r in failed_batch:
        on_event(
            {
                "kind": "candidate_result",
                "gen": gen,
                "regen_round": regen_round,
                "valid": False,
                "configuration": dict(r.configuration),
                "evaluation_attempted": r.evaluation_attempted,
                "failure_phase": r.failure_phase,
                "error": r.error_message,
            }
        )


def _build_search_result(
    all_valid: List[CandidateResult],
    all_candidates_meta: List[tuple],
    objectives: Sequence[ObjectiveSpec],
    history: List[Dict[str, Any]],
    *,
    best_per_generation: Optional[List[Candidate]] = None,
    evaluations: int = 0,
    candidate_key: Callable[[Dict[str, Any]], str] = _default_candidate_key,
) -> SearchResult:
    pareto_results = compute_pareto_front(
        [result_to_candidate(r) for r in all_valid], objectives
    )
    pareto_configs = {candidate_key(c.configuration) for c in pareto_results}

    all_candidates: List[Candidate] = []
    for cr, meta in all_candidates_meta:
        meta_copy = dict(meta)
        if candidate_key(cr.configuration) in pareto_configs:
            meta_copy["is_pareto"] = True
        all_candidates.append(result_to_candidate(cr, meta_copy))

    pareto_list = [
        Candidate(configuration=c.configuration, objectives=c.objectives, metadata={"is_pareto": True})
        for c in pareto_results
    ]

    best_candidate = select_minimax_rank(pareto_results, objectives)
    if best_candidate is None:
        best_candidate = Candidate(configuration={}, objectives={}, metadata={})

    return SearchResult(
        objectives=list(objectives),
        best=best_candidate,
        pareto_front=pareto_list,
        all_candidates=all_candidates,
        history=history,
        best_per_generation=list(best_per_generation or []),
        evaluations=evaluations,
    )
