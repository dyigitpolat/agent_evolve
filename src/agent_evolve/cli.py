"""``agent_evolve`` command line: ``run``, ``check``, ``version``.

``check`` is the one worth reading about. It runs the model against an
uninformed sampler on *your* problem, at the same budget, with the same
evaluator, and reports whether the model actually beat it.

That command exists because this project repeatedly measured how easy it is to
believe an optimizer is working when nothing is: on one benchmark a single
median random draw already accounted for roughly 80 percent of the
hypervolume that a full run produced, and the entire model-guided phase past
the initial design was worth about one percent. No amount of reading a paper
tells anyone whether their own problem is like that. Half a minute of
``agent_evolve check`` does.

A problem is named as ``module:attribute``, e.g.::

    agent_evolve check examples.knapsack.problem_def:problem --budget 40
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from typing import Any, List, Optional, Sequence

from agent_evolve.contract import as_problem
from agent_evolve.core.results import compute_pareto_front

__all__ = ["main"]


def _load_problem(spec: str) -> Any:
    """Import ``module:attribute`` and return the problem object."""
    if ":" not in spec:
        raise SystemExit(
            f"could not read {spec!r}. Name a problem as module:attribute, "
            "e.g. examples.knapsack.problem_def:problem"
        )
    module_name, _, attribute = spec.partition(":")
    sys.path.insert(0, "")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise SystemExit(f"could not import {module_name!r}: {error}") from error
    try:
        candidate = getattr(module, attribute)
    except AttributeError as error:
        raise SystemExit(f"{module_name!r} has no attribute {attribute!r}") from error
    problem = candidate() if isinstance(candidate, type) else candidate
    try:
        return as_problem(problem)
    except TypeError as error:
        raise SystemExit(str(error)) from error


def _front_values(result: Any, objective: str) -> List[float]:
    return [c.objectives[objective] for c in result.pareto_front if objective in c.objectives]


def _summarise(label: str, result: Any, objectives: Sequence[Any]) -> str:
    lines = [f"  {label}:"]
    lines.append(f"    evaluations   {result.evaluations}")
    lines.append(f"    pareto front  {len(result.pareto_front)}")
    for spec in objectives:
        values = _front_values(result, spec.name)
        if not values:
            continue
        best = max(values) if spec.goal == "max" else min(values)
        lines.append(f"    best {spec.name} ({spec.goal})  {best:g}")
    return "\n".join(lines)


def _verdict(
    model_result: Any,
    baseline_results: Sequence[Any],
    objectives: Sequence[Any],
) -> List[str]:
    """State plainly whether the model beat chance, per objective."""
    out: List[str] = []
    beaten = 0
    counted = 0
    for spec in objectives:
        model_values = _front_values(model_result, spec.name)
        if not model_values:
            continue
        pick = max if spec.goal == "max" else min
        model_best = pick(model_values)
        draws = [
            pick(_front_values(r, spec.name))
            for r in baseline_results
            if _front_values(r, spec.name)
        ]
        if not draws:
            continue
        counted += 1
        better = sum(
            1 for d in draws if (d > model_best if spec.goal == "max" else d < model_best)
        )
        # Fraction of uninformed runs that matched or beat the model. This is
        # the chance baseline every claim in this project travels with.
        p = (better + 1) / (len(draws) + 1)
        median = statistics.median(draws)
        won = (model_best > median) if spec.goal == "max" else (model_best < median)
        beaten += int(won)
        out.append(
            f"    {spec.name}: model {model_best:g} vs random median {median:g} "
            f"over {len(draws)} runs  (p = {p:.2f})"
        )
    if counted:
        out.append("")
        if beaten == counted:
            out.append("  The model beat the uninformed baseline on every objective.")
        elif beaten == 0:
            out.append(
                "  The model did not beat the uninformed baseline on any objective.\n"
                "  On this problem, at this budget, it is not earning its cost."
            )
        else:
            out.append(
                f"  The model beat the uninformed baseline on {beaten} of "
                f"{counted} objectives. Mixed, and worth a larger budget "
                "before concluding either way."
            )
        out.append(
            "  A p near 1.00 means uninformed sampling routinely does as well.\n"
            "  Few runs is weak evidence; raise --repeats to sharpen it."
        )
    return out


def _model_line(model: Optional[str]) -> str:
    """Name the model and its price before anything is billed.

    A default nobody saw is a default nobody consented to, so the resolved
    model is printed whether or not the caller chose it, and it is marked as a
    default when they did not.
    """
    from agent_evolve.settings import AgentEvolveSettings, model_price

    resolved = model or AgentEvolveSettings.from_env().model
    origin = "" if model else "  (default)"
    price = model_price(resolved)
    cost = (
        f"  ${price[0]:.2f}/M in, ${price[1]:.2f}/M out"
        if price
        else "  price unknown"
    )
    return f"model {resolved}{origin}{cost}"


def _cmd_check(args: argparse.Namespace) -> int:
    from agent_evolve.api import optimize

    problem = _load_problem(args.problem)
    objectives = list(problem.objectives)
    quiet = (lambda _m: None) if not args.verbose else (lambda m: print(m, flush=True))

    print(f"agent_evolve check: {args.problem}")
    print(f"budget {args.budget} evaluations per run, {args.repeats} baseline runs")
    if args.baseline_only:
        print("baseline only: no model, no credentials, no cost\n")
    else:
        print(f"{_model_line(args.model)}\n")

    baselines = []
    for i in range(args.repeats):
        baselines.append(
            optimize(problem, budget=args.budget, proposer="random", seed=i, on_progress=quiet)
        )
    print(_summarise(f"random baseline (run 1 of {args.repeats})", baselines[0], objectives))

    if args.baseline_only:
        print(
            "\n  Baseline only. Re-run without --baseline-only, with a provider "
            "credential set, to compare a model against it."
        )
        return 0

    print()
    try:
        model_result = optimize(
            problem,
            budget=args.budget,
            model=args.model,
            proposer="llm",
            seed=0,
            on_progress=quiet,
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised, so the baseline still stands
        print(f"  model run failed: {type(error).__name__}: {error}")
        print("\n  The baseline above still stands, and cost nothing.")
        return 1
    print(_summarise("model", model_result, objectives))
    print("\n  verdict")
    for line in _verdict(model_result, baselines, objectives):
        print(line)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from agent_evolve.api import optimize

    problem = _load_problem(args.problem)
    if args.proposer != "random":
        print(_model_line(args.model))
    result = optimize(
        problem,
        budget=args.budget,
        model=args.model,
        proposer=args.proposer,
        seed=args.seed,
        on_progress=(lambda m: print(m, flush=True)) if args.verbose else print,
    )
    print(f"\nbest        {result.best.configuration}")
    print(f"objectives  {result.best.objectives}")
    print(f"pareto      {len(result.pareto_front)}")
    print(f"evaluations {result.evaluations}")
    for i, c in enumerate(result.pareto_front, 1):
        print(f"  {i}. {c.configuration} -> {c.objectives}")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    try:
        from importlib.metadata import version

        print(version("agent_evolve"))
    except Exception:
        print("unknown (not installed as a distribution)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_evolve",
        description="Multi-objective optimization driven by a language model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="does the model beat uninformed sampling on your problem?",
        description=(
            "Runs an uninformed sampler and a model against the same problem, "
            "the same budget and the same evaluator, then says which won. Run "
            "this before deciding to spend anything."
        ),
    )
    check.add_argument("problem", help="module:attribute naming your problem")
    check.add_argument("--budget", type=int, default=40, help="evaluations per run")
    check.add_argument("--repeats", type=int, default=5, help="baseline runs (default 5)")
    check.add_argument("--model", default=None, help="model id for the model arm")
    check.add_argument(
        "--baseline-only",
        action="store_true",
        help="run only the free baseline; no credentials needed",
    )
    check.add_argument("--verbose", action="store_true")
    check.set_defaults(func=_cmd_check)

    run = sub.add_parser("run", help="optimize a problem")
    run.add_argument("problem", help="module:attribute naming your problem")
    run.add_argument("--budget", type=int, default=40)
    run.add_argument("--model", default=None)
    run.add_argument(
        "--proposer",
        default="auto",
        choices=("auto", "llm", "random"),
        help="'random' needs no credentials and is the honest baseline",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=_cmd_run)

    ver = sub.add_parser("version", help="print the installed version")
    ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
