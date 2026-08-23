"""``agent_evolve`` command line: ``init``, ``diagnose``, ``check``, ``run``,
``version``.

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
import json
import statistics
import sys
from typing import Any, List, Optional, Sequence

from agent_evolve.contract import as_problem
from agent_evolve.core.results import compute_pareto_front

__all__ = ["main"]


def _authorship_presets() -> tuple:
    """The preset names, read from the table that defines them."""

    from agent_evolve.session.authorship import PRESETS

    return tuple(PRESETS)


def _structure_budget_arg(value: str) -> Any:
    """``auto`` or an evaluation count; the sentinel is resolved by the API.

    argparse converts a string default through ``type``, so a plain ``int``
    type would try to parse the sentinel itself. Resolving it here instead
    would put a second copy of the sizing rule in the CLI, where it could
    drift from the one the library announces.
    """

    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--structure-budget takes 'auto' or an evaluation count, got "
            f"{value!r}"
        ) from None


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


def _verdict_rows(
    model_result: Any,
    baseline_results: Sequence[Any],
    objectives: Sequence[Any],
) -> List[dict]:
    """The comparison itself, as data: one row per objective it could judge.

    The prose verdict and the ``--json`` document are two renderings of this
    one computation. A second copy of the rule -- which draws count, how ``p``
    is formed, what "won" means -- could disagree with the first, and the
    disagreement would be invisible until someone compared two outputs of the
    same run.
    """
    rows: List[dict] = []
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
        better = sum(
            1 for d in draws if (d > model_best if spec.goal == "max" else d < model_best)
        )
        # Fraction of uninformed runs that matched or beat the model. This is
        # the chance baseline every claim in this project travels with.
        p = (better + 1) / (len(draws) + 1)
        median = statistics.median(draws)
        won = (model_best > median) if spec.goal == "max" else (model_best < median)
        rows.append({
            "objective": spec.name,
            "goal": spec.goal,
            "model_best": model_best,
            "random_median": median,
            "baseline_runs": len(draws),
            "baseline_better": better,
            "p": p,
            "model_won": won,
        })
    return rows


def _winner(rows: Sequence[dict]) -> Optional[str]:
    """``model``, ``baseline`` or ``mixed`` -- ``None`` when nothing was judged."""
    if not rows:
        return None
    beaten = sum(1 for row in rows if row["model_won"])
    if beaten == len(rows):
        return "model"
    if beaten == 0:
        return "baseline"
    return "mixed"


def _verdict(
    model_result: Any,
    baseline_results: Sequence[Any],
    objectives: Sequence[Any],
) -> List[str]:
    """State plainly whether the model beat chance, per objective."""
    out: List[str] = []
    rows = _verdict_rows(model_result, baseline_results, objectives)
    counted = len(rows)
    beaten = sum(1 for row in rows if row["model_won"])
    for row in rows:
        out.append(
            f"    {row['objective']}: model {row['model_best']:g} vs random "
            f"median {row['random_median']:g} over {row['baseline_runs']} runs "
            f" (p = {row['p']:.2f})"
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


def _arm_outcome(result: Any, objectives: Sequence[Any], seed: int) -> dict:
    """One arm's run as data: what it spent and the best it reached.

    The Pareto front's rows are deliberately not here -- ``run --json`` is the
    command that hands you a front, and duplicating it under five baseline
    repeats would bury the one thing ``check`` exists to answer.
    """
    import dataclasses

    best = {}
    for spec in objectives:
        values = _front_values(result, spec.name)
        if values:
            best[spec.name] = (max if spec.goal == "max" else min)(values)
    return {
        "seed": seed,
        "evaluations": result.evaluations,
        "pareto_front": len(result.pareto_front),
        "best": best,
        "provider_usage": (
            dataclasses.asdict(result.provider_usage)
            if result.provider_usage is not None else None
        ),
    }


def _check_json(
    args: argparse.Namespace,
    objectives: Sequence[Any],
    baselines: Sequence[Any],
    model_result: Any,
    model_error: Optional[BaseException],
) -> str:
    """One machine-readable document for a ``check`` verdict.

    Same conventions as ``run --json``: a block nobody could populate
    serializes as ``null`` rather than being omitted, so a reader can tell
    "measured nothing" from "nobody looked" -- ``verdict: null`` under
    ``--baseline-only`` is the second of those, and it is the honest answer
    when no model ever ran.

    The resolved model and its price ride in the document, because ``check``'s
    contract is that nobody is billed by a default they never saw, and a
    machine-readable mode that dropped the price would quietly break it.
    """
    from agent_evolve.settings import AgentEvolveSettings, model_price

    model_arm: Optional[dict] = None
    resolved: Optional[str] = None
    price = None
    if not args.baseline_only:
        resolved = args.model or AgentEvolveSettings.from_env().model
        price = model_price(resolved)
        if model_error is not None:
            model_arm = {
                "seed": 0,
                "error": f"{type(model_error).__name__}: {model_error}",
            }
        else:
            model_arm = _arm_outcome(model_result, objectives, seed=0)
            model_arm["error"] = None

    rows = (
        _verdict_rows(model_result, baselines, objectives)
        if model_result is not None else []
    )
    verdict = None
    if model_arm is not None and model_error is None:
        verdict = {
            "objectives": rows,
            "objectives_judged": len(rows),
            "objectives_won": sum(1 for row in rows if row["model_won"]),
            "winner": _winner(rows),
        }

    payload = {
        "command": "check",
        "problem": args.problem,
        "budget": args.budget,
        "repeats": args.repeats,
        "baseline_only": bool(args.baseline_only),
        "model": resolved,
        "model_is_default": (None if resolved is None else args.model is None),
        "model_price_per_mtok": (
            None if price is None else {"input": price[0], "output": price[1]}
        ),
        "objectives": [
            {"name": spec.name, "goal": spec.goal} for spec in objectives
        ],
        "arms": {
            "baseline": {
                "proposer": "random",
                "runs": [
                    _arm_outcome(result, objectives, seed=i)
                    for i, result in enumerate(baselines)
                ],
            },
            "model": model_arm,
        },
        "verdict": verdict,
        "provider_usage": (
            None if model_arm is None else model_arm.get("provider_usage")
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _cmd_check(args: argparse.Namespace) -> int:
    from agent_evolve.api import optimize

    problem = _load_problem(args.problem)
    objectives = list(problem.objectives)
    # `run --json`'s convention: one parseable document on stdout and nothing
    # else. The prose does not disappear, it moves to stderr -- including the
    # model's price, which `check` states BEFORE it spends, and which a
    # machine-readable mode has no business making quieter. `2>/dev/null` then
    # leaves exactly the document.
    stream = sys.stderr if args.json else sys.stdout

    def emit(message: str = "") -> None:
        print(message, file=stream, flush=True)

    quiet = (lambda _m: None) if not args.verbose else emit

    emit(f"agent_evolve check: {args.problem}")
    emit(f"budget {args.budget} evaluations per run, {args.repeats} baseline runs")
    if args.baseline_only:
        emit("baseline only: no model, no credentials, no cost\n")
    else:
        emit(f"{_model_line(args.model)}\n")

    baselines = []
    for i in range(args.repeats):
        baselines.append(
            optimize(problem, budget=args.budget, proposer="random", seed=i, on_progress=quiet)
        )
    emit(_summarise(f"random baseline (run 1 of {args.repeats})", baselines[0], objectives))

    if args.baseline_only:
        emit(
            "\n  Baseline only. Re-run without --baseline-only, with a provider "
            "credential set, to compare a model against it."
        )
        if args.json:
            print(_check_json(args, objectives, baselines, None, None))
        return 0

    emit()
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
        emit(f"  model run failed: {type(error).__name__}: {error}")
        emit("\n  The baseline above still stands, and cost nothing.")
        if args.json:
            print(_check_json(args, objectives, baselines, None, error))
        return 1
    emit(_summarise("model", model_result, objectives))
    emit("\n  verdict")
    for line in _verdict(model_result, baselines, objectives):
        emit(line)
    if args.json:
        print(_check_json(args, objectives, baselines, model_result, None))
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Probe the problem itself: search space, pipeline health, and headroom.

    Where ``check`` asks whether a *model* beats uninformed sampling, this asks
    the prior question: whether *anything* could demonstrate an advantage on
    this problem at this budget. It needs no model and no credentials, and it
    spends at most ``--probe`` evaluations.
    """
    from agent_evolve.policies.check import check as check_problem

    problem = _load_problem(args.problem)
    print(f"agent_evolve diagnose: {args.problem}\n")
    report = check_problem(problem, args.budget, probe=args.probe, seed=args.seed)
    print(report.render())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from agent_evolve.api import optimize

    problem = _load_problem(args.problem)
    if args.json:
        # One parseable document on stdout and nothing else: progress moves to
        # stderr under --verbose and is dropped otherwise.
        progress = (
            (lambda m: print(m, file=sys.stderr, flush=True))
            if args.verbose else (lambda _m: None)
        )
    else:
        if args.proposer != "random":
            print(_model_line(args.model))
        progress = (lambda m: print(m, flush=True)) if args.verbose else print
    result = optimize(
        problem,
        budget=args.budget,
        model=args.model,
        proposer=args.proposer,
        strategy=args.strategy,
        seed=args.seed,
        seal=args.seal,
        structure_budget=args.structure_budget,
        prior=args.prior,
        chooser=args.chooser,
        effort=args.effort,
        journal=args.journal,
        authorship=args.authorship,
        on_progress=progress,
    )
    if args.json:
        from agent_evolve.core.formatting import result_to_json

        print(result_to_json(result))
        return 0
    print(f"\nbest        {result.best.configuration}")
    print(f"objectives  {result.best.objectives}")
    print(f"pareto      {len(result.pareto_front)}")
    print(f"evaluations {result.evaluations}")
    for i, c in enumerate(result.pareto_front, 1):
        print(f"  {i}. {c.configuration} -> {c.objectives}")
    return 0


#: The five obligations with this project's problem removed and yours left to
#: write. It is the knapsack example's shape -- the same order, the same
#: comments about why each obligation exists -- with the knapsack taken out.
#:
#: It imports and it is a valid ``Problem`` the moment it lands, so ``diagnose``
#: can be run against it immediately; only ``evaluate`` refuses, by name,
#: because measuring is the one obligation nothing can guess for you. A template
#: that returned a plausible number instead would let a run look like it worked.
_SCAFFOLD = '''"""Your problem, as the five obligations ``agent_evolve`` asks for.

Fill in the parts marked TODO. The README section "Describing your problem:
five obligations" explains each one and what it buys you; the worked reference
is ``examples/knapsack/problem_def.py``.

    candidate_model   the schema a proposal must satisfy
    objectives        what is optimized, and which way
    seeds()           where to start
    validate()        cheap rejection that explains itself
    materialize()     candidate -> the artifact that gets measured
    evaluate()        artifact -> objective values

Then, before spending anything::

    agent_evolve diagnose problem_def:problem --budget 40
    agent_evolve run problem_def:problem --budget 40 --proposer random
"""

from pydantic import BaseModel, Field

from agent_evolve import ObjectiveSpec, ValidationOutcome


class CandidateConfig(BaseModel):
    """One candidate configuration.

    TODO: your decision variables. Declare their domains here -- an enum, a
    ``Literal``, a bounded number -- because everything that reads this schema,
    including the uninformed sampler your run is measured against, draws only
    from what it declares. A field with no finite reading is one the operators
    leave frozen, and that is the commonest reason a run goes nowhere.
    """

    workers: int = Field(..., ge=1, le=64, description="TODO: describe this axis")


class MyProblem:
    """TODO: one line saying what is being optimized, and under what limit."""

    candidate_model = CandidateConfig

    # -- 1. what is being optimized ---------------------------------------
    @property
    def objectives(self):
        # TODO: name each objective and its direction. Direction is declared,
        # never encoded by negating a value.
        return [
            ObjectiveSpec("throughput", "max"),
            ObjectiveSpec("cost", "min"),
        ]

    # -- 2. where to start -------------------------------------------------
    def seeds(self):
        """Configurations you would have tried anyway.

        Seeds are evaluated before anything is proposed, so the result answers
        "did this beat what I already had" rather than leaving it assumed.
        Return ``[]`` if you have none.
        """
        # TODO
        return [{"workers": 8}]

    # -- 3. cheap rejection that explains itself ---------------------------
    def validate(self, config) -> ValidationOutcome:
        """Reject what cannot work, and say what would.

        The message is fed back to the proposer verbatim, so state what is
        wrong AND what would be acceptable. A rejection costs no evaluation.
        """
        # TODO: your feasibility rules, e.g.
        #   return ValidationOutcome(
        #       False, "constraint",
        #       "workers above 32 needs the sharded strategy; reduce workers",
        #   )
        return ValidationOutcome(True)

    # -- 4. candidate -> the artifact that gets measured -------------------
    def materialize(self, config):
        """Canonicalise to the thing that actually gets measured.

        Two configurations often produce the same artifact -- the same build,
        the same mapping, the same deployment. Materializing first means the
        second one is free instead of being paid for twice. Put anything cheap
        and deterministic here and keep ``evaluate`` for the expensive part.
        """
        # TODO
        return (config["workers"],)

    # -- 5. measure it -----------------------------------------------------
    def evaluate(self, artifact):
        """Measure the artifact and return one value per objective."""
        # TODO: run the build, the simulation, the benchmark -- the expensive
        # thing. `budget` counts calls to this method, so this is what you are
        # paying for.
        raise NotImplementedError(
            "evaluate() is the one obligation nothing can guess for you: "
            "return {\\"throughput\\": ..., \\"cost\\": ...} for this artifact"
        )

    # -- optional: prose context for a model-driven proposer ---------------
    def search_space_description(self):
        # TODO, or delete: what a model should know about this space that the
        # schema cannot say. Only the `llm` proposer reads it.
        return ""


# `module:attribute` on the command line resolves to this name.
problem = MyProblem()
'''


def _cmd_init(args: argparse.Namespace) -> int:
    """Write the five-obligation template, and refuse to overwrite anything.

    A scaffold that clobbers is a scaffold nobody can run twice, and the file
    it would clobber is the one thing in the directory nobody else can rewrite.
    """
    from pathlib import Path

    target = Path(args.path)
    if target.is_dir() or target.suffix != ".py":
        target = target / "problem_def.py"
    if target.exists():
        raise SystemExit(
            f"refusing to overwrite {target}. Name another path, or move the "
            "existing file first."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SCAFFOLD, encoding="utf-8")
    module = target.stem
    print(f"wrote {target}")
    print("")
    print("Fill in the parts marked TODO, then, before spending anything:")
    print(f"  agent_evolve diagnose {module}:problem --budget 40")
    print(f"  agent_evolve run {module}:problem --budget 40 --proposer random")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    try:
        from importlib.metadata import version

        # The DISTRIBUTION name; the import stays agent_evolve. See pyproject.
        print(version("agentevolve"))
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
    check.add_argument(
        "--repeats", type=int, default=5,
        help=(
            "uninformed BASELINE runs (default 5, seeds 0..N-1). The model arm "
            "is one run at seed 0 and this does not repeat it, so the spread "
            "you see is chance's, not the model's"
        ),
    )
    check.add_argument("--model", default=None, help="model id for the model arm")
    check.add_argument(
        "--baseline-only",
        action="store_true",
        help="run only the free baseline; no credentials needed",
    )
    check.add_argument(
        "--json", action="store_true",
        help=(
            "print one machine-readable JSON document of the verdict on "
            "stdout; the prose moves to stderr rather than being dropped"
        ),
    )
    check.add_argument("--verbose", action="store_true")
    check.set_defaults(func=_cmd_check)

    diagnose = sub.add_parser(
        "diagnose",
        help="could ANY optimizer show an advantage on your problem at this budget?",
        description=(
            "Probes the problem with schema-uniform draws through its own "
            "validate/materialize/evaluate pipeline and reports the locus and "
            "domain structure, failure rate, evaluation cost, per-objective "
            "spread, and whether best-of-budget random draws already reach the "
            "best the probe found. Spends at most --probe evaluations; needs "
            "no model and no credentials. Run it before `check`, which spends "
            "model money to answer the next question."
        ),
    )
    diagnose.add_argument("problem", help="module:attribute naming your problem")
    diagnose.add_argument(
        "--budget", type=int, default=40,
        help="the optimizer budget being assessed (not the probe's spend)",
    )
    diagnose.add_argument(
        "--probe", type=int, default=120,
        help="schema-uniform draws the probe spends (default 120)",
    )
    diagnose.add_argument("--seed", type=int, default=None)
    diagnose.set_defaults(func=_cmd_diagnose)

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
    run.add_argument(
        "--strategy",
        default="auto",
        choices=("auto", "genetic", "authoring"),
        help="'auto' prefers the genetic loop when the problem has seeds",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--seal",
        default=None,
        metavar="PATH",
        help=(
            "write the run's chained proposal journal here; requires the "
            "authoring strategy (the genetic loop refuses it by name)"
        ),
    )
    run.add_argument(
        "--structure-budget", type=_structure_budget_arg, default="auto",
        dest="structure_budget",
        help=(
            "evaluations to spend on a crossed screen before the population; "
            "charged against --budget, not free. 'auto' (default) skips it "
            "below a budget of 48 and sizes it from the budget above that"
        ),
    )
    run.add_argument(
        "--prior",
        default="auto",
        choices=("auto", "rule", "rule-weighted", "llm", "llm-weighted"),
        help=(
            "who turns the screen into a sampling prior; the llm forms fall "
            "back to their rule comparator, out loud, without a credential. "
            "'auto' (default) is 'rule' offline and 'llm-weighted' on a model "
            "run, announced either way"
        ),
    )
    run.add_argument(
        "--chooser",
        default="off",
        choices=("off", "llm"),
        help=(
            "who picks parents and cut points. 'llm' spends one model call per "
            "offspring; it returned ten sealed null verdicts at 107-171x the "
            "cost of the run it advises, and consumed 61%% of the six-arm "
            "ablation's ledger for 0.94x the speed of doing nothing. 'off' "
            "(default) is the random control it never beat"
        ),
    )
    run.add_argument(
        "--effort", default=None,
        help="reasoning-effort pin for every model call (e.g. low, high)",
    )
    run.add_argument(
        "--journal", default=None, metavar="PATH",
        help="write one JSON line per completed model call (model, usage)",
    )
    run.add_argument(
        "--authorship",
        default="auto",
        # Enumerated from the preset table, never repeated: a mechanism that
        # is reachable from the library but not the CLI is half-shipped.
        choices=("auto",) + tuple(_authorship_presets()),
        help=(
            "authored machinery: 'surrogate[-llm]' turns on virtual "
            "pre-screening (model-written surrogates screen only when they "
            "out-validate the rules); 'operators[-llm]' runs variation arms "
            "under survival credit; 'generation-llm' lets the model write the "
            "sampler every candidate is drawn from, 'generative' puts that "
            "sampler under the authored screen; 'guided' is what 'auto' "
            "resolves to on a model run (authored surrogate + model-proposed "
            "initialization, the two measured winners); 'full' is surrogate + "
            "operators + init, model-authored"
        ),
    )
    run.add_argument(
        "--json", action="store_true",
        help="print one machine-readable JSON document instead of prose",
    )
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=_cmd_run)

    init = sub.add_parser(
        "init",
        help="write a problem_def.py template: the five obligations, blank",
        description=(
            "Writes the five-obligation template -- the shipped knapsack "
            "example's shape with the knapsack removed. PATH may be a "
            "directory (problem_def.py is written inside it) or a .py file to "
            "write. An existing file is never overwritten."
        ),
    )
    init.add_argument(
        "path", nargs="?", default=".",
        help="directory to write problem_def.py into, or a .py path (default .)",
    )
    init.set_defaults(func=_cmd_init)

    ver = sub.add_parser("version", help="print the installed version")
    ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
