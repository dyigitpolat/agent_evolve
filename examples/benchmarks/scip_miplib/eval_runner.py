"""In-venv SCIP panel runner for the scip_miplib workload.

This script executes inside the isolated SCIP domain virtualenv
(``domain_envs/scip/venv``); it uses only the standard library plus pyscipopt.
Importing it from the agent_evolve environment performs no solver work and
must not require pyscipopt.

Frozen Gate-A evaluation policy (jul27 standup memo, section 1):
``limits/time`` per instance from the request (30 s production default),
``randomization/randomseedshift`` pinned, single-threaded SCIP default
``optimize()``, quiet output.  The objective evidence per instance is the
primal-dual integral average percent parsed from ``writeStatistics()`` with
the final relative gap as fallback.  Parameter application order is emphasis
meta-settings first, then the curated raw parameters in sorted order; any
``setParam`` rejection aborts the run as a typed ``param_error`` before
``optimize()`` so a partially applied configuration is never measured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback


SCHEMA_VERSION = 1
EVALUATOR_ID = "scip-miplib-panel-config-v1"
_EMPHASIS_GROUPS = ("separating", "heuristics", "presolve")
_EMPHASIS_LEVELS = ("off", "default", "fast", "aggressive")


def _parse_pd_integral(stats_path):
    """Parse the 'primal-dual' integral line (total, avg%) from SCIP stats."""

    try:
        with open(stats_path, "r", errors="replace") as handle:
            for line in handle:
                if "primal-dual" in line:
                    numbers = re.findall(
                        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line
                    )
                    if len(numbers) >= 2:
                        return float(numbers[0]), float(numbers[1])
    except OSError:
        pass
    return None, None


def _solve_one(instance_path, setparams, emphasis, time_limit_s, seed):
    from pyscipopt import Model, SCIP_PARAMSETTING

    levels = {
        "off": SCIP_PARAMSETTING.OFF,
        "default": SCIP_PARAMSETTING.DEFAULT,
        "fast": SCIP_PARAMSETTING.FAST,
        "aggressive": SCIP_PARAMSETTING.AGGRESSIVE,
    }
    started = time.perf_counter()
    model = Model()
    model.hideOutput()
    try:
        model.readProblem(instance_path)
    except Exception as exc:  # noqa: BLE001 - typed invalid outcome
        return {
            "instance": os.path.basename(instance_path),
            "status": "read_error",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.perf_counter() - started, 3),
        }
    setters = {
        "separating": model.setSeparating,
        "heuristics": model.setHeuristics,
        "presolve": model.setPresolve,
    }
    for group in _EMPHASIS_GROUPS:
        level = emphasis.get(group, "default")
        if level != "default":
            setters[group](levels[level])
    model.setParam("limits/time", float(time_limit_s))
    model.setParam("randomization/randomseedshift", int(seed))
    model.setParam("display/verblevel", 0)
    param_errors = {}
    for key in sorted(setparams):
        try:
            model.setParam(key, setparams[key])
        except Exception as exc:  # noqa: BLE001 - collected per key
            param_errors[key] = f"{type(exc).__name__}: {exc}"
    if param_errors:
        model.freeProb()
        return {
            "instance": os.path.basename(instance_path),
            "status": "param_error",
            "param_errors": param_errors,
            "wall_s": round(time.perf_counter() - started, 3),
        }
    model.optimize()
    stats_fd, stats_path = tempfile.mkstemp(suffix=".stats")
    os.close(stats_fd)
    pd_total = pd_avg = None
    try:
        model.writeStatistics(stats_path)
        pd_total, pd_avg = _parse_pd_integral(stats_path)
    except Exception:  # noqa: BLE001 - statistics are best-effort evidence
        pass
    finally:
        if os.path.exists(stats_path):
            os.unlink(stats_path)
    try:
        gap = model.getGap()
    except Exception:  # noqa: BLE001
        gap = None
    if gap is not None and gap >= 1e18:
        gap = None
    record = {
        "instance": os.path.basename(instance_path),
        "status": model.getStatus(),
        "wall_s": round(time.perf_counter() - started, 3),
        "solving_s": round(model.getSolvingTime(), 3),
        "gap": gap,
        "primal": model.getPrimalbound(),
        "dual": model.getDualbound(),
        "nodes": model.getNNodes(),
        "pd_integral_total": pd_total,
        "pd_integral_avg_pct": pd_avg,
    }
    model.freeProb()
    return record


def _run(request):
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    instances = request["instances"]
    setparams = request["setparams"]
    emphasis = request["emphasis"]
    time_limit_s = float(request["time_limit_s"])
    seed = int(request["random_seed_shift"])
    if type(instances) is not list or not instances:
        raise ValueError("instances must be a non-empty list of paths")
    if type(setparams) is not dict or type(emphasis) is not dict:
        raise ValueError("setparams and emphasis must be objects")
    for group, level in emphasis.items():
        if group not in _EMPHASIS_GROUPS or level not in _EMPHASIS_LEVELS:
            raise ValueError(f"invalid emphasis entry: {group}={level}")
    results = [
        _solve_one(path, setparams, emphasis, time_limit_s, seed)
        for path in instances
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "policy": {
            "time_limit_s": time_limit_s,
            "random_seed_shift": seed,
            "threads": "scip_default_single_thread_optimize",
            "application_order": "emphasis_then_sorted_raw_setparams",
            "objective_evidence": "pd_integral_avg_pct_fallback_final_gap",
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=EVALUATOR_ID)
    parser.add_argument("--request", required=True, help="request JSON path")
    args = parser.parse_args()
    try:
        with open(args.request, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        response = _run(request)
    except Exception:  # noqa: BLE001 - single typed contract failure channel
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "evaluator_id": EVALUATOR_ID,
                    "status": "runner_error",
                    "error": traceback.format_exc(limit=20),
                }
            )
        )
        sys.exit(1)
    print(json.dumps(response, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
