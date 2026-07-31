"""In-venv PyBaMM fast-charging runner for the pybamm_fastcharge workload.

This script executes inside the isolated PyBaMM domain virtualenv
(``domain_envs/pybamm/venv``); it uses only the standard library plus pybamm
and numpy.  Importing it from the agent_evolve environment performs no solver
work and must not require pybamm.

Frozen Gate-A evaluation policy (jul27 standup memo, sections 2 and 6): DFN
with lumped thermal on the Chen2020 (LG M50) parameter set from 10% SOC, the
IDAKLU solver, mesh factor as the documented fidelity knob (production 2,
test 1), charge to 4.2 V through three CC stages with two switch voltages,
then CV hold to C/50.  The runner reports termination facts verbatim; the
adapter-side evaluator enforces the termination-validity gate because IDAKLU
convergence failures can return a silently early-terminated solution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback


SCHEMA_VERSION = 1
EVALUATOR_ID = "pybamm-fastcharge-dfn-idaklu-v1"
MODEL = "DFN"
THERMAL = "lumped"
PARAMETER_SET = "Chen2020"
SOLVER = "idaklu"
VMAX_V = 4.2
CV_CUTOFF = "C/50"
PLATING_PROXY_VARS = (
    "X-averaged negative electrode surface potential difference [V]",
    "Negative electrode surface potential difference at separator interface [V]",
    "X-averaged negative electrode reaction overpotential [V]",
)


def _solve(request):
    import numpy as np
    import pybamm

    protocol = request["protocol"]
    c1 = float(protocol["stage1_c_rate"])
    c2 = float(protocol["stage2_c_rate"])
    c3 = float(protocol["stage3_c_rate"])
    v1 = float(protocol["switch_v1"])
    v2 = float(protocol["switch_v2"])
    mesh_factor = int(request["mesh_factor"])
    rtol = float(request["rtol"])
    atol = float(request["atol"])
    initial_soc = float(request["initial_soc"])
    started = time.perf_counter()
    try:
        model = pybamm.lithium_ion.DFN(options={"thermal": THERMAL})
        params = pybamm.ParameterValues(PARAMETER_SET)
        var_pts = {
            key: int(value * mesh_factor)
            for key, value in model.default_var_pts.items()
        }
        experiment = pybamm.Experiment(
            [
                f"Charge at {c1}C until {v1} V",
                f"Charge at {c2}C until {v2} V",
                f"Charge at {c3}C until {VMAX_V} V",
                f"Hold at {VMAX_V} V until {CV_CUTOFF}",
            ]
        )
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        simulation = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=params,
            var_pts=var_pts,
            solver=solver,
        )
        solution = simulation.solve(initial_soc=initial_soc)
        times = solution["Time [s]"].entries
        voltage = solution["Voltage [V]"].entries
        temperature = solution["X-averaged cell temperature [K]"].entries
        current = solution["Current [A]"].entries
        plating_min = None
        plating_var = None
        for name in PLATING_PROXY_VARS:
            try:
                plating_min = float(np.min(solution[name].entries))
                plating_var = name
                break
            except Exception:  # noqa: BLE001 - try the next candidate name
                continue
        return {
            "status": "ok",
            "termination": str(solution.termination),
            "n_steps_solved": len(solution.cycles),
            "charge_time_s": round(float(times[-1] - times[0]), 3),
            "peak_temp_rise_K": round(
                float(np.max(temperature) - temperature[0]), 4
            ),
            "plating_proxy_min_V": plating_min,
            "plating_proxy_var": plating_var,
            "final_voltage_V": round(float(voltage[-1]), 6),
            "final_current_A": round(float(current[-1]), 6),
            "v_max_seen_V": round(float(np.max(voltage)), 6),
            "n_timesteps": int(len(times)),
            "solve_wall_s": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - typed invalid outcome
        return {
            "status": "solver_error",
            "error": f"{type(exc).__name__}: {exc}",
            "solve_wall_s": round(time.perf_counter() - started, 3),
        }


def _run(request):
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    if type(request.get("protocol")) is not dict:
        raise ValueError("protocol must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "policy": {
            "model": MODEL,
            "thermal": THERMAL,
            "parameter_set": PARAMETER_SET,
            "solver": SOLVER,
            "vmax_v": VMAX_V,
            "cv_cutoff": CV_CUTOFF,
            "mesh_factor": int(request["mesh_factor"]),
            "rtol": float(request["rtol"]),
            "atol": float(request["atol"]),
            "initial_soc": float(request["initial_soc"]),
        },
        "result": _solve(request),
    }


def main():
    parser = argparse.ArgumentParser(description=EVALUATOR_ID)
    parser.add_argument("--request", required=True, help="request JSON path")
    args = parser.parse_args()
    try:
        with open(args.request, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        payload = json.dumps(_run(request), allow_nan=False, sort_keys=True)
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
    print(payload)


if __name__ == "__main__":
    main()
