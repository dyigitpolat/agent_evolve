"""Frozen ngspice evaluator boundary for the analog_sizing workload.

SIMULATOR PIN IS LOAD-BEARING.  This adapter requires **ngspice >= 42, built
from source**.  AnalogGym states that ngspice-41 -- which is the newest build
available from conda-forge -- is wrong for DC sweeps involving temperature, and
this testbench sweeps -40..125 C to measure the temperature coefficient.  A
conda install therefore produces a silently wrong result rather than an error,
so the version is asserted at construction and recorded in the preflight
receipt.

Everything below is a workload constant and belongs to the adapter, not to the
framework.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


DEFAULT_ROOT = os.environ.get(
    "ANALOG_ROOT", "/home/yigit/repos/research_stuff/domain_envs/analog"
)
TOPOLOGY = "Leung_NMCF_Pin_3"
CL_PF = 500.0            # PARAM_CLOAD in AnalogGym's shipped AC/DC testbench
MIN_NGSPICE_MAJOR = 42   # AnalogGym's own floor; 41 is wrong for DC temp sweeps

# AnalogGym spec targets, transcribed verbatim from
# RGNN_RL/ckt_graphs.py::GraphAMPNMCF.  Power is in mW; the .meas returns W.
TARGETS = {
    "phase_margin": 60.0, "dcgain": 130.0, "psr_p": -80.0, "psr_n": -80.0,
    "cmrr": -80.0, "vos": 0.06e-3, "tc": 10e-6, "power": 0.3, "gbw": 1.2e6,
}

_CONTROL = """.control
set filetype=ascii
DC temp -40 125 0.5
ac dec 10 0.1 1G
print dcgain gain_bandwidth_product phase_in_deg cmrrdc DCPSRp DCPSRn TC Power vos25
.endc"""

_MEAS_KEYS = {
    "dcgain": "dcgain", "gain_bandwidth_product": "gbw", "phase_in_deg": "phase",
    "cmrrdc": "cmrr", "dcpsrp": "psr_p", "dcpsrn": "psr_n",
    "tc": "tc", "power": "power", "vos25": "vos",
}
_LINE = re.compile(r"\s*([a-z_0-9]+)\s*=\s*(-?[\d.]+e?[-+]?\d*)\s*$", re.I)


@dataclass(frozen=True, slots=True)
class AnalogEvaluatorSettings:
    root: str = DEFAULT_ROOT
    topology: str = TOPOLOGY
    timeout_s: int = 300
    cl_pf: float = CL_PF

    @property
    def amp_dir(self) -> str:
        return f"{self.root}/AnalogGym/AnalogGym/Amplifier"

    @property
    def pdk_dir(self) -> str:
        return f"{self.root}/pdk"

    @property
    def ngspice(self) -> str:
        return f"{self.root}/ngspice-install/bin/ngspice"


def _lo_better(target: float, x: float) -> float:
    return min((target - x) / (target + x), 0.0) if (target + x) else 0.0


def _hi_better(target: float, x: float) -> float:
    return min((x - target) / (x + target), 0.0) if (x + target) else 0.0


def _db_spec(target: float, x: float) -> float:
    if x > 0:
        return -1.0
    if x < target:
        return 0.0
    return min((x - target) / (x + target), 0.0) if (x + target) else 0.0


def analoggym_scores(m: dict[str, float]) -> dict[str, float]:
    """AnalogGym's own reward, 9 of its 11 terms.

    ``sr`` and ``settlingTime`` require the separate transient testbench and are
    DECLARED MISSING rather than imputed.  Each term is <= 0 and equals 0 exactly
    when the spec is met, so the reward is maximised at 0.
    """

    s = {
        "s_tc": _lo_better(TARGETS["tc"], m.get("tc", 1e9)),
        "s_power": _lo_better(TARGETS["power"], m.get("power", 1e9) * 1e3),
        "s_vos": _lo_better(TARGETS["vos"], abs(m.get("vos", 1e9))),
        "s_cmrr": _db_spec(TARGETS["cmrr"], m.get("cmrr", 1.0)),
        "s_psr_p": _db_spec(TARGETS["psr_p"], m.get("psr_p", 1.0)),
        "s_psr_n": _db_spec(TARGETS["psr_n"], m.get("psr_n", 1.0)),
    }
    if m.get("dcgain", 0.0) > 0:
        s["s_dcgain"] = _hi_better(TARGETS["dcgain"], m["dcgain"])
        s["s_gbw"] = _hi_better(TARGETS["gbw"], m.get("gbw", 0.0))
        s["s_pm"] = _hi_better(TARGETS["phase_margin"], m.get("phase_margin", 0.0))
    else:
        s["s_dcgain"] = s["s_gbw"] = s["s_pm"] = -1.0
    s["reward9"] = sum(s.values())
    s["specs_met"] = float(
        sum(1 for k, v in s.items() if k.startswith("s_") and v == 0.0)
    )
    return s


class NgspiceSubprocessEvaluator:
    """One typed sizing -> one AnalogGym AC/DC simulation -> objective values."""

    def __init__(self, settings: AnalogEvaluatorSettings) -> None:
        if type(settings) is not AnalogEvaluatorSettings:
            raise TypeError("settings must be exact AnalogEvaluatorSettings")
        self.settings = settings
        self._template: str | None = None
        self.ngspice_version = self._assert_simulator()

    def _assert_simulator(self) -> str:
        exe = self.settings.ngspice
        if not os.path.exists(exe):
            raise RuntimeError(
                f"ngspice not found at {exe}; this workload requires ngspice "
                f">= {MIN_NGSPICE_MAJOR} BUILT FROM SOURCE (conda-forge ships 41, "
                "which is wrong for the DC temperature sweep this testbench uses)"
            )
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=60).stdout
        m = re.search(r"ngspice-(\d+)", out)
        if m is None:
            raise RuntimeError(f"could not parse an ngspice version from: {out!r}")
        major = int(m.group(1))
        if major < MIN_NGSPICE_MAJOR:
            raise RuntimeError(
                f"ngspice-{major} is below the AnalogGym floor of "
                f"{MIN_NGSPICE_MAJOR}: version 41 silently mis-simulates DC "
                "temperature sweeps, which this testbench relies on for TC"
            )
        return f"ngspice-{major}"

    def _deck_template(self) -> str:
        if self._template is None:
            s = self.settings
            tb = open(f"{s.amp_dir}/amp_spice_testbench/TB_Amplifier_ACDC.cir").read()
            tb = tb.replace("../spice_netlist/HoiLee_AFFC_Pin_3",
                            f"{s.amp_dir}/spice_netlist/{s.topology}")
            tb = tb.replace("../design_variables/HoiLee_AFFC_Pin_3", "__PARAMS__")
            tb = tb.replace("../mosfet_model/sky130_pdk", f"{s.pdk_dir}/sky130_pdk")
            tb = tb.replace("HoiLee_AFFC_Pin_3", s.topology)
            tb = re.sub(r"\.control.*?\.endc", _CONTROL, tb, flags=re.S)
            self._template = tb
        return self._template

    def materialize(self, params: dict[str, float]) -> str:
        """Render the typed sizing into a complete, self-contained SPICE deck."""

        block = ".PARAM\n" + "".join(f"+ {k}={v:.8g}\n" for k, v in params.items())
        return self._deck_template().replace(".include __PARAMS__", block)

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in text.split("\n"):
            m = _LINE.match(line.strip())
            if m and m.group(1).lower() in _MEAS_KEYS:
                try:
                    out[_MEAS_KEYS[m.group(1).lower()]] = float(m.group(2))
                except ValueError:
                    pass
        return out

    def evaluate(self, params: dict[str, float]) -> dict[str, float]:
        d = tempfile.mkdtemp(prefix="analog_", dir="/dev/shm")
        try:
            deck = os.path.join(d, "tb.cir")
            with open(deck, "w") as fh:
                fh.write(self.materialize(params))
            env = dict(os.environ, OMP_NUM_THREADS="1", NGSPICE_MAXTHREADS="1")
            r = subprocess.run(
                [self.settings.ngspice, "-b", deck], cwd=d, env=env,
                capture_output=True, text=True, timeout=self.settings.timeout_s,
            )
            met = self._parse(r.stdout + "\n" + r.stderr)
            if not all(k in met for k in ("dcgain", "gbw", "phase", "power")):
                raise RuntimeError(
                    "ngspice produced no usable measurements for this sizing "
                    f"(got {sorted(met)}); the candidate is electrically "
                    "non-functional rather than schema-invalid"
                )
            ph = met["phase"]
            # AnalogGym takes the measured phase AS the margin, clamping
            # out-of-range values to 0 (RGNN_RL/AMP_NMCF.py:516-522).
            met["phase_margin"] = ph if 0.0 <= ph <= 180.0 else 0.0
            met["fom_s"] = (
                (met["gbw"] / 1e6) * self.settings.cl_pf / (met["power"] * 1e3)
                if met["power"] > 0 and met["gbw"] > 0 else 0.0
            )
            met.update(analoggym_scores(met))
            return met
        finally:
            shutil.rmtree(d, ignore_errors=True)


__all__ = [
    "AnalogEvaluatorSettings",
    "CL_PF",
    "MIN_NGSPICE_MAJOR",
    "NgspiceSubprocessEvaluator",
    "TARGETS",
    "TOPOLOGY",
    "analoggym_scores",
]
