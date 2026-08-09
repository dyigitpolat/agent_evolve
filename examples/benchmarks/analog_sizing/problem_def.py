"""Typed candidate schema and problem binding for the analog_sizing workload.

Domain: AnalogGym (Li et al., ICCAD 2024, BSD-3-Clause), amplifier category --
the half that supports open ngspice + SkyWater sky130.  Topology
``Leung_NMCF_Pin_3`` (the "NMCF" row of the paper's Tables 7 and 8).

Every constant in this file is a WORKLOAD constant and lives here, in the
adapter package.  Nothing under ``src/agent_evolve`` learns anything about
amplifiers.

Design space is taken verbatim from AnalogGym's own reference RL implementation
(``RGNN_RL/ckt_graphs.py::GraphAMPNMCF``, ``action_space_low`` /
``action_space_high``), so the box we search is exactly the box the published
specialist searches -- representation containment is 1.0 by construction, which
is the inverse of the Heat2D situation.

OBJECTIVES.  Chosen on measured conflict over 3,970 designed uniform draws, not
by assumption (see ``domain_scouting/data/analog_gates_nmcf.json``):

    fom_s         AnalogGym's own efficiency figure, GBW*CL/Power   (maximise)
    phase_margin  AnalogGym's own stability spec                    (maximise)
    reward9       AnalogGym's own spec-attainment reward            (maximise)

rho(fom_s, phase_margin) = -0.751, the strongest genuine conflict in the domain;
|F| = 55 over the sampled corpus.  Two rejected alternatives are recorded so the
choice is auditable: ``(reward9, fom_s)`` gives |F| = 9 -- the Heat2D number --
and ``(dcgain, phase_margin)`` gives |F| = 26.

THREE HAZARDS CARRIED FORWARD, all measured rather than assumed:

1.  ACTIVE AREA IS DELIBERATELY EXCLUDED.  Area = sum(W*L*M) is an exact
    analytic function of the decision variables with no simulation involved, so
    admitting it as an objective would create a partial identity map and cap the
    frontier the way ``material_fraction`` capped Heat2D's at nine points.  The
    hazard is *avoided here, not resolved*: any formulation that adds area MUST
    re-run the identity-map check first.
2.  EFFECTIVE OBJECTIVE COUNT IS ~6, NOT 9.  ``cmrr``, ``psr_p`` and ``tc`` are
    near-collinear (Spearman up to +0.963), so they enter only through
    ``reward9`` and never as separate frontier axes.  Any reported Pareto front
    must state this.
3.  SPEC-COVERAGE SCORES ARE TRIVIALLY SATURABLE.  Uniform random at N=38
    saturates a five-objective spec subscore with probability 1.000 while all
    three published specialists score worse on it.  Everything here is therefore
    denominated in AnalogGym's own quantities and never in a coverage score of
    our construction.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from agent_evolve.agentic import ObjectiveSpec

from .evaluator import AnalogEvaluatorSettings, NgspiceSubprocessEvaluator


OBJECTIVE_FOM_S = "fom_s"
OBJECTIVE_PHASE_MARGIN = "phase_margin"
OBJECTIVE_REWARD9 = "reward9"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DeviceGroup(_StrictFrozenModel):
    """One sized transistor group: width, length, and integer multiplier."""

    w: float = Field(ge=0.5, le=10.0)
    l: float = Field(ge=0.5, le=5.0)
    m: int = Field(ge=1, le=50)


class OutputDeviceGroup(_StrictFrozenModel):
    """M11 carries a much wider multiplier range in AnalogGym's own box."""

    w: float = Field(ge=0.5, le=10.0)
    l: float = Field(ge=0.5, le=5.0)
    m: int = Field(ge=100, le=500)


class Compensation(_StrictFrozenModel):
    """Miller compensation capacitors, in farads."""

    c0: float = Field(ge=1.8228e-12, le=5.4684e-11)
    c1: float = Field(ge=1.8228e-12, le=5.4684e-11)


class AnalogSizingCandidate(_StrictFrozenModel):
    """A complete NMCF sizing, 24 typed scalars."""

    bias_pmos: DeviceGroup
    input_pair: DeviceGroup
    gm2_pmos: DeviceGroup
    gmf2_output: OutputDeviceGroup
    bias_nmos: DeviceGroup
    load2_nmos: DeviceGroup
    gm3_nmos: DeviceGroup
    compensation: Compensation
    bias_current: float = Field(ge=1e-6, le=3e-5)

    def spice_params(self) -> dict[str, float]:
        """Map the typed schema onto the netlist's own ``.PARAM`` names."""

        g = {
            "MOSFET_0_8_%s_BIASCM_PMOS": self.bias_pmos,
            "MOSFET_8_2_%s_gm1_PMOS": self.input_pair,
            "MOSFET_10_1_%s_gm2_PMOS": self.gm2_pmos,
            "MOSFET_11_1_%s_gmf2_PMOS": self.gmf2_output,
            "MOSFET_17_7_%s_BIASCM_NMOS": self.bias_nmos,
            "MOSFET_21_2_%s_LOAD2_NMOS": self.load2_nmos,
            "MOSFET_23_1_%s_gm3_NMOS": self.gm3_nmos,
        }
        out: dict[str, float] = {}
        for tmpl, dev in g.items():
            out[tmpl % "W"] = float(dev.w)
            out[tmpl % "L"] = float(dev.l)
            out[tmpl % "M"] = float(dev.m)
        out["CURRENT_0_BIAS"] = float(self.bias_current)
        out["CAPACITOR_0"] = float(self.compensation.c0)
        out["CAPACITOR_1"] = float(self.compensation.c1)
        out["CLOAD"] = 10e-12
        out["VCM"] = 0.3
        return out


# AnalogGym's shipped default sizing (Amplifier/design_variables/Leung_NMCF_Pin_3).
SEED_ANALOGGYM_DEFAULT = AnalogSizingCandidate(
    bias_pmos=DeviceGroup(w=1.0, l=1.0, m=4),
    input_pair=DeviceGroup(w=1.0, l=1.0, m=4),
    gm2_pmos=DeviceGroup(w=1.0, l=1.0, m=4),
    gmf2_output=OutputDeviceGroup(w=1.0, l=1.0, m=100),
    bias_nmos=DeviceGroup(w=1.0, l=1.0, m=4),
    load2_nmos=DeviceGroup(w=1.0, l=1.0, m=4),
    gm3_nmos=DeviceGroup(w=1.0, l=1.0, m=4),
    compensation=Compensation(c0=5e-12, c1=3e-12),
    bias_current=2e-5,
)

# Two further seeds, each ONE documented engineering move from the shipped
# default, chosen to span the measured fom_s/phase_margin conflict rather than
# picked for quality out of a sampled corpus (which would leak search effort
# into the starting point).  Measured values are recorded so the span is
# auditable:
#   analoggym_default  fom_s  544.4  PM 38.66  reward9 -3.5419
#   heavier_comp       fom_s  250.7  PM 57.77  reward9 -3.6922   (margin end)
#   wide_input         fom_s 1070.1  PM 14.28  reward9 -1.8817   (efficiency end)

# Move: quadruple both Miller capacitors -- buy phase margin, lose bandwidth.
SEED_HEAVIER_COMPENSATION = SEED_ANALOGGYM_DEFAULT.model_copy(
    update={"compensation": Compensation(c0=1.45824e-11, c1=1.45824e-11)}
)

# Move: widen and multiply the input pair, with a matching compensation bump --
# buy transconductance and bandwidth, lose phase margin.
SEED_WIDE_INPUT = SEED_ANALOGGYM_DEFAULT.model_copy(
    update={
        "input_pair": DeviceGroup(w=3.4, l=1.0, m=16),
        "compensation": Compensation(c0=7.2912e-12, c1=7.2912e-12),
    }
)


def normalize_candidate(value: object) -> AnalogSizingCandidate:
    if type(value) is AnalogSizingCandidate:
        return value
    if isinstance(value, BaseModel):
        return AnalogSizingCandidate.model_validate(value.model_dump(mode="python"))
    return AnalogSizingCandidate.model_validate(value)


def canonical_candidate_json(value: object) -> str:
    payload = normalize_candidate(value).model_dump(mode="python")
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


class AnalogSizingProblem:
    """Bind one amplifier topology under a frozen evaluator contract."""

    candidate_model = AnalogSizingCandidate
    example_config = SEED_ANALOGGYM_DEFAULT
    constraints_description = (
        "closed 24-scalar sizing surface for one amplifier topology: seven "
        "transistor groups of (width, length, integer multiplier), two Miller "
        "compensation capacitors, and a bias current; every field is typed and "
        "bounded to AnalogGym's own declared search box, so unrepresentable "
        "sizings cannot be expressed; the testbench, PDK corner, load "
        "capacitance, supply, and common-mode ratio are frozen evaluator facts"
    )

    def __init__(
        self,
        settings: AnalogEvaluatorSettings,
        *,
        evaluator: object | None = None,
    ) -> None:
        if type(settings) is not AnalogEvaluatorSettings:
            raise TypeError("settings must be exact AnalogEvaluatorSettings")
        self.settings = settings
        self._evaluator = evaluator

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (
            ObjectiveSpec(
                OBJECTIVE_FOM_S, "max",
                description=(
                    "AnalogGym's efficiency figure of merit, GBW*CL/Power "
                    "-- bandwidth delivered per unit supply power"
                ),
            ),
            ObjectiveSpec(
                OBJECTIVE_PHASE_MARGIN, "max",
                description=(
                    "closed-loop stability margin in degrees; raising "
                    "bandwidth (e.g. by shrinking the compensation "
                    "capacitor) typically spends this margin"
                ),
            ),
            ObjectiveSpec(
                OBJECTIVE_REWARD9, "max",
                description=(
                    "AnalogGym's spec-attainment reward: the sum of nine "
                    "clipped per-spec terms, each 0 when its spec is met "
                    "and negative below it -- maximised at 0 = every spec "
                    "met simultaneously"
                ),
            ),
        )

    @property
    def evaluator(self) -> object:
        if self._evaluator is None:
            self._evaluator = NgspiceSubprocessEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> dict[str, float]:
        candidate = normalize_candidate(config)
        return self.evaluator.evaluate(candidate.spice_params())

    def evaluate(self, config: object) -> dict[str, float]:
        detail = self.evaluate_detailed(config)
        return {
            OBJECTIVE_FOM_S: float(detail[OBJECTIVE_FOM_S]),
            OBJECTIVE_PHASE_MARGIN: float(detail[OBJECTIVE_PHASE_MARGIN]),
            OBJECTIVE_REWARD9: float(detail[OBJECTIVE_REWARD9]),
        }

    @staticmethod
    def candidate_key(config: object) -> str:
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:analog-sizing-candidate-key:v1\x00")
        digest.update(canonical_candidate_json(config).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def render_candidate(config: object) -> str:
        c = normalize_candidate(config)
        return (
            f"in({c.input_pair.w:.2f}/{c.input_pair.l:.2f}x{c.input_pair.m}) "
            f"out({c.gmf2_output.w:.2f}/{c.gmf2_output.l:.2f}x{c.gmf2_output.m}) "
            f"gm2({c.gm2_pmos.w:.2f}/{c.gm2_pmos.l:.2f}x{c.gm2_pmos.m}) "
            f"gm3({c.gm3_nmos.w:.2f}/{c.gm3_nmos.l:.2f}x{c.gm3_nmos.m}) "
            f"Cc({c.compensation.c0*1e12:.1f}p/{c.compensation.c1*1e12:.1f}p) "
            f"Ib({c.bias_current*1e6:.1f}u)"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Transistor sizing for the Leung NMCF three-stage nested-Miller "
            "amplifier in the open SkyWater sky130 process, simulated with "
            "ngspice at the typical corner on AnalogGym's own AC/DC testbench. "
            "The candidate sets width, length and integer multiplier for seven "
            "device groups (bias PMOS/NMOS mirrors, the input pair, the second "
            "and third gain stages, the feedforward output device, and the "
            "second-stage load), plus two Miller compensation capacitors and "
            "the bias current. Maximize AnalogGym's small-signal figure of "
            "merit (gain-bandwidth times load over power), the phase margin, "
            "and AnalogGym's own multi-spec attainment reward. Efficiency and "
            "stability are strongly opposed (rank correlation -0.75): buying "
            "bandwidth per watt costs phase margin, and compensation buys "
            "margin back at the cost of bandwidth."
        )


def default_settings(**kwargs: object) -> AnalogEvaluatorSettings:
    return AnalogEvaluatorSettings(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "OBJECTIVE_FOM_S",
    "OBJECTIVE_PHASE_MARGIN",
    "OBJECTIVE_REWARD9",
    "AnalogSizingCandidate",
    "AnalogSizingProblem",
    "Compensation",
    "DeviceGroup",
    "OutputDeviceGroup",
    "SEED_ANALOGGYM_DEFAULT",
    "SEED_HEAVIER_COMPENSATION",
    "SEED_WIDE_INPUT",
    "canonical_candidate_json",
    "default_settings",
    "normalize_candidate",
]
