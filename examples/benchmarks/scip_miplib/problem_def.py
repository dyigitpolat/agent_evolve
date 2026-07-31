"""Typed SCIP solver-configuration problem over a curated MIPLIB panel.

The candidate is a grouped, bounded subset of SCIP 10.0's 3082-parameter
surface: 20 raw parameters across the memo's five mechanism groups plus the
three emphasis meta-settings.  Every raw parameter name, default, and grid
value was dry-``setParam``-verified against the installed pyscipopt 6.2.1
build during the jul27 standup.  The development/held-out instance families
are frozen in ``instance_split.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_evolve.agentic import ObjectiveSpec

from .evaluator import (
    OBJECTIVE_GAP_TIME,
    OBJECTIVE_PD_INTEGRAL,
    ScipEvaluatorSettings,
    ScipPanelEvaluation,
    ScipSubprocessEvaluator,
    default_domain_python,
    default_instances_root,
)


SCHEMA_VERSION = 1
REPRESENTATION_ID = "scip-miplib-curated-params-v1"
Emphasis = Literal["off", "default", "fast", "aggressive"]
FreqInt = Annotated[int, Field(strict=True, ge=-1, le=100)]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class SeparatingConfig(_StrictFrozenModel):
    emphasis: Emphasis
    maxrounds: Annotated[int, Field(strict=True, ge=-1, le=100)]
    maxroundsroot: Annotated[int, Field(strict=True, ge=-1, le=200)]
    maxcuts: Annotated[int, Field(strict=True, ge=0, le=1000)]
    maxcutsroot: Annotated[int, Field(strict=True, ge=0, le=20000)]
    gomory_freq: FreqInt
    zerohalf_freq: FreqInt
    clique_freq: FreqInt


class HeuristicsConfig(_StrictFrozenModel):
    emphasis: Emphasis
    rins_freq: FreqInt
    rens_freq: FreqInt
    feaspump_freq: FreqInt
    shifting_freq: FreqInt
    localbranching_freq: FreqInt


class PresolvingConfig(_StrictFrozenModel):
    emphasis: Emphasis
    maxrounds: Annotated[int, Field(strict=True, ge=-1, le=100)]
    maxrestarts: Annotated[int, Field(strict=True, ge=-1, le=50)]


class BranchingConfig(_StrictFrozenModel):
    scorefunc: Literal["s", "p", "q"]
    scorefac: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    preferbinary: bool


class LpNodeConfig(_StrictFrozenModel):
    pricing: Literal["l", "a", "f", "p", "s", "q", "d"]
    initalgorithm: Literal["s", "p", "d", "b", "c"]
    childsel: Literal["d", "u", "p", "i", "l", "r", "h"]


# Dotted candidate locus -> raw SCIP parameter name (the closed whitelist).
SCIP_PARAM_BY_LOCUS: dict[str, str] = {
    "separating.maxrounds": "separating/maxrounds",
    "separating.maxroundsroot": "separating/maxroundsroot",
    "separating.maxcuts": "separating/maxcuts",
    "separating.maxcutsroot": "separating/maxcutsroot",
    "separating.gomory_freq": "separating/gomory/freq",
    "separating.zerohalf_freq": "separating/zerohalf/freq",
    "separating.clique_freq": "separating/clique/freq",
    "heuristics.rins_freq": "heuristics/rins/freq",
    "heuristics.rens_freq": "heuristics/rens/freq",
    "heuristics.feaspump_freq": "heuristics/feaspump/freq",
    "heuristics.shifting_freq": "heuristics/shifting/freq",
    "heuristics.localbranching_freq": "heuristics/localbranching/freq",
    "presolving.maxrounds": "presolving/maxrounds",
    "presolving.maxrestarts": "presolving/maxrestarts",
    "branching.scorefunc": "branching/scorefunc",
    "branching.scorefac": "branching/scorefac",
    "branching.preferbinary": "branching/preferbinary",
    "lp_node.pricing": "lp/pricing",
    "lp_node.initalgorithm": "lp/initalgorithm",
    "lp_node.childsel": "nodeselection/childsel",
}
_EMPHASIS_KEY_BY_GROUP = {
    "separating": "separating",
    "heuristics": "heuristics",
    "presolving": "presolve",
}


class ScipConfigCandidate(_StrictFrozenModel):
    separating: SeparatingConfig
    heuristics: HeuristicsConfig
    presolving: PresolvingConfig
    branching: BranchingConfig
    lp_node: LpNodeConfig

    def scip_setparams(self) -> dict[str, object]:
        payload = self.model_dump(mode="python")
        return {
            scip_name: payload[locus.split(".")[0]][locus.split(".")[1]]
            for locus, scip_name in SCIP_PARAM_BY_LOCUS.items()
        }

    def scip_emphasis(self) -> dict[str, str]:
        payload = self.model_dump(mode="python")
        return {
            _EMPHASIS_KEY_BY_GROUP[group]: payload[group]["emphasis"]
            for group in _EMPHASIS_KEY_BY_GROUP
        }


_DEFAULTS: dict[str, object] = {
    "separating": {
        "emphasis": "default", "maxrounds": -1, "maxroundsroot": -1,
        "maxcuts": 100, "maxcutsroot": 2000, "gomory_freq": 10,
        "zerohalf_freq": 10, "clique_freq": 0,
    },
    "heuristics": {
        "emphasis": "default", "rins_freq": 15, "rens_freq": 0,
        "feaspump_freq": 20, "shifting_freq": 5, "localbranching_freq": -1,
    },
    "presolving": {"emphasis": "default", "maxrounds": -1, "maxrestarts": -1},
    "branching": {"scorefunc": "p", "scorefac": 0.167, "preferbinary": False},
    "lp_node": {"pricing": "l", "initalgorithm": "s", "childsel": "h"},
}

SEED_DEFAULT: dict[str, object] = json.loads(json.dumps(_DEFAULTS))

# Gate-A probe config "no_cuts_no_heur": emphasis OFF plus the exact curated
# values the OFF meta-setting was observed to induce on this build.
SEED_NO_CUTS_NO_HEUR: dict[str, object] = json.loads(json.dumps(_DEFAULTS))
SEED_NO_CUTS_NO_HEUR["separating"].update(
    {"emphasis": "off", "gomory_freq": -1, "zerohalf_freq": -1, "clique_freq": -1}
)
SEED_NO_CUTS_NO_HEUR["heuristics"].update(
    {"emphasis": "off", "rins_freq": -1, "rens_freq": -1, "feaspump_freq": -1,
     "shifting_freq": -1, "localbranching_freq": -1}
)

# Gate-A probe config "aggressive": emphasis AGGRESSIVE plus its observed
# induced curated values (separating/maxcutsroot 5000, clique 20, ...).
SEED_AGGRESSIVE: dict[str, object] = json.loads(json.dumps(_DEFAULTS))
SEED_AGGRESSIVE["separating"].update(
    {"emphasis": "aggressive", "maxcutsroot": 5000, "clique_freq": 20}
)
SEED_AGGRESSIVE["heuristics"].update(
    {"emphasis": "aggressive", "rins_freq": 8, "rens_freq": 20,
     "feaspump_freq": 10, "shifting_freq": 3, "localbranching_freq": 20}
)


def normalize_candidate(value: object) -> ScipConfigCandidate:
    if isinstance(value, ScipConfigCandidate):
        value = value.model_dump(mode="python")
    return ScipConfigCandidate.model_validate(
        value, strict=True, by_alias=False, by_name=True
    )


def canonical_candidate_json(value: object) -> str:
    return json.dumps(
        normalize_candidate(value).model_dump(mode="python"),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )


def load_instance_split() -> dict[str, object]:
    path = Path(__file__).resolve().parent / "instance_split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    development = split["development"]
    held_out = split["held_out"]
    dev_families = {record["family"] for record in development}
    held_families = {record["family"] for record in held_out}
    if len(development) != 8 or len(held_out) != 4:
        raise ValueError("frozen split must declare 8 development / 4 held-out")
    if dev_families & held_families:
        raise ValueError("development and held-out families must be disjoint")
    return split


def development_panel() -> tuple[tuple[str, str], ...]:
    return tuple(
        (record["file"], record["sha256"])
        for record in load_instance_split()["development"]
    )


def held_out_panel() -> tuple[tuple[str, str], ...]:
    return tuple(
        (record["file"], record["sha256"])
        for record in load_instance_split()["held_out"]
    )


class ScipMiplibProblem:
    candidate_model = ScipConfigCandidate
    example_config = SEED_DEFAULT
    constraints_description = (
        "closed 20-parameter curated SCIP surface plus three emphasis "
        "meta-settings; every field is typed and bounded; unknown parameters "
        "are unrepresentable; the evaluation panel, time limit, seed, and "
        "thread policy are frozen evaluator facts"
    )

    def __init__(
        self,
        settings: ScipEvaluatorSettings,
        *,
        evaluator: object | None = None,
    ) -> None:
        if type(settings) is not ScipEvaluatorSettings:
            raise TypeError("settings must be exact ScipEvaluatorSettings")
        self.settings = settings
        self._evaluator = evaluator

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (
            ObjectiveSpec(OBJECTIVE_PD_INTEGRAL, "min"),
            ObjectiveSpec(OBJECTIVE_GAP_TIME, "min"),
        )

    @property
    def evaluator(self) -> object:
        if self._evaluator is None:
            self._evaluator = ScipSubprocessEvaluator(self.settings)
        return self._evaluator

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> ScipPanelEvaluation:
        candidate = normalize_candidate(config)
        return self.evaluator.evaluate_panel(
            setparams=candidate.scip_setparams(),
            emphasis=candidate.scip_emphasis(),
        )

    def evaluate(self, config: object) -> dict[str, float]:
        return dict(self.evaluate_detailed(config).objective_values)

    @staticmethod
    def candidate_key(config: object) -> str:
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:scip-miplib-candidate-key:v1\x00")
        digest.update(canonical_candidate_json(config).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def render_candidate(config: object) -> str:
        candidate = normalize_candidate(config)
        return (
            f"sep({candidate.separating.emphasis},"
            f"cuts={candidate.separating.maxcuts}/"
            f"{candidate.separating.maxcutsroot}) "
            f"heur({candidate.heuristics.emphasis},"
            f"rins={candidate.heuristics.rins_freq}) "
            f"presolve({candidate.presolving.emphasis}) "
            f"branch({candidate.branching.scorefunc},"
            f"{candidate.branching.scorefac:.3f}) "
            f"lp({candidate.lp_node.pricing},{candidate.lp_node.initalgorithm},"
            f"{candidate.lp_node.childsel})"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "SCIP 10 solver configuration for a fixed MIPLIB 2017 development "
            "panel solved at a frozen per-instance time limit with a pinned "
            "seed and single-threaded search.  The candidate tunes separation "
            "(emphasis, round/cut limits, three separator frequencies), primal "
            "heuristics (emphasis, five heuristic frequencies), presolving "
            "(emphasis, rounds, restarts), branching (score function and "
            "factor, binary preference), and LP/node-selection strategy "
            "characters.  Minimize the panel-mean primal-dual integral and "
            "the panel-mean best-gap-or-time-to-best composite."
        )


def default_settings(
    *,
    panel: tuple[tuple[str, str], ...] | None = None,
    time_limit_s: float = 30.0,
) -> ScipEvaluatorSettings:
    return ScipEvaluatorSettings(
        domain_python=default_domain_python(),
        instances_root=default_instances_root(),
        panel=development_panel() if panel is None else panel,
        time_limit_s=time_limit_s,
    )


__all__ = [
    "REPRESENTATION_ID",
    "SCHEMA_VERSION",
    "SCIP_PARAM_BY_LOCUS",
    "SEED_AGGRESSIVE",
    "SEED_DEFAULT",
    "SEED_NO_CUTS_NO_HEUR",
    "ScipConfigCandidate",
    "ScipMiplibProblem",
    "canonical_candidate_json",
    "default_settings",
    "development_panel",
    "held_out_panel",
    "load_instance_split",
    "normalize_candidate",
]
