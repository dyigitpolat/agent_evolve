#!/usr/bin/env python3
"""Prepare or execute an evaluator-identical finite-pool qLogNEHVI control.

The optimizer-facing implementation is workload-neutral.  This file owns the
Heat2D codec, legal pool construction, real evaluator, and fixed hypervolume
frame.  The same finite-acquisition policy can therefore be registered later
as an AgentEvolve expert without moving any Heat semantics into the core.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import botorch
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(
    os.environ.get("AGENT_EVOLVE_WORKSPACE_ROOT", str(SOURCE_ROOT.parent))
).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.integrations.botorch import (  # noqa: E402
    BotorchQLogNehviFiniteAcquisition,
)
from agent_evolve.ports.finite_acquisition import (  # noqa: E402
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionRequest,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineHypervolume2DSpec,
    AffineHypervolumeSnapshot2D,
    AffineObjectiveAxis,
)
from examples.benchmarks.heat2d_constructive.candidate import (  # noqa: E402
    CandidateConfig,
    canonical_candidate_json,
    normalize_candidate,
    seed_layouts,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (  # noqa: E402
    LOCUS_GRIDS,
    Heat2DFiniteVariationCatalog,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (  # noqa: E402
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    create_multiobjective_problem,
)
from examples.benchmarks.heat2d_constructive.problem_def import (  # noqa: E402
    Heat2DDirectV3Settings,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/strong_baselines/qlognehvi"
)
PINNED_EVALUATOR_PYTHON = (
    WORKSPACE_ROOT / "agent_evolve/.venv/bin/python"
)
SCHEMA_VERSION = 1
METHOD_ID = "heat2d_finite_pool_qlognehvi_b38"
METHOD_VERSION = 1
TOTAL_EVALUATIONS = 38
INITIAL_SEED_COUNT = 2
BATCH_SIZE = 4
POST_SEED_WAVES = 9
WARMUP_WAVES = 1
DEFAULT_POOL_SIZE = 8192
DEFAULT_MC_SAMPLES = 128
THERMAL_IDEAL = 0.0
THERMAL_REFERENCE = 0.005
MATERIAL_IDEAL = 0.30
MATERIAL_REFERENCE = 0.61
OBJECTIVE_QUANTUM = Decimal("0.000000000001")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()


def _source_identity() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        SOURCE_ROOT / "src/agent_evolve/ports/finite_acquisition.py",
        SOURCE_ROOT / "src/agent_evolve/integrations/botorch/finite_qlognehvi.py",
        SOURCE_ROOT / "examples/benchmarks/heat2d_constructive/candidate.py",
        SOURCE_ROOT
        / "examples/benchmarks/heat2d_constructive/finite_variation_catalog.py",
        SOURCE_ROOT / "examples/benchmarks/heat2d_constructive/multiobjective_v1.py",
        SOURCE_ROOT / "examples/benchmarks/heat2d_constructive/problem_def.py",
    )
    rows = [
        {
            "relative_path": path.relative_to(SOURCE_ROOT).as_posix(),
            "sha256": _file_sha(path),
        }
        for path in paths
    ]
    return {"files": rows, "aggregate_sha256": _sha(rows)}


def _affine_spec() -> AffineHypervolume2DSpec:
    return AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis(
                THERMAL_OBJECTIVE_NAME,
                "min",
                THERMAL_IDEAL,
                THERMAL_REFERENCE,
            ),
            AffineObjectiveAxis(
                MATERIAL_OBJECTIVE_NAME,
                "min",
                MATERIAL_IDEAL,
                MATERIAL_REFERENCE,
            ),
        ),
        reference_provenance=(
            "identical to the frozen AgentEvolve Heat2D B38 campaign: "
            "thermal ideal/reference 0/0.005 and material 0.30/0.61"
        ),
    )


def _acquisition_objectives() -> tuple[FiniteAcquisitionObjective, ...]:
    return tuple(
        sorted(
            (
                FiniteAcquisitionObjective(
                    THERMAL_OBJECTIVE_NAME,
                    "min",
                    THERMAL_IDEAL,
                    THERMAL_REFERENCE,
                ),
                FiniteAcquisitionObjective(
                    MATERIAL_OBJECTIVE_NAME,
                    "min",
                    MATERIAL_IDEAL,
                    MATERIAL_REFERENCE,
                ),
            ),
            key=lambda value: value.metric_id,
        )
    )


def _rounded_objectives(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (
                metric_id,
                float(
                    Decimal(str(raw_value)).quantize(
                        OBJECTIVE_QUANTUM,
                        rounding=ROUND_HALF_EVEN,
                    )
                ),
            )
            for metric_id, raw_value in values.items()
        )
    )


def _set_locus(payload: dict[str, Any], locus: str, value: float) -> None:
    path = locus.split(".")
    cursor = payload
    for name in path[:-1]:
        nested = cursor[name]
        if type(nested) is not dict:
            raise RuntimeError("Heat2D locus traversed a non-object")
        cursor = nested
    cursor[path[-1]] = value


def _get_locus(payload: dict[str, Any], locus: str) -> float:
    value: object = payload
    for name in locus.split("."):
        if type(value) is not dict:
            raise RuntimeError("Heat2D locus traversed a non-object")
        value = value[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Heat2D locus is not numeric")
    return float(value)


_SEEDS = seed_layouts()
_LINEAGE_SIGNATURES = tuple(
    (
        value.trunk.start_x,
        value.trunk.start_y,
        value.trunk.end_x,
        value.trunk.end_y,
    )
    for value in _SEEDS
)


def _lineage(candidate: CandidateConfig) -> int:
    signature = (
        candidate.trunk.start_x,
        candidate.trunk.start_y,
        candidate.trunk.end_x,
        candidate.trunk.end_y,
    )
    try:
        return _LINEAGE_SIGNATURES.index(signature)
    except ValueError as error:
        raise ValueError("candidate escaped the two sealed Heat lineages") from error


def _features(candidate: CandidateConfig) -> tuple[float, ...]:
    payload = candidate.model_dump(mode="python")
    features = [float(_lineage(candidate))]
    for locus, grid, _family in LOCUS_GRIDS:
        raw = _get_locus(payload, locus)
        normalized = (raw - grid[0]) / (grid[-1] - grid[0])
        if not -1.0e-12 <= normalized <= 1.0 + 1.0e-12:
            raise ValueError("Heat2D candidate lies outside the sealed feature frame")
        features.append(min(1.0, max(0.0, float(normalized))))
    return tuple(features)


def _configuration_sha(candidate: CandidateConfig) -> str:
    return hashlib.sha256(
        b"agent-evolve:heat2d-qlognehvi-configuration:v1\x00"
        + canonical_candidate_json(candidate).encode("ascii")
    ).hexdigest()


def _candidate_id(configuration_sha256: str) -> str:
    return f"heat-qnehvi-{configuration_sha256[:32]}"


def _candidate_row(candidate: CandidateConfig) -> FiniteAcquisitionCandidate:
    identity = _configuration_sha(candidate)
    return FiniteAcquisitionCandidate(
        candidate_id=_candidate_id(identity),
        configuration_sha256=identity,
        features=_features(candidate),
    )


def _materialize_sobol(row: torch.Tensor) -> CandidateConfig:
    values = tuple(float(value) for value in row.tolist())
    if len(values) != 1 + len(LOCUS_GRIDS):
        raise ValueError("Sobol row differs from the Heat feature dimension")
    lineage = 0 if values[0] < 0.5 else 1
    payload = _SEEDS[lineage].model_dump(mode="python")
    for coordinate, (locus, grid, _family) in zip(
        values[1:],
        LOCUS_GRIDS,
        strict=True,
    ):
        index = min(len(grid) - 1, int(math.floor(coordinate * len(grid))))
        _set_locus(payload, locus, grid[index])
    return normalize_candidate(payload)


def _global_candidates(*, wave_index: int, seed: int, count: int) -> tuple[CandidateConfig, ...]:
    engine = torch.quasirandom.SobolEngine(
        dimension=1 + len(LOCUS_GRIDS),
        scramble=True,
        seed=seed + 1009 * wave_index,
    )
    selected: dict[str, CandidateConfig] = {}
    draw_count = max(count * 2, 64)
    while len(selected) < count:
        for row in engine.draw(draw_count, dtype=torch.double):
            candidate = _materialize_sobol(row)
            selected.setdefault(_configuration_sha(candidate), candidate)
            if len(selected) == count:
                break
        draw_count = min(draw_count * 2, count * 8)
    return tuple(selected[key] for key in sorted(selected))


def _local_candidates(
    evaluated: tuple["_Observed", ...],
) -> tuple[CandidateConfig, ...]:
    catalog = Heat2DFiniteVariationCatalog()
    selected: dict[str, CandidateConfig] = {}
    for observation in evaluated:
        frozen = freeze_json(observation.candidate.model_dump(mode="python"))
        if type(frozen) is not FrozenJsonObject:
            raise AssertionError("Heat candidate did not freeze to an object")
        for option in catalog.options(frozen):
            candidate = normalize_candidate(thaw_json(option.child_configuration))
            selected.setdefault(_configuration_sha(candidate), candidate)
    return tuple(selected[key] for key in sorted(selected))


def _candidate_pool(
    *,
    evaluated: tuple["_Observed", ...],
    wave_index: int,
    seed: int,
    global_pool_size: int,
) -> tuple[tuple[FiniteAcquisitionCandidate, ...], dict[str, CandidateConfig]]:
    observed_hashes = {value.configuration_sha256 for value in evaluated}
    configurations: dict[str, CandidateConfig] = {}
    for candidate in (
        *_global_candidates(wave_index=wave_index, seed=seed, count=global_pool_size),
        *_local_candidates(evaluated),
    ):
        identity = _configuration_sha(candidate)
        if identity not in observed_hashes:
            configurations.setdefault(identity, candidate)
    by_features: dict[tuple[float, ...], tuple[FiniteAcquisitionCandidate, CandidateConfig]] = {}
    for identity in sorted(configurations):
        candidate = configurations[identity]
        row = _candidate_row(candidate)
        by_features.setdefault(row.features, (row, candidate))
    rows = tuple(value[0] for value in by_features.values())
    materializations = {
        value[0].candidate_id: value[1] for value in by_features.values()
    }
    if len(rows) < BATCH_SIZE:
        raise RuntimeError("Heat qLogNEHVI candidate pool underfilled")
    return rows, materializations


class _Observed:
    __slots__ = ("candidate", "configuration_sha256", "objectives")

    def __init__(
        self,
        candidate: CandidateConfig,
        configuration_sha256: str,
        objectives: tuple[tuple[str, float], ...],
    ) -> None:
        self.candidate = candidate
        self.configuration_sha256 = configuration_sha256
        self.objectives = objectives

    def acquisition_row(self, ordinal: int) -> FiniteAcquisitionObservation:
        return FiniteAcquisitionObservation(
            candidate_id=f"observed-{ordinal:04d}-{self.configuration_sha256[:16]}",
            configuration_sha256=self.configuration_sha256,
            features=_features(self.candidate),
            objectives=self.objectives,
        )

    def to_record(self, ordinal: int) -> dict[str, object]:
        return {
            "ordinal": ordinal,
            "configuration_sha256": self.configuration_sha256,
            "configuration": self.candidate.model_dump(mode="json"),
            "features_hex": [value.hex() for value in _features(self.candidate)],
            "objectives": [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.objectives
            ],
        }


def _space_filling_selection(
    *,
    evaluated: tuple[_Observed, ...],
    candidates: tuple[FiniteAcquisitionCandidate, ...],
) -> tuple[str, ...]:
    anchors = [value.acquisition_row(index).features for index, value in enumerate(evaluated)]
    available = list(candidates)
    selected: list[str] = []
    while len(selected) < BATCH_SIZE:
        winner = max(
            available,
            key=lambda candidate: (
                min(
                    sum((left - right) ** 2 for left, right in zip(candidate.features, anchor, strict=True))
                    for anchor in anchors
                ),
                candidate.candidate_id,
            ),
        )
        selected.append(winner.candidate_id)
        anchors.append(winner.features)
        available.remove(winner)
    return tuple(selected)


def _hypervolume(evaluated: tuple[_Observed, ...]) -> float:
    points = tuple(dict(value.objectives) for value in evaluated)
    return AffineHypervolumeSnapshot2D.create(
        spec=_affine_spec(),
        archive_points=points,
    ).base_hypervolume


def _preregistration(*, outer_seed: int, pool_size: int, mc_samples: int) -> dict[str, object]:
    source = _source_identity()
    policy = BotorchQLogNehviFiniteAcquisition(mc_samples=mc_samples)
    unsigned: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "outer_seed": outer_seed,
        "budget": {
            "initial_seed_count": INITIAL_SEED_COUNT,
            "post_seed_waves": POST_SEED_WAVES,
            "batch_size": BATCH_SIZE,
            "total_real_evaluations": TOTAL_EVALUATIONS,
            "warmup_waves": WARMUP_WAVES,
        },
        "candidate_pool": {
            "global_sobol_count_per_wave": pool_size,
            "local_neighborhood": "all_one_locus_children_of_prior_evaluations",
            "global_materialization": "two_lineage_full_17_locus_grid",
            "deduplication": "configuration_then_feature_identity",
        },
        "acquisition": {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "definition_sha256": policy.definition_sha256,
            "mc_samples": mc_samples,
            "objectives": [value.to_record() for value in _acquisition_objectives()],
        },
        "objective_resolution": {
            "decimal_origin": "0",
            "decimal_quantum": str(OBJECTIVE_QUANTUM),
            "rounding": "nearest_ties_to_even",
        },
        "source": source,
        "environment": {
            "botorch_version": str(botorch.__version__),
            "torch_version": str(torch.__version__),
            "python": sys.version,
            "pinned_evaluator_python": str(PINNED_EVALUATOR_PYTHON),
        },
        "claim_boundary": (
            "evaluator-identical same-domain strong baseline; one seed is developmental, "
            "not a SOTA conclusion"
        ),
    }
    return {**unsigned, "preparation_sha256": _sha(unsigned)}


def prepare(*, run_id: str, outer_seed: int, pool_size: int, mc_samples: int) -> Path:
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    preflight_problem = create_multiobjective_problem(
        Heat2DDirectV3Settings(
            output_root=run_dir / "provider_free_preflight",
            resolution=1001,
            cpu_set="8",
            timeout_s=180.0,
            python_executable=PINNED_EVALUATOR_PYTHON,
            required_numpy_version="2.3.5",
            external_concurrency=1,
        )
    )
    evaluator_preflight = preflight_problem.preflight()
    preregistration = _preregistration(
        outer_seed=outer_seed,
        pool_size=pool_size,
        mc_samples=mc_samples,
    )
    _write_json(run_dir / "preregistration.json", preregistration)
    _write_json(run_dir / "evaluator_preflight.json", evaluator_preflight)
    _write_json(
        run_dir / "status.json",
        {
            "status": "provider_and_pde_free_prepared",
            "provider_calls": 0,
            "pde_evaluations": 0,
            "evaluator_preflight_sha256": _sha(evaluator_preflight),
            "preparation_sha256": preregistration["preparation_sha256"],
        },
    )
    return run_dir


def _load_preregistration(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("preregistration must contain one object")
    recorded = value.pop("preparation_sha256", None)
    if recorded != _sha(value):
        raise ValueError("preregistration authentication failed")
    value["preparation_sha256"] = recorded
    if value.get("source") != _source_identity():
        raise ValueError("source changed after baseline preparation")
    return value


def live(*, run_id: str, preregistration_path: Path) -> Path:
    preregistration = _load_preregistration(preregistration_path)
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "preregistration_identity.json", preregistration)
    pool_size = int(preregistration["candidate_pool"]["global_sobol_count_per_wave"])
    mc_samples = int(preregistration["acquisition"]["mc_samples"])
    outer_seed = int(preregistration["outer_seed"])
    settings = Heat2DDirectV3Settings(
        output_root=run_dir / "pde",
        resolution=1001,
        cpu_set="8",
        timeout_s=180.0,
        python_executable=PINNED_EVALUATOR_PYTHON,
        required_numpy_version="2.3.5",
        external_concurrency=1,
    )
    problem = create_multiobjective_problem(settings)
    _write_json(run_dir / "evaluator_preflight.json", problem.preflight())
    observation_log = run_dir / "observations.jsonl"
    decision_log = run_dir / "decisions.jsonl"
    trajectory_log = run_dir / "trajectory.jsonl"
    evaluated: list[_Observed] = []

    def evaluate(candidate: CandidateConfig, *, stage: str, wave_index: int) -> None:
        started = time.monotonic()
        detailed = problem.evaluate_detailed(candidate)
        elapsed = time.monotonic() - started
        objectives = _rounded_objectives(detailed.objective_values)
        observation = _Observed(candidate, _configuration_sha(candidate), objectives)
        evaluated.append(observation)
        _append_jsonl(
            observation_log,
            {
                **observation.to_record(len(evaluated)),
                "stage": stage,
                "wave_index": wave_index,
                "raw_objectives": detailed.objective_values,
                "adapter_elapsed_s_hex": float(elapsed).hex(),
                "evaluator_elapsed_s_hex": detailed.direct_v3.evaluator_elapsed_s.hex(),
                "manifest_relative_path": detailed.direct_v3.output_dir.relative_to(
                    run_dir
                ).as_posix(),
            },
        )
        _append_jsonl(
            trajectory_log,
            {
                "evaluation_count": len(evaluated),
                "wave_index": wave_index,
                "normalized_hypervolume_hex": _hypervolume(tuple(evaluated)).hex(),
            },
        )

    for seed in _SEEDS:
        evaluate(seed, stage="initial_seed", wave_index=0)

    acquisition = BotorchQLogNehviFiniteAcquisition(mc_samples=mc_samples)
    for wave_index in range(1, POST_SEED_WAVES + 1):
        candidates, materializations = _candidate_pool(
            evaluated=tuple(evaluated),
            wave_index=wave_index,
            seed=outer_seed,
            global_pool_size=pool_size,
        )
        if wave_index <= WARMUP_WAVES:
            selected_ids = _space_filling_selection(
                evaluated=tuple(evaluated),
                candidates=candidates,
            )
            decision_record: dict[str, object] = {
                "schema_version": 1,
                "wave_index": wave_index,
                "policy_id": "deterministic_maximin_sobol_warmup",
                "candidate_pool_size": len(candidates),
                "selected_candidate_ids": list(selected_ids),
            }
        else:
            request = FiniteAcquisitionRequest(
                campaign_scope_sha256=str(preregistration["preparation_sha256"]),
                cutoff_index=wave_index,
                batch_size=BATCH_SIZE,
                seed=outer_seed + wave_index,
                objectives=_acquisition_objectives(),
                observations=tuple(
                    value.acquisition_row(index)
                    for index, value in enumerate(evaluated)
                ),
                candidates=candidates,
            )
            decision = acquisition.select(request)
            selected_ids = tuple(value.candidate_id for value in decision.selected)
            decision_record = {
                "schema_version": 1,
                "wave_index": wave_index,
                "candidate_pool_size": len(candidates),
                "request": request.to_record(),
                "decision": decision.to_record(),
            }
        _append_jsonl(decision_log, decision_record)
        for candidate_id in selected_ids:
            evaluate(
                materializations[candidate_id],
                stage=("space_filling_warmup" if wave_index <= WARMUP_WAVES else "qlognehvi"),
                wave_index=wave_index,
            )
    if len(evaluated) != TOTAL_EVALUATIONS:
        raise RuntimeError("qLogNEHVI baseline did not consume its exact B38 budget")
    final_hv = _hypervolume(tuple(evaluated))
    summary = {
        "schema_version": 1,
        "status": "completed",
        "method_id": METHOD_ID,
        "preparation_sha256": preregistration["preparation_sha256"],
        "real_evaluations": len(evaluated),
        "final_normalized_hypervolume_hex": final_hv.hex(),
        "final_normalized_hypervolume": final_hv,
        "provider_calls": 0,
        "decision_waves": POST_SEED_WAVES,
        "qlognehvi_waves": POST_SEED_WAVES - WARMUP_WAVES,
    }
    _write_json(run_dir / "summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--outer-seed", type=int, default=20260723)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    args = parser.parse_args()
    if args.pool_size < BATCH_SIZE:
        parser.error("--pool-size must be at least the batch size")
    if args.mc_samples < 16:
        parser.error("--mc-samples must be at least 16")
    if args.mode == "prepare":
        if args.prereg is not None:
            parser.error("prepare does not accept --prereg")
        output = prepare(
            run_id=args.run_id,
            outer_seed=args.outer_seed,
            pool_size=args.pool_size,
            mc_samples=args.mc_samples,
        )
    else:
        if args.prereg is None:
            parser.error("live requires --prereg")
        output = live(run_id=args.run_id, preregistration_path=args.prereg)
    print(output)


if __name__ == "__main__":
    main()
