#!/usr/bin/env python3
"""Run the frozen 40-child BOiLS local oracle from research artifact 60.

This is a finite, development-only neighborhood audit around parent C.  It
does not call an LLM, use an optimizer, or support adaptive candidate changes.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256  # noqa: E402

from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    SEQUENCE_LENGTH,
    config_sha256,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    BoilsEvaluationObservation,
)
from examples.development import run_agentic_probe as support  # noqa: E402
from examples.development import run_boils_agentic_pilot as v1  # noqa: E402
from examples.development import run_boils_agentic_pilot_v2 as v2  # noqa: E402
from examples.development.corpus_paths import resolve_corpus_path  # noqa: E402


ORACLE_CPUS = (8, 9, 10, 11)
WORKER_COUNT = 4
CHILD_ROUNDS = 10
CHILD_COUNT = 40
TOTAL_EVALUATIONS = 41
PER_CANDIDATE_TIMEOUT_SECONDS = 60
QUALITY_HORIZON_SECONDS = 300
HARD_CLEANUP_DEADLINE_SECONDS = 720
REFERENCE_POINT = (8_028, 71)
EXPECTED_PARENT_OBJECTIVES = (7_944, 69)
MUTATION_INDICES = (1, 7, 12, 18)

EXPECTED_PARENT_BOILS_SHA256 = v2.EXPECTED_PARENT_BOILS_SHA256
EXPECTED_PARENT_TYPED_SHA256 = v2.EXPECTED_PARENT_TYPED_SHA256
EXPECTED_LEGAL_FILE_SHA256 = v2.EXPECTED_LEGAL_FILE_SHA256
EXPECTED_ABC_SHA256 = v2.EXPECTED_ABC_SHA256
EXPECTED_CIRCUIT_SHA256 = v2.EXPECTED_CIRCUIT_SHA256

V2_RUN_DIR = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "boils_agentic_development"
    / "boils_patch_native_pilot_v2_20260713"
)
PREREGISTRATION_PATH = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "60_boils_local_oracle_and_baseline_preregistration.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "73269daa83a44c4141651e33f70bd0346bef599558ff15f735610424507219b7"
)
EXPECTED_V2_FINALIZED_SHA256 = (
    "018ef03e7202bb27669e2d1f4c5aaad6094a285a5db7d008d4d6f5607f91e245"
)
EXPECTED_V2_FILES: dict[str, dict[str, object]] = {
    "evaluations.jsonl": {
        "bytes": 12_453,
        "lines": 5,
        "sha256": "36e5c216dcec9bd7d5d175207015fc6b1082e5826e52ef5b4c300da2de87d4d4",
    },
    "events.jsonl": {
        "bytes": 61_760,
        "lines": 32,
        "sha256": "d1851eb26d64a428dc19ce222e73a784bbd3aa0dc7fc08119ba06cdd2739d1e7",
    },
    "legal_children.json": {
        "bytes": 11_780,
        "sha256": EXPECTED_LEGAL_FILE_SHA256,
    },
    "manifest.json": {
        "bytes": 32_392,
        "sha256": "032880d5b1f17d11e008c580b029ae9a1c48ad8e9c5b521741dcc48686a32eef",
    },
    "queue_outcomes.jsonl": {
        "bytes": 2_104,
        "lines": 5,
        "sha256": "b7c1fb98818eb6340f24a6a3d9fd3fbad40de89b26847a47bff72562f4fee261",
    },
    "runner_source.py": {
        "bytes": 45_959,
        "sha256": "927fb986718351538184cc8b43eaf99b326cb55f4ef73d443367d92c33b0f07f",
    },
    "summary.json": {
        "bytes": 50_868,
        "sha256": "502d24d7eaf9c28733522ab91af55d9ed7dd90b8725d6a4f21532c3561d2d51f",
    },
}

DEFAULT_LOG_ROOT = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "boils_agentic_development"
)


class OracleEvaluator(Protocol):
    """Small synchronous evaluator port used by the fixed scheduler."""

    def evaluate(self, config: object) -> BoilsEvaluation: ...


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    label: str
    frozen_order: int
    sequence: tuple[str, ...]
    boils_configuration_sha256: str
    typed_json_configuration_sha256: str
    index: int | None = None
    legal_ordinal: int | None = None
    replacement: str | None = None

    @property
    def configuration(self) -> dict[str, object]:
        return {"sequence": list(self.sequence)}

    def identity_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "frozen_order": self.frozen_order,
            "index": self.index,
            "legal_ordinal": self.legal_ordinal,
            "replacement": self.replacement,
            "boils_configuration_sha256": self.boils_configuration_sha256,
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
        }


@dataclass(frozen=True, slots=True)
class SealedV2Choice:
    index: int
    replacement: str
    boils_configuration_sha256: str
    typed_json_configuration_sha256: str
    objectives: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "replacement": self.replacement,
            "boils_configuration_sha256": self.boils_configuration_sha256,
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
            "objectives": {
                "total_lut_count": self.objectives[0],
                "total_levels": self.objectives[1],
            },
        }


class SeedGateError(RuntimeError):
    """The required parent preflight failed before any child was submitted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(resolve_corpus_path(path).read_bytes())


def _as_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"{label} is not a finite integer-valued objective")
    return int(numeric)


def _materialize_schedule() -> tuple[CandidateSpec, ...]:
    parent_sequence = tuple(v2.PARENT_C["sequence"])
    parent = CandidateSpec(
        label="parent_c",
        frozen_order=0,
        sequence=parent_sequence,
        boils_configuration_sha256=EXPECTED_PARENT_BOILS_SHA256,
        typed_json_configuration_sha256=EXPECTED_PARENT_TYPED_SHA256,
    )
    indices_document = v2.LEGAL_CHILD_UNIVERSE["indices"]
    children: list[CandidateSpec] = []
    for ordinal in range(CHILD_ROUNDS):
        for position, index in enumerate(MUTATION_INDICES):
            row = indices_document[str(index)]["legal_children"][ordinal]
            replacement = row["replacement"]
            child = copy.deepcopy(v2.PARENT_C)
            child["sequence"][index] = replacement
            spec = CandidateSpec(
                label=f"index_{index}_legal_{ordinal}",
                frozen_order=1 + ordinal * WORKER_COUNT + position,
                sequence=tuple(child["sequence"]),
                boils_configuration_sha256=row["boils_configuration_sha256"],
                typed_json_configuration_sha256=row[
                    "typed_json_configuration_sha256"
                ],
                index=index,
                legal_ordinal=ordinal,
                replacement=replacement,
            )
            if config_sha256(spec.configuration) != spec.boils_configuration_sha256:
                raise RuntimeError("oracle schedule BOiLS identity failed to materialize")
            if (
                typed_json_sha256(freeze_json(spec.configuration))
                != spec.typed_json_configuration_sha256
            ):
                raise RuntimeError("oracle schedule typed identity failed to materialize")
            children.append(spec)
    schedule = (parent, *children)
    if len(schedule) != TOTAL_EVALUATIONS:
        raise RuntimeError("oracle schedule must contain C plus exactly 40 children")
    if tuple(spec.frozen_order for spec in schedule) != tuple(
        range(TOTAL_EVALUATIONS)
    ):
        raise RuntimeError("oracle frozen-order sequence is not contiguous")
    if len({spec.boils_configuration_sha256 for spec in schedule}) != len(schedule):
        raise RuntimeError("oracle schedule contains duplicate physical identities")
    return schedule


FROZEN_SCHEDULE = _materialize_schedule()


class TraceRecorder:
    """Thread-safe global event ordering over scheduler and worker events."""

    def __init__(
        self,
        writer: v1.DurableJsonlWriter,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._writer = writer
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._sequence = 0
        self._started_ns: int | None = None

    def begin(self, started_ns: int) -> None:
        with self._lock:
            if self._started_ns is not None:
                raise RuntimeError("trace recorder was already started")
            self._started_ns = started_ns

    def emit(self, event_type: str, **fields: object) -> dict[str, object]:
        with self._lock:
            if self._started_ns is None:
                raise RuntimeError("trace recorder has not started")
            self._sequence += 1
            record = {
                "schema_version": 1,
                "stream_sequence": self._sequence,
                "recorded_at_utc": _utc_now(),
                "elapsed_ns": max(0, self._clock_ns() - self._started_ns),
                "event_type": event_type,
                **fields,
            }
            self._writer.write(record)
            return record


class EvaluationPublicationRecorder:
    """Durably record evaluator callback time after cleanup and affinity release."""

    def __init__(
        self,
        writer: v1.DurableJsonlWriter,
        trace: TraceRecorder,
        *,
        schedule: Sequence[CandidateSpec] = FROZEN_SCHEDULE,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._writer = writer
        self._trace = trace
        self._clock_ns = clock_ns
        self._specs = {
            spec.boils_configuration_sha256: spec for spec in schedule
        }
        self._lock = threading.Lock()
        self._started_ns: int | None = None
        self._sequence = 0
        self._records: dict[str, dict[str, object]] = {}

    def begin(self, started_ns: int) -> None:
        with self._lock:
            if self._started_ns is not None:
                raise RuntimeError("evaluation publication recorder was already started")
            self._started_ns = started_ns

    def __call__(self, observation: BoilsEvaluationObservation) -> None:
        if type(observation) is BoilsEvaluation:
            status = "succeeded"
        elif type(observation) is BoilsEvaluationFailure:
            status = "candidate_local_failure"
        else:  # pragma: no cover - the evaluator exposes a closed observation union.
            raise TypeError("unknown BOiLS observation type")
        configuration_hash = observation.configuration_sha256
        spec = self._specs.get(configuration_hash)
        if spec is None:
            raise RuntimeError("evaluator published an identity outside the frozen oracle")
        if tuple(observation.sequence) != spec.sequence:
            raise RuntimeError("evaluator observation sequence disagrees with its identity")
        with self._lock:
            if self._started_ns is None:
                raise RuntimeError("evaluation recorder has not started")
            if configuration_hash in self._records:
                raise RuntimeError("evaluator published a duplicate oracle observation")
            self._sequence += 1
            published_elapsed_ns = max(0, self._clock_ns() - self._started_ns)
            record = {
                "schema_version": 1,
                "publication_sequence": self._sequence,
                "recorded_at_utc": _utc_now(),
                "published_elapsed_ns": published_elapsed_ns,
                "status": status,
                "candidate": spec.identity_record(),
                "observation": observation.as_dict(),
            }
            self._writer.write(record)
            self._records[configuration_hash] = copy.deepcopy(record)
        self._trace.emit(
            "evaluation_published",
            label=spec.label,
            frozen_order=spec.frozen_order,
            publication_sequence=record["publication_sequence"],
            published_elapsed_ns=published_elapsed_ns,
            status=status,
            boils_configuration_sha256=configuration_hash,
        )

    def require(self, spec: CandidateSpec) -> dict[str, object]:
        with self._lock:
            record = self._records.get(spec.boils_configuration_sha256)
            if record is None:
                raise RuntimeError("evaluator returned without a durable publication callback")
            return copy.deepcopy(record)

    def records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(
                copy.deepcopy(record)
                for record in sorted(
                    self._records.values(),
                    key=lambda item: int(item["publication_sequence"]),
                )
            )


def _validate_v2_terminal_file(run_dir: Path, name: str, expected: Mapping[str, object]) -> None:
    path = run_dir / name
    if not path.is_file():
        raise RuntimeError(f"sealed v2 terminal file is missing: {name}")
    payload = resolve_corpus_path(path).read_bytes()
    if len(payload) != expected["bytes"] or _sha256_bytes(payload) != expected["sha256"]:
        raise RuntimeError(f"sealed v2 terminal hash/size mismatch: {name}")
    if "lines" in expected and len(payload.splitlines()) != expected["lines"]:
        raise RuntimeError(f"sealed v2 terminal line-count mismatch: {name}")


def load_sealed_v2_choices(
    run_dir: Path = V2_RUN_DIR,
) -> tuple[SealedV2Choice, ...]:
    """Verify the complete sealed v2 terminal index, then load its four choices."""

    finalized_path = run_dir / "finalized.json"
    if not finalized_path.is_file():
        raise RuntimeError("sealed v2 finalized.json is missing")
    if _sha256(finalized_path) != EXPECTED_V2_FINALIZED_SHA256:
        raise RuntimeError("sealed v2 finalized.json hash changed")
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    if finalized.get("status") != "succeeded" or finalized.get("files") != EXPECTED_V2_FILES:
        raise RuntimeError("sealed v2 terminal index changed")
    for name, expected in EXPECTED_V2_FILES.items():
        _validate_v2_terminal_file(run_dir, name, expected)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "succeeded" or summary.get("acceptance_passed") is not True:
        raise RuntimeError("sealed v2 summary did not pass its acceptance gates")
    if summary.get("fixed_mutation_order") != list(MUTATION_INDICES):
        raise RuntimeError("sealed v2 mutation order changed")
    rows = summary.get("generation_one_patch_native")
    if type(rows) is not list or len(rows) != len(MUTATION_INDICES):
        raise RuntimeError("sealed v2 must contain exactly four mutation outcomes")

    choices: list[SealedV2Choice] = []
    for expected_index, row in zip(MUTATION_INDICES, rows, strict=True):
        legal = row.get("legal_child")
        candidate = row.get("candidate")
        if type(legal) is not dict or type(candidate) is not dict:
            raise RuntimeError("sealed v2 choice is missing candidate/legal identity")
        if legal.get("index") != expected_index or candidate.get("valid") is not True:
            raise RuntimeError("sealed v2 choice order or validity changed")
        replacement = legal.get("replacement")
        boils_hash = legal.get("boils_configuration_sha256")
        typed_hash = legal.get("typed_json_configuration_sha256")
        if not all(type(value) is str for value in (replacement, boils_hash, typed_hash)):
            raise RuntimeError("sealed v2 choice identity has an unexpected type")
        matching = [
            spec
            for spec in FROZEN_SCHEDULE
            if spec.index == expected_index and spec.replacement == replacement
        ]
        if len(matching) != 1:
            raise RuntimeError("sealed v2 choice is outside the legal oracle universe")
        spec = matching[0]
        if (
            boils_hash != spec.boils_configuration_sha256
            or typed_hash != spec.typed_json_configuration_sha256
        ):
            raise RuntimeError("sealed v2 choice hashes disagree with the legal universe")
        objectives = candidate.get("objectives")
        if type(objectives) is not dict:
            raise RuntimeError("sealed v2 choice objectives are missing")
        choices.append(
            SealedV2Choice(
                index=expected_index,
                replacement=replacement,
                boils_configuration_sha256=boils_hash,
                typed_json_configuration_sha256=typed_hash,
                objectives=(
                    _as_exact_int(objectives.get("total_lut_count"), "v2 LUT count"),
                    _as_exact_int(objectives.get("total_levels"), "v2 levels"),
                ),
            )
        )
    return tuple(choices)


def _outcome_from_publication(
    spec: CandidateSpec,
    publication: Mapping[str, object],
) -> dict[str, object]:
    observation = publication["observation"]
    base: dict[str, object] = {
        **spec.identity_record(),
        "status": publication["status"],
        "valid": publication["status"] == "succeeded",
        "publication_sequence": publication["publication_sequence"],
        "published_elapsed_ns": publication["published_elapsed_ns"],
        "objectives": None,
        "cec_passed": False,
        "candidate_local_failure_status": None,
        "evaluation_elapsed_s": float(observation["elapsed_s"]),
        "affinity_queue_wait_s": float(observation["affinity_queue_wait_s"]),
        "cpu_affinity": observation.get("cpu_affinity"),
    }
    if publication["status"] == "succeeded":
        circuit_results = observation["circuit_results"]
        cec_passed = bool(circuit_results) and all(
            result["diagnostics"]["status"] == "passed"
            and result["diagnostics"]["equivalent"] is True
            for result in circuit_results
        )
        base["objectives"] = {
            "total_lut_count": _as_exact_int(
                observation["total_lut_count"], "oracle LUT count"
            ),
            "total_levels": _as_exact_int(
                observation["total_levels"], "oracle levels"
            ),
        }
        base["cec_passed"] = cec_passed
    else:
        base["candidate_local_failure_status"] = observation["diagnostics"]["status"]
    return base


def _evaluate_one(
    *,
    evaluator: OracleEvaluator,
    recorder: EvaluationPublicationRecorder,
    spec: CandidateSpec,
) -> dict[str, object]:
    try:
        result = evaluator.evaluate(spec.configuration)
    except AbcEvaluationError:
        publication = recorder.require(spec)
        if publication["status"] != "candidate_local_failure":
            raise RuntimeError("ABC error lacks its candidate-local publication")
        return _outcome_from_publication(spec, publication)
    if type(result) is not BoilsEvaluation:
        raise TypeError("oracle evaluator must return an exact BoilsEvaluation")
    publication = recorder.require(spec)
    if publication["status"] != "succeeded":
        raise RuntimeError("successful evaluator return disagrees with publication")
    return _outcome_from_publication(spec, publication)


def _objective_tuple(outcome: Mapping[str, object]) -> tuple[int, int]:
    objectives = outcome.get("objectives")
    if outcome.get("valid") is not True or type(objectives) is not dict:
        raise ValueError("invalid outcomes do not have admissible objectives")
    return (int(objectives["total_lut_count"]), int(objectives["total_levels"]))


def _dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left != right


def _weakly_dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1]


def _pareto_outcomes(
    outcomes: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    valid = tuple(outcome for outcome in outcomes if outcome.get("valid") is True)
    return tuple(
        sorted(
            (
                outcome
                for outcome in valid
                if not any(
                    other is not outcome
                    and _dominates(_objective_tuple(other), _objective_tuple(outcome))
                    for other in valid
                )
            ),
            key=lambda item: (*_objective_tuple(item), int(item["frozen_order"])),
        )
    )


def _pareto_layers(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    remaining = list(outcome for outcome in outcomes if outcome.get("valid") is True)
    layers: dict[str, int] = {}
    layer = 1
    while remaining:
        front = [
            outcome
            for outcome in remaining
            if not any(
                other is not outcome
                and _dominates(_objective_tuple(other), _objective_tuple(outcome))
                for other in remaining
            )
        ]
        if not front:  # pragma: no cover - finite strict partial order invariant.
            raise RuntimeError("Pareto layer construction made no progress")
        for outcome in front:
            layers[str(outcome["boils_configuration_sha256"])] = layer
            remaining.remove(outcome)
        layer += 1
    return layers


def hypervolume(
    points: Sequence[tuple[int, int]],
    reference: tuple[int, int] = REFERENCE_POINT,
) -> int:
    """Exact two-objective minimization hypervolume inside a fixed rectangle."""

    admissible = sorted(
        set(
            point
            for point in points
            if point[0] < reference[0] and point[1] < reference[1]
        )
    )
    area = 0
    incumbent_y = reference[1]
    for x_value, y_value in admissible:
        if y_value < incumbent_y:
            area += (reference[0] - x_value) * (incumbent_y - y_value)
            incumbent_y = y_value
    return area


def _probability_record(count: int, denominator: int) -> dict[str, object]:
    fraction = Fraction(count, denominator)
    return {
        "count": count,
        "denominator": denominator,
        "fraction": f"{fraction.numerator}/{fraction.denominator}",
        "value": float(fraction),
    }


def _fraction_record(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "fraction": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


def _type7_quantile(sorted_values: Sequence[int], numerator: int, denominator: int) -> Fraction:
    if not sorted_values:
        raise ValueError("quantile input cannot be empty")
    position = Fraction((len(sorted_values) - 1) * numerator, denominator)
    lower = position.numerator // position.denominator
    upper = math.ceil(position)
    weight = position - lower
    return Fraction(sorted_values[lower]) * (1 - weight) + Fraction(
        sorted_values[upper]
    ) * weight


def _outcome_reference(outcome: Mapping[str, object]) -> dict[str, object]:
    return {
        "label": outcome["label"],
        "frozen_order": outcome["frozen_order"],
        "index": outcome["index"],
        "legal_ordinal": outcome["legal_ordinal"],
        "replacement": outcome["replacement"],
        "boils_configuration_sha256": outcome["boils_configuration_sha256"],
        "typed_json_configuration_sha256": outcome[
            "typed_json_configuration_sha256"
        ],
        "objectives": copy.deepcopy(outcome["objectives"]),
    }


def _fixed_order_hv(outcomes: Sequence[Mapping[str, object]]) -> list[dict[str, int]]:
    points: list[tuple[int, int]] = []
    records: list[dict[str, int]] = []
    for k_value, outcome in enumerate(outcomes, start=1):
        if outcome.get("valid") is True:
            points.append(_objective_tuple(outcome))
        records.append({"k": k_value, "hypervolume": hypervolume(points)})
    return records


def _wall_clock_hv_auc(
    outcomes: Sequence[Mapping[str, object]],
    *,
    horizon_seconds: int = QUALITY_HORIZON_SECONDS,
) -> dict[str, object]:
    horizon_ns = horizon_seconds * 1_000_000_000
    publication_order = sorted(
        outcomes,
        key=lambda outcome: (
            int(outcome["published_elapsed_ns"]),
            int(outcome["publication_sequence"]),
        ),
    )
    points: list[tuple[int, int]] = []
    incumbent_hv = 0
    previous_ns = 0
    auc_hv_ns = 0
    trace: list[dict[str, int]] = [
        {"elapsed_ns": 0, "publication_sequence": 0, "hypervolume": 0}
    ]
    publications_within_horizon = 0
    for outcome in publication_order:
        elapsed_ns = int(outcome["published_elapsed_ns"])
        if elapsed_ns > horizon_ns:
            break
        effective_ns = max(previous_ns, elapsed_ns)
        auc_hv_ns += incumbent_hv * (effective_ns - previous_ns)
        if outcome.get("valid") is True:
            points.append(_objective_tuple(outcome))
            incumbent_hv = hypervolume(points)
        previous_ns = effective_ns
        publications_within_horizon += 1
        trace.append(
            {
                "elapsed_ns": effective_ns,
                "publication_sequence": int(outcome["publication_sequence"]),
                "hypervolume": incumbent_hv,
            }
        )
    auc_hv_ns += incumbent_hv * (horizon_ns - previous_ns)
    return {
        "horizon_seconds": horizon_seconds,
        "publication_time_basis": (
            "evaluator observer callback after cleanup and affinity release"
        ),
        "incumbent_carry_forward": True,
        "publications_within_horizon": publications_within_horizon,
        "terminal_hypervolume_at_horizon": incumbent_hv,
        "auc_hv_nanoseconds": auc_hv_ns,
        "auc_hv_seconds": auc_hv_ns / 1_000_000_000,
        "mean_hypervolume": auc_hv_ns / horizon_ns,
        "trace": trace,
    }


def _constrained_best(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    valid = tuple(outcome for outcome in outcomes if outcome.get("valid") is True)
    lut_feasible = tuple(
        outcome for outcome in valid if _objective_tuple(outcome)[1] <= 69
    )
    level_feasible = tuple(
        outcome for outcome in valid if _objective_tuple(outcome)[0] <= 7_944
    )
    best_lut = min((_objective_tuple(item)[0] for item in lut_feasible), default=None)
    best_level = min(
        (_objective_tuple(item)[1] for item in level_feasible), default=None
    )
    return {
        "best_lut_subject_to_levels_le_69": {
            "value": best_lut,
            "ties": [
                _outcome_reference(item)
                for item in lut_feasible
                if _objective_tuple(item)[0] == best_lut
            ],
        },
        "best_levels_subject_to_luts_le_7944": {
            "value": best_level,
            "ties": [
                _outcome_reference(item)
                for item in level_feasible
                if _objective_tuple(item)[1] == best_level
            ],
        },
    }


def _v2_analysis(
    outcomes: Sequence[Mapping[str, object]],
    choices: Sequence[SealedV2Choice],
    oracle_hv: int,
) -> dict[str, object]:
    by_hash = {
        str(outcome["boils_configuration_sha256"]): outcome for outcome in outcomes
    }
    parent = outcomes[0]
    all_valid = tuple(outcome for outcome in outcomes if outcome.get("valid") is True)
    local_oracle_ideal = (
        min(_objective_tuple(outcome)[0] for outcome in all_valid),
        min(_objective_tuple(outcome)[1] for outcome in all_valid),
    )
    sealed_points = [EXPECTED_PARENT_OBJECTIVES, *(choice.objectives for choice in choices)]
    sealed_terminal_hv = hypervolume(sealed_points)
    current_choice_outcomes: list[Mapping[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for choice in choices:
        selected = by_hash.get(choice.boils_configuration_sha256)
        if selected is None:
            raise RuntimeError("sealed v2 choice is missing from oracle outcomes")
        current_choice_outcomes.append(selected)
        same_path = tuple(
            outcome for outcome in outcomes if outcome.get("index") == choice.index
        )
        valid_path = tuple(
            outcome for outcome in same_path if outcome.get("valid") is True
        )
        layers = _pareto_layers(valid_path)
        path_with_parent = (parent, *valid_path)
        layers_with_parent = _pareto_layers(path_with_parent)
        selected_valid = selected.get("valid") is True
        selected_objectives = _objective_tuple(selected) if selected_valid else None
        path_front = _pareto_outcomes(valid_path)
        path_front_with_parent = _pareto_outcomes(path_with_parent)
        path_hvs = {
            str(item["boils_configuration_sha256"]): hypervolume(
                [_objective_tuple(parent), _objective_tuple(item)]
            )
            for item in valid_path
        }
        best_path_hv = max(path_hvs.values(), default=hypervolume([EXPECTED_PARENT_OBJECTIVES]))
        same_path_ideal = (
            min((_objective_tuple(item)[0] for item in valid_path), default=None),
            min((_objective_tuple(item)[1] for item in valid_path), default=None),
        )
        selected_path_hv = (
            path_hvs[choice.boils_configuration_sha256] if selected_valid else None
        )
        path_rows.append(
            {
                "sealed_choice": choice.as_dict(),
                "oracle_reevaluation": _outcome_reference(selected)
                if selected_valid
                else copy.deepcopy(dict(selected)),
                "objective_reproduction_exact": (
                    selected_valid and selected_objectives == choice.objectives
                ),
                "valid": selected_valid,
                "path_pareto_status_among_ten_children": selected in path_front,
                "path_pareto_layer_rank_among_ten_children": (
                    layers.get(choice.boils_configuration_sha256)
                    if selected_valid
                    else None
                ),
                "path_pareto_status_with_parent_c": selected in path_front_with_parent,
                "path_pareto_layer_rank_with_parent_c": (
                    layers_with_parent.get(choice.boils_configuration_sha256)
                    if selected_valid
                    else None
                ),
                "parent_c_strictly_dominates_choice": (
                    _dominates(_objective_tuple(parent), selected_objectives)
                    if selected_objectives is not None
                    else None
                ),
                "path_dominating_child_count": (
                    sum(
                        _dominates(_objective_tuple(item), selected_objectives)
                        for item in valid_path
                        if item is not selected
                    )
                    if selected_valid and selected_objectives is not None
                    else None
                ),
                "path_hv_rank": (
                    1
                    + sum(value > selected_path_hv for value in path_hvs.values())
                    if selected_path_hv is not None
                    else None
                ),
                "path_hv_rank_tie_count": (
                    sum(value == selected_path_hv for value in path_hvs.values())
                    if selected_path_hv is not None
                    else None
                ),
                "single_child_archive_hypervolume": selected_path_hv,
                "best_same_path_single_child_archive_hypervolume": best_path_hv,
                "hypervolume_regret_to_best_same_path_child": (
                    best_path_hv - selected_path_hv
                    if selected_path_hv is not None
                    else None
                ),
                "same_path_ideal_point": {
                    "total_lut_count": same_path_ideal[0],
                    "total_levels": same_path_ideal[1],
                },
                "objective_regret_to_same_path_ideal": (
                    {
                        "total_lut_count": selected_objectives[0] - same_path_ideal[0],
                        "total_levels": selected_objectives[1] - same_path_ideal[1],
                    }
                    if selected_objectives is not None
                    and same_path_ideal[0] is not None
                    and same_path_ideal[1] is not None
                    else None
                ),
                "objective_regret_to_local_oracle_ideal": (
                    {
                        "total_lut_count": (
                            selected_objectives[0] - local_oracle_ideal[0]
                        ),
                        "total_levels": (
                            selected_objectives[1] - local_oracle_ideal[1]
                        ),
                    }
                    if selected_objectives is not None
                    else None
                ),
                "path_pareto_front_among_ten_children": [
                    _outcome_reference(item) for item in path_front
                ],
                "path_pareto_front_with_parent_c": [
                    _outcome_reference(item) for item in path_front_with_parent
                ],
            }
        )
    current_points = [
        _objective_tuple(parent),
        *(
            _objective_tuple(outcome)
            for outcome in current_choice_outcomes
            if outcome.get("valid") is True
        ),
    ]
    current_terminal_hv = hypervolume(current_points)
    return {
        "sealed_terminal_integrity": {
            "run_dir": str(V2_RUN_DIR),
            "finalized_sha256": EXPECTED_V2_FINALIZED_SHA256,
            "summary_sha256": EXPECTED_V2_FILES["summary.json"]["sha256"],
            "all_terminal_files_verified": True,
        },
        "sealed_choices": [choice.as_dict() for choice in choices],
        "sealed_terminal_hypervolume": sealed_terminal_hv,
        "oracle_reevaluated_terminal_hypervolume": current_terminal_hv,
        "all_four_objectives_reproduced_exactly": all(
            row["objective_reproduction_exact"] for row in path_rows
        ),
        "local_oracle_ideal_point": {
            "total_lut_count": local_oracle_ideal[0],
            "total_levels": local_oracle_ideal[1],
        },
        "hypervolume_regret_to_local_oracle": {
            "sealed_v2": oracle_hv - sealed_terminal_hv,
            "oracle_reevaluated_v2": oracle_hv - current_terminal_hv,
        },
        "path_conditional": path_rows,
    }


def _exact_policy_distribution(
    outcomes: Sequence[Mapping[str, object]],
    choices: Sequence[SealedV2Choice],
    v2_analysis: Mapping[str, object],
) -> dict[str, object]:
    parent = outcomes[0]
    if parent.get("valid") is not True:
        raise RuntimeError("policy analysis requires a valid parent C")
    by_path: dict[int, tuple[Mapping[str, object], ...]] = {
        index: tuple(
            sorted(
                (outcome for outcome in outcomes if outcome.get("index") == index),
                key=lambda item: int(item["legal_ordinal"]),
            )
        )
        for index in MUTATION_INDICES
    }
    if any(len(rows) != CHILD_ROUNDS for rows in by_path.values()):
        raise RuntimeError("policy enumeration requires ten fixed outcomes per path")
    v2_current = {
        choice.index: next(
            outcome
            for outcome in by_path[choice.index]
            if outcome["boils_configuration_sha256"]
            == choice.boils_configuration_sha256
        )
        for choice in choices
    }
    denominator = CHILD_ROUNDS ** len(MUTATION_INDICES)
    hv_values: list[int] = []
    support: Counter[int] = Counter()
    strict_dominance_counts = {index: 0 for index in MUTATION_INDICES}
    weak_coverage_counts = {index: 0 for index in MUTATION_INDICES}
    parent_point = _objective_tuple(parent)
    for ordinals in itertools.product(range(CHILD_ROUNDS), repeat=len(MUTATION_INDICES)):
        selected = tuple(
            by_path[index][ordinal]
            for index, ordinal in zip(MUTATION_INDICES, ordinals, strict=True)
        )
        valid_selected = tuple(item for item in selected if item.get("valid") is True)
        points = (parent_point, *(_objective_tuple(item) for item in valid_selected))
        hv_value = hypervolume(points)
        hv_values.append(hv_value)
        support[hv_value] += 1
        for choice in choices:
            target = v2_current[choice.index]
            if target.get("valid") is not True:
                continue
            target_point = _objective_tuple(target)
            strict_dominance_counts[choice.index] += any(
                _dominates(point, target_point) for point in points
            )
            weak_coverage_counts[choice.index] += any(
                _weakly_dominates(point, target_point) for point in points
            )
    if len(hv_values) != denominator:
        raise RuntimeError("exact policy enumeration did not produce 10,000 policies")
    hv_values.sort()
    sealed_threshold = int(v2_analysis["sealed_terminal_hypervolume"])
    reevaluated_threshold = int(v2_analysis["oracle_reevaluated_terminal_hypervolume"])

    def threshold_record(threshold: int) -> dict[str, object]:
        below = sum(value < threshold for value in hv_values)
        equal = support.get(threshold, 0)
        above = denominator - below - equal
        return {
            "threshold_hypervolume": threshold,
            "strictly_below": _probability_record(below, denominator),
            "equal": _probability_record(equal, denominator),
            "matching_or_exceeding": _probability_record(equal + above, denominator),
            "strictly_exceeding": _probability_record(above, denominator),
            "v2_percentile_ecdf_inclusive": {
                **_probability_record(below + equal, denominator),
                "percent": 100.0 * (below + equal) / denominator,
            },
        }

    return {
        "policy_definition": (
            "C plus one uniformly selected legal child per path; invalid children "
            "remain in the denominator and add no admissible point"
        ),
        "dominance_probability_archive_includes_parent_c": True,
        "policy_count": denominator,
        "invalid_child_occurrences_remain_in_denominator": True,
        "hypervolume": {
            "mean": _fraction_record(Fraction(sum(hv_values), denominator)),
            "minimum": hv_values[0],
            "q1_type7": _fraction_record(_type7_quantile(hv_values, 1, 4)),
            "median_type7": _fraction_record(_type7_quantile(hv_values, 1, 2)),
            "q3_type7": _fraction_record(_type7_quantile(hv_values, 3, 4)),
            "maximum": hv_values[-1],
            "complete_support": [
                {
                    "hypervolume": value,
                    **_probability_record(count, denominator),
                }
                for value, count in sorted(support.items())
            ],
        },
        "comparison_to_sealed_v2": threshold_record(sealed_threshold),
        "comparison_to_oracle_reevaluated_v2": threshold_record(
            reevaluated_threshold
        ),
        "probability_policy_archive_strictly_dominates_v2_child": {
            str(index): _probability_record(
                strict_dominance_counts[index], denominator
            )
            for index in MUTATION_INDICES
        },
        "probability_policy_archive_weakly_covers_v2_child": {
            str(index): _probability_record(weak_coverage_counts[index], denominator)
            for index in MUTATION_INDICES
        },
    }


def _resource_summary(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    service_seconds = sum(float(item["evaluation_elapsed_s"]) for item in outcomes)
    affinity_queue_wait_seconds = sum(
        float(item["affinity_queue_wait_s"]) for item in outcomes
    )
    final_publication_ns = max(int(item["published_elapsed_ns"]) for item in outcomes)
    critical_path_seconds = final_publication_ns / 1_000_000_000
    affinities = [json.dumps(item["cpu_affinity"], separators=(",", ":")) for item in outcomes]
    failure_statuses = Counter(
        str(item["candidate_local_failure_status"])
        for item in outcomes
        if item["candidate_local_failure_status"] is not None
    )
    return {
        "physical_evaluations": len(outcomes),
        "valid_evaluations": sum(item["valid"] is True for item in outcomes),
        "candidate_local_invalids": sum(item["valid"] is False for item in outcomes),
        "candidate_local_failure_statuses": dict(sorted(failure_statuses.items())),
        "timeouts": failure_statuses.get("timeout", 0),
        "cec_passed_valid_evaluations": sum(
            item["valid"] is True and item["cec_passed"] is True for item in outcomes
        ),
        "worker_count": WORKER_COUNT,
        "logical_cpus": list(ORACLE_CPUS),
        "evaluator_service_core_seconds": service_seconds,
        "total_affinity_queue_wait_seconds": affinity_queue_wait_seconds,
        "maximum_service_budget_core_seconds": (
            TOTAL_EVALUATIONS * PER_CANDIDATE_TIMEOUT_SECONDS
        ),
        "critical_path_seconds": critical_path_seconds,
        "service_utilization_over_four_core_critical_path": (
            service_seconds / (WORKER_COUNT * critical_path_seconds)
            if critical_path_seconds > 0
            else None
        ),
        "affinity_publication_counts": {
            affinity: affinities.count(affinity) for affinity in sorted(set(affinities))
        },
    }


def analyze_oracle(
    outcomes: Sequence[Mapping[str, object]],
    choices: Sequence[SealedV2Choice],
) -> dict[str, object]:
    """Run every preregistered deterministic analysis over fixed outcomes."""

    if len(outcomes) != TOTAL_EVALUATIONS:
        raise RuntimeError("oracle analysis requires exactly 41 outcomes")
    ordered = tuple(sorted(outcomes, key=lambda item: int(item["frozen_order"])))
    if tuple(int(item["frozen_order"]) for item in ordered) != tuple(
        range(TOTAL_EVALUATIONS)
    ):
        raise RuntimeError("oracle outcomes disagree with the frozen order")
    parent = ordered[0]
    if parent.get("valid") is not True or _objective_tuple(parent) != EXPECTED_PARENT_OBJECTIVES:
        raise SeedGateError("parent C objective gate failed")
    if parent.get("cec_passed") is not True:
        raise SeedGateError("parent C mandatory CEC gate failed")
    if hypervolume([EXPECTED_PARENT_OBJECTIVES]) != 168:
        raise RuntimeError("fixed-reference C hypervolume invariant changed")

    pareto = _pareto_outcomes(ordered)
    valid_points = [
        _objective_tuple(outcome) for outcome in ordered if outcome.get("valid") is True
    ]
    oracle_hv = hypervolume(valid_points)
    v2_result = _v2_analysis(ordered, choices, oracle_hv)
    policy_result = _exact_policy_distribution(ordered, choices, v2_result)
    invalids = [
        copy.deepcopy(dict(outcome))
        for outcome in ordered
        if outcome.get("valid") is not True
    ]
    return {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Exact four-path local oracle around v1-informed parent C; not a "
            "cold-start optimizer, BOiLS reproduction, SOTA, genericity, or "
            "equal-budget wall-clock claim."
        ),
        "schedule": {
            "seed_alone_and_verified_before_children": True,
            "child_rounds": CHILD_ROUNDS,
            "children_per_round": WORKER_COUNT,
            "fixed_index_order": list(MUTATION_INDICES),
            "physical_evaluations": TOTAL_EVALUATIONS,
            "empty_cache": True,
            "retries": 0,
            "replacements_after_outcomes": 0,
        },
        "outcomes_frozen_order": [copy.deepcopy(dict(item)) for item in ordered],
        "publication_order": [
            {
                "publication_sequence": item["publication_sequence"],
                "published_elapsed_ns": item["published_elapsed_ns"],
                "frozen_order": item["frozen_order"],
                "label": item["label"],
                "status": item["status"],
            }
            for item in sorted(
                ordered, key=lambda row: int(row["publication_sequence"])
            )
        ],
        "invalid_outcomes": invalids,
        "pareto_front": [_outcome_reference(item) for item in pareto],
        "constrained_best": _constrained_best(ordered),
        "hypervolume": {
            "objective_direction": "minimize_both",
            "reference_point": {
                "total_lut_count": REFERENCE_POINT[0],
                "total_levels": REFERENCE_POINT[1],
            },
            "parent_c": 168,
            "terminal_local_oracle": oracle_hv,
            "delta_from_parent_c": oracle_hv - 168,
            "hv_at_k_frozen_order": _fixed_order_hv(ordered),
            "wall_clock_auc": _wall_clock_hv_auc(ordered),
        },
        "v2": v2_result,
        "exact_random_policy_distribution": policy_result,
        "resources": _resource_summary(ordered),
    }


def run_oracle(
    *,
    evaluator: OracleEvaluator,
    recorder: EvaluationPublicationRecorder,
    trace: TraceRecorder,
    v2_choices: Sequence[SealedV2Choice] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Execute the nonadaptive seed-plus-ten-round schedule and analyze it."""

    choices = tuple(v2_choices) if v2_choices is not None else load_sealed_v2_choices()
    if len(choices) != len(MUTATION_INDICES):
        raise RuntimeError("oracle requires exactly four sealed v2 choices")
    started_ns = clock_ns()
    recorder.begin(started_ns)
    trace.begin(started_ns)
    trace.emit(
        "oracle_started",
        quality_horizon_ns=QUALITY_HORIZON_SECONDS * 1_000_000_000,
        hard_cleanup_deadline_ns=HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
    )

    parent = FROZEN_SCHEDULE[0]
    trace.emit("candidate_submitted", **parent.identity_record(), round=None)
    try:
        parent_outcome = _evaluate_one(
            evaluator=evaluator,
            recorder=recorder,
            spec=parent,
        )
    except BaseException as exc:
        raise SeedGateError("parent C evaluation failed before child submission") from exc
    if (
        parent_outcome["valid"] is not True
        or parent_outcome["cec_passed"] is not True
        or _objective_tuple(parent_outcome) != EXPECTED_PARENT_OBJECTIVES
    ):
        raise SeedGateError("parent C identity/objective/CEC gate failed")
    trace.emit("seed_gate_passed", **parent.identity_record())

    outcomes: list[dict[str, object]] = [parent_outcome]
    with ThreadPoolExecutor(
        max_workers=WORKER_COUNT,
        thread_name_prefix="boils-local-oracle",
    ) as executor:
        for round_index in range(CHILD_ROUNDS):
            elapsed_ns = clock_ns() - started_ns
            latest_safe_round_start_ns = (
                HARD_CLEANUP_DEADLINE_SECONDS - PER_CANDIDATE_TIMEOUT_SECONDS
            ) * 1_000_000_000
            if elapsed_ns >= latest_safe_round_start_ns:
                raise RuntimeError(
                    "oracle cannot start another fixed round within its hard "
                    "cleanup deadline"
                )
            round_specs = FROZEN_SCHEDULE[
                1 + round_index * WORKER_COUNT : 1 + (round_index + 1) * WORKER_COUNT
            ]
            if tuple(spec.index for spec in round_specs) != MUTATION_INDICES:
                raise RuntimeError("oracle round escaped the fixed index order")
            if any(spec.legal_ordinal != round_index for spec in round_specs):
                raise RuntimeError("oracle round escaped the legal-list ordinal")
            trace.emit(
                "round_started",
                round=round_index,
                frozen_orders=[spec.frozen_order for spec in round_specs],
            )
            futures: list[tuple[CandidateSpec, Future[dict[str, object]]]] = []
            for spec in round_specs:
                trace.emit(
                    "candidate_submitted",
                    **spec.identity_record(),
                    round=round_index,
                )
                futures.append(
                    (
                        spec,
                        executor.submit(
                            _evaluate_one,
                            evaluator=evaluator,
                            recorder=recorder,
                            spec=spec,
                        ),
                    )
                )
            round_outcomes = [future.result() for _, future in futures]
            outcomes.extend(round_outcomes)
            trace.emit(
                "round_completed",
                round=round_index,
                frozen_orders=[spec.frozen_order for spec, _ in futures],
                candidate_local_invalids=sum(
                    outcome["valid"] is not True for outcome in round_outcomes
                ),
            )
    if clock_ns() - started_ns > HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000:
        raise RuntimeError("oracle exceeded its hard cleanup deadline")
    if len(recorder.records()) != TOTAL_EVALUATIONS:
        raise RuntimeError("oracle durable publication count is not exactly 41")
    trace.emit("oracle_fixed_block_completed", physical_evaluations=len(outcomes))
    summary = analyze_oracle(outcomes, choices)
    trace.emit(
        "oracle_analysis_completed",
        terminal_hypervolume=summary["hypervolume"]["terminal_local_oracle"],
        invalid_outcomes=len(summary["invalid_outcomes"]),
    )
    return summary


def _assert_evaluator_provenance(evaluator: BoilsAbcEvaluator) -> None:
    provenance = evaluator.provenance()
    if provenance.get("abc_binary_sha256") != EXPECTED_ABC_SHA256:
        raise RuntimeError("oracle evaluator ABC identity changed")
    circuits = provenance.get("circuits")
    if type(circuits) is not list or len(circuits) != 1:
        raise RuntimeError("oracle evaluator must contain exactly log2")
    if circuits[0].get("name") != "log2" or circuits[0].get("sha256") != EXPECTED_CIRCUIT_SHA256:
        raise RuntimeError("oracle evaluator circuit identity changed")
    if provenance.get("lut_inputs") != 6:
        raise RuntimeError("oracle evaluator LUT mapping changed")
    if provenance.get("per_circuit_timeout_s") != float(PER_CANDIDATE_TIMEOUT_SECONDS):
        raise RuntimeError("oracle evaluator timeout changed")
    if provenance.get("affinity_sets") != [[cpu] for cpu in ORACLE_CPUS]:
        raise RuntimeError("oracle evaluator affinity declaration changed")


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "v1_durable_helpers": Path(v1.__file__).resolve(),
        "v2_legal_helpers": Path(v2.__file__).resolve(),
        "boils_actions": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/actions.py",
        "boils_evaluator": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        "legal_children": v2.LEGAL_CHILD_PATH,
        "sealed_v2_finalized": V2_RUN_DIR / "finalized.json",
        "sealed_v2_summary": V2_RUN_DIR / "summary.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _manifest(
    *,
    run_id: str,
    evaluator: BoilsAbcEvaluator,
    choices: Sequence[SealedV2Choice],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "preregistration": {
            "source": str(PREREGISTRATION_PATH),
            "durable_copy": "preregistration.md",
            "sha256": EXPECTED_PREREGISTRATION_SHA256,
        },
        "claim_boundary": (
            "Exact local four-path oracle around C; not an optimizer, SOTA, "
            "genericity, or equal-budget wall-clock comparison."
        ),
        "frozen_schedule": {
            "parent_alone_first": FROZEN_SCHEDULE[0].identity_record(),
            "required_parent_objectives": {
                "total_lut_count": EXPECTED_PARENT_OBJECTIVES[0],
                "total_levels": EXPECTED_PARENT_OBJECTIVES[1],
            },
            "child_rounds": CHILD_ROUNDS,
            "within_round_index_order": list(MUTATION_INDICES),
            "schedule": [spec.identity_record() for spec in FROZEN_SCHEDULE],
            "worker_count": WORKER_COUNT,
            "logical_cpus": list(ORACLE_CPUS),
            "per_candidate_timeout_seconds": PER_CANDIDATE_TIMEOUT_SECONDS,
            "quality_horizon_seconds": QUALITY_HORIZON_SECONDS,
            "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
            "empty_cache": True,
            "retries": 0,
            "replacement_candidates": 0,
        },
        "analysis": {
            "objective_direction": "minimize_both",
            "reference_point": list(REFERENCE_POINT),
            "parent_c_hypervolume": 168,
            "policy_count": CHILD_ROUNDS ** len(MUTATION_INDICES),
            "policy_quartile_definition": "Hyndman-Fan type 7",
            "invalid_policy_member_rule": "denominator retained; no admissible point",
        },
        "legal_child_universe": {
            "source": str(v2.LEGAL_CHILD_PATH),
            "sha256": EXPECTED_LEGAL_FILE_SHA256,
            "row_count": CHILD_COUNT,
        },
        "sealed_v2": {
            "run_dir": str(V2_RUN_DIR),
            "finalized_sha256": EXPECTED_V2_FINALIZED_SHA256,
            "terminal_files": copy.deepcopy(EXPECTED_V2_FILES),
            "choices": [choice.as_dict() for choice in choices],
        },
        "evaluator_provenance": evaluator.provenance(),
        "source_sha256": _source_hashes(),
        "python_source_snapshot": support._source_snapshot(
            (
                AGENT_EVOLVE_ROOT / "src",
                AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc",
                AGENT_EVOLVE_ROOT / "examples/development",
            )
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "process_affinity_at_start": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "network_or_llm_calls": 0,
        },
    }


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        "preregistration.md",
        "legal_children.json",
        "sealed_v2_finalized.json",
        "sealed_v2_summary.json",
        "events.jsonl",
        "evaluations.jsonl",
        "summary.json",
        "failure.json",
    )
    files: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if not path.exists():
            continue
        payload = resolve_corpus_path(path).read_bytes()
        record: dict[str, object] = {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        if name.endswith(".jsonl"):
            record["lines"] = len(payload.splitlines())
        files[name] = record
    support._write_json(
        run_dir / "finalized.json",
        {
            "schema_version": 1,
            "status": status,
            "completed_at_utc": _utc_now(),
            "files": files,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cpus", default=",".join(str(cpu) for cpu in ORACLE_CPUS))
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument(
        "--per-candidate-timeout-seconds",
        type=int,
        default=PER_CANDIDATE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--quality-horizon-seconds",
        type=int,
        default=QUALITY_HORIZON_SECONDS,
    )
    parser.add_argument(
        "--hard-cleanup-deadline-seconds",
        type=int,
        default=HARD_CLEANUP_DEADLINE_SECONDS,
    )
    return parser


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "cpus": ",".join(str(cpu) for cpu in ORACLE_CPUS),
        "workers": WORKER_COUNT,
        "per_candidate_timeout_seconds": PER_CANDIDATE_TIMEOUT_SECONDS,
        "quality_horizon_seconds": QUALITY_HORIZON_SECONDS,
        "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise SystemExit(
                f"BOiLS local oracle freezes --{name.replace('_', '-')}={value}"
            )


def main() -> None:
    args = _parser().parse_args()
    _assert_frozen_cli(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "boils_local_oracle_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), run_dir / "runner_source.py")
    shutil.copyfile(PREREGISTRATION_PATH, run_dir / "preregistration.md")
    shutil.copyfile(v2.LEGAL_CHILD_PATH, run_dir / "legal_children.json")
    shutil.copyfile(V2_RUN_DIR / "finalized.json", run_dir / "sealed_v2_finalized.json")
    shutil.copyfile(V2_RUN_DIR / "summary.json", run_dir / "sealed_v2_summary.json")
    event_writer = v1.DurableJsonlWriter(run_dir / "events.jsonl")
    evaluation_writer = v1.DurableJsonlWriter(run_dir / "evaluations.jsonl")
    trace = TraceRecorder(event_writer)
    recorder = EvaluationPublicationRecorder(evaluation_writer, trace)
    status = "failed"
    try:
        if _sha256(run_dir / "runner_source.py") != _sha256(Path(__file__).resolve()):
            raise RuntimeError("durable runner copy failed its hash gate")
        if (
            _sha256(run_dir / "preregistration.md")
            != EXPECTED_PREREGISTRATION_SHA256
        ):
            raise RuntimeError("durable preregistration copy failed its hash gate")
        if _sha256(run_dir / "legal_children.json") != EXPECTED_LEGAL_FILE_SHA256:
            raise RuntimeError("durable legal-child copy failed its hash gate")
        if (
            _sha256(run_dir / "sealed_v2_finalized.json")
            != EXPECTED_V2_FINALIZED_SHA256
        ):
            raise RuntimeError("durable sealed-v2 finalization copy failed its hash gate")
        if (
            _sha256(run_dir / "sealed_v2_summary.json")
            != EXPECTED_V2_FILES["summary.json"]["sha256"]
        ):
            raise RuntimeError("durable sealed-v2 summary copy failed its hash gate")
        choices = load_sealed_v2_choices()
        settings = AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("log2",),
            affinity_sets=tuple((cpu,) for cpu in ORACLE_CPUS),
            per_circuit_timeout_s=float(PER_CANDIDATE_TIMEOUT_SECONDS),
        )
        evaluator = BoilsAbcEvaluator(settings, observer=recorder)
        _assert_evaluator_provenance(evaluator)
        support._write_json(
            run_dir / "manifest.json",
            _manifest(run_id=run_id, evaluator=evaluator, choices=choices),
        )
        summary = run_oracle(
            evaluator=evaluator,
            recorder=recorder,
            trace=trace,
            v2_choices=choices,
        )
        support._write_json(run_dir / "summary.json", summary)
        status = "succeeded"
    except BaseException as exc:
        support._write_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "safe_message": (
                    str(exc)
                    if type(exc).__module__.startswith("examples")
                    else "BOiLS local oracle failed; inspect durable local traces"
                ),
            },
        )
        raise
    finally:
        event_writer.close()
        evaluation_writer.close()
        _finalize(run_dir, status)
    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()
