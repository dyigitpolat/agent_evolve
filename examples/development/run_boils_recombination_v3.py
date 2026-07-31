#!/usr/bin/env python3
"""Run the frozen BOiLS three-edit deterministic recombination block.

The LLM predicts the three already-sealed engine children but cannot create,
select, reorder, cancel, or replace them.  This is a post-hoc development
mechanism test, not optimizer, memory, genericity, SOTA, or wall-clock evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # noqa: E402

from agent_evolve.domain.ids import CandidateId, LLMCallId  # noqa: E402
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, ReplaceScalar  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    freeze_json,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    create_production_queued_runner,
)
from agent_evolve.policies.variation.typed_patch import (  # noqa: E402
    apply_patch,
    derive_patch,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)

from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    SEQUENCE_LENGTH,
    config_sha256,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
)
from examples.development import run_agentic_probe as support  # noqa: E402
from examples.development import run_boils_agentic_pilot as v1  # noqa: E402
from examples.development import run_boils_agentic_pilot_v2 as v2  # noqa: E402
from examples.development import run_boils_local_oracle as oracle  # noqa: E402
from examples.development.corpus_paths import corpus_path_or_none, resolve_corpus_path  # noqa: E402


MODEL = "deepseek/deepseek-v4-pro"
RESOLVED_PROVIDER = "Together"
PROVIDER_ORDER = ("together",)
MAX_OUTPUT_TOKENS = 800
TEMPERATURE = 0.2
QUEUE_MAX_IN_FLIGHT = 1
QUEUE_MAX_PENDING = 0
QUEUE_MAX_ATTEMPTS = 2
QUEUE_ATTEMPT_TIMEOUT_SECONDS = 60
QUEUE_BASE_BACKOFF_SECONDS = 1
QUEUE_MAX_BACKOFF_SECONDS = 8
MAX_SUCCESSFUL_RESPONSE_COST_USD = Decimal("0.01")

PHYSICAL_CPUS = (8, 9, 10, 11)
CHILD_WORKERS = 3
PER_CANDIDATE_TIMEOUT_SECONDS = 60
QUALITY_HORIZON_SECONDS = 120
HARD_CLEANUP_DEADLINE_SECONDS = 300
REFERENCE_POINT = (8_028, 71)
EXPECTED_PHYSICAL_EVALUATIONS = 4
NEW_ARM_ORDER = ("AD", "BD", "ABD")
ALL_ARM_ORDER = ("C", "A", "B", "D", "AB", "AD", "BD", "ABD")

EXPECTED_ABC_SHA256 = v2.EXPECTED_ABC_SHA256
EXPECTED_CIRCUIT_SHA256 = v2.EXPECTED_CIRCUIT_SHA256
EXPECTED_SEED_OBJECTIVES = (7_944, 69)
EXPECTED_PREREGISTRATION_SHA256 = (
    "1288a4ada394c05c91a268d198695df1c10975255507df8e209c28ca30da2529"
)
EXPECTED_CORRECTION_SHA256 = (
    "dc14f1fd95c154bc2c729c9df2b930c8bc8961e8e1efe60be57e5b2af4c3075b"
)

ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
PREREGISTRATION_PATH = (
    ARTIFACT_ROOT / "66_boils_three_edit_recombination_preregistration.md"
)
CORRECTION_PATH = (
    ARTIFACT_ROOT
    / "67_boils_shadow_and_recombination_preregistration_corrections.md"
)
DEVELOPMENT_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "boils_agentic_development"
V1_RUN_DIR = DEVELOPMENT_LOG_ROOT / "boils_agentic_pilot_v1_20260713"
V2_RUN_DIR = DEVELOPMENT_LOG_ROOT / "boils_patch_native_pilot_v2_20260713"
ORACLE_RUN_DIR = DEVELOPMENT_LOG_ROOT / "boils_local_oracle_v1_20260714"
DEFAULT_LOG_ROOT = DEVELOPMENT_LOG_ROOT

EVIDENCE_SOURCES: dict[str, tuple[Path, str]] = {
    "preregistration": (PREREGISTRATION_PATH, EXPECTED_PREREGISTRATION_SHA256),
    "protocol_correction": (CORRECTION_PATH, EXPECTED_CORRECTION_SHA256),
    "v1_finalized": (
        V1_RUN_DIR / "finalized.json",
        "e0d9cbbcba0e87f16e86d255c82ec71201161bb87875a9c51f8cb0d0affab35f",
    ),
    "v1_summary": (
        V1_RUN_DIR / "summary.json",
        "bd7b3d2b56c809821e8185845371ea0d1b29c0d26b5fa8682b156e4fde5303b4",
    ),
    "v2_finalized": (
        V2_RUN_DIR / "finalized.json",
        "018ef03e7202bb27669e2d1f4c5aaad6094a285a5db7d008d4d6f5607f91e245",
    ),
    "v2_summary": (
        V2_RUN_DIR / "summary.json",
        "502d24d7eaf9c28733522ab91af55d9ed7dd90b8725d6a4f21532c3561d2d51f",
    ),
    "oracle_finalized": (
        ORACLE_RUN_DIR / "finalized.json",
        "627db6494ed38133ebb8478b0954216d741b28340342b30c46de0aa331f6be38",
    ),
    "oracle_summary": (
        ORACLE_RUN_DIR / "summary.json",
        "63e144b597f662b606ea4272e9816a3a1ff8e5c7962685d6751e2d9dcc040b0d",
    ),
}

PARENT_C = copy.deepcopy(v2.PARENT_C)
BRANCH_EDITS: dict[str, tuple[int, str]] = {
    "A": (1, "rewrite_z"),
    "B": (12, "refactor_z"),
    "D": (7, "resub"),
}
ARM_BRANCHES: dict[str, tuple[str, ...]] = {
    "C": (),
    "A": ("A",),
    "B": ("B",),
    "D": ("D",),
    "AB": ("A", "B"),
    "AD": ("A", "D"),
    "BD": ("B", "D"),
    "ABD": ("A", "B", "D"),
}
EXPECTED_IDENTITIES: dict[str, tuple[str, str]] = {
    "C": (
        "e954b02443e92dbed5cc7aa21b8d452531400017d602bf5dcdc938fb84e5237e",
        "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a",
    ),
    "A": (
        "bd71137843f397e063798cb94ca6ec4cb34e565ce9c2ad0c7ddba5f592016372",
        "c9564429b5d6980aaccedd9c665b8de7e82065b32bde6c677fa5ed863a1ebfca",
    ),
    "B": (
        "5fb1adfa2cb0aeeacbfefa1a9f5aace3a838ec01f350934c248321e066fb3378",
        "4b34befa0309aed3f4d929773cc03a90f89765e418a0d6b2b372da045f148c62",
    ),
    "D": (
        "249cb63b8d1487a355a1a6f00317e3fdc644bebf41ce287766fe212f272d8bc5",
        "066e763e7d591937a6828bd6edbf1c025e4c888503aaff3a4cb4d3beb77a6ab9",
    ),
    "AB": (
        "df54c93433c38c2b2d839f9947631459d73a4995d27f617c5d7729bd45ce1609",
        "3858b19c033cee8f5583c88a86429c4269a722324640e9d6b608395b26b9370e",
    ),
    "AD": (
        "8aed74a01b6ba7725996ec78e60e9b50d39447a6a3b511c847bf04d52d2d2e04",
        "1df2f132c72655f2a96488e5d63ce3887bd5453ce9b519fa2ae6fdbbb5b618f6",
    ),
    "BD": (
        "d8ba5385b476f93e672bb6095a0d02fbd230bf12c0dc65371d8acc7115ddde2d",
        "9d267611524da973695026ed99620fe0d8c7a9186b9b2e81b8afd75fa43afea1",
    ),
    "ABD": (
        "44765f69d8242a20622c3c502909550ea4e853b90c14a9d5c286b696b4ae85ab",
        "ea74efecec5dfa3eeaf25bcc7c757da3da7c98a9b315185fadc9fa1d6d14c859",
    ),
}
KNOWN_OBJECTIVES: dict[str, tuple[int, int]] = {
    "C": (7_944, 69),
    "A": (7_935, 69),
    "B": (7_931, 69),
    "D": (7_925, 69),
    "AB": (7_918, 70),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _path(index: int) -> JsonPath:
    return JsonPath((ObjectKey("sequence"), ArrayIndex(index)))


def _path_text(index: int) -> str:
    return f"$.sequence[{index}]"


@dataclass(frozen=True, slots=True)
class CubeArm:
    label: str
    frozen_order: int
    branches: tuple[str, ...]
    sequence: tuple[str, ...]
    boils_configuration_sha256: str
    typed_json_configuration_sha256: str
    patch_hash: str | None
    patch_record: Mapping[str, object] | None
    known_objectives: tuple[int, int] | None

    @property
    def configuration(self) -> dict[str, object]:
        return {"sequence": list(self.sequence)}

    def identity_record(self) -> dict[str, object]:
        return {
            "arm": self.label,
            "frozen_order": self.frozen_order,
            "branches": list(self.branches),
            "boils_configuration_sha256": self.boils_configuration_sha256,
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
            "patch_hash": self.patch_hash,
        }


def materialize_cube(
    expected_identities: Mapping[str, tuple[str, str]] = EXPECTED_IDENTITIES,
) -> tuple[CubeArm, ...]:
    """Engine-derive and exact-replay every ancestor-relative cube patch."""

    if tuple(expected_identities) != ALL_ARM_ORDER:
        raise RuntimeError("frozen cube identity order changed")
    if len(PARENT_C.get("sequence", ())) != SEQUENCE_LENGTH:
        raise RuntimeError("frozen parent C escaped the BOiLS sequence schema")
    frozen_parent = freeze_json(PARENT_C)
    parent_id = CandidateId("candidate_boils_recombination_v3_C")
    arms: list[CubeArm] = []
    for frozen_order, label in enumerate(ALL_ARM_ORDER):
        target = copy.deepcopy(PARENT_C)
        for branch in ARM_BRANCHES[label]:
            index, replacement = BRANCH_EDITS[branch]
            target["sequence"][index] = replacement
        boils_hash = config_sha256(target)
        typed_hash = typed_json_sha256(freeze_json(target))
        if (boils_hash, typed_hash) != expected_identities[label]:
            raise RuntimeError(f"cube arm {label} failed its frozen identity gate")

        patch_hash: str | None = None
        patch_record: dict[str, object] | None = None
        if label == "C":
            identity_patch = derive_patch(
                frozen_parent,
                freeze_json(target),
                base_candidate_id=parent_id,
                target_candidate_id=CandidateId(
                    "candidate_boils_recombination_v3_C_reproduction"
                ),
            )
            if identity_patch.operations:
                raise RuntimeError("cube arm C did not derive an empty identity patch")
            replayed = apply_patch(frozen_parent, identity_patch)
            if not typed_json_equal(replayed, frozen_parent):
                raise RuntimeError("cube arm C identity patch failed exact replay")
            patch_hash = identity_patch.patch_hash
            patch_record = {
                "schema_version": identity_patch.schema_version,
                "base_candidate_id": identity_patch.base_candidate_id.value,
                "target_candidate_id": identity_patch.target_candidate_id.value,
                "base_hash": identity_patch.base_hash,
                "target_hash": identity_patch.target_hash,
                "patch_hash": patch_hash,
                "operation_count": 0,
                "operations": [],
                "materialization_kind": "identity_reproduction",
                "attribution_provenance": "system_derived",
                "replay_verified": True,
            }
        else:
            target_id = CandidateId(f"candidate_boils_recombination_v3_{label}")
            patch = derive_patch(
                frozen_parent,
                freeze_json(target),
                base_candidate_id=parent_id,
                target_candidate_id=target_id,
            )
            expected_indices = tuple(
                sorted(BRANCH_EDITS[branch][0] for branch in ARM_BRANCHES[label])
            )
            if len(patch.operations) != len(expected_indices):
                raise RuntimeError(f"cube arm {label} did not derive one operation per edit")
            operation_indices: list[int] = []
            operations: list[dict[str, object]] = []
            branch_by_index = {
                BRANCH_EDITS[branch][0]: branch for branch in ARM_BRANCHES[label]
            }
            for operation in patch.operations:
                if type(operation) is not ReplaceScalar:
                    raise RuntimeError(f"cube arm {label} derived a non-scalar operation")
                if (
                    len(operation.path.segments) != 2
                    or type(operation.path.segments[0]) is not ObjectKey
                    or operation.path.segments[0].value != "sequence"
                    or type(operation.path.segments[1]) is not ArrayIndex
                ):
                    raise RuntimeError(f"cube arm {label} derived an unexpected path")
                index = operation.path.segments[1].value
                operation_indices.append(index)
                branch = branch_by_index.get(index)
                if branch is None:
                    raise RuntimeError(f"cube arm {label} attributed an unsealed edit")
                expected_replacement = BRANCH_EDITS[branch][1]
                expected_old = PARENT_C["sequence"][index]
                if thaw_json(operation.old_value) != expected_old or thaw_json(
                    operation.new_value
                ) != expected_replacement:
                    raise RuntimeError(f"cube arm {label} operation values changed")
                operations.append(
                    {
                        "operation_kind": "replace_scalar",
                        "path": _path_text(index),
                        "old_value": expected_old,
                        "new_value": expected_replacement,
                        "old_value_hash": typed_json_sha256(operation.old_value),
                        "new_value_hash": typed_json_sha256(operation.new_value),
                        "innovation_source_arm": branch,
                        "attribution_provenance": "system_derived_from_frozen_path",
                    }
                )
            if tuple(operation_indices) != expected_indices:
                raise RuntimeError(f"cube arm {label} patch order or paths changed")
            replayed = apply_patch(frozen_parent, patch)
            if not typed_json_equal(replayed, freeze_json(target)):
                raise RuntimeError(f"cube arm {label} patch failed exact replay")
            if typed_json_sha256(replayed) != typed_hash:
                raise RuntimeError(f"cube arm {label} replay hash changed")
            patch_hash = patch.patch_hash
            patch_record = {
                "schema_version": patch.schema_version,
                "base_candidate_id": patch.base_candidate_id.value,
                "target_candidate_id": patch.target_candidate_id.value,
                "base_hash": patch.base_hash,
                "target_hash": patch.target_hash,
                "patch_hash": patch_hash,
                "operation_count": len(operations),
                "operations": operations,
                "materialization_kind": "ancestor_relative_patch_union",
                "attribution_provenance": "system_derived",
                "replay_verified": True,
            }
        arms.append(
            CubeArm(
                label=label,
                frozen_order=frozen_order,
                branches=ARM_BRANCHES[label],
                sequence=tuple(target["sequence"]),
                boils_configuration_sha256=boils_hash,
                typed_json_configuration_sha256=typed_hash,
                patch_hash=patch_hash,
                patch_record=patch_record,
                known_objectives=KNOWN_OBJECTIVES.get(label),
            )
        )
    if len({arm.boils_configuration_sha256 for arm in arms}) != len(arms):
        raise RuntimeError("cube contains duplicate physical identities")
    return tuple(arms)


CUBE = materialize_cube()
CUBE_BY_LABEL = {arm.label: arm for arm in CUBE}
PHYSICAL_SCHEDULE = tuple(
    oracle.CandidateSpec(
        label=label,
        frozen_order=order,
        sequence=CUBE_BY_LABEL[label].sequence,
        boils_configuration_sha256=CUBE_BY_LABEL[label].boils_configuration_sha256,
        typed_json_configuration_sha256=CUBE_BY_LABEL[label].typed_json_configuration_sha256,
    )
    for order, label in enumerate(("C", *NEW_ARM_ORDER))
)


def _recursive_records(value: object):
    if type(value) is dict:
        yield value
        for child in value.values():
            yield from _recursive_records(child)
    elif type(value) is list:
        for child in value:
            yield from _recursive_records(child)


def _contains_result(document: object, arm: str) -> bool:
    expected_hash = EXPECTED_IDENTITIES[arm][0]
    expected_objectives = KNOWN_OBJECTIVES[arm]
    for row in _recursive_records(document):
        observed_hash = row.get(
            "boils_configuration_sha256",
            row.get("boils_schema_configuration_sha256"),
        )
        objectives = row.get("objectives")
        if observed_hash != expected_hash or type(objectives) is not dict:
            continue
        try:
            observed = (
                oracle._as_exact_int(objectives.get("total_lut_count"), "sealed LUT"),
                oracle._as_exact_int(objectives.get("total_levels"), "sealed levels"),
            )
        except (TypeError, ValueError):
            continue
        if observed == expected_objectives:
            return True
    return False


def verify_evidence_bundle(
    sources: Mapping[str, tuple[Path, str]] = EVIDENCE_SOURCES,
) -> dict[str, dict[str, object]]:
    """Hash-bind all sources while parsing only pre-oracle v1/v2 evidence.

    Oracle files are deliberately treated as opaque bytes here.  Their JSON is
    parsed only by :func:`verify_deferred_oracle_evidence`, after the prediction
    response has been durably published.
    """

    if set(sources) != set(EVIDENCE_SOURCES):
        raise RuntimeError("sealed evidence source set changed")
    records: dict[str, dict[str, object]] = {}
    parsed: dict[str, object] = {}
    for name in EVIDENCE_SOURCES:
        path, expected_hash = sources[name]
        if not corpus_path_or_none(path) is not None:
            raise RuntimeError(f"sealed evidence source is missing: {name}")
        payload = resolve_corpus_path(path).read_bytes()
        observed_hash = _sha256_bytes(payload)
        if observed_hash != expected_hash:
            raise RuntimeError(f"sealed evidence source hash changed: {name}")
        records[name] = {
            "source": str(path),
            "sha256": observed_hash,
            "bytes": len(payload),
        }
        if name in {
            "v1_finalized",
            "v1_summary",
            "v2_finalized",
            "v2_summary",
        }:
            parsed[name] = json.loads(payload)
    for name in ("v1_finalized", "v2_finalized"):
        document = parsed[name]
        if type(document) is not dict or document.get("status") != "succeeded":
            raise RuntimeError(f"sealed terminal index is not successful: {name}")
    expected_membership = {
        "v1_summary": ("C", "A", "B", "AB"),
        "v2_summary": ("C", "D"),
    }
    for name, arms in expected_membership.items():
        if not all(_contains_result(parsed[name], arm) for arm in arms):
            raise RuntimeError(f"sealed source facts changed or are incomplete: {name}")
    return records


def _load_json_path(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_deferred_oracle_evidence(
    sources: Mapping[str, tuple[Path, str]] = EVIDENCE_SOURCES,
    *,
    json_loader: Callable[[Path], object] | None = None,
) -> dict[str, object]:
    """Parse and verify sealed oracle JSON at the post-prediction boundary."""

    loader = _load_json_path if json_loader is None else json_loader
    if not callable(loader):
        raise TypeError("json_loader must be callable")
    records: dict[str, str] = {}
    for name in ("oracle_finalized", "oracle_summary"):
        if name not in sources:
            raise RuntimeError(f"deferred oracle source is missing: {name}")
        path, expected_hash = sources[name]
        if not corpus_path_or_none(path) is not None or support._sha256(path) != expected_hash:
            raise RuntimeError(f"deferred oracle source hash changed: {name}")
        records[name] = expected_hash
    finalized = loader(sources["oracle_finalized"][0])
    summary = loader(sources["oracle_summary"][0])
    if type(finalized) is not dict or finalized.get("status") != "succeeded":
        raise RuntimeError("deferred oracle terminal index is not successful")
    if type(summary) is not dict or not all(
        _contains_result(summary, arm) for arm in ("C", "A", "B", "D")
    ):
        raise RuntimeError("deferred oracle source facts changed or are incomplete")
    hypervolume = summary.get("hypervolume")
    front = summary.get("pareto_front")
    if (
        type(hypervolume) is not dict
        or hypervolume.get("terminal_local_oracle") != 700
        or type(front) is not list
        or len(front) != 5
    ):
        raise RuntimeError("deferred full-oracle sensitivity checkpoint changed")
    sensitivity_front = copy.deepcopy(front)
    sensitivity_points: list[tuple[int, int]] = []
    for row in sensitivity_front:
        if type(row) is not dict or type(row.get("objectives")) is not dict:
            raise RuntimeError("deferred full-oracle front has an invalid row")
        objectives = row["objectives"]
        sensitivity_points.append(
            (
                oracle._as_exact_int(
                    objectives.get("total_lut_count"), "oracle-front LUT count"
                ),
                oracle._as_exact_int(
                    objectives.get("total_levels"), "oracle-front levels"
                ),
            )
        )
    if oracle.hypervolume(sensitivity_points, REFERENCE_POINT) != 700:
        raise RuntimeError("deferred full-oracle front does not reproduce HV 700")
    return {
        "verified": True,
        "source_sha256": records,
        "confirmed_arms": ["C", "A", "B", "D"],
        "full_oracle_sensitivity": {
            "available_only_after_durable_prediction": True,
            "reference_point": list(REFERENCE_POINT),
            "hypervolume": 700,
            "front": sensitivity_front,
        },
    }


ArmLabel = Literal["AD", "BD", "ABD"]
DirectionLabel = Literal["decrease", "same", "increase"]


class DirectionProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decrease: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    same: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    increase: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "DirectionProbabilities":
        total = self.decrease + self.same + self.increase
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("categorical probabilities must sum to one within 1e-6")
        return self


class ArmPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    total_lut_count: DirectionProbabilities
    total_levels: DirectionProbabilities


class RecombinationPrediction(BaseModel):
    """Compact ex-ante contract: predictions only, never configurations/actions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ranking: list[ArmLabel] = Field(
        min_length=3,
        max_length=3,
        json_schema_extra={"uniqueItems": True},
    )
    AD: ArmPrediction
    BD: ArmPrediction
    ABD: ArmPrediction

    @field_validator("ranking")
    @classmethod
    def _exact_permutation(cls, values: list[str]) -> list[str]:
        del cls
        if len(values) != 3 or set(values) != set(NEW_ARM_ORDER):
            raise ValueError("ranking must be one exact AD/BD/ABD permutation")
        return values


def _machine_fact(arm: CubeArm, *, include_objectives: bool) -> dict[str, object]:
    record = {
        **arm.identity_record(),
        "engine_patch": copy.deepcopy(arm.patch_record),
    }
    if include_objectives:
        if arm.known_objectives is None:
            raise RuntimeError("attempted to expose an unseen objective")
        record["objectives"] = {
            "total_lut_count": arm.known_objectives[0],
            "total_levels": arm.known_objectives[1],
        }
    else:
        record["objectives"] = "SEALED_AND_NOT_EVALUATED"
    record["fact_id"] = _sha256_json(record)
    return record


def prediction_prompt() -> str:
    payload = {
        "task": {
            "domain": "BOiLS/ABC log2 length-20",
            "mapping": "LUT-6 with mandatory CEC",
            "objectives": [
                {"name": "total_lut_count", "goal": "minimize"},
                {"name": "total_levels", "goal": "minimize"},
            ],
            "comparison_parent": "C",
        },
        "known_machine_facts": [
            _machine_fact(CUBE_BY_LABEL[label], include_objectives=True)
            for label in ("C", "A", "B", "D", "AB")
        ],
        "sealed_unseen_children": [
            _machine_fact(CUBE_BY_LABEL[label], include_objectives=False)
            for label in NEW_ARM_ORDER
        ],
    }
    return (
        "Predict the three fixed engine-materialized recombination children. "
        "Rank AD, BD, ABD best-to-worst exactly once. Best-to-worst means "
        "expected marginal search value relative to the known pre-block cube "
        "archive {C,A,B,D,AB}, whose front is {D,AB}, whose fixed-reference "
        "point is (8028,71), and whose hypervolume is 213. The "
        "observed target ordering will be descending marginal hypervolume from "
        "adding the arm to that archive, then lower Pareto layer in the combined "
        "cube, then lower LUT count, lower level count, then fixed arm order "
        "AD,BD,ABD. For each child and each objective, return probabilities "
        "that its value will decrease, stay the same, or increase versus C. "
        "Lower is better. Do not produce a configuration, selection, rationale, "
        "mechanism claim, or confidence scalar. Every child will be evaluated "
        "regardless of your ranking.\n\n"
        "MACHINE FACTS\n"
        + _canonical_json(payload)
    )


class Predictor(Protocol):
    async def __call__(
        self,
        request: StructuredGenerationRequest[RecombinationPrediction],
    ) -> (
        StructuredGenerationResponse[RecombinationPrediction]
        | AttemptedStructuredGenerationResponse[RecombinationPrediction]
    ): ...


def _provider_record(
    result: StructuredGenerationResponse[RecombinationPrediction]
    | AttemptedStructuredGenerationResponse[RecombinationPrediction],
) -> tuple[StructuredGenerationResponse[RecombinationPrediction], int, dict[str, object]]:
    if type(result) is AttemptedStructuredGenerationResponse:
        response = result.response
        attempts = result.attempt_count
    elif type(result) is StructuredGenerationResponse:
        response = result
        attempts = 1
    else:
        raise TypeError("predictor returned an unsupported response envelope")
    StructuredGenerationResponse.__post_init__(response)
    if type(response.value) is not RecombinationPrediction:
        raise TypeError("prediction response violates its exact output type")
    RecombinationPrediction.model_validate(response.value, strict=True)
    cost = response.cost_usd
    if response.requested_model != MODEL or response.resolved_model != MODEL:
        raise RuntimeError("prediction call did not use the exact frozen model")
    if response.resolved_provider != RESOLVED_PROVIDER:
        raise RuntimeError("prediction call did not resolve to Together")
    if attempts > QUEUE_MAX_ATTEMPTS:
        raise RuntimeError("prediction call exceeded its attempt budget")
    if cost is None or cost > MAX_SUCCESSFUL_RESPONSE_COST_USD:
        raise RuntimeError("prediction successful-response cost gate failed")
    record = {
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "resolved_provider": response.resolved_provider,
        "provider_response_id": response.provider_response_id,
        "finish_reason": response.finish_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
        "cost_usd": str(cost),
        "latency_ns": response.latency_ns,
        "attempt_count": attempts,
    }
    return response, attempts, record


def _assert_evaluator_provenance(evaluator: object) -> dict[str, object]:
    provenance_method = getattr(evaluator, "provenance", None)
    if not callable(provenance_method):
        raise RuntimeError("evaluator does not expose provenance")
    provenance = provenance_method()
    if type(provenance) is not dict:
        raise RuntimeError("evaluator provenance is not a mapping")
    circuits = provenance.get("circuits")
    if provenance.get("abc_binary_sha256") != EXPECTED_ABC_SHA256:
        raise RuntimeError("evaluator ABC provenance changed")
    if not (
        type(circuits) is list
        and len(circuits) == 1
        and circuits[0].get("name") == "log2"
        and circuits[0].get("sha256") == EXPECTED_CIRCUIT_SHA256
    ):
        raise RuntimeError("evaluator circuit provenance changed")
    if provenance.get("lut_inputs") != 6:
        raise RuntimeError("evaluator LUT mapping changed")
    if provenance.get("per_circuit_timeout_s") != float(
        PER_CANDIDATE_TIMEOUT_SECONDS
    ):
        raise RuntimeError("evaluator timeout changed")
    if provenance.get("affinity_sets") != [[cpu] for cpu in PHYSICAL_CPUS]:
        raise RuntimeError("evaluator affinity declaration changed")
    return copy.deepcopy(provenance)


def _objective_tuple(row: Mapping[str, object]) -> tuple[int, int]:
    objectives = row.get("objectives")
    if row.get("valid") is not True or type(objectives) is not dict:
        raise ValueError("invalid cube arms have no admissible objective tuple")
    return (
        oracle._as_exact_int(objectives.get("total_lut_count"), "cube LUT count"),
        oracle._as_exact_int(objectives.get("total_levels"), "cube levels"),
    )


def _dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left != right


def _front(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("valid") is True]
    return [
        copy.deepcopy(dict(row))
        for row in sorted(valid, key=lambda item: (*_objective_tuple(item), str(item["arm"])))
        if not any(
            other is not row
            and _dominates(_objective_tuple(other), _objective_tuple(row))
            for other in valid
        )
    ]


def _pareto_layers(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    remaining = [row for row in rows if row.get("valid") is True]
    result: dict[str, int] = {}
    layer = 1
    while remaining:
        current = [
            row
            for row in remaining
            if not any(
                other is not row
                and _dominates(_objective_tuple(other), _objective_tuple(row))
                for other in remaining
            )
        ]
        if not current:
            raise RuntimeError("Pareto layer construction made no progress")
        for row in current:
            result[str(row["arm"])] = layer
            remaining.remove(row)
        layer += 1
    return result


def _direction(value: int, baseline: int) -> DirectionLabel:
    if value < baseline:
        return "decrease"
    if value > baseline:
        return "increase"
    return "same"


def _prediction_for(
    prediction: RecombinationPrediction,
    arm: str,
    objective: str,
) -> DirectionProbabilities:
    arm_prediction = getattr(prediction, arm)
    return getattr(arm_prediction, objective)


def _categorical_calibration(
    prediction: RecombinationPrediction,
    rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    categories = ("decrease", "same", "increase")
    baseline = EXPECTED_SEED_OBJECTIVES
    details: list[dict[str, object]] = []
    for arm in NEW_ARM_ORDER:
        if rows[arm].get("valid") is not True:
            continue
        observed = _objective_tuple(rows[arm])
        for position, objective in enumerate(("total_lut_count", "total_levels")):
            distribution = _prediction_for(prediction, arm, objective)
            probabilities = distribution.model_dump(mode="python")
            observed_category = _direction(observed[position], baseline[position])
            predicted_category = min(
                categories,
                key=lambda category: (-float(probabilities[category]), categories.index(category)),
            )
            brier = sum(
                (float(probabilities[category]) - (category == observed_category)) ** 2
                for category in categories
            )
            observed_probability = float(probabilities[observed_category])
            details.append(
                {
                    "arm": arm,
                    "objective": objective,
                    "observed_category": observed_category,
                    "predicted_argmax": predicted_category,
                    "argmax_correct": predicted_category == observed_category,
                    "probability_assigned_to_observed": observed_probability,
                    "multiclass_brier": brier,
                    "distribution": probabilities,
                }
            )
    count = len(details)
    return {
        "cells": details,
        "cell_count": count,
        "missing_arms": [
            arm for arm in NEW_ARM_ORDER if rows[arm].get("valid") is not True
        ],
        "categorical_accuracy": (
            None
            if count == 0
            else sum(bool(row["argmax_correct"]) for row in details) / count
        ),
        "mean_probability_assigned_to_observed": (
            None
            if count == 0
            else sum(
                float(row["probability_assigned_to_observed"]) for row in details
            )
            / count
        ),
        "mean_multiclass_brier": (
            None
            if count == 0
            else sum(float(row["multiclass_brier"]) for row in details) / count
        ),
        "brier_definition": "sum over the three categorical cells",
        "argmax_tie_order": list(categories),
    }


def _interaction(
    values: Mapping[str, tuple[int, int]],
    terms: Sequence[tuple[int, str]],
) -> tuple[int, int]:
    return tuple(
        sum(coefficient * values[label][objective] for coefficient, label in terms)
        for objective in range(2)
    )


def analyze_cube(
    *,
    physical_outcomes: Sequence[Mapping[str, object]],
    prediction: RecombinationPrediction,
    provider: Mapping[str, object],
    deferred_oracle: Mapping[str, object],
    started_ns: int,
    completed_ns: int,
) -> dict[str, object]:
    if len(physical_outcomes) != EXPECTED_PHYSICAL_EVALUATIONS:
        raise RuntimeError("cube analysis requires C plus exactly three new arms")
    by_physical_label = {str(row["label"]): row for row in physical_outcomes}
    if tuple(by_physical_label) != ("C", *NEW_ARM_ORDER):
        raise RuntimeError("physical outcomes escaped the fixed report order")
    if by_physical_label["C"].get("valid") is not True or by_physical_label[
        "C"
    ].get("cec_passed") is not True:
        raise RuntimeError("fresh C became invalid before analysis")
    if _objective_tuple(by_physical_label["C"]) != EXPECTED_SEED_OBJECTIVES:
        raise RuntimeError("fresh C objective gate changed before analysis")
    for label in NEW_ARM_ORDER:
        physical = by_physical_label[label]
        if (
            physical.get("valid") is not True
            and physical.get("candidate_local_failure_status")
            == "cec_failed_or_missing"
        ):
            raise RuntimeError("mandatory CEC failed in the physical block")

    rows: list[dict[str, object]] = []
    for arm in CUBE:
        if arm.known_objectives is not None:
            objective = arm.known_objectives
            source = "sealed_pre_block"
            physical = by_physical_label.get(arm.label)
            if physical is not None and _objective_tuple(physical) != objective:
                raise RuntimeError("fresh C disagrees with its sealed objective")
            publication = None if physical is None else physical["publication_sequence"]
            elapsed = None if physical is None else physical["published_elapsed_ns"]
            affinity = None if physical is None else physical["cpu_affinity"]
            valid = True
            cec_passed = True
            failure_status = None
        else:
            physical = by_physical_label[arm.label]
            source = "fresh_recombination_block"
            publication = physical["publication_sequence"]
            elapsed = physical["published_elapsed_ns"]
            affinity = physical["cpu_affinity"]
            valid = physical.get("valid") is True
            cec_passed = physical.get("cec_passed") is True
            failure_status = physical.get("candidate_local_failure_status")
            objective = _objective_tuple(physical) if valid else None
        rows.append(
            {
                **arm.identity_record(),
                "valid": valid,
                "cec_passed": cec_passed,
                "candidate_local_failure_status": failure_status,
                "objective_source": source,
                "objectives": (
                    None
                    if objective is None
                    else {
                        "total_lut_count": objective[0],
                        "total_levels": objective[1],
                    }
                ),
                "publication_sequence": publication,
                "published_elapsed_ns": elapsed,
                "cpu_affinity": affinity,
            }
        )
    row_by_arm = {str(row["arm"]): row for row in rows}
    values = {
        label: _objective_tuple(row_by_arm[label])
        for label in ALL_ARM_ORDER
        if row_by_arm[label].get("valid") is True
    }

    interaction_terms: dict[str, tuple[tuple[int, str], ...]] = {
        "I_AB": ((1, "AB"), (-1, "A"), (-1, "B"), (1, "C")),
        "I_AD": ((1, "AD"), (-1, "A"), (-1, "D"), (1, "C")),
        "I_BD": ((1, "BD"), (-1, "B"), (-1, "D"), (1, "C")),
        "I_ABD": (
            (1, "ABD"),
            (-1, "AB"),
            (-1, "AD"),
            (-1, "BD"),
            (1, "A"),
            (1, "B"),
            (1, "D"),
            (-1, "C"),
        ),
    }
    interactions: dict[str, dict[str, object]] = {}
    interaction_values: dict[str, tuple[int, int]] = {}
    for name, terms in interaction_terms.items():
        required = tuple(dict.fromkeys(label for _, label in terms))
        missing = [label for label in required if label not in values]
        if missing:
            interactions[name] = {
                "available": False,
                "missing_arms": missing,
                "total_lut_count": None,
                "total_levels": None,
            }
            continue
        value = _interaction(values, terms)
        interaction_values[name] = value
        interactions[name] = {
            "available": True,
            "missing_arms": [],
            "total_lut_count": value[0],
            "total_levels": value[1],
            "sign_interpretation": (
                "negative=favorable synergy; positive=antagonism for minimized objective"
            ),
        }

    complete_cube = all(label in values for label in ALL_ARM_ORDER)
    if complete_cube:
        additive = _interaction(
            values,
            ((1, "A"), (1, "B"), (1, "D"), (-2, "C")),
        )
        main_plus_pair = _interaction(
            values,
            (
                (1, "AB"),
                (1, "AD"),
                (1, "BD"),
                (-1, "A"),
                (-1, "B"),
                (-1, "D"),
                (1, "C"),
            ),
        )
        triple = values["ABD"]
        prediction_arithmetic: dict[str, object] = {
            "available": True,
            "missing_arms": [],
            "observed_ABD": list(triple),
            "additive_main_effect_prediction": list(additive),
            "additive_main_effect_error_observed_minus_predicted": [
                triple[index] - additive[index] for index in range(2)
            ],
            "main_plus_pair_effect_prediction": list(main_plus_pair),
            "main_plus_pair_error_observed_minus_predicted": [
                triple[index] - main_plus_pair[index] for index in range(2)
            ],
            "third_order_residual_equals_main_plus_pair_error": (
                tuple(triple[index] - main_plus_pair[index] for index in range(2))
                == interaction_values["I_ABD"]
            ),
            "objective_order": ["total_lut_count", "total_levels"],
        }
    else:
        prediction_arithmetic = {
            "available": False,
            "missing_arms": [label for label in NEW_ARM_ORDER if label not in values],
            "objective_order": ["total_lut_count", "total_levels"],
        }

    preblock = [row_by_arm[label] for label in ("C", "A", "B", "D", "AB")]
    new_rows = [row_by_arm[label] for label in NEW_ARM_ORDER]
    preblock_front = _front(preblock)
    combined_front = _front(rows)
    preblock_hv = oracle.hypervolume(
        [_objective_tuple(row) for row in preblock], REFERENCE_POINT
    )
    terminal_hv = oracle.hypervolume(
        [_objective_tuple(row) for row in rows if row.get("valid") is True],
        REFERENCE_POINT,
    )
    if preblock_hv != 213 or {row["arm"] for row in preblock_front} != {"D", "AB"}:
        raise RuntimeError("known cube archive did not reproduce front {D,AB}/HV 213")
    combined_front_arms = {str(row["arm"]) for row in combined_front}
    valid_objectives = [
        _objective_tuple(row) for row in rows if row.get("valid") is True
    ]
    pareto_decisions = []
    for row in new_rows:
        if row.get("valid") is not True:
            pareto_decisions.append(
                {
                    "arm": row["arm"],
                    "valid": False,
                    "candidate_local_failure_status": row[
                        "candidate_local_failure_status"
                    ],
                    "dominated_by_preblock_arms": None,
                    "dominates_preblock_arms": None,
                    "nondominated_against_preblock": None,
                    "marginal_fixed_reference_hv_gain": None,
                    "enters_combined_front": False,
                    "unique_objective_vector_on_combined_cube_front": False,
                    "contributes_search_value": False,
                }
            )
            continue
        point = _objective_tuple(row)
        dominators = [
            str(other["arm"])
            for other in preblock
            if _dominates(_objective_tuple(other), point)
        ]
        dominated = [
            str(other["arm"])
            for other in preblock
            if _dominates(point, _objective_tuple(other))
        ]
        arm_hv = oracle.hypervolume(
            [*(_objective_tuple(other) for other in preblock), point], REFERENCE_POINT
        )
        enters_front = str(row["arm"]) in combined_front_arms
        unique_front_vector = enters_front and valid_objectives.count(point) == 1
        marginal_gain = arm_hv - preblock_hv
        contributes = unique_front_vector or marginal_gain > 0
        pareto_decisions.append(
            {
                "arm": row["arm"],
                "valid": True,
                "candidate_local_failure_status": None,
                "dominated_by_preblock_arms": dominators,
                "dominates_preblock_arms": dominated,
                "nondominated_against_preblock": not dominators,
                "marginal_fixed_reference_hv_gain": marginal_gain,
                "enters_combined_front": enters_front,
                "unique_objective_vector_on_combined_cube_front": unique_front_vector,
                "contributes_search_value": contributes,
            }
        )

    layers = _pareto_layers(rows)
    decision_by_arm = {str(row["arm"]): row for row in pareto_decisions}
    valid_new_labels = [
        label for label in NEW_ARM_ORDER if row_by_arm[label].get("valid") is True
    ]
    observed_ranking = sorted(
        valid_new_labels,
        key=lambda label: (
            -int(decision_by_arm[label]["marginal_fixed_reference_hv_gain"]),
            layers[label],
            *_objective_tuple(row_by_arm[label]),
            NEW_ARM_ORDER.index(label),
        ),
    )
    predicted_ranking = list(prediction.ranking)
    projected_predicted_ranking = [
        label for label in predicted_ranking if label in valid_new_labels
    ]
    observed_position = {label: index for index, label in enumerate(observed_ranking)}
    predicted_position = {
        label: index for index, label in enumerate(projected_predicted_ranking)
    }
    pairs = tuple(
        (left, right)
        for left_index, left in enumerate(valid_new_labels)
        for right in valid_new_labels[left_index + 1 :]
    )
    pairwise = [
        {
            "pair": [left, right],
            "concordant": (
                (predicted_position[left] < predicted_position[right])
                == (observed_position[left] < observed_position[right])
            ),
        }
        for left, right in pairs
    ]
    rank_calibration = {
        "predicted_ranking": predicted_ranking,
        "projected_predicted_ranking_over_valid_arms": projected_predicted_ranking,
        "observed_ranking": observed_ranking,
        "missing_arms": [
            label for label in NEW_ARM_ORDER if label not in valid_new_labels
        ],
        "observed_order_definition": (
            "descending marginal HV gain versus the pre-block archive, then "
            "combined Pareto layer, LUTs, levels, and frozen arm order"
        ),
        "top_one_correct": (
            None
            if not observed_ranking
            else projected_predicted_ranking[0] == observed_ranking[0]
        ),
        "exact_permutation_correct": (
            predicted_ranking == observed_ranking if complete_cube else None
        ),
        "pairwise_cells": pairwise,
        "pairwise_accuracy": (
            None
            if not pairwise
            else sum(bool(row["concordant"]) for row in pairwise) / len(pairwise)
        ),
    }
    categorical = _categorical_calibration(prediction, row_by_arm)

    recombination_advances = any(
        row["contributes_search_value"] is True for row in pareto_decisions
    )
    interaction_advances = complete_cube

    sensitivity = deferred_oracle.get("full_oracle_sensitivity")
    if type(sensitivity) is not dict or sensitivity.get("hypervolume") != 700:
        raise RuntimeError("post-prediction full-oracle sensitivity is unavailable")
    sensitivity_front = sensitivity.get("front")
    if type(sensitivity_front) is not list:
        raise RuntimeError("post-prediction full-oracle sensitivity front is invalid")
    sensitivity_points: list[tuple[int, int]] = []
    for sensitivity_row in sensitivity_front:
        if type(sensitivity_row) is not dict or type(
            sensitivity_row.get("objectives")
        ) is not dict:
            raise RuntimeError("full-oracle sensitivity contains an invalid front row")
        objectives = sensitivity_row["objectives"]
        sensitivity_points.append(
            (
                oracle._as_exact_int(objectives.get("total_lut_count"), "sensitivity LUT"),
                oracle._as_exact_int(objectives.get("total_levels"), "sensitivity levels"),
            )
        )
    if oracle.hypervolume(sensitivity_points, REFERENCE_POINT) != 700:
        raise RuntimeError("full-oracle sensitivity failed to reproduce HV 700")
    sensitivity_arm_rows = []
    for label in NEW_ARM_ORDER:
        row = row_by_arm[label]
        if row.get("valid") is not True:
            sensitivity_arm_rows.append(
                {"arm": label, "valid": False, "marginal_hv_gain": None}
            )
            continue
        point = _objective_tuple(row)
        arm_hv = oracle.hypervolume([*sensitivity_points, point], REFERENCE_POINT)
        sensitivity_arm_rows.append(
            {"arm": label, "valid": True, "marginal_hv_gain": arm_hv - 700}
        )
    sensitivity_terminal_hv = oracle.hypervolume(
        [
            *sensitivity_points,
            *(
                _objective_tuple(row_by_arm[label])
                for label in NEW_ARM_ORDER
                if row_by_arm[label].get("valid") is True
            ),
        ],
        REFERENCE_POINT,
    )
    elapsed_ns = completed_ns - started_ns
    child_affinities = [
        tuple(row_by_arm[label]["cpu_affinity"] or ()) for label in NEW_ARM_ORDER
    ]
    affinity_gate = all(
        len(affinity) == 1 and affinity[0] in PHYSICAL_CPUS
        for affinity in child_affinities
    ) and len(set(child_affinities)) == len(child_affinities)
    protocol_gates = {
        "fresh_seed_exact": values["C"] == EXPECTED_SEED_OBJECTIVES,
        "fixed_physical_order_and_count": tuple(by_physical_label)
        == ("C", *NEW_ARM_ORDER),
        "engine_patch_replay_and_identity": all(
            arm.patch_record is not None
            and arm.patch_record["replay_verified"] is True
            and arm.patch_record["target_hash"]
            == arm.typed_json_configuration_sha256
            for arm in CUBE
        ),
        "prediction_exact_model_provider_and_cost": (
            provider["requested_model"] == MODEL
            and provider["resolved_model"] == MODEL
            and provider["resolved_provider"] == RESOLVED_PROVIDER
            and Decimal(str(provider["cost_usd"])) <= MAX_SUCCESSFUL_RESPONSE_COST_USD
            and int(provider["attempt_count"]) <= QUEUE_MAX_ATTEMPTS
        ),
        "child_affinity_leases_exact_and_distinct": affinity_gate,
        "deferred_oracle_parsed_after_durable_prediction": deferred_oracle.get(
            "verified"
        )
        is True,
        "hard_cleanup_deadline_met": elapsed_ns
        <= HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
    }
    quality_horizon_met = elapsed_ns <= QUALITY_HORIZON_SECONDS * 1_000_000_000
    scientific_completeness = {
        "all_three_new_arms_valid": complete_cube,
        "complete_cube_arithmetic_available": complete_cube,
        "quality_horizon_met": quality_horizon_met,
    }
    invalid_arms = [
        {
            "arm": label,
            "candidate_local_failure_status": row_by_arm[label][
                "candidate_local_failure_status"
            ],
        }
        for label in NEW_ARM_ORDER
        if row_by_arm[label].get("valid") is not True
    ]
    return {
        "schema_version": 1,
        "status": "succeeded" if complete_cube else "partial_candidate_local_invalid",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "protocol_acceptance_passed": all(protocol_gates.values()),
        "claim_boundary": (
            "One post-hoc three-edit BOiLS/log2 recombination cube; not an "
            "optimizer, memory, genericity, SOTA, or wall-clock claim."
        ),
        "cube_outcomes": rows,
        "engine_materialized_patches": {
            arm.label: copy.deepcopy(arm.patch_record) for arm in CUBE
        },
        "partial_negative_record": {
            "present": bool(invalid_arms),
            "invalid_arms": invalid_arms,
            "fixed_arms_consumed_without_replacement": bool(invalid_arms),
        },
        "interactions": interactions,
        "triple_prediction_arithmetic": prediction_arithmetic,
        "pareto": {
            "primary_comparison_archive": ["C", "A", "B", "D", "AB"],
            "preblock_front": ["D", "AB"],
            "combined_front": [row["arm"] for row in combined_front],
            "new_arm_decisions": pareto_decisions,
            "triple_enters_combined_development_front": any(
                row["arm"] == "ABD" for row in combined_front
            ),
        },
        "hypervolume": {
            "reference_point": list(REFERENCE_POINT),
            "preblock": preblock_hv,
            "terminal": terminal_hv,
            "delta": terminal_hv - preblock_hv,
        },
        "full_oracle_sensitivity_post_prediction_only": {
            "primary_decision_uses_this": False,
            "reference_point": list(REFERENCE_POINT),
            "preblock_hypervolume": 700,
            "terminal_hypervolume": sensitivity_terminal_hv,
            "delta": sensitivity_terminal_hv - 700,
            "preblock_front": copy.deepcopy(sensitivity_front),
            "new_arm_marginals": sensitivity_arm_rows,
        },
        "model_prediction": prediction.model_dump(mode="json"),
        "model_rank_calibration": rank_calibration,
        "model_categorical_calibration": categorical,
        "provider_call": copy.deepcopy(dict(provider)),
        "decision": {
            "deterministic_disjoint_recombination_advances": recombination_advances,
            "interaction_recording_advances": interaction_advances,
            "llm_ranking_retained_as_calibration_only": True,
            "llm_affected_physical_selection_or_order": False,
        },
        "resources": {
            "physical_evaluations": EXPECTED_PHYSICAL_EVALUATIONS,
            "logical_llm_calls": 1,
            "elapsed_ns": elapsed_ns,
            "quality_horizon_ns": QUALITY_HORIZON_SECONDS * 1_000_000_000,
            "hard_cleanup_deadline_ns": HARD_CLEANUP_DEADLINE_SECONDS
            * 1_000_000_000,
            "new_arm_publications_within_quality_horizon": sum(
                int(row_by_arm[label]["published_elapsed_ns"])
                <= QUALITY_HORIZON_SECONDS * 1_000_000_000
                for label in NEW_ARM_ORDER
            ),
            "quality_horizon_met": quality_horizon_met,
            "quality_horizon_failure": not quality_horizon_met,
            "valid_new_arms": len(valid_new_labels),
            "candidate_local_invalid_arms": len(invalid_arms),
            "child_cpu_affinities": [list(value) for value in child_affinities],
        },
        "limitations": [
            "The cube combines sealed v1/v2/oracle outcomes with three fresh arms.",
            "The cube was selected after the local oracle even though its branch facts existed earlier.",
            "Cross-run arithmetic is exact descriptive interaction evidence, not a randomized same-block factorial estimate.",
            "Ranking calibration uses the preregistration-compatible deterministic operationalization recorded above.",
        ],
        "protocol_gates": protocol_gates,
        "scientific_completeness": scientific_completeness,
        "gates": {**protocol_gates, **scientific_completeness},
    }


async def run_block(
    *,
    evaluator: object,
    recorder: oracle.EvaluationPublicationRecorder,
    trace: oracle.TraceRecorder,
    predictor: Predictor,
    evidence_bundle: Mapping[str, Mapping[str, object]] | None = None,
    deferred_oracle_loader: Callable[[], Mapping[str, object]] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Execute seed, one advisory prediction, and the fixed three-arm wave."""

    provenance = _assert_evaluator_provenance(evaluator)
    evidence = (
        copy.deepcopy(dict(evidence_bundle))
        if evidence_bundle is not None
        else verify_evidence_bundle()
    )
    if set(evidence) != set(EVIDENCE_SOURCES):
        raise RuntimeError("evidence bundle omitted a frozen source")
    for name, (_, expected_hash) in EVIDENCE_SOURCES.items():
        row = evidence[name]
        if type(row) is not dict or row.get("sha256") != expected_hash:
            raise RuntimeError(f"evidence bundle identity changed: {name}")
    started_ns = clock_ns()
    recorder.begin(started_ns)
    trace.begin(started_ns)
    trace.emit(
        "recombination_block_started",
        preregistration_sha256=EXPECTED_PREREGISTRATION_SHA256,
        protocol_correction_sha256=EXPECTED_CORRECTION_SHA256,
        physical_arm_order=["C", *NEW_ARM_ORDER],
        quality_horizon_ns=QUALITY_HORIZON_SECONDS * 1_000_000_000,
        hard_cleanup_deadline_ns=HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
        evaluator_provenance=provenance,
        evidence_source_sha256={name: row["sha256"] for name, row in evidence.items()},
    )

    parent_spec = PHYSICAL_SCHEDULE[0]
    trace.emit("candidate_submitted", arm="C", **parent_spec.identity_record())
    try:
        parent_outcome = await asyncio.to_thread(
            oracle._evaluate_one,
            evaluator=evaluator,
            recorder=recorder,
            spec=parent_spec,
        )
    except BaseException as exc:
        raise oracle.SeedGateError("fresh C failed before the model call") from exc
    if (
        parent_outcome.get("valid") is not True
        or parent_outcome.get("cec_passed") is not True
        or _objective_tuple(parent_outcome) != EXPECTED_SEED_OBJECTIVES
        or parent_outcome["boils_configuration_sha256"]
        != EXPECTED_IDENTITIES["C"][0]
        or parent_outcome["typed_json_configuration_sha256"]
        != EXPECTED_IDENTITIES["C"][1]
    ):
        raise oracle.SeedGateError("fresh C identity/objective/CEC gate failed")
    trace.emit(
        "fresh_seed_gate_passed",
        arm="C",
        objectives={
            "total_lut_count": EXPECTED_SEED_OBJECTIVES[0],
            "total_levels": EXPECTED_SEED_OBJECTIVES[1],
        },
        boils_configuration_sha256=EXPECTED_IDENTITIES["C"][0],
        typed_json_configuration_sha256=EXPECTED_IDENTITIES["C"][1],
    )

    prompt = prediction_prompt()
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_boils_recombination_v3_prediction"),
        operation="recombination_prediction",
        prompt=prompt,
        output_type=RecombinationPrediction,
        output_tool_name="return_recombination_prediction",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    trace.emit(
        "prediction_requested",
        call_id=request.call_id.value,
        requested_model=MODEL,
        prompt=prompt,
        prompt_sha256=_sha256_bytes(prompt.encode("utf-8")),
        sealed_new_arm_order=list(NEW_ARM_ORDER),
        physical_selection_owner="engine_fixed_schedule",
    )
    result = await predictor(request)
    response, _, provider_record = _provider_record(result)
    prediction = response.value
    trace.emit(
        "prediction_completed",
        call_id=request.call_id.value,
        prediction=prediction.model_dump(mode="json"),
        prediction_sha256=_sha256_json(prediction.model_dump(mode="json")),
        provider=provider_record,
        physical_selection_owner="engine_fixed_schedule",
    )
    deferred_oracle = dict(
        verify_deferred_oracle_evidence()
        if deferred_oracle_loader is None
        else deferred_oracle_loader()
    )
    if (
        deferred_oracle.get("verified") is not True
        or deferred_oracle.get("confirmed_arms") != ["C", "A", "B", "D"]
    ):
        raise RuntimeError("deferred oracle verification returned an invalid record")
    trace.emit(
        "deferred_oracle_evidence_verified",
        verification=copy.deepcopy(deferred_oracle),
        chronology="after_durable_prediction_before_child_submission",
    )

    latest_safe_start_ns = (
        HARD_CLEANUP_DEADLINE_SECONDS - PER_CANDIDATE_TIMEOUT_SECONDS
    ) * 1_000_000_000
    if clock_ns() - started_ns >= latest_safe_start_ns:
        raise RuntimeError("insufficient hard-deadline budget to submit all fixed children")
    child_specs = PHYSICAL_SCHEDULE[1:]
    if tuple(spec.label for spec in child_specs) != NEW_ARM_ORDER:
        raise RuntimeError("physical child schedule escaped AD/BD/ABD order")
    trace.emit(
        "fixed_child_wave_started",
        submission_order=list(NEW_ARM_ORDER),
        model_ranking=list(prediction.ranking),
        model_controls_submission=False,
    )
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(
        max_workers=CHILD_WORKERS,
        thread_name_prefix="boils-recombination-v3",
    ) as executor:
        futures = []
        for spec in child_specs:
            trace.emit(
                "candidate_submitted",
                arm=spec.label,
                **spec.identity_record(),
                model_rank=prediction.ranking.index(spec.label) + 1,
                submission_selected_by="engine_fixed_schedule",
            )
            futures.append(
                loop.run_in_executor(
                    executor,
                    lambda current=spec: oracle._evaluate_one(
                        evaluator=evaluator,
                        recorder=recorder,
                        spec=current,
                    ),
                )
            )
        # gather preserves the fixed report order, not completion order.
        child_outcomes = await asyncio.gather(*futures)
    completed_ns = clock_ns()
    if completed_ns - started_ns > HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000:
        raise RuntimeError("recombination block exceeded its hard cleanup deadline")
    if len(recorder.records()) != EXPECTED_PHYSICAL_EVALUATIONS:
        raise RuntimeError("durable physical publication count is not exactly four")
    trace.emit(
        "fixed_child_wave_completed",
        report_order=list(NEW_ARM_ORDER),
        candidate_local_invalids=sum(
            outcome.get("valid") is not True for outcome in child_outcomes
        ),
    )
    summary = analyze_cube(
        physical_outcomes=(parent_outcome, *child_outcomes),
        prediction=prediction,
        provider=provider_record,
        deferred_oracle=deferred_oracle,
        started_ns=started_ns,
        completed_ns=completed_ns,
    )
    summary["sealed_evidence_bundle"] = copy.deepcopy(dict(evidence))
    summary["deferred_oracle_verification"] = copy.deepcopy(deferred_oracle)
    summary["evaluator_provenance"] = provenance
    if summary["protocol_acceptance_passed"] is not True:
        raise RuntimeError("a strict recombination-block acceptance gate failed")
    trace.emit(
        "recombination_analysis_completed",
        protocol_acceptance_passed=summary["protocol_acceptance_passed"],
        terminal_hypervolume=summary["hypervolume"]["terminal"],
        recombination_advances=summary["decision"][
            "deterministic_disjoint_recombination_advances"
        ],
    )
    return summary


async def _run_live(
    *,
    evaluator: BoilsAbcEvaluator,
    recorder: oracle.EvaluationPublicationRecorder,
    trace: oracle.TraceRecorder,
    queue_writer: v1.DurableJsonlWriter,
    evidence_bundle: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=float(QUEUE_ATTEMPT_TIMEOUT_SECONDS),
        provider_options={"order": list(PROVIDER_ORDER), "allow_fallbacks": False},
        app_title="AgentEvolve AAAI 2027 BOiLS recombination v3",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
        base_backoff_ns=QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
        max_backoff_ns=QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
        close_generator=True,
        outcome_sink=lambda outcome: queue_writer.write(
            support._queue_outcome_record(outcome)
        ),
    )
    async with runner:
        return await run_block(
            evaluator=evaluator,
            recorder=recorder,
            trace=trace,
            predictor=runner,
            evidence_bundle=evidence_bundle,
        )


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "v1_durable_helpers": Path(v1.__file__).resolve(),
        "v2_cube_parent": Path(v2.__file__).resolve(),
        "oracle_evaluation_audit": Path(oracle.__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/actions.py",
        "evaluator": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        "typed_patch": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/variation/typed_patch.py",
        "patch_domain": AGENT_EVOLVE_ROOT / "src/agent_evolve/domain/patch.py",
        "structured_port": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/structured_generator.py",
        "queue": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/llm_task_queue.py",
        "queued_runner": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
    }
    return {name: support._sha256(path) for name, path in paths.items()}


def _manifest(
    *,
    run_id: str,
    evaluator: BoilsAbcEvaluator,
    evidence_bundle: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "One post-hoc BOiLS/log2 deterministic recombination cube; not an "
            "optimizer, memory, genericity, SOTA, or wall-clock claim."
        ),
        "preregistration": {
            "source": str(PREREGISTRATION_PATH),
            "durable_copy": "preregistration.md",
            "sha256": EXPECTED_PREREGISTRATION_SHA256,
        },
        "protocol_correction": {
            "source": str(CORRECTION_PATH),
            "durable_copy": "protocol_correction.md",
            "sha256": EXPECTED_CORRECTION_SHA256,
            "supersedes_only_enumerated_artifact_66_clauses": True,
        },
        "sealed_evidence_bundle": copy.deepcopy(dict(evidence_bundle)),
        "cube": {
            "all_arm_order": list(ALL_ARM_ORDER),
            "physical_order": ["C", *NEW_ARM_ORDER],
            "arms": [
                {
                    **arm.identity_record(),
                    "configuration": arm.configuration,
                    "known_objectives": (
                        None
                        if arm.known_objectives is None
                        else {
                            "total_lut_count": arm.known_objectives[0],
                            "total_levels": arm.known_objectives[1],
                        }
                    ),
                    "engine_patch": copy.deepcopy(arm.patch_record),
                }
                for arm in CUBE
            ],
            "model_controls_physical_selection": False,
        },
        "task": {
            "circuit": "log2",
            "circuit_sha256": EXPECTED_CIRCUIT_SHA256,
            "abc_sha256": EXPECTED_ABC_SHA256,
            "mapping": "LUT-6 followed by mandatory CEC",
            "logical_cpus": list(PHYSICAL_CPUS),
            "per_candidate_timeout_seconds": PER_CANDIDATE_TIMEOUT_SECONDS,
            "quality_horizon_seconds": QUALITY_HORIZON_SECONDS,
            "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
            "private_cache_free": True,
            "physical_evaluations": EXPECTED_PHYSICAL_EVALUATIONS,
            "retries": 0,
            "replacement_candidates": 0,
        },
        "model": {
            "requested": MODEL,
            "required_resolved": MODEL,
            "required_provider": RESOLVED_PROVIDER,
            "provider_options": {
                "order": list(PROVIDER_ORDER),
                "allow_fallbacks": False,
            },
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "successful_response_cost_ceiling_usd": str(
                MAX_SUCCESSFUL_RESPONSE_COST_USD
            ),
            "logical_calls": 1,
            "output_contract": RecombinationPrediction.model_json_schema(),
        },
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_ns": QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
            "base_backoff_ns": QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
            "max_backoff_ns": QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
            "retry_owner": "AsyncLLMTaskQueue",
            "sdk_retries": 0,
            "pydantic_ai_retries": 0,
        },
        "analysis": {
            "reference_point": list(REFERENCE_POINT),
            "interactions": ["I_AB", "I_AD", "I_BD", "I_ABD"],
            "objective_order": ["total_lut_count", "total_levels"],
            "primary_comparison_archive": ["C", "A", "B", "D", "AB"],
            "required_primary_preblock_front": ["D", "AB"],
            "required_primary_preblock_hypervolume": 213,
            "rank_operationalization": (
                "descending marginal HV gain versus pre-block archive, then "
                "combined Pareto layer, LUTs, levels, frozen arm order"
            ),
            "search_value_rule": (
                "unique objective vector on combined cube front OR positive "
                "marginal HV versus known cube archive; duplicate alone insufficient"
            ),
            "candidate_local_invalid_rule": (
                "fixed arm consumed; retain partial negative summary; other valid "
                "new arms remain eligible for the search-value rule"
            ),
            "full_oracle_sensitivity_role": (
                "values unavailable in this pre-call manifest; construct and "
                "report only after durable prediction; never primary decision"
            ),
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
            "credential_variable": "OPENROUTER_API_KEY",
            "packages": {
                name: support._package_version(name)
                for name in ("pydantic", "pydantic-ai", "openai", "httpx")
            },
        },
    }


_DURABLE_COPY_NAMES = {
    "preregistration": "preregistration.md",
    "protocol_correction": "protocol_correction.md",
    "v1_finalized": "sealed_v1_finalized.json",
    "v1_summary": "sealed_v1_summary.json",
    "v2_finalized": "sealed_v2_finalized.json",
    "v2_summary": "sealed_v2_summary.json",
    "oracle_finalized": "sealed_oracle_finalized.json",
    "oracle_summary": "sealed_oracle_summary.json",
}


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        *_DURABLE_COPY_NAMES.values(),
        "events.jsonl",
        "queue_outcomes.jsonl",
        "evaluations.jsonl",
        "summary.json",
        "failure.json",
    )
    files: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if not corpus_path_or_none(path) is not None:
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
            "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "protocol_correction_sha256": EXPECTED_CORRECTION_SHA256,
            "files": files,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cpus", default=",".join(str(cpu) for cpu in PHYSICAL_CPUS))
    parser.add_argument("--max-attempts", type=int, default=QUEUE_MAX_ATTEMPTS)
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=int,
        default=QUEUE_ATTEMPT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--quality-horizon-seconds", type=int, default=QUALITY_HORIZON_SECONDS
    )
    parser.add_argument(
        "--hard-cleanup-deadline-seconds",
        type=int,
        default=HARD_CLEANUP_DEADLINE_SECONDS,
    )
    return parser


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "model": MODEL,
        "cpus": ",".join(str(cpu) for cpu in PHYSICAL_CPUS),
        "max_attempts": QUEUE_MAX_ATTEMPTS,
        "attempt_timeout_seconds": QUEUE_ATTEMPT_TIMEOUT_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "quality_horizon_seconds": QUALITY_HORIZON_SECONDS,
        "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
    }
    for name, expected_value in expected.items():
        if getattr(args, name) != expected_value:
            raise SystemExit(
                f"BOiLS recombination v3 freezes --{name.replace('_', '-')}="
                f"{expected_value}"
            )


def main() -> None:
    args = _parser().parse_args()
    _assert_frozen_cli(args)
    evidence_bundle = verify_evidence_bundle()
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "boils_recombination_v3_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status = "failed"
    event_writer: v1.DurableJsonlWriter | None = None
    evaluation_writer: v1.DurableJsonlWriter | None = None
    queue_writer: v1.DurableJsonlWriter | None = None
    try:
        shutil.copyfile(Path(__file__).resolve(), run_dir / "runner_source.py")
        for name, destination in _DURABLE_COPY_NAMES.items():
            shutil.copyfile(EVIDENCE_SOURCES[name][0], run_dir / destination)
            if support._sha256(run_dir / destination) != EVIDENCE_SOURCES[name][1]:
                raise RuntimeError(
                    f"durable evidence copy failed its hash gate: {name}"
                )
        event_writer = v1.DurableJsonlWriter(run_dir / "events.jsonl")
        evaluation_writer = v1.DurableJsonlWriter(run_dir / "evaluations.jsonl")
        queue_writer = v1.DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
        trace = oracle.TraceRecorder(event_writer)
        recorder = oracle.EvaluationPublicationRecorder(
            evaluation_writer,
            trace,
            schedule=PHYSICAL_SCHEDULE,
        )
        load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
        if support._sha256(run_dir / "runner_source.py") != support._sha256(
            Path(__file__).resolve()
        ):
            raise RuntimeError("durable runner copy failed its hash gate")
        settings = AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("log2",),
            affinity_sets=tuple((cpu,) for cpu in PHYSICAL_CPUS),
            per_circuit_timeout_s=float(PER_CANDIDATE_TIMEOUT_SECONDS),
        )
        evaluator = BoilsAbcEvaluator(settings, observer=recorder)
        _assert_evaluator_provenance(evaluator)
        support._write_json(
            run_dir / "manifest.json",
            _manifest(
                run_id=run_id,
                evaluator=evaluator,
                evidence_bundle=evidence_bundle,
            ),
        )
        summary = asyncio.run(
            _run_live(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                queue_writer=queue_writer,
                evidence_bundle=evidence_bundle,
            )
        )
        queue_writer.close()
        queue_summary = support._queue_log_summary(run_dir / "queue_outcomes.jsonl")
        if queue_summary["terminal_outcomes"] != 1:
            raise RuntimeError("prediction queue did not seal exactly one logical outcome")
        summary["queue"] = queue_summary
        summary["evaluator_observations"] = v1._evaluation_log_summary(
            run_dir / "evaluations.jsonl"
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
                    or type(exc).__module__.startswith("agent_evolve")
                    else "BOiLS recombination v3 failed; inspect durable sanitized traces"
                ),
            },
        )
        raise
    finally:
        if queue_writer is not None:
            queue_writer.close()
        if event_writer is not None:
            event_writer.close()
        if evaluation_writer is not None:
            evaluation_writer.close()
        _finalize(run_dir, status)
    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()
