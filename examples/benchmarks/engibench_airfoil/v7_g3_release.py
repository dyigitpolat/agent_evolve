"""Provider-free release preparation for the Airfoil-v7 G3 mechanism screen.

This module is deliberately domain-local.  It owns Airfoil card reconstruction,
the outcome-blind parent/sham/mate laws, no-CFD geometry admission, an absolute
endpoint, and the benchmark hypothesis compiler.  It never reads credentials,
calls an LLM, or invokes the physical evaluator.

The historical freshness boundary is split in two:

* :func:`build_historical_denylist` is an offline one-shot authority.  It may
  traverse outcome-bearing historical JSON/JSONL, but emits only hashes and a
  recursively bound source manifest.
* :func:`build_authenticated_trim_card_bank` is a separate offline authority.
  It opens exactly three allowlisted frozen-v2 files and emits four fully typed,
  provenance-bound historical memory entries.
* :func:`prepare_release` consumes only those two sealed exports.  It never
  opens their historical source paths, traverses candidate/evaluator logs, or
  exposes current-run or held-out outcomes to any selector.

The resulting record is launch preparation, not an authorization to launch and
not efficacy evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.executable_hypothesis import (
    registered_source_evidence_sha256,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightOrigin,
    context_stratum_hash,
)
from agent_evolve.application.g3_causal_screen import FrozenDiagnosticPermutation
from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.memory.treatment_compliance import (
    InsightTreatmentRequirement,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentInsightEvidence,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchMaterialization,
    DisjointPatchRecombiner,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    validate_hypothesis_compilation,
    validate_hypothesis_compiler_identity,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    REPRESENTATION_ID,
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    ARCHIVE_DEFINITION_SHA256,
    DELTA_F,
    PHENOTYPE_DEFINITION_SHA256,
    PHENOTYPE_POLICY_ID,
    TASK_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
    decoded_float32_le_bytes,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7UnionVariationCatalog,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
HISTORICAL_RUN = (
    ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_v7"
    / "two_stage"
    / "ae7_generic_two_stage_generation_v2_20260715"
)
DEFAULT_DENYLIST_PATH = (
    ARTIFACT_ROOT
    / "airfoil_g3_release"
    / "airfoil_v7_historical_membership_v1.json"
)
DEFAULT_CARD_BANK_PATH = (
    ARTIFACT_ROOT
    / "airfoil_g3_release"
    / "airfoil_v7_authenticated_trim_card_bank_v1.json"
)
DEFAULT_RELEASE_PATH = (
    ARTIFACT_ROOT
    / "airfoil_g3_release"
    / "airfoil_v7_g3_provider_free_release_v1.json"
)
DEFAULT_FREEZE_RECEIPT_PATH = (
    ARTIFACT_ROOT
    / "airfoil_g3_release"
    / "airfoil_v7_g3_prelaunch_freeze_v1.json"
)
DEFAULT_HISTORICAL_SOURCE_ROOTS = (
    ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7",
    ARTIFACT_ROOT / "experiment_logs" / "benchmark_q0" / "engibench_airfoil",
    ARTIFACT_ROOT / "experiment_logs" / "benchmark_q1" / "engibench_airfoil",
)

EXPECTED_REFLECTION_RESULT_SHA256 = (
    "b1b114ce7104fb669db27b47afbe7aa95ce69000e8791ce07713051000d3db8f"
)
EXPECTED_PROVIDER_FREE_READINESS_SHA256 = (
    "d5047b200b97a6843d821007655c7d6a42c1644f8d67b9cfd3485fc468bb5612"
)
EXPECTED_FINALIZED_FILE_SHA256 = (
    "8779680b658ca32a712048e6ab323ba8a1a53e9510a02a28125366c610ea8c0c"
)
EXPECTED_FINALIZATION_SHA256 = (
    "03418756ebd5031222261c9e06d776a54a0c6102d3da155e1bb8efff464ca752"
)
EXPECTED_RECURSIVE_CONTENT_SHA256 = (
    "c6daea0851af5dcb0eb9c2983ca6d41cc3d1f5a5f3a4fef32242bff8fee26124"
)
EXPECTED_HISTORICAL_STATUS = "transport_incomplete"
EXPECTED_HISTORICAL_MEMBERSHIP_SHA256 = (
    "4b5a2bda8ce647458f4d2300bd10606f372174fdfeaf7222a749e5464e4a7e67"
)
EXPECTED_HISTORICAL_MEMBERSHIP_FILE_SHA256 = (
    "d9c30bb7a218550300bf1bed7c86c9a565bfba2238b64de43f953e4d0f259b23"
)
EXPECTED_HISTORICAL_SOURCE_MANIFEST_SHA256 = (
    "618463b07c9f353bcf3d4c221dece08f10030f436b0796c6b5c0fa0b58169427"
)
EXPECTED_CARD_BANK_SHA256 = (
    "4d61692ebe5584830786e534ce74f98ca19874cf1c1fea4bc91a55666db470bd"
)
EXPECTED_CARD_BANK_FILE_SHA256 = (
    "6576ab6d8664eaf942ae0e949b72f49608b227bda688786ae45df5ea328acccf"
)

_HISTORICAL_RUN_RELATIVE = HISTORICAL_RUN.relative_to(WORKSPACE_ROOT).as_posix()
EXPECTED_CARD_SOURCE_FILES = (
    (
        f"{_HISTORICAL_RUN_RELATIVE}/finalized.json",
        2871,
        EXPECTED_FINALIZED_FILE_SHA256,
    ),
    (
        f"{_HISTORICAL_RUN_RELATIVE}/provider_free_readiness.json",
        156124,
        EXPECTED_PROVIDER_FREE_READINESS_SHA256,
    ),
    (
        f"{_HISTORICAL_RUN_RELATIVE}/reflection_result.json",
        19039,
        EXPECTED_REFLECTION_RESULT_SHA256,
    ),
)

EXPECTED_AUTHENTICATED_CARD_IDENTITIES = {
    "insight_airfoil_twostage_cards_000001": (
        1,
        "11ad780d8971865541f9094f00a4262ad28677573daacbb2d317d5fbff3cca65",
        "dbc72209e687af27497ce7b7f9f95bfa59fdd44fed8d95b4ab0fec3335cb2c64",
        "02a9541858d4484a550942964f6f896f270cc01eb7609b1913c39355d7d0488d",
    ),
    "insight_airfoil_twostage_cards_000002": (
        1,
        "45612e513d6a72d66cfb2d99c58541cfb60982d9ab17cd83fed6b5e63e84052c",
        "f3f05b9b6034f856a37f01266ff0ffb60201d78c4393b60cdf29d4cdea5f299c",
        "eff38873b63a4f0f5fc97b3e8a3c32d599b4ca069360a3ec966ec1c328fd22ed",
    ),
    "insight_airfoil_twostage_cards_000006": (
        1,
        "f54be79027599f9a31fa766b871c804de6ffc5668f79fae0320126dadade3993",
        "8b0d0aebeeb4bd7429e59269bbea11a09dc49a4475c17338fe469716883964e4",
        "06b8b98a9a980d17c69b4ae3a3ec85b27480804b2f9e6d8a848ece0f92cf4162",
    ),
    "insight_airfoil_twostage_cards_000007": (
        1,
        "425e23a51a5ed9831defe58faf22095f302ecd06aa9d23138c4f7a8f7271cfb5",
        "abb0451f5cbc2956879ef5632977f574e2de29097bd57b8801ca3e5fd15d9940",
        "5d65afae5afb9714e2d6defe0eec15e68747abe61ecf404db2673add33327b2f",
    ),
}

TRIM_PATHS = (
    "$.alpha_deg[0]",
    "$.alpha_deg[1]",
    "$.alpha_deg[2]",
)
SHAPE_HELD_FIXED_PATHS = (
    "$.lower_coefficients",
    "$.representation_id",
    "$.upper_coefficients",
)
REQUIRED_METRIC_IDS = (
    "objective:normalized_multipoint_drag",
    "violation:normalized_lift_equality",
)
EXPECTED_TRIM_DRAFT_SHA256S = (
    "11ad780d8971865541f9094f00a4262ad28677573daacbb2d317d5fbff3cca65",
    "425e23a51a5ed9831defe58faf22095f302ecd06aa9d23138c4f7a8f7271cfb5",
    "45612e513d6a72d66cfb2d99c58541cfb60982d9ab17cd83fed6b5e63e84052c",
    "f54be79027599f9a31fa766b871c804de6ffc5668f79fae0320126dadade3993",
)
EXPECTED_LEGACY_READINESS_CONTENT_BY_DRAFT = {
    "11ad780d8971865541f9094f00a4262ad28677573daacbb2d317d5fbff3cca65": (
        "06bc94f1214e43cd201936f4d331f3d8e148be97881b0d8a4a192ae2b1ed094a"
    ),
    "45612e513d6a72d66cfb2d99c58541cfb60982d9ab17cd83fed6b5e63e84052c": (
        "9d93b5c05448832007a792c52aebe6e8e9598168b140beae78c46a9022368814"
    ),
    "f54be79027599f9a31fa766b871c804de6ffc5668f79fae0320126dadade3993": (
        "81b8025d3c76aa776591fe8ac4d8ac987db0bf64cf1df0163248dfe6fec4c380"
    ),
    "425e23a51a5ed9831defe58faf22095f302ecd06aa9d23138c4f7a8f7271cfb5": (
        "0304a14c8bfc412531aba743675618092e48e6cd22a68ccb6e664620904d79bd"
    ),
}

MAX_LOGICAL_LLM_CALLS = 6
MAX_UNIQUE_EVALUATIONS = 11
LOGICAL_CANDIDATE_OCCURRENCES = 12
EXPECTED_RAW_RECEIPTS = 11
EXPECTED_SOLVER_POINT_CALLS = 33
PARENT_GRID_NONCES = tuple(range(1, 256))

_LOWER_SHA256_CHARS = frozenset("0123456789abcdef")
_MEMBERSHIP_DOMAIN = b"agent-evolve:airfoil-g3-historical-membership:v1\x00"
_CARD_BANK_DOMAIN = b"agent-evolve:airfoil-g3-authenticated-card-bank:v1\x00"
_SOURCE_MANIFEST_DOMAIN = b"agent-evolve:airfoil-g3-source-manifest:v1\x00"
_PARENT_GENERATOR_DOMAIN = b"agent-evolve:airfoil-v7-heldout-parent:v1\x00"
_PARENT_SELECTION_DOMAIN = b"agent-evolve:airfoil-g3-parent-grid:v1\x00"
_CARD_SELECTION_DOMAIN = b"agent-evolve:airfoil-g3-card-selection:v1\x00"
_CARD_SELECTION_RECEIPT_DOMAIN = (
    b"agent-evolve:airfoil-g3-card-selection-receipt:v1\x00"
)
_SHAM_SELECTION_DOMAIN = b"agent-evolve:airfoil-g3-sham-selection:v1\x00"
_MATE_SELECTION_DOMAIN = b"agent-evolve:airfoil-g3-mate-selection:v1\x00"
_RELEASE_DOMAIN = b"agent-evolve:airfoil-g3-provider-free-release:v1\x00"
_FREEZE_RECEIPT_DOMAIN = b"agent-evolve:airfoil-g3-prelaunch-freeze:v1\x00"
_DIAGNOSTIC_PERMUTATION_SELECTION_DOMAIN = (
    b"agent-evolve:airfoil-g3-diagnostic-permutation-selection:v1\x00"
)
_ENDPOINT_DOMAIN = b"agent-evolve:airfoil-g3-absolute-endpoint:v1\x00"
_COMPILER_DOMAIN = b"agent-evolve:airfoil-g3-hypothesis-compiler:v1\x00"
_CONTEXT_DOMAIN = b"agent-evolve:airfoil-g3-memory-context:v1\x00"
_SHAM_DOMAIN = b"agent-evolve:airfoil-g3-neutral-sham:v1\x00"
_UTC_SECONDS = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)

_DIAGNOSTIC_PERMUTATION_DEFINITION = {
    "policy_id": "airfoil_v7_g3_public_diagnostic_permutation",
    "policy_version": 1,
    "support": [0, 1],
    "rank_law": "uint256(public_commitment_sha256) mod 2",
    "public_inputs": [
        "deterministic_release_sha256",
        "canonical_selected_card_reference",
        "selected_card_draft_content_sha256",
        "selected_card_registered_source_evidence_sha256",
    ],
    "current_or_heldout_outcome_access": False,
    "sampling_time": "prelaunch_chronology_freeze",
}
_CONFIGURATION_HASH_KEYS = frozenset(
    {
        "candidate_configuration_sha256",
        "child_configuration_sha256",
        "configuration_hash",
        "configuration_sha256",
        "finite_child_configuration_sha256",
        "finite_parent_configuration_sha256",
        "parent_configuration_hash",
        "parent_configuration_sha256",
        "requested_configuration_hash",
        "target_configuration_hash",
        "typed_child_configuration_sha256",
        "typed_configuration_sha256",
    }
)
_CANDIDATE_HASH_KEYS = frozenset(
    {
        "candidate_hash",
        "candidate_sha256",
        "raw_candidate_sha256",
        "task_candidate_sha256",
    }
)
_AIRFOIL_FIELDS = frozenset(
    {
        "representation_id",
        "upper_coefficients",
        "lower_coefficients",
        "alpha_deg",
    }
)


class AirfoilG3ReleaseError(RuntimeError):
    """A release-critical provider-free invariant failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


DIAGNOSTIC_PERMUTATION_DEFINITION_SHA256 = _hash(
    _DIAGNOSTIC_PERMUTATION_SELECTION_DOMAIN,
    _DIAGNOSTIC_PERMUTATION_DEFINITION,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_LOWER_SHA256_CHARS)
    )


def _workspace_relative(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        return resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError as exc:
        raise AirfoilG3ReleaseError(
            f"release source is outside the workspace: {resolved}"
        ) from exc


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AirfoilG3ReleaseError(f"cannot decode {path}: {exc}") from exc
    if type(value) is not dict:
        raise AirfoilG3ReleaseError(f"{path} must contain one JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: object) -> None:
    """Atomically publish one canonical JSON object without replacement."""

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.tmp-freeze-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise AirfoilG3ReleaseError(
                "prelaunch freeze receipt is write-once and already exists"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SourceFileBinding:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or self.path.startswith("/"):
            raise ValueError("source binding path must be workspace-relative")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("source binding size must be non-negative")
        if not _is_sha256(self.sha256):
            raise ValueError("source binding sha256 is malformed")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class HistoricalMembership:
    source_roots: tuple[str, ...]
    source_files: tuple[SourceFileBinding, ...]
    configuration_sha256s: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
    phenotype_value_sha256s: tuple[str, ...]
    source_manifest_sha256: str
    membership_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_roots != tuple(sorted(set(self.source_roots))):
            raise ValueError("source roots must be unique and canonical")
        if self.source_files != tuple(sorted(self.source_files, key=lambda x: x.path)):
            raise ValueError("source files must be canonically ordered")
        if len({value.path for value in self.source_files}) != len(self.source_files):
            raise ValueError("source file paths must be unique")
        for value in self.source_files:
            value.__post_init__()
        for name in (
            "configuration_sha256s",
            "candidate_sha256s",
            "phenotype_value_sha256s",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))) or any(
                not _is_sha256(value) for value in values
            ):
                raise ValueError(f"{name} must be canonical SHA-256 values")
        expected_manifest = _hash(
            _SOURCE_MANIFEST_DOMAIN,
            {
                "source_roots": list(self.source_roots),
                "source_files": [value.to_record() for value in self.source_files],
            },
        )
        if self.source_manifest_sha256 != expected_manifest:
            raise ValueError("historical source manifest identity changed")
        object.__setattr__(
            self,
            "membership_sha256",
            _hash(_MEMBERSHIP_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": "airfoil_v7_historical_hash_membership",
                "policy_version": 1,
                "json_extensions": [".json", ".jsonl"],
                "candidate_object_detection": "exact_airfoil_fields_projection",
                "explicit_configuration_hash_keys": sorted(
                    _CONFIGURATION_HASH_KEYS
                ),
                "explicit_candidate_hash_keys": sorted(_CANDIDATE_HASH_KEYS),
                "phenotype_policy_id": PHENOTYPE_POLICY_ID,
                "outcome_values_retained": False,
            },
            "source_roots": list(self.source_roots),
            "source_files": [value.to_record() for value in self.source_files],
            "source_manifest_sha256": self.source_manifest_sha256,
            "configuration_sha256s": list(self.configuration_sha256s),
            "candidate_sha256s": list(self.candidate_sha256s),
            "phenotype_value_sha256s": list(self.phenotype_value_sha256s),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "membership_sha256": self.membership_sha256}

    def rejects(
        self,
        *,
        configuration_sha256: str,
        candidate_sha256_value: str,
        phenotype_value_sha256: str,
    ) -> bool:
        self.__post_init__()
        return (
            configuration_sha256 in self.configuration_sha256s
            or candidate_sha256_value in self.candidate_sha256s
            or phenotype_value_sha256 in self.phenotype_value_sha256s
        )


def _parse_json_documents(path: Path, content: bytes) -> Iterable[object]:
    if path.suffix == ".json":
        try:
            yield json.loads(content)
        except json.JSONDecodeError as exc:
            raise AirfoilG3ReleaseError(f"malformed JSON source {path}: {exc}") from exc
        return
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise AirfoilG3ReleaseError(
                f"malformed JSONL source {path}:{line_number}: {exc}"
            ) from exc


def _candidate_projection(value: Mapping[str, object]) -> dict[str, Any] | None:
    if not _AIRFOIL_FIELDS.issubset(value):
        return None
    projection = {name: value[name] for name in _AIRFOIL_FIELDS}
    try:
        return normalize_candidate(projection)
    except (TypeError, ValueError):
        return None


def _collect_membership(
    value: object,
    *,
    configurations: set[str],
    candidates: set[str],
    phenotypes: set[str],
    phenotype_policy: AirfoilV7PhenotypeIdentityPolicy,
) -> None:
    if type(value) is dict:
        projection = _candidate_projection(value)
        if projection is not None:
            frozen = freeze_json(projection)
            if type(frozen) is not FrozenJsonObject:
                raise AssertionError("Airfoil candidate must freeze to an object")
            configurations.add(typed_json_sha256(frozen))
            candidates.add(candidate_sha256(projection))
            phenotypes.add(phenotype_policy.identify(frozen).value_sha256)
        for key, item in value.items():
            if key in _CONFIGURATION_HASH_KEYS and _is_sha256(item):
                configurations.add(item)
            if key in _CANDIDATE_HASH_KEYS and _is_sha256(item):
                candidates.add(item)
            if (
                key == "value_sha256"
                and value.get("policy_id") == PHENOTYPE_POLICY_ID
                and _is_sha256(item)
            ):
                phenotypes.add(item)
            _collect_membership(
                item,
                configurations=configurations,
                candidates=candidates,
                phenotypes=phenotypes,
                phenotype_policy=phenotype_policy,
            )
    elif type(value) is list:
        for item in value:
            _collect_membership(
                item,
                configurations=configurations,
                candidates=candidates,
                phenotypes=phenotypes,
                phenotype_policy=phenotype_policy,
            )


def build_historical_denylist(
    source_roots: Sequence[Path] = DEFAULT_HISTORICAL_SOURCE_ROOTS,
) -> HistoricalMembership:
    """Build the offline hash-only historical authority.

    This is the only membership authority in the release slice allowed to
    traverse historical run data.  The separate card-bank authority may open
    its exact three-file allowlist.  The returned membership intentionally
    contains no objective, reward, validity, rank, timing, response, or
    evaluator outcome.
    """

    if isinstance(source_roots, (str, bytes)) or not isinstance(
        source_roots, Sequence
    ):
        raise TypeError("source_roots must be a sequence of paths")
    resolved_roots = tuple(
        sorted({Path(value).expanduser().resolve(strict=True) for value in source_roots})
    )
    if not resolved_roots:
        raise ValueError("at least one historical source root is required")
    for root in resolved_roots:
        if not root.is_dir():
            raise AirfoilG3ReleaseError(f"historical source root is not a directory: {root}")

    paths: set[Path] = set()
    for root in resolved_roots:
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
    if not paths:
        raise AirfoilG3ReleaseError("historical source roots contain no JSON/JSONL")

    configurations: set[str] = set()
    candidates: set[str] = set()
    phenotypes: set[str] = set()
    bindings: list[SourceFileBinding] = []
    phenotype_policy = AirfoilV7PhenotypeIdentityPolicy()
    for path in sorted(paths):
        if path.is_symlink():
            raise AirfoilG3ReleaseError(f"historical source must not be a symlink: {path}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise AirfoilG3ReleaseError(f"cannot read historical source {path}") from exc
        bindings.append(
            SourceFileBinding(
                path=_workspace_relative(path),
                size_bytes=len(content),
                sha256=_sha256_bytes(content),
            )
        )
        for document in _parse_json_documents(path, content):
            _collect_membership(
                document,
                configurations=configurations,
                candidates=candidates,
                phenotypes=phenotypes,
                phenotype_policy=phenotype_policy,
            )

    roots_record = tuple(_workspace_relative(value) for value in resolved_roots)
    files_record = tuple(sorted(bindings, key=lambda value: value.path))
    source_manifest = _hash(
        _SOURCE_MANIFEST_DOMAIN,
        {
            "source_roots": list(roots_record),
            "source_files": [value.to_record() for value in files_record],
        },
    )
    return HistoricalMembership(
        source_roots=roots_record,
        source_files=files_record,
        configuration_sha256s=tuple(sorted(configurations)),
        candidate_sha256s=tuple(sorted(candidates)),
        phenotype_value_sha256s=tuple(sorted(phenotypes)),
        source_manifest_sha256=source_manifest,
    )


def write_historical_denylist(
    membership: HistoricalMembership,
    path: Path = DEFAULT_DENYLIST_PATH,
) -> str:
    membership.__post_init__()
    _write_json_atomic(path, membership.to_record())
    return _sha256_file(path)


def historical_membership_from_record(value: Mapping[str, object]) -> HistoricalMembership:
    try:
        files = tuple(
            SourceFileBinding(
                path=str(item["path"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in value["source_files"]
            if type(item) is dict
        )
        if len(files) != len(value["source_files"]):
            raise TypeError("one source binding is not an object")
        membership = HistoricalMembership(
            source_roots=tuple(str(item) for item in value["source_roots"]),
            source_files=files,
            configuration_sha256s=tuple(
                str(item) for item in value["configuration_sha256s"]
            ),
            candidate_sha256s=tuple(str(item) for item in value["candidate_sha256s"]),
            phenotype_value_sha256s=tuple(
                str(item) for item in value["phenotype_value_sha256s"]
            ),
            source_manifest_sha256=str(value["source_manifest_sha256"]),
        )
        if value["membership_sha256"] != membership.membership_sha256:
            raise ValueError("historical membership SHA-256 changed")
        return membership
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError("historical membership file is malformed") from exc


def load_historical_denylist(path: Path = DEFAULT_DENYLIST_PATH) -> HistoricalMembership:
    """Load only the sealed hash membership; never traverse its source paths."""

    value = _load_json_object(path.expanduser().resolve(strict=True))
    membership = historical_membership_from_record(value)
    membership.__post_init__()
    return membership


# The no-CFD gate mirrors the frozen external adapter's deterministic geometry
# checks but does not import the broad historical experiment harness.
_INITIAL_AREA = 0.04632803061919573
_AREA_RATIO_BOUNDS = (0.8873697327569672, 1.2)
_RAW_X_BOUNDS = (-1.0e-3, 1.001)
_RAW_CHORD_BOUNDS = (0.99, 1.01)
_PREPROCESSED_Y_BOUNDS = (-0.25, 0.25)
_GEOMETRY_ATOL = 1.0e-10
_MIN_SEGMENT_LENGTH = 1.0e-8
_MIN_AREA = 1.0e-8


@dataclass(frozen=True, slots=True)
class NoCFDGeometryReceipt:
    candidate_sha256: str
    decoded_coords_sha256: str
    area: float
    area_ratio: float
    preprocessed_area: float
    minimum_segment_length: float

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "decoded_coords_sha256": self.decoded_coords_sha256,
            "area_hex": self.area.hex(),
            "area_ratio_hex": self.area_ratio.hex(),
            "preprocessed_area_hex": self.preprocessed_area.hex(),
            "minimum_segment_length_hex": self.minimum_segment_length.hex(),
            "checks": [
                "exact_representation_and_bounds",
                "decoded_shape_and_finiteness",
                "raw_x_range",
                "raw_chord_span",
                "task_area_ratio",
                "preprocessed_y_range",
                "preprocessed_closure",
                "preprocessed_segment_length",
                "preprocessed_no_self_intersection",
                "preprocessed_positive_area",
            ],
        }


def _shoelace(x_values: list[float], y_values: list[float]) -> float:
    return abs(
        sum(
            x_values[index] * y_values[(index + 1) % len(x_values)]
            - y_values[index] * x_values[(index + 1) % len(x_values)]
            for index in range(len(x_values))
        )
    ) / 2.0


def _is_blunted(x_values: list[float], tolerance: float = 1.0e-5) -> bool:
    x_gate = max(x_values) * 0.99
    matches: set[int] = set()
    size = len(x_values)
    for index, value in enumerate(x_values):
        if abs(value - x_values[(index + 1) % size]) < tolerance:
            matches.add(index)
        if abs(value - x_values[(index - 1) % size]) < tolerance:
            matches.add(index)
    return len(tuple(index for index in matches if x_values[index] >= x_gate)) > 1


def _trailing_edge_indices(x_values: list[float], tolerance: float) -> list[int]:
    x_gate = max(x_values) * 0.99
    matches: set[int] = set()
    size = len(x_values)
    for index, value in enumerate(x_values):
        if abs(value - x_values[(index + 1) % size]) < tolerance:
            matches.add(index)
        if abs(value - x_values[(index - 1) % size]) < tolerance:
            matches.add(index)
    return sorted(index for index in matches if x_values[index] >= x_gate)


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    return left[0] - right[0], left[1] - right[1]


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if abs(_cross(_subtract(end, start), _subtract(point, start))) > _GEOMETRY_ATOL:
        return False
    return all(
        min(start[axis], end[axis]) - _GEOMETRY_ATOL
        <= point[axis]
        <= max(start[axis], end[axis]) + _GEOMETRY_ATOL
        for axis in (0, 1)
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_direction = _subtract(first_end, first_start)
    second_direction = _subtract(second_end, second_start)
    o1 = _cross(first_direction, _subtract(second_start, first_start))
    o2 = _cross(first_direction, _subtract(second_end, first_start))
    o3 = _cross(second_direction, _subtract(first_start, second_start))
    o4 = _cross(second_direction, _subtract(first_end, second_start))
    first_straddles = (o1 > _GEOMETRY_ATOL and o2 < -_GEOMETRY_ATOL) or (
        o1 < -_GEOMETRY_ATOL and o2 > _GEOMETRY_ATOL
    )
    second_straddles = (o3 > _GEOMETRY_ATOL and o4 < -_GEOMETRY_ATOL) or (
        o3 < -_GEOMETRY_ATOL and o4 > _GEOMETRY_ATOL
    )
    return (
        (first_straddles and second_straddles)
        or abs(o1) <= _GEOMETRY_ATOL
        and _point_on_segment(second_start, first_start, first_end)
        or abs(o2) <= _GEOMETRY_ATOL
        and _point_on_segment(second_end, first_start, first_end)
        or abs(o3) <= _GEOMETRY_ATOL
        and _point_on_segment(first_start, second_start, second_end)
        or abs(o4) <= _GEOMETRY_ATOL
        and _point_on_segment(first_end, second_start, second_end)
    )


def _first_self_intersection(
    points: list[tuple[float, float]],
) -> tuple[int, int] | None:
    if math.dist(points[0], points[-1]) <= _GEOMETRY_ATOL:
        points = points[:-1]
    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue
            if _segments_intersect(
                points[first],
                points[(first + 1) % count],
                points[second],
                points[(second + 1) % count],
            ):
                return first, second
    return None


def validate_no_cfd_geometry(configuration: object) -> NoCFDGeometryReceipt:
    try:
        candidate_value = thaw_json(configuration)
    except TypeError:
        candidate_value = configuration
    candidate = normalize_candidate(candidate_value)
    raw = decoded_float32_le_bytes(candidate)
    values = struct.unpack("<384f", raw)
    x_values = list(values[:192])
    y_values = list(values[192:])
    if not all(math.isfinite(value) for value in values):
        raise AirfoilG3ReleaseError("decoded coordinates are non-finite")
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min < _RAW_X_BOUNDS[0] or x_max > _RAW_X_BOUNDS[1]:
        raise AirfoilG3ReleaseError("raw x range is outside adapter bounds")
    chord = x_max - x_min
    if not _RAW_CHORD_BOUNDS[0] <= chord <= _RAW_CHORD_BOUNDS[1]:
        raise AirfoilG3ReleaseError("raw chord is outside adapter bounds")
    area = _shoelace(x_values, y_values)
    area_ratio = area / _INITIAL_AREA
    if not _AREA_RATIO_BOUNDS[0] <= area_ratio <= _AREA_RATIO_BOUNDS[1]:
        raise AirfoilG3ReleaseError("decoded task area ratio is outside bounds")

    blunted = _is_blunted(x_values)
    leading = min(range(len(x_values)), key=x_values.__getitem__)
    xcut = 0.99 if blunted else 1.0
    processed_x = [xcut * (value - x_min) / chord for value in x_values]
    processed_y = [value - y_values[leading] for value in y_values]
    processed_x[0] = xcut
    processed_x[-1] = xcut
    processed_y[-1] = processed_y[0]
    if blunted:
        trailing = _trailing_edge_indices(processed_x, 1.0e-5)
        tolerance = 1.0e-4
        while len(trailing) < 6:
            trailing = _trailing_edge_indices(processed_x, tolerance)
            tolerance *= 1.5
            if tolerance > 1.0e-3:
                break
        if set(trailing[1:-1]):
            raise AirfoilG3ReleaseError("blunted preprocessing changes cardinality")
    if (
        min(processed_y) < _PREPROCESSED_Y_BOUNDS[0]
        or max(processed_y) > _PREPROCESSED_Y_BOUNDS[1]
    ):
        raise AirfoilG3ReleaseError("preprocessed y range is outside bounds")
    points = list(zip(processed_x, processed_y, strict=True))
    if math.dist(points[0], points[-1]) > _GEOMETRY_ATOL:
        raise AirfoilG3ReleaseError("preprocessed curve is not closed")
    minimum_segment = min(
        math.dist(points[index], points[index + 1])
        for index in range(len(points) - 1)
    )
    if minimum_segment < _MIN_SEGMENT_LENGTH:
        raise AirfoilG3ReleaseError("preprocessed curve has a degenerate segment")
    intersection = _first_self_intersection(points)
    if intersection is not None:
        raise AirfoilG3ReleaseError(
            f"preprocessed curve self-intersects at {intersection}"
        )
    processed_area = _shoelace(processed_x, processed_y)
    if not math.isfinite(processed_area) or processed_area <= _MIN_AREA:
        raise AirfoilG3ReleaseError("preprocessed area is nonpositive")
    return NoCFDGeometryReceipt(
        candidate_sha256=candidate_sha256(candidate),
        decoded_coords_sha256=_sha256_bytes(raw),
        area=area,
        area_ratio=area_ratio,
        preprocessed_area=processed_area,
        minimum_segment_length=minimum_segment,
    )


_ENDPOINT_DEFINITION = {
    "endpoint_id": "airfoil_v7_absolute_bounded_violation_first",
    "endpoint_version": 1,
    "larger_is_better": True,
    "valid_formula": "-V/(1+V) + 0.001*tanh((1-f)/0.001)",
    "f": "normalized_multipoint_drag",
    "V": "normalized_lift_equality",
    "required_valid_domain": {"f": "finite", "V": "finite_and_nonnegative"},
    "invalid_noncompliant_or_no_yield": -2.0,
    "valid_range": "(-1.001, 0.001]",
    "parent_independent": True,
    "operator_scope": [
        "seed",
        "typed_mutation",
        "engine_mate",
        "reproduction",
        "three_way_recombination",
    ],
    "drag_scale": DELTA_F,
    "violation_priority_rationale": (
        "At V approximately 0.5, the complete bounded drag term can offset "
        "only about 0.0045 change in V, matching the frozen DELTA_V resolution."
    ),
    "raw_vector_retained": True,
    "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
}
ABSOLUTE_Q_DEFINITION_SHA256 = _hash(_ENDPOINT_DOMAIN, _ENDPOINT_DEFINITION)


def absolute_airfoil_q(
    *,
    normalized_multipoint_drag: object,
    normalized_lift_equality: object,
    valid: bool,
) -> float:
    """Return the total, bounded, absolute Airfoil G3 endpoint."""

    if type(valid) is not bool:
        raise TypeError("valid must be an exact bool")
    if not valid:
        return -2.0
    if isinstance(normalized_multipoint_drag, bool) or isinstance(
        normalized_lift_equality, bool
    ):
        return -2.0
    try:
        drag = float(normalized_multipoint_drag)
        violation = float(normalized_lift_equality)
    except (TypeError, ValueError, OverflowError):
        return -2.0
    if not math.isfinite(drag) or not math.isfinite(violation) or violation < 0.0:
        return -2.0
    score = -violation / (1.0 + violation) + 0.001 * math.tanh(
        (1.0 - drag) / DELTA_F
    )
    if not math.isfinite(score) or not -1.001 < score <= 0.001:
        raise AssertionError("absolute Airfoil Q escaped its proved valid range")
    return score


def airfoil_g3_absolute_reward(
    child: EvolutionCandidate,
    parents: tuple[EvolutionCandidate, ...],
    objectives: Sequence[object],
) -> float:
    """Engine binding for Q; ``parents`` are intentionally semantically unused."""

    del parents
    declarations = tuple(
        (getattr(value, "name", None), getattr(value, "goal", None))
        for value in objectives
    )
    if declarations != (("normalized_multipoint_drag", "min"),):
        raise ValueError("Airfoil G3 absolute Q received the wrong objective declaration")
    if type(child) is not EvolutionCandidate:
        raise TypeError("child must be an exact EvolutionCandidate")
    if not child.valid or not child.operator_compliant or not child.evidence_compliant:
        return -2.0
    detailed = child.detailed_evaluation
    if detailed is None or not detailed.success:
        return -2.0
    objectives_by_name = dict(detailed.objectives)
    violations_by_name = dict(detailed.violations)
    if set(objectives_by_name) != {"normalized_multipoint_drag"} or set(
        violations_by_name
    ) != {"normalized_lift_equality"}:
        raise ValueError("Airfoil G3 absolute Q detailed metric identity changed")
    return absolute_airfoil_q(
        normalized_multipoint_drag=objectives_by_name["normalized_multipoint_drag"],
        normalized_lift_equality=violations_by_name["normalized_lift_equality"],
        valid=True,
    )


AIRFOIL_G3_ABSOLUTE_REWARD = RewardPolicyBinding(
    score=airfoil_g3_absolute_reward,
    definition_hash=ABSOLUTE_Q_DEFINITION_SHA256,
    failure_score=-2.0,
)


AIRFOIL_G3_RUNTIME_PROBLEM_ID = (
    "examples.benchmarks.engibench_airfoil.v7_problem_def.AirfoilV7Problem"
)
AIRFOIL_G3_RUNTIME_PHASE = "g3_causal_screen"
_CONTEXT_DEFINITION = {
    "context_id": "airfoil_v7_g3_trim_transfer",
    "context_version": 1,
    "benchmark": "airfoil_v7",
    "runtime_problem_id": AIRFOIL_G3_RUNTIME_PROBLEM_ID,
    "runtime_phase": AIRFOIL_G3_RUNTIME_PHASE,
    "operator": OperatorKind.TYPED_MUTATION.value,
    "action_family": "trim_only",
    "evaluator_semantics": "airfoil_v7_convergence_projection",
    "parent_identity_retained_separately": True,
    "diagnostic_and_heldout_share_score_stratum": True,
}
CONTEXT_DEFINITION_SHA256 = _hash(_CONTEXT_DOMAIN, _CONTEXT_DEFINITION)
CONTEXT_PROJECTION_SHA256 = context_stratum_hash(
    problem_id=str(_CONTEXT_DEFINITION["runtime_problem_id"]),
    operator_kind=OperatorKind.TYPED_MUTATION.value,
    phase=str(_CONTEXT_DEFINITION["runtime_phase"]),
)


def _draft_from_record(value: object) -> InsightDraft:
    if type(value) is not dict:
        raise AirfoilG3ReleaseError("historical reflection draft is malformed")
    try:
        raw_predictions = value["effect_predictions"]
        if type(raw_predictions) is not list:
            raise TypeError("effect_predictions must be a list")
        predictions = tuple(
            MetricEffectPrediction(
                metric_id=str(item["metric_id"]),
                direction=MetricEffectDirection(str(item["direction"])),
            )
            for item in raw_predictions
            if type(item) is dict
        )
        if len(predictions) != len(raw_predictions):
            raise TypeError("one effect prediction is not an object")
        return InsightDraft(
            claim=str(value["claim"]),
            trigger=str(value["trigger"]),
            mechanism=str(value["mechanism"]),
            affected_paths=tuple(str(item) for item in value["affected_paths"]),
            evidence_summary=str(value["evidence_summary"]),
            confidence=float(value["confidence"]),
            evidence_contrast_ids=tuple(
                str(item) for item in value["evidence_contrast_ids"]
            ),
            effect_predictions=predictions,
            recommended_option_families=tuple(
                str(item) for item in value["recommended_option_families"]
            ),
            recommended_option_ids=tuple(
                str(item) for item in value["recommended_option_ids"]
            ),
            action_template=str(value["action_template"]),
            falsification_condition=str(value["falsification_condition"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError(
            "historical reflection draft failed typed reconstruction"
        ) from exc


def _finite_binding_from_record(value: object) -> FiniteActionEvidenceBinding:
    if type(value) is not dict:
        raise AirfoilG3ReleaseError("historical finite action binding is malformed")
    try:
        binding = FiniteActionEvidenceBinding(
            contrast_id=str(value["contrast_id"]),
            option_id=str(value["option_id"]),
            family=str(value["family"]),
            option_identity_sha256=str(value["option_identity_sha256"]),
            contract_identity_sha256=str(value["contract_identity_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError(
            "historical finite action binding failed reconstruction"
        ) from exc
    if value.get("binding_identity_sha256") != binding.identity_sha256:
        raise AirfoilG3ReleaseError("historical finite action binding hash changed")
    return binding


def _empirical_snapshot_from_record(value: object) -> EmpiricalEvidenceSnapshot:
    if type(value) is not dict:
        raise AirfoilG3ReleaseError("historical empirical snapshot is malformed")
    try:
        facts = freeze_json(value["facts"])
        if type(facts) is not FrozenJsonObject:
            raise TypeError("empirical facts must freeze to an object")
        snapshot = EmpiricalEvidenceSnapshot(
            contrast_id=str(value["contrast_id"]),
            fact_schema_id=str(value["fact_schema_id"]),
            fact_schema_version=int(value["fact_schema_version"]),
            fact_schema_definition_sha256=str(
                value["fact_schema_definition_sha256"]
            ),
            facts=facts,
            optimization_semantics_definition_sha256=(
                None
                if value.get("optimization_semantics_definition_sha256") is None
                else str(value["optimization_semantics_definition_sha256"])
            ),
            action_semantics_definition_sha256=(
                None
                if value.get("action_semantics_definition_sha256") is None
                else str(value["action_semantics_definition_sha256"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError(
            "historical empirical snapshot failed reconstruction"
        ) from exc
    if value.get("snapshot_sha256") != snapshot.snapshot_sha256:
        raise AirfoilG3ReleaseError("historical empirical snapshot hash changed")
    return snapshot


def _lineage_from_record(value: object) -> InsightEvidenceLineage:
    if type(value) is not dict:
        raise AirfoilG3ReleaseError("historical insight lineage is malformed")
    try:
        lineage = InsightEvidenceLineage(
            reflection_call_id=LLMCallId(str(value["reflection_call_id"])),
            source_operator_invocation_ids=tuple(
                OperatorInvocationId(str(item))
                for item in value["source_operator_invocation_ids"]
            ),
            source_candidate_ids=tuple(
                CandidateId(str(item)) for item in value["source_candidate_ids"]
            ),
            available_contrast_ids=tuple(
                str(item) for item in value["available_contrast_ids"]
            ),
            cited_contrast_ids=tuple(
                str(item) for item in value["cited_contrast_ids"]
            ),
            finite_action_bindings=tuple(
                _finite_binding_from_record(item)
                for item in value["finite_action_bindings"]
            ),
            empirical_evidence=tuple(
                _empirical_snapshot_from_record(item)
                for item in value["empirical_evidence"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError(
            "historical insight lineage failed typed reconstruction"
        ) from exc
    if value.get("lineage_identity_sha256") != lineage.identity_sha256:
        raise AirfoilG3ReleaseError("historical insight lineage hash changed")
    if value != lineage.to_record():
        raise AirfoilG3ReleaseError("historical insight lineage record changed")
    return lineage


def reconstruct_authenticated_trim_entries(
    run_dir: Path = HISTORICAL_RUN,
) -> tuple[InsightMemoryEntry, ...]:
    """Reconstruct the four exact accepted DeepSeek trim hypotheses.

    The historical run's terminal status was transport-incomplete *after* the
    accepted/finalized reflection.  This function authenticates that status and
    the exact reflection/readiness bytes rather than upgrading the run's claim.
    """

    root = run_dir.expanduser().resolve(strict=True)
    paths = {
        "reflection": root / "reflection_result.json",
        "readiness": root / "provider_free_readiness.json",
        "finalized": root / "finalized.json",
    }
    expected_files = {
        "reflection": EXPECTED_REFLECTION_RESULT_SHA256,
        "readiness": EXPECTED_PROVIDER_FREE_READINESS_SHA256,
        "finalized": EXPECTED_FINALIZED_FILE_SHA256,
    }
    for name, path in paths.items():
        if _sha256_file(path) != expected_files[name]:
            raise AirfoilG3ReleaseError(f"frozen historical {name} bytes changed")

    finalized = _load_json_object(paths["finalized"])
    if (
        finalized.get("status") != EXPECTED_HISTORICAL_STATUS
        or finalized.get("finalization_sha256") != EXPECTED_FINALIZATION_SHA256
        or finalized.get("recursive_content_sha256")
        != EXPECTED_RECURSIVE_CONTENT_SHA256
    ):
        raise AirfoilG3ReleaseError("frozen historical finalization changed")
    finalized_files = finalized.get("files")
    if type(finalized_files) is not dict:
        raise AirfoilG3ReleaseError("frozen finalization lacks a file manifest")
    for filename, expected in (
        ("reflection_result.json", EXPECTED_REFLECTION_RESULT_SHA256),
        ("provider_free_readiness.json", EXPECTED_PROVIDER_FREE_READINESS_SHA256),
    ):
        binding = finalized_files.get(filename)
        if type(binding) is not dict or binding.get("sha256") != expected:
            raise AirfoilG3ReleaseError(
                f"frozen finalization does not bind {filename}"
            )

    reflection = _load_json_object(paths["reflection"])
    if (
        reflection.get("schema_version") != 1
        or reflection.get("logical_calls_used") != 1
        or reflection.get("logical_call_ids") != ["call_ae7x4v2_000001"]
    ):
        raise AirfoilG3ReleaseError("historical reflection call identity changed")
    raw_cards = reflection.get("cards")
    if type(raw_cards) is not list or len(raw_cards) != 8:
        raise AirfoilG3ReleaseError("historical reflection must contain eight cards")
    drafts_by_contrast: dict[str, InsightDraft] = {}
    for row in raw_cards:
        if type(row) is not dict:
            raise AirfoilG3ReleaseError("historical reflection card is malformed")
        draft = _draft_from_record(row.get("draft"))
        if row.get("draft_content_sha256") != draft.content_sha256:
            raise AirfoilG3ReleaseError("historical reflection draft hash changed")
        if row.get("call_id") != "call_ae7x4v2_000001":
            raise AirfoilG3ReleaseError("historical reflection card call changed")
        contrast_id = str(row.get("contrast_id"))
        if contrast_id in drafts_by_contrast:
            raise AirfoilG3ReleaseError("historical reflection contrast repeats")
        drafts_by_contrast[contrast_id] = draft

    readiness = _load_json_object(paths["readiness"])
    try:
        raw_entries = readiness["call_preview"]["arms"]["entries"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise AirfoilG3ReleaseError("historical readiness entry bank is missing") from exc
    if type(raw_entries) is not list or len(raw_entries) != 8:
        raise AirfoilG3ReleaseError("historical readiness must bind eight entries")

    trim_entries: list[InsightMemoryEntry] = []
    for row in raw_entries:
        if type(row) is not dict or type(row.get("reference")) is not dict:
            raise AirfoilG3ReleaseError("historical readiness entry is malformed")
        lineage = _lineage_from_record(row.get("lineage"))
        if len(lineage.cited_contrast_ids) != 1:
            raise AirfoilG3ReleaseError("historical card must cite one contrast")
        contrast_id = lineage.cited_contrast_ids[0]
        try:
            draft = drafts_by_contrast[contrast_id]
            reference_record = row["reference"]
            reference = InsightRef(
                InsightId(str(reference_record["insight_id"])),
                int(reference_record["version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AirfoilG3ReleaseError(
                "historical entry reference/draft binding failed"
            ) from exc
        legacy_content = row.get("content_sha256")
        expected_legacy = EXPECTED_LEGACY_READINESS_CONTENT_BY_DRAFT.get(
            draft.content_sha256
        )
        if draft.recommended_option_families == ("trim_only",) and (
            expected_legacy is None or legacy_content != expected_legacy
        ):
            raise AirfoilG3ReleaseError(
                "historical readiness legacy content binding changed"
            )
        entry = InsightMemoryEntry(
            reference=reference,
            draft=draft,
            initial_score=0.0,
            applicable_operator_kinds=("mutation",),
            lifecycle_state=InsightLifecycleState.QUARANTINED,
            origin=InsightOrigin.REFLECTION,
            evidence_lineage=lineage,
            relations=(),
        )
        entry.__post_init__()
        if draft.recommended_option_families == ("trim_only",):
            if (
                draft.affected_paths != TRIM_PATHS
                or len(draft.recommended_option_ids) != 1
                or len(lineage.finite_action_bindings) != 1
                or lineage.finite_action_bindings[0].option_id
                != draft.recommended_option_ids[0]
            ):
                raise AirfoilG3ReleaseError(
                    "historical trim hypothesis lost its exact treatment"
                )
            trim_entries.append(entry)
    result = tuple(sorted(trim_entries, key=lambda value: value.reference))
    if len(result) != 4 or tuple(
        sorted(value.draft.content_sha256 for value in result)
    ) != tuple(sorted(EXPECTED_TRIM_DRAFT_SHA256S)):
        raise AirfoilG3ReleaseError("exact four-card trim bank changed")
    return result


@dataclass(frozen=True, slots=True)
class AuthenticatedTrimCardBank:
    """Offline export of the only historical memory allowed into G3 prep."""

    source_files: tuple[SourceFileBinding, ...]
    entries: tuple[InsightMemoryEntry, ...]
    card_bank_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_files != tuple(sorted(self.source_files, key=lambda x: x.path)):
            raise ValueError("card-bank source files must be canonical")
        for value in self.source_files:
            value.__post_init__()
        source_identity = tuple(
            (value.path, value.size_bytes, value.sha256) for value in self.source_files
        )
        if source_identity != EXPECTED_CARD_SOURCE_FILES:
            raise ValueError("card bank differs from its exact source-file allowlist")
        if len(self.entries) != 4 or self.entries != tuple(
            sorted(self.entries, key=lambda value: value.reference)
        ):
            raise ValueError("card bank must contain four canonical trim entries")
        if len({value.reference for value in self.entries}) != 4:
            raise ValueError("card-bank references must be unique")
        for entry in self.entries:
            entry.__post_init__()
            lineage = entry.evidence_lineage
            if (
                entry.origin is not InsightOrigin.REFLECTION
                or entry.lifecycle_state is not InsightLifecycleState.QUARANTINED
                or entry.applicable_operator_kinds != ("mutation",)
                or lineage is None
                or entry.draft.recommended_option_families != ("trim_only",)
                or entry.draft.content_sha256 not in EXPECTED_TRIM_DRAFT_SHA256S
            ):
                raise ValueError("card bank contains a non-authenticated trim entry")
        card_identities = {
            entry.reference.insight_id.value: (
                entry.reference.version,
                entry.draft.content_sha256,
                entry.evidence_lineage.identity_sha256,  # type: ignore[union-attr]
                registered_source_evidence_sha256(entry),
            )
            for entry in self.entries
        }
        if card_identities != EXPECTED_AUTHENTICATED_CARD_IDENTITIES:
            raise ValueError("card bank differs from the preregistered card identities")
        object.__setattr__(
            self,
            "card_bank_sha256",
            _hash(_CARD_BANK_DOMAIN, self._identity_record()),
        )
        if self.card_bank_sha256 != EXPECTED_CARD_BANK_SHA256:
            raise ValueError("card bank differs from its preregistered identity")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "airfoil_v7_authenticated_trim_card_bank",
            "policy": {
                "policy_id": "airfoil_v7_frozen_v2_trim_card_export",
                "policy_version": 1,
                "historical_memory_is_development_input": True,
                "card_hypotheses_are_not_engine_truth": True,
                "source_scalar_observations_are_historical_not_current_run": True,
                "allowed_entry_count": 4,
            },
            "source_run": {
                "path": _workspace_relative(HISTORICAL_RUN),
                "status": EXPECTED_HISTORICAL_STATUS,
                "finalization_sha256": EXPECTED_FINALIZATION_SHA256,
                "recursive_content_sha256": EXPECTED_RECURSIVE_CONTENT_SHA256,
            },
            "source_files": [value.to_record() for value in self.source_files],
            "entries": [
                {
                    "reference": {
                        "insight_id": entry.reference.insight_id.value,
                        "version": entry.reference.version,
                    },
                    "draft": entry.draft.content_record(),
                    "draft_content_sha256": entry.draft.content_sha256,
                    "legacy_readiness_content_sha256": (
                        EXPECTED_LEGACY_READINESS_CONTENT_BY_DRAFT[
                            entry.draft.content_sha256
                        ]
                    ),
                    "initial_score_hex": entry.initial_score.hex(),
                    "applicable_operator_kinds": list(
                        entry.applicable_operator_kinds
                    ),
                    "lifecycle_state": entry.lifecycle_state.value,
                    "origin": entry.origin.value,
                    "evidence_lineage": entry.evidence_lineage.to_record(),  # type: ignore[union-attr]
                    "relations": [],
                    "registered_source_evidence_sha256": (
                        registered_source_evidence_sha256(entry)
                    ),
                }
                for entry in self.entries
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "card_bank_sha256": self.card_bank_sha256}


def build_authenticated_trim_card_bank(
    run_dir: Path = HISTORICAL_RUN,
) -> AuthenticatedTrimCardBank:
    """Offline authority: authenticate and export the exact typed card bank."""

    root = run_dir.expanduser().resolve(strict=True)
    bindings = tuple(
        sorted(
            (
                SourceFileBinding(
                    path=_workspace_relative(root / filename),
                    size_bytes=(root / filename).stat().st_size,
                    sha256=_sha256_file(root / filename),
                )
                for filename in (
                    "finalized.json",
                    "provider_free_readiness.json",
                    "reflection_result.json",
                )
            ),
            key=lambda value: value.path,
        )
    )
    return AuthenticatedTrimCardBank(
        source_files=bindings,
        entries=reconstruct_authenticated_trim_entries(root),
    )


def write_authenticated_trim_card_bank(
    card_bank: AuthenticatedTrimCardBank,
    path: Path = DEFAULT_CARD_BANK_PATH,
) -> str:
    card_bank.__post_init__()
    _write_json_atomic(path, card_bank.to_record())
    return _sha256_file(path)


def _entry_from_card_bank_record(value: object) -> InsightMemoryEntry:
    if type(value) is not dict or type(value.get("reference")) is not dict:
        raise AirfoilG3ReleaseError("card-bank entry is malformed")
    try:
        reference_record = value["reference"]
        entry = InsightMemoryEntry(
            reference=InsightRef(
                InsightId(str(reference_record["insight_id"])),
                int(reference_record["version"]),
            ),
            draft=_draft_from_record(value["draft"]),
            initial_score=float.fromhex(str(value["initial_score_hex"])),
            applicable_operator_kinds=tuple(
                str(item) for item in value["applicable_operator_kinds"]
            ),
            lifecycle_state=InsightLifecycleState(str(value["lifecycle_state"])),
            origin=InsightOrigin(str(value["origin"])),
            evidence_lineage=_lineage_from_record(value["evidence_lineage"]),
            relations=(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError(
            "card-bank entry failed typed reconstruction"
        ) from exc
    if (
        value.get("draft_content_sha256") != entry.draft.content_sha256
        or value.get("legacy_readiness_content_sha256")
        != EXPECTED_LEGACY_READINESS_CONTENT_BY_DRAFT.get(
            entry.draft.content_sha256
        )
        or value.get("registered_source_evidence_sha256")
        != registered_source_evidence_sha256(entry)
        or value.get("relations") != []
    ):
        raise AirfoilG3ReleaseError("card-bank entry identity changed")
    return entry


def load_authenticated_trim_card_bank(
    path: Path = DEFAULT_CARD_BANK_PATH,
) -> AuthenticatedTrimCardBank:
    """Load the sealed typed card export without opening its source run."""

    record = _load_json_object(path.expanduser().resolve(strict=True))
    try:
        raw_sources = record["source_files"]
        raw_entries = record["entries"]
        if type(raw_sources) is not list or type(raw_entries) is not list:
            raise TypeError("card-bank lists are malformed")
        source_files = tuple(
            SourceFileBinding(
                path=str(item["path"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in raw_sources
            if type(item) is dict
        )
        if len(source_files) != len(raw_sources):
            raise TypeError("one card-bank source is not an object")
        bank = AuthenticatedTrimCardBank(
            source_files=source_files,
            entries=tuple(_entry_from_card_bank_record(item) for item in raw_entries),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError("authenticated card-bank file is malformed") from exc
    if record.get("card_bank_sha256") != bank.card_bank_sha256:
        raise AirfoilG3ReleaseError("authenticated card-bank SHA-256 changed")
    if record != bank.to_record():
        raise AirfoilG3ReleaseError("authenticated card-bank record changed")
    return bank


_COMPILER_DEFINITION = {
    "policy_id": "airfoil_v7_trim_template_compiler",
    "policy_version": 1,
    "source_operator_projection": {
        "source_exact": ["mutation"],
        "executable_exact": [OperatorKind.TYPED_MUTATION.value],
        "source_is_not_rewritten": True,
    },
    "required_family": "trim_only",
    "required_affected_paths": list(TRIM_PATHS),
    "allowed_action_cardinality": 1,
    "action_binding": "exact_recommended_template_in_current_parent_contract",
    "held_fixed_paths": list(SHAPE_HELD_FIXED_PATHS),
    "outcome_access": False,
}
COMPILER_DEFINITION_SHA256 = _hash(_COMPILER_DOMAIN, _COMPILER_DEFINITION)


class AirfoilV7TrimHypothesisCompiler:
    """Bind a portable historical trim template to one exact current parent."""

    policy_id = "airfoil_v7_trim_template_compiler"
    policy_version = 1
    definition_sha256 = COMPILER_DEFINITION_SHA256

    def _inapplicable(
        self,
        request: HypothesisCompilationRequest,
        *reason_codes: str,
    ) -> HypothesisCompilationReceipt:
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.INAPPLICABLE,
            reason_codes=tuple(sorted(set(reason_codes))),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=None,
        )

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        if type(request) is not HypothesisCompilationRequest:
            raise TypeError("request must be an exact HypothesisCompilationRequest")
        request.__post_init__()
        reasons: list[str] = []
        if request.requested_operator_kind != OperatorKind.TYPED_MUTATION.value:
            reasons.append("foreign_executable_operator")
        if request.source_operator_kinds != ("mutation",):
            reasons.append("foreign_source_operator_scope")
        insight = request.insight
        if insight.recommended_option_families != ("trim_only",):
            reasons.append("foreign_option_family")
        if tuple(sorted(insight.affected_paths)) != TRIM_PATHS:
            reasons.append("foreign_affected_paths")
        if tuple(value.metric_id for value in insight.effect_predictions) != (
            REQUIRED_METRIC_IDS
        ):
            reasons.append("foreign_metric_scope")
        if len(insight.recommended_option_ids) != 1:
            reasons.append("non_singleton_template")
        option = None
        if len(insight.recommended_option_ids) == 1:
            try:
                option = request.finite_contract.resolve(
                    insight.recommended_option_ids[0]
                )
            except ValueError:
                reasons.append("template_absent_from_parent")
            else:
                if option.family != "trim_only":
                    reasons.append("template_resolves_to_foreign_family")
        if reasons:
            return self._inapplicable(request, *reasons)
        assert option is not None
        spec = ExecutableHypothesisTestSpec(
            request_sha256=request.request_sha256,
            reference=request.reference,
            insight_content_sha256=insight.content_sha256,
            source_evidence_sha256=request.source_evidence_sha256,
            requested_operator_kind=request.requested_operator_kind,
            source_operator_kinds=request.source_operator_kinds,
            executable_operator_kinds=(request.requested_operator_kind,),
            parent_candidate_id=request.parent_candidate_id,
            parent_configuration_sha256=request.parent_configuration_sha256,
            finite_contract_sha256=request.finite_contract.identity_sha256,
            context_projection_sha256=request.context_projection_sha256,
            endpoint_definition_sha256=request.endpoint_definition_sha256,
            allowed_actions=(
                TreatmentActionBinding(option.option_id, option.identity_sha256),
            ),
            recommended_option_families=insight.recommended_option_families,
            affected_paths=TRIM_PATHS,
            held_fixed_paths=SHAPE_HELD_FIXED_PATHS,
            effect_predictions=insight.effect_predictions,
            falsification_condition=str(insight.falsification_condition),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )
        receipt = HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.APPLICABLE,
            reason_codes=(),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=spec,
        )
        validate_hypothesis_compiler_identity(self, receipt)
        validate_hypothesis_compilation(request, receipt)
        return receipt


@dataclass(frozen=True, slots=True)
class CandidateMaterialization:
    label: str
    configuration: FrozenJsonObject
    configuration_sha256: str
    candidate_sha256: str
    phenotype_value_sha256: str
    geometry: NoCFDGeometryReceipt

    @classmethod
    def from_configuration(
        cls, label: str, configuration: object
    ) -> "CandidateMaterialization":
        try:
            candidate_value = thaw_json(configuration)
        except TypeError:
            candidate_value = configuration
        candidate = normalize_candidate(candidate_value)
        frozen = freeze_json(candidate)
        if type(frozen) is not FrozenJsonObject:
            raise AssertionError("Airfoil candidate must freeze to an object")
        return cls(
            label=label,
            configuration=frozen,
            configuration_sha256=typed_json_sha256(frozen),
            candidate_sha256=candidate_sha256(candidate),
            phenotype_value_sha256=AirfoilV7PhenotypeIdentityPolicy()
            .identify(frozen)
            .value_sha256,
            geometry=validate_no_cfd_geometry(frozen),
        )

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("candidate materialization label must be non-empty")
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("candidate configuration must be frozen typed JSON")
        if typed_json_sha256(self.configuration) != self.configuration_sha256:
            raise ValueError("candidate materialization configuration hash changed")
        candidate = normalize_candidate(thaw_json(self.configuration))
        if candidate_sha256(candidate) != self.candidate_sha256:
            raise ValueError("candidate materialization task hash changed")
        if (
            AirfoilV7PhenotypeIdentityPolicy()
            .identify(self.configuration)
            .value_sha256
            != self.phenotype_value_sha256
        ):
            raise ValueError("candidate materialization phenotype changed")

    def to_record(self, *, include_configuration: bool = True) -> dict[str, object]:
        self.__post_init__()
        return {
            "label": self.label,
            "configuration": (
                thaw_json(self.configuration) if include_configuration else None
            ),
            "configuration_sha256": self.configuration_sha256,
            "candidate_sha256": self.candidate_sha256,
            "phenotype_value_sha256": self.phenotype_value_sha256,
            "geometry": self.geometry.to_record(),
        }


@dataclass(frozen=True, slots=True)
class SelectedParent:
    role: str
    nonce: int
    selection_sha256: str
    candidate: CandidateMaterialization

    def to_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "nonce": self.nonce,
            "selection_sha256": self.selection_sha256,
            "candidate": self.candidate.to_record(),
        }


def _parent_sign(nonce: int, field: str, index: int) -> float:
    digest = hashlib.sha256(
        _PARENT_GENERATOR_DOMAIN
        + TASK_SHA256.encode("ascii")
        + b"\x00"
        + str(nonce).encode("ascii")
        + b"\x00"
        + field.encode("ascii")
        + b"\x00"
        + str(index).encode("ascii")
    ).digest()
    return 1.0 if digest[0] & 1 else -1.0


def parent_grid_candidate(nonce: int) -> dict[str, object]:
    """Outcome-blind, near-neutral, all-real-field parent grid."""

    if type(nonce) is not int or nonce not in PARENT_GRID_NONCES:
        raise ValueError("nonce is outside the frozen parent grid")
    upper = [0.0] * 10
    lower = [0.0] * 10
    for index in range(1, 9):
        upper[index] = _parent_sign(nonce, "upper", index) * 0.0015
        lower[index] = _parent_sign(nonce, "lower", index) * 0.0015
    alpha = [2.5 + _parent_sign(nonce, "alpha", index) * 0.25 for index in range(3)]
    return normalize_candidate(
        {
            "representation_id": REPRESENTATION_ID,
            "upper_coefficients": upper,
            "lower_coefficients": lower,
            "alpha_deg": alpha,
        }
    )


def select_parents(membership: HistoricalMembership) -> tuple[SelectedParent, SelectedParent]:
    membership.__post_init__()
    eligible: list[tuple[str, int, CandidateMaterialization]] = []
    for nonce in PARENT_GRID_NONCES:
        try:
            materialized = CandidateMaterialization.from_configuration(
                f"parent_grid_nonce_{nonce:03d}", parent_grid_candidate(nonce)
            )
        except (TypeError, ValueError, AirfoilG3ReleaseError):
            continue
        candidate = normalize_candidate(thaw_json(materialized.configuration))
        if any(abs(value) > 0.022 for value in candidate["upper_coefficients"]):
            continue
        if any(abs(value) > 0.022 for value in candidate["lower_coefficients"]):
            continue
        if any(not 0.5 <= value <= 9.5 for value in candidate["alpha_deg"]):
            continue
        if membership.rejects(
            configuration_sha256=materialized.configuration_sha256,
            candidate_sha256_value=materialized.candidate_sha256,
            phenotype_value_sha256=materialized.phenotype_value_sha256,
        ):
            continue
        selection_sha256 = _hash(
            _PARENT_SELECTION_DOMAIN,
            {
                "policy_id": "airfoil_v7_g3_parent_grid",
                "policy_version": 1,
                "task_sha256": TASK_SHA256,
                "nonce": nonce,
                "configuration_sha256": materialized.configuration_sha256,
                "candidate_sha256": materialized.candidate_sha256,
                "phenotype_value_sha256": materialized.phenotype_value_sha256,
                "historical_membership_sha256": membership.membership_sha256,
            },
        )
        eligible.append((selection_sha256, nonce, materialized))
    eligible.sort(key=lambda value: (value[0], value[1]))
    if len(eligible) < 2:
        raise AirfoilG3ReleaseError("fewer than two novel safe parents remain")
    selected = eligible[:2]
    diagnostic = SelectedParent("P_D", selected[0][1], selected[0][0], selected[0][2])
    heldout = SelectedParent("P_H", selected[1][1], selected[1][0], selected[1][2])
    if (
        diagnostic.candidate.configuration_sha256
        == heldout.candidate.configuration_sha256
        or diagnostic.candidate.phenotype_value_sha256
        == heldout.candidate.phenotype_value_sha256
    ):
        raise AirfoilG3ReleaseError("selected parents collide")
    return diagnostic, heldout


def _union_contract(parent: SelectedParent) -> FiniteVariationContract:
    catalog = AirfoilV7UnionVariationCatalog()
    frozen = parent.candidate.configuration
    return FiniteVariationContract(
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.catalog_version,
        catalog_definition_sha256=catalog.definition_sha256,
        parent_configuration=frozen,
        options=catalog.options(frozen),
    )


def build_hypothesis_compilation_request(
    *,
    entry: InsightMemoryEntry,
    parent: SelectedParent,
    contract: FiniteVariationContract,
) -> HypothesisCompilationRequest:
    return HypothesisCompilationRequest(
        reference=entry.reference,
        insight=entry.draft,
        source_evidence_sha256=registered_source_evidence_sha256(entry),
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        source_operator_kinds=entry.applicable_operator_kinds,
        parent_candidate_id=CandidateId(f"candidate_airfoil_g3_{parent.role.lower()}"),
        parent_configuration_sha256=contract.parent_configuration_sha256,
        finite_contract=contract,
        context_projection_sha256=CONTEXT_PROJECTION_SHA256,
        endpoint_definition_sha256=ABSOLUTE_Q_DEFINITION_SHA256,
    )


def _compile_on_both(
    entry: InsightMemoryEntry,
    *,
    compiler: AirfoilV7TrimHypothesisCompiler,
    diagnostic: SelectedParent,
    heldout: SelectedParent,
    diagnostic_contract: FiniteVariationContract,
    heldout_contract: FiniteVariationContract,
) -> tuple[HypothesisCompilationReceipt, HypothesisCompilationReceipt] | None:
    receipts: list[HypothesisCompilationReceipt] = []
    for parent, contract in (
        (diagnostic, diagnostic_contract),
        (heldout, heldout_contract),
    ):
        request = build_hypothesis_compilation_request(
            entry=entry,
            parent=parent,
            contract=contract,
        )
        receipt = compiler.compile(request)
        validate_hypothesis_compiler_identity(compiler, receipt)
        validate_hypothesis_compilation(request, receipt)
        if not receipt.applicable:
            return None
        receipts.append(receipt)
    return receipts[0], receipts[1]


@dataclass(frozen=True, slots=True)
class SelectedCard:
    entry: InsightMemoryEntry
    selection_sha256: str
    diagnostic_receipt: HypothesisCompilationReceipt
    heldout_receipt: HypothesisCompilationReceipt

    def to_record(self) -> dict[str, object]:
        lineage = self.entry.evidence_lineage
        assert lineage is not None
        return {
            "reference": {
                "insight_id": self.entry.reference.insight_id.value,
                "version": self.entry.reference.version,
            },
            "draft_content_sha256": self.entry.draft.content_sha256,
            "registered_source_evidence_sha256": registered_source_evidence_sha256(
                self.entry
            ),
            "source_operator_kinds": list(self.entry.applicable_operator_kinds),
            "lifecycle_state": self.entry.lifecycle_state.value,
            "origin": self.entry.origin.value,
            "lineage_identity_sha256": lineage.identity_sha256,
            "selection_sha256": self.selection_sha256,
            "option_id": self.entry.draft.recommended_option_ids[0],
            "diagnostic_compiler_receipt": {
                **self.diagnostic_receipt.to_record(),
                "receipt_sha256": self.diagnostic_receipt.receipt_sha256,
            },
            "heldout_compiler_receipt": {
                **self.heldout_receipt.to_record(),
                "receipt_sha256": self.heldout_receipt.receipt_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class CardSelectionReceipt:
    """Complete eligible ranking and exact two-card public-hash decision."""

    eligible_ranking: tuple[SelectedCard, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.eligible_ranking) is not tuple or len(self.eligible_ranking) != 4:
            raise ValueError("card selection must rank the exact four-card bank")
        if any(type(value) is not SelectedCard for value in self.eligible_ranking):
            raise TypeError("eligible ranking must contain exact SelectedCard values")
        expected = tuple(
            sorted(
                self.eligible_ranking,
                key=lambda value: (value.selection_sha256, value.entry.reference),
            )
        )
        if self.eligible_ranking != expected:
            raise ValueError("eligible card ranking is not canonical")
        references = tuple(value.entry.reference for value in self.eligible_ranking)
        options = tuple(
            value.entry.draft.recommended_option_ids[0]
            for value in self.eligible_ranking
        )
        if len(set(references)) != 4 or len(set(options)) != 4:
            raise ValueError("eligible card ranking contains duplicate treatments")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_CARD_SELECTION_RECEIPT_DOMAIN, self._identity_record()),
        )

    @property
    def selected_cards(self) -> tuple[SelectedCard, SelectedCard]:
        return self.eligible_ranking[0], self.eligible_ranking[1]

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": "airfoil_v7_g3_public_card_hash",
            "policy_version": 1,
            "eligible_count": 4,
            "selected_count": 2,
            "current_or_heldout_outcome_access": False,
            "scalar_reward_or_rank_input": False,
            "ranking": [
                {
                    "rank": rank,
                    "selected": rank <= 2,
                    **value.to_record(),
                }
                for rank, value in enumerate(self.eligible_ranking, start=1)
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "receipt_sha256": self.receipt_sha256}


def select_cards_with_receipt(
    entries: tuple[InsightMemoryEntry, ...],
    *,
    compiler: AirfoilV7TrimHypothesisCompiler,
    diagnostic: SelectedParent,
    heldout: SelectedParent,
    diagnostic_contract: FiniteVariationContract,
    heldout_contract: FiniteVariationContract,
) -> tuple[tuple[SelectedCard, SelectedCard], CardSelectionReceipt]:
    ranked: list[
        tuple[
            str,
            InsightMemoryEntry,
            tuple[HypothesisCompilationReceipt, HypothesisCompilationReceipt],
        ]
    ] = []
    for entry in entries:
        compiled = _compile_on_both(
            entry,
            compiler=compiler,
            diagnostic=diagnostic,
            heldout=heldout,
            diagnostic_contract=diagnostic_contract,
            heldout_contract=heldout_contract,
        )
        if compiled is None:
            continue
        lineage = entry.evidence_lineage
        if lineage is None:
            raise AirfoilG3ReleaseError("active card lacks evidence lineage")
        selection_sha256 = _hash(
            _CARD_SELECTION_DOMAIN,
            {
                "policy_id": "airfoil_v7_g3_public_card_hash",
                "policy_version": 1,
                "reference": {
                    "insight_id": entry.reference.insight_id.value,
                    "version": entry.reference.version,
                },
                "draft_content_sha256": entry.draft.content_sha256,
                "registered_source_evidence_sha256": registered_source_evidence_sha256(
                    entry
                ),
                "lineage_identity_sha256": lineage.identity_sha256,
                "diagnostic_spec_sha256": compiled[0].spec.spec_sha256,  # type: ignore[union-attr]
                "heldout_spec_sha256": compiled[1].spec.spec_sha256,  # type: ignore[union-attr]
            },
        )
        ranked.append((selection_sha256, entry, compiled))
    ranked.sort(key=lambda value: (value[0], value[1].reference))
    if len(ranked) < 2:
        raise AirfoilG3ReleaseError("fewer than two cards compile on both parents")
    selected = tuple(
        SelectedCard(entry, selection, compiled[0], compiled[1])
        for selection, entry, compiled in ranked
    )
    receipt = CardSelectionReceipt(selected)
    first, second = receipt.selected_cards
    if first.entry.reference == second.entry.reference or (
        first.entry.draft.recommended_option_ids
        == second.entry.draft.recommended_option_ids
    ):
        raise AirfoilG3ReleaseError("selected cards do not define distinct treatments")
    return receipt.selected_cards, receipt


def select_cards(
    entries: tuple[InsightMemoryEntry, ...],
    *,
    compiler: AirfoilV7TrimHypothesisCompiler,
    diagnostic: SelectedParent,
    heldout: SelectedParent,
    diagnostic_contract: FiniteVariationContract,
    heldout_contract: FiniteVariationContract,
) -> tuple[SelectedCard, SelectedCard]:
    selected, _ = select_cards_with_receipt(
        entries,
        compiler=compiler,
        diagnostic=diagnostic,
        heldout=heldout,
        diagnostic_contract=diagnostic_contract,
        heldout_contract=heldout_contract,
    )
    return selected


def _is_historically_novel(
    configuration: object,
    membership: HistoricalMembership,
) -> bool:
    materialized = CandidateMaterialization.from_configuration("novelty_probe", configuration)
    return not membership.rejects(
        configuration_sha256=materialized.configuration_sha256,
        candidate_sha256_value=materialized.candidate_sha256,
        phenotype_value_sha256=materialized.phenotype_value_sha256,
    )


def select_sham_option(
    *,
    selected_cards: tuple[SelectedCard, SelectedCard],
    diagnostic_contract: FiniteVariationContract,
    heldout_contract: FiniteVariationContract,
    membership: HistoricalMembership,
) -> tuple[str, str]:
    excluded = {
        value.entry.draft.recommended_option_ids[0] for value in selected_cards
    }
    ranked: list[tuple[str, str]] = []
    for option in heldout_contract.options:
        if option.family != "trim_only" or option.option_id in excluded:
            continue
        diagnostic_option = diagnostic_contract.resolve(option.option_id)
        if diagnostic_option.family != "trim_only":
            continue
        if not _is_historically_novel(option.child_configuration, membership):
            continue
        if not _is_historically_novel(
            diagnostic_option.child_configuration, membership
        ):
            continue
        selection_sha256 = _hash(
            _SHAM_SELECTION_DOMAIN,
            {
                "policy_id": "airfoil_v7_g3_neutral_trim_sham",
                "policy_version": 1,
                "option_id": option.option_id,
                "diagnostic_option_identity_sha256": diagnostic_option.identity_sha256,
                "heldout_option_identity_sha256": option.identity_sha256,
                "excluded_active_option_ids": sorted(excluded),
            },
        )
        ranked.append((selection_sha256, option.option_id))
    ranked.sort()
    if not ranked:
        raise AirfoilG3ReleaseError("no novel distinct trim sham remains")
    return ranked[0][1], ranked[0][0]


def build_sham_entry(option_id: str) -> InsightMemoryEntry:
    draft = InsightDraft(
        claim=(
            "This evidence-free control assigns one exact pointwise trim pattern "
            "without making a directional performance claim."
        ),
        trigger=(
            f"Administer exact finite trim action {option_id} on the current parent."
        ),
        mechanism=(
            "No aerodynamic benefit or harm mechanism is asserted; this card is a "
            "schema- and action-cardinality-matched neutral administration control."
        ),
        affected_paths=TRIM_PATHS,
        evidence_summary="No empirical evidence, outcome, rank, or source contrast is attached.",
        confidence=0.0,
        evidence_contrast_ids=(),
        effect_predictions=tuple(
            MetricEffectPrediction(
                metric_id=metric_id,
                direction=MetricEffectDirection.UNKNOWN,
            )
            for metric_id in REQUIRED_METRIC_IDS
        ),
        recommended_option_families=("trim_only",),
        recommended_option_ids=(option_id,),
        action_template=(
            f"Select the current-parent finite option {option_id}; do not alter shape."
        ),
        falsification_condition=(
            "The structural sham is falsified if the exact assigned trim does not "
            "change all three declared angle paths while preserving both shape arrays."
        ),
    )
    entry = InsightMemoryEntry(
        reference=InsightRef(InsightId("insight_airfoil_g3_sham_000001"), 1),
        draft=draft,
        initial_score=0.0,
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        lifecycle_state=InsightLifecycleState.QUARANTINED,
        origin=InsightOrigin.MANUAL,
        evidence_lineage=None,
        relations=(),
    )
    entry.__post_init__()
    return entry


def build_strict_sham_requirement(
    entry: InsightMemoryEntry,
    contract: FiniteVariationContract,
) -> InsightTreatmentRequirement:
    if entry.origin is not InsightOrigin.MANUAL or entry.evidence_lineage is not None:
        raise AirfoilG3ReleaseError("sham entry is not evidence-free and manual")
    option = contract.resolve(entry.draft.recommended_option_ids[0])
    evidence = TreatmentInsightEvidence(
        reference=entry.reference,
        insight_content_sha256=entry.draft.content_sha256,
        applicable_operator_kinds=entry.applicable_operator_kinds,
        affected_paths=TRIM_PATHS,
        recommended_option_families=("trim_only",),
        recommended_option_ids=(option.option_id,),
    )
    return InsightTreatmentRequirement(
        insight_bindings=(evidence.binding(),),
        finite_contract_sha256=contract.identity_sha256,
        allowed_actions=(
            TreatmentActionBinding(option.option_id, option.identity_sha256),
        ),
        claim_mode=TreatmentClaimMode.EXACT_REQUIRED,
        assignment_role=TreatmentAssignmentRole.SHAM_CONTROL,
        require_option_family_match=True,
        require_changed_path_overlap=True,
    )


def select_shape_mate(
    *,
    heldout_contract: FiniteVariationContract,
    membership: HistoricalMembership,
) -> tuple[str, str]:
    ranked: list[tuple[str, str]] = []
    for option in heldout_contract.options:
        if option.family != "shape_only":
            continue
        try:
            validate_no_cfd_geometry(option.child_configuration)
        except (TypeError, ValueError, AirfoilG3ReleaseError):
            continue
        if not _is_historically_novel(option.child_configuration, membership):
            continue
        selection_sha256 = _hash(
            _MATE_SELECTION_DOMAIN,
            {
                "policy_id": "airfoil_v7_g3_orthogonal_shape_mate",
                "policy_version": 1,
                "option_id": option.option_id,
                "option_identity_sha256": option.identity_sha256,
                "parent_configuration_sha256": (
                    heldout_contract.parent_configuration_sha256
                ),
                "treatment_paths": list(TRIM_PATHS),
                "shape_paths_are_disjoint": True,
            },
        )
        ranked.append((selection_sha256, option.option_id))
    ranked.sort()
    if not ranked:
        raise AirfoilG3ReleaseError("no safe novel orthogonal shape mate remains")
    return ranked[0][1], ranked[0][0]


@dataclass(frozen=True, slots=True)
class ProspectiveFreshnessProof:
    physical_candidates: tuple[CandidateMaterialization, ...]
    occurrence_schedule: tuple[tuple[str, str, str], ...]
    recombinations: tuple[DisjointPatchMaterialization, ...]

    def __post_init__(self) -> None:
        if len(self.physical_candidates) != MAX_UNIQUE_EVALUATIONS:
            raise ValueError("prospective proof must contain exactly 11 physical candidates")
        if len(self.occurrence_schedule) != LOGICAL_CANDIDATE_OCCURRENCES:
            raise ValueError("prospective proof must contain exactly 12 occurrences")
        for materialization in self.recombinations:
            materialization.revalidate()
        for identity in (
            tuple(value.configuration_sha256 for value in self.physical_candidates),
            tuple(value.candidate_sha256 for value in self.physical_candidates),
            tuple(value.phenotype_value_sha256 for value in self.physical_candidates),
        ):
            if len(set(identity)) != MAX_UNIQUE_EVALUATIONS:
                raise ValueError("prospective physical candidates collide")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "expected_unique_physical_candidates": MAX_UNIQUE_EVALUATIONS,
            "expected_logical_occurrences": LOGICAL_CANDIDATE_OCCURRENCES,
            "physical_candidates": [
                value.to_record() for value in self.physical_candidates
            ],
            "occurrence_schedule": [
                {"stage": stage, "label": label, "cache_disposition": disposition}
                for stage, label, disposition in self.occurrence_schedule
            ],
            "recombination_receipts": [
                {
                    "receipt_sha256": value.receipt_sha256,
                    "union_patch_sha256": value.union_patch.patch_hash,
                    "left_patch_sha256": value.classification.left_patch_hash,
                    "right_patch_sha256": value.classification.right_patch_hash,
                    "preservation_replay_verified": True,
                }
                for value in self.recombinations
            ],
        }


def _freshness_proof(
    *,
    diagnostic: SelectedParent,
    heldout: SelectedParent,
    selected_cards: tuple[SelectedCard, SelectedCard],
    sham_option_id: str,
    mate_option_id: str,
    diagnostic_contract: FiniteVariationContract,
    heldout_contract: FiniteVariationContract,
    membership: HistoricalMembership,
) -> ProspectiveFreshnessProof:
    h1_id = selected_cards[0].entry.draft.recommended_option_ids[0]
    h2_id = selected_cards[1].entry.draft.recommended_option_ids[0]
    g1_h1 = diagnostic_contract.resolve(h1_id)
    g1_h2 = diagnostic_contract.resolve(h2_id)
    g2_h1 = heldout_contract.resolve(h1_id)
    g2_h2 = heldout_contract.resolve(h2_id)
    g2_n = heldout_contract.resolve(sham_option_id)
    mate = heldout_contract.resolve(mate_option_id)
    if any(value.family != "trim_only" for value in (g1_h1, g1_h2, g2_h1, g2_h2, g2_n)):
        raise AirfoilG3ReleaseError("one treatment materialization is not trim-only")
    if mate.family != "shape_only":
        raise AirfoilG3ReleaseError("orthogonal mate is not shape-only")

    recombiner = DisjointPatchRecombiner()
    ancestor_id = CandidateId("candidate_airfoil_g3_ph")
    mate_id = CandidateId("candidate_airfoil_g3_e")
    recombinations: list[DisjointPatchMaterialization] = []
    for label, branch, branch_id in (
        ("h1_union_e", g2_h1, CandidateId("candidate_airfoil_g3_h1_ph")),
        ("h2_union_e", g2_h2, CandidateId("candidate_airfoil_g3_h2_ph")),
        ("n_union_e", g2_n, CandidateId("candidate_airfoil_g3_n_ph")),
    ):
        recombinations.append(
            recombiner.materialize(
                ancestor=heldout.candidate.configuration,
                ancestor_candidate_id=ancestor_id,
                left=branch.child_configuration,
                left_candidate_id=branch_id,
                right=mate.child_configuration,
                right_candidate_id=mate_id,
                target_candidate_id=CandidateId(f"candidate_airfoil_g3_{label}"),
            )
        )

    physical = (
        diagnostic.candidate,
        heldout.candidate,
        CandidateMaterialization.from_configuration("G1.H1@P_D", g1_h1.child_configuration),
        CandidateMaterialization.from_configuration("G1.H2@P_D", g1_h2.child_configuration),
        CandidateMaterialization.from_configuration("G2.H1@P_H", g2_h1.child_configuration),
        CandidateMaterialization.from_configuration("G2.H2@P_H", g2_h2.child_configuration),
        CandidateMaterialization.from_configuration("G2.N@P_H", g2_n.child_configuration),
        CandidateMaterialization.from_configuration("G2.E@P_H", mate.child_configuration),
        CandidateMaterialization.from_configuration("G3.H1+E", recombinations[0].configuration),
        CandidateMaterialization.from_configuration("G3.H2+E", recombinations[1].configuration),
        CandidateMaterialization.from_configuration("G3.N+E", recombinations[2].configuration),
    )
    for value in physical:
        if membership.rejects(
            configuration_sha256=value.configuration_sha256,
            candidate_sha256_value=value.candidate_sha256,
            phenotype_value_sha256=value.phenotype_value_sha256,
        ):
            raise AirfoilG3ReleaseError(
                f"prospective candidate {value.label} is not historically fresh"
            )
    schedule = (
        ("G0", "P_D", "MISS"),
        ("G0", "P_H", "MISS"),
        ("G1", "H1@P_D", "MISS"),
        ("G1", "H2@P_D", "MISS"),
        ("G2", "H1@P_H", "MISS"),
        ("G2", "H2@P_H", "MISS"),
        ("G2", "N@P_H", "MISS"),
        ("G2", "E@P_H", "MISS"),
        ("G3", "P_H_REPRODUCTION", "HIT"),
        ("G3", "H1+E", "MISS"),
        ("G3", "H2+E", "MISS"),
        ("G3", "N+E", "MISS"),
    )
    return ProspectiveFreshnessProof(physical, schedule, tuple(recombinations))


def _preparation_source_code_bindings() -> tuple[SourceFileBinding, ...]:
    relative_paths = (
        "agent_evolve/examples/benchmarks/engibench_airfoil/problem_def.py",
        "agent_evolve/examples/benchmarks/engibench_airfoil/v7_contract.py",
        "agent_evolve/examples/benchmarks/engibench_airfoil/v7_g3_release.py",
        "agent_evolve/examples/benchmarks/engibench_airfoil/v7_variation_catalog.py",
        "agent_evolve/src/agent_evolve/ports/executable_hypothesis.py",
        "agent_evolve/src/agent_evolve/policies/memory/treatment_compliance.py",
        "agent_evolve/src/agent_evolve/policies/variation/disjoint_recombination.py",
    )
    values = []
    for relative in relative_paths:
        path = (WORKSPACE_ROOT / relative).resolve(strict=True)
        values.append(
            SourceFileBinding(relative, path.stat().st_size, _sha256_file(path))
        )
    return tuple(values)


@dataclass(frozen=True, slots=True)
class AirfoilG3ReleasePreparation:
    membership: HistoricalMembership
    membership_file_sha256: str
    card_bank: AuthenticatedTrimCardBank
    card_bank_file_sha256: str
    diagnostic_parent: SelectedParent
    heldout_parent: SelectedParent
    diagnostic_contract: FiniteVariationContract
    heldout_contract: FiniteVariationContract
    authenticated_trim_entries: tuple[InsightMemoryEntry, ...]
    selected_cards: tuple[SelectedCard, SelectedCard]
    card_selection_receipt: CardSelectionReceipt
    sham_entry: InsightMemoryEntry
    sham_selection_sha256: str
    sham_requirement: InsightTreatmentRequirement
    mate_option_id: str
    mate_selection_sha256: str
    freshness: ProspectiveFreshnessProof
    release_sha256: str = field(init=False)

    def _identity_record(self) -> dict[str, object]:
        source_entries = []
        for entry in self.authenticated_trim_entries:
            lineage = entry.evidence_lineage
            assert lineage is not None
            source_entries.append(
                {
                    "reference": {
                        "insight_id": entry.reference.insight_id.value,
                        "version": entry.reference.version,
                    },
                    "draft_content_sha256": entry.draft.content_sha256,
                    "legacy_readiness_content_sha256": (
                        EXPECTED_LEGACY_READINESS_CONTENT_BY_DRAFT[
                            entry.draft.content_sha256
                        ]
                    ),
                    "registered_source_evidence_sha256": (
                        registered_source_evidence_sha256(entry)
                    ),
                    "source_operator_kinds": list(entry.applicable_operator_kinds),
                    "lifecycle_state": entry.lifecycle_state.value,
                    "origin": entry.origin.value,
                    "lineage": lineage.to_record(),
                }
            )
        source_code = _preparation_source_code_bindings()
        return {
            "schema_version": 1,
            "kind": "airfoil_v7_g3_provider_free_release_preparation",
            "claim_boundary": {
                "provider_called": False,
                "credentials_read": False,
                "physical_evaluator_called": False,
                "scientific_result_eligible": False,
                "launch_authorized": False,
                "meaning": "release preparation only; no efficacy or wall-clock result",
            },
            "historical_membership": {
                "file_sha256": self.membership_file_sha256,
                "code_pinned_expected_file_sha256": (
                    EXPECTED_HISTORICAL_MEMBERSHIP_FILE_SHA256
                ),
                "membership_sha256": self.membership.membership_sha256,
                "code_pinned_expected_membership_sha256": (
                    EXPECTED_HISTORICAL_MEMBERSHIP_SHA256
                ),
                "source_manifest_sha256": self.membership.source_manifest_sha256,
                "source_file_count": len(self.membership.source_files),
                "configuration_hash_count": len(
                    self.membership.configuration_sha256s
                ),
                "candidate_hash_count": len(self.membership.candidate_sha256s),
                "phenotype_hash_count": len(
                    self.membership.phenotype_value_sha256s
                ),
                "parent_and_freshness_selectors_opened_source_paths": False,
                "general_outcome_logs_opened_by_preparation": False,
                "membership_only": True,
                "frozen_before_g3_live_launch": True,
            },
            "historical_card_source": {
                "sealed_card_bank_file_sha256": self.card_bank_file_sha256,
                "code_pinned_expected_card_bank_file_sha256": (
                    EXPECTED_CARD_BANK_FILE_SHA256
                ),
                "sealed_card_bank_sha256": self.card_bank.card_bank_sha256,
                "code_pinned_expected_card_bank_sha256": (
                    EXPECTED_CARD_BANK_SHA256
                ),
                "run_path": _workspace_relative(HISTORICAL_RUN),
                "status": EXPECTED_HISTORICAL_STATUS,
                "reflection_result_sha256": EXPECTED_REFLECTION_RESULT_SHA256,
                "provider_free_readiness_sha256": (
                    EXPECTED_PROVIDER_FREE_READINESS_SHA256
                ),
                "finalization_sha256": EXPECTED_FINALIZATION_SHA256,
                "recursive_content_sha256": EXPECTED_RECURSIVE_CONTENT_SHA256,
                "accepted_reflection_is_not_run_success": True,
                "preparation_opened_source_run": False,
                "offline_card_authority_opened_allowlisted_sources": True,
                "card_source_outcomes_exposed_to_parent_selection": False,
                "frozen_historical_memory_is_explicit_development_input": True,
                "zero_current_run_or_heldout_outcome_access": True,
                "frozen_before_g3_live_launch": True,
                "entries": source_entries,
            },
            "absolute_endpoint": {
                **_ENDPOINT_DEFINITION,
                "definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
            },
            "memory_context": {
                **_CONTEXT_DEFINITION,
                "rich_definition_sha256": CONTEXT_DEFINITION_SHA256,
                "runtime_context_stratum_sha256": CONTEXT_PROJECTION_SHA256,
                "projection_sha256": CONTEXT_PROJECTION_SHA256,
            },
            "selection_policies": {
                "parent": {
                    "domain_ascii": _PARENT_SELECTION_DOMAIN.decode("ascii"),
                    "grid_nonces": [PARENT_GRID_NONCES[0], PARENT_GRID_NONCES[-1]],
                    "near_neutral_coefficient_magnitude": 0.0015,
                    "near_neutral_alpha_center": 2.5,
                    "near_neutral_alpha_offset": 0.25,
                    "membership_and_geometry_only": True,
                },
                "cards": {
                    "domain_ascii": _CARD_SELECTION_DOMAIN.decode("ascii"),
                    "eligible_if_compiles_on_both_parents": True,
                    "selected_count": 2,
                    "selection_inputs_include_frozen_evidence_identity_hashes": True,
                    "scalar_reward_or_rank_input": False,
                    "current_run_or_heldout_outcome_access": False,
                },
                "sham": {
                    "domain_ascii": _SHAM_SELECTION_DOMAIN.decode("ascii"),
                    "evidence_free": True,
                    "singleton_trim_support": True,
                },
                "mate": {
                    "domain_ascii": _MATE_SELECTION_DOMAIN.decode("ascii"),
                    "engine_authored": True,
                    "shape_only": True,
                    "disjoint_from_all_trim_paths": True,
                },
            },
            "parents": [
                self.diagnostic_parent.to_record(),
                self.heldout_parent.to_record(),
            ],
            "finite_contracts": {
                "P_D": {
                    "identity_sha256": self.diagnostic_contract.identity_sha256,
                    "option_count": len(self.diagnostic_contract.options),
                },
                "P_H": {
                    "identity_sha256": self.heldout_contract.identity_sha256,
                    "option_count": len(self.heldout_contract.options),
                },
            },
            "hypothesis_compiler": {
                **_COMPILER_DEFINITION,
                "definition_sha256": COMPILER_DEFINITION_SHA256,
            },
            "selected_cards": [value.to_record() for value in self.selected_cards],
            "card_selection_receipt": self.card_selection_receipt.to_record(),
            "sham": {
                "reference": {
                    "insight_id": self.sham_entry.reference.insight_id.value,
                    "version": self.sham_entry.reference.version,
                },
                "draft_content_sha256": self.sham_entry.draft.content_sha256,
                "registered_source_evidence_sha256": (
                    registered_source_evidence_sha256(self.sham_entry)
                ),
                "selection_sha256": self.sham_selection_sha256,
                "option_id": self.sham_entry.draft.recommended_option_ids[0],
                "origin": self.sham_entry.origin.value,
                "lifecycle_state": self.sham_entry.lifecycle_state.value,
                "evidence_lineage": None,
                "effect_directions": [
                    value.direction.value
                    for value in self.sham_entry.draft.effect_predictions
                ],
                "requirement": {
                    **self.sham_requirement.to_record(),
                    "requirement_sha256": self.sham_requirement.requirement_sha256,
                },
                "causal_credit_eligible": False,
            },
            "orthogonal_mate": {
                "option_id": self.mate_option_id,
                "selection_sha256": self.mate_selection_sha256,
                "family": "shape_only",
                "model_authored": False,
            },
            "prospective_freshness": self.freshness.to_record(),
            "exact_budget": {
                "max_logical_llm_calls": MAX_LOGICAL_LLM_CALLS,
                "proposal_calls": 5,
                "post_g3_curation_calls": 1,
                "max_unique_evaluations": MAX_UNIQUE_EVALUATIONS,
                "logical_candidate_occurrences": LOGICAL_CANDIDATE_OCCURRENCES,
                "g1_proposal_concurrency": 2,
                "g2_proposal_concurrency": 3,
                "evaluator_concurrency": 1,
                "max_physical_attempts_per_logical_call": 2,
                "schema_repair_or_logical_rerun": False,
                "max_output_tokens_per_call": 384000,
                "planner_no_yield_reward": -2.0,
                "planner_no_yield_equals_absolute_q_failure": True,
            },
            "live_only_terminal_requirements": {
                "complete_launch_manifest": {
                    "required": True,
                    "status": "unobserved_until_live_run",
                    "must_bind_transitive_runtime": [
                        "generic_g3_causal_screen",
                        "budgeted_optimizer_and_agentic_engine",
                        "executable_hypothesis_and_treatment_lifecycle",
                        "evaluation_cache",
                        "provider_queue_and_pydantic_generator",
                        "airfoil_live_evaluator_and_adapter",
                        "boundary_runner_and_result_analyzer",
                    ],
                    "must_bind_environment": [
                        "dependency_lock",
                        "python_runtime",
                        "operating_system",
                        "solver_or_container_identity",
                        "git_tree_and_dirty_patch_or_equivalent_file_manifest",
                    ],
                },
                "pre_run_cache": {
                    "cached_entries": 0,
                    "in_flight": 0,
                    "hits": 0,
                    "misses": 0,
                    "coalesced": 0,
                    "evictions": 0,
                },
                "cache_events": {
                    "distinct_misses": MAX_UNIQUE_EVALUATIONS,
                    "hits": 1,
                    "only_hit": "P_H_REPRODUCTION",
                    "coalesced": 0,
                    "evictions": 0,
                },
                "post_run_cache": {
                    "cached_entries": MAX_UNIQUE_EVALUATIONS,
                    "in_flight": 0,
                    "hits": 1,
                    "misses": MAX_UNIQUE_EVALUATIONS,
                    "coalesced": 0,
                    "evictions": 0,
                },
                "raw_receipts": EXPECTED_RAW_RECEIPTS,
                "solver_point_calls_per_receipt": 3,
                "total_solver_point_calls": EXPECTED_SOLVER_POINT_CALLS,
                "single_fresh_evaluator_run_id": True,
                "miss_to_raw_receipt_bijection_required": True,
                "reproduction_creates_raw_receipt": False,
                "status": "unobserved_until_live_run",
            },
            "preparation_source_code": {
                "scope": (
                    "provider-free release construction, Airfoil compiler, "
                    "freshness proof, and directly imported generic contracts only"
                ),
                "complete_live_launch_manifest": False,
                "files": [value.to_record() for value in source_code],
                "manifest_sha256": _hash(
                    _SOURCE_MANIFEST_DOMAIN,
                    [value.to_record() for value in source_code],
                ),
            },
        }

    def __post_init__(self) -> None:
        self.membership.__post_init__()
        if not _is_sha256(self.membership_file_sha256):
            raise ValueError("membership file hash is malformed")
        self.card_bank.__post_init__()
        if not _is_sha256(self.card_bank_file_sha256):
            raise ValueError("card-bank file hash is malformed")
        if self.authenticated_trim_entries != self.card_bank.entries:
            raise ValueError("release entries differ from the sealed card bank")
        if len(self.authenticated_trim_entries) != 4:
            raise ValueError("release preparation requires four source trim cards")
        if len(self.selected_cards) != 2:
            raise ValueError("release preparation requires two selected cards")
        self.card_selection_receipt.__post_init__()
        if self.selected_cards != self.card_selection_receipt.selected_cards:
            raise ValueError("selected cards differ from the complete ranking receipt")
        self.sham_requirement.__post_init__()
        self.freshness.__post_init__()
        object.__setattr__(
            self,
            "release_sha256",
            _hash(_RELEASE_DOMAIN, self._identity_record()),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "release_sha256": self.release_sha256}


def prepare_release(
    membership_path: Path = DEFAULT_DENYLIST_PATH,
    card_bank_path: Path = DEFAULT_CARD_BANK_PATH,
) -> AirfoilG3ReleasePreparation:
    """Prepare the release using only two sealed offline authority files."""

    resolved_membership = membership_path.expanduser().resolve(strict=True)
    membership_file_sha256 = _sha256_file(resolved_membership)
    if membership_file_sha256 != EXPECTED_HISTORICAL_MEMBERSHIP_FILE_SHA256:
        raise AirfoilG3ReleaseError(
            "historical membership file differs from its preregistered digest"
        )
    membership = load_historical_denylist(resolved_membership)
    if (
        membership.membership_sha256 != EXPECTED_HISTORICAL_MEMBERSHIP_SHA256
        or membership.source_manifest_sha256
        != EXPECTED_HISTORICAL_SOURCE_MANIFEST_SHA256
    ):
        raise AirfoilG3ReleaseError(
            "historical membership authority differs from its preregistered identity"
        )
    resolved_card_bank = card_bank_path.expanduser().resolve(strict=True)
    card_bank_file_sha256 = _sha256_file(resolved_card_bank)
    if card_bank_file_sha256 != EXPECTED_CARD_BANK_FILE_SHA256:
        raise AirfoilG3ReleaseError(
            "card-bank file differs from its preregistered digest"
        )
    card_bank = load_authenticated_trim_card_bank(resolved_card_bank)
    if card_bank.card_bank_sha256 != EXPECTED_CARD_BANK_SHA256:
        raise AirfoilG3ReleaseError(
            "card-bank authority differs from its preregistered identity"
        )
    diagnostic, heldout = select_parents(membership)
    diagnostic_contract = _union_contract(diagnostic)
    heldout_contract = _union_contract(heldout)
    if len(diagnostic_contract.options) != 80 or len(heldout_contract.options) != 80:
        raise AirfoilG3ReleaseError("Airfoil union catalog must contain exactly 80 options")
    entries = card_bank.entries
    compiler = AirfoilV7TrimHypothesisCompiler()
    cards, card_selection_receipt = select_cards_with_receipt(
        entries,
        compiler=compiler,
        diagnostic=diagnostic,
        heldout=heldout,
        diagnostic_contract=diagnostic_contract,
        heldout_contract=heldout_contract,
    )
    sham_option_id, sham_selection_sha256 = select_sham_option(
        selected_cards=cards,
        diagnostic_contract=diagnostic_contract,
        heldout_contract=heldout_contract,
        membership=membership,
    )
    sham_entry = build_sham_entry(sham_option_id)
    sham_requirement = build_strict_sham_requirement(sham_entry, heldout_contract)
    mate_option_id, mate_selection_sha256 = select_shape_mate(
        heldout_contract=heldout_contract,
        membership=membership,
    )
    freshness = _freshness_proof(
        diagnostic=diagnostic,
        heldout=heldout,
        selected_cards=cards,
        sham_option_id=sham_option_id,
        mate_option_id=mate_option_id,
        diagnostic_contract=diagnostic_contract,
        heldout_contract=heldout_contract,
        membership=membership,
    )
    return AirfoilG3ReleasePreparation(
        membership=membership,
        membership_file_sha256=membership_file_sha256,
        card_bank=card_bank,
        card_bank_file_sha256=card_bank_file_sha256,
        diagnostic_parent=diagnostic,
        heldout_parent=heldout,
        diagnostic_contract=diagnostic_contract,
        heldout_contract=heldout_contract,
        authenticated_trim_entries=entries,
        selected_cards=cards,
        card_selection_receipt=card_selection_receipt,
        sham_entry=sham_entry,
        sham_selection_sha256=sham_selection_sha256,
        sham_requirement=sham_requirement,
        mate_option_id=mate_option_id,
        mate_selection_sha256=mate_selection_sha256,
        freshness=freshness,
    )


def write_release_preparation(
    preparation: AirfoilG3ReleasePreparation,
    path: Path = DEFAULT_RELEASE_PATH,
) -> str:
    preparation.__post_init__()
    _write_json_atomic(path, preparation.to_record())
    return _sha256_file(path)


def _diagnostic_permutation_public_record(
    *,
    release_sha256: str,
    card_identities: tuple[tuple[str, int, str, str], ...],
) -> dict[str, object]:
    return {
        "definition_sha256": DIAGNOSTIC_PERMUTATION_DEFINITION_SHA256,
        "release_sha256": release_sha256,
        "cards": [
            {
                "insight_id": insight_id,
                "version": version,
                "draft_content_sha256": draft_sha256,
                "registered_source_evidence_sha256": source_sha256,
            }
            for insight_id, version, draft_sha256, source_sha256 in card_identities
        ],
    }


def freeze_diagnostic_permutation(
    preparation: AirfoilG3ReleasePreparation,
) -> tuple[
    FrozenDiagnosticPermutation,
    str,
    tuple[tuple[str, int, str, str], ...],
]:
    """Realize the public two-slot rank before any current-run outcome exists."""

    preparation.__post_init__()
    entries = tuple(
        sorted(
            (value.entry for value in preparation.selected_cards),
            key=lambda value: value.reference,
        )
    )
    card_identities = tuple(
        (
            entry.reference.insight_id.value,
            entry.reference.version,
            entry.draft.content_sha256,
            registered_source_evidence_sha256(entry),
        )
        for entry in entries
    )
    selection_sha256 = _hash(
        _DIAGNOSTIC_PERMUTATION_SELECTION_DOMAIN,
        _diagnostic_permutation_public_record(
            release_sha256=preparation.release_sha256,
            card_identities=card_identities,
        ),
    )
    permutation = FrozenDiagnosticPermutation(
        active_references=(entries[0].reference, entries[1].reference),
        permutation_rank=int(selection_sha256, 16) % 2,
        randomization_policy_id=str(
            _DIAGNOSTIC_PERMUTATION_DEFINITION["policy_id"]
        ),
        randomization_policy_version=int(
            _DIAGNOSTIC_PERMUTATION_DEFINITION["policy_version"]
        ),
        randomization_definition_sha256=(
            DIAGNOSTIC_PERMUTATION_DEFINITION_SHA256
        ),
    )
    return permutation, selection_sha256, card_identities


@dataclass(frozen=True, slots=True)
class AirfoilG3PrelaunchFreezeReceipt:
    """Chronology root created after prep and before any G3 live outcome."""

    frozen_at_utc: str
    membership_path: str
    membership_file_sha256: str
    membership_sha256: str
    membership_source_manifest_sha256: str
    card_bank_path: str
    card_bank_file_sha256: str
    card_bank_sha256: str
    release_path: str
    release_file_sha256: str
    release_sha256: str
    diagnostic_permutation: FrozenDiagnosticPermutation
    diagnostic_permutation_selection_sha256: str
    diagnostic_permutation_card_identities: tuple[
        tuple[str, int, str, str], ...
    ]
    preparation_module: SourceFileBinding
    preparation_source_manifest_sha256: str
    freeze_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _UTC_SECONDS.fullmatch(self.frozen_at_utc) is None:
            raise ValueError("freeze time must be UTC RFC3339 with whole seconds")
        try:
            datetime.strptime(self.frozen_at_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError("freeze time is not a real UTC calendar instant") from exc
        expected_paths = (
            _workspace_relative(DEFAULT_DENYLIST_PATH),
            _workspace_relative(DEFAULT_CARD_BANK_PATH),
            _workspace_relative(DEFAULT_RELEASE_PATH),
        )
        if (
            self.membership_path,
            self.card_bank_path,
            self.release_path,
        ) != expected_paths:
            raise ValueError("freeze receipt must bind the canonical release paths")
        expected_hashes = (
            EXPECTED_HISTORICAL_MEMBERSHIP_FILE_SHA256,
            EXPECTED_HISTORICAL_MEMBERSHIP_SHA256,
            EXPECTED_HISTORICAL_SOURCE_MANIFEST_SHA256,
            EXPECTED_CARD_BANK_FILE_SHA256,
            EXPECTED_CARD_BANK_SHA256,
        )
        observed_hashes = (
            self.membership_file_sha256,
            self.membership_sha256,
            self.membership_source_manifest_sha256,
            self.card_bank_file_sha256,
            self.card_bank_sha256,
        )
        if observed_hashes != expected_hashes:
            raise ValueError("freeze receipt differs from a preregistered trust root")
        for value in (
            self.release_file_sha256,
            self.release_sha256,
            self.diagnostic_permutation_selection_sha256,
            self.preparation_source_manifest_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("freeze receipt contains a malformed SHA-256")
        self.diagnostic_permutation.__post_init__()
        card_identities = self.diagnostic_permutation_card_identities
        if (
            type(card_identities) is not tuple
            or len(card_identities) != 2
            or card_identities
            != tuple(sorted(set(card_identities), key=lambda value: value[:2]))
        ):
            raise ValueError("diagnostic permutation card identities are not canonical")
        for insight_id, version, draft_sha256, source_sha256 in card_identities:
            InsightRef(InsightId(insight_id), version).__post_init__()
            if not _is_sha256(draft_sha256) or not _is_sha256(source_sha256):
                raise ValueError("diagnostic permutation card hash is malformed")
        references = tuple(
            InsightRef(InsightId(value[0]), value[1]) for value in card_identities
        )
        if self.diagnostic_permutation.active_references != references:
            raise ValueError("diagnostic permutation references differ from its cards")
        expected_selection = _hash(
            _DIAGNOSTIC_PERMUTATION_SELECTION_DOMAIN,
            _diagnostic_permutation_public_record(
                release_sha256=self.release_sha256,
                card_identities=card_identities,
            ),
        )
        if self.diagnostic_permutation_selection_sha256 != expected_selection:
            raise ValueError("diagnostic permutation public selection hash changed")
        if self.diagnostic_permutation.permutation_rank != int(
            expected_selection, 16
        ) % 2:
            raise ValueError("diagnostic permutation rank differs from its public hash")
        self.preparation_module.__post_init__()
        expected_module_path = (
            "agent_evolve/examples/benchmarks/engibench_airfoil/"
            "v7_g3_release.py"
        )
        if self.preparation_module.path != expected_module_path:
            raise ValueError("freeze receipt binds the wrong preparation module")
        object.__setattr__(
            self,
            "freeze_receipt_sha256",
            _hash(_FREEZE_RECEIPT_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "airfoil_v7_g3_prelaunch_chronology_freeze",
            "frozen_at_utc": self.frozen_at_utc,
            "chronology_attestation": {
                "freeze_precedes_g3_live_launch": True,
                "current_run_id_assigned": False,
                "current_run_provider_responses_existing_at_freeze": 0,
                "current_run_physical_receipts_existing_at_freeze": 0,
                "current_run_solver_point_calls_existing_at_freeze": 0,
                "historical_memory_is_explicit_development_input": True,
                "statement": (
                    "At this freeze instant no G3 current-run provider response, "
                    "CFD receipt, or solver point outcome existed."
                ),
            },
            "membership": {
                "path": self.membership_path,
                "file_sha256": self.membership_file_sha256,
                "membership_sha256": self.membership_sha256,
                "source_manifest_sha256": self.membership_source_manifest_sha256,
            },
            "card_bank": {
                "path": self.card_bank_path,
                "file_sha256": self.card_bank_file_sha256,
                "card_bank_sha256": self.card_bank_sha256,
            },
            "release_preparation": {
                "path": self.release_path,
                "file_sha256": self.release_file_sha256,
                "release_sha256": self.release_sha256,
                "preparation_module": self.preparation_module.to_record(),
                "preparation_source_manifest_sha256": (
                    self.preparation_source_manifest_sha256
                ),
            },
            "diagnostic_permutation": {
                "definition": _DIAGNOSTIC_PERMUTATION_DEFINITION,
                "definition_sha256": DIAGNOSTIC_PERMUTATION_DEFINITION_SHA256,
                "public_selection_inputs": _diagnostic_permutation_public_record(
                    release_sha256=self.release_sha256,
                    card_identities=self.diagnostic_permutation_card_identities,
                ),
                "public_selection_sha256": (
                    self.diagnostic_permutation_selection_sha256
                ),
                "frozen_receipt": {
                    **self.diagnostic_permutation.to_record(),
                    "receipt_sha256": self.diagnostic_permutation.receipt_sha256,
                },
                "selected_before_g0_or_g1_outcomes": True,
            },
            "live_launcher_input_contract": {
                "receipt_must_be_verified_before_credentials_or_evaluator_access": True,
                "bound_files_must_match_byte_and_semantic_hashes": True,
                "complete_live_launch_manifest_still_required": True,
                "launcher_must_bind_this_freeze_receipt_sha256": True,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._identity_record(),
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
        }


def create_prelaunch_freeze_receipt(
    preparation: AirfoilG3ReleasePreparation,
    *,
    frozen_at_utc: str,
    membership_path: Path = DEFAULT_DENYLIST_PATH,
    card_bank_path: Path = DEFAULT_CARD_BANK_PATH,
    release_path: Path = DEFAULT_RELEASE_PATH,
) -> AirfoilG3PrelaunchFreezeReceipt:
    """Bind the finalized deterministic prep to one explicit prelaunch instant."""

    preparation.__post_init__()
    resolved_membership = membership_path.expanduser().resolve(strict=True)
    resolved_card_bank = card_bank_path.expanduser().resolve(strict=True)
    resolved_release = release_path.expanduser().resolve(strict=True)
    if _load_json_object(resolved_release) != preparation.to_record():
        raise AirfoilG3ReleaseError(
            "release file differs from the in-memory deterministic preparation"
        )
    module_path = Path(__file__).resolve(strict=True)
    preparation_record = preparation.to_record()
    source_record = preparation_record.get("preparation_source_code")
    if type(source_record) is not dict or not _is_sha256(
        source_record.get("manifest_sha256")
    ):
        raise AirfoilG3ReleaseError("preparation source manifest is missing")
    permutation, permutation_selection, permutation_cards = (
        freeze_diagnostic_permutation(preparation)
    )
    return AirfoilG3PrelaunchFreezeReceipt(
        frozen_at_utc=frozen_at_utc,
        membership_path=_workspace_relative(resolved_membership),
        membership_file_sha256=_sha256_file(resolved_membership),
        membership_sha256=preparation.membership.membership_sha256,
        membership_source_manifest_sha256=(
            preparation.membership.source_manifest_sha256
        ),
        card_bank_path=_workspace_relative(resolved_card_bank),
        card_bank_file_sha256=_sha256_file(resolved_card_bank),
        card_bank_sha256=preparation.card_bank.card_bank_sha256,
        release_path=_workspace_relative(resolved_release),
        release_file_sha256=_sha256_file(resolved_release),
        release_sha256=preparation.release_sha256,
        diagnostic_permutation=permutation,
        diagnostic_permutation_selection_sha256=permutation_selection,
        diagnostic_permutation_card_identities=permutation_cards,
        preparation_module=SourceFileBinding(
            path=_workspace_relative(module_path),
            size_bytes=module_path.stat().st_size,
            sha256=_sha256_file(module_path),
        ),
        preparation_source_manifest_sha256=str(source_record["manifest_sha256"]),
    )


def prelaunch_freeze_receipt_from_record(
    value: Mapping[str, object],
) -> AirfoilG3PrelaunchFreezeReceipt:
    try:
        membership = value["membership"]
        card_bank = value["card_bank"]
        release_record = value["release_preparation"]
        permutation_record = value["diagnostic_permutation"]
        if not all(
            type(item) is dict
            for item in (
                membership,
                card_bank,
                release_record,
                permutation_record,
            )
        ):
            raise TypeError("freeze receipt sections must be objects")
        module_record = release_record["preparation_module"]
        if type(module_record) is not dict:
            raise TypeError("preparation module binding must be an object")
        frozen_permutation = permutation_record["frozen_receipt"]
        public_inputs = permutation_record["public_selection_inputs"]
        if type(frozen_permutation) is not dict or type(public_inputs) is not dict:
            raise TypeError("diagnostic permutation records must be objects")
        raw_references = frozen_permutation["active_references"]
        raw_cards = public_inputs["cards"]
        if type(raw_references) is not list or type(raw_cards) is not list:
            raise TypeError("diagnostic permutation identities must be lists")
        diagnostic_permutation = FrozenDiagnosticPermutation(
            active_references=tuple(
                InsightRef(
                    InsightId(str(item["insight_id"])),
                    int(item["version"]),
                )
                for item in raw_references
                if type(item) is dict
            ),
            permutation_rank=int(frozen_permutation["permutation_rank"]),
            randomization_policy_id=str(
                frozen_permutation["randomization_policy_id"]
            ),
            randomization_policy_version=int(
                frozen_permutation["randomization_policy_version"]
            ),
            randomization_definition_sha256=str(
                frozen_permutation["randomization_definition_sha256"]
            ),
        )
        if len(diagnostic_permutation.active_references) != len(raw_references):
            raise TypeError("one permutation reference is not an object")
        permutation_cards = tuple(
            (
                str(item["insight_id"]),
                int(item["version"]),
                str(item["draft_content_sha256"]),
                str(item["registered_source_evidence_sha256"]),
            )
            for item in raw_cards
            if type(item) is dict
        )
        if len(permutation_cards) != len(raw_cards):
            raise TypeError("one permutation card is not an object")
        receipt = AirfoilG3PrelaunchFreezeReceipt(
            frozen_at_utc=str(value["frozen_at_utc"]),
            membership_path=str(membership["path"]),
            membership_file_sha256=str(membership["file_sha256"]),
            membership_sha256=str(membership["membership_sha256"]),
            membership_source_manifest_sha256=str(
                membership["source_manifest_sha256"]
            ),
            card_bank_path=str(card_bank["path"]),
            card_bank_file_sha256=str(card_bank["file_sha256"]),
            card_bank_sha256=str(card_bank["card_bank_sha256"]),
            release_path=str(release_record["path"]),
            release_file_sha256=str(release_record["file_sha256"]),
            release_sha256=str(release_record["release_sha256"]),
            diagnostic_permutation=diagnostic_permutation,
            diagnostic_permutation_selection_sha256=str(
                permutation_record["public_selection_sha256"]
            ),
            diagnostic_permutation_card_identities=permutation_cards,
            preparation_module=SourceFileBinding(
                path=str(module_record["path"]),
                size_bytes=int(module_record["size_bytes"]),
                sha256=str(module_record["sha256"]),
            ),
            preparation_source_manifest_sha256=str(
                release_record["preparation_source_manifest_sha256"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AirfoilG3ReleaseError("prelaunch freeze receipt is malformed") from exc
    if value.get("freeze_receipt_sha256") != receipt.freeze_receipt_sha256:
        raise AirfoilG3ReleaseError("prelaunch freeze receipt SHA-256 changed")
    if dict(value) != receipt.to_record():
        raise AirfoilG3ReleaseError("prelaunch freeze receipt record changed")
    return receipt


def load_prelaunch_freeze_receipt(
    path: Path = DEFAULT_FREEZE_RECEIPT_PATH,
) -> AirfoilG3PrelaunchFreezeReceipt:
    return prelaunch_freeze_receipt_from_record(
        _load_json_object(path.expanduser().resolve(strict=True))
    )


def write_prelaunch_freeze_receipt(
    receipt: AirfoilG3PrelaunchFreezeReceipt,
    path: Path = DEFAULT_FREEZE_RECEIPT_PATH,
) -> str:
    receipt.__post_init__()
    _write_json_exclusive(path, receipt.to_record())
    return _sha256_file(path)


__all__ = [
    "ABSOLUTE_Q_DEFINITION_SHA256",
    "AIRFOIL_G3_ABSOLUTE_REWARD",
    "AIRFOIL_G3_RUNTIME_PHASE",
    "AIRFOIL_G3_RUNTIME_PROBLEM_ID",
    "AirfoilG3ReleaseError",
    "AirfoilG3ReleasePreparation",
    "AirfoilG3PrelaunchFreezeReceipt",
    "AirfoilV7TrimHypothesisCompiler",
    "AuthenticatedTrimCardBank",
    "COMPILER_DEFINITION_SHA256",
    "CardSelectionReceipt",
    "CONTEXT_PROJECTION_SHA256",
    "CONTEXT_DEFINITION_SHA256",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_DENYLIST_PATH",
    "DEFAULT_CARD_BANK_PATH",
    "DEFAULT_HISTORICAL_SOURCE_ROOTS",
    "DEFAULT_RELEASE_PATH",
    "DIAGNOSTIC_PERMUTATION_DEFINITION_SHA256",
    "HistoricalMembership",
    "NoCFDGeometryReceipt",
    "ProspectiveFreshnessProof",
    "absolute_airfoil_q",
    "build_authenticated_trim_card_bank",
    "build_historical_denylist",
    "build_hypothesis_compilation_request",
    "build_sham_entry",
    "build_strict_sham_requirement",
    "create_prelaunch_freeze_receipt",
    "freeze_diagnostic_permutation",
    "historical_membership_from_record",
    "load_historical_denylist",
    "load_authenticated_trim_card_bank",
    "load_prelaunch_freeze_receipt",
    "parent_grid_candidate",
    "prepare_release",
    "prelaunch_freeze_receipt_from_record",
    "reconstruct_authenticated_trim_entries",
    "select_parents",
    "select_cards_with_receipt",
    "validate_no_cfd_geometry",
    "write_historical_denylist",
    "write_authenticated_trim_card_bank",
    "write_prelaunch_freeze_receipt",
    "write_release_preparation",
]
