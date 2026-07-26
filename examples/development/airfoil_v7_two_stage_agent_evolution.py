"""Deterministic preparation for one real two-stage Airfoil-v7 AgentEvolve run.

This module is deliberately benchmark-side.  It adapts the sealed Airfoil-v7
finite contract, optimization semantics, and development oracle to generic
AgentEvolve ports without teaching the application core about aerodynamics.
No function in this module reads credentials, constructs a provider client, or
calls an LLM.

The scientific prompt boundary is intentionally narrow:

* G1 is the frozen outcome-blind, family-stratified sample.
* The sealed oracle emulates the already-paid CFD evaluations in development.
* Reflection sees only the eight requested G1 outcomes, never oracle ranks or
  any unselected outcome.
* Exact source actions live in ``FiniteActionEvidenceBinding`` values.  Card
  prose is scrubbed of exact action/contract identities before source binding.
* M is pristine source memory, P is one coherent deranged donor permutation,
  and N is a genuinely card-free catalog-only forecast request.
* The deterministic allocator receives a benchmark-owned utility binding; the
  generic allocator remains benchmark-neutral.

The returned records are sufficient to inspect and preregister the run before
the queued provider runner is wired in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re

from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
    InsightMemoryBank,
    InsightMemoryEntry,
    ReflectedInsightBatchItem,
    compose_epistemic_prompt_payload,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    bind_portfolio_experimental_view,
    portfolio_card_from_insight_entry,
)
from agent_evolve.application.reflection_workflow import (
    ReflectionPromptShard,
    ReflectionWorkflowRequest,
    ReflectionWorkflowResult,
)
from agent_evolve.application.two_stage_action_evolution import (
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
)
from agent_evolve.application.action_allocation_frame_commit import (
    validate_frame_action_allocation_phase_commit,
)
from agent_evolve.application.paired_allocation_comparison import (
    AllocationComparisonMethodWave,
    validate_paired_allocation_comparison_commitment,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
    bind_finite_action_evidence,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.diagnostic_sampling import (
    DiagnosticActionSample,
    HashStratifiedDiagnosticSampler,
    validate_diagnostic_action_sample,
)
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionForecastAllocationFrameKind,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.paired_allocation_comparison import (
    PairedAllocationComparisonCommitment,
)
from agent_evolve.ports.postcommit_rank_authority import (
    PostcommitRankAuthorization,
    PostcommitRankRequest,
    RankReferenceObservation,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastRequest,
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionForecastBatch,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    ReflectionInsightContract,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioCardSourceRegistry,
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    PortfolioExperimentalViewReceipt,
    derive_portfolio_card_view,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from examples.benchmarks.engibench_airfoil.v7_contract import DELTA_F, DELTA_V
from examples.benchmarks.engibench_airfoil.v7_finite_oracle import (
    OBJECTIVE_NAME,
    ORACLE_FINALIZATION_FRAMING,
    ORACLE_MANIFEST_FRAMING,
    ORACLE_RECORD_FRAMING,
    PARENT_METRICS,
    VIOLATION_NAME,
)
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    materialize_held_out_parent,
)
from examples.benchmarks.engibench_airfoil.v7_launch import DEFAULT_LIVE_LOG_ROOT
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_ACTION_SEMANTICS,
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7UnionVariationCatalog,
)


G1_SAMPLE_SEED = 20_260_715
G1_SAMPLE_DESIGN_KEY = "two_stage_action_evolution_v1"
G1_SAMPLE_SIZE = 8
G2_PORTFOLIO_SIZE = 3
MAX_OUTPUT_TOKENS = 384_000
ALLOCATOR_RISK_AVERSION = 0.5
ALLOCATOR_DIVERSITY_WEIGHT = 0.25
DEFAULT_SEALED_ORACLE_DIR = (
    DEFAULT_LIVE_LOG_ROOT / "finite_oracles" / "ae7_finite_oracle_0715_0143"
)

OBJECTIVE_METRIC_ID = f"objective:{OBJECTIVE_NAME}"
VIOLATION_METRIC_ID = f"violation:{VIOLATION_NAME}"
REQUIRED_METRIC_IDS = tuple(sorted((OBJECTIVE_METRIC_ID, VIOLATION_METRIC_ID)))

PREPARATION_POLICY_ID = "airfoil_v7_two_stage_preparation"
PREPARATION_POLICY_VERSION = 2
_PREPARATION_DEFINITION = (
    b"agent-evolve:airfoil-v7-two-stage-preparation:v2:"
    b"g1-hash-stratified-8;outcome-opaque-seal-verification;"
    b"decode-only-g1-terminals-predecision;engine-issued-empirical-snapshots;"
    b"unverified-model-hypotheses;hash-bound-action-semantics;"
    b"one-source-bound-card-per-action;rank-free-metric-magnitudes;"
    b"strict-batched-reflection;m-pristine;"
    b"p-prompt-evidence-score-donor-derangement-with-source-action-binding;"
    b"n-catalog-only;g1-excluded-from-g2"
)
PREPARATION_DEFINITION_SHA256 = hashlib.sha256(_PREPARATION_DEFINITION).hexdigest()

EXPERIMENTAL_VIEW_POLICY_ID = "airfoil_v7_two_stage_mpn"
EXPERIMENTAL_VIEW_POLICY_VERSION = 2
_EXPERIMENTAL_VIEW_DEFINITION = (
    b"agent-evolve:airfoil-v7-two-stage-mpn:v2:"
    b"m-pristine-source-cards;"
    b"p-binding-sha-sorted-cyclic-prompt-evidence-score-donor-derangement-"
    b"while-retaining-source-finite-action-evidence;"
    b"n-card-free-catalog-only"
)
EXPERIMENTAL_VIEW_DEFINITION_SHA256 = hashlib.sha256(
    _EXPERIMENTAL_VIEW_DEFINITION
).hexdigest()

UTILITY_POLICY_ID = "airfoil_v7_forecast_usefulness_probability"
UTILITY_POLICY_VERSION = 1
_UTILITY_DEFINITION = (
    b"agent-evolve:airfoil-v7-forecast-usefulness-probability:v1:"
    b"for-each-member-quality=clip(-delta_v/0.005,-60,60)"
    b"+0.05*tanh(-delta_f/0.001);"
    b"usefulness=sigmoid(quality);effective-success=probability_valid*usefulness;"
    b"portfolio-utility=1-product(1-effective-success);range=[0,1];"
    b"higher-is-better;quantile-selected-by-generic-allocator"
)
UTILITY_DEFINITION_SHA256 = hashlib.sha256(_UTILITY_DEFINITION).hexdigest()

EVALUATOR_POLICY_ID = "airfoil_v7_sealed_oracle_development"
EVALUATOR_POLICY_VERSION = 1
_EVALUATOR_DEFINITION = (
    b"agent-evolve:airfoil-v7-sealed-oracle-development-evaluator:v1:"
    b"authenticate-recursive-bytes-without-outcome-decode;g1-exact-allowlist;"
    b"postdecision-fsynced-mpn-commitment;one-shot-selected-g2-decode;"
    b"hide-rank-and-unselected-outcomes;zero-provider-calls;zero-new-cfd"
)
EVALUATOR_DEFINITION_SHA256 = hashlib.sha256(_EVALUATOR_DEFINITION).hexdigest()

_CONTRAST_DOMAIN = b"agent-evolve:two-stage-action-observation:v1\x00"
_CARD_EVIDENCE_DOMAIN = b"agent-evolve:airfoil-two-stage-card-evidence:v1\x00"
_EMPIRICAL_FACT_SCHEMA_ID = "airfoil_v7_metric_delta_receipt"
_EMPIRICAL_FACT_SCHEMA_VERSION = 1
_EMPIRICAL_FACT_SCHEMA_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v7-metric-delta-receipt:v1:"
    b"validity;ordered-required-metric-parent-child-and-child-minus-parent;"
    b"authenticated-terminal-evaluation-receipt"
).hexdigest()
_ALLOCATION_COMMITMENT_DOMAIN = (
    b"agent-evolve:airfoil-two-stage-allocation-commitment:v1\x00"
)
_PAIRED_ALLOCATION_COMMITMENT_DOMAIN = (
    b"agent-evolve:airfoil-paired-allocation-commitment:v1\x00"
)
_ACTION_ALLOCATION_REQUEST_DOMAIN = (
    b"agent-evolve:action-allocation-request:v1\x00"
)
_ACTION_PORTFOLIO_DECISION_DOMAIN = (
    b"agent-evolve:action-portfolio-decision:v1\x00"
)
_ELIGIBLE_ACTION_SET_DOMAIN = b"agent-evolve:eligible-action-set:v1\x00"
_IDENTITY_SENTINEL = "[exact-action-identity]"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _frozen_object(value: Mapping[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(dict(value))
    if type(frozen) is not FrozenJsonObject:
        raise AssertionError("expected a frozen JSON object")
    return frozen


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _verify_self_hash(
    record: Mapping[str, object],
    *,
    field: str,
    framing: bytes,
    label: str,
) -> None:
    unsigned = dict(record)
    claimed = unsigned.pop(field, None)
    expected = hashlib.sha256(framing + _canonical_bytes(unsigned)).hexdigest()
    if claimed != expected:
        raise ValueError(f"{label} self-hash failed")


def _recursive_content_binding(
    run_dir: Path,
) -> tuple[dict[str, dict[str, object]], str]:
    """Hash all sealed bytes without decoding any outcome-bearing file."""

    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(ORACLE_FINALIZATION_FRAMING)
    paths = sorted(
        (
            item
            for item in run_dir.rglob("*")
            if item.is_file() and item != run_dir / "finalized.json"
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    )
    for path in paths:
        if path.is_symlink():
            raise ValueError("sealed oracle contains a symbolic-link file")
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        files[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        encoded = relative.encode("utf-8", errors="strict")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return files, aggregate.hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedAirfoilPredecisionOracle:
    """Outcome-opaque seal/contract view available before G2 decisions.

    Recursive verification necessarily reads every byte, but no outcome file is
    JSON-decoded here.  Only ``finalized.json`` and ``oracle_manifest.json`` are
    decoded.  Selected terminal records are decoded later by phase capabilities.
    """

    run_dir: Path
    contract: FiniteVariationContract
    run_id: str
    manifest_sha256: str
    source_sha256: str
    recursive_content_sha256: str
    recursive_file_count: int
    oracle_result_file_sha256: str
    file_bindings: Mapping[str, Mapping[str, object]] = field(
        repr=False,
        compare=False,
    )

    def seal_record(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "recursive_content_sha256": self.recursive_content_sha256,
            "recursive_file_count": self.recursive_file_count,
            "oracle_result_file_sha256": self.oracle_result_file_sha256,
            "finite_contract_identity_sha256": self.contract.identity_sha256,
            "structural_seal_verification_decoded_outcome_file_count": 0,
        }

    def terminal_path(self, option_id: str) -> Path:
        option = self.contract.resolve(option_id)
        ordinal = next(
            index
            for index, candidate in enumerate(self.contract.options, start=1)
            if candidate.option_id == option.option_id
        )
        relative = f"options/{ordinal:03d}-{option.option_id}/terminal.json"
        binding = self.file_bindings.get(relative)
        if type(binding) is not dict:
            raise ValueError("sealed finalization lacks the selected terminal")
        path = self.run_dir / relative
        content = path.read_bytes()
        if (
            binding.get("sha256") != hashlib.sha256(content).hexdigest()
            or binding.get("bytes") != len(content)
        ):
            raise ValueError("selected terminal differs from the recursive seal")
        return path


def verify_airfoil_v7_predecision_oracle(
    run_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> VerifiedAirfoilPredecisionOracle:
    """Authenticate the seal/contract without decoding an outcome-bearing file."""

    resolved = run_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError("sealed oracle path is not a regular directory")
    finalized = _load_object(resolved / "finalized.json", label="finalization")
    _verify_self_hash(
        finalized,
        field="record_sha256",
        framing=ORACLE_RECORD_FRAMING,
        label="finalization",
    )
    files, recursive_sha256 = _recursive_content_binding(resolved)
    if (
        finalized.get("status") != "completed_80_action_oracle"
        or finalized.get("files") != files
        or finalized.get("recursive_file_count") != len(files)
        or finalized.get("recursive_content_sha256") != recursive_sha256
    ):
        raise ValueError("recursive oracle seal changed")
    manifest = _load_object(resolved / "oracle_manifest.json", label="manifest")
    _verify_self_hash(
        manifest,
        field="manifest_sha256",
        framing=ORACLE_MANIFEST_FRAMING,
        label="manifest",
    )
    oracle_record = manifest.get("oracle")
    source_record = manifest.get("source_snapshot")
    if type(oracle_record) is not dict or type(source_record) is not dict:
        raise ValueError("oracle manifest is malformed")
    catalog = oracle_record.get("catalog")
    if type(catalog) is not dict:
        raise ValueError("oracle manifest lacks its catalog")
    parent = materialize_held_out_parent()
    frozen_parent = freeze_json(parent.candidate)
    if type(frozen_parent) is not FrozenJsonObject:
        raise AssertionError("held-out parent must freeze to an object")
    contract = bind_finite_variation_catalog(
        AirfoilV7UnionVariationCatalog(),
        frozen_parent,
    )
    if catalog.get("contract") != contract.evidence_record():
        raise ValueError("live finite contract differs from the sealed manifest")
    evaluation_order = catalog.get("evaluation_order")
    if type(evaluation_order) is not list or len(evaluation_order) != len(
        contract.options
    ):
        raise ValueError("manifest evaluation order is incomplete")
    for ordinal, (option, row) in enumerate(
        zip(contract.options, evaluation_order, strict=True),
        start=1,
    ):
        if type(row) is not dict or any(
            row.get(name) != expected
            for name, expected in {
                "ordinal": ordinal,
                "option_id": option.option_id,
                "family": option.family,
                "option_identity_sha256": option.identity_sha256,
                "typed_child_configuration_sha256": (
                    option.child_configuration_sha256
                ),
            }.items()
        ):
            raise ValueError("manifest evaluation order differs from the contract")
    result_binding = files.get("oracle_result.json")
    if type(result_binding) is not dict:
        raise ValueError("recursive seal lacks the opaque oracle result bytes")
    return VerifiedAirfoilPredecisionOracle(
        run_dir=resolved,
        contract=contract,
        run_id=str(oracle_record.get("run_id")),
        manifest_sha256=str(manifest["manifest_sha256"]),
        source_sha256=str(source_record.get("sha256")),
        recursive_content_sha256=recursive_sha256,
        recursive_file_count=len(files),
        oracle_result_file_sha256=str(result_binding["sha256"]),
        file_bindings=files,
    )


@dataclass(frozen=True, slots=True)
class AirfoilObservedMetric:
    """One requested G1 metric in the published optimization metric space."""

    metric_id: str
    parent_value: float
    child_value: float

    def __post_init__(self) -> None:
        if self.metric_id not in REQUIRED_METRIC_IDS:
            raise ValueError("observed metric is outside Airfoil-v7 semantics")
        for name in ("parent_value", "child_value"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")

    @property
    def delta(self) -> float:
        self.__post_init__()
        return self.child_value - self.parent_value

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "parent_value_hex": self.parent_value.hex(),
            "child_value_hex": self.child_value.hex(),
            "child_minus_parent_delta_hex": self.delta.hex(),
        }

    def prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "parent_value": self.parent_value,
            "child_value": self.child_value,
            "child_minus_parent_delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class AirfoilDevelopmentEvaluation:
    """Rank-free projection of one requested sealed development evaluation."""

    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str
    metrics: tuple[AirfoilObservedMetric, ...]
    terminal_record_sha256: str
    raw_receipt_sha256: str
    active_wall_seconds: float
    outer_wall_seconds: float

    def __post_init__(self) -> None:
        if type(self.metrics) is not tuple or tuple(
            metric.metric_id for metric in self.metrics
        ) != REQUIRED_METRIC_IDS:
            raise ValueError("evaluation must cover exact canonical Airfoil metrics")
        for metric in self.metrics:
            AirfoilObservedMetric.__post_init__(metric)
        for name in (
            "option_identity_sha256",
            "child_configuration_sha256",
            "terminal_record_sha256",
            "raw_receipt_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("active_wall_seconds", "outer_wall_seconds"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        """Return only requested outcome evidence; performance rank is absent."""

        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
            "valid": True,
            "metrics": [metric.to_record() for metric in self.metrics],
            "terminal_record_sha256": self.terminal_record_sha256,
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "active_wall_seconds_hex": self.active_wall_seconds.hex(),
            "outer_wall_seconds_hex": self.outer_wall_seconds.hex(),
        }


class AirfoilV7SealedOracleDevelopmentEvaluator:
    """Authenticated, rank-blind emulator for already-paid Airfoil CFD calls.

    The private table is never rendered into a prompt or preparation record.
    ``evaluate`` returns exactly the requested option IDs in caller order.
    """

    def __init__(self, oracle: VerifiedAirfoilPredecisionOracle) -> None:
        if type(oracle) is not VerifiedAirfoilPredecisionOracle:
            raise TypeError(
                "oracle must be an exact VerifiedAirfoilPredecisionOracle"
            )
        self._contract = oracle.contract
        self._oracle = oracle
        self._seal = oracle.seal_record()
        self._g1_option_ids: tuple[str, ...] | None = None
        self._postdecision_opened = False

    @property
    def contract(self) -> FiniteVariationContract:
        return self._contract

    def binding_record(self) -> dict[str, object]:
        return {
            "policy_id": EVALUATOR_POLICY_ID,
            "policy_version": EVALUATOR_POLICY_VERSION,
            "definition_sha256": EVALUATOR_DEFINITION_SHA256,
            "source_seal": dict(self._seal),
            "mode": "development_oracle_emulation",
            "provider_calls": 0,
            "new_cfd_calls": 0,
            "rank_exposed": False,
            "unselected_outcomes_exposed": False,
        }

    def firewall_record(self) -> dict[str, object]:
        if self._g1_option_ids is None:
            raise RuntimeError("initial G1 capability is not yet bound")
        return {
            "schema_version": 1,
            "evaluator_definition_sha256": EVALUATOR_DEFINITION_SHA256,
            "authenticated_seal": dict(self._seal),
            "g1_option_ids": list(self._g1_option_ids),
            "g1_outcomes_materialized": True,
            "g1_materialized_count": len(self._g1_option_ids),
            "non_g1_outcomes_materialized": False,
            "g2_opened": self._postdecision_opened,
            "predecision_oracle_result_json_decoded": False,
        }

    def authorize_initial_g1(self, sample: DiagnosticActionSample) -> None:
        """Irreversibly bind the online evaluation surface to exact G1 IDs."""

        if self._g1_option_ids is not None:
            raise RuntimeError("initial G1 evaluation capability is already bound")
        validate_diagnostic_action_sample(self._contract, sample)
        self._g1_option_ids = tuple(member.option_id for member in sample.members)

    def evaluate_g1(
        self,
        option_ids: Sequence[str],
    ) -> tuple[AirfoilDevelopmentEvaluation, ...]:
        """Evaluate only members of the prospectively bound G1 sample."""

        if self._g1_option_ids is None:
            raise RuntimeError("initial G1 evaluation capability is not bound")
        requested = tuple(option_ids)
        if not set(requested).issubset(self._g1_option_ids):
            raise PermissionError(
                "non-G1 outcomes are unavailable before an allocation commitment"
            )
        return self._evaluate_exact(requested)

    def _evaluate_exact(
        self,
        option_ids: Sequence[str],
    ) -> tuple[AirfoilDevelopmentEvaluation, ...]:
        if isinstance(option_ids, (str, bytes)) or not isinstance(
            option_ids, Sequence
        ):
            raise TypeError("option_ids must be a sequence of option IDs")
        requested = tuple(option_ids)
        if not requested or any(type(value) is not str for value in requested):
            raise ValueError("option_ids must contain exact non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("one evaluator batch cannot repeat an option")

        evaluations: list[AirfoilDevelopmentEvaluation] = []
        for option_id in requested:
            option = self._contract.resolve(option_id)
            terminal_path = self._oracle.terminal_path(option_id)
            terminal = _load_object(
                terminal_path,
                label=f"selected terminal {option_id}",
            )
            _verify_self_hash(
                terminal,
                field="record_sha256",
                framing=ORACLE_RECORD_FRAMING,
                label=f"selected terminal {option_id}",
            )
            ordinal = next(
                index
                for index, candidate in enumerate(self._contract.options, start=1)
                if candidate.option_id == option_id
            )
            if (
                terminal.get("disposition") != "success"
                or terminal.get("option_id") != option.option_id
                or terminal.get("ordinal") != ordinal
                or terminal.get("option_identity_sha256") != option.identity_sha256
                or terminal.get("typed_child_configuration_sha256")
                != option.child_configuration_sha256
            ):
                raise ValueError("selected terminal differs from the sealed option")
            payload = terminal.get("payload")
            raw_receipt = terminal.get("raw_receipt")
            if type(payload) is not dict or type(raw_receipt) is not dict:
                raise ValueError("sealed evaluation projection is malformed")
            objectives = payload.get("objectives")
            violations = payload.get("violations")
            if type(objectives) is not dict or type(violations) is not dict:
                raise ValueError("selected terminal lacks exact metrics")
            assert isinstance(objectives, dict)
            assert isinstance(violations, dict)
            assert isinstance(payload, dict)
            assert isinstance(raw_receipt, dict)
            metrics = tuple(
                sorted(
                    (
                        AirfoilObservedMetric(
                            metric_id=OBJECTIVE_METRIC_ID,
                            parent_value=float(PARENT_METRICS[OBJECTIVE_NAME]),
                            child_value=_finite(
                                objectives.get(OBJECTIVE_NAME),
                                OBJECTIVE_NAME,
                            ),
                        ),
                        AirfoilObservedMetric(
                            metric_id=VIOLATION_METRIC_ID,
                            parent_value=float(PARENT_METRICS[VIOLATION_NAME]),
                            child_value=_finite(
                                violations.get(VIOLATION_NAME),
                                VIOLATION_NAME,
                            ),
                        ),
                    ),
                    key=lambda metric: metric.metric_id,
                )
            )
            evaluations.append(
                AirfoilDevelopmentEvaluation(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    child_configuration_sha256=option.child_configuration_sha256,
                    family=option.family,
                    metrics=metrics,
                    terminal_record_sha256=str(terminal["record_sha256"]),
                    raw_receipt_sha256=str(raw_receipt["sha256"]),
                    active_wall_seconds=_finite(
                        payload.get("active_wall_seconds"),
                        "active_wall_seconds",
                    ),
                    outer_wall_seconds=_finite(
                        terminal.get("outer_wall_seconds"),
                        "outer_wall_seconds",
                    ),
                )
            )
        return tuple(evaluations)

    def open_postdecision_evaluation(
        self,
        commitment: AirfoilMpnAllocationCommitment,
    ) -> AirfoilV7PostDecisionEvaluationCapability:
        """Open one exact, one-shot G2 capability after durable allocation."""

        if self._postdecision_opened:
            raise RuntimeError("post-decision evaluation capability already opened")
        if type(commitment) is not AirfoilMpnAllocationCommitment:
            raise TypeError("commitment must be an exact issued allocation commitment")
        commitment.__post_init__()
        if commitment.finite_contract_identity_sha256 != self._contract.identity_sha256:
            raise ValueError("allocation commitment names another finite contract")
        if self._g1_option_ids is None:
            raise RuntimeError("G1 phase must be bound before post-decision evaluation")
        if set(commitment.selected_option_ids).intersection(self._g1_option_ids):
            raise ValueError("post-decision commitment contains a G1 action")
        self._postdecision_opened = True
        return AirfoilV7PostDecisionEvaluationCapability(self, commitment)

    def open_paired_postdecision_evaluation(
        self,
        commitment: AirfoilPairedAllocationCommitment,
    ) -> AirfoilV7PairedPostDecisionEvaluationCapability:
        """Open one union-only capability for the paired v2/v3 comparison."""

        if self._postdecision_opened:
            raise RuntimeError("post-decision evaluation capability already opened")
        if type(commitment) is not AirfoilPairedAllocationCommitment:
            raise TypeError("commitment must be an exact Airfoil paired commitment")
        commitment.__post_init__()
        if commitment.finite_contract_identity_sha256 != self._contract.identity_sha256:
            raise ValueError("paired commitment names another finite contract")
        if self._g1_option_ids is None:
            raise RuntimeError("G1 phase must be bound before post-decision evaluation")
        if set(commitment.selected_option_ids).intersection(self._g1_option_ids):
            raise ValueError("paired post-decision commitment contains a G1 action")
        self._postdecision_opened = True
        return AirfoilV7PairedPostDecisionEvaluationCapability(self, commitment)

    def open_postcommit_rank_reference(
        self,
        *,
        request: PostcommitRankRequest,
        authorization: PostcommitRankAuthorization,
    ) -> AirfoilV7PostcommitRankReferenceCapability:
        """Open a scalar-only eligible reference capability after authorization.

        Raw evaluations stay inside the capability.  The caller supplies one
        endpoint projection and receives only its finite scalar contribution
        plus the authenticated terminal receipt.
        """

        if self._postdecision_opened:
            raise RuntimeError("post-decision evaluation capability already opened")
        if type(request) is not PostcommitRankRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(authorization) is not PostcommitRankAuthorization:
            raise TypeError("authorization must be exact")
        authorization.__post_init__()
        if authorization.request_sha256 != request.request_sha256:
            raise ValueError("rank authorization names another request")
        if request.reference_source_sha256 != self._oracle.recursive_content_sha256:
            raise ValueError("rank request names another sealed oracle")
        if any(
            self._contract.resolve(option_id).option_id != option_id
            for option_id in request.eligible_item_ids
        ):
            raise ValueError("rank request contains an unknown Airfoil option")
        self._postdecision_opened = True
        return AirfoilV7PostcommitRankReferenceCapability(
            self,
            request=request,
        )


@dataclass(frozen=True, slots=True, init=False)
class AirfoilMpnAllocationCommitment:
    """Closed receipt for all three allocations after a durable phase commit."""

    finite_contract_identity_sha256: str
    phase_commit_receipt_sha256: str
    arm_allocation_pairs: tuple[tuple[str, str, str], ...]
    selected_option_ids: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("allocation commitments are issued by the benchmark adapter")

    def __post_init__(self) -> None:
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        require_sha256(
            self.phase_commit_receipt_sha256,
            "phase_commit_receipt_sha256",
        )
        if type(self.arm_allocation_pairs) is not tuple or tuple(
            value[0] for value in self.arm_allocation_pairs
        ) != ("m", "p", "n"):
            raise ValueError("allocation commitment requires canonical M/P/N arms")
        for arm, request_sha256, decision_sha256 in self.arm_allocation_pairs:
            if arm not in {"m", "n", "p"}:
                raise ValueError("allocation commitment contains a foreign arm")
            require_sha256(request_sha256, "allocation_request_sha256")
            require_sha256(decision_sha256, "allocation_decision_sha256")
        if (
            type(self.selected_option_ids) is not tuple
            or not self.selected_option_ids
            or self.selected_option_ids != tuple(sorted(set(self.selected_option_ids)))
        ):
            raise ValueError("selected option IDs must be non-empty and canonical")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "phase_commit_receipt_sha256": self.phase_commit_receipt_sha256,
            "arm_allocations": [
                {
                    "control_arm": arm,
                    "allocation_request_sha256": request_sha256,
                    "allocation_decision_sha256": decision_sha256,
                }
                for arm, request_sha256, decision_sha256 in self.arm_allocation_pairs
            ],
            "selected_option_ids": list(self.selected_option_ids),
        }

    @property
    def commitment_sha256(self) -> str:
        return _hash(_ALLOCATION_COMMITMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "commitment_sha256": self.commitment_sha256,
        }


def _issue_allocation_commitment(
    *,
    finite_contract_identity_sha256: str,
    phase_commit_receipt_sha256: str,
    arm_allocation_pairs: tuple[tuple[str, str, str], ...],
    selected_option_ids: tuple[str, ...],
) -> AirfoilMpnAllocationCommitment:
    commitment = object.__new__(AirfoilMpnAllocationCommitment)
    object.__setattr__(
        commitment,
        "finite_contract_identity_sha256",
        finite_contract_identity_sha256,
    )
    object.__setattr__(
        commitment,
        "phase_commit_receipt_sha256",
        phase_commit_receipt_sha256,
    )
    object.__setattr__(commitment, "arm_allocation_pairs", arm_allocation_pairs)
    object.__setattr__(commitment, "selected_option_ids", selected_option_ids)
    commitment.__post_init__()
    return commitment


class AirfoilV7PostDecisionEvaluationCapability:
    """One-shot access to exactly the durably committed union of G2 actions."""

    def __init__(
        self,
        evaluator: AirfoilV7SealedOracleDevelopmentEvaluator,
        commitment: AirfoilMpnAllocationCommitment,
    ) -> None:
        self._evaluator = evaluator
        self.commitment = commitment
        self._used = False

    def evaluate_selected(self) -> tuple[AirfoilDevelopmentEvaluation, ...]:
        if self._used:
            raise RuntimeError("post-decision evaluation capability is one-shot")
        self._used = True
        return self._evaluator._evaluate_exact(self.commitment.selected_option_ids)


@dataclass(frozen=True, slots=True, init=False)
class AirfoilPairedAllocationCommitment:
    """Closed Airfoil projection of the generic paired-method commitment."""

    finite_contract_identity_sha256: str
    paired_comparison_commitment_sha256: str
    schedule_binding_sha256: str
    method_commit_pairs: tuple[tuple[str, str], ...]
    logical_slot_count: int
    selected_option_ids: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("paired commitments are issued by the Airfoil adapter")

    def __post_init__(self) -> None:
        for name in (
            "finite_contract_identity_sha256",
            "paired_comparison_commitment_sha256",
            "schedule_binding_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.method_commit_pairs) is not tuple or len(
            self.method_commit_pairs
        ) != 2:
            raise ValueError("Airfoil paired commitment requires exactly two methods")
        if self.method_commit_pairs != tuple(sorted(self.method_commit_pairs)):
            raise ValueError("Airfoil method commitments must be canonical")
        if len({value[0] for value in self.method_commit_pairs}) != 2:
            raise ValueError("Airfoil comparison method IDs cannot repeat")
        for method_id, phase_receipt in self.method_commit_pairs:
            if type(method_id) is not str or not method_id:
                raise ValueError("Airfoil comparison method ID must be non-empty")
            require_sha256(phase_receipt, "allocation phase receipt")
        if type(self.logical_slot_count) is not int or not (
            1 <= self.logical_slot_count <= 18
        ):
            raise ValueError("Airfoil paired logical slots must lie in [1,18]")
        if (
            type(self.selected_option_ids) is not tuple
            or not self.selected_option_ids
            or self.selected_option_ids != tuple(sorted(set(self.selected_option_ids)))
            or len(self.selected_option_ids) > self.logical_slot_count
        ):
            raise ValueError(
                "Airfoil selected union must be canonical and slot-bounded"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "paired_comparison_commitment_sha256": (
                self.paired_comparison_commitment_sha256
            ),
            "schedule_binding_sha256": self.schedule_binding_sha256,
            "method_commits": [
                {
                    "comparison_method_id": method_id,
                    "allocation_phase_commit_receipt_sha256": phase_receipt,
                }
                for method_id, phase_receipt in self.method_commit_pairs
            ],
            "logical_slot_count": self.logical_slot_count,
            "selected_option_ids": list(self.selected_option_ids),
            "raw_outcome_authority": "selected_union_only",
            "unselected_outcomes_exposed": False,
        }

    @property
    def commitment_sha256(self) -> str:
        return _hash(
            _PAIRED_ALLOCATION_COMMITMENT_DOMAIN,
            self._unsigned_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "commitment_sha256": self.commitment_sha256,
        }


def _issue_airfoil_paired_allocation_commitment(
    *,
    finite_contract_identity_sha256: str,
    paired_comparison_commitment_sha256: str,
    schedule_binding_sha256: str,
    method_commit_pairs: tuple[tuple[str, str], ...],
    logical_slot_count: int,
    selected_option_ids: tuple[str, ...],
) -> AirfoilPairedAllocationCommitment:
    result = object.__new__(AirfoilPairedAllocationCommitment)
    for name, value in (
        ("finite_contract_identity_sha256", finite_contract_identity_sha256),
        (
            "paired_comparison_commitment_sha256",
            paired_comparison_commitment_sha256,
        ),
        ("schedule_binding_sha256", schedule_binding_sha256),
        ("method_commit_pairs", method_commit_pairs),
        ("logical_slot_count", logical_slot_count),
        ("selected_option_ids", selected_option_ids),
    ):
        object.__setattr__(result, name, value)
    result.__post_init__()
    return result


class AirfoilV7PairedPostDecisionEvaluationCapability:
    """One-shot raw access to only the committed v2/v3 selected union."""

    def __init__(
        self,
        evaluator: AirfoilV7SealedOracleDevelopmentEvaluator,
        commitment: AirfoilPairedAllocationCommitment,
    ) -> None:
        self._evaluator = evaluator
        self.commitment = commitment
        self._used = False

    def evaluate_selected_union(self) -> tuple[AirfoilDevelopmentEvaluation, ...]:
        if self._used:
            raise RuntimeError("paired post-decision evaluation capability is one-shot")
        self._used = True
        return self._evaluator._evaluate_exact(self.commitment.selected_option_ids)


class AirfoilV7PostcommitRankReferenceCapability:
    """One-read-per-item scalar projection over an authorized eligible set."""

    def __init__(
        self,
        evaluator: AirfoilV7SealedOracleDevelopmentEvaluator,
        *,
        request: PostcommitRankRequest,
    ) -> None:
        self._evaluator = evaluator
        self._eligible_item_ids = request.eligible_item_ids
        self._read_item_ids: set[str] = set()

    @property
    def exact_read_count(self) -> int:
        return len(self._read_item_ids)

    def evaluate_component(
        self,
        item_id: str,
        projector: Callable[[AirfoilDevelopmentEvaluation], float],
    ) -> RankReferenceObservation:
        if type(item_id) is not str or item_id not in self._eligible_item_ids:
            raise PermissionError("rank reference item is outside the eligible set")
        if item_id in self._read_item_ids:
            raise RuntimeError("rank reference item cannot be decoded twice")
        if not callable(projector):
            raise TypeError("rank endpoint projector must be callable")
        self._read_item_ids.add(item_id)
        evaluation = self._evaluator._evaluate_exact((item_id,))[0]
        component = projector(evaluation)
        if type(component) is not float or not math.isfinite(component):
            raise TypeError("rank endpoint projector must return a finite float")
        return RankReferenceObservation(
            item_id=item_id,
            endpoint_component=component,
            source_receipt_sha256=evaluation.terminal_record_sha256,
        )


@dataclass(frozen=True, slots=True)
class AirfoilG1ActionObservation:
    """One source-bound G1 action/outcome contrast."""

    diagnostic_rank: int
    operator_invocation_id: OperatorInvocationId
    parent_candidate_id: CandidateId
    child_candidate_id: CandidateId
    option_id: str
    family: str
    option_identity_sha256: str
    child_configuration_sha256: str
    evaluation: AirfoilDevelopmentEvaluation
    contrast_id: str
    action_binding: FiniteActionEvidenceBinding

    def __post_init__(self) -> None:
        if type(self.diagnostic_rank) is not int or self.diagnostic_rank <= 0:
            raise ValueError("diagnostic_rank must be a positive exact integer")
        if type(self.operator_invocation_id) is not OperatorInvocationId:
            raise TypeError("operator_invocation_id must be exact")
        if type(self.parent_candidate_id) is not CandidateId or type(
            self.child_candidate_id
        ) is not CandidateId:
            raise TypeError("candidate IDs must be exact CandidateId values")
        if self.evaluation.option_id != self.option_id:
            raise ValueError("observation evaluation names another action")
        if (
            self.evaluation.option_identity_sha256 != self.option_identity_sha256
            or self.evaluation.child_configuration_sha256
            != self.child_configuration_sha256
            or self.evaluation.family != self.family
        ):
            raise ValueError("observation option binding differs from its evaluation")
        if self.action_binding.contrast_id != self.contrast_id:
            raise ValueError("action binding names another contrast")
        if (
            self.action_binding.option_id != self.option_id
            or self.action_binding.option_identity_sha256
            != self.option_identity_sha256
            or self.action_binding.family != self.family
        ):
            raise ValueError("action binding differs from the observed option")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "diagnostic_rank": self.diagnostic_rank,
            "operator_invocation_id": self.operator_invocation_id.value,
            "parent_candidate_id": self.parent_candidate_id.value,
            "child_candidate_id": self.child_candidate_id.value,
            "contrast_id": self.contrast_id,
            "finite_action_evidence": self.action_binding.to_record(),
            "evaluation": self.evaluation.to_record(),
        }

    def prompt_record(self, contract: FiniteVariationContract) -> dict[str, object]:
        """Expose this action only; no oracle-wide facts or rank fields."""

        self.__post_init__()
        option = contract.resolve(self.option_id)
        if option.identity_sha256 != self.option_identity_sha256:
            raise ValueError("prompt contract differs from the observation")
        return {
            "diagnostic_index": self.diagnostic_rank,
            "contrast_id": self.contrast_id,
            "executed_action": option.prompt_record(),
            "finite_action_evidence": self.action_binding.to_record(),
            "valid": True,
            "observed_metrics": [
                metric.prompt_record() for metric in self.evaluation.metrics
            ],
            "evaluation_receipt_sha256": self.evaluation.terminal_record_sha256,
        }

    def empirical_evidence_snapshot(self) -> EmpiricalEvidenceSnapshot:
        """Issue rank-free facts without copying finite-action attribution."""

        self.__post_init__()
        return EmpiricalEvidenceSnapshot(
            contrast_id=self.contrast_id,
            fact_schema_id=_EMPIRICAL_FACT_SCHEMA_ID,
            fact_schema_version=_EMPIRICAL_FACT_SCHEMA_VERSION,
            fact_schema_definition_sha256=(
                _EMPIRICAL_FACT_SCHEMA_DEFINITION_SHA256
            ),
            facts=_frozen_object(
                {
                    "valid": True,
                    "observed_metric_deltas": [
                        metric.prompt_record()
                        for metric in self.evaluation.metrics
                    ],
                    "evaluation_receipt_sha256": (
                        self.evaluation.terminal_record_sha256
                    ),
                }
            ),
            optimization_semantics_definition_sha256=(
                AIRFOIL_V7_OPTIMIZATION_SEMANTICS.definition_sha256
            ),
            action_semantics_definition_sha256=(
                AIRFOIL_V7_ACTION_SEMANTICS.definition_sha256
            ),
        )


def _build_observations(
    *,
    contract: FiniteVariationContract,
    sample: DiagnosticActionSample,
    evaluations: tuple[AirfoilDevelopmentEvaluation, ...],
) -> tuple[AirfoilG1ActionObservation, ...]:
    if len(evaluations) != len(sample.members):
        raise ValueError("G1 evaluations differ from the frozen sample size")
    parent_candidate_id = CandidateId("candidate_airfoil_twostage_parent")
    observations: list[AirfoilG1ActionObservation] = []
    for member, evaluation in zip(sample.members, evaluations, strict=True):
        if member.option_id != evaluation.option_id:
            raise ValueError("G1 evaluator reordered the frozen sample")
        operator_id = OperatorInvocationId(
            f"operator_airfoil_twostage_g1_{member.rank:02d}"
        )
        child_id = CandidateId(f"candidate_airfoil_twostage_g1_{member.rank:02d}")
        contrast_unsigned = {
            "schema_version": 1,
            "sample_receipt_sha256": sample.receipt_sha256,
            "diagnostic_rank": member.rank,
            "operator_invocation_id": operator_id.value,
            "parent_candidate_id": parent_candidate_id.value,
            "child_candidate_id": child_id.value,
            "option_identity_sha256": member.option_identity_sha256,
            "child_configuration_sha256": member.child_configuration_sha256,
            "terminal_record_sha256": evaluation.terminal_record_sha256,
            "metrics": [metric.to_record() for metric in evaluation.metrics],
        }
        contrast_id = _hash(_CONTRAST_DOMAIN, contrast_unsigned)
        binding = bind_finite_action_evidence(
            contrast_id=contrast_id,
            contract=contract,
            option_id=member.option_id,
        )
        observations.append(
            AirfoilG1ActionObservation(
                diagnostic_rank=member.rank,
                operator_invocation_id=operator_id,
                parent_candidate_id=parent_candidate_id,
                child_candidate_id=child_id,
                option_id=member.option_id,
                family=member.family,
                option_identity_sha256=member.option_identity_sha256,
                child_configuration_sha256=member.child_configuration_sha256,
                evaluation=evaluation,
                contrast_id=contrast_id,
                action_binding=binding,
            )
        )
    return tuple(observations)


def _reflection_instruction() -> str:
    return (
        "Infer exactly one falsifiable intervention card per observed contrast. "
        "Each card must cite only its full contrast_id, recommend exactly the "
        "executed option_id and its family, predict child-minus-parent direction "
        "for every required metric, describe a mechanism rather than merely "
        "restate numbers, and include a concrete held-out falsification condition. "
        "Treat mechanism prose as an unverified hypothesis; observed metrics "
        "remain authoritative facts. "
        "Do not infer or describe outcomes for actions absent from the observed "
        "contrast set. Return exactly eight distinct cards."
    )


def _reflection_prompt(
    *,
    contract: FiniteVariationContract,
    observations: tuple[AirfoilG1ActionObservation, ...],
    insight_contract: ReflectionInsightContract,
) -> str:
    record = {
        "instruction": _reflection_instruction(),
        "optimization_semantics": AIRFOIL_V7_OPTIMIZATION_SEMANTICS.to_record(),
        "action_semantics": AIRFOIL_V7_ACTION_SEMANTICS.to_record(),
        "parent_metric_values": {
            OBJECTIVE_METRIC_ID: float(PARENT_METRICS[OBJECTIVE_NAME]),
            VIOLATION_METRIC_ID: float(PARENT_METRICS[VIOLATION_NAME]),
        },
        "reflection_insight_contract": insight_contract.to_record(),
        "observed_contrasts": [
            observation.prompt_record(contract) for observation in observations
        ],
    }
    return _canonical_bytes(record).decode("ascii")


def _build_reflection_request(
    *,
    contract: FiniteVariationContract,
    observations: tuple[AirfoilG1ActionObservation, ...],
) -> ReflectionWorkflowRequest:
    insight_contract = ReflectionInsightContract(
        required_metric_ids=REQUIRED_METRIC_IDS,
        allowed_option_families=tuple(
            sorted({observation.family for observation in observations})
        ),
        allowed_option_ids=tuple(
            sorted(observation.option_id for observation in observations)
        ),
    )
    batch_prompt = _reflection_prompt(
        contract=contract,
        observations=observations,
        insight_contract=insight_contract,
    )
    shards = tuple(
        ReflectionPromptShard(
            contrast_id=observation.contrast_id,
            prompt=_reflection_prompt(
                contract=contract,
                observations=(observation,),
                insight_contract=ReflectionInsightContract(
                    required_metric_ids=REQUIRED_METRIC_IDS,
                    allowed_option_families=(observation.family,),
                    allowed_option_ids=(observation.option_id,),
                ),
            ),
        )
        for observation in observations
    )
    return ReflectionWorkflowRequest(
        operation="extract_insights",
        shards=shards,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        insight_contract=insight_contract,
        batch_prompt=batch_prompt,
    )


def _parent_metric_values() -> tuple[ParentMetricValue, ...]:
    return tuple(
        sorted(
            (
                ParentMetricValue(
                    OBJECTIVE_METRIC_ID,
                    float(PARENT_METRICS[OBJECTIVE_NAME]),
                ),
                ParentMetricValue(
                    VIOLATION_METRIC_ID,
                    float(PARENT_METRICS[VIOLATION_NAME]),
                ),
            ),
            key=lambda value: value.metric_id,
        )
    )


def _metric_scales() -> tuple[MetricForecastScale, ...]:
    definitions = {
        OBJECTIVE_METRIC_ID: (
            DELTA_F,
            "Airfoil-v7 preregistered practically resolved drag delta_f=0.001.",
        ),
        VIOLATION_METRIC_ID: (
            DELTA_V,
            "Airfoil-v7 preregistered practically resolved violation delta_v=0.005.",
        ),
    }
    return tuple(
        MetricForecastScale(
            metric_id=metric_id,
            delta_scale=float(value),
            definition_sha256=hashlib.sha256(description.encode("ascii")).hexdigest(),
        )
        for metric_id, (value, description) in sorted(definitions.items())
    )


@dataclass(frozen=True, slots=True)
class AirfoilV7ForecastPortfolioUtility:
    """Airfoil-owned probability-like usefulness acquisition over a set.

    The result is explicitly bounded in ``[0,1]``.  Consequently the generic
    allocator's frozen risk coefficient ``0.5`` and diversity coefficient
    ``0.25`` have stable, inspectable scales instead of depending on the raw
    magnitude of Airfoil forecast deltas.
    """

    violation_weight: float = 1.0
    objective_weight: float = 0.05
    standardized_clip: float = 60.0

    def __post_init__(self) -> None:
        expected = (1.0, 0.05, 60.0)
        observed = (
            self.violation_weight,
            self.objective_weight,
            self.standardized_clip,
        )
        if observed != expected:
            raise ValueError("Airfoil-v7 utility parameters are frozen")

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        self.__post_init__()
        if request.optimization_semantics.identity != (
            AIRFOIL_V7_OPTIMIZATION_SEMANTICS.identity
        ):
            raise ValueError("utility received another optimization semantics")
        parent = {value.metric_id: value.value for value in request.parent_metric_values}
        scales = {value.metric_id: value.delta_scale for value in request.metric_scales}
        if set(parent) != set(REQUIRED_METRIC_IDS) or scales != {
            OBJECTIVE_METRIC_ID: float(DELTA_F),
            VIOLATION_METRIC_ID: float(DELTA_V),
        }:
            raise ValueError("utility received another metric/scaling contract")
        attribute = {
            ForecastQuantile.P10: "p10_delta",
            ForecastQuantile.P50: "p50_delta",
            ForecastQuantile.P90: "p90_delta",
        }[request.quantile]
        joint_failure_probability = 1.0
        for member in request.members:
            forecast = {
                value.metric_id: value for value in member.metric_forecasts
            }
            if set(forecast) != set(REQUIRED_METRIC_IDS):
                raise ValueError("utility member has incomplete metric forecasts")
            delta_v = getattr(forecast[VIOLATION_METRIC_ID], attribute)
            delta_f = getattr(forecast[OBJECTIVE_METRIC_ID], attribute)
            quality = max(
                -self.standardized_clip,
                min(
                    self.standardized_clip,
                    -self.violation_weight * delta_v / DELTA_V,
                ),
            ) + self.objective_weight * math.tanh(-delta_f / DELTA_F)
            usefulness = (
                1.0 / (1.0 + math.exp(-quality))
                if quality >= 0.0
                else math.exp(quality) / (1.0 + math.exp(quality))
            )
            effective_success = member.probability_valid * usefulness
            joint_failure_probability *= 1.0 - effective_success
        result = 1.0 - joint_failure_probability
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError("Airfoil forecast utility escaped [0,1]")
        return float(result)


AIRFOIL_V7_FORECAST_UTILITY = ForecastPortfolioUtilityBinding(
    utility=AirfoilV7ForecastPortfolioUtility(),
    policy_id=UTILITY_POLICY_ID,
    policy_version=UTILITY_POLICY_VERSION,
    definition_sha256=UTILITY_DEFINITION_SHA256,
)


@dataclass(frozen=True, slots=True)
class PreparedAirfoilTwoStageGeneration:
    """Complete provider-free state immediately before batched reflection."""

    contract: FiniteVariationContract
    sample: DiagnosticActionSample
    observations: tuple[AirfoilG1ActionObservation, ...]
    reflection_request: ReflectionWorkflowRequest
    parent_metric_values: tuple[ParentMetricValue, ...]
    metric_scales: tuple[MetricForecastScale, ...]
    eligible_g2_option_ids: tuple[str, ...]
    oracle_seal: FrozenJsonObject
    evaluator: AirfoilV7SealedOracleDevelopmentEvaluator = field(
        repr=False,
        compare=False,
    )
    utility: ForecastPortfolioUtilityBinding = field(
        default=AIRFOIL_V7_FORECAST_UTILITY,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validate_diagnostic_action_sample(self.contract, self.sample)
        if type(self.observations) is not tuple or len(self.observations) != G1_SAMPLE_SIZE:
            raise ValueError("preparation requires exactly eight G1 observations")
        if tuple(value.diagnostic_rank for value in self.observations) != tuple(
            range(1, G1_SAMPLE_SIZE + 1)
        ):
            raise ValueError("G1 observations must retain sample order")
        sampled = {member.option_id for member in self.sample.members}
        expected_eligible = tuple(
            sorted(
                option.option_id
                for option in self.contract.options
                if option.option_id not in sampled
            )
        )
        if self.eligible_g2_option_ids != expected_eligible:
            raise ValueError("G2 eligibility must be exactly the non-G1 actions")
        if len(self.eligible_g2_option_ids) != len(self.contract.options) - G1_SAMPLE_SIZE:
            raise ValueError("G2 eligibility has the wrong cardinality")
        if self.reflection_request.max_output_tokens != MAX_OUTPUT_TOKENS:
            raise ValueError("reflection output budget changed")
        if self.reflection_request.insight_contract is None:
            raise ValueError("reflection requires an exact insight contract")
        if self.evaluator.contract != self.contract:
            raise ValueError("development evaluator names another finite contract")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        prompt = self.reflection_request.batch_prompt
        assert prompt is not None
        return {
            "schema_version": 1,
            "policy_id": PREPARATION_POLICY_ID,
            "policy_version": PREPARATION_POLICY_VERSION,
            "definition_sha256": PREPARATION_DEFINITION_SHA256,
            "finite_contract_identity_sha256": self.contract.identity_sha256,
            "finite_option_count": len(self.contract.options),
            "optimization_semantics": AIRFOIL_V7_OPTIMIZATION_SEMANTICS.to_record(),
            "g1_sample": self.sample.to_record(),
            "g1_observations": [value.to_record() for value in self.observations],
            "reflection": {
                "workflow_policy": "strict_batched_reflection",
                "logical_call_count": 1,
                "contrast_ids": sorted(
                    value.contrast_id for value in self.observations
                ),
                "batch_prompt_sha256": hashlib.sha256(prompt.encode("ascii")).hexdigest(),
                "max_output_tokens": self.reflection_request.max_output_tokens,
                "temperature": self.reflection_request.temperature,
                "insight_contract": (
                    self.reflection_request.insight_contract.to_record()
                ),
            },
            "parent_metric_values": [
                value.to_record() for value in self.parent_metric_values
            ],
            "metric_scales": [value.to_record() for value in self.metric_scales],
            "g2_eligible_option_ids": list(self.eligible_g2_option_ids),
            "g2_eligible_count": len(self.eligible_g2_option_ids),
            "oracle_seal": dict(self.oracle_seal.items),
            "evaluator": self.evaluator.binding_record(),
            "predecision_firewall": self.evaluator.firewall_record(),
            "utility": self.utility.to_record(),
            "provider_calls": 0,
            "credentials_read": False,
        }


def prepare_airfoil_v7_two_stage_generation(
    oracle_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> PreparedAirfoilTwoStageGeneration:
    """Authenticate inputs, emulate exactly eight G1 evaluations, and stop."""

    oracle = verify_airfoil_v7_predecision_oracle(oracle_dir)
    evaluator = AirfoilV7SealedOracleDevelopmentEvaluator(oracle)
    contract = oracle.contract
    sample = HashStratifiedDiagnosticSampler(
        seed=G1_SAMPLE_SEED,
        design_key=G1_SAMPLE_DESIGN_KEY,
    ).sample(contract, sample_size=G1_SAMPLE_SIZE)
    validate_diagnostic_action_sample(contract, sample)
    evaluator.authorize_initial_g1(sample)
    evaluations = evaluator.evaluate_g1(
        tuple(member.option_id for member in sample.members)
    )
    observations = _build_observations(
        contract=contract,
        sample=sample,
        evaluations=evaluations,
    )
    sampled = {member.option_id for member in sample.members}
    preparation = PreparedAirfoilTwoStageGeneration(
        contract=contract,
        sample=sample,
        observations=observations,
        reflection_request=_build_reflection_request(
            contract=contract,
            observations=observations,
        ),
        parent_metric_values=_parent_metric_values(),
        metric_scales=_metric_scales(),
        eligible_g2_option_ids=tuple(
            sorted(
                option.option_id
                for option in contract.options
                if option.option_id not in sampled
            )
        ),
        oracle_seal=_frozen_object(oracle.seal_record()),
        evaluator=evaluator,
    )
    preparation.__post_init__()
    return preparation


def _replace_exact_identities(text: str, contract: FiniteVariationContract) -> str:
    """Remove exact action/contract attribution while preserving useful prose."""

    if type(text) is not str:
        raise TypeError("card prose must be exact text")
    result = text
    for option in sorted(
        contract.options,
        key=lambda value: len(value.option_id),
        reverse=True,
    ):
        result = re.sub(
            re.escape(option.option_id),
            _IDENTITY_SENTINEL,
            result,
            flags=re.IGNORECASE,
        )
    forbidden_hashes = {
        contract.identity_sha256,
        *(option.identity_sha256 for option in contract.options),
        *(option.child_configuration_sha256 for option in contract.options),
    }
    for value in forbidden_hashes:
        result = re.sub(re.escape(value), _IDENTITY_SENTINEL, result, flags=re.IGNORECASE)
    return result


def _scrubbed_unverified_hypothesis(
    draft: InsightDraft,
    contract: FiniteVariationContract,
) -> FrozenJsonObject:
    """Project model prose while removing its exact option attribution."""

    draft.__post_init__()
    record = draft.hypothesis_record()
    record.pop("recommended_option_ids", None)
    record.pop("recommended_option_families", None)
    record.pop("evidence_contrast_ids", None)

    def scrub(value: object) -> object:
        if type(value) is str:
            return _replace_exact_identities(value, contract)
        if type(value) is list:
            return [scrub(item) for item in value]
        if type(value) is dict:
            return {str(key): scrub(item) for key, item in value.items()}
        return value

    scrubbed = scrub(record)
    if type(scrubbed) is not dict:
        raise AssertionError("hypothesis projection must remain an object")
    return _frozen_object(scrubbed)


def _epistemic_card_payload(
    draft: InsightDraft,
    contract: FiniteVariationContract,
    empirical_evidence: tuple[EmpiricalEvidenceSnapshot, ...],
) -> FrozenJsonObject:
    """Pair trusted metric facts with separately labeled model hypotheses."""

    hypothesis = thaw_json(
        _scrubbed_unverified_hypothesis(draft, contract)
    )
    if type(hypothesis) is not dict:
        raise AssertionError("hypothesis projection must thaw to an object")
    hypothesis["empirical_snapshot_sha256s"] = [
        snapshot.snapshot_sha256 for snapshot in empirical_evidence
    ]
    composed = compose_epistemic_prompt_payload(
        empirical_evidence=empirical_evidence,
        hypothesis=_frozen_object(hypothesis),
    )
    prompt_projection = thaw_json(composed)
    if type(prompt_projection) is not dict:
        raise AssertionError("epistemic payload must thaw to an object")
    facts = prompt_projection.get("empirical_facts")
    if type(facts) is not list:
        raise AssertionError("epistemic payload must contain empirical facts")
    for fact in facts:
        if type(fact) is not dict:
            raise AssertionError("empirical fact projection must be an object")
        fact.pop("contrast_id", None)
        fact["contrast_binding"] = "structured_finite_action_evidence"
    return _frozen_object(prompt_projection)


def _forecast_context(preparation: PreparedAirfoilTwoStageGeneration) -> FrozenJsonObject:
    return _frozen_object(
        {
            "benchmark": "airfoil_v7",
            "stage": "g2_all_option_forecast",
            "development_evidence_boundary": "eight_requested_g1_outcomes_only",
            "g1_sample_receipt_sha256": preparation.sample.receipt_sha256,
            "g1_observation_count": len(preparation.observations),
            "target_option_count": len(preparation.contract.options),
            "allocation_eligible_option_count": len(
                preparation.eligible_g2_option_ids
            ),
            "source_oracle_result_file_sha256": dict(preparation.oracle_seal.items)[
                "oracle_result_file_sha256"
            ],
        }
    )


def _forecast_instruction() -> str:
    return (
        "Forecast every sealed target action using only the closed ordinal "
        "validity, median-effect, and asymmetric-uncertainty codes. Use the "
        "published optimization semantics and metric scales; trusted code "
        "derives p10/p50/p90 child-minus-parent deltas. When evidence cards are "
        "present, choose exactly one primary prompt-visible evidence slot for "
        "every option-metric cell; analogical evidence may support a different "
        "target action. When cards are absent, emit no evidence slot. Follow "
        "the positional output contract exactly: probability_valid_codes[i] "
        "maps to ordered option i, and every code matrix cell [i][j] maps to "
        "ordered option i and required metric j. Do not omit, add, or reorder "
        "vector entries, matrix rows, or matrix cells."
    )


def _forecast_request(
    *,
    preparation: PreparedAirfoilTwoStageGeneration,
    call_id: LLMCallId,
    cards: tuple[PortfolioCard, ...],
    registry: PortfolioCardSourceRegistry | None,
    evidence_mode: ActionForecastEvidenceMode,
    receipt: PortfolioExperimentalViewReceipt | None,
) -> ActionForecastRequest:
    return ActionForecastRequest(
        call_id=call_id,
        operation="forecast_all_actions",
        instruction=_forecast_instruction(),
        context=_forecast_context(preparation),
        optimization_semantics=AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
        action_semantics=AIRFOIL_V7_ACTION_SEMANTICS,
        finite_variation_contract=preparation.contract,
        cards=cards,
        source_registry=registry,
        evidence_mode=evidence_mode,
        experimental_view_receipt=receipt,
        parent_metric_values=preparation.parent_metric_values,
        metric_scales=preparation.metric_scales,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
    )


@dataclass(frozen=True, slots=True)
class AirfoilTwoStageForecastArms:
    """Source-bound M/P and card-free N requests after live reflection."""

    preparation: PreparedAirfoilTwoStageGeneration
    entries: tuple[InsightMemoryEntry, ...]
    source_cards: tuple[PortfolioCard, ...]
    placebo_cards: tuple[PortfolioCard, ...]
    source_registry: PortfolioCardSourceRegistry
    memory_receipt: PortfolioExperimentalViewReceipt
    placebo_receipt: PortfolioExperimentalViewReceipt
    memory_request: ActionForecastRequest
    placebo_request: ActionForecastRequest
    catalog_only_request: ActionForecastRequest

    def __post_init__(self) -> None:
        if len(self.entries) != G1_SAMPLE_SIZE:
            raise ValueError("forecast arms require eight reflected entries")
        if len(self.source_cards) != len(self.entries) or len(
            self.placebo_cards
        ) != len(self.entries):
            raise ValueError("M/P card populations must be complete")
        if self.memory_receipt.arm is not PortfolioExperimentalArm.MEMORY:
            raise ValueError("memory receipt names another arm")
        if self.placebo_receipt.arm is not PortfolioExperimentalArm.PERMUTED_PLACEBO:
            raise ValueError("placebo receipt names another arm")
        if self.memory_request.evidence_mode is not ActionForecastEvidenceMode.GROUNDED:
            raise ValueError("M must be grounded")
        if self.placebo_request.evidence_mode is not ActionForecastEvidenceMode.GROUNDED:
            raise ValueError("P must be grounded")
        if (
            self.catalog_only_request.evidence_mode
            is not ActionForecastEvidenceMode.CATALOG_ONLY
            or self.catalog_only_request.cards
            or self.catalog_only_request.source_registry is not None
            or self.catalog_only_request.experimental_view_receipt is not None
        ):
            raise ValueError("N must be genuinely catalog-only")
        contexts = {
            request.context_sha256
            for request in (
                self.memory_request,
                self.placebo_request,
                self.catalog_only_request,
            )
        }
        if len(contexts) != 1:
            raise ValueError("M/P/N requests must share one non-arm context")

    def request(self, arm: str) -> ActionForecastRequest:
        return {
            "m": self.memory_request,
            "p": self.placebo_request,
            "n": self.catalog_only_request,
        }[arm.casefold()]

    def allocation_request(
        self,
        arm: str,
        forecasts: ResolvedActionForecastBatch,
        *,
        portfolio_size: int = G2_PORTFOLIO_SIZE,
    ) -> ActionAllocationRequest:
        return ActionAllocationRequest(
            forecast_request=self.request(arm),
            forecasts=forecasts,
            eligible_option_ids=self.preparation.eligible_g2_option_ids,
            portfolio_size=portfolio_size,
            utility=self.preparation.utility,
        )

    def open_postdecision_evaluation(
        self,
        durable_allocation_phase_commit: TwoStageActionPhaseCommit,
    ) -> AirfoilV7PostDecisionEvaluationCapability:
        """Open G2 from the allocation commit after the sink has fsynced it."""

        commitment = bind_airfoil_mpn_allocation_commitment(
            self,
            durable_allocation_phase_commit,
        )
        return self.preparation.evaluator.open_postdecision_evaluation(commitment)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_registry": self.source_registry.to_record(),
            "entries": [
                {
                    "reference": {
                        "insight_id": entry.reference.insight_id.value,
                        "version": entry.reference.version,
                    },
                    "content_sha256": entry.draft.content_sha256,
                    "lineage": entry.evidence_lineage.to_record()
                    if entry.evidence_lineage is not None
                    else None,
                }
                for entry in self.entries
            ],
            "arms": {
                "m": {
                    "request_sha256": self.memory_request.request_sha256,
                    "view_receipt": self.memory_receipt.to_record(),
                    "card_snapshot_sha256": self.memory_request.card_snapshot_sha256,
                },
                "p": {
                    "request_sha256": self.placebo_request.request_sha256,
                    "view_receipt": self.placebo_receipt.to_record(),
                    "card_snapshot_sha256": self.placebo_request.card_snapshot_sha256,
                },
                "n": {
                    "request_sha256": self.catalog_only_request.request_sha256,
                    "evidence_mode": "catalog_only",
                    "card_count": 0,
                    "source_registry": None,
                    "experimental_view_receipt": None,
                },
            },
            "target_option_count_per_arm": len(self.preparation.contract.options),
            "eligible_g2_option_ids": list(
                self.preparation.eligible_g2_option_ids
            ),
            "eligible_g2_count": len(self.preparation.eligible_g2_option_ids),
            "max_output_tokens_per_forecast": MAX_OUTPUT_TOKENS,
            "provider_calls_completed_by_this_builder": 0,
        }


def bind_airfoil_mpn_allocation_commitment(
    arms: AirfoilTwoStageForecastArms,
    durable_allocation_phase_commit: TwoStageActionPhaseCommit,
) -> AirfoilMpnAllocationCommitment:
    """Issue the only capability that can reveal selected G2 outcomes.

    This function is intended to be called inside the phase-commit sink *after*
    that sink has written and fsynced ``durable_allocation_phase_commit``.  It
    accepts the actual generic commit object, validates its hash-bound payload,
    exact canonical M/P/N order, request bindings, and selected finite actions,
    then issues a closed benchmark capability.  An arbitrary SHA string cannot
    open the firewall.
    """

    if type(arms) is not AirfoilTwoStageForecastArms:
        raise TypeError("arms must be exact AirfoilTwoStageForecastArms")
    arms.__post_init__()
    if type(durable_allocation_phase_commit) is not TwoStageActionPhaseCommit:
        raise TypeError("phase commit must be exact TwoStageActionPhaseCommit")
    durable_allocation_phase_commit.__post_init__()
    if (
        durable_allocation_phase_commit.receipt.phase
        is not TwoStageActionPhase.ALLOCATE
    ):
        raise ValueError("only an ALLOCATE phase commit can open G2 evaluation")
    payload = thaw_json(durable_allocation_phase_commit.payload)
    if (
        type(payload) is not dict
        or payload.get("phase") != TwoStageActionPhase.ALLOCATE.value
        or type(payload.get("arm_executions")) is not list
    ):
        raise ValueError("allocation phase payload is malformed")
    executions = payload["arm_executions"]
    assert type(executions) is list
    if [
        execution.get("arm") if type(execution) is dict else None
        for execution in executions
    ] != ["m", "p", "n"]:
        raise ValueError("allocation phase must use canonical M/P/N order")
    pairs: list[tuple[str, str, str]] = []
    selected: set[str] = set()
    for arm, execution in zip(("m", "p", "n"), executions, strict=True):
        if type(execution) is not dict:
            raise ValueError("allocation arm execution is malformed")
        request = execution.get("allocation_request")
        decision = execution.get("decision")
        if type(request) is not dict or type(decision) is not dict:
            raise ValueError("allocation arm lacks request/decision payloads")
        expected_forecast = arms.request(arm)
        expected_eligible_sha256 = _hash(
            _ELIGIBLE_ACTION_SET_DOMAIN,
            {
                "eligible_option_ids": list(
                    arms.preparation.eligible_g2_option_ids
                )
            },
        )
        if (
            request.get("forecast_request_sha256") != expected_forecast.request_sha256
            or request.get("eligible_option_ids")
            != list(arms.preparation.eligible_g2_option_ids)
            or request.get("eligible_options_sha256")
            != expected_eligible_sha256
            or request.get("portfolio_size") != G2_PORTFOLIO_SIZE
            or request.get("utility") != arms.preparation.utility.to_record()
        ):
            raise ValueError("committed allocation differs from its frozen arm")
        request_sha256 = _hash(_ACTION_ALLOCATION_REQUEST_DOMAIN, request)
        if decision.get("allocation_request_sha256") != request_sha256:
            raise ValueError("allocation decision differs from its request payload")
        unsigned_decision = dict(decision)
        decision_sha256 = unsigned_decision.pop("receipt_sha256", None)
        if decision_sha256 != _hash(
            _ACTION_PORTFOLIO_DECISION_DOMAIN,
            unsigned_decision,
        ):
            raise ValueError("allocation decision receipt self-hash failed")
        if (
            decision.get("finite_contract_identity_sha256")
            != arms.preparation.contract.identity_sha256
            or decision.get("forecast_receipt_sha256")
            != request.get("forecast_receipt_sha256")
            or decision.get("eligible_options_sha256")
            != expected_eligible_sha256
            or decision.get("utility_policy")
            != arms.preparation.utility.to_record()
            or type(decision.get("members")) is not list
            or len(decision["members"]) != G2_PORTFOLIO_SIZE
        ):
            raise ValueError("allocation decision has the wrong contract/cardinality")
        members = decision["members"]
        assert type(members) is list
        option_ids = tuple(
            member.get("option_id") if type(member) is dict else None
            for member in members
        )
        if any(type(option_id) is not str for option_id in option_ids):
            raise ValueError("allocation decision contains a malformed option ID")
        if len(set(option_ids)) != G2_PORTFOLIO_SIZE or [
            member.get("rank") if type(member) is dict else None
            for member in members
        ] != [1, 2, 3]:
            raise ValueError("allocation decision members are not unique/canonical")
        if not set(option_ids).issubset(arms.preparation.eligible_g2_option_ids):
            raise ValueError("committed allocation contains a G1 or foreign action")
        for member in members:
            assert type(member) is dict
            option = arms.preparation.contract.resolve(str(member["option_id"]))
            if (
                member.get("option_identity_sha256") != option.identity_sha256
                or member.get("child_configuration_sha256")
                != option.child_configuration_sha256
                or member.get("family") != option.family
            ):
                raise ValueError("committed member differs from its finite action")
        selected.update(option_ids)
        assert type(decision_sha256) is str
        pairs.append((arm, request_sha256, decision_sha256))
    return _issue_allocation_commitment(
        finite_contract_identity_sha256=arms.preparation.contract.identity_sha256,
        phase_commit_receipt_sha256=(
            durable_allocation_phase_commit.receipt.receipt_sha256
        ),
        arm_allocation_pairs=tuple(pairs),
        selected_option_ids=tuple(sorted(selected)),
    )


def bind_airfoil_mpn_frame_allocation_commitment(
    arms: AirfoilTwoStageForecastArms,
    executions: tuple[FrameActionAllocationTreatmentExecution, ...],
    durable_allocation_phase_commit: TwoStageActionPhaseCommit,
) -> AirfoilMpnAllocationCommitment:
    """Open Airfoil G2 only from exact passing M/P/N subset allocations.

    The generic validator proves equality between the durable payload and the
    supplied complete request/decision/audit executions.  This benchmark-side
    adapter then adds only Airfoil facts: canonical M/P/N treatment order, the
    frozen arm requests, one common authenticated partition subset containing
    exactly G2, and finite-contract identities for every selected member.
    """

    if type(arms) is not AirfoilTwoStageForecastArms:
        raise TypeError("arms must be exact AirfoilTwoStageForecastArms")
    arms.__post_init__()
    if type(executions) is not tuple or any(
        type(value) is not FrameActionAllocationTreatmentExecution
        for value in executions
    ):
        raise TypeError("executions must be an exact frame execution tuple")
    if len(executions) != 3:
        raise ValueError("Airfoil frame allocation requires exactly M/P/N")
    if tuple(
        value.treatment_occurrence.treatment_id.value for value in executions
    ) != ("m", "p", "n"):
        raise ValueError("Airfoil frame allocation requires canonical M/P/N order")

    # This rejects non-ALLOCATE commits, failed audits, reordered treatments,
    # and any request/decision/audit record that differs from durable payload.
    validate_frame_action_allocation_phase_commit(
        executions,
        durable_allocation_phase_commit,
    )

    contract = arms.preparation.contract
    global_g2_set = set(arms.preparation.eligible_g2_option_ids)
    common_partition_identity: tuple[object, ...] | None = None
    pairs: list[tuple[str, str, str]] = []
    selected: set[str] = set()
    for arm, execution in zip(("m", "p", "n"), executions, strict=True):
        request = execution.request
        frame = request.frame
        expected_forecast_request = arms.request(arm)
        if (
            frame.frame_kind
            is not ActionForecastAllocationFrameKind.PARTITION_BLOCK_SUBSET
            or frame.block_request is None
            or frame.resolved_block is None
            or frame.subset_policy is None
        ):
            raise ValueError(
                "Airfoil frame allocation requires an authenticated block subset"
            )
        if (
            frame.request.request_sha256
            != expected_forecast_request.request_sha256
            or frame.block_request.request.request_sha256
            != expected_forecast_request.request_sha256
        ):
            raise ValueError("frame allocation names another Airfoil arm request")
        block = frame.block_request.block
        expected_global_rows = tuple(
            index
            for index in range(block.global_row_start, block.global_row_stop)
            if contract.options[index].option_id in global_g2_set
        )
        expected_eligible = tuple(
            sorted(contract.options[index].option_id for index in expected_global_rows)
        )
        if (
            request.eligible_option_ids != expected_eligible
            or frame.global_row_indices != expected_global_rows
            or tuple(value.option_id for value in frame.forecasts)
            != tuple(contract.options[index].option_id for index in expected_global_rows)
            or request.portfolio_size != G2_PORTFOLIO_SIZE
            or request.utility.to_record() != arms.preparation.utility.to_record()
        ):
            raise ValueError("frame allocation differs from the exact common G2 set")
        partition_identity = (
            frame.block_request.layout.layout_sha256,
            block.block_spec_sha256,
            frame.global_row_indices,
            frame.subset_policy.binding_sha256,
        )
        if common_partition_identity is None:
            common_partition_identity = partition_identity
        elif partition_identity != common_partition_identity:
            raise ValueError("M/P/N frames do not share one partition subset")
        if not execution.result.audit.passes:
            raise ValueError("Airfoil allocation surface audit did not pass")

        decision = execution.result.decision
        if len(decision.members) != G2_PORTFOLIO_SIZE:
            raise ValueError("Airfoil allocation decision has the wrong cardinality")
        for member in decision.members:
            if member.option_id not in expected_eligible:
                raise ValueError("Airfoil frame allocation selected G1 or foreign action")
            option = contract.resolve(member.option_id)
            if (
                member.option_identity_sha256 != option.identity_sha256
                or member.child_configuration_sha256
                != option.child_configuration_sha256
                or member.family != option.family
            ):
                raise ValueError("committed member differs from its finite action")
            selected.add(member.option_id)
        pairs.append(
            (
                arm,
                request.request_sha256,
                decision.receipt_sha256,
            )
        )
    return _issue_allocation_commitment(
        finite_contract_identity_sha256=contract.identity_sha256,
        phase_commit_receipt_sha256=(
            durable_allocation_phase_commit.receipt.receipt_sha256
        ),
        arm_allocation_pairs=tuple(pairs),
        selected_option_ids=tuple(sorted(selected)),
    )


def bind_airfoil_mpn_paired_allocation_commitment(
    arms: AirfoilTwoStageForecastArms,
    methods: tuple[AllocationComparisonMethodWave, ...],
    paired_commitment: PairedAllocationComparisonCommitment,
    *,
    expected_schedule_binding_sha256: str,
) -> AirfoilPairedAllocationCommitment:
    """Project exact v2/v3 M/P/N commits into union-only Airfoil authority."""

    if type(arms) is not AirfoilTwoStageForecastArms:
        raise TypeError("arms must be exact AirfoilTwoStageForecastArms")
    arms.__post_init__()
    require_sha256(
        expected_schedule_binding_sha256,
        "expected_schedule_binding_sha256",
    )
    verified = validate_paired_allocation_comparison_commitment(
        methods,
        paired_commitment,
    )
    if (
        verified.methods[0].schedule_binding_sha256
        != expected_schedule_binding_sha256
    ):
        raise ValueError("paired allocation names another Airfoil schedule")
    if verified.logical_slot_count != 18:
        raise ValueError("Airfoil paired M/P/N comparison requires 18 logical slots")

    contract = arms.preparation.contract
    global_g2 = set(arms.preparation.eligible_g2_option_ids)
    common_partition_identity: tuple[object, ...] | None = None
    selected: set[str] = set()
    for method in methods:
        if len(method.executions) != 3 or tuple(
            value.treatment_occurrence.treatment_id.value
            for value in method.executions
        ) != ("m", "p", "n"):
            raise ValueError("Airfoil paired allocation requires canonical M/P/N")
        for arm, execution in zip(
            ("m", "p", "n"),
            method.executions,
            strict=True,
        ):
            if type(execution) is FrameActionAllocationTreatmentExecution:
                request = execution.request
            elif type(execution) is (
                OperationalFrameActionAllocationTreatmentExecution
            ):
                request = execution.request.allocation
            else:
                raise TypeError("Airfoil comparison contains a foreign execution")
            frame = request.frame
            expected_forecast_request = arms.request(arm)
            if (
                frame.frame_kind
                is not ActionForecastAllocationFrameKind.PARTITION_BLOCK_SUBSET
                or frame.block_request is None
                or frame.resolved_block is None
                or frame.subset_policy is None
            ):
                raise ValueError(
                    "Airfoil paired allocation requires a block-subset frame"
                )
            if (
                frame.request.request_sha256
                != expected_forecast_request.request_sha256
                or frame.block_request.request.request_sha256
                != expected_forecast_request.request_sha256
            ):
                raise ValueError("paired allocation names another Airfoil arm")
            block = frame.block_request.block
            expected_rows = tuple(
                index
                for index in range(block.global_row_start, block.global_row_stop)
                if contract.options[index].option_id in global_g2
            )
            expected_eligible = tuple(
                sorted(contract.options[index].option_id for index in expected_rows)
            )
            if (
                frame.global_row_indices != expected_rows
                or request.eligible_option_ids != expected_eligible
                or request.portfolio_size != G2_PORTFOLIO_SIZE
                or request.utility.to_record()
                != arms.preparation.utility.to_record()
                or request.utility.utility is not arms.preparation.utility.utility
            ):
                raise ValueError(
                    "paired allocation differs from the Airfoil frame/utility/budget"
                )
            partition_identity = (
                frame.block_request.layout.layout_sha256,
                block.block_spec_sha256,
                frame.global_row_indices,
                frame.subset_policy.binding_sha256,
            )
            if common_partition_identity is None:
                common_partition_identity = partition_identity
            elif partition_identity != common_partition_identity:
                raise ValueError("paired methods do not share one Airfoil frame")
            if len(execution.result.decision.members) != G2_PORTFOLIO_SIZE:
                raise ValueError("paired Airfoil allocation has the wrong budget")
            for member in execution.result.decision.members:
                if member.option_id not in expected_eligible:
                    raise ValueError("paired allocation selected G1 or foreign action")
                option = contract.resolve(member.option_id)
                if (
                    member.option_identity_sha256 != option.identity_sha256
                    or member.child_configuration_sha256
                    != option.child_configuration_sha256
                    or member.family != option.family
                ):
                    raise ValueError("paired selected member changed finite identity")
                selected.add(member.option_id)
    if tuple(sorted(selected)) != verified.selected_option_ids:
        raise ValueError("Airfoil selected union differs from the paired commitment")
    if len(selected) > 18:
        raise ValueError("Airfoil paired selected union exceeds 18 logical slots")
    return _issue_airfoil_paired_allocation_commitment(
        finite_contract_identity_sha256=contract.identity_sha256,
        paired_comparison_commitment_sha256=verified.commitment_sha256,
        schedule_binding_sha256=expected_schedule_binding_sha256,
        method_commit_pairs=tuple(
            (
                value.comparison_method_id,
                value.allocation_phase_commit_receipt_sha256,
            )
            for value in verified.methods
        ),
        logical_slot_count=verified.logical_slot_count,
        selected_option_ids=verified.selected_option_ids,
    )


def build_airfoil_v7_forecast_arms(
    preparation: PreparedAirfoilTwoStageGeneration,
    reflection: ReflectionWorkflowResult,
) -> AirfoilTwoStageForecastArms:
    """Validate one live reflection result and build all three forecast calls."""

    preparation.__post_init__()
    if type(reflection) is not ReflectionWorkflowResult:
        raise TypeError("reflection must be an exact ReflectionWorkflowResult")
    ReflectionWorkflowResult.__post_init__(reflection)
    by_contrast = {value.contrast_id: value for value in preparation.observations}
    returned = tuple(shard.contrast_id for shard in reflection.shards)
    if set(returned) != set(by_contrast) or len(returned) != len(by_contrast):
        raise ValueError("reflection result differs from the exact G1 contrast set")
    contract = preparation.reflection_request.insight_contract
    assert contract is not None
    available = tuple(sorted(by_contrast))
    staged: list[ReflectedInsightBatchItem] = []
    for shard in reflection.shards:
        observation = by_contrast[shard.contrast_id]
        empirical_snapshot = observation.empirical_evidence_snapshot()
        draft = shard.draft
        validate_reflection_insight_draft(draft, contract)
        if draft.evidence_contrast_ids != (observation.contrast_id,):
            raise ValueError("reflected card must cite exactly its source contrast")
        if draft.recommended_option_ids != (observation.option_id,):
            raise ValueError("reflected card recommends another exact action")
        if draft.recommended_option_families != (observation.family,):
            raise ValueError("reflected card recommends another action family")
        source_candidate_ids = tuple(
            sorted(
                {
                    observation.parent_candidate_id,
                    observation.child_candidate_id,
                }
            )
        )
        if len(source_candidate_ids) != 2:
            raise ValueError("one G1 contrast requires unique parent and child IDs")
        staged.append(
            ReflectedInsightBatchItem(
                draft=draft,
                evidence_lineage=InsightEvidenceLineage(
                    reflection_call_id=shard.call_id,
                    source_operator_invocation_ids=(
                        observation.operator_invocation_id,
                    ),
                    source_candidate_ids=source_candidate_ids,
                    available_contrast_ids=available,
                    cited_contrast_ids=(observation.contrast_id,),
                    finite_action_bindings=(observation.action_binding,),
                    empirical_evidence=(empirical_snapshot,),
                ),
            )
        )
    memory = InsightMemoryBank(
        id_factory=DeterministicIdFactory("airfoil_twostage_cards")
    )
    entries = memory.add_reflection_batch(
        tuple(staged),
        initial_score=0.0,
        applicable_operator_kinds=("mutation",),
    )
    entry_by_contrast = {
        entry.evidence_lineage.cited_contrast_ids[0]: entry
        for entry in entries
        if entry.evidence_lineage is not None
    }
    source_cards: list[PortfolioCard] = []
    ordered_entries: list[InsightMemoryEntry] = []
    for observation in preparation.observations:
        entry = entry_by_contrast[observation.contrast_id]
        lineage = entry.evidence_lineage
        assert lineage is not None
        evidence_sha256 = _hash(
            _CARD_EVIDENCE_DOMAIN,
            {
                "observation": observation.to_record(),
                "draft_content_sha256": entry.draft.content_sha256,
                "lineage_identity_sha256": lineage.identity_sha256,
            },
        )
        source_cards.append(
            portfolio_card_from_insight_entry(
                entry,
                card_key=f"card.g1.{observation.diagnostic_rank:04d}",
                prompt_payload=_epistemic_card_payload(
                    entry.draft,
                    preparation.contract,
                    lineage.empirical_evidence,
                ),
                evidence_sha256=evidence_sha256,
                source_receipt_sha256=(
                    observation.evaluation.terminal_record_sha256
                ),
                score_components=(),
                assigned_score=None,
            )
        )
        ordered_entries.append(entry)
    cards = tuple(sorted(source_cards, key=lambda value: value.card_key))
    entries_tuple = tuple(ordered_entries)
    registry = admit_portfolio_card_sources(entries_tuple, cards)
    memory_receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.MEMORY,
        cards=cards,
        finite_variation_contract=preparation.contract,
        source_registry=registry,
        policy_id=EXPERIMENTAL_VIEW_POLICY_ID,
        policy_version=EXPERIMENTAL_VIEW_POLICY_VERSION,
        policy_definition_sha256=EXPERIMENTAL_VIEW_DEFINITION_SHA256,
    )

    transforms = tuple(
        sorted(
            (
                PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
                PortfolioCardViewTransform.PROMPT_PERMUTATION,
                PortfolioCardViewTransform.SCORE_PERMUTATION,
            ),
            key=lambda value: value.value,
        )
    )
    source_order = tuple(
        sorted(
            cards,
            key=lambda card: card.source_binding.binding_sha256
            if card.source_binding is not None
            else "",
        )
    )
    rotated_donors = (*source_order[1:], source_order[0])
    donor_by_source_binding = {
        source.source_binding.binding_sha256: donor
        for source, donor in zip(source_order, rotated_donors, strict=True)
        if source.source_binding is not None
    }
    donors = tuple(
        donor_by_source_binding[source.source_binding.binding_sha256]
        for source in cards
        if source.source_binding is not None
    )
    if len(donors) != len(cards):
        raise ValueError("P donor rotation requires every source binding")
    placebo_cards = tuple(
        derive_portfolio_card_view(
            source,
            prompt_payload=donor.prompt_payload,
            evidence_sha256=donor.evidence_sha256,
            score_components=donor.score_components,
            assigned_score=donor.assigned_score,
            transforms=transforms,
            policy_id=EXPERIMENTAL_VIEW_POLICY_ID,
            policy_version=EXPERIMENTAL_VIEW_POLICY_VERSION,
            policy_definition_sha256=EXPERIMENTAL_VIEW_DEFINITION_SHA256,
            prompt_source_card=donor,
            evidence_source_card=donor,
            score_source_card=donor,
        )
        for source, donor in zip(cards, donors, strict=True)
    )
    placebo_receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
        cards=placebo_cards,
        finite_variation_contract=preparation.contract,
        source_registry=registry,
        policy_id=EXPERIMENTAL_VIEW_POLICY_ID,
        policy_version=EXPERIMENTAL_VIEW_POLICY_VERSION,
        policy_definition_sha256=EXPERIMENTAL_VIEW_DEFINITION_SHA256,
    )

    arms = AirfoilTwoStageForecastArms(
        preparation=preparation,
        entries=entries_tuple,
        source_cards=cards,
        placebo_cards=placebo_cards,
        source_registry=registry,
        memory_receipt=memory_receipt,
        placebo_receipt=placebo_receipt,
        memory_request=_forecast_request(
            preparation=preparation,
            call_id=LLMCallId("call_airfoil_twostage_forecast_001"),
            cards=cards,
            registry=registry,
            evidence_mode=ActionForecastEvidenceMode.GROUNDED,
            receipt=memory_receipt,
        ),
        placebo_request=_forecast_request(
            preparation=preparation,
            call_id=LLMCallId("call_airfoil_twostage_forecast_002"),
            cards=placebo_cards,
            registry=registry,
            evidence_mode=ActionForecastEvidenceMode.GROUNDED,
            receipt=placebo_receipt,
        ),
        catalog_only_request=_forecast_request(
            preparation=preparation,
            call_id=LLMCallId("call_airfoil_twostage_forecast_003"),
            cards=(),
            registry=None,
            evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
            receipt=None,
        ),
    )
    arms.__post_init__()
    return arms


def live_wiring_record() -> dict[str, object]:
    """Exact remaining live work after provider-free preparation succeeds."""

    return {
        "provider_calls": {
            "count": 4,
            "stages": [
                {
                    "stage": "strict_batched_reflection",
                    "logical_calls": 1,
                    "concurrency_group": 1,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                {
                    "stage": "all_option_forecast_m_p_n",
                    "logical_calls": 3,
                    "concurrency_group": 2,
                    "run_concurrently": True,
                    "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
                },
            ],
        },
        "after_provider": [
            "resolve each complete typed forecast batch against its request",
            "allocate three unseen G2 actions independently for M, P, and N",
            "evaluate each arm decision through the injected development evaluator",
            "seal traces and compare rank-free online decisions post hoc to oracle ranks",
        ],
        "allocator": {
            "risk_aversion": ALLOCATOR_RISK_AVERSION,
            "diversity_weight": ALLOCATOR_DIVERSITY_WEIGHT,
            "benchmark_utility_range": [0.0, 1.0],
        },
        "provider_free_preparation_calls": 0,
        "credentials_read_by_this_module": False,
    }


__all__ = [
    "ALLOCATOR_DIVERSITY_WEIGHT",
    "ALLOCATOR_RISK_AVERSION",
    "AIRFOIL_V7_FORECAST_UTILITY",
    "AirfoilDevelopmentEvaluation",
    "AirfoilG1ActionObservation",
    "AirfoilObservedMetric",
    "AirfoilPairedAllocationCommitment",
    "AirfoilTwoStageForecastArms",
    "AirfoilV7ForecastPortfolioUtility",
    "AirfoilV7PostDecisionEvaluationCapability",
    "AirfoilV7PostcommitRankReferenceCapability",
    "AirfoilV7PairedPostDecisionEvaluationCapability",
    "AirfoilV7SealedOracleDevelopmentEvaluator",
    "AirfoilMpnAllocationCommitment",
    "EVALUATOR_DEFINITION_SHA256",
    "EVALUATOR_POLICY_ID",
    "EXPERIMENTAL_VIEW_DEFINITION_SHA256",
    "G1_SAMPLE_DESIGN_KEY",
    "G1_SAMPLE_SEED",
    "G1_SAMPLE_SIZE",
    "G2_PORTFOLIO_SIZE",
    "MAX_OUTPUT_TOKENS",
    "OBJECTIVE_METRIC_ID",
    "PREPARATION_DEFINITION_SHA256",
    "PreparedAirfoilTwoStageGeneration",
    "REQUIRED_METRIC_IDS",
    "UTILITY_DEFINITION_SHA256",
    "VIOLATION_METRIC_ID",
    "build_airfoil_v7_forecast_arms",
    "bind_airfoil_mpn_allocation_commitment",
    "bind_airfoil_mpn_frame_allocation_commitment",
    "bind_airfoil_mpn_paired_allocation_commitment",
    "live_wiring_record",
    "prepare_airfoil_v7_two_stage_generation",
    "verify_airfoil_v7_predecision_oracle",
]
