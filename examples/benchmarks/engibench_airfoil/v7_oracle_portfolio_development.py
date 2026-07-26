"""Provider-ready, oracle-backed development harness for Airfoil-v7.

This module is intentionally benchmark-local.  It authenticates the completed
finite-action oracle, constructs the prospectively frozen Stage-A evidence
shards from artifact 115, fans eight reflection requests out through an
injected :class:`AgenticGenerator`, and projects accepted cards into the
framework-neutral portfolio-selection port.  It never loads environment files,
reads credentials, constructs a provider client, or evaluates a candidate.

The oracle is adaptation evidence in this harness.  Scores produced here are
trace-debugging diagnostics and must not be described as held-out efficacy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    InsightDraft,
    MetricEffectDirection,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.id_factory import IdFactory
from agent_evolve.ports.portfolio_selection import (
    CardScoreComponent,
    PortfolioCard,
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    validate_ranked_portfolio_decision,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_contract import DELTA_F, DELTA_V
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    materialize_held_out_parent,
)
from examples.benchmarks.engibench_airfoil.v7_finite_oracle import (
    EXPECTED_OPTION_COUNT,
    EXPECTED_RANS_CALLS,
    OBJECTIVE_NAME,
    ORACLE_FINALIZATION_FRAMING,
    ORACLE_MANIFEST_FRAMING,
    ORACLE_RECORD_FRAMING,
    ORACLE_RESULT_FRAMING,
    PARENT_METRICS,
    VIOLATION_NAME,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7UnionVariationCatalog,
)


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEALED_ORACLE_DIR = (
    _WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "airfoil_v7"
    / "finite_oracles"
    / "ae7_finite_oracle_0715_0143"
)

SHARD_DESIGN_ID = "airfoil_v7_family_contiguous_oracle_shards_v1"
SHARD_ORDERING = (
    "filter union task-keyed display order by family, then four contiguous "
    "equal blocks"
)
EXPECTED_SHARD_MAPPING_SHA256 = (
    "a019c0e9379d249aa2905f562d86b90072abe0ef3d8124b568ed30b014884dea"
)
BASE_DEVELOPMENT_DESIGN_ID = "airfoil_v7_oracle_portfolio_stage_a_v1"
DEVELOPMENT_DESIGN_ID = (
    "airfoil_v7_oracle_portfolio_stage_a_v1r2_provider_grammar"
)
EXECUTION_REVISION_CLASS = "pre_treatment_provider_grammar_repair"
MECHANISM_REVISION_ORDINAL = 0
METRIC_IDS = tuple(sorted((OBJECTIVE_NAME, VIOLATION_NAME)))
PORTFOLIO_SIZE = 3
VIEW_IDS = ("M", "P", "N")
UNIFORM_SEED = "agent-evolve:airfoil-v7-oracle-portfolio-uniform:v1"
STAGE_A_MAX_OUTPUT_TOKENS = 384_000

_SCORE_DEFINITIONS: dict[str, str] = {
    "evidence_count": "Number of distinct sealed origin actions for this card.",
    "global_adaptation_rank_mean": (
        "Arithmetic mean of exact scientific ranks in the 80-action adaptation "
        "oracle; lower is better."
    ),
    "native_parent_relation_mean": (
        "Mean Airfoil archive relation against the nonce-0 parent, encoded "
        "better=1, equivalent=0, worse=-1."
    ),
    f"{VIOLATION_NAME}_delta_mean": (
        "Mean child-minus-parent normalized lift-equality violation over sealed "
        "origin actions."
    ),
    f"{OBJECTIVE_NAME}_delta_mean": (
        "Mean child-minus-parent normalized multipoint drag over sealed origin "
        "actions."
    ),
    "within_shard_rank_mean": (
        "Mean exact lexicographic rank among actions in the card's frozen shard; "
        "lower is better."
    ),
}


class OracleDevelopmentContractError(ValueError):
    """A sealed input, generated card, or development decision is invalid."""


class OracleDevelopmentBatchError(RuntimeError):
    """One or more concurrent logical calls failed after all siblings settled."""

    def __init__(self, failures: tuple[tuple[str, str], ...]) -> None:
        self.failures = failures
        super().__init__("oracle portfolio development batch was incomplete")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, object], framing: bytes) -> str:
    return hashlib.sha256(framing + _canonical_bytes(dict(value))).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleDevelopmentContractError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise OracleDevelopmentContractError(f"{label} root must be an object")
    return value


def _verify_self_hash(
    record: dict[str, object],
    *,
    field: str,
    framing: bytes,
    label: str,
) -> None:
    claimed = record.get(field)
    unsigned = dict(record)
    unsigned.pop(field, None)
    if type(claimed) is not str or claimed != _self_hash(unsigned, framing):
        raise OracleDevelopmentContractError(f"{label} self-hash failed")


def _recursive_content_binding(
    run_dir: Path,
) -> tuple[dict[str, dict[str, object]], str]:
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
            raise OracleDevelopmentContractError(
                "sealed oracle contains a symbolic-link file"
            )
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
class VerifiedSealedOracle:
    """Authenticated immutable oracle plus a live reconstruction of its catalog."""

    run_dir: Path
    manifest: dict[str, object]
    result: dict[str, object]
    finalized: dict[str, object]
    contract: FiniteVariationContract
    rows: tuple[dict[str, object], ...]

    @property
    def run_id(self) -> str:
        return str(self.result["run_id"])

    @property
    def result_sha256(self) -> str:
        return str(self.result["result_sha256"])

    @property
    def recursive_content_sha256(self) -> str:
        return str(self.finalized["recursive_content_sha256"])

    @property
    def rows_by_id(self) -> dict[str, dict[str, object]]:
        return {str(row["option_id"]): row for row in self.rows}

    def seal_record(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.result["manifest_sha256"],
            "result_sha256": self.result_sha256,
            "source_sha256": self.result["source_sha256"],
            "recursive_content_sha256": self.recursive_content_sha256,
            "recursive_file_count": self.finalized["recursive_file_count"],
            "finite_contract_identity_sha256": self.contract.identity_sha256,
        }


def verify_sealed_finite_oracle(
    run_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> VerifiedSealedOracle:
    """Authenticate sealed bytes and reconstruct the exact 80-action contract.

    The original manifest binds the source snapshot used for the completed run.
    This verifier deliberately does not require today's source tree to retain
    that historical hash.  Instead it verifies the recursive seal, every
    record self-hash, and then checks that today's benchmark reconstructs the
    same parent-bound contract and row identities before development proceeds.
    """

    resolved = run_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise OracleDevelopmentContractError("sealed oracle path is not a directory")
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
        or finalized.get("recursive_file_count") != len(files)
        or finalized.get("recursive_content_sha256") != recursive_sha256
        or finalized.get("files") != files
    ):
        raise OracleDevelopmentContractError("recursive oracle seal changed")

    manifest = _load_object(resolved / "oracle_manifest.json", label="manifest")
    _verify_self_hash(
        manifest,
        field="manifest_sha256",
        framing=ORACLE_MANIFEST_FRAMING,
        label="manifest",
    )
    result = _load_object(resolved / "oracle_result.json", label="oracle result")
    _verify_self_hash(
        result,
        field="result_sha256",
        framing=ORACLE_RESULT_FRAMING,
        label="oracle result",
    )
    source = manifest.get("source_snapshot")
    if type(source) is not dict:
        raise OracleDevelopmentContractError("manifest source snapshot is malformed")
    if (
        result.get("manifest_sha256") != manifest.get("manifest_sha256")
        or result.get("source_sha256") != source.get("sha256")
        or result.get("status") != "completed_80_action_oracle"
        or result.get("complete_ranking_available") is not True
        or result.get("provider_calls") != 0
        or result.get("credentials_read") is not False
        or result.get("candidate_attempts") != EXPECTED_OPTION_COUNT
        or result.get("successful_candidates") != EXPECTED_OPTION_COUNT
        or result.get("raw_solver_calls") != EXPECTED_RANS_CALLS
    ):
        raise OracleDevelopmentContractError("oracle completion identity changed")

    parent = materialize_held_out_parent()
    frozen_parent = freeze_json(parent.candidate)
    if type(frozen_parent) is not FrozenJsonObject:
        raise AssertionError("Airfoil parent must be a typed JSON object")
    contract = bind_finite_variation_catalog(
        AirfoilV7UnionVariationCatalog(), frozen_parent
    )
    catalog = _load_object(
        resolved / "catalog_contract.json", label="catalog contract"
    )
    manifest_oracle = manifest.get("oracle")
    if type(manifest_oracle) is not dict or manifest_oracle.get("catalog") != catalog:
        raise OracleDevelopmentContractError("manifest catalog binding changed")
    if catalog.get("contract") != contract.evidence_record():
        raise OracleDevelopmentContractError("live finite contract drifted from oracle")
    evaluation_order = catalog.get("evaluation_order")
    rows_value = result.get("results")
    if (
        type(evaluation_order) is not list
        or type(rows_value) is not list
        or len(evaluation_order) != EXPECTED_OPTION_COUNT
        or len(rows_value) != EXPECTED_OPTION_COUNT
        or len(contract.options) != EXPECTED_OPTION_COUNT
    ):
        raise OracleDevelopmentContractError("oracle action table is incomplete")

    rows: list[dict[str, object]] = []
    seen_contrasts: set[str] = set()
    for ordinal, (option, catalog_row, raw_row) in enumerate(
        zip(contract.options, evaluation_order, rows_value, strict=True),
        start=1,
    ):
        if type(catalog_row) is not dict or type(raw_row) is not dict:
            raise OracleDevelopmentContractError("oracle action row is malformed")
        expected = {
            "ordinal": ordinal,
            "option_id": option.option_id,
            "family": option.family,
            "option_identity_sha256": option.identity_sha256,
            "typed_child_configuration_sha256": option.child_configuration_sha256,
            "raw_candidate_sha256": candidate_sha256(
                thaw_json(option.child_configuration)
            ),
        }
        for key, value in expected.items():
            if catalog_row.get(key) != value or raw_row.get(key) != value:
                raise OracleDevelopmentContractError(
                    f"oracle row {ordinal} differs on {key}"
                )
        contrast = raw_row.get("terminal_record_sha256")
        if type(contrast) is not str or len(contrast) != 64:
            raise OracleDevelopmentContractError("terminal contrast ID is malformed")
        if contrast in seen_contrasts:
            raise OracleDevelopmentContractError("terminal contrast IDs repeat")
        seen_contrasts.add(contrast)
        for section, metric in (
            ("objectives", OBJECTIVE_NAME),
            ("violations", VIOLATION_NAME),
        ):
            values = raw_row.get(section)
            if type(values) is not dict or set(values) != {metric}:
                raise OracleDevelopmentContractError("oracle metrics are malformed")
            value = values[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OracleDevelopmentContractError("oracle metric is non-numeric")
            if not math.isfinite(float(value)):
                raise OracleDevelopmentContractError("oracle metric is non-finite")
        rows.append(dict(raw_row))
    return VerifiedSealedOracle(
        run_dir=resolved,
        manifest=manifest,
        result=result,
        finalized=finalized,
        contract=contract,
        rows=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class OracleEvidenceShard:
    shard_id: str
    family: str
    rows: tuple[dict[str, object], ...]

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(str(row["option_id"]) for row in self.rows)

    @property
    def evidence_contrast_ids(self) -> tuple[str, ...]:
        return tuple(str(row["terminal_record_sha256"]) for row in self.rows)

    def mapping_record(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "family": self.family,
            "option_ids": list(self.option_ids),
            "evidence_contrast_ids": list(self.evidence_contrast_ids),
        }


@dataclass(frozen=True, slots=True)
class OracleShardDesign:
    source_run_id: str
    source_result_sha256: str
    shards: tuple[OracleEvidenceShard, ...]

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "design_id": SHARD_DESIGN_ID,
            "source_run_id": self.source_run_id,
            "source_result_sha256": self.source_result_sha256,
            "ordering": SHARD_ORDERING,
            "shards": [shard.mapping_record() for shard in self.shards],
        }

    @property
    def mapping_sha256(self) -> str:
        return _sha256(self.unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self.unsigned_record(),
            "mapping_sha256": self.mapping_sha256,
        }


def build_oracle_evidence_shards(
    oracle: VerifiedSealedOracle,
) -> OracleShardDesign:
    """Apply the frozen outcome-independent family-contiguous shard rule."""

    shards: list[OracleEvidenceShard] = []
    for family, prefix, expected_size in (
        ("shape_only", "shape", 4),
        ("trim_only", "trim", 16),
    ):
        family_rows = tuple(row for row in oracle.rows if row["family"] == family)
        if len(family_rows) != expected_size * 4:
            raise OracleDevelopmentContractError(
                f"family {family} cannot form four equal frozen shards"
            )
        for index in range(4):
            start = index * expected_size
            shards.append(
                OracleEvidenceShard(
                    shard_id=f"{prefix}_{index}",
                    family=family,
                    rows=family_rows[start : start + expected_size],
                )
            )
    design = OracleShardDesign(
        source_run_id=oracle.run_id,
        source_result_sha256=oracle.result_sha256,
        shards=tuple(shards),
    )
    if design.mapping_sha256 != EXPECTED_SHARD_MAPPING_SHA256:
        raise OracleDevelopmentContractError("frozen shard mapping digest changed")
    return design


def _native_parent_relation(row: Mapping[str, object]) -> str:
    violations = row["violations"]
    objectives = row["objectives"]
    assert isinstance(violations, dict) and isinstance(objectives, dict)
    pair = (float(violations[VIOLATION_NAME]), float(objectives[OBJECTIVE_NAME]))
    parent = (
        float(PARENT_METRICS[VIOLATION_NAME]),
        float(PARENT_METRICS[OBJECTIVE_NAME]),
    )
    if pair < parent:
        return "better"
    if pair > parent:
        return "worse"
    return "equivalent"


def _within_shard_ranks(
    shard: OracleEvidenceShard,
) -> dict[str, int]:
    pairs = {
        str(row["option_id"]): (
            float(cast_dict(row["violations"])[VIOLATION_NAME]),
            float(cast_dict(row["objectives"])[OBJECTIVE_NAME]),
        )
        for row in shard.rows
    }
    return {
        option_id: 1 + sum(other < pair for other in pairs.values())
        for option_id, pair in pairs.items()
    }


def cast_dict(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise OracleDevelopmentContractError("expected an exact JSON object")
    return value


def _reflection_evidence_record(
    shard: OracleEvidenceShard,
    contract: FiniteVariationContract,
) -> dict[str, object]:
    within = _within_shard_ranks(shard)
    rows: list[dict[str, object]] = []
    for row in shard.rows:
        option_id = str(row["option_id"])
        option = contract.resolve(option_id)
        objectives = cast_dict(row["objectives"])
        violations = cast_dict(row["violations"])
        rows.append(
            {
                "evidence_contrast_id": row["terminal_record_sha256"],
                "option": option.prompt_record(),
                "outcome": {
                    OBJECTIVE_NAME: float(objectives[OBJECTIVE_NAME]),
                    VIOLATION_NAME: float(violations[VIOLATION_NAME]),
                    f"delta_{OBJECTIVE_NAME}": float(objectives[OBJECTIVE_NAME])
                    - float(PARENT_METRICS[OBJECTIVE_NAME]),
                    f"delta_{VIOLATION_NAME}": float(violations[VIOLATION_NAME])
                    - float(PARENT_METRICS[VIOLATION_NAME]),
                    "native_parent_relation": _native_parent_relation(row),
                    "contextual_parent_reward": row["contextual_parent_reward"],
                    "within_shard_rank": within[option_id],
                    "global_adaptation_rank": row["rank"],
                },
            }
        )
    return {
        "schema_version": 1,
        "development_claim_boundary": (
            "post_hoc adaptation evidence; not held-out efficacy"
        ),
        "shard_id": shard.shard_id,
        "family": shard.family,
        "parent_metrics": dict(PARENT_METRICS),
        "metric_direction_resolution_thresholds": {
            OBJECTIVE_NAME: DELTA_F,
            VIOLATION_NAME: DELTA_V,
        },
        "optimization_order": [
            f"{VIOLATION_NAME}:ascending",
            f"{OBJECTIVE_NAME}:ascending",
        ],
        "action_evidence": rows,
    }


def _render_reflection_prompt(
    shard: OracleEvidenceShard,
    contract: FiniteVariationContract,
) -> str:
    evidence = _reflection_evidence_record(shard, contract)
    encoded = _canonical_bytes(evidence).decode("ascii")
    return "\n".join(
        (
            "Create exactly one causal action card from the sealed adaptation "
            "evidence below.",
            "The card must cite at least one supplied evidence_contrast_id, "
            "recommend at least one exact supplied option_id, recommend only the "
            f"family {shard.family}, and predict every required metric.",
            "Use increase/decrease for child-minus-parent metric direction; use "
            "unchanged only inside the supplied resolution threshold. Do not invent "
            "actions or claim held-out efficacy.",
            "SEALED SHARD EVIDENCE",
            encoded,
        )
    )


@dataclass(frozen=True, slots=True)
class PreparedReflectionCall:
    shard: OracleEvidenceShard
    request: ReflectionGenerationRequest

    def to_record(self) -> dict[str, object]:
        contract = self.request.insight_contract
        assert contract is not None
        return {
            "shard_id": self.shard.shard_id,
            "call_id": self.request.call_id.value,
            "operation": self.request.operation,
            "prompt": self.request.prompt,
            "prompt_sha256": hashlib.sha256(
                self.request.prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
            "available_contrast_ids": list(
                self.request.available_contrast_ids
            ),
            "insight_contract": contract.to_record(),
            "min_insights": self.request.min_insights,
            "max_insights": self.request.max_insights,
            "max_output_tokens": self.request.max_output_tokens,
            "temperature": self.request.temperature,
        }


def prepare_oracle_reflection_requests(
    oracle: VerifiedSealedOracle,
    design: OracleShardDesign,
    *,
    id_factory: IdFactory,
    max_output_tokens: int = STAGE_A_MAX_OUTPUT_TOKENS,
    temperature: float | None = 0.0,
) -> tuple[PreparedReflectionCall, ...]:
    """Render and contract-bind all eight logical calls before execution."""

    if not isinstance(id_factory, IdFactory):
        raise TypeError("id_factory must implement IdFactory")
    _validate_stage_a_generation_settings(max_output_tokens, temperature)
    prepared: list[PreparedReflectionCall] = []
    for shard in design.shards:
        allowed_ids = tuple(sorted(shard.option_ids))
        insight_contract = ReflectionInsightContract(
            required_metric_ids=METRIC_IDS,
            allowed_option_families=(shard.family,),
            allowed_option_ids=allowed_ids,
        )
        request = ReflectionGenerationRequest(
            call_id=id_factory.new_llm_call_id(),
            operation="oracle_portfolio_reflect",
            prompt=_render_reflection_prompt(shard, oracle.contract),
            min_insights=1,
            max_insights=1,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            available_contrast_ids=tuple(sorted(shard.evidence_contrast_ids)),
            insight_contract=insight_contract,
        )
        prepared.append(PreparedReflectionCall(shard=shard, request=request))
    return tuple(prepared)


@dataclass(frozen=True, slots=True)
class AcceptedOracleCard:
    shard: OracleEvidenceShard
    request: ReflectionGenerationRequest
    draft: InsightDraft
    telemetry: AgenticCallTelemetry

    @property
    def card_key(self) -> str:
        family = "shape" if self.shard.family == "shape_only" else "trim"
        return f"card.{family}.{self.shard.shard_id[-1]}"

    def full_record(self) -> dict[str, object]:
        return {
            "card_key": self.card_key,
            "shard_id": self.shard.shard_id,
            "request_call_id": self.request.call_id.value,
            "request_prompt_sha256": hashlib.sha256(
                self.request.prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
            "content_sha256": self.draft.content_sha256,
            "content": self.draft.content_record(),
            "telemetry": telemetry_record(self.telemetry),
        }


def _validate_card_for_call(
    prepared: PreparedReflectionCall,
    result: object,
) -> AcceptedOracleCard:
    if type(result) is not ReflectionGenerationResult or len(result.insights) != 1:
        raise OracleDevelopmentContractError(
            "a reflection shard must return exactly one card"
        )
    draft = result.insights[0]
    if type(draft) is not InsightDraft:
        raise OracleDevelopmentContractError("reflection returned a non-card value")
    contract = prepared.request.insight_contract
    assert contract is not None
    validate_reflection_insight_draft(draft, contract)
    contrasts = set(draft.evidence_contrast_ids)
    if not contrasts or not contrasts.issubset(
        prepared.request.available_contrast_ids
    ):
        raise OracleDevelopmentContractError(
            "card citations escape its sealed shard"
        )
    if not set(draft.recommended_option_ids).issubset(prepared.shard.option_ids):
        raise OracleDevelopmentContractError(
            "card recommendations escape its sealed shard"
        )
    return AcceptedOracleCard(
        shard=prepared.shard,
        request=prepared.request,
        draft=draft,
        telemetry=result.telemetry,
    )


async def run_oracle_reflections(
    prepared: tuple[PreparedReflectionCall, ...],
    *,
    generator: AgenticGenerator,
) -> tuple[AcceptedOracleCard, ...]:
    """Run all eight reflections concurrently and publish all-or-nothing."""

    if len(prepared) != 8:
        raise OracleDevelopmentContractError("exactly eight reflections are required")
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    raw = await asyncio.gather(
        *(generator.reflect(item.request) for item in prepared),
        return_exceptions=True,
    )
    accepted: list[AcceptedOracleCard] = []
    failures: list[tuple[str, str]] = []
    for item, value in zip(prepared, raw, strict=True):
        if isinstance(value, asyncio.CancelledError):
            raise value
        if isinstance(value, BaseException):
            failures.append((item.shard.shard_id, type(value).__name__))
            continue
        try:
            accepted.append(_validate_card_for_call(item, value))
        except (TypeError, ValueError) as exc:
            failures.append((item.shard.shard_id, type(exc).__name__))
    if failures:
        raise OracleDevelopmentBatchError(tuple(failures))
    return tuple(sorted(accepted, key=lambda card: card.card_key))


def _score_definition_sha256(score_id: str) -> str:
    return _sha256(
        {
            "schema_version": 1,
            "benchmark": "airfoil_v7",
            "score_id": score_id,
            "definition": _SCORE_DEFINITIONS[score_id],
        }
    )


def _origin_rows(
    card: AcceptedOracleCard,
) -> tuple[dict[str, object], ...]:
    cited = set(card.draft.evidence_contrast_ids)
    recommended = set(card.draft.recommended_option_ids)
    selected = tuple(
        row
        for row in card.shard.rows
        if row["terminal_record_sha256"] in cited
        or row["option_id"] in recommended
    )
    if not selected:
        raise OracleDevelopmentContractError("card has no sealed origin action")
    return selected


def _card_evidence_bundle(card: AcceptedOracleCard) -> dict[str, object]:
    origins = _origin_rows(card)
    within = _within_shard_ranks(card.shard)
    evidence_rows: list[dict[str, object]] = []
    for row in origins:
        objectives = cast_dict(row["objectives"])
        violations = cast_dict(row["violations"])
        evidence_rows.append(
            {
                "evidence_contrast_id": row["terminal_record_sha256"],
                "metric_deltas": {
                    OBJECTIVE_NAME: float(objectives[OBJECTIVE_NAME])
                    - float(PARENT_METRICS[OBJECTIVE_NAME]),
                    VIOLATION_NAME: float(violations[VIOLATION_NAME])
                    - float(PARENT_METRICS[VIOLATION_NAME]),
                },
                "native_parent_relation": _native_parent_relation(row),
                "within_shard_rank": within[str(row["option_id"])],
                "global_adaptation_rank": row["rank"],
            }
        )
    return {
        "schema_version": 1,
        "confidence": float(card.draft.confidence),
        "effect_predictions": [
            {
                "metric_id": prediction.metric_id,
                "direction": prediction.direction.value,
            }
            for prediction in card.draft.effect_predictions
        ],
        "origin_evidence": evidence_rows,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise OracleDevelopmentContractError("cannot score empty evidence")
    return float(sum(values) / len(values))


def _score_components(card: AcceptedOracleCard) -> tuple[CardScoreComponent, ...]:
    rows = _origin_rows(card)
    within = _within_shard_ranks(card.shard)
    receipts = tuple(sorted(str(row["terminal_record_sha256"]) for row in rows))
    count = len(rows)
    relation_value = {"better": 1.0, "equivalent": 0.0, "worse": -1.0}
    values = {
        "evidence_count": float(count),
        "global_adaptation_rank_mean": _mean(
            [float(row["rank"]) for row in rows]
        ),
        "native_parent_relation_mean": _mean(
            [relation_value[_native_parent_relation(row)] for row in rows]
        ),
        f"{VIOLATION_NAME}_delta_mean": _mean(
            [
                float(cast_dict(row["violations"])[VIOLATION_NAME])
                - float(PARENT_METRICS[VIOLATION_NAME])
                for row in rows
            ]
        ),
        f"{OBJECTIVE_NAME}_delta_mean": _mean(
            [
                float(cast_dict(row["objectives"])[OBJECTIVE_NAME])
                - float(PARENT_METRICS[OBJECTIVE_NAME])
                for row in rows
            ]
        ),
        "within_shard_rank_mean": _mean(
            [float(within[str(row["option_id"])]) for row in rows]
        ),
    }
    return tuple(
        CardScoreComponent(
            score_id=score_id,
            value=values[score_id],
            definition_sha256=_score_definition_sha256(score_id),
            evidence_count=count,
            receipt_sha256s=receipts,
        )
        for score_id in sorted(values)
    )


@dataclass(frozen=True, slots=True)
class CardProjectionSource:
    accepted: AcceptedOracleCard
    evidence_bundle: dict[str, object]
    evidence_sha256: str
    score_components: tuple[CardScoreComponent, ...]

    @property
    def action_binding(self) -> dict[str, object]:
        return {
            "family": self.accepted.shard.family,
            "recommended_option_ids": list(
                self.accepted.draft.recommended_option_ids
            ),
        }


def build_card_projection_sources(
    accepted: tuple[AcceptedOracleCard, ...],
) -> tuple[CardProjectionSource, ...]:
    if len(accepted) != 8:
        raise OracleDevelopmentContractError("a complete card bank requires 8 cards")
    sources = []
    for card in sorted(accepted, key=lambda item: item.card_key):
        bundle = _card_evidence_bundle(card)
        sources.append(
            CardProjectionSource(
                accepted=card,
                evidence_bundle=bundle,
                evidence_sha256=_sha256(bundle),
                score_components=_score_components(card),
            )
        )
    return tuple(sources)


def _portfolio_card(
    target: CardProjectionSource,
    *,
    evidence_source: CardProjectionSource | None,
    view_id: str,
) -> PortfolioCard:
    if view_id == "N":
        payload_value = {
            "schema_version": 1,
            "action_binding": {
                "family": target.accepted.shard.family,
                "recommended_option_ids": [],
            },
            "selector_evidence_bundle": {
                "metric_ids": list(METRIC_IDS),
                "score_component_ids": sorted(_SCORE_DEFINITIONS),
            },
        }
        evidence_sha256 = _sha256(payload_value["selector_evidence_bundle"])
        score_components: tuple[CardScoreComponent, ...] = ()
    else:
        if evidence_source is None:
            raise AssertionError("evidence-bearing views require a source")
        payload_value = {
            "schema_version": 1,
            "action_binding": target.action_binding,
            "selector_evidence_bundle": evidence_source.evidence_bundle,
        }
        evidence_sha256 = evidence_source.evidence_sha256
        score_components = evidence_source.score_components
    frozen = freeze_json(payload_value)
    if type(frozen) is not FrozenJsonObject:
        raise AssertionError("card view must be a typed JSON object")
    target_card = target.accepted
    insight_id = InsightId(
        "insight_airfoil_oracle_" + target_card.shard.shard_id
    )
    return PortfolioCard(
        card_key=target_card.card_key,
        reference=InsightRef(insight_id=insight_id, version=1),
        content_sha256=target_card.draft.content_sha256,
        evidence_sha256=evidence_sha256,
        prompt_payload=frozen,
        score_components=score_components,
        assigned_score=None,
    )


def build_selector_card_views(
    sources: tuple[CardProjectionSource, ...],
) -> dict[str, tuple[PortfolioCard, ...]]:
    by_family: dict[str, list[CardProjectionSource]] = {
        "shape_only": [],
        "trim_only": [],
    }
    for source in sources:
        by_family[source.accepted.shard.family].append(source)
    rotated: dict[str, CardProjectionSource] = {}
    for family_sources in by_family.values():
        ordered = sorted(family_sources, key=lambda source: source.accepted.card_key)
        if len(ordered) != 4:
            raise OracleDevelopmentContractError(
                "P view requires four cards in each family"
            )
        for index, target in enumerate(ordered):
            rotated[target.accepted.card_key] = ordered[(index + 1) % len(ordered)]
    canonical = tuple(sorted(sources, key=lambda item: item.accepted.card_key))
    return {
        "M": tuple(
            _portfolio_card(source, evidence_source=source, view_id="M")
            for source in canonical
        ),
        "P": tuple(
            _portfolio_card(
                source,
                evidence_source=rotated[source.accepted.card_key],
                view_id="P",
            )
            for source in canonical
        ),
        "N": tuple(
            _portfolio_card(source, evidence_source=None, view_id="N")
            for source in canonical
        ),
    }


def _selector_context(oracle: VerifiedSealedOracle) -> FrozenJsonObject:
    value = freeze_json(
        {
            "schema_version": 1,
            "benchmark_id": "engibench_airfoil_v7",
            "development_stage": "post_hoc_oracle_adaptation",
            "claim_boundary": "not_held_out_efficacy",
            "source_run_id": oracle.run_id,
            "source_result_sha256": oracle.result_sha256,
            "parent_metrics": dict(PARENT_METRICS),
            "metric_direction_resolution_thresholds": {
                OBJECTIVE_NAME: DELTA_F,
                VIOLATION_NAME: DELTA_V,
            },
            "optimization_order": [
                f"{VIOLATION_NAME}:ascending",
                f"{OBJECTIVE_NAME}:ascending",
            ],
        }
    )
    if type(value) is not FrozenJsonObject:
        raise AssertionError("selector context must be an object")
    return value


@dataclass(frozen=True, slots=True)
class PreparedSelectorViews:
    requests: tuple[tuple[str, PortfolioSelectionRequest], ...]

    def by_id(self) -> dict[str, PortfolioSelectionRequest]:
        return dict(self.requests)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "views": [
                {
                    "view_id": view_id,
                    "request": request.to_record(),
                    "context": thaw_json(request.context),
                    "cards": [card.prompt_record() for card in request.cards],
                    "ordered_options": list(
                        request.finite_variation_contract.prompt_records()
                    ),
                }
                for view_id, request in self.requests
            ],
        }


def prepare_selector_views(
    oracle: VerifiedSealedOracle,
    sources: tuple[CardProjectionSource, ...],
    *,
    id_factory: IdFactory,
    max_output_tokens: int = STAGE_A_MAX_OUTPUT_TOKENS,
    temperature: float | None = 0.0,
) -> PreparedSelectorViews:
    _validate_stage_a_generation_settings(max_output_tokens, temperature)
    cards = build_selector_card_views(sources)
    instruction = (
        "Select exactly three distinct sealed actions in best-first order for the "
        "supplied parent. Jointly compare all supplied structured cards and the "
        "complete action catalog. No family quota applies. Cite the cards that "
        "materially support each choice and predict every requested metric."
    )
    requests = []
    for view_id in VIEW_IDS:
        requests.append(
            (
                view_id,
                PortfolioSelectionRequest(
                    call_id=id_factory.new_llm_call_id(),
                    operation="select_portfolio",
                    instruction=instruction,
                    context=_selector_context(oracle),
                    finite_variation_contract=oracle.contract,
                    cards=cards[view_id],
                    portfolio_size=PORTFOLIO_SIZE,
                    required_metric_ids=METRIC_IDS,
                    min_distinct_families=None,
                    require_supporting_cards=True,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
        )
    return PreparedSelectorViews(requests=tuple(requests))


async def run_selector_views(
    prepared: PreparedSelectorViews,
    *,
    selector: PortfolioSelectionPolicy,
) -> dict[str, PortfolioSelectionResult]:
    """Run M/P/N concurrently through one injected, provider-neutral policy."""

    if not isinstance(selector, PortfolioSelectionPolicy):
        raise TypeError("selector must implement PortfolioSelectionPolicy")
    values = await asyncio.gather(
        *(selector.select(request) for _, request in prepared.requests),
        return_exceptions=True,
    )
    failures: list[tuple[str, str]] = []
    results: dict[str, PortfolioSelectionResult] = {}
    for (view_id, request), value in zip(prepared.requests, values, strict=True):
        if isinstance(value, asyncio.CancelledError):
            raise value
        if isinstance(value, BaseException):
            failures.append((view_id, type(value).__name__))
            continue
        if type(value) is not PortfolioSelectionResult:
            failures.append((view_id, "InvalidPortfolioSelectionResult"))
            continue
        try:
            validate_ranked_portfolio_decision(request, value.decision)
        except (TypeError, ValueError) as exc:
            failures.append((view_id, type(exc).__name__))
            continue
        results[view_id] = value
    if failures:
        raise OracleDevelopmentBatchError(tuple(failures))
    return results


def _actual_metric_direction(metric_id: str, row: Mapping[str, object]) -> str:
    if metric_id == OBJECTIVE_NAME:
        value = float(cast_dict(row["objectives"])[OBJECTIVE_NAME])
        parent = float(PARENT_METRICS[OBJECTIVE_NAME])
        threshold = DELTA_F
    elif metric_id == VIOLATION_NAME:
        value = float(cast_dict(row["violations"])[VIOLATION_NAME])
        parent = float(PARENT_METRICS[VIOLATION_NAME])
        threshold = DELTA_V
    else:
        raise OracleDevelopmentContractError("unknown Airfoil metric")
    delta = value - parent
    if delta >= threshold:
        return MetricEffectDirection.INCREASE.value
    if delta <= -threshold:
        return MetricEffectDirection.DECREASE.value
    return MetricEffectDirection.UNCHANGED.value


def score_oracle_portfolio(
    oracle: VerifiedSealedOracle,
    option_ids: tuple[str, ...],
    *,
    predictions: tuple[tuple[tuple[str, str], ...], ...] | None = None,
) -> dict[str, object]:
    """Score one committed portfolio exactly against the sealed adaptation table."""

    if len(option_ids) != PORTFOLIO_SIZE or len(set(option_ids)) != PORTFOLIO_SIZE:
        raise OracleDevelopmentContractError("portfolio must contain 3 distinct IDs")
    rows_by_id = oracle.rows_by_id
    try:
        rows = tuple(rows_by_id[option_id] for option_id in option_ids)
    except KeyError as exc:
        raise OracleDevelopmentContractError("portfolio contains a foreign ID") from exc
    ranks = tuple(int(row["rank"]) for row in rows)
    best_rank = min(ranks)
    mass = cast_dict(
        cast_dict(oracle.result["three_action_portfolios"])["best_rank_mass"]
    )
    strictly_better = sum(
        int(count) for rank, count in mass.items() if int(rank) < best_rank
    )
    denominator = int(
        cast_dict(oracle.result["three_action_portfolios"])["combination_count"]
    )
    direction_total = 0
    direction_matches = 0
    prediction_rows: list[dict[str, object]] = []
    if predictions is not None:
        if len(predictions) != PORTFOLIO_SIZE:
            raise OracleDevelopmentContractError("prediction count differs from portfolio")
        for row, predicted in zip(rows, predictions, strict=True):
            predicted_map = dict(predicted)
            if set(predicted_map) != set(METRIC_IDS):
                raise OracleDevelopmentContractError("prediction metrics are incomplete")
            actual = {
                metric_id: _actual_metric_direction(metric_id, row)
                for metric_id in METRIC_IDS
            }
            matches = {
                metric_id: predicted_map[metric_id] == actual[metric_id]
                for metric_id in METRIC_IDS
            }
            direction_total += len(METRIC_IDS)
            direction_matches += sum(matches.values())
            prediction_rows.append(
                {
                    "option_id": row["option_id"],
                    "predicted": predicted_map,
                    "actual": actual,
                    "matches": matches,
                }
            )
    return {
        "option_ids_best_first": list(option_ids),
        "action_ranks": list(ranks),
        "best_rank": best_rank,
        "mean_action_rank": sum(ranks) / PORTFOLIO_SIZE,
        "top_one_rank": ranks[0],
        "top_one_regret_to_own_best": ranks[0] - best_rank,
        "top_one_is_own_best": ranks[0] == best_rank,
        "contains_top_5_action": any(rank <= 5 for rank in ranks),
        "family_composition": {
            family: sum(row["family"] == family for row in rows)
            for family in ("shape_only", "trim_only")
        },
        "uniform_portfolio_percentile_0_best": strictly_better / denominator,
        "uniform_portfolios_strictly_better": strictly_better,
        "uniform_portfolio_count": denominator,
        "direction_prediction_rows": prediction_rows,
        "direction_accuracy": (
            None if direction_total == 0 else direction_matches / direction_total
        ),
    }


def _uniform_option_ids(oracle: VerifiedSealedOracle) -> tuple[str, ...]:
    keyed = sorted(
        oracle.contract.options,
        key=lambda option: (
            hashlib.sha256(
                b"\x00".join(
                    (
                        UNIFORM_SEED.encode("ascii"),
                        oracle.result_sha256.encode("ascii"),
                        option.option_id.encode("ascii"),
                    )
                )
            ).digest(),
            option.option_id,
        ),
    )
    return tuple(option.option_id for option in keyed[:PORTFOLIO_SIZE])


def _empirical_ceiling_option_ids(
    oracle: VerifiedSealedOracle,
) -> tuple[str, ...]:
    ordered = sorted(
        oracle.rows,
        key=lambda row: (
            float(cast_dict(row["violations"])[VIOLATION_NAME]),
            float(cast_dict(row["objectives"])[OBJECTIVE_NAME]),
            str(row["option_id"]).encode("ascii"),
        ),
    )
    return tuple(str(row["option_id"]) for row in ordered[:PORTFOLIO_SIZE])


def engine_baseline_analysis(
    oracle: VerifiedSealedOracle,
) -> dict[str, object]:
    uniform = _uniform_option_ids(oracle)
    ceiling = _empirical_ceiling_option_ids(oracle)
    return {
        "U": {
            "policy": "sha256_seeded_uniform_without_replacement",
            "seed": UNIFORM_SEED,
            "score": score_oracle_portfolio(oracle, uniform),
        },
        "E": {
            "policy": "adaptation_table_top_three_leaked_empirical_ceiling",
            "efficacy_comparator": False,
            "score": score_oracle_portfolio(oracle, ceiling),
        },
    }


def selector_result_record(
    oracle: VerifiedSealedOracle,
    results: Mapping[str, PortfolioSelectionResult],
) -> dict[str, object]:
    views = []
    for view_id in VIEW_IDS:
        result = results[view_id]
        members = result.decision.members
        predictions = tuple(
            tuple(
                (prediction.metric_id, prediction.direction.value)
                for prediction in member.effect_predictions
            )
            for member in members
        )
        views.append(
            {
                "view_id": view_id,
                "decision": result.decision.to_record(),
                "telemetry": (
                    None
                    if result.telemetry is None
                    else telemetry_record(result.telemetry)
                ),
                "score": score_oracle_portfolio(
                    oracle,
                    tuple(member.option_id for member in members),
                    predictions=predictions,
                ),
            }
        )
    by_view = {str(row["view_id"]): cast_dict(row["score"]) for row in views}
    m = by_view["M"]
    p = by_view["P"]
    n = by_view["N"]
    survival = {
        "m_contains_top_5": bool(m["contains_top_5_action"]),
        "m_mean_rank_at_most_20": float(m["mean_action_rank"]) <= 20.0,
        "m_strictly_beats_p_best_and_mean": (
            int(m["best_rank"]) < int(p["best_rank"])
            and float(m["mean_action_rank"]) < float(p["mean_action_rank"])
        ),
        "m_strictly_beats_n_best_and_mean": (
            int(m["best_rank"]) < int(n["best_rank"])
            and float(m["mean_action_rank"]) < float(n["mean_action_rank"])
        ),
        "m_top_one_is_own_best": bool(m["top_one_is_own_best"]),
        "m_direction_accuracy_at_least_075": (
            m["direction_accuracy"] is not None
            and float(m["direction_accuracy"]) >= 0.75
        ),
    }
    return {
        "schema_version": 1,
        "design_id": DEVELOPMENT_DESIGN_ID,
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "execution_revision_class": EXECUTION_REVISION_CLASS,
        "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
        "source_oracle": oracle.seal_record(),
        "claim_boundary": "post_hoc_development_not_held_out_efficacy",
        "views": views,
        "engine_baselines": engine_baseline_analysis(oracle),
        "survival_gates": survival,
        "survives_stage_a_v1": all(survival.values()),
    }


def telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    value.__post_init__()
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": None if value.cost_usd is None else str(value.cost_usd),
        "latency_ns": value.latency_ns,
        "attempt_count": value.attempt_count,
    }


def development_plan_record(
    oracle: VerifiedSealedOracle,
    design: OracleShardDesign,
    reflection_calls: tuple[PreparedReflectionCall, ...],
) -> dict[str, object]:
    """Return the complete no-provider commitment record for Stage A."""

    return {
        "schema_version": 1,
        "design_id": DEVELOPMENT_DESIGN_ID,
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "execution_revision_class": EXECUTION_REVISION_CLASS,
        "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
        "status": "provider_ready_not_launched",
        "source_oracle": oracle.seal_record(),
        "shard_design": design.to_record(),
        "selector_view_design": {
            "M": "correct structured action bindings and evidence bundles",
            "P": (
                "action bindings fixed; complete evidence/effect/score bundle "
                "rotated to the next card within family"
            ),
            "P_precommitted_bundle_sources": {
                **{
                    f"card.shape.{index}": f"card.shape.{(index + 1) % 4}"
                    for index in range(4)
                },
                **{
                    f"card.trim.{index}": f"card.trim.{(index + 1) % 4}"
                    for index in range(4)
                },
            },
            "N": "typed evidence-redacted placeholders and complete catalog",
            "free_card_prose_selector_visible": False,
            "source_action_ids_inside_evidence_bundle": False,
        },
        "reflection_calls": [call.to_record() for call in reflection_calls],
        "logical_reflection_calls": len(reflection_calls),
        "planned_selector_calls": len(VIEW_IDS),
        "new_candidate_evaluations": 0,
        "provider_calls_observed": 0,
        "credentials_read": False,
        "engine_baselines": engine_baseline_analysis(oracle),
    }


def reflection_result_record(
    oracle: VerifiedSealedOracle,
    accepted: tuple[AcceptedOracleCard, ...],
    sources: tuple[CardProjectionSource, ...],
    selector_views: PreparedSelectorViews,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "design_id": DEVELOPMENT_DESIGN_ID,
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "execution_revision_class": EXECUTION_REVISION_CLASS,
        "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
        "source_oracle": oracle.seal_record(),
        "cards": [card.full_record() for card in accepted],
        "card_projection_sources": [
            {
                "card_key": source.accepted.card_key,
                "action_binding": source.action_binding,
                "evidence_sha256": source.evidence_sha256,
                "evidence_bundle": source.evidence_bundle,
                "score_components": [
                    component.to_record() for component in source.score_components
                ],
            }
            for source in sources
        ],
        "selector_views": selector_views.to_record(),
    }


def write_durable_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish canonical JSON and fsync the file and parent directory."""

    if type(value) is not dict:
        value = dict(value)
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(value) + b"\n"
    if target.exists():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == content:
            return
        raise FileExistsError(f"durable record already exists with different bytes: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@runtime_checkable
class DevelopmentRecordSink(Protocol):
    def __call__(self, name: str, record: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class DirectoryDevelopmentRecordSink:
    root: Path

    def __call__(self, name: str, record: Mapping[str, object]) -> None:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz_0123456789" for character in name):
            raise ValueError("record name must be a lowercase token")
        write_durable_json(self.root / f"{name}.json", record)


def _validate_stage_a_generation_settings(
    max_output_tokens: int,
    temperature: float | None,
) -> None:
    if (
        type(max_output_tokens) is not int
        or not 1 <= max_output_tokens <= STAGE_A_MAX_OUTPUT_TOKENS
    ):
        raise ValueError(
            "Stage-A max_output_tokens must lie in "
            f"[1,{STAGE_A_MAX_OUTPUT_TOKENS}]"
        )
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0 <= float(temperature) <= 2
    ):
        raise ValueError("Stage-A temperature must be finite in [0,2] or None")


def prepare_provider_ready_stage_a(
    *,
    run_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
    id_factory: IdFactory | None = None,
    sink: DevelopmentRecordSink | None = None,
) -> tuple[
    VerifiedSealedOracle,
    OracleShardDesign,
    tuple[PreparedReflectionCall, ...],
]:
    """Authenticate and durably freeze Stage A without making a model call."""

    oracle = verify_sealed_finite_oracle(run_dir)
    design = build_oracle_evidence_shards(oracle)
    ids = id_factory or DeterministicIdFactory("airfoil_oracle_stage_a")
    calls = prepare_oracle_reflection_requests(oracle, design, id_factory=ids)
    if sink is not None:
        sink("development_plan", development_plan_record(oracle, design, calls))
    return oracle, design, calls


async def execute_provider_ready_stage_a(
    *,
    generator: AgenticGenerator,
    selector: PortfolioSelectionPolicy,
    run_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
    id_factory: IdFactory | None = None,
    sink: DevelopmentRecordSink | None = None,
) -> dict[str, object]:
    """Execute the complete 8-reflection/3-selector Stage-A batch.

    Both agentic dependencies are injected.  A caller may compose them with the
    queued Pydantic-AI adapters, but this benchmark module has no provider or
    credential capability of its own.  All eight reflection calls settle
    before any card bank is published; M/P/N likewise publish all-or-nothing.
    """

    ids = id_factory or DeterministicIdFactory("airfoil_oracle_stage_a")
    oracle, design, calls = prepare_provider_ready_stage_a(
        run_dir=run_dir,
        id_factory=ids,
        sink=sink,
    )
    accepted = await run_oracle_reflections(calls, generator=generator)
    sources = build_card_projection_sources(accepted)
    selector_views = prepare_selector_views(
        oracle,
        sources,
        id_factory=ids,
    )
    reflection_record = reflection_result_record(
        oracle,
        accepted,
        sources,
        selector_views,
    )
    if sink is not None:
        sink("reflection_results", reflection_record)
    selections = await run_selector_views(selector_views, selector=selector)
    result = selector_result_record(oracle, selections)
    if sink is not None:
        sink("selector_results", result)
    return result


__all__ = [
    "AcceptedOracleCard",
    "BASE_DEVELOPMENT_DESIGN_ID",
    "CardProjectionSource",
    "DEFAULT_SEALED_ORACLE_DIR",
    "DEVELOPMENT_DESIGN_ID",
    "DirectoryDevelopmentRecordSink",
    "EXECUTION_REVISION_CLASS",
    "EXPECTED_SHARD_MAPPING_SHA256",
    "OracleDevelopmentBatchError",
    "OracleDevelopmentContractError",
    "OracleEvidenceShard",
    "OracleShardDesign",
    "MECHANISM_REVISION_ORDINAL",
    "PreparedReflectionCall",
    "PreparedSelectorViews",
    "SHARD_DESIGN_ID",
    "STAGE_A_MAX_OUTPUT_TOKENS",
    "VerifiedSealedOracle",
    "build_card_projection_sources",
    "build_oracle_evidence_shards",
    "build_selector_card_views",
    "development_plan_record",
    "engine_baseline_analysis",
    "execute_provider_ready_stage_a",
    "prepare_oracle_reflection_requests",
    "prepare_provider_ready_stage_a",
    "prepare_selector_views",
    "reflection_result_record",
    "run_oracle_reflections",
    "run_selector_views",
    "score_oracle_portfolio",
    "selector_result_record",
    "verify_sealed_finite_oracle",
    "write_durable_json",
]
