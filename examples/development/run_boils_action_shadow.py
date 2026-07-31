#!/usr/bin/env python3
"""Run the frozen 12-call BOiLS oracle-scored action-ranking shadow.

The proposal phase has no oracle parser or evaluator.  Only after every logical
call is terminal, the queue is closed, and a durable phase-close record exists
does the scorer open the already sealed 40-child development oracle.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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
from types import ModuleType
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, model_validator  # noqa: E402

from agent_evolve.domain.ids import LLMCallId  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    create_production_queued_runner,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter  # noqa: E402
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)

from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    ACTION_IDS,
    ActionId,
)
from examples.development.boils_action_shadow_contracts import (  # noqa: E402
    ProposalClosureReceipt,
)
from examples.development import run_agentic_probe as support  # noqa: E402
from examples.development import run_boils_agentic_pilot as v1  # noqa: E402
from examples.development import run_boils_agentic_pilot_v2 as v2  # noqa: E402
from examples.development.corpus_paths import resolve_corpus_path  # noqa: E402


MODEL = "deepseek/deepseek-v4-pro"
RESOLVED_PROVIDER = "Together"
PROVIDER_ORDER = ("together",)
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 1_800
QUEUE_MAX_IN_FLIGHT = 8
QUEUE_MAX_PENDING = 16
QUEUE_MAX_ATTEMPTS = 2
QUEUE_ATTEMPT_TIMEOUT_SECONDS = 60
QUEUE_BASE_BACKOFF_SECONDS = 1
QUEUE_MAX_BACKOFF_SECONDS = 8
JITTER_SEED = 20_260_714
JITTER_DOMAIN = "boils-shadow-jitter-v1"
MAX_SUCCESSFUL_RESPONSE_COST_USD = Decimal("0.08")

CONDITIONS = ("names_only", "action_cards_niches", "evidence_portfolio")
PATHS = (1, 7, 12, 18)
REFERENCE_POINT = (8_028, 71)
PARENT_OBJECTIVES = (7_944, 69)
RANDOM_POLICY_MEDIAN = 232
EXTENDED_ACTIONS = frozenset(("sopb", "blut", "dsdb"))

CORRECTED_TASK_ORDER: tuple[tuple[str, int], ...] = (
    ("names_only", 7),
    ("action_cards_niches", 7),
    ("action_cards_niches", 1),
    ("names_only", 18),
    ("evidence_portfolio", 18),
    ("names_only", 1),
    ("evidence_portfolio", 1),
    ("evidence_portfolio", 12),
    ("evidence_portfolio", 7),
    ("action_cards_niches", 18),
    ("names_only", 12),
    ("action_cards_niches", 12),
)
ROLE_BY_PATH = {
    18: "balanced Pareto contribution",
    12: "area minimization subject to total_levels<=69",
    1: "depth minimization subject to total_lut_count<=7944",
    7: "action-family exploration",
}
V2_ACTION_BY_PATH = {1: "refactor_z", 7: "resub", 12: "rewrite_z", 18: "rewrite"}

ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DATA_ROOT = ARTIFACT_ROOT / "data"
DEVELOPMENT_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "boils_agentic_development"
DEFAULT_LOG_ROOT = DEVELOPMENT_LOG_ROOT
PREREGISTRATION_PATH = ARTIFACT_ROOT / "65_boils_oracle_scored_action_shadow_preregistration.md"
CORRECTION_PATH = ARTIFACT_ROOT / "67_boils_shadow_and_recombination_preregistration_corrections.md"
CATALOG_PATH = DATA_ROOT / "boils_action_shadow_catalog_v1.json"
EVIDENCE_PATH = DATA_ROOT / "boils_preoracle_action_evidence_v1.json"
ORACLE_RUN_DIR = DEVELOPMENT_LOG_ROOT / "boils_local_oracle_v1_20260714"
SCORER_SOURCE_PATH = (
    AGENT_EVOLVE_ROOT / "examples/development/run_boils_action_shadow_scorer.py"
)
CONTRACT_SOURCE_PATH = (
    AGENT_EVOLVE_ROOT / "examples/development/boils_action_shadow_contracts.py"
)
CANONICAL_MODULE_NAME = "examples.development.run_boils_action_shadow"

EXPECTED_INPUT_SHA256: dict[str, tuple[Path, str]] = {
    "preregistration": (
        PREREGISTRATION_PATH,
        "634f3ba77149f1753d60a135898f40547eeb9eb44a8f3efb5947daee46669702",
    ),
    "correction": (
        CORRECTION_PATH,
        "dc14f1fd95c154bc2c729c9df2b930c8bc8961e8e1efe60be57e5b2af4c3075b",
    ),
    "catalog": (
        CATALOG_PATH,
        "661418880c10999a95eac7b38d42f7c0a3f2973d97fa4ee43d701d86302118de",
    ),
    "preoracle_evidence": (
        EVIDENCE_PATH,
        "3c9b4632721d97a2f3c5085dac2742be4d66a96f8229d9a44c2373ac3b3a2038",
    ),
    "oracle_finalized": (
        ORACLE_RUN_DIR / "finalized.json",
        "627db6494ed38133ebb8478b0954216d741b28340342b30c46de0aa331f6be38",
    ),
    "oracle_evaluations": (
        ORACLE_RUN_DIR / "evaluations.jsonl",
        "5ce1e50f0966ac517ac4f5b14feecc24f4024e500111ed50ee9a1bea63fe1e08",
    ),
    "oracle_summary": (
        ORACLE_RUN_DIR / "summary.json",
        "63e144b597f662b606ea4272e9816a3a1ff8e5c7962685d6751e2d9dcc040b0d",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def hash_bind_inputs(
    sources: Mapping[str, tuple[Path, str]] = EXPECTED_INPUT_SHA256,
) -> dict[str, dict[str, object]]:
    """Read every sealed input only as opaque bytes and check its identity."""

    if set(sources) != set(EXPECTED_INPUT_SHA256):
        raise RuntimeError("shadow sealed-input source set changed")
    records: dict[str, dict[str, object]] = {}
    for name in EXPECTED_INPUT_SHA256:
        path, expected = sources[name]
        payload = resolve_corpus_path(path).read_bytes()
        observed = _sha256_bytes(payload)
        if observed != expected:
            raise RuntimeError(f"shadow sealed-input hash changed: {name}")
        records[name] = {"source": str(path), "bytes": len(payload), "sha256": observed}
    return records


def _load_nonoracle_json(path: Path, expected_sha256: str) -> dict[str, object]:
    payload = resolve_corpus_path(path).read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise RuntimeError(f"frozen non-oracle JSON changed: {path.name}")
    parsed = json.loads(payload)
    if type(parsed) is not dict:
        raise RuntimeError(f"frozen non-oracle JSON is not an object: {path.name}")
    return parsed


CATALOG = _load_nonoracle_json(CATALOG_PATH, EXPECTED_INPUT_SHA256["catalog"][1])
PREORACLE_EVIDENCE = _load_nonoracle_json(
    EVIDENCE_PATH, EXPECTED_INPUT_SHA256["preoracle_evidence"][1]
)


def _validate_catalog() -> tuple[dict[str, object], ...]:
    actions = CATALOG.get("actions")
    order = CATALOG.get("canonical_action_order")
    if (
        CATALOG.get("schema_version") != 1
        or type(actions) is not list
        or tuple(order) != ACTION_IDS
        or tuple(row.get("action_id") for row in actions if type(row) is dict)
        != ACTION_IDS
    ):
        raise RuntimeError("frozen action-card catalog changed shape or order")
    result = tuple(copy.deepcopy(row) for row in actions)
    for row in result:
        if (
            set(row) != {"action_id", "commands", "family", "semantics", "extended_action"}
            or type(row["commands"]) is not list
            or type(row["family"]) is not str
            or type(row["semantics"]) is not str
            or type(row["extended_action"]) is not bool
        ):
            raise RuntimeError("frozen action card has an unexpected shape")
    return result


ACTION_CARDS = _validate_catalog()
CARD_BY_ACTION = {str(row["action_id"]): row for row in ACTION_CARDS}
if frozenset(
    action for action, row in CARD_BY_ACTION.items() if row["extended_action"] is True
) != EXTENDED_ACTIONS:  # pragma: no cover - import-time frozen invariant.
    raise RuntimeError("extended-action catalog flags changed")


DirectionLabel = Literal["decrease", "same", "increase"]


class DirectionProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decrease: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    same: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    increase: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "DirectionProbabilities":
        if not math.isclose(
            self.decrease + self.same + self.increase,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("categorical probabilities must sum to one within 1e-6")
        return self


class ObjectiveDirectionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    total_lut_count: DirectionProbabilities
    total_levels: DirectionProbabilities


Path1ActionId = Literal[
    "rewrite_z",
    "refactor",
    "refactor_z",
    "resub",
    "resub_z",
    "balance",
    "fraig",
    "sopb",
    "blut",
    "dsdb",
]
Path7ActionId = Literal[
    "rewrite",
    "rewrite_z",
    "refactor",
    "resub",
    "resub_z",
    "balance",
    "fraig",
    "sopb",
    "blut",
    "dsdb",
]
Path12ActionId = Literal[
    "rewrite",
    "rewrite_z",
    "refactor_z",
    "resub",
    "resub_z",
    "balance",
    "fraig",
    "sopb",
    "blut",
    "dsdb",
]
Path18ActionId = Literal[
    "rewrite",
    "refactor",
    "refactor_z",
    "resub",
    "resub_z",
    "balance",
    "fraig",
    "sopb",
    "blut",
    "dsdb",
]


class Path1Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rewrite_z: ObjectiveDirectionPrediction
    refactor: ObjectiveDirectionPrediction
    refactor_z: ObjectiveDirectionPrediction
    resub: ObjectiveDirectionPrediction
    resub_z: ObjectiveDirectionPrediction
    balance: ObjectiveDirectionPrediction
    fraig: ObjectiveDirectionPrediction
    sopb: ObjectiveDirectionPrediction
    blut: ObjectiveDirectionPrediction
    dsdb: ObjectiveDirectionPrediction


class Path7Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rewrite: ObjectiveDirectionPrediction
    rewrite_z: ObjectiveDirectionPrediction
    refactor: ObjectiveDirectionPrediction
    resub: ObjectiveDirectionPrediction
    resub_z: ObjectiveDirectionPrediction
    balance: ObjectiveDirectionPrediction
    fraig: ObjectiveDirectionPrediction
    sopb: ObjectiveDirectionPrediction
    blut: ObjectiveDirectionPrediction
    dsdb: ObjectiveDirectionPrediction


class Path12Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rewrite: ObjectiveDirectionPrediction
    rewrite_z: ObjectiveDirectionPrediction
    refactor_z: ObjectiveDirectionPrediction
    resub: ObjectiveDirectionPrediction
    resub_z: ObjectiveDirectionPrediction
    balance: ObjectiveDirectionPrediction
    fraig: ObjectiveDirectionPrediction
    sopb: ObjectiveDirectionPrediction
    blut: ObjectiveDirectionPrediction
    dsdb: ObjectiveDirectionPrediction


class Path18Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rewrite: ObjectiveDirectionPrediction
    refactor: ObjectiveDirectionPrediction
    refactor_z: ObjectiveDirectionPrediction
    resub: ObjectiveDirectionPrediction
    resub_z: ObjectiveDirectionPrediction
    balance: ObjectiveDirectionPrediction
    fraig: ObjectiveDirectionPrediction
    sopb: ObjectiveDirectionPrediction
    blut: ObjectiveDirectionPrediction
    dsdb: ObjectiveDirectionPrediction


class ActionRankingResponse(BaseModel):
    """Base response whose subclasses freeze the current action by path."""

    model_config = ConfigDict(extra="forbid", strict=True)
    expected_actions: ClassVar[frozenset[str]] = frozenset()

    ranking: list[ActionId] = Field(min_length=10, max_length=10)
    predictions: BaseModel

    @model_validator(mode="after")
    def _exact_path_permutations(self) -> "ActionRankingResponse":
        ranking = list(self.ranking)
        predicted = set(type(self.predictions).model_fields)
        if (
            len(set(ranking)) != 10
            or set(ranking) != self.expected_actions
            or predicted != self.expected_actions
        ):
            raise ValueError("ranking and predictions must each be the exact ten-action path permutation")
        return self


class Path1RankingResponse(ActionRankingResponse):
    expected_actions: ClassVar[frozenset[str]] = frozenset(
        action for action in ACTION_IDS if action != v2.PARENT_C["sequence"][1]
    )
    ranking: list[Path1ActionId] = Field(
        min_length=10,
        max_length=10,
        json_schema_extra={"uniqueItems": True},
    )
    predictions: Path1Predictions


class Path7RankingResponse(ActionRankingResponse):
    expected_actions: ClassVar[frozenset[str]] = frozenset(
        action for action in ACTION_IDS if action != v2.PARENT_C["sequence"][7]
    )
    ranking: list[Path7ActionId] = Field(
        min_length=10,
        max_length=10,
        json_schema_extra={"uniqueItems": True},
    )
    predictions: Path7Predictions


class Path12RankingResponse(ActionRankingResponse):
    expected_actions: ClassVar[frozenset[str]] = frozenset(
        action for action in ACTION_IDS if action != v2.PARENT_C["sequence"][12]
    )
    ranking: list[Path12ActionId] = Field(
        min_length=10,
        max_length=10,
        json_schema_extra={"uniqueItems": True},
    )
    predictions: Path12Predictions


class Path18RankingResponse(ActionRankingResponse):
    expected_actions: ClassVar[frozenset[str]] = frozenset(
        action for action in ACTION_IDS if action != v2.PARENT_C["sequence"][18]
    )
    ranking: list[Path18ActionId] = Field(
        min_length=10,
        max_length=10,
        json_schema_extra={"uniqueItems": True},
    )
    predictions: Path18Predictions


OUTPUT_TYPE_BY_PATH: dict[int, type[ActionRankingResponse]] = {
    1: Path1RankingResponse,
    7: Path7RankingResponse,
    12: Path12RankingResponse,
    18: Path18RankingResponse,
}


@dataclass(frozen=True, slots=True)
class ShadowTask:
    ordinal: int
    condition: str
    path: int
    schedule_sha256: str
    call_id: str
    prompt: str
    prompt_sha256: str
    schema_sha256: str
    output_type: type[ActionRankingResponse]

    def manifest_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "condition": self.condition,
            "path": self.path,
            "json_path": f"$.sequence[{self.path}]",
            "schedule_sha256": self.schedule_sha256,
            "call_id": self.call_id,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "output_type": self.output_type.__name__,
            "output_schema": self.output_type.model_json_schema(),
            "output_schema_sha256": self.schema_sha256,
        }


def _schedule_digest(condition: str, path: int) -> str:
    return _sha256_bytes(f"20260714:{condition}:{path}".encode("utf-8"))


def _noncurrent_actions(path: int) -> tuple[str, ...]:
    current = str(v2.PARENT_C["sequence"][path])
    return tuple(action for action in ACTION_IDS if action != current)


def build_prompt(condition: str, path: int) -> str:
    if condition not in CONDITIONS or path not in PATHS:
        raise ValueError("unknown frozen shadow condition or path")
    actions = _noncurrent_actions(path)
    payload: dict[str, object] = {
        "task": {
            "domain": "BOiLS Berkeley ABC logic-synthesis sequence co-optimization",
            "circuit_panel": "log2",
            "mapping_and_validity": "LUT-6 mapping followed by mandatory CEC",
            "objectives": [
                {"name": "total_lut_count", "goal": "minimize"},
                {"name": "total_levels", "goal": "minimize"},
            ],
        },
        "parent": {
            "sequence": list(v2.PARENT_C["sequence"]),
            "objectives": {
                "total_lut_count": PARENT_OBJECTIVES[0],
                "total_levels": PARENT_OBJECTIVES[1],
            },
        },
        "editable_path": f"$.sequence[{path}]",
        "current_action": v2.PARENT_C["sequence"][path],
        "rank_exactly_once": list(actions),
    }
    if condition != "names_only":
        payload["assigned_portfolio_role"] = ROLE_BY_PATH[path]
        payload["action_cards"] = [copy.deepcopy(CARD_BY_ACTION[action]) for action in actions]
    if condition == "evidence_portfolio":
        payload["preoracle_machine_evidence"] = copy.deepcopy(PREORACLE_EVIDENCE)
        payload["deterministic_coordinator"] = {
            "uses_all_four_path_rankings": True,
            "minimum_distinct_families": 3,
            "requires_extended_action": True,
            "your_call_does_not_select_or_materialize_a_configuration": True,
        }
    condition_instruction = {
        "names_only": (
            "Use only the action names and flat parent facts supplied. Rank by expected "
            "contribution to the two-objective archive; no prior outcomes or action semantics are supplied."
        ),
        "action_cards_niches": (
            "Use the exact source-derived action cards and assigned role. The cards describe commands, "
            "not observed performance. Rank all legal actions for this path and role."
        ),
        "evidence_portfolio": (
            "Use the source-derived cards, assigned role, and only the explicitly supplied pre-oracle "
            "machine evidence. Your ranking feeds the stated deterministic portfolio coordinator."
        ),
    }[condition]
    return (
        "You are a compact evolutionary action-ranking component. "
        + condition_instruction
        + " Return all ten non-current action IDs in a strict best-to-worst ranking exactly once. "
        "For every action, return separate decrease/same/increase probability distributions versus the "
        "parent for LUT count and levels. Each triplet must sum to one. Return no configuration, rationale, "
        "mechanism prose, confidence scalar, citation, or memory claim.\n\nFROZEN TASK\n"
        + _canonical_json(payload)
    )


def build_tasks() -> tuple[ShadowTask, ...]:
    derived = tuple(
        sorted(
            ((condition, path) for condition in CONDITIONS for path in PATHS),
            key=lambda row: _schedule_digest(*row),
        )
    )
    if derived != CORRECTED_TASK_ORDER:
        raise RuntimeError("corrected shadow task order no longer matches its hash rule")
    tasks: list[ShadowTask] = []
    for ordinal, (condition, path) in enumerate(derived, start=1):
        prompt = build_prompt(condition, path)
        output_type = OUTPUT_TYPE_BY_PATH[path]
        schema = output_type.model_json_schema()
        tasks.append(
            ShadowTask(
                ordinal=ordinal,
                condition=condition,
                path=path,
                schedule_sha256=_schedule_digest(condition, path),
                call_id=f"call_boils_action_shadow_20260714_{ordinal:02d}",
                prompt=prompt,
                prompt_sha256=_sha256_bytes(prompt.encode("utf-8")),
                schema_sha256=_sha256_json(schema),
                output_type=output_type,
            )
        )
    return tuple(tasks)


FROZEN_TASKS = build_tasks()


class Predictor(Protocol):
    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any] | AttemptedStructuredGenerationResponse[Any]: ...


def _provider_envelope(
    value: StructuredGenerationResponse[Any] | AttemptedStructuredGenerationResponse[Any],
) -> tuple[StructuredGenerationResponse[Any], int]:
    if type(value) is AttemptedStructuredGenerationResponse:
        return value.response, value.attempt_count
    if type(value) is StructuredGenerationResponse:
        return value, 1
    raise TypeError("shadow predictor returned an unsupported response envelope")


def _closure_event(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ordered = sorted(records, key=lambda row: int(row["ordinal"]))
    return {
        "schema_version": 1,
        "event_type": "proposal_phase_closed",
        "recorded_at_utc": _utc_now(),
        "terminal_logical_calls": len(ordered),
        "queue_closed": True,
        "oracle_parser_constructed": False,
        "terminal_response_hashes": [_sha256_json(row) for row in ordered],
    }


def _receipt_from_closure_event(event: Mapping[str, object]) -> ProposalClosureReceipt:
    if (
        event.get("event_type") != "proposal_phase_closed"
        or event.get("queue_closed") is not True
        or event.get("oracle_parser_constructed") is not False
        or event.get("terminal_logical_calls") != 12
        or type(event.get("terminal_response_hashes")) is not list
    ):
        raise RuntimeError("durable proposal closure event has an invalid shape")
    return ProposalClosureReceipt(
        queue_closed=True,
        terminal_logical_calls=12,
        terminal_response_hashes=tuple(str(value) for value in event["terminal_response_hashes"]),
        closure_event_sha256=_sha256_json(event),
    )


def _normalize_terminal_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Replay identity and provider gates instead of trusting stored booleans."""

    if len(records) != 12:
        raise RuntimeError("shadow replay requires exactly twelve response records")
    by_ordinal: dict[int, Mapping[str, object]] = {}
    for record in records:
        ordinal = record.get("ordinal")
        if type(ordinal) is not int or ordinal in by_ordinal:
            raise RuntimeError("shadow replay found a duplicate or invalid ordinal")
        by_ordinal[ordinal] = record
    if set(by_ordinal) != set(range(1, 13)):
        raise RuntimeError("shadow replay response ordinals are incomplete")
    normalized: list[dict[str, object]] = []
    for task in FROZEN_TASKS:
        source = by_ordinal[task.ordinal]
        if (
            source.get("condition") != task.condition
            or source.get("path") != task.path
            or source.get("call_id") != task.call_id
            or source.get("prompt_sha256") != task.prompt_sha256
            or source.get("schema_sha256") != task.schema_sha256
        ):
            raise RuntimeError("shadow response identity or prompt/schema binding changed")
        row = copy.deepcopy(dict(source))
        if row.get("status") == "succeeded":
            output = row.get("output")
            if type(output) is not dict:
                raise RuntimeError("successful shadow response has no typed output")
            parsed = task.output_type.model_validate(output, strict=True)
            output_hash = _sha256_json(parsed.model_dump(mode="json"))
            if row.get("output_sha256") != output_hash:
                raise RuntimeError("shadow response output hash changed")
            attempt_count = row.get("attempt_count")
            model_gate = (
                row.get("requested_model") == MODEL
                and row.get("resolved_model") == MODEL
                and row.get("resolved_provider") == RESOLVED_PROVIDER
                and type(attempt_count) is int
                and 1 <= attempt_count <= QUEUE_MAX_ATTEMPTS
            )
            if row.get("cost_usd") is None:
                cost_gate = False
            else:
                try:
                    cost = Decimal(str(row["cost_usd"]))
                except Exception as exc:
                    raise RuntimeError("shadow successful response has invalid cost") from exc
                cost_gate = cost.is_finite() and cost >= 0
            recomputed_valid = model_gate and cost_gate
            if (
                row.get("output_contract_valid") is not True
                or row.get("model_provider_attempt_gate") is not model_gate
                or row.get("reported_cost_present") is not cost_gate
                or row.get("valid_for_scoring") is not recomputed_valid
            ):
                raise RuntimeError("shadow stored provider/scoring gates do not replay")
            row["valid_for_scoring"] = recomputed_valid
        elif row.get("status") == "failed":
            if row.get("valid_for_scoring") is not False or row.get("output") is not None:
                raise RuntimeError("failed shadow response contains admissible output")
            row["valid_for_scoring"] = False
        else:
            raise RuntimeError("shadow response has an unknown terminal status")
        normalized.append(row)
    return tuple(normalized)


def verify_durable_proposal_logs(
    *,
    responses_path: Path,
    events_path: Path,
    queue_path: Path,
) -> tuple[tuple[dict[str, object], ...], ProposalClosureReceipt]:
    """Reconstruct the proposal capability exclusively from fsynced logs."""

    responses = tuple(
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    normalized = _normalize_terminal_records(responses)
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(events) != 26:
        raise RuntimeError("durable proposal event log has an unexpected length")
    closures = [row for row in events if row.get("event_type") == "proposal_phase_closed"]
    if len(closures) != 1 or events[-1] is not closures[0]:
        raise RuntimeError("durable event log must end with one proposal closure")
    closure = closures[0]
    start = events[0]
    if (
        start.get("schema_version") != 1
        or start.get("event_type") != "proposal_phase_started"
        or start.get("logical_calls") != 12
        or start.get("oracle_parser_constructed") is not False
    ):
        raise RuntimeError("durable proposal start event is invalid")
    submission_events = events[1:13]
    for task, event in zip(FROZEN_TASKS, submission_events, strict=True):
        if (
            event.get("schema_version") != 1
            or event.get("event_type") != "shadow_call_submitted"
            or event.get("ordinal") != task.ordinal
            or event.get("condition") != task.condition
            or event.get("path") != task.path
            or event.get("call_id") != task.call_id
            or event.get("schedule_sha256") != task.schedule_sha256
        ):
            raise RuntimeError("durable proposal submissions changed order or identity")
    terminal_events = events[13:25]
    if any(
        event.get("schema_version") != 1
        or event.get("event_type") != "shadow_call_terminal"
        for event in terminal_events
    ):
        raise RuntimeError("durable proposal terminal event grammar is invalid")
    expected_hashes = tuple(_sha256_json(row) for row in normalized)
    if tuple(closure.get("terminal_response_hashes", ())) != expected_hashes:
        raise RuntimeError("proposal closure hashes do not bind durable responses")
    terminal_by_call = {str(row.get("call_id")): row for row in terminal_events}
    if set(terminal_by_call) != {task.call_id for task in FROZEN_TASKS}:
        raise RuntimeError("durable terminal events do not cover all frozen calls")
    for row in normalized:
        terminal = terminal_by_call[str(row["call_id"])]
        if (
            terminal.get("ordinal") != row["ordinal"]
            or terminal.get("condition") != row["condition"]
            or terminal.get("path") != row["path"]
            or terminal.get("status") != row["status"]
            or terminal.get("valid_for_scoring") is not row["valid_for_scoring"]
            or terminal.get("response_record_sha256") != _sha256_json(row)
        ):
            raise RuntimeError("durable terminal event does not bind its response")
    queue_records = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(queue_records) != 12
        or {str(row.get("task_id")) for row in queue_records}
        != {task.call_id for task in FROZEN_TASKS}
    ):
        raise RuntimeError("durable queue log does not cover all twelve frozen calls")
    queue_by_call = {str(row["task_id"]): row for row in queue_records}
    for row in normalized:
        queue_row = queue_by_call[str(row["call_id"])]
        queue_succeeded = queue_row.get("status") == "succeeded"
        if queue_succeeded != (row.get("status") == "succeeded"):
            raise RuntimeError("durable response and queue terminal statuses disagree")
        attempts = queue_row.get("attempts")
        if type(attempts) is not list or (
            queue_succeeded and len(attempts) != int(row["attempt_count"])
        ):
            raise RuntimeError("durable response and queue attempt counts disagree")
    receipt = _receipt_from_closure_event(closure)
    if receipt.terminal_response_hashes != expected_hashes:
        raise RuntimeError("proposal receipt response binding changed")
    return normalized, receipt


async def execute_proposal_tasks(
    *,
    predictor: Predictor,
    response_writer: v1.DurableJsonlWriter,
    event_writer: v1.DurableJsonlWriter,
) -> tuple[dict[str, object], ...]:
    """Submit the immutable task list and durably close every logical outcome."""

    async def execute(task: ShadowTask) -> dict[str, object]:
        request = StructuredGenerationRequest(
            call_id=LLMCallId(task.call_id),
            operation="boils_action_shadow",
            prompt=task.prompt,
            output_type=task.output_type,
            output_tool_name="return_action_ranking",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        try:
            result = await predictor(request)
            response, attempt_count = _provider_envelope(result)
            StructuredGenerationResponse.__post_init__(response)
            output_valid = type(response.value) is task.output_type
            if output_valid:
                task.output_type.model_validate(response.value, strict=True)
            model_provider_valid = (
                response.requested_model == MODEL
                and response.resolved_model == MODEL
                and response.resolved_provider == RESOLVED_PROVIDER
                and 1 <= attempt_count <= QUEUE_MAX_ATTEMPTS
            )
            cost_valid = (
                response.cost_usd is not None
                and response.cost_usd.is_finite()
                and response.cost_usd >= 0
            )
            record: dict[str, object] = {
                "schema_version": 1,
                "status": "succeeded",
                "ordinal": task.ordinal,
                "condition": task.condition,
                "path": task.path,
                "call_id": task.call_id,
                "prompt_sha256": task.prompt_sha256,
                "schema_sha256": task.schema_sha256,
                "output": (
                    response.value.model_dump(mode="json") if output_valid else None
                ),
                "output_sha256": (
                    _sha256_json(response.value.model_dump(mode="json"))
                    if output_valid
                    else None
                ),
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
                "cost_usd": None if response.cost_usd is None else str(response.cost_usd),
                "latency_ns": response.latency_ns,
                "attempt_count": attempt_count,
                "output_contract_valid": output_valid,
                "model_provider_attempt_gate": model_provider_valid,
                "reported_cost_present": cost_valid,
                "valid_for_scoring": output_valid and model_provider_valid and cost_valid,
            }
        except Exception as exc:
            record = {
                "schema_version": 1,
                "status": "failed",
                "ordinal": task.ordinal,
                "condition": task.condition,
                "path": task.path,
                "call_id": task.call_id,
                "prompt_sha256": task.prompt_sha256,
                "schema_sha256": task.schema_sha256,
                "failure_type": type(exc).__name__,
                "safe_message": "logical shadow call failed; inspect sanitized queue telemetry",
                "valid_for_scoring": False,
            }
        response_writer.write(record)
        event_writer.write(
            {
                "schema_version": 1,
                "event_type": "shadow_call_terminal",
                "recorded_at_utc": _utc_now(),
                "ordinal": task.ordinal,
                "condition": task.condition,
                "path": task.path,
                "call_id": task.call_id,
                "status": record["status"],
                "response_record_sha256": _sha256_json(record),
                "valid_for_scoring": record["valid_for_scoring"],
            }
        )
        return record

    pending: list[asyncio.Task[dict[str, object]]] = []
    for task in FROZEN_TASKS:
        event_writer.write(
            {
                "schema_version": 1,
                "event_type": "shadow_call_submitted",
                "recorded_at_utc": _utc_now(),
                "ordinal": task.ordinal,
                "condition": task.condition,
                "path": task.path,
                "call_id": task.call_id,
                "schedule_sha256": task.schedule_sha256,
            }
        )
        pending.append(asyncio.create_task(execute(task), name=task.call_id))
    records = tuple(await asyncio.gather(*pending))
    if len(records) != 12 or tuple(int(row["ordinal"]) for row in records) != tuple(range(1, 13)):
        raise RuntimeError("shadow proposal phase did not close all twelve frozen tasks")
    return records


async def _run_live(
    *,
    response_writer: v1.DurableJsonlWriter,
    event_writer: v1.DurableJsonlWriter,
    queue_writer: v1.DurableJsonlWriter,
) -> tuple[dict[str, object], ...]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=float(QUEUE_ATTEMPT_TIMEOUT_SECONDS),
        provider_options={"order": list(PROVIDER_ORDER), "allow_fallbacks": False},
        app_title="AgentEvolve AAAI 2027 BOiLS action-ranking shadow",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
        base_backoff_ns=QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
        max_backoff_ns=QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
        jitter_policy=DeterministicHashJitter(seed=JITTER_SEED, domain=JITTER_DOMAIN),
        close_generator=True,
        outcome_sink=lambda outcome: queue_writer.write(support._queue_outcome_record(outcome)),
    )
    async with runner:
        records = await execute_proposal_tasks(
            predictor=runner,
            response_writer=response_writer,
            event_writer=event_writer,
        )
    event_writer.write(_closure_event(records))
    return records


def _bind_canonical_composition_root() -> None:
    """Make direct-script and canonical imports resolve to one live module."""

    current = sys.modules.get(__name__)
    if current is None:
        raise RuntimeError("running shadow module is missing from sys.modules")
    existing = sys.modules.get(CANONICAL_MODULE_NAME)
    if existing is not None and existing is not current:
        raise RuntimeError("a second shadow composition-root module is already loaded")
    sys.modules[CANONICAL_MODULE_NAME] = current


def _load_frozen_post_closure_scorer(
    source_path: Path,
    *,
    expected_sha256: str,
) -> ModuleType:
    """Execute the durably copied scorer source only after proposal closure."""

    import importlib.util

    digest = support._sha256(source_path)
    if digest != expected_sha256:
        raise RuntimeError("frozen post-closure scorer source hash changed")
    module_name = f"_boils_action_shadow_scorer_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct the frozen post-closure scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/actions.py",
        "queue": AGENT_EVOLVE_ROOT / "src/agent_evolve/application/llm_task_queue.py",
        "backoff": AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/llm_backoff.py",
        "queued_runner": AGENT_EVOLVE_ROOT / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        "structured_adapter": AGENT_EVOLVE_ROOT / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
        "post_closure_scorer": SCORER_SOURCE_PATH,
        "closure_contract": CONTRACT_SOURCE_PATH,
    }
    return {name: support._sha256(path) for name, path in paths.items()}


def _manifest(
    run_id: str,
    sealed_inputs: Mapping[str, Mapping[str, object]],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Post-hoc log2 oracle-scored shadow with zero ABC evaluations; not held-out, "
            "optimizer, memory-utility, SOTA, or wall-clock evidence."
        ),
        "sealed_inputs": copy.deepcopy(dict(sealed_inputs)),
        "tasks": [task.manifest_record() for task in FROZEN_TASKS],
        "model": {
            "requested": MODEL,
            "required_resolved": MODEL,
            "required_provider": RESOLVED_PROVIDER,
            "provider_options": {"order": list(PROVIDER_ORDER), "allow_fallbacks": False},
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "successful_response_cost_ceiling_usd": str(MAX_SUCCESSFUL_RESPONSE_COST_USD),
        },
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_ns": QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
            "base_backoff_ns": QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
            "max_backoff_ns": QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
            "retry_owner": "AsyncLLMTaskQueue",
            "jitter": {"kind": "task_keyed_sha256", "seed": JITTER_SEED, "domain": JITTER_DOMAIN},
            "sdk_retries": 0,
            "pydantic_ai_retries": 0,
        },
        "catalog_sha256": EXPECTED_INPUT_SHA256["catalog"][1],
        "preoracle_evidence_sha256": EXPECTED_INPUT_SHA256["preoracle_evidence"][1],
        "source_sha256": dict(source_hashes),
        "python_source_snapshot": support._source_snapshot(
            (AGENT_EVOLVE_ROOT / "src", AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc", AGENT_EVOLVE_ROOT / "examples/development")
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "credential_variable": "OPENROUTER_API_KEY",
            "packages": {
                name: support._package_version(name)
                for name in ("pydantic", "pydantic-ai", "openai", "httpx")
            },
        },
    }


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        "scorer_source.py",
        "closure_contract_source.py",
        "preregistration.md",
        "correction.md",
        "action_catalog.json",
        "preoracle_evidence.json",
        "sealed_oracle_finalized.json",
        "sealed_oracle_evaluations.jsonl",
        "sealed_oracle_summary.json",
        "prompts.jsonl",
        "events.jsonl",
        "queue_outcomes.jsonl",
        "responses.jsonl",
        "scoring.json",
        "summary.json",
        "failure.json",
    )
    files: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if not path.exists():
            continue
        payload = resolve_corpus_path(path).read_bytes()
        record: dict[str, object] = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        if name.endswith(".jsonl"):
            record["lines"] = len(payload.splitlines())
        files[name] = record
    support._write_json(
        run_dir / "finalized.json",
        {
            "schema_version": 1,
            "status": status,
            "completed_at_utc": _utc_now(),
            "preregistration_sha256": EXPECTED_INPUT_SHA256["preregistration"][1],
            "correction_sha256": EXPECTED_INPUT_SHA256["correction"][1],
            "files": files,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-in-flight", type=int, default=QUEUE_MAX_IN_FLIGHT)
    parser.add_argument("--max-pending", type=int, default=QUEUE_MAX_PENDING)
    parser.add_argument("--max-attempts", type=int, default=QUEUE_MAX_ATTEMPTS)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=QUEUE_ATTEMPT_TIMEOUT_SECONDS)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    return parser


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "model": MODEL,
        "max_in_flight": QUEUE_MAX_IN_FLIGHT,
        "max_pending": QUEUE_MAX_PENDING,
        "max_attempts": QUEUE_MAX_ATTEMPTS,
        "attempt_timeout_seconds": QUEUE_ATTEMPT_TIMEOUT_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise SystemExit(f"BOiLS action shadow freezes --{name.replace('_', '-')}={value}")


def main() -> None:
    _bind_canonical_composition_root()
    args = _parser().parse_args()
    _assert_frozen_cli(args)
    sealed_inputs = hash_bind_inputs()
    source_hashes = _source_hashes()
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "boils_action_shadow_v1_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status = "failed"
    event_writer: v1.DurableJsonlWriter | None = None
    response_writer: v1.DurableJsonlWriter | None = None
    queue_writer: v1.DurableJsonlWriter | None = None
    try:
        copies = {
            Path(__file__).resolve(): "runner_source.py",
            SCORER_SOURCE_PATH: "scorer_source.py",
            CONTRACT_SOURCE_PATH: "closure_contract_source.py",
            PREREGISTRATION_PATH: "preregistration.md",
            CORRECTION_PATH: "correction.md",
            CATALOG_PATH: "action_catalog.json",
            EVIDENCE_PATH: "preoracle_evidence.json",
        }
        for source, name in copies.items():
            shutil.copyfile(source, run_dir / name)
        if support._sha256(run_dir / "runner_source.py") != source_hashes["runner"]:
            raise RuntimeError("durable shadow runner copy failed its source hash gate")
        if (
            support._sha256(run_dir / "scorer_source.py")
            != source_hashes["post_closure_scorer"]
        ):
            raise RuntimeError("durable shadow scorer copy failed its source hash gate")
        if (
            support._sha256(run_dir / "closure_contract_source.py")
            != source_hashes["closure_contract"]
        ):
            raise RuntimeError("durable shadow contract copy failed its source hash gate")
        prompt_writer = v1.DurableJsonlWriter(run_dir / "prompts.jsonl")
        for task in FROZEN_TASKS:
            prompt_writer.write(task.manifest_record())
        prompt_writer.close()
        support._write_json(
            run_dir / "manifest.json",
            _manifest(run_id, sealed_inputs, source_hashes),
        )
        event_writer = v1.DurableJsonlWriter(run_dir / "events.jsonl")
        response_writer = v1.DurableJsonlWriter(run_dir / "responses.jsonl")
        queue_writer = v1.DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
        event_writer.write(
            {
                "schema_version": 1,
                "event_type": "proposal_phase_started",
                "recorded_at_utc": _utc_now(),
                "logical_calls": 12,
                "oracle_parser_constructed": False,
            }
        )
        load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
        records = asyncio.run(
            _run_live(
                response_writer=response_writer,
                event_writer=event_writer,
                queue_writer=queue_writer,
            )
        )
        queue_writer.close()
        queue_summary = support._queue_log_summary(run_dir / "queue_outcomes.jsonl")
        if queue_summary["terminal_outcomes"] != 12:
            raise RuntimeError("shadow queue did not durably close exactly twelve logical tasks")
        replayed_records, proposal_receipt = verify_durable_proposal_logs(
            responses_path=run_dir / "responses.jsonl",
            events_path=run_dir / "events.jsonl",
            queue_path=run_dir / "queue_outcomes.jsonl",
        )
        if tuple(_sha256_json(row) for row in records) != tuple(
            _sha256_json(row) for row in replayed_records
        ):
            raise RuntimeError("in-memory and durable shadow responses disagree")
        # Oracle copies and JSON parsing occur only after the durable phase-close event.
        event_writer.write(
            {
                "schema_version": 1,
                "event_type": "oracle_scoring_started",
                "recorded_at_utc": _utc_now(),
                "proposal_phase_closed": True,
                "queue_terminal_outcomes": 12,
            }
        )
        scorer = _load_frozen_post_closure_scorer(
            run_dir / "scorer_source.py",
            expected_sha256=source_hashes["post_closure_scorer"],
        )

        oracle_copies = {
            "oracle_finalized": "sealed_oracle_finalized.json",
            "oracle_evaluations": "sealed_oracle_evaluations.jsonl",
            "oracle_summary": "sealed_oracle_summary.json",
        }
        for key, name in oracle_copies.items():
            source, expected = EXPECTED_INPUT_SHA256[key]
            shutil.copyfile(source, run_dir / name)
            if support._sha256(run_dir / name) != expected:
                raise RuntimeError(f"durable oracle copy failed its hash gate: {key}")
        scoring = scorer.score_shadow(
            replayed_records,
            proposal_receipt=proposal_receipt,
            oracle_loader=lambda: scorer.load_oracle_table(
                run_dir / "sealed_oracle_summary.json"
            ),
        )
        support._write_json(run_dir / "scoring.json", scoring)
        event_writer.write(
            {
                "schema_version": 1,
                "event_type": "oracle_scoring_completed",
                "recorded_at_utc": _utc_now(),
                "oracle_reproduction_gates": copy.deepcopy(
                    scoring["oracle_reproduction_gates"]
                ),
                "advanced_condition": scoring["decision"]["advanced_condition"],
            }
        )
        support._write_json(
            run_dir / "summary.json",
            {
                **scoring,
                "queue": queue_summary,
                "sealed_inputs": sealed_inputs,
                "prompt_sha256": [task.prompt_sha256 for task in FROZEN_TASKS],
                "schema_sha256": [task.schema_sha256 for task in FROZEN_TASKS],
                "zero_abc_evaluations": True,
            },
        )
        status = "succeeded"
    except BaseException as exc:
        support._write_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "safe_message": "BOiLS action shadow failed; inspect sanitized durable traces",
            },
        )
        raise
    finally:
        for writer in (queue_writer, response_writer, event_writer):
            if writer is not None:
                writer.close()
        _finalize(run_dir, status)
    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()
