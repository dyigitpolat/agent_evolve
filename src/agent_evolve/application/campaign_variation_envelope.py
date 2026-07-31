"""Generic batch boundary for adding proposal experts to campaign variation.

Parent-local benchmark catalogs remain the legal structural foundation.  An
optional envelope policy may add sealed full-child options after observing the
strictly prior campaign archive.  The policy is invoked once per portfolio
generation, across all parent lanes, so global acquisition and restart experts
can partition candidates without duplicating them independently in each lane.

This boundary owns no workload, model, provider, objective, or acquisition
implementation.  It only authenticates chronology, lane identity, preservation
of the base legal support, and the exact enriched contracts returned to the
ordinary portfolio workflow.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import OptimizerState
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    ParentVariationBinding,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_REQUEST_DOMAIN = b"agent-evolve:campaign-variation-envelope-request:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:campaign-variation-envelope-result:v1\x00"
_CONTEXT_DOMAIN = b"agent-evolve:campaign-variation-envelope-context:v1\x00"
_TRACE_DOMAIN = b"agent-evolve:campaign-variation-envelope-trace:v2\x00"
_TRACE_PAYLOAD_MAX_UTF8_BYTES = 64 * 1024 * 1024
_TRACE_PAYLOAD_MAX_COMPRESSED_BYTES = 4 * 1024 * 1024
_TRACE_PAYLOAD_CHUNK_CHARACTERS = 32 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _policy_identity(value: object) -> tuple[str, int, str]:
    policy_id = getattr(value, "policy_id", None)
    policy_version = getattr(value, "policy_version", None)
    definition_sha256 = getattr(value, "definition_sha256", None)
    if definition_sha256 is None:
        definition_sha256 = getattr(value, "policy_definition_sha256", None)
    if type(policy_id) is not str or _TOKEN.fullmatch(policy_id) is None:
        raise ValueError("variation-envelope policy_id has invalid syntax")
    if type(policy_version) is not int or policy_version <= 0:
        raise ValueError("variation-envelope policy_version must be positive")
    require_sha256(definition_sha256, "variation-envelope definition_sha256")
    return policy_id, policy_version, definition_sha256


@dataclass(frozen=True, slots=True)
class CampaignVariationEnvelopeLane:
    """One parent lane and its outcome-blind benchmark catalog binding."""

    lane_id: str
    parent: EvolutionCandidate
    base_variation: ParentVariationBinding

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or _TOKEN.fullmatch(self.lane_id) is None:
            raise ValueError("lane_id must use the closed token grammar")
        if type(self.parent) is not EvolutionCandidate:
            raise TypeError("parent must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(self.parent)
        if type(self.base_variation) is not ParentVariationBinding:
            raise TypeError("base_variation must be exact")
        ParentVariationBinding.__post_init__(self.base_variation)
        if (
            self.base_variation.parent_configuration_sha256
            != self.parent.occurrence.configuration_hash
        ):
            raise ValueError("base variation is bound to another parent")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "lane_id": self.lane_id,
            "parent_candidate_id": self.parent.candidate_id.value,
            "parent_configuration_sha256": (
                self.parent.occurrence.configuration_hash
            ),
            "base_variation": self.base_variation.to_record(),
        }


@dataclass(frozen=True, slots=True)
class CampaignVariationEnvelopeRequest:
    """Authenticated prior-only input to one global proposal envelope."""

    campaign_scope_sha256: str
    generation: int
    evaluation_slots_per_lane: int
    state: OptimizerState
    archive_utility: ArchiveUtilitySnapshot
    lanes: tuple[CampaignVariationEnvelopeLane, ...]
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if (
            type(self.evaluation_slots_per_lane) is not int
            or self.evaluation_slots_per_lane <= 0
        ):
            raise ValueError("evaluation_slots_per_lane must be positive")
        if type(self.state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        OptimizerState.__post_init__(self.state)
        if self.state.generation != self.generation - 1:
            raise ValueError("variation envelope received a non-prior state")
        if type(self.archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("archive_utility must be exact")
        ArchiveUtilitySnapshot.__post_init__(self.archive_utility)
        if self.archive_utility.generation != self.generation:
            raise ValueError("archive utility is stale for the envelope generation")
        if type(self.lanes) is not tuple or not self.lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        if any(type(value) is not CampaignVariationEnvelopeLane for value in self.lanes):
            raise TypeError("lanes must contain exact envelope lanes")
        for value in self.lanes:
            CampaignVariationEnvelopeLane.__post_init__(value)
        lane_ids = tuple(value.lane_id for value in self.lanes)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("variation-envelope lanes must be unique and sorted")
        benchmark_ids = {
            value.base_variation.benchmark_sha256 for value in self.lanes
        }
        if len(benchmark_ids) != 1:
            raise ValueError("variation-envelope lanes target different benchmarks")
        if self.archive_utility.benchmark_sha256 not in benchmark_ids:
            raise ValueError("archive utility targets a different benchmark")
        computed = _hash(_REQUEST_DOMAIN, self._unsigned_record())
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "generation": self.generation,
            "evaluation_slots_per_lane": self.evaluation_slots_per_lane,
            "optimizer_state": {
                "generation": self.state.generation,
                "archive_snapshot_sha256": self.state.archive_snapshot_hash,
                "unique_evaluations": self.state.unique_evaluations,
                "logical_llm_calls": self.state.logical_llm_calls,
                "candidate_occurrences": [
                    {
                        "candidate_id": value.candidate_id.value,
                        "configuration_sha256": (
                            value.occurrence.configuration_hash
                        ),
                        "objectives_hex": [
                            [metric_id, metric_value.hex()]
                            for metric_id, metric_value in value.objectives
                        ],
                        "valid": value.valid,
                    }
                    for value in self.state.candidates
                ],
            },
            "archive_utility_sha256": self.archive_utility.snapshot_sha256,
            "lanes": [value.to_record() for value in self.lanes],
            "current_generation_outcomes_exposed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class CampaignVariationEnvelopeLaneResult:
    """One enriched binding, preserving every base option by identity."""

    lane_id: str
    variation: ParentVariationBinding

    def __post_init__(self) -> None:
        if type(self.lane_id) is not str or _TOKEN.fullmatch(self.lane_id) is None:
            raise ValueError("lane_id must use the closed token grammar")
        if type(self.variation) is not ParentVariationBinding:
            raise TypeError("variation must be exact")
        ParentVariationBinding.__post_init__(self.variation)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"lane_id": self.lane_id, "variation": self.variation.to_record()}


@dataclass(frozen=True, slots=True)
class CampaignVariationEnvelopeResult:
    """Exact per-lane enriched contracts and policy evidence."""

    request_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    lanes: tuple[CampaignVariationEnvelopeLaneResult, ...]
    evidence: FrozenJsonObject
    result_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _policy_identity(self)
        if type(self.lanes) is not tuple or not self.lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        if any(
            type(value) is not CampaignVariationEnvelopeLaneResult
            for value in self.lanes
        ):
            raise TypeError("lanes must contain exact result lanes")
        for value in self.lanes:
            CampaignVariationEnvelopeLaneResult.__post_init__(value)
        lane_ids = tuple(value.lane_id for value in self.lanes)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("result lanes must be unique and sorted")
        if type(self.evidence) is not FrozenJsonObject or freeze_json(
            self.evidence
        ) is not self.evidence:
            raise TypeError("evidence must be an exact frozen object")
        computed = _hash(_RESULT_DOMAIN, self._unsigned_record())
        if self.result_sha256 not in ("", computed):
            raise ValueError("result_sha256 does not authenticate the result")
        object.__setattr__(self, "result_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "lanes": [value.to_record() for value in self.lanes],
            "evidence": thaw_json(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "result_sha256": self.result_sha256}


@runtime_checkable
class CampaignVariationEnvelopePolicy(Protocol):
    """Add globally coordinated proposal support before ordinary selection."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def enrich(
        self,
        request: CampaignVariationEnvelopeRequest,
    ) -> CampaignVariationEnvelopeResult: ...


def validate_campaign_variation_envelope_result(
    *,
    policy: CampaignVariationEnvelopePolicy,
    request: CampaignVariationEnvelopeRequest,
    result: CampaignVariationEnvelopeResult,
) -> None:
    """Revalidate policy identity, chronology, and support preservation."""

    if not isinstance(policy, CampaignVariationEnvelopePolicy):
        raise TypeError("policy must implement CampaignVariationEnvelopePolicy")
    policy_identity = _policy_identity(policy)
    if type(request) is not CampaignVariationEnvelopeRequest:
        raise TypeError("request must be exact")
    CampaignVariationEnvelopeRequest.__post_init__(request)
    if type(result) is not CampaignVariationEnvelopeResult:
        raise TypeError("result must be exact")
    CampaignVariationEnvelopeResult.__post_init__(result)
    if result.request_sha256 != request.request_sha256:
        raise ValueError("variation-envelope result targets another request")
    if (
        result.policy_id,
        result.policy_version,
        result.policy_definition_sha256,
    ) != policy_identity:
        raise ValueError("variation-envelope result has a foreign policy identity")
    if tuple(value.lane_id for value in result.lanes) != tuple(
        value.lane_id for value in request.lanes
    ):
        raise ValueError("variation-envelope result changed the lane universe")

    for source, enriched in zip(request.lanes, result.lanes, strict=True):
        base = source.base_variation
        variation = enriched.variation
        if (
            variation.benchmark_sha256 != base.benchmark_sha256
            or variation.parent_configuration_sha256
            != base.parent_configuration_sha256
            or variation.known_phenotype_sha256s
            != base.known_phenotype_sha256s
        ):
            raise ValueError("enriched variation escaped its lane binding")
        base_options = {
            value.identity_sha256 for value in base.contract.options
        }
        enriched_options = {
            value.identity_sha256 for value in variation.contract.options
        }
        if not base_options <= enriched_options:
            raise ValueError("variation envelope removed base legal support")


def campaign_variation_envelope_context_record(
    result: CampaignVariationEnvelopeResult,
) -> dict[str, object]:
    """Project a result to compact prompt evidence without contract duplication.

    The selection request already carries the exact enriched contract for its
    lane.  Repeating every lane's full options inside the context increases
    model input while revealing no additional fact.  This receipt retains the
    authenticated result identity, policy, compact policy evidence, and exact
    per-lane contract identities/counts.
    """

    if type(result) is not CampaignVariationEnvelopeResult:
        raise TypeError("result must be an exact variation-envelope result")
    CampaignVariationEnvelopeResult.__post_init__(result)
    unsigned = {
        "schema_version": 1,
        "request_sha256": result.request_sha256,
        "result_sha256": result.result_sha256,
        "policy_id": result.policy_id,
        "policy_version": result.policy_version,
        "policy_definition_sha256": result.policy_definition_sha256,
        "lanes": [
            {
                "lane_id": value.lane_id,
                "parent_configuration_sha256": (
                    value.variation.parent_configuration_sha256
                ),
                "finite_contract_identity_sha256": (
                    value.variation.contract.identity_sha256
                ),
                "eligible_option_count": len(value.variation.contract.options),
            }
            for value in result.lanes
        ],
        "evidence": thaw_json(result.evidence),
        "full_variation_contracts_repeated_in_context": False,
    }
    return {
        **unsigned,
        "context_receipt_sha256": _hash(_CONTEXT_DOMAIN, unsigned),
    }


def campaign_variation_envelope_trace_record(
    *,
    request: CampaignVariationEnvelopeRequest,
    result: CampaignVariationEnvelopeResult,
) -> dict[str, object]:
    """Record the complete eligible expert union for replay and support audits.

    Prompt context deliberately omits full contracts.  The durable stage trace
    has a different purpose: it must preserve external-expert configurations
    that were legal but not selected, otherwise support and displacement regret
    cannot be measured after the run.  Both the workload-owned base support and
    options added by external proposal experts are therefore included with an
    explicit origin. Evaluated dispositions remain in the ordinary
    portfolio-wave receipts and join exactly through ``option_identity_sha256``.
    """

    if type(request) is not CampaignVariationEnvelopeRequest:
        raise TypeError("request must be an exact variation-envelope request")
    CampaignVariationEnvelopeRequest.__post_init__(request)
    if type(result) is not CampaignVariationEnvelopeResult:
        raise TypeError("result must be an exact variation-envelope result")
    CampaignVariationEnvelopeResult.__post_init__(result)
    if result.request_sha256 != request.request_sha256:
        raise ValueError("variation-envelope trace joins a foreign result")
    if tuple(value.lane_id for value in result.lanes) != tuple(
        value.lane_id for value in request.lanes
    ):
        raise ValueError("variation-envelope trace changed the lane universe")

    lane_records: list[dict[str, object]] = []
    added_count = 0
    eligible_count = 0
    for source, enriched in zip(request.lanes, result.lanes, strict=True):
        base_options = {
            value.identity_sha256 for value in source.base_variation.contract.options
        }
        eligibility = enriched.variation.eligibility_receipt
        phenotype_by_identity = (
            {}
            if eligibility is None
            else {
                value.option_identity_sha256: value.phenotype_identity_sha256
                for value in eligibility.option_phenotypes
            }
        )
        eligible_options = []
        for option in enriched.variation.contract.options:
            support_origin = (
                "base"
                if option.identity_sha256 in base_options
                else "envelope_addition"
            )
            eligible_options.append(
                {
                    "option": option.evidence_record(),
                    "child_configuration": thaw_json(option.child_configuration),
                    "phenotype_identity_sha256": phenotype_by_identity.get(
                        option.identity_sha256
                    ),
                    "eligibility_disposition": "eligible",
                    "support_origin": support_origin,
                }
            )
        lane_added_count = sum(
            value["support_origin"] == "envelope_addition"
            for value in eligible_options
        )
        added_count += lane_added_count
        eligible_count += len(eligible_options)
        lane_records.append(
            {
                "lane_id": source.lane_id,
                "parent_candidate_id": source.parent.candidate_id.value,
                "parent_configuration_sha256": (
                    source.base_variation.parent_configuration_sha256
                ),
                "base_contract_identity_sha256": (
                    source.base_variation.contract.identity_sha256
                ),
                "enriched_contract_identity_sha256": (
                    enriched.variation.contract.identity_sha256
                ),
                "base_option_count": len(source.base_variation.contract.options),
                "eligible_option_count": len(enriched.variation.contract.options),
                "eligible_added_option_count": lane_added_count,
                "eligible_options": eligible_options,
            }
        )

    payload = {
        "schema_version": 1,
        "request_sha256": request.request_sha256,
        "result_sha256": result.result_sha256,
        "lanes": lane_records,
        "evidence": thaw_json(result.evidence),
    }
    raw_payload = _canonical_json(payload)
    if len(raw_payload) > _TRACE_PAYLOAD_MAX_UTF8_BYTES:
        raise ValueError("variation-envelope trace payload exceeds its ceiling")
    compressed_payload = zlib.compress(raw_payload, level=9)
    if len(compressed_payload) > _TRACE_PAYLOAD_MAX_COMPRESSED_BYTES:
        raise ValueError("variation-envelope compressed trace exceeds its ceiling")
    encoded_payload = base64.b64encode(compressed_payload).decode("ascii")
    payload_chunks = [
        encoded_payload[offset : offset + _TRACE_PAYLOAD_CHUNK_CHARACTERS]
        for offset in range(0, len(encoded_payload), _TRACE_PAYLOAD_CHUNK_CHARACTERS)
    ]
    unsigned = {
        "schema_version": 2,
        "request_sha256": request.request_sha256,
        "result_sha256": result.result_sha256,
        "policy_id": result.policy_id,
        "policy_version": result.policy_version,
        "policy_definition_sha256": result.policy_definition_sha256,
        "generation": request.generation,
        "eligible_option_occurrence_count": eligible_count,
        "eligible_added_option_count": added_count,
        "lanes": [
            {
                key: value
                for key, value in lane.items()
                if key != "eligible_options"
            }
            for lane in lane_records
        ],
        "payload_encoding": "zlib_base64_chunks_v1",
        "payload_base64_chunks": payload_chunks,
        "payload_utf8_bytes": len(raw_payload),
        "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "payload_compressed_bytes": len(compressed_payload),
        "payload_compressed_sha256": hashlib.sha256(
            compressed_payload
        ).hexdigest(),
        "full_child_configurations_included": True,
        "full_expert_union_included": True,
        "exposed_to_model_prompt": False,
        "evaluated_disposition_join": {
            "record": "portfolio_wave_receipts.action_attributions",
            "key": "option_identity_sha256",
        },
    }
    return {**unsigned, "trace_receipt_sha256": _hash(_TRACE_DOMAIN, unsigned)}


def decode_campaign_variation_envelope_trace_record(
    record: dict[str, object],
) -> dict[str, object]:
    """Authenticate and recover one losslessly compressed expert-union trace."""

    if type(record) is not dict:
        raise TypeError("variation-envelope trace record must be an exact object")
    if record.get("schema_version") != 2:
        raise ValueError("variation-envelope trace record has an unknown schema")
    trace_receipt_sha256 = record.get("trace_receipt_sha256")
    require_sha256(trace_receipt_sha256, "trace_receipt_sha256")
    unsigned = {
        key: value for key, value in record.items() if key != "trace_receipt_sha256"
    }
    if _hash(_TRACE_DOMAIN, unsigned) != trace_receipt_sha256:
        raise ValueError("variation-envelope trace receipt authentication failed")
    if record.get("payload_encoding") != "zlib_base64_chunks_v1":
        raise ValueError("variation-envelope trace payload encoding is unsupported")
    chunks = record.get("payload_base64_chunks")
    if (
        type(chunks) is not list
        or not chunks
        or any(type(value) is not str or not value for value in chunks)
    ):
        raise ValueError("variation-envelope trace chunks are malformed")
    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("variation-envelope trace base64 is malformed") from exc
    expected_compressed_bytes = record.get("payload_compressed_bytes")
    if (
        type(expected_compressed_bytes) is not int
        or isinstance(expected_compressed_bytes, bool)
        or expected_compressed_bytes <= 0
        or expected_compressed_bytes > _TRACE_PAYLOAD_MAX_COMPRESSED_BYTES
        or len(compressed) != expected_compressed_bytes
        or hashlib.sha256(compressed).hexdigest()
        != record.get("payload_compressed_sha256")
    ):
        raise ValueError("variation-envelope compressed payload authentication failed")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(
            compressed,
            _TRACE_PAYLOAD_MAX_UTF8_BYTES + 1,
        )
    except zlib.error as exc:
        raise ValueError("variation-envelope trace compression is malformed") from exc
    if (
        len(raw) > _TRACE_PAYLOAD_MAX_UTF8_BYTES
        or decompressor.unconsumed_tail
        or not decompressor.eof
        or decompressor.unused_data
    ):
        raise ValueError("variation-envelope trace payload exceeds or violates framing")
    try:
        raw += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("variation-envelope trace compression is malformed") from exc
    if len(raw) > _TRACE_PAYLOAD_MAX_UTF8_BYTES:
        raise ValueError("variation-envelope trace payload exceeds its ceiling")
    expected_utf8_bytes = record.get("payload_utf8_bytes")
    if (
        type(expected_utf8_bytes) is not int
        or isinstance(expected_utf8_bytes, bool)
        or expected_utf8_bytes <= 0
        or len(raw) != expected_utf8_bytes
        or hashlib.sha256(raw).hexdigest() != record.get("payload_sha256")
    ):
        raise ValueError("variation-envelope trace payload authentication failed")
    try:
        payload = json.loads(raw.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("variation-envelope trace payload is not canonical JSON") from exc
    if type(payload) is not dict:
        raise TypeError("variation-envelope trace payload must be an object")
    if _canonical_json(payload) != raw:
        raise ValueError("variation-envelope trace payload is not canonical JSON")
    if (
        payload.get("schema_version") != 1
        or payload.get("request_sha256") != record.get("request_sha256")
        or payload.get("result_sha256") != record.get("result_sha256")
    ):
        raise ValueError("variation-envelope trace payload joins a foreign receipt")
    lanes = payload.get("lanes")
    if type(lanes) is not list or any(type(value) is not dict for value in lanes):
        raise TypeError("variation-envelope trace payload lanes are malformed")
    compact_lanes = record.get("lanes")
    if (
        type(compact_lanes) is not list
        or len(compact_lanes) != len(lanes)
        or any(type(value) is not dict for value in compact_lanes)
    ):
        raise TypeError("variation-envelope trace lane summaries are malformed")
    eligible_options: list[dict[str, object]] = []
    for lane, compact_lane in zip(lanes, compact_lanes, strict=True):
        raw_options = lane.get("eligible_options")
        if type(raw_options) is not list or any(
            type(value) is not dict for value in raw_options
        ):
            raise TypeError("variation-envelope trace eligible options are malformed")
        lane_summary = {key: value for key, value in lane.items() if key != "eligible_options"}
        if lane_summary != compact_lane:
            raise ValueError("variation-envelope trace lane summary is inconsistent")
        eligible_options.extend(raw_options)
    if len(eligible_options) != record.get("eligible_option_occurrence_count"):
        raise ValueError("variation-envelope trace eligible count is inconsistent")
    if (
        sum(
            value.get("support_origin") == "envelope_addition"
            for value in eligible_options
        )
        != record.get("eligible_added_option_count")
    ):
        raise ValueError("variation-envelope trace addition count is inconsistent")
    return payload


__all__ = [
    "CampaignVariationEnvelopeLane",
    "CampaignVariationEnvelopeLaneResult",
    "CampaignVariationEnvelopePolicy",
    "CampaignVariationEnvelopeRequest",
    "CampaignVariationEnvelopeResult",
    "campaign_variation_envelope_context_record",
    "campaign_variation_envelope_trace_record",
    "decode_campaign_variation_envelope_trace_record",
    "validate_campaign_variation_envelope_result",
]
