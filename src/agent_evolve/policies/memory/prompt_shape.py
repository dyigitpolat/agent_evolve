"""Treatment-blinded prompt-shape commitments for causal memory arms.

An exact rendered-prompt hash cannot match across adaptive and control arms:
the selected insight text is the treatment.  This policy instead commits to a
closed projection of every non-treatment input plus only the treatment payload
cardinality.  Application code owns projection of rich parent/schema objects
to the exact hashes accepted here; the renderer pairing remains explicit and
versioned.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_COMMITMENT_DOMAIN = b"agent-evolve:prompt-shape-commitment:v1\x00"
_RENDERED_STRUCTURE_DOMAIN = b"agent-evolve:rendered-prompt-structure:v1\x00"
_RENDERED_RECEIPT_DOMAIN = b"agent-evolve:matched-prompt-structure-receipt:v1\x00"


def _require_sha256(value: str, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _json_shape(value: object) -> object:
    """Project JSON to keys/cardinalities/scalar types, never scalar values."""

    if type(value) is dict:
        return {key: _json_shape(value[key]) for key in sorted(value)}
    if type(value) is list:
        return {
            "kind": "array",
            "length": len(value),
            "items": [_json_shape(item) for item in value],
        }
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        if not math.isfinite(value):  # json.loads rejects no values by itself.
            raise ValueError("rendered prompt JSON must contain finite numbers")
        return "number"
    if type(value) is str:
        return "string"
    raise TypeError("rendered prompt contains an unsupported JSON value")


def rendered_prompt_structure(prompt: str) -> tuple[object, ...]:
    """Return the renderer-bound structural projection of one actual prompt.

    The default renderer emits each structured payload as compact JSON on one
    line.  Those payloads are compared by object keys, array cardinalities, and
    scalar types; all surrounding renderer text remains exact.  This catches
    arm-specific headings, fields, optional values, and list lengths without
    hashing the treatment prose itself into the matched-arm estimand.
    """

    if type(prompt) is not str or not prompt:
        raise ValueError("prompt must be non-empty exact text")
    rows: list[object] = []
    for line in prompt.split("\n"):
        if line == "":
            rows.append(("blank",))
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            rows.append(("text", line))
        else:
            rows.append(("json", _json_shape(parsed)))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MatchedPromptStructureReceipt:
    """Receipt proving actual rendered prompts share one structural shape."""

    renderer_policy_id: str
    renderer_policy_version: int
    prompt_sha256s: tuple[str, ...]
    structure_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.renderer_policy_id, "renderer_policy_id")
        if (
            type(self.renderer_policy_version) is not int
            or self.renderer_policy_version <= 0
        ):
            raise ValueError("renderer_policy_version must be positive")
        if type(self.prompt_sha256s) is not tuple or not self.prompt_sha256s:
            raise ValueError("prompt_sha256s must be a non-empty exact tuple")
        for value in self.prompt_sha256s:
            _require_sha256(value, "prompt_sha256s entry")
        if len(set(self.prompt_sha256s)) != len(self.prompt_sha256s):
            raise ValueError("matched prompts must be distinct rendered treatments")
        _require_sha256(self.structure_sha256, "structure_sha256")
        record = self.to_record()
        object.__setattr__(
            self,
            "receipt_sha256",
            hashlib.sha256(
                _RENDERED_RECEIPT_DOMAIN + _canonical_json(record)
            ).hexdigest(),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "renderer_policy_id": self.renderer_policy_id,
            "renderer_policy_version": self.renderer_policy_version,
            "prompt_sha256s": list(self.prompt_sha256s),
            "structure_sha256": self.structure_sha256,
        }


def seal_matched_prompt_structure(
    prompts: tuple[str, ...],
    *,
    renderer_policy_id: str = "default_evidence_prompt",
    renderer_policy_version: int = 1,
) -> MatchedPromptStructureReceipt:
    """Fail closed unless distinct actual prompts have identical structure."""

    if type(prompts) is not tuple or not prompts:
        raise ValueError("prompts must be a non-empty exact tuple")
    if any(type(prompt) is not str or not prompt for prompt in prompts):
        raise TypeError("prompts must contain non-empty exact strings")
    structures = tuple(rendered_prompt_structure(prompt) for prompt in prompts)
    structure_hashes = tuple(
        hashlib.sha256(
            _RENDERED_STRUCTURE_DOMAIN + _canonical_json(structure)
        ).hexdigest()
        for structure in structures
    )
    if len(set(structure_hashes)) != 1:
        raise ValueError("rendered treatment prompts do not share one structure")
    return MatchedPromptStructureReceipt(
        renderer_policy_id=renderer_policy_id,
        renderer_policy_version=renderer_policy_version,
        prompt_sha256s=tuple(
            hashlib.sha256(prompt.encode("utf-8", errors="strict")).hexdigest()
            for prompt in prompts
        ),
        structure_sha256=structure_hashes[0],
    )


@dataclass(frozen=True, slots=True)
class PromptShapeInputs:
    """Complete non-treatment facts required to render one variation prompt.

    Hash fields are deliberately computed outside this pure policy.  In
    particular, ``parent_evidence_sha256s`` must identify the exact ordered
    parent projections exposed by the renderer, while
    ``candidate_schema_sha256`` identifies the candidate-validation boundary.
    Restricted output representations are additionally bound by their mode
    and finite/exact contract identities.  No insight ID, text, score, arm,
    block, or randomization result is accepted by this type.
    """

    problem_description_sha256: str
    exact_context_hash: str
    parent_evidence_sha256s: tuple[str, ...]
    common_ancestor_evidence_sha256: str | None
    operator_kind: str
    operator_version: int
    phase: str
    allowed_top_level: tuple[str, ...]
    mutation_contract_sha256: str | None
    mutation_response_mode: str
    atomic_replacement_option_sha256s: tuple[str, ...]
    candidate_schema_sha256: str
    selected_insight_count: int
    reward_definition_hash: str
    max_output_tokens: int
    temperature: float | None
    finite_variation_contract_sha256: str | None = None
    crossover_response_mode: str = "full_configuration"
    exact_parent_crossover_contract_sha256: str | None = None
    exact_parent_import_exclusions_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "problem_description_sha256",
            "exact_context_hash",
            "candidate_schema_sha256",
            "reward_definition_hash",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.parent_evidence_sha256s) is not tuple
            or not self.parent_evidence_sha256s
        ):
            raise ValueError("parent_evidence_sha256s must be a non-empty exact tuple")
        for value in self.parent_evidence_sha256s:
            _require_sha256(value, "parent_evidence_sha256s entry")
        if self.common_ancestor_evidence_sha256 is not None:
            _require_sha256(
                self.common_ancestor_evidence_sha256,
                "common_ancestor_evidence_sha256",
            )
        for name in (
            "operator_kind",
            "mutation_response_mode",
            "crossover_response_mode",
        ):
            _require_token(getattr(self, name), name)
        if type(self.operator_version) is not int or self.operator_version <= 0:
            raise ValueError("operator_version must be a positive exact integer")
        if (
            type(self.phase) is not str
            or not self.phase
            or self.phase != self.phase.strip()
        ):
            raise ValueError("phase must be canonical non-empty text")
        if type(self.allowed_top_level) is not tuple or any(
            type(value) is not str or not value or value != value.strip()
            for value in self.allowed_top_level
        ):
            raise TypeError(
                "allowed_top_level must be an exact tuple of canonical strings"
            )
        if len(set(self.allowed_top_level)) != len(self.allowed_top_level):
            raise ValueError("allowed_top_level cannot contain duplicates")
        if self.mutation_contract_sha256 is not None:
            _require_sha256(
                self.mutation_contract_sha256,
                "mutation_contract_sha256",
            )
        if type(self.atomic_replacement_option_sha256s) is not tuple:
            raise TypeError("atomic_replacement_option_sha256s must be an exact tuple")
        for value in self.atomic_replacement_option_sha256s:
            _require_sha256(value, "atomic replacement option digest")
        if len(set(self.atomic_replacement_option_sha256s)) != len(
            self.atomic_replacement_option_sha256s
        ):
            raise ValueError("atomic replacement option digests cannot repeat")
        if self.finite_variation_contract_sha256 is not None:
            _require_sha256(
                self.finite_variation_contract_sha256,
                "finite_variation_contract_sha256",
            )
        if self.exact_parent_crossover_contract_sha256 is not None:
            _require_sha256(
                self.exact_parent_crossover_contract_sha256,
                "exact_parent_crossover_contract_sha256",
            )
        if self.exact_parent_import_exclusions_sha256 is not None:
            _require_sha256(
                self.exact_parent_import_exclusions_sha256,
                "exact_parent_import_exclusions_sha256",
            )
        if self.crossover_response_mode == "exact_parent_import_v1":
            if self.exact_parent_crossover_contract_sha256 is None:
                raise ValueError(
                    "exact parent import requires a crossover contract digest"
                )
            if self.exact_parent_import_exclusions_sha256 is None:
                raise ValueError("exact parent import requires an exclusions digest")
        elif (
            self.exact_parent_crossover_contract_sha256 is not None
            or self.exact_parent_import_exclusions_sha256 is not None
        ):
            raise ValueError("an exact crossover digest requires exact parent import")
        if type(self.selected_insight_count) is not int:
            raise TypeError("selected_insight_count must be an exact integer")
        if self.selected_insight_count < 0:
            raise ValueError("selected_insight_count cannot be negative")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive exact integer")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
        ):
            raise ValueError("temperature must be finite or None")

    def to_record(self) -> dict[str, object]:
        """Return the complete treatment-blinded JSON-ready projection."""

        record: dict[str, object] = {
            "problem_description_sha256": self.problem_description_sha256,
            "exact_context_hash": self.exact_context_hash,
            "parent_evidence_sha256s": list(self.parent_evidence_sha256s),
            "common_ancestor_evidence_sha256": (self.common_ancestor_evidence_sha256),
            "operator_kind": self.operator_kind,
            "operator_version": self.operator_version,
            "phase": self.phase,
            "allowed_top_level": list(self.allowed_top_level),
            "mutation_contract_sha256": self.mutation_contract_sha256,
            "mutation_response_mode": self.mutation_response_mode,
            "atomic_replacement_option_sha256s": list(
                self.atomic_replacement_option_sha256s
            ),
            "candidate_schema_sha256": self.candidate_schema_sha256,
            "selected_insight_count": self.selected_insight_count,
            "reward_definition_hash": self.reward_definition_hash,
            "max_output_tokens": self.max_output_tokens,
            "temperature_hex": (
                None if self.temperature is None else float(self.temperature).hex()
            ),
        }
        # Optional extension preserves byte-identical commitments for all
        # pre-existing full-configuration and atomic plans.
        if self.finite_variation_contract_sha256 is not None:
            record["finite_variation_contract_sha256"] = (
                self.finite_variation_contract_sha256
            )
        # The historical omission is the canonical full-configuration mode.
        # Keeping it implicit preserves byte-identical commitments for every
        # pre-existing plan.  Any bounded crossover representation is explicit
        # and therefore cannot collide with that legacy meaning.
        if self.crossover_response_mode != "full_configuration":
            record["crossover_response_mode"] = self.crossover_response_mode
        if self.exact_parent_crossover_contract_sha256 is not None:
            record["exact_parent_crossover_contract_sha256"] = (
                self.exact_parent_crossover_contract_sha256
            )
        if self.exact_parent_import_exclusions_sha256 is not None:
            record["exact_parent_import_exclusions_sha256"] = (
                self.exact_parent_import_exclusions_sha256
            )
        return record


@runtime_checkable
class PromptShapeCommitmentPolicy(Protocol):
    """Versioned pure policy paired with one trusted prompt renderer."""

    policy_id: str
    policy_version: int
    renderer_policy_id: str
    renderer_policy_version: int

    def commit(self, inputs: PromptShapeInputs) -> str: ...


@dataclass(frozen=True, slots=True)
class DefaultEvidencePromptShapePolicyV1:
    """Shape policy paired exclusively with ``default_evidence_prompt`` v1."""

    policy_id: str = "treatment_blinded_prompt_shape"
    policy_version: int = 1
    renderer_policy_id: str = "default_evidence_prompt"
    renderer_policy_version: int = 1

    def __post_init__(self) -> None:
        if self.policy_id != "treatment_blinded_prompt_shape":
            raise ValueError("unsupported prompt-shape policy_id")
        if self.policy_version != 1:
            raise ValueError("unsupported prompt-shape policy_version")
        if self.renderer_policy_id != "default_evidence_prompt":
            raise ValueError("shape policy is paired with default_evidence_prompt")
        if self.renderer_policy_version != 1:
            raise ValueError("unsupported renderer_policy_version")

    def commit(self, inputs: PromptShapeInputs) -> str:
        if type(inputs) is not PromptShapeInputs:
            raise TypeError("inputs must be exact PromptShapeInputs")
        PromptShapeInputs.__post_init__(inputs)
        record = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "renderer_policy_id": self.renderer_policy_id,
            "renderer_policy_version": self.renderer_policy_version,
            "treatment_payload": {
                "kind": "selected_insight_records",
                "committed_property": "cardinality_only",
                "cardinality": inputs.selected_insight_count,
            },
            "non_treatment_inputs": inputs.to_record(),
        }
        return hashlib.sha256(_COMMITMENT_DOMAIN + _canonical_json(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class DefaultEvidencePromptShapePolicyV2:
    """Shape policy paired with rooted candidate-component paths in renderer v2."""

    policy_id: str = "treatment_blinded_prompt_shape"
    policy_version: int = 2
    renderer_policy_id: str = "default_evidence_prompt"
    renderer_policy_version: int = 2

    def __post_init__(self) -> None:
        if self.policy_id != "treatment_blinded_prompt_shape":
            raise ValueError("unsupported prompt-shape policy_id")
        if self.policy_version != 2:
            raise ValueError("unsupported prompt-shape policy_version")
        if self.renderer_policy_id != "default_evidence_prompt":
            raise ValueError("shape policy is paired with default_evidence_prompt")
        if self.renderer_policy_version != 2:
            raise ValueError("unsupported renderer_policy_version")

    def commit(self, inputs: PromptShapeInputs) -> str:
        if type(inputs) is not PromptShapeInputs:
            raise TypeError("inputs must be exact PromptShapeInputs")
        PromptShapeInputs.__post_init__(inputs)
        record = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "renderer_policy_id": self.renderer_policy_id,
            "renderer_policy_version": self.renderer_policy_version,
            "treatment_payload": {
                "kind": "selected_insight_records",
                "committed_property": "cardinality_only",
                "cardinality": inputs.selected_insight_count,
            },
            "non_treatment_inputs": inputs.to_record(),
        }
        return hashlib.sha256(_COMMITMENT_DOMAIN + _canonical_json(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class DefaultEvidencePromptShapePolicyV3:
    """Shape policy paired with discriminating crossover evidence in renderer v3."""

    policy_id: str = "treatment_blinded_prompt_shape"
    policy_version: int = 3
    renderer_policy_id: str = "default_evidence_prompt"
    renderer_policy_version: int = 3

    def __post_init__(self) -> None:
        if self.policy_id != "treatment_blinded_prompt_shape":
            raise ValueError("unsupported prompt-shape policy_id")
        if self.policy_version != 3:
            raise ValueError("unsupported prompt-shape policy_version")
        if self.renderer_policy_id != "default_evidence_prompt":
            raise ValueError("shape policy is paired with default_evidence_prompt")
        if self.renderer_policy_version != 3:
            raise ValueError("unsupported renderer_policy_version")

    def commit(self, inputs: PromptShapeInputs) -> str:
        if type(inputs) is not PromptShapeInputs:
            raise TypeError("inputs must be exact PromptShapeInputs")
        PromptShapeInputs.__post_init__(inputs)
        record = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "renderer_policy_id": self.renderer_policy_id,
            "renderer_policy_version": self.renderer_policy_version,
            "treatment_payload": {
                "kind": "selected_insight_records",
                "committed_property": "cardinality_only",
                "cardinality": inputs.selected_insight_count,
            },
            "non_treatment_inputs": inputs.to_record(),
        }
        return hashlib.sha256(_COMMITMENT_DOMAIN + _canonical_json(record)).hexdigest()


def with_selected_insight_count(
    inputs: PromptShapeInputs,
    selected_insight_count: int,
) -> PromptShapeInputs:
    """Return an immutable cardinality variant without exposing treatment data."""

    if type(inputs) is not PromptShapeInputs:
        raise TypeError("inputs must be exact PromptShapeInputs")
    return replace(inputs, selected_insight_count=selected_insight_count)


__all__ = [
    "DefaultEvidencePromptShapePolicyV1",
    "DefaultEvidencePromptShapePolicyV2",
    "DefaultEvidencePromptShapePolicyV3",
    "MatchedPromptStructureReceipt",
    "PromptShapeCommitmentPolicy",
    "PromptShapeInputs",
    "rendered_prompt_structure",
    "seal_matched_prompt_structure",
    "with_selected_insight_count",
]
