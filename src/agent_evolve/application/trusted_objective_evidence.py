"""Auditable binary64 objective evidence with LLM-readable decimal values.

Persistent AgentEvolve receipts use ``float.hex`` strings because they are
exact and replayable.  Those strings are a poor cognitive interface: an LLM
can easily read a hexadecimal significand as an ordinary decimal number.
This codec preserves the exact machine representation while providing a
separate JSON numeric field and round-trip decimal text for reasoning.

The codec is objective-, workload-, model-, provider-, and prompt-neutral.
Adapters decide which already-observed values are disclosed; this module only
prevents their representation from changing their meaning.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Mapping

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)


TRUSTED_OBJECTIVE_EVIDENCE_CODEC_ID = (
    "binary64_decimal_objective_evidence"
)
TRUSTED_OBJECTIVE_EVIDENCE_CODEC_VERSION = 1
TRUSTED_OBJECTIVE_EVIDENCE_CODEC_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:binary64-decimal-objective-evidence:v1;"
    b"reasoning-value=json-finite-number;"
    b"roundtrip-text=17-significant-decimal-digits;"
    b"audit-value=python-binary64-hex;"
    b"consistency=all-three-representations-must-roundtrip-identically;"
    b"hex-is-explicitly-machine-audit-only;"
    b"workload-objective-model-provider-branches=false"
).hexdigest()

_METRIC = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,255}$")


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    return value


def _metric_id(value: object) -> str:
    if type(value) is not str or _METRIC.fullmatch(value) is None:
        raise ValueError("metric_id must use the closed metric grammar")
    return value


@dataclass(frozen=True, slots=True)
class TrustedObjectiveEvidenceCodec:
    """Render known objective values without asking a model to parse hex."""

    codec_id: str = field(
        init=False,
        default=TRUSTED_OBJECTIVE_EVIDENCE_CODEC_ID,
    )
    codec_version: int = field(
        init=False,
        default=TRUSTED_OBJECTIVE_EVIDENCE_CODEC_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=TRUSTED_OBJECTIVE_EVIDENCE_CODEC_DEFINITION_SHA256,
    )

    def prompt_contract(self) -> FrozenJsonObject:
        value = freeze_json(
            {
                "codec_id": self.codec_id,
                "codec_version": self.codec_version,
                "definition_sha256": self.definition_sha256,
                "reasoning_field": "numeric_values",
                "exact_audit_field": "exact_binary64_hex",
                "instructions": [
                    (
                        "Use numeric_values for comparisons, arithmetic, "
                        "forecasting, and optimization reasoning."
                    ),
                    (
                        "decimal_text is a round-trip decimal rendering of "
                        "the same binary64 values."
                    ),
                    (
                        "exact_binary64_hex is machine-audit evidence only; "
                        "never interpret its digits as a decimal magnitude."
                    ),
                ],
                "representations_consistency_checked": True,
                "candidate_outcomes_disclosed_by_codec": False,
                "workload_objective_model_provider_branches": False,
            }
        )
        if type(value) is not FrozenJsonObject:
            raise TypeError("prompt contract must have an object root")
        return value

    def encode_point(
        self,
        objectives: Mapping[str, float],
    ) -> FrozenJsonObject:
        """Encode one already-authorized objective mapping."""

        if not isinstance(objectives, Mapping) or not objectives:
            raise ValueError("objectives must be a non-empty mapping")
        ordered: list[tuple[str, float]] = []
        for metric_id, raw_value in objectives.items():
            ordered.append(
                (
                    _metric_id(metric_id),
                    _finite_float(
                        raw_value,
                        name=f"objective[{metric_id!r}]",
                    ),
                )
            )
        ordered.sort(key=lambda value: value[0])
        if len({name for name, _ in ordered}) != len(ordered):
            raise ValueError("objectives repeat a metric")

        value = freeze_json(
            {
                "numeric_values": {
                    metric_id: metric_value
                    for metric_id, metric_value in ordered
                },
                "decimal_text": {
                    metric_id: format(metric_value, ".17g")
                    for metric_id, metric_value in ordered
                },
                "exact_binary64_hex": {
                    metric_id: metric_value.hex()
                    for metric_id, metric_value in ordered
                },
                "reasoning_representation": "ordinary_json_numbers",
                "exact_hex_is_machine_audit_only": True,
            }
        )
        if type(value) is not FrozenJsonObject:
            raise TypeError("objective evidence must have an object root")
        self.verify_point(value)
        return value

    def verify_point(
        self,
        evidence: FrozenJsonObject,
    ) -> tuple[tuple[str, float], ...]:
        """Authenticate all public representations and return exact values."""

        if (
            type(evidence) is not FrozenJsonObject
            or freeze_json(evidence) is not evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        record = thaw_json(evidence)
        numeric = record.get("numeric_values")
        decimal = record.get("decimal_text")
        exact = record.get("exact_binary64_hex")
        if not all(
            isinstance(value, dict)
            for value in (numeric, decimal, exact)
        ):
            raise ValueError("objective evidence representations are absent")
        assert isinstance(numeric, dict)
        assert isinstance(decimal, dict)
        assert isinstance(exact, dict)
        if not numeric or set(numeric) != set(decimal) or set(numeric) != set(exact):
            raise ValueError("objective evidence metric sets differ")
        verified: list[tuple[str, float]] = []
        for metric_id in sorted(numeric):
            name = _metric_id(metric_id)
            value = _finite_float(
                numeric[metric_id],
                name=f"numeric_values[{name!r}]",
            )
            text = decimal[metric_id]
            hex_value = exact[metric_id]
            if type(text) is not str or type(hex_value) is not str:
                raise TypeError("objective text/hex evidence must be strings")
            try:
                from_decimal = float(text)
                from_hex = float.fromhex(hex_value)
            except ValueError as error:
                raise ValueError(
                    "objective evidence contains an invalid representation"
                ) from error
            if (
                not math.isfinite(from_decimal)
                or not math.isfinite(from_hex)
                or from_decimal.hex() != value.hex()
                or from_hex.hex() != value.hex()
                or text != format(value, ".17g")
                or hex_value != value.hex()
            ):
                raise ValueError(
                    "objective evidence representations are inconsistent"
                )
            verified.append((name, value))
        if record.get("exact_hex_is_machine_audit_only") is not True:
            raise ValueError("objective evidence omitted the hex warning")
        return tuple(verified)


__all__ = [
    "TRUSTED_OBJECTIVE_EVIDENCE_CODEC_DEFINITION_SHA256",
    "TRUSTED_OBJECTIVE_EVIDENCE_CODEC_ID",
    "TRUSTED_OBJECTIVE_EVIDENCE_CODEC_VERSION",
    "TrustedObjectiveEvidenceCodec",
]
