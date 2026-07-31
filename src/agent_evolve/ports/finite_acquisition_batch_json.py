"""Strict JSON codecs for finite acquisition slate scoring."""

from __future__ import annotations

from typing import Any

from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlate,
    FiniteAcquisitionSlateScore,
)
from agent_evolve.ports.finite_acquisition_json import (
    finite_acquisition_request_from_record,
)


def _object(value: object, *, name: str, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} must be an exact JSON object with canonical fields")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return value


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact JSON string")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact JSON integer")
    return value


def _hex_float(value: object, *, name: str) -> float:
    text = _string(value, name=name)
    try:
        parsed = float.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical hexadecimal float") from error
    if parsed.hex() != text:
        raise ValueError(f"{name} must use Python's canonical float.hex form")
    return parsed


def _slate_from_record(value: object, *, name: str) -> FiniteAcquisitionSlate:
    record = _object(value, name=name, keys={"candidate_ids"})
    return FiniteAcquisitionSlate(
        tuple(
            _string(item, name=f"{name}.candidate_id")
            for item in _list(record["candidate_ids"], name=f"{name}.candidate_ids")
        )
    )


def finite_acquisition_batch_score_request_from_record(
    value: object,
) -> FiniteAcquisitionBatchScoreRequest:
    record = _object(
        value,
        name="finite acquisition batch-score request",
        keys={"schema_version", "base_request", "slates", "request_sha256"},
    )
    if record["schema_version"] != 1:
        raise ValueError("unsupported finite acquisition batch-score request schema")
    base = finite_acquisition_request_from_record(record["base_request"])
    request = FiniteAcquisitionBatchScoreRequest(
        campaign_scope_sha256=base.campaign_scope_sha256,
        cutoff_index=base.cutoff_index,
        seed=base.seed,
        objectives=base.objectives,
        observations=base.observations,
        candidates=base.candidates,
        slates=tuple(
            _slate_from_record(raw, name=f"slate[{index}]")
            for index, raw in enumerate(_list(record["slates"], name="slates"))
        ),
    )
    if request.request_sha256 != _string(
        record["request_sha256"], name="request_sha256"
    ):
        raise ValueError("finite acquisition batch-score request authentication failed")
    return request


def finite_acquisition_batch_score_decision_from_record(
    value: object,
) -> FiniteAcquisitionBatchScoreDecision:
    record = _object(
        value,
        name="finite acquisition batch-score decision",
        keys={
            "schema_version",
            "request_sha256",
            "policy",
            "scores",
            "diagnostics",
            "decision_sha256",
        },
    )
    if record["schema_version"] != 1:
        raise ValueError("unsupported finite acquisition batch-score decision schema")
    policy = _object(
        record["policy"],
        name="policy",
        keys={"policy_id", "policy_version", "definition_sha256"},
    )
    scores: list[FiniteAcquisitionSlateScore] = []
    for index, raw in enumerate(_list(record["scores"], name="scores")):
        row = _object(
            raw,
            name=f"score[{index}]",
            keys={"slate", "log_acquisition_value_hex"},
        )
        scores.append(
            FiniteAcquisitionSlateScore(
                slate=_slate_from_record(row["slate"], name=f"score[{index}].slate"),
                log_acquisition_value=_hex_float(
                    row["log_acquisition_value_hex"],
                    name=f"score[{index}].log_acquisition_value_hex",
                ),
            )
        )
    diagnostics: list[tuple[str, str]] = []
    for index, raw in enumerate(_list(record["diagnostics"], name="diagnostics")):
        row = _object(
            raw,
            name=f"diagnostic[{index}]",
            keys={"key", "value"},
        )
        diagnostics.append(
            (
                _string(row["key"], name="diagnostic key"),
                _string(row["value"], name="diagnostic value"),
            )
        )
    decision = FiniteAcquisitionBatchScoreDecision(
        request_sha256=_string(record["request_sha256"], name="request_sha256"),
        policy_id=_string(policy["policy_id"], name="policy_id"),
        policy_version=_integer(policy["policy_version"], name="policy_version"),
        policy_definition_sha256=_string(
            policy["definition_sha256"], name="definition_sha256"
        ),
        scores=tuple(scores),
        diagnostics=tuple(diagnostics),
    )
    if decision.decision_sha256 != _string(
        record["decision_sha256"], name="decision_sha256"
    ):
        raise ValueError("finite acquisition batch-score decision authentication failed")
    return decision


__all__ = [
    "finite_acquisition_batch_score_decision_from_record",
    "finite_acquisition_batch_score_request_from_record",
]
