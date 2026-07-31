"""Strict JSON codecs for the finite-acquisition ask/tell boundary."""

from __future__ import annotations

from typing import Any

from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionDecision,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionRequest,
    FiniteAcquisitionSelection,
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


def finite_acquisition_request_from_record(
    value: object,
) -> FiniteAcquisitionRequest:
    """Reconstruct and authenticate a request from its canonical JSON record."""

    record = _object(
        value,
        name="finite acquisition request",
        keys={
            "schema_version",
            "campaign_scope_sha256",
            "cutoff_index",
            "batch_size",
            "seed",
            "objectives",
            "observations",
            "candidates",
            "request_sha256",
        },
    )
    if record["schema_version"] != 1:
        raise ValueError("unsupported finite acquisition request schema")
    objectives = []
    for index, raw in enumerate(_list(record["objectives"], name="objectives")):
        row = _object(
            raw,
            name=f"objective[{index}]",
            keys={"metric_id", "sense", "ideal_hex", "reference_hex"},
        )
        objectives.append(
            FiniteAcquisitionObjective(
                metric_id=_string(row["metric_id"], name="metric_id"),
                sense=_string(row["sense"], name="sense"),
                ideal=_hex_float(row["ideal_hex"], name="ideal_hex"),
                reference=_hex_float(row["reference_hex"], name="reference_hex"),
            )
        )
    observations = []
    for index, raw in enumerate(
        _list(record["observations"], name="observations")
    ):
        row = _object(
            raw,
            name=f"observation[{index}]",
            keys={
                "candidate_id",
                "configuration_sha256",
                "features_hex",
                "objectives",
            },
        )
        objective_values = []
        for metric_index, raw_metric in enumerate(
            _list(row["objectives"], name="observation objectives")
        ):
            metric = _object(
                raw_metric,
                name=f"observation[{index}].objective[{metric_index}]",
                keys={"metric_id", "value_hex"},
            )
            objective_values.append(
                (
                    _string(metric["metric_id"], name="metric_id"),
                    _hex_float(metric["value_hex"], name="value_hex"),
                )
            )
        observations.append(
            FiniteAcquisitionObservation(
                candidate_id=_string(row["candidate_id"], name="candidate_id"),
                configuration_sha256=_string(
                    row["configuration_sha256"], name="configuration_sha256"
                ),
                features=tuple(
                    _hex_float(item, name="feature")
                    for item in _list(row["features_hex"], name="features_hex")
                ),
                objectives=tuple(objective_values),
            )
        )
    candidates = []
    for index, raw in enumerate(_list(record["candidates"], name="candidates")):
        row = _object(
            raw,
            name=f"candidate[{index}]",
            keys={"candidate_id", "configuration_sha256", "features_hex"},
        )
        candidates.append(
            FiniteAcquisitionCandidate(
                candidate_id=_string(row["candidate_id"], name="candidate_id"),
                configuration_sha256=_string(
                    row["configuration_sha256"], name="configuration_sha256"
                ),
                features=tuple(
                    _hex_float(item, name="feature")
                    for item in _list(row["features_hex"], name="features_hex")
                ),
            )
        )
    request = FiniteAcquisitionRequest(
        campaign_scope_sha256=_string(
            record["campaign_scope_sha256"], name="campaign_scope_sha256"
        ),
        cutoff_index=_integer(record["cutoff_index"], name="cutoff_index"),
        batch_size=_integer(record["batch_size"], name="batch_size"),
        seed=_integer(record["seed"], name="seed"),
        objectives=tuple(objectives),
        observations=tuple(observations),
        candidates=tuple(candidates),
    )
    if request.request_sha256 != _string(
        record["request_sha256"], name="request_sha256"
    ):
        raise ValueError("finite acquisition request authentication failed")
    return request


def finite_acquisition_decision_from_record(
    value: object,
) -> FiniteAcquisitionDecision:
    """Reconstruct and authenticate a decision from its canonical JSON record."""

    record = _object(
        value,
        name="finite acquisition decision",
        keys={
            "schema_version",
            "request_sha256",
            "policy",
            "selected",
            "diagnostics",
            "decision_sha256",
        },
    )
    if record["schema_version"] != 1:
        raise ValueError("unsupported finite acquisition decision schema")
    policy = _object(
        record["policy"],
        name="policy",
        keys={"policy_id", "policy_version", "definition_sha256"},
    )
    selected = []
    for index, raw in enumerate(_list(record["selected"], name="selected")):
        row = _object(
            raw,
            name=f"selection[{index}]",
            keys={
                "candidate_id",
                "configuration_sha256",
                "acquisition_value_hex",
            },
        )
        selected.append(
            FiniteAcquisitionSelection(
                candidate_id=_string(row["candidate_id"], name="candidate_id"),
                configuration_sha256=_string(
                    row["configuration_sha256"], name="configuration_sha256"
                ),
                acquisition_value=_hex_float(
                    row["acquisition_value_hex"], name="acquisition_value_hex"
                ),
            )
        )
    diagnostics = []
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
    decision = FiniteAcquisitionDecision(
        request_sha256=_string(record["request_sha256"], name="request_sha256"),
        policy_id=_string(policy["policy_id"], name="policy_id"),
        policy_version=_integer(policy["policy_version"], name="policy_version"),
        policy_definition_sha256=_string(
            policy["definition_sha256"], name="definition_sha256"
        ),
        selected=tuple(selected),
        diagnostics=tuple(diagnostics),
    )
    if decision.decision_sha256 != _string(
        record["decision_sha256"], name="decision_sha256"
    ):
        raise ValueError("finite acquisition decision authentication failed")
    return decision


__all__ = [
    "finite_acquisition_decision_from_record",
    "finite_acquisition_request_from_record",
]
