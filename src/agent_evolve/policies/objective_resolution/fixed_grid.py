"""Workload-neutral fixed-decimal-grid objective resolution.

Evaluators retain authority over raw physical measurements.  This policy only
defines the coarser values on which optimization decisions are made.  Each
metric has a prospectively declared decimal origin, positive decimal quantum,
and nearest-neighbour tie law.  Candidate configuration and objective goal are
deliberately outside the transform, so neither can silently change resolution.

The implementation uses exact integer arithmetic.  It first interprets a
finite Python float through its shortest round-trip decimal text, then rounds
the resulting exact rational grid index.  It therefore does not depend on the
process-global :mod:`decimal` context.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionRequest,
    ObjectiveResolutionResult,
)


FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_ID = "fixed_decimal_grid_objectives"
FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_VERSION = 1

_DEFINITION_DOMAIN = b"agent-evolve:fixed-decimal-grid-objectives:def:v1\x00"
_FLOAT_DECIMAL_CONVERSION = "python_float_shortest_round_trip_decimal_text_v1"
_CONFIGURATION_DEPENDENCE = "none"
_MAX_DECIMAL_DIGITS = 128
_MAX_DECIMAL_ABS_EXPONENT = 1_024
_MAX_METRIC_ID_BYTES = 512


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _metric_sort_key(metric_id: str) -> bytes:
    return metric_id.encode("utf-8", errors="strict")


def _require_metric_id(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("metric_id must be a non-empty canonical string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("metric_id must be valid UTF-8 text") from exc
    if len(encoded) > _MAX_METRIC_ID_BYTES:
        raise ValueError("metric_id exceeds the bounded identifier policy")
    return value


def _require_decimal(value: object, *, name: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError(f"{name} must be an exact finite Decimal")
    decimal_tuple = value.as_tuple()
    if (
        len(decimal_tuple.digits) > _MAX_DECIMAL_DIGITS
        or type(decimal_tuple.exponent) is not int
        or not (
            -_MAX_DECIMAL_ABS_EXPONENT
            <= decimal_tuple.exponent
            <= _MAX_DECIMAL_ABS_EXPONENT
        )
    ):
        raise ValueError(f"{name} exceeds the bounded decimal policy")
    return value


def _decimal_components(value: Decimal) -> tuple[int, int]:
    """Return exact ``coefficient, exponent`` with trailing zeros removed."""

    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    if type(exponent) is not int:  # already rejected at the public boundary
        raise TypeError("finite Decimal exponent must be an exact integer")
    coefficient = 0
    for digit in decimal_tuple.digits:
        coefficient = coefficient * 10 + digit
    if decimal_tuple.sign:
        coefficient = -coefficient
    if coefficient == 0:
        return 0, 0
    while coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    return coefficient, exponent


def _canonical_decimal_text(value: Decimal) -> str:
    coefficient, exponent = _decimal_components(value)
    if coefficient == 0:
        return "0"
    sign = "-" if coefficient < 0 else ""
    digits = str(abs(coefficient))
    if exponent >= 0:
        return sign + digits + "0" * exponent
    point = len(digits) + exponent
    if point > 0:
        return sign + digits[:point] + "." + digits[point:]
    return sign + "0." + "0" * (-point) + digits


def _decimal_from_components(coefficient: int, exponent: int) -> Decimal:
    if coefficient == 0:
        return Decimal(0)
    sign = 1 if coefficient < 0 else 0
    digits = tuple(int(digit) for digit in str(abs(coefficient)))
    return Decimal((sign, digits, exponent))


def _common_integer_components(
    *values: Decimal,
) -> tuple[tuple[int, ...], int]:
    components = tuple(_decimal_components(value) for value in values)
    common_exponent = min(exponent for _, exponent in components)
    integers = tuple(
        coefficient * 10 ** (exponent - common_exponent)
        for coefficient, exponent in components
    )
    return integers, common_exponent


class FixedGridRoundingLaw(str, Enum):
    """Closed nearest-grid tie semantics.

    Both laws choose the nearest grid point away from exact ties.  At an exact
    half-grid boundary, ``NEAREST_TIES_TO_EVEN`` chooses an even integer grid
    index while ``NEAREST_TIES_AWAY_FROM_ZERO`` increases its magnitude.
    """

    NEAREST_TIES_TO_EVEN = "nearest_ties_to_even"
    NEAREST_TIES_AWAY_FROM_ZERO = "nearest_ties_away_from_zero"


@dataclass(frozen=True, slots=True)
class FixedGridMetricSpec:
    """One metric's exact benchmark-declared decision grid."""

    metric_id: str
    decimal_origin: Decimal
    decimal_quantum: Decimal
    rounding_law: FixedGridRoundingLaw

    def __post_init__(self) -> None:
        _require_metric_id(self.metric_id)
        _require_decimal(self.decimal_origin, name="decimal_origin")
        quantum = _require_decimal(self.decimal_quantum, name="decimal_quantum")
        if quantum <= 0:
            raise ValueError("decimal_quantum must be strictly positive")
        if type(self.rounding_law) is not FixedGridRoundingLaw:
            raise TypeError("rounding_law must be an exact FixedGridRoundingLaw")

    def revalidate(self) -> None:
        if type(self) is not FixedGridMetricSpec:
            raise TypeError("metric grid must be an exact FixedGridMetricSpec")
        FixedGridMetricSpec.__post_init__(self)

    def to_record(self) -> dict[str, str]:
        self.revalidate()
        return {
            "metric_id": self.metric_id,
            "decimal_origin": _canonical_decimal_text(self.decimal_origin),
            "decimal_quantum": _canonical_decimal_text(self.decimal_quantum),
            "rounding_law": self.rounding_law.value,
        }


def _rounded_grid_index(
    *,
    value: Decimal,
    spec: FixedGridMetricSpec,
) -> int:
    integers, _ = _common_integer_components(
        value,
        spec.decimal_origin,
        spec.decimal_quantum,
    )
    value_integer, origin_integer, quantum_integer = integers
    numerator = value_integer - origin_integer
    sign = -1 if numerator < 0 else 1
    magnitude, remainder = divmod(abs(numerator), quantum_integer)
    twice_remainder = 2 * remainder
    if twice_remainder > quantum_integer:
        magnitude += 1
    elif twice_remainder == quantum_integer:
        if spec.rounding_law is FixedGridRoundingLaw.NEAREST_TIES_AWAY_FROM_ZERO:
            magnitude += 1
        elif magnitude % 2 == 1:
            magnitude += 1
    return sign * magnitude


def _grid_value(spec: FixedGridMetricSpec, grid_index: int) -> Decimal:
    integers, common_exponent = _common_integer_components(
        spec.decimal_origin,
        spec.decimal_quantum,
    )
    origin_integer, quantum_integer = integers
    return _decimal_from_components(
        origin_integer + grid_index * quantum_integer,
        common_exponent,
    )


@dataclass(frozen=True, slots=True)
class FixedGridObjectiveResolution:
    """Resolve every declared objective onto a fixed decimal measurement grid.

    ``metric_specs`` use UTF-8 bytewise metric order.  At resolution time they
    must cover exactly the request's objective IDs.  Decision values always
    follow the objective declaration order required by the port.
    """

    metric_specs: tuple[FixedGridMetricSpec, ...]
    policy_id: str = field(
        init=False,
        default=FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_ID,
    )
    policy_version: int = field(
        init=False,
        default=FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_VERSION,
    )
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.metric_specs) is not tuple or not self.metric_specs:
            raise ValueError("metric_specs must be a non-empty exact tuple")
        if any(type(spec) is not FixedGridMetricSpec for spec in self.metric_specs):
            raise TypeError("metric_specs must contain exact FixedGridMetricSpec values")
        for spec in self.metric_specs:
            spec.revalidate()
        metric_ids = tuple(spec.metric_id for spec in self.metric_specs)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric_specs cannot repeat a metric_id")
        if metric_ids != tuple(sorted(metric_ids, key=_metric_sort_key)):
            raise ValueError("metric_specs must use canonical UTF-8 metric order")
        definition = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "float_decimal_conversion": _FLOAT_DECIMAL_CONVERSION,
            "configuration_dependence": _CONFIGURATION_DEPENDENCE,
            "metric_specs": [spec.to_record() for spec in self.metric_specs],
        }
        computed = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 differs from the fixed-grid policy")
        object.__setattr__(self, "definition_sha256", computed)

    def revalidate(self) -> None:
        if type(self) is not FixedGridObjectiveResolution:
            raise TypeError("policy must be an exact FixedGridObjectiveResolution")
        FixedGridObjectiveResolution.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "float_decimal_conversion": _FLOAT_DECIMAL_CONVERSION,
            "configuration_dependence": _CONFIGURATION_DEPENDENCE,
            "metric_specs": [spec.to_record() for spec in self.metric_specs],
        }

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult:
        self.revalidate()
        if type(request) is not ObjectiveResolutionRequest:
            raise TypeError("request must be an exact ObjectiveResolutionRequest")
        ObjectiveResolutionRequest.__post_init__(request)
        requested_ids = tuple(objective.name for objective in request.objectives)
        configured_ids = tuple(spec.metric_id for spec in self.metric_specs)
        requested_set = set(requested_ids)
        configured_set = set(configured_ids)
        if requested_set != configured_set:
            missing = sorted(requested_set - configured_set, key=_metric_sort_key)
            extra = sorted(configured_set - requested_set, key=_metric_sort_key)
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise ValueError(
                "metric_specs must cover every objective exactly ("
                + "; ".join(details)
                + ")"
            )
        spec_by_metric = {spec.metric_id: spec for spec in self.metric_specs}
        raw_by_metric = dict(request.raw_objectives)
        decision_values: list[tuple[str, float]] = []
        metric_evidence: list[dict[str, object]] = []
        for metric_id in requested_ids:
            spec = spec_by_metric[metric_id]
            raw_value = raw_by_metric[metric_id]
            input_decimal = Decimal(str(raw_value))
            grid_index = _rounded_grid_index(value=input_decimal, spec=spec)
            resolved_decimal = _grid_value(spec, grid_index)
            decision_value = float(resolved_decimal)
            if not math.isfinite(decision_value):
                raise ValueError(
                    f"resolved grid value for {metric_id!r} is not a finite float"
                )
            if resolved_decimal.is_zero():
                decision_value = 0.0
            decision_values.append((metric_id, decision_value))
            metric_evidence.append(
                {
                    **spec.to_record(),
                    "raw_value_hex": raw_value.hex(),
                    "input_decimal": _canonical_decimal_text(input_decimal),
                    "grid_index": str(grid_index),
                    "resolved_decimal": _canonical_decimal_text(resolved_decimal),
                    "decision_value_hex": decision_value.hex(),
                    "changed": raw_value.hex() != decision_value.hex(),
                }
            )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.definition_sha256,
                "float_decimal_conversion": _FLOAT_DECIMAL_CONVERSION,
                "configuration_dependence": _CONFIGURATION_DEPENDENCE,
                "metrics": metric_evidence,
            }
        )
        if type(evidence) is not FrozenJsonObject:
            raise TypeError("fixed-grid evidence must freeze to an object")
        return ObjectiveResolutionResult(
            request_sha256=request.request_sha256,
            decision_objectives=tuple(decision_values),
            evidence=evidence,
        )


__all__ = [
    "FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_ID",
    "FIXED_GRID_OBJECTIVE_RESOLUTION_POLICY_VERSION",
    "FixedGridMetricSpec",
    "FixedGridObjectiveResolution",
    "FixedGridRoundingLaw",
]
