"""Project residual prompt evidence to an unambiguous numeric interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agent_evolve.application.trusted_objective_evidence import (
    TRUSTED_OBJECTIVE_EVIDENCE_CODEC_DEFINITION_SHA256,
    TrustedObjectiveEvidenceCodec,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


_OBJECTIVE_MAP_KEYS = frozenset({"objectives", "real_objectives"})


def _canonical_hex_float(value: object) -> float | None:
    if type(value) is not str:
        return None
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed.hex() != value:
        return None
    return parsed


def _hex_collection(value: object) -> object | None:
    scalar = _canonical_hex_float(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list) and value:
        parsed = [_canonical_hex_float(item) for item in value]
        if all(item is not None for item in parsed):
            return parsed
    if isinstance(value, dict) and value:
        parsed_map = {
            str(key): _canonical_hex_float(item)
            for key, item in value.items()
        }
        if all(item is not None for item in parsed_map.values()):
            return parsed_map
    return None


@dataclass(frozen=True, slots=True)
class TrustedResidualPromptContextProjection:
    """Add numeric twins for disclosed hex values without adding evidence."""

    codec: TrustedObjectiveEvidenceCodec = field(
        default_factory=TrustedObjectiveEvidenceCodec
    )

    def project(self, context: FrozenJsonObject) -> FrozenJsonObject:
        if (
            type(context) is not FrozenJsonObject
            or freeze_json(context) is not context
        ):
            raise TypeError("context must be an exact frozen object")
        source_sha256 = typed_json_sha256(context)
        conversion_count = 0
        objective_point_count = 0

        def visit(value: object) -> object:
            nonlocal conversion_count, objective_point_count
            if isinstance(value, list):
                return [visit(item) for item in value]
            if not isinstance(value, dict):
                return value
            rendered: dict[str, object] = {}
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if (
                    key in _OBJECTIVE_MAP_KEYS
                    and isinstance(raw_value, dict)
                    and raw_value
                ):
                    point: dict[str, float] = {}
                    for metric_id, metric_value in raw_value.items():
                        parsed = _canonical_hex_float(metric_value)
                        if parsed is None:
                            if (
                                type(metric_value) is not float
                                or not math.isfinite(metric_value)
                            ):
                                raise ValueError(
                                    "objective prompt values must be finite "
                                    "floats or canonical binary64 hex"
                                )
                            parsed = metric_value
                        point[str(metric_id)] = parsed
                    evidence = thaw_json(self.codec.encode_point(point))
                    rendered[key] = evidence["numeric_values"]
                    rendered[f"{key}_evidence"] = evidence
                    objective_point_count += 1
                    continue
                converted = visit(raw_value)
                rendered[key] = converted
                if key.endswith("_hex"):
                    numeric = _hex_collection(raw_value)
                    if numeric is not None:
                        numeric_key = key[:-4]
                        if numeric_key in value or numeric_key in rendered:
                            raise ValueError(
                                "hex prompt evidence collides with an "
                                "existing numeric field"
                            )
                        rendered[numeric_key] = numeric
                        conversion_count += 1
            return rendered

        source = thaw_json(context)
        projected = visit(source)
        if not isinstance(projected, dict):
            raise TypeError("projected context must have an object root")
        projected["trusted_objective_evidence_contract"] = thaw_json(
            self.codec.prompt_contract()
        )
        projected["trusted_prompt_projection"] = {
            "schema_version": 1,
            "source_context_sha256": source_sha256,
            "codec_definition_sha256": (
                TRUSTED_OBJECTIVE_EVIDENCE_CODEC_DEFINITION_SHA256
            ),
            "hex_scalar_or_collection_conversion_count": conversion_count,
            "objective_point_conversion_count": objective_point_count,
            "new_candidate_outcomes_disclosed": False,
            "workload_objective_model_provider_branches": False,
        }
        result = freeze_json(projected)
        if type(result) is not FrozenJsonObject:
            raise TypeError("projected context must have an object root")
        return result


__all__ = ["TrustedResidualPromptContextProjection"]
