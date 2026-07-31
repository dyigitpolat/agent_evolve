from __future__ import annotations

import json

import pytest

from agent_evolve.application.trusted_objective_evidence import (
    TrustedObjectiveEvidenceCodec,
)
from agent_evolve.domain.typed_json import freeze_json, thaw_json


def test_objective_evidence_is_human_numeric_and_machine_exact() -> None:
    codec = TrustedObjectiveEvidenceCodec()
    evidence = codec.encode_point(
        {
            "energy": float(12_345.625),
            "latency": float(0.000_031_25),
        }
    )

    verified = dict(codec.verify_point(evidence))
    assert verified == {
        "energy": 12_345.625,
        "latency": 0.000_031_25,
    }
    record = thaw_json(evidence)
    assert record["numeric_values"] == verified
    assert record["decimal_text"]["energy"] == "12345.625"
    assert record["exact_binary64_hex"]["energy"] == float(12_345.625).hex()
    rendered = json.dumps(record, sort_keys=True)
    assert '"energy": 12345.625' in rendered
    assert '"latency": 3.125e-05' in rendered
    contract = thaw_json(codec.prompt_contract())
    assert contract["reasoning_field"] == "numeric_values"
    assert contract["exact_audit_field"] == "exact_binary64_hex"
    assert "never interpret" in contract["instructions"][2]


def test_objective_evidence_rejects_representation_disagreement() -> None:
    codec = TrustedObjectiveEvidenceCodec()
    inconsistent = freeze_json(
        {
            "numeric_values": {"cost": 10.0},
            "decimal_text": {"cost": "11"},
            "exact_binary64_hex": {"cost": float(10.0).hex()},
            "reasoning_representation": "ordinary_json_numbers",
            "exact_hex_is_machine_audit_only": True,
        }
    )

    with pytest.raises(ValueError, match="representations are inconsistent"):
        codec.verify_point(inconsistent)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_objective_evidence_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(TypeError, match="finite exact float"):
        TrustedObjectiveEvidenceCodec().encode_point({"cost": value})
