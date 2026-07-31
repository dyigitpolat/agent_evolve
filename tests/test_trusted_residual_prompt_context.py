from __future__ import annotations

from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.integrations.pydantic_ai.trusted_residual_prompt_context import (
    TrustedResidualPromptContextProjection,
)


def test_prompt_projection_adds_numeric_twins_without_new_outcomes() -> None:
    source = freeze_json(
        {
            "archive_state": {
                "normalized_hypervolume_hex": float(0.8125).hex(),
                "front": [
                    {
                        "candidate_id": "candidate_a",
                        "objectives": {
                            "energy": float(1_024.5).hex(),
                            "latency": float(0.000_25).hex(),
                        },
                    }
                ],
            },
            "reachable_parents": [
                {
                    "candidate_id": "candidate_a",
                    "objectives": {
                        "energy": float(1_024.5).hex(),
                        "latency": float(0.000_25).hex(),
                    },
                }
            ],
            "raw_trace_memory": [
                {
                    "real_objectives": {
                        "energy": float(1_010.0).hex(),
                        "latency": float(0.000_24).hex(),
                    },
                    "forecast_probability_valid_hex": float(0.75).hex(),
                    "normalized_archive_gain_hex": float(0.01).hex(),
                    "positive_archive_contribution": True,
                }
            ],
        }
    )

    projected = thaw_json(
        TrustedResidualPromptContextProjection().project(source)
    )

    archive = projected["archive_state"]
    assert archive["normalized_hypervolume"] == 0.8125
    assert archive["front"][0]["objectives"] == {
        "energy": 1_024.5,
        "latency": 0.000_25,
    }
    assert (
        archive["front"][0]["objectives_evidence"][
            "exact_hex_is_machine_audit_only"
        ]
        is True
    )
    trace = projected["raw_trace_memory"][0]
    assert trace["forecast_probability_valid"] == 0.75
    assert trace["normalized_archive_gain"] == 0.01
    assert trace["real_objectives"]["energy"] == 1_010.0
    assert (
        projected["trusted_prompt_projection"][
            "new_candidate_outcomes_disclosed"
        ]
        is False
    )
    assert (
        projected["trusted_objective_evidence_contract"][
            "reasoning_field"
        ]
        == "numeric_values"
    )
