"""Prospectively frozen affine utility for Timeloop-v2 campaigns.

The utility definition is workload configuration, not orchestration policy.
Keeping it beside the benchmark lets provider-free preparations, treatments,
and matched controls bind the same three-objective endpoint without importing
one another's launch scripts.
"""

from __future__ import annotations

from agent_evolve.policies.reward.affine_hypervolume import AffineObjectiveAxis
from agent_evolve.policies.reward.affine_hypervolume_3d import (
    AffineHypervolume3DSpec,
)


def timeloop_v2_affine_hypervolume_spec() -> AffineHypervolume3DSpec:
    return AffineHypervolume3DSpec(
        axes=(
            AffineObjectiveAxis("energy_joules", "min", 0.0, 0.75),
            AffineObjectiveAxis("latency_seconds", "min", 0.0, 2.50),
            AffineObjectiveAxis("area_square_meters", "min", 0.0, 5.00e-7),
        ),
        reference_provenance=(
            "prospectively refrozen after the prior 1.20e-7 area assumption "
            "was falsified, using three authenticated real-Docker qualification "
            "runs plus a pre-experiment authenticated historical envelope: "
            "pessimistic energy=0.5694582627867033 J; historical latency="
            "2.147483648 s; componentwise-maximum architecture area="
            "4.6319712e-7 m^2; reference=(0.75,2.50,5.00e-7)"
        ),
    )


__all__ = ["timeloop_v2_affine_hypervolume_spec"]
