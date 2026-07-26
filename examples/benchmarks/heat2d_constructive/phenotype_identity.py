"""Evaluator-semantic identity for constructive Heat2D candidates."""

from __future__ import annotations

from dataclasses import dataclass

from agent_evolve.agentic import PhenotypeIdentity

from .artifact_boundary import decode_mapping
from .candidate import normalize_candidate


@dataclass(frozen=True, slots=True)
class Heat2DPhenotypeIdentityPolicy:
    """Identify the exact projected material field consumed by direct-v3."""

    resolution: int = 1001
    policy_version: int = 1

    def __post_init__(self) -> None:
        if type(self.resolution) is not int or self.resolution < 3:
            raise ValueError("phenotype identity resolution must be at least three")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("phenotype identity policy_version must be positive")

    @property
    def policy_id(self) -> str:
        return f"heat2d_constructive_field_r{self.resolution}"

    def identify(self, configuration: object) -> PhenotypeIdentity:
        candidate = normalize_candidate(configuration)
        decoded = decode_mapping(
            candidate.decoder_mapping(),
            resolution=self.resolution,
        )
        # The qualified decoder's phenotype digest binds its public
        # representation spec, resolution, material fraction, and exact dense
        # field hash.  Use it directly as the policy's value digest.
        return PhenotypeIdentity(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            value_sha256=decoded.phenotype_sha256,
        )


__all__ = ["Heat2DPhenotypeIdentityPolicy"]
