"""Reward policies for auditable evolutionary credit assignment."""

from .frozen_archive import (
    FrozenArchiveMarginalHypervolumeReward,
    FrozenArchiveRewardRecord,
    FrozenArchiveSnapshot2D,
    hypervolume_2d,
)
from .frozen_wave_archive import (
    FrozenArchiveJointWaveHypervolumeReward,
    FrozenArchiveWaveRewardRecord,
    FrozenArchiveWaveSnapshot2D,
    WaveRewardCandidate,
)
from .affine_hypervolume import AffineObjectiveAxis
from .affine_hypervolume_3d import (
    AffineHypervolume3DSpec,
    AffineHypervolumeArchiveUtility3D,
    AffineHypervolumeSnapshot3D,
    audit_affine_reference_envelope_3d,
    hypervolume_3d,
)
from .contextual_marginal_utility import (
    CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256,
    CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID,
    CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION,
    FixedReferenceContextualMarginalUtilityProjector,
    MarginalUtilitySnapshot,
    ReplayableArchiveUtility,
)

__all__ = [
    "FrozenArchiveMarginalHypervolumeReward",
    "FrozenArchiveRewardRecord",
    "FrozenArchiveSnapshot2D",
    "FrozenArchiveJointWaveHypervolumeReward",
    "FrozenArchiveWaveRewardRecord",
    "FrozenArchiveWaveSnapshot2D",
    "WaveRewardCandidate",
    "hypervolume_2d",
    "AffineObjectiveAxis",
    "AffineHypervolume3DSpec",
    "AffineHypervolumeArchiveUtility3D",
    "AffineHypervolumeSnapshot3D",
    "audit_affine_reference_envelope_3d",
    "hypervolume_3d",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_DEFINITION_SHA256",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_ID",
    "CONTEXTUAL_MARGINAL_UTILITY_PROJECTOR_VERSION",
    "FixedReferenceContextualMarginalUtilityProjector",
    "MarginalUtilitySnapshot",
    "ReplayableArchiveUtility",
]
