"""BOiLS sequence optimization on a pinned Berkeley ABC evaluator.

The package separates the candidate/action contract from the subprocess
boundary so experiments can mutate typed sequences without exposing arbitrary
ABC commands.
"""

from .actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
    CandidateConfig,
    canonical_config_bytes,
    config_sha256,
    expand_abc_commands,
    normalize_candidate,
)
from .evaluator import (
    ABC_SOURCE_COMMIT,
    CURRENT_ABC_SHA256,
    EPFL_SOURCE_COMMIT,
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    CircuitEvaluation,
    CircuitSpec,
    ProvenanceMismatchError,
)
from .variation_catalog import (
    ACTION_DEFINITION_SHA256,
    ACTION_FAMILIES,
    CATALOG_SCHEMA_ID,
    CATALOG_SOURCE_SHA256,
    BoilsAtomicVariationCatalog,
)
from .finite_variation_catalog import (
    FINITE_CATALOG_DEFINITION_SHA256,
    FINITE_CATALOG_ID,
    FINITE_CATALOG_SCHEMA_ID,
    FINITE_CATALOG_VERSION,
    BoilsFiniteVariationCatalog,
)
from .campaign_workload import (
    EVIDENCE_PROJECTION_DEFINITION_SHA256,
    WORKLOAD_DEFINITION_SHA256,
    compose_boils_campaign_workload,
)

__all__ = [
    "ABC_SOURCE_COMMIT",
    "ACTION_COMMANDS",
    "ACTION_DEFINITION_SHA256",
    "ACTION_FAMILIES",
    "ACTION_IDS",
    "CURRENT_ABC_SHA256",
    "DEFAULT_ACTION_SEQUENCE",
    "EPFL_SOURCE_COMMIT",
    "EVIDENCE_PROJECTION_DEFINITION_SHA256",
    "FINITE_CATALOG_DEFINITION_SHA256",
    "FINITE_CATALOG_ID",
    "FINITE_CATALOG_SCHEMA_ID",
    "FINITE_CATALOG_VERSION",
    "SEQUENCE_LENGTH",
    "WORKLOAD_DEFINITION_SHA256",
    "AbcEvaluationError",
    "AbcEvaluatorSettings",
    "BoilsAbcEvaluator",
    "BoilsEvaluation",
    "BoilsFiniteVariationCatalog",
    "CandidateConfig",
    "CATALOG_SCHEMA_ID",
    "CATALOG_SOURCE_SHA256",
    "CircuitEvaluation",
    "CircuitSpec",
    "BoilsAtomicVariationCatalog",
    "ProvenanceMismatchError",
    "canonical_config_bytes",
    "compose_boils_campaign_workload",
    "config_sha256",
    "expand_abc_commands",
    "normalize_candidate",
]
