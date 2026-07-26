"""Immutable v2 Timeloop architecture/mapspace-policy boundary.

Version 1 remains the architecture-only developmental adapter. This package
contains the typed candidate/compiler boundary, frozen outcome-blind network
panels, pinned runtime evaluator, and workload-owned campaign composition for
the v2 task.
"""

from .candidate import (
    DEFAULT_CANDIDATE,
    CandidateConfig,
    MappingPolicyBlock,
    candidate_sha256,
    normalize_candidate,
)
from .campaign_workload import (
    EVIDENCE_PROJECTION_DEFINITION_SHA256,
    WORKLOAD_DEFINITION_SHA256,
    compose_timeloop_v2_campaign_workload,
)
from .compiler import (
    COMPILER_DEFINITION_SHA256,
    CompiledPlan,
    CompilationResult,
    TimeloopV2Compiler,
    audit_compiler_injectivity,
    compiled_plan_sha256,
)
from .evaluator import (
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2ContractError,
    TimeloopV2DockerEvaluator,
    TimeloopV2Evaluation,
    TimeloopV2EvaluatorPort,
    TimeloopV2InfeasibleEvaluation,
    TimeloopV2Settings,
    TimeloopV2StaticInfeasibleEvaluation,
    TimeloopV2StaticMapspaceWitness,
    analyze_static_mapspace_feasibility,
    build_evaluation_bundle,
)
from .detailed_evaluation import (
    TIMELOOP_V2_DETAILED_EVALUATOR_ID,
    TIMELOOP_V2_DETAILED_EVALUATOR_VERSION,
    TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION,
    TimeloopV2DetailedEvaluationAdapter,
    TimeloopV2DetailedProjectionError,
    compose_timeloop_v2_detailed_benchmark,
    timeloop_v2_evaluator_context_record,
    timeloop_v2_evaluator_identity,
)
from .finite_variation_catalog import TimeloopV2FiniteVariationCatalog
from .frozen_panels import (
    FROZEN_EXTRACTION_RECEIPT_SHA256,
    FROZEN_NETWORK_PANELS,
    FROZEN_PANEL_SHA256,
    frozen_network_panel,
)
from .network_panel import (
    NETWORK_ASSETS,
    ConvLayerShape,
    LayerMedoid,
    NetworkLayerPanel,
    panel_sha256,
    verify_network_asset,
)
from .panel_extractor import (
    CLUSTERING_DEFINITION_SHA256,
    EXTRACTION_DEFINITION_SHA256,
    PanelExtractionReceipt,
    PanelExtractionResult,
    extract_network_panel,
    extraction_receipt_sha256,
)
from .runtime_manifest import (
    RUNTIME_TRANSLATOR_DEFINITION_SHA256,
    RuntimeLayerManifest,
    compile_runtime_layer_manifests,
    runtime_layer_manifest_sha256,
)

__all__ = [
    "COMPILER_DEFINITION_SHA256",
    "CLUSTERING_DEFINITION_SHA256",
    "DEFAULT_CANDIDATE",
    "EVIDENCE_PROJECTION_DEFINITION_SHA256",
    "EXTRACTION_DEFINITION_SHA256",
    "FROZEN_EXTRACTION_RECEIPT_SHA256",
    "FROZEN_NETWORK_PANELS",
    "FROZEN_PANEL_SHA256",
    "NETWORK_ASSETS",
    "CandidateConfig",
    "CompilationResult",
    "CompiledPlan",
    "ConvLayerShape",
    "LayerMedoid",
    "MappingPolicyBlock",
    "NetworkLayerPanel",
    "PanelExtractionReceipt",
    "PanelExtractionResult",
    "RUNTIME_TRANSLATOR_DEFINITION_SHA256",
    "RuntimeLayerManifest",
    "TimeloopV2Compiler",
    "TimeloopV2CandidateInfeasibleError",
    "TimeloopV2ContractError",
    "TimeloopV2DetailedEvaluationAdapter",
    "TimeloopV2DetailedProjectionError",
    "TimeloopV2DockerEvaluator",
    "TimeloopV2Evaluation",
    "TimeloopV2EvaluatorPort",
    "TimeloopV2FiniteVariationCatalog",
    "TimeloopV2InfeasibleEvaluation",
    "TimeloopV2Settings",
    "TimeloopV2StaticInfeasibleEvaluation",
    "TimeloopV2StaticMapspaceWitness",
    "TIMELOOP_V2_DETAILED_EVALUATOR_ID",
    "TIMELOOP_V2_DETAILED_EVALUATOR_VERSION",
    "TIMELOOP_V2_DETAILED_RECEIPT_SCHEMA_VERSION",
    "WORKLOAD_DEFINITION_SHA256",
    "audit_compiler_injectivity",
    "analyze_static_mapspace_feasibility",
    "candidate_sha256",
    "build_evaluation_bundle",
    "compiled_plan_sha256",
    "compose_timeloop_v2_detailed_benchmark",
    "compose_timeloop_v2_campaign_workload",
    "compile_runtime_layer_manifests",
    "extract_network_panel",
    "extraction_receipt_sha256",
    "frozen_network_panel",
    "normalize_candidate",
    "panel_sha256",
    "runtime_layer_manifest_sha256",
    "timeloop_v2_evaluator_context_record",
    "timeloop_v2_evaluator_identity",
    "verify_network_asset",
]
