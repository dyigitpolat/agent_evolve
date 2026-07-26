"""Provider-free inputs for the Airfoil multi-option evolutionary pilot.

This module is an Airfoil-only composition boundary.  It upgrades fresh frozen
G3 runtime inputs with the canonical post-G3 v2 card, authenticates three prior
development A/U result artifacts, prospectively promotes that tested card, seals
the complete two-card by two-parent K=8 authority matrix, and exposes immutable
launch commitments.  Import does no I/O.  Composition reads only those frozen
result artifacts; it does not read credentials, contact a model, invoke an
evaluator, or import an evolutionary planner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_evolve.agentic import (
    AgenticBenchmark,
    CandidateId,
    CompiledHypothesisTreatment,
    DeterministicIdFactory,
    FiniteActionSetAuthority,
    FrozenJsonObject,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightLifecycleState,
    InsightOrigin,
    OperatorKind,
    ParentBoundActionChoice,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.application.insight_memory import context_stratum_hash
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareOpenRouterConfig,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
    build_openrouter_config,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    AIRFOIL_G3_RUNTIME_PROBLEM_ID,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AIRFOIL_G3_MODEL_CATALOG_ID,
    AirfoilG3RuntimeInputs,
    FrozenPrefixIdFactory,
    load_frozen_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_stage_b_action_set import (
    AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256,
    AirfoilTrimLocalSupportCompiler,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_hypothesis import (
    AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256,
    AirfoilV8ReflectionNativeTrimHypothesisCompiler,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
    AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID,
    AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256,
    AIRFOIL_V8_STAGE_B_CARD_VERSION,
    AIRFOIL_V8_STAGE_B_CATALOG_ID,
    AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    CANONICAL_G3_TRACE_PATH,
    build_airfoil_v8_stage_b_benchmark,
    replay_canonical_g3_adaptive_revision,
)


AIRFOIL_V10_RUN_SEED = 2_026_071_510
AIRFOIL_V10_MULTI_OPTION_PHASE = "airfoil_v10_multi_option"
AIRFOIL_V10_CONTEXT_PROJECTION_SHA256 = context_stratum_hash(
    problem_id=AIRFOIL_G3_RUNTIME_PROBLEM_ID,
    operator_kind=OperatorKind.TYPED_MUTATION.value,
    phase=AIRFOIL_V10_MULTI_OPTION_PHASE,
)
AIRFOIL_V10_OTHER_CARD_INSIGHT_ID = "insight_airfoil_twostage_cards_000007"
AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION = 1
AIRFOIL_V10_OTHER_CARD_VERSION = 2
AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256 = (
    "425e23a51a5ed9831defe58faf22095f302ecd06aa9d23138c4f7a8f7271cfb5"
)
AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS = (
    OperatorKind.TYPED_MUTATION.value,
)
AIRFOIL_V10_OTHER_CARD_REVISION_NOTE = (
    "Legacy applicability migration: authenticated G3 compiler and treatment "
    "traces executed this unchanged card as typed_mutation; v2 changes only "
    "the registered operator applicability from mutation to typed_mutation."
)
AIRFOIL_V10_DIAGNOSTIC_SEED_CONFIGURATION_SHA256 = (
    "4e17a2c2d5efce96e554858f4baad762de76626aec3df0d90ee63711545f9122"
)
AIRFOIL_V10_HYPOTHESIS_SEED_CONFIGURATION_SHA256 = (
    "cb601c3588ca6f17e527b8f7961c1b22e3ae12a138fd33c34bc869d1c6b852a0"
)
AIRFOIL_V10_MATE_OPTION_ID = "shape.camber_aft.n0030"
AIRFOIL_V10_MATE_SELECTION_SHA256 = (
    "007c29a5d3ff469a641b3c16b2cee201dec9a9d54aa321477bedef772e3a3db4"
)

AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "airfoil_stage_b_development"
)
AIRFOIL_V10_PROMOTION_REASON = (
    "Promote the canonical learned v2 card for prospective v10 development "
    "after three authenticated matched A/U blocks showed A > U, including "
    "two fresh-parent transfers; this is development evidence, not paper evidence."
)
AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "airfoil_g3"
    / "airfoil_v7_g3_20260715t1204z"
)
AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON = (
    "Promote the unchanged applicability-migration card 000007@2 to "
    "retrievable for prospective v10 after authenticated finalized G3 traces "
    "proved its 000007@1 predecessor in two valid, evidence-compliant "
    "typed-mutation executions; this claims tested compatibility only, not "
    "efficacy or paper evidence."
)
_AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS = (
    (
        "optimizer_result.json",
        "34e349e2b0989cf7f2051d335068c6d6c0e4a6a485399531fb3a9383bb0a0b51",
    ),
    (
        "analysis_receipt.json",
        "c045321082603de0eb36dc7d82a19ec71bede5cd6b61bae1197c8cd4cf349a79",
    ),
    (
        "finalized.json",
        "f0e93b83119e49abbe003e18a796a441ed16e6433d7eca71adb8bcbe1c6a91c3",
    ),
    (
        "execution_traces.jsonl",
        "45e77289802e484dc90f958d38f879a1e3583aa0361da758ed04286e9053f6f8",
    ),
)
_AIRFOIL_V10_OTHER_CARD_TEST_OBSERVATIONS = (
    {
        "stage": "g1_diagnostic",
        "call_id": "call_airfoil_g3_runtime_000001",
        "candidate_id": "candidate_airfoil_g3_runtime_000003",
        "operator_invocation_id": "operator_airfoil_g3_runtime_000001",
        "objective_hex": "0x1.00fd283e2485bp+0",
        "scalar_reward_hex": "-0x1.5686c7958f482p-2",
    },
    {
        "stage": "g2_score_shuffled",
        "call_id": "call_airfoil_g3_runtime_000004",
        "candidate_id": "candidate_airfoil_g3_runtime_000007",
        "operator_invocation_id": "operator_airfoil_g3_runtime_000004",
        "objective_hex": "0x1.f887393b910fdp-1",
        "scalar_reward_hex": "-0x1.75bd33180cb1cp-2",
    },
)
_AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS = (
    {
        "evidence_id": "airfoil_v8_source_parent",
        "relative_path": (
            "airfoil_v8_stage_b_dev_20260715t1343z/result.json"
        ),
        "result_file_sha256": (
            "aeb8745f607e4c1bdcb675c1dd4c318c2ab4767510597a7d3dd08e4b11cc0ad2"
        ),
        "optimizer_result_sha256": (
            "3282d5f5141f8a3780e8be1a65df9d0c4c57f3c165cc46f8064fb78e186bbf30"
        ),
        "claim_boundary": "development_not_fresh_paper_evidence",
        "fresh_transfer_parent": False,
    },
    {
        "evidence_id": "airfoil_v9_fresh_parent_t000",
        "relative_path": (
            "airfoil_v9_stage_b_t000_20260715t1403z/result.json"
        ),
        "result_file_sha256": (
            "58807f0943f7f54586f3c4050916ac0c4e71e047a841b6b3d827423955959d06"
        ),
        "optimizer_result_sha256": (
            "350e5bfc3fbd181da6c2c28c7ca2e3c14c98d8acf96b8cccca4538215065a043"
        ),
        "claim_boundary": (
            "fresh_parent_single_block_development_not_replicated_paper_evidence"
        ),
        "fresh_transfer_parent": True,
    },
    {
        "evidence_id": "airfoil_v9_fresh_parent_t017",
        "relative_path": (
            "airfoil_v9_stage_b_dev_t017_20260715t1424z/result.json"
        ),
        "result_file_sha256": (
            "fe8fa954eba694355d0c5b1dae6eb1ed8ee09a756efbb9e163f6635df0f7b333"
        ),
        "optimizer_result_sha256": (
            "c530aaa32d5a35b56e73e684050d15c1838281b919194e9c6c5cd4e47d796360"
        ),
        "claim_boundary": (
            "fresh_parent_single_block_development_not_replicated_paper_evidence"
        ),
        "fresh_transfer_parent": True,
    },
)

AIRFOIL_V10_SEED_ROLES = ("diagnostic_parent", "hypothesis_parent")
AIRFOIL_V10_CARD_ROLES = ("learned_v2", "other_migrated_v2")
AIRFOIL_V10_AUTHORITY_MATRIX_ORDER = tuple(
    (seed_role, card_role)
    for seed_role in AIRFOIL_V10_SEED_ROLES
    for card_role in AIRFOIL_V10_CARD_ROLES
)

_DEFINITION_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:def:v1\x00"
_TASK_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:task:v1\x00"
_PROBE_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:probe:v1\x00"
_SCHEDULE_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:schedule:v1\x00"
_PRE_OUTCOME_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:pre-outcome:v1\x00"
_INPUT_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:inputs:v1\x00"
_GPT_RECORD_DOMAIN = b"agent-evolve:airfoil-v10-multi-option:gpt-profile:v1\x00"
_PROMOTION_EVIDENCE_DOMAIN = (
    b"agent-evolve:airfoil-v10-promotion-evidence:v1\x00"
)
_PROMOTION_EVIDENCE_SET_DOMAIN = (
    b"agent-evolve:airfoil-v10-promotion-evidence-set:v1\x00"
)
_OTHER_CARD_EVIDENCE_DOMAIN = (
    b"agent-evolve:airfoil-v10-other-card-evidence:v1\x00"
)


class AirfoilV10MultiOptionInputError(RuntimeError):
    """A frozen source, authority, or no-outcome commitment drifted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE = tuple(
    sorted(
        (
            *(
                "optimizer-result-sha256:"
                + str(spec["optimizer_result_sha256"])
                for spec in _AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS
            ),
            *(
                "result-file-sha256:" + str(spec["result_file_sha256"])
                for spec in _AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS
            ),
        )
    )
)
AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256 = _hash(
    _PROMOTION_EVIDENCE_SET_DOMAIN,
    {
        "schema_version": 1,
        "evidence_specs": list(_AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS),
        "promotion_reason": AIRFOIL_V10_PROMOTION_REASON,
        "supporting_evidence": list(
            AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE
        ),
        "target_lifecycle_state": InsightLifecycleState.PROMOTED.value,
        "retrievable_required": True,
        "paper_evidence_eligible": False,
    },
)
AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256 = _hash(
    _OTHER_CARD_EVIDENCE_DOMAIN,
    {
        "schema_version": 1,
        "target": {
            "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
            "version": AIRFOIL_V10_OTHER_CARD_VERSION,
            "content_sha256": AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256,
            "applicable_operator_kinds": list(
                AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
            ),
            "origin": InsightOrigin.MANUAL.value,
            "revision_note": AIRFOIL_V10_OTHER_CARD_REVISION_NOTE,
        },
        "tested_predecessor": {
            "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
            "version": AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION,
            "content_sha256": AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256,
        },
        "files": [
            {"relative_path": path, "file_sha256": digest}
            for path, digest in _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
        ],
        "observations": list(_AIRFOIL_V10_OTHER_CARD_TEST_OBSERVATIONS),
        "optimizer_result_sha256": (
            "f2da2b41223b488f10b17f2280a9f439c03af1190c0b699c2b66fab897f32f02"
        ),
        "analysis_sha256": (
            "3c2527263140ee4b2f881272d8e1a4b5d75848e1cba0710d75c1bf8de7287361"
        ),
        "finalization_sha256": (
            "820a0418ef8e9ec1c979daf59031dcdac78b6142ccde8594142edccdcfd72f6a"
        ),
        "reason": AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON,
        "claim": "tested_retrievable_only",
        "efficacy_claim": False,
        "paper_evidence_eligible": False,
    },
)
AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE = tuple(
    sorted(
        (
            *(
                "g3-file-sha256:" + digest
                for _, digest in _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
            ),
            "tested-card-evidence-sha256:"
            + AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256,
        )
    )
)


_DEFINITION = {
    "schema_version": 1,
    "status": "provider_and_evaluator_free_preflight",
    "source": "frozen_airfoil_g3_release_plus_canonical_postseal_v2_revision",
    "g0_seeds": [
        {
            "role": AIRFOIL_V10_SEED_ROLES[0],
            "configuration_sha256": (
                AIRFOIL_V10_DIAGNOSTIC_SEED_CONFIGURATION_SHA256
            ),
        },
        {
            "role": AIRFOIL_V10_SEED_ROLES[1],
            "configuration_sha256": (
                AIRFOIL_V10_HYPOTHESIS_SEED_CONFIGURATION_SHA256
            ),
        },
    ],
    "active_cards": [
        {
            "role": AIRFOIL_V10_CARD_ROLES[0],
            "insight_id": AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID,
            "version": AIRFOIL_V8_STAGE_B_CARD_VERSION,
            "content_sha256": AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
        },
        {
            "role": AIRFOIL_V10_CARD_ROLES[1],
            "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
            "version": AIRFOIL_V10_OTHER_CARD_VERSION,
            "content_sha256": AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256,
        },
    ],
    "learned_v2_promotion": {
        "evidence_definition_sha256": (
            AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256
        ),
        "evidence_specs": list(_AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS),
        "reason": AIRFOIL_V10_PROMOTION_REASON,
        "supporting_evidence": list(
            AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE
        ),
        "prior_state": InsightLifecycleState.QUARANTINED.value,
        "new_state": InsightLifecycleState.PROMOTED.value,
        "retrievable_required": True,
        "development_evidence_only": True,
        "paper_evidence_eligible": False,
    },
    "other_migrated_v2_promotion": {
        "evidence_definition_sha256": (
            AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256
        ),
        "files": [
            {"relative_path": path, "file_sha256": digest}
            for path, digest in _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
        ],
        "observations": list(_AIRFOIL_V10_OTHER_CARD_TEST_OBSERVATIONS),
        "revision": {
            "source_version": AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION,
            "target_version": AIRFOIL_V10_OTHER_CARD_VERSION,
            "draft_content_unchanged": True,
            "applicable_operator_kinds": list(
                AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
            ),
            "origin": InsightOrigin.MANUAL.value,
            "note": AIRFOIL_V10_OTHER_CARD_REVISION_NOTE,
        },
        "reason": AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON,
        "supporting_evidence": list(
            AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE
        ),
        "prior_state": InsightLifecycleState.QUARANTINED.value,
        "new_state": InsightLifecycleState.PROMOTED.value,
        "retrievable_required": True,
        "claim": "tested_retrievable_only",
        "efficacy_claim": False,
        "paper_evidence_eligible": False,
    },
    "authority_matrix_order": [
        list(value) for value in AIRFOIL_V10_AUTHORITY_MATRIX_ORDER
    ],
    "runtime_problem_id": AIRFOIL_G3_RUNTIME_PROBLEM_ID,
    "phase": AIRFOIL_V10_MULTI_OPTION_PHASE,
    "context_projection_sha256": AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
    "catalog_id": AIRFOIL_V8_STAGE_B_CATALOG_ID,
    "support_cardinality": AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    "support_compiler_sha256": AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256,
    "hypothesis_compiler_sha256": (
        AIRFOIL_V8_STAGE_B_HYPOTHESIS_COMPILER_DEFINITION_SHA256
    ),
    "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
    "orthogonal_mate": {
        "option_id": AIRFOIL_V10_MATE_OPTION_ID,
        "selection_sha256": AIRFOIL_V10_MATE_SELECTION_SHA256,
        "family": "shape_only",
        "parent_role": AIRFOIL_V10_SEED_ROLES[1],
    },
    "current_outcome_access": False,
}
AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256 = _hash(
    _DEFINITION_DOMAIN,
    _DEFINITION,
)
AIRFOIL_V10_MULTI_OPTION_TASK_SHA256 = _hash(
    _TASK_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
        "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
        "context_projection_sha256": AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
        "current_outcome_access": False,
    },
)
AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS = tuple(
    CandidateId(
        "candidate_"
        + _hash(
            _PROBE_DOMAIN,
            {
                "task_sha256": AIRFOIL_V10_MULTI_OPTION_TASK_SHA256,
                "opaque_ordinal": ordinal,
            },
        )
    )
    for ordinal in range(len(AIRFOIL_V10_AUTHORITY_MATRIX_ORDER))
)
AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256 = _hash(
    _SCHEDULE_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
        "task_sha256": AIRFOIL_V10_MULTI_OPTION_TASK_SHA256,
        "run_seed": AIRFOIL_V10_RUN_SEED,
        "g0_seed_order": list(AIRFOIL_V10_SEED_ROLES),
        "authority_matrix_order": [
            {
                "seed_role": seed_role,
                "card_role": card_role,
                "opaque_probe_candidate_id": probe.value,
            }
            for (seed_role, card_role), probe in zip(
                AIRFOIL_V10_AUTHORITY_MATRIX_ORDER,
                AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS,
                strict=True,
            )
        ],
        "current_outcome_access": False,
    },
)
AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256 = _hash(
    _PRE_OUTCOME_DOMAIN,
    {
        "definition_sha256": AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
        "task_sha256": AIRFOIL_V10_MULTI_OPTION_TASK_SHA256,
        "schedule_sha256": AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256,
        "new_candidate_outcomes_observed": False,
        "current_outcome_access": False,
    },
)


def _forbid_mode_or_pro_fields(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise AirfoilV10MultiOptionInputError(
                    "GPT profile record keys must be exact strings"
                )
            if key.casefold() in {"mode", "pro"}:
                raise AirfoilV10MultiOptionInputError(
                    f"unsupported reasoning field at {path}.{key}"
                )
            _forbid_mode_or_pro_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _forbid_mode_or_pro_fields(child, path=f"{path}[{index}]")


def build_airfoil_v10_gpt_openrouter_config() -> ProgressAwareOpenRouterConfig:
    """Project the authenticated GPT profile into the v10 queue envelope."""

    profile = GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE
    profile.__post_init__()
    config = replace(
        build_openrouter_config(profile),
        jitter_seed=AIRFOIL_V10_RUN_SEED,
        jitter_domain="airfoil-v10-multi-option-evolution-v1",
        app_title="AgentEvolve AAAI 2027 Airfoil v10 multi-option pilot",
    )
    config.__post_init__()
    if (
        config.model_name != profile.model_alias
        or config.provider_only != (profile.provider_slug,)
        or config.provider_require_parameters is not True
        or config.reasoning_config is None
        or config.reasoning_config.to_model_setting() != {"effort": "xhigh"}
    ):
        raise AirfoilV10MultiOptionInputError("GPT xhigh profile projection drifted")
    return config


def airfoil_v10_gpt_profile_config_record() -> dict[str, object]:
    """Return a fresh provider-free GPT profile/config manifest projection."""

    profile = GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE
    config = build_airfoil_v10_gpt_openrouter_config()
    transport = config.to_manifest_record()
    record: dict[str, object] = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "requested_model": profile.model_alias,
        "canonical_model": profile.canonical_model,
        "provider_slug": profile.provider_slug,
        "resolved_provider": profile.resolved_provider,
        "provider_options": config.provider_options,
        "reasoning": transport["reasoning"],
        "temperature": profile.temperature,
        "max_input_tokens": profile.max_input_tokens,
        "max_output_tokens": profile.max_output_tokens,
        "max_reasoning_tokens": profile.max_reasoning_tokens,
        "artificial_output_cap": False,
        "transport": transport,
    }
    if record["reasoning"] != {"effort": "xhigh"}:
        raise AirfoilV10MultiOptionInputError("GPT reasoning effort is not xhigh")
    _forbid_mode_or_pro_fields(record)
    return record


AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256 = _hash(
    _GPT_RECORD_DOMAIN,
    airfoil_v10_gpt_profile_config_record(),
)


@dataclass(frozen=True, slots=True)
class AirfoilV10PromotionEvidenceRecord:
    """Authenticated development-only A/U result used for v2 promotion."""

    evidence_id: str
    relative_path: str
    result_file_sha256: str
    optimizer_result_sha256: str
    claim_boundary: str
    fresh_transfer_parent: bool
    adaptive_minus_uniform_reward_hex: str
    authority_sha256: str
    support_sha256: str
    transfer_parent_configuration_sha256: str | None
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        specs = {
            str(spec["evidence_id"]): spec
            for spec in _AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS
        }
        if type(self.evidence_id) is not str or self.evidence_id not in specs:
            raise ValueError("unknown Airfoil v10 promotion evidence ID")
        expected = specs[self.evidence_id]
        for name in (
            "relative_path",
            "result_file_sha256",
            "optimizer_result_sha256",
            "claim_boundary",
        ):
            if getattr(self, name) != expected[name]:
                raise ValueError(f"promotion evidence {name} changed")
        if (
            type(self.fresh_transfer_parent) is not bool
            or self.fresh_transfer_parent is not expected["fresh_transfer_parent"]
        ):
            raise ValueError("promotion evidence transfer-parent status changed")
        _require_sha256(
            self.result_file_sha256,
            name="promotion result_file_sha256",
        )
        _require_sha256(
            self.optimizer_result_sha256,
            name="promotion optimizer_result_sha256",
        )
        _require_sha256(
            self.authority_sha256,
            name="promotion authority_sha256",
        )
        _require_sha256(
            self.support_sha256,
            name="promotion support_sha256",
        )
        if (
            type(self.adaptive_minus_uniform_reward_hex) is not str
            or not self.adaptive_minus_uniform_reward_hex
        ):
            raise ValueError("promotion A-U reward must be a canonical hex float")
        try:
            improvement = float.fromhex(
                self.adaptive_minus_uniform_reward_hex
            )
        except ValueError as exc:
            raise ValueError("promotion A-U reward is not a hex float") from exc
        if (
            improvement <= 0.0
            or improvement.hex() != self.adaptive_minus_uniform_reward_hex
        ):
            raise ValueError("promotion evidence does not prove canonical A > U")
        if self.fresh_transfer_parent:
            _require_sha256(
                self.transfer_parent_configuration_sha256,
                name="promotion transfer_parent_configuration_sha256",
            )
        elif self.transfer_parent_configuration_sha256 is not None:
            raise ValueError("source-parent evidence cannot claim a transfer parent")
        object.__setattr__(
            self,
            "record_sha256",
            _hash(_PROMOTION_EVIDENCE_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "evidence_id": self.evidence_id,
            "relative_path": self.relative_path,
            "result_file_sha256": self.result_file_sha256,
            "optimizer_result_sha256": self.optimizer_result_sha256,
            "claim_boundary": self.claim_boundary,
            "fresh_transfer_parent": self.fresh_transfer_parent,
            "adaptive_beats_uniform": True,
            "a_u_alias": False,
            "adaptive_minus_uniform_reward_hex": (
                self.adaptive_minus_uniform_reward_hex
            ),
            "authority_sha256": self.authority_sha256,
            "support_sha256": self.support_sha256,
            "transfer_parent_configuration_sha256": (
                self.transfer_parent_configuration_sha256
            ),
            "development_evidence_only": True,
            "paper_evidence_eligible": False,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._identity_record(), "record_sha256": self.record_sha256}


def _exact_result_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise AirfoilV10MultiOptionInputError(
            f"promotion result {name} must be an exact JSON object"
        )
    return value


def _canonical_reward(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise AirfoilV10MultiOptionInputError(
            f"promotion result {name} must be a hex float"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise AirfoilV10MultiOptionInputError(
            f"promotion result {name} is not a hex float"
        ) from exc
    if parsed.hex() != value:
        raise AirfoilV10MultiOptionInputError(
            f"promotion result {name} is not canonical"
        )
    return parsed


def authenticate_airfoil_v10_promotion_evidence(
    root: Path = AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT,
) -> tuple[
    AirfoilV10PromotionEvidenceRecord,
    AirfoilV10PromotionEvidenceRecord,
    AirfoilV10PromotionEvidenceRecord,
]:
    """Read and authenticate the three historical development A/U results."""

    if not isinstance(root, Path):
        raise TypeError("promotion evidence root must be a Path")
    records: list[AirfoilV10PromotionEvidenceRecord] = []
    for spec in _AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS:
        path = root / str(spec["relative_path"])
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise AirfoilV10MultiOptionInputError(
                "required promotion evidence result is unavailable"
            ) from exc
        observed_file_sha256 = hashlib.sha256(content).hexdigest()
        if observed_file_sha256 != spec["result_file_sha256"]:
            raise AirfoilV10MultiOptionInputError(
                "promotion evidence result file SHA-256 changed"
            )
        try:
            result = json.loads(content.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AirfoilV10MultiOptionInputError(
                "promotion evidence result is not canonical UTF-8 JSON"
            ) from exc
        result = _exact_result_mapping(result, name="root")
        if (
            result.get("optimizer_result_sha256")
            != spec["optimizer_result_sha256"]
            or result.get("claim_boundary") != spec["claim_boundary"]
            or result.get("adaptive_beats_uniform") is not True
            or result.get("a_u_alias") is not False
            or result.get("support_cardinality")
            != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
            or result.get("logical_llm_calls") != 1
            or result.get("unique_evaluations") != 3
            or result.get("stop_reason") != "generation_limit_reached"
        ):
            raise AirfoilV10MultiOptionInputError(
                "promotion evidence result identity or A/U status changed"
            )
        arms = _exact_result_mapping(result.get("arms"), name="arms")
        adaptive = _exact_result_mapping(arms.get("A"), name="arms.A")
        uniform = _exact_result_mapping(arms.get("U"), name="arms.U")
        adaptive_candidate = _exact_result_mapping(
            adaptive.get("candidate"),
            name="arms.A.candidate",
        )
        uniform_candidate = _exact_result_mapping(
            uniform.get("candidate"),
            name="arms.U.candidate",
        )
        if (
            adaptive.get("role") != "adaptive_card_model_choice"
            or adaptive.get("selector_kind") != "model"
            or uniform.get("role") != "prospective_uniform_same_support"
            or uniform.get("selector_kind") != "engine"
            or adaptive.get("failure_stage") is not None
            or uniform.get("failure_stage") is not None
            or adaptive_candidate.get("valid") is not True
            or uniform_candidate.get("valid") is not True
            or adaptive.get("option_id") == uniform.get("option_id")
            or adaptive_candidate.get("configuration_hash")
            == uniform_candidate.get("configuration_hash")
        ):
            raise AirfoilV10MultiOptionInputError(
                "promotion evidence arms are aliased, invalid, or mislabeled"
            )
        adaptive_reward = _canonical_reward(
            adaptive.get("reward_hex"),
            name="arms.A.reward_hex",
        )
        uniform_reward = _canonical_reward(
            uniform.get("reward_hex"),
            name="arms.U.reward_hex",
        )
        declared_improvement = _canonical_reward(
            result.get("adaptive_minus_uniform_reward_hex"),
            name="adaptive_minus_uniform_reward_hex",
        )
        if (
            adaptive_reward <= uniform_reward
            or adaptive_reward - uniform_reward != declared_improvement
            or declared_improvement <= 0.0
        ):
            raise AirfoilV10MultiOptionInputError(
                "promotion evidence arithmetic does not prove A > U"
            )
        fresh_transfer_parent = bool(spec["fresh_transfer_parent"])
        transfer_configuration_sha256: str | None = None
        if fresh_transfer_parent:
            transfer = _exact_result_mapping(
                result.get("transfer_parent"),
                name="transfer_parent",
            )
            prospective_uniform = _exact_result_mapping(
                result.get("prospective_uniform"),
                name="prospective_uniform",
            )
            selector_scope = _exact_result_mapping(
                result.get("selector_evidence_scope"),
                name="selector_evidence_scope",
            )
            if (
                transfer.get("g3_source_parent") is not False
                or transfer.get("outcomes_read_by_selection") is not False
                or prospective_uniform.get("outcomes_read") is not False
                or selector_scope.get(
                    "current_parent_outcomes_in_learned_card"
                )
                is not False
            ):
                raise AirfoilV10MultiOptionInputError(
                    "fresh-parent promotion evidence leaked current outcomes"
                )
            transfer_configuration_sha256 = _require_sha256(
                transfer.get("configuration_sha256"),
                name="transfer_parent.configuration_sha256",
            )
        authority_sha256 = _require_sha256(
            result.get("authority_sha256"),
            name="promotion authority_sha256",
        )
        support_sha256 = _require_sha256(
            result.get("support_sha256"),
            name="promotion support_sha256",
        )
        records.append(
            AirfoilV10PromotionEvidenceRecord(
                evidence_id=str(spec["evidence_id"]),
                relative_path=str(spec["relative_path"]),
                result_file_sha256=observed_file_sha256,
                optimizer_result_sha256=str(
                    spec["optimizer_result_sha256"]
                ),
                claim_boundary=str(spec["claim_boundary"]),
                fresh_transfer_parent=fresh_transfer_parent,
                adaptive_minus_uniform_reward_hex=str(
                    result["adaptive_minus_uniform_reward_hex"]
                ),
                authority_sha256=authority_sha256,
                support_sha256=support_sha256,
                transfer_parent_configuration_sha256=(
                    transfer_configuration_sha256
                ),
            )
        )
    values = tuple(records)
    if len(values) != 3:
        raise AssertionError("v10 promotion evidence cardinality changed")
    return values  # type: ignore[return-value]


def _other_card_promotion_evidence_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_definition_sha256": (
            AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256
        ),
        "target": {
            "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
            "version": AIRFOIL_V10_OTHER_CARD_VERSION,
            "content_sha256": AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256,
            "applicable_operator_kinds": list(
                AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
            ),
            "origin": InsightOrigin.MANUAL.value,
            "revision_note": AIRFOIL_V10_OTHER_CARD_REVISION_NOTE,
        },
        "tested_predecessor": {
            "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
            "version": AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION,
            "content_sha256": AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256,
        },
        "files": [
            {"relative_path": path, "file_sha256": digest}
            for path, digest in _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
        ],
        "observations": [
            {
                **observation,
                "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
                "insight_version": AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION,
                "valid": True,
                "evidence_compliant": True,
                "operator_compliant": True,
            }
            for observation in _AIRFOIL_V10_OTHER_CARD_TEST_OBSERVATIONS
        ],
        "optimizer_result_sha256": (
            "f2da2b41223b488f10b17f2280a9f439c03af1190c0b699c2b66fab897f32f02"
        ),
        "analysis_sha256": (
            "3c2527263140ee4b2f881272d8e1a4b5d75848e1cba0710d75c1bf8de7287361"
        ),
        "finalization_sha256": (
            "820a0418ef8e9ec1c979daf59031dcdac78b6142ccde8594142edccdcfd72f6a"
        ),
        "claim": "tested_retrievable_only",
        "efficacy_claim": False,
        "paper_evidence_eligible": False,
    }


AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256 = typed_json_sha256(
    freeze_json(_other_card_promotion_evidence_record())
)


def authenticate_airfoil_v10_other_card_promotion_evidence(
    root: Path = AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_ROOT,
) -> FrozenJsonObject:
    """Authenticate finalized G3 proof that 000007@1 was validly tested twice."""

    if not isinstance(root, Path):
        raise TypeError("other-card promotion evidence root must be a Path")
    contents: dict[str, bytes] = {}
    for relative_path, expected_sha256 in (
        _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
    ):
        try:
            content = (root / relative_path).read_bytes()
        except OSError as exc:
            raise AirfoilV10MultiOptionInputError(
                "required finalized G3 promotion evidence is unavailable"
            ) from exc
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise AirfoilV10MultiOptionInputError(
                "finalized G3 promotion evidence file SHA-256 changed"
            )
        contents[relative_path] = content
    try:
        optimizer = json.loads(contents["optimizer_result.json"])
        analysis = json.loads(contents["analysis_receipt.json"])
        finalized = json.loads(contents["finalized.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AirfoilV10MultiOptionInputError(
            "finalized G3 promotion evidence is not valid JSON"
        ) from exc
    optimizer = _exact_result_mapping(optimizer, name="G3 optimizer result")
    analysis = _exact_result_mapping(analysis, name="G3 analysis receipt")
    finalized = _exact_result_mapping(finalized, name="G3 finalization")
    claim_boundary = _exact_result_mapping(
        analysis.get("claim_boundary"),
        name="G3 claim boundary",
    )
    if (
        optimizer.get("optimizer_result_sha256")
        != "f2da2b41223b488f10b17f2280a9f439c03af1190c0b699c2b66fab897f32f02"
        or optimizer.get("stop_reason") != "generation_limit_reached"
        or analysis.get("optimizer_result_sha256")
        != optimizer.get("optimizer_result_sha256")
        or analysis.get("analysis_sha256")
        != "3c2527263140ee4b2f881272d8e1a4b5d75848e1cba0710d75c1bf8de7287361"
        or claim_boundary.get("paper_ready_sota_claim") is not False
        or claim_boundary.get("genericity_claim") is not False
        or finalized.get("status") != "completed"
        or finalized.get("finalization_sha256")
        != "820a0418ef8e9ec1c979daf59031dcdac78b6142ccde8594142edccdcfd72f6a"
    ):
        raise AirfoilV10MultiOptionInputError(
            "finalized G3 promotion evidence semantic identity changed"
        )
    finalized_files = _exact_result_mapping(
        finalized.get("files"),
        name="G3 finalized files",
    )
    for relative_path, expected_sha256 in (
        _AIRFOIL_V10_OTHER_CARD_PROMOTION_FILE_SPECS
    ):
        if relative_path == "finalized.json":
            continue
        file_record = _exact_result_mapping(
            finalized_files.get(relative_path),
            name=f"G3 finalized file {relative_path}",
        )
        if file_record.get("sha256") != expected_sha256:
            raise AirfoilV10MultiOptionInputError(
                "G3 finalization does not bind a promotion evidence file"
            )
    expected_by_call = {
        str(value["call_id"]): value
        for value in _AIRFOIL_V10_OTHER_CARD_TEST_OBSERVATIONS
    }
    observed: dict[str, dict[str, object]] = {}
    try:
        trace_lines = contents["execution_traces.jsonl"].decode(
            "utf-8", errors="strict"
        ).splitlines()
        for line in trace_lines:
            event = json.loads(line)
            if (
                type(event) is dict
                and event.get("event_type") == "invocation_completed"
                and event.get("call_id") in expected_by_call
            ):
                call_id = str(event["call_id"])
                if call_id in observed:
                    raise AirfoilV10MultiOptionInputError(
                        "finalized G3 trace duplicates a tested-card outcome"
                    )
                observed[call_id] = event
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AirfoilV10MultiOptionInputError(
            "finalized G3 execution trace is invalid JSONL"
        ) from exc
    if set(observed) != set(expected_by_call):
        raise AirfoilV10MultiOptionInputError(
            "finalized G3 trace lacks a tested-card outcome"
        )
    for call_id, expected in expected_by_call.items():
        event = observed[call_id]
        detailed = _exact_result_mapping(
            event.get("detailed_evaluation"),
            name="G3 tested-card detailed evaluation",
        )
        objectives = _exact_result_mapping(
            detailed.get("objectives"),
            name="G3 tested-card objectives",
        )
        objective = objectives.get("normalized_multipoint_drag")
        reward = event.get("scalar_reward")
        if (
            type(objective) is not float
            or objective.hex() != expected["objective_hex"]
            or type(reward) is not float
            or reward.hex() != expected["scalar_reward_hex"]
            or event.get("candidate_id") != expected["candidate_id"]
            or event.get("operator_invocation_id")
            != expected["operator_invocation_id"]
            or event.get("operator_kind") != OperatorKind.TYPED_MUTATION.value
            or event.get("assignment_kind") != "resolved_causal"
            or event.get("valid") is not True
            or event.get("evidence_compliant") is not True
            or event.get("operator_compliant") is not True
            or event.get("failure_stage") is not None
            or event.get("selected_insights")
            != [
                {
                    "insight_id": AIRFOIL_V10_OTHER_CARD_INSIGHT_ID,
                    "version": AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION,
                }
            ]
        ):
            raise AirfoilV10MultiOptionInputError(
                "finalized G3 tested-card trace fields changed"
            )
    evidence = freeze_json(_other_card_promotion_evidence_record())
    if (
        type(evidence) is not FrozenJsonObject
        or typed_json_sha256(evidence)
        != AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256
    ):
        raise AssertionError("other-card promotion evidence record drifted")
    return evidence


@dataclass(frozen=True, slots=True)
class AirfoilV10G0Seed:
    """One frozen G0 configuration and its pre-outcome compilation context."""

    role: str
    configuration: FrozenJsonObject
    configuration_sha256: str
    context_projection_sha256: str

    def __post_init__(self) -> None:
        if self.role not in AIRFOIL_V10_SEED_ROLES:
            raise ValueError("unknown Airfoil v10 seed role")
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("G0 configuration must be a frozen typed-JSON object")
        if freeze_json(self.configuration) is not self.configuration:
            raise TypeError("G0 configuration must already be frozen")
        _require_sha256(self.configuration_sha256, name="configuration_sha256")
        _require_sha256(
            self.context_projection_sha256,
            name="context_projection_sha256",
        )
        if typed_json_sha256(self.configuration) != self.configuration_sha256:
            raise ValueError("G0 configuration hash changed")
        if self.context_projection_sha256 != AIRFOIL_V10_CONTEXT_PROJECTION_SHA256:
            raise ValueError("G0 seed uses a foreign planner context")

    @property
    def configuration_dict(self) -> dict[str, Any]:
        value = thaw_json(self.configuration)
        if type(value) is not dict:  # pragma: no cover - type closes the root.
            raise TypeError("G0 configuration root is not an object")
        return value

    def to_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "configuration_sha256": self.configuration_sha256,
            "context_projection_sha256": self.context_projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class AirfoilV10AuthorityBinding:
    """One exact card/parent compilation and its sealed K=8 support."""

    seed_role: str
    card_role: str
    seed_configuration_sha256: str
    card: InsightMemoryEntry
    probe_candidate_id: CandidateId
    compiled_treatment: CompiledHypothesisTreatment
    authority: FiniteActionSetAuthority

    def __post_init__(self) -> None:
        if (self.seed_role, self.card_role) not in AIRFOIL_V10_AUTHORITY_MATRIX_ORDER:
            raise ValueError("authority binding has an unknown matrix coordinate")
        _require_sha256(
            self.seed_configuration_sha256,
            name="seed_configuration_sha256",
        )
        if type(self.card) is not InsightMemoryEntry:
            raise TypeError("authority card must be an exact memory entry")
        self.card.__post_init__()
        if type(self.probe_candidate_id) is not CandidateId:
            raise TypeError("probe_candidate_id must be an exact CandidateId")
        self.probe_candidate_id.__post_init__()
        if type(self.compiled_treatment) is not CompiledHypothesisTreatment:
            raise TypeError("compiled_treatment must be exact")
        self.compiled_treatment.__post_init__()
        if type(self.authority) is not FiniteActionSetAuthority:
            raise TypeError("authority must be an exact finite action set")
        self.authority.__post_init__()
        request = self.compiled_treatment.request
        support = self.authority.support
        if (
            request.parent_candidate_id != self.probe_candidate_id
            or request.parent_configuration_sha256
            != self.seed_configuration_sha256
            or request.reference != self.card.reference
            or request.insight.content_sha256 != self.card.draft.content_sha256
            or request.requested_operator_kind
            != OperatorKind.TYPED_MUTATION.value
            or request.context_projection_sha256
            != AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
            or support.parent_candidate_id != self.probe_candidate_id
            or support.parent_configuration_sha256
            != self.seed_configuration_sha256
            or support.context_projection_sha256
            != AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
            or self.authority.card.reference != self.card.reference
            or self.authority.card.card_content_sha256
            != self.card.draft.content_sha256
            or support.cardinality != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
            or self.authority.current_outcome_access is not False
        ):
            raise ValueError("compiled K=8 authority lost its exact card/parent binding")
        option_ids = tuple(row.option.option_id for row in support.options)
        child_hashes = tuple(
            row.option.child_configuration_sha256 for row in support.options
        )
        phenotypes = tuple(
            row.phenotype_identity_sha256 for row in support.options
        )
        if any(len(set(values)) != AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY for values in (
            option_ids,
            child_hashes,
            phenotypes,
        )):
            raise ValueError("K=8 support contains a duplicate action or phenotype")

    def to_record(self) -> dict[str, object]:
        return {
            "seed_role": self.seed_role,
            "card_role": self.card_role,
            "seed_configuration_sha256": self.seed_configuration_sha256,
            "card": {
                "insight_id": self.card.reference.insight_id.value,
                "version": self.card.reference.version,
                "content_sha256": self.card.draft.content_sha256,
            },
            "probe_candidate_id": self.probe_candidate_id.value,
            "compiled_treatment_sha256": self.compiled_treatment.binding_sha256,
            "authority_sha256": self.authority.authority_sha256,
            "support_sha256": self.authority.support.support_sha256,
            "anchor_option_id": self.authority.support.anchor_option_id,
            "option_ids": [
                row.option.option_id for row in self.authority.support.options
            ],
            "child_configuration_sha256s": [
                row.option.child_configuration_sha256
                for row in self.authority.support.options
            ],
            "phenotype_identity_sha256s": [
                row.phenotype_identity_sha256
                for row in self.authority.support.options
            ],
            "current_outcome_access": self.authority.current_outcome_access,
        }


@dataclass(frozen=True, slots=True)
class AirfoilV10MultiOptionInputs:
    """Complete provider/evaluator-free inputs for a future generic planner."""

    benchmark: AgenticBenchmark
    id_factory: Any
    memory: InsightMemoryBank
    promotion_evidence: tuple[
        AirfoilV10PromotionEvidenceRecord,
        AirfoilV10PromotionEvidenceRecord,
        AirfoilV10PromotionEvidenceRecord,
    ]
    other_card_promotion_evidence: FrozenJsonObject
    active_cards: tuple[InsightMemoryEntry, InsightMemoryEntry]
    g0_seeds: tuple[AirfoilV10G0Seed, AirfoilV10G0Seed]
    authority_bindings: tuple[
        AirfoilV10AuthorityBinding,
        AirfoilV10AuthorityBinding,
        AirfoilV10AuthorityBinding,
        AirfoilV10AuthorityBinding,
    ]
    mate_choice: ParentBoundActionChoice
    mate_configuration: FrozenJsonObject
    source_runtime_inputs_sha256: str
    source_release_sha256: str
    phase: str = AIRFOIL_V10_MULTI_OPTION_PHASE
    context_projection_sha256: str = AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
    task_sha256: str = AIRFOIL_V10_MULTI_OPTION_TASK_SHA256
    schedule_sha256: str = AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256
    pre_outcome_commit_sha256: str = (
        AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256
    )
    current_outcome_access: bool = False
    promotion_evidence_set_sha256: str = field(init=False)
    inputs_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.benchmark.validate_binding()
        if (
            type(self.benchmark.hypothesis_compiler)
            is not AirfoilV8ReflectionNativeTrimHypothesisCompiler
            or type(self.benchmark.finite_action_set_compiler)
            is not AirfoilTrimLocalSupportCompiler
        ):
            raise ValueError("v10 benchmark is not the exact Stage-B binding")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if (
            type(self.promotion_evidence) is not tuple
            or len(self.promotion_evidence) != 3
            or any(
                type(record) is not AirfoilV10PromotionEvidenceRecord
                for record in self.promotion_evidence
            )
        ):
            raise ValueError("v10 requires three exact promotion evidence records")
        for record in self.promotion_evidence:
            record.__post_init__()
        if tuple(
            (
                record.evidence_id,
                record.relative_path,
                record.result_file_sha256,
                record.optimizer_result_sha256,
                record.claim_boundary,
                record.fresh_transfer_parent,
            )
            for record in self.promotion_evidence
        ) != tuple(
            (
                spec["evidence_id"],
                spec["relative_path"],
                spec["result_file_sha256"],
                spec["optimizer_result_sha256"],
                spec["claim_boundary"],
                spec["fresh_transfer_parent"],
            )
            for spec in _AIRFOIL_V10_PROMOTION_EVIDENCE_SPECS
        ):
            raise ValueError("v10 promotion evidence order or identity changed")
        promotion_evidence_set_sha256 = _hash(
            _PROMOTION_EVIDENCE_SET_DOMAIN,
            {
                "schema_version": 1,
                "definition_sha256": (
                    AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256
                ),
                "records": [
                    record.to_record() for record in self.promotion_evidence
                ],
            },
        )
        object.__setattr__(
            self,
            "promotion_evidence_set_sha256",
            promotion_evidence_set_sha256,
        )
        if (
            type(self.other_card_promotion_evidence) is not FrozenJsonObject
            or typed_json_sha256(self.other_card_promotion_evidence)
            != AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256
            or thaw_json(self.other_card_promotion_evidence)
            != _other_card_promotion_evidence_record()
        ):
            raise ValueError("other-card tested promotion evidence changed")
        if type(self.active_cards) is not tuple or len(self.active_cards) != 2:
            raise ValueError("v10 requires exactly two active cards")
        for card in self.active_cards:
            card.__post_init__()
        learned, other = self.active_cards
        if (
            learned.reference.insight_id.value
            != AIRFOIL_V8_STAGE_B_CARD_INSIGHT_ID
            or learned.reference.version != AIRFOIL_V8_STAGE_B_CARD_VERSION
            or learned.draft.content_sha256
            != AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256
            or learned.evidence_lineage is None
            or learned.evidence_lineage.identity_sha256
            != AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256
            or learned.lifecycle_state is not InsightLifecycleState.PROMOTED
            or learned.retrievable is not True
            or other.reference.insight_id.value
            != AIRFOIL_V10_OTHER_CARD_INSIGHT_ID
            or other.reference.version != AIRFOIL_V10_OTHER_CARD_VERSION
            or other.draft.content_sha256
            != AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256
            or other.applicable_operator_kinds
            != AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
            or other.origin is not InsightOrigin.MANUAL
            or other.evidence_lineage is not None
            or len(other.relations) != 1
            or other.relations[0].kind.value != "revises"
            or other.relations[0].target.insight_id != other.reference.insight_id
            or other.relations[0].target.version
            != AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION
            or other.relations[0].note != AIRFOIL_V10_OTHER_CARD_REVISION_NOTE
            or other.lifecycle_state is not InsightLifecycleState.PROMOTED
            or other.retrievable is not True
            or learned.reference.insight_id == other.reference.insight_id
            or self.memory.entries_for(tuple(card.reference for card in self.active_cards))
            != self.active_cards
        ):
            raise ValueError("active learned/migrated v2 card pair changed")
        source_other_versions = tuple(
            entry
            for entry in self.memory.entries
            if entry.reference.insight_id == other.reference.insight_id
            and entry.reference.version == AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION
        )
        if len(source_other_versions) != 1:
            raise ValueError("other-card source v1 disappeared during migration")
        source_other = source_other_versions[0]
        if (
            source_other.lifecycle_state is not InsightLifecycleState.QUARANTINED
            or source_other.retrievable
            or source_other.draft != other.draft
            or source_other.applicable_operator_kinds != ("mutation",)
        ):
            raise ValueError("other-card source v1 was mutated by migration")
        transitions = self.memory.transitions
        if len(transitions) != 2:
            raise ValueError("v10 requires two exact evidence-backed promotions")
        transition = transitions[0]
        transition.__post_init__()
        if (
            transition.sequence != 1
            or transition.reference != learned.reference
            or transition.prior_state is not InsightLifecycleState.QUARANTINED
            or transition.new_state is not InsightLifecycleState.PROMOTED
            or transition.reason != AIRFOIL_V10_PROMOTION_REASON
            or transition.supporting_evidence
            != AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE
        ):
            raise ValueError("learned-v2 promotion transition changed")
        other_transition = transitions[1]
        other_transition.__post_init__()
        if (
            other_transition.sequence != 2
            or other_transition.reference != other.reference
            or other_transition.prior_state
            is not InsightLifecycleState.QUARANTINED
            or other_transition.new_state is not InsightLifecycleState.PROMOTED
            or other_transition.reason
            != AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON
            or other_transition.supporting_evidence
            != AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE
        ):
            raise ValueError("other-card tested promotion transition changed")
        if (
            type(self.g0_seeds) is not tuple
            or len(self.g0_seeds) != 2
            or tuple(seed.role for seed in self.g0_seeds)
            != AIRFOIL_V10_SEED_ROLES
        ):
            raise ValueError("G0 seed order changed")
        for seed in self.g0_seeds:
            seed.__post_init__()
        if tuple(seed.configuration_sha256 for seed in self.g0_seeds) != (
            AIRFOIL_V10_DIAGNOSTIC_SEED_CONFIGURATION_SHA256,
            AIRFOIL_V10_HYPOTHESIS_SEED_CONFIGURATION_SHA256,
        ):
            raise ValueError("G0 seed identity changed")
        expected_context = context_stratum_hash(
            problem_id=AIRFOIL_G3_RUNTIME_PROBLEM_ID,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=self.phase,
        )
        if (
            self.phase != AIRFOIL_V10_MULTI_OPTION_PHASE
            or self.context_projection_sha256
            != AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
            or self.context_projection_sha256 != expected_context
            or any(
                seed.context_projection_sha256 != self.context_projection_sha256
                for seed in self.g0_seeds
            )
        ):
            raise ValueError("v10 phase/context projection binding changed")
        seed_phenotypes = tuple(
            self.benchmark.phenotype_identity.identify(seed.configuration)
            for seed in self.g0_seeds
        )
        if len({value.identity_sha256 for value in seed_phenotypes}) != 2:
            raise ValueError("G0 seed phenotypes collide")
        if (
            type(self.authority_bindings) is not tuple
            or len(self.authority_bindings) != 4
            or tuple(
                (binding.seed_role, binding.card_role)
                for binding in self.authority_bindings
            )
            != AIRFOIL_V10_AUTHORITY_MATRIX_ORDER
        ):
            raise ValueError("2x2 authority matrix order changed")
        for binding in self.authority_bindings:
            binding.__post_init__()
        if tuple(
            binding.probe_candidate_id for binding in self.authority_bindings
        ) != AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS:
            raise ValueError("opaque authority probe IDs changed")
        if len(
            {binding.authority.authority_sha256 for binding in self.authority_bindings}
        ) != 4 or len(
            {
                binding.authority.support.support_sha256
                for binding in self.authority_bindings
            }
        ) != 4:
            raise ValueError("authority or support receipts collide")
        all_children = tuple(
            row.option.child_configuration_sha256
            for binding in self.authority_bindings
            for row in binding.authority.support.options
        )
        all_phenotypes = tuple(
            row.phenotype_identity_sha256
            for binding in self.authority_bindings
            for row in binding.authority.support.options
        )
        if len(set(all_children)) != 32 or len(set(all_phenotypes)) != 32:
            raise ValueError("the 2x2 matrix has colliding children or phenotypes")
        for seed_role in AIRFOIL_V10_SEED_ROLES:
            pair = tuple(
                binding
                for binding in self.authority_bindings
                if binding.seed_role == seed_role
            )
            if len(pair) != 2:
                raise ValueError("seed does not have both active-card authorities")
            supports = tuple(
                {
                    row.option.option_id
                    for row in binding.authority.support.options
                }
                for binding in pair
            )
            phenotypes = tuple(
                {
                    row.phenotype_identity_sha256
                    for row in binding.authority.support.options
                }
                for binding in pair
            )
            if supports[0].intersection(supports[1]) or phenotypes[0].intersection(
                phenotypes[1]
            ):
                raise ValueError("active cards have overlapping local supports")
        if type(self.mate_choice) is not ParentBoundActionChoice:
            raise TypeError("mate_choice must be exact")
        self.mate_choice.__post_init__()
        if type(self.mate_configuration) is not FrozenJsonObject:
            raise TypeError("mate configuration must be a frozen object")
        hypothesis_seed = self.g0_seeds[1]
        union_contract = self.benchmark.bind_finite_variation(
            AIRFOIL_G3_MODEL_CATALOG_ID,
            hypothesis_seed.configuration,
        )
        self.mate_choice.validate_contract(union_contract)
        mate_option = union_contract.resolve(self.mate_choice.option_id)
        if (
            self.mate_choice.option_id != AIRFOIL_V10_MATE_OPTION_ID
            or self.mate_choice.selection_policy_id
            != "airfoil_v7_g3_sealed_mate"
            or mate_option.family != "shape_only"
            or mate_option.child_configuration != self.mate_configuration
            or any(
                row.phenotype_identity_sha256
                == self.benchmark.phenotype_identity.identify(
                    self.mate_configuration
                ).identity_sha256
                for binding in self.authority_bindings
                for row in binding.authority.support.options
            )
        ):
            raise ValueError("existing shape-only disjoint mate changed")
        for name, expected in (
            ("task_sha256", AIRFOIL_V10_MULTI_OPTION_TASK_SHA256),
            ("schedule_sha256", AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256),
            (
                "pre_outcome_commit_sha256",
                AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256,
            ),
        ):
            value = getattr(self, name)
            _require_sha256(value, name=name)
            if value != expected:
                raise ValueError(f"{name} changed")
        _require_sha256(
            self.source_runtime_inputs_sha256,
            name="source_runtime_inputs_sha256",
        )
        _require_sha256(self.source_release_sha256, name="source_release_sha256")
        if type(self.current_outcome_access) is not bool or self.current_outcome_access:
            raise ValueError("v10 input preparation cannot access current outcomes")
        if any(
            binding.authority.current_outcome_access
            for binding in self.authority_bindings
        ):
            raise ValueError("one authority accessed a current outcome")
        object.__setattr__(
            self,
            "inputs_sha256",
            _hash(_INPUT_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "definition_sha256": AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
            "source_runtime_inputs_sha256": self.source_runtime_inputs_sha256,
            "source_release_sha256": self.source_release_sha256,
            "task_sha256": self.task_sha256,
            "schedule_sha256": self.schedule_sha256,
            "pre_outcome_commit_sha256": self.pre_outcome_commit_sha256,
            "phase": self.phase,
            "context_projection_sha256": self.context_projection_sha256,
            "g0_seeds": [seed.to_record() for seed in self.g0_seeds],
            "active_cards": [
                {
                    "insight_id": card.reference.insight_id.value,
                    "version": card.reference.version,
                    "content_sha256": card.draft.content_sha256,
                    "applicable_operator_kinds": list(
                        card.applicable_operator_kinds
                    ),
                    "origin": card.origin.value,
                    "lifecycle_state": card.lifecycle_state.value,
                    "retrievable": card.retrievable,
                }
                for card in self.active_cards
            ],
            "promotion": {
                "evidence_definition_sha256": (
                    AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256
                ),
                "evidence_set_sha256": self.promotion_evidence_set_sha256,
                "evidence": [
                    record.to_record() for record in self.promotion_evidence
                ],
                "transition": self._promotion_transition_record(),
                "development_evidence_only": True,
                "paper_evidence_eligible": False,
            },
            "other_card_promotion": {
                "evidence_definition_sha256": (
                    AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256
                ),
                "evidence_record_sha256": (
                    AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256
                ),
                "evidence": thaw_json(self.other_card_promotion_evidence),
                "transition": self._promotion_transition_record(1),
                "claim": "tested_retrievable_only",
                "efficacy_claim": False,
                "paper_evidence_eligible": False,
            },
            "authority_bindings": [
                binding.to_record() for binding in self.authority_bindings
            ],
            "mate_choice_sha256": self.mate_choice.choice_sha256,
            "mate_configuration_sha256": typed_json_sha256(
                self.mate_configuration
            ),
            "gpt_profile_config_sha256": (
                AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256
            ),
            "current_outcome_access": self.current_outcome_access,
        }

    def _promotion_transition_record(self, index: int = 0) -> dict[str, object]:
        if type(index) is not int or index not in (0, 1):
            raise ValueError("promotion transition index must be zero or one")
        transition = self.memory.transitions[index]
        return {
            "sequence": transition.sequence,
            "insight_id": transition.reference.insight_id.value,
            "version": transition.reference.version,
            "prior_state": transition.prior_state.value,
            "new_state": transition.new_state.value,
            "reason": transition.reason,
            "supporting_evidence": list(transition.supporting_evidence),
            "retrievable_after_transition": self.active_cards[index].retrievable,
        }

    @property
    def seed_configurations(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return tuple(seed.configuration_dict for seed in self.g0_seeds)  # type: ignore[return-value]

    @property
    def mate_configuration_dict(self) -> dict[str, Any]:
        value = thaw_json(self.mate_configuration)
        if type(value) is not dict:  # pragma: no cover - type closes the root.
            raise TypeError("mate configuration root is not an object")
        return value

    def authority_for(
        self,
        *,
        seed_role: str,
        card_role: str,
    ) -> AirfoilV10AuthorityBinding:
        matches = tuple(
            binding
            for binding in self.authority_bindings
            if binding.seed_role == seed_role and binding.card_role == card_role
        )
        if len(matches) != 1:
            raise KeyError("unknown Airfoil v10 authority coordinate")
        return matches[0]


def _clone_source_memory(
    source: AirfoilG3RuntimeInputs,
) -> tuple[FrozenPrefixIdFactory, InsightMemoryBank]:
    """Copy G3 entries before adding v2, preserving source revalidation."""

    source_entries = source.memory.entries
    ids = FrozenPrefixIdFactory(
        delegate=DeterministicIdFactory("airfoil_v10_multi_option"),
        frozen_insight_ids=tuple(
            entry.reference.insight_id for entry in source_entries
        ),
    )
    memory = InsightMemoryBank(id_factory=ids)
    for expected in source_entries:
        observed, added = memory.add(
            expected.draft,
            initial_score=expected.initial_score,
            applicable_operator_kinds=expected.applicable_operator_kinds,
            origin=expected.origin,
            lifecycle_state=expected.lifecycle_state,
            evidence_lineage=expected.evidence_lineage,
            relations=expected.relations,
        )
        if not added or observed != expected:
            raise AirfoilV10MultiOptionInputError(
                "v10 memory clone changed a frozen G3 entry"
            )
    if ids.frozen_insight_ids_consumed != len(source_entries):
        raise AirfoilV10MultiOptionInputError(
            "v10 memory clone did not consume its frozen insight IDs"
        )
    return ids, memory


def compose_airfoil_v10_multi_option_inputs(
    source: AirfoilG3RuntimeInputs,
    *,
    revision_trace_path: Path = CANONICAL_G3_TRACE_PATH,
    promotion_evidence_root: Path = AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT,
    other_card_promotion_evidence_root: Path = (
        AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_ROOT
    ),
) -> AirfoilV10MultiOptionInputs:
    """Authenticate, promote, and seal without provider or CFD access."""

    if type(source) is not AirfoilG3RuntimeInputs:
        raise TypeError("source must be exact AirfoilG3RuntimeInputs")
    source.__post_init__()
    benchmark = build_airfoil_v8_stage_b_benchmark(source.benchmark)
    ids, memory = _clone_source_memory(source)
    promotion_evidence = authenticate_airfoil_v10_promotion_evidence(
        promotion_evidence_root
    )
    other_card_promotion_evidence = (
        authenticate_airfoil_v10_other_card_promotion_evidence(
            other_card_promotion_evidence_root
        )
    )
    replayed_learned = replay_canonical_g3_adaptive_revision(
        memory,
        trace_path=revision_trace_path,
    )
    if (
        replayed_learned.lifecycle_state
        is not InsightLifecycleState.QUARANTINED
        or replayed_learned.retrievable
    ):
        raise AirfoilV10MultiOptionInputError(
            "canonical learned v2 did not enter quarantine before promotion"
        )
    learned = memory.promote(
        replayed_learned.reference,
        reason=AIRFOIL_V10_PROMOTION_REASON,
        supporting_evidence=AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE,
    )
    if (
        learned.lifecycle_state is not InsightLifecycleState.PROMOTED
        or not learned.retrievable
    ):
        raise AirfoilV10MultiOptionInputError(
            "authenticated learned v2 promotion did not become retrievable"
        )
    other_cards = tuple(
        entry
        for entry in source.active_entries
        if entry.reference.insight_id != learned.reference.insight_id
    )
    if len(other_cards) != 1:
        raise AirfoilV10MultiOptionInputError(
            "canonical learned v2 card has no unique other active v1 card"
        )
    source_other = other_cards[0]
    if (
        source_other.reference.version
        != AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION
        or source_other.lifecycle_state is not InsightLifecycleState.QUARANTINED
        or source_other.retrievable
        or source_other.applicable_operator_kinds != ("mutation",)
    ):
        raise AirfoilV10MultiOptionInputError(
            "legacy other card does not have the frozen v1 metadata"
        )
    migrated_other = memory.add_revision(
        source_other.reference,
        source_other.draft,
        initial_score=source_other.initial_score,
        applicable_operator_kinds=(
            AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
        ),
        origin=InsightOrigin.MANUAL,
        revision_note=AIRFOIL_V10_OTHER_CARD_REVISION_NOTE,
    )
    if (
        migrated_other.reference.insight_id
        != source_other.reference.insight_id
        or migrated_other.reference.version != AIRFOIL_V10_OTHER_CARD_VERSION
        or migrated_other.draft != source_other.draft
        or migrated_other.draft.content_sha256
        != source_other.draft.content_sha256
        or migrated_other.applicable_operator_kinds
        != AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS
        or migrated_other.origin is not InsightOrigin.MANUAL
        or migrated_other.lifecycle_state
        is not InsightLifecycleState.QUARANTINED
        or migrated_other.retrievable
    ):
        raise AirfoilV10MultiOptionInputError(
            "other-card applicability migration changed scientific content"
        )
    other = memory.promote(
        migrated_other.reference,
        reason=AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON,
        supporting_evidence=(
            AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE
        ),
    )
    if (
        other.lifecycle_state is not InsightLifecycleState.PROMOTED
        or not other.retrievable
    ):
        raise AirfoilV10MultiOptionInputError(
            "authenticated other-card promotion did not become retrievable"
        )
    active_cards = (learned, other)
    seeds = (
        AirfoilV10G0Seed(
            role=AIRFOIL_V10_SEED_ROLES[0],
            configuration=source.preparation.diagnostic_parent.candidate.configuration,
            configuration_sha256=(
                source.preparation.diagnostic_parent.candidate.configuration_sha256
            ),
            context_projection_sha256=AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
        ),
        AirfoilV10G0Seed(
            role=AIRFOIL_V10_SEED_ROLES[1],
            configuration=source.preparation.heldout_parent.candidate.configuration,
            configuration_sha256=(
                source.preparation.heldout_parent.candidate.configuration_sha256
            ),
            context_projection_sha256=AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
        ),
    )
    card_by_role = dict(zip(AIRFOIL_V10_CARD_ROLES, active_cards, strict=True))
    seed_by_role = {seed.role: seed for seed in seeds}
    bindings: list[AirfoilV10AuthorityBinding] = []
    for (seed_role, card_role), probe_id in zip(
        AIRFOIL_V10_AUTHORITY_MATRIX_ORDER,
        AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS,
        strict=True,
    ):
        seed = seed_by_role[seed_role]
        card = card_by_role[card_role]
        compiled = benchmark.compile_registered_hypothesis_treatment(
            catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
            parent_candidate_id=probe_id,
            parent_configuration=seed.configuration,
            entry=card,
            requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
            context_projection_sha256=seed.context_projection_sha256,
            endpoint_definition_sha256=ABSOLUTE_Q_DEFINITION_SHA256,
        )
        authority, _ = benchmark.compile_finite_action_set(
            compiled_anchor=compiled,
            required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        )
        bindings.append(
            AirfoilV10AuthorityBinding(
                seed_role=seed_role,
                card_role=card_role,
                seed_configuration_sha256=seed.configuration_sha256,
                card=card,
                probe_candidate_id=probe_id,
                compiled_treatment=compiled,
                authority=authority,
            )
        )
    heldout_union = benchmark.bind_finite_variation(
        AIRFOIL_G3_MODEL_CATALOG_ID,
        seeds[1].configuration,
    )
    source.mate_choice.validate_contract(heldout_union)
    mate_configuration = heldout_union.resolve(
        source.mate_choice.option_id
    ).child_configuration
    if type(mate_configuration) is not FrozenJsonObject:
        raise AirfoilV10MultiOptionInputError("mate configuration is not an object")
    return AirfoilV10MultiOptionInputs(
        benchmark=benchmark,
        id_factory=ids,
        memory=memory,
        promotion_evidence=promotion_evidence,
        other_card_promotion_evidence=other_card_promotion_evidence,
        active_cards=active_cards,
        g0_seeds=seeds,
        authority_bindings=tuple(bindings),  # type: ignore[arg-type]
        mate_choice=source.mate_choice,
        mate_configuration=mate_configuration,
        source_runtime_inputs_sha256=source.runtime_inputs_sha256,
        source_release_sha256=source.preparation.release_sha256,
    )


def load_frozen_airfoil_v10_multi_option_inputs(
    *,
    problem: object,
) -> AirfoilV10MultiOptionInputs:
    """Load frozen G3 inputs and compose v10 without provider/evaluator use."""

    source = load_frozen_airfoil_g3_runtime_inputs(problem=problem)
    return compose_airfoil_v10_multi_option_inputs(source)


def airfoil_v10_multi_option_readiness_record(
    inputs: AirfoilV10MultiOptionInputs,
) -> dict[str, object]:
    """Return the complete provider/CFD-free input and authority preflight."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    return {
        "schema_version": 1,
        "ready": True,
        "claim_boundary": {
            "provider_called": False,
            "credentials_read": False,
            "physical_evaluator_called": False,
            "scientific_result_eligible": False,
            "meaning": "input and authority preflight only",
        },
        "definition_sha256": AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
        "task_sha256": inputs.task_sha256,
        "schedule_sha256": inputs.schedule_sha256,
        "pre_outcome_commit_sha256": inputs.pre_outcome_commit_sha256,
        "phase": inputs.phase,
        "context_projection_sha256": inputs.context_projection_sha256,
        "inputs_sha256": inputs.inputs_sha256,
        "source_runtime_inputs_sha256": inputs.source_runtime_inputs_sha256,
        "source_release_sha256": inputs.source_release_sha256,
        "g0_seeds": [seed.to_record() for seed in inputs.g0_seeds],
        "active_cards": [
            {
                "role": role,
                "insight_id": card.reference.insight_id.value,
                "version": card.reference.version,
                "content_sha256": card.draft.content_sha256,
                "applicable_operator_kinds": list(
                    card.applicable_operator_kinds
                ),
                "origin": card.origin.value,
                "lifecycle_state": card.lifecycle_state.value,
                "retrievable": card.retrievable,
            }
            for role, card in zip(
                AIRFOIL_V10_CARD_ROLES,
                inputs.active_cards,
                strict=True,
            )
        ],
        "learned_v2_promotion": {
            "evidence_definition_sha256": (
                AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256
            ),
            "evidence_set_sha256": inputs.promotion_evidence_set_sha256,
            "evidence": [
                record.to_record() for record in inputs.promotion_evidence
            ],
            "transition": inputs._promotion_transition_record(),
            "all_results_adaptive_beats_uniform": True,
            "fresh_transfer_result_count": 2,
            "development_evidence_only": True,
            "paper_evidence_eligible": False,
        },
        "other_migrated_v2_promotion": {
            "evidence_definition_sha256": (
                AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256
            ),
            "evidence_record_sha256": (
                AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256
            ),
            "evidence": thaw_json(inputs.other_card_promotion_evidence),
            "transition": inputs._promotion_transition_record(1),
            "claim": "tested_retrievable_only",
            "efficacy_claim": False,
            "paper_evidence_eligible": False,
        },
        "authority_matrix": [
            binding.to_record() for binding in inputs.authority_bindings
        ],
        "distinctness": {
            "authority_count": len(inputs.authority_bindings),
            "support_cardinality_each": AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
            "globally_distinct_child_configurations": 32,
            "globally_distinct_phenotypes": 32,
            "within_seed_card_supports_disjoint": True,
        },
        "orthogonal_mate": {
            **inputs.mate_choice.to_record(),
            "choice_sha256": inputs.mate_choice.choice_sha256,
            "family": "shape_only",
            "configuration_sha256": typed_json_sha256(
                inputs.mate_configuration
            ),
        },
        "current_outcome_access": inputs.current_outcome_access,
        "provider": {
            **airfoil_v10_gpt_profile_config_record(),
            "profile_config_sha256": AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256,
        },
    }


__all__ = [
    "AIRFOIL_V10_AUTHORITY_MATRIX_ORDER",
    "AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS",
    "AIRFOIL_V10_CARD_ROLES",
    "AIRFOIL_V10_CONTEXT_PROJECTION_SHA256",
    "AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256",
    "AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256",
    "AIRFOIL_V10_MULTI_OPTION_PHASE",
    "AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256",
    "AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256",
    "AIRFOIL_V10_MULTI_OPTION_TASK_SHA256",
    "AIRFOIL_V10_OTHER_CARD_APPLICABLE_OPERATOR_KINDS",
    "AIRFOIL_V10_OTHER_CARD_CONTENT_SHA256",
    "AIRFOIL_V10_OTHER_CARD_INSIGHT_ID",
    "AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256",
    "AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT",
    "AIRFOIL_V10_PROMOTION_REASON",
    "AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE",
    "AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256",
    "AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256",
    "AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_ROOT",
    "AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON",
    "AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE",
    "AIRFOIL_V10_OTHER_CARD_REVISION_NOTE",
    "AIRFOIL_V10_OTHER_CARD_SOURCE_VERSION",
    "AIRFOIL_V10_OTHER_CARD_VERSION",
    "AIRFOIL_V10_RUN_SEED",
    "AIRFOIL_V10_SEED_ROLES",
    "AirfoilV10AuthorityBinding",
    "AirfoilV10G0Seed",
    "AirfoilV10MultiOptionInputError",
    "AirfoilV10MultiOptionInputs",
    "AirfoilV10PromotionEvidenceRecord",
    "authenticate_airfoil_v10_promotion_evidence",
    "authenticate_airfoil_v10_other_card_promotion_evidence",
    "airfoil_v10_gpt_profile_config_record",
    "airfoil_v10_multi_option_readiness_record",
    "build_airfoil_v10_gpt_openrouter_config",
    "compose_airfoil_v10_multi_option_inputs",
    "load_frozen_airfoil_v10_multi_option_inputs",
]
