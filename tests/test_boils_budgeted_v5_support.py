"""Offline integrity tests for the narrow BOiLS budgeted-v5 support layer."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction

import pytest

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
    default_evidence_prompt,
)
from agent_evolve.application.budgeted_optimizer import SeedGateContext
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import InsightDraft
from examples.benchmarks.boils_abc.actions import config_sha256
from examples.benchmarks.boils_abc import budgeted_v5_support as support


def _artifact_hash(configuration) -> str:
    return hashlib.sha256(canonical_typed_json_bytes(configuration)).hexdigest()


def _candidate(
    config: dict[str, object] | None = None,
    *,
    objectives: tuple[tuple[str, float], ...] = support.PARENT_C_OBJECTIVES,
    valid: bool = True,
) -> EvolutionCandidate:
    frozen = freeze_json(support.parent_c_config() if config is None else config)
    digest = typed_json_sha256(frozen)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId("candidate_boils_v5_support_seed"),
            configuration_hash=digest,
            configuration_artifact_hash=_artifact_hash(frozen),
            proposal_sequence=1,
        ),
        configuration=frozen,
        objectives=objectives if valid else (),
        valid=valid,
        generation=0,
        label="seed_0",
    )


def _context(
    *,
    requested_hash: str = support.PARENT_C_TYPED_JSON_SHA256,
    before: int = 0,
    after: int = 1,
) -> SeedGateContext:
    return SeedGateContext(
        seed_index=0,
        label="seed_0",
        requested_configuration_hash=requested_hash,
        unique_evaluations_before=before,
        unique_evaluations_after=after,
    )


def test_parent_c_reference_and_all_three_hash_boundaries_are_exact() -> None:
    config = support.parent_c_config()
    frozen = freeze_json(config)

    assert support.REFERENCE_POINT == (8_028, 71)
    assert support.PARENT_C_OBJECTIVES == (
        ("total_lut_count", 7_944.0),
        ("total_levels", 69.0),
    )
    assert config_sha256(config) == support.PARENT_C_BOILS_CONFIGURATION_SHA256
    assert typed_json_sha256(frozen) == support.PARENT_C_TYPED_JSON_SHA256
    assert _artifact_hash(frozen) == support.PARENT_C_CONFIGURATION_ARTIFACT_SHA256

    config["sequence"][0] = "fraig"  # type: ignore[index]
    assert support.parent_c_config()["sequence"][0] == "balance"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        support.PARENT_C_CONFIGURATION.items = ()  # type: ignore[misc]


def test_exact_c_seed_gate_binds_external_provenance_and_admits_only_one_eval() -> None:
    provenance_hash = hashlib.sha256(b"boils-v5-test-provenance").hexdigest()
    policy = support.ExactCSeedAdmissionPolicy(provenance_hash)
    accepted = policy.assess(_candidate(), _context())

    assert accepted.admitted is True
    assert accepted.policy_id == "boils_abc_exact_c_seed"
    assert dict(accepted.evidence)["evaluator_provenance_sha256"] == provenance_hash
    assert dict(accepted.evidence)["boils_configuration_sha256"] == (
        support.PARENT_C_BOILS_CONFIGURATION_SHA256
    )
    assert accepted.to_trace_record()["decision_hash"] == (
        "bbea01dec66d4b895ab3eeda34f5adff78a75a5572fad30852ca33a43f12b17a"
    )

    changed = support.parent_c_config()
    changed["sequence"][0] = "rewrite"  # type: ignore[index]
    cases = (
        policy.assess(_candidate(changed), _context()),
        policy.assess(
            _candidate(
                objectives=(
                    ("total_lut_count", 7_943.0),
                    ("total_levels", 69.0),
                )
            ),
            _context(),
        ),
        policy.assess(_candidate(), _context(after=0)),
        policy.assess(
            _candidate(),
            _context(requested_hash=hashlib.sha256(b"wrong-parent").hexdigest()),
        ),
    )
    assert all(decision.admitted is False for decision in cases)
    assert "boils_configuration_sha256" in cases[0].reason
    assert "objectives" in cases[1].reason
    assert "single_physical_seed_evaluation" in cases[2].reason
    assert "requested_configuration_sha256" in cases[3].reason

    with pytest.raises(ValueError, match="SHA-256"):
        support.ExactCSeedAdmissionPolicy("not-a-hash")


def _draft_shape_size(draft: InsightDraft) -> int:
    return sum(
        len(getattr(draft, name))
        for name in ("claim", "trigger", "mechanism", "evidence_summary")
    )


def test_real_and_placebo_cards_are_immutable_path_exact_and_shape_matched() -> None:
    assert support.AREA_REAL_CARD.affected_paths == (support.AREA_PATH_TEXT,)
    assert support.AREA_PLACEBO_CARD.affected_paths == (support.AREA_PATH_TEXT,)
    assert support.DEPTH_REAL_CARD.affected_paths == (support.DEPTH_PATH_TEXT,)
    assert support.DEPTH_PLACEBO_CARD.affected_paths == (support.DEPTH_PATH_TEXT,)
    assert support.AREA_REAL_CARD.evidence_contrast_ids == (
        support.AREA_V2_CONTRAST_ID,
    )
    assert support.AREA_PLACEBO_CARD.evidence_contrast_ids == ()
    assert support.DEPTH_REAL_CARD.evidence_contrast_ids == ()
    assert support.DEPTH_PLACEBO_CARD.evidence_contrast_ids == ()
    assert "resub" in support.AREA_REAL_CARD.claim
    assert support.DEPTH_REQUIRED_ACTION == "fraig"
    assert support.DEPTH_TRANSFER_SOURCE_PATH_TEXT in support.DEPTH_REAL_CARD.claim
    assert support.DEPTH_PATH_TEXT in support.DEPTH_REAL_CARD.claim
    assert "no result is known" in support.DEPTH_REAL_CARD.claim

    schema = tuple(field.name for field in fields(InsightDraft))
    for real, placebo in (
        (support.AREA_REAL_CARD, support.AREA_PLACEBO_CARD),
        (support.DEPTH_REAL_CARD, support.DEPTH_PLACEBO_CARD),
    ):
        assert tuple(field.name for field in fields(type(real))) == schema
        assert tuple(field.name for field in fields(type(placebo))) == schema
        sizes = (_draft_shape_size(real), _draft_shape_size(placebo))
        assert max(sizes) / min(sizes) < 1.6

    with pytest.raises(FrozenInstanceError):
        support.AREA_REAL_CARD.claim = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        support.AREA_CARD_PAIR[0].card_id = "changed"  # type: ignore[misc]


def test_support_contains_only_declared_predecessor_card_evidence() -> None:
    source = inspect.getsource(support)
    contrast_ids = {
        contrast_id
        for definition in support.INSIGHT_CARD_DEFINITIONS
        for contrast_id in definition.draft.evidence_contrast_ids
    }
    assert contrast_ids == {support.AREA_V2_CONTRAST_ID}
    assert (
        "c35ab6f3e49f2d9cfd1ebdeffd3df1999e39c9a196bd6985a211b75619b051f3" not in source
    )
    assert "boils_local_oracle" not in source
    assert "artifact63" not in source
    assert "artifact69" not in source
    assert support.card_manifest_record()["evidence_boundary"] == (
        "named v1/v2 predecessor records only"
    )
    transfer = support.card_manifest_record()["sources"]["depth_v1_positional_transfer"]
    assert transfer["target_result_known"] is False


def test_fixed_roles_paths_actions_and_predecessor_exposures_are_complete() -> None:
    assert (
        support.AREA_PATH_INDEX,
        support.DEPTH_PATH_INDEX,
        support.UNCERTAINTY_PATH_INDEX,
        support.COVERAGE_PATH_INDEX,
    ) == (7, 1, 12, 18)
    assert support.AREA_REQUIRED_ACTION == "resub"
    assert support.DEPTH_REQUIRED_ACTION == "fraig"
    assert support.PREORACLE_PATH_FAMILY_EXPOSURES == (
        ("$.sequence[1]", "aig_refactor", 1),
        ("$.sequence[4]", "aig_functional_reduce", 1),
        ("$.sequence[7]", "aig_resubstitute", 1),
        ("$.sequence[12]", "aig_rewrite", 1),
        ("$.sequence[15]", "aig_resubstitute", 1),
        ("$.sequence[18]", "aig_rewrite", 1),
    )
    assert (
        support.AREA_PHASE,
        support.DEPTH_PHASE,
        support.UNCERTAINTY_PHASE,
    ) == (
        "boils_v5.g1.area",
        "boils_v5.g1.depth",
        "boils_v5.g1.uncertainty",
    )


def test_memory_builder_freezes_balanced_real_placebo_assignment() -> None:
    memory, references = support.build_v5_insight_memory(
        DeterministicIdFactory("v5_support_probe")
    )
    assert isinstance(memory, InsightMemoryBank)
    assert support.ENGINE_RNG_SEED == 6
    assert support.MEMORY_ASSIGNMENT_SEED_RATIONALE == (
        "first nonnegative seed satisfying one-per-pair and cross-pair position "
        "counterbalance"
    )
    assert support.MEMORY_SUBSET_SIZE == 1
    assert support.MEMORY_EXPLORATION_PROBABILITY == Fraction(1, 1)
    assert tuple(card_id for card_id, _ in references.entries) == tuple(
        definition.card_id for definition in support.INSIGHT_CARD_DEFINITIONS
    )
    assert tuple(slot for slot, _ in references.expected_slot_references()) == (
        "G1-A1",
        "G1-A2",
        "G1-D1",
        "G1-D2",
    )
    assert tuple(
        reference.insight_id.value
        for _, reference in references.expected_slot_references()
    ) == (
        "insight_v5_support_probe_000001",
        "insight_v5_support_probe_000002",
        "insight_v5_support_probe_000004",
        "insight_v5_support_probe_000003",
    )

    expected = tuple(
        references.reference_for(card_id)
        for _, card_id in support.EXPECTED_MEMORY_ASSIGNMENTS
    )
    context_hash = hashlib.sha256(b"selector-minimality").hexdigest()

    def selections(seed: int):
        rng = random.Random(seed)
        result = []
        for pair in (support.AREA_CARD_PAIR, support.DEPTH_CARD_PAIR):
            eligible = tuple(references.reference_for(card.card_id) for card in pair)
            for _ in range(2):
                decision = memory.select(
                    context_hash=context_hash,
                    subset_size=1,
                    rng=rng,
                    exploration_probability=Fraction(1, 1),
                    eligible_references=eligible,
                )
                result.append(decision.selected[0])
        return tuple(result)

    def is_position_counterbalanced(values) -> bool:
        a1, a2, d1, d2 = values
        return a1 != a2 and d1 != d2 and (a1 == expected[0]) != (d1 == expected[3])

    assert selections(6) == expected
    assert is_position_counterbalanced(selections(6))
    assert all(not is_position_counterbalanced(selections(seed)) for seed in range(6))
    assert references.to_manifest_record()["manifest_sha256"] == (
        "eb7dbd687fdf2ec29082dd347530406f0aa3ceb6b1181004e1090d9832cbd872"
    )


class _PromptProblem:
    objectives = (
        ObjectiveSpec("total_lut_count", "min"),
        ObjectiveSpec("total_levels", "min"),
    )

    @staticmethod
    def search_space_description() -> str:
        return "Offline BOiLS prompt fixture."


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - prompt-only fake.
        raise AssertionError(request)

    async def reflect(self, request):  # pragma: no cover - prompt-only fake.
        raise AssertionError(request)


def _prepared(role: support.BoilsV5Role, *, phase: str | None = None):
    namespace = {
        support.AREA_ROLE: "bv5a",
        support.DEPTH_ROLE: "bv5d",
        support.UNCERTAINTY_ROLE: "bv5u",
    }[role]
    ids = DeterministicIdFactory(namespace)
    engine = AgenticEvolutionEngine(
        problem=_PromptProblem(),
        generator=_NeverGenerator(),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=1,
    )
    parent = _candidate()
    path = {
        support.AREA_ROLE: support.AREA_PATH,
        support.DEPTH_ROLE: support.DEPTH_PATH,
        support.UNCERTAINTY_ROLE: support.UNCERTAINTY_PATH,
    }[role]
    options = {
        support.AREA_ROLE: ("resub", "fraig", "balance"),
        support.DEPTH_ROLE: ("fraig", "resub", "balance"),
        support.UNCERTAINTY_ROLE: ("dsdb", "blut", "sopb"),
    }[role]
    role_phase = {
        support.AREA_ROLE: support.AREA_PHASE,
        support.DEPTH_ROLE: support.DEPTH_PHASE,
        support.UNCERTAINTY_ROLE: support.UNCERTAINTY_PHASE,
    }[role]
    plan = InvocationPlan(
        OperatorKind.TYPED_MUTATION,
        (parent,),
        generation=1,
        label=f"fixture_{role.value}",
        allowed_top_level=("sequence",),
        mutation_contract=MutationContract(
            (path,),
            max_changed_paths=1,
            max_operations=1,
            allow_abstention=False,
        ),
        mutation_response_mode=MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1,
        atomic_replacement_options=options,
        phase=role_phase if phase is None else phase,
    )
    return engine._prepare(plan), engine.problem_description


@pytest.mark.parametrize(
    "role",
    (support.AREA_ROLE, support.DEPTH_ROLE, support.UNCERTAINTY_ROLE),
)
def test_role_prompt_router_wraps_generic_prompt_and_traces_exact_route(role) -> None:
    prepared, description = _prepared(role)
    router = support.BoilsV5RolePromptRouter()
    built = router.build(description, prepared, ())
    base = default_evidence_prompt(description, prepared, ())

    assert built.prompt.startswith(base)
    assert "FROZEN PORTFOLIO ROLE" in built.prompt
    assert role.value in built.prompt
    assert "adds no benchmark measurement" in built.prompt
    if role is support.AREA_ROLE:
        assert "total_levels <= 69" in built.prompt
    if role is support.DEPTH_ROLE:
        assert "total_lut_count <= 7944" in built.prompt
    assert router(description, prepared, ()) == built.prompt
    trace = built.to_trace_record()
    assert trace["route"] == f"role:{role.value}"
    assert (
        trace["target_path"]
        == {
            support.AREA_ROLE: support.AREA_PATH_TEXT,
            support.DEPTH_ROLE: support.DEPTH_PATH_TEXT,
            support.UNCERTAINTY_ROLE: support.UNCERTAINTY_PATH_TEXT,
        }[role]
    )
    assert len(trace["trace_sha256"]) == 64


def test_prompt_router_rejects_unknown_model_phase_and_defaults_non_model() -> None:
    unknown, description = _prepared(support.AREA_ROLE, phase="boils_v5.unknown")
    router = support.BoilsV5RolePromptRouter()
    with pytest.raises(ValueError, match="unknown model-authored"):
        router.build(description, unknown, ())

    non_model = replace(
        unknown,
        proposal_authority=ProposalAuthority.ENGINE,
        call_id=None,
    )
    built = router.build(description, non_model, ())
    assert built.route == "default_non_model"
    assert built.role is None
    assert built.prompt == default_evidence_prompt(description, non_model, ())


def test_manifests_and_prompt_trace_records_have_fixed_replay_hashes() -> None:
    provenance_hash = hashlib.sha256(b"boils-v5-test-provenance").hexdigest()
    cards = support.card_manifest_record()
    manifest = support.support_manifest_record(provenance_hash)
    prepared, description = _prepared(support.AREA_ROLE)
    prompt_trace = (
        support.BoilsV5RolePromptRouter()
        .build(description, prepared, ())
        .to_trace_record()
    )

    json.dumps(cards, allow_nan=False, sort_keys=True)
    json.dumps(manifest, allow_nan=False, sort_keys=True)
    json.dumps(prompt_trace, allow_nan=False, sort_keys=True)
    assert cards["schema_id"] == support.SUPPORT_SCHEMA_ID
    assert manifest["schema_id"] == support.SUPPORT_SCHEMA_ID
    assert support.SUPPORT_SCHEMA_ID == "boils_abc_budgeted_v5_support_v2"
    assert cards["manifest_sha256"] == (
        "a83be1efd95be0a124464a0a34885098db66c0152600affa877b95ad3df7eaaa"
    )
    assert manifest["manifest_sha256"] == (
        "f27708253069b9ed4407ce50930e0925e1dad22c526c27984ecc3eeff1b943bf"
    )
    assert manifest["uncertainty_palette_obligation"] == {
        "obligation_id": support.UNCERTAINTY_COVERAGE_OBLIGATION_ID,
        "obligation_version": 1,
        "path": support.UNCERTAINTY_PATH_TEXT,
        "required_action": "dsdb",
        "required_family": "gia_dsd_balance",
        "rationale": support.UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE,
    }
    assert prompt_trace["trace_sha256"] == (
        "13f04ba7862ea8df56643a1e1bb6ce529fa3b1d8fac9f7eb1441086f404a7a4d"
    )
    assert support.support_manifest_record(provenance_hash) == manifest
