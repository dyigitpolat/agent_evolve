"""Sealed BOiLS inputs shared by the budgeted-v5 composition root.

This module deliberately contains no generation planner, provider client, or
experiment runner.  It binds the exact development parent, a strict seed gate,
two matched evidence-card pairs, and a thin role decorator around the generic
AgentEvolve prompt.  The only performance observations in the card manifest
come from the named v1/v2 predecessor records.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import ClassVar

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    MutationResponseMode,
    PreparedInvocation,
    ProposalAuthority,
    default_evidence_prompt,
)
from agent_evolve.application.budgeted_optimizer import (
    SeedGateContext,
    SeedGateDecision,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import InsightDraft
from examples.benchmarks.boils_abc.actions import config_sha256


SUPPORT_SCHEMA_ID = "boils_abc_budgeted_v5_support_v2"
_HASH_DOMAIN = b"boils-abc:budgeted-v5-support:v2\x00"

REFERENCE_POINT: tuple[int, int] = (8_028, 71)

# Prospective attempt-4 mechanism identities. One raw unit is the smallest
# meaningful hypervolume quantum for BOiLS' exact integer objectives.
FRONT_EXTENSION_RAW_CREDIT = 1.0
FRONT_ALIGNED_REWARD_POLICY_ID = "boils_v5_frozen_hv_plus_front_extension"
FRONT_ALIGNED_REWARD_POLICY_VERSION = 1
FAILED_SLOT_CONTINUATION_POLICY_ID = "boils_v5_missing_candidate_no_substitution"
FAILED_SLOT_CONTINUATION_POLICY_VERSION = 1
BATCH_INCREMENTAL_COVERAGE_POLICY_ID = (
    "boils_v5_exploit_then_batch_incremental_coverage"
)
BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION = 1

PARENT_C_SEQUENCE: tuple[str, ...] = (
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "fraig",
    "rewrite_z",
    "balance",
    "refactor_z",
    "rewrite_z",
    "balance",
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "rewrite",
    "resub_z",
    "balance",
    "refactor_z",
    "rewrite_z",
    "balance",
)
PARENT_C_CONFIGURATION: FrozenJsonObject = freeze_json(
    {"sequence": list(PARENT_C_SEQUENCE)}
)  # type: ignore[assignment]
PARENT_C_OBJECTIVES: tuple[tuple[str, float], ...] = (
    ("total_lut_count", 7_944.0),
    ("total_levels", 69.0),
)
PARENT_C_BOILS_CONFIGURATION_SHA256 = (
    "e954b02443e92dbed5cc7aa21b8d452531400017d602bf5dcdc938fb84e5237e"
)
PARENT_C_TYPED_JSON_SHA256 = (
    "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a"
)
PARENT_C_CONFIGURATION_ARTIFACT_SHA256 = (
    "78c782b594ec17b8bb0ef3471ae822ab7b85944321cce14d198979eec79f0a22"
)


def _sequence_path(index: int) -> JsonPath:
    return JsonPath((ObjectKey("sequence"), ArrayIndex(index)))


AREA_PATH_INDEX = 7
DEPTH_PATH_INDEX = 1
UNCERTAINTY_PATH_INDEX = 12
COVERAGE_PATH_INDEX = 18

AREA_PATH = _sequence_path(AREA_PATH_INDEX)
DEPTH_PATH = _sequence_path(DEPTH_PATH_INDEX)
UNCERTAINTY_PATH = _sequence_path(UNCERTAINTY_PATH_INDEX)
COVERAGE_PATH = _sequence_path(COVERAGE_PATH_INDEX)

AREA_PATH_TEXT = "$.sequence[7]"
DEPTH_PATH_TEXT = "$.sequence[1]"
UNCERTAINTY_PATH_TEXT = "$.sequence[12]"
COVERAGE_PATH_TEXT = "$.sequence[18]"

AREA_EVIDENCE_ACTION = "resub"
AREA_REQUIRED_ACTION = AREA_EVIDENCE_ACTION
DEPTH_REQUIRED_ACTION = "fraig"
DEPTH_TRANSFER_SOURCE_PATH_TEXT = "$.sequence[4]"
UNCERTAINTY_REQUIRED_ACTION = "dsdb"
UNCERTAINTY_REQUIRED_FAMILY = "gia_dsd_balance"
UNCERTAINTY_COVERAGE_OBLIGATION_ID = "boils_v5.g1.uncertainty.extended_family_coverage"
UNCERTAINTY_COVERAGE_OBLIGATION_VERSION = 1
UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE = (
    "one frozen decomposition-based extended family option in the "
    "epistemic-uncertainty palette; remaining capacity and order stay task-keyed"
)


def protocol_correction_record() -> dict[str, object]:
    """Disclose the development-only chronology of the corrected U palette."""

    return {
        "correction_id": (
            "boils_v5_u_extended_family_after_matched_support_inspection_v1"
        ),
        "trigger": "sealed_matched_random_support_median_equalled_maximum",
        "matched_random_support_source": "computed_from_sealed_local_oracle",
        "outcome_aware_design": True,
        "outcome_aware_after_sealed_distribution_inspection": True,
        "frozen_before_live_calls": True,
        "correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt": False,
        "development_only": True,
        "confirmatory": False,
        "required_action": UNCERTAINTY_REQUIRED_ACTION,
        "required_family": UNCERTAINTY_REQUIRED_FAMILY,
    }


# Counts are strictly predecessor-run exposures, not a quality table.  The
# planner converts these JSON-safe cells into its own policy-domain values.
PREORACLE_PATH_FAMILY_EXPOSURES: tuple[tuple[str, str, int], ...] = (
    (DEPTH_PATH_TEXT, "aig_refactor", 1),
    ("$.sequence[4]", "aig_functional_reduce", 1),
    (AREA_PATH_TEXT, "aig_resubstitute", 1),
    (UNCERTAINTY_PATH_TEXT, "aig_rewrite", 1),
    ("$.sequence[15]", "aig_resubstitute", 1),
    (COVERAGE_PATH_TEXT, "aig_rewrite", 1),
)

AREA_PHASE = "boils_v5.g1.area"
DEPTH_PHASE = "boils_v5.g1.depth"
UNCERTAINTY_PHASE = "boils_v5.g1.uncertainty"
ENGINE_RNG_SEED = 6
MEMORY_ASSIGNMENT_SEED_RATIONALE = (
    "first nonnegative seed satisfying one-per-pair and cross-pair position "
    "counterbalance"
)
MEMORY_SUBSET_SIZE = 1
MEMORY_EXPLORATION_PROBABILITY = Fraction(1, 1)

AREA_V2_CONTRAST_ID = "98c52b918c8dcafe432b257ebc3421034b4875aa511b91839fd7bddad2f78445"
V1_TRANSFER_PARENT_BOILS_SHA256 = (
    "2f1b2c40172a4dd83e8d056a2b6581948ea0983055fea63d930791108509eef4"
)
V1_TRANSFER_CHILD_BOILS_SHA256 = (
    "3c20d80b43bdf0e0842f8bc02d5739a156d14b110299806c18ceb0a58876b871"
)
V1_TRANSFER_EVALUATIONS_SHA256 = (
    "e18ac211243f0bcccae37281146e366492e40a452203acec4a00474d8153acd8"
)
V2_AREA_EVALUATIONS_SHA256 = (
    "36e5c216dcec9bd7d5d175207015fc6b1082e5826e52ef5b4c300da2de87d4d4"
)


class BoilsV5Role(str, Enum):
    AREA = "area_extreme"
    DEPTH = "depth_extreme"
    UNCERTAINTY = "epistemic_uncertainty"
    COVERAGE = "representation_coverage"


AREA_ROLE = BoilsV5Role.AREA
DEPTH_ROLE = BoilsV5Role.DEPTH
UNCERTAINTY_ROLE = BoilsV5Role.UNCERTAINTY
COVERAGE_ROLE = BoilsV5Role.COVERAGE

_ROLE_PATHS = {
    AREA_ROLE: AREA_PATH,
    DEPTH_ROLE: DEPTH_PATH,
    UNCERTAINTY_ROLE: UNCERTAINTY_PATH,
    COVERAGE_ROLE: COVERAGE_PATH,
}
_ROLE_PATH_TEXT = {
    AREA_ROLE: AREA_PATH_TEXT,
    DEPTH_ROLE: DEPTH_PATH_TEXT,
    UNCERTAINTY_ROLE: UNCERTAINTY_PATH_TEXT,
    COVERAGE_ROLE: COVERAGE_PATH_TEXT,
}
_MODEL_ROLE_INSTRUCTIONS = {
    AREA_ROLE: (
        "AREA EXTREME: seek a lower total_lut_count subject to the exact "
        "parent-C feasibility ceiling total_levels <= 69."
    ),
    DEPTH_ROLE: (
        "DEPTH EXTREME: seek a lower total_levels subject to the exact "
        "parent-C feasibility ceiling total_lut_count <= 7944."
    ),
    UNCERTAINTY_ROLE: (
        "EPISTEMIC UNCERTAINTY: all supplied options are preselected "
        "under-explored legal alternatives; choose the option with the "
        "clearest action-specific falsifiable local hypothesis."
    ),
}


AREA_REAL_CARD = InsightDraft(
    claim=(
        "A sealed v2 single-operation contrast found that replacing refactor_z "
        "with resub at $.sequence[7] reduced total_lut_count by 19 and left "
        "total_levels unchanged for exact parent C."
    ),
    trigger=(
        "When mutating $.sequence[7] of exact parent C, where the current "
        "action is refactor_z and resub is a legal supplied option."
    ),
    mechanism=(
        "Test resub as a falsifiable local re-substitution hypothesis; the "
        "single observed association need not transfer to another parent."
    ),
    affected_paths=(AREA_PATH_TEXT,),
    evidence_summary=(
        "V2 contrast 98c52b918c8dcafe432b257ebc3421034b4875aa511b91839fd7bddad2f78445: "
        "refactor_z to resub at $.sequence[7] changed LUTs 7944 to 7925 and "
        "levels 69 to 69."
    ),
    confidence=0.85,
    evidence_contrast_ids=(AREA_V2_CONTRAST_ID,),
)

AREA_PLACEBO_CARD = InsightDraft(
    claim=(
        "This matched control assigns one legal replacement test at "
        "$.sequence[7] and supplies no performance evidence for any action "
        "available in that slot."
    ),
    trigger=(
        "When mutating $.sequence[7] of exact parent C, where the current "
        "action is refactor_z and a finite legal palette is supplied."
    ),
    mechanism=(
        "Use a falsifiable local structure hypothesis from the supplied "
        "actions; their presentation order carries no quality information."
    ),
    affected_paths=(AREA_PATH_TEXT,),
    evidence_summary=(
        "Neutral matched-shape control: no evaluated contrast is supplied, "
        "and no replacement action is endorsed or discouraged."
    ),
    confidence=0.0,
    evidence_contrast_ids=(),
)

DEPTH_REAL_CARD = InsightDraft(
    claim=(
        "A sealed v1 positional-transfer observation found that rewrite to "
        "fraig at $.sequence[4] changed (LUTs, levels) from (8028,71) to "
        "(7952,69); no result is known for fraig at $.sequence[1]."
    ),
    trigger=(
        "When mutating $.sequence[1] of exact parent C, where the current "
        "action is rewrite and fraig is the required transfer test."
    ),
    mechanism=(
        "Test whether early functional reduction transfers across positions; "
        "position dependence makes fraig at the target a new hypothesis."
    ),
    affected_paths=(DEPTH_PATH_TEXT,),
    evidence_summary=(
        "V1 source arm changed only $.sequence[4] from rewrite to fraig, with "
        "delta (-76 LUTs,-2 levels). It is rationale, not outcome evidence, "
        "for $.sequence[1]."
    ),
    confidence=0.45,
    evidence_contrast_ids=(),
)

DEPTH_PLACEBO_CARD = InsightDraft(
    claim=(
        "This matched control assigns one legal replacement test at "
        "$.sequence[1] and supplies no performance evidence for any action "
        "available in that slot."
    ),
    trigger=(
        "When mutating $.sequence[1] of exact parent C, where the current "
        "action is rewrite and a finite legal palette is supplied."
    ),
    mechanism=(
        "Use a falsifiable local depth hypothesis from the supplied actions; "
        "their presentation order carries no quality information."
    ),
    affected_paths=(DEPTH_PATH_TEXT,),
    evidence_summary=(
        "Neutral matched-shape control: no evaluated contrast is supplied, "
        "and no replacement action is endorsed or discouraged."
    ),
    confidence=0.0,
    evidence_contrast_ids=(),
)


class CardTreatment(str, Enum):
    REAL = "sealed_predecessor_evidence"
    PLACEBO = "neutral_matched_placebo"


@dataclass(frozen=True, slots=True)
class InsightCardDefinition:
    """Stable treatment identity around one immutable ``InsightDraft``."""

    card_id: str
    role: BoilsV5Role
    treatment: CardTreatment
    draft: InsightDraft

    def __post_init__(self) -> None:
        if (
            type(self.card_id) is not str
            or not self.card_id
            or self.card_id != self.card_id.strip()
        ):
            raise ValueError("card_id must be canonical non-empty text")
        if type(self.role) is not BoilsV5Role or self.role not in {
            AREA_ROLE,
            DEPTH_ROLE,
        }:
            raise ValueError("cards are defined only for area and depth roles")
        if type(self.treatment) is not CardTreatment:
            raise TypeError("treatment must be a CardTreatment")
        if type(self.draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(self.draft)
        expected_path = _ROLE_PATH_TEXT[self.role]
        if self.draft.affected_paths != (expected_path,):
            raise ValueError("card affected_paths do not match its frozen role")

    def to_manifest_record(self) -> dict[str, object]:
        body = {
            "card_id": self.card_id,
            "role": self.role.value,
            "target_path": _ROLE_PATH_TEXT[self.role],
            "treatment": self.treatment.value,
            "draft": _insight_draft_record(self.draft),
        }
        return {
            **body,
            "card_sha256": _record_sha256("insight-card", body),
        }


AREA_CARD_PAIR = (
    InsightCardDefinition(
        "boils_v5.area.path7.real.v1",
        AREA_ROLE,
        CardTreatment.REAL,
        AREA_REAL_CARD,
    ),
    InsightCardDefinition(
        "boils_v5.area.path7.placebo.v1",
        AREA_ROLE,
        CardTreatment.PLACEBO,
        AREA_PLACEBO_CARD,
    ),
)
DEPTH_CARD_PAIR = (
    InsightCardDefinition(
        "boils_v5.depth.path1.transfer_real.v1",
        DEPTH_ROLE,
        CardTreatment.REAL,
        DEPTH_REAL_CARD,
    ),
    InsightCardDefinition(
        "boils_v5.depth.path1.placebo.v1",
        DEPTH_ROLE,
        CardTreatment.PLACEBO,
        DEPTH_PLACEBO_CARD,
    ),
)
INSIGHT_CARD_DEFINITIONS = (*AREA_CARD_PAIR, *DEPTH_CARD_PAIR)

EXPECTED_MEMORY_ASSIGNMENTS: tuple[tuple[str, str], ...] = (
    ("G1-A1", AREA_CARD_PAIR[0].card_id),
    ("G1-A2", AREA_CARD_PAIR[1].card_id),
    ("G1-D1", DEPTH_CARD_PAIR[1].card_id),
    ("G1-D2", DEPTH_CARD_PAIR[0].card_id),
)


@dataclass(frozen=True, slots=True)
class CardMemoryReferences:
    """Exact card-to-memory identities allocated by one deterministic factory."""

    entries: tuple[tuple[str, InsightRef], ...]

    def __post_init__(self) -> None:
        expected_ids = tuple(card.card_id for card in INSIGHT_CARD_DEFINITIONS)
        if (
            type(self.entries) is not tuple
            or tuple(card_id for card_id, _ in self.entries) != expected_ids
        ):
            raise ValueError("memory references must follow the card manifest order")
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not InsightRef
            for item in self.entries
        ):
            raise TypeError("entries must contain exact card-ID/reference pairs")
        references = tuple(reference for _, reference in self.entries)
        if len(set(references)) != len(references):
            raise ValueError("card memory references must be unique")

    def reference_for(self, card_id: str) -> InsightRef:
        if type(card_id) is not str:
            raise TypeError("card_id must be an exact string")
        matches = tuple(
            reference for current_id, reference in self.entries if current_id == card_id
        )
        if len(matches) != 1:
            raise ValueError("unknown card_id")
        return matches[0]

    def expected_slot_references(self) -> tuple[tuple[str, InsightRef], ...]:
        return tuple(
            (slot_id, self.reference_for(card_id))
            for slot_id, card_id in EXPECTED_MEMORY_ASSIGNMENTS
        )

    def to_manifest_record(self) -> dict[str, object]:
        body = {
            "engine_rng_seed": ENGINE_RNG_SEED,
            "engine_rng_seed_rationale": MEMORY_ASSIGNMENT_SEED_RATIONALE,
            "subset_size": MEMORY_SUBSET_SIZE,
            "exploration_probability": {
                "numerator": MEMORY_EXPLORATION_PROBABILITY.numerator,
                "denominator": MEMORY_EXPLORATION_PROBABILITY.denominator,
            },
            "card_references": [
                {
                    "card_id": card_id,
                    "insight_id": reference.insight_id.value,
                    "version": reference.version,
                }
                for card_id, reference in self.entries
            ],
            "expected_assignments": [
                {
                    "slot_id": slot_id,
                    "card_id": card_id,
                    "insight_id": self.reference_for(card_id).insight_id.value,
                    "version": self.reference_for(card_id).version,
                }
                for slot_id, card_id in EXPECTED_MEMORY_ASSIGNMENTS
            ],
        }
        return {
            **body,
            "manifest_sha256": _record_sha256("card-memory", body),
        }


def build_v5_insight_memory(
    id_factory: object,
) -> tuple[InsightMemoryBank, CardMemoryReferences]:
    """Build the four-card randomized memory with equal frozen priors."""

    memory = InsightMemoryBank(
        id_factory=id_factory,
        exploration_probability=MEMORY_EXPLORATION_PROBABILITY,
    )
    entries: list[tuple[str, InsightRef]] = []
    for definition in INSIGHT_CARD_DEFINITIONS:
        entry, added = memory.add(
            definition.draft,
            initial_score=0.0,
            applicable_operator_kinds=("typed_mutation",),
            origin=InsightOrigin.SEED,
        )
        if not added:
            raise RuntimeError("frozen card claims unexpectedly deduplicated")
        entries.append((definition.card_id, entry.reference))
    references = CardMemoryReferences(tuple(entries))

    # Fail closed if Python's task-keyed RNG behavior or insertion ordering no
    # longer produces the preregistered balanced assignments.
    rng = random.Random(ENGINE_RNG_SEED)
    for pair, expected in (
        (
            tuple(references.reference_for(card.card_id) for card in AREA_CARD_PAIR),
            EXPECTED_MEMORY_ASSIGNMENTS[:2],
        ),
        (
            tuple(references.reference_for(card.card_id) for card in DEPTH_CARD_PAIR),
            EXPECTED_MEMORY_ASSIGNMENTS[2:],
        ),
    ):
        for _, expected_card_id in expected:
            decision = memory.select(
                context_hash=hashlib.sha256(
                    b"boils-v5-memory-assignment-preflight"
                ).hexdigest(),
                subset_size=MEMORY_SUBSET_SIZE,
                rng=rng,
                exploration_probability=MEMORY_EXPLORATION_PROBABILITY,
                eligible_references=pair,
            )
            if decision.selected != (references.reference_for(expected_card_id),):
                raise RuntimeError("frozen balanced memory assignment changed")
    return memory, references


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _record_sha256(kind: str, value: object) -> str:
    if type(kind) is not str or not kind:
        raise ValueError("record kind must be non-empty")
    return hashlib.sha256(
        _HASH_DOMAIN + kind.encode("ascii") + b"\x00" + _canonical_json_bytes(value)
    ).hexdigest()


def _insight_draft_record(draft: InsightDraft) -> dict[str, object]:
    if type(draft) is not InsightDraft:
        raise TypeError("draft must be an exact InsightDraft")
    InsightDraft.__post_init__(draft)
    return {
        "claim": draft.claim,
        "trigger": draft.trigger,
        "mechanism": draft.mechanism,
        "affected_paths": list(draft.affected_paths),
        "evidence_summary": draft.evidence_summary,
        "confidence_hex": float(draft.confidence).hex(),
        "evidence_contrast_ids": list(draft.evidence_contrast_ids),
    }


@dataclass(frozen=True, slots=True)
class ExactCSeedAdmissionPolicy:
    """Admit only exact evaluated C and bind external evaluator provenance.

    The composition root computes and validates the evaluator provenance record,
    then injects its canonical digest here.  This policy makes that exact digest
    part of every seed-gate receipt; it does not reinterpret the external record.
    """

    evaluator_provenance_sha256: str

    policy_id: ClassVar[str] = "boils_abc_exact_c_seed"
    policy_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        require_sha256(
            self.evaluator_provenance_sha256,
            "evaluator_provenance_sha256",
        )

    def assess(
        self,
        candidate: EvolutionCandidate,
        context: SeedGateContext,
    ) -> SeedGateDecision:
        if type(candidate) is not EvolutionCandidate:
            raise TypeError("candidate must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(candidate)
        if type(context) is not SeedGateContext:
            raise TypeError("context must be an exact SeedGateContext")
        SeedGateContext.__post_init__(context)

        typed_hash = typed_json_sha256(candidate.configuration)
        artifact_hash = hashlib.sha256(
            canonical_typed_json_bytes(candidate.configuration)
        ).hexdigest()
        configuration = thaw_json(candidate.configuration)
        if type(configuration) is dict:
            try:
                boils_hash = config_sha256(configuration)
            except (TypeError, ValueError):
                boils_hash = "configuration-is-outside-boils-schema"
        else:  # pragma: no cover - EvolutionCandidate permits other JSON roots.
            boils_hash = "configuration-root-is-not-an-object"
        objectives = candidate.objective_map
        checks = {
            "boils_configuration_sha256": (
                boils_hash == PARENT_C_BOILS_CONFIGURATION_SHA256
            ),
            "candidate_configuration_artifact_sha256": (
                artifact_hash == PARENT_C_CONFIGURATION_ARTIFACT_SHA256
                and candidate.occurrence.configuration_artifact_hash
                == PARENT_C_CONFIGURATION_ARTIFACT_SHA256
            ),
            "candidate_typed_json_sha256": (
                typed_hash == PARENT_C_TYPED_JSON_SHA256
                and candidate.occurrence.configuration_hash
                == PARENT_C_TYPED_JSON_SHA256
            ),
            "generation_is_seed": (
                candidate.generation == 0
                and candidate.operator_kind is None
                and not candidate.parent_ids
                and candidate.occurrence.operator_invocation_id is None
            ),
            "objectives": (
                candidate.valid
                and candidate.operator_compliant
                and candidate.evidence_compliant
                and set(objectives) == {"total_lut_count", "total_levels"}
                and objectives.get("total_lut_count") == 7_944.0
                and objectives.get("total_levels") == 69.0
            ),
            "requested_configuration_sha256": (
                context.requested_configuration_hash == PARENT_C_TYPED_JSON_SHA256
            ),
            "single_physical_seed_evaluation": (
                context.unique_evaluations_after
                == context.unique_evaluations_before + 1
            ),
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        evidence = tuple(
            sorted(
                (
                    ("boils_configuration_sha256", boils_hash),
                    (
                        "candidate_configuration_artifact_sha256",
                        candidate.occurrence.configuration_artifact_hash,
                    ),
                    ("candidate_typed_json_sha256", typed_hash),
                    (
                        "evaluator_provenance_sha256",
                        self.evaluator_provenance_sha256,
                    ),
                    (
                        "requested_configuration_sha256",
                        context.requested_configuration_hash,
                    ),
                    (
                        "total_levels_hex",
                        float(objectives.get("total_levels", float("nan"))).hex(),
                    ),
                    (
                        "total_lut_count_hex",
                        float(objectives.get("total_lut_count", float("nan"))).hex(),
                    ),
                )
            )
        )
        return SeedGateDecision(
            admitted=not failed,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            reason=(
                "exact C identity, objective vector, physical evaluation, and "
                "injected evaluator provenance digest are bound"
                if not failed
                else "exact C gate failed: " + ",".join(failed)
            ),
            evidence=evidence,
        )

    def to_manifest_record(self) -> dict[str, object]:
        body = {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "evaluator_provenance_sha256": self.evaluator_provenance_sha256,
            "required_parent_boils_configuration_sha256": (
                PARENT_C_BOILS_CONFIGURATION_SHA256
            ),
            "required_parent_typed_json_sha256": PARENT_C_TYPED_JSON_SHA256,
            "required_parent_configuration_artifact_sha256": (
                PARENT_C_CONFIGURATION_ARTIFACT_SHA256
            ),
            "required_objectives": [list(item) for item in PARENT_C_OBJECTIVES],
        }
        return {
            **body,
            "manifest_sha256": _record_sha256("seed-admission-policy", body),
        }


@dataclass(frozen=True, slots=True)
class RolePromptBuild:
    """One decorated prompt plus its durable, content-addressed trace row."""

    role: BoilsV5Role
    target_path: str
    prompt: str
    base_prompt_sha256: str
    role_instruction_sha256: str
    decorated_prompt_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.role) is not BoilsV5Role
            or self.role not in _MODEL_ROLE_INSTRUCTIONS
        ):
            raise ValueError("prompt builds support only model-authored roles")
        if self.target_path != _ROLE_PATH_TEXT[self.role]:
            raise ValueError("prompt target path differs from role contract")
        if type(self.prompt) is not str or not self.prompt:
            raise ValueError("prompt must be non-empty")
        for name in (
            "base_prompt_sha256",
            "role_instruction_sha256",
            "decorated_prompt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != (
            self.decorated_prompt_sha256
        ):
            raise ValueError("decorated_prompt_sha256 does not identify prompt")

    def to_trace_record(self) -> dict[str, object]:
        body = {
            "prompt_policy_id": BoilsV5RolePromptDecorator.policy_id,
            "prompt_policy_version": BoilsV5RolePromptDecorator.policy_version,
            "role": self.role.value,
            "target_path": self.target_path,
            "base_prompt_sha256": self.base_prompt_sha256,
            "role_instruction_sha256": self.role_instruction_sha256,
            "decorated_prompt_sha256": self.decorated_prompt_sha256,
        }
        return {
            **body,
            "trace_sha256": _record_sha256("role-prompt-build", body),
        }


@dataclass(frozen=True, slots=True)
class BoilsV5RolePromptDecorator:
    """Wrap the generic prompt with one outcome-free portfolio role label."""

    role: BoilsV5Role

    policy_id: ClassVar[str] = "boils_abc_budgeted_v5_role_prompt"
    policy_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if (
            type(self.role) is not BoilsV5Role
            or self.role not in _MODEL_ROLE_INSTRUCTIONS
        ):
            raise ValueError("decorator role must be area, depth, or uncertainty")

    def build(
        self,
        problem_description: str,
        prepared: PreparedInvocation,
        selected_insights: tuple[dict[str, object], ...],
    ) -> RolePromptBuild:
        if type(prepared) is not PreparedInvocation:
            raise TypeError("prepared must be an exact PreparedInvocation")
        contract = prepared.plan.mutation_contract
        if (
            prepared.plan.mutation_response_mode
            is not MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
            or contract is None
            or contract.editable_paths != (_ROLE_PATHS[self.role],)
        ):
            raise ValueError("invocation does not match the role's atomic path")
        base = default_evidence_prompt(
            problem_description,
            prepared,
            selected_insights,
        )
        instruction = _MODEL_ROLE_INSTRUCTIONS[self.role]
        decoration = "\n".join(
            (
                "",
                "FROZEN PORTFOLIO ROLE",
                self.role.value,
                instruction,
                "This role label adds no benchmark measurement. Choose only "
                "from the provider-enforced replacement palette; its order is "
                "not a quality ranking.",
            )
        )
        prompt = base + decoration
        return RolePromptBuild(
            role=self.role,
            target_path=_ROLE_PATH_TEXT[self.role],
            prompt=prompt,
            base_prompt_sha256=hashlib.sha256(base.encode("utf-8")).hexdigest(),
            role_instruction_sha256=hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            decorated_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )

    def __call__(
        self,
        problem_description: str,
        prepared: PreparedInvocation,
        selected_insights: tuple[dict[str, object], ...],
    ) -> str:
        return self.build(problem_description, prepared, selected_insights).prompt

    def to_manifest_record(self) -> dict[str, object]:
        instruction = _MODEL_ROLE_INSTRUCTIONS[self.role]
        body = {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "role": self.role.value,
            "target_path": _ROLE_PATH_TEXT[self.role],
            "role_instruction": instruction,
            "role_instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "wrapped_prompt_policy": (
                "agent_evolve.application.agentic_evolution.default_evidence_prompt"
            ),
        }
        return {
            **body,
            "manifest_sha256": _record_sha256("role-prompt-policy", body),
        }


@dataclass(frozen=True, slots=True)
class RoutedPromptBuild:
    """Trace projection for one phase-routed prompt build."""

    phase: str
    proposal_authority: ProposalAuthority
    route: str
    role: BoilsV5Role | None
    target_path: str | None
    prompt: str
    base_prompt_sha256: str
    decorated_prompt_sha256: str

    def __post_init__(self) -> None:
        if type(self.phase) is not str or not self.phase:
            raise ValueError("phase must be non-empty")
        if type(self.proposal_authority) is not ProposalAuthority:
            raise TypeError("proposal_authority must be a ProposalAuthority")
        if type(self.route) is not str or not self.route:
            raise ValueError("route must be non-empty")
        if type(self.prompt) is not str or not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.role is None:
            if self.proposal_authority is ProposalAuthority.MODEL:
                raise ValueError("model-authored prompt builds require a role")
            if self.target_path is not None:
                raise ValueError("non-model default routes have no target path")
        elif (
            type(self.role) is not BoilsV5Role
            or self.target_path != _ROLE_PATH_TEXT[self.role]
        ):
            raise ValueError("routed role/path contract changed")
        for name in ("base_prompt_sha256", "decorated_prompt_sha256"):
            require_sha256(getattr(self, name), name)
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != (
            self.decorated_prompt_sha256
        ):
            raise ValueError("decorated prompt hash does not identify prompt")

    def to_trace_record(self) -> dict[str, object]:
        body = {
            "router_policy_id": BoilsV5RolePromptRouter.policy_id,
            "router_policy_version": BoilsV5RolePromptRouter.policy_version,
            "phase": self.phase,
            "proposal_authority": self.proposal_authority.value,
            "route": self.route,
            "role": None if self.role is None else self.role.value,
            "target_path": self.target_path,
            "base_prompt_sha256": self.base_prompt_sha256,
            "decorated_prompt_sha256": self.decorated_prompt_sha256,
        }
        return {
            **body,
            "trace_sha256": _record_sha256("routed-prompt-build", body),
        }


@dataclass(frozen=True, slots=True)
class BoilsV5RolePromptRouter:
    """Single engine prompt policy dispatching exact model-authored phases."""

    policy_id: ClassVar[str] = "boils_abc_budgeted_v5_role_prompt_router"
    policy_version: ClassVar[int] = 1

    def build(
        self,
        problem_description: str,
        prepared: PreparedInvocation,
        selected_insights: tuple[dict[str, object], ...],
    ) -> RoutedPromptBuild:
        if type(prepared) is not PreparedInvocation:
            raise TypeError("prepared must be an exact PreparedInvocation")
        phase_roles = {
            AREA_PHASE: AREA_ROLE,
            DEPTH_PHASE: DEPTH_ROLE,
            UNCERTAINTY_PHASE: UNCERTAINTY_ROLE,
        }
        base = default_evidence_prompt(
            problem_description,
            prepared,
            selected_insights,
        )
        base_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
        if prepared.proposal_authority is ProposalAuthority.MODEL:
            try:
                role = phase_roles[prepared.plan.phase]
            except KeyError as exc:
                raise ValueError(
                    "unknown model-authored BOiLS v5 phase: " + prepared.plan.phase
                ) from exc
            role_build = BoilsV5RolePromptDecorator(role).build(
                problem_description,
                prepared,
                selected_insights,
            )
            if role_build.base_prompt_sha256 != base_hash:
                raise RuntimeError("role decorator did not wrap the generic prompt")
            return RoutedPromptBuild(
                phase=prepared.plan.phase,
                proposal_authority=prepared.proposal_authority,
                route=f"role:{role.value}",
                role=role,
                target_path=_ROLE_PATH_TEXT[role],
                prompt=role_build.prompt,
                base_prompt_sha256=base_hash,
                decorated_prompt_sha256=role_build.decorated_prompt_sha256,
            )

        return RoutedPromptBuild(
            phase=prepared.plan.phase,
            proposal_authority=prepared.proposal_authority,
            route="default_non_model",
            role=None,
            target_path=None,
            prompt=base,
            base_prompt_sha256=base_hash,
            decorated_prompt_sha256=base_hash,
        )

    def __call__(
        self,
        problem_description: str,
        prepared: PreparedInvocation,
        selected_insights: tuple[dict[str, object], ...],
    ) -> str:
        return self.build(problem_description, prepared, selected_insights).prompt

    def to_manifest_record(self) -> dict[str, object]:
        body = {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "model_phase_routes": [
                {"phase": phase, "role": role.value}
                for phase, role in (
                    (AREA_PHASE, AREA_ROLE),
                    (DEPTH_PHASE, DEPTH_ROLE),
                    (UNCERTAINTY_PHASE, UNCERTAINTY_ROLE),
                )
            ],
            "unknown_model_phase_policy": "reject",
            "non_model_route": "default_evidence_prompt",
        }
        return {
            **body,
            "manifest_sha256": _record_sha256("role-prompt-router", body),
        }


def parent_c_config() -> dict[str, object]:
    """Return a fresh mutable interchange copy of exact C."""

    result = thaw_json(PARENT_C_CONFIGURATION)
    if type(result) is not dict:  # pragma: no cover - import invariant.
        raise RuntimeError("parent C root changed")
    return result


def card_manifest_record() -> dict[str, object]:
    """Return the stable JSON-safe evidence/control manifest."""

    body = {
        "schema_id": SUPPORT_SCHEMA_ID,
        "evidence_boundary": "named v1/v2 predecessor records only",
        "sources": {
            "area_v2": {
                "contrast_id": AREA_V2_CONTRAST_ID,
                "evaluations_sha256": V2_AREA_EVALUATIONS_SHA256,
            },
            "depth_v1_positional_transfer": {
                "source_path": DEPTH_TRANSFER_SOURCE_PATH_TEXT,
                "target_path": DEPTH_PATH_TEXT,
                "required_target_action": DEPTH_REQUIRED_ACTION,
                "source_parent_boils_sha256": V1_TRANSFER_PARENT_BOILS_SHA256,
                "source_child_boils_sha256": V1_TRANSFER_CHILD_BOILS_SHA256,
                "evaluations_sha256": V1_TRANSFER_EVALUATIONS_SHA256,
                "target_result_known": False,
            },
        },
        "cards": [
            definition.to_manifest_record() for definition in INSIGHT_CARD_DEFINITIONS
        ],
    }
    return {
        **body,
        "manifest_sha256": _record_sha256("card-manifest", body),
    }


def support_manifest_record(
    evaluator_provenance_sha256: str,
) -> dict[str, object]:
    """Return one launch-ready support manifest bound to evaluator identity."""

    gate = ExactCSeedAdmissionPolicy(evaluator_provenance_sha256)
    body = {
        "schema_id": SUPPORT_SCHEMA_ID,
        "reference_point": list(REFERENCE_POINT),
        "parent_c": {
            "configuration": parent_c_config(),
            "boils_configuration_sha256": PARENT_C_BOILS_CONFIGURATION_SHA256,
            "typed_json_sha256": PARENT_C_TYPED_JSON_SHA256,
            "configuration_artifact_sha256": (PARENT_C_CONFIGURATION_ARTIFACT_SHA256),
            "objectives": [list(item) for item in PARENT_C_OBJECTIVES],
        },
        "role_paths": {role.value: _ROLE_PATH_TEXT[role] for role in BoilsV5Role},
        "predecessor_path_family_exposures": [
            {
                "path": path,
                "family": family,
                "exposure_count": count,
            }
            for path, family, count in PREORACLE_PATH_FAMILY_EXPOSURES
        ],
        "uncertainty_palette_obligation": {
            "obligation_id": UNCERTAINTY_COVERAGE_OBLIGATION_ID,
            "obligation_version": UNCERTAINTY_COVERAGE_OBLIGATION_VERSION,
            "path": UNCERTAINTY_PATH_TEXT,
            "required_action": UNCERTAINTY_REQUIRED_ACTION,
            "required_family": UNCERTAINTY_REQUIRED_FAMILY,
            "rationale": UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE,
        },
        "post_hoc_development_protocol_correction": True,
        "protocol_correction": protocol_correction_record(),
        "memory_assignment_policy": {
            "engine_rng_seed": ENGINE_RNG_SEED,
            "engine_rng_seed_rationale": MEMORY_ASSIGNMENT_SEED_RATIONALE,
            "subset_size": MEMORY_SUBSET_SIZE,
            "exploration_probability": {
                "numerator": MEMORY_EXPLORATION_PROBABILITY.numerator,
                "denominator": MEMORY_EXPLORATION_PROBABILITY.denominator,
            },
            "expected_assignments": [
                {"slot_id": slot_id, "card_id": card_id}
                for slot_id, card_id in EXPECTED_MEMORY_ASSIGNMENTS
            ],
        },
        "card_manifest": card_manifest_record(),
        "seed_admission_policy": gate.to_manifest_record(),
        "role_prompt_router": BoilsV5RolePromptRouter().to_manifest_record(),
        "model_role_prompt_policies": [
            BoilsV5RolePromptDecorator(role).to_manifest_record()
            for role in (AREA_ROLE, DEPTH_ROLE, UNCERTAINTY_ROLE)
        ],
    }
    return {
        **body,
        "manifest_sha256": _record_sha256("support-manifest", body),
    }


if type(PARENT_C_CONFIGURATION) is not FrozenJsonObject:  # pragma: no cover.
    raise RuntimeError("parent C must freeze to an object")
if config_sha256(parent_c_config()) != PARENT_C_BOILS_CONFIGURATION_SHA256:
    raise RuntimeError("parent C BOiLS hash constant changed")
if typed_json_sha256(PARENT_C_CONFIGURATION) != PARENT_C_TYPED_JSON_SHA256:
    raise RuntimeError("parent C typed-JSON hash constant changed")
if (
    hashlib.sha256(canonical_typed_json_bytes(PARENT_C_CONFIGURATION)).hexdigest()
    != PARENT_C_CONFIGURATION_ARTIFACT_SHA256
):
    raise RuntimeError("parent C artifact hash constant changed")


__all__ = [
    "AREA_CARD_PAIR",
    "AREA_EVIDENCE_ACTION",
    "AREA_PHASE",
    "AREA_PATH",
    "AREA_PATH_INDEX",
    "AREA_PATH_TEXT",
    "AREA_PLACEBO_CARD",
    "AREA_REAL_CARD",
    "AREA_REQUIRED_ACTION",
    "AREA_ROLE",
    "AREA_V2_CONTRAST_ID",
    "BoilsV5Role",
    "BoilsV5RolePromptDecorator",
    "BoilsV5RolePromptRouter",
    "BATCH_INCREMENTAL_COVERAGE_POLICY_ID",
    "BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION",
    "COVERAGE_PATH",
    "COVERAGE_PATH_INDEX",
    "COVERAGE_PATH_TEXT",
    "COVERAGE_ROLE",
    "CardMemoryReferences",
    "CardTreatment",
    "DEPTH_CARD_PAIR",
    "DEPTH_PHASE",
    "DEPTH_PATH",
    "DEPTH_PATH_INDEX",
    "DEPTH_PATH_TEXT",
    "DEPTH_PLACEBO_CARD",
    "DEPTH_REAL_CARD",
    "DEPTH_REQUIRED_ACTION",
    "DEPTH_ROLE",
    "DEPTH_TRANSFER_SOURCE_PATH_TEXT",
    "ENGINE_RNG_SEED",
    "EXPECTED_MEMORY_ASSIGNMENTS",
    "ExactCSeedAdmissionPolicy",
    "FAILED_SLOT_CONTINUATION_POLICY_ID",
    "FAILED_SLOT_CONTINUATION_POLICY_VERSION",
    "FRONT_ALIGNED_REWARD_POLICY_ID",
    "FRONT_ALIGNED_REWARD_POLICY_VERSION",
    "FRONT_EXTENSION_RAW_CREDIT",
    "INSIGHT_CARD_DEFINITIONS",
    "InsightCardDefinition",
    "MEMORY_EXPLORATION_PROBABILITY",
    "MEMORY_ASSIGNMENT_SEED_RATIONALE",
    "MEMORY_SUBSET_SIZE",
    "PARENT_C_BOILS_CONFIGURATION_SHA256",
    "PARENT_C_CONFIGURATION",
    "PARENT_C_CONFIGURATION_ARTIFACT_SHA256",
    "PARENT_C_OBJECTIVES",
    "PARENT_C_SEQUENCE",
    "PARENT_C_TYPED_JSON_SHA256",
    "PREORACLE_PATH_FAMILY_EXPOSURES",
    "REFERENCE_POINT",
    "RolePromptBuild",
    "RoutedPromptBuild",
    "SUPPORT_SCHEMA_ID",
    "UNCERTAINTY_PHASE",
    "UNCERTAINTY_COVERAGE_OBLIGATION_ID",
    "UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE",
    "UNCERTAINTY_COVERAGE_OBLIGATION_VERSION",
    "UNCERTAINTY_PATH",
    "UNCERTAINTY_PATH_INDEX",
    "UNCERTAINTY_PATH_TEXT",
    "UNCERTAINTY_REQUIRED_ACTION",
    "UNCERTAINTY_REQUIRED_FAMILY",
    "UNCERTAINTY_ROLE",
    "build_v5_insight_memory",
    "card_manifest_record",
    "parent_c_config",
    "protocol_correction_record",
    "support_manifest_record",
]
