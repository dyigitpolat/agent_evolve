"""Thin Airfoil-v7 composition boundary for the generic G3 causal screen.

This module owns only benchmark wiring.  It does not read credentials, create a
provider client, call a model, or evaluate an Airfoil.  The generic engine,
optimizer, G3 planner, queue, and runner remain injected through public ports.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    OperatorKind,
)
from agent_evolve.application.g3_causal_screen import (
    G3CausalScreenPlanner,
    FrozenDiagnosticPermutation,
    ParentBoundActionChoice,
    PreparedHypothesisMatrix,
)
from agent_evolve.application.g3_postseal_curation import (
    G3CurationSourceScope,
    G3PostsealCurationFactory,
    G3PostsealCurationInterceptor,
    G3PostsealCurationSpec,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightMemoryEntry,
)
from agent_evolve.domain.ids import (
    CandidateId,
    CorrelationId,
    EvaluationAttemptId,
    EvaluationId,
    EventId,
    GenerationId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
    ProviderAttemptId,
    RunId,
)
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.id_factory import IdFactory
from agent_evolve.ports.agentic_generator import ReflectionInsightContract
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    AIRFOIL_G3_ABSOLUTE_REWARD,
    AIRFOIL_G3_RUNTIME_PHASE,
    AIRFOIL_G3_RUNTIME_PROBLEM_ID,
    DEFAULT_CARD_BANK_PATH,
    DEFAULT_DENYLIST_PATH,
    DEFAULT_RELEASE_PATH,
    AirfoilG3ReleaseError,
    AirfoilG3ReleasePreparation,
    AirfoilV7TrimHypothesisCompiler,
    build_hypothesis_compilation_request,
    freeze_diagnostic_permutation,
    load_prelaunch_freeze_receipt,
    prepare_release,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
)


AIRFOIL_G3_MODEL_CATALOG_ID = AirfoilV7UnionVariationCatalog.catalog_id
_CHOICE_POLICY_DOMAIN = b"agent-evolve:airfoil-g3-runtime-choice:v1\x00"
_ESTIMAND_DOMAIN = b"agent-evolve:airfoil-g3-transfer-estimand:v1\x00"
_RUNTIME_INPUT_DOMAIN = b"agent-evolve:airfoil-g3-runtime-inputs:v1\x00"
_CURATION_SCOPE_DEFINITION_SHA256 = hashlib.sha256(
    b"airfoil-v7-g3-curation-source:v1:direct-adaptive-g2-outcome-only;"
    b"exclude-composite-union-and-nonadaptive-controls"
).hexdigest()


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


_ESTIMAND_DEFINITION = {
    "estimand_id": "airfoil_v7_g3_absolute_q_card_transfer",
    "estimand_version": 1,
    "diagnostic_parent_role": "P_D",
    "heldout_parent_role": "P_H",
    "treatment": "one exact parent-bound trim option",
    "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
    "endpoint_is_absolute_and_parent_independent": True,
    "g1_identification": (
        "randomized matched two-slot pairwise ordering of H1@P_D versus H2@P_D; "
        "it does not identify either card's individual effect versus P_D"
    ),
    "g2_transfer_test": (
        "use the sealed G1 comparative ordering to allocate the same two exact "
        "card templates compiled on P_H"
    ),
    "sham": "evidence-free exact trim action with no directional claim",
    "orthogonal_mate": "engine-owned shape-only option",
    "current_run_outcome_access_during_preparation": False,
}
AIRFOIL_G3_ESTIMAND_STRATUM_SHA256 = _hash(
    _ESTIMAND_DOMAIN,
    _ESTIMAND_DEFINITION,
)


class FrozenPrefixIdFactory:
    """Serve exact preregistered insight IDs, then delegate every allocation."""

    def __init__(
        self,
        *,
        delegate: IdFactory,
        frozen_insight_ids: tuple[InsightId, ...],
    ) -> None:
        if not isinstance(delegate, IdFactory):
            raise TypeError("delegate must implement IdFactory")
        if (
            type(frozen_insight_ids) is not tuple
            or not frozen_insight_ids
            or any(type(value) is not InsightId for value in frozen_insight_ids)
            or len(set(frozen_insight_ids)) != len(frozen_insight_ids)
        ):
            raise ValueError("frozen insight IDs must be a unique non-empty tuple")
        self.delegate = delegate
        self.frozen_insight_ids = frozen_insight_ids
        self._insight_index = 0
        self._lock = threading.Lock()

    @property
    def frozen_insight_ids_consumed(self) -> int:
        with self._lock:
            return self._insight_index

    def new_insight_id(self) -> InsightId:
        with self._lock:
            if self._insight_index < len(self.frozen_insight_ids):
                value = self.frozen_insight_ids[self._insight_index]
                self._insight_index += 1
                return value
        return self.delegate.new_insight_id()

    def new_run_id(self) -> RunId:
        return self.delegate.new_run_id()

    def new_event_id(self) -> EventId:
        return self.delegate.new_event_id()

    def new_generation_id(self) -> GenerationId:
        return self.delegate.new_generation_id()

    def new_candidate_id(self) -> CandidateId:
        return self.delegate.new_candidate_id()

    def new_operator_invocation_id(self) -> OperatorInvocationId:
        return self.delegate.new_operator_invocation_id()

    def new_llm_call_id(self) -> LLMCallId:
        return self.delegate.new_llm_call_id()

    def new_provider_attempt_id(self) -> ProviderAttemptId:
        return self.delegate.new_provider_attempt_id()

    def new_evaluation_id(self) -> EvaluationId:
        return self.delegate.new_evaluation_id()

    def new_evaluation_attempt_id(self) -> EvaluationAttemptId:
        return self.delegate.new_evaluation_attempt_id()

    def new_correlation_id(self) -> CorrelationId:
        return self.delegate.new_correlation_id()


def _register_exact_entry(
    memory: InsightMemoryBank,
    expected: InsightMemoryEntry,
) -> InsightMemoryEntry:
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
        raise AirfoilG3ReleaseError(
            "runtime memory registration changed an exact frozen entry"
        )
    return observed


def _prepared_matrices(
    preparation: AirfoilG3ReleasePreparation,
    active_entries: tuple[InsightMemoryEntry, InsightMemoryEntry],
) -> tuple[PreparedHypothesisMatrix, PreparedHypothesisMatrix]:
    cards_by_reference = {
        value.entry.reference: value for value in preparation.selected_cards
    }
    values: list[PreparedHypothesisMatrix] = []
    for role, parent, contract, receipt_name in (
        (
            "diagnostic_parent",
            preparation.diagnostic_parent,
            preparation.diagnostic_contract,
            "diagnostic_receipt",
        ),
        (
            "hypothesis_parent",
            preparation.heldout_parent,
            preparation.heldout_contract,
            "heldout_receipt",
        ),
    ):
        requests = tuple(
            build_hypothesis_compilation_request(
                entry=entry,
                parent=parent,
                contract=contract,
            )
            for entry in active_entries
        )
        receipts = tuple(
            getattr(cards_by_reference[entry.reference], receipt_name)
            for entry in active_entries
        )
        values.append(
            PreparedHypothesisMatrix(
                parent_role=role,
                requests=requests,
                receipts=receipts,
            )
        )
    return values[0], values[1]


def _choice_definition_sha256(
    preparation: AirfoilG3ReleasePreparation,
    *,
    role: str,
    selection_sha256: str,
    option_id: str,
) -> str:
    return _hash(
        _CHOICE_POLICY_DOMAIN,
        {
            "policy_id": "airfoil_v7_g3_sealed_release_choice",
            "policy_version": 1,
            "role": role,
            "release_sha256": preparation.release_sha256,
            "source_selection_sha256": selection_sha256,
            "option_id": option_id,
            "current_or_heldout_outcome_access": False,
        },
    )


def build_airfoil_g3_curation_spec(
    preparation: AirfoilG3ReleasePreparation,
) -> G3PostsealCurationSpec:
    """Inject only Airfoil's metric/action vocabulary and evidence scope."""

    preparation.__post_init__()
    option_ids = tuple(
        sorted(
            value.entry.draft.recommended_option_ids[0]
            for value in preparation.selected_cards
        )
    )
    if len(option_ids) != 2 or len(set(option_ids)) != 2:
        raise AirfoilG3ReleaseError(
            "Airfoil G3 curation requires two distinct frozen trim actions"
        )
    return G3PostsealCurationSpec(
        insight_contract=ReflectionInsightContract(
            required_metric_ids=(
                "objective:normalized_multipoint_drag",
                "violation:normalized_lift_equality",
            ),
            allowed_option_families=("trim_only",),
            allowed_option_ids=option_ids,
        ),
        source_scope=G3CurationSourceScope(
            policy_id="airfoil_v7_g2_adaptive_only",
            policy_version=1,
            policy_definition_sha256=_CURATION_SCOPE_DEFINITION_SHA256,
            slot_ids=("g2_adaptive",),
        ),
    )


@dataclass(frozen=True, slots=True)
class AirfoilG3RuntimeInputs:
    """Provider/evaluator-free objects injected into generic runtime compose."""

    preparation: AirfoilG3ReleasePreparation
    diagnostic_permutation: FrozenDiagnosticPermutation
    benchmark: AgenticBenchmark
    id_factory: FrozenPrefixIdFactory
    memory: InsightMemoryBank
    active_entries: tuple[InsightMemoryEntry, InsightMemoryEntry]
    neutral_entry: InsightMemoryEntry
    prepared_hypothesis_matrices: tuple[
        PreparedHypothesisMatrix,
        PreparedHypothesisMatrix,
    ]
    neutral_choice: ParentBoundActionChoice
    mate_choice: ParentBoundActionChoice
    feedback_interceptor_factory: G3PostsealCurationFactory
    freeze_receipt_sha256: str | None
    runtime_inputs_sha256: str = field(init=False)
    planner_trace_sink: Any = None

    def __post_init__(self) -> None:
        self.preparation.__post_init__()
        expected_permutation, _, _ = freeze_diagnostic_permutation(self.preparation)
        if self.diagnostic_permutation != expected_permutation:
            raise ValueError("runtime permutation differs from the sealed public law")
        if self.freeze_receipt_sha256 is not None and (
            type(self.freeze_receipt_sha256) is not str
            or len(self.freeze_receipt_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.freeze_receipt_sha256
            )
        ):
            raise ValueError("freeze receipt SHA-256 is malformed")
        self.benchmark.validate_binding()
        problem_id = (
            f"{type(self.benchmark.problem).__module__}."
            f"{type(self.benchmark.problem).__qualname__}"
        )
        expected_catalogs = (
            AirfoilV7ShapeVariationCatalog(),
            AirfoilV7TrimVariationCatalog(),
            AirfoilV7UnionVariationCatalog(),
        )
        expected_catalog_identities = tuple(
            (
                value.catalog_id,
                value.catalog_version,
                value.definition_sha256,
            )
            for value in expected_catalogs
        )
        if (
            problem_id != AIRFOIL_G3_RUNTIME_PROBLEM_ID
            or self.benchmark.reward.binding_sha256
            != AIRFOIL_G3_ABSOLUTE_REWARD.binding_sha256
            or self.benchmark.outcome_relation is not AIRFOIL_V7_ARCHIVE_RELATION
            or self.benchmark.detailed_evaluator
            is not getattr(self.benchmark.problem, "detailed_evaluator", None)
            or self.benchmark.optimization_semantics
            is not getattr(self.benchmark.problem, "optimization_semantics", None)
            or self.benchmark.action_semantics
            is not getattr(self.benchmark.problem, "action_semantics", None)
            or type(self.benchmark.phenotype_identity)
            is not AirfoilV7PhenotypeIdentityPolicy
            or self.benchmark.finite_variation_catalog_identities
            != expected_catalog_identities
            or type(self.benchmark.hypothesis_compiler)
            is not AirfoilV7TrimHypothesisCompiler
        ):
            raise ValueError("runtime benchmark differs from the Airfoil G3 binding")
        if (
            type(self.active_entries) is not tuple
            or len(self.active_entries) != 2
            or len({value.reference for value in self.active_entries}) != 2
            or self.active_entries
            != tuple(sorted(self.active_entries, key=lambda value: value.reference))
        ):
            raise ValueError("active runtime entries must be two canonical cards")
        expected_active = tuple(
            sorted(
                (value.entry for value in self.preparation.selected_cards),
                key=lambda value: value.reference,
            )
        )
        if self.active_entries != expected_active:
            raise ValueError("active runtime memory differs from the sealed cards")
        if self.neutral_entry != self.preparation.sham_entry:
            raise ValueError("runtime neutral memory differs from the sealed sham")
        if type(self.feedback_interceptor_factory) is not G3PostsealCurationFactory:
            raise ValueError("runtime feedback factory is not the generic G3 policy")
        expected_initial_memory = tuple(
            sorted((*self.active_entries, self.neutral_entry), key=lambda x: x.reference)
        )
        factory = self.feedback_interceptor_factory
        interceptor = factory.interceptor
        expected_memory = expected_initial_memory
        if interceptor is not None:
            if type(interceptor) is not G3PostsealCurationInterceptor:
                raise ValueError("runtime curation factory holds a foreign interceptor")
            if interceptor.curation_receipt is not None:
                expected_memory = tuple(
                    sorted(
                        (*expected_initial_memory, *interceptor.curated_entries),
                        key=lambda value: value.reference,
                    )
                )
        if self.memory.entries != expected_memory:
            raise ValueError("runtime memory contains foreign or missing entries")
        if (
            self.id_factory.frozen_insight_ids
            != tuple(
                value.reference.insight_id
                for value in (*self.active_entries, self.neutral_entry)
            )
            or self.id_factory.frozen_insight_ids_consumed != 3
        ):
            raise ValueError("runtime ID factory differs from exact memory registration")
        expected_matrices = _prepared_matrices(self.preparation, self.active_entries)
        if self.prepared_hypothesis_matrices != expected_matrices:
            raise ValueError("runtime prepared matrices differ from sealed compilation")
        for value in self.prepared_hypothesis_matrices:
            value.__post_init__()
        self.neutral_choice.__post_init__()
        self.mate_choice.__post_init__()
        contract = self.benchmark.bind_finite_variation(
            AIRFOIL_G3_MODEL_CATALOG_ID,
            self.preparation.heldout_parent.candidate.configuration,
        )
        self.neutral_choice.validate_contract(contract)
        self.mate_choice.validate_contract(contract)
        expected_neutral_definition = _choice_definition_sha256(
            self.preparation,
            role="neutral_sham",
            selection_sha256=self.preparation.sham_selection_sha256,
            option_id=self.preparation.sham_entry.draft.recommended_option_ids[0],
        )
        expected_mate_definition = _choice_definition_sha256(
            self.preparation,
            role="orthogonal_mate",
            selection_sha256=self.preparation.mate_selection_sha256,
            option_id=self.preparation.mate_option_id,
        )
        if (
            self.neutral_choice.option_id
            != self.preparation.sham_entry.draft.recommended_option_ids[0]
            or self.mate_choice.option_id != self.preparation.mate_option_id
            or (
                self.neutral_choice.selection_policy_id,
                self.neutral_choice.selection_policy_version,
                self.neutral_choice.selection_policy_definition_sha256,
            )
            != (
                "airfoil_v7_g3_sealed_sham",
                1,
                expected_neutral_definition,
            )
            or (
                self.mate_choice.selection_policy_id,
                self.mate_choice.selection_policy_version,
                self.mate_choice.selection_policy_definition_sha256,
            )
            != (
                "airfoil_v7_g3_sealed_mate",
                1,
                expected_mate_definition,
            )
            or type(self.feedback_interceptor_factory)
            is not G3PostsealCurationFactory
            or self.feedback_interceptor_factory.spec
            != build_airfoil_g3_curation_spec(self.preparation)
        ):
            raise ValueError("runtime control choices or feedback factory changed")
        object.__setattr__(
            self,
            "runtime_inputs_sha256",
            _hash(_RUNTIME_INPUT_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "release_sha256": self.preparation.release_sha256,
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
            "problem_id": AIRFOIL_G3_RUNTIME_PROBLEM_ID,
            "phase": AIRFOIL_G3_RUNTIME_PHASE,
            "reward_binding_sha256": AIRFOIL_G3_ABSOLUTE_REWARD.binding_sha256,
            "benchmark_catalog_identities": [
                list(value)
                for value in self.benchmark.finite_variation_catalog_identities
            ],
            "compiler_definition_sha256": (
                self.benchmark.hypothesis_compiler.definition_sha256
            ),
            "active_source_evidence_sha256s": [
                request.source_evidence_sha256
                for request in self.prepared_hypothesis_matrices[0].requests
            ],
            "prepared_matrix_commitment_sha256s": [
                value.commitment_sha256
                for value in self.prepared_hypothesis_matrices
            ],
            "diagnostic_permutation_receipt_sha256": (
                self.diagnostic_permutation.receipt_sha256
            ),
            "neutral_choice_sha256": self.neutral_choice.choice_sha256,
            "mate_choice_sha256": self.mate_choice.choice_sha256,
            "estimand_stratum_sha256": AIRFOIL_G3_ESTIMAND_STRATUM_SHA256,
            "feedback_policy": {
                "policy_id": self.feedback_interceptor_factory.spec.policy_id,
                "policy_version": self.feedback_interceptor_factory.spec.policy_version,
                "policy_definition_sha256": (
                    self.feedback_interceptor_factory.spec.policy_definition_sha256
                ),
                "curation_spec_sha256": (
                    self.feedback_interceptor_factory.spec.spec_sha256
                ),
                "source_scope_sha256": (
                    self.feedback_interceptor_factory.spec.source_scope.scope_sha256
                ),
                "insight_contract_sha256": (
                    self.feedback_interceptor_factory.spec.insight_contract.identity_sha256
                ),
                "logical_calls": 1,
                "release_safe": True,
            },
            "preparation_source_manifest_is_not_live_manifest": True,
        }

    @property
    def active_references(self):
        return tuple(value.reference for value in self.active_entries)

    @property
    def seed_configurations(self) -> tuple[dict[str, Any], dict[str, Any]]:
        values = (
            thaw_json(self.preparation.diagnostic_parent.candidate.configuration),
            thaw_json(self.preparation.heldout_parent.candidate.configuration),
        )
        if any(type(value) is not dict for value in values):
            raise TypeError("Airfoil G3 seed configuration must be an object")
        return values  # type: ignore[return-value]

    def build(
        self,
        *,
        benchmark: AgenticBenchmark,
        engine: AgenticEvolutionEngine,
        id_factory: IdFactory,
        memory: InsightMemoryBank,
    ) -> G3CausalScreenPlanner:
        """Late-bind the exact engine instance through the generic factory seam."""

        if type(engine) is not AgenticEvolutionEngine:
            raise TypeError("engine must be an exact AgenticEvolutionEngine")
        if (
            benchmark is not self.benchmark
            or id_factory is not self.id_factory
            or memory is not self.memory
            or engine.problem is not benchmark.problem
            or engine.ids is not id_factory
            or engine.memory is not memory
            or engine.problem_id != AIRFOIL_G3_RUNTIME_PROBLEM_ID
            or engine.reward_binding.binding_sha256
            != AIRFOIL_G3_ABSOLUTE_REWARD.binding_sha256
        ):
            raise ValueError("engine differs from frozen Airfoil G3 runtime inputs")
        return G3CausalScreenPlanner(
            benchmark=self.benchmark,
            engine=engine,
            ids=self.id_factory,
            memory=self.memory,
            reward_binding=AIRFOIL_G3_ABSOLUTE_REWARD,
            active_references=self.active_references,
            neutral_reference=self.neutral_entry.reference,
            diagnostic_permutation=self.diagnostic_permutation,
            prepared_hypothesis_matrices=self.prepared_hypothesis_matrices,
            model_catalog_id=AIRFOIL_G3_MODEL_CATALOG_ID,
            neutral_choice=self.neutral_choice,
            mate_choice=self.mate_choice,
            diagnostic_parent_configuration_sha256=(
                self.preparation.diagnostic_parent.candidate.configuration_sha256
            ),
            hypothesis_parent_configuration_sha256=(
                self.preparation.heldout_parent.candidate.configuration_sha256
            ),
            endpoint_definition_sha256=ABSOLUTE_Q_DEFINITION_SHA256,
            estimand_stratum_sha256=AIRFOIL_G3_ESTIMAND_STRATUM_SHA256,
            phase=AIRFOIL_G3_RUNTIME_PHASE,
            no_yield_reward=-2.0,
            trace_sink=self.planner_trace_sink,
        )


def compose_airfoil_g3_runtime_inputs(
    *,
    problem: object,
    preparation: AirfoilG3ReleasePreparation,
    diagnostic_permutation: FrozenDiagnosticPermutation,
    delegate_id_factory: IdFactory | None = None,
    freeze_receipt_sha256: str | None = None,
    planner_trace_sink=None,
) -> AirfoilG3RuntimeInputs:
    """Compose sealed benchmark inputs without provider or evaluator execution."""

    preparation.__post_init__()
    expected_permutation, _, _ = freeze_diagnostic_permutation(preparation)
    if diagnostic_permutation != expected_permutation:
        raise AirfoilG3ReleaseError(
            "runtime diagnostic permutation differs from public prelaunch law"
        )
    problem_id = f"{type(problem).__module__}.{type(problem).__qualname__}"
    if problem_id != AIRFOIL_G3_RUNTIME_PROBLEM_ID:
        raise AirfoilG3ReleaseError(
            f"Airfoil G3 requires exact runtime problem identity {AIRFOIL_G3_RUNTIME_PROBLEM_ID}"
        )
    detailed_evaluator = getattr(problem, "detailed_evaluator", None)
    optimization_semantics = getattr(problem, "optimization_semantics", None)
    action_semantics = getattr(problem, "action_semantics", None)
    compiler = AirfoilV7TrimHypothesisCompiler()
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=AIRFOIL_G3_ABSOLUTE_REWARD,
        detailed_evaluator=detailed_evaluator,
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        optimization_semantics=optimization_semantics,
        action_semantics=action_semantics,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(
            AirfoilV7ShapeVariationCatalog(),
            AirfoilV7TrimVariationCatalog(),
            AirfoilV7UnionVariationCatalog(),
        ),
        hypothesis_compiler=compiler,
    )
    benchmark.validate_binding()

    active_source = tuple(
        sorted(
            (value.entry for value in preparation.selected_cards),
            key=lambda value: value.reference,
        )
    )
    exact_insight_ids = tuple(
        value.reference.insight_id for value in (*active_source, preparation.sham_entry)
    )
    ids = FrozenPrefixIdFactory(
        delegate=(
            DeterministicIdFactory("airfoil_g3_runtime")
            if delegate_id_factory is None
            else delegate_id_factory
        ),
        frozen_insight_ids=exact_insight_ids,
    )
    memory = InsightMemoryBank(id_factory=ids)
    active_entries = tuple(
        _register_exact_entry(memory, value) for value in active_source
    )
    neutral_entry = _register_exact_entry(memory, preparation.sham_entry)
    if ids.frozen_insight_ids_consumed != 3:
        raise AirfoilG3ReleaseError("runtime did not consume every frozen insight ID")
    prepared = _prepared_matrices(preparation, active_entries)

    union_contract = benchmark.bind_finite_variation(
        AIRFOIL_G3_MODEL_CATALOG_ID,
        preparation.heldout_parent.candidate.configuration,
    )
    if union_contract.identity_sha256 != preparation.heldout_contract.identity_sha256:
        raise AirfoilG3ReleaseError("runtime union catalog differs from sealed P_H")
    neutral_choice = ParentBoundActionChoice.seal(
        role="neutral_sham",
        contract=union_contract,
        option_id=preparation.sham_entry.draft.recommended_option_ids[0],
        selection_policy_id="airfoil_v7_g3_sealed_sham",
        selection_policy_version=1,
        selection_policy_definition_sha256=_choice_definition_sha256(
            preparation,
            role="neutral_sham",
            selection_sha256=preparation.sham_selection_sha256,
            option_id=preparation.sham_entry.draft.recommended_option_ids[0],
        ),
    )
    mate_choice = ParentBoundActionChoice.seal(
        role="orthogonal_mate",
        contract=union_contract,
        option_id=preparation.mate_option_id,
        selection_policy_id="airfoil_v7_g3_sealed_mate",
        selection_policy_version=1,
        selection_policy_definition_sha256=_choice_definition_sha256(
            preparation,
            role="orthogonal_mate",
            selection_sha256=preparation.mate_selection_sha256,
            option_id=preparation.mate_option_id,
        ),
    )
    return AirfoilG3RuntimeInputs(
        preparation=preparation,
        diagnostic_permutation=diagnostic_permutation,
        benchmark=benchmark,
        id_factory=ids,
        memory=memory,
        active_entries=active_entries,  # type: ignore[arg-type]
        neutral_entry=neutral_entry,
        prepared_hypothesis_matrices=prepared,
        neutral_choice=neutral_choice,
        mate_choice=mate_choice,
        feedback_interceptor_factory=G3PostsealCurationFactory(
            spec=build_airfoil_g3_curation_spec(preparation),
        ),
        freeze_receipt_sha256=freeze_receipt_sha256,
        planner_trace_sink=planner_trace_sink,
    )


def load_frozen_airfoil_g3_runtime_inputs(
    *,
    problem: object,
    planner_trace_sink=None,
) -> AirfoilG3RuntimeInputs:
    """Live-facing loader: require and cross-bind the chronology receipt."""

    preparation = prepare_release()
    freeze_receipt = load_prelaunch_freeze_receipt()
    release_path = DEFAULT_RELEASE_PATH.expanduser().resolve(strict=True)
    bound_files = (
        (DEFAULT_DENYLIST_PATH, freeze_receipt.membership_file_sha256),
        (DEFAULT_CARD_BANK_PATH, freeze_receipt.card_bank_file_sha256),
        (release_path, freeze_receipt.release_file_sha256),
    )
    for path, expected_sha256 in bound_files:
        resolved = Path(path).expanduser().resolve(strict=True)
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256:
            raise AirfoilG3ReleaseError(
                "canonical release input bytes differ from the freeze receipt"
            )
    try:
        persisted_release = json.loads(release_path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilG3ReleaseError("canonical release JSON is malformed") from exc
    if (
        freeze_receipt.release_sha256 != preparation.release_sha256
        or freeze_receipt.membership_sha256
        != preparation.membership.membership_sha256
        or freeze_receipt.card_bank_sha256 != preparation.card_bank.card_bank_sha256
        or persisted_release != preparation.to_record()
    ):
        raise AirfoilG3ReleaseError(
            "prelaunch chronology receipt differs from deterministic preparation"
        )
    return compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=freeze_receipt.diagnostic_permutation,
        freeze_receipt_sha256=freeze_receipt.freeze_receipt_sha256,
        planner_trace_sink=planner_trace_sink,
    )


__all__ = [
    "AIRFOIL_G3_ESTIMAND_STRATUM_SHA256",
    "AIRFOIL_G3_MODEL_CATALOG_ID",
    "AirfoilG3RuntimeInputs",
    "FrozenPrefixIdFactory",
    "build_airfoil_g3_curation_spec",
    "compose_airfoil_g3_runtime_inputs",
    "load_frozen_airfoil_g3_runtime_inputs",
]
