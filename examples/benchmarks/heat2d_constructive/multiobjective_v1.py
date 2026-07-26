"""Pareto formulation over the qualified serialized Heat2D evaluator.

This module gives the scientific workload its own immutable identity while
reusing the already-qualified direct-v3 PDE boundary unchanged.  Candidate
semantics stay local to the benchmark: callers receive only the public
``AgenticBenchmark`` bundle, two seed configurations, and the finite catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog, ObjectiveSpec

from .artifact_boundary import artifact_scripts_dir, decoder_source_sha256
from .candidate import (
    CandidateConfig,
    SEED_LAYOUT_A,
    canonical_candidate_json,
    normalize_candidate,
    seed_layouts,
)
from .finite_variation_catalog import (
    CATALOG_DEFINITION_SHA256,
    CATALOG_ID,
    Heat2DFiniteVariationCatalog,
)
from .phenotype_identity import Heat2DPhenotypeIdentityPolicy
from .problem_def import (
    EVALUATOR_ID,
    OBJECTIVE_NAME as DIRECT_V3_OBJECTIVE_NAME,
    DirectV3ContractError,
    DirectV3Evaluator,
    Heat2DDirectV3Evaluation,
    Heat2DDirectV3Settings,
    Heat2DEvaluator,
)


WORKLOAD_ID = "engibench-heatconduction2d-constructive-pareto-v1"
WORKLOAD_VERSION = 1
THERMAL_OBJECTIVE_NAME = "thermal_term"
MATERIAL_OBJECTIVE_NAME = "material_fraction"
OBJECTIVE_NAMES = (THERMAL_OBJECTIVE_NAME, MATERIAL_OBJECTIVE_NAME)
EXACT_VOLUME_POLICY_ID = "unit_square_cg1_binary64_exact_volume_identity"
EXACT_VOLUME_POLICY_VERSION = 1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


_FORMULATION_DEFINITION = {
    "workload_id": WORKLOAD_ID,
    "workload_version": WORKLOAD_VERSION,
    "underlying_evaluator_id": EVALUATOR_ID,
    "candidate_decoder_sha256": decoder_source_sha256(),
    "finite_catalog": {
        "catalog_id": CATALOG_ID,
        "definition_sha256": CATALOG_DEFINITION_SHA256,
    },
    "objectives": [
        {"name": THERMAL_OBJECTIVE_NAME, "goal": "min"},
        {"name": MATERIAL_OBJECTIVE_NAME, "goal": "min"},
    ],
    "thermal_projection": "direct_v3.container_result.result.thermal_term",
    "material_projection": (
        "exact_scaled_numerator/(mesh_mass_denominator*2^binary64_exponent)"
    ),
    "admission": "all direct-v3 checks plus exact-volume receipt validation",
}
FORMULATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:heat2d-constructive-pareto:v1\x00"
    + _canonical_bytes(_FORMULATION_DEFINITION)
).hexdigest()


def _exact_object(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise DirectV3ContractError(f"multiobjective {name} must be an exact object")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectV3ContractError(f"multiobjective {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DirectV3ContractError(f"multiobjective {name} must be finite")
    return number


def _sha256_text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DirectV3ContractError(f"multiobjective {name} is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class Heat2DMultiObjectiveV1Evaluation:
    """Trusted two-objective view plus the complete direct-v3 evidence."""

    objective_values: dict[str, float]
    thermal_term: float
    regularization_term: float
    combined_direct_v3_objective: float
    material_fraction: float
    exact_volume_numerator_decimal: str
    exact_volume_denominator: int
    exact_volume_contract_sha256: str
    direct_v3: Heat2DDirectV3Evaluation


def _project_evaluation(
    candidate: CandidateConfig,
    direct: Heat2DDirectV3Evaluation,
    settings: Heat2DDirectV3Settings,
) -> Heat2DMultiObjectiveV1Evaluation:
    if type(direct) is not Heat2DDirectV3Evaluation:
        raise TypeError("evaluator must return an exact Heat2DDirectV3Evaluation")
    manifest = _exact_object(direct.manifest, "direct-v3 manifest")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("evaluator_id") != EVALUATOR_ID
        or manifest.get("all_checks_pass") is not True
        or manifest.get("full_pde_solve_count") != 1
    ):
        raise DirectV3ContractError(
            "multiobjective projection requires a passing direct-v3 manifest"
        )
    checks = _exact_object(manifest.get("checks"), "direct-v3 checks")
    agreement = _exact_object(manifest.get("volume_agreement"), "volume agreement")
    if (
        checks.get("exact_cross_runtime_fe_volume_identity_matches") is not True
        or agreement.get("exact_identity_matches") is not True
        or agreement.get("admission_policy") != EXACT_VOLUME_POLICY_ID
        or agreement.get("admission_policy_version") != EXACT_VOLUME_POLICY_VERSION
    ):
        raise DirectV3ContractError(
            "multiobjective projection requires exact cross-runtime volume identity"
        )

    container = _exact_object(manifest.get("container_result"), "container result")
    result = _exact_object(container.get("result"), "scientific result")
    combined = _finite_number(
        result.get(DIRECT_V3_OBJECTIVE_NAME), DIRECT_V3_OBJECTIVE_NAME
    )
    thermal = _finite_number(result.get(THERMAL_OBJECTIVE_NAME), "thermal term")
    regularization = _finite_number(
        result.get("regularization_term"), "regularization term"
    )
    residual = _finite_number(
        result.get("decomposition_residual"), "decomposition residual"
    )
    if thermal <= 0.0 or regularization < 0.0:
        raise DirectV3ContractError(
            "multiobjective thermal and regularization terms are out of range"
        )
    direct_combined = direct.objective_values.get(DIRECT_V3_OBJECTIVE_NAME)
    if (
        direct_combined != combined
        or abs(combined - thermal - regularization) > 1.0e-12
        or abs(residual) > 1.0e-12
    ):
        raise DirectV3ContractError(
            "multiobjective projection observed an invalid objective decomposition"
        )

    contract = _exact_object(
        container.get("exact_volume_contract"), "exact-volume contract"
    )
    if (
        contract.get("schema_version") != 1
        or contract.get("policy_id") != EXACT_VOLUME_POLICY_ID
        or contract.get("policy_version") != EXACT_VOLUME_POLICY_VERSION
        or contract.get("resolution") != settings.resolution
        or contract.get("cells_per_axis") != settings.resolution - 1
        or contract.get("binary64_common_denominator_exponent") != 1074
        or contract.get("mesh_mass_denominator") != 6 * (settings.resolution - 1) ** 2
    ):
        raise DirectV3ContractError(
            "multiobjective projection observed exact-volume contract drift"
        )
    numerator_decimal = contract.get("exact_scaled_numerator_decimal")
    if (
        type(numerator_decimal) is not str
        or not numerator_decimal
        or not numerator_decimal.isascii()
        or not numerator_decimal.isdecimal()
        or (len(numerator_decimal) > 1 and numerator_decimal.startswith("0"))
    ):
        raise DirectV3ContractError("exact-volume numerator is not canonical decimal")
    numerator = int(numerator_decimal)
    mesh_denominator = contract["mesh_mass_denominator"]
    denominator = mesh_denominator * (1 << 1074)
    material = float(Fraction(numerator, denominator))
    if (
        not math.isfinite(material)
        or material <= 0.0
        or abs(material - candidate.material_fraction) > settings.volume_tolerance
        or abs(material - direct.finite_element_volume) > 1.0e-12
    ):
        raise DirectV3ContractError(
            "exact material usage does not bind the candidate and decoded field"
        )
    contract_sha256 = _sha256_text(
        contract.get("contract_sha256"), "exact-volume contract digest"
    )
    _sha256_text(
        contract.get("exact_scaled_numerator_sha256"),
        "exact-volume numerator digest",
    )

    return Heat2DMultiObjectiveV1Evaluation(
        objective_values={
            THERMAL_OBJECTIVE_NAME: thermal,
            MATERIAL_OBJECTIVE_NAME: material,
        },
        thermal_term=thermal,
        regularization_term=regularization,
        combined_direct_v3_objective=combined,
        material_fraction=material,
        exact_volume_numerator_decimal=numerator_decimal,
        exact_volume_denominator=denominator,
        exact_volume_contract_sha256=contract_sha256,
        direct_v3=direct,
    )


class Heat2DMultiObjectiveV1Problem:
    """Public two-objective problem backed by one serialized direct-v3 solve."""

    candidate_model = CandidateConfig
    example_config = SEED_LAYOUT_A
    workload_id = WORKLOAD_ID
    workload_version = WORKLOAD_VERSION
    formulation_definition_sha256 = FORMULATION_DEFINITION_SHA256
    constraints_description = (
        "fixed four-primitive CSG topology; finite bounded scalars; exact CG1 "
        "material-volume identity; primitive identity, operation, kind, and order "
        "are engine-owned; direct-v3 PDE evaluations are serialized"
    )

    def __init__(
        self,
        settings: Heat2DDirectV3Settings,
        *,
        evaluator: Heat2DEvaluator | None = None,
    ) -> None:
        if type(settings) is not Heat2DDirectV3Settings:
            raise TypeError("settings must be exact Heat2DDirectV3Settings")
        if evaluator is not None and not isinstance(evaluator, Heat2DEvaluator):
            raise TypeError("evaluator must implement Heat2DEvaluator")
        self.settings = settings
        self._evaluator: Heat2DEvaluator = (
            DirectV3Evaluator(settings) if evaluator is None else evaluator
        )

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (
            ObjectiveSpec(THERMAL_OBJECTIVE_NAME, "min"),
            ObjectiveSpec(MATERIAL_OBJECTIVE_NAME, "min"),
        )

    @property
    def evaluator(self) -> Heat2DEvaluator:
        return self._evaluator

    def preflight(self) -> dict[str, object]:
        preflight = getattr(self._evaluator, "preflight", None)
        if not callable(preflight):
            raise DirectV3ContractError(
                "multiobjective evaluator must expose a provider/PDE-free preflight"
            )
        record = preflight()
        if type(record) is not dict:
            raise DirectV3ContractError(
                "evaluator preflight must return an exact object"
            )
        return {
            "workload_id": WORKLOAD_ID,
            "workload_version": WORKLOAD_VERSION,
            "formulation_definition_sha256": FORMULATION_DEFINITION_SHA256,
            "underlying_evaluator_id": EVALUATOR_ID,
            "objectives": [{"name": name, "goal": "min"} for name in OBJECTIVE_NAMES],
            "external_concurrency": self.settings.external_concurrency,
            "evaluator_preflight": record,
        }

    def validate(self, config: object) -> bool:
        normalize_candidate(config)
        return True

    def evaluate_detailed(self, config: object) -> Heat2DMultiObjectiveV1Evaluation:
        candidate = normalize_candidate(config)
        direct = self._evaluator.evaluate(candidate)
        return _project_evaluation(candidate, direct, self.settings)

    def evaluate(self, config: object) -> dict[str, float]:
        return dict(self.evaluate_detailed(config).objective_values)

    @staticmethod
    def candidate_key(config: object) -> str:
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:heat2d-constructive-pareto-candidate:v1\x00")
        digest.update(bytes.fromhex(FORMULATION_DEFINITION_SHA256))
        digest.update(canonical_candidate_json(config).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def render_candidate(config: object) -> str:
        candidate = normalize_candidate(config)
        return (
            f"Heat2D Pareto CSG material={candidate.material_fraction:.3f}, "
            f"trunk_radius={candidate.trunk.radius:.3f}, "
            f"hole=({candidate.central_hole.radius_x:.3f},"
            f"{candidate.central_hole.radius_y:.3f})"
        )

    @staticmethod
    def search_space_description() -> str:
        return (
            "Constructive HeatConduction2D Pareto co-optimization over a fixed "
            "engine-owned CSG topology. Bounded geometry and requested material "
            "fraction are varied through the workload-owned finite catalog. The "
            "qualified decoder projects a dense CG1 field and direct-v3 proves "
            "its exact cross-runtime integral. Minimize the physical thermal term "
            "and exact projected material fraction as separate objectives."
        )


def default_multiobjective_settings() -> Heat2DDirectV3Settings:
    research_artifacts = artifact_scripts_dir().parent
    return Heat2DDirectV3Settings(
        output_root=(
            research_artifacts
            / "experiment_logs"
            / "benchmark_q1"
            / "engibench_heat2d"
            / "agent_evolve_constructive_pareto_v1_development"
        ),
    )


def create_multiobjective_problem(
    settings: Heat2DDirectV3Settings | None = None,
    *,
    evaluator: Heat2DEvaluator | None = None,
) -> Heat2DMultiObjectiveV1Problem:
    selected = default_multiobjective_settings() if settings is None else settings
    return Heat2DMultiObjectiveV1Problem(selected, evaluator=evaluator)


def create_multiobjective_benchmark(
    settings: Heat2DDirectV3Settings | None = None,
    *,
    evaluator: Heat2DEvaluator | None = None,
) -> AgenticBenchmark:
    selected_problem = create_multiobjective_problem(settings, evaluator=evaluator)
    selected_catalog: FiniteVariationCatalog = Heat2DFiniteVariationCatalog()
    return AgenticBenchmark(
        problem=selected_problem,
        phenotype_identity=Heat2DPhenotypeIdentityPolicy(
            resolution=selected_problem.settings.resolution
        ),
        finite_variation_catalogs=(selected_catalog,),
    )


def prepare_multiobjective_benchmark(
    selected_benchmark: AgenticBenchmark,
) -> dict[str, object]:
    """Prepare exact seeds/catalogs without a provider call or PDE solve."""

    if type(selected_benchmark) is not AgenticBenchmark:
        raise TypeError("selected_benchmark must be an exact AgenticBenchmark")
    selected_problem = selected_benchmark.problem
    if type(selected_problem) is not Heat2DMultiObjectiveV1Problem:
        raise TypeError("benchmark does not contain the exact Pareto-v1 problem")
    preflight = selected_problem.preflight()
    seed_records: list[dict[str, object]] = []
    for ordinal, seed in enumerate(seed_layouts()):
        payload = seed.model_dump(mode="python")
        identity = selected_benchmark.phenotype_identity.identify(payload)
        contract = selected_benchmark.bind_finite_variation(CATALOG_ID, payload)
        seed_records.append(
            {
                "ordinal": ordinal,
                "candidate_key": selected_problem.candidate_key(payload),
                "candidate_json_sha256": hashlib.sha256(
                    canonical_candidate_json(payload).encode("ascii")
                ).hexdigest(),
                "requested_material_fraction": seed.material_fraction,
                "phenotype_sha256": identity.value_sha256,
                "finite_contract_sha256": contract.identity_sha256,
                "finite_option_count": len(contract.options),
            }
        )
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "provider_and_pde_free_prepared",
        "preflight": preflight,
        "catalog_identities": [
            {
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "definition_sha256": definition_sha256,
            }
            for catalog_id, catalog_version, definition_sha256 in (
                selected_benchmark.finite_variation_catalog_identities
            )
        ],
        "seeds": seed_records,
        "provider_calls": 0,
        "pde_solves": 0,
    }
    receipt_sha256 = hashlib.sha256(
        b"agent-evolve:heat2d-constructive-pareto-prepare:v1\x00"
        + _canonical_bytes(record)
    ).hexdigest()
    return {**record, "receipt_sha256": receipt_sha256}


problem = create_multiobjective_problem()
finite_variation_catalog: FiniteVariationCatalog = Heat2DFiniteVariationCatalog()
benchmark = AgenticBenchmark(
    problem=problem,
    phenotype_identity=Heat2DPhenotypeIdentityPolicy(
        resolution=problem.settings.resolution
    ),
    finite_variation_catalogs=(finite_variation_catalog,),
)


__all__ = [
    "EXACT_VOLUME_POLICY_ID",
    "FORMULATION_DEFINITION_SHA256",
    "MATERIAL_OBJECTIVE_NAME",
    "OBJECTIVE_NAMES",
    "THERMAL_OBJECTIVE_NAME",
    "WORKLOAD_ID",
    "WORKLOAD_VERSION",
    "Heat2DMultiObjectiveV1Evaluation",
    "Heat2DMultiObjectiveV1Problem",
    "benchmark",
    "create_multiobjective_benchmark",
    "create_multiobjective_problem",
    "default_multiobjective_settings",
    "finite_variation_catalog",
    "prepare_multiobjective_benchmark",
    "problem",
]
