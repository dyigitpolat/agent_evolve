"""No-repair compiler from v2 decisions to operational Timeloop plans."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import product
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate import (
    ARCHITECTURE_FIELD_GRIDS,
    POLICY_BLOCK_FIELDS,
    CandidateConfig,
    MappingPolicyBlock,
    architecture_cardinality,
    candidate_cardinality,
    candidate_sha256,
    iter_policy_blocks,
    normalize_candidate,
    policy_cardinality,
)
from .network_panel import (
    LayerMedoid,
    NetworkLayerPanel,
    panel_sha256,
)


COMPILER_ID = "timeloop_network_mapspace_policy_compiler"
COMPILER_VERSION = 1
_PLAN_HASH_DOMAIN = b"agent-evolve:timeloop-compiled-plan:v1\x00"
_PROOF_HASH_DOMAIN = b"agent-evolve:timeloop-cardinality-proof:v1\x00"
_DIMENSIONS = ("N", "C", "M", "R", "S", "P", "Q")
_DATASPACES = ("Inputs", "Weights", "Outputs")

PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


_DATAFLOW_TEMPLATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "weight_stationary": ("NMPQCRS", ("Inputs", "Outputs")),
    "output_stationary": ("NCRSPQM", ("Inputs", "Weights")),
    "row_stationary": ("NMCQSPR", ("Outputs",)),
    "no_local_reuse": ("NMCPQRS", _DATASPACES),
}

_OUTER_LOOP_TEMPLATES: dict[str, str] = {
    "channel_major": "NMCPQRS",
    "channel_then_spatial": "NMCQPRS",
    "input_channel_major": "NCPQMRS",
    "output_spatial_major": "NMPQCRS",
    "filter_major": "NCRSMPQ",
    "spatial_major": "NPQMCRS",
}

_RESIDENCY_TEMPLATES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "balanced": (_DATASPACES, ()),
    "input_reuse": (("Inputs",), ("Weights", "Outputs")),
    "weight_reuse": (("Weights",), ("Inputs", "Outputs")),
    "output_reuse": (("Outputs",), ("Inputs", "Weights")),
}

_UTILIZATION_FRACTIONS: dict[str, Fraction] = {
    "low": Fraction(1, 4),
    "medium": Fraction(1, 2),
    "high": Fraction(3, 4),
    "full": Fraction(1, 1),
}

_COMPILER_DEFINITION = {
    "compiler_id": COMPILER_ID,
    "compiler_version": COMPILER_VERSION,
    "repair_policy": "none_fail_closed",
    "dimension_order": list(_DIMENSIONS),
    "dataspace_order": list(_DATASPACES),
    "dataflow_templates": {
        key: {"permutation": value[0], "no_reuse": list(value[1])}
        for key, value in _DATAFLOW_TEMPLATES.items()
    },
    "outer_loop_templates": _OUTER_LOOP_TEMPLATES,
    "residency_templates": {
        key: {"keep": list(value[0]), "bypass": list(value[1])}
        for key, value in _RESIDENCY_TEMPLATES.items()
    },
    "utilization_fractions": {
        key: [value.numerator, value.denominator]
        for key, value in _UTILIZATION_FRACTIONS.items()
    },
    "local_temporal_target_law": {
        "register_enabled": "register_file",
        "register_disabled": "inter_pe_temporal_level",
        "rationale": (
            "Timeloop temporal constraints target storage levels; without the "
            "optional register the innermost temporal level is the existing "
            "inter-PE dummy storage, never the MAC compute component"
        ),
    },
    "spatial_semantics": (
        "force every non-primary dimension to factor one; leave the primary "
        "axis open; require the selected utilization fraction of the maximum "
        "fanout reachable by that axis"
    ),
}
COMPILER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-v2-compiler-definition:v1\x00"
    + _canonical_bytes(_COMPILER_DEFINITION)
).hexdigest()


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class CompiledArchitecture(_ClosedModel):
    global_buffer_depth: Literal[256, 512, 1024, 2048]
    global_buffer_width: Literal[64, 128, 256]
    pe_mesh_x: Literal[4, 8, 16, 32]
    datawidth_bits: Literal[8, 16]
    register_enabled: bool


class CompiledTemporalConstraint(_ClosedModel):
    target_role: Literal[
        "register_file", "inter_pe_temporal_level", "global_buffer"
    ]
    permutation: str
    no_reuse: tuple[Literal["Inputs", "Weights", "Outputs"], ...] = ()

    @model_validator(mode="after")
    def _validate_permutation(self) -> "CompiledTemporalConstraint":
        if len(self.permutation) != len(_DIMENSIONS) or set(self.permutation) != set(
            _DIMENSIONS
        ):
            raise ValueError("temporal permutation must contain every dimension once")
        if len(set(self.no_reuse)) != len(self.no_reuse):
            raise ValueError("no_reuse must not contain duplicates")
        canonical_no_reuse = tuple(
            dataspace for dataspace in _DATASPACES if dataspace in self.no_reuse
        )
        if self.no_reuse != canonical_no_reuse:
            raise ValueError("no_reuse must use canonical dataspace order")
        return self


class CompiledSpatialConstraint(_ClosedModel):
    target_role: Literal["processing_element_mesh"] = "processing_element_mesh"
    permutation: str
    unit_factors: tuple[str, ...]
    minimum_parallelism_numerator: PositiveStrictInt
    minimum_parallelism_denominator: PositiveStrictInt

    @model_validator(mode="after")
    def _validate_constraint(self) -> "CompiledSpatialConstraint":
        if len(self.permutation) != len(_DIMENSIONS) or set(self.permutation) != set(
            _DIMENSIONS
        ):
            raise ValueError("spatial permutation must contain every dimension once")
        if len(self.unit_factors) != len(_DIMENSIONS) - 1:
            raise ValueError("exactly one spatial dimension must remain open")
        expected = {f"{dimension}=1" for dimension in _DIMENSIONS}
        observed = set(self.unit_factors)
        if len(observed) != len(self.unit_factors) or not observed < expected:
            raise ValueError("unit_factors must be six unique canonical constraints")
        canonical_factors = tuple(
            f"{dimension}=1"
            for dimension in _DIMENSIONS
            if f"{dimension}=1" in observed
        )
        if self.unit_factors != canonical_factors:
            raise ValueError("unit_factors must use canonical dimension order")
        if (
            Fraction(
                self.minimum_parallelism_numerator,
                self.minimum_parallelism_denominator,
            )
            > 1
        ):
            raise ValueError("minimum parallelism cannot exceed one")
        return self


class CompiledDataspaceConstraint(_ClosedModel):
    target_role: Literal["global_buffer"] = "global_buffer"
    keep: tuple[Literal["Inputs", "Weights", "Outputs"], ...]
    bypass: tuple[Literal["Inputs", "Weights", "Outputs"], ...]

    @model_validator(mode="after")
    def _validate_partition(self) -> "CompiledDataspaceConstraint":
        if set(self.keep).intersection(self.bypass) or set(self.keep).union(
            self.bypass
        ) != set(_DATASPACES):
            raise ValueError("keep and bypass must partition all dataspaces")
        canonical_keep = tuple(
            dataspace for dataspace in _DATASPACES if dataspace in self.keep
        )
        canonical_bypass = tuple(
            dataspace for dataspace in _DATASPACES if dataspace in self.bypass
        )
        if self.keep != canonical_keep or self.bypass != canonical_bypass:
            raise ValueError("dataspaces must use canonical order")
        return self


class CompiledPolicyPlan(_ClosedModel):
    problem_instance: dict[str, PositiveStrictInt]
    layer_multiplicity: PositiveStrictInt
    local_temporal: CompiledTemporalConstraint
    spatial: CompiledSpatialConstraint
    global_temporal: CompiledTemporalConstraint
    global_buffer_residency: CompiledDataspaceConstraint

    @model_validator(mode="after")
    def _validate_problem(self) -> "CompiledPolicyPlan":
        expected = {
            "N",
            "C",
            "M",
            "R",
            "S",
            "P",
            "Q",
            "Hstride",
            "Wstride",
            "Hdilation",
            "Wdilation",
        }
        if set(self.problem_instance) != expected:
            raise ValueError("problem_instance has missing or extra dimensions")
        return self


class CompiledPlan(_ClosedModel):
    """Only operational data; raw decision labels and provenance are excluded."""

    schema_version: Literal[1] = 1
    compiler_definition_sha256: Literal[COMPILER_DEFINITION_SHA256] = (
        COMPILER_DEFINITION_SHA256
    )
    architecture: CompiledArchitecture
    medoid_0: CompiledPolicyPlan
    medoid_1: CompiledPolicyPlan
    medoid_2: CompiledPolicyPlan


def canonical_compiled_plan_bytes(plan: CompiledPlan) -> bytes:
    if type(plan) is not CompiledPlan:
        raise TypeError("plan must be an exact CompiledPlan")
    return _canonical_bytes(plan.model_dump(mode="python"))


def compiled_plan_sha256(plan: CompiledPlan) -> str:
    return hashlib.sha256(
        _PLAN_HASH_DOMAIN + canonical_compiled_plan_bytes(plan)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: CompiledPlan
    compiled_plan_sha256: str
    candidate_sha256: str
    panel_sha256: str
    repair_count: Literal[0] = 0

    def __post_init__(self) -> None:
        if type(self.plan) is not CompiledPlan:
            raise TypeError("plan must be an exact CompiledPlan")
        if self.compiled_plan_sha256 != compiled_plan_sha256(self.plan):
            raise ValueError("compiled plan digest mismatch")
        for value in (
            self.compiled_plan_sha256,
            self.candidate_sha256,
            self.panel_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("compilation digests must be lowercase SHA-256")
        if self.repair_count != 0:
            raise ValueError("the v2 compiler does not permit repair")


def _compile_architecture(candidate: CandidateConfig) -> CompiledArchitecture:
    return CompiledArchitecture(
        global_buffer_depth=candidate.global_buffer_depth,
        global_buffer_width=candidate.global_buffer_width,
        pe_mesh_x=candidate.pe_mesh_x,
        datawidth_bits=candidate.datawidth_bits,
        register_enabled=candidate.register_enabled,
    )


def _compile_policy(
    policy: MappingPolicyBlock,
    medoid: LayerMedoid,
    architecture: CompiledArchitecture,
) -> CompiledPolicyPlan:
    local_permutation, no_reuse = _DATAFLOW_TEMPLATES[policy.dataflow_family]
    inner_target = (
        "register_file"
        if architecture.register_enabled
        else "inter_pe_temporal_level"
    )

    primary_axis = policy.primary_spatial_axis
    other_dimensions = tuple(
        dimension for dimension in _DIMENSIONS if dimension != primary_axis
    )
    spatial_extent = medoid.shape.spatial_extent(primary_axis)
    reachable_fanout = min(spatial_extent, architecture.pe_mesh_x)
    requested_fraction = _UTILIZATION_FRACTIONS[policy.spatial_utilization]
    minimum_parallelism = requested_fraction * Fraction(
        reachable_fanout,
        architecture.pe_mesh_x,
    )
    keep, bypass = _RESIDENCY_TEMPLATES[policy.buffer_residency_bias]

    return CompiledPolicyPlan(
        problem_instance=medoid.shape.timeloop_instance(),
        layer_multiplicity=medoid.multiplicity,
        local_temporal=CompiledTemporalConstraint(
            target_role=inner_target,
            permutation=local_permutation,
            no_reuse=no_reuse,
        ),
        spatial=CompiledSpatialConstraint(
            permutation=primary_axis + "".join(other_dimensions),
            unit_factors=tuple(f"{dimension}=1" for dimension in other_dimensions),
            minimum_parallelism_numerator=minimum_parallelism.numerator,
            minimum_parallelism_denominator=minimum_parallelism.denominator,
        ),
        global_temporal=CompiledTemporalConstraint(
            target_role="global_buffer",
            permutation=_OUTER_LOOP_TEMPLATES[policy.outer_loop_order],
        ),
        global_buffer_residency=CompiledDataspaceConstraint(
            keep=keep,
            bypass=bypass,
        ),
    )


class TimeloopV2Compiler:
    """Pure compiler; it canonicalizes decisions and never repairs them."""

    compiler_id = COMPILER_ID
    compiler_version = COMPILER_VERSION
    definition_sha256 = COMPILER_DEFINITION_SHA256

    @staticmethod
    def compile(candidate_value: object, panel: NetworkLayerPanel) -> CompilationResult:
        candidate = normalize_candidate(candidate_value)
        if type(panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        architecture = _compile_architecture(candidate)
        policies = tuple(getattr(candidate, field) for field in POLICY_BLOCK_FIELDS)
        medoids = panel.medoids()
        compiled = tuple(
            _compile_policy(policy, medoid, architecture)
            for policy, medoid in zip(policies, medoids, strict=True)
        )
        plan = CompiledPlan(
            architecture=architecture,
            medoid_0=compiled[0],
            medoid_1=compiled[1],
            medoid_2=compiled[2],
        )
        return CompilationResult(
            plan=plan,
            compiled_plan_sha256=compiled_plan_sha256(plan),
            candidate_sha256=candidate_sha256(candidate),
            panel_sha256=panel_sha256(panel),
        )


def _architecture_records() -> set[bytes]:
    fields = tuple(field for field, _, _ in ARCHITECTURE_FIELD_GRIDS)
    grids = tuple(values for _, values, _ in ARCHITECTURE_FIELD_GRIDS)
    records: set[bytes] = set()
    for values in product(*grids):
        payload = dict(zip(fields, values, strict=True))
        candidate = normalize_candidate(
            {
                **CandidateConfig().model_dump(mode="python"),
                **payload,
            }
        )
        records.add(_canonical_bytes(_compile_architecture(candidate).model_dump()))
    return records


def audit_compiler_injectivity(panel: NetworkLayerPanel) -> dict[str, object]:
    """Exhaust components, never the 6.96e11-element Cartesian product."""

    if type(panel) is not NetworkLayerPanel:
        raise TypeError("panel must be an exact NetworkLayerPanel")
    architecture_records = _architecture_records()
    if len(architecture_records) != architecture_cardinality():
        raise RuntimeError("architecture compiler contains an operational alias")

    policies = iter_policy_blocks()
    palette_counts: dict[str, int] = {}
    for mesh_x in (4, 8, 16, 32):
        for register_enabled in (False, True):
            architecture = CompiledArchitecture(
                global_buffer_depth=512,
                global_buffer_width=128,
                pe_mesh_x=mesh_x,
                datawidth_bits=8,
                register_enabled=register_enabled,
            )
            for medoid_ordinal, medoid in enumerate(panel.medoids()):
                records = {
                    _canonical_bytes(
                        _compile_policy(policy, medoid, architecture).model_dump(
                            mode="python"
                        )
                    )
                    for policy in policies
                }
                key = (
                    f"mesh_{mesh_x}.register_{str(register_enabled).lower()}."
                    f"medoid_{medoid_ordinal}"
                )
                palette_counts[key] = len(records)
                if len(records) != policy_cardinality():
                    raise RuntimeError(
                        f"mapping-policy compiler contains an alias in {key}"
                    )

    exact_count = candidate_cardinality()
    proof_record = {
        "schema_version": 1,
        "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
        "panel_sha256": panel_sha256(panel),
        "architecture_cardinality": architecture_cardinality(),
        "policy_cardinality_per_medoid": policy_cardinality(),
        "medoid_count": len(POLICY_BLOCK_FIELDS),
        "exact_compiled_record_cardinality": exact_count,
        "architecture_records_checked": len(architecture_records),
        "policy_contexts_checked": len(palette_counts),
        "policy_records_checked_per_context": palette_counts,
        "whole_space_enumerated": False,
        "repair_policy": "none",
        "proof_argument": (
            "The architecture projection is injective over 192 records. In every "
            "mesh/register/medoid context, the five policy loci map injectively "
            "to operational temporal, spatial, utilization, residency, and outer "
            "loop constraints. Three operationally distinct medoid slots occupy "
            "distinct plan paths. Therefore the compiler is injective over the "
            "Cartesian product: 192 * 1536^3."
        ),
        "hash_identity_assumption": "SHA-256 collision resistance",
        "timeloop_runtime_feasibility_proven": False,
    }
    proof_record["proof_sha256"] = hashlib.sha256(
        _PROOF_HASH_DOMAIN + _canonical_bytes(proof_record)
    ).hexdigest()
    return proof_record


__all__ = [
    "COMPILER_DEFINITION_SHA256",
    "COMPILER_ID",
    "COMPILER_VERSION",
    "CompilationResult",
    "CompiledArchitecture",
    "CompiledDataspaceConstraint",
    "CompiledPlan",
    "CompiledPolicyPlan",
    "CompiledSpatialConstraint",
    "CompiledTemporalConstraint",
    "TimeloopV2Compiler",
    "audit_compiler_injectivity",
    "canonical_compiled_plan_bytes",
    "compiled_plan_sha256",
]
