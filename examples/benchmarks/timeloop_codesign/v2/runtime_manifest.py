"""Pure translation from compiled v2 plans to pinned Timeloop-v4 inputs."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .compiler import (
    COMPILER_DEFINITION_SHA256,
    CompilationResult,
    CompiledArchitecture,
    CompiledPolicyPlan,
)


RUNTIME_TRANSLATOR_ID = "timeloop_v2_compiled_plan_to_v4_manifest"
RUNTIME_TRANSLATOR_VERSION = 1
RUNTIME_TEMPLATE_ID = "timeloop-python-tutorial-2024-dse"
RUNTIME_TEMPLATE_SHA256 = (
    "603d41cacb09e6de6542fd3531c1132feeb295b1cf7228a8c19e3b8a1fb25fe6"
)
_MANIFEST_HASH_DOMAIN = b"agent-evolve:timeloop-v2-runtime-layer-manifest:v1\x00"
_TRANSLATOR_HASH_DOMAIN = b"agent-evolve:timeloop-v2-runtime-translator:v1\x00"
_DIMENSIONS = ("N", "C", "M", "R", "S", "P", "Q", "G")

_TRANSLATOR_DEFINITION = {
    "translator_id": RUNTIME_TRANSLATOR_ID,
    "translator_version": RUNTIME_TRANSLATOR_VERSION,
    "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
    "template_id": RUNTIME_TEMPLATE_ID,
    "template_sha256": RUNTIME_TEMPLATE_SHA256,
    "target_role_mapping": {
        "register_file": "reg",
        "inter_pe_temporal_level": "PE",
        "processing_element_mesh": "PE",
        "global_buffer": "buffer",
    },
    "fixed_group_semantics": {
        "G": 1,
        "H": 1,
        "W": 1,
        "Hpad": 0,
        "Wpad": 0,
    },
    "permutation_completion": "append_fixed_G",
    "spatial_factor_completion": "append_G=1",
    "minimum_parallelism_translation": (
        "ceil(compiled_minimum_parallelism_fraction_times_pe_mesh_x)"
    ),
    "maximum_parallelism_translation": (
        "explicit_primary_axis_upper_bound_at_physical_pe_mesh_x"
    ),
    "repair_policy": "none_fail_closed",
}
RUNTIME_TRANSLATOR_DEFINITION_SHA256 = hashlib.sha256(
    _TRANSLATOR_HASH_DOMAIN
    + json.dumps(
        _TRANSLATOR_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()

PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class RuntimeTemporalConstraint(_ClosedModel):
    type: Literal["temporal"] = "temporal"
    target: Literal["reg", "PE", "buffer"]
    permutation: tuple[str, ...]
    no_reuse: tuple[Literal["Inputs", "Weights", "Outputs"], ...] = ()

    @model_validator(mode="after")
    def _complete_permutation(self) -> "RuntimeTemporalConstraint":
        if self.permutation != tuple(self.permutation):  # pragma: no cover
            raise ValueError("permutation must be immutable")
        if len(self.permutation) != len(_DIMENSIONS) or set(self.permutation) != set(
            _DIMENSIONS
        ):
            raise ValueError("temporal permutation must contain every runtime rank")
        return self


class RuntimeSpatialConstraint(_ClosedModel):
    type: Literal["spatial"] = "spatial"
    target: Literal["PE"] = "PE"
    permutation: tuple[str, ...]
    factors: tuple[str, ...]
    minimum_parallelism: PositiveStrictInt
    maximum_parallelism: PositiveStrictInt

    @model_validator(mode="after")
    def _validate_spatial(self) -> "RuntimeSpatialConstraint":
        if len(self.permutation) != len(_DIMENSIONS) or set(self.permutation) != set(
            _DIMENSIONS
        ):
            raise ValueError("spatial permutation must contain every runtime rank")
        equalities = tuple(
            item for item in self.factors if item.endswith("=1") and ">=" not in item
        )
        lower_bounds = tuple(item for item in self.factors if ">=" in item)
        upper_bounds = tuple(item for item in self.factors if "<=" in item)
        if len(equalities) != 7 or len(lower_bounds) != 1 or len(upper_bounds) != 1:
            raise ValueError("spatial factors must close seven ranks and bound one")
        primary = self.permutation[0]
        if self.factors[-2:] != (
            f"{primary}>={self.minimum_parallelism}",
            f"{primary}<={self.maximum_parallelism}",
        ):
            raise ValueError("parallelism bounds must apply to the primary rank")
        if self.minimum_parallelism > self.maximum_parallelism:
            raise ValueError("parallelism lower bound exceeds the upper bound")
        return self


class RuntimeDataspaceConstraint(_ClosedModel):
    type: Literal["dataspace"] = "dataspace"
    target: Literal["buffer"] = "buffer"
    keep: tuple[Literal["Inputs", "Weights", "Outputs"], ...]
    bypass: tuple[Literal["Inputs", "Weights", "Outputs"], ...]

    @model_validator(mode="after")
    def _partition(self) -> "RuntimeDataspaceConstraint":
        if set(self.keep).intersection(self.bypass) or set(self.keep).union(
            self.bypass
        ) != {"Inputs", "Weights", "Outputs"}:
            raise ValueError("runtime keep/bypass fields must partition dataspaces")
        return self


RuntimeConstraint = (
    RuntimeTemporalConstraint | RuntimeSpatialConstraint | RuntimeDataspaceConstraint
)


class RuntimeLayerManifest(_ClosedModel):
    schema_version: Literal[1] = 1
    translator_id: Literal[RUNTIME_TRANSLATOR_ID] = RUNTIME_TRANSLATOR_ID
    translator_version: Literal[RUNTIME_TRANSLATOR_VERSION] = RUNTIME_TRANSLATOR_VERSION
    translator_definition_sha256: Literal[RUNTIME_TRANSLATOR_DEFINITION_SHA256] = (
        RUNTIME_TRANSLATOR_DEFINITION_SHA256
    )
    template_id: Literal[RUNTIME_TEMPLATE_ID] = RUNTIME_TEMPLATE_ID
    template_sha256: Literal[RUNTIME_TEMPLATE_SHA256] = RUNTIME_TEMPLATE_SHA256
    compiled_plan_sha256: str
    candidate_sha256: str
    panel_sha256: str
    medoid_ordinal: Literal[0, 1, 2]
    layer_multiplicity: PositiveStrictInt
    architecture: CompiledArchitecture
    problem_instance: dict[str, int]
    constraints: tuple[RuntimeConstraint, ...]

    @model_validator(mode="after")
    def _validate_closed_runtime(self) -> "RuntimeLayerManifest":
        for value in (
            self.compiled_plan_sha256,
            self.candidate_sha256,
            self.panel_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("runtime provenance must use lowercase SHA-256")
        expected_problem = {
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
            "G",
            "H",
            "W",
            "Hpad",
            "Wpad",
        }
        if set(self.problem_instance) != expected_problem:
            raise ValueError("runtime problem has missing or extra fields")
        if (
            any(
                type(value) is not int or value <= 0
                for key, value in self.problem_instance.items()
                if key not in {"Hpad", "Wpad"}
            )
            or self.problem_instance["Hpad"] != 0
            or self.problem_instance["Wpad"] != 0
        ):
            raise ValueError("runtime problem dimensions are invalid")
        if len(self.constraints) != 4:
            raise ValueError("each layer requires exactly four runtime constraints")
        local, spatial, global_temporal, dataspace = self.constraints
        if not isinstance(local, RuntimeTemporalConstraint) or local.target not in {
            "reg",
            "PE",
        }:
            raise ValueError("first constraint must be the local temporal policy")
        if not isinstance(spatial, RuntimeSpatialConstraint):
            raise ValueError("second constraint must be the PE spatial policy")
        if (
            not isinstance(global_temporal, RuntimeTemporalConstraint)
            or global_temporal.target != "buffer"
        ):
            raise ValueError("third constraint must be the global temporal policy")
        if not isinstance(dataspace, RuntimeDataspaceConstraint):
            raise ValueError("fourth constraint must be the buffer residency policy")
        if local.target != (
            "reg" if self.architecture.register_enabled else "PE"
        ):
            raise ValueError("local temporal target disagrees with architecture")
        if spatial.minimum_parallelism > self.architecture.pe_mesh_x:
            raise ValueError("minimum parallelism exceeds the PE mesh")
        if spatial.maximum_parallelism != self.architecture.pe_mesh_x:
            raise ValueError("maximum parallelism must equal the physical PE mesh")
        return self


def canonical_runtime_layer_manifest_bytes(manifest: RuntimeLayerManifest) -> bytes:
    if type(manifest) is not RuntimeLayerManifest:
        raise TypeError("manifest must be an exact RuntimeLayerManifest")
    return json.dumps(
        manifest.model_dump(mode="python"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def runtime_layer_manifest_sha256(manifest: RuntimeLayerManifest) -> str:
    return hashlib.sha256(
        _MANIFEST_HASH_DOMAIN + canonical_runtime_layer_manifest_bytes(manifest)
    ).hexdigest()


def _runtime_problem(policy: CompiledPolicyPlan) -> dict[str, int]:
    return {
        **policy.problem_instance,
        "G": 1,
        "H": 1,
        "W": 1,
        "Hpad": 0,
        "Wpad": 0,
    }


def _runtime_constraints(
    policy: CompiledPolicyPlan,
    architecture: CompiledArchitecture,
) -> tuple[RuntimeConstraint, ...]:
    spatial = policy.spatial
    primary_axis = spatial.permutation[0]
    fraction = Fraction(
        spatial.minimum_parallelism_numerator,
        spatial.minimum_parallelism_denominator,
    )
    minimum_parallelism = math.ceil(fraction * architecture.pe_mesh_x)
    local_target = (
        "reg" if architecture.register_enabled else "PE"
    )
    return (
        RuntimeTemporalConstraint(
            target=local_target,
            permutation=tuple(policy.local_temporal.permutation) + ("G",),
            no_reuse=policy.local_temporal.no_reuse,
        ),
        RuntimeSpatialConstraint(
            permutation=tuple(spatial.permutation) + ("G",),
            factors=(
                *spatial.unit_factors,
                "G=1",
                f"{primary_axis}>={minimum_parallelism}",
                f"{primary_axis}<={architecture.pe_mesh_x}",
            ),
            minimum_parallelism=minimum_parallelism,
            maximum_parallelism=architecture.pe_mesh_x,
        ),
        RuntimeTemporalConstraint(
            target="buffer",
            permutation=tuple(policy.global_temporal.permutation) + ("G",),
            no_reuse=policy.global_temporal.no_reuse,
        ),
        RuntimeDataspaceConstraint(
            keep=policy.global_buffer_residency.keep,
            bypass=policy.global_buffer_residency.bypass,
        ),
    )


def compile_runtime_layer_manifests(
    compilation: CompilationResult,
) -> tuple[RuntimeLayerManifest, RuntimeLayerManifest, RuntimeLayerManifest]:
    """Translate all three medoid plans without changing any selected decision."""

    if type(compilation) is not CompilationResult:
        raise TypeError("compilation must be an exact CompilationResult")
    architecture = compilation.plan.architecture
    policies = (
        compilation.plan.medoid_0,
        compilation.plan.medoid_1,
        compilation.plan.medoid_2,
    )
    manifests = tuple(
        RuntimeLayerManifest(
            compiled_plan_sha256=compilation.compiled_plan_sha256,
            candidate_sha256=compilation.candidate_sha256,
            panel_sha256=compilation.panel_sha256,
            medoid_ordinal=ordinal,
            layer_multiplicity=policy.layer_multiplicity,
            architecture=architecture,
            problem_instance=_runtime_problem(policy),
            constraints=_runtime_constraints(policy, architecture),
        )
        for ordinal, policy in enumerate(policies)
    )
    return manifests  # type: ignore[return-value]


__all__ = [
    "RUNTIME_TEMPLATE_ID",
    "RUNTIME_TEMPLATE_SHA256",
    "RUNTIME_TRANSLATOR_DEFINITION_SHA256",
    "RUNTIME_TRANSLATOR_ID",
    "RUNTIME_TRANSLATOR_VERSION",
    "RuntimeConstraint",
    "RuntimeDataspaceConstraint",
    "RuntimeLayerManifest",
    "RuntimeSpatialConstraint",
    "RuntimeTemporalConstraint",
    "canonical_runtime_layer_manifest_bytes",
    "compile_runtime_layer_manifests",
    "runtime_layer_manifest_sha256",
]
