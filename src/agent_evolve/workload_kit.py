"""Small public integration surface for campaign-capable workloads.

The strict campaign ports deliberately authenticate every boundary.  Benchmark
authors should not need to assemble those ports directly, however.  A
``WorkloadKit`` names the four facts a new domain must provide:

1. an :class:`~agent_evolve.agentic.AgenticBenchmark`;
2. one or more canonical seeds;
3. a finite variation catalog (selected by ID only when several exist); and
4. externally acquired evaluator/resource admission receipts.

Evidence projections and workload semantics prompting are optional extension
points.  Omitting evidence selects a workload-neutral, schema-derived empty
memory/context projection.  The optional prompt extension remains governed by
``workload_prompt`` provenance and ablation rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import CampaignSeed
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.workload_prompt import WorkloadPromptExtensionView


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_WORKLOAD_DEFINITION_DOMAIN = b"agent-evolve:workload-kit-definition:v1\x00"
_INTEGRATION_RECEIPT_DOMAIN = b"agent-evolve:workload-kit-receipt:v1\x00"
GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID = "generic_schema_evidence"
# v2: the default projection now emits ONE schema-derived bootstrap card.
#
# v1 returned no cards, and `CampaignPortfolioWaveContext` requires
# `evidence_cards` to be a non-empty tuple, so **every workload that omitted an
# evidence projection was unable to run a portfolio campaign at all** -- it
# raised "portfolio context requires evidence cards" at the first stage.  That
# is four shipped workloads, not one: analog_sizing, heat2d, pybamm_fastcharge
# and scip_miplib all report `uses_default_schema_evidence`.  It stayed
# invisible because every workload with a hand-written runner supplies its own
# projection, so the default was never driven to a portfolio stage until a
# second workload went through the generic driver.
#
# The card is derived entirely from material already in scope -- the benchmark
# descriptor, the finite variation contract and the parent/memory hashes -- so
# the projection remains workload-neutral and carries no workload constant.
GENERIC_SCHEMA_EVIDENCE_PROJECTION_VERSION = 2
GENERIC_SCHEMA_EVIDENCE_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:generic-schema-evidence:v2;"
    b"empty-bootstrap-memory=true;objective-declarations=true;"
    b"catalog-identity-and-cardinality=true;parent-and-memory-hashes=true;"
    b"workload-prose=false;cards=one-schema-derived-bootstrap-card"
).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def campaign_seed(
    seed_id: str,
    candidate: BaseModel | FrozenJsonObject | dict[str, object],
) -> CampaignSeed:
    """Freeze one named candidate into the strict campaign seed type."""

    if isinstance(candidate, BaseModel):
        candidate = candidate.model_dump(mode="python", by_alias=False)
    if type(candidate) not in {dict, FrozenJsonObject}:
        raise TypeError(
            "candidate must be an exact dict, FrozenJsonObject, or Pydantic model"
        )
    frozen = freeze_json(candidate)
    if type(frozen) is not FrozenJsonObject:
        raise TypeError("a campaign seed candidate must be a JSON object")
    return CampaignSeed(seed_id=seed_id, configuration=frozen)


def _generic_memory(benchmark, session, seeds) -> FrozenJsonObject:
    del benchmark
    descriptor = thaw_json(session.benchmark)
    record = {
        "schema_version": 1,
        "projection_id": GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID,
        "workload_id": descriptor["workload_id"],
        "objective_ids": [item["name"] for item in descriptor["objectives"]],
        "seed_configuration_sha256s": [
            seed.configuration_sha256 for seed in seeds.seeds
        ],
        "insights": [],
    }
    frozen = freeze_json(record)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover
        raise AssertionError("generic memory did not freeze as an object")
    return frozen


def _generic_context(
    benchmark,
    session,
    parent,
    variation,
    memory,
) -> FrozenJsonObject:
    del benchmark
    descriptor = thaw_json(session.benchmark)
    record = {
        "schema_version": 1,
        "projection_id": GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID,
        "workload_id": descriptor["workload_id"],
        "objectives": descriptor["objectives"],
        "optimization_semantics_identity": descriptor.get(
            "optimization_semantics_identity"
        ),
        "action_semantics_identity": descriptor.get("action_semantics_identity"),
        "parent_configuration_sha256": typed_json_sha256(parent),
        "memory_sha256": typed_json_sha256(memory),
        "finite_variation": {
            "catalog_id": variation.contract.catalog_id,
            "catalog_version": variation.contract.catalog_version,
            "catalog_definition_sha256": (
                variation.contract.catalog_definition_sha256
            ),
            "contract_identity_sha256": variation.contract.identity_sha256,
            "eligible_option_count": len(variation.contract.options),
        },
    }
    frozen = freeze_json(record)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover
        raise AssertionError("generic context did not freeze as an object")
    return frozen


def _generic_cards(
    benchmark,
    session,
    parent,
    variation,
    memory,
) -> tuple[FrozenJsonObject, ...]:
    """One schema-derived bootstrap card.

    The portfolio wave context requires at least one evidence card, so a
    projection that returns none cannot reach a portfolio stage.  This card
    states only what the schema already declares: which objectives are being
    optimized, which finite catalogue the options come from, how many options
    are eligible, and which parent and memory the selection is conditioned on.
    It asserts nothing about the workload's semantics and contains no workload
    constant -- a workload that wants richer evidence supplies its own
    projection, which is what every hand-written runner already does.
    """

    del benchmark
    descriptor = thaw_json(session.benchmark)
    contract = variation.contract
    record = {
        "schema_version": 1,
        "card_kind": "schema_bootstrap",
        "projection_id": GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID,
        "claim": (
            "Selection is conditioned on the declared objectives and the sealed "
            "finite option contract below; no workload-specific evidence has "
            "been supplied by this projection."
        ),
        "workload_id": descriptor["workload_id"],
        "objectives": descriptor["objectives"],
        "finite_variation": {
            "catalog_id": contract.catalog_id,
            "catalog_version": contract.catalog_version,
            "catalog_definition_sha256": contract.catalog_definition_sha256,
            "contract_identity_sha256": contract.identity_sha256,
            "eligible_option_count": len(contract.options),
        },
        "parent_configuration_sha256": typed_json_sha256(parent),
        "memory_sha256": typed_json_sha256(memory),
    }
    frozen = freeze_json(record)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover
        raise AssertionError("generic card did not freeze as an object")
    return (frozen,)


def generic_schema_evidence_projections() -> AgenticCampaignEvidenceProjections:
    """Return the workload-neutral default evidence implementation."""

    return AgenticCampaignEvidenceProjections(
        projection_id=GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID,
        projection_version=GENERIC_SCHEMA_EVIDENCE_PROJECTION_VERSION,
        definition_sha256=(
            GENERIC_SCHEMA_EVIDENCE_PROJECTION_DEFINITION_SHA256
        ),
        initialize_memory=_generic_memory,
        context=_generic_context,
        cards=_generic_cards,
    )


@dataclass(frozen=True, slots=True)
class WorkloadKit:
    """Declarative high-level adapter compiled into authenticated campaign ports."""

    workload_id: str
    workload_version: int
    benchmark: AgenticBenchmark
    seeds: tuple[CampaignSeed, ...]
    evaluator_concurrency_cap: int
    evaluator_preflight_receipt: FrozenJsonObject
    resource_lease_receipt: FrozenJsonObject
    finite_catalog_id: str | None = None
    evidence: AgenticCampaignEvidenceProjections | None = None
    prompt_extension: WorkloadPromptExtensionView | None = None

    def __post_init__(self) -> None:
        if type(self.workload_id) is not str or _TOKEN.fullmatch(
            self.workload_id
        ) is None:
            raise ValueError("workload_id must use the closed campaign token grammar")
        if type(self.workload_version) is not int or self.workload_version <= 0:
            raise ValueError("workload_version must be a positive exact integer")
        if type(self.benchmark) is not AgenticBenchmark:
            raise TypeError("benchmark must be an exact AgenticBenchmark")
        self.benchmark.validate_binding()
        if type(self.seeds) is not tuple or not self.seeds:
            raise ValueError("seeds must be a non-empty exact tuple")
        if any(type(seed) is not CampaignSeed for seed in self.seeds):
            raise TypeError("seeds must contain exact CampaignSeed values")
        if (
            type(self.evaluator_concurrency_cap) is not int
            or self.evaluator_concurrency_cap <= 0
        ):
            raise ValueError("evaluator_concurrency_cap must be positive")
        for name in (
            "evaluator_preflight_receipt",
            "resource_lease_receipt",
        ):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
                raise TypeError(f"{name} must be an exact frozen typed-JSON object")
        if self.finite_catalog_id is not None and type(
            self.finite_catalog_id
        ) is not str:
            raise TypeError("finite_catalog_id must be an exact string or None")
        if self.evidence is not None and type(
            self.evidence
        ) is not AgenticCampaignEvidenceProjections:
            raise TypeError(
                "evidence must be exact AgenticCampaignEvidenceProjections or None"
            )
        if self.prompt_extension is not None and type(
            self.prompt_extension
        ) is not WorkloadPromptExtensionView:
            raise TypeError(
                "prompt_extension must be exact WorkloadPromptExtensionView or None"
            )
        # Compile eagerly so ambiguous catalogs and noncanonical seeds fail at
        # the public construction boundary rather than during a paid run.
        self.to_campaign_workload()

    @property
    def selected_finite_catalog_id(self) -> str:
        identities = self.benchmark.finite_variation_catalog_identities
        if self.finite_catalog_id is not None:
            return self.finite_catalog_id
        if len(identities) != 1:
            raise ValueError(
                "finite_catalog_id is required when the benchmark publishes "
                "zero or multiple catalogs"
            )
        return identities[0][0]

    @property
    def workload_definition_sha256(self) -> str:
        """Identity of this versioned public integration contract."""

        return _hash(
            _WORKLOAD_DEFINITION_DOMAIN,
            {
                "schema_version": 1,
                "workload_id": self.workload_id,
                "workload_version": self.workload_version,
                "adapter": "workload_kit",
            },
        )

    def to_campaign_workload(self) -> AgenticCampaignWorkloadConfig:
        """Compile the small public declaration into the strict inverted API."""

        evidence = self.evidence or generic_schema_evidence_projections()
        return AgenticCampaignWorkloadConfig(
            workload_id=self.workload_id,
            workload_version=self.workload_version,
            definition_sha256=self.workload_definition_sha256,
            benchmark=self.benchmark,
            seeds=self.seeds,
            finite_catalog_id=self.selected_finite_catalog_id,
            evaluator_concurrency_cap=self.evaluator_concurrency_cap,
            evaluator_preflight_receipt=self.evaluator_preflight_receipt,
            resource_lease_receipt=self.resource_lease_receipt,
            evidence=evidence,
            prompt_extension=self.prompt_extension,
        )

    def integration_receipt(self) -> FrozenJsonObject:
        """Publish the inspectable adapter obligations and extension usage."""

        config = self.to_campaign_workload()
        record: dict[str, object] = {
            "schema_version": 1,
            "adapter": "workload_kit",
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "required_obligations": [
                "agentic_benchmark",
                "canonical_seeds",
                "finite_variation_catalog",
                "evaluator_resource_admission",
            ],
            "required_obligation_count": 4,
            "seed_count": len(self.seeds),
            "finite_catalog_id": self.selected_finite_catalog_id,
            "uses_default_schema_evidence": self.evidence is None,
            "uses_custom_evidence_projection": self.evidence is not None,
            "uses_optional_prompt_extension": self.prompt_extension is not None,
            "prompt_extension_view_sha256": (
                None
                if self.prompt_extension is None
                else self.prompt_extension.view_sha256
            ),
            "campaign_configuration_sha256": config.configuration_sha256,
        }
        record["receipt_sha256"] = _hash(_INTEGRATION_RECEIPT_DOMAIN, record)
        frozen = freeze_json(record)
        if type(frozen) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("integration receipt did not freeze as an object")
        return frozen


__all__ = [
    "GENERIC_SCHEMA_EVIDENCE_PROJECTION_DEFINITION_SHA256",
    "GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID",
    "GENERIC_SCHEMA_EVIDENCE_PROJECTION_VERSION",
    "WorkloadKit",
    "campaign_seed",
    "generic_schema_evidence_projections",
]
