"""Public bridge from :class:`AgenticBenchmark` to campaign workload ports.

The campaign application owns chronology, budgets, and runtime preflight.  A
benchmark owns candidate semantics, a finite variation catalog, seed
configurations, evaluator-resource facts, and prompt evidence.  This module is
the narrow adapter between those two boundaries.

Construction and :meth:`EvolutionCampaign.prepare` are deliberately
provider- and evaluator-free.  The selected finite catalog is materialized
only when ``CampaignCatalogPort.bind`` is called for a concrete parent after
preparation.  Evidence memory, context, and cards are likewise delegated to
injected projections rather than embedding workload rules in the campaign
orchestrator.

Library integrations should import :class:`AgenticCampaignWorkloadConfig` and
:class:`AgenticCampaignEvidenceProjections` from the top-level
``agent_evolve`` facade.  This implementation module remains importable for
type-directed tooling, but it is not the intended composition root.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from agent_evolve.agentic import (
    AgenticBenchmark,
    FiniteVariationContract,
    OptionPhenotypeBinding,
    PhenotypeIdentity,
    eligible_finite_variation_view,
)
from agent_evolve.application.evolution_campaign import (
    BenchmarkSessionRequest,
    CampaignBenchmarkSession,
    CampaignSeed,
    CampaignSeedBatch,
    CampaignWorkloadPorts,
    ParentVariationBinding,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.workload_prompt import (
    WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY,
    WorkloadPromptExtensionView,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_CONFIG_DOMAIN = b"agent-evolve:agentic-campaign-workload-config:v1\x00"
_PORT_DOMAIN = b"agent-evolve:agentic-campaign-workload-port:v1\x00"
_PORT_ADAPTER_IMPLEMENTATION_REVISION = 2
_BINDING_KEY_DOMAIN = (
    b"agent-evolve:agentic-campaign-workload-binding-key:v1\x00"
)
_OPTION_PHENOTYPE_SET_DOMAIN = (
    b"agent-evolve:agentic-campaign-option-phenotype-set:v1\x00"
)
_ISSUED_BINDING_DOMAIN = (
    b"agent-evolve:agentic-campaign-issued-binding:v1\x00"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_token(value: object, *, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed campaign-token grammar")
    return value


def _require_object(value: object, *, name: str) -> FrozenJsonObject:
    if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
        raise TypeError(f"{name} must be an exact frozen typed-JSON object")
    return value


MemoryProjection = Callable[
    [AgenticBenchmark, CampaignBenchmarkSession, CampaignSeedBatch],
    FrozenJsonObject,
]
ContextProjection = Callable[
    [
        AgenticBenchmark,
        CampaignBenchmarkSession,
        FrozenJsonObject,
        ParentVariationBinding,
        FrozenJsonObject,
    ],
    FrozenJsonObject,
]
CardProjection = Callable[
    [
        AgenticBenchmark,
        CampaignBenchmarkSession,
        FrozenJsonObject,
        ParentVariationBinding,
        FrozenJsonObject,
    ],
    tuple[FrozenJsonObject, ...],
]


@dataclass(frozen=True, slots=True)
class AgenticCampaignEvidenceProjections:
    """Injected workload evidence projections with an immutable identity."""

    projection_id: str
    projection_version: int
    definition_sha256: str
    initialize_memory: MemoryProjection
    context: ContextProjection
    cards: CardProjection

    def __post_init__(self) -> None:
        _require_token(self.projection_id, name="projection_id")
        if type(self.projection_version) is not int or self.projection_version <= 0:
            raise ValueError("projection_version must be a positive exact integer")
        _require_sha256(self.definition_sha256, name="definition_sha256")
        for name in ("initialize_memory", "context", "cards"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class AgenticCampaignWorkloadConfig:
    """Complete configuration for one benchmark-owned campaign boundary.

    ``evaluator_preflight_receipt`` and ``resource_lease_receipt`` are facts
    acquired outside this adapter.  Supplying those immutable receipts keeps
    preparation free of evaluator work while still making resource admission
    replay-identifiable.
    """

    workload_id: str
    workload_version: int
    definition_sha256: str
    benchmark: AgenticBenchmark
    seeds: tuple[CampaignSeed, ...]
    finite_catalog_id: str
    evaluator_concurrency_cap: int
    evaluator_preflight_receipt: FrozenJsonObject
    resource_lease_receipt: FrozenJsonObject
    evidence: AgenticCampaignEvidenceProjections
    prompt_extension: WorkloadPromptExtensionView | None = None

    def __post_init__(self) -> None:
        _require_token(self.workload_id, name="workload_id")
        if type(self.workload_version) is not int or self.workload_version <= 0:
            raise ValueError("workload_version must be a positive exact integer")
        _require_sha256(self.definition_sha256, name="definition_sha256")
        if type(self.benchmark) is not AgenticBenchmark:
            raise TypeError("benchmark must be an exact AgenticBenchmark")
        self.benchmark.validate_binding()
        if type(self.seeds) is not tuple or not self.seeds:
            raise ValueError("seeds must be a non-empty exact tuple")
        if any(type(seed) is not CampaignSeed for seed in self.seeds):
            raise TypeError("seeds must contain exact CampaignSeed values")
        if len({seed.seed_id for seed in self.seeds}) != len(self.seeds):
            raise ValueError("seed IDs must be unique")
        if len({seed.configuration_sha256 for seed in self.seeds}) != len(self.seeds):
            raise ValueError("seed configurations must be unique")
        for seed in self.seeds:
            CampaignSeed.__post_init__(seed)
            self._validate_seed_schema(seed)
        _require_token(self.finite_catalog_id, name="finite_catalog_id")
        identities = self.benchmark.finite_variation_catalog_identities
        if sum(identity[0] == self.finite_catalog_id for identity in identities) != 1:
            raise ValueError(
                "finite_catalog_id must identify exactly one benchmark catalog"
            )
        if (
            type(self.evaluator_concurrency_cap) is not int
            or self.evaluator_concurrency_cap <= 0
        ):
            raise ValueError("evaluator_concurrency_cap must be positive")
        _require_object(
            self.evaluator_preflight_receipt,
            name="evaluator_preflight_receipt",
        )
        _require_object(
            self.resource_lease_receipt,
            name="resource_lease_receipt",
        )
        if type(self.evidence) is not AgenticCampaignEvidenceProjections:
            raise TypeError("evidence must be exact AgenticCampaignEvidenceProjections")
        AgenticCampaignEvidenceProjections.__post_init__(self.evidence)
        if self.prompt_extension is not None:
            if type(self.prompt_extension) is not WorkloadPromptExtensionView:
                raise TypeError(
                    "prompt_extension must be an exact "
                    "WorkloadPromptExtensionView or None"
                )
            WorkloadPromptExtensionView.__post_init__(self.prompt_extension)
        # Force construction now so an unsupported benchmark fact fails before
        # a campaign can acquire a session.
        self._benchmark_record()

    def _validate_seed_schema(self, seed: CampaignSeed) -> None:
        candidate_model = self.benchmark.problem.candidate_model
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model, BaseModel
        ):
            raise TypeError("benchmark candidate_model must be a Pydantic model")
        parsed = candidate_model.model_validate(
            thaw_json(seed.configuration),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        round_trip = freeze_json(parsed.model_dump(mode="python", by_alias=False))
        if type(round_trip) is not FrozenJsonObject:
            raise TypeError("candidate_model did not publish an object")
        if typed_json_sha256(round_trip) != seed.configuration_sha256:
            raise ValueError(
                f"seed {seed.seed_id!r} is not canonical under candidate_model"
            )

    @property
    def selected_catalog_identity(self) -> tuple[str, int, str]:
        return next(
            identity
            for identity in self.benchmark.finite_variation_catalog_identities
            if identity[0] == self.finite_catalog_id
        )

    def _benchmark_record(self) -> dict[str, object]:
        self.benchmark.validate_binding()
        candidate_model = self.benchmark.problem.candidate_model
        candidate_schema = freeze_json(
            candidate_model.model_json_schema(by_alias=False)
        )
        if type(candidate_schema) is not FrozenJsonObject:
            raise TypeError("candidate schema must be an object")
        evaluator_identity = (
            None
            if self.benchmark.detailed_evaluator is None
            else self.benchmark.detailed_evaluator.evaluator_identity.to_record()
        )
        relation_identity = (
            None
            if self.benchmark.outcome_relation is None
            else self.benchmark.outcome_relation.to_record()
        )
        optimization_semantics = self.benchmark.optimization_semantics
        action_semantics = self.benchmark.action_semantics
        selected_id, selected_version, selected_definition = (
            self.selected_catalog_identity
        )
        record: dict[str, object] = {
            "schema_version": 1,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "definition_sha256": self.definition_sha256,
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in self.benchmark.objectives
            ],
            "candidate_schema_sha256": typed_json_sha256(candidate_schema),
            "reward_binding_sha256": self.benchmark.reward.binding_sha256,
            "evaluator_identity": evaluator_identity,
            "outcome_relation": relation_identity,
            "phenotype_identity": {
                "policy_id": self.benchmark.phenotype_identity.policy_id,
                "policy_version": self.benchmark.phenotype_identity.policy_version,
            },
            "optimization_semantics_identity": (
                None
                if optimization_semantics is None
                else list(optimization_semantics.identity)
            ),
            "action_semantics_identity": (
                None if action_semantics is None else list(action_semantics.identity)
            ),
            "selected_finite_catalog": {
                "catalog_id": selected_id,
                "catalog_version": selected_version,
                "definition_sha256": selected_definition,
            },
            "finite_catalog_identities": [
                {
                    "catalog_id": catalog_id,
                    "catalog_version": catalog_version,
                    "definition_sha256": definition_sha256,
                }
                for catalog_id, catalog_version, definition_sha256 in (
                    self.benchmark.finite_variation_catalog_identities
                )
            ],
        }
        if self.benchmark.objective_resolution is not None:
            policy = self.benchmark.objective_resolution
            record["objective_resolution"] = {
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
                "definition_sha256": policy.definition_sha256,
            }
        if self.prompt_extension is not None:
            record["workload_prompt_extension"] = (
                self.prompt_extension.to_binding_record()
            )
        return record

    @property
    def benchmark_record(self) -> FrozenJsonObject:
        value = freeze_json(self._benchmark_record())
        if type(value) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("benchmark record did not freeze as an object")
        return value

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "schema_version": 1,
            "benchmark": thaw_json(self.benchmark_record),
            "benchmark_sha256": typed_json_sha256(self.benchmark_record),
            "seeds": [seed.to_record() for seed in self.seeds],
            "finite_catalog_id": self.finite_catalog_id,
            "evaluator_concurrency_cap": self.evaluator_concurrency_cap,
            "evaluator_preflight_receipt_sha256": typed_json_sha256(
                self.evaluator_preflight_receipt
            ),
            "resource_lease_receipt_sha256": typed_json_sha256(
                self.resource_lease_receipt
            ),
            "evidence": self.evidence.to_record(),
        }
        if self.prompt_extension is not None:
            record["workload_prompt_extension"] = (
                self.prompt_extension.to_binding_record()
            )
        return record

    @property
    def configuration_sha256(self) -> str:
        return _sha256(_CONFIG_DOMAIN, self.to_record())

    def build_ports(self) -> CampaignWorkloadPorts:
        """Create the four campaign ports without opening or evaluating anything."""

        self.__post_init__()
        registry = _AuthenticatedBindingRegistry(self)
        return CampaignWorkloadPorts(
            benchmark=_AgenticBenchmarkSessionPort(self, registry),
            seeds=_AgenticSeedPort(self, registry),
            catalog=_AgenticCatalogPort(self, registry),
            evidence=_AgenticEvidencePort(self, registry),
        )


def _port_definition(config: AgenticCampaignWorkloadConfig, role: str) -> str:
    return _sha256(
        _PORT_DOMAIN,
        {
            "schema_version": 1,
            "adapter": "agentic_benchmark_campaign_workload",
            "adapter_implementation_revision": _PORT_ADAPTER_IMPLEMENTATION_REVISION,
            "role": role,
            "configuration_sha256": config.configuration_sha256,
        },
    )


def _benchmark_option_phenotype_bindings(
    benchmark: AgenticBenchmark,
    contract: FiniteVariationContract,
) -> tuple[OptionPhenotypeBinding, ...]:
    """Project every child through the benchmark's declared phenotype law."""

    benchmark.validate_binding()
    policy = benchmark.phenotype_identity
    expected_policy = (policy.policy_id, policy.policy_version)
    bindings = []
    for option in contract.options:
        identity = policy.identify(thaw_json(option.child_configuration))
        if type(identity) is not PhenotypeIdentity:
            raise TypeError(
                "benchmark phenotype policy must return exact PhenotypeIdentity"
            )
        PhenotypeIdentity.__post_init__(identity)
        if (identity.policy_id, identity.policy_version) != expected_policy:
            raise ValueError("phenotype policy returned a foreign identity law")
        bindings.append(
            OptionPhenotypeBinding(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                phenotype_identity_sha256=identity.value_sha256,
            )
        )
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class _BaseBindingKey:
    """Exact immutable authority key before a novelty cutoff is applied."""

    configuration_sha256: str
    benchmark_sha256: str
    phenotype_policy_id: str
    phenotype_policy_version: int
    catalog_id: str
    catalog_version: int
    catalog_definition_sha256: str
    parent_configuration_sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "configuration_sha256": self.configuration_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "phenotype_policy_id": self.phenotype_policy_id,
            "phenotype_policy_version": self.phenotype_policy_version,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "catalog_definition_sha256": self.catalog_definition_sha256,
            "parent_configuration_sha256": self.parent_configuration_sha256,
        }


@dataclass(frozen=True, slots=True)
class _BaseBinding:
    contract: FiniteVariationContract
    option_phenotypes: tuple[OptionPhenotypeBinding, ...]
    contract_identity_sha256: str
    option_phenotypes_sha256: str


@dataclass(frozen=True, slots=True)
class _IssuedBinding:
    binding: ParentVariationBinding
    key_sha256: str
    base_contract_identity_sha256: str
    option_phenotypes_sha256: str
    eligible_contract_identity_sha256: str
    eligibility_receipt_sha256: str
    binding_authenticator_sha256: str


class _AuthenticatedBindingRegistry:
    """Private provenance and memoization boundary shared by one port set.

    Structural equality is intentionally insufficient.  Evidence projections
    accept only the exact object issued by their sibling catalog port and also
    recheck a cryptographic fingerprint, closing both coherent forgery and
    post-issuance mutation.  The registry is private to one ``build_ports``
    invocation, so bindings cannot cross a campaign port/session boundary.
    """

    __slots__ = (
        "_base_bindings",
        "_benchmark_sha256",
        "_bindings",
        "_catalog_identity",
        "_config",
        "_configuration_sha256",
        "_phenotype_policy_identity",
        "_sessions",
    )

    def __init__(self, config: AgenticCampaignWorkloadConfig) -> None:
        self._config = config
        self._configuration_sha256 = config.configuration_sha256
        self._benchmark_sha256 = typed_json_sha256(config.benchmark_record)
        policy = config.benchmark.phenotype_identity
        self._phenotype_policy_identity = (policy.policy_id, policy.policy_version)
        self._catalog_identity = config.selected_catalog_identity
        self._base_bindings: dict[_BaseBindingKey, _BaseBinding] = {}
        self._bindings: dict[tuple[_BaseBindingKey, tuple[str, ...]], _IssuedBinding] = {}
        self._sessions: dict[str, tuple[CampaignBenchmarkSession, str]] = {}

    def issue_session(
        self,
        session: CampaignBenchmarkSession,
    ) -> CampaignBenchmarkSession:
        request_sha256 = session.request_sha256
        cached = self._sessions.get(request_sha256)
        if cached is not None:
            cached_session, session_sha256 = cached
            if cached_session.session_sha256 != session_sha256:
                raise ValueError("cached campaign session was mutated")
            return cached_session
        self._sessions[request_sha256] = (session, session.session_sha256)
        return session

    def require_session(self, session: CampaignBenchmarkSession) -> None:
        cached = self._sessions.get(session.request_sha256)
        if cached is None or cached[0] is not session:
            raise ValueError(
                "session was not issued by this exact campaign port set"
            )
        if session.session_sha256 != cached[1]:
            raise ValueError("issued campaign session failed authentication")

    def _validate_authority(self, benchmark: FrozenJsonObject) -> None:
        if benchmark != self._config.benchmark_record:
            raise ValueError("catalog request is bound to a foreign benchmark")
        if self._config.configuration_sha256 != self._configuration_sha256:
            raise ValueError("campaign workload configuration drifted after port build")
        if typed_json_sha256(benchmark) != self._benchmark_sha256:
            raise ValueError("campaign benchmark identity drifted after port build")
        policy = self._config.benchmark.phenotype_identity
        if (policy.policy_id, policy.policy_version) != (
            self._phenotype_policy_identity
        ):
            raise ValueError("phenotype identity policy drifted after port build")
        if self._config.selected_catalog_identity != self._catalog_identity:
            raise ValueError("finite catalog identity drifted after port build")

    def _base_key(
        self,
        benchmark: FrozenJsonObject,
        parent: FrozenJsonObject,
    ) -> _BaseBindingKey:
        self._validate_authority(benchmark)
        policy_id, policy_version = self._phenotype_policy_identity
        catalog_id, catalog_version, catalog_definition_sha256 = (
            self._catalog_identity
        )
        return _BaseBindingKey(
            configuration_sha256=self._configuration_sha256,
            benchmark_sha256=self._benchmark_sha256,
            phenotype_policy_id=policy_id,
            phenotype_policy_version=policy_version,
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            catalog_definition_sha256=catalog_definition_sha256,
            parent_configuration_sha256=typed_json_sha256(parent),
        )

    @staticmethod
    def _option_phenotypes_sha256(
        values: tuple[OptionPhenotypeBinding, ...],
    ) -> str:
        return _sha256(
            _OPTION_PHENOTYPE_SET_DOMAIN,
            [value.to_record() for value in values],
        )

    @staticmethod
    def _key_sha256(
        base_key: _BaseBindingKey,
        known_phenotype_sha256s: tuple[str, ...],
    ) -> str:
        return _sha256(
            _BINDING_KEY_DOMAIN,
            {
                **base_key.to_record(),
                "known_phenotype_sha256s": list(known_phenotype_sha256s),
            },
        )

    @staticmethod
    def _binding_authenticator(
        *,
        binding: ParentVariationBinding,
        key_sha256: str,
        base_contract_identity_sha256: str,
        option_phenotypes_sha256: str,
        eligibility_receipt_sha256: str,
    ) -> str:
        return _sha256(
            _ISSUED_BINDING_DOMAIN,
            {
                "key_sha256": key_sha256,
                "base_contract_identity_sha256": (
                    base_contract_identity_sha256
                ),
                "option_phenotypes_sha256": option_phenotypes_sha256,
                "eligible_contract_identity_sha256": (
                    binding.contract.identity_sha256
                ),
                "eligibility_receipt_sha256": eligibility_receipt_sha256,
                "binding": binding.to_record(),
            },
        )

    def _base_binding(
        self,
        key: _BaseBindingKey,
        parent: FrozenJsonObject,
    ) -> _BaseBinding:
        cached = self._base_bindings.get(key)
        if cached is not None:
            if (
                cached.contract.identity_sha256
                != cached.contract_identity_sha256
                or self._option_phenotypes_sha256(cached.option_phenotypes)
                != cached.option_phenotypes_sha256
            ):
                raise ValueError("cached finite catalog authority was mutated")
            return cached
        contract = self._config.benchmark.bind_finite_variation(
            self._config.finite_catalog_id,
            thaw_json(parent),
        )
        option_phenotypes = _benchmark_option_phenotype_bindings(
            self._config.benchmark,
            contract,
        )
        created = _BaseBinding(
            contract=contract,
            option_phenotypes=option_phenotypes,
            contract_identity_sha256=contract.identity_sha256,
            option_phenotypes_sha256=self._option_phenotypes_sha256(
                option_phenotypes
            ),
        )
        self._base_bindings[key] = created
        return created

    def bind(
        self,
        benchmark: FrozenJsonObject,
        parent: FrozenJsonObject,
        known_phenotype_sha256s: tuple[str, ...],
    ) -> ParentVariationBinding:
        base_key = self._base_key(benchmark, parent)
        cache_key = (base_key, known_phenotype_sha256s)
        cached = self._bindings.get(cache_key)
        if cached is not None:
            self._validate_issued(cached, cached.binding)
            return cached.binding

        base = self._base_binding(base_key, parent)
        eligibility = eligible_finite_variation_view(
            contract=base.contract,
            option_phenotypes=base.option_phenotypes,
            known_phenotype_sha256s=known_phenotype_sha256s,
        )
        binding = ParentVariationBinding(
            benchmark_sha256=self._benchmark_sha256,
            parent_configuration_sha256=base_key.parent_configuration_sha256,
            known_phenotype_sha256s=known_phenotype_sha256s,
            contract=eligibility.contract,
            eligibility_receipt=eligibility.receipt,
        )
        key_sha256 = self._key_sha256(base_key, known_phenotype_sha256s)
        receipt_sha256 = eligibility.receipt.receipt_sha256
        issued = _IssuedBinding(
            binding=binding,
            key_sha256=key_sha256,
            base_contract_identity_sha256=base.contract_identity_sha256,
            option_phenotypes_sha256=base.option_phenotypes_sha256,
            eligible_contract_identity_sha256=eligibility.contract.identity_sha256,
            eligibility_receipt_sha256=receipt_sha256,
            binding_authenticator_sha256=self._binding_authenticator(
                binding=binding,
                key_sha256=key_sha256,
                base_contract_identity_sha256=base.contract_identity_sha256,
                option_phenotypes_sha256=base.option_phenotypes_sha256,
                eligibility_receipt_sha256=receipt_sha256,
            ),
        )
        self._bindings[cache_key] = issued
        return binding

    def require_issued(
        self,
        benchmark: FrozenJsonObject,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
    ) -> None:
        base_key = self._base_key(benchmark, parent)
        cache_key = (base_key, variation.known_phenotype_sha256s)
        issued = self._bindings.get(cache_key)
        if issued is None or issued.binding is not variation:
            raise ValueError(
                "variation is not the selected catalog's exact eligible view "
                "issued by this campaign port set"
            )
        self._validate_issued(issued, variation)

    def _validate_issued(
        self,
        issued: _IssuedBinding,
        variation: ParentVariationBinding,
    ) -> None:
        ParentVariationBinding.__post_init__(variation)
        receipt = variation.eligibility_receipt
        if receipt is None:
            raise ValueError("issued variation omitted its eligibility receipt")
        if (
            variation.contract.identity_sha256
            != issued.eligible_contract_identity_sha256
            or receipt.base_contract_identity_sha256
            != issued.base_contract_identity_sha256
            or self._option_phenotypes_sha256(receipt.option_phenotypes)
            != issued.option_phenotypes_sha256
            or receipt.receipt_sha256 != issued.eligibility_receipt_sha256
            or self._binding_authenticator(
                binding=variation,
                key_sha256=issued.key_sha256,
                base_contract_identity_sha256=(
                    issued.base_contract_identity_sha256
                ),
                option_phenotypes_sha256=issued.option_phenotypes_sha256,
                eligibility_receipt_sha256=issued.eligibility_receipt_sha256,
            )
            != issued.binding_authenticator_sha256
        ):
            raise ValueError("issued variation authority failed authentication")


@dataclass(frozen=True, slots=True)
class _AgenticBenchmarkSessionPort:
    config: AgenticCampaignWorkloadConfig
    registry: _AuthenticatedBindingRegistry
    port_id = "agentic_benchmark_session"
    port_version = 1

    @property
    def definition_sha256(self) -> str:
        return _port_definition(self.config, "benchmark")

    def open(self, request: BenchmarkSessionRequest) -> CampaignBenchmarkSession:
        if type(request) is not BenchmarkSessionRequest:
            raise TypeError("request must be an exact BenchmarkSessionRequest")
        BenchmarkSessionRequest.__post_init__(request)
        self.config.benchmark.validate_binding()
        return self.registry.issue_session(
            CampaignBenchmarkSession(
                request_sha256=request.request_sha256,
                benchmark=self.config.benchmark_record,
                evaluator_concurrency_cap=self.config.evaluator_concurrency_cap,
                preflight_receipt=self.config.evaluator_preflight_receipt,
                resource_lease=self.config.resource_lease_receipt,
            )
        )


@dataclass(frozen=True, slots=True)
class _AgenticSeedPort:
    config: AgenticCampaignWorkloadConfig
    registry: _AuthenticatedBindingRegistry
    port_id = "agentic_benchmark_seeds"
    port_version = 1

    @property
    def definition_sha256(self) -> str:
        return _port_definition(self.config, "seeds")

    def load(self, session: CampaignBenchmarkSession) -> CampaignSeedBatch:
        _validate_session(self.config, session)
        self.registry.require_session(session)
        return CampaignSeedBatch(
            session_sha256=session.session_sha256,
            seeds=self.config.seeds,
        )


@dataclass(frozen=True, slots=True)
class _AgenticCatalogPort:
    config: AgenticCampaignWorkloadConfig
    registry: _AuthenticatedBindingRegistry
    port_id = "agentic_benchmark_catalog"
    port_version = 1

    @property
    def definition_sha256(self) -> str:
        return _port_definition(self.config, "catalog")

    def bind(
        self,
        benchmark: FrozenJsonObject,
        parent: FrozenJsonObject,
        known_phenotype_sha256s: tuple[str, ...],
    ) -> ParentVariationBinding:
        if benchmark != self.config.benchmark_record:
            raise ValueError("catalog request is bound to a foreign benchmark")
        _require_object(parent, name="parent")
        if type(known_phenotype_sha256s) is not tuple:
            raise TypeError("known_phenotype_sha256s must be an exact tuple")
        for value in known_phenotype_sha256s:
            _require_sha256(value, name="known_phenotype_sha256")
        if known_phenotype_sha256s != tuple(sorted(set(known_phenotype_sha256s))):
            raise ValueError("known phenotype hashes must be unique and canonical")
        return self.registry.bind(benchmark, parent, known_phenotype_sha256s)


@dataclass(frozen=True, slots=True)
class _AgenticEvidencePort:
    config: AgenticCampaignWorkloadConfig
    registry: _AuthenticatedBindingRegistry
    port_id = "agentic_benchmark_evidence"
    port_version = 1

    @property
    def definition_sha256(self) -> str:
        return _port_definition(self.config, "evidence")

    def initialize_memory(
        self,
        session: CampaignBenchmarkSession,
        seeds: CampaignSeedBatch,
    ) -> FrozenJsonObject:
        _validate_session(self.config, session)
        self.registry.require_session(session)
        if type(seeds) is not CampaignSeedBatch:
            raise TypeError("seeds must be an exact CampaignSeedBatch")
        CampaignSeedBatch.__post_init__(seeds)
        if seeds.session_sha256 != session.session_sha256:
            raise ValueError("seed batch is bound to a foreign session")
        if seeds.seeds != self.config.seeds:
            raise ValueError("evidence seed batch differs from workload seeds")
        result = self.config.evidence.initialize_memory(
            self.config.benchmark,
            session,
            seeds,
        )
        return _require_object(result, name="initialized memory")

    def context(
        self,
        session: CampaignBenchmarkSession,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
        memory: FrozenJsonObject,
    ) -> FrozenJsonObject:
        self._validate_projection_request(session, parent, variation, memory)
        result = self.config.evidence.context(
            self.config.benchmark,
            session,
            parent,
            variation,
            memory,
        )
        context = _require_object(result, name="evidence context")
        extension = self.config.prompt_extension
        if extension is None:
            return context
        mutable_context = thaw_json(context)
        if type(mutable_context) is not dict:  # pragma: no cover - exact guard above
            raise AssertionError("evidence context did not thaw as an object")
        if WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY in mutable_context:
            raise ValueError(
                "evidence projection used the reserved workload prompt "
                "extension context key"
            )
        mutable_context[WORKLOAD_PROMPT_EXTENSION_CONTEXT_KEY] = (
            extension.to_prompt_record()
        )
        attached = freeze_json(mutable_context)
        return _require_object(attached, name="extended evidence context")

    def cards(
        self,
        session: CampaignBenchmarkSession,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
        memory: FrozenJsonObject,
    ) -> tuple[FrozenJsonObject, ...]:
        self._validate_projection_request(session, parent, variation, memory)
        result = self.config.evidence.cards(
            self.config.benchmark,
            session,
            parent,
            variation,
            memory,
        )
        if type(result) is not tuple or any(
            type(card) is not FrozenJsonObject for card in result
        ):
            raise TypeError("evidence cards must be an exact tuple of frozen objects")
        for card in result:
            _require_object(card, name="evidence card")
        return result

    def _validate_projection_request(
        self,
        session: CampaignBenchmarkSession,
        parent: FrozenJsonObject,
        variation: ParentVariationBinding,
        memory: FrozenJsonObject,
    ) -> None:
        _validate_session(self.config, session)
        self.registry.require_session(session)
        _require_object(parent, name="parent")
        _require_object(memory, name="memory")
        if type(variation) is not ParentVariationBinding:
            raise TypeError("variation must be an exact ParentVariationBinding")
        ParentVariationBinding.__post_init__(variation)
        if variation.benchmark_sha256 != typed_json_sha256(session.benchmark):
            raise ValueError("variation is bound to a foreign benchmark")
        if variation.parent_configuration_sha256 != typed_json_sha256(parent):
            raise ValueError("variation is bound to a foreign parent")
        self.registry.require_issued(session.benchmark, parent, variation)


def _validate_session(
    config: AgenticCampaignWorkloadConfig,
    session: CampaignBenchmarkSession,
) -> None:
    if type(session) is not CampaignBenchmarkSession:
        raise TypeError("session must be an exact CampaignBenchmarkSession")
    CampaignBenchmarkSession.__post_init__(session)
    if session.benchmark != config.benchmark_record:
        raise ValueError("session is bound to a foreign benchmark")
    if session.evaluator_concurrency_cap != config.evaluator_concurrency_cap:
        raise ValueError("session evaluator concurrency changed")
    if session.preflight_receipt != config.evaluator_preflight_receipt:
        raise ValueError("session preflight receipt changed")
    if session.resource_lease != config.resource_lease_receipt:
        raise ValueError("session resource lease changed")


__all__ = [
    "AgenticCampaignEvidenceProjections",
    "AgenticCampaignWorkloadConfig",
    "CardProjection",
    "ContextProjection",
    "MemoryProjection",
]
