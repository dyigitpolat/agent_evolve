"""Immutable experiment profiles for workload- and model-transfer campaigns.

The workload boundary owns candidate and evaluator semantics.  A campaign
profile owns every optimizer decision that must remain fixed when the same
method is transferred between workloads: depth/width, proposal support,
parent selection, acquisition, memory, recombination, reflection, and prompt
arm.  Model routing and concurrency are bound in the same inspectable object,
but have a separate digest so systems changes cannot be mistaken for method
changes.

The executable objects inside :class:`CampaignPolicyBinding` are deliberately
excluded from identity by that type.  Their authenticated public policy
identities are included.  This permits workload-specific composition objects
to implement the same inverted port without silently creating a new method.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from agent_evolve.application.evolution_campaign import (
    ArchiveUtilityPort,
    CampaignAgentRuntimePort,
    CampaignJournalPort,
    CampaignPolicyBinding,
    CampaignReflectionSupervisionPolicy,
    EvolutionCampaign,
    PreparedEvolutionCampaign,
)
from agent_evolve.campaign_presets import (
    DelayedPortfolioCampaignPreset,
    PortfolioCampaignBehavior,
    PortfolioScaleShape,
)
from agent_evolve.campaign_workload import AgenticCampaignWorkloadConfig
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    OpenRouterModelExecutionProfile,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjector,
)
from agent_evolve.workload_kit import WorkloadKit
from agent_evolve.workload_prompt import WorkloadPromptArm


_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_METHOD_DOMAIN = b"agent-evolve:campaign-method-profile:v2\x00"
_METHOD_DOMAIN_V3 = b"agent-evolve:campaign-method-profile:v3\x00"
_EXECUTION_DOMAIN = b"agent-evolve:campaign-experiment-profile:v2\x00"
_EXECUTION_DOMAIN_V3 = b"agent-evolve:campaign-experiment-profile:v3\x00"
_CONFORMANCE_DOMAIN = b"agent-evolve:campaign-profile-conformance:v1\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class CampaignExperimentProfile:
    """One complete, replay-identifiable campaign composition contract.

    ``method_definition_sha256`` intentionally excludes model routing and
    concurrency.  It must remain identical across a workload/model transfer
    block.  ``experiment_definition_sha256`` additionally binds those systems
    settings and therefore identifies an executable cell.
    """

    profile_id: str
    profile_version: int
    method_id: str
    method_version: int
    scale_shape: PortfolioScaleShape
    candidate_pool_size: int | None
    model_selection_size: int
    prompt_arm: WorkloadPromptArm
    parent_selection: CampaignPolicyBinding
    memory_assignment: CampaignPolicyBinding
    portfolio_selection: CampaignPolicyBinding
    recombination: CampaignPolicyBinding
    reflection: CampaignPolicyBinding
    model_execution: OpenRouterModelExecutionProfile
    archive_context: CampaignPolicyBinding | None = None
    variation_topology: CampaignPolicyBinding | None = None
    contextual_outcomes: CampaignPolicyBinding | None = None
    evaluator_concurrency: int = 1
    agent_concurrency: int = 3
    agent_queue_capacity: int = 8
    reflection_supervision: CampaignReflectionSupervisionPolicy = (
        CampaignReflectionSupervisionPolicy()
    )

    def __post_init__(self) -> None:
        for name in ("profile_id", "method_id"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the closed lowercase token grammar")
        for name in ("profile_version", "method_version"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.scale_shape) is not PortfolioScaleShape:
            raise TypeError("scale_shape must be an exact PortfolioScaleShape")
        self.scale_shape.__post_init__()
        if self.candidate_pool_size is not None and (
            type(self.candidate_pool_size) is not int
            or self.candidate_pool_size <= 0
        ):
            raise ValueError(
                "candidate_pool_size must be a positive exact integer or None"
            )
        if type(self.model_selection_size) is not int or (
            self.model_selection_size <= 0
        ):
            raise ValueError("model_selection_size must be a positive exact integer")
        if not (
            self.model_selection_size >= self.scale_shape.portfolio_width
            and (
                self.candidate_pool_size is None
                or self.candidate_pool_size >= self.model_selection_size
            )
        ):
            raise ValueError(
                "candidate/model/evaluated widths must satisfy M >= K >= k"
            )
        if type(self.prompt_arm) is not WorkloadPromptArm:
            raise TypeError("prompt_arm must be an exact WorkloadPromptArm")
        for name in (
            "parent_selection",
            "memory_assignment",
            "portfolio_selection",
            "recombination",
            "reflection",
        ):
            binding = getattr(self, name)
            if type(binding) is not CampaignPolicyBinding:
                raise TypeError(f"{name} must be an exact CampaignPolicyBinding")
            CampaignPolicyBinding.__post_init__(binding)
        if self.archive_context is not None:
            if type(self.archive_context) is not CampaignPolicyBinding:
                raise TypeError(
                    "archive_context must be an exact CampaignPolicyBinding or None"
                )
            CampaignPolicyBinding.__post_init__(self.archive_context)
            if not isinstance(
                self.archive_context.implementation,
                CampaignPortfolioArchiveContextProjector,
            ):
                raise TypeError(
                    "archive_context implementation must satisfy the generic "
                    "archive-context projector port"
                )
        for name in ("variation_topology", "contextual_outcomes"):
            binding = getattr(self, name)
            if binding is not None:
                if type(binding) is not CampaignPolicyBinding:
                    raise TypeError(
                        f"{name} must be an exact CampaignPolicyBinding or None"
                    )
                CampaignPolicyBinding.__post_init__(binding)
        if type(self.model_execution) is not OpenRouterModelExecutionProfile:
            raise TypeError(
                "model_execution must be an exact OpenRouterModelExecutionProfile"
            )
        self.model_execution.__post_init__()
        for name in ("evaluator_concurrency", "agent_concurrency"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.agent_queue_capacity) is not int
            or self.agent_queue_capacity < self.agent_concurrency
        ):
            raise ValueError("agent_queue_capacity must cover agent_concurrency")
        if (
            self.model_execution.route_concurrency_cap is not None
            and self.agent_concurrency
            > self.model_execution.route_concurrency_cap
        ):
            raise ValueError(
                "agent_concurrency exceeds the authenticated model-route cap"
            )
        if type(self.reflection_supervision) is not (
            CampaignReflectionSupervisionPolicy
        ):
            raise TypeError(
                "reflection_supervision must be exact "
                "CampaignReflectionSupervisionPolicy"
            )
        CampaignReflectionSupervisionPolicy.__post_init__(
            self.reflection_supervision
        )

    def _method_record(self) -> dict[str, object]:
        self.__post_init__()
        extended_identity = (
            self.variation_topology is not None
            or self.contextual_outcomes is not None
        )
        record = {
            "schema_version": 3 if extended_identity else 2,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "scale_shape": self.scale_shape.to_record(),
            "candidate_pool_size": self.candidate_pool_size,
            "candidate_pool_mode": (
                "complete_finite_contract"
                if self.candidate_pool_size is None
                else "fixed_size"
            ),
            "model_selection_size": self.model_selection_size,
            "evaluated_portfolio_width": self.scale_shape.portfolio_width,
            "prompt_arm": self.prompt_arm.value,
            "policies": {
                "parent_selection": self.parent_selection.to_record(),
                "memory_assignment": self.memory_assignment.to_record(),
                "portfolio_selection": self.portfolio_selection.to_record(),
                "recombination": self.recombination.to_record(),
                "reflection": self.reflection.to_record(),
                "archive_context": (
                    None
                    if self.archive_context is None
                    else self.archive_context.to_record()
                ),
                "reflection_supervision": self.reflection_supervision.to_record(),
            },
            "workload_specific_fields": [],
            "model_specific_optimizer_fields": [],
        }
        if extended_identity:
            policies = record["policies"]
            assert type(policies) is dict
            policies["variation_topology"] = (
                None
                if self.variation_topology is None
                else self.variation_topology.to_record()
            )
            policies["contextual_outcomes"] = (
                None
                if self.contextual_outcomes is None
                else self.contextual_outcomes.to_record()
            )
        return record

    @property
    def method_definition_sha256(self) -> str:
        """Identity compared across every workload and model quality cell."""

        record = self._method_record()
        return _digest(
            _METHOD_DOMAIN_V3 if record["schema_version"] == 3 else _METHOD_DOMAIN,
            record,
        )

    def _execution_record(self) -> dict[str, object]:
        self.__post_init__()
        extended_identity = (
            self.variation_topology is not None
            or self.contextual_outcomes is not None
        )
        return {
            "schema_version": 3 if extended_identity else 2,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "method_definition_sha256": self.method_definition_sha256,
            "model_execution_profile_sha256": self.model_execution.profile_sha256,
            "systems": {
                "evaluator_concurrency": self.evaluator_concurrency,
                "agent_concurrency": self.agent_concurrency,
                "agent_queue_capacity": self.agent_queue_capacity,
            },
        }

    @property
    def experiment_definition_sha256(self) -> str:
        """Identity of the method plus model route and systems settings."""

        record = self._execution_record()
        return _digest(
            (
                _EXECUTION_DOMAIN_V3
                if record["schema_version"] == 3
                else _EXECUTION_DOMAIN
            ),
            record,
        )

    def to_record(self) -> dict[str, object]:
        """Publish both identities so quality and systems changes stay separate."""

        return {
            **self._execution_record(),
            "method": self._method_record(),
            "model_execution": self.model_execution.to_record(),
            "experiment_definition_sha256": self.experiment_definition_sha256,
        }

    def behavior(self, *, archive_utility: ArchiveUtilityPort) -> PortfolioCampaignBehavior:
        """Bind workload-owned objective utility to the frozen generic method."""

        self.__post_init__()
        if not isinstance(archive_utility, ArchiveUtilityPort):
            raise TypeError("archive_utility must implement ArchiveUtilityPort")
        return PortfolioCampaignBehavior(
            parent_selection=self.parent_selection,
            memory_assignment=self.memory_assignment,
            portfolio_selection=self.portfolio_selection,
            recombination=self.recombination,
            reflection=self.reflection,
            archive_utility=archive_utility,
            reflection_supervision=self.reflection_supervision,
        )

    @property
    def archive_context_projector(
        self,
    ) -> CampaignPortfolioArchiveContextProjector | None:
        """Return the executable projector authenticated by this method profile."""

        self.__post_init__()
        if self.archive_context is None:
            return None
        implementation = self.archive_context.implementation
        if not isinstance(implementation, CampaignPortfolioArchiveContextProjector):
            raise AssertionError("validated archive-context implementation drifted")
        return implementation

    def preset(self, *, outer_seed: int) -> DelayedPortfolioCampaignPreset:
        """Construct the exact scale schedule without workload constants."""

        self.__post_init__()
        return DelayedPortfolioCampaignPreset.scale_shape(
            self.scale_shape,
            outer_seed=outer_seed,
            evaluator_concurrency=self.evaluator_concurrency,
            agent_concurrency=self.agent_concurrency,
            agent_queue_capacity=self.agent_queue_capacity,
        )

    def compose(
        self,
        *,
        outer_seed: int,
        workload: WorkloadKit | AgenticCampaignWorkloadConfig,
        archive_utility: ArchiveUtilityPort,
        runtime: CampaignAgentRuntimePort,
        journals: tuple[CampaignJournalPort, ...],
    ) -> EvolutionCampaign:
        """Compose one workload through the frozen experiment profile."""

        return self.preset(outer_seed=outer_seed).compose(
            workload=workload,
            behavior=self.behavior(archive_utility=archive_utility),
            runtime=runtime,
            journals=journals,
        )

    def prepared_conformance_record(
        self,
        *,
        prepared: PreparedEvolutionCampaign,
        archive_utility: ArchiveUtilityPort,
        outer_seed: int,
    ) -> dict[str, object]:
        """Authenticate that a workload runner executed this profile.

        Runners may retain a workload-facing protocol label, but they may not
        silently change the optimizer shape, budget, policy identities, or
        concurrency.  The runtime implementations behind policy bindings are
        intentionally swappable and do not affect these scientific identities.
        """

        self.__post_init__()
        if type(prepared) is not PreparedEvolutionCampaign:
            raise TypeError("prepared must be an exact PreparedEvolutionCampaign")
        prepared.__post_init__()
        if type(outer_seed) is not int or outer_seed < 0:
            raise ValueError("outer_seed must be a non-negative exact integer")
        preset = self.preset(outer_seed=outer_seed)
        expected_protocol = preset.protocol(
            required_seed_count=prepared.protocol.required_seed_count
        )
        expected_budget = preset.budget(
            required_seed_count=prepared.protocol.required_seed_count
        )
        expected_concurrency = preset.concurrency()
        expected_policies = self.behavior(
            archive_utility=archive_utility
        ).bind()
        shape_fields = (
            "outer_seed",
            "generation_count",
            "required_seed_count",
            "parents_per_portfolio_generation",
            "portfolio_width",
            "recombinations_per_parent",
            "reflections_per_recombination_generation",
            "reflection_promotion_block_pairs",
            "terminal_reflection_policy",
        )
        gates = {
            "optimizer_shape_exact": all(
                getattr(prepared.protocol, name)
                == getattr(expected_protocol, name)
                for name in shape_fields
            ),
            "derived_budget_exact": prepared.budget == expected_budget,
            "concurrency_exact": prepared.concurrency == expected_concurrency,
            "policy_identity_exact": (
                prepared.policies_sha256 == expected_policies.policies_sha256
            ),
        }
        unsigned = {
            "schema_version": 1,
            "method_definition_sha256": self.method_definition_sha256,
            "experiment_definition_sha256": self.experiment_definition_sha256,
            "preparation_sha256": prepared.preparation_sha256,
            "gates": gates,
            "pass": all(gates.values()),
        }
        record = {
            **unsigned,
            "conformance_sha256": _digest(_CONFORMANCE_DOMAIN, unsigned),
        }
        if not record["pass"]:
            failed = ", ".join(name for name, value in gates.items() if not value)
            raise RuntimeError(f"prepared campaign violates profile: {failed}")
        return record

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is CampaignExperimentProfile
            and self.experiment_definition_sha256
            == other.experiment_definition_sha256
        )

    __hash__ = None


__all__ = ["CampaignExperimentProfile"]
