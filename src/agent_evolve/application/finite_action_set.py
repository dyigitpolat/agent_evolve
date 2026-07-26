"""Trusted sealer for benchmark-compiled matched finite action sets."""

from __future__ import annotations

import hashlib
import json

from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
)
from agent_evolve.domain.finite_action_set import (
    FiniteActionCardAuthority,
    FiniteActionOptionAuthority,
    FiniteActionPresentationAuthority,
    FiniteActionSetAuthority,
    FiniteActionSourceMode,
    FiniteActionSupportAuthority,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.finite_action_set import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetCompiler,
    FiniteActionSetDraft,
    validate_finite_action_set_compiler_identity,
    validate_finite_action_set_draft,
)


_PROMPT_CARD_DOMAIN = b"agent-evolve:finite-action-prompt-card:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


def _validate_compiled_anchor_request(
    *,
    compiled_anchor: CompiledHypothesisTreatment,
    request: FiniteActionSetCompilationRequest,
) -> tuple[str, str]:
    if type(compiled_anchor) is not CompiledHypothesisTreatment:
        raise TypeError("compiled_anchor must be an exact CompiledHypothesisTreatment")
    CompiledHypothesisTreatment.__post_init__(compiled_anchor)
    if type(request) is not FiniteActionSetCompilationRequest:
        raise TypeError("request must be an exact FiniteActionSetCompilationRequest")
    FiniteActionSetCompilationRequest.__post_init__(request)
    spec = compiled_anchor.receipt.spec
    if spec is None:  # pragma: no cover - compiled treatment closes this branch.
        raise ValueError("compiled anchor lost its executable spec")
    if len(compiled_anchor.requirement.allowed_actions) != 1:
        raise ValueError("Stage-B requires one unchanged exact semantic anchor")
    exact_action = compiled_anchor.requirement.allowed_actions[0]
    expected = (
        compiled_anchor.request.parent_candidate_id,
        compiled_anchor.request.finite_contract.identity_sha256,
        exact_action.option_id,
        exact_action.option_identity_sha256,
        compiled_anchor.requirement.requirement_sha256,
        compiled_anchor.request.reference,
        compiled_anchor.request.insight.content_sha256,
        compiled_anchor.request.context_projection_sha256,
        compiled_anchor.request.endpoint_definition_sha256,
    )
    observed = (
        request.parent_candidate_id,
        request.finite_contract.identity_sha256,
        request.anchor_option_id,
        request.anchor_option_identity_sha256,
        request.exact_anchor_requirement_sha256,
        request.card_reference,
        request.card_content_sha256,
        request.context_projection_sha256,
        request.endpoint_definition_sha256,
    )
    if observed != expected:
        raise ValueError("finite action request differs from its exact compiled anchor")
    return spec.spec_sha256, compiled_anchor.receipt.receipt_sha256


def _identify_phenotype(
    policy: PhenotypeIdentityPolicy,
    configuration: object,
) -> PhenotypeIdentity:
    identity = policy.identify(configuration)
    if type(identity) is not PhenotypeIdentity:
        raise TypeError("phenotype policy must return an exact PhenotypeIdentity")
    PhenotypeIdentity.__post_init__(identity)
    expected_policy = (
        getattr(policy, "policy_id", None),
        getattr(policy, "policy_version", None),
    )
    if (identity.policy_id, identity.policy_version) != expected_policy:
        raise ValueError("phenotype policy returned a foreign identity")
    return identity


def compile_and_seal_finite_action_set(
    *,
    compiled_anchor: CompiledHypothesisTreatment,
    request: FiniteActionSetCompilationRequest,
    compiler: FiniteActionSetCompiler,
    phenotype_identity: PhenotypeIdentityPolicy,
    source_mode: FiniteActionSourceMode,
) -> tuple[FiniteActionSetAuthority, FiniteActionSetDraft]:
    """Compile K opaque IDs and seal their full children without outcome access."""

    if source_mode not in {
        FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
        FiniteActionSourceMode.COMPILED_SHUFFLED_CARD,
    }:
        raise ValueError("compiled finite action sealing requires a compiled card mode")
    spec_sha256, compilation_receipt_sha256 = _validate_compiled_anchor_request(
        compiled_anchor=compiled_anchor,
        request=request,
    )
    compiler_identity = validate_finite_action_set_compiler_identity(compiler)
    phenotype_policy_identity = (
        getattr(phenotype_identity, "policy_id", None),
        getattr(phenotype_identity, "policy_version", None),
    )
    # Constructing an exact value validates both policy fields without importing
    # a concrete policy class into this application service.
    PhenotypeIdentity(
        policy_id=phenotype_policy_identity[0],
        policy_version=phenotype_policy_identity[1],
        value_sha256="0" * 64,
    )
    request_sha256_before = request.request_sha256
    anchor_binding_before = compiled_anchor.binding_sha256
    source_contract_before = request.finite_contract.identity_sha256

    draft = compiler.compile(request)
    validate_finite_action_set_draft(request, draft)

    if validate_finite_action_set_compiler_identity(compiler) != compiler_identity:
        raise ValueError("finite action set compiler identity changed during compile")
    if (
        request.request_sha256 != request_sha256_before
        or request.finite_contract.identity_sha256 != source_contract_before
        or compiled_anchor.binding_sha256 != anchor_binding_before
    ):
        raise ValueError("finite action inputs changed during support compilation")

    selected_options = tuple(
        request.finite_contract.resolve(option_id)
        for option_id in draft.ordered_option_ids
    )
    support_contract = FiniteVariationContract(
        catalog_id=request.finite_contract.catalog_id,
        catalog_version=request.finite_contract.catalog_version,
        catalog_definition_sha256=request.finite_contract.catalog_definition_sha256,
        parent_configuration=request.finite_contract.parent_configuration,
        options=selected_options,
    )
    spec = compiled_anchor.receipt.spec
    assert spec is not None
    allowed_families = set(spec.recommended_option_families)
    affected_paths = spec.affected_paths
    held_fixed_paths = spec.held_fixed_paths
    probe = CandidateId("candidate_finite_action_set_probe")
    if probe == request.parent_candidate_id:
        probe = CandidateId("candidate_finite_action_set_probe_alternate")
    option_authorities: list[FiniteActionOptionAuthority] = []
    for option in support_contract.options:
        if option.family not in allowed_families:
            raise ValueError("finite action support escaped the anchor option family")
        patch = derive_patch(
            support_contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=request.parent_candidate_id,
            target_candidate_id=probe,
        )
        changed_paths = tuple(
            sorted({_path_text(operation.path) for operation in patch.operations})
        )
        if not changed_paths or any(
            not any(_paths_overlap(path, affected) for affected in affected_paths)
            for path in changed_paths
        ):
            raise ValueError("finite action support escaped the card's affected paths")
        if any(
            _paths_overlap(path, held)
            for path in changed_paths
            for held in held_fixed_paths
        ):
            raise ValueError("finite action support changed a held-fixed path")
        phenotype = _identify_phenotype(
            phenotype_identity,
            option.child_configuration,
        )
        option_authorities.append(
            FiniteActionOptionAuthority(
                option=option,
                changed_paths=changed_paths,
                phenotype_policy_id=phenotype.policy_id,
                phenotype_policy_version=phenotype.policy_version,
                phenotype_identity_sha256=phenotype.identity_sha256,
            )
        )
    if (
        getattr(phenotype_identity, "policy_id", None),
        getattr(phenotype_identity, "policy_version", None),
    ) != phenotype_policy_identity:
        raise ValueError("phenotype identity policy changed during support sealing")

    presentation = FiniteActionPresentationAuthority(
        policy_id=draft.presentation_policy_id,
        policy_version=draft.presentation_policy_version,
        definition_sha256=draft.presentation_definition_sha256,
        ordered_option_ids=draft.ordered_option_ids,
        ordered_prompt_record_sha256s=tuple(
            value.prompt_record_sha256 for value in option_authorities
        ),
        prompt_shape_sha256=draft.prompt_shape_sha256,
    )
    support = FiniteActionSupportAuthority(
        parent_candidate_id=request.parent_candidate_id,
        source_contract_sha256=request.finite_contract.identity_sha256,
        support_contract=support_contract,
        endpoint_definition_sha256=request.endpoint_definition_sha256,
        context_projection_sha256=request.context_projection_sha256,
        options=tuple(option_authorities),
        anchor_option_id=request.anchor_option_id,
        presentation=presentation,
        compatible_option_count=len(option_authorities),
    )
    prompt_card_record = {
        "schema_version": 1,
        "reference": {
            "insight_id": compiled_anchor.request.reference.insight_id.value,
            "version": compiled_anchor.request.reference.version,
        },
        "card_content_sha256": compiled_anchor.request.insight.content_sha256,
        "compiled_anchor_binding_sha256": compiled_anchor.binding_sha256,
        "exact_anchor_requirement_sha256": (
            compiled_anchor.requirement.requirement_sha256
        ),
    }
    card = FiniteActionCardAuthority(
        source_mode=source_mode,
        reference=compiled_anchor.request.reference,
        card_content_sha256=compiled_anchor.request.insight.content_sha256,
        registered_source_evidence_sha256=(
            compiled_anchor.request.source_evidence_sha256
        ),
        exact_anchor_requirement_sha256=(
            compiled_anchor.requirement.requirement_sha256
        ),
        compilation_request_sha256=compiled_anchor.request.request_sha256,
        compilation_receipt_sha256=compilation_receipt_sha256,
        executable_spec_sha256=spec_sha256,
        prompt_card_record_sha256=_hash(_PROMPT_CARD_DOMAIN, prompt_card_record),
    )
    authority = FiniteActionSetAuthority(
        support=support,
        card=card,
        support_compilation_request_sha256=request.request_sha256,
        support_compilation_draft_sha256=draft.draft_sha256,
        support_compiler_policy_id=compiler_identity[0],
        support_compiler_policy_version=compiler_identity[1],
        support_compiler_definition_sha256=compiler_identity[2],
        current_outcome_access=False,
    )
    return authority, draft


__all__ = ["compile_and_seal_finite_action_set"]
