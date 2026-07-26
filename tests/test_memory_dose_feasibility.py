from __future__ import annotations

import hashlib

from agent_evolve.policies.selection.memory_dose_feasibility import (
    find_memory_dose_attribution_feasibility_witness,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseCardSupport,
    PortfolioMemoryDoseStage,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _support(card_key: str, option_id: str) -> PortfolioMemoryDoseCardSupport:
    return PortfolioMemoryDoseCardSupport(
        card_key=card_key,
        card_content_sha256=_sha(f"content {card_key}"),
        finite_contract_identity_sha256=_sha("finite contract"),
        compatible_options=((option_id, _sha(f"identity {option_id}")),),
        support_policy_id="test_exact_support",
        support_policy_version=1,
        support_policy_definition_sha256=_sha("test exact support"),
    )


def test_finds_canonical_exact_evaluated_card_assignment() -> None:
    contract = BoundedPortfolioMemoryDoseContract(
        card_supports=(
            _support("card.a", "option.a"),
            _support("card.b", "option.c"),
        ),
        proposed_supported_member_bounds=(2, 2),
        evaluated_supported_member_bounds=(2, 2),
        minimum_unattributed_proposed_members=2,
        minimum_unattributed_evaluated_members=2,
    )
    members = tuple(
        (option_id, _sha(f"identity {option_id}"))
        for option_id in ("option.d", "option.b", "option.c", "option.a")
    )

    witness = find_memory_dose_attribution_feasibility_witness(
        contract,
        stage=PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO,
        member_option_identities=members,
    )

    assert witness is not None
    assert tuple(value[0] for value in witness.member_option_identities) == (
        "option.a",
        "option.b",
        "option.c",
        "option.d",
    )
    assert witness.attributions == (
        ("option.a", ("card.a",)),
        ("option.b", ()),
        ("option.c", ("card.b",)),
        ("option.d", ()),
    )
    assert witness.supported_member_count == 2
    assert witness.unattributed_member_count == 2
    assert witness.covered_card_keys == ("card.a", "card.b")
    assert witness.to_record()["objective_values_consulted"] is False


def test_returns_none_when_fixed_subset_cannot_cover_assigned_card() -> None:
    contract = BoundedPortfolioMemoryDoseContract(
        card_supports=(_support("card.a", "option.missing"),),
        proposed_supported_member_bounds=(1, 1),
        evaluated_supported_member_bounds=(1, 1),
        minimum_unattributed_proposed_members=3,
        minimum_unattributed_evaluated_members=3,
    )

    assert (
        find_memory_dose_attribution_feasibility_witness(
            contract,
            stage=PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO,
            member_option_identities=tuple(
                (option_id, _sha(f"identity {option_id}"))
                for option_id in (
                    "option.a",
                    "option.b",
                    "option.c",
                    "option.d",
                )
            ),
        )
        is None
    )
