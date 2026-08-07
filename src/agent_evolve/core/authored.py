"""Model-authored machinery, as data.

An authored artifact is SOURCE CODE written -- by a model, or by a shipped
rule standing in for one -- against a narrow typed contract: a surrogate's
``fit_predict``, a variation operator's ``vary``. Here it is only data;
nothing in ``core/`` can execute it. Execution lives behind
``infrastructure/authored_runtime.py``: out of process, under resource
limits, imports allowlisted, every outcome typed and counted.

The candidate-authoring line survives this widening, restated rather than
relaxed: an artifact never emits a candidate into the run directly. A
surrogate only orders candidates the loop already constructed; an operator's
output is validated value-by-value against the declared domains and the
parents' own material, and it earns real evaluations only through the
governed, credit-assigned portfolio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

__all__ = ["AuthoredArtifact", "AuthoredContract", "CONTRACTS", "authored_artifact"]


@dataclass(frozen=True, slots=True)
class AuthoredContract:
    """The narrow surface an authored artifact must implement."""

    kind: str
    entry_point: str
    positional_arity: int
    #: The exact signature line shown to whoever authors the artifact.
    description: str


CONTRACTS: Mapping[str, AuthoredContract] = {
    "surrogate": AuthoredContract(
        kind="surrogate",
        entry_point="fit_predict",
        positional_arity=3,
        description=(
            "fit_predict(train_x: list[dict], train_y: list[dict], "
            "test_x: list[dict]) -> list[dict]  # stateless fit + predict; "
            "one dict of objective_name -> float per test_x row"
        ),
    ),
    "operator": AuthoredContract(
        kind="operator",
        entry_point="vary",
        positional_arity=5,
        description=(
            "vary(parent_a: dict, parent_b: dict, loci: list[str], "
            "domains: dict[str, list], seed: int) -> dict  # pure and "
            "deterministic given seed; every value from domains or a parent"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class AuthoredArtifact:
    """One authored source, hashed so a run can name exactly what it executed."""

    kind: str
    name: str
    entry_point: str
    source: str
    source_sha256: str
    authored_by: str          # "llm" | "rule"


def authored_artifact(
    kind: str, source: str, *, name: str, authored_by: str
) -> AuthoredArtifact:
    """Package *source* under the contract for *kind*, hashing it for identity."""

    contract = CONTRACTS.get(kind)
    if contract is None:
        raise ValueError(
            f"kind must be one of {sorted(CONTRACTS)}, got {kind!r}"
        )
    return AuthoredArtifact(
        kind=kind,
        name=name,
        entry_point=contract.entry_point,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        authored_by=authored_by,
    )
