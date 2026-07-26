"""Transactional composition for independent campaign outcome projections."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_evolve.application.campaign_execution import CampaignStageRequest
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignPortfolioOutcomePreparation,
    CampaignPortfolioOutcomeUpdater,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("composite outcome evidence did not freeze to an object")
    return result


@dataclass(slots=True)
class CompositeCampaignPortfolioOutcomeUpdater:
    """Stage several zero-call updaters behind one atomic runtime port.

    Each component receives the memory projection produced by its predecessor.
    No component is published until every preparation succeeds.  Abort proceeds
    in reverse order; commit proceeds in declaration order.
    """

    updaters: tuple[CampaignPortfolioOutcomeUpdater, ...]
    _prepared: dict[
        str,
        tuple[
            CampaignPortfolioOutcomePreparation,
            tuple[
                tuple[
                    CampaignPortfolioOutcomeUpdater,
                    CampaignPortfolioOutcomePreparation,
                ],
                ...,
            ],
        ],
    ] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.updaters) is not tuple or len(self.updaters) < 2:
            raise ValueError("composite outcome updater requires at least two components")
        if any(
            not isinstance(value, CampaignPortfolioOutcomeUpdater)
            for value in self.updaters
        ):
            raise TypeError("every component must implement the outcome updater port")

    async def prepare_update(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        prior_memory: FrozenJsonObject,
    ) -> CampaignPortfolioOutcomePreparation:
        self.__post_init__()
        staged: list[
            tuple[CampaignPortfolioOutcomeUpdater, CampaignPortfolioOutcomePreparation]
        ] = []
        memory = prior_memory
        try:
            for updater in self.updaters:
                preparation = await updater.prepare_update(
                    request,
                    waves,
                    results,
                    memory,
                )
                if type(preparation) is not CampaignPortfolioOutcomePreparation:
                    raise TypeError(
                        "composite component returned a foreign preparation"
                    )
                preparation.__post_init__()
                staged.append((updater, preparation))
                memory = preparation.updated_memory
        except BaseException:
            for updater, preparation in reversed(staged):
                updater.abort_update(preparation)
            raise
        outer = CampaignPortfolioOutcomePreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                value.selection_request.request_sha256 for value in waves
            ),
            result_receipt_sha256s=tuple(
                value.receipt.receipt_sha256 for value in results
            ),
            prior_memory_sha256=typed_json_sha256(prior_memory),
            updated_memory=memory,
            evidence=_object(
                {
                    "schema_version": 1,
                    "composition": "sequential_prepare_atomic_publish",
                    "component_preparations": [
                        value.to_record() for _updater, value in staged
                    ],
                    "component_count": len(staged),
                    "provider_calls": 0,
                }
            ),
        )
        if outer.preparation_sha256 in self._prepared:
            for updater, preparation in reversed(staged):
                updater.abort_update(preparation)
            raise ValueError("composite outcome preparation is already pending")
        self._prepared[outer.preparation_sha256] = (outer, tuple(staged))
        return outer

    def commit_update(self, preparation: CampaignPortfolioOutcomePreparation) -> None:
        if type(preparation) is not CampaignPortfolioOutcomePreparation:
            raise TypeError("preparation must be exact")
        preparation.__post_init__()
        pending = self._prepared.pop(preparation.preparation_sha256, None)
        if pending is None or pending[0] != preparation:
            raise ValueError("composite outcome preparation is foreign or not pending")
        for updater, component in pending[1]:
            updater.commit_update(component)

    def abort_update(self, preparation: CampaignPortfolioOutcomePreparation) -> None:
        if type(preparation) is not CampaignPortfolioOutcomePreparation:
            raise TypeError("preparation must be exact")
        preparation.__post_init__()
        pending = self._prepared.pop(preparation.preparation_sha256, None)
        if pending is None or pending[0] != preparation:
            raise ValueError("composite outcome preparation is foreign or not pending")
        for updater, component in reversed(pending[1]):
            updater.abort_update(component)


__all__ = ["CompositeCampaignPortfolioOutcomeUpdater"]
