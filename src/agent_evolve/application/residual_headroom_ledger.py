"""Conserved, workload-opaque learning of residual evolutionary headroom.

The ledger closes a gap between one-stage outcome-adaptive racing and a
multi-generation optimizer.  It converts authenticated conditional set gains
into conserved action credit, learns decayed posteriors over opaque action
cells, and exposes those posteriors through an optional adaptive-market
projector.  It never inspects workload names, objective names, configuration
fields, prompts, providers, or model-name strings.

The module deliberately separates three responsibilities:

* ``ConservedResidualHeadroomProjector`` closes one evaluated stage without
  manufacturing more credit than the stage actually earned;
* ``ConservedResidualHeadroomLedger`` stores immutable closures and estimates
  context-conditioned residual value; and
* ``ResidualHeadroomAdaptiveMarketProjector`` injects the prior-only estimate
  into any existing adaptive market without changing the workload adapter.

Predicted values are selection evidence only.  They never enter an
authoritative archive or replace a real evaluator outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

from agent_evolve.application.materialized_action_broker import (
    BrokerActionScore,
    MaterializedActionDescriptor,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionFactorCell,
    AdaptiveActionOutcome,
    AdaptiveActionRacingDecision,
    AdaptiveActionSetOutcome,
)
from agent_evolve.application.outcome_adaptive_residual_portfolio_evolution import (
    AdaptiveActionMarketProjectorPort,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256


RESIDUAL_HEADROOM_LEDGER_ID = "conserved_residual_headroom_ledger"
RESIDUAL_HEADROOM_LEDGER_VERSION = 1
RESIDUAL_HEADROOM_LEDGER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:conserved-residual-headroom-ledger:v1;"
    b"evidence=authenticated-real-action-and-conditional-set-outcomes;"
    b"credit=wave-conditional-gain-proportional-to-isolated-real-gain;"
    b"conservation=sum-action-credit-equals-sum-conditional-set-gain;"
    b"attribution=equal-share-over-opaque-portable-action-cells;"
    b"posterior=context-conditioned-decayed-clipped-ipw;"
    b"headroom=expected-gain-plus-uncertainty-plus-late-bloom;"
    b"risk=redundancy-plus-saturation-plus-invalidity;"
    b"archive-authority=real-evaluations-only;"
    b"workload-objective-model-provider-prompt-config-branches=false"
).hexdigest()

RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_ID = (
    "residual_headroom_adaptive_market_projector"
)
RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_VERSION = 1

_OBSERVATION_DOMAIN = b"agent-evolve:residual-headroom-observation:v1\x00"
_CLOSURE_DOMAIN = b"agent-evolve:residual-headroom-stage-closure:v1\x00"
_STATE_DOMAIN = b"agent-evolve:residual-headroom-ledger-state:v1\x00"
_CONFIG_DOMAIN = b"agent-evolve:residual-headroom-ledger-config:v1\x00"
_ESTIMATE_DOMAIN = b"agent-evolve:residual-headroom-estimate:v1\x00"
_MARKET_PROJECTOR_DOMAIN = (
    b"agent-evolve:residual-headroom-market-projector:v1\x00"
)
_MARKET_STATE_DOMAIN = (
    b"agent-evolve:residual-headroom-market-projector-state:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _require_nonnegative(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _attribution_cells(
    action: AdaptiveActionDescriptor,
) -> tuple[AdaptiveActionFactorCell, ...]:
    """Project one portable action into conserved-credit attribution cells."""

    action.__post_init__()
    cells = set(action.factor_cells)
    cells.add(
        AdaptiveActionFactorCell(
            family_id="portable_lane",
            level_id=action.lane_id,
        )
    )
    cells.add(
        AdaptiveActionFactorCell(
            family_id="portable_operator",
            level_id=action.operator_id,
        )
    )
    cells.add(
        AdaptiveActionFactorCell(
            family_id="portable_parent_origin",
            level_id=(
                "current_run"
                if action.parent_generated_in_current_run
                else "prior_archive"
            ),
        )
    )
    if not any(
        value.family_id == "materialized_rank_layer"
        for value in cells
    ):
        layer = min(
            2,
            ((action.native_rank - 1) * 3) // action.lane_size,
        )
        cells.add(
            AdaptiveActionFactorCell(
                family_id="materialized_rank_layer",
                level_id=f"layer{layer}",
            )
        )
    for value in action.semantic_cell_ids:
        cells.add(
            AdaptiveActionFactorCell(
                family_id="portable_semantic_cell",
                level_id=value,
            )
        )
    return tuple(sorted(cells))


@dataclass(frozen=True, slots=True)
class ResidualHeadroomObservation:
    """One action's conserved share of a real conditional wave gain."""

    context_sha256: str
    residual_request_sha256: str
    generation_index: int
    wave_index: int
    action_sha256: str
    evaluation_sha256: str
    outcome_sha256: str
    set_outcome_sha256: str
    decision_sha256: str
    selection_propensity: float
    propensity_identified: bool
    feasible: bool
    isolated_gain: float
    conditional_credit: float
    normalized_conditional_credit: float
    redundancy_fraction: float
    synergy_fraction: float
    attribution_cells: tuple[AdaptiveActionFactorCell, ...]
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "context_sha256",
            "residual_request_sha256",
            "action_sha256",
            "evaluation_sha256",
            "outcome_sha256",
            "set_outcome_sha256",
            "decision_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.generation_index) is not int or self.generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("wave_index must be non-negative")
        if (
            type(self.selection_propensity) is not float
            or not math.isfinite(self.selection_propensity)
            or not 0.0 < self.selection_propensity <= 1.0
        ):
            raise ValueError("selection_propensity must lie in (0, 1]")
        if type(self.propensity_identified) is not bool:
            raise TypeError("propensity_identified must be exact")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be exact")
        for name in (
            "isolated_gain",
            "conditional_credit",
            "normalized_conditional_credit",
            "redundancy_fraction",
            "synergy_fraction",
        ):
            _require_nonnegative(getattr(self, name), name=name)
        for name in ("redundancy_fraction", "synergy_fraction"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if not self.feasible and (
            self.isolated_gain != 0.0
            or self.conditional_credit != 0.0
        ):
            raise ValueError("infeasible actions cannot receive positive credit")
        if (
            type(self.attribution_cells) is not tuple
            or not self.attribution_cells
            or self.attribution_cells
            != tuple(sorted(set(self.attribution_cells)))
        ):
            raise ValueError(
                "attribution_cells must be non-empty, unique, and canonical"
            )
        for value in self.attribution_cells:
            if type(value) is not AdaptiveActionFactorCell:
                raise TypeError("attribution cells must be exact")
            value.__post_init__()
        object.__setattr__(
            self,
            "observation_sha256",
            _hash(_OBSERVATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "context_sha256": self.context_sha256,
            "residual_request_sha256": self.residual_request_sha256,
            "generation_index": self.generation_index,
            "wave_index": self.wave_index,
            "action_sha256": self.action_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "outcome_sha256": self.outcome_sha256,
            "set_outcome_sha256": self.set_outcome_sha256,
            "decision_sha256": self.decision_sha256,
            "selection_propensity_hex": self.selection_propensity.hex(),
            "propensity_identified": self.propensity_identified,
            "feasible": self.feasible,
            "isolated_gain_hex": self.isolated_gain.hex(),
            "conditional_credit_hex": self.conditional_credit.hex(),
            "normalized_conditional_credit_hex": (
                self.normalized_conditional_credit.hex()
            ),
            "redundancy_fraction_hex": self.redundancy_fraction.hex(),
            "synergy_fraction_hex": self.synergy_fraction.hex(),
            "attribution_cells": [
                value.to_record() for value in self.attribution_cells
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }

    @classmethod
    def from_record(
        cls,
        record: dict[str, object],
    ) -> "ResidualHeadroomObservation":
        if type(record) is not dict:
            raise TypeError("observation record must be an exact object")
        cells = record["attribution_cells"]
        if (
            type(cells) is not list
            or any(type(value) is not dict for value in cells)
        ):
            raise TypeError("attribution_cells record must be a list")
        value = cls(
            context_sha256=str(record["context_sha256"]),
            residual_request_sha256=str(
                record["residual_request_sha256"]
            ),
            generation_index=int(record["generation_index"]),
            wave_index=int(record["wave_index"]),
            action_sha256=str(record["action_sha256"]),
            evaluation_sha256=str(record["evaluation_sha256"]),
            outcome_sha256=str(record["outcome_sha256"]),
            set_outcome_sha256=str(record["set_outcome_sha256"]),
            decision_sha256=str(record["decision_sha256"]),
            selection_propensity=float.fromhex(
                str(record["selection_propensity_hex"])
            ),
            propensity_identified=bool(record["propensity_identified"]),
            feasible=bool(record["feasible"]),
            isolated_gain=float.fromhex(
                str(record["isolated_gain_hex"])
            ),
            conditional_credit=float.fromhex(
                str(record["conditional_credit_hex"])
            ),
            normalized_conditional_credit=float.fromhex(
                str(record["normalized_conditional_credit_hex"])
            ),
            redundancy_fraction=float.fromhex(
                str(record["redundancy_fraction_hex"])
            ),
            synergy_fraction=float.fromhex(
                str(record["synergy_fraction_hex"])
            ),
            attribution_cells=tuple(
                AdaptiveActionFactorCell(
                    family_id=str(cell["family_id"]),
                    level_id=str(cell["level_id"]),
                )
                for cell in cells
            ),
        )
        if value.observation_sha256 != str(
            record["observation_sha256"]
        ):
            raise ValueError("observation record hash does not authenticate")
        return value


@dataclass(frozen=True, slots=True)
class ResidualHeadroomStageClosure:
    """Authenticated, conserved learning evidence for one real stage."""

    context_sha256: str
    residual_request_sha256: str
    generation_index: int
    reference_gain_scale: float
    reference_gain_evidence_sha256: str
    decision_sha256s: tuple[str, ...]
    set_outcome_sha256s: tuple[str, ...]
    observations: tuple[ResidualHeadroomObservation, ...]
    total_conditional_gain: float
    closure_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "context_sha256",
            "residual_request_sha256",
            "reference_gain_evidence_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.generation_index) is not int or self.generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if (
            type(self.reference_gain_scale) is not float
            or not math.isfinite(self.reference_gain_scale)
            or self.reference_gain_scale <= 0.0
        ):
            raise ValueError("reference_gain_scale must be positive")
        for values, name in (
            (self.decision_sha256s, "decision_sha256s"),
            (self.set_outcome_sha256s, "set_outcome_sha256s"),
        ):
            if (
                type(values) is not tuple
                or not values
                or values != tuple(dict.fromkeys(values))
            ):
                raise ValueError(f"{name} must be non-empty and ordered unique")
            for value in values:
                require_sha256(value, name)
        if len(self.decision_sha256s) != len(self.set_outcome_sha256s):
            raise ValueError("each decision must bind one set outcome")
        if type(self.observations) is not tuple or not self.observations:
            raise ValueError("observations must be a non-empty exact tuple")
        identities: list[str] = []
        for value in self.observations:
            if type(value) is not ResidualHeadroomObservation:
                raise TypeError("observations must contain exact values")
            value.__post_init__()
            if (
                value.context_sha256 != self.context_sha256
                or value.residual_request_sha256
                != self.residual_request_sha256
                or value.generation_index != self.generation_index
                or value.decision_sha256
                not in self.decision_sha256s
                or value.set_outcome_sha256
                not in self.set_outcome_sha256s
            ):
                raise ValueError("an observation names another stage closure")
            if not _close(
                value.normalized_conditional_credit,
                value.conditional_credit / self.reference_gain_scale,
            ):
                raise ValueError("normalized credit differs from stage scale")
            identities.append(value.action_sha256)
        if len(identities) != len(set(identities)):
            raise ValueError("stage closure repeats an action")
        _require_nonnegative(
            self.total_conditional_gain,
            name="total_conditional_gain",
        )
        attributed = math.fsum(
            value.conditional_credit for value in self.observations
        )
        if not _close(attributed, self.total_conditional_gain):
            raise ValueError("action credit does not conserve conditional gain")
        object.__setattr__(
            self,
            "closure_sha256",
            _hash(_CLOSURE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ledger_definition_sha256": (
                RESIDUAL_HEADROOM_LEDGER_DEFINITION_SHA256
            ),
            "context_sha256": self.context_sha256,
            "residual_request_sha256": self.residual_request_sha256,
            "generation_index": self.generation_index,
            "reference_gain_scale_hex": self.reference_gain_scale.hex(),
            "reference_gain_evidence_sha256": (
                self.reference_gain_evidence_sha256
            ),
            "decision_sha256s": list(self.decision_sha256s),
            "set_outcome_sha256s": list(self.set_outcome_sha256s),
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
            "total_conditional_gain_hex": (
                self.total_conditional_gain.hex()
            ),
            "credit_conserved": True,
            "predicted_values_admitted_to_archive": False,
            "workload_objective_model_provider_prompt_config_fields": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "observations": [
                value.to_record() for value in self.observations
            ],
            "closure_sha256": self.closure_sha256,
        }

    @classmethod
    def from_record(
        cls,
        record: dict[str, object],
    ) -> "ResidualHeadroomStageClosure":
        if type(record) is not dict:
            raise TypeError("closure record must be an exact object")
        observations = record["observations"]
        if (
            type(observations) is not list
            or any(type(value) is not dict for value in observations)
        ):
            raise TypeError("closure observations must be exact objects")
        value = cls(
            context_sha256=str(record["context_sha256"]),
            residual_request_sha256=str(
                record["residual_request_sha256"]
            ),
            generation_index=int(record["generation_index"]),
            reference_gain_scale=float.fromhex(
                str(record["reference_gain_scale_hex"])
            ),
            reference_gain_evidence_sha256=str(
                record["reference_gain_evidence_sha256"]
            ),
            decision_sha256s=tuple(
                str(item) for item in record["decision_sha256s"]
            ),
            set_outcome_sha256s=tuple(
                str(item) for item in record["set_outcome_sha256s"]
            ),
            observations=tuple(
                ResidualHeadroomObservation.from_record(value)
                for value in observations
            ),
            total_conditional_gain=float.fromhex(
                str(record["total_conditional_gain_hex"])
            ),
        )
        if value.closure_sha256 != str(record["closure_sha256"]):
            raise ValueError("closure record hash does not authenticate")
        return value


@dataclass(frozen=True, slots=True)
class ConservedResidualHeadroomProjector:
    """Compile one outcome-adaptive stage into conserved ledger evidence."""

    def project(
        self,
        *,
        context_sha256: str,
        generation_index: int,
        reference_gain_scale: float,
        reference_gain_evidence_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        diagnostic_decision: AdaptiveActionRacingDecision,
        continuation_decisions: tuple[
            AdaptiveActionRacingDecision, ...
        ],
        outcomes: tuple[AdaptiveActionOutcome, ...],
        set_outcomes: tuple[AdaptiveActionSetOutcome, ...],
    ) -> ResidualHeadroomStageClosure:
        require_sha256(context_sha256, "context_sha256")
        require_sha256(
            reference_gain_evidence_sha256,
            "reference_gain_evidence_sha256",
        )
        if type(generation_index) is not int or generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if (
            type(reference_gain_scale) is not float
            or not math.isfinite(reference_gain_scale)
            or reference_gain_scale <= 0.0
        ):
            raise ValueError("reference_gain_scale must be positive")
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        action_by_sha256: dict[str, AdaptiveActionDescriptor] = {}
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact values")
            value.__post_init__()
            action_by_sha256[value.action_sha256] = value
        if len(action_by_sha256) != len(actions):
            raise ValueError("actions repeat an identity")
        if type(diagnostic_decision) is not AdaptiveActionRacingDecision:
            raise TypeError("diagnostic_decision must be exact")
        diagnostic_decision.__post_init__()
        if type(continuation_decisions) is not tuple:
            raise TypeError("continuation_decisions must be an exact tuple")
        decisions = (diagnostic_decision, *continuation_decisions)
        for value in decisions:
            if type(value) is not AdaptiveActionRacingDecision:
                raise TypeError("continuation decisions must be exact")
            value.__post_init__()
        request_sha256 = diagnostic_decision.residual_request_sha256
        if any(
            value.residual_request_sha256 != request_sha256
            for value in decisions
        ):
            raise ValueError("decisions name different residual requests")
        if (
            type(outcomes) is not tuple
            or not outcomes
            or any(type(value) is not AdaptiveActionOutcome for value in outcomes)
        ):
            raise ValueError("outcomes must be a non-empty exact tuple")
        outcome_by_action: dict[str, AdaptiveActionOutcome] = {}
        for value in outcomes:
            value.__post_init__()
            outcome_by_action[value.action_sha256] = value
        if len(outcome_by_action) != len(outcomes):
            raise ValueError("outcomes repeat an action")
        if (
            type(set_outcomes) is not tuple
            or len(set_outcomes) != len(decisions)
        ):
            raise ValueError("each decision requires one ordered set outcome")

        observations: list[ResidualHeadroomObservation] = []
        for wave_index, (decision, set_outcome) in enumerate(
            zip(decisions, set_outcomes, strict=True)
        ):
            if type(set_outcome) is not AdaptiveActionSetOutcome:
                raise TypeError("set outcomes must contain exact values")
            set_outcome.__post_init__()
            selected = tuple(
                value[0]
                for value in set_outcome.current_action_evaluation_bindings
            )
            if set(selected) != set(decision.selected_action_sha256s):
                raise ValueError(
                    "decision selection differs from its real set outcome"
                )
            wave_outcomes: list[AdaptiveActionOutcome] = []
            for action_sha256, evaluation_sha256 in (
                set_outcome.current_action_evaluation_bindings
            ):
                action = action_by_sha256.get(action_sha256)
                outcome = outcome_by_action.get(action_sha256)
                if (
                    action is None
                    or outcome is None
                    or outcome.evaluation_sha256 != evaluation_sha256
                ):
                    raise ValueError(
                        "set outcome lacks its action/evaluation binding"
                    )
                wave_outcomes.append(outcome)
            isolated_sum = math.fsum(
                value.marginal_archive_gain for value in wave_outcomes
            )
            conditional = set_outcome.conditional_set_gain
            if conditional == 0.0:
                credits = [0.0 for _ in wave_outcomes]
            elif isolated_sum > 0.0:
                credits = [
                    conditional
                    * value.marginal_archive_gain
                    / isolated_sum
                    for value in wave_outcomes
                ]
            else:
                credits = [
                    conditional / len(wave_outcomes)
                    for _ in wave_outcomes
                ]
            if credits:
                credits[-1] = max(
                    0.0,
                    conditional - math.fsum(credits[:-1]),
                )
            fixed = set_outcome.current_wave_fixed_set_gain
            redundancy_fraction = (
                0.0
                if fixed <= 0.0
                else min(
                    set_outcome.prior_conditioned_redundancy / fixed,
                    1.0,
                )
            )
            synergy_fraction = (
                0.0
                if conditional <= 0.0
                else min(
                    set_outcome.prior_conditioned_synergy / conditional,
                    1.0,
                )
            )
            identified = (
                len(decision.selected_action_sha256s) == 1
                or decision.selection_propensity == 1.0
            )
            effective_propensity = (
                decision.selection_propensity if identified else 1.0
            )
            for outcome, credit in zip(
                wave_outcomes,
                credits,
                strict=True,
            ):
                observations.append(
                    ResidualHeadroomObservation(
                        context_sha256=context_sha256,
                        residual_request_sha256=request_sha256,
                        generation_index=generation_index,
                        wave_index=wave_index,
                        action_sha256=outcome.action_sha256,
                        evaluation_sha256=outcome.evaluation_sha256,
                        outcome_sha256=outcome.outcome_sha256,
                        set_outcome_sha256=(
                            set_outcome.set_outcome_sha256
                        ),
                        decision_sha256=decision.decision_sha256,
                        selection_propensity=effective_propensity,
                        propensity_identified=identified,
                        feasible=outcome.feasible,
                        isolated_gain=outcome.marginal_archive_gain,
                        conditional_credit=float(credit),
                        normalized_conditional_credit=float(
                            credit / reference_gain_scale
                        ),
                        redundancy_fraction=redundancy_fraction,
                        synergy_fraction=synergy_fraction,
                        attribution_cells=_attribution_cells(
                            action_by_sha256[outcome.action_sha256]
                        ),
                    )
                )
        total_conditional_gain = math.fsum(
            value.conditional_set_gain for value in set_outcomes
        )
        return ResidualHeadroomStageClosure(
            context_sha256=context_sha256,
            residual_request_sha256=request_sha256,
            generation_index=generation_index,
            reference_gain_scale=reference_gain_scale,
            reference_gain_evidence_sha256=(
                reference_gain_evidence_sha256
            ),
            decision_sha256s=tuple(
                value.decision_sha256 for value in decisions
            ),
            set_outcome_sha256s=tuple(
                value.set_outcome_sha256 for value in set_outcomes
            ),
            observations=tuple(observations),
            total_conditional_gain=float(total_conditional_gain),
        )


@dataclass(frozen=True, slots=True)
class ResidualHeadroomLedgerConfig:
    """Portable posterior and risk controls for residual-headroom learning."""

    generation_decay: float = 0.85
    cross_context_weight: float = 0.0
    maximum_inverse_propensity: float = 10.0
    prior_strength: float = 2.0
    prior_normalized_gain: float = 1.0
    positive_prior_alpha: float = 1.0
    positive_prior_beta: float = 1.0
    invalid_prior_alpha: float = 1.0
    invalid_prior_beta: float = 9.0
    uncertainty_strength: float = 1.0
    late_bloom_strength: float = 0.5
    saturation_strength: float = 0.5
    redundancy_strength: float = 0.5
    invalidity_strength: float = 0.5
    exploration_floor: float = 0.25
    maximum_abs_slope: float = 2.0
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.generation_decay) is not float
            or not math.isfinite(self.generation_decay)
            or not 0.0 < self.generation_decay <= 1.0
        ):
            raise ValueError("generation_decay must lie in (0, 1]")
        if (
            type(self.cross_context_weight) is not float
            or not math.isfinite(self.cross_context_weight)
            or not 0.0 <= self.cross_context_weight <= 1.0
        ):
            raise ValueError("cross_context_weight must lie in [0, 1]")
        for value, name, positive in (
            (
                self.maximum_inverse_propensity,
                "maximum_inverse_propensity",
                True,
            ),
            (self.prior_strength, "prior_strength", True),
            (
                self.prior_normalized_gain,
                "prior_normalized_gain",
                False,
            ),
            (
                self.positive_prior_alpha,
                "positive_prior_alpha",
                True,
            ),
            (
                self.positive_prior_beta,
                "positive_prior_beta",
                True,
            ),
            (
                self.invalid_prior_alpha,
                "invalid_prior_alpha",
                True,
            ),
            (
                self.invalid_prior_beta,
                "invalid_prior_beta",
                True,
            ),
            (
                self.uncertainty_strength,
                "uncertainty_strength",
                False,
            ),
            (
                self.late_bloom_strength,
                "late_bloom_strength",
                False,
            ),
            (
                self.saturation_strength,
                "saturation_strength",
                False,
            ),
            (
                self.redundancy_strength,
                "redundancy_strength",
                False,
            ),
            (
                self.invalidity_strength,
                "invalidity_strength",
                False,
            ),
            (
                self.exploration_floor,
                "exploration_floor",
                False,
            ),
            (
                self.maximum_abs_slope,
                "maximum_abs_slope",
                True,
            ),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
                or (positive and value <= 0.0)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(_CONFIG_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ledger_id": RESIDUAL_HEADROOM_LEDGER_ID,
            "ledger_version": RESIDUAL_HEADROOM_LEDGER_VERSION,
            "ledger_definition_sha256": (
                RESIDUAL_HEADROOM_LEDGER_DEFINITION_SHA256
            ),
            "generation_decay_hex": self.generation_decay.hex(),
            "cross_context_weight_hex": self.cross_context_weight.hex(),
            "maximum_inverse_propensity_hex": (
                self.maximum_inverse_propensity.hex()
            ),
            "prior_strength_hex": self.prior_strength.hex(),
            "prior_normalized_gain_hex": (
                self.prior_normalized_gain.hex()
            ),
            "positive_prior_alpha_hex": (
                self.positive_prior_alpha.hex()
            ),
            "positive_prior_beta_hex": self.positive_prior_beta.hex(),
            "invalid_prior_alpha_hex": self.invalid_prior_alpha.hex(),
            "invalid_prior_beta_hex": self.invalid_prior_beta.hex(),
            "uncertainty_strength_hex": self.uncertainty_strength.hex(),
            "late_bloom_strength_hex": self.late_bloom_strength.hex(),
            "saturation_strength_hex": self.saturation_strength.hex(),
            "redundancy_strength_hex": self.redundancy_strength.hex(),
            "invalidity_strength_hex": self.invalidity_strength.hex(),
            "exploration_floor_hex": self.exploration_floor.hex(),
            "maximum_abs_slope_hex": self.maximum_abs_slope.hex(),
            "workload_objective_model_provider_prompt_config_branches": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResidualHeadroomLedgerState:
    """Immutable append-only set of conserved stage closures."""

    config_definition_sha256: str
    closures: tuple[ResidualHeadroomStageClosure, ...] = ()
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(
            self.config_definition_sha256,
            "config_definition_sha256",
        )
        if type(self.closures) is not tuple:
            raise TypeError("closures must be an exact tuple")
        closure_ids: list[str] = []
        stage_ids: list[tuple[str, str]] = []
        latest_generation_by_context: dict[str, int] = {}
        for value in self.closures:
            if type(value) is not ResidualHeadroomStageClosure:
                raise TypeError("closures must contain exact values")
            value.__post_init__()
            prior = latest_generation_by_context.get(value.context_sha256)
            if prior is not None and value.generation_index < prior:
                raise ValueError(
                    "closure generations must not regress within a context"
                )
            latest_generation_by_context[value.context_sha256] = (
                value.generation_index
            )
            closure_ids.append(value.closure_sha256)
            stage_ids.append(
                (value.context_sha256, value.residual_request_sha256)
            )
        if len(closure_ids) != len(set(closure_ids)):
            raise ValueError("ledger repeats a closure")
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("ledger repeats a context/request stage")
        object.__setattr__(
            self,
            "state_sha256",
            _hash(_STATE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ledger_definition_sha256": (
                RESIDUAL_HEADROOM_LEDGER_DEFINITION_SHA256
            ),
            "config_definition_sha256": self.config_definition_sha256,
            "closure_sha256s": [
                value.closure_sha256 for value in self.closures
            ],
            "append_only": True,
            "predicted_values_admitted_to_archive": False,
        }

    def to_record(self, *, include_closures: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "state_sha256": self.state_sha256,
        }
        if include_closures:
            record["closures"] = [
                value.to_record() for value in self.closures
            ]
        return record


@dataclass(frozen=True, slots=True)
class ResidualHeadroomCellPosterior:
    cell: AdaptiveActionFactorCell
    posterior_mean: float
    uncertainty: float
    positive_probability: float
    invalid_probability: float
    redundancy_fraction: float
    late_bloom_slope: float
    saturation_slope: float
    effective_sample_size: float
    raw_observation_count: int

    def __post_init__(self) -> None:
        if type(self.cell) is not AdaptiveActionFactorCell:
            raise TypeError("cell must be exact")
        self.cell.__post_init__()
        for name in (
            "posterior_mean",
            "uncertainty",
            "positive_probability",
            "invalid_probability",
            "redundancy_fraction",
            "late_bloom_slope",
            "saturation_slope",
            "effective_sample_size",
        ):
            _require_nonnegative(getattr(self, name), name=name)
        for name in (
            "positive_probability",
            "invalid_probability",
            "redundancy_fraction",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if (
            type(self.raw_observation_count) is not int
            or self.raw_observation_count < 0
        ):
            raise ValueError("raw_observation_count must be non-negative")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "cell": self.cell.to_record(),
            "posterior_mean_hex": self.posterior_mean.hex(),
            "uncertainty_hex": self.uncertainty.hex(),
            "positive_probability_hex": (
                self.positive_probability.hex()
            ),
            "invalid_probability_hex": self.invalid_probability.hex(),
            "redundancy_fraction_hex": self.redundancy_fraction.hex(),
            "late_bloom_slope_hex": self.late_bloom_slope.hex(),
            "saturation_slope_hex": self.saturation_slope.hex(),
            "effective_sample_size_hex": (
                self.effective_sample_size.hex()
            ),
            "raw_observation_count": self.raw_observation_count,
        }


@dataclass(frozen=True, slots=True)
class ResidualHeadroomEstimate:
    context_sha256: str
    action_sha256: str
    generation_index: int
    expected_normalized_gain: float
    uncertainty: float
    positive_probability: float
    invalid_probability: float
    redundancy_fraction: float
    late_bloom_headroom: float
    saturation_risk: float
    acquisition_score: float
    cell_posteriors: tuple[ResidualHeadroomCellPosterior, ...]
    ledger_state_sha256: str
    config_definition_sha256: str
    estimate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "context_sha256",
            "action_sha256",
            "ledger_state_sha256",
            "config_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.generation_index) is not int or self.generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        for name in (
            "expected_normalized_gain",
            "uncertainty",
            "positive_probability",
            "invalid_probability",
            "redundancy_fraction",
            "late_bloom_headroom",
            "saturation_risk",
            "acquisition_score",
        ):
            _require_nonnegative(getattr(self, name), name=name)
        for name in (
            "positive_probability",
            "invalid_probability",
            "redundancy_fraction",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if (
            type(self.cell_posteriors) is not tuple
            or not self.cell_posteriors
            or tuple(value.cell for value in self.cell_posteriors)
            != tuple(sorted({value.cell for value in self.cell_posteriors}))
        ):
            raise ValueError("cell posteriors must be non-empty and canonical")
        for value in self.cell_posteriors:
            if type(value) is not ResidualHeadroomCellPosterior:
                raise TypeError("cell posteriors must be exact")
            value.__post_init__()
        object.__setattr__(
            self,
            "estimate_sha256",
            _hash(_ESTIMATE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "context_sha256": self.context_sha256,
            "action_sha256": self.action_sha256,
            "generation_index": self.generation_index,
            "expected_normalized_gain_hex": (
                self.expected_normalized_gain.hex()
            ),
            "uncertainty_hex": self.uncertainty.hex(),
            "positive_probability_hex": (
                self.positive_probability.hex()
            ),
            "invalid_probability_hex": self.invalid_probability.hex(),
            "redundancy_fraction_hex": self.redundancy_fraction.hex(),
            "late_bloom_headroom_hex": self.late_bloom_headroom.hex(),
            "saturation_risk_hex": self.saturation_risk.hex(),
            "acquisition_score_hex": self.acquisition_score.hex(),
            "cell_posteriors": [
                value.to_record() for value in self.cell_posteriors
            ],
            "ledger_state_sha256": self.ledger_state_sha256,
            "config_definition_sha256": self.config_definition_sha256,
            "predicted_value_admitted_to_archive": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "estimate_sha256": self.estimate_sha256,
        }


@dataclass(frozen=True, slots=True)
class ConservedResidualHeadroomLedger:
    """Fold conserved closures and query action-level residual headroom."""

    config: ResidualHeadroomLedgerConfig = field(
        default_factory=ResidualHeadroomLedgerConfig
    )

    def __post_init__(self) -> None:
        if type(self.config) is not ResidualHeadroomLedgerConfig:
            raise TypeError("config must be exact")
        self.config.__post_init__()

    def empty_state(self) -> ResidualHeadroomLedgerState:
        self.__post_init__()
        return ResidualHeadroomLedgerState(
            config_definition_sha256=self.config.definition_sha256,
        )

    def append(
        self,
        state: ResidualHeadroomLedgerState,
        closure: ResidualHeadroomStageClosure,
    ) -> ResidualHeadroomLedgerState:
        self.__post_init__()
        if type(state) is not ResidualHeadroomLedgerState:
            raise TypeError("state must be exact")
        state.__post_init__()
        if (
            state.config_definition_sha256
            != self.config.definition_sha256
        ):
            raise ValueError("state belongs to another ledger configuration")
        if type(closure) is not ResidualHeadroomStageClosure:
            raise TypeError("closure must be exact")
        closure.__post_init__()
        return ResidualHeadroomLedgerState(
            config_definition_sha256=self.config.definition_sha256,
            closures=(*state.closures, closure),
        )

    def _cell_posterior(
        self,
        *,
        state: ResidualHeadroomLedgerState,
        context_sha256: str,
        generation_index: int,
        cell: AdaptiveActionFactorCell,
        candidate_cell_count: int,
    ) -> ResidualHeadroomCellPosterior:
        samples: list[tuple[ResidualHeadroomObservation, float, float]] = []
        for closure in state.closures:
            for observation in closure.observations:
                if cell not in observation.attribution_cells:
                    continue
                if observation.context_sha256 == context_sha256:
                    if observation.generation_index > generation_index:
                        continue
                    context_weight = 1.0
                    decay = self.config.generation_decay ** (
                        generation_index - observation.generation_index
                    )
                else:
                    context_weight = self.config.cross_context_weight
                    decay = 1.0
                if context_weight == 0.0:
                    continue
                inverse_propensity = (
                    min(
                        self.config.maximum_inverse_propensity,
                        1.0 / observation.selection_propensity,
                    )
                    if observation.propensity_identified
                    else 1.0
                )
                weight = context_weight * decay * inverse_propensity
                target = (
                    observation.normalized_conditional_credit
                    / len(observation.attribution_cells)
                )
                samples.append((observation, weight, target))

        prior_mean = (
            self.config.prior_normalized_gain / candidate_cell_count
        )
        weighted_count = math.fsum(value[1] for value in samples)
        denominator = self.config.prior_strength + weighted_count
        posterior_mean = (
            self.config.prior_strength * prior_mean
            + math.fsum(weight * target for _, weight, target in samples)
        ) / denominator
        variance = math.fsum(
            weight * (target - posterior_mean) ** 2
            for _, weight, target in samples
        ) / denominator
        uncertainty = (
            math.sqrt(max(variance, 0.0) / denominator)
            + max(prior_mean, 1.0e-12) / math.sqrt(denominator)
        )
        positive_probability = (
            self.config.positive_prior_alpha
            + math.fsum(
                weight * float(observation.conditional_credit > 0.0)
                for observation, weight, _ in samples
            )
        ) / (
            self.config.positive_prior_alpha
            + self.config.positive_prior_beta
            + weighted_count
        )
        invalid_probability = (
            self.config.invalid_prior_alpha
            + math.fsum(
                weight * float(not observation.feasible)
                for observation, weight, _ in samples
            )
        ) / (
            self.config.invalid_prior_alpha
            + self.config.invalid_prior_beta
            + weighted_count
        )
        redundancy_fraction = (
            0.0
            if weighted_count == 0.0
            else math.fsum(
                weight * observation.redundancy_fraction
                for observation, weight, _ in samples
            )
            / weighted_count
        )
        slope = 0.0
        distinct_waves = {
            observation.wave_index for observation, _, _ in samples
        }
        if weighted_count > 0.0 and len(distinct_waves) >= 2:
            mean_wave = math.fsum(
                weight * observation.wave_index
                for observation, weight, _ in samples
            ) / weighted_count
            mean_target = math.fsum(
                weight * target for _, weight, target in samples
            ) / weighted_count
            wave_variance = math.fsum(
                weight * (observation.wave_index - mean_wave) ** 2
                for observation, weight, _ in samples
            )
            if wave_variance > 0.0:
                slope = math.fsum(
                    weight
                    * (observation.wave_index - mean_wave)
                    * (target - mean_target)
                    for observation, weight, target in samples
                ) / wave_variance
                slope = max(
                    -self.config.maximum_abs_slope,
                    min(self.config.maximum_abs_slope, slope),
                )
        sum_weight_squared = math.fsum(
            weight * weight for _, weight, _ in samples
        )
        effective_sample_size = (
            0.0
            if sum_weight_squared == 0.0
            else weighted_count * weighted_count / sum_weight_squared
        )
        return ResidualHeadroomCellPosterior(
            cell=cell,
            posterior_mean=float(posterior_mean),
            uncertainty=float(uncertainty),
            positive_probability=float(positive_probability),
            invalid_probability=float(invalid_probability),
            redundancy_fraction=float(redundancy_fraction),
            late_bloom_slope=float(max(slope, 0.0)),
            saturation_slope=float(max(-slope, 0.0)),
            effective_sample_size=float(effective_sample_size),
            raw_observation_count=len(samples),
        )

    def estimate(
        self,
        *,
        state: ResidualHeadroomLedgerState,
        context_sha256: str,
        generation_index: int,
        action: AdaptiveActionDescriptor,
    ) -> ResidualHeadroomEstimate:
        self.__post_init__()
        if type(state) is not ResidualHeadroomLedgerState:
            raise TypeError("state must be exact")
        state.__post_init__()
        if (
            state.config_definition_sha256
            != self.config.definition_sha256
        ):
            raise ValueError("state belongs to another ledger configuration")
        require_sha256(context_sha256, "context_sha256")
        if type(generation_index) is not int or generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if type(action) is not AdaptiveActionDescriptor:
            raise TypeError("action must be exact")
        cells = _attribution_cells(action)
        posteriors = tuple(
            self._cell_posterior(
                state=state,
                context_sha256=context_sha256,
                generation_index=generation_index,
                cell=cell,
                candidate_cell_count=len(cells),
            )
            for cell in cells
        )
        expected = math.fsum(value.posterior_mean for value in posteriors)
        uncertainty = math.sqrt(
            math.fsum(value.uncertainty**2 for value in posteriors)
        )
        raw_count = sum(
            value.raw_observation_count for value in posteriors
        )
        uncertainty += self.config.exploration_floor / math.sqrt(
            1.0 + raw_count
        )
        positive_probability = math.fsum(
            value.positive_probability for value in posteriors
        ) / len(posteriors)
        invalid_probability = math.fsum(
            value.invalid_probability for value in posteriors
        ) / len(posteriors)
        redundancy = math.fsum(
            value.redundancy_fraction for value in posteriors
        ) / len(posteriors)
        late_bloom = math.fsum(
            value.late_bloom_slope for value in posteriors
        )
        saturation = math.fsum(
            value.saturation_slope for value in posteriors
        )
        acquisition = max(
            0.0,
            expected
            + self.config.uncertainty_strength * uncertainty
            + self.config.late_bloom_strength * late_bloom
            - self.config.saturation_strength * saturation
            - self.config.redundancy_strength * expected * redundancy
            - self.config.invalidity_strength
            * max(self.config.prior_normalized_gain, expected)
            * invalid_probability,
        )
        return ResidualHeadroomEstimate(
            context_sha256=context_sha256,
            action_sha256=action.action_sha256,
            generation_index=generation_index,
            expected_normalized_gain=float(expected),
            uncertainty=float(uncertainty),
            positive_probability=float(positive_probability),
            invalid_probability=float(invalid_probability),
            redundancy_fraction=float(redundancy),
            late_bloom_headroom=float(late_bloom),
            saturation_risk=float(saturation),
            acquisition_score=float(acquisition),
            cell_posteriors=posteriors,
            ledger_state_sha256=state.state_sha256,
            config_definition_sha256=self.config.definition_sha256,
        )


@dataclass(frozen=True, slots=True)
class ResidualHeadroomAdaptiveMarketProjector:
    """Blend prior-only headroom ranks into any portable adaptive market."""

    delegate: AdaptiveActionMarketProjectorPort = field(
        repr=False,
        compare=False,
    )
    ledger: ConservedResidualHeadroomLedger = field(
        repr=False,
        compare=False,
    )
    ledger_state: ResidualHeadroomLedgerState
    context_sha256: str
    base_prior_weight: float = 1.0
    headroom_weight: float = 1.0
    projector_id: str = RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_ID
    projector_version: int = (
        RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_VERSION
    )
    definition_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.delegate, AdaptiveActionMarketProjectorPort):
            raise TypeError("delegate must implement the adaptive market port")
        self.delegate.__post_init__()
        if type(self.ledger) is not ConservedResidualHeadroomLedger:
            raise TypeError("ledger must be exact")
        self.ledger.__post_init__()
        if type(self.ledger_state) is not ResidualHeadroomLedgerState:
            raise TypeError("ledger_state must be exact")
        self.ledger_state.__post_init__()
        if (
            self.ledger_state.config_definition_sha256
            != self.ledger.config.definition_sha256
        ):
            raise ValueError("ledger state belongs to another configuration")
        require_sha256(self.context_sha256, "context_sha256")
        for value, name in (
            (self.base_prior_weight, "base_prior_weight"),
            (self.headroom_weight, "headroom_weight"),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.base_prior_weight + self.headroom_weight <= 0.0:
            raise ValueError("at least one projector weight must be positive")
        if (
            self.projector_id
            != RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_ID
            or self.projector_version
            != RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_VERSION
        ):
            raise ValueError("projector identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _MARKET_PROJECTOR_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "delegate_definition_sha256": (
                        self.delegate.definition_sha256
                    ),
                    "ledger_config_definition_sha256": (
                        self.ledger.config.definition_sha256
                    ),
                    "base_prior_weight_hex": self.base_prior_weight.hex(),
                    "headroom_weight_hex": self.headroom_weight.hex(),
                    "normalization": (
                        "within-sealed-market-tie-preserving-rank-percentile"
                    ),
                    "candidate_outcomes_observed": False,
                    "predicted_values_admitted_to_archive": False,
                    "workload_objective_model_provider_prompt_config_branches": (
                        False
                    ),
                },
            ),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _hash(
                _MARKET_STATE_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_definition_sha256": self.definition_sha256,
                    "delegate_state_sha256": self.delegate.state_sha256,
                    "ledger_state_sha256": self.ledger_state.state_sha256,
                    "context_sha256": self.context_sha256,
                },
            ),
        )

    @staticmethod
    def _rank_percentiles(
        values: dict[str, float],
    ) -> dict[str, float]:
        unique = sorted(set(values.values()))
        if len(unique) == 1:
            return {key: 0.5 for key in values}
        percentile = {
            value: index / (len(unique) - 1)
            for index, value in enumerate(unique)
        }
        return {key: percentile[value] for key, value in values.items()}

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: tuple[BrokerActionScore, ...],
        required_action_sha256s: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        self.__post_init__()
        projected = await self.delegate.project(
            request,
            proposals,
            actions,
            scores,
            required_action_sha256s,
        )
        # A cold ledger contains no evidence with which to alter the existing
        # prior.  Returning the delegate values exactly avoids manufacturing a
        # ranking from attribution-cardinality or hash tie breaks.
        if not self.ledger_state.closures:
            return projected
        estimates = {
            action.action_sha256: self.ledger.estimate(
                state=self.ledger_state,
                context_sha256=self.context_sha256,
                generation_index=request.decision_index,
                action=action,
            ).acquisition_score
            for action in projected
        }
        headroom_percentiles = self._rank_percentiles(estimates)
        denominator = self.base_prior_weight + self.headroom_weight
        return tuple(
            replace(
                action,
                prior_score=float(
                    (
                        self.base_prior_weight * action.prior_score
                        + self.headroom_weight
                        * headroom_percentiles[action.action_sha256]
                    )
                    / denominator
                ),
            )
            for action in projected
        )


__all__ = [
    "ConservedResidualHeadroomLedger",
    "ConservedResidualHeadroomProjector",
    "ResidualHeadroomAdaptiveMarketProjector",
    "ResidualHeadroomCellPosterior",
    "ResidualHeadroomEstimate",
    "ResidualHeadroomLedgerConfig",
    "ResidualHeadroomLedgerState",
    "ResidualHeadroomObservation",
    "ResidualHeadroomStageClosure",
    "RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_ID",
    "RESIDUAL_HEADROOM_ADAPTIVE_MARKET_PROJECTOR_VERSION",
    "RESIDUAL_HEADROOM_LEDGER_DEFINITION_SHA256",
    "RESIDUAL_HEADROOM_LEDGER_ID",
    "RESIDUAL_HEADROOM_LEDGER_VERSION",
]
