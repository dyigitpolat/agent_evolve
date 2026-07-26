"""Randomized fixed-size insight retrieval with logged causal propensities.

The policy mixes a deterministic top-k subset with a uniformly random k-subset.
That small, explicit exploration component gives every eligible insight a known
conditional inclusion probability. Accumulated trials can therefore estimate
selected-versus-unselected marginal effects without pretending that an unseen
insight caused the outcome of one batch.

The estimators are intentionally modest. They provide stabilized inverse-
propensity contrasts for one context stratum and an optional two-insight
interaction contrast. They do not claim to solve nonstationarity, interference,
or arbitrary high-order subset interactions; the complete decisions remain
available for richer offline models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from numbers import Real
from typing import Mapping, Optional, Protocol, Sequence, Tuple

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef

_LOWER_SHA256 = frozenset("0123456789abcdef")


class RandomSubsetSource(Protocol):
    """The narrow random-source surface needed by the selector."""

    def randrange(self, stop: int) -> int: ...
    def sample(self, population: Sequence[InsightRef], k: int) -> list[InsightRef]: ...


class InsightSelectionMode(str, Enum):
    EXPLOIT = "exploit"
    EXPLORE_UNIFORM = "explore_uniform"


def _require_hash(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_score(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sorted_refs(values: Sequence[InsightRef]) -> Tuple[InsightRef, ...]:
    result = tuple(sorted(values))
    if any(not isinstance(value, InsightRef) for value in result):
        raise TypeError("insight collections must contain InsightRef values")
    if len(set(result)) != len(result):
        raise ValueError("insight collections cannot contain duplicates")
    return result


def _top_k(
    score_snapshot: Tuple[Tuple[InsightRef, float], ...],
    subset_size: int,
) -> Tuple[InsightRef, ...]:
    ranked = sorted(
        score_snapshot,
        key=lambda item: (-item[1], item[0].insight_id.value, item[0].version),
    )
    return tuple(sorted(reference for reference, _ in ranked[:subset_size]))


@dataclass(frozen=True, slots=True)
class InsightSelectionDecision:
    """One replayable retrieval decision and its conditional assignment law."""

    context_hash: str
    eligible: Tuple[InsightRef, ...]
    selected: Tuple[InsightRef, ...]
    exploitation_subset: Tuple[InsightRef, ...]
    score_snapshot: Tuple[Tuple[InsightRef, float], ...]
    subset_size: int
    exploration_probability: Fraction
    mode: InsightSelectionMode
    selected_subset_probability: Fraction
    policy_id: str = "epsilon_greedy_uniform_k_subset"
    policy_version: int = 1

    def __post_init__(self) -> None:
        _require_hash(self.context_hash, "context_hash")
        eligible = _sorted_refs(self.eligible)
        selected = _sorted_refs(self.selected)
        exploitation = _sorted_refs(self.exploitation_subset)
        if eligible != self.eligible or selected != self.selected:
            raise ValueError("eligible and selected insights must use canonical sorted order")
        if exploitation != self.exploitation_subset:
            raise ValueError("exploitation_subset must use canonical sorted order")
        if type(self.subset_size) is not int or self.subset_size < 0:
            raise ValueError("subset_size must be a non-negative integer")
        if self.subset_size > len(eligible):
            raise ValueError("subset_size cannot exceed the eligible set")
        if len(selected) != self.subset_size or len(exploitation) != self.subset_size:
            raise ValueError("selected subsets must have exactly subset_size members")
        if not set(selected).issubset(eligible) or not set(exploitation).issubset(eligible):
            raise ValueError("selected subsets must be drawn from eligible insights")
        if type(self.exploration_probability) is not Fraction:
            raise TypeError("exploration_probability must be an exact Fraction")
        if not Fraction(0) <= self.exploration_probability <= Fraction(1):
            raise ValueError("exploration_probability must lie in [0,1]")
        if not isinstance(self.mode, InsightSelectionMode):
            raise TypeError("mode must be an InsightSelectionMode")
        if (
            self.exploration_probability == 0
            and self.mode is not InsightSelectionMode.EXPLOIT
        ):
            raise ValueError("zero exploration probability requires exploit mode")
        if (
            self.exploration_probability == 1
            and self.mode is not InsightSelectionMode.EXPLORE_UNIFORM
        ):
            raise ValueError("unit exploration probability requires uniform-exploration mode")
        if type(self.selected_subset_probability) is not Fraction:
            raise TypeError("selected_subset_probability must be an exact Fraction")
        if self.selected_subset_probability <= 0 or self.selected_subset_probability > 1:
            raise ValueError("selected_subset_probability must lie in (0,1]")
        if type(self.policy_id) is not str or self.policy_id != "epsilon_greedy_uniform_k_subset":
            raise ValueError("unsupported insight selection policy_id")
        if type(self.policy_version) is not int or self.policy_version != 1:
            raise ValueError("unsupported insight selection policy_version")

        if type(self.score_snapshot) is not tuple:
            raise TypeError("score_snapshot must be an immutable tuple")
        score_refs = []
        canonical_scores = []
        for item in self.score_snapshot:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("score_snapshot entries must be (InsightRef, score) tuples")
            reference, score = item
            if not isinstance(reference, InsightRef):
                raise TypeError("score_snapshot keys must be InsightRef values")
            score_refs.append(reference)
            canonical_scores.append((reference, _finite_score(score, "insight score")))
        if tuple(canonical_scores) != self.score_snapshot:
            raise TypeError("score_snapshot scores must already be canonical floats")
        if tuple(score_refs) != eligible:
            raise ValueError("score_snapshot must align exactly with canonical eligible insights")
        if _top_k(self.score_snapshot, self.subset_size) != exploitation:
            raise ValueError("exploitation_subset does not match the recorded scores")

        expected_subset_probability = self._subset_probability(selected)
        if self.selected_subset_probability != expected_subset_probability:
            raise ValueError("selected_subset_probability does not match the policy law")
        if self.mode is InsightSelectionMode.EXPLOIT and selected != exploitation:
            raise ValueError("exploit mode must select the exploitation subset")

    @property
    def credit_identifiable(self) -> bool:
        """Whether individual inclusion has overlap under this decision law."""

        return any(
            Fraction(0) < self.inclusion_probability(reference) < Fraction(1)
            for reference in self.eligible
        )

    def _uniform_subset_probability(self) -> Fraction:
        count = math.comb(len(self.eligible), self.subset_size)
        return Fraction(1, count)

    def _subset_probability(self, subset: Tuple[InsightRef, ...]) -> Fraction:
        probability = self.exploration_probability * self._uniform_subset_probability()
        if subset == self.exploitation_subset:
            probability += 1 - self.exploration_probability
        return probability

    def inclusion_probability(self, reference: InsightRef) -> Fraction:
        """Return the exact conditional probability that ``reference`` is selected."""

        if reference not in self.eligible:
            raise ValueError("insight was not eligible for this decision")
        count = len(self.eligible)
        uniform = Fraction(self.subset_size, count) if count else Fraction(0)
        exploit = Fraction(int(reference in self.exploitation_subset))
        return self.exploration_probability * uniform + (1 - self.exploration_probability) * exploit

    def joint_cell_probability(
        self,
        first: InsightRef,
        second: InsightRef,
        first_selected: bool,
        second_selected: bool,
    ) -> Fraction:
        """Exact probability for one two-insight inclusion/exclusion cell."""

        if first == second:
            raise ValueError("pair probabilities require two distinct insights")
        if type(first_selected) is not bool or type(second_selected) is not bool:
            raise TypeError("pair cell flags must be bool")
        if first not in self.eligible or second not in self.eligible:
            raise ValueError("both insights must be eligible for this decision")
        count = len(self.eligible)
        if count < 2:
            raise ValueError("pair probabilities require at least two eligible insights")
        uniform_both = Fraction(
            self.subset_size * (self.subset_size - 1),
            count * (count - 1),
        )
        exploit_both = Fraction(
            int(first in self.exploitation_subset and second in self.exploitation_subset)
        )
        both = (
            self.exploration_probability * uniform_both
            + (1 - self.exploration_probability) * exploit_both
        )
        first_probability = self.inclusion_probability(first)
        second_probability = self.inclusion_probability(second)
        cells = {
            (True, True): both,
            (True, False): first_probability - both,
            (False, True): second_probability - both,
            (False, False): 1 - first_probability - second_probability + both,
        }
        probability = cells[(first_selected, second_selected)]
        if probability < 0:  # pragma: no cover - algebraic implementation guard.
            raise RuntimeError("invalid negative pair-cell probability")
        return probability


@dataclass(frozen=True, slots=True)
class EpsilonGreedySubsetSelector:
    """Mix deterministic score exploitation with uniform fixed-size exploration."""

    exploration_probability: Fraction = Fraction(1, 4)

    def __post_init__(self) -> None:
        if type(self.exploration_probability) is not Fraction:
            raise TypeError("exploration_probability must be an exact Fraction")
        if not Fraction(0) <= self.exploration_probability <= Fraction(1):
            raise ValueError("exploration_probability must lie in [0,1]")

    def select(
        self,
        *,
        context_hash: str,
        eligible: Sequence[InsightRef],
        scores: Mapping[InsightRef, Real],
        subset_size: int,
        rng: RandomSubsetSource,
    ) -> InsightSelectionDecision:
        """Select a subset and record the complete conditional assignment law."""

        _require_hash(context_hash, "context_hash")
        canonical_eligible = _sorted_refs(eligible)
        if type(subset_size) is not int or subset_size < 0:
            raise ValueError("subset_size must be a non-negative integer")
        if subset_size > len(canonical_eligible):
            raise ValueError("subset_size cannot exceed the eligible set")
        if set(scores) != set(canonical_eligible):
            raise ValueError("scores must contain exactly the eligible insight references")
        score_snapshot = tuple(
            (reference, _finite_score(scores[reference], "insight score"))
            for reference in canonical_eligible
        )
        exploitation = _top_k(score_snapshot, subset_size)

        if self.exploration_probability == 0:
            mode = InsightSelectionMode.EXPLOIT
        elif self.exploration_probability == 1:
            mode = InsightSelectionMode.EXPLORE_UNIFORM
        else:
            # Integer sampling preserves the exact logged Fraction law.  A
            # float threshold would silently implement a nearby dyadic law for
            # most rational epsilon values and can double very small rates.
            denominator = self.exploration_probability.denominator
            draw = rng.randrange(denominator)
            if type(draw) is not int:
                raise TypeError("random source randrange must return an integer")
            if draw < 0 or draw >= denominator:
                raise ValueError("random source randrange result is out of bounds")
            mode = (
                InsightSelectionMode.EXPLORE_UNIFORM
                if draw < self.exploration_probability.numerator
                else InsightSelectionMode.EXPLOIT
            )

        if mode is InsightSelectionMode.EXPLORE_UNIFORM:
            sampled = rng.sample(canonical_eligible, subset_size)
            selected = _sorted_refs(sampled)
            if len(selected) != subset_size or not set(selected).issubset(canonical_eligible):
                raise ValueError("random source returned an invalid uniform subset")
        else:
            selected = exploitation

        uniform_probability = Fraction(1, math.comb(len(canonical_eligible), subset_size))
        selected_probability = self.exploration_probability * uniform_probability
        if selected == exploitation:
            selected_probability += 1 - self.exploration_probability
        return InsightSelectionDecision(
            context_hash=context_hash,
            eligible=canonical_eligible,
            selected=selected,
            exploitation_subset=exploitation,
            score_snapshot=score_snapshot,
            subset_size=subset_size,
            exploration_probability=self.exploration_probability,
            mode=mode,
            selected_subset_probability=selected_probability,
        )


@dataclass(frozen=True, slots=True)
class InsightTrial:
    """One randomized selection unit and its predeclared aggregate reward.

    A single operator invocation may generate several candidates from the same
    selected insight subset. Those candidates must be aggregated into this one
    credit unit; copying the reward into one trial per child would be
    pseudoreplication and is rejected through unique invocation/candidate checks.
    """

    credit_unit_id: OperatorInvocationId
    candidate_ids: Tuple[CandidateId, ...]
    reward_definition_hash: str
    decision: InsightSelectionDecision
    reward: float
    treatment_binding_sha256: str | None = None
    generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.credit_unit_id, OperatorInvocationId):
            raise TypeError("credit_unit_id must be an OperatorInvocationId")
        if type(self.candidate_ids) is not tuple:
            raise TypeError("candidate_ids must be an immutable tuple")
        if any(not isinstance(value, CandidateId) for value in self.candidate_ids):
            raise TypeError("candidate_ids must contain only CandidateId values")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids cannot contain duplicates")
        _require_hash(self.reward_definition_hash, "reward_definition_hash")
        if not isinstance(self.decision, InsightSelectionDecision):
            raise TypeError("decision must be an InsightSelectionDecision")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if self.treatment_binding_sha256 is not None:
            _require_hash(
                self.treatment_binding_sha256,
                "treatment_binding_sha256",
            )
        if self.generation is not None and (
            type(self.generation) is not int or self.generation <= 0
        ):
            raise ValueError("generation must be a positive exact integer or None")


@dataclass(frozen=True, slots=True)
class MarginalEffectEstimate:
    insight: InsightRef
    context_hash: Optional[str]
    subset_size: Optional[int]
    exploration_probability: Optional[Fraction]
    reward_definition_hash: Optional[str]
    policy_id: Optional[str]
    policy_version: Optional[int]
    effect: Optional[float]
    treated_mean: Optional[float]
    control_mean: Optional[float]
    treated_trials: int
    control_trials: int
    treated_effective_sample_size: float
    control_effective_sample_size: float
    eligible_trials: int
    overlap_trials: int

    @property
    def identified(self) -> bool:
        return self.effect is not None


@dataclass(frozen=True, slots=True)
class PairSynergyEstimate:
    first: InsightRef
    second: InsightRef
    context_hash: Optional[str]
    subset_size: Optional[int]
    exploration_probability: Optional[Fraction]
    reward_definition_hash: Optional[str]
    policy_id: Optional[str]
    policy_version: Optional[int]
    synergy: Optional[float]
    cell_means: Tuple[Tuple[str, Optional[float]], ...]
    cell_trials: Tuple[Tuple[str, int], ...]
    cell_effective_sample_sizes: Tuple[Tuple[str, float], ...]
    eligible_trials: int
    overlap_trials: int

    @property
    def identified(self) -> bool:
        return self.synergy is not None


@dataclass(frozen=True, slots=True)
class _WeightedSummary:
    mean: Optional[float]
    exact_mean: Optional[Fraction]
    trials: int
    effective_sample_size: float


def _validate_trials(trials: Sequence[InsightTrial]) -> None:
    credit_unit_ids = []
    candidate_ids = []
    for trial in trials:
        if not isinstance(trial, InsightTrial):
            raise TypeError("trials must contain InsightTrial values")
        credit_unit_ids.append(trial.credit_unit_id)
        candidate_ids.extend(trial.candidate_ids)
    if len(set(credit_unit_ids)) != len(credit_unit_ids):
        raise ValueError("an operator invocation may appear in the credit trial set only once")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("a candidate may appear in the insight-credit trial set only once")


def _validate_reward_definition(trials: Sequence[InsightTrial]) -> Optional[str]:
    definitions = {trial.reward_definition_hash for trial in trials}
    if len(definitions) > 1:
        raise ValueError("an insight-credit estimate cannot mix reward definitions")
    return next(iter(definitions), None)


def _weighted_summary(observations: Sequence[Tuple[float, Fraction]]) -> _WeightedSummary:
    if not observations:
        return _WeightedSummary(None, None, 0, 0.0)
    # Accumulate the complete Hájek numerator and denominator as rationals.
    # Normalizing weights and rewards in separate float domains can double-
    # underflow a product even when the final weighted mean is representable.
    # Fraction.from_float preserves the exact admitted binary-float reward.
    exact_weights = [1 / probability for _, probability in observations]
    rewards = [reward for reward, _ in observations]
    weight_sum = sum(exact_weights, Fraction(0))
    weighted_reward_sum = sum(
        (
            weight * Fraction.from_float(reward)
            for weight, reward in zip(exact_weights, rewards, strict=True)
        ),
        Fraction(0),
    )
    exact_mean = weighted_reward_sum / weight_sum
    mean = _exact_to_finite_float(exact_mean, "weighted mean")
    squared_weight_sum = sum(
        (weight * weight for weight in exact_weights), Fraction(0)
    )
    exact_effective_sample_size = weight_sum * weight_sum / squared_weight_sum
    effective_sample_size = _exact_to_finite_float(
        exact_effective_sample_size, "effective sample size"
    )
    return _WeightedSummary(
        mean, exact_mean, len(observations), effective_sample_size
    )


def _validate_estimand_stratum(
    trials: Sequence[InsightTrial],
    requested_context_hash: Optional[str],
    assignment_trials: Sequence[InsightTrial],
) -> tuple[
    Optional[str], Optional[int], Optional[Fraction], Optional[str], Optional[int]
]:
    """Reject silent pooling across distinct policy-relative estimands."""

    contexts = {trial.decision.context_hash for trial in trials}
    if requested_context_hash is None:
        if len(contexts) > 1:
            raise ValueError(
                "an insight-credit estimate cannot mix context strata"
            )
        resolved_context = next(iter(contexts), None)
    else:
        resolved_context = requested_context_hash
        if any(context != requested_context_hash for context in contexts):
            raise ValueError("included trial does not match the requested context stratum")

    subset_sizes = {trial.decision.subset_size for trial in trials}
    if len(subset_sizes) > 1:
        raise ValueError(
            "an insight-credit estimate cannot mix subset-size estimands"
        )
    exploration_probabilities = {
        trial.decision.exploration_probability for trial in assignment_trials
    }
    if len(exploration_probabilities) > 1:
        raise ValueError(
            "an insight-credit estimate cannot mix exploration-policy strata"
        )
    policies = {
        (trial.decision.policy_id, trial.decision.policy_version) for trial in trials
    }
    if len(policies) > 1:
        raise ValueError("an insight-credit estimate cannot mix policy versions")
    policy_id, policy_version = next(iter(policies), (None, None))
    return (
        resolved_context,
        next(iter(subset_sizes), None),
        next(iter(exploration_probabilities), None),
        policy_id,
        policy_version,
    )


def _exact_to_finite_float(value: Fraction, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} cannot be represented as a finite float") from None
    if not math.isfinite(result) or (value != 0 and result == 0.0):
        raise ValueError(f"{name} cannot be represented as a finite float")
    return result


def _finite_linear_contrast(
    values: Sequence[Fraction], coefficients: Sequence[int]
) -> float:
    if len(values) != len(coefficients) or not values:
        raise ValueError("linear contrast inputs are invalid")
    result = sum(
        (
            coefficient * value
            for value, coefficient in zip(values, coefficients, strict=True)
        ),
        Fraction(0),
    )
    return _exact_to_finite_float(result, "effect contrast")


def estimate_marginal_effect(
    trials: Sequence[InsightTrial],
    insight: InsightRef,
    *,
    context_hash: Optional[str] = None,
) -> MarginalEffectEstimate:
    """Estimate selected-minus-unselected reward with stabilized IPW means.

    Propensities are conditional on each trial's eligible set and score state.
    Decisions with deterministic inclusion or exclusion provide no overlap and
    are reported but excluded from the contrast.
    """

    _validate_trials(trials)
    if not isinstance(insight, InsightRef):
        raise TypeError("insight must be an InsightRef")
    if context_hash is not None:
        _require_hash(context_hash, "context_hash")
    eligible_trials = 0
    overlap_trials = 0
    assignment_trials: list[InsightTrial] = []
    treated: list[Tuple[float, Fraction]] = []
    control: list[Tuple[float, Fraction]] = []
    included_trials: list[InsightTrial] = []
    for trial in trials:
        decision = trial.decision
        if context_hash is not None and decision.context_hash != context_hash:
            continue
        if insight not in decision.eligible:
            continue
        included_trials.append(trial)
        eligible_trials += 1
        inclusion = decision.inclusion_probability(insight)
        if inclusion <= 0 or inclusion >= 1:
            continue
        overlap_trials += 1
        assignment_trials.append(trial)
        if insight in decision.selected:
            treated.append((trial.reward, inclusion))
        else:
            control.append((trial.reward, 1 - inclusion))
    reward_definition_hash = _validate_reward_definition(included_trials)
    (
        resolved_context,
        subset_size,
        exploration_probability,
        policy_id,
        policy_version,
    ) = _validate_estimand_stratum(
        included_trials, context_hash, assignment_trials
    )
    treated_summary = _weighted_summary(treated)
    control_summary = _weighted_summary(control)
    effect = None
    if (
        treated_summary.exact_mean is not None
        and control_summary.exact_mean is not None
    ):
        effect = _finite_linear_contrast(
            (treated_summary.exact_mean, control_summary.exact_mean), (1, -1)
        )
    return MarginalEffectEstimate(
        insight=insight,
        context_hash=resolved_context,
        subset_size=subset_size,
        exploration_probability=exploration_probability,
        reward_definition_hash=reward_definition_hash,
        policy_id=policy_id,
        policy_version=policy_version,
        effect=effect,
        treated_mean=treated_summary.mean,
        control_mean=control_summary.mean,
        treated_trials=treated_summary.trials,
        control_trials=control_summary.trials,
        treated_effective_sample_size=treated_summary.effective_sample_size,
        control_effective_sample_size=control_summary.effective_sample_size,
        eligible_trials=eligible_trials,
        overlap_trials=overlap_trials,
    )


def estimate_pair_synergy(
    trials: Sequence[InsightTrial],
    first: InsightRef,
    second: InsightRef,
    *,
    context_hash: Optional[str] = None,
) -> PairSynergyEstimate:
    """Estimate ``E11 - E10 - E01 + E00`` for two insight versions.

    Only decisions assigning positive probability to all four pair cells are
    eligible for this interaction contrast. This makes fixed-size designs with
    structurally impossible cells fail closed instead of fabricating synergy.
    """

    _validate_trials(trials)
    if not isinstance(first, InsightRef) or not isinstance(second, InsightRef):
        raise TypeError("pair members must be InsightRef values")
    if first == second:
        raise ValueError("pair synergy requires two distinct insight versions")
    if second < first:
        first, second = second, first
    if context_hash is not None:
        _require_hash(context_hash, "context_hash")
    cell_order = ((True, True), (True, False), (False, True), (False, False))
    cell_names = {
        (True, True): "11",
        (True, False): "10",
        (False, True): "01",
        (False, False): "00",
    }
    observations: dict[Tuple[bool, bool], list[Tuple[float, Fraction]]] = {
        cell: [] for cell in cell_order
    }
    eligible_trials = 0
    overlap_trials = 0
    assignment_trials: list[InsightTrial] = []
    included_trials: list[InsightTrial] = []
    for trial in trials:
        decision = trial.decision
        if context_hash is not None and decision.context_hash != context_hash:
            continue
        if first not in decision.eligible or second not in decision.eligible:
            continue
        included_trials.append(trial)
        eligible_trials += 1
        probabilities = {
            cell: decision.joint_cell_probability(first, second, *cell)
            for cell in cell_order
        }
        if any(probability <= 0 for probability in probabilities.values()):
            continue
        overlap_trials += 1
        assignment_trials.append(trial)
        observed_cell = (first in decision.selected, second in decision.selected)
        observations[observed_cell].append(
            (trial.reward, probabilities[observed_cell])
        )
    reward_definition_hash = _validate_reward_definition(included_trials)
    (
        resolved_context,
        subset_size,
        exploration_probability,
        policy_id,
        policy_version,
    ) = _validate_estimand_stratum(
        included_trials, context_hash, assignment_trials
    )
    summaries = {cell: _weighted_summary(observations[cell]) for cell in cell_order}
    means = {cell: summaries[cell].mean for cell in cell_order}
    exact_means = {cell: summaries[cell].exact_mean for cell in cell_order}
    synergy = None
    if all(exact_means[cell] is not None for cell in cell_order):
        synergy = _finite_linear_contrast(
            tuple(exact_means[cell] for cell in cell_order),  # type: ignore[arg-type]
            (1, -1, -1, 1),
        )
    return PairSynergyEstimate(
        first=first,
        second=second,
        context_hash=resolved_context,
        subset_size=subset_size,
        exploration_probability=exploration_probability,
        reward_definition_hash=reward_definition_hash,
        policy_id=policy_id,
        policy_version=policy_version,
        synergy=synergy,
        cell_means=tuple((cell_names[cell], summaries[cell].mean) for cell in cell_order),
        cell_trials=tuple((cell_names[cell], summaries[cell].trials) for cell in cell_order),
        cell_effective_sample_sizes=tuple(
            (cell_names[cell], summaries[cell].effective_sample_size) for cell in cell_order
        ),
        eligible_trials=eligible_trials,
        overlap_trials=overlap_trials,
    )
