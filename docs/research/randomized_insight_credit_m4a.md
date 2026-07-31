# M4a randomized insight-subset retrieval and causal credit

M4a is an isolated, provider-free policy foundation. It implements the smallest
statistically defensible version of evidence-scored memory before wiring memory
into the active workflow.

## The attribution problem

Suppose insights `A,B,C,D` are eligible and one generation uses `B,D`. A strong
generation is evidence about that *selected subset*. By itself it is not evidence
that `A,C` would have performed worse, so rewarding `B,D` and penalizing `A,C`
from this one observation fabricates a counterfactual.

The policy instead mixes:

- deterministic top-k exploitation under the current scores; and
- a uniformly random k-subset with exact probability `epsilon`.

For `n` eligible insights and fixed subset size `k`, the conditional inclusion
probability for insight `i` is

```text
pi_i = epsilon * k/n + (1-epsilon) * I[i is in the top-k subset].
```

The exact probability of every selected subset and all four inclusion cells for
any insight pair are recorded as `fractions.Fraction` values. This supports
stabilized inverse-propensity selected-versus-unselected contrasts and a pairwise
interaction contrast:

```text
tau_i = E[reward | i selected] - E[reward | i not selected]
synergy_ij = E11 - E10 - E01 + E00.
```

With fixed-size subsets, `tau_i` is an effect of including `i` *in place of the
policy's randomized alternatives*. It is not the effect of adding an insight to
an otherwise unchanged prompt. Higher-order interactions and a different subset
size require separate estimands.

## Research-unit semantics

Randomization happens once per operator invocation. If that invocation produces
several children from the same insight subset, their predeclared aggregate reward
forms one `InsightTrial`. Copying the subset/reward into one row per child would
inflate the apparent sample size; duplicate invocation or candidate membership is
therefore rejected. An invocation that produces no child is still a randomized
unit and must be retained with an empty candidate tuple and the predeclared
failure/no-yield reward. Dropping it would condition credit on survival.

Each trial binds:

- one `OperatorInvocationId` credit unit;
- every resulting `CandidateId` exactly once, or an empty tuple for a zero-child
  invocation;
- a SHA-256 identity for the reward definition;
- the complete selection decision and conditional law; and
- one finite canonical-float reward.

An estimate cannot mix reward-definition hashes. Context filtering uses an opaque
SHA-256 identity; the future workflow must make that identity bind the relevant
benchmark/task, parent cohort, operator, phase, evidence schema, model/prompt
condition, and reward horizon. A test outcome must never update memory used for
that same held-out test protocol.

## Implemented boundary

- `domain.insight.InsightRef` binds a logical typed `InsightId` and positive
  version. Revised insight text cannot inherit evidence silently.
- `EpsilonGreedySubsetSelector` requires an exact rational exploration rate,
  stable score/ID/version tie breaking, exact eligibility-score agreement, an
  exact-uniform integer `randrange` branch draw, and a random source whose
  `sample` contract is uniform without replacement. It never converts epsilon
  to a float.
- `InsightSelectionDecision` revalidates canonical ordering, scores, top-k
  membership, mode, subset probability, and pair-cell algebra. Forged probability
  records are rejected.
- Deterministically included/excluded insights have no overlap and contribute no
  causal contrast. Selecting every eligible insight is explicitly
  non-identifiable.
- `estimate_marginal_effect` reports selected/control weighted means, raw trial
  counts, effective sample sizes, overlap, and a contrast when both cells exist.
- `estimate_pair_synergy` requires positive probability for all four pair cells;
  impossible fixed-k interactions fail closed.
- Both estimators reject mixed context hashes and mixed subset sizes. A caller
  may explicitly filter one context, but cannot silently average different
  policy-relative estimands.
- Estimates also reject mixed exploration probabilities and expose the exact
  epsilon, reward-definition hash, and policy ID/version in their result. Inverse propensities stay
  rational and admitted binary-float rewards are converted exactly to rationals
  for the complete Hájek numerator/denominator. Only the final mean, effective
  sample size, or contrast is converted back to float; true underflow, overflow,
  or any other nonfinite result fails closed.

M4a deliberately exposes no standard error. A prior weighted-residual formula
was removed during independent review because it was neither a valid Hajek/IPW
variance estimator nor honest confirmatory inference. Scientific runs need
randomization-unit bootstrap or randomization inference, context/task clustering,
sequential/adaptive-design handling, multiplicity control, and a frozen analysis
plan before uncertainty is added back.

## Tests and current hold

Focused tests cover exact rational exploration sampling, exploitation/exploration probability algebra,
pair-cell probabilities, deterministic non-overlap, stable ties, invalid and
forged records, context filtering, reward-identity separation, batch-level credit,
pseudoreplication rejection, zero-child units, estimand-stratum separation, known
marginal effects, known pair interaction, and structurally impossible interaction
cells. The exact post-audit test count is recorded after re-audit rather than
preserving the superseded 242-test checkpoint as a current result.

M4a is not connected to prompts, an insight database, the event vocabulary, or
the legacy generation loop. It made no evaluator, provider, model, or network
call. Before integration it still needs:

1. an independently reviewed event schema for eligibility, selection, trial
   reward, and credit updates, with rational propensities encoded canonically;
2. a versioned memory repository and immutable insight-body artifacts;
3. a frozen reward definition (parent-relative quality/HV contribution,
   validity, cost, and horizon) plus delayed-outcome handling;
4. shrinkage/uncertainty-aware scoring and an explicit cold-start prior;
5. replay and crash-resume equivalence; and
6. synthetic nonstationarity, interaction, propensity, and calibration kill tests.

Until those gates pass, no live memory update or paid model pilot is authorized.

## Independent freeze

The final read-only audit gives GO to this isolated policy at implementation hash
`69aed83eb925a1fb1da9c23e238767a68f778c92ca591a6b7261771fd65cfe26`
and focused-test hash
`7135389ce2b06d70c98c58b73bf069e038fba95ef7b8458fad081e64318850b0`.
It passed 28/28 focused tests, 262/262 full offline tests, 135/135 exhaustive
assignment-law cases, and an independent extreme pair-cell numeric check.

This GO does not cover workflow integration. Exact rational arithmetic must gain
bit-length/trial-count limits (or audited propensity grouping) before it accepts
durable replay data. Events, reward definitions, inference, RNG state, crash
idempotence, and all provider/model execution remain held.
