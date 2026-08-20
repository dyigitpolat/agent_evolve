# The measured record, vendored

> **The authoritative record is the paper's research archive** — the sealed
> preregistrations, the spend ledgers, the numbers bundles — and it is **not
> distributed with this package**. Rows in it assert and re-verify their own
> absolute paths, so a copy of them is a copy and never the record. The wave
> names in the tables below are that archive's addresses.
>
> **This file quotes [`README.md`](../README.md) verbatim**, and the two places
> it does not are marked as quoting [`CHANGELOG.md`](../CHANGELOG.md). It exists
> so that a standalone release carries its own scope with it instead of pointing
> at a tree the reader does not have. Nothing here is restated from memory and
> nothing here is a number the README does not already print. **If this file and
> the README ever disagree, the README is the copy of record and this one is
> stale.**

Every claim below is **venue-scoped**: it is a statement about the named domain,
budget, estimand and seed count, and nothing wider. There is no all-domains
sentence here because there is none in the README and none in the paper.

**Of ten measured venues, four host a general claim.** The reasons the other six
host nothing — a venue that cannot separate any two optimizers, a sweep struck
for venue validity, an axis that read null — are published rather than omitted.

## Which substrate these rows were measured on

Quoting `CHANGELOG.md`, 0.5.0:

> **Every sealed row measured to date was measured on the 0.4 substrate and
> remains quotable as such**, from branch `release/v0.4-sweep`, which preserves
> it. A row measured on 0.5.0 is a row measured on a different search space, and
> the two are not interchangeable.

## What fires

- **The dominance row.** On a real RTL-to-GDS physical-design flow: **1.886×**
  fewer charged evaluations than NSGA-II to a sealed target (median charges,
  `N = 48` in four blocks, `p = 0.0013`), and **1.959×** fewer than our own
  credential-free loop, which does not separate from NSGA-II at all. A second
  dominance row stands on accelerator architecture search.
  *And it still misses the bar that venue's own specialists set: 54.0 against
  43.5.* This is a cheap generic loop priced accordingly, not a domain
  specialist.
- **Authoring quality scales on two axes.** Model class and test-time reasoning
  effort, by the same amount on one venue, stack, estimand and scorer; composed
  on one grid the two are super-additive. At **zero** effort the capability jump
  is not measurable at all.
- **The buy on the effort axis is `none` → `high`**: median exact-front recall
  `0.426471 → 0.588235`, step **+0.161765** at 2.75× its own bar, `N = 48` per
  arm, `p = 5×10⁻⁶`, achieved power 1.000.

## The authored objects, and what arbitration did to each

The floor is the classical baseline and the ceiling is measured, not promised.
These are the rows where the guarantee was watched firing.

| the authored object | what arbitration did to it | evidence |
| --- | --- | --- |
| model-written **surrogate** for pre-screening | Admitted, then beaten. Across the two columns with plentiful cross-validation evidence it **cleared the gate on 46 refreshes and won 0 of them**; the cheap 100-mapping fidelity took **all 313** installs. In the richest column alone it cleared **41 of 188** refreshes and won none. The guided and rule arms came out **byte-identical on 36/36 paired seeds at both budgets** — a contribution of literally zero. | wave L, `aug15_llm_value_tlv2.md` |
| measurement-conditioned **re-authoring channel** | Fired **447 times** and came out **below its own shuffled-evidence control**. Does not ship. | wave L, `aug15_measurement_conditioned.md` |
| graded **locus prior** | Neither beat the stack without it nor separated from its own control. Ships as documented venue knowledge, not as run-time reasoning. | wave N, `aug16_p1b_margin.md` |
| model **choosing among evolutionary operators** | Eight mechanisms, eight sealed verdicts: guided runs sit at the *unguided* median. The published result is the negative one. | the paper, §ledger |

An artifact that clears a gate 46 times and never wins is not being blocked, it
is being beaten — and *the fallback is the arm that actually ran*.

## What the shipped configuration is, seat by seat

What `proposer="llm"` buys is the configuration the ablations picked:

- **Model-proposed initialization** — the six-arm ablation's strongest arm, at
  11× fewer evaluations to target, better on **40 of 40** paired seeds, for
  **one** call.
- **Model-authored surrogate screening** — the shipped default, unchanged, and
  still admitted only when it out-validates the rules, which is the arbitration
  the table above describes rather than a win claim.
- At `budget >= 48`, **a crossed screen sized from the budget, read by a
  model-weighted graded prior**: the ablation's guidance arm, 4.60×. Below 48
  the screen is skipped and the prior stays the rule form, because the prior
  only ever acts on a screen's evidence.

**What it does not buy is the per-offspring chooser**, which is `chooser="off"`
by default: a model call per offspring returned **ten sealed null verdicts** and
consumed 61% of that ablation's whole ledger for **0.94×** the speed of doing
nothing.

Quoting `CHANGELOG.md`, 0.5.0, for the seat names those two winners carry in the
research record: model-proposed initialization is **the A4 seat**, and the
crossed screen read by the model-weighted graded prior is **the A5 seat, 4.60×**
(auto-sized `min(16, max(8, budget // 6))`; the ablation screened at 15
evaluations of 96).

## What is not established, stated as plainly as what is

- **Saturation above `high` is UNRESOLVED, not established.** `high` → `xhigh`
  did not separate: +0.0230, `p = 0.245`, achieved power **0.073**. `max` is
  descriptive only (`n = 20`, never laddered, in no test).
- **`xhigh` is a request, not a dose.** At the same label one model reasoned
  32,156 tokens where another reasoned 53,698. The label names what you asked
  for, not what you got, and the two are not the same variable.
- **The step from mechanism to endpoint is attenuated 35.2×.** The dose that
  moves an authored artifact by **+0.161765** median recall moves the search
  endpoint by **+0.004596** — real, overwhelmingly significant (`p = 5×10⁻⁶`,
  `N = 48`), non-inferior and superior, and *small*. We price that step rather
  than assuming it.
- **Absolute endpoint levels are not comparable across configurations**, and we
  do not quote them across them.
- **The ladders rest on one venue family.**
- **Model guidance of *operator choice* did not work.** Eight mechanisms, eight
  sealed verdicts, guided at the unguided median. Since 0.5.0 it is **off by
  default** and reachable only by asking for it by name (`chooser="llm"`): a
  negative result belongs in the product as a switch nobody is billed for, not
  as a default.
- **Two of our own published sentences are withdrawn**, with the defect in each
  stated in the paper's discussion. We do not re-assert them from the archive.

## Regime guidance is a different file

Which *shape* of problem this tool suits, what it costs in wall-clock, and where
it loses outright are in [`scope.md`](scope.md), drawn from earlier campaigns.
The claim of record is the dominance row above, not the `5.16×` there.

And the fastest answer about *your* problem is not in any of these files:

```bash
agent_evolve diagnose yourpkg.problem:problem --budget 40   # no model, no credential
agent_evolve check    yourpkg.problem:problem --budget 40
```
