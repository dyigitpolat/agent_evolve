# Where this helps, and where it does not

Every number below was measured, most of them while trying to prove the
opposite. They are published because knowing which problems this tool suits is
worth more to you than a claim that it suits all of them, and because the
fastest way to find out about *your* problem is one command:

```bash
agent_evolve check yourpkg.problem:problem --budget 40
```

That runs an uninformed sampler and a model against the same problem, the same
budget and the same evaluator, and tells you which won.

## The short version

| Your problem | Expect |
| --- | --- |
| Categorical or combinatorial, awkward constraints, tiny budget | The strongest case. Up to **5.16x** fewer evaluations than the best classical arm. |
| Expensive evaluator (minutes per candidate) | Worth it. Break-even is around **290 s per evaluation**. |
| Smooth and continuous | Marginal. **1.28x**, and uninformed sampling matches it **31%** of the time. |
| Cheap evaluator (seconds or less) | **Do not use this.** You will pay more in model latency than you save. |
| You need the answer fast in wall-clock | Classical optimizers were **8-9x faster** in head-to-head tests. |

## Where it is strongest

**Categorical spaces with semantic structure, at budgets too small for volume
to substitute.** On an accelerator-mapping problem -- 20 closed categorical
genes, about 7x10^11 configurations, 38 evaluations, 22.65% of draws
statically infeasible -- the best classical arm needed **5.16x more
evaluations** to reach the same endpoint. Uniform random search **never**
reached it on any of six seeds within 606-629 evaluations. A bootstrap put its
median at **1,665 evaluations (43.8x)**, with P(reach within 38) = **0.000**
over 500 trajectories.

This is the regime the tool is for: the flag and mapping selection problems
where blind operators are worst, the encoding carries meaning a reader can
exploit, and each evaluation is expensive enough that you only get a few dozen.

**When your evaluator is expensive.** The model's per-evaluation overhead is
roughly 75 s. Break-even against a classical optimizer sits near **290 s per
evaluation** (per-seed range 207-419 s). Above that, the model earns its
latency; below it, it cannot.

## Where it does not help

**Smooth continuous problems.** On a thermal-design problem the margin was
**1.28x**, and uninformed sampling matched the endpoint within 38 evaluations
**31%** of the time. This is natural Bayesian-optimization territory and you
should use Bayesian optimization.

**Wall-clock, in almost every case.** Measured head-to-head on thermal design:
NSGA-II was **9.1x faster**, uniform random **8.2x faster**, and a
domain-specialized optimizer matched our quality in **one fifth** the
wall-clock. At equal wall-clock, NSGA-II had reached 0.8975 over 455
evaluations against our 38-evaluation endpoint.

A faster model does not fix this. Gemini Flash is 3.2x faster per call and
moves break-even only from 290 s to 226 s, with a floor of 199.5 s at zero
provider latency, because **68.8% of the gap is host-side orchestration**, not
the provider. Nor does a more expensive evaluator: at 25.75x the evaluator
cost, the classical arms reached our endpoint in 13-16 evaluations, which makes
break-even undefined rather than favourable.

**Where it loses outright.** On logic synthesis (BOiLS/ABC), uniform random
search reached our endpoint in **0.55x** our evaluations -- it beat us -- and
the official BOiLS implementation beat us on 3 of 4 seeds.

**Where no optimizer can be told from another.** On the BOiLS `multiplier`
circuit, a single median random draw already accounts for roughly **80%** of
the hypervolume a full 38-evaluation run produces, and the entire model-guided
phase past the initial design is worth about **+1.03%**. That benchmark cannot
separate any two optimizers, and neither can any benchmark like it. This is the
single most useful thing we learned, and it is why `agent_evolve check` exists:
your problem may be one of these, and half a minute will tell you.

## Cost

About **$2.10-2.30** per 38-evaluation campaign on a mid-tier model. The
`random` proposer costs nothing, needs no credential, and touches no network.

## The check that cannot fail

Most of the corrections above came from one mistake, made five times. It
deserves a name, because naming it is the only defence that has worked.

**The pattern: a check that looks like a guarantee and cannot come out any
other way.** Not a wrong check -- a check on something *adjacent* to the claim,
which therefore passes whether or not the claim holds.

Four of the five are in the research record: a `"provider_calls": 0` written as
a literal in a receipt, which evidences nothing because it cannot be anything
else; a comparison against a constant that the constant guaranteed; a pooled
statistic whose denominator dropped the declines, so abstention scored as
success; and an aggregate check whose components cancelled, reproducing a known
total while both halves were wrong.

The fifth is the clearest, and it is in this repository. A guard was written to
skip tests when the research corpus is absent. It checked that the corpus
*directory* existed. Then the corpus was reorganised -- the files moved intact
into `archive/` -- and the directory still existed, so the guard reported the
corpus present while every file it protected was unreadable. Tests failed at
collection again, which is precisely what the guard had been written to
prevent. It was written by someone who had just spent a week removing the other
four.

That is the useful part. This is not carelessness, and knowing about it does
not prevent it. It is the shape a check naturally takes when you write it
quickly: you verify the thing that is easy to reach -- the directory, the
constant, the total -- instead of the thing you actually depend on. The
question that catches it is cheap and worth making a habit:

> If the thing I am checking for were broken right now, would this check fail?

For the corpus guard the answer was no, and one line of thought would have
found it. The guard now resolves paths by calling the same helper the loaders
call, so it cannot answer differently from the code it is protecting.

**And it is not rare.** The count passed eight when a CI workflow was run for
the first time and immediately found four more: metadata claiming support for
a Python version the code could not run on, a hand-written list of the failures
someone happened to see in their own environment, a test asserting which
exception the interpreter raises rather than what the code guarantees, and
three existence guards that resolved a path differently from the read two lines
below them. All four came from a single afternoon of exercising something that
had never been run.

That is the lesson, and it is not that the codebase is bad. This is simply what
unexercised checks look like. A check that has never been made to fail is a
check nobody has evidence about, and the only thing that reliably finds these is
running the thing you claimed works -- on the version you claimed, in the
environment you claimed, from the artifact you actually ship.

This is also why `agent_evolve check` exists and why the `random` proposer
ships as a first-class arm: an optimizer compared only against itself is the
same pathology at experiment scale.

## What we cannot claim

Three limits on the numbers above, stated because they bound what you should
conclude:

- The 5.16x rests on a **single sealed seed** compared against six independent
  classical runs -- best-versus-typical, not like-for-like -- and that campaign
  is **not currently relaunchable from its own receipts**.
- That comparison has **no external comparator**. Every arm in it is one we ran
  ourselves.
- Against a domain-specialized optimizer the honest word is *approaches*, never
  *beats*. On a corrected panel over a genuinely contested window we retrieved
  0.608 of the comparator and won one case in four.

Separately, the margin on the mapping result belongs substantially to the
numerical acquisition engine inside the harness rather than to the model: the
evaluated set of that campaign was seeds plus qLogNEHVI reproducing 99.7% of
the gain. Both proposal sources are kept because they contribute differently --
the model supplies hit *rate*, the engine supplies the rare large hit that
carries a heavy-tailed total -- but if you are here for the model specifically,
that distinction is the one to keep in mind.
