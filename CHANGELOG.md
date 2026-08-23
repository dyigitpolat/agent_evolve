# Changelog

## 0.5.0 — 2026-08-24

Two defects the end-to-end evaluation row found and the test suite could not:
the package could not sell its own measured result, and it could not search a
bounded number. Both were written down in `PACKAGING_TODO.md` §6 as known and
unfixed. Both are fixed here, and the second one moves the substrate under the
search — which is stated rather than absorbed.

### Changed — the distribution is named `agentevolve`, and why that is a fix

The metadata said `name = "agent_evolve"`, which PEP 503 normalizes to
`agent-evolve` — and that name is **already taken on PyPI** by an unrelated
third-party package. Every install line this README has ever suggested
(`pip install agent_evolve`) therefore resolved to **someone else's code**:
not a hypothetical, a supply-chain footgun, found by checking the index
before the first publish rather than after. The distribution is now
**`agentevolve`** (free, and the project's own name as it has always been
written); the import is `agent_evolve` exactly as before, so no user code
changes. `agent_evolve version` and the packaging tests read the new
distribution name, and the install hints in `bootstrap` name the new extra
spellings.

### Fixed — the mechanisms that measured could only be bought with the one that did not

`optimize()` built the completion seam *inside* the per-offspring chooser's own
branch. So `proposer="llm"` handed you the chooser whether or not you wanted it,
and `proposer="random"` left `complete=None`, which silently disabled every
authoring seam. The consequence is worth stating plainly: **the configuration
that produced this project's end-to-end results was not reachable from the
public API.** Not undocumented — unreachable.

The per-offspring chooser is the one mechanism here that has never earned its
price: **ten sealed null verdicts**, Θ(offspring) model calls rather than one,
and **61% of the six-arm ablation's whole ledger consumed for 0.94×** the speed
of doing nothing. It is now a separate parameter, `chooser`, defaulting to
`"off"` — the random control it never beat. `chooser="llm"` still buys it, and
asking for it on a run that makes no model call is refused by name rather than
quietly ignored.

What `proposer="llm"` purchases instead is the configuration the ablations
picked:

- **model-proposed initialization** — the six-arm ablation's strongest arm (the
  A4 seat): **11× fewer evaluations to target, better on 40 of 40 paired
  seeds**, for **one** call;
- **model-authored surrogate screening** — the sealed default, unchanged, still
  admitted only when it out-validates the shipped rules;
- at `budget >= 48`, a **crossed screen** (auto-sized `min(16, max(8, budget //
  6))`; the ablation screened at 15 evaluations of 96) read by a
  **model-weighted graded prior** — the ablation's guidance arm, the A5 seat,
  4.60×.

Below budget 48 the screen is skipped and the prior stays the rule form. That is
the seat's precondition rather than a rounded ratio: the prior only ever acts on
a screen's evidence, so the model prior is bought exactly when a screen will
run, and announcing a model prior beside `structure_budget=0` would be a promise
the run never cashes.

`prior` and `structure_budget` therefore default to an `"auto"` sentinel instead
of to literals, resolved after the seam is either built or comes back empty —
because what the sentinels should mean depends on what the run can actually do,
not on what the caller hoped for. Without a model call they resolve to `"rule"`
and `0`, which are the literal pre-sentinel defaults: the credential-free path
draws the same candidates in the same order and the byte-fossil stream does not
move. That was verified, not assumed. `tests/test_headline_defaults.py`.

### Fixed — a bounded number was not a search space

`policies.genetic._declared_domain` returned `()` for any field that was not a
finite enum or a bool, so `uniform_candidate` and `mutate` left the locus frozen
at whatever the seed happened to carry, and nothing said so. Measured on a
hyperparameter benchmark: **12 of 13 axes identical across 64 draws.** An entire
class of problems could be handed to this optimizer and get back a run that
varied almost nothing.

Bounded numeric loci now project onto a finite domain:

- **integers** — every value when the span is at most 64, otherwise 64 points
  evenly spread with both extremes kept. `exclusiveMinimum`/`exclusiveMaximum`
  tighten by one, which is what they mean on an integer axis.
- **floats** — a 16-point grid with a point on each inclusive endpoint. Sixteen
  because that is the resolution `from_pymoo` has quantized continuous decision
  variables onto since it shipped: one resolution rule for the package rather
  than two. An excluded endpoint is not a legal value, so it is not emitted.
- **`multipleOf`** restricts to multiples, so every value the projection emits
  is one the schema would actually accept.
- **`anyOf`/`oneOf`** are scanned for the first branch with such a reading, with
  the `null` branch skipped — `Optional[int]` with bounds is searchable, and no
  sampler writes `None` where the problem declared a range.
- **`const` is read as a one-value domain.** It was ignored, so a single-valued
  field — which pydantic renders as `{"const": ...}` rather than a one-element
  `enum` — read as undeclared to every reader in the package.
- **A per-field override**, `Field(json_schema_extra={"agent_evolve": {"grid":
  N}})`, clamped to `[2, 256]`: below two is a constant, and above 256 is not a
  domain an operator can draw from or a prompt can render.

`diagnose` names such an axis as `inline_threshold:64(projected)` rather than
`inline_threshold:?`, because "64 values" means something different when the
schema declared 64 values than when it declared an interval. A locus with no
finite reading — an open `str`, a one-sided bound — is still left alone, and
`diagnose` still names those.

**This changes search behaviour on any venue with a numeric locus.** The shipped
`build_flags` example now actually varies `inline_threshold`: at the same
40-evaluation budget its front grew from 6 rows to 11. The `knapsack` example is
no longer the honest-negative for undeclared domains — its selection locus
projects to 10 values and `diagnose` reports `undeclared domains: none`.

**Every sealed row measured to date was measured on the 0.4 substrate and
remains quotable as such**, from branch `release/v0.4-sweep`, which preserves
it. A row measured on 0.5.0 is a row measured on a different search space, and
the two are not interchangeable. `PACKAGING_TODO.md`'s 2026-08-18 deferral of
exactly this fix — deferred on the ICLR deadline, explicitly not on the merits —
is **superseded by the owner's 2026-08-19 direction** to develop heavily. The
byte-fossil test passes unmodified: its venue is enum-only, so nothing it pins
can move. `tests/test_numeric_domains.py`.

### Added

- **Authorship preset `"guided"`** = `{surrogate: "llm", initialization:
  "llm"}` — what `authorship="auto"` resolves to on a model run, named so a
  campaign can *state* its configuration instead of inheriting it. The two seams
  the six-arm ablation and the sealed luna-clear row measured as winners, and
  neither of the per-decision seams that did not.
- **`run --chooser {off,llm}`**, whose `--help` carries the same ten sealed null
  verdicts the default rests on, so nobody has to read this file to find out why
  it is off.
- **The `"auto"` sentinels are reachable from the CLI**: `--structure-budget`
  takes `auto` or an evaluation count, `--prior` takes `auto` beside the four
  named forms. The CLI parses the sentinel and does not resolve it — a second
  copy of the sizing rule there could drift from the one the library announces.
  Every resolution is announced through `on_progress`; offline both resolve to
  the pre-sentinel literals, so the credential-free stream stays byte-identical.

### Added — the revision channel: guidance that reads what the run measured

**Experimental, opt-in, and under measurement. Nothing below is a result.** It
is off by default and stays out of `authorship="auto"` until its first measured
row, because every default in this package comes from a measured row and this
seam's row — the W1 pilot — is in progress.

The end-to-end evaluation row's one decisive loss has a sealed diagnosis:
guidance is authored before the first evaluation and nothing refreshes it
against what the run measures, so on the one live physical simulator tested the
guided stack is the best arm at 20, 40 and 80 charges and the worst by 160. It
is not wasting budget — zero failed evaluations against uniform's 1.25% — it is
working from a stale picture.

- **`authorship.adaptation="llm"`, and the preset `"adaptive"`** (`guided` plus
  this one channel). On a declared cadence in charged evaluations, one call
  reads the domain card, the current per-field weights and a pure rendering of
  what the run measured, and replies with a revised weighted prior — the
  *classical breeding path's* prior, which initialization and mutation both draw
  through. This is deliberately not the sampler-re-authoring channel, which lost
  to its own shuffled-evidence control and does not ship.
- **Admitted or refused whole**, under the same taxonomy: undeclared names,
  invalid weights, in-field concentration above 8, or a zero on any value a
  rank-0 measured configuration holds. An admitted reply is then **step-damped**
  into the installed weights — `w = (1-a)*prev + a*proposal` on normalized
  per-field distributions, `adapt_damping=0.5` by default. Damping below 1 makes
  an introduced exclusion impossible, and a convex mix of two within-cap vectors
  stays within cap (the mediant inequality), so the worst case a wrong revision
  can reach is a bounded tilt inside declared domains, forever.
- **The revert rule now binds.** Its first live measurement found it impotent:
  across six revision-carrying runs on a three-objective simulator it admitted
  **23 revisions and reverted 0**, because the bet asked only that some
  post-event row be rank-0 in the pooled rows — and on three objectives almost
  every fresh point is non-dominated. One run's revisions locked it flat for 120
  charges while its one-factor-apart frozen twin kept improving. The bet is now
  the loop's own unwind semantics: a revision survives its window only if some
  row measured after the event **strictly dominates** a member of the pre-event
  front.
- **Evidence v2: occupancy where the front lives.** At the 40–90 rows a mid-run
  revision actually has, over a 24-field space, per-locus rank correlations are
  noise — the model re-tilted the same fields across a flat 120-charge window —
  while the sealed graded-prior seat's own working format was never shown to it.
  The bundle now carries per-value occupancy among the current non-dominated
  configurations against occupancy among everything measured, rendered by a
  pure, exactly-tested renderer that never sees an objective value.
- **Every event retains the bundle it was shown, verbatim** (`note`'s
  `evidence_text`), because the oracle instrument proved late checkpoints
  unreconstructible from cells: `state.evaluated` includes cache-served repeats
  the charge log cannot recover (2 of 4 digests reproduced on one flagship, 0 of
  4 on another). Each call also journals the sha256 of that evidence.
- **`adapt_gate_reads_view`** (default `False`, the product's stance). The live
  gate reads reality by design — no revision may zero a value some configuration
  *this run* measured onto the front, whatever the prompt showed. But a
  shuffled-evidence control whose prompt reads donor rows while its gate reads
  this run's front accrues refusals the arm it controls never meets (two of four
  revisions refused on the control alone, W1 pilot seed 20370103), so its
  refusal rate stops being comparable. A control arm is the only sane user of
  `True`, and the docstring says both halves.
- **Off is byte-identical to the pre-seam loop** — no call fires, no evidence is
  rendered, no counter moves, and the fossil holds unmodified.
  `tests/test_reguidance.py`, `tests/test_measurement_conditioned.py`.

### Added — the revision channel, v3: what two oracle studies forced

The W1/W2 pilot measured the channel's first form null-to-harmful, and two
live oracle-trace studies (R10) localized why. All three repairs are in, and
the re-pilot verdicts travel with them rather than being promised.

- **Silence keeps mass at the gate.** The admission check treated an
  unlisted value of a named field as a zero and refused the whole reply —
  and 10 of 11 live refusals were exactly that, on subset replies, including
  the oracle's own hindsight replies at the late checkpoints of both
  studies. The mixture already makes implicit exclusion impossible below
  damping 1, so the gate now refuses only an EXPLICIT zero on a front-held
  value; at `damping >= 1` — where that premise fails — silence reads as a
  zero again. Re-pilot: refusals collapsed to 1 in 60 events.
- **Immigrants are required-k, and shortfall is counted.** An optional
  clause drew zero proposals in ~30 calls from the live model and the oracle
  alike; the oracle named per-field independence as unable to carry the
  interaction structure three times. The clause now demands exactly k
  complete configurations, grounded in the occupancy table, and states how
  many configurations the run has already measured; a shorter reply keeps
  its admissible weights half and counts `immigrants_shortfall`. Rejections
  split by reason in the event note — which immediately paid for itself: on
  a 24-field schema every rejection is `shape`, not the dedup collision the
  pilot memo first inferred, and that correction is in the memo.
- **At most four tilted fields.** Live replies tilted or freed all 24 fields
  where the oracle tilts 2–8 grounded ones; the prompt now asks for the
  table's strongest cases only, and `tilt_breadth` is journaled per event
  (re-pilot median and max: exactly 4). No new refusal mode.
- Events self-identify with `mechanism: "v3"`; the evidence format is
  unchanged (`"v2"`).

### Added — four tuning knobs, each carrying its measured verdict

Every default still comes from a measured row; these ship as knobs with
their pilot reads stated, not as silent improvements.

- **High-budget sizing.** Above budget 384 — the largest budget any sealed
  row was measured at, now the named constant `_SEALED_BUDGET_CEILING` — the
  population grows one member per 32 charges (floor 12, ceiling 64) instead
  of capping at twelve. At or below 384 the literal old expression runs,
  re-verified at every integer budget in the range. The cap was a throttle:
  at budget 2000, six of six cells spent 969–1212 charges while the uniform
  comparator spent ~1800, and the matched-budget comparison was decided by
  spend, not guidance. Re-measured after the fix: 5W/1T/0L against uniform
  (median recall 0.833 vs 0.667) with the budget genuinely spendable.
- **`init_style="split"`** (default `"joint"`, byte-identical prompt pinned):
  one call, two labeled sub-asks — strongest bets and coverage — with
  per-half telemetry. Its pilot read is negative and stated: it did not
  remove the effort axis's pool-median reversal and read 0W/3T/5L against
  the joint ask at the endpoint on the one venue tried. It stays off.
- **`prior="llm-weighted-committed"`** (API-only; the cautious prompt is
  byte-identical-pinned): removes the leave-free caution that one model
  tier's hedging phenotype obeys most literally, asking instead for
  evidence-proportional commitment. Pilot read, both halves: it repaired
  that tier's endpoint 4W/1T/1L with three rescues to ≤0.01 — while
  excluding an optimal value at the artifact level on 7 of 8 seeds, the
  third measured case of artifact-level and endpoint-level readings
  disagreeing. The defaults are untouched; the one-factor decomposition is
  the measurement campaign's question.

### Added — the release tail

- **`agent_evolve check --json`.** `run` has had a machine-readable document
  since 0.3.0 and `check` printed prose only, so scripted use of the *verdict* —
  the one thing this package asks you to run before spending anything — meant
  parsing sentences. It emits one document on stdout carrying the arms and their
  budgets, each arm's outcome, the per-objective comparison, the winner and the
  provider usage; the prose moves to stderr rather than being dropped, so the
  price is still stated before the spend and `2>/dev/null` leaves exactly the
  document. `run --json`'s conventions throughout: a block nobody could populate
  serializes as `null` rather than being omitted, so `verdict: null` under
  `--baseline-only` reads as "no model ever ran" and not as a loss. The prose
  verdict and the document are two renderings of one computation.
  `tests/test_public_knobs.py`.
- **`agent_evolve init [PATH]`** writes the five-obligation `problem_def.py`
  template — the knapsack example's shape with the knapsack removed. It lands as
  a valid `Problem`, so `diagnose` can be pointed at it the same minute, with
  exactly one obligation refusing by name: a template that returned a plausible
  number from `evaluate` would let a run look like it worked. An existing file is
  never overwritten. `tests/test_init_scaffold.py`.
- **The terra and sol rungs are declared routes now, not half-declared ones.**
  `openai/gpt-5.6-terra` gains its execution profile (`max_output_tokens`
  128000; provenance: the OpenRouter catalog's
  `top_provider.max_completion_tokens`, fetched 2026-08-20, reads 128000 for
  all three tiers) — without it a live terra cell ran at the provider's silent
  ~65k default while luna and sol ran a declared 128k, and truncation selects
  against exactly the terra/high rung a capability ladder exists to measure.
  Both routes also enter
  `MODEL_PRICES_PER_MTOK` — terra `$1.00`/`$6.00`, sol `$5.00`/`$30.00` per
  million, in both spellings, from this project's own routing table — so a
  terra or sol run reports a derived `cost_usd` instead of `null`. Unknown
  staying unknown was honest and is no longer necessary.
  `tests/test_openrouter_model_execution_profile.py`,
  `tests/test_provider_cost_reporting.py`.

## 0.4.0 — unreleased (the release cut)

The cut that makes the package installable, swappable and honest about its own
scope. Four defects found by *running* what the docs claimed, which is the only
method that has ever worked here.

### Fixed — four things that were described but not exercised

- **The reference example did not run.** `agent_evolve run
  examples.knapsack.problem_def:problem` and `python examples/knapsack/run.py`
  both crashed deterministically, for every seed, with `ValueError: mask has 1
  bits but the candidate has 3 loci`. A locus count is a property of a
  *candidate*, not of a problem — a sequence field contributes one locus per
  element — and the loop read it once from `seeds[0]`. `crossover` is now
  defined on ragged parents (a locus the donor lacks is simply not inheritable)
  and the mask is fitted to the parent it is applied to. Fixed-length genomes
  are byte-identical, which is every measured row in the research record.
  `tests/test_ragged_genome_recombination.py`.
- **Every model call on the default route was rejected.** The completion seam
  posts to OpenRouter's REST API, whose model IDs carry no provider prefix, but
  the shipped default is `openrouter:openai/gpt-5.6-luna` — so each call
  returned `HTTP 400: not a valid model ID`, four times, and the caller fell
  back to the classical path. The fallback guarantee is why nothing crashed and
  also why nobody noticed: the run produced a result and the ledger said
  `calls: 0`. `wire_model()` strips only the `openrouter:` prefix; other vendor
  prefixes are left alone so a mistake fails where it was made.
  `tests/test_completion_wire_model.py`.
- **`proposer="auto"` stopped choosing the credential-free path on machines with
  no provider credential.** `credentials_present()` asked "does this look like a
  secret", which is the *redaction* question, and `VSCODE_GIT_IPC_AUTH_TOKEN`
  and `CLAUDE_CODE_MESSAGING_TOKEN` answered yes. Routing now requires a name a
  model provider actually documents (`PROVIDER_CREDENTIAL_VARS`), or the
  `AGENTEVOLVE_*` prefix as an explicit escape hatch. Redaction keeps the broad
  rule, because over-classifying is safe there and unsafe here.
  `tests/test_provider_credential_routing.py`.
- **`cost_usd` was null on runs whose price was in the table.** Cost is now
  derived from the provider's token counts and the package's published prices,
  with `reported_by` naming both halves. An unpriced route still reports
  unknown, never zero. `tests/test_provider_cost_reporting.py`.

### Added — the drop-in claim, as a test

- **`examples/pymoo_swap/`** is now two files that optimize the *same* pymoo
  problem object: `nsga2_baseline.py` and `agentevolve_swap.py`. They differ by
  six lines — one docstring, two imports, three lines of API contact — and the
  problem definition does not change at all.
  `tests/test_pymoo_swap_acceptance.py` recomputes that diff on every suite run
  and fails if it grows, asserts both files still build the same problem, and
  runs both arms. It deliberately does not assert which arm wins. The
  single-file `swap_demo.py` it replaces is removed.

### Changed — packaging and honesty

- **Version is single-sourced** from `agent_evolve.__version__` via `dynamic =
  ["version"]`; the metadata cannot drift from the attribute again.
- **The `llm` extra pins `pydantic-ai==1.107.1`, `pydantic==2.13.4`,
  `pydantic-core==2.46.4`** — the exact versions `boundary_codec` fails closed
  on. The old `>=1.0,<2` range resolved 1.107.5 on a clean install and 27
  boundary tests failed on first run. `tests/test_packaging_metadata.py` keeps
  the metadata and the code's constants equal.
- **New `pymoo` extra** for the swap example and its acceptance test.
- **PEP 639 licence metadata** (`license = "MIT"`, `project.license-files`), and
  `LICENSE.draft` is no longer shipped inside the wheel beside the real licence.
  The build is now warning-free.
- **`README.md` leads with the fallback guarantee**, states the venue-scoped
  claims the paper states, and quotes real terminal output rather than
  paraphrase. `README.oss.md` is removed: its scope table rested on a venue that
  is struck for venue validity, and a draft that contradicts the paper is worse
  than no draft. `docs/scope.md`'s two blocks on that venue are replaced by a
  non-hosting note, so the absence cannot read as an oversight.

## 0.3.0 — superseded by 0.4.0 before release

The release that turns a research artifact into something a stranger can
install and get value from.

### Added — the authorship substrate (2026-08-07)

The LLM's job moves from choosing among enumerated operations (measured
rule-matchable) to AUTHORING machinery that real evaluations arbitrate:

- **Wiring and telemetry honesty.** `proposer='llm'` on the genetic path now
  actually consults the model (the chooser was built and dropped — a live
  bug); every mechanism's counters reach `SearchResult.telemetry`; provider
  usage is measured from the completion seam's own journal; a fossil test
  pins the credential-free default stream byte-for-byte across all of this.
- **Weighted sampling priors** (`policies/weighted_prior.py`): the generic
  graded form of `DomainRestriction`, which is provably its 0/1 special case
  (uniform weights take the identical `r.choice` path). Rule comparator
  `statistical_weighted_prior` ships beside `llm_weighted_prior_proposer`.
- **Pooled structure screens** (`structure_pooled=True`): pure + spiked
  designs over a sequence field's shared vocabulary; attribution counts
  every (candidate, position) pair, so short screens measure value effects
  sequences could never cross.
- **Authored-code runtime** (`infrastructure/authored_runtime.py`): model-
  written source runs out of process under CPU/memory/wall limits with an
  AST import allowlist; every failure is a typed, counted outcome.
- **Virtual pre-screening** (`authorship='surrogate'`/`'surrogate-llm'`):
  build `pool_factor`× the affordable offspring, order them with a surrogate
  that must beat the train-mean baseline on EVERY objective on held-out data
  (re-validated per generation, best-passing wins — model-authored
  surrogates screen only when they out-validate the shipped additive/kNN
  rules), measure the exploration floor plus the top of the order. Virtual
  and real evaluations are ledgered separately.
- **Operator portfolio** (`authorship='operators'`/`'operators-llm'`/
  `'full'`): variation arms under UCB1 survival credit with the classical
  arm always in the run, model-authored `vary()` operators whose children
  pass a parental/declared-material check or fall back (counted), and a
  preregistered retirement rule.
- **Public knobs**: `optimize(structure_budget=, prior=, effort=, journal=,
  authorship=)`, CLI `--structure-budget --prior --effort --journal
  --authorship --json`. Genetic-only knobs refuse the authoring strategy by
  name; llm forms fall back to their rule comparators out loud.

### Added — the authored generator (2026-08-12)

The model writes the SAMPLER, not the samples
(`authorship='generation-llm'` / `'generative'`,
`policies/llm_generator.py`). One authoring call before any evaluation
produces `propose(archive, n, domains, seed)`, and that one fixed cost then
shapes every candidate the run draws — the amortization per-decision
guidance cannot have, whose leverage falls as 1/budget.

- **Mass generation is free by construction.** The generator is handed a
  template, the declared domains and the archive — never the problem, never
  the evaluation cache — so a pool of 5,000 costs zero evaluations and the
  budget cap still binds. The pool then goes through the existing screening
  path when a screen is on (its exploration floor now holds the generator's
  own first picks), or is taken from the top when it is not.
- **Every emitted candidate is validated value-by-value** against the
  declared domains and the template's shape, exactly as authored initial
  members are; rejects are counted and their slots fall back to
  schema-uniform draws, so a broken generator degrades to the
  credential-free sampler rather than to nothing.
- **A diversity/novelty guard** measures the batch as a set: duplicates
  within it and candidates already measured in this run are dropped and
  counted, so a collapsed sampler reads as `duplicates == emitted - 1` in
  the telemetry and in the per-generation history instead of hiding behind a
  flat curve.
- **It evolves from measured feedback.** The revision hook shows the
  generator its own source plus what the harness measured — acceptance,
  duplicate and archive-overlap rates, and how many of its candidates
  survived selection — under the identical authoring gate, fired only on a
  measured defect and capped (`generation_revisions=0` ablates it, which is
  what one-shot ladder cells need).
- `generation` and `operators` both construct a generation's candidates, so
  asking for both is refused by name rather than silently running one.

### Fixed — authored code was paying for the whole package on every batch

`AuthoredRuntime` launched its worker as `python -m
agent_evolve.infrastructure.authored_worker`, which imported all of
`agent_evolve` (pydantic included) in the child — for a worker that imports
nothing but `ast`, `json`, `sys` and `traceback`. Measured at **922 ms per
spawn against 21 ms** when launched by path: 45× on every screen refresh,
every authored operator generation and every generated pool. A B=1,000
generator run went from 78 s to 2.35 s.

`screen_offspring` counted dominators in O(n²) through the contract-level
`core.results.dominates`, which re-validates every objective on every call.
It now counts over DISTINCT objective vectors weighted by multiplicity
(same integers, O(k²)) on pre-oriented float tuples: a B=300 run screening
pools of 2,000 went from 25.6 s to 2.55 s.

`Screening.refresh` refitted and re-validated every builder on EVERY
measurement so far, which makes one refresh O(n) and a run O(n²) — unusable
exactly at the budgets an authored generator is for. It now reads the most
recent `max_training_rows` (default 1,024, public as
`AuthorshipConfig(screen_training_rows=)`), which is also the better
statistics for a distribution the search keeps moving. Every campaign run to
date (B ≤ 150) sees every row it always did. The screen re-arbitrates once
per generation, so at high budgets this window — not the pool — is what
decides whether the screen's cost is constant or grows with the run.

Measured together, B=10,000 on a trivial lookup problem (12 loci, 2
objectives), all charging exactly 10,000 evaluations: default loop 5.3 s;
authored generator at the default pool 21.5 s (39,952 candidates generated);
authored generator at 2,000 candidates per generation 52.9 s (1,998,000
generated); generator plus the rule screen 31.1 s at
`screen_training_rows=128`, against over 15 minutes unfinished at the 1,024
default — the screen's per-generation re-arbitration is what sizes a
high-budget run, and the window is the dial.

### Breaking

**Python 3.11 is now the minimum, raised from 3.10.**

This is a narrowing, so here is the reasoning rather than just the number.

`requires-python` said `>=3.10`. The shipped code did not run on 3.10:
`src/agent_evolve/infrastructure/exception_provenance.py` uses
`BaseExceptionGroup`, which is a 3.11 builtin. The package had been claiming
support it did not have, and nothing caught it because every machine it was
ever tested on was newer than the claim. A matrix CI run found it in minutes
the first time one was ever executed.

There were two ways to fix it. Adding the `exceptiongroup` backport would
have kept 3.10, at the cost of a new dependency and edits to code that the
research measurements were taken against — and this repository's single most
valuable property is that the tree which ships is the tree that was measured,
because that is what makes a release claim checkable at all. Raising the floor
costs two months of compatibility: Python 3.10 reaches end of life in October
2026. Trading a verified equivalence for that is not a good trade, and nobody
asked for it.

So the floor moved. If you are on 3.10, `pip` will decline to install rather
than installing something that breaks at the first exception — which is the
behaviour the metadata was supposed to guarantee all along.

A test now asserts that the declared floor covers every version-gated builtin
the source actually uses, so the claim and the code cannot diverge again
silently.

**`optimize()` takes one problem and five keyword options, not eighteen.**
`budget` — the number of artifacts you are willing to measure — replaces
`pop_size`, `generations` and `candidates_per_batch`, and is a hard cap rather
than a between-generation check. `proposer` replaces `harness` and also
accepts any registered harness id. The reflection ablation switches, `extra`,
`on_event` and `settings` are gone from the public surface.

**652 top-level exports became 11.** The rest resolve lazily through
`__getattr__`, so the research stack keeps importing exactly what it always
did; they are simply no longer advertised. Import drops from 233 modules to
32, and from 0.73 s to 0.24 s.

**Harness implementations are now checked.** `HarnessBase` declared itself
abstract and enforced nothing, which is how a proposer shipped with three
wrong signatures that only failed mid-run. Every operation is now verified for
existence and positional arity at class-definition time.

### Added

- **The five obligations as a typed contract** (`agent_evolve.contract`):
  `candidate_model`, `objectives`, `seeds()`, `validate()`, `materialize()`,
  `evaluate()`. Two of these had no public representation before. Problems
  written against the old two-method contract keep working unchanged.
- **`materialize()`**, on measured grounds rather than symmetry: the evaluation
  cache keyed on the configuration, so two configurations producing the same
  artifact were paid for twice. It now keys on artifact identity and reports
  what that saved.
- **`agent_evolve check`** — runs an uninformed sampler and a model against
  your problem at the same budget with the same evaluator, and says which won.
  `--baseline-only` needs no credentials.
- **A `random` proposer** that needs no provider, no network and no cost. It is
  both the credential-free path and the control arm.
- **`optimize(..., seal="journal.jsonl")`** — writes one chained,
  self-authenticating line per model call, holding the configuration that was
  emitted, the digest of the prompt that produced it, the digest of the schema
  it was drawn from, and the verdict `validate` returned. The run then replays
  from that file with no provider and no credential; a drifted prompt, an
  edited line, a broken chain, or a feasibility rule that moved since the
  recording all fail loudly rather than falling back to a live call.
- **`agent_evolve.proposal_mode`** — names the two ways a model can act on a
  problem and what each seals with. `generative`: the model emits candidates
  against `candidate_model`. `catalogue`: the engine enumerates parent-relative
  edits and the model selects one, which certifies feasibility by construction
  but leaves the model no way to author anything.
  `require_matched_support` refuses to compare two arms that draw from
  different spaces — the mistake is easy to make, because the wrong null still
  runs and still returns an ordinary-looking number.
- **A CLI**: `check`, `run`, `version`.
- **CI**, and `scripts/ci-local.sh` to run the same workflow file locally.
- **`docs/scope.md`** — where this helps and where it does not, from measured
  boundaries, including the claims we cannot make.
- **A `LICENSE` file.** MIT had been declared in metadata since 0.2.0 with no
  file present.
- `py.typed`, so the typing is visible to consumers.

### Changed

- **The default model is now `openrouter:openai/gpt-5.6-luna`** ($0.10/M input,
  $0.60/M output), replacing `openai:gpt-4o`. `check` prints the model and its
  price before spending, marked as a default when you did not choose it.
- **Dependencies are ranges, not exact pins.** `pydantic==2.13.4` in a library
  conflicts with most of the ecosystem.
- **`pydantic-ai` moved to an optional `[llm]` extra.** The package installs
  and runs with no provider at all.
- **`evaluations` counts artifacts actually measured.** Counting cache hits made
  an honoured budget look breached.
- **The test suite splits.** The default run is the shipped package, offline;
  the research suite is opt-in behind `-m research`.
- Environment inputs on the public path: three that affect behaviour
  (`AGENTEVOLVE_MODEL`, `AGENTEVOLVE_HARNESS`, `AGENTEVOLVE_TEMPERATURE`) plus
  two credential-safety controls.
- **Seeds now seed.** `optimize()` evaluated the caller's starting points and
  then sampled its first batch blind, so the one thing the caller supplied and
  paid to measure had no effect on what was proposed next. With valid seeds the
  first proposal is now bred from them, and the insight derived from them
  carries forward instead of restarting from nothing. Runs with no seeds are
  unchanged.

### Fixed

- **A failed measurement is now charged to the budget.** A candidate whose
  evaluation raised was recorded as a failure and not counted, so a run
  continued until it had accumulated `budget` *successes*, however many
  artifacts that took, and then reported the budget as honoured. On an
  evaluator whose failure mode is a timeout the uncharged evaluations are the
  most expensive ones in the run.
- **The generic prompt wording stopped describing one problem's structure.**
  `DefaultDirectives` instructed every model on every problem to make
  "per-dimension products EXACTLY correct" — the constraint of the single search
  space the backbone was first written against. Problems with no dimensions and
  no products were still told to verify them.
- **The library no longer searches upward for a `.env`.** `load_dotenv()` with
  no argument walks up the directory tree until it finds any `.env`, so a run
  inside a monorepo silently adopted an unrelated project's credentials — and a
  run launched specifically to prove it made no provider call was handed the key
  back by the very next line. Nothing is loaded now unless you name it, and
  `AGENTEVOLVE_SCRUBBED` names variables that must stay unset.
- **The test suite collects without the research corpus.** It used to abort
  entirely on absolute paths into the research tree, so someone who cloned this
  repository could run no test at all.
- **Corpus paths resolve across the 2026-07-28 archive split.** Chosen as a
  fallback rather than a repointed root, because the files are read behind
  frozen content hashes: an archived copy either is the sealed bytes or fails
  loudly on the hash.
- **Asking for the model proposer without the `[llm]` extra** now fails up front
  naming the install command, instead of raising `ModuleNotFoundError` from
  inside the first call.
- A failed optional integration is recorded and re-reported, rather than
  surfacing only as `Unknown harness 'pydantic_ai'. Registered: []`.

### Repository

- **One tree that is both the shipped artifact and the measured one.** 34 of the
  55 modules the paper's scripts import did not exist here, so no release claim
  was checkable against the code that shipped. All 55 now resolve.
