# Changelog

## 0.3.0 — unreleased

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
