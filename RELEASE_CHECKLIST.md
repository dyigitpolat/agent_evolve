# RELEASE_CHECKLIST — 0.5.0

**Everything checked below was verified by running it, not by reading the code,
and each item names the test or the transcript that says so.** The unchecked
items at the bottom are the ones this repository cannot close by itself: they
need the project owner. That is the whole purpose of the file — to make the
remaining distance a short, named list rather than a feeling.

Two standing facts about scope, so nothing here reads as more than it is:

- **The release-tail changes described here are in the working tree.**
  Committing, tagging and pushing are owner actions and are listed as residue.
- **`PACKAGING_TODO.md` remains the reasoning**; this file is the state. Where
  an item ships as a documented limitation rather than a fix, it is under
  "Shipping as a stated limitation" and not quietly under a tick.

## The package

- [x] **Version is single-sourced** from `agent_evolve.__version__` via
      `dynamic = ["version"]`; the metadata cannot drift from the attribute.
      `tests/test_packaging_metadata.py::test_version_is_single_sourced_from_the_code`
      and `::test_installed_distribution_version_matches_the_attribute`.
- [x] **The build is warning-free** and produces both artifacts.
      `python -I -m build` → `agent_evolve-0.5.0-py3-none-any.whl` (2.27 MB) and
      `agent_evolve-0.5.0.tar.gz` (3.14 MB), zero warnings on stdout or stderr.
      The `-I` is not decoration: a stale `build/` artifact directory sits in the
      repository root and shadows the `build` *package* on `sys.path`, so a plain
      `python -m build` from the root fails with "'build' is a package and cannot
      be directly executed". Run it isolated, or from anywhere else.
- [x] **sdist hygiene**, checked by listing the archive rather than by
      intention: no `abc.history`, no `.probe_launch/`, no `LICENSE.draft`, no
      `dist/`, no virtualenv. 782 entries — `src/`, `tests/`, and at the root
      `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `setup.cfg`, and
      nothing else. (`CHANGELOG.md`, `docs/` and this file are repository
      documents and do not travel in the sdist; the GitHub tree is where they
      live, which is also where `docs/measurements.md` is meant to be read.)
- [x] **A licence file is actually shipped**, PEP 639 (`license = "MIT"`,
      `project.license-files = ["LICENSE"]`), and `LICENSE.draft` — whose first
      line read "DRAFT — NOT IN EFFECT" — is gone rather than merely unlisted.
      The CI `packaging` job asserts `licenses/LICENSE` and `py.typed` are
      inside the wheel.
- [x] **The `llm` extra pins exactly what the boundary codec supports**, and a
      test keeps the metadata and the code's constants equal.
      `tests/test_packaging_metadata.py::test_the_llm_extra_pins_exactly_what_the_boundary_codec_supports`.
- [x] **The `pymoo` extra exists** because the swap example needs it.
      `tests/test_packaging_metadata.py::test_the_pymoo_extra_exists_because_the_swap_example_needs_it`.
- [x] **Dependency floors are verified, not asserted** — a fresh venv at the
      declared floors runs the CLI end to end (2026-08-18 transcript).
- [x] **The console entry point names a real callable.**
      `tests/test_packaging_metadata.py::test_the_console_entry_point_names_a_real_callable`.

## The command line

- [x] **`run --seal PATH` and `run --strategy {auto,genetic,authoring}`** reach
      the API knobs of the same name; the genetic path refuses `seal` by name
      rather than writing nothing. `tests/test_optimize_strategy.py`.
- [x] **`run --json`** prints one machine-readable document carrying a derived
      `cost_usd` with its provenance. `tests/test_public_knobs.py`,
      `tests/test_provider_cost_reporting.py`.
- [x] **`check --json`** prints the verdict as one document — arms, budgets,
      per-arm outcomes, the per-objective comparison, the winner,
      `provider_usage` — with the prose moved to stderr, so `2>/dev/null`
      leaves exactly the document and the price is still stated before the
      spend. The prose verdict and the document are two renderings of one
      computation. `tests/test_public_knobs.py`, four tests, run against
      `examples/knapsack` in its own process.
- [x] **`agent_evolve init [PATH]`** writes the five-obligation
      `problem_def.py`, lands as a valid `Problem` that `diagnose` can be
      pointed at immediately, leaves `evaluate` refusing by name, and never
      overwrites an existing file. `tests/test_init_scaffold.py`, seven tests.
- [x] **`check --repeats` says what it repeats.** Its `--help` now states that
      the count is baseline runs and that the model arm is one run at seed 0,
      which was a silent asymmetry in every earlier cut.
- [x] **The mechanisms are reachable by name**: `--chooser {off,llm}`,
      `--structure-budget auto|N`, `--prior auto|…`, `--authorship` enumerated
      from the preset table rather than from a repeated list.
      `tests/test_public_knobs.py`, `tests/test_surrogate_screen.py`.

## The mechanisms the documentation describes

- [x] **`proposer="llm"` resolves to the measured stack** — model-proposed
      initialization, the authored surrogate, and at `budget >= 48` the crossed
      screen read by the model-weighted graded prior — while the per-offspring
      chooser moves to its own parameter, defaulting to `"off"`. Offline the
      `"auto"` sentinels resolve to the pre-sentinel literals and the
      credential-free stream is byte-identical. `tests/test_headline_defaults.py`.
- [x] **Bounded numeric loci are searchable** (integers enumerated or projected
      onto 64 points, doubly bounded floats onto 16, `multipleOf` honoured,
      exclusive bounds excluded, `const` read as a one-value domain), and
      `diagnose` marks a projection as such. `tests/test_numeric_domains.py`.
- [x] **The revision channel is opt-in and off by default**, and off is
      byte-identical to the pre-seam loop. Commits `adf1be0` (mechanism),
      `b752796` (docstring correction), `f2d02c7` (the revert rule now binds on
      strict dominance), `3485b5a` (`gate_reads_view`), `cbed1c5` (evidence v2
      + retained bundles). `tests/test_reguidance.py`,
      `tests/test_measurement_conditioned.py`.
- [x] **The terra and sol rungs are fully declared routes** — terra's execution
      profile (`max_output_tokens` 128000, provenance in the commit message and
      in the code) and both routes' prices in `MODEL_PRICES_PER_MTOK`, in both
      spellings, so a terra or sol run reports a derived cost instead of
      `null`. `tests/test_openrouter_model_execution_profile.py`,
      `tests/test_provider_cost_reporting.py`.

## The documentation

- [x] **`README.md` describes what ships, truthfully**: the chooser's ten sealed
      null verdicts beside the switch that still buys it, the `guided` preset
      named rather than merely resolved to, and adaptive revision documented as
      experimental, opt-in and **under measurement**, with no result claimed and
      no default changed.
- [x] **`docs/measurements.md`** — the vendored scope snapshot, so a standalone
      release carries its own numbers instead of pointing at a tree the reader
      does not have. It quotes `README.md` verbatim and says at the top that the
      archive is the authoritative record and the README the copy of record.
- [x] **`CHANGELOG.md` 0.5.0** carries the revision channel, `gate_reads_view`,
      evidence v2 and the retained bundles, the terra profile and prices,
      `check --json` and `init`.
- [x] **The README's licence line matches the settled decision** (owner,
      2026-08-18) instead of still asking for it.

## Verified by running, this cut

- [x] **Full offline suite green** on the development interpreter.
      `.venv/bin/python -m pytest tests/ -p no:cacheprovider --tb=short`
      → **3192 passed, 1 skipped in 172.43 s** (Python 3.12.3, research corpus
      present in this clone, so more modules collect than in a bare checkout;
      the `research`-marked tests stay deselected by `addopts` either way).
- [x] **Build artifacts**, listed above, zero warnings.
- [ ] **`scripts/ci-local.sh` is not green**, and the three failures are all
      pre-existing at `cbed1c5`. Verdict verbatim, and the triage, below. None
      of them is closed by this cut, and none of them is caused by it.

### `scripts/ci-local.sh` — the verdict, verbatim

```
[CI/packaging                                 ] 🏁  Job succeeded
[CI/a stranger can install it and get a result]   | the failure did not name the fix
[CI/a stranger can install it and get a result]   ❌  Failure - Main Asking for a model without the extra says how to fix it [789.648076ms]
[CI/a stranger can install it and get a result] 🏁  Job failed
[CI/tests (py3.11)-1                          ]   | FAILED tests/test_provider_credential_routing.py::test_credentials_present_is_true_for_a_real_provider_key - assert False is True
[CI/tests (py3.11)-1                          ]   ❌  Failure - Main Test [3m12.612076642s]
[CI/tests (py3.11)-1                          ] 🏁  Job failed
[CI/tests (py3.13)-2                          ]   | FAILED tests/test_multi_option_evolution.py::test_full_multi_option_evolution_runs_through_reflection - assert [(0, 0, 0), (...0, 5, 0), ...] == [(0, 0, 0), (...2, 0, 1), ...]
[CI/tests (py3.13)-2                          ]   | FAILED tests/test_provider_credential_routing.py::test_credentials_present_is_true_for_a_real_provider_key - assert False is True
[CI/tests (py3.13)-2                          ]   ❌  Failure - Main Test [3m16.682233053s]
[CI/tests (py3.13)-2                          ] 🏁  Job failed
```

### The triage — each one reproduced on pristine `cbed1c5`

**1. The stranger job is the serious one, and it is a product defect, not a CI
defect.** `agent_evolve run PROBLEM --proposer llm`, on a wheel installed
without the `[llm]` extra and with no credential, **runs to completion on the
classical path and says nothing about either**. Reproduced twice: from the wheel
built above in a clean 3.12 venv, and from a pristine `git archive cbed1c5` tree
with none of this cut's changes on it. Both printed the model line and its price,
ran the loop to budget exhaustion, and exited 0.

The mechanism is visible in `api.optimize`: on the genetic path (any problem
with seeds) `kind == "llm"` builds the completion seam directly and never calls
`_build_harness`, which is the only place that consults
`bootstrap.requirement_failure` and names `pip install 'agent_evolve[llm]'`. So
the 0.4.0 entry "Asking for the model proposer without the `[llm]` extra now
fails up front naming the install command" is **no longer true on the path most
problems take**, and a request for a model that cannot be honoured is now a
silent fallback rather than a refusal — the exact class of defect the seal
refusal, the chooser refusal and the genetic-knob refusal all exist to prevent.
Not fixed here: the fix belongs in `optimize`'s own resolution, which is under
live measurement, and it should be measured rather than patched at the edge.

**2. `test_credentials_present_is_true_for_a_real_provider_key` fails only under
the local runner**, because `scripts/ci-local.sh` injects
`AGENTEVOLVE_SCRUBBED=OPENAI_API_KEY,…,OPENROUTER_API_KEY,…` into every
container so that a host key cannot make the stranger job pass for the wrong
reason. The scrub rule outranks detection by design — the very next test in that
file asserts exactly that — so the test that sets `OPENROUTER_API_KEY` and
expects it to count cannot pass with the flag set. Isolated on pristine: the
file is 20/20 green without the variable and fails that one test with it. GitHub
does not set it, so this job is green on GitHub and red locally. It is worth
recording anyway: `ci-local.sh`'s own claim is that there is no second
definition of what CI does and nothing to drift, and this is a drift of exactly
that kind, in the runner rather than in the workflow.

**3. `test_full_multi_option_evolution_runs_through_reflection` fails on 3.13
only**, and reproduces on pristine `cbed1c5` under a clean 3.13 environment
(`At index 5 diff: (0, 5, 0) != (2, 0, 1)`). A declared-order divergence in a
fake-driven pipeline, on the newest supported interpreter, in a test nothing in
this cut touches. It is a real matrix failure and it is not this cut's.

## Shipping as a stated limitation, not as a blocker

Each of these is known, documented where a user would meet it, and deliberately
not fixed in this cut. They are listed here so nobody has to rediscover that the
decision was made.

- **651 `_LEGACY_EXPORTS`** resolve lazily through `__getattr__`. Ship-as-is:
  trimming them breaks the property that the tree which ships is the tree that
  was measured, which is what makes any release claim checkable.
- **`HarnessBase` is not public while `harness_registry` is.** A harness can be
  registered with no supported way to name its type.
- **`check`'s model arm is a single run at `seed = 0`** while `--repeats`
  repeats the baseline. Now said out loud in `--help`; a `--model-repeats` is
  the fix, and it is not in this cut.
- **`abc.history`** sits untracked in the working tree. It is gitignored and
  was confirmed absent from the sdist; removing it is repository hygiene, not a
  release blocker.

## Owner-only residue — nothing here can be closed from this repository

- [ ] **Push `main`.** `origin` carries `main` alone and this cut is not on it.
- [ ] **Push `release/v0.4-sweep`** (at `4d0b7b0`, `__version__ = "0.4.0"`),
      which preserves the substrate every sealed row to date was measured on.
      **It is local-only today**, so the preservation is real in this clone and
      in no published one — and it is a precondition for the paper citing the
      substrate a row belongs to.
- [ ] **PyPI publication.** Nothing has been published; the name is unclaimed
      as far as this repository knows.
- [ ] **`Development Status :: 4 - Beta`** — still the declared classifier.
      Confirm it is the intended signal for a first public cut.
- [ ] **The paper's venue**, and with it the "available at" sentence, which
      becomes writable only once the branches above are pushed.

## Finalization pass — 2026-08-24 (this working tree, ready to tag)

Everything below was verified by running it on this tree, after the
mechanism-v3 and tuning-knob commits that postdate the first cut of this
checklist.

- [x] **The distribution is `agentevolve-optimizer`, and that is a
      supply-chain fix — settled by two PyPI rules, the second found by a
      live upload.** `agent_evolve` PEP-503-normalizes to `agent-evolve`,
      TAKEN by an unrelated package — every install line the README ever
      suggested would have fetched someone else's code. The obvious repair,
      plain `agentevolve`, was then refused at upload time: Warehouse's
      similarity check strips separators before comparing, so a name
      differing from a squat only by an underscore's absence is "too similar
      to an existing project" by rule (owner-observed on the real upload,
      2026-08-24). The suffix clears the ultranormalized comparison; the
      owner chose `-optimizer`. Pinned with the full rationale by
      `tests/test_packaging_metadata.py::test_the_distribution_name_is_the_unclaimed_spelling`.
      The IMPORT stays `agent_evolve`; the console script stays
      `agent_evolve`; only `pip install` changes, and every install line in
      README, the pymoo-swap README, `bootstrap`'s hints, the refusal
      message and the CI stranger grep carries the final spelling. Artifacts
      rebuilt (`agentevolve_optimizer-0.5.0` wheel + sdist, twine PASSED),
      wheel re-smoked in a clean venv, `uv.lock` regenerated, tag moved
      (local-only, never pushed).
- [x] **The stranger's refusal names every fix that applies.** On a core
      install with no credential, `--proposer llm` names the extra
      (`pip install 'agentevolve[llm]'`) AND the credential AND the
      credential-free way out; with the extra present it does not tell you to
      reinstall it. `api._llm_refusal_message`, both renderings unit-tested;
      the CI stranger job's grep is the behavioral gate and passes.
- [x] **CHANGELOG is complete through the tree being tagged**: the rename,
      mechanism v3 (with its re-pilot verdicts), and the four tuning knobs
      (each carrying its measured pilot read, including the negative ones) —
      and the 0.5.0 section is dated.
- [x] **`uv.lock` regenerated** at the renamed distribution (dev-only; not in
      the sdist).
- [x] **Clean build from an empty `dist/`**: `agentevolve-0.5.0-py3-none-any.whl`
      + `agentevolve-0.5.0.tar.gz`, zero warnings, `twine check` PASSED on
      both.
- [x] **Wheel smoke in a clean venv** (the stranger's path, scripted):
      `agent_evolve version` → 0.5.0; import name unchanged; `init` scaffold
      writes and refuses overwrite; five-obligation problem defined from
      scratch optimizes offline; explicit-llm-without-credential refuses
      loudly naming the fixes; `[pymoo]` extra installs and the swap runs.
- [x] **Full offline suite green at the release state** (see the final commit
      message for the count), including the new name-pin and refusal-message
      tests.
- [x] **CI matrix state, honestly**: `stranger` PASSES after the refusal fix;
      the two remaining local-runner reds are pre-existing and named —
      `test_credentials_present_is_true_for_a_real_provider_key` fails ONLY
      under `act`'s injected `AGENTEVOLVE_SCRUBBED` (green on GitHub), and one
      py3.13-only research-path ordering drift
      (`test_full_multi_option_evolution_runs_through_reflection`) reproduces
      on a pristine tree and predates this release.
- [x] **`Development Status :: 4 - Beta` stands** as the declared signal for a
      first public cut — honest for a tool whose adaptive channel is
      documented as under measurement.

## Owner actions — the complete remaining distance

- [ ] `git push origin main`, then `git push origin v0.5.0` — the tag push
      is the publish trigger.
- [ ] `git push origin release/v0.4-sweep` — the sealed-substrate branch the
      paper cites; local-only until pushed.
- [ ] PyPI, via the pending Trusted Publisher the owner already registered
      (2026-08-24): verify its four fields read exactly `agentevolve-optimizer`
      / `dyigitpolat` / `agent_evolve` / `publish.yml`, with environment
      `pypi` (pending publishers are freely editable until first use). Then
      the two pushes below make the release: the tag-push runs
      `.github/workflows/publish.yml`, whose first successful OIDC upload is
      what CREATES the project. No token exists anywhere in this flow. A
      `twine upload` with an API token also works but would create the
      project OUTSIDE the pending publisher, invalidating it — if that path
      is ever taken, re-add the publisher afterward under the project's own
      settings. In parallel and without blocking anything: a PEP 541
      name-transfer request against the squatted `agent-evolve` (placeholder
      URLs, single release) can be filed at github.com/pypi/support; if it
      ever succeeds, `agentevolve` can become an alias later.
- [ ] GitHub release notes: the 0.5.0 CHANGELOG section is written to be
      pasted.
