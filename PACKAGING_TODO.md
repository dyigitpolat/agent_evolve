# PACKAGING_TODO — OSS release blockers and decisions (draft, nothing published)

State as measured on 2026-08-06 against this working tree (`pyproject.toml`
version `0.3.0`, venv `.venv/`). This draft added only `LICENSE.draft`,
`README.oss.md`, and this file; no `src/` file and no `pyproject.toml` change.

## 1. Exact-pin dependencies to relax

The good news first: **`[project.dependencies]` is already ranges, not pins**
(`pydantic>=2.7,<3`, `python-dotenv>=1.0,<2`; extras `llm`/`pydantic_ai` =
`pydantic-ai>=1.0,<2`, `research` = `botorch>=0.11`, `numpy>=1.24`, `dev`,
`all`). Nothing to relax in the published metadata itself. The exact pins that
remain live around it:

- [ ] **`uv.lock` is stale and exact by nature** — 525 locked packages, and it
      records `agent-evolve` at version `0.2.0` while `pyproject.toml` says
      `0.3.0`. Regenerate before release; confirm it stays a dev-only file and
      never enters the sdist/wheel as an install constraint.
- [ ] **The venv's editable install is pinned to the old version** —
      `site-packages` carries `agent_evolve-0.2.0.dist-info` and
      `__editable__.agent_evolve-0.2.0.pth`. Consequence today:
      `agent_evolve version` (CLI, via `importlib.metadata`) prints `0.2.0`
      while `agent_evolve.__version__` is `0.3.0`. Reinstall editable;
      consider single-sourcing the version so this class of skew cannot ship.
- [ ] **`examples/development/systematic_workload_contracts/heat2d.json` and
      `heat2d_v2.json` pin `numpy==2.3.5`** inside their environment spec.
      Either relax to a range or annotate them as frozen reproduction
      contracts that are exempt on purpose.
- [ ] **`pymoo 0.6.2` is installed in the dev venv but declared nowhere**,
      and the adapter that needs it now exists in the working tree
      (`src/agent_evolve/integrations/pymoo_adapter.py`, uncommitted). Add an
      optional extra (suggest `pymoo = ["pymoo>=0.6,<0.7"]`) when it ships;
      keep the demo/tests behind a guarded import so the core install stays
      two-dependency.

## 2. Export count: today vs target (~12)

- **Today: `__all__` = 11 names** (10 symbols + `__version__`):
  `optimize`, `Problem`, `ObjectiveSpec`, `ValidationOutcome`,
  `SearchResult`, `Candidate`, `as_problem`, `artifact_key`,
  `RandomProposer`, `harness_registry`, `__version__`.
- **Target: ~12** — room for exactly one more, and there is a real candidate:
  - [ ] `harness_registry` is public but the base class an out-of-tree
        proposer must subclass (`agent_evolve.harness.base.HarnessBase` /
        `Harness`) is not. You can register a harness today without any
        supported way to name its type. Promote one of the two, or document
        the deliberate omission.
  - [ ] Decide whether `as_evaluator` (contract sibling of `as_problem`)
        stays internal.
- [ ] **651 legacy lazy re-exports** resolve through `__getattr__`
      (`_LEGACY_EXPORTS` in `src/agent_evolve/__init__.py`), explicitly
      unsupported but load-bearing for "the measured research stack runs
      against the shipped tree". Decide for the OSS cut: ship all 651 as-is
      (current behavior, zero import cost), or trim — trimming breaks that
      equivalence claim, so the default should be ship-as-is with the
      unsupported status stated in the README.

## 3. CLI gaps

Today's subcommands: `check` / `diagnose` / `run` / `version` (`diagnose` is
an uncommitted working-tree addition, additive-only, backed by
`policies/check.py` + `tests/test_check.py`).

- [ ] **`run` does not expose `seal=`** although `api.optimize(...)` has it —
      the provider-free journal/replay story (the package's audit answer) has
      no CLI path at all. Additive: `run --seal PATH`, plus a new `replay`
      (or `verify`) subcommand for an existing journal.
- [ ] **`run` does not expose `strategy=`** (`auto`/`genetic`/`authoring`);
      the API parameter is unreachable from the CLI.
- [ ] **No machine-readable output**: `run` and `check` print prose only; a
      `--json` dump of `SearchResult` / the check verdict is needed for any
      scripted use.
- [ ] **`version` skew**: prints `0.2.0` from the stale editable metadata
      (see §1); after the reinstall, consider printing both distribution and
      `__version__` so skew is visible instead of silent.
- [ ] **`check` model arm runs at fixed `seed=0`** and single shot; `--repeats`
      applies to the baseline only. Fine for a first release, but say so in
      `--help` or add `--model-repeats`.
- [ ] Nice-to-have: `agent_evolve init` scaffold that writes a five-obligation
      `problem_def.py` template (the knapsack file, blanked).

## 4. Pointers this draft leaves dangling (must close before publishing)

- [ ] **pymoo swap demo**: landed in the working tree while this draft was
      being written — `examples/pymoo_swap/swap_demo.py`,
      `src/agent_evolve/integrations/pymoo_adapter.py` (`from_pymoo`,
      continuous axes quantized onto `Literal` grids),
      `tests/test_pymoo_adapter.py` — all **uncommitted**. Commit them
      together with `README.oss.md`, whose MIGRATION section points at them,
      and add the `pymoo` extra from §1.
- [ ] **Scope-table sources** are quoted from
      `papers/agent_evolve_aaai_2027/research_artifacts/…`, which is outside
      this package. A standalone GitHub release needs either a vendored
      snapshot (e.g. `docs/measurements.md` with the same exact numbers) or a
      link to the paper; the numbers themselves must not be retyped from
      memory when that snapshot is made.
- [x] **LICENSE holder line — SETTLED 2026-08-18 by the owner.** The released
      holder is the named individual: `LICENSE` stands unchanged at
      `Copyright (c) 2026 Yigit Polat`, MIT. `LICENSE.draft` and its impersonal
      "AgentEvolve authors" form are discarded, not promoted.

## 5. Noticed while drafting (pre-release polish, not this draft's scope)

- [ ] `result.pareto_front` returns duplicate entries: the credential-free
      quickstart at `budget=32` returned a 59-row front holding 4 distinct
      configurations. First thing a new user prints; dedupe or document.
- [ ] Untracked working files that must not reach an sdist: `abc.history`,
      `.probe_launch/`. Also untracked and undecided:
      `examples/development/run_analog_sizing_driver_cell.py` — commit or drop.
- [ ] Wheel metadata says `Development Status :: 4 - Beta` — confirm that is
      still the intended signal for the first public cut.

---

## State update 2026-08-07 (append-only; line references above are the 2026-08-06 snapshot)

- FIXED: `pareto_front` duplicate entries — `compute_pareto_front` now collapses
  exact duplicates (same configuration identity + same objectives, first
  occurrence wins) while keeping noisy re-evaluations distinct; quickstart
  repro prints 4 rows / 4 distinct configs. Tests in `tests/test_results.py`.
- FIXED: version skew — editable install re-synced; `importlib.metadata`,
  `__version__`, and `uv.lock` all agree on 0.3.0.
- FIXED: CLI gaps — `run` now takes `--strategy {auto,genetic,authoring}` and
  `--seal PATH`. `optimize(seal=)` on the genetic path now REFUSES by name
  (the seal format records generative proposals, not operator choices) instead
  of silently writing nothing; tests in `tests/test_optimize_strategy.py`.
- FIXED: `swap_demo.py` docstring said "ten lines each way" — now "three code
  lines each way", matching the file and the scorecard/strategy claim.
- Still open: HarnessBase not public while `harness_registry` is; `--json`
  output; `check` model-arm single-shot; `_LEGACY_EXPORTS` trim decision;
  LICENSE owner review; Beta classifier; untracked-file hygiene for sdist.

---

## State update 2026-08-18 — the V6 release cut (append-only)

Everything below was **verified by running it**, not by reading the code. The
transcripts are in `papers/agent_evolve_iclr_2027/V6_RELEASE.md`.

### Closed

- **Version skew, structurally.** `pyproject.toml` is now `dynamic =
  ["version"]` reading `agent_evolve.__version__`. One literal, in the code.
  `tests/test_packaging_metadata.py::test_version_is_single_sourced_from_the_code`
  refuses a literal in the metadata; a second test asserts the installed
  distribution and the attribute agree.
- **The `pymoo` extra exists** (`pymoo>=0.6,<0.7`), and is in `all`.
- **The swap demo is committed and wired into the suite** — and rebuilt as two
  sibling files whose *diff* is the assertion. See `examples/pymoo_swap/README.md`.
- **`--json`**: `run --json` verified end to end on a clean wheel install,
  parses, and now carries a derived `cost_usd` with its provenance.
- **`LICENSE.draft` is no longer distributed.** `project.license-files =
  ["LICENSE"]` (PEP 639). The wheel had been shipping a file whose first line
  reads "DRAFT — NOT IN EFFECT" beside the real licence.
- **sdist/wheel hygiene.** Neither `abc.history` nor `.probe_launch/` reaches
  the sdist (checked by listing the archive). Build emits zero warnings.
- **Dependency floors are verified, not asserted.** A fresh venv at the declared
  floors (`pydantic==2.7.4`, `python-dotenv==1.0.1`) runs the CLI end to end.
- **The `llm` extra no longer declares a range the code refuses.** It pins the
  three versions `boundary_codec` fails closed on; a test keeps metadata and
  code equal.
- **`README.oss.md` is deleted.** Six of its ten scope rows quoted a venue that
  the paper's struck-citation gate strikes for venue validity. Its useful
  content (the migration table, the five-obligation table) is in `README.md`.

### Still open, and now explicit

- [x] **LICENSE holder line — CLOSED 2026-08-18.** The owner decided the
      released holder is the named individual. `LICENSE` stands as written
      (`Copyright (c) 2026 Yigit Polat`, MIT); `LICENSE.draft` is deleted. This
      was the item blocking a public cut, so the paper's "available at"
      sentence becomes writable once the branch is pushed — which remains the
      owner's action, not this repository's.
- [ ] **Integer loci with `ge`/`le` bounds declare no finite domain**, so the
      operators leave them alone — `diagnose` says so out loud, and the knapsack
      example searches nothing because of it. Making bounded integers
      enumerable is a genuine generic improvement *and* it would change search
      behaviour on every venue with an integer locus, including measured ones.
      That is a research-integrity call, not a packaging one.
      **DEFERRED UNTIL AFTER SUBMISSION, by the owner, 2026-08-18.** The
      reasoning is the deadline, not the merits: changing it now would move the
      substrate under sealed rows six weeks before ICLR (full paper 2026-09-25),
      and re-verifying every venue with an integer locus is not a pre-deadline
      purchase. It ships as a DOCUMENTED LIMITATION — `diagnose` already says it
      out loud, which is the behaviour that makes deferring honest rather than
      quiet. Revisit after submission, and when it lands, re-run the affected
      sealed cells and disclose any number that moves rather than absorbing it.
- [ ] **`Development Status :: 4 - Beta`** — still the declared signal; confirm
      it is the intended one for a first public cut.
- [ ] **651 `_LEGACY_EXPORTS`** — ship-as-is remains the default; unchanged.
- [ ] **`HarnessBase` is not public while `harness_registry` is** — unchanged.
- [ ] **`check`'s model arm is single-shot at `seed=0`** while `--repeats`
      applies to the baseline only — unchanged, and `--help` still does not say so.
- [ ] **`check` has no `--json`.** `run` does. Scripted use of the verdict still
      has to parse prose.
- [ ] **A scope snapshot inside the package.** `README.md` now carries the
      numbers with their wave names, but the archive itself is deliberately not
      distributed. A `docs/measurements.md` vendored snapshot is still the
      cleanest answer for a standalone GitHub release.

## 6. Known defects found by measurement, 2026-08-19 — not fixed

These were found by the end-to-end evaluation row, not by the test suite, and
each one is a real limit on what the package can currently do. They are listed
here because the next person will hit them.

- [ ] **`api.optimize` cannot sell the working mechanisms without the worst
      one.** `proposer="auto"` builds the completion seam *and* the per-offspring
      chooser; `proposer="random"` leaves `complete=None`, which silently
      disables the authoring seams. The authoring jobs are where the measured
      wins are; the per-offspring chooser has ten null verdicts and costs
      107–171x the run it advises. **No drop-in user can currently run the
      configuration that produced this project's end-to-end results.** One-flag
      fix; it is the single most valuable thing to do here.
- [ ] **Continuous parameters cannot be searched at all.**
      `policies.genetic._declared_domain` returns `()` for any field that is not
      a finite enum or bool, so `uniform_candidate` and `mutate` leave it
      frozen. Measured: 12 of 13 axes on a hyperparameter benchmark were
      identical across 64 draws. This blocks an entire class of problems, and
      the failure is silent. Needs either a finite projection or a bounded
      perturbation.
- [ ] **A single-valued field renders as `{"const": ...}` rather than
      `{"enum": [...]}`,** and the same reader ignores `const`. Harmless where
      there is nothing to search, but it is the same one-line gap as above.

**A measured direction, not a defect.** Guidance is authored once at the start
and nothing refreshes it against what the run subsequently measures. On the one
live physical simulator tested, the guided stack is the best arm at 20, 40 and
80 evaluations and the worst by 160 — it is not wasting budget (zero failed
evaluations against uniform's 1.25%), it is working from a stale picture. The
seam that would fix this (`reauthor_every` / `evidence_min_rows` in
`policies/llm_generator.py`) is landed and **off by default**; making it earn
its place is the clearest open research direction.

---

## State update 2026-08-19 (W0) — append-only

Sections above are left as written; the entries below say what moved. Substrate
is now `0.5.0` (`agent_evolve.__version__`, single-sourced; editable install
re-synced, `tests/test_packaging_metadata.py` green).

### §6 — all three defects FIXED

- **FIXED — `api.optimize` cannot sell the working mechanisms without the worst
  one.** The completion seam is now built once for the run instead of inside the
  chooser's branch, and the per-offspring chooser moves to its own parameter,
  `chooser`, defaulting to `"off"`. `proposer="llm"` now resolves to the
  measured stack — model-proposed initialization (A4: 11× fewer evaluations to
  target, 40 of 40 paired seeds, one call) + the sealed authored surrogate +, at
  `budget >= 48`, an auto-sized crossed screen read by the model-weighted graded
  prior (A5, 4.60×). `prior` and `structure_budget` become `"auto"` sentinels
  that resolve to the literal old defaults (`"rule"`, `0`) whenever no model
  call is possible, so the credential-free stream is unchanged — verified
  against the byte fossil, not asserted. `chooser="llm"` still reaches the
  mechanism; on a run that makes no model call it is refused by name.
  `tests/test_headline_defaults.py`, CLI `run --chooser {off,llm}`, authorship
  preset `"guided"`.
- **FIXED — continuous parameters cannot be searched at all.**
  `policies.genetic._declared_domain` now projects bounded numeric loci onto
  finite domains: integers enumerated when the span is ≤ 64 and otherwise spread
  over a 64-point grid, doubly bounded floats onto a 16-point inclusive grid
  (the resolution `from_pymoo` has always used — note the adapter deliberately
  does *not* delegate to the new code: its grid is `linspace`'s full binary
  float, the projection's is rounded through `%.10g` so one value renders
  identically in a prompt, a card and a comparison, and moving every pymoo grid
  point to share one rule was judged not worth it), `multipleOf` honoured,
  exclusive bounds excluded, `anyOf`/`oneOf` scanned for the first such branch
  with the `null` branch skipped, and a per-field override
  `Field(json_schema_extra={"agent_evolve": {"grid": N}})` clamped to `[2, 256]`.
  `locus_is_projected` lets a report distinguish a projection from a declared
  set; `diagnose` prints `inline_threshold:64(projected)`.
  `tests/test_numeric_domains.py`.
- **FIXED — a single-valued field renders as `{"const": ...}` and the reader
  ignored it.** `_node_domain` reads `const` as a one-value domain, so such a
  field is declared rather than undeclared to every reader in the package —
  including the `xl_i == xu_i` collapse `from_pymoo` produces.

### Obsolete characterizations in the sections above

- **"the knapsack example searches nothing"** (carried in §5's neighbourhood and
  restated in the 2026-08-18 deferral) is **obsolete**. Its `selection` locus
  declares `ge=0, le=9`, which now projects to 10 values; `diagnose` reports
  `selection[0]:10(projected)` and `undeclared domains: none`, and the probe
  varies it. The example remains an honest negative, for the real reason rather
  than the mechanical one: 40 random draws are expected to reach the same best
  value (150) and the same best weight (5) the probe found, so the verdict is
  still no-headroom.
- The README's `build_flags` and `diagnose` transcripts were re-run and replaced
  rather than edited: the front grew from 6 rows to 11 at the same 40-evaluation
  budget, and the verdict's "1 of 6 loci declare no finite domain" footnote is
  gone because there are none.

### The 2026-08-18 DEFERRAL is superseded

The open item **"Integer loci with `ge`/`le` bounds declare no finite domain …
DEFERRED UNTIL AFTER SUBMISSION, by the owner, 2026-08-18"** is **superseded on
2026-08-19 by the owner's direction to develop heavily.** The deferral's own
reasoning was the deadline and explicitly not the merits, so the reversal needs
no new argument about the merits — only a place to stand for the rows already
sealed.

That place is **branch `release/v0.4-sweep`** (at `4d0b7b0`,
`__version__ = "0.4.0"`), which preserves the 0.4 substrate. Every sealed row
measured to date was measured there and remains quotable as such; a row measured
on 0.5.0 is a row measured on a different search space, and the two are not
interchangeable. **That branch is local-only today** — `origin` carries `main`
alone — so the preservation is real in this clone and not yet in any published
one. Pushing it is the owner's action, and it is a precondition for the paper
citing the substrate a sealed row was measured on. The deferral's own closing
instruction still binds and is now the open work: **re-run the affected sealed
cells on 0.5.0 and disclose any number that moves rather than absorbing it.**
The byte-fossil test passes unmodified — its venue is enum-only, so nothing it
pins can move — which bounds the exposure to venues with a numeric locus but
does not measure it.
