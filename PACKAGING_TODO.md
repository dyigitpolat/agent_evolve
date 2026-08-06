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
- [ ] **`LICENSE.draft` holder line** — "AgentEvolve authors" is deliberately
      impersonal; owner decides the released holder line, then the draft
      replaces `LICENSE` (currently a personal-name MIT) or is discarded.

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
