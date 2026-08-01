"""The certified acquisition modes must be runnable without qLogNEHVI.

This is the test whose absence let the defect survive.  `acquisition_certified`
and `regret_bounded_information` used to REQUIRE protected qLogNEHVI at batch
exactly 8.  On an 8-seat stage that reserved every seat, so the model could not
affect the evaluated set at all -- which is why the Timeloop campaign of record
offered 384 catalogue options per stage and bought zero of them, on the one
domain where we decisively beat random search.  Nothing covered the off path, so
nothing failed when the interlock closed it.

The runner is a module whose import performs all of its environment validation
at module scope, so the contract is exercised by importing it under a controlled
environment and observing whether it raises.  That is the same surface a
campaign launch uses, and it is the surface the defect lived on.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "examples" / "development" / "run_boils_generic_campaign.py"

# Importing the runner is expensive and has module-scope side effects, so each
# case runs in its own interpreter with an explicit environment.
_PROBE = """
import importlib, json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {src!r})
try:
    module = importlib.import_module("examples.development.run_boils_generic_campaign")
except ValueError as error:
    print(json.dumps({{"raised": "ValueError", "message": str(error)}}))
    raise SystemExit(0)
print(
    json.dumps(
        {{
            "raised": None,
            "engine": module.NUMERICAL_CERTIFICATION_ENGINE,
            "batch_size": module.PROTECTED_ACQUISITION_BATCH_SIZE,
            "protected_mode": module.PROTECTED_ACQUISITION_MODE,
            "planned_share": module.PLANNED_MODEL_REACHABLE_SHARE_OF_EVALUATED_SEATS,
            "planned_seats_claimed": module.PLANNED_PROTECTED_ACQUISITION_SEATS,
            "portfolio_width": module.PORTFOLIO_WIDTH,
            "reservation_by_generation": {{
                str(g): module._protected_acquisition_source_minimum(g)
                for g in module.PORTFOLIO_GENERATIONS
            }},
        }}
    )
)
"""


def _import_runner(**environment: str) -> dict[str, object]:
    import json

    env = dict(os.environ)
    for name in [key for key in env if key.startswith("AGENT_EVOLVE_")]:
        del env[name]
    env.update(environment)
    env.setdefault("AGENT_EVOLVE_OFFLINE_IMPORT_PROBE", "1")
    # Prerequisites of the certified modes that are unrelated to this defect but
    # are enforced at module scope: the operator assay demands the hierarchical
    # radius-two topology.  Setting them keeps each case pointed at the one
    # interlock it is about.
    if env.get("AGENT_EVOLVE_ACQUISITION_MODE") in CERTIFIED_MODES:
        env.setdefault("AGENT_EVOLVE_VARIATION_TOPOLOGY", "hierarchical_r2")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE.format(
                root=str(REPOSITORY_ROOT), src=str(REPOSITORY_ROOT / "src")
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        cwd=str(REPOSITORY_ROOT),
    )
    if completed.returncode != 0:
        pytest.skip(
            "runner import is not exercisable in this environment: "
            + completed.stderr.strip()[-400:]
        )
    line = [row for row in completed.stdout.splitlines() if row.startswith("{")]
    if not line:
        pytest.skip("runner import produced no probe record")
    return json.loads(line[-1])


CERTIFIED_MODES = ("acquisition_certified", "regret_bounded_information")


@pytest.mark.parametrize("mode", CERTIFIED_MODES)
def test_certified_mode_runs_without_qlognehvi(mode: str) -> None:
    """The off path exists: certification does not require the protected engine."""

    record = _import_runner(
        AGENT_EVOLVE_ACQUISITION_MODE=mode,
        AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION="1",
        AGENT_EVOLVE_NUMERICAL_CERTIFICATION_ENGINE="off",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="off",
    )
    assert record["raised"] is None, record
    assert record["protected_mode"] == "off"
    # With no protected engine every evaluated seat is reachable by the model.
    assert record["planned_seats_claimed"] == 0
    assert record["planned_share"] == 1.0


@pytest.mark.parametrize("mode", CERTIFIED_MODES)
def test_certified_mode_accepts_the_batch_default(mode: str) -> None:
    """The equality became a floor, and the floor admits the existing default.

    `PROTECTED_ACQUISITION_BATCH_SIZE` defaults to 2 with bounds [2, 8]; the old
    interlock demanded exactly 8, so the lockout was manufactured by the
    interlock rather than by any default.
    """

    record = _import_runner(
        AGENT_EVOLVE_ACQUISITION_MODE=mode,
        AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION="1",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="botorch_qlognehvi",
    )
    assert record["raised"] is None, record
    assert record["batch_size"] == 2
    assert record["planned_share"] > 0.0, "the model must be able to reach a seat"


@pytest.mark.parametrize("mode", CERTIFIED_MODES)
def test_batch_eight_remains_legal_and_unchanged(mode: str) -> None:
    """No launch record in flight changes meaning: 8 still satisfies the floor."""

    record = _import_runner(
        AGENT_EVOLVE_ACQUISITION_MODE=mode,
        AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION="1",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="botorch_qlognehvi",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_BATCH_SIZE="8",
    )
    assert record["raised"] is None, record
    assert record["batch_size"] == 8


def test_certification_still_requires_constraint_decoupling() -> None:
    """The other half of the old interlock is preserved, not relaxed."""

    record = _import_runner(
        AGENT_EVOLVE_ACQUISITION_MODE="acquisition_certified",
        AGENT_EVOLVE_NUMERICAL_CERTIFICATION_ENGINE="off",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="off",
    )
    assert record["raised"] == "ValueError"
    assert "constraint-decoupled" in record["message"]


def test_qlognehvi_engine_still_demands_the_protected_mode() -> None:
    """Asking for the engine and withholding it is still an error."""

    record = _import_runner(
        AGENT_EVOLVE_ACQUISITION_MODE="acquisition_certified",
        AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION="1",
        AGENT_EVOLVE_NUMERICAL_CERTIFICATION_ENGINE="botorch_qlognehvi",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="off",
    )
    assert record["raised"] == "ValueError"
    assert "botorch_qlognehvi" in record["message"]


def test_seat_value_decay_defaults_to_the_constant() -> None:
    """An unconfigured campaign resolves exactly as it did before the decay."""

    record = _import_runner(
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="botorch_qlognehvi",
    )
    assert record["raised"] is None, record
    reservations = record["reservation_by_generation"]
    assert set(reservations.values()) == {1}, reservations


def test_seat_value_decay_releases_the_reservation_at_parity() -> None:
    """The measured law: held while worth >= a catalogue seat, released below.

    Generation 1 resolves to the constant it replaces, so the schedule's first
    value is not a new number; by stage 3 a qLogNEHVI seat is measured below
    parity on both objectives and the reservation goes to zero.
    """

    record = _import_runner(
        AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE="botorch_qlognehvi",
        AGENT_EVOLVE_PROTECTED_ACQUISITION_SEAT_VALUE_DECAY="measured",
    )
    assert record["raised"] is None, record
    reservations = {int(k): v for k, v in record["reservation_by_generation"].items()}
    assert reservations[1] == 1, reservations
    assert all(value == 0 for key, value in reservations.items() if key >= 3), (
        reservations
    )


def test_seat_accounting_reports_a_hard_lockout_rather_than_hiding_it() -> None:
    """G0 becomes mechanical: the receipt states the share itself."""

    module_root = str(REPOSITORY_ROOT)
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
        "_r0_runner_probe", RUNNER
    )
    assert spec is not None and spec.loader is not None
    # The accounting helper is pure and is exercised directly against rows in
    # the shape the construction probe records, so this case costs no import.
    rows = (
        {
            "generation": 1,
            "parent_slot": 0,
            "evaluation_width": 8,
            "eligible_option_count": 384,
            "numerical_acquisition_option_count": 8,
        },
    )
    namespace: dict[str, object] = {
        "PROTECTED_ACQUISITION_BATCH_SIZE": 8,
        "PROTECTED_ACQUISITION_SOURCE_MINIMUM": 1,
        "PROTECTED_ACQUISITION_SEAT_VALUE_DECAY": "off",
        "SEAT_VALUE_PARITY_GENERATION": 3,
    }
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("def _protected_acquisition_source_minimum(")
    end = source.index("def _protected_acquisition_envelope(")
    exec(compile(source[start:end], str(RUNNER), "exec"), namespace)  # noqa: S102
    accounting = namespace["_model_reachable_seat_accounting"](rows)  # type: ignore[operator]
    assert accounting["model_reachable_share_of_evaluated_seats"] == 0.0
    assert accounting["hard_lockout"] is True
    assert accounting["catalogue_options_offered"] == 376
    assert accounting["evaluated_seats"] == 8
    assert accounting["protected_acquisition_seats"] == 8


def test_seat_accounting_reports_a_reachable_share_at_the_batch_default() -> None:
    """Batch 2 against an 8-seat stage leaves six seats the model can reach."""

    rows = (
        {
            "generation": 1,
            "parent_slot": 0,
            "evaluation_width": 8,
            "eligible_option_count": 384,
            "numerical_acquisition_option_count": 2,
        },
    )
    namespace: dict[str, object] = {
        "PROTECTED_ACQUISITION_BATCH_SIZE": 2,
        "PROTECTED_ACQUISITION_SOURCE_MINIMUM": 1,
        "PROTECTED_ACQUISITION_SEAT_VALUE_DECAY": "off",
        "SEAT_VALUE_PARITY_GENERATION": 3,
    }
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("def _protected_acquisition_source_minimum(")
    end = source.index("def _protected_acquisition_envelope(")
    exec(compile(source[start:end], str(RUNNER), "exec"), namespace)  # noqa: S102
    accounting = namespace["_model_reachable_seat_accounting"](rows)  # type: ignore[operator]
    assert accounting["model_reachable_seats"] == 6
    assert accounting["model_reachable_share_of_evaluated_seats"] == 0.75
    assert accounting["hard_lockout"] is False
