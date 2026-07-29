"""Provider-free must be a measurement, not a constant.

The defect these cover: a sealed receipt asserted ``"zero_provider_calls":
PROVIDER_CALLS == 0`` where ``PROVIDER_CALLS = 0`` was a module constant. The
expression cannot evaluate to False. It sat in a health block beside genuine
checks like ``len(wave_records) == 6``, so it read as verified.
"""

from __future__ import annotations

import json

import pytest

from agent_evolve.provider_accounting import (
    DeclaredJournalMissing,
    ProviderUsage,
    ensure_declared_journals,
    measure_provider_usage,
)

DECLARED = {
    "outcome": "queue_outcomes.jsonl",
    "outbound": "outbound_requests.jsonl",
    "campaign": "campaign_events.jsonl",
}


def _write(run_dir, filename, rows):
    path = run_dir / filename
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_a_declared_journal_that_was_never_written_fails_loudly(tmp_path):
    """242 sealed runs declared a journal and never created it; that is a fault."""
    _write(tmp_path, "queue_outcomes.jsonl", [])
    _write(tmp_path, "campaign_events.jsonl", [])
    # outbound_requests.jsonl declared but absent

    with pytest.raises(DeclaredJournalMissing) as excinfo:
        measure_provider_usage(tmp_path, DECLARED)

    assert "outbound" in str(excinfo.value)
    assert "outbound_requests.jsonl" in str(excinfo.value)


def test_an_empty_journal_is_a_measured_zero(tmp_path):
    """The whole point: absent proves nothing, present-and-empty proves zero."""
    for filename in DECLARED.values():
        _write(tmp_path, filename, [])

    usage = measure_provider_usage(tmp_path, DECLARED)

    assert usage.provider_calls == 0
    assert usage.provider_free is True
    assert usage.cost_usd == "0"
    assert usage.to_record()["measured_from"] == {
        "campaign": 0,
        "outbound": 0,
        "outcome": 0,
    }


def test_provider_traffic_is_counted_not_declared(tmp_path):
    _write(
        tmp_path,
        "queue_outcomes.jsonl",
        [
            {"response": {"input_tokens": 100, "output_tokens": 20, "cost_usd": "0.0031"}},
            {"response": {"input_tokens": 50, "output_tokens": 5, "cost_usd": "0.0007"}},
        ],
    )
    _write(tmp_path, "outbound_requests.jsonl", [{"url": "x"}, {"url": "y"}])
    _write(tmp_path, "campaign_events.jsonl", [])

    usage = measure_provider_usage(tmp_path, DECLARED)

    assert usage.provider_calls == 2
    assert usage.outbound_rows == 2
    assert usage.input_tokens == 150
    assert usage.output_tokens == 25
    assert usage.cost_usd == "0.0038", "exact decimal, not float drift"
    assert usage.provider_free is False


def test_a_run_that_never_called_out_still_leaves_the_evidence(tmp_path):
    """A provider-free run must create its journals so zero can be read back."""
    created = ensure_declared_journals(tmp_path, DECLARED)

    assert sorted(created) == ["campaign", "outbound", "outcome"]
    usage = measure_provider_usage(tmp_path, DECLARED)
    assert usage.provider_free is True


def test_ensure_declared_journals_never_truncates_existing_evidence(tmp_path):
    _write(tmp_path, "queue_outcomes.jsonl", [{"response": {"input_tokens": 7}}])

    created = ensure_declared_journals(tmp_path, DECLARED)

    assert "outcome" not in created
    assert measure_provider_usage(tmp_path, DECLARED).input_tokens == 7


def test_an_outcome_row_without_a_response_is_not_a_provider_call(tmp_path):
    """A queued-then-cancelled row must not inflate the count."""
    _write(tmp_path, "queue_outcomes.jsonl", [{"cancelled": True}, {"response": {}}])
    _write(tmp_path, "outbound_requests.jsonl", [])
    _write(tmp_path, "campaign_events.jsonl", [])

    usage = measure_provider_usage(tmp_path, DECLARED)

    assert usage.outcome_rows == 2
    assert usage.provider_calls == 0
    assert usage.provider_free is False, "rows exist, so this is not a clean zero"


def test_no_runner_certifies_itself_against_its_own_constant():
    """A ratchet for the exact shape of the original defect.

    ``PROVIDER_CALLS = 0`` at module scope, then ``PROVIDER_CALLS == 0`` in a
    health block, is an expression that cannot evaluate to False. It sat beside
    genuine checks and was indistinguishable from one.
    """
    import ast
    from pathlib import Path

    import agent_evolve

    repo_root = Path(agent_evolve.__file__).parents[2]
    offenders = []
    for path in sorted((repo_root / "examples").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        literal_constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if not literal_constants:
            continue
        # Only dict *values* matter. A comparison inside `if ... raise` is a
        # defensive guard against a constant drifting, which is legitimate; a
        # comparison serialized into a receipt is evidence, and evidence that
        # cannot come out False is not evidence.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(value, ast.Compare) and len(value.comparators) == 1):
                    continue
                if not isinstance(value.ops[0], ast.Eq):
                    continue
                left, right = value.left, value.comparators[0]
                if not (isinstance(left, ast.Name) and isinstance(right, ast.Constant)):
                    continue
                if literal_constants.get(left.id, object()) != right.value:
                    continue
                field = key.value if isinstance(key, ast.Constant) else "<computed>"
                offenders.append(
                    f"{path.relative_to(repo_root)}:{value.lineno} "
                    f'"{field}": {left.id} == {right.value!r} (always True)'
                )

    assert offenders == [], (
        "these seal a constant compared to its own value as a receipt field, so "
        "they cannot fail; measure it instead (agent_evolve.provider_accounting):\n  "
        + "\n  ".join(offenders)
    )


def test_provider_free_cannot_be_satisfied_by_a_constant():
    """A ProviderUsage built with traffic can never report provider_free."""
    usage = ProviderUsage(
        provider_calls=0,
        outcome_rows=0,
        outbound_rows=3,
        input_tokens=0,
        output_tokens=0,
        cost_usd="0",
        journal_rows={"outbound": 3},
    )
    assert usage.provider_free is False, "outbound traffic must defeat the claim"
