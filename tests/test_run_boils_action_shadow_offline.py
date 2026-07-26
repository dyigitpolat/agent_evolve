"""Provider- and evaluator-free gates for the frozen BOiLS action shadow."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse
from examples.development import run_boils_action_shadow as shadow
from examples.development import run_boils_action_shadow_scorer as scorer
from examples.development import run_boils_agentic_pilot as v1
from examples.development import run_boils_local_oracle as oracle


def _valid_output(
    output_type: type[shadow.ActionRankingResponse],
    *,
    ranking: list[str] | None = None,
) -> shadow.ActionRankingResponse:
    actions = list(shadow.ACTION_IDS)
    current = next(action for action in actions if action not in output_type.expected_actions)
    legal = [action for action in actions if action != current]
    ordered = legal if ranking is None else ranking
    return output_type.model_validate(
        {
            "ranking": ordered,
            "predictions": {
                action: {
                    "total_lut_count": {
                        "decrease": 0.34,
                        "same": 0.33,
                        "increase": 0.33,
                    },
                    "total_levels": {
                        "decrease": 0.33,
                        "same": 0.34,
                        "increase": 0.33,
                    },
                }
                for action in legal
            },
        },
        strict=True,
    )


def _record(
    condition: str,
    path: int,
    output: shadow.ActionRankingResponse,
    ordinal: int,
) -> dict[str, object]:
    task = next(
        task
        for task in shadow.FROZEN_TASKS
        if task.condition == condition and task.path == path
    )
    dumped = output.model_dump(mode="json")
    assert ordinal == task.ordinal
    return {
        "schema_version": 1,
        "status": "succeeded",
        "ordinal": ordinal,
        "condition": condition,
        "path": path,
        "call_id": task.call_id,
        "prompt_sha256": task.prompt_sha256,
        "schema_sha256": task.schema_sha256,
        "output": dumped,
        "output_sha256": shadow._sha256_json(dumped),
        "requested_model": shadow.MODEL,
        "resolved_model": shadow.MODEL,
        "resolved_provider": shadow.RESOLVED_PROVIDER,
        "attempt_count": 1,
        "cost_usd": "0.001",
        "output_contract_valid": True,
        "model_provider_attempt_gate": True,
        "reported_cost_present": True,
        "valid_for_scoring": True,
    }


def _receipt(records: list[dict[str, object]]) -> shadow.ProposalClosureReceipt:
    return shadow._receipt_from_closure_event(shadow._closure_event(records))


def _mark_failed(record: dict[str, object]) -> None:
    keep = {
        "schema_version",
        "ordinal",
        "condition",
        "path",
        "call_id",
        "prompt_sha256",
        "schema_sha256",
    }
    identity = {key: value for key, value in record.items() if key in keep}
    record.clear()
    record.update(
        {
            **identity,
            "status": "failed",
            "failure_type": "OfflineMissing",
            "safe_message": "offline missing response",
            "valid_for_scoring": False,
        }
    )


def _oracle_ranked_records() -> list[dict[str, object]]:
    table = scorer.load_oracle_table()
    records: list[dict[str, object]] = []
    for task in shadow.FROZEN_TASKS:
        path_hvs = {
            action: oracle.hypervolume(
                [table.parent, scorer._objective(row)],
                shadow.REFERENCE_POINT,
            )
            for action, row in table.rows_by_path[task.path].items()
        }
        ranking = sorted(
            path_hvs,
            key=lambda action: (-path_hvs[action], shadow.ACTION_IDS.index(action)),
        )
        records.append(
            _record(
                task.condition,
                task.path,
                _valid_output(task.output_type, ranking=ranking),
                task.ordinal,
            )
        )
    return records


def test_corrected_order_roles_and_prompt_interventions_are_exact_and_nonleaking():
    assert tuple((task.condition, task.path) for task in shadow.FROZEN_TASKS) == shadow.CORRECTED_TASK_ORDER
    assert [task.schedule_sha256 for task in shadow.FROZEN_TASKS] == sorted(
        task.schedule_sha256 for task in shadow.FROZEN_TASKS
    )
    assert shadow.ROLE_BY_PATH == {
        18: "balanced Pareto contribution",
        12: "area minimization subject to total_levels<=69",
        1: "depth minimization subject to total_lut_count<=7944",
        7: "action-family exploration",
    }
    for task in shadow.FROZEN_TASKS:
        if task.condition == "names_only":
            assert '"action_cards"' not in task.prompt
            assert '"assigned_portfolio_role"' not in task.prompt
            assert '"preoracle_machine_evidence"' not in task.prompt
        elif task.condition == "action_cards_niches":
            assert '"action_cards"' in task.prompt
            assert '"assigned_portfolio_role"' in task.prompt
            assert '"preoracle_machine_evidence"' not in task.prompt
        else:
            assert '"action_cards"' in task.prompt
            assert '"preoracle_machine_evidence"' in task.prompt
            assert '"deterministic_coordinator"' in task.prompt
        for oracle_only_number in ("7745", "7906", "8016", "8480", "8684"):
            assert oracle_only_number not in task.prompt


def test_catalog_evidence_and_all_sealed_input_hashes_reproduce():
    bound = shadow.hash_bind_inputs()
    assert {name: row["sha256"] for name, row in bound.items()} == {
        name: expected for name, (_, expected) in shadow.EXPECTED_INPUT_SHA256.items()
    }
    assert tuple(row["action_id"] for row in shadow.ACTION_CARDS) == shadow.ACTION_IDS
    assert {
        action
        for action, row in shadow.CARD_BY_ACTION.items()
        if row["extended_action"] is True
    } == set(shadow.EXTENDED_ACTIONS)
    interaction = shadow.PREORACLE_EVIDENCE["interaction_facts"][0]
    assert interaction["not_a_single_operation_contrast"] is True
    assert interaction["residual"] == {"total_lut_count": -4, "total_levels": 1}


def test_response_contract_rejects_current_duplicate_nonfinite_and_bad_probability_sum():
    good = _valid_output(shadow.Path1RankingResponse)
    assert type(good) is shadow.Path1RankingResponse
    payload = good.model_dump(mode="python")

    duplicate = json.loads(json.dumps(payload))
    duplicate["ranking"][1] = duplicate["ranking"][0]
    with pytest.raises(ValidationError):
        shadow.Path1RankingResponse.model_validate(duplicate, strict=True)

    current = json.loads(json.dumps(payload))
    current["ranking"][0] = shadow.v2.PARENT_C["sequence"][1]
    with pytest.raises(ValidationError):
        shadow.Path1RankingResponse.model_validate(current, strict=True)

    nonfinite = json.loads(json.dumps(payload))
    first_action = payload["ranking"][0]
    nonfinite["predictions"][first_action]["total_levels"]["same"] = math.inf
    with pytest.raises(ValidationError):
        shadow.Path1RankingResponse.model_validate(nonfinite, strict=True)

    bad_sum = json.loads(json.dumps(payload))
    bad_sum["predictions"][first_action]["total_lut_count"] = {
        "decrease": 0.5,
        "same": 0.5,
        "increase": 0.5,
    }
    with pytest.raises(ValidationError):
        shadow.Path1RankingResponse.model_validate(bad_sum, strict=True)


def test_provider_json_schemas_exactly_enumerate_each_path_action_universe():
    for path, output_type in shadow.OUTPUT_TYPE_BY_PATH.items():
        legal = set(shadow._noncurrent_actions(path))
        schema = output_type.model_json_schema()
        ranking = schema["properties"]["ranking"]
        assert ranking["minItems"] == ranking["maxItems"] == 10
        assert ranking["uniqueItems"] is True
        assert set(ranking["items"]["enum"]) == legal
        prediction_ref = schema["properties"]["predictions"]["$ref"]
        prediction_name = prediction_ref.removeprefix("#/$defs/")
        predictions = schema["$defs"][prediction_name]
        assert predictions["additionalProperties"] is False
        assert set(predictions["required"]) == legal
        assert set(predictions["properties"]) == legal


def test_proposal_module_has_no_oracle_or_scorer_capability_at_import_time():
    code = """
import sys
from examples.development import run_boils_action_shadow as proposal
assert 'examples.development.run_boils_action_shadow_scorer' not in sys.modules
assert 'examples.development.run_boils_local_oracle' not in sys.modules
assert 'oracle' not in proposal.__dict__
for name in ('OracleTable', 'load_oracle_table', 'score_shadow'):
    assert not hasattr(proposal, name), name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=shadow.AGENT_EVOLVE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_noncanonical_cli_module_and_delayed_scorer_share_one_composition_root():
    code = f"""
import importlib.util
import sys
from pathlib import Path

source = Path({str(shadow.Path(shadow.__file__).resolve())!r})
name = 'shadow_cli_sim'
spec = importlib.util.spec_from_file_location(name, source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
assert module.CANONICAL_MODULE_NAME not in sys.modules
module._bind_canonical_composition_root()
assert sys.modules[module.CANONICAL_MODULE_NAME] is module
scorer = module._load_frozen_post_closure_scorer(
    module.SCORER_SOURCE_PATH,
    expected_sha256=module._source_hashes()['post_closure_scorer'],
)
assert scorer.proposal is module
receipt = module.ProposalClosureReceipt(
    queue_closed=True,
    terminal_logical_calls=12,
    terminal_response_hashes=tuple('0' * 64 for _ in range(12)),
    closure_event_sha256='1' * 64,
)
assert type(receipt) is scorer.proposal.ProposalClosureReceipt
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=shadow.AGENT_EVOLVE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_coordinator_exhausts_cartesian_policy_and_is_byte_deterministic():
    rankings = {
        path: list(_valid_output(shadow.OUTPUT_TYPE_BY_PATH[path]).ranking)
        for path in shadow.PATHS
    }
    first = scorer.coordinate_evidence_portfolio(rankings)
    second = scorer.coordinate_evidence_portfolio(rankings)
    assert first == second
    assert 0 < first["feasible_count"] <= 10_000
    assert first["family_count"] >= 3
    assert first["extended_count"] >= 1
    actions = first["actions_by_path"]
    assert tuple(int(path) for path in actions) == shadow.PATHS


def test_oracle_scorer_rederives_all_historical_gates_and_front():
    table = scorer.load_oracle_table()
    assert all(table.reproduction_gates.values())
    assert len(table.random_policy_hvs) == 10_000
    assert table.random_policy_hvs[0] == 168
    assert table.random_policy_hvs[-1] == 700
    assert {
        (row["index"], row["replacement"], scorer._objective(row))
        for row in table.raw_front
    } == {
        (18, "blut", (7745, 69)),
        (1, "fraig", (7906, 68)),
        (12, "dsdb", (8016, 67)),
        (1, "sopb", (8480, 60)),
        (7, "sopb", (8684, 59)),
    }


def test_oracle_loader_is_poisoned_until_durable_proposal_closure():
    touched = False

    def poison() -> scorer.OracleTable:
        nonlocal touched
        touched = True
        raise AssertionError("oracle loader must not be touched")

    with pytest.raises(RuntimeError, match="before proposal phase closure"):
        scorer.score_shadow(
            _oracle_ranked_records(),
            proposal_receipt=None,
            oracle_loader=poison,
        )
    assert touched is False


def test_scoring_definitions_and_equal_challenger_decision_are_exact():
    records = _oracle_ranked_records()
    scored = scorer.score_shadow(
        records,
        proposal_receipt=_receipt(records),
    )
    assert all(scored["oracle_reproduction_gates"].values())
    names = scored["conditions"]["names_only"]
    cards = scored["conditions"]["action_cards_niches"]
    evidence = scored["conditions"]["evidence_portfolio"]
    assert names["selected_archive_hypervolume"] == 688
    assert cards["selected_archive_hypervolume"] == 688
    assert evidence["selected_archive_hypervolume"] == 688
    assert scored["provider_block"]["successful_response_cost_usd"] == "0.012"
    assert scored["decision"] == {
        "passing_challengers": [],
        "advanced_condition": None,
        "kill_unconstrained_low_level_llm_ranking": True,
        "decision_rule_applied": True,
    }
    for condition in shadow.CONDITIONS:
        row = scored["conditions"][condition]
        assert 0 <= row["mean_multiclass_brier"] <= 2
        assert 0 <= row["mean_categorical_accuracy"] <= 1
        assert 0 <= row["mean_ndcg"] <= 1


def test_incomplete_names_control_does_not_trigger_a_kill_decision():
    records = _oracle_ranked_records()
    _mark_failed(records[0])
    scored = scorer.score_shadow(records, proposal_receipt=_receipt(records))
    assert scored["conditions"]["names_only"]["complete"] is False
    assert scored["decision"]["decision_rule_applied"] is False
    assert scored["decision"]["kill_unconstrained_low_level_llm_ranking"] is None


def test_incomplete_challenger_makes_the_three_condition_decision_inconclusive():
    records = _oracle_ranked_records()
    missing = next(
        row
        for row in records
        if row["condition"] == "evidence_portfolio" and row["path"] == 18
    )
    _mark_failed(missing)
    scored = scorer.score_shadow(records, proposal_receipt=_receipt(records))
    assert scored["conditions"]["names_only"]["complete"] is True
    assert scored["conditions"]["action_cards_niches"]["complete"] is True
    assert scored["conditions"]["evidence_portfolio"]["complete"] is False
    assert scored["decision"]["decision_rule_applied"] is False
    assert scored["decision"]["passing_challengers"] == []
    assert scored["decision"]["advanced_condition"] is None
    assert scored["decision"]["kill_unconstrained_low_level_llm_ranking"] is None


def test_missing_reported_cost_is_inadmissible_and_makes_decision_inconclusive():
    records = _oracle_ranked_records()
    records[0]["cost_usd"] = None
    records[0]["reported_cost_present"] = False
    records[0]["valid_for_scoring"] = False
    scored = scorer.score_shadow(records, proposal_receipt=_receipt(records))
    assert scored["provider_block"]["successful_responses_without_cost"] == 1
    assert scored["provider_block"]["cost_gate_passed"] is False
    assert scored["provider_block"]["failed_or_invalid_cells"] == 1
    assert scored["conditions"][str(records[0]["condition"])]["complete"] is False
    assert scored["decision"] == {
        "passing_challengers": [],
        "advanced_condition": None,
        "kill_unconstrained_low_level_llm_ranking": None,
        "decision_rule_applied": False,
    }


class _FakePredictor:
    def __init__(self) -> None:
        self.call_ids: list[str] = []

    async def __call__(self, request):
        self.call_ids.append(request.call_id.value)
        await asyncio.sleep(0)
        response = StructuredGenerationResponse(
            value=_valid_output(request.output_type),
            requested_model=shadow.MODEL,
            resolved_model=shadow.MODEL,
            resolved_provider=shadow.RESOLVED_PROVIDER,
            provider_response_id=f"provider_{request.call_id.value}",
            finish_reason="tool_call",
            input_tokens=100,
            output_tokens=200,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=1,
        )
        return AttemptedStructuredGenerationResponse(response=response, attempt_count=1)


def _write_fake_proposal_logs(
    tmp_path: Path,
) -> tuple[tuple[dict[str, object], ...], _FakePredictor]:
    response_writer = v1.DurableJsonlWriter(tmp_path / "responses.jsonl")
    event_writer = v1.DurableJsonlWriter(tmp_path / "events.jsonl")
    event_writer.write(
        {
            "schema_version": 1,
            "event_type": "proposal_phase_started",
            "recorded_at_utc": shadow._utc_now(),
            "logical_calls": 12,
            "oracle_parser_constructed": False,
        }
    )
    predictor = _FakePredictor()
    records = asyncio.run(
        shadow.execute_proposal_tasks(
            predictor=predictor,
            response_writer=response_writer,
            event_writer=event_writer,
        )
    )
    event_writer.write(shadow._closure_event(records))
    response_writer.close()
    event_writer.close()
    queue_writer = v1.DurableJsonlWriter(tmp_path / "queue.jsonl")
    for task in shadow.FROZEN_TASKS:
        queue_writer.write(
            {"task_id": task.call_id, "status": "succeeded", "attempts": [{}]}
        )
    queue_writer.close()
    return records, predictor


def test_fake_proposal_executes_all_twelve_once_and_durably_records_terminals(tmp_path: Path):
    records, predictor = _write_fake_proposal_logs(tmp_path)
    assert len(records) == 12
    assert predictor.call_ids == [task.call_id for task in shadow.FROZEN_TASKS]
    assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 12
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(row["event_type"] == "shadow_call_submitted" for row in events) == 12
    assert sum(row["event_type"] == "shadow_call_terminal" for row in events) == 12
    assert all(row["valid_for_scoring"] is True for row in records)
    replayed, receipt = shadow.verify_durable_proposal_logs(
        responses_path=tmp_path / "responses.jsonl",
        events_path=tmp_path / "events.jsonl",
        queue_path=tmp_path / "queue.jsonl",
    )
    assert tuple(row["call_id"] for row in replayed) == tuple(
        task.call_id for task in shadow.FROZEN_TASKS
    )
    assert receipt.queue_closed is True


@pytest.mark.parametrize("mutation", ("drop_start", "append_after_close"))
def test_durable_replay_rejects_incomplete_or_postclosure_event_grammar(
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_fake_proposal_logs(tmp_path)
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if mutation == "drop_start":
        events = events[1:]
    else:
        events.append(
            {
                "schema_version": 1,
                "event_type": "shadow_call_terminal",
                "recorded_at_utc": shadow._utc_now(),
            }
        )
    events_path.write_text(
        "".join(shadow._canonical_json(event) + "\n" for event in events),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="event log"):
        shadow.verify_durable_proposal_logs(
            responses_path=tmp_path / "responses.jsonl",
            events_path=events_path,
            queue_path=tmp_path / "queue.jsonl",
        )


def test_scorer_rejects_tampered_provider_and_prompt_bindings_before_oracle_access():
    records = _oracle_ranked_records()
    receipt = _receipt(records)
    records[0]["resolved_provider"] = "unexpected"
    touched = False

    def poison() -> scorer.OracleTable:
        nonlocal touched
        touched = True
        raise AssertionError("oracle must remain unopened on replay failure")

    with pytest.raises(RuntimeError, match="stored provider/scoring gates"):
        scorer.score_shadow(
            records,
            proposal_receipt=receipt,
            oracle_loader=poison,
        )
    assert touched is False
