"""Post-closure scorer for the BOiLS action-ranking shadow.

This module is intentionally not imported by the live proposal composition root
until its queue and all durable proposal logs are closed and replay-verified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
import itertools
import json
import math
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

from examples.development import run_boils_action_shadow as proposal
from examples.development import run_boils_local_oracle as oracle


@dataclass(frozen=True, slots=True)
class OracleTable:
    parent: tuple[int, int]
    rows_by_path: Mapping[int, Mapping[str, Mapping[str, object]]]
    random_policy_hvs: tuple[int, ...]
    raw_front: tuple[Mapping[str, object], ...]
    reproduction_gates: Mapping[str, bool]


def _objective(row: Mapping[str, object]) -> tuple[int, int]:
    objectives = row.get("objectives")
    if row.get("valid") is not True or type(objectives) is not dict:
        raise RuntimeError("sealed oracle row is not a valid objective observation")
    return (
        oracle._as_exact_int(objectives.get("total_lut_count"), "shadow LUT count"),
        oracle._as_exact_int(objectives.get("total_levels"), "shadow levels"),
    )


def _dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left != right


def _front(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        row
        for row in rows
        if not any(
            other is not row and _dominates(_objective(other), _objective(row))
            for other in rows
        )
    )


def load_oracle_table(
    summary_path: Path = proposal.EXPECTED_INPUT_SHA256["oracle_summary"][0],
) -> OracleTable:
    """Open the sealed outcome table only in the post-closure scorer process phase."""

    payload = summary_path.read_bytes()
    if proposal._sha256_bytes(payload) != proposal.EXPECTED_INPUT_SHA256["oracle_summary"][1]:
        raise RuntimeError("sealed shadow oracle summary hash changed")
    summary = json.loads(payload)
    if type(summary) is not dict or summary.get("status") != "succeeded":
        raise RuntimeError("sealed shadow oracle summary is not successful")
    outcomes = summary.get("outcomes_frozen_order")
    if type(outcomes) is not list or len(outcomes) != 41:
        raise RuntimeError("sealed shadow oracle must contain C plus forty children")
    parent = _objective(outcomes[0])
    rows_by_path: dict[int, dict[str, Mapping[str, object]]] = {
        path: {} for path in proposal.PATHS
    }
    for row in outcomes[1:]:
        path = row.get("index")
        action = row.get("replacement")
        if path not in proposal.PATHS or action not in proposal.ACTION_IDS:
            raise RuntimeError("sealed shadow oracle child escaped its legal universe")
        rows_by_path[int(path)][str(action)] = row
    if any(
        set(rows_by_path[path]) != set(proposal._noncurrent_actions(path))
        for path in proposal.PATHS
    ):
        raise RuntimeError("sealed shadow oracle path/action matrix is incomplete")
    all_points = [
        parent,
        *(
            _objective(row)
            for path in proposal.PATHS
            for row in rows_by_path[path].values()
        ),
    ]
    oracle_hv = oracle.hypervolume(all_points, proposal.REFERENCE_POINT)
    v2_points = [
        parent,
        *(
            _objective(rows_by_path[path][action])
            for path, action in proposal.V2_ACTION_BY_PATH.items()
        ),
    ]
    v2_hv = oracle.hypervolume(v2_points, proposal.REFERENCE_POINT)
    random_hvs = tuple(
        sorted(
            oracle.hypervolume(
                [
                    parent,
                    *(
                        _objective(rows_by_path[path][action])
                        for path, action in zip(
                            proposal.PATHS, actions, strict=True
                        )
                    ),
                ],
                proposal.REFERENCE_POINT,
            )
            for actions in itertools.product(
                *(proposal._noncurrent_actions(path) for path in proposal.PATHS)
            )
        )
    )
    if len(random_hvs) != 10_000:
        raise RuntimeError("exact shadow random-policy support changed")
    q1 = random_hvs[2_499] * 0.25 + random_hvs[2_500] * 0.75
    median = (random_hvs[4_999] + random_hvs[5_000]) / 2
    q3 = random_hvs[7_499] * 0.75 + random_hvs[7_500] * 0.25
    mean = sum(random_hvs) / len(random_hvs)
    v2_percentile = 100 * sum(value <= v2_hv for value in random_hvs) / len(
        random_hvs
    )
    gates = {
        "parent_c_exact": parent == proposal.PARENT_OBJECTIVES,
        "oracle_hv_exact": oracle_hv == 700,
        "v2_hv_exact": v2_hv == 206,
        "random_q1_exact": q1 == 206,
        "random_median_exact": median == 232,
        "random_q3_exact": q3 == 324,
        "random_mean_exact": mean == 281.542,
        "v2_inclusive_percentile_exact": v2_percentile == 27.09,
    }
    if not all(gates.values()):
        raise RuntimeError("sealed shadow oracle failed a historical reproduction gate")
    return OracleTable(
        parent=parent,
        rows_by_path=rows_by_path,
        random_policy_hvs=random_hvs,
        raw_front=_front(tuple(outcomes[1:])),
        reproduction_gates=gates,
    )


def _output(record: Mapping[str, object]) -> proposal.ActionRankingResponse:
    path = int(record["path"])
    output = record.get("output")
    if type(output) is not dict:
        raise RuntimeError("complete shadow record has no typed output")
    return proposal.OUTPUT_TYPE_BY_PATH[path].model_validate(output, strict=True)


def _family(action: str) -> str:
    return str(proposal.CARD_BY_ACTION[action]["family"])


def coordinate_evidence_portfolio(
    rankings: Mapping[int, Sequence[str]],
) -> dict[str, object]:
    if set(rankings) != set(proposal.PATHS):
        raise ValueError("coordinator requires all four frozen paths")
    rank_maps = {
        path: {action: rank for rank, action in enumerate(rankings[path], start=1)}
        for path in proposal.PATHS
    }
    best_sum: int | None = None
    ties: list[tuple[str, tuple[str, ...]]] = []
    feasible = 0
    for actions in itertools.product(*(rankings[path] for path in proposal.PATHS)):
        families = {_family(action) for action in actions}
        if len(families) < 3 or not any(
            action in proposal.EXTENDED_ACTIONS for action in actions
        ):
            continue
        feasible += 1
        rank_sum = sum(
            rank_maps[path][action]
            for path, action in zip(proposal.PATHS, actions, strict=True)
        )
        if best_sum is None or rank_sum < best_sum:
            best_sum = rank_sum
            ties = []
        if rank_sum == best_sum:
            payload = {
                "schema": "boils_shadow_portfolio_tiebreak_v1",
                "pairs": [
                    {"path": f"$.sequence[{path}]", "action": action}
                    for path, action in zip(proposal.PATHS, actions, strict=True)
                ],
            }
            ties.append((proposal._sha256_json(payload), tuple(actions)))
    if best_sum is None or not ties:
        raise RuntimeError("shadow portfolio coordinator found no feasible tuple")
    digest, chosen = min(ties, key=lambda row: row[0])
    families = {_family(action) for action in chosen}
    return {
        "feasible_count": feasible,
        "minimum_rank_sum": best_sum,
        "minimum_sum_tie_count": len(ties),
        "chosen_digest": digest,
        "actions_by_path": {
            str(path): action
            for path, action in zip(proposal.PATHS, chosen, strict=True)
        },
        "families": sorted(families),
        "family_count": len(families),
        "extended_count": sum(
            action in proposal.EXTENDED_ACTIONS for action in chosen
        ),
    }


def _average_descending_ranks(values: Mapping[str, int]) -> dict[str, float]:
    return {
        action: 1
        + sum(other > value for other in values.values())
        + (sum(other == value for other in values.values()) - 1) / 2
        for action, value in values.items()
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(
        a * b for a, b in zip(centered_left, centered_right, strict=True)
    ) / denominator


def _ndcg(ranking: Sequence[str], gains: Mapping[str, int]) -> float:
    def dcg(order: Sequence[str]) -> float:
        return sum(
            gains[action] / math.log2(rank + 1)
            for rank, action in enumerate(order, start=1)
        )

    ideal = sorted(
        gains,
        key=lambda action: (-gains[action], proposal.ACTION_IDS.index(action)),
    )
    denominator = dcg(ideal)
    return 1.0 if denominator == 0 else dcg(ranking) / denominator


def _direction(value: int, parent: int) -> proposal.DirectionLabel:
    if value < parent:
        return "decrease"
    if value > parent:
        return "increase"
    return "same"


def _recall(
    ranking: Sequence[str], front_actions: set[str], k: int
) -> float | None:
    if not front_actions:
        return None
    return len(set(ranking[:k]) & front_actions) / len(front_actions)


def _score_path(
    *,
    path: int,
    selected_action: str,
    output: proposal.ActionRankingResponse,
    table: OracleTable,
) -> dict[str, object]:
    rows = table.rows_by_path[path]
    parent_hv = oracle.hypervolume([table.parent], proposal.REFERENCE_POINT)
    path_hv = {
        action: oracle.hypervolume(
            [table.parent, _objective(row)], proposal.REFERENCE_POINT
        )
        for action, row in rows.items()
    }
    gains = {action: value - parent_hv for action, value in path_hv.items()}
    selected_hv = path_hv[selected_action]
    rank = 1 + sum(value > selected_hv for value in path_hv.values())
    tie_count = sum(value == selected_hv for value in path_hv.values())
    child_front = _front(tuple(rows.values()))
    with_parent_front = _front(
        (
            {
                "valid": True,
                "objectives": {
                    "total_lut_count": table.parent[0],
                    "total_levels": table.parent[1],
                },
                "replacement": "__parent_c__",
            },
            *rows.values(),
        )
    )
    child_front_actions = {str(row["replacement"]) for row in child_front}
    with_parent_actions = {
        str(row["replacement"])
        for row in with_parent_front
        if row.get("replacement") != "__parent_c__"
    }
    predicted_ranks = {
        action: index for index, action in enumerate(output.ranking, start=1)
    }
    true_ranks = _average_descending_ranks(gains)
    spearman = _pearson(
        [float(predicted_ranks[action]) for action in output.ranking],
        [float(true_ranks[action]) for action in output.ranking],
    )
    categories = ("decrease", "same", "increase")
    calibration: dict[str, object] = {}
    for objective_index, objective_name in enumerate(
        ("total_lut_count", "total_levels")
    ):
        cells: list[dict[str, object]] = []
        for action in output.ranking:
            observed = _direction(
                _objective(rows[action])[objective_index], table.parent[objective_index]
            )
            action_prediction = getattr(output.predictions, action)
            probabilities = getattr(action_prediction, objective_name).model_dump(
                mode="python"
            )
            predicted = min(
                categories,
                key=lambda category: (
                    -float(probabilities[category]),
                    categories.index(category),
                ),
            )
            brier = sum(
                (float(probabilities[category]) - (category == observed)) ** 2
                for category in categories
            )
            cells.append(
                {
                    "action": action,
                    "observed": observed,
                    "predicted_argmax": predicted,
                    "correct": predicted == observed,
                    "brier": brier,
                    "probabilities": probabilities,
                }
            )
        calibration[objective_name] = {
            "mean_multiclass_brier": sum(
                float(cell["brier"]) for cell in cells
            )
            / 10,
            "categorical_accuracy": sum(
                bool(cell["correct"]) for cell in cells
            )
            / 10,
            "cells": cells,
        }
    return {
        "path": path,
        "selected_action": selected_action,
        "selected_objectives": {
            "total_lut_count": _objective(rows[selected_action])[0],
            "total_levels": _objective(rows[selected_action])[1],
        },
        "selected_same_path_hv": selected_hv,
        "same_path_hv_competition_rank": rank,
        "same_path_hv_tie_count": tie_count,
        "same_path_hv_regret": max(path_hv.values()) - selected_hv,
        "selected_on_primary_child_front": selected_action in child_front_actions,
        "selected_on_with_parent_front": selected_action in with_parent_actions,
        "primary_child_front_actions": sorted(
            child_front_actions, key=proposal.ACTION_IDS.index
        ),
        "with_parent_front_actions": sorted(
            with_parent_actions, key=proposal.ACTION_IDS.index
        ),
        "top_k_primary_front_recall": {
            str(k): _recall(output.ranking, child_front_actions, k)
            for k in (1, 3, 5)
        },
        "top_k_with_parent_front_recall": {
            str(k): _recall(output.ranking, with_parent_actions, k)
            for k in (1, 3, 5)
        },
        "spearman_predicted_vs_true_hv_rank": spearman,
        "ndcg_linear_same_path_hv_gain": _ndcg(output.ranking, gains),
        "calibration": calibration,
    }


def score_shadow(
    records: Sequence[Mapping[str, object]],
    *,
    proposal_receipt: proposal.ProposalClosureReceipt,
    oracle_loader: Callable[[], OracleTable] = load_oracle_table,
) -> dict[str, object]:
    if type(proposal_receipt) is not proposal.ProposalClosureReceipt:
        raise RuntimeError(
            "oracle scorer cannot be constructed before proposal phase closure"
        )
    normalized = proposal._normalize_terminal_records(records)
    expected_hashes = tuple(proposal._sha256_json(row) for row in normalized)
    if proposal_receipt.terminal_response_hashes != expected_hashes:
        raise RuntimeError("proposal closure receipt does not bind scoring records")
    table = oracle_loader()
    by_condition_path = {
        (str(row["condition"]), int(row["path"])): row for row in normalized
    }
    if len(by_condition_path) != 12:
        raise RuntimeError("shadow terminal records contain duplicate condition/path cells")
    successful_costs = [
        Decimal(str(row["cost_usd"]))
        for row in normalized
        if row.get("status") == "succeeded" and row.get("cost_usd") is not None
    ]
    successful_count = sum(row.get("status") == "succeeded" for row in normalized)
    block_cost = sum(successful_costs, Decimal(0))
    cost_gate = (
        len(successful_costs) == successful_count
        and block_cost <= proposal.MAX_SUCCESSFUL_RESPONSE_COST_USD
    )
    conditions: dict[str, dict[str, object]] = {}
    for condition in proposal.CONDITIONS:
        condition_records = {
            path: by_condition_path[(condition, path)] for path in proposal.PATHS
        }
        complete = all(
            row.get("valid_for_scoring") is True
            for row in condition_records.values()
        )
        result: dict[str, object] = {
            "complete": complete,
            "valid_response_paths": [
                path
                for path, row in condition_records.items()
                if row.get("valid_for_scoring") is True
            ],
        }
        if complete:
            outputs = {
                path: _output(row) for path, row in condition_records.items()
            }
            rankings = {
                path: list(outputs[path].ranking) for path in proposal.PATHS
            }
            if condition == "evidence_portfolio":
                coordinator = coordinate_evidence_portfolio(rankings)
                selected = {
                    int(path): str(action)
                    for path, action in coordinator["actions_by_path"].items()
                }
            else:
                coordinator = None
                selected = {
                    path: str(rankings[path][0]) for path in proposal.PATHS
                }
            selected_hv = oracle.hypervolume(
                [
                    table.parent,
                    *(
                        _objective(table.rows_by_path[path][selected[path]])
                        for path in proposal.PATHS
                    ),
                ],
                proposal.REFERENCE_POINT,
            )
            path_scores = [
                _score_path(
                    path=path,
                    selected_action=selected[path],
                    output=outputs[path],
                    table=table,
                )
                for path in proposal.PATHS
            ]
            families = {_family(action) for action in selected.values()}
            spearman_values = [
                float(row["spearman_predicted_vs_true_hv_rank"])
                for row in path_scores
                if row["spearman_predicted_vs_true_hv_rank"] is not None
            ]
            result.update(
                {
                    "selected_actions": {
                        str(path): selected[path] for path in proposal.PATHS
                    },
                    "selected_archive_includes_parent_c": True,
                    "selected_archive_hypervolume": selected_hv,
                    "exact_random_policy_inclusive_percentile": 100
                    * sum(value <= selected_hv for value in table.random_policy_hvs)
                    / len(table.random_policy_hvs),
                    "family_count": len(families),
                    "families": sorted(families),
                    "extended_action_count": sum(
                        action in proposal.EXTENDED_ACTIONS
                        for action in selected.values()
                    ),
                    "coordinator": coordinator,
                    "path_scores": path_scores,
                    "mean_ndcg": sum(
                        float(row["ndcg_linear_same_path_hv_gain"])
                        for row in path_scores
                    )
                    / 4,
                    "mean_spearman": (
                        None
                        if not spearman_values
                        else sum(spearman_values) / len(spearman_values)
                    ),
                    "mean_multiclass_brier": sum(
                        float(
                            row["calibration"][objective][
                                "mean_multiclass_brier"
                            ]
                        )
                        for row in path_scores
                        for objective in ("total_lut_count", "total_levels")
                    )
                    / 8,
                    "mean_categorical_accuracy": sum(
                        float(
                            row["calibration"][objective]["categorical_accuracy"]
                        )
                        for row in path_scores
                        for objective in ("total_lut_count", "total_levels")
                    )
                    / 8,
                }
            )
        conditions[condition] = result
    names = conditions["names_only"]
    decision_rule_applied = cost_gate and all(
        conditions[condition].get("complete") is True
        for condition in proposal.CONDITIONS
    )
    passing: list[str] = []
    if decision_rule_applied:
        for challenger in ("action_cards_niches", "evidence_portfolio"):
            result = conditions[challenger]
            if (
                int(result["selected_archive_hypervolume"])
                > int(names["selected_archive_hypervolume"])
                and int(result["selected_archive_hypervolume"])
                > proposal.RANDOM_POLICY_MEDIAN
                and int(result["family_count"]) >= int(names["family_count"])
            ):
                passing.append(challenger)
    advanced = None
    if passing:
        advanced = sorted(
            passing,
            key=lambda name: (
                -int(conditions[name]["selected_archive_hypervolume"]),
                -float(conditions[name]["mean_ndcg"]),
                float(conditions[name]["mean_multiclass_brier"]),
                name,
            ),
        )[0]
    raw_front = [
        {
            "path": row["index"],
            "action": row["replacement"],
            "objectives": copy.deepcopy(row["objectives"]),
            "boils_configuration_sha256": row["boils_configuration_sha256"],
        }
        for row in sorted(
            table.raw_front,
            key=lambda row: (*_objective(row), int(row["index"])),
        )
    ]
    return {
        "schema_version": 1,
        "status": "scored",
        "completed_at_utc": proposal._utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Post-hoc log2 prompt/workflow diagnostic with one realization per cell; "
            "not held-out, optimizer, memory-utility, SOTA, or wall-clock evidence."
        ),
        "oracle_reproduction_gates": copy.deepcopy(
            dict(table.reproduction_gates)
        ),
        "historical": {
            "v2_hypervolume": 206,
            "oracle_hypervolume": 700,
            "random_q1_median_mean_q3": [206, 232, 281.542, 324],
            "v2_inclusive_percentile": 27.09,
        },
        "raw_oracle_front": raw_front,
        "conditions": conditions,
        "provider_block": {
            "logical_calls": 12,
            "successful_calls": successful_count,
            "successful_response_cost_usd": str(block_cost),
            "successful_responses_without_cost": successful_count
            - len(successful_costs),
            "cost_gate_passed": cost_gate,
            "failed_or_invalid_cells": sum(
                row.get("valid_for_scoring") is not True for row in normalized
            ),
        },
        "decision": {
            "passing_challengers": passing,
            "advanced_condition": advanced,
            "kill_unconstrained_low_level_llm_ranking": (
                None if not decision_rule_applied else advanced is None
            ),
            "decision_rule_applied": decision_rule_applied,
        },
        "limitations": [
            "The experiment was designed after observing the log2 local oracle.",
            "Condition B changes cards and niches together.",
            "Condition C changes evidence and coordination together.",
            "The evidence condition manually injects sealed facts and is not a retrieval or memory-utility test.",
            "One model realization per condition provides no reliability estimate.",
        ],
    }


__all__ = [
    "OracleTable",
    "coordinate_evidence_portfolio",
    "load_oracle_table",
    "score_shadow",
]
