"""Compare a frozen plan GNN with minimum-edit and linear ranking baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from osrd_ops_intelligence.features.replay_graph import ACTION_SUMMARY_FEATURE_NAMES
from osrd_ops_intelligence.models.replay_plan_gnn import ReplayPlanGNN
from osrd_ops_intelligence.training.train_replay_plan_gnn import _load_graph_partition


def _linear_features(graph: Any, signature: str) -> list[float]:
    action = json.loads(signature)
    node = graph.x.to(torch.float64)
    node_features = torch.cat((node.mean(0), node.amax(0), node.std(0, unbiased=False)))
    edge = graph.edge_attr.to(torch.float64)
    if edge.numel():
        edge_features = torch.cat((edge.mean(0), edge.amax(0)))
    else:
        edge_features = torch.zeros(graph.edge_attr.shape[1] * 2, dtype=torch.float64)
    retimings = action.get("retimings", [])
    stops = action.get("stops", [])
    departure_time = float(action.get("departure_time_ms", 0)) / 1000 % 86400
    scalar_features = torch.tensor(
        [
            float(action.get("frequency_s", 0)) / 600,
            float(action.get("departure_phase_s", 0)) / 120,
            float(action.get("operating_window_s", 0)) / 1800,
            float(action.get("requested_service_count", 0)) / 8,
            len(stops) / 4,
            sum(float(stop.get("dwell_s", 0)) for stop in stops) / 600,
            len(action.get("route_via", [])) / 4,
            len(retimings) / 8,
            sum(abs(int(item.get("departure_shift_s", 0))) for item in retimings) / 2400,
            torch.sin(torch.tensor(2 * torch.pi * departure_time / 86400)).item(),
            torch.cos(torch.tensor(2 * torch.pi * departure_time / 86400)).item(),
        ],
        dtype=torch.float64,
    )
    return torch.cat((node_features, edge_features, scalar_features)).tolist()


def _edit_key(signature: str) -> tuple[float, float, int]:
    action = json.loads(signature)
    shifts = [int(item.get("departure_shift_s", 0)) for item in action.get("retimings", [])]
    frequency = int(action.get("frequency_s", 0))
    phase = int(action.get("departure_phase_s", 0))
    stops = action.get("stops", [])
    route_via = action.get("route_via", [])
    count = (
        sum(shift != 0 for shift in shifts)
        + int(frequency > 0 and frequency != 300)
        + int(phase != 0)
        + len(stops)
        + len(route_via)
    )
    magnitude = (
        sum(abs(shift) for shift in shifts) / 60
        + (abs(frequency - 300) / 60 if frequency > 0 else 0)
        + phase / 60
        + sum(1 + float(stop.get("dwell_s", 0)) / 60 for stop in stops)
        + len(route_via)
    )
    return float(count), magnitude, 0


def _normalized_average_ranks(keys: list[Any]) -> list[float]:
    """Map comparable keys to [0, 1] average ranks, preserving ties."""
    ordered = sorted(range(len(keys)), key=lambda index: keys[index])
    ranks = [0.0] * len(keys)
    denominator = max(len(keys) - 1, 1)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and keys[ordered[end]] == keys[ordered[start]]:
            end += 1
        normalized_rank = ((start + end - 1) / 2) / denominator
        for position in range(start, end):
            ranks[ordered[position]] = normalized_rank
        start = end
    return ranks


def _fit_linear_ranker(
    contexts: list[list[tuple[Any, tuple[float, float], str]]], alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[tuple[float, float]] = []
    for context in contexts:
        for graph, outcome, signature in context:
            features.append(_linear_features(graph, signature))
            targets.append((outcome[0], outcome[1] / 300.0))
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    normalized = (x - mean) / scale
    design = torch.cat((normalized, torch.ones((len(normalized), 1), dtype=torch.float64)), 1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * alpha
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, weights


def _linear_prediction(
    context: list[tuple[Any, tuple[float, float], str]],
    mean: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
) -> list[tuple[float, float]]:
    x = torch.tensor(
        [_linear_features(graph, signature) for graph, _, signature in context],
        dtype=torch.float64,
    )
    design = torch.cat((((x - mean) / scale), torch.ones((len(x), 1), dtype=torch.float64)), 1)
    return [tuple(row) for row in (design @ weights).tolist()]


def _load_gnn(path: Path) -> tuple[dict[str, Any], ReplayPlanGNN]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    model = ReplayPlanGNN(
        input_channels=int(architecture["input_channels"]),
        hidden_channels=int(architecture["hidden_channels"]),
        heads=int(architecture["heads"]),
        edge_channels=int(architecture["edge_channels"]),
        action_channels=int(architecture.get("action_channels", 0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _gnn_predictions(
    context: list[tuple[Any, tuple[float, float], str]],
    payload: dict[str, Any],
    model: ReplayPlanGNN,
) -> tuple[list[tuple[float, float]], list[float]]:
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32)
    scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32)
    graphs = []
    for graph, _, _ in context:
        graph.x = (graph.x - mean) / scale
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    with torch.inference_mode():
        action_features = getattr(batch, "action_features", None)
        if model.action_channels:
            if action_features is None or tuple(payload.get("action_summary_feature_names", [])) != ACTION_SUMMARY_FEATURE_NAMES:
                raise ValueError("Model action-summary features do not match the evaluator schema")
        outcomes, scores = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, action_features
        )
    return (
        [tuple(float(value) for value in row) for row in outcomes],
        [float(value) for value in scores],
    )


def _context_metrics(
    context: list[tuple[Any, tuple[float, float], str]],
    predictions: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    labels = [item[1] for item in context]
    metrics: dict[str, dict[str, float]] = {}
    for method, keys in predictions.items():
        selected = min(range(len(context)), key=lambda index: keys[index])
        best_conflicts = min(item[0] for item in labels)
        actual_order = sorted(
            range(len(context)),
            key=lambda index: (
                labels[index][0],
                labels[index][1],
                *_edit_key(context[index][2])[:2],
            ),
        )
        selected_actual = (
            labels[selected][0],
            labels[selected][1],
            *_edit_key(context[selected][2])[:2],
        )
        correct = comparable = 0
        for left in range(len(context)):
            for right in range(left + 1, len(context)):
                actual_left = (labels[left][0], labels[left][1])
                actual_right = (labels[right][0], labels[right][1])
                if actual_left == actual_right:
                    continue
                comparable += 1
                predicted_left, predicted_right = keys[left], keys[right]
                if predicted_left == predicted_right:
                    continue
                correct += int((predicted_left < predicted_right) == (actual_left < actual_right))
        oracle = (
            labels[actual_order[0]][0],
            labels[actual_order[0]][1],
            *_edit_key(context[actual_order[0]][2])[:2],
        )
        metrics[method] = {
            "pairwise_concordance": correct / max(comparable, 1),
            "correct_pair_count": float(correct),
            "comparable_pair_count": float(comparable),
            "selected_min_conflict_fraction": float(labels[selected][0] == best_conflicts),
            "top1_conflict_regret": float(labels[selected][0] - best_conflicts),
            "exact_lexicographic_objective_match": float(selected_actual == oracle),
        }
    return metrics


def _aggregate(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    methods = sorted(rows[0])
    result: dict[str, dict[str, float]] = {}
    for method in methods:
        aggregate = {
            metric: sum(row[method][metric] for row in rows) / len(rows)
            for metric in rows[0][method]
            if metric not in {"correct_pair_count", "comparable_pair_count", "pairwise_concordance"}
        }
        correct = sum(row[method]["correct_pair_count"] for row in rows)
        comparable = sum(row[method]["comparable_pair_count"] for row in rows)
        aggregate["correct_pair_count"] = correct
        aggregate["comparable_pair_count"] = comparable
        aggregate["pairwise_concordance"] = correct / max(comparable, 1)
        result[method] = aggregate
    return result


def _macro_average_groups(
    group_metrics: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    methods = sorted(group_metrics[0])
    return {
        method: {
            metric: sum(summary[method][metric] for summary in group_metrics)
            / len(group_metrics)
            for metric in group_metrics[0][method]
            if metric not in {"correct_pair_count", "comparable_pair_count"}
        }
        for method in methods
    }


def evaluate(
    model_path: Path,
    train_roots: list[Path],
    test_roots: list[Path],
    ridge_rank_blend_weight: float | None = None,
) -> dict[str, Any]:
    if ridge_rank_blend_weight is not None and not 0 <= ridge_rank_blend_weight <= 1:
        raise ValueError("Ridge rank blend weight must be between 0 and 1")
    train_contexts, _, _, _, train_groups, train_hashes, _ = _load_graph_partition(
        train_roots, "train"
    )
    test_contexts, _, _, _, test_groups, test_hashes, group_ids = _load_graph_partition(
        test_roots, "test"
    )
    if set(train_groups) & set(test_groups):
        raise ValueError("Training and test scenario groups overlap")
    linear_mean, linear_scale, linear_weights = _fit_linear_ranker(train_contexts)
    del train_contexts
    payload, gnn = _load_gnn(model_path)
    by_group: dict[str, list[dict[str, dict[str, float]]]] = defaultdict(list)
    by_network: dict[str, list[dict[str, dict[str, float]]]] = defaultdict(list)
    rows: list[dict[str, dict[str, float]]] = []
    with torch.inference_mode():
        for group, context in zip(group_ids, test_contexts, strict=True):
            linear = _linear_prediction(context, linear_mean, linear_scale, linear_weights)
            outcome_predictions, rank_scores = _gnn_predictions(context, payload, gnn)
            predictions = {
                "minimum_edit": [_edit_key(item[2]) for item in context],
                "linear_ridge": [tuple(value) for value in linear],
                "plan_gnn": [(value,) for value in rank_scores],
                "plan_gnn_outcome": outcome_predictions,
            }
            if ridge_rank_blend_weight is not None:
                gnn_ranks = _normalized_average_ranks([(value,) for value in rank_scores])
                ridge_ranks = _normalized_average_ranks([tuple(value) for value in linear])
                blend = [
                    (1 - ridge_rank_blend_weight) * gnn_ranks[index]
                    + ridge_rank_blend_weight * ridge_ranks[index]
                    for index in range(len(context))
                ]
                predictions["plan_gnn_ridge_blend"] = [(value,) for value in blend]
            metrics = _context_metrics(context, predictions)
            rows.append(metrics)
            by_group[group].append(metrics)
            if group.startswith("belgium-"):
                network = "belgium"
            elif group.startswith("switzerland-"):
                network = "switzerland"
            elif group.startswith("france-"):
                network = "france"
            elif group.startswith("netherlands-"):
                network = "netherlands"
            else:
                raise ValueError(f"Cannot infer network for scenario group {group!r}")
            by_network[network].append(metrics)
    group_summary = {group: _aggregate(items) for group, items in sorted(by_group.items())}
    return {
        "schema": "osrd-plan-ranker-spatial-generalization-evaluation/v1",
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "linear_baseline": {
            "method": "fixed ridge regression on train-only normalized action and graph summaries",
            "alpha": 1.0,
            "targets": ["new_conflict_count", "positive_arrival_delay_seconds/300"],
            "normalization_fit": "training split only",
            "hyperparameters_selected_on_test": False,
        },
        "ridge_rank_blend": (
            {
                "weight_on_ridge_average_rank": ridge_rank_blend_weight,
                "weight_on_gnn_average_rank": 1 - ridge_rank_blend_weight,
                "rank_ties": "average",
                "rank_normalization": "per-context to [0,1], lower is better",
            }
            if ridge_rank_blend_weight is not None
            else None
        ),
        "split": {
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "train_episode_hashes": sorted(train_hashes),
            "test_episode_hashes": sorted(test_hashes),
        },
        "pooled_context_metrics": _aggregate(rows),
        "macro_average_by_site": _macro_average_groups(list(group_summary.values())),
        "worst_site_by_pairwise_concordance": {
            method: min(
                (
                    (summary[method]["pairwise_concordance"], group)
                    for group, summary in group_summary.items()
                ),
                default=(0.0, ""),
            )
                for method in (
                    "minimum_edit",
                    "linear_ridge",
                    "plan_gnn",
                    "plan_gnn_outcome",
                    *(["plan_gnn_ridge_blend"] if ridge_rank_blend_weight is not None else []),
                )
        },
        "by_network": {
            network: _macro_average_groups(
                [group_summary[group] for group in group_summary if group.startswith(f"{network}-")]
            )
            for network in sorted(by_network)
        },
        "by_scenario_group": group_summary,
        "limitations": [
            "Evaluation labels are OSRD simulated counterfactual outcomes, not observed operator outcomes.",
            "The ridge baseline is a compact tabular reference, not a calibrated impact forecast.",
            "OSRD pathfinding, simulation, and conflict validation remain authoritative.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, action="append", required=True)
    parser.add_argument("--test-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-rank-blend-weight", type=float)
    args = parser.parse_args()
    report = evaluate(
        args.model, args.train_root, args.test_root, args.ridge_rank_blend_weight
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled_context_metrics"], indent=2, sort_keys=True))
    print(f"Wrote evaluation to {args.output}")


if __name__ == "__main__":
    main()
