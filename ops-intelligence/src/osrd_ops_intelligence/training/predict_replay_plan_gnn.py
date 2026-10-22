"""Rank a supplied set of complete candidate replay episodes with a plan GNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch_geometric.data import Batch, Data

from osrd_ops_intelligence.features.replay_graph import (
    ACTION_SUMMARY_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    action_summary_features,
    interaction_graph,
)
from osrd_ops_intelligence.features.replay_v1 import FEATURE_NAMES, feature_rows_for_episode
from osrd_ops_intelligence.models.replay_plan_gnn import ReplayPlanGNN


def predict(model_path: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("At least one candidate episode is required")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "osrd-replay-plan-gnn/v1":
        raise ValueError("Unsupported plan GNN model schema")
    feature_names = tuple(payload.get("feature_names", []))
    if not feature_names or not set(feature_names).issubset(FEATURE_NAMES):
        raise ValueError("Model feature schema contains features unsupported by this predictor")
    if tuple(payload.get("edge_feature_names", [])) != EDGE_FEATURE_NAMES:
        raise ValueError("Model edge schema does not match the installed predictor")
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
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32)
    scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32)
    graphs: list[Data] = []
    for episode in episodes:
        rows = feature_rows_for_episode(episode, feature_names)
        x = torch.tensor(
            [[float(row["features"][name]) for name in feature_names] for row in rows],
            dtype=torch.float32,
        )
        edge_index, edge_attr = interaction_graph(episode, feature_names=EDGE_FEATURE_NAMES)
        graph = Data(x=(x - mean) / scale, edge_index=edge_index, edge_attr=edge_attr)
        if model.action_channels:
            if tuple(payload.get("action_summary_feature_names", [])) != ACTION_SUMMARY_FEATURE_NAMES:
                raise ValueError("Model action-summary features do not match the predictor schema")
            action = episode["inputs"].get("action") or episode["inputs"].get("incident") or {}
            graph.action_features = action_summary_features(action).reshape(1, -1)
        graphs.append(graph)
    batch = Batch.from_data_list(cast(Any, graphs))
    batch_any = cast(Any, batch)
    with torch.inference_mode():
        outcome_logits, rank_scores = model(
            batch_any.x,
            batch_any.edge_index,
            batch_any.edge_attr,
            batch_any.batch,
            getattr(batch_any, "action_features", None),
        )
    conflict_estimates = torch.nn.functional.softplus(outcome_logits[:, 0]).tolist()
    delay_estimates = (torch.nn.functional.softplus(outcome_logits[:, 1]) * 300.0).tolist()
    order = sorted(range(len(episodes)), key=lambda index: (float(rank_scores[index]), index))
    records = []
    for index, episode in enumerate(episodes):
        action = episode["inputs"].get("action", {})
        candidate_id = str(episode.get("episode_id") or f"candidate-{index + 1}")
        records.append(
            {
                "candidate_id": candidate_id,
                "rank_score": float(rank_scores[index]),
                "predicted_new_conflict_count": float(conflict_estimates[index]),
                "predicted_positive_existing_arrival_delay_s": float(delay_estimates[index]),
                "action": action,
                "rank": order.index(index) + 1,
                "osrd_validation_required": True,
            }
        )
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return {
        "schema": "osrd-replay-plan-gnn-predictions/v1",
        "model_sha256": model_hash,
        "ranking_order": [records[index]["candidate_id"] for index in order],
        "candidates": sorted(records, key=lambda item: item["rank"]),
        "safety": {
            "research_only": True,
            "osrd_validation_required": True,
            "automatic_plan_acceptance": False,
            "rank_score_lower_is_better": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = json.loads(args.candidates.read_text(encoding="utf-8"))
        episodes = raw if isinstance(raw, list) else raw.get("episodes", [])
        result = predict(args.model, episodes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
