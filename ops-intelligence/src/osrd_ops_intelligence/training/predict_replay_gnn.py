"""Score a proposed replay timetable with a trained advisory impact GNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from osrd_ops_intelligence.features.replay_graph import (
    EDGE_FEATURE_NAMES,
    LEGACY_EDGE_FEATURE_NAMES,
    interaction_graph,
)
from osrd_ops_intelligence.features.replay_v1 import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_V6_NAMES,
    FEATURE_V6_SCHEMA,
    FEATURE_V7_NAMES,
    FEATURE_V7_SCHEMA,
    LEGACY_FEATURE_NAMES,
    LEGACY_FEATURE_SCHEMA,
    feature_rows_for_episode,
)
from osrd_ops_intelligence.models.replay_impact_gnn import ReplayImpactGNN


def predict_episode(
    model_path: Path,
    episode_path: Path | dict[str, Any],
    *,
    threshold: float | None = None,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Load an artifact and return per-service advisory estimates for a candidate episode."""
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in {
        "osrd-ops-replay-impact-gnn/v1",
        "osrd-ops-replay-impact-gnn/v9",
        "osrd-ops-replay-impact-gnn/v10",
        "osrd-ops-replay-impact-gnn/v11",
        "osrd-ops-replay-impact-gnn/v12",
    }:
        raise ValueError("Unsupported model artifact schema")
    supported_features = {
        LEGACY_FEATURE_SCHEMA: LEGACY_FEATURE_NAMES,
        FEATURE_V6_SCHEMA: FEATURE_V6_NAMES,
        FEATURE_V7_SCHEMA: FEATURE_V7_NAMES,
        FEATURE_SCHEMA: FEATURE_NAMES,
    }
    feature_names = supported_features.get(payload.get("feature_schema"))
    if feature_names is None or payload.get("feature_names") != list(feature_names):
        raise ValueError("Model feature contract does not match this inference implementation")
    edge_feature_names = tuple(payload.get("edge_feature_names", LEGACY_EDGE_FEATURE_NAMES))
    if edge_feature_names not in (EDGE_FEATURE_NAMES, LEGACY_EDGE_FEATURE_NAMES):
        raise ValueError("Model graph-edge feature contract does not match this inference version")
    episode = (
        json.loads(episode_path.read_text()) if isinstance(episode_path, Path) else episode_path
    )
    rows = feature_rows_for_episode(episode, feature_names)
    if not rows:
        raise ValueError("Candidate episode contains no services to score")

    device = torch.device(device_name)
    mean = torch.tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    scale = torch.tensor(payload["feature_scale"], dtype=torch.float32, device=device)
    if mean.numel() != len(feature_names) or scale.numel() != len(feature_names):
        raise ValueError("Artifact normalization dimensions do not match feature contract")
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    x = (x - mean) / scale
    edge_index, edge_attr = interaction_graph(
        episode, device=device, feature_names=edge_feature_names
    )

    architecture = payload.get("architecture", payload)
    model = ReplayImpactGNN(
        input_channels=len(feature_names),
        hidden_channels=int(architecture["hidden_channels"]),
        heads=int(architecture["heads"]),
        dropout=float(architecture.get("dropout", 0.1)),
        edge_channels=int(architecture.get("edge_channels", len(edge_feature_names))),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        output = model(x, edge_index, edge_attr)
        calibration = payload.get("conflict_probability_calibration")
        conflict_logits = output[:, 1]
        if calibration is not None:
            if calibration.get("method") != "positive-slope-Platt-scaling/v1":
                raise ValueError("Unsupported conflict probability calibration method")
            conflict_logits = conflict_logits * float(calibration["slope"]) + float(
                calibration["intercept"]
            )
        probabilities = torch.sigmoid(conflict_logits).cpu().tolist()
        arrival_delta_s = (
            (output[:, 0] * float(payload.get("arrival_delta_target_scale_s", 1.0))).cpu().tolist()
        )

    cutoff = float(payload.get("decision_threshold", 0.5) if threshold is None else threshold)
    if not 0 <= cutoff <= 1:
        raise ValueError("Conflict threshold must be between 0 and 1")
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    promotion = payload.get("promotion_assessment", {})
    inference_infra_id = episode.get("inputs", {}).get("scenario", {}).get("infra_id")
    calibrated_infra_ids = payload.get("calibrated_infra_ids", [])
    offline_ranking_eligible = bool(promotion.get("offline_candidate_ranking_gate_passed", False))
    threshold_applicable = (
        offline_ranking_eligible
        and inference_infra_id is not None
        and int(inference_infra_id) in calibrated_infra_ids
    )
    arrival_infrastructure_applicable = (
        inference_infra_id is not None and int(inference_infra_id) in calibrated_infra_ids
    )
    arrival_delta_eligible = bool(
        promotion.get("arrival_delta_prediction_gate_passed", False)
        and arrival_infrastructure_applicable
    )
    services = [
        {
            "train_index": row["train_index"],
            "is_added_service": row["train_index"] >= baseline_count,
            "conflict_probability": float(probabilities[index]),
            "conflict_risk_at_threshold": (
                bool(probabilities[index] >= cutoff) if threshold_applicable else None
            ),
            "estimated_arrival_delta_s": (
                float(arrival_delta_s[index]) if arrival_delta_eligible else None
            ),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "schema": "osrd-ops-replay-impact-prediction/v2",
        "episode_id": episode["episode_id"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_promotion_assessment": promotion or {"status": "unassessed_legacy_artifact"},
        "offline_candidate_ranking_eligible": offline_ranking_eligible,
        "arrival_delta_prediction_eligible": arrival_delta_eligible,
        "decision_threshold": cutoff,
        "decision_threshold_applicable": threshold_applicable,
        "arrival_delta_infrastructure_applicable": arrival_infrastructure_applicable,
        "conflict_probability_calibration_applied": calibration is not None,
        "advisory_only": True,
        "osrd_validation_required": True,
        "services": services,
        "limitations": [
            "Predictions are research estimates and are not timetable feasibility results.",
            "Run the complete candidate through OSRD simulation and conflict detection.",
            "The current model was trained on synthetic replay data; held-out fixture "
            "routes do not establish operational calibration or cross-network transfer.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="trained model.pt artifact")
    parser.add_argument("--episode", type=Path, required=True, help="candidate replay episode JSON")
    parser.add_argument("--output", type=Path, help="write JSON here (default: stdout)")
    parser.add_argument(
        "--threshold", type=float, help="override validation-selected risk threshold"
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    args = parser.parse_args()
    result = predict_episode(
        args.model, args.episode, threshold=args.threshold, device_name=args.device
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
