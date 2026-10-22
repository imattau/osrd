"""Evaluate a trained RIDE GNN once on its sealed provider test split."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from osrd_ops_intelligence.models.ride_gnn import RideHeterogeneousGNN
from osrd_ops_intelligence.training.ride_training_data import _download_file, _hash_file
from osrd_ops_intelligence.training.train_ride_gnn_streaming import (
    REVISION,
    _inverse_target,
)

TEST_GRAPH = "gnn/test/graphs_part_00000.pt"
TEST_SHA256 = "084b9f070c0d68814be9be8ac37c62646b6eb834f8379b23d3d4e0cb21396b9c"


def _day(graph: Any) -> str:
    return datetime.fromtimestamp(int(graph.snapshot_ts) / 1_000_000_000, tz=UTC).date().isoformat()


def _metric(abs_sum: float, sq_sum: float, count: int) -> dict[str, float | int]:
    if count == 0:
        return {"mae_seconds": float("nan"), "rmse_seconds": float("nan"), "targets": 0}
    return {
        "mae_seconds": abs_sum / count,
        "rmse_seconds": (sq_sum / count) ** 0.5,
        "targets": count,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")

    if args.graph_file.is_file():
        actual_sha = _hash_file(args.graph_file)
        if actual_sha != TEST_SHA256:
            raise ValueError(f"Test graph failed its pinned SHA-256 check: {args.graph_file}")
    else:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading the pinned RIDE test graph for final evaluation...", flush=True)
        _download_file(args.data_dir, TEST_GRAPH, TEST_SHA256)

    checkpoint_sha = _hash_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RideHeterogeneousGNN(
        train_features=config["train_features"],
        station_features=config["station_features"],
        hidden_channels=config["hidden_channels"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested, but this PyTorch installation has no available CUDA device"
        )
    model.to(device).eval()

    print(f"Verifying and loading test graphs: {args.graph_file}", flush=True)
    graphs = torch.load(args.graph_file, map_location="cpu", mmap=True, weights_only=False)
    overall = {"mae": [0.0, 0.0, 0], "zero_delta_baseline": [0.0, 0.0, 0]}
    by_horizon: dict[str, dict[str, list[float | int]]] = {}
    days: list[str] = []
    total_graphs = len(graphs)

    with torch.inference_mode():
        for start in range(0, total_graphs, args.batch_size):
            graph_batch = graphs[start : start + args.batch_size]
            days.extend(_day(graph) for graph in graph_batch)
            batch = Batch.from_data_list(graph_batch).to(device)
            target = batch[("train", "future", "station")].y
            prediction = model(batch)
            valid = torch.isfinite(target) & torch.isfinite(prediction)
            future_rank = batch[("train", "future", "station")].future_rank
            actual_seconds = _inverse_target(target[valid])
            predicted_seconds = _inverse_target(prediction[valid])
            error = predicted_seconds - actual_seconds
            count = int(valid.sum())
            overall["mae"][0] += float(error.abs().sum())
            overall["mae"][1] += float(error.square().sum())
            overall["mae"][2] += count
            baseline_error = actual_seconds
            overall["zero_delta_baseline"][0] += float(baseline_error.abs().sum())
            overall["zero_delta_baseline"][1] += float(baseline_error.square().sum())
            overall["zero_delta_baseline"][2] += count

            valid_ranks = future_rank[valid]
            for rank in torch.unique(valid_ranks).tolist():
                rank_mask = valid_ranks == rank
                rank_error = error[rank_mask]
                key = str(int(rank))
                metrics = by_horizon.setdefault(
                    key, {"mae": [0.0, 0.0, 0], "zero_delta_baseline": [0.0, 0.0, 0]}
                )
                metrics["mae"][0] += float(rank_error.abs().sum())
                metrics["mae"][1] += float(rank_error.square().sum())
                metrics["mae"][2] += int(rank_mask.sum())
                rank_actual = actual_seconds[rank_mask]
                metrics["zero_delta_baseline"][0] += float(rank_actual.abs().sum())
                metrics["zero_delta_baseline"][1] += float(rank_actual.square().sum())
                metrics["zero_delta_baseline"][2] += int(rank_mask.sum())

            del graph_batch, batch, target, prediction, error, actual_seconds, predicted_seconds
            if (start // args.batch_size + 1) % 100 == 0:
                done = min(start + args.batch_size, total_graphs)
                print(f"  evaluated {done}/{total_graphs} snapshots", flush=True)
                gc.collect()

    report = {
        "schema": "osrd-ride-gnn-test-report/v1",
        "dataset_revision": REVISION,
        "split": "provider test; evaluated once after model selection",
        "test_graph_sha256": TEST_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "graphs": total_graphs,
        "targets": overall["mae"][2],
        "test_dates": [min(days), max(days)],
        "overall": {
            "model": _metric(*overall["mae"]),
            "zero_delay_change_baseline": _metric(*overall["zero_delta_baseline"]),
        },
        "by_future_rank": {
            rank: {
                "model": _metric(*values["mae"]),
                "zero_delay_change_baseline": _metric(*values["zero_delta_baseline"]),
            }
            for rank, values in sorted(by_horizon.items(), key=lambda item: int(item[0]))
        },
        "test_graphs_used_for_training_or_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ride-gnn.pt"))
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=Path("data/rail/ride-gold-lite-gnn-test/gnn/test/graphs_part_00000.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/rail/ride-gold-lite-gnn-test"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ride-gnn-test-report.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
