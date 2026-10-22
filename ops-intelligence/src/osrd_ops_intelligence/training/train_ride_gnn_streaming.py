"""Train the RIDE GNN while bounding graph memory and preserving resumable state."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from osrd_ops_intelligence.models.ride_gnn import RideHeterogeneousGNN
from osrd_ops_intelligence.training.ride_gnn_data import FILES
from osrd_ops_intelligence.training.ride_training_data import _hash_file

DEFAULT_DATA = Path("data/rail/ride-gold-lite-gnn-training/gnn/train")
DEFAULT_SHARDS = ("graphs_part_00000.pt", "graphs_part_00001.pt")
REVISION = "e5c1ed1126bca1784eb6ea93816dd6cc89afa544"
TARGET_MEAN = 1.9409597608521583
TARGET_STD = 10.236780419279581


def _load_graph_file(path: Path) -> list[HeteroData]:
    # Only checksum-verified dataset files or locally generated chunks reach here.
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _snapshot_day(graph: HeteroData) -> str:
    timestamp = int(graph.snapshot_ts)
    return datetime.fromtimestamp(timestamp / 1_000_000_000, tz=UTC).date().isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cache_manifest_is_usable(
    path: Path, source_hashes: dict[str, str], chunk_size: int, limit: int | None
) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("revision") != REVISION
        or manifest.get("source_sha256") != source_hashes
        or manifest.get("chunk_size") != chunk_size
        or manifest.get("max_graphs") != limit
    ):
        return False
    return all((path.parent / item["file"]).is_file() for item in manifest.get("chunks", []))


def _prepare_chunks(
    shards: list[Path], cache_dir: Path, chunk_size: int, max_graphs: int | None
) -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    print("Verifying source shard hashes...", flush=True)
    for path in shards:
        relative_path = f"gnn/train/{path.name}"
        if _hash_file(path) != FILES[relative_path]:
            raise ValueError(f"Training shard failed its pinned SHA-256 check: {path}")
        source_hashes[path.name] = FILES[relative_path]

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "chunks.json"
    if _cache_manifest_is_usable(manifest_path, source_hashes, chunk_size, max_graphs):
        print(f"Reusing prepared graph chunks in {cache_dir}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    chunks: list[dict[str, Any]] = []
    total = 0
    for shard_id, source in enumerate(shards):
        if max_graphs is not None and total >= max_graphs:
            break
        print(f"Preparing chunks from {source.name} (one source shard at a time)", flush=True)
        graphs = _load_graph_file(source)
        take = len(graphs) if max_graphs is None else min(len(graphs), max_graphs - total)
        for start in range(0, take, chunk_size):
            end = min(start + chunk_size, take)
            name = f"part-{shard_id:02d}-{start:05d}-{end:05d}.pt"
            destination = cache_dir / name
            temporary = cache_dir / (name + ".tmp")
            torch.save(graphs[start:end], temporary)
            os.replace(temporary, destination)
            chunks.append(
                {
                    "file": name,
                    "shard_id": shard_id,
                    "start": start,
                    "end": end,
                    "snapshot_days": [_snapshot_day(graph) for graph in graphs[start:end]],
                }
            )
            total += end - start
            if len(chunks) % 10 == 0:
                print(f"  prepared {total} snapshots", flush=True)
        del graphs
        gc.collect()

    manifest = {
        "schema": "osrd-ride-gnn-chunks/v1",
        "revision": REVISION,
        "source_sha256": source_hashes,
        "chunk_size": chunk_size,
        "max_graphs": max_graphs,
        "graphs": total,
        "chunks": chunks,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _date_split(chunks: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    dates = sorted({day for chunk in chunks for day in chunk["snapshot_days"]})
    if len(dates) < 2:
        raise ValueError("At least two snapshot dates are needed for a date-based split")
    validation_start = min(len(dates) - 1, max(1, int(len(dates) * 0.8)))
    return set(dates[:validation_start]), set(dates[validation_start:])


def _inverse_target(values: Tensor) -> Tensor:
    signed_sqrt = values * TARGET_STD + TARGET_MEAN
    return torch.sign(signed_sqrt) * signed_sqrt.square()


def _step(
    model: RideHeterogeneousGNN,
    graphs: list[HeteroData],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    batch = Batch.from_data_list(graphs).to(device)
    target = batch[("train", "future", "station")].y
    prediction = model(batch)
    valid = torch.isfinite(target) & torch.isfinite(prediction)
    if optimizer is not None and valid.any():
        loss = torch.nn.functional.smooth_l1_loss(prediction[valid], target[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if valid.any():
        error = _inverse_target(prediction[valid]) - _inverse_target(target[valid])
        return error.abs().sum().detach(), error.square().sum().detach(), int(valid.sum())
    zero = torch.zeros((), device=device)
    return zero, zero, 0


def _save_resume(
    path: Path,
    model: RideHeterogeneousGNN,
    optimizer: torch.optim.Optimizer,
    progress: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            **progress,
        },
        temporary,
    )
    os.replace(temporary, path)


def _score_chunks(
    model: RideHeterogeneousGNN,
    chunks: list[dict[str, Any]],
    date_group: set[str],
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.no_grad():
        for chunk_id, chunk in enumerate(chunks):
            if not any(day in date_group for day in chunk["snapshot_days"]):
                continue
            graphs = _load_graph_file(cache_dir / chunk["file"])
            selected = [graph for graph in graphs if _snapshot_day(graph) in date_group]
            for start in range(0, len(selected), batch_size):
                abs_sum, sq_sum, n = _step(
                    model, selected[start : start + batch_size], None, device
                )
                absolute += float(abs_sum)
                squared += float(sq_sum)
                count += n
            del selected, graphs
            if chunk_id % 10 == 0:
                gc.collect()
    if count == 0:
        raise ValueError("Validation date group contains no targets")
    return {
        "mae_seconds": absolute / count,
        "rmse_seconds": (squared / count) ** 0.5,
        "targets": float(count),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    shards = [args.data_dir / name for name in DEFAULT_SHARDS]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pinned training shard(s): " + ", ".join(missing))
    cache_dir = (
        args.cache_dir
        or args.data_dir.parent / f"chunks-{args.chunk_size}-{args.max_graphs or 'all'}"
    )
    manifest = _prepare_chunks(shards, cache_dir, args.chunk_size, args.max_graphs)
    chunks = manifest["chunks"]
    train_dates, validation_dates = _date_split(chunks)
    train_chunk_ids = [
        i
        for i, chunk in enumerate(chunks)
        if any(day in train_dates for day in chunk["snapshot_days"])
    ]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested, but this PyTorch installation has no available CUDA device"
        )
    model = RideHeterogeneousGNN(hidden_channels=args.hidden_channels, layers=args.layers).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    last_path = args.output.with_suffix(".last.pt")
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_chunk_pos = 0
    resume_batch_start = 0
    global_step = 0
    training_config = {
        "chunk_size": args.chunk_size,
        "max_graphs": args.max_graphs,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "layers": args.layers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
    }

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint found at {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("source_sha256") != manifest["source_sha256"]:
            raise ValueError("Resume checkpoint belongs to different source graph shards")
        if state.get("training_config") != training_config:
            raise ValueError("Resume settings differ from the settings used by the checkpoint")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        resume_chunk_pos = state["chunk_order_position"]
        resume_batch_start = state["next_batch_start"]
        best_validation = state["best_validation"]
        best_epoch = state["best_epoch"]
        stale_epochs = state["stale_epochs"]
        history = state["history"]
        global_step = state["global_step"]
        print(f"Resuming at epoch {start_epoch}, training chunk {resume_chunk_pos}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        order = train_chunk_ids.copy()
        random.Random(args.seed + epoch).shuffle(order)
        offset = resume_batch_start if epoch == start_epoch else 0
        print(f"Epoch {epoch}/{args.epochs}: {len(order)} training chunks", flush=True)
        model.train()
        for chunk_pos, chunk_id in enumerate(order):
            if chunk_pos < resume_chunk_pos:
                continue
            chunk = chunks[chunk_id]
            graphs = _load_graph_file(cache_dir / chunk["file"])
            selected = [graph for graph in graphs if _snapshot_day(graph) in train_dates]
            random.Random(args.seed + epoch * 1_000_003 + chunk_id).shuffle(selected)
            start = offset if chunk_pos == resume_chunk_pos else 0
            if selected:
                for batch_start in range(start, len(selected), args.batch_size):
                    batch_graphs = selected[batch_start : batch_start + args.batch_size]
                    _step(model, batch_graphs, optimizer, device)
                    global_step += 1
                    next_start = batch_start + len(batch_graphs)
                    if global_step % args.checkpoint_every_steps == 0:
                        _save_resume(
                            last_path,
                            model,
                            optimizer,
                            {
                                "source_sha256": manifest["source_sha256"],
                                "training_config": training_config,
                                "epoch": epoch,
                                "chunk_order_position": chunk_pos,
                                "next_batch_start": next_start,
                                "best_validation": best_validation,
                                "best_epoch": best_epoch,
                                "stale_epochs": stale_epochs,
                                "history": history,
                                "global_step": global_step,
                            },
                        )
                        print(
                            f"  saved recovery checkpoint at optimizer step {global_step}",
                            flush=True,
                        )
            del selected, graphs
            gc.collect()
            offset = 0
            if chunk_pos % 10 == 0:
                _save_resume(
                    last_path,
                    model,
                    optimizer,
                    {
                        "source_sha256": manifest["source_sha256"],
                        "training_config": training_config,
                        "epoch": epoch,
                        "chunk_order_position": chunk_pos + 1,
                        "next_batch_start": 0,
                        "best_validation": best_validation,
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                        "history": history,
                        "global_step": global_step,
                    },
                )

        validation = _score_chunks(
            model, chunks, validation_dates, cache_dir, args.batch_size, device
        )
        row = {"epoch": epoch, "validation": validation}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        resume_chunk_pos = resume_batch_start = 0

        if validation["mae_seconds"] < best_validation:
            best_validation = validation["mae_seconds"]
            best_epoch = epoch
            stale_epochs = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            best_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            torch.save(
                {
                    "state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "config": {
                        "train_features": 32,
                        "station_features": 17,
                        "hidden_channels": args.hidden_channels,
                        "layers": args.layers,
                        "heads": 4,
                        "dropout": 0.15,
                    },
                    "dataset_revision": REVISION,
                    "target_mean": TARGET_MEAN,
                    "target_std": TARGET_STD,
                    "validation_dates": sorted(validation_dates),
                },
                best_temporary,
            )
            os.replace(best_temporary, args.output)
        else:
            stale_epochs += 1

        _save_resume(
            last_path,
            model,
            optimizer,
            {
                "source_sha256": manifest["source_sha256"],
                "training_config": training_config,
                "epoch": epoch + 1,
                "chunk_order_position": 0,
                "next_batch_start": 0,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "global_step": global_step,
            },
        )
        if stale_epochs >= args.patience:
            break

    report = {
        "schema": "osrd-ride-gnn-training-report/v1",
        "dataset_revision": REVISION,
        "graphs": manifest["graphs"],
        "train_dates": sorted(train_dates),
        "validation_dates": sorted(validation_dates),
        "best_epoch": best_epoch,
        "best_validation_mae_seconds": best_validation,
        "history": history,
        "checkpoint": str(args.output),
        "recovery_checkpoint": str(last_path),
        "test_graphs_used": False,
    }
    report_path = args.output.with_suffix(".report.json")
    _write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ride-gnn.pt"))
    parser.add_argument("--epochs", type=int, default=34)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from artifacts/ride-gnn.last.pt"
    )
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
