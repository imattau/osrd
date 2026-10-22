"""Train a replay-impact GNN with grouped validation, early stopping and a final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from osrd_ops_intelligence.features.replay_graph import EDGE_FEATURE_NAMES, interaction_graph
from osrd_ops_intelligence.features.replay_v1 import FEATURE_NAMES, FEATURE_SCHEMA, rows_for_episode
from osrd_ops_intelligence.models.linear import predict_probability
from osrd_ops_intelligence.models.replay_impact_gnn import ReplayImpactGNN
from osrd_ops_intelligence.training.overlap_baseline import load_episodes

DELAY_TARGET_SCALE_S = 60.0
MIN_OFFLINE_AP_IMPROVEMENT = 0.05


def _fit_logistic_reference(rows: list[list[float]], labels: list[int]) -> dict[str, Any]:
    """Fit the standardized L2 reference with vectorized float64 LBFGS operations."""
    if not rows or len(rows) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Logistic fitting requires paired rows and both target classes")
    x = torch.tensor(rows, dtype=torch.float64)
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-12)
    x = (x - mean) / scale
    y = torch.tensor(labels, dtype=torch.float64)
    weights = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer: Any = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = (x @ weights + bias).clamp(-35.0, 35.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.5 * 0.02 * weights.square().sum()
        cast(Any, loss).backward()
        return loss

    optimizer_step: Any = optimizer.step
    optimizer_step(closure)
    return {
        "weights": cast(list[float], cast(Any, weights.detach()).tolist()),
        "bias": float(bias.detach()),
        "means": cast(list[float], cast(Any, mean).tolist()),
        "scales": cast(list[float], cast(Any, scale).tolist()),
    }


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _split_groups(
    episodes: list[dict[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    validation_groups: list[str] | None,
    test_groups: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assign whole scenario groups to train/validation/test, never individual episodes."""
    group_by_episode = {
        episode["episode_id"]: str(episode.get("scenario_group") or episode["episode_id"])
        for episode in episodes
    }
    groups = sorted(set(group_by_episode.values()))
    if len(groups) < 3:
        raise ValueError(
            "Full training requires at least three distinct scenario groups for "
            "train, validation, and final test partitions."
        )

    test_set = set(test_groups or [])
    unknown_test = sorted(test_set - set(groups))
    if unknown_test:
        raise ValueError(f"Unknown test scenario groups: {', '.join(unknown_test)}")
    if not test_set:
        test_count = max(1, round(len(groups) * test_fraction))
        ordered = sorted(groups, key=_stable_hash)
        test_set = set(ordered[:test_count])

    remaining = set(groups) - test_set
    if not remaining:
        raise ValueError("At least one scenario group must remain outside the test partition")
    validation_set = set(validation_groups or [])
    unknown_validation = sorted(validation_set - set(groups))
    if unknown_validation:
        raise ValueError(f"Unknown validation scenario groups: {', '.join(unknown_validation)}")
    overlap = test_set & validation_set
    if overlap:
        raise ValueError(
            f"Groups cannot be in both validation and test: {', '.join(sorted(overlap))}"
        )
    if not validation_set:
        validation_count = max(1, round(len(groups) * validation_fraction))
        validation_set = set(sorted(remaining, key=_stable_hash)[:validation_count])
    if not validation_set <= remaining:
        raise ValueError("Validation groups must be separate from test groups")

    train_set = remaining - validation_set
    if not train_set:
        raise ValueError("At least one scenario group must remain for training")

    train = [
        episode for episode in episodes if group_by_episode[episode["episode_id"]] in train_set
    ]
    validation = [
        episode for episode in episodes if group_by_episode[episode["episode_id"]] in validation_set
    ]
    test = [episode for episode in episodes if group_by_episode[episode["episode_id"]] in test_set]
    return (
        train,
        validation,
        test,
        {
            "method": (
                "stable scenario-group split; explicit groups override deterministic assignment"
            ),
            "group_count": len(groups),
            "train_groups": sorted(train_set),
            "validation_groups": sorted(validation_set),
            "test_groups": sorted(test_set),
            "train_episode_ids": [episode["episode_id"] for episode in train],
            "validation_episode_ids": [episode["episode_id"] for episode in validation],
            "test_episode_ids": [episode["episode_id"] for episode in test],
            "train_episode_count": len(train),
            "validation_episode_count": len(validation),
            "test_episode_count": len(test),
        },
    )


def _validate_sources(episodes: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    if not episodes:
        raise ValueError("No replay episodes were loaded")
    for source in sources:
        if not source.get("episodes_sha256"):
            raise ValueError(f"Source manifest has no episode checksum: {source['path']}")
    ids = [episode.get("episode_id") for episode in episodes]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be present and unique across all training sources")
    for episode in episodes:
        if episode.get("schema") not in (
            "osrd-ops-replay-episode/v2",
            "osrd-ops-replay-episode/v3",
            "osrd-ops-replay-episode/v4",
        ):
            raise ValueError(f"Unsupported replay schema in episode {episode.get('episode_id')}")
        paths = episode["inputs"].get("resource_paths_by_train") or []
        if len(paths) != len(episode["inputs"].get("schedule_payloads", [])):
            raise ValueError(
                f"Episode {episode['episode_id']} must contain one OSRD resource path per service"
            )
        if any(not path.get("track_section_ranges") for path in paths):
            raise ValueError(
                f"Episode {episode['episode_id']} is missing pre-simulation OSRD track resources"
            )
        rows_for_episode(episode)  # validates target and feature structure before training starts


def _episode_graph(episode: dict[str, Any], mean: Tensor, scale: Tensor) -> Data:
    rows = rows_for_episode(episode)
    feature_matrix = torch.tensor(
        [[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=torch.float32
    )
    feature_matrix = (feature_matrix - mean) / scale
    targets = [row["arrival_delta_ms"] for row in rows]
    delay_mask = torch.tensor([value is not None for value in targets], dtype=torch.bool)
    # Model targets use normalized seconds; reports are converted back to seconds.
    delay = torch.tensor(
        [
            float(value) / 1000.0 / DELAY_TARGET_SCALE_S if value is not None else 0.0
            for value in targets
        ],
        dtype=torch.float32,
    )
    labels = episode["labels"]["paired_by_train_index"]
    conflict = torch.tensor(
        [int(bool(label["new_conflict"])) for label in labels], dtype=torch.float32
    )

    edge_index, edge_attr = interaction_graph(episode)
    return Data(
        x=feature_matrix,
        edge_index=edge_index,
        edge_attr=edge_attr,
        delay_y=delay,
        delay_mask=delay_mask,
        conflict_y=conflict,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(labels: Tensor, probabilities: Tensor, threshold: float) -> dict[str, Any]:
    actual = [int(value) for value in labels.tolist()]
    scores = [float(value) for value in probabilities.tolist()]
    predicted = [int(value >= threshold) for value in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(actual, predicted, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(actual, predicted, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(actual, predicted, strict=True))
    tn = sum(y == 0 and p == 0 for y, p in zip(actual, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives = sum(actual)
    ranked = sorted(zip(scores, actual, strict=True), key=lambda item: item[0], reverse=True)
    seen_positive = 0
    seen_total = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        end = index
        while end < len(ranked) and ranked[end][0] == score:
            seen_positive += ranked[end][1]
            seen_total += 1
            end += 1
        current_recall = seen_positive / positives if positives else 0.0
        precision_at_threshold = seen_positive / seen_total if seen_total else 0.0
        average_precision += (current_recall - previous_recall) * precision_at_threshold
        previous_recall = current_recall
        index = end
    brier = sum((score - label) ** 2 for score, label in zip(scores, actual, strict=True)) / max(
        len(actual), 1
    )
    return {
        "count": len(actual),
        "positive_count": positives,
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "brier_score": brier,
    }


def _best_threshold(labels: Tensor, scores: Tensor) -> float:
    """Maximize F1 in O(n log n), grouping tied scores as one threshold."""
    actual = [int(value) for value in labels.tolist()]
    ranked = sorted(
        zip((float(value) for value in scores.tolist()), actual, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(actual)
    tp = 0
    fp = 0
    highest = max((score for score, _ in ranked), default=0.0)
    best = (0.0, 0.0, math.nextafter(highest, math.inf))
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        end = index
        group_positives = 0
        while end < len(ranked) and ranked[end][0] == threshold:
            group_positives += ranked[end][1]
            end += 1
        tp += group_positives
        fp += end - index - group_positives
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, (f1, precision, threshold))
        index = end
    return best[2]


def _tune_threshold(labels: Tensor, probabilities: Tensor) -> float:
    return _best_threshold(labels, probabilities)


def _fit_conflict_probability_calibrator(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    """Fit positive-slope Platt scaling on validation predictions only."""
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("Calibration logits and labels must be matching one-dimensional tensors")
    if labels.unique().numel() < 2:
        raise ValueError("Conflict probability calibration requires both validation classes")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Calibration logits and labels must be finite")

    log_slope = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    calibration_logits = logits.detach().to(dtype=torch.float64)
    calibration_labels = labels.detach().to(dtype=torch.float64)
    optimizer = torch.optim.LBFGS(
        [log_slope, intercept],
        lr=0.5,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        slope = torch.exp(log_slope)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            slope * calibration_logits + intercept, calibration_labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_slope).detach()), float(intercept.detach())


def _predict(
    model: ReplayImpactGNN,
    graphs: list[Data],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model.eval()
    predictions: list[Tensor] = []
    delays: list[Tensor] = []
    delay_masks: list[Tensor] = []
    conflicts: list[Tensor] = []
    with torch.inference_mode():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            predictions.append(model(batch.x, batch.edge_index, batch.edge_attr).cpu())
            delays.append(batch.delay_y.cpu())
            delay_masks.append(batch.delay_mask.cpu())
            conflicts.append(batch.conflict_y.cpu())
    return (
        torch.cat(predictions),
        torch.cat(delays),
        torch.cat(delay_masks),
        torch.cat(conflicts),
    )


def _loss_components(
    predictions: Tensor,
    delay: Tensor,
    delay_mask: Tensor,
    conflict: Tensor,
    positive_weight: Tensor,
    node_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if node_mask is not None:
        predictions = predictions[node_mask]
        delay = delay[node_mask]
        delay_mask = delay_mask[node_mask]
        conflict = conflict[node_mask]
    if bool(delay_mask.any()):
        delay_loss = torch.nn.functional.smooth_l1_loss(
            predictions[:, 0][delay_mask], delay[delay_mask]
        )
    else:
        delay_loss = predictions[:, 0].sum() * 0.0
    conflict_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions[:, 1], conflict, pos_weight=positive_weight
    )
    return delay_loss, conflict_loss


def _evaluate_loss(
    model: ReplayImpactGNN,
    graphs: list[Data],
    *,
    batch_size: int,
    device: torch.device,
    positive_weight: Tensor,
) -> tuple[float, float, float]:
    predictions, delays, masks, conflicts = _predict(
        model, graphs, batch_size=batch_size, device=device
    )
    delay_loss, conflict_loss = _loss_components(
        predictions, delays, masks, conflicts, positive_weight.cpu()
    )
    return float(delay_loss), float(conflict_loss), float(delay_loss + conflict_loss)


def _regression_report(prediction: Tensor, actual: Tensor) -> dict[str, float | int]:
    errors = (prediction - actual).tolist()
    return {
        "count": len(errors),
        "mae_s": sum(abs(value) for value in errors) / max(len(errors), 1),
        "rmse_s": math.sqrt(sum(value * value for value in errors) / max(len(errors), 1)),
    }


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _resume_signature(
    sources: list[dict[str, Any]],
    split: dict[str, Any],
    mean: Tensor,
    scale: Tensor,
    args: argparse.Namespace,
) -> str:
    metadata = {
        "source_digests": sorted(source["episodes_sha256"] for source in sources),
        "split_groups": {
            "train": split["train_groups"],
            "validation": split["validation_groups"],
            "test": split["test_groups"],
        },
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": FEATURE_NAMES,
        "edge_feature_schema": "osrd-ops-replay-graph-edges/v2",
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "seed": args.seed,
        "hidden_channels": args.hidden_channels,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "max_positive_weight": args.max_positive_weight,
        "gradient_clip": args.gradient_clip,
        "lr_patience": args.lr_patience,
        "min_delta": args.min_delta,
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def train(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    episodes, sources = load_episodes(args.data_dirs)
    _validate_sources(episodes, sources)
    train_eps, validation_eps, test_eps, split = _split_groups(
        episodes,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        validation_groups=args.validation_scenario_groups,
        test_groups=args.test_scenario_groups,
    )

    training_rows = [row for episode in train_eps for row in rows_for_episode(episode)]
    raw_features = torch.tensor(
        [[row["features"][name] for name in FEATURE_NAMES] for row in training_rows],
        dtype=torch.float32,
    )
    mean = raw_features.mean(dim=0)
    scale = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_graphs = [_episode_graph(episode, mean, scale) for episode in train_eps]
    validation_graphs = [_episode_graph(episode, mean, scale) for episode in validation_eps]

    _set_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ReplayImpactGNN(
        input_channels=len(FEATURE_NAMES), hidden_channels=args.hidden_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_learning_rate
    )
    nodes = sum(graph.num_nodes for graph in train_graphs)
    positives = sum(float(graph.conflict_y.sum()) for graph in train_graphs)
    negatives = nodes - positives
    if positives == 0:
        raise ValueError("Training groups contain no positive conflict labels")
    positive_weight = torch.tensor(
        min(args.max_positive_weight, max(1.0, negatives / positives)), dtype=torch.float32
    ).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=generator)
    resume_signature = _resume_signature(sources, split, mean, scale, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint-last.pt"
    model_path = args.output_dir / "model.pt"
    report_path = args.output_dir / "report.json"
    if (model_path.exists() or report_path.exists()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output already contains a completed run: {args.output_dir}")

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("resume_signature") != resume_signature:
            raise ValueError(
                "Checkpoint data, split, seed, architecture, or optimizer settings do not "
                "match this run. Use the original settings or start a fresh output directory."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_state = checkpoint["best_state"]
        stale_epochs = int(checkpoint["stale_epochs"])
        history = checkpoint["history"]
        generator.set_state(checkpoint["loader_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    print(
        f"device={device}; episodes train/validation/test="
        f"{len(train_eps)}/{len(validation_eps)}/{len(test_eps)}; "
        f"groups train/validation/test="
        f"{len(split['train_groups'])}/{len(split['validation_groups'])}/{len(split['test_groups'])}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch.x, batch.edge_index, batch.edge_attr)
            delay_loss, conflict_loss = _loss_components(
                prediction,
                batch.delay_y,
                batch.delay_mask,
                batch.conflict_y,
                positive_weight,
            )
            loss = delay_loss + conflict_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        val_delay_loss, val_conflict_loss, val_loss = _evaluate_loss(
            model,
            validation_graphs,
            batch_size=args.batch_size,
            device=device,
            positive_weight=positive_weight,
        )
        scheduler.step(val_loss)
        improved = val_loss < best_validation_loss - args.min_delta
        if improved:
            best_validation_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": sum(batch_losses) / max(len(batch_losses), 1),
            "validation_loss": val_loss,
            "validation_delay_loss": val_delay_loss,
            "validation_conflict_loss": val_conflict_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_epoch": best_epoch,
        }
        history.append(record)
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"epoch {epoch}: train={record['train_loss']:.5f} "
                f"validation={val_loss:.5f} best_epoch={best_epoch} "
                f"lr={record['learning_rate']:.2g}",
                flush=True,
            )
        checkpoint = {
            "schema": "osrd-replay-gnn-training-checkpoint/v1",
            "resume_signature": resume_signature,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_state": best_state,
            "stale_epochs": stale_epochs,
            "history": history,
            "loader_rng_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        if stale_epochs >= args.patience:
            print(f"early stopping after {epoch} epochs; best epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    # Threshold and reference threshold are selected on validation only.
    val_prediction, val_delay, val_mask, val_conflict = _predict(
        model, validation_graphs, batch_size=args.batch_size, device=device
    )
    conflict_calibration_slope, conflict_calibration_intercept = (
        _fit_conflict_probability_calibrator(val_prediction[:, 1], val_conflict)
    )
    calibrated_val_conflict_logits = (
        val_prediction[:, 1] * conflict_calibration_slope + conflict_calibration_intercept
    )
    calibrated_val_conflict_probabilities = torch.sigmoid(calibrated_val_conflict_logits)
    conflict_threshold = _tune_threshold(val_conflict, calibrated_val_conflict_probabilities)
    val_rows = [row for episode in validation_eps for row in rows_for_episode(episode)]
    train_rows = training_rows
    train_labels = [int(row["conflict_label"]) for row in train_rows]
    validation_labels = [int(row["conflict_label"]) for row in val_rows]
    linear = _fit_logistic_reference(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in train_rows],
        train_labels,
    )
    validation_linear_probabilities = torch.tensor(
        [
            predict_probability(linear, [float(row["features"][name]) for name in FEATURE_NAMES])
            for row in val_rows
        ],
        dtype=torch.float32,
    )
    linear_threshold = _tune_threshold(
        torch.tensor(validation_labels), validation_linear_probabilities
    )
    gap_threshold = -_best_threshold(
        val_conflict,
        -torch.tensor([float(row["features"]["nearest_departure_gap_s"]) for row in val_rows]),
    )

    # The final test groups are touched only after early stopping and threshold selection.
    test_graphs = [_episode_graph(episode, mean, scale) for episode in test_eps]
    test_prediction, test_delay, test_mask, test_conflict = _predict(
        model, test_graphs, batch_size=args.batch_size, device=device
    )
    calibrated_test_conflict_logits = (
        test_prediction[:, 1] * conflict_calibration_slope + conflict_calibration_intercept
    )
    test_probabilities = torch.sigmoid(calibrated_test_conflict_logits)
    test_delay_prediction = test_prediction[:, 0][test_mask]
    test_delay_targets = test_delay[test_mask]
    test_rows = [row for episode in test_eps for row in rows_for_episode(episode)]
    test_gaps = torch.tensor(
        [float(row["features"]["nearest_departure_gap_s"]) for row in test_rows]
    )
    test_linear_probabilities = torch.tensor(
        [
            predict_probability(linear, [float(row["features"][name]) for name in FEATURE_NAMES])
            for row in test_rows
        ],
        dtype=torch.float32,
    )
    zero_count = int(test_conflict.numel())
    test_metrics = {
        "conflict": _binary_metrics(test_conflict, test_probabilities, conflict_threshold),
        "majority_zero_reference": _binary_metrics(test_conflict, torch.zeros(zero_count), 0.5),
        "departure_gap_reference": _binary_metrics(
            test_conflict, (test_gaps <= gap_threshold).float(), 0.5
        ),
        "linear_logistic_reference": _binary_metrics(
            test_conflict, test_linear_probabilities, linear_threshold
        ),
        "arrival_delta": _regression_report(
            test_delay_prediction * DELAY_TARGET_SCALE_S,
            test_delay_targets * DELAY_TARGET_SCALE_S,
        ),
        "zero_change_arrival_reference": _regression_report(
            torch.zeros_like(test_delay_targets) * DELAY_TARGET_SCALE_S,
            test_delay_targets * DELAY_TARGET_SCALE_S,
        ),
    }
    test_metrics_by_scenario_group: dict[str, Any] = {}
    scenario_groups = [str(row["scenario_group"]) for row in test_rows]
    for group in sorted(set(scenario_groups)):
        indices = torch.tensor(
            [index for index, value in enumerate(scenario_groups) if value == group],
            dtype=torch.long,
        )
        group_mask = test_mask[indices]
        group_delay_prediction = test_prediction[indices, 0][group_mask]
        group_delay_targets = test_delay[indices][group_mask]
        test_metrics_by_scenario_group[group] = {
            "conflict": _binary_metrics(
                test_conflict[indices], test_probabilities[indices], conflict_threshold
            ),
            "linear_logistic_reference": _binary_metrics(
                test_conflict[indices], test_linear_probabilities[indices], linear_threshold
            ),
            "departure_gap_reference": _binary_metrics(
                test_conflict[indices],
                (test_gaps[indices] <= gap_threshold).float(),
                0.5,
            ),
            "arrival_delta": _regression_report(
                group_delay_prediction * DELAY_TARGET_SCALE_S,
                group_delay_targets * DELAY_TARGET_SCALE_S,
            ),
            "zero_change_arrival_reference": _regression_report(
                torch.zeros_like(group_delay_targets) * DELAY_TARGET_SCALE_S,
                group_delay_targets * DELAY_TARGET_SCALE_S,
            ),
        }
    validation_metrics = {
        "conflict": _binary_metrics(
            val_conflict, calibrated_val_conflict_probabilities, conflict_threshold
        ),
        "conflict_raw_uncalibrated": _binary_metrics(
            val_conflict, torch.sigmoid(val_prediction[:, 1]), conflict_threshold
        ),
        "departure_gap_reference": _binary_metrics(
            val_conflict,
            torch.tensor(
                [
                    float(row["features"]["nearest_departure_gap_s"]) <= gap_threshold
                    for row in val_rows
                ]
            ),
            0.5,
        ),
        "linear_logistic_reference": _binary_metrics(
            val_conflict, validation_linear_probabilities, linear_threshold
        ),
        "arrival_delta": _regression_report(
            val_prediction[:, 0][val_mask] * DELAY_TARGET_SCALE_S,
            val_delay[val_mask] * DELAY_TARGET_SCALE_S,
        ),
        "zero_change_arrival_reference": _regression_report(
            torch.zeros_like(val_delay[val_mask]) * DELAY_TARGET_SCALE_S,
            val_delay[val_mask] * DELAY_TARGET_SCALE_S,
        ),
    }

    # Conflict ranking and arrival-delay regression have separate eligibility.
    # A weak arrival head must not be used, but it need not disable a validated
    # offline conflict ranker. Production remains blocked pending broader study.
    best_reference_ap = max(
        test_metrics["departure_gap_reference"]["average_precision"],
        test_metrics["linear_logistic_reference"]["average_precision"],
    )
    group_ap_improvements = {
        group: metrics["conflict"]["average_precision"]
        - max(
            metrics["linear_logistic_reference"]["average_precision"],
            metrics["departure_gap_reference"]["average_precision"],
        )
        for group, metrics in test_metrics_by_scenario_group.items()
    }
    minimum_group_ap_improvement = min(group_ap_improvements.values(), default=0.0)
    impact_gate = {
        "conflict_average_precision_exceeds_references": (
            test_metrics["conflict"]["average_precision"] > best_reference_ap
        ),
        "arrival_delta_mae_beats_zero_change": (
            test_metrics["arrival_delta"]["mae_s"]
            < test_metrics["zero_change_arrival_reference"]["mae_s"]
        ),
        "test_contains_conflict_positive": bool(test_conflict.sum() > 0),
        "each_test_group_exceeds_references_by_minimum_margin": (
            minimum_group_ap_improvement >= MIN_OFFLINE_AP_IMPROVEMENT
        ),
    }
    eligible_for_offline_ranking = (
        impact_gate["conflict_average_precision_exceeds_references"]
        and impact_gate["test_contains_conflict_positive"]
        and impact_gate["each_test_group_exceeds_references_by_minimum_margin"]
    )

    model_payload = {
        "schema": "osrd-ops-replay-impact-gnn/v12",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "edge_feature_schema": "osrd-ops-replay-graph-edges/v2",
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "arrival_delta_target_scale_s": DELAY_TARGET_SCALE_S,
        "offline_ranking_minimum_ap_improvement": MIN_OFFLINE_AP_IMPROVEMENT,
        "calibrated_infra_ids": sorted(
            {
                int(episode["snapshot"]["infra_id"])
                for episode in validation_eps
                if episode.get("snapshot", {}).get("infra_id") is not None
            }
        ),
        "architecture": {
            "type": "two-layer GATv2",
            "hidden_channels": args.hidden_channels,
            "heads": 2,
            "dropout": 0.1,
            "edge_channels": len(EDGE_FEATURE_NAMES),
        },
        "decision_threshold": conflict_threshold,
        "threshold_selection": (
            "validation-group F1 on validation-calibrated probability, "
            "tie-broken by precision then threshold"
        ),
        "conflict_probability_calibration": {
            "method": "positive-slope-Platt-scaling/v1",
            "fit_partition": "validation scenario groups only",
            "validation_row_count": int(val_conflict.numel()),
            "slope": conflict_calibration_slope,
            "intercept": conflict_calibration_intercept,
        },
        "promotion_assessment": {
            "offline_candidate_ranking_gate_passed": eligible_for_offline_ranking,
            "arrival_delta_prediction_gate_passed": impact_gate[
                "arrival_delta_mae_beats_zero_change"
            ],
            "test_group_average_precision_improvements": group_ap_improvements,
            "minimum_test_group_average_precision_improvement": minimum_group_ap_improvement,
            "checks": impact_gate,
            "production_promotion_eligible": False,
            "production_blocker": (
                "Requires evaluation on independent networks, calibrated uncertainty, "
                "and an end-to-end search benefit study."
            ),
        },
        "selected_epoch": best_epoch,
        "seed": args.seed,
        "training_scenario_groups": split["train_groups"],
        "state_dict": best_state,
    }
    run_config = {
        "seed": args.seed,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_positive_weight": args.max_positive_weight,
        "device": str(device),
    }
    report = {
        "schema": "osrd-ops-replay-impact-gnn-report/v4",
        "feature_schema": FEATURE_SCHEMA,
        "sources": sources,
        "split": split,
        "run_config": run_config,
        "resume_signature": resume_signature,
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "threshold_selected_on_validation": conflict_threshold,
        "conflict_probability_calibration": model_payload["conflict_probability_calibration"],
        "departure_gap_threshold_selected_on_validation_s": gap_threshold,
        "linear_logistic_reference": {
            "algorithm": "standardized L2 logistic regression fit on training groups only",
            "threshold_selected_on_validation": linear_threshold,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_scenario_group": test_metrics_by_scenario_group,
        "promotion_assessment": {
            "offline_candidate_ranking_gate_passed": eligible_for_offline_ranking,
            "arrival_delta_prediction_gate_passed": impact_gate[
                "arrival_delta_mae_beats_zero_change"
            ],
            "test_group_average_precision_improvements": group_ap_improvements,
            "minimum_test_group_average_precision_improvement": minimum_group_ap_improvement,
            "checks": impact_gate,
            "production_promotion_eligible": False,
            "production_blocker": (
                "Requires independent real-network evaluation, calibrated uncertainty, "
                "and an end-to-end search benefit study."
            ),
        },
        "data_counts": {
            "train_episode_count": len(train_eps),
            "validation_episode_count": len(validation_eps),
            "test_episode_count": len(test_eps),
            "train_node_count": sum(graph.num_nodes for graph in train_graphs),
            "validation_node_count": sum(graph.num_nodes for graph in validation_graphs),
            "test_node_count": sum(graph.num_nodes for graph in test_graphs),
            "train_conflict_positive_count": int(positives),
            "validation_conflict_positive_count": int(
                sum(graph.conflict_y.sum() for graph in validation_graphs)
            ),
            "test_conflict_positive_count": int(test_conflict.sum()),
        },
        "training_history": history,
        "limitations": [
            "Current replay targets are synthetic OSRD simulation outcomes, not field outcomes.",
            "A scenario-group holdout on one infrastructure does not establish network transfer.",
            "The model scores an already specified insertion and does not construct a timetable.",
            "OSRD simulation and conflict validation remain authoritative.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(model_payload, model_path)
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report["artifacts"] = {"model_path": model_path.name, "model_sha256": model_digest}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "training-history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history), encoding="utf-8"
    )
    metric_rows = "".join(
        "| "
        f"{group} | {metrics['conflict']['average_precision']:.3f} | "
        f"{metrics['linear_logistic_reference']['average_precision']:.3f} | "
        f"{metrics['conflict']['brier_score']:.3f} | "
        f"{metrics['linear_logistic_reference']['brier_score']:.3f} |\n"
        for group, metrics in test_metrics_by_scenario_group.items()
    )
    model_card = (
        "# OSRD service-insertion impact GNN\n\n"
        "Offline research artifact trained on paired synthetic OSRD replay simulations. "
        "It ranks existing services likely to become newly conflict-involved after a "
        "proposed insertion; it does not construct a timetable or recommend retimings.\n\n"
        "## Training and split\n\n"
        "- Architecture: two-layer GATv2 with explicit resource-count and candidate-interaction "
        "edge features; "
        f"node schema `{FEATURE_SCHEMA}`, edge schema `osrd-ops-replay-graph-edges/v2`.\n"
        f"- Episodes: {len(train_eps)} train / {len(validation_eps)} validation / "
        f"{len(test_eps)} held-out test.\n"
        f"- Selected epoch: {best_epoch} / {len(history)} completed.\n"
        f"- Train groups: {', '.join(split['train_groups'])}.\n"
        f"- Validation groups: {', '.join(split['validation_groups'])}.\n"
        f"- Test groups: {', '.join(split['test_groups'])}.\n"
        f"- Probability calibration: validation-only positive-slope Platt scaling "
        f"(slope {conflict_calibration_slope:.4f}, intercept "
        f"{conflict_calibration_intercept:.4f}).\n"
        f"- Offline ranking gate passed: {eligible_for_offline_ranking}.\n"
        f"- A threshold decision is enabled only when the offline gate passes and "
        f"the infrastructure is in validation scope "
        f"{model_payload['calibrated_infra_ids']}.\n"
        f"- Model SHA-256: `{model_digest}`.\n\n"
        "## Held-out results\n\n"
        "| Scenario group | GNN AP | Linear AP | GNN Brier | Linear Brier |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{metric_rows}"
        f"| Aggregate | {test_metrics['conflict']['average_precision']:.3f} | "
        f"{test_metrics['linear_logistic_reference']['average_precision']:.3f} | "
        f"{test_metrics['conflict']['brier_score']:.3f} | "
        f"{test_metrics['linear_logistic_reference']['brier_score']:.3f} |\n\n"
        f"Arrival-delta MAE is {test_metrics['arrival_delta']['mae_s']:.2f} s versus "
        f"{test_metrics['zero_change_arrival_reference']['mae_s']:.2f} s for "
        "zero-change; arrival predictions are disabled. Production promotion "
        "remains false.\n\n"
        "All fixtures are synthetic. Outputs are advisory rankings; OSRD simulation "
        "and conflict detection remain authoritative. Real-network data, reliable "
        "cross-network calibration, and end-to-end search-benefit evaluation remain "
        "outstanding.\n"
    )
    (args.output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_epoch",
                    "epochs_completed",
                    "data_counts",
                    "validation_metrics",
                    "test_metrics",
                    "artifacts",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return model_payload, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-scenario-groups", nargs="+")
    parser.add_argument("--test-scenario-groups", nargs="+")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr-patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from output-dir/checkpoint-last.pt"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace completed artifacts in output-dir"
    )
    args = parser.parse_args()
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(args, name)
        if not 0.05 <= value <= 0.4:
            parser.error(f"--{name.replace('_', '-')} must be between 0.05 and 0.4")
    if args.validation_fraction + args.test_fraction >= 0.8:
        parser.error("validation and test fractions must leave at least 20% for training")
    if min(args.epochs, args.patience, args.batch_size, args.hidden_channels, args.log_every) <= 0:
        parser.error(
            "epochs, patience, batch size, hidden channels and log interval must be positive"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    train(args)


if __name__ == "__main__":
    main()
