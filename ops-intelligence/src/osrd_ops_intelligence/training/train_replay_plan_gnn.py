"""Fit an action-conditioned graph scorer on matched OSRD timetable alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from osrd_ops_intelligence.features.replay_graph import (
    ACTION_SUMMARY_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    action_summary_features,
    interaction_graph,
)
from osrd_ops_intelligence.features.replay_v1 import FEATURE_NAMES, rows_for_episode
from osrd_ops_intelligence.models.replay_plan_gnn import ReplayPlanGNN

MODEL_SCHEMA = "osrd-replay-plan-gnn/v1"
DELAY_SCALE_S = 300.0


def _context_key(episode: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(episode["scenario_group"]),
        int(episode["seed"]),
        int(episode["episode_index"]),
    )


def _outcomes(episode: dict[str, Any]) -> tuple[float, float]:
    labels = episode["labels"]
    conflicts = float(labels["new_conflict_count"])
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    positive_delay = sum(
        max(0.0, float(row["arrival_delta_ms"] or 0) / 1000.0)
        for row in labels["paired_by_train_index"]
        if int(row["train_index"]) < baseline_count
    )
    return conflicts, positive_delay


def _make_graph(episode: dict[str, Any]) -> tuple[Data, tuple[float, float]]:
    rows = rows_for_episode(episode)
    x = torch.tensor(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=torch.float32,
    )
    edge_index, edge_attr = interaction_graph(episode)
    action = episode["inputs"].get("action") or episode["inputs"].get("incident") or {}
    action_features = action_summary_features(action).reshape(1, -1)
    conflict_count, positive_delay_s = _outcomes(episode)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        action_features=action_features,
    ), (conflict_count, positive_delay_s)


GraphContext = list[tuple[Data, tuple[float, float], str]]


def _action_coverage(
    action_sets: list[list[dict[str, Any]]], alternatives_per_partition: list[list[int]]
) -> dict[str, Any]:
    actions = [action for partition in action_sets for action in partition]
    alternatives_per_context = sorted(
        {count for partition in alternatives_per_partition for count in partition}
    )
    stop_patterns = {
        json.dumps(action.get("stops", []), sort_keys=True, separators=(",", ":"))
        for action in actions
    }
    route_patterns = {
        json.dumps(action.get("route_via", []), sort_keys=True, separators=(",", ":"))
        for action in actions
    }
    dwell_patterns = {
        tuple(int(stop.get("dwell_s", 0)) for stop in action.get("stops", []))
        for action in actions
        if action.get("stops")
    }
    retiming_patterns = {
        tuple(
            sorted(
                (int(item.get("train_index", -1)), int(item.get("departure_shift_s", 0)))
                for item in action.get("retimings", [])
            )
        )
        for action in actions
    }
    return {
        "requested_frequency_seconds": sorted(
            {int(action.get("frequency_s", 0)) for action in actions}
        ),
        "departure_phase_seconds": sorted(
            {int(action.get("departure_phase_s", 0)) for action in actions}
        ),
        "matched_alternatives_per_context": alternatives_per_context,
        "retiming_patterns": [
            [list(edit) for edit in sorted(pattern)] for pattern in sorted(retiming_patterns)
        ],
        "route_waypoint_action_count": sum(bool(action.get("route_via")) for action in actions),
        "distinct_nonempty_route_patterns": sum(
            bool(json.loads(pattern)) for pattern in route_patterns
        ),
        "distinct_stop_patterns": len(stop_patterns),
        "station_stop_action_count": sum(bool(action.get("stops")) for action in actions),
        "distinct_nonzero_dwell_patterns": len(dwell_patterns),
        "station_operational_point_ids": sorted(
            {
                str(stop["operational_point_id"])
                for action in actions
                for stop in action.get("stops", [])
                if stop.get("operational_point_id") is not None
            }
        ),
    }


def _cohort_directories(roots: list[Path], partition: str) -> list[Path]:
    directories: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{partition} cohort root does not exist: {root}")
        found = sorted({path.parent for path in root.rglob("manifest.json")})
        if not found:
            raise ValueError(f"{partition} cohort root has no manifest directories: {root}")
        directories.extend(found)
    if len(set(directories)) != len(directories):
        raise ValueError(f"{partition} cohort directory is included more than once")
    return sorted(directories)


def _load_graph_partition(
    roots: list[Path], partition: str
) -> tuple[
    list[GraphContext], torch.Tensor, torch.Tensor, int, list[str], list[str], list[str]
]:
    """Verify and convert JSONL incrementally; only compact graph tensors are retained."""
    contexts: dict[tuple[str, int, int], GraphContext] = defaultdict(list)
    sums = torch.zeros(len(FEATURE_NAMES), dtype=torch.float64)
    square_sums = torch.zeros(len(FEATURE_NAMES), dtype=torch.float64)
    node_count = 0
    hashes: list[str] = []
    group_names: set[str] = set()
    episode_ids: set[str] = set()
    for directory in _cohort_directories(roots, partition):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = directory / "episodes.jsonl"
        digest = hashlib.sha256()
        cohort_groups: set[str] = set()
        cohort_episode_count = 0
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                episode = json.loads(raw_line)
                cohort_episode_count += 1
                episode_id = str(episode.get("episode_id") or "")
                if not episode_id or episode_id in episode_ids:
                    raise ValueError(f"Missing or duplicate episode id in {path}:{line_number}")
                episode_ids.add(episode_id)
                cohort_groups.add(str(episode.get("scenario_group")))
                rows = rows_for_episode(episode)
                feature_rows = [
                    [float(row["features"][feature]) for feature in FEATURE_NAMES] for row in rows
                ]
                matrix64 = torch.tensor(feature_rows, dtype=torch.float64)
                sums += matrix64.sum(0)
                square_sums += matrix64.square().sum(0)
                node_count += len(feature_rows)
                graph, targets = _make_graph(episode)
                signature = json.dumps(
                    episode["inputs"]["action"], sort_keys=True, separators=(",", ":")
                )
                contexts[_context_key(episode)].append((graph, targets, signature))
        actual_digest = digest.hexdigest()
        if actual_digest != manifest.get("episodes_sha256"):
            raise ValueError(f"Episode checksum does not match manifest: {path}")
        if int(manifest.get("episode_count", -1)) != cohort_episode_count:
            raise ValueError(f"Episode count does not match manifest: {path}")
        if len(cohort_groups) != 1:
            raise ValueError(f"Expected exactly one scenario group in cohort {path}")
        group_names.update(cohort_groups)
        hashes.append(actual_digest)
    if not contexts or not group_names:
        raise ValueError(f"No graph contexts were loaded for {partition}")
    output: list[GraphContext] = []
    context_groups: list[str] = []
    for key, alternatives in sorted(contexts.items()):
        signatures = [item[2] for item in alternatives]
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"Duplicate action alternatives in matched context {key}")
        if len(alternatives) >= 2:
            output.append(alternatives)
            context_groups.append(key[0])
    if not output:
        raise ValueError(f"Partition {partition} has no matched contexts with alternatives")
    return output, sums, square_sums, node_count, sorted(group_names), hashes, context_groups


def _normalize_contexts(
    contexts: list[GraphContext], mean: torch.Tensor, scale: torch.Tensor
) -> list[tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]]:
    batches: list[tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for context in contexts:
        graphs: list[Data] = []
        targets: list[tuple[float, float]] = []
        edit_costs: list[tuple[int, int]] = []
        for graph, outcome, action_signature in context:
            action = json.loads(action_signature)
            graph.x = (cast(torch.Tensor, graph.x) - mean) / scale
            graphs.append(graph)
            targets.append(outcome)
            shifts = [int(item.get("departure_shift_s", 0)) for item in action.get("retimings", [])]
            stops = action.get("stops", [])
            route_via = action.get("route_via", [])
            frequency = int(action.get("frequency_s", 0))
            phase = int(action.get("departure_phase_s", 0))
            edit_count = (
                sum(shift != 0 for shift in shifts)
                + int(frequency > 0 and frequency != 300)
                + int(phase != 0)
                + len(stops)
                + len(route_via)
            )
            edit_magnitude = (
                sum(abs(shift) for shift in shifts) / 60
                + (abs(frequency - 300) / 60 if frequency > 0 else 0)
                + phase / 60
                + sum(1 + float(stop.get("dwell_s", 0)) / 60 for stop in stops)
                + len(route_via)
            )
            edit_costs.append((edit_count, round(edit_magnitude)))
        batches.append(
            (
                Batch.from_data_list(graphs),
                torch.tensor([item[0] for item in targets], dtype=torch.float32),
                torch.tensor([item[1] for item in targets], dtype=torch.float32),
                torch.tensor(edit_costs, dtype=torch.float32),
            )
        )
    return batches


def _rank_loss(
    scores: torch.Tensor, conflicts: torch.Tensor, delay_s: torch.Tensor
) -> torch.Tensor:
    left, right = torch.triu_indices(len(scores), len(scores), offset=1, device=scores.device)
    conflict_difference = conflicts[left] - conflicts[right]
    delay_difference = delay_s[left] - delay_s[right]
    difference = torch.where(conflict_difference != 0, conflict_difference, delay_difference)
    comparable = difference.abs() > 1e-6
    if not bool(comparable.any()):
        return scores.sum() * 0.0
    preference = difference[comparable].sign()
    return F.softplus(-preference * (scores[left[comparable]] - scores[right[comparable]])).mean()


def _loss(
    outcomes: torch.Tensor,
    scores: torch.Tensor,
    conflicts: torch.Tensor,
    delay_s: torch.Tensor,
) -> torch.Tensor:
    conflict_loss = F.smooth_l1_loss(F.softplus(outcomes[:, 0]), torch.log1p(conflicts))
    delay_loss = F.smooth_l1_loss(F.softplus(outcomes[:, 1]), torch.log1p(delay_s / DELAY_SCALE_S))
    return _rank_loss(scores, conflicts, delay_s) + 0.25 * conflict_loss + 0.1 * delay_loss


def _forward_context_batch(
    model: ReplayPlanGNN,
    samples: list[tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    graphs: list[Data] = []
    conflicts: list[torch.Tensor] = []
    delays: list[torch.Tensor] = []
    sizes: list[int] = []
    for graph_batch, context_conflicts, context_delays, _ in samples:
        graphs.extend(cast(list[Data], graph_batch.to_data_list()))
        conflicts.append(context_conflicts)
        delays.append(context_delays)
        sizes.append(len(context_conflicts))
    combined = Batch.from_data_list(cast(Any, graphs)).to(device)
    conflict_targets = torch.cat(conflicts).to(device)
    delay_targets = torch.cat(delays).to(device)
    combined_any = cast(Any, combined)
    outcomes, scores = model(
        combined_any.x,
        combined_any.edge_index,
        combined_any.edge_attr,
        combined_any.batch,
        getattr(combined_any, "action_features", None),
    )
    return outcomes, scores, conflict_targets, delay_targets, sizes


def _context_batch_loss(
    outcomes: torch.Tensor,
    scores: torch.Tensor,
    conflicts: torch.Tensor,
    delay_s: torch.Tensor,
    sizes: list[int],
) -> torch.Tensor:
    losses = []
    offset = 0
    for size in sizes:
        end = offset + size
        losses.append(
            _loss(
                outcomes[offset:end],
                scores[offset:end],
                conflicts[offset:end],
                delay_s[offset:end],
            )
        )
        offset = end
    return torch.stack(losses).mean()


def _evaluate(
    model: ReplayPlanGNN,
    batches: list[tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    context_batch_size: int,
    scenario_group_ids: list[str],
) -> dict[str, Any]:
    model.eval()
    correct_pairs = 0
    comparable_pairs = 0
    top1_conflict_regret: list[float] = []
    equal_conflict_delay_regret: list[float] = []
    min_conflict_selections = 0
    min_edit_conflict_regret: list[float] = []
    min_edit_min_conflict_selections = 0
    conflict_errors: list[float] = []
    delay_errors: list[float] = []
    total_loss = 0.0
    total_context_loss_count = 0
    group_loss_totals: dict[str, float] = defaultdict(float)
    group_loss_counts: dict[str, int] = defaultdict(int)
    if len(scenario_group_ids) != len(batches):
        raise ValueError("Each evaluation context must have one scenario-group identifier")
    with torch.inference_mode():
        for offset in range(0, len(batches), context_batch_size):
            samples = batches[offset : offset + context_batch_size]
            outcomes, scores, conflicts, delay_s, sizes = _forward_context_batch(
                model, samples, device
            )
            conflict_prediction = F.softplus(outcomes[:, 0])
            delay_prediction = F.softplus(outcomes[:, 1]) * DELAY_SCALE_S
            conflict_errors.extend((conflict_prediction - conflicts).abs().tolist())
            delay_errors.extend((delay_prediction - delay_s).abs().tolist())
            offset_graphs = 0
            for context_index, size in enumerate(sizes):
                end_graphs = offset_graphs + size
                context_loss = _loss(
                    outcomes[offset_graphs:end_graphs],
                    scores[offset_graphs:end_graphs],
                    conflicts[offset_graphs:end_graphs],
                    delay_s[offset_graphs:end_graphs],
                )
                group = scenario_group_ids[offset + context_index]
                group_loss_totals[group] += float(context_loss)
                group_loss_counts[group] += 1
                total_loss += float(context_loss)
                total_context_loss_count += 1
                context_scores = scores[offset_graphs:end_graphs]
                context_conflicts = conflicts[offset_graphs:end_graphs]
                context_delays = delay_s[offset_graphs:end_graphs]
                selected = int(torch.argmin(context_scores))
                minimum_conflicts = context_conflicts.min()
                top1_conflict_regret.append(float(context_conflicts[selected] - minimum_conflicts))
                if context_conflicts[selected] == minimum_conflicts:
                    min_conflict_selections += 1
                    minimum_delay = context_delays[context_conflicts == minimum_conflicts].min()
                    equal_conflict_delay_regret.append(
                        float(context_delays[selected] - minimum_delay)
                    )
                edit_cost = samples[context_index][3]
                minimum_edit = min(
                    range(size),
                    key=lambda index: (
                        float(edit_cost[index, 0]),
                        float(edit_cost[index, 1]),
                        index,
                    ),
                )
                min_edit_conflict_regret.append(
                    float(context_conflicts[minimum_edit] - minimum_conflicts)
                )
                min_edit_min_conflict_selections += int(
                    context_conflicts[minimum_edit] == minimum_conflicts
                )
                left, right = torch.triu_indices(size, size, offset=1, device=device)
                conflict_difference = context_conflicts[left] - context_conflicts[right]
                delay_difference = context_delays[left] - context_delays[right]
                actual = torch.where(
                    conflict_difference != 0, conflict_difference, delay_difference
                )
                mask = actual.abs() > 1e-6
                comparable_pairs += int(mask.sum())
                correct_pairs += int(
                    (
                        (context_scores[left[mask]] - context_scores[right[mask]]) * actual[mask]
                        > 0
                    ).sum()
                )
                offset_graphs = end_graphs
    group_mean_losses = {
        group: group_loss_totals[group] / group_loss_counts[group]
        for group in sorted(group_loss_totals)
    }
    return {
        "context_count": len(batches),
        "pairwise_concordance": correct_pairs / max(comparable_pairs, 1),
        "comparable_pair_count": comparable_pairs,
        "mean_top1_conflict_regret": sum(top1_conflict_regret) / max(len(top1_conflict_regret), 1),
        "mean_delay_regret_when_min_conflicts_s": sum(equal_conflict_delay_regret)
        / max(len(equal_conflict_delay_regret), 1),
        "selected_min_conflict_fraction": min_conflict_selections / max(len(batches), 1),
        "minimum_edit_mean_conflict_regret": sum(min_edit_conflict_regret)
        / max(len(min_edit_conflict_regret), 1),
        "minimum_edit_min_conflict_fraction": min_edit_min_conflict_selections
        / max(len(batches), 1),
        "conflict_count_mae": sum(conflict_errors) / max(len(conflict_errors), 1),
        "positive_delay_seconds_mae": sum(delay_errors) / max(len(delay_errors), 1),
        "loss": total_loss / max(total_context_loss_count, 1),
        "scenario_group_macro_loss": sum(group_mean_losses.values())
        / max(len(group_mean_losses), 1),
        "scenario_group_worst_loss": max(group_mean_losses.values(), default=0.0),
        "scenario_group_count": len(group_mean_losses),
        "scenario_group_losses": group_mean_losses,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    (
        train_contexts,
        sums,
        square_sums,
        node_count,
        train_groups_list,
        train_hashes,
        _,
    ) = _load_graph_partition(args.train_root, "train")
    train_actions = [
        json.loads(signature) for context in train_contexts for _, _, signature in context
    ]
    train_alternatives = [len(context) for context in train_contexts]
    train_groups = set(train_groups_list)
    mean64 = sums / max(node_count, 1)
    variance = (square_sums / max(node_count, 1) - mean64.square()).clamp_min(1e-12)
    mean, scale = mean64.to(torch.float32), variance.sqrt().to(torch.float32).clamp_min(1e-6)
    train_batches = _normalize_contexts(train_contexts, mean, scale)
    del train_contexts, sums, square_sums

    (
        validation_contexts,
        _,
        _,
        _,
        validation_groups_list,
        validation_hashes,
        validation_context_group_ids,
    ) = _load_graph_partition(args.validation_root, "validation")
    validation_actions = [
        json.loads(signature) for context in validation_contexts for _, _, signature in context
    ]
    validation_alternatives = [len(context) for context in validation_contexts]
    validation_groups = set(validation_groups_list)
    if train_groups & validation_groups:
        raise ValueError("Scenario groups must be disjoint across train and validation")
    validation_batches = _normalize_contexts(validation_contexts, mean, scale)
    del validation_contexts

    (
        test_contexts,
        _,
        _,
        _,
        test_groups_list,
        test_hashes,
        test_context_group_ids,
    ) = _load_graph_partition(args.test_root, "test")
    test_actions = [
        json.loads(signature) for context in test_contexts for _, _, signature in context
    ]
    test_alternatives = [len(context) for context in test_contexts]
    test_groups = set(test_groups_list)
    if train_groups & test_groups or validation_groups & test_groups:
        raise ValueError("Scenario groups must be disjoint across train, validation and test")
    test_batches = _normalize_contexts(test_contexts, mean, scale)
    del test_contexts
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    model = ReplayPlanGNN(
        len(FEATURE_NAMES),
        args.hidden_channels,
        args.heads,
        action_channels=len(ACTION_SUMMARY_FEATURE_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir: Path = args.output_dir
    model_path = output_dir / "model.pt"
    report_path = output_dir / "report.json"
    if (model_path.exists() or report_path.exists()) and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output already contains a completed run: {output_dir}")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint-last.pt"
    source_hashes = sorted(train_hashes + validation_hashes + test_hashes)
    signature_payload = {
        "source_hashes": source_hashes,
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "test_groups": sorted(test_groups),
        "features": FEATURE_NAMES,
        "edges": EDGE_FEATURE_NAMES,
        "action_summary_features": ACTION_SUMMARY_FEATURE_NAMES,
        "hidden_channels": args.hidden_channels,
        "heads": args.heads,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "context_batch_size": args.context_batch_size,
        "checkpoint_selection_metric": "scenario_group_macro_loss",
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    resume_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("resume_signature") != resume_signature:
            raise ValueError("Checkpoint data, split or training settings do not match this run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_loss = float(checkpoint["best_loss"])
        stale = int(checkpoint["stale"])
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = list(range(len(train_batches)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        for offset in range(0, len(order), args.context_batch_size):
            samples = [
                train_batches[index] for index in order[offset : offset + args.context_batch_size]
            ]
            optimizer.zero_grad(set_to_none=True)
            outcomes, scores, conflicts, delay_s, sizes = _forward_context_batch(
                model, samples, device
            )
            loss = _context_batch_loss(outcomes, scores, conflicts, delay_s, sizes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = _evaluate(
            model,
            validation_batches,
            device,
            args.context_batch_size,
            validation_context_group_ids,
        )
        # Select for transfer across held-out scenario groups, not for the pooled
        # validation distribution where large groups can dominate checkpoint choice.
        selection_loss = float(validation["scenario_group_macro_loss"])
        improved = selection_loss < best_loss - args.min_delta
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "validation_loss": float(validation["loss"]),
            "validation_scenario_group_macro_loss": selection_loss,
            "validation_pairwise_concordance": float(validation["pairwise_concordance"]),
            "best_epoch": best_epoch,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % args.log_every == 0:
            print(
                f"epoch={epoch} train_loss={record['train_loss']:.4f} "
                f"validation_loss={record['validation_loss']:.4f} best_epoch={best_epoch}",
                flush=True,
            )
        torch.save(
            {
                "epoch": epoch,
                "resume_signature": resume_signature,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "stale": stale,
                "history": history,
                "feature_mean": mean,
                "feature_scale": scale,
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_path.with_suffix(".pt.tmp"),
        )
        checkpoint_path.with_suffix(".pt.tmp").replace(checkpoint_path)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_report = _evaluate(
        model,
        validation_batches,
        device,
        args.context_batch_size,
        validation_context_group_ids,
    )
    # The final test partition is evaluated only after model selection is frozen.
    test_report = _evaluate(
        model,
        test_batches,
        device,
        args.context_batch_size,
        test_context_group_ids,
    )
    payload = {
        "schema": MODEL_SCHEMA,
        "architecture": {
            "name": "two-layer edge-aware GATv2 with mean/max graph readout",
            "input_channels": len(FEATURE_NAMES),
            "hidden_channels": args.hidden_channels,
            "heads": args.heads,
            "edge_channels": len(EDGE_FEATURE_NAMES),
            "action_channels": len(ACTION_SUMMARY_FEATURE_NAMES),
        },
        "feature_names": list(FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "action_summary_feature_names": list(ACTION_SUMMARY_FEATURE_NAMES),
        "feature_mean": mean,
        "feature_scale": scale,
        "state_dict": model.state_dict(),
        "ranking_objective": [
            "minimize new conflicts first",
            "minimize positive existing-service arrival delay among equal-conflict plans",
        ],
        "safety": {
            "research_only": True,
            "osrd_validation_required": True,
            "automatic_plan_acceptance": False,
        },
    }
    torch.save(payload, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema": "osrd-replay-plan-gnn-report/v1",
        "model_sha256": digest,
        "train_context_count": len(train_batches),
        "validation_context_count": len(validation_batches),
        "test_context_count": len(test_batches),
        "candidate_episode_count": {
            "train": sum(len(batch[1]) for batch in train_batches),
            "validation": sum(len(batch[1]) for batch in validation_batches),
            "test": sum(len(batch[1]) for batch in test_batches),
        },
        "action_coverage": _action_coverage(
            [train_actions, validation_actions, test_actions],
            [train_alternatives, validation_alternatives, test_alternatives],
        ),
        "cohort_count": {
            "train": len(train_hashes),
            "validation": len(validation_hashes),
            "test": len(test_hashes),
        },
        "episode_sha256_by_partition": {
            "train": sorted(train_hashes),
            "validation": sorted(validation_hashes),
            "test_diagnostic_only": sorted(test_hashes),
        },
        "train_scenario_groups": sorted(train_groups),
        "validation_scenario_groups": sorted(validation_groups),
        "test_scenario_groups": sorted(test_groups),
        "training_config": {
            "seed": args.seed,
            "hidden_channels": args.hidden_channels,
            "attention_heads": args.heads,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "context_batch_size": args.context_batch_size,
            "patience": args.patience,
            "minimum_validation_improvement": args.min_delta,
            "checkpoint_selection_metric": "scenario_group_macro_loss",
            "maximum_epochs": args.epochs,
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation": validation_report,
        "test_diagnostic_only": test_report,
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "OSRD simulation labels are synthetic counterfactuals, not observed outcomes.",
            "The objective encodes conflict-first ordering and is not operator preference.",
            "Test cohorts used in this corpus may already have been inspected in prior research.",
            "All candidate plans still require complete OSRD validation.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    validation = report["validation"]
    diagnostic = report["test_diagnostic_only"]
    train_line = (
        f"| Train | {report['candidate_episode_count']['train']} | "
        f"{report['train_context_count']} | {report['cohort_count']['train']} | "
        f"{len(report['train_scenario_groups'])} |"
    )
    validation_line = (
        f"| Validation | {report['candidate_episode_count']['validation']} | "
        f"{report['validation_context_count']} | {report['cohort_count']['validation']} | "
        f"{len(report['validation_scenario_groups'])} |"
    )
    test_line = (
        f"| Test, diagnostic only | {report['candidate_episode_count']['test']} | "
        f"{report['test_context_count']} | {report['cohort_count']['test']} | "
        f"{len(report['test_scenario_groups'])} |"
    )
    coverage = report["action_coverage"]
    frequency_text = ", ".join(map(str, coverage["requested_frequency_seconds"]))
    phase_text = ", ".join(map(str, coverage["departure_phase_seconds"]))
    alternative_text = ", ".join(map(str, coverage["matched_alternatives_per_context"]))
    station_ids = coverage["station_operational_point_ids"]
    model_card = f"""# Action-conditioned Replay Plan GNN model card

**Status:** research-only; not promoted or connected to the planning API.

## Intended use

Order a supplied set of action-conditioned candidate timetable replay graphs for
planner review and OSRD simulation. The model does not generate a timetable or
route, does not certify feasibility, and must not accept or publish a plan.

## Architecture and targets

Two edge-aware GATv2 message-passing layers encode services and their shared
track/route/block interactions. Mean and max graph pooling are fused with a
separately encoded graph-level action summary. Plan-level heads predict new
conflict count and positive arrival-delay sum and produce a direct pairwise
ranking score. Ranking supervision uses lexicographic labels: minimize new
conflicts first, then positive arrival delay among equal-conflict alternatives.
Action and node features are computed from proposed inputs and the baseline
replay; treatment labels are used only as targets.

## Training data

| Partition | Candidate episodes | Matched contexts | Cohorts | Scenario groups |
|---|---:|---:|---:|---:|
{train_line}
{validation_line}
{test_line}

Inputs are checksum-verified OSRD simulations conditioned on public
OSM-derived infrastructure and empirical delay-profile strata. OSM RailJSON
metres are converted to Editoast millimetres; all replay groups use deterministic,
seed-derived departure anchors. They are synthetic counterfactual labels, not
observed alternative-timetable outcomes. Exact episode SHA-256 hashes and group
assignments are in `report.json`.

The collected action grid covers {frequency_text}-second frequency, departure
phases {phase_text} seconds, and {alternative_text} matched alternatives per
context. It includes {coverage["station_stop_action_count"]} station-stop
episodes across {len(station_ids)} passenger operational points, with
{coverage["distinct_nonzero_dwell_patterns"]} distinct nonzero dwell patterns.
There are {coverage["route_waypoint_action_count"]} route-waypoint actions in
this corpus. These counts are derived from the actual episode actions.

## Results

Validation: pairwise concordance {validation["pairwise_concordance"]:.3f};
selected plan had minimum conflict count in
{validation["selected_min_conflict_fraction"]:.1%} of contexts, versus
{validation["minimum_edit_min_conflict_fraction"]:.1%} for minimum-change order;
mean conflict regret was {validation["mean_top1_conflict_regret"]:.3f}, versus
{validation["minimum_edit_mean_conflict_regret"]:.3f} for minimum-change order.

Held-out test, diagnostic only (three corridors within the two imported OSM
networks, not independent operator networks): pairwise concordance
{diagnostic["pairwise_concordance"]:.3f}; minimum-conflict selection
{diagnostic["selected_min_conflict_fraction"]:.1%}, versus
{diagnostic["minimum_edit_min_conflict_fraction"]:.1%} for minimum-change order;
mean conflict regret {diagnostic["mean_top1_conflict_regret"]:.3f}, versus
{diagnostic["minimum_edit_mean_conflict_regret"]:.3f} for minimum-change order.
These test figures are not a promotion result. The number of locations and
contexts is small, and no prospective planner or independent operator-network
evaluation has been performed.

## Limitations and safeguards

- The ranking objective is a research policy, not an operator preference model.
- Delay-head error is high for aggregate positive delay and is not a calibrated
  real-world forecast.
- No operator-authorized plan choices or real counterfactual outcomes are in
  the training data; public actuals only condition synthetic OSRD simulations.
- The direct score is not a probability and applies only to the supplied set.
- Every candidate requires complete OSRD pathfinding, simulation, and conflict
  validation. The OSRD result remains authoritative.
- The test corridors were exercised in earlier pipeline smoke work; treat test
  results as diagnostic only. New spatial holdouts and prospective planner
  evaluation are required before considering integration.

## Artifact identity

- Model SHA-256: `{digest}`
- Best epoch: {best_epoch}
- Epochs completed: {len(history)}
- Training seed: {args.seed}
- Elapsed fit time: {report["elapsed_seconds"]:.2f} seconds
"""
    (output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("best_epoch", "validation", "test_diagnostic_only", "elapsed_seconds")
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, action="append", required=True)
    parser.add_argument("--validation-root", type=Path, action="append", required=True)
    parser.add_argument("--test-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--context-batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.context_batch_size < 1:
        parser.error("epochs, patience and context batch size must be positive")
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
