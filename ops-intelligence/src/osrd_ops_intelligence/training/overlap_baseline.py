"""Episode-grouped evaluation of a transparent departure-overlap baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_episodes(data_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in data_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        episode_path = directory / manifest.get("episode_file", "episodes.jsonl")
        content = episode_path.read_bytes()
        expected_hash = manifest.get("episodes_sha256")
        if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Episode checksum mismatch in {episode_path}")
        sources.append(
            {
                "path": str(directory),
                "seed": manifest.get("seed"),
                "episode_count": manifest.get("episode_count"),
                "episodes_sha256": expected_hash,
                "parameters": manifest.get("parameters", {}),
            }
        )
        records = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        declared_count = manifest.get("episode_count")
        if declared_count is not None and int(declared_count) != len(records):
            raise ValueError(
                f"Episode count mismatch in {episode_path}: "
                f"manifest says {declared_count}, file contains {len(records)}"
            )
        for line in records:
            episode = json.loads(line)
            episode_id = episode["episode_id"]
            if episode_id in seen:
                raise ValueError(f"Duplicate episode id across source runs: {episode_id}")
            seen.add(episode_id)
            episodes.append(episode)
    return episodes, sources


def rows_for_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    schedule_ids = episode["outputs"]["schedule_ids"]
    payloads = episode["inputs"]["schedule_payloads"]
    if len(schedule_ids) != len(payloads):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched schedule ids and inputs")
    paired_labels = {
        int(row["train_index"]): row for row in episode["labels"]["paired_by_train_index"]
    }
    if len(paired_labels) != len(schedule_ids):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched paired labels")
    starts = [payload["start_time"] / 1000 for payload in payloads]
    summaries = episode["outputs"].get("simulation_summaries", {})
    rows: list[dict[str, Any]] = []
    for index, schedule_id in enumerate(schedule_ids):
        neighbors = [abs(starts[index] - other) for j, other in enumerate(starts) if j != index]
        nearest_gap = min(neighbors) if neighbors else float("inf")
        summary = summaries.get(str(schedule_id), {}).get("train_schedule", {})
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "nearest_departure_gap_s": nearest_gap,
                "travel_time_s": (summary.get("time") or 0) / 1000,
                "label": int(bool(paired_labels[index]["new_conflict"])),
            }
        )
    return rows


def split_for_episode(episode_id: str, test_fraction: float) -> bool:
    digest = hashlib.sha256(episode_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < test_fraction


def metrics(actual: list[int], predicted: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(actual, predicted, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(actual, predicted, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(actual, predicted, strict=True))
    tn = sum(y == 0 and p == 0 for y, p in zip(actual, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(actual),
        "positive_count": sum(actual),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int]:
    if not actual:
        return {"count": 0, "mae_ms": 0.0, "rmse_ms": 0.0}
    errors = [prediction - value for value, prediction in zip(actual, predicted, strict=True)]
    return {
        "count": len(actual),
        "mae_ms": sum(abs(error) for error in errors) / len(errors),
        "rmse_ms": (sum(error * error for error in errors) / len(errors)) ** 0.5,
    }


def split_by_scenario_group(
    episodes: list[dict[str, Any]],
    test_fraction: float,
    test_scenario_groups: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    group_for_episode = {
        episode["episode_id"]: episode.get("scenario_group", episode["episode_id"])
        for episode in episodes
    }
    groups = sorted(set(group_for_episode.values()))
    if len(groups) < 2:
        raise ValueError("Need at least two scenario groups for a scenario-held-out evaluation")
    if test_scenario_groups:
        unknown_groups = sorted(set(test_scenario_groups) - set(groups))
        if unknown_groups:
            raise ValueError(f"Unknown test scenario groups: {', '.join(unknown_groups)}")
        test_groups = set(test_scenario_groups)
        if len(test_groups) == len(groups):
            raise ValueError("At least one scenario group must remain for training")
        split_method = "explicit scenario groups held out"
    else:
        test_group_count = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
        test_groups = set(
            sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())[
                :test_group_count
            ]
        )
        split_method = "stable hash assignment by scenario_group; entire groups held out"
    train_episodes = [
        episode
        for episode in episodes
        if group_for_episode[episode["episode_id"]] not in test_groups
    ]
    test_episodes = [
        episode for episode in episodes if group_for_episode[episode["episode_id"]] in test_groups
    ]
    if not train_episodes or not test_episodes:
        raise ValueError("Episode split produced an empty train or test partition")
    split = {
        "method": split_method,
        "test_fraction_target": test_fraction,
        "scenario_group_count": len(groups),
        "train_scenario_groups": sorted(
            set(group_for_episode[episode["episode_id"]] for episode in train_episodes)
        ),
        "test_scenario_groups": sorted(test_groups),
        "train_episode_count": len(train_episodes),
        "test_episode_count": len(test_episodes),
        "train_episode_ids": [episode["episode_id"] for episode in train_episodes],
        "test_episode_ids": [episode["episode_id"] for episode in test_episodes],
    }
    return train_episodes, test_episodes, split


def evaluate(
    data_dirs: list[Path],
    test_fraction: float,
    test_scenario_groups: list[str] | None = None,
) -> dict[str, Any]:
    episodes, sources = load_episodes(data_dirs)
    train_episodes, test_episodes, split = split_by_scenario_group(
        episodes, test_fraction, test_scenario_groups
    )
    train = [row for episode in train_episodes for row in rows_for_episode(episode)]
    test = [row for episode in test_episodes for row in rows_for_episode(episode)]
    candidates = sorted({0.0, *(row["nearest_departure_gap_s"] for row in train)})
    candidates += [candidates[-1] + 1.0]
    best_threshold = min(
        candidates,
        key=lambda threshold: (
            -metrics(
                [row["label"] for row in train],
                [int(row["nearest_departure_gap_s"] <= threshold) for row in train],
            )["f1"],
            -metrics(
                [row["label"] for row in train],
                [int(row["nearest_departure_gap_s"] <= threshold) for row in train],
            )["precision"],
            threshold,
        ),
    )
    y_test = [row["label"] for row in test]
    overlap_predictions = [int(row["nearest_departure_gap_s"] <= best_threshold) for row in test]
    majority = int(sum(row["label"] for row in train) * 2 >= len(train))
    train_delay_deltas = [
        float(label["arrival_delta_ms"])
        for episode in train_episodes
        for label in episode.get("labels", {}).get("paired_by_train_index", [])
        if label.get("arrival_delta_ms") is not None
    ]
    test_delay_deltas = [
        float(label["arrival_delta_ms"])
        for episode in test_episodes
        for label in episode.get("labels", {}).get("paired_by_train_index", [])
        if label.get("arrival_delta_ms") is not None
    ]
    mean_delay_delta = (
        sum(train_delay_deltas) / len(train_delay_deltas) if train_delay_deltas else 0.0
    )
    return {
        "schema": "osrd-ops-overlap-baseline-report/v1",
        "source_runs": sources,
        "split": split,
        "target": "train schedule is involved in at least one OSRD-reported timetable conflict",
        "baseline": {
            "type": "departure-gap threshold",
            "rule": "predict conflict when nearest other departure gap <= threshold",
            "threshold_s_tuned_on_train_episodes": best_threshold,
            "train_metrics": metrics(
                [row["label"] for row in train],
                [int(row["nearest_departure_gap_s"] <= best_threshold) for row in train],
            ),
            "test_metrics": metrics(y_test, overlap_predictions),
        },
        "majority_class_reference": {
            "predicted_class": majority,
            "test_metrics": metrics(y_test, [majority] * len(y_test)),
        },
        "paired_arrival_delta_baseline": {
            "target": "treatment arrival time minus matched baseline arrival time (milliseconds)",
            "zero_change_test_metrics": regression_metrics(
                test_delay_deltas, [0.0] * len(test_delay_deltas)
            ),
            "training_mean_test_metrics": regression_metrics(
                test_delay_deltas, [mean_delay_delta] * len(test_delay_deltas)
            ),
            "training_mean_prediction_ms": mean_delay_delta,
        },
        "data_counts": {
            "episode_count": len(episodes),
            "schedule_count": len(train) + len(test),
            "conflict_involved_schedule_count": sum(row["label"] for row in train + test),
        },
        "limitations": [
            "Synthetic paired delays only; no field operating-delay outcomes.",
            "The threshold is tuned on this small benchmark and is not a calibrated risk model.",
            ("OSRD participants are positive; all other schedules are negative."),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="+", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--test-scenario-groups",
        nargs="+",
        help="Explicit scenario groups to hold out (overrides hash-based test fraction).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("test-fraction must be between 0.05 and 0.5")
    report = evaluate(args.data_dirs, args.test_fraction, args.test_scenario_groups)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
