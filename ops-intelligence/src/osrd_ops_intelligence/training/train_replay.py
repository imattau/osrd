"""Train and evaluate compact non-LLM models from paired replay datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.features.replay_v1 import FEATURE_NAMES, FEATURE_SCHEMA, rows_for_episode
from osrd_ops_intelligence.models.linear import (
    fit_logistic,
    fit_ridge,
    predict_probability,
    predict_regression,
)
from osrd_ops_intelligence.training.overlap_baseline import (
    evaluate as evaluate_overlap_baseline,
)
from osrd_ops_intelligence.training.overlap_baseline import (
    load_episodes,
    metrics,
    regression_metrics,
    split_by_scenario_group,
)


def flatten(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for episode in episodes for row in rows_for_episode(episode)]


def tune_probability_threshold(labels: list[int], probabilities: list[float]) -> float:
    candidates = sorted({0.0, *probabilities, 0.5, 1.0})
    return min(
        candidates,
        key=lambda threshold: (
            -metrics(labels, [int(value >= threshold) for value in probabilities])["f1"],
            -metrics(labels, [int(value >= threshold) for value in probabilities])["precision"],
            threshold,
        ),
    )


def train(
    data_dirs: list[Path],
    test_fraction: float,
    test_scenario_groups: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes, sources = load_episodes(data_dirs)
    train_episodes, test_episodes, split = split_by_scenario_group(
        episodes, test_fraction, test_scenario_groups
    )
    train_rows = flatten(train_episodes)
    test_rows = flatten(test_episodes)
    train_features = [
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in train_rows
    ]
    test_features = [[float(row["features"][name]) for name in FEATURE_NAMES] for row in test_rows]
    train_labels = [int(row["conflict_label"]) for row in train_rows]
    test_labels = [int(row["conflict_label"]) for row in test_rows]
    classifier = fit_logistic(train_features, train_labels)
    train_probabilities = [predict_probability(classifier, row) for row in train_features]
    test_probabilities = [predict_probability(classifier, row) for row in test_features]
    threshold = tune_probability_threshold(train_labels, train_probabilities)
    overlap_report = evaluate_overlap_baseline(data_dirs, test_fraction, test_scenario_groups)

    delay_train = [row for row in train_rows if row["arrival_delta_ms"] is not None]
    delay_test = [row for row in test_rows if row["arrival_delta_ms"] is not None]
    if not delay_train or not delay_test:
        raise ValueError("Paired arrival-delta targets are required in train and test groups")
    delay_train_features = [
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in delay_train
    ]
    delay_test_features = [
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in delay_test
    ]
    delay_train_targets = [float(row["arrival_delta_ms"]) for row in delay_train]
    delay_test_targets = [float(row["arrival_delta_ms"]) for row in delay_test]
    regressor = fit_ridge(delay_train_features, delay_train_targets)
    delay_predictions = [predict_regression(regressor, row) for row in delay_test_features]
    delay_train_mean = sum(delay_train_targets) / len(delay_train_targets)

    model = {
        "schema": "osrd-ops-linear-model/v1",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "training_scenario_groups": split["train_scenario_groups"],
        "source_runs": sources,
        "conflict_classifier": {
            "algorithm": "standardized logistic regression; full-batch gradient descent; L2",
            **classifier,
            "decision_threshold_tuned_on_training_groups": threshold,
        },
        "arrival_delta_regressor": {
            "algorithm": "standardized ridge regression; full-batch gradient descent; L2",
            **regressor,
        },
    }
    report = {
        "schema": "osrd-ops-model-evaluation/v1",
        "feature_schema": FEATURE_SCHEMA,
        "source_runs": sources,
        "split": split,
        "conflict_target": "train newly conflict-involved in treatment relative to paired baseline",
        "conflict_classifier": {
            "threshold": threshold,
            "train_metrics": metrics(
                train_labels, [int(value >= threshold) for value in train_probabilities]
            ),
            "test_metrics": metrics(
                test_labels, [int(value >= threshold) for value in test_probabilities]
            ),
            "test_mean_probability": sum(test_probabilities) / len(test_probabilities),
        },
        "departure_gap_reference": {
            "threshold_s": overlap_report["baseline"]["threshold_s_tuned_on_train_episodes"],
            "test_metrics": overlap_report["baseline"]["test_metrics"],
        },
        "arrival_delta_regressor": {
            "target": "paired treatment minus baseline simulated arrival (milliseconds)",
            "test_metrics": regression_metrics(delay_test_targets, delay_predictions),
            "training_mean_reference_metrics": regression_metrics(
                delay_test_targets, [delay_train_mean] * len(delay_test_targets)
            ),
            "zero_change_reference_metrics": regression_metrics(
                delay_test_targets, [0.0] * len(delay_test_targets)
            ),
        },
        "data_counts": {
            "train_episode_count": len(train_episodes),
            "test_episode_count": len(test_episodes),
            "train_schedule_count": len(train_rows),
            "test_schedule_count": len(test_rows),
            "train_arrival_delta_count": len(delay_train),
            "test_arrival_delta_count": len(delay_test),
        },
        "limitations": [
            "Synthetic scenarios only; these metrics do not estimate field performance.",
            (
                "Only a few scenario groups are available; one held-out split gives weak "
                "uncertainty estimates."
            ),
            "Feature inputs assume a paired baseline simulation is available before scoring.",
            "The model is an offline benchmark and is not approved for operational decisions.",
        ],
    }
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="+", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--test-scenario-groups",
        nargs="+",
        help="Explicit scenario groups to hold out (overrides hash-based test fraction).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("test-fraction must be between 0.05 and 0.5")
    model, report = train(args.data_dirs, args.test_fraction, args.test_scenario_groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "report.json").write_text(rendered_report, encoding="utf-8")
    print(rendered_report, end="")


if __name__ == "__main__":
    main()
