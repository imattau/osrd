"""Train and evaluate an offline OSRD candidate-feasibility ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.linear import fit_logistic, predict_probability
from osrd_ops_intelligence.training.plan_ranker_data import (
    DATASET_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
)

MODEL_SCHEMA = "osrd-ops-plan-feasibility-model/v1"


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float | int]:
    predicted = [int(probability >= 0.5) for probability in probabilities]
    tp = sum(label == 1 and guess == 1 for label, guess in zip(labels, predicted, strict=True))
    fp = sum(label == 0 and guess == 1 for label, guess in zip(labels, predicted, strict=True))
    fn = sum(label == 1 and guess == 0 for label, guess in zip(labels, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(labels),
        "positive_count": sum(labels),
        "positive_rate": sum(labels) / len(labels) if labels else 0.0,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
        "brier_score": (
            sum(
                (probability - label) ** 2
                for label, probability in zip(labels, probabilities, strict=True)
            )
            / len(labels)
            if labels
            else 0.0
        ),
    }


def _top_k_rates(
    rows: list[dict[str, Any]], probabilities: list[float], budgets: tuple[int, ...] = (1, 3, 5)
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        grouped[str(row["scenario_group"])].append((row, probability))
    report: dict[str, Any] = {}
    for budget in budgets:
        learned_rates: list[float] = []
        objective_rates: list[float] = []
        oracle_rates: list[float] = []
        for alternatives in grouped.values():
            count = min(budget, len(alternatives))
            if not count:
                continue
            learned = sorted(
                alternatives,
                key=lambda item: (
                    -item[1],
                    float(item[0]["features"]["objective_score"]),
                    str(item[0]["candidate_id"]),
                ),
            )[:count]
            objective = sorted(
                alternatives,
                key=lambda item: (
                    float(item[0]["features"]["objective_score"]),
                    str(item[0]["candidate_id"]),
                ),
            )[:count]
            oracle = sorted(
                alternatives,
                key=lambda item: (
                    -int(bool(item[0]["targets"]["osrd_validated"])),
                    str(item[0]["candidate_id"]),
                ),
            )[:count]
            learned_rates.append(
                sum(bool(row["targets"]["osrd_validated"]) for row, _ in learned) / count
            )
            objective_rates.append(
                sum(bool(row["targets"]["osrd_validated"]) for row, _ in objective) / count
            )
            oracle_rates.append(
                sum(bool(row["targets"]["osrd_validated"]) for row, _ in oracle) / count
            )
        report[str(budget)] = {
            "scenario_group_count": len(learned_rates),
            "learned_mean_valid_rate": (
                sum(learned_rates) / len(learned_rates) if learned_rates else 0.0
            ),
            "objective_order_mean_valid_rate": (
                sum(objective_rates) / len(objective_rates) if objective_rates else 0.0
            ),
            "oracle_upper_bound_mean_valid_rate": (
                sum(oracle_rates) / len(oracle_rates) if oracle_rates else 0.0
            ),
        }
    return report


def train(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("Unsupported plan candidate dataset schema")
    if manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Unsupported plan candidate feature schema")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Dataset feature names do not match this trainer")
    dataset_path = dataset_dir / str(manifest.get("dataset_file", "candidates.jsonl"))
    content = dataset_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != manifest.get("dataset_sha256"):
        raise ValueError("Candidate dataset checksum mismatch")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Candidate dataset contains no rows")
    split = manifest.get("split", {})
    train_groups = set(split.get("train_scenario_groups", []))
    test_groups = set(split.get("test_scenario_groups", []))
    if not train_groups or not test_groups or train_groups & test_groups:
        raise ValueError("Dataset must contain disjoint non-empty scenario-group splits")
    for row in rows:
        if row.get("schema") != DATASET_SCHEMA or row.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Candidate row schema does not match manifest")
        if row.get("scenario_group") not in train_groups | test_groups:
            raise ValueError("Candidate row references a scenario group outside the manifest split")
        if set(row.get("features", {})) != set(FEATURE_NAMES):
            raise ValueError("Candidate row feature keys do not match the declared feature schema")

    train_rows = [row for row in rows if row["scenario_group"] in train_groups]
    test_rows = [row for row in rows if row["scenario_group"] in test_groups]
    train_labels = [int(bool(row["targets"]["osrd_validated"])) for row in train_rows]
    test_labels = [int(bool(row["targets"]["osrd_validated"])) for row in test_rows]
    if len(set(train_labels)) < 2:
        raise ValueError("Training split needs both OSRD-valid and OSRD-invalid candidates")
    train_features = [
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in train_rows
    ]
    test_features = [[float(row["features"][name]) for name in FEATURE_NAMES] for row in test_rows]
    classifier = fit_logistic(train_features, train_labels)
    train_probabilities = [predict_probability(classifier, row) for row in train_features]
    test_probabilities = [predict_probability(classifier, row) for row in test_features]
    model = {
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "target": "OSRD complete candidate validation succeeds with zero conflicts",
        "dataset_sha256": digest,
        "training_scenario_groups": sorted(train_groups),
        "classifier": {
            "algorithm": "standardized logistic regression; full-batch gradient descent; L2",
            **classifier,
        },
        "safety": {
            "use": "prioritize candidates for OSRD validation",
            "osrd_validation_required": True,
            "not_approved_for_automatic_plan_acceptance": True,
        },
    }
    model_bytes = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    model["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    report = {
        "schema": "osrd-ops-plan-ranker-evaluation/v1",
        "feature_schema": FEATURE_SCHEMA,
        "model_sha256": model["model_sha256"],
        "dataset_sha256": digest,
        "split": split,
        "train_metrics_at_threshold_0_5": _metrics(train_labels, train_probabilities),
        "test_metrics_at_threshold_0_5": _metrics(test_labels, test_probabilities),
        "test_validation_budget_rates": _top_k_rates(test_rows, test_probabilities),
        "limitations": manifest.get("limitations", []),
    }
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        model, report = train(args.dataset_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
