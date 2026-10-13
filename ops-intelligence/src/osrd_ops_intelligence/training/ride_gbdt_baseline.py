"""Fit a gradient-boosted quantile baseline on the same RIDE partitions as the MLP.

The design requires the neural forecaster to beat simpler baselines. This module reports
GBDT metrics in the same units, partitions and split-conformal calibration as
`osrd-ride-train --model mlp`, so the two reports are directly comparable. It writes a
report only; the GBDT is not exported as a servable artifact.
"""

# NumPy ships partial stubs; keep strict checks focused on the surrounding contract.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.calibration import calibrate_interval_expansion
from osrd_ops_intelligence.models.gbdt import fit_gbdt_quantiles, predict_gbdt
from osrd_ops_intelligence.training.train_ride_forecast import (
    FORECAST_QUANTILES,
    prepare_partitions,
)

REPORT_SCHEMA = "osrd-ops-ride-gbdt-baseline/v1"


def quantile_report(
    predictions: Any, targets: Any, mask: Any, levels: tuple[float, ...]
) -> dict[str, Any]:
    """Per-horizon masked pinball loss, outer-interval coverage and width."""
    import numpy as np

    level_array = np.asarray(levels, dtype=np.float32)
    per_horizon: list[dict[str, Any]] = []
    for horizon in range(targets.shape[1]):
        valid = mask[:, horizon].astype(bool)
        actual = targets[valid, horizon]
        values = predictions[valid, horizon, :]
        residual = actual[:, None] - values
        pinball = np.maximum(level_array * residual, (level_array - 1.0) * residual).mean(axis=0)
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "valid_count": int(valid.sum()),
                "pinball_loss_normalized": [float(value) for value in pinball],
                "median_mae_normalized": float(
                    np.mean(np.abs(actual - values[:, len(levels) // 2]))
                ),
                "outer_interval_coverage": float(
                    np.mean((actual >= values[:, 0]) & (actual <= values[:, -1]))
                ),
                "mean_outer_interval_width_normalized": float(
                    np.mean(values[:, -1] - values[:, 0])
                ),
            }
        )
    return {
        "by_horizon": per_horizon,
        "mean_pinball_loss_normalized": float(
            np.mean([np.mean(row["pinball_loss_normalized"]) for row in per_horizon])
        ),
        "mean_outer_interval_coverage": float(
            np.mean([row["outer_interval_coverage"] for row in per_horizon])
        ),
    }


def mlp_mean_pinball(report: dict[str, Any]) -> float:
    """Mean pinball over horizons and quantiles from an `osrd-ride-train` MLP report."""
    rows = report.get("quantile_metrics")
    if report.get("model_type") != "mlp" or not rows:
        raise ValueError("Comparison report must come from `osrd-ride-train --model mlp`")
    return sum(
        sum(row["pinball_loss_normalized"]) / len(row["pinball_loss_normalized"]) for row in rows
    ) / len(rows)


def run(
    dataset_dir: Path,
    *,
    output_dir: Path,
    validation_start: date = date(2024, 10, 1),
    max_train_rows: int = 20_000,
    max_validation_rows: int = 5_000,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    seed: int = 42,
    compare_report: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    parts = prepare_partitions(
        dataset_dir,
        validation_start=validation_start,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        rng=rng,
    )
    model = fit_gbdt_quantiles(
        parts.x_train,
        parts.y_train,
        parts.m_train,
        validation_features=parts.x_early,
        validation_targets=parts.y_early,
        validation_mask=parts.m_early,
        quantiles=FORECAST_QUANTILES,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
    )
    raw = predict_gbdt(model, parts.x_evaluation)
    coverage = FORECAST_QUANTILES[-1] - FORECAST_QUANTILES[0]
    expansion = np.asarray(
        calibrate_interval_expansion(
            predict_gbdt(model, parts.x_calibration),
            parts.y_calibration,
            parts.m_calibration,
            coverage=coverage,
        ),
        dtype=np.float32,
    )
    calibrated = raw.copy()
    calibrated[:, :, 0] -= expansion
    calibrated[:, :, -1] += expansion

    train_mean = np.asarray(
        [parts.y_train[parts.m_train[:, h], h].mean() for h in range(parts.y_train.shape[1])]
    )
    valid = parts.m_evaluation
    mean_baseline_mae = [
        float(np.mean(np.abs(train_mean[h] - parts.y_evaluation[valid[:, h], h])))
        for h in range(valid.shape[1])
    ]

    raw_report = quantile_report(raw, parts.y_evaluation, valid, FORECAST_QUANTILES)
    calibrated_report = quantile_report(calibrated, parts.y_evaluation, valid, FORECAST_QUANTILES)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "dataset_revision": parts.manifest["revision"],
        "validation_start_service_date": parts.cutoff,
        "seed": seed,
        "quantiles": list(FORECAST_QUANTILES),
        "gbdt_config": {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "iterations_selected_on": "early_stop partition (chronological), not a random split",
            "best_iterations": model.best_iterations,
        },
        "rows": {
            "train": len(parts.train_indices),
            "early_stop": len(parts.early_stop_indices),
            "calibration": len(parts.calibration_indices),
            "evaluation": len(parts.evaluation_indices),
        },
        "uncalibrated": raw_report,
        "calibrated": calibrated_report,
        "interval_expansion_by_horizon": [float(value) for value in expansion],
        "train_mean_baseline_mae_by_horizon": mean_baseline_mae,
        "metric_space": "provider-normalized target values (same as osrd-ride-train)",
        "test_period_used": False,
        "limitations": [
            "Observed-delay forecaster only; not an OSRD candidate feasibility model.",
            "A bounded deterministic sample is used; rerun with larger limits before conclusions.",
            "The GBDT is not exported as a servable artifact.",
        ],
    }
    if compare_report is not None:
        mlp_report = json.loads(compare_report.read_text(encoding="utf-8"))
        if (
            mlp_report.get("dataset_revision") != report["dataset_revision"]
            or mlp_report.get("seed") != seed
        ):
            raise ValueError("Comparison report must use the same dataset revision and seed")
        mlp_loss = mlp_mean_pinball(mlp_report)
        gbdt_loss = calibrated_report["mean_pinball_loss_normalized"]
        report["comparison_with_mlp"] = {
            "mlp_report_model_sha256": mlp_report.get("model_sha256"),
            "mlp_mean_pinball_loss_normalized": mlp_loss,
            "gbdt_mean_pinball_loss_normalized": gbdt_loss,
            "mlp_improvement_over_gbdt_pct": float(100 * (gbdt_loss - mlp_loss) / gbdt_loss),
            "note": (
                "Both are calibrated, on identical evaluation rows when seed, row limits and "
                "validation start match. Positive means the MLP is better."
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir", type=Path, nargs="?", default=Path("data/rail/ride-gold-lite-training")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/rail/ride-forecast-gbdt"))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2024, 10, 1))
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    parser.add_argument("--max-validation-rows", type=int, default=5_000)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-report", type=Path, help="report.json from `osrd-ride-train --model mlp`"
    )
    args = parser.parse_args()
    try:
        report = run(
            args.dataset_dir,
            output_dir=args.output_dir,
            validation_start=args.validation_start,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            seed=args.seed,
            compare_report=args.compare_report,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
