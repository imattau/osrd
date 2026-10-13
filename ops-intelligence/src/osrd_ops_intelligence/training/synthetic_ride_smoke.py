"""Generate a deterministic synthetic forecast fixture and MLP smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_SCHEMA = "osrd-ops-ride-delay-forecast/v2"
FEATURE_COUNT = 12
HORIZONS = 15


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output_dir: Path, *, rows: int = 1024, seed: int = 73) -> dict[str, Any]:
    """Create toy data with learnable nonlinear temporal targets and train an MLP.

    The generated variables have no correspondence to railway or RIDE fields. This is
    only for checking file formats and the model training/inference path.
    """
    import numpy as np

    from osrd_ops_intelligence.models.calibration import calibrate_artifact
    from osrd_ops_intelligence.models.forecast import predict
    from osrd_ops_intelligence.models.ride_mlp import fit_masked_mlp, predict_mlp

    if rows < 80:
        raise ValueError("rows must be at least 80 to form four chronological data splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, FEATURE_COUNT)).astype(np.float32)
    train_end = int(rows * 0.60)
    validation_end = int(rows * 0.75)
    calibration_end = int(rows * 0.875)
    if min(train_end, validation_end - train_end, calibration_end - validation_end, rows - calibration_end) < 8:
        raise ValueError("Each synthetic data split must contain at least eight rows")

    base_signal = 0.65 * features[:, 0] - 0.35 * features[:, 1]
    nonlinear_signal = 0.4 * features[:, 2] * features[:, 3] + 0.2 * np.maximum(
        features[:, 4], 0
    )
    targets = np.empty((rows, HORIZONS), dtype=np.float32)
    for horizon in range(HORIZONS):
        decay = np.exp(-horizon / 12.0)
        targets[:, horizon] = (
            decay * base_signal
            + (1.0 - decay) * nonlinear_signal
            + rng.normal(scale=0.08 + horizon * 0.002, size=rows)
        )

    mask = rng.random((rows, HORIZONS)) > 0.08
    partitions = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "calibration": (validation_end, calibration_end),
        "test": (calibration_end, rows),
    }
    arrays: dict[str, tuple[Any, Any, Any]] = {}
    for name, (start, end) in partitions.items():
        part_mask = mask[start:end]
        for horizon in range(HORIZONS):
            part_mask[0, horizon] = True
        arrays[name] = (features[start:end], targets[start:end], part_mask)
    train_features, train_targets, train_mask = arrays["train"]
    validation_features, validation_targets, validation_mask = arrays["validation"]
    calibration_features, calibration_targets, calibration_mask = arrays["calibration"]
    test_features, test_targets, test_mask = arrays["test"]

    feature_names = [f"synthetic_feature_{index:02d}" for index in range(FEATURE_COUNT)]
    schema = (
        "# Synthetic smoke fixture schema; these fields are not railway features.\n"
        "features:\n"
        + "".join(f"  - {name}\n" for name in feature_names)
    ).encode()
    normalization = (
        "# Identity transform; synthetic fixture only.\n"
        "features: standard_normal\n"
        "targets: synthetic_normalized_delay_units\n"
    ).encode()
    (output_dir / "feature-schema.yaml").write_bytes(schema)
    (output_dir / "normalization.yaml").write_bytes(normalization)
    for name, (part_features, part_targets, part_mask) in arrays.items():
        np.save(output_dir / f"features-{name}.npy", part_features, allow_pickle=False)
        np.save(output_dir / f"targets-{name}.npy", part_targets, allow_pickle=False)
        np.save(output_dir / f"mask-{name}.npy", part_mask, allow_pickle=False)

    input_mean, input_scale, weights, biases, fit_metadata = fit_masked_mlp(
        train_features,
        train_targets,
        train_mask,
        hidden_dims=(32, 16),
        epochs=30,
        batch_size=64,
        learning_rate=0.003,
        l2=0.0001,
        seed=seed,
        validation_features=validation_features,
        validation_targets=validation_targets,
        validation_mask=validation_mask,
    )
    parameters: dict[str, Any] = {
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "weights": [layer.tolist() for layer in weights],
        "biases": [layer.tolist() for layer in biases],
        "hidden_dims": [32, 16],
        "activation": "relu",
        "loss": "horizon-wise masked pinball loss",
        "quantiles": [0.1, 0.5, 0.9],
    }
    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "dataset": "synthetic-smoke-fixture",
        "dataset_revision": f"seed-{seed}",
        "target": "synthetic nonlinear target at the next 15 horizons",
        "model_type": "mlp",
        "algorithm": "synthetic-only feed-forward MLP; not trained on railway data",
        "feature_schema_sha256": _sha256(schema),
        "normalization_sha256": _sha256(normalization),
        **parameters,
        "data_provenance": {
            "synthetic": True,
            "seed": seed,
            "feature_names": feature_names,
            "meaning": "Generated numeric variables; no real rail observations or IDs.",
        },
        "safety": {
            "use": "pipeline smoke checks only",
            "synthetic_data_only": True,
            "operationally_approved": False,
            "not_a_plan_feasibility_certificate": True,
            "osrd_validation_required_for_plans": True,
        },
    }
    model["model_sha256"] = _sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    )
    (output_dir / "model-uncalibrated.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    calibration_predictions = predict_mlp(calibration_features, model)
    np.save(
        output_dir / "predictions-calibration.npy", calibration_predictions, allow_pickle=False
    )
    calibrated_model = calibrate_artifact(
        model,
        calibration_predictions,
        calibration_targets,
        calibration_mask,
        coverage=0.8,
    )
    (output_dir / "model.json").write_text(
        json.dumps(calibrated_model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_predictions = predict_mlp(test_features, calibrated_model, sort_quantiles=False)
    quantile_crossing = (
        (raw_predictions[:, :, 0] > raw_predictions[:, :, 1])
        | (raw_predictions[:, :, 1] > raw_predictions[:, :, 2])
    )
    predictions = predict(test_features, calibrated_model)
    np.save(output_dir / "predictions.npy", predictions, allow_pickle=False)
    horizon_metrics = []
    quantile_metrics = []
    for horizon in range(HORIZONS):
        valid = test_mask[:, horizon]
        actual = test_targets[valid, horizon]
        horizon_predictions = predictions[valid, horizon, :]
        errors = horizon_predictions[:, 1] - actual
        train_valid = train_mask[:, horizon]
        train_mean = float(train_targets[train_valid, horizon].mean())
        baseline_errors = train_mean - actual
        model_mae = float(np.mean(np.abs(errors)))
        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        horizon_metrics.append(
            {
                "horizon": horizon + 1,
                "valid_count": int(valid.sum()),
                "mlp_mae_synthetic_units": model_mae,
                "rmse_synthetic_units": float(np.sqrt(np.mean(errors**2))),
                "train_mean_baseline_mae_synthetic_units": baseline_mae,
                "mae_improvement_over_train_mean_pct": (
                    float(100 * (baseline_mae - model_mae) / baseline_mae)
                    if baseline_mae > 0
                    else 0.0
                ),
            }
        )
        residuals = actual[:, None] - horizon_predictions
        levels = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
        pinball = np.maximum(levels * residuals, (levels - 1) * residuals).mean(axis=0)
        quantile_metrics.append(
            {
                "horizon": horizon + 1,
                "valid_count": int(valid.sum()),
                "calibrated_p10_p90_coverage": float(
                    np.mean(
                        (actual >= horizon_predictions[:, 0])
                        & (actual <= horizon_predictions[:, 2])
                    )
                ),
                "nominal_p10_p90_coverage": 0.8,
                "quantile_crossing_rate_before_rearrangement": float(
                    quantile_crossing[valid].mean()
                ),
                "mean_p10_p90_width_synthetic_units": float(
                    np.mean(horizon_predictions[:, 2] - horizon_predictions[:, 0])
                ),
                "pinball_loss_by_quantile": [float(value) for value in pinball],
            }
        )
    report = {
        "schema": "osrd-synthetic-forecast-smoke-report/v1",
        "synthetic_only": True,
        "seed": seed,
        "rows": rows,
        "split_rows": {name: end - start for name, (start, end) in partitions.items()},
        "fit_metadata": fit_metadata,
        "model_sha256": calibrated_model["model_sha256"],
        "calibration": calibrated_model["interval_calibration"],
        "metrics": horizon_metrics,
        "quantile_metrics": quantile_metrics,
        "quantile_crossing_rate_before_rearrangement": float(
            quantile_crossing[test_mask].mean()
        ),
        "limitations": [
            "Generated variables are not RIDE features or railway observations.",
            "Metrics only confirm a runnable synthetic training path; they say nothing about rail forecasting quality.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rail/ride-synthetic-smoke"))
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    try:
        report = generate(args.output_dir, rows=args.rows, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
