"""Fit and evaluate bounded CPU delay-forecast baselines on the pinned RIDE split."""

# NumPy ships partial stubs; keep strict checks focused on the surrounding contract.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.calibration import calibrate_artifact
from osrd_ops_intelligence.models.forecast import predict as predict_forecast
from osrd_ops_intelligence.models.ride_mlp import fit_masked_mlp, predict_mlp
from osrd_ops_intelligence.training.ride_training_data import FILES

MODEL_SCHEMA = "osrd-ops-ride-delay-forecast/v2"
DATA_SCHEMA = "osrd-ride-training-inputs/v1"
FORECAST_QUANTILES = (0.1, 0.5, 0.9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(indices: Any) -> str:
    """Hash sampled row indices in a platform-independent integer encoding."""
    import numpy as np

    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _load_arrays(dataset_dir: Path) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Install the optional numeric dependencies: uv sync --extra rail-data"
        ) from error

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATA_SCHEMA:
        raise ValueError("Unsupported RIDE training-input manifest")
    if manifest.get("revision") != "e5c1ed1126bca1784eb6ea93816dd6cc89afa544":
        raise ValueError("Unexpected RIDE dataset revision")
    for relative_path, expected_digest in FILES.items():
        path = dataset_dir / relative_path
        if not path.is_file() or _sha256(path) != expected_digest:
            raise ValueError(f"Missing or invalid pinned RIDE file: {relative_path}")

    root = dataset_dir / "tabular" / "train"
    features = np.load(root / "x.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(root / "y.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(root / "y_mask.npy", mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or targets.ndim != 2 or mask.shape != targets.shape:
        raise ValueError("RIDE feature, target and validity-mask dimensions are inconsistent")
    if features.shape[0] != targets.shape[0] or targets.shape[1] != 15:
        raise ValueError("RIDE arrays must align and contain the documented 15 target horizons")

    # md.npy is object-dtype in this exact public release. Its SHA-256 is checked
    # above against the pinned upstream artifact before NumPy deserializes it.
    metadata = np.load(root / "md.npy", allow_pickle=True)
    if metadata.shape != (features.shape[0], 3):
        raise ValueError("RIDE metadata does not match the documented row count and 3 fields")
    service_dates = [_normalize_service_date(value) for value in metadata[:, 2]]
    if not all(_is_iso_date(value) for value in service_dates):
        raise ValueError("RIDE metadata contains an invalid service date")
    return features, targets, mask, service_dates, manifest


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _normalize_service_date(value: Any) -> str:
    """Normalize RIDE's ``DDMMMYYYY`` metadata date to ISO format."""
    text = str(value)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        try:
            return datetime.strptime(text, "%d%b%Y").date().isoformat()
        except ValueError as error:
            raise ValueError(f"Invalid RIDE service date: {text!r}") from error


def _inverse_ride_delay(values: Any, normalization: dict[str, Any]) -> Any:
    """Invert RIDE's z-score-after-signed-sqrt transform to delay seconds."""
    import numpy as np

    if normalization.get("norm") != "zscore" or normalization.get("sqrt") is not True:
        raise ValueError("Expected RIDE's signed-square-root z-score for delay values")
    root_values = np.asarray(values, dtype=np.float64) * normalization["std"] + normalization[
        "mean"
    ]
    return np.sign(root_values) * np.square(root_values)


def _sample_indices(candidates: list[int], maximum: int, rng: Any) -> Any:
    import numpy as np

    if len(candidates) <= maximum:
        return np.asarray(candidates, dtype=np.int64)
    return np.sort(rng.choice(candidates, size=maximum, replace=False))


@dataclass(frozen=True)
class Partitions:
    """Chronologically split, sampled RIDE arrays shared by all forecast baselines."""

    manifest: Any
    service_dates: Any
    cutoff: Any
    train_candidates: Any
    candidate_groups: Any
    train_indices: Any
    early_stop_indices: Any
    calibration_indices: Any
    evaluation_indices: Any
    x_train: Any
    y_train: Any
    m_train: Any
    x_early: Any
    y_early: Any
    m_early: Any
    x_calibration: Any
    y_calibration: Any
    m_calibration: Any
    x_evaluation: Any
    y_evaluation: Any
    m_evaluation: Any


def prepare_partitions(
    dataset_dir: Path,
    *,
    validation_start: date,
    max_train_rows: int,
    max_validation_rows: int,
    train_rng: Any,
    validation_rng: Any,
) -> Partitions:
    """Load RIDE arrays and build train/early-stop/calibration/evaluation partitions."""
    import numpy as np

    features, targets, mask, service_dates, manifest = _load_arrays(dataset_dir)
    cutoff = validation_start.isoformat()
    train_candidates = [i for i, day in enumerate(service_dates) if day < cutoff]
    validation_candidates = [i for i, day in enumerate(service_dates) if day >= cutoff]
    if not train_candidates or not validation_candidates:
        raise ValueError("Chronological split is empty; inspect RIDE service-date coverage")
    validation_days = sorted({service_dates[index] for index in validation_candidates})
    if len(validation_days) < 3:
        raise ValueError("At least 3 validation service dates are required for grouped splits")
    first_boundary = len(validation_days) // 3
    second_boundary = (2 * len(validation_days)) // 3
    date_groups = (
        set(validation_days[:first_boundary]),
        set(validation_days[first_boundary:second_boundary]),
        set(validation_days[second_boundary:]),
    )
    if any(not group for group in date_groups):
        raise ValueError("Could not form early-stop, calibration, and evaluation date groups")
    candidate_groups = tuple(
        [index for index in validation_candidates if service_dates[index] in group]
        for group in date_groups
    )
    if any(not group for group in candidate_groups):
        raise ValueError("A grouped validation partition contains no rows")

    train_indices = _sample_indices(train_candidates, max_train_rows, train_rng)
    per_partition_limit = max(1, max_validation_rows // 3)
    early_stop_indices, calibration_indices, evaluation_indices = tuple(
        _sample_indices(group, per_partition_limit, validation_rng) for group in candidate_groups
    )
    x_train = np.asarray(features[train_indices], dtype=np.float32)
    y_train = np.asarray(targets[train_indices], dtype=np.float32)
    m_train = np.asarray(mask[train_indices], dtype=bool)
    x_early = np.asarray(features[early_stop_indices], dtype=np.float32)
    y_early = np.asarray(targets[early_stop_indices], dtype=np.float32)
    m_early = np.asarray(mask[early_stop_indices], dtype=bool)
    x_calibration = np.asarray(features[calibration_indices], dtype=np.float32)
    y_calibration = np.asarray(targets[calibration_indices], dtype=np.float32)
    m_calibration = np.asarray(mask[calibration_indices], dtype=bool)
    x_evaluation = np.asarray(features[evaluation_indices], dtype=np.float32)
    y_evaluation = np.asarray(targets[evaluation_indices], dtype=np.float32)
    m_evaluation = np.asarray(mask[evaluation_indices], dtype=bool)
    if not all(
        np.isfinite(array).all()
        for array in (
            x_train,
            y_train[m_train],
            x_early,
            y_early[m_early],
            x_calibration,
            y_calibration[m_calibration],
            x_evaluation,
            y_evaluation[m_evaluation],
        )
    ):
        raise ValueError("RIDE arrays contain non-finite values in usable training examples")
    if not all(
        validity.any(axis=0).all() for validity in (m_train, m_early, m_calibration, m_evaluation)
    ):
        raise ValueError("Each horizon needs valid train, early-stop, calibration and eval targets")
    return Partitions(
        manifest=manifest,
        service_dates=service_dates,
        cutoff=cutoff,
        train_candidates=train_candidates,
        candidate_groups=candidate_groups,
        train_indices=train_indices,
        early_stop_indices=early_stop_indices,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        x_train=x_train,
        y_train=y_train,
        m_train=m_train,
        x_early=x_early,
        y_early=y_early,
        m_early=m_early,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
        m_calibration=m_calibration,
        x_evaluation=x_evaluation,
        y_evaluation=y_evaluation,
        m_evaluation=m_evaluation,
    )


def train(
    dataset_dir: Path,
    *,
    output_dir: Path,
    validation_start: date = date(2024, 10, 1),
    max_train_rows: int = 20_000,
    max_validation_rows: int = 5_000,
    epochs: int = 40,
    batch_size: int = 1024,
    seed: int = 42,
    validation_seed: int = 42,
    learning_rate: float | None = None,
    l2: float | None = None,
    model_type: str = "linear",
    hidden_dims: tuple[int, ...] = (256, 128),
) -> dict[str, Any]:
    import numpy as np

    effective_learning_rate = (
        learning_rate if learning_rate is not None else (0.001 if model_type == "mlp" else 0.01)
    )
    effective_l2 = l2 if l2 is not None else (0.0001 if model_type == "mlp" else 0.01)
    if effective_learning_rate <= 0 or effective_l2 < 0:
        raise ValueError("learning rate must be positive and L2 penalty cannot be negative")
    if min(max_train_rows, epochs, batch_size) <= 0 or max_validation_rows < 3:
        raise ValueError("Training limits must be positive and validation rows must be at least 3")
    train_rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(validation_seed)
    optimization_rng = np.random.default_rng(seed)
    parts = prepare_partitions(
        dataset_dir,
        validation_start=validation_start,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        train_rng=train_rng,
        validation_rng=validation_rng,
    )
    manifest = parts.manifest
    service_dates = parts.service_dates
    cutoff = parts.cutoff
    train_candidates = parts.train_candidates
    candidate_groups = parts.candidate_groups
    train_indices = parts.train_indices
    early_stop_indices = parts.early_stop_indices
    calibration_indices = parts.calibration_indices
    evaluation_indices = parts.evaluation_indices
    x_train = parts.x_train
    y_train = parts.y_train
    m_train = parts.m_train
    x_early = parts.x_early
    y_early = parts.y_early
    m_early = parts.m_early
    x_calibration = parts.x_calibration
    y_calibration = parts.y_calibration
    m_calibration = parts.m_calibration
    x_evaluation = parts.x_evaluation
    y_evaluation = parts.y_evaluation
    m_evaluation = parts.m_evaluation

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Install the optional data dependencies: uv sync --extra rail-data"
        ) from error
    scheme = yaml.safe_load((dataset_dir / "tabular" / "scheme.yaml").read_text())
    normalization = yaml.safe_load(
        (dataset_dir / "tabular" / "normalization.yaml").read_text()
    )
    feature_columns = scheme["x_columns"]
    last_delay_column = feature_columns.index("past_delay_sec_1")
    last_known_delay_seconds = _inverse_ride_delay(
        x_evaluation[:, last_delay_column], normalization["past_delay_sec_1"]
    )
    target_delay_change_seconds = np.column_stack(
        [
            _inverse_ride_delay(y_evaluation[:, horizon], normalization[f"future_delay_delta_{horizon + 1}"])
            for horizon in range(y_evaluation.shape[1])
        ]
    )
    # RIDE's learned labels are future delay changes relative to the last known
    # delay, while its shared evaluation target is absolute future delay.
    target_seconds = target_delay_change_seconds + last_known_delay_seconds[:, None]

    # A train-only target mean is a stronger and leakage-safe reference than
    # assuming normalized zero is the mean for every horizon.
    target_means = np.asarray(
        [y_train[m_train[:, horizon], horizon].mean() for horizon in range(y_train.shape[1])],
        dtype=np.float32,
    )

    if model_type == "linear":
        # Optimize only observed labels; missing targets never become zero-delay labels.
        y_train_filled = np.where(m_train, y_train, 0.0)
        weights = np.zeros((x_train.shape[1], y_train.shape[1]), dtype=np.float32)
        bias = np.zeros(y_train.shape[1], dtype=np.float32)
        first_moment_w = np.zeros_like(weights)
        second_moment_w = np.zeros_like(weights)
        first_moment_b = np.zeros_like(bias)
        second_moment_b = np.zeros_like(bias)
        step = 0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for _epoch in range(1, epochs + 1):
            order = optimization_rng.permutation(len(x_train))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                x_batch = x_train[batch]
                valid_batch = m_train[batch]
                counts = np.maximum(valid_batch.sum(axis=0), 1).astype(np.float32)
                errors = (x_batch @ weights + bias - y_train_filled[batch]) * valid_batch
                gradient_w = (x_batch.T @ errors) / counts + effective_l2 * weights
                gradient_b = errors.sum(axis=0) / counts
                step += 1
                first_moment_w = beta1 * first_moment_w + (1 - beta1) * gradient_w
                second_moment_w = beta2 * second_moment_w + (1 - beta2) * gradient_w**2
                first_moment_b = beta1 * first_moment_b + (1 - beta1) * gradient_b
                second_moment_b = beta2 * second_moment_b + (1 - beta2) * gradient_b**2
                correction1 = 1 - beta1**step
                correction2 = 1 - beta2**step
                weights -= (
                    effective_learning_rate
                    * (first_moment_w / correction1)
                    / (np.sqrt(second_moment_w / correction2) + epsilon)
                )
                bias -= (
                    effective_learning_rate
                    * (first_moment_b / correction1)
                    / (np.sqrt(second_moment_b / correction2) + epsilon)
                )
        predictions = x_evaluation @ weights + bias
        model_parameters: dict[str, Any] = {"weights": weights.tolist(), "bias": bias.tolist()}
        algorithm = "masked multi-output linear regression with Adam and L2 penalty"
    elif model_type == "mlp":
        input_mean, input_scale, layer_weights, layer_biases, fit_metadata = fit_masked_mlp(
            x_train,
            y_train,
            m_train,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=effective_learning_rate,
            l2=effective_l2,
            seed=seed,
            validation_features=x_early,
            validation_targets=y_early,
            validation_mask=m_early,
            quantiles=FORECAST_QUANTILES,
        )
        model_parameters = {
            "input_mean": input_mean.tolist(),
            "input_scale": input_scale.tolist(),
            "weights": [layer.tolist() for layer in layer_weights],
            "biases": [layer.tolist() for layer in layer_biases],
            "hidden_dims": list(hidden_dims),
            "activation": "relu",
            "loss": "horizon-wise masked pinball loss",
            "quantiles": list(FORECAST_QUANTILES),
            "fit_metadata": fit_metadata,
        }
        algorithm = "masked p10/p50/p90 feed-forward quantile MLP with Adam and L2 penalty"
    else:
        raise ValueError("model_type must be 'linear' or 'mlp'")

    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "dataset_manifest_sha256": _sha256(dataset_dir / "manifest.json"),
        "source_training_files": manifest["files"],
        "target": "RIDE provider-normalized future_delay_delta targets at the next 15 events",
        "prediction_space": "provider-normalized RIDE future_delay_delta target units",
        "model_type": model_type,
        "feature_schema_sha256": FILES["tabular/scheme.yaml"],
        "normalization_sha256": FILES["tabular/normalization.yaml"],
        "algorithm": algorithm,
        "training_config": {
            "validation_start_service_date": cutoff,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate": effective_learning_rate,
            "l2_penalty": effective_l2,
            "max_train_rows": max_train_rows,
            "max_validation_rows_total": max_validation_rows,
            "sample_index_sha256": {
                name: hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
                for name, indices in (
                    ("train", train_indices),
                    ("early_stop", early_stop_indices),
                    ("calibration", calibration_indices),
                    ("evaluation", evaluation_indices),
                )
            },
        },
        **model_parameters,
        "safety": {
            "use": "offline delay forecast research and candidate prioritization only",
            "not_a_plan_feasibility_certificate": True,
            "osrd_validation_required_for_plans": True,
        },
    }
    model["model_sha256"] = hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    quantile_crossing: Any = None
    quantile_predictions: Any = None
    if model_type == "mlp":
        calibration_predictions = predict_mlp(x_calibration, model)
        model = calibrate_artifact(
            model,
            calibration_predictions,
            y_calibration,
            m_calibration,
            coverage=0.8,
        )
        raw_quantile_predictions = predict_mlp(x_evaluation, model, sort_quantiles=False)
        quantile_crossing = (
            raw_quantile_predictions[:, :, 0] > raw_quantile_predictions[:, :, 1]
        ) | (raw_quantile_predictions[:, :, 1] > raw_quantile_predictions[:, :, 2])
        quantile_predictions = predict_forecast(x_evaluation, model)
        predictions = quantile_predictions[:, :, 1]

    predicted_delay_change_seconds = np.column_stack(
        [
            _inverse_ride_delay(predictions[:, horizon], normalization[f"future_delay_delta_{horizon + 1}"])
            for horizon in range(y_evaluation.shape[1])
        ]
    )
    predictions_seconds = predicted_delay_change_seconds + last_known_delay_seconds[:, None]
    persistence_seconds_by_horizon = np.repeat(
        last_known_delay_seconds[:, None], y_evaluation.shape[1], axis=1
    )

    per_horizon: list[dict[str, float | int]] = []
    macro_mae_seconds: list[float] = []
    macro_rmse_seconds: list[float] = []
    macro_persistence_mae_seconds: list[float] = []
    for horizon in range(y_evaluation.shape[1]):
        valid = m_evaluation[:, horizon]
        errors = predictions[valid, horizon] - y_evaluation[valid, horizon]
        baseline_errors = target_means[horizon] - y_evaluation[valid, horizon]
        zero_errors = -y_evaluation[valid, horizon]
        errors_seconds = predictions_seconds[valid, horizon] - target_seconds[valid, horizon]
        persistence_errors_seconds = (
            persistence_seconds_by_horizon[valid, horizon] - target_seconds[valid, horizon]
        )
        model_mae = float(np.mean(np.abs(errors)))
        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        model_mae_seconds = float(np.mean(np.abs(errors_seconds)))
        model_rmse_seconds = float(np.sqrt(np.mean(errors_seconds**2)))
        persistence_mae_seconds = float(np.mean(np.abs(persistence_errors_seconds)))
        macro_mae_seconds.append(model_mae_seconds)
        macro_rmse_seconds.append(model_rmse_seconds)
        macro_persistence_mae_seconds.append(persistence_mae_seconds)
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "valid_count": int(valid.sum()),
                "mae_normalized": model_mae,
                "rmse_normalized": float(np.sqrt(np.mean(errors**2))),
                "bias_normalized": float(np.mean(errors)),
                "train_mean_baseline_mae_normalized": baseline_mae,
                "train_mean_baseline_rmse_normalized": float(np.sqrt(np.mean(baseline_errors**2))),
                "mae_improvement_over_train_mean_pct": (
                    float(100 * (baseline_mae - model_mae) / baseline_mae)
                    if baseline_mae > 0
                    else 0.0
                ),
                "zero_normalized_baseline_mae": float(np.mean(np.abs(zero_errors))),
                "mae_seconds": model_mae_seconds,
                "rmse_seconds": model_rmse_seconds,
                "bias_seconds": float(np.mean(errors_seconds)),
                "persistence_baseline_mae_seconds": persistence_mae_seconds,
                "persistence_baseline_rmse_seconds": float(
                    np.sqrt(np.mean(persistence_errors_seconds**2))
                ),
                "mae_improvement_over_persistence_pct": (
                    float(
                        100
                        * (persistence_mae_seconds - model_mae_seconds)
                        / persistence_mae_seconds
                    )
                    if persistence_mae_seconds > 0
                    else 0.0
                ),
            }
        )

    evaluation_months = sorted({service_dates[index][:7] for index in evaluation_indices})
    by_service_month: list[dict[str, Any]] = []
    for month in evaluation_months:
        month_rows = np.asarray(
            [service_dates[index][:7] == month for index in evaluation_indices], dtype=bool
        )
        month_horizons: list[dict[str, float | int]] = []
        for horizon in range(y_evaluation.shape[1]):
            valid = m_evaluation[:, horizon] & month_rows
            if not valid.any():
                continue
            errors = predictions[valid, horizon] - y_evaluation[valid, horizon]
            month_horizons.append(
                {
                    "horizon": horizon + 1,
                    "valid_count": int(valid.sum()),
                    "mae_normalized": float(np.mean(np.abs(errors))),
                    "rmse_normalized": float(np.sqrt(np.mean(errors**2))),
                    "bias_normalized": float(np.mean(errors)),
                }
            )
        by_service_month.append({"service_month": month, "by_horizon": month_horizons})

    quantile_report: list[dict[str, Any]] = []
    if model_type == "mlp":
        for horizon in range(y_evaluation.shape[1]):
            valid = m_evaluation[:, horizon]
            actual = y_evaluation[valid, horizon]
            quantile_values = quantile_predictions[valid, horizon, :]
            residuals = actual[:, None] - quantile_values
            levels = np.asarray(FORECAST_QUANTILES, dtype=np.float32)
            pinball = np.maximum(levels * residuals, (levels - 1) * residuals).mean(axis=0)
            actual_seconds = target_seconds[valid, horizon]
            quantile_seconds = _inverse_ride_delay(
                quantile_values, normalization[f"future_delay_delta_{horizon + 1}"]
            ) + last_known_delay_seconds[valid, None]
            residuals_seconds = actual_seconds[:, None] - quantile_seconds
            pinball_seconds = np.maximum(
                levels * residuals_seconds, (levels - 1) * residuals_seconds
            ).mean(axis=0)
            quantile_report.append(
                {
                    "horizon": horizon + 1,
                    "valid_count": int(valid.sum()),
                    "quantile_levels": list(FORECAST_QUANTILES),
                    "pinball_loss_normalized": [float(value) for value in pinball],
                    "pinball_loss_seconds": [float(value) for value in pinball_seconds],
                    "calibrated_p10_p90_coverage": float(
                        np.mean(
                            (actual >= quantile_values[:, 0]) & (actual <= quantile_values[:, 2])
                        )
                    ),
                    "nominal_p10_p90_coverage": 0.8,
                    "quantile_crossing_rate_before_rearrangement": float(
                        quantile_crossing[valid].mean()
                    ),
                    "mean_p10_p90_width_normalized": float(
                        np.mean(quantile_values[:, 2] - quantile_values[:, 0])
                    ),
                    "mean_p10_p90_width_seconds": float(
                        np.mean(quantile_seconds[:, 2] - quantile_seconds[:, 0])
                    ),
                }
            )

    report: dict[str, Any] = {
        "schema": "osrd-ops-ride-delay-evaluation/v1",
        "model_sha256": model["model_sha256"],
        "dataset_revision": manifest["revision"],
        "validation_start_service_date": cutoff,
        "train_service_date_range": [
            min(service_dates[i] for i in train_indices),
            max(service_dates[i] for i in train_indices),
        ],
        "early_stop_service_date_range": [
            min(service_dates[i] for i in early_stop_indices),
            max(service_dates[i] for i in early_stop_indices),
        ],
        "calibration_service_date_range": [
            min(service_dates[i] for i in calibration_indices),
            max(service_dates[i] for i in calibration_indices),
        ],
        "evaluation_service_date_range": [
            min(service_dates[i] for i in evaluation_indices),
            max(service_dates[i] for i in evaluation_indices),
        ],
        "train_sample_rows": len(train_indices),
        "early_stop_sample_rows": len(early_stop_indices),
        "calibration_sample_rows": len(calibration_indices),
        "evaluation_sample_rows": len(evaluation_indices),
        "sampled_row_indices_sha256": {
            "train": _indices_sha256(train_indices),
            "early_stop": _indices_sha256(early_stop_indices),
            "calibration": _indices_sha256(calibration_indices),
            "evaluation": _indices_sha256(evaluation_indices),
        },
        "training_rows_available_before_sampling": len(train_candidates),
        "validation_rows_available_before_sampling": {
            name: len(group)
            for name, group in zip(
                ("early_stop", "calibration", "evaluation"), candidate_groups, strict=True
            )
        },
        "epochs_requested": epochs,
        "epochs_run": (model_parameters.get("fit_metadata", {}).get("epochs_run", epochs)),
        "learning_rate": effective_learning_rate,
        "l2_penalty": effective_l2,
        "model_type": model_type,
        "hidden_dims": list(hidden_dims) if model_type == "mlp" else [],
        "seed": seed,
        "validation_seed": validation_seed,
        "metrics": per_horizon,
        "target_unit": "absolute future delay in seconds (last known delay plus inverse-transformed RIDE delay change)",
        "training_label_unit": "future delay change relative to last known delay, in provider-normalized units",
        "macro_metrics_seconds": {
            "mae": float(np.mean(macro_mae_seconds)),
            "rmse": float(np.mean(macro_rmse_seconds)),
            "persistence_baseline_mae": float(np.mean(macro_persistence_mae_seconds)),
            "mae_improvement_over_persistence_pct": (
                float(
                    100
                    * (np.mean(macro_persistence_mae_seconds) - np.mean(macro_mae_seconds))
                    / np.mean(macro_persistence_mae_seconds)
                )
                if np.mean(macro_persistence_mae_seconds) > 0
                else 0.0
            ),
        },
        "quantile_metrics": quantile_report,
        "interval_calibration": model.get("interval_calibration"),
        "quantile_crossing_rate_before_rearrangement": (
            float(quantile_crossing[m_evaluation].mean()) if model_type == "mlp" else None
        ),
        "metric_space": (
            "provider-normalized target values; apply pinned normalization to convert to seconds"
        ),
        "evaluation_by_service_month": by_service_month,
        "test_period_used": False,
        "limitations": [
            "This is an observed-delay forecaster, not an OSRD candidate feasibility model.",
            "A bounded deterministic sample is used; rerun with larger limits before conclusions.",
            "The official 2025 holdout was not used for fitting or tuning.",
            "Internal early-stop, calibration, and evaluation partitions are date-grouped "
            "and sampled separately.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir", type=Path, nargs="?", default=Path("data/rail/ride-gold-lite-training")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2024, 10, 1))
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    parser.add_argument("--max-validation-rows", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--model", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dimensions", type=int, nargs="+", default=[256, 128])
    args = parser.parse_args()
    default_output = (
        Path("data/rail/ride-forecast-model")
        if args.model == "linear"
        else Path("data/rail/ride-forecast-mlp")
    )
    output_dir = args.output_dir or default_output
    try:
        report = train(
            args.dataset_dir,
            output_dir=output_dir,
            validation_start=args.validation_start,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_seed=args.validation_seed,
            learning_rate=args.learning_rate,
            l2=args.l2,
            model_type=args.model,
            hidden_dims=tuple(args.hidden_dimensions),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
