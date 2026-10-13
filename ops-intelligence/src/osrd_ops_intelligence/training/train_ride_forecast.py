"""Fit a bounded CPU masked linear forecaster on the pinned RIDE training split."""

# NumPy ships partial stubs; keep strict checks focused on the surrounding contract.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.ride_mlp import fit_masked_mlp, predict_mlp
from osrd_ops_intelligence.training.ride_training_data import FILES

MODEL_SCHEMA = "osrd-ops-ride-delay-forecast/v1"
DATA_SCHEMA = "osrd-ride-training-inputs/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    service_dates = [str(value)[:10] for value in metadata[:, 2]]
    if not all(_is_iso_date(value) for value in service_dates):
        raise ValueError("RIDE metadata contains an invalid service date")
    return features, targets, mask, service_dates, manifest


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _sample_indices(candidates: list[int], maximum: int, rng: Any) -> Any:
    import numpy as np

    if len(candidates) <= maximum:
        return np.asarray(candidates, dtype=np.int64)
    return np.sort(rng.choice(candidates, size=maximum, replace=False))


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
    learning_rate: float | None = None,
    l2: float = 0.01,
    model_type: str = "linear",
    hidden_dims: tuple[int, ...] = (256, 128),
) -> dict[str, Any]:
    import numpy as np

    effective_learning_rate = (
        learning_rate if learning_rate is not None else (0.001 if model_type == "mlp" else 0.01)
    )
    if min(max_train_rows, max_validation_rows, epochs, batch_size) <= 0:
        raise ValueError("Row limits, epochs, and batch size must be positive")
    features, targets, mask, service_dates, manifest = _load_arrays(dataset_dir)
    cutoff = validation_start.isoformat()
    train_candidates = [i for i, day in enumerate(service_dates) if day < cutoff]
    validation_candidates = [i for i, day in enumerate(service_dates) if day >= cutoff]
    if not train_candidates or not validation_candidates:
        raise ValueError("Chronological split is empty; inspect RIDE service-date coverage")

    rng = np.random.default_rng(seed)
    train_indices = _sample_indices(train_candidates, max_train_rows, rng)
    validation_indices = _sample_indices(validation_candidates, max_validation_rows, rng)
    x_train = np.asarray(features[train_indices], dtype=np.float32)
    y_train = np.asarray(targets[train_indices], dtype=np.float32)
    m_train = np.asarray(mask[train_indices], dtype=bool)
    x_valid = np.asarray(features[validation_indices], dtype=np.float32)
    y_valid = np.asarray(targets[validation_indices], dtype=np.float32)
    m_valid = np.asarray(mask[validation_indices], dtype=bool)
    if not all(
        np.isfinite(array).all() for array in (x_train, y_train[m_train], x_valid, y_valid[m_valid])
    ):
        raise ValueError("RIDE arrays contain non-finite values in usable training examples")
    if not m_train.any(axis=0).all() or not m_valid.any(axis=0).all():
        raise ValueError("Each of the 15 horizons needs valid train and validation targets")

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
            order = rng.permutation(len(x_train))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                x_batch = x_train[batch]
                valid_batch = m_train[batch]
                counts = np.maximum(valid_batch.sum(axis=0), 1).astype(np.float32)
                errors = (x_batch @ weights + bias - y_train_filled[batch]) * valid_batch
                gradient_w = (x_batch.T @ errors) / counts + l2 * weights
                gradient_b = errors.sum(axis=0) / counts
                step += 1
                first_moment_w = beta1 * first_moment_w + (1 - beta1) * gradient_w
                second_moment_w = beta2 * second_moment_w + (1 - beta2) * gradient_w**2
                first_moment_b = beta1 * first_moment_b + (1 - beta1) * gradient_b
                second_moment_b = beta2 * second_moment_b + (1 - beta2) * gradient_b**2
                correction1 = 1 - beta1**step
                correction2 = 1 - beta2**step
                weights -= effective_learning_rate * (first_moment_w / correction1) / (
                    np.sqrt(second_moment_w / correction2) + epsilon
                )
                bias -= effective_learning_rate * (first_moment_b / correction1) / (
                    np.sqrt(second_moment_b / correction2) + epsilon
                )
        predictions = x_valid @ weights + bias
        model_parameters: dict[str, Any] = {"weights": weights.tolist(), "bias": bias.tolist()}
        algorithm = "masked multi-output linear regression with Adam and L2 penalty"
    elif model_type == "mlp":
        input_mean, input_scale, layer_weights, layer_biases = fit_masked_mlp(
            x_train,
            y_train,
            m_train,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=effective_learning_rate,
            l2=l2,
            seed=seed,
        )
        model_parameters = {
            "input_mean": input_mean.tolist(),
            "input_scale": input_scale.tolist(),
            "weights": [layer.tolist() for layer in layer_weights],
            "biases": [layer.tolist() for layer in layer_biases],
            "hidden_dims": list(hidden_dims),
            "activation": "relu",
            "loss": "horizon-wise masked mean squared error",
        }
        predictions = predict_mlp(x_valid, model_parameters)
        algorithm = "masked feed-forward MLP with Adam and L2 penalty"
    else:
        raise ValueError("model_type must be 'linear' or 'mlp'")
    per_horizon: list[dict[str, float | int]] = []
    for horizon in range(y_valid.shape[1]):
        valid = m_valid[:, horizon]
        errors = predictions[valid, horizon] - y_valid[valid, horizon]
        baseline_errors = target_means[horizon] - y_valid[valid, horizon]
        zero_errors = -y_valid[valid, horizon]
        model_mae = float(np.mean(np.abs(errors)))
        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "valid_count": int(valid.sum()),
                "mae_normalized": model_mae,
                "rmse_normalized": float(np.sqrt(np.mean(errors**2))),
                "bias_normalized": float(np.mean(errors)),
                "train_mean_baseline_mae_normalized": baseline_mae,
                "train_mean_baseline_rmse_normalized": float(
                    np.sqrt(np.mean(baseline_errors**2))
                ),
                "mae_improvement_over_train_mean_pct": (
                    float(100 * (baseline_mae - model_mae) / baseline_mae)
                    if baseline_mae > 0
                    else 0.0
                ),
                "zero_normalized_baseline_mae": float(np.mean(np.abs(zero_errors))),
            }
        )

    validation_months = sorted(
        {service_dates[index][:7] for index in validation_indices}
    )
    by_service_month: list[dict[str, Any]] = []
    for month in validation_months:
        month_rows = np.asarray(
            [service_dates[index][:7] == month for index in validation_indices], dtype=bool
        )
        month_horizons: list[dict[str, float | int]] = []
        for horizon in range(y_valid.shape[1]):
            valid = m_valid[:, horizon] & month_rows
            if not valid.any():
                continue
            errors = predictions[valid, horizon] - y_valid[valid, horizon]
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

    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "dataset_manifest_sha256": _sha256(dataset_dir / "manifest.json"),
        "source_training_files": manifest["files"],
        "target": "RIDE provider-normalized delay at the next 15 events",
        "feature_schema_sha256": FILES["tabular/scheme.yaml"],
        "normalization_sha256": FILES["tabular/normalization.yaml"],
        "algorithm": algorithm,
        **model_parameters,
        "safety": {
            "use": "offline delay forecast research and candidate prioritization only",
            "not_a_plan_feasibility_certificate": True,
            "osrd_validation_required_for_plans": True,
        },
    }
    model_bytes = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    model["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    report: dict[str, Any] = {
        "schema": "osrd-ops-ride-delay-evaluation/v1",
        "model_sha256": model["model_sha256"],
        "dataset_revision": manifest["revision"],
        "validation_start_service_date": cutoff,
        "train_service_date_range": [
            min(service_dates[i] for i in train_indices),
            max(service_dates[i] for i in train_indices),
        ],
        "validation_service_date_range": [
            min(service_dates[i] for i in validation_indices),
            max(service_dates[i] for i in validation_indices),
        ],
        "train_sample_rows": len(train_indices),
        "validation_sample_rows": len(validation_indices),
        "training_rows_available_before_sampling": len(train_candidates),
        "validation_rows_available_before_sampling": len(validation_candidates),
        "epochs": epochs,
        "learning_rate": effective_learning_rate,
        "model_type": model_type,
        "hidden_dims": list(hidden_dims) if model_type == "mlp" else [],
        "seed": seed,
        "metrics": per_horizon,
        "metric_space": (
            "provider-normalized target values; apply pinned normalization to convert to seconds"
        ),
        "validation_by_service_month": by_service_month,
        "test_period_used": False,
        "limitations": [
            "This is an observed-delay forecaster, not an OSRD candidate feasibility model.",
            "A bounded deterministic sample is used; rerun with larger limits before conclusions.",
            "The official 2025 holdout was not used for fitting or tuning.",
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
    parser.add_argument("--learning-rate", type=float)
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
            learning_rate=args.learning_rate,
            model_type=args.model,
            hidden_dims=tuple(args.hidden_dimensions),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
