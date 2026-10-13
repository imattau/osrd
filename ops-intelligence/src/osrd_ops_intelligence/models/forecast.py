"""Load and run an offline RIDE delay forecast artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.ride_mlp import predict_mlp

SUPPORTED_MODEL_SCHEMAS = {
    "osrd-ops-ride-delay-forecast/v1",
    "osrd-ops-ride-delay-forecast/v2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> dict[str, Any]:
    """Load a model artifact only after verifying its canonical content hash."""
    model = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(model, dict) or model.get("schema") not in SUPPORTED_MODEL_SCHEMAS:
        raise ValueError("Unsupported delay forecast model schema")
    expected_hash = model.get("model_sha256")
    if not isinstance(expected_hash, str):
        raise ValueError("Model artifact is missing model_sha256")
    unsigned = {key: value for key, value in model.items() if key != "model_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != expected_hash:
        raise ValueError("Model artifact hash check failed")
    safety = model.get("safety")
    if not isinstance(safety, dict) or safety.get("osrd_validation_required_for_plans") is not True:
        raise ValueError("Model artifact is missing the required OSRD validation safety flag")
    return model


def predict(features: Any, model: dict[str, Any], *, batch_size: int = 4096) -> Any:
    """Predict normalized delay targets for a batch of provider-schema feature rows."""
    import numpy as np

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Features must be a non-empty 2D array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model.get("schema") not in SUPPORTED_MODEL_SCHEMAS:
        raise ValueError("Unsupported delay forecast model schema")

    model_type = model.get("model_type", "linear")
    if model_type not in {"linear", "mlp"}:
        raise ValueError(f"Unsupported forecast model type: {model_type}")
    quantile_levels = model.get("quantiles") if model_type == "mlp" else None
    if quantile_levels is not None and (
        not isinstance(quantile_levels, list) or not quantile_levels
    ):
        raise ValueError("MLP quantile levels must be a non-empty list")
    predictions = np.empty(
        (values.shape[0], 15, len(quantile_levels))
        if quantile_levels is not None
        else (values.shape[0], 15),
        dtype=np.float32,
    )
    weights: Any = None
    bias: Any = None
    if model_type == "mlp":
        expected_width = len(model.get("input_mean", []))
        if values.shape[1] != expected_width:
            raise ValueError(f"Expected {expected_width} feature columns, got {values.shape[1]}")
        if quantile_levels is not None and (
            any(not isinstance(level, (int, float)) for level in quantile_levels)
            or any(not 0.0 < float(level) < 1.0 for level in quantile_levels)
            or tuple(sorted(quantile_levels)) != tuple(quantile_levels)
            or len(set(quantile_levels)) != len(quantile_levels)
        ):
            raise ValueError("MLP quantile levels must be unique, ordered probabilities")
    else:
        weights = np.asarray(model.get("weights"), dtype=np.float32)
        bias = np.asarray(model.get("bias"), dtype=np.float32)
        if weights.ndim != 2 or bias.shape != (weights.shape[1],):
            raise ValueError("Linear model parameter dimensions are invalid")
        if values.shape[1] != weights.shape[0]:
            raise ValueError(f"Expected {weights.shape[0]} feature columns, got {values.shape[1]}")
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        batch = values[start:end]
        if not np.isfinite(batch).all():
            raise ValueError(f"Features contain non-finite values near row {start}")
        if model_type == "mlp":
            batch_predictions = predict_mlp(batch, model)
            predictions[start:end] = batch_predictions
        else:
            predictions[start:end] = batch @ weights + bias
    calibration = model.get("interval_calibration")
    if calibration is not None:
        if quantile_levels is None or not isinstance(calibration, dict):
            raise ValueError("Interval calibration requires an MLP quantile model")
        expansion = np.asarray(calibration.get("expansion_by_horizon"), dtype=np.float32)
        expected_coverage = float(quantile_levels[-1]) - float(quantile_levels[0])
        if (
            expansion.shape != (15,)
            or not np.isfinite(expansion).all()
            or (expansion < 0).any()
            or abs(float(calibration.get("nominal_coverage", -1)) - expected_coverage) > 1e-6
        ):
            raise ValueError("Interval calibration metadata is incompatible with model quantiles")
        predictions[:, :, 0] -= expansion
        predictions[:, :, -1] += expansion
    expected_ndim = 3 if quantile_levels is not None else 2
    if predictions.ndim != expected_ndim or predictions.shape[:2] != (values.shape[0], 15):
        raise ValueError("Model must produce predictions for 15 forecast horizons")
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="trained model.json artifact")
    parser.add_argument("--features", type=Path, required=True, help="2D float .npy feature matrix")
    parser.add_argument("--feature-schema", type=Path, required=True, help="RIDE scheme.yaml")
    parser.add_argument("--normalization", type=Path, required=True, help="RIDE normalization.yaml")
    parser.add_argument("--output", type=Path, required=True, help="destination predictions .npy")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    try:
        model = load_model(args.model)
        if _sha256(args.feature_schema) != model.get("feature_schema_sha256"):
            raise ValueError("Feature schema does not match the model artifact")
        if _sha256(args.normalization) != model.get("normalization_sha256"):
            raise ValueError("Feature normalization does not match the model artifact")
        import numpy as np

        features = np.load(args.features, mmap_mode="r", allow_pickle=False)
        predictions = predict(features, model, batch_size=args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, predictions, allow_pickle=False)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "model_sha256": model["model_sha256"],
                "model_type": model.get("model_type", "linear"),
                "rows": int(predictions.shape[0]),
                "horizons": 15,
                "quantiles": model.get("quantiles"),
                "interval_calibrated": model.get("interval_calibration") is not None,
                "prediction_space": (
                    "synthetic target units"
                    if model.get("data_provenance", {}).get("synthetic") is True
                    else "RIDE provider-normalized delay target units"
                ),
                "output": str(args.output),
                "operationally_approved": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
