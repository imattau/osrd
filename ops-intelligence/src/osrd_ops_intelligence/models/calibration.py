"""Split-conformal expansion for multi-horizon quantile forecast intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.models.forecast import load_model


def calibrate_interval_expansion(
    predictions: Any,
    targets: Any,
    mask: Any,
    *,
    coverage: float,
) -> list[float]:
    """Return horizon-specific nonnegative split-conformal interval expansions."""
    import numpy as np

    values = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] < 2:
        raise ValueError("Quantile predictions must have shape [rows, horizons, quantiles]")
    if actual.shape != values.shape[:2] or valid.shape != actual.shape:
        raise ValueError("Calibration targets and mask must match prediction rows and horizons")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    if not np.isfinite(values).all() or not np.isfinite(actual[valid]).all():
        raise ValueError("Calibration arrays contain non-finite values")

    expansions: list[float] = []
    for horizon in range(values.shape[1]):
        horizon_mask = valid[:, horizon]
        count = int(horizon_mask.sum())
        order = math.ceil((count + 1) * coverage)
        if count == 0 or order > count:
            raise ValueError(
                f"Horizon {horizon + 1} needs more calibration labels for coverage {coverage}"
            )
        lower = values[horizon_mask, horizon, 0]
        upper = values[horizon_mask, horizon, -1]
        observed = actual[horizon_mask, horizon]
        scores = np.maximum(np.maximum(lower - observed, observed - upper), 0.0)
        expansions.append(float(np.sort(scores)[order - 1]))
    return expansions


def _artifact_hash(model: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in model.items() if key != "model_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def calibrate_artifact(
    model: dict[str, Any], predictions: Any, targets: Any, mask: Any, *, coverage: float
) -> dict[str, Any]:
    """Attach calibration offsets and refresh the artifact digest."""
    import numpy as np

    if model.get("schema") not in {
        "osrd-ops-ride-delay-forecast/v1",
        "osrd-ops-ride-delay-forecast/v2",
    }:
        raise ValueError("Unsupported forecast model schema for calibration")
    if model.get("model_type") != "mlp" or "quantiles" not in model:
        raise ValueError("Interval calibration requires an MLP quantile model")
    if "interval_calibration" in model:
        raise ValueError("Start interval calibration from an uncalibrated model artifact")
    levels = model["quantiles"]
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("Model artifact has invalid quantile levels")
    if tuple(sorted(levels)) != tuple(levels):
        raise ValueError("Model quantile levels must be ordered")
    prediction_shape = np.asarray(predictions).shape
    if len(prediction_shape) != 3 or prediction_shape[-1] != len(levels):
        raise ValueError("Prediction quantile dimension does not match model artifact")
    if abs((levels[-1] - levels[0]) - coverage) > 1e-6:
        raise ValueError("Requested coverage must match the model's outer quantile levels")
    offsets = calibrate_interval_expansion(
        predictions, targets, mask, coverage=coverage
    )
    actual = np.asarray(targets)
    valid = np.asarray(mask, dtype=bool)
    data_digest = hashlib.sha256(
        np.asarray(actual[valid], dtype="<f4").tobytes()
        + np.asarray(valid, dtype=np.uint8).tobytes()
    ).hexdigest()
    calibrated = dict(model)
    calibrated["interval_calibration"] = {
        "method": "split-conformal-nonnegative-expansion/v1",
        "nominal_coverage": coverage,
        "outer_quantiles": [levels[0], levels[-1]],
        "expansion_by_horizon": offsets,
        "calibration_row_count": int(actual.shape[0]),
        "valid_count_by_horizon": [int(value) for value in valid.sum(axis=0)],
        "calibration_targets_and_mask_sha256": data_digest,
        "warning": (
            "Use calibration data independent of fitting and evaluation. Finite-sample "
            "marginal coverage assumes exchangeability; temporal drift can invalidate it."
        ),
    }
    calibrated["model_sha256"] = _artifact_hash(calibrated)
    return calibrated


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="[rows, 15, quantiles] .npy")
    parser.add_argument("--targets", type=Path, required=True, help="[rows, 15] target .npy")
    parser.add_argument("--mask", type=Path, required=True, help="[rows, 15] validity mask .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.8)
    args = parser.parse_args()
    try:
        model = load_model(args.model)
        predictions = np.load(args.predictions, allow_pickle=False)
        targets = np.load(args.targets, allow_pickle=False)
        mask = np.load(args.mask, allow_pickle=False)
        calibrated = calibrate_artifact(
            model, predictions, targets, mask, coverage=args.coverage
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(calibrated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "calibrated_model_sha256": calibrated["model_sha256"],
                "nominal_coverage": args.coverage,
                "expansion_by_horizon": calibrated["interval_calibration"][
                    "expansion_by_horizon"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
