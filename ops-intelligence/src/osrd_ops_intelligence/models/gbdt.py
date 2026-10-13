"""Gradient-boosted quantile baseline: one histogram GBDT per horizon and quantile."""

# scikit-learn ships no complete type stubs; keep strict checks on the surrounding contract.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any


@dataclass
class GbdtQuantileModel:
    """Fitted estimators indexed [horizon][quantile] with a chosen iteration count each."""

    quantiles: tuple[float, ...]
    estimators: list[list[Any]]
    best_iterations: list[list[int]]


def _pinball(actual: Any, predicted: Any, level: float) -> float:
    import numpy as np

    residual = actual - predicted
    return float(np.mean(np.maximum(level * residual, (level - 1.0) * residual)))


def _predict_at(estimator: Any, features: Any, iterations: int) -> Any:
    """Prediction using only the first `iterations` boosting stages."""
    last = deque(islice(estimator.staged_predict(features), iterations), maxlen=1)
    return last[0]


def fit_gbdt_quantiles(
    features: Any,
    targets: Any,
    mask: Any,
    *,
    validation_features: Any,
    validation_targets: Any,
    validation_mask: Any,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    min_samples_leaf: int = 40,
    l2_regularization: float = 1.0,
    seed: int = 42,
) -> GbdtQuantileModel:
    """Fit masked per-horizon quantile GBDTs.

    Only valid target entries are used for each horizon. The number of boosting stages is
    chosen on the supplied chronological validation partition, never an internal random
    split, so temporal leakage does not leak into model selection.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    if not quantiles or tuple(sorted(quantiles)) != quantiles:
        raise ValueError("quantiles must be a non-empty increasing tuple")
    if any(not 0.0 < level < 1.0 for level in quantiles):
        raise ValueError("quantile levels must be between 0 and 1")
    if features.ndim != 2 or targets.ndim != 2 or mask.shape != targets.shape:
        raise ValueError("GBDT expects 2D features/targets and a matching mask")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("GBDT features and targets must have the same row count")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    estimators: list[list[Any]] = []
    best_iterations: list[list[int]] = []
    for horizon in range(targets.shape[1]):
        rows = mask[:, horizon].astype(bool)
        validation_rows = validation_mask[:, horizon].astype(bool)
        if not rows.any() or not validation_rows.any():
            raise ValueError(f"Horizon {horizon + 1} has no valid train or validation labels")
        horizon_estimators: list[Any] = []
        horizon_iterations: list[int] = []
        for level in quantiles:
            estimator = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=level,
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                early_stopping=False,
                random_state=seed,
            )
            estimator.fit(features[rows], targets[rows, horizon])
            scores = [
                _pinball(validation_targets[validation_rows, horizon], stage, level)
                for stage in estimator.staged_predict(validation_features[validation_rows])
            ]
            horizon_iterations.append(int(np.argmin(scores)) + 1)
            horizon_estimators.append(estimator)
        estimators.append(horizon_estimators)
        best_iterations.append(horizon_iterations)
    return GbdtQuantileModel(quantiles, estimators, best_iterations)


def predict_gbdt(model: GbdtQuantileModel, features: Any) -> Any:
    """Return quantile predictions shaped [rows, horizons, quantiles], sorted per row."""
    import numpy as np

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("GBDT features must be a non-empty 2D array")
    output = np.empty(
        (features.shape[0], len(model.estimators), len(model.quantiles)), dtype=np.float32
    )
    for horizon, horizon_estimators in enumerate(model.estimators):
        for index, estimator in enumerate(horizon_estimators):
            output[:, horizon, index] = _predict_at(
                estimator, features, model.best_iterations[horizon][index]
            )
    return np.sort(output, axis=-1)
