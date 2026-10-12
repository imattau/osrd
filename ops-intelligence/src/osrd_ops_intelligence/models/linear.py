"""Small regularized linear models with explicit, serializable parameters."""

from __future__ import annotations

import math
from typing import Any


def fit_standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    if not rows:
        raise ValueError("Cannot fit a model without feature rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Feature rows have inconsistent widths")
    means = [sum(row[column] for row in rows) / len(rows) for column in range(width)]
    scales: list[float] = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in rows) / len(rows)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def standardize(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    if len(row) != len(means) or len(means) != len(scales):
        raise ValueError("Feature width does not match the fitted model")
    return [(value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)]


def sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def fit_logistic(
    rows: list[list[float]],
    labels: list[int],
    *,
    epochs: int = 3000,
    learning_rate: float = 0.04,
    l2: float = 0.02,
) -> dict[str, Any]:
    if not rows or len(rows) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Logistic fitting requires paired rows and both target classes")
    means, scales = fit_standardizer(rows)
    normalized = [standardize(row, means, scales) for row in rows]
    weights = [0.0] * len(means)
    bias = 0.0
    count = len(labels)
    for _ in range(epochs):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, label in zip(normalized, labels, strict=True):
            probability = sigmoid(bias + sum(w * x for w, x in zip(weights, row, strict=True)))
            error = probability - label
            bias_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        bias -= learning_rate * bias_gradient / count
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradients[index] / count + l2 * weights[index])
    return {"weights": weights, "bias": bias, "means": means, "scales": scales}


def fit_ridge(
    rows: list[list[float]],
    targets: list[float],
    *,
    epochs: int = 3000,
    learning_rate: float = 0.03,
    l2: float = 0.05,
) -> dict[str, Any]:
    if not rows or len(rows) != len(targets):
        raise ValueError("Ridge fitting requires paired feature and target rows")
    means, scales = fit_standardizer(rows)
    normalized = [standardize(row, means, scales) for row in rows]
    target_mean = sum(targets) / len(targets)
    target_scale = (
        math.sqrt(sum((target - target_mean) ** 2 for target in targets) / len(targets)) or 1.0
    )
    normalized_targets = [(target - target_mean) / target_scale for target in targets]
    weights = [0.0] * len(means)
    bias = 0.0
    count = len(targets)
    for _ in range(epochs):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, target in zip(normalized, normalized_targets, strict=True):
            error = bias + sum(w * x for w, x in zip(weights, row, strict=True)) - target
            bias_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        bias -= learning_rate * bias_gradient / count
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradients[index] / count + l2 * weights[index])
    return {
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def predict_probability(model: dict[str, Any], row: list[float]) -> float:
    values = standardize(row, model["means"], model["scales"])
    return sigmoid(
        model["bias"] + sum(w * x for w, x in zip(model["weights"], values, strict=True))
    )


def predict_regression(model: dict[str, Any], row: list[float]) -> float:
    values = standardize(row, model["means"], model["scales"])
    normalized = model["bias"] + sum(w * x for w, x in zip(model["weights"], values, strict=True))
    return model["target_mean"] + model["target_scale"] * normalized
