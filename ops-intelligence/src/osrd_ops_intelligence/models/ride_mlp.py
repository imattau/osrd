"""Small masked multi-output MLP used for the RIDE tabular forecast experiment."""

from __future__ import annotations

from typing import Any


def fit_masked_mlp(
    features: Any,
    targets: Any,
    mask: Any,
    *,
    hidden_dims: tuple[int, ...] = (256, 128),
    epochs: int = 40,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    l2: float = 0.0001,
    seed: int = 42,
) -> tuple[Any, Any, list[Any], list[Any]]:
    """Fit a ReLU MLP with horizon-wise masked MSE and Adam.

    Returns train-fitted input means/scales and layer parameters. Only valid target
    entries contribute to the loss; per-horizon normalization prevents sparse later
    horizons from being silently underweighted.
    """
    import numpy as np

    if features.ndim != 2 or targets.ndim != 2 or mask.shape != targets.shape:
        raise ValueError("MLP expects 2D feature/target arrays and a matching target mask")
    if features.shape[0] != targets.shape[0] or not mask.any(axis=0).all():
        raise ValueError("MLP requires aligned rows and at least one valid label per horizon")
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("MLP hidden dimensions must be positive")

    rng = np.random.default_rng(seed)
    means = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    dimensions = (features.shape[1], *hidden_dims, targets.shape[1])
    weights = [
        (rng.standard_normal((left, right)) * np.sqrt(2.0 / left)).astype(np.float32)
        for left, right in zip(dimensions[:-1], dimensions[1:], strict=True)
    ]
    biases = [np.zeros(width, dtype=np.float32) for width in dimensions[1:]]
    first_w = [np.zeros_like(value) for value in weights]
    second_w = [np.zeros_like(value) for value in weights]
    first_b = [np.zeros_like(value) for value in biases]
    second_b = [np.zeros_like(value) for value in biases]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    for _epoch in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = (features[indices] - means) / scales
            valid = mask[indices].astype(np.float32)
            y = np.where(valid.astype(bool), targets[indices], 0.0)
            counts = np.maximum(valid.sum(axis=0), 1.0)
            activations = [x]
            preactivations = []
            for layer in range(len(weights) - 1):
                z = activations[-1] @ weights[layer] + biases[layer]
                preactivations.append(z)
                activations.append(np.maximum(z, 0.0))
            predictions = activations[-1] @ weights[-1] + biases[-1]
            delta = (2.0 * (predictions - y) * valid) / counts
            grad_w = [np.empty_like(value) for value in weights]
            grad_b = [np.empty_like(value) for value in biases]
            grad_w[-1] = activations[-1].T @ delta + l2 * weights[-1]
            grad_b[-1] = delta.sum(axis=0)
            for layer in range(len(weights) - 2, -1, -1):
                delta = (delta @ weights[layer + 1].T) * (preactivations[layer] > 0)
                grad_w[layer] = activations[layer].T @ delta + l2 * weights[layer]
                grad_b[layer] = delta.sum(axis=0)

            step += 1
            correction1, correction2 = 1 - beta1**step, 1 - beta2**step
            for layer in range(len(weights)):
                first_w[layer] = beta1 * first_w[layer] + (1 - beta1) * grad_w[layer]
                second_w[layer] = beta2 * second_w[layer] + (1 - beta2) * grad_w[layer] ** 2
                first_b[layer] = beta1 * first_b[layer] + (1 - beta1) * grad_b[layer]
                second_b[layer] = beta2 * second_b[layer] + (1 - beta2) * grad_b[layer] ** 2
                weights[layer] -= learning_rate * (first_w[layer] / correction1) / (
                    np.sqrt(second_w[layer] / correction2) + epsilon
                )
                biases[layer] -= learning_rate * (first_b[layer] / correction1) / (
                    np.sqrt(second_b[layer] / correction2) + epsilon
                )
    return means, scales, weights, biases


def predict_mlp(features: Any, model: dict[str, Any]) -> Any:
    import numpy as np

    values = (features - np.asarray(model["input_mean"])) / np.asarray(model["input_scale"])
    weights = [np.asarray(layer) for layer in model["weights"]]
    biases = [np.asarray(layer) for layer in model["biases"]]
    for layer in range(len(weights) - 1):
        values = np.maximum(values @ weights[layer] + biases[layer], 0.0)
    return values @ weights[-1] + biases[-1]
