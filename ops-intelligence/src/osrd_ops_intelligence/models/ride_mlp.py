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
    validation_features: Any | None = None,
    validation_targets: Any | None = None,
    validation_mask: Any | None = None,
    patience: int = 5,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> tuple[Any, Any, list[Any], list[Any], dict[str, Any]]:
    """Fit a ReLU MLP with horizon-wise masked quantile loss and Adam.

    Returns train-fitted input means/scales, layer parameters, and fit metadata. Only
    valid target entries contribute to the loss; per-horizon normalization prevents
    sparse later horizons from being silently underweighted. When validation arrays
    are supplied, retain the best masked validation pinball-loss checkpoint and stop
    early. The output layer predicts all requested quantiles for each horizon.
    """
    import numpy as np

    if features.ndim != 2 or targets.ndim != 2 or mask.shape != targets.shape:
        raise ValueError("MLP expects 2D feature/target arrays and a matching target mask")
    if features.shape[0] != targets.shape[0] or not mask.any(axis=0).all():
        raise ValueError("MLP requires aligned rows and at least one valid label per horizon")
    if not np.isfinite(features).all() or not np.isfinite(targets[mask.astype(bool)]).all():
        raise ValueError("MLP training arrays contain non-finite usable values")
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("MLP hidden dimensions must be positive")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("MLP epochs and batch_size must be positive")
    if learning_rate <= 0 or l2 < 0:
        raise ValueError("MLP learning rate must be positive and L2 penalty cannot be negative")
    if (
        not quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
        or tuple(sorted(quantiles)) != quantiles
    ):
        raise ValueError("quantiles must be an increasing tuple of values between 0 and 1")
    validation_supplied = any(
        value is not None for value in (validation_features, validation_targets, validation_mask)
    )
    if validation_supplied and any(
        value is None for value in (validation_features, validation_targets, validation_mask)
    ):
        raise ValueError("All validation arrays must be supplied together")
    if validation_supplied:
        assert validation_features is not None
        assert validation_targets is not None
        assert validation_mask is not None
        if (
            validation_features.ndim != 2
            or validation_targets.ndim != 2
            or validation_mask.shape != validation_targets.shape
            or validation_features.shape[0] != validation_targets.shape[0]
            or validation_features.shape[1] != features.shape[1]
            or validation_targets.shape[1] != targets.shape[1]
            or not validation_mask.any(axis=0).all()
        ):
            raise ValueError("MLP validation arrays are inconsistent with the training arrays")
        if not np.isfinite(validation_features).all() or not np.isfinite(
            validation_targets[validation_mask.astype(bool)]
        ).all():
            raise ValueError("MLP validation arrays contain non-finite usable values")
    if patience <= 0:
        raise ValueError("patience must be positive")

    rng = np.random.default_rng(seed)
    means = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    dimensions = (features.shape[1], *hidden_dims, targets.shape[1] * len(quantiles))
    quantile_values = np.asarray(quantiles, dtype=np.float32).reshape(1, 1, -1)
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
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_weights = [value.copy() for value in weights]
    best_biases = [value.copy() for value in biases]
    validation_history: list[float] = []

    for epoch in range(1, epochs + 1):
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
            predictions = (activations[-1] @ weights[-1] + biases[-1]).reshape(
                len(indices), targets.shape[1], len(quantiles)
            )
            below_target = y[:, :, None] < predictions
            quantile_gradient = (
                (below_target.astype(np.float32) - quantile_values)
                * valid[:, :, None]
                / counts[None, :, None]
                / len(quantiles)
            )
            delta = quantile_gradient.reshape(len(indices), -1)
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

        if validation_supplied:
            validation_x = (validation_features - means) / scales
            activations = validation_x
            for layer in range(len(weights) - 1):
                activations = np.maximum(activations @ weights[layer] + biases[layer], 0.0)
            validation_predictions = (activations @ weights[-1] + biases[-1]).reshape(
                len(validation_features), targets.shape[1], len(quantiles)
            )
            valid = validation_mask.astype(bool)
            safe_targets = np.where(valid, validation_targets, 0.0)
            horizon_counts = np.maximum(valid.sum(axis=0), 1)
            residual = safe_targets[:, :, None] - validation_predictions
            pinball = np.maximum(
                quantile_values * residual, (quantile_values - 1.0) * residual
            )
            horizon_loss = (
                (pinball * valid[:, :, None]).sum(axis=(0, 2))
                / (horizon_counts * len(quantiles))
            )
            score = float(horizon_loss.mean())
            if not np.isfinite(score):
                raise ValueError("MLP validation loss became non-finite")
            validation_history.append(score)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                best_weights = [value.copy() for value in weights]
                best_biases = [value.copy() for value in biases]
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
    if validation_supplied:
        weights, biases = best_weights, best_biases
    fit_metadata = {
        "epochs_requested": epochs,
        "epochs_run": epoch,
        "best_epoch": best_epoch if validation_supplied else epoch,
        "best_validation_masked_pinball_loss": best_score if validation_supplied else None,
        "validation_masked_pinball_loss_history": validation_history,
        "early_stopping_patience": patience if validation_supplied else None,
    }
    return means, scales, weights, biases, fit_metadata


def predict_mlp(features: Any, model: dict[str, Any], *, sort_quantiles: bool = True) -> Any:
    import numpy as np

    values = np.asarray(features, dtype=np.float32)
    means = np.asarray(model["input_mean"], dtype=np.float32)
    scales = np.asarray(model["input_scale"], dtype=np.float32)
    weights = [np.asarray(layer) for layer in model["weights"]]
    biases = [np.asarray(layer) for layer in model["biases"]]
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("MLP features must be a non-empty 2D array")
    if means.ndim != 1 or scales.shape != means.shape or values.shape[1] != means.size:
        raise ValueError("MLP feature normalization does not match input width")
    if not np.isfinite(values).all() or not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("MLP features or normalization contain non-finite values")
    if (scales <= 0).any() or not weights or len(weights) != len(biases):
        raise ValueError("MLP normalization or layer parameters are invalid")
    previous_width = means.size
    for weight, bias in zip(weights, biases, strict=True):
        if weight.ndim != 2 or weight.shape[0] != previous_width or bias.shape != (weight.shape[1],):
            raise ValueError("MLP layer dimensions are invalid")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("MLP layer parameters contain non-finite values")
        previous_width = weight.shape[1]
    values = (values - means) / scales
    for layer in range(len(weights) - 1):
        values = np.maximum(values @ weights[layer] + biases[layer], 0.0)
    output = values @ weights[-1] + biases[-1]
    if "quantiles" in model:
        quantiles = model["quantiles"]
        if (
            not isinstance(quantiles, list)
            or not quantiles
            or any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in quantiles)
            or tuple(sorted(quantiles)) != tuple(quantiles)
            or len(set(quantiles)) != len(quantiles)
            or output.shape[1] % len(quantiles)
        ):
            raise ValueError("MLP quantile levels or output dimensions are invalid")
        output = output.reshape(len(values), -1, len(quantiles))
        return np.sort(output, axis=-1) if sort_quantiles else output
    return output
