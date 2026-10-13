from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from osrd_ops_intelligence.models.gbdt import fit_gbdt_quantiles, predict_gbdt  # noqa: E402
from osrd_ops_intelligence.training.ride_gbdt_baseline import (  # noqa: E402
    mlp_mean_pinball,
    quantile_report,
)

HORIZONS = 3
LEVELS = (0.1, 0.5, 0.9)


def _data(rows: int, seed: int) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, 3)).astype(np.float32)
    signal = np.maximum(x[:, :1], 0) * 2 + x[:, 1:2] * x[:, 2:3]  # nonlinear
    y = (signal + 0.3 * rng.standard_normal((rows, HORIZONS))).astype(np.float32)
    mask = rng.random((rows, HORIZONS)) > 0.15
    return x, y, mask


def _fit(seed: int = 0, **kwargs: Any) -> Any:
    x, y, mask = _data(1500, seed)
    vx, vy, vm = _data(500, seed + 1)
    return fit_gbdt_quantiles(
        x,
        y,
        mask,
        validation_features=vx,
        validation_targets=vy,
        validation_mask=vm,
        quantiles=LEVELS,
        max_iter=60,
        min_samples_leaf=20,
        **kwargs,
    )


def test_gbdt_predictions_are_sorted_and_cover_roughly_nominal() -> None:
    model = _fit()
    x, y, mask = _data(1500, 7)
    out = predict_gbdt(model, x)
    assert out.shape == (1500, HORIZONS, 3)
    assert (np.diff(out, axis=-1) >= 0).all()
    report = quantile_report(out, y, mask, LEVELS)
    assert 0.65 <= report["mean_outer_interval_coverage"] <= 0.95


def test_gbdt_is_deterministic() -> None:
    x, _, _ = _data(50, 3)
    assert np.array_equal(predict_gbdt(_fit(), x), predict_gbdt(_fit(), x))


def test_gbdt_beats_constant_quantiles_on_nonlinear_signal() -> None:
    model = _fit()
    x, y, mask = _data(1500, 9)
    learned = quantile_report(predict_gbdt(model, x), y, mask, LEVELS)
    constant = np.empty((len(x), HORIZONS, 3), dtype=np.float32)
    for h in range(HORIZONS):
        constant[:, h, :] = np.quantile(y[mask[:, h], h], LEVELS)
    baseline = quantile_report(constant, y, mask, LEVELS)
    assert learned["mean_pinball_loss_normalized"] < 0.8 * baseline["mean_pinball_loss_normalized"]


def test_gbdt_selects_iterations_on_validation_partition() -> None:
    model = _fit()
    assert all(1 <= n <= 60 for row in model.best_iterations for n in row)


def test_gbdt_ignores_masked_targets() -> None:
    x, y, mask = _data(600, 4)
    vx, vy, vm = _data(200, 5)
    kwargs: dict[str, Any] = {
        "validation_features": vx,
        "validation_targets": vy,
        "validation_mask": vm,
        "quantiles": LEVELS,
        "max_iter": 20,
        "min_samples_leaf": 20,
    }
    clean = fit_gbdt_quantiles(x, y, mask, **kwargs)
    garbage = fit_gbdt_quantiles(x, np.where(mask, y, 1e6).astype(np.float32), mask, **kwargs)
    assert np.allclose(predict_gbdt(clean, vx), predict_gbdt(garbage, vx))


def test_gbdt_rejects_bad_arguments() -> None:
    x, y, mask = _data(100, 0)
    kwargs: dict[str, Any] = {
        "validation_features": x,
        "validation_targets": y,
        "validation_mask": mask,
    }
    with pytest.raises(ValueError):
        fit_gbdt_quantiles(x, y, mask, quantiles=(0.9, 0.1), **kwargs)
    with pytest.raises(ValueError):
        fit_gbdt_quantiles(x, y, np.zeros_like(mask), **kwargs)


def test_quantile_report_values() -> None:
    pred = np.tile(np.array([-1.0, 0.0, 1.0], dtype=np.float32), (4, 1, 1))
    y = np.array([[0.0], [0.5], [2.0], [-3.0]], dtype=np.float32)
    mask = np.array([[True], [True], [True], [False]])
    report = quantile_report(pred, y, mask, LEVELS)["by_horizon"][0]
    assert report["valid_count"] == 3
    assert report["outer_interval_coverage"] == pytest.approx(2 / 3)
    assert report["mean_outer_interval_width_normalized"] == pytest.approx(2.0)


def test_mlp_mean_pinball_requires_mlp_report() -> None:
    with pytest.raises(ValueError):
        mlp_mean_pinball({"model_type": "linear"})
    report = {
        "model_type": "mlp",
        "quantile_metrics": [{"pinball_loss_normalized": [0.1, 0.2, 0.3]}],
    }
    assert mlp_mean_pinball(report) == pytest.approx(0.2)
