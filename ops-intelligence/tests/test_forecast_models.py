import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from osrd_ops_intelligence.models.calibration import (  # noqa: E402
    calibrate_artifact,
    calibrate_interval_expansion,
)
from osrd_ops_intelligence.models.forecast import load_model, predict  # noqa: E402
from osrd_ops_intelligence.models.ride_mlp import fit_masked_mlp, predict_mlp  # noqa: E402

HORIZONS = 15
QUANTILES = (0.1, 0.5, 0.9)


def _data(rows: int = 400, seed: int = 0) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, 4)).astype(np.float32)
    base = x[:, :1] * np.linspace(0.5, 1.5, HORIZONS, dtype=np.float32)
    y = (base + 0.1 * rng.standard_normal((rows, HORIZONS))).astype(np.float32)
    mask = rng.random((rows, HORIZONS)) > 0.2
    mask[:2] = True
    return x, y, mask


def _artifact(**overrides: Any) -> dict[str, Any]:
    x, y, mask = _data()
    mean, scale, weights, biases, meta = fit_masked_mlp(
        x, y, mask, hidden_dims=(16,), epochs=5, batch_size=64, quantiles=QUANTILES
    )
    model: dict[str, Any] = {
        "schema": "osrd-ops-ride-delay-forecast/v2",
        "model_type": "mlp",
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases],
        "quantiles": list(QUANTILES),
        "fit_metadata": meta,
        "safety": {"osrd_validation_required_for_plans": True},
    }
    model.update(overrides)
    return _sign(model)


def _sign(model: dict[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in model.items() if k != "model_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return {**unsigned, "model_sha256": hashlib.sha256(canonical).hexdigest()}


def _write(path: Path, model: dict[str, Any]) -> Path:
    path.write_text(json.dumps(model), encoding="utf-8")
    return path


# --- masked quantile MLP -------------------------------------------------------------


def test_mlp_training_is_deterministic_and_learns() -> None:
    x, y, mask = _data()
    kwargs: dict[str, Any] = {
        "hidden_dims": (16,),
        "epochs": 15,
        "batch_size": 64,
        "validation_features": x,
        "validation_targets": y,
        "validation_mask": mask,
        "quantiles": QUANTILES,
    }
    first = fit_masked_mlp(x, y, mask, **kwargs)
    second = fit_masked_mlp(x, y, mask, **kwargs)
    assert all(np.array_equal(a, b) for a, b in zip(first[2], second[2], strict=True))
    history = first[4]["validation_masked_pinball_loss_history"]
    assert history[-1] < history[0]
    assert first[4]["best_epoch"] >= 1


def test_mlp_ignores_masked_targets() -> None:
    x, y, mask = _data()
    corrupted = np.where(mask, y, 1e6).astype(np.float32)  # masked entries are garbage
    kwargs: dict[str, Any] = {"hidden_dims": (8,), "epochs": 3, "quantiles": QUANTILES}
    clean = fit_masked_mlp(x, y, mask, **kwargs)
    noisy = fit_masked_mlp(x, corrupted, mask, **kwargs)
    assert all(np.allclose(a, b) for a, b in zip(clean[2], noisy[2], strict=True))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x, y, m: (x[:, 0], y, m),  # 1D features
        lambda x, y, m: (x, y, m[:, :-1]),  # mask shape mismatch
        lambda x, y, m: (x, y, np.zeros_like(m)),  # no valid labels
        lambda x, y, m: (np.where(x > 0, np.nan, x), y, m),  # non-finite features
    ],
)
def test_mlp_rejects_bad_inputs(mutate: Any) -> None:
    x, y, mask = _data()
    with pytest.raises(ValueError):
        fit_masked_mlp(*mutate(x, y, mask), epochs=1, hidden_dims=(4,))


def test_mlp_requires_all_validation_arrays_together() -> None:
    x, y, mask = _data()
    with pytest.raises(ValueError, match="validation"):
        fit_masked_mlp(x, y, mask, epochs=1, hidden_dims=(4,), validation_features=x)


def test_predict_mlp_sorts_quantiles_and_validates_width() -> None:
    model = _artifact()
    x, _, _ = _data(rows=20)
    out = predict_mlp(x, model)
    assert out.shape == (20, HORIZONS, 3)
    assert (np.diff(out, axis=-1) >= 0).all()
    with pytest.raises(ValueError):
        predict_mlp(x[:, :2], model)


# --- split-conformal calibration -----------------------------------------------------


def test_expansion_matches_conformal_order_statistic() -> None:
    # One horizon, 9 rows, lower=-1, upper=1; observed 1+k for k=1..9 -> scores k.
    n = 9
    preds = np.tile(np.array([-1.0, 0.0, 1.0]), (n, 1, 1)).reshape(n, 1, 3)
    targets = (2.0 + np.arange(n)).reshape(n, 1)
    mask = np.ones((n, 1), dtype=bool)
    # ceil((9+1)*0.8) = 8 -> 8th smallest score = 8
    assert calibrate_interval_expansion(preds, targets, mask, coverage=0.8) == [8.0]


def test_expansion_is_zero_when_already_covered() -> None:
    preds = np.tile(np.array([-5.0, 0.0, 5.0]), (10, 1, 1)).reshape(10, 1, 3)
    targets = np.zeros((10, 1))
    assert calibrate_interval_expansion(preds, targets, np.ones((10, 1), bool), coverage=0.8) == [
        0.0
    ]


def test_expansion_needs_enough_labels_for_coverage() -> None:
    preds = np.zeros((3, 1, 3))
    with pytest.raises(ValueError, match="more calibration labels"):
        calibrate_interval_expansion(preds, np.zeros((3, 1)), np.ones((3, 1), bool), coverage=0.9)


def test_expansion_ignores_masked_targets() -> None:
    n = 10
    preds = np.zeros((n, 1, 3))
    targets = np.full((n, 1), 100.0)
    mask = np.zeros((n, 1), dtype=bool)
    mask[:9] = True
    targets[9] = 0.0  # masked row must not influence the result
    with_mask = calibrate_interval_expansion(preds, targets, mask, coverage=0.8)
    assert with_mask == [100.0]


def test_calibrate_artifact_adds_offsets_and_resigns() -> None:
    model = _artifact()
    x, y, mask = _data(rows=300, seed=1)
    preds = predict_mlp(x, model)
    calibrated = calibrate_artifact(model, preds, y, mask, coverage=0.8)
    cal = calibrated["interval_calibration"]
    assert len(cal["expansion_by_horizon"]) == HORIZONS
    assert all(value >= 0 for value in cal["expansion_by_horizon"])
    unsigned = {k: v for k, v in calibrated.items() if k != "model_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert calibrated["model_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert calibrated["model_sha256"] != model["model_sha256"]
    with pytest.raises(ValueError, match="uncalibrated"):
        calibrate_artifact(calibrated, preds, y, mask, coverage=0.8)


def test_calibrate_artifact_rejects_coverage_mismatch() -> None:
    model = _artifact()
    x, y, mask = _data(rows=300, seed=1)
    with pytest.raises(ValueError, match="outer quantile"):
        calibrate_artifact(model, predict_mlp(x, model), y, mask, coverage=0.5)


def test_calibration_achieves_coverage_on_exchangeable_data() -> None:
    model = _artifact()
    x, y, mask = _data(rows=1500, seed=2)
    cal_x, cal_y, cal_m = x[:700], y[:700], mask[:700]
    calibrated = calibrate_artifact(model, predict_mlp(cal_x, model), cal_y, cal_m, coverage=0.8)
    out = predict(x[700:], calibrated)
    covered = (y[700:] >= out[:, :, 0]) & (y[700:] <= out[:, :, 2])
    valid = mask[700:]
    assert covered[valid].mean() >= 0.75  # nominal 0.8 with finite-sample slack


# --- artifact loading and prediction -------------------------------------------------


def test_load_model_accepts_valid_artifact(tmp_path: Path) -> None:
    model = _artifact()
    assert load_model(_write(tmp_path / "m.json", model))["model_sha256"] == model["model_sha256"]


def test_load_model_rejects_tampered_artifact(tmp_path: Path) -> None:
    model = _artifact()
    tampered = copy.deepcopy(model)
    tampered["weights"][0][0][0] += 1.0
    with pytest.raises(ValueError, match="hash"):
        load_model(_write(tmp_path / "m.json", tampered))


def test_load_model_rejects_missing_hash_schema_and_safety_flag(tmp_path: Path) -> None:
    model = _artifact()
    no_hash = {k: v for k, v in model.items() if k != "model_sha256"}
    with pytest.raises(ValueError, match="model_sha256"):
        load_model(_write(tmp_path / "a.json", no_hash))
    with pytest.raises(ValueError, match="schema"):
        load_model(_write(tmp_path / "b.json", _artifact(schema="other/v1")))
    with pytest.raises(ValueError, match="safety"):
        load_model(_write(tmp_path / "c.json", _artifact(safety={})))


def test_predict_applies_calibration_to_outer_quantiles_only() -> None:
    model = _artifact()
    x, _, _ = _data(rows=10)
    base = predict(x, model)
    expansion = [0.25] * HORIZONS
    calibrated = _sign(
        {
            **model,
            "interval_calibration": {"nominal_coverage": 0.8, "expansion_by_horizon": expansion},
        }
    )
    out = predict(x, calibrated)
    assert np.allclose(out[:, :, 0], base[:, :, 0] - 0.25)
    assert np.allclose(out[:, :, 2], base[:, :, 2] + 0.25)
    assert np.allclose(out[:, :, 1], base[:, :, 1])


def test_predict_rejects_incompatible_calibration_and_bad_features() -> None:
    model = _artifact()
    x, _, _ = _data(rows=5)
    bad = {
        **model,
        "interval_calibration": {"nominal_coverage": 0.5, "expansion_by_horizon": [0.0] * 15},
    }
    with pytest.raises(ValueError, match="incompatible"):
        predict(x, bad)
    negative = {
        **model,
        "interval_calibration": {"nominal_coverage": 0.8, "expansion_by_horizon": [-1.0] * 15},
    }
    with pytest.raises(ValueError, match="incompatible"):
        predict(x, negative)
    with pytest.raises(ValueError, match="feature columns"):
        predict(x[:, :2], model)
    nan = x.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        predict(nan, model)
