import pytest

from osrd_ops_intelligence.models.linear import (
    fit_logistic,
    fit_ridge,
    predict_probability,
    predict_regression,
    sigmoid,
    standardize,
)


def test_sigmoid_is_bounded_for_extreme_inputs() -> None:
    assert sigmoid(1e6) < 1.0 + 1e-12
    assert sigmoid(-1e6) > 0.0
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_standardize_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError):
        standardize([1.0, 2.0], [0.0], [1.0])


def test_ridge_recovers_linear_relation() -> None:
    rows = [[float(x), float(x % 3)] for x in range(30)]
    targets = [2.0 * row[0] + 5.0 for row in rows]
    model = fit_ridge(rows, targets, epochs=2000, l2=0.0)
    assert predict_regression(model, [10.0, 1.0]) == pytest.approx(25.0, abs=0.5)


def test_logistic_separates_classes() -> None:
    rows = [[float(x)] for x in range(-10, 10)]
    labels = [int(row[0] > 0) for row in rows]
    model = fit_logistic(rows, labels, epochs=1500)
    assert predict_probability(model, [8.0]) > 0.9
    assert predict_probability(model, [-8.0]) < 0.1


def test_logistic_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        fit_logistic([[1.0], [2.0]], [1, 1])
