import pytest

from osrd_ops_intelligence.training.plan_ranker_data import _scenario_split
from osrd_ops_intelligence.training.train_plan_ranker import _metrics, _top_k_rates


def _row(group: str, cid: str, score: float, valid: bool) -> dict[str, object]:
    return {
        "scenario_group": group,
        "candidate_id": cid,
        "features": {"objective_score": score},
        "targets": {"osrd_validated": valid},
    }


def test_metrics_precision_recall_brier() -> None:
    m = _metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1])
    assert m["precision_at_0_5"] == pytest.approx(0.5)
    assert m["recall_at_0_5"] == pytest.approx(0.5)
    assert m["brier_score"] == pytest.approx((0.01 + 0.64 + 0.64 + 0.01) / 4)


def test_metrics_handle_empty_predictions() -> None:
    assert _metrics([], [])["brier_score"] == 0.0


def test_top_k_compares_learned_objective_and_oracle() -> None:
    rows = [_row("s", "a", 1.0, False), _row("s", "b", 2.0, True), _row("s", "c", 3.0, False)]
    report = _top_k_rates(rows, [0.1, 0.9, 0.2], budgets=(1,))["1"]
    assert report["learned_mean_valid_rate"] == 1.0  # ranker picks the valid candidate
    assert report["objective_order_mean_valid_rate"] == 0.0  # lowest score is invalid
    assert report["oracle_upper_bound_mean_valid_rate"] == 1.0


def test_scenario_split_keeps_groups_disjoint_and_stable() -> None:
    groups = [f"s{i}" for i in range(10)] * 3
    split = _scenario_split(groups, 0.3)
    train, test = set(split["train_scenario_groups"]), set(split["test_scenario_groups"])
    assert train and test and not train & test
    assert train | test == set(groups)
    assert split == _scenario_split(list(reversed(groups)), 0.3)


def test_scenario_split_needs_two_groups() -> None:
    with pytest.raises(ValueError):
        _scenario_split(["only"], 0.5)
