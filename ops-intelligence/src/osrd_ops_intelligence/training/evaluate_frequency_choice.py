"""Evaluate model-guided insertion-frequency choice on matched OSRD episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.training.evaluate_route_choice import (
    _actual_existing_conflicts,
    _actual_positive_arrival_delay_s,
    _baseline_signature,
    _mean_predicted_existing_risk,
    _read_cohort,
)


def _frequency_neutral_baseline_signature(episode: dict[str, Any]) -> str:
    normalized = dict(episode)
    inputs = dict(episode["inputs"])
    action = dict(inputs["action"])
    action["frequency_s"] = 300
    action["departure_phase_s"] = 0
    inputs["action"] = action
    normalized["inputs"] = inputs
    return _baseline_signature(normalized)


def evaluate(
    *,
    model_path: Path,
    frequency_a_directories: list[Path],
    frequency_b_directories: list[Path],
    frequency_a_s: int,
    frequency_b_s: int,
    expected_track: str,
) -> dict[str, Any]:
    model_report = json.loads((model_path.parent / "report.json").read_text(encoding="utf-8"))
    test_ids = set(model_report["split"]["test_episode_ids"])
    if not test_ids:
        raise ValueError("Training report has no test episode IDs")

    cohorts: list[dict[tuple[int, int], dict[str, Any]]] = []
    for directories, expected_frequency in (
        (frequency_a_directories, frequency_a_s),
        (frequency_b_directories, frequency_b_s),
    ):
        episodes: dict[tuple[int, int], dict[str, Any]] = {}
        for directory in directories:
            cohort = _read_cohort(directory, expected_track)
            if episodes.keys() & cohort.keys():
                raise ValueError(f"Cohorts contain overlapping episode keys: {directory}")
            for episode in cohort.values():
                actual = int(episode["inputs"]["action"]["frequency_s"])
                if actual != expected_frequency:
                    raise ValueError(
                        f"Expected frequency {expected_frequency}s, got {actual}s "
                        f"for {episode['episode_id']}"
                    )
            episodes.update(cohort)
        cohorts.append(episodes)

    frequency_a, frequency_b = cohorts
    matched_keys = sorted(frequency_a.keys() & frequency_b.keys())
    if not matched_keys or set(frequency_a) != set(frequency_b):
        raise ValueError("Frequency cohorts must contain the same matched seed/episode keys")

    pairs: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = {
        name: [] for name in ("frequency_a", "frequency_b", "model", "random", "oracle")
    }
    conflicts: dict[str, list[int]] = {
        name: [] for name in ("frequency_a", "frequency_b", "model", "random", "oracle")
    }
    delays: dict[str, list[float]] = {
        name: [] for name in ("frequency_a", "frequency_b", "model", "random", "oracle")
    }
    correct: list[float] = []
    model_digests: set[str] = set()
    eligibility: set[bool] = set()

    for key in matched_keys:
        a, b = frequency_a[key], frequency_b[key]
        if _frequency_neutral_baseline_signature(a) != _frequency_neutral_baseline_signature(b):
            raise ValueError(f"Matched frequencies do not share the same baseline: {key}")
        if a["episode_id"] not in test_ids or b["episode_id"] not in test_ids:
            raise ValueError(f"A matched frequency candidate was used outside test: {key}")
        risk_a, delay_a, digest_a, eligible_a = _mean_predicted_existing_risk(model_path, a)
        risk_b, delay_b, digest_b, eligible_b = _mean_predicted_existing_risk(model_path, b)
        if digest_a != digest_b:
            raise ValueError("Model artifact changed during frequency-choice evaluation")
        model_digests.update((digest_a, digest_b))
        eligibility.update((eligible_a, eligible_b))

        conflicts_a, conflicts_b = _actual_existing_conflicts(a), _actual_existing_conflicts(b)
        actual_delay_a = _actual_positive_arrival_delay_s(a)
        actual_delay_b = _actual_positive_arrival_delay_s(b)
        selected = 0 if (risk_a, delay_a) <= (risk_b, delay_b) else 1
        oracle_actual = min((conflicts_a, actual_delay_a), (conflicts_b, actual_delay_b))
        if (conflicts_a, actual_delay_a) == (conflicts_b, actual_delay_b):
            correct.append(0.5)
        else:
            chosen = (
                (conflicts_a, actual_delay_a) if selected == 0 else (conflicts_b, actual_delay_b)
            )
            correct.append(float(chosen == oracle_actual))

        actual_conflicts = (conflicts_a, conflicts_b)
        actual_delays = (actual_delay_a, actual_delay_b)
        predicted_risks = (risk_a, risk_b)
        predicted_delays = (delay_a, delay_b)
        chosen_conflicts, chosen_delay = actual_conflicts[selected], actual_delays[selected]
        pairs.append(
            {
                "seed": key[0],
                "episode_index": key[1],
                "frequencies_s": [frequency_a_s, frequency_b_s],
                "predicted_mean_conflict_probability": list(predicted_risks),
                "predicted_positive_arrival_delay_s": list(predicted_delays),
                "actual_new_conflicted_existing_services": list(actual_conflicts),
                "actual_positive_arrival_delay_s": list(actual_delays),
                "selected_frequency_s": frequency_a_s if selected == 0 else frequency_b_s,
                "selected_objective": [chosen_conflicts, chosen_delay],
                "oracle_objective": list(oracle_actual),
            }
        )
        values = {
            "frequency_a": (risk_a, conflicts_a, actual_delay_a),
            "frequency_b": (risk_b, conflicts_b, actual_delay_b),
            "model": (predicted_risks[selected], chosen_conflicts, chosen_delay),
            "random": (
                (risk_a + risk_b) / 2,
                (conflicts_a + conflicts_b) / 2,
                (actual_delay_a + actual_delay_b) / 2,
            ),
            "oracle": (
                min(risk_a, risk_b),
                oracle_actual[0],
                oracle_actual[1],
            ),
        }
        for name, (risk, conflict, delay) in values.items():
            scores[name].append(risk)
            conflicts[name].append(conflict)
            delays[name].append(delay)

    count = len(matched_keys)

    def mean(values: list[float]) -> float:
        return sum(values) / count

    return {
        "schema": "osrd-frequency-choice-evaluation/v1",
        "model_sha256": next(iter(model_digests)),
        "heldout_pair_count": count,
        "candidate_action": "service-insertion frequency on a fixed route and window",
        "route_track": expected_track,
        "frequency_options_s": [frequency_a_s, frequency_b_s],
        "test_only_episode_ids_verified": True,
        "matched_baselines_verified_after_frequency_normalization": True,
        "operational_ranking_eligible": all(eligibility),
        "ranking_gate_status": model_report["promotion_assessment"][
            "offline_candidate_ranking_gate_passed"
        ],
        "pairwise_objective_accuracy_with_half_credit_for_actual_ties": mean(correct),
        "mean_metrics": {
            name: {
                "predicted_mean_conflict_probability": mean(scores[name]),
                "actual_new_conflicted_existing_services": mean(conflicts[name]),
                "actual_positive_arrival_delay_s": mean(delays[name]),
            }
            for name in scores
        },
        "pairs": pairs,
        "limitations": [
            "Episodes are OSRD simulations on one OSM-derived Belgian infrastructure.",
            "Four held-out switch locations do not establish independent-network transfer.",
            "Predictions do not authorize schedule ranking; validate the full plan in OSRD.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frequency-a-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--frequency-b-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--frequency-a-s", type=int, required=True)
    parser.add_argument("--frequency-b-s", type=int, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        model_path=args.model,
        frequency_a_directories=args.frequency_a_data_dir,
        frequency_b_directories=args.frequency_b_data_dir,
        frequency_a_s=args.frequency_a_s,
        frequency_b_s=args.frequency_b_s,
        expected_track=args.track,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
