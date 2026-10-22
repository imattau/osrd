"""Evaluate model-guided first-departure phase choice on matched OSRD episodes."""

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


def _phase_neutral_baseline_signature(episode: dict[str, Any]) -> str:
    normalized = dict(episode)
    inputs = dict(episode["inputs"])
    action = dict(inputs["action"])
    action["departure_phase_s"] = 0
    inputs["action"] = action
    normalized["inputs"] = inputs
    return _baseline_signature(normalized)


def evaluate(
    *,
    model_path: Path,
    phase_a_directories: list[Path],
    phase_b_directories: list[Path],
    phase_a_s: int,
    phase_b_s: int,
    expected_track: str,
) -> dict[str, Any]:
    model_report = json.loads(
        (model_path.parent / "report.json").read_text(encoding="utf-8")
    )
    test_ids = set(model_report["split"]["test_episode_ids"])
    if not test_ids:
        raise ValueError("Training report has no test episode IDs")
    candidates: list[dict[tuple[int, int], dict[str, Any]]] = []
    for directories, expected_phase in (
        (phase_a_directories, phase_a_s),
        (phase_b_directories, phase_b_s),
    ):
        cohort_rows: dict[tuple[int, int], dict[str, Any]] = {}
        for directory in directories:
            cohort = _read_cohort(directory, expected_track)
            if cohort_rows.keys() & cohort.keys():
                raise ValueError(f"Cohorts contain overlapping episode keys: {directory}")
            for _key, episode in cohort.items():
                actual_phase = int(episode["inputs"]["action"].get("departure_phase_s", 0))
                if actual_phase != expected_phase:
                    raise ValueError(
                        f"Expected departure phase {expected_phase}s, got {actual_phase}s "
                        f"for {episode['episode_id']}"
                    )
            cohort_rows.update(cohort)
        candidates.append(cohort_rows)

    phase_a, phase_b = candidates
    matched_keys = sorted(phase_a.keys() & phase_b.keys())
    if not matched_keys or set(phase_a) != set(phase_b):
        raise ValueError("Phase cohorts must contain the same matched seed/episode keys")

    pair_rows: list[dict[str, Any]] = []
    model_digests: set[str] = set()
    operational_eligibility: set[bool] = set()
    empty_metrics = {
        "risk": 0.0,
        "predicted_delay": 0.0,
        "actual_conflicts": 0.0,
        "actual_delay": 0.0,
    }
    candidate_metrics = {
        name: dict(empty_metrics)
        for name in ("phase_a", "phase_b", "model", "oracle", "random")
    }
    objective_correct: list[float] = []
    model_ties = 0

    for key in matched_keys:
        a, b = phase_a[key], phase_b[key]
        if _phase_neutral_baseline_signature(a) != _phase_neutral_baseline_signature(b):
            raise ValueError(f"Matched phases do not share the same baseline: {key}")
        if a["episode_id"] not in test_ids or b["episode_id"] not in test_ids:
            raise ValueError(f"A matched phase candidate was used outside test: {key}")
        risk_a, delay_a, digest_a, eligible_a = _mean_predicted_existing_risk(model_path, a)
        risk_b, delay_b, digest_b, eligible_b = _mean_predicted_existing_risk(model_path, b)
        if digest_a != digest_b:
            raise ValueError("Model artifact changed during phase-choice evaluation")
        model_digests.add(digest_a)
        operational_eligibility.update((eligible_a, eligible_b))
        conflicts_a, conflicts_b = _actual_existing_conflicts(a), _actual_existing_conflicts(b)
        actual_delay_a = _actual_positive_arrival_delay_s(a)
        actual_delay_b = _actual_positive_arrival_delay_s(b)
        # Lexicographic planner objective: fewer new conflicts, then less positive delay.
        objective_a = (risk_a, delay_a)
        objective_b = (risk_b, delay_b)
        if objective_a == objective_b:
            model_ties += 1
        selected = "phase_a" if objective_a <= objective_b else "phase_b"
        actual_a = (conflicts_a, actual_delay_a)
        actual_b = (conflicts_b, actual_delay_b)
        oracle = min(actual_a, actual_b)
        selected_actual = actual_a if selected == "phase_a" else actual_b
        if actual_a == actual_b:
            objective_correct.append(0.5)
        else:
            objective_correct.append(float(selected_actual == oracle))

        values = {
            "phase_a": (risk_a, delay_a, conflicts_a, actual_delay_a),
            "phase_b": (risk_b, delay_b, conflicts_b, actual_delay_b),
            "model": (
                risk_a if selected == "phase_a" else risk_b,
                delay_a if selected == "phase_a" else delay_b,
                selected_actual[0],
                selected_actual[1],
            ),
            "oracle": (min(risk_a, risk_b), min(delay_a, delay_b), oracle[0], oracle[1]),
            "random": (
                (risk_a + risk_b) / 2,
                (delay_a + delay_b) / 2,
                (conflicts_a + conflicts_b) / 2,
                (actual_delay_a + actual_delay_b) / 2,
            ),
        }
        for name, metrics in values.items():
            for metric, value in zip(
                ("risk", "predicted_delay", "actual_conflicts", "actual_delay"),
                metrics,
                strict=True,
            ):
                candidate_metrics[name][metric] += value
        pair_rows.append(
            {
                "seed": key[0],
                "episode_index": key[1],
                "frequency_s": int(a["inputs"]["action"]["frequency_s"]),
                "phase_a_s": phase_a_s,
                "phase_b_s": phase_b_s,
                "predicted_mean_conflict_probability": [risk_a, risk_b],
                "predicted_positive_arrival_delay_s": [delay_a, delay_b],
                "actual_new_conflicted_existing_services": [conflicts_a, conflicts_b],
                "actual_positive_arrival_delay_s": [actual_delay_a, actual_delay_b],
                "selected_phase": phase_a_s if selected == "phase_a" else phase_b_s,
                "oracle_objective": list(oracle),
                "selected_objective": list(selected_actual),
            }
        )

    pair_count = len(matched_keys)
    means = {
        name: {metric: value / pair_count for metric, value in values.items()}
        for name, values in candidate_metrics.items()
    }
    return {
        "schema": "osrd-phase-choice-evaluation/v1",
        "model_sha256": next(iter(model_digests)),
        "heldout_pair_count": pair_count,
        "candidate_action": "first-departure phase on a fixed route",
        "route_track": expected_track,
        "phase_options_s": [phase_a_s, phase_b_s],
        "test_only_episode_ids_verified": True,
        "baseline_schedules_perturbations_and_other_actions_match_verified": True,
        "operational_ranking_eligible": all(operational_eligibility),
        "ranking_gate_status": model_report["promotion_assessment"][
            "offline_candidate_ranking_gate_passed"
        ],
        "mean_metrics": means,
        "pairwise_objective_accuracy_with_half_credit_for_actual_ties": sum(
            objective_correct
        )
        / pair_count,
        "model_score_ties": model_ties,
        "pairs": pair_rows,
        "limitations": [
            "This is a synthetic one-infrastructure timetable-phase benchmark.",
            "Arrival-head scores are research diagnostics outside the model's calibrated "
            "infrastructure scope.",
            "Predictions do not authorize service ranking; validate each whole timetable in OSRD.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase-a-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--phase-b-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--phase-a-s", type=int, required=True)
    parser.add_argument("--phase-b-s", type=int, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        model_path=args.model,
        phase_a_directories=args.phase_a_data_dir,
        phase_b_directories=args.phase_b_data_dir,
        phase_a_s=args.phase_a_s,
        phase_b_s=args.phase_b_s,
        expected_track=args.track,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
