"""Evaluate model-guided choice between matched OSRD route alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from osrd_ops_intelligence.features.replay_graph import (
    EDGE_FEATURE_NAMES,
    LEGACY_EDGE_FEATURE_NAMES,
    interaction_graph,
)
from osrd_ops_intelligence.features.replay_v1 import feature_rows_for_episode
from osrd_ops_intelligence.models.replay_impact_gnn import ReplayImpactGNN
from osrd_ops_intelligence.training.predict_replay_gnn import predict_episode


def _read_cohort(directory: Path, expected_track: str) -> dict[tuple[int, int], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    episodes_path = directory / "episodes.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_bytes = episodes_path.read_bytes()
    if hashlib.sha256(episode_bytes).hexdigest() != manifest.get("episodes_sha256"):
        raise ValueError(f"Episode checksum does not match the manifest: {episodes_path}")
    episodes: dict[tuple[int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(episode_bytes.splitlines(), start=1):
        episode = json.loads(line)
        inputs = episode["inputs"]
        action = inputs["action"]
        schedules = inputs["schedule_payloads"]
        baseline = inputs["baseline_schedule_payloads"]
        index = int(action["train_index"])
        if action.get("type") != "service-plan" or index != len(baseline):
            raise ValueError(f"Unexpected action in {episodes_path}:{line_number}")
        route_track = schedules[index]["path"][1]["location"]["track"]
        if route_track != expected_track:
            raise ValueError(
                f"Expected route via {expected_track}, got {route_track} "
                f"in {episodes_path}:{line_number}"
            )
        key = (int(episode["seed"]), int(episode["episode_index"]))
        if key in episodes:
            raise ValueError(f"Duplicate matched-pair key {key} in {episodes_path}")
        episodes[key] = episode
    if manifest.get("episode_count") != len(episodes):
        raise ValueError(f"Episode count does not match the manifest: {episodes_path}")
    return episodes


def _baseline_signature(episode: dict[str, Any]) -> str:
    inputs = episode["inputs"]
    baseline = inputs["baseline_schedule_payloads"]
    first_departure = baseline[0]["start_time"]
    normalized_schedules = [
        {**schedule, "start_time": schedule["start_time"] - first_departure}
        for schedule in baseline
    ]
    context = {
        "schedules": normalized_schedules,
        "perturbations": inputs["background_perturbations"],
        "headway_s": inputs["scenario"]["headway_s"],
        "rolling_stock": inputs["scenario"]["rolling_stock"],
        "frequency_s": inputs["action"]["frequency_s"],
        "departure_phase_s": inputs["action"].get("departure_phase_s", 0),
        "operating_window_s": inputs["action"]["operating_window_s"],
        "retimings": inputs["action"].get("retimings", []),
    }
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _actual_existing_conflicts(episode: dict[str, Any]) -> int:
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    return sum(
        int(bool(row["new_conflict"]))
        for row in episode["labels"]["paired_by_train_index"]
        if int(row["train_index"]) < baseline_count
    )


def _actual_positive_arrival_delay_s(episode: dict[str, Any]) -> float:
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    deltas = [
        max(0.0, float(row["arrival_delta_ms"]) / 1000)
        for row in episode["labels"]["paired_by_train_index"]
        if int(row["train_index"]) < baseline_count and row.get("arrival_delta_ms") is not None
    ]
    return sum(deltas)


def _raw_positive_arrival_delay_prediction_s(
    model_path: Path, episode: dict[str, Any]
) -> float:
    """Read the research arrival head without granting deployment eligibility."""
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_names = tuple(payload["feature_names"])
    rows = feature_rows_for_episode(episode, feature_names)
    mean = torch.tensor(payload["feature_mean"], dtype=torch.float32)
    scale = torch.tensor(payload["feature_scale"], dtype=torch.float32)
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows], dtype=torch.float32
    )
    x = (x - mean) / scale
    edge_feature_names = tuple(payload.get("edge_feature_names", LEGACY_EDGE_FEATURE_NAMES))
    if edge_feature_names not in (EDGE_FEATURE_NAMES, LEGACY_EDGE_FEATURE_NAMES):
        raise ValueError("Model graph-edge feature contract is unsupported")
    edge_index, edge_attr = interaction_graph(
        episode, device="cpu", feature_names=edge_feature_names
    )
    architecture = payload.get("architecture", payload)
    model = ReplayImpactGNN(
        input_channels=len(feature_names),
        hidden_channels=int(architecture["hidden_channels"]),
        heads=int(architecture["heads"]),
        dropout=float(architecture.get("dropout", 0.1)),
        edge_channels=int(architecture.get("edge_channels", len(edge_feature_names))),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        arrival_delta_s = (
            model(x, edge_index, edge_attr)[:, 0]
            * float(payload.get("arrival_delta_target_scale_s", 1.0))
        ).tolist()
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    return sum(max(0.0, float(value)) for value in arrival_delta_s[:baseline_count])


def _mean_predicted_existing_risk(
    model_path: Path, episode: dict[str, Any]
) -> tuple[float, float, str, bool]:
    prediction = predict_episode(model_path, episode)
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    probabilities = [
        float(row["conflict_probability"])
        for row in prediction["services"]
        if int(row["train_index"]) < baseline_count
    ]
    if len(probabilities) != baseline_count:
        raise ValueError("Prediction does not cover every baseline service")
    return (
        sum(probabilities) / len(probabilities),
        _raw_positive_arrival_delay_prediction_s(model_path, episode),
        str(prediction["model_sha256"]),
        bool(prediction["offline_candidate_ranking_eligible"]),
    )


def evaluate(
    *,
    model_path: Path,
    tc0_directories: list[Path],
    tc1_directories: list[Path],
) -> dict[str, Any]:
    model_report_path = model_path.parent / "report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    test_ids = set(model_report["split"]["test_episode_ids"])
    if not test_ids:
        raise ValueError("Training report has no test episode IDs")

    tc0: dict[tuple[int, int], dict[str, Any]] = {}
    tc1: dict[tuple[int, int], dict[str, Any]] = {}
    for directory in tc0_directories:
        cohort = _read_cohort(directory, "TC0")
        if tc0.keys() & cohort.keys():
            raise ValueError(f"TC0 cohorts contain overlapping episode keys: {directory}")
        tc0.update(cohort)
    for directory in tc1_directories:
        cohort = _read_cohort(directory, "TC1")
        if tc1.keys() & cohort.keys():
            raise ValueError(f"TC1 cohorts contain overlapping episode keys: {directory}")
        tc1.update(cohort)

    matched_keys = sorted(tc0.keys() & tc1.keys())
    if not matched_keys or set(tc0) != set(tc1):
        raise ValueError("Route cohorts must contain the same matched seed/episode keys")
    rows: list[dict[str, Any]] = []
    model_digests: set[str] = set()
    ranking_eligible: set[bool] = set()
    selected_regret: list[int] = []
    selected_delay_regret: list[float] = []
    fixed_tc0_regret: list[int] = []
    fixed_tc1_regret: list[int] = []
    route_choice_correct: list[float] = []
    delay_choice_correct: list[float] = []
    conflict_sensitive_choice_correct: list[float] = []
    delay_sensitive_choice_correct: list[float] = []
    conflict_sensitive_pairs = 0
    delay_sensitive_pairs = 0
    fixed_tc0_delay_regret: list[float] = []
    fixed_tc1_delay_regret: list[float] = []
    tc0_delay_total = 0.0
    tc1_delay_total = 0.0
    model_ties = 0
    tc0_actual_total = 0
    tc1_actual_total = 0
    for key in matched_keys:
        episode0, episode1 = tc0[key], tc1[key]
        if _baseline_signature(episode0) != _baseline_signature(episode1):
            raise ValueError(f"Matched alternatives do not share the same baseline: {key}")
        if episode0["episode_id"] not in test_ids or episode1["episode_id"] not in test_ids:
            raise ValueError(f"A matched alternative was used outside the held-out split: {key}")
        score0, predicted_delay0, digest0, eligible0 = _mean_predicted_existing_risk(
            model_path, episode0
        )
        score1, predicted_delay1, digest1, eligible1 = _mean_predicted_existing_risk(
            model_path, episode1
        )
        if digest0 != digest1:
            raise ValueError("Model artifact changed during route-choice evaluation")
        model_digests.add(digest0)
        ranking_eligible.update((eligible0, eligible1))
        actual0 = _actual_existing_conflicts(episode0)
        actual1 = _actual_existing_conflicts(episode1)
        actual_delay0 = _actual_positive_arrival_delay_s(episode0)
        actual_delay1 = _actual_positive_arrival_delay_s(episode1)
        tc0_actual_total += actual0
        tc1_actual_total += actual1
        tc0_delay_total += actual_delay0
        tc1_delay_total += actual_delay1
        oracle = min(actual0, actual1)
        regret0 = actual0 - oracle
        regret1 = actual1 - oracle
        fixed_tc0_regret.append(regret0)
        fixed_tc1_regret.append(regret1)
        selected_route = "TC0" if score0 <= score1 else "TC1"
        if score0 == score1:
            model_ties += 1
        selected_actual = actual0 if selected_route == "TC0" else actual1
        selected_regret.append(selected_actual - oracle)
        if actual0 == actual1:
            route_choice_correct.append(0.5)
        else:
            conflict_sensitive_pairs += 1
            conflict_sensitive_choice_correct.append(
                float(selected_actual < (actual0 if selected_route == "TC1" else actual1))
            )
            route_choice_correct.append(float(selected_actual < max(actual0, actual1)))
        selected_delay_route = "TC0" if predicted_delay0 <= predicted_delay1 else "TC1"
        selected_actual_delay = actual_delay0 if selected_delay_route == "TC0" else actual_delay1
        oracle_delay = min(actual_delay0, actual_delay1)
        selected_delay_regret.append(selected_actual_delay - oracle_delay)
        fixed_tc0_delay_regret.append(actual_delay0 - oracle_delay)
        fixed_tc1_delay_regret.append(actual_delay1 - oracle_delay)
        if actual_delay0 == actual_delay1:
            delay_choice_correct.append(0.5)
        else:
            delay_sensitive_pairs += 1
            delay_sensitive_choice_correct.append(
                float(
                    selected_actual_delay
                    < (actual_delay0 if selected_delay_route == "TC1" else actual_delay1)
                )
            )
            delay_choice_correct.append(
                float(selected_actual_delay < max(actual_delay0, actual_delay1))
            )
        rows.append(
            {
                "seed": key[0],
                "episode_index": key[1],
                "frequency_s": int(episode0["inputs"]["action"]["frequency_s"]),
                "tc0_mean_existing_conflict_probability": score0,
                "tc1_mean_existing_conflict_probability": score1,
                "tc0_predicted_positive_arrival_delay_s": predicted_delay0,
                "tc1_predicted_positive_arrival_delay_s": predicted_delay1,
                "tc0_new_conflicted_existing_services": actual0,
                "tc1_new_conflicted_existing_services": actual1,
                "tc0_positive_arrival_delay_s": actual_delay0,
                "tc1_positive_arrival_delay_s": actual_delay1,
                "selected_route": selected_route,
                "selected_delay_route": selected_delay_route,
                "oracle_new_conflicts": oracle,
                "selected_new_conflicts": selected_actual,
                "oracle_positive_arrival_delay_s": oracle_delay,
                "selected_positive_arrival_delay_s": selected_actual_delay,
            }
        )

    pair_count = len(matched_keys)
    random_regret = [
        (left + right) / 2 for left, right in zip(fixed_tc0_regret, fixed_tc1_regret, strict=True)
    ]
    def average(values: list[float | int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "schema": "osrd-route-choice-evaluation/v1",
        "model_sha256": next(iter(model_digests)),
        "heldout_pair_count": pair_count,
        "conflict_sensitive_pair_count": conflict_sensitive_pairs,
        "arrival_delay_sensitive_pair_count": delay_sensitive_pairs,
        "pairs_with_any_route_outcome_difference": sum(
            1
            for row in rows
            if row["tc0_new_conflicted_existing_services"]
            != row["tc1_new_conflicted_existing_services"]
            or row["tc0_positive_arrival_delay_s"] != row["tc1_positive_arrival_delay_s"]
        ),
        "candidate_routes": ["TC0", "TC1"],
        "test_only_episode_ids_verified": True,
        "baseline_schedule_and_perturbation_match_verified": True,
        "operational_ranking_eligible": all(ranking_eligible),
        "ranking_gate_status": model_report["promotion_assessment"][
            "offline_candidate_ranking_gate_passed"
        ],
        "mean_new_conflicted_existing_services": {
            "always_tc0": tc0_actual_total / pair_count,
            "always_tc1": tc1_actual_total / pair_count,
            "random_route_expectation": (tc0_actual_total + tc1_actual_total) / (2 * pair_count),
            "model_selected": average(
                [row["selected_new_conflicts"] for row in rows]
            ),
            "oracle_lower_bound": average(
                [row["oracle_new_conflicts"] for row in rows]
            ),
        },
        "mean_regret_new_conflicts_vs_pairwise_oracle": {
            "always_tc0": average(fixed_tc0_regret),
            "always_tc1": average(fixed_tc1_regret),
            "random_route_expectation": average(random_regret),
            "model_selected": average(selected_regret),
        },
        "mean_positive_arrival_delay_seconds": {
            "always_tc0": tc0_delay_total / pair_count,
            "always_tc1": tc1_delay_total / pair_count,
            "random_route_expectation": (tc0_delay_total + tc1_delay_total) / (2 * pair_count),
            "model_selected_by_arrival_head": average(
                [row["selected_positive_arrival_delay_s"] for row in rows]
            ),
            "oracle_lower_bound": average(
                [row["oracle_positive_arrival_delay_s"] for row in rows]
            ),
        },
        "mean_positive_arrival_delay_regret_seconds": {
            "always_tc0": average(fixed_tc0_delay_regret),
            "always_tc1": average(fixed_tc1_delay_regret),
            "random_route_expectation": average(
                [(left + right) / 2 for left, right in zip(
                    fixed_tc0_delay_regret, fixed_tc1_delay_regret, strict=True
                )]
            ),
            "model_selected_by_arrival_head": average(selected_delay_regret),
        },
        "pairwise_route_choice_accuracy_with_half_credit_for_actual_ties": average(
            route_choice_correct
        ),
        "pairwise_route_choice_accuracy_on_conflict_sensitive_pairs": average(
            conflict_sensitive_choice_correct
        ),
        "arrival_based_route_choice_accuracy_with_half_credit_for_actual_ties": average(
            delay_choice_correct
        ),
        "arrival_based_route_choice_accuracy_on_delay_sensitive_pairs": average(
            delay_sensitive_choice_correct
        ),
        "model_score_ties": model_ties,
        "pairs": rows,
        "limitations": [
            "This is a synthetic one-infrastructure, two-platform-track benchmark.",
            "Predictions are scored offline and do not authorize candidate ranking in the service.",
            "Arrival-head scores are research diagnostics outside the model's calibrated "
            "infrastructure scope.",
            "Whole-timetable OSRD simulation and conflict detection remain authoritative.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tc0-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--tc1-data-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        model_path=args.model,
        tc0_directories=args.tc0_data_dir,
        tc1_directories=args.tc1_data_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
