"""Evaluate GNN choice across matched OSRD service-frequency alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
from osrd_ops_intelligence.training.evaluate_route_choice import (
    _actual_existing_conflicts,
    _actual_positive_arrival_delay_s,
    _baseline_signature,
    _read_cohort,
)


def _predict_existing_risk(
    model: ReplayImpactGNN,
    episode: dict[str, Any],
    *,
    feature_names: tuple[str, ...],
    edge_feature_names: tuple[str, ...],
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    arrival_scale_s: float,
    calibration: dict[str, Any] | None,
) -> tuple[float, float]:
    rows = feature_rows_for_episode(episode, feature_names)
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows], dtype=torch.float32
    )
    x = (x - feature_mean) / feature_scale
    edge_index, edge_attr = interaction_graph(
        episode, device="cpu", feature_names=edge_feature_names
    )
    with torch.inference_mode():
        output = model(x, edge_index, edge_attr)
    conflict_logits = output[:, 1]
    if calibration is not None:
        conflict_logits = conflict_logits * float(calibration["slope"]) + float(
            calibration["intercept"]
        )
    probabilities = torch.sigmoid(conflict_logits).tolist()
    arrival_deltas = (output[:, 0] * arrival_scale_s).tolist()
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    mean_risk = sum(probabilities[:baseline_count]) / baseline_count
    positive_delay_s = sum(max(0.0, float(value)) for value in arrival_deltas[:baseline_count])
    return mean_risk, positive_delay_s


def evaluate(
    *,
    model_path: Path,
    data_root: Path,
    switch_ids: list[int],
    frequency_options_s: list[int],
    fixed_baseline_frequencies_s: list[int],
) -> dict[str, Any]:
    if len(set(switch_ids)) != len(switch_ids) or not switch_ids:
        raise ValueError("Switch IDs must be a non-empty unique list")
    if len(set(frequency_options_s)) != len(frequency_options_s) or len(frequency_options_s) < 2:
        raise ValueError("At least two unique frequency options are required")
    if not set(fixed_baseline_frequencies_s) <= set(frequency_options_s):
        raise ValueError("Fixed baseline frequencies must be included in the candidate grid")

    model_report = json.loads((model_path.parent / "report.json").read_text(encoding="utf-8"))
    test_ids = set(model_report["split"]["test_episode_ids"])
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_names = tuple(payload["feature_names"])
    edge_feature_names = tuple(payload.get("edge_feature_names", LEGACY_EDGE_FEATURE_NAMES))
    if edge_feature_names not in (EDGE_FEATURE_NAMES, LEGACY_EDGE_FEATURE_NAMES):
        raise ValueError("Model graph-edge feature contract is unsupported")
    feature_mean = torch.tensor(payload["feature_mean"], dtype=torch.float32)
    feature_scale = torch.tensor(payload["feature_scale"], dtype=torch.float32)
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
    calibration = payload.get("conflict_probability_calibration")
    arrival_scale_s = float(payload.get("arrival_delta_target_scale_s", 1.0))

    contexts_by_site: dict[int, dict[tuple[int, int], dict[int, dict[str, Any]]]] = {}
    track_by_site: dict[int, str] = {}
    for switch_id in switch_ids:
        contexts: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
        for frequency_s in frequency_options_s:
            cohort_dir = data_root / f"switch-{switch_id}-test-freq-{frequency_s}-v1"
            sample_path = cohort_dir / "episodes.jsonl"
            sample = json.loads(sample_path.open(encoding="utf-8").readline())
            action_index = int(sample["inputs"]["action"]["train_index"])
            track_id = sample["inputs"]["schedule_payloads"][action_index]["path"][1]["location"][
                "track"
            ]
            track_by_site[switch_id] = track_id
            episodes = _read_cohort(cohort_dir, track_id)
            for key, episode in episodes.items():
                if episode["episode_id"] not in test_ids:
                    raise ValueError(
                        f"Episode {episode['episode_id']} is outside the frozen test split"
                    )
                action = dict(episode["inputs"]["action"])
                action["frequency_s"] = 300
                action["departure_phase_s"] = 0
                normalized = dict(episode)
                inputs = dict(episode["inputs"])
                inputs["action"] = action
                normalized["inputs"] = inputs
                risk, delay_s = _predict_existing_risk(
                    model,
                    episode,
                    feature_names=feature_names,
                    edge_feature_names=edge_feature_names,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    arrival_scale_s=arrival_scale_s,
                    calibration=calibration,
                )
                contexts.setdefault(key, {})[frequency_s] = {
                    "baseline_signature": _baseline_signature(normalized),
                    "predicted_objective": (risk, delay_s),
                    "actual_objective": (
                        _actual_existing_conflicts(episode),
                        _actual_positive_arrival_delay_s(episode),
                    ),
                }
            if len(episodes) != 40:
                raise ValueError(f"Expected 40 episodes in {cohort_dir}; found {len(episodes)}")
        contexts_by_site[switch_id] = contexts

    context_rows: list[dict[str, Any]] = []
    per_frequency: dict[int, dict[str, list[float]]] = {
        frequency_s: {
            "predicted_mean_conflict_probability": [],
            "predicted_positive_arrival_delay_s": [],
            "actual_new_conflicts": [],
            "actual_positive_arrival_delay_s": [],
        }
        for frequency_s in frequency_options_s
    }
    pairwise_scores: list[float] = []
    site_summaries: dict[str, dict[str, Any]] = {}
    selected_frequency_counts: Counter[int] = Counter()
    exact_matches: list[float] = []
    fixed_baseline_matches: dict[int, list[float]] = {
        frequency_s: [] for frequency_s in fixed_baseline_frequencies_s
    }
    selected_conflicts: list[float] = []
    baseline_conflicts: dict[int, list[float]] = {
        frequency_s: [] for frequency_s in fixed_baseline_frequencies_s
    }
    oracle_conflicts: list[float] = []

    for switch_id, contexts in contexts_by_site.items():
        if len(contexts) != 40:
            raise ValueError(f"Expected 40 matched contexts for switch {switch_id}")
        site_pairs: list[float] = []
        site_exact: list[float] = []
        for key, candidates in sorted(contexts.items()):
            if set(candidates) != set(frequency_options_s):
                raise ValueError(f"Frequency candidates are unmatched for {switch_id}, {key}")
            if len({row["baseline_signature"] for row in candidates.values()}) != 1:
                raise ValueError(
                    f"Frequency candidates do not share a baseline for {switch_id}, {key}"
                )
            selected_frequency = min(
                frequency_options_s,
                key=lambda value: (*candidates[value]["predicted_objective"], value),
            )
            selected_objective = candidates[selected_frequency]["actual_objective"]
            oracle_objective = min(row["actual_objective"] for row in candidates.values())
            selected_frequency_counts[selected_frequency] += 1
            site_exact.append(float(selected_objective == oracle_objective))
            exact_matches.append(float(selected_objective == oracle_objective))
            selected_conflicts.append(float(selected_objective[0]))
            oracle_conflicts.append(float(oracle_objective[0]))
            for baseline_frequency in fixed_baseline_frequencies_s:
                baseline_objective = candidates[baseline_frequency]["actual_objective"]
                fixed_baseline_matches[baseline_frequency].append(
                    float(baseline_objective == oracle_objective)
                )
                baseline_conflicts[baseline_frequency].append(float(baseline_objective[0]))
            for frequency_s, row in candidates.items():
                predicted = row["predicted_objective"]
                actual = row["actual_objective"]
                per_frequency[frequency_s]["predicted_mean_conflict_probability"].append(
                    predicted[0]
                )
                per_frequency[frequency_s]["predicted_positive_arrival_delay_s"].append(
                    predicted[1]
                )
                per_frequency[frequency_s]["actual_new_conflicts"].append(actual[0])
                per_frequency[frequency_s]["actual_positive_arrival_delay_s"].append(actual[1])
            frequencies = list(frequency_options_s)
            for index, frequency_a in enumerate(frequencies):
                for frequency_b in frequencies[index + 1 :]:
                    candidate_a, candidate_b = candidates[frequency_a], candidates[frequency_b]
                    actual_a, actual_b = (
                        candidate_a["actual_objective"],
                        candidate_b["actual_objective"],
                    )
                    predicted_a, predicted_b = (
                        candidate_a["predicted_objective"],
                        candidate_b["predicted_objective"],
                    )
                    score = (
                        0.5
                        if actual_a == actual_b or predicted_a == predicted_b
                        else float((actual_a < actual_b) == (predicted_a < predicted_b))
                    )
                    site_pairs.append(score)
                    pairwise_scores.append(score)
            context_rows.append(
                {
                    "switch_id": switch_id,
                    "seed": key[0],
                    "episode_index": key[1],
                    "selected_frequency_s": selected_frequency,
                    "selected_objective": list(selected_objective),
                    "oracle_objective": list(oracle_objective),
                    "six_way_exact_objective_match": selected_objective == oracle_objective,
                }
            )
        site_summaries[str(switch_id)] = {
            "track_id": track_by_site[switch_id],
            "matched_context_count": len(contexts),
            "pairwise_objective_accuracy_half_credit_ties": sum(site_pairs) / len(site_pairs),
            "six_way_exact_objective_match_rate": sum(site_exact) / len(site_exact),
        }

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    aggregate: dict[str, Any] = {
        "site_count": len(switch_ids),
        "matched_context_count": len(context_rows),
        "candidate_frequency_count": len(frequency_options_s),
        "candidate_episode_count": len(context_rows) * len(frequency_options_s),
        "pairwise_comparison_count": len(pairwise_scores),
        "pairwise_objective_accuracy_half_credit_ties": mean(pairwise_scores),
        "gnn_six_way_exact_objective_match_rate": mean(exact_matches),
        "selected_frequency_distribution_s": {
            str(frequency): count for frequency, count in sorted(selected_frequency_counts.items())
        },
        "mean_actual_new_conflicts": {
            "gnn": mean(selected_conflicts),
            "oracle": mean(oracle_conflicts),
            **{str(frequency): mean(values) for frequency, values in baseline_conflicts.items()},
        },
        "exact_objective_match_rate_by_fixed_frequency": {
            str(frequency): mean(values) for frequency, values in fixed_baseline_matches.items()
        },
    }
    return {
        "schema": "osrd-frequency-grid-evaluation/v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "promotion_assessment": {
            "offline_candidate_ranking_gate_passed": model_report["promotion_assessment"][
                "offline_candidate_ranking_gate_passed"
            ],
            "production_promotion_eligible": model_report["promotion_assessment"][
                "production_promotion_eligible"
            ],
        },
        "frequency_options_s": frequency_options_s,
        "test_only_episode_ids_verified": True,
        "matched_baselines_verified": True,
        "selection_objective": (
            "Minimize predicted mean existing-service conflict probability, then predicted sum "
            "of positive arrival-delay changes; frequency is the deterministic tie-break. "
            "Realized objective is lexicographic (newly conflicted existing services, sum of "
            "positive arrival delays)."
        ),
        "aggregate": aggregate,
        "sites": site_summaries,
        "by_frequency": {
            str(frequency): {key: mean(values) for key, values in metrics.items()}
            for frequency, metrics in per_frequency.items()
        },
        "context_results": context_rows,
        "limitations": [
            "This evaluates candidate ordering, not complete timetable synthesis.",
            "OSRD simulation remains authoritative for every candidate timetable.",
            "Held-out sites belong to one OSM-derived Swiss infrastructure.",
            "Swiss actual observations condition synthetic replays; they are not "
            "counterfactual labels.",
            "Frequency ranking does not encode passenger demand or required service intensity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--switch-id", type=int, action="append", required=True)
    parser.add_argument("--frequency-options-s", type=int, nargs="+", required=True)
    parser.add_argument("--fixed-baseline-frequency-s", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        model_path=args.model,
        data_root=args.data_root,
        switch_ids=args.switch_id,
        frequency_options_s=args.frequency_options_s,
        fixed_baseline_frequencies_s=args.fixed_baseline_frequency_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
