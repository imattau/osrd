"""Evaluate fixed-frequency departure-phase choices on matched OSRD replays."""

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
)
from osrd_ops_intelligence.models.replay_impact_gnn import ReplayImpactGNN
from osrd_ops_intelligence.training.evaluate_frequency_grid import _predict_existing_risk
from osrd_ops_intelligence.training.evaluate_route_choice import (
    _actual_existing_conflicts,
    _actual_positive_arrival_delay_s,
    _baseline_signature,
    _read_cohort,
)


def evaluate(
    *,
    model_path: Path,
    phase_roots: dict[int, Path],
    switch_ids: list[int],
    frequency_options_s: list[int],
) -> dict[str, Any]:
    phases = sorted(phase_roots)
    if len(phases) < 2 or len(set(phases)) != len(phases):
        raise ValueError("At least two unique phase roots are required")
    if len(set(switch_ids)) != len(switch_ids) or not switch_ids:
        raise ValueError("Switch IDs must be a non-empty unique list")
    if len(set(frequency_options_s)) != len(frequency_options_s) or not frequency_options_s:
        raise ValueError("Frequency options must be a non-empty unique list")

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

    cases_by_group: dict[tuple[int, int, int, int], dict[int, dict[str, Any]]] = {}
    track_by_switch: dict[int, str] = {}
    for switch_id in switch_ids:
        for frequency_s in frequency_options_s:
            for phase_s, root in phase_roots.items():
                suffix = (
                    f"switch-{switch_id}-test-freq-{frequency_s}-v1"
                    if phase_s == 0
                    else f"switch-{switch_id}-test-freq-{frequency_s}-phase-{phase_s}-v1"
                )
                cohort_dir = root / suffix
                sample = json.loads(
                    (cohort_dir / "episodes.jsonl").open(encoding="utf-8").readline()
                )
                train_index = int(sample["inputs"]["action"]["train_index"])
                track_id = sample["inputs"]["schedule_payloads"][train_index]["path"][1][
                    "location"
                ]["track"]
                track_by_switch[switch_id] = track_id
                cohort = _read_cohort(cohort_dir, track_id)
                if len(cohort) != 40:
                    raise ValueError(f"Expected 40 episodes in {cohort_dir}; found {len(cohort)}")
                for key, episode in cohort.items():
                    if episode["episode_id"] not in test_ids:
                        raise ValueError(
                            f"Episode {episode['episode_id']} is outside the frozen test split"
                        )
                    action = dict(episode["inputs"]["action"])
                    if int(action["frequency_s"]) != frequency_s:
                        raise ValueError("Frequency does not match the cohort name")
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
                    actual_conflicts = _actual_existing_conflicts(episode)
                    paired_rows = episode["labels"]["paired_by_train_index"]
                    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
                    simulation_valid = all(
                        row.get("treatment_status") == "success"
                        and (
                            int(row["train_index"]) >= baseline_count
                            or row.get("baseline_status") == "success"
                        )
                        for row in paired_rows
                    )
                    actual_new_conflicts = int(episode["labels"]["new_conflict_count"])
                    group_key = (switch_id, frequency_s, key[0], key[1])
                    cases_by_group.setdefault(group_key, {})[phase_s] = {
                        "seed": key[0],
                        "baseline_signature": _baseline_signature(normalized),
                        "predicted_objective": (risk, delay_s),
                        "actual_objective": (
                            actual_conflicts,
                            _actual_positive_arrival_delay_s(episode),
                        ),
                        "accepted": simulation_valid and actual_new_conflicts == 0,
                    }

    context_rows: list[dict[str, Any]] = []
    selected_phases: Counter[int] = Counter()
    pairwise_scores: list[float] = []
    selected_exact: list[float] = []
    fixed_exact: dict[int, list[float]] = {phase: [] for phase in phases}
    selected_conflicts: list[float] = []
    fixed_conflicts: dict[int, list[float]] = {phase: [] for phase in phases}
    oracle_conflicts: list[float] = []
    model_validation_calls: list[int] = []
    phase_first_validation_calls: list[int] = []
    feasible_context_count = 0
    by_frequency: dict[int, dict[str, list[float]]] = {
        frequency: {"accuracy": [], "selected_conflicts": [], "oracle_conflicts": []}
        for frequency in frequency_options_s
    }
    per_site: dict[str, dict[str, Any]] = {}

    def mean(values: list[float] | list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    for switch_id in switch_ids:
        site_contexts: list[dict[str, Any]] = []
        site_accuracy: list[float] = []
        for frequency_s in frequency_options_s:
            keys = sorted(
                key for key in cases_by_group if key[0] == switch_id and key[1] == frequency_s
            )
            if len(keys) != 40:
                raise ValueError(
                    f"Expected 40 matched contexts for switch {switch_id}, "
                    f"frequency {frequency_s}; found {len(keys)}"
                )
            for group_key in keys:
                candidates = cases_by_group[group_key]
                if set(candidates) != set(phases):
                    raise ValueError(f"Phase alternatives are unmatched for {group_key}")
                if len({row["baseline_signature"] for row in candidates.values()}) != 1:
                    raise ValueError(f"Phase alternatives do not share a baseline for {group_key}")
                selected = min(
                    phases,
                    key=lambda phase: (*candidates[phase]["predicted_objective"], phase),
                )
                selected_actual = candidates[selected]["actual_objective"]
                oracle = min(row["actual_objective"] for row in candidates.values())
                model_order = sorted(
                    phases,
                    key=lambda phase: (*candidates[phase]["predicted_objective"], phase),
                )
                deterministic_order = sorted(phases)

                def validations_until_accept(
                    order: list[int], candidate_rows: dict[int, dict[str, Any]]
                ) -> int:
                    return next(
                        (
                            index
                            for index, phase in enumerate(order, start=1)
                            if candidate_rows[phase]["accepted"]
                        ),
                        len(order),
                    )

                model_calls = validations_until_accept(model_order, candidates)
                phase_first_calls = validations_until_accept(deterministic_order, candidates)
                model_validation_calls.append(model_calls)
                phase_first_validation_calls.append(phase_first_calls)
                feasible_context_count += int(any(row["accepted"] for row in candidates.values()))
                selected_phases[selected] += 1
                selected_exact.append(float(selected_actual == oracle))
                selected_conflicts.append(float(selected_actual[0]))
                oracle_conflicts.append(float(oracle[0]))
                for phase in phases:
                    actual = candidates[phase]["actual_objective"]
                    fixed_conflicts[phase].append(float(actual[0]))
                    fixed_exact[phase].append(float(actual == oracle))
                phase_scores: list[float] = []
                for index, phase_a in enumerate(phases):
                    for phase_b in phases[index + 1 :]:
                        actual_a = candidates[phase_a]["actual_objective"]
                        actual_b = candidates[phase_b]["actual_objective"]
                        pred_a = candidates[phase_a]["predicted_objective"]
                        pred_b = candidates[phase_b]["predicted_objective"]
                        score = (
                            0.5
                            if actual_a == actual_b or pred_a == pred_b
                            else float((actual_a < actual_b) == (pred_a < pred_b))
                        )
                        phase_scores.append(score)
                        pairwise_scores.append(score)
                site_accuracy.extend(phase_scores)
                by_frequency[frequency_s]["accuracy"].append(sum(phase_scores) / len(phase_scores))
                by_frequency[frequency_s]["selected_conflicts"].append(float(selected_actual[0]))
                by_frequency[frequency_s]["oracle_conflicts"].append(float(oracle[0]))
                row = {
                    "switch_id": switch_id,
                    "frequency_s": frequency_s,
                    "seed": group_key[2],
                    "episode_index": group_key[3],
                    "selected_phase_s": selected,
                    "selected_objective": list(selected_actual),
                    "oracle_objective": list(oracle),
                    "fixed_phase_objectives": {
                        str(phase): list(candidates[phase]["actual_objective"]) for phase in phases
                    },
                    "candidate_scores_and_acceptance": {
                        str(phase): {
                            "predicted_objective": list(candidates[phase]["predicted_objective"]),
                            "actual_objective": list(candidates[phase]["actual_objective"]),
                            "accepted_by_zero_new_conflicts_and_successful_simulation": candidates[
                                phase
                            ]["accepted"],
                        }
                        for phase in phases
                    },
                    "gnn_first_accept_validation_calls": model_calls,
                    "phase_ascending_first_accept_validation_calls": phase_first_calls,
                }
                context_rows.append(row)
                site_contexts.append(row)
        per_site[str(switch_id)] = {
            "track_id": track_by_switch[switch_id],
            "matched_context_count": len(site_contexts),
            "pairwise_objective_accuracy_half_credit_ties": (
                sum(site_accuracy) / len(site_accuracy)
            ),
            "six_way_exact_objective_match_rate": sum(
                row["selected_objective"] == row["oracle_objective"] for row in site_contexts
            )
            / len(site_contexts),
            "mean_validation_calls_to_first_accepted_candidate": {
                "gnn_order": mean(
                    [row["gnn_first_accept_validation_calls"] for row in site_contexts]
                ),
                "ascending_phase": mean(
                    [row["phase_ascending_first_accept_validation_calls"] for row in site_contexts]
                ),
            },
        }

    promotion = model_report["promotion_assessment"]
    return {
        "schema": "osrd-phase-grid-evaluation/v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "phase_options_s": phases,
        "frequency_options_s": frequency_options_s,
        "test_only_episode_ids_verified": True,
        "matched_baselines_verified": True,
        "promotion_assessment": {
            "offline_candidate_ranking_gate_passed": promotion[
                "offline_candidate_ranking_gate_passed"
            ],
            "production_promotion_eligible": promotion["production_promotion_eligible"],
        },
        "aggregate": {
            "independent_switch_sites": len(switch_ids),
            "matched_context_count": len(context_rows),
            "candidate_episode_count": len(context_rows) * len(phases),
            "pairwise_comparison_count": len(pairwise_scores),
            "pairwise_objective_accuracy_half_credit_ties": mean(pairwise_scores),
            "gnn_exact_objective_match_rate": mean(selected_exact),
            "selected_phase_distribution_s": {
                str(phase): count for phase, count in sorted(selected_phases.items())
            },
            "mean_actual_new_conflicts": {
                "gnn": mean(selected_conflicts),
                "oracle": mean(oracle_conflicts),
                **{str(phase): mean(values) for phase, values in fixed_conflicts.items()},
            },
            "fixed_phase_exact_objective_match_rate": {
                str(phase): mean(values) for phase, values in fixed_exact.items()
            },
            "retrospective_validation_order_study": {
                "definition": (
                    "Count candidates in rank order until one has successful paired simulation "
                    "and zero new conflicts; count all alternatives when none qualifies."
                ),
                "contexts_with_any_accepted_candidate": feasible_context_count,
                "gnn_order_mean_osrd_validations": mean(model_validation_calls),
                "ascending_phase_mean_osrd_validations": mean(phase_first_validation_calls),
                "mean_validations_saved_by_gnn_order": mean(
                    [
                        baseline - model
                        for baseline, model in zip(
                            phase_first_validation_calls, model_validation_calls, strict=True
                        )
                    ]
                ),
                "warning": (
                    "All candidates were already simulated to create this benchmark; this is a "
                    "retrospective ordering estimate, not observed reduction in OSRD runs."
                ),
            },
        },
        "by_frequency": {
            str(frequency): {key: mean(values) for key, values in metrics.items()}
            for frequency, metrics in by_frequency.items()
        },
        "sites": per_site,
        "context_results": context_rows,
        "limitations": [
            "Phase ranking holds service frequency and route fixed.",
            "The four held-out switch sites belong to one OSM-derived Swiss infrastructure.",
            "Swiss observations condition synthetic OSRD replays; they are not "
            "counterfactual labels.",
            "OSRD simulation remains authoritative for every complete candidate timetable.",
            "The model’s offline promotion gate remains false; service-side ranking is disabled.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase-root", nargs=2, action="append", required=True)
    parser.add_argument("--switch-id", type=int, action="append", required=True)
    parser.add_argument("--frequency-options-s", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {int(phase): Path(root) for phase, root in args.phase_root}
    if len(roots) != len(args.phase_root):
        parser.error("Each --phase-root phase must be unique")
    result = evaluate(
        model_path=args.model,
        phase_roots=roots,
        switch_ids=args.switch_id,
        frequency_options_s=args.frequency_options_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
