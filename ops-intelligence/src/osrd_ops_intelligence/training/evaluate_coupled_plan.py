"""Evaluate GNN ordering of matched service plans with downstream retimings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
    _actual_positive_arrival_delay_s,
)


def _matched_baseline_key(episode: dict[str, Any]) -> str:
    """Hash the shared pre-treatment timetable and empirical context."""
    inputs = episode["inputs"]
    baseline = inputs["baseline_schedule_payloads"]
    first_departure = baseline[0]["start_time"]
    normalized = [
        {**schedule, "start_time": schedule["start_time"] - first_departure}
        for schedule in baseline
    ]
    context = {
        "scenario_group": episode.get("scenario_group"),
        "schedules": normalized,
        "perturbations": inputs["background_perturbations"],
        "headway_s": inputs["scenario"]["headway_s"],
        "rolling_stock": inputs["scenario"]["rolling_stock"],
        "empirical_profile_sha256": inputs["scenario"].get("empirical_delay_profile_sha256"),
    }
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_model(model_path: Path) -> tuple[dict[str, Any], ReplayImpactGNN]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_names = tuple(payload["feature_names"])
    edge_names = tuple(payload.get("edge_feature_names", LEGACY_EDGE_FEATURE_NAMES))
    if edge_names not in (EDGE_FEATURE_NAMES, LEGACY_EDGE_FEATURE_NAMES):
        raise ValueError("Model graph-edge feature contract is unsupported")
    architecture = payload.get("architecture", payload)
    model = ReplayImpactGNN(
        input_channels=len(feature_names),
        hidden_channels=int(architecture["hidden_channels"]),
        heads=int(architecture["heads"]),
        dropout=float(architecture.get("dropout", 0.1)),
        edge_channels=int(architecture.get("edge_channels", len(edge_names))),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _predict(
    model: ReplayImpactGNN,
    episode: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[float, float]:
    feature_names = tuple(payload["feature_names"])
    rows = feature_rows_for_episode(episode, feature_names)
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows], dtype=torch.float32
    )
    x = (x - torch.tensor(payload["feature_mean"], dtype=torch.float32)) / torch.tensor(
        payload["feature_scale"], dtype=torch.float32
    )
    edge_names = tuple(payload.get("edge_feature_names", LEGACY_EDGE_FEATURE_NAMES))
    edge_index, edge_attr = interaction_graph(episode, device="cpu", feature_names=edge_names)
    with torch.inference_mode():
        output = model(x, edge_index, edge_attr)
    logits = output[:, 1]
    calibration = payload.get("conflict_probability_calibration")
    if calibration is not None:
        logits = logits * float(calibration["slope"]) + float(calibration["intercept"])
    probabilities = torch.sigmoid(logits).tolist()
    baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
    arrival_scale_s = float(payload.get("arrival_delta_target_scale_s", 1.0))
    delay_s = sum(
        max(0.0, float(value) * arrival_scale_s) for value in output[:baseline_count, 0].tolist()
    )
    # Include inserted services in the conflict-risk score: their conflicts are
    # part of whether the proposed timetable is acceptable.
    return sum(float(value) for value in probabilities) / len(probabilities), delay_s


def _action_cost(episode: dict[str, Any]) -> tuple[int, int]:
    shifts = [
        int(row["departure_shift_s"]) for row in episode["inputs"]["action"].get("retimings", [])
    ]
    return len(shifts), sum(abs(value) for value in shifts)


def evaluate(
    model_path: Path,
    data_root: Path,
    *,
    partition: str = "test",
    action_cost_weight: float = 0.0,
    max_retime_s: int = 900,
) -> dict[str, Any]:
    if partition not in ("validation", "test"):
        raise ValueError("partition must be validation or test")
    if action_cost_weight < 0 or max_retime_s <= 0:
        raise ValueError("action-cost weight must be nonnegative and max retime must be positive")
    report = json.loads((model_path.parent / "report.json").read_text(encoding="utf-8"))
    episode_ids = set(report["split"][f"{partition}_episode_ids"])
    payload, model = _load_model(model_path)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    source_count = 0
    selected_episode_count = 0

    for episode_path in sorted(data_root.rglob("episodes.jsonl")):
        manifest_path = episode_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing cohort manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = episode_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get("episodes_sha256"):
            raise ValueError(f"Episode checksum does not match manifest: {episode_path}")
        source_count += 1
        for line in raw.splitlines():
            episode = json.loads(line)
            if episode["episode_id"] not in episode_ids:
                continue
            selected_episode_count += 1
            action = episode["inputs"]["action"]
            context_key = (
                _matched_baseline_key(episode),
                int(episode["seed"]),
                int(episode["episode_index"]),
            )
            row = {
                "episode": episode,
                "prediction": _predict(model, episode, payload),
                "baseline_key": context_key,
                "frequency_s": int(action.get("frequency_s", 0)),
                "phase_s": int(action.get("departure_phase_s", 0)),
                "retimings": action.get("retimings", []),
                "action_key": "",
            }
            shifts = [int(item["departure_shift_s"]) for item in action.get("retimings", [])]
            if not shifts:
                row["action_key"] = "none"
            elif len(shifts) == 1:
                row["action_key"] = f"one-{action['retimings'][0]['train_index']}:{shifts[0]}"
            elif (
                len(shifts) == len(episode["inputs"]["baseline_schedule_payloads"])
                and len(set(shifts)) == 1
            ):
                row["action_key"] = f"all-{shifts[0]}"
            else:
                row["action_key"] = "mixed"
            grouped[
                (
                    str(episode.get("scenario_group")),
                    int(episode["seed"]),
                    int(episode["episode_index"]),
                )
            ].append(row)

    if not grouped:
        raise ValueError("No episodes from the frozen model test split were found under data root")

    contexts: list[dict[str, Any]] = []
    pairwise: list[float] = []
    selected_exact: list[float] = []
    selected_conflicts: list[float] = []
    oracle_conflicts: list[float] = []
    selected_action_costs: list[float] = []
    oracle_action_costs: list[float] = []
    first_accepted_costs: dict[str, list[float]] = {
        "gnn_order": [],
        "minimum_edit_order": [],
        "all-450_first": [],
        "all-900_first": [],
    }
    first_accepted_actions: dict[str, Counter[str]] = {
        key: Counter() for key in first_accepted_costs
    }
    model_calls: list[int] = []
    minimal_edit_calls: list[int] = []
    accepted_contexts = 0
    candidate_count = 0
    selected_actions: Counter[str] = Counter()
    fixed_action_calls: dict[str, list[int]] = {"all-450": [], "all-900": []}
    fixed_action_costs: dict[str, list[float]] = {"all-450": [], "all-900": []}

    for group_key, rows in sorted(grouped.items()):
        # Each directory is one action alternative. A duplicated candidate for a
        # matched context would contaminate ranking and is rejected explicitly.
        signatures: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            episode = row["episode"]
            signature = json.dumps(
                {
                    "scenario": episode["scenario_group"],
                    "frequency": row["frequency_s"],
                    "phase": row["phase_s"],
                    "retimings": row["retimings"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in signatures:
                raise ValueError(f"Duplicate action alternative for matched context {group_key}")
            signatures.add(signature)
            labels = episode["labels"]
            paired = labels["paired_by_train_index"]
            baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
            success = all(
                item.get("treatment_status") == "success"
                and (
                    int(item["train_index"]) >= baseline_count
                    or item.get("baseline_status") == "success"
                )
                for item in paired
            )
            retimed_count, shift_magnitude = _action_cost(episode)
            baseline_count = len(episode["inputs"]["baseline_schedule_payloads"])
            normalized_action_cost = retimed_count / max(baseline_count, 1) + shift_magnitude / max(
                baseline_count * max_retime_s, 1
            )
            actual_conflicts = int(labels["new_conflict_count"])
            actual_delay = _actual_positive_arrival_delay_s(episode)
            prediction_risk, predicted_delay = row["prediction"]
            candidates.append(
                {
                    **row,
                    "normalized_action_cost": normalized_action_cost,
                    "actual_objective": (
                        actual_conflicts,
                        actual_delay,
                        retimed_count,
                        shift_magnitude,
                    ),
                    "predicted_objective": (
                        prediction_risk + action_cost_weight * normalized_action_cost,
                        predicted_delay,
                        retimed_count,
                        shift_magnitude,
                    ),
                    "accepted": success and actual_conflicts == 0,
                }
            )
        candidates.sort(key=lambda item: item["predicted_objective"])
        if len({item["baseline_key"] for item in candidates}) != 1:
            raise ValueError(f"Candidates are not baseline-matched for context {group_key}")
        candidate_count += len(candidates)
        selected = candidates[0]
        selected_actions[selected["action_key"]] += 1
        oracle_objective = min(item["actual_objective"] for item in candidates)
        selected_exact.append(float(selected["actual_objective"] == oracle_objective))
        selected_conflicts.append(float(selected["actual_objective"][0]))
        oracle_conflicts.append(float(oracle_objective[0]))
        selected_action_costs.append(float(selected["normalized_action_cost"]))
        oracle_action_costs.append(
            min(
                float(item["normalized_action_cost"])
                for item in candidates
                if item["actual_objective"] == oracle_objective
            )
        )
        accepted_contexts += int(any(item["accepted"] for item in candidates))

        def validations_to_accept(ordered: list[dict[str, Any]]) -> int:
            return next(
                (index for index, item in enumerate(ordered, start=1) if item["accepted"]),
                len(ordered),
            )

        def first_accepted(ordered: list[dict[str, Any]]) -> dict[str, Any] | None:
            return next((item for item in ordered if item["accepted"]), None)

        by_edits = sorted(
            candidates,
            key=lambda item: (
                len(item["retimings"]),
                sum(abs(int(row["departure_shift_s"])) for row in item["retimings"]),
                item["frequency_s"],
                item["phase_s"],
            ),
        )
        model_calls.append(validations_to_accept(candidates))
        minimal_edit_calls.append(validations_to_accept(by_edits))
        orders = {
            "gnn_order": candidates,
            "minimum_edit_order": by_edits,
        }
        for fixed_action in fixed_action_calls:
            fixed = [item for item in candidates if item["action_key"] == fixed_action]
            if fixed:
                fixed_action_costs[fixed_action].append(float(fixed[0]["normalized_action_cost"]))
            fixed_order = [
                *fixed,
                *(item for item in by_edits if item["action_key"] != fixed_action),
            ]
            fixed_action_calls[fixed_action].append(validations_to_accept(fixed_order))
            orders[f"{fixed_action}_first"] = fixed_order
        for order_name, ordered in orders.items():
            accepted = first_accepted(ordered)
            if accepted is not None:
                first_accepted_costs[order_name].append(float(accepted["normalized_action_cost"]))
                first_accepted_actions[order_name][accepted["action_key"]] += 1
        context_pairs: list[float] = []
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                actual_left = left["actual_objective"]
                actual_right = right["actual_objective"]
                predicted_left = left["predicted_objective"]
                predicted_right = right["predicted_objective"]
                score = (
                    0.5
                    if actual_left == actual_right or predicted_left == predicted_right
                    else float((actual_left < actual_right) == (predicted_left < predicted_right))
                )
                pairwise.append(score)
                context_pairs.append(score)
        contexts.append(
            {
                "scenario_group": group_key[0],
                "seed": group_key[1],
                "episode_index": group_key[2],
                "candidate_count": len(candidates),
                "pairwise_objective_accuracy_half_credit_ties": (
                    sum(context_pairs) / len(context_pairs) if context_pairs else None
                ),
                "model_selected_objective": selected["actual_objective"],
                "oracle_objective": oracle_objective,
                "model_selected_frequency_s": selected["frequency_s"],
                "model_selected_phase_s": selected["phase_s"],
                "model_selected_retimings": selected["retimings"],
                "model_selected_action_key": selected["action_key"],
                "model_selected_normalized_action_cost": selected["normalized_action_cost"],
                "model_order_validation_calls": model_calls[-1],
                "minimal_edit_validation_calls": minimal_edit_calls[-1],
                "accepted_candidate_count": sum(item["accepted"] for item in candidates),
            }
        )

    return {
        "schema": "osrd-coupled-plan-evaluation/v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "partition_episode_ids_verified": True,
        "partition": partition,
        "action_cost_weight": action_cost_weight,
        "max_retime_s_for_cost_normalization": max_retime_s,
        "matched_contexts": len(contexts),
        "data_root": str(data_root),
        "cohort_directory_count": source_count,
        "partition_episode_count": selected_episode_count,
        "candidate_episode_count": candidate_count,
        "aggregate": {
            "pairwise_objective_comparisons": len(pairwise),
            "pairwise_objective_accuracy_half_credit_ties": (
                sum(pairwise) / len(pairwise) if pairwise else None
            ),
            "exact_oracle_objective_match_rate": sum(selected_exact) / len(selected_exact),
            "mean_new_conflicts": {
                "model_selected": sum(selected_conflicts) / len(selected_conflicts),
                "oracle": sum(oracle_conflicts) / len(oracle_conflicts),
            },
            "mean_normalized_action_cost": {
                "model_selected": sum(selected_action_costs) / len(selected_action_costs),
                "oracle": sum(oracle_action_costs) / len(oracle_action_costs),
                "first_accepted_by_order": {
                    order: sum(costs) / len(costs) if costs else None
                    for order, costs in first_accepted_costs.items()
                },
                **{
                    f"fixed_{action}": sum(costs) / len(costs)
                    for action, costs in fixed_action_costs.items()
                    if costs
                },
            },
            "contexts_with_any_accepted_candidate": accepted_contexts,
            "mean_validations_to_first_accepted_candidate": {
                "gnn_order": sum(model_calls) / len(model_calls),
                "minimum_edit_order": sum(minimal_edit_calls) / len(minimal_edit_calls),
                **{
                    f"{action}_first": sum(calls) / len(calls)
                    for action, calls in fixed_action_calls.items()
                },
            },
            "gnn_selected_action_distribution": dict(sorted(selected_actions.items())),
            "first_accepted_action_distribution_by_order": {
                order: dict(sorted(counts.items()))
                for order, counts in first_accepted_actions.items()
            },
        },
        "context_results": contexts,
        "limitations": [
            "Acceptance requires successful simulations and zero newly introduced conflicts.",
            "Conflict risk is scored by averaging per-service predictions over the candidate plan.",
            "Validation-call results are retrospective because all candidates were already "
            "simulated.",
            "Swiss Actual delay and planned dwell condition OSRD replays but do not provide "
            "counterfactual conflict labels.",
            "OSRD simulation remains authoritative for final timetable acceptance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--action-cost-weight", type=float, default=0.0)
    parser.add_argument("--max-retime-s", type=int, default=900)
    args = parser.parse_args()
    result = evaluate(
        args.model,
        args.data_root,
        partition=args.partition,
        action_cost_weight=args.action_cost_weight,
        max_retime_s=args.max_retime_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
