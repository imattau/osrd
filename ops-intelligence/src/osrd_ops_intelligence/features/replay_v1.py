"""Fixed numeric features derived from paired OSRD replay v2 episodes."""

from __future__ import annotations

import math
from typing import Any

FEATURE_SCHEMA = "osrd-ops-replay-features/v1"
FEATURE_NAMES = (
    "nearest_departure_gap_s",
    "baseline_travel_time_s",
    "train_position_fraction",
    "train_count",
    "scenario_headway_s",
    "incident_delay_s",
    "is_incident_train",
    "background_departure_shift_s",
    "destination_dwell_s",
)


def rows_for_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Build pre-treatment numeric features and paired targets for one episode."""
    if episode.get("schema") != "osrd-ops-replay-episode/v2":
        raise ValueError(f"Unsupported replay schema in episode {episode.get('episode_id')}")
    inputs = episode["inputs"]
    baseline_payloads = inputs["baseline_schedule_payloads"]
    treatment_payloads = inputs["schedule_payloads"]
    pair_labels = episode["labels"]["paired_by_train_index"]
    if not (len(baseline_payloads) == len(treatment_payloads) == len(pair_labels)):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched paired rows")
    paired_outputs = episode["outputs"]["paired"]
    baseline_ids = paired_outputs["baseline"]["schedule_ids"]
    baseline_summaries = paired_outputs["baseline"]["simulation_summaries"]
    if len(baseline_ids) != len(baseline_payloads):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched baseline schedule ids")

    starts_s = [float(payload["start_time"]) / 1000 for payload in treatment_payloads]
    train_count = len(treatment_payloads)
    scenario = inputs["scenario"]
    incident = inputs["incident"]
    perturbations = {int(item["train_index"]): item for item in inputs["background_perturbations"]}
    incident_index = int(incident["train_index"])
    result: list[dict[str, Any]] = []
    for index, (baseline_id, pair) in enumerate(zip(baseline_ids, pair_labels, strict=True)):
        nearest_gap_s = min(
            (
                abs(starts_s[index] - other)
                for other_index, other in enumerate(starts_s)
                if other_index != index
            ),
            default=0.0,
        )
        summary = baseline_summaries.get(str(baseline_id), {}).get("train_schedule", {})
        travel_time_s = float(summary.get("time") or 0) / 1000
        perturbation = perturbations.get(index, {})
        features = {
            "nearest_departure_gap_s": nearest_gap_s,
            "baseline_travel_time_s": travel_time_s,
            "train_position_fraction": index / max(train_count - 1, 1),
            "train_count": float(train_count),
            "scenario_headway_s": float(scenario["headway_s"]),
            "incident_delay_s": float(incident["delay_s"]),
            "is_incident_train": float(index == incident_index),
            "background_departure_shift_s": float(
                perturbation.get("background_departure_shift_s", 0)
            ),
            "destination_dwell_s": float(perturbation.get("destination_dwell_s", 0)),
        }
        if any(not math.isfinite(value) for value in features.values()):
            raise ValueError(f"Episode {episode['episode_id']} has non-finite feature values")
        result.append(
            {
                "episode_id": episode["episode_id"],
                "scenario_group": episode.get("scenario_group", episode["episode_id"]),
                "train_index": index,
                "features": features,
                "conflict_label": int(bool(pair["treatment_conflict"])),
                "arrival_delta_ms": pair.get("arrival_delta_ms"),
            }
        )
    return result
