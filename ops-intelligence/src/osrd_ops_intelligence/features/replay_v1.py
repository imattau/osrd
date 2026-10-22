"""Numeric features derived from paired OSRD replay episodes with optional added trains."""

from __future__ import annotations

import math
from typing import Any

LEGACY_FEATURE_SCHEMA = "osrd-ops-replay-features/v5"
LEGACY_FEATURE_NAMES = (
    "nearest_departure_gap_s",
    "baseline_travel_time_s",
    "train_position_fraction",
    "train_count",
    "scenario_headway_s",
    "incident_delay_s",
    "is_incident_train",
    "is_added_service",
    "background_departure_shift_s",
    "destination_dwell_s",
    "path_length_m",
    "path_block_count",
    "path_route_count",
    "path_track_count",
    "has_resource_path",
    "resource_progress_fraction",
)
FEATURE_V6_SCHEMA = "osrd-ops-replay-features/v6"
FEATURE_V7_SCHEMA = "osrd-ops-replay-features/v7"
FEATURE_V8_SCHEMA = "osrd-ops-replay-features/v8"
ACTION_FEATURE_NAMES_V6 = (
    "requested_frequency_s",
    "requested_window_duration_s",
    "requested_service_count",
    "requested_stop_count",
    "requested_mean_dwell_s",
    "requested_time_of_day_sin",
    "requested_time_of_day_cos",
    "inserted_service_slot_fraction",
    "service_time_of_day_sin",
    "service_time_of_day_cos",
)
ACTION_FEATURE_NAMES_V7 = ACTION_FEATURE_NAMES_V6 + (
    "planned_departure_shift_s",
    "planned_service_edit_count",
)
ACTION_FEATURE_NAMES = ACTION_FEATURE_NAMES_V7 + ("requested_departure_phase_s",)
FEATURE_V6_NAMES = LEGACY_FEATURE_NAMES + ACTION_FEATURE_NAMES_V6
FEATURE_V7_NAMES = LEGACY_FEATURE_NAMES + ACTION_FEATURE_NAMES_V7
ACTION_FEATURE_NAMES_V8 = ACTION_FEATURE_NAMES + (
    "requested_station_stop_count",
    "requested_first_stop_route_fraction",
    "requested_last_stop_route_fraction",
    "requested_mean_stop_route_fraction",
)
ACTION_FEATURE_NAMES_V9 = ACTION_FEATURE_NAMES_V8 + (
    "requested_route_waypoint_count",
    "requested_first_route_waypoint_fraction",
    "requested_last_route_waypoint_fraction",
    "requested_mean_route_waypoint_fraction",
)
FEATURE_V8_NAMES = LEGACY_FEATURE_NAMES + ACTION_FEATURE_NAMES_V8
FEATURE_V9_NAMES = LEGACY_FEATURE_NAMES + ACTION_FEATURE_NAMES_V9
FEATURE_SCHEMA = "osrd-ops-replay-features/v10"
FEATURE_NAMES = FEATURE_V9_NAMES


def _waypoint_route_fractions(
    waypoints: list[dict[str, Any]], path: dict[str, Any]
) -> list[float]:
    ranges = path.get("track_section_ranges", [])
    lengths = [abs(float(item["end"]) - float(item["begin"])) for item in ranges]
    total_length = sum(lengths)
    if total_length <= 0:
        return []
    fractions: list[float] = []
    for waypoint in waypoints:
        track = str(waypoint.get("track", ""))
        offset_mm = float(waypoint.get("offset_mm", 0))
        distance_before_range = 0.0
        for item, segment_length in zip(ranges, lengths, strict=True):
            if str(item.get("track_section")) == track:
                begin = float(item["begin"])
                end = float(item["end"])
                if min(begin, end) <= offset_mm <= max(begin, end):
                    within_segment = abs(offset_mm - begin)
                    fractions.append(
                        min(1.0, max(0.0, (distance_before_range + within_segment) / total_length))
                    )
                    break
            distance_before_range += segment_length
    return fractions


def feature_rows_for_episode(
    episode: dict[str, Any], feature_names: tuple[str, ...] = FEATURE_NAMES
) -> list[dict[str, Any]]:
    """Build pre-treatment feature rows without requiring treatment labels or outcomes.

    This is the inference interface: the candidate schedule and baseline replay are
    available, while treatment simulation results and paired labels may not exist yet.
    """
    if episode.get("schema") not in (
        "osrd-ops-replay-episode/v2",
        "osrd-ops-replay-episode/v3",
        "osrd-ops-replay-episode/v4",
    ):
        raise ValueError(f"Unsupported replay schema in episode {episode.get('episode_id')}")
    inputs = episode["inputs"]
    baseline_payloads = inputs["baseline_schedule_payloads"]
    treatment_payloads = inputs["schedule_payloads"]
    paired_outputs = episode["outputs"]["paired"]
    baseline_ids = paired_outputs["baseline"]["schedule_ids"]
    baseline_summaries = paired_outputs["baseline"]["simulation_summaries"]
    if len(baseline_ids) != len(baseline_payloads):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched baseline schedule ids")

    starts_s = [float(payload["start_time"]) / 1000 for payload in treatment_payloads]
    train_count = len(treatment_payloads)
    scenario = inputs["scenario"]
    incident = inputs["incident"]
    action = inputs.get("action", incident)
    perturbations = {int(item["train_index"]): item for item in inputs["background_perturbations"]}
    incident_index = int(action["train_index"])
    added_count = max(0, train_count - len(baseline_payloads))
    time_of_day = float(action.get("departure_time_ms") or 0) / 1000 % 86400
    stop_count = float(action.get("stop_count", 0))
    requested_frequency = float(action.get("frequency_s", 0))
    requested_window = float(action.get("operating_window_s", 0))
    requested_mean_dwell = float(action.get("mean_dwell_s", 0))
    requested_departure_phase = float(action.get("departure_phase_s", 0))
    requested_service_count = float(action.get("requested_service_count", added_count))
    resource_path = inputs.get("resource_path") or {}
    stop_fractions = _waypoint_route_fractions(action.get("stops", []), resource_path)
    route_waypoint_fractions = _waypoint_route_fractions(
        action.get("route_via", []), resource_path
    )
    planned_shifts = {
        int(item["train_index"]): int(item["departure_shift_s"])
        for item in action.get("retimings", [])
    }
    planned_edit_count = float(sum(shift != 0 for shift in planned_shifts.values()))
    result: list[dict[str, Any]] = []
    for index, _treatment_payload in enumerate(treatment_payloads):
        paths_by_train = inputs.get("resource_paths_by_train") or []
        resource_path = (
            paths_by_train[index]
            if index < len(paths_by_train)
            else inputs.get("resource_path") or {}
        )
        baseline_id = baseline_ids[index] if index < len(baseline_ids) else None
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
            "incident_delay_s": float(action.get("departure_shift_s", incident.get("delay_s", 0))),
            "is_incident_train": float(index == incident_index),
            "is_added_service": float(baseline_id is None),
            "background_departure_shift_s": float(
                perturbation.get("background_departure_shift_s", 0)
            ),
            "destination_dwell_s": float(perturbation.get("destination_dwell_s", 0)),
            "path_length_m": float(resource_path.get("length", 0)) / 1000,
            "path_block_count": float(len(resource_path.get("blocks", []))),
            "path_route_count": float(len(resource_path.get("routes", []))),
            "path_track_count": float(len(resource_path.get("track_section_ranges", []))),
            "has_resource_path": float(bool(resource_path.get("track_section_ranges"))),
            "resource_progress_fraction": 0.0,
            "requested_frequency_s": requested_frequency,
            "requested_window_duration_s": requested_window,
            "requested_service_count": requested_service_count,
            "requested_stop_count": stop_count,
            "requested_mean_dwell_s": requested_mean_dwell,
            "requested_departure_phase_s": requested_departure_phase,
            "requested_time_of_day_sin": math.sin(2 * math.pi * time_of_day / 86400),
            "requested_time_of_day_cos": math.cos(2 * math.pi * time_of_day / 86400),
            "inserted_service_slot_fraction": (
                (index - len(baseline_payloads)) / max(added_count - 1, 1)
                if index >= len(baseline_payloads)
                else 0.0
            ),
            "service_time_of_day_sin": math.sin(2 * math.pi * (starts_s[index] % 86400) / 86400),
            "service_time_of_day_cos": math.cos(2 * math.pi * (starts_s[index] % 86400) / 86400),
            "planned_departure_shift_s": float(planned_shifts.get(index, 0)),
            "planned_service_edit_count": planned_edit_count,
            "requested_station_stop_count": float(
                sum(
                    stop.get("operational_point_id") is not None for stop in action.get("stops", [])
                )
            ),
            "requested_first_stop_route_fraction": min(stop_fractions, default=0.0),
            "requested_last_stop_route_fraction": max(stop_fractions, default=0.0),
            "requested_mean_stop_route_fraction": (
                sum(stop_fractions) / len(stop_fractions) if stop_fractions else 0.0
            ),
            "requested_route_waypoint_count": float(len(action.get("route_via", []))),
            "requested_first_route_waypoint_fraction": (
                route_waypoint_fractions[0] if route_waypoint_fractions else 0.0
            ),
            "requested_last_route_waypoint_fraction": (
                route_waypoint_fractions[-1] if route_waypoint_fractions else 0.0
            ),
            "requested_mean_route_waypoint_fraction": (
                sum(route_waypoint_fractions) / len(route_waypoint_fractions)
                if route_waypoint_fractions
                else 0.0
            ),
        }
        if any(not math.isfinite(value) for value in features.values()):
            raise ValueError(f"Episode {episode['episode_id']} has non-finite feature values")
        result.append(
            {
                "episode_id": episode["episode_id"],
                "scenario_group": episode.get("scenario_group", episode["episode_id"]),
                "train_index": index,
                "features": {name: features[name] for name in feature_names},
            }
        )
    return result


def rows_for_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Build feature rows and paired targets for training/evaluation."""
    features = feature_rows_for_episode(episode)
    pair_labels = episode["labels"]["paired_by_train_index"]
    if len(features) != len(pair_labels):
        raise ValueError(f"Episode {episode['episode_id']} has mismatched treatment rows")
    pairs_by_index = {int(pair["train_index"]): pair for pair in pair_labels}
    result = []
    for row in features:
        pair = pairs_by_index[row["train_index"]]
        result.append(
            {
                **row,
                "conflict_label": int(bool(pair["new_conflict"])),
                "arrival_delta_ms": pair.get("arrival_delta_ms"),
            }
        )
    return result
