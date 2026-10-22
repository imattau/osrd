"""Build outcome-free replay inputs for advisory insertion-model evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from osrd_ops_intelligence.api.contracts import (
    ServicePlanCandidate,
    ServicePlanRequest,
    SnapshotBinding,
)

REPLAY_TRAINING_HEADWAY_SECONDS = 240


class ReplayEpisodeInputError(ValueError):
    """Raised when live-plan context cannot satisfy the replay-model contract."""


def build_service_plan_action(
    request: ServicePlanRequest,
    candidate: ServicePlanCandidate,
    baseline_schedule_ids: list[int],
) -> dict[str, Any]:
    """Translate a generated service plan into replay-model action features."""
    if not candidate.proposed_schedules or not candidate.inserted_departures:
        raise ReplayEpisodeInputError("Service-plan candidate has no proposed departures")
    if len(candidate.proposed_schedules) != len(candidate.inserted_departures):
        raise ReplayEpisodeInputError("Proposed schedules and inserted departures are misaligned")
    start = request.service.operating_window_start
    end = request.service.operating_window_end
    if (
        start.tzinfo is None
        or start.utcoffset() is None
        or end.tzinfo is None
        or end.utcoffset() is None
        or end <= start
    ):
        raise ReplayEpisodeInputError("Operating window must be an increasing timezone-aware range")
    if request.service.frequency_seconds <= 0:
        raise ReplayEpisodeInputError("Requested service frequency must be positive")
    departures = candidate.inserted_departures
    if any(departure.tzinfo is None or departure.utcoffset() is None for departure in departures):
        raise ReplayEpisodeInputError("Candidate departure times must include a time zone")
    departure_ms = [int(departure.timestamp() * 1000) for departure in departures]
    for index, (schedule, expected_start_ms) in enumerate(
        zip(candidate.proposed_schedules, departure_ms, strict=True)
    ):
        schedule_start_ms = schedule.get("start_time")
        if (
            not isinstance(schedule_start_ms, int)
            or isinstance(schedule_start_ms, bool)
            or schedule_start_ms != expected_start_ms
        ):
            raise ReplayEpisodeInputError(
                f"Proposed schedule {index} start time does not match its candidate departure"
            )
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    if any(value < start_ms or value >= end_ms for value in departure_ms):
        raise ReplayEpisodeInputError("Candidate departures must fall inside the requested window")
    if any(
        right - left != request.service.frequency_seconds * 1000
        for left, right in zip(departure_ms[:-1], departure_ms[1:], strict=True)
    ):
        raise ReplayEpisodeInputError("Candidate departures must follow the requested frequency")
    if not baseline_schedule_ids or any(
        not isinstance(schedule_id, int) or isinstance(schedule_id, bool)
        for schedule_id in baseline_schedule_ids
    ):
        raise ReplayEpisodeInputError("Baseline schedule IDs must be integers")
    if len(set(baseline_schedule_ids)) != len(baseline_schedule_ids):
        raise ReplayEpisodeInputError("Baseline schedule IDs must be non-empty and unique")

    baseline_index = {
        str(schedule_id): index for index, schedule_id in enumerate(baseline_schedule_ids)
    }
    retimings = []
    for edit in candidate.edits:
        if edit.train_schedule_id not in baseline_index:
            raise ReplayEpisodeInputError(
                f"Candidate edit references unknown baseline schedule {edit.train_schedule_id}"
            )
        retimings.append(
            {
                "train_index": baseline_index[edit.train_schedule_id],
                "departure_shift_s": edit.shift_seconds,
            }
        )

    proposed_path = candidate.proposed_schedules[0].get("path")
    if not isinstance(proposed_path, list):
        raise ReplayEpisodeInputError("Proposed schedule has no ordered path")
    resolved_by_key = {
        item["key"]: item["location"]
        for item in proposed_path
        if isinstance(item, dict)
        and isinstance(item.get("key"), str)
        and isinstance(item.get("location"), dict)
    }
    stops = []
    for waypoint in request.service.path[1:-1]:
        dwell = waypoint.stop_for_seconds
        if dwell is None:
            continue
        location = resolved_by_key.get(waypoint.key)
        if not isinstance(location, dict):
            raise ReplayEpisodeInputError(
                f"Could not resolve requested stop waypoint {waypoint.key} in candidate path"
            )
        stop: dict[str, Any] = {
            "track": location.get("track", ""),
            "offset_mm": location.get("offset", 0),
            "dwell_s": dwell,
        }
        operational_point = location.get("operational_point")
        if isinstance(operational_point, dict):
            operational_point_id = operational_point.get("operational_point")
            if isinstance(operational_point_id, str):
                stop["operational_point_id"] = operational_point_id
        stops.append(stop)

    route_via = []
    for waypoint in request.service.path[1:-1]:
        if waypoint.stop_for_seconds is not None:
            continue
        location = resolved_by_key.get(waypoint.key)
        if not isinstance(location, dict):
            raise ReplayEpisodeInputError(
                f"Could not resolve requested pass-through waypoint {waypoint.key}"
            )
        offset = location.get("offset")
        route_via.append(
            {
                "track": location.get("track", ""),
                "offset_mm": (
                    offset if isinstance(offset, int) and not isinstance(offset, bool) else 0
                ),
            }
        )
    dwell_values = [int(stop["dwell_s"]) for stop in stops]
    window_seconds = int(
        (
            request.service.operating_window_end - request.service.operating_window_start
        ).total_seconds()
    )
    first_departure = departures[0]
    departure_time_ms = int(first_departure.timestamp() * 1000)
    return {
        "type": "service-plan",
        "train_index": len(baseline_schedule_ids),
        "departure_shift_s": 0,
        "departure_time_ms": departure_time_ms,
        "frequency_s": request.service.frequency_seconds,
        "departure_phase_s": candidate.departure_phase_seconds,
        "operating_window_s": window_seconds,
        "requested_service_count": len(candidate.proposed_schedules),
        "stop_count": len(stops),
        "stops": stops,
        "mean_dwell_s": sum(dwell_values) / len(dwell_values) if dwell_values else 0,
        "route_via": route_via,
        "retimings": retimings,
    }


def build_service_plan_replay_episode(
    *,
    snapshot: SnapshotBinding,
    baseline_schedule_ids: list[int],
    baseline_schedule_payloads: list[dict[str, Any]],
    candidate_schedule_payloads: list[dict[str, Any]],
    baseline_simulation_summaries: dict[str, Any],
    baseline_conflicts: list[dict[str, Any]],
    resource_paths_by_train: list[dict[str, Any]],
    action: dict[str, Any],
) -> dict[str, Any]:
    """Assemble outcome-free inputs for a complete plan with retimings and insertions.

    Existing schedules must remain in source order; this adapter supports departure
    shifts and appended calendar services, but not paced schedules or removals.
    """
    baseline_count = len(baseline_schedule_payloads)
    candidate_count = len(candidate_schedule_payloads)
    if baseline_count == 0 or len(baseline_schedule_ids) != baseline_count:
        raise ReplayEpisodeInputError("Baseline IDs and schedules must be non-empty and aligned")
    if any(
        not isinstance(schedule_id, int) or isinstance(schedule_id, bool)
        for schedule_id in baseline_schedule_ids
    ) or len(set(baseline_schedule_ids)) != baseline_count:
        raise ReplayEpisodeInputError("Baseline schedule IDs must be unique integers")
    if candidate_count <= baseline_count:
        raise ReplayEpisodeInputError(
            "Service plans must preserve the baseline and append a proposed service"
        )
    if len(resource_paths_by_train) != candidate_count or any(
        not isinstance(path, dict) or not path.get("track_section_ranges")
        for path in resource_paths_by_train
    ):
        raise ReplayEpisodeInputError("Every candidate service needs an OSRD resource path")
    if not isinstance(action, dict) or action.get("type") != "service-plan":
        raise ReplayEpisodeInputError("Action must describe a complete service plan")
    retimings = action.get("retimings", [])
    if not isinstance(retimings, list):
        raise ReplayEpisodeInputError("Action retimings must be a list")
    shifts: dict[int, int] = {}
    for retiming in retimings:
        if not isinstance(retiming, dict):
            raise ReplayEpisodeInputError("Each planned retiming must be an object")
        index, shift = retiming.get("train_index"), retiming.get("departure_shift_s")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < baseline_count
            or not isinstance(shift, int)
            or isinstance(shift, bool)
            or shift == 0
            or abs(shift) > 7200
            or index in shifts
        ):
            raise ReplayEpisodeInputError(
                "Planned retimings must identify unique baseline services"
            )
        shifts[index] = shift
    for index in range(baseline_count):
        baseline = baseline_schedule_payloads[index]
        candidate = candidate_schedule_payloads[index]
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ReplayEpisodeInputError("Schedule payloads must be objects")
        if baseline.get("paced") is not None or candidate.get("paced") is not None:
            raise ReplayEpisodeInputError("Replay plan inputs do not support paced schedules")
        baseline_start = baseline.get("start_time")
        if not isinstance(baseline_start, int) or isinstance(baseline_start, bool):
            raise ReplayEpisodeInputError(
                "Baseline schedule start times must be integer milliseconds"
            )
        expected = dict(baseline)
        expected["start_time"] = baseline_start + shifts.get(index, 0) * 1000
        if candidate != expected:
            raise ReplayEpisodeInputError(
                "Candidate baseline services may only include the declared departure shifts"
            )
    if not all(isinstance(schedule, dict) for schedule in candidate_schedule_payloads):
        raise ReplayEpisodeInputError("Candidate schedule payloads must be objects")
    if any(schedule.get("paced") is not None for schedule in candidate_schedule_payloads):
        raise ReplayEpisodeInputError("Replay plan inputs do not support paced schedules")
    if not isinstance(baseline_simulation_summaries, dict) or not all(
        isinstance(conflict, dict) for conflict in baseline_conflicts
    ):
        raise ReplayEpisodeInputError("Baseline simulation and conflict context is invalid")
    if not isinstance(action.get("train_index"), int) or isinstance(
        action.get("train_index"), bool
    ) or not 0 <= action["train_index"] < candidate_count:
        raise ReplayEpisodeInputError("Action train_index is outside the candidate schedule list")
    if action["train_index"] != baseline_count:
        raise ReplayEpisodeInputError("Service-plan action must target the first appended service")
    requested_count = action.get("requested_service_count")
    if (
        not isinstance(requested_count, int)
        or isinstance(requested_count, bool)
        or requested_count != candidate_count - baseline_count
    ):
        raise ReplayEpisodeInputError(
            "Service-plan action count must match the appended proposed services"
        )

    scenario_group = f"{snapshot.infra_id}/{snapshot.timetable_id}"
    payload = {
        "schema": "osrd-ops-replay-episode/v4",
        "scenario_group": scenario_group,
        "snapshot": snapshot.model_dump(mode="json"),
        "inputs": {
            "scenario": {
                "infra_id": int(snapshot.infra_id),
                "headway_s": REPLAY_TRAINING_HEADWAY_SECONDS,
            },
            "baseline_schedule_payloads": baseline_schedule_payloads,
            "schedule_payloads": candidate_schedule_payloads,
            "resource_paths_by_train": resource_paths_by_train,
            "resource_path": resource_paths_by_train[-1],
            "background_perturbations": [],
            "incident": {
                "type": action["type"],
                "train_index": action["train_index"],
                "delay_s": 0,
            },
            "action": dict(action),
        },
        "outputs": {
            "paired": {
                "baseline": {
                    "schedule_ids": baseline_schedule_ids,
                    "simulation_summaries": baseline_simulation_summaries,
                    "conflicts": baseline_conflicts,
                }
            }
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["episode_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_insertion_replay_episode(
    *,
    snapshot: SnapshotBinding,
    baseline_schedule_ids: list[int],
    baseline_schedule_payloads: list[dict[str, Any]],
    candidate_schedule_payloads: list[dict[str, Any]],
    baseline_simulation_summaries: dict[str, Any],
    baseline_conflicts: list[dict[str, Any]],
    resource_paths_by_train: list[dict[str, Any]],
    action: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a model input only for one inserted service and an unchanged baseline.

    All OSRD-derived inputs are supplied by the caller. Treatment simulations,
    labels, and treatment timetable identifiers are deliberately excluded.
    """
    baseline_count = len(baseline_schedule_payloads)
    if baseline_count == 0 or len(baseline_schedule_ids) != baseline_count:
        raise ReplayEpisodeInputError(
            "Baseline schedule IDs and payloads must be non-empty and aligned"
        )
    if any(
        not isinstance(schedule_id, int) or isinstance(schedule_id, bool)
        for schedule_id in baseline_schedule_ids
    ):
        raise ReplayEpisodeInputError("Baseline schedule IDs must be integers")
    if len(set(baseline_schedule_ids)) != baseline_count:
        raise ReplayEpisodeInputError("Baseline schedule IDs must be unique")
    if not all(isinstance(schedule, dict) for schedule in baseline_schedule_payloads):
        raise ReplayEpisodeInputError("Baseline schedule payloads must be objects")
    if len(candidate_schedule_payloads) != baseline_count + 1:
        raise ReplayEpisodeInputError("Insertion ranking requires exactly one appended service")
    if candidate_schedule_payloads[:baseline_count] != baseline_schedule_payloads:
        raise ReplayEpisodeInputError("Candidate must preserve every baseline schedule unchanged")
    if not all(isinstance(schedule, dict) for schedule in candidate_schedule_payloads):
        raise ReplayEpisodeInputError("Candidate schedule payloads must be objects")
    if len(resource_paths_by_train) != len(candidate_schedule_payloads):
        raise ReplayEpisodeInputError(
            "Provide one OSRD resource path per baseline and inserted service"
        )
    if any(
        not isinstance(path, dict) or not path.get("track_section_ranges")
        for path in resource_paths_by_train
    ):
        raise ReplayEpisodeInputError("Every service must have an OSRD track-resource path")
    action_index = action.get("train_index") if isinstance(action, dict) else None
    if (
        not isinstance(action, dict)
        or action.get("type") != "service-insertion"
        or not isinstance(action_index, int)
        or isinstance(action_index, bool)
        or action_index != baseline_count
    ):
        raise ReplayEpisodeInputError("Action must describe the appended service insertion")
    if not isinstance(baseline_simulation_summaries, dict):
        raise ReplayEpisodeInputError("Baseline simulation summaries must be an object")
    if not all(isinstance(conflict, dict) for conflict in baseline_conflicts):
        raise ReplayEpisodeInputError("Baseline conflicts are invalid")

    scenario_group = f"{snapshot.infra_id}/{snapshot.timetable_id}"
    safe_action = dict(action)
    safe_action["train_index"] = baseline_count
    payload = {
        "schema": "osrd-ops-replay-episode/v2",
        "scenario_group": scenario_group,
        "snapshot": snapshot.model_dump(mode="json"),
        "inputs": {
            "scenario": {
                "infra_id": int(snapshot.infra_id),
                "headway_s": REPLAY_TRAINING_HEADWAY_SECONDS,
            },
            "baseline_schedule_payloads": baseline_schedule_payloads,
            "schedule_payloads": candidate_schedule_payloads,
            "resource_paths_by_train": resource_paths_by_train,
            "resource_path": resource_paths_by_train[-1],
            "background_perturbations": [],
            "incident": {"type": "service-insertion", "train_index": baseline_count, "delay_s": 0},
            "action": safe_action,
        },
        "outputs": {
            "paired": {
                "baseline": {
                    "schedule_ids": baseline_schedule_ids,
                    "simulation_summaries": baseline_simulation_summaries,
                    "conflicts": baseline_conflicts,
                }
            }
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["episode_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
