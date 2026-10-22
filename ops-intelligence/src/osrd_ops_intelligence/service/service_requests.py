"""Build and validate bounded frequency-service requests against OSRD."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, TypeVar, cast
from zoneinfo import ZoneInfo

from osrd_ops_intelligence.api.contracts import (
    CompleteTimetableCandidate,
    CompleteTimetablePatternResult,
    CompleteTimetablePlanRequest,
    CompleteTimetablePlanResponse,
    DemandCoverage,
    ExistingServiceDwellEdit,
    ExistingServiceDwellOptions,
    ExistingServiceRetimingPlanRequest,
    ExistingServiceRetimingPlanResponse,
    ExistingServiceTiming,
    HourlyStationCallTiming,
    HourlyServicePlanCandidate,
    HourlyServicePlanResponse,
    HourlyServiceRequest,
    InsertionRankingCandidate,
    InsertionRankingRequest,
    PatronageProfile,
    PlanCandidateEdit,
    PlanCandidateValidation,
    ServiceDepartureTiming,
    ServicePathWaypoint,
    ServicePlanCandidate,
    ServicePlanRequest,
    ServicePlanResponse,
    ServiceRequest,
    ServiceRequestValidationResponse,
    StationCallTiming,
)
from osrd_ops_intelligence.optimizer.departure_shifts import (
    DepartureShiftCandidate,
    generate_service_retiming_candidates,
)
from osrd_ops_intelligence.service.insertion_ranking import rank_insertion_candidates
from osrd_ops_intelligence.service.osrd_client import (
    CandidateValidation,
    OSRDClient,
    ReplayBaselineContext,
    SnapshotRef,
    UnsupportedSnapshot,
)
from osrd_ops_intelligence.service.patronage import score_station_call_coverage
from osrd_ops_intelligence.service.replay_episode import (
    build_insertion_replay_episode,
    build_service_plan_action,
    build_service_plan_replay_episode,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _sample_evenly_spaced(values: list[T], count: int) -> list[T]:
    """Select deterministic representatives across an ordered candidate dimension."""
    if not values or count < 1:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[0]]
    last_index = len(values) - 1
    return [
        values[int(index * last_index / (count - 1) + 0.5)]
        for index in range(count)
    ]


def _complete_plan_stop_options(
    service: ServiceRequest, limit: int
) -> list[tuple[tuple[str, ...], dict[str, int]]]:
    """Sample bounded optional-stop and dwell-time combinations for one service."""
    dimensions: list[tuple[ServicePathWaypoint, list[int | None]]] = []
    for waypoint in service.path[1:-1]:
        dwell_values = waypoint.dwell_options_seconds or (
            [waypoint.stop_for_seconds]
            if waypoint.stop_for_seconds is not None and waypoint.stop_for_seconds > 0
            else []
        )
        if waypoint.optional_stop:
            choices: list[int | None] = [None, *dwell_values]
        elif waypoint.dwell_options_seconds:
            choices = list(waypoint.dwell_options_seconds)
        elif dwell_values:
            choices = dwell_values
        else:
            continue
        dimensions.append((waypoint, choices))

    combination_count = math.prod(len(choices) for _, choices in dimensions)
    if not dimensions:
        return [((), {})]
    indices = (
        list(range(combination_count))
        if combination_count <= limit
        else [
            int(index * (combination_count - 1) / (limit - 1) + 0.5)
            for index in range(limit)
        ]
        if limit > 1
        else [0]
    )
    results: list[tuple[tuple[str, ...], dict[str, int]]] = []
    for index in indices:
        remaining = index
        decisions: list[int | None] = []
        for _, choices in reversed(dimensions):
            remaining, digit = divmod(remaining, len(choices))
            decisions.append(choices[digit])
        decisions.reverse()
        selected_optional: list[str] = []
        dwell_by_key: dict[str, int] = {}
        for (waypoint, _), dwell in zip(dimensions, decisions, strict=True):
            if dwell is None:
                continue
            dwell_by_key[waypoint.key] = dwell
            if waypoint.optional_stop:
                selected_optional.append(waypoint.key)
        results.append((tuple(selected_optional), dwell_by_key))
    return results


def _duration(seconds: int) -> str:
    return f"PT{seconds}S"


def _stop_for_seconds(value: object) -> int:
    if value is None or value in ("P0D", "PT0S"):
        return 0
    if not isinstance(value, str):
        raise ValueError("Existing service stop dwell must be an ISO-8601 duration")
    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        value,
    )
    if match is None or not any(match.groups()):
        raise ValueError("Existing service stop dwell must use whole ISO-8601 seconds")
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds > 3600:
        raise ValueError("Existing service stop dwell exceeds the supported one-hour limit")
    return total_seconds


def _station_call_timings(
    schedule: dict[str, Any],
    *,
    start_time_ms: int,
    path_item_times_ms: list[int] | None,
) -> list[StationCallTiming] | None:
    """Map successful OSRD final path-item times back to explicit station calls."""
    schedule_items = schedule.get("schedule")
    path = schedule.get("path")
    if (
        not isinstance(schedule_items, list)
        or not isinstance(path, list)
        or path_item_times_ms is None
        or len(schedule_items) != len(path_item_times_ms)
    ):
        return None
    path_by_key = {
        str(item["key"]): item
        for item in path
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    train_name = str(schedule.get("train_name", ""))
    timings: list[StationCallTiming] = []
    for index, (schedule_item, offset_ms) in enumerate(
        zip(schedule_items, path_item_times_ms, strict=True)
    ):
        if not isinstance(schedule_item, dict):
            return None
        key = schedule_item.get("at")
        waypoint = path_by_key.get(str(key))
        if waypoint is None:
            return None
        stop_for = schedule_item.get("stop_for")
        is_endpoint = index == 0 or index == len(schedule_items) - 1
        is_scheduled_stop = isinstance(stop_for, str) and stop_for != "PT0S"
        if not is_endpoint and not is_scheduled_stop:
            continue
        location = waypoint.get("location")
        operational_point_id: str | None = None
        if (
            isinstance(location, dict)
            and location.get("type") == "operational_point_part_reference"
        ):
            point_ref = location.get("operational_point")
            if isinstance(point_ref, dict) and point_ref.get("type") == "id":
                point_id = point_ref.get("operational_point")
                if isinstance(point_id, (str, int)) and not isinstance(point_id, bool):
                    operational_point_id = str(point_id)
        dwell_match = re.fullmatch(r"PT(\d+)S", str(stop_for or "PT0S"))
        if stop_for is not None and dwell_match is None:
            return None
        dwell_seconds = int(dwell_match.group(1)) if dwell_match is not None else 0
        arrival_time = datetime.fromtimestamp(
            (start_time_ms + offset_ms) / 1000,
            tz=UTC,
        )
        timings.append(
            StationCallTiming(
                train_name=train_name,
                operational_point_id=operational_point_id,
                arrival_time=arrival_time,
                departure_time=arrival_time + timedelta(seconds=dwell_seconds),
                boarding_allowed=index < len(schedule_items) - 1,
                alighting_allowed=index > 0,
            )
        )
    return timings


def _candidate_patronage_coverage(
    profile: PatronageProfile | None,
    schedules: list[dict[str, Any]],
    path_times_by_train: dict[str, list[int]],
) -> tuple[list[StationCallTiming], DemandCoverage | None, str]:
    if profile is None:
        return [], None, "not_configured"
    calls: list[StationCallTiming] = []
    for schedule in schedules:
        train_name = str(schedule.get("train_name", ""))
        start_time_ms = schedule.get("start_time")
        if not isinstance(start_time_ms, int) or isinstance(start_time_ms, bool):
            return [], None, "simulation_times_unavailable"
        timings = _station_call_timings(
            schedule,
            start_time_ms=start_time_ms,
            path_item_times_ms=path_times_by_train.get(train_name),
        )
        if timings is None:
            return [], None, "simulation_times_unavailable"
        calls.extend(timings)
    coverage = score_station_call_coverage(profile, calls)
    if calls and coverage.unmapped_station_call_count == len(calls):
        return calls, None, "mapping_unavailable"
    return calls, coverage, "available"


def _hourly_patronage_coverage(
    request: HourlyServiceRequest,
    schedule: dict[str, Any],
    path_item_times_ms: list[int] | None,
    *,
    phase_seconds: int,
) -> tuple[list[HourlyStationCallTiming], DemandCoverage | None, str]:
    """Map one simulated paced-service path onto declared local weekdays and cycle time."""
    profile = request.patronage_profile
    if profile is None:
        return [], None, "not_configured"
    if path_item_times_ms is None:
        return [], None, "simulation_times_unavailable"
    zone = ZoneInfo(profile.timezone)
    recurring_count = (request.time_window_seconds + request.frequency_seconds - 1) // request.frequency_seconds
    week_monday = date(2026, 1, 5)
    calls: list[StationCallTiming] = []
    for weekday in request.operating_weekdays:
        cycle_start = datetime(
            week_monday.year, week_monday.month, week_monday.day, tzinfo=zone
        ) + timedelta(days=weekday, minutes=request.cycle_start_minute)
        for departure_index in range(recurring_count):
            departure = cycle_start + timedelta(
                seconds=phase_seconds + departure_index * request.frequency_seconds
            )
            timings = _station_call_timings(
                schedule,
                start_time_ms=int(departure.timestamp() * 1000),
                path_item_times_ms=path_item_times_ms,
            )
            if timings is None:
                return [], None, "simulation_times_unavailable"
            calls.extend(timings)
    if not calls:
        return [], None, "simulation_times_unavailable"
    coverage = score_station_call_coverage(profile, calls)
    if coverage.unmapped_station_call_count == len(calls):
        return [], None, "mapping_unavailable"
    local_calls = [
        HourlyStationCallTiming(
            train_name=call.train_name,
            operational_point_id=call.operational_point_id,
            arrival_weekday=call.arrival_time.astimezone(zone).weekday(),
            arrival_minute=(
                call.arrival_time.astimezone(zone).hour * 60
                + call.arrival_time.astimezone(zone).minute
            ),
            departure_weekday=call.departure_time.astimezone(zone).weekday(),
            departure_minute=(
                call.departure_time.astimezone(zone).hour * 60
                + call.departure_time.astimezone(zone).minute
            ),
            boarding_allowed=call.boarding_allowed,
            alighting_allowed=call.alighting_allowed,
        )
        for call in calls
    ]
    return local_calls, coverage, "available"


def _conflict_fingerprint(signatures: tuple[str, ...]) -> str:
    """Hash the sorted canonical conflict multiset for compact exact comparisons."""
    signature_hashes = sorted(
        hashlib.sha256(signature.encode("utf-8")).hexdigest() for signature in signatures
    )
    payload = json.dumps(signature_hashes, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _explain_service_candidate(
    *,
    candidate: DepartureShiftCandidate,
    request: ServicePlanRequest,
    phase_seconds: int,
    route_index: int,
    departure_count: int,
    feasible: bool,
    pathfinding_valid: bool,
    simulation_valid: bool,
    new_conflict_count: int,
    failure_reasons: list[str],
) -> str:
    """Summarize measurable service-plan trade-offs without inventing benefits."""
    retiming = request.retiming
    objective = {
        "minimize_changed_services": "fewest existing services changed",
        "minimize_simulated_runtime_change": (
            "lowest net simulated running-time change for retimed services"
        ),
        "minimize_weighted_shift": "lowest weighted retiming cost",
        "minimize_total_shift": "least total shift to existing services",
    }[retiming.objective]
    if candidate.edits:
        edits = ", ".join(
            f"{edit.train_schedule_id} {edit.shift_seconds:+d}s" for edit in candidate.edits
        )
        existing = (
            f"Retimes {len(candidate.edits)} existing service(s) ({edits}; "
            f"{candidate.total_shift_seconds}s cumulative absolute shift)."
        )
    else:
        existing = "Leaves existing service departure times unchanged."
    weighted_score = _retiming_priority_score(candidate, request)
    if weighted_score is not None:
        existing += f" Priority-weighted retiming cost: {weighted_score:g}."

    phase = (
        "at the requested window start"
        if phase_seconds == 0
        else f"{phase_seconds}s after the requested window start"
    )
    route = (
        "the requested route"
        if route_index == 0
        else f"route alternative {route_index + 1}"
    )
    if feasible:
        outcome = "OSRD pathfinding and simulation passed, with no new timetable conflicts."
    else:
        outcome = (
            "OSRD checks: "
            f"pathfinding {'passed' if pathfinding_valid else 'failed'}, "
            f"simulation {'passed' if simulation_valid else 'failed'}, "
            f"{new_conflict_count} new conflict(s)."
        )
    details = [reason for reason in failure_reasons if reason]
    if details:
        outcome += " " + " ".join(details)
    return (
        f"Adds {departure_count} requested departure(s) using {route}, first departure "
        f"{phase}. {existing} Ranked by feasibility and new conflicts, then to prioritize "
        f"{objective}. {outcome}"
    )


def _retiming_priority_score(
    candidate: DepartureShiftCandidate,
    request: ServicePlanRequest | HourlyServiceRequest | CompleteTimetablePlanRequest,
) -> float | None:
    if request.retiming.objective != "minimize_weighted_shift":
        return None
    weights = request.retiming.schedule_priority_weights
    return float(
        sum(abs(edit.shift_seconds) * weights[edit.train_schedule_id] for edit in candidate.edits)
    )


def build_service_schedules(
    request: ServiceRequest,
    *,
    departure_phase_seconds: int = 0,
    path: list[ServicePathWaypoint] | None = None,
) -> list[dict[str, object]]:
    """Expand each operating window and phase into concrete calendar departures."""
    departures = _service_departure_times(request, departure_phase_seconds)
    if not 1 <= len(departures) <= 50:
        raise ValueError("Operating windows and frequency must produce between 1 and 50 services")
    service_path = path or request.path
    serialized_path = [
        {"key": waypoint.key, "location": waypoint.location} for waypoint in service_path
    ]
    schedule = [
        {
            "at": waypoint.key,
            **(
                {"stop_for": _duration(waypoint.stop_for_seconds)}
                if waypoint.stop_for_seconds is not None
                else {}
            ),
        }
        for waypoint in service_path
    ]
    return [
        {
            "train_name": f"{request.train_name}-{index + 1:02d}",
            "rolling_stock_name": request.rolling_stock_name,
            "start_time": int(departure.timestamp() * 1000),
            "comfort": "STANDARD",
            "constraint_distribution": "STANDARD",
            "initial_speed": 0,
            "labels": ["osrd-ops-service-request", str(request.request_id)],
            "options": {"use_electrical_profiles": False},
            "path": serialized_path,
            "power_restrictions": [],
            "schedule": schedule,
        }
        for index, departure in enumerate(departures)
    ]


def _service_departure_times(
    request: ServiceRequest, departure_phase_seconds: int = 0
) -> list[datetime]:
    minimum_window = request.minimum_operating_window_seconds
    if not 0 <= departure_phase_seconds < min(request.frequency_seconds, minimum_window):
        raise ValueError("Departure phase must be below the frequency and each operating window")
    departures: list[datetime] = []
    for start, end in request.operating_windows():
        departure = start + timedelta(seconds=departure_phase_seconds)
        while departure < end:
            departures.append(departure)
            departure += timedelta(seconds=request.frequency_seconds)
    if not departures:
        raise ValueError("Operating windows and frequency must produce at least one service")
    if len(departures) > 50:
        raise ValueError("Operating windows and frequency must produce at most 50 services")
    return departures


def build_hourly_service_schedule(
    request: ServiceRequest | HourlyServiceRequest,
    *,
    time_window_seconds: int,
    start_offset_seconds: int = 0,
    path: list[ServicePathWaypoint] | None = None,
) -> dict[str, object]:
    """Build one OSRD paced schedule using cycle-relative milliseconds.

    Hourly timetable start times are offsets within the recurring interval, not
    Unix timestamps. Keep this representation separate from calendar expansion.
    """
    interval_seconds = request.frequency_seconds
    if not 1 <= interval_seconds <= time_window_seconds <= 86_400:
        raise ValueError("Hourly interval and time window must be ordered within 24 hours")
    if not 0 <= start_offset_seconds < interval_seconds:
        raise ValueError("Hourly start offset must be within the recurring interval")

    service_path = path or request.path
    serialized_path = [
        {"key": waypoint.key, "location": waypoint.location} for waypoint in service_path
    ]
    schedule = [
        {
            "at": waypoint.key,
            **(
                {"stop_for": _duration(waypoint.stop_for_seconds)}
                if waypoint.stop_for_seconds is not None
                else {}
            ),
        }
        for waypoint in service_path
    ]
    return {
        "train_name": request.train_name,
        "rolling_stock_name": request.rolling_stock_name,
        "start_time": start_offset_seconds * 1000,
        "comfort": "STANDARD",
        "constraint_distribution": "STANDARD",
        "initial_speed": 0,
        "labels": ["osrd-ops-service-request", str(request.request_id)],
        "options": {"use_electrical_profiles": False},
        "path": serialized_path,
        "power_restrictions": [],
        "schedule": schedule,
        "paced": {
            "interval": _duration(interval_seconds),
            "time_window": _duration(time_window_seconds),
            "exceptions": [],
        },
    }


async def plan_hourly_service_request(
    request: HourlyServiceRequest,
    client: OSRDClient,
) -> HourlyServicePlanResponse:
    """Evaluate recurring phase options for an OSRD HOURLY timetable."""
    snapshot = SnapshotRef(**request.snapshot.model_dump())
    metadata = await client.get_schedule_metadata(snapshot=snapshot)
    if metadata.timetable_type != "HOURLY":
        raise UnsupportedSnapshot("Hourly service planning requires an HOURLY timetable")
    policy = request.retiming
    if (
        policy.objective == "minimize_simulated_runtime_change"
        and policy.max_changed_services > 0
    ):
        unsupported = sorted(
            set(policy.eligible_train_schedule_ids) & metadata.runtime_objective_unsupported_ids
        )
        if unsupported:
            raise ValueError(
                "The simulated running-time objective cannot compare paced services with "
                "timing-affecting exceptions. Choose another objective or remove these "
                f"schedule IDs from the eligible set: {', '.join(unsupported)}"
            )
    shift_caps = {
        schedule_id: min(
            policy.max_shift_seconds,
            policy.max_shift_seconds_by_schedule.get(schedule_id, policy.max_shift_seconds),
            metadata.max_hourly_shift_seconds_by_schedule.get(schedule_id, 0),
        )
        for schedule_id in policy.eligible_train_schedule_ids
    }
    advance_caps = {
        schedule_id: min(
            policy.max_shift_seconds,
            policy.max_shift_seconds_by_schedule.get(
                schedule_id, policy.max_shift_seconds
            ),
            policy.max_advance_seconds_by_schedule.get(
                schedule_id, policy.max_shift_seconds
            ),
            metadata.max_hourly_advance_seconds_by_schedule.get(schedule_id, 0),
        )
        for schedule_id in policy.eligible_train_schedule_ids
    }
    retiming_candidates = generate_service_retiming_candidates(
        service_request_id=str(request.request_id),
        snapshot=request.snapshot,
        schedule_start_times_ms=metadata.start_times_ms,
        eligible_schedule_ids=policy.eligible_train_schedule_ids,
        protected_schedule_ids=policy.protected_train_schedule_ids,
        max_edits=policy.max_changed_services,
        max_shift_seconds=policy.max_shift_seconds,
        max_shift_seconds_by_schedule=shift_caps,
        max_advance_seconds_by_schedule=advance_caps,
        max_candidates=policy.max_candidates,
        step_seconds=policy.step_seconds,
    )
    if request.max_route_alternatives == 1:
        route_options = [(request.path, [])]
    else:
        legs: list[list[dict[str, Any]]] = []
        for origin, destination in zip(request.path, request.path[1:], strict=False):
            legs.append(
                await client.get_route_alternatives(
                    snapshot=snapshot,
                    starting=origin.location,
                    ending=destination.location,
                    limit=request.max_route_alternatives,
                )
            )
        route_indices = _k_best_route_indices(legs, request.max_route_alternatives)
        if not route_indices:
            raise UnsupportedSnapshot("OSRD found no route alternatives for the hourly service")
        route_options = [_route_waypoints(request, legs, indices) for indices in route_indices]
    phases = list(range(0, request.frequency_seconds, request.phase_step_seconds))
    has_retiming_candidates = any(candidate.edits for candidate in retiming_candidates)
    phase_budget = (
        max(1, policy.max_candidates // 2)
        if has_retiming_candidates
        else policy.max_candidates
    )
    phases = _sample_evenly_spaced(phases, min(request.max_phase_options, phase_budget))
    if not phases:
        raise ValueError("Hourly service planning produced no valid phase options")

    route_phase_options = [
        (route_index, phase)
        for phase in phases
        for route_index in range(len(route_options))
    ]
    baseline_route_phases = _sample_evenly_spaced(route_phase_options, phase_budget)
    plans = [
        (route_index, phase, retiming_candidates[0])
        for route_index, phase in baseline_route_phases
    ]
    for retiming_candidate in retiming_candidates[1:]:
        for route_index, phase in route_phase_options:
            if len(plans) >= policy.max_candidates:
                break
            plans.append((route_index, phase, retiming_candidate))
        if len(plans) >= policy.max_candidates:
            break

    baseline_travel_times_ms = (
        await client.get_schedule_travel_times(
            snapshot=snapshot,
            schedule_ids=policy.eligible_train_schedule_ids,
        )
        if policy.max_changed_services > 0
        and policy.eligible_train_schedule_ids
        else {}
    )
    objective_text = {
        "minimize_total_shift": "least total shift to existing services",
        "minimize_changed_services": "fewest existing services changed",
        "minimize_simulated_runtime_change": "lowest net simulated running-time change",
        "minimize_weighted_shift": "lowest weighted retiming cost",
    }[policy.objective]

    results: list[HourlyServicePlanCandidate] = []
    baseline_seen: tuple[str, ...] | None = None
    for route_index, phase, retiming_candidate in plans:
        route_path, route_track_sections = route_options[route_index]
        schedule = build_hourly_service_schedule(
            request,
            time_window_seconds=request.time_window_seconds,
            start_offset_seconds=phase,
            path=route_path,
        )
        identity = (
            f"{request.request_id}-hourly-route-{route_index}-phase-{phase}-"
            f"{retiming_candidate.candidate_id}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        candidate_id = f"hourly-service-plan-{digest}"
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload={
                "candidate_id": candidate_id,
                "snapshot": asdict(snapshot),
                "edits": [asdict(edit) for edit in retiming_candidate.edits],
                "insertions": [schedule],
            },
        )
        baseline_signatures = Counter(validation.baseline_conflict_signatures)
        candidate_signatures = Counter(validation.conflict_signatures)
        new_conflicts = sum((candidate_signatures - baseline_signatures).values())
        baseline_count = len(validation.baseline_conflict_signatures)
        canonical_baseline = tuple(sorted(validation.baseline_conflict_signatures))
        if baseline_seen is not None and canonical_baseline != baseline_seen:
            raise UnsupportedSnapshot(
                "Baseline conflicts changed during hourly candidate validation"
            )
        baseline_seen = canonical_baseline
        feasible = (
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflicts == 0
            and not any("cleanup failed" in reason.lower() for reason in validation.failure_reasons)
        )
        runtime_ms = validation.inserted_travel_times_ms.get(request.train_name)
        station_call_timings, patronage_coverage, patronage_coverage_status = (
            _hourly_patronage_coverage(
                request,
                schedule,
                validation.inserted_path_item_times_ms.get(request.train_name),
                phase_seconds=phase,
            )
        )
        edit_records = [
            PlanCandidateEdit(
                train_schedule_id=edit.train_schedule_id,
                train_name=metadata.train_names.get(edit.train_schedule_id),
                before_start_time_ms=edit.before_start_time_ms,
                after_start_time_ms=edit.after_start_time_ms,
                shift_seconds=edit.shift_seconds,
            )
            for edit in retiming_candidate.edits
        ]
        existing_runtime_deltas = [
            validation.existing_travel_times_ms.get(edit.train_schedule_id)
            - baseline_travel_times_ms[edit.train_schedule_id]
            for edit in retiming_candidate.edits
            if edit.train_schedule_id in validation.existing_travel_times_ms
            and edit.train_schedule_id in baseline_travel_times_ms
        ]
        existing_service_timings: list[ExistingServiceTiming] = []
        unavailable_retimed_service_count = 0
        for edit in retiming_candidate.edits:
            candidate_time_ms = validation.existing_travel_times_ms.get(edit.train_schedule_id)
            baseline_time_ms = baseline_travel_times_ms.get(edit.train_schedule_id)
            source_start_ms = metadata.start_times_ms.get(edit.train_schedule_id)
            if candidate_time_ms is None or baseline_time_ms is None or source_start_ms is None:
                unavailable_retimed_service_count += 1
                continue
            existing_service_timings.append(
                ExistingServiceTiming(
                    train_schedule_id=edit.train_schedule_id,
                    train_name=metadata.train_names.get(edit.train_schedule_id),
                    departure_time=datetime.fromtimestamp(
                        edit.after_start_time_ms / 1000, tz=UTC
                    ),
                    arrival_time=datetime.fromtimestamp(
                        (edit.after_start_time_ms + candidate_time_ms) / 1000, tz=UTC
                    ),
                    travel_time_seconds=round(candidate_time_ms / 1000),
                    baseline_travel_time_seconds=round(baseline_time_ms / 1000),
                    travel_time_delta_seconds=round(
                        (candidate_time_ms - baseline_time_ms) / 1000
                    ),
                    arrival_time_delta_seconds=round(
                        (
                            edit.after_start_time_ms
                            + candidate_time_ms
                            - source_start_ms
                            - baseline_time_ms
                        )
                        / 1000
                    ),
                    retimed=True,
                )
            )
        runtime_change = (
            round(sum(existing_runtime_deltas) / 1000)
            if len(existing_runtime_deltas) == len(retiming_candidate.edits)
            and retiming_candidate.edits
            else (0 if not retiming_candidate.edits else None)
        )
        failure_text = " ".join(validation.failure_reasons)
        if feasible:
            outcome = "OSRD pathfinding and simulation passed, with no new timetable conflicts."
        else:
            outcome = (
                "OSRD checks: "
                f"pathfinding {'passed' if validation.pathfinding_valid else 'failed'}, "
                f"simulation {'passed' if validation.simulation_valid else 'failed'}, "
                f"{new_conflicts} new conflict(s). {failure_text}"
            ).strip()
        recurring_count = (
            request.time_window_seconds + request.frequency_seconds - 1
        ) // request.frequency_seconds
        results.append(
            HourlyServicePlanCandidate(
                candidate_id=candidate_id,
                rank=1,
                route_alternative_index=route_index,
                route_track_sections=route_track_sections,
                departure_offset_seconds=phase,
                recurring_departure_count=recurring_count,
                proposed_schedule=schedule,
                edits=edit_records,
                changed_service_count=len(edit_records),
                total_shift_seconds=retiming_candidate.total_shift_seconds,
                simulated_runtime_change_seconds=runtime_change,
                retiming_priority_score=_retiming_priority_score(retiming_candidate, request),
                simulated_travel_time_seconds=(
                    round(runtime_ms / 1000) if runtime_ms is not None else None
                ),
                existing_service_timings=existing_service_timings,
                existing_service_effects_status=(
                    "not_applicable"
                    if not retiming_candidate.edits
                    else "partial"
                    if unavailable_retimed_service_count
                    else "complete"
                ),
                unavailable_retimed_service_count=unavailable_retimed_service_count,
                station_call_timings=station_call_timings,
                patronage_coverage_status=patronage_coverage_status,
                patronage_coverage=patronage_coverage,
                baseline_conflict_count=baseline_count,
                new_conflict_count=new_conflicts,
                validation=PlanCandidateValidation(
                    pathfinding_valid=validation.pathfinding_valid,
                    simulation_valid=validation.simulation_valid,
                    conflict_count=validation.conflict_count,
                    baseline_conflict_count=baseline_count,
                    new_conflict_count=new_conflicts,
                    failure_reasons=list(validation.failure_reasons),
                    conflicting_existing_service_ids=list(
                        validation.conflicting_existing_service_ids
                    ),
                    conflict_fingerprint=_conflict_fingerprint(validation.conflict_signatures),
                    baseline_conflict_fingerprint=_conflict_fingerprint(
                        validation.baseline_conflict_signatures
                    ),
                ),
                feasible=feasible,
                explanation=(
                    f"Adds a recurring service every {request.frequency_seconds}s for "
                    f"{request.time_window_seconds}s ({recurring_count} departures), "
                    f"starting at offset {phase}s in the cycle. "
                    + (
                        f"Retimes {len(edit_records)} existing service(s), with "
                        f"{retiming_candidate.total_shift_seconds}s cumulative absolute shift. "
                        if edit_records
                        else "Leaves existing service departure offsets unchanged. "
                    )
                    + (
                        "Ranked by feasibility and new conflicts, then to prioritize "
                        f"{objective_text}. "
                    )
                    + (
                        f"Priority-weighted retiming cost: "
                        f"{_retiming_priority_score(retiming_candidate, request):g}. "
                        if _retiming_priority_score(retiming_candidate, request) is not None
                        else ""
                    )
                    + (
                        "After OSRD feasibility and conflict checks, this option is ordered "
                        "by the matched share of profile boarding/alighting demand buckets. "
                        if request.candidate_order == "station_patronage_coverage"
                        else ""
                    )
                    + outcome
                ),
            )
        )

    objective = policy.objective
    results.sort(
        key=lambda item: (
            not item.feasible,
            item.new_conflict_count,
            (
                item.patronage_coverage_status != "available"
                if request.candidate_order == "station_patronage_coverage"
                else False
            ),
            (
                -(item.patronage_coverage.combined_event_coverage or 0)
                if request.candidate_order == "station_patronage_coverage"
                and item.patronage_coverage is not None
                else 0
            ),
            (
                item.total_shift_seconds
                if objective == "minimize_total_shift"
                else item.changed_service_count
                if objective == "minimize_changed_services"
                else item.retiming_priority_score
                if objective == "minimize_weighted_shift"
                else item.simulated_runtime_change_seconds is None
            ),
            item.simulated_runtime_change_seconds
            if objective == "minimize_simulated_runtime_change"
            and item.simulated_runtime_change_seconds is not None
            else 0,
            item.total_shift_seconds,
            item.route_alternative_index,
            item.departure_offset_seconds,
        )
    )
    for rank, item in enumerate(results, start=1):
        item.rank = rank
    feasible_count = sum(item.feasible for item in results)
    return HourlyServicePlanResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        generated_candidate_count=len(phases) * len(route_options) * len(retiming_candidates),
        evaluated_candidate_count=len(results),
        feasible_candidate_count=feasible_count,
        selected_candidate_id=next((item.candidate_id for item in results if item.feasible), None),
        candidates=results,
    )


async def assemble_service_plan_replay_episode(
    request: ServicePlanRequest,
    candidate: ServicePlanCandidate,
    client: OSRDClient,
    *,
    baseline_context: ReplayBaselineContext | None = None,
) -> dict[str, Any]:
    """Assemble outcome-free replay inputs from a candidate and exact OSRD baseline.

    This is offline adapter groundwork only. It does not rank candidates and never
    includes candidate simulation outcomes in model inputs.
    """
    snapshot = SnapshotRef(**request.service.snapshot.model_dump())
    baseline = baseline_context or await client.get_replay_baseline_context(snapshot=snapshot)
    if baseline.snapshot != snapshot:
        raise UnsupportedSnapshot("Replay baseline context does not match the requested snapshot")
    if (
        len(baseline.schedule_ids) != len(baseline.schedule_payloads)
        or len(set(baseline.schedule_ids)) != len(baseline.schedule_ids)
    ):
        raise UnsupportedSnapshot("OSRD baseline context has mismatched schedule IDs and forms")

    candidate_schedules = [dict(schedule) for schedule in baseline.schedule_payloads]
    baseline_index = {
        str(schedule_id): index for index, schedule_id in enumerate(baseline.schedule_ids)
    }
    seen_edits: set[str] = set()
    for edit in candidate.edits:
        index = baseline_index.get(edit.train_schedule_id)
        if index is None or edit.train_schedule_id in seen_edits:
            raise UnsupportedSnapshot(
                f"Replay candidate contains an unknown or duplicate edit: {edit.train_schedule_id}"
            )
        seen_edits.add(edit.train_schedule_id)
        source_start = candidate_schedules[index].get("start_time")
        if not isinstance(source_start, int) or isinstance(source_start, bool):
            raise UnsupportedSnapshot(
                f"Baseline start time is invalid for schedule {edit.train_schedule_id}"
            )
        if source_start != edit.before_start_time_ms:
            raise UnsupportedSnapshot(
                f"Replay candidate edit is stale for baseline schedule {edit.train_schedule_id}"
            )
        expected_start = source_start + edit.shift_seconds * 1000
        if expected_start != edit.after_start_time_ms:
            raise UnsupportedSnapshot(
                f"Replay candidate shift does not match schedule {edit.train_schedule_id}"
            )
        candidate_schedules[index]["start_time"] = expected_start
    candidate_schedules.extend(dict(schedule) for schedule in candidate.proposed_schedules)
    action = build_service_plan_action(request, candidate, baseline.schedule_ids)
    resource_paths = await client.get_schedule_resource_paths(
        snapshot=snapshot, schedules=candidate_schedules
    )
    return build_service_plan_replay_episode(
        snapshot=request.service.snapshot,
        baseline_schedule_ids=baseline.schedule_ids,
        baseline_schedule_payloads=baseline.schedule_payloads,
        candidate_schedule_payloads=candidate_schedules,
        baseline_simulation_summaries=baseline.simulation_summaries,
        baseline_conflicts=baseline.conflicts,
        resource_paths_by_train=resource_paths,
        action=action,
    )


def _route_waypoints(
    service: ServiceRequest | HourlyServiceRequest,
    legs: list[list[dict[str, Any]]],
    route_indices: tuple[int, ...],
) -> tuple[list[ServicePathWaypoint], list[str]]:
    """Constrain OSRD to a selected sequence of track ranges using ordered waypoints."""
    result = [service.path[0]]
    track_sections: list[str] = []
    used_keys = {waypoint.key for waypoint in service.path}
    for leg_index, route_index in enumerate(route_indices):
        ranges_value = legs[leg_index][route_index].get("track_ranges")
        if not isinstance(ranges_value, list):
            raise UnsupportedSnapshot("OSRD returned an empty route alternative")
        ranges = cast(list[object], ranges_value)
        if not ranges:
            raise UnsupportedSnapshot("OSRD returned an empty route alternative")
        for range_index, range_value in enumerate(ranges):
            if not isinstance(range_value, dict):
                raise UnsupportedSnapshot("OSRD returned an invalid route track range")
            track_range = cast(dict[str, Any], range_value)
            track = track_range.get("track")
            begin, end = track_range.get("begin"), track_range.get("end")
            if (
                not isinstance(track, str)
                or not isinstance(begin, int | float)
                or not isinstance(end, int | float)
            ):
                raise UnsupportedSnapshot("OSRD returned an invalid route track range")
            track_sections.append(track)
            # The k-shortest endpoint returns RailJSON metres; train schedule
            # track_offset waypoints are in millimetres.
            offset = round((begin + end) * 500)
            location = {"type": "track_offset", "track": track, "offset": offset}
            previous = result[-1]
            if previous.location == location:
                continue
            key = f"__osrd_route_{leg_index}_{range_index}"
            suffix = 1
            while key in used_keys:
                suffix += 1
                key = f"__osrd_route_{leg_index}_{range_index}_{suffix}"
            used_keys.add(key)
            result.append(ServicePathWaypoint(key=key, location=location))
        destination = service.path[leg_index + 1]
        if result[-1].location == destination.location:
            result[-1] = destination
        else:
            result.append(destination)
    return result, track_sections


def _route_length_m(route: dict[str, Any]) -> float:
    ranges_value = route.get("track_ranges")
    if not isinstance(ranges_value, list) or not ranges_value:
        raise UnsupportedSnapshot("OSRD returned an empty route alternative")
    length_m = 0.0
    for range_value in cast(list[object], ranges_value):
        if not isinstance(range_value, dict):
            raise UnsupportedSnapshot("OSRD returned an invalid route track range")
        track_range = cast(dict[str, Any], range_value)
        begin, end = track_range.get("begin"), track_range.get("end")
        if not isinstance(begin, int | float) or not isinstance(end, int | float):
            raise UnsupportedSnapshot("OSRD returned an invalid route track range")
        length_m += abs(float(end) - float(begin))
    return length_m


def _k_best_route_indices(legs: list[list[dict[str, Any]]], limit: int) -> list[tuple[int, ...]]:
    """Enumerate the shortest combinations without materializing the Cartesian product."""
    if not legs or any(not choices for choices in legs):
        return []
    lengths = [[_route_length_m(route) for route in choices] for choices in legs]
    initial = tuple(0 for _ in legs)
    queue: list[tuple[float, tuple[int, ...]]] = [(sum(row[0] for row in lengths), initial)]
    seen = {initial}
    results: list[tuple[int, ...]] = []
    while queue and len(results) < limit:
        score, indices = heapq.heappop(queue)
        results.append(indices)
        for leg_index in range(len(legs)):
            next_index = indices[leg_index] + 1
            if next_index >= len(legs[leg_index]):
                continue
            neighbor = (*indices[:leg_index], next_index, *indices[leg_index + 1 :])
            if neighbor not in seen:
                seen.add(neighbor)
                next_score = (
                    score - lengths[leg_index][indices[leg_index]] + lengths[leg_index][next_index]
                )
                heapq.heappush(queue, (next_score, neighbor))
    return results


async def validate_service_request(
    request: ServiceRequest,
    client: OSRDClient,
) -> ServiceRequestValidationResponse:
    """Create a temporary full timetable with the requested departures and validate it."""
    if any(waypoint.optional_stop for waypoint in request.path):
        raise ValueError("optional stops are only supported for complete timetable plans")
    if request.max_travel_time_seconds is not None:
        raise ValueError(
            "max_travel_time_seconds is only supported for complete timetable plans"
        )
    snapshot = SnapshotRef(**request.snapshot.model_dump())
    schedules = build_service_schedules(request)
    validation: CandidateValidation = await client.validate_candidate(
        snapshot=snapshot,
        candidate_payload={
            "candidate_id": f"service-request-{request.request_id}",
            "snapshot": asdict(snapshot),
            "edits": [],
            "insertions": schedules,
            "work_schedule_group_id": request.work_schedule_group_id,
        },
    )
    departures = _service_departure_times(request)
    baseline_signatures = Counter(validation.baseline_conflict_signatures)
    candidate_signatures = Counter(validation.conflict_signatures)
    new_conflict_count = sum((candidate_signatures - baseline_signatures).values())
    return ServiceRequestValidationResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        requested_departure_count=len(schedules),
        departures=departures,
        baseline_conflict_count=len(validation.baseline_conflict_signatures),
        new_conflict_count=new_conflict_count,
        validation=PlanCandidateValidation(
            pathfinding_valid=validation.pathfinding_valid,
            simulation_valid=validation.simulation_valid,
            conflict_count=validation.conflict_count,
            baseline_conflict_count=len(validation.baseline_conflict_signatures),
            new_conflict_count=new_conflict_count,
            failure_reasons=list(validation.failure_reasons),
            conflicting_existing_service_ids=list(
                validation.conflicting_existing_service_ids
            ),
        ),
        feasible=(
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflict_count == 0
            and not any("cleanup failed" in reason for reason in validation.failure_reasons)
        ),
    )


async def plan_service_request(
    request: ServicePlanRequest,
    client: OSRDClient,
    *,
    insertion_model_path: Path | None = None,
) -> ServicePlanResponse:
    """Enumerate bounded timetable edits and validate each complete insertion in OSRD."""
    service = request.service
    snapshot = SnapshotRef(**service.snapshot.model_dump())
    metadata = await client.get_schedule_metadata(snapshot=snapshot)
    if metadata.timetable_type != "CALENDAR":
        raise UnsupportedSnapshot(
            "Frequency service planning currently requires a calendar timetable; "
            "hourly timetable insertion needs recurring paced-service candidates"
        )
    starts = metadata.start_times_ms
    train_names = metadata.train_names
    policy = request.retiming
    retiming_possible = policy.max_changed_services > 0 and policy.max_shift_seconds > 0
    if (
        policy.objective == "minimize_simulated_runtime_change"
        and retiming_possible
    ):
        unsupported = sorted(
            set(policy.eligible_train_schedule_ids)
            & metadata.runtime_objective_unsupported_ids
        )
        if unsupported:
            raise ValueError(
                "The simulated running-time objective cannot compare paced services with "
                "timing-affecting exceptions. Choose another objective or remove these "
                f"schedule IDs from the eligible set: {', '.join(unsupported)}"
            )
    route_options: list[tuple[list[ServicePathWaypoint], list[str]]]
    if request.max_route_alternatives == 1:
        route_options = [(service.path, [])]
    else:
        legs: list[list[dict[str, Any]]] = []
        for origin, destination in zip(service.path, service.path[1:], strict=False):
            legs.append(
                await client.get_route_alternatives(
                    snapshot=snapshot,
                    starting=origin.location,
                    ending=destination.location,
                    limit=request.max_route_alternatives,
                )
            )
        route_indices = _k_best_route_indices(legs, request.max_route_alternatives)
        if not route_indices:
            raise UnsupportedSnapshot("OSRD found no route alternatives for the requested path")
        route_options = [_route_waypoints(service, legs, indices) for indices in route_indices]
    baseline_travel_times_ms = (
        await client.get_schedule_travel_times(
            snapshot=snapshot,
            schedule_ids=policy.eligible_train_schedule_ids,
        )
        if retiming_possible and policy.eligible_train_schedule_ids
        else {}
    )
    candidates: list[DepartureShiftCandidate] = generate_service_retiming_candidates(
        service_request_id=str(service.request_id),
        snapshot=service.snapshot,
        schedule_start_times_ms=starts,
        eligible_schedule_ids=policy.eligible_train_schedule_ids,
        protected_schedule_ids=policy.protected_train_schedule_ids,
        max_edits=policy.max_changed_services,
        max_shift_seconds=policy.max_shift_seconds,
        max_shift_seconds_by_schedule=policy.max_shift_seconds_by_schedule,
        max_advance_seconds_by_schedule=policy.max_advance_seconds_by_schedule,
        max_candidates=policy.max_candidates,
        step_seconds=policy.step_seconds,
    )
    phase_limit = min(service.frequency_seconds, service.minimum_operating_window_seconds)
    has_retiming_candidates = any(candidate.edits for candidate in candidates)
    phase_budget = (
        max(1, policy.max_candidates // 2) if has_retiming_candidates else policy.max_candidates
    )
    available_phases = list(range(0, phase_limit, request.phase_step_seconds))
    phase_count = min(
        request.max_phase_options,
        max(1, phase_budget // len(route_options)),
        len(available_phases),
    )
    phase_options = _sample_evenly_spaced(available_phases, phase_count)
    if not phase_options:
        raise ValueError("Service planning produced no valid departure phases")
    # Cover every first-departure phase before spending the remaining bounded
    # validation budget on combinations with existing-service retimings.
    route_phase_options = [
        (route_index, phase) for phase in phase_options for route_index in range(len(route_options))
    ]
    if len(route_phase_options) > policy.max_candidates:
        route_phase_options = _sample_evenly_spaced(
            route_phase_options, policy.max_candidates
        )
    plans: list[tuple[int, int, DepartureShiftCandidate]] = [
        (route_index, phase, candidate)
        for route_index, phase in route_phase_options
        for candidate in candidates
        if not candidate.edits
    ]
    retiming_candidates = [candidate for candidate in candidates if candidate.edits]
    for candidate in retiming_candidates:
        for route_index, phase in route_phase_options:
            if len(plans) >= policy.max_candidates:
                break
            plans.append((route_index, phase, candidate))
        if len(plans) >= policy.max_candidates:
            break
    plans = plans[: policy.max_candidates]
    baseline_signatures_seen: tuple[str, ...] | None = None
    results: list[ServicePlanCandidate] = []
    for route_index, phase_seconds, candidate in plans:
        route_path, route_track_sections = route_options[route_index]
        insertions = build_service_schedules(
            service, departure_phase_seconds=phase_seconds, path=route_path
        )
        identity = f"{candidate.candidate_id}-route-{route_index}-phase-{phase_seconds}"
        candidate_digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        candidate_id = f"service-plan-{candidate_digest}"
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload={
                "candidate_id": candidate_id,
                "snapshot": asdict(snapshot),
                "edits": [asdict(edit) for edit in candidate.edits],
                "insertions": insertions,
                "work_schedule_group_id": service.work_schedule_group_id,
            },
        )
        baseline_signatures = Counter(validation.baseline_conflict_signatures)
        candidate_signatures = Counter(validation.conflict_signatures)
        new_conflict_count = sum((candidate_signatures - baseline_signatures).values())
        current_baseline_count = len(validation.baseline_conflict_signatures)
        canonical_baseline = tuple(sorted(validation.baseline_conflict_signatures))
        if baseline_signatures_seen is not None and canonical_baseline != baseline_signatures_seen:
            raise UnsupportedSnapshot("Baseline conflicts changed during candidate validation")
        baseline_signatures_seen = canonical_baseline
        feasible = (
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflict_count == 0
            and not any("cleanup failed" in reason for reason in validation.failure_reasons)
        )
        departures = [
            datetime.fromtimestamp(int(insertion["start_time"]) / 1000, tz=UTC)
            for insertion in insertions
        ]
        departure_timings = []
        for insertion in insertions:
            train_name = str(insertion["train_name"])
            travel_time_ms = validation.inserted_travel_times_ms.get(train_name)
            if travel_time_ms is None:
                continue
            start_time_ms = int(insertion["start_time"])
            departure_time = datetime.fromtimestamp(start_time_ms / 1000, tz=UTC)
            arrival_time = datetime.fromtimestamp((start_time_ms + travel_time_ms) / 1000, tz=UTC)
            departure_timings.append(
                {
                    "train_name": train_name,
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "travel_time_seconds": round(travel_time_ms / 1000),
                }
            )
        existing_service_timings = []
        for edit in candidate.edits:
            travel_time_ms = validation.existing_travel_times_ms.get(edit.train_schedule_id)
            if travel_time_ms is None:
                continue
            departure_time = datetime.fromtimestamp(edit.after_start_time_ms / 1000, tz=UTC)
            baseline_travel_time_ms = baseline_travel_times_ms.get(edit.train_schedule_id)
            existing_service_timings.append(
                ExistingServiceTiming(
                    train_schedule_id=edit.train_schedule_id,
                    train_name=train_names.get(edit.train_schedule_id),
                    departure_time=departure_time,
                    arrival_time=datetime.fromtimestamp(
                        (edit.after_start_time_ms + travel_time_ms) / 1000, tz=UTC
                    ),
                    travel_time_seconds=round(travel_time_ms / 1000),
                    baseline_travel_time_seconds=(
                        round(baseline_travel_time_ms / 1000)
                        if baseline_travel_time_ms is not None
                        else None
                    ),
                    travel_time_delta_seconds=(
                        round((travel_time_ms - baseline_travel_time_ms) / 1000)
                        if baseline_travel_time_ms is not None
                        else None
                    ),
                    arrival_time_delta_seconds=(
                        round(
                            (
                                edit.after_start_time_ms
                                + travel_time_ms
                                - edit.before_start_time_ms
                                - baseline_travel_time_ms
                            )
                            / 1000
                        )
                        if baseline_travel_time_ms is not None
                        else None
                    ),
                    retimed=True,
                )
            )
        runtime_change_is_complete = len(existing_service_timings) == len(candidate.edits) and all(
            timing.travel_time_delta_seconds is not None for timing in existing_service_timings
        )
        simulated_runtime_change_seconds = (
            sum(
                timing.travel_time_delta_seconds or 0
                for timing in existing_service_timings
            )
            if runtime_change_is_complete
            else None
        )
        station_call_timings, patronage_coverage, patronage_coverage_status = (
            _candidate_patronage_coverage(
                request.patronage_profile,
                insertions,
                validation.inserted_path_item_times_ms,
            )
        )
        results.append(
            ServicePlanCandidate(
                candidate_id=candidate_id,
                rank=1,
                explanation=_explain_service_candidate(
                    candidate=candidate,
                    request=request,
                    phase_seconds=phase_seconds,
                    route_index=route_index,
                    departure_count=len(departures),
                    feasible=feasible,
                    pathfinding_valid=validation.pathfinding_valid,
                    simulation_valid=validation.simulation_valid,
                    new_conflict_count=new_conflict_count,
                    failure_reasons=list(validation.failure_reasons),
                )
                + (
                    " After feasibility and conflict checks, this option is ordered by the share "
                    "of observed boarding/alighting events in profile time buckets matched by "
                    "its simulated station calls; this is a coverage proxy, not passenger benefit."
                    if request.candidate_order == "station_patronage_coverage"
                    else ""
                ),
                route_alternative_index=route_index,
                route_track_sections=route_track_sections,
                proposed_schedules=insertions,
                departure_phase_seconds=phase_seconds,
                edits=[
                    PlanCandidateEdit(
                        train_schedule_id=edit.train_schedule_id,
                        train_name=train_names.get(edit.train_schedule_id),
                        before_start_time_ms=edit.before_start_time_ms,
                        after_start_time_ms=edit.after_start_time_ms,
                        shift_seconds=edit.shift_seconds,
                    )
                    for edit in candidate.edits
                ],
                total_shift_seconds=candidate.total_shift_seconds,
                changed_service_count=len(candidate.edits),
                simulated_runtime_change_seconds=simulated_runtime_change_seconds,
                retiming_priority_score=_retiming_priority_score(candidate, request),
                inserted_departures=departures,
                departure_timings=departure_timings,
                station_call_timings=station_call_timings,
                patronage_coverage_status=patronage_coverage_status,
                patronage_coverage=patronage_coverage,
                existing_service_timings=existing_service_timings,
                baseline_conflict_count=current_baseline_count,
                new_conflict_count=new_conflict_count,
                validation=PlanCandidateValidation(
                    pathfinding_valid=validation.pathfinding_valid,
                    simulation_valid=validation.simulation_valid,
                    conflict_count=validation.conflict_count,
                    baseline_conflict_count=current_baseline_count,
                    new_conflict_count=new_conflict_count,
                    failure_reasons=list(validation.failure_reasons),
                    conflicting_existing_service_ids=list(
                        validation.conflicting_existing_service_ids
                    ),
                    conflict_fingerprint=_conflict_fingerprint(
                        validation.conflict_signatures
                    ),
                    baseline_conflict_fingerprint=_conflict_fingerprint(
                        validation.baseline_conflict_signatures
                    ),
                ),
                feasible=feasible,
            )
        )

    def sort_key(result: ServicePlanCandidate) -> tuple[object, ...]:
        patronage_coverage = (
            result.patronage_coverage.combined_event_coverage
            if result.patronage_coverage is not None
            else None
        )
        demand_order = (
            -patronage_coverage
            if request.candidate_order == "station_patronage_coverage"
            and patronage_coverage is not None
            else (
                float("inf")
                if request.candidate_order == "station_patronage_coverage"
                else 0.0
            )
        )
        if policy.objective == "minimize_total_shift":
            objective: float = result.total_shift_seconds
        elif policy.objective == "minimize_changed_services":
            objective = result.changed_service_count
        elif policy.objective == "minimize_weighted_shift":
            objective = float(result.retiming_priority_score or 0)
        else:
            objective = (
                float(result.simulated_runtime_change_seconds)
                if result.simulated_runtime_change_seconds is not None
                else float("inf")
            )
        return (
            not result.feasible,
            result.new_conflict_count,
            demand_order,
            objective,
            result.changed_service_count,
            result.total_shift_seconds,
            result.departure_phase_seconds,
            result.route_alternative_index,
            result.candidate_id,
        )

    results.sort(key=sort_key)
    for index, result in enumerate(results, start=1):
        result.rank = index
    selected = next((result.candidate_id for result in results if result.feasible), None)
    model_status = "not_configured" if insertion_model_path is None else "not_applicable"
    model_suggested_candidate_id: str | None = None
    model_sha256: str | None = None
    model_note: str | None = (
        "No research insertion model is configured; options use deterministic ordering."
        if insertion_model_path is None
        else None
    )
    model_candidates = [
        result
        for result in results
        if result.feasible
        and not result.edits
        and len(result.proposed_schedules) == 1
    ]
    # The v5 insertion model has no stop, multi-departure, route-choice, or retiming
    # training coverage. Keep its advisory out of all broader planning requests.
    model_supported_request = (
        insertion_model_path is not None
        and request.candidate_order == "operational"
        and request.patronage_profile is None
        and service.work_schedule_group_id is None
        and request.max_route_alternatives == 1
        and len(service.path) == 2
        and request.retiming.max_changed_services == 0
        and len(model_candidates) >= 2
        and len(model_candidates) <= 16
        and len(model_candidates) == sum(result.feasible for result in results)
    )
    if insertion_model_path is not None and not model_supported_request:
        model_note = (
            "No model suggestion for this request: the experimental ranker has not been evaluated "
            "with station-patronage inputs or candidate ordering."
            if (
                request.candidate_order == "station_patronage_coverage"
                or request.patronage_profile is not None
            )
            else (
                "No model suggestion for this request: the experimental ranker supports "
                "2–16 feasible single-departure options with a direct origin-destination path, "
                "one requested route, "
                "no existing-service retiming, and no work-schedule group."
            )
        )
    if insertion_model_path is not None and model_supported_request:
        try:
            baseline = await client.get_replay_baseline_context(snapshot=snapshot)
            baseline_resource_paths = await client.get_schedule_resource_paths(
                snapshot=snapshot,
                schedules=baseline.schedule_payloads,
            )
            proposed_resource_path = await client.get_schedule_resource_paths(
                snapshot=snapshot,
                schedules=[dict(model_candidates[0].proposed_schedules[0])],
            )
            model_inputs: list[InsertionRankingCandidate] = []
            for candidate in model_candidates:
                proposed_schedules = [dict(candidate.proposed_schedules[0])]
                schedules = [dict(schedule) for schedule in baseline.schedule_payloads]
                schedules.extend(proposed_schedules)
                resource_paths = [*baseline_resource_paths, *proposed_resource_path]
                episode = build_insertion_replay_episode(
                    snapshot=service.snapshot,
                    baseline_schedule_ids=baseline.schedule_ids,
                    baseline_schedule_payloads=baseline.schedule_payloads,
                    candidate_schedule_payloads=schedules,
                    baseline_simulation_summaries=baseline.simulation_summaries,
                    baseline_conflicts=baseline.conflicts,
                    resource_paths_by_train=resource_paths,
                    action={
                        "type": "service-insertion",
                        "train_index": len(baseline.schedule_ids),
                        "departure_time_ms": int(candidate.proposed_schedules[0]["start_time"]),
                    },
                )
                model_inputs.append(
                    InsertionRankingCandidate(
                        candidate_id=candidate.candidate_id,
                        replay_episode=episode,
                    )
                )
            model_result = await asyncio.to_thread(
                rank_insertion_candidates,
                InsertionRankingRequest(candidates=model_inputs),
                model_path=insertion_model_path,
            )
            model_sha256 = model_result.model_sha256
            model_suggested_candidate_id = model_result.ranked_candidates[0].candidate_id
            model_scores = {
                score.candidate_id: score
                for score in model_result.ranked_candidates
            }
            for rank, candidate in enumerate(model_result.ranked_candidates, start=1):
                result = next(
                    row for row in model_candidates if row.candidate_id == candidate.candidate_id
                )
                result.research_model_rank = rank
                result.research_model_mean_conflict_risk = (
                    candidate.mean_existing_service_conflict_probability
                )
                result.research_model_max_conflict_risk = (
                    candidate.max_existing_service_conflict_probability
                )
                highest_risk_service = max(
                    candidate.affected_services,
                    key=lambda impact: (
                        impact.conflict_probability,
                        -impact.baseline_schedule_id,
                    ),
                )
                highest_risk_schedule_id = str(highest_risk_service.baseline_schedule_id)
                result.research_model_highest_risk_schedule_id = highest_risk_schedule_id
                result.research_model_highest_risk_train_name = train_names.get(
                    highest_risk_schedule_id
                )
            if len(model_scores) == len(model_candidates):
                model_status = "used"
                model_note = (
                    "The experimental model ranks eligible options by predicted conflict risk "
                    "to existing services. It was trained and evaluated on synthetic OSRD "
                    "fixtures; scores are not calibrated estimates for real operations. "
                    "OSRD validation and deterministic option order remain authoritative."
                )
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            # A research suggestion must never make an already validated plan unavailable.
            logger.warning("Optional research insertion ranking unavailable: %s", error)
            model_status = "unavailable"
            model_note = (
                "The optional research model was unavailable; deterministic planning results "
                "are shown."
            )
            model_suggested_candidate_id = None
            model_sha256 = None
    return ServicePlanResponse(
        request_id=service.request_id,
        snapshot=service.snapshot,
        generated_candidate_count=len(plans),
        evaluated_candidate_count=len(results),
        feasible_candidate_count=sum(result.feasible for result in results),
        selected_candidate_id=selected,
        candidate_order=request.candidate_order,
        research_model_status=model_status,
        research_model_suggested_candidate_id=model_suggested_candidate_id,
        research_model_sha256=model_sha256,
        research_model_note=model_note,
        candidates=results,
    )


async def plan_complete_timetable(
    request: CompleteTimetablePlanRequest,
    client: OSRDClient,
) -> CompleteTimetablePlanResponse:
    """Jointly search bounded service phases and eligible-service retimings through OSRD."""
    first_service = request.services[0]
    snapshot = SnapshotRef(**first_service.snapshot.model_dump())
    metadata = await client.get_schedule_metadata(snapshot=snapshot)
    if metadata.timetable_type != "CALENDAR":
        raise UnsupportedSnapshot(
            "Complete timetable planning currently requires a calendar timetable"
        )

    policy = request.retiming
    if (
        policy.objective == "minimize_simulated_runtime_change"
        and policy.max_changed_services > 0
        and policy.max_shift_seconds > 0
    ):
        unsupported = sorted(
            set(policy.eligible_train_schedule_ids)
            & metadata.runtime_objective_unsupported_ids
        )
        if unsupported:
            raise ValueError(
                "The simulated running-time objective cannot compare paced services with "
                "timing-affecting exceptions. Remove these schedule IDs or choose another "
                f"objective: {', '.join(unsupported)}"
            )
    baseline_travel_times_ms = await client.get_schedule_travel_times(
        snapshot=snapshot,
        schedule_ids=list(metadata.start_times_ms),
    )
    dwell_dimensions: list[tuple[ExistingServiceDwellOptions, int, list[int]]] = []
    protected_ids = set(policy.protected_train_schedule_ids)
    for dwell_request in request.existing_service_dwell_options:
        schedule_id = dwell_request.train_schedule_id
        if schedule_id in protected_ids:
            raise ValueError(
                f"Protected existing service {schedule_id} cannot receive a dwell edit"
            )
        form = metadata.schedule_forms_by_id.get(schedule_id)
        if form is None:
            raise ValueError(f"Unknown existing train schedule ID for dwell edit: {schedule_id}")
        path = form.get("path")
        schedule_rows = form.get("schedule")
        if (
            not isinstance(path, list)
            or not isinstance(schedule_rows, list)
            or not all(isinstance(item, dict) for item in schedule_rows)
        ):
            raise UnsupportedSnapshot(
                f"Existing service {schedule_id} has no editable path or station-call schedule"
            )
        path_keys = [
            item.get("key") for item in path if isinstance(item, dict)
        ]
        if len(path_keys) != len(path) or not all(isinstance(key, str) for key in path_keys):
            raise UnsupportedSnapshot(f"Existing service {schedule_id} has invalid path keys")
        if len(path_keys) != len(set(path_keys)):
            raise UnsupportedSnapshot(f"Existing service {schedule_id} has duplicate path keys")
        try:
            waypoint_index = path_keys.index(dwell_request.waypoint_key)
        except ValueError as error:
            raise ValueError(
                f"Waypoint {dwell_request.waypoint_key} is not in existing service {schedule_id}"
            ) from error
        if waypoint_index == 0 or waypoint_index == len(path_keys) - 1:
            raise ValueError("Existing-service dwell edits must target an intermediate waypoint")
        matching_rows = [
            item
            for item in schedule_rows
            if isinstance(item, dict) and item.get("at") == dwell_request.waypoint_key
        ]
        if len(matching_rows) > 1:
            raise UnsupportedSnapshot(
                f"Existing service {schedule_id} has duplicate station-call entries"
            )
        before_dwell = (
            _stop_for_seconds(matching_rows[0].get("stop_for"))
            if matching_rows
            else 0
        )
        choices = list(dict.fromkeys([before_dwell, *dwell_request.dwell_seconds]))
        dwell_dimensions.append((dwell_request, before_dwell, choices))

    retiming_candidates = generate_service_retiming_candidates(
        service_request_id=str(request.request_id),
        snapshot=first_service.snapshot,
        schedule_start_times_ms=metadata.start_times_ms,
        eligible_schedule_ids=policy.eligible_train_schedule_ids,
        protected_schedule_ids=policy.protected_train_schedule_ids,
        max_edits=policy.max_changed_services,
        max_shift_seconds=policy.max_shift_seconds,
        max_shift_seconds_by_schedule=policy.max_shift_seconds_by_schedule,
        max_advance_seconds_by_schedule=policy.max_advance_seconds_by_schedule,
        max_candidates=policy.max_candidates,
        step_seconds=policy.step_seconds,
    )
    if not retiming_candidates:
        raise ValueError("Retiming policy produced no candidate, including the unchanged baseline")

    phase_options: list[list[int]] = []
    stop_options_by_service: list[
        list[tuple[tuple[str, ...], dict[str, int]]]
    ] = []
    route_options_by_service: list[list[tuple[list[ServicePathWaypoint], list[str]]]] = []
    for service in request.services:
        window = service.minimum_operating_window_seconds
        phase_limit = min(service.frequency_seconds, window)
        phases = list(range(0, phase_limit, request.phase_step_seconds))
        phase_options.append(
            _sample_evenly_spaced(phases, request.max_phase_options_per_service)
        )
        stop_options_by_service.append(
            _complete_plan_stop_options(
                service, request.max_stop_pattern_options_per_service
            )
        )
        if request.max_route_alternatives_per_service == 1:
            route_options_by_service.append([(service.path, [])])
            continue
        legs: list[list[dict[str, Any]]] = []
        for origin, destination in zip(service.path, service.path[1:], strict=False):
            legs.append(
                await client.get_route_alternatives(
                    snapshot=snapshot,
                    starting=origin.location,
                    ending=destination.location,
                    limit=request.max_route_alternatives_per_service,
                )
            )
        route_indices = _k_best_route_indices(
            legs, request.max_route_alternatives_per_service
        )
        if not route_indices:
            raise UnsupportedSnapshot(
                f"OSRD found no route alternatives for complete-plan service {service.train_name}"
            )
        route_options_by_service.append(
            [_route_waypoints(service, legs, indices) for indices in route_indices]
        )
    service_variant_options = [
        [
            (phase, route_index, stop_index)
            for phase in phases
            for route_index in range(len(route_options))
            for stop_index in range(len(stop_options))
        ]
        for phases, route_options, stop_options in zip(
            phase_options, route_options_by_service, stop_options_by_service, strict=True
        )
    ]
    total_combinations = math.prod(len(options) for options in phase_options)
    total_route_combinations = math.prod(
        len(options) for options in route_options_by_service
    )
    total_route_phase_combinations = math.prod(
        len(phases) * len(routes)
        for phases, routes in zip(phase_options, route_options_by_service, strict=True)
    )
    total_stop_pattern_combinations = math.prod(len(options) for options in stop_options_by_service)
    total_route_phase_stop_combinations = math.prod(
        len(options) for options in service_variant_options
    )
    dwell_candidate_count = math.prod(len(choices) for _, _, choices in dwell_dimensions)
    total_candidate_combinations = (
        total_route_phase_stop_combinations
        * len(retiming_candidates)
        * dwell_candidate_count
    )
    if total_candidate_combinations <= request.max_candidates:
        plans = [
            (service_variants, retiming_candidate, dwell_index)
            for service_variants in product(*service_variant_options)
            for retiming_candidate in retiming_candidates
            for dwell_index in range(dwell_candidate_count)
        ]
    else:
        if request.max_candidates == 1:
            sampled_indices = [0]
        else:
            sampled_indices = [
                round(
                    index
                    * (total_candidate_combinations - 1)
                    / (request.max_candidates - 1)
                )
                for index in range(request.max_candidates)
            ]
        plans = []
        for candidate_index in sampled_indices:
            variant_index, remaining = divmod(
                candidate_index, len(retiming_candidates) * dwell_candidate_count
            )
            retiming_index, dwell_index = divmod(remaining, dwell_candidate_count)
            remainder = variant_index
            digits: list[int] = []
            for options in reversed(service_variant_options):
                digits.append(remainder % len(options))
                remainder //= len(options)
            service_variant_choice = tuple(
                    options[digit]
                    for options, digit in zip(
                        service_variant_options, reversed(digits), strict=True
                    )
                )
            plans.append(
                (
                    service_variant_choice,
                    retiming_candidates[retiming_index],
                    dwell_index,
                )
            )
    baseline_signatures: tuple[str, ...] | None = None
    candidates: list[CompleteTimetableCandidate] = []
    for service_variants, retiming_candidate, dwell_index in plans:
        phases = tuple(variant[0] for variant in service_variants)
        route_indices = tuple(variant[1] for variant in service_variants)
        stop_indices = tuple(variant[2] for variant in service_variants)
        insertions: list[dict[str, object]] = []
        pattern_train_names: list[list[str]] = []
        pattern_results: list[CompleteTimetablePatternResult] = []
        for service, phase, route_index, stop_index, route_options, stop_options in zip(
            request.services,
            phases,
            route_indices,
            stop_indices,
            route_options_by_service,
            stop_options_by_service,
            strict=True,
        ):
            route_path, route_track_sections = route_options[route_index]
            selected_stops, selected_dwell_by_key = stop_options[stop_index]
            selected_stop_set = set(selected_stops)
            schedule_path = [
                waypoint.model_copy(
                    update={
                        "optional_stop": False,
                        "stop_for_seconds": (
                            selected_dwell_by_key.get(waypoint.key, waypoint.stop_for_seconds)
                            if not waypoint.optional_stop or waypoint.key in selected_stop_set
                            else None
                        ),
                    }
                )
                for waypoint in route_path
            ]
            schedules = build_service_schedules(
                service,
                departure_phase_seconds=phase,
                path=schedule_path,
            )
            pattern_train_names.append([str(schedule["train_name"]) for schedule in schedules])
            insertions.extend(schedules)
            departures = _service_departure_times(service, phase)
            pattern_results.append(
                CompleteTimetablePatternResult(
                    train_name=service.train_name,
                    departure_phase_seconds=phase,
                    departure_count=len(schedules),
                    departures=departures,
                    route_alternative_index=route_index,
                    route_track_sections=route_track_sections,
                    selected_optional_stop_keys=list(selected_stops),
                    omitted_optional_stop_keys=[
                        waypoint.key
                        for waypoint in service.path
                        if waypoint.optional_stop and waypoint.key not in selected_stop_set
                    ],
                    stop_dwell_seconds_by_key=selected_dwell_by_key,
                )
            )
        remaining_dwell_index = dwell_index
        selected_dwell_values: list[int] = []
        for _, _, choices in reversed(dwell_dimensions):
            remaining_dwell_index, choice_index = divmod(remaining_dwell_index, len(choices))
            selected_dwell_values.append(choices[choice_index])
        selected_dwell_values.reverse()
        existing_service_dwell_edits = [
            ExistingServiceDwellEdit(
                train_schedule_id=dwell_request.train_schedule_id,
                train_name=metadata.train_names.get(dwell_request.train_schedule_id),
                waypoint_key=dwell_request.waypoint_key,
                before_dwell_seconds=before_dwell,
                after_dwell_seconds=after_dwell,
            )
            for (dwell_request, before_dwell, _), after_dwell in zip(
                dwell_dimensions, selected_dwell_values, strict=True
            )
            if after_dwell != before_dwell
        ]
        digest = hashlib.sha256(
            json.dumps(
                [
                    str(request.request_id),
                    list(phases),
                    list(route_indices),
                    [
                        {
                            "optional_stops": list(stop_options[index][0]),
                            "dwell_seconds": stop_options[index][1],
                        }
                        for stop_options, index in zip(
                            stop_options_by_service, stop_indices, strict=True
                        )
                    ],
                    retiming_candidate.candidate_id,
                    [edit.model_dump() for edit in existing_service_dwell_edits],
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        candidate_id = f"complete-timetable-{digest}"
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload={
                "candidate_id": candidate_id,
                "snapshot": asdict(snapshot),
                "edits": [asdict(edit) for edit in retiming_candidate.edits],
                "existing_service_dwell_edits": [
                    edit.model_dump() for edit in existing_service_dwell_edits
                ],
                "insertions": insertions,
                "work_schedule_group_id": first_service.work_schedule_group_id,
            },
        )
        current_baseline = tuple(sorted(validation.baseline_conflict_signatures))
        if baseline_signatures is not None and current_baseline != baseline_signatures:
            raise UnsupportedSnapshot("Baseline conflicts changed during candidate validation")
        baseline_signatures = current_baseline
        new_conflicts = sum(
            (Counter(validation.conflict_signatures)
             - Counter(validation.baseline_conflict_signatures)).values()
        )
        feasible = (
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflicts == 0
            and not any("cleanup failed" in reason.lower() for reason in validation.failure_reasons)
        )
        request_constraint_violations: list[str] = []
        request_constraint_unverified_departures: list[str] = []
        has_request_constraints = False
        for service, train_names in zip(request.services, pattern_train_names, strict=True):
            maximum = service.max_travel_time_seconds
            if maximum is None:
                continue
            has_request_constraints = True
            for train_name in train_names:
                simulated_time_ms = validation.inserted_travel_times_ms.get(train_name)
                if simulated_time_ms is None:
                    request_constraint_unverified_departures.append(train_name)
                elif simulated_time_ms > maximum * 1000:
                    request_constraint_violations.append(
                        f"{train_name} simulated end-to-end time is "
                        f"{round(simulated_time_ms / 1000)}s, above the {maximum}s maximum"
                    )
        if request_constraint_violations:
            request_constraint_status = "violated"
        elif request_constraint_unverified_departures:
            request_constraint_status = "unknown"
        elif has_request_constraints:
            request_constraint_status = "satisfied"
        else:
            request_constraint_status = "not_configured"
        feasible = feasible and request_constraint_status in {"not_configured", "satisfied"}
        station_call_timings, patronage_coverage, patronage_coverage_status = (
            _candidate_patronage_coverage(
                request.patronage_profile,
                insertions,
                validation.inserted_path_item_times_ms,
            )
        )
        departure_timings: list[ServiceDepartureTiming] = []
        for insertion in insertions:
            train_name = str(insertion["train_name"])
            travel_time_ms = validation.inserted_travel_times_ms.get(train_name)
            start_time_ms = insertion.get("start_time")
            if travel_time_ms is None or not isinstance(start_time_ms, int):
                continue
            departure_timings.append(
                ServiceDepartureTiming(
                    train_name=train_name,
                    departure_time=datetime.fromtimestamp(start_time_ms / 1000, tz=UTC),
                    arrival_time=datetime.fromtimestamp(
                        (start_time_ms + travel_time_ms) / 1000, tz=UTC
                    ),
                    travel_time_seconds=round(travel_time_ms / 1000),
                )
            )
        existing_service_timings: list[ExistingServiceTiming] = []
        edits_by_schedule_id = {
            edit.train_schedule_id: edit for edit in retiming_candidate.edits
        }
        dwell_edited_schedule_ids = {
            edit.train_schedule_id for edit in existing_service_dwell_edits
        }
        compared_existing_service_count = 0
        unavailable_existing_service_count = 0
        for schedule_id, source_departure_ms in metadata.start_times_ms.items():
            baseline_travel_time_ms = baseline_travel_times_ms.get(schedule_id)
            candidate_travel_time_ms = validation.existing_travel_times_ms.get(schedule_id)
            edit = edits_by_schedule_id.get(schedule_id)
            if baseline_travel_time_ms is None or candidate_travel_time_ms is None:
                unavailable_existing_service_count += 1
            else:
                compared_existing_service_count += 1
            if candidate_travel_time_ms is None:
                continue
            delta_ms = (
                candidate_travel_time_ms - baseline_travel_time_ms
                if baseline_travel_time_ms is not None
                else None
            )
            dwell_changed = schedule_id in dwell_edited_schedule_ids
            if edit is None and not dwell_changed and (delta_ms is None or delta_ms == 0):
                continue
            candidate_departure_ms = (
                edit.after_start_time_ms if edit is not None else source_departure_ms
            )
            existing_service_timings.append(
                ExistingServiceTiming(
                    train_schedule_id=schedule_id,
                    train_name=metadata.train_names.get(schedule_id),
                    departure_time=datetime.fromtimestamp(
                        candidate_departure_ms / 1000, tz=UTC
                    ),
                    arrival_time=datetime.fromtimestamp(
                        (candidate_departure_ms + candidate_travel_time_ms) / 1000,
                        tz=UTC,
                    ),
                    travel_time_seconds=round(candidate_travel_time_ms / 1000),
                    baseline_travel_time_seconds=(
                        round(baseline_travel_time_ms / 1000)
                        if baseline_travel_time_ms is not None
                        else None
                    ),
                    travel_time_delta_seconds=(
                        round(delta_ms / 1000) if delta_ms is not None else None
                    ),
                    arrival_time_delta_seconds=(
                        round(
                            (
                                candidate_departure_ms
                                + candidate_travel_time_ms
                                - source_departure_ms
                                - baseline_travel_time_ms
                            )
                            / 1000
                        )
                        if baseline_travel_time_ms is not None
                        else None
                    ),
                    retimed=edit is not None,
                    dwell_changed=dwell_changed,
                )
            )
        edited_schedule_ids = set(edits_by_schedule_id) | dwell_edited_schedule_ids
        runtime_deltas_ms = [
            validation.existing_travel_times_ms[schedule_id]
            - baseline_travel_times_ms[schedule_id]
            for schedule_id in edited_schedule_ids
            if schedule_id in validation.existing_travel_times_ms
            and schedule_id in baseline_travel_times_ms
        ]
        runtime_change_is_complete = len(runtime_deltas_ms) == len(edited_schedule_ids)
        simulated_runtime_change_seconds = (
            round(sum(runtime_deltas_ms) / 1000)
            if runtime_change_is_complete
            else None
        )
        candidates.append(
            CompleteTimetableCandidate(
                candidate_id=candidate_id,
                rank=1,
                patterns=pattern_results,
                proposed_schedules=insertions,
                edits=[
                    PlanCandidateEdit(
                        train_schedule_id=edit.train_schedule_id,
                        train_name=metadata.train_names.get(edit.train_schedule_id),
                        before_start_time_ms=edit.before_start_time_ms,
                        after_start_time_ms=edit.after_start_time_ms,
                        shift_seconds=edit.shift_seconds,
                    )
                    for edit in retiming_candidate.edits
                ],
                existing_service_dwell_edits=existing_service_dwell_edits,
                changed_service_count=len(edited_schedule_ids),
                total_shift_seconds=retiming_candidate.total_shift_seconds,
                retiming_priority_score=_retiming_priority_score(
                    retiming_candidate, request
                ),
                simulated_runtime_change_seconds=simulated_runtime_change_seconds,
                departure_timings=departure_timings,
                station_call_timings=station_call_timings,
                patronage_coverage_status=patronage_coverage_status,
                patronage_coverage=patronage_coverage,
                existing_service_timings=existing_service_timings,
                existing_service_effects_status=(
                    "partial"
                    if unavailable_existing_service_count
                    else "complete"
                ),
                source_service_count=len(metadata.start_times_ms),
                compared_existing_service_count=compared_existing_service_count,
                affected_existing_service_count=len(existing_service_timings),
                unavailable_existing_service_count=unavailable_existing_service_count,
                baseline_conflict_count=len(validation.baseline_conflict_signatures),
                new_conflict_count=new_conflicts,
                request_constraint_status=request_constraint_status,
                request_constraint_violations=request_constraint_violations,
                request_constraint_unverified_departures=request_constraint_unverified_departures,
                validation=PlanCandidateValidation(
                    pathfinding_valid=validation.pathfinding_valid,
                    simulation_valid=validation.simulation_valid,
                    conflict_count=validation.conflict_count,
                    baseline_conflict_count=len(validation.baseline_conflict_signatures),
                    new_conflict_count=new_conflicts,
                    failure_reasons=list(validation.failure_reasons),
                    conflicting_existing_service_ids=list(
                        validation.conflicting_existing_service_ids
                    ),
                    conflict_fingerprint=_conflict_fingerprint(validation.conflict_signatures),
                    baseline_conflict_fingerprint=_conflict_fingerprint(
                        validation.baseline_conflict_signatures
                    ),
                ),
                feasible=feasible,
            )
        )
    def sort_key(candidate: CompleteTimetableCandidate) -> tuple[object, ...]:
        if policy.objective == "minimize_total_shift":
            objective: float = candidate.total_shift_seconds
        elif policy.objective == "minimize_changed_services":
            objective = candidate.changed_service_count
        elif policy.objective == "minimize_weighted_shift":
            objective = float(candidate.retiming_priority_score or 0)
        else:
            objective = (
                float(candidate.simulated_runtime_change_seconds)
                if candidate.simulated_runtime_change_seconds is not None
                else float("inf")
            )
        coverage_score = (
            candidate.patronage_coverage.combined_event_coverage
            if candidate.patronage_coverage_status == "available"
            and candidate.patronage_coverage is not None
            else None
        )
        coverage_is_comparable = coverage_score is not None
        return (
            not candidate.feasible,
            candidate.new_conflict_count,
            (
                not coverage_is_comparable,
                -float(coverage_score)
                if coverage_score is not None
                else float("inf"),
            ) if request.candidate_order == "station_patronage_coverage" else (),
            objective,
            candidate.changed_service_count,
            candidate.total_shift_seconds,
            sum(pattern.departure_phase_seconds for pattern in candidate.patterns),
            tuple(pattern.route_alternative_index for pattern in candidate.patterns),
            candidate.candidate_id,
        )

    candidates.sort(key=sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return CompleteTimetablePlanResponse(
        request_id=request.request_id,
        snapshot=first_service.snapshot,
        candidates=candidates,
        selected_candidate_id=next(
            (candidate.candidate_id for candidate in candidates if candidate.feasible), None
        ),
        validation_budget=request.max_candidates,
        total_phase_combinations=total_combinations,
        total_route_combinations=total_route_combinations,
        total_route_phase_combinations=total_route_phase_combinations,
        total_stop_pattern_combinations=total_stop_pattern_combinations,
        total_route_phase_stop_combinations=total_route_phase_stop_combinations,
        retiming_candidate_count=len(retiming_candidates),
        total_candidate_combinations=total_candidate_combinations,
        search_complete=total_candidate_combinations <= request.max_candidates,
    )


async def plan_existing_service_retimings(
    request: ExistingServiceRetimingPlanRequest,
    client: OSRDClient,
) -> ExistingServiceRetimingPlanResponse:
    """Search and validate changes to existing services without adding departures."""
    snapshot = SnapshotRef(**request.snapshot.model_dump())
    metadata = await client.get_schedule_metadata(snapshot=snapshot)
    if metadata.timetable_type != "CALENDAR":
        raise UnsupportedSnapshot("Retiming-only planning currently requires a calendar timetable")
    policy = request.retiming
    if (
        policy.objective == "minimize_simulated_runtime_change"
        and policy.max_changed_services > 0
        and policy.max_shift_seconds > 0
    ):
        unsupported = sorted(
            set(policy.eligible_train_schedule_ids)
            & metadata.runtime_objective_unsupported_ids
        )
        if unsupported:
            raise ValueError(
                "The simulated running-time objective cannot compare paced services with "
                "timing-affecting exceptions. Remove these schedule IDs or choose another "
                f"objective: {', '.join(unsupported)}"
            )
    baseline_travel_times_ms = await client.get_schedule_travel_times(
        snapshot=snapshot,
        schedule_ids=list(metadata.start_times_ms),
    )
    retiming_candidates = generate_service_retiming_candidates(
        service_request_id=str(request.request_id),
        snapshot=request.snapshot,
        schedule_start_times_ms=metadata.start_times_ms,
        eligible_schedule_ids=policy.eligible_train_schedule_ids,
        protected_schedule_ids=policy.protected_train_schedule_ids,
        max_edits=policy.max_changed_services,
        max_shift_seconds=policy.max_shift_seconds,
        max_shift_seconds_by_schedule=policy.max_shift_seconds_by_schedule,
        max_advance_seconds_by_schedule=policy.max_advance_seconds_by_schedule,
        max_candidates=policy.max_candidates,
        step_seconds=policy.step_seconds,
    )
    if not retiming_candidates:
        raise ValueError("Retiming policy produced no candidate options")

    baseline_signatures: tuple[str, ...] | None = None
    candidates: list[CompleteTimetableCandidate] = []
    for retiming_candidate in retiming_candidates:
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload={
                "candidate_id": retiming_candidate.candidate_id,
                "snapshot": asdict(snapshot),
                "edits": [asdict(edit) for edit in retiming_candidate.edits],
                "existing_service_dwell_edits": [],
                "insertions": [],
                "work_schedule_group_id": request.work_schedule_group_id,
            },
        )
        current_baseline = tuple(sorted(validation.baseline_conflict_signatures))
        if baseline_signatures is not None and current_baseline != baseline_signatures:
            raise UnsupportedSnapshot("Baseline conflicts changed during candidate validation")
        baseline_signatures = current_baseline
        new_conflicts = sum(
            (Counter(validation.conflict_signatures)
             - Counter(validation.baseline_conflict_signatures)).values()
        )
        feasible = (
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflicts == 0
            and not any("cleanup failed" in reason.lower() for reason in validation.failure_reasons)
        )
        edits_by_id = {edit.train_schedule_id: edit for edit in retiming_candidate.edits}
        existing_timings: list[ExistingServiceTiming] = []
        compared_count = 0
        unavailable_count = 0
        for schedule_id, source_start_ms in metadata.start_times_ms.items():
            baseline_time_ms = baseline_travel_times_ms.get(schedule_id)
            candidate_time_ms = validation.existing_travel_times_ms.get(schedule_id)
            edit = edits_by_id.get(schedule_id)
            if baseline_time_ms is None or candidate_time_ms is None:
                unavailable_count += 1
            else:
                compared_count += 1
            if candidate_time_ms is None:
                continue
            delta_ms = (
                candidate_time_ms - baseline_time_ms
                if baseline_time_ms is not None else None
            )
            if edit is None and (delta_ms is None or delta_ms == 0):
                continue
            departure_ms = edit.after_start_time_ms if edit is not None else source_start_ms
            existing_timings.append(
                ExistingServiceTiming(
                    train_schedule_id=schedule_id,
                    train_name=metadata.train_names.get(schedule_id),
                    departure_time=datetime.fromtimestamp(departure_ms / 1000, tz=UTC),
                    arrival_time=datetime.fromtimestamp(
                        (departure_ms + candidate_time_ms) / 1000, tz=UTC
                    ),
                    travel_time_seconds=round(candidate_time_ms / 1000),
                    baseline_travel_time_seconds=(
                        round(baseline_time_ms / 1000) if baseline_time_ms is not None else None
                    ),
                    travel_time_delta_seconds=(
                        round(delta_ms / 1000) if delta_ms is not None else None
                    ),
                    arrival_time_delta_seconds=(
                        round(
                            (
                                departure_ms
                                + candidate_time_ms
                                - source_start_ms
                                - baseline_time_ms
                            )
                            / 1000
                        )
                        if baseline_time_ms is not None
                        else None
                    ),
                    retimed=edit is not None,
                )
            )
        edited_ids = set(edits_by_id)
        runtime_deltas = [
            validation.existing_travel_times_ms[schedule_id]
            - baseline_travel_times_ms[schedule_id]
            for schedule_id in edited_ids
            if schedule_id in validation.existing_travel_times_ms
            and schedule_id in baseline_travel_times_ms
        ]
        runtime_change = (
            round(sum(runtime_deltas) / 1000)
            if len(runtime_deltas) == len(edited_ids) else None
        )
        if validation.conflict_count is None:
            raise UnsupportedSnapshot("OSRD did not return a candidate conflict count")
        candidates.append(
            CompleteTimetableCandidate(
                candidate_id=retiming_candidate.candidate_id,
                rank=1,
                patterns=[],
                proposed_schedules=[],
                edits=[
                    PlanCandidateEdit(
                        train_schedule_id=edit.train_schedule_id,
                        train_name=metadata.train_names.get(edit.train_schedule_id),
                        before_start_time_ms=edit.before_start_time_ms,
                        after_start_time_ms=edit.after_start_time_ms,
                        shift_seconds=edit.shift_seconds,
                    )
                    for edit in retiming_candidate.edits
                ],
                changed_service_count=len(edited_ids),
                total_shift_seconds=retiming_candidate.total_shift_seconds,
                retiming_priority_score=(
                    float(sum(
                        abs(edit.shift_seconds) * policy.schedule_priority_weights[edit.train_schedule_id]
                        for edit in retiming_candidate.edits
                    )) if policy.objective == "minimize_weighted_shift" else None
                ),
                simulated_runtime_change_seconds=runtime_change,
                existing_service_timings=existing_timings,
                existing_service_effects_status="partial" if unavailable_count else "complete",
                source_service_count=len(metadata.start_times_ms),
                compared_existing_service_count=compared_count,
                affected_existing_service_count=len(existing_timings),
                unavailable_existing_service_count=unavailable_count,
                baseline_conflict_count=len(validation.baseline_conflict_signatures),
                new_conflict_count=new_conflicts,
                validation=PlanCandidateValidation(
                    pathfinding_valid=validation.pathfinding_valid,
                    simulation_valid=validation.simulation_valid,
                    conflict_count=validation.conflict_count,
                    baseline_conflict_count=len(validation.baseline_conflict_signatures),
                    new_conflict_count=new_conflicts,
                    failure_reasons=list(validation.failure_reasons),
                    conflicting_existing_service_ids=list(
                        validation.conflicting_existing_service_ids
                    ),
                    conflict_fingerprint=_conflict_fingerprint(validation.conflict_signatures),
                    baseline_conflict_fingerprint=_conflict_fingerprint(
                        validation.baseline_conflict_signatures
                    ),
                ),
                feasible=feasible,
            )
        )

    def objective(candidate: CompleteTimetableCandidate) -> tuple[object, ...]:
        if policy.objective == "minimize_changed_services":
            cost: float = candidate.changed_service_count
        elif policy.objective == "minimize_weighted_shift":
            cost = candidate.retiming_priority_score or 0.0
        elif policy.objective == "minimize_simulated_runtime_change":
            cost = (
                float(candidate.simulated_runtime_change_seconds)
                if candidate.simulated_runtime_change_seconds is not None
                else float("inf")
            )
        else:
            cost = candidate.total_shift_seconds
        return (not candidate.feasible, candidate.new_conflict_count, cost,
                candidate.changed_service_count, candidate.total_shift_seconds,
                candidate.candidate_id)

    candidates.sort(key=objective)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return ExistingServiceRetimingPlanResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        generated_candidate_count=len(retiming_candidates),
        evaluated_candidate_count=len(candidates),
        feasible_candidate_count=sum(candidate.feasible for candidate in candidates),
        selected_candidate_id=next(
            (candidate.candidate_id for candidate in candidates if candidate.feasible), None
        ),
        candidates=candidates,
    )
