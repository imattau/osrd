"""Generate deterministic OSRD replay episodes for the first training-data slice.

This is an offline data-collection tool. It creates isolated calendar timetables
and records OSRD's simulation and conflict outputs; it never applies an
operational plan to an existing timetable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

OSRD_COMMIT = "26d19cb55bf18a26d2f0da7a9fc586c310d3ecf0"
SCHEMA_SHA256 = "2164e74a522514e92c0b6fb6f252c7132e174b5a658e7aedf290fbf8e8d51b6b"
EPISODE_SCHEMA = "osrd-ops-replay-episode/v2"
SERVICE_PLAN_EPISODE_SCHEMA = "osrd-ops-replay-episode/v3"
EMPIRICAL_SERVICE_PLAN_EPISODE_SCHEMA = "osrd-ops-replay-episode/v4"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "data_contracts" / "osrd" / OSRD_COMMIT / "openapi.yaml"
)


class Editoast:
    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-osrd-skip-authz": "true",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Editoast {method} {path}: HTTP {exc.code}: {detail}") from exc


def make_schedule(
    *,
    index: int,
    seed: int,
    departure_ms: int,
    rolling_stock: str,
    start_track: str,
    start_offset: int,
    end_track: str,
    end_offset: int,
    dwell_seconds: int,
    intermediate_stops: list[tuple[str, int, int]] | None = None,
    intermediate_route_waypoints: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    stops = intermediate_stops or []
    route_waypoints = intermediate_route_waypoints or []
    path = [
        {
            "key": "origin",
            "location": {"type": "track_offset", "track": start_track, "offset": start_offset},
        }
    ]
    schedule = [{"at": "origin"}]
    for waypoint_index, (track, offset) in enumerate(route_waypoints, start=1):
        key = f"route-via-{waypoint_index}"
        path.append(
            {
                "key": key,
                "location": {"type": "track_offset", "track": track, "offset": offset},
            }
        )
        schedule.append({"at": key})
    for stop_index, (track, offset, stop_dwell_seconds) in enumerate(stops, start=1):
        key = f"stop-{stop_index}"
        path.append(
            {
                "key": key,
                "location": {"type": "track_offset", "track": track, "offset": offset},
            }
        )
        schedule.append({"at": key, "stop_for": f"PT{stop_dwell_seconds}S"})
    path.append(
        {
            "key": "destination",
            "location": {"type": "track_offset", "track": end_track, "offset": end_offset},
        }
    )
    schedule.append({"at": "destination", "stop_for": f"PT{dwell_seconds}S"})
    return {
        "train_name": f"replay-{seed}-{index:03d}",
        "rolling_stock_name": rolling_stock,
        "start_time": departure_ms,
        "comfort": "STANDARD",
        "constraint_distribution": "STANDARD",
        "initial_speed": 0,
        "labels": ["ops-intelligence", f"seed:{seed}"],
        "options": {"use_electrical_profiles": False},
        "path": path,
        "power_restrictions": [],
        "schedule": schedule,
    }


def _route_key(schedule: dict[str, Any]) -> tuple[str, int, str, int]:
    origin, destination = schedule["path"][0], schedule["path"][-1]
    return (
        str(origin["location"]["track"]),
        int(origin["location"]["offset"]),
        str(destination["location"]["track"]),
        int(destination["location"]["offset"]),
    )


def create_arm(
    client: Editoast,
    *,
    args: argparse.Namespace,
    arm: str,
    episode_index: int,
    schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    timetable = client.request("POST", "/timetable", {"timetable_type": "CALENDAR"})
    timetable_id = timetable["timetable_id"]
    schedule_set = client.request(
        "POST",
        "/train_schedule_sets",
        {
            "description": (f"ops-intelligence {arm} seed={args.seed} episode={episode_index}"),
            "name": f"ops-replay-{args.seed}-{episode_index}-{arm}",
            "published": False,
            "timetable_type": "CALENDAR",
        },
    )
    set_id = schedule_set["id"]
    client.request(
        "POST",
        f"/timetable/{timetable_id}/train_schedule_sets",
        {"train_schedule_set_ids": [set_id]},
    )
    created = client.request("POST", f"/train_schedule_sets/{set_id}/train_schedules", schedules)
    ids = [row["id"] for row in created]
    summaries = client.request(
        "POST",
        "/train_schedules/simulation_summary",
        {"infra_id": args.infra_id, "timetable_id": timetable_id, "ids": ids},
    )
    conflicts = client.request(
        "GET", f"/timetable/{timetable_id}/conflicts?infra_id={args.infra_id}"
    )
    return {
        "timetable_id": timetable_id,
        "train_schedule_set_id": set_id,
        "schedule_ids": ids,
        "simulation_summaries": summaries,
        "conflicts": conflicts,
    }


def pathfinding_request(
    *,
    rolling_stock: dict[str, Any],
    start_track: str,
    start_offset: int,
    end_track: str,
    end_offset: int,
    intermediate_stops: list[tuple[str, int, int]] | None = None,
    intermediate_route_waypoints: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Build OSRD pathfinding inputs from the same rolling stock and route as replay."""
    modes = rolling_stock.get("effort_curves", {}).get("modes", {})
    mode_name = rolling_stock.get("default_mode")
    mode = modes.get(mode_name, {})
    locations = (
        [(start_track, start_offset)]
        + list(intermediate_route_waypoints or [])
        + [(track, offset) for track, offset, _ in intermediate_stops or []]
        + [(end_track, end_offset)]
    )
    return {
        "path_items": [
            {
                "location": {"type": "track_offset", "track": track, "offset": offset},
                "can_backtrack": False,
            }
            for track, offset in locations
        ],
        "rolling_stock_loading_gauge": rolling_stock.get("loading_gauge", "G1"),
        "rolling_stock_is_thermal": not bool(mode.get("is_electric", False)),
        "rolling_stock_supported_electrifications": sorted(
            {
                source.get("electrical_profile")
                for source in rolling_stock.get("energy_sources", [])
                if source.get("electrical_profile")
            }
        ),
        "rolling_stock_supported_signaling_systems": [
            item["type"] if isinstance(item, dict) else item
            for item in rolling_stock.get("supported_signaling_systems", [])
        ],
        "rolling_stock_maximum_speed": float(rolling_stock.get("max_speed", 0)),
        "rolling_stock_length": round(float(rolling_stock.get("length", 0)) * 1000),
    }


def conflict_participants(conflicts: list[dict[str, Any]]) -> set[int]:
    return {
        int(train_id["train_schedule_id"])
        for conflict in conflicts
        for train_id in conflict.get("train_ids", [])
        if "train_schedule_id" in train_id
    }


def _load_empirical_profile(path_value: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not path_value:
        return None, None
    path = Path(path_value)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema") not in (
        "osrd-infrabel-origin-departure-delay-profile/v1",
        "osrd-empirical-origin-departure-delay-profile/v1",
        "osrd-empirical-origin-departure-delay-profile/v2",
    ):
        raise ValueError("Unsupported empirical departure-delay profile schema")
    delays = profile.get("delay_seconds")
    dwells = profile.get("stop_dwell_seconds")
    if (
        not isinstance(delays, list)
        or not delays
        or any(not isinstance(value, int) or isinstance(value, bool) for value in delays)
        or not isinstance(dwells, list)
        or not dwells
        or any(not isinstance(value, int) or isinstance(value, bool) for value in dwells)
    ):
        raise ValueError("Empirical profile must contain integer delay and dwell samples")
    return profile, hashlib.sha256(path.read_bytes()).hexdigest()


def collect(args: argparse.Namespace) -> Path:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Pinned Editoast schema is missing: {SCHEMA_PATH}")
    actual_schema_hash = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    if actual_schema_hash != SCHEMA_SHA256:
        raise RuntimeError(
            "Pinned Editoast schema hash mismatch: "
            f"expected {SCHEMA_SHA256}, got {actual_schema_hash}"
        )
    client = Editoast(args.editoast_url)
    rng = random.Random(args.seed)
    empirical_profile, empirical_profile_sha256 = _load_empirical_profile(
        args.empirical_delay_profile
    )
    # Matched action cohorts must share absolute baseline departures, not just
    # relative gaps. A seed-derived anchor keeps time-of-day and date stable
    # across separate collection invocations for each action alternative.
    anchor = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
        days=args.seed % 365, seconds=args.seed % 86400
    )
    now_ms = int(anchor.timestamp() * 1000)
    created_at = datetime.now(UTC).isoformat()
    infra = client.request("GET", f"/infra/{args.infra_id}")
    rolling_stock = client.request(
        "GET", f"/rolling_stock/name/{quote(args.rolling_stock, safe='')}"
    )
    primary_route = (args.start_track, args.start_offset, args.end_track, args.end_offset)
    background_routes = [
        tuple([str(parts[0]), int(parts[1]), str(parts[2]), int(parts[3])])
        for value in args.background_route
        if (parts := value.split(":")) and len(parts) == 4
    ]
    if len(background_routes) != len(args.background_route):
        raise ValueError("Background routes must use START_TRACK:START_OFFSET:END_TRACK:END_OFFSET")
    insertion_stops: list[tuple[str, int, int]] = []
    insertion_stop_details: list[dict[str, Any]] = []
    for stop_value in args.insertion_stop:
        parts = stop_value.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(
                "Insertion stops must use TRACK:OFFSET:DWELL_SECONDS[:OPERATIONAL_POINT_ID]"
            )
        stop = (str(parts[0]), int(parts[1]), int(parts[2]))
        if not stop[0] or stop[1] < 0 or stop[2] < 0:
            raise ValueError(
                "Insertion stop track, offset, and dwell must be valid nonnegative values"
            )
        insertion_stops.append(stop)
        insertion_stop_details.append(
            {
                "track": stop[0],
                "offset_mm": stop[1],
                "dwell_s": stop[2],
                "operational_point_id": parts[3] if len(parts) == 4 else None,
            }
        )
    insertion_route_waypoints: list[tuple[str, int]] = []
    for waypoint_value in args.insertion_route_via:
        parts = waypoint_value.split(":")
        if len(parts) != 2:
            raise ValueError("Insertion route waypoints must use TRACK:OFFSET")
        waypoint = (str(parts[0]), int(parts[1]))
        if not waypoint[0] or waypoint[1] < 0:
            raise ValueError("Insertion route waypoint track and offset must be valid")
        insertion_route_waypoints.append(waypoint)
    all_routes = sorted(set([primary_route, *background_routes]))
    route_paths: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for route in all_routes:
        route_response = client.request(
            "POST",
            f"/infra/{args.infra_id}/pathfinding/blocks",
            pathfinding_request(
                rolling_stock=rolling_stock,
                start_track=route[0],
                start_offset=route[1],
                end_track=route[2],
                end_offset=route[3],
                intermediate_stops=insertion_stops if route == primary_route else None,
            ),
        )
        if route_response.get("status") != "success" or not route_response.get("path"):
            raise RuntimeError(
                f"OSRD pathfinding failed for replay route {route}: {route_response}"
            )
        route_paths[route] = {
            **route_response["path"],
            "length": route_response.get("length", 0),
        }
    route_path = route_paths[primary_route]
    insertion_route_path = None
    if insertion_route_waypoints:
        route_response = client.request(
            "POST",
            f"/infra/{args.infra_id}/pathfinding/blocks",
            pathfinding_request(
                rolling_stock=rolling_stock,
                start_track=primary_route[0],
                start_offset=primary_route[1],
                end_track=primary_route[2],
                end_offset=primary_route[3],
                intermediate_route_waypoints=insertion_route_waypoints,
            ),
        )
        if route_response.get("status") != "success" or not route_response.get("path"):
            raise RuntimeError(f"OSRD pathfinding failed for insertion route: {route_response}")
        insertion_route_path = {
            **route_response["path"],
            "length": route_response.get("length", 0),
        }
    headway_options = args.headway_options_s or [args.headway_s]
    scenario_group = args.scenario_group or (
        f"infra-{args.infra_id}-{args.start_track}-{args.end_track}-headways-"
        + "-".join(str(value) for value in sorted(set(headway_options)))
    )
    episodes: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        baseline_schedules: list[dict[str, Any]] = []
        dwell_inputs: list[dict[str, Any]] = []
        episode_headway_s = rng.choice(headway_options)
        base_departure = now_ms + episode_index * 3_600_000
        for train_index in range(args.trains):
            train_route = rng.choice(background_routes) if background_routes else primary_route
            observed_departure_delay_s: int | None = None
            if train_index == 0:
                departure_shift_s = 0
            elif empirical_profile is not None:
                observed_departure_delay_s = rng.choice(empirical_profile["delay_seconds"])
                departure_shift_s = max(
                    -args.max_departure_shift_s,
                    min(args.max_departure_shift_s, observed_departure_delay_s),
                )
            else:
                departure_shift_s = rng.randint(
                    -args.max_departure_shift_s, args.max_departure_shift_s
                )
            observed_dwell_s: int | None = None
            if train_index == 0:
                dwell_s = 0
            elif empirical_profile is not None:
                observed_dwell_s = rng.choice(empirical_profile["stop_dwell_seconds"])
                dwell_s = min(args.max_dwell_s, observed_dwell_s)
            else:
                dwell_s = rng.randint(0, args.max_dwell_s)
            baseline_schedules.append(
                make_schedule(
                    index=train_index,
                    seed=args.seed + episode_index,
                    departure_ms=base_departure
                    + train_index * episode_headway_s * 1000
                    + departure_shift_s * 1000,
                    rolling_stock=args.rolling_stock,
                    start_track=train_route[0],
                    start_offset=train_route[1],
                    end_track=train_route[2],
                    end_offset=train_route[3],
                    dwell_seconds=dwell_s,
                )
            )
            perturbation: dict[str, Any] = {
                "train_index": train_index,
                "background_departure_shift_s": departure_shift_s,
                "destination_dwell_s": dwell_s,
            }
            if empirical_profile is not None:
                perturbation.update(
                    {
                        "perturbation_source": "empirical_origin_delay_and_planned_dwell/v1",
                        "observed_departure_delay_s": observed_departure_delay_s,
                        "observed_planned_dwell_s": observed_dwell_s,
                    }
                )
            dwell_inputs.append(perturbation)

        treatment_schedules = json.loads(json.dumps(baseline_schedules))
        incident_train_index = rng.randrange(args.trains)
        insertion_departure_ms: int | None = None
        insertion_phase_s = 0
        if args.intervention_type in ("service-insertion", "service-plan"):
            incident_train_index = args.trains
            if args.insertion_frequency_s:
                if args.insertion_phase_s is not None:
                    insertion_phase_s = args.insertion_phase_s
                elif args.randomize_insertion_phase:
                    phase_limit_s = min(
                        episode_headway_s,
                        args.insertion_frequency_s,
                        args.insertion_window_s,
                    )
                    insertion_phase_s = rng.randrange(phase_limit_s)
                insertion_departure_ms = (
                    baseline_schedules[0]["start_time"] + insertion_phase_s * 1000
                )
                insertion_count = (
                    args.insertion_window_s - insertion_phase_s + args.insertion_frequency_s - 1
                ) // args.insertion_frequency_s
            else:
                insertion_departure_ms = rng.randint(
                    baseline_schedules[0]["start_time"], baseline_schedules[-1]["start_time"]
                )
                insertion_count = 1
            for insertion_index in range(insertion_count):
                treatment_schedules.append(
                    make_schedule(
                        index=incident_train_index + insertion_index,
                        seed=args.seed + episode_index,
                        departure_ms=(
                            insertion_departure_ms
                            + insertion_index * args.insertion_frequency_s * 1000
                            if args.insertion_frequency_s
                            else insertion_departure_ms
                        ),
                        rolling_stock=args.rolling_stock,
                        start_track=primary_route[0],
                        start_offset=primary_route[1],
                        end_track=primary_route[2],
                        end_offset=primary_route[3],
                        dwell_seconds=args.insertion_dwell_s,
                        intermediate_stops=insertion_stops,
                        intermediate_route_waypoints=insertion_route_waypoints,
                    )
                )
            incident_delay_s = 0
        elif args.intervention_type == "service-retime":
            incident_delay_s = rng.choice(
                [value for value in range(-args.max_retime_s, args.max_retime_s + 1) if value]
            )
        else:
            incident_delay_s = rng.randint(args.min_incident_delay_s, args.max_incident_delay_s)
        planned_retimings: list[dict[str, int]] = []
        if args.intervention_type == "service-plan":
            if args.plan_retiming_action:
                if args.plan_retiming_action == ["none"]:
                    retiming_actions: list[tuple[int, int]] = []
                else:
                    retiming_actions = []
                    for value in args.plan_retiming_action:
                        if value == "none":
                            raise ValueError("'none' cannot be combined with retiming actions")
                        parts = value.split(":")
                        if len(parts) != 2:
                            raise ValueError(
                                "Plan retiming actions must use TRAIN_INDEX:SHIFT_SECONDS"
                            )
                        train_index, shift_seconds = map(int, parts)
                        if not 0 <= train_index < args.trains:
                            raise ValueError(
                                f"Plan retiming train index is outside 0..{args.trains - 1}"
                            )
                        if not 0 < abs(shift_seconds) <= args.max_retime_s:
                            raise ValueError(
                                "Plan retiming shift must be nonzero and within --max-retime-s"
                            )
                        retiming_actions.append((train_index, shift_seconds))
                    if len({index for index, _ in retiming_actions}) != len(retiming_actions):
                        raise ValueError("A plan may retime each existing service only once")
            else:
                retime_count = min(args.max_plan_retimed_services, args.trains)
                retiming_actions = [
                    (
                        retime_index,
                        rng.choice(
                            [
                                value
                                for value in range(-args.max_retime_s, args.max_retime_s + 1)
                                if value
                            ]
                        ),
                    )
                    for retime_index in sorted(rng.sample(range(args.trains), retime_count))
                ]
            for retime_index, shift_seconds in retiming_actions:
                treatment_schedules[retime_index]["start_time"] += shift_seconds * 1000
                planned_retimings.append(
                    {"train_index": retime_index, "departure_shift_s": shift_seconds}
                )
        elif args.intervention_type not in ("service-insertion", "service-plan"):
            treatment_schedules[incident_train_index]["start_time"] += incident_delay_s * 1000

        resource_paths_by_train = [
            route_paths[_route_key(schedule)] for schedule in treatment_schedules
        ]
        if insertion_route_path is not None:
            for index in range(len(baseline_schedules), len(resource_paths_by_train)):
                resource_paths_by_train[index] = insertion_route_path

        baseline_arm = create_arm(
            client,
            args=args,
            arm="baseline",
            episode_index=episode_index,
            schedules=baseline_schedules,
        )
        treatment_arm = create_arm(
            client,
            args=args,
            arm="treatment",
            episode_index=episode_index,
            schedules=treatment_schedules,
        )
        baseline_ids = baseline_arm["schedule_ids"]
        treatment_ids = treatment_arm["schedule_ids"]
        baseline_summaries = baseline_arm["simulation_summaries"]
        treatment_summaries = treatment_arm["simulation_summaries"]
        baseline_conflicted = conflict_participants(baseline_arm["conflicts"])
        treatment_conflicted = conflict_participants(treatment_arm["conflicts"])

        simulation_labels: dict[str, dict[str, Any]] = {}
        paired_labels: list[dict[str, Any]] = []
        for index, treatment_id in enumerate(treatment_ids):
            base_id = baseline_ids[index] if index < len(baseline_ids) else None
            base_result = (
                baseline_summaries.get(str(base_id), {}).get("train_schedule", {})
                if base_id is not None
                else {}
            )
            result = treatment_summaries.get(str(treatment_id), {}).get("train_schedule", {})
            baseline_arrival_ms = (
                baseline_schedules[index]["start_time"]
                + (base_result.get("path_item_times_final") or [None])[-1]
                if base_id is not None and base_result.get("path_item_times_final")
                else None
            )
            treatment_arrival_ms = (
                treatment_schedules[index]["start_time"]
                + (result.get("path_item_times_final") or [None])[-1]
                if result.get("path_item_times_final")
                else None
            )
            base_conflict = base_id in baseline_conflicted if base_id is not None else False
            treatment_conflict = treatment_id in treatment_conflicted
            paired_labels.append(
                {
                    "train_index": index,
                    "baseline_schedule_id": base_id,
                    "treatment_schedule_id": treatment_id,
                    "baseline_status": base_result.get("status", "unknown"),
                    "treatment_status": result.get("status", "unknown"),
                    "baseline_arrival_ms": baseline_arrival_ms,
                    "treatment_arrival_ms": treatment_arrival_ms,
                    "arrival_delta_ms": (
                        treatment_arrival_ms - baseline_arrival_ms
                        if baseline_arrival_ms is not None and treatment_arrival_ms is not None
                        else None
                    ),
                    "travel_time_delta_ms": (
                        result["time"] - base_result["time"]
                        if result.get("time") is not None and base_result.get("time") is not None
                        else None
                    ),
                    "energy_delta_kwh": (
                        result["energy_consumption"] - base_result["energy_consumption"]
                        if result.get("energy_consumption") is not None
                        and base_result.get("energy_consumption") is not None
                        else None
                    ),
                    "baseline_conflict": base_conflict,
                    "treatment_conflict": treatment_conflict,
                    "new_conflict": treatment_conflict and not base_conflict,
                }
            )
            simulation_labels[str(treatment_id)] = {
                "status": result.get("status", "unknown"),
                "travel_time_ms": result.get("time"),
                "energy_consumption_kwh": result.get("energy_consumption"),
                "path_item_times_final_ms": result.get("path_item_times_final"),
                "path_item_times_base_ms": result.get("path_item_times_base"),
            }
        episode_schema = EPISODE_SCHEMA
        if empirical_profile is not None:
            episode_schema = EMPIRICAL_SERVICE_PLAN_EPISODE_SCHEMA
        elif args.intervention_type == "service-plan":
            episode_schema = SERVICE_PLAN_EPISODE_SCHEMA
        episodes.append(
            {
                "schema": episode_schema,
                "episode_id": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        "osrd-replay:"
                        + hashlib.sha256(
                            json.dumps(
                                {
                                    "schema": episode_schema,
                                    "scenario_group": scenario_group,
                                    "seed": args.seed,
                                    "profile": empirical_profile_sha256 or "uniform",
                                    "episode_index": episode_index,
                                    "action": {
                                        "frequency_s": args.insertion_frequency_s or 0,
                                        "phase_s": insertion_phase_s,
                                        "operating_window_s": args.insertion_window_s,
                                        "stops": insertion_stop_details,
                                        "route_via": insertion_route_waypoints,
                                        "retimings": planned_retimings,
                                    },
                                },
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest(),
                    )
                ),
                "scenario_group": scenario_group,
                "created_at": created_at,
                "seed": args.seed,
                "episode_index": episode_index,
                "snapshot": {
                    "osrd_commit": OSRD_COMMIT,
                    "openapi_sha256": SCHEMA_SHA256,
                    "infra_id": args.infra_id,
                    "baseline_timetable_id": baseline_arm["timetable_id"],
                    "baseline_train_schedule_set_id": baseline_arm["train_schedule_set_id"],
                    "treatment_timetable_id": treatment_arm["timetable_id"],
                    "treatment_train_schedule_set_id": treatment_arm["train_schedule_set_id"],
                    "infra_version": infra.get("version"),
                    "infra_generated_version": infra.get("generated_version"),
                    "rolling_stock_id": rolling_stock.get("id"),
                    "rolling_stock_version": rolling_stock.get("version"),
                },
                "inputs": {
                    "scenario": {
                        "scenario_group": scenario_group,
                        "track_offset_unit": "mm",
                        "departure_anchor": "seed-derived-utc-2026-01-01/v1",
                        "headway_s": episode_headway_s,
                        "headway_options_s": headway_options,
                        "infra_id": args.infra_id,
                        "start_track": args.start_track,
                        "start_offset": args.start_offset,
                        "end_track": args.end_track,
                        "end_offset": args.end_offset,
                        "rolling_stock": args.rolling_stock,
                        "empirical_delay_profile_sha256": empirical_profile_sha256,
                    },
                    "schedule_payloads": treatment_schedules,
                    "baseline_schedule_payloads": baseline_schedules,
                    "background_perturbations": dwell_inputs,
                    # OSRD-derived route resources are collected before either arm is simulated.
                    # They therefore describe topology, never a treatment outcome.
                    "resource_path": insertion_route_path or route_path,
                    "resource_paths_by_train": resource_paths_by_train,
                    "incident": {
                        "type": args.intervention_type,
                        "train_index": incident_train_index,
                        "delay_s": incident_delay_s,
                    },
                    "action": {
                        "type": args.intervention_type,
                        "train_index": incident_train_index,
                        "departure_shift_s": incident_delay_s,
                        "departure_time_ms": insertion_departure_ms,
                        "frequency_s": args.insertion_frequency_s or 0,
                        "departure_phase_s": insertion_phase_s,
                        "operating_window_s": args.insertion_window_s,
                        "requested_service_count": len(treatment_schedules)
                        - len(baseline_schedules),
                        "stop_count": len(insertion_stops),
                        "stops": insertion_stop_details,
                        "mean_dwell_s": (
                            sum(stop[2] for stop in insertion_stops) / len(insertion_stops)
                            if insertion_stops
                            else args.insertion_dwell_s
                        ),
                        "route_via": [
                            {"track": track, "offset_mm": offset}
                            for track, offset in insertion_route_waypoints
                        ],
                        "retimings": planned_retimings,
                    },
                },
                "outputs": {
                    "schedule_ids": treatment_ids,
                    "simulation_summaries": treatment_summaries,
                    "conflicts": treatment_arm["conflicts"],
                    "paired": {
                        "baseline": baseline_arm,
                        "treatment": treatment_arm,
                    },
                },
                "labels": {
                    "simulation_failed_train_ids": [
                        int(k)
                        for k, result in treatment_summaries.items()
                        if result.get("train_schedule", {}).get("status") not in (None, "success")
                    ],
                    "simulation_by_train_schedule_id": simulation_labels,
                    "paired_by_train_index": paired_labels,
                    "conflict_count": len(treatment_arm["conflicts"]),
                    "has_conflicts": bool(treatment_arm["conflicts"]),
                    "new_conflict_count": sum(
                        int(label["new_conflict"]) for label in paired_labels
                    ),
                },
            }
        )
        print(
            f"episode {episode_index + 1}/{args.episodes}: "
            f"scenario={scenario_group}, trains={len(treatment_ids)}, "
            f"new_conflicts={sum(int(label['new_conflict']) for label in paired_labels)}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "osrd-ops-replay-manifest/v1",
        "episode_schema": (
            EMPIRICAL_SERVICE_PLAN_EPISODE_SCHEMA
            if empirical_profile is not None and args.intervention_type == "service-plan"
            else SERVICE_PLAN_EPISODE_SCHEMA
            if args.intervention_type == "service-plan"
            else EPISODE_SCHEMA
        ),
        "created_at": created_at,
        "seed": args.seed,
        "episode_count": len(episodes),
        "episode_file": "episodes.jsonl",
        "format": "jsonl",
        "osrd_commit": OSRD_COMMIT,
        "openapi_sha256": SCHEMA_SHA256,
        "parameters": vars(args),
        "empirical_delay_profile": (
            {
                "path": args.empirical_delay_profile,
                "sha256": empirical_profile_sha256,
                "profile_schema": empirical_profile["schema"],
                "source_months": empirical_profile.get("source_months", []),
                "source_month_sha256": empirical_profile.get("source_month_sha256", {}),
                "source_periods": empirical_profile.get("source_periods", []),
                "source_files": empirical_profile.get("source_files", []),
                "license": empirical_profile["license"],
                "attribution": empirical_profile["attribution"],
                "sampling_unit": empirical_profile.get("sampling_unit"),
            }
            if empirical_profile is not None
            else None
        ),
        "label_definitions": {
            "simulation_failed_train_ids": (
                "IDs whose summary status is not success; missing status is unknown."
            ),
            "conflict_count": "Number of timetable conflict records returned by Editoast.",
            "has_conflicts": "Whether conflict_count is greater than zero.",
            "arrival_delta_ms": "Treatment minus matched baseline simulated arrival time.",
            "new_conflict": "Train is conflict-involved in treatment but not in baseline.",
            "action": (
                "Synthetic service insertion, optional frequency/window expansion, signed "
                "single-service retiming, or coupled insertion and explicitly recorded "
                "existing-service retimings in the treatment arm."
            ),
            "inserted_stop_pattern": (
                "Ordered OSRD track-offset waypoints and dwell seconds "
                "requested for inserted services."
            ),
            "added_service": "Treatment-only service; no paired baseline arrival delta exists.",
        },
        "paired_arms": True,
        "scenario_group": scenario_group,
    }
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, sort_keys=True, separators=(",", ":")) + "\n")
    manifest["episodes_sha256"] = hashlib.sha256(
        (output / "episodes.jsonl").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editoast-url", default="http://localhost:4000/api")
    parser.add_argument("--infra-id", type=int, required=True)
    parser.add_argument("--rolling-stock", required=True)
    parser.add_argument("--start-track", required=True)
    parser.add_argument("--start-offset", type=int, required=True)
    parser.add_argument("--end-track", required=True)
    parser.add_argument("--end-offset", type=int, required=True)
    parser.add_argument(
        "--background-route",
        action="append",
        default=[],
        metavar="START:OFFSET:END:OFFSET",
        help=(
            "Optional route for background services, formatted as "
            "START_TRACK:START_OFFSET:END_TRACK:END_OFFSET. Repeat to mix corridors."
        ),
    )
    parser.add_argument("--output", default="data/replays/run-001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--trains", type=int, default=8)
    parser.add_argument("--headway-s", type=int, default=240)
    parser.add_argument("--headway-options-s", type=int, nargs="+")
    parser.add_argument("--max-departure-shift-s", type=int, default=180)
    parser.add_argument("--max-dwell-s", type=int, default=120)
    parser.add_argument("--min-incident-delay-s", type=int, default=60)
    parser.add_argument("--max-incident-delay-s", type=int, default=300)
    parser.add_argument(
        "--intervention-type",
        choices=("incident-delay", "service-retime", "service-insertion", "service-plan"),
        default="incident-delay",
        help=(
            "Use an incident delay, signed retiming, service insertion, or insertion "
            "with coupled retimings."
        ),
    )
    parser.add_argument("--max-retime-s", type=int, default=300)
    parser.add_argument("--max-plan-retimed-services", type=int, default=2)
    parser.add_argument(
        "--plan-retiming-action",
        action="append",
        default=[],
        metavar="TRAIN_INDEX:SHIFT_SECONDS",
        help=(
            "Set an explicit coupled retiming for a service-plan episode; repeat for multiple "
            "services. Use 'none' for the unretimed alternative. Without this option, retimings "
            "remain randomly sampled."
        ),
    )
    parser.add_argument(
        "--insertion-frequency-s",
        type=int,
        help="Optional frequency for repeated inserted services; requires an operating window.",
    )
    parser.add_argument(
        "--randomize-insertion-phase",
        action="store_true",
        help=(
            "Sample the first inserted departure phase uniformly within one headway and "
            "frequency interval."
        ),
    )
    parser.add_argument(
        "--insertion-phase-s",
        type=int,
        help="Fixed offset of the first repeated departure from the operating-window start.",
    )
    parser.add_argument("--insertion-window-s", type=int, default=0)
    parser.add_argument("--insertion-dwell-s", type=int, default=0)
    parser.add_argument(
        "--insertion-stop",
        action="append",
        default=[],
        metavar="TRACK:OFFSET_MM:DWELL_SECONDS",
        help=(
            "Intermediate stop for each inserted train; repeat in route order. Offset is "
            "millimetres. Optional fourth field records an operational point ID: "
            "TRACK:OFFSET_MM:DWELL[:OP_ID]."
        ),
    )
    parser.add_argument(
        "--insertion-route-via",
        action="append",
        default=[],
        metavar="TRACK:OFFSET",
        help=(
            "Constrain inserted services through ordered track offsets. Repeat to record a "
            "route-choice treatment; OSRD pathfinding validates the combined ordered path."
        ),
    )
    parser.add_argument("--scenario-group")
    parser.add_argument(
        "--empirical-delay-profile",
        "--infrabel-delay-profile",
        dest="empirical_delay_profile",
        help=(
            "Sample baseline departure delays and planned stop dwells from a checksum-bound "
            "Infrabel empirical profile."
        ),
    )
    args = parser.parse_args()
    if (
        min(args.episodes, args.trains, args.headway_s) <= 0
        or min(args.max_departure_shift_s, args.max_dwell_s) < 0
        or args.min_incident_delay_s <= 0
        or args.max_incident_delay_s < args.min_incident_delay_s
        or args.max_retime_s <= 0
        or args.max_plan_retimed_services < 1
        or args.insertion_window_s < 0
        or args.insertion_dwell_s < 0
    ):
        parser.error(
            "episodes, trains, and headway must be positive; perturbation maxima cannot be negative"
        )
    if args.headway_options_s and min(args.headway_options_s) <= 0:
        parser.error("headway options must all be positive")
    if args.insertion_frequency_s is not None:
        if args.intervention_type not in ("service-insertion", "service-plan"):
            parser.error("insertion frequency requires service-insertion or service-plan episodes")
        if args.insertion_frequency_s <= 0 or args.insertion_window_s <= 0:
            parser.error("insertion frequency and operating window must be positive")
        insertion_count = (
            args.insertion_window_s + args.insertion_frequency_s - 1
        ) // args.insertion_frequency_s
        if insertion_count > 50:
            parser.error("frequency request would insert more than 50 services per episode")
    if args.insertion_stop and args.intervention_type not in ("service-insertion", "service-plan"):
        parser.error("insertion stops require service-insertion or service-plan episodes")
    if args.insertion_route_via and args.intervention_type not in (
        "service-insertion",
        "service-plan",
    ):
        parser.error("insertion route waypoints require service-insertion or service-plan episodes")
    if args.randomize_insertion_phase and args.insertion_frequency_s is None:
        parser.error("randomized insertion phase requires a repeated-service frequency")
    if args.insertion_phase_s is not None:
        if args.insertion_frequency_s is None:
            parser.error("a fixed insertion phase requires a repeated-service frequency")
        minimum_headway_s = min(args.headway_options_s or [args.headway_s])
        if (
            not 0
            <= args.insertion_phase_s
            < min(minimum_headway_s, args.insertion_frequency_s, args.insertion_window_s)
        ):
            parser.error(
                "insertion phase must be nonnegative and less than the minimum headway, "
                "frequency, and operating-window duration"
            )
        if args.randomize_insertion_phase:
            parser.error("choose either a fixed or randomized insertion phase")
    print(f"Wrote replay data to {collect(args)}")


if __name__ == "__main__":
    main()
