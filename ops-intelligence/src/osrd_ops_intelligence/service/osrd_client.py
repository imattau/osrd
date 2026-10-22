"""Snapshot-bound Editoast adapter for isolated candidate validation."""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, cast
from urllib.parse import quote

MAX_ROUTE_ENDPOINT_COMBINATIONS = 25
MAX_CONCURRENT_PATHFINDING_REQUESTS = 5
MAX_SCHEDULE_SIMULATION_BATCH = 200


@dataclass(frozen=True)
class SnapshotRef:
    infra_id: str
    infra_version: str
    timetable_id: str
    timetable_revision: str
    osrd_version: str


@dataclass(frozen=True)
class ScheduleMetadata:
    """Planner-facing schedule fields read from one bound timetable snapshot."""

    start_times_ms: dict[str, int]
    train_names: dict[str, str]
    schedule_forms_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    timetable_type: str = "CALENDAR"
    runtime_objective_unsupported_ids: frozenset[str] = frozenset()
    max_hourly_shift_seconds_by_schedule: dict[str, int] = field(default_factory=dict)
    max_hourly_advance_seconds_by_schedule: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayBaselineContext:
    """Snapshot-bound source inputs needed to build outcome-free replay episodes."""

    snapshot: SnapshotRef
    schedule_ids: list[int]
    schedule_payloads: list[dict[str, Any]]
    simulation_summaries: dict[str, Any]
    conflicts: list[dict[str, Any]]


@dataclass(frozen=True)
class CandidateValidation:
    pathfinding_valid: bool
    simulation_valid: bool
    conflict_count: int | None
    failure_reasons: tuple[str, ...] = ()
    conflict_signatures: tuple[str, ...] = ()
    baseline_conflict_signatures: tuple[str, ...] = ()
    inserted_travel_times_ms: dict[str, int] = field(default_factory=dict)
    existing_travel_times_ms: dict[str, int] = field(default_factory=dict)
    inserted_path_item_times_ms: dict[str, list[int]] = field(default_factory=dict)
    conflicting_existing_service_ids: tuple[str, ...] = ()

    @property
    def validated(self) -> bool:
        return (
            self.pathfinding_valid
            and self.simulation_valid
            and self.conflict_count == 0
            and not self.failure_reasons
        )


class OSRDClient(Protocol):
    """Minimal async interface needed by the future planning orchestrator."""

    async def describe_snapshot(self, *, infra_id: str, timetable_id: str) -> SnapshotRef:
        """Return a current binding for an infrastructure and timetable pair."""
        ...

    async def get_schedule_start_times(self, *, snapshot: SnapshotRef) -> dict[str, int]:
        """Load train IDs and start times after checking the exact snapshot binding."""
        ...

    async def get_schedule_metadata(self, *, snapshot: SnapshotRef) -> ScheduleMetadata:
        """Load bounded-shift and naming metadata from one exact timetable snapshot."""
        ...

    async def get_schedule_travel_times(
        self, *, snapshot: SnapshotRef, schedule_ids: list[str]
    ) -> dict[str, int]:
        """Return successful OSRD baseline travel times for selected schedules."""
        ...

    async def get_schedule_resource_paths(
        self, *, snapshot: SnapshotRef, schedules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return OSRD-derived topology resource paths for ordered schedule forms."""
        ...

    async def get_replay_baseline_context(
        self, *, snapshot: SnapshotRef
    ) -> ReplayBaselineContext:
        """Read schedule forms and OSRD baseline evidence from one exact snapshot."""
        ...

    async def get_route_alternatives(
        self,
        *,
        snapshot: SnapshotRef,
        starting: dict[str, Any],
        ending: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return OSRD's shortest topological paths between two track offsets."""
        ...

    async def validate_candidate(
        self,
        *,
        snapshot: SnapshotRef,
        candidate_payload: dict[str, object],
    ) -> CandidateValidation:
        """Validate a complete candidate timetable through OSRD."""
        ...


class EditoastError(RuntimeError):
    """Raised when Editoast rejects a validation operation."""


class UnsupportedSnapshot(ValueError):
    """Raised when this prototype cannot preserve the source timetable semantics."""


_TRAIN_SCHEDULE_FIELDS = (
    "category",
    "comfort",
    "constraint_distribution",
    "initial_speed",
    "labels",
    "margins",
    "options",
    "paced",
    "path",
    "power_restrictions",
    "rolling_stock_name",
    "schedule",
    "speed_limit_tag",
    "start_time",
    "train_name",
)
_EXCEPTION_CHANGE_GROUP_FIELDS = (
    "train_name",
    "rolling_stock",
    "rolling_stock_category",
    "labels",
    "speed_limit_tag",
    "start_time",
    "constraint_distribution",
    "initial_speed",
    "options",
    "path_and_schedule",
)


def _exception_change_groups(exception: dict[str, Any]) -> dict[str, Any]:
    nested = exception.get("change_groups")
    if nested is not None:
        if not isinstance(nested, dict):
            raise UnsupportedSnapshot("Paced exception change groups are invalid")
        return cast(dict[str, Any], nested)
    known_fields = set(_EXCEPTION_CHANGE_GROUP_FIELDS) | {
        "id",
        "key",
        "occurrence_index",
        "disabled",
    }
    unknown_fields = set(exception) - known_fields
    if unknown_fields:
        raise UnsupportedSnapshot(
            f"Paced exception contains unsupported fields: {sorted(unknown_fields)}"
        )
    return {key: exception[key] for key in _EXCEPTION_CHANGE_GROUP_FIELDS if key in exception}


def _schedule_form(schedule: dict[str, Any]) -> dict[str, Any]:
    form = {key: schedule[key] for key in _TRAIN_SCHEDULE_FIELDS if key in schedule}
    required = {"train_name", "rolling_stock_name", "start_time", "path", "constraint_distribution"}
    missing = required - form.keys()
    if missing:
        raise UnsupportedSnapshot(f"Train schedule is missing required fields: {sorted(missing)}")
    paced = form.get("paced")
    if paced is not None:
        if not isinstance(paced, dict):
            raise UnsupportedSnapshot("Paced train data has an invalid shape")
        paced_data = cast(dict[str, Any], paced)
        exceptions = paced_data.get("exceptions", [])
        if not isinstance(exceptions, list):
            raise UnsupportedSnapshot("Paced train exceptions have an invalid shape")
        exception_rows = cast(list[object], exceptions)
        if not all(isinstance(exception, dict) for exception in exception_rows):
            raise UnsupportedSnapshot("Paced train exceptions have an invalid shape")
    return form


def _duration_milliseconds(value: object) -> int:
    """Parse the day/time subset of ISO-8601 durations used by paced intervals."""
    if not isinstance(value, str):
        raise UnsupportedSnapshot("Paced train interval must be an ISO 8601 duration")
    match = re.fullmatch(
        r"P(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?"
        r"(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?",
        value,
    )
    if match is None:
        raise UnsupportedSnapshot("Paced train interval has an unsupported duration")
    try:
        days, hours, minutes, seconds = (
            Decimal(part or "0") for part in match.groups()
        )
    except InvalidOperation as error:
        raise UnsupportedSnapshot("Paced train interval has an invalid duration") from error
    milliseconds = (days * 86_400 + hours * 3_600 + minutes * 60 + seconds) * 1000
    if milliseconds <= 0 or milliseconds != milliseconds.to_integral_value():
        raise UnsupportedSnapshot("Paced train interval must be a positive whole millisecond")
    return int(milliseconds)


def _stop_duration_seconds(value: object) -> int:
    if value is None or value in ("P0D", "PT0S"):
        return 0
    milliseconds = _duration_milliseconds(value)
    if milliseconds % 1000 or milliseconds > 3_600_000:
        raise UnsupportedSnapshot("Existing station dwell must be whole seconds up to one hour")
    return milliseconds // 1000


def compute_timetable_revision(timetable_type: str, schedules: list[dict[str, Any]]) -> str:
    """Hash normalized schedule forms and their source identities for snapshot binding."""

    normalized: list[dict[str, Any]] = [
        {
            "train_schedule_id": int(schedule["id"]),
            "train_schedule_set_id": int(schedule["train_schedule_set_id"]),
            "form": _schedule_form(schedule),
        }
        for schedule in schedules
    ]
    normalized.sort(key=lambda row: int(row["train_schedule_id"]))
    payload = {"timetable_type": timetable_type, "schedules": normalized}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"sha256:{digest}"


def _conflict_signatures(
    conflicts: list[dict[str, Any]], schedule_id_map: dict[str, str]
) -> tuple[str, ...]:
    """Canonicalize OSRD conflicts by participants, zones, and occupied interval."""
    signatures: list[str] = []
    for conflict in conflicts:
        work_schedule_ids = conflict.get("work_schedule_ids", [])
        if not isinstance(work_schedule_ids, list) or any(
            not isinstance(schedule_id, int) or isinstance(schedule_id, bool)
            for schedule_id in work_schedule_ids
        ):
            raise EditoastError("OSRD returned invalid work-schedule conflict IDs")
        participants = []
        for train in conflict.get("train_ids", []):
            if not isinstance(train, dict) or "train_schedule_id" not in train:
                raise EditoastError("OSRD returned an invalid conflict participant")
            schedule_id = str(train["train_schedule_id"])
            participants.append(
                {
                    "schedule": schedule_id_map.get(schedule_id, f"unknown:{schedule_id}"),
                    "index": train.get("index", 0),
                }
            )
        canonical = {
            "type": conflict.get("conflict_type"),
            "participants": sorted(participants, key=lambda row: (row["schedule"], row["index"])),
            "work_schedules": sorted(work_schedule_ids),
            "zones": sorted(
                str(requirement["zone"])
                for requirement in conflict.get("requirements", [])
                if isinstance(requirement, dict) and requirement.get("zone") is not None
            ),
            "start_time": conflict.get("start_time"),
            "duration": conflict.get("duration"),
        }
        signatures.append(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    return tuple(sorted(signatures))


def _new_conflict_existing_service_ids(
    conflicts: list[dict[str, Any]],
    baseline_conflict_signatures: tuple[str, ...],
    candidate_id_map: dict[str, str],
) -> tuple[str, ...]:
    """Return source schedule IDs participating in conflicts absent from the baseline."""
    baseline_remaining = Counter(baseline_conflict_signatures)
    affected_ids: set[str] = set()
    for conflict in conflicts:
        signature = _conflict_signatures([conflict], candidate_id_map)[0]
        if baseline_remaining[signature]:
            baseline_remaining[signature] -= 1
            continue
        for train in conflict.get("train_ids", []):
            schedule_label = candidate_id_map.get(str(train["train_schedule_id"]), "")
            if schedule_label.startswith("source:"):
                affected_ids.add(schedule_label.removeprefix("source:"))
    return tuple(sorted(affected_ids))


def _conflicts_path(
    snapshot: SnapshotRef, timetable_id: int, work_schedule_group_id: object = None
) -> str:
    path = f"/timetable/{timetable_id}/conflicts?infra_id={int(snapshot.infra_id)}"
    if work_schedule_group_id is None:
        return path
    if (
        not isinstance(work_schedule_group_id, int)
        or isinstance(work_schedule_group_id, bool)
        or work_schedule_group_id < 1
    ):
        raise ValueError("work_schedule_group_id must be a positive integer")
    return f"{path}&work_schedule_group_id={work_schedule_group_id}"


def _conflict_failure_reason(
    conflict: dict[str, Any],
    *,
    schedule_id_map: dict[str, str],
    source_by_id: dict[str, dict[str, Any]],
    is_new: bool,
) -> str:
    participants = []
    for train in conflict.get("train_ids", []):
        schedule_id = str(train["train_schedule_id"])
        label = schedule_id_map.get(schedule_id, f"unknown schedule {schedule_id}")
        if label.startswith("source:"):
            source_id = label.removeprefix("source:")
            schedule = source_by_id.get(source_id, {})
            name = str(schedule.get("train_name") or f"schedule {source_id}")
            label = f"{name} (ID {source_id})"
        elif label.startswith("inserted:"):
            label = f"new service {label.removeprefix('inserted:')}"
        index = train.get("index")
        if index not in (None, 0):
            label += f" (index {index})"
        participants.append(label)

    zones = sorted(
        {
            str(requirement["zone"])
            for requirement in conflict.get("requirements", [])
            if isinstance(requirement, dict) and requirement.get("zone") is not None
        }
    )
    kind = conflict.get("conflict_type", "unknown")
    status = "new" if is_new else "pre-existing"
    description = f"{status} OSRD {kind} conflict"
    if participants:
        description += f" involving {', '.join(participants)}"
    work_schedule_ids = conflict.get("work_schedule_ids", [])
    if isinstance(work_schedule_ids, list) and work_schedule_ids:
        description += " involving work schedule(s) " + ", ".join(
            str(schedule_id) for schedule_id in sorted(work_schedule_ids)
        )
    if zones:
        description += f" in zone(s) {', '.join(zones)}"
    interval = []
    if conflict.get("start_time") is not None:
        interval.append(f"start_time={conflict['start_time']}")
    if conflict.get("duration") is not None:
        interval.append(f"duration={conflict['duration']}")
    if interval:
        description += f" ({', '.join(interval)})"
    return description


def _candidate_service_label(
    schedule_id: int,
    candidate_id_map: dict[str, str],
    source_by_id: dict[str, dict[str, Any]],
) -> str:
    label = candidate_id_map.get(str(schedule_id), f"schedule {schedule_id}")
    if label.startswith("source:"):
        source_id = label.removeprefix("source:")
        schedule = source_by_id.get(source_id, {})
        name = str(schedule.get("train_name") or f"schedule {source_id}")
        return f"{name} (schedule {source_id})"
    if label.startswith("inserted:"):
        return f"new service {label.removeprefix('inserted:')}"
    return label


def _simulation_failure_reason(summary: dict[str, Any], service_label: str) -> str | None:
    """Turn OSRD's structured simulation-summary errors into planner guidance."""
    status = summary.get("status", "missing_summary")
    if status == "success":
        return None

    error_type = summary.get("error_type")
    if status == "pathfinding_input_error":
        if error_type == "rolling_stock_not_found":
            stock = summary.get("rolling_stock_name")
            if isinstance(stock, str) and stock:
                return (
                    f"OSRD could not find rolling stock '{stock}' for {service_label}. "
                    "Choose a rolling stock available in this infrastructure."
                )
        elif error_type == "unauthorized_rolling_stock":
            stock_id = summary.get("rolling_stock_id")
            if isinstance(stock_id, int) and not isinstance(stock_id, bool):
                return (
                    f"OSRD rejected rolling stock ID {stock_id} as unauthorized for "
                    f"{service_label}. Choose an authorized rolling stock."
                )
        elif error_type == "invalid_path_items":
            items = summary.get("items")
            indices = (
                [
                    item["index"] + 1
                    for item in items
                    if isinstance(item, dict)
                    and isinstance(item.get("index"), int)
                    and not isinstance(item.get("index"), bool)
                    and item["index"] >= 0
                ]
                if isinstance(items, list)
                else []
            )
            if indices:
                positions = ", ".join(str(index) for index in indices)
                return (
                    f"OSRD rejected path item(s) {positions} for {service_label}. "
                    "Check the selected endpoints and waypoint order."
                )
        elif error_type == "not_enough_path_items":
            return (
                f"OSRD needs an origin and destination path item for {service_label}. "
                "Choose at least two ordered path points."
            )
        elif error_type == "zero_length_path":
            return (
                f"OSRD resolved {service_label} to a zero-length path. "
                "Choose distinct ordered endpoints."
            )
        detail = f" ({error_type})" if isinstance(error_type, str) and error_type else ""
        return f"OSRD rejected the pathfinding input for {service_label}{detail}."

    if status == "pathfinding_not_found":
        guidance = {
            "not_found_in_blocks": "no route was found through the available blocks",
            "not_found_in_routes": "no compatible route was found",
            "not_found_in_tracks": "no continuous track path was found",
            "incompatible_constraints": "the requested route constraints cannot be met together",
        }.get(error_type)
        if guidance:
            return f"OSRD could not find a path for {service_label}: {guidance}."

    if status == "simulation_failed" and isinstance(error_type, str) and error_type:
        return f"OSRD simulation failed for {service_label} ({error_type})."
    if status == "missing_summary":
        return f"OSRD returned no simulation summary for {service_label}."
    detail = f" ({error_type})" if isinstance(error_type, str) and error_type else ""
    return f"OSRD could not simulate {service_label} (status: {status}{detail})."


class EditoastOSRDClient:
    """Validate candidate start-time edits on temporary, unpublished timetables.

    Source schedule sets are read through Editoast's timetable linking endpoint,
    and every candidate is built from all schedules in those linked sets.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    async def check_ready(self) -> bool:
        """Probe Editoast's dependency-aware health endpoint with a short timeout."""
        return await asyncio.to_thread(self._check_ready)

    def _check_ready(self) -> bool:
        request = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Accept": "text/plain", **self.headers},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 3)) as response:
                return response.status == 200 and response.read(64).strip() == b"ok"
        except (
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
            ValueError,
        ):
            return False

    async def get_schedule_resource_paths(
        self, *, snapshot: SnapshotRef, schedules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_schedule_resource_paths, snapshot, schedules)

    async def get_replay_baseline_context(
        self, *, snapshot: SnapshotRef
    ) -> ReplayBaselineContext:
        return await asyncio.to_thread(self._get_replay_baseline_context, snapshot)

    def _get_replay_baseline_context(self, snapshot: SnapshotRef) -> ReplayBaselineContext:
        _, schedules = self._load_source(snapshot)
        ordered = sorted(schedules, key=lambda schedule: int(schedule["id"]))
        schedule_ids = [int(schedule["id"]) for schedule in ordered]
        schedule_payloads = [_schedule_form(schedule) for schedule in ordered]
        summaries: dict[str, Any] = {}
        for start in range(0, len(schedule_ids), 200):
            batch = schedule_ids[start : start + 200]
            response = self._request(
                "POST",
                "/train_schedules/simulation_summary",
                {
                    "infra_id": int(snapshot.infra_id),
                    "timetable_id": int(snapshot.timetable_id),
                    "ids": batch,
                },
            )
            if not isinstance(response, dict):
                raise EditoastError("Editoast returned invalid baseline simulation summaries")
            summaries.update(cast(dict[str, Any], response))
        missing_or_failed = []
        for schedule_id in schedule_ids:
            summary = summaries.get(str(schedule_id))
            train_summary = summary.get("train_schedule") if isinstance(summary, dict) else None
            if not isinstance(train_summary, dict) or train_summary.get("status") != "success":
                missing_or_failed.append(schedule_id)
        if missing_or_failed:
            raise UnsupportedSnapshot(
                "Replay context requires successful baseline simulations for every schedule; "
                f"missing or failed IDs: {missing_or_failed[:20]}"
            )
        conflicts_value = self._request(
            "GET",
            f"/timetable/{int(snapshot.timetable_id)}/conflicts?infra_id={int(snapshot.infra_id)}",
        )
        if not isinstance(conflicts_value, list) or not all(
            isinstance(conflict, dict) for conflict in conflicts_value
        ):
            raise EditoastError("Editoast returned an invalid baseline conflict response")
        self._load_source(snapshot)
        return ReplayBaselineContext(
            snapshot=snapshot,
            schedule_ids=schedule_ids,
            schedule_payloads=schedule_payloads,
            simulation_summaries=summaries,
            conflicts=cast(list[dict[str, Any]], conflicts_value),
        )

    def _get_schedule_resource_paths(
        self, snapshot: SnapshotRef, schedules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self._load_source(snapshot)
        route_cache: dict[str, dict[str, Any]] = {}
        rolling_stock_cache: dict[str, dict[str, Any]] = {}
        paths: list[dict[str, Any]] = []
        for schedule in schedules:
            path_items = schedule.get("path")
            rolling_stock_name = schedule.get("rolling_stock_name")
            if (
                not isinstance(path_items, list)
                or len(path_items) < 2
                or not isinstance(rolling_stock_name, str)
                or not rolling_stock_name
            ):
                raise UnsupportedSnapshot(
                    "Schedule needs a path and rolling stock for resource extraction"
                )
            speed_limit_tag = schedule.get("speed_limit_tag")
            options = schedule.get("options", {})
            if speed_limit_tag is not None and not isinstance(speed_limit_tag, str):
                raise UnsupportedSnapshot("Schedule has an invalid speed limit tag")
            if not isinstance(options, dict):
                raise UnsupportedSnapshot("Schedule has invalid pathfinding options")
            stops_at_end_of_block = options.get("stops_at_end_of_block", False)
            if type(stops_at_end_of_block) is not bool:
                raise UnsupportedSnapshot("Schedule has an invalid stop-at-end-of-block option")
            locations: list[dict[str, Any]] = []
            for item in path_items:
                location = item.get("location") if isinstance(item, dict) else None
                if not isinstance(location, dict):
                    raise UnsupportedSnapshot(
                        "Schedule path waypoint needs an OSRD location object"
                    )
                location_type = location.get("type")
                if location_type == "track_offset":
                    track, offset = location.get("track"), location.get("offset")
                    if (
                        not isinstance(track, str)
                        or not track
                        or not isinstance(offset, int)
                        or isinstance(offset, bool)
                        or offset < 0
                    ):
                        raise UnsupportedSnapshot("Schedule has an invalid track-offset waypoint")
                elif location_type == "operational_point_part_reference":
                    operational_point = location.get("operational_point")
                    if not isinstance(operational_point, dict):
                        raise UnsupportedSnapshot(
                            "Schedule has an invalid operational-point reference"
                        )
                else:
                    raise UnsupportedSnapshot("Schedule has an unsupported path location type")
                locations.append(
                    {
                        "location": location,
                        "can_backtrack": False,
                    }
                )
            route_key = json.dumps(
                [
                    rolling_stock_name,
                    locations,
                    speed_limit_tag,
                    stops_at_end_of_block,
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
            resource_path = route_cache.get(route_key)
            if resource_path is None:
                rolling_stock = rolling_stock_cache.get(rolling_stock_name)
                if rolling_stock is None:
                    rolling_stock_value = self._request(
                        "GET", f"/rolling_stock/name/{quote(rolling_stock_name, safe='')}"
                    )
                    if not isinstance(rolling_stock_value, dict):
                        raise EditoastError("Editoast returned an invalid rolling stock record")
                    rolling_stock = cast(dict[str, Any], rolling_stock_value)
                    rolling_stock_cache[rolling_stock_name] = rolling_stock
                resource_path = self._path_resource(
                    snapshot,
                    locations,
                    rolling_stock,
                    speed_limit_tag=speed_limit_tag,
                    stops_at_end_of_block=stops_at_end_of_block,
                )
                route_cache[route_key] = resource_path
            paths.append(resource_path)
        self._load_source(snapshot)
        return paths

    def _path_resource(
        self,
        snapshot: SnapshotRef,
        path_items: list[dict[str, Any]],
        rolling_stock: dict[str, Any],
        *,
        speed_limit_tag: str | None,
        stops_at_end_of_block: bool,
    ) -> dict[str, Any]:
        effort_curves = rolling_stock.get("effort_curves", {})
        modes = effort_curves.get("modes", {}) if isinstance(effort_curves, dict) else {}
        default_mode = rolling_stock.get("default_mode")
        mode = modes.get(default_mode, {}) if isinstance(modes, dict) else {}
        energy_sources = rolling_stock.get("energy_sources", [])
        signaling_systems = rolling_stock.get("supported_signaling_systems", [])
        if (
            not isinstance(mode, dict)
            or not isinstance(energy_sources, list)
            or not isinstance(signaling_systems, list)
        ):
            raise UnsupportedSnapshot("Rolling stock has invalid pathfinding capabilities")
        request = {
            "path_items": path_items,
            "rolling_stock_loading_gauge": rolling_stock.get("loading_gauge", "G1"),
            "rolling_stock_is_thermal": not bool(mode.get("is_electric", False)),
            "rolling_stock_supported_electrifications": sorted(
                {
                    source.get("electrical_profile")
                    for source in energy_sources
                    if isinstance(source, dict) and source.get("electrical_profile")
                }
            ),
            "rolling_stock_supported_signaling_systems": [
                item["type"] if isinstance(item, dict) and "type" in item else item
                for item in signaling_systems
            ],
            "rolling_stock_maximum_speed": float(rolling_stock.get("max_speed", 0)),
            "rolling_stock_length": round(float(rolling_stock.get("length", 0)) * 1000),
            "speed_limit_tag": speed_limit_tag,
            "stops_at_end_of_block": stops_at_end_of_block,
        }
        response = self._request(
            "POST", f"/infra/{int(snapshot.infra_id)}/pathfinding/blocks", request
        )
        if not isinstance(response, dict) or response.get("status") != "success":
            raise UnsupportedSnapshot("OSRD pathfinding could not derive a schedule resource path")
        path = response.get("path")
        if not isinstance(path, dict) or not isinstance(path.get("track_section_ranges"), list):
            raise EditoastError("OSRD returned an invalid pathfinding resource path")
        ranges = path["track_section_ranges"]
        if not ranges or not all(isinstance(item, dict) for item in ranges):
            raise EditoastError("OSRD returned a path without track-section ranges")
        for item in ranges:
            begin, end = item.get("begin"), item.get("end")
            if (
                not isinstance(item.get("track_section"), str)
                or not item["track_section"]
                or not isinstance(begin, int | float)
                or isinstance(begin, bool)
                or not math.isfinite(begin)
                or not isinstance(end, int | float)
                or isinstance(end, bool)
                or not math.isfinite(end)
            ):
                raise EditoastError("OSRD returned an invalid track-section resource range")
        length = response.get("length", path.get("length"))
        if (
            not isinstance(length, int | float)
            or isinstance(length, bool)
            or not math.isfinite(length)
            or length < 0
        ):
            length = sum(abs(float(item["end"]) - float(item["begin"])) for item in ranges)
        return {**path, "length": length}

    def _track_offset_options(
        self, snapshot: SnapshotRef, location: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if location.get("type") == "track_offset":
            track, offset = location.get("track"), location.get("offset")
            if (
                isinstance(track, str)
                and track
                and isinstance(offset, int)
                and not isinstance(offset, bool)
                and offset >= 0
            ):
                return [location]
            raise UnsupportedSnapshot("Route alternatives received an invalid track offset")

        operational_point = location.get("operational_point")
        if (
            location.get("type") != "operational_point_part_reference"
            or not isinstance(operational_point, dict)
            or operational_point.get("type") != "id"
            or not isinstance(operational_point.get("operational_point"), str)
        ):
            raise UnsupportedSnapshot(
                "Route alternatives require track offsets or ID-based operational points"
            )
        operational_point_id = operational_point["operational_point"]
        response = self._request(
            "POST",
            f"/infra/{int(snapshot.infra_id)}/objects/operational_point",
            [operational_point_id],
        )
        if (
            not isinstance(response, list)
            or len(response) != 1
            or not isinstance(response[0], dict)
        ):
            raise EditoastError("Editoast returned an invalid operational-point response")
        railjson = response[0].get("railjson")
        if not isinstance(railjson, dict) or not isinstance(railjson.get("parts"), list):
            raise EditoastError("Editoast returned an operational point without track parts")
        requested_local_track = location.get("local_track_name")
        options: list[dict[str, Any]] = []
        for part in railjson["parts"]:
            if not isinstance(part, dict):
                continue
            if (
                requested_local_track is not None
                and part.get("local_track_name") != requested_local_track
            ):
                continue
            track, position = part.get("track"), part.get("position")
            if (
                isinstance(track, str)
                and track
                and isinstance(position, int | float)
                and not isinstance(position, bool)
                and math.isfinite(position)
                and position >= 0
            ):
                options.append(
                    {"type": "track_offset", "track": track, "offset": round(position * 1000)}
                )
        if not options:
            raise UnsupportedSnapshot(
                f"Operational point {operational_point_id} has no usable track parts"
            )
        return options

    async def get_route_alternatives(
        self,
        *,
        snapshot: SnapshotRef,
        starting: dict[str, Any],
        ending: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Query the infra k-shortest-path endpoint for one waypoint pair."""
        if not 1 <= limit <= 5:
            raise ValueError("OSRD supports between one and five route alternatives")
        starting_options = await asyncio.to_thread(self._track_offset_options, snapshot, starting)
        ending_options = await asyncio.to_thread(self._track_offset_options, snapshot, ending)
        combinations = [(start, end) for start in starting_options for end in ending_options]
        if len(combinations) > MAX_ROUTE_ENDPOINT_COMBINATIONS:
            raise UnsupportedSnapshot(
                "Operational-point route search spans too many track-part combinations; "
                "use the advanced path editor to specify track-offset waypoints"
            )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PATHFINDING_REQUESTS)

        async def find_routes(start: dict[str, Any], end: dict[str, Any]) -> list[dict[str, Any]]:
            payload = {
                "starting": {
                    "track": start["track"],
                    # This legacy endpoint uses metres; train schedule offsets use millimetres.
                    "position": float(start["offset"]) / 1000,
                },
                "ending": {
                    "track": end["track"],
                    "position": float(end["offset"]) / 1000,
                },
            }
            async with semaphore:
                response: object = await asyncio.to_thread(
                    self._request,
                    "POST",
                    f"/infra/{int(snapshot.infra_id)}/pathfinding/?number={limit}",
                    payload,
                )
            if not isinstance(response, list) or not all(
                isinstance(row, dict) for row in response
            ):
                raise EditoastError("Editoast returned an invalid route-alternative response")
            routes = cast(list[dict[str, Any]], response)
            for route in routes:
                ranges = route.get("track_ranges")
                if not isinstance(ranges, list) or not ranges:
                    raise EditoastError("Editoast returned a route without track ranges")
                for track_range in ranges:
                    if not isinstance(track_range, dict):
                        raise EditoastError("Editoast returned an invalid route track range")
                    track = track_range.get("track")
                    begin, end = track_range.get("begin"), track_range.get("end")
                    if (
                        not isinstance(track, str)
                        or not track
                        or not isinstance(begin, int | float)
                        or isinstance(begin, bool)
                        or not math.isfinite(begin)
                        or not isinstance(end, int | float)
                        or isinstance(end, bool)
                        or not math.isfinite(end)
                    ):
                        raise EditoastError("Editoast returned an invalid route track range")
            return routes

        route_groups = await asyncio.gather(
            *(find_routes(start, end) for start, end in combinations)
        )
        unique_routes: dict[str, dict[str, Any]] = {}
        for routes in route_groups:
            for route in routes:
                signature = json.dumps(route["track_ranges"], sort_keys=True, separators=(",", ":"))
                unique_routes.setdefault(signature, route)

        def route_key(item: tuple[str, dict[str, Any]]) -> tuple[float, str]:
            signature, route = item
            track_ranges = route["track_ranges"]
            length = sum(
                abs(float(track_range["end"]) - float(track_range["begin"]))
                for track_range in track_ranges
                if isinstance(track_range, dict)
                and isinstance(track_range.get("begin"), int | float)
                and isinstance(track_range.get("end"), int | float)
            )
            return length, signature

        return [route for _, route in sorted(unique_routes.items(), key=route_key)[:limit]]

    def _request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json", **self.headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read()
                return json.loads(content) if content else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            raise EditoastError(f"Editoast {method} {path}: HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise EditoastError(f"Editoast {method} {path}: {error.reason}") from error

    def _read_source_material(
        self, infra_id: str, timetable_id: str
    ) -> tuple[str, str, str, list[dict[str, Any]]]:
        infra_number = int(infra_id)
        timetable_number = int(timetable_id)
        infra = cast(dict[str, Any], self._request("GET", f"/infra/{infra_number}"))
        version = cast(dict[str, Any], self._request("GET", "/version"))
        actual_infra_version = str(infra["version"])
        actual_osrd_version = version.get("git_describe")
        if not isinstance(actual_osrd_version, str) or not actual_osrd_version:
            raise UnsupportedSnapshot("OSRD did not report a build version")

        sets = self._request("GET", f"/timetable/{timetable_number}/train_schedule_sets")
        if not isinstance(sets, list):
            raise UnsupportedSnapshot("Source timetable has no linked train schedule sets")
        set_rows = cast(list[object], sets)
        if not set_rows or not all(isinstance(item, dict) for item in set_rows):
            raise UnsupportedSnapshot("Source timetable has invalid linked schedule sets")
        schedule_sets = cast(list[dict[str, Any]], set_rows)
        timetable_types = {str(item.get("timetable_type")) for item in schedule_sets}
        if len(timetable_types) != 1 or "None" in timetable_types:
            raise UnsupportedSnapshot("Source timetable has inconsistent timetable types")
        timetable_type = next(iter(timetable_types))

        schedules: list[dict[str, Any]] = []
        page = 1
        while True:
            page_result = cast(
                dict[str, Any],
                self._request(
                    "GET",
                    f"/timetable/{timetable_number}/train_schedules?page={page}&page_size=200",
                ),
            )
            results = page_result.get("results")
            if not isinstance(results, list):
                raise EditoastError("Editoast returned an invalid paginated schedule response")
            result_rows = cast(list[object], results)
            if not all(isinstance(row, dict) for row in result_rows):
                raise EditoastError("Editoast returned an invalid paginated schedule response")
            schedules.extend(cast(list[dict[str, Any]], result_rows))
            next_page = page_result.get("next")
            if next_page is None:
                break
            page = int(next_page)
        return actual_infra_version, actual_osrd_version, timetable_type, schedules

    def _load_source(self, snapshot: SnapshotRef) -> tuple[str, list[dict[str, Any]]]:
        infra_version, osrd_version, timetable_type, schedules = self._read_source_material(
            snapshot.infra_id, snapshot.timetable_id
        )
        if infra_version != snapshot.infra_version:
            raise UnsupportedSnapshot("Infrastructure version does not match the request binding")
        if osrd_version != snapshot.osrd_version:
            raise UnsupportedSnapshot("OSRD build does not match the request binding")
        revision = compute_timetable_revision(timetable_type, schedules)
        if revision != snapshot.timetable_revision:
            raise UnsupportedSnapshot("Timetable content does not match the request revision")
        return timetable_type, schedules

    async def describe_snapshot(self, *, infra_id: str, timetable_id: str) -> SnapshotRef:
        return await asyncio.to_thread(self._describe_snapshot, infra_id, timetable_id)

    def _describe_snapshot(self, infra_id: str, timetable_id: str) -> SnapshotRef:
        infra_version, osrd_version, timetable_type, schedules = self._read_source_material(
            infra_id, timetable_id
        )
        return SnapshotRef(
            infra_id=infra_id,
            infra_version=infra_version,
            timetable_id=timetable_id,
            timetable_revision=compute_timetable_revision(timetable_type, schedules),
            osrd_version=osrd_version,
        )

    async def get_schedule_start_times(self, *, snapshot: SnapshotRef) -> dict[str, int]:
        return await asyncio.to_thread(self._get_schedule_start_times, snapshot)

    def _get_schedule_start_times(self, snapshot: SnapshotRef) -> dict[str, int]:
        _, schedules = self._load_source(snapshot)
        return {str(item["id"]): int(item["start_time"]) for item in schedules}

    async def get_schedule_metadata(self, *, snapshot: SnapshotRef) -> ScheduleMetadata:
        return await asyncio.to_thread(self._get_schedule_metadata, snapshot)

    def _get_schedule_metadata(self, snapshot: SnapshotRef) -> ScheduleMetadata:
        timetable_type, schedules = self._load_source(snapshot)
        runtime_objective_unsupported_ids = set()
        max_hourly_shift_seconds_by_schedule = {}
        max_hourly_advance_seconds_by_schedule = {}
        runtime_affecting_exception_fields = {
            "constraint_distribution",
            "initial_speed",
            "options",
            "path_and_schedule",
            "rolling_stock",
            "rolling_stock_category",
            "speed_limit_tag",
        }
        for schedule in schedules:
            paced = schedule.get("paced")
            if timetable_type == "HOURLY":
                if not isinstance(paced, dict):
                    raise UnsupportedSnapshot("Hourly timetable contains an unpaced train schedule")
                interval_ms = _duration_milliseconds(paced.get("interval"))
                start_time = schedule.get("start_time")
                if (
                    not isinstance(start_time, int)
                    or isinstance(start_time, bool)
                    or not 0 <= start_time < interval_ms
                ):
                    raise UnsupportedSnapshot("Hourly paced train has an invalid start time")
                remaining_ms = interval_ms - start_time - 1
                max_hourly_shift_seconds_by_schedule[str(schedule["id"])] = min(
                    7200, remaining_ms // 1000
                )
                advance_seconds = min(7200, start_time // 1000)
                max_hourly_advance_seconds_by_schedule[str(schedule["id"])] = advance_seconds
            if not isinstance(paced, dict):
                continue
            exceptions = paced.get("exceptions", [])
            if not isinstance(exceptions, list):
                raise UnsupportedSnapshot("Paced train exceptions have an invalid shape")
            for exception in exceptions:
                if not isinstance(exception, dict):
                    raise UnsupportedSnapshot("Paced train exceptions have an invalid shape")
                if exception.get("disabled", False):
                    continue
                groups = _exception_change_groups(exception)
                if timetable_type == "HOURLY" and "start_time" in groups:
                    start_group = groups["start_time"]
                    exception_start = (
                        start_group.get("value") if isinstance(start_group, dict) else None
                    )
                    if (
                        isinstance(exception_start, int)
                        and not isinstance(exception_start, bool)
                    ):
                        max_hourly_advance_seconds_by_schedule[str(schedule["id"])] = min(
                            max_hourly_advance_seconds_by_schedule[str(schedule["id"])],
                            exception_start // 1000,
                        )
                        max_hourly_shift_seconds_by_schedule[str(schedule["id"])] = min(
                            max_hourly_shift_seconds_by_schedule[str(schedule["id"])],
                            max(0, interval_ms - exception_start - 1) // 1000,
                        )
                if runtime_affecting_exception_fields.intersection(groups):
                    runtime_objective_unsupported_ids.add(str(schedule["id"]))
                    break
        return ScheduleMetadata(
            start_times_ms={str(item["id"]): int(item["start_time"]) for item in schedules},
            train_names={str(item["id"]): str(item["train_name"]) for item in schedules},
            schedule_forms_by_id={
                str(item["id"]): _schedule_form(item) for item in schedules
            },
            timetable_type=timetable_type,
            runtime_objective_unsupported_ids=frozenset(runtime_objective_unsupported_ids),
            max_hourly_shift_seconds_by_schedule=max_hourly_shift_seconds_by_schedule,
            max_hourly_advance_seconds_by_schedule=max_hourly_advance_seconds_by_schedule,
        )

    async def get_schedule_travel_times(
        self, *, snapshot: SnapshotRef, schedule_ids: list[str]
    ) -> dict[str, int]:
        return await asyncio.to_thread(self._get_schedule_travel_times, snapshot, schedule_ids)

    def _get_schedule_travel_times(
        self, snapshot: SnapshotRef, schedule_ids: list[str]
    ) -> dict[str, int]:
        if not schedule_ids:
            return {}
        self._load_source(snapshot)
        selected_ids = list(dict.fromkeys(schedule_ids))
        travel_times: dict[str, int] = {}
        for batch_start in range(0, len(selected_ids), MAX_SCHEDULE_SIMULATION_BATCH):
            batch_ids = selected_ids[
                batch_start : batch_start + MAX_SCHEDULE_SIMULATION_BATCH
            ]
            response = self._request(
                "POST",
                "/train_schedules/simulation_summary",
                {
                    "infra_id": int(snapshot.infra_id),
                    "timetable_id": int(snapshot.timetable_id),
                    "ids": [int(schedule_id) for schedule_id in batch_ids],
                },
            )
            if not isinstance(response, dict):
                raise EditoastError("Editoast returned invalid baseline simulation summaries")
            summaries = cast(dict[str, Any], response)
            for schedule_id in batch_ids:
                value = summaries.get(schedule_id)
                summary = value.get("train_schedule") if isinstance(value, dict) else None
                travel_time = summary.get("time") if isinstance(summary, dict) else None
                if (
                    isinstance(travel_time, int)
                    and not isinstance(travel_time, bool)
                    and travel_time >= 0
                    and isinstance(summary, dict)
                    and summary.get("status") == "success"
                ):
                    travel_times[schedule_id] = travel_time
        self._load_source(snapshot)
        return travel_times

    async def validate_candidate(
        self,
        *,
        snapshot: SnapshotRef,
        candidate_payload: dict[str, object],
    ) -> CandidateValidation:
        return await asyncio.to_thread(self._validate_candidate, snapshot, candidate_payload)

    def _validate_candidate(
        self,
        snapshot: SnapshotRef,
        candidate_payload: dict[str, object],
    ) -> CandidateValidation:
        if candidate_payload.get("snapshot") != asdict(snapshot):
            raise UnsupportedSnapshot("Candidate snapshot binding differs from validation request")
        edits = candidate_payload.get("edits")
        insertion_values = candidate_payload.get("insertions", [])
        candidate_id = candidate_payload.get("candidate_id")
        if not isinstance(edits, list) or not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Candidate payload must include an ID and an edits list")
        if not isinstance(insertion_values, list) or len(insertion_values) > 50:
            raise ValueError("Candidate insertions must be a list containing at most 50 services")
        timetable_type, source_schedules = self._load_source(snapshot)
        inserted_forms: list[dict[str, Any]] = []
        inserted_names: set[str] = set()
        for value in insertion_values:
            if not isinstance(value, dict):
                raise ValueError("Each inserted service must be an object")
            unknown_fields = set(value) - set(_TRAIN_SCHEDULE_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"Inserted service has unsupported fields: {sorted(unknown_fields)}"
                )
            form = _schedule_form(value)
            is_paced = form.get("paced") is not None
            if is_paced != (timetable_type == "HOURLY"):
                expected = "paced" if timetable_type == "HOURLY" else "calendar"
                raise ValueError(f"Inserted services must use {expected} schedules")
            name = str(form["train_name"])
            if name in inserted_names:
                raise ValueError("Inserted service names must be unique within a candidate")
            inserted_names.add(name)
            inserted_forms.append(form)
        edit_rows = cast(list[object], edits)
        work_schedule_group_id = candidate_payload.get("work_schedule_group_id")
        if work_schedule_group_id is not None and timetable_type == "HOURLY":
            raise UnsupportedSnapshot(
                "Work-schedule conflict checks are not supported for hourly timetables"
            )

        # An empty edit list is the pass-through baseline: clone and validate the
        # exact source timetable through the same OSRD checks as edited candidates.
        baseline_conflicts_value = self._request(
            "GET",
            _conflicts_path(snapshot, int(snapshot.timetable_id), work_schedule_group_id),
        )
        if not isinstance(baseline_conflicts_value, list) or not all(
            isinstance(row, dict) for row in baseline_conflicts_value
        ):
            raise EditoastError("Editoast returned an invalid baseline conflict response")
        baseline_conflict_rows = cast(list[dict[str, Any]], baseline_conflicts_value)
        baseline_id_map = {
            str(schedule["id"]): f"source:{schedule['id']}" for schedule in source_schedules
        }
        baseline_conflict_signatures = _conflict_signatures(baseline_conflict_rows, baseline_id_map)
        source_by_id = {str(schedule["id"]): schedule for schedule in source_schedules}
        if len(source_by_id) != len(source_schedules):
            raise UnsupportedSnapshot("Source timetable contains duplicate train schedule IDs")
        forms = {schedule_id: _schedule_form(row) for schedule_id, row in source_by_id.items()}
        existing_names = {str(form["train_name"]) for form in forms.values()}
        if existing_names & inserted_names:
            raise ValueError("Inserted train names must be unique against the baseline timetable")
        exceptions_by_source_id: dict[str, list[dict[str, Any]]] = {}
        for schedule_id, form in forms.items():
            paced = form.get("paced")
            if isinstance(paced, dict):
                paced_data = cast(dict[str, Any], paced)
                exceptions_by_source_id[schedule_id] = cast(
                    list[dict[str, Any]], paced_data.get("exceptions", [])
                )
        edited_ids: set[str] = set()
        for edit_value in edit_rows:
            edit = cast(dict[str, Any], edit_value) if isinstance(edit_value, dict) else None
            if not isinstance(edit, dict):
                raise ValueError("Candidate edit must be an object")
            schedule_id = str(edit.get("train_schedule_id", ""))
            before = edit.get("before_start_time_ms")
            after = edit.get("after_start_time_ms")
            shift = edit.get("shift_seconds")
            if schedule_id in edited_ids or schedule_id not in forms:
                raise ValueError("Candidate contains a duplicate or unknown train schedule ID")
            if (
                not isinstance(before, int)
                or isinstance(before, bool)
                or not isinstance(after, int)
                or isinstance(after, bool)
                or not isinstance(shift, int)
                or isinstance(shift, bool)
            ):
                raise ValueError("Candidate departure times and shifts must be integer values")
            if shift == 0 or abs(shift) > 7200 or after - before != shift * 1000:
                raise ValueError("Candidate departure shift is outside supported bounds")
            if forms[schedule_id]["start_time"] != before:
                raise UnsupportedSnapshot(
                    "Candidate edit was generated from a stale departure time"
                )
            forms[schedule_id]["start_time"] = after
            paced_value = forms[schedule_id].get("paced")
            if isinstance(paced_value, dict):
                for exception in exceptions_by_source_id[schedule_id]:
                    groups = _exception_change_groups(exception)
                    start_time_group = groups.get("start_time")
                    if start_time_group is not None:
                        if not isinstance(start_time_group, dict):
                            raise UnsupportedSnapshot(
                                "Paced exception start-time change is invalid"
                            )
                        start_group = cast(dict[str, Any], start_time_group)
                        exception_start = start_group.get("value")
                        if not isinstance(exception_start, int) or isinstance(
                            exception_start, bool
                        ):
                            raise UnsupportedSnapshot(
                                "Paced exception start time is not integer milliseconds"
                            )
                        start_group["value"] = exception_start + shift * 1000
            edited_ids.add(schedule_id)

        dwell_edit_values = candidate_payload.get("existing_service_dwell_edits", [])
        if not isinstance(dwell_edit_values, list):
            raise ValueError("Candidate existing_service_dwell_edits must be a list")
        seen_dwell_targets: set[tuple[str, str]] = set()
        for edit_value in dwell_edit_values:
            if not isinstance(edit_value, dict):
                raise ValueError("Existing-service dwell edit must be an object")
            schedule_id = str(edit_value.get("train_schedule_id", ""))
            waypoint_key = edit_value.get("waypoint_key")
            before = edit_value.get("before_dwell_seconds")
            after = edit_value.get("after_dwell_seconds")
            target = (schedule_id, waypoint_key) if isinstance(waypoint_key, str) else None
            if (
                schedule_id not in forms
                or target is None
                or target in seen_dwell_targets
                or not isinstance(before, int)
                or isinstance(before, bool)
                or not isinstance(after, int)
                or isinstance(after, bool)
                or not 0 <= before <= 3600
                or not 0 <= after <= 3600
                or before == after
            ):
                raise ValueError("Existing-service dwell edit has an invalid target or duration")
            seen_dwell_targets.add(target)
            form = forms[schedule_id]
            if form.get("paced") is not None:
                raise UnsupportedSnapshot("Existing-service dwell edits require calendar schedules")
            path = form.get("path")
            schedule = form.get("schedule")
            if (
                not isinstance(path, list)
                or not isinstance(schedule, list)
                or not all(isinstance(row, dict) for row in schedule)
            ):
                raise UnsupportedSnapshot("Existing service has no editable path or schedule")
            path_keys = [
                item.get("key") for item in path if isinstance(item, dict)
            ]
            if len(path_keys) != len(path) or waypoint_key not in path_keys:
                raise ValueError("Dwell edit waypoint is not in the existing service path")
            waypoint_index = path_keys.index(waypoint_key)
            if waypoint_index == 0 or waypoint_index == len(path_keys) - 1:
                raise ValueError(
                    "Existing-service dwell edits must target an intermediate waypoint"
                )
            matching_indices = [
                index
                for index, row in enumerate(schedule)
                if isinstance(row, dict) and row.get("at") == waypoint_key
            ]
            if len(matching_indices) > 1:
                raise UnsupportedSnapshot("Existing service has duplicate station-call entries")
            source_dwell = (
                _stop_duration_seconds(schedule[matching_indices[0]].get("stop_for"))
                if matching_indices
                else 0
            )
            if source_dwell != before:
                raise UnsupportedSnapshot("Candidate dwell edit was generated from stale stop data")
            if after == 0:
                if matching_indices:
                    schedule.pop(matching_indices[0])
                continue
            if matching_indices:
                schedule[matching_indices[0]]["stop_for"] = f"PT{after}S"
            else:
                schedule.append({"at": waypoint_key, "stop_for": f"PT{after}S"})
                schedule.sort(key=lambda row: path_keys.index(str(row.get("at"))))

        candidate_timetable_id: int | None = None
        candidate_set_id: int | None = None
        cleanup_errors: list[str] = []
        validation: CandidateValidation | None = None
        try:
            timetable = self._request("POST", "/timetable", {"timetable_type": timetable_type})
            candidate_timetable_id = int(timetable["timetable_id"])
            schedule_set = self._request(
                "POST",
                "/train_schedule_sets",
                {
                    "description": f"AI candidate validation {candidate_id}",
                    "name": f"ai-candidate-{candidate_id}",
                    "published": False,
                    "timetable_type": timetable_type,
                },
            )
            candidate_set_id = int(schedule_set["id"])
            self._request(
                "POST",
                f"/timetable/{candidate_timetable_id}/train_schedule_sets",
                {"train_schedule_set_ids": [candidate_set_id]},
            )
            candidate_forms: dict[str, dict[str, Any]] = {}
            for schedule_id, form in forms.items():
                candidate_form = dict(form)
                paced = form.get("paced")
                if isinstance(paced, dict):
                    candidate_paced = dict(cast(dict[str, Any], paced))
                    candidate_paced["exceptions"] = []
                    candidate_form["paced"] = candidate_paced
                candidate_forms[schedule_id] = candidate_form

            candidate_schedule_ids: list[int] = []
            candidate_schedule_by_source_id: dict[str, int] = {}
            candidate_id_map: dict[str, str] = {}
            schedules_with_exceptions = {
                schedule_id
                for schedule_id, exceptions in exceptions_by_source_id.items()
                if exceptions
            }
            regular_source_ids = sorted(set(forms) - schedules_with_exceptions)
            if regular_source_ids:
                created = self._request(
                    "POST",
                    f"/train_schedule_sets/{candidate_set_id}/train_schedules",
                    [candidate_forms[schedule_id] for schedule_id in regular_source_ids],
                )
                regular_ids = self._created_schedule_ids(created, len(regular_source_ids))
                candidate_schedule_ids.extend(regular_ids)
                candidate_id_map.update(
                    {
                        str(candidate_id): f"source:{source_id}"
                        for source_id, candidate_id in zip(
                            regular_source_ids, regular_ids, strict=True
                        )
                    }
                )

            if inserted_forms:
                created = self._request(
                    "POST",
                    f"/train_schedule_sets/{candidate_set_id}/train_schedules",
                    inserted_forms,
                )
                inserted_ids = self._created_schedule_ids(created, len(inserted_forms))
                candidate_schedule_ids.extend(inserted_ids)
                candidate_id_map.update(
                    {
                        str(candidate_id): f"inserted:{form['train_name']}"
                        for form, candidate_id in zip(inserted_forms, inserted_ids, strict=True)
                    }
                )

            for source_id in sorted(schedules_with_exceptions):
                created = self._request(
                    "POST",
                    f"/train_schedule_sets/{candidate_set_id}/train_schedules",
                    [candidate_forms[source_id]],
                )
                new_schedule_id = self._created_schedule_ids(created, 1)[0]
                candidate_schedule_by_source_id[source_id] = new_schedule_id
                candidate_schedule_ids.append(new_schedule_id)
                candidate_id_map[str(new_schedule_id)] = f"source:{source_id}"

            expected_exception_ids: dict[int, set[str]] = {}
            for source_id in sorted(schedules_with_exceptions):
                new_schedule_id = candidate_schedule_by_source_id[source_id]
                new_exception_ids: set[str] = set()
                for exception in exceptions_by_source_id[source_id]:
                    occurrence_index = exception.get("occurrence_index")
                    groups = _exception_change_groups(exception)
                    created_exception = self._request(
                        "POST",
                        f"/timetable/{candidate_timetable_id}/train_schedule_exception",
                        {
                            "train_schedule_id": new_schedule_id,
                            "occurrence_index": occurrence_index,
                            "disabled": bool(exception.get("disabled", False)),
                            "change_groups": groups,
                        },
                    )
                    if not isinstance(created_exception, dict) or "id" not in created_exception:
                        raise EditoastError("Editoast did not return the created exception ID")
                    exception_result = cast(dict[str, Any], created_exception)
                    new_exception_ids.add(str(exception_result["id"]))
                expected_exception_ids[new_schedule_id] = new_exception_ids

            summaries = cast(
                dict[str, Any],
                self._request(
                    "POST",
                    "/train_schedules/simulation_summary",
                    {
                        "infra_id": int(snapshot.infra_id),
                        "timetable_id": candidate_timetable_id,
                        "ids": candidate_schedule_ids,
                    },
                ),
            )
            statuses: list[str] = []
            simulation_failure_reasons: list[str] = []
            for schedule_id in candidate_schedule_ids:
                service_label = _candidate_service_label(
                    schedule_id, candidate_id_map, source_by_id
                )
                schedule_summary_value = summaries.get(str(schedule_id))
                schedule_summary = (
                    cast(dict[str, Any], schedule_summary_value)
                    if isinstance(schedule_summary_value, dict)
                    else None
                )
                if not isinstance(schedule_summary, dict):
                    statuses.append("missing_summary")
                    simulation_failure_reasons.append(
                        _simulation_failure_reason({}, service_label)
                        or f"OSRD returned no simulation summary for {service_label}."
                    )
                    continue
                base_summary_value = schedule_summary.get("train_schedule")
                base_summary = (
                    cast(dict[str, Any], base_summary_value)
                    if isinstance(base_summary_value, dict)
                    else None
                )
                base_status = (
                    str(base_summary.get("status", "missing_summary"))
                    if base_summary is not None
                    else "missing_summary"
                )
                statuses.append(base_status)
                if base_status != "success":
                    simulation_failure_reasons.append(
                        _simulation_failure_reason(base_summary or {}, service_label)
                        or f"OSRD could not simulate {service_label}."
                    )
                exception_summaries = schedule_summary.get("exceptions", {})
                if not isinstance(exception_summaries, dict):
                    statuses.append("missing_summary")
                    simulation_failure_reasons.append(
                        f"OSRD returned invalid exception summaries for {service_label}."
                    )
                    continue
                exception_summary_map = cast(dict[str, Any], exception_summaries)
                for exception_id in expected_exception_ids.get(schedule_id, set()):
                    exception_summary_value = exception_summary_map.get(exception_id)
                    exception_summary = (
                        cast(dict[str, Any], exception_summary_value)
                        if isinstance(exception_summary_value, dict)
                        else None
                    )
                    exception_status = (
                        str(exception_summary.get("status", "missing_summary"))
                        if exception_summary is not None
                        else "missing_summary"
                    )
                    statuses.append(exception_status)
                    if exception_status != "success":
                        scope = f"{service_label} exception {exception_id}"
                        simulation_failure_reasons.append(
                            _simulation_failure_reason(exception_summary or {}, scope)
                            or f"OSRD could not simulate {scope}."
                        )
            pathfinding_valid = all(
                status
                not in {"pathfinding_not_found", "pathfinding_input_error", "missing_summary"}
                for status in statuses
            )
            simulation_valid = all(status == "success" for status in statuses)
            inserted_travel_times_ms: dict[str, int] = {}
            existing_travel_times_ms: dict[str, int] = {}
            inserted_path_item_times_ms: dict[str, list[int]] = {}
            for schedule_id, label in candidate_id_map.items():
                summary_value = summaries.get(schedule_id)
                summary = (
                    summary_value.get("train_schedule")
                    if isinstance(summary_value, dict)
                    else None
                )
                travel_time = summary.get("time") if isinstance(summary, dict) else None
                if (
                    isinstance(travel_time, int)
                    and not isinstance(travel_time, bool)
                    and travel_time >= 0
                    and isinstance(summary, dict)
                    and summary.get("status") == "success"
                ):
                    if label.startswith("inserted:"):
                        inserted_travel_times_ms[label.removeprefix("inserted:")] = travel_time
                    elif label.startswith("source:"):
                        existing_travel_times_ms[label.removeprefix("source:")] = travel_time
                path_item_times = (
                    summary.get("path_item_times_final")
                    if isinstance(summary, dict)
                    else None
                )
                if (
                    label.startswith("inserted:")
                    and isinstance(path_item_times, list)
                    and all(
                        isinstance(item, int) and not isinstance(item, bool) and item >= 0
                        for item in path_item_times
                    )
                    and isinstance(summary, dict)
                    and summary.get("status") == "success"
                ):
                    inserted_path_item_times_ms[label.removeprefix("inserted:")] = path_item_times
            conflicts = self._request(
                "GET",
                _conflicts_path(snapshot, candidate_timetable_id, work_schedule_group_id),
            )
            if not isinstance(conflicts, list):
                raise EditoastError("Editoast returned an invalid conflict response")
            conflict_values = cast(list[object], conflicts)
            if not all(isinstance(conflict, dict) for conflict in conflict_values):
                raise EditoastError("Editoast returned an invalid conflict response")
            conflict_rows = cast(list[dict[str, Any]], conflict_values)
            candidate_conflict_signatures = _conflict_signatures(conflict_rows, candidate_id_map)
            conflicting_existing_service_ids = _new_conflict_existing_service_ids(
                conflict_rows,
                baseline_conflict_signatures,
                candidate_id_map,
            )
            reasons = simulation_failure_reasons
            baseline_remaining = Counter(baseline_conflict_signatures)
            for index, conflict in enumerate(conflict_rows):
                signature = _conflict_signatures([conflict], candidate_id_map)[0]
                is_new = baseline_remaining[signature] == 0
                if not is_new:
                    baseline_remaining[signature] -= 1
                if index < 20:
                    reasons.append(
                        _conflict_failure_reason(
                            conflict,
                            schedule_id_map=candidate_id_map,
                            source_by_id=source_by_id,
                            is_new=is_new,
                        )
                    )
            if len(conflict_rows) > 20:
                reasons.append(f"{len(conflict_rows) - 20} additional OSRD conflicts omitted")
            validation = CandidateValidation(
                pathfinding_valid=pathfinding_valid,
                simulation_valid=simulation_valid,
                conflict_count=len(conflict_rows),
                failure_reasons=tuple(reasons),
                conflict_signatures=candidate_conflict_signatures,
                baseline_conflict_signatures=baseline_conflict_signatures,
                inserted_travel_times_ms=inserted_travel_times_ms,
                existing_travel_times_ms=existing_travel_times_ms,
                inserted_path_item_times_ms=inserted_path_item_times_ms,
                conflicting_existing_service_ids=conflicting_existing_service_ids,
            )
        finally:
            if candidate_timetable_id is not None:
                try:
                    self._request("DELETE", f"/timetable/{candidate_timetable_id}")
                except Exception as error:  # cleanup must not hide original validation failures
                    cleanup_errors.append(f"temporary timetable cleanup failed: {error}")
            if candidate_set_id is not None:
                try:
                    self._request("DELETE", f"/train_schedule_sets/{candidate_set_id}")
                except Exception as error:  # cleanup must not hide original validation failures
                    cleanup_errors.append(f"temporary schedule-set cleanup failed: {error}")
        # Reject results if another user changed the timetable or infrastructure during
        # simulation; the recommendation must still be bound to the exact source snapshot.
        self._load_source(snapshot)
        return CandidateValidation(
            pathfinding_valid=validation.pathfinding_valid,
            simulation_valid=validation.simulation_valid,
            conflict_count=validation.conflict_count,
            failure_reasons=validation.failure_reasons + tuple(cleanup_errors),
            conflict_signatures=validation.conflict_signatures,
            baseline_conflict_signatures=validation.baseline_conflict_signatures,
            inserted_travel_times_ms=validation.inserted_travel_times_ms,
            existing_travel_times_ms=validation.existing_travel_times_ms,
            inserted_path_item_times_ms=validation.inserted_path_item_times_ms,
            conflicting_existing_service_ids=validation.conflicting_existing_service_ids,
        )

    @staticmethod
    def _created_schedule_ids(created: Any, expected_count: int) -> list[int]:
        if not isinstance(created, list):
            raise EditoastError("Editoast did not create the complete candidate timetable")
        rows = cast(list[object], created)
        if len(rows) != expected_count or not all(isinstance(row, dict) for row in rows):
            raise EditoastError("Editoast did not create the complete candidate timetable")
        schedules = cast(list[dict[str, Any]], rows)
        try:
            return [int(schedule["id"]) for schedule in schedules]
        except (KeyError, TypeError, ValueError) as error:
            raise EditoastError("Editoast returned a schedule without an ID") from error
