"""Snapshot-bound Editoast adapter for isolated candidate validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast


@dataclass(frozen=True)
class SnapshotRef:
    infra_id: str
    infra_version: str
    timetable_id: str
    timetable_revision: str
    osrd_version: str


@dataclass(frozen=True)
class CandidateValidation:
    pathfinding_valid: bool
    simulation_valid: bool
    conflict_count: int | None
    failure_reasons: tuple[str, ...] = ()

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
    "path",
    "power_restrictions",
    "rolling_stock_name",
    "schedule",
    "speed_limit_tag",
    "start_time",
    "train_name",
)


def _schedule_form(schedule: dict[str, Any]) -> dict[str, Any]:
    form = {key: schedule[key] for key in _TRAIN_SCHEDULE_FIELDS if key in schedule}
    required = {"train_name", "rolling_stock_name", "start_time", "path", "constraint_distribution"}
    missing = required - form.keys()
    if missing:
        raise UnsupportedSnapshot(f"Train schedule is missing required fields: {sorted(missing)}")
    if schedule.get("paced") is not None:
        raise UnsupportedSnapshot("Paced trains are not supported by the isolated validator")
    return form


def compute_timetable_revision(
    timetable_type: str, schedules: list[dict[str, Any]]
) -> str:
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
        if not schedules:
            raise UnsupportedSnapshot("Source timetable contains no train schedules")
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
        candidate_id = candidate_payload.get("candidate_id")
        if not isinstance(edits, list) or not isinstance(candidate_id, str):
            raise ValueError("Candidate payload must include an ID and non-empty edits")
        edit_rows = cast(list[object], edits)
        if not edit_rows:
            raise ValueError("Candidate payload must include an ID and non-empty edits")

        timetable_type, source_schedules = self._load_source(snapshot)
        source_by_id = {str(schedule["id"]): schedule for schedule in source_schedules}
        if len(source_by_id) != len(source_schedules):
            raise UnsupportedSnapshot("Source timetable contains duplicate train schedule IDs")
        forms = {schedule_id: _schedule_form(row) for schedule_id, row in source_by_id.items()}
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
            if shift <= 0 or shift > 7200 or after - before != shift * 1000:
                raise ValueError("Candidate departure shift is outside supported bounds")
            if forms[schedule_id]["start_time"] != before:
                raise UnsupportedSnapshot(
                    "Candidate edit was generated from a stale departure time"
                )
            forms[schedule_id]["start_time"] = after
            edited_ids.add(schedule_id)

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
            created = self._request(
                "POST",
                f"/train_schedule_sets/{candidate_set_id}/train_schedules",
                [forms[schedule_id] for schedule_id in sorted(forms)],
            )
            if not isinstance(created, list):
                raise EditoastError("Editoast did not create the complete candidate timetable")
            created_rows = cast(list[object], created)
            if (
                not all(isinstance(item, dict) for item in created_rows)
                or len(created_rows) != len(forms)
            ):
                raise EditoastError("Editoast did not create the complete candidate timetable")
            created_schedules = cast(list[dict[str, Any]], created_rows)
            candidate_schedule_ids = [int(item["id"]) for item in created_schedules]
            summaries = cast(dict[str, Any], self._request(
                "POST",
                "/train_schedules/simulation_summary",
                {
                    "infra_id": int(snapshot.infra_id),
                    "timetable_id": candidate_timetable_id,
                    "ids": candidate_schedule_ids,
                },
            ))
            summary_rows = [
                summaries.get(str(schedule_id), {}).get("train_schedule", {})
                for schedule_id in candidate_schedule_ids
            ]
            statuses = [row.get("status", "missing_summary") for row in summary_rows]
            pathfinding_valid = all(
                status
                not in {"pathfinding_not_found", "pathfinding_input_error", "missing_summary"}
                for status in statuses
            )
            simulation_valid = all(status == "success" for status in statuses)
            conflicts = self._request(
                "GET",
                f"/timetable/{candidate_timetable_id}/conflicts?infra_id={int(snapshot.infra_id)}",
            )
            if not isinstance(conflicts, list):
                raise EditoastError("Editoast returned an invalid conflict response")
            conflict_values = cast(list[object], conflicts)
            if not all(isinstance(conflict, dict) for conflict in conflict_values):
                raise EditoastError("Editoast returned an invalid conflict response")
            conflict_rows = cast(list[dict[str, Any]], conflict_values)
            reasons = [
                f"train schedule simulation status: {status}"
                for status in sorted(set(statuses))
                if status != "success"
            ]
            reasons.extend(
                f"OSRD conflict: {conflict.get('conflict_type', 'unknown')}"
                for conflict in conflict_rows
            )
            validation = CandidateValidation(
                pathfinding_valid=pathfinding_valid,
                simulation_valid=simulation_valid,
                conflict_count=len(conflict_rows),
                failure_reasons=tuple(reasons),
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
        return CandidateValidation(
            pathfinding_valid=validation.pathfinding_valid,
            simulation_valid=validation.simulation_valid,
            conflict_count=validation.conflict_count,
            failure_reasons=validation.failure_reasons + tuple(cleanup_errors),
        )
