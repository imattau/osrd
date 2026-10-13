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
        if not isinstance(edits, list) or not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Candidate payload must include an ID and an edits list")
        edit_rows = cast(list[object], edits)

        # An empty edit list is the pass-through baseline: clone and validate the
        # exact source timetable through the same OSRD checks as edited candidates.
        timetable_type, source_schedules = self._load_source(snapshot)
        source_by_id = {str(schedule["id"]): schedule for schedule in source_schedules}
        if len(source_by_id) != len(source_schedules):
            raise UnsupportedSnapshot("Source timetable contains duplicate train schedule IDs")
        forms = {schedule_id: _schedule_form(row) for schedule_id, row in source_by_id.items()}
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
            if shift <= 0 or shift > 7200 or after - before != shift * 1000:
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

            for source_id in sorted(schedules_with_exceptions):
                created = self._request(
                    "POST",
                    f"/train_schedule_sets/{candidate_set_id}/train_schedules",
                    [candidate_forms[source_id]],
                )
                new_schedule_id = self._created_schedule_ids(created, 1)[0]
                candidate_schedule_by_source_id[source_id] = new_schedule_id
                candidate_schedule_ids.append(new_schedule_id)

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
            for schedule_id in candidate_schedule_ids:
                schedule_summary_value = summaries.get(str(schedule_id))
                schedule_summary = (
                    cast(dict[str, Any], schedule_summary_value)
                    if isinstance(schedule_summary_value, dict)
                    else None
                )
                if not isinstance(schedule_summary, dict):
                    statuses.append("missing_summary")
                    continue
                base_summary_value = schedule_summary.get("train_schedule")
                base_summary = (
                    cast(dict[str, Any], base_summary_value)
                    if isinstance(base_summary_value, dict)
                    else None
                )
                statuses.append(
                    str(base_summary.get("status", "missing_summary"))
                    if base_summary is not None
                    else "missing_summary"
                )
                exception_summaries = schedule_summary.get("exceptions", {})
                if not isinstance(exception_summaries, dict):
                    statuses.append("missing_summary")
                    continue
                exception_summary_map = cast(dict[str, Any], exception_summaries)
                for exception_id in expected_exception_ids.get(schedule_id, set()):
                    exception_summary_value = exception_summary_map.get(exception_id)
                    exception_summary = (
                        cast(dict[str, Any], exception_summary_value)
                        if isinstance(exception_summary_value, dict)
                        else None
                    )
                    statuses.append(
                        str(exception_summary.get("status", "missing_summary"))
                        if exception_summary is not None
                        else "missing_summary"
                    )
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
