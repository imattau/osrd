from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from osrd_ops_intelligence.service.osrd_client import (
    MAX_SCHEDULE_SIMULATION_BATCH,
    EditoastError,
    EditoastOSRDClient,
    SnapshotRef,
    _conflict_signatures,
    _new_conflict_existing_service_ids,
    _simulation_failure_reason,
    compute_timetable_revision,
)


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            {
                "status": "pathfinding_input_error",
                "error_type": "rolling_stock_not_found",
                "rolling_stock_name": "slow_rolling_stock",
            },
            "could not find rolling stock 'slow_rolling_stock'",
        ),
        (
            {
                "status": "pathfinding_input_error",
                "error_type": "invalid_path_items",
                "items": [{"index": 0, "path_item": {}}, {"index": 2, "path_item": {}}],
            },
            "path item(s) 1, 3",
        ),
        (
            {"status": "pathfinding_input_error", "error_type": "not_enough_path_items"},
            "origin and destination",
        ),
        (
            {"status": "pathfinding_not_found", "error_type": "not_found_in_tracks"},
            "no continuous track path was found",
        ),
        (
            {"status": "simulation_failed", "error_type": "invalid_physics"},
            "simulation failed for new service Express 2 (invalid_physics)",
        ),
    ],
)
def test_simulation_failure_reason_explains_osrd_validation_error(
    summary: dict[str, Any], expected: str
) -> None:
    reason = _simulation_failure_reason(summary, "new service Express 2")

    assert reason is not None
    assert expected in reason


def test_successful_simulation_has_no_failure_reason() -> None:
    assert _simulation_failure_reason({"status": "success"}, "new service Express 2") is None


def test_new_osrd_conflicts_identify_only_affected_source_services() -> None:
    candidate_id_map = {
        "100": "inserted:New service-01",
        "101": "source:7",
        "201": "source:8",
    }
    baseline_conflict = {
        "conflict_type": "same_zone",
        "train_ids": [
            {"train_schedule_id": 101, "index": 0},
            {"train_schedule_id": 201, "index": 0},
        ],
        "start_time": 10_000,
        "duration": 60_000,
        "work_schedule_ids": [],
        "requirements": [],
    }
    new_conflict = {
        **baseline_conflict,
        "train_ids": [
            {"train_schedule_id": 100, "index": 0},
            {"train_schedule_id": 101, "index": 0},
        ],
        "start_time": 20_000,
    }
    baseline = _conflict_signatures([baseline_conflict], candidate_id_map)

    affected = _new_conflict_existing_service_ids(
        [baseline_conflict, new_conflict], baseline, candidate_id_map
    )

    assert affected == ("7",)


def test_schedule_travel_time_reads_are_batched_to_bound_request_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EditoastOSRDClient("http://unused")
    snapshot = SnapshotRef("1", "v1", "2", "revision", "test")
    requested_batches: list[list[int]] = []

    def fake_request(method: str, path: str, payload: Any) -> dict[str, Any]:
        assert method == "POST"
        assert path == "/train_schedules/simulation_summary"
        assert isinstance(payload, dict)
        ids = payload["ids"]
        assert isinstance(ids, list)
        requested_batches.append(ids)
        return {
            str(schedule_id): {
                "train_schedule": {"status": "success", "time": schedule_id * 1_000}
            }
            for schedule_id in ids
        }

    monkeypatch.setattr(client, "_load_source", lambda _snapshot: None)
    monkeypatch.setattr(client, "_request", fake_request)
    schedule_ids = [str(schedule_id) for schedule_id in range(1, MAX_SCHEDULE_SIMULATION_BATCH + 2)]

    result = client._get_schedule_travel_times(snapshot, schedule_ids)

    assert [len(batch) for batch in requested_batches] == [MAX_SCHEDULE_SIMULATION_BATCH, 1]
    assert len(result) == len(schedule_ids)
    assert result[schedule_ids[-1]] == (MAX_SCHEDULE_SIMULATION_BATCH + 1) * 1_000


def test_empty_linked_timetable_is_a_valid_snapshot_for_from_scratch_planning() -> None:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, body: Any) -> None:
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/infra/1":
                return self._respond({"version": 4})
            if self.path == "/version":
                return self._respond({"git_describe": "test-build"})
            if self.path == "/timetable/42/train_schedule_sets":
                return self._respond([{"id": 420, "timetable_type": "HOURLY"}])
            if self.path.startswith("/timetable/42/train_schedules?"):
                return self._respond({"results": [], "next": None})
            self.send_error(404)

        def log_message(self, _format: str, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EditoastOSRDClient(f"http://127.0.0.1:{server.server_port}")
        snapshot = client._describe_snapshot("1", "42")
        metadata = client._get_schedule_metadata(snapshot)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert snapshot.timetable_revision == compute_timetable_revision("HOURLY", [])
    assert metadata.timetable_type == "HOURLY"
    assert metadata.start_times_ms == {}
    assert metadata.train_names == {}
    assert metadata.schedule_forms_by_id == {}


def test_candidate_validation_copies_linked_schedules_and_cleans_up() -> None:
    source_schedules: list[dict[str, Any]] = [
        {
            "id": 31,
            "train_schedule_set_id": 21,
            "train_name": "service-31",
            "rolling_stock_name": "fast_rolling_stock",
            "start_time": 1_000,
            "path": [{"track": "A", "offset": 0}],
            "constraint_distribution": {"distribution": "MARECO"},
            "comfort": "STANDARD",
            "paced": {
                "time_window": "PT2H",
                "interval": "PT15M",
                "exceptions": [
                    {
                        "id": 777,
                        "key": "source-exception",
                        "occurrence_index": 1,
                        "disabled": False,
                        "labels": {"value": ["exception-label"]},
                        "start_time": {"value": 3_000},
                    }
                ],
            },
        },
        {
            "id": 32,
            "train_schedule_set_id": 22,
            "train_name": "service-32",
            "rolling_stock_name": "fast_rolling_stock",
            "start_time": 2_000,
            "path": [{"track": "B", "offset": 0}],
            "constraint_distribution": {"distribution": "MARECO"},
            "comfort": "STANDARD",
        },
    ]
    calls: list[tuple[str, str, Any]] = []
    created_schedule_ids: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, body: Any = None, status: int = 200) -> None:
            encoded = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            if encoded:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def do_GET(self) -> None:
            calls.append(("GET", self.path, None))
            if self.path == "/infra/1":
                return self._respond({"version": 4})
            if self.path == "/version":
                return self._respond({"git_describe": "test-build"})
            if self.path == "/timetable/10/train_schedule_sets":
                return self._respond(
                    [
                        {"id": 21, "timetable_type": "STANDARD"},
                        {"id": 22, "timetable_type": "STANDARD"},
                    ]
                )
            if self.path.startswith("/timetable/10/train_schedules?"):
                return self._respond({"results": source_schedules, "next": None})
            if self.path == "/timetable/10/conflicts?infra_id=1":
                return self._respond([])
            if self.path == "/timetable/900/conflicts?infra_id=1":
                return self._respond([])
            self._respond({"error": f"unexpected GET {self.path}"}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            calls.append(("POST", self.path, body))
            if self.path == "/timetable":
                return self._respond({"timetable_id": 900}, 201)
            if self.path == "/train_schedule_sets":
                return self._respond({"id": 901}, 201)
            if self.path == "/timetable/900/train_schedule_sets":
                return self._respond(None, 204)
            if self.path == "/train_schedule_sets/901/train_schedules":
                created = []
                for form in body:
                    new_id = 9002 if form["train_name"] == "service-31" else 9001
                    created_schedule_ids[form["train_name"]] = new_id
                    created.append({"id": new_id})
                return self._respond(created, 201)
            if self.path == "/timetable/900/train_schedule_exception":
                assert body["train_schedule_id"] == 9002
                assert body["occurrence_index"] == 1
                assert body["change_groups"]["labels"] == {"value": ["exception-label"]}
                assert body["change_groups"]["start_time"] == {"value": 63_000}
                return self._respond({"id": 9100}, 200)
            if self.path == "/train_schedules/simulation_summary":
                response = {}
                for schedule_id in body["ids"]:
                    response[str(schedule_id)] = {
                        "train_schedule": {"status": "success"},
                        "exceptions": (
                            {"9100": {"status": "success"}} if schedule_id == 9002 else {}
                        ),
                    }
                return self._respond(response)
            self._respond({"error": f"unexpected POST {self.path}"}, 404)

        def do_DELETE(self) -> None:
            calls.append(("DELETE", self.path, None))
            self._respond(None, 204)

        def log_message(self, _format: str, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EditoastOSRDClient(f"http://127.0.0.1:{server.server_port}")
        snapshot = client._describe_snapshot("1", "10")
        assert snapshot.timetable_revision == compute_timetable_revision(
            "STANDARD", source_schedules
        )
        result = client._validate_candidate(
            snapshot,
            {
                "candidate_id": "candidate-1",
                "snapshot": {
                    "infra_id": snapshot.infra_id,
                    "infra_version": snapshot.infra_version,
                    "timetable_id": snapshot.timetable_id,
                    "timetable_revision": snapshot.timetable_revision,
                    "osrd_version": snapshot.osrd_version,
                },
                "edits": [
                    {
                        "train_schedule_id": "31",
                        "before_start_time_ms": 1_000,
                        "after_start_time_ms": 61_000,
                        "shift_seconds": 60,
                    }
                ],
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result.validated
    copies = [
        form
        for method, path, body in calls
        if method == "POST" and path == "/train_schedule_sets/901/train_schedules"
        for form in body
    ]
    assert len(copies) == 2
    assert {form["train_name"] for form in copies} == {"service-31", "service-32"}
    copied_paced = next(form for form in copies if form["train_name"] == "service-31")
    assert copied_paced["start_time"] == 61_000
    assert copied_paced["paced"]["exceptions"] == []
    exception_calls = [
        body
        for method, path, body in calls
        if method == "POST" and path == "/timetable/900/train_schedule_exception"
    ]
    assert len(exception_calls) == 1
    assert exception_calls[0]["train_schedule_id"] == created_schedule_ids["service-31"]
    assert exception_calls[0]["change_groups"]["labels"]["value"] == ["exception-label"]
    assert exception_calls[0]["change_groups"]["start_time"]["value"] == 63_000
    assert ("DELETE", "/timetable/900", None) in calls
    assert ("DELETE", "/train_schedule_sets/901", None) in calls


def test_candidate_validation_cleans_up_after_partial_creation_failure() -> None:
    source_schedule = {
        "id": 31,
        "train_schedule_set_id": 21,
        "train_name": "service-31",
        "rolling_stock_name": "fast_rolling_stock",
        "start_time": 1_000,
        "path": [{"track": "A", "offset": 0}],
        "constraint_distribution": {"distribution": "MARECO"},
    }
    calls: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, body: Any = None, status: int = 200) -> None:
            encoded = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/infra/1":
                return self._respond({"version": 4})
            if self.path == "/version":
                return self._respond({"git_describe": "test-build"})
            if self.path == "/timetable/10/train_schedule_sets":
                return self._respond([{"id": 21, "timetable_type": "STANDARD"}])
            if self.path.startswith("/timetable/10/train_schedules?"):
                return self._respond({"results": [source_schedule], "next": None})
            if self.path == "/timetable/10/conflicts?infra_id=1":
                return self._respond([])
            self._respond({}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if self.path == "/timetable":
                return self._respond({"timetable_id": 900}, 201)
            if self.path == "/train_schedule_sets":
                return self._respond({"id": 901}, 201)
            if self.path == "/timetable/900/train_schedule_sets":
                return self._respond(None, 204)
            if self.path == "/train_schedule_sets/901/train_schedules":
                return self._respond({"error": "injected creation failure"}, 500)
            self._respond({}, 404)

        def do_DELETE(self) -> None:
            calls.append(("DELETE", self.path))
            self._respond(None, 204)

        def log_message(self, _format: str, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EditoastOSRDClient(f"http://127.0.0.1:{server.server_port}")
        snapshot = SnapshotRef(
            infra_id="1",
            infra_version="4",
            timetable_id="10",
            timetable_revision=compute_timetable_revision("STANDARD", [source_schedule]),
            osrd_version="test-build",
        )
        with pytest.raises(EditoastError, match="injected creation failure"):
            client._validate_candidate(
                snapshot,
                {
                    "candidate_id": "candidate-2",
                    "snapshot": {
                        "infra_id": snapshot.infra_id,
                        "infra_version": snapshot.infra_version,
                        "timetable_id": snapshot.timetable_id,
                        "timetable_revision": snapshot.timetable_revision,
                        "osrd_version": snapshot.osrd_version,
                    },
                    "edits": [
                        {
                            "train_schedule_id": "31",
                            "before_start_time_ms": 1_000,
                            "after_start_time_ms": 61_000,
                            "shift_seconds": 60,
                        }
                    ],
                },
            )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert ("DELETE", "/timetable/900") in calls
    assert ("DELETE", "/train_schedule_sets/901") in calls


@pytest.mark.parametrize(
    ("source_schedule", "before_seconds", "after_seconds", "expected_schedule"),
    [
        (
            [{"at": "mid", "stop_for": "PT30S"}, {"at": "destination", "stop_for": "P0D"}],
            30,
            0,
            [{"at": "destination", "stop_for": "P0D"}],
        ),
        (
            [{"at": "destination", "stop_for": "P0D"}],
            0,
            90,
            [{"at": "mid", "stop_for": "PT90S"}, {"at": "destination", "stop_for": "P0D"}],
        ),
    ],
)
def test_candidate_validation_applies_existing_service_dwell_changes(
    monkeypatch: pytest.MonkeyPatch,
    source_schedule: list[dict[str, Any]],
    before_seconds: int,
    after_seconds: int,
    expected_schedule: list[dict[str, Any]],
) -> None:
    source = {
        "id": 42,
        "train_schedule_set_id": 21,
        "train_name": "existing",
        "rolling_stock_name": "test_stock",
        "start_time": 1_000,
        "path": [
            {"key": "origin", "location": {"type": "track_offset", "track": "A", "offset": 0}},
            {"key": "mid", "location": {"type": "track_offset", "track": "B", "offset": 500}},
            {
                "key": "destination",
                "location": {"type": "track_offset", "track": "C", "offset": 1000},
            },
        ],
        "schedule": source_schedule,
        "constraint_distribution": "STANDARD",
    }
    client = EditoastOSRDClient("http://unused")
    monkeypatch.setattr(client, "_load_source", lambda _snapshot: ("CALENDAR", [source]))
    created_forms: list[dict[str, Any]] = []

    def fake_request(method: str, path: str, body: Any = None) -> Any:
        if method == "GET" and "/conflicts?" in path:
            return []
        if method == "POST" and path == "/timetable":
            return {"timetable_id": 900}
        if method == "POST" and path == "/train_schedule_sets":
            return {"id": 901}
        if method == "POST" and path == "/timetable/900/train_schedule_sets":
            return None
        if method == "POST" and path == "/train_schedule_sets/901/train_schedules":
            created_forms.extend(body)
            return [{"id": 9001}]
        if method == "POST" and path == "/train_schedules/simulation_summary":
            return {"9001": {"train_schedule": {"status": "success", "time": 120_000}}}
        if method == "DELETE":
            return None
        raise AssertionError(f"Unexpected OSRD request: {method} {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    snapshot = SnapshotRef("1", "4", "10", "revision", "test-build")
    result = client._validate_candidate(
        snapshot,
        {
            "candidate_id": "dwell-candidate",
            "snapshot": {
                "infra_id": snapshot.infra_id,
                "infra_version": snapshot.infra_version,
                "timetable_id": snapshot.timetable_id,
                "timetable_revision": snapshot.timetable_revision,
                "osrd_version": snapshot.osrd_version,
            },
            "edits": [],
            "existing_service_dwell_edits": [
                {
                    "train_schedule_id": "42",
                    "waypoint_key": "mid",
                    "before_dwell_seconds": before_seconds,
                    "after_dwell_seconds": after_seconds,
                }
            ],
            "insertions": [],
        },
    )

    assert result.simulation_valid
    assert created_forms[0]["schedule"] == expected_schedule
