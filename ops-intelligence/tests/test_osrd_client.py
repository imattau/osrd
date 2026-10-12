from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from osrd_ops_intelligence.service.osrd_client import (
    EditoastError,
    EditoastOSRDClient,
    SnapshotRef,
    compute_timetable_revision,
)


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
                return self._respond(
                    [{"id": 9001 + index} for index, _ in enumerate(body)], 201
                )
            if self.path == "/train_schedules/simulation_summary":
                return self._respond(
                    {
                        str(schedule_id): {"train_schedule": {"status": "success"}}
                        for schedule_id in body["ids"]
                    }
                )
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
    copied = next(
        body
        for method, path, body in calls
        if method == "POST" and path == "/train_schedule_sets/901/train_schedules"
    )
    assert len(copied) == 2
    assert {form["train_name"] for form in copied} == {"service-31", "service-32"}
    assert copied[0]["start_time"] == 61_000
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

