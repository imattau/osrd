"""Append-only SQLite journal for completed advisory planning requests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from osrd_ops_intelligence.api.contracts import (
    PlanAuditRecord,
    PlanningRequest,
    PlanningResponse,
)


class AuditStoreError(RuntimeError):
    """The audit journal could not safely read or persist a plan record."""


class AuditConflict(RuntimeError):
    """A request ID was reused for a different immutable planning request."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class PlanAuditStore:
    """Store complete request/result pairs once, keyed by the client request ID.

    Update and delete triggers make records immutable through this service's SQLite
    connection. Production deployments should place this journal on a durable volume
    or replace the adapter with Editoast's authoritative audit storage.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.path, timeout=30) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS plan_records (
                        request_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        input_sha256 TEXT NOT NULL,
                        result_sha256 TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    )"""
                )
                connection.executescript(
                    """CREATE TRIGGER IF NOT EXISTS plan_records_no_update
                       BEFORE UPDATE ON plan_records
                       BEGIN SELECT RAISE(ABORT, 'plan records are immutable'); END;
                       CREATE TRIGGER IF NOT EXISTS plan_records_no_delete
                       BEFORE DELETE ON plan_records
                       BEGIN SELECT RAISE(ABORT, 'plan records are immutable'); END;"""
                )
        except sqlite3.Error as error:
            raise AuditStoreError(
                f"Could not initialize planning audit journal: {error}"
            ) from error

    @staticmethod
    def request_digest(request: PlanningRequest) -> str:
        request_json = _canonical_json(request.model_dump(mode="json"))
        return _digest(request_json)

    def get(self, request_id: UUID) -> PlanAuditRecord | None:
        try:
            with sqlite3.connect(self.path, timeout=30) as connection:
                row = connection.execute(
                    """SELECT created_at, input_sha256, result_sha256, request_json, result_json
                       FROM plan_records WHERE request_id = ?""",
                    (str(request_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise AuditStoreError(f"Could not read planning audit journal: {error}") from error
        if row is None:
            return None
        created_at, input_sha256, result_sha256, request_json, result_json = row
        if _digest(request_json) != input_sha256 or _digest(result_json) != result_sha256:
            raise AuditStoreError(f"Planning audit record {request_id} failed digest verification")
        return PlanAuditRecord.model_validate(
            {
                "request_id": request_id,
                "status": "completed",
                "created_at": created_at,
                "input_sha256": input_sha256,
                "result_sha256": result_sha256,
                "request": json.loads(request_json),
                "result": json.loads(result_json),
            }
        )

    def replay(self, request: PlanningRequest) -> PlanAuditRecord | None:
        record = self.get(request.request_id)
        if record is not None and record.input_sha256 != self.request_digest(request):
            raise AuditConflict("request_id already identifies a different immutable plan request")
        return record

    def record(self, request: PlanningRequest, result: PlanningResponse) -> PlanAuditRecord:
        request_json = _canonical_json(request.model_dump(mode="json"))
        result_json = _canonical_json(result.model_dump(mode="json"))
        input_sha256 = _digest(request_json)
        result_sha256 = _digest(result_json)
        created_at = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.path, timeout=30) as connection:
                connection.execute(
                    """INSERT INTO plan_records (
                        request_id, created_at, input_sha256, result_sha256,
                        request_json, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(request.request_id),
                        created_at,
                        input_sha256,
                        result_sha256,
                        request_json,
                        result_json,
                    ),
                )
        except sqlite3.IntegrityError:
            prior = self.replay(request)
            if prior is not None:
                return prior
            raise AuditStoreError("Could not persist completed planning record") from None
        except sqlite3.Error as error:
            raise AuditStoreError(
                f"Could not persist completed planning record: {error}"
            ) from error
        record = self.get(request.request_id)
        if record is None:
            raise AuditStoreError("Planning record was not readable after persistence")
        return record
