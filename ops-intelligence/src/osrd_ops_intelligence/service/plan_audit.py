"""Append-only SQLite journal for plans and explicitly submitted reviews."""

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
    ServicePlanReviewReceipt,
    ServicePlanReviewSubmission,
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
    """Store immutable plan results and explicitly submitted service-plan reviews.

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
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS service_plan_reviews (
                        review_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        review_sha256 TEXT NOT NULL,
                        record_json TEXT NOT NULL
                    )"""
                )
                connection.execute(
                    """CREATE UNIQUE INDEX IF NOT EXISTS service_plan_reviews_request_id
                       ON service_plan_reviews (request_id)"""
                )
                connection.executescript(
                    """CREATE TRIGGER IF NOT EXISTS plan_records_no_update
                       BEFORE UPDATE ON plan_records
                       BEGIN SELECT RAISE(ABORT, 'plan records are immutable'); END;
                       CREATE TRIGGER IF NOT EXISTS plan_records_no_delete
                       BEFORE DELETE ON plan_records
                       BEGIN SELECT RAISE(ABORT, 'plan records are immutable'); END;
                       CREATE TRIGGER IF NOT EXISTS service_plan_reviews_no_update
                       BEFORE UPDATE ON service_plan_reviews
                       BEGIN SELECT RAISE(ABORT, 'service plan reviews are immutable'); END;
                       CREATE TRIGGER IF NOT EXISTS service_plan_reviews_no_delete
                       BEFORE DELETE ON service_plan_reviews
                       BEGIN SELECT RAISE(ABORT, 'service plan reviews are immutable'); END;"""
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

    def record_service_plan_review(
        self, submission: ServicePlanReviewSubmission
    ) -> ServicePlanReviewReceipt:
        record_json = _canonical_json(submission.record)
        review_sha256 = _digest(record_json)
        nested_request = submission.record["request"]
        service_request = nested_request.get("service") or nested_request.get("hourly_service")
        if not isinstance(service_request, dict):
            raise AuditStoreError("Validated service-plan review has no timetable request")
        request_id = UUID(service_request["request_id"])
        created_at = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.path, timeout=30) as connection:
                connection.execute(
                    """INSERT INTO service_plan_reviews (
                        review_id, request_id, created_at, review_sha256, record_json
                    ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(submission.review_id),
                        str(request_id),
                        created_at,
                        review_sha256,
                        record_json,
                    ),
                )
        except sqlite3.IntegrityError:
            try:
                with sqlite3.connect(self.path, timeout=30) as connection:
                    row = connection.execute(
                        """SELECT request_id, created_at, review_sha256, record_json
                           FROM service_plan_reviews WHERE review_id = ?""",
                        (str(submission.review_id),),
                    ).fetchone()
            except sqlite3.Error as error:
                raise AuditStoreError(
                    f"Could not read saved service-plan review: {error}"
                ) from error
            if row is None or row[0] != str(request_id) or row[3] != record_json:
                raise AuditConflict(
                    "request or review ID already identifies a different planner review"
                ) from None
            created_at, review_sha256 = row[1], row[2]
        except sqlite3.Error as error:
            raise AuditStoreError(f"Could not persist service-plan review: {error}") from error
        return ServicePlanReviewReceipt(
            review_id=submission.review_id,
            request_id=request_id,
            created_at=created_at,
            review_sha256=review_sha256,
        )
