from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from osrd_ops_intelligence.api.contracts import (
    AffectedResource,
    Incident,
    IncidentKind,
    PlanningRequest,
    SnapshotBinding,
)

NOW = datetime(2026, 10, 13, 8, tzinfo=UTC)


def _incident(**overrides: object) -> Incident:
    fields: dict[str, object] = {
        "incident_id": "inc-1",
        "kind": IncidentKind.TRACK_CLOSURE,
        "affected_resources": [AffectedResource(resource_type="track_section", resource_id="TA0")],
        "starts_at": NOW,
        "ends_at": NOW + timedelta(hours=1),
        "confidence": 0.9,
        "source": "test",
    }
    fields.update(overrides)
    return Incident.model_validate(fields)


def test_incident_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        _incident(starts_at=datetime(2026, 10, 13, 8))


def test_incident_interval_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        _incident(ends_at=NOW)


def test_resource_offsets_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        AffectedResource(
            resource_type="track_section", resource_id="TA0", start_offset_m=10, end_offset_m=5
        )


def test_request_rejects_unknown_fields() -> None:
    snapshot = SnapshotBinding(
        infra_id="1",
        infra_version="v1",
        timetable_id="1",
        timetable_revision="r1",
        osrd_version="x",
    )
    with pytest.raises(ValidationError):
        PlanningRequest.model_validate(
            {
                "scenario_id": "1",
                "snapshot": snapshot.model_dump(),
                "incident": _incident().model_dump(),
                "horizon_start": NOW,
                "horizon_end": NOW + timedelta(hours=2),
                "surprise": True,
            }
        )
