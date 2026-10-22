from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from osrd_ops_intelligence.api.contracts import (
    AffectedResource,
    CompleteTimetablePlanRequest,
    Incident,
    IncidentKind,
    HourlyServiceRequest,
    PatronageProfile,
    PatronageSource,
    PatronageStation,
    PlanningRequest,
    ServicePathWaypoint,
    ServicePlanRequest,
    ServiceRequest,
    SnapshotBinding,
)
from osrd_ops_intelligence.service.service_requests import build_service_schedules

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


def test_max_journey_time_constraint_is_scoped_to_complete_plans() -> None:
    snapshot = SnapshotBinding(
        infra_id="1",
        infra_version="v1",
        timetable_id="1",
        timetable_revision="r1",
        osrd_version="x",
    )
    service = _service(snapshot).model_copy(update={"max_travel_time_seconds": 1_800})
    with pytest.raises(ValidationError, match="only supported for complete timetable plans"):
        ServicePlanRequest(service=service)


def _service(snapshot: SnapshotBinding, name: str = "Pilot") -> ServiceRequest:
    return ServiceRequest(
        snapshot=snapshot,
        train_name=name,
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin", location={"type": "track_offset", "track": "T1", "offset": 0}
            ),
            ServicePathWaypoint(
                key="destination",
                location={"type": "track_offset", "track": "T2", "offset": 1000},
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=1800,
    )


def test_service_weekdays_expand_daily_local_windows_and_reset_frequency() -> None:
    snapshot = SnapshotBinding(
        infra_id="1", infra_version="v1", timetable_id="1", timetable_revision="r1", osrd_version="x"
    )
    service = _service(snapshot).model_copy(update={
        "operating_window_start": datetime(2026, 10, 12, 8, tzinfo=UTC),
        "operating_window_end": datetime(2026, 10, 16, 10, tzinfo=UTC),
        "frequency_seconds": 3600,
        "operating_weekdays": [0, 2, 4],
    })

    windows = service.operating_windows()

    assert [start.weekday() for start, _ in windows] == [0, 2, 4]
    assert [start.hour for start, _ in windows] == [8, 8, 8]
    assert service.departure_count == 6
    schedules = build_service_schedules(service)
    assert [
        datetime.fromtimestamp(schedule["start_time"] / 1000, tz=UTC).weekday()
        for schedule in schedules
    ] == [0, 0, 2, 2, 4, 4]


def test_service_rejects_weekday_that_does_not_occur_in_date_range() -> None:
    snapshot = SnapshotBinding(
        infra_id="1", infra_version="v1", timetable_id="1", timetable_revision="r1", osrd_version="x"
    )
    service = _service(snapshot).model_copy(update={"operating_weekdays": [6]})
    with pytest.raises(ValueError, match="select no days"):
        service.operating_windows()


def _patronage_profile(infra_id: str = "1") -> PatronageProfile:
    return PatronageProfile(
        schema_version="station_patronage_profile_v1",
        dataset_id="pilot",
        infra_id=infra_id,
        infra_version="v1",
        source=PatronageSource(
            name="Pilot counts",
            license="CC0",
            observation_start="2026-01-01",
            observation_end="2026-01-31",
        ),
        timezone="UTC",
        stations=[PatronageStation(
            operational_point_id="op1",
            source_station_id="station1",
            name="Station 1",
            mapping_method="curated_crosswalk",
            buckets=[],
        )],
    )


def test_complete_plan_patronage_order_requires_matching_profile() -> None:
    snapshot = SnapshotBinding(
        infra_id="1", infra_version="v1", timetable_id="1",
        timetable_revision="r1", osrd_version="x",
    )
    service = _service(snapshot)
    with pytest.raises(ValidationError, match="requires a patronage profile"):
        CompleteTimetablePlanRequest(
            services=[service], candidate_order="station_patronage_coverage"
        )
    with pytest.raises(ValidationError, match="match the service snapshot"):
        CompleteTimetablePlanRequest(
            services=[service], candidate_order="station_patronage_coverage",
            patronage_profile=_patronage_profile("different-infra"),
        )
    request = CompleteTimetablePlanRequest(
        services=[service], candidate_order="station_patronage_coverage",
        patronage_profile=_patronage_profile(),
    )
    assert request.candidate_order == "station_patronage_coverage"


def test_hourly_patronage_requires_operating_weekdays_and_matching_snapshot() -> None:
    snapshot = SnapshotBinding(
        infra_id="1", infra_version="v1", timetable_id="1", timetable_revision="r1", osrd_version="x"
    )
    service = _service(snapshot)
    with pytest.raises(ValidationError, match="requires at least one operating weekday"):
        HourlyServiceRequest(
            snapshot=snapshot,
            train_name=service.train_name,
            rolling_stock_name=service.rolling_stock_name,
            path=service.path,
            frequency_seconds=1800,
            time_window_seconds=3600,
            patronage_profile=_patronage_profile(),
        )
    with pytest.raises(ValidationError, match="match the hourly service snapshot"):
        HourlyServiceRequest(
            snapshot=snapshot,
            train_name=service.train_name,
            rolling_stock_name=service.rolling_stock_name,
            path=service.path,
            frequency_seconds=1800,
            time_window_seconds=3600,
            operating_weekdays=[0],
            patronage_profile=_patronage_profile("different-infra"),
        )
