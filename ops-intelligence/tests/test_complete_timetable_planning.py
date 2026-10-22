import asyncio
from datetime import UTC, datetime
from typing import Any

from osrd_ops_intelligence.api.contracts import (
    CompleteTimetablePlanRequest,
    ExistingServiceRetimingPlanRequest,
    ExistingServiceDwellOptions,
    HourlyServiceRequest,
    PatronageBucket,
    PatronageProfile,
    PatronageSource,
    PatronageStation,
    ServicePathWaypoint,
    ServiceRequest,
    ServicePlanRequest,
    ServiceRetimingPolicy,
    SnapshotBinding,
)
from osrd_ops_intelligence.service.osrd_client import (
    CandidateValidation,
    ScheduleMetadata,
)
from osrd_ops_intelligence.service.service_requests import (
    plan_complete_timetable,
    plan_existing_service_retimings,
    plan_hourly_service_request,
    plan_service_request,
)


class StubOSRDClient:
    def __init__(
        self,
        *,
        simulated_travel_time_ms: int = 1_000,
        omit_travel_times: bool = False,
        schedule_metadata: ScheduleMetadata | None = None,
        baseline_travel_times_ms: dict[str, int] | None = None,
        existing_travel_times_ms: dict[str, int] | None = None,
    ) -> None:
        self.candidate_payloads: list[dict[str, object]] = []
        self.simulated_travel_time_ms = simulated_travel_time_ms
        self.omit_travel_times = omit_travel_times
        self.schedule_metadata = schedule_metadata or ScheduleMetadata(
            start_times_ms={}, train_names={}
        )
        self.baseline_travel_times_ms = baseline_travel_times_ms or {}
        self.existing_travel_times_ms = existing_travel_times_ms or {}

    async def get_schedule_metadata(self, *, snapshot: Any) -> ScheduleMetadata:
        return self.schedule_metadata

    async def get_schedule_travel_times(
        self, *, snapshot: Any, schedule_ids: list[str]
    ) -> dict[str, int]:
        return {
            schedule_id: self.baseline_travel_times_ms[schedule_id]
            for schedule_id in schedule_ids
            if schedule_id in self.baseline_travel_times_ms
        }

    async def get_route_alternatives(
        self,
        *,
        snapshot: Any,
        starting: dict[str, Any],
        ending: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return [
            {"track_ranges": [{"track": "TA", "begin": 0, "end": 1_000}]},
            {"track_ranges": [{"track": "TB", "begin": 0, "end": 1_200}]},
        ][:limit]

    async def validate_candidate(
        self, *, snapshot: Any, candidate_payload: dict[str, object]
    ) -> CandidateValidation:
        self.candidate_payloads.append(candidate_payload)
        insertions = candidate_payload["insertions"]
        assert isinstance(insertions, list)
        path_times = {
            str(insertion["train_name"]): [0, self.simulated_travel_time_ms]
            for insertion in insertions
            if isinstance(insertion, dict)
        } if not self.omit_travel_times else {}
        travel_times = {
            train_name: self.simulated_travel_time_ms for train_name in path_times
        }
        return CandidateValidation(
            pathfinding_valid=True,
            simulation_valid=True,
            conflict_count=0,
            inserted_travel_times_ms=travel_times,
            existing_travel_times_ms=self.existing_travel_times_ms,
            inserted_path_item_times_ms=path_times,
        )


def test_single_service_candidate_reports_weekday_expanded_departures() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Weekday Pilot",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin",
                location={"type": "track_offset", "track": "TA", "offset": 0},
            ),
            ServicePathWaypoint(
                key="destination",
                location={"type": "track_offset", "track": "TB", "offset": 1_000},
            ),
        ],
        operating_time_zone="UTC",
        operating_window_start=datetime(2026, 10, 12, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 16, 10, tzinfo=UTC),
        operating_weekdays=[0, 2, 4],
        frequency_seconds=3_600,
    )
    request = ServicePlanRequest(
        service=service,
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        phase_step_seconds=3_600,
        max_phase_options=1,
    )

    response = asyncio.run(plan_service_request(request, StubOSRDClient()))

    departures = response.candidates[0].inserted_departures
    assert [departure.weekday() for departure in departures] == [0, 0, 2, 2, 4, 4]
    assert [departure.hour for departure in departures] == [8, 9, 8, 9, 8, 9]


def test_hourly_patronage_coverage_uses_declared_weekdays_and_cycle_start() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = HourlyServiceRequest(
        snapshot=snapshot,
        train_name="Hourly Pilot",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin",
                location={
                    "type": "operational_point_part_reference",
                    "operational_point": {"type": "id", "operational_point": "OP1"},
                },
            ),
            ServicePathWaypoint(
                key="destination",
                location={
                    "type": "operational_point_part_reference",
                    "operational_point": {"type": "id", "operational_point": "OP2"},
                },
            ),
        ],
        frequency_seconds=1_800,
        time_window_seconds=3_600,
        phase_step_seconds=600,
        max_phase_options=3,
        candidate_order="station_patronage_coverage",
        operating_weekdays=[0, 2],
        cycle_start_minute=480,
        patronage_profile=PatronageProfile(
            schema_version="station_patronage_profile_v1",
            dataset_id="hourly-weekday-pilot",
            infra_id="infra-a",
            infra_version="version-1",
            source=PatronageSource(
                name="Synthetic hourly demand",
                license="CC0",
                observation_start="2026-01-01",
                observation_end="2026-01-31",
            ),
            timezone="UTC",
            stations=[
                PatronageStation(
                    operational_point_id="OP1",
                    source_station_id="station-1",
                    name="Origin",
                    mapping_method="curated_crosswalk",
                    buckets=[
                        PatronageBucket(
                            weekday=0,
                            start_minute=490,
                            duration_minutes=1,
                            mean_boardings=100,
                            mean_alightings=0,
                            observation_count=10,
                        ),
                        PatronageBucket(
                            weekday=0,
                            start_minute=520,
                            duration_minutes=1,
                            mean_boardings=100,
                            mean_alightings=0,
                            observation_count=10,
                        ),
                        PatronageBucket(
                            weekday=2,
                            start_minute=490,
                            duration_minutes=1,
                            mean_boardings=100,
                            mean_alightings=0,
                            observation_count=10,
                        ),
                        PatronageBucket(
                            weekday=2,
                            start_minute=520,
                            duration_minutes=1,
                            mean_boardings=100,
                            mean_alightings=0,
                            observation_count=10,
                        ),
                    ],
                ),
                PatronageStation(
                    operational_point_id="OP2",
                    source_station_id="station-2",
                    name="Destination",
                    mapping_method="curated_crosswalk",
                    buckets=[],
                ),
            ],
        ),
    )

    hourly_client = StubOSRDClient(
        schedule_metadata=ScheduleMetadata(start_times_ms={}, train_names={}, timetable_type="HOURLY")
    )
    response = asyncio.run(plan_hourly_service_request(service, hourly_client))

    assert response.candidates[0].departure_offset_seconds == 600
    assert response.candidates[0].patronage_coverage_status == "available"
    assert response.candidates[0].patronage_coverage is not None
    assert response.candidates[0].patronage_coverage.boarding_coverage == 1.0
    assert response.candidates[-1].patronage_coverage is not None
    assert response.candidates[-1].patronage_coverage.boarding_coverage == 0.0
    assert {call.arrival_weekday for call in response.candidates[0].station_call_timings} == {0, 2}


def test_complete_plan_uses_joint_patronage_coverage_after_osrd_checks() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Pilot",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin",
                location={
                    "type": "operational_point_part_reference",
                    "operational_point": {"type": "id", "operational_point": "OP1"},
                },
            ),
            ServicePathWaypoint(
                key="destination",
                location={
                    "type": "operational_point_part_reference",
                    "operational_point": {"type": "id", "operational_point": "OP2"},
                },
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 10, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    profile = PatronageProfile(
        schema_version="station_patronage_profile_v1",
        dataset_id="weekday-peak-pilot",
        infra_id="infra-a",
        infra_version="version-1",
        source=PatronageSource(
            name="Aggregated weekday counts",
            license="CC0",
            observation_start="2026-01-01",
            observation_end="2026-01-31",
        ),
        timezone="UTC",
        stations=[
            PatronageStation(
                operational_point_id="OP1",
                source_station_id="station-1",
                name="Origin",
                mapping_method="curated_crosswalk",
                buckets=[
                    PatronageBucket(
                        weekday=1,
                        start_minute=480,
                        duration_minutes=15,
                        mean_boardings=100,
                        mean_alightings=0,
                        observation_count=10,
                    ),
                    PatronageBucket(
                        weekday=1,
                        start_minute=540,
                        duration_minutes=15,
                        mean_boardings=100,
                        mean_alightings=0,
                        observation_count=10,
                    ),
                ],
            )
        ],
    )
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        candidate_order="station_patronage_coverage",
        patronage_profile=profile,
        phase_step_seconds=900,
        max_phase_options_per_service=2,
        max_candidates=2,
    )

    response = asyncio.run(plan_complete_timetable(request, StubOSRDClient()))

    assert len(response.candidates) == 2
    assert all(candidate.feasible for candidate in response.candidates)
    assert all(candidate.new_conflict_count == 0 for candidate in response.candidates)
    assert response.candidates[0].patronage_coverage is not None
    assert response.candidates[1].patronage_coverage is not None
    assert response.candidates[0].patronage_coverage.combined_event_coverage == 1.0
    assert response.candidates[1].patronage_coverage.combined_event_coverage == 0.0
    assert response.candidates[0].rank == 1
    assert response.candidates[0].patterns[0].departure_phase_seconds == 0


def test_complete_plan_searches_route_alternatives_jointly() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Route choice",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin",
                location={"type": "track_offset", "track": "TO", "offset": 0},
            ),
            ServicePathWaypoint(
                key="destination",
                location={"type": "track_offset", "track": "TD", "offset": 1_000_000},
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_route_alternatives_per_service=2,
        max_phase_options_per_service=1,
        max_candidates=4,
    )
    client = StubOSRDClient()

    response = asyncio.run(plan_complete_timetable(request, client))

    assert response.total_route_combinations == 2
    assert response.total_route_phase_combinations == 2
    assert response.total_candidate_combinations == 2
    assert {
        candidate.patterns[0].route_alternative_index for candidate in response.candidates
    } == {0, 1}
    assert {
        tuple(candidate.patterns[0].route_track_sections)
        for candidate in response.candidates
    } == {("TA",), ("TB",)}
    assert len(client.candidate_payloads) == 2
    inserted_tracks = {
        insertion["path"][1]["location"]["track"]
        for payload in client.candidate_payloads
        for insertion in payload["insertions"]
        if isinstance(insertion, dict)
    }
    assert inserted_tracks == {"TA", "TB"}


def test_complete_plan_searches_optional_stop_patterns_jointly() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Optional calls",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin", location={"type": "track_offset", "track": "TO", "offset": 0}
            ),
            ServicePathWaypoint(
                key="mid-a",
                location={"type": "track_offset", "track": "TA", "offset": 500},
                stop_for_seconds=45,
                optional_stop=True,
            ),
            ServicePathWaypoint(
                key="mid-b",
                location={"type": "track_offset", "track": "TB", "offset": 500},
                stop_for_seconds=60,
                optional_stop=True,
            ),
            ServicePathWaypoint(
                key="destination", location={"type": "track_offset", "track": "TD", "offset": 1_000}
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    client = StubOSRDClient()
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_phase_options_per_service=1,
        max_stop_pattern_options_per_service=4,
        max_candidates=4,
    )

    response = asyncio.run(plan_complete_timetable(request, client))

    assert response.total_stop_pattern_combinations == 4
    assert response.total_route_phase_stop_combinations == 4
    assert response.total_candidate_combinations == 4
    patterns = [candidate.patterns[0] for candidate in response.candidates]
    assert {tuple(pattern.selected_optional_stop_keys) for pattern in patterns} == {
        (), ("mid-a",), ("mid-b",), ("mid-a", "mid-b")
    }
    assert {tuple(pattern.omitted_optional_stop_keys) for pattern in patterns} == {
        (), ("mid-a",), ("mid-b",), ("mid-a", "mid-b")
    }
    scheduled_waypoints = {
        stop["at"]
        for payload in client.candidate_payloads
        for insertion in payload["insertions"]
        if isinstance(insertion, dict)
        for stop in insertion["schedule"]
        if "stop_for" in stop
    }
    assert scheduled_waypoints == {"mid-a", "mid-b"}


def test_complete_plan_searches_dwell_alternatives_and_reports_selected_values() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Dwell alternatives",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin", location={"type": "track_offset", "track": "TO", "offset": 0}
            ),
            ServicePathWaypoint(
                key="mid-a",
                location={"type": "track_offset", "track": "TA", "offset": 500},
                dwell_options_seconds=[30, 60],
            ),
            ServicePathWaypoint(
                key="mid-b",
                location={"type": "track_offset", "track": "TB", "offset": 500},
                optional_stop=True,
                dwell_options_seconds=[45, 90],
            ),
            ServicePathWaypoint(
                key="destination", location={"type": "track_offset", "track": "TD", "offset": 1_000}
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    client = StubOSRDClient()
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_phase_options_per_service=1,
        max_stop_pattern_options_per_service=6,
        max_candidates=6,
    )

    response = asyncio.run(plan_complete_timetable(request, client))

    assert response.total_stop_pattern_combinations == 6
    assert len(response.candidates) == 6
    selected_dwells = {
        tuple(sorted(candidate.patterns[0].stop_dwell_seconds_by_key.items()))
        for candidate in response.candidates
    }
    assert selected_dwells == {
        (("mid-a", 30),),
        (("mid-a", 30), ("mid-b", 45)),
        (("mid-a", 30), ("mid-b", 90)),
        (("mid-a", 60),),
        (("mid-a", 60), ("mid-b", 45)),
        (("mid-a", 60), ("mid-b", 90)),
    }
    scheduled_dwells = {
        (stop["at"], stop["stop_for"])
        for payload in client.candidate_payloads
        for insertion in payload["insertions"]
        if isinstance(insertion, dict)
        for stop in insertion["schedule"]
        if "stop_for" in stop
    }
    assert scheduled_dwells == {
        ("mid-a", "PT30S"),
        ("mid-a", "PT60S"),
        ("mid-b", "PT45S"),
        ("mid-b", "PT90S"),
    }


def test_complete_plan_proposes_existing_service_dwell_edits_and_flow_on() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    schedule_form = {
        "train_name": "Existing service",
        "start_time": 1_000,
        "path": [
            {"key": "origin", "location": {"type": "track_offset", "track": "TO", "offset": 0}},
            {"key": "mid", "location": {"type": "track_offset", "track": "TM", "offset": 500}},
            {
                "key": "destination",
                "location": {"type": "track_offset", "track": "TD", "offset": 1_000},
            },
        ],
        "schedule": [{"at": "mid", "stop_for": "PT30S"}],
    }
    metadata = ScheduleMetadata(
        start_times_ms={"42": 1_000},
        train_names={"42": "Existing service"},
        schedule_forms_by_id={"42": schedule_form},
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="New service",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin",
                location={"type": "track_offset", "track": "TN0", "offset": 0},
            ),
            ServicePathWaypoint(
                key="destination",
                location={"type": "track_offset", "track": "TN1", "offset": 1000},
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    client = StubOSRDClient(
        schedule_metadata=metadata,
        baseline_travel_times_ms={"42": 100_000},
        existing_travel_times_ms={"42": 115_000},
    )
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        existing_service_dwell_options=[
            ExistingServiceDwellOptions(
                train_schedule_id="42", waypoint_key="mid", dwell_seconds=[0, 90]
            )
        ],
        max_phase_options_per_service=1,
        max_candidates=3,
    )

    response = asyncio.run(plan_complete_timetable(request, client))

    assert response.total_candidate_combinations == 3
    by_dwell = {
        candidate.existing_service_dwell_edits[0].after_dwell_seconds: candidate
        for candidate in response.candidates
        if candidate.existing_service_dwell_edits
    }
    assert set(by_dwell) == {0, 90}
    assert all(candidate.validation.simulation_valid for candidate in by_dwell.values())
    assert all(candidate.new_conflict_count == 0 for candidate in by_dwell.values())
    changed = by_dwell[90]
    assert changed.changed_service_count == 1
    assert changed.existing_service_timings[0].dwell_changed
    assert changed.existing_service_timings[0].travel_time_delta_seconds == 15
    payload_edits = [
        payload["existing_service_dwell_edits"]
        for payload in client.candidate_payloads
        if payload["existing_service_dwell_edits"]
    ]
    assert {rows[0]["after_dwell_seconds"] for rows in payload_edits} == {0, 90}


def test_complete_plan_marks_simulated_journey_time_limit_violation_infeasible() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="Bounded journey",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin", location={"type": "track_offset", "track": "TO", "offset": 0}
            ),
            ServicePathWaypoint(
                key="destination", location={"type": "track_offset", "track": "TD", "offset": 1_000}
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
        max_travel_time_seconds=60,
    )
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_phase_options_per_service=1,
        max_candidates=1,
    )

    response = asyncio.run(
        plan_complete_timetable(
            request, StubOSRDClient(simulated_travel_time_ms=65_000)
        )
    )

    candidate = response.candidates[0]
    assert candidate.validation.pathfinding_valid
    assert candidate.validation.simulation_valid
    assert candidate.request_constraint_status == "violated"
    assert "65s" in candidate.request_constraint_violations[0]
    assert not candidate.feasible

    satisfied_request = CompleteTimetablePlanRequest(
        services=[service.model_copy(update={"max_travel_time_seconds": 70})],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_phase_options_per_service=1,
        max_candidates=1,
    )
    satisfied = asyncio.run(
        plan_complete_timetable(
            satisfied_request, StubOSRDClient(simulated_travel_time_ms=65_000)
        )
    ).candidates[0]
    assert satisfied.request_constraint_status == "satisfied"
    assert satisfied.feasible

    unverified = asyncio.run(
        plan_complete_timetable(
            satisfied_request, StubOSRDClient(omit_travel_times=True)
        )
    ).candidates[0]
    assert unverified.request_constraint_status == "unknown"
    assert unverified.request_constraint_unverified_departures == [
        "Bounded journey-01"
    ]
    assert not unverified.feasible


def test_complete_plan_reports_non_retimed_existing_service_flow_on_effects() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    service = ServiceRequest(
        snapshot=snapshot,
        train_name="New service",
        rolling_stock_name="test_stock",
        path=[
            ServicePathWaypoint(
                key="origin", location={"type": "track_offset", "track": "TO", "offset": 0}
            ),
            ServicePathWaypoint(
                key="destination", location={"type": "track_offset", "track": "TD", "offset": 1_000}
            ),
        ],
        operating_window_start=datetime(2026, 10, 13, 8, tzinfo=UTC),
        operating_window_end=datetime(2026, 10, 13, 9, tzinfo=UTC),
        frequency_seconds=3_600,
    )
    client = StubOSRDClient(
        schedule_metadata=ScheduleMetadata(
            start_times_ms={"101": 1_792_000_000_000, "102": 1_792_003_600_000},
            train_names={"101": "Existing unchanged", "102": "Existing affected"},
        ),
        baseline_travel_times_ms={"101": 300_000, "102": 400_000},
        existing_travel_times_ms={"101": 300_000, "102": 425_000},
    )
    request = CompleteTimetablePlanRequest(
        services=[service],
        retiming=ServiceRetimingPolicy(max_changed_services=0),
        max_phase_options_per_service=1,
        max_candidates=1,
    )

    candidate = asyncio.run(plan_complete_timetable(request, client)).candidates[0]

    assert candidate.existing_service_effects_status == "complete"
    assert candidate.source_service_count == 2
    assert candidate.compared_existing_service_count == 2
    assert candidate.affected_existing_service_count == 1
    assert candidate.unavailable_existing_service_count == 0
    effect = candidate.existing_service_timings[0]
    assert effect.train_schedule_id == "102"
    assert effect.train_name == "Existing affected"
    assert effect.travel_time_delta_seconds == 25
    assert effect.retimed is False

    partial_client = StubOSRDClient(
        schedule_metadata=ScheduleMetadata(
            start_times_ms={"103": 1_792_007_200_000},
            train_names={"103": "Baseline unavailable"},
        ),
        existing_travel_times_ms={"103": 300_000},
    )
    partial = asyncio.run(plan_complete_timetable(request, partial_client)).candidates[0]
    assert partial.existing_service_effects_status == "partial"
    assert partial.source_service_count == 1
    assert partial.compared_existing_service_count == 0
    assert partial.unavailable_existing_service_count == 1
    assert partial.affected_existing_service_count == 0
    assert partial.existing_service_timings == []


def test_retiming_only_plan_validates_existing_services_without_insertions() -> None:
    snapshot = SnapshotBinding(
        infra_id="infra-a",
        infra_version="version-1",
        timetable_id="timetable-a",
        timetable_revision="revision-1",
        osrd_version="test",
    )
    source_start = int(datetime(2026, 10, 21, 8, tzinfo=UTC).timestamp() * 1000)
    client = StubOSRDClient(
        schedule_metadata=ScheduleMetadata(
            start_times_ms={"schedule-1": source_start},
            train_names={"schedule-1": "Morning service"},
        ),
        baseline_travel_times_ms={"schedule-1": 600_000},
        existing_travel_times_ms={"schedule-1": 600_000},
    )
    request = ExistingServiceRetimingPlanRequest(
        snapshot=snapshot,
        retiming=ServiceRetimingPolicy(
            eligible_train_schedule_ids=["schedule-1"],
            max_changed_services=1,
            max_shift_seconds=120,
            max_candidates=3,
        ),
    )

    response = asyncio.run(plan_existing_service_retimings(request, client))

    assert response.generated_candidate_count == 3
    assert response.evaluated_candidate_count == 3
    assert response.feasible_candidate_count == 3
    assert response.candidates[0].edits == []
    assert all(payload["insertions"] == [] for payload in client.candidate_payloads)
    retimed = next(candidate for candidate in response.candidates if candidate.edits)
    assert retimed.edits[0].train_name == "Morning service"
    assert retimed.existing_service_timings[0].retimed is True
    assert all(candidate.patterns == [] and candidate.proposed_schedules == [] for candidate in response.candidates)
