"""Versioned HTTP payloads for operations-planning requests and results."""

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from math import isfinite
from typing import Any, Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Reject unknown fields so contract drift is visible to API clients."""

    model_config = ConfigDict(extra="forbid")


class PatronageBucket(StrictModel):
    weekday: StrictInt = Field(ge=0, le=6, description="Monday 0 through Sunday 6.")
    start_minute: StrictInt = Field(ge=0, le=1439)
    duration_minutes: StrictInt = Field(ge=1, le=1440)
    mean_boardings: float = Field(ge=0)
    mean_alightings: float = Field(ge=0)
    observation_count: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def valid_values(self) -> "PatronageBucket":
        if not isfinite(self.mean_boardings) or not isfinite(self.mean_alightings):
            raise ValueError("patronage values must be finite")
        if self.start_minute + self.duration_minutes > 1440:
            raise ValueError("patronage buckets must not cross local midnight")
        return self


class PatronageStation(StrictModel):
    operational_point_id: str = Field(min_length=1)
    source_station_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mapping_method: Literal["source_id", "curated_crosswalk"]
    buckets: list[PatronageBucket] = Field(max_length=2000)


class PatronageSource(StrictModel):
    name: str = Field(min_length=1)
    url: str | None = None
    license: str = Field(min_length=1)
    observation_start: date
    observation_end: date
    retrieved_at: datetime | None = None

    @model_validator(mode="after")
    def observation_period_is_valid(self) -> "PatronageSource":
        if self.observation_end < self.observation_start:
            raise ValueError("observation_end must not precede observation_start")
        if self.retrieved_at is not None and self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must include a timezone")
        return self


class PatronageProfile(StrictModel):
    schema_version: Literal["station_patronage_profile_v1"]
    dataset_id: str = Field(min_length=1)
    infra_id: str = Field(min_length=1)
    infra_version: str = Field(min_length=1)
    source: PatronageSource
    timezone: str = Field(min_length=1)
    stations: list[PatronageStation] = Field(max_length=5000)

    @model_validator(mode="after")
    def profile_is_mappable(self) -> "PatronageProfile":
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        station_ids = [station.operational_point_id for station in self.stations]
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("operational_point_id values must be unique within a profile")
        for station in self.stations:
            buckets_by_day: dict[int, list[tuple[int, int]]] = {}
            for bucket in station.buckets:
                buckets_by_day.setdefault(bucket.weekday, []).append(
                    (bucket.start_minute, bucket.start_minute + bucket.duration_minutes)
                )
            for day_buckets in buckets_by_day.values():
                day_buckets.sort()
                if any(
                    current[0] < previous[1]
                    for previous, current in zip(day_buckets, day_buckets[1:], strict=False)
                ):
                    raise ValueError(
                        f"overlapping time buckets for station {station.operational_point_id}"
                    )
        return self


class StationCallTiming(StrictModel):
    train_name: str = Field(min_length=1)
    operational_point_id: str | None
    arrival_time: datetime
    departure_time: datetime
    boarding_allowed: bool
    alighting_allowed: bool

    @model_validator(mode="after")
    def call_times_are_aware(self) -> "StationCallTiming":
        if self.arrival_time.tzinfo is None or self.departure_time.tzinfo is None:
            raise ValueError("station call times must include a timezone")
        if self.departure_time < self.arrival_time:
            raise ValueError("station call departure must not precede arrival")
        return self


class HourlyStationCallTiming(StrictModel):
    """Cycle-relative station call mapped to local weekday and minute-of-day."""

    train_name: str = Field(min_length=1)
    operational_point_id: str | None
    arrival_weekday: StrictInt = Field(ge=0, le=6)
    arrival_minute: StrictInt = Field(ge=0, le=1439)
    departure_weekday: StrictInt = Field(ge=0, le=6)
    departure_minute: StrictInt = Field(ge=0, le=1439)
    boarding_allowed: bool
    alighting_allowed: bool


class DemandCoverage(StrictModel):
    dataset_id: str
    covered_boardings: float
    covered_alightings: float
    total_boardings: float
    total_alightings: float
    boarding_coverage: float | None = Field(ge=0, le=1)
    alighting_coverage: float | None = Field(ge=0, le=1)
    combined_event_coverage: float | None = Field(ge=0, le=1)
    matched_bucket_count: int
    total_bucket_count: int
    unmatched_call_count: int
    unmapped_station_call_count: int
    no_matching_time_bucket_event_count: int
    active_service_day_count: int
    metric: Literal["observed_station_time_bucket_coverage_v1"] = (
        "observed_station_time_bucket_coverage_v1"
    )
    interpretation: Literal["coverage_proxy_not_passenger_benefit"] = (
        "coverage_proxy_not_passenger_benefit"
    )


class IncidentKind(StrEnum):
    DELAY = "delay"
    TRACK_CLOSURE = "track_closure"
    SPEED_RESTRICTION = "speed_restriction"
    ROLLING_STOCK_UNAVAILABLE = "rolling_stock_unavailable"


class PlanningObjective(StrEnum):
    MINIMIZE_TOTAL_SHIFT = "minimize_total_shift"
    MINIMIZE_CHANGED_SERVICES = "minimize_changed_services"
    MINIMIZE_WEIGHTED_SHIFT = "minimize_weighted_shift"


class CandidateValidationOrder(StrEnum):
    OBJECTIVE_FIRST = "objective_first"
    DIVERSE = "diverse"


class AffectedResource(StrictModel):
    resource_type: Literal["track_section", "operational_point", "route", "rolling_stock"]
    resource_id: str = Field(min_length=1)
    start_offset_m: float | None = Field(default=None, ge=0)
    end_offset_m: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> "AffectedResource":
        if self.start_offset_m is not None and self.end_offset_m is not None:
            if self.end_offset_m < self.start_offset_m:
                raise ValueError("end_offset_m must be greater than or equal to start_offset_m")
        return self


class Incident(StrictModel):
    incident_id: str = Field(min_length=1)
    kind: IncidentKind
    affected_resources: list[AffectedResource] = Field(min_length=1)
    starts_at: datetime
    ends_at: datetime
    confidence: float = Field(ge=0, le=1)
    source: str = Field(min_length=1)
    delay_seconds: int | None = Field(default=None, ge=1, le=7200)
    affected_train_schedule_ids: list[str] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def interval_is_valid_and_timezone_aware(self) -> "Incident":
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("incident timestamps must include a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be later than starts_at")
        if len(set(self.affected_train_schedule_ids)) != len(self.affected_train_schedule_ids):
            raise ValueError("affected_train_schedule_ids must be unique")
        if self.kind != IncidentKind.DELAY and (
            self.delay_seconds is not None or self.affected_train_schedule_ids
        ):
            raise ValueError("delay details may only be supplied for delay incidents")
        return self


class SnapshotBinding(StrictModel):
    infra_id: str = Field(min_length=1)
    infra_version: str = Field(min_length=1)
    timetable_id: str = Field(min_length=1)
    timetable_revision: str = Field(
        min_length=1,
        description=(
            "SHA-256 of canonical JSON containing timetable type and sorted source train "
            "schedule IDs, set IDs, and create-form payloads; formatted as sha256:<hex>."
        ),
    )
    osrd_version: str = Field(min_length=1)


class PlanningPolicy(StrictModel):
    max_candidates: int = Field(default=10, ge=1, le=50)
    max_departure_shift_seconds: int = Field(default=900, ge=0, le=7200)
    protected_train_schedule_ids: list[str] = Field(default_factory=list, max_length=500)
    objective: PlanningObjective = PlanningObjective.MINIMIZE_TOTAL_SHIFT
    schedule_priority_weights: dict[str, float] = Field(default_factory=dict, max_length=500)
    max_shift_seconds_by_schedule: dict[str, StrictInt] = Field(
        default_factory=dict, max_length=500
    )
    candidate_validation_order: CandidateValidationOrder = CandidateValidationOrder.OBJECTIVE_FIRST

    @model_validator(mode="after")
    def objective_weights_are_valid(self) -> "PlanningPolicy":
        if len(set(self.protected_train_schedule_ids)) != len(self.protected_train_schedule_ids):
            raise ValueError("protected_train_schedule_ids must be unique")
        for schedule_id, weight in self.schedule_priority_weights.items():
            if not schedule_id or not isfinite(weight) or weight <= 0 or weight > 1000:
                raise ValueError("schedule priority weights must be finite values in (0, 1000]")
        for schedule_id, maximum in self.max_shift_seconds_by_schedule.items():
            if not schedule_id or isinstance(maximum, bool) or not 0 <= maximum <= 7200:
                raise ValueError("per-schedule shift limits must be integer values in [0, 7200]")
        if self.objective == PlanningObjective.MINIMIZE_WEIGHTED_SHIFT:
            if not self.schedule_priority_weights:
                raise ValueError("minimize_weighted_shift requires explicit schedule priorities")
        elif self.schedule_priority_weights:
            raise ValueError("schedule_priority_weights require minimize_weighted_shift")
        return self


class PlanningRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    scenario_id: str = Field(min_length=1)
    snapshot: SnapshotBinding
    incident: Incident
    horizon_start: datetime
    horizon_end: datetime
    policy: PlanningPolicy = Field(default_factory=PlanningPolicy)

    @model_validator(mode="after")
    def horizon_is_valid_and_timezone_aware(self) -> "PlanningRequest":
        if self.horizon_start.tzinfo is None or self.horizon_end.tzinfo is None:
            raise ValueError("planning horizon timestamps must include a timezone")
        if self.horizon_end <= self.horizon_start:
            raise ValueError("horizon_end must be later than horizon_start")
        return self


class PlanProblem(StrictModel):
    code: str
    message: str
    request_id: UUID | None = None


class PlanCandidateEdit(StrictModel):
    train_schedule_id: str
    train_name: str | None = None
    before_start_time_ms: int
    after_start_time_ms: int
    shift_seconds: int


class PlanCandidateValidation(StrictModel):
    pathfinding_valid: bool
    simulation_valid: bool
    conflict_count: int | None
    baseline_conflict_count: int = Field(ge=0)
    new_conflict_count: int = Field(ge=0)
    failure_reasons: list[str]
    conflicting_existing_service_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Existing train schedule IDs involved in new OSRD conflicts for this candidate; "
            "they may be included in a subsequent bounded retiming search."
        ),
    )
    conflict_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint of the canonical OSRD conflicts for this candidate.",
    )
    baseline_conflict_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint of the canonical OSRD conflicts for the source timetable.",
    )


class PlanCandidateResult(StrictModel):
    candidate_id: str
    strategy: str
    edits: list[PlanCandidateEdit]
    total_shift_seconds: int
    objective_score: float = Field(ge=0)
    validation: PlanCandidateValidation


class PlanningResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    objective: PlanningObjective
    candidate_validation_order: CandidateValidationOrder = CandidateValidationOrder.OBJECTIVE_FIRST
    generated_candidate_count: int
    evaluated_candidate_count: int
    baseline: PlanCandidateResult
    candidates: list[PlanCandidateResult]
    rejected_candidates: list[PlanCandidateResult]

    @model_validator(mode="after")
    def candidate_validation_is_consistent(self) -> "PlanningResponse":
        evaluated = [*self.candidates, *self.rejected_candidates]
        if self.evaluated_candidate_count != len(evaluated):
            raise ValueError("evaluated_candidate_count must match returned candidate results")
        if self.generated_candidate_count < self.evaluated_candidate_count:
            raise ValueError("generated_candidate_count cannot be below evaluated_candidate_count")
        ids = [self.baseline.candidate_id, *(candidate.candidate_id for candidate in evaluated)]
        if len(ids) != len(set(ids)):
            raise ValueError("planning response candidate IDs must be unique")
        baseline_conflicts = self.baseline.validation.conflict_count
        if baseline_conflicts is None:
            raise ValueError("planning response requires a known baseline conflict count")
        if self.baseline.validation.baseline_conflict_count != baseline_conflicts:
            raise ValueError("baseline conflict count must match the baseline result")
        for candidate in evaluated:
            validation = candidate.validation
            if validation.conflict_count is None:
                raise ValueError("planning response requires a conflict count for every candidate")
            if (
                validation.baseline_conflict_count
                != self.baseline.validation.baseline_conflict_count
            ):
                raise ValueError(
                    "candidate baseline conflict counts must match the baseline result"
                )
            if validation.new_conflict_count > validation.conflict_count:
                raise ValueError("new conflict count cannot exceed total candidate conflicts")
        for candidate in self.candidates:
            validation = candidate.validation
            if (
                not validation.pathfinding_valid
                or not validation.simulation_valid
                or validation.new_conflict_count != 0
                or any("cleanup failed" in reason for reason in validation.failure_reasons)
            ):
                raise ValueError("accepted candidates must pass OSRD checks without new conflicts")
        for candidate in self.rejected_candidates:
            validation = candidate.validation
            if (
                validation.pathfinding_valid
                and validation.simulation_valid
                and validation.new_conflict_count == 0
                and not any("cleanup failed" in reason for reason in validation.failure_reasons)
            ):
                raise ValueError(
                    "rejected candidates must fail OSRD checks, add conflicts, or have "
                    "cleanup failures"
                )
        return self


class InsertionRankingCandidate(StrictModel):
    """Outcome-free replay context for one proposed single-service insertion."""

    candidate_id: str = Field(min_length=1, max_length=128)
    replay_episode: dict[str, Any]


class InsertionRankingRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    candidates: list[InsertionRankingCandidate] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> "InsertionRankingRequest":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        return self


class InsertionServiceImpact(StrictModel):
    baseline_schedule_id: int
    conflict_probability: float = Field(ge=0, le=1)
    conflict_risk_at_threshold: bool | None


class InsertionCandidateScore(StrictModel):
    rank: int = Field(ge=1)
    candidate_id: str
    mean_existing_service_conflict_probability: float = Field(ge=0, le=1)
    max_existing_service_conflict_probability: float = Field(ge=0, le=1)
    affected_services: list[InsertionServiceImpact]


class InsertionRankingResponse(StrictModel):
    request_id: UUID
    model_sha256: str
    offline_candidate_ranking_eligible: Literal[True]
    generated_candidate_count: int = Field(ge=1)
    ranked_candidates: list[InsertionCandidateScore]
    advisory_only: Literal[True] = True
    osrd_validation_required: Literal[True] = True


class ServicePathWaypoint(StrictModel):
    key: str = Field(min_length=1, max_length=128)
    location: dict[str, Any]
    stop_for_seconds: StrictInt | None = Field(default=None, ge=0, le=3600)
    dwell_options_seconds: list[StrictInt] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Optional dwell alternatives considered by complete timetable search. "
            "When omitted, stop_for_seconds remains fixed."
        ),
    )
    optional_stop: bool = False

    @model_validator(mode="after")
    def location_matches_osrd_schema(self) -> "ServicePathWaypoint":
        location_type = self.location.get("type")
        if location_type == "track_offset":
            offset = self.location.get("offset")
            if (
                not isinstance(self.location.get("track"), str)
                or not self.location["track"]
                or not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
            ):
                raise ValueError("track_offset locations require a track and integer offset")
        elif location_type == "operational_point_part_reference":
            if not isinstance(self.location.get("operational_point"), dict):
                raise ValueError(
                    "operational_point_part_reference locations require an operational_point"
                )
        else:
            raise ValueError("location type must be an OSRD track offset or operational point")
        if len(set(self.dwell_options_seconds)) != len(self.dwell_options_seconds):
            raise ValueError("dwell_options_seconds must contain unique values")
        if any(value <= 0 for value in self.dwell_options_seconds):
            raise ValueError("dwell_options_seconds values must be positive")
        if self.optional_stop and not (
            self.dwell_options_seconds
            or (self.stop_for_seconds is not None and self.stop_for_seconds > 0)
        ):
            raise ValueError(
                "optional stops require a positive fixed dwell or dwell alternative"
            )
        return self


class ServiceRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    snapshot: SnapshotBinding
    work_schedule_group_id: StrictInt | None = Field(
        default=None,
        ge=1,
        description="Optional OSRD work-schedule group included in calendar conflict checks.",
    )
    train_name: str = Field(
        min_length=1,
        max_length=125,
        description=(
            "Base service name; generated departure names append a three-character -NN suffix."
        ),
    )
    rolling_stock_name: str = Field(min_length=1, max_length=255)
    path: list[ServicePathWaypoint] = Field(min_length=2, max_length=30)
    operating_time_zone: str = Field(
        default="UTC",
        min_length=1,
        max_length=64,
        description="IANA time zone used to interpret the requested wall-clock window.",
    )
    operating_window_start: datetime
    operating_window_end: datetime
    operating_weekdays: list[StrictInt] = Field(
        default_factory=list,
        max_length=7,
        description=(
            "Optional local weekdays (Monday 0 through Sunday 6) on which the daily "
            "window between the start and end wall-clock times repeats. Empty preserves "
            "the single continuous UTC window."
        ),
    )
    frequency_seconds: StrictInt = Field(ge=60, le=3600)
    max_travel_time_seconds: StrictInt | None = Field(
        default=None,
        ge=60,
        le=86_400,
        description=(
            "Optional hard maximum for OSRD-simulated end-to-end time, including scheduled stops. "
            "Currently enforced for complete timetable plans."
        ),
    )

    @model_validator(mode="after")
    def service_request_is_well_formed(self) -> "ServiceRequest":
        try:
            zone = ZoneInfo(self.operating_time_zone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("operating_time_zone must be a valid IANA time zone") from exc
        if self.operating_window_start.tzinfo is None or self.operating_window_end.tzinfo is None:
            raise ValueError("operating window timestamps must include a timezone")
        if self.operating_window_start.utcoffset() != timedelta(
            0
        ) or self.operating_window_end.utcoffset() != timedelta(0):
            raise ValueError("operating window timestamps must be expressed in UTC")
        for timestamp in (self.operating_window_start, self.operating_window_end):
            local_wall_time = timestamp.astimezone(zone).replace(tzinfo=None)
            valid_offsets = []
            for fold in (0, 1):
                candidate = local_wall_time.replace(tzinfo=zone, fold=fold)
                round_trip = candidate.astimezone(UTC).astimezone(zone)
                if round_trip.replace(tzinfo=None) == local_wall_time:
                    valid_offsets.append(candidate.utcoffset())
            if not valid_offsets:
                raise ValueError(
                    "operating window timestamps must not map to a nonexistent local time"
                )
            if len(set(valid_offsets)) > 1:
                raise ValueError(
                    "operating window timestamps must not map to an ambiguous repeated local time"
                )
        if self.operating_window_end <= self.operating_window_start:
            raise ValueError("operating_window_end must be later than operating_window_start")
        if len(set(self.operating_weekdays)) != len(self.operating_weekdays):
            raise ValueError("operating_weekdays must contain unique local weekday values")
        if any(day < 0 or day > 6 for day in self.operating_weekdays):
            raise ValueError("operating_weekdays values must be Monday 0 through Sunday 6")
        windows = self.operating_windows()
        if any((end - start).total_seconds() < 1 for start, end in windows):
            raise ValueError("each operating window must be at least one second long")
        departure_count = self.departure_count
        if departure_count > 50:
            raise ValueError(
                "operating window and frequency must produce at most 50 services"
            )
        keys = [waypoint.key for waypoint in self.path]
        if len(keys) != len(set(keys)):
            raise ValueError("path waypoint keys must be unique")
        return self

    def operating_windows(self) -> list[tuple[datetime, datetime]]:
        """Return UTC operating windows, repeating wall-clock bounds on selected weekdays."""
        if not self.operating_weekdays:
            return [(self.operating_window_start, self.operating_window_end)]
        zone = ZoneInfo(self.operating_time_zone)
        local_start = self.operating_window_start.astimezone(zone)
        local_end = self.operating_window_end.astimezone(zone)
        start_time = local_start.timetz().replace(tzinfo=None)
        end_time = local_end.timetz().replace(tzinfo=None)
        if end_time <= start_time:
            raise ValueError(
                "weekday service windows must end later than they start on the same local day"
            )

        windows: list[tuple[datetime, datetime]] = []
        current = local_start.date()
        end_date = local_end.date()
        while current <= end_date:
            if current.weekday() in self.operating_weekdays:
                start = _resolve_local_window_time(current, start_time, zone)
                end = _resolve_local_window_time(current, end_time, zone)
                if end <= start:
                    raise ValueError("weekday operating window is invalid after timezone conversion")
                windows.append((start, end))
            current += timedelta(days=1)
        if not windows:
            raise ValueError("operating_weekdays select no days in the requested date range")
        return windows

    @property
    def departure_count(self) -> int:
        interval_microseconds = self.frequency_seconds * 1_000_000
        total = 0
        for start, end in self.operating_windows():
            window = end - start
            duration_microseconds = (
                (window.days * 86_400 + window.seconds) * 1_000_000 + window.microseconds
            )
            total += (duration_microseconds + interval_microseconds - 1) // interval_microseconds
        return total

    @property
    def minimum_operating_window_seconds(self) -> int:
        return min(
            int((end - start).total_seconds()) for start, end in self.operating_windows()
        )


def _resolve_local_window_time(day: date, wall_time: time, zone: ZoneInfo) -> datetime:
    local_wall_time = datetime.combine(day, wall_time)
    candidates: list[datetime] = []
    offsets = set()
    for fold in (0, 1):
        candidate = local_wall_time.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone)
        if round_trip.replace(tzinfo=None) == local_wall_time:
            candidates.append(candidate)
            offsets.add(candidate.utcoffset())
    if not candidates:
        raise ValueError("weekday operating window maps to a nonexistent local time")
    if len(offsets) > 1:
        raise ValueError("weekday operating window maps to an ambiguous repeated local time")
    return candidates[0].astimezone(UTC)


class ExistingServiceTiming(StrictModel):
    train_schedule_id: str
    train_name: str | None = None
    departure_time: datetime
    arrival_time: datetime
    travel_time_seconds: StrictInt = Field(ge=0)
    baseline_travel_time_seconds: StrictInt | None = Field(default=None, ge=0)
    travel_time_delta_seconds: int | None = Field(
        default=None,
        description=(
            "Candidate minus source-snapshot simulated travel time; positive means slower."
        ),
    )
    arrival_time_delta_seconds: int | None = Field(
        default=None,
        description=(
            "Candidate minus source-snapshot simulated arrival time; positive means later."
        ),
    )
    retimed: bool = Field(
        default=False,
        description="Whether this service also has an explicit departure-time edit.",
    )
    dwell_changed: bool = Field(
        default=False,
        description="Whether this service has an explicit station dwell-time edit.",
    )


class HourlyServicePlanCandidate(StrictModel):
    candidate_id: str
    rank: int = Field(ge=1)
    route_alternative_index: int = Field(default=0, ge=0)
    route_track_sections: list[str] = Field(default_factory=list)
    departure_offset_seconds: StrictInt = Field(ge=0)
    recurring_departure_count: StrictInt = Field(ge=1)
    proposed_schedule: dict[str, Any]
    edits: list[PlanCandidateEdit] = Field(default_factory=list)
    changed_service_count: int = Field(ge=0)
    total_shift_seconds: int = Field(ge=0)
    simulated_runtime_change_seconds: int | None = None
    retiming_priority_score: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Weighted sum of absolute retiming seconds and planner-supplied service priorities."
        ),
    )
    simulated_travel_time_seconds: StrictInt | None = Field(default=None, ge=0)
    existing_service_timings: list[ExistingServiceTiming] = Field(default_factory=list)
    existing_service_effects_status: Literal["not_applicable", "complete", "partial"] = (
        "not_applicable"
    )
    unavailable_retimed_service_count: StrictInt = Field(default=0, ge=0)
    station_call_timings: list[HourlyStationCallTiming] = Field(default_factory=list)
    patronage_coverage_status: Literal[
        "not_configured", "available", "simulation_times_unavailable", "mapping_unavailable"
    ] = "not_configured"
    patronage_coverage: DemandCoverage | None = None
    baseline_conflict_count: int = Field(ge=0)
    new_conflict_count: int = Field(ge=0)
    validation: PlanCandidateValidation
    feasible: bool
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def feasibility_matches_validation(self) -> "HourlyServicePlanCandidate":
        if self.validation.conflict_count is None:
            raise ValueError("hourly candidate conflict count must be known")
        if self.validation.baseline_conflict_count != self.baseline_conflict_count:
            raise ValueError("hourly candidate baseline conflict count is inconsistent")
        if self.validation.new_conflict_count != self.new_conflict_count:
            raise ValueError("hourly candidate new conflict count is inconsistent")
        if self.total_shift_seconds != sum(abs(edit.shift_seconds) for edit in self.edits):
            raise ValueError("hourly candidate total shift must match its explicit edits")
        if self.changed_service_count != len(self.edits):
            raise ValueError("hourly candidate changed-service count must match its edits")
        if self.changed_service_count == 0:
            if (
                self.existing_service_effects_status != "not_applicable"
                or self.existing_service_timings
                or self.unavailable_retimed_service_count
            ):
                raise ValueError("hourly candidates without retimings cannot report retiming effects")
        else:
            if (
                len(self.existing_service_timings) + self.unavailable_retimed_service_count
                != self.changed_service_count
            ):
                raise ValueError("hourly retiming effects must account for each edited service")
            if (self.existing_service_effects_status == "complete") != (
                self.unavailable_retimed_service_count == 0
            ):
                raise ValueError("hourly effect status must match comparison coverage")
            if self.existing_service_effects_status == "not_applicable":
                raise ValueError("hourly retimings require an effect coverage status")
        expected = (
            self.validation.pathfinding_valid
            and self.validation.simulation_valid
            and self.new_conflict_count == 0
            and not any(
                "cleanup failed" in item.lower() for item in self.validation.failure_reasons
            )
        )
        if self.feasible != expected:
            raise ValueError("hourly candidate feasibility must match OSRD validation")
        if (self.patronage_coverage_status == "available") != (self.patronage_coverage is not None):
            raise ValueError("hourly patronage coverage status must match its score")
        return self


class HourlyServicePlanResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    generated_candidate_count: int = Field(ge=1)
    evaluated_candidate_count: int = Field(ge=1)
    feasible_candidate_count: int = Field(ge=0)
    selected_candidate_id: str | None
    candidates: list[HourlyServicePlanCandidate]
    advisory_only: Literal[True] = True
    osrd_validation_required: Literal[True] = True

    @model_validator(mode="after")
    def candidate_set_is_consistent(self) -> "HourlyServicePlanResponse":
        ids = [item.candidate_id for item in self.candidates]
        if (
            len(ids) != len(set(ids))
            or self.generated_candidate_count < len(ids)
            or self.evaluated_candidate_count != len(ids)
            or self.feasible_candidate_count != sum(item.feasible for item in self.candidates)
            or [item.rank for item in self.candidates] != list(range(1, len(ids) + 1))
        ):
            raise ValueError("hourly candidate counts, IDs, and ranks must be consistent")
        selected = next((item.candidate_id for item in self.candidates if item.feasible), None)
        if selected != self.selected_candidate_id:
            raise ValueError("selected hourly candidate must be the first feasible option")
        return self


class ServiceRequestValidationResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    requested_departure_count: int = Field(ge=1, le=50)
    departures: list[datetime]
    baseline_conflict_count: int = Field(ge=0)
    new_conflict_count: int = Field(ge=0)
    validation: PlanCandidateValidation
    feasible: bool
    advisory_only: Literal[True] = True


class ServiceRetimingPolicy(StrictModel):
    eligible_train_schedule_ids: list[str] = Field(default_factory=list, max_length=12)
    protected_train_schedule_ids: list[str] = Field(default_factory=list, max_length=500)
    max_changed_services: StrictInt = Field(
        default=12,
        ge=0,
        le=12,
        description=(
            "Hard upper bound on how many eligible existing services one candidate may retime."
        ),
    )
    max_shift_seconds: StrictInt = Field(default=300, ge=0, le=7200)
    max_shift_seconds_by_schedule: dict[str, StrictInt] = Field(
        default_factory=dict,
        max_length=12,
        description="Optional hard shift caps for individual eligible schedules.",
    )
    max_advance_seconds_by_schedule: dict[str, StrictInt] = Field(
        default_factory=dict,
        max_length=12,
        description="Optional hard advance caps for individual eligible schedules.",
    )
    step_seconds: StrictInt = Field(default=60, ge=1, le=600)
    max_candidates: StrictInt = Field(default=20, ge=1, le=50)
    objective: Literal[
        "minimize_total_shift",
        "minimize_changed_services",
        "minimize_simulated_runtime_change",
        "minimize_weighted_shift",
    ] = "minimize_total_shift"
    schedule_priority_weights: dict[str, StrictFloat | StrictInt] = Field(
        default_factory=dict, max_length=12
    )

    @model_validator(mode="after")
    def ids_are_valid(self) -> "ServiceRetimingPolicy":
        if any(not value for value in self.eligible_train_schedule_ids):
            raise ValueError("eligible_train_schedule_ids must not contain empty IDs")
        if any(not value for value in self.protected_train_schedule_ids):
            raise ValueError("protected_train_schedule_ids must not contain empty IDs")
        if len(set(self.eligible_train_schedule_ids)) != len(self.eligible_train_schedule_ids):
            raise ValueError("eligible_train_schedule_ids must be unique")
        if len(set(self.protected_train_schedule_ids)) != len(self.protected_train_schedule_ids):
            raise ValueError("protected_train_schedule_ids must be unique")
        if set(self.eligible_train_schedule_ids) & set(self.protected_train_schedule_ids):
            raise ValueError("a train cannot be both eligible and protected")
        for schedule_id, maximum in self.max_shift_seconds_by_schedule.items():
            if not schedule_id:
                raise ValueError("per-schedule shift caps require non-empty schedule IDs")
            if schedule_id not in self.eligible_train_schedule_ids:
                raise ValueError("per-schedule shift caps may only target eligible services")
            if not 0 <= maximum <= self.max_shift_seconds:
                raise ValueError(
                    "per-schedule shift caps must be between zero and the global shift cap"
                )
        for schedule_id, maximum in self.max_advance_seconds_by_schedule.items():
            if not schedule_id:
                raise ValueError("per-schedule advance caps require non-empty schedule IDs")
            if schedule_id not in self.eligible_train_schedule_ids:
                raise ValueError("per-schedule advance caps may only target eligible services")
            if not 0 <= maximum <= self.max_shift_seconds:
                raise ValueError(
                    "per-schedule advance caps must be between zero and the global shift cap"
                )
        if self.objective == "minimize_weighted_shift":
            if not self.eligible_train_schedule_ids:
                raise ValueError("minimize_weighted_shift requires eligible services")
            if set(self.schedule_priority_weights) != set(self.eligible_train_schedule_ids):
                raise ValueError(
                    "minimize_weighted_shift requires a priority for every eligible service"
                )
            if any(
                not isfinite(weight) or weight <= 0 or weight > 1000
                for weight in self.schedule_priority_weights.values()
            ):
                raise ValueError("schedule priority weights must be finite values in (0, 1000]")
        elif self.schedule_priority_weights:
            raise ValueError("schedule_priority_weights require minimize_weighted_shift")
        return self


class ExistingServiceDwellOptions(StrictModel):
    """Candidate dwell values for one existing service path waypoint; zero removes the stop."""

    train_schedule_id: str = Field(min_length=1)
    waypoint_key: str = Field(min_length=1, max_length=128)
    dwell_seconds: list[StrictInt] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def alternatives_are_unique_and_positive(self) -> "ExistingServiceDwellOptions":
        if len(set(self.dwell_seconds)) != len(self.dwell_seconds):
            raise ValueError("existing-service dwell alternatives must be unique")
        if any(value < 0 or value > 3600 for value in self.dwell_seconds):
            raise ValueError("existing-service dwell alternatives must be in [0, 3600]")
        return self


class ExistingServiceDwellEdit(StrictModel):
    train_schedule_id: str
    train_name: str | None = None
    waypoint_key: str
    before_dwell_seconds: StrictInt = Field(ge=0, le=3600)
    after_dwell_seconds: StrictInt = Field(ge=0, le=3600)


class HourlyServiceRequest(StrictModel):
    """Recurring service request for an OSRD HOURLY timetable."""

    request_id: UUID = Field(default_factory=uuid4)
    snapshot: SnapshotBinding
    train_name: str = Field(min_length=1, max_length=128)
    rolling_stock_name: str = Field(min_length=1, max_length=255)
    path: list[ServicePathWaypoint] = Field(min_length=2, max_length=30)
    frequency_seconds: StrictInt = Field(ge=60, le=3600)
    time_window_seconds: StrictInt = Field(ge=60, le=86_400)
    phase_step_seconds: StrictInt = Field(default=60, ge=1, le=3600)
    max_phase_options: StrictInt = Field(default=8, ge=1, le=32)
    max_route_alternatives: StrictInt = Field(default=1, ge=1, le=5)
    retiming: ServiceRetimingPolicy = Field(default_factory=ServiceRetimingPolicy)
    candidate_order: Literal["operational", "station_patronage_coverage"] = "operational"
    patronage_profile: PatronageProfile | None = None
    operating_weekdays: list[StrictInt] = Field(default_factory=list, max_length=7)
    cycle_start_minute: StrictInt = Field(
        default=0,
        ge=0,
        le=1439,
        description="Local minute of day when the recurring cycle begins in the profile timezone.",
    )

    @model_validator(mode="after")
    def hourly_pattern_is_well_formed(self) -> "HourlyServiceRequest":
        if self.frequency_seconds > self.time_window_seconds:
            raise ValueError("frequency interval must not exceed the hourly time window")
        if len(set(self.operating_weekdays)) != len(self.operating_weekdays):
            raise ValueError("operating_weekdays must contain unique local weekday values")
        if any(day < 0 or day > 6 for day in self.operating_weekdays):
            raise ValueError("operating_weekdays values must be Monday 0 through Sunday 6")
        profile = self.patronage_profile
        if profile is not None:
            if profile.infra_id != self.snapshot.infra_id or profile.infra_version != self.snapshot.infra_version:
                raise ValueError("patronage profile must match the hourly service snapshot infrastructure")
            if not self.operating_weekdays:
                raise ValueError("hourly patronage comparison requires at least one operating weekday")
        if self.candidate_order == "station_patronage_coverage":
            if profile is None:
                raise ValueError("station_patronage_coverage candidate order requires a patronage profile")
            if not self.operating_weekdays:
                raise ValueError("station_patronage_coverage candidate order requires operating_weekdays")
        keys = [waypoint.key for waypoint in self.path]
        if len(keys) != len(set(keys)):
            raise ValueError("path waypoint keys must be unique")
        if any(waypoint.optional_stop for waypoint in self.path):
            raise ValueError("optional stops are only supported for complete timetable plans")
        if any(waypoint.dwell_options_seconds for waypoint in self.path):
            raise ValueError("dwell alternatives are only supported for complete timetable plans")
        return self


class ServicePlanRequest(StrictModel):
    service: ServiceRequest
    retiming: ServiceRetimingPolicy = Field(default_factory=ServiceRetimingPolicy)
    candidate_order: Literal["operational", "station_patronage_coverage"] = "operational"
    patronage_profile: PatronageProfile | None = Field(
        default=None,
        description=(
            "Optional aggregated station demand profile for calendar service-plan comparisons. "
            "It must be bound to the same OSRD infrastructure ID and version as the snapshot."
        ),
    )
    max_route_alternatives: StrictInt = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum topological route alternatives per requested path.",
    )
    phase_step_seconds: StrictInt = Field(
        default=60,
        ge=1,
        le=3600,
        description="Granularity for candidate first-departure phases in seconds.",
    )
    max_phase_options: StrictInt = Field(
        default=6,
        ge=1,
        le=16,
        description="Maximum phase candidates sampled across one frequency cycle.",
    )

    @model_validator(mode="after")
    def patronage_profile_matches_snapshot(self) -> "ServicePlanRequest":
        if any(waypoint.optional_stop for waypoint in self.service.path):
            raise ValueError("optional stops are only supported for complete timetable plans")
        if any(waypoint.dwell_options_seconds for waypoint in self.service.path):
            raise ValueError("dwell alternatives are only supported for complete timetable plans")
        if self.service.max_travel_time_seconds is not None:
            raise ValueError(
                "max_travel_time_seconds is only supported for complete timetable plans"
            )
        profile = self.patronage_profile
        if self.candidate_order == "station_patronage_coverage" and profile is None:
            raise ValueError(
                "station_patronage_coverage candidate order requires a patronage profile"
            )
        if profile is not None and (
            profile.infra_id != self.service.snapshot.infra_id
            or profile.infra_version != self.service.snapshot.infra_version
        ):
            raise ValueError("patronage profile must match the service snapshot infrastructure")
        return self


class CompleteTimetablePlanRequest(StrictModel):
    """A bounded, jointly validated set of requested calendar service patterns."""

    request_id: UUID = Field(default_factory=uuid4)
    services: list[ServiceRequest] = Field(min_length=1, max_length=10)
    retiming: ServiceRetimingPolicy = Field(default_factory=ServiceRetimingPolicy)
    existing_service_dwell_options: list[ExistingServiceDwellOptions] = Field(
        default_factory=list, max_length=12
    )
    max_route_alternatives_per_service: StrictInt = Field(default=1, ge=1, le=5)
    max_stop_pattern_options_per_service: StrictInt = Field(default=8, ge=1, le=16)
    candidate_order: Literal["operational", "station_patronage_coverage"] = "operational"
    patronage_profile: PatronageProfile | None = None
    phase_step_seconds: StrictInt = Field(default=60, ge=1, le=3600)
    max_phase_options_per_service: StrictInt = Field(default=4, ge=1, le=8)
    max_candidates: StrictInt = Field(default=12, ge=1, le=32)

    @model_validator(mode="after")
    def complete_plan_is_consistent(self) -> "CompleteTimetablePlanRequest":
        profile = self.patronage_profile
        if self.candidate_order == "station_patronage_coverage" and profile is None:
            raise ValueError(
                "station_patronage_coverage candidate order requires a patronage profile"
            )
        if profile is not None and (
            profile.infra_id != self.services[0].snapshot.infra_id
            or profile.infra_version != self.services[0].snapshot.infra_version
        ):
            raise ValueError("patronage profile must match the service snapshot infrastructure")
        snapshots = {service.snapshot.model_dump_json() for service in self.services}
        if len(snapshots) != 1:
            raise ValueError("all requested services must use the same OSRD snapshot")
        names = [service.train_name for service in self.services]
        if len(names) != len(set(names)):
            raise ValueError("each requested service must have a unique base train name")
        groups = {service.work_schedule_group_id for service in self.services}
        if len(groups) > 1:
            raise ValueError(
                "all service patterns must use the same work-schedule group selection"
            )
        dwell_targets = [
            (item.train_schedule_id, item.waypoint_key)
            for item in self.existing_service_dwell_options
        ]
        if len(dwell_targets) != len(set(dwell_targets)):
            raise ValueError("existing-service dwell options must target unique calls")
        if sum(service.departure_count for service in self.services) > 50:
            raise ValueError("a complete plan may contain at most 50 requested departures")
        if any(
            service.path[0].optional_stop
            or service.path[-1].optional_stop
            or service.path[0].dwell_options_seconds
            or service.path[-1].dwell_options_seconds
            for service in self.services
        ):
            raise ValueError(
                "origin and destination waypoints cannot be optional or dwell-choice stops"
            )
        return self


class ExistingServiceRetimingPlanRequest(StrictModel):
    """Retiming-only plan against an existing calendar timetable snapshot."""

    request_id: UUID = Field(default_factory=uuid4)
    snapshot: SnapshotBinding
    retiming: ServiceRetimingPolicy
    work_schedule_group_id: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def retiming_targets_are_explicit(self) -> "ExistingServiceRetimingPlanRequest":
        if not self.retiming.eligible_train_schedule_ids:
            raise ValueError("Select at least one existing service that may be retimed")
        return self


class PlanIntentServiceDraft(StrictModel):
    """Unresolved service terms extracted from a planner's natural-language request."""

    train_name: str | None
    origin: str | None
    destination: str | None
    intermediate_stops: list[str] = Field(max_length=28)
    rolling_stock: str | None
    operating_start_date: str | None = Field(
        default=None,
        description="Start calendar date as YYYY-MM-DD when explicitly stated; otherwise null.",
    )
    operating_end_date: str | None = Field(
        default=None,
        description="End calendar date as YYYY-MM-DD when explicitly stated; otherwise null.",
    )
    operating_start: str | None = Field(
        default=None,
        description="Start wall-clock time as HH:MM, without a timezone suffix.",
    )
    operating_end: str | None = Field(
        default=None,
        description="End wall-clock time as HH:MM, without a timezone suffix.",
    )
    time_zone: str | None = Field(
        description="Explicit IANA timezone from the request; null if unspecified."
    )
    weekdays: list[StrictInt] = Field(
        max_length=7,
        description=(
            "Monday 0 through Sunday 6. Interpret 'weekdays' as Monday-Friday [0,1,2,3,4]."
        ),
    )
    frequency_minutes: StrictInt | None = Field(default=None, ge=1, le=60)
    max_journey_minutes: StrictInt | None = Field(default=None, ge=1, le=1440)
    patronage_preference: Literal["unspecified", "station_time_bucket_coverage"]

    @field_validator("operating_start_date", "operating_end_date")
    @classmethod
    def date_is_real(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @field_validator("operating_start", "operating_end")
    @classmethod
    def clock_time_is_real(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = time.fromisoformat(value)
            if parsed.tzinfo is not None:
                raise ValueError("operating wall-clock times must not include a timezone")
        return value

    @model_validator(mode="after")
    def weekdays_are_unique(self) -> "PlanIntentServiceDraft":
        if any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekdays must use Monday 0 through Sunday 6")
        if len(set(self.weekdays)) != len(self.weekdays):
            raise ValueError("weekdays must be unique")
        if (
            self.operating_start_date is not None
            and self.operating_end_date is not None
            and self.operating_end_date < self.operating_start_date
        ):
            raise ValueError("operating_end_date must not precede operating_start_date")
        return self


class PlanIntentExistingServiceChangeDraft(StrictModel):
    """A named, unresolved existing service the planner may include in retiming search."""

    train_name: str = Field(min_length=1, max_length=125)
    direction: Literal["earlier", "later", "either"]
    max_shift_minutes: StrictInt | None = Field(
        default=None,
        ge=1,
        le=120,
        description=(
            "Explicit upper bound from the request; do not convert a requested exact shift "
            "into an upper bound."
        ),
    )


class PlanIntentDraft(StrictModel):
    status: Literal["ready_for_review", "needs_clarification"]
    summary: str = Field(min_length=1, max_length=1000)
    services: list[PlanIntentServiceDraft] = Field(max_length=10)
    existing_service_changes: list[PlanIntentExistingServiceChangeDraft] = Field(
        default_factory=list, max_length=12
    )
    questions: list[str] = Field(max_length=20)
    assumptions: list[str] = Field(max_length=20)
    unresolved_terms: list[str] = Field(max_length=30)

    @model_validator(mode="after")
    def status_matches_completeness(self) -> "PlanIntentDraft":
        if self.status == "ready_for_review" and (
            (not self.services and not self.existing_service_changes) or self.questions
        ):
            raise ValueError(
                "ready_for_review requires service requirements or existing-service changes "
                "and no outstanding questions"
            )
        if self.status == "needs_clarification" and not self.questions:
            raise ValueError("needs_clarification requires at least one question")
        names = [change.train_name.casefold() for change in self.existing_service_changes]
        if len(names) != len(set(names)):
            raise ValueError("existing-service change targets must be unique")
        return self


class PlanIntentRequest(StrictModel):
    request_text: str = Field(min_length=8, max_length=4000)
    response_language: Literal["de", "en", "fr", "pt"] = "en"
    external_processing_confirmed: bool = False


class PlanIntentResponse(StrictModel):
    draft: PlanIntentDraft
    advisory_only: Literal[True] = True
    requires_human_review: Literal[True] = True


class CompleteTimetablePatternResult(StrictModel):
    train_name: str
    departure_phase_seconds: StrictInt = Field(ge=0)
    departure_count: StrictInt = Field(ge=1)
    departures: list[datetime]
    route_alternative_index: StrictInt = Field(default=0, ge=0)
    route_track_sections: list[str] = Field(default_factory=list)
    selected_optional_stop_keys: list[str] = Field(default_factory=list)
    omitted_optional_stop_keys: list[str] = Field(default_factory=list)
    stop_dwell_seconds_by_key: dict[str, StrictInt] = Field(default_factory=dict)


class CompleteTimetableCandidate(StrictModel):
    candidate_id: str
    rank: StrictInt = Field(ge=1)
    patterns: list[CompleteTimetablePatternResult]
    proposed_schedules: list[dict[str, Any]]
    edits: list[PlanCandidateEdit] = Field(default_factory=list)
    existing_service_dwell_edits: list[ExistingServiceDwellEdit] = Field(default_factory=list)
    changed_service_count: StrictInt = Field(ge=0)
    total_shift_seconds: StrictInt = Field(ge=0)
    retiming_priority_score: float | None = Field(default=None, ge=0)
    simulated_runtime_change_seconds: int | None = None
    departure_timings: list["ServiceDepartureTiming"] = Field(default_factory=list)
    station_call_timings: list[StationCallTiming] = Field(default_factory=list)
    patronage_coverage_status: Literal[
        "not_configured", "available", "simulation_times_unavailable", "mapping_unavailable"
    ] = "not_configured"
    patronage_coverage: DemandCoverage | None = None
    existing_service_timings: list["ExistingServiceTiming"] = Field(default_factory=list)
    existing_service_effects_status: Literal["complete", "partial"] = "complete"
    source_service_count: StrictInt = Field(default=0, ge=0)
    compared_existing_service_count: StrictInt = Field(default=0, ge=0)
    affected_existing_service_count: StrictInt = Field(default=0, ge=0)
    unavailable_existing_service_count: StrictInt = Field(default=0, ge=0)
    baseline_conflict_count: StrictInt = Field(ge=0)
    new_conflict_count: StrictInt = Field(ge=0)
    request_constraint_status: Literal[
        "not_configured", "satisfied", "violated", "unknown"
    ] = "not_configured"
    request_constraint_violations: list[str] = Field(default_factory=list)
    request_constraint_unverified_departures: list[str] = Field(default_factory=list)
    validation: PlanCandidateValidation
    feasible: bool

    @model_validator(mode="after")
    def candidate_matches_validation(self) -> "CompleteTimetableCandidate":
        changed_service_ids = {edit.train_schedule_id for edit in self.edits} | {
            edit.train_schedule_id for edit in self.existing_service_dwell_edits
        }
        if self.changed_service_count != len(changed_service_ids):
            raise ValueError("changed_service_count must match the distinct edited services")
        if self.total_shift_seconds != sum(abs(edit.shift_seconds) for edit in self.edits):
            raise ValueError("total_shift_seconds must match the explicit timetable edits")
        if self.validation.conflict_count is None:
            raise ValueError("candidate conflict count must be known")
        if self.validation.baseline_conflict_count != self.baseline_conflict_count:
            raise ValueError("candidate baseline conflict count is inconsistent")
        if self.validation.new_conflict_count != self.new_conflict_count:
            raise ValueError("candidate new conflict count is inconsistent")
        if (
            self.compared_existing_service_count + self.unavailable_existing_service_count
            != self.source_service_count
        ):
            raise ValueError("existing-service effect counts must cover the source timetable")
        if self.affected_existing_service_count != len(self.existing_service_timings):
            raise ValueError("affected existing-service count must match its timing details")
        if (self.existing_service_effects_status == "complete") != (
            self.unavailable_existing_service_count == 0
        ):
            raise ValueError("existing-service effect status must match comparison coverage")
        if self.request_constraint_status == "violated" and not self.request_constraint_violations:
            raise ValueError("violated request constraints need an explanation")
        if self.request_constraint_status == "unknown" and not (
            self.request_constraint_unverified_departures
        ):
            raise ValueError("unknown request constraints need unverified departures")
        if self.request_constraint_status in {"satisfied", "not_configured"} and (
            self.request_constraint_violations or self.request_constraint_unverified_departures
        ):
            raise ValueError("satisfied or unconfigured constraints cannot have violations")
        expected = (
            self.validation.pathfinding_valid
            and self.validation.simulation_valid
            and self.new_conflict_count == 0
            and self.request_constraint_status in {"not_configured", "satisfied"}
            and not any(
                "cleanup failed" in reason.lower()
                for reason in self.validation.failure_reasons
            )
        )
        if self.feasible is not expected:
            raise ValueError("candidate feasibility must match OSRD validation")
        return self


class CompleteTimetablePlanResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    candidates: list[CompleteTimetableCandidate]
    selected_candidate_id: str | None
    validation_budget: StrictInt = Field(ge=1)
    total_phase_combinations: StrictInt = Field(ge=1)
    total_route_combinations: StrictInt = Field(ge=1)
    total_route_phase_combinations: StrictInt = Field(ge=1)
    total_stop_pattern_combinations: StrictInt = Field(ge=1)
    total_route_phase_stop_combinations: StrictInt = Field(ge=1)
    retiming_candidate_count: StrictInt = Field(ge=1)
    total_candidate_combinations: StrictInt = Field(ge=1)
    search_complete: bool


class ExistingServiceRetimingPlanResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    generated_candidate_count: StrictInt = Field(ge=1)
    evaluated_candidate_count: StrictInt = Field(ge=1)
    feasible_candidate_count: StrictInt = Field(ge=0)
    selected_candidate_id: str | None
    candidates: list[CompleteTimetableCandidate]

    @model_validator(mode="after")
    def candidate_set_is_consistent(self) -> "ExistingServiceRetimingPlanResponse":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if (
            len(ids) != len(set(ids))
            or self.generated_candidate_count < len(ids)
            or self.evaluated_candidate_count != len(ids)
            or self.feasible_candidate_count != sum(candidate.feasible for candidate in self.candidates)
            or [candidate.rank for candidate in self.candidates] != list(range(1, len(ids) + 1))
        ):
            raise ValueError("retiming candidate counts, IDs, and ranks must be consistent")
        selected = next((candidate.candidate_id for candidate in self.candidates if candidate.feasible), None)
        if selected != self.selected_candidate_id:
            raise ValueError("selected candidate must be the first feasible retiming option")
        return self


class ServiceDepartureTiming(StrictModel):
    train_name: str
    departure_time: datetime
    arrival_time: datetime
    travel_time_seconds: StrictInt = Field(ge=0)


class ServicePlanCandidate(StrictModel):
    candidate_id: str
    rank: int = Field(ge=1)
    explanation: str = Field(
        min_length=1,
        description=(
            "Planner-facing summary of the requested service, timetable edits, objective "
            "trade-off, and OSRD validation outcome."
        ),
    )
    route_alternative_index: int = Field(default=0, ge=0)
    route_track_sections: list[str] = Field(default_factory=list)
    proposed_schedules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Complete OSRD train-schedule forms for the requested service departures.",
    )
    departure_phase_seconds: int = Field(ge=0)
    edits: list[PlanCandidateEdit]
    total_shift_seconds: int = Field(ge=0)
    changed_service_count: int = Field(ge=0)
    simulated_runtime_change_seconds: int | None = Field(
        default=None,
        description=(
            "Net candidate-minus-source OSRD running-time change across retimed services; "
            "null when a required simulation is unavailable."
        ),
    )
    retiming_priority_score: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Weighted sum of absolute retiming seconds and planner-supplied service priorities."
        ),
    )
    inserted_departures: list[datetime]
    departure_timings: list[ServiceDepartureTiming] = Field(
        default_factory=list,
        description=(
            "OSRD-simulated departure, destination arrival, and travel duration per insertion."
        ),
    )
    existing_service_timings: list[ExistingServiceTiming] = Field(
        default_factory=list,
        description=(
            "OSRD-simulated departure, destination arrival, and travel duration for each "
            "existing service retimed by this candidate."
        ),
    )
    station_call_timings: list[StationCallTiming] = Field(default_factory=list)
    patronage_coverage_status: Literal[
        "not_configured",
        "available",
        "mapping_unavailable",
        "simulation_times_unavailable",
    ] = "not_configured"
    patronage_coverage: DemandCoverage | None = None
    baseline_conflict_count: int = Field(ge=0)
    new_conflict_count: int = Field(ge=0)
    validation: PlanCandidateValidation
    feasible: bool
    research_model_rank: StrictInt | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional advisory rank from an offline-eligible single-service insertion model; "
            "it does not change OSRD validation or the deterministic option order."
        ),
    )
    research_model_mean_conflict_risk: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Optional model score for conflict involvement among existing services; "
            "research-only and not a calibrated operational probability."
        ),
    )
    research_model_max_conflict_risk: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Maximum per-existing-service model conflict score; research-only and not a "
            "calibrated operational probability."
        ),
    )
    research_model_highest_risk_schedule_id: str | None = Field(default=None, min_length=1)
    research_model_highest_risk_train_name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def research_model_scores_are_consistent(self) -> "ServicePlanCandidate":
        conflict_count = self.validation.conflict_count
        if conflict_count is None or self.new_conflict_count > conflict_count:
            raise ValueError("candidate conflict counts must be complete and consistent")
        osrd_feasible = (
            self.validation.pathfinding_valid
            and self.validation.simulation_valid
            and self.new_conflict_count == 0
            and not any(
                "cleanup failed" in reason.lower()
                for reason in self.validation.failure_reasons
            )
        )
        if self.feasible is not osrd_feasible:
            raise ValueError("candidate feasibility must match OSRD validation")
        if (self.patronage_coverage_status == "available") != (
            self.patronage_coverage is not None
        ):
            raise ValueError(
                "available station-patronage coverage must include its score, and other "
                "statuses must not"
            )
        if (self.research_model_rank is None) != (
            self.research_model_mean_conflict_risk is None
        ):
            raise ValueError("model rank and mean risk score must appear together")
        if (self.research_model_max_conflict_risk is None) != (
            self.research_model_highest_risk_schedule_id is None
        ):
            raise ValueError("maximum model risk and highest-risk schedule ID must appear together")
        if self.research_model_max_conflict_risk is not None:
            if self.research_model_rank is None:
                raise ValueError("per-service model risk details require a model rank")
            if (
                self.research_model_mean_conflict_risk is not None
                and self.research_model_max_conflict_risk
                < self.research_model_mean_conflict_risk
            ):
                raise ValueError("maximum model risk cannot be lower than its mean risk")
        if (
            self.research_model_highest_risk_train_name is not None
            and self.research_model_highest_risk_schedule_id is None
        ):
            raise ValueError("highest-risk train name requires its schedule ID")
        return self


class ServicePlanResponse(StrictModel):
    request_id: UUID
    snapshot: SnapshotBinding
    generated_candidate_count: int = Field(ge=1)
    evaluated_candidate_count: int = Field(ge=1)
    feasible_candidate_count: int = Field(ge=0)
    selected_candidate_id: str | None
    candidate_order: Literal["operational", "station_patronage_coverage"] = "operational"
    research_model_status: Literal["not_configured", "not_applicable", "used", "unavailable"] = (
        "not_configured"
    )
    research_model_suggested_candidate_id: str | None = None
    research_model_sha256: str | None = None
    research_model_note: str | None = None
    candidates: list[ServicePlanCandidate]
    advisory_only: Literal[True] = True
    osrd_validation_required: Literal[True] = True

    @model_validator(mode="after")
    def research_model_response_is_consistent(self) -> "ServicePlanResponse":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if (
            len(candidate_ids) != len(set(candidate_ids))
            or self.evaluated_candidate_count != len(self.candidates)
            or self.generated_candidate_count < self.evaluated_candidate_count
            or self.feasible_candidate_count
            != sum(candidate.feasible for candidate in self.candidates)
            or [candidate.rank for candidate in self.candidates]
            != list(range(1, len(self.candidates) + 1))
        ):
            raise ValueError("service-plan candidate counts, IDs, and ranks must be consistent")
        infeasible_seen = False
        for candidate in self.candidates:
            if not candidate.feasible:
                infeasible_seen = True
            elif infeasible_seen:
                raise ValueError("feasible candidates must rank ahead of infeasible candidates")
        selected = next(
            (candidate.candidate_id for candidate in self.candidates if candidate.feasible),
            None,
        )
        if self.selected_candidate_id != selected:
            raise ValueError("selected candidate must be the first OSRD-feasible ranked option")
        scored = [
            candidate
            for candidate in self.candidates
            if candidate.research_model_rank is not None
        ]
        if self.research_model_status == "used":
            digest = self.research_model_sha256
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or len(scored) < 2
                or {candidate.candidate_id for candidate in scored}
                != {candidate.candidate_id for candidate in self.candidates if candidate.feasible}
                or sorted(int(candidate.research_model_rank) for candidate in scored)
                != list(range(1, len(scored) + 1))
            ):
                raise ValueError(
                    "used research model requires a digest and complete feasible ranks"
                )
            suggested = next(
                (
                    candidate
                    for candidate in scored
                    if candidate.candidate_id == self.research_model_suggested_candidate_id
                ),
                None,
            )
            if suggested is None or suggested.research_model_rank != 1 or not suggested.feasible:
                raise ValueError("research model suggestion must be its rank-one feasible option")
        elif (
            self.research_model_sha256 is not None
            or self.research_model_suggested_candidate_id is not None
            or scored
        ):
            raise ValueError("research model scores require a used model status")
        return self


class PlanAuditRecord(StrictModel):
    request_id: UUID
    status: Literal["completed"]
    created_at: datetime
    input_sha256: str
    result_sha256: str
    request: PlanningRequest
    result: PlanningResponse


class ServicePlanReviewSubmission(StrictModel):
    record_schema: Literal["osrd-service-plan-review-record/v1"] = Field(alias="schema")
    review_id: UUID = Field(default_factory=uuid4)
    record: dict[str, Any]

    @model_validator(mode="after")
    def review_record_is_consistent(self) -> "ServicePlanReviewSubmission":
        if self.record.get("schema") != self.record_schema:
            raise ValueError("submission schema must match the review record schema")
        review_protocol = self.record.get("review_protocol", "legacy_unblinded_or_unknown")
        if not isinstance(review_protocol, str) or review_protocol not in {
            "blind_planner_choice_v1",
            "standard_review_v1",
            "legacy_unblinded_or_unknown",
        }:
            raise ValueError("review record has an unsupported review protocol")
        nested_request = self.record.get("request")
        service = None
        if isinstance(nested_request, dict):
            service = nested_request.get("service") or nested_request.get("hourly_service")
            if "service" in nested_request and "hourly_service" in nested_request:
                raise ValueError("review request must describe one timetable mode")
        is_hourly_review = (
            isinstance(nested_request, dict) and "hourly_service" in nested_request
        )
        ranking_method = self.record.get("ranking_method")
        if is_hourly_review and ranking_method != "deterministic_hourly_service_objective_v1":
            raise ValueError(
                "hourly review records must identify their deterministic ranking method"
            )
        if not is_hourly_review and ranking_method == "deterministic_hourly_service_objective_v1":
            raise ValueError("hourly ranking provenance requires an hourly service request")
        choice = self.record.get("planner_choice")
        candidates = self.record.get("candidate_set")
        if not isinstance(service, dict) or not isinstance(service.get("request_id"), str):
            raise ValueError("review record must include its service request ID")
        try:
            UUID(service["request_id"])
        except ValueError as error:
            raise ValueError("review record service request ID must be a UUID") from error
        if not isinstance(choice, dict) or choice.get("disposition") not in {
            "chosen_for_review",
            "no_feasible_option",
            "no_suitable_option",
        }:
            raise ValueError("review record must include a supported planner disposition")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("review record must include the evaluated candidate set")
        if not all(isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("review record candidates must be objects")
        snapshot = service.get("snapshot")
        if not isinstance(snapshot, dict) or self.record.get("snapshot") != snapshot:
            raise ValueError("review record snapshot must match its planning request")
        candidate_by_id: dict[str, dict[str, Any]] = {}
        feasible_ids: set[str] = set()
        candidate_order: list[str] = []
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in candidate_by_id
            ):
                raise ValueError("review record candidate IDs must be non-empty and unique")
            if type(candidate.get("feasible")) is not bool:
                raise ValueError("review record candidates must include a feasible flag")
            model_rank = candidate.get("research_model_rank")
            model_mean_risk = candidate.get("research_model_mean_conflict_risk")
            model_max_risk = candidate.get("research_model_max_conflict_risk")
            highest_risk_schedule_id = candidate.get(
                "research_model_highest_risk_schedule_id"
            )
            highest_risk_train_name = candidate.get("research_model_highest_risk_train_name")
            if (model_rank is None) != (model_mean_risk is None):
                raise ValueError("review record model rank and mean risk must appear together")
            if model_rank is not None and (
                not isinstance(model_rank, int)
                or isinstance(model_rank, bool)
                or model_rank < 1
                or not isinstance(model_mean_risk, int | float)
                or isinstance(model_mean_risk, bool)
                or not isfinite(model_mean_risk)
                or not 0 <= model_mean_risk <= 1
            ):
                raise ValueError("review record model rank and mean risk are invalid")
            if (model_max_risk is None) != (highest_risk_schedule_id is None):
                raise ValueError(
                    "review record maximum model risk and schedule ID must appear together"
                )
            if model_max_risk is not None and (
                model_rank is None
                or not isinstance(model_max_risk, int | float)
                or isinstance(model_max_risk, bool)
                or not isfinite(model_max_risk)
                or not 0 <= model_max_risk <= 1
                or not isinstance(highest_risk_schedule_id, str)
                or not highest_risk_schedule_id
                or model_max_risk < model_mean_risk
                or highest_risk_train_name is not None
                and (
                    not isinstance(highest_risk_train_name, str)
                    or not highest_risk_train_name
                )
            ):
                raise ValueError("review record maximum model risk details are invalid")
            validation = candidate.get("validation")
            if not isinstance(validation, dict):
                raise ValueError("review record candidates must include OSRD validation")
            pathfinding = validation.get("pathfinding_valid")
            simulation = validation.get("simulation_valid")
            conflict_count = validation.get("conflict_count")
            new_conflict_count = candidate.get("new_conflict_count")
            failure_reasons = validation.get("failure_reasons")
            if type(pathfinding) is not bool or type(simulation) is not bool:
                raise ValueError(
                    "review record OSRD pathfinding and simulation values must be boolean"
                )
            if (
                not isinstance(conflict_count, int)
                or isinstance(conflict_count, bool)
                or conflict_count < 0
                or not isinstance(new_conflict_count, int)
                or isinstance(new_conflict_count, bool)
                or not 0 <= new_conflict_count <= conflict_count
            ):
                raise ValueError("review record conflict counts are invalid")
            if not isinstance(failure_reasons, list) or not all(
                isinstance(reason, str) for reason in failure_reasons
            ):
                raise ValueError("review record OSRD failure reasons must be strings")
            expected_feasible = (
                pathfinding
                and simulation
                and new_conflict_count == 0
                and not any("cleanup failed" in reason.lower() for reason in failure_reasons)
            )
            if candidate["feasible"] is not expected_feasible:
                raise ValueError("review record feasibility must match its OSRD validation")
            candidate_by_id[candidate_id] = candidate
            candidate_order.append(candidate_id)
            rank = candidate.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                raise ValueError("review record candidates must preserve their displayed rank")
            if expected_feasible:
                feasible_ids.add(candidate_id)
        if [candidate["rank"] for candidate in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError("review candidate order must preserve contiguous displayed ranks")
        display_order = self.record.get("display_order_candidate_ids")
        if display_order is not None and (
            not isinstance(display_order, list)
            or not all(isinstance(candidate_id, str) for candidate_id in display_order)
            or len(display_order) != len(set(display_order))
            or set(display_order) != set(candidate_by_id)
            or display_order != candidate_order
        ):
            raise ValueError("review display order must match the displayed candidate ranks")
        research_model = self.record.get("research_model")
        if review_protocol == "blind_planner_choice_v1":
            if not isinstance(research_model, dict) or research_model.get("status") == "used":
                raise ValueError("blind review records cannot include an exposed model suggestion")
            study_group = self.record.get("study_infrastructure_group")
        if study_group is not None and (
            not isinstance(study_group, str) or not study_group.strip()
        ):
            raise ValueError("study infrastructure group must be a non-empty string")
            if not isinstance(study_group, str) or not study_group.strip():
                raise ValueError("blind review records must include a study infrastructure group")
            if any(candidate.get("research_model_rank") is not None for candidate in candidates):
                raise ValueError("blind review records cannot contain model scores")
        counts = self.record.get("plan_counts")
        if not isinstance(counts, dict) or any(
            not isinstance(counts.get(key), int)
            or isinstance(counts.get(key), bool)
            or counts[key] < 0
            for key in ("generated", "evaluated", "feasible")
        ):
            raise ValueError("review record plan counts must be nonnegative integers")
        if (
            counts["generated"] < counts["evaluated"]
            or counts["evaluated"] != len(candidates)
            or counts["feasible"] != len(feasible_ids)
        ):
            raise ValueError("review record plan counts do not match its candidates")
        selected_id = choice.get("candidate_id")
        selected_candidate = self.record.get("selected_candidate")
        if choice["disposition"] == "chosen_for_review":
            if (
                not isinstance(selected_id, str)
                or selected_id not in feasible_ids
                or selected_candidate != candidate_by_id[selected_id]
            ):
                raise ValueError("review choice must select one feasible candidate")
        elif selected_id is not None or selected_candidate is not None:
            raise ValueError("decline records must not select a candidate")
        elif choice["disposition"] == "no_feasible_option" and feasible_ids:
            raise ValueError("no-feasible-option record cannot include a feasible candidate")
        elif choice["disposition"] == "no_suitable_option" and not feasible_ids:
            raise ValueError("no-suitable-option record requires a feasible candidate")
        return self


class ServicePlanReviewReceipt(StrictModel):
    review_id: UUID
    request_id: UUID
    created_at: datetime
    review_sha256: str
