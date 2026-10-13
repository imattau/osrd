"""Versioned HTTP payloads for operations-planning requests and results."""

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so contract drift is visible to API clients."""

    model_config = ConfigDict(extra="forbid")


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
    before_start_time_ms: int
    after_start_time_ms: int
    shift_seconds: int


class PlanCandidateValidation(StrictModel):
    pathfinding_valid: bool
    simulation_valid: bool
    conflict_count: int | None
    failure_reasons: list[str]


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


class PlanAuditRecord(StrictModel):
    request_id: UUID
    status: Literal["completed"]
    created_at: datetime
    input_sha256: str
    result_sha256: str
    request: PlanningRequest
    result: PlanningResponse
