"""Versioned HTTP payloads for operations-planning requests and results."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so contract drift is visible to API clients."""

    model_config = ConfigDict(extra="forbid")


class IncidentKind(StrEnum):
    DELAY = "delay"
    TRACK_CLOSURE = "track_closure"
    SPEED_RESTRICTION = "speed_restriction"
    ROLLING_STOCK_UNAVAILABLE = "rolling_stock_unavailable"


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

    @model_validator(mode="after")
    def interval_is_valid_and_timezone_aware(self) -> "Incident":
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("incident timestamps must include a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be later than starts_at")
        return self


class SnapshotBinding(StrictModel):
    infra_id: str = Field(min_length=1)
    infra_version: str = Field(min_length=1)
    timetable_id: str = Field(min_length=1)
    timetable_revision: str = Field(min_length=1)
    osrd_version: str = Field(min_length=1)


class PlanningPolicy(StrictModel):
    max_candidates: int = Field(default=10, ge=1, le=50)
    max_departure_shift_seconds: int = Field(default=900, ge=0, le=7200)
    protected_train_schedule_ids: list[str] = Field(default_factory=list)


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
