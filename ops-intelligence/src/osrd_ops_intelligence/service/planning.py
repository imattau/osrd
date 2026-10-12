"""Synchronous-request orchestration for validated departure-shift proposals."""

from __future__ import annotations

from dataclasses import asdict

from osrd_ops_intelligence.api.contracts import (
    IncidentKind,
    PlanCandidateResult,
    PlanningRequest,
    PlanningResponse,
)
from osrd_ops_intelligence.optimizer.departure_shifts import (
    DepartureShiftCandidate,
    generate_departure_shift_candidates,
)
from osrd_ops_intelligence.service.osrd_client import (
    CandidateValidation,
    OSRDClient,
    SnapshotRef,
)


class PlanningInputError(ValueError):
    """Raised when a request lacks inputs supported by this first planner slice."""


def _render_candidate(
    candidate: DepartureShiftCandidate, validation: CandidateValidation
) -> PlanCandidateResult:
    # Kept local so API results expose typed fields, not the mutable OSRD request payload.
    return PlanCandidateResult.model_validate(
        {
            "candidate_id": candidate.candidate_id,
            "edits": [asdict(edit) for edit in candidate.edits],
            "total_shift_seconds": candidate.total_shift_seconds,
            "validation": {
                "pathfinding_valid": validation.pathfinding_valid,
                "simulation_valid": validation.simulation_valid,
                "conflict_count": validation.conflict_count,
                "failure_reasons": list(validation.failure_reasons),
            },
        }
    )


async def plan(request: PlanningRequest, client: OSRDClient) -> PlanningResponse:
    if request.incident.kind != IncidentKind.DELAY:
        raise PlanningInputError("The first planner slice supports delay incidents only")
    if request.incident.delay_seconds is None or not request.incident.affected_train_schedule_ids:
        raise PlanningInputError(
            "Delay planning requires delay_seconds and affected_train_schedule_ids"
        )

    snapshot = SnapshotRef(**request.snapshot.model_dump())
    start_times = await client.get_schedule_start_times(snapshot=snapshot)
    candidates = generate_departure_shift_candidates(
        incident_id=request.incident.incident_id,
        snapshot=request.snapshot,
        schedule_start_times_ms=start_times,
        affected_schedule_ids=request.incident.affected_train_schedule_ids,
        protected_schedule_ids=request.policy.protected_train_schedule_ids,
        delay_seconds=request.incident.delay_seconds,
        max_departure_shift_seconds=request.policy.max_departure_shift_seconds,
        max_candidates=request.policy.max_candidates,
    )

    accepted: list[PlanCandidateResult] = []
    rejected: list[PlanCandidateResult] = []
    for candidate in candidates:
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload=candidate.to_dict(),
        )
        rendered = _render_candidate(candidate, validation)
        (accepted if validation.validated else rejected).append(rendered)

    return PlanningResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        generated_candidate_count=len(candidates),
        candidates=accepted,
        rejected_candidates=rejected,
    )
