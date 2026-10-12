"""Synchronous-request orchestration for validated departure-shift proposals."""

from __future__ import annotations

from dataclasses import asdict

from osrd_ops_intelligence.api.contracts import (
    IncidentKind,
    PlanCandidateResult,
    PlanningObjective,
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
    candidate: DepartureShiftCandidate,
    validation: CandidateValidation,
    objective_score: float,
) -> PlanCandidateResult:
    # Kept local so API results expose typed fields, not the mutable OSRD request payload.
    return PlanCandidateResult.model_validate(
        {
            "candidate_id": candidate.candidate_id,
            "edits": [asdict(edit) for edit in candidate.edits],
            "total_shift_seconds": candidate.total_shift_seconds,
            "objective_score": objective_score,
            "validation": {
                "pathfinding_valid": validation.pathfinding_valid,
                "simulation_valid": validation.simulation_valid,
                "conflict_count": validation.conflict_count,
                "failure_reasons": list(validation.failure_reasons),
            },
        }
    )


def _objective_score(
    candidate: DepartureShiftCandidate,
    objective: PlanningObjective,
    schedule_priority_weights: dict[str, float],
) -> float:
    if objective == PlanningObjective.MINIMIZE_TOTAL_SHIFT:
        return float(candidate.total_shift_seconds)
    if objective == PlanningObjective.MINIMIZE_CHANGED_SERVICES:
        return float(len(candidate.edits))
    return float(
        sum(
            edit.shift_seconds * schedule_priority_weights[edit.train_schedule_id]
            for edit in candidate.edits
        )
    )


def _ranking_key(
    candidate: DepartureShiftCandidate,
    score: float,
) -> tuple[float, int, int, tuple[str, ...]]:
    return (
        score,
        candidate.total_shift_seconds,
        len(candidate.edits),
        tuple(sorted(edit.train_schedule_id for edit in candidate.edits)),
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
    affected_ids = set(request.incident.affected_train_schedule_ids)
    protected_ids = set(request.policy.protected_train_schedule_ids)
    weights = request.policy.schedule_priority_weights
    unknown_policy_ids = (protected_ids | set(weights)) - set(start_times)
    if unknown_policy_ids:
        raise PlanningInputError(
            f"Policy references schedules outside the bound timetable: {sorted(unknown_policy_ids)}"
        )
    if request.policy.objective == PlanningObjective.MINIMIZE_WEIGHTED_SHIFT:
        missing_weights = (affected_ids - protected_ids) - set(weights)
        if missing_weights:
            raise PlanningInputError(
                "Weighted-shift objective needs priorities for affected schedules: "
                f"{sorted(missing_weights)}"
            )

    # Generate the bounded one-service and coordinated variants, rank them under
    # operator policy, then use the validation budget only on the best candidates.
    proposal_budget = max(request.policy.max_candidates, 4 * len(affected_ids) + 4)
    generated = generate_departure_shift_candidates(
        incident_id=request.incident.incident_id,
        snapshot=request.snapshot,
        schedule_start_times_ms=start_times,
        affected_schedule_ids=request.incident.affected_train_schedule_ids,
        protected_schedule_ids=sorted(protected_ids),
        delay_seconds=request.incident.delay_seconds,
        max_departure_shift_seconds=request.policy.max_departure_shift_seconds,
        max_candidates=proposal_budget,
    )
    scored = [
        (
            candidate,
            _objective_score(candidate, request.policy.objective, weights),
        )
        for candidate in generated
    ]
    scored.sort(key=lambda item: _ranking_key(item[0], item[1]))
    selected = scored[: request.policy.max_candidates]

    accepted: list[PlanCandidateResult] = []
    rejected: list[PlanCandidateResult] = []
    for candidate, score in selected:
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload=candidate.to_dict(),
        )
        rendered = _render_candidate(candidate, validation, score)
        (accepted if validation.validated else rejected).append(rendered)

    return PlanningResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        objective=request.policy.objective,
        generated_candidate_count=len(generated),
        evaluated_candidate_count=len(selected),
        candidates=accepted,
        rejected_candidates=rejected,
    )
