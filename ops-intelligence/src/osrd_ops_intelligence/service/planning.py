"""Synchronous-request orchestration for validated departure-shift proposals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict

from osrd_ops_intelligence.api.contracts import (
    CandidateValidationOrder,
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
    UnsupportedSnapshot,
)


class PlanningInputError(ValueError):
    """Raised when a request lacks inputs supported by this first planner slice."""


def _render_candidate(
    candidate: DepartureShiftCandidate,
    validation: CandidateValidation,
    objective_score: float,
    baseline_conflict_count: int,
    new_conflict_count: int,
) -> PlanCandidateResult:
    # Kept local so API results expose typed fields, not the mutable OSRD request payload.
    return PlanCandidateResult.model_validate(
        {
            "candidate_id": candidate.candidate_id,
            "strategy": candidate.strategy,
            "edits": [asdict(edit) for edit in candidate.edits],
            "total_shift_seconds": candidate.total_shift_seconds,
            "objective_score": objective_score,
            "validation": {
                "pathfinding_valid": validation.pathfinding_valid,
                "simulation_valid": validation.simulation_valid,
                "conflict_count": validation.conflict_count,
                "baseline_conflict_count": baseline_conflict_count,
                "new_conflict_count": new_conflict_count,
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


def _pass_through_candidate(request: PlanningRequest) -> DepartureShiftCandidate:
    identity = {
        "incident_id": request.incident.incident_id,
        "snapshot": request.snapshot.model_dump(mode="json"),
        "strategy": "pass_through_rules_v1",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return DepartureShiftCandidate(
        candidate_id=f"pass-through-{digest}",
        incident_id=request.incident.incident_id,
        snapshot=request.snapshot,
        strategy="pass_through_rules_v1",
        edits=(),
        total_shift_seconds=0,
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


def _result_ranking_key(
    candidate: PlanCandidateResult,
) -> tuple[bool, bool, int, float, int, int, tuple[str, ...]]:
    return (
        not candidate.validation.pathfinding_valid,
        not candidate.validation.simulation_valid,
        candidate.validation.new_conflict_count,
        candidate.objective_score,
        candidate.total_shift_seconds,
        len(candidate.edits),
        tuple(sorted(edit.train_schedule_id for edit in candidate.edits)),
    )


def _diverse_validation_order(
    scored: list[tuple[DepartureShiftCandidate, float]],
) -> list[tuple[DepartureShiftCandidate, float]]:
    """Round-robin candidate edit patterns, retaining objective order within each pattern."""
    groups: dict[tuple[str, ...], list[tuple[DepartureShiftCandidate, float]]] = {}
    for item in scored:
        signature = tuple(sorted(edit.train_schedule_id for edit in item[0].edits))
        groups.setdefault(signature, []).append(item)
    ordered_signatures = sorted(groups, key=lambda signature: (len(signature), signature))
    result: list[tuple[DepartureShiftCandidate, float]] = []
    for rank in range(max((len(groups[key]) for key in ordered_signatures), default=0)):
        result.extend(groups[key][rank] for key in ordered_signatures if rank < len(groups[key]))
    return result


async def plan(request: PlanningRequest, client: OSRDClient) -> PlanningResponse:
    if request.incident.kind != IncidentKind.DELAY:
        raise PlanningInputError("The first planner slice supports delay incidents only")
    if request.incident.delay_seconds is None or not request.incident.affected_train_schedule_ids:
        raise PlanningInputError(
            "Delay planning requires delay_seconds and affected_train_schedule_ids"
        )

    snapshot = SnapshotRef(**request.snapshot.model_dump())
    baseline_candidate = _pass_through_candidate(request)
    metadata = await client.get_schedule_metadata(snapshot=snapshot)
    start_times = metadata.start_times_ms
    affected_ids = set(request.incident.affected_train_schedule_ids)
    protected_ids = set(request.policy.protected_train_schedule_ids)
    weights = request.policy.schedule_priority_weights
    shift_limits = dict(request.policy.max_shift_seconds_by_schedule)
    if metadata.timetable_type == "HOURLY":
        for schedule_id, hourly_limit in metadata.max_hourly_shift_seconds_by_schedule.items():
            shift_limits[schedule_id] = min(
                shift_limits.get(schedule_id, hourly_limit), hourly_limit
            )
    unknown_policy_ids = (protected_ids | set(weights) | set(shift_limits)) - set(start_times)
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
        max_shift_seconds_by_schedule=shift_limits,
    )
    scored = [
        (
            candidate,
            _objective_score(candidate, request.policy.objective, weights),
        )
        for candidate in generated
    ]
    scored.sort(key=lambda item: _ranking_key(item[0], item[1]))
    validation_pool = (
        _diverse_validation_order(scored)
        if request.policy.candidate_validation_order == CandidateValidationOrder.DIVERSE
        else scored
    )
    selected = validation_pool[: request.policy.max_candidates]

    baseline_validation = await client.validate_candidate(
        snapshot=snapshot,
        candidate_payload=baseline_candidate.to_dict(),
    )
    baseline_signatures = Counter(baseline_validation.conflict_signatures)
    baseline = _render_candidate(
        baseline_candidate,
        baseline_validation,
        0.0,
        len(baseline_validation.conflict_signatures),
        0,
    )
    accepted: list[PlanCandidateResult] = []
    rejected: list[PlanCandidateResult] = []
    for candidate, score in selected:
        validation = await client.validate_candidate(
            snapshot=snapshot,
            candidate_payload=candidate.to_dict(),
        )
        observed_baseline = Counter(validation.baseline_conflict_signatures)
        if observed_baseline != baseline_signatures:
            raise UnsupportedSnapshot("Baseline conflicts changed during candidate validation")
        new_conflict_count = sum(
            (Counter(validation.conflict_signatures) - baseline_signatures).values()
        )
        rendered = _render_candidate(
            candidate,
            validation,
            score,
            len(baseline_signatures),
            new_conflict_count,
        )
        candidate_is_acceptable = (
            validation.pathfinding_valid
            and validation.simulation_valid
            and new_conflict_count == 0
            and not any("cleanup failed" in reason for reason in validation.failure_reasons)
        )
        (accepted if candidate_is_acceptable else rejected).append(rendered)

    accepted.sort(key=_result_ranking_key)
    rejected.sort(key=_result_ranking_key)

    return PlanningResponse(
        request_id=request.request_id,
        snapshot=request.snapshot,
        objective=request.policy.objective,
        candidate_validation_order=request.policy.candidate_validation_order,
        generated_candidate_count=len(generated),
        evaluated_candidate_count=len(selected),
        baseline=baseline,
        candidates=accepted,
        rejected_candidates=rejected,
    )
