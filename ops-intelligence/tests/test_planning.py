from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from osrd_ops_intelligence.api.contracts import PlanningRequest
from osrd_ops_intelligence.service.osrd_client import CandidateValidation, SnapshotRef
from osrd_ops_intelligence.service.planning import PlanningInputError, plan


class StubOSRDClient:
    def __init__(self) -> None:
        self.validated_candidate_ids: list[str] = []

    async def describe_snapshot(self, *, infra_id: str, timetable_id: str) -> SnapshotRef:
        return SnapshotRef(infra_id, "1", timetable_id, "revision", "osrd-build")

    async def get_schedule_start_times(self, *, snapshot: SnapshotRef) -> dict[str, int]:
        return {"101": 1_000, "202": 2_000}

    async def validate_candidate(
        self, *, snapshot: SnapshotRef, candidate_payload: dict[str, object]
    ) -> CandidateValidation:
        self.validated_candidate_ids.append(str(candidate_payload["candidate_id"]))
        return CandidateValidation(True, True, 0)


def _request(policy: dict[str, Any], affected_ids: list[str] | None = None) -> PlanningRequest:
    now = datetime(2026, 10, 13, 8, tzinfo=UTC)
    return PlanningRequest.model_validate(
        {
            "scenario_id": "ranking-test",
            "snapshot": {
                "infra_id": "1",
                "infra_version": "1",
                "timetable_id": "10",
                "timetable_revision": "revision",
                "osrd_version": "osrd-build",
            },
            "incident": {
                "incident_id": "delay-1",
                "kind": "delay",
                "affected_resources": [{"resource_type": "route", "resource_id": "R1"}],
                "starts_at": now,
                "ends_at": now + timedelta(hours=1),
                "confidence": 1.0,
                "source": "test",
                "delay_seconds": 60,
                "affected_train_schedule_ids": affected_ids or ["101", "202"],
            },
            "horizon_start": now,
            "horizon_end": now + timedelta(hours=2),
            "policy": {"max_candidates": 1, "max_departure_shift_seconds": 60, **policy},
        }
    )


def test_weighted_objective_ranks_lower_priority_cost_first() -> None:
    client = StubOSRDClient()
    request = _request(
        {
            "objective": "minimize_weighted_shift",
            "schedule_priority_weights": {"101": 100.0, "202": 1.0},
        }
    )

    response = asyncio.run(plan(request, client))

    assert response.objective == "minimize_weighted_shift"
    assert response.baseline.strategy == "pass_through_rules_v1"
    assert response.baseline.edits == []
    assert response.generated_candidate_count == 3
    assert response.evaluated_candidate_count == 1
    assert response.candidates[0].edits[0].train_schedule_id == "202"
    assert response.candidates[0].objective_score == 60.0
    assert len(client.validated_candidate_ids) == 2
    assert client.validated_candidate_ids[0].startswith("pass-through-")


def test_protected_services_are_excluded_before_ranking() -> None:
    request = _request(
        {
            "objective": "minimize_changed_services",
            "protected_train_schedule_ids": ["101"],
        }
    )

    response = asyncio.run(plan(request, StubOSRDClient()))

    assert response.generated_candidate_count == 1
    assert response.candidates[0].edits[0].train_schedule_id == "202"
    assert response.candidates[0].objective_score == 1.0


def test_weighted_objective_requires_weights_for_unprotected_affected_services() -> None:
    request = _request(
        {
            "objective": "minimize_weighted_shift",
            "schedule_priority_weights": {"101": 2.0},
        }
    )

    with pytest.raises(PlanningInputError, match="needs priorities"):
        asyncio.run(plan(request, StubOSRDClient()))
