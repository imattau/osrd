from fastapi.testclient import TestClient

from osrd_ops_intelligence.service.app import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/healthz").json()["status"] == "ok"


def test_create_plan_rejects_invalid_contract() -> None:
    assert client.post("/v1/operations/plans", json={}).status_code == 422


def test_create_plan_is_not_implemented() -> None:
    body = {
        "scenario_id": "1",
        "snapshot": {
            "infra_id": "1",
            "infra_version": "v1",
            "timetable_id": "1",
            "timetable_revision": "r1",
            "osrd_version": "x",
        },
        "incident": {
            "incident_id": "i",
            "kind": "delay",
            "affected_resources": [{"resource_type": "route", "resource_id": "R"}],
            "starts_at": "2026-10-13T08:00:00Z",
            "ends_at": "2026-10-13T09:00:00Z",
            "confidence": 1,
            "source": "test",
        },
        "horizon_start": "2026-10-13T08:00:00Z",
        "horizon_end": "2026-10-13T12:00:00Z",
    }
    response = client.post("/v1/operations/plans", json=body)
    assert response.status_code == 501
    assert response.json()["code"] == "planning_not_configured"
