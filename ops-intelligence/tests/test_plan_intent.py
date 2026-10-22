from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import pytest

from osrd_ops_intelligence.api.contracts import PlanIntentRequest
from osrd_ops_intelligence.service import plan_intent
from osrd_ops_intelligence.service.app import create_app
from osrd_ops_intelligence.service.plan_intent import PlanIntentUnavailable


def test_intent_interpretation_requires_per_request_external_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_ENABLED", "true")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_PROVIDER", "openai")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def unexpected_call(*_args: object) -> str:
        raise AssertionError("provider must not be called without request consent")

    monkeypatch.setattr(plan_intent, "_call_responses_api", unexpected_call)
    with pytest.raises(PlanIntentUnavailable, match="Confirm"):
        asyncio.run(
            plan_intent.interpret_plan_intent(
                PlanIntentRequest(request_text="Add weekday rail service")
            )
        )


def test_interpretation_returns_reviewable_unresolved_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_ENABLED", "true")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_PROVIDER", "openai")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[tuple[str, str, str]] = []

    def fake_provider(api_key: str, model: str, request_text: str) -> str:
        calls.append((api_key, model, request_text))
        return (
            '{"status":"needs_clarification","summary":"A weekday service request",'
            '"services":[{"train_name":null,"origin":"West Station",'
            '"destination":"Mid East Station","intermediate_stops":[],"rolling_stock":null,'
            '"operating_start_date":null,"operating_end_date":null,'
            '"operating_start":"07:00","operating_end":"10:00","time_zone":null,'
            '"weekdays":[0,1,2,3,4],"frequency_minutes":30,"max_journey_minutes":null,'
            '"patronage_preference":"unspecified"}],'
            '"questions":["Which local time zone should apply?"],"assumptions":[],'
            '"unresolved_terms":["West Station","Mid East Station"]}'
        )

    async def run_inline(function: object, *args: object) -> object:
        assert callable(function)
        return function(*args)

    monkeypatch.setattr(plan_intent, "_call_responses_api", fake_provider)
    monkeypatch.setattr(plan_intent.asyncio, "to_thread", run_inline)
    draft = asyncio.run(
        plan_intent.interpret_plan_intent(
            PlanIntentRequest(
                request_text="Weekday service from West Station to Mid East Station",
                external_processing_confirmed=True,
            )
        )
    )

    assert draft.status == "needs_clarification"
    assert draft.services[0].origin == "West Station"
    assert draft.services[0].weekdays == [0, 1, 2, 3, 4]
    assert "Which local time zone should apply?" in draft.questions
    assert any("calendar start and end date" in question for question in draft.questions)
    assert calls == [
        (
            "test-key",
            "test-model",
            "Weekday service from West Station to Mid East Station",
        )
    ]


def test_strict_provider_schema_requires_every_nullable_field() -> None:
    schema = plan_intent._strict_schema()
    assert schema["required"] == list(schema["properties"])
    service_schema = schema["$defs"]["PlanIntentServiceDraft"]
    assert service_schema["required"] == list(service_schema["properties"])


def test_retiming_only_request_discards_invented_new_service_pattern() -> None:
    response = {
        "status": "needs_clarification",
        "summary": "A retiming and service request",
        "services": [{
            "train_name": None,
            "origin": "West_station",
            "destination": "Mid_East_station",
            "intermediate_stops": [],
            "rolling_stock": None,
            "operating_start_date": None,
            "operating_end_date": None,
            "operating_start": None,
            "operating_end": None,
            "time_zone": None,
            "weekdays": [],
            "frequency_minutes": None,
            "max_journey_minutes": None,
            "patronage_preference": "unspecified",
        }],
        "existing_service_changes": [{
            "train_name": "AI local demo regional service rerun-01",
            "direction": "either",
            "max_shift_minutes": None,
        }],
        "questions": [
            "Is the requested shift an exact target or a permitted maximum?",
            "What are the origin and destination stations?",
            "Please specify the rolling stock and calendar dates.",
            "What is the operating window for this service?",
        ],
        "assumptions": [],
        "unresolved_terms": ["West_station", "Mid_East_station"],
    }

    draft = plan_intent._validated_draft(
        json.dumps(response),
        "Move the existing service AI local demo regional service rerun-01 later "
        "by up to 2 minutes. Do not add a new service.",
    )

    assert draft.status == "ready_for_review"
    assert draft.services == []
    assert draft.existing_service_changes[0].direction == "later"
    assert draft.existing_service_changes[0].max_shift_minutes == 2
    assert draft.questions == []
    assert draft.unresolved_terms == []


def test_plan_intent_endpoint_rejects_unconsented_request_text() -> None:
    app = create_app()
    route = next(route for route in app.routes if route.path == "/v1/operations/plan-intent")
    response = asyncio.run(
        route.endpoint(
            PlanIntentRequest(
                request_text="Add a weekday rail service",
                external_processing_confirmed=False,
            )
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)["code"] == "external_processing_not_confirmed"


def test_ollama_provider_uses_local_structured_output_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_ENABLED", "true")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_PROVIDER", "ollama")
    monkeypatch.setenv("OSRD_OPS_PLAN_INTENT_MODEL", "qwen3.5:latest")
    monkeypatch.setenv("OSRD_OPS_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called: list[tuple[str, str, str]] = []

    def fake_ollama(base_url: str, model: str, request_text: str) -> str:
        called.append((base_url, model, request_text))
        return (
            '{"status":"needs_clarification","summary":"A weekday service request",'
            '"services":[{"train_name":null,"origin":"West Station",'
            '"destination":"Mid East Station","intermediate_stops":[],"rolling_stock":null,'
            '"operating_start_date":null,"operating_end_date":null,'
            '"operating_start":"07:00","operating_end":"10:00","time_zone":null,'
            '"weekdays":[0,1,2,3,4],"frequency_minutes":30,"max_journey_minutes":null,'
            '"patronage_preference":"unspecified"}],'
            '"questions":["Which local time zone should apply?"],"assumptions":[],'
            '"unresolved_terms":["West Station","Mid East Station"]}'
        )

    async def run_inline(function: object, *args: object) -> object:
        assert callable(function)
        return function(*args)

    monkeypatch.setattr(plan_intent, "_call_ollama_api", fake_ollama)
    monkeypatch.setattr(plan_intent.asyncio, "to_thread", run_inline)
    draft = asyncio.run(
        plan_intent.interpret_plan_intent(
            PlanIntentRequest(
                request_text="Weekday service from West Station to Mid East Station",
                external_processing_confirmed=True,
            )
        )
    )

    assert draft.status == "needs_clarification"
    assert called == [
        (
            "http://127.0.0.1:11434",
            "qwen3.5:latest",
            "Weekday service from West Station to Mid East Station",
        )
    ]


def test_ollama_request_uses_constrained_schema_and_suppresses_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message":{"content":"{}"}}'

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(plan_intent, "urlopen", fake_urlopen)
    content = plan_intent._call_ollama_api(
        "http://127.0.0.1:11434", "qwen3.5:latest", "Synthetic service request"
    )
    payload: Any = captured["payload"]

    assert content == "{}"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert isinstance(payload, dict)
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"]["type"] == "object"
    assert payload["messages"][1]["content"] == "Synthetic service request"


def test_model_cannot_invent_patronage_objective_or_placeholder_terms() -> None:
    content = json.dumps(
        {
            "status": "needs_clarification",
            "summary": "A half-hourly rail service",
            "services": [
                {
                    "train_name": "[Unresolved]",
                    "origin": "West Station",
                    "destination": "Mid East Station",
                    "intermediate_stops": [],
                    "rolling_stock": None,
                    "operating_start_date": None,
                    "operating_end_date": None,
                    "operating_start": "07:00",
                    "operating_end": "10:00",
                    "time_zone": None,
                    "weekdays": [0, 1, 2, 3, 4],
                    "frequency_minutes": 30,
                    "max_journey_minutes": None,
                    "patronage_preference": "station_time_bucket_coverage",
                }
            ],
            "questions": ["What is the local time zone?"],
            "assumptions": [],
            "unresolved_terms": ["West Station", "[Unresolved]"],
        }
    )

    draft = plan_intent._validated_draft(
        content,
        "Weekday half-hourly service from West Station to Mid East Station, 07:00–10:00.",
    )

    assert draft.services[0].patronage_preference == "unspecified"
    assert draft.services[0].train_name is None
    assert draft.unresolved_terms == ["West Station"]
    assert len([question for question in draft.questions if "time zone" in question]) == 1


def test_numeric_service_constraints_are_grounded_in_request_text() -> None:
    response = {
        "status": "ready_for_review",
        "summary": "A weekday service",
        "services": [{
            "train_name": "West to Mid East",
            "origin": "West Station",
            "destination": "Mid East Station",
            "intermediate_stops": [],
            "rolling_stock": "DMU Regional train",
            "operating_start_date": "2026-10-22",
            "operating_end_date": "2026-10-22",
            "operating_start": "10:00",
            "operating_end": "11:00",
            "time_zone": "UTC",
            "weekdays": [3],
            "frequency_minutes": 30,
            "max_journey_minutes": 45,
            "patronage_preference": "unspecified",
        }],
        "existing_service_changes": [],
        "questions": [],
        "assumptions": [],
        "unresolved_terms": [],
    }

    inferred_limit = plan_intent._validated_draft(
        json.dumps(response),
        "Run the West to Mid East service from West Station to Mid East Station using "
        "DMU Regional train on 2026-10-22, 10:00 to 11:00 UTC, every 20 minutes.",
    )

    assert inferred_limit.services[0].frequency_minutes == 20
    assert inferred_limit.services[0].max_journey_minutes is None

    explicit_limit = plan_intent._validated_draft(
        json.dumps(response),
        "Run the West to Mid East service from West Station to Mid East Station using "
        "DMU Regional train on 2026-10-22, 10:00 to 11:00 UTC, every 20 minutes, "
        "with a maximum journey time of 40 minutes.",
    )

    assert explicit_limit.services[0].frequency_minutes == 20
    assert explicit_limit.services[0].max_journey_minutes == 40

    false_questions = json.loads(json.dumps(response))
    false_questions["status"] = "needs_clarification"
    false_questions["questions"] = [
        "The 30-minute frequency conflicts with the 10:00 to 11:00 operating window.",
        "Is the requested shift for any existing services an exact target or a maximum?",
        "Is the operating window from 10:00 to 11:00 UTC inclusive?",
        "Is the maximum journey time of 40 minutes a hard limit or a target?",
    ]
    normalized = plan_intent._validated_draft(
        json.dumps(false_questions),
        "Run the West to Mid East service from West Station to Mid East Station using "
        "DMU Regional train on 2026-10-22, 10:00 to 11:00 UTC, every 20 minutes, "
        "with a maximum journey time of 40 minutes.",
    )

    assert normalized.status == "ready_for_review"
    assert normalized.questions == []

    incomplete_model = json.loads(json.dumps(response))
    incomplete_model["status"] = "needs_clarification"
    incomplete_model["services"][0]["rolling_stock"] = "DMU Regional"
    incomplete_model["services"][0]["time_zone"] = None
    incomplete_model["questions"] = ["Please specify the IANA time zone for this service."]
    incomplete_model["unresolved_terms"] = ["West Station", "Mid East Station", "DMU Regional"]
    recovered = plan_intent._validated_draft(
        json.dumps(incomplete_model),
        "Run the West to Mid East service from West Station to Mid East Station using "
        "DMU Regional train on 2026-10-22, 10:00 to 11:00 UTC, every 20 minutes, "
        "with a maximum journey time of 40 minutes.",
    )

    assert recovered.status == "ready_for_review"
    assert recovered.services[0].rolling_stock == "DMU Regional train"
    assert recovered.services[0].time_zone == "UTC"
    assert "DMU Regional train" in recovered.unresolved_terms
    assert recovered.questions == []


def test_missing_frequency_is_asked_instead_of_taken_from_model() -> None:
    response = {
        "status": "ready_for_review",
        "summary": "A weekday service",
        "services": [{
            "train_name": "West to Mid East",
            "origin": "West Station",
            "destination": "Mid East Station",
            "intermediate_stops": [],
            "rolling_stock": "DMU Regional train",
            "operating_start_date": "2026-10-22",
            "operating_end_date": "2026-10-22",
            "operating_start": "10:00",
            "operating_end": "11:00",
            "time_zone": "UTC",
            "weekdays": [3],
            "frequency_minutes": 20,
            "max_journey_minutes": 30,
            "patronage_preference": "unspecified",
        }],
        "existing_service_changes": [],
        "questions": [],
        "assumptions": [],
        "unresolved_terms": [],
    }

    draft = plan_intent._validated_draft(
        json.dumps(response),
        "Run the West to Mid East service from West Station to Mid East Station using "
        "DMU Regional train on 2026-10-22, 10:00 to 11:00 UTC.",
    )

    assert draft.status == "needs_clarification"
    assert draft.services[0].frequency_minutes is None
    assert draft.services[0].max_journey_minutes is None
    assert any("service frequency" in question for question in draft.questions)


def test_unstated_dates_times_and_timezone_are_not_taken_from_model() -> None:
    response = {
        "status": "ready_for_review",
        "summary": "A weekday service",
        "services": [{
            "train_name": "West to Mid East",
            "origin": "West Station",
            "destination": "Mid East Station",
            "intermediate_stops": [],
            "rolling_stock": "DMU Regional train",
            "operating_start_date": "2027-04-01",
            "operating_end_date": "2027-04-30",
            "operating_start": "07:00",
            "operating_end": "10:00",
            "time_zone": "Europe/Paris",
            "weekdays": [0, 1, 2, 3, 4],
            "frequency_minutes": 30,
            "max_journey_minutes": 45,
            "patronage_preference": "unspecified",
        }],
        "existing_service_changes": [],
        "questions": [],
        "assumptions": [],
        "unresolved_terms": [],
    }

    draft = plan_intent._validated_draft(
        json.dumps(response),
        "Add a weekday service from West Station to Mid East Station using DMU Regional train.",
    )

    service = draft.services[0]
    assert draft.status == "needs_clarification"
    assert service.operating_start_date is None
    assert service.operating_end_date is None
    assert service.operating_start is None
    assert service.operating_end is None
    assert service.time_zone is None
    assert service.frequency_minutes is None
    assert service.max_journey_minutes is None


def test_explicit_natural_date_and_clock_ranges_are_parsed_deterministically() -> None:
    assert plan_intent._explicit_operating_dates(
        "Operate from 22 October 2026 through October 24, 2026."
    ) == (date(2026, 10, 22), date(2026, 10, 24))
    assert plan_intent._explicit_operating_time_ranges(
        "Operate every weekday between 10 am and 11:30 a.m. UTC."
    ) == [("10:00", "11:30")]


def test_explicit_named_patterns_exclude_the_following_action_verb() -> None:
    request = (
        "Create two patterns: Pattern named Harbor Peak runs from West Station to Mid East Station; "
        "Pattern named Airport Connector runs from North Station to South Station."
    )

    assert plan_intent._explicit_named_services(request) == [
        "Harbor Peak",
        "Airport Connector",
    ]


def test_stock_extraction_ignores_introductory_with_clauses() -> None:
    request = (
        "Create a complete timetable with two named weekday service patterns in "
        "Australia/Melbourne using DMU Regional train. Pattern named Harbor Peak runs "
        "from West Station to Mid East Station with a maximum journey time of 25 minutes. "
        "Pattern named Airport Connector runs from North Station to South Station "
        "with an Electric EMU."
    )

    assert plan_intent._explicit_rolling_stock_names(request) == [
        "DMU Regional train",
        "Electric EMU",
    ]


def test_multi_pattern_draft_rejects_ungrounded_assumptions_and_asks_for_all_dates() -> None:
    response = {
        "status": "needs_clarification",
        "summary": "Two weekday service patterns",
        "services": [
            {
                "train_name": "Harbor Peak runs",
                "origin": "West_station",
                "destination": "Mid_East_station",
                "intermediate_stops": [],
                "rolling_stock": "two named weekday service patterns in Australia/Melbourne using DMU Regional train",
                "operating_start_date": "2025-04-01",
                "operating_end_date": "2025-04-30",
                "operating_start": "06:00",
                "operating_end": "08:00",
                "time_zone": None,
                "weekdays": [0, 1, 2, 3, 4],
                "frequency_minutes": 30,
                "max_journey_minutes": 25,
                "patronage_preference": "unspecified",
            },
            {
                "train_name": "Airport Connector runs",
                "origin": "West_station",
                "destination": "Mid_East_station",
                "intermediate_stops": [],
                "rolling_stock": "two named weekday service patterns in Australia/Melbourne using DMU Regional train",
                "operating_start_date": "2025-04-01",
                "operating_end_date": "2025-04-30",
                "operating_start": "06:30",
                "operating_end": "08:30",
                "time_zone": None,
                "weekdays": [0, 1, 2, 3, 4],
                "frequency_minutes": 60,
                "max_journey_minutes": 35,
                "patronage_preference": "unspecified",
            },
        ],
        "existing_service_changes": [],
        "questions": ["Please specify the calendar start and end date for Harbor Peak runs."],
        "assumptions": [
            "The operating dates are fixed as 2025-04-01.",
            "The time zones are not specified, so they are omitted.",
        ],
        "unresolved_terms": [
            "West_station",
            "Mid_East_station",
            "two named weekday service patterns in Australia/Melbourne using DMU Regional train",
        ],
    }
    request = (
        "Create a complete timetable with two named weekday service patterns in "
        "Australia/Melbourne using DMU Regional train. Pattern named Harbor Peak runs "
        "from West_station to Mid_East_station every 30 minutes from 06:00 to 08:00, "
        "with a maximum journey time of 25 minutes. Pattern named Airport Connector runs "
        "from West_station to Mid_East_station every 60 minutes from 06:30 to 08:30, "
        "with a maximum journey time of 35 minutes. Keep existing services unchanged."
    )

    draft = plan_intent._validated_draft(json.dumps(response), request)

    assert [service.train_name for service in draft.services] == [
        "Harbor Peak",
        "Airport Connector",
    ]
    assert [service.rolling_stock for service in draft.services] == [
        "DMU Regional train",
        "DMU Regional train",
    ]
    assert [service.frequency_minutes for service in draft.services] == [30, 60]
    assert [service.max_journey_minutes for service in draft.services] == [25, 35]
    assert all(service.operating_start_date is None for service in draft.services)
    assert all(service.operating_end_date is None for service in draft.services)
    assert draft.assumptions == ["Interpreted 'weekdays' as Monday-Friday."]
    assert draft.questions == [
        "Please specify the calendar start and end date for each service pattern."
    ]


def test_multiple_explicit_journey_caps_remove_redundant_target_questions() -> None:
    response = {
        "status": "needs_clarification",
        "summary": "Two explicit service patterns",
        "services": [
            {
                "train_name": "Regional 1", "origin": "West Station",
                "destination": "Mid East Station", "intermediate_stops": [],
                "rolling_stock": "DMU Regional train", "operating_start_date": "2026-10-22",
                "operating_end_date": "2026-10-22", "operating_start": "10:00",
                "operating_end": "11:00", "time_zone": "UTC", "weekdays": [3],
                "frequency_minutes": 30, "max_journey_minutes": 40,
                "patronage_preference": "unspecified",
            },
            {
                "train_name": "Express 2", "origin": "North Station",
                "destination": "South Station", "intermediate_stops": [],
                "rolling_stock": "Electric EMU", "operating_start_date": "2026-10-22",
                "operating_end_date": "2026-10-22", "operating_start": "12:00",
                "operating_end": "13:00", "time_zone": "UTC", "weekdays": [3],
                "frequency_minutes": 15, "max_journey_minutes": 60,
                "patronage_preference": "unspecified",
            },
        ],
        "existing_service_changes": [],
        "questions": [
            "Is the maximum journey time of 40 minutes for Regional 1 an exact target or a permitted maximum?",
            "Is the maximum journey time of 60 minutes for Express 2 an exact target or a permitted maximum?",
        ],
        "assumptions": [],
        "unresolved_terms": [
            "West Station", "Mid East Station", "North Station", "South Station",
            "DMU Regional train", "Electric EMU",
        ],
    }

    draft = plan_intent._validated_draft(
        json.dumps(response),
        "Create two weekday services on 2026-10-22 UTC. Regional 1 runs from West Station "
        "to Mid East Station using DMU Regional train, every 30 minutes between 10:00 and "
        "11:00, with a maximum journey time of 40 minutes. Express 2 runs from North Station "
        "to South Station using Electric EMU, every 15 minutes between 12:00 and 13:00, "
        "with a maximum journey time of 60 minutes.",
    )

    assert draft.services[0].frequency_minutes == 30
    assert draft.services[0].max_journey_minutes == 40
    assert draft.services[1].frequency_minutes == 15
    assert draft.services[1].max_journey_minutes == 60
    assert not any("exact target" in question for question in draft.questions)
