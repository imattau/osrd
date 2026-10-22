import json
from pathlib import Path

from osrd_ops_intelligence.training.service_plan_preferences import _review_event
from osrd_ops_intelligence.training.train_service_plan_preferences import (
    TIMETABLE_MODES,
    _choice_sets,
)


def test_complete_calendar_reviews_export_separate_pattern_features() -> None:
    snapshot = {
        "infra_id": "infra-a",
        "infra_version": "version-1",
        "timetable_id": "timetable-a",
        "timetable_revision": "sha256:revision",
        "osrd_version": "test",
    }
    candidates = [
        {
            "candidate_id": "complete-a",
            "rank": 1,
            "feasible": True,
            "patterns": [
                {
                    "train_name": "A",
                    "departure_count": 4,
                    "departure_phase_seconds": 60,
                    "route_alternative_index": 1,
                    "selected_optional_stop_keys": ["stop"],
                    "stop_dwell_seconds_by_key": {"stop": 60},
                },
                {
                    "train_name": "B",
                    "departure_count": 2,
                    "departure_phase_seconds": 180,
                    "route_alternative_index": 0,
                },
            ],
            "proposed_schedules": [{"train_name": "A"}] * 6,
            "departure_timings": [],
            "existing_service_timings": [
                {"travel_time_delta_seconds": 20},
                {"travel_time_delta_seconds": -5},
            ],
            "existing_service_dwell_edits": [
                {
                    "train_schedule_id": "existing-1",
                    "waypoint_key": "stop",
                    "before_dwell_seconds": 45,
                    "after_dwell_seconds": 60,
                }
            ],
            "existing_service_effects_status": "partial",
            "source_service_count": 4,
            "compared_existing_service_count": 3,
            "affected_existing_service_count": 2,
            "new_conflict_count": 0,
            "baseline_conflict_count": 0,
            "changed_service_count": 0,
            "total_shift_seconds": 0,
            "validation": {
                "pathfinding_valid": True,
                "simulation_valid": True,
                "conflict_count": 0,
                "failure_reasons": [],
            },
        },
        {
            "candidate_id": "complete-b",
            "rank": 2,
            "feasible": True,
            "patterns": [
                {"train_name": "A", "departure_count": 4, "departure_phase_seconds": 0},
                {
                    "train_name": "B",
                    "departure_count": 2,
                    "departure_phase_seconds": 240,
                    "stop_dwell_seconds_by_key": {"stop": 30},
                },
            ],
            "proposed_schedules": [{"train_name": "A"}] * 6,
            "departure_timings": [],
            "existing_service_timings": [],
            "existing_service_dwell_edits": [],
            "existing_service_effects_status": "complete",
            "source_service_count": 3,
            "compared_existing_service_count": 3,
            "affected_existing_service_count": 0,
            "new_conflict_count": 0,
            "baseline_conflict_count": 0,
            "changed_service_count": 0,
            "total_shift_seconds": 0,
            "validation": {
                "pathfinding_valid": True,
                "simulation_valid": True,
                "conflict_count": 0,
                "failure_reasons": [],
            },
        },
    ]
    record = {
        "schema": "osrd-service-plan-review-record/v1",
        "review_protocol": "blind_planner_choice_v1",
        "study_infrastructure_group": "independent-network-a",
        "ranking_method": "bounded_joint_phase_search_v1",
        "research_model": {"status": "not_applicable"},
        "request": {
            "request_id": "request-1",
            "existing_service_dwell_options": [
                {
                    "train_schedule_id": "existing-1",
                    "waypoint_key": "stop",
                    "dwell_seconds": [0, 60, 120],
                }
            ],
            "services": [
                {
                    "request_id": "request-1",
                    "snapshot": snapshot,
                    "frequency_seconds": 1800,
                    "path": [
                        {"key": "origin", "stop_for_seconds": None},
                        {
                            "key": "stop",
                            "stop_for_seconds": 45,
                            "dwell_options_seconds": [30, 60, 90],
                            "optional_stop": True,
                        },
                    ],
                },
                {
                    "request_id": "request-1",
                    "snapshot": snapshot,
                    "frequency_seconds": 3600,
                    "path": [{"key": "origin"}, {"key": "destination"}],
                },
            ],
        },
        "snapshot": snapshot,
        "plan_counts": {"generated": 2, "evaluated": 2, "feasible": 2},
        "candidate_set": candidates,
        "display_order_candidate_ids": ["complete-a", "complete-b"],
        "planner_choice": {"candidate_id": "complete-a", "disposition": "chosen_for_review"},
        "selected_candidate": candidates[0],
    }

    event = _review_event(Path("complete-review.json"), json.dumps(record).encode())

    assert "complete_calendar" in TIMETABLE_MODES
    assert event["timetable_mode"] == "complete_calendar"
    features = event["candidates"][0]["features"]
    assert features["service_pattern_count"] == 2.0
    assert features["proposed_departure_count"] == 6.0
    assert features["mean_pattern_departure_count"] == 3.0
    assert features["departure_phase_seconds"] == 120.0
    assert features["pattern_phase_span_seconds"] == 120.0
    assert features["distinct_pattern_phase_count"] == 2.0
    assert features["route_alternative_index"] == 0.5
    assert features["distinct_route_alternative_count"] == 2.0
    assert features["mean_requested_frequency_seconds"] == 2700.0
    assert features["requested_scheduled_stop_count"] == 1.0
    assert features["requested_optional_stop_count"] == 1.0
    assert features["selected_optional_stop_count"] == 1.0
    assert features["requested_dwell_choice_count"] == 3.0
    assert features["requested_dwell_choice_stop_count"] == 1.0
    assert features["mean_selected_stop_dwell_seconds"] == 60.0
    assert features["max_selected_stop_dwell_seconds"] == 60.0
    assert features["requested_existing_dwell_call_count"] == 1.0
    assert features["requested_existing_dwell_option_count"] == 3.0
    assert features["existing_service_dwell_edit_count"] == 1.0
    assert features["mean_existing_dwell_change_seconds"] == 15.0
    assert features["total_existing_dwell_change_seconds"] == 15.0
    assert features["existing_service_stop_removal_count"] == 0.0
    assert features["existing_service_stop_addition_count"] == 0.0
    assert features["existing_service_effect_coverage"] == 0.75
    assert features["affected_existing_service_count"] == 2.0
    assert features["existing_service_effects_complete"] == 0.0
    assert features["max_positive_existing_runtime_change_seconds"] == 20.0
    assert features["total_positive_existing_runtime_change_seconds"] == 20.0
    assert features["total_negative_existing_runtime_change_seconds"] == -5.0
    no_effect_features = event["candidates"][1]["features"]
    assert no_effect_features["existing_service_effect_coverage"] == 1.0
    assert no_effect_features["affected_existing_service_count"] == 0.0
    assert no_effect_features["existing_service_effects_complete"] == 1.0
    assert no_effect_features["max_positive_existing_runtime_change_seconds"] == 0.0
    choice_sets = _choice_sets([event])
    assert len(choice_sets) == 1
    assert choice_sets[0]["timetable_mode"] == "complete_calendar"
    assert choice_sets[0]["features"][0]["service_pattern_count"] == 2.0
