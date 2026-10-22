"""Export planner review selections as separate, explicitly weak preference events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DATASET_SCHEMA = "osrd-ops-service-plan-review-events/v9"
REVIEW_RECORD_SCHEMA = "osrd-service-plan-review-record/v1"
RANKING_METHOD = "deterministic_service_plan_objective_v1"
HOURLY_RANKING_METHOD = "deterministic_hourly_service_objective_v1"
COMPLETE_RANKING_METHOD = "bounded_joint_stop_route_phase_search_v1"
LEGACY_COMPLETE_RANKING_METHOD = "bounded_joint_phase_search_v1"
COMPLETE_PATRONAGE_RANKING_METHOD = "deterministic_complete_timetable_patronage_coverage_v1"
CHOICE_REASONS = frozenset(
    {
        "service_request_fit",
        "fewer_existing_changes",
        "lower_existing_service_impact",
        "preferred_route_or_timing",
        "other",
    }
)
DECLINE_REASONS = frozenset(
    {
        "service_needs_not_met",
        "too_many_existing_changes",
        "unwanted_route_or_timing",
        "other",
    }
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _candidate_features(
    candidate: dict[str, Any], request: dict[str, Any] | None = None
) -> dict[str, float | None]:
    patterns = candidate.get("patterns")
    if patterns is not None and not isinstance(patterns, list):
        raise ValueError("Candidate patterns must be a list")
    pattern_rows = patterns if isinstance(patterns, list) else []
    if not all(isinstance(pattern, dict) for pattern in pattern_rows):
        raise ValueError("Candidate patterns must contain objects")
    selected_stop_keys = [
        pattern.get("selected_optional_stop_keys", []) for pattern in pattern_rows
    ]
    if not all(
        isinstance(keys, list) and all(isinstance(key, str) for key in keys)
        for keys in selected_stop_keys
    ):
        raise ValueError("Selected optional stop keys must be lists of strings")
    selected_dwell_values: list[int] = []
    for pattern in pattern_rows:
        dwell_map = pattern.get("stop_dwell_seconds_by_key", {})
        if not isinstance(dwell_map, dict):
            raise ValueError("Selected stop dwell values must be a key-to-seconds object")
        for key, seconds in dwell_map.items():
            if (
                not isinstance(key, str)
                or not isinstance(seconds, int)
                or isinstance(seconds, bool)
                or seconds <= 0
            ):
                raise ValueError("Selected dwell values must be positive integer seconds")
            selected_dwell_values.append(seconds)
    existing_service_timings = candidate.get("existing_service_timings", [])
    if not isinstance(existing_service_timings, list) or not all(
        isinstance(timing, dict) for timing in existing_service_timings
    ):
        raise ValueError("Candidate existing_service_timings must be a list of objects")
    existing_runtime_deltas: list[float] = []
    for timing in existing_service_timings:
        delta = timing.get("travel_time_delta_seconds")
        if delta is None:
            continue
        if (
            not isinstance(delta, int | float)
            or isinstance(delta, bool)
            or not math.isfinite(delta)
        ):
            raise ValueError("Existing service runtime deltas must be finite numbers")
        existing_runtime_deltas.append(float(delta))
    existing_dwell_edits = candidate.get("existing_service_dwell_edits", [])
    if not isinstance(existing_dwell_edits, list) or not all(
        isinstance(edit, dict) for edit in existing_dwell_edits
    ):
        raise ValueError("Candidate existing_service_dwell_edits must be a list of objects")
    existing_dwell_deltas: list[float] = []
    existing_stop_removal_count = 0
    existing_stop_addition_count = 0
    for edit in existing_dwell_edits:
        before = edit.get("before_dwell_seconds")
        after = edit.get("after_dwell_seconds")
        if (
            not isinstance(edit.get("train_schedule_id"), str)
            or not isinstance(edit.get("waypoint_key"), str)
            or not isinstance(before, int)
            or isinstance(before, bool)
            or not isinstance(after, int)
            or isinstance(after, bool)
            or not 0 <= before <= 3600
            or not 0 <= after <= 3600
            or before == after
        ):
            raise ValueError("Existing-service dwell edits have invalid waypoint or duration data")
        existing_dwell_deltas.append(float(abs(after - before)))
        existing_stop_removal_count += int(before > 0 and after == 0)
        existing_stop_addition_count += int(before == 0 and after > 0)
    pattern_departure_counts = [
        pattern.get("departure_count") for pattern in pattern_rows
    ]
    pattern_phases = [
        pattern.get("departure_phase_seconds") for pattern in pattern_rows
    ]
    pattern_route_indices = [
        pattern.get("route_alternative_index", 0) for pattern in pattern_rows
    ]
    valid_pattern_counts = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in pattern_departure_counts
    )
    valid_pattern_phases = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in pattern_phases
    )
    valid_pattern_routes = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in pattern_route_indices
    )
    if pattern_rows and (
        not valid_pattern_counts or not valid_pattern_phases or not valid_pattern_routes
    ):
        raise ValueError(
            "Complete timetable patterns need nonnegative integer counts, phases, and route indices"
        )
    mean_pattern_phase = (
        sum(pattern_phases) / len(pattern_rows)
        if pattern_rows and valid_pattern_phases
        else None
    )
    mean_pattern_route = (
        sum(pattern_route_indices) / len(pattern_rows) if pattern_rows else None
    )
    requested_services = request.get("services") if isinstance(request, dict) else None
    if requested_services is not None and not isinstance(requested_services, list):
        raise ValueError("Complete timetable request services must be a list")
    requested_frequencies: list[int] = []
    requested_stop_count = 0
    requested_optional_stop_count = 0
    requested_dwell_choice_count = 0
    requested_dwell_choice_stop_count = 0
    existing_dwell_requests = (
        request.get("existing_service_dwell_options") if isinstance(request, dict) else None
    )
    requested_existing_dwell_option_count = 0
    if existing_dwell_requests is not None:
        if not isinstance(existing_dwell_requests, list) or not all(
            isinstance(item, dict) for item in existing_dwell_requests
        ):
            raise ValueError("Existing-service dwell options must be a list of objects")
        for item in existing_dwell_requests:
            options = item.get("dwell_seconds")
            if not isinstance(options, list) or any(
                not isinstance(seconds, int)
                or isinstance(seconds, bool)
                or not 0 <= seconds <= 3600
                for seconds in options
            ):
                raise ValueError(
                    "Existing-service dwell choices must be whole seconds in [0, 3600]"
                )
            requested_existing_dwell_option_count += len(options)
    if isinstance(requested_services, list):
        for service in requested_services:
            if not isinstance(service, dict):
                raise ValueError("Complete timetable request services must contain objects")
            frequency = service.get("frequency_seconds")
            path = service.get("path")
            if (
                not isinstance(frequency, int)
                or isinstance(frequency, bool)
                or frequency <= 0
                or not isinstance(path, list)
                or not all(isinstance(waypoint, dict) for waypoint in path)
            ):
                raise ValueError("Complete timetable request has invalid frequency or path data")
            requested_frequencies.append(frequency)
            requested_stop_count += sum(
                isinstance(waypoint.get("stop_for_seconds"), int)
                and not isinstance(waypoint.get("stop_for_seconds"), bool)
                and waypoint["stop_for_seconds"] > 0
                for waypoint in path
            )
            requested_optional_stop_count += sum(
                waypoint.get("optional_stop") is True for waypoint in path
            )
            for waypoint in path:
                dwell_options = waypoint.get("dwell_options_seconds", [])
                if not isinstance(dwell_options, list) or any(
                    not isinstance(seconds, int)
                    or isinstance(seconds, bool)
                    or seconds <= 0
                    for seconds in dwell_options
                ):
                    raise ValueError(
                        "Requested dwell alternatives must be positive integer seconds"
                    )
                requested_dwell_choice_count += len(dwell_options)
                requested_dwell_choice_stop_count += int(bool(dwell_options))
    proposed_schedules = candidate.get("proposed_schedules")
    if proposed_schedules is not None and not isinstance(proposed_schedules, list):
        raise ValueError("Candidate proposed_schedules must be a list")
    if isinstance(requested_services, list) and pattern_rows:
        if len(pattern_rows) != len(requested_services):
            raise ValueError("Complete candidate pattern count does not match its request")
        if (
            isinstance(proposed_schedules, list)
            and valid_pattern_counts
            and sum(pattern_departure_counts) != len(proposed_schedules)
        ):
            raise ValueError(
                "Complete candidate departure count does not match its pattern summaries"
            )
    departure_timings = candidate.get("departure_timings", [])
    if not isinstance(departure_timings, list):
        raise ValueError("Candidate departure_timings must be a list")
    travel_times: list[float] = []
    for row in departure_timings:
        if not isinstance(row, dict):
            raise ValueError("Candidate departure timing must be an object")
        travel_time = row.get("travel_time_seconds")
        if (
            not isinstance(travel_time, int | float)
            or isinstance(travel_time, bool)
            or not math.isfinite(travel_time)
            or travel_time < 0
        ):
            raise ValueError("Candidate travel time must be finite nonnegative seconds")
        travel_times.append(float(travel_time))
    if not travel_times and candidate.get("simulated_travel_time_seconds") is not None:
        travel_time = candidate["simulated_travel_time_seconds"]
        if (
            not isinstance(travel_time, int | float)
            or isinstance(travel_time, bool)
            or not math.isfinite(travel_time)
            or travel_time < 0
        ):
            raise ValueError(
                "Candidate simulated_travel_time_seconds must be finite and nonnegative"
            )
        travel_times.append(float(travel_time))

    def number(
        name: str,
        *,
        default: int | float | None = 0,
        optional: bool = False,
    ) -> float | None:
        value = candidate.get(name, default)
        if value is None and optional:
            return None
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Candidate {name} must be a finite number")
        return float(value)

    runtime_change = candidate.get("simulated_runtime_change_seconds")
    runtime_value = number(
        "simulated_runtime_change_seconds", optional=True
    ) if runtime_change is not None else None
    priority_score = (
        number("retiming_priority_score", optional=True)
        if candidate.get("retiming_priority_score") is not None
        else None
    )
    departure_phase = (
        number("departure_phase_seconds")
        if candidate.get("departure_phase_seconds") is not None
        else number("departure_offset_seconds")
        if candidate.get("departure_offset_seconds") is not None
        else mean_pattern_phase or 0.0
    )
    if priority_score is not None and priority_score < 0:
        raise ValueError("Candidate retiming priority score must be nonnegative")
    features = {
        "route_alternative_index": (
            number("route_alternative_index")
            if candidate.get("route_alternative_index") is not None
            else mean_pattern_route or 0.0
        ),
        "departure_phase_seconds": departure_phase,
        "changed_service_count": number("changed_service_count") or 0.0,
        "total_shift_seconds": number("total_shift_seconds") or 0.0,
        "new_conflict_count": number("new_conflict_count") or 0.0,
        "baseline_conflict_count": number("baseline_conflict_count") or 0.0,
    }
    if any(value < 0 for value in features.values()):
        raise ValueError("Candidate indices, counts, and shift must be nonnegative")
    coverage = candidate.get("patronage_coverage")
    if coverage is not None and not isinstance(coverage, dict):
        raise ValueError("Candidate patronage_coverage must be an object when present")
    coverage = coverage if isinstance(coverage, dict) else None

    def coverage_number(name: str) -> float | None:
        if coverage is None:
            return None
        value = coverage.get(name)
        if value is None:
            return None
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Candidate patronage coverage {name} must be finite")
        return float(value)

    matched_buckets = coverage_number("matched_bucket_count")
    total_buckets = coverage_number("total_bucket_count")
    active_days = coverage_number("active_service_day_count")
    source_service_count = number(
        "source_service_count", default=None, optional=True
    )
    compared_existing_service_count = number(
        "compared_existing_service_count", default=None, optional=True
    )
    affected_existing_service_count = number(
        "affected_existing_service_count", default=None, optional=True
    )
    if affected_existing_service_count is None and existing_service_timings:
        affected_existing_service_count = float(len(existing_service_timings))
    if (
        source_service_count is not None
        and compared_existing_service_count is not None
        and source_service_count > 0
    ):
        existing_service_effect_coverage = (
            compared_existing_service_count / source_service_count
        )
    elif candidate.get("existing_service_effects_status") == "complete":
        existing_service_effect_coverage = 1.0
    else:
        existing_service_effect_coverage = None
    effects_are_complete = candidate.get("existing_service_effects_status") == "complete"
    return {
        **features,
        "simulated_runtime_change_seconds": runtime_value,
        "retiming_priority_score": priority_score,
        "mean_inserted_travel_time_seconds": (
            sum(travel_times) / len(travel_times) if travel_times else None
        ),
        "patronage_boarding_coverage": coverage_number("boarding_coverage"),
        "patronage_alighting_coverage": coverage_number("alighting_coverage"),
        "patronage_combined_event_coverage": coverage_number("combined_event_coverage"),
        "patronage_coverage_available": 1.0 if coverage is not None else 0.0,
        "patronage_matched_bucket_fraction": (
            matched_buckets / total_buckets
            if matched_buckets is not None and total_buckets
            else None
        ),
        "patronage_unmatched_call_count": coverage_number("unmatched_call_count"),
        "patronage_unmapped_station_call_count": coverage_number("unmapped_station_call_count"),
        "patronage_active_service_day_count": active_days,
        "service_pattern_count": float(len(pattern_rows)) if pattern_rows else None,
        "proposed_departure_count": (
            float(len(proposed_schedules))
            if isinstance(proposed_schedules, list)
            else None
        ),
        "mean_pattern_departure_count": (
            sum(float(value) for value in pattern_departure_counts) / len(pattern_rows)
            if pattern_rows and valid_pattern_counts
            else None
        ),
        "pattern_phase_span_seconds": (
            float(max(pattern_phases) - min(pattern_phases))
            if pattern_rows and valid_pattern_phases
            else None
        ),
        "distinct_pattern_phase_count": (
            float(len(set(pattern_phases)))
            if pattern_rows and valid_pattern_phases
            else None
        ),
        "distinct_route_alternative_count": (
            float(len(set(pattern_route_indices))) if pattern_rows else None
        ),
        "mean_requested_frequency_seconds": (
            sum(requested_frequencies) / len(requested_frequencies)
            if requested_frequencies
            else None
        ),
        "requested_scheduled_stop_count": (
            float(requested_stop_count) if requested_services is not None else None
        ),
        "requested_optional_stop_count": (
            float(requested_optional_stop_count) if requested_services is not None else None
        ),
        "selected_optional_stop_count": (
            float(sum(len(keys) for keys in selected_stop_keys))
            if pattern_rows
            else None
        ),
        "requested_dwell_choice_count": (
            float(requested_dwell_choice_count) if requested_services is not None else None
        ),
        "requested_dwell_choice_stop_count": (
            float(requested_dwell_choice_stop_count) if requested_services is not None else None
        ),
        "mean_selected_stop_dwell_seconds": (
            sum(selected_dwell_values) / len(selected_dwell_values)
            if selected_dwell_values
            else None
        ),
        "max_selected_stop_dwell_seconds": (
            float(max(selected_dwell_values)) if selected_dwell_values else None
        ),
        "requested_existing_dwell_call_count": (
            float(len(existing_dwell_requests)) if existing_dwell_requests is not None else None
        ),
        "requested_existing_dwell_option_count": (
            float(requested_existing_dwell_option_count)
            if existing_dwell_requests is not None
            else None
        ),
        "existing_service_dwell_edit_count": (
            float(len(existing_dwell_edits))
            if "existing_service_dwell_edits" in candidate
            else None
        ),
        "mean_existing_dwell_change_seconds": (
            sum(existing_dwell_deltas) / len(existing_dwell_deltas)
            if existing_dwell_deltas
            else None
        ),
        "total_existing_dwell_change_seconds": (
            sum(existing_dwell_deltas) if "existing_service_dwell_edits" in candidate else None
        ),
        "existing_service_stop_removal_count": (
            float(existing_stop_removal_count)
            if "existing_service_dwell_edits" in candidate
            else None
        ),
        "existing_service_stop_addition_count": (
            float(existing_stop_addition_count)
            if "existing_service_dwell_edits" in candidate
            else None
        ),
        "existing_service_effect_coverage": existing_service_effect_coverage,
        "affected_existing_service_count": affected_existing_service_count,
        "existing_service_effects_complete": (
            1.0 if effects_are_complete else 0.0
            if candidate.get("existing_service_effects_status") == "partial"
            else None
        ),
        "max_positive_existing_runtime_change_seconds": (
            max((delta for delta in existing_runtime_deltas if delta > 0), default=0.0)
            if existing_runtime_deltas or effects_are_complete
            else None
        ),
        "total_positive_existing_runtime_change_seconds": (
            sum(delta for delta in existing_runtime_deltas if delta > 0)
            if existing_runtime_deltas or effects_are_complete
            else None
        ),
        "total_negative_existing_runtime_change_seconds": (
            sum(delta for delta in existing_runtime_deltas if delta < 0)
            if existing_runtime_deltas or effects_are_complete
            else None
        ),
        "feasible": float(candidate["feasible"]),
    }


def _review_event(path: Path, raw: bytes) -> dict[str, Any]:
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: not a valid UTF-8 JSON review record") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}: review record must be a JSON object")
    if record.get("schema") != REVIEW_RECORD_SCHEMA:
        raise ValueError(f"{path}: unsupported service-plan review-record schema")
    review_protocol = record.get("review_protocol", "legacy_unblinded_or_unknown")
    if not isinstance(review_protocol, str) or review_protocol not in {
        "blind_planner_choice_v1",
        "standard_review_v1",
        "legacy_unblinded_or_unknown",
    }:
        raise ValueError(f"{path}: unsupported review protocol")
    ranking_method = record.get("ranking_method", "unspecified")
    if not isinstance(ranking_method, str) or ranking_method not in {
        RANKING_METHOD,
        HOURLY_RANKING_METHOD,
        COMPLETE_RANKING_METHOD,
        LEGACY_COMPLETE_RANKING_METHOD,
        COMPLETE_PATRONAGE_RANKING_METHOD,
        "unspecified",
    }:
        raise ValueError(f"{path}: unsupported ranking method")
    choice = record.get("planner_choice")
    if not isinstance(choice, dict):
        raise ValueError(f"{path}: missing planner choice")
    disposition = choice.get("disposition")
    selected_id = choice.get("candidate_id")
    choice_reason = choice.get("choice_reason")
    decline_reason = choice.get("decline_reason")
    candidates = record.get("candidate_set")
    if not isinstance(candidates, list):
        raise ValueError(f"{path}: candidate set is missing")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError(f"{path}: candidate set must contain objects")
    candidate_rows = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if not candidate_rows:
        raise ValueError(f"{path}: candidate set must not be empty")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidate_rows]
    if not all(isinstance(candidate_id, str) and candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"{path}: every candidate needs a non-empty string ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{path}: candidate IDs must be unique")
    display_order = record.get("display_order_candidate_ids", candidate_ids)
    if (
        not isinstance(display_order, list)
        or not all(isinstance(candidate_id, str) for candidate_id in display_order)
        or len(display_order) != len(set(display_order))
        or display_order != candidate_ids
    ):
        raise ValueError(f"{path}: display order must contain each candidate exactly once")
    research_model = record.get("research_model")
    if research_model is not None:
        if not isinstance(research_model, dict):
            raise ValueError(f"{path}: research_model must be an object")
        model_status = research_model.get("status")
        if not isinstance(model_status, str) or model_status not in {
            "not_configured",
            "not_applicable",
            "used",
            "unavailable",
        }:
            raise ValueError(f"{path}: research model status is unsupported")
        suggested_id = research_model.get("suggested_candidate_id")
        if suggested_id is not None and suggested_id not in candidate_ids:
            raise ValueError(f"{path}: model suggestion must identify a reviewed candidate")
        model_sha256 = research_model.get("model_sha256")
        if model_sha256 is not None and (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in model_sha256)
        ):
            raise ValueError(f"{path}: research model digest must be a lowercase SHA-256")
        note = research_model.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"{path}: research model note must be a string or null")
        if model_status == "used" and (
            suggested_id is None or model_sha256 is None
        ):
            raise ValueError(f"{path}: a used research model needs its suggestion and digest")
    if any(type(candidate.get("feasible")) is not bool for candidate in candidate_rows):
        raise ValueError(f"{path}: every candidate needs a boolean feasible value")
    for candidate in candidate_rows:
        model_rank = candidate.get("research_model_rank")
        model_risk = candidate.get("research_model_mean_conflict_risk")
        model_max_risk = candidate.get("research_model_max_conflict_risk")
        highest_risk_schedule_id = candidate.get("research_model_highest_risk_schedule_id")
        highest_risk_train_name = candidate.get("research_model_highest_risk_train_name")
        if model_rank is not None and (
            not isinstance(model_rank, int) or isinstance(model_rank, bool) or model_rank < 1
        ):
            raise ValueError(f"{path}: research model ranks must be positive integers")
        if model_risk is not None and (
            not isinstance(model_risk, int | float)
            or isinstance(model_risk, bool)
            or not math.isfinite(model_risk)
            or not 0 <= model_risk <= 1
        ):
            raise ValueError(f"{path}: research model risk scores must be finite values in [0, 1]")
        if (model_rank is None) != (model_risk is None):
            raise ValueError(f"{path}: model rank and model risk score must appear together")
        if model_max_risk is not None and (
            not isinstance(model_max_risk, int | float)
            or isinstance(model_max_risk, bool)
            or not math.isfinite(model_max_risk)
            or not 0 <= model_max_risk <= 1
            or model_rank is None
            or model_risk is not None and model_max_risk < model_risk
        ):
            raise ValueError(f"{path}: maximum model risk score must be finite and in [0, 1]")
        if (model_max_risk is None) != (highest_risk_schedule_id is None):
            raise ValueError(
                f"{path}: maximum model risk and highest-risk schedule ID must appear together"
            )
        if highest_risk_schedule_id is not None and (
            not isinstance(highest_risk_schedule_id, str) or not highest_risk_schedule_id
        ):
            raise ValueError(f"{path}: highest-risk schedule ID must be a non-empty string")
        if highest_risk_train_name is not None and (
            not isinstance(highest_risk_train_name, str) or not highest_risk_train_name
        ):
            raise ValueError(f"{path}: highest-risk train name must be a non-empty string")
        if highest_risk_train_name is not None and highest_risk_schedule_id is None:
            raise ValueError(f"{path}: highest-risk train name requires its schedule ID")
    for candidate in candidate_rows:
        validation = candidate.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"{path}: every candidate needs its OSRD validation result")
        pathfinding_valid = validation.get("pathfinding_valid")
        simulation_valid = validation.get("simulation_valid")
        conflict_count = validation.get("conflict_count")
        new_conflict_count = candidate.get("new_conflict_count")
        failure_reasons = validation.get("failure_reasons")
        if type(pathfinding_valid) is not bool or type(simulation_valid) is not bool:
            raise ValueError(
                f"{path}: candidate pathfinding and simulation results must be boolean"
            )
        if (
            not isinstance(conflict_count, int)
            or isinstance(conflict_count, bool)
            or conflict_count < 0
            or not isinstance(new_conflict_count, int)
            or isinstance(new_conflict_count, bool)
            or not 0 <= new_conflict_count <= conflict_count
        ):
            raise ValueError(f"{path}: candidate conflict counts are invalid")
        if not isinstance(failure_reasons, list) or not all(
            isinstance(reason, str) for reason in failure_reasons
        ):
            raise ValueError(f"{path}: candidate validation reasons must be strings")
        expected_feasible = (
            pathfinding_valid
            and simulation_valid
            and new_conflict_count == 0
            and not any("cleanup failed" in reason.lower() for reason in failure_reasons)
        )
        if candidate["feasible"] is not expected_feasible:
            raise ValueError(f"{path}: candidate feasible flag disagrees with OSRD validation")
    feasible_ids = [
        candidate["candidate_id"]
        for candidate in candidate_rows
        if candidate.get("feasible") is True
    ]
    model_scored_candidates = [
        candidate
        for candidate in candidate_rows
        if candidate.get("research_model_rank") is not None
    ]
    if research_model is not None:
        if model_status == "used":
            expected_ranks = list(range(1, len(model_scored_candidates) + 1))
            actual_ranks = sorted(
                candidate["research_model_rank"] for candidate in model_scored_candidates
            )
            suggested = next(
                (
                    candidate
                    for candidate in model_scored_candidates
                    if candidate["candidate_id"] == research_model["suggested_candidate_id"]
                ),
                None,
            )
            if (
                set(candidate["candidate_id"] for candidate in model_scored_candidates)
                != set(feasible_ids)
                or actual_ranks != expected_ranks
                or len(model_scored_candidates) < 2
                or suggested is None
                or suggested["research_model_rank"] != 1
            ):
                raise ValueError(f"{path}: model ranks and suggested candidate are inconsistent")
        elif model_scored_candidates:
            raise ValueError(f"{path}: candidate model scores require a used research model")
    plan_counts = record.get("plan_counts")
    if not isinstance(plan_counts, dict):
        raise ValueError(f"{path}: plan counts are missing")
    counts = {name: plan_counts.get(name) for name in ("generated", "evaluated", "feasible")}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise ValueError(f"{path}: plan counts must be nonnegative integers")
    if (
        counts["generated"] < counts["evaluated"]
        or counts["evaluated"] != len(candidate_rows)
        or counts["feasible"] != len(feasible_ids)
    ):
        raise ValueError(f"{path}: plan counts do not match the candidate set")
    if disposition == "chosen_for_review":
        if not isinstance(selected_id, str):
            raise ValueError(f"{path}: selected candidate ID is missing")
        if decline_reason is not None:
            raise ValueError(f"{path}: a chosen option cannot include a decline reason")
        if choice_reason is not None and (
            not isinstance(choice_reason, str) or choice_reason not in CHOICE_REASONS
        ):
            raise ValueError(f"{path}: planner choice has an unsupported choice_reason")
        chosen = [
            candidate
            for candidate in candidate_rows
            if candidate.get("candidate_id") == selected_id
        ]
        if len(chosen) != 1 or chosen[0].get("feasible") is not True:
            raise ValueError(f"{path}: choice must identify one feasible candidate")
        if record.get("selected_candidate") != chosen[0]:
            raise ValueError(f"{path}: selected_candidate does not match the candidate set")
        choice_set = [
            candidate_id for candidate_id in display_order if candidate_id in feasible_ids
        ]
        event = "chosen_for_review"
    elif disposition == "no_feasible_option":
        if selected_id is not None or record.get("selected_candidate") is not None:
            raise ValueError(f"{path}: no-feasible-option record must not select a candidate")
        if choice_reason is not None or decline_reason is not None:
            raise ValueError(f"{path}: no-feasible-option record cannot include a reason")
        if feasible_ids:
            raise ValueError(f"{path}: every candidate must fail OSRD validation")
        choice_set = []
        event = "no_feasible_option"
    elif disposition == "no_suitable_option":
        if selected_id is not None or record.get("selected_candidate") is not None:
            raise ValueError(f"{path}: no-suitable-option record must not select a candidate")
        if choice_reason is not None:
            raise ValueError(f"{path}: no-suitable-option record cannot include a choice reason")
        if not feasible_ids:
            raise ValueError(f"{path}: no-suitable-option record requires feasible candidates")
        if decline_reason is not None and (
            not isinstance(decline_reason, str) or decline_reason not in DECLINE_REASONS
        ):
            raise ValueError(f"{path}: planner decline has an unsupported decline_reason")
        choice_set = [
            candidate_id for candidate_id in display_order if candidate_id in feasible_ids
        ]
        event = "no_suitable_option"
    else:
        raise ValueError(f"{path}: unsupported planner choice disposition")
    request = record.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"{path}: original planning request is missing")
    if "service" in request and "hourly_service" in request:
        raise ValueError(f"{path}: review request must describe one timetable mode")
    complete_request = isinstance(request.get("services"), list) and bool(request["services"])
    timetable_mode = (
        "complete_calendar"
        if complete_request
        else "hourly"
        if "hourly_service" in request
        else "calendar"
    )
    if complete_request and ("service" in request or "hourly_service" in request):
        raise ValueError(
            f"{path}: complete timetable review must not include a single service request"
        )
    if timetable_mode == "hourly" and ranking_method != HOURLY_RANKING_METHOD:
        raise ValueError(f"{path}: hourly review has mismatched ranking provenance")
    if timetable_mode == "calendar" and ranking_method != RANKING_METHOD:
        raise ValueError(f"{path}: calendar review has mismatched ranking provenance")
    if timetable_mode == "complete_calendar" and ranking_method not in {
        COMPLETE_RANKING_METHOD,
        LEGACY_COMPLETE_RANKING_METHOD,
        COMPLETE_PATRONAGE_RANKING_METHOD,
    }:
        raise ValueError(
            f"{path}: complete timetable review has mismatched ranking provenance"
        )
    if timetable_mode != "complete_calendar" and ranking_method in {
        COMPLETE_RANKING_METHOD,
        LEGACY_COMPLETE_RANKING_METHOD,
        COMPLETE_PATRONAGE_RANKING_METHOD,
    }:
        raise ValueError(
            f"{path}: complete-timetable ranking provenance requires a complete request"
        )
    service = (
        request["services"][0]
        if complete_request
        else request.get("service") or request.get("hourly_service")
    )
    snapshot = service.get("snapshot") if isinstance(service, dict) else None
    if not isinstance(snapshot, dict) or not all(
        snapshot.get(key)
        for key in (
            "infra_id",
            "infra_version",
            "timetable_id",
            "timetable_revision",
            "osrd_version",
        )
    ):
        raise ValueError(f"{path}: source infrastructure binding is missing")
    if record.get("snapshot") != snapshot:
        raise ValueError(f"{path}: exported snapshot does not match the planning request")
    study_group = record.get("study_infrastructure_group")
    if study_group is not None and (
        not isinstance(study_group, str) or not study_group.strip()
    ):
        raise ValueError(f"{path}: study infrastructure group must be a non-empty string")
    if review_protocol == "blind_planner_choice_v1" and (
        not isinstance(study_group, str) or not study_group.strip()
    ):
        raise ValueError(f"{path}: blind review must include a study infrastructure group")
    application = record.get("application")
    if event != "chosen_for_review" and application is not None:
        raise ValueError(f"{path}: only a chosen feasible option can have a draft application")
    if application is not None:
        if (
            not isinstance(application, dict)
            or application.get("disposition") != "created_as_draft"
            or not isinstance(application.get("draft_scenario_id"), str)
            or not application.get("draft_scenario_id")
            or not isinstance(application.get("applied_at"), str)
            or not application.get("applied_at")
        ):
            raise ValueError(f"{path}: draft application record is malformed")
        try:
            applied_at = datetime.fromisoformat(application["applied_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{path}: draft application timestamp must be ISO 8601") from error
        if applied_at.tzinfo is None or applied_at.utcoffset() is None:
            raise ValueError(f"{path}: draft application timestamp must include a timezone")
    request_id = request.get("request_id") if complete_request else service.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"{path}: service request ID is missing")
    suggestion_selected_for_review = (
        selected_id == research_model.get("suggested_candidate_id")
        if event == "chosen_for_review"
        and isinstance(research_model, dict)
        and research_model.get("status") == "used"
        else None
    )
    return {
        "schema": DATASET_SCHEMA,
        "source_file": path.name,
        "source_sha256": _digest(raw),
        "request_id": request_id,
        "infrastructure_group": study_group or str(snapshot["infra_id"]),
        "study_infrastructure_group": study_group,
        "source_infra_id": str(snapshot["infra_id"]),
        "source_timetable_group": f"{snapshot['infra_id']}/{snapshot['timetable_id']}",
        "snapshot": snapshot,
        "request": request,
        "timetable_mode": timetable_mode,
        "ranking_method": ranking_method,
        "review_protocol": review_protocol,
        "display_order_candidate_ids": display_order,
        "research_model": research_model,
        "research_model_suggestion_selected_for_review": suggestion_selected_for_review,
        "event": event,
        "chosen_candidate_id": selected_id,
        "choice_reason": choice_reason,
        "decline_reason": decline_reason,
        "choice_set_candidate_ids": choice_set,
        "application": application,
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "feasible": bool(candidate.get("feasible")),
                "features": _candidate_features(candidate, request),
                "explanation": candidate.get("explanation", ""),
                "research_model_rank": candidate.get("research_model_rank"),
                "research_model_mean_conflict_risk": candidate.get(
                    "research_model_mean_conflict_risk"
                ),
                "research_model_max_conflict_risk": candidate.get(
                    "research_model_max_conflict_risk"
                ),
                "research_model_highest_risk_schedule_id": candidate.get(
                    "research_model_highest_risk_schedule_id"
                ),
                "research_model_highest_risk_train_name": candidate.get(
                    "research_model_highest_risk_train_name"
                ),
            }
            for candidate in candidate_rows
        ],
        "unselected_candidate_interpretation": (
            "unknown; unselected does not mean rejected"
            if event == "chosen_for_review"
            else (
                "planner declined the feasible option set; individual alternatives remain "
                "unclassified"
                if event == "no_suitable_option"
                else "not applicable; no candidate passed OSRD validation"
            )
        ),
    }


def export_reviews(
    review_files: list[Path], output_dir: Path, *, review_db: Path | None = None
) -> dict[str, Any]:
    sources = [(path, path.read_bytes()) for path in review_files]
    if review_db is not None:
        try:
            with sqlite3.connect(review_db) as connection:
                saved = connection.execute(
                    """SELECT review_id, record_json, review_sha256
                       FROM service_plan_reviews
                       ORDER BY created_at, review_id"""
                ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table: service_plan_reviews" in str(error).lower():
                saved = []
            else:
                raise ValueError(
                    f"Could not read saved planner reviews from {review_db}: {error}"
                ) from error
        except sqlite3.Error as error:
            raise ValueError(
                f"Could not read saved planner reviews from {review_db}: {error}"
            ) from error
        for review_id, record_json, expected_hash in saved:
            raw = record_json.encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"Saved planner review {review_id} failed digest verification")
            sources.append((Path(f"{review_db}#review:{review_id}"), raw))
    if not sources and review_db is None:
        raise ValueError("Provide review JSON files or a saved-review SQLite database")
    rows = []
    seen_requests: set[str] = set()
    source_hash = hashlib.sha256()
    for path, raw in sources:
        row = _review_event(path, raw)
        request_id = row["request_id"]
        if request_id in seen_requests:
            raise ValueError(f"Duplicate request ID in review records: {request_id}")
        seen_requests.add(request_id)
        rows.append(row)
    rows.sort(key=lambda row: row["request_id"])
    for row in rows:
        source_hash.update(f"{row['request_id']}:{row['source_sha256']}\n".encode())
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        group = row["infrastructure_group"]
        mode = row["timetable_mode"]
        group_coverage = coverage.setdefault(group, {})
        mode_coverage = group_coverage.setdefault(
            mode,
            {
                "review_record_count": 0,
                "blind_review_record_count": 0,
                "blind_multi_option_choice_count": 0,
                "no_feasible_option_count": 0,
                "no_suitable_option_count": 0,
            },
        )
        mode_coverage["review_record_count"] += 1
        if row["review_protocol"] == "blind_planner_choice_v1":
            mode_coverage["blind_review_record_count"] += 1
            if (
                row["event"] == "chosen_for_review"
                and len(row["choice_set_candidate_ids"]) > 1
            ):
                mode_coverage["blind_multi_option_choice_count"] += 1
        if row["event"] == "no_feasible_option":
            mode_coverage["no_feasible_option_count"] += 1
        elif row["event"] == "no_suitable_option":
            mode_coverage["no_suitable_option_count"] += 1
    blind_choice_groups = {
        row["infrastructure_group"]
        for row in rows
        if row["review_protocol"] == "blind_planner_choice_v1"
        and row["event"] == "chosen_for_review"
        and len(row["choice_set_candidate_ids"]) > 1
    }
    coverage_sufficient = len(blind_choice_groups) >= 8 and all(
        coverage.get(group, {}).get(mode, {}).get("blind_multi_option_choice_count", 0) >= 30
        for group in blind_choice_groups
        for mode in ("calendar", "hourly")
    )
    model_choice_rows = [
        row
        for row in rows
        if row["research_model_suggestion_selected_for_review"] is not None
    ]
    model_suggestion_matches = sum(
        row["research_model_suggestion_selected_for_review"] is True
        for row in model_choice_rows
    )
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "review-events.jsonl"
    dataset_path.write_bytes(content)
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_file": dataset_path.name,
        "dataset_sha256": _digest(content),
        "source_record_count": len(rows),
        "source_records_sha256": source_hash.hexdigest(),
        "infrastructure_groups": sorted({row["infrastructure_group"] for row in rows}),
        "coverage_by_infrastructure_and_mode": coverage,
        "blind_multi_option_infrastructure_group_count": len(blind_choice_groups),
        "minimum_blind_multi_option_choices_per_group_and_mode": 30,
        "coverage_sufficient_for_preference_fit": coverage_sufficient,
        "source_timetable_group_count": len(
            {row["source_timetable_group"] for row in rows}
        ),
        "choice_event_count": sum(row["event"] == "chosen_for_review" for row in rows),
        "no_feasible_option_event_count": sum(
            row["event"] == "no_feasible_option" for row in rows
        ),
        "no_suitable_option_event_count": sum(
            row["event"] == "no_suitable_option" for row in rows
        ),
        "multi_option_choice_event_count": sum(
            row["event"] == "chosen_for_review"
            and len(row["choice_set_candidate_ids"]) > 1
            for row in rows
        ),
        "reasoned_choice_event_count": sum(row["choice_reason"] is not None for row in rows),
        "reasoned_decline_event_count": sum(row["decline_reason"] is not None for row in rows),
        "choice_reason_counts": {
            reason: sum(row["choice_reason"] == reason for row in rows)
            for reason in sorted(CHOICE_REASONS)
            if any(row["choice_reason"] == reason for row in rows)
        },
        "decline_reason_counts": {
            reason: sum(row["decline_reason"] == reason for row in rows)
            for reason in sorted(DECLINE_REASONS)
            if any(row["decline_reason"] == reason for row in rows)
        },
        "created_as_draft_count": sum(row["application"] is not None for row in rows),
        "ranking_method_counts": {
            method: sum(row["ranking_method"] == method for row in rows)
            for method in sorted({row["ranking_method"] for row in rows})
        },
        "timetable_mode_counts": {
            mode: sum(row["timetable_mode"] == mode for row in rows)
            for mode in ("calendar", "hourly")
            if any(row["timetable_mode"] == mode for row in rows)
        },
        "research_model_status_counts": {
            status: sum(
                isinstance(row["research_model"], dict)
                and row["research_model"].get("status") == status
                for row in rows
            )
            for status in ("not_configured", "not_applicable", "used", "unavailable")
            if any(
                isinstance(row["research_model"], dict)
                and row["research_model"].get("status") == status
                for row in rows
            )
        },
        "research_model_choice_comparison_count": len(model_choice_rows),
        "research_model_suggestion_selected_for_review_count": model_suggestion_matches,
        "research_model_suggestion_review_choice_match_rate": (
            model_suggestion_matches / len(model_choice_rows) if model_choice_rows else None
        ),
        "candidate_option_count": sum(len(row["candidates"]) for row in rows),
        "selection_semantics": (
            "One feasible option was chosen for review from the listed feasible choice set. "
            "An optional categorical reason records the planner's stated rationale, not an "
            "outcome; when absent, no inference is made about why the option was selected. "
            "A no_feasible_option event records that no candidate passed OSRD validation; "
            "it is not a preference label. A no_suitable_option event records planner "
            "rejection of the feasible option set; individual alternatives remain unclassified. "
            "An application field, when present, records creation of a draft scenario only, "
            "not publication or acceptance."
        ),
        "training_ready": False,
        "limitations": [
            "Unselected alternatives are not negative labels.",
            "A choice for review is not evidence of acceptance, service utility, or outcome.",
            (
                "Model suggestion alignment measures review choices only; it is not a measure "
                "of model quality or operational acceptance."
            ),
            "No-feasible-option events do not identify whether constraints should change.",
            "No-suitable-option events do not identify candidate-level rejection; the optional "
            "reason applies to the set as a whole.",
            "Split future evaluations by infrastructure and timetable context before fitting.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_records", nargs="*", type=Path)
    parser.add_argument(
        "--review-db",
        type=Path,
        help="Also import explicitly submitted records from the local planning journal.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export_reviews(args.review_records, args.output_dir, review_db=args.review_db)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
