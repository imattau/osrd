"""Fit an offline listwise ranker from explicitly chosen planner review options.

The target is a planner's choice for review within one displayed feasible set. It is
weak preference evidence, not operational acceptance or an outcome label. Artifacts
from this trainer are research-only and are not loaded by the live planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

EVENT_SCHEMAS = {
    "osrd-ops-service-plan-review-events/v3",
    "osrd-ops-service-plan-review-events/v4",
    "osrd-ops-service-plan-review-events/v5",
    "osrd-ops-service-plan-review-events/v6",
    "osrd-ops-service-plan-review-events/v7",
    "osrd-ops-service-plan-review-events/v8",
    "osrd-ops-service-plan-review-events/v9",
}
MODEL_SCHEMA = "osrd-ops-service-plan-preference-ranker/v7"
TIMETABLE_MODES = ("calendar", "hourly", "complete_calendar")
FEATURES = (
    "route_alternative_index",
    "departure_phase_seconds",
    "changed_service_count",
    "total_shift_seconds",
    "new_conflict_count",
    "baseline_conflict_count",
    "simulated_runtime_change_seconds",
    "retiming_priority_score",
    "mean_inserted_travel_time_seconds",
    "patronage_boarding_coverage",
    "patronage_alighting_coverage",
    "patronage_combined_event_coverage",
    "patronage_coverage_available",
    "patronage_matched_bucket_fraction",
    "patronage_unmatched_call_count",
    "patronage_unmapped_station_call_count",
    "patronage_active_service_day_count",
    "service_pattern_count",
    "proposed_departure_count",
    "mean_pattern_departure_count",
    "pattern_phase_span_seconds",
    "distinct_pattern_phase_count",
    "distinct_route_alternative_count",
    "mean_requested_frequency_seconds",
    "requested_scheduled_stop_count",
    "requested_optional_stop_count",
    "selected_optional_stop_count",
    "requested_dwell_choice_count",
    "requested_dwell_choice_stop_count",
    "mean_selected_stop_dwell_seconds",
    "max_selected_stop_dwell_seconds",
    "requested_existing_dwell_call_count",
    "requested_existing_dwell_option_count",
    "existing_service_dwell_edit_count",
    "mean_existing_dwell_change_seconds",
    "total_existing_dwell_change_seconds",
    "existing_service_stop_removal_count",
    "existing_service_stop_addition_count",
    "existing_service_effect_coverage",
    "affected_existing_service_count",
    "existing_service_effects_complete",
    "max_positive_existing_runtime_change_seconds",
    "total_positive_existing_runtime_change_seconds",
    "total_negative_existing_runtime_change_seconds",
)
# A regularized quadratic listwise model can learn trade-offs such as patronage
# coverage versus retiming cost. Its interaction basis remains offline/research-only.
MODEL_FEATURES = FEATURES + tuple(
    f"{left}*{right}"
    for index, left in enumerate(FEATURES)
    for right in FEATURES[index:]
)
MIN_INFRASTRUCTURE_GROUPS = 8
MIN_EVENTS_PER_INFRASTRUCTURE = 30
MIN_TRAIN_EVENTS = 12
MIN_VALIDATION_EVENTS = 3
MIN_TEST_EVENTS = 5
METRIC_PROTOCOL = {
    "schema": "osrd-service-plan-preference-metrics/v1",
    "primary": "top1_accuracy",
    "secondary": ["mean_reciprocal_rank", "listwise_log_loss"],
    "references": ["displayed_deterministic_order", "uniform_random_choice"],
    "reported_by_infrastructure": True,
    "promotion_decision": "none; preference fit does not establish operational utility",
}


def _read_events(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON event") from error
        if not isinstance(row, dict) or row.get("schema") not in EVENT_SCHEMAS:
            raise ValueError(f"{path}:{line_number}: unsupported review event schema")
        rows.append(row)
    return rows, digest


def _choice_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    for row in rows:
        if (
            row.get("event") != "chosen_for_review"
            or row.get("review_protocol") != "blind_planner_choice_v1"
        ):
            continue
        if (
            not isinstance(row.get("study_infrastructure_group"), str)
            or not row["study_infrastructure_group"].strip()
            or row.get("infrastructure_group") != row.get("study_infrastructure_group")
        ):
            raise ValueError("Blind choice event lacks its explicit study infrastructure group")
        research_model = row.get("research_model")
        if (
            isinstance(research_model, dict)
            and research_model.get("status") == "used"
        ):
            raise ValueError("Blind review choice set contains an exposed model suggestion")
        request_id = row.get("request_id")
        group = row.get("infrastructure_group")
        timetable_mode = row.get("timetable_mode")
        chosen_id = row.get("chosen_candidate_id")
        choices = row.get("choice_set_candidate_ids")
        candidates = row.get("candidates")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(group, str)
            or not group
            or timetable_mode not in TIMETABLE_MODES
            or not isinstance(chosen_id, str)
            or not isinstance(choices, list)
            or not all(isinstance(value, str) and value for value in choices)
            or len(choices) != len(set(choices))
            or not isinstance(candidates, list)
        ):
            raise ValueError("Review event has an invalid choice-set binding")
        if request_id in seen_requests:
            raise ValueError(f"Duplicate planner choice for request {request_id}")
        seen_requests.add(request_id)
        if len(choices) < 2:
            continue
        by_id = {
            candidate.get("candidate_id"): candidate
            for candidate in candidates
            if isinstance(candidate, dict)
        }
        if chosen_id not in choices or any(value not in by_id for value in choices):
            raise ValueError(f"Review event {request_id} has inconsistent candidates")
        choice_candidates = [by_id[value] for value in choices]
        if any(candidate.get("feasible") is not True for candidate in choice_candidates):
            raise ValueError(f"Review event {request_id} includes an infeasible choice")
        vectors: list[dict[str, float | None]] = []
        for candidate in choice_candidates:
            features = candidate.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"Review event {request_id} is missing candidate features")
            vector: dict[str, float | None] = {}
            for feature in FEATURES:
                value = features.get(feature)
                if value is None:
                    vector[feature] = None
                    continue
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Review event {request_id} has invalid {feature}")
                if feature.endswith("_coverage") and not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"Review event {request_id} has out-of-range {feature}")
                vector[feature] = float(value)
            vectors.append(vector)
        events.append(
            {
                "request_id": request_id,
                "infrastructure_group": group,
                "source_infra_id": row.get("source_infra_id"),
                "timetable_mode": timetable_mode,
                "candidate_ids": choices,
                "chosen_index": choices.index(chosen_id),
                "features": vectors,
            }
        )
    return events


def _group_split(events: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[event["infrastructure_group"]].append(event)
    group_names = sorted(groups)
    if len(group_names) < MIN_INFRASTRUCTURE_GROUPS:
        raise ValueError(
            f"Need at least {MIN_INFRASTRUCTURE_GROUPS} infrastructure groups with multi-option "
            f"choices; found {len(group_names)}"
        )
    sparse_groups = {
        group: len(group_events)
        for group, group_events in groups.items()
        if len(group_events) < MIN_EVENTS_PER_INFRASTRUCTURE
    }
    if sparse_groups:
        raise ValueError(
            "Need at least 30 blind multi-option choices per infrastructure; "
            f"sparse groups: {sparse_groups}"
        )
    random.Random(seed).shuffle(group_names)
    test_count = max(1, round(len(group_names) * 0.2))
    validation_count = max(1, round(len(group_names) * 0.2))
    test_groups = set(group_names[:test_count])
    validation_groups = set(group_names[test_count : test_count + validation_count])
    train_groups = set(group_names) - test_groups - validation_groups
    split = {
        "train": [event for event in events if event["infrastructure_group"] in train_groups],
        "validation": [
            event for event in events if event["infrastructure_group"] in validation_groups
        ],
        "test": [event for event in events if event["infrastructure_group"] in test_groups],
    }
    minimums = {
        "train": MIN_TRAIN_EVENTS,
        "validation": MIN_VALIDATION_EVENTS,
        "test": MIN_TEST_EVENTS,
    }
    for name, minimum in minimums.items():
        if len(split[name]) < minimum:
            raise ValueError(
                f"Need at least {minimum} multi-option choices in the {name} split; "
                f"found {len(split[name])}"
            )
    return split


def _validate_infrastructure_registry(
    path: Path, events: list[dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Verify study labels against a reviewed physical-network registry."""
    raw = path.read_bytes()
    try:
        registry = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Infrastructure registry is not valid JSON: {error}") from error
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != "osrd-service-plan-infrastructure-registry/v1"
        or not isinstance(registry.get("groups"), list)
    ):
        raise ValueError("Unsupported or malformed OSRD infrastructure registry")

    by_label: dict[str, dict[str, Any]] = {}
    physical_ids: dict[str, str] = {}
    for item in registry["groups"]:
        if not isinstance(item, dict):
            raise ValueError("Infrastructure registry groups must be objects")
        label = item.get("study_infrastructure_group")
        physical_id = item.get("physical_network_id")
        name = item.get("network_name")
        basis = item.get("independence_basis")
        reviewer = item.get("reviewed_by_role")
        reviewed_at = item.get("reviewed_at_utc")
        source_ids = item.get("source_infrastructure_ids")
        required_strings = (label, physical_id, name, basis, reviewer, reviewed_at)
        if not all(isinstance(value, str) and value.strip() for value in required_strings):
            raise ValueError(
                "Each registry group needs a label, physical identity, name, "
                "independence basis, reviewer role, and review time"
            )
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not all(isinstance(value, str) and value.strip() for value in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError(f"Registry group {label!r} needs unique source infrastructure IDs")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Registry group {label!r} has an invalid review timestamp") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"Registry group {label!r} review timestamp must include a timezone")
        if label in by_label:
            raise ValueError(f"Duplicate study infrastructure label {label!r}")
        canonical_id = physical_id.strip().casefold()
        if canonical_id in physical_ids:
            raise ValueError(
                f"Study groups {physical_ids[canonical_id]!r} and {label!r} "
                "identify the same physical network"
            )
        by_label[label] = item
        physical_ids[canonical_id] = label

    observed_labels = {event["infrastructure_group"] for event in events}
    if set(by_label) != observed_labels:
        raise ValueError(
            "Infrastructure registry labels must exactly match this mode's blind choice groups"
        )
    observed_sources: dict[str, set[str]] = defaultdict(set)
    for event in events:
        source_id = event.get("source_infra_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Blind choice event lacks its source OSRD infrastructure ID")
        observed_sources[event["infrastructure_group"]].add(source_id)
    for label, source_ids in observed_sources.items():
        if not source_ids.issubset(set(by_label[label]["source_infrastructure_ids"])):
            raise ValueError(
                f"Registry source infrastructure IDs do not cover study group {label!r}"
            )
    if len(by_label) < MIN_INFRASTRUCTURE_GROUPS:
        raise ValueError(
            f"Infrastructure registry must identify at least {MIN_INFRASTRUCTURE_GROUPS} "
            "independent physical networks"
        )
    return hashlib.sha256(raw).hexdigest(), by_label


def freeze_split(
    input_path: Path,
    output_path: Path,
    *,
    seed: int,
    timetable_mode: str,
    infrastructure_registry_path: Path,
    require_patronage: bool = False,
) -> dict[str, Any]:
    """Write the dataset-bound infrastructure assignment before model fitting."""
    rows, dataset_sha256 = _read_events(input_path)
    if timetable_mode not in TIMETABLE_MODES:
        raise ValueError(f"timetable_mode must be one of {', '.join(TIMETABLE_MODES)}")
    events = [
        event for event in _choice_sets(rows) if event["timetable_mode"] == timetable_mode
    ]
    if require_patronage and timetable_mode == "hourly":
        raise ValueError(
            "Patronage-specific fitting is unavailable for hourly timetables until "
            "weekday-aware hourly demand matching is implemented"
        )
    registry_sha256, registry_groups = _validate_infrastructure_registry(
        infrastructure_registry_path, events
    )
    split = _group_split(events, seed)
    patronage_choice_counts = {
        name: sum(
            bool(event["features"])
            and all(
                candidate["patronage_coverage_available"] == 1.0
                for candidate in event["features"]
            )
            for event in part
        )
        for name, part in split.items()
    }
    if require_patronage and any(count < 30 for count in patronage_choice_counts.values()):
        raise ValueError(
            "Patronage split requires at least 30 fully profile-backed choice sets in each "
            f"infrastructure-held-out split; counts: {patronage_choice_counts}"
        )
    manifest = {
        "schema": "osrd-service-plan-preference-split/v2",
        "timetable_mode": timetable_mode,
        "dataset_sha256": dataset_sha256,
        "infrastructure_registry_sha256": registry_sha256,
        "physical_network_ids_by_group": {
            label: registry_groups[label]["physical_network_id"]
            for label in sorted(registry_groups)
        },
        "seed": seed,
        "groups_by_split": {
            name: sorted({event["infrastructure_group"] for event in part})
            for name, part in split.items()
        },
        "choice_sets_by_split": {name: len(part) for name, part in split.items()},
        "minimum_infrastructure_groups": MIN_INFRASTRUCTURE_GROUPS,
        "require_patronage": require_patronage,
        "patronage_profile_choice_set_count_by_split": patronage_choice_counts,
        "minimum_choices_per_infrastructure": MIN_EVENTS_PER_INFRASTRUCTURE,
        "metric_protocol": METRIC_PROTOCOL,
        "created_before_model_fit": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["split_manifest_sha256"] = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    return manifest


def _normalization(
    train_events: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    scales: list[float] = []
    for feature in FEATURES:
        values = [
            float(value)
            for event in train_events
            for candidate in event["features"]
            if (value := candidate[feature]) is not None
        ]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return means, scales


def _vectors(
    events: list[dict[str, Any]], means: list[float], scales: list[float]
) -> list[dict[str, Any]]:
    result = []
    for event in events:
        normalized = [
            _expand_quadratic(
                [
                    ((means[index] if value is None else float(value)) - means[index])
                    / scales[index]
                    for index, feature in enumerate(FEATURES)
                    for value in [candidate[feature]]
                ]
            )
            for candidate in event["features"]
        ]
        result.append({**event, "features": normalized})
    return result


def _expand_quadratic(values: list[float]) -> list[float]:
    """Append second-order terms to a train-normalized base feature vector."""
    return values + [
        left * right
        for index, left in enumerate(values)
        for right in values[index:]
    ]


def _fit(
    events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    learning_rate: float = 0.04,
    regularization: float = 0.05,
    epochs: int = 800,
    patience: int = 60,
) -> tuple[list[float], int]:
    weights = [0.0] * len(MODEL_FEATURES)
    best_weights = weights.copy()
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(epochs):
        gradients = [0.0] * len(MODEL_FEATURES)
        for event in events:
            scores = [
                sum(weight * value for weight, value in zip(weights, row, strict=True))
                for row in event["features"]
            ]
            peak = max(scores)
            exp_scores = [math.exp(score - peak) for score in scores]
            denominator = sum(exp_scores)
            probabilities = [value / denominator for value in exp_scores]
            for index, row in enumerate(event["features"]):
                target = 1.0 if index == event["chosen_index"] else 0.0
                coefficient = probabilities[index] - target
                for feature_index, value in enumerate(row):
                    gradients[feature_index] += coefficient * value
        rate = learning_rate / math.sqrt(1.0 + epoch / 50.0)
        divisor = len(events)
        weights = [
            weight - rate * (gradient / divisor + regularization * weight)
            for weight, gradient in zip(weights, gradients, strict=True)
        ]
        validation_loss = _metrics(validation_events, weights)["listwise_log_loss"]
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_weights = weights.copy()
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return best_weights, best_epoch


def _metrics(events: list[dict[str, Any]], weights: list[float]) -> dict[str, float | int]:
    correct = 0
    reciprocal_rank = 0.0
    random_top1 = 0.0
    random_reciprocal_rank = 0.0
    random_listwise_log_loss = 0.0
    displayed_order_correct = 0
    displayed_order_reciprocal_rank = 0.0
    loss = 0.0
    for event in events:
        scores = [
            sum(weight * value for weight, value in zip(weights, row, strict=True))
            for row in event["features"]
        ]
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        correct += ranked[0] == event["chosen_index"]
        reciprocal_rank += 1.0 / (ranked.index(event["chosen_index"]) + 1)
        displayed_order_correct += event["chosen_index"] == 0
        displayed_order_reciprocal_rank += 1.0 / (event["chosen_index"] + 1)
        random_top1 += 1.0 / len(scores)
        random_reciprocal_rank += (
            sum(1.0 / rank for rank in range(1, len(scores) + 1)) / len(scores)
        )
        random_listwise_log_loss += math.log(len(scores))
        peak = max(scores)
        log_partition = peak + math.log(sum(math.exp(score - peak) for score in scores))
        loss += log_partition - scores[event["chosen_index"]]
    count = len(events)
    return {
        "choice_set_count": count,
        "top1_accuracy": correct / count,
        "mean_reciprocal_rank": reciprocal_rank / count,
        "displayed_order_top1_accuracy": displayed_order_correct / count,
        "displayed_order_mean_reciprocal_rank": displayed_order_reciprocal_rank / count,
        "random_mean_reciprocal_rank_reference": random_reciprocal_rank / count,
        "listwise_log_loss": loss / count,
        "random_listwise_log_loss_reference": random_listwise_log_loss / count,
        "random_top1_reference": random_top1 / count,
    }


def train(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int | None,
    timetable_mode: str,
    split_manifest_path: Path,
    infrastructure_registry_path: Path,
    require_patronage: bool = False,
) -> dict[str, Any]:
    rows, dataset_sha256 = _read_events(input_path)
    if timetable_mode not in TIMETABLE_MODES:
        raise ValueError(f"timetable_mode must be one of {', '.join(TIMETABLE_MODES)}")
    events = [
        event for event in _choice_sets(rows) if event["timetable_mode"] == timetable_mode
    ]
    registry_sha256, registry_groups = _validate_infrastructure_registry(
        infrastructure_registry_path, events
    )
    if not events:
        raise ValueError(
            f"No multi-option chosen-for-review events are available for {timetable_mode}"
        )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("schema") != "osrd-service-plan-preference-split/v2":
        raise ValueError("Unsupported preference split manifest schema")
    if split_manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Frozen split manifest belongs to a different review dataset")
    if split_manifest.get("timetable_mode") != timetable_mode:
        raise ValueError("Frozen split manifest belongs to a different timetable mode")
    if require_patronage and split_manifest.get("require_patronage") is not True:
        raise ValueError("Patronage-required fitting needs a patronage-gated frozen split")
    if split_manifest.get("infrastructure_registry_sha256") != registry_sha256:
        raise ValueError("Frozen split manifest belongs to a different infrastructure registry")
    expected_physical_ids = {
        label: registry_groups[label]["physical_network_id"]
        for label in sorted(registry_groups)
    }
    if split_manifest.get("physical_network_ids_by_group") != expected_physical_ids:
        raise ValueError("Frozen split physical-network audit does not match the registry")
    if split_manifest.get("created_before_model_fit") is not True:
        raise ValueError("Frozen split manifest must be created before model fitting")
    if split_manifest.get("metric_protocol") != METRIC_PROTOCOL:
        raise ValueError("Frozen split manifest does not match the preference metric protocol")
    split_seed = split_manifest.get("seed")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("Frozen split manifest has an invalid seed")
    if seed is not None and seed != split_seed:
        raise ValueError("Training seed must match the frozen split manifest")
    groups_by_split = split_manifest.get("groups_by_split")
    if not isinstance(groups_by_split, dict) or set(groups_by_split) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Frozen split manifest must assign train, validation, and test groups")
    normalized_groups: dict[str, set[str]] = {}
    for name, groups in groups_by_split.items():
        if (
            not isinstance(groups, list)
            or not groups
            or not all(isinstance(group, str) and group for group in groups)
            or len(groups) != len(set(groups))
        ):
            raise ValueError(f"Frozen split has invalid {name} infrastructure groups")
        normalized_groups[name] = set(groups)
    all_assigned = set().union(*normalized_groups.values())
    if sum(len(groups) for groups in normalized_groups.values()) != len(all_assigned):
        raise ValueError("Frozen split assigns an infrastructure to more than one partition")
    observed_groups = {event["infrastructure_group"] for event in events}
    if all_assigned != observed_groups:
        raise ValueError("Frozen split groups do not exactly cover this mode's blind choice data")
    if groups_by_split != {
        name: sorted(groups) for name, groups in groups_by_split.items()
    }:
        raise ValueError("Frozen split infrastructure groups must be sorted")
    splits = {
        name: [event for event in events if event["infrastructure_group"] in groups]
        for name, groups in normalized_groups.items()
    }
    if any(not part for part in splits.values()):
        raise ValueError("Frozen split must include train, validation, and test choice sets")
    if any(len(part) < minimum for part, minimum in (
        (splits["train"], MIN_TRAIN_EVENTS),
        (splits["validation"], MIN_VALIDATION_EVENTS),
        (splits["test"], MIN_TEST_EVENTS),
    )):
        raise ValueError("Frozen split does not meet the per-partition choice-set minimums")
    for group in observed_groups:
        count = sum(event["infrastructure_group"] == group for event in events)
        if count < MIN_EVENTS_PER_INFRASTRUCTURE:
            raise ValueError(f"Infrastructure {group} has fewer than 30 blind choices")
    declared_counts = split_manifest.get("choice_sets_by_split")
    actual_counts = {name: len(part) for name, part in splits.items()}
    if declared_counts != actual_counts:
        raise ValueError("Frozen split choice-set counts do not match the dataset")
    if len(observed_groups) < MIN_INFRASTRUCTURE_GROUPS:
        raise ValueError("Frozen split does not meet the eight-infrastructure minimum")
    patronage_counts = {
        name: sum(
            bool(event["features"])
            and all(
                candidate["patronage_coverage_available"] == 1.0
                for candidate in event["features"]
            )
            for event in part
        )
        for name, part in splits.items()
    }
    if require_patronage and any(count < 30 for count in patronage_counts.values()):
        raise ValueError(
            "Patronage ranker requires at least 30 fully profile-backed choice sets in each "
            f"infrastructure-held-out split; counts: {patronage_counts}"
        )
    split_manifest_sha256 = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    means, scales = _normalization(splits["train"])
    normalized = {name: _vectors(part, means, scales) for name, part in splits.items()}
    weights, selected_epoch = _fit(
        normalized["train"], normalized["validation"]
    )
    report = {
        "schema": MODEL_SCHEMA,
        "timetable_mode": timetable_mode,
        "dataset_sha256": dataset_sha256,
        "infrastructure_registry_sha256": registry_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split_seed": split_seed,
        "split_created_before_model_fit": split_manifest.get("created_before_model_fit") is True,
        "metric_protocol": METRIC_PROTOCOL,
        "source_event_count": len(rows),
        "blind_review_event_count": sum(
            row.get("review_protocol") == "blind_planner_choice_v1" for row in rows
        ),
        "unblinded_or_unknown_event_count": sum(
            row.get("review_protocol") != "blind_planner_choice_v1" for row in rows
        ),
        "model_assisted_choice_event_count_excluded": sum(
            row.get("event") == "chosen_for_review"
            and row.get("timetable_mode") == timetable_mode
            and isinstance(row.get("research_model"), dict)
            and row["research_model"].get("status") == "used"
            for row in rows
        ),
        "multi_option_choice_count": len(events),
        "blind_review_choice_count": len(events),
        "minimum_blind_choices_per_infrastructure": MIN_EVENTS_PER_INFRASTRUCTURE,
        "minimum_infrastructure_groups": MIN_INFRASTRUCTURE_GROUPS,
        "selected_epoch": selected_epoch,
        "quadratic_feature_count": len(MODEL_FEATURES),
        "patronage_profile_choice_set_count_by_split": patronage_counts,
        "trained_with_patronage_requirement": require_patronage,
        "standardized_model_coefficients": {
            name: weight for name, weight in zip(MODEL_FEATURES, weights, strict=True)
        },
        "split_infrastructure_groups": {
            name: sorted({event["infrastructure_group"] for event in part})
            for name, part in splits.items()
        },
        "metrics": {name: _metrics(part, weights) for name, part in normalized.items()},
        "metrics_by_infrastructure": {
            split_name: {
                group: _metrics(
                    [
                        event
                        for event in split_events
                        if event["infrastructure_group"] == group
                    ],
                    weights,
                )
                for group in sorted(
                    {event["infrastructure_group"] for event in split_events}
                )
            }
            for split_name, split_events in normalized.items()
        },
        "limitations": [
            "Targets are planner choices for review, not accepted plans or operational outcomes.",
            "Alternatives are compared only within the feasible choice set shown to the planner.",
            "Groups are split by infrastructure, but the available review sample may not "
            "represent network operations.",
            "This model is research-only and is not wired into live planner ranking.",
        ],
        "live_promotion_eligible": False,
    }
    model = {
        "schema": MODEL_SCHEMA,
        "timetable_mode": timetable_mode,
        "base_features": list(FEATURES),
        "features": list(MODEL_FEATURES),
        "feature_expansion": "standardized_quadratic_v1",
        "means": means,
        "scales": scales,
        "weights": weights,
        "dataset_sha256": dataset_sha256,
        "infrastructure_registry_sha256": registry_sha256,
        "research_only": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    model_content = json.dumps(model, indent=2, sort_keys=True) + "\n"
    model_path.write_text(model_content, encoding="utf-8")
    report["model_sha256"] = hashlib.sha256(model_content.encode()).hexdigest()
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "review_events", type=Path, help="Review events JSONL from the review exporter"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timetable-mode", choices=TIMETABLE_MODES, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--infrastructure-registry", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--require-patronage",
        action="store_true",
        help=(
            "Require 30 choice sets with profile-backed coverage for every alternative "
            "in each held-out split"
        ),
    )
    args = parser.parse_args()
    try:
        report = train(
            args.review_events,
            args.output_dir,
            seed=args.seed,
            timetable_mode=args.timetable_mode,
            split_manifest_path=args.split_manifest,
            infrastructure_registry_path=args.infrastructure_registry,
            require_patronage=args.require_patronage,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


def freeze_main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a dataset-bound infrastructure split before preference-model fitting"
    )
    parser.add_argument("review_events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timetable-mode", choices=TIMETABLE_MODES, required=True)
    parser.add_argument("--infrastructure-registry", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-patronage", action="store_true")
    args = parser.parse_args()
    try:
        report = freeze_split(
            args.review_events,
            args.output,
            seed=args.seed,
            timetable_mode=args.timetable_mode,
            infrastructure_registry_path=args.infrastructure_registry,
            require_patronage=args.require_patronage,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
