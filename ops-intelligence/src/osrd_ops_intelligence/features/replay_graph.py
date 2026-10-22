"""Train interaction graph derived from OSRD pathfinding resources."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

LEGACY_EDGE_FEATURE_NAMES = (
    "departure_gap_over_headway",
    "shared_track_jaccard",
    "shared_route_jaccard",
    "shared_block_jaccard",
)
EDGE_FEATURE_NAMES = (
    "departure_gap_over_headway",
    "absolute_departure_gap_over_headway",
    "shared_track_jaccard",
    "shared_route_jaccard",
    "shared_block_jaccard",
    "shared_track_count",
    "shared_route_count",
    "shared_block_count",
    "candidate_service_interaction",
)
ACTION_SUMMARY_FEATURE_NAMES = (
    "requested_frequency_fraction_of_600s",
    "departure_phase_fraction_of_120s",
    "operating_window_fraction_of_1800s",
    "requested_service_count_fraction_of_8",
    "requested_stop_count_fraction_of_4",
    "requested_total_dwell_fraction_of_600s",
    "route_waypoint_count_fraction_of_4",
    "retimed_train_count_fraction_of_8",
    "retiming_signed_sum_fraction_of_2400s",
    "retiming_magnitude_sum_fraction_of_2400s",
    "departure_time_of_day_sin",
    "departure_time_of_day_cos",
)


def action_summary_features(action: dict[str, Any]) -> Tensor:
    """Return a compact, normalized summary of the proposed plan action."""
    retimings = [
        int(item.get("departure_shift_s", 0)) for item in action.get("retimings", [])
    ]
    stops = action.get("stops", [])
    departure_time = float(action.get("departure_time_ms", 0)) / 1000 % 86400
    return torch.tensor(
        [
            float(action.get("frequency_s", 0)) / 600,
            float(action.get("departure_phase_s", 0)) / 120,
            float(action.get("operating_window_s", 0)) / 1800,
            float(action.get("requested_service_count", 0)) / 8,
            len(stops) / 4,
            sum(float(stop.get("dwell_s", 0)) for stop in stops) / 600,
            len(action.get("route_via", [])) / 4,
            len(retimings) / 8,
            sum(retimings) / 2400,
            sum(abs(shift) for shift in retimings) / 2400,
            torch.sin(torch.tensor(2 * torch.pi * departure_time / 86400)).item(),
            torch.cos(torch.tensor(2 * torch.pi * departure_time / 86400)).item(),
        ],
        dtype=torch.float32,
    )


def interaction_graph(
    episode: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES,
) -> tuple[Tensor, Tensor]:
    """Connect time-neighbour or resource-sharing trains, without reading outcomes."""
    schedules = episode["inputs"]["schedule_payloads"]
    paths_by_train = episode["inputs"].get("resource_paths_by_train") or []
    default_path = episode["inputs"].get("resource_path") or {}
    paths = [
        paths_by_train[index] if index < len(paths_by_train) else default_path
        for index in range(len(schedules))
    ]

    def resource_set(path: dict[str, Any], collection: str, key: str) -> set[str]:
        return {str(resource[key]) for resource in path.get(collection, []) if resource.get(key)}

    tracks = [resource_set(path, "track_section_ranges", "track_section") for path in paths]
    routes = [resource_set(path, "routes", "id") for path in paths]
    blocks = [resource_set(path, "blocks", "id") for path in paths]
    starts = [float(schedule["start_time"]) / 1000 for schedule in schedules]
    ordered = sorted(range(len(starts)), key=lambda index: (starts[index], index))
    pairs = {tuple(sorted(pair)) for pair in zip(ordered, ordered[1:], strict=False)}
    for left in range(len(starts)):
        for right in range(left + 1, len(starts)):
            if (
                tracks[left] & tracks[right]
                or routes[left] & routes[right]
                or blocks[left] & blocks[right]
            ):
                pairs.add((left, right))

    def jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    headway = max(float(episode["inputs"]["scenario"]["headway_s"]), 1.0)
    baseline_count = len(episode["inputs"].get("baseline_schedule_payloads", []))
    action = episode["inputs"].get("action") or episode["inputs"].get("incident") or {}
    candidate_indices = {
        int(item["train_index"])
        for item in action.get("retimings", [])
        if int(item.get("departure_shift_s", 0)) != 0
    }
    if action.get("train_index") is not None:
        candidate_indices.add(int(action["train_index"]))
    candidate_indices.update(range(baseline_count, len(schedules)))
    edge_rows: list[tuple[float, ...]] = []
    for left, right in sorted(pairs):
        gap = (starts[right] - starts[left]) / headway
        shared_tracks = tracks[left] & tracks[right]
        shared_routes = routes[left] & routes[right]
        shared_blocks = blocks[left] & blocks[right]
        edge_features = (
            jaccard(tracks[left], tracks[right]),
            jaccard(routes[left], routes[right]),
            jaccard(blocks[left], blocks[right]),
            float(len(shared_tracks)),
            float(len(shared_routes)),
            float(len(shared_blocks)),
            float(left in candidate_indices or right in candidate_indices),
        )
        if feature_names == LEGACY_EDGE_FEATURE_NAMES:
            forward_features = (gap, *edge_features[:3])
            backward_features = (-gap, *edge_features[:3])
        elif feature_names == EDGE_FEATURE_NAMES:
            forward_features = (gap, abs(gap), *edge_features)
            backward_features = (-gap, abs(gap), *edge_features)
        else:
            raise ValueError(f"Unsupported replay graph edge schema: {feature_names}")
        edge_rows.extend(
            (
                (left, right, *forward_features),
                (right, left, *backward_features),
            )
        )
    edge_index = torch.tensor(
        [[edge[0] for edge in edge_rows], [edge[1] for edge in edge_rows]],
        dtype=torch.long,
        device=device,
    ).reshape(2, -1)
    edge_attr = torch.tensor(
        [edge[2:] for edge in edge_rows], dtype=torch.float32, device=device
    ).reshape(-1, len(feature_names))
    return edge_index, edge_attr
