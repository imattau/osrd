"""Create a bounded, referentially valid RailJSON crop for OSRD route-search experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _route_corridor(
    tracks: list[dict[str, Any]], switches: list[dict[str, Any]], start: str, end: str
) -> tuple[set[str], set[str]]:
    known = {str(track["id"]) for track in tracks}
    adjacency: dict[str, list[tuple[str, str]]] = {track_id: [] for track_id in known}
    for switch in switches:
        ports = switch.get("ports")
        if not isinstance(ports, dict):
            continue
        port_tracks = {
            port_name: str(port.get("track"))
            for port_name, port in ports.items()
            if isinstance(port, dict) and str(port.get("track")) in known
        }
        switch_id = str(switch["id"])
        switch_type = switch.get("switch_type")
        if switch_type == "link":
            allowed_pairs = [("A", "B")]
        elif switch_type == "point_switch":
            allowed_pairs = [("A", "B1"), ("A", "B2")]
        elif switch_type == "double_slip_switch":
            allowed_pairs = [(left, right) for left in ("A1", "A2") for right in ("B1", "B2")]
        elif switch_type == "crossing":
            allowed_pairs = [("A1", "B1"), ("A2", "B2")]
        else:
            names = sorted(port_tracks)
            allowed_pairs = [
                (left, right) for i, left in enumerate(names) for right in names[i + 1 :]
            ]
        for left_port, right_port in allowed_pairs:
            left, right = port_tracks.get(left_port), port_tracks.get(right_port)
            if left is not None and right is not None and left != right:
                adjacency[left].append((right, switch_id))
                adjacency[right].append((left, switch_id))
    if start not in known or end not in known:
        raise ValueError("Route endpoint track is absent from the source infrastructure")
    pending = [start]
    previous: dict[str, tuple[str, str] | None] = {start: None}
    for current in pending:
        if current == end:
            break
        for neighbor, switch_id in adjacency[current]:
            if neighbor not in previous:
                previous[neighbor] = (current, switch_id)
                pending.append(neighbor)
    if end not in previous:
        raise ValueError(f"No switch-connected path from {start} to {end}")
    corridor_tracks = {end}
    corridor_switches: set[str] = set()
    current = end
    while current != start:
        predecessor = previous[current]
        if predecessor is None:
            raise ValueError("Invalid path predecessor chain")
        current, switch_id = predecessor
        corridor_tracks.add(current)
        corridor_switches.add(switch_id)
    switch_by_id = {str(switch["id"]): switch for switch in switches}
    # Include every port track for selected junctions so each retained switch
    # remains valid; these branch tracks terminate at the corridor boundary.
    for switch_id in corridor_switches:
        for port in switch_by_id[switch_id]["ports"].values():
            corridor_tracks.add(str(port["track"]))
    return corridor_tracks, corridor_switches


def crop(
    input_path: Path,
    output_path: Path,
    bounds: tuple[float, float, float, float] | None = None,
    route_endpoints: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Keep a geographic crop or a switch-connected route corridor."""
    raw: Any = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RailJSON root must be an object")
    document = cast(dict[str, Any], raw)
    if (bounds is None) == (route_endpoints is None):
        raise ValueError("Provide exactly one of bounds or route endpoints")

    def intersects(section: dict[str, Any]) -> bool:
        geo = section.get("geo")
        coordinates = geo.get("coordinates") if isinstance(geo, dict) else None
        if not isinstance(coordinates, list) or not coordinates:
            return False
        assert bounds is not None
        west, south, east, north = bounds
        for coordinate in coordinates:
            if not isinstance(coordinate, list) or len(coordinate) < 2:
                raise ValueError(f"Track {section.get('id')} has invalid geometry")
            lon, lat = coordinate[0], coordinate[1]
            if west <= lon <= east and south <= lat <= north:
                return True
        # Include a section that crosses the crop even when no source vertex lies
        # inside it. Segment bounding-box overlap is a conservative spatial filter.
        for start, end in zip(coordinates, coordinates[1:], strict=False):
            if (
                max(start[0], end[0]) >= west
                and min(start[0], end[0]) <= east
                and max(start[1], end[1]) >= south
                and min(start[1], end[1]) <= north
            ):
                return True
        return False

    source_tracks = cast(list[dict[str, Any]], document.get("track_sections", []))
    source_switches = cast(list[dict[str, Any]], document.get("switches", []))
    corridor_switch_ids: set[str] | None = None
    if route_endpoints is not None:
        route_track_ids, corridor_switch_ids = _route_corridor(
            source_tracks, source_switches, *route_endpoints
        )
        kept_track_sections = [
            track for track in source_tracks if str(track["id"]) in route_track_ids
        ]
    else:
        assert bounds is not None
        west, south, east, north = bounds
        if west >= east or south >= north:
            raise ValueError("Bounds must be ordered west, south, east, north")
        kept_track_sections = [track for track in source_tracks if intersects(track)]
    track_ids = {str(track["id"]) for track in kept_track_sections}
    if not track_ids:
        raise ValueError("No track sections intersect the requested crop")

    kept_switches = []
    for switch in source_switches:
        ports = switch.get("ports")
        if (
            isinstance(ports, dict)
            and (corridor_switch_ids is None or str(switch.get("id")) in corridor_switch_ids)
            and all(
                isinstance(port, dict) and str(port.get("track")) in track_ids
                for port in ports.values()
            )
        ):
            kept_switches.append(switch)

    def on_kept_track(item: dict[str, Any]) -> bool:
        return str(item.get("track")) in track_ids

    def keep_range_section(item: dict[str, Any]) -> bool:
        ranges = item.get("track_ranges")
        return isinstance(ranges, list) and any(
            isinstance(value, dict) and str(value.get("track")) in track_ids for value in ranges
        )

    kept_buffer_stops = [
        item
        for item in cast(list[dict[str, Any]], document.get("buffer_stops", []))
        if on_kept_track(item)
    ]
    kept_detectors = [
        item
        for item in cast(list[dict[str, Any]], document.get("detectors", []))
        if on_kept_track(item)
    ]
    # Routes encode generated block topology and are intentionally discarded. OSRD
    # regenerates them when the cropped infrastructure is imported.
    removed_counts: dict[str, int] = {"routes": len(document.get("routes", []))}
    document["routes"] = []
    document["track_sections"] = kept_track_sections
    document["switches"] = kept_switches
    document["buffer_stops"] = kept_buffer_stops
    document["detectors"] = kept_detectors
    for key in ("signals",):
        values = cast(list[dict[str, Any]], document.get(key, []))
        selected = [item for item in values if on_kept_track(item)]
        removed_counts[key] = len(values) - len(selected)
        document[key] = selected
    for key in ("speed_sections", "electrifications", "neutral_sections"):
        values = cast(list[dict[str, Any]], document.get(key, []))
        selected = []
        for item in values:
            ranges = item.get("track_ranges")
            if not isinstance(ranges, list):
                continue
            clipped = [
                value
                for value in ranges
                if isinstance(value, dict) and str(value.get("track")) in track_ids
            ]
            if clipped:
                copy = dict(item)
                copy["track_ranges"] = clipped
                selected.append(copy)
        removed_counts[key] = len(values) - len(selected)
        document[key] = selected
    points = cast(list[dict[str, Any]], document.get("operational_points", []))
    selected_points = []
    for point in points:
        parts = point.get("parts")
        if not isinstance(parts, list):
            continue
        kept_parts = [
            part for part in parts if isinstance(part, dict) and str(part.get("track")) in track_ids
        ]
        if kept_parts:
            copy = dict(point)
            copy["parts"] = kept_parts
            selected_points.append(copy)
    removed_counts["operational_points"] = len(points) - len(selected_points)
    document["operational_points"] = selected_points
    crossings = cast(list[dict[str, Any]], document.get("level_crossings", []))
    selected_crossings = [
        item
        for item in crossings
        if any(
            str(port.get("track")) in track_ids
            for port in item.get("track", [])
            if isinstance(port, dict)
        )
    ]
    removed_counts["level_crossings"] = len(crossings) - len(selected_crossings)
    document["level_crossings"] = selected_crossings

    # Fail closed on the references that can be checked from this RailJSON subset.
    for switch in kept_switches:
        if any(str(port.get("track")) not in track_ids for port in switch["ports"].values()):
            raise ValueError(f"Dangling track reference in switch {switch.get('id')}")
    for item in [*document["signals"], *kept_detectors, *kept_buffer_stops]:
        if str(item.get("track")) not in track_ids:
            raise ValueError(f"Dangling track reference in {item.get('id')}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return {
        "schema": "osrd-ops-osm-railjson-crop/v1",
        "source_sha256": _sha256(input_path),
        "output_sha256": _sha256(output_path),
        "bounds_wgs84": (
            {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3]}
            if bounds is not None
            else None
        ),
        "route_track_endpoints": list(route_endpoints) if route_endpoints else None,
        "track_section_count": len(kept_track_sections),
        "switch_count": len(kept_switches),
        "operational_point_count": len(selected_points),
        "detector_count": len(kept_detectors),
        "buffer_stop_count": len(kept_buffer_stops),
        "removed_counts": removed_counts,
        "route_policy": "discard generated routes; regenerate on OSRD import",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--route-start")
    parser.add_argument("--route-end")
    args = parser.parse_args()
    route = None
    if args.route_start is not None or args.route_end is not None:
        if args.route_start is None or args.route_end is None:
            parser.error("--route-start and --route-end must be supplied together")
        route = (args.route_start, args.route_end)
    bounds = tuple(args.bounds) if args.bounds is not None else None
    print(json.dumps(crop(args.input, args.output, bounds, route), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
