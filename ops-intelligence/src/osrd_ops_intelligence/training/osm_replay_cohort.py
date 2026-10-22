"""Discover and collect geographically split OSM switch-level replay cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, cast


def _track_center(track: dict[str, Any]) -> tuple[float, float]:
    geo = track.get("geo")
    coordinates = geo.get("coordinates") if isinstance(geo, dict) else None
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError(f"Track {track.get('id')} has no geometry")
    points = [point for point in coordinates if isinstance(point, list) and len(point) >= 2]
    if not points:
        raise ValueError(f"Track {track.get('id')} has invalid geometry")
    return (
        sum(float(point[1]) for point in points) / len(points),
        sum(float(point[0]) for point in points) / len(points),
    )


def candidates(
    source_path: Path, *, margin_m: int = 100, test_latitude: float = 50.7
) -> list[dict[str, Any]]:
    """Choose one valid switch-connected route per 0.25-degree geographic cell."""
    document: Any = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("RailJSON root must be an object")
    data = cast(dict[str, Any], document)
    tracks = cast(list[dict[str, Any]], data.get("track_sections", []))
    switches = cast(list[dict[str, Any]], data.get("switches", []))
    operational_points = cast(list[dict[str, Any]], data.get("operational_points", []))
    track_by_id = {str(track["id"]): track for track in tracks}
    stations_by_track: dict[str, list[dict[str, Any]]] = {}
    for point in operational_points:
        if not point.get("is_passenger_station"):
            continue
        for part in point.get("parts", []):
            if not isinstance(part, dict) or "track" not in part or "position" not in part:
                continue
            stations_by_track.setdefault(str(part["track"]), []).append(
                {
                    "id": str(point.get("id", "")),
                    "name": str(point.get("name", "")),
                    "track": str(part["track"]),
                    "offset_m": float(part["position"]),
                    "offset_mm": round(float(part["position"]) * 1000),
                }
            )
    best_by_cell: dict[tuple[int, int], dict[str, Any]] = {}

    def port_pairs(switch: dict[str, Any]) -> list[tuple[str, str]]:
        switch_type = switch.get("switch_type")
        if switch_type == "link":
            return [("A", "B")]
        if switch_type == "point_switch":
            return [("A", "B1"), ("A", "B2")]
        if switch_type == "double_slip_switch":
            return [(left, right) for left in ("A1", "A2") for right in ("B1", "B2")]
        if switch_type == "crossing":
            return [("A1", "B1"), ("A2", "B2")]
        return []

    for switch in switches:
        ports = switch.get("ports")
        if not isinstance(ports, dict):
            continue
        for start_port, end_port in port_pairs(switch):
            start = ports.get(start_port)
            end = ports.get(end_port)
            if not isinstance(start, dict) or not isinstance(end, dict):
                continue
            start_track = track_by_id.get(str(start.get("track")))
            end_track = track_by_id.get(str(end.get("track")))
            if start_track is None or end_track is None:
                continue
            start_length = float(start_track["length"])
            end_length = float(end_track["length"])
            if start_length < margin_m * 2 or end_length < margin_m * 2:
                continue
            start_offset_m = margin_m if start.get("endpoint") == "END" else start_length - margin_m
            end_offset_m = margin_m if end.get("endpoint") == "END" else end_length - margin_m
            start_offset_mm = round(start_offset_m * 1000)
            end_offset_mm = round(end_offset_m * 1000)
            latitude, longitude = _track_center(start_track)
            cell = (int(latitude * 4), int(longitude * 4))
            candidate = {
                "switch_id": str(switch["id"]),
                "switch_type": str(switch["switch_type"]),
                "start_track": str(start["track"]),
                "start_offset_mm": start_offset_mm,
                "start_endpoint": str(start["endpoint"]),
                "end_track": str(end["track"]),
                "end_offset_mm": end_offset_mm,
                "end_endpoint": str(end["endpoint"]),
                "latitude": latitude,
                "longitude": longitude,
                "spatial_cell": f"{cell[0]}:{cell[1]}",
                "minimum_track_length_m": min(start_length, end_length),
                "start_track_length_m": start_length,
                "end_track_length_m": end_length,
                "station_stops": [],
            }
            # Keep stop choices on the directed route between the route endpoint
            # and the switch. This avoids inventing arbitrary mid-track stops.
            for station in stations_by_track.get(str(start["track"]), []):
                offset = float(station["offset_m"])
                direction = 1 if start.get("endpoint") == "END" else -1
                if (
                    25
                    <= (offset - start_offset_m) * direction
                    <= max(25, start_length - 2 * margin_m)
                ):
                    candidate["station_stops"].append({**station, "route_leg": "approach"})
            for station in stations_by_track.get(str(end["track"]), []):
                offset = float(station["offset_m"])
                direction = 1 if end.get("endpoint") == "END" else -1
                if 25 <= (offset - end_offset_m) * direction <= max(25, end_length - 2 * margin_m):
                    candidate["station_stops"].append({**station, "route_leg": "departure"})
            current = best_by_cell.get(cell)
            if (
                current is None
                or candidate["minimum_track_length_m"] > current["minimum_track_length_m"]
            ):
                best_by_cell[cell] = candidate

    rows = sorted(best_by_cell.values(), key=lambda row: (row["latitude"], row["longitude"]))
    for row in rows:
        # Hold out the northern latitude band to prevent nearby cells from
        # crossing partitions; all episodes on one switch stay in one group.
        row["partition"] = "test" if row["latitude"] >= test_latitude else "train"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for row in rows:
        row["source_railjson_sha256"] = source_hash
    return rows


def _spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    return [rows[index] for index in indices]


def _select_candidates(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if (args.partition == "all" or row["partition"] == args.partition)
        and row["switch_id"] not in set(args.exclude_switch_ids)
    ]
    if args.include_switch_ids:
        if len(args.include_switch_ids) != len(set(args.include_switch_ids)):
            raise ValueError("Included switch IDs must be unique")
        by_id = {row["switch_id"]: row for row in eligible}
        missing = sorted(set(args.include_switch_ids) - by_id.keys())
        if missing:
            raise ValueError(
                "Requested switch IDs are not eligible in this partition: " + ", ".join(missing)
            )
        return [by_id[switch_id] for switch_id in args.include_switch_ids]
    return _spread(eligible, args.limit)


def collect(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    selected = _select_candidates(rows, args)
    if not selected:
        raise ValueError(f"No candidates selected for partition {args.partition}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "route-candidates.json").write_text(
        json.dumps(
            {
                "schema": "osrd-ops-osm-switch-route-candidates/v1",
                "track_offset_unit": "mm",
                "source_railjson": str(args.input),
                "source_railjson_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "network_name": args.network_name,
                "infra_id": args.infra_id,
                "episodes_per_cohort": args.episodes,
                "trains_per_baseline": args.trains,
                "insertion_frequency_options_s": args.frequency_options_s,
                "insertion_stop_dwell_options_s": args.stop_dwell_options_s,
                "station_stop_dwell_options_s": args.station_stop_dwell_options_s,
                "station_stop_limit": args.station_stop_limit,
                "stop_position_method": (
                    "Passenger operational point from RailJSON; metres converted to Editoast mm."
                    if args.station_stop_dwell_options_s
                    else (
                        "Synthetic stop 50m downstream of inserted service start; "
                        "not station-mapped."
                        if args.stop_dwell_options_s
                        else None
                    )
                ),
                "operating_window_s": args.window_s,
                "phase_options_s": args.phase_options_s,
                "route_via_options": args.route_via_options,
                "plan_retiming_options": args.plan_retiming_actions,
                "max_retime_s": args.max_retime_s,
                "delay_profile": str(args.delay_profile) if args.delay_profile else None,
                "base_seed": args.seed,
                "spatial_test_partition": f"latitude >= {args.test_latitude}",
                "excluded_switch_ids": args.exclude_switch_ids,
                "included_switch_ids": args.include_switch_ids,
                "routes": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    replay_script = Path(__file__).with_name("replay.py")
    cohort_count = 0
    stop_options = args.stop_dwell_options_s or [None]
    retiming_options = args.plan_retiming_actions or ["random"]
    for index, row in enumerate(selected):
        for frequency_s in args.frequency_options_s:
            for phase_s in args.phase_options_s:
                station_options = _spread(row["station_stops"], args.station_stop_limit)
                stop_cases: list[tuple[dict[str, Any] | None, int | None]] = [(None, None)]
                if args.station_stop_dwell_options_s:
                    stop_cases.extend(
                        (station, dwell)
                        for station in station_options
                        for dwell in args.station_stop_dwell_options_s
                    )
                if args.stop_dwell_options_s:
                    stop_cases.extend((None, dwell) for dwell in args.stop_dwell_options_s)
                for station, station_dwell_s in stop_cases:
                    dwell_cases = [station_dwell_s] if station_dwell_s is not None else stop_options
                    for stop_dwell_s in dwell_cases:
                        for retiming_option, (route_via_index, route_via_option) in product(
                            retiming_options, enumerate(args.route_via_options)
                        ):
                            frequency_suffix = (
                                f"-freq-{frequency_s}"
                                if len(args.frequency_options_s) > 1 or frequency_s != 300
                                else ""
                            )
                            phase_suffix = (
                                f"-phase-{phase_s}"
                                if len(args.phase_options_s) > 1 or phase_s != 0
                                else ""
                            )
                            station_suffix = (
                                f"-station-{station['id']}-dwell-{station_dwell_s}"
                                if station is not None
                                else ""
                            )
                            stop_suffix = (
                                f"-stop-{stop_dwell_s}" if stop_dwell_s is not None else ""
                            )
                            route_via_suffix = (
                                f"-route-via-{route_via_index:02d}"
                                if route_via_option != "none"
                                else ""
                            )
                            retiming_suffix = (
                                "-retime-random"
                                if retiming_option == "random"
                                else "-retime-"
                                + retiming_option.replace(":", "-").replace(",", "_")
                            )
                            suffix = (
                                f"{frequency_suffix}{phase_suffix}{stop_suffix}"
                                f"{route_via_suffix}{retiming_suffix}"
                            )
                            scenario = (
                                f"{args.network_name}-switch-{row['switch_id']}-"
                                f"{row['partition']}-matched-v2"
                            )
                            output = output_root / (
                                f"switch-{row['switch_id']}-{row['partition']}{suffix}"
                                f"{station_suffix}-v2"
                            )
                            command = [
                                sys.executable,
                                str(replay_script),
                                "--editoast-url",
                                args.editoast_url,
                                "--infra-id",
                                str(args.infra_id),
                                "--rolling-stock",
                                args.rolling_stock,
                                "--start-track",
                                row["start_track"],
                                "--start-offset",
                                str(row["start_offset_mm"]),
                                "--end-track",
                                row["end_track"],
                                "--end-offset",
                                str(row["end_offset_mm"]),
                                "--output",
                                str(output),
                                "--scenario-group",
                                scenario,
                                "--seed",
                                str(args.seed + index),
                                "--episodes",
                                str(args.episodes),
                                "--trains",
                                str(args.trains),
                                "--intervention-type",
                                "service-plan",
                                "--insertion-frequency-s",
                                str(frequency_s),
                                "--insertion-window-s",
                                str(args.window_s),
                                "--insertion-phase-s",
                                str(phase_s),
                                "--max-retime-s",
                                str(args.max_retime_s),
                            ]
                            if retiming_option != "random":
                                actions = (
                                    [] if retiming_option == "none" else retiming_option.split(",")
                                )
                                for action in actions or ["none"]:
                                    command.extend(["--plan-retiming-action", action])
                            if route_via_option != "none":
                                command.extend(
                                    ["--insertion-route-via", *route_via_option.split(",")]
                                )
                            if args.delay_profile is not None:
                                command.extend(
                                    ["--empirical-delay-profile", str(args.delay_profile)]
                                )
                            if stop_dwell_s is not None:
                                if station is not None:
                                    stop_track = station["track"]
                                    stop_offset = station["offset_mm"]
                                    stop_arg = (
                                        f"{stop_track}:{stop_offset}:{stop_dwell_s}:{station['id']}"
                                    )
                                else:
                                    direction = 1 if row["start_endpoint"] == "END" else -1
                                    stop_track = row["start_track"]
                                    stop_offset = row["start_offset_mm"] + direction * 50_000
                                    if not 0 <= stop_offset <= row["start_track_length_m"] * 1000:
                                        raise ValueError(
                                            "Cannot place a 50m downstream stop on switch "
                                            f"{row['switch_id']}"
                                        )
                                    stop_arg = f"{stop_track}:{stop_offset}:{stop_dwell_s}"
                                command.extend(["--insertion-stop", stop_arg])
                            cohort_count += 1
                            print(
                                f"Collecting {cohort_count}: switch={row['switch_id']} "
                                f"cell={row['spatial_cell']} partition={row['partition']} "
                                f"frequency={frequency_s}s phase={phase_s}s stop={stop_dwell_s}s "
                                f"route_via={route_via_option} retiming={retiming_option}",
                                flush=True,
                            )
                            subprocess.run(command, check=True)
    return cohort_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/osm-switch-cohorts-v2"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--margin-m",
        type=int,
        default=100,
        help="Keep generated route endpoints this many metres from each track end.",
    )
    parser.add_argument("--test-latitude", type=float, default=50.7)
    parser.add_argument("--exclude-switch-ids", nargs="*", default=[])
    parser.add_argument(
        "--include-switch-ids",
        nargs="+",
        help="Collect these exact eligible switch routes instead of spatially spreading --limit.",
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--infra-id", type=int, default=13)
    parser.add_argument("--network-name", default="belgium-osm")
    parser.add_argument("--editoast-url", default="http://localhost:8090")
    parser.add_argument("--rolling-stock", default="fast_rolling_stock")
    parser.add_argument("--partition", choices=("train", "test", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--trains", type=int, default=8)
    parser.add_argument("--phase-options-s", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--route-via-options",
        nargs="+",
        default=["none"],
        metavar="NONE_OR_TRACK:OFFSET[,TRACK:OFFSET...]",
        help=(
            "Matched route-waypoint actions. Include 'none' and comma-separated ordered "
            "TRACK:OFFSET_MM points for each constrained option."
        ),
    )
    parser.add_argument("--frequency-s", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--frequency-options-s", nargs="+", type=int)
    parser.add_argument(
        "--stop-dwell-options-s",
        nargs="+",
        type=int,
        default=[],
        help="Synthetic stop stress cases 50m into the candidate route leg; not station-mapped.",
    )
    parser.add_argument(
        "--station-stop-dwell-options-s",
        nargs="+",
        type=int,
        default=[],
        help="Collect station-mapped passenger-stop alternatives from RailJSON operational points.",
    )
    parser.add_argument("--station-stop-limit", type=int, default=2)
    parser.add_argument("--window-s", type=int, default=900)
    parser.add_argument(
        "--plan-retiming-actions",
        nargs="+",
        help=(
            "Matched coupled-plan alternatives. Use 'none', 'random', or comma-separated "
            "TRAIN_INDEX:SHIFT_SECONDS entries, e.g. none 0:60 1:-60 0:60,1:-60. "
            "All alternatives share their baseline seed and scenario group."
        ),
    )
    parser.add_argument("--max-retime-s", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--delay-profile",
        type=Path,
        default=Path("data/rail/infrabel-punctuality/profiles/origin-delay-profile-2025.json"),
    )
    parser.add_argument(
        "--no-delay-profile",
        action="store_true",
        help="Do not condition the replay on the default Belgian Infrabel delay profile.",
    )
    args = parser.parse_args()
    args.frequency_options_s = args.frequency_options_s or [args.frequency_s]
    if args.no_delay_profile:
        args.delay_profile = None
    rows = candidates(args.input, margin_m=args.margin_m, test_latitude=args.test_latitude)
    if args.limit < 1 or args.episodes < 1 or args.trains < 2:
        parser.error("limit and episodes must be positive; trains must be at least 2")
    if any(frequency <= 0 for frequency in args.frequency_options_s):
        parser.error("frequency options must be positive")
    if len(args.route_via_options) != len(set(args.route_via_options)):
        parser.error("route-via options must be unique")
    if "none" not in args.route_via_options:
        parser.error("route-via options must include 'none' as the unconstrained candidate")
    for option in args.route_via_options:
        if option == "none":
            continue
        seen_waypoints: set[tuple[str, int]] = set()
        for waypoint_value in option.split(","):
            parts = waypoint_value.split(":")
            if len(parts) != 2 or not parts[0]:
                parser.error(f"Invalid route-via option: {option}")
            try:
                offset = int(parts[1])
            except ValueError:
                parser.error(f"Invalid route-via offset in option: {option}")
            waypoint = (parts[0], offset)
            if offset < 0 or waypoint in seen_waypoints:
                parser.error(f"Route-via waypoints must be nonnegative and unique: {option}")
            seen_waypoints.add(waypoint)
    if any(dwell <= 0 for dwell in args.stop_dwell_options_s):
        parser.error("stop dwell options must be positive")
    if any(dwell <= 0 for dwell in args.station_stop_dwell_options_s):
        parser.error("station stop dwell options must be positive")
    if args.station_stop_limit < 1:
        parser.error("--station-stop-limit must be positive")
    if args.max_retime_s <= 0:
        parser.error("--max-retime-s must be positive")
    if args.plan_retiming_actions:
        if "random" in args.plan_retiming_actions and len(args.plan_retiming_actions) != 1:
            parser.error("'random' cannot be combined with deterministic retiming alternatives")
        for option in args.plan_retiming_actions:
            if option in ("none", "random"):
                continue
            seen_indices: set[int] = set()
            for action in option.split(","):
                parts = action.split(":")
                if len(parts) != 2:
                    parser.error(f"Invalid retiming alternative: {option}")
                try:
                    train_index, shift_s = map(int, parts)
                except ValueError:
                    parser.error(f"Invalid retiming alternative: {option}")
                if not 0 <= train_index < args.trains or not 0 < abs(shift_s) <= args.max_retime_s:
                    parser.error(
                        f"Retiming must target an existing service and fit --max-retime-s: {option}"
                    )
                if train_index in seen_indices:
                    parser.error(f"Retiming alternative repeats service {train_index}: {option}")
                seen_indices.add(train_index)
    minimum_headway_s = 240  # replay.py default; phase must be below its minimum headway.
    if any(
        phase < 0 or phase >= min(minimum_headway_s, frequency, args.window_s)
        for phase in args.phase_options_s
        for frequency in args.frequency_options_s
    ):
        parser.error(
            "phase options must be nonnegative and below minimum headway, frequency, and window"
        )
    if not args.collect:
        print(json.dumps(_select_candidates(rows, args), indent=2, sort_keys=True))
        return
    count = collect(args, rows)
    print(f"Collected {count} spatially separated switch cohorts in {args.output_root}")


if __name__ == "__main__":
    main()
