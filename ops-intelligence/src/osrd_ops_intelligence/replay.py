"""Generate deterministic OSRD replay episodes for the first training-data slice.

This is an offline data-collection tool. It creates isolated calendar timetables
and records OSRD's simulation and conflict outputs; it never applies an
operational plan to an existing timetable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OSRD_COMMIT = "26d19cb55bf18a26d2f0da7a9fc586c310d3ecf0"
SCHEMA_SHA256 = "2164e74a522514e92c0b6fb6f252c7132e174b5a658e7aedf290fbf8e8d51b6b"
EPISODE_SCHEMA = "osrd-ops-replay-episode/v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "data_contracts" / "osrd" / OSRD_COMMIT / "openapi.yaml"
)


class Editoast:
    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-osrd-skip-authz": "true",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Editoast {method} {path}: HTTP {exc.code}: {detail}") from exc


def make_schedule(
    *,
    index: int,
    seed: int,
    departure_ms: int,
    rolling_stock: str,
    start_track: str,
    start_offset: int,
    end_track: str,
    end_offset: int,
    dwell_seconds: int,
) -> dict[str, Any]:
    return {
        "train_name": f"replay-{seed}-{index:03d}",
        "rolling_stock_name": rolling_stock,
        "start_time": departure_ms,
        "comfort": "STANDARD",
        "constraint_distribution": "STANDARD",
        "initial_speed": 0,
        "labels": ["ops-intelligence", f"seed:{seed}"],
        "options": {"use_electrical_profiles": False},
        "path": [
            {
                "key": "origin",
                "location": {"type": "track_offset", "track": start_track, "offset": start_offset},
            },
            {
                "key": "destination",
                "location": {"type": "track_offset", "track": end_track, "offset": end_offset},
            },
        ],
        "power_restrictions": [],
        "schedule": [
            {"at": "origin"},
            {"at": "destination", "stop_for": f"PT{dwell_seconds}S"},
        ],
    }


def collect(args: argparse.Namespace) -> Path:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Pinned Editoast schema is missing: {SCHEMA_PATH}")
    actual_schema_hash = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    if actual_schema_hash != SCHEMA_SHA256:
        raise RuntimeError(
            "Pinned Editoast schema hash mismatch: "
            f"expected {SCHEMA_SHA256}, got {actual_schema_hash}"
        )
    client = Editoast(args.editoast_url)
    rng = random.Random(args.seed)
    now_ms = int(time.time() * 1000) + 60_000
    created_at = datetime.now(UTC).isoformat()
    episodes: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        timetable = client.request("POST", "/timetable", {"timetable_type": "CALENDAR"})
        timetable_id = timetable["timetable_id"]
        schedule_set = client.request(
            "POST",
            "/train_schedule_sets",
            {
                "description": f"ops-intelligence replay seed={args.seed} episode={episode_index}",
                "name": f"ops-replay-{args.seed}-{episode_index}",
                "published": False,
                "timetable_type": "CALENDAR",
            },
        )
        set_id = schedule_set["id"]
        client.request(
            "POST",
            f"/timetable/{timetable_id}/train_schedule_sets",
            {"train_schedule_set_ids": [set_id]},
        )

        schedules = []
        perturbations = []
        base_departure = now_ms + episode_index * 3_600_000
        for train_index in range(args.trains):
            # Baseline and perturbed departures are sampled from a fixed seed.
            departure_shift_s = (
                0
                if train_index == 0
                else rng.randint(-args.max_departure_shift_s, args.max_departure_shift_s)
            )
            dwell_s = 0 if train_index == 0 else rng.randint(0, args.max_dwell_s)
            schedules.append(
                make_schedule(
                    index=train_index,
                    seed=args.seed + episode_index,
                    departure_ms=base_departure
                    + train_index * args.headway_s * 1000
                    + departure_shift_s * 1000,
                    rolling_stock=args.rolling_stock,
                    start_track=args.start_track,
                    start_offset=args.start_offset,
                    end_track=args.end_track,
                    end_offset=args.end_offset,
                    dwell_seconds=dwell_s,
                )
            )
            perturbations.append(
                {
                    "train_index": train_index,
                    "departure_shift_s": departure_shift_s,
                    "destination_dwell_s": dwell_s,
                }
            )

        created_schedules = client.request(
            "POST", f"/train_schedule_sets/{set_id}/train_schedules", schedules
        )
        ids = [row["id"] for row in created_schedules]
        summaries = client.request(
            "POST",
            "/train_schedules/simulation_summary",
            {"infra_id": args.infra_id, "timetable_id": timetable_id, "ids": ids},
        )
        conflicts = client.request(
            "GET", f"/timetable/{timetable_id}/conflicts?infra_id={args.infra_id}"
        )
        simulation_labels: dict[str, dict[str, Any]] = {}
        for schedule_id in ids:
            result = summaries.get(str(schedule_id), {}).get("train_schedule", {})
            simulation_labels[str(schedule_id)] = {
                "status": result.get("status", "unknown"),
                "travel_time_ms": result.get("time"),
                "energy_consumption_kwh": result.get("energy_consumption"),
                "path_item_times_final_ms": result.get("path_item_times_final"),
                "path_item_times_base_ms": result.get("path_item_times_base"),
            }
        episodes.append(
            {
                "schema": EPISODE_SCHEMA,
                "episode_id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"osrd-replay:{args.seed}:{episode_index}")
                ),
                "created_at": created_at,
                "seed": args.seed,
                "episode_index": episode_index,
                "snapshot": {
                    "osrd_commit": OSRD_COMMIT,
                    "openapi_sha256": SCHEMA_SHA256,
                    "infra_id": args.infra_id,
                    "timetable_id": timetable_id,
                    "train_schedule_set_id": set_id,
                },
                "inputs": {"schedule_payloads": schedules, "perturbations": perturbations},
                "outputs": {
                    "schedule_ids": ids,
                    "simulation_summaries": summaries,
                    "conflicts": conflicts,
                },
                "labels": {
                    "simulation_failed_train_ids": [
                        int(k)
                        for k, result in summaries.items()
                        if result.get("train_schedule", {}).get("status") not in (None, "success")
                    ],
                    "simulation_by_train_schedule_id": simulation_labels,
                    "conflict_count": len(conflicts),
                    "has_conflicts": bool(conflicts),
                },
            }
        )
        print(
            f"episode {episode_index + 1}/{args.episodes}: "
            f"timetable={timetable_id}, trains={len(ids)}, conflicts={len(conflicts)}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "osrd-ops-replay-manifest/v1",
        "created_at": created_at,
        "seed": args.seed,
        "episode_count": len(episodes),
        "episode_file": "episodes.jsonl",
        "format": "jsonl",
        "osrd_commit": OSRD_COMMIT,
        "openapi_sha256": SCHEMA_SHA256,
        "parameters": vars(args),
        "label_definitions": {
            "simulation_failed_train_ids": (
                "IDs whose summary status is not success; missing status is unknown."
            ),
            "conflict_count": "Number of timetable conflict records returned by Editoast.",
            "has_conflicts": "Whether conflict_count is greater than zero.",
        },
    }
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, sort_keys=True, separators=(",", ":")) + "\n")
    manifest["episodes_sha256"] = hashlib.sha256(
        (output / "episodes.jsonl").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editoast-url", default="http://localhost:4000/api")
    parser.add_argument("--infra-id", type=int, required=True)
    parser.add_argument("--rolling-stock", required=True)
    parser.add_argument("--start-track", required=True)
    parser.add_argument("--start-offset", type=int, required=True)
    parser.add_argument("--end-track", required=True)
    parser.add_argument("--end-offset", type=int, required=True)
    parser.add_argument("--output", default="data/replays/run-001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--trains", type=int, default=8)
    parser.add_argument("--headway-s", type=int, default=240)
    parser.add_argument("--max-departure-shift-s", type=int, default=180)
    parser.add_argument("--max-dwell-s", type=int, default=120)
    args = parser.parse_args()
    if (
        min(args.episodes, args.trains, args.headway_s) <= 0
        or min(args.max_departure_shift_s, args.max_dwell_s) < 0
    ):
        parser.error(
            "episodes, trains, and headway must be positive; perturbation maxima cannot be negative"
        )
    print(f"Wrote replay data to {collect(args)}")


if __name__ == "__main__":
    main()
