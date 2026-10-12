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
from urllib.parse import quote

OSRD_COMMIT = "26d19cb55bf18a26d2f0da7a9fc586c310d3ecf0"
SCHEMA_SHA256 = "2164e74a522514e92c0b6fb6f252c7132e174b5a658e7aedf290fbf8e8d51b6b"
EPISODE_SCHEMA = "osrd-ops-replay-episode/v2"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "data_contracts" / "osrd" / OSRD_COMMIT / "openapi.yaml"
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


def create_arm(
    client: Editoast,
    *,
    args: argparse.Namespace,
    arm: str,
    episode_index: int,
    schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    timetable = client.request("POST", "/timetable", {"timetable_type": "CALENDAR"})
    timetable_id = timetable["timetable_id"]
    schedule_set = client.request(
        "POST",
        "/train_schedule_sets",
        {
            "description": (f"ops-intelligence {arm} seed={args.seed} episode={episode_index}"),
            "name": f"ops-replay-{args.seed}-{episode_index}-{arm}",
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
    created = client.request("POST", f"/train_schedule_sets/{set_id}/train_schedules", schedules)
    ids = [row["id"] for row in created]
    summaries = client.request(
        "POST",
        "/train_schedules/simulation_summary",
        {"infra_id": args.infra_id, "timetable_id": timetable_id, "ids": ids},
    )
    conflicts = client.request(
        "GET", f"/timetable/{timetable_id}/conflicts?infra_id={args.infra_id}"
    )
    return {
        "timetable_id": timetable_id,
        "train_schedule_set_id": set_id,
        "schedule_ids": ids,
        "simulation_summaries": summaries,
        "conflicts": conflicts,
    }


def conflict_participants(conflicts: list[dict[str, Any]]) -> set[int]:
    return {
        int(train_id["train_schedule_id"])
        for conflict in conflicts
        for train_id in conflict.get("train_ids", [])
        if "train_schedule_id" in train_id
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
    infra = client.request("GET", f"/infra/{args.infra_id}")
    rolling_stock = client.request(
        "GET", f"/rolling_stock/name/{quote(args.rolling_stock, safe='')}"
    )
    headway_options = args.headway_options_s or [args.headway_s]
    scenario_group = args.scenario_group or (
        f"infra-{args.infra_id}-{args.start_track}-{args.end_track}-headways-"
        + "-".join(str(value) for value in sorted(set(headway_options)))
    )
    episodes: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        baseline_schedules: list[dict[str, Any]] = []
        dwell_inputs: list[dict[str, int]] = []
        episode_headway_s = rng.choice(headway_options)
        base_departure = now_ms + episode_index * 3_600_000
        for train_index in range(args.trains):
            departure_shift_s = (
                0
                if train_index == 0
                else rng.randint(-args.max_departure_shift_s, args.max_departure_shift_s)
            )
            dwell_s = 0 if train_index == 0 else rng.randint(0, args.max_dwell_s)
            baseline_schedules.append(
                make_schedule(
                    index=train_index,
                    seed=args.seed + episode_index,
                    departure_ms=base_departure
                    + train_index * episode_headway_s * 1000
                    + departure_shift_s * 1000,
                    rolling_stock=args.rolling_stock,
                    start_track=args.start_track,
                    start_offset=args.start_offset,
                    end_track=args.end_track,
                    end_offset=args.end_offset,
                    dwell_seconds=dwell_s,
                )
            )
            dwell_inputs.append(
                {
                    "train_index": train_index,
                    "background_departure_shift_s": departure_shift_s,
                    "destination_dwell_s": dwell_s,
                }
            )

        treatment_schedules = json.loads(json.dumps(baseline_schedules))
        incident_train_index = rng.randrange(args.trains)
        incident_delay_s = rng.randint(args.min_incident_delay_s, args.max_incident_delay_s)
        treatment_schedules[incident_train_index]["start_time"] += incident_delay_s * 1000

        baseline_arm = create_arm(
            client,
            args=args,
            arm="baseline",
            episode_index=episode_index,
            schedules=baseline_schedules,
        )
        treatment_arm = create_arm(
            client,
            args=args,
            arm="treatment",
            episode_index=episode_index,
            schedules=treatment_schedules,
        )
        baseline_ids = baseline_arm["schedule_ids"]
        treatment_ids = treatment_arm["schedule_ids"]
        baseline_summaries = baseline_arm["simulation_summaries"]
        treatment_summaries = treatment_arm["simulation_summaries"]
        baseline_conflicted = conflict_participants(baseline_arm["conflicts"])
        treatment_conflicted = conflict_participants(treatment_arm["conflicts"])

        simulation_labels: dict[str, dict[str, Any]] = {}
        paired_labels: list[dict[str, Any]] = []
        for index, (base_id, treatment_id) in enumerate(
            zip(baseline_ids, treatment_ids, strict=True)
        ):
            base_result = baseline_summaries.get(str(base_id), {}).get("train_schedule", {})
            result = treatment_summaries.get(str(treatment_id), {}).get("train_schedule", {})
            baseline_arrival_ms = (
                baseline_schedules[index]["start_time"]
                + (base_result.get("path_item_times_final") or [None])[-1]
                if base_result.get("path_item_times_final")
                else None
            )
            treatment_arrival_ms = (
                treatment_schedules[index]["start_time"]
                + (result.get("path_item_times_final") or [None])[-1]
                if result.get("path_item_times_final")
                else None
            )
            base_conflict = base_id in baseline_conflicted
            treatment_conflict = treatment_id in treatment_conflicted
            paired_labels.append(
                {
                    "train_index": index,
                    "baseline_schedule_id": base_id,
                    "treatment_schedule_id": treatment_id,
                    "baseline_status": base_result.get("status", "unknown"),
                    "treatment_status": result.get("status", "unknown"),
                    "baseline_arrival_ms": baseline_arrival_ms,
                    "treatment_arrival_ms": treatment_arrival_ms,
                    "arrival_delta_ms": (
                        treatment_arrival_ms - baseline_arrival_ms
                        if baseline_arrival_ms is not None and treatment_arrival_ms is not None
                        else None
                    ),
                    "travel_time_delta_ms": (
                        result["time"] - base_result["time"]
                        if result.get("time") is not None and base_result.get("time") is not None
                        else None
                    ),
                    "energy_delta_kwh": (
                        result["energy_consumption"] - base_result["energy_consumption"]
                        if result.get("energy_consumption") is not None
                        and base_result.get("energy_consumption") is not None
                        else None
                    ),
                    "baseline_conflict": base_conflict,
                    "treatment_conflict": treatment_conflict,
                    "new_conflict": treatment_conflict and not base_conflict,
                }
            )
            simulation_labels[str(treatment_id)] = {
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
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"osrd-replay:v2:{scenario_group}:{args.seed}:{episode_index}",
                    )
                ),
                "scenario_group": scenario_group,
                "created_at": created_at,
                "seed": args.seed,
                "episode_index": episode_index,
                "snapshot": {
                    "osrd_commit": OSRD_COMMIT,
                    "openapi_sha256": SCHEMA_SHA256,
                    "infra_id": args.infra_id,
                    "baseline_timetable_id": baseline_arm["timetable_id"],
                    "baseline_train_schedule_set_id": baseline_arm["train_schedule_set_id"],
                    "treatment_timetable_id": treatment_arm["timetable_id"],
                    "treatment_train_schedule_set_id": treatment_arm["train_schedule_set_id"],
                    "infra_version": infra.get("version"),
                    "infra_generated_version": infra.get("generated_version"),
                    "rolling_stock_id": rolling_stock.get("id"),
                    "rolling_stock_version": rolling_stock.get("version"),
                },
                "inputs": {
                    "scenario": {
                        "scenario_group": scenario_group,
                        "headway_s": episode_headway_s,
                        "headway_options_s": headway_options,
                        "infra_id": args.infra_id,
                        "start_track": args.start_track,
                        "start_offset": args.start_offset,
                        "end_track": args.end_track,
                        "end_offset": args.end_offset,
                        "rolling_stock": args.rolling_stock,
                    },
                    "schedule_payloads": treatment_schedules,
                    "baseline_schedule_payloads": baseline_schedules,
                    "background_perturbations": dwell_inputs,
                    "incident": {
                        "type": "departure_delay",
                        "train_index": incident_train_index,
                        "delay_s": incident_delay_s,
                    },
                },
                "outputs": {
                    "schedule_ids": treatment_ids,
                    "simulation_summaries": treatment_summaries,
                    "conflicts": treatment_arm["conflicts"],
                    "paired": {
                        "baseline": baseline_arm,
                        "treatment": treatment_arm,
                    },
                },
                "labels": {
                    "simulation_failed_train_ids": [
                        int(k)
                        for k, result in treatment_summaries.items()
                        if result.get("train_schedule", {}).get("status") not in (None, "success")
                    ],
                    "simulation_by_train_schedule_id": simulation_labels,
                    "paired_by_train_index": paired_labels,
                    "conflict_count": len(treatment_arm["conflicts"]),
                    "has_conflicts": bool(treatment_arm["conflicts"]),
                    "new_conflict_count": sum(
                        int(label["new_conflict"]) for label in paired_labels
                    ),
                },
            }
        )
        print(
            f"episode {episode_index + 1}/{args.episodes}: "
            f"scenario={scenario_group}, trains={len(treatment_ids)}, "
            f"new_conflicts={sum(int(label['new_conflict']) for label in paired_labels)}"
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
            "arrival_delta_ms": "Treatment minus matched baseline simulated arrival time.",
            "new_conflict": "Train is conflict-involved in treatment but not in baseline.",
        },
        "paired_arms": True,
        "scenario_group": scenario_group,
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
    parser.add_argument("--headway-options-s", type=int, nargs="+")
    parser.add_argument("--max-departure-shift-s", type=int, default=180)
    parser.add_argument("--max-dwell-s", type=int, default=120)
    parser.add_argument("--min-incident-delay-s", type=int, default=60)
    parser.add_argument("--max-incident-delay-s", type=int, default=300)
    parser.add_argument("--scenario-group")
    args = parser.parse_args()
    if (
        min(args.episodes, args.trains, args.headway_s) <= 0
        or min(args.max_departure_shift_s, args.max_dwell_s) < 0
        or args.min_incident_delay_s <= 0
        or args.max_incident_delay_s < args.min_incident_delay_s
    ):
        parser.error(
            "episodes, trains, and headway must be positive; perturbation maxima cannot be negative"
        )
    if args.headway_options_s and min(args.headway_options_s) <= 0:
        parser.error("headway options must all be positive")
    print(f"Wrote replay data to {collect(args)}")


if __name__ == "__main__":
    main()
