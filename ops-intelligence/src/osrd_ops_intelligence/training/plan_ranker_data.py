"""Export OSRD-validated planning alternatives as grouped ranker examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, cast

FEATURE_SCHEMA = "osrd-ops-plan-candidate-features/v1"
DATASET_SCHEMA = "osrd-ops-plan-candidate-dataset/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _scenario_split(groups: list[str], test_fraction: float) -> dict[str, Any]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("Need at least two scenario groups for a grouped ranker dataset")
    test_count = max(1, min(len(unique_groups) - 1, round(len(unique_groups) * test_fraction)))
    test_groups = set(
        sorted(unique_groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())[
            :test_count
        ]
    )
    return {
        "method": (
            "stable hash split by scenario_id; all alternatives from a scenario stay together"
        ),
        "test_fraction_target": test_fraction,
        "train_scenario_groups": [group for group in unique_groups if group not in test_groups],
        "test_scenario_groups": sorted(test_groups),
    }


def _candidate_features(
    *, request: dict[str, Any], result: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    edits = candidate.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("Candidate edits must be a list")
    raw_edits = cast(list[Any], edits)
    if not all(isinstance(edit, dict) for edit in raw_edits):
        raise ValueError("Candidate edits must contain objects")
    edit_rows = cast(list[dict[str, Any]], raw_edits)
    shifts = [float(edit["shift_seconds"]) for edit in edit_rows]
    baseline = result.get("baseline", {}).get("validation", {})
    policy = request.get("policy", {})
    objective = policy.get("objective", "minimize_total_shift")
    features = {
        "edit_count": float(len(shifts)),
        "total_shift_seconds": float(candidate.get("total_shift_seconds", 0)),
        "mean_shift_seconds": sum(shifts) / len(shifts) if shifts else 0.0,
        "max_shift_seconds": max(shifts, default=0.0),
        "objective_score": float(candidate.get("objective_score", 0)),
        "incident_delay_seconds": float(request.get("incident", {}).get("delay_seconds") or 0),
        "max_departure_shift_seconds": float(policy.get("max_departure_shift_seconds", 0)),
        "policy_max_candidates": float(policy.get("max_candidates", 0)),
        "baseline_valid": float(
            bool(baseline.get("pathfinding_valid"))
            and bool(baseline.get("simulation_valid"))
            and baseline.get("conflict_count") == 0
            and not baseline.get("failure_reasons")
        ),
        "baseline_conflict_count": float(baseline.get("conflict_count") or 0),
        "objective_minimize_total_shift": float(objective == "minimize_total_shift"),
        "objective_minimize_changed_services": float(objective == "minimize_changed_services"),
        "objective_minimize_weighted_shift": float(objective == "minimize_weighted_shift"),
    }
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("Planning record produced a non-finite candidate feature")
    return features


def export_dataset(audit_db: Path, output_dir: Path, test_fraction: float = 0.2) -> dict[str, Any]:
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be in [0.05, 0.5]")
    try:
        connection = sqlite3.connect(audit_db)
        try:
            connection.execute("PRAGMA query_only = ON")
            records = connection.execute(
                """SELECT request_id, input_sha256, result_sha256, request_json, result_json
                   FROM plan_records ORDER BY request_id"""
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise ValueError(f"Could not read completed planning records: {error}") from error
    if not records:
        raise ValueError("The planning audit journal contains no completed plan records")

    rows: list[dict[str, Any]] = []
    source_hash = hashlib.sha256()
    for request_id, input_sha256, result_sha256, request_json, result_json in records:
        if _digest(request_json.encode()) != input_sha256:
            raise ValueError(f"Request digest mismatch for planning record {request_id}")
        if _digest(result_json.encode()) != result_sha256:
            raise ValueError(f"Result digest mismatch for planning record {request_id}")
        source_hash.update(f"{request_id}:{input_sha256}:{result_sha256}\n".encode())
        request = json.loads(request_json)
        result = json.loads(result_json)
        scenario_group = str(request["scenario_id"])
        candidates = [*result.get("candidates", []), *result.get("rejected_candidates", [])]
        for candidate in candidates:
            validation = candidate.get("validation", {})
            rows.append(
                {
                    "schema": DATASET_SCHEMA,
                    "feature_schema": FEATURE_SCHEMA,
                    "scenario_group": scenario_group,
                    "request_id": request_id,
                    "candidate_id": candidate["candidate_id"],
                    "features": _candidate_features(
                        request=request, result=result, candidate=candidate
                    ),
                    "targets": {
                        "osrd_validated": bool(
                            validation.get("pathfinding_valid")
                            and validation.get("simulation_valid")
                            and validation.get("conflict_count") == 0
                            and not validation.get("failure_reasons")
                        ),
                        "conflict_count": validation.get("conflict_count"),
                        "failure_reason_count": len(validation.get("failure_reasons", [])),
                    },
                }
            )
    if not rows:
        raise ValueError("Planning records contain no edited candidates to export")

    split = _scenario_split([row["scenario_group"] for row in rows], test_fraction)
    train_groups = set(split["train_scenario_groups"])
    test_groups = set(split["test_scenario_groups"])
    content = "".join(_canonical_json(row) + "\n" for row in rows).encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "candidates.jsonl"
    dataset_path.write_bytes(content)
    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "dataset_file": dataset_path.name,
        "dataset_sha256": _digest(content),
        "source_record_count": len(records),
        "source_records_sha256": source_hash.hexdigest(),
        "candidate_count": len(rows),
        "scenario_group_count": len(train_groups | test_groups),
        "train_candidate_count": sum(row["scenario_group"] in train_groups for row in rows),
        "test_candidate_count": sum(row["scenario_group"] in test_groups for row in rows),
        "split": split,
        "target_definition": (
            "Candidate passes OSRD pathfinding, complete simulation, and has zero conflicts."
        ),
        "limitations": [
            (
                "Only alternatives selected for OSRD validation are represented; "
                "selection bias remains."
            ),
            "The target is validator feasibility, not operator preference or service utility.",
            "Use the learned score only to prioritize validation; OSRD remains authoritative.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-db", type=Path, default=Path("data/plan-audit.sqlite3"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export_dataset(args.audit_db, args.output_dir, args.test_fraction)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
