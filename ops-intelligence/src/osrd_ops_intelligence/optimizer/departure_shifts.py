"""Deterministic, bounded timetable departure-shift candidate generation.

Candidates contain edits only. A caller must apply each edit to the complete
snapshot timetable and accept it only after whole-timetable OSRD validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from osrd_ops_intelligence.api.contracts import SnapshotBinding


@dataclass(frozen=True)
class DepartureShiftEdit:
    train_schedule_id: str
    before_start_time_ms: int
    after_start_time_ms: int
    shift_seconds: int


@dataclass(frozen=True)
class DepartureShiftCandidate:
    candidate_id: str
    incident_id: str
    snapshot: SnapshotBinding
    strategy: str
    edits: tuple[DepartureShiftEdit, ...]
    total_shift_seconds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "incident_id": self.incident_id,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "strategy": self.strategy,
            "edits": [asdict(edit) for edit in self.edits],
            "total_shift_seconds": self.total_shift_seconds,
            "requires_osrd_validation": True,
        }


def _shift_options(delay_seconds: int, maximum_seconds: int, step_seconds: int) -> list[int]:
    if maximum_seconds <= 0:
        return []
    cap = min(delay_seconds, maximum_seconds)
    return sorted(
        {min(step_seconds, cap), min(step_seconds * 2, cap), min(step_seconds * 5, cap), cap}
    )


def generate_departure_shift_candidates(
    *,
    incident_id: str,
    snapshot: SnapshotBinding,
    schedule_start_times_ms: Mapping[str, int],
    affected_schedule_ids: Sequence[str],
    protected_schedule_ids: Sequence[str],
    delay_seconds: int,
    max_departure_shift_seconds: int,
    max_candidates: int,
    step_seconds: int = 60,
) -> list[DepartureShiftCandidate]:
    """Propose small positive retimings for delayed trains, never protected ones.

    The output includes single-train variants and coordinated variants that
    shift all eligible affected trains by the same amount. Candidates are
    ordered by fewest edits, then least total shift, then schedule IDs.
    """

    if not incident_id:
        raise ValueError("incident_id is required")
    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")
    if max_departure_shift_seconds < 0:
        raise ValueError("max_departure_shift_seconds cannot be negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if step_seconds < 1:
        raise ValueError("step_seconds must be positive")
    if any(isinstance(value, bool) for value in schedule_start_times_ms.values()):
        raise ValueError("schedule start times must be integer milliseconds")

    affected = set(affected_schedule_ids)
    unknown = affected - set(schedule_start_times_ms)
    if unknown:
        raise ValueError(f"Affected schedules are missing from the snapshot: {sorted(unknown)}")
    eligible = sorted(affected - set(protected_schedule_ids))
    options = _shift_options(delay_seconds, max_departure_shift_seconds, step_seconds)
    if not eligible or not options:
        return []

    proposals: set[tuple[tuple[str, int], ...]] = set()
    for schedule_id in eligible:
        for shift_seconds in options:
            proposals.add(((schedule_id, shift_seconds),))
    if len(eligible) > 1:
        for shift_seconds in options:
            proposals.add(tuple((schedule_id, shift_seconds) for schedule_id in eligible))

    ordered = sorted(proposals, key=lambda edits: (len(edits), sum(v for _, v in edits), edits))
    candidates: list[DepartureShiftCandidate] = []
    for proposal in ordered[:max_candidates]:
        edits = tuple(
            DepartureShiftEdit(
                train_schedule_id=schedule_id,
                before_start_time_ms=schedule_start_times_ms[schedule_id],
                after_start_time_ms=schedule_start_times_ms[schedule_id] + shift_seconds * 1000,
                shift_seconds=shift_seconds,
            )
            for schedule_id, shift_seconds in proposal
        )
        identity = {
            "incident_id": incident_id,
            "snapshot": snapshot.model_dump(mode="json"),
            "edits": [asdict(edit) for edit in edits],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        candidates.append(
            DepartureShiftCandidate(
                candidate_id=f"departure-shift-{digest}",
                incident_id=incident_id,
                snapshot=snapshot,
                strategy="bounded_departure_shift_v1",
                edits=edits,
                total_shift_seconds=sum(edit.shift_seconds for edit in edits),
            )
        )
    return candidates
