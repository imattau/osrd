"""Deterministic, bounded timetable departure-shift candidate generation.

Candidates contain edits only. A caller must apply each edit to the complete
snapshot timetable and accept it only after whole-timetable OSRD validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

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
    max_shift_seconds_by_schedule: Mapping[str, int] | None = None,
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
    per_schedule_maximum = max_shift_seconds_by_schedule or {}
    for schedule_id, maximum in per_schedule_maximum.items():
        if not schedule_id or isinstance(maximum, bool) or not 0 <= maximum <= 7200:
            raise ValueError("per-schedule shift limits must be integer values in [0, 7200]")
    unknown_limits = set(per_schedule_maximum) - set(schedule_start_times_ms)
    if unknown_limits:
        raise ValueError(f"Shift limits reference missing schedules: {sorted(unknown_limits)}")
    options_by_schedule = {
        schedule_id: _shift_options(
            delay_seconds,
            min(
                max_departure_shift_seconds,
                per_schedule_maximum.get(schedule_id, max_departure_shift_seconds),
            ),
            step_seconds,
        )
        for schedule_id in eligible
    }
    if not eligible or not any(options_by_schedule.values()):
        return []

    proposals: set[tuple[tuple[str, int], ...]] = set()
    for schedule_id in eligible:
        for shift_seconds in options_by_schedule[schedule_id]:
            proposals.add(((schedule_id, shift_seconds),))
    if len(eligible) > 1:
        common_options = set(options_by_schedule[eligible[0]])
        for schedule_id in eligible[1:]:
            common_options.intersection_update(options_by_schedule[schedule_id])
        for shift_seconds in sorted(common_options):
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


def generate_service_retiming_candidates(
    *,
    service_request_id: str,
    snapshot: SnapshotBinding,
    schedule_start_times_ms: Mapping[str, int],
    eligible_schedule_ids: Sequence[str],
    protected_schedule_ids: Sequence[str],
    max_edits: int = 12,
    max_shift_seconds: int,
    max_shift_seconds_by_schedule: Mapping[str, int] | None = None,
    max_advance_seconds_by_schedule: Mapping[str, int] | None = None,
    max_candidates: int,
    step_seconds: int = 60,
) -> list[DepartureShiftCandidate]:
    """Generate bounded positive and negative shifts for explicitly eligible services."""
    if not service_request_id:
        raise ValueError("service_request_id is required")
    if not 0 <= max_edits <= 12:
        raise ValueError("max_edits must be in [0, 12]")
    if not 0 <= max_shift_seconds <= 7200:
        raise ValueError("max_shift_seconds must be in [0, 7200]")
    if max_candidates < 1 or step_seconds < 1:
        raise ValueError("max_candidates and step_seconds must be positive")
    eligible = set(eligible_schedule_ids)
    protected = set(protected_schedule_ids)
    unknown = eligible - set(schedule_start_times_ms)
    if unknown:
        raise ValueError(f"Eligible schedules are missing from the snapshot: {sorted(unknown)}")
    unknown_protected = protected - set(schedule_start_times_ms)
    if unknown_protected:
        raise ValueError(
            f"Protected schedules are missing from the snapshot: {sorted(unknown_protected)}"
        )
    if eligible & protected:
        raise ValueError("Protected schedules cannot be retimed")
    schedule_shift_caps = dict(max_shift_seconds_by_schedule or {})
    schedule_advance_caps = dict(max_advance_seconds_by_schedule or {})
    if set(schedule_shift_caps) - eligible:
        raise ValueError("Per-schedule shift caps may only target eligible services")
    if set(schedule_advance_caps) - eligible:
        raise ValueError("Per-schedule advance caps may only target eligible services")
    if any(not 0 <= maximum <= max_shift_seconds for maximum in schedule_shift_caps.values()):
        raise ValueError("Per-schedule shift caps must be within the global shift limit")
    if any(not 0 <= maximum <= max_shift_seconds for maximum in schedule_advance_caps.values()):
        raise ValueError("Per-schedule advance caps must be within the global shift limit")
    eligible_ordered = sorted(eligible)
    magnitudes = (
        sorted(
            {
                min(step_seconds, max_shift_seconds),
                min(step_seconds * 2, max_shift_seconds),
                min(step_seconds * 5, max_shift_seconds),
                max_shift_seconds,
            }
        )
        if max_shift_seconds
        else []
    )
    schedule_magnitudes = {
        schedule_id: sorted(
            {
                min(step_seconds, maximum),
                min(step_seconds * 2, maximum),
                min(step_seconds * 5, maximum),
                maximum,
            }
        )
        if (maximum := schedule_shift_caps.get(schedule_id, max_shift_seconds))
        else []
        for schedule_id in eligible_ordered
    }
    schedule_advance_magnitudes = {
        schedule_id: sorted(
            {
                min(step_seconds, maximum),
                min(step_seconds * 2, maximum),
                min(step_seconds * 5, maximum),
                maximum,
            }
        )
        if (maximum := schedule_advance_caps.get(schedule_id, max_shift_seconds))
        else []
        for schedule_id in eligible_ordered
    }
    signed_options_by_schedule = {
        schedule_id: sorted(
            {
                *schedule_magnitudes[schedule_id],
                *(-magnitude for magnitude in schedule_advance_magnitudes[schedule_id]),
            }
        )
        for schedule_id in eligible_ordered
    }
    singles: list[tuple[tuple[str, int], ...]] = []
    if max_edits >= 1:
        for schedule_id in eligible_ordered:
            for shift in signed_options_by_schedule[schedule_id]:
                singles.append(((schedule_id, shift),))
    pairwise = (
        [
            ((left_id, left_shift), (right_id, right_shift))
            for left_index, left_id in enumerate(eligible_ordered)
            for right_id in eligible_ordered[left_index + 1 :]
            for left_shift in signed_options_by_schedule[left_id]
            for right_shift in signed_options_by_schedule[right_id]
        ]
        if max_edits >= 2
        else []
    )
    coordinated = [
        tuple(
            (
                schedule_id,
                direction
                * min(
                    magnitude,
                    schedule_advance_caps.get(schedule_id, max_shift_seconds)
                    if direction < 0
                    else schedule_shift_caps.get(schedule_id, max_shift_seconds),
                ),
            )
            for schedule_id in group
        )
        for edit_count in range(3, min(max_edits, len(eligible_ordered)) + 1)
        for group in combinations(eligible_ordered, edit_count)
        for magnitude in magnitudes
        for direction in (-1, 1)
        if all(
            (
                schedule_advance_caps.get(schedule_id, max_shift_seconds)
                if direction < 0
                else schedule_shift_caps.get(schedule_id, max_shift_seconds)
            )
            > 0
            for schedule_id in group
        )
    ]
    mixed_coordinated: list[tuple[tuple[str, int], ...]] = []
    for edit_count in range(3, min(max_edits, len(eligible_ordered)) + 1):
        groups = list(combinations(eligible_ordered, edit_count))
        group_limit = min(4, max_candidates, len(groups))
        if group_limit == 1:
            group_indices = [(len(groups) - 1) // 2]
        else:
            group_indices = sorted(
                {
                    round(index * (len(groups) - 1) / (group_limit - 1))
                    for index in range(group_limit)
                }
            )
        for group_index in group_indices:
            group = groups[group_index]
            sign_patterns: set[tuple[int, ...]] = set()
            for direction in (-1, 1):
                sign_patterns.add(
                    tuple(
                        direction if index % 2 == 0 else -direction
                        for index in range(edit_count)
                    )
                )
                for outlier_index in range(edit_count):
                    sign_patterns.add(
                        tuple(
                            direction if index == outlier_index else -direction
                            for index in range(edit_count)
                        )
                    )
                for split_index in range(1, edit_count):
                    sign_patterns.add(
                        tuple(
                            direction if index < split_index else -direction
                            for index in range(edit_count)
                        )
                    )
            for signs in sorted(sign_patterns):
                for magnitude in magnitudes:
                    proposal = tuple(
                        (
                            schedule_id,
                            sign
                            * min(
                                magnitude,
                                schedule_advance_caps.get(schedule_id, max_shift_seconds)
                                if sign < 0
                                else schedule_shift_caps.get(schedule_id, max_shift_seconds),
                            ),
                        )
                        for schedule_id, sign in zip(group, signs, strict=True)
                    )
                    if all(shift != 0 for _, shift in proposal):
                        mixed_coordinated.append(proposal)

    def proposal_key(proposal: tuple[tuple[str, int], ...]) -> tuple[object, ...]:
        return sum(abs(shift) for _, shift in proposal), proposal

    singles.sort(key=proposal_key)
    pairwise.sort(key=proposal_key)
    coordinated.sort(key=proposal_key)
    mixed_coordinated.sort(key=proposal_key)
    # Reserve validation budget across singleton, coordinated, pairwise, and
    # mixed-direction edits. No candidate exceeds max_edits.
    ordered = [()]
    seen = {()}
    for index in range(
        max(len(singles), len(pairwise), len(coordinated), len(mixed_coordinated))
    ):
        for bucket in (singles, coordinated, pairwise, mixed_coordinated):
            if index < len(bucket) and bucket[index] not in seen:
                ordered.append(bucket[index])
                seen.add(bucket[index])
                if len(ordered) >= max_candidates:
                    break
        if len(ordered) >= max_candidates:
            break
    candidates: list[DepartureShiftCandidate] = []
    for proposal in ordered:
        edits = tuple(
            DepartureShiftEdit(
                train_schedule_id=schedule_id,
                before_start_time_ms=schedule_start_times_ms[schedule_id],
                after_start_time_ms=schedule_start_times_ms[schedule_id] + shift * 1000,
                shift_seconds=shift,
            )
            for schedule_id, shift in proposal
        )
        identity = {
            "service_request_id": service_request_id,
            "snapshot": snapshot.model_dump(mode="json"),
            "edits": [asdict(edit) for edit in edits],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        candidates.append(
            DepartureShiftCandidate(
                candidate_id=f"service-plan-{digest}",
                incident_id=service_request_id,
                snapshot=snapshot,
                strategy="service_request_bounded_retiming_v1",
                edits=edits,
                total_shift_seconds=sum(abs(edit.shift_seconds) for edit in edits),
            )
        )
    return candidates
