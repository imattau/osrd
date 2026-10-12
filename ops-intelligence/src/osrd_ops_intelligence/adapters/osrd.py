"""Typed boundary for the OSRD validation operations used by a planner.

The first scaffold deliberately has no HTTP client implementation. The eventual
adapter must bind every result to the exact infrastructure and timetable versions
in a planning request and return OSRD's own validation outputs unchanged.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SnapshotRef:
    infra_id: str
    infra_version: str
    timetable_id: str
    timetable_revision: str
    osrd_version: str


@dataclass(frozen=True)
class CandidateValidation:
    pathfinding_valid: bool
    simulation_valid: bool
    conflict_count: int | None
    failure_reasons: tuple[str, ...] = ()

    @property
    def validated(self) -> bool:
        return (
            self.pathfinding_valid
            and self.simulation_valid
            and self.conflict_count == 0
            and not self.failure_reasons
        )


class OSRDClient(Protocol):
    """Minimal async interface needed by the future planning orchestrator."""

    async def validate_candidate(
        self,
        *,
        snapshot: SnapshotRef,
        candidate_payload: dict[str, object],
    ) -> CandidateValidation:
        """Validate a complete candidate timetable through OSRD."""
