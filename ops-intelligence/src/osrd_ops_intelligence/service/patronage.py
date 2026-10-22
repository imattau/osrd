"""Transparent demand-coverage calculations for candidate station calls.

This module does not infer passenger journeys or predict passenger benefit. It measures
the share of supplied station/time-bucket patronage represented by at least one candidate
call at the same OSRD operational point during that bucket.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from zoneinfo import ZoneInfo

from osrd_ops_intelligence.api.contracts import (
    DemandCoverage,
    PatronageProfile,
    StationCallTiming,
)


def score_station_call_coverage(
    profile: PatronageProfile,
    calls: list[StationCallTiming],
    *,
    active_dates: set[date] | None = None,
) -> DemandCoverage:
    """Score demand buckets touched by at least one time-matched candidate call.

    Calls are converted to the profile's timezone. A bucket is counted once even if
    multiple candidate trains call within it, preventing frequency from multiplying the
    profile's observed demand. Boarding demand uses simulated departure time, alighting
    demand uses simulated arrival time, and termini are treated directionally. Unmapped
    calls are reported rather than silently ignored.
    """
    zone = ZoneInfo(profile.timezone)
    if active_dates is None:
        active_dates = {
            event_time.astimezone(zone).date()
            for call in calls
            for event_time in (call.arrival_time, call.departure_time)
        }
    active_weekdays = Counter(day.weekday() for day in active_dates)
    station_by_id = {station.operational_point_id: station for station in profile.stations}
    matched_boardings: set[tuple[str, int, int, int]] = set()
    matched_alightings: set[tuple[str, int, int, int]] = set()
    unmapped_calls = 0
    unmatched_time_buckets = 0
    for call in calls:
        station = station_by_id.get(call.operational_point_id)
        if station is None:
            unmapped_calls += 1
            continue
        for event_time, allowed, matched_set in (
            (call.departure_time, call.boarding_allowed, matched_boardings),
            (call.arrival_time, call.alighting_allowed, matched_alightings),
        ):
            if not allowed:
                continue
            local = event_time.astimezone(zone)
            weekday = local.weekday()
            minute = local.hour * 60 + local.minute
            event_matched = False
            for bucket in station.buckets:
                if (
                    bucket.weekday == weekday
                    and bucket.start_minute
                    <= minute
                    < bucket.start_minute + bucket.duration_minutes
                ):
                    matched_set.add(
                        (
                            station.operational_point_id,
                            bucket.weekday,
                            bucket.start_minute,
                            bucket.duration_minutes,
                        )
                    )
                    event_matched = True
            if not event_matched:
                unmatched_time_buckets += 1

    covered_boardings = 0.0
    covered_alightings = 0.0
    total_boardings = 0.0
    total_alightings = 0.0
    matched_bucket_count = 0
    total_bucket_count = 0
    for station in profile.stations:
        for bucket in station.buckets:
            day_weight = active_weekdays.get(bucket.weekday, 0)
            if not day_weight:
                continue
            total_bucket_count += 1
            total_boardings += bucket.mean_boardings * day_weight
            total_alightings += bucket.mean_alightings * day_weight
            key = (
                station.operational_point_id,
                bucket.weekday,
                bucket.start_minute,
                bucket.duration_minutes,
            )
            if key in matched_boardings:
                covered_boardings += bucket.mean_boardings * day_weight
            if key in matched_alightings:
                covered_alightings += bucket.mean_alightings * day_weight
            if key in matched_boardings or key in matched_alightings:
                matched_bucket_count += 1

    return DemandCoverage(
        dataset_id=profile.dataset_id,
        covered_boardings=covered_boardings,
        covered_alightings=covered_alightings,
        total_boardings=total_boardings,
        total_alightings=total_alightings,
        boarding_coverage=(covered_boardings / total_boardings if total_boardings else None),
        alighting_coverage=(covered_alightings / total_alightings if total_alightings else None),
        combined_event_coverage=(
            (covered_boardings + covered_alightings) / (total_boardings + total_alightings)
            if total_boardings + total_alightings
            else None
        ),
        matched_bucket_count=matched_bucket_count,
        total_bucket_count=total_bucket_count,
        unmatched_call_count=unmapped_calls + unmatched_time_buckets,
        unmapped_station_call_count=unmapped_calls,
        no_matching_time_bucket_event_count=unmatched_time_buckets,
        active_service_day_count=len(active_dates),
    )
