"""Build a status-aware replay disturbance profile from Swiss Actual data v2 CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "osrd-empirical-origin-departure-delay-profile/v2"
ATTRIBUTION = "Open Data platform mobility Switzerland: Actual data v2"
TERMS = "opentransportdata.swiss file-based Open Data Terms of Use"
PRODUCT_FILTER = "PRODUKT_ID equals Zug (case-insensitive); other public transport modes excluded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        date_part, time_part = value.split(maxsplit=1)
        day, month, year = (int(part) for part in date_part.split("."))
        time_fields = time_part.split(":")
        hour, minute = int(time_fields[0]), int(time_fields[1])
        second = int(time_fields[2].split(".", maxsplit=1)[0]) if len(time_fields) > 2 else 0
        return datetime(year, month, day, hour, minute, second)
    except (IndexError, ValueError):
        return None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def _swiss_stop(row: dict[str, str]) -> bool:
    bpuic = (row.get("BPUIC") or "").strip()
    sloid = (row.get("SLOID") or "").strip().lower()
    return bpuic.startswith("85") or sloid.startswith("ch:")


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_profile(inputs: list[Path], output: Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one Swiss Actual data CSV is required")

    # Pick the earliest scheduled departure at a Swiss stop per service. Keep its
    # status even when it is not REAL so a later stop cannot silently become origin.
    first_departure: dict[tuple[str, str, str], tuple[datetime, str, datetime | None]] = {}
    dwell_samples: list[int] = []
    status_counts: Counter[str] = Counter()
    row_count = 0
    skipped_cancelled = 0
    non_rail_rows = 0
    non_swiss_rows = 0
    invalid_planned_departures = 0
    source_files: list[dict[str, Any]] = []

    for path in inputs:
        source_files.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "service_date": path.stem.split("_")[0],
            }
        )
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            required = {
                "BETRIEBSTAG",
                "FAHRT_BEZEICHNER",
                "BETREIBER_ID",
                "PRODUKT_ID",
                "FAELLT_AUS_TF",
                "BPUIC",
                "ANKUNFTSZEIT",
                "ABFAHRTSZEIT",
                "AB_PROGNOSE",
                "AB_PROGNOSE_STATUS",
                "DURCHFAHRT_TF",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Swiss Actual CSV {path} is missing fields: {', '.join(sorted(missing))}"
                )
            for row in reader:
                row_count += 1
                # Actual data v2 covers multiple public-transport modes. Only
                # railway journeys should condition the rail replay model.
                if (row.get("PRODUKT_ID") or "").strip().casefold() != "zug":
                    non_rail_rows += 1
                    continue
                status = (row.get("AB_PROGNOSE_STATUS") or "").strip().upper()
                status_counts[status or "EMPTY"] += 1
                if _truthy(row.get("FAELLT_AUS_TF")):
                    skipped_cancelled += 1
                    continue
                if not _swiss_stop(row):
                    non_swiss_rows += 1
                    continue
                if _truthy(row.get("DURCHFAHRT_TF")):
                    continue

                planned_arrival = _parse_time(row.get("ANKUNFTSZEIT"))
                planned_departure = _parse_time(row.get("ABFAHRTSZEIT"))
                if planned_arrival and planned_departure:
                    dwell = round((planned_departure - planned_arrival).total_seconds())
                    if 0 <= dwell <= 3600:
                        dwell_samples.append(dwell)

                if planned_departure is None:
                    invalid_planned_departures += 1
                    continue
                service_key = (
                    (row.get("BETRIEBSTAG") or "").strip(),
                    (row.get("BETREIBER_ID") or "").strip(),
                    (row.get("FAHRT_BEZEICHNER") or "").strip(),
                )
                if not all(service_key):
                    continue
                current = first_departure.get(service_key)
                if current is None or planned_departure < current[0]:
                    first_departure[service_key] = (
                        planned_departure,
                        status,
                        _parse_time(row.get("AB_PROGNOSE")),
                    )

    delays: list[int] = []
    origin_status_counts: Counter[str] = Counter()
    for planned_departure, status, observed_departure in first_departure.values():
        origin_status_counts[status or "EMPTY"] += 1
        # Only effective observed times become delay samples. Forecast and estimated
        # values are counted for coverage reporting but never treated as actual labels.
        if status != "REAL" or observed_departure is None:
            continue
        delay = round((observed_departure - planned_departure).total_seconds())
        if -86_400 <= delay <= 86_400:
            delays.append(delay)

    if len(delays) < 100:
        raise ValueError(f"Only {len(delays)} REAL first-departure samples; need at least 100")
    if not dwell_samples:
        raise ValueError("No valid planned dwell samples were found")

    profile = {
        "schema": PROFILE_SCHEMA,
        "source_files": source_files,
        "source_periods": sorted({entry["service_date"] for entry in source_files}),
        "license": TERMS,
        "attribution": ATTRIBUTION,
        "sampling_unit": (
            "earliest scheduled departure of a Zug journey at a Swiss stop per "
            "service-date/operator/journey; "
            "only AB_PROGNOSE_STATUS=REAL is used for delay samples"
        ),
        "dwell_sampling_unit": (
            "scheduled train departure minus arrival at non-through Swiss stops; 0-3600 seconds"
        ),
        "product_filter": PRODUCT_FILTER,
        "timezone": "Europe/Zurich; source clock values are local wall time",
        "delay_seconds": delays,
        "stop_dwell_seconds": dwell_samples,
        "sample_count": len(delays),
        "statistics": {
            "minimum_delay_seconds": min(delays),
            "maximum_delay_seconds": max(delays),
            "mean_delay_seconds": sum(delays) / len(delays),
            "positive_fraction": sum(value > 0 for value in delays) / len(delays),
            "zero_fraction": sum(value == 0 for value in delays) / len(delays),
            "delay_quantiles_seconds": {
                str(q): _quantile(delays, q) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)
            },
            "dwell_sample_count": len(dwell_samples),
            "dwell_quantiles_seconds": {
                str(q): _quantile(dwell_samples, q) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)
            },
        },
        "quality_report": {
            "input_rows": row_count,
            "non_rail_rows_skipped": non_rail_rows,
            "cancelled_rows_skipped": skipped_cancelled,
            "non_swiss_rows_skipped": non_swiss_rows,
            "invalid_planned_departure_rows": invalid_planned_departures,
            "candidate_service_count": len(first_departure),
            "first_departure_status_counts": dict(sorted(origin_status_counts.items())),
            "all_stop_departure_status_counts": dict(sorted(status_counts.items())),
            "forecast_and_estimated_rows_excluded_from_delay_labels": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "osrd-ops-swiss-actual-profile-manifest/v1",
        "profile_path": str(output),
        "profile_sha256": _sha256(output),
        "input_files": source_files,
        "profile_schema": PROFILE_SCHEMA,
        "attribution": ATTRIBUTION,
        "terms": TERMS,
        "product_filter": PRODUCT_FILTER,
        "sample_count": len(delays),
        "dwell_sample_count": len(dwell_samples),
        "quality_report": profile["quality_report"],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_archive_profile(
    archives: list[Path],
    output: Path,
    *,
    max_delay_samples: int = 250_000,
    max_dwell_samples: int = 250_000,
    seed: int = 20261017,
) -> dict[str, Any]:
    """Build a bounded-size profile from monthly ZIPs, processing one day at a time."""
    if not archives:
        raise ValueError("At least one Swiss Actual monthly ZIP is required")
    if max_delay_samples < 100 or max_dwell_samples < 100:
        raise ValueError("Profile sample caps must be at least 100")

    rng = random.Random(seed)
    delay_sample: list[int] = []
    dwell_sample: list[int] = []
    delay_seen = 0
    dwell_seen = 0
    source_files: list[dict[str, Any]] = []
    source_periods: set[str] = set()
    quality: Counter[str] = Counter()
    origin_statuses: Counter[str] = Counter()
    stop_statuses: Counter[str] = Counter()
    archive_sources: list[dict[str, Any]] = []

    def reservoir(sample: list[int], value: int, seen: int, limit: int) -> None:
        if len(sample) < limit:
            sample.append(value)
        else:
            index = rng.randrange(seen + 1)
            if index < limit:
                sample[index] = value

    with tempfile.TemporaryDirectory(prefix="osrd-swiss-actual-") as temporary:
        temp_root = Path(temporary)
        for archive in archives:
            archive_digest = _sha256(archive)
            year_match = re.search(r"20\d{2}", archive.name)
            archive_record = {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_digest,
            }
            if year_match:
                archive_record["source_url"] = (
                    "https://archive.opentransportdata.swiss/istdaten/"
                    f"{year_match.group()}/{archive.name}"
                )
            archive_sources.append(archive_record)
            with zipfile.ZipFile(archive) as bundle:
                members = sorted(
                    (
                        item
                        for item in bundle.infolist()
                        if not item.is_dir() and item.filename.lower().endswith(".csv")
                    ),
                    key=lambda item: item.filename,
                )
                if not members:
                    raise ValueError(f"No CSV members found in {archive}")
                for index, member in enumerate(members):
                    member_name = Path(member.filename).name
                    service_date = member_name.split("_")[0]
                    print(
                        f"Processing {service_date} from {archive.name} "
                        f"({index + 1}/{len(members)})",
                        flush=True,
                    )
                    day_path = temp_root / f"{service_date}_{index}.csv"
                    with bundle.open(member) as source, day_path.open("wb") as destination:
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                    day_profile_path = temp_root / f"day-{index}.json"
                    build_profile([day_path], day_profile_path)
                    day_profile = json.loads(day_profile_path.read_text(encoding="utf-8"))
                    source_record = day_profile["source_files"][0]
                    service_date = source_record.get("service_date", "")
                    source_periods.add(service_date)
                    source_files.append(
                        {
                            "archive": str(archive),
                            "archive_sha256": archive_digest,
                            "member": member.filename,
                            "member_sha256": source_record["sha256"],
                            "uncompressed_bytes": member.file_size,
                            "service_date": service_date,
                        }
                    )
                    for value in day_profile["delay_seconds"]:
                        reservoir(delay_sample, int(value), delay_seen, max_delay_samples)
                        delay_seen += 1
                    for value in day_profile["stop_dwell_seconds"]:
                        reservoir(dwell_sample, int(value), dwell_seen, max_dwell_samples)
                        dwell_seen += 1
                    quality.update(
                        {
                            key: value
                            for key, value in day_profile["quality_report"].items()
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                    )
                    origin_statuses.update(
                        day_profile["quality_report"]["first_departure_status_counts"]
                    )
                    stop_statuses.update(
                        day_profile["quality_report"]["all_stop_departure_status_counts"]
                    )
                    day_path.unlink()
                    day_profile_path.unlink()

    delay_quantiles = {
        str(q): _quantile(delay_sample, q) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)
    }
    dwell_quantiles = {
        str(q): _quantile(dwell_sample, q) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)
    }
    profile = {
        "schema": PROFILE_SCHEMA,
        "source_archives": archive_sources,
        "source_files": source_files,
        "source_periods": sorted(source_periods),
        "license": TERMS,
        "attribution": ATTRIBUTION,
        "sampling_unit": (
            "uniform reservoir sample of REAL first scheduled Swiss-stop rail departures; "
            "source files may contain estimated and forecast events, excluded from delay labels"
        ),
        "product_filter": PRODUCT_FILTER,
        "dwell_sampling_unit": (
            "uniform reservoir sample of scheduled rail dwell seconds at Swiss stops"
        ),
        "timezone": "Europe/Zurich; source clock values are local wall time",
        "delay_seconds": delay_sample,
        "stop_dwell_seconds": dwell_sample,
        "sample_count": len(delay_sample),
        "population_sample_count": delay_seen,
        "statistics": {
            "minimum_delay_seconds": min(delay_sample),
            "maximum_delay_seconds": max(delay_sample),
            "mean_delay_seconds": sum(delay_sample) / len(delay_sample),
            "positive_fraction": sum(value > 0 for value in delay_sample) / len(delay_sample),
            "zero_fraction": sum(value == 0 for value in delay_sample) / len(delay_sample),
            "delay_quantiles_seconds": delay_quantiles,
            "dwell_sample_count": len(dwell_sample),
            "dwell_population_sample_count": dwell_seen,
            "dwell_quantiles_seconds": dwell_quantiles,
        },
        "quality_report": {
            **dict(quality),
            "candidate_service_count": quality["candidate_service_count"],
            "first_departure_status_counts": dict(sorted(origin_statuses.items())),
            "all_stop_departure_status_counts": dict(sorted(stop_statuses.items())),
            "forecast_and_estimated_rows_excluded_from_delay_labels": True,
            "bounded_reservoir_sample": True,
            "reservoir_seed": seed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "osrd-ops-swiss-actual-profile-manifest/v1",
        "profile_path": str(output),
        "profile_sha256": _sha256(output),
        "archives": archive_sources,
        "input_files": source_files,
        "profile_schema": PROFILE_SCHEMA,
        "attribution": ATTRIBUTION,
        "terms": TERMS,
        "sample_count": len(delay_sample),
        "population_sample_count": delay_seen,
        "dwell_sample_count": len(dwell_sample),
        "dwell_population_sample_count": dwell_seen,
        "quality_report": profile["quality_report"],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--monthly-archives",
        action="store_true",
        help="Treat each input as a monthly ZIP and stream its daily CSVs into a bounded profile.",
    )
    args = parser.parse_args()
    result = (
        build_archive_profile(args.inputs, args.output)
        if args.monthly_archives
        else build_profile(args.inputs, args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
