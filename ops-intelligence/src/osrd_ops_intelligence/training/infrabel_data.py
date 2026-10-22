"""Acquire and normalize Infrabel's public monthly passenger-punctuality events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CATALOG_URL = (
    "https://opendata.infrabel.be/api/explore/v2.1/catalog/datasets/"
    "stiptheid-gegevens-maandelijksebestanden/records"
)
DATASET_URL = (
    "https://opendata.infrabel.be/explore/dataset/stiptheid-gegevens-maandelijksebestanden/"
)
LICENSE = "CC0"
ATTRIBUTION = "Infrabel Open Data: Monthly raw passenger train punctuality files"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
DOWNLOAD_USER_AGENT = "osrd-ops-intelligence/0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _catalogue_files() -> dict[str, str]:
    files: dict[str, str] = {}
    offset = 0
    while True:
        request = urllib.request.Request(
            f"{CATALOG_URL}?limit=100&offset={offset}&order_by=mois",
            headers={"User-Agent": DOWNLOAD_USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Infrabel catalogue response has no results list")
        for record in results:
            if isinstance(record, dict) and record.get("mois") and record.get("link_to_data"):
                files[str(record["mois"])] = str(record["link_to_data"])
        offset += len(results)
        if not results or offset >= int(payload.get("total_count", offset)):
            return files


def _head(url: str) -> tuple[int | None, str | None, str | None]:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": DOWNLOAD_USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw_length = response.headers.get("Content-Length")
        return (
            int(raw_length) if raw_length and raw_length.isdigit() else None,
            response.headers.get("ETag"),
            response.headers.get("Last-Modified"),
        )


def _download(url: str, destination: Path, expected_size: int | None) -> None:
    partial = destination.with_name(f".{destination.name}.part")
    retryable = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": DOWNLOAD_USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        partial.unlink(missing_ok=True)
                        continue
                else:
                    offset = 0
                mode = "ab" if append else "wb"
                total = expected_size or offset + int(response.headers.get("Content-Length", 0))
                started = time.monotonic()
                last_report = started
                size = offset
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                        current = time.monotonic()
                        if current - last_report >= 3:
                            elapsed = max(current - started, 1e-6)
                            rate = (size - offset) / elapsed
                            if total:
                                remaining = max(total - size, 0)
                                eta = remaining / max(rate, 1)
                                print(
                                    f"{destination.name}: {size:,}/{total:,} bytes "
                                    f"({100 * size / total:.1f}%), "
                                    f"{rate / 1024:.0f} KiB/s, ETA {eta:.0f}s",
                                    flush=True,
                                )
                            else:
                                print(f"{destination.name}: {size:,} bytes", flush=True)
                            last_report = current
                    output.flush()
                    os.fsync(output.fileno())
        except urllib.error.HTTPError as error:
            if error.code == 416 and expected_size == offset:
                pass
            elif error.code not in retryable or attempt == 6:
                raise
            else:
                time.sleep(min(2**attempt, 30))
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 6:
                raise
            time.sleep(min(2**attempt, 30))
            continue
        size = partial.stat().st_size
        if expected_size is not None and size != expected_size:
            if size > expected_size:
                partial.unlink()
            if attempt == 6:
                raise OSError(f"Incomplete download: got {size} bytes, expected {expected_size}")
            time.sleep(min(2**attempt, 30))
            continue
        partial.replace(destination)
        return


def fetch_months(months: list[str], output_dir: Path) -> Path:
    if not months or any(not MONTH_PATTERN.fullmatch(month) for month in months):
        raise ValueError("Months must be supplied as YYYY-MM values")
    if len(months) != len(set(months)):
        raise ValueError("Months must be unique")
    catalogue = _catalogue_files()
    missing = sorted(set(months) - set(catalogue))
    if missing:
        raise ValueError(f"Months absent from the Infrabel catalogue: {', '.join(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    old_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = dict(old_manifest.get("files", {}))
    for month in months:
        url = catalogue[month]
        filename = f"Data_raw_punctuality_{month.replace('-', '')}.csv"
        destination = output_dir / filename
        expected_size, etag, modified = _head(url)
        if not destination.exists() or (
            expected_size is not None and destination.stat().st_size != expected_size
        ):
            print(f"Downloading Infrabel {month} ({expected_size or 'unknown'} bytes)", flush=True)
            _download(url, destination, expected_size)
        files[month] = {
            "file": filename,
            "url": url,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "etag": etag,
            "last_modified": modified,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
        }
    manifest = {
        "schema": "osrd-infrabel-punctuality-source/v1",
        "source": ATTRIBUTION,
        "catalogue_url": CATALOG_URL,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "timezone": "Europe/Brussels (source clock values remain local wall time)",
        "files": files,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest_path


def normalize_months(months: list[str], input_dir: Path, output_dir: Path) -> Path:
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Install the rail-data extra to normalize Infrabel CSV files") from error
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Fetch source files first; the checksum manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_files = manifest.get("files", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for month in months:
        if month not in source_files:
            raise ValueError(f"No checksum-pinned source file for {month}")
        source_info = source_files[month]
        source = input_dir / source_info["file"]
        if _sha256(source) != source_info["sha256"]:
            raise ValueError(f"Source checksum mismatch for {month}: {source}")
        destination = output_dir / f"infrabel-punctuality-{month}.parquet"
        writer: Any = None
        row_count = 0
        source_row = 0
        schema: Any = None
        quality: dict[str, dict[str, int]] = {
            suffix: {
                "provider_delay_count": 0,
                "planned_time_count": 0,
                "actual_time_count": 0,
                "clock_comparison_count": 0,
                "within_1_second_count": 0,
                "maximum_absolute_clock_difference_seconds": 0,
            }
            for suffix in ("arrival", "departure")
        }
        for frame in pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            chunksize=100_000,
            encoding="utf-8-sig",
        ):
            chunk_size = len(frame)
            source_row += chunk_size
            normalized: dict[str, Any] = {
                "source_month": [month] * chunk_size,
                "source_row_number": range(source_row - chunk_size + 1, source_row + 1),
                "source_event_key": [
                    f"infrabel:{month}:{number:07d}"
                    for number in range(source_row - chunk_size + 1, source_row + 1)
                ],
                "service_date": pd.to_datetime(frame["DATDEP"], format="%d%b%Y", errors="coerce"),
                "train_number": frame["TRAIN_NO"],
                "operator": frame["TRAIN_SERV"],
                "relation": frame["RELATION"],
                "relation_direction": frame["RELATION_DIRECTION"],
                "stop_id": frame["PTCAR_NO"],
                "stop_name": frame["PTCAR_LG_NM_NL"],
                "line_departure": frame["LINE_NO_DEP"],
                "line_arrival": frame["LINE_NO_ARR"],
                "event_type_code": frame["CIRC_TYP"],
                "stop_type_code": frame["THOP1_COD"],
            }
            for prefix in ("ARR", "DEP"):
                suffix = "arrival" if prefix == "ARR" else "departure"
                for raw_name, target_name in (
                    (f"PLANNED_DATE_{prefix}", f"planned_{suffix}_local"),
                    (f"REAL_DATE_{prefix}", f"actual_{suffix}_local"),
                ):
                    clock_prefix = "PLANNED" if raw_name.startswith("PLANNED") else "REAL"
                    time_column = f"{clock_prefix}_TIME_{prefix}"
                    stamp = frame[raw_name].str.cat(frame[time_column], sep=" ")
                    normalized[target_name] = pd.to_datetime(
                        stamp, format="%d%b%Y %H:%M:%S", errors="coerce"
                    )
                delay_name = f"DELAY_{prefix}"
                normalized[f"delay_{suffix}_seconds"] = pd.to_numeric(
                    frame[delay_name].replace("", pd.NA), errors="coerce"
                ).astype("Int32")
                actual = normalized[f"actual_{suffix}_local"]
                planned = normalized[f"planned_{suffix}_local"]
                normalized[f"clock_delta_{suffix}_seconds"] = (
                    (actual - planned).dt.total_seconds().round().astype("Int32")
                )
            out = pd.DataFrame(normalized)
            for suffix, stats in quality.items():
                provider = out[f"delay_{suffix}_seconds"]
                planned = out[f"planned_{suffix}_local"]
                actual = out[f"actual_{suffix}_local"]
                comparison = (provider - out[f"clock_delta_{suffix}_seconds"]).abs().dropna()
                stats["provider_delay_count"] += int(provider.notna().sum())
                stats["planned_time_count"] += int(planned.notna().sum())
                stats["actual_time_count"] += int(actual.notna().sum())
                stats["clock_comparison_count"] += len(comparison)
                stats["within_1_second_count"] += int((comparison <= 1).sum())
                if not comparison.empty:
                    stats["maximum_absolute_clock_difference_seconds"] = max(
                        stats["maximum_absolute_clock_difference_seconds"],
                        int(comparison.max()),
                    )
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(destination, schema, compression="zstd")
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            row_count += chunk_size
        if writer is None:
            raise ValueError(f"Source month {month} contained no rows")
        writer.close()
        outputs.append(
            {
                "month": month,
                "file": destination.name,
                "rows": row_count,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "source_sha256": source_info["sha256"],
                "source_file": source_info["file"],
                "quality": {
                    suffix: {
                        **stats,
                        "fraction_within_1_second": (
                            stats["within_1_second_count"] / stats["clock_comparison_count"]
                            if stats["clock_comparison_count"]
                            else None
                        ),
                        "difference_over_1_second_count": (
                            stats["clock_comparison_count"] - stats["within_1_second_count"]
                        ),
                    }
                    for suffix, stats in quality.items()
                },
            }
        )
        print(f"Normalized {month}: {row_count:,} event rows -> {destination}", flush=True)
    normalized_manifest = {
        "schema": "osrd-infrabel-punctuality-events/v1",
        "source_manifest_sha256": _sha256(manifest_path),
        "license": manifest["license"],
        "attribution": manifest["source"],
        "timezone": manifest["timezone"],
        "delay_label_semantics": (
            "Preserve provider DELAY_ARR/DELAY_DEP seconds. Local clock-derived deltas are "
            "included as quality diagnostics and may differ by rounding."
        ),
        "files": outputs,
    }
    destination_manifest = output_dir / "manifest.json"
    temporary = destination_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(normalized_manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination_manifest)
    return destination_manifest


def build_departure_delay_profile(months: list[str], input_dir: Path, output: Path) -> Path:
    """Create a train-origin delay sampler from observed Infrabel departure events."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Install the rail-data extra to build an Infrabel delay profile"
        ) from error
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Fetch source files first; the checksum manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    source_hashes: dict[str, str] = {}
    for month in months:
        source_info = manifest.get("files", {}).get(month)
        if not source_info:
            raise ValueError(f"No checksum-pinned source file for {month}")
        parquet_path = input_dir.parent / "events" / f"infrabel-punctuality-{month}.parquet"
        normalized_manifest_path = parquet_path.parent / "manifest.json"
        if not normalized_manifest_path.exists():
            raise ValueError(f"Normalize source month {month} before building its profile")
        normalized_manifest = json.loads(normalized_manifest_path.read_text(encoding="utf-8"))
        normalized_info = next(
            (item for item in normalized_manifest["files"] if item["month"] == month), None
        )
        if not normalized_info or _sha256(parquet_path) != normalized_info["sha256"]:
            raise ValueError(f"Normalized event checksum mismatch for {month}")
        source_hashes[month] = source_info["sha256"]
        frames.append(
            pd.read_parquet(
                parquet_path,
                columns=[
                    "service_date",
                    "operator",
                    "train_number",
                    "relation",
                    "relation_direction",
                    "planned_departure_local",
                    "actual_departure_local",
                    "planned_arrival_local",
                    "actual_arrival_local",
                    "delay_departure_seconds",
                ],
            )
        )
    events = pd.concat(frames, ignore_index=True)
    valid = events.dropna(
        subset=["planned_departure_local", "actual_departure_local", "delay_departure_seconds"]
    ).copy()
    service_key = [
        "service_date",
        "operator",
        "train_number",
        "relation",
        "relation_direction",
    ]
    origins = valid.sort_values("planned_departure_local").drop_duplicates(
        service_key, keep="first"
    )
    samples = origins["delay_departure_seconds"].astype(int).tolist()
    if len(samples) < 100:
        raise ValueError("Too few valid origin departures to build a stable delay profile")
    series = pd.Series(samples, dtype="int64")
    valid_dwells = events.dropna(subset=["planned_arrival_local", "planned_departure_local"])
    dwell_series = (
        (valid_dwells["planned_departure_local"] - valid_dwells["planned_arrival_local"])
        .dt.total_seconds()
        .round()
    )
    dwell_samples = dwell_series[(dwell_series >= 0) & (dwell_series <= 3600)].astype(int).tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "schema": "osrd-infrabel-origin-departure-delay-profile/v1",
        "source_manifest_sha256": _sha256(manifest_path),
        "source_month_sha256": source_hashes,
        "source_months": months,
        "license": manifest["license"],
        "attribution": manifest["source"],
        "sampling_unit": (
            "first observed planned departure per service-date/operator/train/relation/direction"
        ),
        "timezone": manifest["timezone"],
        "sample_count": len(samples),
        "delay_seconds": samples,
        "stop_dwell_seconds": dwell_samples,
        "statistics": {
            "minimum": int(series.min()),
            "maximum": int(series.max()),
            "mean": float(series.mean()),
            "positive_fraction": float((series > 0).mean()),
            "zero_fraction": float((series == 0).mean()),
            "quantiles_seconds": {
                str(q): float(series.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)
            },
            "fraction_within_abs_420_seconds": float((series.abs() <= 420).mean()),
            "dwell_sample_count": len(dwell_samples),
            "dwell_quantiles_seconds": {
                str(q): float(pd.Series(dwell_samples).quantile(q))
                for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, separators=(",", ":")) + "\n")
    temporary.replace(output)
    print(f"Built {len(samples):,} origin-delay samples -> {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Fetch official monthly raw files")
    fetch.add_argument("--months", nargs="+", required=True, metavar="YYYY-MM")
    fetch.add_argument(
        "--output-dir", type=Path, default=Path("data/rail/infrabel-punctuality/monthly")
    )
    normalize = subparsers.add_parser("normalize", help="Normalize raw files to Parquet")
    normalize.add_argument("--months", nargs="+", required=True, metavar="YYYY-MM")
    normalize.add_argument(
        "--input-dir", type=Path, default=Path("data/rail/infrabel-punctuality/monthly")
    )
    normalize.add_argument(
        "--output-dir", type=Path, default=Path("data/rail/infrabel-punctuality/events")
    )
    profile = subparsers.add_parser("profile", help="Build observed origin-delay sampler")
    profile.add_argument("--months", nargs="+", required=True, metavar="YYYY-MM")
    profile.add_argument(
        "--input-dir", type=Path, default=Path("data/rail/infrabel-punctuality/monthly")
    )
    profile.add_argument(
        "--output",
        type=Path,
        default=Path("data/rail/infrabel-punctuality/profiles/origin-delay-profile.json"),
    )
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            print(fetch_months(args.months, args.output_dir))
        elif args.command == "normalize":
            print(normalize_months(args.months, args.input_dir, args.output_dir))
        else:
            print(build_departure_delay_profile(args.months, args.input_dir, args.output))
    except (OSError, ValueError, urllib.error.URLError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
