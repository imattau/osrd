"""Normalize exact duplicate OSM operational-point records in converted RailJSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _part_signature(operational_point: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    raw_parts = operational_point.get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError("Operational point has an invalid parts list")
    parts = cast(list[Any], raw_parts)
    signature: list[tuple[str, float]] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Operational point has an invalid part")
        part_dict = cast(dict[str, Any], part)
        track = part_dict.get("track")
        position = part_dict.get("position")
        if (
            not isinstance(track, str)
            or not track
            or not isinstance(position, int | float)
            or isinstance(position, bool)
            or not math.isfinite(position)
            or position < 0
        ):
            raise ValueError("Operational-point part needs a track and numeric position")
        signature.append((track, float(position)))
    return tuple(sorted(signature))


def normalize(
    input_path: Path,
    output_path: Path,
    *,
    disambiguate_conflicting_osrd_identities: bool = False,
) -> dict[str, Any]:
    """Normalize exact duplicates and optionally disambiguate OSM identity collisions."""
    raw_document: Any = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_document, dict):
        raise ValueError("RailJSON document has no operational_points list")
    document = cast(dict[str, Any], raw_document)
    raw_points = document.get("operational_points")
    if not isinstance(raw_points, list):
        raise ValueError("RailJSON document has no operational_points list")
    points: list[dict[str, Any]] = []
    for raw_point in cast(list[Any], raw_points):
        if not isinstance(raw_point, dict):
            raise ValueError("RailJSON operational_points contains a non-object entry")
        points.append(cast(dict[str, Any], raw_point))
    point_ids = [point.get("id") for point in points]
    if any(not isinstance(point_id, str) or not point_id for point_id in point_ids):
        raise ValueError("RailJSON operational points need non-empty string IDs")
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("RailJSON operational point IDs must be unique")
    remove_ids: set[str] = set()
    duplicate_groups: list[dict[str, Any]] = []
    disambiguated_groups: list[dict[str, Any]] = []
    def groups_for(key_fn: Any) -> dict[tuple[object, ...], list[dict[str, Any]]]:
        result: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            key = key_fn(point)
            if key is not None:
                result[key].append(point)
        return result

    def remove_exact_duplicates(
        identity_type: str,
        identity: tuple[object, ...],
        group: list[dict[str, Any]],
    ) -> bool:
        signatures = [_part_signature(value) for value in group]
        names = {value.get("name") for value in group}
        if len(names) != 1 or any(signature != signatures[0] for signature in signatures[1:]):
            return False
        keeper = min(group, key=lambda value: str(value.get("id", "")))
        removed = [value for value in group if value is not keeper]
        remove_ids.update(str(value.get("id", "")) for value in removed)
        duplicate_groups.append(
            {
                "identity_type": identity_type,
                "identity": list(identity),
                "name": keeper.get("name"),
                "kept_id": keeper.get("id"),
                "removed_ids": sorted(str(value.get("id", "")) for value in removed),
                "part_locations": [list(part) for part in signatures[0]],
            }
        )
        return True

    def disambiguate_group(
        identity_type: str,
        identity: tuple[object, ...],
        group: list[dict[str, Any]],
        field: str,
        suffix_kind: str,
    ) -> None:
        entries = []
        for point in group:
            original = str(point[field])
            suffix_value = point["uic"] if suffix_kind == "UIC" else point["id"]
            normalized = f"{original}-{suffix_kind}{suffix_value}"
            point[field] = normalized
            entries.append(
                {
                    "id": point["id"],
                    "uic": point.get("uic"),
                    "field": field,
                    "original_code": original,
                    "normalized_code": normalized,
                }
            )
        disambiguated_groups.append(
            {"identity_type": identity_type, "identity": list(identity), "entries": entries}
        )

    # OSRD has unique indexes on domestic identity and on UIC+secondary code.
    # Handle exact duplicates first; preserve genuinely distinct points by
    # changing only the conflicting local code when the caller opts in.
    domestic_key = lambda point: (
        point.get("country_code"),
        point.get("main_code"),
        point.get("secondary_code"),
    )
    for identity, group in groups_for(domestic_key).items():
        if len(group) < 2 or remove_exact_duplicates("domestic", identity, group):
            continue
        uics = [point.get("uic") for point in group]
        if (
            not disambiguate_conflicting_osrd_identities
            or not all(isinstance(uic, int) and not isinstance(uic, bool) for uic in uics)
            or len(uics) != len(set(uics))
        ):
            raise ValueError(
                "Conflicting operational points share domestic identity "
                f"{identity!r}; refusing to alter distinct locations"
            )
        disambiguate_group("domestic", identity, group, "main_code", "UIC")

    uic_key = lambda point: (
        (point.get("uic"), point.get("secondary_code"))
        if point.get("uic") is not None
        else None
    )
    for identity, group in groups_for(uic_key).items():
        remaining = [point for point in group if str(point.get("id", "")) not in remove_ids]
        if len(remaining) < 2 or remove_exact_duplicates("uic_secondary", identity, remaining):
            continue
        if not disambiguate_conflicting_osrd_identities:
            raise ValueError(
                "Conflicting operational points share UIC/secondary identity "
                f"{identity!r}; refusing to alter distinct locations"
            )
        disambiguate_group("uic_secondary", identity, remaining, "secondary_code", "OSM")

    plc_key = lambda point: (point.get("plc"),) if point.get("plc") is not None else None
    for identity, group in groups_for(plc_key).items():
        remaining = [point for point in group if str(point.get("id", "")) not in remove_ids]
        if len(remaining) > 1 and not remove_exact_duplicates("plc", identity, remaining):
            raise ValueError(
                f"Conflicting operational points share PLC identity {identity!r}; refusing to alter it"
            )

    document["operational_points"] = [
        value for value in points if str(value.get("id", "")) not in remove_ids
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    report = {
        "schema": "osrd-ops-osm-railjson-normalization/v1",
        "source_path": str(input_path),
        "source_sha256": _sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "source_operational_point_count": len(points),
        "output_operational_point_count": len(document["operational_points"]),
        "removed_duplicate_count": len(remove_ids),
        "duplicate_groups": duplicate_groups,
        "disambiguated_groups": disambiguated_groups,
        "method": (
            "OSRD domestic, UIC+secondary, and PLC unique identities are checked. Exact name and "
            "track-position duplicates are removed. With explicit opt-in, remaining domestic "
            "conflicts use UIC-suffixed main codes and UIC+secondary conflicts use OSM-ID-suffixed "
            "secondary codes; PLC conflicts and unresolvable domestic conflicts fail closed."
        ),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--disambiguate-conflicting-osrd-identities",
        action="store_true",
        help=(
            "Preserve non-exact RailJSON operational points that collide on OSRD identity indexes "
            "by adding stable local UIC or OSM-ID suffixes to their conflicting codes."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            normalize(
                args.input,
                args.output,
                disambiguate_conflicting_osrd_identities=args.disambiguate_conflicting_osrd_identities,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
