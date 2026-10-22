"""Extend a replay GNN fit while preserving its frozen validation and test groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _source_directories(report_path: Path, excluded_roots: list[Path]) -> list[Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Training report contains no source manifests: {report_path}")
    result = [Path(source["path"]) for source in sources]
    if len(result) != len(set(result)):
        raise ValueError("Base report contains duplicate source directories")
    resolved_roots = [root.resolve() for root in excluded_roots]
    matched_roots: set[Path] = set()
    retained: list[Path] = []
    for path in result:
        resolved = path.resolve()
        excluding = next(
            (root for root in resolved_roots if resolved == root or root in resolved.parents), None
        )
        if excluding is None:
            retained.append(path)
        else:
            matched_roots.add(excluding)
    unmatched = sorted(map(str, set(resolved_roots) - matched_roots))
    if unmatched:
        raise ValueError("Exclude roots matched no base report sources: " + ", ".join(unmatched))
    result = retained
    if not result:
        raise ValueError("No base report sources remain after exclusion")
    missing = [str(path) for path in result if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError("Base report sources are missing: " + ", ".join(missing[:5]))
    return result


def _groups(report_path: Path, partition: str) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    groups = report.get("split", {}).get(f"{partition}_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Training report has no frozen {partition} groups: {report_path}")
    return [str(group) for group in groups]


def _additional_directories(roots: list[Path]) -> list[Path]:
    directories: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Additional cohort root does not exist: {root}")
        directories.extend(
            sorted(
                path
                for path in root.iterdir()
                if path.is_dir() and (path / "manifest.json").is_file()
            )
        )
    if not directories:
        raise ValueError("Additional roots contain no cohort manifest directories")
    if len(directories) != len(set(directories)):
        raise ValueError("Additional roots contain duplicate cohort directories")
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path, action="append", required=True)
    parser.add_argument("--exclude-source-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-group", action="append", default=[])
    parser.add_argument("--replace-validation-groups", action="store_true")
    parser.add_argument("--test-group", action="append", default=[])
    parser.add_argument("--replace-test-groups", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    from osrd_ops_intelligence.training.train_replay_gnn import main as train_main

    if args.replace_validation_groups and not args.validation_group:
        parser.error("--replace-validation-groups requires at least one --validation-group")
    if args.replace_test_groups and not args.test_group:
        parser.error("--replace-test-groups requires at least one --test-group")
    base_directories = _source_directories(args.base_report, args.exclude_source_root)
    additions = _additional_directories(args.additional_root)
    overlap = set(base_directories) & set(additions)
    if overlap:
        raise ValueError("Additional roots repeat base sources: " + ", ".join(map(str, overlap)))

    validation_groups = sorted(
        set(
            args.validation_group
            if args.replace_validation_groups
            else _groups(args.base_report, "validation") + args.validation_group
        )
    )
    test_groups = sorted(
        set(
            args.test_group
            if args.replace_test_groups
            else _groups(args.base_report, "test") + args.test_group
        )
    )
    sys.argv = [
        "osrd-replay-gnn-train",
        *(str(path) for path in [*base_directories, *additions]),
        "--output-dir",
        str(args.output_dir),
        "--validation-scenario-groups",
        *validation_groups,
        "--test-scenario-groups",
        *test_groups,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    train_main()


if __name__ == "__main__":
    main()
