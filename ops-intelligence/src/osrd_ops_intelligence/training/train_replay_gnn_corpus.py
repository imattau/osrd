"""Train a replay GNN from explicit train, validation, and test cohort roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _cohorts(roots: list[Path], partition: str) -> tuple[list[Path], list[str]]:
    directories: list[Path] = []
    groups: set[str] = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{partition} cohort root does not exist: {root}")
        candidates = sorted(
            path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()
        )
        if not candidates:
            raise ValueError(f"{partition} cohort root contains no manifest directories: {root}")
        for cohort in candidates:
            if cohort in directories:
                raise ValueError(f"Cohort directory is included more than once: {cohort}")
            manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
            episode_path = cohort / "episodes.jsonl"
            raw = episode_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest.get("episodes_sha256"):
                raise ValueError(f"Episode checksum does not match manifest: {episode_path}")
            episode_groups = {
                str(json.loads(line)["scenario_group"]) for line in raw.splitlines() if line
            }
            if len(episode_groups) != 1:
                raise ValueError(
                    f"Every cohort must contain exactly one scenario group: {episode_path}"
                )
            groups.update(episode_groups)
            directories.append(cohort)
    if not directories or not groups:
        raise ValueError(f"At least one {partition} cohort is required")
    return directories, sorted(groups)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, action="append", required=True)
    parser.add_argument("--validation-root", type=Path, action="append", required=True)
    parser.add_argument("--test-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_dirs, train_groups = _cohorts(args.train_root, "train")
    validation_dirs, validation_groups = _cohorts(args.validation_root, "validation")
    test_dirs, test_groups = _cohorts(args.test_root, "test")
    overlap = (
        (set(train_groups) & set(validation_groups))
        | (set(train_groups) & set(test_groups))
        | (set(validation_groups) & set(test_groups))
    )
    if overlap:
        raise ValueError("Scenario groups cross data partitions: " + ", ".join(sorted(overlap)))
    # Keep the final source order independent of CLI root order and stable for
    # deterministic dataloader shuffling and checkpoint signatures.
    cohort_dirs = sorted(train_dirs + validation_dirs + test_dirs)
    if len(cohort_dirs) != len(set(cohort_dirs)):
        raise ValueError("Cohort directories must not be shared between partitions")

    from osrd_ops_intelligence.training.train_replay_gnn import main as train_main

    sys.argv = [
        "osrd-replay-gnn-train",
        *(str(path) for path in cohort_dirs),
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
        "--hidden-channels",
        str(args.hidden_channels),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--log-every",
        str(args.log_every),
        *(["--resume"] if args.resume else []),
        *(["--overwrite"] if args.overwrite else []),
    ]
    print(
        json.dumps(
            {
                "train_cohorts": len(train_dirs),
                "train_groups": len(train_groups),
                "validation_cohorts": len(validation_dirs),
                "validation_groups": len(validation_groups),
                "test_cohorts": len(test_dirs),
                "test_groups": len(test_groups),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_main()


if __name__ == "__main__":
    main()
