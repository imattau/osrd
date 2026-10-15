"""Fetch the pinned RIDE Gold Lite heterogeneous graph training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osrd_ops_intelligence.training.ride_training_data import (
    ATTRIBUTION,
    DATASET,
    LICENSE,
    REVISION,
    _download_file,
    _hash_file,
)

FILES = {
    "gnn/feature_spec.yaml": "28f636af3bf5ebac9df7190648b57d42bea795558a4842985e5f95fda9f8f9e3",
    "gnn/normalization.yaml": "8ebc3a30585ab2f4dd0f83eb0505375cf8d8aaf7d52694bd2f7bbaecc756fcfb",
    "gnn/train/graphs_part_00000.pt": "df9fef3fc0c72a47e6fea534afd51e2c959864a53de737f43f1d08e25a81a0ef",
    "gnn/train/graphs_part_00001.pt": "930cb01e2c0bc9b6da568b9f75b72791487992520976413aa4392459dbd9f1bc",
}


def fetch(output_dir: Path) -> dict[str, object]:
    """Download and verify only the pinned training graph files and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for relative_path, expected_sha256 in FILES.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            actual = _hash_file(destination)
            if actual != expected_sha256:
                raise ValueError(f"Existing file failed integrity check: {destination}")
            size = destination.stat().st_size
        else:
            size = _download_file(output_dir, relative_path, expected_sha256)
        files[relative_path] = {"sha256": expected_sha256, "bytes": size}

    manifest: dict[str, object] = {
        "schema": "osrd-ride-gnn-training-inputs/v1",
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "representation": "PyTorch Geometric heterogeneous graphs",
        "split": "provider-defined train only; no test graph fetched",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rail/ride-gold-lite-gnn-training"),
        help="Ignored local directory for pinned GNN training shards and provenance manifest",
    )
    args = parser.parse_args()
    try:
        manifest = fetch(args.output_dir)
    except KeyboardInterrupt:
        parser.exit(130, "Download cancelled; verified files and resumable cache were kept.\n")
    except Exception as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
