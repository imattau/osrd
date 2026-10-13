"""Fetch the pinned, compact RIDE Gold Lite evaluation table.

The upstream dataset is licensed CC BY 4.0. Keep the data out of Git and retain
the generated provenance manifest alongside the downloaded file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

DATASET = "orailix/ride-gold-lite"
REVISION = "e5c1ed1126bca1784eb6ea93816dd6cc89afa544"
RELATIVE_PATH = "core/test_eval_table.parquet"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{RELATIVE_PATH}?download=true"
EXPECTED_BYTES = 74_008_518
EXPECTED_SHA256 = "6bd73df24dc459d1d8449995a120601aa9a640b71a8bf1dd5d537d20de9cfaf4"
ATTRIBUTION = "RIDE (OrAilix), based on data from Infrabel and Open-Meteo"


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _download(stream: BinaryIO, destination: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > EXPECTED_BYTES:
                raise ValueError("Download exceeds pinned expected size")
            digest.update(chunk)
            output.write(chunk)
    if size != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(f"Integrity check failed (bytes={size}, sha256={digest.hexdigest()})")


def fetch(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "test_eval_table.parquet"
    if data_path.exists():
        size, digest = _hash_file(data_path)
        if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
            raise ValueError(f"Existing file failed integrity check: {data_path}")
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "osrd-ops-intelligence/0.1"})
        fd, temporary_name = tempfile.mkstemp(prefix=".ride-", suffix=".part", dir=output_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                _download(response, temporary)
            temporary.replace(data_path)
        finally:
            temporary.unlink(missing_ok=True)

    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "path": RELATIVE_PATH,
        "license": "CC-BY-4.0",
        "attribution": ATTRIBUTION,
        "source_url": URL,
        "file": data_path.name,
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "usage": (
            "External observed-delay forecast evaluation only; not OSRD counterfactual labels."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rail/ride-gold-lite-core"),
        help="Ignored local directory for data and its provenance manifest",
    )
    args = parser.parse_args()
    path = fetch(args.output)
    print(f"Verified RIDE Gold Lite core table: {path}")
    print(f"SHA-256: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
