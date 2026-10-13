"""Score a last-observed-delay persistence reference on the RIDE test table."""

# PyArrow currently ships no type stubs; keep the strict project check useful
# for this module while allowing unknown types at the Arrow boundary.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.training.ride_core import (
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    REVISION,
)


def _verify_table(path: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    if size != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(f"RIDE evaluation table failed integrity check: {path}")


def evaluate(table_path: Path) -> dict[str, Any]:
    _verify_table(table_path)
    try:
        import numpy as np
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Install the optional Parquet dependency: uv sync --extra rail-data"
        ) from error

    horizon = 15
    columns = ["last_known_delay", *(f"future_delay_{step}" for step in range(1, horizon + 1))]
    table = pq.read_table(table_path, columns=columns)
    last_delay = pc.fill_null(table["last_known_delay"], math.nan).to_numpy(zero_copy_only=False)
    by_horizon: list[dict[str, float | int]] = []
    all_absolute_errors: list[Any] = []
    all_squared_errors: list[Any] = []
    all_errors: list[Any] = []

    for step in range(1, horizon + 1):
        future = pc.fill_null(table[f"future_delay_{step}"], math.nan).to_numpy(
            zero_copy_only=False
        )
        valid = np.isfinite(last_delay) & np.isfinite(future)
        errors = last_delay[valid] - future[valid]
        absolute_errors = np.abs(errors)
        if errors.size == 0:
            raise ValueError(f"No valid target rows for forecast horizon {step}")
        by_horizon.append(
            {
                "step": step,
                "n": int(errors.size),
                "mae_s": float(np.mean(absolute_errors)),
                "rmse_s": float(np.sqrt(np.mean(np.square(errors)))),
                "bias_s": float(np.mean(errors)),
            }
        )
        all_absolute_errors.append(absolute_errors)
        all_squared_errors.append(np.square(errors))
        all_errors.append(errors)

    absolute = np.concatenate(all_absolute_errors)
    squared = np.concatenate(all_squared_errors)
    errors = np.concatenate(all_errors)
    return {
        "dataset": "RIDE Gold Lite test evaluation table",
        "dataset_revision": REVISION,
        "source_sha256": EXPECTED_SHA256,
        "rows": table.num_rows,
        "target_unit": "seconds",
        "baseline": "last known delay held constant at every future event",
        "overall": {
            "n_targets": int(absolute.size),
            "mae_s": float(np.mean(absolute)),
            "rmse_s": float(np.sqrt(np.mean(squared))),
            "bias_s": float(np.mean(errors)),
        },
        "by_horizon": by_horizon,
        "interpretation": (
            "External observed-delay forecast reference only; not a timetable-change "
            "counterfactual or OSRD conflict label."
        ),
        "attribution": "RIDE (OrAilix), based on data from Infrabel and Open-Meteo; CC BY 4.0",
    }


def main() -> None:
    default_data = Path("data/rail/ride-gold-lite-core")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "table",
        type=Path,
        nargs="?",
        default=default_data / "test_eval_table.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data / "persistence-baseline.json",
    )
    args = parser.parse_args()
    result = evaluate(args.table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], sort_keys=True))
    print(f"Horizon metrics: {args.output}")


if __name__ == "__main__":
    main()
