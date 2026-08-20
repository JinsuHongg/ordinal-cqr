#!/usr/bin/env python3
"""Report post-image-availability class distributions for SuryaBench splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ordinal_cqr.datasets.flare_cls_datasets import (
    _map_goes_class,
    filter_ocqr_flare_rows,
)
from ordinal_cqr.datasets.surya_zarr import (
    discover_surya_year_groups,
    open_surya_year_dataset,
    timestamps_in_surya_year_partition,
    unambiguous_surya_timestamps,
)

SPLITS = {
    "train": "train.csv",
    "validation": "validation.csv",
    "calibration": "leaky_validation.csv",
    "test": "test.csv",
}
CLASS_NAMES = ("FQ/A", "B", "C", "M", "X")


def image_timestamps(zarr_path: str) -> pd.DatetimeIndex:
    """Return unambiguous, calendar-partition-valid timestamps in the Zarr store."""
    partitions: list[pd.DatetimeIndex] = []
    for year in discover_surya_year_groups(zarr_path):
        dataset = open_surya_year_dataset(zarr_path, year)
        array = next(
            value for value in dataset.data_vars.values() if "channel" in value.dims
        )
        coordinate = "timestep" if "timestep" in array.coords else "time"
        timestamps = pd.DatetimeIndex(array.coords[coordinate].values)
        partitions.append(timestamps_in_surya_year_partition(timestamps, year))
    return unambiguous_surya_timestamps(
        pd.DatetimeIndex(np.concatenate(partitions))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    available_timestamps = image_timestamps(args.zarr_path)
    report: dict[str, object] = {
        "schema_version": "surya-image-available-distribution-v1",
        "index_dir": args.index_dir,
        "zarr_path": args.zarr_path,
        "filter_policy": {
            "fq_max_intensity_exclusion_threshold": 1e-7,
            "excluded_goes_classes": ["M0.9"],
        },
        "splits": {},
    }
    for split, filename in SPLITS.items():
        source = pd.read_csv(Path(args.index_dir) / filename)
        retained = filter_ocqr_flare_rows(
            source,
            fq_max_intensity=1e-7,
            excluded_goes_classes=("M0.9",),
        )
        timestamps = pd.to_datetime(retained["timestamp"])
        image_available = retained.loc[timestamps.isin(available_timestamps)].copy()
        labels = image_available["max_goes_class"].map(_map_goes_class)
        counts = [int((labels == class_id).sum()) for class_id in range(5)]
        total = len(image_available)
        report["splits"][split] = {
            "retained_manifest_row_count": int(len(retained)),
            "image_available_row_count": int(total),
            "class_counts": {
                str(class_id): counts[class_id] for class_id in range(5)
            },
            "class_names": {
                str(class_id): CLASS_NAMES[class_id] for class_id in range(5)
            },
            "class_percentages": {
                str(class_id): round(100 * counts[class_id] / total, 4)
                for class_id in range(5)
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["splits"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
