from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from ordinal_cqr.datasets.surya_zarr import (
    consolidate_surya_year_metadata,
    discover_surya_year_groups,
    open_surya_year_dataset,
    timestamps_in_surya_year_partition,
    unambiguous_surya_timestamps,
)


def test_discovers_directory_partitioned_year_groups(tmp_path: Path) -> None:
    for year in ("2024", "2010", "2019"):
        (tmp_path / year / "dataset").mkdir(parents=True)
        (tmp_path / year / "dataset" / ".zgroup").write_text("{}")
    (tmp_path / "notes").mkdir()

    assert discover_surya_year_groups(tmp_path) == ["2010", "2019", "2024"]


def test_rejects_store_without_valid_year_groups(tmp_path: Path) -> None:
    (tmp_path / "2010").mkdir()

    with pytest.raises(ValueError, match="Could not find numeric Surya year groups"):
        discover_surya_year_groups(tmp_path)


def test_filters_all_frames_at_ambiguous_timestamps() -> None:
    timestamps = pd.to_datetime(
        ["2010-01-01", "2010-01-01", "2010-02-01", "2010-03-01", "2010-03-01"]
    )

    result = unambiguous_surya_timestamps(timestamps)

    assert result.tolist() == [pd.Timestamp("2010-02-01")]


def test_filters_timestamps_stored_under_the_wrong_year_partition() -> None:
    timestamps = pd.to_datetime(["2010-12-31 23:00", "2011-01-01 00:00"])

    result = timestamps_in_surya_year_partition(timestamps, "2010")

    assert result.tolist() == [pd.Timestamp("2010-12-31 23:00")]


def _write_zero_filled_year_store(root: Path, year: str = "2024") -> Path:
    store = root / year / "dataset"
    group = zarr.open_group(str(store), mode="w")
    images = group.create_dataset(
        "images",
        shape=(1, 1, 2, 2),
        chunks=(1, 1, 2, 2),
        dtype="float32",
        fill_value=0.0,
    )
    images.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel", "y", "x"]
    images[:] = np.asarray([[[[0.0, 1.0], [-1.0, 0.0]]]], dtype=np.float32)
    time = group.create_dataset(
        "time", shape=(1,), chunks=(1,), dtype="int64"
    )
    time.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    time[:] = np.asarray([0], dtype=np.int64)
    return store


def test_open_preserves_zeros_marked_as_zarr_fill_value(tmp_path: Path) -> None:
    _write_zero_filled_year_store(tmp_path)

    dataset = open_surya_year_dataset(tmp_path, "2024")

    values = dataset["images"].values
    assert np.isfinite(values).all()
    assert values.tolist() == [[[[0.0, 1.0], [-1.0, 0.0]]]]


def test_consolidates_each_year_and_remains_readable(tmp_path: Path) -> None:
    stores = [
        _write_zero_filled_year_store(tmp_path, year)
        for year in ("2023", "2024")
    ]

    result = consolidate_surya_year_metadata(tmp_path)

    assert result == stores
    assert all((store / ".zmetadata").is_file() for store in stores)
    values = open_surya_year_dataset(tmp_path, "2024")["images"].values
    assert np.isfinite(values).all()
