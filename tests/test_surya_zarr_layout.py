from pathlib import Path

import pytest

from ordinal_cqr.datasets.surya_zarr import discover_surya_year_groups


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
