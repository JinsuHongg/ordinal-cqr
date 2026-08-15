"""Layout helpers for the SuryaBench year-partitioned Zarr store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _has_zarr_metadata(path: Path) -> bool:
    """Return whether ``path`` is a Zarr v2 or v3 group/array directory."""
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def discover_surya_year_groups(zarr_path: str | Path) -> list[str]:
    """Discover sorted year groups in either supported Surya Zarr layout.

    The current exported Surya store has numeric year directories containing a
    ``dataset`` Zarr group but no Zarr metadata at the store root. Older stores
    may instead expose those years through a conventional root Zarr group.
    """
    root = Path(zarr_path)
    if not root.is_dir():
        raise ValueError(f"Surya Zarr store is not a directory: {root}")

    years = [
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and entry.name.isdigit()
        and (_has_zarr_metadata(entry) or _has_zarr_metadata(entry / "dataset"))
    ]
    if years:
        return sorted(years, key=int)

    # Preserve compatibility with conventional root-group stores.
    try:
        import zarr

        return sorted(zarr.open_group(str(root), mode="r").group_keys(), key=int)
    except Exception as exc:
        raise ValueError(
            f"Could not find numeric Surya year groups under {root}. "
            "Expected <year>/dataset/.zgroup (or zarr.json)."
        ) from exc


def surya_year_store_path(zarr_path: str | Path, year: str) -> Path:
    """Return the Zarr store path for one Surya year partition."""
    root = Path(zarr_path)
    direct = root / year / "dataset"
    if _has_zarr_metadata(direct):
        return direct
    legacy = root / year
    if _has_zarr_metadata(legacy):
        return legacy
    raise ValueError(f"Surya year {year!r} has no readable Zarr group under {root}")


def open_surya_year_dataset(
    zarr_path: str | Path, year: str, *, chunks: Any = None
) -> Any:
    """Open one year partition without assuming root-level Zarr metadata."""
    import xarray as xr

    return xr.open_zarr(surya_year_store_path(zarr_path, year), chunks=chunks, consolidated=False)


def unambiguous_surya_timestamps(timestamps: Any) -> pd.DatetimeIndex:
    """Return only timestamps represented by exactly one frame in a Zarr partition.

    A label in the flare manifest identifies one observation. Multiple Zarr
    frames at that label are ambiguous, so they must be excluded rather than
    selected arbitrarily or counted multiple times in normalization statistics.
    """
    index = pd.DatetimeIndex(timestamps)
    return index[~index.duplicated(keep=False)]
