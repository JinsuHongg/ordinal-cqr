"""Create a binary solar-disk mask from a representative Surya Zarr frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def _channel_names(data: xr.DataArray) -> list[str]:
    if "channel" not in data.dims:
        return []
    if "channel_names" in data.attrs:
        return [str(name) for name in data.attrs["channel_names"]]
    if "channel" in data.coords:
        return [str(name) for name in data.coords["channel"].values]
    return [str(index) for index in range(data.sizes["channel"])]


def _select_frame(zarr_path: Path, year: str | None, channel: str | None) -> tuple[np.ndarray, str, str]:
    root = zarr.open(zarr_path, mode="r")
    years = sorted(root.group_keys())
    if not years:
        raise ValueError(f"Zarr store has no groups: {zarr_path}")
    selected_year = year or years[0]
    if selected_year not in years:
        raise ValueError(f"Year group {selected_year!r} is absent; available groups: {years}")

    dataset = xr.open_zarr(zarr_path, group=selected_year, chunks="auto")
    if "dataset" in dataset:
        data = dataset["dataset"]
        if "channel" in data.dims and "channel" not in data.coords and "channel_names" in data.attrs:
            data = data.assign_coords(channel=[str(name) for name in data.attrs["channel_names"]])
        names = _channel_names(data)
        if "channel" in data.dims:
            selected_channel = channel or next(
                (name for name in names if name.lower().startswith("hmi")), names[0]
            )
            if selected_channel not in names:
                raise ValueError(f"Channel {selected_channel!r} is absent; available channels: {names}")
            data = data.sel(channel=selected_channel)
        elif channel is not None:
            raise ValueError("--channel was provided but the stacked Zarr data has no channel dimension.")
        else:
            selected_channel = "dataset"
    else:
        names = list(dataset.data_vars)
        selected_channel = channel or next(
            (name for name in names if name.lower().startswith("hmi")), names[0]
        )
        if selected_channel not in dataset:
            raise ValueError(f"Channel {selected_channel!r} is absent; available channels: {names}")
        data = dataset[selected_channel]

    time_dim = next((dim for dim in ("timestep", "time") if dim in data.dims), None)
    if time_dim is None:
        raise ValueError(f"Selected channel has no timestep/time dimension: {data.dims}")
    frame = np.asarray(data.isel({time_dim: 0}).values, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2-D image after selecting time/channel, got shape {frame.shape}.")
    return frame, selected_year, selected_channel


def _disk_geometry(frame: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Estimate a centered disk from finite, non-background HMI pixels.

    Surya HMI frames use zero (or NaN) outside the solar disk. The extrema of
    non-background pixels provide the disk bounding box without adding OpenCV
    as a runtime dependency.
    """
    support = np.isfinite(frame) & (np.abs(frame) > threshold)
    rows, cols = np.nonzero(support)
    if not len(rows):
        raise ValueError(
            "No finite pixels exceeded --support-threshold. Choose an HMI channel "
            "or lower the threshold."
        )
    x_min, x_max = float(cols.min()), float(cols.max())
    y_min, y_max = float(rows.min()), float(rows.max())
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    radius = max(x_max - x_min, y_max - y_min) / 2.0 + 0.5
    if radius <= 1:
        raise ValueError("Estimated solar-disk radius is implausibly small.")
    return center_x, center_y, radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--year", help="Zarr year group; defaults to the first group.")
    parser.add_argument("--channel", help="HMI channel name; defaults to the first name beginning with 'hmi'.")
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument("--center-x", type=float)
    parser.add_argument("--center-y", type=float)
    parser.add_argument("--radius", type=float)
    args = parser.parse_args()

    if not args.zarr.exists():
        raise SystemExit(f"Zarr store does not exist: {args.zarr}")
    supplied_geometry = (args.center_x, args.center_y, args.radius)
    if any(value is not None for value in supplied_geometry) and any(
        value is None for value in supplied_geometry
    ):
        raise SystemExit("Provide --center-x, --center-y, and --radius together.")

    frame, year, channel = _select_frame(args.zarr, args.year, args.channel)
    if all(value is None for value in supplied_geometry):
        center_x, center_y, radius = _disk_geometry(frame, args.support_threshold)
    else:
        center_x, center_y, radius = supplied_geometry
        assert center_x is not None and center_y is not None and radius is not None

    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    mask = (((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask)
    print(
        f"Wrote {mask.shape} limb mask to {args.output}; year={year}, "
        f"channel={channel}, center=({center_x:.2f}, {center_y:.2f}), "
        f"radius={radius:.2f}, selected_pixels={int(mask.sum())}."
    )


if __name__ == "__main__":
    main()
