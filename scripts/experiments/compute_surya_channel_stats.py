"""Compute train-only, per-channel signed-log statistics for SuryaBench."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import yaml

from ordinal_cqr.datasets.surya_zarr import (
    consolidate_surya_year_metadata,
    discover_surya_year_groups,
    open_surya_year_dataset,
    unambiguous_surya_timestamps,
)


def _retained_training_rows(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {args.timestamp_column, args.label_column, args.numeric_target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training index is missing columns: {sorted(missing)}")
    labels = frame[args.label_column].astype("string").str.strip().str.upper()
    values = pd.to_numeric(frame[args.numeric_target_column], errors="coerce")
    keep = ~((labels == "FQ") & (values >= args.fq_max_intensity))
    if args.exclude_goes_class:
        keep &= ~labels.isin({label.upper() for label in args.exclude_goes_class})
    return frame.loc[keep].copy()


def _group_timestamps_by_year(values: pd.Series) -> dict[str, pd.Series]:
    """Parse manifest timestamps and group them by calendar year."""
    timestamps = pd.Series(pd.to_datetime(values, errors="raise"), copy=False)
    return {
        str(year): group
        for year, group in timestamps.groupby(timestamps.dt.year)
    }


def _unambiguous_requested_positions(
    available: pd.DatetimeIndex, requested: pd.DatetimeIndex
) -> np.ndarray:
    """Return integer positions for requested timestamps with one Zarr frame."""
    available = pd.DatetimeIndex(available)
    requested = pd.DatetimeIndex(requested)
    safe = unambiguous_surya_timestamps(available).intersection(requested)
    return np.flatnonzero(available.isin(safe))


def _load_year(
    zarr_path: Path, year: str, channels: list[str] | None
) -> tuple[object, list[str]]:
    dataset = open_surya_year_dataset(zarr_path, year, chunks="auto")
    stacked_name = next(
        (name for name, value in dataset.data_vars.items() if "channel" in value.dims),
        None,
    )
    if stacked_name is not None:
        image_data = dataset[stacked_name]
        if "channel_names" in image_data.attrs:
            image_data = image_data.assign_coords(
                channel=[str(value) for value in image_data.attrs["channel_names"]]
            )
        if "channel" not in image_data.dims:
            raise ValueError(f"Stacked Zarr group {year} has no channel dimension.")
        available = [str(value) for value in image_data.coords["channel"].values]
        selected = channels or available
        missing = set(selected).difference(available)
        if missing:
            raise ValueError(f"Group {year} lacks channels: {sorted(missing)}")
        return image_data.sel(channel=selected), selected

    selected = channels or list(dataset.data_vars)
    missing = set(selected).difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Group {year} lacks channels: {sorted(missing)}")
    return xr.concat(
        [dataset[channel] for channel in selected],
        dim=pd.Index(selected, name="channel"),
    ), selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--limb-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channels", nargs="+", help="Ordered channel names; defaults to Zarr order.")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--label-column", default="max_goes_class")
    parser.add_argument("--numeric-target-column", default="max_intensity")
    parser.add_argument("--fq-max-intensity", type=float, default=1.0e-7)
    parser.add_argument("--exclude-goes-class", action="append", default=["M0.9"])
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help=(
            "Write .zmetadata for each completed yearly group before reading it. "
            "This changes metadata only, not image chunks."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=("all", "matching", "none"),
        default="all",
        help="Apply the limb mask to all channels (default), matching channels, or none.",
    )
    parser.add_argument(
        "--mask-channel",
        action="append",
        default=[],
        help="Exact channel name to disk-mask when --mask-mode=matching.",
    )
    parser.add_argument(
        "--mask-channel-prefix",
        action="append",
        default=[],
        help="Channel-name prefix to disk-mask when --mask-mode=matching (e.g. hmi).",
    )
    args = parser.parse_args()

    for label, path in (("Zarr store", args.zarr), ("training index", args.train_index), ("limb mask", args.limb_mask)):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    train = _retained_training_rows(pd.read_csv(args.train_index), args)
    if args.consolidate_metadata:
        stores = consolidate_surya_year_metadata(args.zarr)
        print(f"Consolidated metadata for {len(stores)} yearly Zarr groups.")

    by_year = _group_timestamps_by_year(train[args.timestamp_column])
    mask = np.load(args.limb_mask).astype(bool)
    if not mask.any():
        raise SystemExit("Limb mask contains no selected pixels.")

    counts: list[da.Array] = []
    sums: list[da.Array] = []
    sum_squares: list[da.Array] = []
    selected_channels = args.channels
    retained_timestamps = 0
    excluded_ambiguous_zarr_frames = 0

    for year in discover_surya_year_groups(args.zarr):
        requested = by_year.get(str(year))
        if requested is None:
            continue
        image_data, resolved_channels = _load_year(args.zarr, year, selected_channels)
        if selected_channels is None:
            selected_channels = resolved_channels
        if args.mask_mode == "all":
            mask_channels = np.ones(len(selected_channels), dtype=bool)
        elif args.mask_mode == "none":
            mask_channels = np.zeros(len(selected_channels), dtype=bool)
        else:
            exact = {name.lower() for name in args.mask_channel}
            prefixes = tuple(prefix.lower() for prefix in args.mask_channel_prefix)
            mask_channels = np.asarray(
                [
                    channel.lower() in exact
                    or channel.lower().startswith(prefixes)
                    for channel in selected_channels
                ],
                dtype=bool,
            )
            if not mask_channels.any():
                raise ValueError(
                    "The selective statistics mask did not match any channels: "
                    f"{selected_channels}."
                )
        time_dim = "timestep" if "timestep" in image_data.dims else "time"
        if time_dim not in image_data.dims:
            raise ValueError(f"Group {year} has no timestep/time dimension.")
        all_available = pd.DatetimeIndex(image_data[time_dim].values)
        available = unambiguous_surya_timestamps(all_available)
        excluded_ambiguous_zarr_frames += len(all_available) - len(available)
        positions = _unambiguous_requested_positions(all_available, requested)
        if not len(positions):
            continue
        spatial_dims = [dim for dim in image_data.dims if dim not in ("channel", time_dim)]
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected two spatial dimensions, found {spatial_dims}.")
        image_data = image_data.isel({time_dim: positions}).transpose(
            "channel", time_dim, *spatial_dims
        )
        array = image_data.data.astype(np.float64)
        if tuple(array.shape[-2:]) != mask.shape:
            raise ValueError(
                f"Limb mask shape {mask.shape} does not match Zarr image shape "
                f"{tuple(array.shape[-2:])}."
            )
        transformed = da.sign(array) * da.log1p(da.fabs(array))
        channel_validity = np.where(
            mask_channels[:, np.newaxis, np.newaxis],
            mask[np.newaxis, :, :],
            True,
        )
        valid = da.isfinite(transformed) & channel_validity[:, np.newaxis, :, :]
        values = da.where(valid, transformed, 0.0)
        counts.append(valid.sum(axis=(1, 2, 3)))
        sums.append(values.sum(axis=(1, 2, 3)))
        sum_squares.append((values * values).sum(axis=(1, 2, 3)))
        retained_timestamps += len(positions)

    if not counts or selected_channels is None or retained_timestamps == 0:
        raise SystemExit("No training timestamps were found in the Zarr store.")
    count, total, squared_total = dask.compute(sum(counts), sum(sums), sum(sum_squares))
    if np.any(count == 0):
        raise SystemExit("At least one channel has no finite training pixels inside the limb mask.")
    mean = total / count
    variance = np.maximum(squared_total / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise SystemExit("At least one channel has a non-positive or non-finite standard deviation.")

    payload = {
        "schema_version": "surya-channel-stats-v2",
        "transform": "sign(x) * log1p(abs(x))",
        "split": "train",
        "channels": selected_channels,
        "xarray_mask_and_scale": False,
        "limb_mask_policy": {
            "mode": args.mask_mode,
            "masked_channels": [
                channel
                for channel, use_mask in zip(selected_channels, mask_channels)
                if use_mask
            ],
        },
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "pixel_count_per_channel": [int(value) for value in count],
        "retained_training_timestamps": retained_timestamps,
        "excluded_ambiguous_zarr_frames": excluded_ambiguous_zarr_frames,
        "source_train_index_sha256": hashlib.sha256(args.train_index.read_bytes()).hexdigest(),
        "zarr_path": str(args.zarr),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Channels (stored order): {selected_channels}")
    print(f"Disk-masked channels: {payload['limb_mask_policy']['masked_channels']}")
    print(f"Wrote per-channel statistics for {len(selected_channels)} channels to {args.output}")


if __name__ == "__main__":
    main()
