"""Fail fast when a SuryaBench run is not portable to the current machine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ordinal_cqr.datasets.surya_zarr import (
    discover_surya_year_groups,
    open_surya_year_dataset,
)


REQUIRED_COLUMNS = {"timestamp", "max_intensity", "max_goes_class"}


def _existing_file(label: str, value: str, errors: list[str]) -> None:
    if not Path(value).is_file():
        errors.append(f"{label} is not a readable file: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Surya YAML configuration file")
    args = parser.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Configuration file not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    try:
        OmegaConf.resolve(cfg)
    except Exception as exc:
        raise SystemExit(f"Could not resolve configuration environment variables: {exc}") from exc

    errors: list[str] = []
    data = cfg.data
    index_dir = Path(str(data.flare_index.path))
    if not index_dir.is_dir():
        errors.append(f"SURYA_INDEX_DIR is not a directory: {index_dir}")
    else:
        for split in ("train", "val", "test"):
            filename = data.flare_index.get(split)
            if filename is None:
                errors.append(f"data.flare_index.{split} is missing")
                continue
            csv_path = index_dir / str(filename)
            _existing_file(f"{split} index", str(csv_path), errors)
            if csv_path.is_file():
                columns = set(pd.read_csv(csv_path, nrows=1).columns)
                missing = REQUIRED_COLUMNS - columns
                if missing:
                    errors.append(f"{csv_path} lacks required columns: {sorted(missing)}")
        cal = data.flare_index.get("cal")
        if cal is not None:
            _existing_file("calibration index", str(index_dir / str(cal)), errors)

    zarr_path = Path(str(data.input_zarr_path))
    if not zarr_path.exists():
        errors.append(f"SURYA_ZARR_PATH does not exist: {zarr_path}")
    _existing_file("statistics file", str(data.input_stat_path), errors)
    _existing_file("limb mask", str(data.limb_mask_path), errors)

    expected_size = data.get("expected_image_size")
    if Path(str(data.limb_mask_path)).is_file() and expected_size is not None:
        mask_shape = tuple(np.load(data.limb_mask_path).shape)
        if mask_shape != tuple(expected_size):
            errors.append(
                f"Limb mask size is {mask_shape}; config requires "
                f"{tuple(expected_size)}."
            )

    expected_channels = data.get("expected_channels")
    if Path(str(data.input_stat_path)).is_file() and expected_channels is not None:
        stats = OmegaConf.load(data.input_stat_path)
        for name in ("mean", "std"):
            values = stats.get(name)
            if not OmegaConf.is_list(values) or len(values) != expected_channels:
                errors.append(
                    f"{data.input_stat_path} must provide {expected_channels} "
                    f"per-channel {name} values for this configuration."
                )

    if zarr_path.exists() and expected_channels is not None:
        try:
            years = discover_surya_year_groups(zarr_path)
            if not years:
                errors.append(f"Zarr store has no year groups: {zarr_path}")
            else:
                sample = open_surya_year_dataset(zarr_path, years[0])
                stacked_name = next(
                    (name for name, value in sample.data_vars.items() if "channel" in value.dims),
                    None,
                )
                if stacked_name is not None:
                    image_data = sample[stacked_name]
                    available_channels = image_data.attrs.get("channel_names")
                    channel_count = (
                        len(available_channels)
                        if available_channels is not None
                        else int(image_data.sizes.get("channel", 1))
                    )
                    spatial_shape = tuple(image_data.shape[-2:])
                else:
                    channel_count = len(sample.data_vars)
                    spatial_shape = tuple(next(iter(sample.data_vars.values())).shape[-2:])
                if channel_count != expected_channels:
                    errors.append(
                        f"Zarr has {channel_count} channels; config requires "
                        f"{expected_channels}."
                    )
                if expected_size is not None and spatial_shape != tuple(expected_size):
                    errors.append(
                        f"Zarr image size is {spatial_shape}; config requires "
                        f"{tuple(expected_size)}."
                    )
        except Exception as exc:
            errors.append(f"Could not inspect Surya Zarr layout: {exc}")

    if errors:
        raise SystemExit("Surya preflight failed:\n- " + "\n- ".join(errors))

    print("Surya preflight passed")
    print(f"  config: {args.config}")
    print(f"  indices: {index_dir}")
    print(f"  zarr: {zarr_path}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}")


if __name__ == "__main__":
    main()
