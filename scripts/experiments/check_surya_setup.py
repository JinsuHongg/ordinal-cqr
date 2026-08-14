"""Fail fast when a SuryaBench run is not portable to the current machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ordinal_cqr.datasets.surya_zarr import (
    discover_surya_year_groups,
    open_surya_year_dataset,
)
from ordinal_cqr.datasets.flare_cls_datasets import (
    _map_goes_class,
    build_ocqr_flare_manifest_audit,
    filter_ocqr_flare_rows,
)


SPLIT_KEYS = ("train", "val", "cal", "test")
FORECAST_WINDOW_HOURS = 24


def _existing_file(label: str, value: str, errors: list[str]) -> None:
    if not Path(value).is_file():
        errors.append(f"{label} is not a readable file: {value}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_with_neighbor_within(
    left: pd.DatetimeIndex, right: pd.DatetimeIndex, hours: int
) -> int:
    """Count left timestamps with any right timestamp less than ``hours`` away."""
    if left.empty or right.empty:
        return 0
    left_ns = np.sort(left.view("int64"))
    right_ns = np.sort(right.view("int64"))
    window = int(pd.Timedelta(hours=hours).value)
    lower = np.searchsorted(right_ns, left_ns - window, side="right")
    upper = np.searchsorted(right_ns, left_ns + window, side="left")
    return int(np.count_nonzero(upper > lower))


def _minimum_separation_seconds(
    left: pd.DatetimeIndex, right: pd.DatetimeIndex
) -> float | None:
    if left.empty or right.empty:
        return None
    left_ns = np.sort(left.view("int64"))
    right_ns = np.sort(right.view("int64"))
    positions = np.searchsorted(right_ns, left_ns)
    distances = []
    valid = positions < len(right_ns)
    if valid.any():
        distances.append(np.abs(left_ns[valid] - right_ns[positions[valid]]))
    valid = positions > 0
    if valid.any():
        distances.append(np.abs(left_ns[valid] - right_ns[positions[valid] - 1]))
    return float(min(np.min(values) for values in distances) / 1e9)


def build_split_audit(cfg, index_dir: Path) -> tuple[dict[str, object], list[str]]:
    """Audit source hashes, retained targets, and pairwise temporal separation."""
    errors: list[str] = []
    timestamps: dict[str, pd.DatetimeIndex] = {}
    splits: dict[str, object] = {}
    expected_hashes = cfg.data.flare_index.get("sha256", {})
    ordinal_column = str(cfg.data.get("ordinal_label_type", "max_goes_class"))
    numeric_column = str(
        cfg.data.get("filter_numeric_target_column", "max_intensity")
    )
    for split in SPLIT_KEYS:
        filename = cfg.data.flare_index.get(split)
        if filename is None:
            errors.append(f"data.flare_index.{split} is missing")
            continue
        path = index_dir / str(filename)
        if not path.is_file():
            errors.append(f"{split} index is not a readable file: {path}")
            continue
        digest = _sha256(path)
        expected = expected_hashes.get(split)
        if expected is None:
            errors.append(f"data.flare_index.sha256.{split} is missing")
        elif digest != str(expected):
            errors.append(f"{split} index SHA-256 mismatch: {digest}")
        frame = pd.read_csv(path)
        required_columns = {"timestamp", ordinal_column, numeric_column}
        missing = required_columns - set(frame.columns)
        if missing:
            errors.append(f"{path} lacks required columns: {sorted(missing)}")
            continue
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
        invalid_timestamps = int(parsed.isna().sum())
        duplicate_timestamps = int(parsed.duplicated().sum())
        if invalid_timestamps:
            errors.append(f"{split} has {invalid_timestamps} invalid timestamps")
        if duplicate_timestamps:
            errors.append(f"{split} has {duplicate_timestamps} duplicate timestamps")
        timestamps[split] = pd.DatetimeIndex(parsed.dropna().sort_values())

        try:
            manifest = build_ocqr_flare_manifest_audit(
                str(path),
                split_name=split,
                ordinal_label_column=ordinal_column,
                numeric_target_column=numeric_column,
                fq_max_intensity=cfg.data.get("fq_max_intensity_exclusion_threshold"),
                excluded_goes_classes=tuple(cfg.data.get("excluded_goes_classes", [])),
            )
            retained = filter_ocqr_flare_rows(
                frame,
                ordinal_label_column=ordinal_column,
                numeric_target_column=numeric_column,
                fq_max_intensity=cfg.data.get("fq_max_intensity_exclusion_threshold"),
                excluded_goes_classes=tuple(cfg.data.get("excluded_goes_classes", [])),
            )
            numeric = pd.to_numeric(retained[numeric_column], errors="coerce").to_numpy()
            invalid_numeric = int((~np.isfinite(numeric) | (numeric <= 0)).sum())
            if invalid_numeric:
                errors.append(f"{split} has {invalid_numeric} invalid retained numeric targets")
            consistent = np.ones(len(retained), dtype=bool)
            if not invalid_numeric:
                z = np.log10(numeric) + 9.0
                derived = np.searchsorted(np.asarray([2.0, 3.0, 4.0, 5.0]), z, side="right")
                supplied = retained[ordinal_column].map(_map_goes_class).to_numpy()
                consistent = derived == supplied
                if not consistent.all():
                    errors.append(
                        f"{split} has {int((~consistent).sum())} retained target-label-bin inconsistencies"
                    )
            splits[split] = {
                **manifest,
                "invalid_timestamp_count": invalid_timestamps,
                "duplicate_timestamp_count": duplicate_timestamps,
                "invalid_retained_numeric_target_count": invalid_numeric,
                "retained_target_label_bin_inconsistency_count": int((~consistent).sum()),
                "timestamp_min": timestamps[split].min().isoformat()
                if len(timestamps[split])
                else None,
                "timestamp_max": timestamps[split].max().isoformat()
                if len(timestamps[split])
                else None,
            }
        except Exception as exc:
            errors.append(f"{split} retained-population audit failed: {exc}")

    pairwise: dict[str, object] = {}
    for left, right in combinations(timestamps, 2):
        direct = int(len(timestamps[left].intersection(timestamps[right])))
        if direct:
            errors.append(f"{left}/{right} have {direct} directly overlapping timestamps")
        pairwise[f"{left}__{right}"] = {
            "direct_timestamp_overlap_count": direct,
            "minimum_separation_seconds": _minimum_separation_seconds(
                timestamps[left], timestamps[right]
            ),
            "left_rows_with_other_split_timestamp_within_24h": _rows_with_neighbor_within(
                timestamps[left], timestamps[right], FORECAST_WINDOW_HOURS
            ),
            "right_rows_with_other_split_timestamp_within_24h": _rows_with_neighbor_within(
                timestamps[right], timestamps[left], FORECAST_WINDOW_HOURS
            ),
        }

    development = [timestamps[key] for key in ("train", "val", "cal") if key in timestamps]
    chronological_test = None
    nonempty_development = [values for values in development if len(values)]
    if nonempty_development and "test" in timestamps and len(timestamps["test"]):
        development_max = max(values.max() for values in nonempty_development)
        chronological_test = bool(development_max < timestamps["test"].min())
        if not chronological_test:
            errors.append("future test timestamps do not strictly follow development timestamps")

    retained_train = (
        splits.get("train", {}).get("retained_manifest", {}).get("row_count", 0)
    )
    planned_steps = math.ceil(retained_train / int(cfg.data.batch_size)) * int(
        cfg.trainer.max_epochs
    )
    return {
        "schema_version": "surya-split-audit-v1",
        "forecast_window_hours": FORECAST_WINDOW_HOURS,
        "splits": splits,
        "pairwise_temporal_checks": pairwise,
        "future_test_strictly_after_development": chronological_test,
        "planned_optimizer_steps_upper_bound_before_image_filtering": planned_steps,
        "validation": {"passed": not errors, "errors": errors},
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("config", type=Path, help="Surya YAML configuration file")
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides")
    args = parser.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Configuration file not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
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
        required_columns = {
            "timestamp",
            str(data.get("filter_numeric_target_column", "max_intensity")),
            str(data.get("ordinal_label_type", "max_goes_class")),
        }
        for split in ("train", "val", "cal", "test"):
            filename = data.flare_index.get(split)
            if filename is None:
                errors.append(f"data.flare_index.{split} is missing")
                continue
            csv_path = index_dir / str(filename)
            _existing_file(f"{split} index", str(csv_path), errors)
            if csv_path.is_file():
                columns = set(pd.read_csv(csv_path, nrows=1).columns)
                missing = required_columns - columns
                if missing:
                    errors.append(f"{csv_path} lacks required columns: {sorted(missing)}")
        split_audit, audit_errors = build_split_audit(cfg, index_dir)
        errors.extend(audit_errors)
        if args.audit_output is not None:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(
                json.dumps(split_audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

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
    if args.audit_output is not None:
        print(f"  split audit: {args.audit_output}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}")


if __name__ == "__main__":
    main()
