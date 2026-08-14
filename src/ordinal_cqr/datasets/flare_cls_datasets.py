import hashlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr

from ordinal_cqr.datasets.surya_zarr import (
    discover_surya_year_groups,
    open_surya_year_dataset,
)
from loguru import logger as lgr_logger
from omegaconf import OmegaConf
try:
    from terratorch_surya.datasets.helio import HelioNetCDFDataset
except ImportError:
    HelioNetCDFDataset = object
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


def filter_ocqr_flare_rows(
    flare_index: pd.DataFrame,
    *,
    ordinal_label_column: str = "max_goes_class",
    numeric_target_column: str = "max_intensity",
    fq_max_intensity: float | None = None,
    excluded_goes_classes: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply a prespecified retained-population rule to a flare index.

    The canonical raw-flux OCQR variant excludes FQ horizons whose background
    flux reaches the B-class threshold and known label/flux inconsistencies.
    Filtering occurs before timestamp availability filtering so every split
    follows the same declared population rule.
    """
    exclusion_masks = ocqr_flare_row_exclusion_masks(
        flare_index,
        ordinal_label_column=ordinal_label_column,
        numeric_target_column=numeric_target_column,
        fq_max_intensity=fq_max_intensity,
        excluded_goes_classes=excluded_goes_classes,
    )
    excluded = pd.Series(False, index=flare_index.index, dtype=bool)
    for mask in exclusion_masks.values():
        excluded |= mask
    return flare_index.loc[~excluded].copy()


def ocqr_flare_row_exclusion_masks(
    flare_index: pd.DataFrame,
    *,
    ordinal_label_column: str = "max_goes_class",
    numeric_target_column: str = "max_intensity",
    fq_max_intensity: float | None = None,
    excluded_goes_classes: tuple[str, ...] = (),
) -> dict[str, pd.Series]:
    """Return named, non-mutating exclusion masks for the OCQR solar policy."""
    required_columns = {ordinal_label_column}
    if fq_max_intensity is not None:
        required_columns.add(numeric_target_column)
    missing_columns = required_columns.difference(flare_index.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Flare index is missing required columns: {missing}.")

    labels = (
        flare_index[ordinal_label_column].astype("string").str.strip().str.upper()
    )
    masks: dict[str, pd.Series] = {}
    if fq_max_intensity is not None:
        numeric_target = pd.to_numeric(
            flare_index[numeric_target_column], errors="coerce"
        )
        invalid_fq_target = (labels == "FQ") & ~np.isfinite(numeric_target)
        if invalid_fq_target.any():
            raise ValueError(
                "FQ rows must have finite numeric targets when applying the "
                "OCQR retained-population rule."
            )
        masks["fq_intensity_at_or_above_threshold"] = (
            (labels == "FQ") & (numeric_target >= fq_max_intensity)
        )
    if excluded_goes_classes:
        excluded = {str(label).strip().upper() for label in excluded_goes_classes}
        masks["excluded_goes_class"] = labels.isin(excluded)
    return masks


def build_ocqr_flare_manifest_audit(
    flare_index_path: str,
    *,
    split_name: str,
    ordinal_label_column: str = "max_goes_class",
    numeric_target_column: str = "max_intensity",
    fq_max_intensity: float | None = None,
    excluded_goes_classes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build reproducible source and retained-manifest provenance for one split."""
    source_path = Path(flare_index_path)
    raw = pd.read_csv(source_path)
    masks = ocqr_flare_row_exclusion_masks(
        raw,
        ordinal_label_column=ordinal_label_column,
        numeric_target_column=numeric_target_column,
        fq_max_intensity=fq_max_intensity,
        excluded_goes_classes=excluded_goes_classes,
    )
    excluded = pd.Series(False, index=raw.index, dtype=bool)
    for mask in masks.values():
        excluded |= mask
    retained = raw.loc[~excluded].copy()
    labels = retained[ordinal_label_column].map(_map_goes_class)
    retained_csv = retained.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "split": split_name,
        "source_index": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "row_count": int(len(raw)),
        },
        "retained_manifest": {
            "sha256": hashlib.sha256(retained_csv).hexdigest(),
            "row_count": int(len(retained)),
            "hash_representation": "UTF-8 CSV after policy filtering; source row order and columns preserved",
            "class_counts": {
                str(class_id): int((labels == class_id).sum())
                for class_id in range(5)
            },
        },
        "exclusions": {
            "total": int(excluded.sum()),
            "by_reason": {
                reason: int(mask.sum()) for reason, mask in masks.items()
            },
        },
    }


def _map_goes_class(value: object) -> int:
    """Map a supplied GOES/FQ class value to the canonical ordinal index."""
    if pd.isna(value):
        raise ValueError("GOES class label must not be missing.")
    normalized_value = str(value).strip().upper()
    if normalized_value == "FQ":
        return 0
    mapping = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    class_name = normalized_value[0]
    if class_name not in mapping:
        raise ValueError(f"Unknown GOES class label: {value!r}.")
    return mapping[class_name]


class FlareHelioviewerRegDataset(Dataset):
    """Dataset for solar flare regression using Helioviewer images.

    This dataset loads sequences of solar images from Helioviewer and pairs them
    with flare intensity labels for regression tasks.

    Args:
        input_index_path: Path to the CSV file containing image metadata.
        input_time_delta: List of time offsets (in minutes) for the input sequence.
        input_stat_path: Path to the YAML file containing data statistics.
        flare_index_path: Path to the CSV file containing flare labels.
        limb_mask_path: Path to the NPY file containing the solar limb mask.
        scaler_mul: Multiplicative factor for input scaling.
        scaler_shift: Additive shift for input scaling.
        scaler_div: Divisor for input scaling.
        label_type: Column name in flare_index for the target label.
        target_norm_type: Type of normalization for the target (e.g., 'log').
        phase: Dataset phase ('train', 'val', or 'test').

    Attributes:
        input_time_delta: List of time offsets for the input sequence.
        stats: Loaded data statistics.
        limb_mask: Solar limb mask array.
        scaler_mul: Multiplicative factor for input scaling.
        scaler_shift: Additive shift for input scaling.
        scaler_div: Divisor for input scaling.
        label_type: Column name for the target label.
        target_norm_type: Type of normalization for the target.
        phase: Dataset phase.
        index: Image metadata index.
        flare_index: Flare labels index.
        augment: Augmentation pipeline for training.
        valid_timestamps: List of timestamps with valid input sequences and labels.
    """

    def __init__(
        self,
        input_index_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        flare_index_path: str,
        limb_mask_path: str,
        scaler_mul: float,
        scaler_shift: float,
        scaler_div: float,
        label_type: str,
        target_norm_type: str,
        phase: str,
        ordinal_label_type: str = "max_goes_class",
    ):
        super().__init__()
        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path)  # input data statistics
        self.limb_mask = np.load(limb_mask_path)
        self.scaler_mul = scaler_mul
        self.scaler_shift = scaler_shift
        self.scaler_div = scaler_div
        self.label_type = label_type
        self.target_norm_type = target_norm_type
        self.phase = phase
        self.ordinal_label_type = ordinal_label_type

        # load index file
        self.index = pd.read_csv(input_index_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"])
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(self.flare_index["timestamp"])
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)
        self._get_valid_indices()
        lgr_logger.info(f"{self.phase} instances: {self.__len__()}")

        # Define Augmentation (Only for Training)
        if self.phase == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=11),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        else:
            self.augment = None  # No augmentation for validation/test

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            The number of valid timestamps.
        """
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int):
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            ``(X, Z, Y_ord, timestamp)``. ``Y_ord`` comes from the supplied
            flare-class field; the timestamp is retained as provenance.
        """
        current_time = self.valid_timestamps[idx]

        # Calculate all timestamps needed for this sample
        # e.g., current_time - 10min, current_time - 0min
        required_times = [
            current_time + pd.Timedelta(minutes=dt) for dt in self.input_time_delta
        ]

        # Load and stack all images
        images = []
        for t in required_times:
            # We know 't' exists because of _get_valid_indices validation
            img_path = self.index.loc[t, "input"]
            img = read_image(img_path)
            images.append(self.transform(img))

        # Stack along the first dimension (C, H, W) -> (Num_Frames, C, H, W)
        x = torch.stack(images, dim=1)
        x = x.float()

        target = self.transform_target(
            self.flare_index.loc[current_time, self.label_type]
        )

        ordinal_label = _map_goes_class(
            self.flare_index.loc[current_time, self.ordinal_label_type]
        )
        return (
            x,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(ordinal_label, dtype=torch.long),
            current_time.value,
        )

    def _get_valid_indices(self):
        time_deltas = pd.to_timedelta(self.input_time_delta, unit="min")
        idx = self.index.index

        valid_mask = np.ones(len(idx), dtype=bool)
        for dt in time_deltas:
            required_times = idx + dt
            has_required_time = required_times.isin(idx)
            valid_mask = valid_mask & has_required_time

        # Get timestamps that have valid input sequences
        valid_sequence_timestamps = idx[valid_mask]

        # Keep only timestamps that are ALSO in flare_index
        # This assumes both indices are DatetimeIndex
        final_valid_timestamps = valid_sequence_timestamps.intersection(
            self.flare_index.index
        )

        self.valid_timestamps = sorted(final_valid_timestamps)

    def transform(self, data):
        """Applies transformations and scaling to the input image.

        Args:
            data: Input image tensor.

        Returns:
            Transformed and scaled image tensor.
        """
        data = data.float()

        # Apply Augmentation (Only if defined)
        if hasattr(self, "augment") and self.augment is not None:
            data = self.augment(data)

        data = data * self.limb_mask + 127.5 * (
            1 - self.limb_mask
        )  # limb regions become zero after the norm
        scaled = data * self.scaler_mul
        shift = scaled + self.scaler_shift

        return shift / self.scaler_div

    def transform_target(self, target):
        """Applies normalization to the target label.

        Args:
            target: Raw target value.

        Returns:
            Normalized target value.
        """
        match self.target_norm_type:
            case "log":
                if target == 0:
                    print("target is zero?")
                return np.log10(target) + 9


class FlareSuryaBenchDataset(Dataset):
    """Dataset for solar flare regression/classification using surya-bench data.

    Loads sequences of HMI magnetogram images from a Zarr store and pairs them
    with flare intensity labels for regression or binary classification tasks.

    Args:
        input_zarr_path: Path to the root Zarr store containing year groups.
        input_time_delta: List of time offsets (in minutes) for the input sequence.
            e.g. [-10, 0] loads two frames: 10 minutes before and at current_time.
        input_stat_path: Path to the YAML file containing dataset mean and std statistics.
        flare_index_path: Path to the CSV file containing flare labels with a
            'timestamp' column and one or more label columns.
        limb_mask_path: Path to the NPY file containing the solar limb mask (512x512).
        scaler_mul: Multiplicative scaler applied during preprocessing.
        scaler_shift: Shift scaler applied during preprocessing.
        scaler_div: Divisive scaler applied during preprocessing.
        label_type: Column name in flare_index for the target label.
        target_norm_type: Normalization strategy for the target label.
            'log'        — log10(target) + 9, returns float.
            'binary'     — casts existing 0/1 column to int, returns long.
            'multi_class'— maps GOES class strings to 0-4 int, returns long.
        phase: Dataset phase, one of 'train', 'val', or 'test'.
            Training phase applies random augmentations.

    Attributes:
        years: Sorted list of year group strings found in the Zarr store.
        index_timestamps: Flat DatetimeIndex of all available image timestamps.
        input_time_delta: List of time offsets (minutes) for the input sequence.
        stats: Loaded data statistics (mean, std) from the YAML file.
        limb_mask: Solar limb mask array of shape (512, 512).
        label_type: Column name for the target label.
        target_norm_type: Normalization strategy for the target label.
        phase: Dataset phase ('train', 'val', or 'test').
        flare_index: DataFrame of flare labels indexed by timestamp.
        augment: Augmentation pipeline for training, or None for val/test.
        valid_timestamps: Sorted list of timestamps that have both a complete
            input sequence and a matching entry in flare_index.
    """

    def __init__(
        self,
        input_zarr_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        flare_index_path: str,
        limb_mask_path: str,
        label_type: str,
        target_norm_type: str,
        phase: str,
        channel: str = "hmi_m",
        channels: list[str] | None = None,
        expected_channels: int | None = None,
        expected_image_size: list[int] | tuple[int, int] | None = None,
        require_per_channel_stats: bool = False,
        apply_limb_mask: bool = False,
        limb_mask_channels: list[str] | None = None,
        limb_mask_channel_prefixes: list[str] | None = None,
        ordinal_label_type: str = "max_goes_class",
        filter_numeric_target_column: str = "max_intensity",
        fq_max_intensity: float | None = None,
        excluded_goes_classes: tuple[str, ...] = (),
    ):
        super().__init__()
        self.channels = list(channels) if channels is not None else None
        self.channel = channel
        self.expected_channels = expected_channels
        self.expected_image_size = (
            tuple(expected_image_size) if expected_image_size is not None else None
        )
        self.require_per_channel_stats = require_per_channel_stats
        self.apply_limb_mask = apply_limb_mask
        self.limb_mask_channels = {
            str(channel).lower() for channel in (limb_mask_channels or [])
        }
        self.limb_mask_channel_prefixes = tuple(
            str(prefix).lower() for prefix in (limb_mask_channel_prefixes or [])
        )

        # Find year groups from Zarr store
        self.years = discover_surya_year_groups(input_zarr_path)

        # Open each year group as a lazy xarray DataArray
        self._arrays: dict[str, xr.DataArray] = {}
        for year in self.years:
            ds = open_surya_year_dataset(input_zarr_path, year)

            # Support the new stacked Zarr structure
            stacked_name = next(
                (name for name, value in ds.data_vars.items() if "channel" in value.dims),
                None,
            )
            if stacked_name is not None:
                da = ds[stacked_name]
                if "channel_names" in da.attrs:
                    da = da.assign_coords(channel=da.attrs["channel_names"])

                available_channels = [str(value) for value in da.coords["channel"].values]
                if self.channels is None:
                    self.channels = available_channels
                    lgr_logger.info(
                        f"Auto-selected Zarr channels in stored order: {self.channels}"
                    )
                selected_channels = self.channels
                missing_channels = set(selected_channels).difference(available_channels)
                if missing_channels:
                    raise ValueError(
                        f"Zarr group {year} is missing requested channels: "
                        f"{sorted(missing_channels)}. Available: {available_channels}."
                    )
                da = da.sel(channel=selected_channels)

                # Ensure the time dimension is named 'timestep'
                if "time" in da.dims:
                    da = da.rename({"time": "timestep"})

                self._arrays[year] = da
            else:
                # Fallback to old format
                if self.channels is None:
                    self.channels = list(ds.data_vars)
                    lgr_logger.info(
                        f"Auto-selected Zarr variables in stored order: {self.channels}"
                    )
                selected_channels = self.channels
                missing_channels = set(selected_channels).difference(ds.data_vars)
                if missing_channels:
                    raise ValueError(
                        f"Zarr group {year} is missing requested channels: "
                        f"{sorted(missing_channels)}."
                    )
                self._arrays[year] = xr.concat(
                    [ds[name] for name in selected_channels],
                    dim=pd.Index(selected_channels, name="channel"),
                )

        first_array = next(iter(self._arrays.values()))
        channel_count = int(first_array.sizes.get("channel", 1))
        if self.expected_channels is not None and channel_count != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} input channels, found {channel_count}."
            )
        spatial_shape = tuple(first_array.shape[-2:])
        if self.expected_image_size is not None and spatial_shape != self.expected_image_size:
            raise ValueError(
                f"Expected image size {self.expected_image_size}, found {spatial_shape}."
            )

        # Build flat index — single concatenated DatetimeIndex
        index_timestamps = []
        for da in self._arrays.values():
            if "timestep" in da.coords:
                index_timestamps.append(pd.DatetimeIndex(da.coords["timestep"].values))
            else:
                lgr_logger.warning("No 'timestep' coordinates found in Zarr store!")
                index_timestamps.append(pd.DatetimeIndex(range(len(da))))
        self.index_timestamps = pd.DatetimeIndex(np.concatenate(index_timestamps))

        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path)
        self.limb_mask = np.load(limb_mask_path)
        if self.limb_mask.shape != spatial_shape:
            raise ValueError(
                f"Limb mask shape {self.limb_mask.shape} does not match image size "
                f"{spatial_shape}."
            )
        self._mask_channel_flags = np.asarray(
            [
                self.apply_limb_mask
                or channel.lower() in self.limb_mask_channels
                or channel.lower().startswith(self.limb_mask_channel_prefixes)
                for channel in self.channels
            ],
            dtype=bool,
        )
        if (
            not self.apply_limb_mask
            and (self.limb_mask_channels or self.limb_mask_channel_prefixes)
            and not self._mask_channel_flags.any()
        ):
            raise ValueError(
                "The configured selective limb-mask policy did not match any "
                f"input channels: {self.channels}."
            )
        self.label_type = label_type
        self.target_norm_type = target_norm_type
        self.phase = phase
        self.ordinal_label_type = ordinal_label_type

        raw_flare_index = pd.read_csv(flare_index_path)
        self.flare_index = filter_ocqr_flare_rows(
            raw_flare_index,
            ordinal_label_column=ordinal_label_type,
            numeric_target_column=filter_numeric_target_column,
            fq_max_intensity=fq_max_intensity,
            excluded_goes_classes=excluded_goes_classes,
        )
        excluded_count = len(raw_flare_index) - len(self.flare_index)
        if excluded_count:
            lgr_logger.info(
                f"{self.phase}: excluded {excluded_count} rows using the "
                "configured OCQR retained-population rule."
            )
        self.flare_index["timestamp"] = pd.to_datetime(self.flare_index["timestamp"])
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        self._get_valid_indices()
        lgr_logger.info(f"{self.phase} instances: {self.__len__()}")

    def get_by_timestamp(self, timestamp: str | pd.Timestamp) -> np.ndarray:
        """Fetch a single HMI image by its exact timestamp.

        Args:
            timestamp: Timestamp string (e.g. '2015-06-21 12:00:00') or pd.Timestamp.

        Returns:
            Image array of shape (C, H, W) as float32.

        Raises:
            KeyError: If the year of the timestamp is not loaded in this dataset.
        """
        ts = pd.Timestamp(timestamp)
        year = str(ts.year)
        if year not in self._arrays:
            raise KeyError(f"Year {year} not loaded in this dataset.")
        image = self._arrays[year].sel(timestep=ts)
        return self._canonicalize_frame(image, timestamp=ts)

    @staticmethod
    def _canonicalize_frame(image: xr.DataArray, *, timestamp: pd.Timestamp) -> np.ndarray:
        """Return one selected Surya frame in ``(channel, height, width)`` order.

        A scalar xarray selection normally removes its temporal dimension. It
        can retain a singleton temporal axis, however, depending on the Zarr
        index layout. Standardization is per channel, so forwarding that axis
        as the leading dimension would incorrectly make a 13-channel image
        look like a one-channel image.
        """
        for time_dim in ("timestep", "time"):
            if time_dim not in image.dims:
                continue
            count = int(image.sizes[time_dim])
            if count != 1:
                raise ValueError(
                    f"Expected exactly one Surya frame at {timestamp}, found "
                    f"{count} along {time_dim!r}."
                )
            image = image.isel({time_dim: 0}, drop=True)

        if "channel" not in image.dims:
            raise ValueError(
                f"Surya frame at {timestamp} has no channel dimension: {image.dims}."
            )
        spatial_dims = [dim for dim in image.dims if dim != "channel"]
        if len(spatial_dims) != 2:
            raise ValueError(
                f"Expected two spatial dimensions for Surya frame at {timestamp}, "
                f"found {spatial_dims}."
            )
        return image.transpose("channel", *spatial_dims).values.astype(np.float32)

    def __len__(self) -> int:
        """Returns the number of valid samples in the dataset.

        Returns:
            Number of valid timestamps.
        """
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of:
            x       — input tensor of shape (C, num_frames, H, W).
                target  — scalar target tensor (float32 for 'log', long for 'binary').
                Y_ord  — ordinal class read from the supplied flare-class field.
                timestamp — int64 nanoseconds retained as sample provenance.
        """
        current_time = self.valid_timestamps[idx]

        required_times = [
            current_time + pd.Timedelta(minutes=dt) for dt in self.input_time_delta
        ]

        images = []
        for t in required_times:
            img = self.get_by_timestamp(t)
            images.append(self.transform(img))

        # (C, H, W) per frame → stack → (C, num_frames, H, W)
        x = torch.stack(images, dim=1).float()

        raw_target = self.flare_index.loc[current_time, self.label_type]
        target = self.transform_target(raw_target)

        dtype = torch.long if self.target_norm_type in ["binary", "multi_class"] else torch.float32
        if self.target_norm_type in ["binary", "multi_class"]:
            ordinal_label = int(target)
        else:
            ordinal_label = _map_goes_class(
                self.flare_index.loc[current_time, self.ordinal_label_type]
            )
        return (
            x,
            torch.tensor(target, dtype=dtype),
            torch.tensor(ordinal_label, dtype=torch.long),
            current_time.value,
        )

    def _get_valid_indices(self) -> None:
        """Builds self.valid_timestamps by filtering index_timestamps to only those
        where every required input frame exists and a flare label is available.
        """
        time_deltas = pd.to_timedelta(self.input_time_delta, unit="min")
        idx = self.index_timestamps

        valid_mask = np.ones(len(idx), dtype=bool)
        for dt in time_deltas:
            required_times = idx + dt
            has_required_time = required_times.isin(idx)
            valid_mask = valid_mask & has_required_time

        valid_sequence_timestamps = idx[valid_mask]
        final_valid_timestamps = valid_sequence_timestamps.intersection(
            self.flare_index.index
        )
        self.valid_timestamps = sorted(final_valid_timestamps)

    def transform(self, arr: np.ndarray) -> torch.Tensor:
        """Applies signed log1p transform and standardization to an input image.

        The signed log1p transform compresses the dynamic range of magnetogram
        values while preserving the sign: f(x) = sign(x) * log1p(|x|).
        The result is then standardized using the precomputed dataset mean and std.

        Args:
            arr: Input image array of shape (C, H, W), float32.

        Returns:
            Transformed image tensor of shape (C, H, W).
        """
        arr_transformed = np.sign(arr) * np.log1p(np.abs(arr))
        mean = np.asarray(self.stats.mean, dtype=np.float32)
        std = np.asarray(self.stats.std, dtype=np.float32)
        if self.require_per_channel_stats and (mean.ndim != 1 or std.ndim != 1):
            raise ValueError(
                "This configuration requires one mean and std per input channel."
            )
        if mean.ndim == 1:
            mean = mean[:, np.newaxis, np.newaxis]
        if std.ndim == 1:
            std = std[:, np.newaxis, np.newaxis]
        if mean.size not in (1, arr.shape[0]) or std.size not in (1, arr.shape[0]):
            raise ValueError(
                "Statistics must be scalar or have one mean/std per input channel; "
                f"got mean shape {mean.shape}, std shape {std.shape}, and "
                f"{arr.shape[0]} channels."
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Input statistics must be finite and have positive standard deviations.")
        arr_normalized = (arr_transformed - mean) / std
        if self._mask_channel_flags.any():
            channel_masks = np.where(
                self._mask_channel_flags[:, np.newaxis, np.newaxis],
                self.limb_mask[np.newaxis, :, :],
                1.0,
            )
            arr_normalized = arr_normalized * channel_masks
        return torch.from_numpy(arr_normalized)

    def transform_target(self, target: float | str) -> float | int:
        """Applies normalization to the target label.

        Args:
            target: Raw target value from flare_index.

        Returns:
            For 'log'         — float: log10(target) + 9.
            For 'binary'      — int: 0 or 1 (cast from existing binary column).
            For 'multi_class' — int: 0-4 (maps GOES class).

        Raises:
            ValueError: If target_norm_type is not 'log', 'binary', or 'multi_class'.
            ValueError: If target_norm_type is 'log' and target is zero.
        """
        match self.target_norm_type:
            case "log":
                if target == 0:
                    raise ValueError(
                        f"target is zero at — log10(0) is undefined. "
                        f"Check your flare_index for label_type='{self.label_type}'."
                    )
                return np.log10(target) + 9

            case "binary":
                return int(target)

            case "multi_class":
                if pd.isna(target) or target == "FQ":
                    return 0
                target_str = str(target).upper()
                return _map_goes_class(target_str)

            case _:
                raise ValueError(
                    f"Unknown target_norm_type: '{self.target_norm_type}'. "
                    f"Expected 'log', 'binary', or 'multi_class'."
                )


class FlareSuryaClsDataset(HelioNetCDFDataset):
    """Dataset for solar flare classification using Surya/SDO data.

    This dataset extends HelioNetCDFDataset to include flare classification
    labels from a flare index file.

    Args:
        sdo_data_root_path: Root directory for SDO data.
        index_path: Path to the NetCDF index file.
        flare_index_path: Path to the CSV file containing flare labels.
        time_delta_input_minutes: List of time offsets for input.
        time_delta_target_minutes: Time offset for the target.
        n_input_timestamps: Number of input timestamps.
        rollout_steps: Number of rollout steps.
        scalers: Scalers for data normalization.
        num_mask_aia_channels: Number of AIA channels to mask.
        drop_hmi_probability: Probability of dropping HMI data.
        use_latitude_in_learned_flow: Whether to use latitude in learned flow.
        channels: List of channels to use.
        phase: Dataset phase ('train', 'val', or 'test').
        pooling: Pooling factor.
        random_vert_flip: Whether to apply random vertical flip.

    Attributes:
        flare_index: Flare labels index.
        valid_indices: List of valid timestamps.
        adjusted_length: Number of valid samples.
    """

    def __init__(
        self,
        sdo_data_root_path: str,
        index_path: str,
        flare_index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
    ):
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
        )

        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self) -> list:
        """Filters timestamps to include only those present in the flare index.

        Returns:
            List of valid timestamps.
        """
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            The adjusted length of the dataset.
        """
        return self.adjusted_length


@hydra.main(config_path="../../configs/", config_name="alexnet_helioviewer_config.yaml")
def main(cfg):
    FlareHelioviewerRegDataset(
        task=cfg.experiment.task,
        index_path=cfg.data.index_path.train,
        input_time_delta=cfg.data.input_time_delta,
        input_stat_path=cfg.data.input_stat_path,
        limb_mask_path=cfg.data.limb_mask_path,
        scaler_mul=cfg.data.scaler_mul,
        scaler_shift=cfg.data.scaler_shift,
        scaler_div=cfg.data.scaler_div,
        label_type=cfg.data.label_type,
        phase="training",
    )


if __name__ == "__main__":
    main()
