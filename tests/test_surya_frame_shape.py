from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ordinal_cqr.datasets.flare_cls_datasets import FlareSuryaBenchDataset


def _frame(*, timestamps: list[pd.Timestamp]) -> xr.DataArray:
    return xr.DataArray(
        np.arange(len(timestamps) * 3 * 2 * 4, dtype=np.float32).reshape(
            len(timestamps), 3, 2, 4
        ),
        dims=("timestep", "channel", "y", "x"),
        coords={"timestep": timestamps, "channel": ["a", "b", "c"]},
    )


def test_canonicalize_frame_removes_singleton_time_axis() -> None:
    timestamp = pd.Timestamp("2010-01-01T00:00:00")

    result = FlareSuryaBenchDataset._canonicalize_frame(
        _frame(timestamps=[timestamp]), timestamp=timestamp
    )

    assert result.shape == (3, 2, 4)
    np.testing.assert_array_equal(result, np.arange(24, dtype=np.float32).reshape(3, 2, 4))


def test_canonicalize_frame_rejects_multiple_frames_for_one_timestamp() -> None:
    timestamp = pd.Timestamp("2010-01-01T00:00:00")

    with pytest.raises(ValueError, match="Expected exactly one Surya frame"):
        FlareSuryaBenchDataset._canonicalize_frame(
            _frame(timestamps=[timestamp, timestamp]), timestamp=timestamp
        )
