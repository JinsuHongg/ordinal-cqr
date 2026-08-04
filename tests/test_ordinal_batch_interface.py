import tempfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from ordinal_cqr.datamodules.retina_mnist import OrdinalTargetDataset
from ordinal_cqr.datasets.adience import AdienceDataset
from ordinal_cqr.datasets.eyepacs import EyePACSDataset
from ordinal_cqr.datasets.flare_cls_datasets import (
    _map_goes_class,
    filter_ocqr_flare_rows,
)
from ordinal_cqr.datasets.utkface import UTKFaceDataset


def _image_tensor(_: Image.Image) -> torch.Tensor:
    return torch.zeros((3, 4, 4), dtype=torch.float32)


def test_class_only_adapter_emits_numeric_and_ordinal_targets() -> None:
    source = TensorDataset(torch.zeros((1, 3, 4, 4)), torch.tensor([2]))
    x, z, y_ord = OrdinalTargetDataset(source)[0]

    assert x.shape == (3, 1, 4, 4)
    assert int(z) == 2
    assert int(y_ord) == 2


def test_utkface_continuous_mode_emits_age_and_age_bin() -> None:
    with tempfile.TemporaryDirectory() as directory:
        filename = "42_0_0_sample.jpg"
        Image.new("RGB", (4, 4)).save(Path(directory) / filename)
        dataset = UTKFaceDataset(
            directory,
            [filename],
            thresholds=[20.0, 40.0, 60.0, 80.0],
            label_type="continuous",
            transform=_image_tensor,
        )

        _, z, y_ord = dataset[0]

    assert float(z) == 42.0
    assert int(y_ord) == 2


def test_class_embedded_datasets_emit_separate_ordinal_label() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (4, 4)).save(root / "eye.jpeg")
        eyepacs = EyePACSDataset(
            directory,
            ["eye.jpeg"],
            [3],
            label_type="continuous",
            transform=_image_tensor,
        )
        _, eye_z, eye_y_ord = eyepacs[0]

        Image.new("RGB", (4, 4)).save(root / "face.jpg")
        frame = pd.DataFrame(
            [{"image_path": "face.jpg", "label": 4, "continuous_label": 28.5}]
        )
        adience = AdienceDataset(
            frame,
            directory,
            transform=_image_tensor,
            label_type="continuous",
        )
        _, age_z, age_y_ord = adience[0]

    assert float(eye_z) == 3.0
    assert int(eye_y_ord) == 3
    assert float(age_z) == 28.5
    assert int(age_y_ord) == 4


def test_supplied_goes_classes_use_canonical_ordinal_mapping() -> None:
    assert [_map_goes_class(value) for value in ["FQ", "A1.0", "B7", "C2", "M1", "X1"]] == [
        0,
        0,
        1,
        2,
        3,
        4,
    ]
    try:
        _map_goes_class(pd.NA)
    except ValueError as error:
        assert "must not be missing" in str(error)
    else:
        raise AssertionError("Expected a missing GOES class label to raise.")


def test_solar_retained_population_rule_is_target_specific_and_inclusive() -> None:
    source = pd.DataFrame(
        {
            "id": ["fq-low", "fq-boundary", "fq-high", "a-high", "m-bad", "m-good"],
            "max_goes_class": [" fq ", "FQ", "FQ", "A1.0", " M0.9 ", "M1.0"],
            "max_intensity": [0.999e-7, 1.0e-7, 1.1e-7, 1.1e-7, 9.0e-6, 1.1e-5],
        },
        index=[10, 11, 12, 13, 14, 15],
    )

    retained = filter_ocqr_flare_rows(
        source,
        fq_max_intensity=1.0e-7,
        excluded_goes_classes=("M0.9",),
    )

    assert retained.index.tolist() == [10, 13, 15]
    assert retained["id"].tolist() == ["fq-low", "a-high", "m-good"]
    assert source["id"].tolist() == [
        "fq-low", "fq-boundary", "fq-high", "a-high", "m-bad", "m-good"
    ]
    assert _map_goes_class(" fq ") == 0


def test_solar_retained_population_rule_requires_numeric_target_column() -> None:
    source = pd.DataFrame({"max_goes_class": ["FQ"]})

    try:
        filter_ocqr_flare_rows(source, fq_max_intensity=1.0e-7)
    except ValueError as error:
        assert "max_intensity" in str(error)
    else:
        raise AssertionError("Expected a missing numeric target column to raise.")


def test_solar_retained_population_rule_is_compatible_when_disabled() -> None:
    source = pd.DataFrame({"max_goes_class": ["FQ", "B1.0"]})

    retained = filter_ocqr_flare_rows(source)

    assert retained.equals(source)
