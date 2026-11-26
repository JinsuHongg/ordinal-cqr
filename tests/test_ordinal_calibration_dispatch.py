import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ordinal_cqr.explainability import APSWrapper, ClsCPWrapper, OrdinalAPSWrapper

SCRIPT = Path(__file__).parents[1] / "scripts/experiments/ordinal_calibration.py"
SPEC = importlib.util.spec_from_file_location("ordinal_calibration", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_build_classifier_uq_wrapper_supports_lac_and_aps() -> None:
    model = torch.nn.Linear(4, 5)
    cfg = OmegaConf.create(
        {
            "uc": {
                "significance_level": 0.1,
                "num_classes": 5,
                "class_wise": False,
                "class_mapping": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
                "thresholds": [0.5, 1.5, 2.5, 3.5],
            }
        }
    )

    assert isinstance(MOD.build_classifier_uq_wrapper("lac", model, cfg), ClsCPWrapper)
    assert isinstance(MOD.build_classifier_uq_wrapper("aps", model, cfg), APSWrapper)
    assert isinstance(
        MOD.build_classifier_uq_wrapper("oaps", model, cfg), OrdinalAPSWrapper
    )
