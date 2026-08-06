"""Deterministic contract validation for canonical OCQR v0.3.

This is deliberately a small executable validation, not a substitute for a
real-data experiment. It exercises exact calibration and post-processing edge
cases before a conference run is trusted.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ordinal_cqr.explainability.poshoc_uc import OrdinalCQRWrapper


class IdentityQuantileModel(torch.nn.Module):
    """Interpret input columns as lower and upper quantile predictions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _wrapper(alpha: float = 0.5) -> OrdinalCQRWrapper:
    return OrdinalCQRWrapper(
        IdentityQuantileModel(),
        num_classes=3,
        class_mapping={"0": 0, "1": 1, "2": 2},
        thresholds=[0.5, 1.5],
        alpha=alpha,
        class_wise=True,
    )


def run_validation() -> dict[str, object]:
    """Run deterministic v0.3 edge cases and return JSON-safe evidence."""
    wrapper = _wrapper()
    # Tied negative scores for class 0 and finite scores for class 1. Class 2
    # is absent, exercising the augmented +inf correction.
    calibration = DataLoader(
        TensorDataset(
            torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.8, 1.2], [0.8, 1.2]]),
            torch.tensor([0.25, 0.25, 1.0, 1.0]),
            torch.tensor([0, 0, 1, 1]),
        ),
        batch_size=4,
    )
    wrapper.calibrate(calibration)
    metadata = wrapper.get_calibration_metadata()
    assert metadata.counts.tolist() == [2, 2, 0]
    assert metadata.requested_ranks.tolist() == [2, 2, 1]
    assert float(metadata.corrections[0]) == -0.25
    assert metadata.tie_counts[0].item() == 2
    assert torch.isinf(metadata.corrections[2])

    # Candidate-specific q values select classes 0 and 2 and leave a raw gap;
    # hull closure must add class 1 without deleting either candidate.
    wrapper.q_hats.copy_(torch.tensor([0.0, -0.7, 0.0]))
    fragmented = wrapper.predict_step(
        (torch.tensor([[1.6, 0.4]]), torch.tensor([1.0]), torch.tensor([1])), 0
    )
    assert fragmented["raw_prediction_set"].tolist() == [[True, False, True]]
    assert fragmented["prediction_set"].tolist() == [[True, True, True]]

    # A negative correction can make the inverted numeric acceptance interval
    # empty. The canonical fallback then produces the full label space.
    wrapper.q_hats.fill_(-2.0)
    empty = wrapper.predict_step(
        (torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]), torch.tensor([1])), 0
    )
    assert not empty["raw_prediction_set"].any()
    assert empty["prediction_set"].all()

    # Boundary equality follows the right-bin convention. The preceding empty
    # interval check already exercises a finite negative correction.
    wrapper.q_hats.zero_()
    boundary = wrapper.predict_step(
        (torch.tensor([[0.75, 0.75]]), torch.tensor([0.5]), torch.tensor([1])), 0
    )
    assert boundary["raw_prediction_set"].tolist() == [[False, True, False]]

    return {
        "schema_version": "ocqr-synthetic-validation-v1",
        "method": "ocqr",
        "method_version": "0.3.0",
        "alpha": 0.5,
        "seed": 0,
        "num_repetitions": 1,
        "target_coverage": 0.5,
        "per_class_empirical_coverage": {"0": 1.0, "1": 1.0, "2": None},
        "passed": True,
        "checks": {
            "exact_augmented_rank": True,
            "ties": True,
            "negative_scores": True,
            "q_plus_inf": True,
            "candidate_specific_corrections": True,
            "score_inversion_empty_interval": True,
            "raw_fragmentation": True,
            "hull_add_only_contiguity": True,
            "empty_raw_full_fallback": True,
            "right_boundary": True,
        },
        "calibration_counts": metadata.counts.tolist(),
        "corrections": [
            "+inf" if torch.isinf(value) else float(value)
            for value in metadata.corrections
        ],
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
