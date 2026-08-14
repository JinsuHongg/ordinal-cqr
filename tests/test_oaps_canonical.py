"""Focused tests for the Lu et al. greedy Ordinal APS implementation."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ordinal_cqr.explainability import (
    OrdinalAPSWrapper,
    oaps_entry_scores,
    oaps_prediction_sets,
)


def test_oaps_grows_from_mode_toward_more_probable_neighbor() -> None:
    probs = torch.tensor([[0.1, 0.4, 0.2, 0.3]])

    scores = oaps_entry_scores(probs)

    # Greedy order is 1 (mode), 2, 3, 0. Scores are the interval mass
    # immediately before each class enters that nested sequence.
    torch.testing.assert_close(scores, torch.tensor([[0.9, 0.0, 0.4, 0.6]]))
    assert oaps_prediction_sets(probs, 0.4).tolist() == [[False, True, True, False]]
    assert oaps_prediction_sets(probs, 0.6).tolist() == [[False, True, True, True]]


def test_oaps_adjacent_probability_ties_choose_upper_neighbor() -> None:
    probs = torch.tensor([[0.2, 0.6, 0.2]])

    scores = oaps_entry_scores(probs)

    # The authors' reference code uses a strict comparison for the left edge,
    # so an equal-probability tie selects the right edge first.
    torch.testing.assert_close(scores, torch.tensor([[0.8, 0.0, 0.6]]))
    assert oaps_prediction_sets(probs, 0.6).tolist() == [[False, True, True]]


def test_oaps_modal_ties_choose_lowest_class() -> None:
    probs = torch.tensor([[0.4, 0.2, 0.4]])

    scores = oaps_entry_scores(probs)

    torch.testing.assert_close(scores, torch.tensor([[0.0, 0.4, 0.6]]))
    assert oaps_prediction_sets(probs, 0.0).tolist() == [[True, False, False]]


def test_oaps_sets_are_nested_nonempty_contiguous_and_contain_mode() -> None:
    probs = torch.tensor(
        [[0.15, 0.10, 0.50, 0.20, 0.05], [0.40, 0.05, 0.10, 0.35, 0.10]]
    )
    previous = torch.zeros_like(probs, dtype=torch.bool)
    modes = probs.argmax(dim=1)

    for threshold in (0.0, 0.2, 0.5, 0.8, float("inf")):
        current = oaps_prediction_sets(probs, threshold)
        assert (previous <= current).all()
        assert current[torch.arange(len(probs)), modes].all()
        for row in current:
            labels = torch.where(row)[0]
            assert len(labels) > 0
            assert int(labels[-1] - labels[0] + 1) == len(labels)
        previous = current


def test_oaps_calibration_uses_supplied_ordinal_label_and_exact_rank() -> None:
    probs = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
    )
    numeric_targets = torch.tensor([99.0, 99.0, 99.0])
    ordinal_labels = torch.tensor([0, 2, 1])
    loader = DataLoader(
        TensorDataset(probs.log(), numeric_targets, ordinal_labels), batch_size=3
    )
    wrapper = OrdinalAPSWrapper(nn.Identity(), num_classes=3, alpha=0.25)

    wrapper.calibrate(loader)

    expected = oaps_entry_scores(probs).gather(1, ordinal_labels[:, None]).squeeze(1)
    torch.testing.assert_close(wrapper.q_hat, torch.kthvalue(expected, 3).values)


def test_oaps_rejects_nonoriginal_classwise_variant() -> None:
    with pytest.raises(ValueError, match="pooled marginal calibration"):
        OrdinalAPSWrapper(nn.Identity(), class_wise=True)


def test_oaps_rejects_nonfinite_probabilities() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        oaps_entry_scores(torch.tensor([[0.5, float("nan")]]))
