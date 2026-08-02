import torch
from torch.utils.data import DataLoader, TensorDataset

from ocqr_solar.explainability.poshoc_uc import OrdinalCQRWrapper
from ocqr_solar.metrics.classification_metrics import ClassificationUQMetrics


class IdentityQuantileModel(torch.nn.Module):
    """Treat the two input columns as lower and upper quantile predictions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_wrapper(
    *,
    class_wise: bool = True,
    alpha: float = 0.5,
    allow_derived_labels: bool = False,
) -> OrdinalCQRWrapper:
    return OrdinalCQRWrapper(
        IdentityQuantileModel(),
        num_classes=3,
        class_mapping={"low": 0, "middle": 1, "high": 2},
        thresholds=[0.5, 1.5],
        alpha=alpha,
        class_wise=class_wise,
        allow_derived_labels=allow_derived_labels,
    )


def test_mondrian_calibration_uses_exact_order_statistic_and_safe_empty_class() -> None:
    wrapper = make_wrapper()
    # Class 0 scores are 0.1 and 0.4. With n=2, alpha=.5, k=2,
    # the finite-sample correction is exactly the second order statistic (0.4).
    predictions = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    targets = torch.tensor([0.0, 0.0, 1.0])
    loader = DataLoader(
        TensorDataset(predictions, targets, torch.tensor([0, 0, 1])), batch_size=3
    )

    wrapper.calibrate(loader)

    torch.testing.assert_close(wrapper.q_hats[:2], torch.tensor([0.4, 0.2]))
    assert torch.isinf(wrapper.q_hats[2])
    assert wrapper.class_supported.tolist() == [True, True, False]
    assert wrapper.calibration_counts.tolist() == [2, 1, 0]
    assert wrapper.calibration_ranks.tolist() == [2, 1, 1]
    metadata = wrapper.get_calibration_metadata()
    assert metadata["counts"].tolist() == [2, 1, 0]
    assert metadata["ranks"].tolist() == [2, 1, 1]
    assert metadata["supported"].tolist() == [True, True, False]
    torch.testing.assert_close(
        metadata["corrections"][:2], torch.tensor([0.4, 0.2])
    )
    assert torch.isinf(metadata["corrections"][2])


def test_candidate_specific_membership_is_hulled_and_crossing_is_ordered() -> None:
    wrapper = make_wrapper()
    wrapper.q_hats.copy_(torch.tensor([0.0, -0.7, 0.0]))
    wrapper.class_supported.fill_(True)

    # Reversed raw quantiles exercise crossing protection. Classes 0 and 2 are
    # candidate matches while class 1 is not; the ordinal hull must fill class 1.
    batch = (torch.tensor([[1.6, 0.4]]), torch.tensor([1.0]), torch.tensor([1]))
    output = wrapper.predict_step(batch, 0)

    assert output["raw_prediction_set"].tolist() == [[True, False, True]]
    assert output["prediction_set"].tolist() == [[True, True, True]]
    assert output["target"].tolist() == [1]


def test_unattainable_rank_and_empty_raw_set_use_conservative_fallbacks() -> None:
    wrapper = make_wrapper(alpha=0.1)
    predictions = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([0.0])
    loader = DataLoader(
        TensorDataset(predictions, targets, torch.zeros(1, dtype=torch.long)), batch_size=1
    )

    wrapper.calibrate(loader)

    assert torch.isinf(wrapper.q_hats).all()
    assert not wrapper.class_supported.any()
    empty = torch.zeros((1, 3), dtype=torch.bool)
    assert wrapper._ordinal_hull(empty).tolist() == [[True, True, True]]


def test_predict_step_empty_raw_set_uses_full_fallback() -> None:
    wrapper = make_wrapper()
    wrapper.q_hats.fill_(-2.0)
    wrapper.class_supported.fill_(True)

    output = wrapper.predict_step(
        (torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]), torch.tensor([1])), 0
    )

    assert output["raw_prediction_set"].tolist() == [[False, False, False]]
    assert output["prediction_set"].tolist() == [[True, True, True]]


def test_marginal_mode_preserves_single_correction_and_threshold_boundary() -> None:
    wrapper = make_wrapper(class_wise=False)
    wrapper.q_hat.fill_(0.0)
    batch = (torch.tensor([[0.5, 0.5]]), torch.tensor([0.5]), torch.tensor([1]))

    output = wrapper.predict_step(batch, 0)

    # A point exactly on 0.5 belongs to the right-hand, half-open class bin.
    assert output["prediction_set"].tolist() == [[False, True, False]]
    assert output["target"].tolist() == [1]


def test_integer_targets_preserve_midpoint_thresholds() -> None:
    wrapper = make_wrapper()

    assert wrapper._class_indices(torch.tensor([0, 1, 2])).tolist() == [0, 1, 2]
    assert wrapper._class_indices(torch.tensor([0.5, 1.5])).tolist() == [1, 2]


def test_two_item_class_only_batch_derives_ordinal_label() -> None:
    wrapper = make_wrapper(class_wise=False, allow_derived_labels=True)
    wrapper.q_hat.zero_()

    output = wrapper.predict_step(
        (torch.tensor([[0.9, 1.1]]), torch.tensor([1])), 0
    )

    assert output["target"].tolist() == [1]
    assert output["numeric_target"].tolist() == [1]


def test_inconsistent_numeric_target_and_ordinal_label_raise() -> None:
    wrapper = make_wrapper()
    batch = (
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([2]),
    )

    try:
        wrapper.predict_step(batch, 0)
    except RuntimeError as error:
        assert "target-label-bin consistency" in str(error)
    else:
        raise AssertionError("Expected inconsistent Z and Y_ord to raise.")


def test_nonfinite_quantile_endpoints_raise() -> None:
    wrapper = make_wrapper()
    batch = (
        torch.tensor([[float("nan"), 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([0]),
    )

    try:
        wrapper.predict_step(batch, 0)
    except RuntimeError as error:
        assert "quantile endpoints must be finite" in str(error)
    else:
        raise AssertionError("Expected nonfinite endpoints to raise.")


def test_nonfinite_calibration_target_raise() -> None:
    wrapper = make_wrapper()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([float("inf")]),
            torch.tensor([0]),
        ),
        batch_size=1,
    )

    try:
        wrapper.calibrate(loader)
    except RuntimeError as error:
        assert "Z must be finite" in str(error)
    else:
        raise AssertionError("Expected a nonfinite calibration target to raise.")


def test_uq_metrics_use_segment_count_and_coverage_aware_ccr() -> None:
    metrics = ClassificationUQMetrics(num_classes=3)
    prediction_sets = torch.tensor(
        [[True, True, False], [True, False, True], [True, False, False]]
    )
    targets = torch.tensor([1, 1, 2])

    metrics.update(prediction_sets, targets)
    result = metrics.compute()

    torch.testing.assert_close(result["avg_sfs"], torch.tensor(4.0 / 3.0))
    torch.testing.assert_close(result["avg_mdj"], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(result["ccr"], torch.tensor(1.0 / 3.0))
