import torch

from ordinal_cqr.metrics.classification_metrics import OrdinalCQRMetrics


def test_raw_fallback_hull_metrics_have_exact_decomposition() -> None:
    metrics = OrdinalCQRMetrics(num_classes=3)
    raw = torch.tensor(
        [[True, True, False], [True, False, True], [False, False, False]]
    )
    final = torch.tensor(
        [[True, True, False], [True, True, True], [True, True, True]]
    )
    target = torch.tensor([1, 1, 2])

    metrics.update(raw, final, target)
    result = metrics.compute()

    torch.testing.assert_close(result["num_samples"], torch.tensor(3.0))
    torch.testing.assert_close(result["raw_coverage"], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(result["marginal_coverage"], torch.tensor(1.0))
    torch.testing.assert_close(result["avg_raw_set_size"], torch.tensor(4.0 / 3.0))
    torch.testing.assert_close(result["avg_set_size"], torch.tensor(8.0 / 3.0))
    torch.testing.assert_close(result["raw_empty_rate"], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(
        result["raw_fragmented_rate"], torch.tensor(1.0 / 3.0)
    )
    torch.testing.assert_close(
        result["avg_fallback_inflation"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        result["avg_hull_inflation"], torch.tensor(1.0 / 3.0)
    )
    torch.testing.assert_close(
        result["avg_total_inflation"], torch.tensor(4.0 / 3.0)
    )
    torch.testing.assert_close(result["full_set_rate"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(result["ccr"], torch.tensor(1.0))
    assert result["per_class_count"].tolist() == [0.0, 2.0, 1.0]
    assert torch.isnan(result["per_class_coverage"][0])
    torch.testing.assert_close(
        result["per_class_avg_set_size"][1:], torch.tensor([2.5, 3.0])
    )


def test_partitioned_updates_equal_single_update() -> None:
    raw = torch.tensor([[True, False, True], [False, False, False]])
    final = torch.ones((2, 3), dtype=torch.bool)
    target = torch.tensor([0, 2])
    combined = OrdinalCQRMetrics(num_classes=3)
    partitioned = OrdinalCQRMetrics(num_classes=3)

    combined.update(raw, final, target)
    partitioned.update(raw[:1], final[:1], target[:1])
    partitioned.update(raw[1:], final[1:], target[1:])

    combined_result = combined.compute()
    partitioned_result = partitioned.compute()
    for name in combined_result:
        torch.testing.assert_close(
            combined_result[name], partitioned_result[name], equal_nan=True
        )
