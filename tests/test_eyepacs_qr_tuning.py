from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torchvision import models

from ordinal_cqr.datamodules.eyepacs import build_eyepacs_sample_weights
from ordinal_cqr.metrics.regression_metrics import PerClassQRValidationMetrics
from ordinal_cqr.models.backbone import ResNet18Regressor
from ordinal_cqr.utils.losses import PinballLoss


def test_pinball_loss_none_matches_mean_reduction() -> None:
    predictions = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([1.0, 1.0])
    elementwise = PinballLoss([0.05, 0.5, 0.95], reduction="none")(
        predictions, target
    )

    assert elementwise.shape == (2, 3)
    torch.testing.assert_close(
        elementwise.mean(), PinballLoss([0.05, 0.5, 0.95])(predictions, target)
    )


def test_eyepacs_sampling_weights_cover_declared_strategies() -> None:
    labels = [0, 0, 0, 0, 1, 2, 3, 4]

    assert build_eyepacs_sample_weights(labels, "natural") is None
    torch.testing.assert_close(
        build_eyepacs_sample_weights(labels, "inverse_frequency"),
        torch.tensor(
            [0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        build_eyepacs_sample_weights(labels, "sqrt_inverse_frequency"),
        torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
    )


def test_per_class_qr_metrics_use_ordinal_label_groups() -> None:
    metric = PerClassQRValidationMetrics([0.05, 0.5, 0.95], num_classes=2)
    predictions = torch.tensor([[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]])
    target = torch.tensor([0.5, 1.0])
    ordinal_label = torch.tensor([1, 0])

    metric.update(predictions, target, ordinal_label)
    result = metric.compute()

    torch.testing.assert_close(result["class_count"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(result["per_class_coverage"], torch.ones(2))
    torch.testing.assert_close(result["per_class_width"], torch.tensor([2.0, 1.0]))


def test_pretrained_resnet18_keeps_the_imagenet_input_layer() -> None:
    constructor = models.resnet18
    captured = {}

    def build_without_download(*, weights):
        captured["weights"] = weights
        model = constructor(weights=None)
        captured["conv1"] = model.conv1
        return model

    with patch(
        "ordinal_cqr.models.backbone.models.resnet18", side_effect=build_without_download
    ):
        model = ResNet18Regressor(
            in_channels=3, time_steps=1, num_classes=3, weights="DEFAULT"
        )

    assert captured["weights"] is models.ResNet18_Weights.DEFAULT
    assert model.resnet[0] is captured["conv1"]


def test_pretrained_resnet18_rejects_incompatible_channels() -> None:
    with pytest.raises(ValueError, match=r"in_channels \* time_steps == 3"):
        ResNet18Regressor(in_channels=1, time_steps=1, weights="DEFAULT")


def test_eyepacs_sweep_config_declares_tuning_controls() -> None:
    cfg = OmegaConf.load("configs/qr/QR_resnet18_train_eyepacs_sweep.yaml")

    assert cfg.data.sampling_strategy == "natural"
    assert cfg.model.resnet18.weights == "DEFAULT"
    assert cfg.model.resnet18.p_drop == 0.2
    assert cfg.optimizer.lr == pytest.approx(1e-4)
    assert cfg.optimizer.weight_decay == pytest.approx(0.01)
    assert cfg.scheduler.cosine_warmup.warmup_ratio == pytest.approx(0.02)
    assert cfg.model.qr.validation_diagnostics.num_classes == 5
