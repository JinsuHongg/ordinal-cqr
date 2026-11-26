import torch
from torchmetrics import Metric


class PerClassQRValidationMetrics(Metric):
    """Accumulate vectorized per-class quantile-regression diagnostics."""

    def __init__(self, quantiles: list[float], num_classes: int):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32), persistent=False
        )
        self.add_state(
            "class_count",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pinball_sum",
            default=torch.zeros(num_classes, len(quantiles)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "coverage_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "width_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, ordinal_label: torch.Tensor
    ) -> None:
        if preds.ndim != 2 or preds.shape[1] != self.quantiles.numel():
            raise ValueError("preds must have shape [batch, num_quantiles].")
        target = target.view(-1).to(device=preds.device, dtype=preds.dtype)
        ordinal_label = ordinal_label.view(-1).to(device=preds.device, dtype=torch.long)
        if target.shape[0] != preds.shape[0] or ordinal_label.shape[0] != preds.shape[0]:
            raise ValueError("preds, target, and ordinal_label must share a batch size.")
        errors = target.unsqueeze(1) - preds
        quantiles = self.quantiles.to(device=preds.device, dtype=preds.dtype).view(1, -1)
        pinball = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        lower = torch.minimum(preds[:, 0], preds[:, -1])
        upper = torch.maximum(preds[:, 0], preds[:, -1])
        coverage = ((target >= lower) & (target <= upper)).to(dtype=preds.dtype)
        width = upper - lower

        class_indicator = torch.nn.functional.one_hot(
            ordinal_label, num_classes=self.num_classes
        ).to(dtype=preds.dtype)
        self.class_count += class_indicator.sum(dim=0)
        self.pinball_sum += class_indicator.transpose(0, 1) @ pinball
        self.coverage_sum += class_indicator.transpose(0, 1) @ coverage
        self.width_sum += class_indicator.transpose(0, 1) @ width

    def compute(self) -> dict[str, torch.Tensor]:
        denominator = self.class_count.clamp_min(1.0)
        per_class_pinball = self.pinball_sum / denominator.unsqueeze(1)
        per_class_coverage = self.coverage_sum / denominator
        per_class_width = self.width_sum / denominator
        observed = self.class_count > 0
        class_pinball = per_class_pinball.mean(dim=1)
        return {
            "class_count": self.class_count,
            "per_class_pinball": per_class_pinball,
            "per_class_coverage": per_class_coverage,
            "per_class_width": per_class_width,
            "macro_pinball": class_pinball[observed].mean(),
            "worst_class_pinball": class_pinball[observed].max(),
        }
