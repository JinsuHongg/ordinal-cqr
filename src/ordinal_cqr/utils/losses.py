import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression.

    This loss function is used to train models to predict specific quantiles
     of the target distribution.

    Args:
        quantiles: List of quantiles to estimate (e.g., [0.05, 0.5, 0.95]).

    Attributes:
        quantiles: List of quantiles to estimate.
    """

    def __init__(self, quantiles: list[float], reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be either 'mean' or 'none'.")
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, preds, target):
        """Calculates the pinball loss.

        Args:
            preds: Predicted quantiles of shape (Batch, Num_Quantiles).
            target: Ground truth values of shape (Batch).

        Returns:
            The calculated mean pinball loss.
        """
        # Ensure target shape matches preds for broadcasting
        # Target: [Batch] -> [Batch, 1]
        target = target.view(-1, 1)

        errors = target - preds
        quantiles = preds.new_tensor(self.quantiles).view(1, -1)
        losses = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        return losses.mean() if self.reduction == "mean" else losses
