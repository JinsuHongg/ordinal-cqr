import torch
import torch.nn as nn
from torchmetrics import Metric


class MultiClassClassificationMetrics(Metric):
    """Multi-class classification metrics including Skill Scores.

    Args:
        num_classes: Number of classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "conf_matrix",
            default=torch.zeros(num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix."""
        preds = torch.argmax(preds, dim=1)
        # Assuming target is not one-hot encoded
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=preds.device)
        for p, t in zip(preds, target):
            cm[t, p] += 1
        self.conf_matrix += cm

    def compute(self):
        """Compute all metrics."""
        cm = self.conf_matrix
        tp = cm.diag()
        row_sum = cm.sum(dim=1)
        col_sum = cm.sum(dim=0)
        n = cm.sum()

        # Standard metrics
        accuracy = tp.sum() / n
        
        # Balanced accuracy
        balanced_accuracy = (tp / row_sum).mean()

        # Macro metrics
        precision = (tp / (col_sum + 1e-12)).mean()
        recall = (tp / (row_sum + 1e-12)).mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Skill Scores
        # TSS = (sum(tp) - sum(row_i * col_i)/n) / (n - sum(row_i^2)/n)
        # HSS = (n * sum(tp) - sum(row_i * col_i)) / (n^2 - sum(row_i * col_i))
        
        sum_tp = tp.sum()
        sum_product_marginals = (row_sum * col_sum).sum()
        sum_row_sq = (row_sum**2).sum()
        
        hss = (n * sum_tp - sum_product_marginals) / (n**2 - sum_product_marginals + 1e-12)
        tss = (n * sum_tp - sum_product_marginals) / (n**2 - sum_row_sq + 1e-12)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "tss": tss,
            "hss": hss,
        }


class ClassificationUQMetrics(Metric):
    """Uncertainty Quantification metrics for multi-class classification.

    Args:
        num_classes: Number of classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("coverage", default=[], dist_reduce_fx="cat")
        self.add_state("set_sizes", default=[], dist_reduce_fx="cat")
        self.add_state("sfs", default=[], dist_reduce_fx="cat")
        self.add_state("mdj", default=[], dist_reduce_fx="cat")

    def update(self, prediction_sets: torch.Tensor, target: torch.Tensor):
        """Update metrics state.

        Args:
            prediction_sets: Boolean mask of shape [Batch, Num_Classes].
            target: Ground truth labels of shape [Batch].
        """
        # Marginal Coverage
        # Check if true class is in prediction set
        coverage = torch.gather(prediction_sets, 1, target.unsqueeze(1)).squeeze(1)
        self.coverage.append(coverage.float())

        # Set Size
        set_sizes = prediction_sets.sum(dim=1)
        self.set_sizes.append(set_sizes.float())

        # SFS is the number of connected selected segments. A nonempty
        # contiguous ordinal set therefore has SFS == 1.
        previous_selected = torch.cat(
            (
                torch.zeros_like(prediction_sets[:, :1]),
                prediction_sets[:, :-1],
            ),
            dim=1,
        )
        segment_starts = prediction_sets & ~previous_selected
        batch_sfs = segment_starts.sum(dim=1).float()

        # MDJ is the largest number of omitted labels between consecutive
        # selected classes. Compute it without per-sample Python loops.
        positions = torch.arange(
            self.num_classes, device=prediction_sets.device
        ).expand_as(prediction_sets)
        selected_positions = torch.where(
            prediction_sets, positions, torch.full_like(positions, -1)
        )
        last_selected = selected_positions.cummax(dim=1).values
        previous_index = torch.cat(
            (torch.full_like(last_selected[:, :1], -1), last_selected[:, :-1]),
            dim=1,
        )
        gaps = torch.where(
            prediction_sets & (previous_index >= 0),
            positions - previous_index - 1,
            torch.zeros_like(positions),
        )
        batch_mdj = gaps.max(dim=1).values.float()

        self.sfs.append(batch_sfs)
        self.mdj.append(batch_mdj)

    def compute(self):
        """Compute all UQ metrics."""
        coverage = torch.cat(self.coverage)
        set_sizes = torch.cat(self.set_sizes)
        sfs = torch.cat(self.sfs)
        mdj = torch.cat(self.mdj)

        return {
            "marginal_coverage": coverage.mean(),
            "avg_set_size": set_sizes.mean(),
            "avg_sfs": sfs.mean(),
            "avg_mdj": mdj.mean(),
            "ccr": ((sfs == 1) & coverage.bool()).float().mean(),
        }


class OrdinalCQRMetrics(Metric):
    """Aggregate raw, fallback, hull, and final OCQR evaluation metrics."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        scalar_states = (
            "num_samples",
            "raw_coverage_sum",
            "final_coverage_sum",
            "raw_size_sum",
            "final_size_sum",
            "raw_empty_count",
            "raw_fragmented_count",
            "hull_inflation_sum",
            "fallback_inflation_sum",
            "total_inflation_sum",
            "full_set_count",
            "final_sfs_sum",
            "final_mdj_sum",
            "ccr_count",
        )
        for name in scalar_states:
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "per_class_count",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "per_class_coverage_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "per_class_set_size_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )

    def _segment_count(self, prediction_sets: torch.Tensor) -> torch.Tensor:
        previous = torch.cat(
            (torch.zeros_like(prediction_sets[:, :1]), prediction_sets[:, :-1]),
            dim=1,
        )
        return (prediction_sets & ~previous).sum(dim=1)

    def _maximum_disjoint_jump(self, prediction_sets: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            self.num_classes, device=prediction_sets.device
        ).expand_as(prediction_sets)
        selected = torch.where(
            prediction_sets, positions, torch.full_like(positions, -1)
        )
        last_selected = selected.cummax(dim=1).values
        previous = torch.cat(
            (torch.full_like(last_selected[:, :1], -1), last_selected[:, :-1]),
            dim=1,
        )
        gaps = torch.where(
            prediction_sets & (previous >= 0),
            positions - previous - 1,
            torch.zeros_like(positions),
        )
        return gaps.max(dim=1).values

    def update(
        self,
        raw_prediction_sets: torch.Tensor,
        final_prediction_sets: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Accumulate metrics from raw and post-processed prediction sets."""
        if raw_prediction_sets.shape != final_prediction_sets.shape:
            raise ValueError("Raw and final prediction sets must have matching shapes.")
        if raw_prediction_sets.ndim != 2 or raw_prediction_sets.shape[1] != self.num_classes:
            raise ValueError("Prediction sets must have shape [batch, num_classes].")

        raw = raw_prediction_sets.bool()
        final = final_prediction_sets.bool()
        target = target.view(-1).long()
        if target.shape[0] != raw.shape[0]:
            raise ValueError("Targets and prediction sets must share the batch size.")
        torch._assert_async(
            ((target >= 0) & (target < self.num_classes)).all(),
            "OCQR targets must be in [0, num_classes).",
        )

        raw_active = raw.any(dim=1)
        fallback = torch.where(raw_active[:, None], raw, torch.ones_like(raw))
        raw_size = raw.sum(dim=1).float()
        fallback_size = fallback.sum(dim=1).float()
        final_size = final.sum(dim=1).float()
        raw_coverage = torch.gather(raw, 1, target[:, None]).squeeze(1)
        final_coverage = torch.gather(final, 1, target[:, None]).squeeze(1)
        raw_sfs = self._segment_count(raw)
        final_sfs = self._segment_count(final)
        final_mdj = self._maximum_disjoint_jump(final)
        torch._assert_async(
            ((~raw) | final).all(), "Final OCQR sets must contain their raw sets."
        )
        torch._assert_async(
            ((~fallback) | final).all(),
            "Final OCQR sets must contain the fallback-adjusted sets.",
        )
        torch._assert_async(final.any(dim=1).all(), "Final OCQR sets must be nonempty.")
        torch._assert_async(
            (final_sfs == 1).all(), "Final OCQR sets must be ordinally contiguous."
        )

        self.num_samples += raw.shape[0]
        self.raw_coverage_sum += raw_coverage.float().sum()
        self.final_coverage_sum += final_coverage.float().sum()
        self.raw_size_sum += raw_size.sum()
        self.final_size_sum += final_size.sum()
        self.raw_empty_count += (~raw_active).float().sum()
        self.raw_fragmented_count += (raw_sfs > 1).float().sum()
        self.hull_inflation_sum += (final_size - fallback_size).sum()
        self.fallback_inflation_sum += (fallback_size - raw_size).sum()
        self.total_inflation_sum += (final_size - raw_size).sum()
        self.full_set_count += (final_size == self.num_classes).float().sum()
        self.final_sfs_sum += final_sfs.float().sum()
        self.final_mdj_sum += final_mdj.float().sum()
        self.ccr_count += ((final_sfs == 1) & final_coverage).float().sum()

        ones = torch.ones_like(target, dtype=torch.float32)
        self.per_class_count.scatter_add_(0, target, ones)
        self.per_class_coverage_sum.scatter_add_(
            0, target, final_coverage.float()
        )
        self.per_class_set_size_sum.scatter_add_(0, target, final_size)

    def compute(self) -> dict[str, torch.Tensor]:
        """Return aggregate and per-class OCQR metrics."""
        denominator = self.num_samples.clamp_min(1.0)
        class_denominator = self.per_class_count.clamp_min(1.0)
        nan_values = torch.full_like(self.per_class_count, torch.nan)
        per_class_coverage = torch.where(
            self.per_class_count > 0,
            self.per_class_coverage_sum / class_denominator,
            nan_values,
        )
        per_class_set_size = torch.where(
            self.per_class_count > 0,
            self.per_class_set_size_sum / class_denominator,
            nan_values,
        )
        return {
            "num_samples": self.num_samples,
            "raw_coverage": self.raw_coverage_sum / denominator,
            "marginal_coverage": self.final_coverage_sum / denominator,
            "avg_raw_set_size": self.raw_size_sum / denominator,
            "avg_set_size": self.final_size_sum / denominator,
            "raw_empty_rate": self.raw_empty_count / denominator,
            "raw_fragmented_rate": self.raw_fragmented_count / denominator,
            "avg_hull_inflation": self.hull_inflation_sum / denominator,
            "avg_fallback_inflation": self.fallback_inflation_sum / denominator,
            "avg_total_inflation": self.total_inflation_sum / denominator,
            "normalized_hull_inflation": (
                self.hull_inflation_sum / denominator / self.num_classes
            ),
            "normalized_fallback_inflation": (
                self.fallback_inflation_sum / denominator / self.num_classes
            ),
            "normalized_total_inflation": (
                self.total_inflation_sum / denominator / self.num_classes
            ),
            "full_set_rate": self.full_set_count / denominator,
            "avg_sfs": self.final_sfs_sum / denominator,
            "avg_mdj": self.final_mdj_sum / denominator,
            "ccr": self.ccr_count / denominator,
            "per_class_count": self.per_class_count,
            "per_class_coverage": per_class_coverage,
            "per_class_avg_set_size": per_class_set_size,
        }
