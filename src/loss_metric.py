"""
Loss functions and evaluation metrics for BraTS 2021 segmentation.
Includes standard Dice and Dice-CrossEntropy losses, and multi-channel
BraTS evaluation metrics (Tumor Core, Whole Tumor, Enhancing Tumor).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric


def build_loss_fn(loss_type: str = "dice_ce") -> nn.Module:
    """
    Construct multi-label segmentation loss function with sigmoid activation.

    Args:
        loss_type: One of 'dice_ce', 'dice', or 'dice_focal'.

    Returns:
        torch.nn.Module loss instance.
    """
    loss_key = loss_type.lower()
    if loss_key == "dice":
        return DiceLoss(
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    elif loss_key == "dice_focal":
        return DiceFocalLoss(
            sigmoid=True,
            gamma=2.0
        )
    elif loss_key == "dice_ce":
        return DiceCELoss(
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Choose from ['dice', 'dice_ce', 'dice_focal'].")


class BraTSMetric:
    """
    BraTS evaluation metric manager.
    Computes individual Dice scores for:
    - TC (Tumor Core: channel 0)
    - WT (Whole Tumor: channel 1)
    - ET (Enhancing Tumor: channel 2)
    and their arithmetic mean.
    """

    def __init__(self, include_background: bool = True):
        # reduction="none" retains per-channel scores across the batch
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="none"
        )

    def reset(self) -> None:
        """Reset internal metric accumulator."""
        self.dice_metric.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update metric with predictions and ground truth.

        Args:
            y_pred: Predicted logits or probabilities (B, 3, H, W, D).
            y: Ground truth binary targets (B, 3, H, W, D).
        """
        # Threshold predictions at 0.5
        if y_pred.dtype != torch.bool:
            preds = (torch.sigmoid(y_pred) > 0.5).float()
        else:
            preds = y_pred.float()

        self.dice_metric(y_pred=preds, y=y.float())

    def aggregate(self) -> Dict[str, float]:
        """
        Aggregate accumulated scores across all evaluation batches.

        Returns:
            Dictionary with dice_tc, dice_wt, dice_et, and dice_mean.
        """
        # score shape: (N_samples, 3)
        scores = self.dice_metric.aggregate()
        if isinstance(scores, list):
            scores = scores[0]

        channel_means = torch.nanmean(scores, dim=0)

        dice_tc = float(channel_means[0].item()) if scores.shape[-1] > 0 else 0.0
        dice_wt = float(channel_means[1].item()) if scores.shape[-1] > 1 else 0.0
        dice_et = float(channel_means[2].item()) if scores.shape[-1] > 2 else 0.0
        dice_mean = float(torch.nanmean(channel_means).item())

        return {
            "dice_tc": round(dice_tc, 4),
            "dice_wt": round(dice_wt, 4),
            "dice_et": round(dice_et, 4),
            "dice_mean": round(dice_mean, 4)
        }

