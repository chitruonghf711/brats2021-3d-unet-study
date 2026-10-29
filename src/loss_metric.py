"""
Loss functions and evaluation metrics for BraTS 2021 3D segmentation,
including Soft-Label-compatible losses, Noise-Robust losses (SCE, GCE),
and Probability Calibration / Uncertainty metrics (ECE, Brier Score).
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric


class SoftDiceLoss(nn.Module):
    """
    Continuous L2-denominator Soft Dice Loss for multi-label 3D segmentation.
    Reaches its exact minimum when sigmoid(logits) == soft_target for any y in [0, 1],
    preserving probability calibration on soft boundary labels.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        # Flatten spatial dimensions: (B, C, N)
        probs_flat = probs.flatten(start_dim=2)
        targets_flat = targets.flatten(start_dim=2)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        denom = (probs_flat.pow(2) + targets_flat.pow(2)).sum(dim=2)
        soft_dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - soft_dice.mean()


class SoftDiceBCELoss(nn.Module):
    """
    Combined Soft Dice + Channel-wise Soft Binary Cross-Entropy Loss.
    Properly handles overlapping BraTS channels (TC, WT, ET) and continuous targets in [0, 1]
    without cross-channel Softmax competition.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.soft_dice = SoftDiceLoss(smooth=smooth)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        loss_dice = self.soft_dice(logits, targets_f)
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets_f)
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce


class DiceSymmetricBCELoss(nn.Module):
    """
    Noise-Robust Symmetric Cross-Entropy (Wang et al., ICCV 2019) + Soft Dice Loss
    adapted for 3D multi-label segmentation under boundary and flip label noise.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        dice_weight: float = 0.5,
        eps: float = 1e-4,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_weight = dice_weight
        self.eps = eps
        self.soft_dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f)

        probs = torch.clamp(torch.sigmoid(logits), self.eps, 1.0 - self.eps)
        t_clip = torch.clamp(targets_f, self.eps, 1.0 - self.eps)
        rce = -(probs * torch.log(t_clip) + (1.0 - probs) * torch.log(1.0 - t_clip)).mean()

        sce = self.alpha * bce + self.beta * rce
        loss_dice = self.soft_dice(logits, targets_f)
        return self.dice_weight * loss_dice + (1.0 - self.dice_weight) * sce


class DiceGeneralizedCELoss(nn.Module):
    """
    Noise-Robust Generalized Cross-Entropy (Zhang & Sabuncu, NeurIPS 2018) + Soft Dice Loss.
    Parameter q in (0, 1] interpolates between BCE (q -> 0) and MAE (q = 1), attenuating
    gradients on corrupted/mislabeled voxels to prevent noise memorization.
    """

    def __init__(self, q: float = 0.7, dice_weight: float = 0.5, eps: float = 1e-6, smooth: float = 1e-5):
        super().__init__()
        if not (0.0 < q <= 1.0):
            raise ValueError(f"GCE parameter q must be in (0, 1], got {q}.")
        self.q = q
        self.dice_weight = dice_weight
        self.eps = eps
        self.soft_dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        probs = torch.sigmoid(logits)
        p_t = targets_f * probs + (1.0 - targets_f) * (1.0 - probs)
        gce = ((1.0 - torch.clamp(p_t, min=self.eps).pow(self.q)) / self.q).mean()
        loss_dice = self.soft_dice(logits, targets_f)
        return self.dice_weight * loss_dice + (1.0 - self.dice_weight) * gce


SUPPORTED_LOSSES = (
    "dice",
    "dice_ce",
    "dice_focal",
    "soft_dice",
    "soft_dice_bce",
    "sce",
    "gce",
)


def build_loss_fn(
    loss_type: str = "dice_ce",
    sce_alpha: float = 1.0,
    sce_beta: float = 0.5,
    gce_q: float = 0.7,
) -> nn.Module:
    """
    Construct multi-label segmentation loss function with sigmoid activation.

    Args:
        loss_type: One of 'dice_ce', 'dice', 'dice_focal', 'soft_dice',
                   'soft_dice_bce', 'sce', or 'gce'.
        sce_alpha: Forward BCE weight for Symmetric Cross-Entropy ('sce').
        sce_beta: Reverse BCE weight for Symmetric Cross-Entropy ('sce').
        gce_q: Box-Cox robustness exponent q in (0, 1] for Generalized CE ('gce').

    Returns:
        torch.nn.Module loss instance.
    """
    loss_key = loss_type.lower()
    if loss_key == "dice":
        return DiceLoss(
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
    elif loss_key == "dice_focal":
        return DiceFocalLoss(
            sigmoid=True,
            gamma=2.0,
        )
    elif loss_key == "dice_ce":
        return DiceCELoss(
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
    elif loss_key == "soft_dice":
        return SoftDiceLoss()
    elif loss_key == "soft_dice_bce":
        return SoftDiceBCELoss()
    elif loss_key == "sce":
        return DiceSymmetricBCELoss(alpha=sce_alpha, beta=sce_beta)
    elif loss_key == "gce":
        return DiceGeneralizedCELoss(q=gce_q)
    else:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Choose from {list(SUPPORTED_LOSSES)}."
        )


def compute_calibration_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute probability calibration and uncertainty metrics on 3D predictions:
    - brier_score: Mean squared probability error E[(p - y)^2] (lower is better).
    - ece: Expected Calibration Error across confidence bins (lower is better).
    - predictive_entropy: Mean binary entropy in nats across active/near-tumor region.
    """
    probs = torch.sigmoid(logits.detach().float()).reshape(-1)
    y = (targets.detach().float() > 0.5).float().reshape(-1)

    brier = float(((probs - y) ** 2).mean().item())

    # Confidence & accuracy formulation for binary multi-label ECE
    conf = torch.maximum(probs, 1.0 - probs)
    pred_bin = (probs >= 0.5).float()
    acc = (pred_bin == y).float()

    bin_boundaries = torch.linspace(0.5, 1.0, steps=num_bins + 1, device=probs.device)
    ece = 0.0
    n_total = float(probs.numel())
    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (conf >= low) & (conf <= high)
        else:
            in_bin = (conf > low) & (conf <= high)
        count = float(in_bin.sum().item())
        if count > 0:
            avg_conf = float(conf[in_bin].mean().item())
            avg_acc = float(acc[in_bin].mean().item())
            ece += (count / n_total) * abs(avg_acc - avg_conf)

    p_clamped = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    entropy = -(p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))

    return {
        "brier_score": round(brier, 5),
        "ece": round(float(ece), 5),
        "mean_entropy": round(float(entropy.mean().item()), 5),
    }


class BraTSMetric:
    """
    BraTS evaluation metric manager.
    Computes individual Dice scores for:
    - TC (Tumor Core: channel 0)
    - WT (Whole Tumor: channel 1)
    - ET (Enhancing Tumor: channel 2)
    and their arithmetic mean, plus optional probability calibration tracking.
    """

    def __init__(self, include_background: bool = True, track_calibration: bool = False):
        # reduction="none" retains per-channel scores across the batch
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="none",
        )
        self.track_calibration = track_calibration
        self._brier_list = []
        self._ece_list = []

    def reset(self) -> None:
        """Reset internal metric accumulator."""
        self.dice_metric.reset()
        self._brier_list = []
        self._ece_list = []

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update metric with predictions and ground truth.

        Args:
            y_pred: Predicted logits or boolean masks (B, 3, H, W, D).
            y: Ground truth binary targets (B, 3, H, W, D).
        """
        y_clean = (y.float() > 0.5).float()
        if y_pred.dtype != torch.bool:
            preds = (torch.sigmoid(y_pred) > 0.5).float()
            if self.track_calibration:
                cal = compute_calibration_stats(y_pred, y_clean)
                self._brier_list.append(cal["brier_score"])
                self._ece_list.append(cal["ece"])
        else:
            preds = y_pred.float()

        self.dice_metric(y_pred=preds, y=y_clean)

    def aggregate(self) -> Dict[str, float]:
        """
        Aggregate accumulated scores across all evaluation batches.

        Returns:
            Dictionary with dice_tc, dice_wt, dice_et, and dice_mean
            (plus brier_score and ece when track_calibration=True).
        """
        scores = self.dice_metric.aggregate()
        if isinstance(scores, list):
            scores = scores[0]

        channel_means = torch.nanmean(scores, dim=0)

        dice_tc = float(channel_means[0].item()) if scores.shape[-1] > 0 else 0.0
        dice_wt = float(channel_means[1].item()) if scores.shape[-1] > 1 else 0.0
        dice_et = float(channel_means[2].item()) if scores.shape[-1] > 2 else 0.0
        dice_mean = float(torch.nanmean(channel_means).item())

        result = {
            "dice_tc": round(dice_tc, 4),
            "dice_wt": round(dice_wt, 4),
            "dice_et": round(dice_et, 4),
            "dice_mean": round(dice_mean, 4),
        }
        if self.track_calibration and self._brier_list:
            result["brier_score"] = round(sum(self._brier_list) / len(self._brier_list), 5)
            result["ece"] = round(sum(self._ece_list) / len(self._ece_list), 5)
        return result
