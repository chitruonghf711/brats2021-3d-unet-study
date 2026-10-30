"""
Loss functions and evaluation metrics for BraTS 2021 3D segmentation,
including Soft-Label-compatible losses, Noise-Robust losses (SCE, GCE),
and Probability Calibration / Uncertainty metrics (ECE, Brier Score).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric


# Official BraTS 2021 channel ordering produced by MONAI ConvertToMultiChannelBasedOnBratsClassesd:
#   Channel 0: TC (Tumor Core = NCR[1] U ET[4])
#   Channel 1: WT (Whole Tumor = NCR[1] U ED[2] U ET[4])
#   Channel 2: ET (Enhancing Tumor = ET[4])
BRATS_CHANNEL_NAMES: Tuple[str, str, str] = ("tc", "wt", "et")

# Physical diagonal of standard SRI24 BraTS volume (240 x 240 x 155 mm^3 at 1x1x1 mm spacing):
# sqrt(240^2 + 240^2 + 155^2) = 373.1287 mm -> official BraTS Challenge penalty for single-empty HD95.
BRATS_MAX_DISTANCE_MM: float = 373.13


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
        probs = torch.sigmoid(logits.float())
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
        logits_f = logits.float()
        targets_f = targets.float()
        loss_dice = self.soft_dice(logits_f, targets_f)
        loss_bce = F.binary_cross_entropy_with_logits(logits_f, targets_f)
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
    "dice_bce",
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
        loss_type: One of 'dice_ce', 'dice_bce', 'dice', 'dice_focal',
                   'soft_dice', 'soft_dice_bce', 'sce', or 'gce'.
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
    elif loss_key in ("dice_bce", "soft_dice_bce"):
        return SoftDiceBCELoss()
    elif loss_key == "sce":
        return DiceSymmetricBCELoss(alpha=sce_alpha, beta=sce_beta)
    elif loss_key == "gce":
        return DiceGeneralizedCELoss(q=gce_q)
    else:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Choose from {list(SUPPORTED_LOSSES)}."
        )


def compute_hd95_3d(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    empty_penalty: float = BRATS_MAX_DISTANCE_MM,
) -> Tuple[float, str]:
    """
    Compute exact undirected 95% Hausdorff Distance (HD95 in mm) on a single 3D binary channel
    following the RSNA-ASNR-MICCAI BraTS 2021 Challenge convention:
      - Both empty (|P| == 0 and |G| == 0)     -> HD95 = 0.0 mm        ('both_empty')
      - Only pred empty (|P| == 0, |G| > 0)    -> HD95 = empty_penalty ('pred_empty_fn')
      - Only GT empty   (|P| > 0, |G| == 0)    -> HD95 = empty_penalty ('gt_empty_fp')
      - Both non-empty  (|P| > 0 and |G| > 0)  -> max(P95(d(dP->dG)), P95(d(dG->dP))) ('both_non_empty')

    Crops around the union bounding box (P | G) with a 1-voxel margin before running
    Euclidean Distance Transform, yielding bit-exact surface distances ~10x faster.
    """
    p = np.asarray(pred_3d, dtype=bool)
    g = np.asarray(gt_3d, dtype=bool)
    p_any = bool(p.any())
    g_any = bool(g.any())

    if not p_any and not g_any:
        return 0.0, "both_empty"
    if not p_any and g_any:
        return float(empty_penalty), "pred_empty_fn"
    if p_any and not g_any:
        return float(empty_penalty), "gt_empty_fp"

    union = p | g
    coords = np.argwhere(union)
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, p.shape)
    bbox = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))

    p_crop = p[bbox]
    g_crop = g[bbox]

    # Extract 6-connected morphological boundary surfaces (guaranteed non-empty via border_value=0)
    p_surface = p_crop ^ ndi.binary_erosion(p_crop, border_value=0)
    g_surface = g_crop ^ ndi.binary_erosion(g_crop, border_value=0)

    dt_to_g = ndi.distance_transform_edt(~g_surface, sampling=spacing)
    dt_to_p = ndi.distance_transform_edt(~p_surface, sampling=spacing)

    d_p_to_g = dt_to_g[p_surface]
    d_g_to_p = dt_to_p[g_surface]

    hd95 = float(max(np.percentile(d_p_to_g, 95), np.percentile(d_g_to_p, 95)))
    return hd95, "both_non_empty"


def compute_binary_volume_metrics(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    compute_hd95: bool = True,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    empty_penalty: float = BRATS_MAX_DISTANCE_MM,
) -> Dict[str, Any]:
    """
    Compute complete voxel-overlap and boundary metrics on a single 3D region (H, W, D)
    with explicit BraTS empty-region handling:
      - Both empty (|P| == 0 & |G| == 0): Dice = 1.0, IoU = 1.0, Sens = 1.0, Prec = 1.0, HD95 = 0.0
      - Single empty (|P| == 0 ^ |G| == 0): Dice = 0.0, IoU = 0.0, HD95 = empty_penalty
    """
    p = np.asarray(pred_3d, dtype=bool)
    g = np.asarray(gt_3d, dtype=bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    pred_vol = tp + fp
    gt_vol = tp + fn

    if pred_vol == 0 and gt_vol == 0:
        dice = 1.0
        iou = 1.0
        sensitivity = 1.0
        precision = 1.0
        status = "both_empty"
    else:
        dice = float((2.0 * tp) / (2.0 * tp + fp + fn))
        iou = float(tp / (tp + fp + fn))
        sensitivity = float(tp / gt_vol) if gt_vol > 0 else 0.0
        precision = float(tp / pred_vol) if pred_vol > 0 else 0.0
        if pred_vol == 0:
            status = "pred_empty_fn"
        elif gt_vol == 0:
            status = "gt_empty_fp"
        else:
            status = "both_non_empty"

    out: Dict[str, Any] = {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "sensitivity": round(sensitivity, 4),
        "precision": round(precision, 4),
        "pred_voxels": pred_vol,
        "gt_voxels": gt_vol,
        "status": status,
    }
    if compute_hd95:
        hd95_val, _ = compute_hd95_3d(p, g, spacing=spacing, empty_penalty=empty_penalty)
        out["hd95"] = round(hd95_val, 4)
    return out


def get_metric_setup_spec(
    threshold: float = 0.5,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hd95_empty_penalty: float = BRATS_MAX_DISTANCE_MM,
    overlap: float = 0.5,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
) -> Dict[str, Any]:
    """Return a self-documenting dictionary specifying the exact BraTS 2021 metric protocol."""
    return {
        "evaluation_unit": "Full 3D patient volume (never 2D slice-wise or random patch-wise)",
        "inference_mode": {
            "method": "MONAI sliding_window_inference",
            "roi_size": list(roi_size),
            "overlap": overlap,
            "blending": "gaussian",
            "activation": "Channel-wise independent Sigmoid: p_c = sigmoid(logit_c)",
            "binarization": f"Hard threshold p_c > {threshold} per channel",
        },
        "target_subregions": {
            "channel_0_TC": "Tumor Core = NCR (label 1) U ET (label 4)",
            "channel_1_WT": "Whole Tumor = NCR (label 1) U ED (label 2) U ET (label 4)",
            "channel_2_ET": "Enhancing Tumor = ET (label 4)",
        },
        "primary_metrics": {
            "Dice": "2 * |P n G| / (|P| + |G|), range [0, 1], higher is better",
            "HD95_mm": (
                f"max(P95(d(dP -> dG)), P95(d(dG -> dP))) in mm at spacing {list(spacing)}, "
                "lower is better"
            ),
        },
        "secondary_metrics": {
            "IoU_Jaccard": "|P n G| / |P u G|, range [0, 1], higher is better",
            "Sensitivity_Recall": "TP / (TP + FN), range [0, 1], higher is better",
            "Precision_PPV": "TP / (TP + FP), range [0, 1], higher is better",
        },
        "empty_region_convention": {
            "both_empty (|P|==0 and |G|==0)": "Dice=1.0, IoU=1.0, Sensitivity=1.0, Precision=1.0, HD95=0.0 mm",
            "single_empty (|P|==0 xor |G|==0)": (
                f"Dice=0.0, IoU=0.0, HD95={hd95_empty_penalty} mm (SRI24 volume diagonal penalty)"
            ),
        },
        "aggregation_and_selection": {
            "step_1": "Compute volumetric metric per patient n and sub-region c in {TC, WT, ET}",
            "step_2": "Macro-average across patients: Metric_c = (1/N) * sum_n Metric(n, c)",
            "step_3": "Macro-average across 3 regions: Mean_Metric = (Metric_TC + Metric_WT + Metric_ET) / 3",
            "checkpoint_selection_criterion": "Highest Validation Mean Dice (val_dice_mean)",
        },
    }


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
    BraTS 2021 3D Segmentation Evaluation Metric Manager.

    Computes region-wise scores over full 3D volumes for:
      - Channel 0: TC (Tumor Core = 1 + 4)
      - Channel 1: WT (Whole Tumor = 1 + 2 + 4)
      - Channel 2: ET (Enhancing Tumor = 4)
    and their macro mean across regions:
      - Always tracks Dice (TC, WT, ET, Mean) with explicit BraTS empty-mask handling
        (ignore_empty=False: both-empty -> 1.0, single-empty -> 0.0).
      - Optionally tracks 95% Hausdorff Distance HD95 in mm (compute_hd95=True).
      - Optionally tracks IoU, Sensitivity, and Precision (compute_confusion_metrics=True).
      - Optionally tracks probability calibration ECE & Brier Score (track_calibration=True).
    """

    def __init__(
        self,
        include_background: bool = True,
        track_calibration: bool = False,
        compute_hd95: bool = False,
        compute_confusion_metrics: bool = False,
        threshold: float = 0.5,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hd95_empty_penalty: float = BRATS_MAX_DISTANCE_MM,
        ignore_empty: bool = False,
    ):
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="none",
            ignore_empty=ignore_empty,
        )
        self.track_calibration = track_calibration
        self.compute_hd95 = compute_hd95
        self.compute_confusion_metrics = compute_confusion_metrics
        self.threshold = float(threshold)
        self.spacing = spacing
        self.hd95_empty_penalty = float(hd95_empty_penalty)

        self._brier_list: List[float] = []
        self._ece_list: List[float] = []
        self._patient_records: List[Dict[str, Any]] = []
        self._sample_counter: int = 0

    def reset(self) -> None:
        """Reset internal metric accumulators."""
        self.dice_metric.reset()
        self._brier_list = []
        self._ece_list = []
        self._patient_records = []
        self._sample_counter = 0

    def update(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        patient_ids: Optional[Sequence[str]] = None,
        from_logits: bool = True,
    ) -> None:
        """
        Update metric with full-volume predictions and clean ground truth.

        Args:
            y_pred: Predicted logits (default), probabilities, or boolean masks (B, 3, H, W, D).
            y: Ground truth binary targets (B, 3, H, W, D).
            patient_ids: Optional sequence of patient identifiers for case-by-case reporting.
            from_logits: If True (default) and y_pred is non-boolean, applies torch.sigmoid first.
        """
        y_clean = (y.detach().float() > 0.5).float()
        if y_pred.dtype == torch.bool:
            preds = y_pred.detach().float()
        elif from_logits:
            preds = (torch.sigmoid(y_pred.detach().float()) > self.threshold).float()
            if self.track_calibration:
                cal = compute_calibration_stats(y_pred, y_clean)
                self._brier_list.append(cal["brier_score"])
                self._ece_list.append(cal["ece"])
        else:
            preds = (y_pred.detach().float() > self.threshold).float()

        self.dice_metric(y_pred=preds, y=y_clean)

        if self.compute_hd95 or self.compute_confusion_metrics or patient_ids is not None:
            preds_np = preds.cpu().numpy()
            gt_np = y_clean.cpu().numpy()
            batch_size = preds_np.shape[0]
            num_ch = min(preds_np.shape[1], len(BRATS_CHANNEL_NAMES))

            for b in range(batch_size):
                pid = (
                    str(patient_ids[b])
                    if patient_ids is not None and b < len(patient_ids)
                    else f"case_{self._sample_counter:04d}"
                )
                self._sample_counter += 1

                rec: Dict[str, Any] = {"patient_id": pid}
                ch_dices: List[float] = []
                ch_ious: List[float] = []
                ch_sens: List[float] = []
                ch_precs: List[float] = []
                ch_hd95s: List[float] = []

                for c_idx in range(num_ch):
                    ch_name = BRATS_CHANNEL_NAMES[c_idx]
                    m = compute_binary_volume_metrics(
                        pred_3d=preds_np[b, c_idx],
                        gt_3d=gt_np[b, c_idx],
                        compute_hd95=self.compute_hd95,
                        spacing=self.spacing,
                        empty_penalty=self.hd95_empty_penalty,
                    )
                    rec[f"dice_{ch_name}"] = m["dice"]
                    rec[f"iou_{ch_name}"] = m["iou"]
                    rec[f"sensitivity_{ch_name}"] = m["sensitivity"]
                    rec[f"precision_{ch_name}"] = m["precision"]
                    rec[f"pred_voxels_{ch_name}"] = m["pred_voxels"]
                    rec[f"gt_voxels_{ch_name}"] = m["gt_voxels"]
                    rec[f"status_{ch_name}"] = m["status"]

                    ch_dices.append(m["dice"])
                    ch_ious.append(m["iou"])
                    ch_sens.append(m["sensitivity"])
                    ch_precs.append(m["precision"])

                    if self.compute_hd95:
                        rec[f"hd95_{ch_name}"] = m["hd95"]
                        ch_hd95s.append(m["hd95"])

                rec["dice_mean"] = round(float(np.mean(ch_dices)), 4) if ch_dices else 0.0
                rec["iou_mean"] = round(float(np.mean(ch_ious)), 4) if ch_ious else 0.0
                rec["sensitivity_mean"] = round(float(np.mean(ch_sens)), 4) if ch_sens else 0.0
                rec["precision_mean"] = round(float(np.mean(ch_precs)), 4) if ch_precs else 0.0
                if self.compute_hd95 and ch_hd95s:
                    rec["hd95_mean"] = round(float(np.mean(ch_hd95s)), 4)

                self._patient_records.append(rec)

    def get_patient_records(self) -> List[Dict[str, Any]]:
        """Return per-patient evaluation dictionaries accumulated since last reset()."""
        return list(self._patient_records)

    def summarize_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        Compute Mean, Std, Median, Min, and Max across evaluated patients for each tracked metric.
        """
        if not self._patient_records:
            return {}

        metric_keys = [
            "dice_tc", "dice_wt", "dice_et", "dice_mean",
            "iou_tc", "iou_wt", "iou_et", "iou_mean",
            "sensitivity_tc", "sensitivity_wt", "sensitivity_et", "sensitivity_mean",
            "precision_tc", "precision_wt", "precision_et", "precision_mean",
        ]
        if self.compute_hd95:
            metric_keys.extend(["hd95_tc", "hd95_wt", "hd95_et", "hd95_mean"])

        summary: Dict[str, Dict[str, float]] = {}
        for k in metric_keys:
            vals = [float(r[k]) for r in self._patient_records if k in r]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                summary[k] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr, ddof=0)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "min": round(float(np.min(arr)), 4),
                    "max": round(float(np.max(arr)), 4),
                }
        return summary

    def aggregate(self) -> Dict[str, float]:
        """
        Aggregate accumulated scores across all evaluation batches.

        Returns:
            Dictionary with dice_tc, dice_wt, dice_et, dice_mean, plus optional
            hd95_*, iou_*, sensitivity_*, precision_*, brier_score, and ece.
        """
        scores = self.dice_metric.aggregate()
        if isinstance(scores, list):
            scores = scores[0]

        channel_means = torch.nanmean(scores, dim=0)

        dice_tc = float(channel_means[0].item()) if scores.shape[-1] > 0 else 0.0
        dice_wt = float(channel_means[1].item()) if scores.shape[-1] > 1 else 0.0
        dice_et = float(channel_means[2].item()) if scores.shape[-1] > 2 else 0.0
        dice_mean = float(torch.nanmean(channel_means).item())

        result: Dict[str, float] = {
            "dice_tc": round(dice_tc, 4),
            "dice_wt": round(dice_wt, 4),
            "dice_et": round(dice_et, 4),
            "dice_mean": round(dice_mean, 4),
        }

        if self._patient_records:
            if self.compute_hd95:
                for k in ("hd95_tc", "hd95_wt", "hd95_et", "hd95_mean"):
                    vals = [float(r[k]) for r in self._patient_records if k in r]
                    if vals:
                        result[k] = round(float(np.mean(vals)), 4)

            if self.compute_confusion_metrics:
                for prefix in ("iou", "sensitivity", "precision"):
                    for suffix in ("tc", "wt", "et", "mean"):
                        k = f"{prefix}_{suffix}"
                        vals = [float(r[k]) for r in self._patient_records if k in r]
                        if vals:
                            result[k] = round(float(np.mean(vals)), 4)

        if self.track_calibration and self._brier_list:
            result["brier_score"] = round(sum(self._brier_list) / len(self._brier_list), 5)
            result["ece"] = round(sum(self._ece_list) / len(self._ece_list), 5)
        return result
