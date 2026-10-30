"""
Utility functions for BraTS 2021 segmentation pipeline:
- Reproducibility & seed configuration
- Model checkpointing (saving & loading)
- Qualitative segmentation visualization (predictions vs ground truth)
- Training history plotting
"""

import os
import random
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return underlying module if wrapped in torch.nn.DataParallel / DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "best_metric_model.pth"
) -> str:
    """
    Save training state dictionary to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model weights and optional optimizer state from checkpoint (supports DataParallel).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    # Strip 'module.' prefix if ever saved directly from DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def visualize_predictions(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize input modalities, ground truth masks, and model predictions.

    Args:
        image: Input modalities array of shape (4, H, W, D) or (H, W, D, 4).
        label: Ground truth target array of shape (3, H, W, D).
        pred: Predicted binary mask array of shape (3, H, W, D).
        slice_idx: Optional axial slice index.
        save_path: Optional output path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    # Ensure channels-first format: (C, H, W, D)
    if image.shape[0] != 4 and image.shape[-1] == 4:
        image = np.moveaxis(image, -1, 0)
    if label.shape[0] != 3 and label.shape[-1] == 3:
        label = np.moveaxis(label, -1, 0)
    if pred.shape[0] != 3 and pred.shape[-1] == 3:
        pred = np.moveaxis(pred, -1, 0)

    # If no slice specified, find slice with maximum ground truth tumor area
    if slice_idx is None:
        wt_mask = label[1]  # Whole Tumor channel
        slice_sums = np.sum(wt_mask, axis=(0, 1))
        slice_idx = int(np.argmax(slice_sums)) if np.max(slice_sums) > 0 else label.shape[-1] // 2

    flair_slice = image[0, :, :, slice_idx]
    t1ce_slice = image[1, :, :, slice_idx]

    # Target channels: 0 = TC, 1 = WT, 2 = ET
    gt_tc = label[0, :, :, slice_idx]
    gt_wt = label[1, :, :, slice_idx]
    gt_et = label[2, :, :, slice_idx]

    pred_tc = pred[0, :, :, slice_idx]
    pred_wt = pred[1, :, :, slice_idx]
    pred_et = pred[2, :, :, slice_idx]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"BraTS Segmentation: Ground Truth vs Model Prediction (Axial Slice: {slice_idx})",
                 fontsize=14, fontweight="bold")

    # Row 1: Ground Truth
    axes[0, 0].imshow(np.rot90(flair_slice), cmap="gray")
    axes[0, 0].set_title("Input: FLAIR", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.rot90(gt_wt), cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("GT: Whole Tumor (WT)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.rot90(gt_tc), cmap="Reds", vmin=0, vmax=1)
    axes[0, 2].set_title("GT: Tumor Core (TC)", fontsize=11)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(np.rot90(gt_et), cmap="YlOrBr", vmin=0, vmax=1)
    axes[0, 3].set_title("GT: Enhancing Tumor (ET)", fontsize=11)
    axes[0, 3].axis("off")

    # Row 2: Predictions
    axes[1, 0].imshow(np.rot90(t1ce_slice), cmap="gray")
    axes[1, 0].set_title("Input: T1ce", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.rot90(pred_wt), cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title("Pred: Whole Tumor (WT)", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.rot90(pred_tc), cmap="Reds", vmin=0, vmax=1)
    axes[1, 2].set_title("Pred: Tumor Core (TC)", fontsize=11)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(np.rot90(pred_et), cmap="YlOrBr", vmin=0, vmax=1)
    axes[1, 3].set_title("Pred: Enhancing Tumor (ET)", fontsize=11)
    axes[1, 3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss and validation Dice scores across epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if "train_loss" in history and len(history["train_loss"]) > 0:
        epochs = list(range(1, len(history["train_loss"]) + 1))
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
        axes[0].set_title("Training Loss Over Epochs", fontsize=12)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

    # Dice metrics curve
    val_epochs = history.get("val_epochs", [])
    if len(val_epochs) > 0:
        if "val_dice_tc" in history:
            axes[1].plot(val_epochs, history["val_dice_tc"], label="Val Dice TC", color="crimson", lw=1.5, marker="o")
        if "val_dice_wt" in history:
            axes[1].plot(val_epochs, history["val_dice_wt"], label="Val Dice WT", color="mediumseagreen", lw=1.5, marker="s")
        if "val_dice_et" in history:
            axes[1].plot(val_epochs, history["val_dice_et"], label="Val Dice ET", color="darkorange", lw=1.5, marker="^")
        if "val_dice_mean" in history:
            axes[1].plot(val_epochs, history["val_dice_mean"], label="Val Dice Mean", color="purple", lw=2.5, linestyle="--")

        axes[1].set_title("Validation Dice Metric Over Epochs", fontsize=12)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Dice Score", fontsize=11)
        axes[1].set_ylim([0.0, 1.0])
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    return fig

