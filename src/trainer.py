"""
Training and validation engine for BraTS 2021 3D segmentation.
Features:
- Automatic Mixed Precision (AMP) support for fast training and low VRAM
- Sliding window inference for memory-efficient 3D volume validation
- Overlap blending using Gaussian weighting
- Best checkpoint tracking and recovery
- Clean progress tracking without icons
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from src.config import BaselineConfig
from src.loss_metric import BraTSMetric
from src.utils import save_checkpoint, unwrap_model


def _to_plain_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert MONAI MetaTensor to sanitized native torch.Tensor on target device (safe for DataParallel)."""
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    t = x.to(device)
    if t.is_floating_point():
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


class BraTSTrainer:
    """
    Trainer class encapsulating training, validation, and checkpointing logic.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: BaselineConfig,
        lr_scheduler: Optional[Any] = None,
        metric: Optional[BraTSMetric] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.metric = (
            metric
            if metric is not None
            else BraTSMetric(
                track_calibration=getattr(config, "track_calibration", False),
                compute_hd95=getattr(config, "compute_hd95", False),
                compute_confusion_metrics=getattr(config, "compute_confusion_metrics", False),
                threshold=getattr(config, "metric_threshold", 0.5),
                spacing=getattr(config, "voxel_spacing", (1.0, 1.0, 1.0)),
                hd95_empty_penalty=getattr(config, "hd95_empty_penalty", 373.13),
            )
        )

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Cross-platform AMP configuration:
        # CUDA uses GradScaler; MPS and CPU run standard precision or autocast without scaler
        self.is_cuda = self.device.type == "cuda"
        self.amp_enabled = config.amp and self.is_cuda
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled) if self.is_cuda else None
        self.autocast_device = "cuda" if self.is_cuda else "cpu"

        self.best_metric = -1.0
        self.best_metric_epoch = -1
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_epochs": [],
            "val_dice_tc": [],
            "val_dice_wt": [],
            "val_dice_et": [],
            "val_dice_mean": [],
            "val_hd95_tc": [],
            "val_hd95_wt": [],
            "val_hd95_et": [],
            "val_hd95_mean": [],
            "val_brier": [],
            "val_ece": [],
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Execute one training epoch.

        Args:
            dataloader: DataLoader supplying training patches.
            epoch: Current epoch number (1-indexed).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        step_count = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{self.config.max_epochs} [Train]",
            leave=False,
            disable=not sys.stdout.isatty(),
        )

        for batch in pbar:
            images = _to_plain_tensor(batch["image"], self.device)
            labels = _to_plain_tensor(batch["label"], self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.autocast_device,
                                    enabled=self.amp_enabled):
                logits = self.model(images)

            # Always compute loss in FP32 to prevent FP16 sum/logit overflow
            loss = self.loss_fn(logits.float(), labels.float())
            if not torch.isfinite(loss):
                continue

            if self.scaler is not None and self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            step_count += 1
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = epoch_loss / max(step_count, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model using sliding window inference over complete 3D volumes.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            Dictionary containing Dice scores (TC, WT, ET, mean).
        """
        self.model.eval()
        self.metric.reset()

        pbar = tqdm(dataloader, desc="Validating (Sliding Window)", leave=False, disable=not sys.stdout.isatty())

        for batch in pbar:
            images = _to_plain_tensor(batch["image"], self.device)
            labels = _to_plain_tensor(batch["label"], self.device)

            with torch.amp.autocast(device_type=self.autocast_device,
                                    enabled=self.amp_enabled):
                preds = sliding_window_inference(
                    inputs=images,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.overlap,
                    mode="gaussian"
                )

            self.metric.update(
                y_pred=preds,
                y=labels,
                patient_ids=batch.get("patient_id"),
            )

        scores = self.metric.aggregate()
        return scores

    def _run_validation_and_checkpoint(self, val_loader: DataLoader, epoch: int) -> None:
        """Run sliding-window validation, update history, and save best_metric_model.pth if improved."""
        val_scores = self.validate(val_loader)
        mean_dice = val_scores["dice_mean"]

        self.history["val_epochs"].append(epoch)
        self.history["val_dice_tc"].append(val_scores["dice_tc"])
        self.history["val_dice_wt"].append(val_scores["dice_wt"])
        self.history["val_dice_et"].append(val_scores["dice_et"])
        self.history["val_dice_mean"].append(mean_dice)
        if "hd95_mean" in val_scores:
            self.history["val_hd95_tc"].append(val_scores["hd95_tc"])
            self.history["val_hd95_wt"].append(val_scores["hd95_wt"])
            self.history["val_hd95_et"].append(val_scores["hd95_et"])
            self.history["val_hd95_mean"].append(val_scores["hd95_mean"])
        if "brier_score" in val_scores:
            self.history["val_brier"].append(val_scores["brier_score"])
        if "ece" in val_scores:
            self.history["val_ece"].append(val_scores["ece"])

        extra_str = ""
        if "hd95_mean" in val_scores:
            extra_str += (
                f" | HD95(mm) Mean: {val_scores['hd95_mean']:.2f} "
                f"(TC:{val_scores['hd95_tc']:.2f}, WT:{val_scores['hd95_wt']:.2f}, ET:{val_scores['hd95_et']:.2f})"
            )
        if "ece" in val_scores:
            extra_str += f" | ECE: {val_scores['ece']:.4f} | Brier: {val_scores['brier_score']:.4f}"

        print(
            f"Validation (Epoch {epoch:03d}) - "
            f"Mean Dice: {mean_dice:.4f} | "
            f"TC: {val_scores['dice_tc']:.4f} | "
            f"WT: {val_scores['dice_wt']:.4f} | "
            f"ET: {val_scores['dice_et']:.4f}{extra_str}"
        )

        if mean_dice > self.best_metric:
            self.best_metric = mean_dice
            self.best_metric_epoch = epoch
            ckpt_path = save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": self.best_metric,
                    "val_scores": val_scores,
                    "config": self.config.__dict__
                },
                checkpoint_dir=self.config.checkpoint_dir,
                filename="best_metric_model.pth"
            )
            print(f"Saved new best model to: {ckpt_path} (Mean Dice: {mean_dice:.4f})")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        val_interval: Optional[int] = None,
        max_wall_hours: Optional[float] = None,
        start_epoch: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Run complete training and validation cycle with optional wall-clock session guardrail.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Number of training epochs (defaults to config.max_epochs).
            val_interval: Validation frequency in epochs (defaults to config.val_interval).
            max_wall_hours: Optional hard wall-clock budget in hours (e.g. 10.5 on Kaggle 12h sessions).
                            Stops cleanly before timeout so Held-Out Test & artifacts always finish saving.
            start_epoch: 1-indexed starting epoch (used when resuming from a checkpoint).

        Returns:
            Dictionary with training history.
        """
        epochs = max_epochs if max_epochs is not None else self.config.max_epochs
        interval = val_interval if val_interval is not None else self.config.val_interval
        wall_limit_h = (
            max_wall_hours
            if max_wall_hours is not None
            else getattr(self.config, "max_wall_hours", None)
        )
        wall_limit_sec = float(wall_limit_h) * 3600.0 if wall_limit_h and wall_limit_h > 0 else None

        budget_msg = f" | Max Wall Budget: {wall_limit_h:.2f}h" if wall_limit_sec else ""
        print(f"Starting training (Epochs {start_epoch}..{epochs}) on device: {self.device}{budget_msg}")
        start_time = time.time()
        last_epoch_sec = 0.0

        for epoch in range(start_epoch, epochs + 1):
            if wall_limit_sec is not None:
                elapsed_sec = time.time() - start_time
                if elapsed_sec + last_epoch_sec >= wall_limit_sec:
                    print(
                        f"[TIME GUARDRAIL] Elapsed {elapsed_sec / 3600.0:.2f}h reached safe budget "
                        f"({wall_limit_h:.2f}h) before Epoch {epoch}. Stopping early to preserve "
                        "time for final validation, Held-Out Test evaluation, and artifact export."
                    )
                    prev_ep = epoch - 1
                    if prev_ep >= start_epoch and (
                        not self.history["val_epochs"] or self.history["val_epochs"][-1] != prev_ep
                    ):
                        self._run_validation_and_checkpoint(val_loader, prev_ep)
                    break

            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            epoch_time = time.time() - epoch_start
            last_epoch_sec = epoch_time
            print(f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f} - Time: {epoch_time:.1f}s")

            # Validation check
            if epoch % interval == 0 or epoch == epochs:
                self._run_validation_and_checkpoint(val_loader, epoch)

            # Always save latest checkpoint
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": (
                        self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    ),
                    "best_metric": self.best_metric,
                    "best_metric_epoch": self.best_metric_epoch,
                    "history": self.history,
                    "config": self.config.__dict__
                },
                checkpoint_dir=self.config.checkpoint_dir,
                filename="latest_model.pth"
            )

        total_time = (time.time() - start_time) / 60.0
        print(f"Training completed in {total_time:.2f} minutes.")
        print(f"Best Mean Dice: {self.best_metric:.4f} at epoch {self.best_metric_epoch}")

        return self.history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, name: str = "Test Set") -> Dict[str, float]:
        """
        Evaluate model on held-out test set using sliding window inference.

        Args:
            dataloader: DataLoader supplying test 3D volumes.
            name: Display name for the evaluation run.

        Returns:
            Dictionary containing Dice scores (TC, WT, ET, mean) and optional HD95/IoU/Sens/Prec.
        """
        self.model.eval()
        self.metric.reset()

        pbar = tqdm(dataloader, desc=f"Evaluating ({name})", leave=False, disable=not sys.stdout.isatty())

        for batch in pbar:
            images = _to_plain_tensor(batch["image"], self.device)
            labels = _to_plain_tensor(batch["label"], self.device)

            with torch.amp.autocast(device_type=self.autocast_device,
                                    enabled=self.amp_enabled):
                preds = sliding_window_inference(
                    inputs=images,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.overlap,
                    mode="gaussian"
                )

            self.metric.update(
                y_pred=preds,
                y=labels,
                patient_ids=batch.get("patient_id"),
            )

        scores = self.metric.aggregate()
        hd_str = ""
        if "hd95_mean" in scores:
            hd_str = (
                f" | HD95(mm) Mean: {scores['hd95_mean']:.2f} "
                f"(TC:{scores['hd95_tc']:.2f}, WT:{scores['hd95_wt']:.2f}, ET:{scores['hd95_et']:.2f})"
            )
        print(
            f"Evaluation on {name} - "
            f"Mean Dice: {scores['dice_mean']:.4f} | "
            f"TC: {scores['dice_tc']:.4f} | "
            f"WT: {scores['dice_wt']:.4f} | "
            f"ET: {scores['dice_et']:.4f}{hd_str}"
        )
        return scores

