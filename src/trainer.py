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
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from src.config import BaselineConfig
from src.loss_metric import BraTSMetric
from src.utils import save_checkpoint


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
        self.metric = metric if metric is not None else BraTSMetric()

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
            leave=False
        )

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.autocast_device,
                                    enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            if self.scaler is not None and self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
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

        pbar = tqdm(dataloader, desc="Validating (Sliding Window)", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

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

            self.metric.update(y_pred=preds, y=labels)

        scores = self.metric.aggregate()
        return scores

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        val_interval: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Run complete training and validation cycle.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Number of training epochs (defaults to config.max_epochs).
            val_interval: Validation frequency in epochs (defaults to config.val_interval).

        Returns:
            Dictionary with training history.
        """
        epochs = max_epochs if max_epochs is not None else self.config.max_epochs
        interval = val_interval if val_interval is not None else self.config.val_interval

        print(f"Starting training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f} - Time: {epoch_time:.1f}s")

            # Validation check
            if epoch % interval == 0 or epoch == epochs:
                val_scores = self.validate(val_loader)
                mean_dice = val_scores["dice_mean"]

                self.history["val_epochs"].append(epoch)
                self.history["val_dice_tc"].append(val_scores["dice_tc"])
                self.history["val_dice_wt"].append(val_scores["dice_wt"])
                self.history["val_dice_et"].append(val_scores["dice_et"])
                self.history["val_dice_mean"].append(mean_dice)

                print(
                    f"Validation (Epoch {epoch:03d}) - "
                    f"Mean Dice: {mean_dice:.4f} | "
                    f"TC: {val_scores['dice_tc']:.4f} | "
                    f"WT: {val_scores['dice_wt']:.4f} | "
                    f"ET: {val_scores['dice_et']:.4f}"
                )

                # Save best checkpoint
                if mean_dice > self.best_metric:
                    self.best_metric = mean_dice
                    self.best_metric_epoch = epoch
                    ckpt_path = save_checkpoint(
                        state={
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_metric": self.best_metric,
                            "val_scores": val_scores,
                            "config": self.config.__dict__
                        },
                        checkpoint_dir=self.config.checkpoint_dir,
                        filename="best_metric_model.pth"
                    )
                    print(f"Saved new best model to: {ckpt_path} (Mean Dice: {mean_dice:.4f})")

            # Always save latest checkpoint
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": self.best_metric,
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

