"""
Smoke tests for BraTS 2021 3D Segmentation pipeline.
Verifies all modules: config, data loading, transformations,
UNet architecture, loss functions, metrics, checkpointing, and 1-step training.
"""

import os
import sys
import shutil
from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

# Ensure project root is in sys.path whether executed via pytest or direct python invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BaselineConfig, get_default_config
from src.eda import scan_brats_dataset, analyze_label_distribution
from src.dataset import (
    get_brats_data_dicts,
    get_train_transforms,
    get_val_transforms
)
from src.model import build_unet_model, get_model_summary
from src.loss_metric import build_loss_fn, BraTSMetric
from src.utils import set_seed, save_checkpoint, load_checkpoint


class SyntheticVolumeDataset(Dataset):
    """Synthetic dataset generating small 3D volume batches for fast testing."""

    def __init__(self, count: int = 2, spatial_shape=(32, 32, 32)):
        self.count = count
        self.spatial_shape = spatial_shape

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # 4 modalities, 3 binary targets
        image = torch.randn(4, *self.spatial_shape, dtype=torch.float32)
        label = (torch.rand(3, *self.spatial_shape) > 0.5).float()
        return {"image": image, "label": label}


def test_config():
    """Verify configuration initialization and defaults."""
    cfg = get_default_config()
    assert cfg.in_channels == 4
    assert cfg.out_channels == 3
    assert cfg.spatial_dims == 3
    assert len(cfg.roi_size) == 3
    assert cfg.roi_size == (96, 96, 96)
    assert cfg.batch_size >= 1
    assert cfg.max_epochs >= 1


def test_scan_dataset():
    """Verify dataset scanning on local workspace directory."""
    data_dir = "dataset/BraTS2021_Training_Data"
    if os.path.exists(data_dir):
        records = scan_brats_dataset(data_dir)
        assert len(records) > 0
        first_record = records[0]
        assert "patient_id" in first_record
        assert "image" in first_record
        assert "label" in first_record
        assert len(first_record["image"]) == 4
        assert os.path.isfile(first_record["label"])


def test_dataset_split():
    """Verify patient-level 80/10/10 split logic with max_samples capping."""
    data_dir = "dataset/BraTS2021_Training_Data"
    if os.path.exists(data_dir):
        train_files, val_files, test_files = get_brats_data_dicts(
            data_dir=data_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            max_samples=10
        )
        assert len(train_files) == 8
        assert len(val_files) == 1
        assert len(test_files) == 1


def test_model_forward_pass():
    """Verify 3D UNet instantiation and forward pass tensor shapes."""
    model = build_unet_model(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
        norm="batch"
    )
    summary = get_model_summary(model)
    assert summary["total_parameters"] > 0

    dummy_input = torch.randn(1, 4, 16, 16, 16)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3, 16, 16, 16)


def test_loss_functions():
    """Verify loss functions compute scalar gradients correctly."""
    pred_logits = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    target = (torch.rand(2, 3, 16, 16, 16) > 0.5).float()

    for loss_name in ["dice_ce", "dice", "dice_focal"]:
        loss_fn = build_loss_fn(loss_name)
        loss = loss_fn(pred_logits, target)
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        loss.backward(retain_graph=True)
        assert pred_logits.grad is not None
        pred_logits.grad.zero_()


def test_brats_metric():
    """Verify BraTS multi-channel Dice evaluation metric."""
    metric = BraTSMetric()
    metric.reset()

    # Exact match should yield Dice ~ 1.0
    labels = torch.zeros(2, 3, 16, 16, 16)
    labels[:, :, 4:12, 4:12, 4:12] = 1.0

    # Large positive logits will produce sigmoid values close to 1
    preds = torch.full((2, 3, 16, 16, 16), -10.0)
    preds[:, :, 4:12, 4:12, 4:12] = 10.0

    metric.update(preds, labels)
    scores = metric.aggregate()

    assert "dice_tc" in scores
    assert "dice_wt" in scores
    assert "dice_et" in scores
    assert "dice_mean" in scores

    assert scores["dice_tc"] > 0.95
    assert scores["dice_wt"] > 0.95
    assert scores["dice_et"] > 0.95
    assert scores["dice_mean"] > 0.95


def test_checkpointing():
    """Verify saving and loading model weights."""
    test_dir = "checkpoints/test_temp"
    os.makedirs(test_dir, exist_ok=True)

    try:
        model = build_unet_model(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(8, 16),
            strides=(2,),
            num_res_units=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        state = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": 0.85
        }
        saved_file = save_checkpoint(state, test_dir, "test_model.pth")
        assert os.path.isfile(saved_file)

        model_loaded = build_unet_model(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(8, 16),
            strides=(2,),
            num_res_units=1
        )
        ckpt = load_checkpoint(saved_file, model_loaded)
        assert ckpt["epoch"] == 5
        assert ckpt["best_metric"] == 0.85

        # Check weights equivalence
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_end_to_end_single_step():
    """
    Execute 1 synthetic training step and 1 sliding window validation step
    to ensure zero runtime exceptions across the entire pipeline.
    """
    device = torch.device("cpu")
    model = build_unet_model(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = build_loss_fn("dice_ce")

    dataset = SyntheticVolumeDataset(count=2, spatial_shape=(16, 16, 16))
    loader = DataLoader(dataset, batch_size=2)

    # 1. Training step
    model.train()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss.item() > 0

    # 2. Validation step with sliding window inference
    model.eval()
    metric = BraTSMetric()
    metric.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds = sliding_window_inference(
                inputs=images,
                roi_size=(8, 8, 8),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25
            )
            metric.update(preds, labels)

    val_scores = metric.aggregate()
    assert "dice_mean" in val_scores
    assert 0.0 <= val_scores["dice_mean"] <= 1.0


def test_real_sample_transform():
    """Verify transforms on 1 real BraTS volume if dataset exists."""
    data_dir = "dataset/BraTS2021_Training_Data"
    if not os.path.exists(data_dir):
        pytest.skip("BraTS data directory not found")

    train_files, _, _ = get_brats_data_dicts(data_dir, max_samples=1)
    assert len(train_files) == 1

    transform = get_train_transforms(roi_size=(96, 96, 96))
    sample = transform(train_files[0])

    assert "image" in sample
    assert "label" in sample
    assert sample["image"].shape == (4, 96, 96, 96)
    assert sample["label"].shape == (3, 96, 96, 96)
    assert sample["label"].dtype in (torch.float32, torch.float64)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))


