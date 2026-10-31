"""
Smoke tests for BraTS 2021 nnU-Net (DynUNet) Baseline pipeline.
Verifies DynUNet architecture, instance normalization, LeakyReLU activation,
multi-stage configuration, forward pass shapes, loss functions, metrics,
and 1-step synthetic end-to-end training.
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BaselineConfig, get_default_config, get_nnunet_config
from src.model import build_nnunet_model, build_model, get_model_summary
from src.loss_metric import build_loss_fn, BraTSMetric
from src.utils import set_seed, save_checkpoint, load_checkpoint


class SyntheticVolumeDataset(Dataset):
    """Synthetic dataset generating small 3D volume batches for fast testing."""

    def __init__(self, count: int = 2, spatial_shape=(16, 16, 16)):
        self.count = count
        self.spatial_shape = spatial_shape

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # 4 modalities, 3 binary targets
        image = torch.randn(4, *self.spatial_shape, dtype=torch.float32)
        label = (torch.rand(3, *self.spatial_shape) > 0.5).float()
        return {"image": image, "label": label}


def test_nnunet_config():
    """Verify nnU-Net configuration defaults."""
    cfg = get_nnunet_config()
    assert cfg.model_name == "nnunet"
    assert "nnunet" in cfg.checkpoint_dir
    assert "nnunet" in cfg.output_dir
    assert cfg.in_channels == 4
    assert cfg.out_channels == 3
    assert cfg.spatial_dims == 3
    assert cfg.nnunet_norm == "instance"
    assert cfg.nnunet_act == "leakyrelu"
    assert len(cfg.nnunet_filters) >= 3
    assert len(cfg.nnunet_strides) == len(cfg.nnunet_filters)


def test_nnunet_model_forward():
    """Verify DynUNet architecture instantiation and tensor output shape."""
    model = build_nnunet_model(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        filters=(8, 16, 32),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        norm_name="instance",
        act_name="leakyrelu",
        deep_supervision=False,
        res_block=False
    )
    summary = get_model_summary(model)
    assert summary["model_class"] == "DynUNet"
    assert summary["total_parameters"] > 0

    dummy_input = torch.randn(1, 4, 16, 16, 16)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3, 16, 16, 16)


def test_unified_model_factory():
    """Verify build_model dispatches to DynUNet when model_name='nnunet'."""
    cfg = get_nnunet_config(
        nnunet_filters=(8, 16, 32),
        nnunet_strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        nnunet_kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3))
    )
    model = build_model(cfg)
    assert model.__class__.__name__ == "DynUNet"


def test_nnunet_loss_and_metrics():
    """Verify DynUNet outputs compute valid gradients with DiceCELoss and BraTSMetric."""
    model = build_nnunet_model(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        filters=(8, 16, 32),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        norm_name="instance"
    )
    loss_fn = build_loss_fn("dice_ce")
    metric = BraTSMetric()

    dummy_input = torch.randn(2, 4, 16, 16, 16)
    target = (torch.rand(2, 3, 16, 16, 16) > 0.5).float()

    model.train()
    logits = model(dummy_input)
    loss = loss_fn(logits, target)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    loss.backward()

    metric.reset()
    metric.update(logits.detach(), target)
    scores = metric.aggregate()

    assert "dice_tc" in scores
    assert "dice_wt" in scores
    assert "dice_et" in scores
    assert "dice_mean" in scores


def test_nnunet_checkpointing():
    """Verify checkpoint saving and reloading for DynUNet."""
    test_dir = "checkpoints/test_nnunet_temp"
    os.makedirs(test_dir, exist_ok=True)

    try:
        model = build_nnunet_model(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            filters=(8, 16, 32),
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        state = {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": 0.88
        }
        saved_file = save_checkpoint(state, test_dir, "nnunet_checkpoint.pth")
        assert os.path.isfile(saved_file)

        loaded_model = build_nnunet_model(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            filters=(8, 16, 32),
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2))
        )
        ckpt = load_checkpoint(saved_file, loaded_model)
        assert ckpt["epoch"] == 3
        assert ckpt["best_metric"] == 0.88

        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_nnunet_end_to_end_step():
    """
    Execute 1 synthetic training step and 1 sliding window validation step
    using DynUNet to verify end-to-end integration without errors.
    """
    device = torch.device("cpu")
    model = build_nnunet_model(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        filters=(8, 16, 32),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        norm_name="instance"
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
        logits = model(images)
        loss = loss_fn(logits, labels)
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

    scores = metric.aggregate()
    assert "dice_mean" in scores
    assert 0.0 <= scores["dice_mean"] <= 1.0


def test_run_nnunet_baseline_cli_dry_run():
    """Verify that scripts/run_nnunet_baseline.py executes cleanly in dry-run mode."""
    import subprocess
    script_path = PROJECT_ROOT / "scripts" / "run_nnunet_baseline.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    assert res.returncode == 0, f"run_nnunet_baseline.py dry-run failed:\n{res.stderr}"
    assert "BRATS 2021 nnU-NET (DynUNet) BASELINE" in res.stdout
    assert "DynUNet" in res.stdout
    assert "Dry Run" in res.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
