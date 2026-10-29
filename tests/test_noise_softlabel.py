"""
Unit and integration tests for the Soft Label & Label Noise 3D Segmentation extensions.
Verifies:
1. Study configuration presets
2. 3D synthetic label noise modes, monotonicity, hierarchy (ET <= TC <= WT), and per-patient determinism
3. Continuous soft label transforms (Label Smoothing, 3D Gaussian Boundary, 3D Signed Distance Function)
4. Soft-label & noise-robust loss functions (SoftDice, SoftDiceBCE, SCE, GCE)
5. Calibration & uncertainty metrics (Brier Score, ECE)
6. End-to-end real BraTS 2021 noisy + soft-label transform & 3D UNet training step
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_softlabel_study_config
from src.dataset import get_brats_data_dicts, get_train_transforms
from src.loss_metric import (
    BraTSMetric,
    SoftDiceLoss,
    build_loss_fn,
    compute_calibration_stats,
)
from src.model import build_unet_model
from src.noise_soft_labels import (
    AddSyntheticLabelNoised,
    ConvertHardToSoftLabeld,
    compute_noise_corruption_stats,
    corrupt_label_3d,
    generate_soft_label_3d,
)


def _make_nested_brats_phantom(shape=(32, 32, 32)) -> torch.Tensor:
    """Create a clean nested 3-channel BraTS phantom: ET (ch 2) <= TC (ch 0) <= WT (ch 1)."""
    h, w, d = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        torch.linspace(-1, 1, d),
        indexing="ij",
    )
    r2 = zz ** 2 + yy ** 2 + xx ** 2
    wt = (r2 <= 0.55 ** 2).float()
    tc = (r2 <= 0.35 ** 2).float()
    et = (r2 <= 0.20 ** 2).float()
    return torch.stack([tc, wt, et], dim=0)


def test_study_presets():
    """Verify all Soft Label & Label Noise configuration presets initialize cleanly."""
    for preset in [
        "clean",
        "noisy_baseline",
        "noisy_soft_sdf",
        "noisy_soft_gaussian",
        "noisy_sce",
        "noisy_gce",
    ]:
        cfg = get_softlabel_study_config(preset=preset, noise_ratio=0.25)
        assert cfg.track_calibration is True
        if preset == "clean":
            assert cfg.noise_type == "none"
            assert cfg.noise_ratio == 0.0
        else:
            assert cfg.noise_ratio == 0.25


def test_label_noise_modes_and_hierarchy():
    """Verify synthetic 3D noise modes preserve shape, binary values, and ET <= TC <= WT nesting."""
    clean = _make_nested_brats_phantom((32, 32, 32))
    image = torch.ones(4, 32, 32, 32)

    # Zero noise must be exact identity
    unchanged = corrupt_label_3d(clean, image=image, noise_type="boundary", noise_ratio=0.0)
    assert torch.equal(clean, unchanged)

    for mode in ["boundary", "flip", "mixed"]:
        rng = np.random.RandomState(123)
        noisy = corrupt_label_3d(
            clean, image=image, noise_type=mode, noise_ratio=0.35, noise_radius=2, rng=rng
        )
        assert noisy.shape == clean.shape
        unique_vals = torch.unique(noisy)
        assert all(float(v) in (0.0, 1.0) for v in unique_vals)
        # Hierarchy: ET (ch 2) <= TC (ch 0) <= WT (ch 1)
        assert torch.all(noisy[2] <= noisy[0])
        assert torch.all(noisy[0] <= noisy[1])
        stats = compute_noise_corruption_stats(clean, noisy)
        assert stats["voxel_disagreement_rate"] > 0.0
        assert stats["label_dice_mean"] < 1.0


def test_boundary_noise_monotonicity_and_determinism():
    """Higher noise_ratio must corrupt more voxels, and same patient_id must yield identical static noise."""
    clean = _make_nested_brats_phantom((40, 40, 40))
    low = corrupt_label_3d(
        clean, noise_type="boundary", noise_ratio=0.10, noise_radius=2, rng=np.random.RandomState(7)
    )
    high = corrupt_label_3d(
        clean, noise_type="boundary", noise_ratio=0.45, noise_radius=2, rng=np.random.RandomState(7)
    )
    stats_low = compute_noise_corruption_stats(clean, low)
    stats_high = compute_noise_corruption_stats(clean, high)
    assert stats_high["voxel_disagreement_rate"] > stats_low["voxel_disagreement_rate"]

    transform = AddSyntheticLabelNoised(
        keys="label", noise_type="mixed", noise_ratio=0.3, seed=42, deterministic_per_patient=True
    )
    out_a1 = transform({"patient_id": "BraTS2021_00005", "label": clean.clone()})["label"]
    out_a2 = transform({"patient_id": "BraTS2021_00005", "label": clean.clone()})["label"]
    out_b = transform({"patient_id": "BraTS2021_00009", "label": clean.clone()})["label"]

    assert torch.equal(out_a1, out_a2)
    assert not torch.equal(out_a1, out_b)


def test_soft_label_methods():
    """Verify continuous soft-label generation methods produce calibrated boundary transitions in [0, 1]."""
    clean = _make_nested_brats_phantom((32, 32, 32))

    # 1. Label smoothing
    ls = generate_soft_label_3d(clean, method="label_smoothing", alpha=0.2)
    assert torch.allclose(ls.min(), torch.tensor(0.1))
    assert torch.allclose(ls.max(), torch.tensor(0.9))

    # 2. Gaussian boundary & SDF
    for method in ["gaussian_boundary", "sdf"]:
        soft = generate_soft_label_3d(clean, method=method, sigma=1.5)
        assert soft.shape == clean.shape
        assert float(soft.min()) >= 0.0
        assert float(soft.max()) <= 1.0
        # Must contain genuinely intermediate values (0.15, 0.85) along the tumor boundary
        mid_voxels = ((soft[1] > 0.15) & (soft[1] < 0.85)).sum().item()
        assert mid_voxels > 50
        # Center of tumor should remain high confidence (> 0.9), corner background low (< 0.1)
        assert float(soft[1, 16, 16, 16]) > 0.90
        assert float(soft[1, 0, 0, 0]) < 0.10


def test_soft_and_robust_losses():
    """Verify SoftDice, SoftDiceBCE, SCE, and GCE compute valid gradients on continuous soft targets."""
    logits = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    hard_target = _make_nested_brats_phantom((16, 16, 16)).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    soft_target = generate_soft_label_3d(hard_target[0], method="gaussian_boundary", sigma=1.2).unsqueeze(0).repeat(2, 1, 1, 1, 1)

    for loss_name in ["soft_dice", "soft_dice_bce", "sce", "gce"]:
        loss_fn = build_loss_fn(loss_name)
        loss = loss_fn(logits, soft_target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        logits.grad.zero_()

    # Verify SoftDiceLoss reaches ~0 when predicted probabilities match soft target
    perfect_logits = torch.logit(torch.clamp(soft_target, 1e-4, 1.0 - 1e-4))
    assert float(SoftDiceLoss()(perfect_logits, soft_target).item()) < 1e-3


def test_calibration_metrics():
    """Verify Brier score and ECE computation in BraTSMetric."""
    target = _make_nested_brats_phantom((16, 16, 16)).unsqueeze(0)
    good_logits = torch.where(target > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
    bad_logits = torch.where(target > 0.5, torch.tensor(-4.0), torch.tensor(4.0))

    cal_good = compute_calibration_stats(good_logits, target)
    cal_bad = compute_calibration_stats(bad_logits, target)
    assert cal_good["brier_score"] < cal_bad["brier_score"]
    assert cal_good["ece"] < cal_bad["ece"]

    metric = BraTSMetric(track_calibration=True)
    metric.reset()
    metric.update(good_logits, target)
    agg = metric.aggregate()
    assert agg["dice_mean"] > 0.99
    assert "brier_score" in agg
    assert "ece" in agg


def test_real_brats_noisy_soft_pipeline_and_step():
    """Verify real BraTS 2021 transform with mixed noise + SDF soft labels + 1 UNet step."""
    data_dir = "dataset/BraTS2021_Training_Data"
    if not os.path.exists(data_dir):
        pytest.skip("BraTS dataset directory not found")

    train_files, _, _ = get_brats_data_dicts(data_dir, max_samples=1)
    transform = get_train_transforms(
        roi_size=(64, 64, 64),
        noise_type="mixed",
        noise_ratio=0.3,
        soft_label_method="sdf",
        soft_sigma=1.5,
        keep_clean_label=True,
    )
    sample = transform(train_files[0])
    assert sample["image"].shape == (4, 64, 64, 64)
    assert sample["label"].shape == (3, 64, 64, 64)
    assert sample["clean_label"].shape == (3, 64, 64, 64)

    model = build_unet_model(channels=(8, 16, 32), strides=(2, 2), num_res_units=1)
    loss_fn = build_loss_fn("soft_dice_bce")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits = model(sample["image"].unsqueeze(0))
    loss = loss_fn(logits, sample["label"].unsqueeze(0))
    loss.backward()
    opt.step()
    assert float(loss.item()) > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))

