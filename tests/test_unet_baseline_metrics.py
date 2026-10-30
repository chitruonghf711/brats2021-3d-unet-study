"""
Unit tests verifying the explicit BraTS 2021 3D UNet Baseline Metric Setup:
- Volumetric Dice, IoU (Jaccard), Sensitivity, Precision
- Exact 3D 95% Hausdorff Distance (HD95 in mm)
- Empty-region edge-case rules (both empty vs single empty)
- Per-patient record tracking and distribution summary (Mean, Std, Median)
- Clean baseline configuration isolation
"""

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_unet_baseline_config
from src.loss_metric import (
    BRATS_MAX_DISTANCE_MM,
    BraTSMetric,
    build_loss_fn,
    compute_binary_volume_metrics,
    compute_hd95_3d,
    get_metric_setup_spec,
)


def test_unet_baseline_config_is_strictly_clean():
    """Verify baseline config enforces clean supervision and enables full BraTS metrics."""
    cfg = get_unet_baseline_config()
    assert cfg.model_name == "unet"
    assert cfg.noise_type == "none"
    assert cfg.noise_ratio == 0.0
    assert cfg.soft_label_method == "none"
    assert cfg.compute_hd95 is True
    assert cfg.compute_confusion_metrics is True
    assert cfg.metric_threshold == 0.5
    assert cfg.voxel_spacing == (1.0, 1.0, 1.0)
    loss_fn = build_loss_fn(cfg.loss_type)
    assert loss_fn is not None


def test_empty_region_conventions_both_and_single_empty():
    """
    Verify official BraTS Challenge empty-mask edge cases:
      1. Both empty (|P|==0, |G|==0) -> Dice=1.0, IoU=1.0, Sens=1.0, Prec=1.0, HD95=0.0
      2. Pred empty, GT non-empty   -> Dice=0.0, IoU=0.0, Sens=0.0, Prec=0.0, HD95=373.13
      3. Pred non-empty, GT empty   -> Dice=0.0, IoU=0.0, Sens=0.0, Prec=0.0, HD95=373.13
    """
    empty = np.zeros((20, 20, 20), dtype=np.float32)
    lesion = np.zeros((20, 20, 20), dtype=np.float32)
    lesion[6:12, 6:12, 6:12] = 1.0

    # 1. Both empty (e.g., LGG patient with no Enhancing Tumor and clean prediction)
    m_both = compute_binary_volume_metrics(empty, empty, compute_hd95=True)
    assert m_both["status"] == "both_empty"
    assert m_both["dice"] == 1.0
    assert m_both["iou"] == 1.0
    assert m_both["sensitivity"] == 1.0
    assert m_both["precision"] == 1.0
    assert m_both["hd95"] == 0.0

    # 2. Complete miss (FN)
    m_fn = compute_binary_volume_metrics(empty, lesion, compute_hd95=True)
    assert m_fn["status"] == "pred_empty_fn"
    assert m_fn["dice"] == 0.0
    assert m_fn["iou"] == 0.0
    assert m_fn["sensitivity"] == 0.0
    assert m_fn["hd95"] == BRATS_MAX_DISTANCE_MM

    # 3. False-positive hallucination on empty GT (FP)
    m_fp = compute_binary_volume_metrics(lesion, empty, compute_hd95=True)
    assert m_fp["status"] == "gt_empty_fp"
    assert m_fp["dice"] == 0.0
    assert m_fp["iou"] == 0.0
    assert m_fp["precision"] == 0.0
    assert m_fp["hd95"] == BRATS_MAX_DISTANCE_MM


def test_hd95_exact_known_translation():
    """Verify exact 3D Euclidean surface HD95 for a known 2-voxel shift along one axis."""
    gt = np.zeros((32, 32, 32), dtype=np.float32)
    pred = np.zeros((32, 32, 32), dtype=np.float32)
    gt[8:20, 8:20, 8:20] = 1.0
    pred[10:22, 8:20, 8:20] = 1.0  # Shifted by exactly +2 voxels along axis 0

    hd95_mm, status = compute_hd95_3d(pred, gt, spacing=(1.0, 1.0, 1.0))
    assert status == "both_non_empty"
    assert abs(hd95_mm - 2.0) < 1e-4


def test_brats_metric_per_patient_and_empty_et_penalization():
    """
    Verify BraTSMetric tracks per-patient Dice + HD95 + IoU + Sens + Prec
    and properly rewards/penalizes empty ET channels across patients.
    """
    metric = BraTSMetric(compute_hd95=True, compute_confusion_metrics=True, ignore_empty=False)
    metric.reset()

    # Patient A: All 3 channels non-empty & exact match -> Dice=1.0, HD95=0.0
    gt_a = torch.zeros(1, 3, 16, 16, 16)
    gt_a[:, :, 4:12, 4:12, 4:12] = 1.0
    logits_a = torch.where(gt_a > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    metric.update(logits_a, gt_a, patient_ids=["BraTS2021_A"])

    # Patient B: TC & WT exact match, ET empty in GT (0 voxels) BUT model hallucinates ET!
    # Under BraTS convention: ET Dice must be 0.0 (not ignored as NaN!) and ET HD95 = 373.13
    gt_b = torch.zeros(1, 3, 16, 16, 16)
    gt_b[:, 0:2, 4:12, 4:12, 4:12] = 1.0  # ch2 (ET) is empty in GT
    logits_b = torch.full((1, 3, 16, 16, 16), -10.0)
    logits_b[:, 0:2, 4:12, 4:12, 4:12] = 10.0
    logits_b[:, 2, 5:9, 5:9, 5:9] = 10.0  # False-positive ET prediction!
    metric.update(logits_b, gt_b, patient_ids=["BraTS2021_B"])

    agg = metric.aggregate()
    records = metric.get_patient_records()
    dist = metric.summarize_distribution()

    assert len(records) == 2
    assert records[0]["patient_id"] == "BraTS2021_A"
    assert records[0]["dice_et"] == 1.0
    assert records[0]["hd95_et"] == 0.0

    assert records[1]["patient_id"] == "BraTS2021_B"
    assert records[1]["dice_et"] == 0.0
    assert records[1]["hd95_et"] == BRATS_MAX_DISTANCE_MM

    # Macro-average across Patient A (1.0) and Patient B (0.0) -> ET Dice = 0.5
    assert abs(agg["dice_tc"] - 1.0) < 1e-4
    assert abs(agg["dice_wt"] - 1.0) < 1e-4
    assert abs(agg["dice_et"] - 0.5) < 1e-4
    assert "hd95_mean" in agg
    assert "iou_mean" in agg
    assert "sensitivity_mean" in agg
    assert "precision_mean" in agg
    assert "dice_mean" in dist and "std" in dist["dice_mean"] and "median" in dist["dice_mean"]

    spec = get_metric_setup_spec()
    assert "primary_metrics" in spec and "empty_region_convention" in spec


def test_kaggle_uncompressed_nii_scanning_and_time_guardrail(tmp_path):
    """
    Verify two Kaggle-critical features:
      1. `scan_brats_dataset` discovers uncompressed `.nii` files (auto-unzipped by Kaggle).
      2. `BraTSTrainer.fit(..., max_wall_hours=...)` gracefully halts before session timeout
         and still triggers final validation + checkpoint saving.
    """
    import nibabel as nib
    from torch.utils.data import DataLoader, Dataset
    from src.eda import scan_brats_dataset
    from src.model import build_unet_model
    from src.trainer import BraTSTrainer

    # 1. Uncompressed .nii Kaggle structure check
    p_dir = tmp_path / "BraTS2021_99999"
    p_dir.mkdir(parents=True)
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    for mod in ("flair", "t1", "t1ce", "t2", "seg"):
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p_dir / f"BraTS2021_99999_{mod}.nii"))

    records = scan_brats_dataset(str(tmp_path))
    assert len(records) == 1
    assert records[0]["patient_id"] == "BraTS2021_99999"
    assert records[0]["flair"].endswith(".nii")

    # 2. Wall-clock guardrail check (tiny budget stops cleanly after epoch 1 instead of running 10 epochs)
    class TinyVolDS(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            img = torch.randn(4, 16, 16, 16)
            lbl = torch.zeros(3, 16, 16, 16)
            lbl[:, 4:10, 4:10, 4:10] = 1.0
            return {"image": img, "label": lbl, "patient_id": f"P_{idx}"}

    cfg = get_unet_baseline_config(
        checkpoint_dir=str(tmp_path / "ckpts"),
        output_dir=str(tmp_path / "outs"),
        roi_size=(16, 16, 16),
        device="cpu",
        amp=False,
    )
    model = build_unet_model(channels=(8, 16), strides=(2,), num_res_units=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss_fn("dice_bce")
    trainer = BraTSTrainer(model=model, optimizer=opt, loss_fn=loss_fn, config=cfg)

    loader = DataLoader(TinyVolDS(), batch_size=1)
    # Budget of 1e-6 hours (~3.6 ms) allows Epoch 1 to complete, then halts cleanly before Epoch 2
    hist = trainer.fit(loader, loader, max_epochs=10, val_interval=5, max_wall_hours=1e-6)
    assert len(hist["train_loss"]) == 1
    assert len(hist["val_epochs"]) == 1
    assert (tmp_path / "ckpts" / "best_metric_model.pth").is_file()

