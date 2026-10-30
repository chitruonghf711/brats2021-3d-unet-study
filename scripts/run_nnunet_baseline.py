"""
Dedicated Standalone Pipeline: nnU-Net (DynUNet) Baseline on BraTS 2021.

Runs a dynamic 3D U-Net (MONAI DynUNet) baseline on BraTS 2021 with:
  1. Full BraTS 2021 dataset support (all 1,251 patient cases by default, or custom subset).
  2. Patient-level 80% / 10% / 10% split (Train / Validation / Held-Out Test).
  3. Dynamic architectural design (nnU-Net standard):
       - Instance Normalization (InstanceNorm3d) for invariant small-batch statistics.
       - LeakyReLU activation (negative_slope = 0.01).
       - Dynamic isotropic/anisotropic convolutional strides and kernels.
       - Optional deep supervision and residual encoder blocks (ResEnc).
  4. Explicit MICCAI BraTS Metric Setup:
       - Overlap Metric : Volumetric Dice Similarity Coefficient (TC, WT, ET, Macro Mean).
       - Boundary Metric: 95% Hausdorff Distance HD95 in mm (3D Euclidean Distance Transform).
       - Diagnostic Set : IoU (Jaccard), Sensitivity (Recall), Precision (PPV).
       - Empty-Region Convention:
           * |P| == 0 & |G| == 0 -> Dice = 1.0, IoU = 1.0, HD95 = 0.0 mm
           * |P| == 0 ^ |G| == 0 -> Dice = 0.0, IoU = 0.0, HD95 = 373.13 mm (SRI24 diagonal)
  5. Checkpointing & Benchmark Reporting:
       - Selection strictly based on best Validation Mean Dice -> best_metric_model.pth.
       - Full unbiased evaluation on 10% Held-Out Test set (~125 cases on full dataset).
       - Generation of metrics report (.json), learning curves (.png), and slice predictions (.png).
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_kaggle_brats_extracted, get_nnunet_config
from src.dataset import build_dataloaders
from src.loss_metric import (
    BRATS_MAX_DISTANCE_MM,
    BraTSMetric,
    build_loss_fn,
    get_metric_setup_spec,
)
from src.model import build_nnunet_model, get_model_summary
from src.trainer import BraTSTrainer
from src.utils import load_checkpoint, set_seed, visualize_predictions


def print_nnunet_metric_setup_banner(
    metric_spec: Dict[str, Any],
    loss_type: str,
    model_info: Dict[str, Any],
    optimizer_name: str,
    total_cases: int,
) -> None:
    """Print the explicit BraTS 2021 nnU-Net Setup protocol to stdout."""
    print("\n" + "=" * 88)
    print("BRATS 2021 nnU-NET (DynUNet) BASELINE - FULL DATASET SPECIFICATION")
    print("=" * 88)
    print(f"  Dataset Scope           : Full BraTS 2021 (Total Cases: {total_cases})")
    print(f"  Architecture            : MONAI DynUNet (nnU-Net formulation)")
    print(f"  Normalization           : InstanceNorm3d (batch-size invariant)")
    print(f"  Activation              : LeakyReLU (negative_slope=0.01)")
    print(f"  Parameters              : {model_info['trainable_parameters']:,} ({model_info['model_size_mb']:.2f} MB)")
    print(f"  Optimizer & Loss        : {optimizer_name.upper()} | Loss: {loss_type}")
    print(f"  Evaluation Unit         : {metric_spec['evaluation_unit']}")
    inf = metric_spec["inference_mode"]
    print(
        f"  Inference Protocol      : {inf['method']} | ROI={tuple(inf['roi_size'])} | "
        f"overlap={inf['overlap']} | blend={inf['blending']}"
    )
    print(f"  Channel Binarization    : {inf['activation']} -> {inf['binarization']}")
    print("  Target Sub-Regions      :")
    for ch_k, desc in metric_spec["target_subregions"].items():
        print(f"    - {ch_k:<14}: {desc}")
    print("  Primary Metrics         :")
    for m_k, desc in metric_spec["primary_metrics"].items():
        print(f"    - {m_k:<14}: {desc}")
    print("  Secondary Metrics       : IoU (Jaccard), Sensitivity (Recall), Precision (PPV)")
    print("  Empty-Mask Edge Cases   :")
    for e_k, desc in metric_spec["empty_region_convention"].items():
        print(f"    - {e_k}: {desc}")
    print(
        "  Checkpoint Selection    : "
        f"{metric_spec['aggregation_and_selection']['checkpoint_selection_criterion']}"
    )
    print("=" * 88 + "\n")


def plot_nnunet_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot 3-panel nnU-Net Learning Curves: Train Loss, Validation Dice, and Validation HD95."""
    has_hd95 = len(history.get("val_hd95_mean", [])) > 0
    ncols = 3 if has_hd95 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.0))

    # Panel 1: Epoch Training Loss
    train_losses = history.get("train_loss", [])
    if train_losses:
        ep_x = list(range(1, len(train_losses) + 1))
        axes[0].plot(ep_x, train_losses, color="#1f77b4", lw=2.2, marker="o", markersize=4, label="Train Loss")
    axes[0].set_title("nnU-Net (DynUNet) Baseline: Training Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel 2: Validation Volumetric Dice (TC, WT, ET, Mean)
    val_ep = history.get("val_epochs", [])
    if val_ep:
        axes[1].plot(val_ep, history["val_dice_wt"], marker="s", lw=1.8, color="#4c78a8", label="WT Dice")
        axes[1].plot(val_ep, history["val_dice_tc"], marker="^", lw=1.8, color="#e45756", label="TC Dice")
        axes[1].plot(val_ep, history["val_dice_et"], marker="d", lw=1.8, color="#f58518", label="ET Dice")
        axes[1].plot(val_ep, history["val_dice_mean"], marker="o", lw=2.6, color="#2ca02c", label="Mean Dice (Selection)")
    axes[1].set_title("Validation Volumetric Dice (Higher is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient [0, 1]")
    axes[1].set_ylim([0.0, 1.0])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # Panel 3: Validation Boundary Error HD95 in mm (TC, WT, ET, Mean)
    if has_hd95 and val_ep:
        axes[2].plot(val_ep, history["val_hd95_wt"], marker="s", lw=1.8, color="#4c78a8", label="WT HD95")
        axes[2].plot(val_ep, history["val_hd95_tc"], marker="^", lw=1.8, color="#e45756", label="TC HD95")
        axes[2].plot(val_ep, history["val_hd95_et"], marker="d", lw=1.8, color="#f58518", label="ET HD95")
        axes[2].plot(val_ep, history["val_hd95_mean"], marker="o", lw=2.6, color="#d62728", label="Mean HD95")
        axes[2].set_title("Validation Boundary HD95 in mm (Lower is Better)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("95% Hausdorff Distance (mm)")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved nnU-Net learning curves to: {save_path}")


def print_test_report_tables(
    patient_records: List[Dict[str, Any]],
    dist_stats: Dict[str, Dict[str, float]],
    aggregate_scores: Dict[str, float],
) -> None:
    """Print formatted per-patient breakdown and aggregate distribution summary tables."""
    print("\n" + "#" * 96)
    print("1. HELD-OUT TEST SET: PER-PATIENT 3D VOLUME METRICS")
    print("#" * 96)
    p_hdr = (
        f"{'Patient ID':<18} | {'Mean Dice':>9} | {'WT Dice':>7} | {'TC Dice':>7} | {'ET Dice':>7} | "
        f"{'Mean HD95':>9} | {'WT HD95':>7} | {'TC HD95':>7} | {'ET HD95':>7}"
    )
    print(p_hdr)
    print("-" * len(p_hdr))
    # Show first 15 and last 5 if large set
    if len(patient_records) > 25:
        display_records = patient_records[:15] + [None] + patient_records[-5:]
    else:
        display_records = patient_records

    for rec in display_records:
        if rec is None:
            print(f"... ({len(patient_records) - 20} more test patients evaluated) ...")
            continue
        print(
            f"{rec['patient_id']:<18} | "
            f"{rec['dice_mean']:>9.4f} | "
            f"{rec['dice_wt']:>7.4f} | "
            f"{rec['dice_tc']:>7.4f} | "
            f"{rec['dice_et']:>7.4f} | "
            f"{rec.get('hd95_mean', 0.0):>8.2f}m | "
            f"{rec.get('hd95_wt', 0.0):>6.2f}m | "
            f"{rec.get('hd95_tc', 0.0):>6.2f}m | "
            f"{rec.get('hd95_et', 0.0):>6.2f}m"
        )

    print("\n" + "#" * 96)
    print("2. HELD-OUT TEST SET: AGGREGATE SUMMARY ACROSS SUB-REGIONS")
    print("#" * 96)
    s_hdr = (
        f"{'Region':<16} | {'Dice (Mean +- Std)':>20} | {'Median Dice':>11} | "
        f"{'HD95 mm (Mean +- Std)':>22} | {'IoU':>7} | {'Sens':>7} | {'Prec':>7}"
    )
    print(s_hdr)
    print("-" * len(s_hdr))
    for label, key in [("Whole Tumor (WT)", "wt"), ("Tumor Core (TC)", "tc"), ("Enhancing (ET)", "et"), ("Macro Mean", "mean")]:
        d_stat = dist_stats.get(f"dice_{key}", {"mean": aggregate_scores.get(f"dice_{key}", 0.0), "std": 0.0, "median": 0.0})
        h_stat = dist_stats.get(f"hd95_{key}", {"mean": aggregate_scores.get(f"hd95_{key}", 0.0), "std": 0.0})
        iou_v = aggregate_scores.get(f"iou_{key}", 0.0)
        sen_v = aggregate_scores.get(f"sensitivity_{key}", 0.0)
        pre_v = aggregate_scores.get(f"precision_{key}", 0.0)
        print(
            f"{label:<16} | "
            f"{d_stat['mean']:>8.4f} +- {d_stat['std']:<8.4f} | "
            f"{d_stat['median']:>11.4f} | "
            f"{h_stat['mean']:>9.2f} +- {h_stat['std']:<9.2f} | "
            f"{iou_v:>7.4f} | "
            f"{sen_v:>7.4f} | "
            f"{pre_v:>7.4f}"
        )
    print("#" * 96 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dedicated nnU-Net (DynUNet) Baseline on full BraTS 2021 with explicit Metric Setup."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to BraTS2021_Training_Data.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total patient cases (default: None = use ALL available cases in data_dir, e.g. 1,251).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--val-interval", type=int, default=5, help="Validation frequency in epochs (default: 5).")
    parser.add_argument("--roi-size", type=int, default=96, help="Cubic 3D patch size (multiple of 16, e.g. 96).")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--sw-batch-size", type=int, default=4, help="Sliding-window inference batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (1e-3 for AdamW, 1e-2 for SGD).")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer type: 'adamw' (default) or 'sgd' (classic nnU-Net SGD with Nesterov momentum 0.99).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="dice_bce",
        choices=["dice_bce", "dice_ce", "dice"],
        help="Training loss ('dice_bce' = Multi-label Dice + Channel-wise BCE; 'dice_ce' = MONAI DiceCELoss).",
    )
    parser.add_argument("--norm", type=str, default="instance", choices=["instance", "batch"], help="Normalization type.")
    parser.add_argument("--act", type=str, default="leakyrelu", help="Activation function (default: leakyrelu).")
    parser.add_argument("--res-block", action="store_true", help="Enable residual units (nnU-Net ResEnc variant).")
    parser.add_argument("--deep-supervision", action="store_true", help="Enable deep supervision multi-scale outputs.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window Gaussian overlap in [0, 1).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid binarization threshold per channel.")
    parser.add_argument("--cache-rate", type=float, default=0.0, help="MONAI CacheDataset rate in [0.0, 1.0].")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--device", type=str, default="", help="Optional device override ('cuda', 'mps', 'cpu').")
    parser.add_argument("--output-dir", type=str, default="", help="Optional custom output directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Optional custom checkpoint directory.")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Disable automatic DataParallel when >1 CUDA GPUs exist.")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 AMP.")
    parser.add_argument("--no-amp", action="store_true", help="Explicitly disable FP16 AMP (pure FP32, default).")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Safe wall-clock training budget in hours (e.g. 11.0 for Kaggle 12h sessions). "
             "Stops training cleanly before timeout so Held-Out Test & artifacts always finish saving.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a latest_model.pth checkpoint to resume training across sessions.",
    )
    parser.add_argument("--keep-latest", action="store_true", help="Retain latest_model.pth alongside best_metric_model.pth.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Metric Setup specification, split summary, and DynUNet architecture without training.",
    )
    args = parser.parse_args()

    resolved_data = ensure_kaggle_brats_extracted(args.data_dir)
    roi_tuple = (args.roi_size, args.roi_size, args.roi_size)

    overrides: Dict[str, Any] = {
        "roi_size": roi_tuple,
        "batch_size": args.batch_size,
        "sw_batch_size": args.sw_batch_size,
        "max_epochs": args.epochs,
        "val_interval": args.val_interval,
        "lr": args.lr,
        "optimizer_type": args.optimizer,
        "loss_type": args.loss_type,
        "overlap": args.overlap,
        "metric_threshold": args.threshold,
        "cache_rate": 0.0 if args.dry_run else args.cache_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "nnunet_norm": args.norm,
        "nnunet_act": args.act,
        "nnunet_res_block": args.res_block,
        "nnunet_deep_supervision": args.deep_supervision,
    }
    if args.device:
        overrides["device"] = args.device
    if args.amp:
        overrides["amp"] = True
    elif args.no_amp:
        overrides["amp"] = False
    if args.max_hours is not None:
        overrides["max_wall_hours"] = args.max_hours
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    config = get_nnunet_config(data_dir=resolved_data, **overrides)
    set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Protect multi-worker 3D MRI DataLoaders in Docker/Kaggle containers with restricted /dev/shm
    if config.num_workers > 0 and os.path.exists("/dev/shm"):
        try:
            shm_free_gb = shutil.disk_usage("/dev/shm").free / (1024 ** 3)
            if shm_free_gb < 2.0:
                torch.multiprocessing.set_sharing_strategy("file_system")
                print(f"[Worker Guard] /dev/shm free={shm_free_gb:.2f} GiB -> switched PyTorch sharing to 'file_system'.")
        except Exception:
            pass

    # 1. Build patient-level DataLoaders (80% Train / 10% Val / 10% Held-Out Test)
    train_loader, val_loader, test_loader, train_files, val_files, test_files = build_dataloaders(
        data_dir=config.data_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=config.num_workers,
        roi_size=config.roi_size,
        cache_rate=config.cache_rate,
        seed=config.seed,
        max_samples=args.max_samples,
        crop_eval_foreground=True,
    )
    total_scanned = len(train_files) + len(val_files) + len(test_files)

    # 2. Instantiate DynUNet Model
    model = build_nnunet_model(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        filters=config.nnunet_filters,
        strides=config.nnunet_strides,
        kernel_size=config.nnunet_kernel_size,
        norm_name=config.nnunet_norm,
        act_name=config.nnunet_act,
        deep_supervision=config.nnunet_deep_supervision,
        res_block=config.nnunet_res_block,
    ).to(config.device)
    model_info = get_model_summary(model)

    n_gpus = torch.cuda.device_count() if config.device == "cuda" else 1
    if config.device == "cuda" and n_gpus > 1 and not args.no_multi_gpu:
        model = torch.nn.DataParallel(model)
        gpu_str = f"cuda DataParallel ({n_gpus} GPUs)"
    else:
        gpu_str = config.device

    metric_spec = get_metric_setup_spec(
        threshold=config.metric_threshold,
        spacing=config.voxel_spacing,
        hd95_empty_penalty=config.hd95_empty_penalty,
        overlap=config.overlap,
        roi_size=config.roi_size,
    )
    print_nnunet_metric_setup_banner(
        metric_spec=metric_spec,
        loss_type=config.loss_type,
        model_info=model_info,
        optimizer_name=config.optimizer_type,
        total_cases=total_scanned,
    )
    print(
        f"Instantiated DynUNet (nnU-Net) | Parameters: {model_info['trainable_parameters']:,} "
        f"({model_info['model_size_mb']:.2f} MB) | Device: {gpu_str}\n"
        f"Patient Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    )

    if args.dry_run:
        print(f"\n[Dry Run] Verification complete. Exiting without training.")
        return

    # 3. Setup Loss, Optimizer, Scheduler, Metric, and Trainer
    loss_fn = build_loss_fn(config.loss_type)
    if config.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.sgd_momentum,
            nesterov=config.sgd_nesterov,
            weight_decay=3e-5,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=1e-5,
    )

    metric = BraTSMetric(
        include_background=True,
        track_calibration=False,
        compute_hd95=True,
        compute_confusion_metrics=True,
        threshold=config.metric_threshold,
        spacing=config.voxel_spacing,
        hd95_empty_penalty=config.hd95_empty_penalty,
        ignore_empty=False,
    )

    trainer = BraTSTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        lr_scheduler=scheduler,
        metric=metric,
    )

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer=optimizer, device=config.device)
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.best_metric = float(ckpt.get("best_metric", -1.0))
        trainer.best_metric_epoch = int(ckpt.get("best_metric_epoch", ckpt.get("epoch", -1)))
        if isinstance(ckpt.get("history"), dict):
            trainer.history.update(ckpt["history"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at Epoch {start_epoch} (Best Val Mean Dice: {trainer.best_metric:.4f})")

    # 4. Train & Validate across full dataset
    t0 = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.max_epochs,
        val_interval=config.val_interval,
        max_wall_hours=config.max_wall_hours,
        start_epoch=start_epoch,
    )
    elapsed_sec = round(time.time() - t0, 2)

    if not args.keep_latest:
        latest_ckpt = os.path.join(config.checkpoint_dir, "latest_model.pth")
        if os.path.exists(latest_ckpt):
            print(f"Retained latest model: {latest_ckpt}")

    # 5. Save Learning Curves
    curves_path = os.path.join(config.output_dir, "nnunet_learning_curves.png")
    plot_nnunet_curves(history, curves_path)

    # 6. Load Best Checkpoint for Held-Out Test Evaluation
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_metric_model.pth")
    if not os.path.isfile(best_ckpt_path):
        print(f"[Warning] Best checkpoint not found at {best_ckpt_path}; using current in-memory model weights.")
        best_epoch = trainer.best_metric_epoch
        best_val_dice = trainer.best_metric
    else:
        best_ckpt = load_checkpoint(best_ckpt_path, model, device=config.device)
        best_epoch = int(best_ckpt.get("epoch", trainer.best_metric_epoch))
        best_val_dice = float(best_ckpt.get("best_metric", trainer.best_metric))
        print(f"\nReloaded best checkpoint from {best_ckpt_path} (Epoch {best_epoch}, Val Mean Dice = {best_val_dice:.4f})")

    # 7. Unbiased Evaluation on Held-Out Test Set (10% ~ 125 cases on full dataset)
    print("\n" + "=" * 88)
    print(f"HELD-OUT TEST SET EVALUATION: {len(test_files)} CASES (DynUNet Best Checkpoint)")
    print("=" * 88)

    test_metric = BraTSMetric(
        include_background=True,
        track_calibration=False,
        compute_hd95=True,
        compute_confusion_metrics=True,
        threshold=config.metric_threshold,
        spacing=config.voxel_spacing,
        hd95_empty_penalty=config.hd95_empty_penalty,
        ignore_empty=False,
    )

    eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    eval_model.eval()

    patient_records: List[Dict[str, Any]] = []
    test_qualitative_sample: Optional[Dict[str, Any]] = None

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)
            pid = test_files[idx]["patient_id"]

            preds = sliding_window_inference(
                inputs=images,
                roi_size=config.roi_size,
                sw_batch_size=config.sw_batch_size,
                predictor=eval_model,
                overlap=config.overlap,
                mode="gaussian",
            )
            case_scores = test_metric.compute_case_metrics(preds[0], labels[0], patient_id=pid)
            patient_records.append(case_scores)
            test_metric.update(preds, labels)

            # Store the first case with non-empty tumor for qualitative visualization
            if test_qualitative_sample is None and float(case_scores.get("dice_wt", 0.0)) > 0.3:
                test_qualitative_sample = {
                    "image": images[0].detach().cpu().numpy(),
                    "label": labels[0].detach().cpu().numpy(),
                    "pred": torch.sigmoid(preds[0]).detach().cpu().numpy(),
                    "pid": pid,
                    "case_scores": case_scores,
                }

    aggregate_test_scores = test_metric.aggregate()

    # Compute distribution statistics (Mean, Std, Median, Min, Max) across patient records
    dist_stats: Dict[str, Dict[str, float]] = {}
    metric_keys = [
        "dice_mean", "dice_wt", "dice_tc", "dice_et",
        "hd95_mean", "hd95_wt", "hd95_tc", "hd95_et",
        "iou_mean", "iou_wt", "iou_tc", "iou_et",
        "sensitivity_mean", "sensitivity_wt", "sensitivity_tc", "sensitivity_et",
        "precision_mean", "precision_wt", "precision_tc", "precision_et",
    ]
    for mk in metric_keys:
        vals = [float(r[mk]) for r in patient_records if mk in r and not np.isnan(r[mk])]
        if vals:
            dist_stats[mk] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    # Print summary tables to stdout
    print_test_report_tables(patient_records, dist_stats, aggregate_test_scores)

    # 8. Generate Qualitative Prediction Slices
    if test_qualitative_sample is not None:
        pred_fig_path = os.path.join(config.output_dir, "nnunet_test_predictions.png")
        visualize_predictions(
            image=test_qualitative_sample["image"],
            clean_label=test_qualitative_sample["label"],
            pred_probs=test_qualitative_sample["pred"],
            title_prefix=f"nnU-Net Baseline — Patient {test_qualitative_sample['pid']} (Dice: {test_qualitative_sample['case_scores']['dice_mean']:.3f})",
            save_path=pred_fig_path,
        )

    # 9. Dump Comprehensive JSON Metrics Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "DynUNet (MONAI nnU-Net)",
        "model_parameters": model_info["trainable_parameters"],
        "model_size_mb": model_info["model_size_mb"],
        "dataset_split": {
            "total_cases": total_scanned,
            "train_cases": len(train_files),
            "val_cases": len(val_files),
            "test_cases": len(test_files),
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
        },
        "training_configuration": {
            "epochs_trained": len(history.get("train_loss", [])),
            "max_epochs": config.max_epochs,
            "val_interval": config.val_interval,
            "batch_size": config.batch_size,
            "sw_batch_size": config.sw_batch_size,
            "roi_size": list(config.roi_size),
            "optimizer": config.optimizer_type,
            "learning_rate": config.lr,
            "loss_function": config.loss_type,
            "precision": "FP32 (amp=False)" if not config.amp else "AMP (FP16)",
            "normalization": config.nnunet_norm,
            "activation": config.nnunet_act,
            "residual_block": config.nnunet_res_block,
            "deep_supervision": config.nnunet_deep_supervision,
            "elapsed_seconds": elapsed_sec,
        },
        "best_val_epoch": best_epoch,
        "best_val_dice_mean": round(best_val_dice, 4),
        "test_summary_mean": {k: round(float(v), 4) for k, v in aggregate_test_scores.items()},
        "test_distribution_stats": dist_stats,
        "patient_test_records": patient_records,
        "metric_setup_specification": metric_spec,
    }

    report_path = os.path.join(config.output_dir, "nnunet_metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nFinal nnU-Net metrics report saved to: {report_path}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
