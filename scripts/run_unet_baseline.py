"""
Dedicated Standalone Pipeline: Standard 3D UNet Baseline on BraTS 2021.

Runs a strictly clean (zero label noise, zero soft-label transformation) 3D UNet
baseline on BraTS 2021 with an explicit, reproducible MICCAI BraTS Metric Setup:
  1. Patient-level 80% / 10% / 10% split (Train / Validation / Held-Out Test).
  2. Full-volume 3D Sliding Window Inference with Gaussian overlap blending.
  3. Independent channel-wise Sigmoid binarization (p_c > 0.5) on overlapping BraTS regions:
       - Channel 0: TC (Tumor Core = NCR[1] U ET[4])
       - Channel 1: WT (Whole Tumor = NCR[1] U ED[2] U ET[4])
       - Channel 2: ET (Enhancing Tumor = ET[4])
  4. Official BraTS 2021 Evaluation Metrics:
       - Primary Overlap Metric   : Volumetric Dice Coefficient (TC, WT, ET, Macro Mean)
       - Primary Boundary Metric  : 95% Hausdorff Distance HD95 in mm (TC, WT, ET, Macro Mean)
       - Secondary Diagnostic Set : IoU (Jaccard), Sensitivity (Recall), Precision (PPV)
       - Empty-Region Convention  :
           * |P| == 0 & |G| == 0 -> Dice = 1.0, IoU = 1.0, HD95 = 0.0 mm
           * |P| == 0 ^ |G| == 0 -> Dice = 0.0, IoU = 0.0, HD95 = 373.13 mm (SRI24 diagonal)
  5. Model Selection & Reporting:
       - Checkpoint `best_metric_model.pth` selected strictly by highest Validation Mean Dice.
       - Reloaded once for final Held-Out Test evaluation (per-patient table + Mean +- Std / Median).
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

from src.config import ensure_kaggle_brats_extracted, get_unet_baseline_config
from src.dataset import build_dataloaders
from src.loss_metric import (
    BRATS_MAX_DISTANCE_MM,
    BraTSMetric,
    build_loss_fn,
    get_metric_setup_spec,
)
from src.model import build_unet_model, get_model_summary
from src.trainer import BraTSTrainer
from src.utils import load_checkpoint, set_seed, visualize_predictions


def print_metric_setup_banner(metric_spec: Dict[str, Any], loss_type: str) -> None:
    """Print the explicit BraTS 2021 Metric Setup protocol to stdout."""
    print("\n" + "=" * 84)
    print("BRATS 2021 3D UNET BASELINE - EXPLICIT METRIC SETUP SPECIFICATION")
    print("=" * 84)
    print(f"  Training Loss Function  : {loss_type}")
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
    print("=" * 84 + "\n")


def plot_baseline_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot 3-panel Baseline Learning Curves: Train Loss, Validation Dice, and Validation HD95."""
    has_hd95 = len(history.get("val_hd95_mean", [])) > 0
    ncols = 3 if has_hd95 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.0))

    # Panel 1: Epoch Training Loss
    train_losses = history.get("train_loss", [])
    if train_losses:
        ep_x = list(range(1, len(train_losses) + 1))
        axes[0].plot(ep_x, train_losses, color="#1f77b4", lw=2.2, marker="o", markersize=4, label="Train Loss")
    axes[0].set_title("3D UNet Baseline: Training Loss", fontsize=12, fontweight="bold")
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
    print(f"Saved baseline learning curves to: {save_path}")


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
    for rec in patient_records:
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
        description="Run dedicated Standard 3D UNet Baseline on BraTS 2021 with explicit Metric Setup."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to BraTS2021_Training_Data.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total patient cases (default: use all available cases in data_dir).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--val-interval", type=int, default=4, help="Validation frequency in epochs.")
    parser.add_argument("--roi-size", type=int, default=96, help="Cubic 3D patch size (multiple of 16, e.g. 80 or 96).")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--sw-batch-size", type=int, default=2, help="Sliding-window inference batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial AdamW learning rate.")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="dice_bce",
        choices=["dice_bce", "dice_ce", "dice", "dice_focal"],
        help="Baseline training loss ('dice_bce' = Multi-label Dice + Channel-wise BCE; 'dice_ce' = MONAI DiceCELoss).",
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window Gaussian overlap in [0, 1).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid binarization threshold per channel.")
    parser.add_argument("--cache-rate", type=float, default=1.0, help="MONAI CacheDataset rate in [0.0, 1.0].")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--device", type=str, default="", help="Optional device override ('cuda', 'mps', 'cpu').")
    parser.add_argument("--output-dir", type=str, default="", help="Optional custom output directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Optional custom checkpoint directory.")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Disable automatic DataParallel when >1 CUDA GPUs exist.")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 AMP (default: False, pure FP32 for numerical stability).")
    parser.add_argument("--no-amp", action="store_true", help="Explicitly disable FP16 AMP.")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Safe wall-clock training budget in hours (e.g. 10.2 for Kaggle 12h sessions). "
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
        help="Print Metric Setup specification, split summary, and 3D UNet architecture without training.",
    )
    args = parser.parse_args()

    resolved_data = ensure_kaggle_brats_extracted(args.data_dir)
    roi_tuple = (args.roi_size, args.roi_size, args.roi_size)

    overrides: Dict[str, Any] = {
        "max_epochs": args.epochs,
        "val_interval": args.val_interval,
        "max_wall_hours": args.max_hours,
        "roi_size": roi_tuple,
        "batch_size": args.batch_size,
        "sw_batch_size": args.sw_batch_size,
        "lr": args.lr,
        "loss_type": args.loss_type,
        "overlap": args.overlap,
        "metric_threshold": args.threshold,
        "cache_rate": 0.0 if args.dry_run else args.cache_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    if args.device:
        overrides["device"] = args.device
    if args.amp:
        overrides["amp"] = True
    elif args.no_amp:
        overrides["amp"] = False
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    config = get_unet_baseline_config(data_dir=resolved_data, **overrides)
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

    metric_spec = get_metric_setup_spec(
        threshold=config.metric_threshold,
        spacing=config.voxel_spacing,
        hd95_empty_penalty=config.hd95_empty_penalty,
        overlap=config.overlap,
        roi_size=config.roi_size,
    )
    print_metric_setup_banner(metric_spec, loss_type=config.loss_type)

    # 1. Build strictly clean patient-level DataLoaders (80% Train / 10% Val / 10% Held-Out Test)
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

    # 2. Instantiate Standard 3D UNet, Loss, Optimizer, Cosine Scheduler, and BraTSMetric
    model = build_unet_model(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        norm=config.norm,
        dropout=config.dropout,
    ).to(config.device)
    model_info = get_model_summary(model)

    n_gpus = torch.cuda.device_count() if config.device == "cuda" else 1
    if config.device == "cuda" and n_gpus > 1 and not args.no_multi_gpu:
        model = torch.nn.DataParallel(model)
        gpu_str = f"cuda DataParallel ({n_gpus} GPUs)"
    else:
        gpu_str = config.device

    print(
        f"Instantiated Standard 3D UNet | Parameters: {model_info['trainable_parameters']:,} "
        f"({model_info['model_size_mb']:.2f} MB) | Device: {gpu_str}"
    )

    if args.dry_run:
        print(
            f"Dry-run verification complete | Patient split: "
            f"Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}."
        )
        return

    loss_fn = build_loss_fn(config.loss_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-5)

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

    # 3. Train & Validate (protected by optional --max-hours Kaggle session guardrail)
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
            os.remove(latest_ckpt)

    # 4. Reload best validation checkpoint and run final Held-Out Test evaluation
    best_ckpt = os.path.join(config.checkpoint_dir, "best_metric_model.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model, device=config.device)

    test_scores = trainer.evaluate(test_loader, name="Held-Out Test Set (3D UNet Baseline)")
    test_patient_records = metric.get_patient_records()
    test_distribution = metric.summarize_distribution()

    print_test_report_tables(
        patient_records=test_patient_records,
        dist_stats=test_distribution,
        aggregate_scores=test_scores,
    )

    # 5. Export plots & self-documenting JSON report
    curves_path = os.path.join(config.output_dir, "baseline_learning_curves.png")
    plot_baseline_curves(history, save_path=curves_path)

    model.eval()
    sample_batch = next(iter(test_loader))
    raw_img, raw_gt = sample_batch["image"], sample_batch["label"]
    sample_img = (raw_img.as_tensor() if hasattr(raw_img, "as_tensor") else raw_img).to(config.device)
    sample_gt = (raw_gt.as_tensor() if hasattr(raw_gt, "as_tensor") else raw_gt).to(config.device)
    with torch.no_grad():
        sample_logits = sliding_window_inference(
            inputs=sample_img,
            roi_size=config.roi_size,
            sw_batch_size=config.sw_batch_size,
            predictor=model,
            overlap=config.overlap,
            mode="gaussian",
        )
        sample_pred = (torch.sigmoid(sample_logits) > config.metric_threshold).float()

    pred_fig_path = os.path.join(config.output_dir, "baseline_test_predictions.png")
    fig_pred = visualize_predictions(
        image=sample_img[0].cpu().numpy(),
        label=sample_gt[0].cpu().numpy(),
        pred=sample_pred[0].cpu().numpy(),
        save_path=pred_fig_path,
    )
    plt.close(fig_pred)

    report_payload = {
        "pipeline": "Standard 3D UNet Baseline (Clean BraTS 2021)",
        "metric_setup": metric_spec,
        "model_summary": model_info,
        "hyperparameters": {
            "epochs": config.max_epochs,
            "val_interval": config.val_interval,
            "roi_size": list(config.roi_size),
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer_type,
            "loss_type": config.loss_type,
            "sliding_window_overlap": config.overlap,
            "metric_threshold": config.metric_threshold,
            "hd95_empty_penalty_mm": config.hd95_empty_penalty,
            "seed": config.seed,
            "device": config.device,
        },
        "patient_splits": {
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "train_patients": [r["patient_id"] for r in train_files],
            "val_patients": [r["patient_id"] for r in val_files],
            "test_patients": [r["patient_id"] for r in test_files],
        },
        "training_time_sec": elapsed_sec,
        "best_val_epoch": int(trainer.best_metric_epoch),
        "best_val_dice_mean": round(float(trainer.best_metric), 4),
        "test_summary_mean": test_scores,
        "test_distribution_stats": test_distribution,
        "test_per_patient_records": test_patient_records,
        "history": history,
    }

    report_json_path = os.path.join(config.output_dir, "baseline_metrics_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved complete baseline metric report to: {report_json_path}")


if __name__ == "__main__":
    main()
