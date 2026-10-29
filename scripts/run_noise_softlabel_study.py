"""
End-to-End Pilot Study Runner: 3D UNet under Label Noise & Soft Labels on BraTS 2021.

Executes a controlled 4-arm comparative experiment on BraTS-Mini (patient-level 80/10/10 split):
  1. 'clean'            : Clean hard ground truth + standard DiceCE loss (Upper-bound anchor)
  2. 'noisy_baseline'   : Corrupted training labels + standard DiceCE loss (Degradation check)
  3. 'noisy_soft_sdf'   : Corrupted training labels + 3D SDF Soft Labels + SoftDiceBCE loss
  4. 'noisy_sce'        : Corrupted training labels + Label Smoothing + Symmetric CE (SCE) loss

All models are validated and tested strictly against CLEAN ground-truth masks on unseen patients,
reporting Dice (TC, WT, ET, Mean) as well as Calibration metrics (ECE, Brier Score), and saving
publication-ready figures and JSON summary artifacts to `outputs/softlabel_study/`.
"""

import argparse
import gc
import json
import os
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
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_softlabel_study_config, resolve_dataset_dir
from src.dataset import build_dataloaders
from src.eda import analyze_label_distribution, plot_patient_modalities, scan_brats_dataset
from src.loss_metric import BraTSMetric, build_loss_fn
from src.model import build_model, get_model_summary
from src.noise_soft_labels import (
    compute_noise_corruption_stats,
    corrupt_label_3d,
    generate_soft_label_3d,
)
from src.trainer import BraTSTrainer
from src.utils import (
    load_checkpoint,
    set_seed,
    visualize_noise_and_soft_labels,
    visualize_uncertainty_and_predictions,
)


EXPERIMENT_DISPLAY_NAMES = {
    "clean": "1. Clean GT + DiceCE (Anchor)",
    "noisy_baseline": "2. Noisy Hard + DiceCE",
    "noisy_soft_sdf": "3. Noisy + SDF Soft + SoftDiceBCE",
    "noisy_soft_gaussian": "3b. Noisy + Gaussian Soft + SoftDiceBCE",
    "noisy_sce": "4. Noisy + Smoothing + SCE Robust",
    "noisy_gce": "4b. Noisy + GCE Robust",
}


def generate_eda_and_noise_visuals(
    data_dir: str,
    output_dir: str,
    noise_ratio: float = 0.35,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate EDA slice plot, noise/soft-label demonstration figure, and corruption table."""
    os.makedirs(output_dir, exist_ok=True)
    records = scan_brats_dataset(data_dir)
    if not records:
        raise FileNotFoundError(f"No valid BraTS records found in {data_dir}")

    sample_rec = records[0]
    eda_path = os.path.join(output_dir, "eda_sample_patient.png")
    fig_eda = plot_patient_modalities(sample_rec, slice_idx=None, save_path=eda_path)
    plt.close(fig_eda)

    base_loader = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])
    loaded = base_loader(sample_rec)
    img_t = loaded["image"]
    clean_t = loaded["label"]

    # Measure empirical label corruption across noise ratios
    noise_curve: Dict[str, Dict[str, float]] = {}
    for eta in [0.10, 0.20, 0.35, 0.50]:
        noisy_eta = corrupt_label_3d(
            clean_t,
            image=img_t,
            noise_type="mixed",
            noise_ratio=eta,
            noise_radius=2,
            rng=np.random.RandomState(seed),
        )
        noise_curve[f"eta_{eta:.2f}"] = compute_noise_corruption_stats(clean_t, noisy_eta)

    # Demo figure at study noise_ratio
    noisy_demo = corrupt_label_3d(
        clean_t,
        image=img_t,
        noise_type="mixed",
        noise_ratio=noise_ratio,
        noise_radius=2,
        rng=np.random.RandomState(seed),
    )
    gauss_soft = generate_soft_label_3d(clean_t, method="gaussian_boundary", sigma=1.8)
    sdf_soft = generate_soft_label_3d(clean_t, method="sdf", sigma=1.8)

    demo_path = os.path.join(output_dir, "label_noise_and_soft_labels_demo.png")
    fig_demo = visualize_noise_and_soft_labels(
        flair_3d=img_t[0].numpy(),
        clean_wt_3d=clean_t[1].numpy(),
        noisy_wt_3d=noisy_demo[1].numpy(),
        gaussian_soft_wt_3d=gauss_soft[1].numpy(),
        sdf_soft_wt_3d=sdf_soft[1].numpy(),
        save_path=demo_path,
    )
    plt.close(fig_demo)

    return {
        "sample_patient_id": sample_rec["patient_id"],
        "total_available_cases": len(records),
        "sample_label_distribution": analyze_label_distribution(sample_rec["seg"]),
        "empirical_noise_severity_curve": noise_curve,
    }


def plot_study_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    """Plot validation learning curves, held-out Test Dice bars, and Calibration ECE/Brier."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    # Panel 1: Validation Mean Dice Curve
    for idx, (preset, res) in enumerate(results.items()):
        hist = res["history"]
        label = EXPERIMENT_DISPLAY_NAMES.get(preset, preset)
        axes[0].plot(
            hist["val_epochs"],
            hist["val_dice_mean"],
            marker="o",
            lw=2.2,
            color=colors[idx % len(colors)],
            label=label,
        )
    axes[0].set_title("Validation Mean Dice Across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Clean Validation Mean Dice")
    axes[0].set_ylim([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8.5)

    # Panel 2: Held-Out Test Dice (TC, WT, ET, Mean)
    presets = list(results.keys())
    x = np.arange(len(presets))
    width = 0.20
    metrics_keys = [("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET"), ("dice_mean", "Mean")]
    bar_colors = ["#4c78a8", "#e45756", "#f58518", "#54a24b"]

    for m_i, (m_key, m_label) in enumerate(metrics_keys):
        vals = [results[p]["test_scores"].get(m_key, 0.0) for p in presets]
        axes[1].bar(x + (m_i - 1.5) * width, vals, width=width, label=m_label, color=bar_colors[m_i])

    short_labels = [p.replace("noisy_", "n_") for p in presets]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=15, fontsize=9.5)
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_title("Held-Out Test Dice (Clean Unseen GT)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Dice Score (Higher is Better)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # Panel 3: Probability Calibration Error (ECE & Brier Score)
    eces = [results[p]["test_scores"].get("ece", 0.0) for p in presets]
    briers = [results[p]["test_scores"].get("brier_score", 0.0) for p in presets]
    axes[2].bar(x - 0.16, eces, width=0.30, label="ECE (Calibration Error)", color="#e15759")
    axes[2].bar(x + 0.16, briers, width=0.30, label="Brier Score", color="#76b7b2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, rotation=15, fontsize=9.5)
    axes[2].set_title("Test Calibration & Uncertainty Error", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Error (Lower is Better)")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparative benchmark plot to: {save_path}")


def run_single_experiment(
    preset: str,
    data_dir: str,
    max_samples: int,
    epochs: int,
    val_interval: int,
    roi_size: tuple,
    lr: float,
    noise_ratio: float,
    seed: int,
    device_override: str = "",
) -> Dict[str, Any]:
    """Train, validate, and evaluate one 3D UNet experimental arm."""
    overrides: Dict[str, Any] = {
        "max_epochs": epochs,
        "val_interval": val_interval,
        "roi_size": roi_size,
        "lr": lr,
        "seed": seed,
        "batch_size": 2,
        "sw_batch_size": 2,
        "overlap": 0.25,
    }
    if device_override:
        overrides["device"] = device_override

    config = get_softlabel_study_config(
        preset=preset,
        data_dir=data_dir,
        noise_ratio=noise_ratio,
        **overrides,
    )
    set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"EXPERIMENT: {EXPERIMENT_DISPLAY_NAMES.get(preset, preset)}")
    print(
        f"  noise_type={config.noise_type} (ratio={config.noise_ratio}) | "
        f"soft_label={config.soft_label_method} | loss={config.loss_type} | device={config.device}"
    )
    print("=" * 78)

    train_loader, val_loader, test_loader, train_files, val_files, test_files = build_dataloaders(
        data_dir=config.data_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0,
        roi_size=config.roi_size,
        cache_rate=1.0,
        seed=config.seed,
        max_samples=max_samples,
        noise_type=config.noise_type,
        noise_ratio=config.noise_ratio,
        noise_radius=config.noise_radius,
        soft_label_method=config.soft_label_method,
        soft_alpha=config.soft_alpha,
        soft_sigma=config.soft_sigma,
        crop_eval_foreground=True,
    )

    model = build_model(config).to(config.device)
    summary = get_model_summary(model)
    loss_fn = build_loss_fn(
        config.loss_type,
        sce_alpha=config.sce_alpha,
        sce_beta=config.sce_beta,
        gce_q=config.gce_q,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    metric = BraTSMetric(track_calibration=True)

    trainer = BraTSTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        lr_scheduler=scheduler,
        metric=metric,
    )

    t0 = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=epochs,
        val_interval=val_interval,
    )
    elapsed_sec = round(time.time() - t0, 2)

    # Remove redundant latest_model.pth to keep disk footprint minimal
    latest_ckpt = os.path.join(config.checkpoint_dir, "latest_model.pth")
    if os.path.exists(latest_ckpt):
        os.remove(latest_ckpt)

    # Load best validation checkpoint for unbiased test evaluation
    best_ckpt = os.path.join(config.checkpoint_dir, "best_metric_model.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model, device=config.device)

    test_scores = trainer.evaluate(test_loader, name=f"Held-Out Test [{preset}]")

    # Save qualitative prediction + uncertainty map on first test patient
    model.eval()
    test_sample = next(iter(test_loader))
    t_img = test_sample["image"].to(config.device)
    t_gt = test_sample["label"].to(config.device)
    with torch.no_grad():
        t_logits = sliding_window_inference(
            inputs=t_img,
            roi_size=config.roi_size,
            sw_batch_size=config.sw_batch_size,
            predictor=model,
            overlap=config.overlap,
            mode="gaussian",
        )
        t_probs = torch.sigmoid(t_logits)

    unc_fig_path = os.path.join(config.output_dir, f"uncertainty_{preset}.png")
    fig_u = visualize_uncertainty_and_predictions(
        image=t_img[0].cpu().numpy(),
        clean_label=t_gt[0].cpu().numpy(),
        pred_probs=t_probs[0].cpu().numpy(),
        title_prefix=EXPERIMENT_DISPLAY_NAMES.get(preset, preset),
        save_path=unc_fig_path,
    )
    plt.close(fig_u)

    result = {
        "preset": preset,
        "display_name": EXPERIMENT_DISPLAY_NAMES.get(preset, preset),
        "model_summary": summary,
        "split_counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "hyperparameters": {
            "epochs": epochs,
            "roi_size": list(roi_size),
            "lr": lr,
            "noise_type": config.noise_type,
            "noise_ratio": config.noise_ratio,
            "soft_label_method": config.soft_label_method,
            "loss_type": config.loss_type,
            "device": config.device,
        },
        "best_val_dice_mean": round(float(trainer.best_metric), 4),
        "best_val_epoch": int(trainer.best_metric_epoch),
        "test_scores": test_scores,
        "training_time_sec": elapsed_sec,
        "history": history,
    }

    # Release cached dataset volumes before next arm
    del train_loader, val_loader, test_loader, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D UNet Soft Label & Label Noise Study.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to BraTS2021_Training_Data.")
    parser.add_argument("--max-samples", type=int, default=20, help="Patient cases to include (80/10/10 split).")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per experimental arm.")
    parser.add_argument("--val-interval", type=int, default=3, help="Validation interval in epochs.")
    parser.add_argument("--roi-size", type=int, default=80, help="Cubic 3D patch size (multiple of 16).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial AdamW learning rate.")
    parser.add_argument("--noise-ratio", type=float, default=0.35, help="Synthetic training noise ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="", help="Optional device override ('mps', 'cuda', 'cpu').")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["clean", "noisy_baseline", "noisy_soft_sdf", "noisy_sce"],
        help="List of study presets to execute.",
    )
    args = parser.parse_args()

    resolved_data = resolve_dataset_dir(args.data_dir)
    output_dir = str((PROJECT_ROOT / "outputs" / "softlabel_study").resolve())
    os.makedirs(output_dir, exist_ok=True)

    print(f"Resolved BraTS Dataset: {resolved_data}")
    eda_meta = generate_eda_and_noise_visuals(
        data_dir=resolved_data,
        output_dir=output_dir,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
    )

    roi_tuple = (args.roi_size, args.roi_size, args.roi_size)
    study_results: Dict[str, Dict[str, Any]] = {}

    for preset in args.presets:
        study_results[preset] = run_single_experiment(
            preset=preset,
            data_dir=resolved_data,
            max_samples=args.max_samples,
            epochs=args.epochs,
            val_interval=args.val_interval,
            roi_size=roi_tuple,
            lr=args.lr,
            noise_ratio=args.noise_ratio,
            seed=args.seed,
            device_override=args.device,
        )

    comp_plot_path = os.path.join(output_dir, "study_comparison.png")
    plot_study_comparison(study_results, save_path=comp_plot_path)

    summary_payload = {
        "eda_and_noise_analysis": eda_meta,
        "experiments": study_results,
    }
    summary_json_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "#" * 78)
    print("FINAL HELD-OUT TEST BENCHMARK SUMMARY (CLEAN UNSEEN GROUND TRUTH)")
    print("#" * 78)
    header = f"{'Experiment':<36} | {'Mean':>6} | {'WT':>6} | {'TC':>6} | {'ET':>6} | {'ECE':>7} | {'Brier':>7}"
    print(header)
    print("-" * len(header))
    for preset, res in study_results.items():
        ts = res["test_scores"]
        print(
            f"{res['display_name']:<36} | "
            f"{ts['dice_mean']:>6.4f} | "
            f"{ts['dice_wt']:>6.4f} | "
            f"{ts['dice_tc']:>6.4f} | "
            f"{ts['dice_et']:>6.4f} | "
            f"{ts.get('ece', 0.0):>7.5f} | "
            f"{ts.get('brier_score', 0.0):>7.5f}"
        )
    print(f"\nFull JSON report saved to: {summary_json_path}")


if __name__ == "__main__":
    main()

