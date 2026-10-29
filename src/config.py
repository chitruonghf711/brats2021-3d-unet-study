"""
Configuration module for BraTS 2021 3D Segmentation Baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import os
import torch


def resolve_dataset_dir(user_path: Optional[str] = None) -> str:
    """
    Dynamically locate the BraTS dataset directory without hardcoding.
    Searches user path, environment variable BRATS_DATA_DIR, relative workspace
    directories, and common Google Colab paths.
    """
    search_paths = []
    if user_path:
        search_paths.append(Path(user_path))
    if os.environ.get("BRATS_DATA_DIR"):
        search_paths.append(Path(os.environ["BRATS_DATA_DIR"]))

    cwd = Path.cwd()
    search_paths.extend([
        cwd / "dataset" / "BraTS2021_Training_Data",
        cwd / "dataset",
        cwd / "data" / "BraTS2021_Training_Data",
        cwd.parent / "dataset" / "BraTS2021_Training_Data",
        Path("dataset/BraTS2021_Training_Data"),
        Path("dataset"),
        Path("/content/BraTS2021_Training_Data"),
        Path("/content/dataset/BraTS2021_Training_Data"),
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/BraTS2021_Training_Data"),
        Path("/content/drive/MyDrive/dataset/BraTS2021_Training_Data"),
    ])

    for p in search_paths:
        if p and p.exists() and p.is_dir():
            entries = [e.name for e in p.iterdir() if e.is_dir()]
            if any(e.startswith("BraTS2021_") for e in entries):
                return str(p.resolve())
            sub = p / "BraTS2021_Training_Data"
            if sub.exists() and sub.is_dir():
                sub_entries = [e.name for e in sub.iterdir() if e.is_dir()]
                if any(e.startswith("BraTS2021_") for e in sub_entries):
                    return str(sub.resolve())

    # Fallback to local relative path
    return str((cwd / "dataset" / "BraTS2021_Training_Data").resolve())


def get_default_device() -> str:
    """
    Detect the most performant device available across platforms:
    - CUDA for Windows/Linux with NVIDIA GPU
    - MPS (Metal Performance Shaders) for macOS Apple Silicon (M1/M2/M3/M4)
    - CPU as fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


@dataclass
class BaselineConfig:
    # Directory paths (resolved dynamically)
    data_dir: str = field(default_factory=resolve_dataset_dir)
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    # Dataset parameters (Patient-level 80/10/10 split)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    roi_size: Tuple[int, int, int] = (96, 96, 96)
    num_samples_per_volume: int = 2
    num_workers: int = 2
    cache_rate: float = 0.0

    # Model architecture selection ('unet' or 'nnunet' / 'dynunet')
    model_name: str = "unet"

    # Standard UNet parameters
    spatial_dims: int = 3
    in_channels: int = 4
    out_channels: int = 3
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    norm: str = "batch"
    dropout: float = 0.1

    # nnU-Net (DynUNet) specific parameters
    nnunet_filters: Tuple[int, ...] = (16, 32, 64, 128, 256)
    nnunet_strides: Tuple[Tuple[int, int, int], ...] = (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    )
    nnunet_kernel_size: Tuple[Tuple[int, int, int], ...] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    )
    nnunet_norm: str = "instance"
    nnunet_act: str = "leakyrelu"
    nnunet_res_block: bool = False
    nnunet_deep_supervision: bool = False

    # Training and optimization parameters
    optimizer_type: str = "adamw"  # 'adamw' or 'sgd'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    sgd_momentum: float = 0.99
    sgd_nesterov: bool = True
    batch_size: int = 2
    val_batch_size: int = 1
    test_batch_size: int = 1
    max_epochs: int = 100
    val_interval: int = 2
    amp: bool = True
    loss_type: str = "dice_ce"

    # Label Noise & Soft Label research parameters
    noise_type: str = "none"              # 'none', 'boundary', 'flip', 'mixed'
    noise_ratio: float = 0.0              # Corruption rate in [0.0, 1.0]
    noise_radius: int = 2                 # Morphological radius (voxels) for boundary noise
    soft_label_method: str = "none"       # 'none', 'label_smoothing', 'gaussian_boundary', 'sdf'
    soft_alpha: float = 0.1               # Smoothing factor epsilon for 'label_smoothing'
    soft_sigma: float = 1.5               # Spatial std / temperature (voxels) for 'gaussian_boundary' / 'sdf'
    sce_alpha: float = 1.0                # Forward BCE weight for Symmetric CE ('sce')
    sce_beta: float = 0.5                 # Reverse BCE weight for Symmetric CE ('sce')
    gce_q: float = 0.7                    # Box-Cox exponent q in (0, 1] for Generalized CE ('gce')
    track_calibration: bool = False       # Compute Brier score & ECE during validation/evaluation

    # Sliding window inference parameters
    sw_batch_size: int = 2
    overlap: float = 0.5

    # Device configuration (auto-detects CUDA, Apple Silicon MPS, or CPU)
    device: str = field(default_factory=get_default_device)


def get_default_config(data_dir: Optional[str] = None, **kwargs) -> BaselineConfig:
    """Return baseline configuration with dynamically resolved paths or custom overrides."""
    if data_dir is not None:
        kwargs["data_dir"] = resolve_dataset_dir(data_dir)
    return BaselineConfig(**kwargs)


def get_nnunet_config(data_dir: Optional[str] = None, **kwargs) -> BaselineConfig:
    """Return configuration tailored for the nnU-Net (DynUNet) baseline."""
    kwargs["model_name"] = "nnunet"
    if "checkpoint_dir" not in kwargs:
        kwargs["checkpoint_dir"] = "checkpoints/nnunet"
    if "output_dir" not in kwargs:
        kwargs["output_dir"] = "outputs/nnunet"
    return get_default_config(data_dir=data_dir, **kwargs)


def get_softlabel_study_config(
    preset: str = "clean",
    data_dir: Optional[str] = None,
    noise_ratio: float = 0.3,
    **kwargs,
) -> BaselineConfig:
    """
    Factory returning pre-configured 3D UNet settings for Soft Label & Label Noise experiments:
      - 'clean': Clean ground-truth labels + standard DiceCE loss (Anchor Benchmark)
      - 'noisy_baseline': Corrupted labels + standard DiceCE loss (Noise Degradation Check)
      - 'noisy_soft_sdf': Corrupted labels + SDF Soft Labels + SoftDiceBCE loss
      - 'noisy_soft_gaussian': Corrupted labels + Gaussian Boundary Soft Labels + SoftDiceBCE loss
      - 'noisy_sce': Corrupted labels + Symmetric Cross-Entropy (SCE) noise-robust loss
      - 'noisy_gce': Corrupted labels + Generalized Cross-Entropy (GCE) noise-robust loss
    """
    preset_key = preset.lower()
    presets = {
        "clean": {
            "noise_type": "none",
            "noise_ratio": 0.0,
            "soft_label_method": "none",
            "loss_type": "dice_ce",
        },
        "noisy_baseline": {
            "noise_type": "mixed",
            "noise_ratio": noise_ratio,
            "soft_label_method": "none",
            "loss_type": "dice_ce",
        },
        "noisy_soft_sdf": {
            "noise_type": "mixed",
            "noise_ratio": noise_ratio,
            "soft_label_method": "sdf",
            "soft_sigma": 1.5,
            "loss_type": "soft_dice_bce",
        },
        "noisy_soft_gaussian": {
            "noise_type": "mixed",
            "noise_ratio": noise_ratio,
            "soft_label_method": "gaussian_boundary",
            "soft_sigma": 1.5,
            "loss_type": "soft_dice_bce",
        },
        "noisy_sce": {
            "noise_type": "mixed",
            "noise_ratio": noise_ratio,
            "soft_label_method": "label_smoothing",
            "soft_alpha": 0.1,
            "loss_type": "sce",
        },
        "noisy_gce": {
            "noise_type": "mixed",
            "noise_ratio": noise_ratio,
            "soft_label_method": "none",
            "loss_type": "gce",
        },
    }
    if preset_key not in presets:
        raise ValueError(f"Unknown study preset '{preset}'. Choose from {list(presets.keys())}.")

    merged = dict(presets[preset_key])
    merged["track_calibration"] = True
    merged.setdefault("checkpoint_dir", f"checkpoints/softlabel_study/{preset_key}")
    merged.setdefault("output_dir", "outputs/softlabel_study")
    merged.update(kwargs)
    return get_default_config(data_dir=data_dir, **merged)

