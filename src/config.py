"""
Configuration module for BraTS 2021 3D Segmentation Baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import os
import subprocess
import tarfile
import torch


def _dir_has_brats_cases(p: Path) -> Optional[str]:
    """Return resolved path string if `p` or `p/BraTS2021_Training_Data` contains BraTS2021_* folders."""
    if not p or not p.exists() or not p.is_dir():
        return None
    try:
        entries = [e.name for e in p.iterdir() if e.is_dir()]
    except OSError:
        return None
    if any(e.startswith("BraTS2021_") for e in entries):
        return str(p.resolve())
    sub = p / "BraTS2021_Training_Data"
    if sub.exists() and sub.is_dir():
        try:
            sub_entries = [e.name for e in sub.iterdir() if e.is_dir()]
        except OSError:
            return None
        if any(e.startswith("BraTS2021_") for e in sub_entries):
            return str(sub.resolve())
    return None


def resolve_dataset_dir(user_path: Optional[str] = None) -> str:
    """
    Dynamically locate the BraTS dataset directory without hardcoding.
    Searches user path, environment variable BRATS_DATA_DIR, relative workspace
    directories, Kaggle (/kaggle/tmp, /kaggle/input), and Google Colab paths.
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
        Path("/kaggle/tmp/BraTS2021_Training_Data"),
        Path("/tmp/BraTS2021_Training_Data"),
        Path("/kaggle/input/brats-2021-task1/BraTS2021_Training_Data"),
        Path("/kaggle/input/brats-2021-task1"),
        Path("/kaggle/input/brats2021-training-data/BraTS2021_Training_Data"),
        Path("/kaggle/input/brats2021-training-data"),
        Path("/kaggle/working/dataset/BraTS2021_Training_Data"),
        Path("/content/BraTS2021_Training_Data"),
        Path("/content/dataset/BraTS2021_Training_Data"),
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/BraTS2021_Training_Data"),
        Path("/content/drive/MyDrive/dataset/BraTS2021_Training_Data"),
    ])

    # Auto-discover any attached BraTS dataset inside /kaggle/input (supports both classic and nested mounts)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and kaggle_input.is_dir():
        for root, dirs, _ in os.walk(str(kaggle_input)):
            depth = Path(root).relative_to(kaggle_input).parts
            if any(d.startswith("BraTS2021_") for d in dirs):
                search_paths.append(Path(root))
                dirs[:] = []  # Do not descend inside 1,251 individual patient folders
            elif len(depth) >= 4:
                dirs[:] = []

    for p in search_paths:
        found = _dir_has_brats_cases(p)
        if found:
            return found

    # Fallback to local relative path
    return str((cwd / "dataset" / "BraTS2021_Training_Data").resolve())


def ensure_kaggle_brats_extracted(user_path: Optional[str] = None) -> str:
    """
    If running on Kaggle where `dschettler8845/brats-2021-task1` provides
    `BraTS2021_Training_Data.tar` inside `/kaggle/input`, automatically extract it
    into ephemeral scratch `/kaggle/tmp/BraTS2021_Training_Data` (preserving 100% of
    `/kaggle/working` 20 GB output quota) and apply single-case replacement tars if present.
    """
    resolved = resolve_dataset_dir(user_path)
    if os.path.isdir(resolved) and _dir_has_brats_cases(Path(resolved)):
        return resolved

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return resolved

    tar_candidates = list(kaggle_input.rglob("BraTS2021_Training_Data.tar"))
    if not tar_candidates:
        return resolved

    main_tar = tar_candidates[0]
    scratch_root = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path("/tmp")
    target_dir = scratch_root / "BraTS2021_Training_Data"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Kaggle Setup] Extracting {main_tar} -> {target_dir} (ephemeral SSD, 0 MB in /kaggle/working)...")
    subprocess.run(["tar", "-xf", str(main_tar), "-C", str(target_dir)], check=True)

    resolved_after = _dir_has_brats_cases(target_dir) or str(target_dir)
    # Apply known fixed patient tars (e.g., BraTS2021_00495.tar, BraTS2021_00621.tar) if alongside main_tar
    for patch_tar in sorted(main_tar.parent.glob("BraTS2021_*.tar")):
        if patch_tar.name == "BraTS2021_Training_Data.tar":
            continue
        case_id = patch_tar.stem  # e.g. BraTS2021_00495
        try:
            with tarfile.open(patch_tar, "r") as tf:
                names = [n.lstrip("./") for n in tf.getnames() if n.strip("./")]
            has_case_dir = any(n == case_id or n.startswith(f"{case_id}/") for n in names)
        except Exception:
            has_case_dir = False
        extract_dest = Path(resolved_after) if has_case_dir else (Path(resolved_after) / case_id)
        extract_dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(patch_tar), "-C", str(extract_dest)], check=False)

    print(f"[Kaggle Setup] Extraction complete: {resolved_after}")
    return resolved_after


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
    max_wall_hours: Optional[float] = None  # Safe wall-clock training cutoff (e.g. 10.5h on Kaggle 12h sessions)
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

    # Official BraTS 2021 Metric Setup parameters
    metric_threshold: float = 0.5         # Independent sigmoid binarization threshold per channel
    compute_hd95: bool = False            # Compute 95% Hausdorff Distance (mm) per sub-region
    compute_confusion_metrics: bool = False  # Compute IoU, Sensitivity, and Precision per sub-region
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Isotropic SRI24 spacing (mm)
    hd95_empty_penalty: float = 373.13    # Diagonal of 240x240x155 mm^3 volume for single-empty HD95

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


def get_unet_baseline_config(data_dir: Optional[str] = None, **kwargs) -> BaselineConfig:
    """
    Return clean, dedicated configuration strictly for the pure 3D UNet Baseline on BraTS 2021:
      - Zero synthetic label noise ('none', ratio=0.0) and zero soft-label distortion ('none')
      - Pure FP32 training by default ('amp': False) to guarantee numerical stability on BatchNorm3d
      - Full BraTS 2021 Metric Setup enabled (Dice + HD95 + IoU + Sensitivity + Precision)
      - Isolated output and checkpoint directories ('checkpoints/unet_baseline', 'outputs/unet_baseline')
    """
    defaults = {
        "model_name": "unet",
        "amp": False,
        "noise_type": "none",
        "noise_ratio": 0.0,
        "soft_label_method": "none",
        "loss_type": "dice_bce",
        "compute_hd95": True,
        "compute_confusion_metrics": True,
        "track_calibration": False,
        "metric_threshold": 0.5,
        "voxel_spacing": (1.0, 1.0, 1.0),
        "hd95_empty_penalty": 373.13,
        "checkpoint_dir": "checkpoints/unet_baseline",
        "output_dir": "outputs/unet_baseline",
    }
    defaults.update(kwargs)
    return get_default_config(data_dir=data_dir, **defaults)


def get_nnunet_config(data_dir: Optional[str] = None, **kwargs) -> BaselineConfig:
    """Return configuration tailored for the nnU-Net (DynUNet) baseline."""
    kwargs["model_name"] = "nnunet"
    if "checkpoint_dir" not in kwargs:
        kwargs["checkpoint_dir"] = "checkpoints/nnunet"
    if "output_dir" not in kwargs:
        kwargs["output_dir"] = "outputs/nnunet"
    return get_default_config(data_dir=data_dir, **kwargs)

