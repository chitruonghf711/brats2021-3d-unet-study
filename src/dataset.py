"""
Dataset and DataLoader module for BraTS 2021 segmentation using MONAI.
Handles data splitting, 4-modality image stacking, multi-channel BraTS label conversion,
spatial cropping, intensity normalization, and augmentation.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
)

from src.eda import scan_brats_dataset


def get_brats_data_dicts(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan dataset directory and split patient records into train, validation, and test sets
    strictly at the patient level (BraTS2021_xxxxx) to avoid any data leakage.

    Default split ratios:
    - 80% Training: used to update model weights (~1,000 cases)
    - 10% Validation: used to tune hyperparameters and track best checkpoint (~125 cases)
    - 10% Test: held-out unseen set for final unbiased benchmark evaluation (~126 cases)

    Args:
        data_dir: Root path to BraTS2021_Training_Data.
        train_ratio: Fraction of patient cases for training (default: 0.8).
        val_ratio: Fraction of patient cases for validation (default: 0.1).
        test_ratio: Fraction of patient cases for held-out test (default: 0.1).
        seed: Random seed for reproducible patient-level shuffling.
        max_samples: Optional cap on total samples (for debugging/smoke testing).

    Returns:
        (train_files, val_files, test_files): Lists of dictionaries with file paths.
    """
    records = scan_brats_dataset(data_dir)
    if len(records) == 0:
        raise ValueError(f"No valid patient records found in {data_dir}")

    if max_samples is not None and max_samples < len(records):
        np.random.seed(seed)
        selected_indices = np.random.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in selected_indices]

    total_count = len(records)
    if total_count <= 2:
        return records, records, records

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(total_count)

    n_train = max(1, int(total_count * train_ratio))
    n_val = max(1, int(total_count * val_ratio))
    if n_train + n_val >= total_count:
        n_train = total_count - 2
        n_val = 1
    n_test = total_count - n_train - n_val

    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]

    train_files = [records[i] for i in train_indices]
    val_files = [records[i] for i in val_indices]
    test_files = [records[i] for i in test_indices] if len(test_indices) > 0 else val_files

    return train_files, val_files, test_files


def get_train_transforms(
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    crop_foreground: bool = True
) -> Compose:
    """
    Build MONAI transformation pipeline for training:
    - Load 4 modalities and segmentation volume.
    - Convert BraTS labels into 3 overlapping target channels:
        Channel 0: TC (Tumor Core: 1 + 4)
        Channel 1: WT (Whole Tumor: 1 + 2 + 4)
        Channel 2: ET (Enhancing Tumor: 4)
    - Foreground cropping (remove irrelevant background margins).
    - Random spatial cropping to target roi_size.
    - Data augmentation: random axis flips.
    - Channel-wise nonzero intensity normalization.
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
    ]

    if crop_foreground:
        transforms.append(CropForegroundd(keys=["image", "label"], source_key="image"))

    transforms.extend([
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    return Compose(transforms)


def get_val_transforms() -> Compose:
    """
    Build MONAI transformation pipeline for validation and evaluation:
    - Load 4 modalities and segmentation volume.
    - Convert BraTS labels into 3 overlapping target channels: TC, WT, ET.
    - Channel-wise nonzero intensity normalization.
    (Full volumes are preserved for sliding window inference).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def build_dataloaders(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 2,
    val_batch_size: int = 1,
    test_batch_size: int = 1,
    num_workers: int = 2,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    cache_rate: float = 0.0,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:
    """
    Build training, validation, and held-out test DataLoaders (80/10/10 split).

    Args:
        data_dir: Path to BraTS2021_Training_Data directory.
        train_ratio: Split ratio for training (default: 0.8).
        val_ratio: Split ratio for validation (default: 0.1).
        test_ratio: Split ratio for testing (default: 0.1).
        batch_size: Batch size for training loader.
        val_batch_size: Batch size for validation loader.
        test_batch_size: Batch size for test loader.
        num_workers: Number of DataLoader subprocesses.
        roi_size: 3D patch dimensions for training crops.
        cache_rate: Fraction of dataset to cache in RAM (0.0 to 1.0).
        seed: Random seed.
        max_samples: Optional limit on patient cases.

    Returns:
        (train_loader, val_loader, test_loader, train_files, val_files, test_files)
    """
    train_files, val_files, test_files = get_brats_data_dicts(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        max_samples=max_samples
    )

    train_transforms = get_train_transforms(roi_size=roi_size)
    eval_transforms = get_val_transforms()

    if cache_rate > 0.0:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=eval_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
        test_ds = CacheDataset(
            data=test_files,
            transform=eval_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=eval_transforms)
        test_ds = Dataset(data=test_files, transform=eval_transforms)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_files, val_files, test_files
