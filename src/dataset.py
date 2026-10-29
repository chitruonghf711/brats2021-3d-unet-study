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
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan dataset directory and split patient records into train and validation sets.

    Args:
        data_dir: Root path to BraTS2021_Training_Data.
        train_ratio: Fraction of cases used for training (default: 0.8).
        seed: Random seed for reproducibility.
        max_samples: Optional cap on total samples (useful for rapid debugging/smoke testing).

    Returns:
        (train_files, val_files): Lists of dictionaries with 'image' and 'label' paths.
    """
    records = scan_brats_dataset(data_dir)
    if len(records) == 0:
        raise ValueError(f"No valid patient records found in {data_dir}")

    if max_samples is not None and max_samples < len(records):
        np.random.seed(seed)
        selected_indices = np.random.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in selected_indices]

    if len(records) <= 1:
        return records, records

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(records))
    split_idx = max(1, int(len(records) * train_ratio))

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    train_files = [records[i] for i in train_indices]
    val_files = [records[i] for i in val_indices] if len(val_indices) > 0 else train_files

    return train_files, val_files


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
    batch_size: int = 2,
    val_batch_size: int = 1,
    num_workers: int = 2,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    cache_rate: float = 0.0,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build training and validation DataLoaders.

    Args:
        data_dir: Path to BraTS2021_Training_Data directory.
        train_ratio: Split ratio for training.
        batch_size: Batch size for training loader.
        val_batch_size: Batch size for validation loader.
        num_workers: Number of DataLoader subprocesses.
        roi_size: 3D patch dimensions for training crops.
        cache_rate: Fraction of dataset to cache in RAM (0.0 to 1.0).
        seed: Random seed.
        max_samples: Optional limit on patient cases.

    Returns:
        (train_loader, val_loader, train_files, val_files)
    """
    train_files, val_files = get_brats_data_dicts(
        data_dir=data_dir,
        train_ratio=train_ratio,
        seed=seed,
        max_samples=max_samples
    )

    train_transforms = get_train_transforms(roi_size=roi_size)
    val_transforms = get_val_transforms()

    if cache_rate > 0.0:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_files, val_files
