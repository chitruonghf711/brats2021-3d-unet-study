"""
Exploratory Data Analysis (EDA) module for BraTS 2021 dataset.
Provides functions to inspect dataset structure, file integrity,
spatial properties, intensity statistics, and 2D/3D slice visualizations.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _resolve_modality_file(patient_dir: str, patient_id: str, suffix: str) -> Optional[str]:
    """Locate either compressed (.nii.gz) or uncompressed (.nii, common on Kaggle) NIfTI file."""
    for ext in (".nii.gz", ".nii"):
        candidate = os.path.join(patient_dir, f"{patient_id}_{suffix}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def scan_brats_dataset(data_dir: str) -> List[Dict[str, str]]:
    """
    Scan the BraTS dataset directory and verify the presence of all required modalities.
    Supports both `.nii.gz` and `.nii` files (auto-unzipped on Kaggle datasets).

    Expected modalities per patient:
    - FLAIR: *_flair.nii.gz or *_flair.nii
    - T1: *_t1.nii.gz or *_t1.nii
    - T1ce: *_t1ce.nii.gz or *_t1ce.nii
    - T2: *_t2.nii.gz or *_t2.nii
    - Segmentation: *_seg.nii.gz or *_seg.nii

    Returns:
        List of dictionaries with patient paths.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    patient_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])

    data_list = []
    missing_records = []

    for patient_id in patient_dirs:
        p_path = os.path.join(data_dir, patient_id)
        flair = _resolve_modality_file(p_path, patient_id, "flair")
        t1 = _resolve_modality_file(p_path, patient_id, "t1")
        t1ce = _resolve_modality_file(p_path, patient_id, "t1ce")
        t2 = _resolve_modality_file(p_path, patient_id, "t2")
        seg = _resolve_modality_file(p_path, patient_id, "seg")

        if all(p is not None for p in (flair, t1, t1ce, t2, seg)):
            data_list.append({
                "patient_id": patient_id,
                "flair": flair,  # type: ignore[dict-item]
                "t1": t1,        # type: ignore[dict-item]
                "t1ce": t1ce,    # type: ignore[dict-item]
                "t2": t2,        # type: ignore[dict-item]
                "seg": seg,      # type: ignore[dict-item]
                "image": [flair, t1ce, t1, t2],  # type: ignore[list-item]
                "label": seg,    # type: ignore[dict-item]
            })
        else:
            missing_records.append(patient_id)

    if missing_records:
        print(f"Warning: {len(missing_records)} cases have missing modalities.")

    return data_list


def compute_volume_statistics(nii_path: str) -> Dict[str, Any]:
    """
    Compute basic statistics for a single NIfTI volume.
    """
    img = nib.load(nii_path)
    data = img.get_fdata()
    header = img.header

    nonzero_data = data[data > 0]
    return {
        "shape": list(data.shape),
        "voxel_spacing": list(header.get_zooms()[:3]),
        "min": float(data.min()),
        "max": float(data.max()),
        "nonzero_mean": float(nonzero_data.mean()) if len(nonzero_data) > 0 else 0.0,
        "nonzero_std": float(nonzero_data.std()) if len(nonzero_data) > 0 else 0.0,
    }


def analyze_label_distribution(seg_path: str) -> Dict[str, Any]:
    """
    Analyze voxel counts for BraTS segmentation labels:
    - 0: Background
    - 1: Necrotic Tumor Core (NCR)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET)

    Sub-regions:
    - Whole Tumor (WT): labels 1 + 2 + 4
    - Tumor Core (TC): labels 1 + 4
    - Enhancing Tumor (ET): label 4
    """
    seg = nib.load(seg_path).get_fdata().astype(np.int32)
    total_voxels = seg.size

    count_bg = int(np.sum(seg == 0))
    count_ncr = int(np.sum(seg == 1))
    count_ed = int(np.sum(seg == 2))
    count_et = int(np.sum(seg == 4))

    count_wt = count_ncr + count_ed + count_et
    count_tc = count_ncr + count_et

    return {
        "total_voxels": total_voxels,
        "background_voxels": count_bg,
        "ncr_voxels": count_ncr,
        "ed_voxels": count_ed,
        "et_voxels": count_et,
        "wt_voxels": count_wt,
        "tc_voxels": count_tc,
        "wt_ratio": count_wt / total_voxels if total_voxels > 0 else 0.0,
        "tc_ratio": count_tc / total_voxels if total_voxels > 0 else 0.0,
        "et_ratio": count_et / total_voxels if total_voxels > 0 else 0.0,
    }


def find_best_axial_slice(seg_data: np.ndarray) -> int:
    """Find axial slice index with the largest tumor area."""
    tumor_mask = seg_data > 0
    slice_sums = np.sum(tumor_mask, axis=(0, 1))
    best_slice = int(np.argmax(slice_sums))
    return best_slice


def plot_patient_modalities(
    patient_record: Dict[str, str],
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 4 MRI modalities and BraTS segmentation targets side-by-side.

    Args:
        patient_record: Dict containing keys flair, t1, t1ce, t2, seg, patient_id.
        slice_idx: Axial slice index. If None, picks the slice with maximum tumor voxels.
        save_path: Optional file path to save the generated figure.

    Returns:
        matplotlib Figure object.
    """
    flair = nib.load(patient_record["flair"]).get_fdata()
    t1 = nib.load(patient_record["t1"]).get_fdata()
    t1ce = nib.load(patient_record["t1ce"]).get_fdata()
    t2 = nib.load(patient_record["t2"]).get_fdata()
    seg = nib.load(patient_record["seg"]).get_fdata().astype(np.int32)

    if slice_idx is None:
        slice_idx = find_best_axial_slice(seg)

    # BraTS target sub-regions
    wt = (seg == 1) | (seg == 2) | (seg == 4)
    tc = (seg == 1) | (seg == 4)
    et = (seg == 4)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    patient_id = patient_record.get("patient_id", "Patient")
    fig.suptitle(f"BraTS 2021 EDA - Patient: {patient_id} (Axial Slice: {slice_idx})", fontsize=14, fontweight="bold")

    # Row 1: MRI modalities
    modalities = [
        ("FLAIR", flair[:, :, slice_idx]),
        ("T1", t1[:, :, slice_idx]),
        ("T1ce (Contrast Enhanced)", t1ce[:, :, slice_idx]),
        ("T2", t2[:, :, slice_idx]),
    ]

    for ax, (name, img_slice) in zip(axes[0], modalities):
        ax.imshow(np.rot90(img_slice), cmap="gray")
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    # Custom colormap for multi-class segmentation:
    # 0: transparent/black, 1: red (NCR), 2: green (ED), 4: yellow (ET)
    cmap_brats = ListedColormap(["black", "#d9534f", "#5cb85c", "white", "#f0ad4e"])

    # Row 2: Segmentation views
    seg_slice = seg[:, :, slice_idx]
    axes[1, 0].imshow(np.rot90(flair[:, :, slice_idx]), cmap="gray")
    axes[1, 0].imshow(np.rot90(seg_slice), cmap=cmap_brats, alpha=0.5, vmin=0, vmax=4)
    axes[1, 0].set_title("Ground Truth (Multiclass Overlay)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.rot90(wt[:, :, slice_idx]), cmap="Blues")
    axes[1, 1].set_title("Whole Tumor (WT: 1+2+4)", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.rot90(tc[:, :, slice_idx]), cmap="Reds")
    axes[1, 2].set_title("Tumor Core (TC: 1+4)", fontsize=12)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(np.rot90(et[:, :, slice_idx]), cmap="YlOrBr")
    axes[1, 3].set_title("Enhancing Tumor (ET: 4)", fontsize=12)
    axes[1, 3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig

