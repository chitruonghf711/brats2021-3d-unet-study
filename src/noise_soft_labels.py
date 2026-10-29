"""
Synthetic Label Noise and Soft Label generation module for 3D Brain Tumor Segmentation.

Supports:
1. Controllable 3D Label Noise simulation (applied to training annotations only):
   - 'none': Clean ground-truth annotations.
   - 'boundary': Smooth 3D morphological dilation/erosion perturbations simulating
     inter-observer boundary over/under-segmentation.
   - 'flip': Random 3D ellipsoidal false-positive lesions and false-negative dropouts.
   - 'mixed': Combined boundary ambiguity and region-level annotation errors.
   Deterministic per-patient seeding ensures static noisy annotations across epochs
   (allowing investigation of noise memorization vs. robustness) without duplicating NIfTI files.

2. Continuous Soft Label generation (target values in [0, 1]):
   - 'none': Hard binary masks {0, 1}.
   - 'label_smoothing': Uniform confidence calibration (1 - alpha) * y + 0.5 * alpha.
   - 'gaussian_boundary': Separable 3D Gaussian boundary softening.
   - 'sdf': Euclidean Signed Distance Function sigmoid softening, modeling spatially
     graded uncertainty around tumor margins.
"""

import hashlib
import math
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.transforms import MapTransform


VALID_NOISE_TYPES = ("none", "boundary", "flip", "mixed")
VALID_SOFT_METHODS = ("none", "label_smoothing", "gaussian_boundary", "sdf")


def _stable_patient_seed(base_seed: int, patient_id: str) -> int:
    """Derive a cross-process deterministic 31-bit seed from base_seed and patient_id."""
    digest = hashlib.md5(f"{base_seed}:{patient_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _copy_like(template: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Preserve MONAI MetaTensor metadata if template is a MetaTensor."""
    out = template.clone()
    out.copy_(data.to(dtype=template.dtype, device=template.device))
    return out


def _as_plain_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract underlying torch.Tensor from MONAI MetaTensor to avoid dispatch overhead."""
    return t.as_tensor() if hasattr(t, "as_tensor") else t


def _morph_dilate_3d(mask_5d: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact separable 3D cubic morphological dilation in O(r) instead of O(r^3) per voxel."""
    if radius <= 0:
        return mask_5d
    k = 2 * radius + 1
    out = F.max_pool3d(mask_5d, kernel_size=(k, 1, 1), stride=1, padding=(radius, 0, 0))
    out = F.max_pool3d(out, kernel_size=(1, k, 1), stride=1, padding=(0, radius, 0))
    out = F.max_pool3d(out, kernel_size=(1, 1, k), stride=1, padding=(0, 0, radius))
    return out


def _morph_erode_3d(mask_5d: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact separable 3D cubic morphological erosion in O(r) instead of O(r^3) per voxel."""
    if radius <= 0:
        return mask_5d
    return -_morph_dilate_3d(-mask_5d, radius)


def _smooth_binary_gate_3d(
    spatial_shape: Tuple[int, int, int],
    active_fraction: float,
    rng: np.random.RandomState,
    coarse_stride: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a smooth 3D spatial indicator mask where approximately `active_fraction`
    of voxels evaluate to 1.0, producing coherent anatomical bulges/indentations
    rather than unrealistic salt-and-pepper single-voxel noise.
    """
    if active_fraction <= 0.0:
        return torch.zeros(spatial_shape, dtype=torch.float32, device=device)
    if active_fraction >= 1.0:
        return torch.ones(spatial_shape, dtype=torch.float32, device=device)

    h, w, d = spatial_shape
    gh = max(2, h // coarse_stride)
    gw = max(2, w // coarse_stride)
    gd = max(2, d // coarse_stride)

    coarse = torch.from_numpy(
        rng.randn(1, 1, gh, gw, gd).astype(np.float32)
    ).to(device)
    smooth = F.interpolate(
        coarse, size=spatial_shape, mode="trilinear", align_corners=False
    )[0, 0]
    flat = smooth.reshape(-1)
    if flat.numel() > 8192:
        step = max(1, flat.numel() // 8192)
        flat = flat[::step]
    threshold = torch.quantile(flat, float(1.0 - active_fraction))
    return (smooth >= threshold).float()


def _enforce_brats_hierarchy(mask: torch.Tensor) -> torch.Tensor:
    """
    Enforce BraTS sub-region nesting on 3-channel binary tensor:
      Channel 0 = TC (1+4), Channel 1 = WT (1+2+4), Channel 2 = ET (4)
      => ET (ch 2) <= TC (ch 0) <= WT (ch 1).
    """
    if mask.shape[0] < 3:
        return mask
    out = mask.clone()
    out[0] = torch.maximum(out[0], out[2])
    out[1] = torch.maximum(out[1], out[0])
    return out


def corrupt_label_3d(
    label: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    noise_type: str = "none",
    noise_ratio: float = 0.0,
    noise_radius: int = 2,
    rng: Optional[np.random.RandomState] = None,
    enforce_hierarchy: bool = True,
) -> torch.Tensor:
    """
    Inject controllable 3D synthetic label noise into multi-channel binary BraTS masks.

    Args:
        label: Binary target tensor of shape (C, H, W, D), values in {0, 1}.
        image: Optional 4-modality image tensor (4, H, W, D) to constrain false-positive
               blobs within the non-zero brain foreground.
        noise_type: One of 'none', 'boundary', 'flip', 'mixed'.
        noise_ratio: Corruption intensity in [0.0, 1.0].
        noise_radius: Base morphological radius (in voxels) for boundary corruption.
        rng: Optional NumPy RandomState for deterministic per-patient corruption.
        enforce_hierarchy: Whether to maintain ET <= TC <= WT consistency.

    Returns:
        Corrupted binary tensor of shape (C, H, W, D) in {0.0, 1.0}.
    """
    ntype = noise_type.lower()
    if ntype not in VALID_NOISE_TYPES:
        raise ValueError(f"Unsupported noise_type '{noise_type}'. Choose from {VALID_NOISE_TYPES}.")

    if ntype == "none" or noise_ratio <= 0.0:
        return label

    ratio = float(np.clip(noise_ratio, 0.0, 1.0))
    if rng is None:
        rng = np.random.RandomState()

    device = label.device
    raw_label = _as_plain_tensor(label)
    raw_image = _as_plain_tensor(image) if image is not None else None
    work = (raw_label.detach().float() > 0.5).float()
    c_dim, h, w, d = work.shape
    spatial_shape = (h, w, d)

    # 1. Smooth 3D morphological boundary noise (over- & under-segmentation)
    if ntype in ("boundary", "mixed"):
        eff_radius = max(1, int(round(noise_radius * (0.75 + 1.5 * ratio))))
        for c in range(c_dim):
            ch_5d = work[c : c + 1].unsqueeze(0)
            if ch_5d.sum() == 0:
                continue
            dilated = _morph_dilate_3d(ch_5d, eff_radius)[0, 0]
            eroded = _morph_erode_3d(ch_5d, eff_radius)[0, 0]

            outer_band = (dilated > 0.5) & (work[c] <= 0.5)
            inner_band = (work[c] > 0.5) & (eroded <= 0.5)

            gate_out = _smooth_binary_gate_3d(spatial_shape, ratio, rng, coarse_stride=8, device=device)
            gate_in = _smooth_binary_gate_3d(spatial_shape, ratio, rng, coarse_stride=8, device=device)

            added = outer_band.float() * gate_out
            removed = inner_band.float() * gate_in
            work[c] = torch.clamp(work[c] + added - removed, 0.0, 1.0)

    # 2. Region-level false-positive blobs & false-negative dropouts
    if ntype in ("flip", "mixed"):
        if raw_image is not None and raw_image.shape[-3:] == spatial_shape:
            brain_fg = (raw_image[0].detach().abs() > 1e-5).to(device)
        else:
            brain_fg = torch.ones(spatial_shape, dtype=torch.bool, device=device)

        fg_coords = torch.nonzero(brain_fg, as_tuple=False)
        num_blobs = max(1, int(round(8 * ratio)))

        if len(fg_coords) > 0:
            zz = torch.arange(h, device=device, dtype=torch.float32).view(h, 1, 1)
            yy = torch.arange(w, device=device, dtype=torch.float32).view(1, w, 1)
            xx = torch.arange(d, device=device, dtype=torch.float32).view(1, 1, d)

            # Inject false-positive 3D ellipsoids in brain foreground
            for _ in range(num_blobs):
                pick = int(rng.randint(0, len(fg_coords)))
                cz, cy, cx = [float(v) for v in fg_coords[pick]]
                rz = float(rng.uniform(3.0, 6.0 + 10.0 * ratio))
                ry = float(rng.uniform(3.0, 6.0 + 10.0 * ratio))
                rx = float(rng.uniform(3.0, 6.0 + 10.0 * ratio))
                ellipsoid = (
                    ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
                ) <= 1.0
                blob = (ellipsoid & brain_fg).float()

                target_ch = int(rng.randint(0, c_dim))
                work[target_ch] = torch.maximum(work[target_ch], blob)

            # Carve out false-negative ellipsoids inside existing tumor foreground
            for c in range(c_dim):
                tumor_coords = torch.nonzero(work[c] > 0.5, as_tuple=False)
                if len(tumor_coords) == 0:
                    continue
                n_drop = max(1, int(round(4 * ratio)))
                for _ in range(n_drop):
                    if rng.rand() < ratio:
                        pick = int(rng.randint(0, len(tumor_coords)))
                        cz, cy, cx = [float(v) for v in tumor_coords[pick]]
                        rz = float(rng.uniform(3.0, 5.0 + 8.0 * ratio))
                        ry = float(rng.uniform(3.0, 5.0 + 8.0 * ratio))
                        rx = float(rng.uniform(3.0, 5.0 + 8.0 * ratio))
                        hole = (
                            ((zz - cz) / rz) ** 2
                            + ((yy - cy) / ry) ** 2
                            + ((xx - cx) / rx) ** 2
                        ) <= 1.0
                        work[c] = work[c] * (~hole).float()

    if enforce_hierarchy:
        work = _enforce_brats_hierarchy(work)

    return _copy_like(label, work)


def _gaussian_blur_3d_channels(tensor_4d: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply fast separable 3D Gaussian smoothing per channel on (C, H, W, D) tensor."""
    if sigma <= 0.0:
        return tensor_4d

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor_4d.device)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    c_dim = tensor_4d.shape[0]
    x = tensor_4d.unsqueeze(0)  # (1, C, H, W, D)

    k_h = kernel_1d.view(1, 1, -1, 1, 1).expand(c_dim, 1, -1, 1, 1)
    k_w = kernel_1d.view(1, 1, 1, -1, 1).expand(c_dim, 1, 1, -1, 1)
    k_d = kernel_1d.view(1, 1, 1, 1, -1).expand(c_dim, 1, 1, 1, -1)

    x = F.pad(x, (0, 0, 0, 0, radius, radius), mode="replicate")
    x = F.conv3d(x, k_h, groups=c_dim)

    x = F.pad(x, (0, 0, radius, radius, 0, 0), mode="replicate")
    x = F.conv3d(x, k_w, groups=c_dim)

    x = F.pad(x, (radius, radius, 0, 0, 0, 0), mode="replicate")
    x = F.conv3d(x, k_d, groups=c_dim)

    return x.squeeze(0)


def generate_soft_label_3d(
    label: torch.Tensor,
    method: str = "none",
    alpha: float = 0.1,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Convert binary segmentation masks into continuous soft probability targets in [0, 1].

    Args:
        label: Binary or soft mask tensor of shape (C, H, W, D).
        method: One of 'none', 'label_smoothing', 'gaussian_boundary', 'sdf'.
        alpha: Smoothing factor epsilon in [0, 0.5) for 'label_smoothing'.
        sigma: Spatial standard deviation / temperature (in voxels) for
               'gaussian_boundary' or 'sdf'.

    Returns:
        Continuous soft target tensor of shape (C, H, W, D) with values in [0.0, 1.0].
    """
    m = method.lower()
    if m not in VALID_SOFT_METHODS:
        raise ValueError(f"Unsupported soft_label_method '{method}'. Choose from {VALID_SOFT_METHODS}.")

    if m == "none":
        return label

    work = _as_plain_tensor(label).detach().float()

    if m == "label_smoothing":
        eps = float(np.clip(alpha, 0.0, 0.49))
        soft = (1.0 - eps) * work + 0.5 * eps
        return _copy_like(label, torch.clamp(soft, 0.0, 1.0))

    if m == "gaussian_boundary":
        soft = _gaussian_blur_3d_channels(work, sigma=max(float(sigma), 1e-3))
        return _copy_like(label, torch.clamp(soft, 0.0, 1.0))

    if m == "sdf":
        tau = max(float(sigma), 1e-3)
        np_mask = (work.cpu().numpy() > 0.5)
        soft_np = np.zeros_like(np_mask, dtype=np.float32)

        for c in range(np_mask.shape[0]):
            ch = np_mask[c]
            if not np.any(ch):
                soft_np[c] = 0.0
            elif np.all(ch):
                soft_np[c] = 1.0
            else:
                dist_in = ndi.distance_transform_edt(ch)
                dist_out = ndi.distance_transform_edt(~ch)
                sdf = dist_in - dist_out
                logits = np.clip(sdf / tau, -20.0, 20.0)
                soft_np[c] = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        soft_tensor = torch.from_numpy(soft_np).to(device=label.device, dtype=torch.float32)
        return _copy_like(label, torch.clamp(soft_tensor, 0.0, 1.0))

    return label


def compute_noise_corruption_stats(
    clean_label: torch.Tensor,
    noisy_label: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure empirical disagreement (Dice overlap & voxel flip rate) between
    clean ground-truth masks and corrupted noisy masks of shape (3, H, W, D).
    """
    clean_b = (clean_label.detach().float() > 0.5).float()
    noisy_b = (noisy_label.detach().float() > 0.5).float()

    eps = 1e-5
    channel_names = ["tc", "wt", "et"]
    stats: Dict[str, float] = {}
    dices = []

    for c, name in enumerate(channel_names[: clean_b.shape[0]]):
        inter = float((clean_b[c] * noisy_b[c]).sum().item())
        denom = float((clean_b[c].sum() + noisy_b[c].sum()).item())
        dice_c = (2.0 * inter + eps) / (denom + eps)
        stats[f"label_dice_{name}"] = round(dice_c, 4)
        dices.append(dice_c)

    stats["label_dice_mean"] = round(float(np.mean(dices)), 4) if dices else 1.0
    stats["voxel_disagreement_rate"] = round(
        float((clean_b != noisy_b).float().mean().item()), 5
    )
    return stats


class AddSyntheticLabelNoised(MapTransform):
    """
    MONAI dictionary transform injecting static per-patient synthetic 3D label noise.
    """

    def __init__(
        self,
        keys: KeysCollection = "label",
        image_key: Optional[str] = "image",
        noise_type: str = "none",
        noise_ratio: float = 0.0,
        noise_radius: int = 2,
        seed: int = 42,
        deterministic_per_patient: bool = True,
        keep_clean_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.noise_radius = noise_radius
        self.seed = seed
        self.deterministic_per_patient = deterministic_per_patient
        self.keep_clean_key = keep_clean_key
        self._call_counter = 0

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        if self.noise_type.lower() == "none" or self.noise_ratio <= 0.0:
            if self.keep_clean_key is not None and "label" in d:
                d[self.keep_clean_key] = d["label"].clone()
            return d

        patient_id = str(d.get("patient_id", ""))
        if self.deterministic_per_patient and patient_id:
            sample_seed = _stable_patient_seed(self.seed, patient_id)
        else:
            sample_seed = (self.seed + self._call_counter) & 0x7FFFFFFF
            self._call_counter += 1

        rng = np.random.RandomState(sample_seed)
        img = d.get(self.image_key) if self.image_key else None

        for key in self.key_iterator(d):
            if self.keep_clean_key is not None:
                d[self.keep_clean_key] = d[key].clone()
            d[key] = corrupt_label_3d(
                label=d[key],
                image=img,
                noise_type=self.noise_type,
                noise_ratio=self.noise_ratio,
                noise_radius=self.noise_radius,
                rng=rng,
            )
        return d


class ConvertHardToSoftLabeld(MapTransform):
    """
    MONAI dictionary transform converting binary masks into continuous soft labels in [0, 1].
    """

    def __init__(
        self,
        keys: KeysCollection = "label",
        method: str = "none",
        alpha: float = 0.1,
        sigma: float = 1.5,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.method = method
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        if self.method.lower() == "none":
            return d

        for key in self.key_iterator(d):
            d[key] = generate_soft_label_3d(
                label=d[key],
                method=self.method,
                alpha=self.alpha,
                sigma=self.sigma,
            )
        return d
