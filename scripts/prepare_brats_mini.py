"""
Prepare a lightweight "BraTS-Mini" subset for rapid 3D UNet prototyping
on Soft Label & Label Noise research without downloading the full ~13 GB archive.

Features:
1. Streaming mode ('stream'): Streams exact N real BraTS 2021 patient volumes
   (~9.5 MB/patient, 4 MRI modalities + ground-truth segmentation) directly over HTTP
   from a public BraTS 2021 tar archive and closes the connection immediately once N
   complete cases are extracted.
2. Synthetic mode ('synthetic'): Generates anatomically structured 3D multi-modal NIfTI
   volumes (nested Whole Tumor -> Tumor Core -> Enhancing Tumor) for offline testing.
3. Disk-space guard: Refuses to fill disk below a configurable safety reserve.
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HF_TAR_URL = (
    "https://huggingface.co/datasets/Su700/Brats2021/resolve/main/BraTS2021_Training_Data.tar"
)
REQUIRED_MODALITIES = {"flair", "t1", "t1ce", "t2", "seg"}
MEMBER_PATTERN = re.compile(
    r"^(?:\./)?(BraTS2021_\d+)/(BraTS2021_\d+_(flair|t1|t1ce|t2|seg)\.nii\.gz)$"
)


def find_existing_complete_cases(output_dir: Path) -> Set[str]:
    """Return set of patient_ids in output_dir that already have all 5 NIfTI files."""
    complete: Set[str] = set()
    if not output_dir.exists():
        return complete

    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        pid = entry.name
        files_ok = all(
            (entry / f"{pid}_{mod}.nii.gz").is_file()
            and (entry / f"{pid}_{mod}.nii.gz").stat().st_size > 0
            for mod in REQUIRED_MODALITIES
        )
        if files_ok:
            complete.add(pid)
    return complete


def check_disk_space(target_dir: Path, estimated_mb: float, min_free_gb: float = 0.8) -> None:
    """Verify sufficient free disk space before writing dataset files."""
    check_path = target_dir if target_dir.exists() else target_dir.parent
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent

    usage = shutil.disk_usage(check_path)
    free_gb = usage.free / (1024 ** 3)
    projected_free_gb = (usage.free - estimated_mb * (1024 ** 2)) / (1024 ** 3)

    print(
        f"[Disk Check] Current free space: {free_gb:.2f} GB | "
        f"Estimated download: {estimated_mb:.1f} MB | "
        f"Projected free space: {projected_free_gb:.2f} GB"
    )
    if projected_free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space: projected free space ({projected_free_gb:.2f} GB) "
            f"would fall below safety threshold ({min_free_gb:.2f} GB). "
            f"Reduce --num-cases or free up disk space."
        )


def stream_brats_mini(
    output_dir: Path,
    num_cases: int = 20,
    tar_url: str = DEFAULT_HF_TAR_URL,
    min_free_gb: float = 0.8,
) -> int:
    """
    Stream exact `num_cases` complete patient folders from remote uncompressed .tar archive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = find_existing_complete_cases(output_dir)

    if len(completed) >= num_cases:
        print(f"[BraTS-Mini] Already found {len(completed)} complete cases in {output_dir}.")
        return len(completed)

    needed = num_cases - len(completed)
    check_disk_space(output_dir, estimated_mb=needed * 11.0, min_free_gb=min_free_gb)

    print(f"[BraTS-Mini] Streaming {needed} real BraTS 2021 patient cases from:")
    print(f"  {tar_url}")
    print(f"  Target directory: {output_dir}")

    patient_mods: Dict[str, Set[str]] = {pid: set(REQUIRED_MODALITIES) for pid in completed}

    req = urllib.request.Request(tar_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with tarfile.open(fileobj=resp, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                match = MEMBER_PATTERN.match(member.name)
                if not match:
                    continue

                patient_id, filename, modality = match.group(1), match.group(2), match.group(3)
                if patient_id in completed:
                    continue

                patient_dir = output_dir / patient_id
                patient_dir.mkdir(parents=True, exist_ok=True)
                dest_file = patient_dir / os.path.basename(filename)

                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                tmp_file = dest_file.with_suffix(".tmp")
                with open(tmp_file, "wb") as out_f:
                    shutil.copyfileobj(extracted, out_f, length=1024 * 256)
                tmp_file.replace(dest_file)

                mods = patient_mods.setdefault(patient_id, set())
                mods.add(modality)

                if mods == REQUIRED_MODALITIES:
                    completed.add(patient_id)
                    print(
                        f"  [{len(completed):02d}/{num_cases:02d}] Downloaded complete case: "
                        f"{patient_id}"
                    )
                    if len(completed) >= num_cases:
                        break

    # Clean up any partially downloaded patient folder if stream stopped mid-patient
    for entry in output_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in completed:
            shutil.rmtree(entry, ignore_errors=True)

    print(f"[BraTS-Mini] Successfully prepared {len(completed)} complete BraTS 2021 cases.")
    return len(completed)


def generate_synthetic_brats_mini(
    output_dir: Path,
    num_cases: int = 20,
    shape: tuple = (128, 128, 96),
    seed: int = 42,
) -> int:
    """
    Generate structured synthetic 3D NIfTI volumes mimicking BraTS multi-modal MRI
    and nested tumor sub-regions (ED=2, NCR=1, ET=4) for offline environments.
    """
    import nibabel as nib
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    h, w, d = shape
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        np.linspace(-1, 1, d, dtype=np.float32),
        indexing="ij",
    )

    brain_mask = (zz ** 2 / 0.72 + yy ** 2 / 0.65 + xx ** 2 / 0.70) <= 1.0
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    for idx in range(num_cases):
        pid = f"BraTS2021_{idx:05d}"
        p_dir = output_dir / pid
        p_dir.mkdir(parents=True, exist_ok=True)

        # Random tumor center inside brain
        cz = rng.uniform(-0.30, 0.30)
        cy = rng.uniform(-0.30, 0.30)
        cx = rng.uniform(-0.25, 0.25)
        r_wt = rng.uniform(0.22, 0.34)
        r_tc = r_wt * rng.uniform(0.55, 0.72)
        r_ncr = r_tc * rng.uniform(0.45, 0.65)

        dist2 = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        wt_mask = (dist2 <= r_wt ** 2) & brain_mask
        tc_mask = (dist2 <= r_tc ** 2) & brain_mask
        ncr_mask = (dist2 <= r_ncr ** 2) & brain_mask
        et_mask = tc_mask & (~ncr_mask)
        ed_mask = wt_mask & (~tc_mask)

        seg = np.zeros(shape, dtype=np.uint8)
        seg[ed_mask] = 2
        seg[ncr_mask] = 1
        seg[et_mask] = 4

        # Synthesize correlated 4-modality MRI signals
        base_tissue = brain_mask.astype(np.float32) * (
            100.0 + 15.0 * np.sin(3.0 * zz) * np.cos(3.0 * yy)
        )
        flair = base_tissue + ed_mask * 95.0 + tc_mask * 60.0
        t1 = base_tissue * 1.1 - ed_mask * 25.0 - ncr_mask * 45.0
        t1ce = t1 + et_mask * 130.0
        t2 = base_tissue * 0.9 + ed_mask * 110.0 + ncr_mask * 80.0 + et_mask * 45.0

        for mod_name, vol in [("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2)]:
            noisy_vol = np.where(
                brain_mask,
                np.clip(vol + rng.normal(0.0, 6.0, size=shape).astype(np.float32), 1.0, None),
                0.0,
            ).astype(np.float32)
            nib.save(nib.Nifti1Image(noisy_vol, affine), str(p_dir / f"{pid}_{mod_name}.nii.gz"))

        nib.save(nib.Nifti1Image(seg, affine), str(p_dir / f"{pid}_seg.nii.gz"))

    print(f"[BraTS-Mini] Generated {num_cases} synthetic BraTS cases in {output_dir}.")
    return num_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS-Mini 3D dataset subset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "BraTS2021_Training_Data"),
        help="Destination directory for BraTS2021_xxxxx patient folders.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=20,
        help="Number of patient cases to download or synthesize (default: 20).",
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "synthetic"],
        default="stream",
        help="'stream' real BraTS 2021 subset over HTTP, or generate 'synthetic' volumes.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.8,
        help="Minimum free disk space (GB) to preserve on the volume.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    if args.mode == "stream":
        stream_brats_mini(
            output_dir=out_path,
            num_cases=args.num_cases,
            min_free_gb=args.min_free_gb,
        )
    else:
        generate_synthetic_brats_mini(output_dir=out_path, num_cases=args.num_cases)


if __name__ == "__main__":
    main()

