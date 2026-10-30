"""Convert all candidate DICOM series in the manifest to NIfTI using dcm2niix.

This script scans data/processed/dicom_batch1 and data/processed/dicom_batch2,
matches the DICOM series for participants in mri_training_manifest_from_xml.csv,
runs dcm2niix to generate uncompressed 3D NIfTI files in data/processed/nifti,
and writes the updated manifest to data/metadata/mri_training_manifest.csv
with verified NIFTI_PATH and T1_CONFIRMED=yes.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "dcm2niix_win" / "dcm2niix.exe"
MANIFEST_IN = ROOT / "data" / "metadata" / "mri_training_manifest_from_xml.csv"
MANIFEST_OUT = ROOT / "data" / "metadata" / "mri_training_manifest.csv"
NIFTI_DIR = ROOT / "data" / "processed" / "nifti"


def find_dicom_directories() -> dict[tuple[str, str], Path]:
    """Index all DICOM series directories by (PTID, ImageUID) and PTID."""
    index = {}
    for batch_name in ["dicom_batch1", "dicom_batch2"]:
        batch_dir = ROOT / "data" / "processed" / batch_name / "ADNI"
        if not batch_dir.exists():
            continue
        for root, dirs, files in os.walk(batch_dir):
            if any(f.lower().endswith(".dcm") for f in files):
                p = Path(root)
                # Structure is typically .../ADNI/<PTID>/<Sequence>/<Date_Time>/<ImageUID>
                parts = p.parts
                # Find PTID which has format DDD_S_DDDD
                ptid = None
                img_uid = p.name  # e.g. I19057
                for part in parts:
                    if "_S_" in part:
                        ptid = part
                        break
                if ptid:
                    index[(ptid, img_uid)] = p
                    index[(ptid, img_uid.lstrip("I"))] = p
                    if ptid not in index:
                        index[ptid] = p
    return index


def main() -> int:
    if not CONVERTER.is_file():
        raise FileNotFoundError(f"dcm2niix executable not found at {CONVERTER}")
    if not MANIFEST_IN.is_file():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_IN}")

    NIFTI_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST_IN)
    dicom_index = find_dicom_directories()
    print(f"Loaded {len(df)} manifest rows. Found {len(dicom_index)} DICOM series indexed.")

    converted_count = 0
    skipped_count = 0
    failed_count = 0

    df["NIFTI_PATH"] = ""
    df["T1_CONFIRMED"] = ""

    for idx, row in df.iterrows():
        ptid = str(row["PTID"]).strip()
        img_id = str(row["IMAGE_DATA_ID"]).strip()
        series_dir = dicom_index.get((ptid, f"I{img_id}")) or dicom_index.get((ptid, img_id)) or dicom_index.get(ptid)

        if not series_dir or not series_dir.is_dir():
            print(f"[{idx+1}/{len(df)}] No DICOM directory found for {ptid} (Image {img_id})")
            continue

        prefix = f"{ptid}_{img_id}"
        expected_nii = NIFTI_DIR / f"{prefix}.nii"

        # Check if already converted
        matching_nii = list(NIFTI_DIR.glob(f"{prefix}*.nii"))
        if not matching_nii:
            matching_nii = list(NIFTI_DIR.glob(f"{ptid}*.nii"))

        if matching_nii and matching_nii[0].is_file():
            target_nii = matching_nii[0]
            skipped_count += 1
        else:
            # Convert series using dcm2niix
            # -z n produces uncompressed .nii for faster read performance
            cmd = [
                str(CONVERTER),
                "-z", "n",
                "-f", prefix,
                "-o", str(NIFTI_DIR),
                str(series_dir),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            matching_nii = list(NIFTI_DIR.glob(f"{prefix}*.nii"))
            if res.returncode == 0 and matching_nii:
                target_nii = matching_nii[0]
                converted_count += 1
            else:
                print(f"[{idx+1}/{len(df)}] Conversion failed for {ptid}: {res.stderr[:200]}")
                failed_count += 1
                continue

        # Verify NIfTI can be loaded with nibabel
        try:
            nib_img = nib.load(str(target_nii))
            shape = nib_img.shape
            if len(shape) < 3 or min(shape[:3]) < 20:
                print(f"[{idx+1}/{len(df)}] Suspicious image dimensions for {ptid}: {shape}")
                failed_count += 1
                continue
            # Store relative path for portability
            rel_path = target_nii.relative_to(ROOT)
            df.at[idx, "NIFTI_PATH"] = str(rel_path).replace("\\", "/")
            df.at[idx, "T1_CONFIRMED"] = "yes"
        except Exception as e:
            print(f"[{idx+1}/{len(df)}] Error loading converted NIfTI for {ptid}: {e}")
            failed_count += 1

    # Keep only rows with confirmed NIfTI
    ready = df[df["T1_CONFIRMED"].eq("yes")].copy()
    # Deduplicate PTID to ensure exactly one scan per participant
    ready = ready.drop_duplicates(subset=["PTID"], keep="first")

    print("\n--- Conversion Summary ---")
    print(f"Total manifest rows: {len(df)}")
    print(f"Successfully converted / verified: {len(ready)}")
    print(f"Newly converted: {converted_count}")
    print(f"Already existed: {skipped_count}")
    print(f"Failed / Missing: {failed_count}")
    print("\nLabel breakdown among verified scans:")
    print(ready["LABEL"].value_counts().to_string())
    print("\nSex breakdown among verified scans:")
    print(ready["SEX"].value_counts(dropna=False).to_string())

    # Save to both manifest locations
    df.to_csv(MANIFEST_IN, index=False)
    ready.to_csv(MANIFEST_OUT, index=False)
    print(f"\nSaved updated training manifest to: {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
