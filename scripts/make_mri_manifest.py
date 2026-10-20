"""Create a local MRI review sheet from the current cohort and MRI candidates.

Run from the project folder:
  py -3.14 .\scripts\make_mri_manifest.py

Fill in nifti_path and set t1_confirmed to yes only after reviewing the scan in IDA.
The resulting file contains participant-level data; keep it local.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
COHORT = META / "master_cohort_clinical_only.csv"
CANDIDATES = META / "mri_scan_candidates_local.csv"
OUT = META / "mri_training_manifest.csv"


def main() -> None:
    if not COHORT.exists() or not CANDIDATES.exists():
        raise FileNotFoundError(
            "Build the clinical cohort and run match_mri_metadata.py before creating the manifest."
        )
    cohort = pd.read_csv(COHORT, low_memory=False)
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    cohort.columns = cohort.columns.str.strip().str.upper()
    candidates.columns = candidates.columns.str.strip().str.upper()
    if "PTID" not in cohort or "PTID" not in candidates:
        raise ValueError("Both input files must contain PTID.")

    clinical_fields = [
        "PTID", "LABEL", "SEX", "AGE_YEARS_APPROX", "EDUCATION_YEARS",
        "MMSE_SCORE", "ADAS_SCORE", "OUTCOME", "BASELINE_DATE",
    ]
    missing = [col for col in clinical_fields if col not in cohort]
    if missing:
        raise ValueError(f"Clinical cohort lacks expected fields: {', '.join(missing)}")
    scan_fields = [
        "PTID", "IMAGE_DATA_ID", "VISIT", "MODALITY", "DESCRIPTION", "TYPE",
        "ACQ_DATE", "FORMAT", "DAYS_FROM_BASELINE", "ABS_DAYS_FROM_BASELINE",
    ]
    scan_fields = [col for col in scan_fields if col in candidates]
    sheet = candidates[scan_fields].merge(
        cohort[clinical_fields], on="PTID", how="inner", validate="many_to_one"
    )
    # Keep the nearest dated scan per participant as a review starting point.
    if "ABS_DAYS_FROM_BASELINE" in sheet:
        sheet = sheet.sort_values("ABS_DAYS_FROM_BASELINE").drop_duplicates("PTID")
    sheet["NIFTI_PATH"] = ""
    sheet["T1_CONFIRMED"] = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(OUT, index=False)
    print(f"Created local MRI review manifest with {len(sheet)} candidate participants:")
    print(OUT)
    print("Fill NIFTI_PATH and set T1_CONFIRMED=yes only for a visually/metadata-verified T1 scan.")


if __name__ == "__main__":
    main()
