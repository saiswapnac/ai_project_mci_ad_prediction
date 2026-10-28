"""Create a local MRI review sheet from the current cohort and MRI candidates.

Run from the project folder:
  py -3.14 .\scripts\make_mri_manifest.py

Fill in nifti_path and set t1_confirmed to yes only after reviewing the scan in IDA.
The resulting file contains participant-level data; keep it local.
"""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
COHORT = META / "master_cohort_clinical_only.csv"
CANDIDATES = META / "mri_scan_candidates_local.csv"
OUT = META / "mri_training_manifest.csv"


def local_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT, help="Local cohort CSV")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES, help="Local MRI candidate CSV")
    parser.add_argument("--output", type=Path, default=OUT, help="Local review manifest output")
    args = parser.parse_args()
    cohort_path = local_path(args.cohort)
    candidates_path = local_path(args.candidates)
    out_path = local_path(args.output)
    if not cohort_path.exists() or not candidates_path.exists():
        raise FileNotFoundError(
            "Build the clinical cohort and run match_mri_metadata.py before creating the manifest."
        )
    cohort = pd.read_csv(cohort_path, low_memory=False)
    candidates = pd.read_csv(candidates_path, low_memory=False)
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
    # Preserve manual review fields only for the exact same selected scan.
    if out_path.exists():
        previous = pd.read_csv(out_path, low_memory=False)
        previous.columns = previous.columns.str.strip().str.upper()
        keys = [key for key in ["PTID", "IMAGE_DATA_ID"] if key in sheet and key in previous]
        manual = [col for col in ["NIFTI_PATH", "T1_CONFIRMED"] if col in previous]
        if keys and manual:
            previous = previous[keys + manual].drop_duplicates(keys, keep="last")
            sheet = sheet.merge(previous, on=keys, how="left", validate="one_to_one")
    for column in ["NIFTI_PATH", "T1_CONFIRMED"]:
        if column not in sheet:
            sheet[column] = ""
        else:
            sheet[column] = sheet[column].fillna("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_path, index=False)
    print(f"Created local MRI review manifest with {len(sheet)} candidate participants:")
    print(out_path)
    print("Fill NIFTI_PATH and set T1_CONFIRMED=yes only for a visually/metadata-verified T1 scan.")


if __name__ == "__main__":
    main()
