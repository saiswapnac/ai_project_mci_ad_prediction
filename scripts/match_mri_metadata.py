"""Match local ADNI MRI metadata to the clinical cohort without printing IDs or rows.

Place the Simple Search CSV at data/raw/MRI_metadata.csv, then run from the
mci_ad_prediction folder:
  py -3.14 .\\scripts\\match_mri_metadata.py

Participant-level candidate output is kept locally in data/metadata/.
"""
from pathlib import Path
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"
COHORT_PATH = META / "master_cohort_clinical_only.csv"
MRI_PATH = RAW / "MRI_metadata.csv"
OUT_PATH = META / "mri_scan_candidates_local.csv"
WINDOW_DAYS = 180


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^A-Z0-9]+", "_", str(c).strip().upper()).strip("_") for c in df.columns]
    return df


def main() -> int:
    if not COHORT_PATH.exists():
        print(f"Clinical cohort not found: {COHORT_PATH}")
        print("Run scripts/build_cohort_from_tables.py first.")
        return 1
    if not MRI_PATH.exists():
        print(f"MRI metadata CSV not found: {MRI_PATH}")
        print("Copy the Simple Search export to data/raw/MRI_metadata.csv and rerun.")
        return 1

    cohort = normalize_headers(pd.read_csv(COHORT_PATH, low_memory=False))
    mri = normalize_headers(pd.read_csv(MRI_PATH, low_memory=False))
    required_cohort = {"PTID", "BASELINE_DATE"}
    required_mri = {"IMAGE_DATA_ID", "SUBJECT", "VISIT", "MODALITY", "DESCRIPTION", "TYPE", "ACQ_DATE", "FORMAT"}
    missing = sorted(required_cohort - set(cohort.columns)) + sorted(required_mri - set(mri.columns))
    if missing:
        print("Missing expected columns after header normalization:", ", ".join(missing))
        print("This script will not print file contents. Check the local CSV headers and adjust the mapping.")
        return 2

    cohort["_subject_key"] = cohort["PTID"].astype("string").str.strip().str.casefold()
    cohort["BASELINE_DATE"] = pd.to_datetime(cohort["BASELINE_DATE"], errors="coerce")
    cohort = cohort.dropna(subset=["_subject_key", "BASELINE_DATE"]).drop_duplicates("_subject_key")
    mri["_subject_key"] = mri["SUBJECT"].astype("string").str.strip().str.casefold()
    mri["ACQ_DATE"] = pd.to_datetime(mri["ACQ_DATE"], errors="coerce")
    mri["_modality"] = mri["MODALITY"].astype("string").str.strip().str.upper()
    mri["_description"] = mri["DESCRIPTION"].fillna("").astype(str)

    cohort_keys = set(cohort["_subject_key"].dropna())
    mri_keys = set(mri["_subject_key"].dropna())
    overlap = cohort_keys & mri_keys
    linked = mri[mri["_subject_key"].isin(overlap)].merge(
        cohort[["_subject_key", "PTID", "BASELINE_DATE"]], on="_subject_key", how="inner", validate="many_to_one"
    )
    linked["days_from_baseline"] = (linked["ACQ_DATE"] - linked["BASELINE_DATE"]).dt.days
    linked["abs_days_from_baseline"] = linked["days_from_baseline"].abs()
    linked["modality_is_mr"] = linked["_modality"].isin({"MR", "MRI"})
    # This is only a text hint for manual review. It does not certify T1 weighting.
    linked["description_t1_hint"] = linked["_description"].str.contains(
        r"MPRAGE|IR\s*[-/]?\s*F?SPGR|T1W|T1.WEIGHT", case=False, regex=True, na=False
    )
    candidates = linked[
        linked["modality_is_mr"] & linked["ACQ_DATE"].notna()
        & linked["abs_days_from_baseline"].le(WINDOW_DAYS)
    ].copy()
    candidates = candidates.sort_values(["PTID", "abs_days_from_baseline", "IMAGE_DATA_ID"])

    # Preserve relevant image-selection fields and local cohort identifiers for private use.
    keep = [c for c in ["PTID", "BASELINE_DATE", "IMAGE_DATA_ID", "SUBJECT", "VISIT", "MODALITY",
                         "DESCRIPTION", "TYPE", "ACQ_DATE", "FORMAT", "days_from_baseline",
                         "abs_days_from_baseline", "description_t1_hint"] if c in candidates.columns]
    META.mkdir(parents=True, exist_ok=True)
    candidates[keep].to_csv(OUT_PATH, index=False)

    matched_ids = set(candidates["_subject_key"])
    print("MRI metadata match summary (aggregate counts only):")
    print("Clinical cohort participants:", cohort["_subject_key"].nunique())
    print("MRI metadata participants:", mri["_subject_key"].nunique())
    print("Exact participant-ID overlap:", len(overlap))
    print("Participants with MR metadata within +/-180 days of baseline:", len(matched_ids))
    print("Candidate scan rows in window:", len(candidates))
    print("Candidate rows with T1-like description text (review required):", int(candidates["description_t1_hint"].sum()))
    if not candidates.empty:
        print("Format counts:")
        print(candidates["FORMAT"].astype("string").fillna("<missing>").value_counts().to_string())
        print("Visit counts (top 15):")
        print(candidates["VISIT"].astype("string").fillna("<missing>").value_counts().head(15).to_string())
    print(f"\nSaved participant-level scan candidates locally to: {OUT_PATH}")
    print("The description hint is not proof of T1. Review T1 weighting and image details in IDA before downloading.")
    if len(overlap) == 0:
        print("No exact Subject-to-PTID matches. A local ROSTER mapping or ID-format normalization may be needed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
